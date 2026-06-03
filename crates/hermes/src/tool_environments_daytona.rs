//! Daytona cloud execution environment.
//!
//! Native Rust port of `tools/environments/daytona.py`.
//!
//! The Python original drives the Daytona Python SDK to run commands in cloud
//! sandboxes.  This port reproduces the same behaviour against the Daytona REST
//! API using `reqwest::blocking`.  It supports persistent sandboxes: when
//! enabled, sandboxes are stopped on cleanup and resumed on next creation,
//! preserving the filesystem across sessions.
//!
//! Faithful behaviours preserved:
//!   * Resource sizing: memory/disk MiB -> ceil(GiB), disk capped at 10 GiB
//!     with a warning.
//!   * Persistent resume flow: try `get(name)` + `start`, then fall back to a
//!     labelled `list` lookup, finally `create`.
//!   * Remote `$HOME` detection; `cwd` rewrite when requested cwd is `~` or
//!     `/home/daytona`.
//!   * Shell timeout wrapper via `bash -c` / `bash -l -c` with shell-quoting.
//!   * File-sync shell-string builders (mkdir -p / rm -f / tar) matching the
//!     `file_sync` helpers, including PID-suffixed remote temp paths.
//!   * cleanup: sync_back, then stop (persistent) or delete (ephemeral).
//!
//! The Daytona SDK surface is abstracted behind the [`DaytonaApi`] trait so the
//! environment logic is testable without a live API.  A reqwest-backed
//! implementation, [`HttpDaytonaApi`], reproduces the REST request construction
//! and response parsing.

use std::sync::{Arc, Mutex};

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
    // shlex._find_unsafe matches anything NOT in this safe set.
    let safe = |c: char| {
        c.is_ascii_alphanumeric() || matches!(c, '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-')
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

// ---------------------------------------------------------------------------
// file_sync shell-string helpers (ported from tools/environments/file_sync.py)
// ---------------------------------------------------------------------------

/// Build a shell `rm -f` command for a batch of remote paths.
pub fn quoted_rm_command(remote_paths: &[String]) -> String {
    let mut s = String::from("rm -f ");
    s.push_str(
        &remote_paths
            .iter()
            .map(|p| shlex_quote(p))
            .collect::<Vec<_>>()
            .join(" "),
    );
    s
}

/// Build a shell `mkdir -p` command for a batch of directories.
pub fn quoted_mkdir_command(dirs: &[String]) -> String {
    let mut s = String::from("mkdir -p ");
    s.push_str(
        &dirs
            .iter()
            .map(|d| shlex_quote(d))
            .collect::<Vec<_>>()
            .join(" "),
    );
    s
}

/// Return the parent directory of a remote path, mirroring `Path(p).parent`.
///
/// `Path("a/b/c.txt").parent == "a/b"`, `Path("c.txt").parent == "."`,
/// `Path("/").parent == "/"`.
pub fn parent_dir(path: &str) -> String {
    // Strip trailing slashes (except a lone root).
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

/// Extract sorted unique parent directories from (host, remote) pairs.
pub fn unique_parent_dirs(files: &[(String, String)]) -> Vec<String> {
    let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (_, remote) in files {
        set.insert(parent_dir(remote));
    }
    set.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Daytona SDK surface (abstracted for testability)
// ---------------------------------------------------------------------------

/// Sandbox lifecycle state (mirrors `daytona.SandboxState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxState {
    Started,
    Stopped,
    Archived,
    Creating,
    Error,
    Unknown,
}

impl SandboxState {
    /// Parse a Daytona API state string (e.g. `"started"`, `"stopped"`).
    pub fn from_api(s: &str) -> SandboxState {
        match s.to_ascii_lowercase().as_str() {
            "started" | "running" => SandboxState::Started,
            "stopped" => SandboxState::Stopped,
            "archived" => SandboxState::Archived,
            "creating" => SandboxState::Creating,
            "error" => SandboxState::Error,
            _ => SandboxState::Unknown,
        }
    }
}

/// Compute resources for a sandbox (mirrors `daytona.Resources`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resources {
    pub cpu: i64,
    /// GiB
    pub memory: i64,
    /// GiB
    pub disk: i64,
}

/// Parameters for creating a sandbox from an image
/// (mirrors `daytona.CreateSandboxFromImageParams`).
#[derive(Debug, Clone)]
pub struct CreateSandboxFromImageParams {
    pub image: String,
    pub name: String,
    pub labels: Vec<(String, String)>,
    pub auto_stop_interval: i64,
    pub resources: Resources,
}

/// A single file to upload in a bulk multipart request
/// (mirrors `daytona.common.filesystem.FileUpload`).
#[derive(Debug, Clone)]
pub struct FileUpload {
    pub source: String,
    pub destination: String,
}

/// Result of a `process.exec` call (mirrors the SDK `ExecuteResponse`).
#[derive(Debug, Clone)]
pub struct ExecResponse {
    pub result: Option<String>,
    pub exit_code: i32,
}

/// Errors surfaced by the Daytona API surface.
#[derive(Debug)]
pub enum DaytonaApiError {
    /// A `DaytonaError` in the SDK (e.g. sandbox not found) — drives the
    /// persistent-resume fallback path.
    Daytona(String),
    /// Any other error (network, IO, etc.).
    Other(String),
}

impl std::fmt::Display for DaytonaApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DaytonaApiError::Daytona(m) => write!(f, "DaytonaError: {m}"),
            DaytonaApiError::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for DaytonaApiError {}

/// A page of sandboxes returned by `list` (mirrors the SDK paged response).
#[derive(Debug, Clone, Default)]
pub struct SandboxPage {
    pub items: Vec<String>, // sandbox ids
}

/// Abstraction over the Daytona SDK / REST API.
///
/// Each method maps onto an SDK call used by the Python original.  Sandbox
/// handles are represented by their string id; the trait carries the
/// per-sandbox operations the environment needs.
pub trait DaytonaApi: Send + Sync {
    /// `Daytona.get(name)` — resolve a sandbox by name. `Err(Daytona)` when
    /// the sandbox does not exist (drives the resume fallback).
    fn get(&self, name: &str) -> Result<String, DaytonaApiError>;

    /// `Daytona.list(labels=..., page=1, limit=1)`.
    fn list(
        &self,
        labels: &[(String, String)],
        page: u32,
        limit: u32,
    ) -> Result<SandboxPage, DaytonaApiError>;

    /// `Daytona.create(CreateSandboxFromImageParams(...))` — returns sandbox id.
    fn create(&self, params: &CreateSandboxFromImageParams) -> Result<String, DaytonaApiError>;

    /// `Daytona.delete(sandbox)`.
    fn delete(&self, sandbox_id: &str) -> Result<(), DaytonaApiError>;

    /// `sandbox.start()`.
    fn start(&self, sandbox_id: &str) -> Result<(), DaytonaApiError>;

    /// `sandbox.stop()`.
    fn stop(&self, sandbox_id: &str) -> Result<(), DaytonaApiError>;

    /// `sandbox.refresh_data()` then read `sandbox.state`.
    fn refresh_state(&self, sandbox_id: &str) -> Result<SandboxState, DaytonaApiError>;

    /// `sandbox.process.exec(cmd, timeout=...)`.
    fn exec(
        &self,
        sandbox_id: &str,
        command: &str,
        timeout: Option<i64>,
    ) -> Result<ExecResponse, DaytonaApiError>;

    /// `sandbox.fs.upload_file(host_path, remote_path)`.
    fn upload_file(
        &self,
        sandbox_id: &str,
        host_path: &str,
        remote_path: &str,
    ) -> Result<(), DaytonaApiError>;

    /// `sandbox.fs.upload_files(uploads)` — single multipart POST.
    fn upload_files(
        &self,
        sandbox_id: &str,
        uploads: &[FileUpload],
    ) -> Result<(), DaytonaApiError>;

    /// `sandbox.fs.download_file(remote_path, dest)`.
    fn download_file(
        &self,
        sandbox_id: &str,
        remote_path: &str,
        dest: &str,
    ) -> Result<(), DaytonaApiError>;
}

// ---------------------------------------------------------------------------
// reqwest-backed implementation
// ---------------------------------------------------------------------------

/// reqwest::blocking implementation of [`DaytonaApi`] against the Daytona REST
/// API.  Reads `DAYTONA_API_KEY` / `DAYTONA_API_URL` from the environment, as
/// the `Daytona()` SDK constructor does.
pub struct HttpDaytonaApi {
    api_url: String,
    api_key: String,
    target: Option<String>,
    client: reqwest::blocking::Client,
}

impl HttpDaytonaApi {
    /// Construct from environment (mirrors `Daytona()` zero-arg constructor).
    pub fn from_env() -> Result<HttpDaytonaApi, DaytonaApiError> {
        let api_key = std::env::var("DAYTONA_API_KEY")
            .map_err(|_| DaytonaApiError::Other("DAYTONA_API_KEY not set".into()))?;
        let api_url = std::env::var("DAYTONA_API_URL")
            .or_else(|_| std::env::var("DAYTONA_SERVER_URL"))
            .unwrap_or_else(|_| "https://app.daytona.io/api".to_string());
        let target = std::env::var("DAYTONA_TARGET").ok();
        let client = reqwest::blocking::Client::builder()
            .build()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        Ok(HttpDaytonaApi {
            api_url: api_url.trim_end_matches('/').to_string(),
            api_key,
            target,
            client,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.api_url, path.trim_start_matches('/'))
    }

    fn auth(
        &self,
        rb: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        rb.bearer_auth(&self.api_key)
    }

    fn check(resp: reqwest::blocking::Response) -> Result<reqwest::blocking::Response, DaytonaApiError> {
        let status = resp.status();
        if status.as_u16() == 404 {
            return Err(DaytonaApiError::Daytona(format!("not found ({status})")));
        }
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(DaytonaApiError::Other(format!("HTTP {status}: {body}")));
        }
        Ok(resp)
    }
}

impl DaytonaApi for HttpDaytonaApi {
    fn get(&self, name: &str) -> Result<String, DaytonaApiError> {
        // GET /sandbox/{name} — 404 maps to DaytonaError (resume fallback).
        let resp = self
            .auth(self.client.get(self.url(&format!("sandbox/{name}"))))
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        let resp = Self::check(resp)?;
        let v: serde_json::Value =
            resp.json().map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        v.get("id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| DaytonaApiError::Daytona("sandbox has no id".into()))
    }

    fn list(
        &self,
        labels: &[(String, String)],
        page: u32,
        limit: u32,
    ) -> Result<SandboxPage, DaytonaApiError> {
        let labels_json = serde_json::to_string(
            &labels
                .iter()
                .cloned()
                .collect::<std::collections::BTreeMap<String, String>>(),
        )
        .unwrap_or_else(|_| "{}".to_string());
        let mut rb = self.auth(self.client.get(self.url("sandbox"))).query(&[
            ("labels", labels_json.as_str()),
            ("page", &page.to_string()),
            ("limit", &limit.to_string()),
        ]);
        if let Some(t) = &self.target {
            rb = rb.query(&[("target", t.as_str())]);
        }
        let resp = rb.send().map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        let resp = Self::check(resp)?;
        let v: serde_json::Value =
            resp.json().map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        // The API may return either a bare array or {items: [...]}.
        let arr = v
            .get("items")
            .and_then(|x| x.as_array())
            .or_else(|| v.as_array())
            .cloned()
            .unwrap_or_default();
        let items = arr
            .iter()
            .filter_map(|s| s.get("id").and_then(|x| x.as_str()).map(|s| s.to_string()))
            .collect();
        Ok(SandboxPage { items })
    }

    fn create(&self, params: &CreateSandboxFromImageParams) -> Result<String, DaytonaApiError> {
        let labels: std::collections::BTreeMap<String, String> =
            params.labels.iter().cloned().collect();
        let body = serde_json::json!({
            "image": params.image,
            "name": params.name,
            "labels": labels,
            "autoStopInterval": params.auto_stop_interval,
            "cpu": params.resources.cpu,
            "memory": params.resources.memory,
            "disk": params.resources.disk,
            "target": self.target,
        });
        let resp = self
            .auth(self.client.post(self.url("sandbox")))
            .json(&body)
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        let resp = Self::check(resp)?;
        let v: serde_json::Value =
            resp.json().map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        v.get("id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| DaytonaApiError::Other("create response missing id".into()))
    }

    fn delete(&self, sandbox_id: &str) -> Result<(), DaytonaApiError> {
        let resp = self
            .auth(self.client.delete(self.url(&format!("sandbox/{sandbox_id}"))))
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        Self::check(resp).map(|_| ())
    }

    fn start(&self, sandbox_id: &str) -> Result<(), DaytonaApiError> {
        let resp = self
            .auth(self.client.post(self.url(&format!("sandbox/{sandbox_id}/start"))))
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        Self::check(resp).map(|_| ())
    }

    fn stop(&self, sandbox_id: &str) -> Result<(), DaytonaApiError> {
        let resp = self
            .auth(self.client.post(self.url(&format!("sandbox/{sandbox_id}/stop"))))
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        Self::check(resp).map(|_| ())
    }

    fn refresh_state(&self, sandbox_id: &str) -> Result<SandboxState, DaytonaApiError> {
        let resp = self
            .auth(self.client.get(self.url(&format!("sandbox/{sandbox_id}"))))
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        let resp = Self::check(resp)?;
        let v: serde_json::Value =
            resp.json().map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        let state = v
            .get("state")
            .and_then(|x| x.as_str())
            .unwrap_or("unknown");
        Ok(SandboxState::from_api(state))
    }

    fn exec(
        &self,
        sandbox_id: &str,
        command: &str,
        timeout: Option<i64>,
    ) -> Result<ExecResponse, DaytonaApiError> {
        let mut body = serde_json::json!({ "command": command });
        if let Some(t) = timeout {
            body["timeout"] = serde_json::json!(t);
        }
        let resp = self
            .auth(
                self.client
                    .post(self.url(&format!("toolbox/{sandbox_id}/toolbox/process/execute"))),
            )
            .json(&body)
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        let resp = Self::check(resp)?;
        let v: serde_json::Value =
            resp.json().map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        let result = v
            .get("result")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        let exit_code = v
            .get("exitCode")
            .or_else(|| v.get("exit_code"))
            .and_then(|x| x.as_i64())
            .unwrap_or(0) as i32;
        Ok(ExecResponse { result, exit_code })
    }

    fn upload_file(
        &self,
        sandbox_id: &str,
        host_path: &str,
        remote_path: &str,
    ) -> Result<(), DaytonaApiError> {
        let bytes = std::fs::read(host_path).map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        let resp = self
            .auth(
                self.client
                    .post(self.url(&format!("toolbox/{sandbox_id}/toolbox/files/upload")))
                    .query(&[("path", remote_path)]),
            )
            .body(bytes)
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        Self::check(resp).map(|_| ())
    }

    fn upload_files(
        &self,
        sandbox_id: &str,
        uploads: &[FileUpload],
    ) -> Result<(), DaytonaApiError> {
        use reqwest::blocking::multipart;
        let mut form = multipart::Form::new();
        for up in uploads {
            let part = multipart::Part::file(&up.source)
                .map_err(|e| DaytonaApiError::Other(e.to_string()))?
                .file_name(up.destination.clone());
            form = form.part("files", part);
        }
        let resp = self
            .auth(
                self.client
                    .post(self.url(&format!("toolbox/{sandbox_id}/toolbox/files/bulk-upload"))),
            )
            .multipart(form)
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        Self::check(resp).map(|_| ())
    }

    fn download_file(
        &self,
        sandbox_id: &str,
        remote_path: &str,
        dest: &str,
    ) -> Result<(), DaytonaApiError> {
        let resp = self
            .auth(
                self.client
                    .get(self.url(&format!("toolbox/{sandbox_id}/toolbox/files/download")))
                    .query(&[("path", remote_path)]),
            )
            .send()
            .map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        let resp = Self::check(resp)?;
        let bytes = resp.bytes().map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        std::fs::write(dest, &bytes).map_err(|e| DaytonaApiError::Other(e.to_string()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// resource sizing
// ---------------------------------------------------------------------------

/// Convert MiB to GiB via ceil, with a floor of 1 (mirrors
/// `max(1, math.ceil(value / 1024))`).
pub fn mib_to_gib(mib: i64) -> i64 {
    let g = (mib as f64 / 1024.0).ceil() as i64;
    g.max(1)
}

/// Daytona platform disk ceiling, in GiB.
pub const DAYTONA_DISK_CAP_GIB: i64 = 10;

/// Compute the [`Resources`] for the requested CPU/memory/disk (MiB).
///
/// Returns the computed resources and a flag indicating whether disk was
/// capped (so the caller can emit the platform-limit warning the Python logs).
pub fn compute_resources(cpu: i64, memory_mib: i64, disk_mib: i64) -> (Resources, bool) {
    let memory_gib = mib_to_gib(memory_mib);
    let mut disk_gib = mib_to_gib(disk_mib);
    let capped = disk_gib > DAYTONA_DISK_CAP_GIB;
    if capped {
        disk_gib = DAYTONA_DISK_CAP_GIB;
    }
    (
        Resources {
            cpu,
            memory: memory_gib,
            disk: disk_gib,
        },
        capped,
    )
}

// ---------------------------------------------------------------------------
// Daytona environment
// ---------------------------------------------------------------------------

/// Configuration for a [`DaytonaEnvironment`] (mirrors `__init__` args).
#[derive(Debug, Clone)]
pub struct DaytonaConfig {
    pub image: String,
    pub cwd: String,
    pub timeout: i64,
    pub cpu: i64,
    /// MiB
    pub memory: i64,
    /// MiB
    pub disk: i64,
    pub persistent_filesystem: bool,
    pub task_id: String,
}

impl Default for DaytonaConfig {
    fn default() -> Self {
        DaytonaConfig {
            image: String::new(),
            cwd: "/home/daytona".to_string(),
            timeout: 60,
            cpu: 1,
            memory: 5120,
            disk: 10240,
            persistent_filesystem: true,
            task_id: "default".to_string(),
        }
    }
}

/// Daytona cloud sandbox execution backend.
///
/// Mirrors `DaytonaEnvironment`.  Holds the API surface, the resolved sandbox
/// id, the resolved remote home / cwd, and a lock guarding lifecycle calls.
pub struct DaytonaEnvironment {
    api: Arc<dyn DaytonaApi>,
    persistent: bool,
    task_id: String,
    sandbox: Mutex<Option<String>>,
    remote_home: String,
    cwd: String,
    timeout: i64,
    /// Sandbox name label key/value for resume + create.
    sandbox_name: String,
    labels: Vec<(String, String)>,
    resources: Resources,
    image: String,
}

impl DaytonaEnvironment {
    /// `_stdin_mode = "heredoc"` from the Python class.
    pub const STDIN_MODE: &'static str = "heredoc";

    /// Construct a Daytona environment, resolving/creating its sandbox.
    ///
    /// Mirrors `DaytonaEnvironment.__init__`: resume an existing persistent
    /// sandbox (by name, then by label), otherwise create one; then detect the
    /// remote home dir and rewrite cwd; finally perform the initial forced
    /// file sync.
    pub fn new(
        api: Arc<dyn DaytonaApi>,
        config: &DaytonaConfig,
    ) -> Result<DaytonaEnvironment, DaytonaApiError> {
        let requested_cwd = config.cwd.clone();

        let (resources, disk_capped) =
            compute_resources(config.cpu, config.memory, config.disk);
        if disk_capped {
            log::warn!(
                "Daytona: requested disk exceeds platform limit (10GB). Capping to 10GB."
            );
        }

        let labels = vec![("hermes_task_id".to_string(), config.task_id.clone())];
        let sandbox_name = format!("hermes-{}", config.task_id);

        let mut sandbox: Option<String> = None;

        if config.persistent_filesystem {
            // Try get(name) + start.
            match api.get(&sandbox_name) {
                Ok(id) => match api.start(&id) {
                    Ok(()) => {
                        log::info!(
                            "Daytona: resumed sandbox {} for task {}",
                            id,
                            config.task_id
                        );
                        sandbox = Some(id);
                    }
                    Err(DaytonaApiError::Daytona(_)) => {
                        sandbox = None;
                    }
                    Err(e) => {
                        log::warn!(
                            "Daytona: failed to resume sandbox for task {}: {}",
                            config.task_id,
                            e
                        );
                        sandbox = None;
                    }
                },
                Err(DaytonaApiError::Daytona(_)) => {
                    sandbox = None;
                }
                Err(e) => {
                    log::warn!(
                        "Daytona: failed to resume sandbox for task {}: {}",
                        config.task_id,
                        e
                    );
                    sandbox = None;
                }
            }

            // Fall back to a labelled list lookup.
            if sandbox.is_none() {
                match api.list(&labels, 1, 1) {
                    Ok(page) => {
                        if let Some(id) = page.items.into_iter().next() {
                            match api.start(&id) {
                                Ok(()) => {
                                    log::info!(
                                        "Daytona: resumed legacy sandbox {} for task {}",
                                        id,
                                        config.task_id
                                    );
                                    sandbox = Some(id);
                                }
                                Err(e) => {
                                    log::debug!(
                                        "Daytona: no legacy sandbox found for task {}: {}",
                                        config.task_id,
                                        e
                                    );
                                    sandbox = None;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::debug!(
                            "Daytona: no legacy sandbox found for task {}: {}",
                            config.task_id,
                            e
                        );
                        sandbox = None;
                    }
                }
            }
        }

        // Create if not resumed.
        if sandbox.is_none() {
            let params = CreateSandboxFromImageParams {
                image: config.image.clone(),
                name: sandbox_name.clone(),
                labels: labels.clone(),
                auto_stop_interval: 0,
                resources,
            };
            let id = api.create(&params)?;
            log::info!("Daytona: created sandbox {} for task {}", id, config.task_id);
            sandbox = Some(id);
        }

        let sandbox_id = sandbox.expect("sandbox must be set after resume/create");

        // Detect remote home dir.
        let mut remote_home = "/root".to_string();
        let mut cwd = config.cwd.clone();
        if let Ok(resp) = api.exec(&sandbox_id, "echo $HOME", None) {
            let home = resp.result.unwrap_or_default().trim().to_string();
            if !home.is_empty() {
                remote_home = home.clone();
                if requested_cwd == "~" || requested_cwd == "/home/daytona" {
                    cwd = home;
                }
            }
        }
        log::info!(
            "Daytona: resolved home to {}, cwd to {}",
            remote_home,
            cwd
        );

        let env = DaytonaEnvironment {
            api,
            persistent: config.persistent_filesystem,
            task_id: config.task_id.clone(),
            sandbox: Mutex::new(Some(sandbox_id)),
            remote_home,
            cwd,
            timeout: config.timeout,
            sandbox_name,
            labels,
            resources,
            image: config.image.clone(),
        };

        Ok(env)
    }

    /// Resolved remote `$HOME`.
    pub fn remote_home(&self) -> &str {
        &self.remote_home
    }

    /// Resolved working directory.
    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// Configured default timeout (seconds).
    pub fn timeout(&self) -> i64 {
        self.timeout
    }

    /// The sandbox name label (`hermes-<task_id>`).
    pub fn sandbox_name(&self) -> &str {
        &self.sandbox_name
    }

    /// Computed sandbox resources.
    pub fn resources(&self) -> Resources {
        self.resources
    }

    /// Currently-resolved sandbox id, if any.
    pub fn sandbox_id(&self) -> Option<String> {
        self.sandbox.lock().unwrap().clone()
    }

    /// Upload a single file via the SDK: `mkdir -p <parent>` then upload.
    /// Mirrors `_daytona_upload`.
    pub fn daytona_upload(&self, host_path: &str, remote_path: &str) -> Result<(), DaytonaApiError> {
        let sid = self.require_sandbox()?;
        let parent = parent_dir(remote_path);
        let _ = self.api.exec(&sid, &format!("mkdir -p {parent}"), None)?;
        self.api.upload_file(&sid, host_path, remote_path)
    }

    /// Bulk-upload many files in a single multipart POST.
    /// Mirrors `_daytona_bulk_upload`: short-circuit on empty, mkdir -p parents,
    /// then `upload_files`.
    pub fn daytona_bulk_upload(&self, files: &[(String, String)]) -> Result<(), DaytonaApiError> {
        if files.is_empty() {
            return Ok(());
        }
        let sid = self.require_sandbox()?;
        let parents = unique_parent_dirs(files);
        if !parents.is_empty() {
            let _ = self.api.exec(&sid, &quoted_mkdir_command(&parents), None)?;
        }
        let uploads: Vec<FileUpload> = files
            .iter()
            .map(|(host, remote)| FileUpload {
                source: host.clone(),
                destination: remote.clone(),
            })
            .collect();
        self.api.upload_files(&sid, &uploads)
    }

    /// Download remote `.hermes/` as a tar archive to `dest`.
    /// Mirrors `_daytona_bulk_download`, including the PID-suffixed remote temp
    /// path and best-effort cleanup.
    pub fn daytona_bulk_download(&self, dest: &str) -> Result<(), DaytonaApiError> {
        let sid = self.require_sandbox()?;
        let rel_base = format!("{}/.hermes", self.remote_home)
            .trim_start_matches('/')
            .to_string();
        let remote_tar = format!("/tmp/.hermes_sync.{}.tar", std::process::id());
        let tar_cmd = format!(
            "tar cf {} -C / {}",
            shlex_quote(&remote_tar),
            shlex_quote(&rel_base)
        );
        let _ = self.api.exec(&sid, &tar_cmd, None)?;
        self.api.download_file(&sid, &remote_tar, dest)?;
        // best-effort cleanup
        let _ = self
            .api
            .exec(&sid, &format!("rm -f {}", shlex_quote(&remote_tar)), None);
        Ok(())
    }

    /// Batch-delete remote files via exec. Mirrors `_daytona_delete`.
    pub fn daytona_delete(&self, remote_paths: &[String]) -> Result<(), DaytonaApiError> {
        let sid = self.require_sandbox()?;
        let _ = self
            .api
            .exec(&sid, &quoted_rm_command(remote_paths), None)?;
        Ok(())
    }

    /// Restart the sandbox if it was stopped/archived. Mirrors
    /// `_ensure_sandbox_ready` (caller holds the lifecycle lock).
    pub fn ensure_sandbox_ready(&self) -> Result<(), DaytonaApiError> {
        let sid = self.require_sandbox()?;
        let state = self.api.refresh_state(&sid)?;
        if matches!(state, SandboxState::Stopped | SandboxState::Archived) {
            self.api.start(&sid)?;
            log::info!("Daytona: restarted sandbox {}", sid);
        }
        Ok(())
    }

    /// Build the shell command string for `_run_bash`.
    ///
    /// `login=true` -> `bash -l -c <quoted>`, else `bash -c <quoted>`.
    pub fn build_shell_cmd(cmd_string: &str, login: bool) -> String {
        if login {
            format!("bash -l -c {}", shlex_quote(cmd_string))
        } else {
            format!("bash -c {}", shlex_quote(cmd_string))
        }
    }

    /// Run a bash command in the sandbox, returning `(result, exit_code)`.
    ///
    /// Mirrors `_run_bash`'s `exec_fn`: wraps the command in a `bash -c`
    /// (or `bash -l -c`) shell and execs it via the SDK with the supplied
    /// timeout.  The Python original returns a `_ThreadedProcessHandle` whose
    /// cancel callback calls `sandbox.stop()`; here the call is blocking and
    /// cancellation is exposed separately via [`cancel`](Self::cancel).
    pub fn run_bash(
        &self,
        cmd_string: &str,
        login: bool,
        timeout: i64,
    ) -> Result<(String, i32), DaytonaApiError> {
        let sid = self.require_sandbox()?;
        let shell_cmd = Self::build_shell_cmd(cmd_string, login);
        let response = self.api.exec(&sid, &shell_cmd, Some(timeout))?;
        Ok((response.result.unwrap_or_default(), response.exit_code))
    }

    /// Cancel a running command by stopping the sandbox (best-effort), matching
    /// the `cancel` closure wired into the Python `_ThreadedProcessHandle`.
    pub fn cancel(&self) {
        if let Some(sid) = self.sandbox.lock().unwrap().clone() {
            let _ = self.api.stop(&sid);
        }
    }

    /// Tear down the environment. Mirrors `cleanup`: under the lock, no-op if
    /// already cleaned; otherwise run `sync_back` (best effort) then stop
    /// (persistent) or delete (ephemeral), clearing the sandbox handle.
    ///
    /// `sync_back` is supplied as a closure so the file-sync layer (not yet
    /// ported) can be injected; pass a no-op `|| Ok(())` when unavailable.
    pub fn cleanup<F>(&self, sync_back: F)
    where
        F: FnOnce() -> Result<(), DaytonaApiError>,
    {
        let mut guard = self.sandbox.lock().unwrap();
        let sid = match guard.clone() {
            Some(id) => id,
            None => return,
        };

        log::info!("Daytona: syncing files from sandbox...");
        if let Err(e) = sync_back() {
            log::warn!("Daytona: sync_back failed: {}", e);
        }

        let result = if self.persistent {
            self.api.stop(&sid).map(|()| {
                log::info!(
                    "Daytona: stopped sandbox {} (filesystem preserved)",
                    sid
                );
            })
        } else {
            self.api.delete(&sid).map(|()| {
                log::info!("Daytona: deleted sandbox {}", sid);
            })
        };
        if let Err(e) = result {
            log::warn!("Daytona: cleanup failed: {}", e);
        }
        *guard = None;
    }

    /// Whether the sandbox filesystem is preserved across sessions.
    pub fn is_persistent(&self) -> bool {
        self.persistent
    }

    /// Task id this environment was created for.
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// Configured image.
    pub fn image(&self) -> &str {
        &self.image
    }

    /// Labels applied to the sandbox.
    pub fn labels(&self) -> &[(String, String)] {
        &self.labels
    }

    fn require_sandbox(&self) -> Result<String, DaytonaApiError> {
        self.sandbox
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| DaytonaApiError::Other("sandbox is not initialized".into()))
    }
}

// ===========================================================================
// tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;

    // ----- shlex_quote -----
    #[test]
    fn test_shlex_quote() {
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("simple"), "simple");
        assert_eq!(shlex_quote("/tmp/file.tar"), "/tmp/file.tar");
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
        assert_eq!(shlex_quote("a;b"), "'a;b'");
    }

    #[test]
    fn test_quoted_commands() {
        let paths = vec!["/a/b".to_string(), "/c d".to_string()];
        assert_eq!(quoted_rm_command(&paths), "rm -f /a/b '/c d'");
        let dirs = vec!["/x".to_string(), "/y z".to_string()];
        assert_eq!(quoted_mkdir_command(&dirs), "mkdir -p /x '/y z'");
    }

    #[test]
    fn test_parent_dir() {
        assert_eq!(parent_dir("a/b/c.txt"), "a/b");
        assert_eq!(parent_dir("/root/.hermes/x"), "/root/.hermes");
        assert_eq!(parent_dir("c.txt"), ".");
        assert_eq!(parent_dir("/file"), "/");
        assert_eq!(parent_dir("/"), "/");
    }

    #[test]
    fn test_unique_parent_dirs_sorted() {
        let files = vec![
            ("h1".to_string(), "/a/b/x".to_string()),
            ("h2".to_string(), "/a/b/y".to_string()),
            ("h3".to_string(), "/a/c/z".to_string()),
        ];
        assert_eq!(unique_parent_dirs(&files), vec!["/a/b", "/a/c"]);
    }

    // ----- resource sizing -----
    #[test]
    fn test_mib_to_gib() {
        assert_eq!(mib_to_gib(5120), 5); // exactly 5 GiB
        assert_eq!(mib_to_gib(5121), 6); // ceil
        assert_eq!(mib_to_gib(0), 1); // floor of 1
        assert_eq!(mib_to_gib(100), 1);
        assert_eq!(mib_to_gib(1024), 1);
        assert_eq!(mib_to_gib(1025), 2);
    }

    #[test]
    fn test_compute_resources_caps_disk() {
        // 10240 MiB = 10 GiB, no cap.
        let (r, capped) = compute_resources(2, 5120, 10240);
        assert_eq!(r.cpu, 2);
        assert_eq!(r.memory, 5);
        assert_eq!(r.disk, 10);
        assert!(!capped);

        // 20480 MiB = 20 GiB -> capped to 10.
        let (r2, capped2) = compute_resources(1, 1024, 20480);
        assert_eq!(r2.disk, 10);
        assert!(capped2);
    }

    #[test]
    fn test_sandbox_state_parse() {
        assert_eq!(SandboxState::from_api("started"), SandboxState::Started);
        assert_eq!(SandboxState::from_api("STOPPED"), SandboxState::Stopped);
        assert_eq!(SandboxState::from_api("archived"), SandboxState::Archived);
        assert_eq!(SandboxState::from_api("weird"), SandboxState::Unknown);
    }

    #[test]
    fn test_build_shell_cmd() {
        assert_eq!(
            DaytonaEnvironment::build_shell_cmd("ls -la", false),
            "bash -c 'ls -la'"
        );
        assert_eq!(
            DaytonaEnvironment::build_shell_cmd("ls -la", true),
            "bash -l -c 'ls -la'"
        );
        assert_eq!(
            DaytonaEnvironment::build_shell_cmd("echo hi", false),
            "bash -c 'echo hi'"
        );
    }

    // ----- mock API for environment lifecycle tests -----
    #[derive(Default)]
    struct MockState {
        log: Vec<String>,
        existing_by_name: HashMap<String, String>,
        list_items: Vec<String>,
        next_create_id: String,
        state: HashMap<String, SandboxState>,
        home: String,
    }

    struct MockApi {
        st: StdMutex<MockState>,
    }

    impl MockApi {
        fn new(st: MockState) -> Self {
            MockApi { st: StdMutex::new(st) }
        }
        fn push(&self, s: &str) {
            self.st.lock().unwrap().log.push(s.to_string());
        }
        fn calls(&self) -> Vec<String> {
            self.st.lock().unwrap().log.clone()
        }
    }

    impl DaytonaApi for MockApi {
        fn get(&self, name: &str) -> Result<String, DaytonaApiError> {
            self.push(&format!("get:{name}"));
            let st = self.st.lock().unwrap();
            match st.existing_by_name.get(name) {
                Some(id) => Ok(id.clone()),
                None => Err(DaytonaApiError::Daytona("not found".into())),
            }
        }
        fn list(
            &self,
            _labels: &[(String, String)],
            _page: u32,
            _limit: u32,
        ) -> Result<SandboxPage, DaytonaApiError> {
            self.push("list");
            Ok(SandboxPage {
                items: self.st.lock().unwrap().list_items.clone(),
            })
        }
        fn create(&self, params: &CreateSandboxFromImageParams) -> Result<String, DaytonaApiError> {
            self.push(&format!("create:{}", params.name));
            Ok(self.st.lock().unwrap().next_create_id.clone())
        }
        fn delete(&self, sandbox_id: &str) -> Result<(), DaytonaApiError> {
            self.push(&format!("delete:{sandbox_id}"));
            Ok(())
        }
        fn start(&self, sandbox_id: &str) -> Result<(), DaytonaApiError> {
            self.push(&format!("start:{sandbox_id}"));
            Ok(())
        }
        fn stop(&self, sandbox_id: &str) -> Result<(), DaytonaApiError> {
            self.push(&format!("stop:{sandbox_id}"));
            Ok(())
        }
        fn refresh_state(&self, sandbox_id: &str) -> Result<SandboxState, DaytonaApiError> {
            self.push(&format!("refresh:{sandbox_id}"));
            Ok(self
                .st
                .lock()
                .unwrap()
                .state
                .get(sandbox_id)
                .copied()
                .unwrap_or(SandboxState::Started))
        }
        fn exec(
            &self,
            sandbox_id: &str,
            command: &str,
            _timeout: Option<i64>,
        ) -> Result<ExecResponse, DaytonaApiError> {
            self.push(&format!("exec:{sandbox_id}:{command}"));
            if command == "echo $HOME" {
                let home = self.st.lock().unwrap().home.clone();
                return Ok(ExecResponse {
                    result: Some(format!("{home}\n")),
                    exit_code: 0,
                });
            }
            Ok(ExecResponse {
                result: Some(String::new()),
                exit_code: 0,
            })
        }
        fn upload_file(&self, _s: &str, _h: &str, _r: &str) -> Result<(), DaytonaApiError> {
            self.push("upload_file");
            Ok(())
        }
        fn upload_files(&self, _s: &str, ups: &[FileUpload]) -> Result<(), DaytonaApiError> {
            self.push(&format!("upload_files:{}", ups.len()));
            Ok(())
        }
        fn download_file(&self, _s: &str, _r: &str, _d: &str) -> Result<(), DaytonaApiError> {
            self.push("download_file");
            Ok(())
        }
    }

    fn cfg() -> DaytonaConfig {
        DaytonaConfig {
            image: "ubuntu:22.04".to_string(),
            cwd: "/home/daytona".to_string(),
            timeout: 60,
            cpu: 1,
            memory: 5120,
            disk: 10240,
            persistent_filesystem: true,
            task_id: "t1".to_string(),
        }
    }

    #[test]
    fn test_resume_by_name() {
        let mut st = MockState::default();
        st.existing_by_name
            .insert("hermes-t1".to_string(), "sb-existing".to_string());
        st.home = "/home/daytona".to_string();
        let api = Arc::new(MockApi::new(st));
        let env = DaytonaEnvironment::new(api.clone(), &cfg()).unwrap();
        assert_eq!(env.sandbox_id(), Some("sb-existing".to_string()));
        let calls = api.calls();
        assert!(calls.iter().any(|c| c == "get:hermes-t1"));
        assert!(calls.iter().any(|c| c == "start:sb-existing"));
        assert!(!calls.iter().any(|c| c.starts_with("create:")));
        // home detection rewrote cwd to home (== /home/daytona path)
        assert_eq!(env.remote_home(), "/home/daytona");
        assert_eq!(env.cwd(), "/home/daytona");
    }

    #[test]
    fn test_resume_legacy_by_label() {
        let mut st = MockState::default();
        st.list_items = vec!["sb-legacy".to_string()];
        st.home = "/home/user".to_string();
        let api = Arc::new(MockApi::new(st));
        let mut c = cfg();
        c.cwd = "~".to_string();
        let env = DaytonaEnvironment::new(api.clone(), &c).unwrap();
        assert_eq!(env.sandbox_id(), Some("sb-legacy".to_string()));
        let calls = api.calls();
        assert!(calls.iter().any(|c| c == "list"));
        assert!(calls.iter().any(|c| c == "start:sb-legacy"));
        // cwd "~" rewritten to detected home
        assert_eq!(env.cwd(), "/home/user");
    }

    #[test]
    fn test_create_when_no_resume() {
        let mut st = MockState::default();
        st.next_create_id = "sb-new".to_string();
        st.home = "/root".to_string();
        let api = Arc::new(MockApi::new(st));
        let env = DaytonaEnvironment::new(api.clone(), &cfg()).unwrap();
        assert_eq!(env.sandbox_id(), Some("sb-new".to_string()));
        assert!(api.calls().iter().any(|c| c == "create:hermes-t1"));
    }

    #[test]
    fn test_create_when_not_persistent_skips_resume() {
        let mut st = MockState::default();
        st.existing_by_name
            .insert("hermes-t1".to_string(), "sb-existing".to_string());
        st.next_create_id = "sb-new".to_string();
        st.home = "/root".to_string();
        let api = Arc::new(MockApi::new(st));
        let mut c = cfg();
        c.persistent_filesystem = false;
        let env = DaytonaEnvironment::new(api.clone(), &c).unwrap();
        assert_eq!(env.sandbox_id(), Some("sb-new".to_string()));
        let calls = api.calls();
        assert!(!calls.iter().any(|c| c.starts_with("get:")));
        assert!(calls.iter().any(|c| c == "create:hermes-t1"));
    }

    #[test]
    fn test_cleanup_persistent_stops() {
        let mut st = MockState::default();
        st.next_create_id = "sb-new".to_string();
        st.home = "/root".to_string();
        let api = Arc::new(MockApi::new(st));
        let env = DaytonaEnvironment::new(api.clone(), &cfg()).unwrap();
        env.cleanup(|| Ok(()));
        assert_eq!(env.sandbox_id(), None);
        let calls = api.calls();
        assert!(calls.iter().any(|c| c == "stop:sb-new"));
        assert!(!calls.iter().any(|c| c.starts_with("delete:")));
        // second cleanup is a no-op
        let before = api.calls().len();
        env.cleanup(|| Ok(()));
        assert_eq!(api.calls().len(), before);
    }

    #[test]
    fn test_cleanup_ephemeral_deletes() {
        let mut st = MockState::default();
        st.next_create_id = "sb-new".to_string();
        st.home = "/root".to_string();
        let api = Arc::new(MockApi::new(st));
        let mut c = cfg();
        c.persistent_filesystem = false;
        let env = DaytonaEnvironment::new(api.clone(), &c).unwrap();
        env.cleanup(|| Ok(()));
        let calls = api.calls();
        assert!(calls.iter().any(|c| c == "delete:sb-new"));
        assert!(!calls.iter().any(|c| c.starts_with("stop:")));
    }

    #[test]
    fn test_ensure_sandbox_ready_restarts_when_stopped() {
        let mut st = MockState::default();
        st.next_create_id = "sb-new".to_string();
        st.home = "/root".to_string();
        st.state.insert("sb-new".to_string(), SandboxState::Stopped);
        let api = Arc::new(MockApi::new(st));
        let env = DaytonaEnvironment::new(api.clone(), &cfg()).unwrap();
        env.ensure_sandbox_ready().unwrap();
        let calls = api.calls();
        // start called once during create-resume path? create path does not
        // start, so the only start is from ensure_sandbox_ready.
        assert!(calls.iter().any(|c| c == "start:sb-new"));
    }

    #[test]
    fn test_run_bash_wraps_and_execs() {
        let mut st = MockState::default();
        st.next_create_id = "sb-new".to_string();
        st.home = "/root".to_string();
        let api = Arc::new(MockApi::new(st));
        let env = DaytonaEnvironment::new(api.clone(), &cfg()).unwrap();
        let (out, code) = env.run_bash("echo hi", false, 120).unwrap();
        assert_eq!(code, 0);
        let _ = out;
        assert!(api
            .calls()
            .iter()
            .any(|c| c == "exec:sb-new:bash -c 'echo hi'"));
    }

    #[test]
    fn test_bulk_download_uses_pid_temp_and_cleans_up() {
        let mut st = MockState::default();
        st.next_create_id = "sb-new".to_string();
        st.home = "/root".to_string();
        let api = Arc::new(MockApi::new(st));
        let env = DaytonaEnvironment::new(api.clone(), &cfg()).unwrap();
        let dir = std::env::temp_dir();
        let dest = dir.join("daytona_test_dl.tar");
        env.daytona_bulk_download(dest.to_str().unwrap()).unwrap();
        let pid = std::process::id();
        let tar = format!("/tmp/.hermes_sync.{pid}.tar");
        let calls = api.calls();
        // tar create command with rel_base "root/.hermes"
        assert!(calls.iter().any(|c| c.contains(&format!(
            "exec:sb-new:tar cf {} -C / root/.hermes",
            shlex_quote(&tar)
        ))));
        assert!(calls.iter().any(|c| c == "download_file"));
        // cleanup rm -f
        assert!(calls
            .iter()
            .any(|c| c.contains(&format!("rm -f {}", shlex_quote(&tar)))));
        let _ = std::fs::remove_file(&dest);
    }

    #[test]
    fn test_bulk_upload_empty_is_noop() {
        let mut st = MockState::default();
        st.next_create_id = "sb-new".to_string();
        st.home = "/root".to_string();
        let api = Arc::new(MockApi::new(st));
        let env = DaytonaEnvironment::new(api.clone(), &cfg()).unwrap();
        let before = api.calls().len();
        env.daytona_bulk_upload(&[]).unwrap();
        assert_eq!(api.calls().len(), before);
    }
}
