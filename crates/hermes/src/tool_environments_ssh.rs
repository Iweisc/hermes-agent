//! SSH remote execution environment with ControlMaster connection persistence.
//!
//! Native Rust port of `tools/environments/ssh.py`.
//!
//! Spawn-per-call: every command spawns a fresh `ssh ... bash -c` process.
//! Uses SSH ControlMaster for connection reuse. The file-sync layer (the
//! Python `FileSyncManager`) is not yet ported to Rust; this module exposes
//! the transport callbacks it needs (`scp_upload`, `ssh_delete`,
//! `ssh_bulk_upload`, `ssh_bulk_download`, `iter_sync_files`) as public
//! methods so a future `FileSyncManager` port can drive them.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use sha2::{Digest, Sha256};

/// Errors raised by the SSH environment, mirroring the Python `RuntimeError`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SshError(pub String);

impl std::fmt::Display for SshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SshError {}

pub type SshResult<T> = Result<T, SshError>;

// ---------------------------------------------------------------------------
// shell quoting (POSIX, equivalent to Python's shlex.quote)
// ---------------------------------------------------------------------------

/// POSIX-shell-quote a string the way Python's `shlex.quote` does.
///
/// Empty strings become `''`. Strings containing only "safe" characters are
/// returned unchanged. Otherwise the string is single-quoted, with embedded
/// single quotes escaped as `'"'"'`.
pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    // shlex's _find_unsafe matches anything NOT in this set.
    let safe = |c: char| {
        c.is_ascii_alphanumeric() || matches!(c, '_' | '@' | '%' | '+' | '=' | ':' | ',' | '.' | '/' | '-')
    };
    if s.chars().all(safe) {
        return s.to_string();
    }
    // Replace ' with '"'"' and wrap in single quotes.
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// Build a shell `rm -f` command for a batch of remote paths.
pub fn quoted_rm_command(remote_paths: &[String]) -> String {
    let mut out = String::from("rm -f");
    for p in remote_paths {
        out.push(' ');
        out.push_str(&shlex_quote(p));
    }
    out
}

/// Build a shell `mkdir -p` command for a batch of directories.
pub fn quoted_mkdir_command(dirs: &[String]) -> String {
    let mut out = String::from("mkdir -p");
    for d in dirs {
        out.push(' ');
        out.push_str(&shlex_quote(d));
    }
    out
}

/// Extract sorted unique parent directories from (host, remote) pairs.
pub fn unique_parent_dirs(files: &[(String, String)]) -> Vec<String> {
    let mut set = std::collections::BTreeSet::new();
    for (_, remote) in files {
        let parent = Path::new(remote)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        // Python's Path.parent of "/" is "/", of "foo" is ".".
        let parent = if parent.is_empty() { ".".to_string() } else { parent };
        set.insert(parent);
    }
    set.into_iter().collect()
}

// ---------------------------------------------------------------------------
// availability check
// ---------------------------------------------------------------------------

/// Return the absolute path of *name* if it is on `PATH` (like `shutil.which`).
fn which(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Fail fast with a clear error when the SSH client is unavailable.
pub fn ensure_ssh_available() -> SshResult<()> {
    if which("ssh").is_none() {
        return Err(SshError(
            "SSH is not installed or not in PATH. Install OpenSSH client: apt install openssh-client"
                .to_string(),
        ));
    }
    if which("scp").is_none() {
        return Err(SshError(
            "SCP is not installed or not in PATH. Install OpenSSH client: apt install openssh-client"
                .to_string(),
        ));
    }
    Ok(())
}

/// Return the system temp directory (equivalent to `tempfile.gettempdir()`).
fn temp_dir() -> PathBuf {
    std::env::temp_dir()
}

// ---------------------------------------------------------------------------
// SSH environment
// ---------------------------------------------------------------------------

/// Run commands on a remote machine over SSH.
///
/// Mirrors the Python `SSHEnvironment`. The constructor establishes the
/// ControlMaster connection, detects the remote home, and creates the base
/// `~/.hermes` directory tree. File-sync (push/pull of skills/credentials)
/// is performed by a separate `FileSyncManager` (not yet ported); this type
/// provides the transport methods that manager invokes.
pub struct SshEnvironment {
    pub host: String,
    pub user: String,
    pub cwd: String,
    pub timeout: u64,
    pub port: u16,
    pub key_path: String,

    pub control_dir: PathBuf,
    pub control_socket: PathBuf,
    pub remote_home: String,
}

impl SshEnvironment {
    /// Construct and connect, mirroring `SSHEnvironment.__init__`.
    ///
    /// Note: unlike the Python original this does *not* drive the file-sync
    /// manager (it is not ported yet). It performs the SSH availability check,
    /// establishes the ControlMaster connection, detects the remote home and
    /// creates the base directory tree.
    pub fn new(
        host: &str,
        user: &str,
        cwd: &str,
        timeout: u64,
        port: u16,
        key_path: &str,
    ) -> SshResult<Self> {
        let control_dir = temp_dir().join("hermes-ssh");
        std::fs::create_dir_all(&control_dir)
            .map_err(|e| SshError(format!("failed to create control dir: {e}")))?;

        // Keep the socket filename short and deterministic so the full path
        // stays under the 104-byte sun_path limit macOS enforces.
        let socket_id = Self::socket_id(user, host, port);
        let control_socket = control_dir.join(format!("{socket_id}.sock"));

        ensure_ssh_available()?;

        let mut env = SshEnvironment {
            host: host.to_string(),
            user: user.to_string(),
            cwd: cwd.to_string(),
            timeout,
            port,
            key_path: key_path.to_string(),
            control_dir,
            control_socket,
            remote_home: String::new(),
        };

        env.establish_connection()?;
        env.remote_home = env.detect_remote_home();
        env.ensure_remote_dirs();

        Ok(env)
    }

    /// Deterministic 16-hex-char socket id derived from `user@host:port`.
    pub fn socket_id(user: &str, host: &str, port: u16) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("{user}@{host}:{port}").as_bytes());
        let digest = hasher.finalize();
        let hex = digest.iter().map(|b| format!("{b:02x}")).collect::<String>();
        hex[..16].to_string()
    }

    /// Build the base `ssh` argv with all ControlMaster/connection options.
    pub fn build_ssh_command(&self, extra_args: &[&str]) -> Vec<String> {
        let mut cmd: Vec<String> = vec!["ssh".to_string()];
        cmd.push("-o".into());
        cmd.push(format!("ControlPath={}", self.control_socket.display()));
        cmd.push("-o".into());
        cmd.push("ControlMaster=auto".into());
        cmd.push("-o".into());
        cmd.push("ControlPersist=300".into());
        cmd.push("-o".into());
        cmd.push("BatchMode=yes".into());
        cmd.push("-o".into());
        cmd.push("StrictHostKeyChecking=accept-new".into());
        cmd.push("-o".into());
        cmd.push("ConnectTimeout=10".into());
        if self.port != 22 {
            cmd.push("-p".into());
            cmd.push(self.port.to_string());
        }
        if !self.key_path.is_empty() {
            cmd.push("-i".into());
            cmd.push(self.key_path.clone());
        }
        for a in extra_args {
            cmd.push((*a).to_string());
        }
        cmd.push(format!("{}@{}", self.user, self.host));
        cmd
    }

    /// Run an argv (first element is the program) capturing stdout/stderr.
    /// Mirrors `subprocess.run(..., capture_output=True, text=True, timeout=...)`.
    /// A timeout maps to an error string identifying the call site.
    fn run_capture(
        argv: &[String],
        timeout: Duration,
        timeout_label: &str,
    ) -> SshResult<CapturedOutput> {
        let mut command = Command::new(&argv[0]);
        command.args(&argv[1..]);
        command.stdin(Stdio::null());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        run_with_timeout(command, timeout, timeout_label)
    }

    /// Establish the ControlMaster connection by running a trivial echo.
    fn establish_connection(&self) -> SshResult<()> {
        let mut cmd = self.build_ssh_command(&[]);
        cmd.push("echo 'SSH connection established'".into());
        match Self::run_capture(&cmd, Duration::from_secs(15), "establish") {
            Ok(result) => {
                if result.status_code != Some(0) {
                    let stderr = result.stderr.trim();
                    let error_msg = if !stderr.is_empty() {
                        stderr.to_string()
                    } else {
                        result.stdout.trim().to_string()
                    };
                    return Err(SshError(format!("SSH connection failed: {error_msg}")));
                }
                Ok(())
            }
            Err(SshError(e)) if e == "establish" => Err(SshError(format!(
                "SSH connection to {}@{} timed out",
                self.user, self.host
            ))),
            Err(e) => Err(e),
        }
    }

    /// Detect the remote user's home directory.
    fn detect_remote_home(&self) -> String {
        let mut cmd = self.build_ssh_command(&[]);
        cmd.push("echo $HOME".into());
        if let Ok(result) = Self::run_capture(&cmd, Duration::from_secs(10), "home") {
            let home = result.stdout.trim();
            if !home.is_empty() && result.status_code == Some(0) {
                log::debug!("SSH: remote home = {home}");
                return home.to_string();
            }
        }
        if self.user == "root" {
            return "/root".to_string();
        }
        format!("/home/{}", self.user)
    }

    /// Create the base `~/.hermes` directory tree on the remote in one call.
    fn ensure_remote_dirs(&self) {
        let base = format!("{}/.hermes", self.remote_home);
        let dirs = vec![
            base.clone(),
            format!("{base}/skills"),
            format!("{base}/credentials"),
            format!("{base}/cache"),
        ];
        let mut cmd = self.build_ssh_command(&[]);
        cmd.push(quoted_mkdir_command(&dirs));
        // Best-effort: Python ignores the result here.
        let _ = Self::run_capture(&cmd, Duration::from_secs(10), "ensure_dirs");
    }

    /// The remote `~/.hermes` base path used as the sync source root.
    pub fn sync_base(&self) -> String {
        format!("{}/.hermes", self.remote_home)
    }

    // ------------------------------------------------------------------
    // File sync transport callbacks (driven by a FileSyncManager)
    // ------------------------------------------------------------------

    /// Upload a single file via `scp` over the ControlMaster socket.
    pub fn scp_upload(&self, host_path: &str, remote_path: &str) -> SshResult<()> {
        let parent = Path::new(remote_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string());
        let mut mkdir_cmd = self.build_ssh_command(&[]);
        mkdir_cmd.push(format!("mkdir -p {}", shlex_quote(&parent)));
        let _ = Self::run_capture(&mkdir_cmd, Duration::from_secs(10), "scp_mkdir");

        let mut scp_cmd: Vec<String> = vec![
            "scp".into(),
            "-o".into(),
            format!("ControlPath={}", self.control_socket.display()),
        ];
        if self.port != 22 {
            scp_cmd.push("-P".into());
            scp_cmd.push(self.port.to_string());
        }
        if !self.key_path.is_empty() {
            scp_cmd.push("-i".into());
            scp_cmd.push(self.key_path.clone());
        }
        scp_cmd.push(host_path.to_string());
        scp_cmd.push(format!("{}@{}:{}", self.user, self.host, remote_path));
        let result = Self::run_capture(&scp_cmd, Duration::from_secs(30), "scp")?;
        if result.status_code != Some(0) {
            return Err(SshError(format!("scp failed: {}", result.stderr.trim())));
        }
        Ok(())
    }

    /// Batch-delete remote files in one SSH call.
    pub fn ssh_delete(&self, remote_paths: &[String]) -> SshResult<()> {
        let mut cmd = self.build_ssh_command(&[]);
        cmd.push(quoted_rm_command(remote_paths));
        let result = Self::run_capture(&cmd, Duration::from_secs(10), "rm")?;
        if result.status_code != Some(0) {
            return Err(SshError(format!(
                "remote rm failed: {}",
                result.stderr.trim()
            )));
        }
        Ok(())
    }

    /// Upload many files in a single `tar`-over-SSH stream.
    ///
    /// Stages each file as a symlink under a temp directory so the absolute
    /// remote layout is preserved, then pipes `tar c` into a remote
    /// `tar xf - --no-overwrite-dir -C /`.
    pub fn ssh_bulk_upload(&self, files: &[(String, String)]) -> SshResult<()> {
        if files.is_empty() {
            return Ok(());
        }

        let parents = unique_parent_dirs(files);
        if !parents.is_empty() {
            let mut cmd = self.build_ssh_command(&[]);
            cmd.push(quoted_mkdir_command(&parents));
            let result = Self::run_capture(&cmd, Duration::from_secs(30), "bulk_mkdir")?;
            if result.status_code != Some(0) {
                return Err(SshError(format!(
                    "remote mkdir failed: {}",
                    result.stderr.trim()
                )));
            }
        }

        // Symlink staging avoids fragile GNU tar --transform rules.
        let staging = TempDir::new("hermes-ssh-bulk-")?;
        for (host_path, remote_path) in files {
            // Strip leading '/' so the staged tree is relative.
            let rel = remote_path.trim_start_matches('/');
            let staged = staging.path().join(rel);
            if let Some(dir) = staged.parent() {
                std::fs::create_dir_all(dir).map_err(|e| {
                    SshError(format!("failed to create staging dir: {e}"))
                })?;
            }
            let abs = std::fs::canonicalize(host_path).unwrap_or_else(|_| {
                // Fall back to absolutising via cwd when canonicalize fails.
                if Path::new(host_path).is_absolute() {
                    PathBuf::from(host_path)
                } else {
                    std::env::current_dir()
                        .unwrap_or_default()
                        .join(host_path)
                }
            });
            symlink(&abs, &staged)
                .map_err(|e| SshError(format!("failed to stage symlink: {e}")))?;
        }

        let tar_cmd = vec![
            "tar".to_string(),
            "-chf".to_string(),
            "-".to_string(),
            "-C".to_string(),
            staging.path().to_string_lossy().to_string(),
            ".".to_string(),
        ];
        let mut ssh_cmd = self.build_ssh_command(&[]);
        // --no-overwrite-dir prevents tar from clobbering existing dir modes.
        ssh_cmd.push("tar xf - --no-overwrite-dir -C /".into());

        pipe_tar_over_ssh(&tar_cmd, &ssh_cmd)?;

        log::debug!("SSH: bulk-uploaded {} file(s) via tar pipe", files.len());
        Ok(())
    }

    /// Download remote `.hermes/` as a tar archive written to *dest*.
    pub fn ssh_bulk_download(&self, dest: &Path) -> SshResult<()> {
        // Tar from / with the full path so archive entries preserve absolute
        // paths (e.g. home/user/.hermes/skills/f.py).
        let rel_base = self.sync_base();
        let rel_base = rel_base.trim_start_matches('/');
        let mut ssh_cmd = self.build_ssh_command(&[]);
        ssh_cmd.push(format!("tar cf - -C / {}", shlex_quote(rel_base)));

        let file = std::fs::File::create(dest)
            .map_err(|e| SshError(format!("failed to open dest: {e}")))?;
        let mut command = Command::new(&ssh_cmd[0]);
        command.args(&ssh_cmd[1..]);
        command.stdin(Stdio::null());
        command.stdout(Stdio::from(file));
        command.stderr(Stdio::piped());
        let result = run_with_timeout(command, Duration::from_secs(120), "download")?;
        if result.status_code != Some(0) {
            return Err(SshError(format!(
                "SSH bulk download failed: {}",
                result.stderr.trim()
            )));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Execution
    // ------------------------------------------------------------------

    /// Build the argv that runs `bash` on the remote host for *cmd_string*.
    ///
    /// The Python original returns a live `subprocess.Popen` via `_popen_bash`;
    /// since process handling lives in the (unported) base environment, this
    /// returns the fully-formed argv so the caller can spawn it however it
    /// drives long-running processes.
    pub fn build_run_bash(&self, cmd_string: &str, login: bool) -> Vec<String> {
        let mut cmd = self.build_ssh_command(&[]);
        cmd.push("bash".into());
        if login {
            cmd.push("-l".into());
        }
        cmd.push("-c".into());
        cmd.push(shlex_quote(cmd_string));
        cmd
    }

    /// Tear down the ControlMaster connection and remove the socket.
    ///
    /// Mirrors `SSHEnvironment.cleanup` (minus the file-sync sync-back, which
    /// belongs to the not-yet-ported `FileSyncManager`).
    pub fn cleanup(&self) {
        if self.control_socket.exists() {
            let cmd = vec![
                "ssh".to_string(),
                "-o".to_string(),
                format!("ControlPath={}", self.control_socket.display()),
                "-O".to_string(),
                "exit".to_string(),
                format!("{}@{}", self.user, self.host),
            ];
            let mut command = Command::new(&cmd[0]);
            command.args(&cmd[1..]);
            command.stdin(Stdio::null());
            command.stdout(Stdio::null());
            command.stderr(Stdio::null());
            let _ = run_with_timeout(command, Duration::from_secs(5), "exit");
            let _ = std::fs::remove_file(&self.control_socket);
        }
    }
}

// ---------------------------------------------------------------------------
// subprocess helpers
// ---------------------------------------------------------------------------

struct CapturedOutput {
    status_code: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Spawn *command*, enforcing a wall-clock *timeout*. On timeout the child is
/// killed and `Err(SshError(timeout_label))` is returned (callers compare the
/// label to detect the timeout case, mirroring `subprocess.TimeoutExpired`).
fn run_with_timeout(
    mut command: Command,
    timeout: Duration,
    timeout_label: &str,
) -> SshResult<CapturedOutput> {
    let mut child = command
        .spawn()
        .map_err(|e| SshError(format!("failed to spawn process: {e}")))?;

    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => break,
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(SshError(timeout_label.to_string()));
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(SshError(format!("wait failed: {e}"))),
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|e| SshError(format!("collect output failed: {e}")))?;
    Ok(CapturedOutput {
        status_code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

/// Pipe `tar c` (local) into `tar x` (remote over ssh), draining stderr.
///
/// Reproduces the Python `_ssh_bulk_upload` process plumbing: tar's stdout is
/// connected to ssh's stdin, and both exit codes are checked with tar first.
fn pipe_tar_over_ssh(tar_cmd: &[String], ssh_cmd: &[String]) -> SshResult<()> {
    let mut tar_proc = Command::new(&tar_cmd[0])
        .args(&tar_cmd[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| SshError(format!("failed to spawn tar: {e}")))?;

    let tar_stdout = tar_proc
        .stdout
        .take()
        .ok_or_else(|| SshError("tar stdout unavailable".to_string()))?;

    let ssh_spawn = Command::new(&ssh_cmd[0])
        .args(&ssh_cmd[1..])
        .stdin(Stdio::from(tar_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    let mut ssh_proc = match ssh_spawn {
        Ok(p) => p,
        Err(e) => {
            let _ = tar_proc.kill();
            let _ = tar_proc.wait();
            return Err(SshError(format!("failed to spawn ssh: {e}")));
        }
    };

    // Wait on ssh with a timeout, draining its stderr.
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(120);
    loop {
        match ssh_proc.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = tar_proc.kill();
                    let _ = ssh_proc.kill();
                    let _ = tar_proc.wait();
                    let _ = ssh_proc.wait();
                    return Err(SshError("SSH bulk upload timed out".to_string()));
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(SshError(format!("ssh wait failed: {e}"))),
        }
    }

    let ssh_output = ssh_proc
        .wait_with_output()
        .map_err(|e| SshError(format!("ssh output failed: {e}")))?;
    let tar_output = tar_proc
        .wait_with_output()
        .map_err(|e| SshError(format!("tar output failed: {e}")))?;

    if tar_output.status.code() != Some(0) {
        return Err(SshError(format!(
            "tar create failed (rc={}): {}",
            tar_output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&tar_output.stderr).trim()
        )));
    }
    if ssh_output.status.code() != Some(0) {
        return Err(SshError(format!(
            "tar extract over SSH failed (rc={}): {}",
            ssh_output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&ssh_output.stderr).trim()
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn symlink(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(src, dst)
}

#[cfg(not(unix))]
fn symlink(src: &Path, dst: &Path) -> std::io::Result<()> {
    // Fall back to a copy on non-unix platforms (tar -h dereferences anyway).
    std::fs::copy(src, dst).map(|_| ())
}

/// Minimal self-cleaning temp directory (like `tempfile.TemporaryDirectory`).
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> SshResult<Self> {
        let base = temp_dir();
        // Derive a unique suffix from time + pid.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        for attempt in 0..1024u32 {
            let candidate = base.join(format!("{prefix}{pid}-{nanos}-{attempt}"));
            match std::fs::create_dir(&candidate) {
                Ok(_) => return Ok(TempDir { path: candidate }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(SshError(format!("failed to create temp dir: {e}")))
                }
            }
        }
        Err(SshError("exhausted temp dir name attempts".to_string()))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

// Silence unused-import warning when the file is compiled without exercising
// Write (used indirectly through process plumbing in some configurations).
#[allow(dead_code)]
fn _assert_write_in_scope() {
    let _f: Option<&dyn Write> = None;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shlex_quote_basic() {
        assert_eq!(shlex_quote(""), "''");
        assert_eq!(shlex_quote("simple"), "simple");
        assert_eq!(shlex_quote("/home/user/.hermes"), "/home/user/.hermes");
        assert_eq!(shlex_quote("a b"), "'a b'");
        assert_eq!(shlex_quote("it's"), "'it'\"'\"'s'");
        assert_eq!(shlex_quote("$HOME"), "'$HOME'");
    }

    #[test]
    fn quoted_mkdir_and_rm() {
        let dirs = vec!["/a b".to_string(), "/c".to_string()];
        assert_eq!(quoted_mkdir_command(&dirs), "mkdir -p '/a b' /c");
        let paths = vec!["/x".to_string(), "/y z".to_string()];
        assert_eq!(quoted_rm_command(&paths), "rm -f /x '/y z'");
    }

    #[test]
    fn unique_parents_sorted_dedup() {
        let files = vec![
            ("h1".to_string(), "/root/.hermes/a.py".to_string()),
            ("h2".to_string(), "/root/.hermes/b.py".to_string()),
            ("h3".to_string(), "/root/.hermes/skills/c.py".to_string()),
        ];
        let parents = unique_parent_dirs(&files);
        assert_eq!(
            parents,
            vec![
                "/root/.hermes".to_string(),
                "/root/.hermes/skills".to_string()
            ]
        );
    }

    #[test]
    fn socket_id_deterministic_and_16_hex() {
        let a = SshEnvironment::socket_id("user", "host", 22);
        let b = SshEnvironment::socket_id("user", "host", 22);
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        // Different port -> different id.
        assert_ne!(a, SshEnvironment::socket_id("user", "host", 2222));
        // Known vector: sha256("user@host:22")[:16].
        let mut h = Sha256::new();
        h.update(b"user@host:22");
        let full: String = h.finalize().iter().map(|x| format!("{x:02x}")).collect();
        assert_eq!(a, full[..16]);
    }

    fn fake_env(port: u16, key: &str) -> SshEnvironment {
        SshEnvironment {
            host: "example.com".to_string(),
            user: "alice".to_string(),
            cwd: "~".to_string(),
            timeout: 60,
            port,
            key_path: key.to_string(),
            control_dir: PathBuf::from("/tmp/hermes-ssh"),
            control_socket: PathBuf::from("/tmp/hermes-ssh/abc.sock"),
            remote_home: "/home/alice".to_string(),
        }
    }

    #[test]
    fn build_ssh_command_default_port_no_key() {
        let env = fake_env(22, "");
        let cmd = env.build_ssh_command(&[]);
        assert_eq!(cmd[0], "ssh");
        assert!(cmd.contains(&"ControlPath=/tmp/hermes-ssh/abc.sock".to_string()));
        assert!(cmd.contains(&"ControlMaster=auto".to_string()));
        assert!(cmd.contains(&"ControlPersist=300".to_string()));
        assert!(cmd.contains(&"BatchMode=yes".to_string()));
        assert!(cmd.contains(&"StrictHostKeyChecking=accept-new".to_string()));
        assert!(cmd.contains(&"ConnectTimeout=10".to_string()));
        // No -p, no -i.
        assert!(!cmd.contains(&"-p".to_string()));
        assert!(!cmd.contains(&"-i".to_string()));
        // Target is last.
        assert_eq!(cmd.last().unwrap(), "alice@example.com");
    }

    #[test]
    fn build_ssh_command_custom_port_and_key() {
        let env = fake_env(2222, "/path/to/key");
        let cmd = env.build_ssh_command(&["-q"]);
        let joined = cmd.join(" ");
        assert!(joined.contains("-p 2222"));
        assert!(joined.contains("-i /path/to/key"));
        assert!(joined.contains("-q"));
        assert_eq!(cmd.last().unwrap(), "alice@example.com");
    }

    #[test]
    fn build_run_bash_login_and_nonlogin() {
        let env = fake_env(22, "");
        // "echo" is shlex-safe (no special chars), so it is passed verbatim.
        let nonlogin = env.build_run_bash("echo", false);
        assert_eq!(&nonlogin[nonlogin.len() - 3..], &["bash", "-c", "echo"]);

        let login = env.build_run_bash("echo", true);
        assert_eq!(&login[login.len() - 4..], &["bash", "-l", "-c", "echo"]);

        // A command with a space gets single-quoted (matches shlex.quote).
        let spaced = env.build_run_bash("echo hi", false);
        assert_eq!(&spaced[spaced.len() - 3..], &["bash", "-c", "'echo hi'"]);

        // Quoting kicks in for unsafe input.
        let quoted = env.build_run_bash("ls -la && rm x", false);
        assert_eq!(quoted.last().unwrap(), "'ls -la && rm x'");
    }

    #[test]
    fn sync_base_path() {
        let env = fake_env(22, "");
        assert_eq!(env.sync_base(), "/home/alice/.hermes");
    }
}
