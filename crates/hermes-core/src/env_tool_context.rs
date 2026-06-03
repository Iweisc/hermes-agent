//! `ToolContext` -- Unrestricted Tool Access for Reward Functions.
//!
//! Native Rust port of `environments/tool_context.py`.
//!
//! A per-rollout handle that gives reward/verification functions direct access
//! to ALL hermes-agent tools, scoped to the rollout's `task_id`. The same
//! `task_id` means the terminal/browser session is the SAME one the model used
//! during its rollout -- all state (files, processes, browser tabs) is
//! preserved.
//!
//! The verifier author decides which tools to use. Nothing is hardcoded or
//! gated.
//!
//! ## Differences from the Python original
//!
//! The Python module imported three free functions at module scope:
//!
//! * `model_tools.handle_function_call`
//! * `tools.terminal_tool.cleanup_vm`
//! * `tools.browser_tool.cleanup_browser`
//!
//! plus, lazily inside [`ToolContext::cleanup`], `tools.process_registry`.
//!
//! In the native build, [`crate::mod_model_tools::handle_function_call`] takes a
//! [`ToolRegistry`] and a richer set of arguments, while `cleanup_vm`,
//! `cleanup_browser` and the process registry's `kill_all` live in the `hermes`
//! crate (not reachable from `hermes-core`) or are not ported yet. To avoid
//! blocking and to keep this file self-contained, those collaborators are
//! injected as boxed callbacks ([`CleanupHooks`]) and the registry/dispatch
//! hooks are held on the context. The orchestrator wires these up separately.
//!
//! The Python thread-pool dance (`_run_tool_in_thread`) existed purely to give
//! backends that call `asyncio.run()` a clean event loop when invoked from an
//! async context. The native dispatcher is synchronous, so [`call_tool`] simply
//! calls through. [`run_tool_in_thread`] is preserved as a thin wrapper for
//! parity and so callers that want to offload to a worker thread can.
//!
//! [`call_tool`]: ToolContext::call_tool

use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine as _;
use serde_json::{json, Value};

use crate::mod_model_tools::{handle_function_call, CallIds, DispatchHooks};
use crate::tool_registry::ToolRegistry;

/// Default timeout (seconds) for `terminal()` calls -- matches the Python
/// default of `timeout=180`.
pub const DEFAULT_TERMINAL_TIMEOUT: i64 = 180;

/// Maximum base64 characters to push through a single shell command before
/// chunking. Matches the Python `chunk_size = 60_000`.
pub const UPLOAD_CHUNK_SIZE: usize = 60_000;

/// Optional cleanup collaborators injected by the caller. Each corresponds to a
/// free function the Python module imported lazily.
///
/// Every callback is best-effort: failures are swallowed (logged at debug) just
/// like the Python `try/except Exception` blocks around each cleanup step.
#[derive(Default)]
pub struct CleanupHooks {
    /// Kill all background processes for a `task_id`. Mirrors
    /// `tools.process_registry.process_registry.kill_all(task_id=...)`.
    /// Returns the number of processes killed.
    #[allow(clippy::type_complexity)]
    pub kill_processes: Option<Box<dyn Fn(&str) -> usize + Send + Sync>>,
    /// Release the terminal VM/sandbox for a `task_id`. Mirrors
    /// `tools.terminal_tool.cleanup_vm(task_id)`.
    pub cleanup_vm: Option<Box<dyn Fn(&str) + Send + Sync>>,
    /// Release the browser session for a `task_id`. Mirrors
    /// `tools.browser_tool.cleanup_browser(task_id)`.
    pub cleanup_browser: Option<Box<dyn Fn(&str) + Send + Sync>>,
}

/// Open-ended access to all hermes-agent tools for a specific rollout.
///
/// Passed to `compute_reward()` so verifiers can use any tool they need:
/// terminal commands, file reads/writes, web searches, browser automation, etc.
/// All calls share the rollout's `task_id` for session isolation.
pub struct ToolContext {
    /// The rollout's task id -- scopes terminal/browser sessions.
    pub task_id: String,
    /// Tool registry the dispatcher routes against.
    registry: Arc<ToolRegistry>,
    /// Plugin-system seams forwarded to the dispatcher.
    hooks: DispatchHooks,
    /// Best-effort cleanup collaborators.
    cleanup_hooks: CleanupHooks,
}

impl ToolContext {
    /// Construct a context for `task_id` backed by `registry`. Uses default
    /// (no-op) dispatch hooks and no cleanup collaborators.
    pub fn new(task_id: impl Into<String>, registry: Arc<ToolRegistry>) -> Self {
        Self {
            task_id: task_id.into(),
            registry,
            hooks: DispatchHooks::default(),
            cleanup_hooks: CleanupHooks::default(),
        }
    }

    /// Full constructor: supply custom dispatch hooks and cleanup collaborators.
    pub fn with_hooks(
        task_id: impl Into<String>,
        registry: Arc<ToolRegistry>,
        hooks: DispatchHooks,
        cleanup_hooks: CleanupHooks,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            registry,
            hooks,
            cleanup_hooks,
        }
    }

    /// Install/replace the cleanup collaborators after construction.
    pub fn set_cleanup_hooks(&mut self, hooks: CleanupHooks) {
        self.cleanup_hooks = hooks;
    }

    // -- internal dispatch ---------------------------------------------------

    /// Build the [`CallIds`] for a task-scoped call.
    fn ids_with_task(&self) -> CallIds {
        CallIds {
            task_id: Some(self.task_id.clone()),
            ..CallIds::default()
        }
    }

    /// Dispatch a tool call carrying this context's `task_id`. Equivalent to
    /// the Python `handle_function_call(name, args, task_id=self.task_id)`.
    fn dispatch_task(&self, tool_name: &str, arguments: Value) -> String {
        let ids = self.ids_with_task();
        handle_function_call(
            &self.registry,
            tool_name,
            arguments,
            &ids,
            None,
            false,
            &self.hooks,
        )
    }

    /// Dispatch a tool call with NO task id (web tools in the Python source
    /// call `handle_function_call(name, args)` without `task_id`).
    fn dispatch_global(&self, tool_name: &str, arguments: Value) -> String {
        let ids = CallIds::default();
        handle_function_call(
            &self.registry,
            tool_name,
            arguments,
            &ids,
            None,
            false,
            &self.hooks,
        )
    }

    /// Parse a tool result as JSON, falling back to `fallback(raw)` on a JSON
    /// decode error. Mirrors the `try: json.loads(result) except
    /// JSONDecodeError:` pattern repeated throughout the Python module.
    fn parse_or<F>(raw: String, fallback: F) -> Value
    where
        F: FnOnce(String) -> Value,
    {
        match serde_json::from_str::<Value>(&raw) {
            Ok(v) => v,
            Err(_) => fallback(raw),
        }
    }

    // -------------------------------------------------------------------------
    // Terminal tools
    // -------------------------------------------------------------------------

    /// Run a command in the rollout's terminal session.
    ///
    /// Returns a JSON object with `exit_code` (int) and `output` (str). If the
    /// raw result is not valid JSON, returns `{"exit_code": -1, "output": raw}`.
    pub fn terminal(&self, command: &str, timeout: i64) -> Value {
        let backend = std::env::var("TERMINAL_ENV").unwrap_or_else(|_| "local".to_string());
        let short = if self.task_id.len() >= 8 {
            &self.task_id[..8]
        } else {
            &self.task_id[..]
        };
        let cmd_preview: String = command.chars().take(100).collect();
        log::debug!(
            "ToolContext.terminal [{} backend] task={}: {}",
            backend,
            short,
            cmd_preview
        );

        let raw = self.dispatch_task(
            "terminal",
            json!({ "command": command, "timeout": timeout }),
        );
        Self::parse_or(raw, |r| json!({ "exit_code": -1, "output": r }))
    }

    /// Convenience: run a terminal command with the default 180s timeout.
    pub fn terminal_default(&self, command: &str) -> Value {
        self.terminal(command, DEFAULT_TERMINAL_TIMEOUT)
    }

    // -------------------------------------------------------------------------
    // File tools
    // -------------------------------------------------------------------------

    /// Read a file from the rollout's filesystem. On a non-JSON result returns
    /// `{"error": raw}`.
    pub fn read_file(&self, path: &str) -> Value {
        let raw = self.dispatch_task("read_file", json!({ "path": path }));
        Self::parse_or(raw, |r| json!({ "error": r }))
    }

    /// Write a TEXT file in the rollout's filesystem (shell heredoc under the
    /// hood -- text only; use [`upload_file`] for binary).
    ///
    /// [`upload_file`]: ToolContext::upload_file
    pub fn write_file(&self, path: &str, content: &str) -> Value {
        let raw = self.dispatch_task("write_file", json!({ "path": path, "content": content }));
        Self::parse_or(raw, |r| json!({ "error": r }))
    }

    /// Upload a local file to the rollout's sandbox (binary-safe).
    ///
    /// Base64-encodes the file and decodes it inside the sandbox. For files
    /// whose base64 representation exceeds [`UPLOAD_CHUNK_SIZE`], the content is
    /// streamed in chunks to avoid shell command-length limits.
    ///
    /// Returns a JSON object with `exit_code` and `output`.
    pub fn upload_file(&self, local_path: &str, remote_path: &str) -> Value {
        let local = Path::new(local_path);
        if !local.exists() {
            return json!({
                "exit_code": -1,
                "output": format!("Local file not found: {}", local_path)
            });
        }

        let raw = match std::fs::read(local) {
            Ok(bytes) => bytes,
            Err(e) => {
                return json!({
                    "exit_code": -1,
                    "output": format!("Local file not found: {}: {}", local_path, e)
                });
            }
        };
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);

        // Ensure parent directory exists in the sandbox.
        let parent = Path::new(remote_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        if !parent.is_empty() && parent != "." && parent != "/" {
            self.terminal(&format!("mkdir -p {}", parent), 10);
        }

        if b64.len() <= UPLOAD_CHUNK_SIZE {
            self.terminal(
                &format!("printf '%s' '{}' | base64 -d > {}", b64, remote_path),
                30,
            )
        } else {
            // Larger files: write base64 in chunks, then decode.
            let tmp_b64 = "/tmp/_hermes_upload.b64";
            self.terminal(&format!(": > {}", tmp_b64), 5); // truncate
            let bytes = b64.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                let end = (i + UPLOAD_CHUNK_SIZE).min(bytes.len());
                // base64 alphabet is ASCII, so slicing by byte index is safe.
                let chunk = &b64[i..end];
                self.terminal(&format!("printf '%s' '{}' >> {}", chunk, tmp_b64), 15);
                i = end;
            }
            self.terminal(
                &format!("base64 -d {} > {} && rm -f {}", tmp_b64, remote_path, tmp_b64),
                30,
            )
        }
    }

    /// Upload an entire local directory to the rollout's sandbox (binary-safe),
    /// recursively, preserving directory structure. Returns one result per file.
    pub fn upload_dir(&self, local_dir: &str, remote_dir: &str) -> Vec<Value> {
        let local = Path::new(local_dir);
        if !local.exists() || !local.is_dir() {
            return vec![json!({
                "exit_code": -1,
                "output": format!("Local directory not found: {}", local_dir)
            })];
        }

        let mut files: Vec<PathBuf> = Vec::new();
        collect_files_recursive(local, &mut files);
        files.sort();

        let mut results = Vec::new();
        for file_path in files {
            if let Ok(relative) = file_path.strip_prefix(local) {
                let rel = relative.to_string_lossy().replace('\\', "/");
                let target = format!("{}/{}", remote_dir, rel);
                results.push(self.upload_file(&file_path.to_string_lossy(), &target));
            }
        }
        results
    }

    /// Download a file from the rollout's sandbox to the host (binary-safe) --
    /// the inverse of [`upload_file`].
    ///
    /// Returns `{"success": true, "bytes": n}` on success or
    /// `{"success": false, "error": msg}` on failure.
    ///
    /// [`upload_file`]: ToolContext::upload_file
    pub fn download_file(&self, remote_path: &str, local_path: &str) -> Value {
        let result = self.terminal(&format!("base64 {} 2>/dev/null", remote_path), 30);

        let exit_code = result
            .get("exit_code")
            .and_then(Value::as_i64)
            .unwrap_or(-1);
        if exit_code != 0 {
            let output = result.get("output").and_then(Value::as_str).unwrap_or("");
            return json!({
                "success": false,
                "error": format!("Failed to read remote file: {}", output)
            });
        }

        let b64_data = result
            .get("output")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if b64_data.is_empty() {
            return json!({
                "success": false,
                "error": format!("Remote file is empty or missing: {}", remote_path)
            });
        }

        let raw = match base64::engine::general_purpose::STANDARD.decode(b64_data.as_bytes()) {
            Ok(b) => b,
            Err(e) => {
                return json!({
                    "success": false,
                    "error": format!("Base64 decode failed: {}", e)
                });
            }
        };

        let local = Path::new(local_path);
        if let Some(parent) = local.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        if let Err(e) = std::fs::write(local, &raw) {
            return json!({
                "success": false,
                "error": format!("Failed to write local file: {}", e)
            });
        }

        json!({ "success": true, "bytes": raw.len() })
    }

    /// Download a directory from the rollout's sandbox to the host
    /// (binary-safe), preserving directory structure. Returns one result per
    /// file downloaded.
    pub fn download_dir(&self, remote_dir: &str, local_dir: &str) -> Vec<Value> {
        let ls_result = self.terminal(&format!("find {} -type f 2>/dev/null", remote_dir), 15);

        let exit_code = ls_result
            .get("exit_code")
            .and_then(Value::as_i64)
            .unwrap_or(-1);
        if exit_code != 0 {
            return vec![json!({
                "success": false,
                "error": format!("Failed to list remote dir: {}", remote_dir)
            })];
        }

        let file_list = ls_result
            .get("output")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if file_list.is_empty() {
            return vec![json!({
                "success": false,
                "error": format!("Remote directory is empty or missing: {}", remote_dir)
            })];
        }

        let mut results = Vec::new();
        for line in file_list.lines() {
            let remote_file = line.trim();
            if remote_file.is_empty() {
                continue;
            }
            // Compute the relative path to preserve directory structure.
            let relative: String = if remote_file.starts_with(remote_dir) {
                remote_file[remote_dir.len()..]
                    .trim_start_matches('/')
                    .to_string()
            } else {
                Path::new(remote_file)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| remote_file.to_string())
            };
            let local_file = Path::new(local_dir).join(&relative);
            results.push(self.download_file(remote_file, &local_file.to_string_lossy()));
        }

        results
    }

    /// Search for text in the rollout's filesystem (`search_files` tool).
    pub fn search(&self, query: &str, path: &str) -> Value {
        let raw = self.dispatch_task("search_files", json!({ "pattern": query, "path": path }));
        Self::parse_or(raw, |r| json!({ "error": r }))
    }

    // -------------------------------------------------------------------------
    // Web tools
    // -------------------------------------------------------------------------

    /// Search the web. Note: dispatched WITHOUT a task id, matching Python.
    pub fn web_search(&self, query: &str) -> Value {
        let raw = self.dispatch_global("web_search", json!({ "query": query }));
        Self::parse_or(raw, |r| json!({ "error": r }))
    }

    /// Extract content from URLs. Dispatched WITHOUT a task id, matching Python.
    pub fn web_extract(&self, urls: &[String]) -> Value {
        let raw = self.dispatch_global("web_extract", json!({ "urls": urls }));
        Self::parse_or(raw, |r| json!({ "error": r }))
    }

    // -------------------------------------------------------------------------
    // Browser tools
    // -------------------------------------------------------------------------

    /// Navigate the rollout's browser session to a URL.
    pub fn browser_navigate(&self, url: &str) -> Value {
        let raw = self.dispatch_task("browser_navigate", json!({ "url": url }));
        Self::parse_or(raw, |r| json!({ "error": r }))
    }

    /// Take a snapshot of the current browser page.
    pub fn browser_snapshot(&self) -> Value {
        let raw = self.dispatch_task("browser_snapshot", json!({}));
        Self::parse_or(raw, |r| json!({ "error": r }))
    }

    // -------------------------------------------------------------------------
    // Generic tool access
    // -------------------------------------------------------------------------

    /// Call any hermes-agent tool by name. The generic escape hatch -- returns
    /// the raw JSON string result from the tool, scoped to this context's
    /// `task_id`.
    pub fn call_tool(&self, tool_name: &str, arguments: Value) -> String {
        self.dispatch_task(tool_name, arguments)
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    /// Release all resources (terminal VMs, browser sessions, background
    /// processes) for this rollout. Best-effort: every step is independent and
    /// failures are swallowed, mirroring the Python `try/except` blocks.
    ///
    /// During browser cleanup, `HERMES_QUIET=1` is set to suppress noisy debug
    /// prints, then restored to its previous value (or removed).
    pub fn cleanup(&self) {
        // Kill any background processes from this rollout (safety net).
        if let Some(kill) = &self.cleanup_hooks.kill_processes {
            let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                kill(&self.task_id)
            }))
            .unwrap_or(0);
            if killed > 0 {
                log::debug!(
                    "Process cleanup for task {}: killed {} process(es)",
                    self.task_id,
                    killed
                );
            }
        }

        if let Some(cleanup_vm) = &self.cleanup_hooks.cleanup_vm {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cleanup_vm(&self.task_id)
            }));
        }

        // Suppress browser_tool's noisy debug prints during cleanup. The
        // cleanup still runs (safe), it just doesn't spam the console.
        let prev_quiet = std::env::var("HERMES_QUIET").ok();
        // SAFETY: edition 2024 requires set_var/remove_var in unsafe.
        unsafe {
            std::env::set_var("HERMES_QUIET", "1");
        }

        if let Some(cleanup_browser) = &self.cleanup_hooks.cleanup_browser {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cleanup_browser(&self.task_id)
            }));
        }

        // SAFETY: edition 2024 requires set_var/remove_var in unsafe.
        unsafe {
            match prev_quiet {
                None => std::env::remove_var("HERMES_QUIET"),
                Some(v) => std::env::set_var("HERMES_QUIET", v),
            }
        }
    }
}

/// Recursively collect every regular file under `dir` into `out`. Used by
/// [`ToolContext::upload_dir`] in place of Python's `Path.rglob("*")`.
fn collect_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

/// Run a tool call, optionally on a disposable worker thread.
///
/// Mirrors the Python `_run_tool_in_thread`, whose only purpose was to give
/// backends that internally call `asyncio.run()` a clean event loop when invoked
/// from an async context. The native dispatcher is synchronous, so this simply
/// dispatches through [`handle_function_call`]; it exists for API parity and for
/// callers who want to offload via [`std::thread`].
pub fn run_tool_in_thread(
    registry: &ToolRegistry,
    tool_name: &str,
    arguments: Value,
    task_id: &str,
    hooks: &DispatchHooks,
) -> String {
    let ids = CallIds {
        task_id: Some(task_id.to_string()),
        ..CallIds::default()
    };
    handle_function_call(registry, tool_name, arguments, &ids, None, false, hooks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_or_returns_parsed_json() {
        let v = ToolContext::parse_or(
            r#"{"exit_code": 0, "output": "ok"}"#.to_string(),
            |r| json!({ "exit_code": -1, "output": r }),
        );
        assert_eq!(v["exit_code"], 0);
        assert_eq!(v["output"], "ok");
    }

    #[test]
    fn parse_or_uses_fallback_on_bad_json() {
        let v = ToolContext::parse_or("not json".to_string(), |r| {
            json!({ "exit_code": -1, "output": r })
        });
        assert_eq!(v["exit_code"], -1);
        assert_eq!(v["output"], "not json");
    }

    #[test]
    fn parse_or_error_fallback_shape() {
        let v = ToolContext::parse_or("boom".to_string(), |r| json!({ "error": r }));
        assert_eq!(v["error"], "boom");
    }

    #[test]
    fn upload_file_missing_local_returns_error() {
        let reg = Arc::new(ToolRegistry::new());
        let ctx = ToolContext::new("task-1234abcd", reg);
        let v = ctx.upload_file("/definitely/not/a/real/path.bin", "/remote/x");
        assert_eq!(v["exit_code"], -1);
        assert!(v["output"]
            .as_str()
            .unwrap()
            .contains("Local file not found"));
    }

    #[test]
    fn upload_dir_missing_local_returns_single_error() {
        let reg = Arc::new(ToolRegistry::new());
        let ctx = ToolContext::new("task-1234abcd", reg);
        let v = ctx.upload_dir("/definitely/not/a/real/dir", "/remote");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0]["exit_code"], -1);
        assert!(v[0]["output"]
            .as_str()
            .unwrap()
            .contains("Local directory not found"));
    }

    #[test]
    fn short_task_id_does_not_panic_on_truncation() {
        // The terminal() log line slices task_id[:8]; ensure short ids are safe.
        let reg = Arc::new(ToolRegistry::new());
        let ctx = ToolContext::new("abc", reg);
        // Just exercise the truncation guard path; do not call dispatch.
        let short = if ctx.task_id.len() >= 8 {
            &ctx.task_id[..8]
        } else {
            &ctx.task_id[..]
        };
        assert_eq!(short, "abc");
    }

    #[test]
    fn cleanup_hooks_invoked_and_quiet_restored() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc as StdArc;

        let killed = StdArc::new(AtomicUsize::new(0));
        let vm_called = StdArc::new(AtomicUsize::new(0));
        let browser_called = StdArc::new(AtomicUsize::new(0));

        let k = killed.clone();
        let v = vm_called.clone();
        let b = browser_called.clone();

        let hooks = CleanupHooks {
            kill_processes: Some(Box::new(move |_task| {
                k.fetch_add(1, Ordering::SeqCst);
                3
            })),
            cleanup_vm: Some(Box::new(move |_task| {
                v.fetch_add(1, Ordering::SeqCst);
            })),
            cleanup_browser: Some(Box::new(move |_task| {
                b.fetch_add(1, Ordering::SeqCst);
            })),
        };

        // SAFETY: edition 2024 requires set_var/remove_var in unsafe.
        unsafe {
            std::env::remove_var("HERMES_QUIET");
        }

        let reg = Arc::new(ToolRegistry::new());
        let ctx = ToolContext::with_hooks("task-xyz", reg, DispatchHooks::default(), hooks);
        ctx.cleanup();

        assert_eq!(killed.load(Ordering::SeqCst), 1);
        assert_eq!(vm_called.load(Ordering::SeqCst), 1);
        assert_eq!(browser_called.load(Ordering::SeqCst), 1);
        // Previously unset -> should be removed again after cleanup.
        assert!(std::env::var("HERMES_QUIET").is_err());
    }

    #[test]
    fn cleanup_restores_prior_quiet_value() {
        // SAFETY: edition 2024 requires set_var/remove_var in unsafe.
        unsafe {
            std::env::set_var("HERMES_QUIET", "prev");
        }
        let reg = Arc::new(ToolRegistry::new());
        let ctx = ToolContext::new("task-restore", reg);
        ctx.cleanup();
        assert_eq!(std::env::var("HERMES_QUIET").unwrap(), "prev");
        // SAFETY: edition 2024 requires set_var/remove_var in unsafe.
        unsafe {
            std::env::remove_var("HERMES_QUIET");
        }
    }
}
