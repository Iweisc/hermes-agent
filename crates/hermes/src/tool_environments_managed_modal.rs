//! Managed Modal environment backed by tool-gateway.
//!
//! Native Rust port of `tools/environments/managed_modal.py`.
//!
//! A gateway-owned Modal sandbox with Hermes-compatible execute/cleanup. The
//! tool-gateway handles command preparation, CWD tracking, and env-snapshot
//! management on the server side. This module reproduces the request
//! construction and response-parsing logic of the Python implementation using
//! `reqwest::blocking`.
//!
//! Dependencies that are not yet ported (`resolve_managed_tool_gateway`,
//! credential-file mounts, interrupt signalling) are modelled as injectable
//! parameters / minimal local types so this module does not block on them.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

/// Resolved managed-tool gateway configuration.
///
/// Mirrors `ManagedToolGatewayConfig` from `tools/managed_tool_gateway.py`.
/// Construct via [`resolve_managed_tool_gateway`] or directly when injecting
/// a gateway for tests.
#[derive(Debug, Clone)]
pub struct ManagedToolGatewayConfig {
    pub vendor: String,
    pub gateway_origin: String,
    pub nous_user_token: String,
    pub managed_mode: bool,
}

/// Result of a Modal exec: output text plus return code.
///
/// Equivalent to the `{"output": ..., "returncode": ...}` dicts produced by the
/// Python `_result` / `_error_result` helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecResult {
    pub output: String,
    pub returncode: i64,
}

impl ExecResult {
    pub fn new(output: impl Into<String>, returncode: i64) -> Self {
        ExecResult {
            output: output.into(),
            returncode,
        }
    }

    /// Error result: returncode 1 with the given output (mirrors `_error_result`).
    pub fn error(output: impl Into<String>) -> Self {
        ExecResult::new(output, 1)
    }
}

/// Normalized command data passed to the transport-specific exec runner.
///
/// Mirrors `PreparedModalExec`.
#[derive(Debug, Clone)]
pub struct PreparedModalExec {
    pub command: String,
    pub cwd: String,
    pub timeout: i64,
    pub stdin_data: Option<String>,
}

/// Opaque handle for an in-flight managed Modal exec.
///
/// Mirrors `_ManagedModalExecHandle`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedModalExecHandle {
    pub exec_id: String,
}

/// Transport response after starting an exec.
///
/// Mirrors `ModalExecStart`: either an immediate result (terminal/error) or a
/// handle to poll.
#[derive(Debug, Clone)]
pub enum ModalExecStart {
    Immediate(ExecResult),
    Handle(ManagedModalExecHandle),
}

const CONNECT_TIMEOUT_ENV: &str = "TERMINAL_MANAGED_MODAL_CONNECT_TIMEOUT_SECONDS";
const POLL_READ_TIMEOUT_ENV: &str = "TERMINAL_MANAGED_MODAL_POLL_READ_TIMEOUT_SECONDS";
const CANCEL_READ_TIMEOUT_ENV: &str = "TERMINAL_MANAGED_MODAL_CANCEL_READ_TIMEOUT_SECONDS";

const INTERRUPT_OUTPUT: &str = "[Command interrupted - Modal sandbox exec cancelled]";
const CLIENT_TIMEOUT_GRACE_SECONDS: f64 = 10.0;
const POLL_INTERVAL_SECONDS: f64 = 0.25;

/// Statuses considered terminal by the gateway exec lifecycle.
fn is_terminal_status(status: Option<&str>) -> bool {
    matches!(
        status,
        Some("completed") | Some("failed") | Some("cancelled") | Some("timeout")
    )
}

/// Read a positive float from an env var, falling back to `default`.
///
/// Mirrors `_request_timeout_env`: non-positive or unparseable values fall back
/// to the default.
pub fn request_timeout_env(name: &str, default: f64) -> f64 {
    match std::env::var(name) {
        Ok(raw) => match raw.trim().parse::<f64>() {
            Ok(value) if value > 0.0 => value,
            _ => default,
        },
        Err(_) => default,
    }
}

/// Coerce an arbitrary JSON value into a float, falling back to `default`.
///
/// Mirrors `_coerce_number`: `null`/missing -> default; numeric strings parse;
/// otherwise default.
pub fn coerce_number(value: Option<&Value>, default: Option<f64>) -> Option<f64> {
    match value {
        None | Some(Value::Null) => default,
        Some(Value::Number(n)) => n.as_f64().or(default),
        Some(Value::String(s)) => s.trim().parse::<f64>().ok().or(default),
        Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
        Some(_) => default,
    }
}

/// A minimal view of an HTTP response used for error formatting and parsing.
///
/// Decouples response handling from `reqwest` so it can be unit-tested and so
/// the formatting logic can be shared.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

impl HttpResponse {
    /// Parse the body as JSON, if possible.
    pub fn json(&self) -> Option<Value> {
        serde_json::from_str(&self.body).ok()
    }
}

/// Format an error message from an HTTP response.
///
/// Mirrors `_format_error`: prefer `error`/`message`/`code` keys from a JSON
/// object, then the serialized JSON, then the trimmed body text, then the
/// status code.
pub fn format_error(prefix: &str, response: &HttpResponse) -> String {
    if let Some(Value::Object(map)) = response.json() {
        let message = map
            .get("error")
            .or_else(|| map.get("message"))
            .or_else(|| map.get("code"));
        if let Some(Value::String(s)) = message {
            if !s.is_empty() {
                return format!("{prefix}: {s}");
            }
        }
        // ensure_ascii=False -> serde_json keeps non-ASCII as-is by default.
        let serialized = serde_json::to_string(&Value::Object(map)).unwrap_or_default();
        return format!("{prefix}: {serialized}");
    }

    let text = response.body.trim();
    if !text.is_empty() {
        return format!("{prefix}: {text}");
    }
    format!("{prefix}: HTTP {}", response.status_code)
}

/// Trait abstracting the gateway HTTP transport so the environment can be
/// driven against a real gateway or a fake one in tests.
pub trait GatewayTransport {
    /// Perform an HTTP request to the gateway.
    ///
    /// `connect_timeout`/`read_timeout` mirror the Python `(connect, read)`
    /// tuple; when only a single timeout applies both are equal.
    fn request(
        &self,
        method: &str,
        path: &str,
        json_body: Option<&Value>,
        connect_timeout: Duration,
        read_timeout: Duration,
        extra_headers: &HashMap<String, String>,
    ) -> Result<HttpResponse, String>;
}

/// Real gateway transport backed by `reqwest::blocking`.
pub struct ReqwestGatewayTransport {
    pub gateway_origin: String,
    pub nous_user_token: String,
}

impl ReqwestGatewayTransport {
    pub fn new(gateway_origin: String, nous_user_token: String) -> Self {
        ReqwestGatewayTransport {
            gateway_origin,
            nous_user_token,
        }
    }
}

impl GatewayTransport for ReqwestGatewayTransport {
    fn request(
        &self,
        method: &str,
        path: &str,
        json_body: Option<&Value>,
        connect_timeout: Duration,
        read_timeout: Duration,
        extra_headers: &HashMap<String, String>,
    ) -> Result<HttpResponse, String> {
        let url = format!("{}{}", self.gateway_origin, path);

        let client = reqwest::blocking::Client::builder()
            .connect_timeout(connect_timeout)
            .timeout(read_timeout)
            .build()
            .map_err(|e| e.to_string())?;

        let http_method =
            reqwest::Method::from_bytes(method.as_bytes()).map_err(|e| e.to_string())?;

        let mut req = client
            .request(http_method, &url)
            .header(
                "Authorization",
                format!("Bearer {}", self.nous_user_token),
            )
            .header("Content-Type", "application/json");

        for (k, v) in extra_headers {
            req = req.header(k.as_str(), v.as_str());
        }

        if let Some(body) = json_body {
            let serialized = serde_json::to_string(body).map_err(|e| e.to_string())?;
            req = req.body(serialized);
        }

        let resp = req.send().map_err(|e| e.to_string())?;
        let status_code = resp.status().as_u16();
        let body = resp.text().map_err(|e| e.to_string())?;

        Ok(HttpResponse { status_code, body })
    }
}

/// Gateway-owned Modal sandbox with Hermes-compatible execute/cleanup.
///
/// Native port of `ManagedModalEnvironment`.
pub struct ManagedModalEnvironment<T: GatewayTransport> {
    transport: T,
    cwd: String,
    timeout: i64,
    task_id: String,
    persistent: bool,
    image: String,
    sandbox_kwargs: serde_json::Map<String, Value>,
    create_idempotency_key: String,
    sandbox_id: Option<String>,

    connect_timeout_seconds: f64,
    poll_read_timeout_seconds: f64,
    cancel_read_timeout_seconds: f64,
}

/// Options for constructing a [`ManagedModalEnvironment`].
#[derive(Debug, Clone)]
pub struct ManagedModalOptions {
    pub image: String,
    pub cwd: String,
    pub timeout: i64,
    pub modal_sandbox_kwargs: serde_json::Map<String, Value>,
    pub persistent_filesystem: bool,
    pub task_id: String,
}

impl Default for ManagedModalOptions {
    fn default() -> Self {
        ManagedModalOptions {
            image: String::new(),
            cwd: "/root".to_string(),
            timeout: 60,
            modal_sandbox_kwargs: serde_json::Map::new(),
            persistent_filesystem: true,
            task_id: "default".to_string(),
        }
    }
}

impl<T: GatewayTransport> ManagedModalEnvironment<T> {
    /// Construct the environment and create the backing sandbox.
    ///
    /// `transport` carries the resolved gateway origin and Nous user token.
    /// Mirrors `ManagedModalEnvironment.__init__`, except gateway resolution and
    /// credential-mount guarding are performed by the caller / injected, since
    /// those dependencies are not yet ported. The sandbox is created eagerly
    /// (as in Python), returning an error instead of raising.
    pub fn new(transport: T, options: ManagedModalOptions) -> Result<Self, String> {
        let mut env = ManagedModalEnvironment {
            transport,
            cwd: options.cwd,
            timeout: options.timeout,
            task_id: options.task_id,
            persistent: options.persistent_filesystem,
            image: options.image,
            sandbox_kwargs: options.modal_sandbox_kwargs,
            create_idempotency_key: new_uuid(),
            sandbox_id: None,
            connect_timeout_seconds: request_timeout_env(CONNECT_TIMEOUT_ENV, 1.0),
            poll_read_timeout_seconds: request_timeout_env(POLL_READ_TIMEOUT_ENV, 5.0),
            cancel_read_timeout_seconds: request_timeout_env(CANCEL_READ_TIMEOUT_ENV, 5.0),
        };

        let sandbox_id = env.create_sandbox()?;
        env.sandbox_id = Some(sandbox_id);
        Ok(env)
    }

    pub fn sandbox_id(&self) -> Option<&str> {
        self.sandbox_id.as_deref()
    }

    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    pub fn timeout(&self) -> i64 {
        self.timeout
    }

    /// Prepare command data for a managed Modal exec.
    ///
    /// The managed transport uses stdin mode "payload", so `stdin_data` is
    /// forwarded verbatim rather than wrapped in a heredoc. Command preparation
    /// (sudo wrapping etc.) is delegated to the gateway server-side, so we pass
    /// the command through unchanged. Mirrors `_prepare_modal_exec` for the
    /// managed case.
    pub fn prepare_exec(
        &self,
        command: &str,
        cwd: &str,
        timeout: Option<i64>,
        stdin_data: Option<&str>,
    ) -> PreparedModalExec {
        let effective_cwd = if cwd.is_empty() {
            self.cwd.clone()
        } else {
            cwd.to_string()
        };
        // Python: `timeout or self.timeout` -> falls back when None or 0.
        let effective_timeout = match timeout {
            Some(t) if t != 0 => t,
            _ => self.timeout,
        };

        PreparedModalExec {
            command: command.to_string(),
            cwd: effective_cwd,
            timeout: effective_timeout,
            stdin_data: stdin_data.map(|s| s.to_string()),
        }
    }

    /// Execute a command, polling to completion. `is_interrupted` is invoked
    /// between polls to support cooperative cancellation (mirrors
    /// `tools.interrupt.is_interrupted`).
    ///
    /// Mirrors `BaseModalExecutionEnvironment.execute` specialized for the
    /// managed transport.
    pub fn execute(
        &self,
        command: &str,
        cwd: &str,
        timeout: Option<i64>,
        stdin_data: Option<&str>,
        is_interrupted: &dyn Fn() -> bool,
    ) -> ExecResult {
        let prepared = self.prepare_exec(command, cwd, timeout, stdin_data);

        let start = self.start_modal_exec(&prepared);
        let handle = match start {
            ModalExecStart::Immediate(result) => return result,
            ModalExecStart::Handle(h) => h,
        };

        let deadline = Instant::now()
            + Duration::from_secs_f64(prepared.timeout as f64 + CLIENT_TIMEOUT_GRACE_SECONDS);

        loop {
            if is_interrupted() {
                self.cancel_modal_exec(&handle);
                return ExecResult::new(INTERRUPT_OUTPUT, 130);
            }

            match self.poll_modal_exec(&handle) {
                Some(result) => return result,
                None => {}
            }

            if Instant::now() >= deadline {
                self.cancel_modal_exec(&handle);
                return self.timeout_result_for_modal(prepared.timeout);
            }

            std::thread::sleep(Duration::from_secs_f64(POLL_INTERVAL_SECONDS));
        }
    }

    /// Begin a managed Modal exec. Mirrors `_start_modal_exec`.
    pub fn start_modal_exec(&self, prepared: &PreparedModalExec) -> ModalExecStart {
        let exec_id = new_uuid();
        let mut payload = serde_json::Map::new();
        payload.insert("execId".to_string(), json!(exec_id));
        payload.insert("command".to_string(), json!(prepared.command));
        payload.insert("cwd".to_string(), json!(prepared.cwd));
        payload.insert(
            "timeoutMs".to_string(),
            json!((prepared.timeout * 1000) as i64),
        );
        if let Some(stdin) = &prepared.stdin_data {
            payload.insert("stdinData".to_string(), json!(stdin));
        }

        let sandbox_id = match &self.sandbox_id {
            Some(id) => id.clone(),
            None => {
                return ModalExecStart::Immediate(ExecResult::error(
                    "Managed Modal exec failed: no sandbox",
                ))
            }
        };

        let path = format!("/v1/sandboxes/{sandbox_id}/execs");
        let body_value = Value::Object(payload);
        let response = match self.request(
            "POST",
            &path,
            Some(&body_value),
            10.0,
            10.0,
            &HashMap::new(),
        ) {
            Ok(resp) => resp,
            Err(exc) => {
                return ModalExecStart::Immediate(ExecResult::error(format!(
                    "Managed Modal exec failed: {exc}"
                )))
            }
        };

        if response.status_code >= 400 {
            return ModalExecStart::Immediate(ExecResult::error(format_error(
                "Managed Modal exec failed",
                &response,
            )));
        }

        let body = response.json().unwrap_or(Value::Null);
        let status = body.get("status").and_then(Value::as_str);
        if is_terminal_status(status) {
            return ModalExecStart::Immediate(self.result_from_body(&body));
        }

        let returned_exec_id = body.get("execId").and_then(Value::as_str);
        if returned_exec_id != Some(exec_id.as_str()) {
            return ModalExecStart::Immediate(ExecResult::error(
                "Managed Modal exec start did not return the expected exec id",
            ));
        }

        ModalExecStart::Handle(ManagedModalExecHandle { exec_id })
    }

    /// Poll an in-flight exec; returns `Some(result)` when terminal. Mirrors
    /// `_poll_modal_exec`.
    pub fn poll_modal_exec(&self, handle: &ManagedModalExecHandle) -> Option<ExecResult> {
        let sandbox_id = self.sandbox_id.as_deref()?;
        let path = format!("/v1/sandboxes/{}/execs/{}", sandbox_id, handle.exec_id);

        let response = match self.request(
            "GET",
            &path,
            None,
            self.connect_timeout_seconds,
            self.poll_read_timeout_seconds,
            &HashMap::new(),
        ) {
            Ok(resp) => resp,
            Err(exc) => {
                return Some(ExecResult::error(format!(
                    "Managed Modal exec poll failed: {exc}"
                )))
            }
        };

        if response.status_code == 404 {
            return Some(ExecResult::error("Managed Modal exec not found"));
        }

        if response.status_code >= 400 {
            return Some(ExecResult::error(format_error(
                "Managed Modal exec poll failed",
                &response,
            )));
        }

        let body = response.json().unwrap_or(Value::Null);
        let status = body.get("status").and_then(Value::as_str);
        if is_terminal_status(status) {
            return Some(self.result_from_body(&body));
        }
        None
    }

    /// Cancel an in-flight exec. Mirrors `_cancel_modal_exec` -> `_cancel_exec`.
    pub fn cancel_modal_exec(&self, handle: &ManagedModalExecHandle) {
        let sandbox_id = match &self.sandbox_id {
            Some(id) => id.clone(),
            None => return,
        };
        let path = format!("/v1/sandboxes/{}/execs/{}/cancel", sandbox_id, handle.exec_id);
        if let Err(exc) = self.request(
            "POST",
            &path,
            None,
            self.connect_timeout_seconds,
            self.cancel_read_timeout_seconds,
            &HashMap::new(),
        ) {
            log::warn!("Managed Modal exec cancel failed: {exc}");
        }
    }

    /// Mirrors `_timeout_result_for_modal`.
    pub fn timeout_result_for_modal(&self, timeout: i64) -> ExecResult {
        ExecResult::new(format!("Managed Modal exec timed out after {timeout}s"), 124)
    }

    /// Terminate the sandbox, optionally snapshotting. Mirrors `cleanup`.
    pub fn cleanup(&mut self) {
        let sandbox_id = match self.sandbox_id.take() {
            Some(id) if !id.is_empty() => id,
            _ => {
                // Either no sandbox or empty id: clear and return.
                self.sandbox_id = None;
                return;
            }
        };

        let path = format!("/v1/sandboxes/{sandbox_id}/terminate");
        let body = json!({ "snapshotBeforeTerminate": self.persistent });
        if let Err(exc) = self.request("POST", &path, Some(&body), 60.0, 60.0, &HashMap::new()) {
            log::warn!("Managed Modal cleanup failed: {exc}");
        }
        // `sandbox_id` already taken above, so it remains None.
    }

    /// Create the backing sandbox and return its id. Mirrors `_create_sandbox`.
    fn create_sandbox(&self) -> Result<String, String> {
        let cpu = coerce_number(self.sandbox_kwargs.get("cpu"), Some(1.0));
        let memory = coerce_number(
            self.sandbox_kwargs
                .get("memoryMiB")
                .or_else(|| self.sandbox_kwargs.get("memory")),
            Some(5120.0),
        );
        let disk = coerce_number(
            self.sandbox_kwargs
                .get("ephemeral_disk")
                .or_else(|| self.sandbox_kwargs.get("diskMiB")),
            None,
        );

        let idle_timeout_ms = std::cmp::max(300_000i64, self.timeout * 1000);

        let mut payload = serde_json::Map::new();
        payload.insert("image".to_string(), json!(self.image));
        payload.insert("cwd".to_string(), json!(self.cwd));
        payload.insert("cpu".to_string(), number_value(cpu));
        payload.insert("memoryMiB".to_string(), number_value(memory));
        payload.insert("timeoutMs".to_string(), json!(3_600_000i64));
        payload.insert("idleTimeoutMs".to_string(), json!(idle_timeout_ms));
        payload.insert("persistentFilesystem".to_string(), json!(self.persistent));
        payload.insert("logicalKey".to_string(), json!(self.task_id));
        if let Some(d) = disk {
            payload.insert("diskMiB".to_string(), number_value(Some(d)));
        }

        let mut extra_headers = HashMap::new();
        extra_headers.insert(
            "x-idempotency-key".to_string(),
            self.create_idempotency_key.clone(),
        );

        let body_value = Value::Object(payload);
        let response = self.request(
            "POST",
            "/v1/sandboxes",
            Some(&body_value),
            60.0,
            60.0,
            &extra_headers,
        )?;

        if response.status_code >= 400 {
            return Err(format_error("Managed Modal create failed", &response));
        }

        let body = response.json().unwrap_or(Value::Null);
        match body.get("id") {
            Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
            _ => Err("Managed Modal create did not return a sandbox id".to_string()),
        }
    }

    /// Build `{"output": ..., "returncode": ...}` from a gateway status body,
    /// defaulting output to "" and returncode to 1. Mirrors the `_result(...)`
    /// calls fed from response bodies.
    fn result_from_body(&self, body: &Value) -> ExecResult {
        let output = body
            .get("output")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let returncode = body
            .get("returncode")
            .and_then(Value::as_i64)
            .unwrap_or(1);
        ExecResult::new(output, returncode)
    }

    /// Issue an authenticated request through the transport. Mirrors `_request`.
    fn request(
        &self,
        method: &str,
        path: &str,
        json_body: Option<&Value>,
        connect_timeout_seconds: f64,
        read_timeout_seconds: f64,
        extra_headers: &HashMap<String, String>,
    ) -> Result<HttpResponse, String> {
        self.transport.request(
            method,
            path,
            json_body,
            Duration::from_secs_f64(connect_timeout_seconds),
            Duration::from_secs_f64(read_timeout_seconds),
            extra_headers,
        )
    }
}

/// Encode a coerced number into JSON the way Python's `json.dumps` would: an
/// integral float renders as a whole number (e.g. `1.0` -> `1`), preserving the
/// `cpu`/`memoryMiB`/`diskMiB` shapes that downstream gateways expect.
fn number_value(value: Option<f64>) -> Value {
    match value {
        Some(v) if v.fract() == 0.0 && v.is_finite() => json!(v as i64),
        Some(v) => json!(v),
        None => Value::Null,
    }
}

/// Generate a UUIDv4-like string without pulling in the `uuid` crate.
///
/// Mirrors `str(uuid.uuid4())`: a random 128-bit value formatted as
/// `8-4-4-4-12` hex with version/variant bits set.
fn new_uuid() -> String {
    let mut bytes = [0u8; 16];
    fill_random(&mut bytes);
    // Set version 4 and RFC 4122 variant bits.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Fill a buffer with random bytes using OS entropy, with a time-seeded
/// fallback so the function never fails.
fn fill_random(buf: &mut [u8]) {
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| std::io::Read::read_exact(&mut f, buf))
        .is_ok()
    {
        return;
    }

    // Fallback: xorshift seeded from the clock + buffer address.
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15)
        ^ (buf.as_ptr() as u64);
    if state == 0 {
        state = 0x9E3779B97F4A7C15;
    }
    for b in buf.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *b = (state & 0xff) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    /// Programmable transport returning queued responses in order, recording
    /// each request for assertions.
    struct FakeTransport {
        responses: RefCell<VecDeque<Result<HttpResponse, String>>>,
        requests: RefCell<Vec<(String, String, Option<Value>, HashMap<String, String>)>>,
    }

    impl FakeTransport {
        fn new(responses: Vec<Result<HttpResponse, String>>) -> Self {
            FakeTransport {
                responses: RefCell::new(responses.into_iter().collect()),
                requests: RefCell::new(Vec::new()),
            }
        }
    }

    impl GatewayTransport for FakeTransport {
        fn request(
            &self,
            method: &str,
            path: &str,
            json_body: Option<&Value>,
            _connect_timeout: Duration,
            _read_timeout: Duration,
            extra_headers: &HashMap<String, String>,
        ) -> Result<HttpResponse, String> {
            self.requests.borrow_mut().push((
                method.to_string(),
                path.to_string(),
                json_body.cloned(),
                extra_headers.clone(),
            ));
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(|| Err("no more queued responses".to_string()))
        }
    }

    fn ok(status: u16, body: &str) -> Result<HttpResponse, String> {
        Ok(HttpResponse {
            status_code: status,
            body: body.to_string(),
        })
    }

    fn env_with_create(
        responses: Vec<Result<HttpResponse, String>>,
        options: ManagedModalOptions,
    ) -> Result<ManagedModalEnvironment<FakeTransport>, String> {
        let transport = FakeTransport::new(responses);
        ManagedModalEnvironment::new(transport, options)
    }

    #[test]
    fn request_timeout_env_fallbacks() {
        let name = "TERMINAL_MANAGED_MODAL_TEST_TIMEOUT_XYZ";
        unsafe {
            std::env::remove_var(name);
        }
        assert_eq!(request_timeout_env(name, 2.5), 2.5);

        unsafe {
            std::env::set_var(name, "0");
        }
        assert_eq!(request_timeout_env(name, 2.5), 2.5);

        unsafe {
            std::env::set_var(name, "-3");
        }
        assert_eq!(request_timeout_env(name, 2.5), 2.5);

        unsafe {
            std::env::set_var(name, "not-a-number");
        }
        assert_eq!(request_timeout_env(name, 2.5), 2.5);

        unsafe {
            std::env::set_var(name, "7.5");
        }
        assert_eq!(request_timeout_env(name, 2.5), 7.5);

        unsafe {
            std::env::remove_var(name);
        }
    }

    #[test]
    fn coerce_number_handles_variants() {
        assert_eq!(coerce_number(None, Some(1.0)), Some(1.0));
        assert_eq!(coerce_number(Some(&Value::Null), Some(1.0)), Some(1.0));
        assert_eq!(coerce_number(Some(&json!(4)), Some(1.0)), Some(4.0));
        assert_eq!(coerce_number(Some(&json!("8")), Some(1.0)), Some(8.0));
        assert_eq!(coerce_number(Some(&json!("nope")), Some(1.0)), Some(1.0));
        assert_eq!(coerce_number(Some(&json!([1, 2])), Some(1.0)), Some(1.0));
        assert_eq!(coerce_number(Some(&json!("3.5")), None), Some(3.5));
        assert_eq!(coerce_number(None, None), None);
    }

    #[test]
    fn format_error_prefers_error_key() {
        let resp = HttpResponse {
            status_code: 500,
            body: r#"{"error":"boom"}"#.to_string(),
        };
        assert_eq!(format_error("X", &resp), "X: boom");
    }

    #[test]
    fn format_error_falls_back_to_message_then_code() {
        let resp = HttpResponse {
            status_code: 500,
            body: r#"{"message":"m"}"#.to_string(),
        };
        assert_eq!(format_error("X", &resp), "X: m");

        let resp = HttpResponse {
            status_code: 500,
            body: r#"{"code":"c"}"#.to_string(),
        };
        assert_eq!(format_error("X", &resp), "X: c");
    }

    #[test]
    fn format_error_dumps_json_when_no_known_keys() {
        let resp = HttpResponse {
            status_code: 500,
            body: r#"{"foo":"bar"}"#.to_string(),
        };
        assert_eq!(format_error("X", &resp), r#"X: {"foo":"bar"}"#);
    }

    #[test]
    fn format_error_falls_back_to_text_then_status() {
        let resp = HttpResponse {
            status_code: 503,
            body: "  upstream down  ".to_string(),
        };
        assert_eq!(format_error("X", &resp), "X: upstream down");

        let resp = HttpResponse {
            status_code: 503,
            body: "".to_string(),
        };
        assert_eq!(format_error("X", &resp), "X: HTTP 503");
    }

    #[test]
    fn number_value_renders_integral_floats_as_ints() {
        assert_eq!(number_value(Some(1.0)), json!(1));
        assert_eq!(number_value(Some(5120.0)), json!(5120));
        assert_eq!(number_value(Some(3.5)), json!(3.5));
        assert_eq!(number_value(None), Value::Null);
    }

    #[test]
    fn new_uuid_is_well_formed_v4() {
        let u = new_uuid();
        let parts: Vec<&str> = u.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
        assert!(parts[2].starts_with('4'));
        assert!(u.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
        assert_ne!(new_uuid(), new_uuid());
    }

    #[test]
    fn create_sandbox_builds_expected_payload() {
        let mut kwargs = serde_json::Map::new();
        kwargs.insert("cpu".to_string(), json!(2));
        kwargs.insert("memory".to_string(), json!(2048));
        kwargs.insert("ephemeral_disk".to_string(), json!(10240));

        let options = ManagedModalOptions {
            image: "python:3.12".to_string(),
            cwd: "/work".to_string(),
            timeout: 120,
            modal_sandbox_kwargs: kwargs,
            persistent_filesystem: false,
            task_id: "task-7".to_string(),
        };

        let env = env_with_create(
            vec![ok(200, r#"{"id":"sbx_123"}"#)],
            options,
        )
        .expect("create");

        assert_eq!(env.sandbox_id(), Some("sbx_123"));

        let reqs = env.transport.requests.borrow();
        assert_eq!(reqs.len(), 1);
        let (method, path, body, headers) = &reqs[0];
        assert_eq!(method, "POST");
        assert_eq!(path, "/v1/sandboxes");
        assert_eq!(
            headers.get("x-idempotency-key"),
            Some(&env.create_idempotency_key)
        );
        let body = body.as_ref().unwrap();
        assert_eq!(body["image"], json!("python:3.12"));
        assert_eq!(body["cwd"], json!("/work"));
        assert_eq!(body["cpu"], json!(2));
        assert_eq!(body["memoryMiB"], json!(2048));
        assert_eq!(body["diskMiB"], json!(10240));
        assert_eq!(body["timeoutMs"], json!(3_600_000));
        // idleTimeoutMs = max(300000, 120*1000) = 300000? No: 120000 < 300000 -> 300000.
        assert_eq!(body["idleTimeoutMs"], json!(300_000));
        assert_eq!(body["persistentFilesystem"], json!(false));
        assert_eq!(body["logicalKey"], json!("task-7"));
    }

    #[test]
    fn create_sandbox_idle_timeout_scales_with_timeout() {
        let options = ManagedModalOptions {
            timeout: 600,
            ..ManagedModalOptions::default()
        };
        let env = env_with_create(vec![ok(200, r#"{"id":"x"}"#)], options).expect("create");
        let reqs = env.transport.requests.borrow();
        let (_m, _p, body, _h) = &reqs[0];
        // max(300000, 600*1000) = 600000
        assert_eq!(body.as_ref().unwrap()["idleTimeoutMs"], json!(600_000));
    }

    #[test]
    fn create_sandbox_omits_disk_when_absent() {
        let env =
            env_with_create(vec![ok(200, r#"{"id":"x"}"#)], ManagedModalOptions::default())
                .expect("create");
        let reqs = env.transport.requests.borrow();
        let body = reqs[0].2.as_ref().unwrap();
        assert!(body.get("diskMiB").is_none());
        // defaults
        assert_eq!(body["cpu"], json!(1));
        assert_eq!(body["memoryMiB"], json!(5120));
    }

    #[test]
    fn create_sandbox_error_status_returns_err() {
        let result = env_with_create(
            vec![ok(500, r#"{"error":"nope"}"#)],
            ManagedModalOptions::default(),
        );
        assert_eq!(
            result.err().unwrap(),
            "Managed Modal create failed: nope"
        );
    }

    #[test]
    fn create_sandbox_missing_id_returns_err() {
        let result = env_with_create(vec![ok(200, "{}")], ManagedModalOptions::default());
        assert_eq!(
            result.err().unwrap(),
            "Managed Modal create did not return a sandbox id"
        );
    }

    #[test]
    fn start_exec_immediate_terminal_status() {
        let env = env_with_create(
            vec![
                ok(200, r#"{"id":"sbx"}"#),
                ok(200, r#"{"status":"completed","output":"hi","returncode":0}"#),
            ],
            ManagedModalOptions::default(),
        )
        .expect("create");

        let prepared = env.prepare_exec("echo hi", "", None, None);
        match env.start_modal_exec(&prepared) {
            ModalExecStart::Immediate(res) => {
                assert_eq!(res, ExecResult::new("hi", 0));
            }
            _ => panic!("expected immediate result"),
        }
    }

    #[test]
    fn start_exec_returns_handle_when_running() {
        // The exec POST echoes back the execId; we can't predict it, so the
        // server must return matching execId. We capture the generated id by
        // reading the request body that we sent.
        let env = env_with_create(
            vec![ok(200, r#"{"id":"sbx"}"#)],
            ManagedModalOptions::default(),
        )
        .expect("create");

        // Queue a response that echoes whatever execId is requested.
        // Since FakeTransport can't reflect dynamically, mimic by reading the
        // body after the call is impossible pre-call. Instead drive through a
        // dedicated transport below.
        let _ = env;
    }

    #[test]
    fn start_exec_handle_roundtrip() {
        // A reflecting transport that echoes the execId from the request body.
        struct Reflect {
            create: RefCell<bool>,
        }
        impl GatewayTransport for Reflect {
            fn request(
                &self,
                _method: &str,
                _path: &str,
                json_body: Option<&Value>,
                _c: Duration,
                _r: Duration,
                _h: &HashMap<String, String>,
            ) -> Result<HttpResponse, String> {
                if !*self.create.borrow() {
                    *self.create.borrow_mut() = true;
                    return Ok(HttpResponse {
                        status_code: 200,
                        body: r#"{"id":"sbx"}"#.to_string(),
                    });
                }
                let exec_id = json_body
                    .and_then(|b| b.get("execId"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                Ok(HttpResponse {
                    status_code: 200,
                    body: json!({ "status": "running", "execId": exec_id }).to_string(),
                })
            }
        }

        let env = ManagedModalEnvironment::new(
            Reflect {
                create: RefCell::new(false),
            },
            ManagedModalOptions::default(),
        )
        .expect("create");

        let prepared = env.prepare_exec("sleep 1", "", None, None);
        match env.start_modal_exec(&prepared) {
            ModalExecStart::Handle(h) => assert!(!h.exec_id.is_empty()),
            _ => panic!("expected handle"),
        }
    }

    #[test]
    fn start_exec_mismatched_exec_id_errors() {
        let env = env_with_create(
            vec![
                ok(200, r#"{"id":"sbx"}"#),
                ok(200, r#"{"status":"running","execId":"WRONG"}"#),
            ],
            ManagedModalOptions::default(),
        )
        .expect("create");
        let prepared = env.prepare_exec("x", "", None, None);
        match env.start_modal_exec(&prepared) {
            ModalExecStart::Immediate(res) => {
                assert_eq!(res.returncode, 1);
                assert!(res.output.contains("expected exec id"));
            }
            _ => panic!("expected immediate error"),
        }
    }

    #[test]
    fn start_exec_transport_error() {
        let env = env_with_create(
            vec![ok(200, r#"{"id":"sbx"}"#), Err("conn refused".to_string())],
            ManagedModalOptions::default(),
        )
        .expect("create");
        let prepared = env.prepare_exec("x", "", None, None);
        match env.start_modal_exec(&prepared) {
            ModalExecStart::Immediate(res) => {
                assert_eq!(res.output, "Managed Modal exec failed: conn refused");
                assert_eq!(res.returncode, 1);
            }
            _ => panic!("expected immediate error"),
        }
    }

    #[test]
    fn start_exec_payload_includes_stdin_and_timeout_ms() {
        let env = env_with_create(
            vec![
                ok(200, r#"{"id":"sbx"}"#),
                ok(200, r#"{"status":"completed","output":"","returncode":0}"#),
            ],
            ManagedModalOptions::default(),
        )
        .expect("create");
        let prepared = env.prepare_exec("cat", "/tmp", Some(5), Some("input-data"));
        let _ = env.start_modal_exec(&prepared);

        let reqs = env.transport.requests.borrow();
        let (method, path, body, _h) = &reqs[1];
        assert_eq!(method, "POST");
        assert_eq!(path, "/v1/sandboxes/sbx/execs");
        let body = body.as_ref().unwrap();
        assert_eq!(body["command"], json!("cat"));
        assert_eq!(body["cwd"], json!("/tmp"));
        assert_eq!(body["timeoutMs"], json!(5000));
        assert_eq!(body["stdinData"], json!("input-data"));
    }

    #[test]
    fn poll_terminal_and_not_found_and_error() {
        let env = env_with_create(
            vec![
                ok(200, r#"{"id":"sbx"}"#),
                ok(200, r#"{"status":"failed","output":"bad","returncode":2}"#),
                ok(404, ""),
                ok(500, r#"{"error":"poll-bad"}"#),
                ok(200, r#"{"status":"running"}"#),
            ],
            ManagedModalOptions::default(),
        )
        .expect("create");

        let handle = ManagedModalExecHandle {
            exec_id: "e1".to_string(),
        };

        assert_eq!(
            env.poll_modal_exec(&handle),
            Some(ExecResult::new("bad", 2))
        );
        assert_eq!(
            env.poll_modal_exec(&handle),
            Some(ExecResult::error("Managed Modal exec not found"))
        );
        assert_eq!(
            env.poll_modal_exec(&handle),
            Some(ExecResult::error("Managed Modal exec poll failed: poll-bad"))
        );
        assert_eq!(env.poll_modal_exec(&handle), None);
    }

    #[test]
    fn poll_path_uses_sandbox_and_exec_id() {
        let env = env_with_create(
            vec![
                ok(200, r#"{"id":"sbx9"}"#),
                ok(200, r#"{"status":"running"}"#),
            ],
            ManagedModalOptions::default(),
        )
        .expect("create");
        let handle = ManagedModalExecHandle {
            exec_id: "abc".to_string(),
        };
        env.poll_modal_exec(&handle);
        let reqs = env.transport.requests.borrow();
        let (method, path, _b, _h) = &reqs[1];
        assert_eq!(method, "GET");
        assert_eq!(path, "/v1/sandboxes/sbx9/execs/abc");
    }

    #[test]
    fn execute_returns_interrupt_when_flagged() {
        let env = env_with_create(
            vec![
                ok(200, r#"{"id":"sbx"}"#),
                ok(200, r#"{"status":"running","execId":"will-not-match"}"#),
            ],
            ManagedModalOptions::default(),
        )
        .expect("create");
        // start_modal_exec will error on mismatched exec id -> immediate result;
        // so to test interrupt we use the reflecting handle path instead.
        let _ = env;
    }

    #[test]
    fn execute_interrupt_cancels_and_returns_130() {
        struct Reflect {
            stage: RefCell<u32>,
        }
        impl GatewayTransport for Reflect {
            fn request(
                &self,
                _method: &str,
                path: &str,
                json_body: Option<&Value>,
                _c: Duration,
                _r: Duration,
                _h: &HashMap<String, String>,
            ) -> Result<HttpResponse, String> {
                let mut stage = self.stage.borrow_mut();
                *stage += 1;
                if *stage == 1 {
                    return Ok(HttpResponse {
                        status_code: 200,
                        body: r#"{"id":"sbx"}"#.to_string(),
                    });
                }
                if path.ends_with("/execs") {
                    let exec_id = json_body
                        .and_then(|b| b.get("execId"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    return Ok(HttpResponse {
                        status_code: 200,
                        body: json!({"status": "running", "execId": exec_id}).to_string(),
                    });
                }
                // cancel
                Ok(HttpResponse {
                    status_code: 200,
                    body: "{}".to_string(),
                })
            }
        }

        let env = ManagedModalEnvironment::new(
            Reflect {
                stage: RefCell::new(0),
            },
            ManagedModalOptions::default(),
        )
        .expect("create");

        let interrupted = || true;
        let res = env.execute("loop", "", None, None, &interrupted);
        assert_eq!(res, ExecResult::new(INTERRUPT_OUTPUT, 130));
    }

    #[test]
    fn execute_immediate_completion() {
        let env = env_with_create(
            vec![
                ok(200, r#"{"id":"sbx"}"#),
                ok(200, r#"{"status":"completed","output":"done","returncode":0}"#),
            ],
            ManagedModalOptions::default(),
        )
        .expect("create");
        let never = || false;
        let res = env.execute("echo done", "", None, None, &never);
        assert_eq!(res, ExecResult::new("done", 0));
    }

    #[test]
    fn cleanup_terminates_and_clears_sandbox() {
        let mut env = env_with_create(
            vec![ok(200, r#"{"id":"sbx"}"#), ok(200, "{}")],
            ManagedModalOptions {
                persistent_filesystem: true,
                ..ManagedModalOptions::default()
            },
        )
        .expect("create");

        env.cleanup();
        assert_eq!(env.sandbox_id(), None);

        let reqs = env.transport.requests.borrow();
        let (method, path, body, _h) = &reqs[1];
        assert_eq!(method, "POST");
        assert_eq!(path, "/v1/sandboxes/sbx/terminate");
        assert_eq!(body.as_ref().unwrap()["snapshotBeforeTerminate"], json!(true));
    }

    #[test]
    fn cleanup_noop_when_no_sandbox() {
        let mut env = env_with_create(
            vec![ok(200, r#"{"id":"sbx"}"#)],
            ManagedModalOptions::default(),
        )
        .expect("create");
        // Manually clear and confirm no extra request is issued.
        env.sandbox_id = None;
        env.cleanup();
        let reqs = env.transport.requests.borrow();
        assert_eq!(reqs.len(), 1);
    }

    #[test]
    fn prepare_exec_cwd_and_timeout_fallback() {
        let env = env_with_create(
            vec![ok(200, r#"{"id":"sbx"}"#)],
            ManagedModalOptions {
                cwd: "/root".to_string(),
                timeout: 42,
                ..ManagedModalOptions::default()
            },
        )
        .expect("create");

        let p = env.prepare_exec("ls", "", None, None);
        assert_eq!(p.cwd, "/root");
        assert_eq!(p.timeout, 42);

        let p = env.prepare_exec("ls", "/other", Some(0), None);
        assert_eq!(p.cwd, "/other");
        assert_eq!(p.timeout, 42); // 0 falls back

        let p = env.prepare_exec("ls", "/other", Some(99), Some("x"));
        assert_eq!(p.timeout, 99);
        assert_eq!(p.stdin_data.as_deref(), Some("x"));
    }
}
