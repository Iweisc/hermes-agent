//! Vision / video analysis tools — native Rust port of `tools/vision_tools.py`.
//!
//! Faithfully reproduces the behaviour of the Python `vision_tools` module:
//!
//! * `vision_analyze_tool` / `video_analyze_tool` download (or read locally) an
//!   image/video, convert it to a base64 data URL, and dispatch a multimodal
//!   chat request through a pluggable LLM caller.
//! * SSRF-hardened downloading with per-redirect re-validation and retry with
//!   exponential backoff (2s, 4s, 8s).
//! * Size guards: a 50 MB download hard cap, a 20 MB base64 hard cap for images
//!   (with Pillow-style auto-resize down to a 5 MB target), and a 50 MB cap for
//!   video payloads.
//! * MIME sniffing from magic bytes (PNG/JPEG/GIF/BMP/WEBP/SVG) for images and
//!   extension-based detection for video.
//! * Error classification that maps provider failures to user-facing analysis
//!   messages (payment, unsupported-vision, rejected-image, generic).
//!
//! ## Differences from the Python original (idiomatic adaptations)
//!
//! * Python `async_call_llm` is a runtime router that this module does not own.
//!   Here the LLM call is abstracted behind the [`LlmCaller`] trait so callers
//!   (or the integration layer) can plug in the real router. A default
//!   [`UnconfiguredCaller`] returns an error mirroring "no vision client".
//! * SSRF + website-policy + interrupt checks are injected through
//!   [`SecurityHooks`]. The ported equivalents live in private modules of the
//!   sibling `hermes-core` crate (`tool_url_safety`, `tool_website_policy`,
//!   `tool_interrupt`) and are not publicly re-exported, so the integration
//!   layer wires them in. A self-contained default [`SecurityHooks::default`]
//!   provides a faithful local SSRF IP check and otherwise fails open, matching
//!   the Python "config unavailable" branch.
//! * Image auto-resize uses the `image` crate (Pillow analogue) behind the
//!   `vision_resize` feature. When unavailable the resize path degrades to
//!   returning the original encoding, exactly like Python's "Pillow not
//!   installed" branch.
//! * Synchronous (`reqwest::blocking`) downloads + a blocking retry/backoff loop
//!   replace the async httpx client.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Configurable limits (ported verbatim from the Python constants).
// ---------------------------------------------------------------------------

/// Default download timeout for `_download_image` (seconds). Overridable via
/// `HERMES_VISION_DOWNLOAD_TIMEOUT` env var or `auxiliary.vision.download_timeout`.
pub const DEFAULT_VISION_DOWNLOAD_TIMEOUT: f64 = 30.0;

/// Hard cap on a downloaded image's size (50 MB). Prevents OOM from
/// attacker-hosted multi-gigabyte files or decompression bombs.
pub const VISION_MAX_DOWNLOAD_BYTES: usize = 50 * 1024 * 1024;

/// Hard limit for vision API payloads (20 MB) — matches the most restrictive
/// major provider (Gemini inline data limit).
pub const MAX_BASE64_BYTES: usize = 20 * 1024 * 1024;

/// Target size when auto-resizing on API failure (5 MB).
pub const RESIZE_TARGET_BYTES: usize = 5 * 1024 * 1024;

/// Default vision LLM timeout (seconds). From `auxiliary.vision.timeout`.
pub const DEFAULT_VISION_TIMEOUT: f64 = 120.0;

/// Default vision LLM temperature. From `auxiliary.vision.temperature`.
pub const DEFAULT_VISION_TEMPERATURE: f64 = 0.1;

/// Video MIME hard cap (50 MB) and warn threshold (20 MB).
pub const MAX_VIDEO_BASE64_BYTES: usize = 50 * 1024 * 1024;
pub const VIDEO_SIZE_WARN_BYTES: usize = 20 * 1024 * 1024;

/// Default video LLM timeout (seconds).
pub const DEFAULT_VIDEO_TIMEOUT: f64 = 180.0;

const MAX_RETRIES: usize = 3;
const MAX_REDIRECTS: usize = 10;

const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

// ---------------------------------------------------------------------------
// LLM caller abstraction.
// ---------------------------------------------------------------------------

/// Keyword arguments for a vision/video LLM call. Mirrors the dict that the
/// Python `vision_analyze_tool` builds and passes to `async_call_llm`.
#[derive(Debug, Clone)]
pub struct LlmCallKwargs {
    pub task: String,
    pub messages: Vec<Value>,
    pub temperature: f64,
    pub max_tokens: u32,
    pub timeout: f64,
    pub model: Option<String>,
}

impl LlmCallKwargs {
    /// Serialise to the JSON kwargs shape the Python router consumes.
    pub fn to_json(&self) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("task".into(), json!(self.task));
        obj.insert("messages".into(), Value::Array(self.messages.clone()));
        obj.insert("temperature".into(), json!(self.temperature));
        obj.insert("max_tokens".into(), json!(self.max_tokens));
        obj.insert("timeout".into(), json!(self.timeout));
        if let Some(model) = &self.model {
            obj.insert("model".into(), json!(model));
        }
        Value::Object(obj)
    }
}

/// Pluggable analogue of Python's `async_call_llm`. Returns the raw provider
/// response JSON (from which [`extract_content_or_reasoning`] pulls text), or an
/// error string. The error string is matched by [`is_image_size_error`] and the
/// error-classification logic, so it should contain the provider's message.
pub trait LlmCaller {
    fn call(&self, kwargs: &LlmCallKwargs) -> Result<Value, String>;
}

/// Default caller used when the integration layer has not wired a real router.
/// Mirrors the "no auxiliary vision model available" failure mode.
pub struct UnconfiguredCaller;

impl LlmCaller for UnconfiguredCaller {
    fn call(&self, _kwargs: &LlmCallKwargs) -> Result<Value, String> {
        Err("No auxiliary vision model available. Configure a supported \
             multimodal backend (OpenRouter, Nous, Codex, Anthropic, or a \
             custom OpenAI-compatible endpoint)."
            .to_string())
    }
}

/// Extract textual content from a chat-completions style response, falling back
/// to a `reasoning` field. Mirrors `agent.auxiliary_client.extract_content_or_reasoning`.
///
/// Prefer [`hermes_core::ag_auxiliary_client::extract_content_or_reasoning`] when
/// a real response is in hand; this local copy keeps the module self-contained
/// for tests and when that crate path is unavailable.
pub fn extract_content_or_reasoning(response: &Value) -> String {
    // OpenAI-style: choices[0].message.content / .reasoning / .reasoning_content
    if let Some(choice) = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
    {
        if let Some(message) = choice.get("message") {
            if let Some(text) = message.get("content").and_then(Value::as_str) {
                if !text.trim().is_empty() {
                    return text.to_string();
                }
            }
            for key in ["reasoning", "reasoning_content"] {
                if let Some(text) = message.get(key).and_then(Value::as_str) {
                    if !text.trim().is_empty() {
                        return text.to_string();
                    }
                }
            }
        }
        // Some providers put text directly on the choice.
        if let Some(text) = choice.get("text").and_then(Value::as_str) {
            if !text.trim().is_empty() {
                return text.to_string();
            }
        }
    }
    // Anthropic-style: content is a list of blocks with type=text.
    if let Some(blocks) = response.get("content").and_then(Value::as_array) {
        let mut out = String::new();
        for block in blocks {
            if block.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    out.push_str(text);
                }
            }
        }
        if !out.trim().is_empty() {
            return out;
        }
    }
    // Flat content string.
    if let Some(text) = response.get("content").and_then(Value::as_str) {
        if !text.trim().is_empty() {
            return text.to_string();
        }
    }
    String::new()
}

// ---------------------------------------------------------------------------
// Security hooks (SSRF / website policy / interrupt).
//
// The Python original imports `tools.url_safety.is_safe_url`,
// `tools.website_policy.check_website_access`, and `tools.interrupt.is_interrupted`.
// Those are ported in the `hermes-core` crate but live in private modules; the
// integration layer can wire the real implementations in here. The defaults
// provide a faithful self-contained SSRF check and otherwise fail open.
// ---------------------------------------------------------------------------

/// Injectable security callbacks. Construct with [`SecurityHooks::default`] for
/// the built-in SSRF check, or override any field to delegate to the ported
/// `hermes-core` implementations.
#[derive(Clone)]
pub struct SecurityHooks {
    /// Returns `true` if the URL is safe to fetch (not private/internal). Maps
    /// to `tools.url_safety.is_safe_url`.
    pub is_safe_url: std::sync::Arc<dyn Fn(&str) -> bool + Send + Sync>,
    /// Returns `Some(message)` if the URL is blocked by website policy. Maps to
    /// `tools.website_policy.check_website_access` (collapsed to the message).
    pub check_website_access: std::sync::Arc<dyn Fn(&str) -> Option<String> + Send + Sync>,
    /// Returns `true` if the current operation has been interrupted. Maps to
    /// `tools.interrupt.is_interrupted`.
    pub is_interrupted: std::sync::Arc<dyn Fn() -> bool + Send + Sync>,
}

impl Default for SecurityHooks {
    fn default() -> Self {
        Self {
            is_safe_url: std::sync::Arc::new(default_is_safe_url),
            check_website_access: std::sync::Arc::new(|_url| None),
            is_interrupted: std::sync::Arc::new(|| false),
        }
    }
}

impl std::fmt::Debug for SecurityHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityHooks").finish_non_exhaustive()
    }
}

/// Self-contained SSRF check: parse the URL, resolve the host, and block any
/// private/loopback/link-local/reserved/CGNAT/metadata address. Faithful subset
/// of `tools.url_safety.is_safe_url` (fails closed on resolution failure).
pub fn default_is_safe_url(url: &str) -> bool {
    let parsed = match url::Url::parse(url) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let host = parsed
        .host_str()
        .unwrap_or("")
        .trim()
        .trim_end_matches('.')
        .to_lowercase();
    if host.is_empty() {
        return false;
    }
    // Always block cloud-metadata hostnames.
    if host == "metadata.google.internal" || host == "metadata.goog" {
        return false;
    }
    // If the host is a literal IP, classify it directly.
    if let Ok(ip) = host.parse::<IpAddr>() {
        return !is_blocked_ip(&ip);
    }
    // Resolve and block if any resolved IP is unsafe (fail closed on failure).
    match (host.as_str(), 0u16).to_socket_addrs() {
        Ok(addrs) => {
            let mut saw = false;
            for sa in addrs {
                saw = true;
                if is_blocked_ip(&sa.ip()) {
                    return false;
                }
            }
            saw
        }
        Err(_) => false,
    }
}

fn is_blocked_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            is_private_v4(v4)
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.octets()[0] >= 240 // reserved 240.0.0.0/4
                || v4.is_multicast()
                || v4.is_unspecified()
                || in_cgnat(v4)
        }
        IpAddr::V6(v6) => {
            is_private_v6(v6)
                || v6.is_loopback()
                || (v6.segments()[0] & 0xffc0) == 0xfe80 // link-local fe80::/10
                || v6.is_multicast()
                || v6.is_unspecified()
        }
    }
}

fn is_private_v4(v4: &Ipv4Addr) -> bool {
    let o = v4.octets();
    if o[0] == 10 {
        return true;
    }
    if o[0] == 172 && (16..=31).contains(&o[1]) {
        return true;
    }
    if o[0] == 192 && o[1] == 168 {
        return true;
    }
    if o[0] == 0 {
        return true;
    }
    if o[0] == 192 && o[1] == 0 && o[2] == 0 {
        return true;
    }
    if o[0] == 198 && (o[1] == 18 || o[1] == 19) {
        return true;
    }
    if o[0] == 192 && o[1] == 0 && o[2] == 2 {
        return true;
    }
    if o[0] == 198 && o[1] == 51 && o[2] == 100 {
        return true;
    }
    if o[0] == 203 && o[1] == 0 && o[2] == 113 {
        return true;
    }
    false
}

fn in_cgnat(v4: &Ipv4Addr) -> bool {
    let o = v4.octets();
    o[0] == 100 && (64..=127).contains(&o[1])
}

fn is_private_v6(v6: &Ipv6Addr) -> bool {
    if (v6.segments()[0] & 0xfe00) == 0xfc00 {
        return true;
    }
    if let Some(v4) = v6.to_ipv4_mapped() {
        return is_private_v4(&v4) || v4.is_loopback() || v4.is_link_local();
    }
    false
}

// ---------------------------------------------------------------------------
// Download-timeout resolution.
// ---------------------------------------------------------------------------

/// Resolve the image download timeout. Resolution order mirrors Python:
/// `HERMES_VISION_DOWNLOAD_TIMEOUT` env → `auxiliary.vision.download_timeout`
/// config → 30s default.
pub fn resolve_download_timeout(config: Option<&serde_yaml::Value>) -> f64 {
    if let Ok(raw) = std::env::var("HERMES_VISION_DOWNLOAD_TIMEOUT") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            if let Ok(value) = trimmed.parse::<f64>() {
                return value;
            }
        }
    }
    if let Some(value) = yaml_number(config, &["auxiliary", "vision", "download_timeout"]) {
        return value;
    }
    DEFAULT_VISION_DOWNLOAD_TIMEOUT
}

fn yaml_number(config: Option<&serde_yaml::Value>, keys: &[&str]) -> Option<f64> {
    let mut node = config?;
    for key in keys {
        node = node.get(serde_yaml::Value::String((*key).to_string()))?;
    }
    match node {
        serde_yaml::Value::Number(n) => n.as_f64(),
        serde_yaml::Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// URL validation + MIME detection.
// ---------------------------------------------------------------------------

/// Basic validation of an image/video URL. Mirrors `_validate_image_url`:
/// requires http(s) scheme, a non-empty netloc, and passes the SSRF
/// `is_safe_url` check (via the default [`SecurityHooks`]).
pub fn validate_image_url(url: &str) -> bool {
    validate_image_url_with_hooks(url, &SecurityHooks::default())
}

/// `validate_image_url` with explicit security hooks for the SSRF check.
pub fn validate_image_url_with_hooks(url: &str, hooks: &SecurityHooks) -> bool {
    if url.is_empty() {
        return false;
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return false;
    }
    let parsed = match url::Url::parse(url) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let netloc = parsed.host_str().unwrap_or("");
    if netloc.is_empty() {
        return false;
    }
    (hooks.is_safe_url)(url)
}

/// Detect an image MIME type from magic bytes + (for SVG) the file body.
/// Mirrors `_detect_image_mime_type`. Returns `None` for non-image files.
pub fn detect_image_mime_type(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    detect_image_mime_from_bytes(&bytes, path)
}

/// Magic-byte sniffing shared by file and in-memory paths.
pub fn detect_image_mime_from_bytes(header: &[u8], path: &Path) -> Option<String> {
    if header.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png".to_string());
    }
    if header.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some("image/jpeg".to_string());
    }
    if header.starts_with(b"GIF87a") || header.starts_with(b"GIF89a") {
        return Some("image/gif".to_string());
    }
    if header.starts_with(b"BM") {
        return Some("image/bmp".to_string());
    }
    if header.len() >= 12 && &header[0..4] == b"RIFF" && &header[8..12] == b"WEBP" {
        return Some("image/webp".to_string());
    }
    if ext_lower(path).as_deref() == Some("svg") {
        let head = String::from_utf8_lossy(&header[..header.len().min(4096)]).to_lowercase();
        if head.contains("<svg") {
            return Some("image/svg+xml".to_string());
        }
    }
    None
}

/// Determine a MIME type from the file extension, defaulting to `image/jpeg`.
/// Mirrors `_determine_mime_type`.
pub fn determine_mime_type(path: &Path) -> String {
    match ext_lower(path).as_deref() {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("gif") => "image/gif",
        Some("bmp") => "image/bmp",
        Some("webp") => "image/webp",
        Some("svg") => "image/svg+xml",
        _ => "image/jpeg",
    }
    .to_string()
}

fn ext_lower(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
}

// ---------------------------------------------------------------------------
// Base64 data URL encoding.
// ---------------------------------------------------------------------------

/// Convert an image file to a base64 data URL. Mirrors `_image_to_base64_data_url`.
pub fn image_to_base64_data_url(path: &Path, mime_type: Option<&str>) -> std::io::Result<String> {
    let data = std::fs::read(path)?;
    let encoded = BASE64.encode(&data);
    let mime = mime_type
        .map(ToString::to_string)
        .unwrap_or_else(|| determine_mime_type(path));
    Ok(format!("data:{mime};base64,{encoded}"))
}

/// Detect if an API error string relates to image or payload size.
/// Mirrors `_is_image_size_error`.
pub fn is_image_size_error(error: &str) -> bool {
    let lower = error.to_lowercase();
    const HINTS: [&str; 9] = [
        "too large",
        "payload",
        "413",
        "content_too_large",
        "request_too_large",
        "image_url",
        "invalid_request",
        "exceeds",
        "size limit",
    ];
    HINTS.iter().any(|h| lower.contains(h))
}

// ---------------------------------------------------------------------------
// Auto-resize (Pillow analogue via the `image` crate).
// ---------------------------------------------------------------------------

/// Convert an image to a base64 data URL, auto-resizing if too large. Mirrors
/// `_resize_image_for_vision`.
///
/// Strategy: if the estimated base64 size already fits, encode directly.
/// Otherwise decode with the `image` crate and progressively halve dimensions
/// (min 64px) for up to 5 rounds, trying JPEG quality steps (85/70/50) for JPEG
/// output. Returns the best candidate found.
pub fn resize_image_for_vision(
    path: &Path,
    mime_type: Option<&str>,
    max_base64_bytes: usize,
) -> std::io::Result<String> {
    let file_size = std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);
    let estimated_b64 = (file_size * 4) / 3 + 100;

    let mut direct: Option<String> = None;
    if estimated_b64 <= max_base64_bytes {
        let url = image_to_base64_data_url(path, mime_type)?;
        if url.len() <= max_base64_bytes {
            return Ok(url);
        }
        direct = Some(url);
    }

    match resize_with_image_crate(path, mime_type, max_base64_bytes) {
        Some(candidate) => Ok(candidate),
        None => {
            // Pillow analogue unavailable / cannot open — fall back to raw encode.
            match direct {
                Some(url) => Ok(url),
                None => image_to_base64_data_url(path, mime_type),
            }
        }
    }
}

#[cfg(feature = "vision_resize")]
fn resize_with_image_crate(
    path: &Path,
    mime_type: Option<&str>,
    max_base64_bytes: usize,
) -> Option<String> {
    use image::ImageFormat;

    let mime = mime_type
        .map(ToString::to_string)
        .unwrap_or_else(|| determine_mime_type(path));
    let to_png = mime == "image/png";
    let (out_format, out_mime) = if to_png {
        (ImageFormat::Png, "image/png")
    } else {
        (ImageFormat::Jpeg, "image/jpeg")
    };

    let loaded = image::open(path).ok()?;
    let mut img = if !to_png {
        image::DynamicImage::ImageRgb8(loaded.to_rgb8())
    } else {
        loaded
    };

    let quality_steps: &[Option<u8>] = if to_png {
        &[None]
    } else {
        &[Some(85), Some(70), Some(50)]
    };

    let mut prev_dims = (img.width(), img.height());
    let mut candidate: Option<String> = None;

    for attempt in 0..5 {
        if attempt > 0 {
            let mut new_w = ((img.width() as f64) * 0.5) as u32;
            let mut new_h = ((img.height() as f64) * 0.5) as u32;
            new_w = new_w.max(64);
            new_h = new_h.max(64);
            if new_w == 64 && img.width() > 0 {
                let eff = 64.0 / img.width() as f64;
                new_h = (((img.height() as f64) * eff) as u32).max(64);
            } else if new_h == 64 && img.height() > 0 {
                let eff = 64.0 / img.height() as f64;
                new_w = (((img.width() as f64) * eff) as u32).max(64);
            }
            if (new_w, new_h) == prev_dims {
                break;
            }
            img = img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3);
            prev_dims = (new_w, new_h);
        }

        for q in quality_steps {
            let mut buf: Vec<u8> = Vec::new();
            let mut cursor = std::io::Cursor::new(&mut buf);
            let write_ok = match (out_format, q) {
                (ImageFormat::Jpeg, Some(quality)) => {
                    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(
                        &mut cursor,
                        *quality,
                    );
                    enc.encode_image(&img).is_ok()
                }
                _ => img.write_to(&mut cursor, out_format).is_ok(),
            };
            if !write_ok {
                continue;
            }
            let encoded = BASE64.encode(&buf);
            let url = format!("data:{out_mime};base64,{encoded}");
            if url.len() <= max_base64_bytes {
                return Some(url);
            }
            candidate = Some(url);
        }
    }
    candidate
}

#[cfg(not(feature = "vision_resize"))]
fn resize_with_image_crate(
    _path: &Path,
    _mime_type: Option<&str>,
    _max_base64_bytes: usize,
) -> Option<String> {
    // Pillow-not-installed analogue: no resize available.
    None
}

// ---------------------------------------------------------------------------
// Downloading with SSRF guard + retry.
// ---------------------------------------------------------------------------

/// Re-validate a redirect target against SSRF rules + website policy.
/// Mirrors the `_ssrf_redirect_guard` event hook (returns `Err(message)` to block).
fn ssrf_redirect_guard(url: &str, hooks: &SecurityHooks) -> Result<(), String> {
    if !(hooks.is_safe_url)(url) {
        return Err(format!(
            "Blocked redirect to private/internal address: {url}"
        ));
    }
    if let Some(message) = (hooks.check_website_access)(url) {
        return Err(message);
    }
    Ok(())
}

/// Download an image to `destination` with retry + SSRF redirect re-validation.
/// Mirrors `_download_image`. `accept` is the Accept header (image vs video).
fn download_to(
    url: &str,
    destination: &Path,
    timeout_secs: f64,
    max_bytes: usize,
    accept: &str,
    max_retries: usize,
    hooks: &SecurityHooks,
) -> Result<PathBuf, String> {
    if let Some(parent) = destination.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut last_error: Option<String> = None;
    for attempt in 0..max_retries {
        match download_once(url, destination, timeout_secs, max_bytes, accept, hooks) {
            Ok(()) => return Ok(destination.to_path_buf()),
            Err(e) => {
                last_error = Some(e.clone());
                if attempt < max_retries - 1 {
                    let wait = 2u64.pow((attempt + 1) as u32);
                    log::warn!(
                        "Download failed (attempt {}/{}): {}",
                        attempt + 1,
                        max_retries,
                        &e[..e.len().min(50)]
                    );
                    log::warn!("Retrying in {}s...", wait);
                    std::thread::sleep(Duration::from_secs(wait));
                } else {
                    log::error!(
                        "Download failed after {} attempts: {}",
                        max_retries,
                        &e[..e.len().min(100)]
                    );
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        format!("download exited retry loop without attempting (max_retries={max_retries})")
    }))
}

fn download_once(
    url: &str,
    destination: &Path,
    timeout_secs: f64,
    max_bytes: usize,
    accept: &str,
    hooks: &SecurityHooks,
) -> Result<(), String> {
    // Pre-flight website policy on the initial URL.
    if let Some(message) = (hooks.check_website_access)(url) {
        return Err(message);
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout_secs.max(1.0)))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| e.to_string())?;

    let mut current = url::Url::parse(url).map_err(|e| e.to_string())?;

    for _ in 0..=MAX_REDIRECTS {
        let response = client
            .get(current.clone())
            .header(reqwest::header::USER_AGENT, USER_AGENT)
            .header(reqwest::header::ACCEPT, accept)
            .send()
            .map_err(|e| e.to_string())?;

        let status = response.status();
        if status.is_redirection() {
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .ok_or_else(|| "Redirect response was missing a Location header.".to_string())?
                .to_str()
                .map_err(|_| "Redirect Location was not valid UTF-8.".to_string())?;
            let next = current
                .join(location)
                .map_err(|e| format!("Invalid redirect target: {e}"))?;
            // SSRF redirect guard (event-hook analogue).
            ssrf_redirect_guard(next.as_str(), hooks)?;
            current = next;
            continue;
        }

        if !status.is_success() {
            return Err(format!("HTTP {} while fetching resource.", status.as_u16()));
        }

        // Reject overly large payloads early via Content-Length.
        if let Some(cl) = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            if cl > max_bytes {
                return Err(format!("Resource too large ({cl} bytes, max {max_bytes})"));
            }
        }

        // Re-validate the final (post-redirect) URL against website policy.
        let final_url = response.url().to_string();
        if let Some(message) = (hooks.check_website_access)(&final_url) {
            return Err(message);
        }

        let body = response.bytes().map_err(|e| e.to_string())?;
        if body.len() > max_bytes {
            return Err(format!(
                "Resource too large ({} bytes, max {max_bytes})",
                body.len()
            ));
        }
        std::fs::write(destination, &body).map_err(|e| e.to_string())?;
        return Ok(());
    }

    Err("Too many redirects while fetching resource.".to_string())
}

// ---------------------------------------------------------------------------
// Vision analysis.
// ---------------------------------------------------------------------------

/// Settings sourced from `auxiliary.vision` config (timeout/temperature).
fn vision_call_params(config: Option<&serde_yaml::Value>) -> (f64, f64) {
    let mut timeout = DEFAULT_VISION_TIMEOUT;
    let mut temperature = DEFAULT_VISION_TEMPERATURE;
    if let Some(v) = yaml_number(config, &["auxiliary", "vision", "timeout"]) {
        timeout = v;
    }
    if let Some(v) = yaml_number(config, &["auxiliary", "vision", "temperature"]) {
        temperature = v;
    }
    (timeout, temperature)
}

/// Strip a `file://` prefix and expand `~`, mirroring `os.path.expanduser`.
fn resolve_local_path(raw: &str) -> PathBuf {
    let resolved = raw.strip_prefix("file://").unwrap_or(raw);
    if let Some(rest) = resolved.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    } else if resolved == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    PathBuf::from(resolved)
}

/// Analyze an image from a URL or local file path. Native port of
/// `vision_analyze_tool`. Returns a JSON string `{success, analysis[, error]}`.
///
/// `config` supplies `auxiliary.vision.*` overrides; `hooks` injects SSRF /
/// website-policy / interrupt behaviour.
pub fn vision_analyze_tool(
    image_url: &str,
    user_prompt: &str,
    model: Option<&str>,
    caller: &dyn LlmCaller,
    config: Option<&serde_yaml::Value>,
    hooks: &SecurityHooks,
) -> String {
    match run_vision_analysis(image_url, user_prompt, model, caller, config, hooks) {
        Ok(analysis) => pretty_json(&json!({
            "success": true,
            "analysis": if analysis.is_empty() {
                "There was a problem with the request and the image could not be analyzed.".to_string()
            } else {
                analysis
            },
        })),
        Err(e) => pretty_json(&classify_image_error(&e, model)),
    }
}

fn run_vision_analysis(
    image_url: &str,
    user_prompt: &str,
    model: Option<&str>,
    caller: &dyn LlmCaller,
    config: Option<&serde_yaml::Value>,
    hooks: &SecurityHooks,
) -> Result<String, String> {
    if (hooks.is_interrupted)() {
        return Err("Interrupted".to_string());
    }

    // Resolve local vs remote.
    let local_path = resolve_local_path(image_url);
    let (temp_path, should_cleanup) = if local_path.is_file() {
        (local_path, false)
    } else if validate_image_url_with_hooks(image_url, hooks) {
        if let Some(message) = (hooks.check_website_access)(image_url) {
            return Err(message);
        }
        let timeout = resolve_download_timeout(config);
        let temp_dir = hermes_cache_dir("cache/vision", "temp_vision_images");
        let temp_path = temp_dir.join(format!("temp_image_{}.jpg", unique_token()));
        download_to(
            image_url,
            &temp_path,
            timeout,
            VISION_MAX_DOWNLOAD_BYTES,
            "image/*,*/*;q=0.8",
            MAX_RETRIES,
            hooks,
        )?;
        (temp_path, true)
    } else {
        return Err(
            "Invalid image source. Provide an HTTP/HTTPS URL or a valid local file path."
                .to_string(),
        );
    };

    let result = (|| -> Result<String, String> {
        let detected_mime = detect_image_mime_type(&temp_path)
            .ok_or_else(|| "Only real image files are supported for vision analysis.".to_string())?;

        let mut image_data_url =
            image_to_base64_data_url(&temp_path, Some(&detected_mime)).map_err(|e| e.to_string())?;

        // Hard 20 MB limit — try resize to 5 MB before giving up.
        if image_data_url.len() > MAX_BASE64_BYTES {
            image_data_url = resize_image_for_vision(&temp_path, Some(&detected_mime), RESIZE_TARGET_BYTES)
                .map_err(|e| e.to_string())?;
            if image_data_url.len() > MAX_BASE64_BYTES {
                return Err(format!(
                    "Image too large for vision API: base64 payload is {:.1} MB \
                     (limit {:.0} MB) even after resizing. Install Pillow for better \
                     auto-resize, or compress the image manually.",
                    image_data_url.len() as f64 / (1024.0 * 1024.0),
                    MAX_BASE64_BYTES as f64 / (1024.0 * 1024.0),
                ));
            }
        }

        let (timeout, temperature) = vision_call_params(config);
        let mut messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        })];

        let mut kwargs = LlmCallKwargs {
            task: "vision".to_string(),
            messages: messages.clone(),
            temperature,
            max_tokens: 2000,
            timeout,
            model: model.map(ToString::to_string),
        };

        // Try full-size first; on size-related rejection, downscale and retry.
        let response = match caller.call(&kwargs) {
            Ok(r) => r,
            Err(api_err) => {
                if is_image_size_error(&api_err) && image_data_url.len() > RESIZE_TARGET_BYTES {
                    let resized =
                        resize_image_for_vision(&temp_path, Some(&detected_mime), RESIZE_TARGET_BYTES)
                            .map_err(|e| e.to_string())?;
                    messages[0]["content"][1]["image_url"]["url"] = json!(resized);
                    kwargs.messages = messages.clone();
                    caller.call(&kwargs)?
                } else {
                    return Err(api_err);
                }
            }
        };

        let mut analysis = extract_content_or_reasoning(&response);
        if analysis.is_empty() {
            log::warn!("Vision LLM returned empty content, retrying once");
            let retry = caller.call(&kwargs)?;
            analysis = extract_content_or_reasoning(&retry);
        }
        Ok(analysis)
    })();

    if should_cleanup && temp_path.exists() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

/// Classify an image-analysis error into a user-facing `{success:false,...}` JSON
/// value. Mirrors the `except` block of `vision_analyze_tool`.
pub fn classify_image_error(error: &str, model: Option<&str>) -> Value {
    let error_msg = format!("Error analyzing image: {error}");
    let lower = error.to_lowercase();
    let analysis = if any_hint(&lower, &["402", "insufficient", "payment required", "credits", "billing"]) {
        format!(
            "Insufficient credits or payment required. Please top up your API \
             provider account and try again. Error: {error}"
        )
    } else if any_hint(
        &lower,
        &[
            "does not support",
            "not support image",
            "content_policy",
            "multimodal",
            "unrecognized request argument",
            "image input",
        ],
    ) {
        format!(
            "{} does not support vision or our request was not accepted by the \
             server. Error: {error}",
            model.unwrap_or("None")
        )
    } else if lower.contains("invalid_request") || lower.contains("image_url") {
        format!(
            "The vision API rejected the image. This can happen when the image is \
             in an unsupported format, corrupted, or still too large after \
             auto-resize. Try a smaller JPEG/PNG and retry. Error: {error}"
        )
    } else {
        format!(
            "There was a problem with the request and the image could not be \
             analyzed. Error: {error}"
        )
    };
    json!({
        "success": false,
        "error": error_msg,
        "analysis": analysis,
    })
}

// ---------------------------------------------------------------------------
// Video analysis.
// ---------------------------------------------------------------------------

/// Extension → MIME for video. avi/mkv fall back to mp4. Mirrors `_VIDEO_MIME_TYPES`.
pub fn detect_video_mime_type(path: &Path) -> Option<String> {
    let mime = match ext_lower(path).as_deref()? {
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/mov",
        "avi" => "video/mp4",
        "mkv" => "video/mp4",
        "mpeg" => "video/mpeg",
        "mpg" => "video/mpeg",
        _ => return None,
    };
    Some(mime.to_string())
}

fn supported_video_extensions() -> &'static str {
    ".avi, .mkv, .mov, .mp4, .mpeg, .mpg, .webm"
}

/// Convert a video file to a base64 data URL. Mirrors `_video_to_base64_data_url`.
pub fn video_to_base64_data_url(path: &Path, mime_type: Option<&str>) -> std::io::Result<String> {
    let data = std::fs::read(path)?;
    let encoded = BASE64.encode(&data);
    let mime = mime_type
        .map(ToString::to_string)
        .or_else(|| detect_video_mime_type(path))
        .unwrap_or_else(|| "video/mp4".to_string());
    Ok(format!("data:{mime};base64,{encoded}"))
}

fn video_call_params(config: Option<&serde_yaml::Value>) -> (f64, f64) {
    let mut timeout = DEFAULT_VIDEO_TIMEOUT;
    let mut temperature = DEFAULT_VISION_TEMPERATURE;
    if let Some(v) = yaml_number(config, &["auxiliary", "vision", "timeout"]) {
        timeout = v.max(DEFAULT_VIDEO_TIMEOUT);
    }
    if let Some(v) = yaml_number(config, &["auxiliary", "vision", "temperature"]) {
        temperature = v;
    }
    (timeout, temperature)
}

/// Analyze a video from a URL or local file path. Native port of `video_analyze_tool`.
pub fn video_analyze_tool(
    video_url: &str,
    user_prompt: &str,
    model: Option<&str>,
    caller: &dyn LlmCaller,
    config: Option<&serde_yaml::Value>,
    hooks: &SecurityHooks,
) -> String {
    match run_video_analysis(video_url, user_prompt, model, caller, config, hooks) {
        Ok(analysis) => pretty_json(&json!({
            "success": true,
            "analysis": if analysis.is_empty() {
                "There was a problem with the request and the video could not be analyzed.".to_string()
            } else {
                analysis
            },
        })),
        Err(e) => pretty_json(&classify_video_error(&e)),
    }
}

fn run_video_analysis(
    video_url: &str,
    user_prompt: &str,
    model: Option<&str>,
    caller: &dyn LlmCaller,
    config: Option<&serde_yaml::Value>,
    hooks: &SecurityHooks,
) -> Result<String, String> {
    if (hooks.is_interrupted)() {
        return Err("Interrupted".to_string());
    }

    let local_path = resolve_local_path(video_url);
    let (temp_path, should_cleanup) = if local_path.is_file() {
        (local_path, false)
    } else if validate_image_url_with_hooks(video_url, hooks) {
        if let Some(message) = (hooks.check_website_access)(video_url) {
            return Err(message);
        }
        let temp_dir = hermes_cache_dir("cache/video", "temp_video_files");
        let temp_path = temp_dir.join(format!("temp_video_{}.mp4", unique_token()));
        download_to(
            video_url,
            &temp_path,
            60.0,
            MAX_VIDEO_BASE64_BYTES,
            "video/*,*/*;q=0.8",
            MAX_RETRIES,
            hooks,
        )?;
        (temp_path, true)
    } else {
        return Err(
            "Invalid video source. Provide an HTTP/HTTPS URL or a valid local file path."
                .to_string(),
        );
    };

    let result = (|| -> Result<String, String> {
        let video_size_bytes = std::fs::metadata(&temp_path).map(|m| m.len() as usize).unwrap_or(0);

        let detected_mime = detect_video_mime_type(&temp_path).ok_or_else(|| {
            let suffix = temp_path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| format!(".{e}"))
                .unwrap_or_default();
            format!(
                "Unsupported video format: '{suffix}'. Supported: {}",
                supported_video_extensions()
            )
        })?;

        if video_size_bytes > VIDEO_SIZE_WARN_BYTES {
            log::warn!(
                "Video is {:.1} MB — may be slow or rejected",
                video_size_bytes as f64 / (1024.0 * 1024.0)
            );
        }

        let video_data_url =
            video_to_base64_data_url(&temp_path, Some(&detected_mime)).map_err(|e| e.to_string())?;

        if video_data_url.len() > MAX_VIDEO_BASE64_BYTES {
            return Err(format!(
                "Video too large for API: base64 payload is {:.1} MB (limit {:.0} MB). \
                 Compress or trim the video and retry.",
                video_data_url.len() as f64 / (1024.0 * 1024.0),
                MAX_VIDEO_BASE64_BYTES as f64 / (1024.0 * 1024.0),
            ));
        }

        let (timeout, temperature) = video_call_params(config);
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "video_url", "video_url": {"url": video_data_url}}
            ]
        })];

        let kwargs = LlmCallKwargs {
            task: "vision".to_string(),
            messages,
            temperature,
            max_tokens: 4000,
            timeout,
            model: model.map(ToString::to_string),
        };

        let response = caller.call(&kwargs)?;
        let mut analysis = extract_content_or_reasoning(&response);
        if analysis.is_empty() {
            log::warn!("Empty video response, retrying once");
            let retry = caller.call(&kwargs)?;
            analysis = extract_content_or_reasoning(&retry);
        }
        Ok(analysis)
    })();

    if should_cleanup && temp_path.exists() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

/// Classify a video-analysis error. Mirrors the `except` block of `video_analyze_tool`.
pub fn classify_video_error(error: &str) -> Value {
    let error_msg = format!("Error analyzing video: {error}");
    let lower = error.to_lowercase();
    let analysis = if any_hint(&lower, &["402", "insufficient", "payment required", "credits", "billing"]) {
        format!(
            "Insufficient credits or payment required. Please top up your API \
             provider account and try again. Error: {error}"
        )
    } else if any_hint(
        &lower,
        &[
            "does not support",
            "not support video",
            "content_policy",
            "multimodal",
            "unrecognized request argument",
            "video input",
            "video_url",
        ],
    ) {
        format!(
            "The model does not support video analysis or the request was rejected. \
             Ensure you're using a video-capable model (e.g. google/gemini-2.5-flash). \
             Error: {error}"
        )
    } else if any_hint(
        &lower,
        &[
            "too large",
            "payload",
            "413",
            "content_too_large",
            "request_too_large",
            "exceeds",
            "size limit",
        ],
    ) {
        format!(
            "The video is too large for the API. Try compressing or trimming the \
             video (max ~50 MB). Error: {error}"
        )
    } else {
        format!(
            "There was a problem with the request and the video could not be \
             analyzed. Error: {error}"
        )
    };
    json!({
        "success": false,
        "error": error_msg,
        "analysis": analysis,
    })
}

// ---------------------------------------------------------------------------
// Schemas + registry handlers.
// ---------------------------------------------------------------------------

/// JSON schema for the `vision_analyze` tool. Mirrors `VISION_ANALYZE_SCHEMA`.
pub fn vision_analyze_schema() -> Value {
    json!({
        "name": "vision_analyze",
        "description": "Inspect an image from a URL, file path, or tool output when you need closer detail than what's visible in the conversation. If the user's image is already attached to the conversation and you can see it, just answer directly — only call this tool for images referenced by URL/path, images returned inside other tool results (browser screenshots, search thumbnails), or when you need a deeper look at a specific region the main model's vision may have missed.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Image URL (http/https) or local file path to analyze."
                },
                "question": {
                    "type": "string",
                    "description": "Your specific question or request about the image to resolve. The AI will automatically provide a complete image description AND answer your specific question."
                }
            },
            "required": ["image_url", "question"]
        }
    })
}

/// JSON schema for the `video_analyze` tool. Mirrors `VIDEO_ANALYZE_SCHEMA`.
pub fn video_analyze_schema() -> Value {
    json!({
        "name": "video_analyze",
        "description": "Analyze a video from a URL or local file path using a multimodal AI model. Sends the video to a video-capable model (e.g. Gemini) for understanding. Use this for video files — for images, use vision_analyze instead. Supports mp4, webm, mov, avi, mkv, mpeg formats. Note: large videos (>20 MB) may be slow; max ~50 MB.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_url": {
                    "type": "string",
                    "description": "Video URL (http/https) or local file path to analyze."
                },
                "question": {
                    "type": "string",
                    "description": "Your specific question about the video. The AI will describe what happens in the video and answer your question."
                }
            },
            "required": ["video_url", "question"]
        }
    })
}

/// Build the full vision prompt from a user question. Mirrors `_handle_vision_analyze`.
pub fn build_vision_prompt(question: &str) -> String {
    format!(
        "Fully describe and explain everything about this image, then answer the \
         following question:\n\n{question}"
    )
}

/// Build the full video prompt from a user question. Mirrors `_handle_video_analyze`.
pub fn build_video_prompt(question: &str) -> String {
    format!(
        "Fully describe and explain everything happening in this video, including \
         visual content, motion, audio cues, text overlays, and scene transitions. \
         Then answer the following question:\n\n{question}"
    )
}

/// Resolve the vision model override from env (`AUXILIARY_VISION_MODEL`).
pub fn env_vision_model() -> Option<String> {
    std::env::var("AUXILIARY_VISION_MODEL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Resolve the video model override from env, falling back to the vision model.
/// Mirrors `_handle_video_analyze`'s model resolution.
pub fn env_video_model() -> Option<String> {
    std::env::var("AUXILIARY_VIDEO_MODEL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(env_vision_model)
}

/// Registry handler for `vision_analyze`. Mirrors `_handle_vision_analyze`.
pub fn handle_vision_analyze(
    args: &Value,
    caller: &dyn LlmCaller,
    config: Option<&serde_yaml::Value>,
    hooks: &SecurityHooks,
) -> String {
    let image_url = args.get("image_url").and_then(Value::as_str).unwrap_or("");
    let question = args.get("question").and_then(Value::as_str).unwrap_or("");
    let prompt = build_vision_prompt(question);
    let model = env_vision_model();
    vision_analyze_tool(image_url, &prompt, model.as_deref(), caller, config, hooks)
}

/// Registry handler for `video_analyze`. Mirrors `_handle_video_analyze`.
pub fn handle_video_analyze(
    args: &Value,
    caller: &dyn LlmCaller,
    config: Option<&serde_yaml::Value>,
    hooks: &SecurityHooks,
) -> String {
    let video_url = args.get("video_url").and_then(Value::as_str).unwrap_or("");
    let question = args.get("question").and_then(Value::as_str).unwrap_or("");
    let prompt = build_video_prompt(question);
    let model = env_video_model();
    video_analyze_tool(video_url, &prompt, model.as_deref(), caller, config, hooks)
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

fn any_hint(haystack: &str, hints: &[&str]) -> bool {
    hints.iter().any(|h| haystack.contains(h))
}

/// JSON pretty-print with `ensure_ascii=False` semantics (serde keeps UTF-8).
fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

/// Resolve a hermes cache directory, mirroring `hermes_constants.get_hermes_dir`:
/// prefer the legacy `old_name` directory if it already exists under HERMES_HOME,
/// otherwise use `new_subpath`. HERMES_HOME defaults to `~/.hermes`.
fn hermes_cache_dir(new_subpath: &str, old_name: &str) -> PathBuf {
    let home = hermes_home();
    let old_path = home.join(old_name);
    if old_path.exists() {
        return old_path;
    }
    home.join(new_subpath)
}

fn hermes_home() -> PathBuf {
    if let Ok(value) = std::env::var("HERMES_HOME") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return resolve_local_path(trimmed);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".hermes")
}

/// A reasonably unique token for temp filenames (uuid analogue, no uuid crate).
fn unique_token() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    format!("{nanos:x}_{pid:x}")
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct StubCaller {
        responses: std::sync::Mutex<Vec<Result<Value, String>>>,
        captured: std::sync::Mutex<Vec<LlmCallKwargs>>,
    }

    impl StubCaller {
        fn new(responses: Vec<Result<Value, String>>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                captured: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl LlmCaller for StubCaller {
        fn call(&self, kwargs: &LlmCallKwargs) -> Result<Value, String> {
            self.captured.lock().unwrap().push(kwargs.clone());
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Err("no more stub responses".to_string())
            } else {
                responses.remove(0)
            }
        }
    }

    fn write_png(dir: &std::path::Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00\x00\x00\x00\x00")
            .unwrap();
        path
    }

    #[test]
    fn detects_image_mime_from_magic_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let png = write_png(tmp.path(), "x.png");
        assert_eq!(detect_image_mime_type(&png).as_deref(), Some("image/png"));

        let jpg = tmp.path().join("y.jpg");
        std::fs::write(&jpg, [0xff, 0xd8, 0xff, 0x00]).unwrap();
        assert_eq!(detect_image_mime_type(&jpg).as_deref(), Some("image/jpeg"));

        let txt = tmp.path().join("z.txt");
        std::fs::write(&txt, b"not an image").unwrap();
        assert!(detect_image_mime_type(&txt).is_none());
    }

    #[test]
    fn webp_riff_detection() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.webp");
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&[0, 0, 0, 0]);
        data.extend_from_slice(b"WEBP");
        std::fs::write(&p, &data).unwrap();
        assert_eq!(detect_image_mime_type(&p).as_deref(), Some("image/webp"));
    }

    #[test]
    fn determine_mime_defaults_to_jpeg() {
        assert_eq!(determine_mime_type(Path::new("a.png")), "image/png");
        assert_eq!(determine_mime_type(Path::new("a.unknown")), "image/jpeg");
        assert_eq!(determine_mime_type(Path::new("a.SVG")), "image/svg+xml");
    }

    #[test]
    fn video_mime_extension_mapping() {
        assert_eq!(detect_video_mime_type(Path::new("a.mp4")).as_deref(), Some("video/mp4"));
        assert_eq!(detect_video_mime_type(Path::new("a.avi")).as_deref(), Some("video/mp4"));
        assert_eq!(detect_video_mime_type(Path::new("a.webm")).as_deref(), Some("video/webm"));
        assert_eq!(detect_video_mime_type(Path::new("a.mov")).as_deref(), Some("video/mov"));
        assert_eq!(detect_video_mime_type(Path::new("a.mpeg")).as_deref(), Some("video/mpeg"));
        assert!(detect_video_mime_type(Path::new("a.txt")).is_none());
    }

    #[test]
    fn size_error_detection() {
        assert!(is_image_size_error("Request entity too large"));
        assert!(is_image_size_error("HTTP 413"));
        assert!(is_image_size_error("invalid_request: image_url"));
        assert!(!is_image_size_error("authentication failed"));
    }

    #[test]
    fn url_validation_rejects_non_http() {
        assert!(!validate_image_url(""));
        assert!(!validate_image_url("ftp://example.com/a.png"));
        assert!(!validate_image_url("not a url"));
    }

    #[test]
    fn data_url_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let png = write_png(tmp.path(), "x.png");
        let url = image_to_base64_data_url(&png, Some("image/png")).unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn local_non_image_is_rejected_before_llm() {
        let tmp = tempfile::tempdir().unwrap();
        let txt = tmp.path().join("secret.txt");
        std::fs::write(&txt, b"TOP SECRET").unwrap();
        let caller = StubCaller::new(vec![]);
        let out = vision_analyze_tool(
            txt.to_str().unwrap(),
            "describe",
            None,
            &caller,
            None,
            &SecurityHooks::default(),
        );
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["success"].as_bool(), Some(false));
        assert!(
            parsed["error"]
                .as_str()
                .unwrap()
                .contains("Only real image files are supported")
        );
        assert!(caller.captured.lock().unwrap().is_empty());
    }

    #[test]
    fn local_image_dispatches_and_returns_analysis() {
        let tmp = tempfile::tempdir().unwrap();
        let png = write_png(tmp.path(), "x.png");
        let caller = StubCaller::new(vec![Ok(json!({
            "choices": [{"message": {"content": "A tiny PNG."}}]
        }))]);
        let out = vision_analyze_tool(
            png.to_str().unwrap(),
            "What is this?",
            Some("google/gemini-3-flash-preview"),
            &caller,
            None,
            &SecurityHooks::default(),
        );
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["success"].as_bool(), Some(true));
        assert_eq!(parsed["analysis"].as_str(), Some("A tiny PNG."));

        let captured = caller.captured.lock().unwrap();
        assert_eq!(captured.len(), 1);
        let kw = &captured[0];
        assert_eq!(kw.task, "vision");
        assert_eq!(kw.max_tokens, 2000);
        assert_eq!(kw.model.as_deref(), Some("google/gemini-3-flash-preview"));
        let content = kw.messages[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert!(
            content[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }

    #[test]
    fn empty_content_triggers_one_retry() {
        let tmp = tempfile::tempdir().unwrap();
        let png = write_png(tmp.path(), "x.png");
        let caller = StubCaller::new(vec![
            Ok(json!({"choices": [{"message": {"content": ""}}]})),
            Ok(json!({"choices": [{"message": {"content": "second try"}}]})),
        ]);
        let out = vision_analyze_tool(
            png.to_str().unwrap(),
            "q",
            None,
            &caller,
            None,
            &SecurityHooks::default(),
        );
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["analysis"].as_str(), Some("second try"));
        assert_eq!(caller.captured.lock().unwrap().len(), 2);
    }

    #[test]
    fn payment_error_classification() {
        let v = classify_image_error("Error 402: insufficient credits", Some("m"));
        assert_eq!(v["success"].as_bool(), Some(false));
        assert!(v["analysis"].as_str().unwrap().contains("Insufficient credits"));
    }

    #[test]
    fn unsupported_vision_classification_includes_model() {
        let v = classify_image_error("model does not support image input", Some("foo/bar"));
        assert!(v["analysis"].as_str().unwrap().contains("foo/bar"));
        assert!(v["analysis"].as_str().unwrap().contains("does not support vision"));
    }

    #[test]
    fn video_size_error_classification() {
        let v = classify_video_error("payload content_too_large 413");
        assert!(v["analysis"].as_str().unwrap().contains("too large for the API"));
    }

    #[test]
    fn video_unsupported_classification() {
        let v = classify_video_error("model does not support video input");
        assert!(
            v["analysis"]
                .as_str()
                .unwrap()
                .contains("does not support video analysis")
        );
    }

    #[test]
    fn build_prompts_match_python() {
        assert!(build_vision_prompt("Q?").starts_with("Fully describe and explain everything about this image"));
        assert!(build_video_prompt("Q?").starts_with("Fully describe and explain everything happening in this video"));
        assert!(build_vision_prompt("Q?").ends_with("Q?"));
    }

    #[test]
    fn env_model_resolution() {
        unsafe {
            std::env::set_var("AUXILIARY_VISION_MODEL", "  vis/model  ");
            std::env::remove_var("AUXILIARY_VIDEO_MODEL");
        }
        assert_eq!(env_vision_model().as_deref(), Some("vis/model"));
        assert_eq!(env_video_model().as_deref(), Some("vis/model"));
        unsafe {
            std::env::set_var("AUXILIARY_VIDEO_MODEL", "vid/model");
        }
        assert_eq!(env_video_model().as_deref(), Some("vid/model"));
        unsafe {
            std::env::remove_var("AUXILIARY_VISION_MODEL");
            std::env::remove_var("AUXILIARY_VIDEO_MODEL");
        }
    }

    #[test]
    fn download_timeout_resolution_env_override() {
        unsafe {
            std::env::set_var("HERMES_VISION_DOWNLOAD_TIMEOUT", "45.5");
        }
        assert_eq!(resolve_download_timeout(None), 45.5);
        unsafe {
            std::env::set_var("HERMES_VISION_DOWNLOAD_TIMEOUT", "  ");
        }
        assert_eq!(resolve_download_timeout(None), DEFAULT_VISION_DOWNLOAD_TIMEOUT);
        unsafe {
            std::env::remove_var("HERMES_VISION_DOWNLOAD_TIMEOUT");
        }
    }

    #[test]
    fn config_timeout_and_temperature() {
        let cfg: serde_yaml::Value = serde_yaml::from_str(
            "auxiliary:\n  vision:\n    timeout: 77\n    temperature: 0.7\n    download_timeout: 12\n",
        )
        .unwrap();
        assert_eq!(resolve_download_timeout(Some(&cfg)), 12.0);
        let (t, temp) = vision_call_params(Some(&cfg));
        assert_eq!(t, 77.0);
        assert_eq!(temp, 0.7);
        // Video clamps timeout up to the 180s floor.
        let (vt, _) = video_call_params(Some(&cfg));
        assert_eq!(vt, 180.0);
    }

    #[test]
    fn schemas_have_expected_names() {
        assert_eq!(vision_analyze_schema()["name"], "vision_analyze");
        assert_eq!(video_analyze_schema()["name"], "video_analyze");
        assert_eq!(
            vision_analyze_schema()["parameters"]["required"],
            json!(["image_url", "question"])
        );
    }

    #[test]
    fn invalid_source_rejected() {
        let caller = StubCaller::new(vec![]);
        let out = vision_analyze_tool(
            "ftp://x/y.png",
            "q",
            None,
            &caller,
            None,
            &SecurityHooks::default(),
        );
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["success"].as_bool(), Some(false));
        assert!(
            parsed["error"]
                .as_str()
                .unwrap()
                .contains("Invalid image source")
        );
    }
}
