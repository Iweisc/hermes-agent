//! Base platform-adapter logic, ported from `gateway/platforms/base.py`.
//!
//! The Python module mixes pure utility functions (proxy resolution, NO_PROXY
//! matching, media/URL extraction, message truncation, caption merging, channel
//! prompt/skill resolution) with an `asyncio`-driven session-lifecycle engine
//! built around `asyncio.Task`/`asyncio.Event`.
//!
//! This port reproduces the *pure*, behavior-defining logic faithfully and
//! idiomatically in Rust. The async task-orchestration machinery
//! (`_process_message_background`, `_keep_typing`, etc.) is intimately tied to
//! the CPython event loop and is therefore modelled here as plain data
//! structures + synchronous helpers (`SessionState`) that mirror the Python
//! state transitions without dragging in a runtime. Network helpers use
//! `reqwest::blocking` and preserve request construction / response parsing.
//!
//! Cross-refs:
//!   - [`crate::mod_utils::normalize_proxy_url`]
//!   - [`crate::tool_url_safety::is_safe_url`]
//!   - [`crate::mod_hermes_constants::get_hermes_dir`]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use regex::Regex;

// ===========================================================================
// Constants
// ===========================================================================

pub const GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE: &str =
    "Secure secret entry is not supported over messaging. \
Load this skill in the local CLI to be prompted, or add the key to ~/.hermes/.env manually.";

/// Audio file extensions Hermes recognizes for native audio delivery.
pub const AUDIO_EXTS: &[&str] = &[".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"];
/// Telegram Bot API sendAudio accepts only MP3 / M4A.
pub const TELEGRAM_AUDIO_ATTACHMENT_EXTS: &[&str] = &[".mp3", ".m4a"];
/// Telegram sendVoice accepts Opus / OGG.
pub const TELEGRAM_VOICE_EXTS: &[&str] = &[".ogg", ".opus"];

/// Error substrings that indicate a transient *connection* failure worth retrying.
pub const RETRYABLE_ERROR_PATTERNS: &[&str] = &[
    "connecterror",
    "connectionerror",
    "connectionreset",
    "connectionrefused",
    "connecttimeout",
    "network",
    "broken pipe",
    "remotedisconnected",
    "eoferror",
];

/// Supported video extension -> mime type.
pub fn supported_video_types() -> &'static [(&'static str, &'static str)] {
    &[
        (".mp4", "video/mp4"),
        (".mov", "video/quicktime"),
        (".webm", "video/webm"),
        (".mkv", "video/x-matroska"),
        (".avi", "video/x-msvideo"),
    ]
}

/// Supported document extension -> mime type.
pub fn supported_document_types() -> &'static [(&'static str, &'static str)] {
    &[
        (".pdf", "application/pdf"),
        (".md", "text/markdown"),
        (".txt", "text/plain"),
        (".csv", "text/csv"),
        (".log", "text/plain"),
        (".json", "application/json"),
        (".xml", "application/xml"),
        (".yaml", "application/yaml"),
        (".yml", "application/yaml"),
        (".toml", "application/toml"),
        (".ini", "text/plain"),
        (".cfg", "text/plain"),
        (".zip", "application/zip"),
        (
            ".docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            ".xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        (
            ".pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
    ]
}

// ===========================================================================
// Platform-name normalization and audio routing
// ===========================================================================

/// Normalize a platform name into a lowercase string. Mirrors `_platform_name`.
pub fn platform_name(platform: &str) -> String {
    platform.trim().to_lowercase()
}

/// Return True when a media file should use the platform's audio sender.
///
/// Mirrors `should_send_media_as_audio`.
pub fn should_send_media_as_audio(platform: &str, ext: &str, is_voice: bool) -> bool {
    let normalized_ext = ext.to_lowercase();
    if !AUDIO_EXTS.contains(&normalized_ext.as_str()) {
        return false;
    }
    if platform_name(platform) == "telegram" {
        if TELEGRAM_VOICE_EXTS.contains(&normalized_ext.as_str()) {
            return is_voice;
        }
        return TELEGRAM_AUDIO_ATTACHMENT_EXTS.contains(&normalized_ext.as_str());
    }
    true
}

// ===========================================================================
// UTF-16 length helpers (Telegram measures message length in UTF-16 units)
// ===========================================================================

/// Count UTF-16 code units in `s`. Mirrors `utf16_len`.
pub fn utf16_len(s: &str) -> usize {
    s.chars().map(|c| c.len_utf16()).sum()
}

/// Return the largest char-index `n` such that `len_fn(&s[..n_bytes]) <= budget`,
/// where the returned value is a *character count* (Python codepoint offset).
///
/// Mirrors `_custom_unit_to_cp`. `len_fn` measures length in custom units.
pub fn custom_unit_to_cp<F>(s: &str, budget: usize, len_fn: &F) -> usize
where
    F: Fn(&str) -> usize,
{
    if len_fn(s) <= budget {
        return s.chars().count();
    }
    let chars: Vec<char> = s.chars().collect();
    let mut lo = 0usize;
    let mut hi = chars.len();
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        let prefix: String = chars[..mid].iter().collect();
        if len_fn(&prefix) <= budget {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Return the longest prefix of `s` whose UTF-16 length <= `limit`.
/// Mirrors `_prefix_within_utf16_limit`.
pub fn prefix_within_utf16_limit(s: &str, limit: usize) -> String {
    if utf16_len(s) <= limit {
        return s.to_string();
    }
    let n = custom_unit_to_cp(s, limit, &(utf16_len as fn(&str) -> usize));
    s.chars().take(n).collect()
}

// ===========================================================================
// Network-accessibility and NO_PROXY logic
// ===========================================================================

/// Return True if `host` would expose the server beyond loopback.
///
/// Mirrors `is_network_accessible`. IP literals are checked directly; hostnames
/// are resolved via DNS and DNS failure fails *open* (returns True).
pub fn is_network_accessible(host: &str) -> bool {
    use std::net::IpAddr;
    if let Ok(addr) = host.parse::<IpAddr>() {
        if addr.is_loopback() {
            return false;
        }
        // IPv4-mapped IPv6 (::ffff:127.0.0.1) — Rust's is_loopback is false for
        // those, so unwrap the mapped IPv4 explicitly.
        if let IpAddr::V6(v6) = addr {
            if let Some(v4) = v6.to_ipv4_mapped() {
                if v4.is_loopback() {
                    return false;
                }
            }
        }
        return true;
    }

    // Hostname: resolve and inspect addresses.
    use std::net::ToSocketAddrs;
    match (host, 0u16).to_socket_addrs() {
        Ok(addrs) => {
            let mut any = false;
            for sa in addrs {
                any = true;
                if !sa.ip().is_loopback() {
                    return true;
                }
            }
            // Resolved but every address is loopback -> not accessible.
            // If it didn't resolve at all (no iterations), fall closed=False
            // matches Python: a successful getaddrinfo with only loopback -> False.
            let _ = any;
            false
        }
        // DNS / OS error -> fail open.
        Err(_) => true,
    }
}

/// Split a `host[:port]` (or URL) into a normalized `(host, port)` pair.
/// Mirrors `_split_host_port`.
pub fn split_host_port(value: &str) -> (String, Option<u16>) {
    let raw = value.trim();
    if raw.is_empty() {
        return (String::new(), None);
    }
    if raw.contains("://") {
        if let Ok(parsed) = url::Url::parse(raw) {
            let host = parsed
                .host_str()
                .unwrap_or("")
                .to_lowercase()
                .trim_end_matches('.')
                .to_string();
            return (host, parsed.port());
        }
        return (String::new(), None);
    }
    if raw.starts_with('[') && raw.contains(']') {
        let inner = &raw[1..];
        if let Some(idx) = inner.find(']') {
            let host = &inner[..idx];
            let rest = &inner[idx + 1..];
            let mut port = None;
            if let Some(stripped) = rest.strip_prefix(':') {
                if !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit()) {
                    port = stripped.parse::<u16>().ok();
                }
            }
            return (host.to_lowercase().trim_end_matches('.').to_string(), port);
        }
    }
    if raw.matches(':').count() == 1 {
        if let Some(idx) = raw.rfind(':') {
            let host = &raw[..idx];
            let maybe_port = &raw[idx + 1..];
            if !maybe_port.is_empty() && maybe_port.chars().all(|c| c.is_ascii_digit()) {
                return (
                    host.to_lowercase().trim_end_matches('.').to_string(),
                    maybe_port.parse::<u16>().ok(),
                );
            }
        }
    }
    (
        raw.to_lowercase()
            .trim_matches(|c| c == '[' || c == ']')
            .trim_end_matches('.')
            .to_string(),
        None,
    )
}

/// Read NO_PROXY/no_proxy entries from the environment. Mirrors `_no_proxy_entries`.
pub fn no_proxy_entries() -> Vec<String> {
    let mut entries = Vec::new();
    for key in ["NO_PROXY", "no_proxy"] {
        if let Ok(raw) = std::env::var(key) {
            for part in raw.split(',') {
                let p = part.trim();
                if !p.is_empty() {
                    entries.push(p.to_string());
                }
            }
        }
    }
    entries
}

/// Does a single NO_PROXY `entry` match `host` (optionally with `port`)?
/// Mirrors `_no_proxy_entry_matches`.
pub fn no_proxy_entry_matches(entry: &str, host: &str, port: Option<u16>) -> bool {
    use std::net::IpAddr;
    let token = entry.trim().to_lowercase();
    if token.is_empty() {
        return false;
    }
    if token == "*" {
        return true;
    }

    let (token_host, token_port) = split_host_port(&token);
    if let Some(tp) = token_port {
        match port {
            Some(p) if tp != p => return false,
            None => return false,
            _ => {}
        }
    }
    if token_host.is_empty() {
        return false;
    }

    // CIDR network match.
    if let Some(network) = parse_cidr(&token_host) {
        if let Ok(addr) = host.parse::<IpAddr>() {
            return cidr_contains(&network, &addr);
        }
        return false;
    }

    // Exact IP literal.
    if let Ok(token_ip) = token_host.parse::<IpAddr>() {
        if let Ok(addr) = host.parse::<IpAddr>() {
            return addr == token_ip;
        }
        return false;
    }

    if let Some(suffix) = token_host.strip_prefix('*') {
        // "*.example.com" -> suffix is ".example.com"
        return host.ends_with(suffix);
    }
    if let Some(rest) = token_host.strip_prefix('.') {
        return host == rest || host.ends_with(&token_host);
    }
    host == token_host || host.ends_with(&format!(".{token_host}"))
}

/// Parsed CIDR network: base address + prefix length.
#[derive(Debug, Clone)]
struct CidrNetwork {
    base: std::net::IpAddr,
    prefix: u8,
}

fn parse_cidr(token: &str) -> Option<CidrNetwork> {
    use std::net::IpAddr;
    let (addr_part, prefix_part) = token.split_once('/')?;
    let base: IpAddr = addr_part.parse().ok()?;
    let prefix: u8 = prefix_part.parse().ok()?;
    let max = match base {
        IpAddr::V4(_) => 32,
        IpAddr::V6(_) => 128,
    };
    if prefix > max {
        return None;
    }
    Some(CidrNetwork { base, prefix })
}

fn cidr_contains(network: &CidrNetwork, addr: &std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match (&network.base, addr) {
        (IpAddr::V4(net), IpAddr::V4(ip)) => {
            let net_bits = u32::from(*net);
            let ip_bits = u32::from(*ip);
            if network.prefix == 0 {
                return true;
            }
            let mask = u32::MAX << (32 - network.prefix as u32);
            (net_bits & mask) == (ip_bits & mask)
        }
        (IpAddr::V6(net), IpAddr::V6(ip)) => {
            let net_bits = u128::from(*net);
            let ip_bits = u128::from(*ip);
            if network.prefix == 0 {
                return true;
            }
            let mask = u128::MAX << (128 - network.prefix as u32);
            (net_bits & mask) == (ip_bits & mask)
        }
        _ => false,
    }
}

/// Return True when NO_PROXY/no_proxy matches at least one target host.
/// Mirrors `should_bypass_proxy`.
pub fn should_bypass_proxy(target_hosts: &[String]) -> bool {
    let entries = no_proxy_entries();
    if entries.is_empty() || target_hosts.is_empty() {
        return false;
    }
    for candidate in target_hosts {
        let (host, port) = split_host_port(candidate);
        if host.is_empty() {
            continue;
        }
        if entries
            .iter()
            .any(|entry| no_proxy_entry_matches(entry, &host, port))
        {
            return true;
        }
    }
    false
}

/// Detect the macOS system HTTP(S) proxy via `scutil --proxy`.
/// Mirrors `_detect_macos_system_proxy`. Returns None on non-macOS or error.
pub fn detect_macos_system_proxy() -> Option<String> {
    if std::env::consts::OS != "macos" {
        return None;
    }
    let out = std::process::Command::new("scutil")
        .arg("--proxy")
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        // Python uses check_output which raises on nonzero; mirror by None.
        // (scutil --proxy returns 0 normally; defensive.)
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut props: HashMap<String, String> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some((key, val)) = line.split_once(" : ") {
            props.insert(key.trim().to_string(), val.trim().to_string());
        }
    }
    for (enable_key, host_key, port_key) in [
        ("HTTPSEnable", "HTTPSProxy", "HTTPSPort"),
        ("HTTPEnable", "HTTPProxy", "HTTPPort"),
    ] {
        if props.get(enable_key).map(|s| s.as_str()) == Some("1") {
            let host = props.get(host_key);
            let port = props.get(port_key);
            if let (Some(h), Some(p)) = (host, port) {
                if !h.is_empty() && !p.is_empty() {
                    return Some(format!("http://{h}:{p}"));
                }
            }
        }
    }
    None
}

/// Return a proxy URL from env vars or the macOS system proxy.
///
/// Mirrors `resolve_proxy_url`. `normalize` is applied to candidate values
/// (pass [`crate::mod_utils::normalize_proxy_url`]).
pub fn resolve_proxy_url(
    platform_env_var: Option<&str>,
    target_hosts: &[String],
) -> Option<String> {
    if let Some(var) = platform_env_var {
        let value = std::env::var(var).unwrap_or_default();
        let value = value.trim();
        if !value.is_empty() {
            if should_bypass_proxy(target_hosts) {
                return None;
            }
            return crate::mod_utils::normalize_proxy_url(Some(value));
        }
    }
    for key in [
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ] {
        let value = std::env::var(key).unwrap_or_default();
        let value = value.trim();
        if !value.is_empty() {
            if should_bypass_proxy(target_hosts) {
                return None;
            }
            return crate::mod_utils::normalize_proxy_url(Some(value));
        }
    }
    let detected =
        crate::mod_utils::normalize_proxy_url(detect_macos_system_proxy().as_deref());
    if detected.is_some() && should_bypass_proxy(target_hosts) {
        return None;
    }
    detected
}

/// Kind of proxy kwargs a bot library expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyKwargs {
    /// No proxy configured.
    None,
    /// HTTP proxy: pass `proxy=url`.
    Http(String),
    /// SOCKS proxy: build a ProxyConnector with rdns=True.
    SocksConnector(String),
}

/// Build proxy kwargs for `commands.Bot()` / `discord.Client()`.
/// Mirrors `proxy_kwargs_for_bot`.
pub fn proxy_kwargs_for_bot(proxy_url: Option<&str>) -> ProxyKwargs {
    match proxy_url {
        None => ProxyKwargs::None,
        Some(url) if url.is_empty() => ProxyKwargs::None,
        Some(url) => {
            if url.to_lowercase().starts_with("socks") {
                ProxyKwargs::SocksConnector(url.to_string())
            } else {
                ProxyKwargs::Http(url.to_string())
            }
        }
    }
}

/// Return True when `hostname` matches a NO_PROXY entry.
///
/// Mirrors `is_host_excluded_by_no_proxy`. Supports comma/whitespace-separated
/// entries with optional leading dots and `*.` wildcards.
pub fn is_host_excluded_by_no_proxy(hostname: &str, no_proxy_value: Option<&str>) -> bool {
    let raw_owned;
    let raw = match no_proxy_value {
        Some(v) => v,
        None => {
            raw_owned = std::env::var("NO_PROXY")
                .ok()
                .filter(|s| !s.is_empty())
                .or_else(|| std::env::var("no_proxy").ok())
                .unwrap_or_default();
            &raw_owned
        }
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return false;
    }
    let lower_hostname = hostname.to_lowercase();
    let re = Regex::new(r"[\s,]+").unwrap();
    for entry in re.split(raw) {
        let mut normalized = entry.trim().to_lowercase();
        if normalized.is_empty() {
            continue;
        }
        if normalized == "*" {
            return true;
        }
        if let Some(rest) = normalized.strip_prefix("*.") {
            normalized = rest.to_string();
        } else if let Some(rest) = normalized.strip_prefix('.') {
            normalized = rest.to_string();
        }
        if lower_hostname == normalized || lower_hostname.ends_with(&format!(".{normalized}")) {
            return true;
        }
    }
    false
}

// ===========================================================================
// URL log sanitization
// ===========================================================================

/// Return a URL string safe for logs (no query/fragment/userinfo).
/// Mirrors `safe_url_for_log`.
pub fn safe_url_for_log(url: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    if url.is_empty() {
        return String::new();
    }
    let raw = url;

    let safe = match url::Url::parse(raw) {
        Ok(parsed) if !parsed.scheme().is_empty() && parsed.has_host() => {
            // netloc without userinfo
            let host = parsed.host_str().unwrap_or("");
            let netloc = match parsed.port() {
                Some(p) => format!("{host}:{p}"),
                None => host.to_string(),
            };
            let base = format!("{}://{}", parsed.scheme(), netloc);
            let path = parsed.path();
            if !path.is_empty() && path != "/" {
                let basename = path.rsplit('/').next().unwrap_or("");
                if !basename.is_empty() {
                    format!("{base}/.../{basename}")
                } else {
                    format!("{base}/...")
                }
            } else {
                base
            }
        }
        _ => raw.to_string(),
    };

    if safe.chars().count() <= max_len {
        return safe;
    }
    if max_len <= 3 {
        return ".".repeat(max_len);
    }
    let truncated: String = safe.chars().take(max_len - 3).collect();
    format!("{truncated}...")
}

// ===========================================================================
// Image / audio / document detection + caching
// ===========================================================================

/// Return True if `data` starts with a known image magic-byte sequence.
/// Mirrors `_looks_like_image`.
pub fn looks_like_image(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    if data.len() >= 8 && &data[..8] == b"\x89PNG\r\n\x1a\n" {
        return true;
    }
    if &data[..3] == b"\xff\xd8\xff" {
        return true;
    }
    if data.len() >= 6 && (&data[..6] == b"GIF87a" || &data[..6] == b"GIF89a") {
        return true;
    }
    if &data[..2] == b"BM" {
        return true;
    }
    if &data[..4] == b"RIFF" && data.len() >= 12 && &data[8..12] == b"WEBP" {
        return true;
    }
    false
}

/// Image cache directory: `{HERMES_HOME}/cache/images` (legacy `image_cache/`).
pub fn image_cache_dir() -> PathBuf {
    crate::mod_hermes_constants::get_hermes_dir("cache/images", "image_cache")
}

/// Return the image cache directory, creating it if missing. Mirrors `get_image_cache_dir`.
pub fn get_image_cache_dir() -> std::io::Result<PathBuf> {
    let dir = image_cache_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Generate a 12-hex-char uuid prefix matching `uuid.uuid4().hex[:12]`.
fn uuid12() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Two random sources combined; only needs to be unique enough for filenames.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    let mixed = nanos
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(pid.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    format!("{:012x}", mixed & 0xFFFF_FFFF_FFFF)
}

/// Save raw image bytes to the cache and return the absolute file path.
/// Mirrors `cache_image_from_bytes`. Errors when `data` is not a valid image.
pub fn cache_image_from_bytes(data: &[u8], ext: &str) -> Result<String, String> {
    if !looks_like_image(data) {
        let snippet_len = data.len().min(80);
        let snippet = String::from_utf8_lossy(&data[..snippet_len]);
        return Err(format!(
            "Refusing to cache non-image data as {ext} (starts with: {snippet:?})"
        ));
    }
    let cache_dir = get_image_cache_dir().map_err(|e| e.to_string())?;
    let filename = format!("img_{}{}", uuid12(), ext);
    let filepath = cache_dir.join(&filename);
    std::fs::write(&filepath, data).map_err(|e| e.to_string())?;
    Ok(filepath.to_string_lossy().into_owned())
}

/// Audio cache directory: `{HERMES_HOME}/cache/audio` (legacy `audio_cache/`).
pub fn audio_cache_dir() -> PathBuf {
    crate::mod_hermes_constants::get_hermes_dir("cache/audio", "audio_cache")
}

/// Return the audio cache directory, creating it if missing.
pub fn get_audio_cache_dir() -> std::io::Result<PathBuf> {
    let dir = audio_cache_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Save raw audio bytes to the cache and return the absolute file path.
/// Mirrors `cache_audio_from_bytes`.
pub fn cache_audio_from_bytes(data: &[u8], ext: &str) -> Result<String, String> {
    let cache_dir = get_audio_cache_dir().map_err(|e| e.to_string())?;
    let filename = format!("audio_{}{}", uuid12(), ext);
    let filepath = cache_dir.join(&filename);
    std::fs::write(&filepath, data).map_err(|e| e.to_string())?;
    Ok(filepath.to_string_lossy().into_owned())
}

/// Video cache directory.
pub fn video_cache_dir() -> PathBuf {
    crate::mod_hermes_constants::get_hermes_dir("cache/videos", "video_cache")
}

/// Return the video cache directory, creating it if missing.
pub fn get_video_cache_dir() -> std::io::Result<PathBuf> {
    let dir = video_cache_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Save raw video bytes to the cache. Mirrors `cache_video_from_bytes`.
pub fn cache_video_from_bytes(data: &[u8], ext: &str) -> Result<String, String> {
    let cache_dir = get_video_cache_dir().map_err(|e| e.to_string())?;
    let filename = format!("video_{}{}", uuid12(), ext);
    let filepath = cache_dir.join(&filename);
    std::fs::write(&filepath, data).map_err(|e| e.to_string())?;
    Ok(filepath.to_string_lossy().into_owned())
}

/// Document cache directory.
pub fn document_cache_dir() -> PathBuf {
    crate::mod_hermes_constants::get_hermes_dir("cache/documents", "document_cache")
}

/// Return the document cache directory, creating it if missing.
pub fn get_document_cache_dir() -> std::io::Result<PathBuf> {
    let dir = document_cache_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Save raw document bytes to the cache, preserving the original name.
/// Mirrors `cache_document_from_bytes`, including path-traversal rejection.
pub fn cache_document_from_bytes(data: &[u8], filename: &str) -> Result<String, String> {
    let cache_dir = get_document_cache_dir().map_err(|e| e.to_string())?;
    // Strip directory components, null bytes, control whitespace.
    let mut safe_name = Path::new(filename)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    safe_name = safe_name.replace('\u{0}', "");
    safe_name = safe_name.trim().to_string();
    if safe_name.is_empty() || safe_name == "." || safe_name == ".." {
        safe_name = "document".to_string();
    }
    let cached_name = format!("doc_{}_{}", uuid12(), safe_name);
    let filepath = cache_dir.join(&cached_name);
    // Final safety check: ensure path stays inside cache dir.
    let resolved_parent = filepath
        .parent()
        .and_then(|p| p.canonicalize().ok())
        .unwrap_or_else(|| cache_dir.clone());
    let resolved_cache = cache_dir.canonicalize().unwrap_or_else(|_| cache_dir.clone());
    if !resolved_parent.starts_with(&resolved_cache) {
        return Err(format!("Path traversal rejected: {filename:?}"));
    }
    std::fs::write(&filepath, data).map_err(|e| e.to_string())?;
    Ok(filepath.to_string_lossy().into_owned())
}

/// Delete cached files older than `max_age_hours`. Returns count removed.
/// Shared by `cleanup_image_cache` / `cleanup_document_cache`.
pub fn cleanup_cache_dir(cache_dir: &Path, max_age_hours: u64) -> usize {
    use std::time::{Duration, SystemTime};
    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(max_age_hours * 3600))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let mut removed = 0;
    let entries = match std::fs::read_dir(cache_dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if mtime < cutoff && std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Delete cached images older than `max_age_hours`. Mirrors `cleanup_image_cache`.
pub fn cleanup_image_cache(max_age_hours: u64) -> usize {
    match get_image_cache_dir() {
        Ok(dir) => cleanup_cache_dir(&dir, max_age_hours),
        Err(_) => 0,
    }
}

/// Delete cached documents older than `max_age_hours`. Mirrors `cleanup_document_cache`.
pub fn cleanup_document_cache(max_age_hours: u64) -> usize {
    match get_document_cache_dir() {
        Ok(dir) => cleanup_cache_dir(&dir, max_age_hours),
        Err(_) => 0,
    }
}

// ===========================================================================
// Enums + message types
// ===========================================================================

/// Types of incoming messages. Mirrors `MessageType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    Text,
    Location,
    Photo,
    Video,
    Audio,
    Voice,
    Document,
    Sticker,
    Command,
}

impl MessageType {
    pub fn value(&self) -> &'static str {
        match self {
            MessageType::Text => "text",
            MessageType::Location => "location",
            MessageType::Photo => "photo",
            MessageType::Video => "video",
            MessageType::Audio => "audio",
            MessageType::Voice => "voice",
            MessageType::Document => "document",
            MessageType::Sticker => "sticker",
            MessageType::Command => "command",
        }
    }
}

/// Result classification for message-processing lifecycle hooks.
/// Mirrors `ProcessingOutcome`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingOutcome {
    Success,
    Failure,
    Cancelled,
}

impl ProcessingOutcome {
    pub fn value(&self) -> &'static str {
        match self {
            ProcessingOutcome::Success => "success",
            ProcessingOutcome::Failure => "failure",
            ProcessingOutcome::Cancelled => "cancelled",
        }
    }
}

/// Minimal session-source descriptor used by [`MessageEvent`].
///
/// The full `gateway.session.SessionSource` is richer; this captures the fields
/// the base adapter actually reads (`chat_type`, `chat_id`, `thread_id`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionSource {
    pub platform: String,
    pub chat_id: String,
    pub chat_name: Option<String>,
    pub chat_type: String,
    pub user_id: Option<String>,
    pub user_name: Option<String>,
    pub thread_id: Option<String>,
    pub chat_topic: Option<String>,
    pub user_id_alt: Option<String>,
    pub chat_id_alt: Option<String>,
    pub is_bot: bool,
    pub guild_id: Option<String>,
    pub parent_chat_id: Option<String>,
    pub message_id: Option<String>,
}

/// Incoming message from a platform. Normalized representation that all
/// adapters produce. Mirrors `MessageEvent`.
#[derive(Debug, Clone)]
pub struct MessageEvent {
    pub text: String,
    pub message_type: MessageType,
    pub source: SessionSource,
    pub message_id: Option<String>,
    pub platform_update_id: Option<i64>,
    pub media_urls: Vec<String>,
    pub media_types: Vec<String>,
    pub reply_to_message_id: Option<String>,
    pub reply_to_text: Option<String>,
    pub auto_skill: Vec<String>,
    pub channel_prompt: Option<String>,
    pub internal: bool,
}

impl Default for MessageEvent {
    fn default() -> Self {
        MessageEvent {
            text: String::new(),
            message_type: MessageType::Text,
            source: SessionSource::default(),
            message_id: None,
            platform_update_id: None,
            media_urls: Vec::new(),
            media_types: Vec::new(),
            reply_to_message_id: None,
            reply_to_text: None,
            auto_skill: Vec::new(),
            channel_prompt: None,
            internal: false,
        }
    }
}

impl MessageEvent {
    /// Check if this is a command message (e.g. `/new`). Mirrors `is_command`.
    pub fn is_command(&self) -> bool {
        self.text.starts_with('/')
    }

    /// Extract the command name if this is a command message. Mirrors `get_command`.
    pub fn get_command(&self) -> Option<String> {
        if !self.is_command() {
            return None;
        }
        let first = self.text.split_whitespace().next()?;
        // Strip the leading '/'.
        let mut raw = first[1..].to_lowercase();
        if raw.is_empty() {
            return None;
        }
        if raw.contains('@') {
            raw = raw.split('@').next().unwrap_or("").to_string();
        }
        // Reject file paths: valid command names never contain '/'.
        if raw.contains('/') {
            return None;
        }
        if raw.is_empty() { None } else { Some(raw) }
    }

    /// Get the arguments after a command. Mirrors `get_command_args`.
    ///
    /// Python uses `str.split(maxsplit=1)`, which strips the leading whitespace
    /// run before the args, then applies the iOS dash-autocorrect rewrites in
    /// the exact order `"——" -> "--"`, `"—" -> "--"`, `"–" -> "-"`.
    pub fn get_command_args(&self) -> String {
        if !self.is_command() {
            return self.text.clone();
        }
        let args = match self.text.splitn(2, char::is_whitespace).nth(1) {
            Some(rest) => rest.trim_start().to_string(),
            None => String::new(),
        };
        args.replace("\u{2014}\u{2014}", "--")
            .replace('\u{2014}', "--")
            .replace('\u{2013}', "-")
    }
}

/// Plaintext gateway-restart patterns. Mirrors `_PLAINTEXT_GATEWAY_RESTART_PATTERNS`.
fn plaintext_gateway_restart_patterns() -> Vec<Regex> {
    vec![
        Regex::new(r"(?i)^(?:please\s+)?restart\s+(?:the\s+)?gateway[.!?\s]*$").unwrap(),
        Regex::new(r"(?i)^(?:please\s+)?restart\s+(?:the\s+)?hermes\s+gateway[.!?\s]*$").unwrap(),
        Regex::new(r"(?i)^(?:please\s+)?restart\s+hermes[.!?\s]*$").unwrap(),
    ]
}

/// Rewrite a tiny set of DM plaintext admin phrases into slash commands.
/// Mirrors `coerce_plaintext_gateway_command`. Mutates `event` in place.
pub fn coerce_plaintext_gateway_command(event: &mut MessageEvent) {
    if event.message_type != MessageType::Text {
        return;
    }
    let text = event.text.trim().to_string();
    if text.is_empty() || text.starts_with('/') {
        return;
    }
    if event.source.chat_type != "dm" {
        return;
    }
    for pattern in plaintext_gateway_restart_patterns() {
        if pattern.is_match(&text) {
            event.text = "/restart".to_string();
            return;
        }
    }
}

/// Result of sending a message. Mirrors `SendResult`.
#[derive(Debug, Clone, Default)]
pub struct SendResult {
    pub success: bool,
    pub message_id: Option<String>,
    pub error: Option<String>,
    pub retryable: bool,
}

impl SendResult {
    pub fn ok(message_id: Option<String>) -> Self {
        SendResult {
            success: true,
            message_id,
            error: None,
            retryable: false,
        }
    }

    pub fn fail(error: impl Into<String>) -> Self {
        SendResult {
            success: false,
            message_id: None,
            error: Some(error.into()),
            retryable: false,
        }
    }

    pub fn not_supported() -> Self {
        SendResult::fail("Not supported")
    }
}

/// System-notice reply that auto-deletes after a TTL. Mirrors `EphemeralReply`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EphemeralReply {
    pub text: String,
    pub ttl_seconds: Option<i64>,
}

impl EphemeralReply {
    pub fn new(text: impl Into<String>, ttl_seconds: Option<i64>) -> Self {
        EphemeralReply {
            text: text.into(),
            ttl_seconds,
        }
    }
}

// ===========================================================================
// Pending-message merge logic
// ===========================================================================

/// Merge a new caption into existing text, avoiding duplicates.
/// Mirrors `BasePlatformAdapter._merge_caption`.
pub fn merge_caption(existing_text: Option<&str>, new_text: &str) -> String {
    match existing_text {
        None | Some("") => new_text.to_string(),
        Some(existing) => {
            let existing_captions: Vec<&str> =
                existing.split("\n\n").map(|c| c.trim()).collect();
            if !existing_captions.contains(&new_text.trim()) {
                format!("{existing}\n\n{new_text}").trim().to_string()
            } else {
                existing.to_string()
            }
        }
    }
}

/// Store or merge a pending event for a session.
/// Mirrors `merge_pending_message_event`.
pub fn merge_pending_message_event(
    pending_messages: &mut HashMap<String, MessageEvent>,
    session_key: &str,
    event: MessageEvent,
    merge_text: bool,
) {
    if let Some(existing) = pending_messages.get_mut(session_key) {
        let existing_is_photo = existing.message_type == MessageType::Photo;
        let incoming_is_photo = event.message_type == MessageType::Photo;
        let existing_has_media = !existing.media_urls.is_empty();
        let incoming_has_media = !event.media_urls.is_empty();

        if existing_is_photo && incoming_is_photo {
            existing.media_urls.extend(event.media_urls);
            existing.media_types.extend(event.media_types);
            if !event.text.is_empty() {
                existing.text = merge_caption(Some(&existing.text), &event.text);
            }
            return;
        }

        if existing_has_media || incoming_has_media {
            if incoming_has_media {
                existing.media_urls.extend(event.media_urls);
                existing.media_types.extend(event.media_types);
            }
            if !event.text.is_empty() {
                if !existing.text.is_empty() {
                    existing.text = merge_caption(Some(&existing.text), &event.text);
                } else {
                    existing.text = event.text;
                }
            }
            if existing_is_photo || incoming_is_photo {
                existing.message_type = MessageType::Photo;
            } else if existing.message_type == MessageType::Text
                && event.message_type != MessageType::Text
            {
                existing.message_type = event.message_type;
            }
            return;
        }

        if merge_text
            && existing.message_type == MessageType::Text
            && event.message_type == MessageType::Text
        {
            if !event.text.is_empty() {
                existing.text = if !existing.text.is_empty() {
                    format!("{}\n{}", existing.text, event.text)
                } else {
                    event.text
                };
            }
            return;
        }
    }

    pending_messages.insert(session_key.to_string(), event);
}

// ===========================================================================
// Channel prompt / skill resolution
// ===========================================================================

/// Resolve a per-channel ephemeral prompt from platform config.
/// Mirrors `resolve_channel_prompt`. `config_extra` is the adapter's
/// `config.extra` dict as JSON.
pub fn resolve_channel_prompt(
    config_extra: &serde_json::Value,
    channel_id: &str,
    parent_id: Option<&str>,
) -> Option<String> {
    let prompts = config_extra.get("channel_prompts")?;
    let prompts = prompts.as_object()?;
    for key in [Some(channel_id), parent_id] {
        let key = match key {
            Some(k) if !k.is_empty() => k,
            _ => continue,
        };
        if let Some(prompt) = prompts.get(key) {
            if prompt.is_null() {
                continue;
            }
            let s = match prompt {
                serde_json::Value::String(s) => s.trim().to_string(),
                other => other.to_string(),
            };
            // For non-string values Python does str(prompt).strip(); JSON
            // numbers/bools are stringified. Keep behavior simple: use string
            // representation for strings, else the raw scalar text.
            let s = if let serde_json::Value::String(orig) = prompt {
                orig.trim().to_string()
            } else {
                s
            };
            if !s.is_empty() {
                return Some(s);
            }
        }
    }
    None
}

/// Resolve auto-loaded skill(s) for a channel/thread from platform config.
/// Mirrors `resolve_channel_skills`.
pub fn resolve_channel_skills(
    config_extra: &serde_json::Value,
    channel_id: &str,
    parent_id: Option<&str>,
) -> Option<Vec<String>> {
    let bindings = config_extra.get("channel_skill_bindings")?;
    let bindings = bindings.as_array()?;
    if bindings.is_empty() {
        return None;
    }
    let mut ids_to_check: Vec<String> = Vec::new();
    if !channel_id.is_empty() {
        ids_to_check.push(channel_id.to_string());
    }
    if let Some(p) = parent_id {
        if !p.is_empty() {
            ids_to_check.push(p.to_string());
        }
    }
    if ids_to_check.is_empty() {
        return None;
    }
    for entry in bindings {
        let entry = match entry.as_object() {
            Some(o) => o,
            None => continue,
        };
        let entry_id = match entry.get("id") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => String::new(),
        };
        if !ids_to_check.contains(&entry_id) {
            continue;
        }
        let skills = entry.get("skills").or_else(|| entry.get("skill"));
        match skills {
            Some(serde_json::Value::String(s)) => {
                let trimmed = s.trim();
                return if trimmed.is_empty() {
                    None
                } else {
                    Some(vec![trimmed.to_string()])
                };
            }
            Some(serde_json::Value::Array(arr)) if !arr.is_empty() => {
                let mut seen: Vec<String> = Vec::new();
                for name in arr {
                    if let serde_json::Value::String(s) = name {
                        let nm = s.trim().to_string();
                        if !nm.is_empty() && !seen.contains(&nm) {
                            seen.push(nm);
                        }
                    }
                }
                return if seen.is_empty() { None } else { Some(seen) };
            }
            _ => {}
        }
    }
    None
}

// ===========================================================================
// Error classification + ephemeral unwrap
// ===========================================================================

/// Return True if the error string looks like a transient network failure.
/// Mirrors `_is_retryable_error`.
pub fn is_retryable_error(error: Option<&str>) -> bool {
    match error {
        None => false,
        Some(e) => {
            let lowered = e.to_lowercase();
            RETRYABLE_ERROR_PATTERNS
                .iter()
                .any(|pat| lowered.contains(pat))
        }
    }
}

/// Return True if the error string indicates a read/write timeout.
/// Mirrors `_is_timeout_error`. Timeouts are NOT retryable.
pub fn is_timeout_error(error: Option<&str>) -> bool {
    match error {
        None => false,
        Some(e) => {
            let lowered = e.to_lowercase();
            lowered.contains("timed out")
                || lowered.contains("readtimeout")
                || lowered.contains("writetimeout")
        }
    }
}

/// Handler response: plain text, ephemeral reply, or nothing.
#[derive(Debug, Clone)]
pub enum HandlerResponse {
    None,
    Text(String),
    Ephemeral(EphemeralReply),
}

/// Unwrap a handler response into `(Option<text>, ttl_seconds)`.
/// Mirrors `_unwrap_ephemeral`.
///
/// `ephemeral_ttl_default` is the configured default (from
/// `display.ephemeral_system_ttl`); `delete_supported` indicates whether the
/// adapter overrides `delete_message` — when it doesn't, ttl is forced to 0.
pub fn unwrap_ephemeral(
    response: &HandlerResponse,
    ephemeral_ttl_default: i64,
    delete_supported: bool,
) -> (Option<String>, i64) {
    match response {
        HandlerResponse::Ephemeral(reply) => {
            let mut ttl = reply.ttl_seconds.unwrap_or(ephemeral_ttl_default);
            if ttl > 0 && !delete_supported {
                ttl = 0;
            }
            (Some(reply.text.clone()), ttl.max(0))
        }
        HandlerResponse::Text(t) => (Some(t.clone()), 0),
        HandlerResponse::None => (None, 0),
    }
}

// ===========================================================================
// Animation / image / media extraction
// ===========================================================================

/// Check if a URL points to an animated GIF. Mirrors `_is_animation_url`.
pub fn is_animation_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    let lower = lower.split('?').next().unwrap_or(&lower);
    lower.ends_with(".gif")
}

/// Extract image URLs from markdown / HTML img tags.
/// Mirrors `BasePlatformAdapter.extract_images`. Returns `(images, cleaned)`.
pub fn extract_images(content: &str) -> (Vec<(String, String)>, String) {
    let mut images: Vec<(String, String)> = Vec::new();

    let md_re = Regex::new(r"!\[([^\]]*)\]\((https?://[^\s\)]+)\)").unwrap();
    let html_re =
        Regex::new(r#"<img\s+src=["']?(https?://[^\s"'<>]+)["']?\s*/?>\s*(?:</img>)?"#).unwrap();

    let img_markers = [
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        "fal.media",
        "fal-cdn",
        "replicate.delivery",
    ];

    for caps in md_re.captures_iter(content) {
        let alt_text = caps.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
        let url = caps.get(2).map(|m| m.as_str()).unwrap_or("").to_string();
        let lower = url.to_lowercase();
        if img_markers
            .iter()
            .any(|ext| lower.ends_with(ext) || lower.contains(ext))
        {
            images.push((url, alt_text));
        }
    }

    for caps in html_re.captures_iter(content) {
        let url = caps.get(1).map(|m| m.as_str()).unwrap_or("").to_string();
        images.push((url, String::new()));
    }

    let mut cleaned = content.to_string();
    if !images.is_empty() {
        let extracted_urls: std::collections::HashSet<String> =
            images.iter().map(|(u, _)| u.clone()).collect();

        // Remove only matched image tags (whose url was extracted).
        cleaned = md_re
            .replace_all(&cleaned, |caps: &regex::Captures| {
                let url = caps.get(2).map(|m| m.as_str()).unwrap_or("");
                if extracted_urls.contains(url) {
                    String::new()
                } else {
                    caps.get(0).unwrap().as_str().to_string()
                }
            })
            .into_owned();
        cleaned = html_re
            .replace_all(&cleaned, |caps: &regex::Captures| {
                let url = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                if extracted_urls.contains(url) {
                    String::new()
                } else {
                    caps.get(0).unwrap().as_str().to_string()
                }
            })
            .into_owned();
        let blank = Regex::new(r"\n{3,}").unwrap();
        cleaned = blank.replace_all(&cleaned, "\n\n").trim().to_string();
    }

    (images, cleaned)
}

/// Expand a leading `~/` to the user's home directory. Mirrors `os.path.expanduser`.
pub fn expanduser(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().into_owned();
        }
    } else if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.to_string_lossy().into_owned();
        }
    }
    path.to_string()
}

/// Extract `MEDIA:<path>` tags and `[[audio_as_voice]]` directives.
/// Mirrors `BasePlatformAdapter.extract_media`. Returns `(media, cleaned)`
/// where each media item is `(expanded_path, is_voice)`.
pub fn extract_media(content: &str) -> (Vec<(String, bool)>, String) {
    let mut media: Vec<(String, bool)> = Vec::new();

    let has_voice_tag = content.contains("[[audio_as_voice]]");
    let mut cleaned = content.replace("[[audio_as_voice]]", "");

    let media_re = Regex::new(
        r#"[`"']?MEDIA:\s*(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|(?:~/|/)\S+(?:[^\S\n]+\S+)*?\.(?:png|jpe?g|gif|webp|mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|txt|csv|apk|ipa)(?:[\s`"',;:)\]}]|$)|\S+)[`"']?"#,
    )
    .unwrap();

    for caps in media_re.captures_iter(content) {
        let mut path = caps
            .name("path")
            .map(|m| m.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        // Strip matching surrounding quote/backtick wrappers.
        let chars: Vec<char> = path.chars().collect();
        if chars.len() >= 2
            && chars[0] == chars[chars.len() - 1]
            && (chars[0] == '`' || chars[0] == '"' || chars[0] == '\'')
        {
            path = chars[1..chars.len() - 1].iter().collect::<String>().trim().to_string();
        }
        path = path
            .trim_start_matches(['`', '"', '\''])
            .trim_end_matches(['`', '"', '\'', ',', '.', ';', ':', ')', '}', ']'])
            .to_string();
        if !path.is_empty() {
            media.push((expanduser(&path), has_voice_tag));
        }
    }

    if !media.is_empty() {
        cleaned = media_re.replace_all(&cleaned, "").into_owned();
        let blank = Regex::new(r"\n{3,}").unwrap();
        cleaned = blank.replace_all(&cleaned, "\n\n").trim().to_string();
    }

    (media, cleaned)
}

/// Detect bare local file paths in response text for native media delivery.
/// Mirrors `BasePlatformAdapter.extract_local_files`. Returns `(paths, cleaned)`.
///
/// `is_file` is injected so callers can validate path existence (defaults to
/// `Path::is_file` via [`extract_local_files`]).
pub fn extract_local_files_with<F>(content: &str, is_file: F) -> (Vec<String>, String)
where
    F: Fn(&str) -> bool,
{
    let local_media_exts = [
        "png", "jpg", "jpeg", "gif", "webp", "mp4", "mov", "avi", "mkv", "webm",
    ];
    let ext_part = local_media_exts.join("|");
    let path_pattern = format!(
        r"(?i)(?:^|[^/:\w.])((?:~/|/)(?:[\w.\-]+/)*[\w.\-]+\.(?:{ext_part})\b)"
    );
    // Rust's regex has no lookbehind; emulate (?<![/:\w.]) by matching a
    // preceding non-class char (or start) in a capturing group and re-using the
    // inner capture for the actual path.
    let path_re = Regex::new(&path_pattern).unwrap();

    // Build spans covered by fenced code blocks and inline code.
    let mut code_spans: Vec<(usize, usize)> = Vec::new();
    let fenced = Regex::new(r"(?s)```[^\n]*\n.*?```").unwrap();
    for m in fenced.find_iter(content) {
        code_spans.push((m.start(), m.end()));
    }
    let inline = Regex::new(r"`[^`\n]+`").unwrap();
    for m in inline.find_iter(content) {
        code_spans.push((m.start(), m.end()));
    }
    let in_code = |pos: usize| code_spans.iter().any(|&(s, e)| s <= pos && pos < e);

    let mut found: Vec<(String, String)> = Vec::new(); // (raw, expanded)
    for caps in path_re.captures_iter(content) {
        let inner = match caps.get(1) {
            Some(m) => m,
            None => continue,
        };
        if in_code(inner.start()) {
            continue;
        }
        let raw = inner.as_str().to_string();
        let expanded = expanduser(&raw);
        if is_file(&expanded) {
            found.push((raw, expanded));
        }
    }

    // Deduplicate by expanded path, preserving discovery order.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut unique: Vec<(String, String)> = Vec::new();
    for (raw, expanded) in found {
        if seen.insert(expanded.clone()) {
            unique.push((raw, expanded));
        }
    }

    let paths: Vec<String> = unique.iter().map(|(_, e)| e.clone()).collect();

    let mut cleaned = content.to_string();
    if !unique.is_empty() {
        for (raw, _exp) in &unique {
            cleaned = cleaned.replace(raw, "");
        }
        let blank = Regex::new(r"\n{3,}").unwrap();
        cleaned = blank.replace_all(&cleaned, "\n\n").trim().to_string();
    }

    (paths, cleaned)
}

/// Convenience wrapper validating paths against the filesystem.
pub fn extract_local_files(content: &str) -> (Vec<String>, String) {
    extract_local_files_with(content, |p| Path::new(p).is_file())
}

// ===========================================================================
// Message truncation (code-block aware)
// ===========================================================================

/// Split a long message into chunks, preserving code-block boundaries.
/// Mirrors `BasePlatformAdapter.truncate_message`.
///
/// `len_fn` measures string length; pass [`utf16_len`] for Telegram. `None`
/// defaults to Unicode code-point count (`chars().count()`).
pub fn truncate_message(
    content: &str,
    max_length: usize,
    len_fn: Option<&dyn Fn(&str) -> usize>,
) -> Vec<String> {
    fn cp_len(s: &str) -> usize {
        s.chars().count()
    }
    let default_len: &dyn Fn(&str) -> usize = &cp_len;
    let len_is_default = len_fn.is_none();
    let len_fn: &dyn Fn(&str) -> usize = len_fn.unwrap_or(default_len);

    if len_fn(content) <= max_length {
        return vec![content.to_string()];
    }

    const INDICATOR_RESERVE: usize = 10;
    const FENCE_CLOSE: &str = "\n```";

    let mut chunks: Vec<String> = Vec::new();
    let mut remaining: Vec<char> = content.chars().collect();
    let mut carry_lang: Option<String> = None;

    while !remaining.is_empty() {
        let prefix = match &carry_lang {
            Some(lang) => format!("```{lang}\n"),
            None => String::new(),
        };
        let prefix_len = len_fn(&prefix);
        let fence_close_len = len_fn(FENCE_CLOSE);

        let mut headroom = max_length
            .saturating_sub(INDICATOR_RESERVE)
            .saturating_sub(prefix_len)
            .saturating_sub(fence_close_len);
        if headroom < 1 {
            headroom = max_length / 2;
        }

        let remaining_str: String = remaining.iter().collect();

        // Everything remaining fits in one final chunk.
        if prefix_len + len_fn(&remaining_str) <= max_length.saturating_sub(INDICATOR_RESERVE) {
            chunks.push(format!("{prefix}{remaining_str}"));
            break;
        }

        // Codepoint-based slice limit.
        let cp_limit = if !len_is_default {
            custom_unit_to_cp(&remaining_str, headroom, &|s: &str| len_fn(s))
        } else {
            headroom
        };
        let cp_limit = cp_limit.min(remaining.len());

        // Find a natural split point (prefer newlines, then spaces). Work in
        // char offsets throughout so the slicing matches Python codepoints.
        let region_chars: Vec<char> = remaining[..cp_limit].to_vec();
        let mut split_idx = rfind_char(&region_chars, '\n');
        if split_idx < (cp_limit / 2) as isize {
            split_idx = rfind_char(&region_chars, ' ');
        }
        if split_idx < 1 {
            split_idx = cp_limit as isize;
        }
        let mut split_at = split_idx as usize;

        // Avoid splitting inside an inline code span.
        let candidate: String = remaining[..split_at.min(remaining.len())].iter().collect();
        let backtick_count =
            count_substr(&candidate, "`") as isize - count_substr(&candidate, "\\`") as isize;
        if backtick_count % 2 == 1 {
            let cand_chars: Vec<char> = candidate.chars().collect();
            let mut last_bt = rfind_char(&cand_chars, '`');
            while last_bt > 0 && cand_chars[(last_bt - 1) as usize] == '\\' {
                last_bt = rfind_char_before(&cand_chars, '`', last_bt as usize);
            }
            if last_bt > 0 {
                let space_split = rfind_char_before(&cand_chars, ' ', last_bt as usize);
                let nl_split = rfind_char_before(&cand_chars, '\n', last_bt as usize);
                let safe_split = space_split.max(nl_split);
                if safe_split > (cp_limit / 4) as isize {
                    split_at = safe_split as usize;
                }
            }
        }

        let split_at = split_at.min(remaining.len());
        let chunk_body: String = remaining[..split_at].iter().collect();
        // remaining = remaining[split_at:].lstrip()
        let rest: String = remaining[split_at..].iter().collect();
        remaining = rest.trim_start().chars().collect();

        let mut full_chunk = format!("{prefix}{chunk_body}");

        // Walk chunk_body to detect open code block.
        let mut in_code = carry_lang.is_some();
        let mut lang = carry_lang.clone().unwrap_or_default();
        for line in chunk_body.split('\n') {
            let stripped = line.trim();
            if stripped.starts_with("```") {
                if in_code {
                    in_code = false;
                    lang = String::new();
                } else {
                    in_code = true;
                    let tag = stripped[3..].trim();
                    lang = tag.split_whitespace().next().unwrap_or("").to_string();
                }
            }
        }

        if in_code {
            full_chunk.push_str(FENCE_CLOSE);
            carry_lang = Some(lang);
        } else {
            carry_lang = None;
        }

        chunks.push(full_chunk);
    }

    if chunks.len() > 1 {
        let total = chunks.len();
        chunks = chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| format!("{chunk} ({}/{})", i + 1, total))
            .collect();
    }

    chunks
}

/// Char-offset of the last occurrence of `needle`, or -1 if absent.
fn rfind_char(chars: &[char], needle: char) -> isize {
    for (i, &c) in chars.iter().enumerate().rev() {
        if c == needle {
            return i as isize;
        }
    }
    -1
}

/// Char-offset of the last occurrence of `needle` strictly before `end`.
fn rfind_char_before(chars: &[char], needle: char, end: usize) -> isize {
    let upper = end.min(chars.len());
    for i in (0..upper).rev() {
        if chars[i] == needle {
            return i as isize;
        }
    }
    -1
}

fn count_substr(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    haystack.matches(needle).count()
}

// ===========================================================================
// Human-delay pacing
// ===========================================================================

/// Mode for human-like response pacing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanDelayMode {
    Off,
    Natural,
    Custom,
}

/// Compute a `(min_ms, max_ms)` range for human-like response pacing from env.
/// Mirrors `_get_human_delay` env parsing (without the RNG draw, which the
/// caller performs over the range). Returns `None` when mode is "off".
pub fn human_delay_range_ms() -> Option<(u64, u64)> {
    let mode = std::env::var("HERMES_HUMAN_DELAY_MODE")
        .unwrap_or_else(|_| "off".to_string())
        .to_lowercase();
    match mode.as_str() {
        "off" => None,
        "natural" => Some((800, 2500)),
        _ => {
            let min_ms = std::env::var("HERMES_HUMAN_DELAY_MIN_MS")
                .ok()
                .and_then(|v| v.parse::<i64>().ok())
                .filter(|&v| v >= 0)
                .map(|v| v as u64)
                .unwrap_or(800);
            let max_ms = std::env::var("HERMES_HUMAN_DELAY_MAX_MS")
                .ok()
                .and_then(|v| v.parse::<i64>().ok())
                .filter(|&v| v >= 0)
                .map(|v| v as u64)
                .unwrap_or(2500);
            Some((min_ms, max_ms))
        }
    }
}

// ===========================================================================
// Session lifecycle state (synchronous model of the asyncio engine)
// ===========================================================================

/// Synchronous model of the per-adapter session-tracking maps from
/// `BasePlatformAdapter`.
///
/// The Python original keys these by `session_key` and uses `asyncio.Event` /
/// `asyncio.Task`. Here the interrupt "event" is a simple boolean flag and
/// task identity is an opaque `u64` token, which is sufficient to reproduce the
/// stale-lock detection, guard-release, and pending-merge state transitions
/// deterministically (the actual coroutine scheduling lives in the async
/// runtime layer that drives this state).
#[derive(Debug, Default)]
pub struct SessionState {
    /// session_key -> interrupt flag (mirrors `_active_sessions`).
    pub active_sessions: HashMap<String, bool>,
    /// session_key -> pending follow-up event (mirrors `_pending_messages`).
    pub pending_messages: HashMap<String, MessageEvent>,
    /// session_key -> owner-task token (mirrors `_session_tasks`).
    pub session_tasks: HashMap<String, u64>,
    /// Tasks (tokens) that have completed (used by stale detection).
    pub done_tasks: std::collections::HashSet<u64>,
    /// Auto-TTS config.
    pub auto_tts_default: bool,
    pub auto_tts_enabled_chats: std::collections::HashSet<String>,
    pub auto_tts_disabled_chats: std::collections::HashSet<String>,
    /// Chats where typing is paused (mirrors `_typing_paused`).
    pub typing_paused: std::collections::HashSet<String>,
}

impl SessionState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether auto-TTS on voice input should fire for `chat_id`.
    /// Mirrors `_should_auto_tts_for_chat`.
    pub fn should_auto_tts_for_chat(&self, chat_id: &str) -> bool {
        if self.auto_tts_enabled_chats.contains(chat_id) {
            return true;
        }
        if self.auto_tts_disabled_chats.contains(chat_id) {
            return false;
        }
        self.auto_tts_default
    }

    pub fn pause_typing_for_chat(&mut self, chat_id: &str) {
        self.typing_paused.insert(chat_id.to_string());
    }

    pub fn resume_typing_for_chat(&mut self, chat_id: &str) {
        self.typing_paused.remove(chat_id);
    }

    /// Release the adapter-level guard for a session. Mirrors `_release_session_guard`.
    /// When `guard` is provided, only release if the entry still equals it
    /// (modelled as: only release when the entry exists; the boolean guard
    /// value is not identity-comparable, so callers should manage this via
    /// task ownership instead).
    pub fn release_session_guard(&mut self, session_key: &str) {
        self.active_sessions.remove(session_key);
    }

    /// Return True if the owner task for `session_key` is done/cancelled.
    /// Mirrors `_session_task_is_stale`.
    pub fn session_task_is_stale(&self, session_key: &str) -> bool {
        match self.session_tasks.get(session_key) {
            None => false,
            Some(token) => self.done_tasks.contains(token),
        }
    }

    /// Clear a stale session lock if the owner task is already gone.
    /// Mirrors `_heal_stale_session_lock`. Returns True if healed.
    pub fn heal_stale_session_lock(&mut self, session_key: &str) -> bool {
        if !self.active_sessions.contains_key(session_key) {
            return false;
        }
        if !self.session_task_is_stale(session_key) {
            return false;
        }
        self.active_sessions.remove(session_key);
        self.pending_messages.remove(session_key);
        self.session_tasks.remove(session_key);
        true
    }

    /// Check if there's a pending interrupt for a session. Mirrors `has_pending_interrupt`.
    pub fn has_pending_interrupt(&self, session_key: &str) -> bool {
        self.active_sessions.get(session_key).copied().unwrap_or(false)
    }

    /// Get and clear any pending message for a session. Mirrors `get_pending_message`.
    pub fn get_pending_message(&mut self, session_key: &str) -> Option<MessageEvent> {
        self.pending_messages.remove(session_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_routing_telegram_and_default() {
        assert!(should_send_media_as_audio("whatsapp", ".mp3", false));
        assert!(should_send_media_as_audio("whatsapp", ".opus", false));
        assert!(!should_send_media_as_audio("whatsapp", ".txt", false));
        // Telegram: opus only as audio when is_voice.
        assert!(!should_send_media_as_audio("telegram", ".opus", false));
        assert!(should_send_media_as_audio("telegram", ".opus", true));
        assert!(should_send_media_as_audio("telegram", ".mp3", false));
        assert!(!should_send_media_as_audio("telegram", ".wav", false));
        // Case-insensitive platform.
        assert!(should_send_media_as_audio("Telegram", ".MP3", false));
    }

    #[test]
    fn utf16_counts_surrogate_pairs() {
        assert_eq!(utf16_len("abc"), 3);
        assert_eq!(utf16_len("😀"), 2); // outside BMP
        assert_eq!(utf16_len("a😀b"), 4);
    }

    #[test]
    fn prefix_within_limit_respects_surrogates() {
        let s = "a😀b";
        // limit 3 should keep "a😀" (1 + 2 = 3 units) not slice the emoji.
        assert_eq!(prefix_within_utf16_limit(s, 3), "a😀");
        // limit 2 cannot fit the emoji (would be 3 units), so just "a".
        assert_eq!(prefix_within_utf16_limit(s, 2), "a");
        assert_eq!(prefix_within_utf16_limit(s, 1), "a");
        // limit 4 fits the whole string.
        assert_eq!(prefix_within_utf16_limit(s, 4), "a😀b");
    }

    #[test]
    fn split_host_port_variants() {
        assert_eq!(split_host_port("Example.COM"), ("example.com".into(), None));
        assert_eq!(
            split_host_port("example.com:8080"),
            ("example.com".into(), Some(8080))
        );
        assert_eq!(
            split_host_port("http://Example.com:9090/path"),
            ("example.com".into(), Some(9090))
        );
        assert_eq!(
            split_host_port("[::1]:443"),
            ("::1".into(), Some(443))
        );
    }

    #[test]
    fn no_proxy_matching() {
        assert!(no_proxy_entry_matches("*", "anything", None));
        assert!(no_proxy_entry_matches("example.com", "example.com", None));
        assert!(no_proxy_entry_matches("example.com", "api.example.com", None));
        assert!(no_proxy_entry_matches(".example.com", "example.com", None));
        assert!(no_proxy_entry_matches("*.example.com", "api.example.com", None));
        assert!(!no_proxy_entry_matches("example.com", "notexample.com", None));
        // CIDR.
        assert!(no_proxy_entry_matches("10.0.0.0/8", "10.1.2.3", None));
        assert!(!no_proxy_entry_matches("10.0.0.0/8", "11.1.2.3", None));
        // IP literal.
        assert!(no_proxy_entry_matches("127.0.0.1", "127.0.0.1", None));
        // port mismatch.
        assert!(!no_proxy_entry_matches("example.com:443", "example.com", Some(80)));
        assert!(no_proxy_entry_matches("example.com:443", "example.com", Some(443)));
    }

    #[test]
    fn host_excluded_by_no_proxy_explicit() {
        assert!(is_host_excluded_by_no_proxy("api.foo.com", Some("*.foo.com")));
        assert!(is_host_excluded_by_no_proxy("foo.com", Some(".foo.com")));
        assert!(is_host_excluded_by_no_proxy("anything", Some("*")));
        assert!(!is_host_excluded_by_no_proxy("bar.com", Some("foo.com")));
        assert!(!is_host_excluded_by_no_proxy("bar.com", Some("")));
    }

    #[test]
    fn network_accessible_loopback() {
        assert!(!is_network_accessible("127.0.0.1"));
        assert!(!is_network_accessible("::1"));
        assert!(is_network_accessible("8.8.8.8"));
    }

    #[test]
    fn safe_url_strips_credentials_and_query() {
        let out = safe_url_for_log("https://user:pass@example.com/a/b/c.png?token=secret", 80);
        assert!(!out.contains("secret"));
        assert!(!out.contains("user"));
        assert!(out.starts_with("https://example.com"));
        assert!(out.contains("c.png"));
    }

    #[test]
    fn looks_like_image_magic_bytes() {
        assert!(looks_like_image(b"\x89PNG\r\n\x1a\n\x00\x00"));
        assert!(looks_like_image(b"\xff\xd8\xff\xe0junk"));
        assert!(looks_like_image(b"GIF89a..."));
        assert!(looks_like_image(b"BMxx"));
        let mut webp = b"RIFF1234WEBP".to_vec();
        webp.extend_from_slice(b"more");
        assert!(looks_like_image(&webp));
        assert!(!looks_like_image(b"<htm"));
        assert!(!looks_like_image(b"ab"));
    }

    #[test]
    fn command_parsing() {
        let mut ev = MessageEvent {
            text: "/new@mybot some args".into(),
            ..Default::default()
        };
        assert!(ev.is_command());
        assert_eq!(ev.get_command().as_deref(), Some("new"));
        assert_eq!(ev.get_command_args(), "some args");

        ev.text = "/path/to/file".into();
        assert_eq!(ev.get_command(), None);

        ev.text = "hello".into();
        assert!(!ev.is_command());
        assert_eq!(ev.get_command(), None);
        assert_eq!(ev.get_command_args(), "hello");

        // iOS dash autocorrect.
        ev.text = "/foo \u{2014}\u{2014}flag".into();
        assert_eq!(ev.get_command_args(), "--flag");
        ev.text = "/foo \u{2013}f".into();
        assert_eq!(ev.get_command_args(), "-f");
    }

    #[test]
    fn coerce_restart_phrases() {
        let mut ev = MessageEvent {
            text: "please restart the gateway".into(),
            ..Default::default()
        };
        ev.source.chat_type = "dm".into();
        coerce_plaintext_gateway_command(&mut ev);
        assert_eq!(ev.text, "/restart");

        // group chat: untouched.
        let mut ev2 = MessageEvent {
            text: "restart hermes".into(),
            ..Default::default()
        };
        ev2.source.chat_type = "group".into();
        coerce_plaintext_gateway_command(&mut ev2);
        assert_eq!(ev2.text, "restart hermes");
    }

    #[test]
    fn merge_caption_dedup() {
        assert_eq!(merge_caption(None, "x"), "x");
        assert_eq!(merge_caption(Some(""), "x"), "x");
        assert_eq!(merge_caption(Some("a"), "b"), "a\n\nb");
        // duplicate line not appended.
        assert_eq!(merge_caption(Some("a\n\nb"), "b"), "a\n\nb");
    }

    #[test]
    fn extract_images_markdown_and_html() {
        let content = "before ![alt](https://x.com/a.png) middle <img src=\"https://y.com/b.jpg\"> after";
        let (images, cleaned) = extract_images(content);
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].0, "https://x.com/a.png");
        assert_eq!(images[0].1, "alt");
        assert_eq!(images[1].0, "https://y.com/b.jpg");
        assert!(!cleaned.contains("a.png"));
        assert!(!cleaned.contains("b.jpg"));
    }

    #[test]
    fn extract_media_voice_tag_and_path() {
        let content = "[[audio_as_voice]]\nMEDIA:/tmp/foo.ogg\ndone";
        let (media, cleaned) = extract_media(content);
        assert_eq!(media.len(), 1);
        assert_eq!(media[0].0, "/tmp/foo.ogg");
        assert!(media[0].1); // voice
        assert!(!cleaned.contains("MEDIA:"));
        assert!(!cleaned.contains("audio_as_voice"));
    }

    #[test]
    fn extract_local_files_skips_code_and_validates() {
        let content = "see /home/u/pic.png and `/home/u/code.png` here";
        let (paths, _cleaned) =
            extract_local_files_with(content, |p| p == "/home/u/pic.png");
        assert_eq!(paths, vec!["/home/u/pic.png".to_string()]);
        // the inline-code path must be skipped.
    }

    #[test]
    fn truncate_short_message_single_chunk() {
        let chunks = truncate_message("hello world", 4096, None);
        assert_eq!(chunks, vec!["hello world".to_string()]);
    }

    #[test]
    fn truncate_long_message_splits_and_indicates() {
        let body = "word ".repeat(200); // 1000 chars
        let chunks = truncate_message(&body, 100, None);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.contains('/')); // indicator like (1/N)
        }
    }

    #[test]
    fn truncate_preserves_code_fence() {
        let mut body = String::from("intro text that is fairly long and goes on. ");
        body.push_str("```python\n");
        body.push_str(&"print('x')\n".repeat(40));
        body.push_str("```\n");
        let chunks = truncate_message(&body, 120, None);
        assert!(chunks.len() > 1);
        // Some chunk should carry a reopened fence.
        assert!(chunks.iter().any(|c| c.contains("```python")));
    }

    #[test]
    fn channel_prompt_resolution() {
        let cfg = serde_json::json!({
            "channel_prompts": {
                "C1": "  hello  ",
                "C2": "   ",
            }
        });
        assert_eq!(
            resolve_channel_prompt(&cfg, "C1", None).as_deref(),
            Some("hello")
        );
        // blank prompt -> None
        assert_eq!(resolve_channel_prompt(&cfg, "C2", None), None);
        // fallback to parent
        assert_eq!(
            resolve_channel_prompt(&cfg, "missing", Some("C1")).as_deref(),
            Some("hello")
        );
    }

    #[test]
    fn channel_skills_resolution() {
        let cfg = serde_json::json!({
            "channel_skill_bindings": [
                {"id": "C1", "skills": ["a", "b", "a"]},
                {"id": "C2", "skill": "solo"},
            ]
        });
        assert_eq!(
            resolve_channel_skills(&cfg, "C1", None),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            resolve_channel_skills(&cfg, "C2", None),
            Some(vec!["solo".to_string()])
        );
        assert_eq!(resolve_channel_skills(&cfg, "missing", None), None);
    }

    #[test]
    fn error_classification() {
        assert!(is_retryable_error(Some("ConnectionReset by peer")));
        assert!(is_retryable_error(Some("network unreachable")));
        assert!(!is_retryable_error(Some("bad request")));
        assert!(!is_retryable_error(None));
        assert!(is_timeout_error(Some("Read timed out")));
        assert!(is_timeout_error(Some("ReadTimeout")));
        assert!(!is_timeout_error(Some("connection reset")));
    }

    #[test]
    fn unwrap_ephemeral_logic() {
        let r = HandlerResponse::Text("hi".into());
        assert_eq!(unwrap_ephemeral(&r, 60, true), (Some("hi".into()), 0));

        let e = HandlerResponse::Ephemeral(EphemeralReply::new("bye", Some(30)));
        assert_eq!(unwrap_ephemeral(&e, 60, true), (Some("bye".into()), 30));
        // delete not supported -> ttl forced to 0
        assert_eq!(unwrap_ephemeral(&e, 60, false), (Some("bye".into()), 0));
        // None ttl -> use default
        let e2 = HandlerResponse::Ephemeral(EphemeralReply::new("def", None));
        assert_eq!(unwrap_ephemeral(&e2, 60, true), (Some("def".into()), 60));

        assert_eq!(unwrap_ephemeral(&HandlerResponse::None, 60, true), (None, 0));
    }

    #[test]
    fn animation_url_detection() {
        assert!(is_animation_url("https://x.com/a.GIF?v=1"));
        assert!(!is_animation_url("https://x.com/a.png"));
    }

    #[test]
    fn human_delay_modes() {
        unsafe {
            std::env::set_var("HERMES_HUMAN_DELAY_MODE", "off");
        }
        assert_eq!(human_delay_range_ms(), None);
        unsafe {
            std::env::set_var("HERMES_HUMAN_DELAY_MODE", "natural");
        }
        assert_eq!(human_delay_range_ms(), Some((800, 2500)));
        unsafe {
            std::env::set_var("HERMES_HUMAN_DELAY_MODE", "custom");
            std::env::set_var("HERMES_HUMAN_DELAY_MIN_MS", "100");
            std::env::set_var("HERMES_HUMAN_DELAY_MAX_MS", "200");
        }
        assert_eq!(human_delay_range_ms(), Some((100, 200)));
        unsafe {
            std::env::remove_var("HERMES_HUMAN_DELAY_MODE");
            std::env::remove_var("HERMES_HUMAN_DELAY_MIN_MS");
            std::env::remove_var("HERMES_HUMAN_DELAY_MAX_MS");
        }
    }

    #[test]
    fn merge_pending_photo_burst() {
        let mut pending: HashMap<String, MessageEvent> = HashMap::new();
        let e1 = MessageEvent {
            message_type: MessageType::Photo,
            media_urls: vec!["a".into()],
            media_types: vec!["photo".into()],
            text: "cap1".into(),
            ..Default::default()
        };
        merge_pending_message_event(&mut pending, "k", e1, false);
        let e2 = MessageEvent {
            message_type: MessageType::Photo,
            media_urls: vec!["b".into()],
            media_types: vec!["photo".into()],
            text: "cap2".into(),
            ..Default::default()
        };
        merge_pending_message_event(&mut pending, "k", e2, false);
        let merged = &pending["k"];
        assert_eq!(merged.media_urls, vec!["a".to_string(), "b".to_string()]);
        assert!(merged.text.contains("cap1"));
        assert!(merged.text.contains("cap2"));
    }

    #[test]
    fn session_state_stale_heal() {
        let mut st = SessionState::new();
        st.active_sessions.insert("s".into(), false);
        st.session_tasks.insert("s".into(), 42);
        // task not done -> not stale.
        assert!(!st.heal_stale_session_lock("s"));
        // mark done -> stale -> healed.
        st.done_tasks.insert(42);
        assert!(st.heal_stale_session_lock("s"));
        assert!(!st.active_sessions.contains_key("s"));
    }

    #[test]
    fn auto_tts_decision() {
        let mut st = SessionState::new();
        assert!(!st.should_auto_tts_for_chat("c"));
        st.auto_tts_default = true;
        assert!(st.should_auto_tts_for_chat("c"));
        st.auto_tts_disabled_chats.insert("c".into());
        assert!(!st.should_auto_tts_for_chat("c"));
        st.auto_tts_enabled_chats.insert("c".into());
        assert!(st.should_auto_tts_for_chat("c")); // enabled wins
    }

    #[test]
    fn proxy_kwargs_classification() {
        assert_eq!(proxy_kwargs_for_bot(None), ProxyKwargs::None);
        assert_eq!(
            proxy_kwargs_for_bot(Some("http://p:1")),
            ProxyKwargs::Http("http://p:1".into())
        );
        assert_eq!(
            proxy_kwargs_for_bot(Some("socks5://p:1")),
            ProxyKwargs::SocksConnector("socks5://p:1".into())
        );
    }
}
