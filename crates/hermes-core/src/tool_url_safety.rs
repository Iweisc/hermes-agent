//! URL safety checks — blocks requests to private/internal network addresses.
//!
//! Native Rust port of `tools/url_safety.py`.
//!
//! Prevents SSRF (Server-Side Request Forgery) where a malicious prompt or
//! skill could trick the agent into fetching internal resources like cloud
//! metadata endpoints (169.254.169.254), localhost services, or private
//! network hosts.
//!
//! The check can be globally disabled via `security.allow_private_urls: true`
//! in config.yaml for environments where DNS resolves external domains to
//! private/benchmark-range IPs (OpenWrt routers, corporate proxies, VPNs that
//! use 198.18.0.0/15 or 100.64.0.0/10). Even when disabled, cloud metadata
//! hostnames (metadata.google.internal, 169.254.169.254) are **always**
//! blocked — those are never legitimate agent targets.
//!
//! Limitations (documented, not fixable at pre-flight level):
//!   - DNS rebinding (TOCTOU): an attacker-controlled DNS server with TTL=0
//!     can return a public IP for the check, then a private IP for the actual
//!     connection. Fixing this requires connection-level validation.
//!   - Redirect-based bypass is mitigated by per-redirect re-validation in the
//!     callers that follow redirects.
//!
//! ## Config integration
//!
//! The Python original reads `security.allow_private_urls` /
//! `browser.allow_private_urls` from `read_raw_config()`. Since that raw config
//! reader is not yet ported, [`is_safe_url`] takes the parsed config as a
//! `serde_yaml::Value` parameter. Pass `None` to behave as if config were
//! unavailable (Python's `except Exception: pass` branch). The env-var override
//! (`HERMES_ALLOW_PRIVATE_URLS`) is honoured exactly as in Python and takes
//! priority over the config.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::sync::Mutex;

use crate::mod_utils::{is_truthy_value, TruthyInput};

/// Hostnames that should always be blocked regardless of IP resolution or any
/// config toggle. These are cloud metadata endpoints that an attacker could use
/// to steal instance credentials.
pub const BLOCKED_HOSTNAMES: [&str; 2] = ["metadata.google.internal", "metadata.goog"];

/// Exact HTTPS hostnames allowed to resolve to private/benchmark-space IPs.
/// Intentionally narrow: QQ media downloads can legitimately resolve to
/// 198.18.0.0/15 behind local proxy/benchmark infrastructure.
pub const TRUSTED_PRIVATE_IP_HOSTS: [&str; 1] = ["multimedia.nt.qq.com.cn"];

/// IPs that should always be blocked regardless of the `allow_private_urls`
/// toggle. Cloud metadata / credential endpoints — the #1 SSRF target.
fn always_blocked_ips() -> [IpAddr; 5] {
    [
        IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254)), // AWS/GCP/Azure/DO/Oracle metadata
        IpAddr::V4(Ipv4Addr::new(169, 254, 170, 2)),   // AWS ECS task metadata (task IAM creds)
        IpAddr::V4(Ipv4Addr::new(169, 254, 169, 253)), // Azure IMDS wire server
        IpAddr::V6(Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0254)), // AWS metadata (IPv6)
        IpAddr::V4(Ipv4Addr::new(100, 100, 100, 200)), // Alibaba Cloud metadata
    ]
}

/// Process-wide cache for the resolved `allow_private_urls` toggle. Mirrors the
/// Python module-level globals `_allow_private_resolved` / `_cached_allow_private`.
static ALLOW_PRIVATE_CACHE: Mutex<Option<bool>> = Mutex::new(None);

// ---------------------------------------------------------------------------
// IP classification helpers
// ---------------------------------------------------------------------------

/// True if `ip` falls inside `169.254.0.0/16` (the entire IPv4 link-local
/// range — `_ALWAYS_BLOCKED_NETWORKS` in Python).
fn in_always_blocked_networks(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.octets()[0] == 169 && v4.octets()[1] == 254,
        IpAddr::V6(_) => false,
    }
}

/// True if `ip` is inside `100.64.0.0/10` (CGNAT / Shared Address Space,
/// RFC 6598). Python's `ipaddress.is_private` returns `False` for this range,
/// so it must be checked explicitly.
fn in_cgnat_network(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            // 100.64.0.0/10 => first octet 100, second octet 64..=127
            o[0] == 100 && (64..=127).contains(&o[1])
        }
        IpAddr::V6(_) => false,
    }
}

/// IPv4 private ranges per Python `ipaddress.IPv4Address.is_private`.
fn is_private_v4(v4: &Ipv4Addr) -> bool {
    let o = v4.octets();
    // 10.0.0.0/8
    if o[0] == 10 {
        return true;
    }
    // 172.16.0.0/12
    if o[0] == 172 && (16..=31).contains(&o[1]) {
        return true;
    }
    // 192.168.0.0/16
    if o[0] == 192 && o[1] == 168 {
        return true;
    }
    // 100.64.0.0/10 is NOT considered private by Python's ipaddress (handled
    // separately via CGNAT check), so it is intentionally excluded here.
    // 0.0.0.0/8 ("this network") is treated as private by Python.
    if o[0] == 0 {
        return true;
    }
    // 192.0.0.0/29, 192.0.0.170/31, 192.0.0.171/32 (IETF protocol assignments)
    if o[0] == 192 && o[1] == 0 && o[2] == 0 {
        return true;
    }
    // 198.18.0.0/15 (benchmarking)
    if o[0] == 198 && (o[1] == 18 || o[1] == 19) {
        return true;
    }
    // 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24 (documentation / TEST-NET)
    if o[0] == 192 && o[1] == 0 && o[2] == 2 {
        return true;
    }
    if o[0] == 198 && o[1] == 51 && o[2] == 100 {
        return true;
    }
    if o[0] == 203 && o[1] == 0 && o[2] == 113 {
        return true;
    }
    // 240.0.0.0/4 (reserved) is reported by is_reserved, not is_private.
    false
}

/// IPv4 reserved range per Python `ipaddress.IPv4Address.is_reserved`:
/// `240.0.0.0/4` (excluding the broadcast address which is is_private elsewhere).
fn is_reserved_v4(v4: &Ipv4Addr) -> bool {
    v4.octets()[0] >= 240
}

/// True if `ip` should be blocked for SSRF protection. Faithful port of
/// `_is_blocked_ip`: private, loopback, link-local, reserved, multicast, or
/// unspecified — plus the CGNAT range not covered by `is_private`.
pub fn is_blocked_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            if is_private_v4(v4)
                || v4.is_loopback()
                || v4.is_link_local()
                || is_reserved_v4(v4)
                || v4.is_multicast()
                || v4.is_unspecified()
            {
                return true;
            }
            in_cgnat_network(ip)
        }
        IpAddr::V6(v6) => {
            if is_private_v6(v6)
                || v6.is_loopback()
                || is_link_local_v6(v6)
                || is_reserved_v6(v6)
                || v6.is_multicast()
                || v6.is_unspecified()
            {
                return true;
            }
            // CGNAT is IPv4-only.
            false
        }
    }
}

/// IPv6 private per Python `ipaddress.IPv6Address.is_private`: unique-local
/// `fc00::/7` (covers `fd00::/8`), plus mapped/embedded private addresses.
fn is_private_v6(v6: &Ipv6Addr) -> bool {
    let seg = v6.segments();
    // fc00::/7 (unique local addresses)
    if (seg[0] & 0xfe00) == 0xfc00 {
        return true;
    }
    // ::ffff:0:0/96 IPv4-mapped — defer to the embedded IPv4 classification.
    if let Some(v4) = v6.to_ipv4_mapped() {
        return is_private_v4(&v4) || v4.is_loopback() || v4.is_link_local();
    }
    false
}

/// IPv6 link-local `fe80::/10` (Python `is_link_local`).
fn is_link_local_v6(v6: &Ipv6Addr) -> bool {
    (v6.segments()[0] & 0xffc0) == 0xfe80
}

/// IPv6 reserved per Python `is_reserved` — a broad set of reserved blocks. We
/// approximate with the commonly-relevant reserved prefixes; combined with the
/// other checks this keeps blocking behaviour conservative (fail-closed).
fn is_reserved_v6(v6: &Ipv6Addr) -> bool {
    let seg = v6.segments();
    // ::/8 reserved (includes loopback/unspecified, already caught) and other
    // 0000::/8 space treated as reserved by Python.
    if (seg[0] & 0xff00) == 0x0000 {
        // Exclude IPv4-mapped/compatible which are handled by is_private_v6.
        if v6.to_ipv4_mapped().is_none() && v6.to_ipv4().is_none() {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Global toggle
// ---------------------------------------------------------------------------

/// Reset the cached toggle — only for tests. Mirrors `_reset_allow_private_cache`.
pub fn reset_allow_private_cache() {
    let mut guard = ALLOW_PRIVATE_CACHE.lock().unwrap();
    *guard = None;
}

/// Return `true` when the user has opted out of private-IP blocking.
///
/// Checks (in priority order):
/// 1. `HERMES_ALLOW_PRIVATE_URLS` env var (`true`/`1`/`yes`)
/// 2. `security.allow_private_urls` in `config`
/// 3. `browser.allow_private_urls` in `config` (legacy / backward compat)
///
/// `config` is the parsed raw config (as the Python `read_raw_config()` would
/// return), or `None` when unavailable (Python's `except Exception` branch).
///
/// The result is cached for the process lifetime, exactly like the Python
/// module-level cache.
pub fn global_allow_private_urls(config: Option<&serde_yaml::Value>) -> bool {
    let mut guard = ALLOW_PRIVATE_CACHE.lock().unwrap();
    if let Some(cached) = *guard {
        return cached;
    }

    // Default-safe: blocking enabled.
    let mut result = false;

    // 1. Env var override (highest priority).
    let env_val = std::env::var("HERMES_ALLOW_PRIVATE_URLS")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    if env_val == "true" || env_val == "1" || env_val == "yes" {
        *guard = Some(true);
        return true;
    }
    if env_val == "false" || env_val == "0" || env_val == "no" {
        // Explicit false — don't fall through to config.
        *guard = Some(false);
        return false;
    }

    // 2. Config file.
    if let Some(cfg) = config {
        if let Some(map) = cfg.as_mapping() {
            // security.allow_private_urls (preferred)
            if let Some(sec) = map.get("security").and_then(|v| v.as_mapping()) {
                let val = yaml_to_truthy(sec.get("allow_private_urls"));
                if is_truthy_value(&val, false) {
                    result = true;
                }
            }
            // browser.allow_private_urls (legacy fallback)
            if !result {
                if let Some(browser) = map.get("browser").and_then(|v| v.as_mapping()) {
                    let val = yaml_to_truthy(browser.get("allow_private_urls"));
                    if is_truthy_value(&val, false) {
                        result = true;
                    }
                }
            }
        }
    }

    *guard = Some(result);
    result
}

/// Convert an optional YAML value into a [`TruthyInput`], matching how Python's
/// `is_truthy_value` interprets `None`/`bool`/`str`/other objects.
fn yaml_to_truthy(value: Option<&serde_yaml::Value>) -> TruthyInput {
    match value {
        None | Some(serde_yaml::Value::Null) => TruthyInput::None,
        Some(serde_yaml::Value::Bool(b)) => TruthyInput::Bool(*b),
        Some(serde_yaml::Value::String(s)) => TruthyInput::Str(s.clone()),
        // Any other YAML scalar/collection — Python coerces via bool(value):
        // empty containers/zero are falsy, everything else truthy.
        Some(other) => {
            let truthy = match other {
                serde_yaml::Value::Number(n) => {
                    n.as_f64().map(|f| f != 0.0).unwrap_or(true)
                }
                serde_yaml::Value::Sequence(s) => !s.is_empty(),
                serde_yaml::Value::Mapping(m) => !m.is_empty(),
                _ => true,
            };
            TruthyInput::Other(truthy)
        }
    }
}

/// True when a trusted HTTPS hostname may bypass IP-class blocking.
/// Faithful port of `_allows_private_ip_resolution`.
pub fn allows_private_ip_resolution(hostname: &str, scheme: &str) -> bool {
    scheme == "https" && TRUSTED_PRIVATE_IP_HOSTS.contains(&hostname)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Return `true` if the URL target is not a private/internal address.
///
/// Resolves the hostname to an IP and checks against private ranges. Fails
/// closed: DNS errors and unexpected errors block the request.
///
/// When `security.allow_private_urls` is enabled (or `HERMES_ALLOW_PRIVATE_URLS=true`),
/// private-IP blocking is skipped. Cloud metadata endpoints remain blocked.
///
/// `config` is the raw parsed config, or `None` if unavailable.
pub fn is_safe_url(url: &str, config: Option<&serde_yaml::Value>) -> bool {
    is_safe_url_with_resolver(url, config, &default_resolver)
}

/// Resolve a hostname to a list of IP addresses. Mirrors Python's
/// `socket.getaddrinfo(hostname, None, AF_UNSPEC, SOCK_STREAM)`. Returns `Err`
/// on DNS failure so the caller can fail closed.
fn default_resolver(hostname: &str) -> Result<Vec<IpAddr>, ()> {
    // Port 0 is required by ToSocketAddrs but unused for IP extraction.
    match (hostname, 0u16).to_socket_addrs() {
        Ok(iter) => Ok(iter.map(|sa| sa.ip()).collect()),
        Err(_) => Err(()),
    }
}

/// Core implementation with an injectable resolver (for tests).
pub fn is_safe_url_with_resolver(
    url: &str,
    config: Option<&serde_yaml::Value>,
    resolver: &dyn Fn(&str) -> Result<Vec<IpAddr>, ()>,
) -> bool {
    // Parse the URL. Any failure => fail closed (Python's outer except).
    let parsed = match url::Url::parse(url) {
        Ok(p) => p,
        Err(_) => return false,
    };

    let hostname = parsed
        .host_str()
        .unwrap_or("")
        .trim()
        .to_lowercase()
        .trim_end_matches('.')
        .to_string();
    let scheme = parsed.scheme().trim().to_lowercase();

    if hostname.is_empty() {
        return false;
    }

    // Block known internal hostnames — ALWAYS, even with toggle on.
    if BLOCKED_HOSTNAMES.contains(&hostname.as_str()) {
        log::warn!("Blocked request to internal hostname: {hostname}");
        return false;
    }

    // Check the global toggle AFTER blocking metadata hostnames.
    let allow_all_private = global_allow_private_urls(config);
    let allow_private_ip = allows_private_ip_resolution(&hostname, &scheme);

    // Resolve and check IPs.
    let addrs = match resolver(&hostname) {
        Ok(a) => a,
        Err(()) => {
            // DNS resolution failed — fail closed.
            log::warn!("Blocked request — DNS resolution failed for: {hostname}");
            return false;
        }
    };

    let blocked_ips = always_blocked_ips();

    for ip in &addrs {
        // Always block cloud metadata IPs and link-local, even with toggle on.
        if blocked_ips.contains(ip) || in_always_blocked_networks(ip) {
            log::warn!("Blocked request to cloud metadata address: {hostname} -> {ip}");
            return false;
        }

        if !allow_all_private && !allow_private_ip && is_blocked_ip(ip) {
            log::warn!("Blocked request to private/internal address: {hostname} -> {ip}");
            return false;
        }
    }

    if allow_all_private {
        log::debug!(
            "Allowing private/internal resolution (security.allow_private_urls=true): {hostname}"
        );
    } else if allow_private_ip {
        log::debug!("Allowing trusted hostname despite private/internal resolution: {hostname}");
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_security_true() -> serde_yaml::Value {
        serde_yaml::from_str("security:\n  allow_private_urls: true\n").unwrap()
    }

    fn cfg_browser_true() -> serde_yaml::Value {
        serde_yaml::from_str("browser:\n  allow_private_urls: true\n").unwrap()
    }

    fn no_resolve(_h: &str) -> Result<Vec<IpAddr>, ()> {
        Err(())
    }

    fn resolve_to(ips: Vec<IpAddr>) -> impl Fn(&str) -> Result<Vec<IpAddr>, ()> {
        move |_h: &str| Ok(ips.clone())
    }

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    fn clear_env() {
        unsafe {
            std::env::remove_var("HERMES_ALLOW_PRIVATE_URLS");
        }
        reset_allow_private_cache();
    }

    #[test]
    fn blocks_metadata_hostname_always() {
        clear_env();
        let r = resolve_to(vec![ip("8.8.8.8")]);
        assert!(!is_safe_url_with_resolver(
            "http://metadata.google.internal/",
            None,
            &r
        ));
        assert!(!is_safe_url_with_resolver(
            "https://metadata.goog/computeMetadata/",
            None,
            &r
        ));
    }

    #[test]
    fn metadata_hostname_blocked_even_with_toggle() {
        clear_env();
        let cfg = cfg_security_true();
        let r = resolve_to(vec![ip("8.8.8.8")]);
        assert!(!is_safe_url_with_resolver(
            "https://metadata.google.internal/",
            Some(&cfg),
            &r
        ));
        clear_env();
    }

    #[test]
    fn blocks_metadata_ip_always() {
        clear_env();
        let cfg = cfg_security_true();
        let r = resolve_to(vec![ip("169.254.169.254")]);
        assert!(!is_safe_url_with_resolver(
            "http://example.com/",
            Some(&cfg),
            &r
        ));
        clear_env();
    }

    #[test]
    fn blocks_ecs_and_alibaba_metadata() {
        clear_env();
        let r1 = resolve_to(vec![ip("169.254.170.2")]);
        assert!(!is_safe_url_with_resolver("http://x.com/", None, &r1));
        let r2 = resolve_to(vec![ip("100.100.100.200")]);
        assert!(!is_safe_url_with_resolver("http://x.com/", None, &r2));
    }

    #[test]
    fn blocks_link_local_range() {
        clear_env();
        let r = resolve_to(vec![ip("169.254.1.5")]);
        assert!(!is_safe_url_with_resolver("http://x.com/", None, &r));
    }

    #[test]
    fn blocks_private_ranges() {
        clear_env();
        for s in ["10.0.0.1", "172.16.5.5", "192.168.1.1", "127.0.0.1", "0.0.0.0"] {
            let r = resolve_to(vec![ip(s)]);
            assert!(
                !is_safe_url_with_resolver("http://x.com/", None, &r),
                "{s} should be blocked"
            );
        }
    }

    #[test]
    fn blocks_cgnat_range() {
        clear_env();
        let r = resolve_to(vec![ip("100.64.0.1")]);
        assert!(!is_safe_url_with_resolver("http://x.com/", None, &r));
        let r2 = resolve_to(vec![ip("100.127.255.255")]);
        assert!(!is_safe_url_with_resolver("http://x.com/", None, &r2));
        // 100.128.0.0 is outside the /10 and public.
        let r3 = resolve_to(vec![ip("100.128.0.1")]);
        assert!(is_safe_url_with_resolver("http://x.com/", None, &r3));
    }

    #[test]
    fn allows_public_ip() {
        clear_env();
        let r = resolve_to(vec![ip("8.8.8.8")]);
        assert!(is_safe_url_with_resolver("https://dns.google/", None, &r));
    }

    #[test]
    fn fails_closed_on_dns_error() {
        clear_env();
        assert!(!is_safe_url_with_resolver(
            "http://nonexistent.invalid/",
            None,
            &no_resolve
        ));
    }

    #[test]
    fn fails_closed_on_bad_url() {
        clear_env();
        let r = resolve_to(vec![ip("8.8.8.8")]);
        assert!(!is_safe_url_with_resolver("not a url", None, &r));
        assert!(!is_safe_url_with_resolver("", None, &r));
    }

    #[test]
    fn toggle_allows_private_via_security() {
        clear_env();
        let cfg = cfg_security_true();
        let r = resolve_to(vec![ip("192.168.1.1")]);
        assert!(is_safe_url_with_resolver(
            "http://internal.example/",
            Some(&cfg),
            &r
        ));
        clear_env();
    }

    #[test]
    fn toggle_allows_private_via_browser_legacy() {
        clear_env();
        let cfg = cfg_browser_true();
        let r = resolve_to(vec![ip("10.1.2.3")]);
        assert!(is_safe_url_with_resolver(
            "http://internal.example/",
            Some(&cfg),
            &r
        ));
        clear_env();
    }

    #[test]
    fn env_var_true_overrides() {
        clear_env();
        unsafe {
            std::env::set_var("HERMES_ALLOW_PRIVATE_URLS", "true");
        }
        reset_allow_private_cache();
        let r = resolve_to(vec![ip("192.168.0.50")]);
        assert!(is_safe_url_with_resolver("http://x.com/", None, &r));
        clear_env();
    }

    #[test]
    fn env_var_false_does_not_fall_through_to_config() {
        clear_env();
        unsafe {
            std::env::set_var("HERMES_ALLOW_PRIVATE_URLS", "false");
        }
        reset_allow_private_cache();
        let cfg = cfg_security_true();
        let r = resolve_to(vec![ip("192.168.0.50")]);
        // Explicit false blocks even though config says allow.
        assert!(!is_safe_url_with_resolver("http://x.com/", Some(&cfg), &r));
        clear_env();
    }

    #[test]
    fn trusted_https_host_bypasses_private_block() {
        clear_env();
        // 198.18.x.x is benchmark/private space.
        let r = resolve_to(vec![ip("198.18.0.1")]);
        assert!(is_safe_url_with_resolver(
            "https://multimedia.nt.qq.com.cn/file",
            None,
            &r
        ));
        clear_env();
    }

    #[test]
    fn trusted_host_only_over_https() {
        clear_env();
        let r = resolve_to(vec![ip("198.18.0.1")]);
        // Over http it is NOT trusted.
        assert!(!is_safe_url_with_resolver(
            "http://multimedia.nt.qq.com.cn/file",
            None,
            &r
        ));
        clear_env();
    }

    #[test]
    fn trusted_host_still_blocks_metadata_ip() {
        clear_env();
        let r = resolve_to(vec![ip("169.254.169.254")]);
        assert!(!is_safe_url_with_resolver(
            "https://multimedia.nt.qq.com.cn/file",
            None,
            &r
        ));
        clear_env();
    }

    #[test]
    fn hostname_trailing_dot_normalised() {
        clear_env();
        let r = resolve_to(vec![ip("8.8.8.8")]);
        // Trailing dot stripped; metadata match still works.
        let r2 = resolve_to(vec![ip("8.8.8.8")]);
        assert!(!is_safe_url_with_resolver(
            "http://metadata.google.internal./",
            None,
            &r2
        ));
        assert!(is_safe_url_with_resolver("https://example.com./", None, &r));
        clear_env();
    }

    #[test]
    fn blocks_ipv6_loopback_and_ula() {
        clear_env();
        let r = resolve_to(vec![ip("::1")]);
        assert!(!is_safe_url_with_resolver("http://x.com/", None, &r));
        let r2 = resolve_to(vec![ip("fd00::1")]);
        assert!(!is_safe_url_with_resolver("http://x.com/", None, &r2));
    }

    #[test]
    fn is_blocked_ip_classification() {
        assert!(is_blocked_ip(&ip("10.0.0.1")));
        assert!(is_blocked_ip(&ip("127.0.0.1")));
        assert!(is_blocked_ip(&ip("169.254.1.1")));
        assert!(is_blocked_ip(&ip("100.64.0.1")));
        assert!(is_blocked_ip(&ip("224.0.0.1"))); // multicast
        assert!(is_blocked_ip(&ip("0.0.0.0"))); // unspecified
        assert!(is_blocked_ip(&ip("240.0.0.1"))); // reserved
        assert!(!is_blocked_ip(&ip("8.8.8.8")));
        assert!(!is_blocked_ip(&ip("1.1.1.1")));
        assert!(is_blocked_ip(&ip("::1")));
        assert!(is_blocked_ip(&ip("fe80::1")));
        assert!(!is_blocked_ip(&ip("2606:4700:4700::1111")));
    }
}
