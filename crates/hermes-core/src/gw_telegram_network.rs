//! Telegram-specific network helpers.
//!
//! Provides a hostname-preserving fallback transport for networks where
//! `api.telegram.org` resolves to an endpoint that is unreachable from the
//! current host. The transport keeps the logical request host and TLS SNI as
//! `api.telegram.org` while retrying the TCP connection against one or more
//! fallback IPv4 addresses.
//!
//! This is the native Rust port of `gateway/platforms/telegram_network.py`.
//!
//! In the Python implementation this is built around an `httpx` async
//! transport that rewrites the request URL host to a fallback IP while
//! preserving the `Host` header and TLS SNI hostname (equivalent to
//! `curl --resolve api.telegram.org:443:<ip>`). With `reqwest::blocking` the
//! same effect is achieved with [`reqwest::blocking::ClientBuilder::resolve`],
//! which pins DNS for a host to a specific socket address while leaving the
//! request URL (and therefore the `Host` header and SNI) unchanged.

use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

/// The logical Telegram Bot API host. Requests are always addressed to this
/// host; only the underlying TCP target is allowed to vary.
pub const TELEGRAM_API_HOST: &str = "api.telegram.org";

/// DoH request timeout. Bounded so `connect()` isn't noticeably delayed.
pub const DOH_TIMEOUT: Duration = Duration::from_secs(4);

/// Last-resort IPs when DoH is also blocked. These are stable Telegram Bot API
/// endpoints in the `149.154.160.0/20` block (same seed used by OpenClaw).
pub const SEED_FALLBACK_IPS: &[&str] = &["149.154.167.220"];

/// A DNS-over-HTTPS provider used to discover Telegram API IPs that may differ
/// from the (potentially unreachable) IP returned by the local system
/// resolver.
#[derive(Debug, Clone)]
pub struct DohProvider {
    /// Endpoint URL queried with a `?name=&type=` query string.
    pub url: &'static str,
    /// Extra request headers (e.g. `Accept: application/dns-json`).
    pub headers: &'static [(&'static str, &'static str)],
}

/// The DoH providers, in query order. Mirrors `_DOH_PROVIDERS` in the Python
/// source (Google then Cloudflare).
pub fn doh_providers() -> Vec<DohProvider> {
    vec![
        DohProvider {
            url: "https://dns.google/resolve",
            headers: &[],
        },
        DohProvider {
            url: "https://cloudflare-dns.com/dns-query",
            headers: &[("Accept", "application/dns-json")],
        },
    ]
}

/// Validate, filter and canonicalise an iterable of candidate fallback IP
/// strings.
///
/// Mirrors `_normalize_fallback_ips`:
/// * whitespace-trimmed empty entries are dropped,
/// * non-parseable addresses are dropped (with a warning),
/// * non-IPv4 addresses are dropped (with a warning),
/// * private / loopback / link-local / unspecified addresses are dropped
///   (with a warning),
/// * surviving addresses are returned in their canonical string form,
///   preserving input order. (Deduplication is intentionally NOT performed
///   here, matching the Python behaviour where dedup happens separately.)
pub fn normalize_fallback_ips<I, S>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut normalized: Vec<String> = Vec::new();
    for value in values {
        let raw = value.as_ref().trim();
        if raw.is_empty() {
            continue;
        }
        let addr: IpAddr = match raw.parse() {
            Ok(a) => a,
            Err(_) => {
                log::warn!("Ignoring invalid Telegram fallback IP: {:?}", raw);
                continue;
            }
        };
        let v4 = match addr {
            IpAddr::V4(v4) => v4,
            IpAddr::V6(_) => {
                log::warn!("Ignoring non-IPv4 Telegram fallback IP: {}", raw);
                continue;
            }
        };
        if is_internal_v4(&v4) {
            log::warn!("Ignoring private/internal Telegram fallback IP: {}", raw);
            continue;
        }
        normalized.push(v4.to_string());
    }
    normalized
}

/// Returns `true` for IPv4 addresses Python's `ipaddress` treats as
/// private / loopback / link-local / unspecified (and which the Python source
/// rejects).
fn is_internal_v4(addr: &Ipv4Addr) -> bool {
    addr.is_private()
        || addr.is_loopback()
        || addr.is_link_local()
        || addr.is_unspecified()
        // Python's `is_private` also covers the wider RFC1918/special ranges;
        // include the additional shared/CGNAT and benchmarking ranges that
        // `ipaddress.ip_address(...).is_private` flags.
        || is_shared_or_special_v4(addr)
}

/// Additional ranges Python's `ipaddress.is_private` flags beyond the std
/// helpers above: 100.64.0.0/10 (CGNAT/shared), 192.0.0.0/24, 198.18.0.0/15
/// (benchmarking) and 192.0.2.0/24-ish documentation ranges are NOT private in
/// Python, so only the shared range is added here.
fn is_shared_or_special_v4(addr: &Ipv4Addr) -> bool {
    let o = addr.octets();
    // 100.64.0.0/10 — RFC 6598 shared address space (private in Python 3.9+).
    o[0] == 100 && (o[1] & 0xC0) == 0x40
}

/// Parse a comma-separated environment-variable value into validated fallback
/// IPs. Mirrors `parse_fallback_ip_env`.
pub fn parse_fallback_ip_env(value: Option<&str>) -> Vec<String> {
    match value {
        None => Vec::new(),
        Some(v) if v.is_empty() => Vec::new(),
        Some(v) => {
            let parts: Vec<&str> = v.split(',').map(|p| p.trim()).collect();
            normalize_fallback_ips(parts)
        }
    }
}

/// Deduplicate a list of IP strings preserving first-seen order.
/// Equivalent to Python's `dict.fromkeys(...)` ordering trick.
pub fn dedup_preserve_order<I, S>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for v in values {
        let s = v.into();
        if seen.insert(s.clone()) {
            out.push(s);
        }
    }
    out
}

/// Resolve the IPv4 addresses the OS resolver returns for `api.telegram.org`.
///
/// Mirrors `_resolve_system_dns`; returns an empty set on any failure.
pub fn resolve_system_dns() -> std::collections::HashSet<String> {
    use std::net::ToSocketAddrs;
    let mut out = std::collections::HashSet::new();
    let target = format!("{}:443", TELEGRAM_API_HOST);
    match target.to_socket_addrs() {
        Ok(addrs) => {
            for a in addrs {
                if let IpAddr::V4(v4) = a.ip() {
                    out.insert(v4.to_string());
                }
            }
        }
        Err(_) => return std::collections::HashSet::new(),
    }
    out
}

/// Build the query string for a DoH provider lookup of the Telegram API host's
/// A records.
fn doh_query_params() -> [(&'static str, &'static str); 2] {
    [("name", TELEGRAM_API_HOST), ("type", "A")]
}

/// Parse a DoH JSON response body into A-record IP strings.
///
/// Split out from the network call so it can be unit-tested directly. Mirrors
/// the answer-filtering logic in `_query_doh_provider`.
pub fn parse_doh_answers(data: &serde_json::Value) -> Vec<String> {
    let mut ips: Vec<String> = Vec::new();
    let answers = match data.get("Answer").and_then(|a| a.as_array()) {
        Some(a) => a,
        None => return ips,
    };
    for answer in answers {
        // Only A records (type == 1).
        if answer.get("type").and_then(|t| t.as_i64()) != Some(1) {
            continue;
        }
        let raw = answer
            .get("data")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .trim();
        if raw.parse::<IpAddr>().is_ok() {
            ips.push(raw.to_string());
        }
    }
    ips
}

/// Query a single DoH provider over HTTPS and return its A-record IPs.
///
/// On any failure an empty vector is returned (matching the Python source,
/// which logs at debug level and swallows the error).
pub fn query_doh_provider(client: &reqwest::blocking::Client, provider: &DohProvider) -> Vec<String> {
    let result = (|| -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut req = client.get(provider.url).query(&doh_query_params());
        for (k, v) in provider.headers {
            req = req.header(*k, *v);
        }
        let resp = req.send()?;
        let resp = resp.error_for_status()?;
        let data: serde_json::Value = resp.json()?;
        Ok(parse_doh_answers(&data))
    })();
    match result {
        Ok(ips) => ips,
        Err(exc) => {
            log::debug!("DoH query to {} failed: {}", provider.url, exc);
            Vec::new()
        }
    }
}

/// Auto-discover Telegram API IPs via DNS-over-HTTPS.
///
/// Mirrors `discover_fallback_ips`: resolves `api.telegram.org` through Google
/// and Cloudflare DoH and returns all unique, validated A records. IPs that
/// match the local system resolver are kept rather than excluded (#14520).
/// Falls back to [`SEED_FALLBACK_IPS`] only when DoH yields no usable answers.
///
/// Unlike the Python version, which runs the DoH queries and the system-DNS
/// lookup concurrently, this runs them sequentially (the calls are individually
/// bounded by [`DOH_TIMEOUT`]). The system-DNS result is only used for logging,
/// matching the Python behaviour.
pub fn discover_fallback_ips() -> Vec<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(DOH_TIMEOUT)
        .build();
    let client = match client {
        Ok(c) => c,
        Err(exc) => {
            log::debug!("Failed to build DoH client: {}", exc);
            return SEED_FALLBACK_IPS.iter().map(|s| s.to_string()).collect();
        }
    };

    let system_ips = resolve_system_dns();

    let mut doh_ips: Vec<String> = Vec::new();
    for provider in doh_providers() {
        doh_ips.extend(query_doh_provider(&client, &provider));
    }

    // Deduplicate preserving order, then validate through normalisation.
    let candidates = dedup_preserve_order(doh_ips);
    let validated = normalize_fallback_ips(candidates);

    if !validated.is_empty() {
        log::debug!(
            "Discovered Telegram fallback IPs via DoH: {}",
            validated.join(", ")
        );
        return validated;
    }

    let system_str = if system_ips.is_empty() {
        "unknown".to_string()
    } else {
        let mut v: Vec<String> = system_ips.into_iter().collect();
        v.sort();
        v.join(", ")
    };
    log::info!(
        "DoH discovery yielded no usable IPs (system DNS: {}); using seed fallback IPs {}",
        system_str,
        SEED_FALLBACK_IPS.join(", ")
    );
    SEED_FALLBACK_IPS.iter().map(|s| s.to_string()).collect()
}

/// A hostname-preserving fallback transport for the Telegram Bot API.
///
/// This is the `reqwest::blocking` analogue of Python's
/// `TelegramFallbackTransport`. Requests continue to target
/// `https://api.telegram.org/...` logically, but the underlying TCP connection
/// can be pinned to a known-reachable fallback IP while keeping the `Host`
/// header and TLS SNI as `api.telegram.org`.
///
/// On connect failures the request is retried against each fallback IP in
/// order. Once a fallback IP succeeds it becomes "sticky" and is tried first on
/// subsequent requests.
pub struct TelegramFallbackTransport {
    fallback_ips: Vec<String>,
    /// A client whose DNS is unpinned (uses the system resolver). Used for the
    /// primary path.
    primary: reqwest::blocking::Client,
    /// Per-fallback-IP clients with DNS pinned to that IP via
    /// [`ClientBuilder::resolve`].
    fallbacks: std::collections::HashMap<String, reqwest::blocking::Client>,
    /// The fallback IP that last succeeded, if any.
    sticky_ip: std::sync::Mutex<Option<String>>,
}

impl TelegramFallbackTransport {
    /// Build a transport over the given fallback IPs (which are normalised and
    /// deduplicated). `request_timeout` bounds every request attempt.
    pub fn new<I, S>(fallback_ips: I, request_timeout: Option<Duration>) -> reqwest::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let normalized = normalize_fallback_ips(fallback_ips);
        let fallback_ips = dedup_preserve_order(normalized);

        let mut primary_builder = reqwest::blocking::Client::builder();
        if let Some(t) = request_timeout {
            primary_builder = primary_builder.timeout(t);
        }
        let primary = primary_builder.build()?;

        let mut fallbacks = std::collections::HashMap::new();
        for ip in &fallback_ips {
            let mut b = reqwest::blocking::Client::builder();
            if let Some(t) = request_timeout {
                b = b.timeout(t);
            }
            // Pin api.telegram.org:443 -> <ip>:443 while preserving Host/SNI.
            if let Ok(parsed) = ip.parse::<IpAddr>() {
                let socket = std::net::SocketAddr::new(parsed, 443);
                b = b.resolve(TELEGRAM_API_HOST, socket);
            }
            fallbacks.insert(ip.clone(), b.build()?);
        }

        Ok(Self {
            fallback_ips,
            primary,
            fallbacks,
            sticky_ip: std::sync::Mutex::new(None),
        })
    }

    /// The validated fallback IPs this transport will try, in order.
    pub fn fallback_ips(&self) -> &[String] {
        &self.fallback_ips
    }

    /// The currently sticky fallback IP, if one has succeeded.
    pub fn sticky_ip(&self) -> Option<String> {
        self.sticky_ip.lock().unwrap().clone()
    }

    /// Compute the order in which connection targets should be attempted.
    ///
    /// `None` represents the primary (system-resolver) path. Mirrors the
    /// `attempt_order` construction in `handle_async_request`: the sticky IP
    /// (or the primary path when there is no sticky IP) is tried first,
    /// followed by every fallback IP not equal to the sticky IP.
    pub fn attempt_order(&self) -> Vec<Option<String>> {
        let sticky = self.sticky_ip.lock().unwrap().clone();
        let mut order: Vec<Option<String>> = Vec::new();
        match &sticky {
            Some(ip) => order.push(Some(ip.clone())),
            None => order.push(None),
        }
        for ip in &self.fallback_ips {
            if Some(ip) != sticky.as_ref() {
                order.push(Some(ip.clone()));
            }
        }
        order
    }

    /// Send a request, trying the primary path and then fallback IPs until one
    /// succeeds. `build` is invoked once per attempt to produce a fresh
    /// `RequestBuilder` on the chosen client (a builder cannot be cloned across
    /// attempts).
    ///
    /// If the URL host is not `api.telegram.org`, or there are no fallback IPs,
    /// the primary path is used directly (mirroring the early-return in the
    /// Python source).
    pub fn send<F>(&self, url: &str, build: F) -> reqwest::Result<reqwest::blocking::Response>
    where
        F: Fn(&reqwest::blocking::Client) -> reqwest::blocking::RequestBuilder,
    {
        let is_telegram = url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(|h| h == TELEGRAM_API_HOST))
            .unwrap_or(false);

        if !is_telegram || self.fallback_ips.is_empty() {
            return build(&self.primary).send();
        }

        let mut last_error: Option<reqwest::Error> = None;
        for target in self.attempt_order() {
            let client = match &target {
                None => &self.primary,
                Some(ip) => match self.fallbacks.get(ip) {
                    Some(c) => c,
                    None => continue,
                },
            };
            match build(client).send() {
                Ok(response) => {
                    if let Some(ip) = &target {
                        let mut guard = self.sticky_ip.lock().unwrap();
                        if guard.as_deref() != Some(ip.as_str()) {
                            *guard = Some(ip.clone());
                            log::warn!(
                                "[Telegram] Primary api.telegram.org path unreachable; using sticky fallback IP {}",
                                ip
                            );
                        }
                    }
                    return Ok(response);
                }
                Err(exc) => {
                    let retryable = is_retryable_connect_error(&exc);
                    match &target {
                        None => {
                            log::warn!(
                                "[Telegram] Primary api.telegram.org connection failed ({}); trying fallback IPs {}",
                                exc,
                                self.fallback_ips.join(", ")
                            );
                        }
                        Some(ip) => {
                            log::warn!("[Telegram] Fallback IP {} failed: {}", ip, exc);
                        }
                    }
                    last_error = Some(exc);
                    if !retryable {
                        // Non-retryable error: surface it immediately.
                        return Err(last_error.unwrap());
                    }
                    continue;
                }
            }
        }

        // All attempts exhausted; surface the last error.
        Err(last_error.expect("attempt_order always contains at least one entry"))
    }
}

/// Whether a request error is a connect-phase failure worth retrying against a
/// fallback IP.
///
/// Mirrors `_is_retryable_connect_error`, which retries on `httpx.ConnectError`
/// / `httpx.ConnectTimeout`. With `reqwest` we treat connect errors and
/// connect timeouts as retryable; once bytes have been exchanged (a response
/// status / body error) the failure is not retried.
pub fn is_retryable_connect_error(err: &reqwest::Error) -> bool {
    err.is_connect() || (err.is_timeout() && !err.is_body())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_keeps_valid_public_ipv4() {
        let out = normalize_fallback_ips(["149.154.167.220", "1.1.1.1"]);
        assert_eq!(out, vec!["149.154.167.220", "1.1.1.1"]);
    }

    #[test]
    fn normalize_trims_and_drops_empty() {
        let out = normalize_fallback_ips(["  8.8.8.8  ", "", "   "]);
        assert_eq!(out, vec!["8.8.8.8"]);
    }

    #[test]
    fn normalize_drops_invalid() {
        let out = normalize_fallback_ips(["not-an-ip", "999.999.0.1", "8.8.4.4"]);
        assert_eq!(out, vec!["8.8.4.4"]);
    }

    #[test]
    fn normalize_drops_ipv6() {
        let out = normalize_fallback_ips(["2001:4860:4860::8888", "8.8.8.8"]);
        assert_eq!(out, vec!["8.8.8.8"]);
    }

    #[test]
    fn normalize_drops_private_loopback_linklocal_unspecified() {
        let out = normalize_fallback_ips([
            "10.0.0.1",      // private
            "192.168.1.1",   // private
            "172.16.5.5",    // private
            "127.0.0.1",     // loopback
            "169.254.1.1",   // link-local
            "0.0.0.0",       // unspecified
            "100.64.0.1",    // CGNAT/shared
            "203.0.113.5",   // documentation but NOT private -> kept
        ]);
        assert_eq!(out, vec!["203.0.113.5"]);
    }

    #[test]
    fn parse_env_none_and_empty() {
        assert!(parse_fallback_ip_env(None).is_empty());
        assert!(parse_fallback_ip_env(Some("")).is_empty());
    }

    #[test]
    fn parse_env_comma_separated() {
        let out = parse_fallback_ip_env(Some("149.154.167.220, 8.8.8.8 ,bad,10.0.0.1"));
        assert_eq!(out, vec!["149.154.167.220", "8.8.8.8"]);
    }

    #[test]
    fn dedup_preserves_first_seen_order() {
        let out = dedup_preserve_order(["a", "b", "a", "c", "b"]);
        assert_eq!(out, vec!["a", "b", "c"]);
    }

    #[test]
    fn parse_doh_answers_filters_a_records() {
        let data = serde_json::json!({
            "Answer": [
                {"type": 1, "data": "149.154.167.220"},
                {"type": 5, "data": "cname.example.com"}, // CNAME, ignored
                {"type": 1, "data": "not-an-ip"},          // bad data, ignored
                {"type": 1, "data": " 8.8.8.8 "},          // trimmed, kept
                {"type": 28, "data": "2001:db8::1"}        // AAAA, ignored
            ]
        });
        let ips = parse_doh_answers(&data);
        assert_eq!(ips, vec!["149.154.167.220", "8.8.8.8"]);
    }

    #[test]
    fn parse_doh_answers_missing_answer_key() {
        let data = serde_json::json!({ "Status": 0 });
        assert!(parse_doh_answers(&data).is_empty());
    }

    #[test]
    fn parse_doh_answers_empty_answer() {
        let data = serde_json::json!({ "Answer": [] });
        assert!(parse_doh_answers(&data).is_empty());
    }

    #[test]
    fn doh_providers_order_and_headers() {
        let providers = doh_providers();
        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0].url, "https://dns.google/resolve");
        assert!(providers[0].headers.is_empty());
        assert_eq!(providers[1].url, "https://cloudflare-dns.com/dns-query");
        assert_eq!(providers[1].headers, &[("Accept", "application/dns-json")]);
    }

    #[test]
    fn doh_query_params_target_telegram_a_record() {
        let p = doh_query_params();
        assert_eq!(p, [("name", "api.telegram.org"), ("type", "A")]);
    }

    #[test]
    fn transport_no_fallbacks_uses_primary_order() {
        let t = TelegramFallbackTransport::new(Vec::<String>::new(), None).unwrap();
        assert!(t.fallback_ips().is_empty());
        // With no fallbacks, attempt order is just the primary path.
        assert_eq!(t.attempt_order(), vec![None]);
    }

    #[test]
    fn transport_attempt_order_no_sticky() {
        let t =
            TelegramFallbackTransport::new(["149.154.167.220", "8.8.8.8"], None).unwrap();
        assert_eq!(
            t.attempt_order(),
            vec![
                None,
                Some("149.154.167.220".to_string()),
                Some("8.8.8.8".to_string())
            ]
        );
    }

    #[test]
    fn transport_attempt_order_with_sticky_promotes_it() {
        let t =
            TelegramFallbackTransport::new(["149.154.167.220", "8.8.8.8"], None).unwrap();
        *t.sticky_ip.lock().unwrap() = Some("8.8.8.8".to_string());
        // Sticky IP first, then remaining fallbacks (sticky excluded).
        assert_eq!(
            t.attempt_order(),
            vec![
                Some("8.8.8.8".to_string()),
                Some("149.154.167.220".to_string())
            ]
        );
    }

    #[test]
    fn transport_dedups_and_validates_input() {
        let t = TelegramFallbackTransport::new(
            ["8.8.8.8", "8.8.8.8", "10.0.0.1", "bad", "1.1.1.1"],
            None,
        )
        .unwrap();
        assert_eq!(t.fallback_ips(), &["8.8.8.8".to_string(), "1.1.1.1".to_string()]);
    }

    #[test]
    fn seed_fallback_is_valid_public_ipv4() {
        let normalized = normalize_fallback_ips(SEED_FALLBACK_IPS.iter().copied());
        assert_eq!(normalized, vec!["149.154.167.220"]);
    }
}
