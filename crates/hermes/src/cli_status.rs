//! Status command for the Hermes CLI.
//!
//! Native Rust port of `hermes_cli/status.py`. Renders the status of all
//! Hermes Agent components to stdout.
//!
//! ## Design
//!
//! The Python original mixes data gathering (talking to the auth store, the
//! gateway runtime, the filesystem) with rendering. Many of those data
//! sources are large, partially-ported subsystems with their own evolving
//! Rust APIs (`auth.rs`, `gateway.rs`, `cli_nous_subscription.rs`, ...). To
//! keep this module self-contained and faithful, the *rendering* is captured
//! here as [`show_status`], driven by a fully-populated [`StatusReport`].
//!
//! The caller (the CLI command dispatcher) is responsible for collecting the
//! live data into a [`StatusReport`] — exactly the values the Python code
//! pulled from `load_config()`, `get_anthropic_key()`,
//! `get_nous_auth_status()`, `get_gateway_runtime_snapshot()`, etc. This keeps
//! the byte-for-byte output contract here while letting the data plumbing
//! evolve independently. Pure helpers ([`check_mark`], [`redact_key`],
//! [`format_iso_timestamp`], [`configured_model_label`]) are exposed for reuse
//! and unit-tested directly.
//!
//! Color and redaction reuse the already-ported helpers
//! (`crate::cli_colors`, `hermes_core::agent_redact`).

use std::collections::BTreeMap;

use chrono::{DateTime, FixedOffset, Local, TimeZone, Utc};

use crate::cli_colors::{color, Colors};

// ---------------------------------------------------------------------------
// Pure helpers (faithful 1:1 ports)
// ---------------------------------------------------------------------------

/// Render a green check or red cross, mirroring Python `check_mark`.
pub fn check_mark(ok: bool) -> String {
    if ok {
        color("\u{2713}", &[Colors::GREEN])
    } else {
        color("\u{2717}", &[Colors::RED])
    }
}

/// Redact an API key for display.
///
/// Preserves the dim `(not set)` placeholder for empty values, matching
/// `hermes config` output. Mirrors `agent.redact.mask_secret` with the
/// default head=4 / tail=4 / floor=12 options (the `hermes-core` port keeps
/// `mask_secret` private, so the small logic is reproduced here).
pub fn redact_key(key: &str) -> String {
    mask_secret(key, 4, 4, 12, "***", &color("(not set)", &[Colors::DIM]))
}

/// Mask a secret for display, preserving `head` and `tail` characters.
///
/// Faithful port of `agent.redact.mask_secret`: empty input returns `empty`;
/// values shorter than `floor` (by char count) return `placeholder`;
/// otherwise `<head>...<tail>`.
fn mask_secret(value: &str, head: usize, tail: usize, floor: usize, placeholder: &str, empty: &str) -> String {
    if value.is_empty() {
        return empty.to_string();
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() < floor {
        return placeholder.to_string();
    }
    let head_str: String = chars.iter().take(head).collect();
    let start = chars.len().saturating_sub(tail);
    let tail_str: String = chars[start..].iter().collect();
    format!("{head_str}...{tail_str}")
}

/// Format an ISO timestamp for status output, converting to the local zone.
///
/// Faithful port of `_format_iso_timestamp`:
/// - empty / whitespace-only input returns `"(unknown)"`;
/// - a trailing `Z` is treated as `+00:00`;
/// - naive timestamps (no offset) are assumed UTC;
/// - on parse failure the original string is returned unchanged;
/// - output format is `%Y-%m-%d %H:%M:%S %Z` in local time.
pub fn format_iso_timestamp(value: Option<&str>) -> String {
    let raw = match value {
        Some(v) => v,
        None => return "(unknown)".to_string(),
    };
    let text = raw.trim();
    if text.is_empty() {
        return "(unknown)".to_string();
    }

    // Normalise a trailing Z to +00:00, matching Python.
    let normalized = if let Some(stripped) = text.strip_suffix('Z') {
        format!("{stripped}+00:00")
    } else {
        text.to_string()
    };

    // Try an offset-aware parse first (handles "+00:00" etc.).
    if let Ok(dt) = DateTime::parse_from_rfc3339(&normalized) {
        return format_local(dt.with_timezone(&Utc));
    }
    // Some ISO variants use a space separator or omit the offset; try a few.
    for fmt in &["%Y-%m-%dT%H:%M:%S%.f%:z", "%Y-%m-%d %H:%M:%S%.f%:z"] {
        if let Ok(dt) = DateTime::parse_from_str(&normalized, fmt) {
            return format_local(dt.with_timezone(&Utc));
        }
    }
    // Naive (no offset) -> assume UTC, mirroring Python's tzinfo=None branch.
    for fmt in &["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(&normalized, fmt) {
            if let Some(dt) = Utc.from_local_datetime(&naive).single() {
                return format_local(dt);
            }
        }
    }

    // Parse failure: return the original (un-normalized) value.
    raw.to_string()
}

fn format_local(dt: DateTime<Utc>) -> String {
    let local: DateTime<Local> = dt.with_timezone(&Local);
    // Python's %Z renders the local tz abbreviation; chrono's %Z on Local
    // yields the offset name. Use the numeric offset as a stable fallback.
    let formatted = local.format("%Y-%m-%d %H:%M:%S").to_string();
    let tz = local.format("%Z").to_string();
    if tz.is_empty() {
        formatted
    } else {
        format!("{formatted} {tz}")
    }
}

/// Convert a millisecond UNIX timestamp into a UTC RFC3339 string,
/// mirroring `datetime.fromtimestamp(ms/1000, tz=utc).isoformat()`.
pub fn ms_to_utc_iso(expires_at_ms: i64) -> String {
    let secs = expires_at_ms.div_euclid(1000);
    let nanos = (expires_at_ms.rem_euclid(1000) * 1_000_000) as u32;
    let dt: DateTime<FixedOffset> = Utc
        .timestamp_opt(secs, nanos)
        .single()
        .unwrap_or_else(Utc::now)
        .with_timezone(&FixedOffset::east_opt(0).unwrap());
    // isoformat() -> e.g. 2026-01-02T03:04:05+00:00
    dt.format("%Y-%m-%dT%H:%M:%S%.f%:z").to_string()
}

/// Return the configured default model from the parsed config value.
///
/// Mirrors `_configured_model_label`: a `model` mapping uses `default` then
/// `name`; a `model` string is used directly; anything else is empty.
pub fn configured_model_label(config: &serde_json::Value) -> String {
    let model = match config.get("model") {
        Some(serde_json::Value::Object(map)) => {
            let default = map.get("default").and_then(|v| v.as_str()).unwrap_or("");
            let name = map.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let picked = if !default.trim().is_empty() {
                default
            } else {
                name
            };
            picked.trim().to_string()
        }
        Some(serde_json::Value::String(s)) => s.trim().to_string(),
        _ => String::new(),
    };
    if model.is_empty() {
        "(not set)".to_string()
    } else {
        model
    }
}

// ---------------------------------------------------------------------------
// Report data model
// ---------------------------------------------------------------------------

/// Status of a single OAuth provider row (Nous / Codex / Qwen / MiniMax).
#[derive(Debug, Default, Clone)]
pub struct OAuthStatus {
    pub logged_in: bool,
    pub error: Option<String>,
    pub portal_base_url: Option<String>,
    pub access_expires_at: Option<String>,
    pub agent_key_expires_at: Option<String>,
    pub has_refresh_token: bool,
    pub auth_store: Option<String>,
    pub last_refresh: Option<String>,
    pub auth_file: Option<String>,
    pub expires_at_ms: Option<i64>,
    pub region: Option<String>,
    pub expires_at: Option<String>,
}

/// A single API-key row: display label and the resolved (possibly empty) value.
#[derive(Debug, Clone)]
pub struct ApiKeyRow {
    pub label: String,
    pub value: String,
}

/// A single API-key provider row (configured / not configured).
#[derive(Debug, Clone)]
pub struct ApiKeyProviderRow {
    pub label: String,
    pub configured: bool,
}

/// A single Nous tool-gateway feature row.
#[derive(Debug, Clone)]
pub struct GatewayFeatureRow {
    pub label: String,
    pub managed_by_nous: bool,
    pub active: bool,
    pub available: bool,
    pub included_by_default: bool,
    pub key: String,
    pub current_provider: Option<String>,
}

/// Nous Tool Gateway section state.
#[derive(Debug, Clone)]
pub struct NousGateway {
    /// `managed_nous_tools_enabled()` — when true the full feature table renders.
    pub managed_enabled: bool,
    pub nous_auth_present: bool,
    pub features: Vec<GatewayFeatureRow>,
}

/// Terminal backend configuration, already resolved from env + config.
#[derive(Debug, Clone, Default)]
pub struct TerminalBackend {
    /// "local", "ssh", "docker", "daytona", "vercel_sandbox", ...
    pub backend: String,
    pub ssh_host: String,
    pub ssh_user: String,
    pub docker_image: String,
    pub daytona_image: String,
    pub vercel_runtime: String,
    pub vercel_persist_enabled: bool,
    pub vercel_sdk_installed: bool,
    pub vercel_auth_ok: bool,
    pub vercel_auth_label: String,
    pub vercel_auth_detail_lines: Vec<String>,
    pub sudo_enabled: bool,
}

/// A messaging-platform row.
#[derive(Debug, Clone)]
pub struct PlatformRow {
    pub label: String,
    pub configured: bool,
    pub home_channel: String,
    /// Plugin rows append " (plugin)" to the status string.
    pub is_plugin: bool,
}

/// Resolved gateway runtime snapshot (mirror of `get_gateway_runtime_snapshot`).
#[derive(Debug, Clone)]
pub struct GatewaySnapshot {
    pub running: bool,
    pub manager: String,
    pub gateway_pids_formatted: Option<String>,
    pub has_process_service_mismatch: bool,
    pub service_installed: bool,
    pub service_running: bool,
    pub has_pids: bool,
}

/// Fallback gateway state when the snapshot could not be obtained.
#[derive(Debug, Clone)]
pub enum GatewaySection {
    Snapshot(GatewaySnapshot),
    /// Snapshot failed; render platform-specific "unknown" lines.
    Unavailable,
}

/// Result of an LM Studio probe (only present when LM Studio is active).
#[derive(Debug, Clone)]
pub struct LmStudioProbe {
    pub ok: bool,
    pub message: String,
}

/// Result of a deep OpenRouter connectivity check.
#[derive(Debug, Clone)]
pub enum OpenRouterCheck {
    Ok,
    HttpError(u16),
    Error(String),
}

/// Deep-check section data.
#[derive(Debug, Clone, Default)]
pub struct DeepChecks {
    pub openrouter: Option<OpenRouterCheck>,
    /// `None` when the socket probe raised an OSError (suppressed in Python).
    pub port_18789_in_use: Option<bool>,
}

/// Fully-resolved data backing one `show_status` render.
#[derive(Debug, Clone)]
pub struct StatusReport {
    pub show_all: bool,
    pub deep: bool,

    // Environment
    pub project_root: String,
    pub python_version: String,
    pub env_file_exists: bool,
    pub configured_model: String,
    pub effective_provider_label: String,

    // API keys (already resolved; Anthropic last).
    pub api_keys: Vec<ApiKeyRow>,
    pub anthropic_value: String,

    // Auth providers
    pub nous: OAuthStatus,
    pub codex: OAuthStatus,
    pub qwen: OAuthStatus,
    pub minimax: OAuthStatus,

    // Nous tool gateway
    pub nous_gateway: NousGateway,

    // API-key providers
    pub api_key_providers: Vec<ApiKeyProviderRow>,
    pub lmstudio: Option<LmStudioProbe>,

    // Terminal
    pub terminal: TerminalBackend,

    // Messaging
    pub platforms: Vec<PlatformRow>,

    // Gateway
    pub gateway: GatewaySection,
    pub is_termux: bool,
    pub platform_is_linux: bool,
    pub platform_is_darwin: bool,

    // Cron / sessions
    pub jobs_active: Option<usize>,
    pub jobs_total: Option<usize>,
    pub jobs_error: bool,
    pub jobs_file_exists: bool,
    pub sessions_count: Option<usize>,
    pub sessions_error: bool,
    pub sessions_file_exists: bool,

    // Deep checks
    pub deep_checks: DeepChecks,
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the full status report to a single string (newline-terminated lines),
/// faithfully mirroring the layout of Python `show_status`.
///
/// Returns the rendered text so callers can either print it or capture it for
/// tests. `show_status` prints it directly.
pub fn render_status(report: &StatusReport) -> String {
    let mut out = StatusWriter::new();

    out.blank();
    out.cyan("\u{250c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}");
    out.cyan("\u{2502}                 \u{2695} Hermes Agent Status                  \u{2502}");
    out.cyan("\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}");

    // Environment ----------------------------------------------------------
    out.blank();
    out.header("\u{25c6} Environment");
    out.line(&format!("  Project:      {}", report.project_root));
    out.line(&format!("  Python:       {}", report.python_version));
    out.line(&format!(
        "  .env file:    {} {}",
        check_mark(report.env_file_exists),
        if report.env_file_exists { "exists" } else { "not found" }
    ));
    out.line(&format!("  Model:        {}", report.configured_model));
    out.line(&format!("  Provider:     {}", report.effective_provider_label));

    // API keys -------------------------------------------------------------
    out.blank();
    out.header("\u{25c6} API Keys");
    for row in &report.api_keys {
        let display = if report.show_all {
            row.value.clone()
        } else {
            redact_key(&row.value)
        };
        out.line(&format!(
            "  {:<12}  {} {}",
            row.label,
            check_mark(!row.value.is_empty()),
            display
        ));
    }
    let anthropic_display = if report.show_all {
        report.anthropic_value.clone()
    } else {
        redact_key(&report.anthropic_value)
    };
    out.line(&format!(
        "  {:<12}  {} {}",
        "Anthropic",
        check_mark(!report.anthropic_value.is_empty()),
        anthropic_display
    ));

    // Auth providers -------------------------------------------------------
    out.blank();
    out.header("\u{25c6} Auth Providers");
    render_nous_auth(&mut out, &report.nous);
    render_codex_auth(&mut out, &report.codex);
    render_qwen_auth(&mut out, &report.qwen);
    render_minimax_auth(&mut out, &report.minimax);

    // Nous Tool Gateway ----------------------------------------------------
    let nous_logged_in = report.nous.logged_in;
    if report.nous_gateway.managed_enabled {
        out.blank();
        out.header("\u{25c6} Nous Tool Gateway");
        if !report.nous_gateway.nous_auth_present {
            out.line("  Nous Portal   \u{2717} not logged in");
        } else {
            out.line("  Nous Portal   \u{2713} managed tools available");
        }
        for feature in &report.nous_gateway.features {
            let state = if feature.managed_by_nous {
                "active via Nous subscription".to_string()
            } else if feature.active {
                let current = feature
                    .current_provider
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("configured provider");
                format!("active via {current}")
            } else if feature.included_by_default && report.nous_gateway.nous_auth_present {
                "included by subscription, not currently selected".to_string()
            } else if feature.key == "modal" && report.nous_gateway.nous_auth_present {
                "available via subscription (optional)".to_string()
            } else {
                "not configured".to_string()
            };
            out.line(&format!(
                "  {:<15} {} {}",
                feature.label,
                check_mark(feature.available || feature.active || feature.managed_by_nous),
                state
            ));
        }
    } else if nous_logged_in {
        out.blank();
        out.header("\u{25c6} Nous Tool Gateway");
        out.line("  Your free-tier Nous account does not include Tool Gateway access.");
        out.line("  Upgrade your subscription to unlock managed web, image, TTS, and browser tools.");
        if let Some(portal) = report.nous.portal_base_url.as_deref() {
            let portal = portal.trim_end_matches('/');
            if !portal.is_empty() {
                out.line(&format!("  Upgrade: {portal}"));
            }
        }
    }

    // API-key providers ----------------------------------------------------
    out.blank();
    out.header("\u{25c6} API-Key Providers");
    for row in &report.api_key_providers {
        let label = if row.configured {
            "configured"
        } else {
            "not configured (run: hermes model)"
        };
        out.line(&format!(
            "  {:<16} {} {}",
            row.label,
            check_mark(row.configured),
            label
        ));
    }
    if let Some(probe) = &report.lmstudio {
        out.line(&format!(
            "  {:<16} {} {}",
            "LM Studio",
            check_mark(probe.ok),
            probe.message
        ));
    }

    // Terminal backend -----------------------------------------------------
    out.blank();
    out.header("\u{25c6} Terminal Backend");
    let t = &report.terminal;
    out.line(&format!("  Backend:      {}", t.backend));
    match t.backend.as_str() {
        "ssh" => {
            out.line(&format!(
                "  SSH Host:     {}",
                if t.ssh_host.is_empty() { "(not set)" } else { &t.ssh_host }
            ));
            out.line(&format!(
                "  SSH User:     {}",
                if t.ssh_user.is_empty() { "(not set)" } else { &t.ssh_user }
            ));
        }
        "docker" => {
            out.line(&format!("  Docker Image: {}", t.docker_image));
        }
        "daytona" => {
            out.line(&format!("  Daytona Image: {}", t.daytona_image));
        }
        "vercel_sandbox" => {
            out.line(&format!("  Runtime:      {}", t.vercel_runtime));
            let sdk_label = if t.vercel_sdk_installed {
                "installed"
            } else {
                "missing (install: pip install 'hermes-agent[vercel]')"
            };
            out.line(&format!(
                "  SDK:          {} {}",
                check_mark(t.vercel_sdk_installed),
                sdk_label
            ));
            out.line(&format!(
                "  Auth:         {} {}",
                check_mark(t.vercel_auth_ok),
                t.vercel_auth_label
            ));
            for line in &t.vercel_auth_detail_lines {
                out.line(&format!("  Auth detail:  {line}"));
            }
            out.line(&format!(
                "  Persistence:  {}",
                if t.vercel_persist_enabled {
                    "snapshot filesystem"
                } else {
                    "ephemeral filesystem"
                }
            ));
            out.line("  Processes:    live processes do not survive cleanup, snapshots, or sandbox recreation");
        }
        _ => {}
    }
    out.line(&format!(
        "  Sudo:         {} {}",
        check_mark(t.sudo_enabled),
        if t.sudo_enabled { "enabled" } else { "disabled" }
    ));

    // Messaging platforms --------------------------------------------------
    out.blank();
    out.header("\u{25c6} Messaging Platforms");
    for p in &report.platforms {
        let mut status = if p.configured {
            "configured".to_string()
        } else {
            "not configured".to_string()
        };
        if !p.home_channel.is_empty() {
            status.push_str(&format!(" (home: {})", p.home_channel));
        }
        if p.is_plugin {
            // Plugin rows in Python use a fixed "configured"/"not configured"
            // string with a " (plugin)" suffix and never a home channel.
            let base = if p.configured { "configured" } else { "not configured" };
            out.line(&format!(
                "  {:<12}  {} {} (plugin)",
                p.label,
                check_mark(p.configured),
                base
            ));
        } else {
            out.line(&format!(
                "  {:<12}  {} {}",
                p.label,
                check_mark(p.configured),
                status
            ));
        }
    }

    // Gateway service ------------------------------------------------------
    out.blank();
    out.header("\u{25c6} Gateway Service");
    match &report.gateway {
        GatewaySection::Snapshot(snap) => {
            out.line(&format!(
                "  Status:       {} {}",
                check_mark(snap.running),
                if snap.running { "running" } else { "stopped" }
            ));
            out.line(&format!("  Manager:      {}", snap.manager));
            if let Some(pids) = &snap.gateway_pids_formatted {
                out.line(&format!("  PID(s):       {pids}"));
            }
            if snap.has_process_service_mismatch {
                out.line("  Service:      installed but not managing the current running gateway");
            } else if report.is_termux && !snap.has_pids {
                out.line("  Start with:   hermes gateway");
                out.line("  Note:         Android may stop background jobs when Termux is suspended");
            } else if snap.service_installed && !snap.service_running {
                out.line("  Service:      installed but stopped");
            }
        }
        GatewaySection::Unavailable => {
            if report.is_termux {
                out.line(&format!("  Status:       {}", color("unknown", &[Colors::DIM])));
                out.line("  Manager:      Termux / manual process");
            } else if report.platform_is_linux {
                out.line(&format!("  Status:       {}", color("unknown", &[Colors::DIM])));
                out.line("  Manager:      systemd/manual");
            } else if report.platform_is_darwin {
                out.line(&format!("  Status:       {}", color("unknown", &[Colors::DIM])));
                out.line("  Manager:      launchd");
            } else {
                out.line(&format!("  Status:       {}", color("N/A", &[Colors::DIM])));
                out.line("  Manager:      (not supported on this platform)");
            }
        }
    }

    // Scheduled jobs -------------------------------------------------------
    out.blank();
    out.header("\u{25c6} Scheduled Jobs");
    if report.jobs_file_exists {
        if report.jobs_error {
            out.line("  Jobs:         (error reading jobs file)");
        } else {
            out.line(&format!(
                "  Jobs:         {} active, {} total",
                report.jobs_active.unwrap_or(0),
                report.jobs_total.unwrap_or(0)
            ));
        }
    } else {
        out.line("  Jobs:         0");
    }

    // Sessions -------------------------------------------------------------
    out.blank();
    out.header("\u{25c6} Sessions");
    if report.sessions_file_exists {
        if report.sessions_error {
            out.line("  Active:       (error reading sessions file)");
        } else {
            out.line(&format!(
                "  Active:       {} session(s)",
                report.sessions_count.unwrap_or(0)
            ));
        }
    } else {
        out.line("  Active:       0");
    }

    // Deep checks ----------------------------------------------------------
    if report.deep {
        out.blank();
        out.header("\u{25c6} Deep Checks");
        if let Some(check) = &report.deep_checks.openrouter {
            match check {
                OpenRouterCheck::Ok => out.line(&format!(
                    "  OpenRouter:   {} reachable",
                    check_mark(true)
                )),
                OpenRouterCheck::HttpError(code) => out.line(&format!(
                    "  OpenRouter:   {} error ({})",
                    check_mark(false),
                    code
                )),
                OpenRouterCheck::Error(e) => out.line(&format!(
                    "  OpenRouter:   {} error: {}",
                    check_mark(false),
                    e
                )),
            }
        }
        if let Some(in_use) = report.deep_checks.port_18789_in_use {
            out.line(&format!(
                "  Port 18789:   {}",
                if in_use { "in use" } else { "available" }
            ));
        }
    }

    // Footer ---------------------------------------------------------------
    out.blank();
    out.line(&color(&"\u{2500}".repeat(60), &[Colors::DIM]));
    out.line(&color("  Run 'hermes doctor' for detailed diagnostics", &[Colors::DIM]));
    out.line(&color("  Run 'hermes setup' to configure", &[Colors::DIM]));
    out.blank();

    out.finish()
}

/// Print the rendered status to stdout, mirroring Python `show_status`.
pub fn show_status(report: &StatusReport) {
    print!("{}", render_status(report));
}

// ---------------------------------------------------------------------------
// Auth provider sub-renderers
// ---------------------------------------------------------------------------

fn render_nous_auth(out: &mut StatusWriter, nous: &OAuthStatus) {
    let logged_in = nous.logged_in;
    let label = if logged_in {
        "logged in"
    } else {
        "not logged in (run: hermes auth add nous --type oauth)"
    };
    out.line(&format!(
        "  {:<12}  {} {}",
        "Nous Portal",
        check_mark(logged_in),
        label
    ));

    let portal_url = nous
        .portal_base_url
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "(unknown)".to_string());
    let access_exp = format_iso_timestamp(nous.access_expires_at.as_deref());
    let key_exp = format_iso_timestamp(nous.agent_key_expires_at.as_deref());
    let refresh_label = if nous.has_refresh_token { "yes" } else { "no" };

    if logged_in || portal_url != "(unknown)" || nous.error.is_some() {
        out.line(&format!("    Portal URL: {portal_url}"));
    }
    if logged_in || nous.access_expires_at.as_deref().map(|s| !s.is_empty()).unwrap_or(false) {
        out.line(&format!("    Access exp: {access_exp}"));
    }
    if logged_in || nous.agent_key_expires_at.as_deref().map(|s| !s.is_empty()).unwrap_or(false) {
        out.line(&format!("    Key exp:    {key_exp}"));
    }
    if logged_in || nous.has_refresh_token {
        out.line(&format!("    Refresh:    {refresh_label}"));
    }
    if let Some(err) = &nous.error {
        if !logged_in {
            out.line(&format!("    Error:      {err}"));
        }
    }
}

fn render_codex_auth(out: &mut StatusWriter, codex: &OAuthStatus) {
    let logged_in = codex.logged_in;
    out.line(&format!(
        "  {:<12}  {} {}",
        "OpenAI Codex",
        check_mark(logged_in),
        if logged_in { "logged in" } else { "not logged in (run: hermes model)" }
    ));
    if let Some(file) = codex.auth_store.as_deref().filter(|s| !s.is_empty()) {
        out.line(&format!("    Auth file:  {file}"));
    }
    if codex.last_refresh.as_deref().map(|s| !s.is_empty()).unwrap_or(false) {
        let refreshed = format_iso_timestamp(codex.last_refresh.as_deref());
        out.line(&format!("    Refreshed:  {refreshed}"));
    }
    if let Some(err) = codex.error.as_deref().filter(|s| !s.is_empty()) {
        if !logged_in {
            out.line(&format!("    Error:      {err}"));
        }
    }
}

fn render_qwen_auth(out: &mut StatusWriter, qwen: &OAuthStatus) {
    let logged_in = qwen.logged_in;
    out.line(&format!(
        "  {:<12}  {} {}",
        "Qwen OAuth",
        check_mark(logged_in),
        if logged_in { "logged in" } else { "not logged in (run: qwen auth qwen-oauth)" }
    ));
    if let Some(file) = qwen.auth_file.as_deref().filter(|s| !s.is_empty()) {
        out.line(&format!("    Auth file:  {file}"));
    }
    if let Some(ms) = qwen.expires_at_ms.filter(|v| *v != 0) {
        out.line(&format!("    Access exp: {}", ms_to_utc_iso(ms)));
    }
    if let Some(err) = qwen.error.as_deref().filter(|s| !s.is_empty()) {
        if !logged_in {
            out.line(&format!("    Error:      {err}"));
        }
    }
}

fn render_minimax_auth(out: &mut StatusWriter, minimax: &OAuthStatus) {
    let logged_in = minimax.logged_in;
    out.line(&format!(
        "  {:<12}  {} {}",
        "MiniMax OAuth",
        check_mark(logged_in),
        if logged_in {
            "logged in"
        } else {
            "not logged in (run: hermes auth add minimax-oauth)"
        }
    ));
    if logged_in {
        if let Some(region) = minimax.region.as_deref().filter(|s| !s.is_empty()) {
            out.line(&format!("    Region:     {region}"));
        }
    }
    if let Some(exp) = minimax.expires_at.as_deref().filter(|s| !s.is_empty()) {
        out.line(&format!("    Access exp: {exp}"));
    }
    if let Some(err) = minimax.error.as_deref().filter(|s| !s.is_empty()) {
        if !logged_in {
            out.line(&format!("    Error:      {err}"));
        }
    }
}

// ---------------------------------------------------------------------------
// Output buffer
// ---------------------------------------------------------------------------

struct StatusWriter {
    lines: Vec<String>,
}

impl StatusWriter {
    fn new() -> Self {
        StatusWriter { lines: Vec::new() }
    }
    fn line(&mut self, s: &str) {
        self.lines.push(s.to_string());
    }
    fn blank(&mut self) {
        self.lines.push(String::new());
    }
    fn cyan(&mut self, s: &str) {
        let c = color(s, &[Colors::CYAN]);
        self.lines.push(c);
    }
    fn header(&mut self, s: &str) {
        let c = color(s, &[Colors::CYAN, Colors::BOLD]);
        self.lines.push(c);
    }
    fn finish(self) -> String {
        let mut s = self.lines.join("\n");
        s.push('\n');
        s
    }
}

// ---------------------------------------------------------------------------
// Default key / provider tables (mirrors Python's literal dicts)
// ---------------------------------------------------------------------------

/// The default API-key display table, in declaration order, as
/// `(label, [env var names])`. Anthropic is intentionally excluded — the
/// Python code skips it in the loop and renders it last via a dedicated
/// lookup. First non-empty env value wins.
pub fn default_api_key_table() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("OpenRouter", vec!["OPENROUTER_API_KEY"]),
        ("OpenAI", vec!["OPENAI_API_KEY"]),
        ("Google / Gemini", vec!["GOOGLE_API_KEY", "GEMINI_API_KEY"]),
        ("DeepSeek", vec!["DEEPSEEK_API_KEY"]),
        ("xAI / Grok", vec!["XAI_API_KEY"]),
        ("NVIDIA NIM", vec!["NVIDIA_API_KEY"]),
        ("Z.AI / GLM", vec!["GLM_API_KEY"]),
        ("Kimi", vec!["KIMI_API_KEY"]),
        ("StepFun Step Plan", vec!["STEPFUN_API_KEY"]),
        ("MiniMax", vec!["MINIMAX_API_KEY"]),
        ("MiniMax-CN", vec!["MINIMAX_CN_API_KEY"]),
        ("Firecrawl", vec!["FIRECRAWL_API_KEY"]),
        ("Tavily", vec!["TAVILY_API_KEY"]),
        ("Browser Use", vec!["BROWSER_USE_API_KEY"]),
        ("Browserbase", vec!["BROWSERBASE_API_KEY"]),
        ("FAL", vec!["FAL_KEY"]),
        ("Tinker", vec!["TINKER_API_KEY"]),
        ("WandB", vec!["WANDB_API_KEY"]),
        ("ElevenLabs", vec!["ELEVENLABS_API_KEY"]),
        ("GitHub", vec!["GITHUB_TOKEN"]),
    ]
}

/// The default API-key-provider table, in declaration order:
/// `(label, [env var names])`. First non-empty env value wins.
pub fn default_api_key_provider_table() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("Z.AI / GLM", vec!["GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"]),
        ("Kimi / Moonshot", vec!["KIMI_API_KEY"]),
        ("StepFun Step Plan", vec!["STEPFUN_API_KEY"]),
        ("MiniMax", vec!["MINIMAX_API_KEY"]),
        ("MiniMax (China)", vec!["MINIMAX_CN_API_KEY"]),
    ]
}

/// The default messaging-platform table: `(label, token_var, home_var?)`.
pub fn default_platform_table() -> Vec<(&'static str, &'static str, Option<&'static str>)> {
    vec![
        ("Telegram", "TELEGRAM_BOT_TOKEN", Some("TELEGRAM_HOME_CHANNEL")),
        ("Discord", "DISCORD_BOT_TOKEN", Some("DISCORD_HOME_CHANNEL")),
        ("WhatsApp", "WHATSAPP_ENABLED", None),
        ("Signal", "SIGNAL_HTTP_URL", Some("SIGNAL_HOME_CHANNEL")),
        ("Slack", "SLACK_BOT_TOKEN", None),
        ("Email", "EMAIL_ADDRESS", Some("EMAIL_HOME_ADDRESS")),
        ("SMS", "TWILIO_ACCOUNT_SID", Some("SMS_HOME_CHANNEL")),
        ("DingTalk", "DINGTALK_CLIENT_ID", None),
        ("Feishu", "FEISHU_APP_ID", Some("FEISHU_HOME_CHANNEL")),
        ("WeCom", "WECOM_BOT_ID", Some("WECOM_HOME_CHANNEL")),
        ("WeCom Callback", "WECOM_CALLBACK_CORP_ID", None),
        ("Weixin", "WEIXIN_ACCOUNT_ID", Some("WEIXIN_HOME_CHANNEL")),
        ("BlueBubbles", "BLUEBUBBLES_SERVER_URL", Some("BLUEBUBBLES_HOME_CHANNEL")),
        ("QQBot", "QQ_APP_ID", Some("QQ_HOME_CHANNEL")),
        ("Yuanbao", "YUANBAO_APP_ID", Some("YUANBAO_HOME_CHANNEL")),
    ]
}

/// Resolve the first non-empty env value among `names`, using `env`.
///
/// `env` maps env var names to their (already-resolved) values, matching the
/// Python `get_env_value` semantics (None / empty treated as absent).
pub fn resolve_first(env: &BTreeMap<String, String>, names: &[&str]) -> String {
    for name in names {
        if let Some(v) = env.get(*name) {
            if !v.is_empty() {
                return v.clone();
            }
        }
    }
    String::new()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn check_mark_no_color_when_disabled() {
        // With NO_COLOR set, color() returns the raw glyphs.
        unsafe {
            std::env::set_var("NO_COLOR", "1");
        }
        assert_eq!(check_mark(true), "\u{2713}");
        assert_eq!(check_mark(false), "\u{2717}");
        unsafe {
            std::env::remove_var("NO_COLOR");
        }
    }

    #[test]
    fn redact_key_empty_shows_not_set() {
        unsafe {
            std::env::set_var("NO_COLOR", "1");
        }
        assert_eq!(redact_key(""), "(not set)");
        // Short keys (< floor 12) collapse to ***.
        assert_eq!(redact_key("short"), "***");
        // Long keys show head...tail.
        assert_eq!(redact_key("sk-abcdefghijklmnop"), "sk-a...mnop");
        unsafe {
            std::env::remove_var("NO_COLOR");
        }
    }

    #[test]
    fn configured_model_label_variants() {
        assert_eq!(
            configured_model_label(&json!({"model": {"default": "gpt-x"}})),
            "gpt-x"
        );
        assert_eq!(
            configured_model_label(&json!({"model": {"name": "named"}})),
            "named"
        );
        // default wins over name
        assert_eq!(
            configured_model_label(&json!({"model": {"default": "d", "name": "n"}})),
            "d"
        );
        assert_eq!(configured_model_label(&json!({"model": "plain"})), "plain");
        assert_eq!(configured_model_label(&json!({})), "(not set)");
        assert_eq!(
            configured_model_label(&json!({"model": {"default": "   "}})),
            "(not set)"
        );
    }

    #[test]
    fn format_iso_unknown_and_passthrough() {
        assert_eq!(format_iso_timestamp(None), "(unknown)");
        assert_eq!(format_iso_timestamp(Some("")), "(unknown)");
        assert_eq!(format_iso_timestamp(Some("   ")), "(unknown)");
        // Unparseable -> original string returned unchanged.
        assert_eq!(format_iso_timestamp(Some("not-a-date")), "not-a-date");
    }

    #[test]
    fn format_iso_parses_z_suffix() {
        // A Z-suffixed UTC timestamp should parse and produce a local-formatted
        // string ending in the local tz token. We just assert it changed shape.
        let out = format_iso_timestamp(Some("2026-01-02T03:04:05Z"));
        assert_ne!(out, "2026-01-02T03:04:05Z");
        assert!(out.starts_with("20"));
        assert!(out.contains(':'));
    }

    #[test]
    fn ms_to_utc_iso_basic() {
        // 0 ms => epoch
        assert_eq!(ms_to_utc_iso(0), "1970-01-01T00:00:00+00:00");
        // 1_000 ms => one second past epoch
        assert_eq!(ms_to_utc_iso(1000), "1970-01-01T00:00:01+00:00");
    }

    #[test]
    fn resolve_first_picks_first_non_empty() {
        let mut env = BTreeMap::new();
        env.insert("A".to_string(), "".to_string());
        env.insert("B".to_string(), "val".to_string());
        assert_eq!(resolve_first(&env, &["A", "B"]), "val");
        assert_eq!(resolve_first(&env, &["A"]), "");
        assert_eq!(resolve_first(&env, &["MISSING"]), "");
    }

    fn minimal_report() -> StatusReport {
        StatusReport {
            show_all: false,
            deep: false,
            project_root: "/proj".to_string(),
            python_version: "3.11.0".to_string(),
            env_file_exists: true,
            configured_model: "(not set)".to_string(),
            effective_provider_label: "Auto".to_string(),
            api_keys: vec![ApiKeyRow {
                label: "OpenRouter".to_string(),
                value: String::new(),
            }],
            anthropic_value: String::new(),
            nous: OAuthStatus::default(),
            codex: OAuthStatus::default(),
            qwen: OAuthStatus::default(),
            minimax: OAuthStatus::default(),
            nous_gateway: NousGateway {
                managed_enabled: false,
                nous_auth_present: false,
                features: vec![],
            },
            api_key_providers: vec![ApiKeyProviderRow {
                label: "Z.AI / GLM".to_string(),
                configured: false,
            }],
            lmstudio: None,
            terminal: TerminalBackend {
                backend: "local".to_string(),
                ..Default::default()
            },
            platforms: vec![PlatformRow {
                label: "Telegram".to_string(),
                configured: false,
                home_channel: String::new(),
                is_plugin: false,
            }],
            gateway: GatewaySection::Unavailable,
            is_termux: false,
            platform_is_linux: false,
            platform_is_darwin: true,
            jobs_active: None,
            jobs_total: None,
            jobs_error: false,
            jobs_file_exists: false,
            sessions_count: None,
            sessions_error: false,
            sessions_file_exists: false,
            deep_checks: DeepChecks::default(),
        }
    }

    #[test]
    fn render_contains_core_sections() {
        unsafe {
            std::env::set_var("NO_COLOR", "1");
        }
        let report = minimal_report();
        let out = render_status(&report);
        assert!(out.contains("Hermes Agent Status"));
        assert!(out.contains("\u{25c6} Environment"));
        assert!(out.contains("  Project:      /proj"));
        assert!(out.contains("  Python:       3.11.0"));
        assert!(out.contains("  .env file:    \u{2713} exists"));
        assert!(out.contains("\u{25c6} API Keys"));
        // Anthropic row always rendered last with (not set).
        assert!(out.contains("Anthropic"));
        assert!(out.contains("(not set)"));
        assert!(out.contains("\u{25c6} Auth Providers"));
        assert!(out.contains("Nous Portal"));
        assert!(out.contains("not logged in (run: hermes auth add nous --type oauth)"));
        assert!(out.contains("\u{25c6} Terminal Backend"));
        assert!(out.contains("  Backend:      local"));
        assert!(out.contains("  Sudo:         \u{2717} disabled"));
        assert!(out.contains("\u{25c6} Messaging Platforms"));
        assert!(out.contains("\u{25c6} Gateway Service"));
        // darwin + Unavailable -> launchd
        assert!(out.contains("  Manager:      launchd"));
        assert!(out.contains("  Jobs:         0"));
        assert!(out.contains("  Active:       0"));
        assert!(out.contains("hermes doctor"));
        // No deep section when deep=false.
        assert!(!out.contains("\u{25c6} Deep Checks"));
        unsafe {
            std::env::remove_var("NO_COLOR");
        }
    }

    #[test]
    fn render_deep_section_and_gateway_snapshot() {
        unsafe {
            std::env::set_var("NO_COLOR", "1");
        }
        let mut report = minimal_report();
        report.deep = true;
        report.deep_checks = DeepChecks {
            openrouter: Some(OpenRouterCheck::HttpError(401)),
            port_18789_in_use: Some(true),
        };
        report.gateway = GatewaySection::Snapshot(GatewaySnapshot {
            running: true,
            manager: "launchd".to_string(),
            gateway_pids_formatted: Some("1234".to_string()),
            has_process_service_mismatch: false,
            service_installed: true,
            service_running: false,
            has_pids: true,
        });
        let out = render_status(&report);
        assert!(out.contains("  Status:       \u{2713} running"));
        assert!(out.contains("  PID(s):       1234"));
        assert!(out.contains("  Service:      installed but stopped"));
        assert!(out.contains("\u{25c6} Deep Checks"));
        assert!(out.contains("  OpenRouter:   \u{2717} error (401)"));
        assert!(out.contains("  Port 18789:   in use"));
        unsafe {
            std::env::remove_var("NO_COLOR");
        }
    }

    #[test]
    fn render_nous_gateway_feature_states() {
        unsafe {
            std::env::set_var("NO_COLOR", "1");
        }
        let mut report = minimal_report();
        report.nous.logged_in = true;
        report.nous_gateway = NousGateway {
            managed_enabled: true,
            nous_auth_present: true,
            features: vec![
                GatewayFeatureRow {
                    label: "Web".to_string(),
                    managed_by_nous: true,
                    active: false,
                    available: true,
                    included_by_default: true,
                    key: "web".to_string(),
                    current_provider: None,
                },
                GatewayFeatureRow {
                    label: "Modal".to_string(),
                    managed_by_nous: false,
                    active: false,
                    available: false,
                    included_by_default: false,
                    key: "modal".to_string(),
                    current_provider: None,
                },
            ],
        };
        let out = render_status(&report);
        assert!(out.contains("Nous Portal   \u{2713} managed tools available"));
        assert!(out.contains("active via Nous subscription"));
        assert!(out.contains("available via subscription (optional)"));
        unsafe {
            std::env::remove_var("NO_COLOR");
        }
    }
}
