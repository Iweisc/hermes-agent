//! Welcome banner, ASCII art, skills summary, and update check for the CLI.
//!
//! Native Rust port of `hermes_cli/banner.py`. The Python module is built on
//! `rich`/`prompt_toolkit`, neither of which exists here; instead we reproduce
//! the *content* faithfully — the same ASCII art, the same Rich-markup strings,
//! the same git/update-check logic, and the same string-formatting rules. A
//! caller (the TUI renderer) is responsible for actually painting the markup.
//!
//! The pure-display helpers carry no `HermesCLI` state, matching the Python
//! module's contract.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::cli_backup::get_hermes_home;
use crate::cli_skin_engine::get_active_skin;

// =========================================================================
// Version metadata (mirrors `hermes_cli.__version__` / `__release_date__`).
// =========================================================================

/// Mirrors `hermes_cli.__version__`.
pub const VERSION: &str = "0.12.0";
/// Mirrors `hermes_cli.__release_date__`.
pub const RELEASE_DATE: &str = "2026.4.30";

// =========================================================================
// ANSI building blocks for conversation display
// =========================================================================

/// True-color `#FFD700` bold.
pub const GOLD: &str = "\x1b[1;38;2;255;215;0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const RST: &str = "\x1b[0m";

// =========================================================================
// Skin-aware color helpers
// =========================================================================

/// Get a color from the active skin, or return `fallback`.
pub fn skin_color(key: &str, fallback: &str) -> String {
    // `get_active_skin` already swallows failures and returns a default skin,
    // so this never panics. `SkinConfig::get_color` applies the fallback.
    get_active_skin().get_color(key, fallback)
}

/// Get a branding string from the active skin, or return `fallback`.
pub fn skin_branding(key: &str, fallback: &str) -> String {
    get_active_skin().get_branding(key, fallback)
}

// =========================================================================
// ASCII Art & Branding
// =========================================================================

pub const HERMES_AGENT_LOGO: &str = "[bold #FFD700]██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD700]██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#FFBF00]███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#FFBF00]██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#CD7F32]██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#CD7F32]╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]";

pub const HERMES_CADUCEUS: &str = "[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀[/]
[#FFBF00]⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]";

// =========================================================================
// Skills scanning
// =========================================================================

/// A single discovered skill, mirroring the dict shape produced by
/// `tools.skills_tool._find_all_skills`. Used as the input to
/// [`group_skills_by_category`].
#[derive(Debug, Clone)]
pub struct SkillInfo {
    pub name: String,
    /// `None` (or empty) is treated as the `"general"` category.
    pub category: Option<String>,
}

/// Group skills by category, mirroring `get_available_skills`.
///
/// The Python version delegates to `_find_all_skills()` (platform/disabled
/// filtering already applied) and then buckets by `category` (defaulting to
/// `"general"`). Since the skill discovery module is not yet ported, the
/// already-filtered list is taken as a parameter. The bucketing/default logic
/// is reproduced exactly.
pub fn group_skills_by_category(all_skills: &[SkillInfo]) -> BTreeMap<String, Vec<String>> {
    let mut skills_by_category: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for skill in all_skills {
        let category = match &skill.category {
            Some(c) if !c.is_empty() => c.clone(),
            _ => "general".to_string(),
        };
        skills_by_category
            .entry(category)
            .or_default()
            .push(skill.name.clone());
    }
    skills_by_category
}

// =========================================================================
// Update check
// =========================================================================

/// Cache update check results for 6 hours to avoid repeated git fetches.
pub const UPDATE_CHECK_CACHE_SECONDS: u64 = 6 * 3600;

/// Sentinel returned when we know an update exists but can't count commits
/// (e.g. nix-built hermes — no local git history to count against).
pub const UPDATE_AVAILABLE_NO_COUNT: i64 = -1;

const UPSTREAM_REPO_URL: &str = "https://github.com/NousResearch/hermes-agent.git";

/// Compare an embedded git revision to upstream main via `git ls-remote`.
///
/// Returns `Some(0)` if up-to-date, `Some(UPDATE_AVAILABLE_NO_COUNT)` if behind,
/// or `None` on failure.
pub fn check_via_rev(local_rev: &str) -> Option<i64> {
    let output = Command::new("git")
        .args(["ls-remote", UPSTREAM_REPO_URL, "refs/heads/main"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.is_empty() {
        return None;
    }
    let upstream_rev = stdout.split_whitespace().next().unwrap_or("");
    if upstream_rev.is_empty() {
        return None;
    }
    Some(if upstream_rev == local_rev {
        0
    } else {
        UPDATE_AVAILABLE_NO_COUNT
    })
}

/// Count commits behind `origin/main` in a local checkout.
pub fn check_via_local_git(repo_dir: &Path) -> Option<i64> {
    // Best-effort fetch; offline/timeout falls through to stale refs.
    let _ = Command::new("git")
        .args(["fetch", "origin", "--quiet"])
        .current_dir(repo_dir)
        .output();

    let output = Command::new("git")
        .args(["rev-list", "--count", "HEAD..origin/main"])
        .current_dir(repo_dir)
        .output()
        .ok()?;
    if output.status.success() {
        let s = String::from_utf8_lossy(&output.stdout);
        return s.trim().parse::<i64>().ok();
    }
    None
}

/// Check whether a Hermes update is available.
///
/// Two paths: if `HERMES_REVISION` is set (nix builds embed it), compare it to
/// upstream main via `git ls-remote`. Otherwise look for a local git checkout
/// and count commits behind `origin/main`.
///
/// Returns the number of commits behind, `UPDATE_AVAILABLE_NO_COUNT` (-1) if
/// behind but the count is unknown, `0` if up-to-date, or `None` if the check
/// failed or doesn't apply. Cached for 6 hours.
///
/// `fallback_repo_dir` is the equivalent of Python's
/// `Path(__file__).parent.parent.resolve()` — the directory of the running
/// install used when no `~/.hermes/hermes-agent` checkout exists. Pass the
/// detected project root (or `None` to skip that fallback).
pub fn check_for_updates(fallback_repo_dir: Option<&Path>) -> Option<i64> {
    let hermes_home = get_hermes_home();
    let cache_file = hermes_home.join(".update_check");
    let embedded_rev = std::env::var("HERMES_REVISION")
        .ok()
        .filter(|s| !s.is_empty());

    let now = now_secs();

    // Read cache — invalidate if the embedded rev has changed since last check.
    if let Ok(text) = std::fs::read_to_string(&cache_file) {
        if let Ok(cached) = serde_json::from_str::<Value>(&text) {
            let ts = cached.get("ts").and_then(Value::as_f64).unwrap_or(0.0);
            let cached_rev = cached.get("rev").and_then(Value::as_str);
            let rev_matches = cached_rev.map(|s| s.to_string()) == embedded_rev;
            if (now - ts) < UPDATE_CHECK_CACHE_SECONDS as f64 && rev_matches {
                return cached
                    .get("behind")
                    .and_then(|b| if b.is_null() { None } else { b.as_i64() });
            }
        }
    }

    let behind: Option<i64> = if let Some(rev) = embedded_rev.as_deref() {
        check_via_rev(rev)
    } else {
        let mut repo_dir = hermes_home.join("hermes-agent");
        if !repo_dir.join(".git").exists() {
            match fallback_repo_dir {
                Some(d) => repo_dir = d.to_path_buf(),
                None => return None,
            }
        }
        if !repo_dir.join(".git").exists() {
            return None;
        }
        check_via_local_git(&repo_dir)
    };

    let payload = json!({
        "ts": now,
        "behind": behind,
        "rev": embedded_rev,
    });
    let _ = std::fs::write(&cache_file, payload.to_string());

    behind
}

/// Return the active Hermes git checkout, or `None` if this isn't a git install.
///
/// `fallback_repo_dir` mirrors Python's `Path(__file__).parent.parent.resolve()`.
pub fn resolve_repo_dir(fallback_repo_dir: Option<&Path>) -> Option<PathBuf> {
    let hermes_home = get_hermes_home();
    let mut repo_dir = hermes_home.join("hermes-agent");
    if !repo_dir.join(".git").exists() {
        repo_dir = fallback_repo_dir?.to_path_buf();
    }
    if repo_dir.join(".git").exists() {
        Some(repo_dir)
    } else {
        None
    }
}

/// Resolve a git revision to an 8-character short hash.
pub fn git_short_hash(repo_dir: &Path, rev: &str) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=8", rev])
        .current_dir(repo_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Upstream/local git hashes for the startup banner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitBannerState {
    pub upstream: String,
    pub local: String,
    pub ahead: i64,
}

/// Return upstream/local git hashes for the startup banner.
///
/// `repo_dir` overrides the resolved checkout; pass `None` to resolve via
/// [`resolve_repo_dir`] (which itself needs `fallback_repo_dir`).
pub fn get_git_banner_state(
    repo_dir: Option<&Path>,
    fallback_repo_dir: Option<&Path>,
) -> Option<GitBannerState> {
    let resolved: PathBuf = match repo_dir {
        Some(d) => d.to_path_buf(),
        None => resolve_repo_dir(fallback_repo_dir)?,
    };

    let upstream = git_short_hash(&resolved, "origin/main")?;
    let local = git_short_hash(&resolved, "HEAD")?;

    let mut ahead = 0i64;
    if let Ok(output) = Command::new("git")
        .args(["rev-list", "--count", "origin/main..HEAD"])
        .current_dir(&resolved)
        .output()
    {
        if output.status.success() {
            let s = String::from_utf8_lossy(&output.stdout);
            let trimmed = s.trim();
            ahead = if trimmed.is_empty() {
                0
            } else {
                trimmed.parse::<i64>().unwrap_or(0)
            };
        }
    }

    Some(GitBannerState {
        upstream,
        local,
        ahead: ahead.max(0),
    })
}

const RELEASE_URL_BASE: &str = "https://github.com/NousResearch/hermes-agent/releases/tag";

// Cached per-process: `None` = not yet looked up; `Some(None)` = looked up and
// no tag (falsy sentinel); `Some(Some(..))` = resolved (tag, url).
static LATEST_RELEASE_CACHE: Mutex<Option<Option<(String, String)>>> = Mutex::new(None);

/// Return `(tag, release_url)` for the latest git tag, or `None`.
///
/// Local-only — runs `git describe --tags --abbrev=0` against the Hermes
/// checkout. Cached per-process. Release URL always points at the canonical
/// NousResearch/hermes-agent repo (forks don't get a link).
pub fn get_latest_release_tag(
    repo_dir: Option<&Path>,
    fallback_repo_dir: Option<&Path>,
) -> Option<(String, String)> {
    {
        let cache = LATEST_RELEASE_CACHE.lock().unwrap();
        if let Some(resolved) = cache.as_ref() {
            return resolved.clone();
        }
    }

    let result = compute_latest_release_tag(repo_dir, fallback_repo_dir);
    *LATEST_RELEASE_CACHE.lock().unwrap() = Some(result.clone());
    result
}

fn compute_latest_release_tag(
    repo_dir: Option<&Path>,
    fallback_repo_dir: Option<&Path>,
) -> Option<(String, String)> {
    let resolved: PathBuf = match repo_dir {
        Some(d) => d.to_path_buf(),
        None => resolve_repo_dir(fallback_repo_dir)?,
    };

    let output = Command::new("git")
        .args(["describe", "--tags", "--abbrev=0"])
        .current_dir(&resolved)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let tag = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if tag.is_empty() {
        return None;
    }
    let url = format!("{RELEASE_URL_BASE}/{tag}");
    Some((tag, url))
}

/// Reset the per-process latest-release cache. Test helper / re-resolve hook.
pub fn reset_latest_release_cache() {
    *LATEST_RELEASE_CACHE.lock().unwrap() = None;
}

/// Return the version label shown in the startup banner title.
pub fn format_banner_version_label(state: Option<&GitBannerState>) -> String {
    let base = format!("Hermes Agent v{VERSION} ({RELEASE_DATE})");
    let state = match state {
        Some(s) => s,
        None => return base,
    };

    let upstream = &state.upstream;
    let local = &state.local;
    let ahead = state.ahead;

    if ahead <= 0 || upstream == local {
        return format!("{base} · upstream {upstream}");
    }

    let carried_word = if ahead == 1 { "commit" } else { "commits" };
    format!(
        "{base} · upstream {upstream} · local {local} (+{ahead} carried {carried_word})"
    )
}

// =========================================================================
// Welcome banner
// =========================================================================

/// Format a token count for display (e.g. 128000 → "128K", 1048576 → "1M").
pub fn format_context_length(tokens: i64) -> String {
    if tokens >= 1_000_000 {
        let val = tokens as f64 / 1_000_000.0;
        let rounded = val.round();
        if (val - rounded).abs() < 0.05 {
            return format!("{}M", rounded as i64);
        }
        return format!("{val:.1}M");
    } else if tokens >= 1_000 {
        let val = tokens as f64 / 1_000.0;
        let rounded = val.round();
        if (val - rounded).abs() < 0.05 {
            return format!("{}K", rounded as i64);
        }
        return format!("{val:.1}K");
    }
    tokens.to_string()
}

/// Normalize internal/legacy toolset identifiers for banner display.
pub fn display_toolset_name(toolset_name: &str) -> String {
    if toolset_name.is_empty() {
        return "unknown".to_string();
    }
    if let Some(stripped) = toolset_name.strip_suffix("_tools") {
        stripped.to_string()
    } else {
        toolset_name.to_string()
    }
}

/// Status of a single MCP server, mirroring entries from `get_mcp_status()`.
#[derive(Debug, Clone)]
pub struct McpServerStatus {
    pub name: String,
    pub transport: String,
    pub connected: bool,
    pub tools: i64,
}

/// An unavailable toolset, mirroring an entry of `check_tool_availability`'s
/// second return value.
#[derive(Debug, Clone, Default)]
pub struct UnavailableToolset {
    /// The `name` field used for `TOOLSET_REQUIREMENTS` lookup.
    pub name: String,
    /// The `id` field used for display (falls back to `name`).
    pub id: Option<String>,
    pub tools: Vec<String>,
    /// Whether this toolset has a lazy `check_fn` (its tools render yellow
    /// rather than red, since they aren't misconfigured — just not yet
    /// initialized).
    pub has_check_fn: bool,
}

/// Inputs needed to build the welcome banner. The Python `build_welcome_banner`
/// pulls these from several modules at call time (`model_tools`,
/// `tools.mcp_tool`, `tools.skills_tool`, `hermes_cli.profiles`,
/// `hermes_cli.config`). Those modules are not all ported yet, so the data is
/// passed in here, keeping this function pure and testable.
#[derive(Debug, Clone, Default)]
pub struct BannerInputs {
    pub model: String,
    pub cwd: String,
    /// Tool display names, in the order returned by the agent. The Python
    /// version reads `tool["function"]["name"]`; pre-extract those names.
    pub tools: Vec<String>,
    pub session_id: Option<String>,
    pub context_length: Option<i64>,
    /// Maps a tool name to its toolset name (the `get_toolset_for_tool`
    /// callable). Tools missing from the map fall back to `"other"`.
    pub toolset_for_tool: BTreeMap<String, String>,
    pub unavailable_toolsets: Vec<UnavailableToolset>,
    pub mcp_status: Vec<McpServerStatus>,
    pub skills: Vec<SkillInfo>,
    /// Active profile name; rendered when not `"default"`.
    pub active_profile_name: Option<String>,
    /// Result of the prefetched update check (commits behind / sentinel).
    pub update_behind: Option<i64>,
    /// `recommended_update_command()` text (used when `behind > 0`).
    pub recommended_update_command: String,
    /// `get_managed_update_command()` text (used for the no-count sentinel).
    pub managed_update_command: Option<String>,
    /// Banner title version label (from [`format_banner_version_label`]).
    pub version_label: String,
    /// `(tag, url)` for the latest release link, if any.
    pub release_info: Option<(String, String)>,
}

/// A fully-rendered welcome banner, as Rich markup strings. The TUI is
/// responsible for laying these into a panel / grid and painting them. This
/// mirrors exactly what `build_welcome_banner` constructs before handing off to
/// `rich`.
#[derive(Debug, Clone)]
pub struct WelcomeBanner {
    /// Left column lines (caduceus + model/cwd/session).
    pub left_lines: Vec<String>,
    /// Right column lines (tools / MCP / skills / summary / update).
    pub right_lines: Vec<String>,
    /// Panel title markup.
    pub title_markup: String,
    /// Border style color.
    pub border_color: String,
    /// The logo to show above the panel when terminal width >= 95.
    pub logo: String,
}

/// Build the welcome banner content. Faithful port of `build_welcome_banner`'s
/// string construction (everything up to the `rich` rendering calls).
pub fn build_welcome_banner(inputs: &BannerInputs) -> WelcomeBanner {
    // Replicate the disabled/lazy partition over unavailable toolsets.
    let mut disabled_tools: std::collections::BTreeSet<String> = Default::default();
    let mut lazy_tools: std::collections::BTreeSet<String> = Default::default();
    for item in &inputs.unavailable_toolsets {
        if item.has_check_fn {
            lazy_tools.extend(item.tools.iter().cloned());
        } else {
            disabled_tools.extend(item.tools.iter().cloned());
        }
    }

    // Resolve skin colors once for the entire banner.
    let accent = skin_color("banner_accent", "#FFBF00");
    let dim = skin_color("banner_dim", "#B8860B");
    let text = skin_color("banner_text", "#FFF8DC");
    let session_color = skin_color("session_border", "#8B8682");

    // Skin's custom caduceus art if provided.
    let skin = get_active_skin();
    let hero = if !skin.banner_hero.is_empty() {
        skin.banner_hero.clone()
    } else {
        HERMES_CADUCEUS.to_string()
    };

    let mut left_lines: Vec<String> = vec![String::new(), hero, String::new()];

    let mut model_short = match inputs.model.rsplit_once('/') {
        Some((_, last)) => last.to_string(),
        None => inputs.model.clone(),
    };
    if let Some(stripped) = model_short.strip_suffix(".gguf") {
        model_short = stripped.to_string();
    }
    if model_short.chars().count() > 28 {
        let truncated: String = model_short.chars().take(25).collect();
        model_short = format!("{truncated}...");
    }

    let ctx_str = match inputs.context_length {
        Some(ctx) if ctx != 0 => format!(
            " [dim {dim}]·[/] [dim {dim}]{} context[/]",
            format_context_length(ctx)
        ),
        _ => String::new(),
    };
    left_lines.push(format!(
        "[{accent}]{model_short}[/]{ctx_str} [dim {dim}]·[/] [dim {dim}]Nous Research[/]"
    ));
    left_lines.push(format!("[dim {dim}]{}[/]", inputs.cwd));
    if let Some(sid) = &inputs.session_id {
        if !sid.is_empty() {
            left_lines.push(format!("[dim {session_color}]Session: {sid}[/]"));
        }
    }

    // ----- Right column -----
    let mut right_lines: Vec<String> = vec![format!("[bold {accent}]Available Tools[/]")];

    let mut toolsets_dict: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for tool_name in &inputs.tools {
        let raw = inputs
            .toolset_for_tool
            .get(tool_name)
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("other");
        let toolset = display_toolset_name(raw);
        toolsets_dict.entry(toolset).or_default().push(tool_name.clone());
    }

    for item in &inputs.unavailable_toolsets {
        let toolset_id = item
            .id
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                if item.name.is_empty() {
                    "unknown".to_string()
                } else {
                    item.name.clone()
                }
            });
        let display_name = display_toolset_name(&toolset_id);
        let entry = toolsets_dict.entry(display_name).or_default();
        for tool_name in &item.tools {
            if !entry.contains(tool_name) {
                entry.push(tool_name.clone());
            }
        }
    }

    // BTreeMap already yields sorted keys, matching `sorted(toolsets_dict.keys())`.
    let sorted_toolsets: Vec<String> = toolsets_dict.keys().cloned().collect();
    let display_toolsets: Vec<&String> = sorted_toolsets.iter().take(8).collect();
    let remaining_toolsets = sorted_toolsets.len() as i64 - 8;

    for toolset in &display_toolsets {
        let mut tool_names = toolsets_dict[*toolset].clone();
        tool_names.sort();

        let color_one = |name: &str| -> String {
            if disabled_tools.contains(name) {
                format!("[red]{name}[/]")
            } else if lazy_tools.contains(name) {
                format!("[yellow]{name}[/]")
            } else {
                format!("[{text}]{name}[/]")
            }
        };

        let colored_names: Vec<String> = tool_names.iter().map(|n| color_one(n)).collect();
        let mut tools_str = colored_names.join(", ");

        // Length check uses the plain (uncolored) joined names.
        let plain_joined = tool_names.join(", ");
        if plain_joined.chars().count() > 45 {
            let mut short_names: Vec<String> = Vec::new();
            let mut length = 0usize;
            for name in &tool_names {
                if length + name.chars().count() + 2 > 42 {
                    short_names.push("...".to_string());
                    break;
                }
                short_names.push(name.clone());
                length += name.chars().count() + 2;
            }
            let colored_short: Vec<String> = short_names
                .iter()
                .map(|name| {
                    if name == "..." {
                        "[dim]...[/]".to_string()
                    } else {
                        color_one(name)
                    }
                })
                .collect();
            tools_str = colored_short.join(", ");
        }

        right_lines.push(format!("[dim {dim}]{toolset}:[/] {tools_str}"));
    }

    if remaining_toolsets > 0 {
        right_lines.push(format!(
            "[dim {dim}](and {remaining_toolsets} more toolsets...)[/]"
        ));
    }

    // MCP Servers section (only if configured).
    if !inputs.mcp_status.is_empty() {
        right_lines.push(String::new());
        right_lines.push(format!("[bold {accent}]MCP Servers[/]"));
        for srv in &inputs.mcp_status {
            if srv.connected {
                right_lines.push(format!(
                    "[dim {dim}]{}[/] [{text}]({})[/] [dim {dim}]—[/] [{text}]{} tool(s)[/]",
                    srv.name, srv.transport, srv.tools
                ));
            } else {
                right_lines.push(format!(
                    "[red]{}[/] [dim]({})[/] [red]— failed[/]",
                    srv.name, srv.transport
                ));
            }
        }
    }

    // Skills section.
    right_lines.push(String::new());
    right_lines.push(format!("[bold {accent}]Available Skills[/]"));
    let skills_by_category = group_skills_by_category(&inputs.skills);
    let total_skills: usize = skills_by_category.values().map(|v| v.len()).sum();

    if !skills_by_category.is_empty() {
        // BTreeMap yields sorted categories, matching `sorted(...keys())`.
        for (category, skill_names) in &skills_by_category {
            let mut skill_names = skill_names.clone();
            skill_names.sort();
            let mut skills_str = if skill_names.len() > 8 {
                let display_names = &skill_names[..8];
                format!(
                    "{} +{} more",
                    display_names.join(", "),
                    skill_names.len() - 8
                )
            } else {
                skill_names.join(", ")
            };
            if skills_str.chars().count() > 50 {
                let truncated: String = skills_str.chars().take(47).collect();
                skills_str = format!("{truncated}...");
            }
            right_lines.push(format!("[dim {dim}]{category}:[/] [{text}]{skills_str}[/]"));
        }
    } else {
        right_lines.push(format!("[dim {dim}]No skills installed[/]"));
    }

    // Summary line.
    right_lines.push(String::new());
    let mcp_connected = inputs.mcp_status.iter().filter(|s| s.connected).count();
    let mut summary_parts: Vec<String> = vec![
        format!("{} tools", inputs.tools.len()),
        format!("{total_skills} skills"),
    ];
    if mcp_connected > 0 {
        summary_parts.push(format!("{mcp_connected} MCP servers"));
    }
    summary_parts.push("/help for commands".to_string());

    // Active profile name when not 'default'.
    if let Some(profile) = &inputs.active_profile_name {
        if !profile.is_empty() && profile != "default" {
            right_lines.push(format!("[bold {accent}]Profile:[/] [{text}]{profile}[/]"));
        }
    }

    right_lines.push(format!("[dim {dim}]{}[/]", summary_parts.join(" · ")));

    // Update check — use prefetched result.
    if let Some(behind) = inputs.update_behind {
        if behind != 0 {
            if behind > 0 {
                let commits_word = if behind == 1 { "commit" } else { "commits" };
                right_lines.push(format!(
                    "[bold yellow]⚠ {behind} {commits_word} behind[/][dim yellow] — run [bold]{}[/bold] to update[/]",
                    inputs.recommended_update_command
                ));
            } else {
                // UPDATE_AVAILABLE_NO_COUNT: nix-built hermes; we know an update
                // exists but not by how much.
                let mut line = "[bold yellow]⚠ update available[/]".to_string();
                if let Some(cmd) = &inputs.managed_update_command {
                    if !cmd.is_empty() {
                        line.push_str(&format!("[dim yellow] — run [bold]{cmd}[/bold][/]"));
                    }
                }
                right_lines.push(line);
            }
        }
    }

    // Panel title.
    let title_color = skin_color("banner_title", "#FFD700");
    let border_color = skin_color("banner_border", "#CD7F32");
    let title_markup = match &inputs.release_info {
        Some((_tag, url)) => format!(
            "[bold {title_color}][link={url}]{}[/link][/]",
            inputs.version_label
        ),
        None => format!("[bold {title_color}]{}[/]", inputs.version_label),
    };

    // Logo (skin override or default).
    let logo = if !skin.banner_logo.is_empty() {
        skin.banner_logo.clone()
    } else {
        HERMES_AGENT_LOGO.to_string()
    };

    WelcomeBanner {
        left_lines,
        right_lines,
        title_markup,
        border_color,
        logo,
    }
}

// =========================================================================
// Helpers
// =========================================================================

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_context_length_basic() {
        assert_eq!(format_context_length(500), "500");
        assert_eq!(format_context_length(128_000), "128K");
        assert_eq!(format_context_length(1_048_576), "1M");
        assert_eq!(format_context_length(1_000_000), "1M");
        assert_eq!(format_context_length(2_000_000), "2M");
    }

    #[test]
    fn format_context_length_fractional() {
        // 1500 / 1000 = 1.5 -> "1.5K"
        assert_eq!(format_context_length(1_500), "1.5K");
        // 1_500_000 / 1_000_000 = 1.5 -> "1.5M"
        assert_eq!(format_context_length(1_500_000), "1.5M");
        // 1024 -> 1.024K, rounds within 0.05? 1.024 vs 1 -> diff 0.024 < 0.05 -> "1K"
        assert_eq!(format_context_length(1_024), "1K");
        // 32768 -> 32.768K -> diff from 33 is 0.232 -> "32.8K"
        assert_eq!(format_context_length(32_768), "32.8K");
    }

    #[test]
    fn display_toolset_name_strips_suffix() {
        assert_eq!(display_toolset_name(""), "unknown");
        assert_eq!(display_toolset_name("web_tools"), "web");
        assert_eq!(display_toolset_name("browser"), "browser");
        assert_eq!(display_toolset_name("_tools"), "");
    }

    #[test]
    fn group_skills_defaults_category() {
        let skills = vec![
            SkillInfo {
                name: "a".into(),
                category: Some("dev".into()),
            },
            SkillInfo {
                name: "b".into(),
                category: None,
            },
            SkillInfo {
                name: "c".into(),
                category: Some(String::new()),
            },
        ];
        let grouped = group_skills_by_category(&skills);
        assert_eq!(grouped.get("dev").unwrap(), &vec!["a".to_string()]);
        assert_eq!(
            grouped.get("general").unwrap(),
            &vec!["b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn version_label_no_state() {
        let label = format_banner_version_label(None);
        assert_eq!(label, format!("Hermes Agent v{VERSION} ({RELEASE_DATE})"));
    }

    #[test]
    fn version_label_upstream_only_when_not_ahead() {
        let state = GitBannerState {
            upstream: "abc12345".into(),
            local: "abc12345".into(),
            ahead: 0,
        };
        let label = format_banner_version_label(Some(&state));
        assert!(label.ends_with("· upstream abc12345"));
        assert!(!label.contains("carried"));
    }

    #[test]
    fn version_label_carried_commits() {
        let state = GitBannerState {
            upstream: "abc12345".into(),
            local: "def67890".into(),
            ahead: 1,
        };
        let label = format_banner_version_label(Some(&state));
        assert!(label.contains("upstream abc12345"));
        assert!(label.contains("local def67890"));
        assert!(label.contains("(+1 carried commit)"));

        let state2 = GitBannerState {
            upstream: "abc12345".into(),
            local: "def67890".into(),
            ahead: 3,
        };
        let label2 = format_banner_version_label(Some(&state2));
        assert!(label2.contains("(+3 carried commits)"));
    }

    #[test]
    fn banner_basic_structure() {
        let mut inputs = BannerInputs::default();
        inputs.model = "anthropic/claude-opus-4".into();
        inputs.cwd = "/tmp/work".into();
        inputs.session_id = Some("sess-1".into());
        inputs.context_length = Some(200_000);
        inputs.tools = vec!["read_file".into(), "write_file".into()];
        inputs
            .toolset_for_tool
            .insert("read_file".into(), "fs_tools".into());
        inputs
            .toolset_for_tool
            .insert("write_file".into(), "fs_tools".into());
        inputs.version_label = "Hermes Agent v0.12.0 (2026.4.30)".into();

        let banner = build_welcome_banner(&inputs);

        // Model short name from path, plus context shown.
        assert!(banner.left_lines.iter().any(|l| l.contains("claude-opus-4")));
        assert!(banner.left_lines.iter().any(|l| l.contains("200K context")));
        assert!(banner.left_lines.iter().any(|l| l.contains("/tmp/work")));
        assert!(banner.left_lines.iter().any(|l| l.contains("Session: sess-1")));

        // Tools grouped under display toolset "fs".
        assert!(banner.right_lines.iter().any(|l| l.contains("fs:")));
        assert!(banner.right_lines.iter().any(|l| l.contains("read_file")));

        // Summary: 2 tools, 0 skills.
        assert!(banner
            .right_lines
            .iter()
            .any(|l| l.contains("2 tools") && l.contains("0 skills")));

        // No skills installed message.
        assert!(banner
            .right_lines
            .iter()
            .any(|l| l.contains("No skills installed")));
    }

    #[test]
    fn banner_disabled_and_lazy_tools_colored() {
        let mut inputs = BannerInputs::default();
        inputs.model = "m".into();
        inputs.cwd = "/x".into();
        inputs.tools = vec!["t1".into()];
        inputs
            .toolset_for_tool
            .insert("t1".into(), "main_tools".into());
        inputs.unavailable_toolsets = vec![
            UnavailableToolset {
                name: "honcho_tools".into(),
                id: Some("honcho_tools".into()),
                tools: vec!["honcho_query".into()],
                has_check_fn: true,
            },
            UnavailableToolset {
                name: "broken_tools".into(),
                id: Some("broken_tools".into()),
                tools: vec!["broken_op".into()],
                has_check_fn: false,
            },
        ];

        let banner = build_welcome_banner(&inputs);
        let joined = banner.right_lines.join("\n");
        // lazy tool -> yellow, disabled tool -> red.
        assert!(joined.contains("[yellow]honcho_query[/]"));
        assert!(joined.contains("[red]broken_op[/]"));
    }

    #[test]
    fn banner_update_behind() {
        let mut inputs = BannerInputs::default();
        inputs.model = "m".into();
        inputs.cwd = "/x".into();
        inputs.update_behind = Some(2);
        inputs.recommended_update_command = "hermes update".into();
        let banner = build_welcome_banner(&inputs);
        assert!(banner
            .right_lines
            .iter()
            .any(|l| l.contains("⚠ 2 commits behind") && l.contains("hermes update")));
    }

    #[test]
    fn banner_update_no_count_sentinel() {
        let mut inputs = BannerInputs::default();
        inputs.model = "m".into();
        inputs.cwd = "/x".into();
        inputs.update_behind = Some(UPDATE_AVAILABLE_NO_COUNT);
        inputs.managed_update_command = Some("nix run".into());
        let banner = build_welcome_banner(&inputs);
        assert!(banner
            .right_lines
            .iter()
            .any(|l| l.contains("⚠ update available") && l.contains("nix run")));
    }

    #[test]
    fn banner_title_with_release_link() {
        let mut inputs = BannerInputs::default();
        inputs.model = "m".into();
        inputs.cwd = "/x".into();
        inputs.version_label = "v0.12.0".into();
        inputs.release_info = Some(("v0.12.0".into(), "https://example/tag/v0.12.0".into()));
        let banner = build_welcome_banner(&inputs);
        assert!(banner.title_markup.contains("[link=https://example/tag/v0.12.0]"));
        assert!(banner.title_markup.contains("v0.12.0"));
    }

    #[test]
    fn model_short_truncation_and_gguf() {
        let mut inputs = BannerInputs::default();
        inputs.cwd = "/x".into();
        inputs.model = "some-really-extremely-long-model-name-that-exceeds".into();
        let banner = build_welcome_banner(&inputs);
        // truncated to 25 chars + "..."
        assert!(banner.left_lines.iter().any(|l| l.contains("...")));

        let mut inputs2 = BannerInputs::default();
        inputs2.cwd = "/x".into();
        inputs2.model = "models/llama-3.gguf".into();
        let banner2 = build_welcome_banner(&inputs2);
        assert!(banner2.left_lines.iter().any(|l| l.contains("llama-3")));
        assert!(!banner2.left_lines.iter().any(|l| l.contains(".gguf")));
    }

    #[test]
    fn profile_shown_only_when_not_default() {
        let mut inputs = BannerInputs::default();
        inputs.model = "m".into();
        inputs.cwd = "/x".into();
        inputs.active_profile_name = Some("default".into());
        let banner = build_welcome_banner(&inputs);
        assert!(!banner.right_lines.iter().any(|l| l.contains("Profile:")));

        let mut inputs2 = BannerInputs::default();
        inputs2.model = "m".into();
        inputs2.cwd = "/x".into();
        inputs2.active_profile_name = Some("work".into());
        let banner2 = build_welcome_banner(&inputs2);
        assert!(banner2.right_lines.iter().any(|l| l.contains("Profile:") && l.contains("work")));
    }
}
