//! Shared constants for Hermes Agent.
//!
//! Native Rust port of `hermes_constants.py`. Import-safe module with no
//! dependencies on the rest of the crate — mirrors the Python module that is
//! safe to import from anywhere without risk of circular imports.
//!
//! Behavioural notes vs. the Python original:
//! * [`apply_ipv4_preference`] cannot monkey-patch `socket.getaddrinfo` (there
//!   is no global resolver to patch in Rust). It is preserved as a no-op that
//!   honours the same `force` gating so call-sites port 1:1; the actual IPv4
//!   preference is expressed at the `reqwest` client-builder level by callers.
//! * The profile-fallback warning in [`get_hermes_home`] is emitted at most
//!   once per process (matching `_profile_fallback_warned`), written directly
//!   to stderr.

use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use std::io::Write as _;

/// One-shot guard so the HERMES_HOME profile-fallback warning fires at most once.
static PROFILE_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

/// Cached WSL detection (`None`=0, `Some(false)`=1, `Some(true)`=2).
static WSL_DETECTED: AtomicU8 = AtomicU8::new(0);

/// Cached container detection (`None`=0, `Some(false)`=1, `Some(true)`=2).
static CONTAINER_DETECTED: AtomicU8 = AtomicU8::new(0);

/// Best-effort home directory, mirroring `Path.home()`.
fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// Return the Hermes home directory (default: `~/.hermes`).
///
/// Reads the `HERMES_HOME` env var, falling back to `~/.hermes`. This is the
/// single source of truth — all other copies should call this.
///
/// When `HERMES_HOME` is unset but an `active_profile` file indicates a
/// non-default profile is active, logs a loud one-shot warning to stderr so
/// cross-profile data corruption is diagnosable instead of silent. Behaviour is
/// unchanged otherwise — we still return `~/.hermes`.
pub fn get_hermes_home() -> PathBuf {
    if let Ok(val) = env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }

    // Guard: if a non-default profile is sticky-active, warn once that the
    // fallback to the default profile is almost certainly wrong.
    if !PROFILE_FALLBACK_WARNED.load(Ordering::Relaxed) {
        let active_path = home_dir().join(".hermes").join("active_profile");
        let active = match std::fs::read_to_string(&active_path) {
            Ok(s) => s.trim().to_string(),
            Err(_) => String::new(),
        };
        if !active.is_empty() && active != "default" {
            // Set the flag first; only emit if we were the one to flip it, to
            // avoid a double-emit race between threads.
            if !PROFILE_FALLBACK_WARNED.swap(true, Ordering::Relaxed) {
                let msg = format!(
                    "[HERMES_HOME fallback] HERMES_HOME is unset but active \
                     profile is {active:?}. Falling back to ~/.hermes, which \
                     is the DEFAULT profile — not {active:?}. Any data this \
                     process writes will land in the wrong profile. The \
                     subprocess spawner should pass HERMES_HOME explicitly \
                     (see issue #18594)."
                );
                let stderr = std::io::stderr();
                let mut lock = stderr.lock();
                let _ = writeln!(lock, "{msg}");
                let _ = lock.flush();
            }
        }
    }

    home_dir().join(".hermes")
}

/// Return the root Hermes directory for profile-level operations.
///
/// * Standard deployments: `~/.hermes`.
/// * Docker / custom deployments where `HERMES_HOME` points outside `~/.hermes`
///   (e.g. `/opt/data`): returns `HERMES_HOME` directly — that *is* the root.
/// * Profile mode where `HERMES_HOME` is `<root>/profiles/<name>`: returns
///   `<root>` so `profile list` can see all profiles. Works for both standard
///   (`~/.hermes/profiles/coder`) and Docker (`/opt/data/profiles/coder`)
///   layouts.
pub fn get_default_hermes_root() -> PathBuf {
    let native_home = home_dir().join(".hermes");
    let env_home = env::var("HERMES_HOME").unwrap_or_default();
    if env_home.is_empty() {
        return native_home;
    }
    let env_path = PathBuf::from(&env_home);

    // Mirror `env_path.resolve().relative_to(native_home.resolve())`: if the
    // resolved env path is under the resolved native home, HERMES_HOME is in
    // normal or profile mode → return native home.
    let env_resolved = env_path
        .canonicalize()
        .unwrap_or_else(|_| lexical_abspath(&env_path));
    let native_resolved = native_home
        .canonicalize()
        .unwrap_or_else(|_| lexical_abspath(&native_home));
    if env_resolved.starts_with(&native_resolved) {
        return native_home;
    }

    // Docker / custom deployment. Check if this is a profile path:
    // `<root>/profiles/<name>` — if the immediate parent dir is `profiles`,
    // the root is the grandparent.
    if env_path.parent().and_then(|p| p.file_name()).map(|n| n == "profiles") == Some(true) {
        // grandparent
        if let Some(root) = env_path.parent().and_then(|p| p.parent()) {
            return root.to_path_buf();
        }
    }

    // Not a profile path — HERMES_HOME itself is the root.
    env_path
}

/// Lexical absolutisation without resolving symlinks (fallback for
/// non-existent paths), mirroring `Path.resolve()`'s lexical aspect.
fn lexical_abspath(p: &Path) -> PathBuf {
    use std::path::Component;
    let base = if p.is_absolute() {
        PathBuf::new()
    } else {
        env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
    };
    let mut out = base;
    for comp in p.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir => {
                out = PathBuf::from("/");
            }
            Component::Prefix(pre) => {
                out = PathBuf::from(pre.as_os_str());
            }
            Component::Normal(seg) => out.push(seg),
        }
    }
    out
}

/// Return the optional-skills directory, honouring package-manager wrappers.
///
/// Packaged installs may ship `optional-skills` outside the package tree and
/// expose it via `HERMES_OPTIONAL_SKILLS`.
pub fn get_optional_skills_dir(default: Option<PathBuf>) -> PathBuf {
    if let Ok(override_val) = env::var("HERMES_OPTIONAL_SKILLS") {
        let override_val = override_val.trim();
        if !override_val.is_empty() {
            return PathBuf::from(override_val);
        }
    }
    if let Some(default) = default {
        return default;
    }
    get_hermes_home().join("optional-skills")
}

/// Resolve a Hermes subdirectory with backward compatibility.
///
/// New installs get the consolidated layout (e.g. `cache/images`). Existing
/// installs that already have the old path (e.g. `image_cache`) keep using it —
/// no migration required.
///
/// Returns the old location if it exists on disk, otherwise the new one.
pub fn get_hermes_dir(new_subpath: &str, old_name: &str) -> PathBuf {
    let home = get_hermes_home();
    let old_path = home.join(old_name);
    if old_path.exists() {
        return old_path;
    }
    home.join(new_subpath)
}

/// Return a user-friendly display string for the current HERMES_HOME.
///
/// Uses `~/` shorthand for readability:
/// * default: `~/.hermes`
/// * profile: `~/.hermes/profiles/coder`
/// * custom:  `/opt/hermes-custom`
pub fn display_hermes_home() -> String {
    let home = get_hermes_home();
    match home.strip_prefix(home_dir()) {
        Ok(rel) => format!("~/{}", rel.display()),
        Err(_) => home.display().to_string(),
    }
}

/// Return a per-profile HOME directory for subprocesses, or `None`.
///
/// When `{HERMES_HOME}/home/` exists on disk, subprocesses should use it as
/// `HOME` so system tools (git, ssh, gh, npm …) write their configs inside the
/// Hermes data directory instead of the OS-level `/root` or `~/`.
///
/// Activation is directory-based: if the `home/` subdirectory doesn't exist,
/// returns `None`.
pub fn get_subprocess_home() -> Option<String> {
    let hermes_home = env::var("HERMES_HOME").unwrap_or_default();
    if hermes_home.is_empty() {
        return None;
    }
    let profile_home = Path::new(&hermes_home).join("home");
    if profile_home.is_dir() {
        return Some(profile_home.to_string_lossy().into_owned());
    }
    None
}

/// Valid reasoning effort levels (excluding the special `"none"` sentinel).
pub const VALID_REASONING_EFFORTS: [&str; 5] =
    ["minimal", "low", "medium", "high", "xhigh"];

/// Parsed reasoning-effort configuration, mirroring the Python dict.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReasoningEffort {
    pub enabled: bool,
    /// Present only when `enabled` is true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

/// Parse a reasoning effort level into a config struct.
///
/// Valid levels: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`.
/// * Returns `None` when the input is empty or unrecognised (caller uses default).
/// * Returns `{enabled: false}` for `"none"`.
/// * Returns `{enabled: true, effort: <level>}` for valid effort levels.
pub fn parse_reasoning_effort(effort: &str) -> Option<ReasoningEffort> {
    let trimmed = effort.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lowered = trimmed.to_lowercase();
    if lowered == "none" {
        return Some(ReasoningEffort {
            enabled: false,
            effort: None,
        });
    }
    if VALID_REASONING_EFFORTS.contains(&lowered.as_str()) {
        return Some(ReasoningEffort {
            enabled: true,
            effort: Some(lowered),
        });
    }
    None
}

/// Return true when running inside a Termux (Android) environment.
///
/// Checks `TERMUX_VERSION` (set by Termux) or the Termux-specific `PREFIX` path.
pub fn is_termux() -> bool {
    let prefix = env::var("PREFIX").unwrap_or_default();
    env::var("TERMUX_VERSION").map(|v| !v.is_empty()).unwrap_or(false)
        || prefix.contains("com.termux/files/usr")
}

/// Return true when running inside WSL (Windows Subsystem for Linux).
///
/// Checks `/proc/version` for the `microsoft` marker that both WSL1 and WSL2
/// inject. Result is cached for the process lifetime.
pub fn is_wsl() -> bool {
    match WSL_DETECTED.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    let detected = match std::fs::read_to_string("/proc/version") {
        Ok(contents) => contents.to_lowercase().contains("microsoft"),
        Err(_) => false,
    };
    WSL_DETECTED.store(if detected { 2 } else { 1 }, Ordering::Relaxed);
    detected
}

/// Return true when running inside a Docker/Podman container.
///
/// Checks `/.dockerenv` (Docker), `/run/.containerenv` (Podman), and
/// `/proc/1/cgroup` for container runtime markers. Result is cached for the
/// process lifetime.
pub fn is_container() -> bool {
    match CONTAINER_DETECTED.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    let detected = detect_container();
    CONTAINER_DETECTED.store(if detected { 2 } else { 1 }, Ordering::Relaxed);
    detected
}

fn detect_container() -> bool {
    if Path::new("/.dockerenv").exists() {
        return true;
    }
    if Path::new("/run/.containerenv").exists() {
        return true;
    }
    if let Ok(cgroup) = std::fs::read_to_string("/proc/1/cgroup") {
        if cgroup.contains("docker") || cgroup.contains("podman") || cgroup.contains("/lxc/") {
            return true;
        }
    }
    false
}

// ─── Well-Known Paths ─────────────────────────────────────────────────────────

/// Return the path to `config.yaml` under HERMES_HOME.
pub fn get_config_path() -> PathBuf {
    get_hermes_home().join("config.yaml")
}

/// Return the path to the skills directory under HERMES_HOME.
pub fn get_skills_dir() -> PathBuf {
    get_hermes_home().join("skills")
}

/// Return the path to the `.env` file under HERMES_HOME.
pub fn get_env_path() -> PathBuf {
    get_hermes_home().join(".env")
}

// ─── Network Preferences ─────────────────────────────────────────────────────

/// Prefer IPv4 connections.
///
/// In Python this monkey-patches `socket.getaddrinfo` so that AF_UNSPEC
/// resolutions skip IPv6. Rust has no global resolver to patch, so this is a
/// gated no-op preserved for 1:1 call-site porting: callers express IPv4
/// preference at the `reqwest::ClientBuilder` level
/// (`.local_address` / `.resolve`) when needed.
///
/// Honours the same `force` gate and is safe to call multiple times.
pub fn apply_ipv4_preference(force: bool) {
    if !force {
        return;
    }
    // No-op: see doc comment. Intentionally left without side effects.
}

// ─── Provider Base URLs ───────────────────────────────────────────────────────

pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// `{OPENROUTER_BASE_URL}/models`.
pub fn openrouter_models_url() -> String {
    format!("{OPENROUTER_BASE_URL}/models")
}

pub const AI_GATEWAY_BASE_URL: &str = "https://ai-gateway.vercel.sh/v1";

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialise env mutation across tests in this module.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn get_hermes_home_honours_env() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_HOME", "/opt/data"); }
        assert_eq!(get_hermes_home(), PathBuf::from("/opt/data"));
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn get_hermes_home_trims_and_falls_back_on_blank() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_HOME", "   "); }
        let home = get_hermes_home();
        assert!(home.ends_with(".hermes"), "{home:?}");
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn get_hermes_home_strips_whitespace_around_value() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_HOME", "  /opt/data  "); }
        assert_eq!(get_hermes_home(), PathBuf::from("/opt/data"));
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn default_root_unset_is_native_home() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::remove_var("HERMES_HOME"); }
        assert_eq!(get_default_hermes_root(), home_dir().join(".hermes"));
    }

    #[test]
    fn default_root_docker_profile_returns_grandparent() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_HOME", "/opt/data/profiles/coder"); }
        assert_eq!(get_default_hermes_root(), PathBuf::from("/opt/data"));
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn default_root_docker_custom_returns_self() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_HOME", "/opt/data"); }
        assert_eq!(get_default_hermes_root(), PathBuf::from("/opt/data"));
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn optional_skills_dir_override_wins() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_OPTIONAL_SKILLS", "/custom/skills"); }
        assert_eq!(
            get_optional_skills_dir(Some(PathBuf::from("/ignored"))),
            PathBuf::from("/custom/skills")
        );
        unsafe { env::remove_var("HERMES_OPTIONAL_SKILLS"); }
    }

    #[test]
    fn optional_skills_dir_default_then_home() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::remove_var("HERMES_OPTIONAL_SKILLS"); }
        assert_eq!(
            get_optional_skills_dir(Some(PathBuf::from("/d"))),
            PathBuf::from("/d")
        );
        unsafe { env::set_var("HERMES_HOME", "/opt/data"); }
        assert_eq!(
            get_optional_skills_dir(None),
            PathBuf::from("/opt/data/optional-skills")
        );
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn well_known_paths() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_HOME", "/opt/data"); }
        assert_eq!(get_config_path(), PathBuf::from("/opt/data/config.yaml"));
        assert_eq!(get_skills_dir(), PathBuf::from("/opt/data/skills"));
        assert_eq!(get_env_path(), PathBuf::from("/opt/data/.env"));
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn display_home_custom() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_HOME", "/opt/hermes-custom"); }
        assert_eq!(display_hermes_home(), "/opt/hermes-custom");
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn subprocess_home_none_when_unset() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::remove_var("HERMES_HOME"); }
        assert_eq!(get_subprocess_home(), None);
    }

    #[test]
    fn subprocess_home_none_when_dir_missing() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::set_var("HERMES_HOME", "/nonexistent-hermes-home-xyz"); }
        assert_eq!(get_subprocess_home(), None);
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn reasoning_effort_empty_is_none() {
        assert_eq!(parse_reasoning_effort(""), None);
        assert_eq!(parse_reasoning_effort("   "), None);
    }

    #[test]
    fn reasoning_effort_none_disables() {
        assert_eq!(
            parse_reasoning_effort("none"),
            Some(ReasoningEffort {
                enabled: false,
                effort: None
            })
        );
        assert_eq!(
            parse_reasoning_effort("  NONE  "),
            Some(ReasoningEffort {
                enabled: false,
                effort: None
            })
        );
    }

    #[test]
    fn reasoning_effort_valid_levels() {
        for level in VALID_REASONING_EFFORTS {
            assert_eq!(
                parse_reasoning_effort(level),
                Some(ReasoningEffort {
                    enabled: true,
                    effort: Some(level.to_string())
                })
            );
        }
        // case-insensitive
        assert_eq!(
            parse_reasoning_effort("HIGH"),
            Some(ReasoningEffort {
                enabled: true,
                effort: Some("high".to_string())
            })
        );
    }

    #[test]
    fn reasoning_effort_unknown_is_none() {
        assert_eq!(parse_reasoning_effort("turbo"), None);
        assert_eq!(parse_reasoning_effort("ultra"), None);
    }

    #[test]
    fn reasoning_effort_serializes_like_python_dict() {
        let on = parse_reasoning_effort("low").unwrap();
        let v = serde_json::to_value(&on).unwrap();
        assert_eq!(v, serde_json::json!({"enabled": true, "effort": "low"}));

        let off = parse_reasoning_effort("none").unwrap();
        let v = serde_json::to_value(&off).unwrap();
        assert_eq!(v, serde_json::json!({"enabled": false}));
    }

    #[test]
    fn ipv4_preference_noop_is_safe() {
        apply_ipv4_preference(false);
        apply_ipv4_preference(true);
        apply_ipv4_preference(true);
    }

    #[test]
    fn provider_urls() {
        assert_eq!(OPENROUTER_BASE_URL, "https://openrouter.ai/api/v1");
        assert_eq!(openrouter_models_url(), "https://openrouter.ai/api/v1/models");
        assert_eq!(AI_GATEWAY_BASE_URL, "https://ai-gateway.vercel.sh/v1");
    }

    #[test]
    fn detection_helpers_are_callable_and_cached() {
        // Just ensure they don't panic and return stable cached values.
        let a = is_wsl();
        assert_eq!(a, is_wsl());
        let b = is_container();
        assert_eq!(b, is_container());
        let _ = is_termux();
    }
}
