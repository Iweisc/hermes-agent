//! Shared helpers for tool backend selection.
//!
//! Native Rust port of `tools/tool_backend_helpers.py`. Reproduces provider /
//! modal-mode normalisation, direct-Modal credential detection, OpenAI audio
//! key resolution, and the direct-vs-managed Modal backend resolution state
//! machine.
//!
//! Several of the Python helpers reach into other subsystems
//! (`hermes_cli.auth`, `hermes_cli.config`) that are not yet ported with
//! matching signatures. To avoid blocking, those cross-cutting decisions are
//! supplied to the pure functions as parameters (e.g. `managed_nous_tools_enabled`
//! is passed in to [`resolve_modal_backend_state`]), and convenience wrappers
//! that consult the environment / config are provided where the inputs are
//! self-contained (env vars, home-dir files).

use std::env;
use std::path::PathBuf;

use crate::mod_utils::{is_truthy_value, TruthyInput};

/// Default browser cloud provider key.
pub const DEFAULT_BROWSER_PROVIDER: &str = "local";
/// Default modal execution mode.
pub const DEFAULT_MODAL_MODE: &str = "auto";
/// The set of valid modal execution modes.
pub const VALID_MODAL_MODES: [&str; 3] = ["auto", "direct", "managed"];

/// Return a normalized browser provider key.
///
/// Mirrors `normalize_browser_cloud_provider`: coerces `None`/empty to the
/// default, trims, lowercases, and falls back to the default when the result
/// is empty.
pub fn normalize_browser_cloud_provider(value: Option<&str>) -> String {
    let raw = match value {
        Some(v) if !v.is_empty() => v,
        // Python: `str(value or _DEFAULT_BROWSER_PROVIDER)` — `None` and the
        // empty string are both falsy and fall back to the default.
        _ => DEFAULT_BROWSER_PROVIDER,
    };
    let provider = raw.trim().to_lowercase();
    if provider.is_empty() {
        DEFAULT_BROWSER_PROVIDER.to_string()
    } else {
        provider
    }
}

/// Return the requested modal mode when valid, else the default.
///
/// Mirrors `coerce_modal_mode`.
pub fn coerce_modal_mode(value: Option<&str>) -> String {
    let raw = match value {
        Some(v) if !v.is_empty() => v,
        _ => DEFAULT_MODAL_MODE,
    };
    let mode = raw.trim().to_lowercase();
    if VALID_MODAL_MODES.contains(&mode.as_str()) {
        mode
    } else {
        DEFAULT_MODAL_MODE.to_string()
    }
}

/// Return a normalized modal execution mode.
///
/// Mirrors `normalize_modal_mode`, which simply delegates to
/// [`coerce_modal_mode`].
pub fn normalize_modal_mode(value: Option<&str>) -> String {
    coerce_modal_mode(value)
}

/// Return true when direct Modal credentials/config are available.
///
/// Mirrors `has_direct_modal_credentials`: true when both `MODAL_TOKEN_ID` and
/// `MODAL_TOKEN_SECRET` are set (to any value, matching Python's truthiness of
/// non-empty strings), or when `~/.modal.toml` exists.
pub fn has_direct_modal_credentials() -> bool {
    let env_creds = !env::var("MODAL_TOKEN_ID").unwrap_or_default().is_empty()
        && !env::var("MODAL_TOKEN_SECRET").unwrap_or_default().is_empty();
    if env_creds {
        return true;
    }
    home_modal_toml_exists()
}

/// Return true when `~/.modal.toml` exists. Mirrors
/// `(Path.home() / ".modal.toml").exists()`.
fn home_modal_toml_exists() -> bool {
    match dirs::home_dir() {
        Some(home) => {
            let mut p = PathBuf::from(home);
            p.push(".modal.toml");
            p.exists()
        }
        None => false,
    }
}

/// Resolved direct-vs-managed Modal backend selection state.
///
/// Faithful mirror of the dict returned by `resolve_modal_backend_state`.
/// `selected_backend` is `None` when neither backend is usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModalBackendState {
    /// The originally requested mode, after validation/coercion.
    pub requested_mode: String,
    /// The normalized mode (currently identical to `requested_mode`).
    pub mode: String,
    /// Whether direct credentials are available.
    pub has_direct: bool,
    /// Whether the managed backend is otherwise ready.
    pub managed_ready: bool,
    /// True when `managed` was requested but managed tools are disabled.
    pub managed_mode_blocked: bool,
    /// The chosen backend: `"managed"`, `"direct"`, or `None`.
    pub selected_backend: Option<String>,
}

/// Resolve direct vs managed Modal backend selection.
///
/// Faithful port of `resolve_modal_backend_state`. Semantics:
/// - `direct` means direct-only
/// - `managed` means managed-only
/// - `auto` prefers managed when available, then falls back to direct
///
/// The Python original calls `managed_nous_tools_enabled()` internally; since
/// that consults un-ported auth/subscription state, it is supplied here as the
/// `managed_nous_tools_enabled` parameter so this function stays pure and
/// testable.
pub fn resolve_modal_backend_state(
    modal_mode: Option<&str>,
    has_direct: bool,
    managed_ready: bool,
    managed_nous_tools_enabled: bool,
) -> ModalBackendState {
    let requested_mode = coerce_modal_mode(modal_mode);
    let normalized_mode = normalize_modal_mode(modal_mode);
    let managed_mode_blocked = requested_mode == "managed" && !managed_nous_tools_enabled;

    let selected_backend: Option<String> = if normalized_mode == "managed" {
        if managed_nous_tools_enabled && managed_ready {
            Some("managed".to_string())
        } else {
            None
        }
    } else if normalized_mode == "direct" {
        if has_direct {
            Some("direct".to_string())
        } else {
            None
        }
    } else {
        // auto
        if managed_nous_tools_enabled && managed_ready {
            Some("managed".to_string())
        } else if has_direct {
            Some("direct".to_string())
        } else {
            None
        }
    };

    ModalBackendState {
        requested_mode,
        mode: normalized_mode,
        has_direct,
        managed_ready,
        managed_mode_blocked,
        selected_backend,
    }
}

/// Prefer the voice-tools key, but fall back to the normal OpenAI key.
///
/// Mirrors `resolve_openai_audio_api_key`: reads `VOICE_TOOLS_OPENAI_KEY` then
/// `OPENAI_API_KEY` (both defaulting to `""`), takes the first non-empty, and
/// strips surrounding whitespace.
pub fn resolve_openai_audio_api_key() -> String {
    let voice = env::var("VOICE_TOOLS_OPENAI_KEY").unwrap_or_default();
    // Python: `a or b` returns `a` when truthy (non-empty), else `b`.
    let chosen = if !voice.is_empty() {
        voice
    } else {
        env::var("OPENAI_API_KEY").unwrap_or_default()
    };
    chosen.trim().to_string()
}

/// Return true when the user opted into the Tool Gateway for this tool's
/// config section.
///
/// Mirrors `prefers_gateway`: reads `<section>.use_gateway` from a loaded
/// config and evaluates it with [`is_truthy_value`] (default `false`). The
/// Python original loads `config.yaml` via `hermes_cli.config.load_config`;
/// since that loader is not yet ported with a matching signature, the parsed
/// config section is supplied here as a parameter. `None` (section missing or
/// not a mapping) yields `false`, never raising.
pub fn prefers_gateway(section: Option<&serde_json::Value>) -> bool {
    if let Some(serde_json::Value::Object(map)) = section {
        let truthy = json_to_truthy(map.get("use_gateway"));
        return is_truthy_value(&truthy, false);
    }
    false
}

/// Return true when a FAL key value is set to a non-whitespace value.
///
/// Mirrors `fal_key_is_configured`: consults the environment first and, when
/// `FAL_KEY` is unset, the supplied `.env` fallback value (Python falls back to
/// `hermes_cli.config.get_env_value("FAL_KEY")`). A whitespace-only value is
/// treated as unset everywhere.
///
/// `env_fallback` should be `Some(value)` when the `.env` lookup succeeded,
/// `None` otherwise (matching Python's `None` on missing/error).
pub fn fal_key_is_configured(env_fallback: Option<&str>) -> bool {
    // os.getenv("FAL_KEY") returns None when truly unset; an empty-string env
    // var is present-but-empty. We distinguish via env::var's Result.
    let value: Option<String> = match env::var("FAL_KEY") {
        Ok(v) => Some(v),
        Err(_) => env_fallback.map(|s| s.to_string()),
    };
    match value {
        Some(v) => !v.trim().is_empty(),
        None => false,
    }
}

/// Convert an optional JSON value into a [`TruthyInput`] for use with
/// [`is_truthy_value`], matching Python's handling of `dict.get(...)`:
/// missing/`null` -> `None`, bool -> `Bool`, string -> `Str`, anything else ->
/// `Other(<python-truthiness>)`.
fn json_to_truthy(value: Option<&serde_json::Value>) -> TruthyInput {
    match value {
        None | Some(serde_json::Value::Null) => TruthyInput::None,
        Some(serde_json::Value::Bool(b)) => TruthyInput::Bool(*b),
        Some(serde_json::Value::String(s)) => TruthyInput::Str(s.clone()),
        Some(serde_json::Value::Number(n)) => {
            // Python bool(number): non-zero is truthy.
            let truthy = n.as_f64().map(|f| f != 0.0).unwrap_or(true);
            TruthyInput::Other(truthy)
        }
        Some(serde_json::Value::Array(a)) => TruthyInput::Other(!a.is_empty()),
        Some(serde_json::Value::Object(o)) => TruthyInput::Other(!o.is_empty()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_browser_provider_defaults_and_normalizes() {
        assert_eq!(normalize_browser_cloud_provider(None), "local");
        assert_eq!(normalize_browser_cloud_provider(Some("")), "local");
        assert_eq!(normalize_browser_cloud_provider(Some("   ")), "local");
        assert_eq!(normalize_browser_cloud_provider(Some("  AWS ")), "aws");
        assert_eq!(normalize_browser_cloud_provider(Some("Cloud")), "cloud");
    }

    #[test]
    fn coerce_modal_mode_validates() {
        assert_eq!(coerce_modal_mode(None), "auto");
        assert_eq!(coerce_modal_mode(Some("")), "auto");
        assert_eq!(coerce_modal_mode(Some(" DIRECT ")), "direct");
        assert_eq!(coerce_modal_mode(Some("Managed")), "managed");
        assert_eq!(coerce_modal_mode(Some("bogus")), "auto");
        assert_eq!(normalize_modal_mode(Some("auto")), "auto");
    }

    #[test]
    fn resolve_managed_mode() {
        // managed requested, enabled + ready -> managed
        let s = resolve_modal_backend_state(Some("managed"), false, true, true);
        assert_eq!(s.selected_backend.as_deref(), Some("managed"));
        assert!(!s.managed_mode_blocked);

        // managed requested but not enabled -> blocked, no backend
        let s = resolve_modal_backend_state(Some("managed"), true, true, false);
        assert_eq!(s.selected_backend, None);
        assert!(s.managed_mode_blocked);

        // managed requested, enabled but not ready -> no backend
        let s = resolve_modal_backend_state(Some("managed"), true, false, true);
        assert_eq!(s.selected_backend, None);
        assert!(!s.managed_mode_blocked);
    }

    #[test]
    fn resolve_direct_mode() {
        let s = resolve_modal_backend_state(Some("direct"), true, true, true);
        assert_eq!(s.selected_backend.as_deref(), Some("direct"));
        assert!(!s.managed_mode_blocked);

        let s = resolve_modal_backend_state(Some("direct"), false, true, true);
        assert_eq!(s.selected_backend, None);
    }

    #[test]
    fn resolve_auto_mode_prefers_managed() {
        // managed available -> managed
        let s = resolve_modal_backend_state(Some("auto"), true, true, true);
        assert_eq!(s.selected_backend.as_deref(), Some("managed"));

        // managed not ready, direct available -> direct
        let s = resolve_modal_backend_state(Some("auto"), true, false, true);
        assert_eq!(s.selected_backend.as_deref(), Some("direct"));

        // nothing available
        let s = resolve_modal_backend_state(None, false, false, false);
        assert_eq!(s.selected_backend, None);
        assert_eq!(s.requested_mode, "auto");
        assert_eq!(s.mode, "auto");
    }

    #[test]
    fn prefers_gateway_reads_section() {
        let cfg = json!({"use_gateway": true});
        assert!(prefers_gateway(Some(&cfg)));

        let cfg = json!({"use_gateway": "yes"});
        assert!(prefers_gateway(Some(&cfg)));

        let cfg = json!({"use_gateway": "no"});
        assert!(!prefers_gateway(Some(&cfg)));

        let cfg = json!({"other": 1});
        assert!(!prefers_gateway(Some(&cfg)));

        // Not a mapping / missing.
        assert!(!prefers_gateway(None));
        let not_map = json!("nope");
        assert!(!prefers_gateway(Some(&not_map)));
    }

    #[test]
    fn fal_key_configured_logic() {
        // env present and non-blank
        unsafe {
            env::set_var("FAL_KEY", "  abc  ");
        }
        assert!(fal_key_is_configured(None));

        // env present but blank
        unsafe {
            env::set_var("FAL_KEY", "   ");
        }
        assert!(!fal_key_is_configured(Some("fallback")));

        // env unset -> use fallback
        unsafe {
            env::remove_var("FAL_KEY");
        }
        assert!(fal_key_is_configured(Some("present")));
        assert!(!fal_key_is_configured(Some("  ")));
        assert!(!fal_key_is_configured(None));
    }

    #[test]
    fn openai_audio_key_prefers_voice() {
        unsafe {
            env::set_var("VOICE_TOOLS_OPENAI_KEY", "  voicekey  ");
            env::set_var("OPENAI_API_KEY", "normalkey");
        }
        assert_eq!(resolve_openai_audio_api_key(), "voicekey");

        unsafe {
            env::remove_var("VOICE_TOOLS_OPENAI_KEY");
        }
        assert_eq!(resolve_openai_audio_api_key(), "normalkey");

        unsafe {
            env::remove_var("OPENAI_API_KEY");
        }
        assert_eq!(resolve_openai_audio_api_key(), "");
    }
}
