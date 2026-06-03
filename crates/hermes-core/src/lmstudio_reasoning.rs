//! LM Studio reasoning-effort resolution shared by the chat-completions
//! transport and run_agent's iteration-limit summary path.
//!
//! LM Studio publishes per-model `capabilities.reasoning.allowed_options` (e.g.
//! `["off","on"]` for toggle-style models, `["off","minimal","low"]` for
//! graduated models). We map the user's `reasoning_config` onto LM Studio's
//! OpenAI-compatible vocabulary, then clamp against the model's allowed set so
//! the server doesn't 400 on an unsupported effort.
//!
//! Native Rust port of `agent/lmstudio_reasoning.py`.

use serde_json::Value;

/// LM Studio accepts these top-level `reasoning_effort` values via its
/// OpenAI-compatible chat.completions endpoint.
const LM_VALID_EFFORTS: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh"];

/// Map a toggle/alias option onto the OpenAI-compatible request vocabulary.
///
/// Toggle-style models publish `allowed_options` as `["off","on"]` in
/// `/api/v1/models`; mirror the Python `_LM_EFFORT_ALIASES` dict.
fn lm_effort_alias(opt: &str) -> &str {
    match opt {
        "off" => "none",
        "on" => "medium",
        other => other,
    }
}

fn is_valid_effort(effort: &str) -> bool {
    LM_VALID_EFFORTS.contains(&effort)
}

/// Return the `reasoning_effort` string to send to LM Studio, or `None`.
///
/// `None` means "omit the field": the user picked a level the model can't
/// honor, so let LM Studio fall back to the model's declared default rather
/// than silently substituting a different effort. When `allowed_options` is
/// falsy (probe failed / empty), skip clamping and send the resolved effort
/// anyway.
///
/// `reasoning_config` is the dynamic user config dict (a JSON object); anything
/// that isn't an object is treated like Python's falsy/non-dict case. Pass
/// `None` for `allowed_options` when the model-capability probe failed; an
/// empty slice is likewise treated as "no clamping" (matching Python's falsy
/// list check).
pub fn resolve_lmstudio_effort(
    reasoning_config: Option<&Value>,
    allowed_options: Option<&[String]>,
) -> Option<String> {
    let mut effort = "medium".to_string();

    if let Some(Value::Object(cfg)) = reasoning_config {
        // `reasoning_config.get("enabled") is False`
        if cfg.get("enabled") == Some(&Value::Bool(false)) {
            effort = "none".to_string();
        } else {
            // `(reasoning_config.get("effort") or "").strip().lower()`
            let raw = cfg
                .get("effort")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_lowercase();
            let raw = lm_effort_alias(&raw);
            if is_valid_effort(raw) {
                effort = raw.to_string();
            }
        }
    }

    // `if allowed_options:` — None or empty list skips clamping.
    if let Some(opts) = allowed_options {
        if !opts.is_empty() {
            let allowed: Vec<&str> = opts.iter().map(|o| lm_effort_alias(o)).collect();
            if !allowed.contains(&effort.as_str()) {
                return None;
            }
        }
    }

    Some(effort)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn default_is_medium_when_no_config_and_no_options() {
        assert_eq!(resolve_lmstudio_effort(None, None).as_deref(), Some("medium"));
    }

    #[test]
    fn disabled_reasoning_maps_to_none_effort() {
        let cfg = json!({ "enabled": false });
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), None).as_deref(),
            Some("none")
        );
    }

    #[test]
    fn enabled_true_with_effort_low() {
        let cfg = json!({ "enabled": true, "effort": "low" });
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), None).as_deref(),
            Some("low")
        );
    }

    #[test]
    fn effort_is_trimmed_and_lowercased() {
        let cfg = json!({ "effort": "  HIGH  " });
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), None).as_deref(),
            Some("high")
        );
    }

    #[test]
    fn toggle_aliases_in_effort() {
        let cfg_off = json!({ "effort": "off" });
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg_off), None).as_deref(),
            Some("none")
        );
        let cfg_on = json!({ "effort": "on" });
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg_on), None).as_deref(),
            Some("medium")
        );
    }

    #[test]
    fn invalid_effort_falls_back_to_medium_default() {
        let cfg = json!({ "effort": "bogus" });
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), None).as_deref(),
            Some("medium")
        );
    }

    #[test]
    fn empty_effort_string_keeps_default() {
        let cfg = json!({ "effort": "" });
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), None).as_deref(),
            Some("medium")
        );
    }

    #[test]
    fn non_object_config_treated_as_no_config() {
        let cfg = json!("not a dict");
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), None).as_deref(),
            Some("medium")
        );
        let cfg_null = Value::Null;
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg_null), None).as_deref(),
            Some("medium")
        );
    }

    #[test]
    fn clamps_to_none_when_effort_not_in_allowed() {
        // graduated model: off/minimal/low -> none/minimal/low
        let cfg = json!({ "effort": "high" });
        let allowed = opts(&["off", "minimal", "low"]);
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), Some(&allowed)),
            None
        );
    }

    #[test]
    fn passes_when_effort_in_allowed_after_alias() {
        // toggle model: off/on -> none/medium; default medium is allowed via "on"
        let allowed = opts(&["off", "on"]);
        assert_eq!(
            resolve_lmstudio_effort(None, Some(&allowed)).as_deref(),
            Some("medium")
        );
        // explicit "off" effort -> none, which is allowed via "off"
        let cfg = json!({ "effort": "off" });
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), Some(&allowed)).as_deref(),
            Some("none")
        );
    }

    #[test]
    fn empty_allowed_options_skips_clamping() {
        let cfg = json!({ "effort": "xhigh" });
        let allowed: Vec<String> = Vec::new();
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), Some(&allowed)).as_deref(),
            Some("xhigh")
        );
    }

    #[test]
    fn disabled_clamped_against_toggle_off() {
        let cfg = json!({ "enabled": false });
        let allowed = opts(&["off", "on"]);
        // none is allowed (off alias), so it passes
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), Some(&allowed)).as_deref(),
            Some("none")
        );
    }

    #[test]
    fn enabled_false_takes_precedence_over_effort() {
        let cfg = json!({ "enabled": false, "effort": "high" });
        assert_eq!(
            resolve_lmstudio_effort(Some(&cfg), None).as_deref(),
            Some("none")
        );
    }
}
