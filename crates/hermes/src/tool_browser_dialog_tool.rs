//! Agent-facing tool: respond to a native JS dialog captured by the CDP supervisor.
//!
//! Native Rust port of `tools/browser_dialog_tool.py`.
//!
//! This tool is response-only — the agent first reads `pending_dialogs` from
//! `browser_snapshot` output, then calls `browser_dialog(action=...)` to accept
//! or dismiss the dialog.
//!
//! Gated on the same CDP availability check as `browser_cdp` so it only appears
//! when a CDP endpoint is reachable (Browserbase with a `connectUrl`, local
//! Chrome via `/browser connect`, or `browser.cdp_url` set in config).
//!
//! See `website/docs/developer-guide/browser-supervisor.md` for the full design.

use std::time::Duration;

use serde_json::{json, Value};

use crate::tool_browser_supervisor::supervisor_registry;

/// Default timeout used when waiting for the supervisor to acknowledge a dialog
/// response. The Python tool delegates the timeout entirely to the supervisor's
/// own command channel; we surface a generous default that matches the
/// supervisor's per-command waits.
pub const DEFAULT_RESPOND_TIMEOUT_SECS: u64 = 30;

/// JSON schema advertised to the model for the `browser_dialog` tool.
///
/// Built lazily (rather than as a `static`) because `serde_json::Value` is not
/// `const`-constructible. The structure mirrors `BROWSER_DIALOG_SCHEMA` in the
/// Python source byte-for-byte in its semantic fields.
pub fn browser_dialog_schema() -> Value {
    json!({
        "name": "browser_dialog",
        "description": concat!(
            "Respond to a native JavaScript dialog (alert / confirm / prompt / ",
            "beforeunload) that is currently blocking the page.\n\n",
            "**Workflow:** call ``browser_snapshot`` first — if a dialog is open, ",
            "it appears in the ``pending_dialogs`` field with ``id``, ``type``, ",
            "and ``message``. Then call this tool with ``action='accept'`` or ",
            "``action='dismiss'``.\n\n",
            "**Prompt dialogs:** pass ``prompt_text`` to supply the response ",
            "string. Ignored for alert/confirm/beforeunload.\n\n",
            "**Multiple dialogs:** if more than one dialog is queued (rare — ",
            "happens when a second dialog fires while the first is still open), ",
            "pass ``dialog_id`` from the snapshot to disambiguate.\n\n",
            "**Availability:** only present when a CDP-capable backend is ",
            "attached — Browserbase sessions, local Chrome via ",
            "``/browser connect``, or ``browser.cdp_url`` in config.yaml. ",
            "Not available on Camofox (REST-only) or the default Playwright ",
            "local browser (CDP port is hidden)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["accept", "dismiss"],
                    "description": concat!(
                        "'accept' clicks OK / returns the prompt text. ",
                        "'dismiss' clicks Cancel / returns null from prompt(). ",
                        "For ``beforeunload`` dialogs: 'accept' allows the ",
                        "navigation, 'dismiss' keeps the page."
                    ),
                },
                "prompt_text": {
                    "type": "string",
                    "description": concat!(
                        "Response string for a ``prompt()`` dialog. Ignored for ",
                        "other dialog types. Defaults to empty string."
                    ),
                },
                "dialog_id": {
                    "type": "string",
                    "description": concat!(
                        "Specific dialog to respond to, from ",
                        "``browser_snapshot.pending_dialogs[].id``. Required ",
                        "only when multiple dialogs are queued."
                    ),
                },
            },
            "required": ["action"],
        },
    })
}

/// Respond to a pending dialog on the active task's CDP supervisor.
///
/// Returns a JSON string (matching the Python `json.dumps(...)` contract):
/// * `{"success": true, "action": ..., "dialog": {...}}` on success,
/// * `{"success": false, "error": "..."}` on any failure.
///
/// `task_id` of `None` resolves to `"default"`, mirroring the Python default.
pub fn browser_dialog(
    action: &str,
    prompt_text: Option<&str>,
    dialog_id: Option<&str>,
    task_id: Option<&str>,
) -> String {
    serde_json::to_string(&browser_dialog_value(
        action,
        prompt_text,
        dialog_id,
        task_id,
    ))
    .unwrap_or_else(|_| {
        // Should be impossible — the value is always plain JSON. Provide a
        // best-effort fallback rather than panicking in a tool handler.
        "{\"success\":false,\"error\":\"failed to serialize response\"}".to_string()
    })
}

/// Same as [`browser_dialog`] but returns the structured [`Value`] instead of a
/// serialized string. Useful for callers that re-embed the result in a larger
/// JSON document, and for tests.
pub fn browser_dialog_value(
    action: &str,
    prompt_text: Option<&str>,
    dialog_id: Option<&str>,
    task_id: Option<&str>,
) -> Value {
    let effective_task_id = task_id.filter(|t| !t.is_empty()).unwrap_or("default");

    let supervisor = match supervisor_registry().get(effective_task_id) {
        Some(s) => s,
        None => {
            return json!({
                "success": false,
                "error": concat!(
                    "No CDP supervisor is attached to this task. Either the ",
                    "browser backend doesn't expose CDP (Camofox, default ",
                    "Playwright) or no browser session has been started yet. ",
                    "Call browser_navigate or /browser connect first."
                ),
            });
        }
    };

    let result = supervisor.respond_to_dialog(
        action,
        prompt_text,
        dialog_id,
        Duration::from_secs(DEFAULT_RESPOND_TIMEOUT_SECS),
    );

    if result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        let dialog = result
            .get("dialog")
            .cloned()
            .unwrap_or_else(|| json!({}));
        json!({
            "success": true,
            "action": action,
            "dialog": dialog,
        })
    } else {
        let error = result
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        json!({"success": false, "error": error})
    }
}

/// Handler shim matching the registry's `(args, task_id)` calling convention.
///
/// Mirrors the Python lambda registered for `browser_dialog`: it pulls
/// `action` / `prompt_text` / `dialog_id` from the args object and the
/// `task_id` from the dispatch kwargs.
pub fn browser_dialog_handler(args: &Value, task_id: Option<&str>) -> String {
    let action = args.get("action").and_then(Value::as_str).unwrap_or("");
    let prompt_text = args.get("prompt_text").and_then(Value::as_str);
    let dialog_id = args.get("dialog_id").and_then(Value::as_str);
    browser_dialog(action, prompt_text, dialog_id, task_id)
}

/// Resolver for the static CDP override URL.
///
/// In Python the availability gate (`_browser_cdp_check`) imports
/// `_get_cdp_override` / `check_browser_requirements` from `browser_tool`,
/// which are not yet ported. To avoid blocking, the default resolver reads the
/// `BROWSER_CDP_URL` environment variable (set by `/browser connect`) and is
/// considered satisfied when it is a non-empty string. Callers with access to
/// fully-ported browser config may pass their own closure to
/// [`browser_dialog_check_with`].
pub type CdpOverrideResolver<'a> = dyn Fn() -> Option<String> + 'a;

/// Default CDP-override resolver: reads `BROWSER_CDP_URL` from the environment.
pub fn default_cdp_override() -> Option<String> {
    match std::env::var("BROWSER_CDP_URL") {
        Ok(v) if !v.trim().is_empty() => Some(v),
        _ => None,
    }
}

/// Availability gate, mirroring Python's `_browser_dialog_check`, which simply
/// delegates to `browser_cdp`'s `_browser_cdp_check`.
///
/// Kept identical in spirit so `browser_cdp` and `browser_dialog` appear and
/// disappear together: the tool is offered only when a reachable CDP override
/// URL is configured right now.
pub fn browser_dialog_check() -> bool {
    browser_dialog_check_with(&default_cdp_override)
}

/// As [`browser_dialog_check`] but with an injectable override resolver, so
/// callers that have ported the full browser config can supply a precise check.
pub fn browser_dialog_check_with(resolver: &CdpOverrideResolver<'_>) -> bool {
    resolver().map(|u| !u.trim().is_empty()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Value {
        serde_json::from_str(s).expect("handler must return valid JSON")
    }

    #[test]
    fn schema_has_expected_shape() {
        let schema = browser_dialog_schema();
        assert_eq!(schema["name"], "browser_dialog");
        assert_eq!(
            schema["parameters"]["required"],
            json!(["action"])
        );
        let enum_vals = &schema["parameters"]["properties"]["action"]["enum"];
        assert_eq!(*enum_vals, json!(["accept", "dismiss"]));
        // prompt_text + dialog_id are advertised but optional.
        assert!(schema["parameters"]["properties"]["prompt_text"].is_object());
        assert!(schema["parameters"]["properties"]["dialog_id"].is_object());
    }

    #[test]
    fn no_supervisor_returns_actionable_error() {
        // No supervisor registered for this unique task id → failure with the
        // "No CDP supervisor is attached" guidance message.
        let out = browser_dialog(
            "accept",
            None,
            None,
            Some("nonexistent-task-for-dialog-test"),
        );
        let v = parse(&out);
        assert_eq!(v["success"], json!(false));
        let err = v["error"].as_str().unwrap();
        assert!(err.contains("No CDP supervisor is attached"));
        assert!(err.contains("browser_navigate"));
    }

    #[test]
    fn missing_task_id_defaults_and_still_fails_cleanly() {
        // task_id None resolves to "default"; in a unit-test process there is
        // no supervisor, so we expect the same clean failure path.
        let v = browser_dialog_value("dismiss", None, None, None);
        assert_eq!(v["success"], json!(false));
        assert!(v["error"].is_string());
    }

    #[test]
    fn empty_task_id_treated_as_default() {
        let v = browser_dialog_value("accept", None, None, Some(""));
        assert_eq!(v["success"], json!(false));
        assert!(v["error"]
            .as_str()
            .unwrap()
            .contains("No CDP supervisor is attached"));
    }

    #[test]
    fn handler_extracts_args() {
        let args = json!({
            "action": "accept",
            "prompt_text": "hello",
            "dialog_id": "d-1",
        });
        let out = browser_dialog_handler(&args, Some("some-task"));
        let v = parse(&out);
        // No supervisor → failure, but the call path is exercised end-to-end.
        assert_eq!(v["success"], json!(false));
    }

    #[test]
    fn handler_missing_action_uses_empty_string() {
        let args = json!({});
        let out = browser_dialog_handler(&args, None);
        let v = parse(&out);
        assert_eq!(v["success"], json!(false));
    }

    #[test]
    fn check_uses_resolver() {
        // Resolver returns None → not available.
        let none_resolver = || None::<String>;
        assert!(!browser_dialog_check_with(&none_resolver));

        // Empty / whitespace URL → not available.
        let blank = || Some("   ".to_string());
        assert!(!browser_dialog_check_with(&blank));

        // Real URL → available.
        let real = || Some("ws://127.0.0.1:9222/devtools".to_string());
        assert!(browser_dialog_check_with(&real));
    }

    #[test]
    fn default_cdp_override_reads_env() {
        let key = "BROWSER_CDP_URL";
        let prev = std::env::var(key).ok();

        unsafe {
            std::env::remove_var(key);
        }
        assert!(default_cdp_override().is_none());

        unsafe {
            std::env::set_var(key, "ws://localhost:9222");
        }
        assert_eq!(
            default_cdp_override().as_deref(),
            Some("ws://localhost:9222")
        );

        // Restore prior environment to avoid cross-test contamination.
        unsafe {
            match prev {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
