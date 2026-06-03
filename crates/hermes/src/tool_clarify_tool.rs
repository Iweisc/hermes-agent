//! Clarify Tool Module - Interactive Clarifying Questions
//!
//! Allows the agent to present structured multiple-choice questions or
//! open-ended prompts to the user. In CLI mode, choices are navigable with
//! arrow keys. On messaging platforms, choices are rendered as a numbered list.
//!
//! The actual user-interaction logic lives in the platform layer (cli for CLI,
//! gateway for messaging). This module defines the schema, validation, and a
//! thin dispatcher that delegates to a platform-provided callback.
//!
//! Faithful port of `tools/clarify_tool.py`.

use serde_json::{json, Value};

/// Maximum number of predefined choices the agent can offer.
///
/// A 5th "Other (type your answer)" option is always appended by the UI.
pub const MAX_CHOICES: usize = 4;

/// Standard error payload, matching the registry's `tool_error` shape used by
/// other ported tools.
///
/// Mirrors `tools.registry.tool_error` (a JSON object with a single `error`
/// key). `serde_json` escapes only what JSON requires and preserves non-ASCII
/// characters verbatim, matching Python's `ensure_ascii=False` behaviour.
pub fn tool_error(message: &str) -> String {
    json!({ "error": message }).to_string()
}

/// Callback signature for the platform-provided UI interaction.
///
/// Mirrors the Python `callback(question, choices) -> str`. The callback
/// receives the (already trimmed) question and the validated choices (`None`
/// for open-ended questions) and returns either the user's raw response
/// (`Ok`) or an error message describing why input could not be obtained
/// (`Err`).
pub type ClarifyCallback<'a> = dyn FnMut(&str, Option<&[String]>) -> Result<String, String> + 'a;

/// Validate and normalise a list of choices.
///
/// - Each choice is converted to its trimmed string form.
/// - Blank choices are dropped.
/// - At most [`MAX_CHOICES`] are kept (extras truncated).
/// - An empty result becomes `None` (open-ended question).
///
/// `None` input passes straight through as `None`.
pub fn normalize_choices(choices: Option<&[String]>) -> Option<Vec<String>> {
    let choices = choices?;
    let mut cleaned: Vec<String> = choices
        .iter()
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    if cleaned.len() > MAX_CHOICES {
        cleaned.truncate(MAX_CHOICES);
    }
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Extract a list of string choices from a JSON value.
///
/// Mirrors the Python `args.get("choices")`: anything that is not a JSON array
/// is treated as a validation error (returns `Err`). A JSON `null` or absent
/// key (passed as `None`) means open-ended. Each element is coerced to its
/// string form (`str(c)` in Python) prior to trimming/validation downstream.
fn choices_from_json(value: Option<&Value>) -> Result<Option<Vec<String>>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => {
            let strs = items
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => {
                        // Match Python str(True) -> "True"
                        if *b { "True".to_string() } else { "False".to_string() }
                    }
                    Value::Null => "None".to_string(),
                    other => other.to_string(),
                })
                .collect();
            Ok(Some(strs))
        }
        Some(_) => Err("choices must be a list of strings.".to_string()),
    }
}

/// Ask the user a question, optionally with multiple-choice options.
///
/// - `question`: the question text to present.
/// - `choices`: up to [`MAX_CHOICES`] predefined answer choices. `None`
///   produces a purely open-ended question.
/// - `callback`: platform-provided function that handles the actual UI
///   interaction. `None` means the tool is unavailable in this context.
///
/// Returns a JSON string with the user's response (or an error payload),
/// matching the Python contract exactly.
pub fn clarify_tool(
    question: &str,
    choices: Option<&[String]>,
    callback: Option<&mut ClarifyCallback>,
) -> String {
    if question.trim().is_empty() {
        return tool_error("Question text is required.");
    }

    let question = question.trim().to_string();

    let choices = normalize_choices(choices);

    let callback = match callback {
        Some(cb) => cb,
        None => {
            return json!({
                "error": "Clarify tool is not available in this execution context."
            })
            .to_string();
        }
    };

    let user_response = match callback(&question, choices.as_deref()) {
        Ok(resp) => resp,
        Err(exc) => {
            return json!({ "error": format!("Failed to get user input: {exc}") }).to_string();
        }
    };

    // `choices_offered` is the (possibly `None`) validated list. `serde_json`
    // serialises `None` as JSON `null`, matching Python's `json.dumps(None)`.
    json!({
        "question": question,
        "choices_offered": choices,
        "user_response": user_response.trim(),
    })
    .to_string()
}

/// Convenience handler matching the registry calling convention.
///
/// Takes the raw tool arguments as a JSON value plus the optional UI callback,
/// validates `choices`, and dispatches to [`clarify_tool`].
pub fn handle(args: &Value, callback: Option<&mut ClarifyCallback>) -> String {
    let question = args.get("question").and_then(Value::as_str).unwrap_or("");

    let choices = match choices_from_json(args.get("choices")) {
        Ok(c) => c,
        Err(msg) => return tool_error(&msg),
    };

    clarify_tool(question, choices.as_deref(), callback)
}

/// Clarify tool has no external requirements -- always available.
pub fn check_clarify_requirements() -> bool {
    true
}

/// OpenAI function-calling schema for the clarify tool.
///
/// Byte-for-byte equivalent (modulo JSON key ordering) to `CLARIFY_SCHEMA` in
/// the Python module.
pub fn clarify_schema() -> Value {
    json!({
        "name": "clarify",
        "description":
            "Ask the user a question when you need clarification, feedback, or a \
decision before proceeding. Supports two modes:\n\n\
1. **Multiple choice** — provide up to 4 choices. The user picks one \
or types their own answer via a 5th 'Other' option.\n\
2. **Open-ended** — omit choices entirely. The user types a free-form \
response.\n\n\
Use this tool when:\n\
- The task is ambiguous and you need the user to choose an approach\n\
- You want post-task feedback ('How did that work out?')\n\
- You want to offer to save a skill or update memory\n\
- A decision has meaningful trade-offs the user should weigh in on\n\n\
Do NOT use this tool for simple yes/no confirmation of dangerous \
commands (the terminal tool handles that). Prefer making a reasonable \
default choice yourself when the decision is low-stakes.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to present to the user.",
                },
                "choices": {
                    "type": "array",
                    "items": { "type": "string" },
                    "maxItems": MAX_CHOICES,
                    "description":
                        "Up to 4 answer choices. Omit this parameter entirely to \
ask an open-ended question. When provided, the UI \
automatically appends an 'Other (type your answer)' option.",
                },
            },
            "required": ["question"],
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(s: &str) -> Value {
        serde_json::from_str(s).expect("valid json")
    }

    #[test]
    fn empty_question_errors() {
        let out = parse(&clarify_tool("   ", None, None));
        assert_eq!(out["error"], "Question text is required.");
    }

    #[test]
    fn missing_callback_errors() {
        let out = parse(&clarify_tool("Pick one", None, None));
        assert_eq!(
            out["error"],
            "Clarify tool is not available in this execution context."
        );
    }

    #[test]
    fn open_ended_success_trims_question_and_response() {
        let mut cb = |q: &str, c: Option<&[String]>| {
            assert_eq!(q, "What next?");
            assert!(c.is_none());
            Ok("  do the thing  ".to_string())
        };
        let out = parse(&clarify_tool(
            "  What next?  ",
            None,
            Some(&mut cb),
        ));
        assert_eq!(out["question"], "What next?");
        assert_eq!(out["choices_offered"], Value::Null);
        assert_eq!(out["user_response"], "do the thing");
    }

    #[test]
    fn choices_trimmed_and_blanks_dropped() {
        let captured = std::cell::RefCell::new(Vec::<String>::new());
        let mut cb = |_q: &str, c: Option<&[String]>| {
            if let Some(c) = c {
                *captured.borrow_mut() = c.to_vec();
            }
            Ok("A".to_string())
        };
        let choices = vec![
            " A ".to_string(),
            "".to_string(),
            "  ".to_string(),
            "B".to_string(),
        ];
        let out = parse(&clarify_tool("Q?", Some(&choices), Some(&mut cb)));
        assert_eq!(out["choices_offered"], json!(["A", "B"]));
        assert_eq!(*captured.borrow(), vec!["A".to_string(), "B".to_string()]);
    }

    #[test]
    fn choices_truncated_to_max() {
        let choices: Vec<String> = (1..=6).map(|i| format!("c{i}")).collect();
        let mut cb = |_q: &str, _c: Option<&[String]>| Ok("x".to_string());
        let out = parse(&clarify_tool("Q?", Some(&choices), Some(&mut cb)));
        assert_eq!(out["choices_offered"], json!(["c1", "c2", "c3", "c4"]));
    }

    #[test]
    fn empty_choices_become_open_ended() {
        let choices = vec!["".to_string(), "  ".to_string()];
        let mut cb = |_q: &str, c: Option<&[String]>| {
            assert!(c.is_none());
            Ok("free".to_string())
        };
        let out = parse(&clarify_tool("Q?", Some(&choices), Some(&mut cb)));
        assert_eq!(out["choices_offered"], Value::Null);
        assert_eq!(out["user_response"], "free");
    }

    #[test]
    fn callback_failure_is_wrapped() {
        let mut cb = |_q: &str, _c: Option<&[String]>| Err("kaboom".to_string());
        let out = parse(&clarify_tool("Q?", None, Some(&mut cb)));
        assert_eq!(out["error"], "Failed to get user input: kaboom");
    }

    #[test]
    fn handle_rejects_non_array_choices() {
        let args = json!({ "question": "Q?", "choices": "notalist" });
        let out = parse(&handle(&args, None));
        assert_eq!(out["error"], "choices must be a list of strings.");
    }

    #[test]
    fn handle_open_ended_from_json() {
        let args = json!({ "question": "Q?" });
        let mut cb = |_q: &str, c: Option<&[String]>| {
            assert!(c.is_none());
            Ok("ok".to_string())
        };
        let out = parse(&handle(&args, Some(&mut cb)));
        assert_eq!(out["user_response"], "ok");
    }

    #[test]
    fn handle_coerces_non_string_choice_elements() {
        let args = json!({ "question": "Q?", "choices": [1, true, "x"] });
        let mut cb = |_q: &str, _c: Option<&[String]>| Ok("done".to_string());
        let out = parse(&handle(&args, Some(&mut cb)));
        assert_eq!(out["choices_offered"], json!(["1", "True", "x"]));
    }

    #[test]
    fn requirements_always_true() {
        assert!(check_clarify_requirements());
    }

    #[test]
    fn schema_shape() {
        let s = clarify_schema();
        assert_eq!(s["name"], "clarify");
        assert_eq!(s["parameters"]["properties"]["choices"]["maxItems"], 4);
        assert_eq!(s["parameters"]["required"], json!(["question"]));
    }
}
