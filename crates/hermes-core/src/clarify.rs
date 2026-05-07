use serde_json::{Value, json};

use crate::tools::{ToolRuntime, tool_error, tool_result};

const MAX_CHOICES: usize = 4;

pub fn clarify_available() -> bool {
    true
}

pub fn clarify_schema() -> Value {
    json!({
        "name": "clarify",
        "description": "Ask the user a question when you need clarification, feedback, or a decision before proceeding. Supports multiple-choice with up to 4 predefined options, or open-ended questions when choices are omitted.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to present to the user."
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": MAX_CHOICES,
                    "description": "Up to 4 answer choices. Omit to ask an open-ended question."
                }
            },
            "required": ["question"]
        }
    })
}

pub fn handle_clarify(args: &Value, runtime: &ToolRuntime) -> String {
    let Some(question) = args
        .get("question")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return tool_error("Question text is required.");
    };

    let choices = match normalize_choices(args.get("choices")) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    let user_response = match runtime.clarify(question, choices.as_deref()) {
        Ok(value) => value.trim().to_string(),
        Err(error) => return tool_error(error),
    };

    tool_result(json!({
        "question": question,
        "choices_offered": choices,
        "user_response": user_response,
    }))
}

fn normalize_choices(value: Option<&Value>) -> Result<Option<Vec<String>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(items) = value.as_array() else {
        return Err("choices must be a list of strings.".to_string());
    };
    let mut trimmed = Vec::new();
    for item in items.iter().take(MAX_CHOICES) {
        let Some(text) = item.as_str() else {
            return Err("choices must be a list of strings.".to_string());
        };
        let text = text.trim();
        if !text.is_empty() {
            trimmed.push(text.to_string());
        }
    }
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clarify_returns_user_response_with_callback() {
        let runtime = ToolRuntime::default().with_clarify_callback(|question, choices| {
            assert_eq!(question, "Pick one");
            assert_eq!(choices.unwrap(), &["A".to_string(), "B".to_string()]);
            Ok("B".to_string())
        });
        let result = handle_clarify(
            &json!({
                "question": "Pick one",
                "choices": ["A", "B"],
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["user_response"], json!("B"));
    }

    #[test]
    fn clarify_errors_without_callback() {
        let runtime = ToolRuntime::default();
        let result = handle_clarify(&json!({"question": "Need input"}), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(parsed["error"].as_str().unwrap().contains("not available"));
    }
}
