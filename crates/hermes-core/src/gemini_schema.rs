//! Helpers for translating OpenAI-style tool schemas to Gemini's schema subset.
//!
//! Gemini's `FunctionDeclaration.parameters` field accepts the `Schema`
//! object, which is only a subset of OpenAPI 3.0 / JSON Schema. Strip fields
//! outside that subset before sending Hermes tool schemas to Google.
//!
//! Ported faithfully from `agent/gemini_schema.py`.

use serde_json::{Map, Value};

/// Keys permitted in Gemini's `Schema` object subset.
const GEMINI_SCHEMA_ALLOWED_KEYS: &[&str] = &[
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "enum",
    "maxItems",
    "minItems",
    "properties",
    "required",
    "minProperties",
    "maxProperties",
    "minLength",
    "maxLength",
    "pattern",
    "example",
    "anyOf",
    "propertyOrdering",
    "default",
    "items",
    "minimum",
    "maximum",
];

fn is_allowed_key(key: &str) -> bool {
    GEMINI_SCHEMA_ALLOWED_KEYS.contains(&key)
}

/// Return a Gemini-compatible copy of a tool parameter schema.
///
/// Hermes tool schemas are OpenAI-flavored JSON Schema and may contain keys
/// such as `$schema` or `additionalProperties` that Google's Gemini `Schema`
/// object rejects. This helper preserves the documented Gemini subset and
/// recursively sanitizes nested `properties` / `items` / `anyOf` definitions.
///
/// Mirrors Python's `sanitize_gemini_schema`: non-object input yields an empty
/// object.
pub fn sanitize_gemini_schema(schema: &Value) -> Value {
    let obj = match schema.as_object() {
        Some(obj) => obj,
        None => return Value::Object(Map::new()),
    };

    let mut cleaned: Map<String, Value> = Map::new();

    for (key, value) in obj {
        if !is_allowed_key(key) {
            continue;
        }

        match key.as_str() {
            "properties" => {
                // Skip entirely when `properties` is not a dict (matches Python `continue`).
                let props_obj = match value.as_object() {
                    Some(props_obj) => props_obj,
                    None => continue,
                };
                let mut props: Map<String, Value> = Map::new();
                // JSON object keys are always strings, so the `isinstance(prop_name, str)`
                // guard in Python is always satisfied here.
                for (prop_name, prop_schema) in props_obj {
                    props.insert(prop_name.clone(), sanitize_gemini_schema(prop_schema));
                }
                cleaned.insert(key.clone(), Value::Object(props));
            }
            "items" => {
                cleaned.insert(key.clone(), sanitize_gemini_schema(value));
            }
            "anyOf" => {
                // Skip entirely when `anyOf` is not a list (matches Python `continue`).
                let arr = match value.as_array() {
                    Some(arr) => arr,
                    None => continue,
                };
                let sanitized: Vec<Value> = arr
                    .iter()
                    .filter(|item| item.is_object())
                    .map(sanitize_gemini_schema)
                    .collect();
                cleaned.insert(key.clone(), Value::Array(sanitized));
            }
            _ => {
                cleaned.insert(key.clone(), value.clone());
            }
        }
    }

    // Gemini's Schema validator requires every `enum` entry to be a string,
    // even when the parent `type` is `integer` / `number` / `boolean`.
    // OpenAI / OpenRouter / Anthropic accept typed enums (e.g. Discord's
    // `auto_archive_duration: {type: integer, enum: [60, 1440, 4320, 10080]}`),
    // so we only drop the `enum` when it would collide with Gemini's rule.
    // Keeping `type: integer` plus the human-readable description gives the
    // model enough guidance; the tool handler still validates the value.
    let drop_enum = {
        let enum_is_list = cleaned.get("enum").map(Value::is_array).unwrap_or(false);
        let type_collides = matches!(
            cleaned.get("type").and_then(Value::as_str),
            Some("integer") | Some("number") | Some("boolean")
        );
        if enum_is_list && type_collides {
            cleaned
                .get("enum")
                .and_then(Value::as_array)
                .map(|arr| arr.iter().any(|item| !item.is_string()))
                .unwrap_or(false)
        } else {
            false
        }
    };
    if drop_enum {
        cleaned.remove("enum");
    }

    Value::Object(cleaned)
}

/// Normalize tool parameters to a valid Gemini object schema.
///
/// Mirrors Python's `sanitize_gemini_tool_parameters`: when sanitizing yields
/// an empty object, fall back to `{"type": "object", "properties": {}}`.
pub fn sanitize_gemini_tool_parameters(parameters: &Value) -> Value {
    let cleaned = sanitize_gemini_schema(parameters);
    let is_empty = cleaned
        .as_object()
        .map(|obj| obj.is_empty())
        .unwrap_or(true);
    if is_empty {
        let mut fallback = Map::new();
        fallback.insert("type".to_string(), Value::String("object".to_string()));
        fallback.insert("properties".to_string(), Value::Object(Map::new()));
        return Value::Object(fallback);
    }
    cleaned
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn non_dict_input_yields_empty_object() {
        assert_eq!(sanitize_gemini_schema(&json!("hello")), json!({}));
        assert_eq!(sanitize_gemini_schema(&json!(42)), json!({}));
        assert_eq!(sanitize_gemini_schema(&json!([1, 2, 3])), json!({}));
        assert_eq!(sanitize_gemini_schema(&json!(null)), json!({}));
    }

    #[test]
    fn strips_disallowed_keys() {
        let input = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "additionalProperties": false,
            "type": "object",
            "description": "a thing"
        });
        let out = sanitize_gemini_schema(&input);
        assert_eq!(out, json!({"type": "object", "description": "a thing"}));
    }

    #[test]
    fn recurses_into_properties() {
        let input = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "additionalProperties": true},
                "age": {"type": "integer", "$comment": "drop me"}
            }
        });
        let out = sanitize_gemini_schema(&input);
        assert_eq!(
            out,
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            })
        );
    }

    #[test]
    fn non_dict_properties_skipped() {
        let input = json!({"type": "object", "properties": "nope"});
        let out = sanitize_gemini_schema(&input);
        assert_eq!(out, json!({"type": "object"}));
    }

    #[test]
    fn recurses_into_items() {
        let input = json!({
            "type": "array",
            "items": {"type": "string", "additionalProperties": false}
        });
        let out = sanitize_gemini_schema(&input);
        assert_eq!(out, json!({"type": "array", "items": {"type": "string"}}));
    }

    #[test]
    fn any_of_filters_non_dict_and_recurses() {
        let input = json!({
            "anyOf": [
                {"type": "string", "$schema": "x"},
                "not-a-dict",
                {"type": "integer"},
                123
            ]
        });
        let out = sanitize_gemini_schema(&input);
        assert_eq!(
            out,
            json!({"anyOf": [{"type": "string"}, {"type": "integer"}]})
        );
    }

    #[test]
    fn non_list_any_of_skipped() {
        let input = json!({"type": "string", "anyOf": {"foo": "bar"}});
        let out = sanitize_gemini_schema(&input);
        assert_eq!(out, json!({"type": "string"}));
    }

    #[test]
    fn typed_enum_with_non_string_entries_dropped() {
        let input = json!({
            "type": "integer",
            "enum": [60, 1440, 4320, 10080],
            "description": "archive duration"
        });
        let out = sanitize_gemini_schema(&input);
        assert_eq!(
            out,
            json!({"type": "integer", "description": "archive duration"})
        );
    }

    #[test]
    fn string_enum_kept() {
        let input = json!({
            "type": "string",
            "enum": ["a", "b", "c"]
        });
        let out = sanitize_gemini_schema(&input);
        assert_eq!(out, json!({"type": "string", "enum": ["a", "b", "c"]}));
    }

    #[test]
    fn typed_enum_all_strings_kept() {
        // type integer but enum entries are strings -> no collision, keep enum.
        let input = json!({
            "type": "integer",
            "enum": ["1", "2"]
        });
        let out = sanitize_gemini_schema(&input);
        assert_eq!(out, json!({"type": "integer", "enum": ["1", "2"]}));
    }

    #[test]
    fn typed_enum_mixed_dropped() {
        let input = json!({
            "type": "number",
            "enum": ["1", 2, "3"]
        });
        let out = sanitize_gemini_schema(&input);
        assert_eq!(out, json!({"type": "number"}));
    }

    #[test]
    fn boolean_typed_enum_with_bool_entries_dropped() {
        let input = json!({"type": "boolean", "enum": [true, false]});
        let out = sanitize_gemini_schema(&input);
        assert_eq!(out, json!({"type": "boolean"}));
    }

    #[test]
    fn tool_parameters_empty_fallback() {
        assert_eq!(
            sanitize_gemini_tool_parameters(&json!("not-a-dict")),
            json!({"type": "object", "properties": {}})
        );
        assert_eq!(
            sanitize_gemini_tool_parameters(&json!({})),
            json!({"type": "object", "properties": {}})
        );
        // Object with only disallowed keys becomes empty -> fallback.
        assert_eq!(
            sanitize_gemini_tool_parameters(&json!({"$schema": "x"})),
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn tool_parameters_passthrough() {
        let input = json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        });
        let out = sanitize_gemini_tool_parameters(&input);
        assert_eq!(out, input);
    }
}
