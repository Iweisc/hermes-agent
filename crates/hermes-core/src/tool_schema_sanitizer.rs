//! Sanitize tool JSON schemas for broad LLM-backend compatibility.
//!
//! Some local inference backends (notably llama.cpp's `json-schema-to-grammar`
//! converter used to build GBNF tool-call parsers) are strict about what JSON
//! Schema shapes they accept. Schemas that OpenAI / Anthropic / most cloud
//! providers silently accept can make llama.cpp fail the entire request with:
//!
//! ```text
//! HTTP 400: Unable to generate parser for this template.
//! Automatic parser generation failed: JSON schema conversion failed:
//! Unrecognized schema: "object"
//! ```
//!
//! The failure modes seen in the wild:
//!
//! * `{"type": "object"}` with no `properties` — rejected as a node the
//!   grammar generator can't constrain.
//! * A schema value that is the bare string `"object"` instead of a dict
//!   (malformed MCP server output, e.g. `additionalProperties: "object"`).
//! * `"type": ["string", "null"]` array types — many converters only accept
//!   single-string `type`.
//! * `anyOf` / `oneOf` unions whose only purpose is to permit `null` for
//!   optional fields (common Pydantic/MCP shape). Anthropic rejects these at
//!   the top of `input_schema`; collapse them to the non-null branch.
//! * Unconstrained `additionalProperties` on objects with empty properties.
//!
//! This module walks the final tool schema tree (after MCP-level normalization
//! and any per-tool dynamic rebuilds) and fixes the known-hostile constructs
//! on a deep copy. It is intentionally conservative: it only modifies shapes
//! the LLM backend couldn't use anyway.
//!
//! This is a faithful native port of `tools/schema_sanitizer.py`.

use serde_json::{Map, Value};

/// Return a copy of `tools` with each tool's parameter schema sanitized.
///
/// Input is an OpenAI-format tool list:
/// `[{"type": "function", "function": {"name": ..., "parameters": {...}}}]`
///
/// The returned list is a deep copy — callers can safely mutate it without
/// affecting the original registry entries.
pub fn sanitize_tool_schemas(tools: &[Value]) -> Vec<Value> {
    if tools.is_empty() {
        return tools.to_vec();
    }
    tools.iter().map(sanitize_single_tool).collect()
}

/// Deep-copy and sanitize a single OpenAI-format tool entry.
fn sanitize_single_tool(tool: &Value) -> Value {
    let mut out = tool.clone();
    // fn = out.get("function") if isinstance(out, dict) else None
    let out_obj = match out.as_object_mut() {
        Some(m) => m,
        None => return out,
    };
    let fn_is_obj = matches!(out_obj.get("function"), Some(Value::Object(_)));
    if !fn_is_obj {
        return out;
    }

    // Operate on a detached function object then write it back.
    let mut fn_obj = match out_obj.remove("function") {
        Some(Value::Object(m)) => m,
        // Unreachable given the check above, but be safe.
        other => {
            if let Some(v) = other {
                out_obj.insert("function".to_string(), v);
            }
            return out;
        }
    };

    let params = fn_obj.get("parameters");
    // Missing / non-dict parameters → substitute the minimal valid shape.
    if !matches!(params, Some(Value::Object(_))) {
        fn_obj.insert("parameters".to_string(), default_object_schema());
        out_obj.insert("function".to_string(), Value::Object(fn_obj));
        return out;
    }

    // path = fn.get("name", "<tool>")
    let path = match fn_obj.get("name") {
        Some(Value::String(s)) => s.clone(),
        _ => "<tool>".to_string(),
    };
    let params_val = fn_obj.remove("parameters").unwrap();
    let sanitized = sanitize_node(params_val, &path);

    // After recursion, guarantee the top-level is an object with properties.
    let top = match sanitized {
        Value::Object(mut m) => {
            if m.get("type") != Some(&Value::String("object".to_string())) {
                m.insert("type".to_string(), Value::String("object".to_string()));
            }
            let props_is_obj = matches!(m.get("properties"), Some(Value::Object(_)));
            if !props_is_obj {
                m.insert("properties".to_string(), Value::Object(Map::new()));
            }
            Value::Object(m)
        }
        _ => default_object_schema(),
    };

    // Final pass: collapse nullable anyOf/oneOf unions that the recursive
    // sanitizer leaves intact (it only handles the array-form
    // `type: [X, "null"]`). Keep the `nullable: true` hint so runtime argument
    // coercion can still map a model-emitted `"null"` string to None.
    let top = strip_nullable_unions(top, true);
    fn_obj.insert("parameters".to_string(), top);
    out_obj.insert("function".to_string(), Value::Object(fn_obj));
    out
}

fn default_object_schema() -> Value {
    let mut m = Map::new();
    m.insert("type".to_string(), Value::String("object".to_string()));
    m.insert("properties".to_string(), Value::Object(Map::new()));
    Value::Object(m)
}

/// Collapse `anyOf` / `oneOf` nullable unions to the non-null branch.
///
/// MCP / Pydantic optional fields commonly arrive as:
///
/// ```text
/// {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}
/// ```
///
/// Anthropic's tool input-schema validator rejects the null branch. Tool
/// optionality is already represented by the parent object's `required` array,
/// so we collapse the union to the single non-null variant.
///
/// Metadata (`title`, `description`, `default`, `examples`) on the outer union
/// node is carried over to the replacement variant.
///
/// When `keep_nullable_hint` is true, sets `nullable: true` on the replacement
/// to preserve the "this field may be None" signal for downstream consumers.
pub fn strip_nullable_unions(schema: Value, keep_nullable_hint: bool) -> Value {
    match schema {
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| strip_nullable_unions(item, keep_nullable_hint))
                .collect(),
        ),
        Value::Object(obj) => {
            // First recurse into every value.
            let mut stripped: Map<String, Value> = Map::new();
            for (k, v) in obj.into_iter() {
                stripped.insert(k, strip_nullable_unions(v, keep_nullable_hint));
            }

            for key in ["anyOf", "oneOf"] {
                let variants = match stripped.get(key) {
                    Some(Value::Array(a)) => a,
                    _ => continue,
                };
                let total = variants.len();
                let non_null: Vec<&Value> = variants
                    .iter()
                    .filter(|item| !is_null_type_object(item))
                    .collect();

                // Only collapse when we actually dropped a null branch AND
                // exactly one non-null branch survives.
                if non_null.len() == 1 && non_null.len() != total {
                    let mut replacement: Map<String, Value> = match non_null[0] {
                        Value::Object(m) => m.clone(),
                        _ => Map::new(),
                    };
                    if keep_nullable_hint {
                        replacement
                            .entry("nullable".to_string())
                            .or_insert(Value::Bool(true));
                    }
                    for meta_key in ["title", "description", "default", "examples"] {
                        if let Some(v) = stripped.get(meta_key) {
                            if !replacement.contains_key(meta_key) {
                                replacement.insert(meta_key.to_string(), v.clone());
                            }
                        }
                    }
                    return strip_nullable_unions(Value::Object(replacement), keep_nullable_hint);
                }
            }
            Value::Object(stripped)
        }
        other => other,
    }
}

/// True for `{"type": "null"}` objects (a null-branch of a union).
fn is_null_type_object(item: &Value) -> bool {
    matches!(item, Value::Object(m) if m.get("type") == Some(&Value::String("null".to_string())))
}

const BARE_SCHEMA_TYPES: &[&str] = &[
    "object", "string", "number", "integer", "boolean", "array", "null",
];

/// Recursively sanitize a JSON-Schema fragment.
///
/// - Replaces bare-string schema values (`"object"`, `"string"`, ...) with
///   `{"type": <value>}` so downstream consumers see a dict.
/// - Injects `properties: {}` into object-typed nodes missing it.
/// - Normalizes `type: [X, "null"]` arrays to single `type: X` (keeping
///   `nullable: true` as a hint).
/// - Recurses into `properties`, `items`, `additionalProperties`, `anyOf`,
///   `oneOf`, `allOf`, and `$defs` / `definitions`.
fn sanitize_node(node: Value, path: &str) -> Value {
    match node {
        // Malformed: the schema position holds a bare string like "object".
        Value::String(s) => {
            if BARE_SCHEMA_TYPES.contains(&s.as_str()) {
                log::debug!(
                    "schema_sanitizer[{}]: replacing bare-string schema {:?} with {{'type': {:?}}}",
                    path,
                    s,
                    s
                );
                if s == "object" {
                    default_object_schema()
                } else {
                    let mut m = Map::new();
                    m.insert("type".to_string(), Value::String(s));
                    Value::Object(m)
                }
            } else {
                // Any other stray string is not a schema — drop it.
                log::debug!(
                    "schema_sanitizer[{}]: replacing non-schema string {:?} with empty object schema",
                    path,
                    s
                );
                default_object_schema()
            }
        }
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .enumerate()
                .map(|(i, item)| sanitize_node(item, &format!("{}[{}]", path, i)))
                .collect(),
        ),
        Value::Object(node) => sanitize_object(node, path),
        other => other,
    }
}

fn sanitize_object(node: Map<String, Value>, path: &str) -> Value {
    let mut out: Map<String, Value> = Map::new();

    for (key, value) in node.into_iter() {
        // type: [X, "null"] → type: X
        if key == "type" {
            if let Value::Array(types) = &value {
                let non_null: Vec<&Value> = types
                    .iter()
                    .filter(|t| **t != Value::String("null".to_string()))
                    .collect();
                if non_null.len() == 1 {
                    if let Value::String(s) = non_null[0] {
                        out.insert("type".to_string(), Value::String(s.clone()));
                        if types.iter().any(|t| *t == Value::String("null".to_string())) {
                            out.entry("nullable".to_string()).or_insert(Value::Bool(true));
                        }
                        continue;
                    }
                }
                // Fallback: pick the first string type that isn't "null".
                let first_str = types.iter().find_map(|t| match t {
                    Value::String(s) if s != "null" => Some(s.clone()),
                    _ => None,
                });
                if let Some(s) = first_str {
                    out.insert("type".to_string(), Value::String(s));
                    continue;
                }
                // All-null or empty list → treat as object.
                out.insert("type".to_string(), Value::String("object".to_string()));
                continue;
            }
            // Non-array type: pass through via the generic branch below.
            out.insert(key, value);
            continue;
        }

        match key.as_str() {
            "properties" | "$defs" | "definitions" if value.is_object() => {
                let sub = match value {
                    Value::Object(m) => m,
                    _ => unreachable!(),
                };
                let mut new_map = Map::new();
                for (sub_k, sub_v) in sub.into_iter() {
                    let sub_path = format!("{}.{}.{}", path, key, sub_k);
                    new_map.insert(sub_k, sanitize_node(sub_v, &sub_path));
                }
                out.insert(key, Value::Object(new_map));
            }
            "items" | "additionalProperties" => {
                if value.is_boolean() {
                    // Keep bool additionalProperties / items as-is.
                    out.insert(key, value);
                } else {
                    let sub_path = format!("{}.{}", path, key);
                    out.insert(key, sanitize_node(value, &sub_path));
                }
            }
            "anyOf" | "oneOf" | "allOf" if value.is_array() => {
                let arr = match value {
                    Value::Array(a) => a,
                    _ => unreachable!(),
                };
                let new_arr: Vec<Value> = arr
                    .into_iter()
                    .enumerate()
                    .map(|(i, item)| {
                        let sub_path = format!("{}.{}[{}]", path, key, i);
                        sanitize_node(item, &sub_path)
                    })
                    .collect();
                out.insert(key, Value::Array(new_arr));
            }
            "required" | "enum" | "examples" => {
                // Schema "sibling" keywords whose values are NOT schemas.
                // Pass through unchanged (deep copy preserved via clone).
                out.insert(key, value);
            }
            _ => {
                if value.is_object() || value.is_array() {
                    let sub_path = format!("{}.{}", path, key);
                    out.insert(key, sanitize_node(value, &sub_path));
                } else {
                    out.insert(key, value);
                }
            }
        }
    }

    let is_object_type = out.get("type") == Some(&Value::String("object".to_string()));

    // Object nodes without properties: inject empty properties dict.
    if is_object_type && !matches!(out.get("properties"), Some(Value::Object(_))) {
        out.insert("properties".to_string(), Value::Object(Map::new()));
    }

    // Prune `required` entries that don't exist in properties.
    if is_object_type {
        if let Some(Value::Array(required)) = out.get("required") {
            let props: Map<String, Value> = match out.get("properties") {
                Some(Value::Object(m)) => m.clone(),
                _ => Map::new(),
            };
            let original_len = required.len();
            let valid: Vec<Value> = required
                .iter()
                .filter(|r| matches!(r, Value::String(s) if props.contains_key(s)))
                .cloned()
                .collect();
            if valid.is_empty() {
                out.remove("required");
            } else if valid.len() != original_len {
                out.insert("required".to_string(), Value::Array(valid));
            }
        }
    }

    Value::Object(out)
}

// =============================================================================
// Reactive strip — only invoked when llama.cpp rejects a schema
// =============================================================================

const STRIP_ON_RECOVERY_KEYS: &[&str] = &["pattern", "format"];

/// Strip `pattern` and `format` JSON Schema keywords from tool schemas.
///
/// This is a *reactive* sanitizer invoked only when llama.cpp's
/// `json-schema-to-grammar` converter has rejected a tool schema with an HTTP
/// 400 grammar-parse error. llama.cpp's regex engine supports only a small
/// subset of ECMAScript regex — it rejects escape classes like `\d`, `\w`,
/// `\s` and most `format` values. Cloud providers accept these keywords fine
/// and rely on them as prompting hints, so we keep them in the default schema
/// and only strip on demand.
///
/// The strip operates on a sibling of `type` (so schema keywords are removed) —
/// a property literally *named* `pattern` is not affected because property
/// names live in the `properties` dict, not as siblings of `type`.
///
/// Returns `(tools, stripped_count)` — the (mutated) list plus a count of how
/// many `pattern`/`format` keywords were removed across all tools.
pub fn strip_pattern_and_format(mut tools: Vec<Value>) -> (Vec<Value>, usize) {
    if tools.is_empty() {
        return (tools, 0);
    }

    let mut stripped: usize = 0;

    for tool in tools.iter_mut() {
        let params = tool
            .as_object_mut()
            .and_then(|m| m.get_mut("function"))
            .and_then(|f| f.as_object_mut())
            .and_then(|f| f.get_mut("parameters"));
        if let Some(p) = params {
            if p.is_object() {
                walk_strip(p, &mut stripped);
            }
        }
    }

    if stripped > 0 {
        log::info!(
            "schema_sanitizer: stripped {} pattern/format keyword(s) from tool schemas (llama.cpp grammar-parse recovery)",
            stripped
        );
    }
    (tools, stripped)
}

fn walk_strip(node: &mut Value, stripped: &mut usize) {
    match node {
        Value::Object(map) => {
            // Only strip as a sibling of `type` — i.e. when this node is itself
            // a schema. This avoids stripping literal property keys named
            // "pattern" because those live inside a `properties` dict.
            let is_schema_node = map.contains_key("type")
                || map.contains_key("anyOf")
                || map.contains_key("oneOf")
                || map.contains_key("allOf");

            // Collect keys to recurse into / strip (Python iterates list(keys)).
            let keys: Vec<String> = map.keys().cloned().collect();
            for key in keys {
                if is_schema_node && STRIP_ON_RECOVERY_KEYS.contains(&key.as_str()) {
                    map.remove(&key);
                    *stripped += 1;
                    continue;
                }
                if let Some(child) = map.get_mut(&key) {
                    walk_strip(child, stripped);
                }
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                walk_strip(item, stripped);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(params: Value) -> Value {
        json!({"type": "function", "function": {"name": "f", "parameters": params}})
    }

    fn sanitized_params(params: Value) -> Value {
        let out = sanitize_tool_schemas(&[tool(params)]);
        out[0]["function"]["parameters"].clone()
    }

    #[test]
    fn empty_tools_passthrough() {
        let tools: Vec<Value> = vec![];
        assert_eq!(sanitize_tool_schemas(&tools), tools);
    }

    #[test]
    fn missing_parameters_become_default_object() {
        let t = json!({"type": "function", "function": {"name": "f"}});
        let out = sanitize_tool_schemas(&[t]);
        assert_eq!(
            out[0]["function"]["parameters"],
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn non_dict_parameters_become_default_object() {
        assert_eq!(
            sanitized_params(json!("nope")),
            json!({"type": "object", "properties": {}})
        );
        assert_eq!(
            sanitized_params(json!(null)),
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn tool_without_function_passthrough() {
        let t = json!({"type": "function"});
        let out = sanitize_tool_schemas(&[t.clone()]);
        assert_eq!(out[0], t);
    }

    #[test]
    fn non_dict_tool_passthrough() {
        let out = sanitize_tool_schemas(&[json!("not a tool"), json!(42)]);
        assert_eq!(out, vec![json!("not a tool"), json!(42)]);
    }

    #[test]
    fn top_level_forced_to_object_with_properties() {
        let out = sanitized_params(json!({}));
        assert_eq!(out["type"], json!("object"));
        assert_eq!(out["properties"], json!({}));
    }

    #[test]
    fn top_level_wrong_type_forced_object() {
        let out = sanitized_params(json!({"type": "string"}));
        assert_eq!(out["type"], json!("object"));
        assert_eq!(out["properties"], json!({}));
    }

    #[test]
    fn bare_string_object_property_replaced() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {"x": "object"}
        }));
        assert_eq!(
            out["properties"]["x"],
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn bare_string_scalar_property_replaced() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {"x": "string"}
        }));
        assert_eq!(out["properties"]["x"], json!({"type": "string"}));
    }

    #[test]
    fn non_schema_string_becomes_empty_object() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {"x": "totally-not-a-type"}
        }));
        assert_eq!(
            out["properties"]["x"],
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn additional_properties_bare_string() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {},
            "additionalProperties": "object"
        }));
        assert_eq!(
            out["additionalProperties"],
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn additional_properties_bool_preserved() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }));
        assert_eq!(out["additionalProperties"], json!(false));
    }

    #[test]
    fn type_array_with_null_collapses_to_single() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {"x": {"type": ["string", "null"]}}
        }));
        assert_eq!(out["properties"]["x"]["type"], json!("string"));
        assert_eq!(out["properties"]["x"]["nullable"], json!(true));
    }

    #[test]
    fn type_array_multiple_non_null_picks_first() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {"x": {"type": ["string", "integer", "null"]}}
        }));
        assert_eq!(out["properties"]["x"]["type"], json!("string"));
        // More than one non-null branch → no nullable hint (falls into fallback).
        assert!(out["properties"]["x"].get("nullable").is_none());
    }

    #[test]
    fn type_array_all_null_becomes_object() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {"x": {"type": ["null"]}}
        }));
        assert_eq!(out["properties"]["x"]["type"], json!("object"));
        // object node without properties → injected.
        assert_eq!(out["properties"]["x"]["properties"], json!({}));
    }

    #[test]
    fn object_node_missing_properties_injected() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {"nested": {"type": "object"}}
        }));
        assert_eq!(out["properties"]["nested"]["properties"], json!({}));
    }

    #[test]
    fn nullable_anyof_union_collapsed() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {
                "maybe": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": null,
                    "description": "an optional"
                }
            }
        }));
        let maybe = &out["properties"]["maybe"];
        assert!(maybe.get("anyOf").is_none());
        assert_eq!(maybe["type"], json!("string"));
        assert_eq!(maybe["nullable"], json!(true));
        // Metadata carried over.
        assert_eq!(maybe["description"], json!("an optional"));
        assert_eq!(maybe["default"], json!(null));
    }

    #[test]
    fn nullable_oneof_union_collapsed() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {
                "maybe": {"oneOf": [{"type": "integer"}, {"type": "null"}]}
            }
        }));
        let maybe = &out["properties"]["maybe"];
        assert!(maybe.get("oneOf").is_none());
        assert_eq!(maybe["type"], json!("integer"));
        assert_eq!(maybe["nullable"], json!(true));
    }

    #[test]
    fn multibranch_union_left_intact() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {
                "u": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
            }
        }));
        // Two non-null branches → not collapsed.
        let u = &out["properties"]["u"];
        assert_eq!(u["anyOf"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn union_without_null_left_intact() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {
                "u": {"anyOf": [{"type": "string"}]}
            }
        }));
        // No null branch dropped → len(non_null) == len(variants) → not collapsed.
        let u = &out["properties"]["u"];
        assert_eq!(u["anyOf"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn required_pruned_to_valid_props() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a", "ghost"]
        }));
        assert_eq!(out["required"], json!(["a"]));
    }

    #[test]
    fn required_all_invalid_removed() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["ghost"]
        }));
        assert!(out.get("required").is_none());
    }

    #[test]
    fn enum_values_not_treated_as_schemas() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["object", "string", "path"]}
            }
        }));
        // The literal enum strings must be preserved, NOT rewritten to schemas.
        assert_eq!(
            out["properties"]["mode"]["enum"],
            json!(["object", "string", "path"])
        );
    }

    #[test]
    fn examples_preserved_verbatim() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {
                "x": {"type": "string", "examples": ["object", "boolean"]}
            }
        }));
        assert_eq!(
            out["properties"]["x"]["examples"],
            json!(["object", "boolean"])
        );
    }

    #[test]
    fn defs_recursed() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {},
            "$defs": {"Foo": {"type": "object"}}
        }));
        // Foo is object → properties injected.
        assert_eq!(out["$defs"]["Foo"]["properties"], json!({}));
    }

    #[test]
    fn items_recursed() {
        let out = sanitized_params(json!({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": ["string", "null"]}}
            }
        }));
        assert_eq!(out["properties"]["tags"]["items"]["type"], json!("string"));
    }

    #[test]
    fn original_not_mutated() {
        let original = tool(json!({"type": "object", "properties": {"x": "string"}}));
        let snapshot = original.clone();
        let _ = sanitize_tool_schemas(&[original.clone()]);
        assert_eq!(original, snapshot);
    }

    // ---- strip_pattern_and_format ----

    #[test]
    fn strip_empty_tools() {
        let (tools, n) = strip_pattern_and_format(vec![]);
        assert!(tools.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn strip_pattern_and_format_from_schema_node() {
        let t = tool(json!({
            "type": "object",
            "properties": {
                "x": {"type": "string", "pattern": "\\d+", "format": "email"}
            }
        }));
        let (out, n) = strip_pattern_and_format(vec![t]);
        assert_eq!(n, 2);
        let x = &out[0]["function"]["parameters"]["properties"]["x"];
        assert!(x.get("pattern").is_none());
        assert!(x.get("format").is_none());
        assert_eq!(x["type"], json!("string"));
    }

    #[test]
    fn strip_does_not_touch_property_named_pattern() {
        // A property literally named "pattern" lives in `properties`, not as a
        // sibling of `type`, so it must survive.
        let t = tool(json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "the regex"}
            },
            "required": ["pattern"]
        }));
        let (out, n) = strip_pattern_and_format(vec![t]);
        assert_eq!(n, 0);
        assert!(out[0]["function"]["parameters"]["properties"]
            .get("pattern")
            .is_some());
    }

    #[test]
    fn strip_inside_anyof_branch() {
        let t = tool(json!({
            "type": "object",
            "properties": {
                "x": {"anyOf": [{"type": "string", "format": "uri"}]}
            }
        }));
        let (out, n) = strip_pattern_and_format(vec![t]);
        assert_eq!(n, 1);
        let branch = &out[0]["function"]["parameters"]["properties"]["x"]["anyOf"][0];
        assert!(branch.get("format").is_none());
    }

    #[test]
    fn strip_missing_function_or_params_noop() {
        let (out, n) = strip_pattern_and_format(vec![
            json!({"type": "function"}),
            json!("nope"),
            json!({"type": "function", "function": {"parameters": "bad"}}),
        ]);
        assert_eq!(n, 0);
        assert_eq!(out.len(), 3);
    }

    // ---- strip_nullable_unions direct ----

    #[test]
    fn strip_nullable_unions_without_hint() {
        let out = strip_nullable_unions(
            json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
            false,
        );
        assert_eq!(out["type"], json!("string"));
        assert!(out.get("nullable").is_none());
    }

    #[test]
    fn strip_nullable_unions_scalar_passthrough() {
        assert_eq!(strip_nullable_unions(json!(42), true), json!(42));
        assert_eq!(strip_nullable_unions(json!("x"), true), json!("x"));
    }
}
