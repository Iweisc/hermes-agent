//! Helpers for translating OpenAI-style tool schemas to Moonshot's schema subset.
//!
//! Moonshot (Kimi) accepts a stricter subset of JSON Schema than standard OpenAI
//! tool calling. Requests that violate it fail with HTTP 400:
//!
//! ```text
//! tools.function.parameters is not a valid moonshot flavored json schema,
//! details: <...>
//! ```
//!
//! Known rejection modes documented at
//! <https://forum.moonshot.ai/t/tool-calling-specification-violation-on-moonshot-api/102>
//! and MoonshotAI/kimi-cli#1595:
//!
//! 1. Every property schema must carry a `type`. Standard JSON Schema allows
//!    type to be omitted (the value is then unconstrained); Moonshot refuses.
//! 2. When `anyOf` is used, `type` must be on the `anyOf` children, not
//!    the parent. Presence of both causes "type should be defined in anyOf
//!    items instead of the parent schema".
//!
//! The `#/definitions/...` → `#/$defs/...` rewrite for draft-07 refs is
//! handled separately at MCP registration time for all providers.
//!
//! This is a faithful native port of `agent/moonshot_schema.py`.

use serde_json::{Map, Value};

/// Keys whose values are maps of name → schema (not schemas themselves).
/// When we recurse, we walk the values of these maps as schemas, but we do
/// NOT apply the missing-type repair to the map itself.
const SCHEMA_MAP_KEYS: &[&str] = &["properties", "patternProperties", "$defs", "definitions"];

/// Keys whose values are lists of schemas.
const SCHEMA_LIST_KEYS: &[&str] = &["anyOf", "oneOf", "allOf", "prefixItems"];

/// Keys whose values are a single nested schema.
const SCHEMA_NODE_KEYS: &[&str] = &["items", "contains", "not", "additionalProperties", "propertyNames"];

fn is_schema_map_key(k: &str) -> bool {
    SCHEMA_MAP_KEYS.contains(&k)
}
fn is_schema_list_key(k: &str) -> bool {
    SCHEMA_LIST_KEYS.contains(&k)
}
fn is_schema_node_key(k: &str) -> bool {
    SCHEMA_NODE_KEYS.contains(&k)
}

/// Recursively apply Moonshot repairs to a schema node.
///
/// `is_schema=true` means this value is a JSON Schema node and gets the
/// missing-type + anyOf-parent repairs applied. `is_schema=false` means
/// it's a container map (e.g. the value of `properties`) and we only
/// recurse into its values.
fn repair_schema(node: Value, is_schema: bool) -> Value {
    match node {
        // Lists only show up under schema-list keys (anyOf/oneOf/allOf), so
        // every element is itself a schema.
        Value::Array(items) => {
            Value::Array(items.into_iter().map(|item| repair_schema(item, true)).collect())
        }
        Value::Object(obj) => repair_object(obj, is_schema),
        other => other,
    }
}

fn repair_object(node: Map<String, Value>, is_schema: bool) -> Value {
    // Walk the dict, deciding per-key whether recursion is into a schema
    // node, a container map, or a scalar.
    let mut repaired: Map<String, Value> = Map::new();
    for (key, value) in node.into_iter() {
        if is_schema_map_key(&key) && value.is_object() {
            // Map of name → schema. Don't treat the map itself as a schema
            // (it has no type / properties of its own), but each value is.
            let sub = match value {
                Value::Object(m) => m,
                _ => unreachable!(),
            };
            let mut new_map = Map::new();
            for (sub_key, sub_val) in sub.into_iter() {
                new_map.insert(sub_key, repair_schema(sub_val, true));
            }
            repaired.insert(key, Value::Object(new_map));
        } else if is_schema_list_key(&key) && value.is_array() {
            let arr = match value {
                Value::Array(a) => a,
                _ => unreachable!(),
            };
            let new_arr: Vec<Value> = arr.into_iter().map(|v| repair_schema(v, true)).collect();
            repaired.insert(key, Value::Array(new_arr));
        } else if is_schema_node_key(&key) {
            // items / not / additionalProperties: single nested schema.
            // additionalProperties can also be a bool — leave those alone.
            if value.is_object() {
                repaired.insert(key, repair_schema(value, true));
            } else {
                repaired.insert(key, value);
            }
        } else {
            // Scalars (description, title, format, enum values, etc.) pass through.
            repaired.insert(key, value);
        }
    }

    if !is_schema {
        return Value::Object(repaired);
    }

    // Rule 2: when anyOf is present, type belongs only on the children.
    // Additionally, Moonshot rejects null-type branches inside anyOf
    // (enum value (<nil>) does not match any type in [string]).
    // Collapse the anyOf to the first non-null branch and infer its type.
    if matches!(repaired.get("anyOf"), Some(Value::Array(_))) {
        repaired.remove("type");
        // Borrow the anyOf array.
        let any_of_len = match repaired.get("anyOf") {
            Some(Value::Array(a)) => a.len(),
            _ => 0,
        };
        // Collect indices of non-null branches.
        let non_null_indices: Vec<usize> = match repaired.get("anyOf") {
            Some(Value::Array(a)) => a
                .iter()
                .enumerate()
                .filter_map(|(i, b)| {
                    if let Value::Object(m) = b {
                        let is_null = matches!(m.get("type"), Some(Value::String(s)) if s == "null");
                        if !is_null {
                            return Some(i);
                        }
                    }
                    None
                })
                .collect(),
            _ => Vec::new(),
        };

        if !non_null_indices.is_empty() && non_null_indices.len() < any_of_len {
            // Drop the anyOf wrapper — keep only the non-null branch(es).
            if non_null_indices.len() == 1 {
                // Promote the single non-null branch and fall through to
                // Rules 1/3 so nullable/enum cleanup still applies to the
                // merged node. merge = repaired without "anyOf"; then
                // merge.update(non_null[0]).
                let idx = non_null_indices[0];
                // Extract the branch object.
                let branch_obj = {
                    let arr = match repaired.get_mut("anyOf") {
                        Some(Value::Array(a)) => a,
                        _ => unreachable!(),
                    };
                    match std::mem::replace(&mut arr[idx], Value::Null) {
                        Value::Object(m) => m,
                        _ => unreachable!(),
                    }
                };
                repaired.remove("anyOf");
                // merge.update(branch): branch keys overwrite repaired keys.
                for (k, v) in branch_obj.into_iter() {
                    repaired.insert(k, v);
                }
                // fall through to Rules 1/3 below
            } else {
                // Keep only the non-null branches.
                let arr = match repaired.remove("anyOf") {
                    Some(Value::Array(a)) => a,
                    _ => unreachable!(),
                };
                let mut kept: Vec<Value> = Vec::with_capacity(non_null_indices.len());
                for (i, v) in arr.into_iter().enumerate() {
                    if non_null_indices.contains(&i) {
                        kept.push(v);
                    }
                }
                repaired.insert("anyOf".to_string(), Value::Array(kept));
                return Value::Object(repaired);
            }
        } else {
            // Nothing to collapse — parent type stripped, children already
            // repaired by the recursive walk above.
            return Value::Object(repaired);
        }
    }

    // Moonshot also rejects non-standard keywords like `nullable` on
    // parameter schemas — strip it.
    repaired.remove("nullable");

    // Rule 1: property schemas without type need one. $ref nodes are exempt
    // — their type comes from the referenced definition.
    // Fill missing type BEFORE Rule 3 so enum cleanup can check the type.
    if !repaired.contains_key("$ref") {
        fill_missing_type(&mut repaired);
    }

    // Rule 3: Moonshot rejects null/empty-string values inside enum arrays
    // when the parent type is a scalar (string, integer, etc.). The error:
    //   "enum value (<nil>) does not match any type in [string]"
    // Strip null and empty-string from enum values, and if the enum becomes
    // empty, drop it entirely.
    if matches!(repaired.get("enum"), Some(Value::Array(_))) {
        let node_type = match repaired.get("type") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        };
        let scalar = matches!(
            node_type.as_deref(),
            Some("string") | Some("integer") | Some("number") | Some("boolean")
        );
        if scalar {
            let arr = match repaired.remove("enum") {
                Some(Value::Array(a)) => a,
                _ => unreachable!(),
            };
            let cleaned: Vec<Value> = arr
                .into_iter()
                .filter(|v| !v.is_null() && !matches!(v, Value::String(s) if s.is_empty()))
                .collect();
            if !cleaned.is_empty() {
                repaired.insert("enum".to_string(), Value::Array(cleaned));
            }
            // else: drop it entirely (do not re-insert)
        }
    }

    Value::Object(repaired)
}

/// Infer a reasonable `type` if this schema node has none.
fn fill_missing_type(node: &mut Map<String, Value>) {
    // if "type" in node and node["type"] not in (None, "")
    if let Some(t) = node.get("type") {
        let is_empty = match t {
            Value::Null => true,
            Value::String(s) => s.is_empty(),
            _ => false,
        };
        if !is_empty {
            return;
        }
    }

    // Heuristic: presence of `properties` → object, `items` → array, `enum`
    // → type of first enum value, else fall back to `string` (safest scalar).
    let inferred: &str = if node.contains_key("properties")
        || node.contains_key("required")
        || node.contains_key("additionalProperties")
    {
        "object"
    } else if node.contains_key("items") || node.contains_key("prefixItems") {
        "array"
    } else if let Some(Value::Array(arr)) = node.get("enum") {
        if let Some(sample) = arr.first() {
            // Python ladder: bool, then int (note bool is subclass of int in
            // Python, so bool is checked first), then float, else string.
            if sample.is_boolean() {
                "boolean"
            } else if sample.is_i64() || sample.is_u64() {
                "integer"
            } else if sample.is_f64() {
                "number"
            } else {
                "string"
            }
        } else {
            // Empty enum: Python's `node["enum"]` is falsy, so the elif is
            // not taken and it falls through to the final else → "string".
            "string"
        }
    } else {
        "string"
    };

    node.insert("type".to_string(), Value::String(inferred.to_string()));
}

/// Normalize tool parameters to a Moonshot-compatible object schema.
///
/// Returns a deep-copied schema with the flavored-JSON-Schema repairs
/// applied. Input is not mutated.
pub fn sanitize_moonshot_tool_parameters(parameters: &Value) -> Value {
    let obj = match parameters {
        Value::Object(_) => parameters.clone(),
        _ => {
            let mut m = Map::new();
            m.insert("type".to_string(), Value::String("object".to_string()));
            m.insert("properties".to_string(), Value::Object(Map::new()));
            return Value::Object(m);
        }
    };

    let repaired = repair_schema(obj, true);
    let mut map = match repaired {
        Value::Object(m) => m,
        _ => {
            let mut m = Map::new();
            m.insert("type".to_string(), Value::String("object".to_string()));
            m.insert("properties".to_string(), Value::Object(Map::new()));
            return Value::Object(m);
        }
    };

    // Top-level must be an object schema.
    let is_object = matches!(map.get("type"), Some(Value::String(s)) if s == "object");
    if !is_object {
        map.insert("type".to_string(), Value::String("object".to_string()));
    }
    if !map.contains_key("properties") {
        map.insert("properties".to_string(), Value::Object(Map::new()));
    }

    Value::Object(map)
}

/// Apply [`sanitize_moonshot_tool_parameters`] to every tool's parameters.
///
/// Returns a new list only if at least one tool's parameters changed;
/// otherwise returns a clone of the original list (mirroring the Python's
/// "return original object when unchanged" semantics).
pub fn sanitize_moonshot_tools(tools: &[Value]) -> Vec<Value> {
    if tools.is_empty() {
        return tools.to_vec();
    }

    let mut sanitized: Vec<Value> = Vec::with_capacity(tools.len());
    let mut any_change = false;

    for tool in tools {
        let tool_obj = match tool {
            Value::Object(m) => m,
            _ => {
                sanitized.push(tool.clone());
                continue;
            }
        };
        let fn_obj = match tool_obj.get("function") {
            Some(Value::Object(m)) => m,
            _ => {
                sanitized.push(tool.clone());
                continue;
            }
        };

        // params = fn.get("parameters")  (may be missing → treated as non-dict
        // by sanitize, yielding the default object schema)
        let params: Value = fn_obj.get("parameters").cloned().unwrap_or(Value::Null);
        let repaired = sanitize_moonshot_tool_parameters(&params);

        // Python compares identity (`repaired is not params`). Since our
        // sanitize always returns a fresh owned value, identity is always
        // "different"; the observable contract is "did the value change".
        // We approximate by value-equality against the original parameters.
        if repaired != params {
            any_change = true;
            let mut new_fn = fn_obj.clone();
            new_fn.insert("parameters".to_string(), repaired);
            let mut new_tool = tool_obj.clone();
            new_tool.insert("function".to_string(), Value::Object(new_fn));
            sanitized.push(Value::Object(new_tool));
        } else {
            sanitized.push(tool.clone());
        }
    }

    if any_change {
        sanitized
    } else {
        tools.to_vec()
    }
}

/// True for any Kimi / Moonshot model slug, regardless of aggregator prefix.
///
/// Matches bare names (`kimi-k2.6`, `moonshotai/Kimi-K2.6`) and aggregator-
/// prefixed slugs (`nous/moonshotai/kimi-k2.6`, `openrouter/moonshotai/...`).
/// Detection by model name covers Nous / OpenRouter / other aggregators that
/// route to Moonshot's inference, where the base URL is the aggregator's, not
/// `api.moonshot.ai`.
pub fn is_moonshot_model(model: Option<&str>) -> bool {
    let model = match model {
        Some(m) => m,
        None => return false,
    };
    if model.is_empty() {
        return false;
    }
    let bare = model.trim().to_lowercase();
    if bare.is_empty() {
        return false;
    }
    // Last path segment (covers aggregator-prefixed slugs)
    let tail = bare.rsplit('/').next().unwrap_or(&bare);
    if tail.starts_with("kimi-") || tail == "kimi" {
        return true;
    }
    // Vendor-prefixed forms commonly used on aggregators
    if bare.contains("moonshot") || bare.contains("/kimi") || bare.starts_with("kimi") {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn non_dict_parameters_become_default_object() {
        assert_eq!(
            sanitize_moonshot_tool_parameters(&json!("nope")),
            json!({"type": "object", "properties": {}})
        );
        assert_eq!(
            sanitize_moonshot_tool_parameters(&json!(null)),
            json!({"type": "object", "properties": {}})
        );
        assert_eq!(
            sanitize_moonshot_tool_parameters(&json!([1, 2, 3])),
            json!({"type": "object", "properties": {}})
        );
    }

    #[test]
    fn top_level_forced_to_object_with_properties() {
        let out = sanitize_moonshot_tool_parameters(&json!({}));
        assert_eq!(out["type"], json!("object"));
        assert_eq!(out["properties"], json!({}));
    }

    #[test]
    fn property_without_type_gets_string() {
        let input = json!({
            "type": "object",
            "properties": {
                "name": {"description": "the name"}
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        assert_eq!(out["properties"]["name"]["type"], json!("string"));
        assert_eq!(out["properties"]["name"]["description"], json!("the name"));
    }

    #[test]
    fn property_with_properties_infers_object() {
        let input = json!({
            "type": "object",
            "properties": {
                "nested": {"properties": {"x": {"type": "integer"}}}
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        assert_eq!(out["properties"]["nested"]["type"], json!("object"));
        assert_eq!(out["properties"]["nested"]["properties"]["x"]["type"], json!("integer"));
    }

    #[test]
    fn property_with_items_infers_array() {
        let input = json!({
            "type": "object",
            "properties": {
                "tags": {"items": {"type": "string"}}
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        assert_eq!(out["properties"]["tags"]["type"], json!("array"));
        assert_eq!(out["properties"]["tags"]["items"]["type"], json!("string"));
    }

    #[test]
    fn enum_infers_type_from_first_value() {
        // integer
        let out = sanitize_moonshot_tool_parameters(&json!({
            "type": "object",
            "properties": {"n": {"enum": [1, 2, 3]}}
        }));
        assert_eq!(out["properties"]["n"]["type"], json!("integer"));

        // boolean
        let out = sanitize_moonshot_tool_parameters(&json!({
            "type": "object",
            "properties": {"b": {"enum": [true, false]}}
        }));
        assert_eq!(out["properties"]["b"]["type"], json!("boolean"));

        // number
        let out = sanitize_moonshot_tool_parameters(&json!({
            "type": "object",
            "properties": {"f": {"enum": [1.5, 2.5]}}
        }));
        assert_eq!(out["properties"]["f"]["type"], json!("number"));

        // string
        let out = sanitize_moonshot_tool_parameters(&json!({
            "type": "object",
            "properties": {"s": {"enum": ["a", "b"]}}
        }));
        assert_eq!(out["properties"]["s"]["type"], json!("string"));
    }

    #[test]
    fn anyof_strips_parent_type_and_drops_null_branch_single() {
        // Single non-null branch → promoted, anyOf removed.
        let input = json!({
            "type": "object",
            "properties": {
                "maybe": {
                    "type": "string",
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                }
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        let maybe = &out["properties"]["maybe"];
        assert!(maybe.get("anyOf").is_none());
        assert_eq!(maybe["type"], json!("string"));
    }

    #[test]
    fn anyof_multiple_non_null_branches_kept_no_parent_type() {
        let input = json!({
            "type": "object",
            "properties": {
                "u": {
                    "type": "string",
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                        {"type": "null"}
                    ]
                }
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        let u = &out["properties"]["u"];
        assert!(u.get("type").is_none(), "parent type should be stripped");
        let branches = u["anyOf"].as_array().unwrap();
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0]["type"], json!("string"));
        assert_eq!(branches[1]["type"], json!("integer"));
    }

    #[test]
    fn anyof_no_null_branch_keeps_all_strips_parent_type() {
        let input = json!({
            "type": "object",
            "properties": {
                "u": {
                    "type": "string",
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"}
                    ]
                }
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        let u = &out["properties"]["u"];
        assert!(u.get("type").is_none());
        assert_eq!(u["anyOf"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn nullable_keyword_stripped() {
        let input = json!({
            "type": "object",
            "properties": {
                "x": {"type": "string", "nullable": true}
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        assert!(out["properties"]["x"].get("nullable").is_none());
        assert_eq!(out["properties"]["x"]["type"], json!("string"));
    }

    #[test]
    fn ref_node_exempt_from_type_fill() {
        let input = json!({
            "type": "object",
            "properties": {
                "ref": {"$ref": "#/$defs/Foo"}
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        assert!(out["properties"]["ref"].get("type").is_none());
        assert_eq!(out["properties"]["ref"]["$ref"], json!("#/$defs/Foo"));
    }

    #[test]
    fn enum_strips_null_and_empty_string_for_scalar() {
        let input = json!({
            "type": "object",
            "properties": {
                "s": {"type": "string", "enum": ["a", "", null, "b"]}
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        assert_eq!(out["properties"]["s"]["enum"], json!(["a", "b"]));
    }

    #[test]
    fn enum_dropped_when_empty_after_cleanup() {
        let input = json!({
            "type": "object",
            "properties": {
                "s": {"type": "string", "enum": ["", null]}
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        assert!(out["properties"]["s"].get("enum").is_none());
        assert_eq!(out["properties"]["s"]["type"], json!("string"));
    }

    #[test]
    fn defs_map_values_repaired_but_not_map_itself() {
        let input = json!({
            "type": "object",
            "properties": {},
            "$defs": {
                "Foo": {"properties": {"x": {"description": "hi"}}}
            }
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        // The $defs map itself must not get a type.
        assert!(out["$defs"].get("type").is_none());
        // The Foo schema is inferred object, its prop x → string.
        assert_eq!(out["$defs"]["Foo"]["type"], json!("object"));
        assert_eq!(out["$defs"]["Foo"]["properties"]["x"]["type"], json!("string"));
    }

    #[test]
    fn additional_properties_bool_left_alone() {
        let input = json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        assert_eq!(out["additionalProperties"], json!(false));
    }

    #[test]
    fn additional_properties_schema_repaired() {
        let input = json!({
            "type": "object",
            "properties": {},
            "additionalProperties": {"description": "free-form"}
        });
        let out = sanitize_moonshot_tool_parameters(&input);
        assert_eq!(out["additionalProperties"]["type"], json!("string"));
    }

    #[test]
    fn sanitize_tools_unchanged_returns_equivalent_list() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let out = sanitize_moonshot_tools(&tools);
        assert_eq!(out, tools);
    }

    #[test]
    fn sanitize_tools_applies_repairs() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"description": "no type"}}
                }
            }
        })];
        let out = sanitize_moonshot_tools(&tools);
        assert_eq!(
            out[0]["function"]["parameters"]["properties"]["a"]["type"],
            json!("string")
        );
    }

    #[test]
    fn sanitize_tools_non_dict_passthrough() {
        let tools = vec![json!("not a tool"), json!(42)];
        let out = sanitize_moonshot_tools(&tools);
        assert_eq!(out, tools);
    }

    #[test]
    fn sanitize_tools_missing_function_passthrough() {
        let tools = vec![json!({"type": "function"})];
        let out = sanitize_moonshot_tools(&tools);
        assert_eq!(out, tools);
    }

    #[test]
    fn sanitize_tools_empty() {
        let tools: Vec<Value> = vec![];
        assert_eq!(sanitize_moonshot_tools(&tools), tools);
    }

    #[test]
    fn is_moonshot_model_cases() {
        assert!(is_moonshot_model(Some("kimi-k2.6")));
        assert!(is_moonshot_model(Some("kimi")));
        assert!(is_moonshot_model(Some("moonshotai/Kimi-K2.6")));
        assert!(is_moonshot_model(Some("nous/moonshotai/kimi-k2.6")));
        assert!(is_moonshot_model(Some("openrouter/moonshotai/kimi-foo")));
        assert!(is_moonshot_model(Some("  KIMI-K2  ")));
        assert!(is_moonshot_model(Some("some/kimi")));

        assert!(!is_moonshot_model(None));
        assert!(!is_moonshot_model(Some("")));
        assert!(!is_moonshot_model(Some("gpt-4o")));
        assert!(!is_moonshot_model(Some("claude-3-5-sonnet")));
        // "akimi" should NOT match: tail doesn't start with kimi-, no
        // "moonshot", no "/kimi", and bare doesn't start with "kimi".
        assert!(!is_moonshot_model(Some("akimi")));
    }

    #[test]
    fn input_not_mutated() {
        let input = json!({
            "type": "object",
            "properties": {"a": {"description": "x"}}
        });
        let snapshot = input.clone();
        let _ = sanitize_moonshot_tool_parameters(&input);
        assert_eq!(input, snapshot);
    }
}
