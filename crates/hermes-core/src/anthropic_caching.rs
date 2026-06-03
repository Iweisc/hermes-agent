//! Anthropic prompt caching (`system_and_3` strategy).
//!
//! Reduces input token costs by ~75% on multi-turn conversations by caching
//! the conversation prefix. Uses 4 `cache_control` breakpoints (Anthropic max):
//!   1. System prompt (stable across all turns)
//!   2-4. Last 3 non-system messages (rolling window)
//!
//! Pure functions — no struct state, no agent dependency. This is a faithful
//! port of `agent/prompt_caching.py`.

use serde_json::{json, Map, Value};

/// Build the ephemeral cache marker for a given TTL.
///
/// Mirrors the Python `marker = {"type": "ephemeral"}` with an optional
/// `ttl` key set to `"1h"` when `cache_ttl == "1h"`.
fn build_cache_marker(cache_ttl: &str) -> Value {
    let mut marker = Map::new();
    marker.insert("type".to_string(), json!("ephemeral"));
    if cache_ttl == "1h" {
        marker.insert("ttl".to_string(), json!("1h"));
    }
    Value::Object(marker)
}

/// Add `cache_control` to a single message, handling all format variations.
///
/// Faithful port of Python `_apply_cache_marker`.
fn apply_cache_marker(msg: &mut Value, cache_marker: &Value, native_anthropic: bool) {
    // role = msg.get("role", "")
    let role = msg
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    if role == "tool" {
        if native_anthropic {
            if let Some(obj) = msg.as_object_mut() {
                obj.insert("cache_control".to_string(), cache_marker.clone());
            }
        }
        return;
    }

    // content = msg.get("content")
    let content = msg.get("content");

    // content is None or content == ""
    let is_none_or_empty = match content {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        _ => false,
    };
    if is_none_or_empty {
        if let Some(obj) = msg.as_object_mut() {
            obj.insert("cache_control".to_string(), cache_marker.clone());
        }
        return;
    }

    // isinstance(content, str)
    if let Some(Value::String(s)) = content {
        let text = s.clone();
        if let Some(obj) = msg.as_object_mut() {
            obj.insert(
                "content".to_string(),
                json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": cache_marker.clone(),
                }]),
            );
        }
        return;
    }

    // isinstance(content, list) and content (non-empty)
    if let Some(Value::Array(_)) = content {
        if let Some(arr) = msg
            .get_mut("content")
            .and_then(Value::as_array_mut)
        {
            if let Some(last) = arr.last_mut() {
                // if isinstance(last, dict)
                if let Some(last_obj) = last.as_object_mut() {
                    last_obj.insert("cache_control".to_string(), cache_marker.clone());
                }
            }
        }
    }
}

/// Apply the `system_and_3` caching strategy to messages for Anthropic models.
///
/// Places up to 4 `cache_control` breakpoints: system prompt + last 3
/// non-system messages.
///
/// Returns a deep copy of `api_messages` with `cache_control` breakpoints
/// injected. The input slice is left untouched.
///
/// Faithful port of Python `apply_anthropic_cache_control`.
pub fn apply_anthropic_cache_control(
    api_messages: &[Value],
    cache_ttl: &str,
    native_anthropic: bool,
) -> Vec<Value> {
    // messages = copy.deepcopy(api_messages)
    let mut messages: Vec<Value> = api_messages.to_vec();
    if messages.is_empty() {
        return messages;
    }

    let marker = build_cache_marker(cache_ttl);

    let mut breakpoints_used = 0usize;

    // if messages[0].get("role") == "system"
    let first_is_system = messages[0]
        .get("role")
        .and_then(Value::as_str)
        == Some("system");
    if first_is_system {
        apply_cache_marker(&mut messages[0], &marker, native_anthropic);
        breakpoints_used += 1;
    }

    let remaining = 4usize.saturating_sub(breakpoints_used);

    // non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    let non_sys: Vec<usize> = (0..messages.len())
        .filter(|&i| messages[i].get("role").and_then(Value::as_str) != Some("system"))
        .collect();

    // for idx in non_sys[-remaining:]
    let start = non_sys.len().saturating_sub(remaining);
    let targets: Vec<usize> = non_sys[start..].to_vec();
    for idx in targets {
        apply_cache_marker(&mut messages[idx], &marker, native_anthropic);
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_messages_returns_empty() {
        let msgs: Vec<Value> = vec![];
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        assert!(out.is_empty());
    }

    #[test]
    fn string_content_becomes_text_block_with_marker() {
        let msgs = vec![json!({"role": "user", "content": "hello"})];
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        assert_eq!(
            out[0],
            json!({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hello",
                    "cache_control": {"type": "ephemeral"},
                }],
            })
        );
    }

    #[test]
    fn ttl_1h_adds_ttl_field() {
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let out = apply_anthropic_cache_control(&msgs, "1h", false);
        let cc = &out[0]["content"][0]["cache_control"];
        assert_eq!(cc, &json!({"type": "ephemeral", "ttl": "1h"}));
    }

    #[test]
    fn default_ttl_has_no_ttl_field() {
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        let cc = &out[0]["content"][0]["cache_control"];
        assert_eq!(cc, &json!({"type": "ephemeral"}));
        assert!(cc.get("ttl").is_none());
    }

    #[test]
    fn empty_string_content_marks_message_directly() {
        let msgs = vec![json!({"role": "user", "content": ""})];
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        assert_eq!(out[0]["cache_control"], json!({"type": "ephemeral"}));
        // content stays the empty string
        assert_eq!(out[0]["content"], json!(""));
    }

    #[test]
    fn null_content_marks_message_directly() {
        let msgs = vec![json!({"role": "user", "content": null})];
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        assert_eq!(out[0]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn missing_content_marks_message_directly() {
        let msgs = vec![json!({"role": "user"})];
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        assert_eq!(out[0]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn list_content_marks_last_dict_element() {
        let msgs = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        })];
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        let arr = out[0]["content"].as_array().unwrap();
        assert!(arr[0].get("cache_control").is_none());
        assert_eq!(arr[1]["cache_control"], json!({"type": "ephemeral"}));
    }

    #[test]
    fn list_content_last_non_dict_is_left_alone() {
        let msgs = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "a"}, "trailing"],
        })];
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        // last element is a string; nothing added, no panic
        assert_eq!(out[0]["content"][1], json!("trailing"));
        assert!(out[0].get("cache_control").is_none());
    }

    #[test]
    fn tool_role_native_anthropic_gets_marker() {
        let msgs = vec![json!({"role": "tool", "content": "result"})];
        let out = apply_anthropic_cache_control(&msgs, "5m", true);
        assert_eq!(out[0]["cache_control"], json!({"type": "ephemeral"}));
        // content untouched (still the raw string)
        assert_eq!(out[0]["content"], json!("result"));
    }

    #[test]
    fn tool_role_non_native_gets_nothing() {
        let msgs = vec![json!({"role": "tool", "content": "result"})];
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        assert!(out[0].get("cache_control").is_none());
        assert_eq!(out[0]["content"], json!("result"));
    }

    #[test]
    fn system_plus_three_breakpoints_rolling_window() {
        // system + 5 user messages; expect system + last 3 non-system cached.
        let mut msgs = vec![json!({"role": "system", "content": "sys"})];
        for i in 0..5 {
            msgs.push(json!({"role": "user", "content": format!("m{i}")}));
        }
        let out = apply_anthropic_cache_control(&msgs, "5m", false);

        // system cached
        assert_eq!(
            out[0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );

        // indices 1,2 (m0,m1) NOT cached; 3,4,5 (m2,m3,m4) cached
        let cached: Vec<bool> = out
            .iter()
            .map(|m| {
                // a message is "cached" if its content[last] has cache_control
                // or the message itself has cache_control
                if let Some(arr) = m["content"].as_array() {
                    arr.last()
                        .and_then(|l| l.get("cache_control"))
                        .is_some()
                } else {
                    m.get("cache_control").is_some()
                }
            })
            .collect();
        assert_eq!(cached, vec![true, false, false, true, true, true]);
    }

    #[test]
    fn no_system_uses_four_breakpoints() {
        // 5 user messages, no system → last 4 cached.
        let mut msgs = vec![];
        for i in 0..5 {
            msgs.push(json!({"role": "user", "content": format!("m{i}")}));
        }
        let out = apply_anthropic_cache_control(&msgs, "5m", false);
        let cached: Vec<bool> = out
            .iter()
            .map(|m| {
                m["content"]
                    .as_array()
                    .and_then(|a| a.last())
                    .and_then(|l| l.get("cache_control"))
                    .is_some()
            })
            .collect();
        assert_eq!(cached, vec![false, true, true, true, true]);
    }

    #[test]
    fn input_is_not_mutated() {
        let msgs = vec![json!({"role": "user", "content": "hello"})];
        let _ = apply_anthropic_cache_control(&msgs, "5m", false);
        // original untouched (still a plain string)
        assert_eq!(msgs[0]["content"], json!("hello"));
    }
}
