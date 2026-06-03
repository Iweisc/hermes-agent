//! Home Assistant tool for controlling smart home devices via REST API.
//!
//! Native Rust port of `tools/homeassistant_tool.py`.
//!
//! Registers four LLM-callable tools:
//! - `ha_list_entities` -- list/filter entities by domain or area
//! - `ha_get_state` -- get detailed state of a single entity
//! - `ha_list_services` -- list available services (actions) per domain
//! - `ha_call_service` -- call a HA service (turn_on, turn_off, set_temperature, etc.)
//!
//! Authentication uses a Long-Lived Access Token via the `HASS_TOKEN` env var.
//! The HA instance URL is read from `HASS_URL`
//! (default: `http://homeassistant.local:8123`).

use std::collections::BTreeMap;
use std::env;
use std::sync::Mutex;
use std::time::Duration;

use regex::Regex;
use reqwest::blocking::Client;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DEFAULT_HASS_URL: &str = "http://homeassistant.local:8123";
const LIST_TIMEOUT_SECS: u64 = 15;
const GET_STATE_TIMEOUT_SECS: u64 = 10;

/// Service domains blocked for security -- these allow arbitrary code/command
/// execution on the HA host or enable SSRF attacks on the local network.
/// HA provides zero service-level access control; all safety must be in our layer.
pub const BLOCKED_DOMAINS: &[&str] = &[
    "shell_command", // arbitrary shell commands as root in HA container
    "command_line",  // sensors/switches that execute shell commands
    "python_script", // sandboxed but can escalate via hass.services.call()
    "pyscript",      // scripting integration with broader access
    "hassio",        // addon control, host shutdown/reboot, stdin to containers
    "rest_command",  // HTTP requests from HA server (SSRF vector)
];

// Overrides kept for backward compatibility (e.g. test monkeypatching);
// prefer the env vars. When non-empty these take precedence over env vars.
static HASS_URL_OVERRIDE: Mutex<String> = Mutex::new(String::new());
static HASS_TOKEN_OVERRIDE: Mutex<String> = Mutex::new(String::new());

/// Set the in-process HASS_URL override (mirrors the Python module-level
/// `_HASS_URL`). An empty string clears the override.
pub fn set_hass_url_override(value: &str) {
    *HASS_URL_OVERRIDE.lock().unwrap() = value.to_string();
}

/// Set the in-process HASS_TOKEN override (mirrors the Python module-level
/// `_HASS_TOKEN`). An empty string clears the override.
pub fn set_hass_token_override(value: &str) {
    *HASS_TOKEN_OVERRIDE.lock().unwrap() = value.to_string();
}

/// Return `(hass_url, hass_token)` from overrides or env vars at call time.
/// The URL has any trailing slashes stripped.
pub fn get_config() -> (String, String) {
    let url_override = HASS_URL_OVERRIDE.lock().unwrap().clone();
    let raw_url = if !url_override.is_empty() {
        url_override
    } else {
        env::var("HASS_URL").unwrap_or_else(|_| DEFAULT_HASS_URL.to_string())
    };
    let url = raw_url.trim_end_matches('/').to_string();

    let token_override = HASS_TOKEN_OVERRIDE.lock().unwrap().clone();
    let token = if !token_override.is_empty() {
        token_override
    } else {
        env::var("HASS_TOKEN").unwrap_or_default()
    };

    (url, token)
}

/// Regex for a valid HA entity_id (e.g. "light.living_room", "sensor.temp_1").
pub fn entity_id_re() -> Regex {
    Regex::new(r"^[a-z_][a-z0-9_]*\.[a-z0-9_]+$").unwrap()
}

/// Regex for valid HA service/domain names (e.g. "light", "turn_on").
///
/// Only lowercase ASCII letters, digits, and underscores -- no slashes, dots,
/// or other characters that could allow path traversal in URL construction.
/// The domain and service are interpolated into
/// `/api/services/{domain}/{service}`, so allowing arbitrary strings would
/// enable SSRF via path traversal (e.g. domain="../../api/config") or
/// blocked-domain bypass (e.g. domain="shell_command/../light").
pub fn service_name_re() -> Regex {
    Regex::new(r"^[a-z][a-z0-9_]*$").unwrap()
}

/// Return a JSON error envelope `{"error": message}` as a string. Mirrors the
/// Python `tool_error` helper.
pub fn tool_error(message: impl Into<String>) -> String {
    json!({ "error": message.into() }).to_string()
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

fn build_client(timeout_secs: u64) -> Result<Client, String> {
    Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| e.to_string())
}

fn get_json(url: &str, token: &str, timeout_secs: u64) -> Result<Value, String> {
    let client = build_client(timeout_secs)?;
    let resp = client
        .get(url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .send()
        .map_err(|e| e.to_string())?;
    let resp = resp.error_for_status().map_err(|e| e.to_string())?;
    resp.json::<Value>().map_err(|e| e.to_string())
}

fn post_json(url: &str, token: &str, body: &Value, timeout_secs: u64) -> Result<Value, String> {
    let client = build_client(timeout_secs)?;
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .json(body)
        .send()
        .map_err(|e| e.to_string())?;
    let resp = resp.error_for_status().map_err(|e| e.to_string())?;
    resp.json::<Value>().map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Core helpers (pure -- mirror the Python helper functions)
// ---------------------------------------------------------------------------

/// Filter raw HA states by domain/area and return a compact summary.
pub fn filter_and_summarize(states: &[Value], domain: Option<&str>, area: Option<&str>) -> Value {
    let mut filtered: Vec<&Value> = states.iter().collect();

    if let Some(domain) = domain {
        let prefix = format!("{domain}.");
        filtered.retain(|s| {
            s.get("entity_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .starts_with(&prefix)
        });
    }

    if let Some(area) = area {
        let area_lower = area.to_lowercase();
        filtered.retain(|s| {
            let attrs = s.get("attributes");
            let friendly = attrs
                .and_then(|a| a.get("friendly_name"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_lowercase();
            let area_attr = attrs
                .and_then(|a| a.get("area"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_lowercase();
            friendly.contains(&area_lower) || area_attr.contains(&area_lower)
        });
    }

    let mut entities = Vec::with_capacity(filtered.len());
    for s in &filtered {
        entities.push(json!({
            "entity_id": s.get("entity_id").cloned().unwrap_or(Value::Null),
            "state": s.get("state").cloned().unwrap_or(Value::Null),
            "friendly_name": s
                .get("attributes")
                .and_then(|a| a.get("friendly_name"))
                .and_then(Value::as_str)
                .unwrap_or_default(),
        }));
    }

    json!({
        "count": entities.len(),
        "entities": entities,
    })
}

/// Build the JSON payload for a HA service call.
/// The `entity_id` parameter takes precedence over `data["entity_id"]`.
pub fn build_service_payload(entity_id: Option<&str>, data: Option<&Value>) -> Value {
    let mut payload: Map<String, Value> = Map::new();
    if let Some(Value::Object(map)) = data {
        for (k, v) in map {
            payload.insert(k.clone(), v.clone());
        }
    }
    if let Some(entity_id) = entity_id {
        payload.insert("entity_id".to_string(), Value::String(entity_id.to_string()));
    }
    Value::Object(payload)
}

/// Parse a HA service-call response into a structured result.
pub fn parse_service_response(domain: &str, service: &str, result: &Value) -> Value {
    let mut affected = Vec::new();
    if let Some(items) = result.as_array() {
        for s in items {
            affected.push(json!({
                "entity_id": s.get("entity_id").and_then(Value::as_str).unwrap_or_default(),
                "state": s.get("state").and_then(Value::as_str).unwrap_or_default(),
            }));
        }
    }

    json!({
        "success": true,
        "service": format!("{domain}.{service}"),
        "affected_entities": affected,
    })
}

// ---------------------------------------------------------------------------
// Network operations
// ---------------------------------------------------------------------------

/// Fetch entity states from HA and optionally filter by domain/area.
pub fn list_entities(domain: Option<&str>, area: Option<&str>) -> Result<Value, String> {
    let (hass_url, token) = get_config();
    let url = format!("{hass_url}/api/states");
    let states = get_json(&url, &token, LIST_TIMEOUT_SECS)?;
    let states = states
        .as_array()
        .cloned()
        .ok_or_else(|| "unexpected states payload".to_string())?;
    Ok(filter_and_summarize(&states, domain, area))
}

/// Fetch detailed state of a single entity.
pub fn get_state(entity_id: &str) -> Result<Value, String> {
    let (hass_url, token) = get_config();
    let url = format!("{hass_url}/api/states/{entity_id}");
    let data = get_json(&url, &token, GET_STATE_TIMEOUT_SECS)?;
    Ok(json!({
        "entity_id": data.get("entity_id").cloned().unwrap_or(Value::Null),
        "state": data.get("state").cloned().unwrap_or(Value::Null),
        "attributes": data.get("attributes").cloned().unwrap_or_else(|| json!({})),
        "last_changed": data.get("last_changed").cloned().unwrap_or(Value::Null),
        "last_updated": data.get("last_updated").cloned().unwrap_or(Value::Null),
    }))
}

/// Call a Home Assistant service.
pub fn call_service(
    domain: &str,
    service: &str,
    entity_id: Option<&str>,
    data: Option<&Value>,
) -> Result<Value, String> {
    let (hass_url, token) = get_config();
    let url = format!("{hass_url}/api/services/{domain}/{service}");
    let payload = build_service_payload(entity_id, data);
    let result = post_json(&url, &token, &payload, LIST_TIMEOUT_SECS)?;
    Ok(parse_service_response(domain, service, &result))
}

/// Fetch available services from HA and optionally filter by domain.
pub fn list_services(domain: Option<&str>) -> Result<Value, String> {
    let (hass_url, token) = get_config();
    let url = format!("{hass_url}/api/services");
    let services = get_json(&url, &token, LIST_TIMEOUT_SECS)?;
    let services = services
        .as_array()
        .cloned()
        .ok_or_else(|| "unexpected services payload".to_string())?;

    let mut result = Vec::new();
    for svc_domain in &services {
        let d = svc_domain
            .get("domain")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if let Some(filter) = domain {
            if d != filter {
                continue;
            }
        }

        // Use a BTreeMap so output ordering is deterministic.
        let mut domain_services: BTreeMap<String, Value> = BTreeMap::new();
        if let Some(svc_map) = svc_domain.get("services").and_then(Value::as_object) {
            for (svc_name, svc_info) in svc_map {
                let description = svc_info
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let mut entry = Map::new();
                entry.insert(
                    "description".to_string(),
                    Value::String(description.to_string()),
                );
                if let Some(fields) = svc_info.get("fields").and_then(Value::as_object) {
                    if !fields.is_empty() {
                        let mut field_map: BTreeMap<String, Value> = BTreeMap::new();
                        for (k, v) in fields {
                            if let Some(obj) = v.as_object() {
                                let fdesc = obj
                                    .get("description")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default();
                                field_map
                                    .insert(k.clone(), Value::String(fdesc.to_string()));
                            }
                        }
                        let field_obj: Map<String, Value> = field_map.into_iter().collect();
                        entry.insert("fields".to_string(), Value::Object(field_obj));
                    }
                }
                domain_services.insert(svc_name.clone(), Value::Object(entry));
            }
        }
        let svc_obj: Map<String, Value> = domain_services.into_iter().collect();
        result.push(json!({ "domain": d, "services": Value::Object(svc_obj) }));
    }

    Ok(json!({ "count": result.len(), "domains": result }))
}

// ---------------------------------------------------------------------------
// Handlers (mirror the Python sync wrappers; return JSON strings)
// ---------------------------------------------------------------------------

fn arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

/// Handler for the `ha_list_entities` tool.
pub fn handle_list_entities(args: &Value) -> String {
    let domain = arg_str(args, "domain");
    let area = arg_str(args, "area");
    match list_entities(domain, area) {
        Ok(result) => json!({ "result": result }).to_string(),
        Err(e) => tool_error(format!("Failed to list entities: {e}")),
    }
}

/// Handler for the `ha_get_state` tool.
pub fn handle_get_state(args: &Value) -> String {
    let entity_id = arg_str(args, "entity_id").unwrap_or_default();
    if entity_id.is_empty() {
        return tool_error("Missing required parameter: entity_id");
    }
    if !entity_id_re().is_match(entity_id) {
        return tool_error(format!("Invalid entity_id format: {entity_id}"));
    }
    match get_state(entity_id) {
        Ok(result) => json!({ "result": result }).to_string(),
        Err(e) => tool_error(format!("Failed to get state for {entity_id}: {e}")),
    }
}

/// Handler for the `ha_list_services` tool.
pub fn handle_list_services(args: &Value) -> String {
    let domain = arg_str(args, "domain");
    match list_services(domain) {
        Ok(result) => json!({ "result": result }).to_string(),
        Err(e) => tool_error(format!("Failed to list services: {e}")),
    }
}

/// Handler for the `ha_call_service` tool.
pub fn handle_call_service(args: &Value) -> String {
    let domain = arg_str(args, "domain").unwrap_or_default();
    let service = arg_str(args, "service").unwrap_or_default();
    if domain.is_empty() || service.is_empty() {
        return tool_error("Missing required parameters: domain and service");
    }

    // Validate domain/service format BEFORE the blocklist check -- prevents
    // path traversal in /api/services/{domain}/{service} and blocklist bypass
    // via payloads like "shell_command/../light".
    let name_re = service_name_re();
    if !name_re.is_match(domain) {
        return tool_error(format!("Invalid domain format: {domain:?}"));
    }
    if !name_re.is_match(service) {
        return tool_error(format!("Invalid service format: {service:?}"));
    }

    if BLOCKED_DOMAINS.contains(&domain) {
        let mut sorted: Vec<&str> = BLOCKED_DOMAINS.to_vec();
        sorted.sort_unstable();
        return json!({
            "error": format!(
                "Service domain '{}' is blocked for security. Blocked domains: {}",
                domain,
                sorted.join(", ")
            )
        })
        .to_string();
    }

    let entity_id = arg_str(args, "entity_id");
    if let Some(entity_id) = entity_id {
        if !entity_id.is_empty() && !entity_id_re().is_match(entity_id) {
            return tool_error(format!("Invalid entity_id format: {entity_id}"));
        }
    }
    // Treat empty entity_id like Python's falsy check (None passed downstream).
    let entity_id = entity_id.filter(|e| !e.is_empty());

    // The `data` parameter may arrive as a JSON object or as a JSON string.
    let data: Option<Value> = match args.get("data") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => {
            if s.trim().is_empty() {
                None
            } else {
                match serde_json::from_str::<Value>(s) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        return tool_error(format!(
                            "Invalid JSON string in 'data' parameter: {e}"
                        ));
                    }
                }
            }
        }
        Some(other) => Some(other.clone()),
    };

    match call_service(domain, service, entity_id, data.as_ref()) {
        Ok(result) => json!({ "result": result }).to_string(),
        Err(e) => tool_error(format!("Failed to call {domain}.{service}: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Availability check
// ---------------------------------------------------------------------------

/// Tool is only available when `HASS_TOKEN` is set (and non-empty).
pub fn check_ha_available() -> bool {
    env::var("HASS_TOKEN")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tool schemas
// ---------------------------------------------------------------------------

pub fn ha_list_entities_schema() -> Value {
    json!({
        "name": "ha_list_entities",
        "description": "List Home Assistant entities. Optionally filter by domain (light, switch, climate, sensor, binary_sensor, cover, fan, etc.) or by area name (living room, kitchen, bedroom, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Entity domain to filter by (e.g. 'light', 'switch', 'climate', 'sensor', 'binary_sensor', 'cover', 'fan', 'media_player'). Omit to list all entities."
                },
                "area": {
                    "type": "string",
                    "description": "Area/room name to filter by (e.g. 'living room', 'kitchen'). Matches against entity friendly names. Omit to list all."
                }
            },
            "required": []
        }
    })
}

pub fn ha_get_state_schema() -> Value {
    json!({
        "name": "ha_get_state",
        "description": "Get the detailed state of a single Home Assistant entity, including all attributes (brightness, color, temperature setpoint, sensor readings, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The entity ID to query (e.g. 'light.living_room', 'climate.thermostat', 'sensor.temperature')."
                }
            },
            "required": ["entity_id"]
        }
    })
}

pub fn ha_list_services_schema() -> Value {
    json!({
        "name": "ha_list_services",
        "description": "List available Home Assistant services (actions) for device control. Shows what actions can be performed on each device type and what parameters they accept. Use this to discover how to control devices found via ha_list_entities.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Filter by domain (e.g. 'light', 'climate', 'switch'). Omit to list services for all domains."
                }
            },
            "required": []
        }
    })
}

pub fn ha_call_service_schema() -> Value {
    json!({
        "name": "ha_call_service",
        "description": "Call a Home Assistant service to control a device. Use ha_list_services to discover available services and their parameters for each domain.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Service domain (e.g. 'light', 'switch', 'climate', 'cover', 'media_player', 'fan', 'scene', 'script')."
                },
                "service": {
                    "type": "string",
                    "description": "Service name (e.g. 'turn_on', 'turn_off', 'toggle', 'set_temperature', 'set_hvac_mode', 'open_cover', 'close_cover', 'set_volume_level')."
                },
                "entity_id": {
                    "type": "string",
                    "description": "Target entity ID (e.g. 'light.living_room'). Some services (like scene.turn_on) may not need this."
                },
                "data": {
                    "type": "string",
                    "description": "Additional service data as a JSON string. Examples: {\"brightness\": 255, \"color_name\": \"blue\"} for lights, {\"temperature\": 22, \"hvac_mode\": \"heat\"} for climate, {\"volume_level\": 0.5} for media players."
                }
            },
            "required": ["domain", "service"]
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_id_regex_accepts_valid_and_rejects_invalid() {
        let re = entity_id_re();
        assert!(re.is_match("light.living_room"));
        assert!(re.is_match("sensor.temperature_1"));
        assert!(re.is_match("_hidden.entity"));
        assert!(!re.is_match("Light.LivingRoom"));
        assert!(!re.is_match("light"));
        assert!(!re.is_match("light.living/room"));
        assert!(!re.is_match("../../api/config"));
    }

    #[test]
    fn service_name_regex_blocks_traversal() {
        let re = service_name_re();
        assert!(re.is_match("light"));
        assert!(re.is_match("turn_on"));
        assert!(re.is_match("shell_command"));
        assert!(!re.is_match("_leading_underscore"));
        assert!(!re.is_match("shell_command/../light"));
        assert!(!re.is_match("../../api/config"));
        assert!(!re.is_match("Light"));
    }

    #[test]
    fn filter_by_domain() {
        let states = vec![
            json!({"entity_id": "light.a", "state": "on", "attributes": {"friendly_name": "A"}}),
            json!({"entity_id": "switch.b", "state": "off", "attributes": {"friendly_name": "B"}}),
        ];
        let out = filter_and_summarize(&states, Some("light"), None);
        assert_eq!(out["count"], json!(1));
        assert_eq!(out["entities"][0]["entity_id"], json!("light.a"));
        assert_eq!(out["entities"][0]["friendly_name"], json!("A"));
    }

    #[test]
    fn filter_by_area_matches_friendly_name_or_area_attr() {
        let states = vec![
            json!({"entity_id": "light.a", "state": "on", "attributes": {"friendly_name": "Living Room Lamp"}}),
            json!({"entity_id": "light.b", "state": "on", "attributes": {"friendly_name": "Lamp", "area": "Kitchen"}}),
            json!({"entity_id": "light.c", "state": "on", "attributes": {"friendly_name": "Garage"}}),
        ];
        let out = filter_and_summarize(&states, None, Some("living"));
        assert_eq!(out["count"], json!(1));
        assert_eq!(out["entities"][0]["entity_id"], json!("light.a"));

        let out2 = filter_and_summarize(&states, None, Some("kitchen"));
        assert_eq!(out2["count"], json!(1));
        assert_eq!(out2["entities"][0]["entity_id"], json!("light.b"));
    }

    #[test]
    fn build_payload_entity_id_takes_precedence() {
        let data = json!({"brightness": 255, "entity_id": "light.old"});
        let payload = build_service_payload(Some("light.new"), Some(&data));
        assert_eq!(payload["entity_id"], json!("light.new"));
        assert_eq!(payload["brightness"], json!(255));
    }

    #[test]
    fn build_payload_without_entity_id_keeps_data() {
        let data = json!({"transition": 2});
        let payload = build_service_payload(None, Some(&data));
        assert_eq!(payload["transition"], json!(2));
        assert!(payload.get("entity_id").is_none());
    }

    #[test]
    fn parse_response_collects_affected_entities() {
        let result = json!([
            {"entity_id": "light.a", "state": "on"},
            {"entity_id": "light.b", "state": "off"},
        ]);
        let parsed = parse_service_response("light", "turn_on", &result);
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["service"], json!("light.turn_on"));
        assert_eq!(parsed["affected_entities"].as_array().unwrap().len(), 2);
        assert_eq!(parsed["affected_entities"][0]["entity_id"], json!("light.a"));
    }

    #[test]
    fn parse_response_non_list_yields_empty() {
        let result = json!({"context": {}});
        let parsed = parse_service_response("scene", "turn_on", &result);
        assert_eq!(parsed["affected_entities"], json!([]));
    }

    #[test]
    fn get_state_requires_entity_id() {
        let out = handle_get_state(&json!({}));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"], json!("Missing required parameter: entity_id"));
    }

    #[test]
    fn get_state_rejects_invalid_entity_id() {
        let out = handle_get_state(&json!({"entity_id": "bad/id"}));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"], json!("Invalid entity_id format: bad/id"));
    }

    #[test]
    fn call_service_requires_domain_and_service() {
        let out = handle_call_service(&json!({"domain": "light"}));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            v["error"],
            json!("Missing required parameters: domain and service")
        );
    }

    #[test]
    fn call_service_blocks_dangerous_domains() {
        let out = handle_call_service(&json!({"domain": "shell_command", "service": "x"}));
        let v: Value = serde_json::from_str(&out).unwrap();
        let msg = v["error"].as_str().unwrap();
        assert!(msg.contains("blocked for security"));
        // Sorted blocked list.
        assert!(msg.contains("command_line, hassio, pyscript, python_script, rest_command, shell_command"));
    }

    #[test]
    fn call_service_rejects_traversal_before_blocklist() {
        let out =
            handle_call_service(&json!({"domain": "shell_command/../light", "service": "turn_on"}));
        let v: Value = serde_json::from_str(&out).unwrap();
        // Format rejection, not blocklist.
        assert!(v["error"].as_str().unwrap().starts_with("Invalid domain format"));
    }

    #[test]
    fn call_service_rejects_invalid_entity_id() {
        let out = handle_call_service(&json!({
            "domain": "light",
            "service": "turn_on",
            "entity_id": "BAD"
        }));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"], json!("Invalid entity_id format: BAD"));
    }

    #[test]
    fn call_service_rejects_bad_json_data() {
        let out = handle_call_service(&json!({
            "domain": "light",
            "service": "turn_on",
            "data": "{not json"
        }));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v["error"]
            .as_str()
            .unwrap()
            .starts_with("Invalid JSON string in 'data' parameter"));
    }

    #[test]
    fn config_strips_trailing_slash_and_defaults() {
        // Use overrides to avoid mutating process env in parallel tests.
        set_hass_url_override("http://example.local:8123///");
        set_hass_token_override("tok");
        let (url, token) = get_config();
        assert_eq!(url, "http://example.local:8123");
        assert_eq!(token, "tok");
        set_hass_url_override("");
        set_hass_token_override("");
    }

    #[test]
    fn schemas_have_expected_names() {
        assert_eq!(ha_list_entities_schema()["name"], json!("ha_list_entities"));
        assert_eq!(ha_get_state_schema()["name"], json!("ha_get_state"));
        assert_eq!(ha_list_services_schema()["name"], json!("ha_list_services"));
        assert_eq!(ha_call_service_schema()["name"], json!("ha_call_service"));
        assert_eq!(
            ha_call_service_schema()["parameters"]["required"],
            json!(["domain", "service"])
        );
    }
}
