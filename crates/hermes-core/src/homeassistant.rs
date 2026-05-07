use std::collections::BTreeMap;
use std::env;
use std::time::Duration;

use reqwest::Url;
use reqwest::blocking::{Client, Response};
use serde_json::{Map, Value, json};

use crate::tools::{ToolRuntime, tool_error, tool_result};

const DEFAULT_HASS_URL: &str = "http://homeassistant.local:8123";
const REQUEST_TIMEOUT_SECS: u64 = 15;
const BLOCKED_DOMAINS: &[&str] = &[
    "shell_command",
    "command_line",
    "python_script",
    "pyscript",
    "hassio",
    "rest_command",
];

pub fn homeassistant_available() -> bool {
    !hass_token().is_empty()
}

pub fn ha_list_entities_schema() -> Value {
    json!({
        "name": "ha_list_entities",
        "description": "List Home Assistant entities. Optionally filter by domain such as light or climate, or by area name such as living room or kitchen.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Optional entity domain filter such as light, switch, climate, sensor, binary_sensor, cover, fan, or media_player."
                },
                "area": {
                    "type": "string",
                    "description": "Optional room or area filter such as living room or kitchen."
                }
            },
            "required": []
        }
    })
}

pub fn ha_get_state_schema() -> Value {
    json!({
        "name": "ha_get_state",
        "description": "Get the detailed state and attributes of a single Home Assistant entity.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Entity id such as light.living_room or climate.thermostat."
                }
            },
            "required": ["entity_id"]
        }
    })
}

pub fn ha_list_services_schema() -> Value {
    json!({
        "name": "ha_list_services",
        "description": "List available Home Assistant services and their fields. Optionally filter by domain such as light or climate.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Optional domain filter."
                }
            },
            "required": []
        }
    })
}

pub fn ha_call_service_schema() -> Value {
    json!({
        "name": "ha_call_service",
        "description": "Call a Home Assistant service to control a device or trigger an action.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Service domain such as light, switch, climate, cover, media_player, fan, scene, or script."
                },
                "service": {
                    "type": "string",
                    "description": "Service name such as turn_on, turn_off, toggle, set_temperature, or set_volume_level."
                },
                "entity_id": {
                    "type": "string",
                    "description": "Optional target entity id."
                },
                "data": {
                    "description": "Optional extra service payload as either a JSON object or a JSON string."
                }
            },
            "required": ["domain", "service"]
        }
    })
}

pub fn handle_ha_list_entities(args: &Value, _runtime: &ToolRuntime) -> String {
    let domain = match optional_trimmed_string(args, "domain") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if let Some(domain) = &domain
        && !is_valid_name(domain)
    {
        return tool_error(format!("Invalid domain format: {domain:?}"));
    }
    let area = match optional_trimmed_string(args, "area") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match list_entities(domain.as_deref(), area.as_deref()) {
        Ok(result) => tool_result(json!({ "success": true, "result": result })),
        Err(error) => tool_error(format!("Failed to list entities: {error}")),
    }
}

pub fn handle_ha_get_state(args: &Value, _runtime: &ToolRuntime) -> String {
    let entity_id = match required_trimmed_string(args, "entity_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if !is_valid_entity_id(&entity_id) {
        return tool_error(format!("Invalid entity_id format: {entity_id}"));
    }

    match get_state(&entity_id) {
        Ok(result) => tool_result(json!({ "success": true, "result": result })),
        Err(error) => tool_error(format!("Failed to get state for {entity_id}: {error}")),
    }
}

pub fn handle_ha_list_services(args: &Value, _runtime: &ToolRuntime) -> String {
    let domain = match optional_trimmed_string(args, "domain") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if let Some(domain) = &domain
        && !is_valid_name(domain)
    {
        return tool_error(format!("Invalid domain format: {domain:?}"));
    }

    match list_services(domain.as_deref()) {
        Ok(result) => tool_result(json!({ "success": true, "result": result })),
        Err(error) => tool_error(format!("Failed to list services: {error}")),
    }
}

pub fn handle_ha_call_service(args: &Value, _runtime: &ToolRuntime) -> String {
    let domain = match required_trimmed_string(args, "domain") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let service = match required_trimmed_string(args, "service") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if !is_valid_name(&domain) {
        return tool_error(format!("Invalid domain format: {domain:?}"));
    }
    if !is_valid_name(&service) {
        return tool_error(format!("Invalid service format: {service:?}"));
    }
    if BLOCKED_DOMAINS.contains(&domain.as_str()) {
        return tool_result(json!({
            "success": false,
            "error": format!(
                "Service domain '{}' is blocked for security. Blocked domains: {}",
                domain,
                BLOCKED_DOMAINS.join(", ")
            ),
        }));
    }

    let entity_id = match optional_trimmed_string(args, "entity_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if let Some(entity_id) = &entity_id
        && !is_valid_entity_id(entity_id)
    {
        return tool_error(format!("Invalid entity_id format: {entity_id}"));
    }

    let data = match optional_service_data(args.get("data")) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    match call_service(&domain, &service, entity_id.as_deref(), data) {
        Ok(result) => tool_result(json!({ "success": true, "result": result })),
        Err(error) => tool_error(format!("Failed to call {domain}.{service}: {error}")),
    }
}

fn list_entities(domain: Option<&str>, area: Option<&str>) -> Result<Value, String> {
    let states = request_json("GET", "/api/states", None)?;
    let states = states
        .as_array()
        .ok_or_else(|| "Home Assistant returned an unexpected states payload".to_string())?;

    let mut entities = Vec::new();
    let area_lower = area.map(|value| value.to_ascii_lowercase());

    for state in states {
        let Some(entity_id) = state.get("entity_id").and_then(Value::as_str) else {
            continue;
        };
        if let Some(domain) = domain
            && !entity_id.starts_with(&format!("{domain}."))
        {
            continue;
        }

        let attributes = state.get("attributes").and_then(Value::as_object);
        if let Some(area_lower) = area_lower.as_deref() {
            let friendly = attributes
                .and_then(|attrs| attrs.get("friendly_name"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();
            let area_name = attributes
                .and_then(|attrs| attrs.get("area"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !friendly.contains(area_lower) && !area_name.contains(area_lower) {
                continue;
            }
        }

        entities.push(json!({
            "entity_id": entity_id,
            "state": state.get("state").and_then(Value::as_str).unwrap_or_default(),
            "friendly_name": attributes
                .and_then(|attrs| attrs.get("friendly_name"))
                .and_then(Value::as_str)
                .unwrap_or_default(),
        }));
    }

    Ok(json!({
        "count": entities.len(),
        "entities": entities,
    }))
}

fn get_state(entity_id: &str) -> Result<Value, String> {
    let payload = request_json("GET", &format!("/api/states/{entity_id}"), None)?;
    Ok(json!({
        "entity_id": payload.get("entity_id").and_then(Value::as_str).unwrap_or(entity_id),
        "state": payload.get("state").and_then(Value::as_str).unwrap_or_default(),
        "attributes": payload.get("attributes").cloned().unwrap_or_else(|| json!({})),
        "last_changed": payload.get("last_changed").cloned().unwrap_or(Value::Null),
        "last_updated": payload.get("last_updated").cloned().unwrap_or(Value::Null),
    }))
}

fn list_services(domain: Option<&str>) -> Result<Value, String> {
    let payload = request_json("GET", "/api/services", None)?;
    let items = payload
        .as_array()
        .ok_or_else(|| "Home Assistant returned an unexpected services payload".to_string())?;

    let mut domains = Vec::new();
    for item in items {
        let Some(item_domain) = item.get("domain").and_then(Value::as_str) else {
            continue;
        };
        if let Some(domain) = domain
            && item_domain != domain
        {
            continue;
        }

        let mut services = BTreeMap::new();
        if let Some(service_map) = item.get("services").and_then(Value::as_object) {
            for (service_name, info) in service_map {
                let description = info
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let fields = info
                    .get("fields")
                    .and_then(Value::as_object)
                    .map(|fields| {
                        fields
                            .iter()
                            .filter_map(|(field, detail)| {
                                detail
                                    .as_object()
                                    .and_then(|detail| detail.get("description"))
                                    .and_then(Value::as_str)
                                    .map(|text| (field.clone(), Value::String(text.to_string())))
                            })
                            .collect::<Map<String, Value>>()
                    })
                    .unwrap_or_default();
                let mut entry = Map::new();
                entry.insert(
                    "description".to_string(),
                    Value::String(description.to_string()),
                );
                if !fields.is_empty() {
                    entry.insert("fields".to_string(), Value::Object(fields));
                }
                services.insert(service_name.clone(), Value::Object(entry));
            }
        }

        domains.push(json!({
            "domain": item_domain,
            "services": services,
        }));
    }

    Ok(json!({
        "count": domains.len(),
        "domains": domains,
    }))
}

fn call_service(
    domain: &str,
    service: &str,
    entity_id: Option<&str>,
    data: Option<Value>,
) -> Result<Value, String> {
    let mut payload = match data {
        None => Map::new(),
        Some(Value::Object(map)) => map,
        Some(_) => return Err("data must be a JSON object or JSON string".to_string()),
    };
    if let Some(entity_id) = entity_id {
        payload.insert(
            "entity_id".to_string(),
            Value::String(entity_id.to_string()),
        );
    }

    let result = request_json(
        "POST",
        &format!("/api/services/{domain}/{service}"),
        Some(Value::Object(payload)),
    )?;
    let affected = result
        .as_array()
        .map(|items| {
            items.iter()
                .map(|item| {
                    json!({
                        "entity_id": item.get("entity_id").and_then(Value::as_str).unwrap_or_default(),
                        "state": item.get("state").and_then(Value::as_str).unwrap_or_default(),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(json!({
        "success": true,
        "service": format!("{domain}.{service}"),
        "affected_entities": affected,
    }))
}

fn request_json(method: &str, path: &str, body: Option<Value>) -> Result<Value, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()
        .map_err(|error| error.to_string())?;
    let base_url = hass_url()?;
    let url = base_url
        .join(path)
        .map_err(|error| format!("Invalid Home Assistant path: {error}"))?;
    let token = hass_token();
    if token.is_empty() {
        return Err("HASS_TOKEN is not set".to_string());
    }

    let mut request = match method {
        "GET" => client.get(url),
        "POST" => client.post(url),
        _ => return Err(format!("Unsupported method: {method}")),
    }
    .bearer_auth(token)
    .header(reqwest::header::CONTENT_TYPE, "application/json");

    if let Some(body) = body {
        request = request.json(&body);
    }

    let response = request.send().map_err(|error| error.to_string())?;
    decode_response(response)
}

fn decode_response(response: Response) -> Result<Value, String> {
    let status = response.status();
    let text = response.text().map_err(|error| error.to_string())?;
    if !status.is_success() {
        let detail = if text.trim().is_empty() {
            format!("HTTP {}", status.as_u16())
        } else {
            format!("HTTP {}: {}", status.as_u16(), text.trim())
        };
        return Err(detail);
    }
    serde_json::from_str(&text).map_err(|error| format!("Invalid JSON response: {error}"))
}

fn hass_url() -> Result<Url, String> {
    let raw = env::var("HASS_URL")
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_HASS_URL.to_string());
    let parsed = Url::parse(&raw).map_err(|error| format!("Invalid HASS_URL: {error}"))?;
    match parsed.scheme() {
        "http" | "https" => Ok(parsed),
        _ => Err("HASS_URL must use http or https".to_string()),
    }
}

fn hass_token() -> String {
    env::var("HASS_TOKEN")
        .ok()
        .map(|value| value.trim().to_string())
        .unwrap_or_default()
}

fn required_trimmed_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

fn optional_trimmed_string(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Some(_) => Err(format!("{key} must be a string")),
    }
}

fn optional_service_data(value: Option<&Value>) -> Result<Option<Value>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(map)) => Ok(Some(Value::Object(map.clone()))),
        Some(Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            let parsed = serde_json::from_str::<Value>(trimmed)
                .map_err(|error| format!("Invalid JSON string in 'data' parameter: {error}"))?;
            if !parsed.is_object() {
                return Err("data JSON string must decode to an object".to_string());
            }
            Ok(Some(parsed))
        }
        Some(_) => Err("data must be a JSON object or JSON string".to_string()),
    }
}

fn is_valid_entity_id(value: &str) -> bool {
    let Some((domain, object_id)) = value.split_once('.') else {
        return false;
    };
    is_valid_domain_name(domain) && is_valid_object_id(object_id)
}

fn is_valid_name(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) if first.is_ascii_lowercase() => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

fn is_valid_domain_name(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(first) if first.is_ascii_lowercase() || first == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

fn is_valid_object_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    use tempfile::TempDir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn rejects_invalid_entity_ids_and_blocked_domains() {
        let runtime = ToolRuntime::default();
        let state = handle_ha_get_state(&json!({"entity_id":"bad/id"}), &runtime);
        let parsed: Value = serde_json::from_str(&state).unwrap();
        assert_eq!(parsed["error"], json!("Invalid entity_id format: bad/id"));

        let blocked = handle_ha_call_service(
            &json!({"domain":"shell_command","service":"turn_on"}),
            &runtime,
        );
        let blocked_json: Value = serde_json::from_str(&blocked).unwrap();
        assert_eq!(blocked_json["success"], json!(false));
        assert!(
            blocked_json["error"]
                .as_str()
                .unwrap()
                .contains("blocked for security")
        );
    }

    #[test]
    fn homeassistant_tools_work_against_fake_server() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        let (base_url, captured, handle) = serve_homeassistant_server();

        unsafe {
            env::set_var("HASS_URL", &base_url);
            env::set_var("HASS_TOKEN", "test-token");
        }

        let entities =
            handle_ha_list_entities(&json!({"domain":"light","area":"living"}), &runtime);
        let entities_json: Value = serde_json::from_str(&entities).unwrap();
        assert_eq!(entities_json["success"], json!(true));
        assert_eq!(entities_json["result"]["count"], json!(1));
        assert_eq!(
            entities_json["result"]["entities"][0]["entity_id"],
            json!("light.living_room")
        );

        let state = handle_ha_get_state(&json!({"entity_id":"light.living_room"}), &runtime);
        let state_json: Value = serde_json::from_str(&state).unwrap();
        assert_eq!(state_json["result"]["state"], json!("on"));

        let services = handle_ha_list_services(&json!({"domain":"light"}), &runtime);
        let services_json: Value = serde_json::from_str(&services).unwrap();
        assert_eq!(services_json["result"]["count"], json!(1));
        assert_eq!(
            services_json["result"]["domains"][0]["services"]["turn_on"]["description"],
            json!("Turn on")
        );

        let called = handle_ha_call_service(
            &json!({
                "domain":"light",
                "service":"turn_on",
                "entity_id":"light.living_room",
                "data":{"brightness":255}
            }),
            &runtime,
        );
        let called_json: Value = serde_json::from_str(&called).unwrap();
        assert_eq!(called_json["success"], json!(true));
        assert_eq!(
            called_json["result"]["affected_entities"][0]["entity_id"],
            json!("light.living_room")
        );

        handle.join().unwrap();
        let captured = captured.lock().unwrap();
        assert!(captured[0].starts_with("GET /api/states"));
        assert!(captured[1].starts_with("GET /api/states/light.living_room"));
        assert!(captured[2].starts_with("GET /api/services"));
        assert!(captured[3].starts_with("POST /api/services/light/turn_on"));
        assert!(captured[3].contains("\"brightness\":255"));
        assert!(captured[3].contains("\"entity_id\":\"light.living_room\""));

        unsafe {
            env::remove_var("HASS_URL");
            env::remove_var("HASS_TOKEN");
        }
    }

    fn serve_homeassistant_server() -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = captured.clone();
        let handle = thread::spawn(move || {
            for _ in 0..4 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 4096];
                loop {
                    let read = stream.read(&mut buffer).unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .unwrap()
                    + 4;
                let headers = String::from_utf8_lossy(&request[..header_end]).to_string();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let mut body = request[header_end..].to_vec();
                while body.len() < content_length {
                    let read = stream.read(&mut buffer).unwrap();
                    if read == 0 {
                        break;
                    }
                    body.extend_from_slice(&buffer[..read]);
                }
                let first_line = headers.lines().next().unwrap_or_default().to_string();
                let body_text = String::from_utf8_lossy(&body).to_string();
                captured_clone
                    .lock()
                    .unwrap()
                    .push(format!("{first_line}\n{body_text}"));

                let response_body = if first_line.starts_with("GET /api/states/light.living_room ")
                {
                    json!({
                        "entity_id": "light.living_room",
                        "state": "on",
                        "attributes": { "friendly_name": "Living Room Light", "brightness": 255 },
                        "last_changed": "2026-05-07T00:00:00+00:00",
                        "last_updated": "2026-05-07T00:00:01+00:00"
                    })
                    .to_string()
                } else if first_line.starts_with("GET /api/states ") {
                    json!([
                        {
                            "entity_id": "light.living_room",
                            "state": "on",
                            "attributes": {
                                "friendly_name": "Living Room Light",
                                "area": "Living Room"
                            }
                        },
                        {
                            "entity_id": "switch.garage",
                            "state": "off",
                            "attributes": {
                                "friendly_name": "Garage Switch",
                                "area": "Garage"
                            }
                        }
                    ])
                    .to_string()
                } else if first_line.starts_with("GET /api/services ") {
                    json!([
                        {
                            "domain": "light",
                            "services": {
                                "turn_on": {
                                    "description": "Turn on",
                                    "fields": {
                                        "brightness": { "description": "Brightness level" }
                                    }
                                }
                            }
                        },
                        {
                            "domain": "switch",
                            "services": {
                                "turn_off": { "description": "Turn off" }
                            }
                        }
                    ])
                    .to_string()
                } else {
                    json!([
                        {
                            "entity_id": "light.living_room",
                            "state": "on"
                        }
                    ])
                    .to_string()
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (format!("http://{addr}"), captured, handle)
    }
}
