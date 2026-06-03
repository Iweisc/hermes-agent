//! Native Rust port of `cli.py` — the Hermes Agent interactive terminal interface.
//!
//! `cli.py` is ~12.5K lines, dominated by an interactive `prompt_toolkit` REPL
//! (`HermesCLI`) and a `fire`-driven `main()` dispatcher. Those layers are bound
//! tightly to the Python TUI/agent runtime and cannot be expressed as a
//! self-contained Rust module. What *is* portable — and ported here faithfully —
//! is the substantial body of pure, side-effect-light helper logic the CLI is
//! built on:
//!
//!   * reasoning/tool-call XML tag stripping (`strip_reasoning_tags`)
//!   * assistant-content text extraction
//!   * config loading + atomic save (`load_cli_config`, `save_config_value`)
//!   * reasoning-effort / service-tier parsing
//!   * file-drop & attachment-path resolution
//!   * leading-path token splitting (quoted / escaped spaces)
//!   * leaked terminal control-response stripping (CPR/DSR + SGR mouse)
//!   * bracketed-paste wrapper stripping
//!   * markdown-syntax stripping + Windows dot-segment preservation
//!   * hex→ANSI color conversion
//!   * image attachment badge formatting
//!   * process-notification formatting
//!   * slash-command detection, skills-argument parsing
//!   * output-history ring buffer + ANSI control stripping
//!   * compact banner construction
//!
//! These mirror the Python originals line-for-line in behavior. The interactive
//! REPL and `main()` dispatch are intentionally out of scope for this flat
//! module (they live in the gateway/TUI surfaces and depend on the live agent).

use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::Mutex;

use regex::Regex;
use regex::RegexBuilder;
use serde_json::Value;

// ============================================================================
// Constants
// ============================================================================

/// Reasoning tags whose paired/unterminated/orphan forms are scrubbed from
/// visible assistant content. Mirrors `_REASONING_TAGS` in cli.py.
pub const REASONING_TAGS: &[&str] = &[
    "REASONING_SCRATCHPAD",
    "think",
    "thinking",
    "reasoning",
    "thought",
];

/// Tool-call XML block tags scrubbed from visible content.
pub const TOOL_CALL_TAGS: &[&str] = &[
    "tool_call",
    "tool_calls",
    "tool_result",
    "function_call",
    "function_calls",
];

/// Known image file extensions (lowercase, with leading dot).
/// Mirrors `_IMAGE_EXTENSIONS`.
pub const IMAGE_EXTENSIONS: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg",
    ".ico",
];

/// Spinner frames used by command/busy indicators.
pub const COMMAND_SPINNER_FRAMES: &[&str] =
    &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Default true-color accent ANSI (bold #FFD700).
pub const ACCENT_ANSI_DEFAULT: &str = "\x1b[1;38;2;255;215;0m";
pub const BOLD: &str = "\x1b[1m";
pub const RST: &str = "\x1b[0m";
/// 4-space indent for streamed response text (matches Panel padding).
pub const STREAM_PAD: &str = "    ";

/// Terminal input-mode reset sequence emitted on recovery. Mirrors
/// `_TERMINAL_INPUT_MODE_RESET_SEQ`.
pub const TERMINAL_INPUT_MODE_RESET_SEQ: &str = concat!(
    "\x1b[?1006l", // disable SGR mouse
    "\x1b[?1003l", // disable any-motion tracking
    "\x1b[?1002l", // disable button-motion tracking
    "\x1b[?1000l", // disable click tracking
    "\x1b[?1004l", // disable focus events
    "\x1b[?2004l", // disable bracketed paste
    "\x1b[?1049l", // leave alt screen (if stuck there)
    "\x1b[<u",     // pop kitty keyboard mode
    "\x1b[>4m",    // reset modifyOtherKeys
    "\x1b[0m",     // reset text attributes
    "\x1b[?25h"    // ensure cursor visible
);

// ============================================================================
// Reasoning / tool-call tag stripping
// ============================================================================

/// Remove reasoning/thinking blocks and tool-call XML from displayed text.
///
/// Faithful port of `_strip_reasoning_tags`. Handles closed pairs,
/// unterminated open tags running to end-of-text, and stray orphan close
/// tags, for each reasoning tag. Then scrubs tool-call XML blocks and
/// boundary-gated `<function name="...">…</function>` emissions, plus stray
/// tool-call close tags. Returns the trimmed result.
pub fn strip_reasoning_tags(text: &str) -> String {
    let mut cleaned = text.to_string();

    for tag in REASONING_TAGS {
        // Closed pair — case-insensitive, DOTALL.
        let closed = RegexBuilder::new(&format!(r"<{0}>.*?</{0}>\s*", regex::escape(tag)))
            .case_insensitive(true)
            .dot_matches_new_line(true)
            .build()
            .unwrap();
        cleaned = closed.replace_all(&cleaned, "").into_owned();

        // Unterminated open tag — strip from the tag to end of text.
        let open = RegexBuilder::new(&format!(r"<{0}>.*$", regex::escape(tag)))
            .case_insensitive(true)
            .dot_matches_new_line(true)
            .build()
            .unwrap();
        cleaned = open.replace_all(&cleaned, "").into_owned();

        // Stray orphan close tag.
        let orphan = RegexBuilder::new(&format!(r"</{0}>\s*", regex::escape(tag)))
            .case_insensitive(true)
            .build()
            .unwrap();
        cleaned = orphan.replace_all(&cleaned, "").into_owned();
    }

    // Tool-call XML blocks.
    for tc in TOOL_CALL_TAGS {
        let re = RegexBuilder::new(&format!(r"<{0}\b[^>]*>.*?</{0}>\s*", regex::escape(tc)))
            .case_insensitive(true)
            .dot_matches_new_line(true)
            .build()
            .unwrap();
        cleaned = re.replace_all(&cleaned, "").into_owned();
    }

    // <function name="..."> — boundary + attribute gated to avoid prose FPs.
    // The Python lookbehind alternation `(?:(?<=^)|(?<=[\n\r.!?:]))` is not
    // supported by Rust's `regex`; emulate with a leading capture group that
    // matches start-of-text OR one of the boundary chars, and re-emit it.
    cleaned = strip_function_named_blocks(&cleaned);

    // Stray tool-call close tags.
    let stray = RegexBuilder::new(
        r"</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*",
    )
    .case_insensitive(true)
    .build()
    .unwrap();
    cleaned = stray.replace_all(&cleaned, "").into_owned();

    cleaned.trim().to_string()
}

/// Emulate the boundary-gated `<function name="...">…</function>` strip.
///
/// Python uses a leading lookbehind alternation `(?:(?<=^)|(?<=[\n\r.!?:]))`
/// plus a negative-lookahead body `(?:(?!</function>).)*`. Rust's `regex`
/// supports neither lookbehind nor lookahead, so we (1) capture the boundary
/// char as a leading group and re-emit it, and (2) use a non-greedy body
/// `.*?` which stops at the first `</function>` — behaviorally identical for
/// well-formed input.
fn strip_function_named_blocks(input: &str) -> String {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        RegexBuilder::new(
            r"(^|[\n\r.!?:])[ \t]*<function\b[^>]*\bname\s*=[^>]*>.*?</function>\s*",
        )
        .case_insensitive(true)
        .dot_matches_new_line(true)
        .build()
        .unwrap()
    });

    RE.replace_all(input, |caps: &regex::Captures| caps[1].to_string())
        .into_owned()
}

/// Coerce assistant `content` (string, list of content parts, or null) into a
/// joined text string. Mirrors `_assistant_content_as_text`.
pub fn assistant_content_as_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let texts: Vec<String> = parts
                .iter()
                .filter_map(|part| {
                    if let Value::Object(map) = part {
                        if map.get("type").and_then(|v| v.as_str()) == Some("text") {
                            return Some(
                                map.get("text")
                                    .map(value_to_string)
                                    .unwrap_or_default(),
                            );
                        }
                    }
                    None
                })
                .filter(|p| !p.is_empty())
                .collect();
            texts.join("\n")
        }
        other => value_to_string(other),
    }
}

/// `str(content)` equivalent for non-string JSON scalars.
fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Strip reasoning tags from the textual rendering of assistant content.
/// Mirrors `_assistant_copy_text`.
pub fn assistant_copy_text(content: &Value) -> String {
    strip_reasoning_tags(&assistant_content_as_text(content))
}

// ============================================================================
// Reasoning effort / service tier parsing
// ============================================================================

/// Parse a persisted service-tier preference into a Responses API value.
/// Mirrors `_parse_service_tier_config`. Returns `None` for normal/default,
/// `Some("priority")` for fast/priority/on, and `None` (with a warning in the
/// Python original) for unknown values.
pub fn parse_service_tier_config(raw: &str) -> Option<String> {
    let value = raw.trim().to_lowercase();
    if value.is_empty()
        || matches!(value.as_str(), "normal" | "default" | "standard" | "off" | "none")
    {
        return None;
    }
    if matches!(value.as_str(), "fast" | "priority" | "on") {
        return Some("priority".to_string());
    }
    // Unknown service_tier — Python logs a warning and returns None.
    None
}

// ============================================================================
// Config loading
// ============================================================================

/// Resolve the Hermes home directory (`~/.hermes` unless `HERMES_HOME` is set).
pub fn hermes_home() -> PathBuf {
    if let Ok(h) = std::env::var("HERMES_HOME") {
        if !h.trim().is_empty() {
            return PathBuf::from(h);
        }
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".hermes")
}

/// Default personalities map (subset of `agent.personalities` defaults).
pub fn default_personalities() -> Vec<(&'static str, &'static str)> {
    vec![
        ("helpful", "You are a helpful, friendly AI assistant."),
        (
            "concise",
            "You are a concise assistant. Keep responses brief and to the point.",
        ),
        (
            "technical",
            "You are a technical expert. Provide detailed, accurate technical information.",
        ),
        (
            "creative",
            "You are a creative assistant. Think outside the box and offer innovative solutions.",
        ),
        (
            "teacher",
            "You are a patient teacher. Explain concepts clearly with examples.",
        ),
    ]
}

/// Build the default CLI configuration tree as a JSON `Value`. This mirrors the
/// `defaults` dict inside `load_cli_config()`.
pub fn default_cli_config() -> Value {
    serde_json::json!({
        "model": {"default": "", "base_url": "", "provider": "auto"},
        "terminal": {
            "env_type": "local",
            "cwd": ".",
            "timeout": 60,
            "lifetime_seconds": 300,
            "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_forward_env": [],
            "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
            "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
            "docker_volumes": [],
            "docker_mount_cwd_to_workspace": false
        },
        "browser": {
            "inactivity_timeout": 120,
            "record_sessions": false,
            "engine": "auto"
        },
        "compression": {"enabled": true, "threshold": 0.50},
        "agent": {
            "max_turns": 90,
            "verbose": false,
            "system_prompt": "",
            "prefill_messages_file": "",
            "reasoning_effort": "",
            "service_tier": "",
            "personalities": {
                "helpful": "You are a helpful, friendly AI assistant.",
                "concise": "You are a concise assistant. Keep responses brief and to the point.",
                "technical": "You are a technical expert. Provide detailed, accurate technical information.",
                "creative": "You are a creative assistant. Think outside the box and offer innovative solutions.",
                "teacher": "You are a patient teacher. Explain concepts clearly with examples."
            }
        },
        "display": {
            "compact": false,
            "resume_display": "full",
            "show_reasoning": false,
            "streaming": true,
            "busy_input_mode": "interrupt",
            "persistent_output": true,
            "persistent_output_max_lines": 200,
            "skin": "default"
        },
        "clarify": {"timeout": 120},
        "code_execution": {"timeout": 300, "max_tool_calls": 50},
        "auxiliary": {
            "vision": {"provider": "auto", "model": "", "base_url": "", "api_key": ""},
            "web_extract": {"provider": "auto", "model": "", "base_url": "", "api_key": ""}
        },
        "delegation": {
            "max_iterations": 45,
            "model": "",
            "provider": "",
            "base_url": "",
            "api_key": ""
        },
        "onboarding": {"seen": {}}
    })
}

/// Deep-merge `src` into `dst` following the cli.py merge rules: dict-into-dict
/// keys are recursively `update`-merged (top-level scalars overwrite), and keys
/// only present in `src` are carried over.
fn dict_update(dst: &mut Value, src: &Value) {
    if let (Value::Object(dmap), Value::Object(smap)) = (dst, src) {
        for (k, v) in smap {
            dmap.insert(k.clone(), v.clone());
        }
    }
}

/// Load CLI configuration mirroring `load_cli_config()`'s file merge logic.
///
/// `user_config` and `project_config` are the parsed YAML/JSON of
/// `~/.hermes/config.yaml` and `./cli-config.yaml` respectively (either may be
/// `None` if absent). `ignore_user_config` corresponds to
/// `HERMES_IGNORE_USER_CONFIG=1`. This function performs the deep-merge into
/// defaults; the env-var bridging side effects of the Python version are
/// returned separately via [`config_env_bridge`] so callers can apply them.
pub fn load_cli_config(
    user_config: Option<&Value>,
    project_config: Option<&Value>,
    ignore_user_config: bool,
) -> Value {
    let mut defaults = default_cli_config();

    let file_config: Option<&Value> = if user_config.is_some() && !ignore_user_config {
        user_config
    } else {
        project_config
    };

    let file_config = match file_config {
        Some(Value::Object(_)) => file_config.unwrap(),
        _ => return defaults,
    };

    // Handle model config — string (new) or dict (old).
    if let Some(model) = file_config.get("model") {
        match model {
            Value::String(s) => {
                defaults["model"]["default"] = Value::String(s.clone());
            }
            Value::Object(mmap) => {
                dict_update(&mut defaults["model"], model);
                // Promote model.model → model.default when default unset.
                if mmap.contains_key("model") && !mmap.contains_key("default") {
                    defaults["model"]["default"] = mmap["model"].clone();
                }
            }
            _ => {}
        }
    }

    // Legacy root-level provider/base_url fallback.
    if defaults["model"]
        .get("provider")
        .and_then(|v| v.as_str())
        .map(|s| s.is_empty())
        .unwrap_or(true)
    {
        if let Some(p) = file_config.get("provider") {
            if !p.is_null() {
                defaults["model"]["provider"] = p.clone();
            }
        }
    }
    if defaults["model"]
        .get("base_url")
        .and_then(|v| v.as_str())
        .map(|s| s.is_empty())
        .unwrap_or(true)
    {
        if let Some(b) = file_config.get("base_url") {
            if !b.is_null() {
                defaults["model"]["base_url"] = b.clone();
            }
        }
    }

    // Deep-merge keys present in both (model already handled).
    let default_keys: Vec<String> = defaults
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    for key in &default_keys {
        if key == "model" {
            continue;
        }
        if let Some(fv) = file_config.get(key) {
            let dv = &mut defaults[key];
            if dv.is_object() && fv.is_object() {
                dict_update(dv, fv);
            } else {
                *dv = fv.clone();
            }
        }
    }

    // Carry over keys from file_config absent in defaults (except model).
    if let Value::Object(fmap) = file_config {
        for (key, val) in fmap {
            if key != "model" && !default_keys.contains(key) {
                defaults[key] = val.clone();
            }
        }
    }

    // Legacy root-level max_turns → agent.max_turns when nested key missing.
    if let Some(mt) = file_config.get("max_turns") {
        let agent_has = file_config
            .get("agent")
            .and_then(|a| a.as_object())
            .and_then(|a| a.get("max_turns"))
            .map(|v| !v.is_null())
            .unwrap_or(false);
        if !agent_has {
            defaults["agent"]["max_turns"] = mt.clone();
        }
    }

    defaults
}

/// Compute the environment-variable bridge that `load_cli_config()` applies as
/// a side effect. Returns `(env_var, value)` pairs that should be exported.
///
/// `is_gateway` corresponds to `_HERMES_GATEWAY=1`; when true `TERMINAL_CWD` is
/// skipped. `file_has_terminal_config` corresponds to whether the loaded config
/// file had a `terminal:` section. `existing_env` lets the caller mirror the
/// "only set if file has config OR env not already set" rule.
pub fn config_env_bridge(
    config: &Value,
    cwd: &str,
    is_gateway: bool,
    file_has_terminal_config: bool,
    existing_env: &dyn Fn(&str) -> bool,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();

    let terminal = config.get("terminal").cloned().unwrap_or(Value::Null);
    let mut terminal = terminal.as_object().cloned().unwrap_or_default();

    // Normalize "backend" → "env_type" (backend takes precedence).
    if let Some(b) = terminal.get("backend").cloned() {
        terminal.insert("env_type".to_string(), b);
    }

    let effective_backend = terminal
        .get("env_type")
        .and_then(|v| v.as_str())
        .unwrap_or("local")
        .to_string();

    let cwd_placeholders = ["", ".", "auto", "cwd"];
    if effective_backend == "local" {
        terminal.insert("cwd".to_string(), Value::String(cwd.to_string()));
    } else if let Some(c) = terminal.get("cwd").and_then(|v| v.as_str()) {
        if cwd_placeholders.contains(&c) {
            terminal.remove("cwd");
        }
    }

    let env_mappings: &[(&str, &str)] = &[
        ("env_type", "TERMINAL_ENV"),
        ("cwd", "TERMINAL_CWD"),
        ("timeout", "TERMINAL_TIMEOUT"),
        ("lifetime_seconds", "TERMINAL_LIFETIME_SECONDS"),
        ("docker_image", "TERMINAL_DOCKER_IMAGE"),
        ("docker_forward_env", "TERMINAL_DOCKER_FORWARD_ENV"),
        ("singularity_image", "TERMINAL_SINGULARITY_IMAGE"),
        ("modal_image", "TERMINAL_MODAL_IMAGE"),
        ("daytona_image", "TERMINAL_DAYTONA_IMAGE"),
        ("vercel_runtime", "TERMINAL_VERCEL_RUNTIME"),
        ("ssh_host", "TERMINAL_SSH_HOST"),
        ("ssh_user", "TERMINAL_SSH_USER"),
        ("ssh_port", "TERMINAL_SSH_PORT"),
        ("ssh_key", "TERMINAL_SSH_KEY"),
        ("container_cpu", "TERMINAL_CONTAINER_CPU"),
        ("container_memory", "TERMINAL_CONTAINER_MEMORY"),
        ("container_disk", "TERMINAL_CONTAINER_DISK"),
        ("container_persistent", "TERMINAL_CONTAINER_PERSISTENT"),
        ("docker_volumes", "TERMINAL_DOCKER_VOLUMES"),
        ("docker_mount_cwd_to_workspace", "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE"),
        ("docker_run_as_host_user", "TERMINAL_DOCKER_RUN_AS_HOST_USER"),
        ("sandbox_dir", "TERMINAL_SANDBOX_DIR"),
        ("persistent_shell", "TERMINAL_PERSISTENT_SHELL"),
        ("sudo_password", "SUDO_PASSWORD"),
    ];

    for (config_key, env_var) in env_mappings {
        if let Some(val) = terminal.get(*config_key) {
            if *env_var == "TERMINAL_CWD" {
                if is_gateway {
                    continue;
                }
                out.push((env_var.to_string(), json_scalar_to_string(val)));
                continue;
            }
            if file_has_terminal_config || !existing_env(env_var) {
                out.push((env_var.to_string(), json_scalar_to_string(val)));
            }
        }
    }

    // Browser config.
    if let Some(browser) = config.get("browser").and_then(|v| v.as_object()) {
        if let Some(v) = browser.get("inactivity_timeout") {
            out.push((
                "BROWSER_INACTIVITY_TIMEOUT".to_string(),
                json_scalar_to_string(v),
            ));
        }
    }

    // Auxiliary task overrides.
    let aux_tasks: &[(&str, [&str; 4])] = &[
        (
            "vision",
            [
                "AUXILIARY_VISION_PROVIDER",
                "AUXILIARY_VISION_MODEL",
                "AUXILIARY_VISION_BASE_URL",
                "AUXILIARY_VISION_API_KEY",
            ],
        ),
        (
            "web_extract",
            [
                "AUXILIARY_WEB_EXTRACT_PROVIDER",
                "AUXILIARY_WEB_EXTRACT_MODEL",
                "AUXILIARY_WEB_EXTRACT_BASE_URL",
                "AUXILIARY_WEB_EXTRACT_API_KEY",
            ],
        ),
        (
            "approval",
            [
                "AUXILIARY_APPROVAL_PROVIDER",
                "AUXILIARY_APPROVAL_MODEL",
                "AUXILIARY_APPROVAL_BASE_URL",
                "AUXILIARY_APPROVAL_API_KEY",
            ],
        ),
    ];

    let auxiliary = config.get("auxiliary").and_then(|v| v.as_object());
    if let Some(auxiliary) = auxiliary {
        for (task_key, env_map) in aux_tasks {
            let task_cfg = match auxiliary.get(*task_key).and_then(|v| v.as_object()) {
                Some(c) => c,
                None => continue,
            };
            let get = |k: &str| {
                task_cfg
                    .get(k)
                    .map(json_scalar_to_string)
                    .unwrap_or_default()
                    .trim()
                    .to_string()
            };
            let prov = get("provider");
            let model = get("model");
            let base_url = get("base_url");
            let api_key = get("api_key");
            if !prov.is_empty() && prov != "auto" {
                out.push((env_map[0].to_string(), prov));
            }
            if !model.is_empty() {
                out.push((env_map[1].to_string(), model));
            }
            if !base_url.is_empty() {
                out.push((env_map[2].to_string(), base_url));
            }
            if !api_key.is_empty() {
                out.push((env_map[3].to_string(), api_key));
            }
        }
    }

    // Security settings.
    if let Some(security) = config.get("security").and_then(|v| v.as_object()) {
        if let Some(redact) = security.get("redact_secrets") {
            if !redact.is_null() {
                let lowered = json_scalar_to_string(redact).to_lowercase();
                out.push(("HERMES_REDACT_SECRETS".to_string(), lowered));
            }
        }
    }

    out
}

/// Render a JSON scalar/list as Python's `str()`/`json.dumps()` bridge does:
/// lists are JSON-serialized, scalars use their string form.
fn json_scalar_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            // Python str(True) -> "True"; the env bridge uses str() for scalars.
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Null => "None".to_string(),
        Value::Array(_) | Value::Object(_) => v.to_string(),
        other => other.to_string(),
    }
}

/// Save a value to the active config file at a dot-separated key path, mirroring
/// `save_config_value`. The caller supplies the target config file path (the
/// Python resolution logic — user config if it exists, else project config —
/// should be applied by the caller via [`resolve_save_config_path`]).
///
/// Returns `Ok(())` on success. The file is written atomically (temp + rename)
/// and chmod 0o600 on Unix.
pub fn save_config_value(
    config_path: &Path,
    key_path: &str,
    value: Value,
) -> std::io::Result<()> {
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut config: Value = if config_path.exists() {
        let raw = std::fs::read_to_string(config_path)?;
        serde_yaml::from_str(&raw).unwrap_or(Value::Object(Default::default()))
    } else {
        Value::Object(Default::default())
    };
    if !config.is_object() {
        config = Value::Object(Default::default());
    }

    let keys: Vec<&str> = key_path.split('.').collect();
    {
        let mut current = &mut config;
        for key in &keys[..keys.len() - 1] {
            let obj = current.as_object_mut().unwrap();
            let entry = obj
                .entry((*key).to_string())
                .or_insert_with(|| Value::Object(Default::default()));
            if !entry.is_object() {
                *entry = Value::Object(Default::default());
            }
            current = entry;
        }
        current
            .as_object_mut()
            .unwrap()
            .insert(keys[keys.len() - 1].to_string(), value);
    }

    // Atomic write: temp file in same dir, then rename.
    let yaml = serde_yaml::to_string(&config)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let parent = config_path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(
        ".{}.tmp",
        config_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("config")
    ));
    std::fs::write(&tmp, yaml)?;
    std::fs::rename(&tmp, config_path)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(config_path, std::fs::Permissions::from_mode(0o600));
    }

    Ok(())
}

/// Resolve the config file the CLI writes to: user config if it exists, else
/// project config. Mirrors `save_config_value`'s path selection.
pub fn resolve_save_config_path(project_dir: &Path) -> PathBuf {
    let user = hermes_home().join("config.yaml");
    if user.exists() {
        user
    } else {
        project_dir.join("cli-config.yaml")
    }
}

// ============================================================================
// Personality resolution
// ============================================================================

/// Accept a string or dict personality value and return the system prompt
/// string. Mirrors `HermesCLI._resolve_personality_prompt`.
pub fn resolve_personality_prompt(value: &Value) -> String {
    if let Value::Object(map) = value {
        let mut parts: Vec<String> = Vec::new();
        parts.push(
            map.get("system_prompt")
                .map(value_to_string)
                .unwrap_or_default(),
        );
        if let Some(tone) = map.get("tone") {
            if !value_to_string(tone).is_empty() {
                parts.push(format!("Tone: {}", value_to_string(tone)));
            }
        }
        if let Some(style) = map.get("style") {
            if !value_to_string(style).is_empty() {
                parts.push(format!("Style: {}", value_to_string(style)));
            }
        }
        return parts
            .into_iter()
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
    }
    value_to_string(value)
}

// ============================================================================
// Slash command / skills argument parsing
// ============================================================================

/// Return true if *text* looks like a slash command, not a file path.
/// Mirrors `_looks_like_slash_command`.
pub fn looks_like_slash_command(text: &str) -> bool {
    if text.is_empty() || !text.starts_with('/') {
        return false;
    }
    let first_word = text.split_whitespace().next().unwrap_or("");
    // After the leading '/', a command name has no slashes; a path does.
    !first_word.get(1..).unwrap_or("").contains('/')
}

/// Normalize a CLI skills flag into a deduplicated list. Mirrors
/// `_parse_skills_argument`. Accepts a single string (comma-separated) or a
/// list of strings.
pub fn parse_skills_argument(skills: &[String]) -> Vec<String> {
    let mut parsed: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for raw in skills {
        for part in raw.split(',') {
            let normalized = part.trim();
            if normalized.is_empty() || seen.contains(normalized) {
                continue;
            }
            seen.insert(normalized.to_string());
            parsed.push(normalized.to_string());
        }
    }
    parsed
}

// ============================================================================
// Path / attachment helpers
// ============================================================================

/// Split a leading file-path token from trailing free-form text. Supports
/// quoted paths and backslash-escaped spaces. Mirrors `_split_path_input`.
pub fn split_path_input(raw: &str) -> (String, String) {
    let raw = raw.trim();
    if raw.is_empty() {
        return (String::new(), String::new());
    }

    let chars: Vec<char> = raw.chars().collect();
    let first = chars[0];

    if first == '"' || first == '\'' {
        let quote = first;
        let mut pos = 1usize;
        while pos < chars.len() {
            let ch = chars[pos];
            if ch == '\\' && pos + 1 < chars.len() {
                pos += 2;
                continue;
            }
            if ch == quote {
                let token: String = chars[1..pos].iter().collect();
                let remainder: String = chars[pos + 1..].iter().collect::<String>().trim().to_string();
                return (token, remainder);
            }
            pos += 1;
        }
        let token: String = chars[1..].iter().collect();
        return (token, String::new());
    }

    let mut pos = 0usize;
    while pos < chars.len() {
        let ch = chars[pos];
        if ch == '\\' && pos + 1 < chars.len() && chars[pos + 1] == ' ' {
            pos += 2;
        } else if ch == ' ' {
            break;
        } else {
            pos += 1;
        }
    }

    let token: String = chars[..pos].iter().collect::<String>().replace("\\ ", " ");
    let remainder: String = chars[pos..].iter().collect::<String>().trim().to_string();
    (token, remainder)
}

/// Return true if the suffix (with leading dot, lowercased) is a known image
/// extension.
pub fn is_image_suffix(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => {
            let dotted = format!(".{}", ext.to_lowercase());
            IMAGE_EXTENSIONS.contains(&dotted.as_str())
        }
        None => false,
    }
}

/// Resolve a user-supplied local attachment path. Accepts quoted/unquoted
/// paths, expands `~` and env vars, resolves relative paths from `TERMINAL_CWD`
/// (or cwd), and returns the resolved path only if it points to an existing
/// file. Mirrors `_resolve_attachment_path`.
///
/// `env_get` resolves env vars (e.g. for `$VAR` expansion and `TERMINAL_CWD`),
/// `home` provides the `~` expansion target, and `cwd` is the working dir.
pub fn resolve_attachment_path(
    raw_path: &str,
    env_get: &dyn Fn(&str) -> Option<String>,
    home: &Path,
    cwd: &Path,
) -> Option<PathBuf> {
    let mut token = raw_path.trim().to_string();
    if token.is_empty() {
        return None;
    }

    if (token.starts_with('"') && token.ends_with('"'))
        || (token.starts_with('\'') && token.ends_with('\''))
    {
        token = token[1..token.len() - 1].trim().to_string();
    }
    token = token.replace("\\ ", " ");
    if token.is_empty() {
        return None;
    }

    let mut expanded = token.clone();
    if token.starts_with("file://") {
        if let Ok(parsed) = url::Url::parse(&token) {
            if parsed.scheme() == "file" {
                expanded = percent_decode(parsed.path());
            }
        }
    }

    expanded = expand_user_and_vars(&expanded, env_get, home);

    // On non-Windows, translate a drive-letter path (D:\... / D:/...) to a
    // WSL-style /mnt/d/... mount.
    if cfg!(not(windows)) {
        let normalized = expanded.replace('\\', "/");
        let nb = normalized.as_bytes();
        if nb.len() >= 3
            && nb[1] == b':'
            && nb[2] == b'/'
            && (nb[0] as char).is_ascii_alphabetic()
        {
            let drive = (nb[0] as char).to_ascii_lowercase();
            expanded = format!("/mnt/{}/{}", drive, &normalized[3..]);
        }
    }

    let mut path = PathBuf::from(&expanded);
    if !path.is_absolute() {
        let base = env_get("TERMINAL_CWD")
            .map(PathBuf::from)
            .unwrap_or_else(|| cwd.to_path_buf());
        path = base.join(&path);
    }

    let resolved = std::fs::canonicalize(&path).unwrap_or(path);

    match std::fs::metadata(&resolved) {
        Ok(md) if md.is_file() => Some(resolved),
        _ => None,
    }
}

/// Minimal percent-decoding for `file://` URL paths.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Expand a leading `~` and `$VAR` / `${VAR}` references, mirroring
/// `os.path.expandvars(os.path.expanduser(...))`.
fn expand_user_and_vars(
    s: &str,
    env_get: &dyn Fn(&str) -> Option<String>,
    home: &Path,
) -> String {
    let mut result = s.to_string();
    if result == "~" {
        result = home.to_string_lossy().into_owned();
    } else if let Some(rest) = result.strip_prefix("~/") {
        result = home.join(rest).to_string_lossy().into_owned();
    }
    expand_env_vars_in_str(&result, env_get)
}

/// Expand `$VAR` and `${VAR}` in a string.
fn expand_env_vars_in_str(s: &str, env_get: &dyn Fn(&str) -> Option<String>) -> String {
    static RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)").unwrap());
    RE.replace_all(s, |caps: &regex::Captures| {
        let name = caps
            .get(1)
            .or_else(|| caps.get(2))
            .map(|m| m.as_str())
            .unwrap_or("");
        env_get(name).unwrap_or_else(|| caps[0].to_string())
    })
    .into_owned()
}

/// Result of a successful file-drop detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileDrop {
    pub path: PathBuf,
    pub is_image: bool,
    pub remainder: String,
}

/// Return true if the stripped text starts like a local file path. Mirrors the
/// `starts_like_path` prefilter inside `_detect_file_drop`.
pub fn starts_like_path(stripped: &str) -> bool {
    let chars: Vec<char> = stripped.chars().collect();
    let drive_at = |i: usize| -> bool {
        chars.len() >= i + 3
            && chars[i + 1] == ':'
            && (chars[i + 2] == '\\' || chars[i + 2] == '/')
            && chars[i].is_ascii_alphabetic()
    };
    stripped.starts_with('/')
        || stripped.starts_with('~')
        || stripped.starts_with("./")
        || stripped.starts_with("../")
        || stripped.starts_with("file://")
        || drive_at(0)
        || stripped.starts_with("\"/")
        || stripped.starts_with("\"~")
        || stripped.starts_with("'/")
        || stripped.starts_with("'~")
        || stripped.starts_with("\"./")
        || stripped.starts_with("\"../")
        || stripped.starts_with("'./")
        || stripped.starts_with("'../")
        || (chars.len() >= 4
            && (chars[0] == '\'' || chars[0] == '"')
            && chars[2] == ':'
            && (chars[3] == '\\' || chars[3] == '/')
            && chars[1].is_ascii_alphabetic())
}

/// Detect whether *user_input* starts with a real local file path. Mirrors
/// `_detect_file_drop`. The path-resolution callbacks match
/// [`resolve_attachment_path`].
pub fn detect_file_drop(
    user_input: &str,
    env_get: &dyn Fn(&str) -> Option<String>,
    home: &Path,
    cwd: &Path,
) -> Option<FileDrop> {
    let stripped = user_input.trim();
    if stripped.is_empty() {
        return None;
    }
    if !starts_like_path(stripped) {
        return None;
    }

    if let Some(direct) = resolve_attachment_path(stripped, env_get, home, cwd) {
        let is_image = is_image_suffix(&direct);
        return Some(FileDrop {
            path: direct,
            is_image,
            remainder: String::new(),
        });
    }

    let (first_token, mut remainder) = split_path_input(stripped);
    let mut drop_path = resolve_attachment_path(&first_token, env_get, home, cwd);

    let first_char = stripped.chars().next().unwrap();
    if drop_path.is_none()
        && stripped.contains(' ')
        && first_char != '\''
        && first_char != '"'
    {
        let byte_positions: Vec<usize> =
            stripped.match_indices(' ').map(|(i, _)| i).collect();
        for &pos in byte_positions.iter().rev() {
            let candidate = stripped[..pos].trim_end();
            if let Some(resolved) = resolve_attachment_path(candidate, env_get, home, cwd) {
                drop_path = Some(resolved);
                remainder = stripped[pos + 1..].trim().to_string();
                break;
            }
        }
    }

    let drop_path = drop_path?;
    let is_image = is_image_suffix(&drop_path);
    Some(FileDrop {
        path: drop_path,
        is_image,
        remainder,
    })
}

/// Collect local image attachments for single-query CLI flows. Mirrors
/// `_collect_query_images`. Returns `(message, deduped_image_paths)`. Returns an
/// `Err(String)` when an explicit `image_arg` is missing or not a supported
/// image (matching the `ValueError`s in the Python version).
pub fn collect_query_images(
    query: Option<&str>,
    image_arg: Option<&str>,
    env_get: &dyn Fn(&str) -> Option<String>,
    home: &Path,
    cwd: &Path,
) -> Result<(String, Vec<PathBuf>), String> {
    let mut message = query.unwrap_or("").to_string();
    let mut images: Vec<PathBuf> = Vec::new();

    if let Some(q) = query {
        if let Some(dropped) = detect_file_drop(q, env_get, home, cwd) {
            if dropped.is_image {
                let name = dropped
                    .path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                images.push(dropped.path.clone());
                message = if !dropped.remainder.is_empty() {
                    dropped.remainder
                } else {
                    format!("[User attached image: {}]", name)
                };
            }
        }
    }

    if let Some(arg) = image_arg {
        if !arg.is_empty() {
            let explicit = resolve_attachment_path(arg, env_get, home, cwd)
                .ok_or_else(|| format!("Image file not found: {}", arg))?;
            if !is_image_suffix(&explicit) {
                return Err(format!("Not a supported image file: {}", explicit.display()));
            }
            images.push(explicit);
        }
    }

    let mut deduped: Vec<PathBuf> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for img in images {
        let key = img.to_string_lossy().into_owned();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        deduped.push(img);
    }

    Ok((message, deduped))
}

/// Auto-attach clipboard images only for image-only paste gestures. Mirrors
/// `_should_auto_attach_clipboard_image_on_paste`.
pub fn should_auto_attach_clipboard_image_on_paste(pasted_text: &str) -> bool {
    pasted_text.trim().is_empty()
}

// ============================================================================
// Terminal leaked-response / paste stripping
// ============================================================================

static DSR_CPR_ESC_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\x1b\[\d+;\d+R").unwrap());
static DSR_CPR_VISIBLE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\^\[\[\d+;\d+R").unwrap());
static SGR_MOUSE_ESC_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\[<\d+;\d+;\d+[Mm]").unwrap());
static SGR_MOUSE_VISIBLE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\^\[\[<\d+;\d+;\d+[Mm]").unwrap());
static SGR_MOUSE_BARE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<\d+;\d+;\d+[Mm]").unwrap());

/// Strip leaked terminal control-response sequences. Returns
/// `(cleaned_text, had_mouse_reports)`. Mirrors
/// `_strip_leaked_terminal_responses_with_meta`.
pub fn strip_leaked_terminal_responses_with_meta(text: &str) -> (String, bool) {
    if text.is_empty() {
        return (text.to_string(), false);
    }

    let has_esc = text.contains("\x1b[");
    let has_visible = text.contains("^[");
    let has_bare_mouse =
        text.contains('<') && text.contains(';') && (text.contains('M') || text.contains('m'));
    if !(has_esc || has_visible || has_bare_mouse) {
        return (text.to_string(), false);
    }

    let mut t = text.to_string();
    let mut had_mouse_reports = false;

    if has_esc {
        t = DSR_CPR_ESC_RE.replace_all(&t, "").into_owned();
        let count = SGR_MOUSE_ESC_RE.find_iter(&t).count();
        t = SGR_MOUSE_ESC_RE.replace_all(&t, "").into_owned();
        had_mouse_reports = had_mouse_reports || count > 0;
    }

    if has_visible {
        t = DSR_CPR_VISIBLE_RE.replace_all(&t, "").into_owned();
        let count = SGR_MOUSE_VISIBLE_RE.find_iter(&t).count();
        t = SGR_MOUSE_VISIBLE_RE.replace_all(&t, "").into_owned();
        had_mouse_reports = had_mouse_reports || count > 0;
    }

    if has_bare_mouse {
        let count = SGR_MOUSE_BARE_RE.find_iter(&t).count();
        t = SGR_MOUSE_BARE_RE.replace_all(&t, "").into_owned();
        had_mouse_reports = had_mouse_reports || count > 0;
    }

    (t, had_mouse_reports)
}

/// Compatibility wrapper returning only the cleaned text. Mirrors
/// `_strip_leaked_terminal_responses`.
pub fn strip_leaked_terminal_responses(text: &str) -> String {
    strip_leaked_terminal_responses_with_meta(text).0
}

/// Strip leaked bracketed-paste wrapper markers. Mirrors
/// `_strip_leaked_bracketed_paste_wrappers`.
pub fn strip_leaked_bracketed_paste_wrappers(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }

    let mut t = text
        .replace("\x1b[200~", "")
        .replace("\x1b[201~", "")
        .replace("^[[200~", "")
        .replace("^[[201~", "");

    static RE_OPEN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(^|[\s\n>:\]\)])\[200~").unwrap());
    static RE_CLOSE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\[201~($|[\s\n<\[\(\):;.,!?])").unwrap());
    static RE_OPEN2: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(^|[\s\n>:\]\)])00~").unwrap());
    static RE_CLOSE2: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"01~($|[\s\n<\[\(\):;.,!?])").unwrap());

    t = RE_OPEN
        .replace_all(&t, |caps: &regex::Captures| caps[1].to_string())
        .into_owned();
    // The Python close patterns use a lookahead; emulate by re-emitting the
    // trailing boundary char captured here.
    t = RE_CLOSE
        .replace_all(&t, |caps: &regex::Captures| caps[1].to_string())
        .into_owned();
    t = RE_OPEN2
        .replace_all(&t, |caps: &regex::Captures| caps[1].to_string())
        .into_owned();
    t = RE_CLOSE2
        .replace_all(&t, |caps: &regex::Captures| caps[1].to_string())
        .into_owned();
    t
}

// ============================================================================
// Markdown stripping / Windows path preservation
// ============================================================================

static WINDOWS_PATH_DOT_RE: LazyLock<Regex> = LazyLock::new(|| {
    RegexBuilder::new(r"(?:\b[a-z]:\\|\\\\)[^\s`]*\\\.[^\s`]*")
        .case_insensitive(true)
        .build()
        .unwrap()
});

/// Keep Windows path separators before hidden directories in Markdown. Mirrors
/// `_preserve_windows_dot_segments_for_markdown`.
pub fn preserve_windows_dot_segments_for_markdown(text: &str) -> String {
    if !text.contains("\\.") {
        return text.to_string();
    }
    // Within each matched path-token, double a single backslash that precedes
    // a dot. Equivalent to the Python inner `re.sub(r"(?<!\\)\\(?=\.)", r"\\\\")`.
    WINDOWS_PATH_DOT_RE
        .replace_all(text, |caps: &regex::Captures| double_backslash_before_dot(&caps[0]))
        .into_owned()
}

/// Double a single (non-doubled) backslash that immediately precedes a `.`,
/// emulating `re.sub(r"(?<!\\)\\(?=\.)", r"\\\\", s)`.
fn double_backslash_before_dot(s: &str) -> String {
    let bytes: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i];
        if ch == '\\' {
            let prev_is_backslash = i > 0 && bytes[i - 1] == '\\';
            let next_is_dot = i + 1 < bytes.len() && bytes[i + 1] == '.';
            if !prev_is_backslash && next_is_dot {
                out.push('\\');
                out.push('\\');
                i += 1;
                continue;
            }
        }
        out.push(ch);
        i += 1;
    }
    out
}

/// Best-effort markdown marker removal for plain-text display. Mirrors
/// `_strip_markdown_syntax`. Operates on the already-plain text (callers should
/// strip ANSI first via [`strip_ansi_control`] if needed).
pub fn strip_markdown_syntax(text: &str) -> String {
    macro_rules! sub {
        ($plain:expr, $pat:expr, $rep:expr) => {{
            static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new($pat).unwrap());
            RE.replace_all($plain, $rep).into_owned()
        }};
    }

    let mut plain = text.to_string();

    // Multiline horizontal rules.
    {
        static RE: LazyLock<Regex> = LazyLock::new(|| {
            RegexBuilder::new(r"^\s{0,3}(?:[-*_]\s*){3,}$")
                .multi_line(true)
                .build()
                .unwrap()
        });
        plain = RE.replace_all(&plain, "").into_owned();
    }
    // Headings.
    {
        static RE: LazyLock<Regex> = LazyLock::new(|| {
            RegexBuilder::new(r"^\s{0,3}#{1,6}\s+")
                .multi_line(true)
                .build()
                .unwrap()
        });
        plain = RE.replace_all(&plain, "").into_owned();
    }

    plain = sub!(&plain, r"(```+|~~~+)", "");
    plain = sub!(&plain, r"`([^`]*)`", "$1");
    plain = sub!(&plain, r"!\[([^\]]*)\]\([^\)]*\)", "$1");
    plain = sub!(&plain, r"\[([^\]]+)\]\([^\)]*\)", "$1");
    plain = sub!(&plain, r"\*\*\*([^*]+)\*\*\*", "$1");
    // (?<!\w)___([^_]+)___(?!\w) — emulate word-boundary lookarounds.
    plain = strip_emphasis_bounded(&plain, "___", "___");
    plain = sub!(&plain, r"\*\*([^*]+)\*\*", "$1");
    plain = strip_emphasis_bounded(&plain, "__", "__");
    plain = sub!(&plain, r"\*([^*]+)\*", "$1");
    plain = strip_emphasis_bounded(&plain, "_", "_");
    plain = sub!(&plain, r"~~([^~]+)~~", "$1");
    plain = sub!(&plain, r"\n{3,}", "\n\n");

    plain.trim_matches('\n').to_string()
}

/// Emulate `(?<!\w)MARKER([^_]+)MARKER(?!\w)` underscore-emphasis removal.
fn strip_emphasis_bounded(text: &str, open: &str, close: &str) -> String {
    // Build a regex with explicit boundary capture in place of lookarounds.
    let pat = format!(
        r"(^|[^A-Za-z0-9_]){}([^_]+){}($|[^A-Za-z0-9_])",
        regex::escape(open),
        regex::escape(close)
    );
    let re = Regex::new(&pat).unwrap();
    // We need overlapping handling because adjacent matches can share a
    // boundary char; loop until stable.
    let mut current = text.to_string();
    loop {
        let next = re
            .replace_all(&current, |caps: &regex::Captures| {
                format!("{}{}{}", &caps[1], &caps[2], &caps[3])
            })
            .into_owned();
        if next == current {
            break;
        }
        current = next;
    }
    current
}

// ============================================================================
// ANSI color helpers
// ============================================================================

/// ANSI control-sequence stripping regex (matches `_ANSI_CONTROL_RE`).
static ANSI_CONTROL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))").unwrap()
});

/// Strip ANSI control sequences from text. Mirrors `_ANSI_CONTROL_RE.sub("")`.
pub fn strip_ansi_control(text: &str) -> String {
    ANSI_CONTROL_RE.replace_all(text, "").into_owned()
}

/// Convert a hex color like `#268bd2` to a true-color ANSI escape. Mirrors
/// `_hex_to_ansi`. On parse failure returns the bold/non-bold default.
pub fn hex_to_ansi(hex_color: &str, bold: bool) -> String {
    let parse = || -> Option<(u8, u8, u8)> {
        if hex_color.len() < 7 {
            return None;
        }
        let r = u8::from_str_radix(hex_color.get(1..3)?, 16).ok()?;
        let g = u8::from_str_radix(hex_color.get(3..5)?, 16).ok()?;
        let b = u8::from_str_radix(hex_color.get(5..7)?, 16).ok()?;
        Some((r, g, b))
    };
    match parse() {
        Some((r, g, b)) => {
            let prefix = if bold { "1;" } else { "" };
            format!("\x1b[{}38;2;{};{};{}m", prefix, r, g, b)
        }
        None => {
            if bold {
                ACCENT_ANSI_DEFAULT.to_string()
            } else {
                "\x1b[38;2;184;134;11m".to_string()
            }
        }
    }
}

// ============================================================================
// Image attachment badges
// ============================================================================

/// Format the attached-image badge row. Mirrors `_format_image_attachment_badges`.
/// `attached_image_names` are the file names (basenames) of attached images,
/// `image_counter` is the running counter, and `width` is the terminal width.
pub fn format_image_attachment_badges(
    attached_image_names: &[String],
    image_counter: i64,
    width: usize,
) -> String {
    if attached_image_names.is_empty() {
        return String::new();
    }

    fn trunc(name: &str, limit: usize) -> String {
        let chars: Vec<char> = name.chars().collect();
        if chars.len() <= limit {
            name.to_string()
        } else {
            let keep = limit.saturating_sub(3).max(1);
            let mut s: String = chars[..keep.min(chars.len())].iter().collect();
            s.push_str("...");
            s
        }
    }

    let n = attached_image_names.len();

    if width < 52 {
        if n == 1 {
            return format!("[📎 {}]", trunc(&attached_image_names[0], 20));
        }
        return format!("[📎 {} images attached]", n);
    }

    if width < 80 {
        if n == 1 {
            return format!("[📎 {}]", trunc(&attached_image_names[0], 32));
        }
        let first = trunc(&attached_image_names[0], 20);
        let extra = n - 1;
        return format!("[📎 {}] [+{}]", first, extra);
    }

    let base = image_counter - n as i64 + 1;
    (0..n)
        .map(|i| format!("[📎 Image #{}]", base + i as i64))
        .collect::<Vec<_>>()
        .join(" ")
}

// ============================================================================
// Process notification formatting
// ============================================================================

/// Format a process notification event into an `[IMPORTANT: ...]` message.
/// Mirrors `_format_process_notification`. Returns `None` only for unknown
/// shapes where the Python version would also produce nothing useful (it never
/// returns None in practice, but the signature allows it).
pub fn format_process_notification(evt: &Value) -> Option<String> {
    let obj = evt.as_object()?;
    let get_str = |k: &str, default: &str| -> String {
        obj.get(k)
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => value_to_string(other),
            })
            .filter(|s| !s.is_empty() || obj.contains_key(k))
            .unwrap_or_else(|| default.to_string())
    };

    let evt_type = obj
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("completion");
    let sid = get_str("session_id", "unknown");
    let cmd = get_str("command", "unknown");

    if evt_type == "watch_disabled" {
        let message = obj
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        return Some(format!("[IMPORTANT: {}]", message));
    }

    if evt_type == "watch_match" {
        let pat = obj.get("pattern").and_then(|v| v.as_str()).unwrap_or("?");
        let out = obj.get("output").and_then(|v| v.as_str()).unwrap_or("");
        let sup = obj
            .get("suppressed")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let mut text = format!(
            "[IMPORTANT: Background process {} matched watch pattern \"{}\".\nCommand: {}\nMatched output:\n{}",
            sid, pat, cmd, out
        );
        if sup != 0 {
            text.push_str(&format!(
                "\n({} earlier matches were suppressed by rate limit)",
                sup
            ));
        }
        text.push(']');
        return Some(text);
    }

    // Default: completion event.
    let exit = obj
        .get("exit_code")
        .map(value_to_string)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "?".to_string());
    let out = obj.get("output").and_then(|v| v.as_str()).unwrap_or("");
    Some(format!(
        "[IMPORTANT: Background process {} completed (exit code {}).\nCommand: {}\nOutput:\n{}]",
        sid, exit, cmd, out
    ))
}

// ============================================================================
// Output history ring buffer
// ============================================================================

struct OutputHistoryState {
    enabled: bool,
    replaying: bool,
    suppressed: bool,
    max_lines: usize,
    buffer: VecDeque<String>,
}

static OUTPUT_HISTORY: LazyLock<Mutex<OutputHistoryState>> = LazyLock::new(|| {
    Mutex::new(OutputHistoryState {
        enabled: true,
        replaying: false,
        suppressed: false,
        max_lines: 200,
        buffer: VecDeque::new(),
    })
});

/// Coerce a numeric output-history limit (min 10, default 200). Mirrors
/// `_coerce_output_history_limit`.
pub fn coerce_output_history_limit(value: Option<i64>) -> usize {
    match value {
        Some(v) => std::cmp::max(10, v) as usize,
        None => 200,
    }
}

/// Configure the recent-output history buffer. Mirrors `_configure_output_history`.
pub fn configure_output_history(enabled: bool, max_lines: Option<i64>) {
    let mut st = OUTPUT_HISTORY.lock().unwrap();
    st.enabled = enabled;
    st.max_lines = coerce_output_history_limit(max_lines);
    st.buffer = VecDeque::new();
}

/// Clear the output history. Mirrors `_clear_output_history`.
pub fn clear_output_history() {
    OUTPUT_HISTORY.lock().unwrap().buffer.clear();
}

/// Set the suppressed flag and return the previous value (for scoped restore).
/// Together with the returned guard, mirrors `_suspend_output_history`.
pub fn suspend_output_history() -> OutputHistoryGuard {
    let mut st = OUTPUT_HISTORY.lock().unwrap();
    let old = st.suppressed;
    st.suppressed = true;
    OutputHistoryGuard { old }
}

/// RAII guard that restores the previous suppression state on drop.
pub struct OutputHistoryGuard {
    old: bool,
}

impl Drop for OutputHistoryGuard {
    fn drop(&mut self) {
        OUTPUT_HISTORY.lock().unwrap().suppressed = self.old;
    }
}

/// Record cleaned text into the history buffer (split per line). Mirrors
/// `_record_output_history`.
pub fn record_output_history(text: &str) {
    let mut st = OUTPUT_HISTORY.lock().unwrap();
    if !st.enabled || st.replaying || st.suppressed {
        return;
    }
    let clean = strip_ansi_control(text)
        .replace('\r', "")
        .trim_end_matches('\n')
        .to_string();
    if clean.is_empty() {
        return;
    }
    let max = st.max_lines;
    for line in clean.split('\n') {
        if st.buffer.len() >= max {
            st.buffer.pop_front();
        }
        st.buffer.push_back(line.to_string());
    }
}

/// Snapshot the current output history lines. Used by replay logic.
pub fn output_history_snapshot() -> Vec<String> {
    OUTPUT_HISTORY.lock().unwrap().buffer.iter().cloned().collect()
}

// ============================================================================
// Compact banner
// ============================================================================

/// Build the compact welcome banner that fits the current terminal width.
/// Mirrors `_build_compact_banner`. Colors/branding are resolved from the
/// supplied skin parameters (the Python version reads these from the skin
/// engine). Output uses Rich-markup tags exactly as the Python original emits
/// them so downstream rendering is identical.
pub fn build_compact_banner(
    skin_name: &str,
    border_color: &str,
    title_color: &str,
    dim_color: &str,
    agent_name: &str,
    version_line: &str,
    terminal_columns: usize,
) -> String {
    let (line1, tiny_line) = if skin_name == "default" {
        (
            "⚕ NOUS HERMES - AI Agent Framework".to_string(),
            "⚕ NOUS HERMES".to_string(),
        )
    } else {
        (
            format!("{} - AI Agent Framework", agent_name),
            agent_name.to_string(),
        )
    };

    let w = std::cmp::min(terminal_columns.saturating_sub(2), 88);
    if w < 30 {
        return format!(
            "\n[{}]{}[/] [dim {}]- Nous Research[/]\n",
            title_color, tiny_line, dim_color
        );
    }

    let inner = w.saturating_sub(2); // inside the box border
    let content_width = inner.saturating_sub(2);
    let bar = "═".repeat(w);

    let line1 = pad_truncate(&line1, content_width);
    let line2 = pad_truncate(version_line, content_width);

    format!(
        "\n[bold {0}]╔{1}╗[/]\n[bold {0}]║[/] [{2}]{3}[/] [bold {0}]║[/]\n[bold {0}]║[/] [dim {4}]{5}[/] [bold {0}]║[/]\n[bold {0}]╚{1}╝[/]\n",
        border_color, bar, title_color, line1, dim_color, line2
    )
}

/// Truncate a string to `width` chars (Python slicing) then left-justify (pad
/// with spaces) to `width`. Mirrors `s[:width].ljust(width)`.
fn pad_truncate(s: &str, width: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    let truncated: String = chars.iter().take(width).collect();
    let len = truncated.chars().count();
    if len >= width {
        truncated
    } else {
        let mut out = truncated;
        out.push_str(&" ".repeat(width - len));
        out
    }
}

// ============================================================================
// Prefill messages loading
// ============================================================================

/// Load ephemeral prefill messages from a JSON file. Mirrors
/// `_load_prefill_messages`. Returns an empty list if the path is empty, the
/// file is missing, or the JSON is not an array.
pub fn load_prefill_messages(file_path: &str, home: &Path) -> Vec<Value> {
    if file_path.is_empty() {
        return Vec::new();
    }
    let mut path = if let Some(rest) = file_path.strip_prefix("~/") {
        home.join(rest)
    } else if file_path == "~" {
        home.to_path_buf()
    } else {
        PathBuf::from(file_path)
    };
    if !path.is_absolute() {
        path = home.join(&path);
    }
    if !path.exists() {
        return Vec::new();
    }
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    match serde_json::from_str::<Value>(&raw) {
        Ok(Value::Array(arr)) => arr,
        _ => Vec::new(),
    }
}

// ============================================================================
// Git worktree helpers (pure-ish logic)
// ============================================================================

/// Return whether a resolved path stays within the expected root. Mirrors
/// `_path_is_within_root`.
pub fn path_is_within_root(path: &Path, root: &Path) -> bool {
    path.strip_prefix(root).is_ok()
}

/// Filter orphaned auto-generated branches (those not in `active_branches` and
/// matching `hermes/hermes-*` or `pr-*`). Mirrors the orphan-selection logic in
/// `_prune_orphaned_branches`.
pub fn select_orphaned_branches(
    all_branches: &[String],
    active_branches: &std::collections::HashSet<String>,
) -> Vec<String> {
    all_branches
        .iter()
        .filter(|b| {
            !active_branches.contains(*b)
                && (b.starts_with("hermes/hermes-") || b.starts_with("pr-"))
        })
        .cloned()
        .collect()
}

/// Build the per-session worktree name and branch name from a short id.
/// Mirrors the naming in `_setup_worktree`.
pub fn worktree_names(short_id: &str) -> (String, String) {
    let wt_name = format!("hermes-{}", short_id);
    let branch_name = format!("hermes/{}", wt_name);
    (wt_name, branch_name)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_strip_reasoning_closed_pair() {
        // The closed-pair regex consumes `<think>…</think>\s*` but not the
        // space preceding the tag, matching the Python original.
        let s = "before <think>secret reasoning</think>after";
        assert_eq!(strip_reasoning_tags(s), "before after");
        let s2 = "before<think>secret</think>after";
        assert_eq!(strip_reasoning_tags(s2), "beforeafter");
    }

    #[test]
    fn test_strip_reasoning_case_insensitive() {
        let s = "x<THINK>hidden</THINK>y";
        assert_eq!(strip_reasoning_tags(s), "xy");
    }

    #[test]
    fn test_strip_reasoning_unterminated() {
        let s = "answer<thinking>runs to the end of the text";
        assert_eq!(strip_reasoning_tags(s), "answer");
    }

    #[test]
    fn test_strip_reasoning_orphan_close() {
        let s = "stuff</think>answer";
        assert_eq!(strip_reasoning_tags(s), "stuffanswer");
    }

    #[test]
    fn test_strip_tool_call_block() {
        let s = "hi <tool_call>{\"x\":1}</tool_call> bye";
        assert_eq!(strip_reasoning_tags(s), "hi bye");
    }

    #[test]
    fn test_assistant_content_as_text_string() {
        assert_eq!(assistant_content_as_text(&json!("hello")), "hello");
    }

    #[test]
    fn test_assistant_content_as_text_list() {
        let c = json!([
            {"type": "text", "text": "a"},
            {"type": "image", "text": "ignored"},
            {"type": "text", "text": "b"}
        ]);
        assert_eq!(assistant_content_as_text(&c), "a\nb");
    }

    #[test]
    fn test_assistant_content_as_text_null() {
        assert_eq!(assistant_content_as_text(&Value::Null), "");
    }

    #[test]
    fn test_parse_service_tier() {
        assert_eq!(parse_service_tier_config(""), None);
        assert_eq!(parse_service_tier_config("normal"), None);
        assert_eq!(parse_service_tier_config("off"), None);
        assert_eq!(parse_service_tier_config("fast"), Some("priority".into()));
        assert_eq!(parse_service_tier_config("PRIORITY"), Some("priority".into()));
        assert_eq!(parse_service_tier_config("on"), Some("priority".into()));
        assert_eq!(parse_service_tier_config("weird"), None);
    }

    #[test]
    fn test_looks_like_slash_command() {
        assert!(looks_like_slash_command("/help"));
        assert!(looks_like_slash_command("/model gpt-4"));
        assert!(looks_like_slash_command("/q"));
        assert!(!looks_like_slash_command("/Users/foo/bar.md fix this"));
        assert!(!looks_like_slash_command("hello"));
        assert!(!looks_like_slash_command(""));
    }

    #[test]
    fn test_parse_skills_argument() {
        let got = parse_skills_argument(&["a,b , c".to_string(), "b,d".to_string()]);
        assert_eq!(got, vec!["a", "b", "c", "d"]);
        assert!(parse_skills_argument(&[]).is_empty());
        assert!(parse_skills_argument(&["".to_string()]).is_empty());
    }

    #[test]
    fn test_split_path_input_plain() {
        let (t, r) = split_path_input("/tmp/pic.png describe this");
        assert_eq!(t, "/tmp/pic.png");
        assert_eq!(r, "describe this");
    }

    #[test]
    fn test_split_path_input_escaped_space() {
        let (t, r) = split_path_input(r"~/My\ Photos/cat.png what is this?");
        assert_eq!(t, "~/My Photos/cat.png");
        assert_eq!(r, "what is this?");
    }

    #[test]
    fn test_split_path_input_quoted() {
        let (t, r) = split_path_input("\"/storage/cat 1.png\" summarize");
        assert_eq!(t, "/storage/cat 1.png");
        assert_eq!(r, "summarize");
    }

    #[test]
    fn test_split_path_input_empty() {
        let (t, r) = split_path_input("   ");
        assert_eq!(t, "");
        assert_eq!(r, "");
    }

    #[test]
    fn test_is_image_suffix() {
        assert!(is_image_suffix(Path::new("/x/a.PNG")));
        assert!(is_image_suffix(Path::new("a.jpeg")));
        assert!(!is_image_suffix(Path::new("a.txt")));
        assert!(!is_image_suffix(Path::new("noext")));
    }

    #[test]
    fn test_starts_like_path() {
        assert!(starts_like_path("/etc/hosts"));
        assert!(starts_like_path("~/file"));
        assert!(starts_like_path("./rel"));
        assert!(starts_like_path("../rel"));
        assert!(starts_like_path("file:///x"));
        assert!(starts_like_path(r"C:\Users\x"));
        assert!(starts_like_path("C:/Users/x"));
        assert!(starts_like_path("\"/quoted"));
        assert!(!starts_like_path("hello world"));
        assert!(!starts_like_path("model gpt"));
    }

    #[test]
    fn test_strip_leaked_terminal_responses() {
        let (cleaned, mouse) = strip_leaked_terminal_responses_with_meta("ab\x1b[12;5Rcd");
        assert_eq!(cleaned, "abcd");
        assert!(!mouse);

        let (cleaned, mouse) =
            strip_leaked_terminal_responses_with_meta("x\x1b[<0;10;20Mtext");
        assert_eq!(cleaned, "xtext");
        assert!(mouse);

        let (cleaned, _) = strip_leaked_terminal_responses_with_meta("plain text");
        assert_eq!(cleaned, "plain text");
    }

    #[test]
    fn test_strip_bracketed_paste() {
        assert_eq!(
            strip_leaked_bracketed_paste_wrappers("\x1b[200~hello\x1b[201~"),
            "hello"
        );
    }

    #[test]
    fn test_hex_to_ansi() {
        assert_eq!(hex_to_ansi("#268bd2", false), "\x1b[38;2;38;139;210m");
        assert_eq!(hex_to_ansi("#268bd2", true), "\x1b[1;38;2;38;139;210m");
        // Invalid → default.
        assert_eq!(hex_to_ansi("nope", true), ACCENT_ANSI_DEFAULT);
        assert_eq!(hex_to_ansi("nope", false), "\x1b[38;2;184;134;11m");
    }

    #[test]
    fn test_strip_ansi_control() {
        assert_eq!(strip_ansi_control("\x1b[31mred\x1b[0m"), "red");
    }

    #[test]
    fn test_strip_markdown_syntax() {
        assert_eq!(strip_markdown_syntax("# Heading"), "Heading");
        assert_eq!(strip_markdown_syntax("**bold**"), "bold");
        assert_eq!(strip_markdown_syntax("`code`"), "code");
        assert_eq!(strip_markdown_syntax("*italic*"), "italic");
        assert_eq!(strip_markdown_syntax("[link](http://x)"), "link");
        assert_eq!(strip_markdown_syntax("~~strike~~"), "strike");
    }

    #[test]
    fn test_preserve_windows_dot_segments() {
        let r = preserve_windows_dot_segments_for_markdown(r"path D:\repo\.ai end");
        assert!(r.contains(r"D:\repo\\.ai"));
        // No backslash-dot → unchanged.
        assert_eq!(
            preserve_windows_dot_segments_for_markdown("normal text"),
            "normal text"
        );
    }

    #[test]
    fn test_format_image_badges_narrow() {
        let names = vec!["cat.png".to_string()];
        assert_eq!(
            format_image_attachment_badges(&names, 1, 40),
            "[📎 cat.png]"
        );
        let many = vec!["a.png".to_string(), "b.png".to_string()];
        assert_eq!(
            format_image_attachment_badges(&many, 2, 40),
            "[📎 2 images attached]"
        );
    }

    #[test]
    fn test_format_image_badges_wide() {
        let names = vec!["a.png".to_string(), "b.png".to_string()];
        assert_eq!(
            format_image_attachment_badges(&names, 5, 120),
            "[📎 Image #4] [📎 Image #5]"
        );
    }

    #[test]
    fn test_format_process_notification_completion() {
        let evt = json!({
            "session_id": "S1",
            "command": "ls",
            "exit_code": 0,
            "output": "files"
        });
        let got = format_process_notification(&evt).unwrap();
        assert!(got.starts_with("[IMPORTANT: Background process S1 completed (exit code 0)."));
        assert!(got.contains("Command: ls"));
        assert!(got.ends_with("files]"));
    }

    #[test]
    fn test_format_process_notification_watch_match() {
        let evt = json!({
            "type": "watch_match",
            "session_id": "S2",
            "command": "tail -f log",
            "pattern": "ERROR",
            "output": "ERROR boom",
            "suppressed": 3
        });
        let got = format_process_notification(&evt).unwrap();
        assert!(got.contains("matched watch pattern \"ERROR\""));
        assert!(got.contains("3 earlier matches were suppressed"));
    }

    #[test]
    fn test_resolve_personality_prompt_string() {
        assert_eq!(resolve_personality_prompt(&json!("be nice")), "be nice");
    }

    #[test]
    fn test_resolve_personality_prompt_dict() {
        let v = json!({"system_prompt": "base", "tone": "warm", "style": "terse"});
        assert_eq!(
            resolve_personality_prompt(&v),
            "base\nTone: warm\nStyle: terse"
        );
        let v2 = json!({"system_prompt": "base"});
        assert_eq!(resolve_personality_prompt(&v2), "base");
    }

    #[test]
    fn test_default_cli_config_shape() {
        let cfg = default_cli_config();
        assert_eq!(cfg["model"]["provider"], json!("auto"));
        assert_eq!(cfg["agent"]["max_turns"], json!(90));
        assert_eq!(cfg["compression"]["threshold"], json!(0.50));
    }

    #[test]
    fn test_load_cli_config_string_model() {
        let user = json!({"model": "claude-3"});
        let cfg = load_cli_config(Some(&user), None, false);
        assert_eq!(cfg["model"]["default"], json!("claude-3"));
        // provider default preserved.
        assert_eq!(cfg["model"]["provider"], json!("auto"));
    }

    #[test]
    fn test_load_cli_config_dict_model_promotes() {
        let user = json!({"model": {"model": "gpt-x"}});
        let cfg = load_cli_config(Some(&user), None, false);
        assert_eq!(cfg["model"]["default"], json!("gpt-x"));
    }

    #[test]
    fn test_load_cli_config_legacy_root_provider() {
        // The legacy root-level provider/base_url fallback only fires when
        // model.provider/base_url is falsy. The default provider is "auto"
        // (truthy), so root provider does NOT override it — matching Python's
        // `if not defaults["model"].get("provider")` guard. base_url default
        // is "" (falsy), so the root base_url fallback DOES apply.
        let user = json!({"provider": "openrouter", "base_url": "http://x"});
        let cfg = load_cli_config(Some(&user), None, false);
        assert_eq!(cfg["model"]["provider"], json!("auto"));
        assert_eq!(cfg["model"]["base_url"], json!("http://x"));

        // When model.provider is explicitly empty, the root fallback applies.
        let user2 = json!({"model": {"provider": ""}, "provider": "openrouter"});
        let cfg2 = load_cli_config(Some(&user2), None, false);
        assert_eq!(cfg2["model"]["provider"], json!("openrouter"));
    }

    #[test]
    fn test_load_cli_config_legacy_max_turns() {
        let user = json!({"max_turns": 12});
        let cfg = load_cli_config(Some(&user), None, false);
        assert_eq!(cfg["agent"]["max_turns"], json!(12));
    }

    #[test]
    fn test_load_cli_config_ignore_user() {
        let user = json!({"model": "u"});
        let project = json!({"model": "p"});
        let cfg = load_cli_config(Some(&user), Some(&project), true);
        assert_eq!(cfg["model"]["default"], json!("p"));
    }

    #[test]
    fn test_load_cli_config_carry_unknown_keys() {
        let user = json!({"honcho": {"enabled": true}});
        let cfg = load_cli_config(Some(&user), None, false);
        assert_eq!(cfg["honcho"]["enabled"], json!(true));
    }

    #[test]
    fn test_config_env_bridge_local_cwd() {
        let cfg = json!({"terminal": {"env_type": "local", "cwd": "."}});
        let no_env = |_: &str| false;
        let bridge = config_env_bridge(&cfg, "/work/dir", false, true, &no_env);
        let cwd = bridge.iter().find(|(k, _)| k == "TERMINAL_CWD");
        assert_eq!(cwd, Some(&("TERMINAL_CWD".to_string(), "/work/dir".to_string())));
    }

    #[test]
    fn test_config_env_bridge_gateway_skips_cwd() {
        let cfg = json!({"terminal": {"env_type": "local"}});
        let no_env = |_: &str| false;
        let bridge = config_env_bridge(&cfg, "/work", true, true, &no_env);
        assert!(bridge.iter().all(|(k, _)| k != "TERMINAL_CWD"));
    }

    #[test]
    fn test_config_env_bridge_auxiliary() {
        let cfg = json!({
            "auxiliary": {"vision": {"provider": "openai", "model": "gpt-4o", "base_url": "", "api_key": ""}}
        });
        let no_env = |_: &str| false;
        let bridge = config_env_bridge(&cfg, "/x", false, false, &no_env);
        assert!(bridge.contains(&("AUXILIARY_VISION_PROVIDER".to_string(), "openai".to_string())));
        assert!(bridge.contains(&("AUXILIARY_VISION_MODEL".to_string(), "gpt-4o".to_string())));
        // empty base_url/api_key not bridged
        assert!(bridge.iter().all(|(k, _)| k != "AUXILIARY_VISION_BASE_URL"));
    }

    #[test]
    fn test_config_env_bridge_auto_provider_skipped() {
        let cfg = json!({
            "auxiliary": {"vision": {"provider": "auto", "model": "", "base_url": "", "api_key": ""}}
        });
        let no_env = |_: &str| false;
        let bridge = config_env_bridge(&cfg, "/x", false, false, &no_env);
        assert!(bridge.iter().all(|(k, _)| k != "AUXILIARY_VISION_PROVIDER"));
    }

    #[test]
    fn test_save_and_reload_config_value() {
        let dir = std::env::temp_dir().join(format!("hermes_cli_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.yaml");
        let _ = std::fs::remove_file(&path);

        save_config_value(&path, "agent.system_prompt", json!("hi there")).unwrap();
        save_config_value(&path, "model.default", json!("gpt-x")).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_yaml::from_str(&raw).unwrap();
        assert_eq!(parsed["agent"]["system_prompt"], json!("hi there"));
        assert_eq!(parsed["model"]["default"], json!("gpt-x"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_output_history() {
        configure_output_history(true, Some(5));
        clear_output_history();
        record_output_history("\x1b[31mline one\x1b[0m\nline two\n");
        let snap = output_history_snapshot();
        assert_eq!(snap, vec!["line one".to_string(), "line two".to_string()]);

        {
            let _g = suspend_output_history();
            record_output_history("suppressed");
        }
        // After guard drop, suppression restored to false; suppressed line not added.
        let snap2 = output_history_snapshot();
        assert_eq!(snap2.len(), 2);
    }

    #[test]
    fn test_coerce_output_history_limit() {
        assert_eq!(coerce_output_history_limit(Some(3)), 10);
        assert_eq!(coerce_output_history_limit(Some(50)), 50);
        assert_eq!(coerce_output_history_limit(None), 200);
    }

    #[test]
    fn test_build_compact_banner_default() {
        let b = build_compact_banner(
            "default", "#FFD700", "#FFBF00", "#B8860B", "Hermes", "v1.0", 100,
        );
        assert!(b.contains("⚕ NOUS HERMES - AI Agent Framework"));
        assert!(b.contains("╔"));
    }

    #[test]
    fn test_build_compact_banner_tiny() {
        let b = build_compact_banner(
            "default", "#FFD700", "#FFBF00", "#B8860B", "Hermes", "v1.0", 20,
        );
        assert!(b.contains("⚕ NOUS HERMES"));
        assert!(b.contains("Nous Research"));
        assert!(!b.contains("╔"));
    }

    #[test]
    fn test_select_orphaned_branches() {
        let all = vec![
            "main".to_string(),
            "hermes/hermes-abc".to_string(),
            "pr-42".to_string(),
            "feature/x".to_string(),
        ];
        let mut active = std::collections::HashSet::new();
        active.insert("main".to_string());
        let orphaned = select_orphaned_branches(&all, &active);
        assert_eq!(orphaned, vec!["hermes/hermes-abc", "pr-42"]);
    }

    #[test]
    fn test_worktree_names() {
        let (wt, br) = worktree_names("deadbeef");
        assert_eq!(wt, "hermes-deadbeef");
        assert_eq!(br, "hermes/hermes-deadbeef");
    }

    #[test]
    fn test_load_prefill_messages_missing() {
        let home = std::env::temp_dir();
        assert!(load_prefill_messages("", &home).is_empty());
        assert!(load_prefill_messages("does-not-exist-xyz.json", &home).is_empty());
    }

    #[test]
    fn test_should_auto_attach_clipboard() {
        assert!(should_auto_attach_clipboard_image_on_paste("   "));
        assert!(!should_auto_attach_clipboard_image_on_paste("hello"));
    }

    #[test]
    fn test_resolve_attachment_path_existing() {
        let dir = std::env::temp_dir().join(format!("hermes_attach_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("img.png");
        std::fs::write(&file, b"x").unwrap();

        let env_get = |_: &str| None;
        let home = std::env::temp_dir();
        let got = resolve_attachment_path(
            file.to_str().unwrap(),
            &env_get,
            &home,
            &dir,
        );
        assert!(got.is_some());

        let missing = resolve_attachment_path("/no/such/file.png", &env_get, &home, &dir);
        assert!(missing.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_detect_file_drop_image() {
        let dir = std::env::temp_dir().join(format!("hermes_drop_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("cat.png");
        std::fs::write(&file, b"x").unwrap();

        let env_get = |_: &str| None;
        let home = std::env::temp_dir();
        let input = format!("{} describe this", file.to_str().unwrap());
        let drop = detect_file_drop(&input, &env_get, &home, &dir).unwrap();
        assert!(drop.is_image);
        assert_eq!(drop.remainder, "describe this");

        assert!(detect_file_drop("hello there", &env_get, &home, &dir).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_collect_query_images_explicit_missing() {
        let env_get = |_: &str| None;
        let home = std::env::temp_dir();
        let cwd = std::env::temp_dir();
        let res = collect_query_images(Some("hi"), Some("/no/such.png"), &env_get, &home, &cwd);
        assert!(res.is_err());
    }

    #[test]
    fn test_path_is_within_root() {
        assert!(path_is_within_root(Path::new("/a/b/c"), Path::new("/a/b")));
        assert!(!path_is_within_root(Path::new("/x/y"), Path::new("/a/b")));
    }

    #[test]
    fn test_env_var_expansion() {
        let env_get = |k: &str| if k == "FOO" { Some("bar".to_string()) } else { None };
        assert_eq!(expand_env_vars_in_str("$FOO/baz", &env_get), "bar/baz");
        assert_eq!(expand_env_vars_in_str("${FOO}x", &env_get), "barx");
        assert_eq!(expand_env_vars_in_str("$MISSING", &env_get), "$MISSING");
    }
}

/// Re-export of the spinner frame helper so other modules can render the same
/// busy indicator the CLI uses. Mirrors `_command_spinner_frame` selection by
/// time (caller supplies an index, typically `elapsed*10 % len`).
pub fn command_spinner_frame(index: usize) -> &'static str {
    COMMAND_SPINNER_FRAMES[index % COMMAND_SPINNER_FRAMES.len()]
}

/// A small typed view over a loaded CLI config for convenient access by other
/// modules. Holds the merged JSON tree.
#[derive(Debug, Clone)]
pub struct CliConfig {
    pub root: Value,
}

impl CliConfig {
    pub fn new(root: Value) -> Self {
        Self { root }
    }

    /// Dot-path getter (e.g. `agent.max_turns`).
    pub fn get(&self, dotted: &str) -> Option<&Value> {
        let mut current = &self.root;
        for key in dotted.split('.') {
            current = current.get(key)?;
        }
        Some(current)
    }

    /// Convenience: model default string.
    pub fn model_default(&self) -> String {
        self.get("model.default")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    }

    /// Convenience: agent max_turns.
    pub fn max_turns(&self) -> i64 {
        self.get("agent.max_turns").and_then(|v| v.as_i64()).unwrap_or(90)
    }

    /// Returns the personalities map as a plain `HashMap<String, Value>`.
    pub fn personalities(&self) -> HashMap<String, Value> {
        self.get("agent.personalities")
            .and_then(|v| v.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    }
}
