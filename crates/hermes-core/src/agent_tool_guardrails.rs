//! Pure tool-call loop guardrail primitives.
//!
//! The controller in this module is intentionally side-effect free: it tracks
//! per-turn tool-call observations and returns decisions. Runtime code owns
//! whether those decisions become warning guidance, synthetic tool results, or
//! controlled turn halts.
//!
//! Native Rust port of `agent/tool_guardrails.py`.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value as Json;
use sha2::{Digest, Sha256};

/// Tools whose output is read-only / idempotent for a fixed set of arguments.
pub const IDEMPOTENT_TOOL_NAMES: &[&str] = &[
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
    "session_search",
    "browser_snapshot",
    "browser_console",
    "browser_get_images",
    "mcp_filesystem_read_file",
    "mcp_filesystem_read_text_file",
    "mcp_filesystem_read_multiple_files",
    "mcp_filesystem_list_directory",
    "mcp_filesystem_list_directory_with_sizes",
    "mcp_filesystem_directory_tree",
    "mcp_filesystem_get_file_info",
    "mcp_filesystem_search_files",
];

/// Tools that mutate state; never treated as idempotent.
pub const MUTATING_TOOL_NAMES: &[&str] = &[
    "terminal",
    "execute_code",
    "write_file",
    "patch",
    "todo",
    "memory",
    "skill_manage",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_scroll",
    "browser_navigate",
    "send_message",
    "cronjob",
    "delegate_task",
    "process",
];

/// Thresholds for per-turn tool-call loop detection.
///
/// Warnings are enabled by default and never prevent tool execution. Hard stops
/// are explicit opt-in so interactive CLI/TUI sessions get a gentle nudge unless
/// the user enables circuit-breaker behavior in `config.yaml`.
#[derive(Debug, Clone)]
pub struct ToolCallGuardrailConfig {
    pub warnings_enabled: bool,
    pub hard_stop_enabled: bool,
    pub exact_failure_warn_after: i64,
    pub exact_failure_block_after: i64,
    pub same_tool_failure_warn_after: i64,
    pub same_tool_failure_halt_after: i64,
    pub no_progress_warn_after: i64,
    pub no_progress_block_after: i64,
    pub idempotent_tools: Vec<String>,
    pub mutating_tools: Vec<String>,
}

impl Default for ToolCallGuardrailConfig {
    fn default() -> Self {
        ToolCallGuardrailConfig {
            warnings_enabled: true,
            hard_stop_enabled: false,
            exact_failure_warn_after: 2,
            exact_failure_block_after: 5,
            same_tool_failure_warn_after: 3,
            same_tool_failure_halt_after: 8,
            no_progress_warn_after: 2,
            no_progress_block_after: 5,
            idempotent_tools: IDEMPOTENT_TOOL_NAMES.iter().map(|s| s.to_string()).collect(),
            mutating_tools: MUTATING_TOOL_NAMES.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl ToolCallGuardrailConfig {
    /// Build config from the `tool_loop_guardrails` config.yaml section.
    pub fn from_mapping(data: Option<&serde_yaml::Value>) -> Self {
        let defaults = Self::default();
        let map = match data {
            Some(serde_yaml::Value::Mapping(m)) => m,
            _ => return defaults,
        };

        let get = |key: &str| map.get(serde_yaml::Value::String(key.to_string()));

        let warn_after = match get("warn_after") {
            Some(serde_yaml::Value::Mapping(m)) => Some(m.clone()),
            _ => None,
        };
        let hard_stop_after = match get("hard_stop_after") {
            Some(serde_yaml::Value::Mapping(m)) => Some(m.clone()),
            _ => None,
        };

        let nested = |m: &Option<serde_yaml::Mapping>, key: &str| -> Option<serde_yaml::Value> {
            m.as_ref()
                .and_then(|mm| mm.get(serde_yaml::Value::String(key.to_string())).cloned())
        };

        // For each tunable: prefer nested warn_after/hard_stop_after value, else
        // fall back to flat key on the top-level mapping.
        let pick = |nested_val: Option<serde_yaml::Value>, flat_key: &str| -> Option<serde_yaml::Value> {
            if nested_val.is_some() {
                nested_val
            } else {
                get(flat_key).cloned()
            }
        };

        ToolCallGuardrailConfig {
            warnings_enabled: as_bool(get("warnings_enabled"), defaults.warnings_enabled),
            hard_stop_enabled: as_bool(get("hard_stop_enabled"), defaults.hard_stop_enabled),
            exact_failure_warn_after: positive_int(
                pick(nested(&warn_after, "exact_failure"), "exact_failure_warn_after").as_ref(),
                defaults.exact_failure_warn_after,
            ),
            same_tool_failure_warn_after: positive_int(
                pick(
                    nested(&warn_after, "same_tool_failure"),
                    "same_tool_failure_warn_after",
                )
                .as_ref(),
                defaults.same_tool_failure_warn_after,
            ),
            no_progress_warn_after: positive_int(
                pick(
                    nested(&warn_after, "idempotent_no_progress"),
                    "no_progress_warn_after",
                )
                .as_ref(),
                defaults.no_progress_warn_after,
            ),
            exact_failure_block_after: positive_int(
                pick(
                    nested(&hard_stop_after, "exact_failure"),
                    "exact_failure_block_after",
                )
                .as_ref(),
                defaults.exact_failure_block_after,
            ),
            same_tool_failure_halt_after: positive_int(
                pick(
                    nested(&hard_stop_after, "same_tool_failure"),
                    "same_tool_failure_halt_after",
                )
                .as_ref(),
                defaults.same_tool_failure_halt_after,
            ),
            no_progress_block_after: positive_int(
                pick(
                    nested(&hard_stop_after, "idempotent_no_progress"),
                    "no_progress_block_after",
                )
                .as_ref(),
                defaults.no_progress_block_after,
            ),
            idempotent_tools: defaults.idempotent_tools,
            mutating_tools: defaults.mutating_tools,
        }
    }
}

/// Stable, non-reversible identity for a tool name plus canonical args.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolCallSignature {
    pub tool_name: String,
    pub args_hash: String,
}

impl ToolCallSignature {
    pub fn from_call(tool_name: &str, args: &Json) -> Self {
        let canonical = canonical_tool_args(args);
        ToolCallSignature {
            tool_name: tool_name.to_string(),
            args_hash: sha256_hex(&canonical),
        }
    }

    /// Return public metadata without raw argument values.
    pub fn to_metadata(&self) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("tool_name".to_string(), self.tool_name.clone());
        m.insert("args_hash".to_string(), self.args_hash.clone());
        m
    }
}

/// Action returned by the guardrail controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailAction {
    Allow,
    Warn,
    Block,
    Halt,
}

impl GuardrailAction {
    pub fn as_str(self) -> &'static str {
        match self {
            GuardrailAction::Allow => "allow",
            GuardrailAction::Warn => "warn",
            GuardrailAction::Block => "block",
            GuardrailAction::Halt => "halt",
        }
    }
}

/// Decision returned by the tool-call guardrail controller.
#[derive(Debug, Clone)]
pub struct ToolGuardrailDecision {
    pub action: GuardrailAction,
    pub code: String,
    pub message: String,
    pub tool_name: String,
    pub count: i64,
    pub signature: Option<ToolCallSignature>,
}

impl Default for ToolGuardrailDecision {
    fn default() -> Self {
        ToolGuardrailDecision {
            action: GuardrailAction::Allow,
            code: "allow".to_string(),
            message: String::new(),
            tool_name: String::new(),
            count: 0,
            signature: None,
        }
    }
}

impl ToolGuardrailDecision {
    /// A plain "allow" decision carrying a tool name / signature.
    fn allow_with(tool_name: &str, count: i64, signature: ToolCallSignature) -> Self {
        ToolGuardrailDecision {
            tool_name: tool_name.to_string(),
            count,
            signature: Some(signature),
            ..Default::default()
        }
    }

    pub fn allows_execution(&self) -> bool {
        matches!(self.action, GuardrailAction::Allow | GuardrailAction::Warn)
    }

    pub fn should_halt(&self) -> bool {
        matches!(self.action, GuardrailAction::Block | GuardrailAction::Halt)
    }

    pub fn to_metadata(&self) -> Json {
        let mut obj = serde_json::Map::new();
        obj.insert("action".to_string(), Json::String(self.action.as_str().to_string()));
        obj.insert("code".to_string(), Json::String(self.code.clone()));
        obj.insert("message".to_string(), Json::String(self.message.clone()));
        obj.insert("tool_name".to_string(), Json::String(self.tool_name.clone()));
        obj.insert("count".to_string(), Json::Number(self.count.into()));
        if let Some(sig) = &self.signature {
            let mut sm = serde_json::Map::new();
            sm.insert("tool_name".to_string(), Json::String(sig.tool_name.clone()));
            sm.insert("args_hash".to_string(), Json::String(sig.args_hash.clone()));
            obj.insert("signature".to_string(), Json::Object(sm));
        }
        Json::Object(obj)
    }
}

/// Return sorted compact JSON for parsed tool arguments.
///
/// Mirrors Python `json.dumps(args, ensure_ascii=False, sort_keys=True,
/// separators=(",", ":"), default=str)`. serde_json serializes compactly by
/// default; we recursively sort object keys to emulate `sort_keys=True`.
pub fn canonical_tool_args(args: &Json) -> String {
    let sorted = sort_json_keys(args);
    serde_json::to_string(&sorted).unwrap_or_else(|_| "{}".to_string())
}

/// Recursively reorder object keys lexicographically.
fn sort_json_keys(value: &Json) -> Json {
    match value {
        Json::Object(map) => {
            let mut sorted: BTreeMap<String, Json> = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), sort_json_keys(v));
            }
            let mut out = serde_json::Map::new();
            for (k, v) in sorted {
                out.insert(k, v);
            }
            Json::Object(out)
        }
        Json::Array(items) => Json::Array(items.iter().map(sort_json_keys).collect()),
        other => other.clone(),
    }
}

/// Safety-fallback classifier used only when callers don't pass `failed`.
///
/// Mirrors `agent.display._detect_tool_failure` so the guardrail never
/// disagrees with the CLI's user-visible `[error]` tag.
///
/// Returns `(failed, tag_suffix)`.
pub fn classify_tool_failure(tool_name: &str, result: Option<&str>) -> (bool, String) {
    let result = match result {
        Some(r) => r,
        None => return (false, String::new()),
    };

    if tool_name == "terminal" {
        if let Some(Json::Object(data)) = safe_json_loads(result) {
            if let Some(exit) = data.get("exit_code") {
                if !exit.is_null() {
                    let nonzero = match exit {
                        Json::Number(n) => n.as_i64().map(|v| v != 0).unwrap_or_else(|| {
                            n.as_f64().map(|v| v != 0.0).unwrap_or(true)
                        }),
                        _ => true,
                    };
                    if nonzero {
                        return (true, format!(" [exit {}]", json_scalar_to_string(exit)));
                    }
                }
            }
        }
        return (false, String::new());
    }

    if tool_name == "memory" {
        if let Some(Json::Object(data)) = safe_json_loads(result) {
            let success_false = matches!(data.get("success"), Some(Json::Bool(false)));
            let err = data
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if success_false && err.contains("exceed the limit") {
                return (true, " [full]".to_string());
            }
        }
    }

    let head: String = result.chars().take(500).collect();
    let lower = head.to_lowercase();
    if lower.contains("\"error\"") || lower.contains("\"failed\"") || result.starts_with("Error") {
        return (true, " [error]".to_string());
    }

    (false, String::new())
}

/// Per-turn controller for repeated failed/non-progressing tool calls.
pub struct ToolCallGuardrailController {
    pub config: ToolCallGuardrailConfig,
    exact_failure_counts: HashMap<ToolCallSignature, i64>,
    same_tool_failure_counts: HashMap<String, i64>,
    // signature -> (result_hash, repeat_count)
    no_progress: HashMap<ToolCallSignature, (String, i64)>,
    halt_decision: Option<ToolGuardrailDecision>,
}

impl ToolCallGuardrailController {
    pub fn new(config: Option<ToolCallGuardrailConfig>) -> Self {
        let mut c = ToolCallGuardrailController {
            config: config.unwrap_or_default(),
            exact_failure_counts: HashMap::new(),
            same_tool_failure_counts: HashMap::new(),
            no_progress: HashMap::new(),
            halt_decision: None,
        };
        c.reset_for_turn();
        c
    }

    pub fn reset_for_turn(&mut self) {
        self.exact_failure_counts.clear();
        self.same_tool_failure_counts.clear();
        self.no_progress.clear();
        self.halt_decision = None;
    }

    pub fn halt_decision(&self) -> Option<&ToolGuardrailDecision> {
        self.halt_decision.as_ref()
    }

    pub fn before_call(&mut self, tool_name: &str, args: Option<&Json>) -> ToolGuardrailDecision {
        let coerced = coerce_args(args);
        let signature = ToolCallSignature::from_call(tool_name, &coerced);

        if !self.config.hard_stop_enabled {
            return ToolGuardrailDecision::allow_with(tool_name, 0, signature);
        }

        let exact_count = *self.exact_failure_counts.get(&signature).unwrap_or(&0);
        if exact_count >= self.config.exact_failure_block_after {
            let decision = ToolGuardrailDecision {
                action: GuardrailAction::Block,
                code: "repeated_exact_failure_block".to_string(),
                message: format!(
                    "Blocked {tool_name}: the same tool call failed {exact_count} \
times with identical arguments. Stop retrying it unchanged; \
change strategy or explain the blocker."
                ),
                tool_name: tool_name.to_string(),
                count: exact_count,
                signature: Some(signature),
            };
            self.halt_decision = Some(decision.clone());
            return decision;
        }

        if self.is_idempotent(tool_name) {
            if let Some((_result_hash, repeat_count)) = self.no_progress.get(&signature).cloned() {
                if repeat_count >= self.config.no_progress_block_after {
                    let decision = ToolGuardrailDecision {
                        action: GuardrailAction::Block,
                        code: "idempotent_no_progress_block".to_string(),
                        message: format!(
                            "Blocked {tool_name}: this read-only call returned the same \
result {repeat_count} times. Stop repeating it unchanged; \
use the result already provided or try a different query."
                        ),
                        tool_name: tool_name.to_string(),
                        count: repeat_count,
                        signature: Some(signature),
                    };
                    self.halt_decision = Some(decision.clone());
                    return decision;
                }
            }
        }

        ToolGuardrailDecision::allow_with(tool_name, 0, signature)
    }

    pub fn after_call(
        &mut self,
        tool_name: &str,
        args: Option<&Json>,
        result: Option<&str>,
        failed: Option<bool>,
    ) -> ToolGuardrailDecision {
        let coerced = coerce_args(args);
        let signature = ToolCallSignature::from_call(tool_name, &coerced);
        let failed = failed.unwrap_or_else(|| classify_tool_failure(tool_name, result).0);

        if failed {
            let exact_count = self.exact_failure_counts.get(&signature).unwrap_or(&0) + 1;
            self.exact_failure_counts.insert(signature.clone(), exact_count);
            self.no_progress.remove(&signature);

            let same_count = self.same_tool_failure_counts.get(tool_name).unwrap_or(&0) + 1;
            self.same_tool_failure_counts
                .insert(tool_name.to_string(), same_count);

            if self.config.hard_stop_enabled
                && same_count >= self.config.same_tool_failure_halt_after
            {
                let decision = ToolGuardrailDecision {
                    action: GuardrailAction::Halt,
                    code: "same_tool_failure_halt".to_string(),
                    message: format!(
                        "Stopped {tool_name}: it failed {same_count} times this turn. \
Stop retrying the same failing tool path and choose a different approach."
                    ),
                    tool_name: tool_name.to_string(),
                    count: same_count,
                    signature: Some(signature),
                };
                self.halt_decision = Some(decision.clone());
                return decision;
            }

            if self.config.warnings_enabled && exact_count >= self.config.exact_failure_warn_after {
                return ToolGuardrailDecision {
                    action: GuardrailAction::Warn,
                    code: "repeated_exact_failure_warning".to_string(),
                    message: format!(
                        "{tool_name} has failed {exact_count} times with identical arguments. \
This looks like a loop; inspect the error and change strategy \
instead of retrying it unchanged."
                    ),
                    tool_name: tool_name.to_string(),
                    count: exact_count,
                    signature: Some(signature),
                };
            }

            if self.config.warnings_enabled
                && same_count >= self.config.same_tool_failure_warn_after
            {
                return ToolGuardrailDecision {
                    action: GuardrailAction::Warn,
                    code: "same_tool_failure_warning".to_string(),
                    message: format!(
                        "{tool_name} has failed {same_count} times this turn. \
This looks like a loop; change approach before retrying."
                    ),
                    tool_name: tool_name.to_string(),
                    count: same_count,
                    signature: Some(signature),
                };
            }

            return ToolGuardrailDecision::allow_with(tool_name, exact_count, signature);
        }

        // Success path.
        self.exact_failure_counts.remove(&signature);
        self.same_tool_failure_counts.remove(tool_name);

        if !self.is_idempotent(tool_name) {
            self.no_progress.remove(&signature);
            return ToolGuardrailDecision::allow_with(tool_name, 0, signature);
        }

        let result_hash = result_hash(result);
        let previous = self.no_progress.get(&signature).cloned();
        let mut repeat_count = 1;
        if let Some((prev_hash, prev_count)) = previous {
            if prev_hash == result_hash {
                repeat_count = prev_count + 1;
            }
        }
        self.no_progress
            .insert(signature.clone(), (result_hash, repeat_count));

        if self.config.warnings_enabled && repeat_count >= self.config.no_progress_warn_after {
            return ToolGuardrailDecision {
                action: GuardrailAction::Warn,
                code: "idempotent_no_progress_warning".to_string(),
                message: format!(
                    "{tool_name} returned the same result {repeat_count} times. \
Use the result already provided or change the query instead of \
repeating it unchanged."
                ),
                tool_name: tool_name.to_string(),
                count: repeat_count,
                signature: Some(signature),
            };
        }

        ToolGuardrailDecision::allow_with(tool_name, repeat_count, signature)
    }

    fn is_idempotent(&self, tool_name: &str) -> bool {
        if self.config.mutating_tools.iter().any(|t| t == tool_name) {
            return false;
        }
        self.config.idempotent_tools.iter().any(|t| t == tool_name)
    }
}

/// Build a synthetic role=tool content string for a blocked tool call.
pub fn toolguard_synthetic_result(decision: &ToolGuardrailDecision) -> String {
    let mut obj = serde_json::Map::new();
    obj.insert("error".to_string(), Json::String(decision.message.clone()));
    obj.insert("guardrail".to_string(), decision.to_metadata());
    serde_json::to_string(&Json::Object(obj)).unwrap_or_else(|_| "{}".to_string())
}

/// Append runtime guidance to the current tool result content.
pub fn append_toolguard_guidance(result: &str, decision: &ToolGuardrailDecision) -> String {
    let is_warn_or_halt =
        matches!(decision.action, GuardrailAction::Warn | GuardrailAction::Halt);
    if !is_warn_or_halt || decision.message.is_empty() {
        return result.to_string();
    }
    let label = if decision.action == GuardrailAction::Halt {
        "Tool loop hard stop"
    } else {
        "Tool loop warning"
    };
    let suffix = format!(
        "\n\n[{label}: {}; count={}; {}]",
        decision.code, decision.count, decision.message
    );
    format!("{result}{suffix}")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn coerce_args(args: Option<&Json>) -> Json {
    match args {
        Some(v @ Json::Object(_)) => v.clone(),
        _ => Json::Object(serde_json::Map::new()),
    }
}

/// Parse JSON, returning None for empty/whitespace-only or invalid input.
/// Mirrors `utils.safe_json_loads`.
fn safe_json_loads(text: &str) -> Option<Json> {
    if text.trim().is_empty() {
        return None;
    }
    serde_json::from_str(text).ok()
}

fn result_hash(result: Option<&str>) -> String {
    let raw = result.unwrap_or("");
    let canonical = match safe_json_loads(raw) {
        Some(parsed) => {
            let sorted = sort_json_keys(&parsed);
            serde_json::to_string(&sorted).unwrap_or_else(|_| raw.to_string())
        }
        None => raw.to_string(),
    };
    sha256_hex(&canonical)
}

fn as_bool(value: Option<&serde_yaml::Value>, default: bool) -> bool {
    match value {
        None | Some(serde_yaml::Value::Null) => default,
        Some(serde_yaml::Value::Bool(b)) => *b,
        Some(serde_yaml::Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(f) = n.as_f64() {
                f != 0.0
            } else {
                default
            }
        }
        Some(serde_yaml::Value::String(s)) => {
            let lowered = s.trim().to_lowercase();
            match lowered.as_str() {
                "1" | "true" | "yes" | "on" | "enabled" => true,
                "0" | "false" | "no" | "off" | "disabled" => false,
                _ => default,
            }
        }
        _ => default,
    }
}

fn positive_int(value: Option<&serde_yaml::Value>, default: i64) -> i64 {
    let parsed: Option<i64> = match value {
        None | Some(serde_yaml::Value::Null) => return default,
        Some(serde_yaml::Value::Number(n)) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else {
                // Truncate floats like Python int(float).
                n.as_f64().map(|f| f as i64)
            }
        }
        Some(serde_yaml::Value::Bool(b)) => Some(if *b { 1 } else { 0 }),
        Some(serde_yaml::Value::String(s)) => s.trim().parse::<i64>().ok(),
        _ => None,
    };
    match parsed {
        Some(p) if p >= 1 => p,
        _ => default,
    }
}

fn json_scalar_to_string(v: &Json) -> String {
    match v {
        Json::String(s) => s.clone(),
        Json::Number(n) => n.to_string(),
        Json::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg_hardstop() -> ToolCallGuardrailConfig {
        ToolCallGuardrailConfig {
            hard_stop_enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn canonical_args_sorts_keys_compactly() {
        let args = json!({"b": 1, "a": {"z": 2, "y": 3}});
        assert_eq!(canonical_tool_args(&args), r#"{"a":{"y":3,"z":2},"b":1}"#);
    }

    #[test]
    fn canonical_args_stable_hash() {
        let a = ToolCallSignature::from_call("read_file", &json!({"x": 1, "y": 2}));
        let b = ToolCallSignature::from_call("read_file", &json!({"y": 2, "x": 1}));
        assert_eq!(a.args_hash, b.args_hash);
        assert_eq!(a, b);
    }

    #[test]
    fn classify_terminal_nonzero_exit() {
        let (failed, tag) = classify_tool_failure("terminal", Some(r#"{"exit_code": 2}"#));
        assert!(failed);
        assert_eq!(tag, " [exit 2]");
    }

    #[test]
    fn classify_terminal_zero_exit_ok() {
        let (failed, tag) = classify_tool_failure("terminal", Some(r#"{"exit_code": 0}"#));
        assert!(!failed);
        assert_eq!(tag, "");
    }

    #[test]
    fn classify_memory_full() {
        let res = r#"{"success": false, "error": "you exceed the limit now"}"#;
        let (failed, tag) = classify_tool_failure("memory", Some(res));
        assert!(failed);
        assert_eq!(tag, " [full]");
    }

    #[test]
    fn classify_error_substring() {
        let (failed, tag) = classify_tool_failure("web_search", Some(r#"{"error": "boom"}"#));
        assert!(failed);
        assert_eq!(tag, " [error]");
        let (failed2, _) = classify_tool_failure("web_search", Some("Error: nope"));
        assert!(failed2);
    }

    #[test]
    fn classify_none_result() {
        let (failed, tag) = classify_tool_failure("read_file", None);
        assert!(!failed);
        assert_eq!(tag, "");
    }

    #[test]
    fn no_hardstop_before_call_allows() {
        let mut c = ToolCallGuardrailController::new(None);
        let d = c.before_call("read_file", Some(&json!({"a": 1})));
        assert_eq!(d.action, GuardrailAction::Allow);
        assert!(d.allows_execution());
    }

    #[test]
    fn exact_failure_warns_then_blocks() {
        let mut c = ToolCallGuardrailController::new(Some(cfg_hardstop()));
        let args = json!({"q": "x"});
        // First failure: no warn yet (warn_after = 2).
        let d1 = c.after_call("write_file", Some(&args), Some(r#"{"error":"e"}"#), Some(true));
        assert_eq!(d1.action, GuardrailAction::Allow);
        // Second failure: warn.
        let d2 = c.after_call("write_file", Some(&args), Some(r#"{"error":"e"}"#), Some(true));
        assert_eq!(d2.action, GuardrailAction::Warn);
        assert_eq!(d2.code, "repeated_exact_failure_warning");
        // Keep failing until block_after (5) reached, then before_call blocks.
        c.after_call("write_file", Some(&args), Some(r#"{"error":"e"}"#), Some(true));
        c.after_call("write_file", Some(&args), Some(r#"{"error":"e"}"#), Some(true));
        c.after_call("write_file", Some(&args), Some(r#"{"error":"e"}"#), Some(true));
        let block = c.before_call("write_file", Some(&args));
        assert_eq!(block.action, GuardrailAction::Block);
        assert!(block.should_halt());
        assert!(c.halt_decision().is_some());
    }

    #[test]
    fn same_tool_failure_halts() {
        let cfg = ToolCallGuardrailConfig {
            hard_stop_enabled: true,
            same_tool_failure_halt_after: 3,
            // raise warn thresholds so we hit halt path cleanly
            exact_failure_warn_after: 99,
            same_tool_failure_warn_after: 99,
            ..Default::default()
        };
        let mut c = ToolCallGuardrailController::new(Some(cfg));
        // Vary args so exact-failure counter stays at 1 each, same-tool grows.
        c.after_call("terminal", Some(&json!({"cmd": "a"})), Some("Error"), Some(true));
        c.after_call("terminal", Some(&json!({"cmd": "b"})), Some("Error"), Some(true));
        let d = c.after_call("terminal", Some(&json!({"cmd": "c"})), Some("Error"), Some(true));
        assert_eq!(d.action, GuardrailAction::Halt);
        assert_eq!(d.code, "same_tool_failure_halt");
    }

    #[test]
    fn idempotent_no_progress_warns_and_blocks() {
        let mut c = ToolCallGuardrailController::new(Some(cfg_hardstop()));
        let args = json!({"path": "/f"});
        let result = r#"{"content": "same"}"#;
        // First success: repeat 1, no warn.
        let d1 = c.after_call("read_file", Some(&args), Some(result), Some(false));
        assert_eq!(d1.action, GuardrailAction::Allow);
        // Second identical: repeat 2 -> warn (no_progress_warn_after=2).
        let d2 = c.after_call("read_file", Some(&args), Some(result), Some(false));
        assert_eq!(d2.action, GuardrailAction::Warn);
        assert_eq!(d2.code, "idempotent_no_progress_warning");
        // Reach block threshold (5) and confirm before_call blocks.
        c.after_call("read_file", Some(&args), Some(result), Some(false));
        c.after_call("read_file", Some(&args), Some(result), Some(false));
        c.after_call("read_file", Some(&args), Some(result), Some(false));
        let block = c.before_call("read_file", Some(&args));
        assert_eq!(block.action, GuardrailAction::Block);
        assert_eq!(block.code, "idempotent_no_progress_block");
    }

    #[test]
    fn success_clears_failure_counters() {
        let mut c = ToolCallGuardrailController::new(Some(cfg_hardstop()));
        let args = json!({"q": "x"});
        c.after_call("write_file", Some(&args), Some("Error"), Some(true));
        c.after_call("write_file", Some(&args), Some("Error"), Some(true));
        // Success resets the exact-failure count for this signature.
        c.after_call("write_file", Some(&args), Some("{}"), Some(false));
        let d = c.after_call("write_file", Some(&args), Some("Error"), Some(true));
        // Back to count 1, below warn threshold.
        assert_eq!(d.action, GuardrailAction::Allow);
        assert_eq!(d.count, 1);
    }

    #[test]
    fn mutating_tool_not_idempotent() {
        let c = ToolCallGuardrailController::new(None);
        assert!(!c.is_idempotent("write_file"));
        assert!(c.is_idempotent("read_file"));
        assert!(!c.is_idempotent("unknown_tool"));
    }

    #[test]
    fn synthetic_result_contains_error_and_metadata() {
        let mut c = ToolCallGuardrailController::new(Some(cfg_hardstop()));
        let args = json!({"q": "x"});
        for _ in 0..5 {
            c.after_call("write_file", Some(&args), Some("Error"), Some(true));
        }
        let block = c.before_call("write_file", Some(&args));
        let s = toolguard_synthetic_result(&block);
        let parsed: Json = serde_json::from_str(&s).unwrap();
        assert!(parsed.get("error").is_some());
        assert_eq!(parsed["guardrail"]["action"], json!("block"));
    }

    #[test]
    fn append_guidance_only_for_warn_or_halt() {
        let warn = ToolGuardrailDecision {
            action: GuardrailAction::Warn,
            code: "x".to_string(),
            message: "msg".to_string(),
            count: 3,
            ..Default::default()
        };
        let out = append_toolguard_guidance("base", &warn);
        assert!(out.starts_with("base"));
        assert!(out.contains("[Tool loop warning: x; count=3; msg]"));

        let allow = ToolGuardrailDecision::default();
        assert_eq!(append_toolguard_guidance("base", &allow), "base");
    }

    #[test]
    fn config_from_mapping_nested_and_flat() {
        let yaml = r#"
warnings_enabled: false
hard_stop_enabled: true
warn_after:
  exact_failure: 4
hard_stop_after:
  same_tool_failure: 12
no_progress_warn_after: 3
"#;
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let cfg = ToolCallGuardrailConfig::from_mapping(Some(&val));
        assert!(!cfg.warnings_enabled);
        assert!(cfg.hard_stop_enabled);
        assert_eq!(cfg.exact_failure_warn_after, 4);
        assert_eq!(cfg.same_tool_failure_halt_after, 12);
        // flat fallback used
        assert_eq!(cfg.no_progress_warn_after, 3);
        // untouched default
        assert_eq!(cfg.exact_failure_block_after, 5);
    }

    #[test]
    fn config_from_mapping_invalid_keeps_defaults() {
        let val = serde_yaml::Value::String("nope".to_string());
        let cfg = ToolCallGuardrailConfig::from_mapping(Some(&val));
        assert!(cfg.warnings_enabled);
        assert!(!cfg.hard_stop_enabled);
    }

    #[test]
    fn positive_int_rejects_zero_and_negative() {
        assert_eq!(
            positive_int(Some(&serde_yaml::from_str("0").unwrap()), 7),
            7
        );
        assert_eq!(
            positive_int(Some(&serde_yaml::from_str("-3").unwrap()), 7),
            7
        );
        assert_eq!(
            positive_int(Some(&serde_yaml::from_str("5").unwrap()), 7),
            5
        );
    }
}
