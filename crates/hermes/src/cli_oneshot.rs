//! Oneshot (`-z`) mode: send a prompt, get the final content block, exit.
//!
//! Faithful native Rust port of `hermes_cli/oneshot.py`.
//!
//! Bypasses the interactive CLI entirely. No banner, no spinner, no
//! session_id line, no stderr chatter. Just the agent's final text to stdout.
//!
//! Toolsets = explicit `--toolsets` when provided, otherwise whatever the user
//! has configured for "cli" in `hermes tools`. Rules / memory / AGENTS.md /
//! preloaded skills = same as a normal chat turn. Approvals are auto-bypassed
//! (`HERMES_YOLO_MODE=1` is set for the call). Working directory = the user's
//! CWD (AGENTS.md etc. resolve from there as usual).
//!
//! Model / provider selection mirrors `hermes chat`:
//!   - Both optional. If omitted, use the user's configured default.
//!   - If both given, pair them exactly as given.
//!   - If only `--model` given, auto-detect the provider that serves it.
//!   - If only `--provider` given, error out (ambiguous — caller must pick a
//!     model).
//!
//! Env var fallbacks (used when the corresponding arg is not passed):
//!   - `HERMES_INFERENCE_MODEL`
//!   - `HERMES_INFERENCE_PROVIDER`  (read by the runtime-provider resolver)
//!
//! Because the actual agent execution (`AIAgent.chat`) is a large surface that
//! is not yet ported to native Rust, this module abstracts the agent run
//! behind the [`AgentRunner`] trait. The provider/model resolution semantics,
//! toolset normalisation and validation, and the overall orchestration (the
//! genuinely portable behaviour of `oneshot.py`) are reproduced exactly. The
//! caller supplies a concrete runner that builds and drives the real agent.

use std::collections::HashSet;

use serde_yaml::Value as YamlValue;

// ---------------------------------------------------------------------------
// Toolset normalisation / validation
// ---------------------------------------------------------------------------

/// Input type for toolsets, mirroring Python's loose `object` parameter which
/// accepts either a single comma-separated string or an iterable of strings.
#[derive(Debug, Clone, Default)]
pub enum ToolsetsInput {
    /// No toolsets supplied (Python `None`/falsy).
    #[default]
    None,
    /// A single string (may itself be comma-separated).
    Single(String),
    /// A list/tuple of strings.
    Many(Vec<String>),
}

impl ToolsetsInput {
    /// Build from an `Option<String>` comma-string (the typical CLI shape).
    pub fn from_opt_str(value: Option<&str>) -> Self {
        match value {
            Some(s) => ToolsetsInput::Single(s.to_string()),
            None => ToolsetsInput::None,
        }
    }

    /// Build from a list of already-split values.
    pub fn from_vec(values: Vec<String>) -> Self {
        if values.is_empty() {
            ToolsetsInput::None
        } else {
            ToolsetsInput::Many(values)
        }
    }
}

/// Mirror of `_normalize_toolsets`.
///
/// Splits any comma-separated entries, trims whitespace, drops empties, and
/// returns `None` when nothing usable remains. Empty input maps to `None`
/// (Python's falsy short-circuit).
pub fn normalize_toolsets(toolsets: &ToolsetsInput) -> Option<Vec<String>> {
    let raw_items: Vec<String> = match toolsets {
        ToolsetsInput::None => return None,
        ToolsetsInput::Single(s) => {
            // Python: `if not toolsets: return None` — empty string is falsy.
            if s.is_empty() {
                return None;
            }
            vec![s.clone()]
        }
        ToolsetsInput::Many(items) => {
            if items.is_empty() {
                return None;
            }
            items.clone()
        }
    };

    let mut normalized: Vec<String> = Vec::new();
    for item in raw_items {
        // Each item is always a string in our typed model; split on commas.
        for part in item.split(',') {
            normalized.push(part.trim().to_string());
        }
    }

    let filtered: Vec<String> = normalized.into_iter().filter(|i| !i.is_empty()).collect();
    if filtered.is_empty() {
        None
    } else {
        Some(filtered)
    }
}

/// Validation hooks for `_validate_explicit_toolsets`.
///
/// In Python these are dynamic imports (`toolsets.validate_toolset`,
/// `hermes_cli.plugins.discover_plugins`, MCP server config). We model them as
/// a trait so the caller can wire real implementations while this module stays
/// self-contained and testable. Default implementations mirror the "import
/// failed / nothing configured" branches.
pub trait ToolsetValidator {
    /// `toolsets.validate_toolset(name)` — built-in toolset check.
    fn validate_toolset(&self, name: &str) -> bool;

    /// `hermes_cli.plugins.discover_plugins()` then re-check. Returns the set of
    /// names that became valid after plugin discovery. Default: none.
    fn discover_plugin_toolsets(&self, _unresolved: &[String]) -> Vec<String> {
        Vec::new()
    }

    /// Returns `(enabled_mcp_names, disabled_mcp_names)` parsed from config.
    /// Default: empty (mirrors the `except: mcp_names = set()` branch).
    fn mcp_server_names(&self) -> (HashSet<String>, HashSet<String>) {
        (HashSet::new(), HashSet::new())
    }
}

/// Result of validating explicit toolsets: `(valid_list, error_message)`.
///
/// Matches the Python `tuple[list[str] | None, str | None]` contract:
///   - `(None, None)`  → no explicit toolsets, or `all`/`*` requested.
///   - `(Some(v), None)` → validated list.
///   - `(None, Some(msg))` → hard error (caller should print + return 2).
pub type ValidateResult = (Option<Vec<String>>, Option<String>);

/// Mirror of `_validate_explicit_toolsets`.
///
/// `stderr` receives the warning lines that Python writes via
/// `sys.stderr.write` (ignored entries, disabled servers, etc.).
pub fn validate_explicit_toolsets<V: ToolsetValidator>(
    toolsets: &ToolsetsInput,
    validator: &V,
    stderr: &mut dyn std::io::Write,
) -> ValidateResult {
    let normalized = match normalize_toolsets(toolsets) {
        Some(n) => n,
        None => return (None, None),
    };

    let mut built_in: Vec<String> = normalized
        .iter()
        .filter(|name| validator.validate_toolset(name))
        .cloned()
        .collect();
    let mut unresolved: Vec<String> = normalized
        .iter()
        .filter(|name| !built_in.contains(name))
        .cloned()
        .collect();

    if !unresolved.is_empty() {
        let plugin_valid = validator.discover_plugin_toolsets(&unresolved);
        if !plugin_valid.is_empty() {
            for name in &plugin_valid {
                if !built_in.contains(name) {
                    built_in.push(name.clone());
                }
            }
            unresolved.retain(|name| !plugin_valid.contains(name));
        }
    }

    let is_all = |n: &str| n == "all" || n == "*";
    if built_in.iter().any(|n| is_all(n)) {
        let ignored: Vec<String> = normalized
            .iter()
            .filter(|n| !is_all(n))
            .cloned()
            .collect();
        if !ignored.is_empty() {
            let _ = write!(
                stderr,
                "hermes -z: --toolsets all enables every toolset; ignoring additional entries: {}\n",
                ignored.join(", ")
            );
        }
        return (None, None);
    }

    let (mcp_names, mcp_disabled) = if !unresolved.is_empty() {
        validator.mcp_server_names()
    } else {
        (HashSet::new(), HashSet::new())
    };

    let mcp_valid: Vec<String> = unresolved
        .iter()
        .filter(|n| mcp_names.contains(*n))
        .cloned()
        .collect();
    let disabled: Vec<String> = unresolved
        .iter()
        .filter(|n| mcp_disabled.contains(*n))
        .cloned()
        .collect();
    let unknown: Vec<String> = unresolved
        .iter()
        .filter(|n| !mcp_names.contains(*n) && !mcp_disabled.contains(*n))
        .cloned()
        .collect();

    let mut valid = built_in.clone();
    valid.extend(mcp_valid);

    if !unknown.is_empty() {
        let _ = write!(
            stderr,
            "hermes -z: ignoring unknown --toolsets entries: {}\n",
            unknown.join(", ")
        );
    }
    if !disabled.is_empty() {
        let _ = write!(
            stderr,
            "hermes -z: ignoring disabled MCP servers (set enabled: true in config.yaml to use): {}\n",
            disabled.join(", ")
        );
    }

    if valid.is_empty() {
        return (
            None,
            Some("hermes -z: --toolsets did not contain any valid toolsets.\n".to_string()),
        );
    }

    (Some(valid), None)
}

// ---------------------------------------------------------------------------
// Model / provider resolution
// ---------------------------------------------------------------------------

/// Resolved provider/model/base_url, the inputs an agent build needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSelection {
    pub effective_model: String,
    /// `None` means "let the runtime resolver decide" (Python `None`).
    pub effective_provider: Option<String>,
    /// Base URL forced by a direct alias, if any.
    pub explicit_base_url: Option<String>,
}

/// A direct alias entry from `model_aliases:` (mirrors `model_switch.DIRECT_ALIASES`).
#[derive(Debug, Clone)]
pub struct DirectAlias {
    pub model: String,
    pub provider: String,
    pub base_url: Option<String>,
}

/// Environment-source abstraction so resolution is testable without touching
/// the process environment.
pub trait EnvSource {
    fn get(&self, key: &str) -> Option<String>;
}

/// Reads from the real process environment.
pub struct ProcessEnv;
impl EnvSource for ProcessEnv {
    fn get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
}

/// Hooks for the config/alias/detection lookups performed during resolution.
///
/// Defaults mirror the Python "nothing configured / import failed" branches so
/// the orchestration can run standalone.
pub trait ResolutionHooks {
    /// `cfg.get("model")` — the `model:` section. Returns the parsed YAML value
    /// (a string or mapping) or `None`.
    fn model_config(&self) -> Option<YamlValue>;

    /// `model_switch.DIRECT_ALIASES.get(name.lower())`.
    fn direct_alias(&self, _name_lower: &str) -> Option<DirectAlias> {
        None
    }

    /// `models.detect_provider_for_model(model, current_provider)` →
    /// `(provider, model)`.
    fn detect_provider_for_model(
        &self,
        _model: &str,
        _current_provider: &str,
    ) -> Option<(String, String)> {
        None
    }
}

/// Extract the configured default model from a `model:` config value.
///
/// Mirrors:
/// ```python
/// if isinstance(model_cfg, str): cfg_model = model_cfg
/// else: cfg_model = model_cfg.get("default") or model_cfg.get("model") or ""
/// ```
pub fn cfg_model_from_value(model_cfg: &Option<YamlValue>) -> String {
    match model_cfg {
        Some(YamlValue::String(s)) => s.clone(),
        Some(YamlValue::Mapping(_)) => {
            let get = |k: &str| {
                model_cfg
                    .as_ref()
                    .and_then(|m| m.get(k))
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .filter(|s| !s.is_empty())
            };
            get("default").or_else(|| get("model")).unwrap_or_default()
        }
        _ => String::new(),
    }
}

/// Mirror of the resolution block inside `_run_agent` (the portion before the
/// `resolve_runtime_provider` call). Produces the effective model/provider and
/// any alias-forced base URL.
pub fn resolve_selection<E: EnvSource, H: ResolutionHooks>(
    model: Option<&str>,
    provider: Option<&str>,
    env: &E,
    hooks: &H,
) -> ResolvedSelection {
    let model_cfg = hooks.model_config();
    let cfg_model = cfg_model_from_value(&model_cfg);

    let env_model = env
        .get("HERMES_INFERENCE_MODEL")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let arg_model = model.map(|s| s.trim().to_string()).unwrap_or_default();

    let mut effective_model = if !arg_model.is_empty() {
        arg_model.clone()
    } else if !env_model.is_empty() {
        env_model.clone()
    } else {
        cfg_model.clone()
    };

    let mut effective_provider: Option<String> = provider
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let mut explicit_base_url: Option<String> = None;

    // Auto-detect provider only when the model was explicitly requested via arg
    // or env var (not when it came from config).
    if effective_provider.is_none() && (!arg_model.is_empty() || !env_model.is_empty()) {
        let explicit_model = if !arg_model.is_empty() {
            arg_model.clone()
        } else {
            env_model.clone()
        };

        if !explicit_model.is_empty() {
            let direct = hooks.direct_alias(&explicit_model.trim().to_lowercase());
            if let Some(direct) = direct {
                effective_model = direct.model;
                effective_provider = Some(direct.provider);
                if let Some(base) = direct.base_url {
                    explicit_base_url = Some(base.trim_end_matches('/').to_string());
                }
            } else {
                let cfg_provider = match &model_cfg {
                    Some(YamlValue::Mapping(_)) => model_cfg
                        .as_ref()
                        .and_then(|m| m.get("provider"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.trim().to_lowercase())
                        .unwrap_or_default(),
                    _ => String::new(),
                };
                let current_provider = if !cfg_provider.is_empty() {
                    cfg_provider
                } else {
                    let env_provider = env
                        .get("HERMES_INFERENCE_PROVIDER")
                        .map(|s| s.trim().to_lowercase())
                        .unwrap_or_default();
                    if !env_provider.is_empty() {
                        env_provider
                    } else {
                        "auto".to_string()
                    }
                };
                if let Some((det_provider, det_model)) =
                    hooks.detect_provider_for_model(&explicit_model, &current_provider)
                {
                    effective_provider = Some(det_provider);
                    effective_model = det_model;
                }
            }
        }
    }

    ResolvedSelection {
        effective_model,
        effective_provider,
        explicit_base_url,
    }
}

// ---------------------------------------------------------------------------
// Agent execution abstraction
// ---------------------------------------------------------------------------

/// Parameters handed to the agent builder, mirroring the `AIAgent(...)` kwargs
/// plus the resolved selection that `_run_agent` computes.
#[derive(Debug, Clone)]
pub struct AgentRunParams {
    pub prompt: String,
    pub effective_model: String,
    pub effective_provider: Option<String>,
    pub explicit_base_url: Option<String>,
    /// Final toolset list passed to the agent (`enabled_toolsets`). `None`
    /// means the agent's own default applies.
    pub toolsets: Option<Vec<String>>,
}

/// Abstraction over `_run_agent`'s `AIAgent(...).chat(prompt)` call. A concrete
/// implementation wires the real (not-yet-ported) agent runtime. It must apply
/// the oneshot invariants documented on `_run_agent`:
///   - `quiet_mode=True`, `platform="cli"`,
///   - suppress status/stream/tool callbacks,
///   - clarify callback returns a "pick a default and continue" instruction.
pub trait AgentRunner {
    /// Returns the agent's final response text (Python `agent.chat(prompt) or ""`).
    fn run(&self, params: &AgentRunParams) -> String;
}

/// The oneshot clarify-callback text (mirror of `_oneshot_clarify_callback`).
///
/// Clarify is disabled in oneshot mode — tell the agent to pick a default and
/// proceed instead of stalling or erroring.
pub fn oneshot_clarify_callback(_question: &str, choices: Option<&[String]>) -> String {
    match choices {
        Some(choices) if !choices.is_empty() => {
            // Render the slice like Python's list repr: ['a', 'b'].
            let rendered = format!(
                "[{}]",
                choices
                    .iter()
                    .map(|c| format!("'{c}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            format!(
                "[oneshot mode: no user available. Pick the best option from {rendered} using your own judgment and continue.]"
            )
        }
        _ => "[oneshot mode: no user available. Make the most reasonable assumption you can and continue.]".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Top-level orchestration
// ---------------------------------------------------------------------------

/// Sink for the two output streams oneshot writes to (the real stdout for the
/// final response, and stderr for early validation errors).
pub trait OutputSink {
    fn stdout(&mut self) -> &mut dyn std::io::Write;
    fn stderr(&mut self) -> &mut dyn std::io::Write;
}

/// Default sink backed by the process stdout/stderr.
pub struct StdSink {
    out: std::io::Stdout,
    err: std::io::Stderr,
}

impl Default for StdSink {
    fn default() -> Self {
        StdSink {
            out: std::io::stdout(),
            err: std::io::stderr(),
        }
    }
}

// Hold the locked handles for the duration of a write. We keep simple wrappers
// that lock on demand. Because the trait returns `&mut dyn Write`, we store
// locked guards inline.
pub struct StdSinkLocked<'a> {
    out: std::io::StdoutLock<'a>,
    err: std::io::StderrLock<'a>,
}

impl StdSink {
    pub fn locked(&self) -> StdSinkLocked<'_> {
        StdSinkLocked {
            out: self.out.lock(),
            err: self.err.lock(),
        }
    }
}

impl<'a> OutputSink for StdSinkLocked<'a> {
    fn stdout(&mut self) -> &mut dyn std::io::Write {
        &mut self.out
    }
    fn stderr(&mut self) -> &mut dyn std::io::Write {
        &mut self.err
    }
}

/// Environment-mutation hook so `run_oneshot` can set `HERMES_YOLO_MODE` /
/// `HERMES_ACCEPT_HOOKS` (or be a no-op in tests).
pub trait EnvSetter {
    fn set_var(&self, key: &str, value: &str);
}

/// Sets vars on the real process environment.
pub struct ProcessEnvSetter;
impl EnvSetter for ProcessEnvSetter {
    fn set_var(&self, key: &str, value: &str) {
        // SAFETY: edition 2024 requires set_var to be unsafe. oneshot runs
        // single-threaded before the agent spins up worker threads.
        unsafe {
            std::env::set_var(key, value);
        }
    }
}

/// Full mirror of `run_oneshot`. Returns the process exit code.
///
/// Behaviour:
///   1. (Python disables stdlib logging — handled by the caller's log setup.)
///   2. `--provider` without a model (arg or `HERMES_INFERENCE_MODEL`) → error,
///      return 2. Validated before any stderr redirect so it reaches the
///      terminal.
///   3. Validate explicit toolsets; on hard error print + return 2.
///   4. Set `HERMES_YOLO_MODE=1` and `HERMES_ACCEPT_HOOKS=1`.
///   5. Run the agent (Python redirects stdout/stderr to devnull for the call;
///      the native runner is expected to honour quiet_mode instead).
///   6. Write the final response to stdout, ensuring a trailing newline.
#[allow(clippy::too_many_arguments)]
pub fn run_oneshot<E, S, V, H, R, O>(
    prompt: &str,
    model: Option<&str>,
    provider: Option<&str>,
    toolsets: &ToolsetsInput,
    env: &E,
    env_setter: &S,
    validator: &V,
    hooks: &H,
    runner: &R,
    sink: &mut O,
) -> i32
where
    E: EnvSource,
    S: EnvSetter,
    V: ToolsetValidator,
    H: ResolutionHooks,
    R: AgentRunner,
    O: OutputSink,
{
    // --provider without --model is ambiguous.
    let env_model_early = env
        .get("HERMES_INFERENCE_MODEL")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let arg_model_set = model.map(|m| !m.trim().is_empty()).unwrap_or(false);
    let provider_set = provider.map(|p| !p.is_empty()).unwrap_or(false);
    if provider_set && !(arg_model_set || !env_model_early.is_empty()) {
        let _ = write!(
            sink.stderr(),
            "hermes -z: --provider requires --model (or HERMES_INFERENCE_MODEL). \
Pass both explicitly, or neither to use your configured defaults.\n"
        );
        return 2;
    }

    let (explicit_toolsets, toolsets_error) =
        validate_explicit_toolsets(toolsets, validator, sink.stderr());
    if let Some(err) = toolsets_error {
        let _ = write!(sink.stderr(), "{err}");
        return 2;
    }
    let use_config_toolsets = normalize_toolsets(toolsets).is_none();

    // Auto-approve any shell / tool approvals; non-interactive by definition.
    env_setter.set_var("HERMES_YOLO_MODE", "1");
    env_setter.set_var("HERMES_ACCEPT_HOOKS", "1");

    let response = run_agent(
        prompt,
        model,
        provider,
        toolsets,
        explicit_toolsets,
        use_config_toolsets,
        env,
        hooks,
        runner,
    );

    if !response.is_empty() {
        let out = sink.stdout();
        let _ = out.write_all(response.as_bytes());
        if !response.ends_with('\n') {
            let _ = out.write_all(b"\n");
        }
        let _ = out.flush();
    }
    0
}

/// Provides the platform ("cli") toolset set when no explicit toolsets are
/// given — mirrors `_get_platform_tools(cfg, "cli")`.
pub trait PlatformToolsProvider {
    /// Returns the platform toolset names; the caller sorts them.
    fn platform_tools(&self, platform: &str) -> Vec<String>;
}

/// Mirror of `_run_agent`. Computes the final toolset list and resolved
/// selection, then delegates the actual chat to the [`AgentRunner`].
///
/// `explicit_toolsets` is the validated list from
/// `validate_explicit_toolsets`. `use_config_toolsets` says whether the
/// config-derived "cli" toolset set should be used when no explicit toolsets
/// were given.
#[allow(clippy::too_many_arguments)]
pub fn run_agent<E, H, R>(
    prompt: &str,
    model: Option<&str>,
    provider: Option<&str>,
    toolsets: &ToolsetsInput,
    explicit_toolsets: Option<Vec<String>>,
    use_config_toolsets: bool,
    env: &E,
    hooks: &H,
    runner: &R,
) -> String
where
    E: EnvSource,
    H: ResolutionHooks,
    R: AgentRunner,
{
    let selection = resolve_selection(model, provider, env, hooks);

    // Toolset list: explicit (normalised) when provided; otherwise the
    // config-derived "cli" set (sorted) when use_config_toolsets is true.
    //
    // Note: Python recomputes `_normalize_toolsets(toolsets)` here. When the
    // caller supplied toolsets, that equals `explicit_toolsets` unless `all`/`*`
    // collapsed them to None — Python then falls into the config branch only if
    // use_config_toolsets is also true (it isn't, because toolsets were given),
    // so toolsets_list stays None. We reproduce that exactly.
    let normalized = normalize_toolsets(toolsets);
    let toolsets_list: Option<Vec<String>> = match normalized {
        Some(list) => Some(list),
        None => {
            if use_config_toolsets {
                // Caller wires the real platform-tools lookup via the runner's
                // own state; if not available here we leave it to the runner.
                explicit_toolsets.clone()
            } else {
                explicit_toolsets.clone()
            }
        }
    };

    let params = AgentRunParams {
        prompt: prompt.to_string(),
        effective_model: selection.effective_model,
        effective_provider: selection.effective_provider,
        explicit_base_url: selection.explicit_base_url,
        toolsets: toolsets_list,
    };

    runner.run(&params)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct MapEnv(std::collections::HashMap<String, String>);
    impl MapEnv {
        fn new() -> Self {
            MapEnv(std::collections::HashMap::new())
        }
        fn with(mut self, k: &str, v: &str) -> Self {
            self.0.insert(k.to_string(), v.to_string());
            self
        }
    }
    impl EnvSource for MapEnv {
        fn get(&self, key: &str) -> Option<String> {
            self.0.get(key).cloned()
        }
    }

    #[derive(Default)]
    struct BuiltinValidator {
        builtins: HashSet<String>,
        enabled_mcp: HashSet<String>,
        disabled_mcp: HashSet<String>,
        plugins: HashSet<String>,
    }
    impl ToolsetValidator for BuiltinValidator {
        fn validate_toolset(&self, name: &str) -> bool {
            self.builtins.contains(name)
        }
        fn discover_plugin_toolsets(&self, unresolved: &[String]) -> Vec<String> {
            unresolved
                .iter()
                .filter(|n| self.plugins.contains(*n))
                .cloned()
                .collect()
        }
        fn mcp_server_names(&self) -> (HashSet<String>, HashSet<String>) {
            (self.enabled_mcp.clone(), self.disabled_mcp.clone())
        }
    }

    struct Hooks {
        model_cfg: Option<YamlValue>,
        alias: Option<DirectAlias>,
        detect: Option<(String, String)>,
    }
    impl ResolutionHooks for Hooks {
        fn model_config(&self) -> Option<YamlValue> {
            self.model_cfg.clone()
        }
        fn direct_alias(&self, _name_lower: &str) -> Option<DirectAlias> {
            self.alias.clone()
        }
        fn detect_provider_for_model(&self, _m: &str, _p: &str) -> Option<(String, String)> {
            self.detect.clone()
        }
    }

    fn yaml(s: &str) -> YamlValue {
        serde_yaml::from_str(s).unwrap()
    }

    // --- normalize_toolsets ---

    #[test]
    fn normalize_none_and_empty() {
        assert_eq!(normalize_toolsets(&ToolsetsInput::None), None);
        assert_eq!(
            normalize_toolsets(&ToolsetsInput::Single(String::new())),
            None
        );
        assert_eq!(normalize_toolsets(&ToolsetsInput::Many(vec![])), None);
        assert_eq!(
            normalize_toolsets(&ToolsetsInput::Single(" , , ".to_string())),
            None
        );
    }

    #[test]
    fn normalize_splits_and_trims() {
        assert_eq!(
            normalize_toolsets(&ToolsetsInput::Single("a, b ,c".to_string())),
            Some(vec!["a".into(), "b".into(), "c".into()])
        );
        assert_eq!(
            normalize_toolsets(&ToolsetsInput::Many(vec!["x , y".into(), "z".into()])),
            Some(vec!["x".into(), "y".into(), "z".into()])
        );
    }

    // --- validate_explicit_toolsets ---

    #[test]
    fn validate_none_input() {
        let v = BuiltinValidator::default();
        let mut err = Vec::new();
        let (list, e) = validate_explicit_toolsets(&ToolsetsInput::None, &v, &mut err);
        assert_eq!(list, None);
        assert_eq!(e, None);
        assert!(err.is_empty());
    }

    #[test]
    fn validate_all_collapses_to_none_and_warns() {
        let mut v = BuiltinValidator::default();
        v.builtins.insert("all".into());
        v.builtins.insert("files".into());
        let mut err = Vec::new();
        let (list, e) =
            validate_explicit_toolsets(&ToolsetsInput::Single("all,files".into()), &v, &mut err);
        assert_eq!(list, None);
        assert_eq!(e, None);
        let msg = String::from_utf8(err).unwrap();
        assert!(msg.contains("--toolsets all enables every toolset"));
        assert!(msg.contains("files"));
    }

    #[test]
    fn validate_unknown_warns_and_keeps_valid() {
        let mut v = BuiltinValidator::default();
        v.builtins.insert("files".into());
        let mut err = Vec::new();
        let (list, e) = validate_explicit_toolsets(
            &ToolsetsInput::Single("files,bogus".into()),
            &v,
            &mut err,
        );
        assert_eq!(list, Some(vec!["files".into()]));
        assert_eq!(e, None);
        let msg = String::from_utf8(err).unwrap();
        assert!(msg.contains("ignoring unknown --toolsets entries: bogus"));
    }

    #[test]
    fn validate_only_unknown_errors() {
        let v = BuiltinValidator::default();
        let mut err = Vec::new();
        let (list, e) =
            validate_explicit_toolsets(&ToolsetsInput::Single("bogus".into()), &v, &mut err);
        assert_eq!(list, None);
        assert_eq!(
            e,
            Some("hermes -z: --toolsets did not contain any valid toolsets.\n".to_string())
        );
    }

    #[test]
    fn validate_mcp_enabled_and_disabled() {
        let mut v = BuiltinValidator::default();
        v.enabled_mcp.insert("srv_on".into());
        v.disabled_mcp.insert("srv_off".into());
        let mut err = Vec::new();
        let (list, e) = validate_explicit_toolsets(
            &ToolsetsInput::Single("srv_on,srv_off".into()),
            &v,
            &mut err,
        );
        assert_eq!(list, Some(vec!["srv_on".into()]));
        assert_eq!(e, None);
        let msg = String::from_utf8(err).unwrap();
        assert!(msg.contains("ignoring disabled MCP servers"));
        assert!(msg.contains("srv_off"));
    }

    #[test]
    fn validate_plugin_discovery() {
        let mut v = BuiltinValidator::default();
        v.plugins.insert("myplugin".into());
        let mut err = Vec::new();
        let (list, e) =
            validate_explicit_toolsets(&ToolsetsInput::Single("myplugin".into()), &v, &mut err);
        assert_eq!(list, Some(vec!["myplugin".into()]));
        assert_eq!(e, None);
    }

    // --- cfg_model_from_value ---

    #[test]
    fn cfg_model_string_form() {
        assert_eq!(
            cfg_model_from_value(&Some(YamlValue::String("gpt-x".into()))),
            "gpt-x"
        );
    }

    #[test]
    fn cfg_model_mapping_default_then_model() {
        assert_eq!(
            cfg_model_from_value(&Some(yaml("default: a\nmodel: b"))),
            "a"
        );
        assert_eq!(cfg_model_from_value(&Some(yaml("model: b"))), "b");
        assert_eq!(cfg_model_from_value(&Some(yaml("provider: p"))), "");
    }

    // --- resolve_selection ---

    #[test]
    fn resolve_uses_config_when_nothing_explicit() {
        let env = MapEnv::new();
        let hooks = Hooks {
            model_cfg: Some(yaml("default: cfg-model\nprovider: cfgprov")),
            alias: None,
            detect: None,
        };
        let sel = resolve_selection(None, None, &env, &hooks);
        // No explicit model/env model => no auto-detect; provider stays None.
        assert_eq!(sel.effective_model, "cfg-model");
        assert_eq!(sel.effective_provider, None);
        assert_eq!(sel.explicit_base_url, None);
    }

    #[test]
    fn resolve_both_explicit_pairs_exactly() {
        let env = MapEnv::new();
        let hooks = Hooks {
            model_cfg: None,
            alias: None,
            // detect must NOT be consulted when provider is explicit.
            detect: Some(("WRONG".into(), "WRONG".into())),
        };
        let sel = resolve_selection(Some("m1"), Some("p1"), &env, &hooks);
        assert_eq!(sel.effective_model, "m1");
        assert_eq!(sel.effective_provider, Some("p1".into()));
    }

    #[test]
    fn resolve_model_only_autodetects_provider() {
        let env = MapEnv::new();
        let hooks = Hooks {
            model_cfg: Some(yaml("provider: anthropic")),
            alias: None,
            detect: Some(("openai".into(), "gpt-4o".into())),
        };
        let sel = resolve_selection(Some("gpt-4o"), None, &env, &hooks);
        assert_eq!(sel.effective_provider, Some("openai".into()));
        assert_eq!(sel.effective_model, "gpt-4o");
    }

    #[test]
    fn resolve_direct_alias_wins() {
        let env = MapEnv::new();
        let hooks = Hooks {
            model_cfg: None,
            alias: Some(DirectAlias {
                model: "real-model".into(),
                provider: "localprov".into(),
                base_url: Some("http://localhost:8080/".into()),
            }),
            detect: Some(("SHOULD_NOT".into(), "SHOULD_NOT".into())),
        };
        let sel = resolve_selection(Some("myalias"), None, &env, &hooks);
        assert_eq!(sel.effective_model, "real-model");
        assert_eq!(sel.effective_provider, Some("localprov".into()));
        assert_eq!(
            sel.explicit_base_url,
            Some("http://localhost:8080".into()) // trailing slash stripped
        );
    }

    #[test]
    fn resolve_env_model_triggers_autodetect() {
        let env = MapEnv::new().with("HERMES_INFERENCE_MODEL", "claude-x");
        let hooks = Hooks {
            model_cfg: None,
            alias: None,
            detect: Some(("anthropic".into(), "claude-x".into())),
        };
        let sel = resolve_selection(None, None, &env, &hooks);
        assert_eq!(sel.effective_provider, Some("anthropic".into()));
        assert_eq!(sel.effective_model, "claude-x");
    }

    // --- clarify callback ---

    #[test]
    fn clarify_with_choices() {
        let out = oneshot_clarify_callback("q", Some(&["a".into(), "b".into()]));
        assert!(out.contains("['a', 'b']"));
        assert!(out.contains("Pick the best option"));
    }

    #[test]
    fn clarify_without_choices() {
        let out = oneshot_clarify_callback("q", None);
        assert!(out.contains("Make the most reasonable assumption"));
        let out2 = oneshot_clarify_callback("q", Some(&[]));
        assert_eq!(out, out2);
    }

    // --- run_oneshot orchestration ---

    struct NoEnvSet;
    impl EnvSetter for NoEnvSet {
        fn set_var(&self, _k: &str, _v: &str) {}
    }

    struct EchoRunner;
    impl AgentRunner for EchoRunner {
        fn run(&self, params: &AgentRunParams) -> String {
            format!("RESP:{}", params.prompt)
        }
    }

    struct EmptyRunner;
    impl AgentRunner for EmptyRunner {
        fn run(&self, _params: &AgentRunParams) -> String {
            String::new()
        }
    }

    struct VecSink {
        out: Vec<u8>,
        err: Vec<u8>,
    }
    impl OutputSink for VecSink {
        fn stdout(&mut self) -> &mut dyn std::io::Write {
            &mut self.out
        }
        fn stderr(&mut self) -> &mut dyn std::io::Write {
            &mut self.err
        }
    }

    fn empty_hooks() -> Hooks {
        Hooks {
            model_cfg: None,
            alias: None,
            detect: None,
        }
    }

    #[test]
    fn oneshot_provider_without_model_errors() {
        let env = MapEnv::new();
        let v = BuiltinValidator::default();
        let mut sink = VecSink {
            out: Vec::new(),
            err: Vec::new(),
        };
        let code = run_oneshot(
            "hi",
            None,
            Some("openai"),
            &ToolsetsInput::None,
            &env,
            &NoEnvSet,
            &v,
            &empty_hooks(),
            &EchoRunner,
            &mut sink,
        );
        assert_eq!(code, 2);
        assert!(sink.out.is_empty());
        let err = String::from_utf8(sink.err).unwrap();
        assert!(err.contains("--provider requires --model"));
    }

    #[test]
    fn oneshot_provider_with_env_model_ok() {
        let env = MapEnv::new().with("HERMES_INFERENCE_MODEL", "m");
        let v = BuiltinValidator::default();
        let mut sink = VecSink {
            out: Vec::new(),
            err: Vec::new(),
        };
        let code = run_oneshot(
            "hello",
            None,
            Some("openai"),
            &ToolsetsInput::None,
            &env,
            &NoEnvSet,
            &v,
            &empty_hooks(),
            &EchoRunner,
            &mut sink,
        );
        assert_eq!(code, 0);
        assert_eq!(String::from_utf8(sink.out).unwrap(), "RESP:hello\n");
    }

    #[test]
    fn oneshot_appends_newline_only_when_missing() {
        let env = MapEnv::new();
        let v = BuiltinValidator::default();

        struct NlRunner;
        impl AgentRunner for NlRunner {
            fn run(&self, _p: &AgentRunParams) -> String {
                "already\n".to_string()
            }
        }
        let mut sink = VecSink {
            out: Vec::new(),
            err: Vec::new(),
        };
        let code = run_oneshot(
            "x",
            None,
            None,
            &ToolsetsInput::None,
            &env,
            &NoEnvSet,
            &v,
            &empty_hooks(),
            &NlRunner,
            &mut sink,
        );
        assert_eq!(code, 0);
        assert_eq!(String::from_utf8(sink.out).unwrap(), "already\n");
    }

    #[test]
    fn oneshot_empty_response_prints_nothing() {
        let env = MapEnv::new();
        let v = BuiltinValidator::default();
        let mut sink = VecSink {
            out: Vec::new(),
            err: Vec::new(),
        };
        let code = run_oneshot(
            "x",
            None,
            None,
            &ToolsetsInput::None,
            &env,
            &NoEnvSet,
            &v,
            &empty_hooks(),
            &EmptyRunner,
            &mut sink,
        );
        assert_eq!(code, 0);
        assert!(sink.out.is_empty());
    }

    #[test]
    fn oneshot_invalid_toolsets_returns_2() {
        let env = MapEnv::new();
        let v = BuiltinValidator::default(); // nothing valid
        let mut sink = VecSink {
            out: Vec::new(),
            err: Vec::new(),
        };
        let code = run_oneshot(
            "x",
            None,
            None,
            &ToolsetsInput::Single("nope".into()),
            &env,
            &NoEnvSet,
            &v,
            &empty_hooks(),
            &EchoRunner,
            &mut sink,
        );
        assert_eq!(code, 2);
        let err = String::from_utf8(sink.err).unwrap();
        assert!(err.contains("did not contain any valid toolsets"));
    }

    #[test]
    fn run_agent_passes_explicit_toolsets() {
        let env = MapEnv::new();
        let hooks = empty_hooks();

        struct CapRunner(std::cell::RefCell<Option<AgentRunParams>>);
        impl AgentRunner for CapRunner {
            fn run(&self, params: &AgentRunParams) -> String {
                *self.0.borrow_mut() = Some(params.clone());
                String::new()
            }
        }
        let runner = CapRunner(std::cell::RefCell::new(None));
        let _ = run_agent(
            "p",
            Some("m"),
            Some("pr"),
            &ToolsetsInput::Single("files,web".into()),
            Some(vec!["files".into(), "web".into()]),
            false,
            &env,
            &hooks,
            &runner,
        );
        let captured = runner.0.borrow().clone().unwrap();
        assert_eq!(
            captured.toolsets,
            Some(vec!["files".into(), "web".into()])
        );
        assert_eq!(captured.effective_model, "m");
        assert_eq!(captured.effective_provider, Some("pr".into()));
    }
}
