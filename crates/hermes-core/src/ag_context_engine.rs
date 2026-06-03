//! Abstract base class for pluggable context engines.
//!
//! A context engine controls how conversation context is managed when
//! approaching the model's token limit. The built-in `ContextCompressor`
//! is the default implementation. Third-party engines (e.g. LCM) can
//! replace it via the plugin system or by being placed in the
//! `plugins/context_engine/<name>/` directory.
//!
//! Selection is config-driven: `context.engine` in config.yaml.
//! Default is `"compressor"` (the built-in). Only one engine is active.
//!
//! The engine is responsible for:
//!   - Deciding when compaction should fire
//!   - Performing compaction (summarization, DAG construction, etc.)
//!   - Optionally exposing tools the agent can call (e.g. `lcm_grep`)
//!   - Tracking token usage from API responses
//!
//! Lifecycle:
//!   1. Engine is instantiated and registered (plugin `register()` or default)
//!   2. `on_session_start()` called when a conversation begins
//!   3. `update_from_response()` called after each API response with usage data
//!   4. `should_compress()` checked after each turn
//!   5. `compress()` called when `should_compress()` returns `true`
//!   6. `on_session_end()` called at real session boundaries (CLI exit, /reset,
//!      gateway session expiry) — NOT per-turn
//!
//! This is a faithful, idiomatic Rust port of `agent/context_engine.py`.
//!
//! In Python the engine is an `ABC` that both declares an interface *and*
//! carries mutable instance state (token counts, thresholds, etc.) plus a
//! number of concrete default methods. Rust traits cannot hold fields, so the
//! mutable state is factored into [`ContextEngineState`], and the interface +
//! default behaviour live on the [`ContextEngine`] trait. The trait requires a
//! `state()` / `state_mut()` accessor so the default methods can read/write the
//! shared fields exactly like the Python base class does.

use serde_json::{json, Value};

/// Default fraction of the context window at which compaction fires.
pub const DEFAULT_THRESHOLD_PERCENT: f64 = 0.75;
/// Default number of leading messages protected from compaction.
pub const DEFAULT_PROTECT_FIRST_N: usize = 3;
/// Default number of trailing messages protected from compaction.
pub const DEFAULT_PROTECT_LAST_N: usize = 6;

/// Mutable token/compaction state carried by every context engine.
///
/// Mirrors the mutable class attributes on the Python `ContextEngine` ABC.
/// `run_agent.py` reads these directly for display/logging, so they are all
/// `pub`.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextEngineState {
    // -- Token state (read by run_agent.py for display/logging) --
    pub last_prompt_tokens: i64,
    pub last_completion_tokens: i64,
    pub last_total_tokens: i64,
    pub threshold_tokens: i64,
    pub context_length: i64,
    pub compression_count: i64,

    // -- Compaction parameters (read by run_agent.py for preflight) --
    pub threshold_percent: f64,
    pub protect_first_n: usize,
    pub protect_last_n: usize,
}

impl Default for ContextEngineState {
    fn default() -> Self {
        ContextEngineState {
            last_prompt_tokens: 0,
            last_completion_tokens: 0,
            last_total_tokens: 0,
            threshold_tokens: 0,
            context_length: 0,
            compression_count: 0,
            threshold_percent: DEFAULT_THRESHOLD_PERCENT,
            protect_first_n: DEFAULT_PROTECT_FIRST_N,
            protect_last_n: DEFAULT_PROTECT_LAST_N,
        }
    }
}

impl ContextEngineState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Optional keyword arguments passed to [`ContextEngine::on_session_start`].
///
/// In Python this is `**kwargs`; here it is an explicit, optional bag of the
/// fields the built-in callers actually pass (`hermes_home`, `platform`,
/// `model`, etc.). Engines that need more can ignore unknown keys.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionStartArgs {
    pub hermes_home: Option<String>,
    pub platform: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
}

/// Base interface all context engines must implement.
///
/// Required methods correspond to the Python `@abstractmethod`s; everything
/// else has a default implementation matching the Python base class.
pub trait ContextEngine {
    // -- State access -------------------------------------------------------
    //
    // Rust traits cannot carry fields, so concrete engines expose their
    // embedded [`ContextEngineState`] here. The default methods below read and
    // mutate state through these accessors, reproducing the Python base
    // class's behaviour where the methods touch `self.<field>` directly.

    fn state(&self) -> &ContextEngineState;
    fn state_mut(&mut self) -> &mut ContextEngineState;

    // -- Identity -----------------------------------------------------------

    /// Short identifier (e.g. `"compressor"`, `"lcm"`).
    fn name(&self) -> &str;

    // -- Core interface (abstract) ------------------------------------------

    /// Update tracked token usage from an API response.
    ///
    /// Called after every LLM call with the usage dict from the response.
    fn update_from_response(&mut self, usage: &Value);

    /// Return `true` if compaction should fire this turn.
    fn should_compress(&self, prompt_tokens: Option<i64>) -> bool;

    /// Compact the message list and return the new message list.
    ///
    /// This is the main entry point. The engine receives the full message
    /// list and returns a (possibly shorter) list that fits within the
    /// context budget. The implementation is free to summarize, build a
    /// DAG, or do anything else — as long as the returned list is a valid
    /// OpenAI-format message sequence.
    ///
    /// `focus_topic` is an optional topic string from manual
    /// `/compress <focus>`. Engines that support guided compression should
    /// prioritise preserving information related to this topic. Engines that
    /// don't support it may simply ignore the argument.
    fn compress(
        &mut self,
        messages: Vec<Value>,
        current_tokens: Option<i64>,
        focus_topic: Option<&str>,
    ) -> Vec<Value>;

    // -- Optional: pre-flight check -----------------------------------------

    /// Quick rough check before the API call (no real token count yet).
    ///
    /// Default returns `false` (skip pre-flight). Override if your engine
    /// can do a cheap estimate.
    fn should_compress_preflight(&self, _messages: &[Value]) -> bool {
        false
    }

    // -- Optional: manual /compress preflight -------------------------------

    /// Quick check: is there anything in `messages` that can be compacted?
    ///
    /// Used by the gateway `/compress` command as a preflight guard —
    /// returning `false` lets the gateway report "nothing to compress yet"
    /// without making an LLM call.
    ///
    /// Default returns `true` (always attempt). Engines with a cheap way to
    /// introspect their own head/tail boundaries should override this to
    /// return `false` when the transcript is still entirely protected.
    fn has_content_to_compress(&self, _messages: &[Value]) -> bool {
        true
    }

    // -- Optional: session lifecycle ----------------------------------------

    /// Called when a new conversation session begins.
    ///
    /// Use this to load persisted state (DAG, store) for the session.
    /// `args` may carry `hermes_home`, `platform`, `model`, etc.
    fn on_session_start(&mut self, _session_id: &str, _args: &SessionStartArgs) {}

    /// Called at real session boundaries (CLI exit, /reset, gateway expiry).
    ///
    /// Use this to flush state, close DB connections, etc.
    /// NOT called per-turn — only when the session truly ends.
    fn on_session_end(&mut self, _session_id: &str, _messages: &[Value]) {}

    /// Called on `/new` or `/reset`. Reset per-session state.
    ///
    /// Default resets `compression_count` and token tracking.
    fn on_session_reset(&mut self) {
        let s = self.state_mut();
        s.last_prompt_tokens = 0;
        s.last_completion_tokens = 0;
        s.last_total_tokens = 0;
        s.compression_count = 0;
    }

    // -- Optional: tools ----------------------------------------------------

    /// Return tool schemas this engine provides to the agent.
    ///
    /// Default returns an empty list (no tools). LCM would return schemas for
    /// `lcm_grep`, `lcm_describe`, `lcm_expand` here.
    fn get_tool_schemas(&self) -> Vec<Value> {
        Vec::new()
    }

    /// Handle a tool call from the agent.
    ///
    /// Only called for tool names returned by [`get_tool_schemas`]. Must
    /// return a JSON string.
    ///
    /// `messages` is the current in-memory message list (for live ingestion).
    ///
    /// [`get_tool_schemas`]: ContextEngine::get_tool_schemas
    fn handle_tool_call(&mut self, name: &str, _args: &Value, _messages: &[Value]) -> String {
        json!({ "error": format!("Unknown context engine tool: {name}") }).to_string()
    }

    // -- Optional: status / display -----------------------------------------

    /// Return status dict for display/logging.
    ///
    /// Default returns the standard fields `run_agent.py` expects.
    fn get_status(&self) -> Value {
        let s = self.state();
        let usage_percent: f64 = if s.context_length != 0 {
            let pct = (s.last_prompt_tokens as f64) / (s.context_length as f64) * 100.0;
            pct.min(100.0)
        } else {
            0.0
        };
        json!({
            "last_prompt_tokens": s.last_prompt_tokens,
            "threshold_tokens": s.threshold_tokens,
            "context_length": s.context_length,
            "usage_percent": usage_percent,
            "compression_count": s.compression_count,
        })
    }

    // -- Optional: model switch support -------------------------------------

    /// Called when the user switches models or on fallback activation.
    ///
    /// Default updates `context_length` and recalculates `threshold_tokens`
    /// from `threshold_percent`. Override if your engine needs more (e.g.
    /// recalculate DAG budgets, switch summary models).
    fn update_model(
        &mut self,
        _model: &str,
        context_length: i64,
        _base_url: &str,
        _api_key: &str,
        _provider: &str,
    ) {
        let threshold_percent = self.state().threshold_percent;
        let s = self.state_mut();
        s.context_length = context_length;
        // Python: int(context_length * threshold_percent) — truncation toward zero.
        s.threshold_tokens = (context_length as f64 * threshold_percent) as i64;
    }
}

/// Extract an integer token field from a usage dict, tolerating missing keys
/// and `null` (mirrors Python `usage.get(key, 0)` with int coercion).
pub fn usage_int(usage: &Value, key: &str) -> i64 {
    usage.get(key).and_then(Value::as_i64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal concrete engine reproducing the typical built-in compressor
    /// behaviour so the trait's default methods can be exercised.
    struct DummyEngine {
        state: ContextEngineState,
        last_handled: Option<String>,
    }

    impl DummyEngine {
        fn new() -> Self {
            DummyEngine {
                state: ContextEngineState::new(),
                last_handled: None,
            }
        }
    }

    impl ContextEngine for DummyEngine {
        fn state(&self) -> &ContextEngineState {
            &self.state
        }
        fn state_mut(&mut self) -> &mut ContextEngineState {
            &mut self.state
        }
        fn name(&self) -> &str {
            "dummy"
        }
        fn update_from_response(&mut self, usage: &Value) {
            self.state.last_prompt_tokens = usage_int(usage, "prompt_tokens");
            self.state.last_completion_tokens = usage_int(usage, "completion_tokens");
            self.state.last_total_tokens = usage_int(usage, "total_tokens");
        }
        fn should_compress(&self, prompt_tokens: Option<i64>) -> bool {
            let pt = prompt_tokens.unwrap_or(self.state.last_prompt_tokens);
            self.state.threshold_tokens > 0 && pt >= self.state.threshold_tokens
        }
        fn compress(
            &mut self,
            messages: Vec<Value>,
            _current_tokens: Option<i64>,
            _focus_topic: Option<&str>,
        ) -> Vec<Value> {
            self.state.compression_count += 1;
            messages
        }
    }

    #[test]
    fn defaults_match_python_base_class() {
        let s = ContextEngineState::default();
        assert_eq!(s.last_prompt_tokens, 0);
        assert_eq!(s.last_completion_tokens, 0);
        assert_eq!(s.last_total_tokens, 0);
        assert_eq!(s.threshold_tokens, 0);
        assert_eq!(s.context_length, 0);
        assert_eq!(s.compression_count, 0);
        assert_eq!(s.threshold_percent, 0.75);
        assert_eq!(s.protect_first_n, 3);
        assert_eq!(s.protect_last_n, 6);
    }

    #[test]
    fn update_model_recalculates_threshold_with_truncation() {
        let mut e = DummyEngine::new();
        e.update_model("gpt", 100_000, "", "", "");
        assert_eq!(e.state().context_length, 100_000);
        // 100000 * 0.75 = 75000
        assert_eq!(e.state().threshold_tokens, 75_000);

        // Non-round value: truncate toward zero like int() in Python.
        e.state_mut().threshold_percent = 0.333;
        e.update_model("gpt", 1001, "", "", "");
        // 1001 * 0.333 = 333.333 -> 333
        assert_eq!(e.state().threshold_tokens, 333);
    }

    #[test]
    fn default_optional_methods() {
        let e = DummyEngine::new();
        assert!(!e.should_compress_preflight(&[]));
        assert!(e.has_content_to_compress(&[]));
        assert!(e.get_tool_schemas().is_empty());
    }

    #[test]
    fn handle_unknown_tool_returns_error_json() {
        let mut e = DummyEngine::new();
        let out = e.handle_tool_call("lcm_grep", &json!({}), &[]);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["error"], "Unknown context engine tool: lcm_grep");
        // Default impl does not record handling.
        assert!(e.last_handled.is_none());
    }

    #[test]
    fn on_session_reset_clears_token_state_and_count() {
        let mut e = DummyEngine::new();
        e.state_mut().last_prompt_tokens = 10;
        e.state_mut().last_completion_tokens = 20;
        e.state_mut().last_total_tokens = 30;
        e.state_mut().compression_count = 4;
        // context_length / threshold should be preserved across reset.
        e.state_mut().context_length = 8192;
        e.state_mut().threshold_tokens = 6144;

        e.on_session_reset();

        assert_eq!(e.state().last_prompt_tokens, 0);
        assert_eq!(e.state().last_completion_tokens, 0);
        assert_eq!(e.state().last_total_tokens, 0);
        assert_eq!(e.state().compression_count, 0);
        assert_eq!(e.state().context_length, 8192);
        assert_eq!(e.state().threshold_tokens, 6144);
    }

    #[test]
    fn get_status_reports_capped_usage_percent() {
        let mut e = DummyEngine::new();
        e.state_mut().context_length = 1000;
        e.state_mut().last_prompt_tokens = 500;
        e.state_mut().threshold_tokens = 750;
        e.state_mut().compression_count = 2;

        let st = e.get_status();
        assert_eq!(st["last_prompt_tokens"], 500);
        assert_eq!(st["threshold_tokens"], 750);
        assert_eq!(st["context_length"], 1000);
        assert_eq!(st["compression_count"], 2);
        assert_eq!(st["usage_percent"].as_f64().unwrap(), 50.0);

        // Over 100% is clamped to 100.
        e.state_mut().last_prompt_tokens = 5000;
        let st = e.get_status();
        assert_eq!(st["usage_percent"].as_f64().unwrap(), 100.0);

        // Zero context_length yields 0 (no division by zero).
        e.state_mut().context_length = 0;
        let st = e.get_status();
        assert_eq!(st["usage_percent"].as_f64().unwrap(), 0.0);
    }

    #[test]
    fn update_from_response_and_should_compress() {
        let mut e = DummyEngine::new();
        e.update_model("m", 1000, "", "", ""); // threshold 750
        e.update_from_response(&json!({
            "prompt_tokens": 800,
            "completion_tokens": 50,
            "total_tokens": 850,
        }));
        assert_eq!(e.state().last_prompt_tokens, 800);
        assert_eq!(e.state().last_total_tokens, 850);
        assert!(e.should_compress(None));
        assert!(!e.should_compress(Some(10)));

        // Missing keys coerce to 0.
        e.update_from_response(&json!({}));
        assert_eq!(e.state().last_prompt_tokens, 0);
    }

    #[test]
    fn compress_increments_count_and_returns_messages() {
        let mut e = DummyEngine::new();
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        let out = e.compress(msgs.clone(), Some(100), Some("billing"));
        assert_eq!(out, msgs);
        assert_eq!(e.state().compression_count, 1);
    }
}
