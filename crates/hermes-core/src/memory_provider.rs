//! Provider abstraction base trait (port of agent/memory_provider.py). Not the manager.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Python **kwargs (hermes_home, platform, agent_context, ...).
pub type Kwargs = Map<String, Value>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError(pub String);
impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for ProviderError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryAction { Add, Replace, Remove }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryTarget { Memory, User }

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryWrite {
    pub action: MemoryAction,
    pub target: MemoryTarget,
    pub content: String,
    pub metadata: Map<String, Value>,
}

/// Serializes to Python's sparse config dict (unset optionals omitted).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfigField {
    pub key: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub secret: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_var: Option<String>,
}
fn is_false(b: &bool) -> bool { !*b }

pub trait MemoryProvider {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn initialize(&mut self, session_id: &str, kwargs: &Kwargs);
    fn system_prompt_block(&self) -> String { String::new() }
    fn prefetch(&self, _query: &str, _session_id: &str) -> String { String::new() }
    fn queue_prefetch(&mut self, _query: &str, _session_id: &str) {}
    fn sync_turn(&mut self, _user: &str, _assistant: &str, _session_id: &str) {}
    fn get_tool_schemas(&self) -> Vec<Value>;
    fn handle_tool_call(&mut self, tool_name: &str, _args: &Map<String, Value>, _kwargs: &Kwargs) -> Result<String, ProviderError> {
        Err(ProviderError(format!("Provider {} does not handle tool {tool_name}", self.name())))
    }
    fn shutdown(&mut self) {}
    fn on_turn_start(&mut self, _turn: i64, _message: &str, _kwargs: &Kwargs) {}
    fn on_session_end(&mut self, _messages: &[Value]) {}
    fn on_session_switch(&mut self, _new_session_id: &str, _parent_session_id: &str, _reset: bool, _kwargs: &Kwargs) {}
    fn on_pre_compress(&mut self, _messages: &[Value]) -> String { String::new() }
    fn on_delegation(&mut self, _task: &str, _result: &str, _child_session_id: &str, _kwargs: &Kwargs) {}
    fn get_config_schema(&self) -> Vec<ConfigField> { Vec::new() }
    fn save_config(&mut self, _values: &Map<String, Value>, _hermes_home: &str) {}
    fn on_memory_write(&mut self, _write: &MemoryWrite) {}
}