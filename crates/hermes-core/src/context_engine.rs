use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ToolDefinition;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextEngineSessionStart {
    pub hermes_home: String,
    pub platform: String,
    pub model: String,
    pub provider: String,
}

pub trait ContextEngine: Send + Sync {
    fn name(&self) -> &str;

    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        Vec::new()
    }

    fn on_session_start(
        &self,
        _session_id: &str,
        _event: &ContextEngineSessionStart,
    ) -> Result<(), String> {
        Ok(())
    }

    fn on_session_end(&self, _session_id: &str, _messages: &[Value]) -> Result<(), String> {
        Ok(())
    }

    fn handle_tool_call(&self, name: &str, args: &Value, messages: &[Value]) -> String;
}
