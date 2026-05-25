use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::browser::{
    browser_available, browser_back_schema, browser_cdp_available, browser_cdp_schema,
    browser_click_schema, browser_console_schema, browser_dialog_schema, browser_get_images_schema,
    browser_navigate_schema, browser_press_schema, browser_scroll_schema, browser_snapshot_schema,
    browser_type_schema, browser_vision_schema, handle_browser_back, handle_browser_cdp,
    handle_browser_click, handle_browser_console, handle_browser_dialog, handle_browser_get_images,
    handle_browser_navigate, handle_browser_press, handle_browser_scroll, handle_browser_snapshot,
    handle_browser_type, handle_browser_vision,
};
use crate::clarify::{clarify_available, clarify_schema, handle_clarify};
use crate::code_execution::{execute_code_available, execute_code_schema, handle_execute_code};
use crate::cronjob::{cronjob_available, cronjob_schema, handle_cronjob};
use crate::delegate::{DelegateTaskRequest, DelegateTaskSpec};
use crate::discord::{
    discord_admin_available, discord_admin_schema, discord_available, discord_core_schema,
    handle_discord, handle_discord_admin,
};
use crate::feishu::{
    feishu_available, feishu_doc_read_schema, feishu_drive_add_comment_schema,
    feishu_drive_list_comment_replies_schema, feishu_drive_list_comments_schema,
    feishu_drive_reply_comment_schema, handle_feishu_doc_read, handle_feishu_drive_add_comment,
    handle_feishu_drive_list_comment_replies, handle_feishu_drive_list_comments,
    handle_feishu_drive_reply_comment,
};
use crate::homeassistant::{
    ha_call_service_schema, ha_get_state_schema, ha_list_entities_schema, ha_list_services_schema,
    handle_ha_call_service, handle_ha_get_state, handle_ha_list_entities, handle_ha_list_services,
    homeassistant_available,
};
use crate::image_gen::{handle_image_generate, image_generate_available, image_generate_schema};
use crate::kanban::{
    handle_kanban_block, handle_kanban_comment, handle_kanban_complete, handle_kanban_create,
    handle_kanban_heartbeat, handle_kanban_link, handle_kanban_show, kanban_available,
    kanban_block_schema, kanban_comment_schema, kanban_complete_schema, kanban_create_schema,
    kanban_heartbeat_schema, kanban_link_schema, kanban_show_schema,
};
use crate::memory::{ExternalMemoryProviderRuntime, MemoryStore};
use crate::moa::{handle_mixture_of_agents, mixture_of_agents_available, mixture_of_agents_schema};
use crate::rl::{
    handle_rl_check_status, handle_rl_edit_config, handle_rl_get_current_config,
    handle_rl_get_results, handle_rl_list_environments, handle_rl_list_runs,
    handle_rl_select_environment, handle_rl_start_training, handle_rl_stop_training,
    handle_rl_test_inference, rl_available, rl_check_status_schema, rl_edit_config_schema,
    rl_get_current_config_schema, rl_get_results_schema, rl_list_environments_schema,
    rl_list_runs_schema, rl_select_environment_schema, rl_start_training_schema,
    rl_stop_training_schema, rl_test_inference_schema,
};
use crate::send_message::{handle_send_message, send_message_available, send_message_schema};
use crate::skills::{
    handle_skill_manage, handle_skill_view, handle_skills_list, skill_manage_schema,
    skill_view_schema, skills_available, skills_list_schema,
};
use crate::spotify::{
    handle_spotify_albums, handle_spotify_devices, handle_spotify_library, handle_spotify_playback,
    handle_spotify_playlists, handle_spotify_queue, handle_spotify_search, spotify_albums_schema,
    spotify_available, spotify_devices_schema, spotify_library_schema, spotify_playback_schema,
    spotify_playlists_schema, spotify_queue_schema, spotify_search_schema,
};
use crate::state::SessionStore;
use crate::terminal::{
    handle_process, handle_terminal, process_schema, terminal_available, terminal_schema,
};
use crate::tts::{handle_text_to_speech, text_to_speech_schema};
use crate::video::{handle_video_analyze, video_analyze_schema};
use crate::vision::{handle_vision_analyze, vision_analyze_schema};
use crate::web::{
    handle_web_extract, handle_web_search, web_extract_schema, web_search_schema,
    web_tools_available,
};
use crate::yuanbao::{
    handle_yb_query_group_info, handle_yb_query_group_members, handle_yb_search_sticker,
    handle_yb_send_dm, handle_yb_send_sticker, yb_query_group_info_schema,
    yb_query_group_members_schema, yb_search_sticker_schema, yb_send_dm_schema,
    yb_send_sticker_schema, yuanbao_available,
};

const DEFAULT_READ_LIMIT: i64 = 500;
const MAX_READ_LIMIT: i64 = 2_000;
const MAX_TOOL_RESULT_CHARS: usize = 100_000;
const DEFAULT_SEARCH_LIMIT: i64 = 50;
const MAX_SEARCH_LIMIT: i64 = 500;
const MAX_CONTEXT_LINES: i64 = 20;
const MAX_PATCH_OPERATIONS: usize = 128;
const VALID_TODO_STATUSES: &[&str] = &["pending", "in_progress", "completed", "cancelled"];

type ToolHandler = fn(&Value, &ToolRuntime) -> String;
type ToolCheck = fn() -> bool;
type ClarifyFn = dyn Fn(&str, Option<&[String]>) -> Result<String, String> + Send + Sync;
type ClarifyRequestFn = dyn Fn(&ClarifyRequest) + Send + Sync;
type DelegateFn = dyn Fn(DelegateTaskRequest, &ToolRuntime) -> Result<Value, String> + Send + Sync;
type ApprovalFn = dyn Fn(&crate::ApprovalRequest) -> Result<String, String> + Send + Sync;
type ApprovalRequestFn = dyn Fn(&crate::ApprovalRequest) + Send + Sync;
type ToolProgressFn = dyn Fn(&ToolProgressUpdate) + Send + Sync;
type StepFn = dyn Fn(&StepUpdate) + Send + Sync;
type DynamicToolDispatchFn = dyn Fn(&str, &Value, &ToolRuntime) -> Option<String> + Send + Sync;
type HookInvokeFn = dyn Fn(&str, &Value, &ToolRuntime) -> Vec<Value> + Send + Sync;

#[derive(Clone)]
struct ClarifyCallback(Arc<ClarifyFn>);

impl std::fmt::Debug for ClarifyCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ClarifyCallback(..)")
    }
}

#[derive(Clone)]
struct ClarifyRequestCallback(Arc<ClarifyRequestFn>);

impl std::fmt::Debug for ClarifyRequestCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ClarifyRequestCallback(..)")
    }
}

#[derive(Clone)]
struct DelegateCallback(Arc<DelegateFn>);

impl std::fmt::Debug for DelegateCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DelegateCallback(..)")
    }
}

#[derive(Clone)]
struct ApprovalCallback(Arc<ApprovalFn>);

impl std::fmt::Debug for ApprovalCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApprovalCallback(..)")
    }
}

#[derive(Clone)]
struct DynamicToolDispatchCallback(Arc<DynamicToolDispatchFn>);

impl std::fmt::Debug for DynamicToolDispatchCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DynamicToolDispatchCallback(..)")
    }
}

#[derive(Clone)]
struct ApprovalRequestCallback(Arc<ApprovalRequestFn>);

impl std::fmt::Debug for ApprovalRequestCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApprovalRequestCallback(..)")
    }
}

#[derive(Clone)]
struct ContextEngineHandle(Arc<dyn crate::ContextEngine>);

impl std::fmt::Debug for ContextEngineHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ContextEngineHandle(..)")
    }
}

#[derive(Clone)]
struct ToolProgressCallback(Arc<ToolProgressFn>);

impl std::fmt::Debug for ToolProgressCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ToolProgressCallback(..)")
    }
}

#[derive(Clone)]
struct StepCallback(Arc<StepFn>);

impl std::fmt::Debug for StepCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StepCallback(..)")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolProgressUpdate {
    pub event_type: String,
    pub function_name: Option<String>,
    pub preview: Option<String>,
    pub function_args: Option<Value>,
    pub duration_ms: Option<u64>,
    pub is_error: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepToolRecord {
    pub name: String,
    pub result: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepUpdate {
    pub iteration: u64,
    pub prev_tools: Vec<StepToolRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClarifyRequest {
    pub question: String,
    pub choices: Option<Vec<String>>,
}

#[derive(Clone)]
struct HookInvokeCallback(Arc<HookInvokeFn>);

impl std::fmt::Debug for HookInvokeCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HookInvokeCallback(..)")
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub toolset: String,
    pub description: String,
    pub emoji: String,
    pub schema: Value,
}

impl ToolDefinition {
    pub fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": self.schema,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolsetInfo {
    pub name: String,
    pub description: String,
    pub direct_tools: Vec<String>,
    pub includes: Vec<String>,
    pub resolved_tools: Vec<String>,
    pub implemented_tools: Vec<String>,
    pub available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TodoItem {
    id: String,
    content: String,
    status: String,
}

#[derive(Debug, Default)]
struct TodoStore {
    items: Vec<TodoItem>,
}

impl TodoStore {
    fn read(&self) -> Vec<TodoItem> {
        self.items.clone()
    }

    fn replace(&mut self, items: Vec<TodoItem>) {
        self.items = items;
    }

    fn write(&mut self, todos: &[Value], merge: bool) -> Vec<TodoItem> {
        let deduped = Self::dedupe_by_id(todos);
        if !merge {
            self.items = deduped.iter().map(Self::validate).collect();
            return self.read();
        }

        let mut existing = self
            .items
            .iter()
            .cloned()
            .map(|item| (item.id.clone(), item))
            .collect::<HashMap<_, _>>();

        for item in &deduped {
            let item_id = item
                .get("id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            let Some(item_id) = item_id else {
                continue;
            };

            if let Some(current) = existing.get_mut(&item_id) {
                if let Some(content) = item
                    .get("content")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    current.content = content.to_string();
                }
                if let Some(status) = item
                    .get("status")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .map(str::to_ascii_lowercase)
                    .filter(|value| VALID_TODO_STATUSES.contains(&value.as_str()))
                {
                    current.status = status;
                }
            } else {
                let validated = Self::validate(item);
                existing.insert(validated.id.clone(), validated.clone());
                self.items.push(validated);
            }
        }

        let mut rebuilt = Vec::new();
        let mut seen = HashSet::new();
        for item in &self.items {
            let current = existing
                .get(&item.id)
                .cloned()
                .unwrap_or_else(|| item.clone());
            if seen.insert(current.id.clone()) {
                rebuilt.push(current);
            }
        }
        self.items = rebuilt;
        self.read()
    }

    fn hydrate_from_messages(&mut self, messages: &[crate::MessageRecord]) {
        for message in messages.iter().rev() {
            if message.role != "tool" || message.tool_name.as_deref() != Some("todo") {
                continue;
            }
            let Some(content) = message.content.as_ref() else {
                continue;
            };
            let Some(payload) = parse_embedded_json(content) else {
                continue;
            };
            let Some(todos) = payload.get("todos").and_then(Value::as_array) else {
                continue;
            };
            self.replace(todos.iter().map(Self::validate).collect());
            break;
        }
    }

    fn validate(item: &Value) -> TodoItem {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("?")
            .to_string();
        let content = item
            .get("content")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("(no description)")
            .to_string();
        let status = item
            .get("status")
            .and_then(Value::as_str)
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .filter(|value| VALID_TODO_STATUSES.contains(&value.as_str()))
            .unwrap_or_else(|| "pending".to_string());
        TodoItem {
            id,
            content,
            status,
        }
    }

    fn dedupe_by_id(todos: &[Value]) -> Vec<Value> {
        let mut last_index = HashMap::new();
        for (index, item) in todos.iter().enumerate() {
            let item_id = item
                .get("id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("?")
                .to_string();
            last_index.insert(item_id, index);
        }
        let mut keep = last_index.into_values().collect::<Vec<_>>();
        keep.sort_unstable();
        keep.into_iter()
            .filter_map(|index| todos.get(index).cloned())
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ToolRuntime {
    cwd: PathBuf,
    hermes_home: PathBuf,
    current_session_id: Option<String>,
    todo_store: Arc<Mutex<TodoStore>>,
    memory_store: Option<Arc<Mutex<MemoryStore>>>,
    external_memory_provider: Option<Arc<Mutex<ExternalMemoryProviderRuntime>>>,
    available_tool_names: Option<BTreeSet<String>>,
    system_prompt_additions: Vec<String>,
    dynamic_tool_definitions: Arc<BTreeMap<String, ToolDefinition>>,
    dynamic_toolsets: Arc<BTreeMap<String, Vec<String>>>,
    clarify_callback: Option<ClarifyCallback>,
    clarify_request_callback: Option<ClarifyRequestCallback>,
    delegate_callback: Option<DelegateCallback>,
    approvals: Option<Arc<Mutex<crate::ApprovalManager>>>,
    approval_callback: Option<ApprovalCallback>,
    approval_request_callback: Option<ApprovalRequestCallback>,
    context_engine: Option<ContextEngineHandle>,
    context_engine_enabled: bool,
    context_session_started: Arc<Mutex<Option<String>>>,
    checkpoints: Option<Arc<Mutex<crate::checkpoints::CheckpointManager>>>,
    shell_hooks: Option<Arc<crate::shell_hooks::ShellHookRunner>>,
    tool_progress_callback: Option<ToolProgressCallback>,
    step_callback: Option<StepCallback>,
    dynamic_tool_dispatch: Option<DynamicToolDispatchCallback>,
    hook_invoke_callback: Option<HookInvokeCallback>,
}

impl ToolRuntime {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            hermes_home: default_hermes_home(),
            current_session_id: None,
            todo_store: Arc::new(Mutex::new(TodoStore::default())),
            memory_store: None,
            external_memory_provider: None,
            available_tool_names: None,
            system_prompt_additions: Vec::new(),
            dynamic_tool_definitions: Arc::new(BTreeMap::new()),
            dynamic_toolsets: Arc::new(BTreeMap::new()),
            clarify_callback: None,
            clarify_request_callback: None,
            delegate_callback: None,
            approvals: None,
            approval_callback: None,
            approval_request_callback: None,
            context_engine: None,
            context_engine_enabled: false,
            context_session_started: Arc::new(Mutex::new(None)),
            checkpoints: None,
            shell_hooks: None,
            tool_progress_callback: None,
            step_callback: None,
            dynamic_tool_dispatch: None,
            hook_invoke_callback: None,
        }
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn hermes_home(&self) -> &Path {
        &self.hermes_home
    }

    pub fn with_hermes_home(mut self, value: impl Into<PathBuf>) -> Self {
        self.hermes_home = value.into();
        self
    }

    pub fn current_session_id(&self) -> Option<&str> {
        self.current_session_id.as_deref()
    }

    pub fn current_session_id_or_empty(&self) -> &str {
        self.current_session_id.as_deref().unwrap_or("")
    }

    pub fn with_current_session_id(mut self, value: Option<String>) -> Self {
        self.current_session_id = value.and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });
        self
    }

    pub fn hydrate_todo_from_messages(&self, messages: &[crate::MessageRecord]) {
        if let Ok(mut store) = self.todo_store.lock() {
            store.hydrate_from_messages(messages);
        }
    }

    pub fn with_memory_store(mut self, value: Option<MemoryStore>) -> Self {
        self.memory_store = value.map(|store| Arc::new(Mutex::new(store)));
        self
    }

    pub fn with_external_memory_provider(
        mut self,
        value: Option<ExternalMemoryProviderRuntime>,
    ) -> Self {
        self.external_memory_provider = value.map(|provider| Arc::new(Mutex::new(provider)));
        self
    }

    pub fn load_memory_store(&mut self, config: &crate::MemoryConfig) -> Result<(), String> {
        if !config.any_enabled() {
            self.memory_store = None;
            return Ok(());
        }

        let mut store = MemoryStore::new(config);
        store.load_from_disk(&self.hermes_home)?;
        self.memory_store = Some(Arc::new(Mutex::new(store)));
        Ok(())
    }

    pub fn memory_system_prompt_block(&self, target: &str) -> Option<String> {
        let store = self.memory_store.as_ref()?;
        let store = store.lock().ok()?;
        store.format_for_system_prompt(target)
    }

    pub fn external_memory_system_prompt_block(&self) -> Option<String> {
        let provider = self.external_memory_provider.as_ref()?;
        let provider = provider.lock().ok()?;
        let block = provider.system_prompt_block().trim();
        (!block.is_empty()).then(|| block.to_string())
    }

    pub fn external_memory_tool_definitions(&self) -> Vec<ToolDefinition> {
        let Some(provider) = self.external_memory_provider.as_ref() else {
            return Vec::new();
        };
        let Ok(provider) = provider.lock() else {
            return Vec::new();
        };
        provider
            .tool_schemas()
            .into_iter()
            .filter_map(|schema| {
                let name = schema
                    .get("name")
                    .and_then(Value::as_str)?
                    .trim()
                    .to_string();
                if name.is_empty() {
                    return None;
                }
                let description = schema
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                Some(ToolDefinition {
                    name,
                    toolset: "memory".to_string(),
                    description,
                    emoji: "x".to_string(),
                    schema,
                })
            })
            .collect()
    }

    pub fn external_memory_has_tool(&self, name: &str) -> bool {
        let Some(provider) = self.external_memory_provider.as_ref() else {
            return false;
        };
        let Ok(provider) = provider.lock() else {
            return false;
        };
        provider.has_tool(name)
    }

    pub fn external_memory_handle_tool_call(
        &self,
        name: &str,
        args: &Value,
    ) -> Result<String, String> {
        let provider = self
            .external_memory_provider
            .as_ref()
            .ok_or_else(|| "external memory provider is not configured".to_string())?;
        let mut provider = provider
            .lock()
            .map_err(|_| "external memory provider lock poisoned".to_string())?;
        provider.handle_tool_call(name, args, self.current_session_id_or_empty())
    }

    pub fn external_memory_on_turn_start(
        &self,
        turn_number: u64,
        message: &str,
    ) -> Result<(), String> {
        let Some(provider) = self.external_memory_provider.as_ref() else {
            return Ok(());
        };
        let mut provider = provider
            .lock()
            .map_err(|_| "external memory provider lock poisoned".to_string())?;
        provider.on_turn_start(turn_number, message, json!({"platform": "rust-agent"}))
    }

    pub fn external_memory_prefetch(&self, query: &str) -> Result<String, String> {
        let Some(provider) = self.external_memory_provider.as_ref() else {
            return Ok(String::new());
        };
        let mut provider = provider
            .lock()
            .map_err(|_| "external memory provider lock poisoned".to_string())?;
        provider.prefetch(query, self.current_session_id_or_empty())
    }

    pub fn external_memory_queue_prefetch(&self, query: &str) -> Result<(), String> {
        let Some(provider) = self.external_memory_provider.as_ref() else {
            return Ok(());
        };
        let mut provider = provider
            .lock()
            .map_err(|_| "external memory provider lock poisoned".to_string())?;
        provider.queue_prefetch(query, self.current_session_id_or_empty())
    }

    pub fn external_memory_sync_turn(
        &self,
        user_content: &str,
        assistant_content: &str,
    ) -> Result<(), String> {
        let Some(provider) = self.external_memory_provider.as_ref() else {
            return Ok(());
        };
        let mut provider = provider
            .lock()
            .map_err(|_| "external memory provider lock poisoned".to_string())?;
        provider.sync_turn(
            user_content,
            assistant_content,
            self.current_session_id_or_empty(),
        )
    }

    pub fn external_memory_on_memory_write(
        &self,
        action: &str,
        target: &str,
        content: &str,
        metadata: Value,
    ) -> Result<(), String> {
        let Some(provider) = self.external_memory_provider.as_ref() else {
            return Ok(());
        };
        let mut provider = provider
            .lock()
            .map_err(|_| "external memory provider lock poisoned".to_string())?;
        provider.on_memory_write(action, target, content, metadata)
    }

    pub fn external_memory_shutdown(&self) -> Result<(), String> {
        let Some(provider) = self.external_memory_provider.as_ref() else {
            return Ok(());
        };
        let mut provider = provider
            .lock()
            .map_err(|_| "external memory provider lock poisoned".to_string())?;
        provider.shutdown()
    }

    pub fn with_clarify_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, Option<&[String]>) -> Result<String, String> + Send + Sync + 'static,
    {
        self.clarify_callback = Some(ClarifyCallback(Arc::new(callback)));
        self
    }

    pub fn with_clarify_request_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&ClarifyRequest) + Send + Sync + 'static,
    {
        self.clarify_request_callback = Some(ClarifyRequestCallback(Arc::new(callback)));
        self
    }

    pub fn emit_clarify_request(&self, request: &ClarifyRequest) {
        if let Some(callback) = &self.clarify_request_callback {
            (callback.0)(request);
        }
    }

    pub fn clarify(&self, question: &str, choices: Option<&[String]>) -> Result<String, String> {
        let Some(callback) = &self.clarify_callback else {
            return Err("Clarify tool is not available in this execution context.".to_string());
        };
        self.emit_clarify_request(&ClarifyRequest {
            question: question.to_string(),
            choices: choices.map(|items| items.to_vec()),
        });
        (callback.0)(question, choices)
    }

    pub fn with_delegate_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(DelegateTaskRequest, &ToolRuntime) -> Result<Value, String> + Send + Sync + 'static,
    {
        self.delegate_callback = Some(DelegateCallback(Arc::new(callback)));
        self
    }

    pub fn with_approval_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&crate::ApprovalRequest) -> Result<String, String> + Send + Sync + 'static,
    {
        self.approval_callback = Some(ApprovalCallback(Arc::new(callback)));
        self
    }

    pub fn with_approval_request_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&crate::ApprovalRequest) + Send + Sync + 'static,
    {
        self.approval_request_callback = Some(ApprovalRequestCallback(Arc::new(callback)));
        self
    }

    pub fn emit_approval_request(&self, request: &crate::ApprovalRequest) {
        if let Some(callback) = &self.approval_request_callback {
            (callback.0)(request);
        }
    }

    pub fn with_context_engine<E>(mut self, engine: E) -> Self
    where
        E: crate::ContextEngine + 'static,
    {
        self.context_engine = Some(ContextEngineHandle(Arc::new(engine)));
        self.context_engine_enabled = false;
        if let Ok(mut started) = self.context_session_started.lock() {
            *started = None;
        }
        self
    }

    pub fn with_tool_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&ToolProgressUpdate) + Send + Sync + 'static,
    {
        self.tool_progress_callback = Some(ToolProgressCallback(Arc::new(callback)));
        self
    }

    pub fn emit_tool_progress(&self, update: ToolProgressUpdate) {
        if let Some(callback) = &self.tool_progress_callback {
            (callback.0)(&update);
        }
    }

    pub fn with_step_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&StepUpdate) + Send + Sync + 'static,
    {
        self.step_callback = Some(StepCallback(Arc::new(callback)));
        self
    }

    pub fn emit_step(&self, update: StepUpdate) {
        if let Some(callback) = &self.step_callback {
            (callback.0)(&update);
        }
    }

    pub fn with_shell_hooks(mut self, value: Option<crate::shell_hooks::ShellHookRunner>) -> Self {
        self.shell_hooks = value.map(Arc::new);
        self
    }

    pub fn with_approvals(mut self, value: Option<crate::ApprovalManager>) -> Self {
        self.approvals = value.map(|manager| Arc::new(Mutex::new(manager)));
        self
    }

    pub fn load_approvals(&mut self, loaded: &crate::LoadedConfig) {
        if self.approvals.is_none() {
            self.approvals = Some(Arc::new(Mutex::new(
                crate::ApprovalManager::load_for_runtime(&self.hermes_home, loaded),
            )));
        }
    }

    pub fn load_context_engine(&mut self, loaded: &crate::LoadedConfig) {
        let configured = loaded
            .cfg_get(&["context", "engine"])
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("compressor");

        self.context_engine_enabled = false;
        if configured.eq_ignore_ascii_case("compressor") {
            if let Ok(mut started) = self.context_session_started.lock() {
                *started = None;
            }
            return;
        }

        let Some(engine) = &self.context_engine else {
            log::warn!(
                "context engine '{configured}' requested but no Rust context engine is registered"
            );
            return;
        };

        if engine.0.name().eq_ignore_ascii_case(configured) {
            self.context_engine_enabled = true;
            return;
        }

        if let Ok(mut started) = self.context_session_started.lock() {
            *started = None;
        }
        log::warn!(
            "context engine '{configured}' requested but Rust runtime only has '{}'",
            engine.0.name()
        );
    }

    pub fn with_checkpoints(
        mut self,
        value: Option<crate::checkpoints::CheckpointManager>,
    ) -> Self {
        self.checkpoints = value.map(|manager| Arc::new(Mutex::new(manager)));
        self
    }

    pub fn load_checkpoints(&mut self, loaded: &crate::LoadedConfig) {
        if self.checkpoints.is_none() {
            self.checkpoints =
                crate::checkpoints::CheckpointManager::load_for_runtime(&self.hermes_home, loaded)
                    .map(|manager| Arc::new(Mutex::new(manager)));
        }
    }

    pub fn begin_tool_turn(&self) {
        if let Some(checkpoints) = &self.checkpoints
            && let Ok(mut manager) = checkpoints.lock()
        {
            manager.new_turn();
        }
    }

    pub fn maybe_checkpoint_before_tool(&self, name: &str, args: &Value) {
        let Some(checkpoints) = &self.checkpoints else {
            return;
        };

        let checkpoint = match name {
            "write_file" | "patch" => {
                let Some(path) = args.get("path").and_then(Value::as_str).map(str::trim) else {
                    return;
                };
                if path.is_empty() {
                    return;
                }
                let Ok(resolved) = self.resolve_path(path) else {
                    return;
                };
                (resolved, format!("before {name}"))
            }
            "terminal" => {
                let Some(command) = args.get("command").and_then(Value::as_str).map(str::trim)
                else {
                    return;
                };
                if !crate::checkpoints::is_destructive_command(command) {
                    return;
                }
                let workdir = args
                    .get("workdir")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .and_then(|value| self.resolve_path(value).ok())
                    .unwrap_or_else(|| self.cwd().to_path_buf());
                (
                    workdir,
                    format!(
                        "before terminal: {}",
                        command.chars().take(60).collect::<String>()
                    ),
                )
            }
            _ => return,
        };

        if let Ok(mut manager) = checkpoints.lock() {
            let workdir = manager.get_working_dir_for_path(&checkpoint.0);
            let _ = manager.ensure_checkpoint(&workdir, &checkpoint.1);
        }
    }

    pub fn check_terminal_command_approval(&self, command: &str) -> crate::ApprovalCheckResult {
        let Some(approvals) = &self.approvals else {
            return crate::ApprovalCheckResult::Approved;
        };
        let callback = self.approval_callback.as_ref().cloned();
        let request_callback = self.approval_request_callback.as_ref().cloned();
        let Ok(mut manager) = approvals.lock() else {
            return crate::ApprovalCheckResult::Blocked {
                message: "BLOCKED: Approval manager is unavailable.".to_string(),
            };
        };
        let wrapped_callback = callback.as_ref().map(|callback| {
            let callback = callback.clone();
            let request_callback = request_callback.clone();
            move |request: &crate::ApprovalRequest| {
                if let Some(observer) = &request_callback {
                    (observer.0)(request);
                }
                (callback.0)(request)
            }
        });
        manager.check_command(
            command,
            "local",
            self.current_session_id(),
            wrapped_callback.as_ref().map(|callback| callback as _),
        )
    }

    pub fn load_shell_hooks(
        &mut self,
        context: &crate::HermesContext,
        loaded: &crate::LoadedConfig,
    ) {
        if self.shell_hooks.is_none() {
            self.shell_hooks =
                crate::shell_hooks::ShellHookRunner::load_for_runtime(context, loaded)
                    .map(Arc::new);
        }
    }

    pub fn delegate(&self, request: DelegateTaskRequest) -> Result<Value, String> {
        let Some(callback) = &self.delegate_callback else {
            return Err("delegate_task requires a parent agent context.".to_string());
        };
        (callback.0)(request, self)
    }

    pub fn with_available_tool_names<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let names = values
            .into_iter()
            .map(Into::into)
            .map(|value: String| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect::<BTreeSet<_>>();
        self.available_tool_names = Some(names);
        self
    }

    pub fn available_tool_names(&self) -> Option<&BTreeSet<String>> {
        self.available_tool_names.as_ref()
    }

    pub fn with_system_prompt_addition(mut self, value: impl Into<String>) -> Self {
        let value = value.into();
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            self.system_prompt_additions.push(trimmed.to_string());
        }
        self
    }

    pub fn system_prompt_additions(&self) -> &[String] {
        &self.system_prompt_additions
    }

    pub fn context_tool_definitions(&self) -> Vec<ToolDefinition> {
        if !self.context_engine_enabled {
            return Vec::new();
        }

        self.context_engine
            .as_ref()
            .map(|engine| {
                engine
                    .0
                    .tool_definitions()
                    .into_iter()
                    .filter(|definition| {
                        !definition.name.trim().is_empty() && definition.schema.is_object()
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn context_tool_schema(&self, name: &str) -> Option<Value> {
        self.context_tool_definitions()
            .into_iter()
            .find(|definition| definition.name == name)
            .map(|definition| definition.schema)
    }

    fn has_context_tool(&self, name: &str) -> bool {
        self.context_tool_schema(name).is_some()
    }

    fn dispatch_context_tool(
        &self,
        name: &str,
        args: &Value,
        messages: &[Value],
    ) -> Option<String> {
        if !self.context_engine_enabled || !self.has_context_tool(name) {
            return None;
        }
        self.context_engine
            .as_ref()
            .map(|engine| engine.0.handle_tool_call(name, args, messages))
    }

    pub fn start_context_engine_session(&self, session_id: &str, model: &str, provider: &str) {
        if !self.context_engine_enabled {
            return;
        }
        let session_id = session_id.trim();
        if session_id.is_empty() {
            return;
        }

        let Some(engine) = &self.context_engine else {
            return;
        };

        let Ok(mut started) = self.context_session_started.lock() else {
            return;
        };
        if started.as_deref() == Some(session_id) {
            return;
        }

        let event = crate::ContextEngineSessionStart {
            hermes_home: self.hermes_home.display().to_string(),
            platform: std::env::var("HERMES_SESSION_PLATFORM")
                .ok()
                .and_then(|value| {
                    let trimmed = value.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_string())
                })
                .unwrap_or_else(|| "cli".to_string()),
            model: model.to_string(),
            provider: provider.to_string(),
        };
        match engine.0.on_session_start(session_id, &event) {
            Ok(()) => *started = Some(session_id.to_string()),
            Err(error) => log::warn!("context engine session start failed: {error}"),
        }
    }

    pub fn with_dynamic_tools(
        mut self,
        definitions: Vec<ToolDefinition>,
        toolsets: BTreeMap<String, Vec<String>>,
    ) -> Self {
        self.dynamic_tool_definitions = Arc::new(
            definitions
                .into_iter()
                .map(|definition| (definition.name.clone(), definition))
                .collect(),
        );
        self.dynamic_toolsets = Arc::new(
            toolsets
                .into_iter()
                .map(|(name, tools)| {
                    let normalized = tools
                        .into_iter()
                        .map(|tool| tool.trim().to_string())
                        .filter(|tool| !tool.is_empty())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>();
                    (name, normalized)
                })
                .collect(),
        );
        self
    }

    pub fn dynamic_tool_definition(&self, name: &str) -> Option<ToolDefinition> {
        self.dynamic_tool_definitions.get(name).cloned()
    }

    pub fn dynamic_toolset_names(&self) -> Vec<String> {
        self.dynamic_toolsets.keys().cloned().collect()
    }

    pub fn resolve_dynamic_toolset(&self, name: &str) -> Vec<String> {
        self.dynamic_toolsets.get(name).cloned().unwrap_or_default()
    }

    pub fn with_dynamic_runtime_from(mut self, other: &Self) -> Self {
        self.dynamic_tool_definitions = Arc::clone(&other.dynamic_tool_definitions);
        self.dynamic_toolsets = Arc::clone(&other.dynamic_toolsets);
        self.dynamic_tool_dispatch = other.dynamic_tool_dispatch.clone();
        self.hook_invoke_callback = other.hook_invoke_callback.clone();
        self
    }

    pub fn resolve_path(&self, raw: &str) -> Result<PathBuf, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("path must not be empty".to_string());
        }

        let path = if trimmed == "~" {
            dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"))
        } else if let Some(rest) = trimmed.strip_prefix("~/") {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("~"))
                .join(rest)
        } else {
            PathBuf::from(trimmed)
        };

        if path.is_absolute() {
            Ok(path)
        } else {
            Ok(self.cwd.join(path))
        }
    }

    pub fn with_dynamic_tool_dispatch<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, &Value, &ToolRuntime) -> Option<String> + Send + Sync + 'static,
    {
        self.dynamic_tool_dispatch = Some(DynamicToolDispatchCallback(Arc::new(callback)));
        self
    }

    pub fn dispatch_dynamic_tool(&self, name: &str, args: &Value) -> Option<String> {
        self.dynamic_tool_dispatch
            .as_ref()
            .and_then(|callback| (callback.0)(name, args, self))
    }

    pub fn with_hook_invoke_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, &Value, &ToolRuntime) -> Vec<Value> + Send + Sync + 'static,
    {
        self.hook_invoke_callback = Some(HookInvokeCallback(Arc::new(callback)));
        self
    }

    pub fn invoke_hook(&self, hook_name: &str, payload: &Value) -> Vec<Value> {
        self.hook_invoke_callback
            .as_ref()
            .map(|callback| (callback.0)(hook_name, payload, self))
            .unwrap_or_default()
    }
}

impl Default for ToolRuntime {
    fn default() -> Self {
        Self::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
}

struct ToolEntry {
    name: &'static str,
    toolset: &'static str,
    description: &'static str,
    emoji: &'static str,
    schema_fn: fn() -> Value,
    handler: ToolHandler,
    check_fn: Option<ToolCheck>,
}

struct ToolsetEntry {
    name: &'static str,
    description: &'static str,
    tools: &'static [&'static str],
    includes: &'static [&'static str],
}

const FILE_TOOLS: &[&str] = &["read_file", "write_file", "patch", "search_files"];
const BROWSER_TOOLS: &[&str] = &[
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_back",
    "browser_press",
    "browser_get_images",
    "browser_console",
    "browser_cdp",
    "browser_dialog",
    "browser_vision",
];
const CLARIFY_TOOLS: &[&str] = &["clarify"];
const CODE_EXECUTION_TOOLS: &[&str] = &["execute_code"];
const CRONJOB_TOOLS: &[&str] = &["cronjob"];
const DELEGATION_TOOLS: &[&str] = &["delegate_task"];
const DISCORD_TOOLS: &[&str] = &["discord"];
const DISCORD_ADMIN_TOOLS: &[&str] = &["discord_admin"];
const FEISHU_DOC_TOOLS: &[&str] = &["feishu_doc_read"];
const FEISHU_DRIVE_TOOLS: &[&str] = &[
    "feishu_drive_list_comments",
    "feishu_drive_list_comment_replies",
    "feishu_drive_reply_comment",
    "feishu_drive_add_comment",
];
const HOMEASSISTANT_TOOLS: &[&str] = &[
    "ha_list_entities",
    "ha_get_state",
    "ha_list_services",
    "ha_call_service",
];
const IMAGE_GEN_TOOLS: &[&str] = &["image_generate"];
const KANBAN_TOOLS: &[&str] = &[
    "kanban_show",
    "kanban_complete",
    "kanban_block",
    "kanban_heartbeat",
    "kanban_comment",
    "kanban_create",
    "kanban_link",
];
const MEMORY_TOOLS: &[&str] = &["memory"];
const MESSAGING_TOOLS: &[&str] = &["send_message"];
const MOA_TOOLS: &[&str] = &["mixture_of_agents"];
const RL_TOOLS: &[&str] = &[
    "rl_list_environments",
    "rl_select_environment",
    "rl_get_current_config",
    "rl_edit_config",
    "rl_start_training",
    "rl_check_status",
    "rl_stop_training",
    "rl_get_results",
    "rl_list_runs",
    "rl_test_inference",
];
const SESSION_SEARCH_TOOLS: &[&str] = &["session_search"];
const SKILLS_TOOLS: &[&str] = &["skills_list", "skill_view", "skill_manage"];
const SPOTIFY_TOOLS: &[&str] = &[
    "spotify_playback",
    "spotify_devices",
    "spotify_queue",
    "spotify_search",
    "spotify_playlists",
    "spotify_albums",
    "spotify_library",
];
const TERMINAL_TOOLS: &[&str] = &["terminal", "process"];
const TTS_TOOLS: &[&str] = &["text_to_speech"];
const TODO_TOOLS: &[&str] = &["todo"];
const VIDEO_TOOLS: &[&str] = &["video_analyze"];
const VISION_TOOLS: &[&str] = &["vision_analyze"];
const WEB_TOOLS: &[&str] = &["web_search", "web_extract"];
const YUANBAO_TOOLS: &[&str] = &[
    "yb_query_group_info",
    "yb_query_group_members",
    "yb_send_dm",
    "yb_search_sticker",
    "yb_send_sticker",
];

const TOOLSETS: &[ToolsetEntry] = &[
    ToolsetEntry {
        name: "file",
        description: "File manipulation tools: read, write, and search text files",
        tools: FILE_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "browser",
        description: "Drive a live local browser session through agent-browser",
        tools: BROWSER_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "terminal",
        description: "Local shell execution and managed background processes",
        tools: TERMINAL_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "clarify",
        description: "Ask the user clarifying questions and collect structured answers",
        tools: CLARIFY_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "code_execution",
        description: "Sandboxed Python execution with RPC access to a limited tool subset",
        tools: CODE_EXECUTION_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "cronjob",
        description: "Create, inspect, and mutate scheduled jobs stored under Hermes home",
        tools: CRONJOB_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "delegation",
        description: "Spawn subagents for isolated delegated tasks",
        tools: DELEGATION_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "discord",
        description: "Discord read and thread-participation tools",
        tools: DISCORD_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "discord_admin",
        description: "Discord server management tools",
        tools: DISCORD_ADMIN_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "feishu_doc",
        description: "Read Feishu or Lark document content",
        tools: FEISHU_DOC_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "feishu_drive",
        description: "List and reply to Feishu or Lark document comments",
        tools: FEISHU_DRIVE_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "homeassistant",
        description: "Home Assistant smart home control and monitoring",
        tools: HOMEASSISTANT_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "memory",
        description: "Persistent built-in memory and user profile storage",
        tools: MEMORY_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "messaging",
        description: "Cross-platform outbound messaging via configured platform credentials",
        tools: MESSAGING_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "moa",
        description: "Mixture-of-agents collaborative reasoning with multiple frontier models",
        tools: MOA_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "rl",
        description: "RL environment discovery, config editing, run management, and process-mode inference testing",
        tools: RL_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "kanban",
        description: "Structured kanban worker and orchestrator coordination tools",
        tools: KANBAN_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "image_gen",
        description: "Generate images from prompts through the configured backend",
        tools: IMAGE_GEN_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "session_search",
        description: "Search or browse prior Rust agent sessions",
        tools: SESSION_SEARCH_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "skills",
        description: "List and inspect local skill documents",
        tools: SKILLS_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "spotify",
        description: "Native Spotify playback, search, playlist, album, and library tools",
        tools: SPOTIFY_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "tts",
        description: "Convert text to speech audio through the configured backend",
        tools: TTS_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "todo",
        description: "Planning and task tracking for the current session",
        tools: TODO_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "vision",
        description: "Analyze local or remote images with a multimodal model",
        tools: VISION_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "video",
        description: "Analyze local or remote videos with a video-capable multimodal model",
        tools: VIDEO_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "web",
        description: "Search the web and extract page content through Firecrawl",
        tools: WEB_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "yuanbao",
        description: "Local Yuanbao sticker catalogue search",
        tools: YUANBAO_TOOLS,
        includes: &[],
    },
    ToolsetEntry {
        name: "hermes-cli",
        description: "Rust CLI toolset bootstrap",
        tools: &[],
        includes: &[
            "file",
            "browser",
            "terminal",
            "clarify",
            "code_execution",
            "cronjob",
            "delegation",
            "homeassistant",
            "kanban",
            "todo",
            "memory",
            "messaging",
            "image_gen",
            "session_search",
            "skills",
            "tts",
            "vision",
            "web",
        ],
    },
    ToolsetEntry {
        name: "hermes-acp",
        description: "Rust ACP bootstrap toolset",
        tools: &[],
        includes: &[
            "file",
            "browser",
            "terminal",
            "clarify",
            "code_execution",
            "cronjob",
            "delegation",
            "kanban",
            "todo",
            "memory",
            "messaging",
            "image_gen",
            "session_search",
            "skills",
            "tts",
            "vision",
            "web",
        ],
    },
    ToolsetEntry {
        name: "hermes-api-server",
        description: "Rust API-server bootstrap toolset",
        tools: &[],
        includes: &[
            "file",
            "browser",
            "terminal",
            "clarify",
            "code_execution",
            "cronjob",
            "delegation",
            "homeassistant",
            "kanban",
            "todo",
            "memory",
            "messaging",
            "image_gen",
            "session_search",
            "skills",
            "tts",
            "vision",
            "web",
        ],
    },
    ToolsetEntry {
        name: "hermes-cron",
        description: "Rust cron bootstrap toolset",
        tools: &[],
        includes: &[
            "file",
            "browser",
            "terminal",
            "code_execution",
            "delegation",
            "homeassistant",
            "kanban",
            "todo",
            "memory",
            "messaging",
            "image_gen",
            "session_search",
            "skills",
            "tts",
            "vision",
            "web",
        ],
    },
    ToolsetEntry {
        name: "hermes-discord",
        description: "Rust Discord bootstrap toolset",
        tools: &[],
        includes: &["hermes-cli", "discord", "discord_admin"],
    },
    ToolsetEntry {
        name: "hermes-feishu",
        description: "Rust Feishu bootstrap toolset",
        tools: &[],
        includes: &["hermes-cli", "feishu_doc", "feishu_drive"],
    },
];

const LEGACY_TOOLSETS: &[(&str, &[&str])] = &[("file_tools", FILE_TOOLS)];

const TOOL_ENTRIES: &[ToolEntry] = &[
    ToolEntry {
        name: "terminal",
        toolset: "terminal",
        description: "Execute local shell commands in the foreground or background",
        emoji: "💻",
        schema_fn: terminal_schema,
        handler: handle_terminal,
        check_fn: Some(terminal_available),
    },
    ToolEntry {
        name: "process",
        toolset: "terminal",
        description: "Manage background processes started by the terminal tool",
        emoji: "⚙️",
        schema_fn: process_schema,
        handler: handle_process,
        check_fn: Some(terminal_available),
    },
    ToolEntry {
        name: "browser_navigate",
        toolset: "browser",
        description: "Open a live page in the local browser and return a compact snapshot",
        emoji: "🌐",
        schema_fn: browser_navigate_schema,
        handler: handle_browser_navigate,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "browser_snapshot",
        toolset: "browser",
        description: "Refresh the current browser page snapshot with interactive refs",
        emoji: "📸",
        schema_fn: browser_snapshot_schema,
        handler: handle_browser_snapshot,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "browser_click",
        toolset: "browser",
        description: "Click an element by ref ID from the browser snapshot",
        emoji: "👆",
        schema_fn: browser_click_schema,
        handler: handle_browser_click,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "browser_type",
        toolset: "browser",
        description: "Fill an input element in the browser by ref ID",
        emoji: "⌨️",
        schema_fn: browser_type_schema,
        handler: handle_browser_type,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "browser_scroll",
        toolset: "browser",
        description: "Scroll the active browser page up or down",
        emoji: "📜",
        schema_fn: browser_scroll_schema,
        handler: handle_browser_scroll,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "browser_back",
        toolset: "browser",
        description: "Navigate back in the active browser history",
        emoji: "◀️",
        schema_fn: browser_back_schema,
        handler: handle_browser_back,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "browser_press",
        toolset: "browser",
        description: "Press a keyboard key in the active browser page",
        emoji: "⌨️",
        schema_fn: browser_press_schema,
        handler: handle_browser_press,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "browser_get_images",
        toolset: "browser",
        description: "List images visible on the current browser page",
        emoji: "🖼️",
        schema_fn: browser_get_images_schema,
        handler: handle_browser_get_images,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "browser_console",
        toolset: "browser",
        description: "Read console output or evaluate JavaScript in the active browser page",
        emoji: "🖥️",
        schema_fn: browser_console_schema,
        handler: handle_browser_console,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "browser_cdp",
        toolset: "browser",
        description: "Send a raw Chrome DevTools Protocol command to a configured browser endpoint",
        emoji: "🧪",
        schema_fn: browser_cdp_schema,
        handler: handle_browser_cdp,
        check_fn: Some(browser_cdp_available),
    },
    ToolEntry {
        name: "browser_dialog",
        toolset: "browser",
        description: "Accept or dismiss a blocking JavaScript dialog on a CDP-capable browser page",
        emoji: "💬",
        schema_fn: browser_dialog_schema,
        handler: handle_browser_dialog,
        check_fn: Some(browser_cdp_available),
    },
    ToolEntry {
        name: "browser_vision",
        toolset: "browser",
        description: "Capture a browser screenshot and inspect it with the configured vision model",
        emoji: "👁️",
        schema_fn: browser_vision_schema,
        handler: handle_browser_vision,
        check_fn: Some(browser_available),
    },
    ToolEntry {
        name: "rl_list_environments",
        toolset: "rl",
        description: "List RL environments discovered from the local repo tree",
        emoji: "🧪",
        schema_fn: rl_list_environments_schema,
        handler: handle_rl_list_environments,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "rl_select_environment",
        toolset: "rl",
        description: "Select one RL environment and persist it for later config or run tools",
        emoji: "🧪",
        schema_fn: rl_select_environment_schema,
        handler: handle_rl_select_environment,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "rl_get_current_config",
        toolset: "rl",
        description: "Show the selected RL environment's configurable and locked fields",
        emoji: "🧪",
        schema_fn: rl_get_current_config_schema,
        handler: handle_rl_get_current_config,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "rl_edit_config",
        toolset: "rl",
        description: "Update one configurable RL field for the selected environment",
        emoji: "🧪",
        schema_fn: rl_edit_config_schema,
        handler: handle_rl_edit_config,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "rl_start_training",
        toolset: "rl",
        description: "Start a tracked RL training run if the local Atropos runtime is installed",
        emoji: "🧪",
        schema_fn: rl_start_training_schema,
        handler: handle_rl_start_training,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "rl_check_status",
        toolset: "rl",
        description: "Inspect one tracked RL run's process state and log paths",
        emoji: "🧪",
        schema_fn: rl_check_status_schema,
        handler: handle_rl_check_status,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "rl_stop_training",
        toolset: "rl",
        description: "Stop a tracked RL training run and terminate its child processes",
        emoji: "🧪",
        schema_fn: rl_stop_training_schema,
        handler: handle_rl_stop_training,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "rl_get_results",
        toolset: "rl",
        description: "Return persisted metadata and log tails for one RL run",
        emoji: "🧪",
        schema_fn: rl_get_results_schema,
        handler: handle_rl_get_results,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "rl_list_runs",
        toolset: "rl",
        description: "List tracked RL runs stored under Hermes home",
        emoji: "🧪",
        schema_fn: rl_list_runs_schema,
        handler: handle_rl_list_runs,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "rl_test_inference",
        toolset: "rl",
        description: "Run a small process-mode RL inference sanity check for the selected environment",
        emoji: "🧪",
        schema_fn: rl_test_inference_schema,
        handler: handle_rl_test_inference,
        check_fn: Some(rl_available),
    },
    ToolEntry {
        name: "yb_query_group_info",
        toolset: "yuanbao",
        description: "Query Yuanbao group name, owner, and member count",
        emoji: "👥",
        schema_fn: yb_query_group_info_schema,
        handler: handle_yb_query_group_info,
        check_fn: Some(yuanbao_available),
    },
    ToolEntry {
        name: "yb_query_group_members",
        toolset: "yuanbao",
        description: "Query Yuanbao group members, bots, or nickname matches",
        emoji: "📋",
        schema_fn: yb_query_group_members_schema,
        handler: handle_yb_query_group_members,
        check_fn: Some(yuanbao_available),
    },
    ToolEntry {
        name: "yb_send_dm",
        toolset: "yuanbao",
        description: "Send a private Yuanbao DM to a group member",
        emoji: "✉️",
        schema_fn: yb_send_dm_schema,
        handler: handle_yb_send_dm,
        check_fn: Some(yuanbao_available),
    },
    ToolEntry {
        name: "yb_search_sticker",
        toolset: "yuanbao",
        description: "Search the built-in Yuanbao sticker catalogue by name, id, or keyword",
        emoji: "🔍",
        schema_fn: yb_search_sticker_schema,
        handler: handle_yb_search_sticker,
        check_fn: None,
    },
    ToolEntry {
        name: "yb_send_sticker",
        toolset: "yuanbao",
        description: "Send a built-in Yuanbao sticker to the current or specified chat",
        emoji: "🎨",
        schema_fn: yb_send_sticker_schema,
        handler: handle_yb_send_sticker,
        check_fn: Some(yuanbao_available),
    },
    ToolEntry {
        name: "ha_list_entities",
        toolset: "homeassistant",
        description: "List Home Assistant entities, optionally filtered by domain or area",
        emoji: "🏠",
        schema_fn: ha_list_entities_schema,
        handler: handle_ha_list_entities,
        check_fn: Some(homeassistant_available),
    },
    ToolEntry {
        name: "ha_get_state",
        toolset: "homeassistant",
        description: "Fetch the detailed state and attributes of one Home Assistant entity",
        emoji: "🏠",
        schema_fn: ha_get_state_schema,
        handler: handle_ha_get_state,
        check_fn: Some(homeassistant_available),
    },
    ToolEntry {
        name: "ha_list_services",
        toolset: "homeassistant",
        description: "List available Home Assistant services and their field descriptions",
        emoji: "🏠",
        schema_fn: ha_list_services_schema,
        handler: handle_ha_list_services,
        check_fn: Some(homeassistant_available),
    },
    ToolEntry {
        name: "ha_call_service",
        toolset: "homeassistant",
        description: "Call a Home Assistant service with optional entity_id and payload data",
        emoji: "🏠",
        schema_fn: ha_call_service_schema,
        handler: handle_ha_call_service,
        check_fn: Some(homeassistant_available),
    },
    ToolEntry {
        name: "kanban_show",
        toolset: "kanban",
        description: "Read a kanban task's state, comments, runs, and worker context",
        emoji: "📋",
        schema_fn: kanban_show_schema,
        handler: handle_kanban_show,
        check_fn: Some(kanban_available),
    },
    ToolEntry {
        name: "kanban_complete",
        toolset: "kanban",
        description: "Mark a kanban task done with a structured handoff",
        emoji: "✔",
        schema_fn: kanban_complete_schema,
        handler: handle_kanban_complete,
        check_fn: Some(kanban_available),
    },
    ToolEntry {
        name: "kanban_block",
        toolset: "kanban",
        description: "Block a kanban task and record the reason",
        emoji: "⏸",
        schema_fn: kanban_block_schema,
        handler: handle_kanban_block,
        check_fn: Some(kanban_available),
    },
    ToolEntry {
        name: "kanban_heartbeat",
        toolset: "kanban",
        description: "Record a heartbeat for a running kanban worker",
        emoji: "💓",
        schema_fn: kanban_heartbeat_schema,
        handler: handle_kanban_heartbeat,
        check_fn: Some(kanban_available),
    },
    ToolEntry {
        name: "kanban_comment",
        toolset: "kanban",
        description: "Append a durable comment to a kanban task thread",
        emoji: "💬",
        schema_fn: kanban_comment_schema,
        handler: handle_kanban_comment,
        check_fn: Some(kanban_available),
    },
    ToolEntry {
        name: "kanban_create",
        toolset: "kanban",
        description: "Create a new kanban task with optional parent dependencies",
        emoji: "➕",
        schema_fn: kanban_create_schema,
        handler: handle_kanban_create,
        check_fn: Some(kanban_available),
    },
    ToolEntry {
        name: "kanban_link",
        toolset: "kanban",
        description: "Link an existing parent task to an existing child task",
        emoji: "🔗",
        schema_fn: kanban_link_schema,
        handler: handle_kanban_link,
        check_fn: Some(kanban_available),
    },
    ToolEntry {
        name: "clarify",
        toolset: "clarify",
        description: "Ask the user a clarifying question or present up to 4 choices",
        emoji: "❓",
        schema_fn: clarify_schema,
        handler: handle_clarify,
        check_fn: Some(clarify_available),
    },
    ToolEntry {
        name: "cronjob",
        toolset: "cronjob",
        description: "Create, list, pause, resume, update, remove, or trigger scheduled jobs",
        emoji: "⏰",
        schema_fn: cronjob_schema,
        handler: handle_cronjob,
        check_fn: Some(cronjob_available),
    },
    ToolEntry {
        name: "delegate_task",
        toolset: "delegation",
        description: "Spawn one or more subagents to work in isolated contexts",
        emoji: "🔀",
        schema_fn: delegate_task_schema,
        handler: handle_delegate_task,
        check_fn: Some(delegate_task_available),
    },
    ToolEntry {
        name: "discord",
        toolset: "discord",
        description: "Read recent Discord channel activity, search members, and create threads",
        emoji: "💬",
        schema_fn: discord_core_schema,
        handler: handle_discord,
        check_fn: Some(discord_available),
    },
    ToolEntry {
        name: "discord_admin",
        toolset: "discord_admin",
        description: "Inspect and manage Discord servers, channels, pins, roles, and members",
        emoji: "🛡️",
        schema_fn: discord_admin_schema,
        handler: handle_discord_admin,
        check_fn: Some(discord_admin_available),
    },
    ToolEntry {
        name: "feishu_doc_read",
        toolset: "feishu_doc",
        description: "Read the plain-text content of a Feishu or Lark document",
        emoji: "📄",
        schema_fn: feishu_doc_read_schema,
        handler: handle_feishu_doc_read,
        check_fn: Some(feishu_available),
    },
    ToolEntry {
        name: "feishu_drive_list_comments",
        toolset: "feishu_drive",
        description: "List comments on a Feishu or Lark document",
        emoji: "💬",
        schema_fn: feishu_drive_list_comments_schema,
        handler: handle_feishu_drive_list_comments,
        check_fn: Some(feishu_available),
    },
    ToolEntry {
        name: "feishu_drive_list_comment_replies",
        toolset: "feishu_drive",
        description: "List replies in a Feishu or Lark document comment thread",
        emoji: "💬",
        schema_fn: feishu_drive_list_comment_replies_schema,
        handler: handle_feishu_drive_list_comment_replies,
        check_fn: Some(feishu_available),
    },
    ToolEntry {
        name: "feishu_drive_reply_comment",
        toolset: "feishu_drive",
        description: "Reply to a Feishu or Lark document comment thread",
        emoji: "✉️",
        schema_fn: feishu_drive_reply_comment_schema,
        handler: handle_feishu_drive_reply_comment,
        check_fn: Some(feishu_available),
    },
    ToolEntry {
        name: "feishu_drive_add_comment",
        toolset: "feishu_drive",
        description: "Add a whole-document Feishu or Lark comment",
        emoji: "✉️",
        schema_fn: feishu_drive_add_comment_schema,
        handler: handle_feishu_drive_add_comment,
        check_fn: Some(feishu_available),
    },
    ToolEntry {
        name: "read_file",
        toolset: "file",
        description: "Read a UTF-8 text file with line numbers and pagination",
        emoji: "📖",
        schema_fn: read_file_schema,
        handler: handle_read_file,
        check_fn: None,
    },
    ToolEntry {
        name: "write_file",
        toolset: "file",
        description: "Write complete content to a file, creating parent directories",
        emoji: "✍️",
        schema_fn: write_file_schema,
        handler: handle_write_file,
        check_fn: None,
    },
    ToolEntry {
        name: "patch",
        toolset: "file",
        description: "Apply targeted string replacements or V4A-style file patches",
        emoji: "🔧",
        schema_fn: patch_schema,
        handler: handle_patch,
        check_fn: None,
    },
    ToolEntry {
        name: "search_files",
        toolset: "file",
        description: "Search file contents with ripgrep or list files by glob",
        emoji: "🔎",
        schema_fn: search_files_schema,
        handler: handle_search_files,
        check_fn: Some(ripgrep_available),
    },
    ToolEntry {
        name: "web_search",
        toolset: "web",
        description: "Search the web for titles, URLs, and snippets via Firecrawl",
        emoji: "🔍",
        schema_fn: web_search_schema,
        handler: handle_web_search,
        check_fn: Some(web_tools_available),
    },
    ToolEntry {
        name: "web_extract",
        toolset: "web",
        description: "Extract markdown content from web pages or PDFs via Firecrawl",
        emoji: "📄",
        schema_fn: web_extract_schema,
        handler: handle_web_extract,
        check_fn: Some(web_tools_available),
    },
    ToolEntry {
        name: "skills_list",
        toolset: "skills",
        description: "List available skills with minimal metadata",
        emoji: "📚",
        schema_fn: skills_list_schema,
        handler: handle_skills_list,
        check_fn: Some(skills_available),
    },
    ToolEntry {
        name: "skill_view",
        toolset: "skills",
        description: "Load full skill content or a linked file within a skill",
        emoji: "📚",
        schema_fn: skill_view_schema,
        handler: handle_skill_view,
        check_fn: Some(skills_available),
    },
    ToolEntry {
        name: "skill_manage",
        toolset: "skills",
        description: "Create, edit, patch, or delete local skills and their supporting files",
        emoji: "📝",
        schema_fn: skill_manage_schema,
        handler: handle_skill_manage,
        check_fn: Some(skills_available),
    },
    ToolEntry {
        name: "session_search",
        toolset: "session_search",
        description: "Search prior sessions or browse recent ones from the Rust state database",
        emoji: "🔍",
        schema_fn: session_search_schema,
        handler: handle_session_search,
        check_fn: None,
    },
    ToolEntry {
        name: "send_message",
        toolset: "messaging",
        description: "Send a message to a configured Telegram, Discord, Slack, Feishu, Matrix, Signal, or Yuanbao target",
        emoji: "📤",
        schema_fn: send_message_schema,
        handler: handle_send_message,
        check_fn: Some(send_message_available),
    },
    ToolEntry {
        name: "spotify_playback",
        toolset: "spotify",
        description: "Control Spotify playback and inspect the active playback session",
        emoji: "🎵",
        schema_fn: spotify_playback_schema,
        handler: handle_spotify_playback,
        check_fn: Some(spotify_available),
    },
    ToolEntry {
        name: "spotify_devices",
        toolset: "spotify",
        description: "List Spotify Connect devices and transfer playback between them",
        emoji: "🔈",
        schema_fn: spotify_devices_schema,
        handler: handle_spotify_devices,
        check_fn: Some(spotify_available),
    },
    ToolEntry {
        name: "spotify_queue",
        toolset: "spotify",
        description: "Inspect or modify the Spotify playback queue",
        emoji: "📻",
        schema_fn: spotify_queue_schema,
        handler: handle_spotify_queue,
        check_fn: Some(spotify_available),
    },
    ToolEntry {
        name: "spotify_search",
        toolset: "spotify",
        description: "Search the Spotify catalog for tracks, albums, artists, playlists, and shows",
        emoji: "🔎",
        schema_fn: spotify_search_schema,
        handler: handle_spotify_search,
        check_fn: Some(spotify_available),
    },
    ToolEntry {
        name: "spotify_playlists",
        toolset: "spotify",
        description: "List, inspect, create, and edit Spotify playlists",
        emoji: "📚",
        schema_fn: spotify_playlists_schema,
        handler: handle_spotify_playlists,
        check_fn: Some(spotify_available),
    },
    ToolEntry {
        name: "spotify_albums",
        toolset: "spotify",
        description: "Fetch Spotify album metadata and tracks",
        emoji: "💿",
        schema_fn: spotify_albums_schema,
        handler: handle_spotify_albums,
        check_fn: Some(spotify_available),
    },
    ToolEntry {
        name: "spotify_library",
        toolset: "spotify",
        description: "List, save, or remove the user's saved Spotify tracks or albums",
        emoji: "❤️",
        schema_fn: spotify_library_schema,
        handler: handle_spotify_library,
        check_fn: Some(spotify_available),
    },
    ToolEntry {
        name: "text_to_speech",
        toolset: "tts",
        description: "Convert text into an audio file and return a MEDIA tag for delivery surfaces",
        emoji: "🔊",
        schema_fn: text_to_speech_schema,
        handler: handle_text_to_speech,
        check_fn: None,
    },
    ToolEntry {
        name: "vision_analyze",
        toolset: "vision",
        description: "Inspect an image from a URL, file path, or file URI with a multimodal model",
        emoji: "👁️",
        schema_fn: vision_analyze_schema,
        handler: handle_vision_analyze,
        check_fn: None,
    },
    ToolEntry {
        name: "video_analyze",
        toolset: "video",
        description: "Inspect a video from a URL, file path, or file URI with a video-capable multimodal model",
        emoji: "🎬",
        schema_fn: video_analyze_schema,
        handler: handle_video_analyze,
        check_fn: None,
    },
    ToolEntry {
        name: "execute_code",
        toolset: "code_execution",
        description: "Run a Python script that can call a subset of Hermes tools programmatically",
        emoji: "🐍",
        schema_fn: execute_code_schema,
        handler: handle_execute_code,
        check_fn: Some(execute_code_available),
    },
    ToolEntry {
        name: "memory",
        toolset: "memory",
        description: "Persist durable user and environment facts across sessions",
        emoji: "🧠",
        schema_fn: memory_schema,
        handler: handle_memory,
        check_fn: None,
    },
    ToolEntry {
        name: "mixture_of_agents",
        toolset: "moa",
        description: "Route a hard problem through multiple frontier models collaboratively",
        emoji: "🧠",
        schema_fn: mixture_of_agents_schema,
        handler: handle_mixture_of_agents,
        check_fn: Some(mixture_of_agents_available),
    },
    ToolEntry {
        name: "image_generate",
        toolset: "image_gen",
        description: "Generate high-quality images from text prompts",
        emoji: "🎨",
        schema_fn: image_generate_schema,
        handler: handle_image_generate,
        check_fn: Some(image_generate_available),
    },
    ToolEntry {
        name: "todo",
        toolset: "todo",
        description: "Manage the current session task list for planning and progress tracking",
        emoji: "📝",
        schema_fn: todo_schema,
        handler: handle_todo,
        check_fn: None,
    },
];

#[derive(Debug, Default)]
struct DynamicToolRegistry {
    toolsets: BTreeMap<String, BTreeSet<String>>,
    tool_to_toolset: BTreeMap<String, String>,
}

static DYNAMIC_TOOL_REGISTRY_CACHE: OnceLock<Mutex<Option<&'static DynamicToolRegistry>>> =
    OnceLock::new();

fn dynamic_tool_registry() -> &'static DynamicToolRegistry {
    let cache = DYNAMIC_TOOL_REGISTRY_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().expect("dynamic tool registry cache poisoned");
    if let Some(existing) = *guard {
        return existing;
    }
    let built = Box::leak(Box::new(build_dynamic_tool_registry()));
    *guard = Some(built);
    built
}

fn build_dynamic_tool_registry() -> DynamicToolRegistry {
    let mut plugins = BTreeMap::<String, Vec<(String, String)>>::new();

    for dir in discover_plugin_dirs(bundled_plugins_root(), 2) {
        if let Some((name, tools)) = parse_plugin_tool_registrations(&dir) {
            plugins.entry(name).or_insert(tools);
        }
    }
    for dir in discover_plugin_dirs(user_plugins_root(), 2) {
        if let Some((name, tools)) = parse_plugin_tool_registrations(&dir) {
            plugins.insert(name, tools);
        }
    }

    let mut registry = DynamicToolRegistry::default();
    for tools in plugins.into_values() {
        for (toolset, name) in tools {
            registry
                .toolsets
                .entry(toolset.clone())
                .or_default()
                .insert(name.clone());
            registry.tool_to_toolset.insert(name, toolset);
        }
    }
    registry
}

fn parse_plugin_tool_registrations(dir: &Path) -> Option<(String, Vec<(String, String)>)> {
    let manifest = manifest_path(dir)?;
    let parsed = serde_yaml::from_str::<serde_yaml::Value>(&fs::read_to_string(manifest).ok()?)
        .unwrap_or(serde_yaml::Value::Null);
    let mapping = parsed.as_mapping()?;
    let plugin_name = mapping
        .get(serde_yaml::Value::String("name".to_string()))
        .and_then(serde_yaml::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            dir.file_name()
                .and_then(|value| value.to_str())
                .map(ToOwned::to_owned)
        })?;
    let provides_tools = mapping
        .get(serde_yaml::Value::String("provides_tools".to_string()))
        .and_then(serde_yaml::Value::as_sequence)
        .map(|values| {
            values
                .iter()
                .filter_map(serde_yaml::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let init_path = dir.join("__init__.py");
    let source = fs::read_to_string(init_path).ok()?;
    let calls = parse_register_tool_calls(&source);
    let mut discovered = Vec::new();
    for (name, toolset) in &calls {
        if let (Some(name), Some(toolset)) = (name.as_ref(), toolset.as_ref()) {
            discovered.push((toolset.clone(), name.clone()));
        }
    }

    let discovered_toolsets = calls
        .iter()
        .filter_map(|(_, toolset)| toolset.clone())
        .collect::<BTreeSet<_>>();
    if !provides_tools.is_empty() {
        if discovered_toolsets.len() == 1 {
            let toolset = discovered_toolsets
                .iter()
                .next()
                .cloned()
                .unwrap_or_default();
            for tool in &provides_tools {
                if !discovered.iter().any(|(current_toolset, current_name)| {
                    current_toolset == &toolset && current_name == tool
                }) {
                    discovered.push((toolset.clone(), tool.clone()));
                }
            }
        } else {
            let inferred = provides_tools
                .iter()
                .filter_map(|tool| tool_entry(tool).map(|entry| entry.toolset.to_string()))
                .collect::<BTreeSet<_>>();
            if inferred.len() == 1 {
                let toolset = inferred.iter().next().cloned().unwrap_or_default();
                for tool in provides_tools {
                    if !discovered.iter().any(|(current_toolset, current_name)| {
                        current_toolset == &toolset && current_name == &tool
                    }) {
                        discovered.push((toolset.clone(), tool));
                    }
                }
            }
        }
    }

    if discovered.is_empty() {
        return None;
    }
    discovered.sort();
    discovered.dedup();
    Some((plugin_name, discovered))
}

fn discover_plugin_dirs(root: PathBuf, max_depth: usize) -> Vec<PathBuf> {
    let mut results = Vec::new();
    collect_plugin_dirs(&root, 0, max_depth, &mut results);
    results.sort();
    results
}

fn collect_plugin_dirs(root: &Path, depth: usize, max_depth: usize, results: &mut Vec<PathBuf>) {
    if depth > max_depth || !root.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !path.is_dir() || name.starts_with(['.', '_']) {
            continue;
        }
        if manifest_path(&path).is_some() && path.join("__init__.py").is_file() {
            results.push(path.clone());
        }
        collect_plugin_dirs(&path, depth + 1, max_depth, results);
    }
}

fn bundled_plugins_root() -> PathBuf {
    std::env::var_os("HERMES_BUNDLED_PLUGINS")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("plugins"))
}

fn user_plugins_root() -> PathBuf {
    std::env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".hermes")))
        .unwrap_or_else(|| PathBuf::from(".hermes"))
        .join("plugins")
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

fn manifest_path(path: &Path) -> Option<PathBuf> {
    let yaml = path.join("plugin.yaml");
    if yaml.is_file() {
        return Some(yaml);
    }
    let yml = path.join("plugin.yml");
    yml.is_file().then_some(yml)
}

fn parse_register_tool_calls(source: &str) -> Vec<(Option<String>, Option<String>)> {
    let mut calls = Vec::new();
    for matched in register_tool_call_regex().find_iter(source) {
        let open_index = matched.end().saturating_sub(1);
        let Some(close_index) = find_python_matching_delimiter(source, open_index, '(', ')') else {
            continue;
        };
        let body = &source[open_index + 1..close_index];
        let args = parse_python_keyword_args(body);
        calls.push((
            args.get("name")
                .and_then(|value| parse_python_string(value)),
            args.get("toolset")
                .and_then(|value| parse_python_string(value)),
        ));
    }
    calls
}

fn parse_python_keyword_args(body: &str) -> BTreeMap<String, String> {
    let mut args = BTreeMap::new();
    for item in split_python_top_level_items(body, ',') {
        let item = strip_python_comment(&item);
        if item.trim().is_empty() {
            continue;
        }
        let Some((key, value)) = split_python_top_level_pair(&item, '=') else {
            continue;
        };
        args.insert(key.trim().to_string(), value.trim().to_string());
    }
    args
}

fn find_python_matching_delimiter(
    source: &str,
    open_index: usize,
    open_char: char,
    close_char: char,
) -> Option<usize> {
    let mut stack = vec![open_char];
    let mut quote = None;
    let mut escaped = false;
    let mut comment = false;

    for (offset, ch) in source[open_index + 1..].char_indices() {
        if comment {
            if ch == '\n' {
                comment = false;
            }
            continue;
        }
        if let Some(active) = quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '#' => comment = true,
            '\'' | '"' => quote = Some(ch),
            '(' | '[' | '{' => stack.push(ch),
            ')' => {
                if stack.pop() != Some('(') {
                    return None;
                }
                if stack.is_empty() && close_char == ')' {
                    return Some(open_index + 1 + offset);
                }
            }
            ']' => {
                if stack.pop() != Some('[') {
                    return None;
                }
                if stack.is_empty() && close_char == ']' {
                    return Some(open_index + 1 + offset);
                }
            }
            '}' => {
                if stack.pop() != Some('{') {
                    return None;
                }
                if stack.is_empty() && close_char == '}' {
                    return Some(open_index + 1 + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_python_top_level_items(body: &str, separator: char) -> Vec<String> {
    let mut items = Vec::new();
    let mut current = String::new();
    let mut stack = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    let mut comment = false;

    for ch in body.chars() {
        if comment {
            if ch == '\n' {
                comment = false;
            }
            continue;
        }
        if let Some(active) = quote {
            current.push(ch);
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '#' => {
                comment = true;
            }
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
            }
            '(' | '[' | '{' => {
                stack.push(ch);
                current.push(ch);
            }
            ')' | ']' | '}' => {
                stack.pop();
                current.push(ch);
            }
            _ if ch == separator && stack.is_empty() => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    items.push(trimmed.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        items.push(trimmed.to_string());
    }
    items
}

fn split_python_top_level_pair(text: &str, separator: char) -> Option<(String, String)> {
    let mut stack = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    let mut comment = false;

    for (index, ch) in text.char_indices() {
        if comment {
            if ch == '\n' {
                comment = false;
            }
            continue;
        }
        if let Some(active) = quote {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '#' => comment = true,
            '\'' | '"' => quote = Some(ch),
            '(' | '[' | '{' => stack.push(ch),
            ')' | ']' | '}' => {
                stack.pop();
            }
            _ if ch == separator && stack.is_empty() => {
                return Some((
                    text[..index].trim().to_string(),
                    text[index + ch.len_utf8()..].trim().to_string(),
                ));
            }
            _ => {}
        }
    }

    None
}

fn strip_python_comment(text: &str) -> String {
    let mut result = String::new();
    let mut quote = None;
    let mut escaped = false;

    for ch in text.chars() {
        if let Some(active) = quote {
            result.push(ch);
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                result.push(ch);
            }
            '#' => break,
            _ => result.push(ch),
        }
    }

    result.trim().to_string()
}

fn parse_python_string(text: &str) -> Option<String> {
    let trimmed = strip_python_comment(text);
    let bytes = trimmed.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let quote = *bytes.first()?;
    if !matches!(quote, b'\'' | b'"') || bytes.last().copied()? != quote {
        return None;
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    Some(
        inner
            .replace("\\\"", "\"")
            .replace("\\'", "'")
            .replace("\\\\", "\\"),
    )
}

fn register_tool_call_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"ctx\.register_tool\s*\(").expect("register_tool regex"))
}

#[cfg(test)]
fn reset_dynamic_tool_registry_for_tests() {
    if let Some(cache) = DYNAMIC_TOOL_REGISTRY_CACHE.get() {
        let mut guard = cache.lock().expect("dynamic tool registry cache poisoned");
        *guard = None;
    }
}

pub fn get_tool_definitions(
    enabled_toolsets: Option<&[String]>,
    disabled_toolsets: Option<&[String]>,
) -> Vec<ToolDefinition> {
    get_tool_definitions_with_runtime(enabled_toolsets, disabled_toolsets, None)
}

pub fn get_tool_definitions_with_runtime(
    enabled_toolsets: Option<&[String]>,
    disabled_toolsets: Option<&[String]>,
    runtime: Option<&ToolRuntime>,
) -> Vec<ToolDefinition> {
    let mut selected = BTreeSet::new();

    if let Some(enabled) = enabled_toolsets {
        for name in enabled {
            for tool in resolve_toolset(name) {
                selected.insert(tool);
            }
            if let Some(runtime) = runtime {
                for tool in runtime.resolve_dynamic_toolset(name) {
                    selected.insert(tool);
                }
            }
        }
    } else {
        for name in get_toolset_names() {
            for tool in resolve_toolset(&name) {
                selected.insert(tool);
            }
        }
        if let Some(runtime) = runtime {
            for toolset in runtime.dynamic_toolset_names() {
                for tool in runtime.resolve_dynamic_toolset(&toolset) {
                    selected.insert(tool);
                }
            }
        }
    }

    if let Some(disabled) = disabled_toolsets {
        for name in disabled {
            for tool in resolve_toolset(name) {
                selected.remove(&tool);
            }
            if let Some(runtime) = runtime {
                for tool in runtime.resolve_dynamic_toolset(name) {
                    selected.remove(&tool);
                }
            }
        }
    }

    selected
        .into_iter()
        .filter_map(|name| {
            if let Some(entry) = tool_entry(&name) {
                return tool_is_available(entry).then(|| build_tool_definition(entry));
            }
            runtime.and_then(|runtime| runtime.dynamic_tool_definition(&name))
        })
        .collect()
}

pub fn get_tool_definitions_for_runtime(
    runtime: &ToolRuntime,
    enabled_toolsets: Option<&[String]>,
    disabled_toolsets: Option<&[String]>,
) -> Vec<ToolDefinition> {
    let mut definitions = get_tool_definitions(enabled_toolsets, disabled_toolsets);
    let mut seen = definitions
        .iter()
        .map(|definition| definition.name.clone())
        .collect::<BTreeSet<_>>();

    for definition in runtime.context_tool_definitions() {
        if seen.insert(definition.name.clone()) {
            definitions.push(definition);
        }
    }

    for definition in runtime.external_memory_tool_definitions() {
        if seen.insert(definition.name.clone()) {
            definitions.push(definition);
        }
    }

    definitions
}

pub fn get_all_tool_names() -> Vec<String> {
    let mut names = TOOL_ENTRIES
        .iter()
        .map(|entry| entry.name.to_string())
        .collect::<BTreeSet<_>>();
    names.extend(dynamic_tool_registry().tool_to_toolset.keys().cloned());
    names.into_iter().collect()
}

pub fn get_toolset_for_tool(name: &str) -> Option<String> {
    tool_entry(name)
        .map(|entry| entry.toolset.to_string())
        .or_else(|| dynamic_tool_registry().tool_to_toolset.get(name).cloned())
}

pub fn get_toolset_names() -> Vec<String> {
    let mut names = TOOLSETS
        .iter()
        .map(|entry| entry.name.to_string())
        .collect::<BTreeSet<_>>();
    names.extend(dynamic_tool_registry().toolsets.keys().cloned());
    names.into_iter().collect()
}

pub fn get_toolset_info(name: &str) -> Option<ToolsetInfo> {
    let dynamic_tools = dynamic_tool_registry()
        .toolsets
        .get(name)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect::<Vec<_>>();

    let (description, mut direct_tools, includes) = if let Some(entry) = toolset_entry(name) {
        let direct_tools = entry
            .tools
            .iter()
            .map(|tool| (*tool).to_string())
            .collect::<Vec<_>>();
        let includes = entry
            .includes
            .iter()
            .map(|toolset| (*toolset).to_string())
            .collect::<Vec<_>>();
        (entry.description.to_string(), direct_tools, includes)
    } else if let Some(legacy) = legacy_toolset(name) {
        (
            "Legacy toolset".to_string(),
            legacy
                .iter()
                .map(|tool| (*tool).to_string())
                .collect::<Vec<_>>(),
            Vec::new(),
        )
    } else if !dynamic_tools.is_empty() {
        (
            "Plugin toolset discovered from plugin registrations".to_string(),
            Vec::new(),
            Vec::new(),
        )
    } else {
        return None;
    };
    direct_tools.extend(dynamic_tools);
    direct_tools.sort();
    direct_tools.dedup();
    let resolved_tools = resolve_toolset(name);
    let implemented_tools = resolved_tools
        .iter()
        .filter_map(|tool| tool_entry(tool))
        .filter(|tool| tool_is_available(tool))
        .map(|tool| tool.name.to_string())
        .collect::<Vec<_>>();

    Some(ToolsetInfo {
        name: name.to_string(),
        description,
        direct_tools,
        includes,
        resolved_tools,
        implemented_tools: implemented_tools.clone(),
        available: !implemented_tools.is_empty(),
    })
}

pub fn get_all_toolsets() -> BTreeMap<String, ToolsetInfo> {
    let mut result = BTreeMap::new();
    for name in get_toolset_names() {
        if let Some(info) = get_toolset_info(&name) {
            result.insert(name, info);
        }
    }
    result
}

pub fn validate_toolset(name: &str) -> bool {
    if matches!(name, "all" | "*") {
        return true;
    }
    toolset_entry(name).is_some()
        || legacy_toolset(name).is_some()
        || dynamic_tool_registry().toolsets.contains_key(name)
}

pub fn resolve_toolset(name: &str) -> Vec<String> {
    let mut visited = HashSet::new();
    resolve_toolset_inner(name, &mut visited)
}

pub fn coerce_tool_args(tool_name: &str, args: Value) -> Value {
    let Some(entry) = tool_entry(tool_name) else {
        return args;
    };
    coerce_tool_args_against_schema(args, &(entry.schema_fn)())
}

fn coerce_tool_args_against_schema(args: Value, schema: &Value) -> Value {
    if !args.is_object() {
        return args;
    }

    let mut args = args;
    let Some(object) = args.as_object_mut() else {
        return args;
    };
    let properties = schema
        .get("parameters")
        .and_then(|params| params.get("properties"))
        .and_then(Value::as_object);
    let Some(properties) = properties else {
        return args;
    };

    for (key, value) in object.iter_mut() {
        let Some(property_schema) = properties.get(key) else {
            continue;
        };
        let expected_type = property_schema.get("type");

        if expected_type == Some(&Value::String("array".to_string()))
            && !value.is_array()
            && !value.is_null()
        {
            if let Some(text) = value.as_str() {
                let coerced = coerce_value(text, expected_type, property_schema);
                if coerced != Value::String(text.to_string()) {
                    *value = coerced;
                    continue;
                }
            }
            *value = Value::Array(vec![value.clone()]);
            continue;
        }

        let Some(text) = value.as_str() else {
            continue;
        };
        let coerced = coerce_value(text, expected_type, property_schema);
        if coerced != Value::String(text.to_string()) {
            *value = coerced;
        }
    }

    args
}

pub fn dispatch_tool(name: &str, args: Value, runtime: &ToolRuntime) -> String {
    dispatch_tool_with_messages(name, args, runtime, &[])
}

pub(crate) fn dispatch_tool_with_messages(
    name: &str,
    args: Value,
    runtime: &ToolRuntime,
    messages: &[Value],
) -> String {
    let static_entry = tool_entry(name);
    let context_schema = runtime.context_tool_schema(name);
    let dynamic_schema = runtime.dynamic_tool_definition(name).map(|definition| definition.schema);
    let Some(schema) = static_entry
        .map(|entry| (entry.schema_fn)())
        .or(context_schema)
        .or(dynamic_schema)
    else {
        if runtime.external_memory_has_tool(name) {
            return runtime
                .external_memory_handle_tool_call(name, &args)
                .unwrap_or_else(tool_error);
        }
        return tool_error(format!("unknown tool: {name}"));
    };
    if let Some(entry) = static_entry
        && !tool_is_available(entry)
    {
        return tool_error(format!("tool unavailable: {name}"));
    }
    let coerced = coerce_tool_args_against_schema(args, &schema);
    let pre_hook_payload = json!({
        "tool_name": name,
        "args": coerced.clone(),
        "task_id": runtime.current_session_id().unwrap_or_default(),
        "session_id": runtime.current_session_id().unwrap_or_default(),
        "tool_call_id": "",
    });
    for hook_result in runtime.invoke_hook("pre_tool_call", &pre_hook_payload) {
        let Some(object) = hook_result.as_object() else {
            continue;
        };
        if object.get("action").and_then(Value::as_str) != Some("block") {
            continue;
        }
        let Some(message) = object.get("message").and_then(Value::as_str) else {
            continue;
        };
        if !message.trim().is_empty() {
            return tool_error(message);
        }
    }
    if let Some(shell_hooks) = runtime.shell_hooks.as_ref()
        && let Some(message) =
            shell_hooks.pre_tool_call(name, &coerced, runtime.current_session_id(), runtime.cwd())
    {
        return tool_error(message);
    }

    runtime.emit_tool_progress(ToolProgressUpdate {
        event_type: "tool.started".to_string(),
        function_name: Some(name.to_string()),
        preview: build_tool_preview(name, &coerced),
        function_args: Some(coerced.clone()),
        duration_ms: None,
        is_error: None,
    });

    let started = std::time::Instant::now();
    let mut result = if let Some(entry) = static_entry {
        (entry.handler)(&coerced, runtime)
    } else if let Some(result) = runtime.dispatch_context_tool(name, &coerced, messages) {
        result
    } else if let Some(result) = runtime.dispatch_dynamic_tool(name, &coerced) {
        result
    } else {
        tool_error(format!("unknown tool: {name}"))
    };
    let duration_ms = started.elapsed().as_millis() as u64;
    if let Some(shell_hooks) = runtime.shell_hooks.as_ref() {
        shell_hooks.post_tool_call(
            name,
            &coerced,
            runtime.current_session_id(),
            runtime.cwd(),
            &result,
            duration_ms,
        );
    }
    let post_hook_payload = json!({
        "tool_name": name,
        "args": coerced,
        "result": result.clone(),
        "task_id": runtime.current_session_id().unwrap_or_default(),
        "session_id": runtime.current_session_id().unwrap_or_default(),
        "tool_call_id": "",
        "duration_ms": duration_ms,
    });
    let _ = runtime.invoke_hook("post_tool_call", &post_hook_payload);
    for hook_result in runtime.invoke_hook("transform_tool_result", &post_hook_payload) {
        if let Some(text) = hook_result.as_str() {
            result = text.to_string();
            break;
        }
    }
    runtime.emit_tool_progress(ToolProgressUpdate {
        event_type: "tool.completed".to_string(),
        function_name: Some(name.to_string()),
        preview: None,
        function_args: None,
        duration_ms: Some(duration_ms),
        is_error: Some(tool_result_is_error(&result)),
    });
    result
}

pub fn tool_error(message: impl Into<String>) -> String {
    json!({ "error": message.into() }).to_string()
}

pub fn tool_result(data: Value) -> String {
    data.to_string()
}

fn build_tool_definition(entry: &ToolEntry) -> ToolDefinition {
    ToolDefinition {
        name: entry.name.to_string(),
        toolset: entry.toolset.to_string(),
        description: entry.description.to_string(),
        emoji: entry.emoji.to_string(),
        schema: (entry.schema_fn)(),
    }
}

fn tool_entry(name: &str) -> Option<&'static ToolEntry> {
    TOOL_ENTRIES.iter().find(|entry| entry.name == name)
}

fn toolset_entry(name: &str) -> Option<&'static ToolsetEntry> {
    TOOLSETS.iter().find(|entry| entry.name == name)
}

fn legacy_toolset(name: &str) -> Option<&'static [&'static str]> {
    LEGACY_TOOLSETS
        .iter()
        .find(|(toolset_name, _)| *toolset_name == name)
        .map(|(_, tools)| *tools)
}

fn tool_is_available(entry: &ToolEntry) -> bool {
    entry.check_fn.is_none_or(|check| check())
}

fn build_tool_preview(name: &str, args: &Value) -> Option<String> {
    let object = args.as_object()?;
    if let Some(command) = object.get("command").and_then(Value::as_str) {
        let trimmed = command.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if let Some(path) = object.get("path").and_then(Value::as_str) {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if let Some(pattern) = object.get("pattern").and_then(Value::as_str) {
        let trimmed = pattern.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    if let Some(query) = object
        .get("query")
        .or_else(|| object.get("q"))
        .and_then(Value::as_str)
    {
        let trimmed = query.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    (!name.is_empty()).then_some(name.to_string())
}

fn tool_result_is_error(result: &str) -> bool {
    serde_json::from_str::<Value>(result)
        .ok()
        .and_then(|value| value.get("error").cloned())
        .is_some_and(|value| !value.is_null())
}

fn resolve_toolset_inner(name: &str, visited: &mut HashSet<String>) -> Vec<String> {
    if matches!(name, "all" | "*") {
        let mut tools = BTreeSet::new();
        for toolset in get_toolset_names() {
            for tool in resolve_toolset_inner(&toolset, &mut HashSet::new()) {
                tools.insert(tool);
            }
        }
        return tools.into_iter().collect();
    }

    if !visited.insert(name.to_string()) {
        return Vec::new();
    }

    if let Some(legacy) = legacy_toolset(name) {
        let mut tools = legacy
            .iter()
            .filter(|tool| tool_entry(tool).is_some())
            .map(|tool| (*tool).to_string())
            .collect::<BTreeSet<_>>();
        if let Some(dynamic) = dynamic_tool_registry().toolsets.get(name) {
            tools.extend(dynamic.iter().cloned());
        }
        return tools.into_iter().collect();
    }

    let mut tools = BTreeSet::new();
    if let Some(entry) = toolset_entry(name) {
        for tool in entry.tools {
            if tool_entry(tool).is_some() {
                tools.insert((*tool).to_string());
            }
        }
        for include in entry.includes {
            for tool in resolve_toolset_inner(include, visited) {
                tools.insert(tool);
            }
        }
    }
    if let Some(dynamic) = dynamic_tool_registry().toolsets.get(name) {
        tools.extend(dynamic.iter().cloned());
    }
    if tools.is_empty() {
        return Vec::new();
    }
    tools.into_iter().collect()
}

fn read_file_schema() -> Value {
    json!({
        "name": "read_file",
        "description": "Read a UTF-8 text file with line numbers and pagination. Use offset and limit for large files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "1-indexed line number to start from",
                    "default": 1,
                    "minimum": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "default": DEFAULT_READ_LIMIT,
                    "maximum": MAX_READ_LIMIT
                }
            },
            "required": ["path"]
        }
    })
}

fn write_file_schema() -> Value {
    json!({
        "name": "write_file",
        "description": "Write complete content to a text file, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Complete file content"
                }
            },
            "required": ["path", "content"]
        }
    })
}

fn patch_schema() -> Value {
    json!({
        "name": "patch",
        "description": "Apply targeted file edits. Use replace mode for exact string replacement, or patch mode for V4A multi-file patches.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["replace", "patch"],
                    "default": "replace",
                    "description": "Edit mode"
                },
                "path": {
                    "type": "string",
                    "description": "Target path for replace mode"
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace in replace mode"
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text for replace mode"
                },
                "replace_all": {
                    "type": "boolean",
                    "default": false,
                    "description": "Replace all matches instead of requiring a unique match"
                },
                "patch": {
                    "type": "string",
                    "description": "V4A patch text for patch mode"
                }
            },
            "required": ["mode"]
        }
    })
}

fn search_files_schema() -> Value {
    json!({
        "name": "search_files",
        "description": "Search file contents with ripgrep or find files by glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern for content search, or glob pattern for file search"
                },
                "target": {
                    "type": "string",
                    "enum": ["content", "files"],
                    "default": "content",
                    "description": "Search inside file contents or list matching files"
                },
                "path": {
                    "type": "string",
                    "default": ".",
                    "description": "Directory or file to search"
                },
                "file_glob": {
                    "type": "string",
                    "description": "Optional file filter for content search"
                },
                "limit": {
                    "type": "integer",
                    "default": DEFAULT_SEARCH_LIMIT,
                    "maximum": MAX_SEARCH_LIMIT
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_only", "count"],
                    "default": "content"
                },
                "context": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "maximum": MAX_CONTEXT_LINES
                }
            },
            "required": ["pattern"]
        }
    })
}

fn session_search_schema() -> Value {
    json!({
        "name": "session_search",
        "description": "Search past sessions by keyword, or browse recent sessions when query is omitted.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "FTS query to search prior sessions. Omit to browse recent sessions."
                },
                "role_filter": {
                    "type": "string",
                    "description": "Optional comma-separated message roles to include, such as 'user,assistant'"
                },
                "limit": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Maximum sessions to return"
                }
            },
            "required": []
        }
    })
}

fn delegate_task_available() -> bool {
    true
}

fn delegate_task_schema() -> Value {
    json!({
        "name": "delegate_task",
        "description": "Spawn one or more subagents to work on tasks in isolated contexts. Each child gets its own conversation and toolset. Provide either `goal` for a single delegated task or `tasks` for parallel batch mode. Results are always returned as an array.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Single-task mode: what the subagent should accomplish"
                },
                "context": {
                    "type": "string",
                    "description": "Optional background information for the subagent"
                },
                "toolsets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional toolsets to enable for the child"
                },
                "tasks": {
                    "type": "array",
                    "description": "Batch mode: tasks to run in parallel",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string"},
                            "context": {"type": "string"},
                            "toolsets": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "role": {
                                "type": "string",
                                "enum": ["leaf", "orchestrator"]
                            }
                        },
                        "required": ["goal"]
                    }
                },
                "role": {
                    "type": "string",
                    "enum": ["leaf", "orchestrator"],
                    "description": "Role for the child agent. Orchestrators can spawn their own leaf workers when delegation.max_spawn_depth allows it."
                }
            },
            "required": []
        }
    })
}

fn memory_schema() -> Value {
    json!({
        "name": "memory",
        "description": "Save durable information to persistent memory that survives across sessions. Use target `user` for user preferences and identity details, and target `memory` for environment facts, project conventions, and lessons learned. Actions: `add`, `replace`, `remove`. Do not store temporary task progress or todo state here.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove"],
                    "description": "The action to perform"
                },
                "target": {
                    "type": "string",
                    "enum": ["memory", "user"],
                    "description": "Which memory store to update"
                },
                "content": {
                    "type": "string",
                    "description": "Entry content for add or replace"
                },
                "old_text": {
                    "type": "string",
                    "description": "Short unique substring identifying the entry to replace or remove"
                }
            },
            "required": ["action", "target"]
        }
    })
}

fn todo_schema() -> Value {
    json!({
        "name": "todo",
        "description": "Manage your task list for the current session. Use it for complex work with multiple steps. Call with no parameters to read the current list. Provide `todos` to write items. With `merge=false` the list is replaced. With `merge=true` existing items are updated by id and new ones are appended. Each item is `{id, content, status}` where status is `pending`, `in_progress`, `completed`, or `cancelled`. Keep list order as priority and keep only one item in progress at a time.",
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Task items to write. Omit to read the current list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique task identifier"
                            },
                            "content": {
                                "type": "string",
                                "description": "Task description"
                            },
                            "status": {
                                "type": "string",
                                "enum": VALID_TODO_STATUSES,
                                "description": "Task status"
                            }
                        }
                    }
                },
                "merge": {
                    "type": "boolean",
                    "default": false,
                    "description": "Update existing items by id instead of replacing the whole list"
                }
            },
            "required": []
        }
    })
}

fn handle_read_file(args: &Value, runtime: &ToolRuntime) -> String {
    let path = match required_non_empty_string(args, "path") {
        Ok(path) => path,
        Err(error) => return tool_error(error),
    };
    let offset = match bounded_integer(args, "offset", 1, 1, i64::MAX) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let limit = match bounded_integer(args, "limit", DEFAULT_READ_LIMIT, 1, MAX_READ_LIMIT) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let resolved = match runtime.resolve_path(&path) {
        Ok(path) => path,
        Err(error) => return tool_error(error),
    };

    let metadata = match fs::metadata(&resolved) {
        Ok(metadata) => metadata,
        Err(error) => {
            return tool_error(format!("read_file: {}: {error}", resolved.display()));
        }
    };
    if !metadata.is_file() {
        return tool_error(format!("read_file: not a file: {}", resolved.display()));
    }

    let bytes = match fs::read(&resolved) {
        Ok(bytes) => bytes,
        Err(error) => {
            return tool_error(format!("read_file: {}: {error}", resolved.display()));
        }
    };
    let text = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => {
            return tool_error(format!(
                "read_file: {} is not a UTF-8 text file",
                resolved.display()
            ));
        }
    };

    let lines = text.lines().collect::<Vec<_>>();
    let total_lines = lines.len() as i64;
    let start_index = (offset - 1).max(0) as usize;
    let mut rendered = Vec::new();
    for (index, line) in lines
        .iter()
        .enumerate()
        .skip(start_index)
        .take(limit as usize)
    {
        rendered.push(format!("{}|{}", index + 1, line));
    }
    let content = rendered.join("\n");
    if content.len() > MAX_TOOL_RESULT_CHARS {
        return tool_error(format!(
            "read_file: result too large ({} chars); reduce limit or increase offset",
            content.len()
        ));
    }

    let returned = rendered.len() as i64;
    let end_line = if returned == 0 {
        offset.saturating_sub(1)
    } else {
        offset + returned - 1
    };

    tool_result(json!({
        "success": true,
        "path": resolved.display().to_string(),
        "offset": offset,
        "limit": limit,
        "start_line": offset,
        "end_line": end_line,
        "total_lines": total_lines,
        "returned_lines": returned,
        "truncated": end_line < total_lines,
        "content": content,
    }))
}

fn handle_write_file(args: &Value, runtime: &ToolRuntime) -> String {
    let path = match required_non_empty_string(args, "path") {
        Ok(path) => path,
        Err(error) => return tool_error(error),
    };
    let content = match required_string(args, "content") {
        Ok(content) => content,
        Err(error) => return tool_error(error),
    };
    let resolved = match runtime.resolve_path(&path) {
        Ok(path) => path,
        Err(error) => return tool_error(error),
    };

    if resolved.is_dir() {
        return tool_error(format!(
            "write_file: target is a directory: {}",
            resolved.display()
        ));
    }
    if let Some(parent) = resolved.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            return tool_error(format!(
                "write_file: creating parent directories for {} failed: {error}",
                resolved.display()
            ));
        }
    }
    if let Err(error) = fs::write(&resolved, content.as_bytes()) {
        return tool_error(format!("write_file: {}: {error}", resolved.display()));
    }

    tool_result(json!({
        "success": true,
        "path": resolved.display().to_string(),
        "bytes_written": content.len(),
        "line_count": content.lines().count(),
    }))
}

fn handle_patch(args: &Value, runtime: &ToolRuntime) -> String {
    let mode = match enum_arg(args, "mode", "replace", &["replace", "patch"]) {
        Ok(mode) => mode,
        Err(error) => return tool_error(error),
    };

    if mode == "replace" {
        let path = match required_non_empty_string(args, "path") {
            Ok(path) => path,
            Err(error) => return tool_error(error),
        };
        let old_string = match required_string(args, "old_string") {
            Ok(value) => value,
            Err(error) => return tool_error(error),
        };
        let new_string = match required_string(args, "new_string") {
            Ok(value) => value,
            Err(error) => return tool_error(error),
        };
        let replace_all = match optional_bool(args, "replace_all") {
            Ok(value) => value.unwrap_or(false),
            Err(error) => return tool_error(error),
        };
        return patch_replace(&path, &old_string, &new_string, replace_all, runtime);
    }

    let patch_text = match required_string(args, "patch") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    apply_v4a_patch(&patch_text, runtime)
}

fn handle_search_files(args: &Value, runtime: &ToolRuntime) -> String {
    if !ripgrep_available() {
        return tool_error("search_files: ripgrep ('rg') is not installed");
    }

    let pattern = match required_non_empty_string(args, "pattern") {
        Ok(pattern) => pattern,
        Err(error) => return tool_error(error),
    };
    let target = match enum_arg(args, "target", "content", &["content", "files"]) {
        Ok(target) => target,
        Err(error) => return tool_error(error),
    };
    let raw_path = optional_non_empty_string(args, "path").unwrap_or_else(|| ".".to_string());
    let search_path = match runtime.resolve_path(&raw_path) {
        Ok(path) => path,
        Err(error) => return tool_error(error),
    };
    if !search_path.exists() {
        return tool_error(format!(
            "search_files: path does not exist: {}",
            search_path.display()
        ));
    }
    let limit = match bounded_integer(args, "limit", DEFAULT_SEARCH_LIMIT, 1, MAX_SEARCH_LIMIT) {
        Ok(limit) => limit,
        Err(error) => return tool_error(error),
    };
    let offset = match bounded_integer(args, "offset", 0, 0, i64::MAX) {
        Ok(offset) => offset,
        Err(error) => return tool_error(error),
    };
    let output_mode = match enum_arg(
        args,
        "output_mode",
        "content",
        &["content", "files_only", "count"],
    ) {
        Ok(mode) => mode,
        Err(error) => return tool_error(error),
    };
    let context = match bounded_integer(args, "context", 0, 0, MAX_CONTEXT_LINES) {
        Ok(context) => context,
        Err(error) => return tool_error(error),
    };
    let file_glob = optional_non_empty_string(args, "file_glob");

    if target == "files" {
        return search_file_names(&pattern, &search_path, limit, offset, runtime);
    }
    search_file_contents(
        &pattern,
        &search_path,
        file_glob.as_deref(),
        limit,
        offset,
        &output_mode,
        context,
        runtime,
    )
}

fn handle_memory(args: &Value, runtime: &ToolRuntime) -> String {
    let action = match required_non_empty_string(args, "action") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let target = match required_non_empty_string(args, "target") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let Some(store) = runtime.memory_store.as_ref() else {
        return tool_error(
            "Memory is not available. It may be disabled in config or this environment.",
        );
    };
    let mut store = match store.lock() {
        Ok(store) => store,
        Err(_) => return tool_error("memory: store lock poisoned"),
    };

    let result = match action.as_str() {
        "add" => {
            let content = match required_non_empty_string(args, "content") {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            store.add(runtime.hermes_home(), &target, &content)
        }
        "replace" => {
            let old_text = match required_non_empty_string(args, "old_text") {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            let content = match required_non_empty_string(args, "content") {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            store.replace(runtime.hermes_home(), &target, &old_text, &content)
        }
        "remove" => {
            let old_text = match required_non_empty_string(args, "old_text") {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            store.remove(runtime.hermes_home(), &target, &old_text)
        }
        other => Err(format!(
            "Unknown action '{other}'. Use: add, replace, remove"
        )),
    };

    match result {
        Ok(value) => {
            if matches!(action.as_str(), "add" | "replace") {
                let metadata = json!({
                    "write_origin": "assistant_tool",
                    "execution_context": "foreground",
                    "session_id": runtime.current_session_id_or_empty(),
                    "tool_name": "memory",
                });
                let content = args
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let _ = runtime.external_memory_on_memory_write(
                    action.as_str(),
                    target.as_str(),
                    content,
                    metadata,
                );
            }
            tool_result(value)
        }
        Err(error) => tool_error(error),
    }
}

fn handle_delegate_task(args: &Value, runtime: &ToolRuntime) -> String {
    let request = match parse_delegate_request(args) {
        Ok(request) => request,
        Err(error) => return tool_error(error),
    };
    match runtime.delegate(request) {
        Ok(value) => tool_result(value),
        Err(error) => tool_error(error),
    }
}

fn handle_todo(args: &Value, runtime: &ToolRuntime) -> String {
    let merge = match optional_bool(args, "merge") {
        Ok(value) => value.unwrap_or(false),
        Err(error) => return tool_error(error),
    };

    let items = match args.get("todos") {
        None | Some(Value::Null) => match runtime.todo_store.lock() {
            Ok(store) => store.read(),
            Err(_) => return tool_error("todo: store lock poisoned"),
        },
        Some(Value::Array(todos)) => match runtime.todo_store.lock() {
            Ok(mut store) => store.write(todos, merge),
            Err(_) => return tool_error("todo: store lock poisoned"),
        },
        Some(_) => return tool_error("todos must be an array"),
    };

    let pending = items.iter().filter(|item| item.status == "pending").count();
    let in_progress = items
        .iter()
        .filter(|item| item.status == "in_progress")
        .count();
    let completed = items
        .iter()
        .filter(|item| item.status == "completed")
        .count();
    let cancelled = items
        .iter()
        .filter(|item| item.status == "cancelled")
        .count();

    tool_result(json!({
        "todos": items,
        "summary": {
            "total": pending + in_progress + completed + cancelled,
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
        }
    }))
}

fn parse_delegate_request(args: &Value) -> Result<DelegateTaskRequest, String> {
    let goal = match args.get("goal") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Some(_) => return Err("goal must be a string".to_string()),
    };
    let context = match args.get("context") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Some(_) => return Err("context must be a string".to_string()),
    };
    let toolsets = parse_optional_string_array(args.get("toolsets"), "toolsets")?;
    let role = match args.get("role") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Some(_) => return Err("role must be a string".to_string()),
    };
    let tasks = match args.get("tasks") {
        None | Some(Value::Null) => None,
        Some(Value::Array(items)) => {
            let mut tasks = Vec::new();
            for (index, item) in items.iter().enumerate() {
                let Some(object) = item.as_object() else {
                    return Err(format!("tasks[{index}] must be an object"));
                };
                let goal = object
                    .get("goal")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| format!("tasks[{index}] is missing a non-empty 'goal'"))?
                    .to_string();
                let context = object
                    .get("context")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned);
                let toolsets = parse_optional_string_array(object.get("toolsets"), "toolsets")?;
                let role = object
                    .get("role")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned);
                tasks.push(DelegateTaskSpec {
                    goal,
                    context,
                    toolsets,
                    role,
                });
            }
            Some(tasks)
        }
        Some(_) => return Err("tasks must be an array".to_string()),
    };

    Ok(DelegateTaskRequest {
        goal,
        context,
        toolsets,
        tasks,
        role,
    })
}

fn parse_optional_string_array(
    value: Option<&Value>,
    field_name: &str,
) -> Result<Option<Vec<String>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(items) = value.as_array() else {
        return Err(format!("{field_name} must be an array of strings"));
    };
    let mut parsed = Vec::new();
    for item in items {
        let Some(text) = item.as_str() else {
            return Err(format!("{field_name} must contain only strings"));
        };
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            parsed.push(trimmed.to_string());
        }
    }
    if parsed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parsed))
    }
}

fn search_file_names(
    pattern: &str,
    search_path: &Path,
    limit: i64,
    offset: i64,
    runtime: &ToolRuntime,
) -> String {
    let search_root = if search_path.is_dir() {
        search_path.to_path_buf()
    } else {
        search_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| runtime.cwd().to_path_buf())
    };
    let restrict_file = if search_path.is_file() {
        Some(search_path.to_path_buf())
    } else {
        None
    };

    let output = match Command::new("rg")
        .current_dir(runtime.cwd())
        .arg("--files")
        .arg(&search_root)
        .arg("-g")
        .arg(pattern)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(error) => return tool_error(format!("search_files: rg failed to start: {error}")),
    };

    if !matches!(output.status.code(), Some(0) | Some(1)) {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return tool_error(format!("search_files: rg failed: {stderr}"));
    }

    let mut results = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let candidate = PathBuf::from(line);
        let full_path = if candidate.is_absolute() {
            candidate
        } else {
            runtime.cwd().join(candidate)
        };
        if let Some(restrict) = restrict_file.as_ref()
            && full_path != *restrict
        {
            continue;
        }
        results.push(json!({
            "path": full_path.display().to_string(),
        }));
    }

    let total_results = results.len();
    let paged = results
        .into_iter()
        .skip(offset as usize)
        .take(limit as usize)
        .collect::<Vec<_>>();

    tool_result(json!({
        "success": true,
        "target": "files",
        "pattern": pattern,
        "path": search_path.display().to_string(),
        "offset": offset,
        "limit": limit,
        "total_results": total_results,
        "returned": paged.len(),
        "results": paged,
    }))
}

fn search_file_contents(
    pattern: &str,
    search_path: &Path,
    file_glob: Option<&str>,
    limit: i64,
    offset: i64,
    output_mode: &str,
    context: i64,
    runtime: &ToolRuntime,
) -> String {
    let mut command = Command::new("rg");
    command
        .current_dir(runtime.cwd())
        .arg("--json")
        .arg("--line-number")
        .arg("--color")
        .arg("never");
    if context > 0 && output_mode == "content" {
        command.arg("--context").arg(context.to_string());
    }
    if let Some(file_glob) = file_glob {
        command.arg("--glob").arg(file_glob);
    }
    command.arg("--").arg(pattern).arg(search_path);

    let output = match command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(error) => return tool_error(format!("search_files: rg failed to start: {error}")),
    };
    if !matches!(output.status.code(), Some(0) | Some(1)) {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return tool_error(format!("search_files: rg failed: {stderr}"));
    }

    let mut rows = Vec::new();
    let mut counts: HashMap<String, i64> = HashMap::new();

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(event) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(event_type) = event.get("type").and_then(Value::as_str) else {
            continue;
        };
        let Some(data) = event.get("data") else {
            continue;
        };

        match event_type {
            "match" => {
                let path = json_path_text(data.get("path"));
                if path.is_empty() {
                    continue;
                }
                let line_number = data.get("line_number").and_then(Value::as_i64);
                let text = data
                    .get("lines")
                    .and_then(|lines| lines.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim_end_matches('\n')
                    .to_string();
                *counts.entry(path.clone()).or_insert(0) += 1;
                if output_mode == "content" {
                    rows.push(json!({
                        "kind": "match",
                        "path": path,
                        "line": line_number,
                        "text": text,
                    }));
                }
            }
            "context" if output_mode == "content" && context > 0 => {
                let path = json_path_text(data.get("path"));
                if path.is_empty() {
                    continue;
                }
                let line_number = data.get("line_number").and_then(Value::as_i64);
                let text = data
                    .get("lines")
                    .and_then(|lines| lines.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim_end_matches('\n')
                    .to_string();
                rows.push(json!({
                    "kind": "context",
                    "path": path,
                    "line": line_number,
                    "text": text,
                }));
            }
            _ => {}
        }
    }

    if output_mode == "count" {
        let mut rows = counts
            .into_iter()
            .map(|(path, count)| json!({ "path": path, "count": count }))
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            right["count"]
                .as_i64()
                .cmp(&left["count"].as_i64())
                .then_with(|| left["path"].as_str().cmp(&right["path"].as_str()))
        });
        let total_results = rows.len();
        let paged = rows
            .into_iter()
            .skip(offset as usize)
            .take(limit as usize)
            .collect::<Vec<_>>();
        return tool_result(json!({
            "success": true,
            "target": "content",
            "output_mode": "count",
            "pattern": pattern,
            "path": search_path.display().to_string(),
            "offset": offset,
            "limit": limit,
            "total_results": total_results,
            "returned": paged.len(),
            "results": paged,
        }));
    }

    if output_mode == "files_only" {
        let mut files = counts.into_keys().collect::<Vec<_>>();
        files.sort();
        let total_results = files.len();
        let paged = files
            .into_iter()
            .skip(offset as usize)
            .take(limit as usize)
            .map(|path| json!({ "path": path }))
            .collect::<Vec<_>>();
        return tool_result(json!({
            "success": true,
            "target": "content",
            "output_mode": "files_only",
            "pattern": pattern,
            "path": search_path.display().to_string(),
            "offset": offset,
            "limit": limit,
            "total_results": total_results,
            "returned": paged.len(),
            "results": paged,
        }));
    }

    let total_results = rows.len();
    let paged = rows
        .into_iter()
        .skip(offset as usize)
        .take(limit as usize)
        .collect::<Vec<_>>();
    tool_result(json!({
        "success": true,
        "target": "content",
        "output_mode": "content",
        "pattern": pattern,
        "path": search_path.display().to_string(),
        "offset": offset,
        "limit": limit,
        "total_results": total_results,
        "returned": paged.len(),
        "results": paged,
    }))
}

fn handle_session_search(args: &Value, runtime: &ToolRuntime) -> String {
    let limit = match bounded_integer(args, "limit", 3, 1, 5) {
        Ok(limit) => limit,
        Err(error) => return tool_error(error),
    };
    let query = optional_non_empty_string(args, "query");
    let role_filter = match optional_non_empty_string(args, "role_filter") {
        Some(value) => match parse_role_filter(&value) {
            Ok(roles) => roles,
            Err(error) => return tool_error(error),
        },
        None => Vec::new(),
    };

    let store = match SessionStore::open(runtime.hermes_home().join("state.db")) {
        Ok(store) => store,
        Err(error) => return tool_error(error.to_string()),
    };

    if query.is_none() {
        let rows = match store.search_sessions(None, limit, 0) {
            Ok(rows) => rows,
            Err(error) => return tool_error(error.to_string()),
        };
        let current = runtime.current_session_id();
        let sessions = rows
            .into_iter()
            .filter(|row| current != Some(row.id.as_str()))
            .take(limit as usize)
            .map(|row| {
                json!({
                    "session_id": row.id,
                    "source": row.source,
                    "model": row.model,
                    "title": row.title,
                    "started_at": row.started_at,
                    "ended_at": row.ended_at,
                    "message_count": row.message_count,
                    "preview": row.preview,
                })
            })
            .collect::<Vec<_>>();
        return tool_result(json!({
            "success": true,
            "mode": "browse",
            "sessions": sessions,
        }));
    }

    let query = query.unwrap_or_default();
    let role_filter_ref = (!role_filter.is_empty()).then_some(role_filter.as_slice());
    let rows = match store.search_messages(&query, None, None, role_filter_ref, limit * 10, 0) {
        Ok(rows) => rows,
        Err(error) => return tool_error(error.to_string()),
    };

    let current = runtime.current_session_id();
    let mut seen = HashSet::new();
    let mut sessions = Vec::new();
    for row in rows {
        if current == Some(row.session_id.as_str()) {
            continue;
        }
        if !seen.insert(row.session_id.clone()) {
            continue;
        }
        let title = match store.get_session(&row.session_id) {
            Ok(Some(session)) => session.title,
            Ok(None) | Err(_) => None,
        };
        sessions.push(json!({
            "session_id": row.session_id,
            "source": row.source,
            "model": row.model,
            "title": title,
            "message_id": row.id,
            "role": row.role,
            "session_started": row.session_started,
            "snippet": row.snippet,
            "context": row.context.iter().map(|item| {
                json!({
                    "role": item.role,
                    "content": item.content,
                })
            }).collect::<Vec<_>>(),
        }));
        if sessions.len() >= limit as usize {
            break;
        }
    }

    tool_result(json!({
        "success": true,
        "mode": "search",
        "query": query,
        "sessions": sessions,
    }))
}

fn patch_replace(
    raw_path: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
    runtime: &ToolRuntime,
) -> String {
    if old_string.is_empty() {
        return tool_error("old_string must not be empty");
    }
    let path = match runtime.resolve_path(raw_path) {
        Ok(path) => path,
        Err(error) => return tool_error(error),
    };
    let content = match read_utf8_file(&path, "patch") {
        Ok(content) => content,
        Err(error) => return tool_error(error),
    };

    let (updated, replacements) = match replace_exact(&content, old_string, new_string, replace_all)
    {
        Ok(result) => result,
        Err(error) => return tool_error(error),
    };
    if let Err(error) = write_text_file(&path, &updated, "patch") {
        return tool_error(error);
    }

    tool_result(json!({
        "success": true,
        "mode": "replace",
        "path": path.display().to_string(),
        "replacements": replacements,
        "bytes_written": updated.len(),
    }))
}

fn apply_v4a_patch(patch_text: &str, runtime: &ToolRuntime) -> String {
    let operations = match parse_v4a_patch(patch_text) {
        Ok(operations) => operations,
        Err(error) => return tool_error(error),
    };
    if operations.len() > MAX_PATCH_OPERATIONS {
        return tool_error(format!(
            "patch contains too many operations ({} > {MAX_PATCH_OPERATIONS})",
            operations.len()
        ));
    }

    let mut applied = Vec::new();
    for operation in operations {
        match apply_v4a_operation(operation, runtime) {
            Ok(result) => applied.push(result),
            Err(error) => return tool_error(error),
        }
    }

    tool_result(json!({
        "success": true,
        "mode": "patch",
        "operations": applied,
    }))
}

#[derive(Debug, Clone)]
enum PatchOperation {
    AddFile {
        path: String,
        content: String,
    },
    DeleteFile {
        path: String,
    },
    UpdateFile {
        path: String,
        move_to: Option<String>,
        hunks: Vec<PatchHunk>,
    },
}

#[derive(Debug, Clone)]
struct PatchHunk {
    old_lines: Vec<String>,
    new_lines: Vec<String>,
}

fn parse_v4a_patch(input: &str) -> Result<Vec<PatchOperation>, String> {
    let lines = input.lines().collect::<Vec<_>>();
    if lines.first().copied() != Some("*** Begin Patch") {
        return Err("patch must start with '*** Begin Patch'".to_string());
    }

    let mut operations = Vec::new();
    let mut index = 1usize;
    let mut found_end = false;

    while index < lines.len() {
        let line = lines[index];
        if line == "*** End Patch" {
            found_end = true;
            index += 1;
            break;
        }
        if line.is_empty() {
            index += 1;
            continue;
        }

        if let Some(path) = line.strip_prefix("*** Add File: ") {
            let path = sanitize_patch_path(path)?;
            index += 1;
            let mut content = Vec::new();
            while index < lines.len() {
                let line = lines[index];
                if line.starts_with("*** ") {
                    break;
                }
                let Some(text) = line.strip_prefix('+') else {
                    return Err(format!(
                        "add file block for '{}' contained a non-add line: {}",
                        path, line
                    ));
                };
                content.push(text.to_string());
                index += 1;
            }
            operations.push(PatchOperation::AddFile {
                path,
                content: content.join("\n"),
            });
            continue;
        }

        if let Some(path) = line.strip_prefix("*** Delete File: ") {
            operations.push(PatchOperation::DeleteFile {
                path: sanitize_patch_path(path)?,
            });
            index += 1;
            continue;
        }

        if let Some(path) = line.strip_prefix("*** Update File: ") {
            let path = sanitize_patch_path(path)?;
            index += 1;
            let mut move_to = None;
            if index < lines.len() {
                if let Some(target) = lines[index].strip_prefix("*** Move to: ") {
                    move_to = Some(sanitize_patch_path(target)?);
                    index += 1;
                }
            }

            let mut hunks = Vec::new();
            while index < lines.len() {
                let line = lines[index];
                if line == "*** End Patch"
                    || line.starts_with("*** Update File: ")
                    || line.starts_with("*** Add File: ")
                    || line.starts_with("*** Delete File: ")
                {
                    break;
                }
                if line == "*** End of File" {
                    index += 1;
                    continue;
                }
                if !line.starts_with("@@") {
                    return Err(format!("unexpected patch line in '{}': {}", path, line));
                }
                index += 1;

                let mut old_lines = Vec::new();
                let mut new_lines = Vec::new();
                while index < lines.len() {
                    let line = lines[index];
                    if line.starts_with("@@")
                        || line == "*** End Patch"
                        || line.starts_with("*** Update File: ")
                        || line.starts_with("*** Add File: ")
                        || line.starts_with("*** Delete File: ")
                    {
                        break;
                    }
                    if line == "*** End of File" {
                        index += 1;
                        break;
                    }
                    let (prefix, rest) = line.split_at(1);
                    match prefix {
                        " " => {
                            old_lines.push(rest.to_string());
                            new_lines.push(rest.to_string());
                        }
                        "-" => old_lines.push(rest.to_string()),
                        "+" => new_lines.push(rest.to_string()),
                        _ => {
                            return Err(format!(
                                "invalid patch line prefix '{}' in '{}'",
                                prefix, path
                            ));
                        }
                    }
                    index += 1;
                }
                hunks.push(PatchHunk {
                    old_lines,
                    new_lines,
                });
            }

            if hunks.is_empty() {
                return Err(format!("update block for '{}' had no hunks", path));
            }
            operations.push(PatchOperation::UpdateFile {
                path,
                move_to,
                hunks,
            });
            continue;
        }

        return Err(format!("unrecognized patch directive: {}", line));
    }

    if !found_end {
        return Err("patch must end with '*** End Patch'".to_string());
    }
    if index != lines.len() {
        return Err("patch contained trailing content after '*** End Patch'".to_string());
    }
    Ok(operations)
}

fn apply_v4a_operation(operation: PatchOperation, runtime: &ToolRuntime) -> Result<Value, String> {
    match operation {
        PatchOperation::AddFile { path, content } => {
            let path = runtime.resolve_path(&path)?;
            if path.exists() {
                return Err(format!(
                    "patch add failed: {} already exists",
                    path.display()
                ));
            }
            write_text_file(&path, &content, "patch add")?;
            Ok(json!({
                "action": "add",
                "path": path.display().to_string(),
                "bytes_written": content.len(),
            }))
        }
        PatchOperation::DeleteFile { path } => {
            let path = runtime.resolve_path(&path)?;
            if !path.exists() {
                return Err(format!(
                    "patch delete failed: {} does not exist",
                    path.display()
                ));
            }
            if !path.is_file() {
                return Err(format!(
                    "patch delete failed: {} is not a file",
                    path.display()
                ));
            }
            fs::remove_file(&path)
                .map_err(|error| format!("deleting {} failed: {error}", path.display()))?;
            Ok(json!({
                "action": "delete",
                "path": path.display().to_string(),
            }))
        }
        PatchOperation::UpdateFile {
            path,
            move_to,
            hunks,
        } => {
            let source_path = runtime.resolve_path(&path)?;
            let mut content = read_utf8_file(&source_path, "patch update")?;
            let mut applied_hunks = 0usize;
            for hunk in hunks {
                let old_block = hunk.old_lines.join("\n");
                let new_block = hunk.new_lines.join("\n");
                let (updated, replacements) =
                    replace_exact(&content, &old_block, &new_block, false)
                        .or_else(|_| {
                            if old_block.is_empty() {
                                Err("empty patch hunk is not supported".to_string())
                            } else {
                                let old_with_newline = format!("{old_block}\n");
                                let new_with_newline = format!("{new_block}\n");
                                replace_exact(&content, &old_with_newline, &new_with_newline, false)
                            }
                        })
                        .map_err(|_| {
                            format!(
                                "patch update failed: hunk did not match {}",
                                source_path.display()
                            )
                        })?;
                content = updated;
                applied_hunks += replacements;
            }

            if let Some(target) = move_to {
                let target_path = runtime.resolve_path(&target)?;
                if target_path != source_path && target_path.exists() {
                    return Err(format!(
                        "patch move failed: {} already exists",
                        target_path.display()
                    ));
                }
                write_text_file(&target_path, &content, "patch move")?;
                if target_path != source_path {
                    fs::remove_file(&source_path).map_err(|error| {
                        format!(
                            "removing {} after move failed: {error}",
                            source_path.display()
                        )
                    })?;
                }
                Ok(json!({
                    "action": "update",
                    "path": source_path.display().to_string(),
                    "moved_to": target_path.display().to_string(),
                    "hunks": applied_hunks,
                }))
            } else {
                write_text_file(&source_path, &content, "patch update")?;
                Ok(json!({
                    "action": "update",
                    "path": source_path.display().to_string(),
                    "hunks": applied_hunks,
                    "bytes_written": content.len(),
                }))
            }
        }
    }
}

fn sanitize_patch_path(path: &str) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("patch path must not be empty".to_string());
    }
    Ok(trimmed.to_string())
}

fn read_utf8_file(path: &Path, action: &str) -> Result<String, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("{action}: {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("{action}: {} is not a file", path.display()));
    }
    let bytes = fs::read(path).map_err(|error| format!("{action}: {}: {error}", path.display()))?;
    String::from_utf8(bytes)
        .map_err(|_| format!("{action}: {} is not a UTF-8 text file", path.display()))
}

fn write_text_file(path: &Path, content: &str, action: &str) -> Result<(), String> {
    if path.is_dir() {
        return Err(format!("{action}: {} is a directory", path.display()));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("{action}: creating {} failed: {error}", parent.display()))?;
    }
    fs::write(path, content.as_bytes())
        .map_err(|error| format!("{action}: writing {} failed: {error}", path.display()))
}

fn replace_exact(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<(String, usize), String> {
    if old.is_empty() {
        return Err("old_string must not be empty".to_string());
    }

    let matches = content.match_indices(old).collect::<Vec<_>>();
    if matches.is_empty() {
        return Err("old_string was not found".to_string());
    }
    if !replace_all && matches.len() > 1 {
        return Err(format!(
            "old_string matched {} locations; set replace_all=true or provide more context",
            matches.len()
        ));
    }

    let updated = if replace_all {
        content.replacen(old, new, matches.len())
    } else {
        content.replacen(old, new, 1)
    };
    let replacements = if replace_all { matches.len() } else { 1 };
    Ok((updated, replacements))
}

fn required_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} must be a string"))
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    let value = required_string(args, key)?;
    if value.trim().is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    Ok(value)
}

fn optional_non_empty_string(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn optional_bool(args: &Value, key: &str) -> Result<Option<bool>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(format!("{key} must be a boolean")),
    }
}

fn parse_role_filter(raw: &str) -> Result<Vec<String>, String> {
    let roles = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if roles.is_empty() {
        return Err("role_filter must contain at least one role".to_string());
    }
    Ok(roles)
}

fn bounded_integer(
    args: &Value,
    key: &str,
    default: i64,
    min: i64,
    max: i64,
) -> Result<i64, String> {
    let value = match args.get(key) {
        Some(value) => value
            .as_i64()
            .ok_or_else(|| format!("{key} must be an integer"))?,
        None => default,
    };
    if value < min || value > max {
        return Err(format!("{key} must be between {min} and {max}"));
    }
    Ok(value)
}

fn enum_arg(args: &Value, key: &str, default: &str, allowed: &[&str]) -> Result<String, String> {
    let value = args
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .trim()
        .to_string();
    if allowed.iter().any(|candidate| *candidate == value) {
        Ok(value)
    } else {
        Err(format!("{key} must be one of: {}", allowed.join(", ")))
    }
}

fn coerce_value(value: &str, expected_type: Option<&Value>, schema: &Value) -> Value {
    if schema_allows_null(schema) && value.trim().eq_ignore_ascii_case("null") {
        return Value::Null;
    }

    match expected_type {
        Some(Value::String(kind)) => match kind.as_str() {
            "integer" => coerce_number(value, true),
            "number" => coerce_number(value, false),
            "boolean" => coerce_boolean(value),
            "array" => coerce_json_array(value),
            "object" => coerce_json_object(value),
            "null" if value.trim().eq_ignore_ascii_case("null") => Value::Null,
            _ => Value::String(value.to_string()),
        },
        Some(Value::Array(kinds)) => {
            for kind in kinds {
                let coerced = coerce_value(value, Some(kind), schema);
                if coerced != Value::String(value.to_string()) {
                    return coerced;
                }
            }
            Value::String(value.to_string())
        }
        _ => Value::String(value.to_string()),
    }
}

fn schema_allows_null(schema: &Value) -> bool {
    if schema.get("type") == Some(&Value::String("null".to_string())) {
        return true;
    }
    if schema
        .get("type")
        .and_then(Value::as_array)
        .is_some_and(|types| types.iter().any(|value| value == "null"))
    {
        return true;
    }
    if schema.get("nullable").and_then(Value::as_bool) == Some(true) {
        return true;
    }

    for key in ["anyOf", "oneOf"] {
        if schema
            .get(key)
            .and_then(Value::as_array)
            .is_some_and(|items| {
                items
                    .iter()
                    .any(|item| item.get("type") == Some(&Value::String("null".to_string())))
            })
        {
            return true;
        }
    }
    false
}

fn coerce_number(value: &str, integer_only: bool) -> Value {
    let Ok(number) = value.parse::<f64>() else {
        return Value::String(value.to_string());
    };
    if !number.is_finite() {
        return Value::String(value.to_string());
    }
    if number.fract() == 0.0 {
        return json!(number as i64);
    }
    if integer_only {
        return Value::String(value.to_string());
    }
    json!(number)
}

fn coerce_boolean(value: &str) -> Value {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => Value::String(value.to_string()),
    }
}

fn coerce_json_array(value: &str) -> Value {
    let Ok(parsed) = serde_json::from_str::<Value>(value) else {
        return Value::String(value.to_string());
    };
    if parsed.is_array() {
        parsed
    } else {
        Value::String(value.to_string())
    }
}

fn coerce_json_object(value: &str) -> Value {
    let Ok(parsed) = serde_json::from_str::<Value>(value) else {
        return Value::String(value.to_string());
    };
    if parsed.is_object() {
        parsed
    } else {
        Value::String(value.to_string())
    }
}

fn parse_embedded_json(value: &Value) -> Option<Value> {
    match value {
        Value::Object(_) | Value::Array(_) => Some(value.clone()),
        Value::String(text) => serde_json::from_str::<Value>(text).ok(),
        _ => None,
    }
}

fn json_path_text(value: Option<&Value>) -> String {
    value
        .and_then(|value| value.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn ripgrep_available() -> bool {
    Command::new("rg")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn default_hermes_home() -> PathBuf {
    std::env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".hermes")))
        .unwrap_or_else(|| PathBuf::from(".hermes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use tempfile::TempDir;

    #[derive(Clone)]
    struct FakeContextEngine {
        calls: Arc<Mutex<Vec<(String, Value, usize)>>>,
    }

    impl crate::ContextEngine for FakeContextEngine {
        fn name(&self) -> &str {
            "lcm"
        }

        fn tool_definitions(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                name: "lcm_expand".to_string(),
                toolset: "context_engine".to_string(),
                description: "Expand prior context.".to_string(),
                emoji: "🧠".to_string(),
                schema: json!({
                    "name": "lcm_expand",
                    "description": "Expand context for a query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"]
                    }
                }),
            }]
        }

        fn handle_tool_call(&self, name: &str, args: &Value, messages: &[Value]) -> String {
            self.calls
                .lock()
                .unwrap()
                .push((name.to_string(), args.clone(), messages.len()));
            json!({
                "success": true,
                "engine": "lcm",
                "name": name,
                "query": args.get("query").cloned().unwrap_or(Value::Null),
            })
            .to_string()
        }
    }

    fn loaded_with_context_engine(name: &str) -> crate::LoadedConfig {
        crate::LoadedConfig {
            path: PathBuf::from("config.yaml"),
            raw: serde_yaml::from_str(&format!("context:\n  engine: {name}\n")).unwrap(),
            config: crate::HermesConfig::default(),
            warnings: Vec::new(),
        }
    }

    fn set_optional_env(name: &str, value: Option<&Path>) -> Option<std::ffi::OsString> {
        let previous = std::env::var_os(name);
        match value {
            Some(path) => unsafe { std::env::set_var(name, path) },
            None => unsafe { std::env::remove_var(name) },
        }
        previous
    }

    fn restore_optional_env(name: &str, value: Option<std::ffi::OsString>) {
        match value {
            Some(value) => unsafe { std::env::set_var(name, value) },
            None => unsafe { std::env::remove_var(name) },
        }
    }

    #[test]
    fn resolves_hermes_cli_to_file_tools() {
        assert_eq!(
            resolve_toolset("hermes-cli"),
            vec![
                "browser_back".to_string(),
                "browser_cdp".to_string(),
                "browser_click".to_string(),
                "browser_console".to_string(),
                "browser_dialog".to_string(),
                "browser_get_images".to_string(),
                "browser_navigate".to_string(),
                "browser_press".to_string(),
                "browser_scroll".to_string(),
                "browser_snapshot".to_string(),
                "browser_type".to_string(),
                "browser_vision".to_string(),
                "clarify".to_string(),
                "cronjob".to_string(),
                "delegate_task".to_string(),
                "execute_code".to_string(),
                "ha_call_service".to_string(),
                "ha_get_state".to_string(),
                "ha_list_entities".to_string(),
                "ha_list_services".to_string(),
                "image_generate".to_string(),
                "kanban_block".to_string(),
                "kanban_comment".to_string(),
                "kanban_complete".to_string(),
                "kanban_create".to_string(),
                "kanban_heartbeat".to_string(),
                "kanban_link".to_string(),
                "kanban_show".to_string(),
                "memory".to_string(),
                "patch".to_string(),
                "process".to_string(),
                "read_file".to_string(),
                "search_files".to_string(),
                "send_message".to_string(),
                "session_search".to_string(),
                "skill_manage".to_string(),
                "skill_view".to_string(),
                "skills_list".to_string(),
                "terminal".to_string(),
                "text_to_speech".to_string(),
                "todo".to_string(),
                "vision_analyze".to_string(),
                "web_extract".to_string(),
                "web_search".to_string(),
                "write_file".to_string(),
            ]
        );
    }

    #[test]
    fn yuanbao_toolset_is_opt_in() {
        assert_eq!(
            resolve_toolset("yuanbao"),
            vec![
                "yb_query_group_info".to_string(),
                "yb_query_group_members".to_string(),
                "yb_search_sticker".to_string(),
                "yb_send_dm".to_string(),
                "yb_send_sticker".to_string(),
            ]
        );
        assert!(!resolve_toolset("hermes-cli").contains(&"yb_search_sticker".to_string()));
    }

    #[test]
    fn rl_toolset_is_opt_in() {
        assert_eq!(
            resolve_toolset("rl"),
            vec![
                "rl_check_status".to_string(),
                "rl_edit_config".to_string(),
                "rl_get_current_config".to_string(),
                "rl_get_results".to_string(),
                "rl_list_environments".to_string(),
                "rl_list_runs".to_string(),
                "rl_select_environment".to_string(),
                "rl_start_training".to_string(),
                "rl_stop_training".to_string(),
                "rl_test_inference".to_string(),
            ]
        );
        assert!(!resolve_toolset("hermes-cli").contains(&"rl_list_environments".to_string()));
    }

    #[test]
    fn video_toolset_is_opt_in() {
        assert_eq!(resolve_toolset("video"), vec!["video_analyze".to_string()]);
        assert!(!resolve_toolset("hermes-cli").contains(&"video_analyze".to_string()));
    }

    #[test]
    fn moa_toolset_is_opt_in() {
        assert_eq!(
            resolve_toolset("moa"),
            vec!["mixture_of_agents".to_string()]
        );
        assert!(!resolve_toolset("hermes-cli").contains(&"mixture_of_agents".to_string()));
    }

    #[test]
    fn spotify_toolset_is_opt_in() {
        assert_eq!(
            resolve_toolset("spotify"),
            vec![
                "spotify_albums".to_string(),
                "spotify_devices".to_string(),
                "spotify_library".to_string(),
                "spotify_playback".to_string(),
                "spotify_playlists".to_string(),
                "spotify_queue".to_string(),
                "spotify_search".to_string(),
            ]
        );
        assert!(!resolve_toolset("hermes-cli").contains(&"spotify_playback".to_string()));
    }

    #[test]
    fn feishu_toolsets_are_opt_in() {
        assert_eq!(
            resolve_toolset("feishu_doc"),
            vec!["feishu_doc_read".to_string()]
        );
        assert_eq!(
            resolve_toolset("feishu_drive"),
            vec![
                "feishu_drive_add_comment".to_string(),
                "feishu_drive_list_comment_replies".to_string(),
                "feishu_drive_list_comments".to_string(),
                "feishu_drive_reply_comment".to_string(),
            ]
        );
        assert!(!resolve_toolset("hermes-cli").contains(&"feishu_doc_read".to_string()));
    }

    #[test]
    fn coerces_string_arguments_from_schema() {
        let args = json!({
            "path": "demo.txt",
            "offset": "12",
            "limit": "4",
            "context": "2",
        });
        let coerced = coerce_tool_args("search_files", args);
        assert_eq!(coerced["offset"], json!(12));
        assert_eq!(coerced["limit"], json!(4));
        assert_eq!(coerced["context"], json!(2));
    }

    #[test]
    fn dynamic_runtime_can_be_inherited() {
        let parent = ToolRuntime::default()
            .with_dynamic_tools(
                vec![ToolDefinition {
                    name: "demo_tool".to_string(),
                    toolset: "demo".to_string(),
                    description: "Demo".to_string(),
                    emoji: "plug".to_string(),
                    schema: json!({
                        "name": "demo_tool",
                        "description": "Demo",
                        "parameters": {"type": "object", "properties": {}},
                    }),
                }],
                BTreeMap::from([(String::from("demo"), vec![String::from("demo_tool")])]),
            )
            .with_dynamic_tool_dispatch(|name, args, _runtime| {
                (name == "demo_tool").then(|| {
                    json!({
                        "success": true,
                        "echo": args.get("message").cloned().unwrap_or(Value::Null),
                    })
                    .to_string()
                })
            })
            .with_hook_invoke_callback(|hook_name, _payload, _runtime| {
                if hook_name == "pre_llm_call" {
                    vec![json!({"context": "from parent"})]
                } else {
                    Vec::new()
                }
            });
        let child = ToolRuntime::new(".").with_dynamic_runtime_from(&parent);

        assert_eq!(
            child.resolve_dynamic_toolset("demo"),
            vec![String::from("demo_tool")]
        );
        let result = child
            .dispatch_dynamic_tool("demo_tool", &json!({"message": "hello"}))
            .unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.get("echo"), Some(&json!("hello")));
        let hook_results = child.invoke_hook("pre_llm_call", &json!({}));
        assert_eq!(hook_results, vec![json!({"context": "from parent"})]);
    }

    #[test]
    fn writes_then_reads_file() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path());

        let write = dispatch_tool(
            "write_file",
            json!({
                "path": "notes/test.txt",
                "content": "alpha\nbeta\ngamma\n",
            }),
            &runtime,
        );
        let write_json: Value = serde_json::from_str(&write).unwrap();
        assert_eq!(write_json["success"], Value::Bool(true));

        let read = dispatch_tool(
            "read_file",
            json!({
                "path": "notes/test.txt",
                "offset": 2,
                "limit": 2,
            }),
            &runtime,
        );
        let read_json: Value = serde_json::from_str(&read).unwrap();
        assert_eq!(read_json["success"], Value::Bool(true));
        assert_eq!(read_json["returned_lines"], json!(2));
        assert_eq!(read_json["content"], json!("2|beta\n3|gamma"));
    }

    #[test]
    fn rejects_invalid_read_ranges() {
        let runtime = ToolRuntime::default();
        let result = dispatch_tool(
            "read_file",
            json!({
                "path": "Cargo.toml",
                "limit": 5001,
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["error"]
                .as_str()
                .unwrap()
                .contains("limit must be between 1 and 2000")
        );
    }

    #[test]
    fn searches_file_content_and_names() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path());
        fs::create_dir_all(temp.path().join("src")).unwrap();
        fs::write(
            temp.path().join("src/main.rs"),
            "fn main() {\n    println!(\"needle\");\n}\n",
        )
        .unwrap();
        fs::write(temp.path().join("README.txt"), "needle in docs\n").unwrap();

        let content = dispatch_tool(
            "search_files",
            json!({
                "pattern": "needle",
                "path": temp.path().display().to_string(),
                "output_mode": "files_only",
            }),
            &runtime,
        );
        let content_json: Value = serde_json::from_str(&content).unwrap();
        if content_json.get("error").is_none() {
            assert_eq!(content_json["success"], Value::Bool(true));
            assert!(content_json["returned"].as_i64().unwrap() >= 1);
        }

        let files = dispatch_tool(
            "search_files",
            json!({
                "pattern": "*.rs",
                "target": "files",
                "path": temp.path().display().to_string(),
            }),
            &runtime,
        );
        let files_json: Value = serde_json::from_str(&files).unwrap();
        if files_json.get("error").is_none() {
            assert_eq!(files_json["success"], Value::Bool(true));
            assert_eq!(files_json["returned"], json!(1));
        }
    }

    #[test]
    fn patch_replace_updates_existing_text() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path());
        fs::write(temp.path().join("demo.txt"), "alpha\nbeta\ngamma\n").unwrap();

        let result = dispatch_tool(
            "patch",
            json!({
                "mode": "replace",
                "path": "demo.txt",
                "old_string": "beta",
                "new_string": "delta",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], Value::Bool(true));
        assert_eq!(
            fs::read_to_string(temp.path().join("demo.txt")).unwrap(),
            "alpha\ndelta\ngamma\n"
        );
    }

    #[test]
    fn patch_v4a_can_add_update_and_delete_files() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path());
        fs::write(temp.path().join("demo.txt"), "one\ntwo\nthree\n").unwrap();
        fs::write(temp.path().join("remove.txt"), "obsolete\n").unwrap();

        let patch = "\
*** Begin Patch
*** Add File: added.txt
+hello
+world
*** Update File: demo.txt
@@
 one
-two
+TWO
 three
*** Delete File: remove.txt
*** End Patch";
        let result = dispatch_tool("patch", json!({"mode": "patch", "patch": patch}), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], Value::Bool(true));
        assert_eq!(
            fs::read_to_string(temp.path().join("added.txt")).unwrap(),
            "hello\nworld"
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("demo.txt")).unwrap(),
            "one\nTWO\nthree\n"
        );
        assert!(!temp.path().join("remove.txt").exists());
    }

    #[test]
    fn dispatch_tool_blocks_when_shell_hook_denies_call() {
        let _guard = crate::test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context =
            crate::HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let script = temp.path().join("block.sh");
        fs::write(
            &script,
            "#!/usr/bin/env bash\ncat >/dev/null\nprintf '{\"decision\":\"block\",\"reason\":\"blocked by rust hook\"}\\n'\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let loaded = crate::LoadedConfig {
            path: context.config_path(),
            raw: serde_yaml::from_str(&format!(
                "hooks:\n  pre_tool_call:\n    - command: \"{}\"\n      matcher: \"^read_file$\"\n",
                script.display()
            ))
            .unwrap(),
            config: crate::HermesConfig::default(),
            warnings: Vec::new(),
        };

        unsafe { std::env::set_var("HERMES_ACCEPT_HOOKS", "1") };
        let mut runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        runtime.load_shell_hooks(&context, &loaded);
        let target = temp.path().join("notes.txt");
        fs::write(&target, "hello\n").unwrap();

        let result = dispatch_tool(
            "read_file",
            json!({ "path": target.display().to_string() }),
            &runtime,
        );
        unsafe { std::env::remove_var("HERMES_ACCEPT_HOOKS") };

        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["error"], json!("blocked by rust hook"));
    }

    #[test]
    fn dispatch_tool_emits_post_tool_shell_hook_payload() {
        let _guard = crate::test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context =
            crate::HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().into()));
        context.ensure_hermes_home().unwrap();
        let capture = temp.path().join("payload.json");
        let script = temp.path().join("capture.sh");
        fs::write(
            &script,
            format!(
                "#!/usr/bin/env bash\ncat - > {}\nprintf '{{}}\\n'\n",
                capture.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script, perms).unwrap();
        }
        let loaded = crate::LoadedConfig {
            path: context.config_path(),
            raw: serde_yaml::from_str(&format!(
                "hooks:\n  post_tool_call:\n    - command: \"{}\"\n      matcher: \"^read_file$\"\n",
                script.display()
            ))
            .unwrap(),
            config: crate::HermesConfig::default(),
            warnings: Vec::new(),
        };

        unsafe { std::env::set_var("HERMES_ACCEPT_HOOKS", "1") };
        let mut runtime = ToolRuntime::new(temp.path())
            .with_hermes_home(temp.path())
            .with_current_session_id(Some("session_123".to_string()));
        runtime.load_shell_hooks(&context, &loaded);
        let target = temp.path().join("notes.txt");
        fs::write(&target, "hello\n").unwrap();

        let result = dispatch_tool(
            "read_file",
            json!({ "path": target.display().to_string() }),
            &runtime,
        );
        unsafe { std::env::remove_var("HERMES_ACCEPT_HOOKS") };

        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], Value::Bool(true));
        let payload: Value = serde_json::from_str(&fs::read_to_string(capture).unwrap()).unwrap();
        assert_eq!(payload["hook_event_name"], json!("post_tool_call"));
        assert_eq!(payload["tool_name"], json!("read_file"));
        assert_eq!(payload["session_id"], json!("session_123"));
        assert_eq!(
            payload["tool_input"]["path"],
            json!(target.display().to_string())
        );
        assert!(payload["extra"]["duration_ms"].as_u64().unwrap() <= 60_000);
    }

    #[test]
    fn dispatch_tool_emits_progress_updates() {
        let temp = TempDir::new().unwrap();
        let events = Arc::new(Mutex::new(Vec::<ToolProgressUpdate>::new()));
        let capture = Arc::clone(&events);
        let runtime = ToolRuntime::new(temp.path()).with_tool_progress_callback(move |update| {
            capture.lock().unwrap().push(update.clone())
        });
        let target = temp.path().join("notes.txt");
        fs::write(&target, "hello\n").unwrap();

        let result = dispatch_tool(
            "read_file",
            json!({ "path": target.display().to_string() }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], Value::Bool(true));

        let events = events.lock().unwrap().clone();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "tool.started");
        assert_eq!(events[0].function_name.as_deref(), Some("read_file"));
        assert_eq!(
            events[0].function_args.as_ref().unwrap()["path"],
            json!(target.display().to_string())
        );
        assert_eq!(events[0].preview, Some(target.display().to_string()));
        assert_eq!(events[1].event_type, "tool.completed");
        assert_eq!(events[1].function_name.as_deref(), Some("read_file"));
        assert!(events[1].duration_ms.unwrap() <= 60_000);
        assert_eq!(events[1].is_error, Some(false));
    }

    #[test]
    fn context_engine_tools_are_injected_only_when_selected() {
        let calls = Arc::new(Mutex::new(Vec::<(String, Value, usize)>::new()));
        let mut runtime = ToolRuntime::default().with_context_engine(FakeContextEngine {
            calls: Arc::clone(&calls),
        });
        runtime.load_context_engine(&loaded_with_context_engine("lcm"));
        let with_engine =
            get_tool_definitions_for_runtime(&runtime, Some(&["hermes-cli".to_string()]), None);
        assert!(with_engine.iter().any(|tool| tool.name == "lcm_expand"));

        let mut without_runtime =
            ToolRuntime::default().with_context_engine(FakeContextEngine { calls });
        without_runtime.load_context_engine(&loaded_with_context_engine("compressor"));
        let without_engine = get_tool_definitions_for_runtime(
            &without_runtime,
            Some(&["hermes-cli".to_string()]),
            None,
        );
        assert!(!without_engine.iter().any(|tool| tool.name == "lcm_expand"));
    }

    #[test]
    fn dispatch_tool_routes_context_engine_calls() {
        let events = Arc::new(Mutex::new(Vec::<ToolProgressUpdate>::new()));
        let events_capture = Arc::clone(&events);
        let calls = Arc::new(Mutex::new(Vec::<(String, Value, usize)>::new()));
        let mut runtime = ToolRuntime::default()
            .with_context_engine(FakeContextEngine {
                calls: Arc::clone(&calls),
            })
            .with_tool_progress_callback(move |update| {
                events_capture.lock().unwrap().push(update.clone())
            });
        runtime.load_context_engine(&loaded_with_context_engine("lcm"));

        let result = dispatch_tool("lcm_expand", json!({ "query": "alpha" }), &runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["engine"], json!("lcm"));
        assert_eq!(parsed["query"], json!("alpha"));

        let calls = calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "lcm_expand");
        assert_eq!(calls[0].1["query"], json!("alpha"));
        assert_eq!(calls[0].2, 0);

        let events = events.lock().unwrap().clone();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "tool.started");
        assert_eq!(events[0].function_name.as_deref(), Some("lcm_expand"));
        assert_eq!(events[1].event_type, "tool.completed");
        assert_eq!(events[1].is_error, Some(false));
    }

    #[test]
    fn execute_code_runs_python_and_calls_allowed_tools() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path())
            .with_hermes_home(temp.path())
            .with_available_tool_names([
                "execute_code",
                "read_file",
                "write_file",
                "search_files",
                "patch",
                "terminal",
            ]);

        let result = dispatch_tool(
            "execute_code",
            json!({
                "code": "from hermes_tools import write_file, read_file\nwrite_file('sandbox.txt', 'hello from sandbox')\nprint(read_file('sandbox.txt')['content'])"
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["status"], json!("success"));
        assert!(
            parsed["output"]
                .as_str()
                .unwrap()
                .contains("1|hello from sandbox")
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("sandbox.txt")).unwrap(),
            "hello from sandbox"
        );
    }

    #[test]
    fn built_in_toolsets_merge_plugin_registered_tools() {
        let _lock = crate::test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled_plugin = temp.path().join("bundled").join("demo-plugin");
        fs::create_dir_all(&bundled_plugin).unwrap();
        fs::write(
            bundled_plugin.join("plugin.yaml"),
            "name: demo-plugin\nversion: 0.1.0\ndescription: demo\n",
        )
        .unwrap();
        fs::write(
            bundled_plugin.join("__init__.py"),
            r#"
def register(ctx):
    ctx.register_tool(
        name="web_search_plus",
        toolset="web",
        schema={},
        handler=None,
    )
"#,
        )
        .unwrap();

        let old_bundled =
            set_optional_env("HERMES_BUNDLED_PLUGINS", Some(&temp.path().join("bundled")));
        let old_home = set_optional_env("HERMES_HOME", Some(&temp.path().join(".hermes")));
        reset_dynamic_tool_registry_for_tests();

        assert!(resolve_toolset("web").contains(&"web_search_plus".to_string()));

        restore_optional_env("HERMES_BUNDLED_PLUGINS", old_bundled);
        restore_optional_env("HERMES_HOME", old_home);
        reset_dynamic_tool_registry_for_tests();
    }

    #[test]
    fn plugin_only_toolsets_are_discovered_from_manifest_and_register_calls() {
        let _lock = crate::test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled_plugin = temp.path().join("bundled").join("google_meet");
        fs::create_dir_all(&bundled_plugin).unwrap();
        fs::write(
            bundled_plugin.join("plugin.yaml"),
            "name: google_meet\nprovides_tools:\n  - meet_join\n  - meet_leave\n",
        )
        .unwrap();
        fs::write(
            bundled_plugin.join("__init__.py"),
            r#"
def register(ctx):
    for name in ("meet_join", "meet_leave"):
        ctx.register_tool(
            name=name,
            toolset="google_meet",
            schema={},
            handler=None,
        )
"#,
        )
        .unwrap();

        let old_bundled =
            set_optional_env("HERMES_BUNDLED_PLUGINS", Some(&temp.path().join("bundled")));
        let old_home = set_optional_env("HERMES_HOME", Some(&temp.path().join(".hermes")));
        reset_dynamic_tool_registry_for_tests();

        assert!(validate_toolset("google_meet"));
        assert_eq!(
            resolve_toolset("google_meet"),
            vec!["meet_join".to_string(), "meet_leave".to_string()]
        );
        let info = get_toolset_info("google_meet").unwrap();
        assert_eq!(info.implemented_tools, Vec::<String>::new());
        assert!(!info.available);

        restore_optional_env("HERMES_BUNDLED_PLUGINS", old_bundled);
        restore_optional_env("HERMES_HOME", old_home);
        reset_dynamic_tool_registry_for_tests();
    }

    #[test]
    fn dynamic_toolsets_reuse_builtin_availability_checks_for_known_tools() {
        let _lock = crate::test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled_plugin = temp.path().join("bundled").join("search-alias");
        fs::create_dir_all(&bundled_plugin).unwrap();
        fs::write(
            bundled_plugin.join("plugin.yaml"),
            "name: search-alias\nversion: 0.1.0\ndescription: alias\n",
        )
        .unwrap();
        fs::write(
            bundled_plugin.join("__init__.py"),
            r#"
def register(ctx):
    ctx.register_tool(
        name="web_search",
        toolset="plugin_search",
        schema={},
        handler=None,
    )
"#,
        )
        .unwrap();

        let old_bundled =
            set_optional_env("HERMES_BUNDLED_PLUGINS", Some(&temp.path().join("bundled")));
        let old_home = set_optional_env("HERMES_HOME", Some(&temp.path().join(".hermes")));
        reset_dynamic_tool_registry_for_tests();

        let info = get_toolset_info("plugin_search").unwrap();
        assert_eq!(info.resolved_tools, vec!["web_search".to_string()]);
        assert_eq!(info.implemented_tools, vec!["web_search".to_string()]);
        assert!(info.available);

        restore_optional_env("HERMES_BUNDLED_PLUGINS", old_bundled);
        restore_optional_env("HERMES_HOME", old_home);
        reset_dynamic_tool_registry_for_tests();
    }

    #[test]
    fn todo_reads_writes_and_merges_items() {
        let runtime = ToolRuntime::default();

        let write = dispatch_tool(
            "todo",
            json!({
                "todos": [
                    {"id": "1", "content": "plan", "status": "pending"},
                    {"id": "2", "content": "build", "status": "in_progress"},
                    {"id": "1", "content": "plan better", "status": "completed"}
                ]
            }),
            &runtime,
        );
        let write_json: Value = serde_json::from_str(&write).unwrap();
        let todos = write_json["todos"].as_array().unwrap();
        assert_eq!(todos.len(), 2);
        assert_eq!(todos[0]["id"], json!("2"));
        assert_eq!(todos[1]["id"], json!("1"));
        assert_eq!(write_json["summary"]["completed"], json!(1));
        assert_eq!(write_json["summary"]["in_progress"], json!(1));

        let merge = dispatch_tool(
            "todo",
            json!({
                "todos": [
                    {"id": "2", "status": "completed"},
                    {"id": "3", "content": "ship", "status": "pending"}
                ],
                "merge": true
            }),
            &runtime,
        );
        let merge_json: Value = serde_json::from_str(&merge).unwrap();
        let todos = merge_json["todos"].as_array().unwrap();
        assert_eq!(todos.len(), 3);
        assert_eq!(todos[0]["id"], json!("2"));
        assert_eq!(todos[0]["status"], json!("completed"));
        assert_eq!(todos[2]["id"], json!("3"));
        assert_eq!(merge_json["summary"]["completed"], json!(2));

        let read = dispatch_tool("todo", json!({}), &runtime);
        let read_json: Value = serde_json::from_str(&read).unwrap();
        assert_eq!(read_json["todos"], merge_json["todos"]);
    }

    #[test]
    fn memory_tool_persists_entries_to_profile_memory_files() {
        let temp = TempDir::new().unwrap();
        let mut runtime = ToolRuntime::new(temp.path()).with_hermes_home(temp.path());
        runtime
            .load_memory_store(&crate::MemoryConfig::default())
            .unwrap();

        let add = dispatch_tool(
            "memory",
            json!({
                "action": "add",
                "target": "memory",
                "content": "Project uses ripgrep by default."
            }),
            &runtime,
        );
        let add_json: Value = serde_json::from_str(&add).unwrap();
        assert_eq!(add_json["success"], Value::Bool(true));
        assert!(temp.path().join("memories/MEMORY.md").exists());

        let replace = dispatch_tool(
            "memory",
            json!({
                "action": "replace",
                "target": "memory",
                "old_text": "ripgrep",
                "content": "Project prefers rg for searches."
            }),
            &runtime,
        );
        let replace_json: Value = serde_json::from_str(&replace).unwrap();
        assert_eq!(replace_json["entry_count"], json!(1));
        let content = fs::read_to_string(temp.path().join("memories/MEMORY.md")).unwrap();
        assert!(content.contains("Project prefers rg for searches."));

        let remove = dispatch_tool(
            "memory",
            json!({
                "action": "remove",
                "target": "memory",
                "old_text": "prefers rg"
            }),
            &runtime,
        );
        let remove_json: Value = serde_json::from_str(&remove).unwrap();
        assert_eq!(remove_json["entry_count"], json!(0));
        assert_eq!(
            fs::read_to_string(temp.path().join("memories/MEMORY.md")).unwrap_or_default(),
            ""
        );
    }

    #[test]
    fn todo_hydrates_from_prior_tool_messages() {
        let runtime = ToolRuntime::default();
        runtime.hydrate_todo_from_messages(&[
            crate::MessageRecord {
                id: 1,
                session_id: "session_1".to_string(),
                role: "tool".to_string(),
                content: Some(Value::String(
                    json!({
                        "todos": [
                            {"id": "1", "content": "restored", "status": "in_progress"}
                        ]
                    })
                    .to_string(),
                )),
                tool_call_id: Some("call_1".to_string()),
                tool_calls: None,
                tool_name: Some("todo".to_string()),
                timestamp: 1.0,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
            crate::MessageRecord {
                id: 2,
                session_id: "session_1".to_string(),
                role: "tool".to_string(),
                content: Some(Value::String("{\"ignored\":true}".to_string())),
                tool_call_id: Some("call_2".to_string()),
                tool_calls: None,
                tool_name: Some("write_file".to_string()),
                timestamp: 2.0,
                token_count: None,
                finish_reason: None,
                reasoning: None,
                reasoning_content: None,
                reasoning_details: None,
                codex_reasoning_items: None,
                codex_message_items: None,
            },
        ]);

        let read = dispatch_tool("todo", json!({}), &runtime);
        let read_json: Value = serde_json::from_str(&read).unwrap();
        let todos = read_json["todos"].as_array().unwrap();
        assert_eq!(todos.len(), 1);
        assert_eq!(todos[0]["content"], json!("restored"));
        assert_eq!(todos[0]["status"], json!("in_progress"));
    }

    #[test]
    fn session_search_browses_and_searches_prior_sessions() {
        let temp = TempDir::new().unwrap();
        let runtime = ToolRuntime::new(temp.path())
            .with_hermes_home(temp.path())
            .with_current_session_id(Some("session_current".to_string()));
        let store = SessionStore::open(temp.path().join("state.db")).unwrap();

        store
            .create_session(&crate::SessionCreate {
                id: "session_old".to_string(),
                source: "rust-agent".to_string(),
                user_id: None,
                model: Some("test-model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .append_message(
                "session_old",
                &crate::MessageAppend {
                    role: "user".to_string(),
                    content: Some(Value::String("alpha beta".to_string())),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();

        store
            .create_session(&crate::SessionCreate {
                id: "session_current".to_string(),
                source: "rust-agent".to_string(),
                user_id: None,
                model: Some("test-model".to_string()),
                model_config: None,
                system_prompt: None,
                parent_session_id: None,
            })
            .unwrap();
        store
            .append_message(
                "session_current",
                &crate::MessageAppend {
                    role: "user".to_string(),
                    content: Some(Value::String("current only".to_string())),
                    tool_call_id: None,
                    tool_calls: None,
                    tool_name: None,
                    token_count: None,
                    finish_reason: None,
                    reasoning: None,
                    reasoning_content: None,
                    reasoning_details: None,
                    codex_reasoning_items: None,
                    codex_message_items: None,
                },
            )
            .unwrap();

        let browse = dispatch_tool("session_search", json!({ "limit": 3 }), &runtime);
        let browse_json: Value = serde_json::from_str(&browse).unwrap();
        assert_eq!(browse_json["success"], Value::Bool(true));
        let browse_sessions = browse_json["sessions"].as_array().unwrap();
        assert_eq!(browse_sessions.len(), 1);
        assert_eq!(browse_sessions[0]["session_id"], json!("session_old"));

        let search = dispatch_tool(
            "session_search",
            json!({
                "query": "alpha OR beta",
                "role_filter": "user",
                "limit": 3
            }),
            &runtime,
        );
        let search_json: Value = serde_json::from_str(&search).unwrap();
        assert_eq!(search_json["success"], Value::Bool(true));
        let search_sessions = search_json["sessions"].as_array().unwrap();
        assert_eq!(search_sessions.len(), 1);
        assert_eq!(search_sessions[0]["session_id"], json!("session_old"));
    }
}
