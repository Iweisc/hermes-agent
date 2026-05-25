use std::collections::BTreeSet;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

use serde_json::{Value, json};

use crate::{
    HermesContext, HermesError, LoadedConfig, ModelOverrides, ToolRuntime, get_toolset_info,
    get_toolset_names, validate_toolset,
};

const BLOCKED_LEAF_TOOLSETS: &[&str] = &["delegation", "clarify", "memory", "code_execution"];
const BLOCKED_ORCHESTRATOR_TOOLSETS: &[&str] = &["clarify", "memory", "code_execution"];

#[derive(Debug, Clone)]
pub struct DelegateTaskRequest {
    pub goal: Option<String>,
    pub context: Option<String>,
    pub toolsets: Option<Vec<String>>,
    pub tasks: Option<Vec<DelegateTaskSpec>>,
    pub role: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DelegateTaskSpec {
    pub goal: String,
    pub context: Option<String>,
    pub toolsets: Option<Vec<String>>,
    pub role: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DelegateExecutor {
    context: HermesContext,
    loaded: LoadedConfig,
    source: String,
    parent_toolsets: Vec<String>,
    overrides: ModelOverrides,
    cwd: PathBuf,
    depth: usize,
    runtime_template: Option<ToolRuntime>,
}

impl DelegateExecutor {
    pub fn new(
        context: HermesContext,
        loaded: LoadedConfig,
        source: impl Into<String>,
        parent_toolsets: Vec<String>,
        overrides: ModelOverrides,
        cwd: impl Into<PathBuf>,
    ) -> Self {
        Self {
            context,
            loaded,
            source: source.into(),
            parent_toolsets,
            overrides,
            cwd: cwd.into(),
            depth: 0,
            runtime_template: None,
        }
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_runtime_template(mut self, runtime: ToolRuntime) -> Self {
        self.runtime_template = Some(runtime);
        self
    }

    pub fn execute(
        &self,
        request: DelegateTaskRequest,
        parent_runtime: &ToolRuntime,
    ) -> Result<Value, String> {
        let config = &self.loaded.config.delegation;
        let max_spawn_depth = config.max_spawn_depth.max(1) as usize;
        if self.depth >= max_spawn_depth {
            return Err(format!(
                "Delegation depth limit reached (depth={}, max_spawn_depth={}).",
                self.depth, max_spawn_depth
            ));
        }

        let top_role = normalize_role(request.role.as_deref())?;
        let mut tasks = if let Some(tasks) = request.tasks {
            if tasks.is_empty() {
                return Err("No tasks provided.".to_string());
            }
            tasks
        } else if let Some(goal) = request
            .goal
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            vec![DelegateTaskSpec {
                goal: goal.to_string(),
                context: request.context,
                toolsets: request.toolsets,
                role: Some(top_role.to_string()),
            }]
        } else {
            return Err("Provide either 'goal' (single task) or 'tasks' (batch).".to_string());
        };

        let max_children = config.max_concurrent_children.max(1) as usize;
        if tasks.len() > max_children {
            return Err(format!(
                "Too many tasks: {} provided, but max_concurrent_children is {}.",
                tasks.len(),
                max_children
            ));
        }
        for (index, task) in tasks.iter().enumerate() {
            if task.goal.trim().is_empty() {
                return Err(format!("Task {index} is missing a 'goal'."));
            }
            if let Some(role) = task.role.as_deref() {
                normalize_role(Some(role))?;
            }
            if let Some(toolsets) = task.toolsets.as_ref() {
                validate_requested_toolsets(toolsets)?;
            }
        }

        let overall_start = Instant::now();
        let results = if tasks.len() == 1 {
            vec![self.run_task(0, tasks.swap_remove(0), top_role)]
        } else {
            let mut handles = Vec::new();
            for (index, task) in tasks.into_iter().enumerate() {
                let executor = self.clone();
                let top_role = top_role.to_string();
                handles.push(thread::spawn(move || {
                    executor.run_task(index, task, &top_role)
                }));
            }
            let mut results = Vec::new();
            for handle in handles {
                match handle.join() {
                    Ok(result) => results.push(result),
                    Err(_) => results.push(json!({
                        "task_index": results.len(),
                        "status": "error",
                        "summary": Value::Null,
                        "error": "Delegated child thread panicked.",
                        "api_calls": 0,
                        "tool_calls": 0,
                        "duration_seconds": 0.0,
                    })),
                }
            }
            results.sort_by_key(|value| {
                value
                    .get("task_index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default()
            });
            results
        };

        let mut public_results = results;
        emit_subagent_stop_hooks(parent_runtime, &mut public_results);

        Ok(json!({
            "results": public_results,
            "total_duration_seconds": overall_start.elapsed().as_secs_f64(),
        }))
    }

    fn run_task(&self, task_index: usize, task: DelegateTaskSpec, top_role: &str) -> Value {
        let start = Instant::now();
        let requested_role = task.role.as_deref().unwrap_or(top_role);
        let child_depth = self.depth + 1;
        let role = effective_role(
            requested_role,
            child_depth,
            self.loaded.config.delegation.max_spawn_depth.max(1) as usize,
            self.loaded.config.delegation.orchestrator_enabled,
        );
        let child_toolsets = match self.child_toolsets(task.toolsets.as_ref(), role) {
            Ok(toolsets) => toolsets,
            Err(error) => {
                return json!({
                    "task_index": task_index,
                    "status": "error",
                    "summary": Value::Null,
                    "error": error,
                    "api_calls": 0,
                    "tool_calls": 0,
                    "duration_seconds": start.elapsed().as_secs_f64(),
                });
            }
        };
        let child_prompt = build_child_prompt(&task.goal, task.context.as_deref());
        let child_overrides = self.child_overrides();
        let child_runtime = self.child_runtime(&child_toolsets, child_overrides.clone(), role);
        let mut child_loaded = self.loaded.clone();
        child_loaded.config.agent.max_turns = self.loaded.config.delegation.max_iterations.max(1);
        let session_store = match self.context.open_session_store() {
            Ok(store) => store,
            Err(error) => {
                return json!({
                    "task_index": task_index,
                    "status": "error",
                    "summary": Value::Null,
                    "error": error_string(error),
                    "api_calls": 0,
                    "tool_calls": 0,
                    "duration_seconds": start.elapsed().as_secs_f64(),
                });
            }
        };
        match self.context.run_chat_completions_turn(
            &child_loaded,
            &child_prompt,
            &child_runtime,
            Some(&child_toolsets),
            &child_overrides,
            None,
            Some(&session_store),
        ) {
            Ok(result) => json!({
                "task_index": task_index,
                "status": "completed",
                "summary": result.final_response,
                "api_calls": result.api_calls,
                "tool_calls": result.tool_calls,
                "duration_seconds": start.elapsed().as_secs_f64(),
                "model": result.model,
                "session_id": result.session_id,
                "exit_reason": "completed",
                "_child_role": role,
            }),
            Err(error) => json!({
                "task_index": task_index,
                "status": "error",
                "summary": Value::Null,
                "error": error_string(error),
                "api_calls": 0,
                "tool_calls": 0,
                "duration_seconds": start.elapsed().as_secs_f64(),
                "_child_role": role,
            }),
        }
    }

    fn child_runtime(
        &self,
        child_toolsets: &[String],
        child_overrides: ModelOverrides,
        role: &str,
    ) -> ToolRuntime {
        let mut runtime =
            ToolRuntime::new(self.cwd.clone()).with_hermes_home(self.context.hermes_home());
        if let Some(template) = self.runtime_template.as_ref() {
            runtime = runtime.with_dynamic_runtime_from(template);
        } else if let Ok(attached) =
            crate::attach_python_plugin_runtime(&self.context.hermes_home(), runtime.clone())
        {
            runtime = attached;
        }
        if role == "orchestrator" {
            let child_executor = Self {
                context: self.context.clone(),
                loaded: self.loaded.clone(),
                source: self.source.clone(),
                parent_toolsets: child_toolsets.to_vec(),
                overrides: child_overrides,
                cwd: self.cwd.clone(),
                depth: self.depth + 1,
                runtime_template: self.runtime_template.clone(),
            };
            runtime = runtime.with_delegate_callback(move |request, parent_runtime| {
                child_executor.execute(request, parent_runtime)
            });
        }
        runtime
    }

    fn child_overrides(&self) -> ModelOverrides {
        let delegation = &self.loaded.config.delegation;
        ModelOverrides {
            model: non_empty(&delegation.model).or_else(|| self.overrides.model.clone()),
            provider: non_empty(&delegation.provider).or_else(|| self.overrides.provider.clone()),
            base_url: non_empty(&delegation.base_url).or_else(|| self.overrides.base_url.clone()),
            api_key: non_empty(&delegation.api_key).or_else(|| self.overrides.api_key.clone()),
            api_mode: non_empty(&delegation.api_mode).or_else(|| self.overrides.api_mode.clone()),
        }
    }

    fn child_toolsets(
        &self,
        requested: Option<&Vec<String>>,
        role: &str,
    ) -> Result<Vec<String>, String> {
        let parent_allowed = expanded_parent_toolsets(&self.parent_toolsets);
        let mut child_toolsets = if let Some(requested) = requested {
            requested
                .iter()
                .filter(|toolset| parent_allowed.contains(*toolset))
                .cloned()
                .collect::<Vec<_>>()
        } else {
            parent_allowed.into_iter().collect::<Vec<_>>()
        };

        child_toolsets.retain(|toolset| {
            let blocked = if role == "orchestrator" {
                BLOCKED_ORCHESTRATOR_TOOLSETS
            } else {
                BLOCKED_LEAF_TOOLSETS
            };
            !blocked.contains(&toolset.as_str())
        });
        if role == "orchestrator" && !child_toolsets.iter().any(|toolset| toolset == "delegation") {
            child_toolsets.push("delegation".to_string());
        }
        child_toolsets.sort();
        child_toolsets.dedup();
        Ok(child_toolsets)
    }
}

fn emit_subagent_stop_hooks(parent_runtime: &ToolRuntime, results: &mut [Value]) {
    let parent_session_id = parent_runtime.current_session_id().unwrap_or_default();
    for entry in results {
        let Some(object) = entry.as_object_mut() else {
            continue;
        };
        let child_role = object.remove("_child_role").unwrap_or(Value::Null);
        let duration_ms = object
            .get("duration_seconds")
            .and_then(Value::as_f64)
            .map(|seconds| (seconds * 1000.0) as u64)
            .unwrap_or_default();
        let payload = json!({
            "parent_session_id": parent_session_id,
            "child_role": child_role,
            "child_summary": object.get("summary").cloned().unwrap_or(Value::Null),
            "child_status": object.get("status").cloned().unwrap_or(Value::Null),
            "duration_ms": duration_ms,
        });
        let _ = parent_runtime.invoke_hook("subagent_stop", &payload);
    }
}

fn build_child_prompt(goal: &str, context: Option<&str>) -> String {
    let mut parts = vec![
        "You are a focused subagent working on a delegated task.".to_string(),
        format!("Goal:\n{}", goal.trim()),
    ];
    if let Some(context) = context.map(str::trim).filter(|value| !value.is_empty()) {
        parts.push(format!("Context:\n{context}"));
    }
    parts.push(
        "Complete the task and return a concise summary of what you did, what you found, any files changed, and any remaining issues.".to_string(),
    );
    parts.join("\n\n")
}

fn normalize_role(value: Option<&str>) -> Result<&str, String> {
    match value.unwrap_or("leaf").trim() {
        "" | "leaf" => Ok("leaf"),
        "orchestrator" => Ok("orchestrator"),
        other => Err(format!(
            "Invalid role '{other}'. Use 'leaf' or 'orchestrator'."
        )),
    }
}

fn effective_role(
    requested_role: &str,
    child_depth: usize,
    max_spawn_depth: usize,
    orchestrator_enabled: bool,
) -> &str {
    if requested_role == "orchestrator" && orchestrator_enabled && child_depth < max_spawn_depth {
        "orchestrator"
    } else {
        "leaf"
    }
}

fn expanded_parent_toolsets(parent_toolsets: &[String]) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    for toolset in parent_toolsets {
        expand_toolset_name(toolset, &mut result);
    }
    if result.is_empty() {
        for toolset in get_toolset_names() {
            if !toolset.starts_with("hermes-") {
                result.insert(toolset);
            }
        }
    }
    result
}

fn expand_toolset_name(name: &str, result: &mut BTreeSet<String>) {
    if !validate_toolset(name) || !result.insert(name.to_string()) {
        return;
    }
    let Some(info) = get_toolset_info(name) else {
        return;
    };
    for include in info.includes {
        expand_toolset_name(&include, result);
    }
    if name.starts_with("hermes-") {
        result.remove(name);
    }
}

fn validate_requested_toolsets(toolsets: &[String]) -> Result<(), String> {
    for toolset in toolsets {
        if !validate_toolset(toolset) {
            return Err(format!("Unknown toolset '{toolset}'."));
        }
    }
    Ok(())
}

fn non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn error_string(error: HermesError) -> String {
    error.to_string()
}
