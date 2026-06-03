//! Todo Tool Module - Planning & Task Management
//!
//! Provides an in-memory task list the agent uses to decompose complex tasks,
//! track progress, and maintain focus across long conversations. The state
//! lives on the agent instance (one per session) and is re-injected into the
//! conversation after context compression events.
//!
//! Design:
//! - Single `todo` tool: provide `todos` param to write, omit to read
//! - Every call returns the full current list
//! - No system prompt mutation, no tool response modification
//! - Behavioral guidance lives entirely in the tool schema description
//!
//! This is a faithful, idiomatic Rust port of `tools/todo_tool.py`.

use serde_json::{json, Value};

/// Valid status values for todo items.
pub const VALID_STATUSES: [&str; 4] = ["pending", "in_progress", "completed", "cancelled"];

/// Returns true if `status` is one of the recognised status values.
pub fn is_valid_status(status: &str) -> bool {
    VALID_STATUSES.contains(&status)
}

/// A single, normalized todo item: `{id, content, status}`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TodoItem {
    pub id: String,
    pub content: String,
    pub status: String,
}

impl TodoItem {
    /// Serialize to a `serde_json::Value` object with `id`/`content`/`status` keys.
    pub fn to_value(&self) -> Value {
        json!({
            "id": self.id,
            "content": self.content,
            "status": self.status,
        })
    }
}

/// Summary counts of a todo list by status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TodoSummary {
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub cancelled: usize,
}

impl TodoSummary {
    /// Compute the summary for a slice of items.
    pub fn from_items(items: &[TodoItem]) -> Self {
        let mut s = TodoSummary {
            total: items.len(),
            ..Default::default()
        };
        for i in items {
            match i.status.as_str() {
                "pending" => s.pending += 1,
                "in_progress" => s.in_progress += 1,
                "completed" => s.completed += 1,
                "cancelled" => s.cancelled += 1,
                _ => {}
            }
        }
        s
    }

    pub fn to_value(&self) -> Value {
        json!({
            "total": self.total,
            "pending": self.pending,
            "in_progress": self.in_progress,
            "completed": self.completed,
            "cancelled": self.cancelled,
        })
    }
}

/// Read a string field from a JSON object, mirroring Python's
/// `str(item.get(key, default)).strip()`.
///
/// Booleans/numbers are stringified like Python does. `null` and missing
/// keys fall back to `default`.
fn get_str_field(item: &Value, key: &str, default: &str) -> String {
    match item.get(key) {
        None | Some(Value::Null) => default.trim().to_string(),
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Bool(b)) => {
            // Python str(True) == "True"
            if *b { "True".to_string() } else { "False".to_string() }
        }
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
    }
}

/// Returns whether a JSON object has a truthy value at `key` (present, not
/// null, and not an empty string / empty container / zero), matching the
/// Python `if "x" in t and t["x"]` idiom for the fields used here.
fn has_truthy(item: &Value, key: &str) -> bool {
    match item.get(key) {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// In-memory todo list. One instance per agent session.
///
/// Items are ordered -- list position is priority. Each item has an `id`
/// (unique, agent-chosen), `content` (description) and `status`.
#[derive(Debug, Clone, Default)]
pub struct TodoStore {
    items: Vec<TodoItem>,
}

impl TodoStore {
    /// Create an empty store.
    pub fn new() -> Self {
        TodoStore { items: Vec::new() }
    }

    /// Write todos. Returns the full current list after writing.
    ///
    /// - `merge == false`: replace the entire list with a fresh plan.
    /// - `merge == true`: update existing items by id, append new ones.
    pub fn write(&mut self, todos: &[Value], merge: bool) -> Vec<TodoItem> {
        if !merge {
            // Replace mode: new list entirely.
            self.items = Self::dedupe_by_id(todos)
                .iter()
                .map(|t| Self::validate(t))
                .collect();
        } else {
            self.merge_write(todos);
        }
        self.read()
    }

    fn merge_write(&mut self, todos: &[Value]) {
        // Mirror of the Python merge logic. `existing` maps id -> item, and is
        // kept in sync with `self.items` (which preserves insertion order).
        use std::collections::HashMap;

        // Index into self.items by id for in-place field updates.
        let mut index_by_id: HashMap<String, usize> = HashMap::new();
        for (i, item) in self.items.iter().enumerate() {
            index_by_id.insert(item.id.clone(), i);
        }

        for t in Self::dedupe_by_id(todos) {
            let item_id = get_str_field(&t, "id", "");
            if item_id.is_empty() {
                continue; // Can't merge without an id.
            }

            if let Some(&idx) = index_by_id.get(&item_id) {
                // Update only the fields the LLM actually provided.
                if has_truthy(&t, "content") {
                    self.items[idx].content = get_str_field(&t, "content", "");
                }
                if has_truthy(&t, "status") {
                    let status = get_str_field(&t, "status", "").to_lowercase();
                    if is_valid_status(&status) {
                        self.items[idx].status = status;
                    }
                }
            } else {
                // New item -- validate fully and append to end.
                let validated = Self::validate(&t);
                index_by_id.insert(validated.id.clone(), self.items.len());
                self.items.push(validated);
            }
        }

        // Rebuild preserving order, dropping any duplicate ids (keeping first
        // occurrence). The Python code does the same via a `seen` set; with the
        // index-based update above duplicates only arise if pre-existing items
        // shared an id, but we replicate the dedupe to stay faithful.
        let mut seen = std::collections::HashSet::new();
        let mut rebuilt = Vec::with_capacity(self.items.len());
        for item in self.items.drain(..) {
            if seen.insert(item.id.clone()) {
                rebuilt.push(item);
            }
        }
        self.items = rebuilt;
    }

    /// Return a copy of the current list.
    pub fn read(&self) -> Vec<TodoItem> {
        self.items.clone()
    }

    /// Check if there are any items in the list.
    pub fn has_items(&self) -> bool {
        !self.items.is_empty()
    }

    /// Render the todo list for post-compression injection.
    ///
    /// Returns a human-readable string to append to the compressed message
    /// history, or `None` if there are no active (pending/in_progress) items.
    pub fn format_for_injection(&self) -> Option<String> {
        if self.items.is_empty() {
            return None;
        }

        // Only inject pending/in_progress items -- completed/cancelled ones
        // cause the model to re-do finished work after compression.
        let active: Vec<&TodoItem> = self
            .items
            .iter()
            .filter(|i| i.status == "pending" || i.status == "in_progress")
            .collect();
        if active.is_empty() {
            return None;
        }

        let mut lines =
            vec!["[Your active task list was preserved across context compression]".to_string()];
        for item in active {
            let marker = match item.status.as_str() {
                "completed" => "[x]",
                "in_progress" => "[>]",
                "pending" => "[ ]",
                "cancelled" => "[~]",
                _ => "[?]",
            };
            lines.push(format!(
                "- {} {}. {} ({})",
                marker, item.id, item.content, item.status
            ));
        }
        Some(lines.join("\n"))
    }

    /// Validate and normalize a todo item.
    ///
    /// Ensures required fields exist and status is valid. Returns a clean
    /// `TodoItem` with only `{id, content, status}`.
    pub fn validate(item: &Value) -> TodoItem {
        let mut id = get_str_field(item, "id", "");
        if id.is_empty() {
            id = "?".to_string();
        }

        let mut content = get_str_field(item, "content", "");
        if content.is_empty() {
            content = "(no description)".to_string();
        }

        let mut status = get_str_field(item, "status", "pending").to_lowercase();
        if !is_valid_status(&status) {
            status = "pending".to_string();
        }

        TodoItem { id, content, status }
    }

    /// Collapse duplicate ids, keeping the last occurrence in its position.
    ///
    /// Mirrors the Python `_dedupe_by_id`: record the last index seen for each
    /// id, then return the items at those indices in ascending index order.
    fn dedupe_by_id(todos: &[Value]) -> Vec<Value> {
        use std::collections::HashMap;
        let mut last_index: HashMap<String, usize> = HashMap::new();
        for (i, item) in todos.iter().enumerate() {
            let mut item_id = get_str_field(item, "id", "");
            if item_id.is_empty() {
                item_id = "?".to_string();
            }
            last_index.insert(item_id, i);
        }
        let mut indices: Vec<usize> = last_index.into_values().collect();
        indices.sort_unstable();
        indices.into_iter().map(|i| todos[i].clone()).collect()
    }
}

/// Build the JSON payload (`{todos, summary}`) for a list of items.
pub fn build_result(items: &[TodoItem]) -> Value {
    let summary = TodoSummary::from_items(items);
    json!({
        "todos": items.iter().map(TodoItem::to_value).collect::<Vec<_>>(),
        "summary": summary.to_value(),
    })
}

/// Standard error payload, matching the registry's `tool_error` shape used by
/// other ported tools.
pub fn tool_error(message: &str) -> String {
    json!({ "error": message }).to_string()
}

/// Single entry point for the todo tool. Reads or writes depending on params.
///
/// - `todos`: if `Some`, write these items; if `None`, read the current list.
/// - `merge`: if `true`, update by id; if `false` (default), replace the list.
/// - `store`: the [`TodoStore`] for the current session.
///
/// Returns a JSON string with the full current list and summary metadata.
pub fn todo_tool(
    todos: Option<&[Value]>,
    merge: bool,
    store: Option<&mut TodoStore>,
) -> String {
    let store = match store {
        Some(s) => s,
        None => return tool_error("TodoStore not initialized"),
    };

    let items = match todos {
        Some(t) => store.write(t, merge),
        None => store.read(),
    };

    // serde_json never emits ensure_ascii-style escaping for non-ASCII; it
    // writes UTF-8 directly, matching `ensure_ascii=False`.
    build_result(&items).to_string()
}

/// Convenience handler matching the registry calling convention: takes the raw
/// tool-call `args` object and a mutable store, dispatches to [`todo_tool`].
pub fn handle(args: &Value, store: Option<&mut TodoStore>) -> String {
    let todos: Option<Vec<Value>> = match args.get("todos") {
        Some(Value::Array(a)) => Some(a.clone()),
        _ => None,
    };
    let merge = args
        .get("merge")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    todo_tool(todos.as_deref(), merge, store)
}

/// Todo tool has no external requirements -- always available.
pub fn check_todo_requirements() -> bool {
    true
}

/// OpenAI function-calling schema for the `todo` tool.
///
/// Behavioral guidance is baked into the description so it's part of the
/// static tool schema (cached, never changes mid-conversation).
pub fn todo_schema() -> Value {
    json!({
        "name": "todo",
        "description": concat!(
            "Manage your task list for the current session. Use for complex tasks ",
            "with 3+ steps or when the user provides multiple tasks. ",
            "Call with no parameters to read the current list.\n\n",
            "Writing:\n",
            "- Provide 'todos' array to create/update items\n",
            "- merge=false (default): replace the entire list with a fresh plan\n",
            "- merge=true: update existing items by id, add any new ones\n\n",
            "Each item: {id: string, content: string, ",
            "status: pending|in_progress|completed|cancelled}\n",
            "List order is priority. Only ONE item in_progress at a time.\n",
            "Mark items completed immediately when done. If something fails, ",
            "cancel it and add a revised item.\n\n",
            "Always returns the full current list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Task items to write. Omit to read current list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique item identifier"
                            },
                            "content": {
                                "type": "string",
                                "description": "Task description"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                                "description": "Current status"
                            }
                        },
                        "required": ["id", "content", "status"]
                    }
                },
                "merge": {
                    "type": "boolean",
                    "description": concat!(
                        "true: update existing items by id, add new ones. ",
                        "false (default): replace the entire list."
                    ),
                    "default": false
                }
            },
            "required": []
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td(id: &str, content: &str, status: &str) -> Value {
        json!({ "id": id, "content": content, "status": status })
    }

    #[test]
    fn write_replace_basic() {
        let mut store = TodoStore::new();
        let items = store.write(&[td("1", "first", "pending"), td("2", "second", "in_progress")], false);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "1");
        assert_eq!(items[1].status, "in_progress");
    }

    #[test]
    fn replace_overwrites_existing() {
        let mut store = TodoStore::new();
        store.write(&[td("1", "first", "pending")], false);
        let items = store.write(&[td("9", "new", "completed")], false);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "9");
    }

    #[test]
    fn validate_defaults() {
        let item = TodoStore::validate(&json!({}));
        assert_eq!(item.id, "?");
        assert_eq!(item.content, "(no description)");
        assert_eq!(item.status, "pending");
    }

    #[test]
    fn validate_invalid_status_falls_back() {
        let item = TodoStore::validate(&td("a", "x", "bogus"));
        assert_eq!(item.status, "pending");
    }

    #[test]
    fn validate_status_lowercased_and_trimmed() {
        let item = TodoStore::validate(&json!({ "id": " a ", "content": " c ", "status": "  IN_PROGRESS " }));
        assert_eq!(item.id, "a");
        assert_eq!(item.content, "c");
        assert_eq!(item.status, "in_progress");
    }

    #[test]
    fn dedupe_keeps_last_in_position() {
        // ids: a, b, a -> last 'a' is index 2, 'b' is index 1.
        // last_index: a->2, b->1 ; sorted indices [1,2] -> [b, a(updated)]
        let mut store = TodoStore::new();
        let items = store.write(
            &[
                td("a", "first-a", "pending"),
                td("b", "b-item", "pending"),
                td("a", "second-a", "completed"),
            ],
            false,
        );
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].id, "b");
        assert_eq!(items[1].id, "a");
        assert_eq!(items[1].content, "second-a");
        assert_eq!(items[1].status, "completed");
    }

    #[test]
    fn merge_updates_existing_and_appends_new() {
        let mut store = TodoStore::new();
        store.write(&[td("1", "one", "pending"), td("2", "two", "pending")], false);
        // Update item 1 status, add item 3.
        let items = store.write(
            &[
                json!({ "id": "1", "status": "completed" }),
                td("3", "three", "in_progress"),
            ],
            true,
        );
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].id, "1");
        assert_eq!(items[0].status, "completed");
        assert_eq!(items[0].content, "one"); // content preserved
        assert_eq!(items[1].id, "2");
        assert_eq!(items[2].id, "3");
    }

    #[test]
    fn merge_ignores_invalid_status_update() {
        let mut store = TodoStore::new();
        store.write(&[td("1", "one", "pending")], false);
        let items = store.write(&[json!({ "id": "1", "status": "nonsense" })], true);
        assert_eq!(items[0].status, "pending");
    }

    #[test]
    fn merge_skips_items_without_id() {
        let mut store = TodoStore::new();
        store.write(&[td("1", "one", "pending")], false);
        let items = store.write(&[json!({ "content": "no id", "status": "completed" })], true);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "1");
    }

    #[test]
    fn format_for_injection_empty_is_none() {
        let store = TodoStore::new();
        assert!(store.format_for_injection().is_none());
    }

    #[test]
    fn format_for_injection_only_active() {
        let mut store = TodoStore::new();
        store.write(
            &[
                td("1", "done", "completed"),
                td("2", "doing", "in_progress"),
                td("3", "todo", "pending"),
                td("4", "killed", "cancelled"),
            ],
            false,
        );
        let out = store.format_for_injection().unwrap();
        assert!(out.contains("[Your active task list was preserved across context compression]"));
        assert!(out.contains("- [>] 2. doing (in_progress)"));
        assert!(out.contains("- [ ] 3. todo (pending)"));
        assert!(!out.contains("done"));
        assert!(!out.contains("killed"));
    }

    #[test]
    fn format_for_injection_none_when_all_finished() {
        let mut store = TodoStore::new();
        store.write(&[td("1", "done", "completed"), td("2", "x", "cancelled")], false);
        assert!(store.format_for_injection().is_none());
    }

    #[test]
    fn todo_tool_no_store_errors() {
        let out = todo_tool(None, false, None);
        assert!(out.contains("TodoStore not initialized"));
    }

    #[test]
    fn todo_tool_read_returns_summary() {
        let mut store = TodoStore::new();
        store.write(
            &[
                td("1", "a", "pending"),
                td("2", "b", "in_progress"),
                td("3", "c", "completed"),
            ],
            false,
        );
        let out = todo_tool(None, false, Some(&mut store));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["summary"]["total"], 3);
        assert_eq!(v["summary"]["pending"], 1);
        assert_eq!(v["summary"]["in_progress"], 1);
        assert_eq!(v["summary"]["completed"], 1);
        assert_eq!(v["summary"]["cancelled"], 0);
        assert_eq!(v["todos"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn handle_dispatches_write() {
        let mut store = TodoStore::new();
        let args = json!({ "todos": [ td("1", "x", "pending") ], "merge": false });
        let out = handle(&args, Some(&mut store));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["summary"]["total"], 1);
        assert!(store.has_items());
    }

    #[test]
    fn handle_no_todos_reads() {
        let mut store = TodoStore::new();
        store.write(&[td("1", "x", "pending")], false);
        let args = json!({});
        let out = handle(&args, Some(&mut store));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["summary"]["total"], 1);
    }

    #[test]
    fn schema_shape() {
        let s = todo_schema();
        assert_eq!(s["name"], "todo");
        assert_eq!(s["parameters"]["properties"]["merge"]["default"], false);
        let enums = s["parameters"]["properties"]["todos"]["items"]["properties"]["status"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(enums.len(), 4);
    }

    #[test]
    fn check_requirements_always_true() {
        assert!(check_todo_requirements());
    }

    #[test]
    fn non_ascii_preserved() {
        let mut store = TodoStore::new();
        store.write(&[td("1", "café ☕", "pending")], false);
        let out = todo_tool(None, false, Some(&mut store));
        assert!(out.contains("café ☕"));
    }
}
