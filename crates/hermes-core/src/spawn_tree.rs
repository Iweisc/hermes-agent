//! Native spawn-tree snapshot store.
//!
//! Port of the `spawn_tree.save` / `spawn_tree.list` / `spawn_tree.load`
//! handlers in `tui_gateway/server.py`. These are pure filesystem operations
//! under `<hermes_home>/spawn-trees/` (per-session snapshot JSON files plus an
//! `_index.jsonl` cache) with no live-session or plugin dependencies, so they
//! port directly and replace the corresponding `DISPATCH_RPC_HELPER` spawns.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde_json::{Map, Value, json};

const SPAWN_TREE_INDEX: &str = "_index.jsonl";

/// Outcome of a spawn-tree RPC: a `result` object, or `(code, message)` error.
pub type SpawnTreeResult = Result<Value, (i64, String)>;

fn spawn_trees_root(hermes_home: &Path) -> PathBuf {
    hermes_home.join("spawn-trees")
}

/// Sanitize a session id for use as a directory name (port of the inline
/// comprehension: keep alphanumerics, `-`, `_`; everything else becomes `_`;
/// empty -> "unknown").
fn sanitize_session_id(session_id: &str) -> String {
    let sanitized: String = session_id
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

fn spawn_tree_session_dir(hermes_home: &Path, session_id: &str) -> std::io::Result<PathBuf> {
    let dir = spawn_trees_root(hermes_home).join(sanitize_session_id(session_id));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn append_index(session_dir: &Path, entry: &Value) {
    // Index is a cache — losing a line just falls back to a directory scan.
    use std::io::Write;
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(session_dir.join(SPAWN_TREE_INDEX))
    {
        let _ = writeln!(file, "{}", serde_json::to_string(entry).unwrap_or_default());
    }
}

fn read_index(session_dir: &Path) -> Vec<Value> {
    let index_path = session_dir.join(SPAWN_TREE_INDEX);
    let Ok(text) = fs::read_to_string(&index_path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect()
}

/// `spawn_tree.save`: persist a snapshot of `subagents` for `session_id` and
/// append an index entry. Returns `{path, session_id}`.
pub fn save(hermes_home: &Path, params: &Map<String, Value>, now: f64) -> SpawnTreeResult {
    let session_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let subagents = params.get("subagents").cloned().unwrap_or(Value::Null);
    let subagents_array = subagents.as_array().cloned().unwrap_or_default();
    if subagents_array.is_empty() {
        return Err((4000, "subagents list required".to_string()));
    }

    let started_at = params.get("started_at").and_then(Value::as_f64);
    let finished_at = params
        .get("finished_at")
        .and_then(Value::as_f64)
        .filter(|v| *v != 0.0)
        .unwrap_or(now);
    let label = params
        .get("label")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let ts = format_utc_compact(finished_at);
    let fname = format!("{ts}.json");
    let dir = spawn_tree_session_dir(hermes_home, if session_id.is_empty() { "default" } else { &session_id })
        .map_err(|e| (5000, format!("spawn_tree.save failed: {e}")))?;
    let path = dir.join(&fname);

    let started_value = match started_at {
        Some(value) => json!(value),
        None => Value::Null,
    };
    let payload = json!({
        "session_id": session_id,
        "started_at": started_value,
        "finished_at": finished_at,
        "label": label,
        "subagents": subagents_array,
    });
    fs::write(&path, serde_json::to_string(&payload).unwrap_or_default())
        .map_err(|e| (5000, format!("spawn_tree.save failed: {e}")))?;

    append_index(
        &dir,
        &json!({
            "path": path.display().to_string(),
            "session_id": session_id,
            "started_at": started_value,
            "finished_at": finished_at,
            "label": label,
            "count": subagents_array.len(),
        }),
    );

    Ok(json!({"path": path.display().to_string(), "session_id": session_id}))
}

/// `spawn_tree.list`: list snapshot index entries (or a legacy directory scan)
/// for a session, or across all sessions. Returns `{entries: [...]}`.
pub fn list(hermes_home: &Path, params: &Map<String, Value>) -> SpawnTreeResult {
    let session_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let limit = params.get("limit").and_then(Value::as_i64).unwrap_or(50).max(0) as usize;
    let cross_session = params
        .get("cross_session")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let roots: Vec<PathBuf> = if cross_session {
        let root = spawn_trees_root(hermes_home);
        let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect()
            })
            .unwrap_or_default();
        dirs.sort();
        dirs
    } else {
        let safe = sanitize_session_id(if session_id.is_empty() { "default" } else { &session_id });
        vec![spawn_trees_root(hermes_home).join(safe)]
    };

    let mut entries: Vec<Value> = Vec::new();
    for dir in roots {
        let indexed = read_index(&dir);
        if !indexed.is_empty() {
            // Skip index entries whose snapshot file was manually deleted.
            for entry in indexed {
                if let Some(path) = entry.get("path").and_then(Value::as_str) {
                    if Path::new(path).exists() {
                        entries.push(entry);
                    }
                }
            }
            continue;
        }
        // Legacy fallback: scan *.json files in the dir.
        let Ok(read) = fs::read_dir(&dir) else {
            continue;
        };
        let mut json_files: Vec<PathBuf> = read
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|x| x.to_str()) == Some("json")
                    && p.file_name().and_then(|n| n.to_str()) != Some(SPAWN_TREE_INDEX)
            })
            .collect();
        json_files.sort();
        for p in json_files {
            let Ok(meta) = fs::metadata(&p) else { continue };
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            let raw: Value = fs::read_to_string(&p)
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok())
                .unwrap_or_else(|| json!({}));
            let subagents = raw.get("subagents").and_then(Value::as_array);
            let count = subagents.map(|s| s.len()).unwrap_or(0);
            let dir_name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            entries.push(json!({
                "path": p.display().to_string(),
                "session_id": raw.get("session_id").and_then(Value::as_str).map(str::to_string).unwrap_or(dir_name),
                "finished_at": raw.get("finished_at").and_then(Value::as_f64).unwrap_or(mtime),
                "started_at": raw.get("started_at").cloned().unwrap_or(Value::Null),
                "label": raw.get("label").and_then(Value::as_str).unwrap_or("").to_string(),
                "count": count,
            }));
        }
    }

    // Sort by finished_at descending; slice to limit.
    entries.sort_by(|a, b| {
        let fa = a.get("finished_at").and_then(Value::as_f64).unwrap_or(0.0);
        let fb = b.get("finished_at").and_then(Value::as_f64).unwrap_or(0.0);
        fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(limit);
    Ok(json!({"entries": entries}))
}

/// `spawn_tree.load`: read a snapshot JSON file, rejecting paths that escape the
/// spawn-trees root. Returns the raw payload.
pub fn load(hermes_home: &Path, params: &Map<String, Value>) -> SpawnTreeResult {
    let raw_path = params
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if raw_path.is_empty() {
        return Err((4000, "path required".to_string()));
    }

    let root = spawn_trees_root(hermes_home);
    let root_resolved = root.canonicalize().unwrap_or(root);
    let resolved = Path::new(&raw_path)
        .canonicalize()
        .map_err(|e| (4030, format!("path outside spawn-trees root: {e}")))?;
    if !resolved.starts_with(&root_resolved) {
        return Err((
            4030,
            "path outside spawn-trees root: not relative".to_string(),
        ));
    }

    let text = fs::read_to_string(&resolved)
        .map_err(|e| (5000, format!("spawn_tree.load failed: {e}")))?;
    let payload: Value = serde_json::from_str(&text)
        .map_err(|e| (5000, format!("spawn_tree.load failed: {e}")))?;
    Ok(payload)
}

/// Format a UNIX timestamp as `%Y%m%dT%H%M%S` in UTC (port of
/// `datetime.utcfromtimestamp(ts).strftime("%Y%m%dT%H%M%S")`).
fn format_utc_compact(timestamp: f64) -> String {
    let secs = timestamp.trunc() as i64;
    DateTime::<Utc>::from_timestamp(secs, 0)
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap())
        .format("%Y%m%dT%H%M%S")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn save_then_list_and_load_round_trip() {
        let temp = TempDir::new().unwrap();
        let home = temp.path();
        let mut params = Map::new();
        params.insert("session_id".to_string(), json!("sess A/1"));
        params.insert("label".to_string(), json!("my run"));
        params.insert("finished_at".to_string(), json!(1_700_000_000.0));
        params.insert("started_at".to_string(), json!(1_699_999_000.0));
        params.insert("subagents".to_string(), json!([{"id": "a"}, {"id": "b"}]));

        let saved = save(home, &params, 1_700_000_000.0).unwrap();
        let path = saved["path"].as_str().unwrap().to_string();
        assert!(Path::new(&path).exists());
        // session id sanitized in dir name (space and slash -> _)
        assert!(path.contains("sess_A_1"));

        // list returns the index entry
        let mut list_params = Map::new();
        list_params.insert("session_id".to_string(), json!("sess A/1"));
        let listed = list(home, &list_params).unwrap();
        let entries = listed["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["count"], 2);
        assert_eq!(entries[0]["label"], "my run");

        // load returns the full payload
        let mut load_params = Map::new();
        load_params.insert("path".to_string(), json!(path));
        let loaded = load(home, &load_params).unwrap();
        assert_eq!(loaded["session_id"], "sess A/1");
        assert_eq!(loaded["subagents"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn save_requires_subagents() {
        let temp = TempDir::new().unwrap();
        let mut params = Map::new();
        params.insert("session_id".to_string(), json!("s"));
        let err = save(temp.path(), &params, 1.0).unwrap_err();
        assert_eq!(err.0, 4000);
    }

    #[test]
    fn load_rejects_path_outside_root() {
        let temp = TempDir::new().unwrap();
        // create the root so canonicalize works
        fs::create_dir_all(spawn_trees_root(temp.path())).unwrap();
        let outside = temp.path().join("secret.json");
        fs::write(&outside, "{}").unwrap();
        let mut params = Map::new();
        params.insert("path".to_string(), json!(outside.display().to_string()));
        let err = load(temp.path(), &params).unwrap_err();
        assert_eq!(err.0, 4030);
    }

    #[test]
    fn list_limit_and_sort() {
        let temp = TempDir::new().unwrap();
        let home = temp.path();
        for (i, ts) in [100.0, 300.0, 200.0].iter().enumerate() {
            let mut params = Map::new();
            params.insert("session_id".to_string(), json!("s"));
            params.insert("finished_at".to_string(), json!(*ts));
            params.insert("subagents".to_string(), json!([{"i": i}]));
            save(home, &params, *ts).unwrap();
        }
        let mut list_params = Map::new();
        list_params.insert("session_id".to_string(), json!("s"));
        list_params.insert("limit".to_string(), json!(2));
        let listed = list(home, &list_params).unwrap();
        let entries = listed["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 2);
        // sorted by finished_at desc: 300 then 200
        assert_eq!(entries[0]["finished_at"], 300.0);
        assert_eq!(entries[1]["finished_at"], 200.0);
    }
}
