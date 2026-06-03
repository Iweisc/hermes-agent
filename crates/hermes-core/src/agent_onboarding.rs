//! Contextual first-touch onboarding hints.
//!
//! Instead of blocking first-run questionnaires, show a one-time hint the
//! *first* time a user hits a behavior fork — message-while-running, first
//! long-running tool, etc. Each hint is shown once per install (tracked in
//! `config.yaml` under `onboarding.seen.<flag>`) and then never again.
//!
//! Direct port of `agent/onboarding.py`. Kept tiny and dependency-light so
//! both the CLI and gateway can use it without pulling in heavy modules.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde_yaml::{Mapping, Value};

// -------------------------------------------------------------------------
// Flag names (stable — used as config.yaml keys under onboarding.seen)
// -------------------------------------------------------------------------

pub const BUSY_INPUT_FLAG: &str = "busy_input_prompt";
pub const TOOL_PROGRESS_FLAG: &str = "tool_progress_prompt";
pub const OPENCLAW_RESIDUE_FLAG: &str = "openclaw_residue_cleanup";

// -------------------------------------------------------------------------
// Hint content
// -------------------------------------------------------------------------

/// Hint shown the first time a user messages while the agent is busy.
///
/// `mode` is the effective busy_input_mode that was just applied, so the
/// message matches reality ("I just interrupted…" vs "I just queued…").
pub fn busy_input_hint_gateway(mode: &str) -> String {
    match mode {
        "queue" => "💡 First-time tip — I queued your message instead of interrupting. \
             Send `/busy interrupt` to make new messages stop the current task \
             immediately, or `/busy status` to check. This notice won't appear again."
            .to_string(),
        "steer" => "💡 First-time tip — I steered your message into the current run; \
             it will arrive after the next tool call instead of interrupting. \
             Send `/busy interrupt` or `/busy queue` to change this, or \
             `/busy status` to check. This notice won't appear again."
            .to_string(),
        _ => "💡 First-time tip — I just interrupted my current task to answer you. \
             Send `/busy queue` to queue follow-ups for after the current task instead, \
             `/busy steer` to inject them mid-run without interrupting, or \
             `/busy status` to check. This notice won't appear again."
            .to_string(),
    }
}

/// CLI version of the busy-input hint (plain text, no markdown).
pub fn busy_input_hint_cli(mode: &str) -> String {
    match mode {
        "queue" => "(tip) Your message was queued for the next turn. \
             Use /busy interrupt to make Enter stop the current run instead, \
             or /busy steer to inject mid-run. This tip only shows once."
            .to_string(),
        "steer" => "(tip) Your message was steered into the current run; it arrives \
             after the next tool call. Use /busy interrupt or /busy queue to \
             change this. This tip only shows once."
            .to_string(),
        _ => "(tip) Your message interrupted the current run. \
             Use /busy queue to queue messages for the next turn instead, \
             or /busy steer to inject mid-run. This tip only shows once."
            .to_string(),
    }
}

pub fn tool_progress_hint_gateway() -> String {
    "💡 First-time tip — that tool took a while and I'm streaming every step. \
     If the progress messages feel noisy, send `/verbose` to cycle modes \
     (all → new → off). This notice won't appear again."
        .to_string()
}

pub fn tool_progress_hint_cli() -> String {
    "(tip) That tool ran for a while. Use /verbose to cycle tool-progress \
     display modes (all -> new -> off -> verbose). This tip only shows once."
        .to_string()
}

/// Banner shown the first time Hermes starts and finds `~/.openclaw/`.
///
/// Points users at `hermes claw migrate` (non-destructive port of config,
/// memory, and skills) first. `hermes claw cleanup` is mentioned as the
/// follow-up step for users who have already migrated and want to archive
/// the old directory — with a warning that archiving breaks OpenClaw.
pub fn openclaw_residue_hint_cli() -> String {
    "A legacy OpenClaw directory was detected at ~/.openclaw/.\n\
     To port your config, memory, and skills over to Hermes, run \
     `hermes claw migrate`.\n\
     If you've already migrated and want to archive the old directory, \
     run `hermes claw cleanup` (renames it to ~/.openclaw.pre-migration — \
     OpenClaw will stop working after this).\n\
     This tip only shows once."
        .to_string()
}

/// Return true if an OpenClaw workspace directory is present in `$HOME`.
///
/// Pure filesystem check — no side effects. `home` override exists for tests.
pub fn detect_openclaw_residue(home: Option<&Path>) -> bool {
    let base: PathBuf = match home {
        Some(h) => h.to_path_buf(),
        None => match dirs::home_dir() {
            Some(h) => h,
            None => return false,
        },
    };
    base.join(".openclaw").is_dir()
}

// -------------------------------------------------------------------------
// State read / write
// -------------------------------------------------------------------------

/// Borrow the `onboarding.seen` mapping from a config value, if present.
fn get_seen_mapping(config: &Value) -> Option<&Mapping> {
    config
        .as_mapping()?
        .get(Value::String("onboarding".to_string()))
        .and_then(Value::as_mapping)?
        .get(Value::String("seen".to_string()))
        .and_then(Value::as_mapping)
}

/// Return true if the user has already been shown this first-touch hint.
///
/// Mirrors Python's `bool(...)` truthiness on the flag value: only a value
/// that is truthy (true, non-zero number, non-empty string/collection)
/// counts as "seen".
pub fn is_seen(config: &Value, flag: &str) -> bool {
    let Some(seen) = get_seen_mapping(config) else {
        return false;
    };
    match seen.get(Value::String(flag.to_string())) {
        Some(value) => is_truthy(value),
        None => false,
    }
}

/// Python-style truthiness for a YAML value.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(u) = n.as_u64() {
                u != 0
            } else {
                n.as_f64().map(|f| f != 0.0).unwrap_or(false)
            }
        }
        Value::String(s) => !s.is_empty(),
        Value::Sequence(s) => !s.is_empty(),
        Value::Mapping(m) => !m.is_empty(),
        Value::Tagged(t) => is_truthy(&t.value),
    }
}

/// Persist `onboarding.seen.<flag> = true` to `config_path`.
///
/// Uses an atomic write (temp file + rename) so a concurrent process can't
/// observe a partially-written file. Returns true on success, false on any
/// error (including the config file being absent — onboarding is best-effort).
pub fn mark_seen(config_path: &Path, flag: &str) -> bool {
    match mark_seen_inner(config_path, flag) {
        Ok(()) => true,
        Err(e) => {
            log::debug!("onboarding: failed to mark flag {}: {}", flag, e);
            false
        }
    }
}

fn mark_seen_inner(config_path: &Path, flag: &str) -> Result<(), String> {
    // Load existing config (or start fresh if absent / empty).
    let mut cfg: Value = if config_path.exists() {
        let contents = std::fs::read_to_string(config_path)
            .map_err(|e| format!("read {}: {}", config_path.display(), e))?;
        if contents.trim().is_empty() {
            Value::Mapping(Mapping::new())
        } else {
            serde_yaml::from_str(&contents).map_err(|e| format!("parse yaml: {}", e))?
        }
    } else {
        Value::Mapping(Mapping::new())
    };

    // `yaml.safe_load` of an explicit `null` document yields {} in the Python
    // code (`... or {}`); mirror that by treating any non-mapping root as {}.
    if !cfg.is_mapping() {
        cfg = Value::Mapping(Mapping::new());
    }

    let root = cfg
        .as_mapping_mut()
        .ok_or_else(|| "config root is not a mapping".to_string())?;

    // Ensure cfg["onboarding"] is a mapping.
    let onboarding_key = Value::String("onboarding".to_string());
    let needs_onboarding = !root
        .get(&onboarding_key)
        .map(Value::is_mapping)
        .unwrap_or(false);
    if needs_onboarding {
        root.insert(onboarding_key.clone(), Value::Mapping(Mapping::new()));
    }
    let onboarding = root
        .get_mut(&onboarding_key)
        .and_then(Value::as_mapping_mut)
        .ok_or_else(|| "onboarding is not a mapping".to_string())?;

    // Ensure onboarding["seen"] is a mapping.
    let seen_key = Value::String("seen".to_string());
    let needs_seen = !onboarding
        .get(&seen_key)
        .map(Value::is_mapping)
        .unwrap_or(false);
    if needs_seen {
        onboarding.insert(seen_key.clone(), Value::Mapping(Mapping::new()));
    }
    let seen = onboarding
        .get_mut(&seen_key)
        .and_then(Value::as_mapping_mut)
        .ok_or_else(|| "seen is not a mapping".to_string())?;

    let flag_key = Value::String(flag.to_string());
    // Already marked exactly `true` — nothing to do (matches Python's
    // `seen.get(flag) is True`).
    if matches!(seen.get(&flag_key), Some(Value::Bool(true))) {
        return Ok(());
    }
    seen.insert(flag_key, Value::Bool(true));

    atomic_yaml_write(config_path, &cfg)
}

/// Write YAML data to a file atomically (temp file + fsync + rename).
///
/// If the process crashes mid-write the previous version of the file remains
/// intact. Mirrors `utils.atomic_yaml_write` (sort_keys=False).
fn atomic_yaml_write(path: &Path, data: &Value) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;

    let serialized = serde_yaml::to_string(data).map_err(|e| format!("serialize yaml: {}", e))?;

    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config".to_string());

    // Unique-ish temp file in the target directory so the rename is atomic
    // (same filesystem). Use pid + nanos to avoid collisions.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_name = format!(".{}_{}_{}.tmp", stem, std::process::id(), nanos);
    let tmp_path = parent.join(tmp_name);

    let write_result = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(serialized.as_bytes())?;
        f.flush()?;
        f.sync_all()?;
        Ok(())
    })();

    if let Err(e) = write_result {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(format!("write temp file: {}", e));
    }

    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(format!("rename into place: {}", e));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yaml(s: &str) -> Value {
        serde_yaml::from_str(s).unwrap()
    }

    #[test]
    fn busy_gateway_modes() {
        assert!(busy_input_hint_gateway("queue").contains("I queued your message"));
        assert!(busy_input_hint_gateway("steer").contains("I steered your message"));
        assert!(busy_input_hint_gateway("interrupt").contains("I just interrupted"));
        // Unknown mode falls through to the interrupt default.
        assert!(busy_input_hint_gateway("whatever").contains("I just interrupted"));
    }

    #[test]
    fn busy_cli_modes() {
        assert!(busy_input_hint_cli("queue").contains("queued for the next turn"));
        assert!(busy_input_hint_cli("steer").contains("steered into the current run"));
        assert!(busy_input_hint_cli("interrupt").contains("interrupted the current run"));
    }

    #[test]
    fn progress_and_openclaw_hints() {
        assert!(tool_progress_hint_gateway().contains("/verbose"));
        assert!(tool_progress_hint_cli().contains("/verbose"));
        let r = openclaw_residue_hint_cli();
        assert!(r.contains("~/.openclaw/"));
        assert!(r.contains("hermes claw migrate"));
        assert!(r.contains("hermes claw cleanup"));
    }

    #[test]
    fn is_seen_reads_nested_flag() {
        let cfg = yaml("onboarding:\n  seen:\n    busy_input_prompt: true\n");
        assert!(is_seen(&cfg, BUSY_INPUT_FLAG));
        assert!(!is_seen(&cfg, TOOL_PROGRESS_FLAG));
    }

    #[test]
    fn is_seen_falsey_values() {
        let cfg = yaml("onboarding:\n  seen:\n    a: false\n    b: 0\n    c: \"\"\n    d: 1\n    e: \"x\"\n");
        assert!(!is_seen(&cfg, "a"));
        assert!(!is_seen(&cfg, "b"));
        assert!(!is_seen(&cfg, "c"));
        assert!(is_seen(&cfg, "d"));
        assert!(is_seen(&cfg, "e"));
    }

    #[test]
    fn is_seen_missing_or_malformed() {
        assert!(!is_seen(&yaml("{}"), BUSY_INPUT_FLAG));
        assert!(!is_seen(&yaml("onboarding: 5"), BUSY_INPUT_FLAG));
        assert!(!is_seen(&yaml("onboarding:\n  seen: notamap\n"), BUSY_INPUT_FLAG));
        assert!(!is_seen(&Value::Null, BUSY_INPUT_FLAG));
    }

    #[test]
    fn detect_openclaw_residue_dir() {
        let dir = std::env::temp_dir().join(format!("hermes_ob_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!detect_openclaw_residue(Some(&dir)));
        std::fs::create_dir_all(dir.join(".openclaw")).unwrap();
        assert!(detect_openclaw_residue(Some(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mark_seen_creates_and_persists() {
        let dir = std::env::temp_dir().join(format!("hermes_ob_ms_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.yaml");

        // File absent: still succeeds and creates it.
        assert!(mark_seen(&path, BUSY_INPUT_FLAG));
        let cfg = yaml(&std::fs::read_to_string(&path).unwrap());
        assert!(is_seen(&cfg, BUSY_INPUT_FLAG));
        assert!(!is_seen(&cfg, TOOL_PROGRESS_FLAG));

        // Idempotent re-mark.
        assert!(mark_seen(&path, BUSY_INPUT_FLAG));
        assert!(is_seen(&yaml(&std::fs::read_to_string(&path).unwrap()), BUSY_INPUT_FLAG));

        // Adding a second flag preserves the first.
        assert!(mark_seen(&path, TOOL_PROGRESS_FLAG));
        let cfg = yaml(&std::fs::read_to_string(&path).unwrap());
        assert!(is_seen(&cfg, BUSY_INPUT_FLAG));
        assert!(is_seen(&cfg, TOOL_PROGRESS_FLAG));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mark_seen_preserves_unrelated_config() {
        let dir = std::env::temp_dir().join(format!("hermes_ob_pres_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.yaml");
        std::fs::write(&path, "model: gpt\nagent:\n  max_turns: 9\n").unwrap();

        assert!(mark_seen(&path, OPENCLAW_RESIDUE_FLAG));
        let cfg = yaml(&std::fs::read_to_string(&path).unwrap());
        assert!(is_seen(&cfg, OPENCLAW_RESIDUE_FLAG));
        assert_eq!(
            cfg.as_mapping()
                .unwrap()
                .get(Value::String("model".into()))
                .and_then(Value::as_str),
            Some("gpt")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mark_seen_overwrites_non_mapping_onboarding() {
        let dir = std::env::temp_dir().join(format!("hermes_ob_nm_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.yaml");
        std::fs::write(&path, "onboarding: 42\n").unwrap();

        assert!(mark_seen(&path, BUSY_INPUT_FLAG));
        assert!(is_seen(&yaml(&std::fs::read_to_string(&path).unwrap()), BUSY_INPUT_FLAG));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
