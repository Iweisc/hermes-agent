//! Skills configuration for Hermes Agent — native Rust port of
//! `hermes_cli/skills_config.py`.
//!
//! `hermes skills` enters this module. It lets the user toggle individual
//! skills or whole categories on/off, globally or per-platform. The config is
//! stored in `~/.hermes/config.yaml` under:
//!
//! ```yaml
//! skills:
//!   disabled: [skill-a, skill-b]          # global disabled list
//!   platform_disabled:                    # per-platform overrides
//!     telegram: [skill-c]
//!     cli: []
//! ```
//!
//! Behavioural parity notes:
//! - `PLATFORMS` is the builtin platform table minus `api_server`, exposed as
//!   `{key: label}`. `get_platforms()` augments it with any live plugin
//!   platforms (other than `api_server` / already-present keys).
//! - `get_disabled_skills` returns a *set* of disabled skill names. The
//!   platform-specific list falls back to the global list only when the
//!   platform key is entirely absent (an empty list disables nothing).
//! - `save_disabled_skills` writes a *sorted* list. The global path writes
//!   `skills.disabled`; the per-platform path writes
//!   `skills.platform_disabled.<platform>`.
//! - The interactive `skills_command` flow mirrors the Python prompts and uses
//!   the shared curses checklist for selection. "Selected" == enabled.

use std::collections::{BTreeSet, HashSet};
use std::io::Write as _;

use serde_yaml::{Mapping, Value};

use crate::cli_colors::{color, Colors};
use crate::cli_curses_ui::curses_checklist;
use crate::cli_platforms::{self, PlatformInfo, PluginEntry};
use crate::tool_skills_tool::{find_all_skills, SkillListEntry};

use hermes_core::cli_config::{load_config, save_config};

// ─── Platform Views ─────────────────────────────────────────────────────────

/// Backward-compatible view of the builtin platform table as
/// `{key: label}`, excluding `api_server`.
///
/// Mirrors the module-level `PLATFORMS` dict in Python:
/// `{k: info.label for k, info in _PLATFORMS.items() if k != "api_server"}`.
/// The returned `Vec` preserves the deterministic builtin ordering.
pub fn platforms() -> Vec<(String, String)> {
    cli_platforms::platforms()
        .into_iter()
        .filter(|(k, _)| k != "api_server")
        .map(|(k, info): (String, PlatformInfo)| (k, info.label))
        .collect()
}

/// Return builtin platform labels plus any live plugin platforms.
///
/// Mirrors `_get_platforms()`. `plugin_entries` is the set of dynamically
/// registered plugin platforms (empty when the registry is unavailable —
/// equivalent to the Python `except Exception: pass` branch). Plugin entries
/// keyed `api_server` or already present in the builtin view are skipped.
pub fn get_platforms_with_plugins(plugin_entries: Vec<PluginEntry>) -> Vec<(String, String)> {
    let mut out = platforms();
    let mut seen: HashSet<String> = out.iter().map(|(k, _)| k.clone()).collect();
    for entry in plugin_entries {
        if entry.name == "api_server" || seen.contains(&entry.name) {
            continue;
        }
        seen.insert(entry.name.clone());
        out.push((entry.name.clone(), entry.display_label()));
    }
    out
}

/// Convenience wrapper for the common no-plugin case.
pub fn get_platforms() -> Vec<(String, String)> {
    get_platforms_with_plugins(Vec::new())
}

/// Look up a platform label in the (plugin-augmented) view, falling back to
/// `"All platforms"` — matching `_get_platforms().get(platform, "All
/// platforms")` in `skills_command`.
pub fn platform_label_or_all(platforms_view: &[(String, String)], platform: &str) -> String {
    platforms_view
        .iter()
        .find(|(k, _)| k == platform)
        .map(|(_, label)| label.clone())
        .unwrap_or_else(|| "All platforms".to_string())
}

// ─── Config Helpers ─────────────────────────────────────────────────────────

/// Traverse `cfg[key1][key2]...`, returning `None` on any miss or non-mapping
/// intermediate. Mirrors `hermes_cli.config.cfg_get`.
fn cfg_get<'a>(cfg: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let mut node = cfg;
    for key in keys {
        let map = node.as_mapping()?;
        node = map.get(Value::String((*key).to_string()))?;
    }
    Some(node)
}

/// Collect the string elements of a YAML value that is (expected to be) a
/// sequence of skill names. Non-string elements are ignored, mirroring the
/// loose Python `set(...)` semantics over a list.
fn value_to_string_set(value: &Value) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if let Some(seq) = value.as_sequence() {
        for item in seq {
            if let Some(s) = item.as_str() {
                out.insert(s.to_string());
            }
        }
    }
    out
}

/// Return disabled skill names. A platform-specific list falls back to the
/// global list when the platform key is *absent* (`None`); an explicit empty
/// list disables nothing.
///
/// Mirrors `get_disabled_skills(config, platform)`.
pub fn get_disabled_skills(config: &Value, platform: Option<&str>) -> BTreeSet<String> {
    let skills_cfg = config
        .as_mapping()
        .and_then(|m| m.get(Value::String("skills".to_string())));

    let global_disabled = skills_cfg
        .and_then(|s| cfg_get(s, &["disabled"]))
        .map(value_to_string_set)
        .unwrap_or_default();

    let platform = match platform {
        None => return global_disabled,
        Some(p) => p,
    };

    let Some(skills_cfg) = skills_cfg else {
        return global_disabled;
    };

    match cfg_get(skills_cfg, &["platform_disabled", platform]) {
        None => global_disabled,
        Some(list) => value_to_string_set(list),
    }
}

/// Persist disabled skill names to `config` (in memory) and to disk via
/// `save_config`. Global writes go to `skills.disabled`; per-platform writes go
/// to `skills.platform_disabled.<platform>`. The stored list is sorted.
///
/// Mirrors `save_disabled_skills(config, disabled, platform)`. Returns the
/// result of the underlying `save_config`.
pub fn save_disabled_skills(
    config: &mut Value,
    disabled: &BTreeSet<String>,
    platform: Option<&str>,
) -> Result<(), String> {
    // config.setdefault("skills", {})
    let root = ensure_mapping(config);
    let skills = ensure_child_mapping(root, "skills");

    // BTreeSet already iterates in sorted order; `sorted(disabled)`.
    let sorted: Vec<Value> = disabled
        .iter()
        .map(|s| Value::String(s.clone()))
        .collect();

    match platform {
        None => {
            skills.insert(
                Value::String("disabled".to_string()),
                Value::Sequence(sorted),
            );
        }
        Some(p) => {
            // config["skills"].setdefault("platform_disabled", {})
            let plat_disabled = ensure_child_mapping(skills, "platform_disabled");
            plat_disabled.insert(Value::String(p.to_string()), Value::Sequence(sorted));
        }
    }

    save_config(config)
}

/// Coerce `value` into a mapping in place, replacing non-mapping values with an
/// empty mapping, and return a mutable reference to it.
fn ensure_mapping(value: &mut Value) -> &mut Mapping {
    if !value.is_mapping() {
        *value = Value::Mapping(Mapping::new());
    }
    value.as_mapping_mut().expect("ensured mapping")
}

/// `dict.setdefault(key, {})` over a `serde_yaml::Mapping`, returning a mutable
/// reference to the (now guaranteed) child mapping.
fn ensure_child_mapping<'a>(map: &'a mut Mapping, key: &str) -> &'a mut Mapping {
    let k = Value::String(key.to_string());
    let entry = map
        .entry(k)
        .or_insert_with(|| Value::Mapping(Mapping::new()));
    if !entry.is_mapping() {
        *entry = Value::Mapping(Mapping::new());
    }
    entry.as_mapping_mut().expect("ensured child mapping")
}

// ─── Skill Discovery ─────────────────────────────────────────────────────────

/// Return all installed skills, ignoring disabled state.
///
/// Mirrors `_list_all_skills()` -> `_find_all_skills(skip_disabled=True)`.
/// Discovery failures degrade to an empty list (the Python `except`).
pub fn list_all_skills() -> Vec<SkillListEntry> {
    find_all_skills(true)
}

/// The category of a skill, treating `None` as `"uncategorized"`.
fn skill_category(skill: &SkillListEntry) -> String {
    match &skill.category {
        Some(c) if !c.is_empty() => c.clone(),
        _ => "uncategorized".to_string(),
    }
}

/// Return sorted unique category names (`None` -> `"uncategorized"`).
///
/// Mirrors `_get_categories(skills)`.
pub fn get_categories(skills: &[SkillListEntry]) -> Vec<String> {
    let set: BTreeSet<String> = skills.iter().map(skill_category).collect();
    set.into_iter().collect()
}

// ─── Category Toggle ─────────────────────────────────────────────────────────

/// Toggle all skills in a category at once.
///
/// A category is "enabled" (pre-checked) when NOT all of its skills are
/// currently disabled. Checking a category removes its skills from the disabled
/// set; unchecking adds them. Mirrors `_toggle_by_category(skills, disabled)`.
pub fn toggle_by_category(
    skills: &[SkillListEntry],
    disabled: &BTreeSet<String>,
) -> BTreeSet<String> {
    let categories = get_categories(skills);

    let mut cat_labels: Vec<String> = Vec::with_capacity(categories.len());
    let mut pre_selected: BTreeSet<usize> = BTreeSet::new();

    for (i, cat) in categories.iter().enumerate() {
        let cat_skills: Vec<&String> = skills
            .iter()
            .filter(|s| &skill_category(s) == cat)
            .map(|s| &s.name)
            .collect();
        cat_labels.push(format!("{} ({} skills)", cat, cat_skills.len()));
        // enabled (checked) when NOT all its skills are disabled
        if !cat_skills.iter().all(|n| disabled.contains(*n)) {
            pre_selected.insert(i);
        }
    }

    let chosen = curses_checklist(
        "Categories — toggle entire categories",
        &cat_labels,
        &pre_selected,
        Some(&pre_selected),
        None,
    );

    let mut new_disabled = disabled.clone();
    for (i, cat) in categories.iter().enumerate() {
        let cat_skills: BTreeSet<String> = skills
            .iter()
            .filter(|s| &skill_category(s) == cat)
            .map(|s| s.name.clone())
            .collect();
        if chosen.contains(&i) {
            // category enabled → remove from disabled
            for name in &cat_skills {
                new_disabled.remove(name);
            }
        } else {
            // category disabled → add to disabled
            new_disabled.extend(cat_skills);
        }
    }
    new_disabled
}

// ─── Individual Toggle ───────────────────────────────────────────────────────

/// Truncate a description to the first 55 chars (by char, not byte), matching
/// the Python slice `s['description'][:55]`.
fn truncate_55(s: &str) -> String {
    s.chars().take(55).collect()
}

/// Build the per-skill checklist labels in the exact format used by Python:
/// `"{name}  ({category})  —  {description[:55]}"`.
pub fn individual_labels(skills: &[SkillListEntry]) -> Vec<String> {
    skills
        .iter()
        .map(|s| {
            format!(
                "{}  ({})  —  {}",
                s.name,
                skill_category(s),
                truncate_55(&s.description)
            )
        })
        .collect()
}

/// Run the individual-skill checklist and return the resulting disabled set
/// (everything NOT chosen). Mirrors the `mode != "2"` branch of
/// `skills_command`.
pub fn toggle_individual(
    skills: &[SkillListEntry],
    disabled: &BTreeSet<String>,
    title: &str,
) -> BTreeSet<String> {
    let labels = individual_labels(skills);
    // "selected" = enabled (not disabled) — matches the [✓] convention.
    let pre_selected: BTreeSet<usize> = skills
        .iter()
        .enumerate()
        .filter(|(_, s)| !disabled.contains(&s.name))
        .map(|(i, _)| i)
        .collect();

    let chosen = curses_checklist(title, &labels, &pre_selected, Some(&pre_selected), None);

    // Anything NOT chosen is disabled.
    skills
        .iter()
        .enumerate()
        .filter(|(i, _)| !chosen.contains(i))
        .map(|(_, s)| s.name.clone())
        .collect()
}

// ─── Platform Selection ──────────────────────────────────────────────────────

/// Parse the platform-selection menu input.
///
/// `options` is the menu in display order, where index 0 is the synthetic
/// `("global", ...)` entry. Returns:
/// - `None` for blank input, out-of-range, non-numeric, or the `"global"`
///   choice (global default),
/// - `Some(key)` for a concrete platform.
///
/// Mirrors the parsing tail of `_select_platform()`.
pub fn parse_platform_choice(options: &[(String, String)], raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None; // global
    }
    let idx: i64 = match raw.parse::<i64>() {
        Ok(n) => n - 1,
        Err(_) => return None,
    };
    if idx >= 0 && (idx as usize) < options.len() {
        let key = &options[idx as usize].0;
        if key == "global" {
            None
        } else {
            Some(key.clone())
        }
    } else {
        None
    }
}

/// Build the platform-selection options list: a leading synthetic `"global"`
/// entry followed by the plugin-augmented platform view.
fn platform_options() -> Vec<(String, String)> {
    let mut options = vec![(
        "global".to_string(),
        "All platforms (global default)".to_string(),
    )];
    options.extend(get_platforms());
    options
}

/// Prompt the user for which platform to configure (or global). Returns
/// `Some(key)` for a concrete platform, `None` for global / cancel.
///
/// Mirrors `_select_platform()`.
fn select_platform() -> Option<String> {
    let options = platform_options();
    println!();
    println!("{}", color("  Configure skills for:", &[Colors::BOLD]));
    for (i, (_key, label)) in options.iter().enumerate() {
        println!("  {}. {}", i + 1, label);
    }
    println!();
    let raw = match prompt(&color("  Select [1]: ", &[Colors::YELLOW])) {
        Some(r) => r,
        None => return None,
    };
    parse_platform_choice(&options, &raw)
}

/// Read a line from stdin after printing `prompt_text` (no newline). Returns
/// `None` on EOF (mirrors the `EOFError`/`KeyboardInterrupt` guards).
fn prompt(prompt_text: &str) -> Option<String> {
    print!("{}", prompt_text);
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) => None, // EOF
        Ok(_) => Some(line.trim_end_matches(['\n', '\r']).to_string()),
        Err(_) => None,
    }
}

// ─── Entry Point ──────────────────────────────────────────────────────────────

/// Entry point for `hermes skills`. Mirrors `skills_command(args)`.
pub fn skills_command() {
    let mut config = load_config();
    let skills = list_all_skills();

    if skills.is_empty() {
        println!("{}", color("  No skills installed.", &[Colors::DIM]));
        return;
    }

    // Step 1: Select platform.
    let platform = select_platform();
    let platforms_view = get_platforms();
    let platform_label = match &platform {
        Some(p) => platform_label_or_all(&platforms_view, p),
        None => "All platforms".to_string(),
    };

    // Step 2: Select mode — individual or by category.
    println!();
    println!(
        "{}",
        color(
            &format!("  Configure for: {}", platform_label),
            &[Colors::DIM]
        )
    );
    println!();
    println!("  1. Toggle individual skills");
    println!("  2. Toggle by category");
    println!();
    let mode = match prompt(&color("  Select [1]: ", &[Colors::YELLOW])) {
        Some(m) => {
            let m = m.trim().to_string();
            if m.is_empty() {
                "1".to_string()
            } else {
                m
            }
        }
        None => return,
    };

    let disabled = get_disabled_skills(&config, platform.as_deref());

    let new_disabled = if mode == "2" {
        toggle_by_category(&skills, &disabled)
    } else {
        let title = format!("Skills for {}", platform_label);
        toggle_individual(&skills, &disabled, &title)
    };

    if new_disabled == disabled {
        println!("{}", color("  No changes.", &[Colors::DIM]));
        return;
    }

    if let Err(e) = save_disabled_skills(&mut config, &new_disabled, platform.as_deref()) {
        eprintln!("{}", color(&format!("  Failed to save: {e}"), &[Colors::RED]));
        return;
    }

    let enabled_count = skills.len().saturating_sub(new_disabled.len());
    println!(
        "{}",
        color(
            &format!(
                "✓ Saved: {} enabled, {} disabled ({}).",
                enabled_count,
                new_disabled.len(),
                platform_label
            ),
            &[Colors::GREEN]
        )
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::Value;

    fn yaml(s: &str) -> Value {
        serde_yaml::from_str(s).unwrap()
    }

    fn skill(name: &str, category: Option<&str>, desc: &str) -> SkillListEntry {
        SkillListEntry {
            name: name.to_string(),
            description: desc.to_string(),
            category: category.map(|c| c.to_string()),
        }
    }

    #[test]
    fn cfg_get_traverses_and_misses() {
        let cfg = yaml("skills:\n  disabled:\n    - a\n    - b\n");
        let v = cfg_get(&cfg, &["skills", "disabled"]).unwrap();
        assert_eq!(v.as_sequence().unwrap().len(), 2);
        assert!(cfg_get(&cfg, &["skills", "missing"]).is_none());
        // non-mapping intermediate
        let cfg2 = yaml("skills: oops\n");
        assert!(cfg_get(&cfg2, &["skills", "disabled"]).is_none());
    }

    #[test]
    fn global_disabled_returned_for_no_platform() {
        let cfg = yaml("skills:\n  disabled: [a, b]\n");
        let got = get_disabled_skills(&cfg, None);
        assert_eq!(
            got,
            ["a", "b"].iter().map(|s| s.to_string()).collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn platform_absent_falls_back_to_global() {
        let cfg = yaml("skills:\n  disabled: [a, b]\n");
        let got = get_disabled_skills(&cfg, Some("telegram"));
        assert_eq!(
            got,
            ["a", "b"].iter().map(|s| s.to_string()).collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn platform_present_empty_overrides_global() {
        // An explicit empty platform list disables nothing, NOT a fallback.
        let cfg = yaml("skills:\n  disabled: [a, b]\n  platform_disabled:\n    telegram: []\n");
        let got = get_disabled_skills(&cfg, Some("telegram"));
        assert!(got.is_empty());
    }

    #[test]
    fn platform_present_with_values() {
        let cfg = yaml(
            "skills:\n  disabled: [a, b]\n  platform_disabled:\n    telegram: [c, d]\n",
        );
        let got = get_disabled_skills(&cfg, Some("telegram"));
        assert_eq!(
            got,
            ["c", "d"].iter().map(|s| s.to_string()).collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn missing_skills_section_returns_empty() {
        let cfg = yaml("other: 1\n");
        assert!(get_disabled_skills(&cfg, None).is_empty());
        assert!(get_disabled_skills(&cfg, Some("cli")).is_empty());
    }

    #[test]
    fn save_global_writes_sorted_list() {
        let mut cfg = yaml("{}");
        let disabled: BTreeSet<String> =
            ["b", "a", "c"].iter().map(|s| s.to_string()).collect();
        // Build expected in-memory mutation without touching disk: emulate the
        // mutation portion only.
        let root = ensure_mapping(&mut cfg);
        let skills = ensure_child_mapping(root, "skills");
        let sorted: Vec<Value> = disabled.iter().map(|s| Value::String(s.clone())).collect();
        skills.insert(
            Value::String("disabled".to_string()),
            Value::Sequence(sorted),
        );

        let written = cfg_get(&cfg, &["skills", "disabled"]).unwrap();
        let names: Vec<&str> = written
            .as_sequence()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn ensure_child_mapping_replaces_non_mapping() {
        let mut cfg = yaml("skills:\n  platform_disabled: oops\n");
        let root = ensure_mapping(&mut cfg);
        let skills = ensure_child_mapping(root, "skills");
        let pd = ensure_child_mapping(skills, "platform_disabled");
        assert!(pd.is_empty());
    }

    #[test]
    fn categories_sorted_unique_with_uncategorized() {
        let skills = vec![
            skill("z", Some("net"), "d"),
            skill("a", None, "d"),
            skill("b", Some("net"), "d"),
            skill("c", Some(""), "d"),
        ];
        let cats = get_categories(&skills);
        assert_eq!(cats, vec!["net".to_string(), "uncategorized".to_string()]);
    }

    #[test]
    fn individual_labels_format_and_truncate() {
        let long = "x".repeat(80);
        let skills = vec![skill("foo", Some("net"), &long), skill("bar", None, "short")];
        let labels = individual_labels(&skills);
        assert!(labels[0].starts_with("foo  (net)  —  "));
        // 55-char truncation of description
        assert!(labels[0].ends_with(&"x".repeat(55)));
        assert_eq!(labels[1], "bar  (uncategorized)  —  short");
    }

    #[test]
    fn parse_platform_choice_cases() {
        let options = vec![
            ("global".to_string(), "All".to_string()),
            ("cli".to_string(), "CLI".to_string()),
            ("telegram".to_string(), "Telegram".to_string()),
        ];
        assert_eq!(parse_platform_choice(&options, ""), None); // blank → global
        assert_eq!(parse_platform_choice(&options, "  "), None);
        assert_eq!(parse_platform_choice(&options, "1"), None); // global entry
        assert_eq!(
            parse_platform_choice(&options, "2"),
            Some("cli".to_string())
        );
        assert_eq!(
            parse_platform_choice(&options, "3"),
            Some("telegram".to_string())
        );
        assert_eq!(parse_platform_choice(&options, "9"), None); // out of range
        assert_eq!(parse_platform_choice(&options, "abc"), None); // non-numeric
        assert_eq!(parse_platform_choice(&options, "0"), None); // idx -1
    }

    #[test]
    fn platforms_excludes_api_server() {
        let view = platforms();
        assert!(view.iter().all(|(k, _)| k != "api_server"));
        // builtin keys present
        assert!(view.iter().any(|(k, _)| k == "cli"));
        assert!(view.iter().any(|(k, _)| k == "telegram"));
    }

    #[test]
    fn plugin_view_skips_api_server_and_dupes() {
        let plugins = vec![
            PluginEntry {
                name: "irc".to_string(),
                label: "IRC".to_string(),
                emoji: "💬".to_string(),
            },
            PluginEntry {
                name: "cli".to_string(), // dup of builtin → skipped
                label: "DUP".to_string(),
                emoji: String::new(),
            },
            PluginEntry {
                name: "api_server".to_string(), // skipped
                label: "API".to_string(),
                emoji: String::new(),
            },
        ];
        let view = get_platforms_with_plugins(plugins);
        assert!(view.iter().any(|(k, l)| k == "irc" && l.contains("IRC")));
        assert!(view.iter().all(|(k, _)| k != "api_server"));
        // cli kept its builtin label, not "DUP"
        let cli = view.iter().find(|(k, _)| k == "cli").unwrap();
        assert!(cli.1 != "DUP");
    }

    #[test]
    fn platform_label_or_all_fallback() {
        let view = vec![("cli".to_string(), "🖥️  CLI".to_string())];
        assert_eq!(platform_label_or_all(&view, "cli"), "🖥️  CLI");
        assert_eq!(platform_label_or_all(&view, "nope"), "All platforms");
    }

    #[test]
    fn toggle_individual_disables_unchosen() {
        // curses_checklist returns the cancel/pre-selected set on a non-tty,
        // which equals "all enabled" → nothing disabled.
        let skills = vec![skill("a", None, "d"), skill("b", None, "d")];
        let disabled = BTreeSet::new();
        let result = toggle_individual(&skills, &disabled, "t");
        // Non-interactive: pre_selected = all enabled → nothing disabled.
        assert!(result.is_empty());
    }

    #[test]
    fn toggle_by_category_noninteractive_is_idempotent() {
        // On non-tty, checklist returns pre_selected. A category is pre-checked
        // (enabled) iff not all its skills are disabled, so the round trip
        // should reproduce `disabled` for already-consistent input.
        let skills = vec![
            skill("a", Some("net"), "d"),
            skill("b", Some("net"), "d"),
            skill("c", Some("fs"), "d"),
        ];
        // net partially disabled → net stays enabled → its skills removed.
        // fs fully enabled → stays enabled.
        let disabled: BTreeSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        let result = toggle_by_category(&skills, &disabled);
        // net pre-checked (a disabled, b not) → enabled → "a" removed.
        assert!(result.is_empty());
    }
}
