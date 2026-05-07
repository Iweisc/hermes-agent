use std::collections::{BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus};

use clap::{Args, Subcommand, ValueEnum};
use hermes_core::HermesContext;
use serde_json::{Map as JsonMap, Value as JsonValue};
use serde_yaml::Value as YamlValue;

use crate::compat_cmd::CompatArgs;
use crate::config_cmd::{read_raw_yaml_mapping, write_yaml_mapping};
use crate::python_bridge::{project_root, resolve_repo_python};

const EXCLUDED_SKILL_DIRS: &[&str] = &[".git", ".github", ".hub", ".archive"];
const MAX_NAME_LENGTH: usize = 64;

#[derive(Subcommand, Debug)]
pub enum SkillsCommand {
    Browse(CompatArgs),
    Search(CompatArgs),
    Install(CompatArgs),
    Inspect(CompatArgs),
    List(ListArgs),
    Config,
    Check(CompatArgs),
    Update(CompatArgs),
    Audit(CompatArgs),
    Uninstall(UninstallArgs),
    Reset(CompatArgs),
    Publish(CompatArgs),
    Snapshot(CompatArgs),
    Tap(CompatArgs),
}

#[derive(Args, Debug, Clone)]
pub struct ListArgs {
    #[arg(long, value_enum, default_value_t = SkillsSourceFilter::All)]
    pub source: SkillsSourceFilter,
    #[arg(long = "enabled-only", default_value_t = false)]
    pub enabled_only: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum SkillsSourceFilter {
    #[value(name = "all")]
    All,
    #[value(name = "hub")]
    Hub,
    #[value(name = "builtin")]
    Builtin,
    #[value(name = "local")]
    Local,
}

#[derive(Args, Debug, Clone)]
pub struct UninstallArgs {
    pub name: String,
}

#[derive(Debug, Clone)]
struct SkillEntry {
    name: String,
    category: Option<String>,
}

#[derive(Debug, Clone)]
struct HubInstalledEntry {
    source: String,
    trust_level: String,
    install_path: String,
    raw: JsonMap<String, JsonValue>,
}

pub fn print_skills(
    context: &HermesContext,
    command: Option<SkillsCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        None => bridge_skills(None, &[]),
        Some(SkillsCommand::Browse(args)) => bridge_prefixed("browse", &args.args),
        Some(SkillsCommand::Search(args)) => bridge_prefixed("search", &args.args),
        Some(SkillsCommand::Install(args)) => bridge_prefixed("install", &args.args),
        Some(SkillsCommand::Inspect(args)) => bridge_prefixed("inspect", &args.args),
        Some(SkillsCommand::List(args)) => print_list(context, args),
        Some(SkillsCommand::Config) => configure_skills(context),
        Some(SkillsCommand::Check(args)) => bridge_prefixed("check", &args.args),
        Some(SkillsCommand::Update(args)) => bridge_prefixed("update", &args.args),
        Some(SkillsCommand::Audit(args)) => bridge_prefixed("audit", &args.args),
        Some(SkillsCommand::Uninstall(args)) => uninstall_skill(context, &args.name),
        Some(SkillsCommand::Reset(args)) => bridge_prefixed("reset", &args.args),
        Some(SkillsCommand::Publish(args)) => bridge_prefixed("publish", &args.args),
        Some(SkillsCommand::Snapshot(args)) => bridge_prefixed("snapshot", &args.args),
        Some(SkillsCommand::Tap(args)) => bridge_prefixed("tap", &args.args),
    }
}

fn bridge_prefixed(action: &str, passthrough: &[String]) -> Result<(), Box<dyn Error>> {
    bridge_skills(Some(action), passthrough)
}

fn bridge_skills(action: Option<&str>, passthrough: &[String]) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_SKILLS_PYTHON"))
        .ok_or("could not find a Python interpreter for skills")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_SKILLS_ACTION", action.unwrap_or(""))
        .arg("-c")
        .arg(SKILLS_BOOTSTRAP)
        .args(passthrough);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("skills", status).into())
}

const SKILLS_BOOTSTRAP: &str = concat!(
    "import argparse\n",
    "import os\n",
    "import sys\n",
    "action = (os.environ.get('HERMES_SKILLS_ACTION') or '').strip()\n",
    "if action == 'config':\n",
    "    from hermes_cli.skills_config import skills_command\n",
    "else:\n",
    "    from hermes_cli.skills_hub import skills_command\n",
    "parser = argparse.ArgumentParser(prog='hermes skills')\n",
    "parser.set_defaults(skills_action=(action or None))\n",
    "if action == 'browse':\n",
    "    parser.add_argument('--page', type=int, default=1)\n",
    "    parser.add_argument('--size', type=int, default=20)\n",
    "    parser.add_argument('--source', default='all', choices=['all','official','skills-sh','well-known','github','clawhub','lobehub'])\n",
    "elif action == 'search':\n",
    "    parser.add_argument('query')\n",
    "    parser.add_argument('--source', default='all', choices=['all','official','skills-sh','well-known','github','clawhub','lobehub'])\n",
    "    parser.add_argument('--limit', type=int, default=10)\n",
    "elif action == 'install':\n",
    "    parser.add_argument('identifier')\n",
    "    parser.add_argument('--category', default='')\n",
    "    parser.add_argument('--name', default='')\n",
    "    parser.add_argument('--force', action='store_true')\n",
    "    parser.add_argument('--yes', '-y', action='store_true', default=False)\n",
    "elif action == 'inspect':\n",
    "    parser.add_argument('identifier')\n",
    "elif action == 'check':\n",
    "    parser.add_argument('name', nargs='?')\n",
    "elif action == 'update':\n",
    "    parser.add_argument('name', nargs='?')\n",
    "elif action == 'audit':\n",
    "    parser.add_argument('name', nargs='?')\n",
    "elif action == 'reset':\n",
    "    parser.add_argument('name')\n",
    "    parser.add_argument('--restore', action='store_true')\n",
    "    parser.add_argument('--yes', '-y', action='store_true', default=False)\n",
    "elif action == 'publish':\n",
    "    parser.add_argument('skill_path')\n",
    "    parser.add_argument('--to', default='github', choices=['github', 'clawhub'])\n",
    "    parser.add_argument('--repo', default='')\n",
    "elif action == 'snapshot':\n",
    "    subparsers = parser.add_subparsers(dest='snapshot_action')\n",
    "    export_p = subparsers.add_parser('export')\n",
    "    export_p.add_argument('output')\n",
    "    import_p = subparsers.add_parser('import')\n",
    "    import_p.add_argument('input')\n",
    "    import_p.add_argument('--force', action='store_true')\n",
    "elif action == 'tap':\n",
    "    subparsers = parser.add_subparsers(dest='tap_action')\n",
    "    subparsers.add_parser('list')\n",
    "    add_p = subparsers.add_parser('add')\n",
    "    add_p.add_argument('repo')\n",
    "    remove_p = subparsers.add_parser('remove')\n",
    "    remove_p.add_argument('name')\n",
    "elif action == 'config':\n",
    "    pass\n",
    "elif action:\n",
    "    raise SystemExit(f'unsupported skills action: {action}')\n",
    "skills_command(parser.parse_args(sys.argv[1:]))\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn print_list(context: &HermesContext, args: ListArgs) -> Result<(), Box<dyn Error>> {
    let raw_config = load_raw_config(context)?;
    let disabled = resolve_disabled_skills(&raw_config);
    let hub_installed = load_hub_lock(context)?;
    let builtin_names = load_builtin_manifest(context)?;
    let skills = discover_all_skills(context, &raw_config)?;

    if skills.is_empty() {
        println!("No skills installed.");
        return Ok(());
    }

    println!(
        "{:<24} {:<16} {:<12} {:<10} Status",
        "Name", "Category", "Source", "Trust"
    );
    println!(
        "{:<24} {:<16} {:<12} {:<10} ------",
        "------------------------", "----------------", "------------", "----------"
    );

    let mut hub_count = 0_usize;
    let mut builtin_count = 0_usize;
    let mut local_count = 0_usize;
    let mut enabled_count = 0_usize;
    let mut disabled_count = 0_usize;

    for skill in skills {
        let source_info = classify_skill(&skill.name, &hub_installed, &builtin_names);
        if args.source != SkillsSourceFilter::All && args.source != source_info.filter {
            continue;
        }

        let is_enabled = !disabled.contains(&skill.name);
        if args.enabled_only && !is_enabled {
            continue;
        }

        match source_info.filter {
            SkillsSourceFilter::Hub => hub_count += 1,
            SkillsSourceFilter::Builtin => builtin_count += 1,
            SkillsSourceFilter::Local => local_count += 1,
            SkillsSourceFilter::All => {}
        }

        if is_enabled {
            enabled_count += 1;
        } else {
            disabled_count += 1;
        }

        println!(
            "{:<24} {:<16} {:<12} {:<10} {}",
            truncate(&skill.name, 24),
            truncate(skill.category.as_deref().unwrap_or(""), 16),
            truncate(&source_info.source_display, 12),
            truncate(&source_info.trust, 10),
            if is_enabled { "enabled" } else { "disabled" }
        );
    }

    let mut summary =
        format!("{hub_count} hub-installed, {builtin_count} builtin, {local_count} local");
    if args.enabled_only {
        summary.push_str(&format!(" — {enabled_count} enabled shown"));
    } else {
        summary.push_str(&format!(
            " — {enabled_count} enabled, {disabled_count} disabled"
        ));
    }
    println!();
    println!("{summary}");
    Ok(())
}

fn uninstall_skill(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let name = validate_skill_name(raw_name)?;
    let mut installed = load_hub_lock(context)?;
    let Some(entry) = installed.remove(name) else {
        return Err(format!("'{}' is not a hub-installed skill (may be a builtin)", name).into());
    };

    if !confirm_prompt(&format!("Uninstall '{name}'? [y/N]: "))? {
        println!("Cancelled.");
        return Ok(());
    }

    let skills_root = context.hermes_home().join("skills");
    let install_path = validated_install_path(&skills_root, &entry.install_path)?;
    if install_path.exists() {
        fs::remove_dir_all(&install_path)?;
    }

    save_hub_lock(context, &installed)?;
    append_audit_log(
        context,
        "UNINSTALL",
        name,
        &entry.source,
        &entry.trust_level,
        "n/a",
        "user_request",
    )?;
    println!("Uninstalled '{name}' from {}", entry.install_path);
    Ok(())
}

const SKILL_CONFIG_PLATFORMS: &[(&str, &str)] = &[
    ("cli", "CLI"),
    ("telegram", "Telegram"),
    ("discord", "Discord"),
    ("slack", "Slack"),
    ("whatsapp", "WhatsApp"),
    ("signal", "Signal"),
    ("bluebubbles", "BlueBubbles"),
    ("email", "Email"),
    ("homeassistant", "Home Assistant"),
    ("mattermost", "Mattermost"),
    ("matrix", "Matrix"),
    ("dingtalk", "DingTalk"),
    ("feishu", "Feishu"),
    ("wecom", "WeCom"),
    ("wecom_callback", "WeCom Callback"),
    ("weixin", "Weixin"),
    ("qqbot", "QQBot"),
    ("yuanbao", "Yuanbao"),
    ("webhook", "Webhook"),
    ("cron", "Cron"),
];

fn configure_skills(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    if !io::stdin().is_terminal() {
        return Err("skills config requires an interactive terminal".into());
    }

    let raw_config = load_raw_config(context)?;
    let skills = discover_all_skills(context, &raw_config)?;
    if skills.is_empty() {
        println!("No skills installed.");
        return Ok(());
    }

    let platform = prompt_skill_platform()?;
    let platform_label = skill_platform_label(platform.as_deref());
    println!();
    println!("Configure for: {platform_label}");
    println!("  1. Toggle individual skills");
    println!("  2. Toggle by category");
    let mode = prompt_menu_choice("Select [1]: ", 2, 1)?;

    let disabled = disabled_skills_for_platform(&raw_config, platform.as_deref());
    let new_disabled = if mode == 2 {
        toggle_skills_by_category(&skills, &disabled)?
    } else {
        toggle_individual_skills(&skills, &disabled, &platform_label)?
    };

    if new_disabled == disabled {
        println!("No changes.");
        return Ok(());
    }

    save_disabled_skills(context, platform.as_deref(), &new_disabled)?;
    let enabled_count = skills.len().saturating_sub(new_disabled.len());
    println!(
        "Saved: {enabled_count} enabled, {} disabled ({platform_label}).",
        new_disabled.len()
    );
    Ok(())
}

fn prompt_skill_platform() -> Result<Option<String>, Box<dyn Error>> {
    println!();
    println!("Configure skills for:");
    println!("  1. All platforms (global default)");
    for (index, (_key, label)) in SKILL_CONFIG_PLATFORMS.iter().enumerate() {
        println!("  {}. {}", index + 2, label);
    }

    let raw = prompt_line("Select [1]: ")?;
    if raw.is_empty() {
        return Ok(None);
    }
    let selection = raw
        .parse::<usize>()
        .map_err(|_| "selection must be a number")?;
    if selection == 1 {
        return Ok(None);
    }
    let offset = selection
        .checked_sub(2)
        .ok_or("selection is out of range")?;
    let Some((key, _label)) = SKILL_CONFIG_PLATFORMS.get(offset) else {
        return Err("selection is out of range".into());
    };
    Ok(Some((*key).to_string()))
}

fn skill_platform_label(platform: Option<&str>) -> String {
    match platform {
        None => String::from("All platforms"),
        Some(name) => SKILL_CONFIG_PLATFORMS
            .iter()
            .find(|(key, _label)| *key == name)
            .map(|(_key, label)| (*label).to_string())
            .unwrap_or_else(|| name.to_string()),
    }
}

fn prompt_menu_choice(prompt: &str, max: usize, default: usize) -> Result<usize, Box<dyn Error>> {
    let raw = prompt_line(prompt)?;
    if raw.is_empty() {
        return Ok(default);
    }
    let selection = raw
        .parse::<usize>()
        .map_err(|_| "selection must be a number")?;
    if !(1..=max).contains(&selection) {
        return Err("selection is out of range".into());
    }
    Ok(selection)
}

fn toggle_skills_by_category(
    skills: &[SkillEntry],
    disabled: &HashSet<String>,
) -> Result<HashSet<String>, Box<dyn Error>> {
    let mut categories = skills
        .iter()
        .map(|skill| {
            skill
                .category
                .clone()
                .unwrap_or_else(|| String::from("uncategorized"))
        })
        .collect::<Vec<_>>();
    categories.sort();
    categories.dedup();

    let labels = categories
        .iter()
        .map(|category| {
            let count = skills
                .iter()
                .filter(|skill| {
                    skill.category.as_deref().unwrap_or("uncategorized") == category.as_str()
                })
                .count();
            format!("{category} ({count} skills)")
        })
        .collect::<Vec<_>>();

    let preselected = categories
        .iter()
        .enumerate()
        .filter_map(|(index, category)| {
            let all_disabled = skills
                .iter()
                .filter(|skill| {
                    skill.category.as_deref().unwrap_or("uncategorized") == category.as_str()
                })
                .all(|skill| disabled.contains(&skill.name));
            (!all_disabled).then_some(index)
        })
        .collect::<BTreeSet<_>>();

    let chosen = prompt_enabled_indices("Categories", &labels, &preselected)?;
    let mut new_disabled = disabled.clone();
    for (index, category) in categories.iter().enumerate() {
        let category_skills = skills
            .iter()
            .filter(|skill| {
                skill.category.as_deref().unwrap_or("uncategorized") == category.as_str()
            })
            .map(|skill| skill.name.clone())
            .collect::<HashSet<_>>();
        if chosen.contains(&index) {
            new_disabled.retain(|name| !category_skills.contains(name));
        } else {
            new_disabled.extend(category_skills);
        }
    }
    Ok(new_disabled)
}

fn toggle_individual_skills(
    skills: &[SkillEntry],
    disabled: &HashSet<String>,
    platform_label: &str,
) -> Result<HashSet<String>, Box<dyn Error>> {
    let labels = skills
        .iter()
        .map(|skill| {
            let category = skill.category.as_deref().unwrap_or("uncategorized");
            format!("{} ({category})", skill.name)
        })
        .collect::<Vec<_>>();
    let preselected = skills
        .iter()
        .enumerate()
        .filter_map(|(index, skill)| (!disabled.contains(&skill.name)).then_some(index))
        .collect::<BTreeSet<_>>();
    let chosen = prompt_enabled_indices(
        &format!("Skills for {platform_label}"),
        &labels,
        &preselected,
    )?;
    Ok(skills
        .iter()
        .enumerate()
        .filter_map(|(index, skill)| (!chosen.contains(&index)).then_some(skill.name.clone()))
        .collect())
}

fn prompt_enabled_indices(
    title: &str,
    labels: &[String],
    preselected: &BTreeSet<usize>,
) -> Result<BTreeSet<usize>, Box<dyn Error>> {
    println!();
    println!("{title}:");
    for (index, label) in labels.iter().enumerate() {
        let marker = if preselected.contains(&index) {
            'x'
        } else {
            ' '
        };
        println!("  {:>2}. [{}] {}", index + 1, marker, label);
    }
    println!("Enter enabled numbers like 1,3-5, 'all', 'none', or press Enter to keep current.");
    let raw = prompt_line("Enabled [keep]: ")?;
    if raw.is_empty() {
        return Ok(preselected.clone());
    }
    parse_enabled_indices(&raw, labels.len())
}

fn parse_enabled_indices(raw: &str, total: usize) -> Result<BTreeSet<usize>, Box<dyn Error>> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok((0..total).collect());
    }
    if trimmed.eq_ignore_ascii_case("none") {
        return Ok(BTreeSet::new());
    }

    let mut selected = BTreeSet::new();
    for segment in trimmed.split(',') {
        let piece = segment.trim();
        if piece.is_empty() {
            return Err("selection contains an empty item".into());
        }
        if let Some((start_raw, end_raw)) = piece.split_once('-') {
            let start = parse_selection_index(start_raw, total)?;
            let end = parse_selection_index(end_raw, total)?;
            if start > end {
                return Err("selection range must be ascending".into());
            }
            for index in start..=end {
                selected.insert(index);
            }
        } else {
            selected.insert(parse_selection_index(piece, total)?);
        }
    }
    Ok(selected)
}

fn parse_selection_index(raw: &str, total: usize) -> Result<usize, Box<dyn Error>> {
    let selection = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| "selection must use numeric entries")?;
    if selection == 0 || selection > total {
        return Err("selection is out of range".into());
    }
    Ok(selection - 1)
}

fn prompt_line(prompt: &str) -> Result<String, Box<dyn Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn save_disabled_skills(
    context: &HermesContext,
    platform: Option<&str>,
    disabled: &HashSet<String>,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let skills_entry = root
        .entry(yaml_key("skills"))
        .or_insert_with(|| YamlValue::Mapping(YamlMapping::new()));
    if !matches!(skills_entry, YamlValue::Mapping(_)) {
        *skills_entry = YamlValue::Mapping(YamlMapping::new());
    }
    let skills = skills_entry
        .as_mapping_mut()
        .ok_or("skills config must be a mapping")?;

    let disabled_value = sorted_string_sequence(disabled);
    match platform.map(str::trim).filter(|value| !value.is_empty()) {
        None => {
            skills.insert(yaml_key("disabled"), YamlValue::Sequence(disabled_value));
        }
        Some(platform) => {
            let platform_entry = skills
                .entry(yaml_key("platform_disabled"))
                .or_insert_with(|| YamlValue::Mapping(YamlMapping::new()));
            if !matches!(platform_entry, YamlValue::Mapping(_)) {
                *platform_entry = YamlValue::Mapping(YamlMapping::new());
            }
            let platform_mapping = platform_entry
                .as_mapping_mut()
                .ok_or("skills.platform_disabled must be a mapping")?;
            platform_mapping.insert(yaml_key(platform), YamlValue::Sequence(disabled_value));
        }
    }

    write_yaml_mapping(&context.config_path(), &root)
}

fn sorted_string_sequence(values: &HashSet<String>) -> Vec<YamlValue> {
    let mut items = values.iter().cloned().collect::<Vec<_>>();
    items.sort();
    items.into_iter().map(YamlValue::String).collect()
}

fn discover_all_skills(
    context: &HermesContext,
    raw_config: &YamlValue,
) -> Result<Vec<SkillEntry>, Box<dyn Error>> {
    let mut dirs = vec![context.hermes_home().join("skills")];
    dirs.extend(external_skills_dirs(context, raw_config));

    let mut seen = HashSet::new();
    let mut skills = Vec::new();
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let mut skill_files = Vec::new();
        collect_skill_files(&dir, &mut skill_files)?;
        for skill_md in skill_files {
            let content = match fs::read_to_string(&skill_md) {
                Ok(content) => content,
                Err(_) => continue,
            };
            let (frontmatter, _body) = parse_frontmatter(&content);
            if !skill_matches_platform(&frontmatter) {
                continue;
            }
            let Some(skill_dir) = skill_md.parent() else {
                continue;
            };
            let fallback_name = skill_dir
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("skill");
            let name = frontmatter
                .get("name")
                .and_then(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| truncate(value, MAX_NAME_LENGTH))
                .unwrap_or_else(|| truncate(fallback_name, MAX_NAME_LENGTH));
            if !seen.insert(name.clone()) {
                continue;
            }
            skills.push(SkillEntry {
                category: category_from_path(&dir, &skill_md),
                name,
            });
        }
    }

    skills.sort_by(|left, right| {
        let left_key = (
            left.category.as_deref().unwrap_or_default(),
            left.name.as_str(),
        );
        let right_key = (
            right.category.as_deref().unwrap_or_default(),
            right.name.as_str(),
        );
        left_key.cmp(&right_key)
    });
    Ok(skills)
}

fn classify_skill(
    name: &str,
    hub_installed: &HashMap<String, HubInstalledEntry>,
    builtin_names: &HashSet<String>,
) -> SkillSourceInfo {
    if let Some(entry) = hub_installed.get(name) {
        return SkillSourceInfo {
            filter: SkillsSourceFilter::Hub,
            source_display: if entry.source.trim().is_empty() {
                "hub".to_string()
            } else {
                entry.source.clone()
            },
            trust: if entry.source == "official" {
                "official".to_string()
            } else if entry.trust_level.trim().is_empty() {
                "community".to_string()
            } else {
                entry.trust_level.clone()
            },
        };
    }
    if builtin_names.contains(name) {
        return SkillSourceInfo {
            filter: SkillsSourceFilter::Builtin,
            source_display: "builtin".to_string(),
            trust: "builtin".to_string(),
        };
    }
    SkillSourceInfo {
        filter: SkillsSourceFilter::Local,
        source_display: "local".to_string(),
        trust: "local".to_string(),
    }
}

struct SkillSourceInfo {
    filter: SkillsSourceFilter,
    source_display: String,
    trust: String,
}

fn load_raw_config(context: &HermesContext) -> Result<YamlValue, Box<dyn Error>> {
    if !context.config_path().exists() {
        return Ok(YamlValue::Null);
    }
    let text = fs::read_to_string(context.config_path())?;
    if text.trim().is_empty() {
        return Ok(YamlValue::Null);
    }
    Ok(serde_yaml::from_str(&text)?)
}

fn resolve_disabled_skills(raw_config: &YamlValue) -> HashSet<String> {
    let resolved_platform = std::env::var("HERMES_PLATFORM")
        .ok()
        .or_else(|| std::env::var("HERMES_SESSION_PLATFORM").ok())
        .unwrap_or_default();
    disabled_skills_for_platform(
        raw_config,
        (!resolved_platform.trim().is_empty()).then_some(resolved_platform.trim()),
    )
}

fn disabled_skills_for_platform(raw_config: &YamlValue, platform: Option<&str>) -> HashSet<String> {
    let Some(root) = raw_config.as_mapping() else {
        return HashSet::new();
    };
    let Some(skills) = root
        .get(&yaml_key("skills"))
        .and_then(YamlValue::as_mapping)
    else {
        return HashSet::new();
    };

    let global_disabled = normalize_string_set(skills.get(&yaml_key("disabled")));
    let Some(platform) = platform.map(str::trim).filter(|value| !value.is_empty()) else {
        return global_disabled;
    };

    skills
        .get(&yaml_key("platform_disabled"))
        .and_then(YamlValue::as_mapping)
        .and_then(|mapping| mapping.get(&yaml_key(platform)))
        .map(|value| normalize_string_set(Some(value)))
        .unwrap_or(global_disabled)
}

fn normalize_string_set(value: Option<&YamlValue>) -> HashSet<String> {
    match value {
        Some(YamlValue::Sequence(items)) => items
            .iter()
            .filter_map(YamlValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        Some(YamlValue::String(value)) => value
            .trim()
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        _ => HashSet::new(),
    }
}

fn external_skills_dirs(context: &HermesContext, raw_config: &YamlValue) -> Vec<PathBuf> {
    let Some(root) = raw_config.as_mapping() else {
        return Vec::new();
    };
    let Some(skills) = root
        .get(&yaml_key("skills"))
        .and_then(YamlValue::as_mapping)
    else {
        return Vec::new();
    };
    let Some(raw_dirs) = skills.get(&yaml_key("external_dirs")) else {
        return Vec::new();
    };

    let values = match raw_dirs {
        YamlValue::Sequence(items) => items
            .iter()
            .filter_map(YamlValue::as_str)
            .map(str::to_string)
            .collect::<Vec<_>>(),
        YamlValue::String(value) => vec![value.clone()],
        _ => return Vec::new(),
    };

    let local_skills = context.hermes_home().join("skills");
    let local_resolved = local_skills
        .canonicalize()
        .unwrap_or_else(|_| local_skills.clone());
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();

    for raw in values {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let expanded = expand_path_like(trimmed);
        let candidate = if Path::new(&expanded).is_absolute() {
            PathBuf::from(expanded)
        } else {
            context.hermes_home().join(expanded)
        };
        let resolved = candidate
            .canonicalize()
            .unwrap_or_else(|_| candidate.clone());
        if resolved == local_resolved || !resolved.is_dir() || !seen.insert(resolved.clone()) {
            continue;
        }
        result.push(resolved);
    }

    result
}

fn load_builtin_manifest(context: &HermesContext) -> Result<HashSet<String>, Box<dyn Error>> {
    let manifest = context
        .hermes_home()
        .join("skills")
        .join(".bundled_manifest");
    if !manifest.exists() {
        return Ok(HashSet::new());
    }
    let mut result = HashSet::new();
    for line in fs::read_to_string(manifest)?.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let name = trimmed
            .split_once(':')
            .map(|(name, _)| name.trim())
            .unwrap_or(trimmed);
        if !name.is_empty() {
            result.insert(name.to_string());
        }
    }
    Ok(result)
}

fn load_hub_lock(
    context: &HermesContext,
) -> Result<HashMap<String, HubInstalledEntry>, Box<dyn Error>> {
    let lock_path = context
        .hermes_home()
        .join("skills")
        .join(".hub")
        .join("lock.json");
    if !lock_path.exists() {
        return Ok(HashMap::new());
    }
    let parsed = serde_json::from_str::<JsonValue>(&fs::read_to_string(lock_path)?)?;
    let Some(installed) = parsed.get("installed").and_then(JsonValue::as_object) else {
        return Ok(HashMap::new());
    };
    let mut result = HashMap::new();
    for (name, value) in installed {
        let Some(object) = value.as_object() else {
            continue;
        };
        let source = object
            .get("source")
            .and_then(JsonValue::as_str)
            .unwrap_or("hub")
            .trim()
            .to_string();
        let trust_level = object
            .get("trust_level")
            .and_then(JsonValue::as_str)
            .unwrap_or("community")
            .trim()
            .to_string();
        let install_path = object
            .get("install_path")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        result.insert(
            name.clone(),
            HubInstalledEntry {
                source,
                trust_level,
                install_path,
                raw: object.clone(),
            },
        );
    }
    Ok(result)
}

fn save_hub_lock(
    context: &HermesContext,
    installed: &HashMap<String, HubInstalledEntry>,
) -> Result<(), Box<dyn Error>> {
    let lock_path = context
        .hermes_home()
        .join("skills")
        .join(".hub")
        .join("lock.json");
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut root = JsonMap::new();
    root.insert("version".to_string(), JsonValue::from(1));
    let mut installed_map = JsonMap::new();
    let mut names = installed.keys().cloned().collect::<Vec<_>>();
    names.sort();
    for name in names {
        let Some(entry) = installed.get(&name) else {
            continue;
        };
        installed_map.insert(name, JsonValue::Object(entry.raw.clone()));
    }
    root.insert("installed".to_string(), JsonValue::Object(installed_map));
    fs::write(
        lock_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&JsonValue::Object(root))?
        ),
    )?;
    Ok(())
}

fn append_audit_log(
    context: &HermesContext,
    action: &str,
    skill_name: &str,
    source: &str,
    trust_level: &str,
    verdict: &str,
    extra: &str,
) -> Result<(), Box<dyn Error>> {
    let path = context
        .hermes_home()
        .join("skills")
        .join(".hub")
        .join("audit.log");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let timestamp = iso8601_now();
    let mut line = format!("{timestamp} {action} {skill_name} {source}:{trust_level} {verdict}");
    if !extra.trim().is_empty() {
        line.push(' ');
        line.push_str(extra.trim());
    }
    line.push('\n');
    let mut contents = if path.exists() {
        fs::read_to_string(&path)?
    } else {
        String::new()
    };
    contents.push_str(&line);
    fs::write(path, contents)?;
    Ok(())
}

fn validated_install_path(
    skills_root: &Path,
    install_path: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let trimmed = install_path.trim();
    if trimmed.is_empty() {
        return Err("hub install_path is empty".into());
    }
    let path = Path::new(trimmed);
    if path.is_absolute() {
        return Err("hub install_path must be relative".into());
    }
    for component in path.components() {
        match component {
            Component::CurDir | Component::Normal(_) => {}
            Component::ParentDir => return Err("hub install_path cannot escape skills root".into()),
            Component::RootDir | Component::Prefix(_) => {
                return Err("hub install_path must stay within skills root".into());
            }
        }
    }
    Ok(skills_root.join(path))
}

fn collect_skill_files(root: &Path, output: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    if !root.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if path.is_dir() {
            if EXCLUDED_SKILL_DIRS.iter().any(|excluded| *excluded == name) {
                continue;
            }
            collect_skill_files(&path, output)?;
        } else if name == "SKILL.md" {
            output.push(path);
        }
    }
    Ok(())
}

fn parse_frontmatter(content: &str) -> (YamlMapping, String) {
    if !content.starts_with("---") {
        return (YamlMapping::new(), content.to_string());
    }
    let tail = &content[3..];
    let Some(end_offset) = tail.find("\n---\n").or_else(|| tail.find("\n---\r\n")) else {
        return (YamlMapping::new(), content.to_string());
    };
    let yaml_content = &tail[..end_offset];
    let body = tail[end_offset + 5..].to_string();
    match serde_yaml::from_str::<YamlValue>(yaml_content) {
        Ok(YamlValue::Mapping(mapping)) => (mapping, body),
        _ => (YamlMapping::new(), body),
    }
}

type YamlMapping = serde_yaml::Mapping;

fn skill_matches_platform(frontmatter: &YamlMapping) -> bool {
    let Some(platforms) = frontmatter.get(&yaml_key("platforms")) else {
        return true;
    };
    let values = match platforms {
        YamlValue::Sequence(items) => items
            .iter()
            .filter_map(YamlValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>(),
        YamlValue::String(value) => vec![value.trim().to_string()],
        _ => Vec::new(),
    };
    if values.is_empty() {
        return true;
    }
    let current = std::env::consts::OS;
    values
        .into_iter()
        .any(|platform| match platform.to_ascii_lowercase().as_str() {
            "macos" => current == "macos",
            "linux" => current == "linux",
            "windows" => current == "windows",
            other => other == current,
        })
}

fn category_from_path(skills_root: &Path, skill_md: &Path) -> Option<String> {
    let rel = skill_md.strip_prefix(skills_root).ok()?;
    let mut parts = rel.components();
    let first = parts.next()?;
    let second = parts.next()?;
    if second.as_os_str() == "SKILL.md" {
        return None;
    }
    match first {
        Component::Normal(value) => Some(value.to_string_lossy().to_string()),
        _ => None,
    }
}

fn validate_skill_name(raw: &str) -> Result<&str, Box<dyn Error>> {
    let name = raw.trim();
    if name.is_empty() {
        return Err("skill name cannot be empty".into());
    }
    Ok(name)
}

fn truncate(value: &str, max_len: usize) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_len {
        return value.to_string();
    }
    if max_len <= 3 {
        return ".".repeat(max_len);
    }
    let prefix = chars[..max_len - 3].iter().collect::<String>();
    format!("{prefix}...")
}

fn expand_path_like(value: &str) -> String {
    let mut expanded = value.to_string();
    if expanded == "~" {
        if let Some(home) = dirs::home_dir() {
            expanded = home.display().to_string();
        }
    } else if let Some(rest) = expanded.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            expanded = home.join(rest).display().to_string();
        }
    }

    let mut rendered = String::new();
    let mut chars = expanded.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next();
            let mut key = String::new();
            while let Some(next) = chars.next() {
                if next == '}' {
                    break;
                }
                key.push(next);
            }
            if key.is_empty() {
                rendered.push_str("${}");
            } else if let Ok(value) = std::env::var(&key) {
                rendered.push_str(&value);
            }
        } else {
            rendered.push(ch);
        }
    }
    rendered
}

fn iso8601_now() -> String {
    let output = std::process::Command::new("date")
        .arg("-u")
        .arg("+%Y-%m-%dT%H:%M:%SZ")
        .output();
    match output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        }
        _ => "1970-01-01T00:00:00Z".to_string(),
    }
}

fn confirm_prompt(prompt: &str) -> Result<bool, Box<dyn Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    Ok(matches!(trimmed.to_ascii_lowercase().as_str(), "y" | "yes"))
}

fn yaml_key(key: &str) -> YamlValue {
    YamlValue::String(key.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(test)]
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::TempDir;

    #[cfg(test)]
    fn test_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(test)]
    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe {
            env::set_var(key, value);
        }
    }

    #[cfg(test)]
    fn remove_env_var(key: &str) {
        unsafe {
            env::remove_var(key);
        }
    }

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-skills-{label}-{unique}"))
    }

    #[test]
    fn list_sees_local_builtin_hub_and_disabled_skills() {
        let home = temp_path("list");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let skills_dir = home.join("skills");
        fs::create_dir_all(skills_dir.join("mlops").join("builtin-skill")).unwrap();
        fs::create_dir_all(skills_dir.join("local-skill")).unwrap();
        fs::create_dir_all(skills_dir.join("hub-skill")).unwrap();
        fs::create_dir_all(skills_dir.join(".hub")).unwrap();
        fs::write(
            skills_dir
                .join("mlops")
                .join("builtin-skill")
                .join("SKILL.md"),
            "---\nname: builtin-skill\ndescription: Builtin\n---\nBody\n",
        )
        .unwrap();
        fs::write(
            skills_dir.join("local-skill").join("SKILL.md"),
            "---\nname: local-skill\ndescription: Local\n---\nBody\n",
        )
        .unwrap();
        fs::write(
            skills_dir.join("hub-skill").join("SKILL.md"),
            "---\nname: hub-skill\ndescription: Hub\n---\nBody\n",
        )
        .unwrap();
        fs::write(skills_dir.join(".bundled_manifest"), "builtin-skill:hash\n").unwrap();
        fs::write(
            skills_dir.join(".hub").join("lock.json"),
            r#"{"version":1,"installed":{"hub-skill":{"source":"official","trust_level":"trusted","install_path":"hub-skill","files":["SKILL.md"]}}}"#,
        )
        .unwrap();
        fs::write(
            context.config_path(),
            "skills:\n  disabled:\n    - local-skill\n",
        )
        .unwrap();

        print_list(
            &context,
            ListArgs {
                source: SkillsSourceFilter::All,
                enabled_only: false,
            },
        )
        .unwrap();

        let disabled = resolve_disabled_skills(&load_raw_config(&context).unwrap());
        assert!(disabled.contains("local-skill"));
        let builtin = load_builtin_manifest(&context).unwrap();
        assert!(builtin.contains("builtin-skill"));
        let hub = load_hub_lock(&context).unwrap();
        assert!(hub.contains_key("hub-skill"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn uninstall_rejects_non_hub_skill() {
        let home = temp_path("uninstall-miss");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let error = uninstall_skill(&context, "missing")
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a hub-installed skill"));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn external_dirs_are_resolved_from_config() {
        let home = temp_path("external");
        let external = temp_path("external-src");
        fs::create_dir_all(&external).unwrap();
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        if let Some(parent) = context.config_path().parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(
            context.config_path(),
            format!(
                "skills:\n  external_dirs:\n    - {}\n    - ./missing\n",
                external.display()
            ),
        )
        .unwrap();
        let dirs = external_skills_dirs(&context, &load_raw_config(&context).unwrap());
        assert_eq!(dirs, vec![external.canonicalize().unwrap()]);
        let _ = fs::remove_dir_all(home);
        let _ = fs::remove_dir_all(external);
    }

    #[test]
    fn disabled_skills_for_platform_falls_back_to_global() {
        let config = serde_yaml::from_str::<YamlValue>(
            "skills:\n  disabled:\n    - global-a\n  platform_disabled:\n    cli:\n      - cli-only\n",
        )
        .unwrap();

        let cli_disabled = disabled_skills_for_platform(&config, Some("cli"));
        assert_eq!(cli_disabled, HashSet::from([String::from("cli-only")]));

        let telegram_disabled = disabled_skills_for_platform(&config, Some("telegram"));
        assert_eq!(telegram_disabled, HashSet::from([String::from("global-a")]));
    }

    #[test]
    fn save_disabled_skills_writes_platform_override_without_clobbering_global() {
        let home = temp_path("save-platform");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        if let Some(parent) = context.config_path().parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(
            context.config_path(),
            "skills:\n  disabled:\n    - global-a\n",
        )
        .unwrap();

        save_disabled_skills(
            &context,
            Some("telegram"),
            &HashSet::from([String::from("tg-a"), String::from("tg-b")]),
        )
        .unwrap();

        let saved =
            serde_yaml::from_str::<YamlValue>(&fs::read_to_string(context.config_path()).unwrap())
                .unwrap();
        assert_eq!(
            disabled_skills_for_platform(&saved, Some("telegram")),
            HashSet::from([String::from("tg-a"), String::from("tg-b")])
        );
        assert_eq!(
            disabled_skills_for_platform(&saved, Some("cli")),
            HashSet::from([String::from("global-a")])
        );

        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn parse_enabled_indices_supports_ranges_and_keywords() {
        assert_eq!(
            parse_enabled_indices("1,3-4", 5).unwrap(),
            BTreeSet::from([0, 2, 3])
        );
        assert_eq!(
            parse_enabled_indices("all", 3).unwrap(),
            BTreeSet::from([0, 1, 2])
        );
        assert!(parse_enabled_indices("4", 3).is_err());
        assert!(parse_enabled_indices("3-2", 3).is_err());
    }

    #[test]
    fn saving_hub_lock_preserves_unknown_metadata_fields() {
        let home = temp_path("lock-preserve");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let hub_dir = home.join("skills").join(".hub");
        fs::create_dir_all(&hub_dir).unwrap();
        fs::write(
            hub_dir.join("lock.json"),
            r#"{"version":1,"installed":{"hub-skill":{"source":"official","trust_level":"trusted","install_path":"hub-skill","files":["SKILL.md"],"identifier":"owner/repo/hub-skill","metadata":{"foo":"bar"}}}}"#,
        )
        .unwrap();

        let installed = load_hub_lock(&context).unwrap();
        save_hub_lock(&context, &installed).unwrap();

        let saved = fs::read_to_string(hub_dir.join("lock.json")).unwrap();
        assert!(saved.contains("\"identifier\": \"owner/repo/hub-skill\""));
        assert!(saved.contains("\"metadata\""));
        assert!(saved.contains("\"foo\": \"bar\""));
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    #[cfg(unix)]
    fn bridge_skills_uses_python_override_and_passes_action_and_args() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  shift 2\n\
  printf 'action=%s argv=%s\\n' \"$HERMES_SKILLS_ACTION\" \"$*\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_SKILLS_PYTHON", &fake_python);
        bridge_prefixed(
            "install",
            &[
                String::from("official/mlops/demo"),
                String::from("--force"),
                String::from("--yes"),
            ],
        )
        .unwrap();
        bridge_skills(None, &[]).unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("action=install argv=official/mlops/demo --force --yes"));
        assert!(output.contains("action= argv="));

        remove_env_var("HERMES_SKILLS_PYTHON");
    }
}
