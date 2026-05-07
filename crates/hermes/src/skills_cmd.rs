use std::collections::{BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus};

use clap::{Args, Subcommand, ValueEnum};
use hermes_core::HermesContext;
use md5::Context as Md5Context;
use serde_json::{Map as JsonMap, Value as JsonValue};
use serde_yaml::Value as YamlValue;
use sha2::Digest;

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
    Inspect(InspectArgs),
    List(ListArgs),
    Config,
    Check(CompatArgs),
    Update(CompatArgs),
    Audit(CompatArgs),
    Uninstall(UninstallArgs),
    Reset(SkillResetArgs),
    Publish(CompatArgs),
    Snapshot(SkillSnapshotArgs),
    Tap(SkillTapArgs),
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

#[derive(Args, Debug, Clone)]
pub struct InspectArgs {
    pub identifier: String,
}

#[derive(Args, Debug, Clone)]
pub struct SkillResetArgs {
    pub name: String,
    #[arg(long, default_value_t = false)]
    pub restore: bool,
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,
}

#[derive(Args, Debug, Clone)]
pub struct SkillSnapshotArgs {
    #[command(subcommand)]
    pub command: Option<SkillSnapshotCommand>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum SkillSnapshotCommand {
    Export {
        output: String,
    },
    Import {
        input: String,
        #[arg(long, default_value_t = false)]
        force: bool,
    },
}

#[derive(Args, Debug, Clone)]
pub struct SkillTapArgs {
    #[command(subcommand)]
    pub command: Option<SkillTapCommand>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum SkillTapCommand {
    List,
    Add { repo: String },
    Remove { name: String },
}

#[derive(Debug, Clone)]
struct SkillEntry {
    name: String,
    category: Option<String>,
}

#[derive(Debug, Clone)]
struct SkillRecord {
    entry: SkillEntry,
    skill_md: PathBuf,
}

#[derive(Debug, Clone)]
struct NativeInspectSkill {
    name: String,
    description: String,
    source: String,
    trust: String,
    identifier: String,
    tags: Vec<String>,
    preview: String,
    path: PathBuf,
}

#[derive(Debug, Clone)]
struct OfficialSkillCandidate {
    name: String,
    source: String,
    trust_level: String,
    scan_verdict: String,
    install_path: String,
    source_dir: PathBuf,
    current_hash: String,
    latest_hash: String,
    files: Vec<String>,
}

#[derive(Debug, Clone)]
struct HubInstalledEntry {
    source: String,
    trust_level: String,
    install_path: String,
    raw: JsonMap<String, JsonValue>,
}

#[derive(Debug, Clone)]
struct TapEntry {
    repo: String,
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
        Some(SkillsCommand::Inspect(args)) => inspect_skill_command(context, &args.identifier),
        Some(SkillsCommand::List(args)) => print_list(context, args),
        Some(SkillsCommand::Config) => configure_skills(context),
        Some(SkillsCommand::Check(args)) => check_skills_command(context, &args.args),
        Some(SkillsCommand::Update(args)) => update_skills_command(context, &args.args),
        Some(SkillsCommand::Audit(args)) => bridge_prefixed("audit", &args.args),
        Some(SkillsCommand::Uninstall(args)) => uninstall_skill(context, &args.name),
        Some(SkillsCommand::Reset(args)) => reset_skill(context, args),
        Some(SkillsCommand::Publish(args)) => bridge_prefixed("publish", &args.args),
        Some(SkillsCommand::Snapshot(args)) => print_snapshot(context, args),
        Some(SkillsCommand::Tap(args)) => print_taps(context, args),
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

fn inspect_skill_command(
    context: &HermesContext,
    raw_identifier: &str,
) -> Result<(), Box<dyn Error>> {
    let identifier = validate_skill_identifier(raw_identifier)?;
    if let Some(skill) = resolve_native_inspect_skill(context, identifier)? {
        print_native_inspect(&skill);
        return Ok(());
    }
    bridge_prefixed("inspect", &[identifier.to_string()])
}

fn check_skills_command(
    context: &HermesContext,
    passthrough: &[String],
) -> Result<(), Box<dyn Error>> {
    let Some(name) = parse_single_name_passthrough(passthrough) else {
        return bridge_prefixed("check", passthrough);
    };

    let installed = load_hub_lock(context)?;
    let targets = if let Some(name) = name {
        let Some(entry) = installed.get(name) else {
            println!("No hub-installed skills to check.");
            println!();
            return Ok(());
        };
        vec![(name.to_string(), entry.source.clone())]
    } else {
        installed
            .iter()
            .map(|(name, entry)| (name.clone(), entry.source.clone()))
            .collect()
    };

    if targets.is_empty() {
        println!("No hub-installed skills to check.");
        println!();
        return Ok(());
    }

    if targets.iter().any(|(_, source)| source != "official") {
        return bridge_prefixed("check", passthrough);
    }

    let target_names = targets
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let candidates = collect_official_skill_candidates(&installed, &target_names)?;
    print_official_check_results(&targets, &candidates);
    Ok(())
}

fn update_skills_command(
    context: &HermesContext,
    passthrough: &[String],
) -> Result<(), Box<dyn Error>> {
    let Some(name) = parse_single_name_passthrough(passthrough) else {
        return bridge_prefixed("update", passthrough);
    };

    let mut installed = load_hub_lock(context)?;
    let target_names = if let Some(name) = name {
        let Some(entry) = installed.get(name) else {
            println!("No updates available.");
            println!();
            return Ok(());
        };
        if entry.source != "official" {
            return bridge_prefixed("update", passthrough);
        }
        vec![name.to_string()]
    } else {
        if installed.is_empty() {
            println!("No updates available.");
            println!();
            return Ok(());
        }
        if installed.values().any(|entry| entry.source != "official") {
            return bridge_prefixed("update", passthrough);
        }
        let mut names = installed.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    };

    let candidates = collect_official_skill_candidates(&installed, &target_names)?;
    let updates = candidates
        .into_values()
        .filter(|candidate| candidate.current_hash != candidate.latest_hash)
        .collect::<Vec<_>>();

    if updates.is_empty() {
        println!("No updates available.");
        println!();
        return Ok(());
    }

    apply_official_updates(context, &mut installed, &updates)?;
    save_hub_lock(context, &installed)?;
    println!("Updated {} skill(s).", updates.len());
    println!();
    Ok(())
}

fn parse_single_name_passthrough<'a>(passthrough: &'a [String]) -> Option<Option<&'a str>> {
    if passthrough.len() > 1 {
        return None;
    }
    let name = passthrough.first().map(String::as_str).map(str::trim);
    let Some(name) = name else {
        return Some(None);
    };
    if name.is_empty() || name.starts_with('-') {
        return None;
    }
    Some(Some(name))
}

fn print_official_check_results(
    targets: &[(String, String)],
    candidates: &HashMap<String, OfficialSkillCandidate>,
) {
    println!("{:<24} {:<12} Status", "Name", "Source");
    println!(
        "{:<24} {:<12} ------",
        "------------------------", "------------"
    );

    let mut updates = 0_usize;
    for (name, source) in targets {
        let status = match candidates.get(name) {
            Some(candidate) if candidate.current_hash == candidate.latest_hash => "up_to_date",
            Some(_) => {
                updates += 1;
                "update_available"
            }
            None => "unavailable",
        };
        println!(
            "{:<24} {:<12} {}",
            truncate(name, 24),
            truncate(source, 12),
            status
        );
    }

    println!();
    println!(
        "{} update(s) available across {} checked skill(s)",
        updates,
        targets.len()
    );
    println!();
}

fn collect_official_skill_candidates(
    installed: &HashMap<String, HubInstalledEntry>,
    target_names: &[String],
) -> Result<HashMap<String, OfficialSkillCandidate>, Box<dyn Error>> {
    let optional = discover_optional_skill_records()?;
    let mut records_by_identifier = HashMap::new();
    let mut records_by_name = HashMap::new();
    for record in &optional {
        let Some(skill_dir) = record.skill_md.parent() else {
            continue;
        };
        let rel = match skill_dir.strip_prefix(optional_skills_dir()) {
            Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
            Err(_) => continue,
        };
        records_by_identifier.insert(format!("official/{rel}"), record);
        records_by_name.insert(record.entry.name.clone(), record);
    }

    let mut result = HashMap::new();
    for name in target_names {
        let Some(entry) = installed.get(name) else {
            continue;
        };
        let identifier = entry
            .raw
            .get("identifier")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let record = identifier
            .and_then(|value| records_by_identifier.get(value).copied())
            .or_else(|| records_by_name.get(name).copied());
        let Some(record) = record else {
            continue;
        };
        let Some(source_dir) = record.skill_md.parent().map(Path::to_path_buf) else {
            continue;
        };
        let files = collect_bundle_file_paths(&source_dir)?;
        let latest_hash = bundle_content_hash_from_dir(&source_dir)?;
        let current_hash = entry
            .raw
            .get("content_hash")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        result.insert(
            name.clone(),
            OfficialSkillCandidate {
                name: name.clone(),
                source: entry.source.clone(),
                trust_level: entry.trust_level.clone(),
                scan_verdict: entry
                    .raw
                    .get("scan_verdict")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("n/a")
                    .to_string(),
                install_path: entry.install_path.clone(),
                source_dir,
                current_hash,
                latest_hash,
                files,
            },
        );
    }

    Ok(result)
}

fn apply_official_updates(
    context: &HermesContext,
    installed: &mut HashMap<String, HubInstalledEntry>,
    updates: &[OfficialSkillCandidate],
) -> Result<(), Box<dyn Error>> {
    let skills_root = context.hermes_home().join("skills");
    for update in updates {
        println!("Updating: {}", update.name);
        let install_path = validated_install_path(&skills_root, &update.install_path)?;
        install_official_bundle(&update.source_dir, &install_path)?;

        let Some(entry) = installed.get_mut(&update.name) else {
            return Err(format!("missing hub lock entry for {}", update.name).into());
        };
        entry.raw.insert(
            "content_hash".to_string(),
            JsonValue::String(update.latest_hash.clone()),
        );
        entry.raw.insert(
            "files".to_string(),
            JsonValue::Array(
                update
                    .files
                    .iter()
                    .cloned()
                    .map(JsonValue::String)
                    .collect(),
            ),
        );
        entry
            .raw
            .insert("updated_at".to_string(), JsonValue::String(iso8601_now()));
        append_audit_log(
            context,
            "UPDATE",
            &update.name,
            &update.source,
            &update.trust_level,
            &update.scan_verdict,
            &update.latest_hash,
        )?;
    }
    Ok(())
}

fn install_official_bundle(source_dir: &Path, install_dir: &Path) -> Result<(), Box<dyn Error>> {
    if install_dir.exists() {
        fs::remove_dir_all(install_dir)?;
    }
    fs::create_dir_all(install_dir)?;
    for rel in collect_bundle_file_paths(source_dir)? {
        let source_path = source_dir.join(&rel);
        let dest_path = install_dir.join(&rel);
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source_path, dest_path)?;
    }
    Ok(())
}

fn bundle_content_hash_from_dir(skill_dir: &Path) -> Result<String, Box<dyn Error>> {
    let mut hasher = sha2::Sha256::new();
    for rel in collect_bundle_file_paths(skill_dir)? {
        hasher.update(fs::read(skill_dir.join(rel))?);
    }
    let digest = sha2::Digest::finalize(hasher);
    Ok(format!("sha256:{:x}", digest)[..23].to_string())
}

fn collect_bundle_file_paths(skill_dir: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let mut output = Vec::new();
    collect_bundle_file_paths_inner(skill_dir, skill_dir, &mut output)?;
    output.sort();
    Ok(output)
}

fn collect_bundle_file_paths_inner(
    root: &Path,
    current: &Path,
    output: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    if !current.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if name.starts_with('.') || name == "__pycache__" {
                continue;
            }
            collect_bundle_file_paths_inner(root, &path, output)?;
            continue;
        }
        if !path.is_file()
            || name.starts_with('.')
            || path.extension().is_some_and(|ext| ext == "pyc")
        {
            continue;
        }
        let rel = path
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        if rel.split('/').any(|part| part == "__pycache__") {
            continue;
        }
        output.push(rel);
    }
    Ok(())
}

fn print_native_inspect(skill: &NativeInspectSkill) {
    println!("Name: {}", skill.name);
    println!("Description: {}", skill.description);
    println!("Source: {}", skill.source);
    println!("Trust: {}", skill.trust);
    println!("Identifier: {}", skill.identifier);
    if !skill.tags.is_empty() {
        println!("Tags: {}", skill.tags.join(", "));
    }
    println!("Path: {}", skill.path.display());
    println!();
    println!("SKILL.md Preview:");
    println!("{}", skill.preview);
}

fn resolve_native_inspect_skill(
    context: &HermesContext,
    identifier: &str,
) -> Result<Option<NativeInspectSkill>, Box<dyn Error>> {
    let raw_config = load_raw_config(context)?;
    let hub_installed = load_hub_lock(context)?;
    let builtin_names = load_builtin_manifest(context)?;
    let installed = discover_skill_records(context, &raw_config)?;

    let mut local_matches = installed
        .iter()
        .filter(|record| skill_record_matches(record, identifier))
        .collect::<Vec<_>>();
    if local_matches.len() == 1 {
        return build_installed_inspect_skill(
            local_matches.remove(0),
            &hub_installed,
            &builtin_names,
        )
        .map(Some);
    }
    if local_matches.len() > 1 {
        return Ok(None);
    }

    let optional = discover_optional_skill_records()?;
    let mut optional_matches = optional
        .iter()
        .filter(|record| optional_skill_matches(record, identifier))
        .collect::<Vec<_>>();
    if optional_matches.len() == 1 {
        return build_optional_inspect_skill(optional_matches.remove(0)).map(Some);
    }
    if optional_matches.len() > 1 {
        return Ok(None);
    }

    Ok(None)
}

fn build_installed_inspect_skill(
    record: &SkillRecord,
    hub_installed: &HashMap<String, HubInstalledEntry>,
    builtin_names: &HashSet<String>,
) -> Result<NativeInspectSkill, Box<dyn Error>> {
    let content = fs::read_to_string(&record.skill_md)?;
    let (frontmatter, _body) = parse_frontmatter(&content);
    let source_info = classify_skill(&record.entry.name, hub_installed, builtin_names);
    let identifier = hub_installed
        .get(&record.entry.name)
        .and_then(|entry| entry.raw.get("identifier"))
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(record.entry.name.as_str())
        .to_string();
    Ok(NativeInspectSkill {
        name: record.entry.name.clone(),
        description: frontmatter_string(&frontmatter, "description")
            .unwrap_or_else(|| String::from("(no description)")),
        source: source_info.source_display,
        trust: source_info.trust,
        identifier,
        tags: extract_tags(&frontmatter),
        preview: preview_lines(&content, 50),
        path: record.skill_md.clone(),
    })
}

fn build_optional_inspect_skill(
    record: &SkillRecord,
) -> Result<NativeInspectSkill, Box<dyn Error>> {
    let content = fs::read_to_string(&record.skill_md)?;
    let (frontmatter, _body) = parse_frontmatter(&content);
    let optional_root = optional_skills_dir();
    let skill_dir = record
        .skill_md
        .parent()
        .ok_or("optional skill is missing a parent directory")?;
    let rel = skill_dir
        .strip_prefix(&optional_root)
        .map_err(|_| "optional skill path is outside optional-skills")?;
    Ok(NativeInspectSkill {
        name: record.entry.name.clone(),
        description: frontmatter_string(&frontmatter, "description")
            .unwrap_or_else(|| String::from("(no description)")),
        source: String::from("official"),
        trust: String::from("official"),
        identifier: format!("official/{}", rel.to_string_lossy().replace('\\', "/")),
        tags: extract_tags(&frontmatter),
        preview: preview_lines(&content, 50),
        path: record.skill_md.clone(),
    })
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

#[derive(Debug)]
struct BundledSyncResult {
    copied: Vec<String>,
    updated: Vec<String>,
}

#[derive(Debug)]
struct ResetSkillResult {
    ok: bool,
    message: String,
    synced: Option<BundledSyncResult>,
}

fn reset_skill(context: &HermesContext, args: SkillResetArgs) -> Result<(), Box<dyn Error>> {
    let name = validate_skill_name(&args.name)?.to_string();
    if args.restore
        && !args.yes
        && !confirm_prompt(&format!(
            "Restore '{name}' from bundled source? This will replace your current copy. [y/N]: "
        ))?
    {
        println!("Cancelled.");
        return Ok(());
    }

    let result = reset_bundled_skill(context, &name, args.restore)?;
    if !result.ok {
        return Err(result.message.into());
    }

    println!("{}", result.message);
    if let Some(synced) = result.synced {
        if !synced.copied.is_empty() {
            println!("Copied: {}", synced.copied.join(", "));
        }
        if !synced.updated.is_empty() {
            println!("Updated: {}", synced.updated.join(", "));
        }
    }
    println!();
    Ok(())
}

fn reset_bundled_skill(
    context: &HermesContext,
    name: &str,
    restore: bool,
) -> Result<ResetSkillResult, Box<dyn Error>> {
    let mut manifest = read_bundled_manifest(context)?;
    let bundled_dir = bundled_skills_dir();
    let bundled_skills = discover_bundled_skills(&bundled_dir)?;
    let bundled_by_name = bundled_skills.into_iter().collect::<HashMap<_, _>>();

    let in_manifest = manifest.contains_key(name);
    let bundled_path = bundled_by_name.get(name).cloned();
    if !in_manifest && bundled_path.is_none() {
        return Ok(ResetSkillResult {
            ok: false,
            message: format!(
                "'{name}' is not a tracked bundled skill. Nothing to reset. (Hub-installed skills use `hermes skills uninstall`.)"
            ),
            synced: None,
        });
    }

    manifest.remove(name);
    write_bundled_manifest(context, &manifest)?;

    let deleted_user_copy = if restore {
        let Some(skill_dir) = bundled_path.as_ref() else {
            return Ok(ResetSkillResult {
                ok: false,
                message: format!(
                    "'{name}' has no bundled source — manifest entry cleared but cannot restore from bundled (skill was removed upstream)."
                ),
                synced: None,
            });
        };
        let dest = bundled_skill_dest(context, &bundled_dir, skill_dir)?;
        if dest.exists() {
            fs::remove_dir_all(&dest)?;
            true
        } else {
            false
        }
    } else {
        false
    };

    let synced = sync_bundled_skills(context, true)?;
    let message = if restore && deleted_user_copy {
        format!("Restored '{name}' from bundled source.")
    } else if restore {
        format!("Restored '{name}' (no prior user copy, re-copied from bundled).")
    } else {
        format!(
            "Cleared manifest entry for '{name}'. Future `hermes update` runs will re-baseline against your current copy and accept upstream changes."
        )
    };

    Ok(ResetSkillResult {
        ok: true,
        message,
        synced: Some(synced),
    })
}

fn sync_bundled_skills(
    context: &HermesContext,
    quiet: bool,
) -> Result<BundledSyncResult, Box<dyn Error>> {
    let bundled_dir = bundled_skills_dir();
    if !bundled_dir.exists() {
        return Ok(BundledSyncResult {
            copied: Vec::new(),
            updated: Vec::new(),
        });
    }

    let skills_root = context.hermes_home().join("skills");
    fs::create_dir_all(&skills_root)?;
    let mut manifest = read_bundled_manifest(context)?;
    let bundled_skills = discover_bundled_skills(&bundled_dir)?;
    let bundled_names = bundled_skills
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<HashSet<_>>();

    let mut copied = Vec::new();
    let mut updated = Vec::new();
    let mut user_modified = Vec::new();

    for (skill_name, skill_src) in &bundled_skills {
        let dest = bundled_skill_dest(context, &bundled_dir, skill_src)?;
        let bundled_hash = dir_hash(skill_src)?;

        if !manifest.contains_key(skill_name) {
            if dest.exists() {
                if dir_hash(&dest)? == bundled_hash {
                    manifest.insert(skill_name.clone(), bundled_hash);
                } else if !quiet {
                    println!(
                        "  ⚠ {skill_name}: bundled version shipped but a local skill with this name already exists — keeping yours"
                    );
                }
            } else {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent)?;
                }
                copy_dir_all(skill_src, &dest)?;
                copied.push(skill_name.clone());
                manifest.insert(skill_name.clone(), bundled_hash);
            }
            continue;
        }

        if !dest.exists() {
            continue;
        }

        let origin_hash = manifest.get(skill_name).cloned().unwrap_or_default();
        let user_hash = dir_hash(&dest)?;
        if origin_hash.is_empty() {
            manifest.insert(skill_name.clone(), user_hash.clone());
            continue;
        }
        if user_hash != origin_hash {
            user_modified.push(skill_name.clone());
            if !quiet {
                println!("  ~ {skill_name} (user-modified, skipping)");
            }
            continue;
        }
        if bundled_hash == origin_hash {
            continue;
        }

        let backup = dest.with_extension("bak");
        if backup.exists() {
            fs::remove_dir_all(&backup)?;
        }
        fs::rename(&dest, &backup)?;
        let update_result = (|| -> Result<(), Box<dyn Error>> {
            copy_dir_all(skill_src, &dest)?;
            manifest.insert(skill_name.clone(), bundled_hash);
            updated.push(skill_name.clone());
            fs::remove_dir_all(&backup)?;
            Ok(())
        })();
        if let Err(error) = update_result {
            if backup.exists() && !dest.exists() {
                fs::rename(&backup, &dest)?;
            }
            return Err(error);
        }
    }

    let mut cleaned = manifest
        .keys()
        .filter(|name| !bundled_names.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    cleaned.sort();
    for name in &cleaned {
        manifest.remove(name);
    }

    copy_bundled_descriptions(&bundled_dir, &skills_root)?;
    write_bundled_manifest(context, &manifest)?;

    Ok(BundledSyncResult { copied, updated })
}

fn print_snapshot(context: &HermesContext, args: SkillSnapshotArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        Some(SkillSnapshotCommand::Export { output }) => export_skill_snapshot(context, &output),
        Some(SkillSnapshotCommand::Import { input, force }) => {
            import_skill_snapshot(context, &input, force)
        }
        None => {
            println!("Usage: hermes skills snapshot [export|import]");
            println!();
            Ok(())
        }
    }
}

fn print_taps(context: &HermesContext, args: SkillTapArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        Some(SkillTapCommand::List) => list_taps(context),
        Some(SkillTapCommand::Add { repo }) => add_tap(context, &repo),
        Some(SkillTapCommand::Remove { name }) => remove_tap(context, &name),
        None => {
            println!("Usage: hermes skills tap [list|add|remove]");
            println!();
            Ok(())
        }
    }
}

fn export_skill_snapshot(context: &HermesContext, output_path: &str) -> Result<(), Box<dyn Error>> {
    let installed = load_hub_lock(context)?;
    let taps = load_taps(context)?;
    let tap_count = taps.len();

    let mut names = installed.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let skills = names
        .into_iter()
        .filter_map(|name| installed.get(&name).map(|entry| (name, entry)))
        .map(|(name, entry)| {
            let mut item = JsonMap::new();
            item.insert("name".to_string(), JsonValue::String(name));
            item.insert(
                "source".to_string(),
                JsonValue::String(entry.source.clone()),
            );
            item.insert(
                "identifier".to_string(),
                entry
                    .raw
                    .get("identifier")
                    .and_then(JsonValue::as_str)
                    .map(str::to_string)
                    .map(JsonValue::String)
                    .unwrap_or(JsonValue::String(String::new())),
            );
            item.insert(
                "category".to_string(),
                JsonValue::String(category_from_install_path(&entry.install_path)),
            );
            JsonValue::Object(item)
        })
        .collect::<Vec<_>>();

    let mut snapshot = JsonMap::new();
    snapshot.insert(
        "hermes_version".to_string(),
        JsonValue::String(env!("CARGO_PKG_VERSION").to_string()),
    );
    snapshot.insert("exported_at".to_string(), JsonValue::String(iso8601_now()));
    snapshot.insert("skills".to_string(), JsonValue::Array(skills));
    snapshot.insert(
        "taps".to_string(),
        JsonValue::Array(
            taps.into_iter()
                .map(|entry| JsonValue::Object(entry.raw))
                .collect(),
        ),
    );

    let payload = format!(
        "{}\n",
        serde_json::to_string_pretty(&JsonValue::Object(snapshot))?
    );
    if output_path == "-" {
        print!("{payload}");
        io::stdout().flush()?;
        return Ok(());
    }

    let output = PathBuf::from(output_path);
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output, payload)?;
    println!("Snapshot exported: {}", output.display());
    println!("{} skill(s), {} tap(s)", installed.len(), tap_count);
    Ok(())
}

fn import_skill_snapshot(
    context: &HermesContext,
    input_path: &str,
    force: bool,
) -> Result<(), Box<dyn Error>> {
    let input = Path::new(input_path);
    if !input.exists() {
        return Err(format!("File not found: {}", input.display()).into());
    }

    let snapshot = serde_json::from_str::<JsonValue>(&fs::read_to_string(input)?)?;
    let taps = snapshot
        .get("taps")
        .and_then(JsonValue::as_array)
        .cloned()
        .unwrap_or_default();
    if !taps.is_empty() {
        import_snapshot_taps(context, &taps)?;
        println!("Restored {} tap(s)", taps.len());
    }

    let skills = snapshot
        .get("skills")
        .and_then(JsonValue::as_array)
        .cloned()
        .unwrap_or_default();
    if skills.is_empty() {
        println!("No skills in snapshot to install.");
        println!();
        return Ok(());
    }

    println!("Importing {} skill(s) from snapshot...", skills.len());
    println!();
    for item in skills {
        let Some(object) = item.as_object() else {
            continue;
        };
        let display_name = object
            .get("name")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("?");
        let identifier = object
            .get("identifier")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(identifier) = identifier else {
            println!("Skipping entry with no identifier: {display_name}");
            continue;
        };

        let category = object
            .get("category")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let mut passthrough = vec![identifier.to_string()];
        if let Some(category) = category {
            passthrough.push(String::from("--category"));
            passthrough.push(category.to_string());
        }
        if force {
            passthrough.push(String::from("--force"));
        }

        println!("--- {display_name} ---");
        bridge_prefixed("install", &passthrough)?;
    }

    println!("Snapshot import complete.");
    println!();
    Ok(())
}

fn list_taps(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let taps = load_taps(context)?;
    if taps.is_empty() {
        println!("No custom taps configured. Using default sources only.");
        println!();
        return Ok(());
    }

    println!("{:<30} Path", "Repo");
    println!("{:<30} ----", "------------------------------");
    for entry in taps {
        let path = entry
            .raw
            .get("path")
            .and_then(JsonValue::as_str)
            .unwrap_or("skills/");
        println!("{:<30} {}", truncate(&entry.repo, 30), path);
    }
    println!();
    Ok(())
}

fn add_tap(context: &HermesContext, raw_repo: &str) -> Result<(), Box<dyn Error>> {
    let repo = validate_tap_repo(raw_repo)?;
    let mut taps = load_taps(context)?;
    if taps.iter().any(|entry| entry.repo == repo) {
        println!("Tap already exists: {repo}");
        println!();
        return Ok(());
    }

    let mut raw = JsonMap::new();
    raw.insert("repo".to_string(), JsonValue::String(repo.to_string()));
    raw.insert(
        "path".to_string(),
        JsonValue::String(String::from("skills/")),
    );
    taps.push(TapEntry {
        repo: repo.to_string(),
        raw,
    });
    save_taps(context, &taps)?;
    println!("Added tap: {repo}");
    println!();
    Ok(())
}

fn remove_tap(context: &HermesContext, raw_repo: &str) -> Result<(), Box<dyn Error>> {
    let repo = validate_tap_repo(raw_repo)?;
    let taps = load_taps(context)?;
    let original_len = taps.len();
    let filtered = taps
        .into_iter()
        .filter(|entry| entry.repo != repo)
        .collect::<Vec<_>>();
    if filtered.len() == original_len {
        return Err(format!("Tap not found: {repo}").into());
    }
    save_taps(context, &filtered)?;
    println!("Removed tap: {repo}");
    println!();
    Ok(())
}

fn import_snapshot_taps(
    context: &HermesContext,
    items: &[JsonValue],
) -> Result<(), Box<dyn Error>> {
    let mut existing = load_taps(context)?;
    let mut seen = existing
        .iter()
        .map(|entry| entry.repo.clone())
        .collect::<HashSet<_>>();

    for item in items {
        let Some(raw) = item.as_object() else {
            continue;
        };
        let repo = raw
            .get("repo")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(repo) = repo else {
            continue;
        };
        let repo = validate_tap_repo(repo)?.to_string();
        if !seen.insert(repo.clone()) {
            continue;
        }
        let path = raw
            .get("path")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("skills/")
            .to_string();
        let mut stored = raw.clone();
        stored.insert("repo".to_string(), JsonValue::String(repo.clone()));
        stored.insert("path".to_string(), JsonValue::String(path));
        existing.push(TapEntry { repo, raw: stored });
    }

    save_taps(context, &existing)
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
    Ok(discover_skill_records(context, raw_config)?
        .into_iter()
        .map(|record| record.entry)
        .collect())
}

fn discover_skill_records(
    context: &HermesContext,
    raw_config: &YamlValue,
) -> Result<Vec<SkillRecord>, Box<dyn Error>> {
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
            skills.push(SkillRecord {
                entry: SkillEntry {
                    category: category_from_path(&dir, &skill_md),
                    name,
                },
                skill_md,
            });
        }
    }

    skills.sort_by(|left, right| {
        let left_key = (
            left.entry.category.as_deref().unwrap_or_default(),
            left.entry.name.as_str(),
        );
        let right_key = (
            right.entry.category.as_deref().unwrap_or_default(),
            right.entry.name.as_str(),
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

fn validate_skill_identifier(raw: &str) -> Result<&str, Box<dyn Error>> {
    let identifier = raw.trim();
    if identifier.is_empty() {
        return Err("skill identifier cannot be empty".into());
    }
    Ok(identifier)
}

fn skill_record_matches(record: &SkillRecord, identifier: &str) -> bool {
    let trimmed = identifier.trim();
    if trimmed.eq_ignore_ascii_case(&record.entry.name) {
        return true;
    }
    let Some(skill_dir) = record.skill_md.parent() else {
        return false;
    };
    skill_dir
        .file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case(trimmed))
        .unwrap_or(false)
}

fn optional_skill_matches(record: &SkillRecord, identifier: &str) -> bool {
    let trimmed = identifier.trim();
    if trimmed.eq_ignore_ascii_case(&record.entry.name) {
        return true;
    }
    let Some(skill_dir) = record.skill_md.parent() else {
        return false;
    };
    let optional_root = optional_skills_dir();
    let rel = match skill_dir.strip_prefix(&optional_root) {
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => return false,
    };
    trimmed == format!("official/{rel}") || trimmed == rel
}

fn optional_skills_dir() -> PathBuf {
    std::env::var_os("HERMES_OPTIONAL_SKILLS")
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root().join("optional-skills"))
}

fn discover_optional_skill_records() -> Result<Vec<SkillRecord>, Box<dyn Error>> {
    let root = optional_skills_dir();
    if !root.is_dir() {
        return Ok(Vec::new());
    }

    let mut seen = HashSet::new();
    let mut skill_files = Vec::new();
    collect_skill_files(&root, &mut skill_files)?;
    let mut skills = Vec::new();
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
        skills.push(SkillRecord {
            entry: SkillEntry {
                category: category_from_path(&root, &skill_md),
                name,
            },
            skill_md,
        });
    }

    skills.sort_by(|left, right| {
        let left_key = (
            left.entry.category.as_deref().unwrap_or_default(),
            left.entry.name.as_str(),
        );
        let right_key = (
            right.entry.category.as_deref().unwrap_or_default(),
            right.entry.name.as_str(),
        );
        left_key.cmp(&right_key)
    });
    Ok(skills)
}

fn frontmatter_string(frontmatter: &YamlMapping, key: &str) -> Option<String> {
    frontmatter
        .get(&yaml_key(key))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn extract_tags(frontmatter: &YamlMapping) -> Vec<String> {
    if let Some(tags) = frontmatter
        .get(&yaml_key("metadata"))
        .and_then(YamlValue::as_mapping)
        .and_then(|mapping| mapping.get(&yaml_key("hermes")))
        .and_then(YamlValue::as_mapping)
        .and_then(|mapping| mapping.get(&yaml_key("tags")))
    {
        let collected = yaml_tags(tags);
        if !collected.is_empty() {
            return collected;
        }
    }
    frontmatter
        .get(&yaml_key("tags"))
        .map(yaml_tags)
        .unwrap_or_default()
}

fn yaml_tags(value: &YamlValue) -> Vec<String> {
    match value {
        YamlValue::Sequence(items) => items
            .iter()
            .filter_map(YamlValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        YamlValue::String(text) => text
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

fn preview_lines(content: &str, limit: usize) -> String {
    let lines = content.lines().collect::<Vec<_>>();
    let visible = lines.iter().take(limit).copied().collect::<Vec<_>>();
    let mut preview = visible.join("\n");
    if lines.len() > limit {
        preview.push_str(&format!("\n\n... ({} more lines)", lines.len() - limit));
    }
    preview
}

struct SkillSourceInfo {
    filter: SkillsSourceFilter,
    source_display: String,
    trust: String,
}

fn bundled_skills_dir() -> PathBuf {
    std::env::var_os("HERMES_BUNDLED_SKILLS")
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root().join("skills"))
}

fn bundled_manifest_path(context: &HermesContext) -> PathBuf {
    context
        .hermes_home()
        .join("skills")
        .join(".bundled_manifest")
}

fn read_bundled_manifest(
    context: &HermesContext,
) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let path = bundled_manifest_path(context);
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let mut manifest = HashMap::new();
    for line in fs::read_to_string(path)?.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some((name, hash)) = trimmed.split_once(':') {
            manifest.insert(name.trim().to_string(), hash.trim().to_string());
        } else {
            manifest.insert(trimmed.to_string(), String::new());
        }
    }
    Ok(manifest)
}

fn write_bundled_manifest(
    context: &HermesContext,
    manifest: &HashMap<String, String>,
) -> Result<(), Box<dyn Error>> {
    let path = bundled_manifest_path(context);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut names = manifest.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut rendered = String::new();
    for name in names {
        rendered.push_str(&format!(
            "{}:{}\n",
            name,
            manifest.get(&name).map(String::as_str).unwrap_or("")
        ));
    }
    fs::write(path, rendered)?;
    Ok(())
}

fn discover_bundled_skills(bundled_dir: &Path) -> Result<Vec<(String, PathBuf)>, Box<dyn Error>> {
    if !bundled_dir.exists() {
        return Ok(Vec::new());
    }
    let mut skill_files = Vec::new();
    collect_skill_files(bundled_dir, &mut skill_files)?;
    let mut result = Vec::new();
    for skill_md in skill_files {
        let Some(skill_dir) = skill_md.parent() else {
            continue;
        };
        let content = fs::read_to_string(&skill_md).unwrap_or_default();
        let (frontmatter, _body) = parse_frontmatter(&content);
        let fallback = skill_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("skill");
        let name = frontmatter
            .get(&yaml_key("name"))
            .and_then(YamlValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(fallback)
            .to_string();
        result.push((name, skill_dir.to_path_buf()));
    }
    Ok(result)
}

fn bundled_skill_dest(
    context: &HermesContext,
    bundled_dir: &Path,
    skill_dir: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    let rel = skill_dir.strip_prefix(bundled_dir)?;
    Ok(context.hermes_home().join("skills").join(rel))
}

fn dir_hash(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut files = Vec::new();
    collect_all_files(path, path, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Md5Context::new();
    for (relative, file) in files {
        hasher.consume(relative.as_bytes());
        hasher.consume(fs::read(file)?);
    }
    Ok(format!("{:x}", hasher.compute()))
}

fn collect_all_files(
    root: &Path,
    current: &Path,
    output: &mut Vec<(String, PathBuf)>,
) -> Result<(), Box<dyn Error>> {
    if !current.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_all_files(root, &path, output)?;
        } else if path.is_file() {
            let relative = path
                .strip_prefix(root)?
                .to_string_lossy()
                .replace('\\', "/");
            output.push((relative, path));
        }
    }
    Ok(())
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

fn copy_bundled_descriptions(bundled_dir: &Path, skills_root: &Path) -> Result<(), Box<dyn Error>> {
    if !bundled_dir.exists() {
        return Ok(());
    }
    let mut stack = vec![bundled_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if entry.file_name().to_string_lossy() != "DESCRIPTION.md" {
                continue;
            }
            let rel = path.strip_prefix(bundled_dir)?;
            let dest = skills_root.join(rel);
            if dest.exists() {
                continue;
            }
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(path, dest)?;
        }
    }
    Ok(())
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

fn load_taps(context: &HermesContext) -> Result<Vec<TapEntry>, Box<dyn Error>> {
    let taps_path = taps_path(context);
    if !taps_path.exists() {
        return Ok(Vec::new());
    }
    let parsed = serde_json::from_str::<JsonValue>(&fs::read_to_string(taps_path)?)?;
    let Some(items) = parsed.get("taps").and_then(JsonValue::as_array) else {
        return Ok(Vec::new());
    };

    let mut taps = Vec::new();
    for item in items {
        let Some(raw) = item.as_object() else {
            continue;
        };
        let repo = raw
            .get("repo")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let Some(repo) = repo else {
            continue;
        };
        taps.push(TapEntry {
            repo,
            raw: raw.clone(),
        });
    }
    Ok(taps)
}

fn save_taps(context: &HermesContext, taps: &[TapEntry]) -> Result<(), Box<dyn Error>> {
    let path = taps_path(context);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let payload = JsonValue::Object(JsonMap::from_iter([(
        "taps".to_string(),
        JsonValue::Array(
            taps.iter()
                .map(|entry| JsonValue::Object(entry.raw.clone()))
                .collect(),
        ),
    )]));
    fs::write(
        path,
        format!("{}\n", serde_json::to_string_pretty(&payload)?),
    )?;
    Ok(())
}

fn taps_path(context: &HermesContext) -> PathBuf {
    context
        .hermes_home()
        .join("skills")
        .join(".hub")
        .join("taps.json")
}

fn validate_tap_repo(raw: &str) -> Result<&str, Box<dyn Error>> {
    let repo = raw.trim();
    if repo.is_empty() {
        return Err("tap repo cannot be empty".into());
    }
    let Some((owner, name)) = repo.split_once('/') else {
        return Err("tap repo must be in owner/repo format".into());
    };
    if !is_valid_repo_segment(owner) || !is_valid_repo_segment(name) {
        return Err("tap repo must be in owner/repo format".into());
    }
    Ok(repo)
}

fn is_valid_repo_segment(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
}

fn category_from_install_path(install_path: &str) -> String {
    let trimmed = install_path.trim();
    if trimmed.is_empty() || !trimmed.contains('/') {
        return String::new();
    }
    Path::new(trimmed)
        .parent()
        .filter(|parent| parent.as_os_str() != ".")
        .map(|parent| parent.to_string_lossy().to_string())
        .unwrap_or_default()
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
    fn tap_round_trip_add_list_remove() {
        let home = temp_path("taps");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        add_tap(&context, "owner/repo").unwrap();
        let taps = load_taps(&context).unwrap();
        assert_eq!(taps.len(), 1);
        assert_eq!(taps[0].repo, "owner/repo");
        assert_eq!(
            taps[0].raw.get("path").and_then(JsonValue::as_str).unwrap(),
            "skills/"
        );

        let duplicate = add_tap(&context, "owner/repo").unwrap();
        let _ = duplicate;
        let taps = load_taps(&context).unwrap();
        assert_eq!(taps.len(), 1);

        remove_tap(&context, "owner/repo").unwrap();
        assert!(load_taps(&context).unwrap().is_empty());
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn export_snapshot_writes_skills_and_taps() {
        let home = temp_path("snapshot-export");
        let out = home.join("snapshot.json");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let hub_dir = home.join("skills").join(".hub");
        fs::create_dir_all(&hub_dir).unwrap();
        fs::write(
            hub_dir.join("lock.json"),
            r#"{"version":1,"installed":{"hub-skill":{"source":"official","identifier":"official/dev/hub-skill","trust_level":"trusted","install_path":"dev/hub-skill","files":["SKILL.md"]}}}"#,
        )
        .unwrap();
        fs::write(
            hub_dir.join("taps.json"),
            r#"{"taps":[{"repo":"owner/repo","path":"skills/"}]}"#,
        )
        .unwrap();

        export_skill_snapshot(&context, out.to_str().unwrap()).unwrap();

        let parsed = serde_json::from_str::<JsonValue>(&fs::read_to_string(out).unwrap()).unwrap();
        let skills = parsed.get("skills").and_then(JsonValue::as_array).unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(
            skills[0].get("name").and_then(JsonValue::as_str),
            Some("hub-skill")
        );
        assert_eq!(
            skills[0].get("category").and_then(JsonValue::as_str),
            Some("dev")
        );
        let taps = parsed.get("taps").and_then(JsonValue::as_array).unwrap();
        assert_eq!(taps.len(), 1);
        assert_eq!(
            taps[0].get("repo").and_then(JsonValue::as_str),
            Some("owner/repo")
        );
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    #[cfg(unix)]
    fn import_snapshot_restores_taps_and_bridges_installs() {
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

        let home = temp_path("snapshot-import");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let input = home.join("snapshot.json");
        fs::create_dir_all(home.join("skills").join(".hub")).unwrap();
        fs::write(
            &input,
            r#"{
  "skills": [
    {"name": "alpha", "identifier": "official/dev/alpha", "category": "dev"},
    {"name": "beta", "identifier": "official/beta", "category": ""}
  ],
  "taps": [
    {"repo": "owner/repo", "path": "skills/"}
  ]
}"#,
        )
        .unwrap();

        set_env_var("HERMES_SKILLS_PYTHON", &fake_python);
        import_skill_snapshot(&context, input.to_str().unwrap(), true).unwrap();

        let taps = load_taps(&context).unwrap();
        assert_eq!(taps.len(), 1);
        assert_eq!(taps[0].repo, "owner/repo");

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("action=install argv=official/dev/alpha --category dev --force"));
        assert!(output.contains("action=install argv=official/beta --force"));

        remove_env_var("HERMES_SKILLS_PYTHON");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn reset_restore_recopies_bundled_skill() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("reset-restore-home");
        let bundled = temp_path("reset-restore-bundled");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let bundled_skill = bundled.join("dev").join("demo");
        fs::create_dir_all(&bundled_skill).unwrap();
        fs::write(
            bundled_skill.join("SKILL.md"),
            "---\nname: demo\ndescription: Bundled\n---\nBundled\n",
        )
        .unwrap();

        let local_skill = home.join("skills").join("dev").join("demo");
        fs::create_dir_all(&local_skill).unwrap();
        fs::write(
            local_skill.join("SKILL.md"),
            "---\nname: demo\ndescription: Local\n---\nLocal\n",
        )
        .unwrap();
        fs::write(
            home.join("skills").join(".bundled_manifest"),
            "demo:stalehash\n",
        )
        .unwrap();

        set_env_var("HERMES_BUNDLED_SKILLS", &bundled);
        let result = reset_bundled_skill(&context, "demo", true).unwrap();
        assert!(result.ok);
        assert!(result.message.contains("Restored 'demo'"));
        let restored = fs::read_to_string(local_skill.join("SKILL.md")).unwrap();
        assert!(restored.contains("Bundled"));
        let manifest = read_bundled_manifest(&context).unwrap();
        assert_eq!(
            manifest.get("demo").cloned().unwrap(),
            dir_hash(&bundled_skill).unwrap()
        );

        remove_env_var("HERMES_BUNDLED_SKILLS");
        let _ = fs::remove_dir_all(home);
        let _ = fs::remove_dir_all(bundled);
    }

    #[test]
    fn inspect_resolves_local_skill_natively() {
        let home = temp_path("inspect-local");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let skill_dir = home.join("skills").join("local-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: local-skill\ndescription: Native local skill\ntags: [alpha, beta]\n---\nline1\nline2\n",
        )
        .unwrap();

        let inspected = resolve_native_inspect_skill(&context, "local-skill")
            .unwrap()
            .unwrap();
        assert_eq!(inspected.name, "local-skill");
        assert_eq!(inspected.description, "Native local skill");
        assert_eq!(inspected.source, "local");
        assert_eq!(inspected.trust, "local");
        assert_eq!(inspected.identifier, "local-skill");
        assert_eq!(
            inspected.tags,
            vec![String::from("alpha"), String::from("beta")]
        );
        assert!(inspected.preview.contains("line1"));

        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn inspect_resolves_optional_official_skill_natively() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("inspect-optional-home");
        let optional = temp_path("inspect-optional-src");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let skill_dir = optional.join("research").join("demo");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: demo\ndescription: Optional demo\nmetadata:\n  hermes:\n    tags:\n      - optional\n      - official\n---\npreview\n",
        )
        .unwrap();

        set_env_var("HERMES_OPTIONAL_SKILLS", &optional);
        let inspected = resolve_native_inspect_skill(&context, "official/research/demo")
            .unwrap()
            .unwrap();
        assert_eq!(inspected.name, "demo");
        assert_eq!(inspected.source, "official");
        assert_eq!(inspected.trust, "official");
        assert_eq!(inspected.identifier, "official/research/demo");
        assert_eq!(
            inspected.tags,
            vec![String::from("optional"), String::from("official")]
        );

        remove_env_var("HERMES_OPTIONAL_SKILLS");
        let _ = fs::remove_dir_all(home);
        let _ = fs::remove_dir_all(optional);
    }

    #[test]
    #[cfg(unix)]
    fn inspect_bridges_when_native_resolution_misses() {
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

        let home = temp_path("inspect-bridge");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));

        set_env_var("HERMES_SKILLS_PYTHON", &fake_python);
        inspect_skill_command(&context, "owner/repo/remote-skill").unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("action=inspect argv=owner/repo/remote-skill"));

        remove_env_var("HERMES_SKILLS_PYTHON");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    fn collect_official_candidates_detects_updates() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("official-check-home");
        let optional = temp_path("official-check-src");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let install_dir = home.join("skills").join("research").join("demo");
        let source_dir = optional.join("research").join("demo");
        fs::create_dir_all(&install_dir).unwrap();
        fs::create_dir_all(&source_dir).unwrap();
        fs::create_dir_all(home.join("skills").join(".hub")).unwrap();

        fs::write(
            install_dir.join("SKILL.md"),
            "---\nname: demo\ndescription: Old\n---\nold\n",
        )
        .unwrap();
        fs::write(
            source_dir.join("SKILL.md"),
            "---\nname: demo\ndescription: New\n---\nnew\n",
        )
        .unwrap();
        fs::write(
            home.join("skills").join(".hub").join("lock.json"),
            r#"{"version":1,"installed":{"demo":{"source":"official","identifier":"official/research/demo","trust_level":"builtin","scan_verdict":"safe","content_hash":"sha256:stale","install_path":"research/demo","files":["SKILL.md"]}}}"#,
        )
        .unwrap();

        set_env_var("HERMES_OPTIONAL_SKILLS", &optional);
        let installed = load_hub_lock(&context).unwrap();
        let candidates =
            collect_official_skill_candidates(&installed, &[String::from("demo")]).unwrap();
        let candidate = candidates.get("demo").unwrap();
        assert_eq!(candidate.name, "demo");
        assert_eq!(candidate.source, "official");
        assert_eq!(candidate.install_path, "research/demo");
        assert_eq!(candidate.current_hash, "sha256:stale");
        assert_ne!(candidate.latest_hash, candidate.current_hash);
        assert_eq!(candidate.files, vec![String::from("SKILL.md")]);

        remove_env_var("HERMES_OPTIONAL_SKILLS");
        let _ = fs::remove_dir_all(home);
        let _ = fs::remove_dir_all(optional);
    }

    #[test]
    fn update_native_official_skill_restores_files_and_lock_hash() {
        let _guard = test_env_lock().lock().unwrap();
        let home = temp_path("official-update-home");
        let optional = temp_path("official-update-src");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        let install_dir = home.join("skills").join("research").join("demo");
        let source_dir = optional.join("research").join("demo");
        fs::create_dir_all(&install_dir).unwrap();
        fs::create_dir_all(&source_dir).unwrap();
        fs::create_dir_all(home.join("skills").join(".hub")).unwrap();

        fs::write(
            install_dir.join("SKILL.md"),
            "---\nname: demo\ndescription: Old\n---\nold\n",
        )
        .unwrap();
        fs::write(install_dir.join("old.txt"), "stale\n").unwrap();
        fs::write(
            source_dir.join("SKILL.md"),
            "---\nname: demo\ndescription: New\n---\nnew\n",
        )
        .unwrap();
        fs::write(source_dir.join("extra.txt"), "fresh\n").unwrap();
        fs::write(
            home.join("skills").join(".hub").join("lock.json"),
            r#"{"version":1,"installed":{"demo":{"source":"official","identifier":"official/research/demo","trust_level":"builtin","scan_verdict":"safe","content_hash":"sha256:stale","install_path":"research/demo","files":["SKILL.md"]}}}"#,
        )
        .unwrap();

        set_env_var("HERMES_OPTIONAL_SKILLS", &optional);
        update_skills_command(&context, &[]).unwrap();

        let installed = load_hub_lock(&context).unwrap();
        let entry = installed.get("demo").unwrap();
        let saved_hash = entry
            .raw
            .get("content_hash")
            .and_then(JsonValue::as_str)
            .unwrap();
        assert_eq!(
            saved_hash,
            bundle_content_hash_from_dir(&source_dir).unwrap()
        );
        assert_eq!(
            fs::read_to_string(install_dir.join("SKILL.md")).unwrap(),
            fs::read_to_string(source_dir.join("SKILL.md")).unwrap()
        );
        assert_eq!(
            fs::read_to_string(install_dir.join("extra.txt")).unwrap(),
            "fresh\n"
        );
        assert!(!install_dir.join("old.txt").exists());

        remove_env_var("HERMES_OPTIONAL_SKILLS");
        let _ = fs::remove_dir_all(home);
        let _ = fs::remove_dir_all(optional);
    }

    #[test]
    #[cfg(unix)]
    fn check_bridges_when_non_official_sources_are_present() {
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

        let home = temp_path("check-bridge");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(home.join("skills").join(".hub")).unwrap();
        fs::write(
            home.join("skills").join(".hub").join("lock.json"),
            r#"{"version":1,"installed":{"demo":{"source":"github","identifier":"owner/repo/demo","trust_level":"community","scan_verdict":"safe","content_hash":"sha256:stale","install_path":"demo","files":["SKILL.md"]}}}"#,
        )
        .unwrap();

        set_env_var("HERMES_SKILLS_PYTHON", &fake_python);
        check_skills_command(&context, &[]).unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("action=check argv="));

        remove_env_var("HERMES_SKILLS_PYTHON");
        let _ = fs::remove_dir_all(home);
    }

    #[test]
    #[cfg(unix)]
    fn update_bridges_when_non_official_sources_are_present() {
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

        let home = temp_path("update-bridge");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(home.join("skills").join(".hub")).unwrap();
        fs::write(
            home.join("skills").join(".hub").join("lock.json"),
            r#"{"version":1,"installed":{"demo":{"source":"github","identifier":"owner/repo/demo","trust_level":"community","scan_verdict":"safe","content_hash":"sha256:stale","install_path":"demo","files":["SKILL.md"]}}}"#,
        )
        .unwrap();

        set_env_var("HERMES_SKILLS_PYTHON", &fake_python);
        update_skills_command(&context, &[]).unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("action=update argv="));

        remove_env_var("HERMES_SKILLS_PYTHON");
        let _ = fs::remove_dir_all(home);
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
