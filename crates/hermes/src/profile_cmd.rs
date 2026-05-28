use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Mutex;

use clap::{Args, Subcommand};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use hermes_core::HermesContext;
use serde_yaml::Value as YamlValue;
use tar::{Archive, Builder, EntryType, Header};
use tempfile::TempDir;

use crate::gateway_cmd::{
    GatewayArgs, GatewayCommand, GatewayServiceArgs, GatewaySystemArgs, print_gateway,
};
use crate::python_bridge::project_root;

const RESERVED_ALIAS_NAMES: &[&str] = &["hermes", "default", "test", "tmp", "root", "sudo"];
const HERMES_SUBCOMMANDS: &[&str] = &[
    "chat",
    "model",
    "gateway",
    "platforms",
    "setup",
    "whatsapp",
    "login",
    "logout",
    "status",
    "cron",
    "doctor",
    "dump",
    "config",
    "pairing",
    "skills",
    "tools",
    "mcp",
    "sessions",
    "insights",
    "version",
    "update",
    "uninstall",
    "profile",
    "plugins",
    "honcho",
    "acp",
];

#[derive(Subcommand, Debug)]
pub enum ProfileCommand {
    Current,
    List,
    Path { name: Option<String> },
    Create(ProfileCreateArgs),
    Use { name: String },
    Delete(ProfileDeleteArgs),
    Show { name: String },
    Alias(ProfileAliasArgs),
    Rename { old_name: String, new_name: String },
    Export(ProfileExportArgs),
    Import(ProfileImportArgs),
}

#[derive(Args, Debug, Clone)]
pub struct ProfileCreateArgs {
    pub name: String,
    #[arg(long, default_value_t = false)]
    pub clone: bool,
    #[arg(long = "clone-all", default_value_t = false)]
    pub clone_all: bool,
    #[arg(long = "clone-from")]
    pub clone_from: Option<String>,
    #[arg(long, default_value_t = false)]
    pub no_alias: bool,
}

#[derive(Args, Debug, Clone)]
pub struct ProfileDeleteArgs {
    pub name: String,
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,
}

#[derive(Args, Debug, Clone)]
pub struct ProfileAliasArgs {
    pub name: String,
    #[arg(long, default_value_t = false)]
    pub remove: bool,
    #[arg(long = "name")]
    pub alias_name: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct ProfileExportArgs {
    pub name: String,
    #[arg(short = 'o', long = "output")]
    pub output: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct ProfileImportArgs {
    pub archive: String,
    #[arg(long = "name")]
    pub import_name: Option<String>,
}

#[derive(Debug, Clone)]
struct ProfileRow {
    name: String,
    path: PathBuf,
    is_default: bool,
    gateway_running: bool,
    model: Option<String>,
    provider: Option<String>,
    has_env: bool,
    skill_count: usize,
    alias_path: Option<PathBuf>,
}

pub fn print_profile(
    context: &HermesContext,
    command: Option<ProfileCommand>,
) -> Result<(), Box<dyn Error>> {
    match command {
        None => print_profile_current(context),
        Some(ProfileCommand::Current) => print_profile_current(context),
        Some(ProfileCommand::List) => print_profile_list(context),
        Some(ProfileCommand::Path { name }) => {
            let selected = name.unwrap_or_else(|| context.current_profile_name());
            println!("profile={selected}");
            println!("path={}", context.profile_dir(&selected)?.display());
            Ok(())
        }
        Some(ProfileCommand::Create(args)) => create_profile_command(context, args),
        Some(ProfileCommand::Use { name }) => {
            context.set_active_profile(&name)?;
            println!("active_profile={}", context.active_profile());
            Ok(())
        }
        Some(ProfileCommand::Delete(args)) => delete_profile_command(context, args),
        Some(ProfileCommand::Show { name }) => show_profile_command(context, &name),
        Some(ProfileCommand::Alias(args)) => alias_profile_command(context, args),
        Some(ProfileCommand::Rename { old_name, new_name }) => {
            rename_profile_command(context, &old_name, &new_name)
        }
        Some(ProfileCommand::Export(args)) => export_profile_command(context, args),
        Some(ProfileCommand::Import(args)) => import_profile_command(context, args),
    }
}

fn profile_current_lines(context: &HermesContext) -> Vec<String> {
    vec![
        format!("active_profile={}", context.active_profile()),
        format!("current_profile={}", context.current_profile_name()),
        format!("display_home={}", context.display_hermes_home()),
    ]
}

fn print_profile_current(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    for line in profile_current_lines(context) {
        println!("{line}");
    }
    Ok(())
}

fn print_profile_list(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let rows = list_profiles(context)?;
    if rows.is_empty() {
        println!("No profiles found.");
        return Ok(());
    }

    let active = context.current_profile_name();
    println!();
    println!(" {:<16} {:<28} {:<12} Alias", "Profile", "Model", "Gateway");
    println!(
        " {:<16} {:<28} {:<12} {:<12}",
        "───────────────", "───────────────────────────", "───────────", "────────────"
    );
    for row in rows {
        let marker = if row.name == active { "◆" } else { " " };
        let model = row.model.as_deref().unwrap_or("—");
        let model = truncate(model, 26);
        let gateway = if row.gateway_running {
            "running"
        } else {
            "stopped"
        };
        let alias = if row.is_default {
            "—".to_string()
        } else if row.alias_path.is_some() {
            row.name.clone()
        } else {
            "—".to_string()
        };
        println!(
            "{}{:<15} {:<28} {:<12} {}",
            marker, row.name, model, gateway, alias
        );
    }
    println!();
    Ok(())
}

fn create_profile_command(
    context: &HermesContext,
    args: ProfileCreateArgs,
) -> Result<(), Box<dyn Error>> {
    let canon = hermes_core::normalize_profile_name(&args.name)?;
    if args.clone && args.clone_all {
        return Err("--clone and --clone-all are mutually exclusive".into());
    }
    let clone_requested = args.clone || args.clone_all || args.clone_from.is_some();
    let path = if args.clone_all {
        let source = resolve_clone_source(context, args.clone_from.as_deref())?;
        clone_profile_tree(context, &canon, &source)?
    } else {
        let created = context.create_profile(&canon)?;
        let profile_context = context.clone().with_hermes_home_env(Some(created.clone()));
        profile_context.ensure_hermes_home()?;
        if clone_requested {
            let source = resolve_clone_source(context, args.clone_from.as_deref())?;
            clone_profile_config(&source, &created)?;
        } else {
            seed_bundled_skills(&created)?;
        }
        created
    };
    println!("created={canon}");
    println!("path={}", path.display());
    if args.clone || args.clone_all {
        let source = args
            .clone_from
            .as_deref()
            .map(str::to_string)
            .unwrap_or_else(|| context.current_profile_name());
        println!("cloned_from={source}");
        if args.clone_all {
            println!("clone_mode=all");
        } else {
            println!("clone_mode=config");
        }
    }
    if !args.no_alias {
        match create_wrapper_script(context.home_dir(), &canon, &canon) {
            Ok(path) => {
                println!("alias_created={}", path.display());
                if !is_wrapper_dir_in_path(context.home_dir()) {
                    println!(
                        "path_warning={} is not on PATH",
                        wrapper_dir(context.home_dir()).display()
                    );
                }
            }
            Err(err) => println!("alias_warning={err}"),
        }
    }
    Ok(())
}

fn delete_profile_command(
    context: &HermesContext,
    args: ProfileDeleteArgs,
) -> Result<(), Box<dyn Error>> {
    let canon = hermes_core::normalize_profile_name(&args.name)?;
    hermes_core::validate_profile_name(&canon)?;
    if canon == "default" {
        return Err("Cannot delete the default profile (~/.hermes). Use: hermes uninstall".into());
    }
    let profile_dir = context.profile_dir(&canon)?;
    if !profile_dir.is_dir() {
        return Err(format!("Profile '{canon}' does not exist.").into());
    }

    let row = describe_profile(context, &canon)?;
    println!("profile={}", row.name);
    println!("path={}", row.path.display());
    if let Some(model) = row.model.as_deref() {
        match row.provider.as_deref() {
            Some(provider) if !provider.is_empty() => println!("model={model} ({provider})"),
            _ => println!("model={model}"),
        }
    }
    println!("skills={}", row.skill_count);
    if row.gateway_running {
        println!("gateway=running");
    }
    if row.alias_path.is_some() {
        println!(
            "alias={}",
            wrapper_dir(context.home_dir()).join(&canon).display()
        );
    }

    if !args.yes {
        print!("Type '{canon}' to confirm: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim() != canon {
            println!("cancelled");
            return Ok(());
        }
    }

    cleanup_gateway_service(&profile_dir)?;
    if remove_wrapper_script(context.home_dir(), &canon)? {
        println!("alias_removed={canon}");
    }
    fs::remove_dir_all(&profile_dir)?;
    if context.active_profile() == canon {
        context.set_active_profile("default")?;
        println!("active_profile=default");
    }
    println!("deleted={canon}");
    Ok(())
}

fn show_profile_command(context: &HermesContext, raw_name: &str) -> Result<(), Box<dyn Error>> {
    let canon = hermes_core::normalize_profile_name(raw_name)?;
    let row = describe_profile(context, &canon)?;
    println!("profile={}", row.name);
    println!("path={}", row.path.display());
    if let Some(model) = row.model.as_deref() {
        match row.provider.as_deref() {
            Some(provider) if !provider.is_empty() => println!("model={model} ({provider})"),
            _ => println!("model={model}"),
        }
    }
    println!(
        "gateway={}",
        if row.gateway_running {
            "running"
        } else {
            "stopped"
        }
    );
    println!("skills={}", row.skill_count);
    println!("env={}", if row.has_env { "exists" } else { "missing" });
    println!(
        "soul={}",
        if row.path.join("SOUL.md").exists() {
            "exists"
        } else {
            "missing"
        }
    );
    if let Some(alias_path) = row.alias_path {
        println!("alias={}", alias_path.display());
    }
    Ok(())
}

fn alias_profile_command(
    context: &HermesContext,
    args: ProfileAliasArgs,
) -> Result<(), Box<dyn Error>> {
    let canon = hermes_core::normalize_profile_name(&args.name)?;
    if !context.profile_exists(&canon) {
        return Err(format!("Profile '{canon}' does not exist.").into());
    }

    let alias_name = match args.alias_name.as_deref() {
        Some(value) => hermes_core::normalize_profile_name(value)?,
        None => canon.clone(),
    };
    if args.remove {
        if remove_wrapper_script(context.home_dir(), &alias_name)? {
            println!("alias_removed={alias_name}");
        } else {
            println!("alias_missing={alias_name}");
        }
        return Ok(());
    }

    if let Some(message) = check_alias_collision(context.home_dir(), &alias_name)? {
        return Err(message.into());
    }
    let path = create_wrapper_script(context.home_dir(), &alias_name, &canon)?;
    println!("alias_created={}", path.display());
    if !is_wrapper_dir_in_path(context.home_dir()) {
        println!(
            "path_warning={} is not on PATH",
            wrapper_dir(context.home_dir()).display()
        );
    }
    Ok(())
}

fn rename_profile_command(
    context: &HermesContext,
    raw_old: &str,
    raw_new: &str,
) -> Result<(), Box<dyn Error>> {
    let old_name = hermes_core::normalize_profile_name(raw_old)?;
    let new_name = hermes_core::normalize_profile_name(raw_new)?;
    hermes_core::validate_profile_name(&old_name)?;
    hermes_core::validate_profile_name(&new_name)?;
    if old_name == "default" {
        return Err("Cannot rename the default profile.".into());
    }
    if new_name == "default" {
        return Err("Cannot rename to 'default' — it is reserved.".into());
    }
    let old_dir = context.profile_dir(&old_name)?;
    let new_dir = context.profile_dir(&new_name)?;
    if !old_dir.is_dir() {
        return Err(format!("Profile '{old_name}' does not exist.").into());
    }
    if new_dir.exists() {
        return Err(format!("Profile '{new_name}' already exists.").into());
    }

    cleanup_gateway_service(&old_dir)?;
    fs::rename(&old_dir, &new_dir)?;
    let _ = remove_wrapper_script(context.home_dir(), &old_name)?;
    if check_alias_collision(context.home_dir(), &new_name)?.is_none() {
        let _ = create_wrapper_script(context.home_dir(), &new_name, &new_name);
    }
    if context.active_profile() == old_name {
        context.set_active_profile(&new_name)?;
        println!("active_profile={new_name}");
    }
    println!("renamed={old_name}->{new_name}");
    println!("path={}", new_dir.display());
    Ok(())
}

fn export_profile_command(
    context: &HermesContext,
    args: ProfileExportArgs,
) -> Result<(), Box<dyn Error>> {
    let canon = hermes_core::normalize_profile_name(&args.name)?;
    hermes_core::validate_profile_name(&canon)?;
    let output = args.output.unwrap_or_else(|| format!("{canon}.tar.gz"));
    let output_path = export_profile(context, &canon, &output)?;
    println!("exported={canon}");
    println!("archive={}", output_path.display());
    Ok(())
}

fn import_profile_command(
    context: &HermesContext,
    args: ProfileImportArgs,
) -> Result<(), Box<dyn Error>> {
    let imported = import_profile_archive(context, &args.archive, args.import_name.as_deref())?;
    println!(
        "imported={}",
        imported
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or_default()
    );
    println!("path={}", imported.display());
    let imported_name = imported
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("imported profile path missing final name")?;
    if check_alias_collision(context.home_dir(), imported_name)?.is_none() {
        if let Ok(path) = create_wrapper_script(context.home_dir(), imported_name, imported_name) {
            println!("alias_created={}", path.display());
        }
    }
    Ok(())
}

fn list_profiles(context: &HermesContext) -> Result<Vec<ProfileRow>, Box<dyn Error>> {
    let mut rows = Vec::new();
    let default_home = context.default_hermes_root();
    if default_home.is_dir() {
        let (model, provider) = read_config_model(&default_home)?;
        rows.push(ProfileRow {
            name: "default".to_string(),
            path: default_home.clone(),
            is_default: true,
            gateway_running: gateway_running(&default_home),
            model,
            provider,
            has_env: default_home.join(".env").exists(),
            skill_count: count_skills(&default_home),
            alias_path: None,
        });
    }

    let profiles_root = context.profiles_root();
    if profiles_root.is_dir() {
        for entry in fs::read_dir(&profiles_root)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if !is_valid_profile_id(&name) {
                continue;
            }
            let (model, provider) = read_config_model(&path)?;
            let alias = wrapper_dir(context.home_dir()).join(&name);
            rows.push(ProfileRow {
                name,
                path: path.clone(),
                is_default: false,
                gateway_running: gateway_running(&path),
                model,
                provider,
                has_env: path.join(".env").exists(),
                skill_count: count_skills(&path),
                alias_path: alias.exists().then_some(alias),
            });
        }
    }
    rows.sort_by(|left, right| {
        left.is_default
            .cmp(&right.is_default)
            .reverse()
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(rows)
}

fn resolve_clone_source(
    context: &HermesContext,
    clone_from: Option<&str>,
) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(source) = clone_from {
        let canon = hermes_core::normalize_profile_name(source)?;
        let dir = context.profile_dir(&canon)?;
        if !dir.is_dir() {
            return Err(format!(
                "Source profile '{canon}' does not exist at {}",
                dir.display()
            )
            .into());
        }
        return Ok(dir);
    }
    let dir = context.hermes_home();
    if !dir.is_dir() {
        return Err(format!(
            "Source profile 'active' does not exist at {}",
            dir.display()
        )
        .into());
    }
    Ok(dir)
}

fn clone_profile_config(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    for name in ["config.yaml", ".env", "SOUL.md"] {
        let src = source.join(name);
        if src.exists() {
            fs::copy(&src, destination.join(name))?;
        }
    }
    let source_skills = source.join("skills");
    if source_skills.is_dir() {
        copy_dir_recursive(&source_skills, &destination.join("skills"), &|_, _| false)?;
    }
    for relative in ["memories/MEMORY.md", "memories/USER.md"] {
        let src = source.join(relative);
        if src.exists() {
            let dest = destination.join(relative);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(src, dest)?;
        }
    }
    Ok(())
}

fn clone_profile_tree(
    context: &HermesContext,
    profile_name: &str,
    source: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    let dest = context.profile_dir(profile_name)?;
    if dest.exists() {
        return Err(format!(
            "Profile '{profile_name}' already exists at {}",
            dest.display()
        )
        .into());
    }
    let staging = TempDir::new()?;
    let staged_dest = staging.path().join(profile_name);
    copy_dir_recursive(source, &staged_dest, &|path, depth| {
        depth == 1 && path.file_name().is_some_and(|name| name == "profiles")
    })?;
    for stale in ["gateway.pid", "gateway_state.json", "processes.json"] {
        let _ = fs::remove_file(staged_dest.join(stale));
    }
    fs::create_dir_all(dest.parent().ok_or("profile destination missing parent")?)?;
    fs::rename(&staged_dest, &dest)?;
    let profile_context = context.clone().with_hermes_home_env(Some(dest.clone()));
    profile_context.ensure_hermes_home()?;
    Ok(dest)
}

fn seed_bundled_skills(profile_dir: &Path) -> Result<(), Box<dyn Error>> {
    let bundled_dir = env::var_os("HERMES_BUNDLED_SKILLS")
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .unwrap_or_else(|| project_root().join("skills"));
    if !bundled_dir.is_dir() {
        return Ok(());
    }
    copy_dir_recursive(&bundled_dir, &profile_dir.join("skills"), &|path, _| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == ".git" || name == ".github" || name == ".hub")
    })
}

fn copy_dir_recursive(
    source: &Path,
    destination: &Path,
    skip: &dyn Fn(&Path, usize) -> bool,
) -> Result<(), Box<dyn Error>> {
    fn walk(
        source: &Path,
        destination: &Path,
        depth: usize,
        skip: &dyn Fn(&Path, usize) -> bool,
    ) -> Result<(), Box<dyn Error>> {
        if skip(source, depth) {
            return Ok(());
        }
        fs::create_dir_all(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let src = entry.path();
            let dest = destination.join(entry.file_name());
            if skip(&src, depth + 1) {
                continue;
            }
            if src.is_dir() {
                walk(&src, &dest, depth + 1, skip)?;
            } else {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(&src, &dest)?;
            }
        }
        Ok(())
    }

    walk(source, destination, 0, skip)
}

fn export_profile(
    context: &HermesContext,
    profile_name: &str,
    output: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let profile_dir = context.profile_dir(profile_name)?;
    if !profile_dir.is_dir() {
        return Err(format!("Profile '{profile_name}' does not exist.").into());
    }
    let output_path = PathBuf::from(output);
    if output_path.as_os_str().is_empty() {
        return Err("profile export output cannot be empty".into());
    }
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = fs::File::create(&output_path)?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut builder = Builder::new(encoder);
    let root_name = if profile_name == "default" {
        "default"
    } else {
        profile_name
    };
    append_directory_to_tar(
        &mut builder,
        &profile_dir,
        Path::new(root_name),
        profile_name == "default",
    )?;
    builder.finish()?;
    Ok(output_path)
}

fn append_directory_to_tar(
    builder: &mut Builder<GzEncoder<fs::File>>,
    source: &Path,
    archive_root: &Path,
    default_profile: bool,
) -> Result<(), Box<dyn Error>> {
    append_dir_entry(builder, archive_root, source)?;
    walk_export_entries(builder, source, archive_root, 0, default_profile)
}

fn walk_export_entries(
    builder: &mut Builder<GzEncoder<fs::File>>,
    source: &Path,
    archive_root: &Path,
    depth: usize,
    default_profile: bool,
) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if should_skip_export_entry(&name, depth, default_profile) {
            continue;
        }
        let archive_path = archive_root.join(entry.file_name());
        if path.is_dir() {
            append_dir_entry(builder, &archive_path, &path)?;
            walk_export_entries(builder, &path, &archive_path, depth + 1, default_profile)?;
        } else if path.is_file() {
            builder.append_path_with_name(&path, &archive_path)?;
        }
    }
    Ok(())
}

fn should_skip_export_entry(name: &str, depth: usize, default_profile: bool) -> bool {
    if matches!(name, "__pycache__" | "package.json" | "package-lock.json") {
        return true;
    }
    if name.ends_with(".sock") || name.ends_with(".tmp") {
        return true;
    }
    if default_profile && depth == 0 {
        return matches!(
            name,
            "hermes-agent"
                | ".worktrees"
                | "profiles"
                | "bin"
                | "node_modules"
                | "state.db"
                | "state.db-shm"
                | "state.db-wal"
                | "hermes_state.db"
                | "response_store.db"
                | "response_store.db-shm"
                | "response_store.db-wal"
                | "gateway.pid"
                | "gateway_state.json"
                | "processes.json"
                | "auth.json"
                | ".env"
                | "auth.lock"
                | "active_profile"
                | ".update_check"
                | "errors.log"
                | ".hermes_history"
                | "image_cache"
                | "audio_cache"
                | "document_cache"
                | "browser_screenshots"
                | "checkpoints"
                | "sandboxes"
                | "logs"
        );
    }
    if !default_profile && depth == 0 && matches!(name, "auth.json" | ".env") {
        return true;
    }
    false
}

fn append_dir_entry(
    builder: &mut Builder<GzEncoder<fs::File>>,
    archive_path: &Path,
    source: &Path,
) -> Result<(), Box<dyn Error>> {
    let metadata = fs::metadata(source)?;
    let mut header = Header::new_gnu();
    header.set_path(archive_path)?;
    header.set_entry_type(EntryType::Directory);
    header.set_size(0);
    #[cfg(unix)]
    header.set_mode(metadata.permissions().mode());
    #[cfg(not(unix))]
    header.set_mode(0o755);
    header.set_cksum();
    builder.append(&header, io::empty())?;
    Ok(())
}

fn import_profile_archive(
    context: &HermesContext,
    archive_path: &str,
    name: Option<&str>,
) -> Result<PathBuf, Box<dyn Error>> {
    let archive_path = PathBuf::from(archive_path);
    if !archive_path.is_file() {
        return Err(format!("Archive not found: {}", archive_path.display()).into());
    }

    let bytes = fs::read(&archive_path)?;
    let top_dirs = inspect_archive_roots(&bytes)?;
    let archive_root = if top_dirs.len() == 1 {
        top_dirs.into_iter().next().unwrap()
    } else {
        return Err("Profile archive must contain exactly one top-level directory.".into());
    };

    let inferred = if let Some(name) = name {
        hermes_core::normalize_profile_name(name)?
    } else {
        archive_root.clone()
    };
    hermes_core::validate_profile_name(&inferred)?;
    if inferred == "default" {
        return Err("Cannot import as 'default' — specify a different name with --name.".into());
    }

    let profile_dir = context.profile_dir(&inferred)?;
    if profile_dir.exists() {
        return Err(format!(
            "Profile '{inferred}' already exists at {}",
            profile_dir.display()
        )
        .into());
    }
    fs::create_dir_all(context.profiles_root())?;

    let staging = TempDir::new()?;
    extract_profile_archive(&bytes, staging.path())?;
    let extracted = staging.path().join(&archive_root);
    if !extracted.is_dir() {
        return Err(format!("Profile archive root is missing or invalid: {archive_root}").into());
    }

    let final_source = if archive_root != inferred {
        let renamed = staging.path().join(&inferred);
        fs::rename(&extracted, &renamed)?;
        renamed
    } else {
        extracted
    };
    fs::rename(&final_source, &profile_dir)?;
    Ok(profile_dir)
}

fn inspect_archive_roots(
    bytes: &[u8],
) -> Result<std::collections::BTreeSet<String>, Box<dyn Error>> {
    let cursor = std::io::Cursor::new(bytes);
    let decoder = GzDecoder::new(cursor);
    let mut archive = Archive::new(decoder);
    let mut roots = std::collections::BTreeSet::new();
    for entry in archive.entries()? {
        let entry = entry?;
        let parts = normalize_archive_parts(&entry.path()?)?;
        if let Some(root) = parts.first() {
            roots.insert(root.clone());
        }
    }
    Ok(roots)
}

fn extract_profile_archive(bytes: &[u8], destination: &Path) -> Result<(), Box<dyn Error>> {
    let cursor = std::io::Cursor::new(bytes);
    let decoder = GzDecoder::new(cursor);
    let mut archive = Archive::new(decoder);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let parts = normalize_archive_parts(&entry.path()?)?;
        let target = parts
            .iter()
            .fold(destination.to_path_buf(), |acc, part| acc.join(part));
        if entry.header().entry_type().is_dir() {
            fs::create_dir_all(&target)?;
            continue;
        }
        if !entry.header().entry_type().is_file() {
            return Err(format!(
                "Unsupported archive member type: {}",
                entry.path()?.display()
            )
            .into());
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::File::create(&target)?;
        io::copy(&mut entry, &mut file)?;
        #[cfg(unix)]
        if let Ok(mode) = entry.header().mode() {
            let mut perms = fs::metadata(&target)?.permissions();
            perms.set_mode(mode);
            let _ = fs::set_permissions(&target, perms);
        }
    }
    Ok(())
}

fn normalize_archive_parts(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::Normal(value) => {
                let value = value.to_string_lossy();
                if value.is_empty() || value == "." {
                    continue;
                }
                if value == ".." {
                    return Err(format!("Unsafe archive member path: {}", path.display()).into());
                }
                parts.push(value.to_string());
            }
            std::path::Component::CurDir => {}
            _ => return Err(format!("Unsafe archive member path: {}", path.display()).into()),
        }
    }
    if parts.is_empty() {
        return Err(format!("Unsafe archive member path: {}", path.display()).into());
    }
    Ok(parts)
}

fn describe_profile(context: &HermesContext, name: &str) -> Result<ProfileRow, Box<dyn Error>> {
    let canon = hermes_core::normalize_profile_name(name)?;
    let path = context.profile_dir(&canon)?;
    if !path.is_dir() && canon != "default" {
        return Err(format!("Profile '{canon}' does not exist.").into());
    }
    let (model, provider) = read_config_model(&path)?;
    let alias = wrapper_dir(context.home_dir()).join(&canon);
    Ok(ProfileRow {
        name: canon.clone(),
        path: path.clone(),
        is_default: canon == "default",
        gateway_running: gateway_running(&path),
        model,
        provider,
        has_env: path.join(".env").exists(),
        skill_count: count_skills(&path),
        alias_path: (!canon.eq("default") && alias.exists()).then_some(alias),
    })
}

fn read_config_model(
    profile_dir: &Path,
) -> Result<(Option<String>, Option<String>), Box<dyn Error>> {
    let config_path = profile_dir.join("config.yaml");
    if !config_path.exists() {
        return Ok((None, None));
    }
    let text = fs::read_to_string(config_path)?;
    if text.trim().is_empty() {
        return Ok((None, None));
    }
    let parsed: YamlValue = serde_yaml::from_str(&text)?;
    let Some(root) = parsed.as_mapping() else {
        return Ok((None, None));
    };
    let Some(model) = root.get(YamlValue::String("model".to_string())) else {
        return Ok((None, None));
    };

    if let Some(value) = model
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Ok((Some(value.to_string()), None));
    }

    let Some(mapping) = model.as_mapping() else {
        return Ok((None, None));
    };
    let default = mapping
        .get(YamlValue::String("default".to_string()))
        .or_else(|| mapping.get(YamlValue::String("model".to_string())))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let provider = mapping
        .get(YamlValue::String("provider".to_string()))
        .and_then(YamlValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    Ok((default, provider))
}

fn gateway_running(profile_dir: &Path) -> bool {
    let pid_file = profile_dir.join("gateway.pid");
    let Ok(raw) = fs::read_to_string(pid_file) else {
        return false;
    };
    let pid = if raw.trim_start().starts_with('{') {
        serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .and_then(|value| value.get("pid").and_then(serde_json::Value::as_i64))
    } else {
        raw.trim().parse::<i64>().ok()
    };
    let Some(pid) = pid else {
        return false;
    };
    process_running(pid)
}

fn process_running(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        false
    }
}

fn count_skills(profile_dir: &Path) -> usize {
    let skills_dir = profile_dir.join("skills");
    if !skills_dir.is_dir() {
        return 0;
    }
    let mut count = 0;
    walk_skill_markdowns(&skills_dir, &mut count);
    count
}

fn walk_skill_markdowns(root: &Path, count: &mut usize) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == ".hub" || name == ".git" {
                continue;
            }
            walk_skill_markdowns(&path, count);
        } else if name == "SKILL.md" {
            *count += 1;
        }
    }
}

fn cleanup_gateway_service(profile_dir: &Path) -> Result<(), Box<dyn Error>> {
    let profile_context =
        HermesContext::new("/tmp").with_hermes_home_env(Some(profile_dir.to_path_buf()));
    for command in [
        GatewayCommand::Stop(GatewayServiceArgs {
            system: false,
            all: false,
        }),
        GatewayCommand::Stop(GatewayServiceArgs {
            system: true,
            all: false,
        }),
        GatewayCommand::Uninstall(GatewaySystemArgs { system: false }),
        GatewayCommand::Uninstall(GatewaySystemArgs { system: true }),
    ] {
        let _ = print_gateway(
            &profile_context,
            GatewayArgs {
                accept_hooks: false,
                command: Some(command),
            },
        );
    }
    Ok(())
}

fn check_alias_collision(
    home_dir: &Path,
    raw_name: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let canon = hermes_core::normalize_profile_name(raw_name)?;
    if RESERVED_ALIAS_NAMES.contains(&canon.as_str()) {
        return Ok(Some(format!("'{canon}' is a reserved name")));
    }
    if HERMES_SUBCOMMANDS.contains(&canon.as_str()) {
        return Ok(Some(format!(
            "'{canon}' conflicts with a hermes subcommand"
        )));
    }

    if let Some(existing) = which_on_path(&canon) {
        let wrapper = wrapper_dir(home_dir).join(&canon);
        if existing == wrapper {
            if let Ok(content) = fs::read_to_string(&wrapper) {
                if content.contains("hermes -p") {
                    return Ok(None);
                }
            }
        }
        return Ok(Some(format!(
            "'{canon}' conflicts with an existing command ({})",
            existing.display()
        )));
    }
    Ok(None)
}

fn create_wrapper_script(
    home_dir: &Path,
    raw_alias: &str,
    raw_profile: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let alias = hermes_core::normalize_profile_name(raw_alias)?;
    let profile = hermes_core::normalize_profile_name(raw_profile)?;
    let wrapper_dir = wrapper_dir(home_dir);
    fs::create_dir_all(&wrapper_dir)?;
    let wrapper_path = wrapper_dir.join(alias);
    fs::write(
        &wrapper_path,
        format!("#!/bin/sh\nexec hermes -p {profile} \"$@\"\n"),
    )?;
    #[cfg(unix)]
    {
        let mut perms = fs::metadata(&wrapper_path)?.permissions();
        perms.set_mode(perms.mode() | 0o111);
        fs::set_permissions(&wrapper_path, perms)?;
    }
    Ok(wrapper_path)
}

fn remove_wrapper_script(home_dir: &Path, raw_alias: &str) -> Result<bool, Box<dyn Error>> {
    let alias = hermes_core::normalize_profile_name(raw_alias)?;
    let wrapper_path = wrapper_dir(home_dir).join(alias);
    if !wrapper_path.exists() {
        return Ok(false);
    }
    let Ok(content) = fs::read_to_string(&wrapper_path) else {
        return Ok(false);
    };
    if !content.contains("hermes -p") {
        return Ok(false);
    }
    fs::remove_file(&wrapper_path)?;
    Ok(true)
}

fn is_wrapper_dir_in_path(home_dir: &Path) -> bool {
    let target = wrapper_dir(home_dir);
    env::split_paths(&env::var_os("PATH").unwrap_or_default()).any(|path| path == target)
}

fn wrapper_dir(home_dir: &Path) -> PathBuf {
    home_dir.join(".local").join("bin")
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH")?;
    for dir in env::split_paths(&paths) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let exe = dir.join(format!("{name}.exe"));
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    None
}

fn is_valid_profile_id(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }
    let first = bytes[0];
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }
    bytes.iter().copied().all(|byte| {
        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
    })
}

fn truncate(value: &str, max_len: usize) -> String {
    let mut chars = value.chars();
    let truncated = chars.by_ref().take(max_len).collect::<String>();
    if chars.next().is_some() {
        truncated
    } else {
        value.to_string()
    }
}

#[cfg(test)]
fn test_env_lock() -> &'static Mutex<()> {
    crate::cli_test_env_lock()
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_context() -> (TempDir, HermesContext) {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        let ctx = HermesContext::new(&home);
        (temp, ctx)
    }

    #[test]
    fn list_profiles_includes_default_and_named_rows() {
        let (_temp, ctx) = test_context();
        fs::create_dir_all(ctx.default_hermes_root()).unwrap();
        fs::create_dir_all(ctx.profiles_root().join("coder").join("skills")).unwrap();
        fs::write(
            ctx.profiles_root().join("coder").join("config.yaml"),
            "model:\n  default: gpt-4.1-mini\n  provider: openai\n",
        )
        .unwrap();
        fs::write(
            ctx.profiles_root()
                .join("coder")
                .join("skills")
                .join("SKILL.md"),
            "---\nname: demo\n---\nbody\n",
        )
        .unwrap();
        create_wrapper_script(ctx.home_dir(), "coder", "coder").unwrap();

        let rows = list_profiles(&ctx).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "default");
        assert_eq!(rows[1].name, "coder");
        assert_eq!(rows[1].model.as_deref(), Some("gpt-4.1-mini"));
        assert_eq!(rows[1].provider.as_deref(), Some("openai"));
        assert_eq!(rows[1].skill_count, 1);
        assert!(rows[1].alias_path.is_some());
    }

    #[test]
    fn alias_create_and_remove_work() {
        let (_temp, ctx) = test_context();
        let path = create_wrapper_script(ctx.home_dir(), "coder", "coder").unwrap();
        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("hermes -p coder"));
        assert!(remove_wrapper_script(ctx.home_dir(), "coder").unwrap());
        assert!(!path.exists());
    }

    #[test]
    fn delete_profile_removes_directory_and_resets_active_profile() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let profile_dir = ctx.create_profile("coder").unwrap();
        create_wrapper_script(ctx.home_dir(), "coder", "coder").unwrap();
        ctx.set_active_profile("coder").unwrap();
        set_env_var("HERMES_PROFILE_PYTHON", "/bin/true");

        delete_profile_command(
            &ctx,
            ProfileDeleteArgs {
                name: "coder".to_string(),
                yes: true,
            },
        )
        .unwrap();

        assert!(!profile_dir.exists());
        assert_eq!(ctx.active_profile(), "default");
        assert!(!wrapper_dir(ctx.home_dir()).join("coder").exists());
        remove_env_var("HERMES_PROFILE_PYTHON");
    }

    #[test]
    fn rename_profile_moves_directory_and_alias() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let old_dir = ctx.create_profile("coder").unwrap();
        create_wrapper_script(ctx.home_dir(), "coder", "coder").unwrap();
        ctx.set_active_profile("coder").unwrap();
        set_env_var("HERMES_PROFILE_PYTHON", "/bin/true");

        rename_profile_command(&ctx, "coder", "builder").unwrap();

        assert!(!old_dir.exists());
        assert!(ctx.profile_dir("builder").unwrap().exists());
        assert!(!wrapper_dir(ctx.home_dir()).join("coder").exists());
        assert!(wrapper_dir(ctx.home_dir()).join("builder").exists());
        assert_eq!(ctx.active_profile(), "builder");
        remove_env_var("HERMES_PROFILE_PYTHON");
    }

    #[test]
    fn create_profile_command_creates_alias_by_default() {
        let _guard = test_env_lock().lock().unwrap();
        let (_temp, ctx) = test_context();
        let bundled = ctx.home_dir().join("bundled");
        fs::create_dir_all(bundled.join("demo")).unwrap();
        fs::write(
            bundled.join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nbody\n",
        )
        .unwrap();
        set_env_var("HERMES_BUNDLED_SKILLS", &bundled);
        create_profile_command(
            &ctx,
            ProfileCreateArgs {
                name: "coder".to_string(),
                clone: false,
                clone_all: false,
                clone_from: None,
                no_alias: false,
            },
        )
        .unwrap();
        assert!(ctx.profile_dir("coder").unwrap().exists());
        assert!(wrapper_dir(ctx.home_dir()).join("coder").exists());
        assert!(
            ctx.profile_dir("coder")
                .unwrap()
                .join("skills")
                .join("demo")
                .join("SKILL.md")
                .exists()
        );
        remove_env_var("HERMES_BUNDLED_SKILLS");
    }

    #[test]
    fn create_profile_clone_copies_config_skills_and_memory() {
        let (_temp, ctx) = test_context();
        let source = ctx.create_profile("source").unwrap();
        fs::write(
            source.join("config.yaml"),
            "model:\n  default: test-model\n",
        )
        .unwrap();
        fs::write(source.join(".env"), "OPENAI_API_KEY=test-key\n").unwrap();
        fs::write(source.join("SOUL.md"), "custom soul").unwrap();
        fs::create_dir_all(source.join("skills").join("team").join("demo")).unwrap();
        fs::write(
            source
                .join("skills")
                .join("team")
                .join("demo")
                .join("SKILL.md"),
            "---\nname: demo\n---\nbody\n",
        )
        .unwrap();
        fs::write(source.join("memories").join("MEMORY.md"), "memory").unwrap();
        fs::write(source.join("memories").join("USER.md"), "user").unwrap();

        create_profile_command(
            &ctx,
            ProfileCreateArgs {
                name: "clone".to_string(),
                clone: true,
                clone_all: false,
                clone_from: Some("source".to_string()),
                no_alias: true,
            },
        )
        .unwrap();

        let dest = ctx.profile_dir("clone").unwrap();
        assert_eq!(
            fs::read_to_string(dest.join("config.yaml")).unwrap(),
            "model:\n  default: test-model\n"
        );
        assert_eq!(
            fs::read_to_string(dest.join(".env")).unwrap(),
            "OPENAI_API_KEY=test-key\n"
        );
        assert_eq!(
            fs::read_to_string(dest.join("SOUL.md")).unwrap(),
            "custom soul"
        );
        assert!(
            dest.join("skills")
                .join("team")
                .join("demo")
                .join("SKILL.md")
                .exists()
        );
        assert_eq!(
            fs::read_to_string(dest.join("memories").join("MEMORY.md")).unwrap(),
            "memory"
        );
        assert_eq!(
            fs::read_to_string(dest.join("memories").join("USER.md")).unwrap(),
            "user"
        );
    }

    #[test]
    fn create_profile_clone_all_skips_nested_profiles_and_runtime_files() {
        let (_temp, ctx) = test_context();
        let default_root = ctx.default_hermes_root();
        fs::create_dir_all(default_root.join("workspace")).unwrap();
        fs::write(default_root.join("workspace").join("note.txt"), "hello").unwrap();
        fs::write(default_root.join("gateway.pid"), "123").unwrap();
        fs::write(default_root.join("processes.json"), "{}").unwrap();
        fs::create_dir_all(default_root.join("profiles").join("other")).unwrap();
        fs::write(
            default_root
                .join("profiles")
                .join("other")
                .join("marker.txt"),
            "ignore me",
        )
        .unwrap();

        create_profile_command(
            &ctx,
            ProfileCreateArgs {
                name: "mirror".to_string(),
                clone: false,
                clone_all: true,
                clone_from: None,
                no_alias: true,
            },
        )
        .unwrap();

        let dest = ctx.profile_dir("mirror").unwrap();
        assert!(dest.join("workspace").join("note.txt").exists());
        assert!(!dest.join("gateway.pid").exists());
        assert!(!dest.join("processes.json").exists());
        assert!(!dest.join("profiles").exists());
    }

    #[test]
    fn export_and_import_profile_archive_round_trips_without_credentials() {
        let (_temp, ctx) = test_context();
        let source = ctx.create_profile("coder").unwrap();
        fs::write(source.join("config.yaml"), "model:\n  default: imported\n").unwrap();
        fs::write(source.join(".env"), "SECRET=1\n").unwrap();
        fs::write(source.join("auth.json"), "{\"token\":1}\n").unwrap();
        fs::create_dir_all(source.join("skills").join("demo")).unwrap();
        fs::write(
            source.join("skills").join("demo").join("SKILL.md"),
            "---\nname: demo\n---\nbody\n",
        )
        .unwrap();

        let archive = ctx.home_dir().join("coder.tar.gz");
        export_profile(&ctx, "coder", archive.to_str().unwrap()).unwrap();
        let imported =
            import_profile_archive(&ctx, archive.to_str().unwrap(), Some("builder")).unwrap();

        assert_eq!(imported, ctx.profile_dir("builder").unwrap());
        assert_eq!(
            fs::read_to_string(imported.join("config.yaml")).unwrap(),
            "model:\n  default: imported\n"
        );
        assert!(
            imported
                .join("skills")
                .join("demo")
                .join("SKILL.md")
                .exists()
        );
        assert!(!imported.join(".env").exists());
        assert!(!imported.join("auth.json").exists());
    }

    #[test]
    fn profile_current_lines_match_active_profile_and_home() {
        let (_temp, ctx) = test_context();
        let profile_home = ctx.create_profile("coder").unwrap();
        ctx.set_active_profile("coder").unwrap();
        let profile_ctx = ctx.with_hermes_home_env(Some(profile_home));

        let lines = profile_current_lines(&profile_ctx);
        assert_eq!(lines[0], "active_profile=coder");
        assert_eq!(lines[1], "current_profile=coder");
        assert_eq!(
            lines[2],
            format!("display_home={}", profile_ctx.display_hermes_home())
        );
    }

    #[test]
    fn check_alias_collision_rejects_reserved_names() {
        let (_temp, ctx) = test_context();
        let message = check_alias_collision(ctx.home_dir(), "hermes")
            .unwrap()
            .unwrap();
        assert!(message.contains("reserved"));
        let message = check_alias_collision(ctx.home_dir(), "platforms")
            .unwrap()
            .unwrap();
        assert!(message.contains("conflicts with a hermes subcommand"));
    }
}
