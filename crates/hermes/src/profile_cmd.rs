use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

use clap::{Args, Subcommand};
use hermes_core::HermesContext;
use serde_yaml::Value as YamlValue;

use crate::python_bridge::launch_python_main_command;

const RESERVED_ALIAS_NAMES: &[&str] = &["hermes", "default", "test", "tmp", "root", "sudo"];
const HERMES_SUBCOMMANDS: &[&str] = &[
    "chat",
    "model",
    "gateway",
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
}

#[derive(Args, Debug, Clone)]
pub struct ProfileCreateArgs {
    pub name: String,
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
        None => print_profile_summary(context),
        Some(ProfileCommand::Current) => {
            println!("active_profile={}", context.active_profile());
            println!("current_profile={}", context.current_profile_name());
            println!("display_home={}", context.display_hermes_home());
            Ok(())
        }
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
    }
}

fn print_profile_summary(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let current = context.current_profile_name();
    println!("active_profile={current}");
    println!("display_home={}", context.display_hermes_home());

    let rows = list_profiles(context)?;
    if let Some(row) = rows.iter().find(|row| row.name == current) {
        if let Some(model) = row.model.as_deref() {
            match row.provider.as_deref() {
                Some(provider) if !provider.is_empty() => {
                    println!("model={model} ({provider})");
                }
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
        if row.alias_path.is_some() && !row.is_default {
            println!("alias={} -> hermes -p {}", row.name, row.name);
        }
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
    let path = context.create_profile(&canon)?;
    println!("created={canon}");
    println!("path={}", path.display());
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
    let env_home = vec![("HERMES_HOME".to_string(), profile_dir.display().to_string())];
    for argv in [
        vec!["stop".to_string()],
        vec!["stop".to_string(), "--system".to_string()],
        vec!["uninstall".to_string()],
        vec!["uninstall".to_string(), "--system".to_string()],
    ] {
        let _ =
            launch_python_main_command("gateway", &argv, Some("HERMES_PROFILE_PYTHON"), &env_home);
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
        let (_temp, ctx) = test_context();
        create_profile_command(
            &ctx,
            ProfileCreateArgs {
                name: "coder".to_string(),
                no_alias: false,
            },
        )
        .unwrap();
        assert!(ctx.profile_dir("coder").unwrap().exists());
        assert!(wrapper_dir(ctx.home_dir()).join("coder").exists());
    }

    #[test]
    fn check_alias_collision_rejects_reserved_names() {
        let (_temp, ctx) = test_context();
        let message = check_alias_collision(ctx.home_dir(), "hermes")
            .unwrap()
            .unwrap();
        assert!(message.contains("reserved"));
    }
}
