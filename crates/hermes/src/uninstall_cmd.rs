use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
#[cfg(not(windows))]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

use clap::Args;
use hermes_core::HermesContext;

use crate::gateway_cmd::{
    GatewayArgs, GatewayCommand, GatewayServiceArgs, GatewaySystemArgs, print_gateway,
};
use crate::python_bridge::project_root as repo_project_root;

#[derive(Args, Debug, Clone)]
pub struct UninstallArgs {
    #[arg(long, default_value_t = false)]
    pub full: bool,
    #[arg(short = 'y', long, default_value_t = false)]
    pub yes: bool,
}

#[derive(Debug, Clone)]
struct NamedProfile {
    name: String,
    path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UninstallMode {
    KeepData,
    Full,
    Cancel,
}

pub fn print_uninstall(context: &HermesContext, args: UninstallArgs) -> Result<(), Box<dyn Error>> {
    if !args.yes && !stdin_is_terminal() {
        return Err("uninstall requires an interactive tty unless --yes is provided".into());
    }

    let project_root = uninstall_project_root();
    let hermes_home = context.hermes_home();
    let current_profile = context.current_profile_name();
    let default_root = context.default_hermes_root();
    let is_default_profile = hermes_home == default_root;
    let named_profiles = if is_default_profile {
        discover_named_profiles(context)?
    } else {
        Vec::new()
    };

    print_summary(&project_root, &hermes_home, &named_profiles);

    let mode = determine_mode(args.full, args.yes)?;
    if mode == UninstallMode::Cancel {
        println!("uninstall cancelled");
        return Ok(());
    }

    let full_uninstall = mode == UninstallMode::Full;
    let remove_named_profiles = if full_uninstall && !named_profiles.is_empty() {
        if args.yes {
            true
        } else {
            prompt_yes_no(
                &format!(
                    "Also stop and remove {} named profile(s)? [y/N]: ",
                    named_profiles.len()
                ),
                false,
            )?
        }
    } else {
        false
    };

    if !args.yes {
        print_final_warning(full_uninstall, remove_named_profiles, named_profiles.len());
        print!("Type 'yes' to confirm: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim() != "yes" {
            println!("uninstall cancelled");
            return Ok(());
        }
    }

    println!("uninstalling...");

    let removed_gateway = cleanup_gateway_for_profile(context, None);
    if removed_gateway {
        println!("gateway_cleanup=done");
    } else {
        println!("gateway_cleanup=skipped");
    }

    let updated_shells = remove_path_from_shell_configs(context.home_dir());
    if updated_shells.is_empty() {
        println!("shell_configs=unchanged");
    } else {
        println!("shell_configs_updated={}", updated_shells.len());
    }

    let removed_wrappers = remove_main_wrappers(context.home_dir());
    if removed_wrappers.is_empty() {
        println!("wrappers=unchanged");
    } else {
        println!("wrappers_removed={}", removed_wrappers.len());
    }

    if full_uninstall && current_profile != "default" && current_profile != "custom" {
        if remove_profile_alias(context.home_dir(), &current_profile) {
            println!("profile_alias_removed={current_profile}");
        }
    }

    remove_path_if_exists(&project_root)?;
    println!("code_removed={}", project_root.display());

    if full_uninstall {
        if remove_named_profiles {
            for profile in &named_profiles {
                let cleaned = cleanup_gateway_for_profile(context, Some(&profile.path));
                let alias_removed = remove_profile_alias(context.home_dir(), &profile.name);
                println!(
                    "profile_cleanup name={} gateway={} alias_removed={}",
                    profile.name,
                    if cleaned { "done" } else { "skipped" },
                    alias_removed
                );
            }
        }
        remove_path_if_exists(&hermes_home)?;
        println!("data_removed={}", hermes_home.display());
    } else {
        println!("data_preserved={}", hermes_home.display());
    }

    println!("uninstall_complete");
    Ok(())
}

fn print_summary(project_root: &Path, hermes_home: &Path, named_profiles: &[NamedProfile]) {
    println!("current_installation");
    println!("  code={}", project_root.display());
    println!("  config={}", hermes_home.join("config.yaml").display());
    println!("  secrets={}", hermes_home.join(".env").display());
    println!("  data={}", hermes_home.display());
    if !named_profiles.is_empty() {
        println!("named_profiles={}", named_profiles.len());
        for profile in named_profiles {
            println!("  profile={} path={}", profile.name, profile.path.display());
        }
    }
}

fn determine_mode(flag_full: bool, skip_prompts: bool) -> Result<UninstallMode, Box<dyn Error>> {
    if flag_full {
        return Ok(UninstallMode::Full);
    }
    if skip_prompts {
        return Ok(UninstallMode::KeepData);
    }

    loop {
        println!("options:");
        println!("  1) keep data");
        println!("  2) full uninstall");
        println!("  3) cancel");
        print!("Select option [1/2/3]: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        match input.trim() {
            "1" => return Ok(UninstallMode::KeepData),
            "2" => return Ok(UninstallMode::Full),
            "3" | "c" | "C" | "cancel" | "q" | "Q" => return Ok(UninstallMode::Cancel),
            _ => println!("invalid selection"),
        }
    }
}

fn print_final_warning(full_uninstall: bool, remove_named_profiles: bool, named_count: usize) {
    if full_uninstall {
        println!(
            "warning: full uninstall will permanently delete configs, sessions, logs, cron, and secrets"
        );
        if remove_named_profiles && named_count > 0 {
            println!("warning: also removing {named_count} named profile(s)");
        }
    } else {
        println!("warning: uninstall will remove the code and command wrappers but preserve data");
    }
}

fn discover_named_profiles(context: &HermesContext) -> Result<Vec<NamedProfile>, Box<dyn Error>> {
    let root = context.profiles_root();
    if !root.is_dir() {
        return Ok(Vec::new());
    }

    let mut profiles = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !is_valid_profile_name(&name) {
            continue;
        }
        profiles.push(NamedProfile { name, path });
    }
    profiles.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(profiles)
}

fn remove_path_from_shell_configs(home_dir: &Path) -> Vec<PathBuf> {
    let mut removed = Vec::new();
    for config_path in shell_configs(home_dir) {
        let Ok(content) = fs::read_to_string(&config_path) else {
            continue;
        };
        let original = content.clone();
        let mut new_lines = Vec::new();
        let mut skip_next = false;

        for line in content.lines() {
            if line.contains("# Hermes Agent") || line.contains("# hermes-agent") {
                skip_next = true;
                continue;
            }
            if skip_next && line.to_ascii_lowercase().contains("hermes") && line.contains("PATH") {
                skip_next = false;
                continue;
            }
            skip_next = false;

            let lower = line.to_ascii_lowercase();
            if lower.contains("hermes") && (line.contains("PATH=") || lower.contains("path=")) {
                continue;
            }
            new_lines.push(line);
        }

        let mut rewritten = new_lines.join("\n");
        while rewritten.contains("\n\n\n") {
            rewritten = rewritten.replace("\n\n\n", "\n\n");
        }
        if content.ends_with('\n') && !rewritten.ends_with('\n') {
            rewritten.push('\n');
        }

        if rewritten != original && fs::write(&config_path, rewritten).is_ok() {
            removed.push(config_path);
        }
    }
    removed
}

fn shell_configs(home_dir: &Path) -> Vec<PathBuf> {
    let candidates = [
        ".bashrc",
        ".bash_profile",
        ".profile",
        ".zshrc",
        ".zprofile",
    ];
    candidates
        .into_iter()
        .map(|name| home_dir.join(name))
        .filter(|path| path.exists())
        .collect()
}

fn remove_main_wrappers(home_dir: &Path) -> Vec<PathBuf> {
    let mut removed = Vec::new();
    for path in [
        wrapper_dir(home_dir).join("hermes"),
        PathBuf::from("/usr/local/bin/hermes"),
    ] {
        if remove_wrapper_if(&path, |content| {
            content.contains("hermes_cli") || content.contains("hermes-agent")
        }) {
            removed.push(path);
        }
    }
    removed
}

fn remove_profile_alias(home_dir: &Path, profile_name: &str) -> bool {
    let path = wrapper_dir(home_dir).join(profile_name);
    remove_wrapper_if(&path, |content| {
        content.contains(&format!(" -p {profile_name} "))
            || content.contains(&format!(" --profile {profile_name} "))
            || content.contains(&format!(" -p {profile_name}\""))
            || content.contains(&format!(" --profile {profile_name}\""))
    })
}

fn remove_wrapper_if(path: &Path, predicate: impl Fn(&str) -> bool) -> bool {
    if !path.exists() {
        return false;
    }
    let Ok(content) = fs::read_to_string(path) else {
        return false;
    };
    if !predicate(&content) {
        return false;
    }
    fs::remove_file(path).is_ok()
}

fn wrapper_dir(home_dir: &Path) -> PathBuf {
    home_dir.join(".local").join("bin")
}

fn cleanup_gateway_for_profile(context: &HermesContext, profile_home: Option<&Path>) -> bool {
    let profile_context = match profile_home {
        Some(path) => {
            HermesContext::new(context.home_dir()).with_hermes_home_env(Some(path.into()))
        }
        None => context.clone(),
    };
    let mut any_success = false;
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
        let ok = print_gateway(
            &profile_context,
            GatewayArgs {
                accept_hooks: false,
                command: Some(command),
            },
        )
        .is_ok();
        any_success |= ok;
    }
    any_success
}

fn remove_path_if_exists(path: &Path) -> Result<(), Box<dyn Error>> {
    if !path.exists() {
        return Ok(());
    }
    fs::remove_dir_all(path)?;
    Ok(())
}

fn uninstall_project_root() -> PathBuf {
    if let Some(override_path) = env::var_os("HERMES_UNINSTALL_PROJECT_ROOT")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
    {
        return override_path;
    }
    repo_project_root()
}

fn prompt_yes_no(prompt: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let trimmed = input.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return Ok(default);
    }
    Ok(matches!(trimmed.as_str(), "y" | "yes"))
}

fn is_valid_profile_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return false;
    }
    if name.len() > 64 {
        return false;
    }
    chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-')
}

fn stdin_is_terminal() -> bool {
    #[cfg(windows)]
    {
        true
    }
    #[cfg(not(windows))]
    {
        unsafe { libc::isatty(io::stdin().as_raw_fd()) == 1 }
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

    fn write_wrapper(path: &Path, body: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }

    #[test]
    fn remove_path_cleanup_strips_hermes_entries() {
        let temp = TempDir::new().unwrap();
        let bashrc = temp.path().join(".bashrc");
        fs::write(
            &bashrc,
            "export PATH=\"/usr/bin:$PATH\"\n# Hermes Agent\nexport PATH=\"$HOME/.hermes/bin:$PATH\"\nexport PATH=\"$HOME/.foo:$PATH\"\n",
        )
        .unwrap();

        let updated = remove_path_from_shell_configs(temp.path());
        assert_eq!(updated, vec![bashrc.clone()]);
        let content = fs::read_to_string(bashrc).unwrap();
        assert!(!content.contains("Hermes Agent"));
        assert!(!content.contains(".hermes/bin"));
        assert!(content.contains(".foo"));
    }

    #[test]
    fn discover_named_profiles_finds_aliases() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        let hermes_home = home.join(".hermes");
        fs::create_dir_all(hermes_home.join("profiles").join("coder")).unwrap();
        write_wrapper(
            &home.join(".local/bin/coder"),
            "#!/bin/sh\nexec hermes -p coder \"$@\"\n",
        );

        let context = HermesContext::new(&home).with_hermes_home_env(Some(hermes_home));
        let profiles = discover_named_profiles(&context).unwrap();
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].name, "coder");
    }

    #[test]
    fn uninstall_keep_data_yes_preserves_home() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        let hermes_home = home.join(".hermes");
        let project_root = temp.path().join("project");
        fs::create_dir_all(&project_root).unwrap();
        fs::create_dir_all(&hermes_home).unwrap();
        fs::write(
            home.join(".bashrc"),
            "export PATH=\"$HOME/.hermes/bin:$PATH\"\n",
        )
        .unwrap();
        write_wrapper(
            &home.join(".local/bin/hermes"),
            "#!/bin/sh\nexec python -m hermes_cli.main \"$@\"\n",
        );

        set_env_var("HERMES_UNINSTALL_PROJECT_ROOT", &project_root);

        let context = HermesContext::new(&home).with_hermes_home_env(Some(hermes_home.clone()));
        print_uninstall(
            &context,
            UninstallArgs {
                full: false,
                yes: true,
            },
        )
        .unwrap();

        assert!(!project_root.exists());
        assert!(hermes_home.exists());
        assert!(!home.join(".local/bin/hermes").exists());
        assert!(
            !fs::read_to_string(home.join(".bashrc"))
                .unwrap()
                .contains(".hermes/bin")
        );

        remove_env_var("HERMES_UNINSTALL_PROJECT_ROOT");
    }

    #[test]
    fn uninstall_full_yes_removes_data_and_profile_aliases() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        let hermes_home = home.join(".hermes");
        let project_root = temp.path().join("project");
        fs::create_dir_all(&project_root).unwrap();
        fs::create_dir_all(hermes_home.join("profiles").join("coder")).unwrap();
        fs::write(
            home.join(".zshrc"),
            "# Hermes Agent\nexport PATH=\"$HOME/.hermes/bin:$PATH\"\n",
        )
        .unwrap();
        write_wrapper(
            &home.join(".local/bin/hermes"),
            "#!/bin/sh\nexec python -m hermes_cli.main \"$@\"\n",
        );
        write_wrapper(
            &home.join(".local/bin/coder"),
            "#!/bin/sh\nexec hermes -p coder \"$@\"\n",
        );

        set_env_var("HERMES_UNINSTALL_PROJECT_ROOT", &project_root);

        let context = HermesContext::new(&home).with_hermes_home_env(Some(hermes_home.clone()));
        print_uninstall(
            &context,
            UninstallArgs {
                full: true,
                yes: true,
            },
        )
        .unwrap();

        assert!(!project_root.exists());
        assert!(!hermes_home.exists());
        assert!(!home.join(".local/bin/hermes").exists());
        assert!(!home.join(".local/bin/coder").exists());

        remove_env_var("HERMES_UNINSTALL_PROJECT_ROOT");
    }

    #[test]
    fn cleanup_gateway_for_profile_removes_named_profile_user_service() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        let hermes_home = home.join(".hermes");
        let profile_home = hermes_home.join("profiles").join("coder");
        let user_unit_dir = home.join(".config").join("systemd").join("user");
        fs::create_dir_all(&profile_home).unwrap();
        fs::create_dir_all(&user_unit_dir).unwrap();
        let unit_path = user_unit_dir.join("hermes-gateway-coder.service");
        fs::write(&unit_path, "[Unit]\nDescription=Hermes Gateway\n").unwrap();

        let fake_bin = home.join("bin");
        fs::create_dir_all(&fake_bin).unwrap();
        let log_path = home.join("systemctl.log");
        let systemctl = fake_bin.join("systemctl");
        fs::write(
            &systemctl,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"{}\"\nif [ \"$1\" = \"--user\" ] && [ \"$2\" = \"is-system-running\" ]; then\n  echo running\n  exit 0\nfi\nif [ \"$1\" = \"is-system-running\" ]; then\n  echo running\n  exit 0\nfi\nexit 0\n",
                log_path.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&systemctl).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&systemctl, perms).unwrap();
        }

        let path = env::var_os("PATH").unwrap_or_default();
        let mut paths = vec![fake_bin.clone()];
        paths.extend(env::split_paths(&path));
        let joined = env::join_paths(paths).unwrap();
        set_env_var("PATH", &joined);

        let context = HermesContext::new(&home).with_hermes_home_env(Some(hermes_home));
        assert!(cleanup_gateway_for_profile(&context, Some(&profile_home)));
        assert!(!unit_path.exists());

        let log = fs::read_to_string(log_path).unwrap();
        assert!(log.contains("--user stop hermes-gateway-coder"));
        assert!(log.contains("--user disable hermes-gateway-coder"));
        assert!(log.contains("--user daemon-reload"));
    }
}
