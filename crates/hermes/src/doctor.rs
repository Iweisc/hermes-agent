use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Args;
use hermes_core::{
    EnvLoadReport, HermesConfig, HermesContext, LoadedConfig, ModelOverrides, SessionStore,
    get_tool_definitions, get_toolset_info, is_container, is_wsl,
};

#[derive(Args, Debug)]
pub struct DoctorArgs {
    #[arg(long)]
    pub fix: bool,
}

pub fn print_doctor(
    context: &HermesContext,
    env_report: &EnvLoadReport,
    loaded: &LoadedConfig,
    session_store: &SessionStore,
    args: DoctorArgs,
) -> Result<(), Box<dyn Error>> {
    let mut issues = Vec::new();
    let mut fixes = Vec::new();

    if args.fix {
        if ensure_env_file(context.env_path())? {
            fixes.push(format!("created {}", context.env_path().display()));
        }
        if ensure_config_file(&context.config_path())? {
            fixes.push(format!("created {}", context.config_path().display()));
        }
    }

    let output = render_doctor(
        context,
        env_report,
        loaded,
        session_store,
        &mut issues,
        &fixes,
    );
    println!("{output}");
    Ok(())
}

fn render_doctor(
    context: &HermesContext,
    env_report: &EnvLoadReport,
    loaded: &LoadedConfig,
    session_store: &SessionStore,
    issues: &mut Vec<String>,
    fixes: &[String],
) -> String {
    let mut lines = Vec::new();
    let config_exists = context.config_path().exists();
    let env_exists = context.env_path().exists();
    let state_db = session_store.path().clone();
    let session_count = session_store.session_count().unwrap_or_default();
    let message_count = session_store.message_count(None).unwrap_or_default();
    let state_size = fs::metadata(&state_db)
        .ok()
        .map(|meta| meta.len())
        .unwrap_or(0);
    let runtime = context.resolve_model_runtime(loaded, &ModelOverrides::default());
    let current_toolsets = if loaded.config.toolsets.is_empty() {
        vec![String::from("hermes-cli")]
    } else {
        loaded.config.toolsets.clone()
    };
    let disabled_toolsets = disabled_memory_toolsets(&loaded.config.memory);
    let available_tools =
        get_tool_definitions(Some(&current_toolsets), disabled_toolsets.as_deref()).len();

    lines.push(String::from("=== Hermes Doctor ==="));
    lines.push(String::new());
    lines.push(String::from("Runtime"));
    lines.push(format!("  rust:          {}", rust_version()));
    lines.push(format!(
        "  profile:       {}",
        context.current_profile_name()
    ));
    lines.push(format!(
        "  home:          {}",
        context.hermes_home().display()
    ));
    lines.push(format!(
        "  config:        {}",
        status_label(config_exists, "present", "missing")
    ));
    lines.push(format!(
        "  env:           {}",
        status_label(env_exists, "present", "missing")
    ));
    lines.push(format!(
        "  env_files:     {}",
        env_report.loaded_paths.len()
    ));
    lines.push(format!("  termux:        {}", context.is_termux()));
    lines.push(format!("  wsl:           {}", is_wsl()));
    lines.push(format!("  container:     {}", is_container()));

    if !config_exists {
        issues.push(format!("missing {}", context.config_path().display()));
    }
    if !env_exists {
        issues.push(format!("missing {}", context.env_path().display()));
    }

    if !loaded.warnings.is_empty() || !env_report.warnings.is_empty() {
        lines.push(String::new());
        lines.push(String::from("Warnings"));
        for warning in &loaded.warnings {
            lines.push(format!("  config:        {warning}"));
            issues.push(warning.clone());
        }
        for warning in &env_report.warnings {
            lines.push(format!("  env:           {warning}"));
            issues.push(warning.clone());
        }
    }

    lines.push(String::new());
    lines.push(String::from("Model Runtime"));
    match runtime {
        Ok(runtime) => {
            lines.push(format!("  status:        ok"));
            lines.push(format!("  provider:      {}", runtime.provider));
            lines.push(format!("  model:         {}", runtime.model));
            lines.push(format!("  api_mode:      {}", runtime.api_mode));
            lines.push(format!("  base_url:      {}", runtime.base_url));
        }
        Err(error) => {
            lines.push(String::from("  status:        fail"));
            lines.push(format!("  detail:        {error}"));
            issues.push(format!("model runtime resolution failed: {error}"));
        }
    }

    lines.push(String::new());
    lines.push(String::from("State"));
    lines.push(format!("  state_db:      {}", state_db.display()));
    lines.push(format!("  db_size:       {}", format_bytes(state_size)));
    lines.push(format!("  sessions:      {session_count}"));
    lines.push(format!("  messages:      {message_count}"));

    lines.push(String::new());
    lines.push(String::from("External Tools"));
    let terminal_backend = loaded.config.terminal.backend.to_ascii_lowercase();
    for (cmd, required, note) in external_tools(&terminal_backend) {
        let present = command_exists(cmd);
        lines.push(format!(
            "  {cmd:<13} {}{}",
            if present { "ok" } else { "missing" },
            if note.is_empty() {
                String::new()
            } else {
                format!(" ({note})")
            }
        ));
        if required && !present {
            issues.push(format!("required external tool '{cmd}' is missing"));
        }
    }

    lines.push(String::new());
    lines.push(String::from("Toolsets"));
    lines.push(format!("  enabled:       {}", current_toolsets.join(", ")));
    if let Some(disabled) = disabled_toolsets.as_ref() {
        if !disabled.is_empty() {
            lines.push(format!("  disabled:      {}", disabled.join(", ")));
        }
    }
    lines.push(format!("  tools:         {available_tools} available"));
    for toolset in &current_toolsets {
        if disabled_toolsets
            .as_ref()
            .is_some_and(|disabled| disabled.iter().any(|item| item == toolset))
        {
            lines.push(format!("  - {toolset}:     disabled by config"));
            continue;
        }
        match get_toolset_info(toolset) {
            Some(info) => lines.push(format!(
                "  - {toolset}:     {} tool(s)",
                info.implemented_tools.len()
            )),
            None => {
                lines.push(format!("  - {toolset}:     unknown"));
                issues.push(format!("unknown toolset '{toolset}'"));
            }
        }
    }

    if !fixes.is_empty() {
        lines.push(String::new());
        lines.push(String::from("Fixes"));
        for fix in fixes {
            lines.push(format!("  - {fix}"));
        }
    }

    lines.push(String::new());
    lines.push(String::from("Summary"));
    if issues.is_empty() {
        lines.push(String::from("  status:        ok"));
        lines.push(String::from("  detail:        all checks passed"));
    } else {
        lines.push(String::from("  status:        issues"));
        lines.push(format!("  count:         {}", issues.len()));
        for issue in issues.iter() {
            lines.push(format!("  - {issue}"));
        }
    }

    lines.join("\n")
}

fn disabled_memory_toolsets(memory: &hermes_core::MemoryConfig) -> Option<Vec<String>> {
    (!memory.any_enabled()).then(|| vec![String::from("memory")])
}

fn ensure_env_file(path: PathBuf) -> Result<bool, Box<dyn Error>> {
    if path.exists() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, b"")?;
    Ok(true)
}

fn ensure_config_file(path: &Path) -> Result<bool, Box<dyn Error>> {
    if path.exists() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let rendered = serde_yaml::to_string(&HermesConfig::default())?;
    atomic_write(path, rendered.as_bytes())?;
    Ok(true)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{unique}"));
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn rust_version() -> String {
    match std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if text.is_empty() {
                String::from("(unknown)")
            } else {
                text
            }
        }
        _ => String::from("(unknown)"),
    }
}

fn status_label(ok: bool, ok_text: &str, missing_text: &str) -> String {
    if ok {
        ok_text.to_string()
    } else {
        missing_text.to_string()
    }
}

fn format_bytes(size: u64) -> String {
    if size >= 1024 * 1024 {
        format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
    } else if size >= 1024 {
        format!("{:.1} KB", size as f64 / 1024.0)
    } else {
        format!("{size} B")
    }
}

fn command_exists(command: &str) -> bool {
    let Some(path_var) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path_var).any(|dir| {
        let full = dir.join(command);
        full.is_file() || cfg!(windows) && full.with_extension("exe").is_file()
    })
}

fn external_tools(terminal_backend: &str) -> Vec<(&'static str, bool, &'static str)> {
    let mut tools = vec![
        ("git", false, "recommended"),
        ("rg", false, "recommended"),
        ("node", false, "browser tools"),
        ("python3", false, "provider bridges"),
        ("bash", true, "shell runtime"),
    ];
    if terminal_backend == "docker" {
        tools.push(("docker", true, "terminal backend"));
    } else {
        tools.push(("docker", false, "optional"));
    }
    tools
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-doctor-{label}-{unique}"))
    }

    #[test]
    fn ensure_env_file_creates_missing_file() {
        let path = temp_path("env").join(".env");
        assert!(ensure_env_file(path.clone()).unwrap());
        assert!(path.exists());
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn ensure_config_file_writes_defaults() {
        let path = temp_path("config").join("config.yaml");
        assert!(ensure_config_file(&path).unwrap());
        let written = fs::read_to_string(&path).unwrap();
        assert!(written.contains("toolsets:"));
        let _ = fs::remove_dir_all(path.parent().unwrap());
    }
}
