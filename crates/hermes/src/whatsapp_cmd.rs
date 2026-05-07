use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use hermes_core::HermesContext;

use crate::python_bridge::project_root;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WhatsAppMode {
    Bot,
    SelfChat,
}

impl WhatsAppMode {
    fn env_value(self) -> &'static str {
        match self {
            Self::Bot => "bot",
            Self::SelfChat => "self-chat",
        }
    }

    fn display_label(self) -> &'static str {
        match self {
            Self::Bot => "separate bot number",
            Self::SelfChat => "personal number (self-chat)",
        }
    }

    fn allows_wildcard(self) -> bool {
        matches!(self, Self::Bot)
    }
}

trait PromptUi {
    fn is_interactive(&self) -> bool;
    fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>>;
    fn prompt(&mut self, prompt: &str) -> Result<Option<String>, Box<dyn Error>>;

    fn blank(&mut self) -> Result<(), Box<dyn Error>> {
        self.line("")
    }
}

struct TerminalUi;

impl PromptUi for TerminalUi {
    fn is_interactive(&self) -> bool {
        io::stdin().is_terminal()
    }

    fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>> {
        println!("{text}");
        Ok(())
    }

    fn prompt(&mut self, prompt: &str) -> Result<Option<String>, Box<dyn Error>> {
        let mut stdout = io::stdout();
        stdout.write_all(prompt.as_bytes())?;
        stdout.flush()?;
        let mut input = String::new();
        let read = io::stdin().read_line(&mut input)?;
        if read == 0 {
            return Ok(None);
        }
        Ok(Some(input.trim().to_string()))
    }
}

pub fn print_whatsapp(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let mut ui = TerminalUi;
    if !ui.is_interactive() {
        return Err(
            "Error: 'hermes whatsapp' requires an interactive terminal.\nIt cannot be run through a pipe or non-interactive subprocess.\nRun it directly in your terminal instead."
                .into(),
        );
    }
    run_whatsapp_setup(context, &mut ui)
}

fn run_whatsapp_setup(
    context: &HermesContext,
    ui: &mut dyn PromptUi,
) -> Result<(), Box<dyn Error>> {
    ui.blank()?;
    ui.line("⚕ WhatsApp Setup")?;
    ui.line("==================================================")?;

    let env_path = context.env_path();
    let mode = match load_mode(&env_path) {
        Some(mode) => {
            ui.blank()?;
            ui.line(&format!("✓ Mode: {}", mode.display_label()))?;
            mode
        }
        None => match prompt_mode(ui)? {
            Some(mode) => {
                save_env_value(&env_path, "WHATSAPP_MODE", mode.env_value())?;
                ui.line(&format!("  ✓ Mode: {}", mode.display_label()))?;
                if matches!(mode, WhatsAppMode::Bot) {
                    ui.blank()?;
                    ui.line("  Use a second WhatsApp number for the bot.")?;
                    ui.line(
                        "  WhatsApp Business on the same phone is usually the simplest option.",
                    )?;
                }
                mode
            }
            None => {
                ui.line("Setup cancelled.")?;
                return Ok(());
            }
        },
    };

    ui.blank()?;
    match load_env_value(&env_path, "WHATSAPP_ENABLED").as_deref() {
        Some(value) if value.eq_ignore_ascii_case("true") => {
            ui.line("✓ WhatsApp is already enabled")?
        }
        _ => {
            save_env_value(&env_path, "WHATSAPP_ENABLED", "true")?;
            ui.line("✓ WhatsApp enabled")?;
        }
    }

    let current_users = load_env_value(&env_path, "WHATSAPP_ALLOWED_USERS");
    configure_allowed_users(ui, &env_path, mode, current_users.as_deref())?;

    let bridge_dir = whatsapp_bridge_dir();
    let bridge_script = bridge_dir.join("bridge.js");
    if !bridge_script.exists() {
        ui.blank()?;
        ui.line(&format!(
            "✗ Bridge script not found at {}",
            bridge_script.display()
        ))?;
        return Ok(());
    }

    if !ensure_bridge_dependencies(ui, &bridge_dir)? {
        return Ok(());
    }

    let session_dir = context.hermes_home().join("whatsapp").join("session");
    fs::create_dir_all(&session_dir)?;
    let creds_path = session_dir.join("creds.json");
    if creds_path.exists() {
        ui.line("✓ Existing WhatsApp session found")?;
        match prompt_yes_no(
            ui,
            "\n  Re-pair? This will clear the existing session. [y/N] ",
            false,
        )? {
            Some(true) => {
                fs::remove_dir_all(&session_dir).ok();
                fs::create_dir_all(&session_dir)?;
                ui.line("  ✓ Session cleared")?;
            }
            Some(false) => {
                ui.blank()?;
                ui.line("✓ WhatsApp is configured and paired!")?;
                ui.line("  Start the gateway with: hermes gateway")?;
                return Ok(());
            }
            None => {
                ui.line("Setup cancelled.")?;
                return Ok(());
            }
        }
    }

    ui.blank()?;
    ui.line("--------------------------------------------------")?;
    match mode {
        WhatsAppMode::Bot => {
            ui.line("Open WhatsApp or WhatsApp Business on the bot phone and scan the QR code.")?;
        }
        WhatsAppMode::SelfChat => {
            ui.line("Open WhatsApp on your phone and scan the QR code.")?;
        }
    }
    ui.blank()?;
    ui.line("Settings -> Linked Devices -> Link a Device")?;
    ui.line("--------------------------------------------------")?;
    ui.blank()?;

    run_pairing(&bridge_dir, &bridge_script, &session_dir)?;

    ui.blank()?;
    if creds_path.exists() {
        ui.line("✓ WhatsApp paired successfully!")?;
        ui.blank()?;
        ui.line("  Next steps:")?;
        ui.line("    1. Start the gateway: hermes gateway")?;
        match mode {
            WhatsAppMode::Bot => {
                ui.line("    2. Send a message to the bot's WhatsApp number")?;
            }
            WhatsAppMode::SelfChat => {
                ui.line("    2. Open WhatsApp -> Message Yourself")?;
            }
        }
        ui.line("    3. Hermes will reply automatically")?;
        ui.blank()?;
        ui.line("  Or install as a service: hermes gateway install")?;
    } else {
        ui.line("⚠ Pairing may not have completed. Run 'hermes whatsapp' to try again.")?;
    }

    Ok(())
}

fn prompt_mode(ui: &mut dyn PromptUi) -> Result<Option<WhatsAppMode>, Box<dyn Error>> {
    ui.blank()?;
    ui.line("How will you use WhatsApp with Hermes?")?;
    ui.blank()?;
    ui.line("  1. Separate bot number")?;
    ui.line("     People message the bot directly.")?;
    ui.blank()?;
    ui.line("  2. Personal number (self-chat)")?;
    ui.line("     You message yourself to talk to Hermes.")?;
    ui.blank()?;

    loop {
        match ui.prompt("  Choose [1/2]: ")? {
            Some(choice) => match choice.trim() {
                "1" => return Ok(Some(WhatsAppMode::Bot)),
                "2" => return Ok(Some(WhatsAppMode::SelfChat)),
                _ => ui.line("  Enter 1 or 2.")?,
            },
            None => return Ok(None),
        }
    }
}

fn configure_allowed_users(
    ui: &mut dyn PromptUi,
    env_path: &Path,
    mode: WhatsAppMode,
    current: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    match current.map(str::trim).filter(|value| !value.is_empty()) {
        Some(value) => {
            ui.line(&format!("✓ Allowed users: {value}"))?;
            match prompt_yes_no(ui, "\n  Update allowed users? [y/N] ", false)? {
                Some(true) => {
                    if let Some(updated) = prompt_allowed_users(ui, mode)? {
                        if !updated.is_empty() {
                            save_env_value(env_path, "WHATSAPP_ALLOWED_USERS", &updated)?;
                            ui.line(&format!("  ✓ Updated to: {updated}"))?;
                        }
                    } else {
                        ui.line("Setup cancelled.")?;
                    }
                }
                Some(false) => {}
                None => ui.line("Setup cancelled.")?,
            }
        }
        None => {
            ui.blank()?;
            let prompt = if mode.allows_wildcard() {
                "  Phone numbers (comma-separated, or * for anyone): "
            } else {
                "  Your phone number (digits only, e.g. 15551234567): "
            };
            match prompt_allowed_users_with_message(ui, prompt, mode)? {
                Some(value) if !value.is_empty() => {
                    save_env_value(env_path, "WHATSAPP_ALLOWED_USERS", &value)?;
                    ui.line(&format!("  ✓ Allowed users set: {value}"))?;
                }
                Some(_) => ui.line(
                    "  ⚠ No allowlist configured — Hermes will respond to all incoming messages.",
                )?,
                None => ui.line("Setup cancelled.")?,
            }
        }
    }

    Ok(())
}

fn prompt_allowed_users(
    ui: &mut dyn PromptUi,
    mode: WhatsAppMode,
) -> Result<Option<String>, Box<dyn Error>> {
    let prompt = if mode.allows_wildcard() {
        "  Phone numbers that can message the bot: "
    } else {
        "  Your phone number (digits only, e.g. 15551234567): "
    };
    prompt_allowed_users_with_message(ui, prompt, mode)
}

fn prompt_allowed_users_with_message(
    ui: &mut dyn PromptUi,
    prompt: &str,
    mode: WhatsAppMode,
) -> Result<Option<String>, Box<dyn Error>> {
    loop {
        match ui.prompt(prompt)? {
            Some(input) => match normalize_allowed_users(&input, mode.allows_wildcard()) {
                Ok(value) => return Ok(Some(value)),
                Err(error) => ui.line(&format!("  {error}"))?,
            },
            None => return Ok(None),
        }
    }
}

fn prompt_yes_no(
    ui: &mut dyn PromptUi,
    prompt: &str,
    default: bool,
) -> Result<Option<bool>, Box<dyn Error>> {
    loop {
        match ui.prompt(prompt)? {
            Some(input) => {
                let normalized = input.trim().to_ascii_lowercase();
                match normalized.as_str() {
                    "" => return Ok(Some(default)),
                    "y" | "yes" => return Ok(Some(true)),
                    "n" | "no" => return Ok(Some(false)),
                    _ => ui.line("  Enter y or n.")?,
                }
            }
            None => return Ok(None),
        }
    }
}

fn normalize_allowed_users(input: &str, allow_wildcard: bool) -> Result<String, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    if trimmed == "*" {
        if allow_wildcard {
            return Ok(String::from("*"));
        }
        return Err(String::from("Wildcard '*' is only allowed in bot mode."));
    }

    let mut entries = Vec::new();
    for raw_entry in trimmed.split(',') {
        let entry = raw_entry.trim();
        if entry.is_empty() {
            return Err(String::from("Phone list contains an empty entry."));
        }
        let cleaned = normalize_phone_number(entry)?;
        entries.push(cleaned);
    }
    Ok(entries.join(","))
}

fn normalize_phone_number(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    let mut cleaned = String::new();
    for ch in trimmed.chars() {
        if ch.is_ascii_digit() {
            cleaned.push(ch);
            continue;
        }
        if ch == '+' || ch == '-' || ch == '(' || ch == ')' || ch.is_whitespace() {
            continue;
        }
        return Err(format!("Invalid phone number: {value}"));
    }
    if cleaned.len() < 7 || cleaned.len() > 20 {
        return Err(format!("Invalid phone number: {value}"));
    }
    Ok(cleaned)
}

fn load_mode(env_path: &Path) -> Option<WhatsAppMode> {
    match load_env_value(env_path, "WHATSAPP_MODE")
        .as_deref()
        .map(str::trim)
        .unwrap_or_default()
    {
        "bot" => Some(WhatsAppMode::Bot),
        "self-chat" => Some(WhatsAppMode::SelfChat),
        _ => None,
    }
}

fn load_env_value(path: &Path, key: &str) -> Option<String> {
    if let Ok(value) = env::var(key) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    let contents = fs::read_to_string(path).ok()?;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix(key) else {
            continue;
        };
        let Some(value) = rest.strip_prefix('=') else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

fn save_env_value(path: &Path, key: &str, value: &str) -> Result<(), Box<dyn Error>> {
    if !is_env_key(key) {
        return Err(format!("invalid environment variable name: {key}").into());
    }
    let sanitized = value.replace('\n', "").replace('\r', "");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut lines = if path.exists() {
        fs::read_to_string(path)?
            .lines()
            .map(|line| format!("{line}\n"))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let mut found = false;
    for line in &mut lines {
        if line
            .strip_prefix(key)
            .is_some_and(|rest| rest.starts_with('='))
        {
            *line = format!("{key}={sanitized}\n");
            found = true;
            break;
        }
    }
    if !found {
        lines.push(format!("{key}={sanitized}\n"));
    }
    atomic_write(path, lines.concat().as_bytes())
}

fn is_env_key(key: &str) -> bool {
    if key.is_empty() {
        return false;
    }
    let mut chars = key.chars();
    match chars.next() {
        Some(ch) if ch.is_ascii_uppercase() || ch == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".tmp-{unique}"));
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn ensure_bridge_dependencies(
    ui: &mut dyn PromptUi,
    bridge_dir: &Path,
) -> Result<bool, Box<dyn Error>> {
    if bridge_dir.join("node_modules").exists() {
        ui.line("✓ Bridge dependencies already installed")?;
        return Ok(true);
    }

    ui.blank()?;
    ui.line("→ Installing WhatsApp bridge dependencies...")?;
    let output = match run_command_capture(
        "npm",
        &["install", "--no-fund", "--no-audit", "--progress=false"],
        bridge_dir,
    ) {
        Ok(output) => output,
        Err(error) => {
            ui.line(&format!("  ✗ npm not available: {error}"))?;
            return Ok(false);
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let preview = stderr
            .lines()
            .rev()
            .take(30)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        ui.line("  ✗ npm install failed:")?;
        ui.line(if preview.trim().is_empty() {
            "(no output)"
        } else {
            &preview
        })?;
        return Ok(false);
    }

    ui.line("  ✓ Dependencies installed")?;
    Ok(true)
}

fn run_pairing(
    bridge_dir: &Path,
    bridge_script: &Path,
    session_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let status = Command::new("node")
        .arg(bridge_script)
        .arg("--pair-only")
        .arg("--session")
        .arg(session_dir)
        .current_dir(bridge_dir)
        .status();
    match status {
        Ok(_) => Ok(()),
        Err(error) => Err(format!("failed to run node pairing bridge: {error}").into()),
    }
}

fn run_command_capture(command: &str, args: &[&str], cwd: &Path) -> Result<Output, Box<dyn Error>> {
    Command::new(command)
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|error| format!("{command}: {error}").into())
}

fn whatsapp_bridge_dir() -> PathBuf {
    env::var_os("HERMES_WHATSAPP_BRIDGE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root().join("scripts").join("whatsapp-bridge"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::sync::{Mutex, OnceLock};
    use tempfile::TempDir;

    fn test_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe {
            env::set_var(key, value);
        }
    }

    fn remove_env_var(key: &str) {
        unsafe {
            env::remove_var(key);
        }
    }

    struct TestUi {
        answers: VecDeque<String>,
        output: String,
    }

    impl TestUi {
        fn new(answers: &[&str]) -> Self {
            Self {
                answers: answers.iter().map(|value| value.to_string()).collect(),
                output: String::new(),
            }
        }
    }

    impl PromptUi for TestUi {
        fn is_interactive(&self) -> bool {
            true
        }

        fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>> {
            self.output.push_str(text);
            self.output.push('\n');
            Ok(())
        }

        fn prompt(&mut self, prompt: &str) -> Result<Option<String>, Box<dyn Error>> {
            self.output.push_str(prompt);
            Ok(self.answers.pop_front())
        }
    }

    fn test_context() -> (TempDir, HermesContext) {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        (temp, context)
    }

    #[test]
    fn normalize_allowed_users_accepts_wildcard_and_phone_lists() {
        assert_eq!(normalize_allowed_users("*", true).unwrap(), "*");
        assert_eq!(
            normalize_allowed_users("+1 (555) 123-4567, 447700900123", false).unwrap(),
            "15551234567,447700900123"
        );
    }

    #[test]
    fn normalize_allowed_users_rejects_invalid_entries() {
        assert!(normalize_allowed_users("*", false).is_err());
        assert!(normalize_allowed_users("abc", true).is_err());
        assert!(normalize_allowed_users("123, ,456", true).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn whatsapp_setup_writes_env_and_pairs_session() {
        let _guard = test_env_lock().lock().unwrap();
        let (temp, context) = test_context();
        let bridge_dir = temp.path().join("wa-bridge");
        let bin_dir = temp.path().join("bin");
        fs::create_dir_all(&bridge_dir).unwrap();
        fs::create_dir_all(&bin_dir).unwrap();
        fs::write(bridge_dir.join("bridge.js"), "console.log('bridge');\n").unwrap();

        let log = temp.path().join("commands.log");
        let npm = bin_dir.join("npm");
        fs::write(
            &npm,
            format!(
                "#!/bin/sh\nprintf 'npm %s\\n' \"$*\" >> '{}'\nmkdir -p node_modules\n",
                log.display()
            ),
        )
        .unwrap();
        let node = bin_dir.join("node");
        fs::write(
            &node,
            format!(
                "#!/bin/sh\nprintf 'node %s\\n' \"$*\" >> '{}'\nSESSION=''\nPREV=''\nfor ARG in \"$@\"; do\n  if [ \"$PREV\" = '1' ]; then SESSION=\"$ARG\"; PREV=''; continue; fi\n  if [ \"$ARG\" = '--session' ]; then PREV='1'; fi\n done\nmkdir -p \"$SESSION\"\nprintf '{{}}' > \"$SESSION/creds.json\"\n",
                log.display()
            ),
        )
        .unwrap();
        for path in [&npm, &node] {
            let mut perms = fs::metadata(path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).unwrap();
        }

        let old_path = env::var_os("PATH");
        set_env_var(
            "PATH",
            format!(
                "{}:{}",
                bin_dir.display(),
                env::var("PATH").unwrap_or_default()
            ),
        );
        set_env_var("HERMES_WHATSAPP_BRIDGE_DIR", &bridge_dir);

        let mut ui = TestUi::new(&["1", "15551234567"]);
        run_whatsapp_setup(&context, &mut ui).unwrap();

        let env_contents = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_contents.contains("WHATSAPP_MODE=bot"));
        assert!(env_contents.contains("WHATSAPP_ENABLED=true"));
        assert!(env_contents.contains("WHATSAPP_ALLOWED_USERS=15551234567"));
        assert!(
            context
                .hermes_home()
                .join("whatsapp")
                .join("session")
                .join("creds.json")
                .exists()
        );

        let commands = fs::read_to_string(&log).unwrap();
        assert!(commands.contains("npm install --no-fund --no-audit --progress=false"));
        assert!(commands.contains("node"));
        assert!(ui.output.contains("WhatsApp paired successfully"));

        match old_path {
            Some(value) => set_env_var("PATH", value),
            None => remove_env_var("PATH"),
        }
        remove_env_var("HERMES_WHATSAPP_BRIDGE_DIR");
    }
}
