use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_yaml::{Mapping, Value as YamlValue};
use unicode_normalization::UnicodeNormalization;

use crate::LoadedConfig;

const DEFAULT_TIMEOUT_SECONDS: u64 = 60;
const DEFAULT_MODE: &str = "manual";
const DEFAULT_CRON_MODE: &str = "deny";
const DEFAULT_SESSION_KEY: &str = "default";
const SENSITIVE_WRITE_TARGET: &str = r#"(?:/etc/|/dev/sd|(?:~|\$home|\$\{home\})/\.ssh(?:/|$)|(?:~\/\.hermes/|(?:\$home|\$\{home\})/\.hermes/|(?:\$hermes_home|\$\{hermes_home\})/)\.env\b|(?:~|\$home|\$\{home\})/\.(?:bashrc|zshrc|profile|bash_profile|zprofile)\b|(?:~|\$home|\$\{home\})/\.(?:netrc|pgpass|npmrc|pypirc)\b)"#;
const PROJECT_SENSITIVE_WRITE_TARGET: &str = r#"(?:(?:(?:/|\.{1,2}/)?(?:[^\s/"'`]+/)*\.env(?:\.[^/\s"'`]+)*)|(?:(?:/|\.{1,2}/)?(?:[^\s/"'`]+/)*config\.yaml))"#;
const COMMAND_TAIL: &str = r"(?:\s*(?:&&|\|\||;).*)?$";
const CMDPOS: &str = r"(?:^|[;&|\n`]|\$\()\s*(?:sudo\s+(?:-[^\s]+\s+)*)?(?:env\s+(?:\w+=\S*\s+)*)?(?:(?:exec|nohup|setsid|time)\s+)*\s*";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub command: String,
    pub description: String,
    pub pattern_keys: Vec<String>,
    pub choices: Vec<String>,
    pub allow_permanent: bool,
}

#[derive(Debug, Clone)]
pub struct ApprovalManager {
    hermes_home: PathBuf,
    mode: ApprovalMode,
    timeout_seconds: u64,
    cron_mode: CronApprovalMode,
    permanent_approved: HashSet<String>,
    session_approved: HashMap<String, HashSet<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApprovalMode {
    Manual,
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CronApprovalMode {
    Approve,
    Deny,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalCheckResult {
    Approved,
    Blocked { message: String },
}

impl ApprovalManager {
    pub fn load_for_runtime(hermes_home: &Path, loaded: &LoadedConfig) -> Self {
        let approvals = loaded
            .raw
            .as_mapping()
            .and_then(|root| mapping_value(root, "approvals"))
            .and_then(YamlValue::as_mapping);
        let mode = approvals
            .and_then(|mapping| mapping_value(mapping, "mode"))
            .map(parse_approval_mode)
            .unwrap_or(ApprovalMode::Manual);
        let timeout_seconds = approvals
            .and_then(|mapping| mapping_value(mapping, "timeout"))
            .and_then(yaml_u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECONDS)
            .clamp(1, 3_600);
        let cron_mode = approvals
            .and_then(|mapping| mapping_value(mapping, "cron_mode"))
            .map(parse_cron_mode)
            .unwrap_or(CronApprovalMode::Deny);
        let permanent_approved = load_allowlist_from_raw(&loaded.raw);

        Self {
            hermes_home: hermes_home.to_path_buf(),
            mode,
            timeout_seconds,
            cron_mode,
            permanent_approved,
            session_approved: HashMap::new(),
        }
    }

    pub fn check_command(
        &mut self,
        command: &str,
        env_type: &str,
        session_id: Option<&str>,
        callback: Option<&(dyn Fn(&ApprovalRequest) -> Result<String, String> + Send + Sync)>,
    ) -> ApprovalCheckResult {
        if matches!(
            env_type,
            "docker" | "singularity" | "modal" | "daytona" | "vercel_sandbox"
        ) {
            return ApprovalCheckResult::Approved;
        }

        let normalized = normalize_command_for_detection(command);
        if let Some(description) = detect_rule_match(&normalized, hardline_rules()) {
            return ApprovalCheckResult::Blocked {
                message: format!(
                    "BLOCKED (hardline): {description}. This command is on the unconditional blocklist and cannot be executed via the agent."
                ),
            };
        }

        if self.mode == ApprovalMode::Off || env_truthy("HERMES_YOLO_MODE") {
            return ApprovalCheckResult::Approved;
        }

        let Some((pattern_key, description)) = detect_dangerous_command(&normalized) else {
            return ApprovalCheckResult::Approved;
        };

        let session_key = session_key(session_id);
        if self.is_approved(session_key, &pattern_key) {
            return ApprovalCheckResult::Approved;
        }

        if env_truthy("HERMES_CRON_SESSION") && self.cron_mode == CronApprovalMode::Deny {
            return ApprovalCheckResult::Blocked {
                message: format!(
                    "BLOCKED: Command flagged as dangerous ({description}) but cron jobs run without a user present to approve it."
                ),
            };
        }

        let Some(callback) = callback else {
            return ApprovalCheckResult::Approved;
        };

        let request = ApprovalRequest {
            command: command.trim().to_string(),
            description: description.clone(),
            pattern_keys: vec![pattern_key.clone()],
            choices: vec![
                "once".to_string(),
                "session".to_string(),
                "always".to_string(),
                "deny".to_string(),
            ],
            allow_permanent: true,
        };

        let choice = match callback(&request) {
            Ok(choice) => normalize_choice(&choice),
            Err(error) => {
                return ApprovalCheckResult::Blocked {
                    message: format!("BLOCKED: Approval request failed: {error}"),
                };
            }
        };

        match choice.as_str() {
            "once" => ApprovalCheckResult::Approved,
            "session" => {
                self.approve_session(session_key, &pattern_key);
                ApprovalCheckResult::Approved
            }
            "always" => {
                self.approve_session(session_key, &pattern_key);
                if let Err(error) = self.approve_permanent(&pattern_key) {
                    return ApprovalCheckResult::Blocked {
                        message: format!("BLOCKED: Failed to persist approval: {error}"),
                    };
                }
                ApprovalCheckResult::Approved
            }
            _ => ApprovalCheckResult::Blocked {
                message: format!(
                    "BLOCKED: User denied this potentially dangerous command ({description}). Do NOT retry."
                ),
            },
        }
    }

    fn approve_session(&mut self, session_key: &str, pattern_key: &str) {
        self.session_approved
            .entry(session_key.to_string())
            .or_default()
            .insert(pattern_key.to_string());
    }

    fn approve_permanent(&mut self, pattern_key: &str) -> Result<(), String> {
        self.permanent_approved.insert(pattern_key.to_string());
        save_allowlist_to_config(&self.hermes_home, &self.permanent_approved)
    }

    fn is_approved(&self, session_key: &str, pattern_key: &str) -> bool {
        self.permanent_approved.contains(pattern_key)
            || self
                .session_approved
                .get(session_key)
                .is_some_and(|items| items.contains(pattern_key))
    }

    pub fn timeout_seconds(&self) -> u64 {
        self.timeout_seconds
    }
}

fn hardline_rules() -> &'static [(Regex, &'static str)] {
    static RULES: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    RULES.get_or_init(|| {
        vec![
            (compile(r"\brm\s+(-[^\s]*\s+)*(/|/\*|/ \*)(\s|$)"), "recursive delete of root filesystem"),
            (compile(r"\brm\s+(-[^\s]*\s+)*(/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*)(\s|$)"), "recursive delete of system directory"),
            (compile(r"\brm\s+(-[^\s]*\s+)*(~|\$HOME)(/?|/\*)?(\s|$)"), "recursive delete of home directory"),
            (compile(r"\bmkfs(\.[a-z0-9]+)?\b"), "format filesystem (mkfs)"),
            (compile(r"\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*"), "dd to raw block device"),
            (compile(r">\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b"), "redirect to raw block device"),
            (compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "fork bomb"),
            (compile(r"\bkill\s+(-[^\s]+\s+)*-1\b"), "kill all processes"),
            (compile(&format!(r"{CMDPOS}(shutdown|reboot|halt|poweroff)\b")), "system shutdown/reboot"),
            (compile(&format!(r"{CMDPOS}init\s+[06]\b")), "init 0/6 (shutdown/reboot)"),
            (compile(&format!(r"{CMDPOS}systemctl\s+(poweroff|reboot|halt|kexec)\b")), "systemctl poweroff/reboot"),
            (compile(&format!(r"{CMDPOS}telinit\s+[06]\b")), "telinit 0/6 (shutdown/reboot)"),
        ]
    })
}

fn dangerous_rules() -> &'static [(Regex, &'static str)] {
    static RULES: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    RULES.get_or_init(|| {
        vec![
            (compile(r"\brm\s+(-[^\s]*\s+)*/"), "delete in root path"),
            (compile(r"\brm\s+-[^\s]*r"), "recursive delete"),
            (compile(r"\brm\s+--recursive\b"), "recursive delete (long flag)"),
            (compile(r"\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b"), "world/other-writable permissions"),
            (compile(r"\bchown\s+(-[^\s]*)?R\s+root"), "recursive chown to root"),
            (compile(r"\bmkfs\b"), "format filesystem"),
            (compile(r"\bdd\s+.*if="), "disk copy"),
            (compile(r">\s*/dev/sd"), "write to block device"),
            (compile(r"\bdrop\s+(table|database)\b"), "SQL DROP"),
            (compile(r"\bdelete\s+from\b"), "SQL DELETE"),
            (compile(r"\btruncate\s+(table)?\s*\w"), "SQL TRUNCATE"),
            (compile(r">\s*/etc/"), "overwrite system config"),
            (compile(r"\bsystemctl\s+(-[^\s]+\s+)*(stop|restart|disable|mask)\b"), "stop/restart system service"),
            (compile(r"\bkill\s+-9\s+-1\b"), "kill all processes"),
            (compile(r"\bpkill\s+-9\b"), "force kill processes"),
            (compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "fork bomb"),
            (compile(r"\b(bash|sh|zsh|ksh)\s+-[^\s]*c(\s+|$)"), "shell command via -c/-lc flag"),
            (compile(r"\b(python[23]?|perl|ruby|node)\s+-[ec]\s+"), "script execution via -e/-c flag"),
            (compile(r"\b(curl|wget)\b.*\|\s*(ba)?sh\b"), "pipe remote content to shell"),
            (compile(r"\b(bash|sh|zsh|ksh)\s+<\s*<?\s*\(\s*(curl|wget)\b"), "execute remote script via process substitution"),
            (compile(&format!(r#"\btee\b.*["']?{SENSITIVE_WRITE_TARGET}"#)), "overwrite system file via tee"),
            (compile(&format!(r#">>?\s*["']?{SENSITIVE_WRITE_TARGET}"#)), "overwrite system file via redirection"),
            (compile(&format!(r#"\btee\b.*["']?{PROJECT_SENSITIVE_WRITE_TARGET}["']?{COMMAND_TAIL}"#)), "overwrite project env/config via tee"),
            (compile(&format!(r#">>?\s*["']?{PROJECT_SENSITIVE_WRITE_TARGET}["']?{COMMAND_TAIL}"#)), "overwrite project env/config via redirection"),
            (compile(r"\bxargs\s+.*\brm\b"), "xargs with rm"),
            (compile(r"\bfind\b.*-exec\s+(/\S*/)?rm\b"), "find -exec rm"),
            (compile(r"\bfind\b.*-delete\b"), "find -delete"),
            (compile(r"\bhermes\s+gateway\s+(stop|restart)\b"), "stop/restart hermes gateway (kills running agents)"),
            (compile(r"\bhermes\s+update\b"), "hermes update (restarts gateway, kills running agents)"),
            (compile(r"\bgateway\s+run\b.*(&\s*$|&\s*;|\bdisown\b|\bsetsid\b)"), "start gateway outside systemd"),
            (compile(r"\bnohup\b.*gateway\s+run\b"), "start gateway outside systemd"),
            (compile(r"\b(pkill|killall)\b.*\b(hermes|gateway|cli\.py)\b"), "kill hermes/gateway process (self-termination)"),
            (compile(r"\bkill\b.*\$\(\s*pgrep\b"), "kill process via pgrep expansion (self-termination)"),
            (compile(r"\bkill\b.*`\s*pgrep\b"), "kill process via backtick pgrep expansion (self-termination)"),
            (compile(r"\b(cp|mv|install)\b.*\s/etc/"), "copy/move file into /etc/"),
            (compile(&format!(r#"\b(cp|mv|install)\b.*\s["']?{PROJECT_SENSITIVE_WRITE_TARGET}["']?{COMMAND_TAIL}"#)), "overwrite project env/config file"),
            (compile(r"\bsed\s+-[^\s]*i.*\s/etc/"), "in-place edit of system config"),
            (compile(r"\bsed\s+--in-place\b.*\s/etc/"), "in-place edit of system config (long flag)"),
            (compile(r"\b(python[23]?|perl|ruby|node)\s+<<"), "script execution via heredoc"),
            (compile(r"\bgit\s+reset\s+--hard\b"), "git reset --hard (destroys uncommitted changes)"),
            (compile(r"\bgit\s+push\b.*--force\b"), "git force push (rewrites remote history)"),
            (compile(r"\bgit\s+push\b.*-f\b"), "git force push short flag (rewrites remote history)"),
            (compile(r"\bgit\s+clean\s+-[^\s]*f"), "git clean with force (deletes untracked files)"),
            (compile(r"\bgit\s+branch\s+-D\b"), "git branch force delete"),
            (compile(r"\bchmod\s+\+x\b.*[;&|]+\s*\./"), "chmod +x followed by immediate execution"),
        ]
    })
}

fn detect_dangerous_command(command: &str) -> Option<(String, String)> {
    detect_rule_match_with_key(command, dangerous_rules())
}

fn detect_rule_match(command: &str, rules: &[(Regex, &'static str)]) -> Option<String> {
    rules
        .iter()
        .find(|(pattern, _)| pattern.is_match(command))
        .map(|(_, description)| (*description).to_string())
}

fn detect_rule_match_with_key(
    command: &str,
    rules: &[(Regex, &'static str)],
) -> Option<(String, String)> {
    rules.iter().find_map(|(pattern, description)| {
        pattern
            .is_match(command)
            .then(|| ((*description).to_string(), (*description).to_string()))
    })
}

fn normalize_command_for_detection(command: &str) -> String {
    strip_ansi(command)
        .replace('\0', "")
        .nfkc()
        .collect::<String>()
        .trim()
        .to_lowercase()
}

fn strip_ansi(input: &str) -> String {
    let mut output = String::new();
    let bytes = input.as_bytes();
    let mut index = 0_usize;
    while index < bytes.len() {
        if bytes[index] == 0x1b {
            index += 1;
            if index < bytes.len() && bytes[index] == b'[' {
                index += 1;
                while index < bytes.len() {
                    let ch = bytes[index];
                    index += 1;
                    if (0x40..=0x7e).contains(&ch) {
                        break;
                    }
                }
                continue;
            }
        }
        output.push(bytes[index] as char);
        index += 1;
    }
    output
}

fn compile(pattern: &str) -> Regex {
    Regex::new(pattern).expect("approval regex must compile")
}

fn parse_approval_mode(value: &YamlValue) -> ApprovalMode {
    match value {
        YamlValue::Bool(false) => ApprovalMode::Off,
        YamlValue::Bool(true) => ApprovalMode::Manual,
        YamlValue::String(text) if text.trim().eq_ignore_ascii_case("off") => ApprovalMode::Off,
        YamlValue::String(text) if text.trim().is_empty() => {
            parse_approval_mode(&YamlValue::String(DEFAULT_MODE.to_string()))
        }
        _ => ApprovalMode::Manual,
    }
}

fn parse_cron_mode(value: &YamlValue) -> CronApprovalMode {
    match value {
        YamlValue::String(text)
            if matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "approve" | "off" | "allow" | "yes"
            ) =>
        {
            CronApprovalMode::Approve
        }
        YamlValue::Bool(true) => CronApprovalMode::Approve,
        YamlValue::String(text) if text.trim().is_empty() => {
            parse_cron_mode(&YamlValue::String(DEFAULT_CRON_MODE.to_string()))
        }
        _ => CronApprovalMode::Deny,
    }
}

fn normalize_choice(choice: &str) -> String {
    match choice.trim().to_ascii_lowercase().as_str() {
        "1" | "o" | "once" | "approve" | "yes" | "y" | "ok" => "once".to_string(),
        "2" | "s" | "session" => "session".to_string(),
        "3" | "a" | "always" => "always".to_string(),
        "4" | "d" | "deny" | "no" | "n" | "" => "deny".to_string(),
        _ => "deny".to_string(),
    }
}

fn session_key(session_id: Option<&str>) -> &str {
    session_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_SESSION_KEY)
}

fn load_allowlist_from_raw(raw: &YamlValue) -> HashSet<String> {
    raw.as_mapping()
        .and_then(|root| mapping_value(root, "command_allowlist"))
        .and_then(YamlValue::as_sequence)
        .map(|items| {
            items
                .iter()
                .filter_map(YamlValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default()
}

fn save_allowlist_to_config(hermes_home: &Path, allowlist: &HashSet<String>) -> Result<(), String> {
    let path = hermes_home.join("config.yaml");
    let mut root = if path.exists() {
        match fs::read_to_string(&path) {
            Ok(contents) => serde_yaml::from_str::<YamlValue>(&contents)
                .unwrap_or_else(|_| YamlValue::Mapping(Mapping::new())),
            Err(error) => return Err(format!("reading {} failed: {error}", path.display())),
        }
    } else {
        YamlValue::Mapping(Mapping::new())
    };

    let mapping = root
        .as_mapping_mut()
        .ok_or_else(|| format!("{} must contain a YAML mapping", path.display()))?;
    let mut values = allowlist.iter().cloned().collect::<Vec<_>>();
    values.sort();
    mapping.insert(
        YamlValue::String("command_allowlist".to_string()),
        YamlValue::Sequence(values.into_iter().map(YamlValue::String).collect()),
    );
    let rendered = serde_yaml::to_string(&root).map_err(|error| error.to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&path, rendered)
        .map_err(|error| format!("writing {} failed: {error}", path.display()))
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn mapping_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

fn yaml_u64(value: &YamlValue) -> Option<u64> {
    match value {
        YamlValue::Number(number) => number.as_u64(),
        YamlValue::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    #[test]
    fn hardline_detector_matches_shutdown() {
        let normalized = normalize_command_for_detection("sudo shutdown now");
        assert_eq!(
            detect_rule_match(&normalized, hardline_rules()).as_deref(),
            Some("system shutdown/reboot")
        );
    }

    #[test]
    fn always_approval_persists_to_config_and_reloads() {
        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config.yaml");
        fs::write(
            &config_path,
            "approvals:\n  mode: manual\ncommand_allowlist:\n  - existing\n",
        )
        .unwrap();
        let loaded = LoadedConfig {
            path: config_path.clone(),
            raw: serde_yaml::from_str(
                "approvals:\n  mode: manual\ncommand_allowlist:\n  - existing\n",
            )
            .unwrap(),
            config: crate::HermesConfig::default(),
            warnings: Vec::new(),
        };
        let mut manager = ApprovalManager::load_for_runtime(temp.path(), &loaded);
        let result = manager.check_command(
            "bash -c \"printf hi\"",
            "local",
            Some("session-1"),
            Some(&|_: &ApprovalRequest| Ok("always".to_string())),
        );
        assert_eq!(result, ApprovalCheckResult::Approved);

        let saved = fs::read_to_string(config_path).unwrap();
        assert!(saved.contains("command_allowlist"));
        assert!(saved.contains("existing"));
        assert!(saved.contains("shell command via -c/-lc flag"));

        let reloaded = LoadedConfig {
            path: temp.path().join("config.yaml"),
            raw: serde_yaml::from_str(&saved).unwrap(),
            config: crate::HermesConfig::default(),
            warnings: Vec::new(),
        };
        let mut manager = ApprovalManager::load_for_runtime(temp.path(), &reloaded);
        let result = manager.check_command(
            "bash -c \"printf hi\"",
            "local",
            Some("other-session"),
            None,
        );
        assert_eq!(result, ApprovalCheckResult::Approved);
    }
}
