use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::{Component, Path, PathBuf};

use chrono::Local;
use hermes_core::{HermesConfig, HermesContext};
use serde::Serialize;
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};

use super::{MigrateArgs, MigratePreset, SkillConflict};

const ENTRY_DELIMITER: &str = "\n§\n";
const DEFAULT_MEMORY_CHAR_LIMIT: usize = 2200;
const DEFAULT_USER_CHAR_LIMIT: usize = 1375;
const SKILL_CATEGORY_DIRNAME: &str = "openclaw-imports";
const SKILL_CATEGORY_DESCRIPTION: &str = "Skills migrated from an OpenClaw workspace.\n";
const SUPPORTED_SECRET_TARGETS: &[&str] = &[
    "TELEGRAM_BOT_TOKEN",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ELEVENLABS_API_KEY",
    "VOICE_TOOLS_OPENAI_KEY",
];
const WORKSPACE_INSTRUCTIONS_FILENAME: &str = "AGENTS.md";
const CONFIG_MUTATING_OPTIONS: &[&str] = &[
    "model-config",
    "tts-config",
    "mcp-servers",
    "command-allowlist",
    "agent-config",
    "session-config",
    "full-providers",
    "deep-channels",
    "browser-config",
    "tools-config",
    "approvals-config",
];
const REASON_TARGET_EXISTS: &str = "Target exists and overwrite is disabled";
const REASON_BLOCKED_BY_APPLY_CONFLICT: &str = "blocked by earlier apply conflict";

const ALL_MIGRATION_OPTIONS: &[&str] = &[
    "soul",
    "workspace-agents",
    "memory",
    "user-profile",
    "messaging-settings",
    "secret-settings",
    "command-allowlist",
    "skills",
    "tts-assets",
    "discord-settings",
    "slack-settings",
    "whatsapp-settings",
    "signal-settings",
    "provider-keys",
    "model-config",
    "tts-config",
    "shared-skills",
    "daily-memory",
    "archive",
    "mcp-servers",
    "plugins-config",
    "cron-jobs",
    "hooks-config",
    "agent-config",
    "gateway-config",
    "session-config",
    "full-providers",
    "deep-channels",
    "browser-config",
    "tools-config",
    "approvals-config",
    "memory-backend",
    "skills-config",
    "ui-identity",
    "logging-config",
];

#[derive(Debug, Clone, Serialize)]
pub struct MigrationItem {
    pub kind: String,
    pub source: Option<String>,
    pub destination: Option<String>,
    pub status: String,
    pub reason: String,
    pub details: JsonValue,
    pub sensitive: bool,
}

#[derive(Debug, Clone)]
pub struct MigrationReport {
    pub summary: BTreeMap<String, usize>,
    pub items: Vec<MigrationItem>,
    pub output_dir: Option<PathBuf>,
    pub preset: String,
    pub warnings: Vec<String>,
    pub next_steps: Vec<String>,
}

pub struct Migrator {
    source_root: PathBuf,
    target_root: PathBuf,
    execute: bool,
    workspace_target: Option<PathBuf>,
    overwrite: bool,
    migrate_secrets: bool,
    output_dir: Option<PathBuf>,
    archive_dir: Option<PathBuf>,
    backup_dir: Option<PathBuf>,
    overflow_dir: Option<PathBuf>,
    selected_options: BTreeSet<String>,
    preset_name: String,
    skill_conflict_mode: SkillConflict,
    items: Vec<MigrationItem>,
    config_apply_blocked: bool,
    memory_limit: usize,
    user_limit: usize,
    custom_workspace: Option<PathBuf>,
}

impl Migrator {
    pub fn new(
        context: &HermesContext,
        source_root: PathBuf,
        execute: bool,
        args: &MigrateArgs,
    ) -> Result<Self, Box<dyn Error>> {
        ensure_target_config_exists(context)?;
        let target_root = context.default_hermes_root();
        let output_dir = execute.then(|| {
            target_root
                .join("migration")
                .join("openclaw")
                .join(Local::now().format("%Y%m%dT%H%M%S").to_string())
        });
        let archive_dir = output_dir.as_ref().map(|path| path.join("archive"));
        let backup_dir = output_dir.as_ref().map(|path| path.join("backups"));
        let overflow_dir = output_dir.as_ref().map(|path| path.join("overflow"));
        let selected_options = selected_options(args.preset);
        let raw_config = load_openclaw_config(&source_root);
        let custom_workspace = configured_workspace(&source_root, &raw_config);
        let loaded = context.load_config_document()?;
        let mem_cfg = &loaded.config.memory;
        Ok(Self {
            source_root,
            target_root,
            execute,
            workspace_target: args.workspace_target.clone(),
            overwrite: args.overwrite,
            migrate_secrets: args.migrate_secrets,
            output_dir,
            archive_dir,
            backup_dir,
            overflow_dir,
            selected_options,
            preset_name: args.preset.as_str().to_string(),
            skill_conflict_mode: args.skill_conflict,
            items: Vec::new(),
            config_apply_blocked: false,
            memory_limit: usize::try_from(mem_cfg.memory_char_limit.max(1_000))
                .unwrap_or(DEFAULT_MEMORY_CHAR_LIMIT),
            user_limit: usize::try_from(mem_cfg.user_char_limit.max(500))
                .unwrap_or(DEFAULT_USER_CHAR_LIMIT),
            custom_workspace,
        })
    }

    pub fn migrate(mut self) -> Result<MigrationReport, Box<dyn Error>> {
        if !self.source_root.exists() {
            self.record(
                "source",
                Some(self.source_root.display().to_string()),
                None,
                "error",
                "OpenClaw directory does not exist",
                JsonValue::Null,
            );
            return Ok(self.build_report());
        }
        let config = load_openclaw_config(&self.source_root);

        self.run_if_selected("soul", |s| s.migrate_soul());
        self.run_if_selected("workspace-agents", |s| s.migrate_workspace_agents());
        self.run_if_selected("memory", |s| {
            let source =
                s.source_candidate(&["workspace/MEMORY.md", "workspace.default/MEMORY.md"]);
            s.migrate_memory(
                source.as_deref(),
                &s.target_root.join("memories").join("MEMORY.md"),
                s.memory_limit,
                "memory",
            )
        });
        self.run_if_selected("user-profile", |s| {
            let source = s.source_candidate(&["workspace/USER.md", "workspace.default/USER.md"]);
            s.migrate_memory(
                source.as_deref(),
                &s.target_root.join("memories").join("USER.md"),
                s.user_limit,
                "user-profile",
            )
        });
        self.run_if_selected("messaging-settings", |s| {
            s.migrate_messaging_settings(&config)
        });
        self.run_if_selected("secret-settings", |s| {
            s.migrate_secret_settings_gate(&config)
        });
        self.run_if_selected("discord-settings", |s| s.migrate_discord_settings(&config));
        self.run_if_selected("slack-settings", |s| s.migrate_slack_settings(&config));
        self.run_if_selected("whatsapp-settings", |s| {
            s.migrate_whatsapp_settings(&config)
        });
        self.run_if_selected("signal-settings", |s| s.migrate_signal_settings(&config));
        self.run_if_selected("provider-keys", |s| s.migrate_provider_keys_gate(&config));
        self.run_if_selected("model-config", |s| s.migrate_model_config(&config));
        self.run_if_selected("tts-config", |s| s.migrate_tts_config(&config));
        self.run_if_selected("command-allowlist", |s| s.migrate_command_allowlist());
        self.run_if_selected("skills", |s| s.migrate_skills());
        self.run_if_selected("shared-skills", |s| s.migrate_shared_skills());
        self.run_if_selected("daily-memory", |s| s.migrate_daily_memory());
        self.run_if_selected("tts-assets", |s| {
            let source = s.source_candidate(&["workspace/tts"]);
            s.copy_tree_non_destructive(
                source.as_deref(),
                &s.target_root.join("tts"),
                "tts-assets",
                &[".venv", "generated", "__pycache__"],
            )
        });
        self.run_if_selected("archive", |s| s.archive_docs());
        self.run_if_selected("mcp-servers", |s| s.migrate_mcp_servers(&config));
        self.run_if_selected("plugins-config", |s| {
            s.archive_json_section(
                "plugins-config",
                "openclaw.json plugins.*",
                config.get("plugins"),
                "plugins-config.json",
                "Plugins config archived for manual review",
            )
        });
        self.run_if_selected("cron-jobs", |s| s.migrate_cron_jobs(&config));
        self.run_if_selected("hooks-config", |s| s.migrate_hooks_config(&config));
        self.run_if_selected("agent-config", |s| s.migrate_agent_config(&config));
        self.run_if_selected("gateway-config", |s| s.migrate_gateway_config(&config));
        self.run_if_selected("session-config", |s| s.migrate_session_config(&config));
        self.run_if_selected("full-providers", |s| s.migrate_full_providers(&config));
        self.run_if_selected("deep-channels", |s| s.migrate_deep_channels(&config));
        self.run_if_selected("browser-config", |s| s.migrate_browser_config(&config));
        self.run_if_selected("tools-config", |s| s.migrate_tools_config(&config));
        self.run_if_selected("approvals-config", |s| s.migrate_approvals_config(&config));
        self.run_if_selected("memory-backend", |s| {
            s.archive_json_section(
                "memory-backend",
                "openclaw.json memory.*",
                config.get("memory"),
                "memory-backend-config.json",
                "Memory backend config archived for manual review",
            )
        });
        self.run_if_selected("skills-config", |s| {
            s.archive_json_section(
                "skills-config",
                "openclaw.json skills.*",
                config.get("skills"),
                "skills-registry-config.json",
                "Skills registry config archived",
            )
        });
        self.run_if_selected("ui-identity", |s| {
            s.archive_json_section(
                "ui-identity",
                "openclaw.json ui.*",
                config.get("ui"),
                "ui-identity-config.json",
                "UI theme and identity settings archived",
            )
        });
        self.run_if_selected("logging-config", |s| s.migrate_logging_config(&config));

        if let Some(out) = self.output_dir.as_ref() {
            fs::create_dir_all(out)?;
        }
        self.generate_migration_notes()?;
        Ok(self.build_report())
    }

    fn run_if_selected<F>(&mut self, option_id: &str, action: F)
    where
        F: FnOnce(&mut Self),
    {
        if !self.selected_options.contains(option_id) {
            self.record(
                option_id,
                None,
                None,
                "skipped",
                "Not selected for this run",
                JsonValue::Null,
            );
            return;
        }
        if self.execute && self.config_apply_blocked && CONFIG_MUTATING_OPTIONS.contains(&option_id)
        {
            self.record(
                option_id,
                None,
                None,
                "skipped",
                REASON_BLOCKED_BY_APPLY_CONFLICT,
                JsonValue::Null,
            );
            return;
        }
        action(self);
    }

    fn build_report(self) -> MigrationReport {
        let mut summary = BTreeMap::from([
            ("migrated".to_string(), 0usize),
            ("archived".to_string(), 0usize),
            ("skipped".to_string(), 0usize),
            ("conflict".to_string(), 0usize),
            ("error".to_string(), 0usize),
        ]);
        for item in &self.items {
            if let Some(value) = summary.get_mut(&item.status) {
                *value += 1;
            }
        }
        let warnings = build_warnings(
            &self.items,
            self.execute,
            self.migrate_secrets,
            self.config_apply_blocked,
        );
        let next_steps = build_next_steps(self.execute, &summary, self.output_dir.as_deref());
        if let Some(output_dir) = self.output_dir.as_ref() {
            let _ = write_report(
                output_dir,
                &self.source_root,
                &self.target_root,
                &self.preset_name,
                self.execute,
                &summary,
                &self.items,
                &warnings,
                &next_steps,
            );
        }
        MigrationReport {
            summary,
            items: self.items,
            output_dir: self.output_dir,
            preset: self.preset_name,
            warnings,
            next_steps,
        }
    }

    fn record(
        &mut self,
        kind: &str,
        source: Option<String>,
        destination: Option<String>,
        status: &str,
        reason: &str,
        details: JsonValue,
    ) {
        if matches!(status, "conflict" | "error")
            && destination.as_deref().is_some_and(|value| {
                value.ends_with("config.yaml") || value.contains("config.yaml ")
            })
        {
            self.config_apply_blocked = true;
        }
        self.items.push(MigrationItem {
            kind: kind.to_string(),
            source,
            destination,
            status: status.to_string(),
            reason: reason.to_string(),
            details,
            sensitive: false,
        });
    }

    fn source_candidate(&self, relatives: &[&str]) -> Option<PathBuf> {
        for rel in relatives {
            let candidate = self.source_root.join(rel);
            if candidate.exists() {
                return Some(candidate);
            }
            if let Some(rest) = rel.strip_prefix("workspace/") {
                for variant in ["workspace-main", "workspace-assistant"] {
                    let alt = self.source_root.join(variant).join(rest);
                    if alt.exists() {
                        return Some(alt);
                    }
                }
            } else if let Some(rest) = rel.strip_prefix("workspace.default/") {
                let alt = self.source_root.join("workspace-main").join(rest);
                if alt.exists() {
                    return Some(alt);
                }
            }
        }
        if let Some(workspace) = self.custom_workspace.as_ref() {
            for rel in relatives {
                let suffix = rel
                    .strip_prefix("workspace/")
                    .or_else(|| rel.strip_prefix("workspace.default/"));
                if let Some(suffix) = suffix {
                    let alt = workspace.join(suffix);
                    if alt.exists() {
                        return Some(alt);
                    }
                }
            }
        }
        None
    }

    fn maybe_backup(&self, path: &Path) -> Result<Option<PathBuf>, Box<dyn Error>> {
        if !self.execute || !path.exists() {
            return Ok(None);
        }
        let Some(root) = self.backup_dir.as_ref() else {
            return Ok(None);
        };
        let rel = if path.is_absolute() {
            path.strip_prefix(Path::new("/")).unwrap_or(path)
        } else {
            path
        };
        let dest = root.join(rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        if path.is_dir() {
            copy_dir_recursive(path, &dest)?;
        } else {
            fs::copy(path, &dest)?;
        }
        Ok(Some(dest))
    }

    fn write_overflow_entries(
        &self,
        kind: &str,
        entries: &[String],
    ) -> Result<Option<PathBuf>, Box<dyn Error>> {
        if entries.is_empty() {
            return Ok(None);
        }
        let Some(root) = self.overflow_dir.as_ref() else {
            return Ok(None);
        };
        fs::create_dir_all(root)?;
        let path = root.join(format!("{}_overflow.txt", kind.replace('-', "_")));
        fs::write(&path, format!("{}\n", entries.join("\n")))?;
        Ok(Some(path))
    }

    fn copy_file(
        &mut self,
        source: &Path,
        destination: &Path,
        kind: &str,
        transform: Option<fn(&str) -> String>,
    ) {
        if !source.exists() {
            return;
        }
        if destination.exists() && !self.overwrite {
            self.record(
                kind,
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "conflict",
                REASON_TARGET_EXISTS,
                JsonValue::Null,
            );
            return;
        }
        if self.execute {
            if let Ok(_backup) = self.maybe_backup(destination) {
                let _ = ensure_parent(destination);
                if let Some(transform) = transform {
                    if let Ok(raw) = fs::read_to_string(source) {
                        let _ = fs::write(destination, transform(&raw));
                    }
                } else {
                    let _ = fs::copy(source, destination);
                }
            }
        }
        self.record(
            kind,
            Some(source.display().to_string()),
            Some(destination.display().to_string()),
            "migrated",
            if self.execute { "" } else { "Would copy" },
            JsonValue::Null,
        );
    }

    fn migrate_soul(&mut self) {
        let Some(source) =
            self.source_candidate(&["workspace/SOUL.md", "workspace.default/SOUL.md"])
        else {
            self.record(
                "soul",
                None,
                Some(self.target_root.join("SOUL.md").display().to_string()),
                "skipped",
                "No OpenClaw SOUL.md found",
                JsonValue::Null,
            );
            return;
        };
        let destination = self.target_root.join("SOUL.md");
        self.copy_file(&source, &destination, "soul", Some(rebrand_text));
    }

    fn migrate_workspace_agents(&mut self) {
        let Some(source) =
            self.source_candidate(&["workspace/AGENTS.md", "workspace.default/AGENTS.md"])
        else {
            self.record(
                "workspace-agents",
                Some("workspace/AGENTS.md".to_string()),
                None,
                "skipped",
                "Source file not found",
                JsonValue::Null,
            );
            return;
        };
        let Some(workspace_target) = self.workspace_target.as_ref() else {
            self.record(
                "workspace-agents",
                Some(source.display().to_string()),
                None,
                "skipped",
                "No workspace target was provided",
                JsonValue::Null,
            );
            return;
        };
        let destination = workspace_target.join(WORKSPACE_INSTRUCTIONS_FILENAME);
        self.copy_file(
            &source,
            &destination,
            "workspace-agents",
            Some(rebrand_text),
        );
    }

    fn migrate_memory(
        &mut self,
        source: Option<&Path>,
        destination: &Path,
        limit: usize,
        kind: &str,
    ) {
        let Some(source) = source.filter(|path| path.exists()) else {
            self.record(
                kind,
                None,
                Some(destination.display().to_string()),
                "skipped",
                "Source file not found",
                JsonValue::Null,
            );
            return;
        };
        let incoming = extract_markdown_entries(&fs::read_to_string(source).unwrap_or_default())
            .into_iter()
            .map(|entry| rebrand_text(&entry))
            .collect::<Vec<_>>();
        if incoming.is_empty() {
            self.record(
                kind,
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "skipped",
                "No importable entries found",
                JsonValue::Null,
            );
            return;
        }
        let existing = parse_existing_memory_entries(destination);
        let (merged, stats, overflowed) = merge_entries(&existing, &incoming, limit);
        let overflow_file = self
            .write_overflow_entries(kind, &overflowed)
            .ok()
            .flatten();
        let details = json!({
            "existing_entries": stats.existing,
            "added_entries": stats.added,
            "duplicate_entries": stats.duplicates,
            "overflowed_entries": stats.overflowed,
            "char_limit": limit,
            "final_char_count": if merged.is_empty() { 0 } else { merged.join(ENTRY_DELIMITER).len() },
            "overflow_file": overflow_file.as_ref().map(|path| path.display().to_string()),
        });
        if self.execute {
            if stats.added == 0 && overflowed.is_empty() {
                self.record(
                    kind,
                    Some(source.display().to_string()),
                    Some(destination.display().to_string()),
                    "skipped",
                    "No new entries to import",
                    details,
                );
                return;
            }
            let _ = self.maybe_backup(destination);
            let _ = ensure_parent(destination);
            let _ = fs::write(destination, format!("{}\n", merged.join(ENTRY_DELIMITER)));
            self.record(
                kind,
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "migrated",
                "",
                details,
            );
        } else {
            self.record(
                kind,
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "migrated",
                "Would merge entries",
                details,
            );
        }
    }

    fn migrate_command_allowlist(&mut self) {
        let source = self.source_root.join("exec-approvals.json");
        let destination = self.target_root.join("config.yaml");
        if !source.exists() {
            self.record(
                "command-allowlist",
                None,
                Some(destination.display().to_string()),
                "skipped",
                "No OpenClaw exec approvals file found",
                JsonValue::Null,
            );
            return;
        }
        let Ok(raw) = fs::read_to_string(&source) else {
            self.record(
                "command-allowlist",
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "error",
                "Could not read exec approvals file",
                JsonValue::Null,
            );
            return;
        };
        let Ok(data) = serde_json::from_str::<JsonValue>(&raw) else {
            self.record(
                "command-allowlist",
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "error",
                "Invalid JSON",
                JsonValue::Null,
            );
            return;
        };
        let mut patterns = BTreeSet::new();
        if let Some(agents) = data.get("agents").and_then(JsonValue::as_object) {
            for agent_data in agents.values() {
                if let Some(allowlist) = agent_data.get("allowlist").and_then(JsonValue::as_array) {
                    for entry in allowlist {
                        if let Some(pattern) = entry.get("pattern").and_then(JsonValue::as_str) {
                            let trimmed = pattern.trim();
                            if !trimmed.is_empty() {
                                patterns.insert(trimmed.to_string());
                            }
                        }
                    }
                }
            }
        }
        if patterns.is_empty() {
            self.record(
                "command-allowlist",
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "skipped",
                "No allowlist patterns found",
                JsonValue::Null,
            );
            return;
        }
        let mut config = load_yaml_file(&destination);
        let current = yaml_sequence_strings(config.get(yaml_key("command_allowlist")));
        let merged = current
            .iter()
            .cloned()
            .chain(patterns.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let added = merged
            .iter()
            .filter(|pattern| !current.contains(pattern))
            .cloned()
            .collect::<Vec<_>>();
        if added.is_empty() {
            self.record(
                "command-allowlist",
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "skipped",
                "All patterns already present",
                JsonValue::Null,
            );
            return;
        }
        if self.execute {
            let _ = self.maybe_backup(&destination);
            config.as_mapping_mut().map(|mapping| {
                mapping.insert(
                    yaml_key("command_allowlist"),
                    YamlValue::Sequence(
                        merged
                            .iter()
                            .map(|value| YamlValue::String(value.clone()))
                            .collect(),
                    ),
                );
            });
            let _ = dump_yaml_file(&destination, &config);
            self.record(
                "command-allowlist",
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "migrated",
                "",
                json!({ "added_patterns": added }),
            );
        } else {
            self.record(
                "command-allowlist",
                Some(source.display().to_string()),
                Some(destination.display().to_string()),
                "migrated",
                "Would merge patterns",
                json!({ "added_patterns": added }),
            );
        }
    }

    fn merge_env_values(
        &mut self,
        additions: &BTreeMap<String, String>,
        kind: &str,
        source_label: &str,
    ) {
        let destination = self.target_root.join(".env");
        let mut env_data = parse_env_file(&destination);
        let mut added = Vec::new();
        let mut conflicts = Vec::new();
        for (key, value) in additions {
            match env_data.get(key) {
                Some(current) if current == value => {}
                Some(_) if !self.overwrite => conflicts.push(key.clone()),
                _ => {
                    env_data.insert(key.clone(), value.clone());
                    added.push(key.clone());
                }
            }
        }
        if conflicts.is_empty() && added.is_empty() {
            self.record(
                kind,
                Some(source_label.to_string()),
                Some(destination.display().to_string()),
                "skipped",
                "All env values already present",
                JsonValue::Null,
            );
            return;
        }
        if !conflicts.is_empty() && added.is_empty() {
            self.record(
                kind,
                Some(source_label.to_string()),
                Some(destination.display().to_string()),
                "conflict",
                "Destination .env already has different values",
                json!({ "conflicting_keys": conflicts }),
            );
            return;
        }
        if self.execute {
            let _ = self.maybe_backup(&destination);
            let _ = save_env_file(&destination, &env_data);
            self.record(
                kind,
                Some(source_label.to_string()),
                Some(destination.display().to_string()),
                "migrated",
                "",
                json!({ "added_keys": added, "conflicting_keys": conflicts }),
            );
        } else {
            self.record(
                kind,
                Some(source_label.to_string()),
                Some(destination.display().to_string()),
                "migrated",
                "Would merge env values",
                json!({ "added_keys": added, "conflicting_keys": conflicts }),
            );
        }
    }

    fn migrate_messaging_settings(&mut self, config: &JsonValue) {
        let mut additions = BTreeMap::new();
        if let Some(workspace) = config
            .pointer("/agents/defaults/workspace")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let path = PathBuf::from(workspace);
            let inside_source = path
                .canonicalize()
                .ok()
                .and_then(|resolved| resolved.strip_prefix(&self.source_root).ok().map(|_| true))
                .unwrap_or(false);
            if !inside_source {
                additions.insert("MESSAGING_CWD".to_string(), workspace.to_string());
            }
        }
        let allowlist_path = self
            .source_root
            .join("credentials")
            .join("telegram-default-allowFrom.json");
        if let Ok(raw) = fs::read_to_string(&allowlist_path)
            && let Ok(value) = serde_json::from_str::<JsonValue>(&raw)
            && let Some(items) = value.get("allowFrom").and_then(JsonValue::as_array)
        {
            let users = items
                .iter()
                .filter_map(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>();
            if !users.is_empty() {
                additions.insert("TELEGRAM_ALLOWED_USERS".to_string(), users.join(","));
            }
        }
        if additions.is_empty() {
            self.record(
                "messaging-settings",
                Some(self.source_root.join("openclaw.json").display().to_string()),
                Some(self.target_root.join(".env").display().to_string()),
                "skipped",
                "No Hermes-compatible messaging settings found",
                JsonValue::Null,
            );
            return;
        }
        self.merge_env_values(
            &additions,
            "messaging-settings",
            &self.source_root.join("openclaw.json").display().to_string(),
        );
    }

    fn migrate_secret_settings_gate(&mut self, config: &JsonValue) {
        if self.migrate_secrets {
            self.migrate_secret_settings(config);
            return;
        }
        self.record("secret-settings", Some(self.source_root.join("openclaw.json").display().to_string()), Some(self.target_root.join(".env").display().to_string()), "skipped", "Secret migration disabled. Re-run with --migrate-secrets to import allowlisted secrets.", json!({ "supported_targets": SUPPORTED_SECRET_TARGETS }));
    }

    fn migrate_secret_settings(&mut self, config: &JsonValue) {
        let mut additions = BTreeMap::new();
        if let Some(token) = channel_field(config, "telegram", "botToken") {
            additions.insert("TELEGRAM_BOT_TOKEN".to_string(), token);
        }
        if additions.is_empty() {
            self.record(
                "secret-settings",
                Some(self.source_root.join("openclaw.json").display().to_string()),
                Some(self.target_root.join(".env").display().to_string()),
                "skipped",
                "No allowlisted Hermes-compatible secrets found",
                json!({ "supported_targets": SUPPORTED_SECRET_TARGETS }),
            );
            return;
        }
        self.merge_env_values(
            &additions,
            "secret-settings",
            &self.source_root.join("openclaw.json").display().to_string(),
        );
    }

    fn migrate_discord_settings(&mut self, config: &JsonValue) {
        let mut additions = BTreeMap::new();
        if let Some(token) = channel_field(config, "discord", "token") {
            additions.insert("DISCORD_BOT_TOKEN".to_string(), token);
        }
        if let Some(users) = channel_list_field(config, "discord", "allowFrom") {
            additions.insert("DISCORD_ALLOWED_USERS".to_string(), users.join(","));
        }
        self.finish_channel_settings("discord-settings", &additions);
    }

    fn migrate_slack_settings(&mut self, config: &JsonValue) {
        let mut additions = BTreeMap::new();
        if let Some(token) = channel_field(config, "slack", "botToken") {
            additions.insert("SLACK_BOT_TOKEN".to_string(), token);
        }
        if let Some(token) = channel_field(config, "slack", "appToken") {
            additions.insert("SLACK_APP_TOKEN".to_string(), token);
        }
        if let Some(users) = channel_list_field(config, "slack", "allowFrom") {
            additions.insert("SLACK_ALLOWED_USERS".to_string(), users.join(","));
        }
        self.finish_channel_settings("slack-settings", &additions);
    }

    fn migrate_whatsapp_settings(&mut self, config: &JsonValue) {
        let mut additions = BTreeMap::new();
        if let Some(users) = channel_list_field(config, "whatsapp", "allowFrom") {
            additions.insert("WHATSAPP_ALLOWED_USERS".to_string(), users.join(","));
        }
        self.finish_channel_settings("whatsapp-settings", &additions);
    }

    fn migrate_signal_settings(&mut self, config: &JsonValue) {
        let mut additions = BTreeMap::new();
        if let Some(account) = channel_field(config, "signal", "account") {
            additions.insert("SIGNAL_ACCOUNT".to_string(), account);
        }
        if let Some(url) = channel_field(config, "signal", "httpUrl") {
            additions.insert("SIGNAL_HTTP_URL".to_string(), url);
        }
        if let Some(users) = channel_list_field(config, "signal", "allowFrom") {
            additions.insert("SIGNAL_ALLOWED_USERS".to_string(), users.join(","));
        }
        self.finish_channel_settings("signal-settings", &additions);
    }

    fn finish_channel_settings(&mut self, kind: &str, additions: &BTreeMap<String, String>) {
        if additions.is_empty() {
            self.record(
                kind,
                Some(self.source_root.join("openclaw.json").display().to_string()),
                Some(self.target_root.join(".env").display().to_string()),
                "skipped",
                &format!("No {} found", kind.replace("-settings", " settings")),
                JsonValue::Null,
            );
            return;
        }
        self.merge_env_values(
            additions,
            kind,
            &self.source_root.join("openclaw.json").display().to_string(),
        );
    }

    fn migrate_provider_keys_gate(&mut self, config: &JsonValue) {
        if !self.migrate_secrets {
            self.record("provider-keys", Some(self.source_root.join("openclaw.json").display().to_string()), Some(self.target_root.join(".env").display().to_string()), "skipped", "Secret migration disabled. Re-run with --migrate-secrets to import provider API keys.", json!({ "supported_targets": SUPPORTED_SECRET_TARGETS }));
            return;
        }
        self.migrate_provider_keys(config);
    }

    fn migrate_provider_keys(&mut self, config: &JsonValue) {
        let mut additions = BTreeMap::new();
        if let Some(providers) = config
            .pointer("/models/providers")
            .and_then(JsonValue::as_object)
        {
            for (name, provider) in providers {
                let api_key = provider.get("apiKey").and_then(|value| {
                    resolve_secret_input(value, &load_openclaw_env(&self.source_root))
                });
                let Some(api_key) = api_key else {
                    continue;
                };
                let base_url = provider
                    .get("baseUrl")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let api_type = provider
                    .get("api")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let env_key =
                    if base_url.contains("openrouter") || name.eq_ignore_ascii_case("openrouter") {
                        Some("OPENROUTER_API_KEY")
                    } else if base_url.contains("openai.com")
                        || name.to_ascii_lowercase().contains("openai")
                    {
                        Some("OPENAI_API_KEY")
                    } else if base_url.contains("anthropic")
                        || api_type == "anthropic-messages"
                        || name.to_ascii_lowercase().contains("anthropic")
                    {
                        Some("ANTHROPIC_API_KEY")
                    } else {
                        None
                    };
                if let Some(env_key) = env_key {
                    additions.insert(env_key.to_string(), api_key);
                }
            }
        }
        let env_map = BTreeMap::from([
            ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
            ("OPENAI_API_KEY", "OPENAI_API_KEY"),
            ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
            ("ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY"),
            ("TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN"),
            ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
            ("GEMINI_API_KEY", "GEMINI_API_KEY"),
            ("ZAI_API_KEY", "ZAI_API_KEY"),
            ("MINIMAX_API_KEY", "MINIMAX_API_KEY"),
        ]);
        let source_env = load_openclaw_env(&self.source_root);
        for (src, dst) in &env_map {
            if let Some(value) = source_env
                .get(*src)
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
            {
                additions
                    .entry((*dst).to_string())
                    .or_insert_with(|| value.to_string());
            }
        }
        if let Some(tts) = config.pointer("/messages/tts") {
            if let Some(key) = tts
                .pointer("/elevenlabs/apiKey")
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                additions.insert("ELEVENLABS_API_KEY".to_string(), key.to_string());
            }
            if let Some(key) = tts
                .pointer("/openai/apiKey")
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                additions.insert("VOICE_TOOLS_OPENAI_KEY".to_string(), key.to_string());
            }
        }
        if additions.is_empty() {
            self.record(
                "provider-keys",
                Some(self.source_root.join("openclaw.json").display().to_string()),
                Some(self.target_root.join(".env").display().to_string()),
                "skipped",
                "No provider API keys found",
                json!({ "supported_targets": SUPPORTED_SECRET_TARGETS }),
            );
            return;
        }
        self.merge_env_values(
            &additions,
            "provider-keys",
            &self.source_root.join("openclaw.json").display().to_string(),
        );
    }

    fn migrate_model_config(&mut self, config: &JsonValue) {
        let destination = self.target_root.join("config.yaml");
        let source_label = self.source_root.join("openclaw.json").display().to_string();
        let Some(mut model_str) = config
            .pointer("/agents/defaults/model")
            .and_then(extract_model_string)
        else {
            self.record(
                "model-config",
                Some(source_label),
                Some(destination.display().to_string()),
                "skipped",
                "No default model found in OpenClaw config",
                JsonValue::Null,
            );
            return;
        };
        if let Some(catalog) = config
            .pointer("/agents/defaults/models")
            .and_then(JsonValue::as_object)
            && !catalog.contains_key(&model_str)
        {
            for (api_id, entry) in catalog {
                if entry.get("alias").and_then(JsonValue::as_str) == Some(model_str.as_str())
                    || entry.as_str() == Some(model_str.as_str())
                {
                    model_str = api_id.clone();
                    break;
                }
            }
        }
        let mut yaml = load_yaml_file(&destination);
        let current = yaml
            .as_mapping()
            .and_then(|mapping| mapping.get(yaml_key("model")));
        if let Some(current) = current {
            if current.as_str() == Some(model_str.as_str())
                || current
                    .as_mapping()
                    .and_then(|mapping| mapping.get(yaml_key("default")))
                    .and_then(YamlValue::as_str)
                    == Some(model_str.as_str())
            {
                self.record(
                    "model-config",
                    Some(source_label),
                    Some(destination.display().to_string()),
                    "skipped",
                    "Model already set to the same value",
                    JsonValue::Null,
                );
                return;
            }
            if !self.overwrite {
                self.record(
                    "model-config",
                    Some(source_label),
                    Some(destination.display().to_string()),
                    "conflict",
                    "Model already set and overwrite is disabled",
                    json!({ "incoming": model_str }),
                );
                return;
            }
        }
        if self.execute {
            let _ = self.maybe_backup(&destination);
            let model_mapping = ensure_mapping_mut(&mut yaml, "model");
            model_mapping.insert(yaml_key("default"), YamlValue::String(model_str.clone()));
            let _ = dump_yaml_file(&destination, &yaml);
            self.record(
                "model-config",
                Some(source_label),
                Some(destination.display().to_string()),
                "migrated",
                "",
                json!({ "model": model_str }),
            );
        } else {
            self.record(
                "model-config",
                Some(source_label),
                Some(destination.display().to_string()),
                "migrated",
                "Would set model",
                json!({ "model": model_str }),
            );
        }
    }

    fn migrate_tts_config(&mut self, config: &JsonValue) {
        let destination = self.target_root.join("config.yaml");
        let source_label = self.source_root.join("openclaw.json").display().to_string();
        let tts = config
            .pointer("/messages/tts")
            .cloned()
            .unwrap_or(JsonValue::Null);
        let talk = config.get("talk").cloned().unwrap_or(JsonValue::Null);
        if !tts.is_object() {
            self.record(
                "tts-config",
                Some(source_label),
                Some(destination.display().to_string()),
                "skipped",
                "No TTS configuration found in OpenClaw config",
                JsonValue::Null,
            );
            return;
        }
        let mut tts_yaml = YamlMapping::new();
        if let Some(provider) = tts
            .get("provider")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| matches!(*value, "elevenlabs" | "openai" | "edge" | "microsoft"))
        {
            let provider = if provider == "microsoft" {
                "edge"
            } else {
                provider
            };
            tts_yaml.insert(
                yaml_key("provider"),
                YamlValue::String(provider.to_string()),
            );
        }
        let providers = tts.get("providers").cloned().unwrap_or(JsonValue::Null);
        let elevenlabs = providers
            .pointer("/elevenlabs")
            .cloned()
            .or_else(|| talk.pointer("/providers/elevenlabs").cloned())
            .or_else(|| tts.pointer("/elevenlabs").cloned())
            .unwrap_or(JsonValue::Null);
        if elevenlabs.is_object() {
            let mut mapping = YamlMapping::new();
            if let Some(voice) = elevenlabs
                .get("voiceId")
                .or_else(|| talk.get("voiceId"))
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                mapping.insert(yaml_key("voice_id"), YamlValue::String(voice.to_string()));
            }
            if let Some(model) = elevenlabs
                .get("modelId")
                .or_else(|| talk.get("modelId"))
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                mapping.insert(yaml_key("model_id"), YamlValue::String(model.to_string()));
            }
            if !mapping.is_empty() {
                tts_yaml.insert(yaml_key("elevenlabs"), YamlValue::Mapping(mapping));
            }
        }
        let openai_tts = providers
            .pointer("/openai")
            .cloned()
            .or_else(|| talk.pointer("/providers/openai").cloned())
            .or_else(|| tts.pointer("/openai").cloned())
            .unwrap_or(JsonValue::Null);
        if openai_tts.is_object() {
            let mut mapping = YamlMapping::new();
            if let Some(model) = openai_tts
                .get("model")
                .or_else(|| openai_tts.get("modelId"))
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                mapping.insert(yaml_key("model"), YamlValue::String(model.to_string()));
            }
            if let Some(voice) = openai_tts
                .get("voice")
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                mapping.insert(yaml_key("voice"), YamlValue::String(voice.to_string()));
            }
            if !mapping.is_empty() {
                tts_yaml.insert(yaml_key("openai"), YamlValue::Mapping(mapping));
            }
        }
        let edge_tts = providers
            .pointer("/edge")
            .cloned()
            .or_else(|| providers.pointer("/microsoft").cloned())
            .or_else(|| tts.pointer("/edge").cloned())
            .or_else(|| tts.pointer("/microsoft").cloned())
            .unwrap_or(JsonValue::Null);
        if let Some(voice) = edge_tts
            .get("voice")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let mut mapping = YamlMapping::new();
            mapping.insert(yaml_key("voice"), YamlValue::String(voice.to_string()));
            tts_yaml.insert(yaml_key("edge"), YamlValue::Mapping(mapping));
        }
        if tts_yaml.is_empty() {
            self.record(
                "tts-config",
                Some(source_label),
                Some(destination.display().to_string()),
                "skipped",
                "No compatible TTS settings found",
                JsonValue::Null,
            );
            return;
        }
        let mut yaml = load_yaml_file(&destination);
        if self.execute {
            let _ = self.maybe_backup(&destination);
            let existing = ensure_mapping_mut(&mut yaml, "tts");
            for (key, value) in tts_yaml {
                existing.insert(key, value);
            }
            let _ = dump_yaml_file(&destination, &yaml);
            self.record(
                "tts-config",
                Some(source_label),
                Some(destination.display().to_string()),
                "migrated",
                "",
                JsonValue::Null,
            );
        } else {
            self.record(
                "tts-config",
                Some(source_label),
                Some(destination.display().to_string()),
                "migrated",
                "Would set TTS config",
                JsonValue::Null,
            );
        }
    }

    fn migrate_shared_skills(&mut self) {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
        let sources = [
            (
                self.source_root.join("skills"),
                "shared-skills",
                "managed skills",
            ),
            (
                home.join(".agents").join("skills"),
                "personal-skills",
                "personal cross-project skills",
            ),
            (
                self.source_root
                    .join("workspace")
                    .join(".agents")
                    .join("skills"),
                "project-skills",
                "project-level shared skills",
            ),
            (
                self.source_root
                    .join("workspace.default")
                    .join(".agents")
                    .join("skills"),
                "project-skills",
                "project-level shared skills",
            ),
        ];
        let mut found = false;
        for (source_root, kind, desc) in sources {
            if source_root.exists() {
                found = true;
                self.import_skill_directory(&source_root, kind, desc);
            }
        }
        if !found {
            self.record(
                "shared-skills",
                None,
                Some(
                    self.target_root
                        .join("skills")
                        .join(SKILL_CATEGORY_DIRNAME)
                        .display()
                        .to_string(),
                ),
                "skipped",
                "No shared OpenClaw skills directories found",
                JsonValue::Null,
            );
        }
    }

    fn import_skill_directory(&mut self, source_root: &Path, kind: &str, description: &str) {
        let destination_root = self.target_root.join("skills").join(SKILL_CATEGORY_DIRNAME);
        let skill_dirs = fs::read_dir(source_root)
            .ok()
            .into_iter()
            .flat_map(|entries| entries.flatten())
            .map(|entry| entry.path())
            .filter(|path| path.is_dir() && path.join("SKILL.md").exists())
            .collect::<Vec<_>>();
        if skill_dirs.is_empty() {
            self.record(
                kind,
                Some(source_root.display().to_string()),
                Some(destination_root.display().to_string()),
                "skipped",
                &format!("No skills with SKILL.md found in {description}"),
                JsonValue::Null,
            );
            return;
        }
        for skill_dir in skill_dirs {
            self.copy_skill_dir(kind, &skill_dir, &destination_root);
        }
        if self.execute {
            let _ = fs::create_dir_all(&destination_root);
            let desc_path = destination_root.join("DESCRIPTION.md");
            if !desc_path.exists() {
                let _ = fs::write(desc_path, SKILL_CATEGORY_DESCRIPTION);
            }
        }
    }

    fn migrate_daily_memory(&mut self) {
        let Some(source_dir) = self.source_candidate(&["workspace/memory"]) else {
            self.record(
                "daily-memory",
                None,
                Some(
                    self.target_root
                        .join("memories")
                        .join("MEMORY.md")
                        .display()
                        .to_string(),
                ),
                "skipped",
                "No workspace/memory/ directory found",
                JsonValue::Null,
            );
            return;
        };
        let destination = self.target_root.join("memories").join("MEMORY.md");
        let files = fs::read_dir(&source_dir)
            .ok()
            .into_iter()
            .flat_map(|entries| entries.flatten())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("md")
            })
            .collect::<Vec<_>>();
        if files.is_empty() {
            self.record(
                "daily-memory",
                Some(source_dir.display().to_string()),
                Some(destination.display().to_string()),
                "skipped",
                "No .md files found in workspace/memory/",
                JsonValue::Null,
            );
            return;
        }
        let incoming = files
            .iter()
            .flat_map(|path| {
                extract_markdown_entries(&fs::read_to_string(path).unwrap_or_default())
            })
            .map(|entry| rebrand_text(&entry))
            .collect::<Vec<_>>();
        let existing = parse_existing_memory_entries(&destination);
        let (merged, stats, overflowed) = merge_entries(&existing, &incoming, self.memory_limit);
        let overflow_file = self
            .write_overflow_entries("daily-memory", &overflowed)
            .ok()
            .flatten();
        let details = json!({
            "source_files": files.len(),
            "existing_entries": stats.existing,
            "added_entries": stats.added,
            "duplicate_entries": stats.duplicates,
            "overflowed_entries": stats.overflowed,
            "overflow_file": overflow_file.as_ref().map(|path| path.display().to_string()),
        });
        if self.execute {
            let _ = self.maybe_backup(&destination);
            let _ = ensure_parent(&destination);
            let _ = fs::write(&destination, format!("{}\n", merged.join(ENTRY_DELIMITER)));
            self.record(
                "daily-memory",
                Some(source_dir.display().to_string()),
                Some(destination.display().to_string()),
                "migrated",
                "",
                details,
            );
        } else {
            self.record(
                "daily-memory",
                Some(source_dir.display().to_string()),
                Some(destination.display().to_string()),
                "migrated",
                "Would merge daily memory entries",
                details,
            );
        }
    }

    fn migrate_skills(&mut self) {
        let Some(source_root) = self.source_candidate(&["workspace/skills"]) else {
            self.record(
                "skills",
                None,
                Some(
                    self.target_root
                        .join("skills")
                        .join(SKILL_CATEGORY_DIRNAME)
                        .display()
                        .to_string(),
                ),
                "skipped",
                "No OpenClaw skills directory found",
                JsonValue::Null,
            );
            return;
        };
        let destination_root = self.target_root.join("skills").join(SKILL_CATEGORY_DIRNAME);
        let skill_dirs = fs::read_dir(&source_root)
            .ok()
            .into_iter()
            .flat_map(|entries| entries.flatten())
            .map(|entry| entry.path())
            .filter(|path| path.is_dir() && path.join("SKILL.md").exists())
            .collect::<Vec<_>>();
        if skill_dirs.is_empty() {
            self.record(
                "skills",
                Some(source_root.display().to_string()),
                Some(destination_root.display().to_string()),
                "skipped",
                "No skills with SKILL.md found",
                JsonValue::Null,
            );
            return;
        }
        for skill_dir in skill_dirs {
            self.copy_skill_dir("skill", &skill_dir, &destination_root);
        }
        if self.execute {
            let _ = fs::create_dir_all(&destination_root);
            let desc_path = destination_root.join("DESCRIPTION.md");
            if !desc_path.exists() {
                let _ = fs::write(desc_path, SKILL_CATEGORY_DESCRIPTION);
            }
        }
    }

    fn copy_skill_dir(&mut self, kind: &str, skill_dir: &Path, destination_root: &Path) {
        let destination = destination_root.join(skill_dir.file_name().unwrap_or_default());
        let final_destination =
            if destination.exists() && self.skill_conflict_mode == SkillConflict::Rename {
                resolve_skill_destination(&destination)
            } else {
                destination.clone()
            };
        if destination.exists() && self.skill_conflict_mode == SkillConflict::Skip {
            self.record(
                kind,
                Some(skill_dir.display().to_string()),
                Some(destination.display().to_string()),
                "conflict",
                "Destination skill already exists",
                JsonValue::Null,
            );
            return;
        }
        if self.execute {
            if final_destination == destination && destination.exists() {
                let _ = self.maybe_backup(&destination);
                let _ = fs::remove_dir_all(&destination);
            }
            let _ = ensure_parent(&final_destination);
            let _ = copy_dir_recursive(skill_dir, &final_destination);
        }
        let mut details = JsonMap::new();
        if final_destination != destination {
            details.insert(
                "renamed_from".to_string(),
                JsonValue::String(destination.display().to_string()),
            );
        }
        self.record(
            kind,
            Some(skill_dir.display().to_string()),
            Some(final_destination.display().to_string()),
            "migrated",
            if self.execute {
                ""
            } else {
                "Would copy skill directory"
            },
            JsonValue::Object(details),
        );
    }

    fn copy_tree_non_destructive(
        &mut self,
        source_root: Option<&Path>,
        destination_root: &Path,
        kind: &str,
        ignore_dir_names: &[&str],
    ) {
        let Some(source_root) = source_root.filter(|path| path.exists()) else {
            self.record(
                kind,
                None,
                Some(destination_root.display().to_string()),
                "skipped",
                "Source directory not found",
                JsonValue::Null,
            );
            return;
        };
        let mut copied = 0usize;
        let mut skipped = 0usize;
        let mut conflicts = 0usize;
        let files = collect_files(source_root);
        for source in files {
            let rel = match source.strip_prefix(source_root) {
                Ok(rel) => rel,
                Err(_) => continue,
            };
            if rel.components().any(|part| {
                matches!(
                    part,
                    Component::Normal(name)
                        if ignore_dir_names.iter().any(|ignore| name == std::ffi::OsStr::new(ignore))
                )
            }) {
                continue;
            }
            let destination = destination_root.join(rel);
            if destination.exists() {
                if file_contents_match(&source, &destination) {
                    skipped += 1;
                    continue;
                }
                if !self.overwrite {
                    conflicts += 1;
                    self.record(
                        kind,
                        Some(source.display().to_string()),
                        Some(destination.display().to_string()),
                        "conflict",
                        REASON_TARGET_EXISTS,
                        JsonValue::Null,
                    );
                    continue;
                }
            }
            if self.execute {
                let _ = self.maybe_backup(&destination);
                let _ = ensure_parent(&destination);
                let _ = fs::copy(&source, &destination);
            }
            copied += 1;
        }
        let status = if copied > 0 {
            "migrated"
        } else if conflicts > 0 {
            "conflict"
        } else {
            "skipped"
        };
        let reason = if copied == 0 && conflicts > 0 {
            "All candidate files conflicted with existing destination files"
        } else if copied == 0 {
            "No new files to copy"
        } else {
            ""
        };
        self.record(
            kind,
            Some(source_root.display().to_string()),
            Some(destination_root.display().to_string()),
            status,
            reason,
            json!({ "copied_files": copied, "unchanged_files": skipped, "conflicts": conflicts }),
        );
    }

    fn archive_docs(&mut self) {
        for candidate in [
            self.source_candidate(&["workspace/IDENTITY.md", "workspace.default/IDENTITY.md"]),
            self.source_candidate(&["workspace/TOOLS.md", "workspace.default/TOOLS.md"]),
            self.source_candidate(&["workspace/HEARTBEAT.md", "workspace.default/HEARTBEAT.md"]),
            self.source_candidate(&["workspace/BOOTSTRAP.md", "workspace.default/BOOTSTRAP.md"]),
        ]
        .into_iter()
        .flatten()
        {
            self.archive_path(
                &candidate,
                "No direct Hermes destination; archived for manual review",
            );
        }
        for rel in ["workspace/.learnings", "workspace/memory"] {
            let candidate = self.source_root.join(rel);
            if candidate.exists() {
                self.archive_path(
                    &candidate,
                    "No direct Hermes destination; archived for manual review",
                );
            }
        }
    }

    fn archive_path(&mut self, source: &Path, reason: &str) {
        let destination = self
            .archive_dir
            .as_ref()
            .map(|root| root.join(relative_label(source, &self.source_root)));
        if self.execute
            && let Some(destination) = destination.as_ref()
        {
            let _ = ensure_parent(destination);
            if source.is_dir() {
                let _ = copy_dir_recursive(source, destination);
            } else {
                let _ = fs::copy(source, destination);
            }
        }
        self.record(
            "archive",
            Some(source.display().to_string()),
            destination.as_ref().map(|path| path.display().to_string()),
            "archived",
            reason,
            JsonValue::Null,
        );
    }

    fn migrate_mcp_servers(&mut self, config: &JsonValue) {
        let Some(servers) = config
            .pointer("/mcp/servers")
            .and_then(JsonValue::as_object)
        else {
            self.record(
                "mcp-servers",
                None,
                None,
                "skipped",
                "No MCP servers found in OpenClaw config",
                JsonValue::Null,
            );
            return;
        };
        let destination = self.target_root.join("config.yaml");
        let mut yaml = load_yaml_file(&destination);
        let mcp_mapping = ensure_mapping_mut(&mut yaml, "mcp_servers");
        let mut added = 0usize;
        for (name, srv) in servers {
            if mcp_mapping.contains_key(yaml_key(name)) && !self.overwrite {
                self.record(
                    "mcp-servers",
                    Some(format!("mcp.servers.{name}")),
                    Some(format!("config.yaml mcp_servers.{name}")),
                    "conflict",
                    "MCP server already exists in Hermes config",
                    JsonValue::Null,
                );
                continue;
            }
            let mut mapping = YamlMapping::new();
            if let Some(command) = srv.get("command").and_then(JsonValue::as_str) {
                mapping.insert(yaml_key("command"), YamlValue::String(command.to_string()));
            }
            if let Some(args) = srv.get("args").and_then(JsonValue::as_array) {
                mapping.insert(
                    yaml_key("args"),
                    YamlValue::Sequence(
                        args.iter()
                            .filter_map(JsonValue::as_str)
                            .map(|value| YamlValue::String(value.to_string()))
                            .collect(),
                    ),
                );
            }
            if let Some(env) = srv.get("env").and_then(json_object_to_yaml_mapping) {
                mapping.insert(yaml_key("env"), YamlValue::Mapping(env));
            }
            if let Some(cwd) = srv.get("cwd").and_then(JsonValue::as_str) {
                mapping.insert(yaml_key("cwd"), YamlValue::String(cwd.to_string()));
            }
            if let Some(url) = srv.get("url").and_then(JsonValue::as_str) {
                mapping.insert(yaml_key("url"), YamlValue::String(url.to_string()));
            }
            if let Some(headers) = srv.get("headers").and_then(json_object_to_yaml_mapping) {
                mapping.insert(yaml_key("headers"), YamlValue::Mapping(headers));
            }
            if let Some(auth) = srv.get("auth").map(json_to_yaml) {
                mapping.insert(yaml_key("auth"), auth);
            }
            if srv.get("enabled").and_then(JsonValue::as_bool) == Some(false) {
                mapping.insert(yaml_key("enabled"), YamlValue::Bool(false));
            }
            if let Some(timeout) = srv.get("timeout").and_then(json_number_to_yaml) {
                mapping.insert(yaml_key("timeout"), timeout);
            }
            if let Some(timeout) = srv.get("connectTimeout").and_then(json_number_to_yaml) {
                mapping.insert(yaml_key("connect_timeout"), timeout);
            }
            if let Some(tools) = srv.get("tools").and_then(JsonValue::as_object) {
                let mut tools_mapping = YamlMapping::new();
                if let Some(include) = tools.get("include").map(json_to_yaml) {
                    tools_mapping.insert(yaml_key("include"), include);
                }
                if let Some(exclude) = tools.get("exclude").map(json_to_yaml) {
                    tools_mapping.insert(yaml_key("exclude"), exclude);
                }
                if !tools_mapping.is_empty() {
                    mapping.insert(yaml_key("tools"), YamlValue::Mapping(tools_mapping));
                }
            }
            mcp_mapping.insert(yaml_key(name), YamlValue::Mapping(mapping));
            added += 1;
            self.record(
                "mcp-servers",
                Some(format!("mcp.servers.{name}")),
                Some(format!("config.yaml mcp_servers.{name}")),
                "migrated",
                "",
                JsonValue::Null,
            );
        }
        if added > 0 && self.execute {
            let _ = self.maybe_backup(&destination);
            let _ = dump_yaml_file(&destination, &yaml);
        }
    }

    fn migrate_cron_jobs(&mut self, config: &JsonValue) {
        let cron = config.get("cron");
        let cron_store = self.source_root.join("cron");
        let mut found = false;
        if cron.is_some() && cron != Some(&JsonValue::Null) {
            found = true;
            self.archive_json_section(
                "cron-jobs",
                "openclaw.json cron.*",
                cron,
                "cron-config.json",
                "Cron config archived. Use 'hermes cron' to recreate jobs manually.",
            );
        }
        if cron_store.is_dir() {
            found = true;
            let destination = self
                .archive_dir
                .as_ref()
                .map(|root| root.join("cron-store"));
            if self.execute
                && let Some(destination) = destination.as_ref()
            {
                let _ = copy_dir_recursive(&cron_store, destination);
            }
            self.record(
                "cron-jobs",
                Some(cron_store.display().to_string()),
                destination.as_ref().map(|path| path.display().to_string()),
                "archived",
                "Cron job store archived",
                JsonValue::Null,
            );
        }
        if !found {
            self.record(
                "cron-jobs",
                None,
                None,
                "skipped",
                "No cron configuration found",
                JsonValue::Null,
            );
        }
    }

    fn migrate_hooks_config(&mut self, config: &JsonValue) {
        let hooks = config.get("hooks");
        if hooks.is_none() || hooks == Some(&JsonValue::Null) {
            self.record(
                "hooks-config",
                None,
                None,
                "skipped",
                "No hooks configuration found",
                JsonValue::Null,
            );
            return;
        }
        self.archive_json_section(
            "hooks-config",
            "openclaw.json hooks.*",
            hooks,
            "hooks-config.json",
            "Hooks config archived for manual review",
        );
        for ws_name in ["workspace", "workspace.default"] {
            let hooks_dir = self.source_root.join(ws_name).join("hooks");
            if hooks_dir.is_dir() {
                let destination = self
                    .archive_dir
                    .as_ref()
                    .map(|root| root.join("workspace-hooks"));
                if self.execute
                    && let Some(destination) = destination.as_ref()
                {
                    let _ = copy_dir_recursive(&hooks_dir, destination);
                }
                self.record(
                    "hooks-config",
                    Some(hooks_dir.display().to_string()),
                    destination.as_ref().map(|path| path.display().to_string()),
                    "archived",
                    "Workspace hooks directory archived",
                    JsonValue::Null,
                );
                break;
            }
        }
    }

    fn migrate_agent_config(&mut self, config: &JsonValue) {
        let defaults = config
            .pointer("/agents/defaults")
            .cloned()
            .unwrap_or(JsonValue::Null);
        let agent_list = config
            .pointer("/agents/list")
            .and_then(JsonValue::as_array)
            .cloned()
            .unwrap_or_default();
        if !defaults.is_object() && agent_list.is_empty() {
            self.record(
                "agent-config",
                None,
                None,
                "skipped",
                "No agent configuration found",
                JsonValue::Null,
            );
            return;
        }
        let destination = self.target_root.join("config.yaml");
        let mut yaml = load_yaml_file(&destination);
        let mut changed = false;
        if let Some(timeout) = defaults.get("timeoutSeconds").and_then(JsonValue::as_i64) {
            let agent = ensure_mapping_mut(&mut yaml, "agent");
            agent.insert(
                yaml_key("max_turns"),
                YamlValue::Number(serde_yaml::Number::from((timeout / 10).clamp(1, 200))),
            );
            changed = true;
        }
        if let Some(verbose) = defaults.get("verboseDefault").and_then(JsonValue::as_bool) {
            let agent = ensure_mapping_mut(&mut yaml, "agent");
            agent.insert(yaml_key("verbose"), YamlValue::Bool(verbose));
            changed = true;
        }
        if let Some(thinking) = defaults.get("thinkingDefault").and_then(JsonValue::as_str) {
            let effort = match thinking {
                "always" | "high" | "xhigh" => "high",
                "auto" | "medium" | "adaptive" => "medium",
                "off" | "low" | "none" | "minimal" => "low",
                _ => "medium",
            };
            let agent = ensure_mapping_mut(&mut yaml, "agent");
            agent.insert(
                yaml_key("reasoning_effort"),
                YamlValue::String(effort.to_string()),
            );
            changed = true;
        }
        if let Some(user_timezone) = defaults
            .get("userTimezone")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            if let Some(root) = yaml.as_mapping_mut() {
                root.insert(
                    yaml_key("timezone"),
                    YamlValue::String(user_timezone.to_string()),
                );
                changed = true;
            }
        }
        if let Some(timeout) = config
            .pointer("/tools/exec/timeoutSec")
            .or_else(|| config.pointer("/tools/exec/timeout"))
            .and_then(json_number_to_yaml)
        {
            let terminal = ensure_mapping_mut(&mut yaml, "terminal");
            terminal.insert(yaml_key("timeout"), timeout);
            changed = true;
        }
        if defaults
            .pointer("/sandbox/backend")
            .and_then(JsonValue::as_str)
            == Some("docker")
        {
            let terminal = ensure_mapping_mut(&mut yaml, "terminal");
            terminal.insert(yaml_key("backend"), YamlValue::String("docker".to_string()));
            if let Some(image) = defaults
                .pointer("/sandbox/docker/image")
                .and_then(JsonValue::as_str)
            {
                terminal.insert(
                    yaml_key("docker_image"),
                    YamlValue::String(image.to_string()),
                );
            }
            changed = true;
        }
        if changed && self.execute {
            let _ = self.maybe_backup(&destination);
            let _ = dump_yaml_file(&destination, &yaml);
        }
        if changed {
            self.record(
                "agent-config",
                Some("openclaw.json agents.defaults".to_string()),
                Some("config.yaml agent/compression/terminal".to_string()),
                "migrated",
                "Agent defaults mapped to Hermes config",
                JsonValue::Null,
            );
        }
        if !agent_list.is_empty() {
            self.archive_value(
                "agent-config",
                "openclaw.json agents.list",
                "agents-list.json",
                &JsonValue::Array(agent_list.clone()),
                &format!(
                    "Multi-agent setup ({}) archived for manual recreation",
                    agent_list.len()
                ),
            );
        }
        if let Some(bindings) = config.get("bindings")
            && bindings.as_array().is_some_and(|items| !items.is_empty())
        {
            self.archive_value(
                "agent-config",
                "openclaw.json bindings",
                "bindings.json",
                bindings,
                "Agent routing bindings archived",
            );
        }
    }

    fn migrate_gateway_config(&mut self, config: &JsonValue) {
        let gateway = config.get("gateway");
        if gateway.is_none() || gateway == Some(&JsonValue::Null) {
            self.record(
                "gateway-config",
                None,
                None,
                "skipped",
                "No gateway configuration found",
                JsonValue::Null,
            );
            return;
        }
        self.archive_json_section(
            "gateway-config",
            "openclaw.json gateway.*",
            gateway,
            "gateway-config.json",
            "Gateway config archived. Use 'hermes gateway' to configure.",
        );
        if self.migrate_secrets
            && let Some(token) = gateway
                .and_then(|value| value.pointer("/auth/token"))
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
        {
            let mut additions = BTreeMap::new();
            additions.insert("HERMES_GATEWAY_TOKEN".to_string(), token.to_string());
            self.merge_env_values(&additions, "env-var", "gateway.auth.token");
        }
    }

    fn migrate_session_config(&mut self, config: &JsonValue) {
        let session = config.get("session");
        if session.is_none() || session == Some(&JsonValue::Null) {
            self.record(
                "session-config",
                None,
                None,
                "skipped",
                "No session configuration found",
                JsonValue::Null,
            );
            return;
        }
        let destination = self.target_root.join("config.yaml");
        let mut yaml = load_yaml_file(&destination);
        let mut changed = false;
        let sr = ensure_mapping_mut(&mut yaml, "session_reset");
        if let Some(reset) = session
            .and_then(|value| value.get("reset"))
            .and_then(JsonValue::as_object)
        {
            if let Some(mode) = reset.get("mode").and_then(JsonValue::as_str) {
                sr.insert(yaml_key("mode"), YamlValue::String(mode.to_string()));
                changed = true;
            }
            if let Some(hour) = reset.get("atHour").and_then(json_number_to_yaml) {
                sr.insert(yaml_key("at_hour"), hour);
                changed = true;
            }
            if let Some(idle) = reset.get("idleMinutes").and_then(json_number_to_yaml) {
                sr.insert(yaml_key("idle_minutes"), idle);
                changed = true;
            }
        } else if let Some(triggers) = session
            .and_then(|value| {
                value
                    .get("resetTriggers")
                    .or_else(|| value.get("reset_triggers"))
            })
            .and_then(JsonValue::as_array)
        {
            let has_daily = triggers.iter().any(|value| value.as_str() == Some("daily"));
            let has_idle = triggers.iter().any(|value| value.as_str() == Some("idle"));
            let mode = if has_daily && has_idle {
                Some("both")
            } else if has_daily {
                Some("daily")
            } else if has_idle {
                Some("idle")
            } else {
                None
            };
            if let Some(mode) = mode {
                sr.insert(yaml_key("mode"), YamlValue::String(mode.to_string()));
                changed = true;
            }
        }
        if changed && self.execute {
            let _ = self.maybe_backup(&destination);
            let _ = dump_yaml_file(&destination, &yaml);
        }
        if changed {
            self.record(
                "session-config",
                Some("openclaw.json session.resetTriggers".to_string()),
                Some("config.yaml session_reset".to_string()),
                "migrated",
                "",
                JsonValue::Null,
            );
        }
        if let Some(session_value) = session
            && let Some(object) = session_value.as_object()
        {
            let complex = object
                .iter()
                .filter(|(key, value)| {
                    [
                        "identityLinks",
                        "threadBindings",
                        "maintenance",
                        "scope",
                        "sendPolicy",
                    ]
                    .contains(&key.as_str())
                        && !value.is_null()
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<JsonMap<_, _>>();
            if !complex.is_empty() {
                self.archive_value(
                    "session-config",
                    "openclaw.json session (advanced)",
                    "session-config.json",
                    &JsonValue::Object(complex),
                    "Advanced session settings archived",
                );
            }
        }
    }

    fn migrate_full_providers(&mut self, config: &JsonValue) {
        let Some(providers) = config
            .pointer("/models/providers")
            .and_then(JsonValue::as_object)
        else {
            self.record(
                "full-providers",
                None,
                None,
                "skipped",
                "No model providers found",
                JsonValue::Null,
            );
            return;
        };
        let destination = self.target_root.join("config.yaml");
        let mut yaml = load_yaml_file(&destination);
        let custom_providers = ensure_sequence_mut(&mut yaml, "custom_providers");
        let mut existing = custom_providers
            .iter()
            .filter_map(|value| value.as_mapping())
            .filter_map(|mapping| mapping.get(yaml_key("name")).and_then(YamlValue::as_str))
            .map(|value| value.to_ascii_lowercase())
            .collect::<HashSet<_>>();
        let well_known = [
            "openrouter",
            "openai",
            "anthropic",
            "deepseek",
            "google",
            "groq",
        ];
        let mut added = 0usize;
        for (name, provider) in providers {
            if self.migrate_secrets {
                if let Some(key) = provider
                    .get("apiKey")
                    .and_then(JsonValue::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    let mut additions = BTreeMap::new();
                    additions.insert(
                        format!("{}_API_KEY", name.to_ascii_uppercase().replace('-', "_")),
                        key.to_string(),
                    );
                    self.merge_env_values(
                        &additions,
                        "env-var",
                        &format!("models.providers.{name}.apiKey"),
                    );
                }
            }
            if well_known.contains(&name.to_ascii_lowercase().as_str())
                || provider
                    .get("baseUrl")
                    .and_then(JsonValue::as_str)
                    .is_none()
            {
                continue;
            }
            if existing.contains(&name.to_ascii_lowercase()) && !self.overwrite {
                self.record(
                    "full-providers",
                    Some(format!("models.providers.{name}")),
                    Some("config.yaml custom_providers".to_string()),
                    "conflict",
                    &format!("Provider '{name}' already exists"),
                    JsonValue::Null,
                );
                continue;
            }
            let mut mapping = YamlMapping::new();
            mapping.insert(yaml_key("name"), YamlValue::String(name.clone()));
            mapping.insert(
                yaml_key("base_url"),
                YamlValue::String(
                    provider
                        .get("baseUrl")
                        .and_then(JsonValue::as_str)
                        .unwrap_or_default()
                        .to_string(),
                ),
            );
            mapping.insert(yaml_key("api_key"), YamlValue::String(String::new()));
            let api_mode = match provider
                .get("apiType")
                .or_else(|| provider.get("api"))
                .or_else(|| provider.get("type"))
                .and_then(JsonValue::as_str)
                .unwrap_or("openai")
            {
                "anthropic" | "anthropic-messages" => "anthropic_messages",
                _ => "chat_completions",
            };
            mapping.insert(
                yaml_key("api_mode"),
                YamlValue::String(api_mode.to_string()),
            );
            custom_providers.push(YamlValue::Mapping(mapping));
            existing.insert(name.to_ascii_lowercase());
            added += 1;
            self.record(
                "full-providers",
                Some(format!("models.providers.{name}")),
                Some(format!("config.yaml custom_providers[{name}]")),
                "migrated",
                "",
                JsonValue::Null,
            );
        }
        if added > 0 && self.execute {
            let _ = self.maybe_backup(&destination);
            let _ = dump_yaml_file(&destination, &yaml);
        }
        if let Some(aliases) = config.pointer("/agents/defaults/models")
            && aliases.as_object().is_some_and(|object| !object.is_empty())
        {
            self.archive_value(
                "full-providers",
                "agents.defaults.models",
                "model-aliases.json",
                aliases,
                "Model aliases/catalog archived",
            );
        }
    }

    fn migrate_deep_channels(&mut self, config: &JsonValue) {
        let Some(channels) = config.get("channels").and_then(JsonValue::as_object) else {
            self.record(
                "deep-channels",
                None,
                None,
                "skipped",
                "No channel configuration found",
                JsonValue::Null,
            );
            return;
        };
        let mut complex = JsonMap::new();
        for (name, channel) in channels {
            let Some(object) = channel.as_object() else {
                continue;
            };
            let filtered = object
                .iter()
                .filter(|(key, value)| {
                    !matches!(
                        key.as_str(),
                        "botToken"
                            | "appToken"
                            | "allowFrom"
                            | "enabled"
                            | "requireMention"
                            | "autoThread"
                    ) && !value.is_null()
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<JsonMap<_, _>>();
            if !filtered.is_empty() {
                complex.insert(name.clone(), JsonValue::Object(filtered));
            }
        }
        if !complex.is_empty() {
            self.archive_value(
                "deep-channels",
                "openclaw.json channels (advanced settings)",
                "channels-deep-config.json",
                &JsonValue::Object(complex),
                "Deep channel config archived",
            );
        } else {
            self.record(
                "deep-channels",
                None,
                None,
                "skipped",
                "No deep channel configuration found",
                JsonValue::Null,
            );
        }
    }

    fn migrate_browser_config(&mut self, config: &JsonValue) {
        let Some(browser) = config.get("browser").and_then(JsonValue::as_object) else {
            self.record(
                "browser-config",
                None,
                None,
                "skipped",
                "No browser configuration found",
                JsonValue::Null,
            );
            return;
        };
        let destination = self.target_root.join("config.yaml");
        let mut yaml = load_yaml_file(&destination);
        let browser_yaml = ensure_mapping_mut(&mut yaml, "browser");
        let mut changed = false;
        if let Some(cdp_url) = browser.get("cdpUrl").and_then(JsonValue::as_str) {
            browser_yaml.insert(yaml_key("cdp_url"), YamlValue::String(cdp_url.to_string()));
            changed = true;
        }
        if let Some(headless) = browser.get("headless").and_then(JsonValue::as_bool) {
            browser_yaml.insert(yaml_key("headless"), YamlValue::Bool(headless));
            changed = true;
        }
        if changed && self.execute {
            let _ = self.maybe_backup(&destination);
            let _ = dump_yaml_file(&destination, &yaml);
        }
        if changed {
            self.record(
                "browser-config",
                Some("openclaw.json browser.*".to_string()),
                Some("config.yaml browser".to_string()),
                "migrated",
                "",
                JsonValue::Null,
            );
        }
        let advanced = browser
            .iter()
            .filter(|(key, value)| {
                !matches!(key.as_str(), "cdpUrl" | "headless") && !value.is_null()
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect::<JsonMap<_, _>>();
        if !advanced.is_empty() {
            self.archive_value(
                "browser-config",
                "openclaw.json browser (advanced)",
                "browser-config.json",
                &JsonValue::Object(advanced),
                "Advanced browser settings archived",
            );
        }
    }

    fn migrate_tools_config(&mut self, config: &JsonValue) {
        let Some(tools) = config.get("tools").and_then(JsonValue::as_object) else {
            self.record(
                "tools-config",
                None,
                None,
                "skipped",
                "No tools configuration found",
                JsonValue::Null,
            );
            return;
        };
        let destination = self.target_root.join("config.yaml");
        let mut yaml = load_yaml_file(&destination);
        let mut changed = false;
        if let Some(timeout) = config
            .pointer("/tools/exec/timeoutSec")
            .or_else(|| config.pointer("/tools/exec/timeout"))
            .and_then(json_number_to_yaml)
        {
            let terminal = ensure_mapping_mut(&mut yaml, "terminal");
            terminal.insert(yaml_key("timeout"), timeout);
            changed = true;
        }
        if self.migrate_secrets
            && let Some(key) = config
                .pointer("/tools/web/search/brave/apiKey")
                .or_else(|| config.pointer("/tools/webSearch/braveApiKey"))
                .or_else(|| config.pointer("/tools/web/braveApiKey"))
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
        {
            let mut additions = BTreeMap::new();
            additions.insert("BRAVE_API_KEY".to_string(), key.to_string());
            self.merge_env_values(&additions, "env-var", "tools.web.search.brave.apiKey");
        }
        if changed && self.execute {
            let _ = self.maybe_backup(&destination);
            let _ = dump_yaml_file(&destination, &yaml);
            self.record(
                "tools-config",
                Some("openclaw.json tools.*".to_string()),
                Some("config.yaml terminal".to_string()),
                "migrated",
                "",
                JsonValue::Null,
            );
        }
        self.archive_value(
            "tools-config",
            "openclaw.json tools (full)",
            "tools-config.json",
            &JsonValue::Object(tools.clone()),
            "Full tools config archived for reference",
        );
    }

    fn migrate_approvals_config(&mut self, config: &JsonValue) {
        let Some(approvals) = config.get("approvals").and_then(JsonValue::as_object) else {
            self.record(
                "approvals-config",
                None,
                None,
                "skipped",
                "No approvals configuration found",
                JsonValue::Null,
            );
            return;
        };
        let mode = approvals
            .get("exec")
            .and_then(|value| value.get("mode"))
            .or_else(|| approvals.get("mode"))
            .or_else(|| approvals.get("defaultMode"))
            .and_then(JsonValue::as_str);
        if let Some(mode) = mode {
            let destination = self.target_root.join("config.yaml");
            let mut yaml = load_yaml_file(&destination);
            let approvals_yaml = ensure_mapping_mut(&mut yaml, "approvals");
            let hermes_mode = match mode {
                "auto" => "off",
                "always" | "manual" => "manual",
                "smart" => "smart",
                _ => "manual",
            };
            approvals_yaml.insert(yaml_key("mode"), YamlValue::String(hermes_mode.to_string()));
            if self.execute {
                let _ = self.maybe_backup(&destination);
                let _ = dump_yaml_file(&destination, &yaml);
            }
            self.record(
                "approvals-config",
                Some("openclaw.json approvals.mode".to_string()),
                Some("config.yaml approvals.mode".to_string()),
                "migrated",
                &format!("Mapped '{mode}' -> '{hermes_mode}'"),
                JsonValue::Null,
            );
        }
        if approvals.len() > 1 {
            self.archive_value(
                "approvals-config",
                "openclaw.json approvals (rules)",
                "approvals-config.json",
                &JsonValue::Object(approvals.clone()),
                "Approvals config archived",
            );
        }
    }

    fn migrate_logging_config(&mut self, config: &JsonValue) {
        let mut combined = JsonMap::new();
        if let Some(logging) = config.get("logging").filter(|value| !value.is_null()) {
            combined.insert("logging".to_string(), logging.clone());
        }
        if let Some(diagnostics) = config.get("diagnostics").filter(|value| !value.is_null()) {
            combined.insert("diagnostics".to_string(), diagnostics.clone());
        }
        if combined.is_empty() {
            self.record(
                "logging-config",
                None,
                None,
                "skipped",
                "No logging/diagnostics configuration found",
                JsonValue::Null,
            );
            return;
        }
        self.archive_value(
            "logging-config",
            "openclaw.json logging/diagnostics",
            "logging-diagnostics-config.json",
            &JsonValue::Object(combined),
            "Logging and diagnostics config archived",
        );
    }

    fn archive_json_section(
        &mut self,
        kind: &str,
        source_label: &str,
        value: Option<&JsonValue>,
        filename: &str,
        reason: &str,
    ) {
        let Some(value) = value.filter(|value| !value.is_null()) else {
            self.record(
                kind,
                None,
                None,
                "skipped",
                &format!("No {} found", kind.replace('-', " ")),
                JsonValue::Null,
            );
            return;
        };
        self.archive_value(kind, source_label, filename, value, reason);
    }

    fn archive_value(
        &mut self,
        kind: &str,
        source_label: &str,
        filename: &str,
        value: &JsonValue,
        reason: &str,
    ) {
        let destination = self.archive_dir.as_ref().map(|root| root.join(filename));
        if self.execute
            && let Some(destination) = destination.as_ref()
        {
            let _ = ensure_parent(destination);
            let _ = fs::write(
                destination,
                format!(
                    "{}\n",
                    serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string())
                ),
            );
        }
        self.record(
            kind,
            Some(source_label.to_string()),
            destination.as_ref().map(|path| path.display().to_string()),
            "archived",
            reason,
            JsonValue::Null,
        );
    }

    fn generate_migration_notes(&self) -> Result<(), Box<dyn Error>> {
        let Some(output_dir) = self.output_dir.as_ref() else {
            return Ok(());
        };
        fs::create_dir_all(output_dir)?;
        let archived = self
            .items
            .iter()
            .filter(|item| item.status == "archived")
            .collect::<Vec<_>>();
        let conflicts = self
            .items
            .iter()
            .filter(|item| item.status == "conflict")
            .collect::<Vec<_>>();
        let mut lines = vec![
            "# OpenClaw -> Hermes Migration Notes".to_string(),
            String::new(),
            "This document lists items that require manual attention after migration.".to_string(),
            String::new(),
            "## PM2 / External Processes".to_string(),
            String::new(),
            "Your PM2 processes are not modified by this migration.".to_string(),
            String::new(),
        ];
        if !archived.is_empty() {
            lines.push("## Archived Items".to_string());
            lines.push(String::new());
            for item in archived {
                lines.push(format!(
                    "- **{}**: `{}` -- {}",
                    item.kind,
                    item.destination.as_deref().unwrap_or("(n/a)"),
                    item.reason
                ));
            }
            lines.push(String::new());
        }
        if !conflicts.is_empty() {
            lines.push("## Conflicts".to_string());
            lines.push(String::new());
            for item in conflicts {
                lines.push(format!(
                    "- **{}**: {} ({})",
                    item.kind,
                    item.destination.as_deref().unwrap_or("(n/a)"),
                    item.reason
                ));
            }
            lines.push(String::new());
        }
        fs::write(
            output_dir.join("migration-notes.md"),
            format!("{}\n", lines.join("\n")),
        )?;
        Ok(())
    }
}

pub fn run_native_preview(
    context: &HermesContext,
    source_root: PathBuf,
    args: &MigrateArgs,
) -> Result<MigrationReport, Box<dyn Error>> {
    Migrator::new(context, source_root, false, args)?.migrate()
}

pub fn run_native_apply(
    context: &HermesContext,
    source_root: PathBuf,
    args: &MigrateArgs,
) -> Result<MigrationReport, Box<dyn Error>> {
    Migrator::new(context, source_root, true, args)?.migrate()
}

fn ensure_target_config_exists(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    context.ensure_hermes_home()?;
    let path = context.config_path();
    if !path.exists() {
        fs::write(&path, serde_yaml::to_string(&HermesConfig::default())?)?;
    }
    Ok(())
}

fn selected_options(preset: MigratePreset) -> BTreeSet<String> {
    let values: &[&str] = match preset {
        MigratePreset::UserData => &[
            "soul",
            "workspace-agents",
            "memory",
            "user-profile",
            "messaging-settings",
            "command-allowlist",
            "skills",
            "tts-assets",
            "discord-settings",
            "slack-settings",
            "whatsapp-settings",
            "signal-settings",
            "model-config",
            "tts-config",
            "shared-skills",
            "daily-memory",
            "archive",
            "mcp-servers",
            "agent-config",
            "session-config",
            "browser-config",
            "tools-config",
            "approvals-config",
            "deep-channels",
            "full-providers",
            "plugins-config",
            "cron-jobs",
            "hooks-config",
            "memory-backend",
            "skills-config",
            "ui-identity",
            "logging-config",
            "gateway-config",
        ],
        MigratePreset::Full => ALL_MIGRATION_OPTIONS,
    };
    values.iter().map(|value| (*value).to_string()).collect()
}

fn configured_workspace(source_root: &Path, config: &JsonValue) -> Option<PathBuf> {
    let workspace = config
        .pointer("/agents/defaults/workspace")
        .and_then(JsonValue::as_str)?
        .trim();
    if workspace.is_empty() {
        return None;
    }
    let path = PathBuf::from(workspace).expand_home();
    let Ok(resolved) = path.canonicalize() else {
        return None;
    };
    if resolved.is_dir() && resolved.strip_prefix(source_root).is_err() {
        Some(resolved)
    } else {
        None
    }
}

fn load_openclaw_config(source_root: &Path) -> JsonValue {
    for name in ["openclaw.json", "clawdbot.json", "moltbot.json"] {
        let path = source_root.join(name);
        if let Ok(raw) = fs::read_to_string(path)
            && let Ok(value) = serde_json::from_str::<JsonValue>(&raw)
            && value.is_object()
        {
            return value;
        }
    }
    JsonValue::Object(JsonMap::new())
}

fn load_openclaw_env(source_root: &Path) -> HashMap<String, String> {
    parse_env_file(&source_root.join(".env"))
}

fn resolve_secret_input(value: &JsonValue, env: &HashMap<String, String>) -> Option<String> {
    match value {
        JsonValue::String(text) => {
            let trimmed = text.trim();
            if let Some(inner) = trimmed
                .strip_prefix("${")
                .and_then(|rest| rest.strip_suffix('}'))
            {
                return env
                    .get(inner)
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty());
            }
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        JsonValue::Object(object) => {
            if object.get("source").and_then(JsonValue::as_str) == Some("env") {
                let id = object.get("id").and_then(JsonValue::as_str)?.trim();
                return env
                    .get(id)
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty());
            }
            None
        }
        _ => None,
    }
}

fn build_warnings(
    items: &[MigrationItem],
    execute: bool,
    migrate_secrets: bool,
    config_apply_blocked: bool,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if items.iter().any(|item| item.status == "conflict") {
        warnings.push("Conflicts were found. Re-run with --overwrite to replace conflicting targets after item-level backups.".to_string());
    }
    if items.iter().any(|item| item.status == "error") {
        warnings.push("One or more items failed. Inspect the report and re-run after fixing the underlying cause.".to_string());
    }
    if config_apply_blocked && execute {
        warnings.push("A config.yaml write hit a conflict or error mid-apply; later config items were skipped to avoid a partial write.".to_string());
    }
    if !migrate_secrets
        && items
            .iter()
            .any(|item| item.kind == "provider-keys" && item.status == "skipped")
    {
        warnings.push("API keys and other credentials were detected but not imported. Re-run with --migrate-secrets to copy supported keys into the Hermes env file.".to_string());
    }
    warnings
}

fn build_next_steps(
    execute: bool,
    summary: &BTreeMap<String, usize>,
    output_dir: Option<&Path>,
) -> Vec<String> {
    if !execute {
        return vec![
            "Re-run without --dry-run to apply the migration.".to_string(),
            "Pass --overwrite to resolve conflicts, or --migrate-secrets to include API keys."
                .to_string(),
        ];
    }
    let mut steps = Vec::new();
    if summary.get("migrated").copied().unwrap_or(0) > 0 {
        if let Some(output_dir) = output_dir {
            steps.push(format!(
                "Review the migration report at {}",
                output_dir.join("summary.md").display()
            ));
        } else {
            steps.push("Review the migration report.".to_string());
        }
        steps.push(
            "Start a new Hermes session (or /reset) to pick up the imported config.".to_string(),
        );
    }
    if summary.get("conflict").copied().unwrap_or(0) > 0 {
        steps.push(
            "Re-run with --overwrite to apply items that were blocked by conflicts.".to_string(),
        );
    }
    steps
}

fn write_report(
    output_dir: &Path,
    source_root: &Path,
    target_root: &Path,
    preset: &str,
    execute: bool,
    summary: &BTreeMap<String, usize>,
    items: &[MigrationItem],
    warnings: &[String],
    next_steps: &[String],
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    let report = json!({
        "timestamp": Local::now().format("%Y%m%dT%H%M%S").to_string(),
        "mode": if execute { "execute" } else { "dry-run" },
        "source_root": source_root.display().to_string(),
        "target_root": target_root.display().to_string(),
        "output_dir": output_dir.display().to_string(),
        "preset": preset,
        "summary": summary,
        "items": items,
        "warnings": warnings,
        "next_steps": next_steps,
    });
    fs::write(
        output_dir.join("report.json"),
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;

    let mut lines = vec![
        "# OpenClaw -> Hermes Migration Report".to_string(),
        String::new(),
        format!("- Mode: {}", if execute { "execute" } else { "dry-run" }),
        format!("- Source: `{}`", source_root.display()),
        format!("- Target: `{}`", target_root.display()),
        String::new(),
        "## Summary".to_string(),
        String::new(),
    ];
    for (key, value) in summary {
        lines.push(format!("- {key}: {value}"));
    }
    if !warnings.is_empty() {
        lines.push(String::new());
        lines.push("## Warnings".to_string());
        lines.push(String::new());
        for warning in warnings {
            lines.push(format!("- {warning}"));
        }
    }
    lines.push(String::new());
    lines.push("## What Was Not Fully Brought Over".to_string());
    lines.push(String::new());
    let skipped = items
        .iter()
        .filter(|item| matches!(item.status.as_str(), "skipped" | "conflict" | "error"))
        .collect::<Vec<_>>();
    if skipped.is_empty() {
        lines.push("- Nothing. All discovered items were either migrated or archived.".to_string());
    } else {
        for item in skipped {
            lines.push(format!(
                "- `{}` -> `{}`: {}",
                item.source.as_deref().unwrap_or("(n/a)"),
                item.destination.as_deref().unwrap_or("(n/a)"),
                item.reason
            ));
        }
    }
    if !next_steps.is_empty() {
        lines.push(String::new());
        lines.push("## Next Steps".to_string());
        lines.push(String::new());
        for step in next_steps {
            lines.push(format!("- {step}"));
        }
    }
    fs::write(
        output_dir.join("summary.md"),
        format!("{}\n", lines.join("\n")),
    )?;
    Ok(())
}

#[derive(Default)]
struct MergeStats {
    existing: usize,
    added: usize,
    duplicates: usize,
    overflowed: usize,
}

fn parse_existing_memory_entries(path: &Path) -> Vec<String> {
    if !path.exists() {
        return Vec::new();
    }
    let raw = fs::read_to_string(path).unwrap_or_default();
    if raw.contains(ENTRY_DELIMITER) {
        return raw
            .split(ENTRY_DELIMITER)
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(str::to_string)
            .collect();
    }
    extract_markdown_entries(&raw)
}

fn extract_markdown_entries(text: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut headings = Vec::<String>::new();
    let mut paragraph = Vec::<String>::new();
    let mut in_code = false;
    let flush = |entries: &mut Vec<String>, headings: &[String], paragraph: &mut Vec<String>| {
        if paragraph.is_empty() {
            return;
        }
        let text = paragraph.join(" ").trim().to_string();
        paragraph.clear();
        if text.is_empty() {
            return;
        }
        let prefix = headings
            .iter()
            .filter(|heading| !heading.is_empty() && !heading.ends_with(".md"))
            .cloned()
            .collect::<Vec<_>>()
            .join(" > ");
        if prefix.is_empty() {
            entries.push(text);
        } else {
            entries.push(format!("{prefix}: {text}"));
        }
    };
    for raw_line in text.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            flush(&mut entries, &headings, &mut paragraph);
            continue;
        }
        if in_code {
            continue;
        }
        if trimmed.is_empty() {
            flush(&mut entries, &headings, &mut paragraph);
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix('#') {
            let level = trimmed.chars().take_while(|ch| *ch == '#').count();
            let heading = rest.trim().to_string();
            flush(&mut entries, &headings, &mut paragraph);
            while headings.len() >= level {
                headings.pop();
            }
            headings.push(heading);
            continue;
        }
        if let Some(bullet) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            flush(&mut entries, &headings, &mut paragraph);
            let prefix = headings.join(" > ");
            if prefix.is_empty() {
                entries.push(bullet.trim().to_string());
            } else {
                entries.push(format!("{prefix}: {}", bullet.trim()));
            }
            continue;
        }
        paragraph.push(trimmed.to_string());
    }
    flush(&mut entries, &headings, &mut paragraph);
    dedupe_strings(entries)
}

fn dedupe_strings(entries: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for entry in entries {
        let key = normalize_text(&entry);
        if seen.insert(key) {
            deduped.push(entry);
        }
    }
    deduped
}

fn merge_entries(
    existing: &[String],
    incoming: &[String],
    limit: usize,
) -> (Vec<String>, MergeStats, Vec<String>) {
    let mut merged = existing.to_vec();
    let mut seen = existing
        .iter()
        .map(|entry| normalize_text(entry))
        .collect::<HashSet<_>>();
    let mut stats = MergeStats {
        existing: existing.len(),
        ..MergeStats::default()
    };
    let mut overflowed = Vec::new();
    let mut current_len = if merged.is_empty() {
        0
    } else {
        merged.join(ENTRY_DELIMITER).len()
    };
    for entry in incoming {
        let normalized = normalize_text(entry);
        if normalized.is_empty() {
            continue;
        }
        if seen.contains(&normalized) {
            stats.duplicates += 1;
            continue;
        }
        let candidate_len = if merged.is_empty() {
            entry.len()
        } else {
            current_len + ENTRY_DELIMITER.len() + entry.len()
        };
        if candidate_len > limit {
            stats.overflowed += 1;
            overflowed.push(entry.clone());
            continue;
        }
        merged.push(entry.clone());
        seen.insert(normalized);
        current_len = candidate_len;
        stats.added += 1;
    }
    (merged, stats, overflowed)
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn rebrand_text(text: &str) -> String {
    let replacements = [
        ("OpenClaw", "Hermes"),
        ("openclaw", "hermes"),
        ("ClawdBot", "Hermes"),
        ("clawdbot", "hermes"),
        ("MoltBot", "Hermes"),
        ("moltbot", "hermes"),
    ];
    let mut output = text.to_string();
    for (from, to) in replacements {
        output = output.replace(from, to);
    }
    output
}

fn parse_env_file(path: &Path) -> HashMap<String, String> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (key, value) = line.split_once('=')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect()
}

fn save_env_file(path: &Path, env: &HashMap<String, String>) -> Result<(), Box<dyn Error>> {
    ensure_parent(path)?;
    let mut lines = env.iter().collect::<Vec<_>>();
    lines.sort_by(|left, right| left.0.cmp(right.0));
    let body = lines
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, format!("{body}\n"))?;
    Ok(())
}

fn load_yaml_file(path: &Path) -> YamlValue {
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_yaml::from_str(&raw).ok())
        .unwrap_or_else(|| YamlValue::Mapping(YamlMapping::new()))
}

fn dump_yaml_file(path: &Path, value: &YamlValue) -> Result<(), Box<dyn Error>> {
    ensure_parent(path)?;
    fs::write(path, serde_yaml::to_string(value)?)?;
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn ensure_mapping_mut<'a>(root: &'a mut YamlValue, key: &str) -> &'a mut YamlMapping {
    if !root.is_mapping() {
        *root = YamlValue::Mapping(YamlMapping::new());
    }
    let mapping = root.as_mapping_mut().expect("mapping");
    let entry = mapping
        .entry(yaml_key(key))
        .or_insert_with(|| YamlValue::Mapping(YamlMapping::new()));
    if !entry.is_mapping() {
        *entry = YamlValue::Mapping(YamlMapping::new());
    }
    entry.as_mapping_mut().expect("mapping")
}

fn ensure_sequence_mut<'a>(root: &'a mut YamlValue, key: &str) -> &'a mut Vec<YamlValue> {
    if !root.is_mapping() {
        *root = YamlValue::Mapping(YamlMapping::new());
    }
    let mapping = root.as_mapping_mut().expect("mapping");
    let entry = mapping
        .entry(yaml_key(key))
        .or_insert_with(|| YamlValue::Sequence(Vec::new()));
    if !entry.is_sequence() {
        *entry = YamlValue::Sequence(Vec::new());
    }
    entry.as_sequence_mut().expect("sequence")
}

fn yaml_key(key: &str) -> YamlValue {
    YamlValue::String(key.to_string())
}

fn yaml_sequence_strings(value: Option<&YamlValue>) -> Vec<String> {
    value
        .and_then(YamlValue::as_sequence)
        .map(|items| {
            items
                .iter()
                .filter_map(YamlValue::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn extract_model_string(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(text) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        JsonValue::Object(object) => object
            .get("primary")
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string),
        _ => None,
    }
}

fn channel_field(config: &JsonValue, channel: &str, field: &str) -> Option<String> {
    let channel = config.pointer(&format!("/channels/{channel}"))?;
    channel
        .get(field)
        .or_else(|| channel.pointer(&format!("/accounts/default/{field}")))
        .and_then(JsonValue::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn channel_list_field(config: &JsonValue, channel: &str, field: &str) -> Option<Vec<String>> {
    let channel = config.pointer(&format!("/channels/{channel}"))?;
    channel
        .get(field)
        .or_else(|| channel.pointer(&format!("/accounts/default/{field}")))
        .and_then(JsonValue::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty())
}

fn json_object_to_yaml_mapping(value: &JsonValue) -> Option<YamlMapping> {
    value.as_object().map(|object| {
        object
            .iter()
            .map(|(key, value)| (yaml_key(key), json_to_yaml(value)))
            .collect()
    })
}

fn json_to_yaml(value: &JsonValue) -> YamlValue {
    match value {
        JsonValue::Null => YamlValue::Null,
        JsonValue::Bool(value) => YamlValue::Bool(*value),
        JsonValue::Number(value) => {
            if let Some(value) = value.as_i64() {
                YamlValue::Number(serde_yaml::Number::from(value))
            } else if let Some(value) = value.as_u64() {
                YamlValue::Number(serde_yaml::Number::from(value))
            } else if let Some(value) = value.as_f64() {
                YamlValue::Number(serde_yaml::Number::from(value))
            } else {
                YamlValue::Null
            }
        }
        JsonValue::String(value) => YamlValue::String(value.clone()),
        JsonValue::Array(values) => YamlValue::Sequence(values.iter().map(json_to_yaml).collect()),
        JsonValue::Object(values) => YamlValue::Mapping(
            values
                .iter()
                .map(|(key, value)| (yaml_key(key), json_to_yaml(value)))
                .collect(),
        ),
    }
}

fn json_number_to_yaml(value: &JsonValue) -> Option<YamlValue> {
    match value {
        JsonValue::Number(number) => {
            if let Some(value) = number.as_i64() {
                Some(YamlValue::Number(serde_yaml::Number::from(value)))
            } else if let Some(value) = number.as_u64() {
                Some(YamlValue::Number(serde_yaml::Number::from(value)))
            } else {
                number
                    .as_f64()
                    .map(|value| YamlValue::Number(serde_yaml::Number::from(value)))
            }
        }
        _ => None,
    }
}

fn relative_label(path: &Path, root: &Path) -> PathBuf {
    path.strip_prefix(root).unwrap_or(path).to_path_buf()
}

fn resolve_skill_destination(destination: &Path) -> PathBuf {
    if !destination.exists() {
        return destination.to_path_buf();
    }
    let mut candidate = destination.with_file_name(format!(
        "{}-imported",
        destination
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("skill")
    ));
    let mut counter = 2usize;
    while candidate.exists() {
        candidate = destination.with_file_name(format!(
            "{}-imported-{counter}",
            destination
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("skill")
        ));
        counter += 1;
    }
    candidate
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else {
            ensure_parent(&destination_path)?;
            fs::copy(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files.extend(collect_files(&path));
        } else if path.is_file() {
            files.push(path);
        }
    }
    files
}

fn file_contents_match(left: &Path, right: &Path) -> bool {
    fs::read(left).ok() == fs::read(right).ok()
}

trait ExpandHome {
    fn expand_home(self) -> PathBuf;
}

impl ExpandHome for PathBuf {
    fn expand_home(self) -> PathBuf {
        let Some(rest) = self.to_str().and_then(|value| value.strip_prefix("~/")) else {
            return self;
        };
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/"))
            .join(rest)
    }
}
