use std::collections::HashSet;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::json;
use serde_yaml::{Mapping, Value as YamlValue};
use sha2::{Digest, Sha256};

use crate::LoadedConfig;

const STORE_DIRNAME: &str = "store";
const INDEXES_DIRNAME: &str = "indexes";
const PROJECTS_DIRNAME: &str = "projects";
const REFS_PREFIX: &str = "refs/hermes";
const MAX_FILES: usize = 50_000;
const DEFAULT_TIMEOUT_SECONDS: u64 = 30;
const MAX_TIMEOUT_SECONDS: u64 = 120;
const DEFAULT_EXCLUDES: &[&str] = &[
    "node_modules/",
    "dist/",
    "build/",
    "target/",
    "out/",
    ".next/",
    ".nuxt/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".cache/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "coverage/",
    ".coverage",
    ".venv/",
    "venv/",
    "env/",
    ".git/",
    ".hg/",
    ".svn/",
    ".worktrees/",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.o",
    "*.a",
    "*.jar",
    "*.class",
    "*.exe",
    "*.obj",
    "*.mp4",
    "*.mov",
    "*.mkv",
    "*.webm",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
    "*.rar",
    "*.iso",
    ".env",
    ".env.*",
    ".env.local",
    ".env.*.local",
    ".DS_Store",
    "Thumbs.db",
    "*.log",
];

#[derive(Debug, Clone)]
pub struct CheckpointManager {
    hermes_home: PathBuf,
    max_snapshots: usize,
    max_total_size_mb: u64,
    max_file_size_mb: u64,
    git_timeout: Duration,
    checkpointed_dirs: HashSet<PathBuf>,
    git_available: Option<bool>,
}

#[derive(Debug)]
struct GitCommandResult {
    success: bool,
    returncode: Option<i32>,
    stdout: String,
    stderr: String,
}

impl CheckpointManager {
    pub fn load_for_runtime(hermes_home: &Path, loaded: &LoadedConfig) -> Option<Self> {
        let root = loaded.raw.as_mapping()?;
        let checkpoints = mapping_value(root, "checkpoints")?.as_mapping()?;
        if !mapping_value(checkpoints, "enabled").is_some_and(yaml_truthy) {
            return None;
        }

        let max_snapshots = mapping_value(checkpoints, "max_snapshots")
            .and_then(yaml_u64)
            .unwrap_or(20)
            .clamp(1, 1_000) as usize;
        let max_total_size_mb = mapping_value(checkpoints, "max_total_size_mb")
            .and_then(yaml_u64)
            .unwrap_or(500);
        let max_file_size_mb = mapping_value(checkpoints, "max_file_size_mb")
            .and_then(yaml_u64)
            .unwrap_or(10);
        let timeout_seconds = env::var("HERMES_CHECKPOINT_TIMEOUT")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECONDS)
            .clamp(1, MAX_TIMEOUT_SECONDS);

        Some(Self {
            hermes_home: hermes_home.to_path_buf(),
            max_snapshots,
            max_total_size_mb,
            max_file_size_mb,
            git_timeout: Duration::from_secs(timeout_seconds),
            checkpointed_dirs: HashSet::new(),
            git_available: None,
        })
    }

    pub fn new_turn(&mut self) {
        self.checkpointed_dirs.clear();
    }

    pub fn get_working_dir_for_path(&self, file_path: &Path) -> PathBuf {
        let candidate = if file_path.is_dir() {
            file_path.to_path_buf()
        } else {
            file_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| file_path.to_path_buf())
        };

        let mut check = candidate.clone();
        while check.parent().is_some() {
            if has_project_marker(&check) {
                return check;
            }
            let Some(parent) = check.parent() else {
                break;
            };
            if parent == check {
                break;
            }
            check = parent.to_path_buf();
        }
        candidate
    }

    pub fn ensure_checkpoint(&mut self, working_dir: &Path, reason: &str) -> bool {
        if !*self.git_available.get_or_insert_with(git_installed) {
            return false;
        }

        let Ok(abs_dir) = fs::canonicalize(working_dir) else {
            return false;
        };
        if is_broad_directory(&abs_dir) || self.checkpointed_dirs.contains(&abs_dir) {
            return false;
        }
        self.checkpointed_dirs.insert(abs_dir.clone());

        match self.take_snapshot(&abs_dir, reason) {
            Ok(taken) => taken,
            Err(error) => {
                log::debug!("checkpoint skipped: {error}");
                false
            }
        }
    }

    fn take_snapshot(&self, working_dir: &Path, reason: &str) -> Result<bool, String> {
        let store = self.store_path();
        self.init_store(&store)?;
        self.touch_project(&store, working_dir)?;

        if dir_file_count(working_dir) > MAX_FILES {
            return Ok(false);
        }

        let dir_hash = project_hash(working_dir);
        let index_file = index_path(&store, &dir_hash);
        let ref_name = ref_name(&dir_hash);

        if index_file.exists() {
            if let Some(ref_commit) = self.git_stdout(
                &store,
                working_dir,
                None,
                &["rev-parse", "--verify", &format!("{ref_name}^{{commit}}")],
                &[128],
            )? {
                let _ = self.git_success(
                    &store,
                    working_dir,
                    Some(&index_file),
                    &["read-tree", &ref_commit],
                    &[128],
                )?;
            } else {
                let _ = fs::remove_file(&index_file);
            }
        } else if let Some(parent) = index_file.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }

        if !self.git_success(&store, working_dir, Some(&index_file), &["add", "-A"], &[])? {
            return Ok(false);
        }

        if self.max_file_size_mb > 0 {
            self.drop_oversize_from_index(&store, working_dir, &index_file)?;
        }

        let ref_commit = self.git_stdout(
            &store,
            working_dir,
            None,
            &["rev-parse", "--verify", &format!("{ref_name}^{{commit}}")],
            &[128],
        )?;
        let has_ref = ref_commit.is_some();

        if let Some(ref_commit) = ref_commit.as_deref() {
            let unchanged = self.git_status_code(
                &store,
                working_dir,
                Some(&index_file),
                &["diff-index", "--cached", "--quiet", ref_commit],
                &[1],
            )? == 0;
            if unchanged {
                return Ok(false);
            }
        } else {
            let cached = self.git_stdout(
                &store,
                working_dir,
                Some(&index_file),
                &["ls-files", "--cached"],
                &[],
            )?;
            if cached.as_deref().is_none_or(str::is_empty) {
                return Ok(false);
            }
        }

        let Some(tree_sha) =
            self.git_stdout(&store, working_dir, Some(&index_file), &["write-tree"], &[])?
        else {
            return Ok(false);
        };

        let mut commit_args = vec![
            "commit-tree".to_string(),
            tree_sha,
            "-m".to_string(),
            reason.to_string(),
            "--no-gpg-sign".to_string(),
        ];
        if let Some(parent) = ref_commit.as_ref() {
            commit_args.splice(2..2, [String::from("-p"), parent.clone()]);
        }
        let commit_args_refs = commit_args.iter().map(String::as_str).collect::<Vec<_>>();
        let Some(new_sha) = self.git_stdout(
            &store,
            working_dir,
            Some(&index_file),
            &commit_args_refs,
            &[],
        )?
        else {
            return Ok(false);
        };

        let mut update_args = vec!["update-ref".to_string(), ref_name.clone(), new_sha];
        if let Some(parent) = ref_commit.as_ref() {
            update_args.push(parent.clone());
        }
        let update_args_refs = update_args.iter().map(String::as_str).collect::<Vec<_>>();
        if !self.git_success(&store, working_dir, None, &update_args_refs, &[])? {
            return Ok(false);
        }

        if has_ref || self.max_snapshots > 1 {
            self.prune_project_history(&store, working_dir, &ref_name)?;
        }
        self.enforce_size_cap(&store, working_dir)?;
        Ok(true)
    }

    fn drop_oversize_from_index(
        &self,
        store: &Path,
        working_dir: &Path,
        index_file: &Path,
    ) -> Result<(), String> {
        let cap = self.max_file_size_mb.saturating_mul(1024 * 1024);
        if cap == 0 {
            return Ok(());
        }
        let Some(stdout) = self.git_stdout(
            store,
            working_dir,
            Some(index_file),
            &["ls-files", "--cached", "-z"],
            &[],
        )?
        else {
            return Ok(());
        };

        let mut oversize = Vec::new();
        for rel in stdout.split('\0').filter(|value| !value.is_empty()) {
            let size = working_dir
                .join(rel)
                .metadata()
                .map(|meta| meta.len())
                .unwrap_or(0);
            if size > cap {
                oversize.push(rel.to_string());
            }
        }
        if oversize.is_empty() {
            return Ok(());
        }

        for chunk in oversize.chunks(200) {
            let mut args = vec![
                "rm".to_string(),
                "--cached".to_string(),
                "--quiet".to_string(),
                "--".to_string(),
            ];
            args.extend(chunk.iter().cloned());
            let arg_refs = args.iter().map(String::as_str).collect::<Vec<_>>();
            let _ = self.git_success(store, working_dir, Some(index_file), &arg_refs, &[128])?;
        }
        Ok(())
    }

    fn prune_project_history(
        &self,
        store: &Path,
        working_dir: &Path,
        ref_name: &str,
    ) -> Result<(), String> {
        let Some(count) = self
            .git_stdout(
                store,
                working_dir,
                None,
                &["rev-list", "--count", ref_name],
                &[128],
            )?
            .and_then(|value| value.parse::<usize>().ok())
        else {
            return Ok(());
        };
        if count <= self.max_snapshots {
            return Ok(());
        }

        let Some(list_out) = self.git_stdout(
            store,
            working_dir,
            None,
            &["rev-list", "--reverse", ref_name],
            &[],
        )?
        else {
            return Ok(());
        };
        let commits = list_out
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if commits.len() <= self.max_snapshots {
            return Ok(());
        }

        let keep = &commits[commits.len().saturating_sub(self.max_snapshots)..];
        let mut new_parent: Option<String> = None;
        for sha in keep {
            let Some(tree_sha) = self.git_stdout(
                store,
                working_dir,
                None,
                &["rev-parse", &format!("{sha}^{{tree}}")],
                &[],
            )?
            else {
                return Ok(());
            };
            let message = self
                .git_stdout(
                    store,
                    working_dir,
                    None,
                    &["log", "--format=%s", "-1", sha],
                    &[],
                )?
                .unwrap_or_else(|| String::from("checkpoint"));
            let mut args = vec![
                "commit-tree".to_string(),
                tree_sha,
                "-m".to_string(),
                message,
                "--no-gpg-sign".to_string(),
            ];
            if let Some(parent) = new_parent.as_ref() {
                args.splice(2..2, [String::from("-p"), parent.clone()]);
            }
            let arg_refs = args.iter().map(String::as_str).collect::<Vec<_>>();
            new_parent = self.git_stdout(store, working_dir, None, &arg_refs, &[])?;
            if new_parent.is_none() {
                return Ok(());
            }
        }

        if let Some(new_parent) = new_parent {
            let _ = self.git_success(
                store,
                working_dir,
                None,
                &["update-ref", ref_name, &new_parent],
                &[],
            )?;
            self.run_gc(store, working_dir)?;
        }
        Ok(())
    }

    fn enforce_size_cap(&self, store: &Path, working_dir: &Path) -> Result<(), String> {
        if self.max_total_size_mb == 0 {
            return Ok(());
        }
        let cap_bytes = self.max_total_size_mb.saturating_mul(1024 * 1024);
        for _ in 0..20 {
            if dir_size_bytes(store) <= cap_bytes {
                break;
            }
            let Some(refs_out) = self.git_stdout(
                store,
                working_dir,
                None,
                &["for-each-ref", "--format=%(refname)", REFS_PREFIX],
                &[],
            )?
            else {
                break;
            };
            let refs = refs_out
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>();
            if refs.is_empty() {
                break;
            }

            let mut dropped = false;
            for ref_name in refs {
                let Some(commits_out) = self.git_stdout(
                    store,
                    working_dir,
                    None,
                    &["rev-list", "--reverse", &ref_name],
                    &[],
                )?
                else {
                    continue;
                };
                let commits = commits_out
                    .lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                if commits.len() <= 1 {
                    continue;
                }

                let keep = &commits[1..];
                let mut new_parent: Option<String> = None;
                let mut failed = false;
                for sha in keep {
                    let Some(tree_sha) = self.git_stdout(
                        store,
                        working_dir,
                        None,
                        &["rev-parse", &format!("{sha}^{{tree}}")],
                        &[],
                    )?
                    else {
                        failed = true;
                        break;
                    };
                    let message = self
                        .git_stdout(
                            store,
                            working_dir,
                            None,
                            &["log", "--format=%s", "-1", sha],
                            &[],
                        )?
                        .unwrap_or_else(|| String::from("checkpoint"));
                    let mut args = vec![
                        "commit-tree".to_string(),
                        tree_sha,
                        "-m".to_string(),
                        message,
                        "--no-gpg-sign".to_string(),
                    ];
                    if let Some(parent) = new_parent.as_ref() {
                        args.splice(2..2, [String::from("-p"), parent.clone()]);
                    }
                    let arg_refs = args.iter().map(String::as_str).collect::<Vec<_>>();
                    new_parent = self.git_stdout(store, working_dir, None, &arg_refs, &[])?;
                    if new_parent.is_none() {
                        failed = true;
                        break;
                    }
                }
                if failed {
                    continue;
                }
                if let Some(new_parent) = new_parent {
                    let _ = self.git_success(
                        store,
                        working_dir,
                        None,
                        &["update-ref", &ref_name, &new_parent],
                        &[],
                    )?;
                    dropped = true;
                }
            }
            if !dropped {
                break;
            }
            self.run_gc(store, working_dir)?;
        }
        Ok(())
    }

    fn init_store(&self, store: &Path) -> Result<(), String> {
        if store.join("HEAD").exists() {
            return Ok(());
        }

        fs::create_dir_all(store).map_err(|error| error.to_string())?;
        fs::create_dir_all(store.join(INDEXES_DIRNAME)).map_err(|error| error.to_string())?;
        fs::create_dir_all(store.join(PROJECTS_DIRNAME)).map_err(|error| error.to_string())?;

        let mut command = Command::new("git");
        command
            .arg("init")
            .arg("--bare")
            .arg(store)
            .env("GIT_CONFIG_GLOBAL", devnull_path())
            .env("GIT_CONFIG_SYSTEM", devnull_path())
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .env_remove("GIT_NAMESPACE")
            .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let output = run_command_with_timeout(command, self.git_timeout)
            .map_err(|error| error.to_string())?;
        if !output.success {
            return Err(format!("checkpoint git init failed: {}", output.stderr));
        }

        let base = store.parent().unwrap_or(store);
        let _ = self.git_success(
            store,
            base,
            None,
            &["config", "user.email", "hermes@local"],
            &[],
        )?;
        let _ = self.git_success(
            store,
            base,
            None,
            &["config", "user.name", "Hermes Checkpoint"],
            &[],
        )?;
        let _ = self.git_success(
            store,
            base,
            None,
            &["config", "commit.gpgsign", "false"],
            &[],
        )?;
        let _ = self.git_success(store, base, None, &["config", "tag.gpgSign", "false"], &[])?;
        let _ = self.git_success(store, base, None, &["config", "gc.auto", "0"], &[])?;

        let info_dir = store.join("info");
        fs::create_dir_all(&info_dir).map_err(|error| error.to_string())?;
        fs::write(
            info_dir.join("exclude"),
            format!("{}\n", DEFAULT_EXCLUDES.join("\n")),
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn touch_project(&self, store: &Path, working_dir: &Path) -> Result<(), String> {
        let normalized = fs::canonicalize(working_dir).map_err(|error| error.to_string())?;
        let meta_path = project_meta_path(store, &project_hash(&normalized));
        let now = now_ts();

        let created_at = fs::read_to_string(&meta_path)
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|value| value.get("created_at").and_then(serde_json::Value::as_f64))
            .unwrap_or(now);

        if let Some(parent) = meta_path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(
            meta_path,
            json!({
                "workdir": normalized.display().to_string(),
                "created_at": created_at,
                "last_touch": now,
            })
            .to_string(),
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn store_path(&self) -> PathBuf {
        self.hermes_home.join("checkpoints").join(STORE_DIRNAME)
    }

    fn run_gc(&self, store: &Path, working_dir: &Path) -> Result<(), String> {
        let _ = self.git_success(
            store,
            working_dir,
            None,
            &["reflog", "expire", "--expire=now", "--all"],
            &[],
        )?;
        let _ = self.git_success(
            store,
            working_dir,
            None,
            &["gc", "--prune=now", "--quiet"],
            &[],
        )?;
        Ok(())
    }

    fn git_success(
        &self,
        store: &Path,
        working_dir: &Path,
        index_file: Option<&Path>,
        args: &[&str],
        allowed_returncodes: &[i32],
    ) -> Result<bool, String> {
        Ok(self
            .run_git(store, working_dir, index_file, args, allowed_returncodes)?
            .success)
    }

    fn git_status_code(
        &self,
        store: &Path,
        working_dir: &Path,
        index_file: Option<&Path>,
        args: &[&str],
        allowed_returncodes: &[i32],
    ) -> Result<i32, String> {
        let result = self.run_git(store, working_dir, index_file, args, allowed_returncodes)?;
        Ok(result.returncode.unwrap_or(-1))
    }

    fn git_stdout(
        &self,
        store: &Path,
        working_dir: &Path,
        index_file: Option<&Path>,
        args: &[&str],
        allowed_returncodes: &[i32],
    ) -> Result<Option<String>, String> {
        let result = self.run_git(store, working_dir, index_file, args, allowed_returncodes)?;
        if !result.success {
            return Ok(None);
        }
        let trimmed = result.stdout.trim();
        Ok((!trimmed.is_empty()).then(|| trimmed.to_string()))
    }

    fn run_git(
        &self,
        store: &Path,
        working_dir: &Path,
        index_file: Option<&Path>,
        args: &[&str],
        allowed_returncodes: &[i32],
    ) -> Result<GitCommandResult, String> {
        if !working_dir.exists() {
            return Err(format!(
                "working directory not found: {}",
                working_dir.display()
            ));
        }
        if !working_dir.is_dir() {
            return Err(format!(
                "working directory is not a directory: {}",
                working_dir.display()
            ));
        }

        let mut command = Command::new("git");
        command
            .args(args)
            .current_dir(working_dir)
            .env("GIT_DIR", store)
            .env("GIT_WORK_TREE", working_dir)
            .env("GIT_CONFIG_GLOBAL", devnull_path())
            .env("GIT_CONFIG_SYSTEM", devnull_path())
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env_remove("GIT_NAMESPACE")
            .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(index_file) = index_file {
            command.env("GIT_INDEX_FILE", index_file);
        } else {
            command.env_remove("GIT_INDEX_FILE");
        }

        let output = run_command_with_timeout(command, self.git_timeout)
            .map_err(|error| error.to_string())?;
        if !output.success
            && !allowed_returncodes.contains(&output.returncode.unwrap_or_default())
            && !output.stderr.is_empty()
        {
            log::debug!(
                "checkpoint git command failed: {:?}: {}",
                args,
                output.stderr
            );
        }
        Ok(output)
    }
}

pub fn is_destructive_command(command: &str) -> bool {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return false;
    }
    destructive_patterns().is_match(trimmed) || redirect_overwrite_pattern().is_match(trimmed)
}

fn git_installed() -> bool {
    Command::new("git")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn has_project_marker(path: &Path) -> bool {
    [
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "Makefile",
        "pom.xml",
        ".hg",
        "Gemfile",
    ]
    .iter()
    .any(|marker| path.join(marker).exists())
}

fn is_broad_directory(path: &Path) -> bool {
    path == Path::new("/") || dirs::home_dir().as_deref() == Some(path)
}

fn destructive_patterns() -> &'static Regex {
    static PATTERN: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(
            r"(?:^|\s|&&|\|\||;|`)(?:rm\s|rmdir\s|cp\s|install\s|mv\s|sed\s+-i|truncate\s|dd\s|shred\s|git\s+(?:reset|clean|checkout)\s)",
        )
        .expect("valid destructive command regex")
    })
}

fn redirect_overwrite_pattern() -> &'static Regex {
    static PATTERN: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"[^>]>[^>]|^>[^>]").expect("valid redirect regex"))
}

fn project_hash(working_dir: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(working_dir.display().to_string().as_bytes());
    let digest = hasher.finalize();
    digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn index_path(store: &Path, dir_hash: &str) -> PathBuf {
    store.join(INDEXES_DIRNAME).join(dir_hash)
}

fn ref_name(dir_hash: &str) -> String {
    format!("{REFS_PREFIX}/{dir_hash}")
}

fn project_meta_path(store: &Path, dir_hash: &str) -> PathBuf {
    store
        .join(PROJECTS_DIRNAME)
        .join(format!("{dir_hash}.json"))
}

fn dir_file_count(path: &Path) -> usize {
    let mut count = 0_usize;
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(current) else {
            continue;
        };
        for entry in entries.flatten() {
            count = count.saturating_add(1);
            if count > MAX_FILES {
                return count;
            }
            let child = entry.path();
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                stack.push(child);
            }
        }
    }
    count
}

fn dir_size_bytes(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total = 0_u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            } else if metadata.is_dir() {
                stack.push(path);
            }
        }
    }
    total
}

fn run_command_with_timeout(
    mut command: Command,
    timeout: Duration,
) -> io::Result<GitCommandResult> {
    let mut child = command.spawn()?;
    let deadline = Instant::now() + timeout;
    loop {
        if child.try_wait()?.is_some() {
            let output = child.wait_with_output()?;
            return Ok(GitCommandResult {
                success: output.status.success(),
                returncode: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            });
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(GitCommandResult {
                success: false,
                returncode: None,
                stdout: String::new(),
                stderr: format!("timed out after {}s", timeout.as_secs()),
            });
        }
        sleep(Duration::from_millis(50));
    }
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs_f64())
        .unwrap_or(0.0)
}

fn devnull_path() -> &'static str {
    #[cfg(windows)]
    {
        "NUL"
    }
    #[cfg(not(windows))]
    {
        "/dev/null"
    }
}

fn mapping_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

fn yaml_u64(value: &YamlValue) -> Option<u64> {
    match value {
        YamlValue::Number(number) => number.as_u64(),
        YamlValue::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

fn yaml_truthy(value: &YamlValue) -> bool {
    match value {
        YamlValue::Bool(flag) => *flag,
        YamlValue::Number(number) => number.as_i64().is_some_and(|value| value != 0),
        YamlValue::String(text) => matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn destructive_command_detector_matches_python_contract() {
        assert!(is_destructive_command("rm -rf build"));
        assert!(is_destructive_command("printf hi > out.txt"));
        assert!(!is_destructive_command("printf hi >> out.txt"));
        assert!(!is_destructive_command("printf hi"));
    }

    #[test]
    fn checkpoint_manager_dedupes_within_turn_and_resets_next_turn() {
        let temp = TempDir::new().unwrap();
        let hermes_home = temp.path().join("home");
        let project = temp.path().join("project");
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("Cargo.toml"), "[package]\nname='demo'\n").unwrap();

        let loaded = LoadedConfig {
            path: hermes_home.join("config.yaml"),
            raw: serde_yaml::from_str(
                "checkpoints:\n  enabled: true\n  max_snapshots: 20\n  max_total_size_mb: 500\n  max_file_size_mb: 10\n",
            )
            .unwrap(),
            config: crate::HermesConfig::default(),
            warnings: Vec::new(),
        };

        let mut manager = CheckpointManager::load_for_runtime(&hermes_home, &loaded).unwrap();
        assert!(manager.ensure_checkpoint(&project, "before write_file"));
        assert!(!manager.ensure_checkpoint(&project, "before write_file"));

        fs::write(project.join("Cargo.toml"), "[package]\nname='demo-two'\n").unwrap();
        manager.new_turn();
        assert!(manager.ensure_checkpoint(&project, "before write_file"));

        let store = hermes_home.join("checkpoints").join(STORE_DIRNAME);
        let ref_name = ref_name(&project_hash(&fs::canonicalize(&project).unwrap()));
        let commits = manager
            .git_stdout(
                &store,
                &project,
                None,
                &["rev-list", "--count", &ref_name],
                &[],
            )
            .unwrap()
            .unwrap();
        assert_eq!(commits, "2");
    }
}
