use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Local, Utc};
use clap::Args;
use hermes_core::HermesContext;
use rusqlite::{Connection, DatabaseName, OpenFlags};
use serde::{Deserialize, Serialize};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

const EXCLUDED_DIRS: &[&str] = &[
    "hermes-agent",
    "__pycache__",
    ".git",
    "node_modules",
    "backups",
    "checkpoints",
];
const EXCLUDED_SUFFIXES: &[&str] = &[".pyc", ".pyo", ".db-wal", ".db-shm", ".db-journal"];
const EXCLUDED_NAMES: &[&str] = &["gateway.pid", "cron.pid"];
const SECRET_FILE_NAMES: &[&str] = &[".env", "auth.json", "state.db"];
const QUICK_STATE_FILES: &[&str] = &[
    "state.db",
    "config.yaml",
    ".env",
    "auth.json",
    "cron/jobs.json",
    "gateway_state.json",
    "channel_directory.json",
    "processes.json",
    "pairing",
    "platforms/pairing",
    "feishu_comment_pairing.json",
];
const QUICK_SNAPSHOTS_DIR: &str = "state-snapshots";
pub(crate) const QUICK_DEFAULT_KEEP: usize = 20;

#[derive(Args, Debug)]
pub struct BackupArgs {
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long, short = 'q')]
    pub quick: bool,
    #[arg(long, short = 'l')]
    pub label: Option<String>,
}

#[derive(Args, Debug)]
pub struct ImportArgs {
    pub zipfile: PathBuf,
    #[arg(long, short = 'y')]
    pub force: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickSnapshotManifest {
    pub id: String,
    pub timestamp: String,
    pub label: Option<String>,
    pub file_count: usize,
    pub total_size: u64,
    pub files: BTreeMap<String, u64>,
}

pub fn print_backup(context: &HermesContext, args: BackupArgs) -> Result<(), Box<dyn Error>> {
    if args.quick {
        return print_quick_backup(context, args.label.as_deref());
    }
    let root = context.default_hermes_root();
    if !root.is_dir() {
        return Err(format!("Hermes home directory not found at {}", root.display()).into());
    }
    let out_path = resolve_output_path(context, args.output.as_deref())?;
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Scanning {} ...", display_path(context, &root));
    let mut files = Vec::new();
    let mut skipped_dirs = Vec::new();
    collect_backup_files(&root, &root, &out_path, &mut files, &mut skipped_dirs)?;

    if files.is_empty() {
        println!("No files to back up.");
        return Ok(());
    }

    files.sort_by(|left, right| left.1.cmp(&right.1));
    skipped_dirs.sort();
    skipped_dirs.dedup();

    println!("Backing up {} files ...", files.len());
    let file = File::create(&out_path)?;
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut total_bytes = 0_u64;
    let mut warnings = Vec::new();

    for (index, (source, relative)) in files.iter().enumerate() {
        let member = zip_member_name(relative)?;
        if source
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("db"))
        {
            let tmp_db = unique_temp_path("backup-db", "db");
            match safe_copy_db(source, &tmp_db) {
                Ok(()) => {
                    let size = fs::metadata(&tmp_db).map(|meta| meta.len()).unwrap_or(0);
                    let mut reader = File::open(&tmp_db)?;
                    zip.start_file(member, options)?;
                    io::copy(&mut reader, &mut zip)?;
                    total_bytes += size;
                    let _ = fs::remove_file(&tmp_db);
                }
                Err(error) => {
                    warnings.push(format!("  {}: {}", relative.display(), error));
                    let _ = fs::remove_file(&tmp_db);
                    continue;
                }
            }
        } else {
            let mut reader = match File::open(source) {
                Ok(file) => file,
                Err(error) => {
                    warnings.push(format!("  {}: {}", relative.display(), error));
                    continue;
                }
            };
            zip.start_file(member, options)?;
            io::copy(&mut reader, &mut zip)?;
            total_bytes += fs::metadata(source).map(|meta| meta.len()).unwrap_or(0);
        }

        let processed = index + 1;
        if processed % 500 == 0 {
            println!("  {processed}/{} files ...", files.len());
        }
    }

    zip.finish()?;
    let zip_size = fs::metadata(&out_path)?.len();

    println!();
    println!("Backup complete: {}", out_path.display());
    println!("  Files:       {}", files.len());
    println!("  Original:    {}", format_size(total_bytes));
    println!("  Compressed:  {}", format_size(zip_size));

    if !skipped_dirs.is_empty() {
        println!("\n  Excluded directories:");
        for dir in skipped_dirs {
            println!("    {dir}/");
        }
    }

    if !warnings.is_empty() {
        println!("\n  Warnings ({} files skipped):", warnings.len());
        for warning in warnings.iter().take(10) {
            println!("{warning}");
        }
        if warnings.len() > 10 {
            println!("  ... and {} more", warnings.len() - 10);
        }
    }

    println!("\nRestore with: hermes import {}", out_path.display());
    Ok(())
}

pub fn print_quick_backup(
    context: &HermesContext,
    label: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let snap_id = create_quick_snapshot(context, label)?;
    if let Some(id) = snap_id {
        println!("State snapshot created: {id}");
        let snapshots = list_quick_snapshots(context, QUICK_DEFAULT_KEEP)?;
        println!(
            "  {} snapshot(s) stored in {}/{}",
            snapshots.len(),
            display_path(context, &context.hermes_home()),
            QUICK_SNAPSHOTS_DIR
        );
        println!("  Restore with: hermes snapshot restore {id}");
    } else {
        println!("No state files found to snapshot.");
    }
    Ok(())
}

pub fn print_import(context: &HermesContext, args: ImportArgs) -> Result<(), Box<dyn Error>> {
    let zip_path = resolve_input_path(context, &args.zipfile)?;
    if !zip_path.is_file() {
        return Err(format!("File not found: {}", zip_path.display()).into());
    }

    let file = File::open(&zip_path)?;
    let mut archive = ZipArchive::new(file)?;
    let members = archive_member_names(&mut archive)?;
    validate_backup_names(&members)?;
    let prefix = detect_prefix(&members);
    let file_count = members.iter().filter(|name| !name.ends_with('/')).count();
    let root = context.default_hermes_root();

    println!("Backup contains {file_count} files");
    println!("Target: {}", display_path(context, &root));
    if let Some(value) = prefix.as_deref() {
        println!("Detected archive prefix: {value:?} (will be stripped)");
    }

    if (root.join("config.yaml").exists() || root.join(".env").exists())
        && !args.force
        && !confirm_import()?
    {
        println!("Aborted.");
        return Ok(());
    }

    println!("\nImporting {file_count} files ...");
    fs::create_dir_all(&root)?;
    let mut restored = 0_usize;
    let mut warnings = Vec::new();

    for index in 0..archive.len() {
        let mut member = archive.by_index(index)?;
        if member.is_dir() {
            continue;
        }
        let raw_name = member.name().to_string();
        let stripped = strip_prefix(&raw_name, prefix.as_deref());
        let Some(name) = stripped else {
            continue;
        };
        if name.is_empty() {
            continue;
        }

        let relative = match sanitize_member_path(&name) {
            Ok(path) => path,
            Err(error) => {
                warnings.push(format!("  {name}: {error}"));
                continue;
            }
        };
        let target = root.join(&relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        match File::create(&target) {
            Ok(mut dst) => {
                io::copy(&mut member, &mut dst)?;
                if is_secret_file(&target) {
                    tighten_secret_permissions(&target)?;
                }
                restored += 1;
            }
            Err(error) => {
                warnings.push(format!("  {}: {}", relative.display(), error));
                continue;
            }
        }

        if restored % 500 == 0 {
            println!("  {restored}/{file_count} files ...");
        }
    }

    println!();
    println!("Import complete: {restored} files restored");
    println!("  Target: {}", display_path(context, &root));

    if !warnings.is_empty() {
        println!("\n  Warnings ({} files skipped):", warnings.len());
        for warning in warnings.iter().take(10) {
            println!("{warning}");
        }
        if warnings.len() > 10 {
            println!("  ... and {} more", warnings.len() - 10);
        }
    }

    println!();
    if !root.join("hermes-agent").is_dir() {
        println!("Note: The hermes-agent codebase was not included in the backup.");
        println!("  If this is a fresh install, run: hermes update");
    }
    println!("Done. Your Hermes configuration has been restored.");
    Ok(())
}

pub fn create_quick_snapshot(
    context: &HermesContext,
    label: Option<&str>,
) -> Result<Option<String>, Box<dyn Error>> {
    let home = context.hermes_home();
    let root = quick_snapshot_root(context);
    fs::create_dir_all(&root)?;

    let timestamp = Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let label = validate_snapshot_label(label)?;
    let snapshot_id = label
        .as_deref()
        .map(|value| format!("{timestamp}-{value}"))
        .unwrap_or(timestamp.clone());
    let snapshot_dir = root.join(&snapshot_id);
    fs::create_dir_all(&snapshot_dir)?;

    let mut manifest_files = BTreeMap::new();
    for relative in QUICK_STATE_FILES {
        let source = home.join(relative);
        if !source.exists() {
            continue;
        }
        if source.is_dir() {
            copy_quick_snapshot_dir(&home, &source, &snapshot_dir, &mut manifest_files)?;
            continue;
        }
        if !source.is_file() {
            continue;
        }
        quick_snapshot_copy_file(&home, &source, &snapshot_dir, &mut manifest_files)?;
    }

    if manifest_files.is_empty() {
        fs::remove_dir_all(&snapshot_dir)?;
        return Ok(None);
    }

    let manifest = QuickSnapshotManifest {
        id: snapshot_id.clone(),
        timestamp,
        label,
        file_count: manifest_files.len(),
        total_size: manifest_files.values().sum(),
        files: manifest_files,
    };
    let manifest_path = snapshot_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    let _ = prune_quick_snapshots(context, QUICK_DEFAULT_KEEP);
    Ok(Some(snapshot_id))
}

pub fn list_quick_snapshots(
    context: &HermesContext,
    limit: usize,
) -> Result<Vec<QuickSnapshotManifest>, Box<dyn Error>> {
    let root = quick_snapshot_root(context);
    if !root.is_dir() {
        return Ok(Vec::new());
    }

    let mut entries = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    entries.reverse();

    let mut snapshots = Vec::new();
    for entry in entries {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let manifest_path = path.join("manifest.json");
        let fallback_id = entry.file_name().to_string_lossy().to_string();
        let manifest = match fs::read(&manifest_path) {
            Ok(bytes) => serde_json::from_slice::<QuickSnapshotManifest>(&bytes).unwrap_or(
                QuickSnapshotManifest {
                    id: fallback_id,
                    timestamp: String::new(),
                    label: None,
                    file_count: 0,
                    total_size: 0,
                    files: BTreeMap::new(),
                },
            ),
            Err(_) => QuickSnapshotManifest {
                id: fallback_id,
                timestamp: String::new(),
                label: None,
                file_count: 0,
                total_size: 0,
                files: BTreeMap::new(),
            },
        };
        snapshots.push(manifest);
        if snapshots.len() >= limit {
            break;
        }
    }

    Ok(snapshots)
}

pub fn restore_quick_snapshot(
    context: &HermesContext,
    snapshot_id: &str,
) -> Result<bool, Box<dyn Error>> {
    let snapshot_id = validate_snapshot_id(snapshot_id)?;
    let home = context.hermes_home();
    let manifest_path = quick_snapshot_root(context)
        .join(&snapshot_id)
        .join("manifest.json");
    if !manifest_path.is_file() {
        return Ok(false);
    }

    let manifest: QuickSnapshotManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let snapshot_dir = manifest_path
        .parent()
        .ok_or("snapshot manifest has no parent directory")?;
    let mut restored = 0_usize;

    for relative in manifest.files.keys() {
        let source = snapshot_dir.join(relative);
        if !source.exists() {
            continue;
        }
        let target = home.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        if target
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("db"))
        {
            let temp = unique_temp_path("snapshot-restore", "db");
            fs::copy(&source, &temp)?;
            let _ = fs::remove_file(&target);
            fs::rename(&temp, &target)?;
        } else {
            fs::copy(&source, &target)?;
        }
        if is_secret_file(&target) {
            tighten_secret_permissions(&target)?;
        }
        restored += 1;
    }

    Ok(restored > 0)
}

pub fn prune_quick_snapshots(
    context: &HermesContext,
    keep: usize,
) -> Result<usize, Box<dyn Error>> {
    let root = quick_snapshot_root(context);
    if !root.is_dir() {
        return Ok(0);
    }

    let mut entries = fs::read_dir(root)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|entry| entry.path().is_dir())
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.file_name());
    entries.reverse();

    let mut deleted = 0_usize;
    for entry in entries.into_iter().skip(keep) {
        fs::remove_dir_all(entry.path())?;
        deleted += 1;
    }
    Ok(deleted)
}

fn collect_backup_files(
    root: &Path,
    current: &Path,
    out_path: &Path,
    files: &mut Vec<(PathBuf, PathBuf)>,
    skipped_dirs: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let mut entries = fs::read_dir(current)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let path = entry.path();
        let relative = path.strip_prefix(root)?.to_path_buf();
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        let name = name.to_string_lossy();

        if file_type.is_dir() {
            if EXCLUDED_DIRS.iter().any(|item| *item == name) {
                skipped_dirs.push(relative.to_string_lossy().replace('\\', "/"));
                continue;
            }
            collect_backup_files(root, &path, out_path, files, skipped_dirs)?;
            continue;
        }

        if should_exclude(&relative) || path == out_path {
            continue;
        }
        if file_type.is_file() {
            files.push((path, relative));
        }
    }
    Ok(())
}

fn quick_snapshot_root(context: &HermesContext) -> PathBuf {
    context.hermes_home().join(QUICK_SNAPSHOTS_DIR)
}

fn copy_quick_snapshot_dir(
    home: &Path,
    source_dir: &Path,
    snapshot_dir: &Path,
    manifest_files: &mut BTreeMap<String, u64>,
) -> Result<(), Box<dyn Error>> {
    for entry in source_dir.read_dir()? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            copy_quick_snapshot_dir(home, &path, snapshot_dir, manifest_files)?;
            continue;
        }
        if path.is_file() {
            quick_snapshot_copy_file(home, &path, snapshot_dir, manifest_files)?;
        }
    }
    Ok(())
}

fn quick_snapshot_copy_file(
    home: &Path,
    source: &Path,
    snapshot_dir: &Path,
    manifest_files: &mut BTreeMap<String, u64>,
) -> Result<(), Box<dyn Error>> {
    let relative = source.strip_prefix(home)?;
    let target = snapshot_dir.join(relative);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    if source
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("db"))
    {
        safe_copy_db(source, &target).map_err(io::Error::other)?;
    } else {
        fs::copy(source, &target)?;
    }
    let key = relative.to_string_lossy().replace('\\', "/");
    let size = fs::metadata(&target)?.len();
    manifest_files.insert(key, size);
    Ok(())
}

fn validate_snapshot_label(label: Option<&str>) -> Result<Option<String>, Box<dyn Error>> {
    let Some(label) = label.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    if label.contains(['/', '\\']) {
        return Err("snapshot label cannot contain path separators".into());
    }
    if label == "." || label == ".." {
        return Err("snapshot label cannot be '.' or '..'".into());
    }
    if label.chars().any(char::is_control) {
        return Err("snapshot label cannot contain control characters".into());
    }
    Ok(Some(label.to_string()))
}

fn validate_snapshot_id(snapshot_id: &str) -> Result<String, Box<dyn Error>> {
    let snapshot_id = snapshot_id.trim();
    if snapshot_id.is_empty() {
        return Err("snapshot id cannot be empty".into());
    }
    if snapshot_id.contains(['/', '\\']) {
        return Err("snapshot id cannot contain path separators".into());
    }
    if snapshot_id == "." || snapshot_id == ".." {
        return Err("snapshot id is invalid".into());
    }
    Ok(snapshot_id.to_string())
}

fn should_exclude(relative: &Path) -> bool {
    if relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .any(|part| EXCLUDED_DIRS.iter().any(|item| *item == part))
    {
        return true;
    }
    let Some(name) = relative.file_name().and_then(|value| value.to_str()) else {
        return false;
    };
    EXCLUDED_NAMES.iter().any(|item| *item == name)
        || EXCLUDED_SUFFIXES
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

fn safe_copy_db(src: &Path, dst: &Path) -> Result<(), String> {
    match Connection::open_with_flags(src, OpenFlags::SQLITE_OPEN_READ_ONLY) {
        Ok(source) => match source.backup(DatabaseName::Main, dst, None) {
            Ok(()) => Ok(()),
            Err(error) => fs::copy(src, dst)
                .map(|_| ())
                .map_err(|fallback| format!("{error}; fallback copy failed: {fallback}")),
        },
        Err(error) => fs::copy(src, dst)
            .map(|_| ())
            .map_err(|fallback| format!("{error}; fallback copy failed: {fallback}")),
    }
}

fn validate_backup_names(names: &[String]) -> Result<(), Box<dyn Error>> {
    if names.is_empty() {
        return Err("zip archive is empty".into());
    }
    let markers = ["config.yaml", ".env", "state.db"];
    if names.iter().any(|name| {
        Path::new(name)
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|base| markers.iter().any(|marker| *marker == base))
    }) {
        return Ok(());
    }
    Err(
        "zip does not appear to be a Hermes backup (no config.yaml, .env, or state.db found)"
            .into(),
    )
}

fn detect_prefix(names: &[String]) -> Option<String> {
    let mut first = None::<String>;
    for name in names.iter().filter(|name| !name.ends_with('/')) {
        let path = Path::new(name);
        let mut components = path.components();
        let Some(Component::Normal(value)) = components.next() else {
            return None;
        };
        if components.next().is_none() {
            return None;
        }
        let part = value.to_string_lossy().to_string();
        match first.as_deref() {
            Some(existing) if existing != part => return None,
            None => first = Some(part),
            _ => {}
        }
    }
    match first.as_deref() {
        Some(".hermes") | Some("hermes") => first.map(|value| format!("{value}/")),
        _ => None,
    }
}

fn archive_member_names<R: Read + io::Seek>(
    archive: &mut ZipArchive<R>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut names = Vec::with_capacity(archive.len());
    for index in 0..archive.len() {
        let member = archive.by_index(index)?;
        names.push(member.name().to_string());
    }
    Ok(names)
}

fn strip_prefix(name: &str, prefix: Option<&str>) -> Option<String> {
    match prefix {
        Some(value) => name.strip_prefix(value).map(ToOwned::to_owned),
        None => Some(name.to_string()),
    }
}

fn sanitize_member_path(name: &str) -> Result<PathBuf, &'static str> {
    if name.contains('\\') {
        return Err("path contains unsupported separators");
    }
    let mut cleaned = PathBuf::new();
    for component in Path::new(name).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(value) => cleaned.push(value),
            Component::ParentDir => return Err("path traversal blocked"),
            Component::RootDir | Component::Prefix(_) => return Err("absolute path blocked"),
        }
    }
    if cleaned.as_os_str().is_empty() {
        return Err("empty path");
    }
    Ok(cleaned)
}

fn confirm_import() -> Result<bool, Box<dyn Error>> {
    println!();
    println!("Warning: Target directory already has Hermes configuration.");
    println!("Importing will overwrite existing files with backup contents.");
    println!();
    print!("Continue? [y/N] ");
    io::stdout().flush()?;
    let mut answer = String::new();
    let read = io::stdin().read_line(&mut answer)?;
    if read == 0 {
        return Ok(false);
    }
    let answer = answer.trim().to_ascii_lowercase();
    Ok(answer == "y" || answer == "yes")
}

fn resolve_output_path(
    context: &HermesContext,
    output: Option<&Path>,
) -> Result<PathBuf, Box<dyn Error>> {
    let stamp = Local::now().format("%Y-%m-%d-%H%M%S").to_string();
    let mut path = match output {
        Some(value) => expand_user_path(context.home_dir(), value),
        None => context
            .home_dir()
            .join(format!("hermes-backup-{stamp}.zip")),
    };
    path = absolutize(path)?;
    if path.is_dir() {
        path = path.join(format!("hermes-backup-{stamp}.zip"));
    }
    if !path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("zip"))
    {
        let mut value = OsString::from(path.as_os_str());
        value.push(".zip");
        path = PathBuf::from(value);
    }
    Ok(path)
}

fn resolve_input_path(context: &HermesContext, input: &Path) -> Result<PathBuf, Box<dyn Error>> {
    absolutize(expand_user_path(context.home_dir(), input))
}

fn expand_user_path(home_dir: &Path, path: &Path) -> PathBuf {
    let raw = path.as_os_str().to_string_lossy();
    if raw == "~" {
        return home_dir.to_path_buf();
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return home_dir.join(rest);
    }
    path.to_path_buf()
}

fn absolutize(path: PathBuf) -> Result<PathBuf, Box<dyn Error>> {
    if path.is_absolute() {
        return Ok(path);
    }
    Ok(std::env::current_dir()?.join(path))
}

fn zip_member_name(path: &Path) -> Result<String, Box<dyn Error>> {
    let rendered = path.to_string_lossy().replace('\\', "/");
    if rendered.is_empty() {
        return Err("zip member name cannot be empty".into());
    }
    Ok(rendered)
}

fn is_secret_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| SECRET_FILE_NAMES.iter().any(|item| *item == name))
}

fn tighten_secret_permissions(path: &Path) -> Result<(), Box<dyn Error>> {
    #[cfg(unix)]
    {
        let permissions = fs::Permissions::from_mode(0o600);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

fn unique_temp_path(label: &str, extension: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("hermes-rs-{label}-{unique}.{extension}"))
}

pub(crate) fn format_size(bytes: u64) -> String {
    let mut size = bytes as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if size < 1024.0 || unit == "GB" {
            return if unit == "B" {
                format!("{} {unit}", bytes)
            } else {
                format!("{size:.1} {unit}")
            };
        }
        size /= 1024.0;
    }
    format!("{size:.1} TB")
}

fn display_path(context: &HermesContext, path: &Path) -> String {
    path.strip_prefix(context.home_dir())
        .ok()
        .map(|relative| {
            if relative.as_os_str().is_empty() {
                "~".to_string()
            } else {
                format!("~/{}", relative.display())
            }
        })
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn should_exclude_transient_paths() {
        assert!(should_exclude(Path::new("backups/old.zip")));
        assert!(should_exclude(Path::new("logs/state.db-wal")));
        assert!(should_exclude(Path::new("gateway.pid")));
        assert!(!should_exclude(Path::new("logs/agent.log")));
    }

    #[test]
    fn detect_prefix_for_wrapped_archive() {
        let names = vec![
            ".hermes/config.yaml".to_string(),
            ".hermes/.env".to_string(),
            ".hermes/state.db".to_string(),
        ];
        assert_eq!(detect_prefix(&names).as_deref(), Some(".hermes/"));
    }

    #[test]
    fn backup_and_import_round_trip() {
        let source_home = TempDir::new().unwrap();
        let source_root = source_home.path().join(".hermes");
        fs::create_dir_all(source_root.join("logs")).unwrap();
        fs::create_dir_all(source_root.join("profiles/demo")).unwrap();
        fs::create_dir_all(source_root.join("backups")).unwrap();
        fs::create_dir_all(source_root.join("checkpoints")).unwrap();
        fs::write(
            source_root.join("config.yaml"),
            "model:\n  provider: auto\n",
        )
        .unwrap();
        fs::write(source_root.join(".env"), "OPENAI_API_KEY=test\n").unwrap();
        fs::write(source_root.join("logs/agent.log"), "hello\n").unwrap();
        fs::write(source_root.join("profiles/demo/config.yaml"), "model:\n").unwrap();
        fs::write(source_root.join("backups/nested.zip"), b"skip").unwrap();
        Connection::open(source_root.join("state.db"))
            .unwrap()
            .execute_batch("CREATE TABLE demo (id INTEGER PRIMARY KEY, value TEXT); INSERT INTO demo (value) VALUES ('ok');")
            .unwrap();

        let archive = source_home.path().join("backup.zip");
        let source_context = HermesContext::new(source_home.path());
        print_backup(
            &source_context,
            BackupArgs {
                output: Some(archive.clone()),
                quick: false,
                label: None,
            },
        )
        .unwrap();
        assert!(archive.is_file());

        let dest_home = TempDir::new().unwrap();
        let dest_context = HermesContext::new(dest_home.path());
        print_import(
            &dest_context,
            ImportArgs {
                zipfile: archive.clone(),
                force: true,
            },
        )
        .unwrap();

        let dest_root = dest_home.path().join(".hermes");
        assert_eq!(
            fs::read_to_string(dest_root.join("config.yaml")).unwrap(),
            "model:\n  provider: auto\n"
        );
        assert_eq!(
            fs::read_to_string(dest_root.join("logs/agent.log")).unwrap(),
            "hello\n"
        );
        assert!(!dest_root.join("backups/nested.zip").exists());
        let db = Connection::open(dest_root.join("state.db")).unwrap();
        let value: String = db
            .query_row("SELECT value FROM demo LIMIT 1", [], |row| row.get(0))
            .unwrap();
        assert_eq!(value, "ok");
        #[cfg(unix)]
        {
            let mode = fs::metadata(dest_root.join(".env"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn quick_snapshot_create_list_restore_and_prune() {
        let home = TempDir::new().unwrap();
        let context = HermesContext::new(home.path());
        let root = context.hermes_home().to_path_buf();
        fs::create_dir_all(root.join("cron")).unwrap();
        fs::create_dir_all(root.join("platforms/pairing")).unwrap();
        fs::write(root.join("config.yaml"), "model:\n  provider: auto\n").unwrap();
        fs::write(root.join(".env"), "OPENAI_API_KEY=one\n").unwrap();
        fs::write(root.join("cron/jobs.json"), "{\"jobs\":[]}\n").unwrap();
        fs::write(root.join("platforms/pairing/demo.json"), "{\"ok\":true}\n").unwrap();
        Connection::open(root.join("state.db"))
            .unwrap()
            .execute_batch("CREATE TABLE demo (id INTEGER PRIMARY KEY, value TEXT); INSERT INTO demo (value) VALUES ('before');")
            .unwrap();

        let snapshot_id = create_quick_snapshot(&context, Some("base"))
            .unwrap()
            .unwrap();
        let snapshots = list_quick_snapshots(&context, 20).unwrap();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].id, snapshot_id);
        assert_eq!(snapshots[0].label.as_deref(), Some("base"));
        assert!(snapshots[0].files.contains_key("config.yaml"));
        assert!(snapshots[0].files.contains_key("state.db"));

        fs::write(root.join("config.yaml"), "model:\n  provider: changed\n").unwrap();
        fs::write(root.join(".env"), "OPENAI_API_KEY=two\n").unwrap();
        Connection::open(root.join("state.db"))
            .unwrap()
            .execute_batch("DELETE FROM demo; INSERT INTO demo (value) VALUES ('after');")
            .unwrap();

        assert!(restore_quick_snapshot(&context, &snapshot_id).unwrap());
        assert_eq!(
            fs::read_to_string(root.join("config.yaml")).unwrap(),
            "model:\n  provider: auto\n"
        );
        assert_eq!(
            fs::read_to_string(root.join(".env")).unwrap(),
            "OPENAI_API_KEY=one\n"
        );
        let db = Connection::open(root.join("state.db")).unwrap();
        let value: String = db
            .query_row("SELECT value FROM demo LIMIT 1", [], |row| row.get(0))
            .unwrap();
        assert_eq!(value, "before");

        for label in ["1", "2", "3"] {
            std::thread::sleep(std::time::Duration::from_millis(5));
            create_quick_snapshot(&context, Some(label)).unwrap();
        }
        let deleted = prune_quick_snapshots(&context, 2).unwrap();
        assert!(deleted >= 2);
        let remaining = list_quick_snapshots(&context, 20).unwrap();
        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn import_blocks_path_traversal_entries() {
        let home = TempDir::new().unwrap();
        let archive_path = home.path().join("bad.zip");
        let file = File::create(&archive_path).unwrap();
        let mut writer = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        writer.start_file("config.yaml", options).unwrap();
        writer.write_all(b"model:\n").unwrap();
        writer.start_file("../evil.txt", options).unwrap();
        writer.write_all(b"nope").unwrap();
        writer.finish().unwrap();

        let target_home = TempDir::new().unwrap();
        let context = HermesContext::new(target_home.path());
        print_import(
            &context,
            ImportArgs {
                zipfile: archive_path,
                force: true,
            },
        )
        .unwrap();

        assert!(!target_home.path().join("evil.txt").exists());
        assert!(
            !target_home
                .path()
                .join(".hermes")
                .join("..")
                .join("evil.txt")
                .exists()
        );
        assert!(
            target_home
                .path()
                .join(".hermes")
                .join("config.yaml")
                .exists()
        );
    }
}
