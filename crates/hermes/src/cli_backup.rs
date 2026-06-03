//! Backup and import commands for hermes CLI (native Rust port of
//! `hermes_cli/backup.py`).
//!
//! `hermes backup` creates a zip archive of the entire `~/.hermes/`
//! directory (excluding the hermes-agent repo and transient files).
//!
//! `hermes import` restores from a backup zip, overlaying onto the current
//! HERMES_HOME root.
//!
//! Also provides "quick" state snapshots (used by the `/snapshot` slash
//! command and `hermes backup --quick`) and the pre-update / pre-migration
//! auto-backup helpers.
//!
//! This module is largely self-contained.  The Hermes home/root resolution
//! helpers mirror `hermes_constants.py` (`get_hermes_home`,
//! `get_default_hermes_root`, `display_hermes_home`) so the module does not
//! have to depend on a not-yet-public re-export from `hermes-core`.  If/when
//! those are exported, callers can swap to `crate::mod_hermes_constants::*`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use chrono::{Local, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

// ---------------------------------------------------------------------------
// Exclusion rules
// ---------------------------------------------------------------------------

/// Directory names to skip entirely (matched against each path component).
pub const EXCLUDED_DIRS: &[&str] = &[
    "hermes-agent", // the codebase repo — re-clone instead
    "__pycache__",  // bytecode caches — regenerated on import
    ".git",         // nested git dirs (safety)
    "node_modules", // js deps if website/ somehow leaks in
    "backups",      // prior auto-backups — don't nest backups exponentially
    "checkpoints",  // session-local trajectory caches — regenerated per session
];

/// File-name suffixes to skip.
pub const EXCLUDED_SUFFIXES: &[&str] = &[
    ".pyc",
    ".pyo",
    // SQLite sidecar files — the backup takes a consistent snapshot of `*.db`
    // via the SQLite backup API, so shipping the live WAL / shared-memory /
    // rollback-journal alongside would pair a fresh snapshot with stale
    // sidecar state and produce a torn restore on the next open.
    ".db-wal",
    ".db-shm",
    ".db-journal",
];

/// File names to skip (runtime state that's meaningless on another machine).
pub const EXCLUDED_NAMES: &[&str] = &["gateway.pid", "cron.pid"];

/// zip extraction drops Unix mode bits; restore tightens these to 0600.
pub const SECRET_FILE_NAMES: &[&str] = &[".env", "auth.json", "state.db"];

/// Return true if *rel_path* (relative to hermes root) should be skipped.
pub fn should_exclude(rel_path: &Path) -> bool {
    for comp in rel_path.components() {
        if let Component::Normal(os) = comp {
            if let Some(s) = os.to_str() {
                if EXCLUDED_DIRS.contains(&s) {
                    return true;
                }
            }
        }
    }

    let name = match rel_path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return false,
    };

    if EXCLUDED_NAMES.contains(&name) {
        return true;
    }

    if EXCLUDED_SUFFIXES.iter().any(|suf| name.ends_with(suf)) {
        return true;
    }

    false
}

// ---------------------------------------------------------------------------
// Hermes home / root resolution (mirrors hermes_constants.py)
// ---------------------------------------------------------------------------

fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// Return the Hermes home directory (default: `~/.hermes`).
///
/// Reads the `HERMES_HOME` env var, falls back to `~/.hermes`.
pub fn get_hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    home_dir().join(".hermes")
}

/// Return the root Hermes directory for profile-level operations.
///
/// In standard deployments this is `~/.hermes`.  In Docker/custom deployments
/// where `HERMES_HOME` points outside `~/.hermes` it returns `HERMES_HOME`
/// directly (or the grandparent if it's a `<root>/profiles/<name>` path).
pub fn get_default_hermes_root() -> PathBuf {
    let native_home = home_dir().join(".hermes");
    let env_home = std::env::var("HERMES_HOME").unwrap_or_default();
    if env_home.is_empty() {
        return native_home;
    }
    let env_path = PathBuf::from(&env_home);

    // HERMES_HOME under ~/.hermes (normal or profile mode) -> native_home.
    let env_resolved = env_path.canonicalize().unwrap_or_else(|_| env_path.clone());
    let native_resolved = native_home
        .canonicalize()
        .unwrap_or_else(|_| native_home.clone());
    if env_resolved.starts_with(&native_resolved) {
        return native_home;
    }

    // Docker / custom deployment.  Profile path: <root>/profiles/<name>.
    if let Some(parent) = env_path.parent() {
        if parent.file_name().and_then(|n| n.to_str()) == Some("profiles") {
            if let Some(grandparent) = parent.parent() {
                return grandparent.to_path_buf();
            }
        }
    }

    env_path
}

/// User-friendly display string for the current HERMES_HOME (uses `~/`).
pub fn display_hermes_home() -> String {
    let home = get_hermes_home();
    let user_home = home_dir();
    match home.strip_prefix(&user_home) {
        Ok(rel) => format!("~/{}", rel.to_string_lossy()),
        Err(_) => home.to_string_lossy().into_owned(),
    }
}

// ---------------------------------------------------------------------------
// SQLite safe copy
// ---------------------------------------------------------------------------

/// Copy a SQLite database safely using the backup API.
///
/// Handles WAL mode — produces a consistent snapshot even while the DB is
/// being written to.  Falls back to a raw byte copy on failure.
pub fn safe_copy_db(src: &Path, dst: &Path) -> bool {
    match sqlite_backup(src, dst) {
        Ok(()) => true,
        Err(exc) => {
            log::warn!("SQLite safe copy failed for {}: {}", src.display(), exc);
            match fs::copy(src, dst) {
                Ok(_) => true,
                Err(exc2) => {
                    log::error!("Raw copy also failed for {}: {}", src.display(), exc2);
                    false
                }
            }
        }
    }
}

fn sqlite_backup(src: &Path, dst: &Path) -> Result<(), rusqlite::Error> {
    use rusqlite::{Connection, DatabaseName, OpenFlags};

    let source = Connection::open_with_flags(src, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    source.backup(DatabaseName::Main, dst, None)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Human-readable file size.
pub fn format_size(nbytes: u64) -> String {
    let mut size = nbytes as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if size < 1024.0 {
            if unit == "B" {
                return format!("{} {}", nbytes, unit);
            }
            return format!("{:.1} {}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.1} TB", size)
}

fn file_size(p: &Path) -> u64 {
    fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

/// Whether a path's extension is exactly `db` (mirrors `Path.suffix == ".db"`).
fn is_db(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()) == Some("db")
}

/// Walk a directory tree, pruning the excluded directory names so we do not
/// descend into them (mirrors the in-place `dirnames[:]` pruning of os.walk).
///
/// Returns (absolute, relative-to-root) pairs for every non-excluded file,
/// plus the set of pruned directory paths (relative to root, posix style).
fn collect_files(
    hermes_root: &Path,
    out_path: Option<&Path>,
    record_skipped: bool,
) -> (Vec<(PathBuf, PathBuf)>, BTreeSet<String>) {
    let mut files: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut skipped_dirs: BTreeSet<String> = BTreeSet::new();
    let out_resolved = out_path.map(|p| p.canonicalize().unwrap_or_else(|_| p.to_path_buf()));

    let mut stack: Vec<PathBuf> = vec![hermes_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let mut subdirs: Vec<PathBuf> = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            // Do not follow symlinks (os.walk followlinks=False).
            let meta = match fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };

            if meta.is_dir() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if EXCLUDED_DIRS.contains(&name_str.as_ref()) {
                    if record_skipped {
                        if let Ok(rel) = path.strip_prefix(hermes_root) {
                            skipped_dirs.insert(format!("{}", rel.to_string_lossy()));
                        }
                    }
                    continue;
                }
                subdirs.push(path);
            } else {
                let rel = match path.strip_prefix(hermes_root) {
                    Ok(r) => r.to_path_buf(),
                    Err(_) => continue,
                };
                if should_exclude(&rel) {
                    continue;
                }
                if let Some(ref outr) = out_resolved {
                    if let Ok(fr) = path.canonicalize() {
                        if &fr == outr {
                            continue;
                        }
                    }
                }
                files.push((path, rel));
            }
        }
        // Push subdirs so traversal stays deterministic-ish (order not relied
        // upon for correctness — the file list is collected fully before use).
        stack.extend(subdirs);
    }

    (files, skipped_dirs)
}

fn arcname(rel: &Path) -> String {
    rel.components()
        .filter_map(|c| match c {
            Component::Normal(s) => s.to_str().map(|s| s.to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn zip_write_file(
    zf: &mut ZipWriter<File>,
    src: &Path,
    arc: &str,
    opts: SimpleFileOptions,
) -> io::Result<u64> {
    zf.start_file(arc, opts)?;
    let mut f = File::open(src)?;
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        zf.write_all(&buf[..n])?;
        total += n as u64;
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Backup
// ---------------------------------------------------------------------------

/// Arguments accepted by [`run_backup`].
#[derive(Debug, Default, Clone)]
pub struct BackupArgs {
    pub output: Option<String>,
}

/// Create a zip backup of the Hermes home directory.
///
/// Returns the exit code (0 on success, 1 on hard failure) so the caller can
/// mirror the Python `sys.exit` semantics without panicking.
pub fn run_backup(args: &BackupArgs) -> i32 {
    let hermes_root = get_default_hermes_root();

    if !hermes_root.is_dir() {
        println!(
            "Error: Hermes home directory not found at {}",
            hermes_root.display()
        );
        return 1;
    }

    // Determine output path.
    let mut out_path: PathBuf = if let Some(ref output) = args.output {
        let expanded = expanduser(output);
        let resolved = expanded.canonicalize().unwrap_or(expanded);
        if resolved.is_dir() {
            let stamp = Local::now().format("%Y-%m-%d-%H%M%S").to_string();
            resolved.join(format!("hermes-backup-{}.zip", stamp))
        } else {
            resolved
        }
    } else {
        let stamp = Local::now().format("%Y-%m-%d-%H%M%S").to_string();
        home_dir().join(format!("hermes-backup-{}.zip", stamp))
    };

    // Ensure the suffix is .zip.
    if out_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        != Some("zip".to_string())
    {
        out_path = append_zip_suffix(&out_path);
    }

    // Ensure parent directory exists.
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    println!("Scanning {} ...", display_hermes_home());
    let (files_to_add, skipped_dirs) = collect_files(&hermes_root, Some(&out_path), true);

    if files_to_add.is_empty() {
        println!("No files to back up.");
        return 0;
    }

    let file_count = files_to_add.len();
    println!("Backing up {} files ...", file_count);

    let mut total_bytes: u64 = 0;
    let mut errors: Vec<String> = Vec::new();
    let t0 = Instant::now();

    let zf_file = match File::create(&out_path) {
        Ok(f) => f,
        Err(exc) => {
            println!("Error: could not create {}: {}", out_path.display(), exc);
            return 1;
        }
    };
    let mut zf = ZipWriter::new(zf_file);
    let opts = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(6i64));

    for (i, (abs_path, rel_path)) in files_to_add.iter().enumerate() {
        let arc = arcname(rel_path);
        let res: io::Result<u64> = if is_db(abs_path) {
            // Safe copy for SQLite databases (handles WAL mode).
            match make_tempfile("db") {
                Ok(tmp_db) => {
                    if safe_copy_db(abs_path, &tmp_db) {
                        let r = zip_write_file(&mut zf, &tmp_db, &arc, opts).map(|_| {
                            let sz = file_size(&tmp_db);
                            sz
                        });
                        let _ = fs::remove_file(&tmp_db);
                        r
                    } else {
                        let _ = fs::remove_file(&tmp_db);
                        errors.push(format!("  {}: SQLite safe copy failed", arc));
                        continue;
                    }
                }
                Err(exc) => {
                    errors.push(format!("  {}: {}", arc, exc));
                    continue;
                }
            }
        } else {
            zip_write_file(&mut zf, abs_path, &arc, opts).map(|_| file_size(abs_path))
        };

        match res {
            Ok(sz) => total_bytes += sz,
            Err(exc) => {
                errors.push(format!("  {}: {}", arc, exc));
                continue;
            }
        }

        let idx = i + 1;
        if idx % 500 == 0 {
            println!("  {}/{} files ...", idx, file_count);
        }
    }

    if let Err(exc) = zf.finish() {
        println!("Error: failed to finalize zip: {}", exc);
        return 1;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let zip_size = file_size(&out_path);

    println!();
    println!("Backup complete: {}", out_path.display());
    println!("  Files:       {}", file_count);
    println!("  Original:    {}", format_size(total_bytes));
    println!("  Compressed:  {}", format_size(zip_size));
    println!("  Time:        {:.1}s", elapsed);

    if !skipped_dirs.is_empty() {
        println!("\n  Excluded directories:");
        for d in &skipped_dirs {
            println!("    {}/", d);
        }
    }

    if !errors.is_empty() {
        println!("\n  Warnings ({} files skipped):", errors.len());
        for e in errors.iter().take(10) {
            println!("{}", e);
        }
        if errors.len() > 10 {
            println!("  ... and {} more", errors.len() - 10);
        }
    }

    let out_name = out_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| out_path.to_string_lossy().into_owned());
    println!("\nRestore with: hermes import {}", out_name);
    0
}

fn expanduser(p: &str) -> PathBuf {
    if let Some(stripped) = p.strip_prefix("~/") {
        home_dir().join(stripped)
    } else if p == "~" {
        home_dir()
    } else {
        PathBuf::from(p)
    }
}

/// Mirror `Path.with_suffix(suffix + ".zip")` — append `.zip` to the final
/// component, preserving any existing extension as part of the stem+suffix.
fn append_zip_suffix(p: &Path) -> PathBuf {
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let new_name = format!("{}.zip", name);
    match p.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(new_name),
        _ => PathBuf::from(new_name),
    }
}

fn make_tempfile(suffix: &str) -> io::Result<PathBuf> {
    let dir = std::env::temp_dir();
    // Use a monotonic-ish nonce; uniqueness is best-effort, matching Python's
    // NamedTemporaryFile (which also relies on the OS for atomic creation).
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let path = dir.join(format!("hermes-backup-{}.{}", nonce, suffix));
    // Touch the file so callers can stat it even before writing.
    File::create(&path)?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/// Arguments accepted by [`run_import`].
#[derive(Debug, Default, Clone)]
pub struct ImportArgs {
    pub zipfile: String,
    pub force: bool,
}

/// Check that a zip looks like a Hermes backup. Returns (ok, reason).
pub fn validate_backup_zip(names: &[String]) -> (bool, String) {
    if names.is_empty() {
        return (false, "zip archive is empty".to_string());
    }

    let markers = ["config.yaml", ".env", "state.db"];
    let mut found = false;
    for n in names {
        let basename = Path::new(n)
            .file_name()
            .and_then(|b| b.to_str())
            .unwrap_or("");
        if markers.contains(&basename) {
            found = true;
            break;
        }
    }

    if !found {
        return (
            false,
            "zip does not appear to be a Hermes backup (no config.yaml, .env, or state databases found)".to_string(),
        );
    }

    (true, String::new())
}

/// Detect a common directory prefix wrapping all entries.
///
/// Some tools zip as `.hermes/config.yaml` instead of `config.yaml`.  Returns
/// the prefix to strip (with trailing `/`), or empty string if none.
pub fn detect_prefix(names: &[String]) -> String {
    let entries: Vec<&String> = names.iter().filter(|n| !n.ends_with('/')).collect();
    if entries.is_empty() {
        return String::new();
    }

    let mut first_parts: BTreeSet<String> = BTreeSet::new();
    for n in &entries {
        let parts: Vec<&str> = Path::new(n)
            .components()
            .filter_map(|c| match c {
                Component::Normal(s) => s.to_str(),
                _ => None,
            })
            .collect();
        if parts.len() > 1 {
            first_parts.insert(parts[0].to_string());
        }
    }

    if first_parts.len() == 1 {
        let prefix = first_parts.into_iter().next().unwrap();
        if prefix == ".hermes" || prefix == "hermes" {
            return format!("{}/", prefix);
        }
    }

    String::new()
}

/// Restore a Hermes backup from a zip file. Returns an exit code.
pub fn run_import(args: &ImportArgs) -> i32 {
    let zip_path = {
        let expanded = expanduser(&args.zipfile);
        expanded.canonicalize().unwrap_or(expanded)
    };

    if !zip_path.is_file() {
        println!("Error: File not found: {}", zip_path.display());
        return 1;
    }

    let file = match File::open(&zip_path) {
        Ok(f) => f,
        Err(_) => {
            println!("Error: Not a valid zip file: {}", zip_path.display());
            return 1;
        }
    };
    let mut archive = match ZipArchive::new(file) {
        Ok(a) => a,
        Err(_) => {
            println!("Error: Not a valid zip file: {}", zip_path.display());
            return 1;
        }
    };

    let hermes_root = get_default_hermes_root();

    let names: Vec<String> = (0..archive.len())
        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
        .collect();

    let (ok, reason) = validate_backup_zip(&names);
    if !ok {
        println!("Error: {}", reason);
        return 1;
    }

    let prefix = detect_prefix(&names);
    let members: Vec<String> = names.iter().filter(|n| !n.ends_with('/')).cloned().collect();
    let file_count = members.len();

    println!("Backup contains {} files", file_count);
    println!("Target: {}", display_hermes_home());

    if !prefix.is_empty() {
        println!("Detected archive prefix: '{}' (will be stripped)", prefix);
    }

    let has_config = hermes_root.join("config.yaml").exists();
    let has_env = hermes_root.join(".env").exists();

    if (has_config || has_env) && !args.force {
        println!();
        println!("Warning: Target directory already has Hermes configuration.");
        println!("Importing will overwrite existing files with backup contents.");
        println!();
        match prompt_continue() {
            Some(answer) => {
                let a = answer.trim().to_lowercase();
                if a != "y" && a != "yes" {
                    println!("Aborted.");
                    return 0;
                }
            }
            None => {
                println!("\nAborted.");
                return 1;
            }
        }
    }

    println!("\nImporting {} files ...", file_count);
    let _ = fs::create_dir_all(&hermes_root);

    let mut errors: Vec<String> = Vec::new();
    let mut restored: usize = 0;
    let t0 = Instant::now();

    let root_resolved = hermes_root
        .canonicalize()
        .unwrap_or_else(|_| hermes_root.clone());

    for member in &members {
        let rel: String = if !prefix.is_empty() && member.starts_with(&prefix) {
            member[prefix.len()..].to_string()
        } else {
            member.clone()
        };

        if rel.is_empty() {
            continue;
        }

        let target = hermes_root.join(&rel);

        // Security: reject absolute paths and traversals.
        if !is_within_root(&target, &root_resolved) {
            errors.push(format!("  {}: path traversal blocked", rel));
            continue;
        }

        let write_result = (|| -> io::Result<()> {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            let idx = match archive.index_for_name(member) {
                Some(i) => i,
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        "member not found",
                    ))
                }
            };
            let mut src = archive
                .by_index(idx)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            let mut dst = File::create(&target)?;
            io::copy(&mut src, &mut dst)?;
            Ok(())
        })();

        match write_result {
            Ok(()) => {
                let name = target
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                if SECRET_FILE_NAMES.contains(&name) {
                    set_mode_600(&target);
                }
                restored += 1;
            }
            Err(exc) => {
                errors.push(format!("  {}: {}", rel, exc));
            }
        }

        if restored != 0 && restored % 500 == 0 {
            println!("  {}/{} files ...", restored, file_count);
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();

    println!();
    println!(
        "Import complete: {} files restored in {:.1}s",
        restored, elapsed
    );
    println!("  Target: {}", display_hermes_home());

    if !errors.is_empty() {
        println!("\n  Warnings ({} files skipped):", errors.len());
        for e in errors.iter().take(10) {
            println!("{}", e);
        }
        if errors.len() > 10 {
            println!("  ... and {} more", errors.len() - 10);
        }
    }

    // Post-import: profile wrapper-script restoration is handled by the Python
    // `hermes_cli.profiles` integration which is not yet ported.  We detect the
    // profiles directory and surface guidance, mirroring the ImportError branch.
    let profiles_dir = hermes_root.join("profiles");
    let mut restored_profiles: Vec<(String, bool)> = Vec::new();
    if profiles_dir.is_dir() {
        // Equivalent of the Python ImportError path: profiles exist but we
        // can't (yet) wire wrapper scripts from native Rust.
        if dir_has_any_entry(&profiles_dir) {
            // Collect profile names eligible for wrapper restoration so the
            // gateway-services hint below can enumerate them.
            if let Ok(entries) = fs::read_dir(&profiles_dir) {
                let mut names: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
                names.sort();
                for entry in names {
                    if !entry.is_dir() {
                        continue;
                    }
                    let has_cfg = entry.join("config.yaml").exists();
                    let has_e = entry.join(".env").exists();
                    if !has_cfg && !has_e {
                        continue;
                    }
                    if let Some(pname) = entry.file_name().and_then(|n| n.to_str()) {
                        restored_profiles.push((pname.to_string(), false));
                    }
                }
            }
            println!("\n  Profiles detected but aliases could not be created.");
            println!("  Run: hermes profile list  (after installing hermes)");
        }
    }

    println!();
    if !hermes_root.join("hermes-agent").is_dir() {
        println!("Note: The hermes-agent codebase was not included in the backup.");
        println!("  If this is a fresh install, run: hermes update");
    }

    if !restored_profiles.is_empty() {
        println!("\nTo re-enable gateway services for profiles:");
        for (pname, _) in &restored_profiles {
            println!("  hermes -p {} gateway install", pname);
        }
    }

    println!("Done. Your Hermes configuration has been restored.");
    0
}

fn prompt_continue() -> Option<String> {
    use std::io::BufRead;
    print!("Continue? [y/N] ");
    let _ = io::stdout().flush();
    let stdin = io::stdin();
    let mut line = String::new();
    match stdin.lock().read_line(&mut line) {
        Ok(0) => None, // EOF
        Ok(_) => Some(line),
        Err(_) => None,
    }
}

/// Mirror Python's `target.resolve().relative_to(hermes_root.resolve())`:
/// resolve the target to a canonical form anchored at the canonical root, then
/// check containment.  Targets need not exist yet (we resolve the longest
/// existing ancestor and lexically normalise the remainder).
fn is_within_root(target: &Path, root_resolved: &Path) -> bool {
    let resolved = resolve_target(target);
    resolved.starts_with(root_resolved)
}

/// Resolve `target` even when it (or its tail) does not exist: canonicalize the
/// deepest existing ancestor, then re-append and lexically normalise the rest.
fn resolve_target(target: &Path) -> PathBuf {
    if let Ok(p) = target.canonicalize() {
        return p;
    }
    // Find the deepest existing ancestor.
    let mut existing = target.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if existing.exists() {
            break;
        }
        match existing.parent() {
            Some(parent) if parent != existing => {
                if let Some(name) = existing.file_name() {
                    tail.push(name.to_os_string());
                }
                existing = parent.to_path_buf();
            }
            _ => break,
        }
    }
    let base = existing.canonicalize().unwrap_or(existing);
    let mut out = base;
    for seg in tail.into_iter().rev() {
        out.push(seg);
    }
    lexical_normalize(&out)
}

/// Lexical path normalisation (resolves `.`/`..` components without touching
/// the filesystem).  `..` at the filesystem root is clamped (matching POSIX
/// `realpath`/Python `resolve`), so `/a/../../etc` becomes `/etc`.
fn lexical_normalize(p: &Path) -> PathBuf {
    let mut out: Vec<Component> = Vec::new();
    for comp in p.components() {
        match comp {
            Component::ParentDir => match out.last() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                // Clamp at root / prefix: drop the `..` entirely.
                Some(Component::RootDir) | Some(Component::Prefix(_)) => {}
                _ => out.push(comp),
            },
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out.iter().collect()
}

#[cfg(unix)]
fn set_mode_600(p: &Path) {
    let _ = fs::set_permissions(p, fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn set_mode_600(_p: &Path) {}

fn dir_has_any_entry(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|mut e| e.next().is_some())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Quick state snapshots
// ---------------------------------------------------------------------------

/// Critical state files to include in quick snapshots (relative to HERMES_HOME).
pub const QUICK_STATE_FILES: &[&str] = &[
    "state.db",
    "config.yaml",
    ".env",
    "auth.json",
    "cron/jobs.json",
    "gateway_state.json",
    "channel_directory.json",
    "processes.json",
    // Pairing stores (generic + per-platform JSONs outside state.db).
    "pairing",
    "platforms/pairing",
    "feishu_comment_pairing.json",
];

const QUICK_SNAPSHOTS_DIR: &str = "state-snapshots";
pub const QUICK_DEFAULT_KEEP: usize = 20;

fn quick_snapshot_root(hermes_home: Option<&Path>) -> PathBuf {
    let home = hermes_home
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_hermes_home);
    home.join(QUICK_SNAPSHOTS_DIR)
}

/// Snapshot manifest written to `manifest.json`.
#[derive(Debug, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub id: String,
    pub timestamp: String,
    pub label: Option<String>,
    pub file_count: usize,
    pub total_size: u64,
    pub files: BTreeMap<String, u64>,
}

/// Create a quick state snapshot of critical files.
///
/// Copies the quick-state file set to a timestamped directory under
/// `state-snapshots/` and auto-prunes old snapshots.  Returns the snapshot ID,
/// or `None` if no files were found.
pub fn create_quick_snapshot(label: Option<&str>, hermes_home: Option<&Path>) -> Option<String> {
    let home = hermes_home
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_hermes_home);
    let root = quick_snapshot_root(Some(&home));

    let ts = Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let snap_id = match label {
        Some(l) => format!("{}-{}", ts, l),
        None => ts.clone(),
    };
    let snap_dir = root.join(&snap_id);
    let _ = fs::create_dir_all(&snap_dir);

    let mut manifest: BTreeMap<String, u64> = BTreeMap::new();

    for rel in QUICK_STATE_FILES {
        let src = home.join(rel);
        if !src.exists() {
            continue;
        }

        if src.is_dir() {
            // Walk the directory; record each file individually so restore can
            // treat them uniformly.
            let mut stack = vec![src.clone()];
            while let Some(dir) = stack.pop() {
                let entries = match fs::read_dir(&dir) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        stack.push(path);
                        continue;
                    }
                    if !path.is_file() {
                        continue;
                    }
                    let sub_rel = match path.strip_prefix(&home) {
                        Ok(r) => arcname(r),
                        Err(_) => continue,
                    };
                    let dst = snap_dir.join(&sub_rel);
                    if let Some(parent) = dst.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    match fs::copy(&path, &dst) {
                        Ok(_) => {
                            manifest.insert(sub_rel, file_size(&dst));
                        }
                        Err(exc) => {
                            log::warn!("Could not snapshot {}: {}", sub_rel, exc);
                        }
                    }
                }
            }
            continue;
        }

        if !src.is_file() {
            continue;
        }

        let dst = snap_dir.join(rel);
        if let Some(parent) = dst.parent() {
            let _ = fs::create_dir_all(parent);
        }

        if is_db(&src) {
            if !safe_copy_db(&src, &dst) {
                continue;
            }
            manifest.insert((*rel).to_string(), file_size(&dst));
        } else {
            match fs::copy(&src, &dst) {
                Ok(_) => {
                    manifest.insert((*rel).to_string(), file_size(&dst));
                }
                Err(exc) => {
                    log::warn!("Could not snapshot {}: {}", rel, exc);
                }
            }
        }
    }

    if manifest.is_empty() {
        let _ = fs::remove_dir_all(&snap_dir);
        return None;
    }

    let total_size: u64 = manifest.values().sum();
    let meta = SnapshotManifest {
        id: snap_id.clone(),
        timestamp: ts,
        label: label.map(|s| s.to_string()),
        file_count: manifest.len(),
        total_size,
        files: manifest.clone(),
    };
    if let Ok(json) = serde_json::to_string_pretty(&meta) {
        let _ = fs::write(snap_dir.join("manifest.json"), json);
    }

    prune_quick_snapshots_inner(&root, QUICK_DEFAULT_KEEP);

    log::info!(
        "State snapshot created: {} ({} files)",
        snap_id,
        manifest.len()
    );
    Some(snap_id)
}

/// List existing quick state snapshots, most recent first.
///
/// Returns the parsed manifest JSON values (or a minimal fallback for snapshots
/// with an unreadable manifest).
pub fn list_quick_snapshots(limit: usize, hermes_home: Option<&Path>) -> Vec<Value> {
    let root = quick_snapshot_root(hermes_home);
    if !root.exists() {
        return Vec::new();
    }

    let mut dirs: Vec<PathBuf> = match fs::read_dir(&root) {
        Ok(e) => e.flatten().map(|x| x.path()).collect(),
        Err(_) => return Vec::new(),
    };
    // Sort by name descending (reverse=True on sorted(iterdir())).
    dirs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    let mut results: Vec<Value> = Vec::new();
    for d in dirs {
        if !d.is_dir() {
            continue;
        }
        let manifest_path = d.join("manifest.json");
        if manifest_path.exists() {
            match fs::read_to_string(&manifest_path)
                .ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            {
                Some(v) => results.push(v),
                None => {
                    let name = d
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();
                    results.push(serde_json::json!({
                        "id": name,
                        "file_count": 0,
                        "total_size": 0,
                    }));
                }
            }
        }
        if results.len() >= limit {
            break;
        }
    }

    results
}

/// Restore state from a quick snapshot.  Returns true if at least one file was
/// restored.
pub fn restore_quick_snapshot(snapshot_id: &str, hermes_home: Option<&Path>) -> bool {
    let home = hermes_home
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_hermes_home);
    let root = quick_snapshot_root(Some(&home));
    let snap_dir = root.join(snapshot_id);

    if !snap_dir.is_dir() {
        return false;
    }

    let manifest_path = snap_dir.join("manifest.json");
    if !manifest_path.exists() {
        return false;
    }

    let meta: Value = match fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
    {
        Some(v) => v,
        None => return false,
    };

    let files = match meta.get("files").and_then(|f| f.as_object()) {
        Some(m) => m,
        None => return false,
    };

    let mut restored = 0usize;
    for rel in files.keys() {
        let src = snap_dir.join(rel);
        if !src.exists() {
            continue;
        }

        let dst = home.join(rel);
        if let Some(parent) = dst.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let result = if is_db(&dst) {
            // Atomic-ish replace for databases.
            let dst_name = dst
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("db")
                .to_string();
            let tmp = dst
                .parent()
                .map(|p| p.join(format!(".{}.snap_restore", dst_name)))
                .unwrap_or_else(|| PathBuf::from(format!(".{}.snap_restore", dst_name)));
            (|| -> io::Result<()> {
                fs::copy(&src, &tmp)?;
                let _ = fs::remove_file(&dst);
                fs::rename(&tmp, &dst)?;
                Ok(())
            })()
        } else {
            fs::copy(&src, &dst).map(|_| ())
        };

        match result {
            Ok(()) => restored += 1,
            Err(exc) => {
                log::error!("Failed to restore {}: {}", rel, exc);
            }
        }
    }

    log::info!("Restored {} files from snapshot {}", restored, snapshot_id);
    restored > 0
}

fn prune_quick_snapshots_inner(root: &Path, keep: usize) -> usize {
    if !root.exists() {
        return 0;
    }

    let mut dirs: Vec<PathBuf> = match fs::read_dir(root) {
        Ok(e) => e.flatten().map(|x| x.path()).filter(|p| p.is_dir()).collect(),
        Err(_) => return 0,
    };
    // Sort by name descending.
    dirs.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    let mut deleted = 0;
    for d in dirs.into_iter().skip(keep) {
        match fs::remove_dir_all(&d) {
            Ok(()) => deleted += 1,
            Err(exc) => {
                let name = d.file_name().and_then(|n| n.to_str()).unwrap_or("");
                log::warn!("Failed to prune snapshot {}: {}", name, exc);
            }
        }
    }

    deleted
}

/// Manually prune quick snapshots. Returns count deleted.
pub fn prune_quick_snapshots(keep: usize, hermes_home: Option<&Path>) -> usize {
    prune_quick_snapshots_inner(&quick_snapshot_root(hermes_home), keep)
}

/// CLI entry point for `hermes backup --quick`.
pub fn run_quick_backup(label: Option<&str>) -> i32 {
    match create_quick_snapshot(label, None) {
        Some(snap_id) => {
            println!("State snapshot created: {}", snap_id);
            let snaps = list_quick_snapshots(20, None);
            println!(
                "  {} snapshot(s) stored in {}/state-snapshots/",
                snaps.len(),
                display_hermes_home()
            );
            println!("  Restore with: /snapshot restore {}", snap_id);
        }
        None => {
            println!("No state files found to snapshot.");
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Shared full-zip backup helper
// ---------------------------------------------------------------------------

/// Write a full zip snapshot of `hermes_root` to `out_path`.
///
/// Uses the same exclusion rules and SQLite safe-copy as [`run_backup`].
/// Returns the output path on success, `None` on failure (nothing to back up,
/// or write error — caller should surface the outcome but not panic).
pub fn write_full_zip_backup(out_path: &Path, hermes_root: &Path) -> Option<PathBuf> {
    let (files_to_add, _) = collect_files(hermes_root, Some(out_path), false);

    if files_to_add.is_empty() {
        return None;
    }

    let zf_file = match File::create(out_path) {
        Ok(f) => f,
        Err(exc) => {
            log::warn!("Full-zip backup: zip write failed: {}", exc);
            return None;
        }
    };
    let mut zf = ZipWriter::new(zf_file);
    let opts = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(6i64));

    for (abs_path, rel_path) in &files_to_add {
        let arc = arcname(rel_path);
        let res: io::Result<()> = if is_db(abs_path) {
            match make_tempfile("db") {
                Ok(tmp_db) => {
                    let r = if safe_copy_db(abs_path, &tmp_db) {
                        zip_write_file(&mut zf, &tmp_db, &arc, opts).map(|_| ())
                    } else {
                        Ok(())
                    };
                    let _ = fs::remove_file(&tmp_db);
                    r
                }
                Err(_) => Ok(()),
            }
        } else {
            zip_write_file(&mut zf, abs_path, &arc, opts).map(|_| ())
        };

        if let Err(exc) = res {
            log::debug!("Skipping {} in zip backup: {}", arc, exc);
            continue;
        }
    }

    if let Err(exc) = zf.finish() {
        log::warn!("Full-zip backup: zip write failed: {}", exc);
        let _ = fs::remove_file(out_path);
        return None;
    }

    Some(out_path.to_path_buf())
}

// ---------------------------------------------------------------------------
// Pre-update auto-backup
// ---------------------------------------------------------------------------

const PRE_UPDATE_BACKUPS_DIR: &str = "backups";
const PRE_UPDATE_PREFIX: &str = "pre-update-";
pub const PRE_UPDATE_DEFAULT_KEEP: usize = 5;

const PRE_MIGRATION_PREFIX: &str = "pre-migration-";
pub const PRE_MIGRATION_DEFAULT_KEEP: usize = 5;

fn pre_update_backup_dir(hermes_home: Option<&Path>) -> PathBuf {
    let home = hermes_home
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_hermes_home);
    home.join(PRE_UPDATE_BACKUPS_DIR)
}

/// Remove oldest pre-update backups beyond the keep limit. Only touches files
/// matching `pre-update-*.zip`.  `keep` is floored to 1.
fn prune_prefixed_backups(backup_dir: &Path, prefix: &str, mut keep: usize, floor_one: bool) -> usize {
    if floor_one && keep < 1 {
        keep = 1;
    }
    if !backup_dir.exists() {
        return 0;
    }

    let mut backups: Vec<PathBuf> = match fs::read_dir(backup_dir) {
        Ok(e) => e
            .flatten()
            .map(|x| x.path())
            .filter(|p| {
                p.is_file()
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with(prefix))
                        .unwrap_or(false)
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.eq_ignore_ascii_case("zip"))
                        .unwrap_or(false)
            })
            .collect(),
        Err(_) => return 0,
    };
    backups.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    let mut deleted = 0;
    for p in backups.into_iter().skip(keep) {
        match fs::remove_file(&p) {
            Ok(()) => deleted += 1,
            Err(exc) => {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                log::warn!("Failed to prune backup {}: {}", name, exc);
            }
        }
    }

    deleted
}

/// Create a full zip backup of HERMES_HOME under `backups/` and auto-prune old
/// pre-update backups.  Returns the path to the created zip, or `None`.  Never
/// panics — the caller (`hermes update`) should continue even if it fails.
pub fn create_pre_update_backup(hermes_home: Option<&Path>, keep: usize) -> Option<PathBuf> {
    let hermes_root = hermes_home
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_default_hermes_root);
    if !hermes_root.is_dir() {
        return None;
    }

    let backup_dir = pre_update_backup_dir(Some(&hermes_root));
    if let Err(exc) = fs::create_dir_all(&backup_dir) {
        log::warn!(
            "Could not create pre-update backup dir {}: {}",
            backup_dir.display(),
            exc
        );
        return None;
    }

    let stamp = Local::now().format("%Y-%m-%d-%H%M%S").to_string();
    let out_path = backup_dir.join(format!("{}{}.zip", PRE_UPDATE_PREFIX, stamp));

    write_full_zip_backup(&out_path, &hermes_root)?;
    prune_prefixed_backups(&backup_dir, PRE_UPDATE_PREFIX, keep, true);
    Some(out_path)
}

/// Create a full zip backup of HERMES_HOME under `backups/` before a
/// `hermes claw migrate` apply, and auto-prune old pre-migration backups.
pub fn create_pre_migration_backup(hermes_home: Option<&Path>, keep: usize) -> Option<PathBuf> {
    let hermes_root = hermes_home
        .map(|p| p.to_path_buf())
        .unwrap_or_else(get_default_hermes_root);
    if !hermes_root.is_dir() {
        return None;
    }

    // Reuses the shared backups/ directory.
    let backup_dir = pre_update_backup_dir(Some(&hermes_root));
    if let Err(exc) = fs::create_dir_all(&backup_dir) {
        log::warn!(
            "Could not create pre-migration backup dir {}: {}",
            backup_dir.display(),
            exc
        );
        return None;
    }

    let stamp = Local::now().format("%Y-%m-%d-%H%M%S").to_string();
    let out_path = backup_dir.join(format!("{}{}.zip", PRE_MIGRATION_PREFIX, stamp));

    write_full_zip_backup(&out_path, &hermes_root)?;
    // pre-migration keep is floored to 0 (allows removing all).
    prune_prefixed_backups(&backup_dir, PRE_MIGRATION_PREFIX, keep, false);
    Some(out_path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn format_size_units() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1024 * 1024), "1.0 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_size(1024u64 * 1024 * 1024 * 1024), "1.0 TB");
    }

    #[test]
    fn should_exclude_matches() {
        assert!(should_exclude(Path::new("hermes-agent/foo.txt")));
        assert!(should_exclude(Path::new("sub/__pycache__/x.pyc")));
        assert!(should_exclude(Path::new("foo.pyc")));
        assert!(should_exclude(Path::new("state.db-wal")));
        assert!(should_exclude(Path::new("state.db-shm")));
        assert!(should_exclude(Path::new("a/state.db-journal")));
        assert!(should_exclude(Path::new("gateway.pid")));
        assert!(should_exclude(Path::new("sub/cron.pid")));
        assert!(should_exclude(Path::new("checkpoints/sess/x")));

        assert!(!should_exclude(Path::new("config.yaml")));
        assert!(!should_exclude(Path::new("state.db")));
        assert!(!should_exclude(Path::new(".env")));
        assert!(!should_exclude(Path::new("sub/data.json")));
    }

    #[test]
    fn validate_backup_zip_logic() {
        let (ok, reason) = validate_backup_zip(&[]);
        assert!(!ok);
        assert_eq!(reason, "zip archive is empty");

        let (ok, _) = validate_backup_zip(&["random.txt".to_string()]);
        assert!(!ok);

        let (ok, _) = validate_backup_zip(&["config.yaml".to_string()]);
        assert!(ok);

        let (ok, _) = validate_backup_zip(&[".hermes/state.db".to_string()]);
        assert!(ok);

        let (ok, _) = validate_backup_zip(&["sub/.env".to_string()]);
        assert!(ok);
    }

    #[test]
    fn detect_prefix_logic() {
        assert_eq!(detect_prefix(&[]), "");
        // single root dir that is a hermes name
        assert_eq!(
            detect_prefix(&[
                ".hermes/config.yaml".to_string(),
                ".hermes/state.db".to_string()
            ]),
            ".hermes/"
        );
        assert_eq!(
            detect_prefix(&[
                "hermes/config.yaml".to_string(),
                "hermes/a/b.txt".to_string()
            ]),
            "hermes/"
        );
        // not a hermes-name prefix -> no strip
        assert_eq!(
            detect_prefix(&["other/config.yaml".to_string()]),
            ""
        );
        // multiple roots -> no strip
        assert_eq!(
            detect_prefix(&[
                ".hermes/config.yaml".to_string(),
                "other/state.db".to_string()
            ]),
            ""
        );
        // top-level files only (len == 1) -> no common prefix
        assert_eq!(
            detect_prefix(&["config.yaml".to_string(), "state.db".to_string()]),
            ""
        );
    }

    #[test]
    fn append_zip_suffix_logic() {
        assert_eq!(
            append_zip_suffix(Path::new("/tmp/backup")),
            PathBuf::from("/tmp/backup.zip")
        );
        // Python with_suffix(suffix + ".zip") appends to the existing name.
        assert_eq!(
            append_zip_suffix(Path::new("/tmp/backup.tar")),
            PathBuf::from("/tmp/backup.tar.zip")
        );
        assert_eq!(
            append_zip_suffix(Path::new("backup")),
            PathBuf::from("backup.zip")
        );
    }

    #[test]
    fn arcname_uses_forward_slashes() {
        assert_eq!(arcname(Path::new("a/b/c.txt")), "a/b/c.txt");
        assert_eq!(arcname(Path::new("config.yaml")), "config.yaml");
    }

    #[test]
    fn lexical_normalize_resolves_traversal() {
        assert_eq!(
            lexical_normalize(Path::new("/root/a/../b")),
            PathBuf::from("/root/b")
        );
        assert_eq!(
            lexical_normalize(Path::new("/root/../../etc")),
            PathBuf::from("/etc")
        );
    }

    #[test]
    fn is_within_root_blocks_traversal() {
        let root = std::env::temp_dir().join("hermes-within-root-test");
        let _ = fs::create_dir_all(&root);
        let root_resolved = root.canonicalize().unwrap_or(root.clone());

        let good = root.join("sub/file.txt");
        assert!(is_within_root(&good, &root_resolved));

        let bad = root.join("../escape.txt");
        assert!(!is_within_root(&bad, &root_resolved));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn get_default_hermes_root_env() {
        // Save and restore HERMES_HOME around the test.
        let saved = std::env::var("HERMES_HOME").ok();

        unsafe {
            std::env::set_var("HERMES_HOME", "/opt/custom-data");
        }
        // Non-existent custom path that isn't under ~/.hermes -> itself.
        assert_eq!(
            get_default_hermes_root(),
            PathBuf::from("/opt/custom-data")
        );

        // Docker-style profile path: <root>/profiles/<name> -> <root>
        unsafe {
            std::env::set_var("HERMES_HOME", "/opt/data/profiles/coder");
        }
        assert_eq!(get_default_hermes_root(), PathBuf::from("/opt/data"));

        // Restore.
        unsafe {
            match saved {
                Some(v) => std::env::set_var("HERMES_HOME", v),
                None => std::env::remove_var("HERMES_HOME"),
            }
        }
    }

    #[test]
    fn quick_snapshot_roundtrip() {
        let base = std::env::temp_dir().join(format!(
            "hermes-quick-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let home = base.join("home");
        fs::create_dir_all(&home).unwrap();

        // Create a couple of quick-state files.
        fs::write(home.join("config.yaml"), b"key: value\n").unwrap();
        fs::write(home.join(".env"), b"SECRET=1\n").unwrap();

        let snap_id =
            create_quick_snapshot(Some("test"), Some(&home)).expect("snapshot created");
        assert!(snap_id.ends_with("-test"));

        let snaps = list_quick_snapshots(20, Some(&home));
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0]["file_count"].as_u64(), Some(2));

        // Mutate then restore.
        fs::write(home.join("config.yaml"), b"changed\n").unwrap();
        assert!(restore_quick_snapshot(&snap_id, Some(&home)));
        let restored = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert_eq!(restored, "key: value\n");

        // No-files case returns None.
        let empty_home = base.join("empty");
        fs::create_dir_all(&empty_home).unwrap();
        assert!(create_quick_snapshot(None, Some(&empty_home)).is_none());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn prune_quick_keeps_newest() {
        let base = std::env::temp_dir().join(format!(
            "hermes-prune-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let root = base.join("state-snapshots");
        fs::create_dir_all(&root).unwrap();

        for name in ["20240101-000001", "20240101-000002", "20240101-000003"] {
            fs::create_dir_all(root.join(name)).unwrap();
        }

        let deleted = prune_quick_snapshots_inner(&root, 2);
        assert_eq!(deleted, 1);
        // The oldest (000001) should be gone; newest two remain.
        assert!(!root.join("20240101-000001").exists());
        assert!(root.join("20240101-000002").exists());
        assert!(root.join("20240101-000003").exists());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn prune_prefixed_floors_keep() {
        let base = std::env::temp_dir().join(format!(
            "hermes-prune-pre-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();

        for n in 1..=3 {
            fs::write(
                base.join(format!("pre-update-2024-01-0{}-000000.zip", n)),
                b"x",
            )
            .unwrap();
        }
        // hand-made zip not matching prefix must never be touched
        fs::write(base.join("manual.zip"), b"y").unwrap();

        // keep 0 should be floored to 1 for pre-update.
        let deleted = prune_prefixed_backups(&base, PRE_UPDATE_PREFIX, 0, true);
        assert_eq!(deleted, 2);
        assert!(base.join("pre-update-2024-01-03-000000.zip").exists());
        assert!(base.join("manual.zip").exists());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn full_zip_backup_roundtrip() {
        let base = std::env::temp_dir().join(format!(
            "hermes-fullzip-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let root = base.join("root");
        fs::create_dir_all(root.join("sub")).unwrap();
        fs::write(root.join("config.yaml"), b"a: b\n").unwrap();
        fs::write(root.join("sub/data.json"), b"{}").unwrap();
        // excluded dir must not appear
        fs::create_dir_all(root.join("hermes-agent")).unwrap();
        fs::write(root.join("hermes-agent/code.py"), b"print()").unwrap();

        let out = base.join("out.zip");
        let result = write_full_zip_backup(&out, &root);
        assert!(result.is_some());
        assert!(out.exists());

        // Verify contents.
        let f = File::open(&out).unwrap();
        let mut archive = ZipArchive::new(f).unwrap();
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();
        assert!(names.contains(&"config.yaml".to_string()));
        assert!(names.contains(&"sub/data.json".to_string()));
        assert!(!names.iter().any(|n| n.contains("hermes-agent")));

        let _ = fs::remove_dir_all(&base);
    }
}
