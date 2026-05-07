use std::cell::RefCell;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once, OnceLock};

use chrono::Local;
use log::{Level, LevelFilter, Log, Metadata, Record};

use crate::config::LoadedConfig;
use crate::{HermesContext, HermesError};

const NOISY_LOGGERS: [&str; 13] = [
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "asyncio",
    "hpack",
    "hpack.hpack",
    "grpc",
    "modal",
    "urllib3",
    "urllib3.connectionpool",
    "websockets",
    "charset_normalizer",
];

thread_local! {
    static SESSION_CONTEXT: RefCell<Option<String>> = const { RefCell::new(None) };
}

static LOGGER_STATE: OnceLock<LoggerState> = OnceLock::new();
static LOGGER_INSTALL: Once = Once::new();
static GLOBAL_LOGGER: HermesGlobalLogger = HermesGlobalLogger;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoggingMode {
    Cli,
    Gateway,
    Cron,
}

#[derive(Debug, Clone)]
pub struct LoggingSetup {
    pub log_dir: PathBuf,
    pub agent_log: PathBuf,
    pub errors_log: PathBuf,
    pub gateway_log: Option<PathBuf>,
    pub level: LevelFilter,
}

impl HermesContext {
    pub fn setup_logging(
        &self,
        config: &LoadedConfig,
        mode: LoggingMode,
    ) -> Result<LoggingSetup, HermesError> {
        self.ensure_hermes_home()?;

        let log_dir = self.hermes_home().join("logs");
        create_dir_all(&log_dir)?;

        let level = parse_level_filter(&config.config.logging.level);
        let gateway_log = matches!(mode, LoggingMode::Gateway).then(|| log_dir.join("gateway.log"));

        if LOGGER_STATE.get().is_none() {
            let state = LoggerState::new(
                log_dir.clone(),
                level,
                config
                    .config
                    .logging
                    .max_size_mb
                    .saturating_mul(1024 * 1024),
                config.config.logging.backup_count as usize,
                gateway_log.clone(),
            )?;
            let _ = LOGGER_STATE.set(state);
        }

        LOGGER_INSTALL.call_once(|| {
            let _ = log::set_logger(&GLOBAL_LOGGER);
        });
        log::set_max_level(level);

        Ok(LoggingSetup {
            agent_log: log_dir.join("agent.log"),
            errors_log: log_dir.join("errors.log"),
            log_dir,
            gateway_log,
            level,
        })
    }
}

pub fn set_session_context(session_id: &str) {
    SESSION_CONTEXT.with(|current| *current.borrow_mut() = Some(session_id.to_string()));
}

pub fn clear_session_context() {
    SESSION_CONTEXT.with(|current| *current.borrow_mut() = None);
}

pub fn enable_verbose_logging() {
    if let Some(state) = LOGGER_STATE.get() {
        state.verbose_stderr.store(true, Ordering::SeqCst);
        log::set_max_level(LevelFilter::Debug);
    }
}

struct HermesGlobalLogger;

impl Log for HermesGlobalLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        LOGGER_STATE
            .get()
            .is_some_and(|state| state.enabled(metadata))
    }

    fn log(&self, record: &Record<'_>) {
        if let Some(state) = LOGGER_STATE.get() {
            let _ = state.write_record(record);
        }
    }

    fn flush(&self) {
        if let Some(state) = LOGGER_STATE.get() {
            state.flush();
        }
    }
}

struct LoggerState {
    level: LevelFilter,
    agent: Mutex<FileSink>,
    errors: Mutex<FileSink>,
    gateway: Option<Mutex<FileSink>>,
    verbose_stderr: AtomicBool,
}

impl LoggerState {
    fn new(
        log_dir: PathBuf,
        level: LevelFilter,
        max_bytes: u64,
        backup_count: usize,
        gateway_log: Option<PathBuf>,
    ) -> Result<Self, HermesError> {
        Ok(Self {
            level,
            agent: Mutex::new(FileSink::new(
                log_dir.join("agent.log"),
                max_bytes.max(1),
                backup_count,
            )?),
            errors: Mutex::new(FileSink::new(
                log_dir.join("errors.log"),
                2 * 1024 * 1024,
                2,
            )?),
            gateway: match gateway_log {
                Some(path) => Some(Mutex::new(FileSink::new(path, 5 * 1024 * 1024, 3)?)),
                None => None,
            },
            verbose_stderr: AtomicBool::new(false),
        })
    }

    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level().to_level_filter() <= self.level
            && !is_noisy_target(metadata.target(), metadata.level())
    }

    fn write_record(&self, record: &Record<'_>) -> Result<(), HermesError> {
        if !self.enabled(record.metadata()) {
            return Ok(());
        }
        self.write_entry(record.level(), record.target(), &record.args().to_string())
    }

    fn write_entry(&self, level: Level, target: &str, message: &str) -> Result<(), HermesError> {
        let session_tag = SESSION_CONTEXT.with(|current| {
            current
                .borrow()
                .as_ref()
                .map_or_else(String::new, |session| format!(" [{session}]"))
        });
        let line = format_log_line(level, target, message, &session_tag);

        self.agent
            .lock()
            .expect("agent log mutex")
            .write_line(&line)?;

        if matches!(level, Level::Warn | Level::Error) {
            self.errors
                .lock()
                .expect("errors log mutex")
                .write_line(&line)?;
        }

        if target.starts_with("gateway") {
            if let Some(gateway) = &self.gateway {
                gateway
                    .lock()
                    .expect("gateway log mutex")
                    .write_line(&line)?;
            }
        }

        if self.verbose_stderr.load(Ordering::SeqCst) {
            eprint!("{line}");
        }

        Ok(())
    }

    fn flush(&self) {
        let _ = self.agent.lock().map(|mut sink| sink.flush());
        let _ = self.errors.lock().map(|mut sink| sink.flush());
        if let Some(gateway) = &self.gateway {
            let _ = gateway.lock().map(|mut sink| sink.flush());
        }
    }
}

struct FileSink {
    path: PathBuf,
    max_bytes: u64,
    backup_count: usize,
    file: File,
}

impl FileSink {
    fn new(path: PathBuf, max_bytes: u64, backup_count: usize) -> Result<Self, HermesError> {
        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }
        let file = open_append_file(&path)?;
        Ok(Self {
            path,
            max_bytes,
            backup_count,
            file,
        })
    }

    fn write_line(&mut self, line: &str) -> Result<(), HermesError> {
        self.rotate_if_needed(line.len() as u64)?;
        self.file
            .write_all(line.as_bytes())
            .map_err(|source| HermesError::Io {
                action: "writing",
                path: self.path.clone(),
                source,
            })?;
        self.file.flush().map_err(|source| HermesError::Io {
            action: "flushing",
            path: self.path.clone(),
            source,
        })?;
        Ok(())
    }

    fn flush(&mut self) {
        let _ = self.file.flush();
    }

    fn rotate_if_needed(&mut self, incoming_len: u64) -> Result<(), HermesError> {
        let current_len = self
            .file
            .metadata()
            .map_err(|source| HermesError::Io {
                action: "reading metadata",
                path: self.path.clone(),
                source,
            })?
            .len();

        if current_len.saturating_add(incoming_len) <= self.max_bytes {
            return Ok(());
        }

        self.file.flush().map_err(|source| HermesError::Io {
            action: "flushing",
            path: self.path.clone(),
            source,
        })?;

        if self.backup_count == 0 {
            self.file = open_truncate_file(&self.path)?;
            return Ok(());
        }

        let oldest = self.rotated_path(self.backup_count);
        if oldest.exists() {
            fs::remove_file(&oldest).map_err(|source| HermesError::Io {
                action: "removing",
                path: oldest.clone(),
                source,
            })?;
        }

        for index in (1..self.backup_count).rev() {
            let src = self.rotated_path(index);
            if src.exists() {
                let dst = self.rotated_path(index + 1);
                fs::rename(&src, &dst).map_err(|source| HermesError::Io {
                    action: "renaming",
                    path: dst.clone(),
                    source,
                })?;
            }
        }

        if self.path.exists() {
            let first_backup = self.rotated_path(1);
            fs::rename(&self.path, &first_backup).map_err(|source| HermesError::Io {
                action: "renaming",
                path: first_backup,
                source,
            })?;
        }

        self.file = open_truncate_file(&self.path)?;
        Ok(())
    }

    fn rotated_path(&self, index: usize) -> PathBuf {
        self.path.with_extension(format!("log.{index}"))
    }
}

fn open_append_file(path: &Path) -> Result<File, HermesError> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|source| HermesError::Io {
            action: "opening",
            path: path.to_path_buf(),
            source,
        })
}

fn open_truncate_file(path: &Path) -> Result<File, HermesError> {
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|source| HermesError::Io {
            action: "opening",
            path: path.to_path_buf(),
            source,
        })
}

fn create_dir_all(path: &Path) -> Result<(), HermesError> {
    fs::create_dir_all(path).map_err(|source| HermesError::Io {
        action: "creating",
        path: path.to_path_buf(),
        source,
    })
}

fn parse_level_filter(level_name: &str) -> LevelFilter {
    match level_name.trim().to_ascii_uppercase().as_str() {
        "TRACE" => LevelFilter::Trace,
        "DEBUG" => LevelFilter::Debug,
        "WARNING" | "WARN" => LevelFilter::Warn,
        "ERROR" => LevelFilter::Error,
        "OFF" => LevelFilter::Off,
        _ => LevelFilter::Info,
    }
}

fn is_noisy_target(target: &str, level: Level) -> bool {
    level.to_level_filter() < LevelFilter::Warn
        && NOISY_LOGGERS
            .iter()
            .any(|prefix| target.starts_with(prefix))
}

fn format_log_line(level: Level, target: &str, message: &str, session_tag: &str) -> String {
    format!(
        "{} {}{} {}: {}\n",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        format_level(level),
        session_tag,
        target,
        message
    )
}

fn format_level(level: Level) -> &'static str {
    match level {
        Level::Error => "ERROR",
        Level::Warn => "WARNING",
        Level::Info => "INFO",
        Level::Debug => "DEBUG",
        Level::Trace => "TRACE",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn logger_state_writes_agent_errors_and_gateway_logs() {
        let temp = TempDir::new().expect("tempdir");
        let state = LoggerState::new(
            temp.path().to_path_buf(),
            LevelFilter::Info,
            1024 * 1024,
            2,
            Some(temp.path().join("gateway.log")),
        )
        .expect("logger state");

        set_session_context("sess-1");
        state
            .write_entry(Level::Info, "hermes_cli", "startup complete")
            .expect("agent log");
        state
            .write_entry(Level::Warn, "tools.exec", "subprocess failed")
            .expect("errors log");
        state
            .write_entry(Level::Info, "gateway.telegram", "delivery started")
            .expect("gateway log");
        clear_session_context();

        let agent = fs::read_to_string(temp.path().join("agent.log")).expect("agent log read");
        let errors = fs::read_to_string(temp.path().join("errors.log")).expect("errors log read");
        let gateway =
            fs::read_to_string(temp.path().join("gateway.log")).expect("gateway log read");

        assert!(agent.contains("INFO [sess-1] hermes_cli: startup complete"));
        assert!(agent.contains("WARNING [sess-1] tools.exec: subprocess failed"));
        assert!(gateway.contains("INFO [sess-1] gateway.telegram: delivery started"));
        assert!(!gateway.contains("hermes_cli: startup complete"));
        assert!(errors.contains("WARNING [sess-1] tools.exec: subprocess failed"));
    }

    #[test]
    fn file_sink_rotates_when_size_limit_is_exceeded() {
        let temp = TempDir::new().expect("tempdir");
        let mut sink = FileSink::new(temp.path().join("agent.log"), 40, 1).expect("file sink");

        sink.write_line("12345678901234567890\n").expect("first");
        sink.write_line("abcdefghijabcdefghij\n").expect("second");

        assert!(temp.path().join("agent.log").exists());
        assert!(temp.path().join("agent.log.1").exists());
    }
}
