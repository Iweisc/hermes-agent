use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::{HermesContext, HermesError};

const CREDENTIAL_SUFFIXES: [&str; 4] = ["_API_KEY", "_TOKEN", "_SECRET", "_KEY"];

#[derive(Debug, Clone, Default)]
pub struct EnvLoadReport {
    pub loaded_paths: Vec<PathBuf>,
    pub warnings: Vec<String>,
}

impl HermesContext {
    pub fn load_hermes_dotenv(
        &self,
        project_env: Option<&Path>,
    ) -> Result<EnvLoadReport, HermesError> {
        let mut report = EnvLoadReport::default();
        let mut warned_keys = BTreeSet::new();
        let user_env = self.env_path();
        let mut loaded_user_env = false;

        if user_env.exists() {
            load_env_file(&user_env, true, &mut warned_keys, &mut report)?;
            loaded_user_env = true;
        }

        if let Some(project_env_path) = project_env.filter(|path| path.exists()) {
            load_env_file(
                project_env_path,
                !loaded_user_env,
                &mut warned_keys,
                &mut report,
            )?;
        }

        Ok(report)
    }
}

fn load_env_file(
    path: &Path,
    override_existing: bool,
    warned_keys: &mut BTreeSet<String>,
    report: &mut EnvLoadReport,
) -> Result<(), HermesError> {
    let bytes = fs::read(path).map_err(|source| HermesError::Io {
        action: "reading",
        path: path.to_path_buf(),
        source,
    })?;
    let contents = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => String::from_utf8_lossy(&error.into_bytes()).into_owned(),
    };

    for (line_number, line) in contents.lines().enumerate() {
        let Some((key, value)) = parse_env_line(line) else {
            continue;
        };
        if !is_valid_env_var_name(&key) {
            report.warnings.push(format!(
                "Ignoring invalid env var name {key:?} in {}:{}",
                path.display(),
                line_number + 1
            ));
            continue;
        }

        let sanitized = sanitize_loaded_credential(&key, &value, warned_keys, &mut report.warnings);
        if override_existing || env::var_os(&key).is_none() {
            // SAFETY: startup-only env mutation before any threads are spawned.
            unsafe { env::set_var(&key, &sanitized) };
        }
    }

    report.loaded_paths.push(path.to_path_buf());
    Ok(())
}

fn parse_env_line(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }

    let without_export = trimmed.strip_prefix("export ").unwrap_or(trimmed);
    let (key, raw_value) = without_export.split_once('=')?;
    let key = key.trim().to_string();
    let value = strip_matching_quotes(raw_value.trim()).to_string();
    Some((key, value))
}

fn strip_matching_quotes(value: &str) -> &str {
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        let first = bytes[0];
        let last = bytes[value.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return &value[1..value.len() - 1];
        }
    }
    value
}

fn is_valid_env_var_name(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn sanitize_loaded_credential(
    key: &str,
    value: &str,
    warned_keys: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> String {
    if !CREDENTIAL_SUFFIXES
        .iter()
        .any(|suffix| key.ends_with(suffix))
    {
        return value.to_string();
    }

    let sanitized: String = value.chars().filter(|ch| ch.is_ascii()).collect();
    if sanitized.len() == value.len() || !warned_keys.insert(key.to_string()) {
        return sanitized;
    }

    warnings.push(format!(
        "{key} contained non-ASCII characters and was sanitized before use"
    ));
    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::TempDir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn test_context() -> (TempDir, HermesContext) {
        let temp = TempDir::new().expect("tempdir");
        let home = temp.path().join("home");
        fs::create_dir_all(&home).expect("home dir");
        (temp, HermesContext::new(home))
    }

    #[test]
    fn user_env_overrides_project_env_and_sanitizes_credentials() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let (temp, ctx) = test_context();
        let hermes_home = ctx.hermes_home();
        fs::create_dir_all(&hermes_home).expect("hermes home");
        fs::write(
            hermes_home.join(".env"),
            "OPENAI_API_KEY=sk-tést\nMODEL_NAME=user\n",
        )
        .expect("user env");
        let project_env = temp.path().join("project.env");
        fs::write(&project_env, "OPENAI_API_KEY=project\nMODEL_NAME=project\n")
            .expect("project env");

        let report = ctx
            .load_hermes_dotenv(Some(&project_env))
            .expect("load dotenv");

        assert_eq!(report.loaded_paths.len(), 2);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("OPENAI_API_KEY"))
        );
        assert_eq!(env::var("OPENAI_API_KEY").expect("api key"), "sk-tst");
        assert_eq!(env::var("MODEL_NAME").expect("model"), "user");
        // SAFETY: test-only cleanup of env vars set by the loader.
        unsafe {
            env::remove_var("OPENAI_API_KEY");
            env::remove_var("MODEL_NAME");
        };
    }

    #[test]
    fn project_env_fills_when_user_env_is_missing() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let (temp, ctx) = test_context();
        let project_env = temp.path().join("project.env");
        fs::write(&project_env, "MODEL_NAME=project\n").expect("project env");

        let report = ctx
            .load_hermes_dotenv(Some(&project_env))
            .expect("load dotenv");

        assert_eq!(report.loaded_paths, vec![project_env]);
        assert_eq!(env::var("MODEL_NAME").expect("model"), "project");
        // SAFETY: test-only cleanup of env vars set by the loader.
        unsafe { env::remove_var("MODEL_NAME") };
    }
}
