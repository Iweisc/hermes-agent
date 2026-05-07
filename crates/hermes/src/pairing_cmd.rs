use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Subcommand;
use hermes_core::HermesContext;
use serde::{Deserialize, Serialize};

const CODE_TTL_SECONDS: f64 = 3600.0;
const LOCKOUT_SECONDS: f64 = 3600.0;
const MAX_FAILED_ATTEMPTS: i64 = 5;

#[derive(Subcommand, Debug)]
pub enum PairingCommand {
    List,
    Approve {
        platform: String,
        code: String,
    },
    Revoke {
        platform: String,
        user_id: String,
    },
    #[command(name = "clear-pending")]
    ClearPending,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PendingEntry {
    user_id: String,
    #[serde(default)]
    user_name: String,
    created_at: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ApprovedEntry {
    #[serde(default)]
    user_name: String,
    approved_at: f64,
}

pub fn print_pairing(
    context: &HermesContext,
    command: Option<PairingCommand>,
) -> Result<(), Box<dyn Error>> {
    let store = PairingStore::new(context);
    match command.unwrap_or(PairingCommand::List) {
        PairingCommand::List => print_list(&store)?,
        PairingCommand::Approve { platform, code } => {
            let platform = validate_platform(&platform)?;
            let code = validate_code(&code)?;
            if let Some(result) = store.approve_code(&platform, &code)? {
                let display = if result.user_name.trim().is_empty() {
                    result.user_id.clone()
                } else {
                    format!("{} ({})", result.user_name, result.user_id)
                };
                println!("Approved! User {display} on {platform} can now use the bot.");
                println!("They'll be recognized automatically on their next message.");
            } else {
                println!("Code '{code}' not found or expired for platform '{platform}'.");
                println!("Run `hermes pairing list` to see pending codes.");
            }
        }
        PairingCommand::Revoke { platform, user_id } => {
            let platform = validate_platform(&platform)?;
            let user_id = validate_user_id(&user_id)?;
            if store.revoke(&platform, &user_id)? {
                println!("Revoked access for user {user_id} on {platform}.");
            } else {
                println!("User {user_id} not found in approved list for {platform}.");
            }
        }
        PairingCommand::ClearPending => {
            let count = store.clear_pending(None)?;
            if count > 0 {
                println!("Cleared {count} pending pairing request(s).");
            } else {
                println!("No pending requests to clear.");
            }
        }
    }
    Ok(())
}

fn print_list(store: &PairingStore) -> Result<(), Box<dyn Error>> {
    let pending = store.list_pending(None)?;
    let approved = store.list_approved(None)?;
    if pending.is_empty() && approved.is_empty() {
        println!("No pairing data found. No one has tried to pair yet~");
        return Ok(());
    }

    if !pending.is_empty() {
        println!();
        println!("  Pending Pairing Requests ({}):", pending.len());
        println!(
            "  {:<12} {:<10} {:<20} {:<20} {}",
            "Platform", "Code", "User ID", "Name", "Age"
        );
        println!(
            "  {:<12} {:<10} {:<20} {:<20} {}",
            "--------", "----", "-------", "----", "---"
        );
        for entry in pending {
            println!(
                "  {:<12} {:<10} {:<20} {:<20} {}m ago",
                entry.platform, entry.code, entry.user_id, entry.user_name, entry.age_minutes
            );
        }
    } else {
        println!();
        println!("  No pending pairing requests.");
    }

    if !approved.is_empty() {
        println!();
        println!("  Approved Users ({}):", approved.len());
        println!("  {:<12} {:<20} {:<20}", "Platform", "User ID", "Name");
        println!("  {:<12} {:<20} {:<20}", "--------", "-------", "----");
        for entry in approved {
            println!(
                "  {:<12} {:<20} {:<20}",
                entry.platform, entry.user_id, entry.user_name
            );
        }
    } else {
        println!();
        println!("  No approved users.");
    }

    println!();
    Ok(())
}

#[derive(Debug, Clone)]
struct PendingDisplayRow {
    platform: String,
    code: String,
    user_id: String,
    user_name: String,
    age_minutes: i64,
}

#[derive(Debug, Clone)]
struct ApprovedDisplayRow {
    platform: String,
    user_id: String,
    user_name: String,
}

#[derive(Debug, Clone)]
struct ApprovedResult {
    user_id: String,
    user_name: String,
}

struct PairingStore {
    dir: PathBuf,
}

impl PairingStore {
    fn new(context: &HermesContext) -> Self {
        Self {
            dir: resolve_pairing_dir(context),
        }
    }

    fn pending_path(&self, platform: &str) -> PathBuf {
        self.dir.join(format!("{platform}-pending.json"))
    }

    fn approved_path(&self, platform: &str) -> PathBuf {
        self.dir.join(format!("{platform}-approved.json"))
    }

    fn rate_limit_path(&self) -> PathBuf {
        self.dir.join("_rate_limits.json")
    }

    fn list_pending(
        &self,
        platform: Option<&str>,
    ) -> Result<Vec<PendingDisplayRow>, Box<dyn Error>> {
        let mut result = Vec::new();
        let platforms = if let Some(platform) = platform {
            vec![platform.to_string()]
        } else {
            self.all_platforms("pending")?
        };
        for platform in platforms {
            self.cleanup_expired(&platform)?;
            let pending = self.load_pending(&platform)?;
            for (code, entry) in pending {
                let age_minutes = ((now_ts() - entry.created_at) / 60.0).max(0.0) as i64;
                result.push(PendingDisplayRow {
                    platform: platform.clone(),
                    code,
                    user_id: entry.user_id,
                    user_name: entry.user_name,
                    age_minutes,
                });
            }
        }
        result.sort_by(|left, right| {
            left.platform
                .cmp(&right.platform)
                .then_with(|| left.code.cmp(&right.code))
        });
        Ok(result)
    }

    fn list_approved(
        &self,
        platform: Option<&str>,
    ) -> Result<Vec<ApprovedDisplayRow>, Box<dyn Error>> {
        let mut result = Vec::new();
        let platforms = if let Some(platform) = platform {
            vec![platform.to_string()]
        } else {
            self.all_platforms("approved")?
        };
        for platform in platforms {
            let approved = self.load_approved(&platform)?;
            for (user_id, entry) in approved {
                result.push(ApprovedDisplayRow {
                    platform: platform.clone(),
                    user_id,
                    user_name: entry.user_name,
                });
            }
        }
        result.sort_by(|left, right| {
            left.platform
                .cmp(&right.platform)
                .then_with(|| left.user_id.cmp(&right.user_id))
        });
        Ok(result)
    }

    fn approve_code(
        &self,
        platform: &str,
        code: &str,
    ) -> Result<Option<ApprovedResult>, Box<dyn Error>> {
        self.cleanup_expired(platform)?;
        let mut pending = self.load_pending(platform)?;
        let Some(entry) = pending.remove(code) else {
            self.record_failed_attempt(platform)?;
            return Ok(None);
        };
        self.save_pending(platform, &pending)?;

        let mut approved = self.load_approved(platform)?;
        approved.insert(
            entry.user_id.clone(),
            ApprovedEntry {
                user_name: entry.user_name.clone(),
                approved_at: now_ts(),
            },
        );
        self.save_approved(platform, &approved)?;

        Ok(Some(ApprovedResult {
            user_id: entry.user_id,
            user_name: entry.user_name,
        }))
    }

    fn revoke(&self, platform: &str, user_id: &str) -> Result<bool, Box<dyn Error>> {
        let mut approved = self.load_approved(platform)?;
        if approved.remove(user_id).is_none() {
            return Ok(false);
        }
        self.save_approved(platform, &approved)?;
        Ok(true)
    }

    fn clear_pending(&self, platform: Option<&str>) -> Result<usize, Box<dyn Error>> {
        let platforms = if let Some(platform) = platform {
            vec![platform.to_string()]
        } else {
            self.all_platforms("pending")?
        };
        let mut count = 0_usize;
        for platform in platforms {
            let pending = self.load_pending(&platform)?;
            count += pending.len();
            self.save_pending(&platform, &BTreeMap::new())?;
        }
        Ok(count)
    }

    fn cleanup_expired(&self, platform: &str) -> Result<(), Box<dyn Error>> {
        let mut pending = self.load_pending(platform)?;
        let cutoff = now_ts() - CODE_TTL_SECONDS;
        let before = pending.len();
        pending.retain(|_, entry| entry.created_at >= cutoff);
        if pending.len() != before {
            self.save_pending(platform, &pending)?;
        }
        Ok(())
    }

    fn record_failed_attempt(&self, platform: &str) -> Result<(), Box<dyn Error>> {
        let mut limits = self.load_rate_limits()?;
        let fail_key = format!("_failures:{platform}");
        let lockout_key = format!("_lockout:{platform}");
        let failures = limits.get(&fail_key).copied().unwrap_or(0.0) as i64 + 1;
        if failures >= MAX_FAILED_ATTEMPTS {
            limits.insert(fail_key, 0.0);
            limits.insert(lockout_key, now_ts() + LOCKOUT_SECONDS);
        } else {
            limits.insert(fail_key, failures as f64);
        }
        self.save_rate_limits(&limits)
    }

    fn all_platforms(&self, suffix: &str) -> Result<Vec<String>, Box<dyn Error>> {
        if !self.dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut platforms = BTreeMap::<String, ()>::new();
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            let marker = format!("-{suffix}.json");
            let Some(platform) = name.strip_suffix(&marker) else {
                continue;
            };
            if platform.starts_with('_') || platform.is_empty() {
                continue;
            }
            platforms.insert(platform.to_string(), ());
        }
        Ok(platforms.into_keys().collect())
    }

    fn load_pending(
        &self,
        platform: &str,
    ) -> Result<BTreeMap<String, PendingEntry>, Box<dyn Error>> {
        load_json_map(&self.pending_path(platform))
    }

    fn save_pending(
        &self,
        platform: &str,
        data: &BTreeMap<String, PendingEntry>,
    ) -> Result<(), Box<dyn Error>> {
        save_json_map(&self.pending_path(platform), data)
    }

    fn load_approved(
        &self,
        platform: &str,
    ) -> Result<BTreeMap<String, ApprovedEntry>, Box<dyn Error>> {
        load_json_map(&self.approved_path(platform))
    }

    fn save_approved(
        &self,
        platform: &str,
        data: &BTreeMap<String, ApprovedEntry>,
    ) -> Result<(), Box<dyn Error>> {
        save_json_map(&self.approved_path(platform), data)
    }

    fn load_rate_limits(&self) -> Result<BTreeMap<String, f64>, Box<dyn Error>> {
        load_json_map(&self.rate_limit_path())
    }

    fn save_rate_limits(&self, data: &BTreeMap<String, f64>) -> Result<(), Box<dyn Error>> {
        save_json_map(&self.rate_limit_path(), data)
    }
}

fn resolve_pairing_dir(context: &HermesContext) -> PathBuf {
    let home = context.hermes_home();
    let legacy = home.join("pairing");
    if legacy.exists() {
        return legacy;
    }
    home.join("platforms").join("pairing")
}

fn load_json_map<T>(path: &Path) -> Result<BTreeMap<String, T>, Box<dyn Error>>
where
    T: for<'de> Deserialize<'de>,
{
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let bytes = fs::read(path)?;
    let parsed = serde_json::from_slice::<BTreeMap<String, T>>(&bytes).unwrap_or_default();
    Ok(parsed)
}

fn save_json_map<T>(path: &Path, data: &BTreeMap<String, T>) -> Result<(), Box<dyn Error>>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{unique}"));
    let bytes = serde_json::to_vec_pretty(data)?;
    fs::write(&tmp, bytes)?;
    tighten_permissions(&tmp)?;
    fs::rename(&tmp, path)?;
    tighten_permissions(path)?;
    Ok(())
}

fn tighten_permissions(path: &Path) -> Result<(), Box<dyn Error>> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn validate_platform(value: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = value.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return Err("platform cannot be empty".into());
    }
    if trimmed.contains(['/', '\\']) {
        return Err("platform cannot contain path separators".into());
    }
    Ok(trimmed)
}

fn validate_code(value: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = value.trim().to_ascii_uppercase();
    if trimmed.is_empty() {
        return Err("code cannot be empty".into());
    }
    if trimmed.contains(['/', '\\']) {
        return Err("code cannot contain path separators".into());
    }
    Ok(trimmed)
}

fn validate_user_id(value: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("user_id cannot be empty".into());
    }
    Ok(trimmed.to_string())
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn pairing_store_lists_approves_revokes_and_clears() {
        let home = TempDir::new().unwrap();
        let context = HermesContext::new(home.path());
        let store = PairingStore::new(&context);

        let mut pending = BTreeMap::new();
        pending.insert(
            "ABCD1234".to_string(),
            PendingEntry {
                user_id: "u-1".to_string(),
                user_name: "Alice".to_string(),
                created_at: now_ts(),
            },
        );
        store.save_pending("telegram", &pending).unwrap();

        let listed = store.list_pending(None).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].platform, "telegram");

        let approved = store.approve_code("telegram", "ABCD1234").unwrap().unwrap();
        assert_eq!(approved.user_id, "u-1");
        assert!(store.list_pending(None).unwrap().is_empty());
        let approved_rows = store.list_approved(None).unwrap();
        assert_eq!(approved_rows.len(), 1);
        assert_eq!(approved_rows[0].user_name, "Alice");

        assert!(store.revoke("telegram", "u-1").unwrap());
        assert!(!store.revoke("telegram", "u-1").unwrap());

        let mut pending_again = BTreeMap::new();
        pending_again.insert(
            "WXYZ9999".to_string(),
            PendingEntry {
                user_id: "u-2".to_string(),
                user_name: "".to_string(),
                created_at: now_ts(),
            },
        );
        store.save_pending("telegram", &pending_again).unwrap();
        assert_eq!(store.clear_pending(None).unwrap(), 1);
        assert!(store.list_pending(None).unwrap().is_empty());
    }

    #[test]
    fn pairing_store_uses_legacy_dir_when_present() {
        let home = TempDir::new().unwrap();
        let context = HermesContext::new(home.path());
        fs::create_dir_all(context.hermes_home().join("pairing")).unwrap();
        let store = PairingStore::new(&context);
        assert_eq!(store.dir, context.hermes_home().join("pairing"));
    }

    #[test]
    fn invalid_code_records_failures_and_lockout() {
        let home = TempDir::new().unwrap();
        let context = HermesContext::new(home.path());
        let store = PairingStore::new(&context);
        for _ in 0..MAX_FAILED_ATTEMPTS {
            assert!(store.approve_code("slack", "MISSING").unwrap().is_none());
        }
        let limits = store.load_rate_limits().unwrap();
        assert_eq!(limits.get("_failures:slack").copied().unwrap_or(-1.0), 0.0);
        assert!(limits.get("_lockout:slack").copied().unwrap_or(0.0) > now_ts());
    }
}
