//! CLI commands for the DM pairing system.
//!
//! Mirrors `hermes_cli/pairing.py`.
//!
//! Usage:
//!   hermes pairing list                          # Show all pending + approved users
//!   hermes pairing approve <platform> <code>     # Approve a pairing code
//!   hermes pairing revoke <platform> <user_id>   # Revoke user access
//!   hermes pairing clear-pending                 # Clear all expired/pending codes
//!
//! The Python module delegates all storage logic to `gateway.pairing.PairingStore`,
//! whose native equivalent is [`hermes_core::gw_pairing::PairingStore`]. This module
//! is purely the command-dispatch + presentation (printing) layer.

use hermes_core::gw_pairing::PairingStore;

/// The parsed `pairing` subcommand action.
///
/// The Python code reads `args.pairing_action` plus optional positional
/// `args.platform`, `args.code`, `args.user_id`. We model that as an explicit
/// enum so callers (the CLI arg parser) can construct it directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingAction {
    List,
    Approve { platform: String, code: String },
    Revoke { platform: String, user_id: String },
    ClearPending,
    /// Unknown / missing action -> prints usage.
    Usage,
}

/// Handle the `hermes pairing` subcommands.
///
/// Mirrors `pairing_command(args)`. Output is written via `println!`
/// (i.e. to stdout) exactly as the Python `print(...)` calls do.
pub fn pairing_command(action: &PairingAction) {
    let store = PairingStore::new();
    dispatch(&store, action);
}

/// Dispatch against an explicit store (used for tests/injection).
pub fn dispatch(store: &PairingStore, action: &PairingAction) {
    match action {
        PairingAction::List => cmd_list(store),
        PairingAction::Approve { platform, code } => cmd_approve(store, platform, code),
        PairingAction::Revoke { platform, user_id } => cmd_revoke(store, platform, user_id),
        PairingAction::ClearPending => cmd_clear_pending(store),
        PairingAction::Usage => {
            println!("Usage: hermes pairing {{list|approve|revoke|clear-pending}}");
            println!("Run 'hermes pairing --help' for details.");
        }
    }
}

/// List all pending and approved users.
///
/// Mirrors `_cmd_list`. Returns the rendered text so it can be unit-tested
/// without capturing stdout; the side effect is the `println!` of that text.
fn cmd_list(store: &PairingStore) {
    let out = render_list(store);
    print!("{out}");
}

/// Build the text `_cmd_list` would print. Split out for testability.
pub fn render_list(store: &PairingStore) -> String {
    let pending = store.list_pending(None);
    let approved = store.list_approved(None);

    if pending.is_empty() && approved.is_empty() {
        return "No pairing data found. No one has tried to pair yet~\n".to_string();
    }

    let mut out = String::new();

    if !pending.is_empty() {
        out.push_str(&format!(
            "\n  Pending Pairing Requests ({}):\n",
            pending.len()
        ));
        // f"  {'Platform':<12} {'Code':<10} {'User ID':<20} {'Name':<20} {'Age'}"
        out.push_str(&format!(
            "  {:<12} {:<10} {:<20} {:<20} {}\n",
            "Platform", "Code", "User ID", "Name", "Age"
        ));
        out.push_str(&format!(
            "  {:<12} {:<10} {:<20} {:<20} {}\n",
            "--------", "----", "-------", "----", "---"
        ));
        for p in &pending {
            out.push_str(&format!(
                "  {:<12} {:<10} {:<20} {:<20} {}m ago\n",
                p.platform, p.code, p.user_id, p.user_name, p.age_minutes
            ));
        }
    } else {
        out.push_str("\n  No pending pairing requests.\n");
    }

    if !approved.is_empty() {
        out.push_str(&format!("\n  Approved Users ({}):\n", approved.len()));
        // f"  {'Platform':<12} {'User ID':<20} {'Name':<20}"
        out.push_str(&format!(
            "  {:<12} {:<20} {:<20}\n",
            "Platform", "User ID", "Name"
        ));
        out.push_str(&format!(
            "  {:<12} {:<20} {:<20}\n",
            "--------", "-------", "----"
        ));
        for a in &approved {
            out.push_str(&format!(
                "  {:<12} {:<20} {:<20}\n",
                a.platform, a.user_id, a.user_name
            ));
        }
    } else {
        out.push_str("\n  No approved users.\n");
    }

    // Trailing print()
    out.push('\n');
    out
}

/// Approve a pairing code. Mirrors `_cmd_approve`.
fn cmd_approve(store: &PairingStore, platform: &str, code: &str) {
    print!("{}", render_approve(store, platform, code));
}

/// Build the text `_cmd_approve` would print. Performs the approval as a side effect.
pub fn render_approve(store: &PairingStore, platform: &str, code: &str) -> String {
    let platform = platform.to_lowercase();
    let platform = platform.trim();
    let code = code.to_uppercase();
    let code = code.trim();

    match store.approve_code(platform, code) {
        Some(result) => {
            let uid = &result.user_id;
            let name = &result.user_name;
            let display = if !name.is_empty() {
                format!("{name} ({uid})")
            } else {
                uid.clone()
            };
            format!(
                "\n  Approved! User {display} on {platform} can now use the bot~\n  \
                 They'll be recognized automatically on their next message.\n"
            )
        }
        None => format!(
            "\n  Code '{code}' not found or expired for platform '{platform}'.\n  \
             Run 'hermes pairing list' to see pending codes.\n"
        ),
    }
}

/// Revoke a user's access. Mirrors `_cmd_revoke`.
fn cmd_revoke(store: &PairingStore, platform: &str, user_id: &str) {
    print!("{}", render_revoke(store, platform, user_id));
}

/// Build the text `_cmd_revoke` would print. Performs the revoke as a side effect.
pub fn render_revoke(store: &PairingStore, platform: &str, user_id: &str) -> String {
    let platform = platform.to_lowercase();
    let platform = platform.trim();

    if store.revoke(platform, user_id) {
        format!("\n  Revoked access for user {user_id} on {platform}.\n")
    } else {
        format!("\n  User {user_id} not found in approved list for {platform}.\n")
    }
}

/// Clear all pending pairing codes. Mirrors `_cmd_clear_pending`.
fn cmd_clear_pending(store: &PairingStore) {
    print!("{}", render_clear_pending(store));
}

/// Build the text `_cmd_clear_pending` would print. Performs the clear as a side effect.
pub fn render_clear_pending(store: &PairingStore) -> String {
    let count = store.clear_pending(None);
    if count > 0 {
        format!("\n  Cleared {count} pending pairing request(s).\n")
    } else {
        "\n  No pending requests to clear.\n".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_store() -> (PairingStore, PathBuf) {
        let mut dir = std::env::temp_dir();
        let unique = format!(
            "hermes_cli_pairing_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        dir.push(unique);
        let store = PairingStore::with_dir(dir.clone());
        (store, dir)
    }

    #[test]
    fn list_empty_message() {
        let (store, dir) = tmp_store();
        let out = render_list(&store);
        assert_eq!(out, "No pairing data found. No one has tried to pair yet~\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn approve_then_list_then_revoke() {
        let (store, dir) = tmp_store();

        // Generate a pending code, then approve via the CLI layer.
        let code = store
            .generate_code("discord", "user-123", "Alice")
            .expect("code generated");

        // Pending list should render a header and the row.
        let listed = render_list(&store);
        assert!(listed.contains("Pending Pairing Requests (1)"));
        assert!(listed.contains("discord"));
        assert!(listed.contains("user-123"));
        assert!(listed.contains(&code));

        // Approve using mixed case / whitespace to exercise normalization.
        let approve_out = render_approve(&store, "  Discord ", &format!("  {} ", code.to_lowercase()));
        assert!(approve_out.contains("Approved!"));
        assert!(approve_out.contains("Alice (user-123)"));
        assert!(approve_out.contains("on discord"));

        // Now listed as approved, not pending.
        let listed2 = render_list(&store);
        assert!(listed2.contains("Approved Users (1)"));
        assert!(listed2.contains("No pending pairing requests."));

        // Revoke (platform normalized to lowercase).
        let revoke_out = render_revoke(&store, "DISCORD", "user-123");
        assert!(revoke_out.contains("Revoked access for user user-123 on discord."));

        // Revoking again -> not found.
        let revoke_out2 = render_revoke(&store, "discord", "user-123");
        assert!(revoke_out2.contains("not found in approved list for discord."));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn approve_missing_code() {
        let (store, dir) = tmp_store();
        let out = render_approve(&store, "telegram", "nope");
        assert!(out.contains("Code 'NOPE' not found or expired for platform 'telegram'."));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn clear_pending_counts() {
        let (store, dir) = tmp_store();
        let _ = store.generate_code("slack", "u1", "");
        let _ = store.generate_code("slack", "u2", "");

        let out = render_clear_pending(&store);
        assert!(out.contains("Cleared 2 pending pairing request(s)."));

        let out2 = render_clear_pending(&store);
        assert!(out2.contains("No pending requests to clear."));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn usage_action_does_not_panic() {
        let (store, dir) = tmp_store();
        dispatch(&store, &PairingAction::Usage);
        let _ = std::fs::remove_dir_all(dir);
    }
}
