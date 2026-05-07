use std::error::Error;

use clap::Subcommand;
use hermes_core::HermesContext;

use crate::backup::{
    QUICK_DEFAULT_KEEP, create_quick_snapshot, format_size, list_quick_snapshots,
    prune_quick_snapshots, restore_quick_snapshot,
};

#[derive(Subcommand, Debug)]
pub enum SnapshotCommand {
    #[command(alias = "ls")]
    List,
    Create {
        label: Vec<String>,
    },
    Restore {
        id: String,
    },
    Prune {
        keep: Option<usize>,
    },
}

pub fn print_snapshot(
    context: &HermesContext,
    command: Option<SnapshotCommand>,
) -> Result<(), Box<dyn Error>> {
    match command.unwrap_or(SnapshotCommand::List) {
        SnapshotCommand::List => print_snapshot_list(context)?,
        SnapshotCommand::Create { label } => {
            let label = join_label(&label);
            let snapshot_id = create_quick_snapshot(context, label.as_deref())?;
            if let Some(id) = snapshot_id {
                println!("Snapshot created: {id}");
            } else {
                println!("No state files found to snapshot.");
            }
        }
        SnapshotCommand::Restore { id } => {
            let snapshot_id = resolve_snapshot_id(context, &id)?;
            if restore_quick_snapshot(context, &snapshot_id)? {
                println!("Restored state from: {snapshot_id}");
                println!("Restart recommended for state.db changes to take effect.");
            } else {
                println!("Snapshot not found: {snapshot_id}");
            }
        }
        SnapshotCommand::Prune { keep } => {
            let keep = keep.unwrap_or(QUICK_DEFAULT_KEEP);
            let deleted = prune_quick_snapshots(context, keep)?;
            println!("Pruned {deleted} old snapshot(s) (keeping {keep}).");
        }
    }
    Ok(())
}

fn print_snapshot_list(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let snapshots = list_quick_snapshots(context, QUICK_DEFAULT_KEEP)?;
    if snapshots.is_empty() {
        println!("No state snapshots yet.");
        println!("Create one: hermes snapshot create [label]");
        return Ok(());
    }

    println!(
        "State snapshots ({}/state-snapshots/):",
        display_hermes_home(context)
    );
    println!();
    println!(
        "  {:>3}  {:<35} {:>5} {:>10} Label",
        "#", "ID", "Files", "Size"
    );
    println!(
        "  {:>3}  {:<35} {:>5} {:>10} {:<20}",
        "───", "───────────────────────────────────", "─────", "──────────", "────────────────────"
    );
    for (index, snapshot) in snapshots.iter().enumerate() {
        println!(
            "  {:>3}  {:<35} {:>5} {:>10} {}",
            index + 1,
            snapshot.id,
            snapshot.file_count,
            format_size(snapshot.total_size),
            snapshot.label.as_deref().unwrap_or("")
        );
    }
    Ok(())
}

fn resolve_snapshot_id(context: &HermesContext, raw: &str) -> Result<String, Box<dyn Error>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("snapshot id cannot be empty".into());
    }
    if let Ok(index) = trimmed.parse::<usize>() {
        let snapshots = list_quick_snapshots(context, QUICK_DEFAULT_KEEP)?;
        if !(1..=snapshots.len()).contains(&index) {
            return Err(format!(
                "invalid snapshot number {index}; available range is 1-{}",
                snapshots.len()
            )
            .into());
        }
        return Ok(snapshots[index - 1].id.clone());
    }
    Ok(trimmed.to_string())
}

fn join_label(parts: &[String]) -> Option<String> {
    let joined = parts.join(" ");
    let trimmed = joined.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn display_hermes_home(context: &HermesContext) -> String {
    context
        .hermes_home()
        .strip_prefix(context.home_dir())
        .ok()
        .map(|relative| {
            if relative.as_os_str().is_empty() {
                "~".to_string()
            } else {
                format!("~/{}", relative.display())
            }
        })
        .unwrap_or_else(|| context.hermes_home().display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct SnapshotHarness {
        #[command(subcommand)]
        command: Option<SnapshotCommand>,
    }

    #[test]
    fn snapshot_create_collects_multi_word_label() {
        let parsed =
            SnapshotHarness::try_parse_from(["snapshot", "create", "before", "migration"]).unwrap();
        match parsed.command.unwrap() {
            SnapshotCommand::Create { label } => {
                assert_eq!(join_label(&label).as_deref(), Some("before migration"));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }
}
