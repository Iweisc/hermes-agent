//! Top-level argument-parser construction for the `hermes` CLI.
//!
//! This is a native Rust port of `hermes_cli/_parser.py`. The original lives in
//! its own module so other modules (e.g. `relaunch`) can introspect the parser
//! to discover which flags exist without running `main`.
//!
//! Only the top-level parser and the `chat` subparser live here. Every other
//! subparser (model, gateway, sessions, …) is built inline in `main`, because
//! its dispatch is tightly coupled to module-level `cmd_*` functions.
//!
//! The Python source used `argparse`; this port uses `clap`. Argparse's
//! per-action `inherit_on_relaunch` attribute is reproduced here via the
//! [`InheritedFlag`] table and the [`inherited_flags`] / [`chat_inherited_flags`]
//! functions, so the relaunch table builder can find them via introspection
//! exactly as the Python code did.

use clap::{Arg, ArgAction, Command};

/// `--profile` / `-p` is consumed before argparse/clap runs (it sets
/// `HERMES_HOME` and strips itself from argv), so it isn't on the parser.
/// Listed here so all "carry over on relaunch" metadata lives in one file.
///
/// Each tuple is `(flag, takes_value)`.
pub const PRE_ARGPARSE_INHERITED_FLAGS: &[(&str, bool)] = &[("--profile", true), ("-p", true)];

/// Metadata for a flag that `relaunch` should carry over when the CLI re-execs
/// itself (e.g. after `sessions browse` picks a session, or after the setup
/// wizard launches chat).
///
/// This is the structured equivalent of the Python `inherit_on_relaunch = True`
/// attribute tagged onto an argparse Action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InheritedFlag {
    /// The primary long flag, e.g. `--model`.
    pub long: &'static str,
    /// The short flag if any, e.g. `-m`.
    pub short: Option<&'static str>,
    /// Whether the flag takes a value (`true`) or is a store_true switch.
    pub takes_value: bool,
    /// Whether the flag may be repeated (argparse `action="append"`).
    pub appendable: bool,
}

/// The epilogue shown in `hermes --help`. Mirrors `_EPILOGUE` verbatim.
pub const EPILOGUE: &str = "\nExamples:\n    hermes                        Start interactive chat\n    hermes chat -q \"Hello\"        Single query mode\n    hermes -c                     Resume the most recent session\n    hermes -c \"my project\"        Resume a session by name (latest in lineage)\n    hermes --resume <session_id>  Resume a specific session by ID\n    hermes setup                  Run setup wizard\n    hermes logout                 Clear stored authentication\n    hermes auth add <provider>    Add a pooled credential\n    hermes auth list              List pooled credentials\n    hermes auth remove <p> <t>    Remove pooled credential by index, id, or label\n    hermes auth reset <provider>  Clear exhaustion status for a provider\n    hermes model                  Select default model\n    hermes fallback [list]        Show fallback provider chain\n    hermes fallback add           Add a fallback provider (same picker as `hermes model`)\n    hermes fallback remove        Remove a fallback provider from the chain\n    hermes config                 View configuration\n    hermes config edit            Edit config in $EDITOR\n    hermes config set model gpt-4 Set a config value\n    hermes gateway                Run messaging gateway\n    hermes -s hermes-agent-dev,github-auth\n    hermes -w                     Start in isolated git worktree\n    hermes gateway install        Install gateway background service\n    hermes sessions list          List past sessions\n    hermes sessions browse        Interactive session picker\n    hermes sessions rename ID T   Rename/title a session\n    hermes logs                   View agent.log (last 50 lines)\n    hermes logs -f                Follow agent.log in real time\n    hermes logs errors            View errors.log\n    hermes logs --since 1h        Lines from the last hour\n    hermes debug share             Upload debug report for support\n    hermes update                 Update to latest version\n\nFor more help on a command:\n    hermes <command> --help\n";

/// The list of top-level flags tagged `inherit_on_relaunch = True` in the
/// Python source. Order matches registration order in `build_top_level_parser`.
pub fn inherited_flags() -> Vec<InheritedFlag> {
    vec![
        InheritedFlag {
            long: "--model",
            short: Some("-m"),
            takes_value: true,
            appendable: false,
        },
        InheritedFlag {
            long: "--provider",
            short: None,
            takes_value: true,
            appendable: false,
        },
        InheritedFlag {
            long: "--accept-hooks",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--skills",
            short: Some("-s"),
            takes_value: true,
            appendable: true,
        },
        InheritedFlag {
            long: "--yolo",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--pass-session-id",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--ignore-user-config",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--ignore-rules",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--tui",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--dev",
            short: None,
            takes_value: false,
            appendable: false,
        },
    ]
}

/// The list of `chat` subparser flags tagged `inherit_on_relaunch = True`.
/// Order matches registration order in the `chat` block of the Python source.
pub fn chat_inherited_flags() -> Vec<InheritedFlag> {
    vec![
        InheritedFlag {
            long: "--model",
            short: Some("-m"),
            takes_value: true,
            appendable: false,
        },
        InheritedFlag {
            long: "--skills",
            short: Some("-s"),
            takes_value: true,
            appendable: true,
        },
        InheritedFlag {
            long: "--provider",
            short: None,
            takes_value: true,
            appendable: false,
        },
        InheritedFlag {
            long: "--accept-hooks",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--yolo",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--pass-session-id",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--ignore-user-config",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--ignore-rules",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--tui",
            short: None,
            takes_value: false,
            appendable: false,
        },
        InheritedFlag {
            long: "--dev",
            short: None,
            takes_value: false,
            appendable: false,
        },
    ]
}

/// Build the `chat` subcommand, mirroring the `chat_parser` block of the Python
/// source. The caller wires dispatch (the equivalent of
/// `chat_parser.set_defaults(func=cmd_chat)`).
pub fn build_chat_command() -> Command {
    Command::new("chat")
        .about("Interactive chat with the agent")
        .long_about("Start an interactive chat session with Hermes Agent")
        .arg(
            Arg::new("query")
                .short('q')
                .long("query")
                .help("Single query (non-interactive mode)"),
        )
        .arg(
            Arg::new("image")
                .long("image")
                .help("Optional local image path to attach to a single query"),
        )
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .help("Model to use (e.g., anthropic/claude-sonnet-4)"),
        )
        .arg(
            Arg::new("toolsets")
                .short('t')
                .long("toolsets")
                .help("Comma-separated toolsets to enable"),
        )
        .arg(
            // action="append" -> repeatable, collects multiple values.
            Arg::new("skills")
                .short('s')
                .long("skills")
                .action(ArgAction::Append)
                .help("Preload one or more skills for the session (repeat flag or comma-separate)"),
        )
        .arg(
            Arg::new("provider")
                .long("provider")
                .help("Inference provider (default: auto). Built-in or a user-defined name from `providers:` in config.yaml."),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(ArgAction::SetTrue)
                .help("Verbose output"),
        )
        .arg(
            Arg::new("quiet")
                .short('Q')
                .long("quiet")
                .action(ArgAction::SetTrue)
                .help("Quiet mode for programmatic use: suppress banner, spinner, and tool previews. Only output the final response and session info."),
        )
        .arg(
            Arg::new("resume")
                .short('r')
                .long("resume")
                .value_name("SESSION_ID")
                .help("Resume a previous session by ID (shown on exit)"),
        )
        .arg(
            // nargs="?", const=True: optional value; presence-without-value is
            // a sentinel meaning "most recent". Modelled here with num_args(0..=1).
            Arg::new("continue_last")
                .short('c')
                .long("continue")
                .num_args(0..=1)
                .value_name("SESSION_NAME")
                .default_missing_value("\u{0}continue-last")
                .help("Resume a session by name, or the most recent if no name given"),
        )
        .arg(
            Arg::new("worktree")
                .short('w')
                .long("worktree")
                .action(ArgAction::SetTrue)
                .help("Run in an isolated git worktree (for parallel agents on the same repo)"),
        )
        .arg(
            Arg::new("accept-hooks")
                .long("accept-hooks")
                .action(ArgAction::SetTrue)
                .help("Auto-approve any unseen shell hooks declared in config.yaml without a TTY prompt (see also HERMES_ACCEPT_HOOKS env var and hooks_auto_accept: in config.yaml)."),
        )
        .arg(
            Arg::new("checkpoints")
                .long("checkpoints")
                .action(ArgAction::SetTrue)
                .help("Enable filesystem checkpoints before destructive file operations (use /rollback to restore)"),
        )
        .arg(
            Arg::new("max-turns")
                .long("max-turns")
                .value_name("N")
                .value_parser(clap::value_parser!(i64))
                .help("Maximum tool-calling iterations per conversation turn (default: 90, or agent.max_turns in config)"),
        )
        .arg(
            Arg::new("yolo")
                .long("yolo")
                .action(ArgAction::SetTrue)
                .help("Bypass all dangerous command approval prompts (use at your own risk)"),
        )
        .arg(
            Arg::new("pass-session-id")
                .long("pass-session-id")
                .action(ArgAction::SetTrue)
                .help("Include the session ID in the agent's system prompt"),
        )
        .arg(
            Arg::new("ignore-user-config")
                .long("ignore-user-config")
                .action(ArgAction::SetTrue)
                .help("Ignore ~/.hermes/config.yaml and fall back to built-in defaults (credentials in .env are still loaded). Useful for isolated CI runs, reproduction, and third-party integrations."),
        )
        .arg(
            Arg::new("ignore-rules")
                .long("ignore-rules")
                .action(ArgAction::SetTrue)
                .help("Skip auto-injection of AGENTS.md, SOUL.md, .cursorrules, memory, and preloaded skills. Combine with --ignore-user-config for a fully isolated run."),
        )
        .arg(
            Arg::new("source")
                .long("source")
                .help("Session source tag for filtering (default: cli). Use 'tool' for third-party integrations that should not appear in user session lists."),
        )
        .arg(
            Arg::new("tui")
                .long("tui")
                .action(ArgAction::SetTrue)
                .help("Launch the modern TUI instead of the classic REPL"),
        )
        .arg(
            // dest="tui_dev"
            Arg::new("tui_dev")
                .long("dev")
                .action(ArgAction::SetTrue)
                .help("With --tui: run TypeScript sources via tsx (skip dist build)"),
        )
}

/// Build the top-level parser plus the `chat` subcommand, mirroring
/// `build_top_level_parser`.
///
/// In Python this returned `(parser, subparsers, chat_parser)` so the caller
/// could keep registering other subparsers. In clap, subcommands are attached
/// to the returned [`Command`]; the caller registers other subcommands via
/// `cmd.subcommand(...)`. The `chat` subcommand is already attached.
pub fn build_top_level_parser() -> Command {
    Command::new("hermes")
        .about("Hermes Agent - AI assistant with tool-calling capabilities")
        .after_help(EPILOGUE)
        .arg(
            Arg::new("version")
                .short('V')
                .long("version")
                .action(ArgAction::SetTrue)
                .help("Show version and exit"),
        )
        .arg(
            Arg::new("oneshot")
                .short('z')
                .long("oneshot")
                .value_name("PROMPT")
                .help("One-shot mode: send a single prompt and print ONLY the final response text to stdout. No banner, no spinner, no tool previews, no session_id line. Tools, memory, rules, and AGENTS.md in the CWD are loaded as normal; approvals are auto-bypassed. Intended for scripts / pipes."),
        )
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .help("Model override for this invocation (e.g. anthropic/claude-sonnet-4.6). Applies to -z/--oneshot and --tui. Also settable via HERMES_INFERENCE_MODEL env var."),
        )
        .arg(
            Arg::new("provider")
                .long("provider")
                .help("Provider override for this invocation (e.g. openrouter, anthropic). Applies to -z/--oneshot and --tui. Also settable via HERMES_INFERENCE_PROVIDER env var."),
        )
        .arg(
            Arg::new("toolsets")
                .short('t')
                .long("toolsets")
                .help("Comma-separated toolsets to enable for this invocation. Applies to -z/--oneshot and --tui."),
        )
        .arg(
            Arg::new("resume")
                .short('r')
                .long("resume")
                .value_name("SESSION")
                .help("Resume a previous session by ID or title"),
        )
        .arg(
            Arg::new("continue_last")
                .short('c')
                .long("continue")
                .num_args(0..=1)
                .value_name("SESSION_NAME")
                .default_missing_value("\u{0}continue-last")
                .help("Resume a session by name, or the most recent if no name given"),
        )
        .arg(
            Arg::new("worktree")
                .short('w')
                .long("worktree")
                .action(ArgAction::SetTrue)
                .help("Run in an isolated git worktree (for parallel agents)"),
        )
        .arg(
            Arg::new("accept-hooks")
                .long("accept-hooks")
                .action(ArgAction::SetTrue)
                .help("Auto-approve any unseen shell hooks declared in config.yaml without a TTY prompt.  Equivalent to HERMES_ACCEPT_HOOKS=1 or hooks_auto_accept: true in config.yaml.  Use on CI / headless runs that can't prompt."),
        )
        .arg(
            Arg::new("skills")
                .short('s')
                .long("skills")
                .action(ArgAction::Append)
                .help("Preload one or more skills for the session (repeat flag or comma-separate)"),
        )
        .arg(
            Arg::new("yolo")
                .long("yolo")
                .action(ArgAction::SetTrue)
                .help("Bypass all dangerous command approval prompts (use at your own risk)"),
        )
        .arg(
            Arg::new("pass-session-id")
                .long("pass-session-id")
                .action(ArgAction::SetTrue)
                .help("Include the session ID in the agent's system prompt"),
        )
        .arg(
            Arg::new("ignore-user-config")
                .long("ignore-user-config")
                .action(ArgAction::SetTrue)
                .help("Ignore ~/.hermes/config.yaml and fall back to built-in defaults (credentials in .env are still loaded)"),
        )
        .arg(
            Arg::new("ignore-rules")
                .long("ignore-rules")
                .action(ArgAction::SetTrue)
                .help("Skip auto-injection of AGENTS.md, SOUL.md, .cursorrules, memory, and preloaded skills"),
        )
        .arg(
            Arg::new("tui")
                .long("tui")
                .action(ArgAction::SetTrue)
                .help("Launch the modern TUI instead of the classic REPL"),
        )
        .arg(
            Arg::new("tui_dev")
                .long("dev")
                .action(ArgAction::SetTrue)
                .help("With --tui: run TypeScript sources via tsx (skip dist build)"),
        )
        .subcommand(build_chat_command())
}

/// Sentinel value stored when `--continue` / `-c` is passed without an argument
/// (argparse `const=True`). Callers should treat this as "resume most recent".
pub const CONTINUE_LAST_SENTINEL: &str = "\u{0}continue-last";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_builds_and_has_chat_subcommand() {
        let cmd = build_top_level_parser();
        assert_eq!(cmd.get_name(), "hermes");
        assert!(
            cmd.get_subcommands().any(|s| s.get_name() == "chat"),
            "chat subcommand must be registered"
        );
    }

    #[test]
    fn top_level_oneshot_and_model_parse() {
        let m = build_top_level_parser()
            .try_get_matches_from(["hermes", "-z", "hello", "-m", "anthropic/x"])
            .expect("should parse");
        assert_eq!(m.get_one::<String>("oneshot").map(String::as_str), Some("hello"));
        assert_eq!(m.get_one::<String>("model").map(String::as_str), Some("anthropic/x"));
    }

    #[test]
    fn version_flag_is_store_true() {
        let m = build_top_level_parser()
            .try_get_matches_from(["hermes", "-V"])
            .expect("parse");
        assert!(m.get_flag("version"));
    }

    #[test]
    fn continue_without_value_yields_sentinel() {
        let m = build_top_level_parser()
            .try_get_matches_from(["hermes", "-c"])
            .expect("parse");
        assert_eq!(
            m.get_one::<String>("continue_last").map(String::as_str),
            Some(CONTINUE_LAST_SENTINEL)
        );
    }

    #[test]
    fn continue_with_value_keeps_value() {
        let m = build_top_level_parser()
            .try_get_matches_from(["hermes", "-c", "my project"])
            .expect("parse");
        assert_eq!(
            m.get_one::<String>("continue_last").map(String::as_str),
            Some("my project")
        );
    }

    #[test]
    fn skills_are_appendable() {
        let m = build_top_level_parser()
            .try_get_matches_from(["hermes", "-s", "a", "-s", "b"])
            .expect("parse");
        let vals: Vec<&String> = m.get_many::<String>("skills").unwrap().collect();
        assert_eq!(vals, vec!["a", "b"]);
    }

    #[test]
    fn chat_subcommand_parses_query_and_flags() {
        let cmd = build_top_level_parser();
        let m = cmd
            .try_get_matches_from(["hermes", "chat", "-q", "hi", "-v", "--max-turns", "5"])
            .expect("parse");
        let (name, sub) = m.subcommand().expect("subcommand present");
        assert_eq!(name, "chat");
        assert_eq!(sub.get_one::<String>("query").map(String::as_str), Some("hi"));
        assert!(sub.get_flag("verbose"));
        assert_eq!(sub.get_one::<i64>("max-turns").copied(), Some(5));
    }

    #[test]
    fn inherited_flag_tables_match_python_tags() {
        let top = inherited_flags();
        // 10 top-level flags tagged in the Python source.
        assert_eq!(top.len(), 10);
        assert!(top.iter().any(|f| f.long == "--model" && f.short == Some("-m")));
        assert!(top.iter().any(|f| f.long == "--skills" && f.appendable));
        assert!(top.iter().any(|f| f.long == "--yolo" && !f.takes_value));

        let chat = chat_inherited_flags();
        assert_eq!(chat.len(), 10);
        assert!(chat.iter().any(|f| f.long == "--provider" && f.takes_value));
    }

    #[test]
    fn pre_argparse_inherited_flags_present() {
        assert_eq!(
            PRE_ARGPARSE_INHERITED_FLAGS,
            &[("--profile", true), ("-p", true)]
        );
    }
}
