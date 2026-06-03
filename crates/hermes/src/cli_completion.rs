//! Shell completion script generation for the hermes CLI.
//!
//! Native Rust port of `hermes_cli/completion.py`.
//!
//! The Python version walks the live `argparse` parser tree to generate
//! accurate, always-up-to-date completion scripts. This port operates on an
//! equivalent in-memory parser tree ([`ParserTree`]) so callers can build the
//! tree from whatever command source they have (clap, a hand-built tree, or a
//! tree deserialised from the Python parser) and emit identical bash, zsh, and
//! fish completion scripts.
//!
//! Supports bash, zsh, and fish.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// A single parser node: its flags plus its (canonical-name keyed)
/// subcommands. Mirrors the dict returned by the Python `_walk` helper, i.e.
/// `{"flags": [...], "subcommands": {...}, "help": "..."}`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParserTree {
    /// Option strings beginning with `-` (in definition order, like argparse).
    pub flags: Vec<String>,
    /// Subcommands keyed by their canonical name. A `BTreeMap` keeps the keys
    /// sorted, matching Python's `sorted(tree["subcommands"])` iteration used
    /// throughout the generators.
    pub subcommands: BTreeMap<String, ParserTree>,
    /// Cleaned help text for this node (empty for the root).
    pub help: String,
}

impl ParserTree {
    /// Create an empty parser tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience builder: add a flag.
    pub fn with_flag(mut self, flag: impl Into<String>) -> Self {
        self.flags.push(flag.into());
        self
    }

    /// Convenience builder: add a subcommand.
    pub fn with_subcommand(mut self, name: impl Into<String>, sub: ParserTree) -> Self {
        self.subcommands.insert(name.into(), sub);
        self
    }

    /// Convenience builder: set help text (will be cleaned).
    pub fn with_help(mut self, help: &str) -> Self {
        self.help = clean(help, DEFAULT_MAXLEN);
        self
    }
}

/// Default truncation length used by [`clean`], matching Python's `maxlen=60`.
pub const DEFAULT_MAXLEN: usize = 60;

/// Strip shell-unsafe characters and truncate.
///
/// Port of Python `_clean`: removes single quotes, double quotes and
/// backslashes, then truncates to `maxlen` characters.
pub fn clean(text: &str, maxlen: usize) -> String {
    text.chars()
        .filter(|c| *c != '\'' && *c != '"' && *c != '\\')
        .take(maxlen)
        .collect()
}

// ---------------------------------------------------------------------------
// Bash
// ---------------------------------------------------------------------------

/// Generate a bash completion script for the given parser tree.
///
/// Port of Python `generate_bash`.
pub fn generate_bash(tree: &ParserTree) -> String {
    // Keys of a BTreeMap are already sorted; collect for the top-level list.
    let top_cmds = tree
        .subcommands
        .keys()
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");

    let mut cases: Vec<String> = Vec::new();
    for (cmd, info) in &tree.subcommands {
        if cmd == "profile" && !info.subcommands.is_empty() {
            // Profile subcommand: complete actions, then profile names for
            // actions that accept a profile argument.
            let subcmds = info
                .subcommands
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            let profile_actions = "use delete show alias rename export";
            let profile_actions_pat = profile_actions.replace(' ', "|");
            cases.push(format!(
                "        profile)\n\
                 \x20           case \"$prev\" in\n\
                 \x20               profile)\n\
                 \x20                   COMPREPLY=($(compgen -W \"{subcmds}\" -- \"$cur\"))\n\
                 \x20                   return\n\
                 \x20                   ;;\n\
                 \x20               {profile_actions_pat})\n\
                 \x20                   COMPREPLY=($(compgen -W \"$(_hermes_profiles)\" -- \"$cur\"))\n\
                 \x20                   return\n\
                 \x20                   ;;\n\
                 \x20           esac\n\
                 \x20           ;;",
            ));
        } else if !info.subcommands.is_empty() {
            let subcmds = info
                .subcommands
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            cases.push(format!(
                "        {cmd})\n\
                 \x20           COMPREPLY=($(compgen -W \"{subcmds}\" -- \"$cur\"))\n\
                 \x20           return\n\
                 \x20           ;;",
            ));
        } else if !info.flags.is_empty() {
            let flags = info.flags.join(" ");
            cases.push(format!(
                "        {cmd})\n\
                 \x20           COMPREPLY=($(compgen -W \"{flags}\" -- \"$cur\"))\n\
                 \x20           return\n\
                 \x20           ;;",
            ));
        }
    }

    let cases_str = cases.join("\n");

    format!(
        "# Hermes Agent bash completion\n\
         # Add to ~/.bashrc:\n\
         #   eval \"$(hermes completion bash)\"\n\
         \n\
         _hermes_profiles() {{\n\
         \x20   local profiles_dir=\"$HOME/.hermes/profiles\"\n\
         \x20   local profiles=\"default\"\n\
         \x20   if [ -d \"$profiles_dir\" ]; then\n\
         \x20       profiles=\"$profiles $(ls \"$profiles_dir\" 2>/dev/null)\"\n\
         \x20   fi\n\
         \x20   echo \"$profiles\"\n\
         }}\n\
         \n\
         _hermes_completion() {{\n\
         \x20   local cur prev\n\
         \x20   COMPREPLY=()\n\
         \x20   cur=\"${{COMP_WORDS[COMP_CWORD]}}\"\n\
         \x20   prev=\"${{COMP_WORDS[COMP_CWORD-1]}}\"\n\
         \n\
         \x20   # Complete profile names after -p / --profile\n\
         \x20   if [[ \"$prev\" == \"-p\" || \"$prev\" == \"--profile\" ]]; then\n\
         \x20       COMPREPLY=($(compgen -W \"$(_hermes_profiles)\" -- \"$cur\"))\n\
         \x20       return\n\
         \x20   fi\n\
         \n\
         \x20   if [[ $COMP_CWORD -ge 2 ]]; then\n\
         \x20       case \"${{COMP_WORDS[1]}}\" in\n\
         {cases_str}\n\
         \x20       esac\n\
         \x20   fi\n\
         \n\
         \x20   if [[ $COMP_CWORD -eq 1 ]]; then\n\
         \x20       COMPREPLY=($(compgen -W \"{top_cmds}\" -- \"$cur\"))\n\
         \x20   fi\n\
         }}\n\
         \n\
         complete -F _hermes_completion hermes\n",
    )
}

// ---------------------------------------------------------------------------
// Zsh
// ---------------------------------------------------------------------------

/// Generate a zsh completion script for the given parser tree.
///
/// Port of Python `generate_zsh`.
pub fn generate_zsh(tree: &ParserTree) -> String {
    let mut top_cmds_lines: Vec<String> = Vec::new();
    for (cmd, info) in &tree.subcommands {
        let help_text = clean(&info.help, DEFAULT_MAXLEN);
        top_cmds_lines.push(format!("                '{cmd}:{help_text}'"));
    }
    let top_cmds_str = top_cmds_lines.join("\n");

    let mut sub_cases: Vec<String> = Vec::new();
    for (cmd, info) in &tree.subcommands {
        if info.subcommands.is_empty() {
            continue;
        }
        if cmd == "profile" {
            // Profile subcommand: complete actions, then profile names for
            // actions that accept a profile argument.
            let mut sub_lines: Vec<String> = Vec::new();
            for (sc, sinfo) in &info.subcommands {
                let sh = clean(&sinfo.help, DEFAULT_MAXLEN);
                sub_lines.push(format!("                        '{sc}:{sh}'"));
            }
            let sub_str = sub_lines.join("\n");
            sub_cases.push(format!(
                "                profile)\n\
                 \x20                   case ${{line[2]}} in\n\
                 \x20                       use|delete|show|alias|rename|export)\n\
                 \x20                           _hermes_profiles\n\
                 \x20                           ;;\n\
                 \x20                       *)\n\
                 \x20                           local -a profile_cmds\n\
                 \x20                           profile_cmds=(\n\
                 {sub_str}\n\
                 \x20                           )\n\
                 \x20                           _describe 'profile command' profile_cmds\n\
                 \x20                           ;;\n\
                 \x20                   esac\n\
                 \x20                   ;;",
            ));
        } else {
            let mut sub_lines: Vec<String> = Vec::new();
            for (sc, sinfo) in &info.subcommands {
                let sh = clean(&sinfo.help, DEFAULT_MAXLEN);
                sub_lines.push(format!("                    '{sc}:{sh}'"));
            }
            let sub_str = sub_lines.join("\n");
            let safe = cmd.replace('-', "_");
            sub_cases.push(format!(
                "                {cmd})\n\
                 \x20                   local -a {safe}_cmds\n\
                 \x20                   {safe}_cmds=(\n\
                 {sub_str}\n\
                 \x20                   )\n\
                 \x20                   _describe '{cmd} command' {safe}_cmds\n\
                 \x20                   ;;",
            ));
        }
    }
    let sub_cases_str = sub_cases.join("\n");

    format!(
        "#compdef hermes\n\
         # Hermes Agent zsh completion\n\
         # Add to ~/.zshrc:\n\
         #   eval \"$(hermes completion zsh)\"\n\
         \n\
         _hermes_profiles() {{\n\
         \x20   local -a profiles\n\
         \x20   profiles=(default)\n\
         \x20   if [[ -d \"$HOME/.hermes/profiles\" ]]; then\n\
         \x20       profiles+=(\"${{(@f)$(ls $HOME/.hermes/profiles 2>/dev/null)}}\")\n\
         \x20   fi\n\
         \x20   _describe 'profile' profiles\n\
         }}\n\
         \n\
         _hermes() {{\n\
         \x20   local context state line\n\
         \x20   typeset -A opt_args\n\
         \n\
         \x20   _arguments -C \\\n\
         \x20       '(-h --help){{-h,--help}}[Show help and exit]' \\\n\
         \x20       '(-V --version){{-V,--version}}[Show version and exit]' \\\n\
         \x20       '(-p --profile){{-p,--profile}}[Profile name]:profile:_hermes_profiles' \\\n\
         \x20       '1:command:->commands' \\\n\
         \x20       '*::arg:->args'\n\
         \n\
         \x20   case $state in\n\
         \x20       commands)\n\
         \x20           local -a subcmds\n\
         \x20           subcmds=(\n\
         {top_cmds_str}\n\
         \x20           )\n\
         \x20           _describe 'hermes command' subcmds\n\
         \x20           ;;\n\
         \x20       args)\n\
         \x20           case ${{line[1]}} in\n\
         {sub_cases_str}\n\
         \x20           esac\n\
         \x20           ;;\n\
         \x20   esac\n\
         }}\n\
         \n\
         _hermes \"$@\"\n",
    )
}

// ---------------------------------------------------------------------------
// Fish
// ---------------------------------------------------------------------------

/// Profile actions that should complete profile names in fish.
fn profile_name_actions() -> BTreeSet<&'static str> {
    ["use", "delete", "show", "alias", "rename", "export"]
        .into_iter()
        .collect()
}

/// Generate a fish completion script for the given parser tree.
///
/// Port of Python `generate_fish`.
pub fn generate_fish(tree: &ParserTree) -> String {
    let top_cmds = tree.subcommands.keys().cloned().collect::<Vec<_>>();
    let top_cmds_str = top_cmds.join(" ");

    let mut lines: Vec<String> = vec![
        "# Hermes Agent fish completion".to_string(),
        "# Add to your config:".to_string(),
        "#   hermes completion fish | source".to_string(),
        String::new(),
        "# Helper: list available profiles".to_string(),
        "function __hermes_profiles".to_string(),
        "    echo default".to_string(),
        "    if test -d $HOME/.hermes/profiles".to_string(),
        "        ls $HOME/.hermes/profiles 2>/dev/null".to_string(),
        "    end".to_string(),
        "end".to_string(),
        String::new(),
        "# Disable file completion by default".to_string(),
        "complete -c hermes -f".to_string(),
        String::new(),
        "# Complete profile names after -p / --profile".to_string(),
        "complete -c hermes -f -s p -l profile -d 'Profile name' -xa '(__hermes_profiles)'"
            .to_string(),
        String::new(),
        "# Top-level subcommands".to_string(),
    ];

    for cmd in &top_cmds {
        let info = &tree.subcommands[cmd];
        let help_text = clean(&info.help, DEFAULT_MAXLEN);
        lines.push(format!(
            "complete -c hermes -f \
             -n 'not __fish_seen_subcommand_from {top_cmds_str}' \
             -a {cmd} -d '{help_text}'"
        ));
    }

    lines.push(String::new());
    lines.push("# Subcommand completions".to_string());

    let profile_actions = profile_name_actions();

    for cmd in &top_cmds {
        let info = &tree.subcommands[cmd];
        if info.subcommands.is_empty() {
            continue;
        }
        lines.push(format!("# {cmd}"));
        for (sc, sinfo) in &info.subcommands {
            let sh = clean(&sinfo.help, DEFAULT_MAXLEN);
            lines.push(format!(
                "complete -c hermes -f \
                 -n '__fish_seen_subcommand_from {cmd}' \
                 -a {sc} -d '{sh}'"
            ));
        }
        // For profile subcommand, complete profile names for relevant actions.
        if cmd == "profile" {
            for action in &profile_actions {
                lines.push(format!(
                    "complete -c hermes -f \
                     -n '__fish_seen_subcommand_from {action}; \
                     and __fish_seen_subcommand_from profile' \
                     -a '(__hermes_profiles)' -d 'Profile name'"
                ));
            }
        }
    }

    lines.push(String::new());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a parser tree resembling the hermes CLI for exercising the
    /// generators, including a `profile` subcommand with nested actions.
    fn sample_tree() -> ParserTree {
        let mut root = ParserTree::new();

        // A flag-only leaf command.
        let mut debug = ParserTree::new().with_help("Debug helpers");
        debug.flags = vec!["-v".to_string(), "--verbose".to_string()];
        root.subcommands.insert("debug".to_string(), debug);

        // A command with subcommands (non-profile).
        let mut sessions = ParserTree::new().with_help("Manage sessions");
        sessions
            .subcommands
            .insert("browse".to_string(), ParserTree::new().with_help("Browse"));
        sessions
            .subcommands
            .insert("resume".to_string(), ParserTree::new().with_help("Resume"));
        root.subcommands.insert("sessions".to_string(), sessions);

        // The profile command with profile-name actions.
        let mut profile = ParserTree::new().with_help("Profile management");
        for action in ["use", "delete", "show", "alias", "rename", "export", "current"] {
            profile.subcommands.insert(
                action.to_string(),
                ParserTree::new().with_help(&format!("{action} profile")),
            );
        }
        root.subcommands.insert("profile".to_string(), profile);

        // A hyphenated command name to exercise the zsh safe-name path.
        let mut set_home = ParserTree::new().with_help("Set home");
        set_home
            .subcommands
            .insert("show".to_string(), ParserTree::new().with_help("Show"));
        root.subcommands.insert("set-home".to_string(), set_home);

        root
    }

    #[test]
    fn clean_strips_unsafe_chars_and_truncates() {
        assert_eq!(clean("a'b\"c\\d", DEFAULT_MAXLEN), "abcd");
        let long = "x".repeat(100);
        assert_eq!(clean(&long, 60).len(), 60);
        // Truncation counts characters, not bytes.
        let s = clean("hello world this is plain", DEFAULT_MAXLEN);
        assert_eq!(s, "hello world this is plain");
    }

    #[test]
    fn bash_completion_lists_top_level_commands() {
        let tree = sample_tree();
        let script = generate_bash(&tree);
        assert!(script.contains("complete -F _hermes_completion hermes"));
        assert!(script.contains("_hermes_profiles"));
        // Top-level commands are sorted alphabetically.
        assert!(script.contains(
            "COMPREPLY=($(compgen -W \"debug profile sessions set-home\" -- \"$cur\"))"
        ));
        // Leaf with flags emits a flag case.
        assert!(script.contains("compgen -W \"-v --verbose\""));
    }

    #[test]
    fn bash_completion_has_profile_special_case() {
        let tree = sample_tree();
        let script = generate_bash(&tree);
        assert!(script.contains("        profile)"));
        assert!(script.contains("case \"$prev\" in"));
        // Profile-name actions are pipe-joined.
        assert!(script.contains("use|delete|show|alias|rename|export)"));
    }

    #[test]
    fn bash_completion_non_profile_subcommand_case() {
        let tree = sample_tree();
        let script = generate_bash(&tree);
        // sessions subcommands sorted alphabetically.
        assert!(script.contains("        sessions)"));
        assert!(script.contains("compgen -W \"browse resume\""));
    }

    #[test]
    fn zsh_completion_includes_profile_helper_and_case() {
        let tree = sample_tree();
        let script = generate_zsh(&tree);
        assert!(script.starts_with("#compdef hermes\n"));
        assert!(script.contains("_hermes_profiles"));
        assert!(script.contains("_describe 'profile command' profile_cmds"));
        assert!(script.contains("use|delete|show|alias|rename|export)"));
    }

    #[test]
    fn zsh_completion_safe_name_for_hyphenated_command() {
        let tree = sample_tree();
        let script = generate_zsh(&tree);
        // "set-home" -> "set_home_cmds"
        assert!(script.contains("local -a set_home_cmds"));
        assert!(script.contains("_describe 'set-home command' set_home_cmds"));
    }

    #[test]
    fn zsh_top_level_entries_have_help() {
        let tree = sample_tree();
        let script = generate_zsh(&tree);
        assert!(script.contains("'debug:Debug helpers'"));
        assert!(script.contains("'sessions:Manage sessions'"));
    }

    #[test]
    fn fish_completion_contains_subcommand_entries() {
        let tree = sample_tree();
        let script = generate_fish(&tree);
        assert!(script.contains("function __hermes_profiles"));
        assert!(script.contains("complete -c hermes -f -s p -l profile"));
        assert!(script.contains("__fish_seen_subcommand_from sessions"));
        assert!(script.contains("-a browse -d 'Browse'"));
        // Disable-file-completion line present.
        assert!(script.contains("# Disable file completion by default"));
    }

    #[test]
    fn fish_completion_profile_name_actions() {
        let tree = sample_tree();
        let script = generate_fish(&tree);
        // For each profile-name action, a profile-name completion line exists.
        for action in ["use", "delete", "show", "alias", "rename", "export"] {
            let needle = format!(
                "-n '__fish_seen_subcommand_from {action}; \
                 and __fish_seen_subcommand_from profile' \
                 -a '(__hermes_profiles)' -d 'Profile name'"
            );
            assert!(
                script.contains(&needle),
                "missing fish profile action line for {action}"
            );
        }
    }

    #[test]
    fn fish_top_level_uses_not_seen_guard() {
        let tree = sample_tree();
        let script = generate_fish(&tree);
        assert!(script.contains(
            "-n 'not __fish_seen_subcommand_from debug profile sessions set-home'"
        ));
    }

    #[test]
    fn builders_compose() {
        let tree = ParserTree::new()
            .with_flag("--x")
            .with_subcommand("foo", ParserTree::new().with_help("Foo command"));
        assert_eq!(tree.flags, vec!["--x".to_string()]);
        assert_eq!(tree.subcommands["foo"].help, "Foo command");
    }
}
