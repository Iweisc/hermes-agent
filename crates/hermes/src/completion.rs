use std::error::Error;

use clap::{Args, CommandFactory, ValueEnum};

use crate::Cli;

#[derive(Args, Debug)]
pub struct CompletionArgs {
    #[arg(value_enum, default_value_t = CompletionShell::Bash)]
    pub shell: CompletionShell,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum CompletionShell {
    Bash,
    Zsh,
    Fish,
}

#[derive(Debug, Clone, Default)]
struct CommandInfo {
    names: Vec<String>,
    help: String,
    flags: Vec<String>,
    subcommands: Vec<CommandInfo>,
}

pub fn print_completion(args: CompletionArgs) -> Result<(), Box<dyn Error>> {
    let tree = walk_root_command(&mut Cli::command());
    let script = match args.shell {
        CompletionShell::Bash => generate_bash(&tree),
        CompletionShell::Zsh => generate_zsh(&tree),
        CompletionShell::Fish => generate_fish(&tree),
    };
    println!("{script}");
    Ok(())
}

fn walk_root_command(command: &mut clap::Command) -> CommandInfo {
    let mut info = walk_command(command);
    info.names.clear();
    info
}

fn walk_command(command: &mut clap::Command) -> CommandInfo {
    command.build();
    let mut names = vec![command.get_name().to_string()];
    names.extend(command.get_all_aliases().map(str::to_string));
    names.sort();
    names.dedup();
    if let Some(index) = names.iter().position(|name| name == command.get_name()) {
        names.swap(0, index);
    }
    let mut info = CommandInfo {
        names,
        help: clean(
            command
                .get_about()
                .map(|value| value.to_string())
                .unwrap_or_default(),
        ),
        flags: command_flags(command),
        subcommands: Vec::new(),
    };
    for subcommand in command.get_subcommands_mut() {
        if subcommand.is_hide_set() {
            continue;
        }
        info.subcommands.push(walk_command(subcommand));
    }
    merge_compat_subcommands(&mut info);
    info.subcommands.sort_by(|left, right| {
        left.canonical_name()
            .cmp(right.canonical_name())
            .then_with(|| left.names.cmp(&right.names))
    });
    info
}

fn command_flags(command: &clap::Command) -> Vec<String> {
    let mut flags = command
        .get_arguments()
        .filter(|arg| !arg.is_hide_set())
        .flat_map(|arg| {
            let mut values = Vec::new();
            if let Some(short) = arg.get_short() {
                values.push(format!("-{short}"));
            }
            if let Some(short_aliases) = arg.get_all_short_aliases() {
                values.extend(short_aliases.into_iter().map(|alias| format!("-{alias}")));
            }
            if let Some(long) = arg.get_long() {
                values.push(format!("--{long}"));
            }
            if let Some(long_aliases) = arg.get_all_aliases() {
                values.extend(long_aliases.into_iter().map(|alias| format!("--{alias}")));
            }
            values
        })
        .collect::<Vec<_>>();
    flags.sort();
    flags.dedup();
    flags
}

impl CommandInfo {
    fn canonical_name(&self) -> &str {
        self.names.first().map(String::as_str).unwrap_or("")
    }

    fn case_pattern(&self) -> String {
        self.names.join("|")
    }

    fn all_subcommand_names(&self) -> Vec<String> {
        self.subcommands
            .iter()
            .flat_map(|info| info.names.iter().cloned())
            .collect()
    }
}

fn compat_command(name: &str, aliases: &[&str], subcommands: Vec<CommandInfo>) -> CommandInfo {
    let mut names = vec![name.to_string()];
    names.extend(aliases.iter().map(|alias| alias.to_string()));
    CommandInfo {
        names,
        help: String::new(),
        flags: Vec::new(),
        subcommands,
    }
}

fn merge_compat_subcommands(info: &mut CommandInfo) {
    for compat in compat_subcommands_for(info.canonical_name()) {
        if info
            .subcommands
            .iter()
            .any(|existing| existing.canonical_name() == compat.canonical_name())
        {
            continue;
        }
        info.subcommands.push(compat);
    }
}

fn compat_subcommands_for(name: &str) -> Vec<CommandInfo> {
    match name {
        "approve" => vec![
            compat_command("all", &[], Vec::new()),
            compat_command("session", &[], Vec::new()),
            compat_command("always", &[], Vec::new()),
        ],
        "browser" => vec![
            compat_command("connect", &[], Vec::new()),
            compat_command("disconnect", &[], Vec::new()),
            compat_command("status", &[], Vec::new()),
        ],
        "deny" => vec![compat_command("all", &[], Vec::new())],
        "busy" => vec![
            compat_command("queue", &[], Vec::new()),
            compat_command("steer", &[], Vec::new()),
            compat_command("interrupt", &[], Vec::new()),
            compat_command("status", &[], Vec::new()),
        ],
        "fast" => vec![
            compat_command("normal", &[], Vec::new()),
            compat_command("fast", &[], Vec::new()),
            compat_command("status", &[], Vec::new()),
            compat_command("on", &[], Vec::new()),
            compat_command("off", &[], Vec::new()),
        ],
        "footer" => vec![
            compat_command("on", &[], Vec::new()),
            compat_command("off", &[], Vec::new()),
            compat_command("status", &[], Vec::new()),
        ],
        "goal" => vec![
            compat_command("status", &[], Vec::new()),
            compat_command("pause", &[], Vec::new()),
            compat_command("resume", &[], Vec::new()),
            compat_command("clear", &[], Vec::new()),
        ],
        "rollback" => vec![compat_command("diff", &[], Vec::new())],
        "indicator" => vec![
            compat_command("kaomoji", &[], Vec::new()),
            compat_command("emoji", &[], Vec::new()),
            compat_command("unicode", &[], Vec::new()),
            compat_command("ascii", &[], Vec::new()),
        ],
        "kanban" => vec![
            compat_command(
                "boards",
                &[],
                vec![
                    compat_command("list", &["ls"], Vec::new()),
                    compat_command("create", &[], Vec::new()),
                    compat_command("remove", &["rm"], Vec::new()),
                    compat_command("switch", &[], Vec::new()),
                    compat_command("show", &[], Vec::new()),
                    compat_command("rename", &[], Vec::new()),
                ],
            ),
            compat_command("list", &["ls"], Vec::new()),
            compat_command("show", &[], Vec::new()),
            compat_command("create", &[], Vec::new()),
            compat_command("assign", &[], Vec::new()),
            compat_command("link", &[], Vec::new()),
            compat_command("unlink", &[], Vec::new()),
            compat_command("claim", &[], Vec::new()),
            compat_command("comment", &[], Vec::new()),
            compat_command("complete", &[], Vec::new()),
            compat_command("block", &[], Vec::new()),
            compat_command("unblock", &[], Vec::new()),
            compat_command("archive", &[], Vec::new()),
            compat_command("tail", &[], Vec::new()),
            compat_command("dispatch", &[], Vec::new()),
            compat_command("context", &[], Vec::new()),
            compat_command("init", &[], Vec::new()),
            compat_command("gc", &[], Vec::new()),
        ],
        "reasoning" => vec![
            compat_command("none", &[], Vec::new()),
            compat_command("minimal", &[], Vec::new()),
            compat_command("low", &[], Vec::new()),
            compat_command("medium", &[], Vec::new()),
            compat_command("high", &[], Vec::new()),
            compat_command("xhigh", &[], Vec::new()),
            compat_command("show", &[], Vec::new()),
            compat_command("hide", &[], Vec::new()),
            compat_command("on", &[], Vec::new()),
            compat_command("off", &[], Vec::new()),
        ],
        "topic" => vec![
            compat_command("help", &[], Vec::new()),
            compat_command("off", &[], Vec::new()),
        ],
        "voice" => vec![
            compat_command("on", &[], Vec::new()),
            compat_command("off", &[], Vec::new()),
            compat_command("tts", &[], Vec::new()),
            compat_command("status", &[], Vec::new()),
        ],
        _ => Vec::new(),
    }
}

fn clean(text: String) -> String {
    text.replace(['\'', '"', '\\'], "")
        .chars()
        .take(60)
        .collect()
}

fn generate_bash(tree: &CommandInfo) -> String {
    let top_cmds = tree
        .subcommands
        .iter()
        .flat_map(|info| info.names.iter().cloned())
        .collect::<Vec<_>>();
    let top_cmds_str = top_cmds.join(" ");
    let top_flags = tree.flags.join(" ");
    let mut cases = Vec::new();
    for info in &tree.subcommands {
        cases.push(build_bash_case(info, 1));
    }
    format!(
        "# Hermes Agent bash completion\n\
         # Add to ~/.bashrc:\n\
         #   eval \"$(hermes completion bash)\"\n\
         \n\
         _hermes_profiles() {{\n\
             local profiles_dir=\"$HOME/.hermes/profiles\"\n\
             local profiles=\"default\"\n\
             if [ -d \"$profiles_dir\" ]; then\n\
                 profiles=\"$profiles $(ls \"$profiles_dir\" 2>/dev/null)\"\n\
             fi\n\
             echo \"$profiles\"\n\
         }}\n\
         \n\
         _hermes_completion() {{\n\
             local cur prev cmd subcmd\n\
             COMPREPLY=()\n\
             cur=\"${{COMP_WORDS[COMP_CWORD]}}\"\n\
             prev=\"${{COMP_WORDS[COMP_CWORD-1]}}\"\n\
             cmd=\"${{COMP_WORDS[1]}}\"\n\
             subcmd=\"${{COMP_WORDS[2]}}\"\n\
         \n\
             if [[ \"$prev\" == \"-p\" || \"$prev\" == \"--profile\" ]]; then\n\
                 COMPREPLY=($(compgen -W \"$(_hermes_profiles)\" -- \"$cur\"))\n\
                 return\n\
             fi\n\
         \n\
             if [[ \"$cmd\" == \"profile\" && ( \"$prev\" == \"use\" || \"$prev\" == \"path\" ) ]]; then\n\
                 COMPREPLY=($(compgen -W \"$(_hermes_profiles)\" -- \"$cur\"))\n\
                 return\n\
             fi\n\
         \n\
             if [[ $COMP_CWORD -eq 1 ]]; then\n\
                 COMPREPLY=($(compgen -W \"{top_cmds_str} {top_flags}\" -- \"$cur\"))\n\
                 return\n\
             fi\n\
         \n\
             case \"$cmd\" in\n\
         {cases}\n\
             esac\n\
         }}\n\
         \n\
         complete -F _hermes_completion hermes",
        cases = cases.join("\n")
    )
}

fn build_bash_case(info: &CommandInfo, depth: usize) -> String {
    let indent = "        ".repeat(depth);
    let inner = format!("{indent}    ");
    let flags = info.flags.join(" ");
    if info.subcommands.is_empty() {
        return format!(
            "{indent}{})\n\
             {inner}COMPREPLY=($(compgen -W \"{flags}\" -- \"$cur\"))\n\
             {inner}return\n\
             {inner};;",
            info.case_pattern(),
        );
    }
    let subcommands = info.all_subcommand_names();
    let subcommand_str = subcommands.join(" ");
    let nested = info
        .subcommands
        .iter()
        .map(|sub_info| build_bash_case(sub_info, depth + 1))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{indent}{})\n\
         {inner}if [[ $COMP_CWORD -eq {} ]]; then\n\
         {inner}    COMPREPLY=($(compgen -W \"{subcommand_str} {flags}\" -- \"$cur\"))\n\
         {inner}    return\n\
         {inner}fi\n\
         {inner}case \"{}\" in\n\
         {nested}\n\
         {inner}esac\n\
         {inner}COMPREPLY=($(compgen -W \"{flags}\" -- \"$cur\"))\n\
         {inner}return\n\
         {inner};;",
        info.case_pattern(),
        depth + 1,
        bash_word_expr(depth + 1),
        nested = nested
    )
}

fn generate_zsh(tree: &CommandInfo) -> String {
    let mut top_cmds = Vec::new();
    let mut sub_cases = Vec::new();
    for info in &tree.subcommands {
        top_cmds.extend(zsh_entries(std::slice::from_ref(info), 4));
        if info.canonical_name() != "profile" && !info.subcommands.is_empty() {
            sub_cases.push(build_zsh_case(info, 1));
        }
    }
    format!(
        "#compdef hermes\n\
         # Hermes Agent zsh completion\n\
         # Add to ~/.zshrc:\n\
         #   eval \"$(hermes completion zsh)\"\n\
         \n\
         _hermes_profiles() {{\n\
             local -a profiles\n\
             profiles=(default)\n\
             if [[ -d \"$HOME/.hermes/profiles\" ]]; then\n\
                 profiles+=(\"${{(@f)$(ls $HOME/.hermes/profiles 2>/dev/null)}}\")\n\
             fi\n\
             _describe 'profile' profiles\n\
         }}\n\
         \n\
         _hermes() {{\n\
             local context state line\n\
             typeset -A opt_args\n\
         \n\
             _arguments -C \\\n\
                 '(-h --help){{-h,--help}}[Show help and exit]' \\\n\
                 '(-V --version){{-V,--version}}[Show version and exit]' \\\n\
                 '(-p --profile){{-p,--profile}}[Profile name]:profile:_hermes_profiles' \\\n\
                 '1:command:->commands' \\\n\
                 '*::arg:->args'\n\
         \n\
             case $state in\n\
                 commands)\n\
                     local -a subcmds\n\
                     subcmds=(\n\
         {top_cmds}\n\
                     )\n\
                     _describe 'hermes command' subcmds\n\
                     ;;\n\
                 args)\n\
                     case ${{line[1]}} in\n\
                         profile)\n\
                             case ${{line[2]}} in\n\
                                 use|path)\n\
                                     _hermes_profiles\n\
                                     ;;\n\
                                 *)\n\
                                     local -a profile_cmds\n\
                                     profile_cmds=(\n\
                                         'current:Show active profile'\n\
                                         'path:Print profile path'\n\
                                         'create:Create a profile'\n\
                                         'use:Switch active profile'\n\
                                     )\n\
                                     _describe 'profile command' profile_cmds\n\
                                     ;;\n\
                             esac\n\
                             ;;\n\
         {sub_cases}\n\
                     esac\n\
                     ;;\n\
             esac\n\
         }}\n\
         \n\
         _hermes \"$@\"",
        top_cmds = top_cmds.join("\n"),
        sub_cases = sub_cases.join("\n")
    )
}

fn generate_fish(tree: &CommandInfo) -> String {
    let top_cmds = tree
        .subcommands
        .iter()
        .flat_map(|info| info.names.iter().cloned())
        .collect::<Vec<_>>();
    let top_cmds_str = top_cmds.join(" ");
    let mut lines = vec![
        "# Hermes Agent fish completion".to_string(),
        "# Add to your config:".to_string(),
        "#   hermes completion fish | source".to_string(),
        "".to_string(),
        "function __hermes_profiles".to_string(),
        "    echo default".to_string(),
        "    if test -d $HOME/.hermes/profiles".to_string(),
        "        ls $HOME/.hermes/profiles 2>/dev/null".to_string(),
        "    end".to_string(),
        "end".to_string(),
        "".to_string(),
        "complete -c hermes -f".to_string(),
        "complete -c hermes -f -s p -l profile -d 'Profile name' -xa '(__hermes_profiles)'"
            .to_string(),
        "".to_string(),
        "# Top-level subcommands".to_string(),
    ];
    for info in &tree.subcommands {
        for (index, name) in info.names.iter().enumerate() {
            let description = if index == 0 {
                info.help.clone()
            } else {
                format!("Alias for {}", info.canonical_name())
            };
            lines.push(format!(
                "complete -c hermes -f -n 'not __fish_seen_subcommand_from {top_cmds_str}' -a {name} -d '{}'",
                description
            ));
        }
    }
    lines.push("".to_string());
    lines.push("# Subcommand completions".to_string());
    for info in &tree.subcommands {
        push_fish_subcommand_lines(&mut lines, info, &[]);
    }
    lines.push(
        "complete -c hermes -f -n '__fish_seen_subcommand_from use path; and __fish_seen_subcommand_from profile' -a '(__hermes_profiles)' -d 'Profile name'"
            .to_string(),
    );
    lines.join("\n")
}

fn bash_word_expr(depth: usize) -> String {
    match depth {
        1 => "$cmd".to_string(),
        2 => "$subcmd".to_string(),
        _ => format!("${{COMP_WORDS[{depth}]}}"),
    }
}

fn build_zsh_case(info: &CommandInfo, depth: usize) -> String {
    let indent = zsh_indent(depth);
    let inner = format!("{indent}    ");
    let nested_cases = info
        .subcommands
        .iter()
        .filter(|sub_info| !sub_info.subcommands.is_empty())
        .map(|sub_info| build_zsh_case(sub_info, depth + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let entries = zsh_entries(&info.subcommands, depth + 3).join("\n");
    let safe = zsh_safe_name(info.canonical_name(), depth);
    if nested_cases.is_empty() {
        return format!(
            "{indent}{})\n\
             {inner}local -a {safe}_cmds\n\
             {inner}{safe}_cmds=(\n\
             {entries}\n\
             {inner})\n\
             {inner}_describe '{} command' {safe}_cmds\n\
             {inner};;",
            zsh_case_pattern(&info.names),
            info.canonical_name(),
        );
    }
    format!(
        "{indent}{})\n\
         {inner}case ${{line[{}}} in\n\
         {nested_cases}\n\
         {inner}    *)\n\
         {inner}        local -a {safe}_cmds\n\
         {inner}        {safe}_cmds=(\n\
         {entries}\n\
         {inner}        )\n\
         {inner}        _describe '{} command' {safe}_cmds\n\
         {inner}        ;;\n\
         {inner}esac\n\
         {inner};;",
        zsh_case_pattern(&info.names),
        depth + 1,
        info.canonical_name(),
    )
}

fn zsh_entries(commands: &[CommandInfo], depth: usize) -> Vec<String> {
    let indent = zsh_indent(depth);
    let mut entries = Vec::new();
    for info in commands {
        entries.push(format!("{indent}'{}:{}'", info.canonical_name(), info.help));
        for alias in info.names.iter().skip(1) {
            entries.push(format!(
                "{indent}'{}:Alias for {}'",
                alias,
                info.canonical_name()
            ));
        }
    }
    entries
}

fn zsh_indent(depth: usize) -> String {
    format!("                {}", "    ".repeat(depth.saturating_sub(1)))
}

fn zsh_safe_name(name: &str, depth: usize) -> String {
    format!("{}_{}_cmds", name.replace('-', "_"), depth)
}

fn push_fish_subcommand_lines(
    lines: &mut Vec<String>,
    info: &CommandInfo,
    ancestor_conditions: &[String],
) {
    if info.subcommands.is_empty() {
        return;
    }

    lines.push(format!("# {}", info.canonical_name()));
    let mut condition_parts = ancestor_conditions.to_vec();
    condition_parts.push(format!(
        "__fish_seen_subcommand_from {}",
        info.names.join(" ")
    ));
    let child_names = info.all_subcommand_names().join(" ");
    if !child_names.is_empty() {
        condition_parts.push(format!("not __fish_seen_subcommand_from {child_names}"));
    }
    let condition = condition_parts.join("; and ");
    for sub_info in &info.subcommands {
        for (index, sub_name) in sub_info.names.iter().enumerate() {
            let description = if index == 0 {
                sub_info.help.clone()
            } else {
                format!("Alias for {}", sub_info.canonical_name())
            };
            lines.push(format!(
                "complete -c hermes -f -n '{condition}' -a {sub_name} -d '{}'",
                description
            ));
        }
    }

    let mut next_conditions = ancestor_conditions.to_vec();
    next_conditions.push(format!(
        "__fish_seen_subcommand_from {}",
        info.names.join(" ")
    ));
    for sub_info in &info.subcommands {
        push_fish_subcommand_lines(lines, sub_info, &next_conditions);
    }
}

fn zsh_case_pattern(names: &[String]) -> String {
    if names.len() <= 1 {
        names.first().cloned().unwrap_or_default()
    } else {
        format!("({})", names.join("|"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bash_completion_lists_top_level_commands() {
        let tree = walk_root_command(&mut Cli::command());
        let script = generate_bash(&tree);
        assert!(script.contains("debug"));
        assert!(script.contains("completion"));
        assert!(script.contains("_hermes_profiles"));
    }

    #[test]
    fn zsh_completion_includes_profile_helper() {
        let tree = walk_root_command(&mut Cli::command());
        let script = generate_zsh(&tree);
        assert!(script.contains("_hermes_profiles"));
        assert!(script.contains("profile command"));
    }

    #[test]
    fn fish_completion_contains_subcommand_entries() {
        let tree = walk_root_command(&mut Cli::command());
        let script = generate_fish(&tree);
        assert!(script.contains("complete -c hermes"));
        assert!(script.contains("sessions"));
    }

    #[test]
    fn completions_include_sessions_browse_and_fallback_subcommands() {
        let tree = walk_root_command(&mut Cli::command());
        let bash = generate_bash(&tree);
        let fish = generate_fish(&tree);
        assert!(bash.contains("browse"));
        assert!(bash.contains("resume"));
        assert!(bash.contains("fallback"));
        assert!(fish.contains("__fish_seen_subcommand_from sessions"));
        assert!(fish.contains("-a resume -d ''"));
        assert!(fish.contains("__fish_seen_subcommand_from fallback"));
        assert!(fish.contains(" -a browse "));
        assert!(fish.contains(" -a add "));
        assert!(fish.contains(" -a remove "));
    }

    #[test]
    fn completions_include_alias_subcommands_and_flag_aliases() {
        let tree = walk_root_command(&mut Cli::command());
        let bash = generate_bash(&tree);
        let fish = generate_fish(&tree);
        let zsh = generate_zsh(&tree);

        assert!(bash.contains("create|add"));
        assert!(bash.contains("remove|"));
        assert!(bash.contains("|delete"));
        assert!(bash.contains("|rm"));
        assert!(bash.contains("--synchronous"));

        assert!(fish.contains("__fish_seen_subcommand_from fallback"));
        assert!(fish.contains(" -a ls "));
        assert!(fish.contains(" -a rm "));

        assert!(zsh.contains("'add:Alias for create'"));
        assert!(zsh.contains("'rm:Alias for remove'"));
    }

    #[test]
    fn completions_include_top_level_aliases_and_nested_subcommands() {
        let tree = walk_root_command(&mut Cli::command());
        let bash = generate_bash(&tree);
        let fish = generate_fish(&tree);
        let zsh = generate_zsh(&tree);

        assert!(bash.contains("${COMP_WORDS[3]}"));
        assert!(bash.contains("export import"));

        assert!(fish.contains("-a provider -d 'Alias for model'"));
        assert!(fish.contains("-a snap -d 'Alias for snapshot'"));
        assert!(fish.contains(
            "__fish_seen_subcommand_from skills; and __fish_seen_subcommand_from snapshot"
        ));
        assert!(fish.contains(" -a export "));
        assert!(fish.contains(" -a import "));

        assert!(zsh.contains("'provider:Alias for model'"));
        assert!(zsh.contains("'snap:Alias for snapshot'"));
        assert!(zsh.contains("_describe 'snapshot command'"));
        assert!(zsh.contains("'export:"));
    }

    #[test]
    fn completions_include_kanban_compat_subcommands() {
        let tree = walk_root_command(&mut Cli::command());
        let bash = generate_bash(&tree);
        let fish = generate_fish(&tree);
        let zsh = generate_zsh(&tree);

        assert!(bash.contains("boards"));
        assert!(bash.contains("assign"));
        assert!(bash.contains("switch"));

        assert!(fish.contains("-a platforms -d 'Alias for gateway'"));
        assert!(fish.contains("__fish_seen_subcommand_from kanban"));
        assert!(fish.contains(" -a boards "));
        assert!(fish.contains(" -a switch "));

        assert!(zsh.contains("'platforms:Alias for gateway'"));
        assert!(zsh.contains("_describe 'kanban command'"));
        assert!(zsh.contains("'boards:"));
        assert!(zsh.contains("'switch:"));
    }

    #[test]
    fn completions_include_slash_compat_commands_and_subcommands() {
        let tree = walk_root_command(&mut Cli::command());
        let bash = generate_bash(&tree);
        let fish = generate_fish(&tree);
        let zsh = generate_zsh(&tree);

        assert!(bash.contains("reasoning"));
        assert!(bash.contains("voice"));
        assert!(bash.contains("gquota"));
        assert!(bash.contains("goal"));
        assert!(bash.contains("clear"));
        assert!(bash.contains("compress"));
        assert!(bash.contains("image"));
        assert!(bash.contains("new"));
        assert!(bash.contains("paste"));
        assert!(bash.contains("queue"));
        assert!(bash.contains("redraw"));
        assert!(bash.contains("rollback"));
        assert!(bash.contains("restart"));
        assert!(bash.contains("retry"));
        assert!(bash.contains("steer"));
        assert!(bash.contains("status"));
        assert!(bash.contains("title"));
        assert!(bash.contains("undo"));
        assert!(bash.contains("background"));
        assert!(bash.contains("approve"));
        assert!(bash.contains("deny"));
        assert!(bash.contains("set-home"));
        assert!(bash.contains("show"));
        assert!(bash.contains("hide"));
        assert!(bash.contains("connect"));
        assert!(bash.contains("disconnect"));
        assert!(bash.contains("status"));
        assert!(bash.contains("topic"));

        assert!(fish.contains("-a bg -d 'Alias for background'"));
        assert!(fish.contains("-a tasks -d 'Alias for agents'"));
        assert!(fish.contains("-a fork -d 'Alias for branch'"));
        assert!(fish.contains("-a q -d 'Alias for queue'"));
        assert!(fish.contains("-a reset -d 'Alias for new'"));
        assert!(fish.contains("-a set-home -d 'Alias for sethome'"));
        assert!(fish.contains("-a gquota -d ''"));
        assert!(fish.contains("-a reload_mcp -d 'Alias for reload-mcp'"));
        assert!(fish.contains("__fish_seen_subcommand_from approve"));
        assert!(fish.contains(" -a always "));
        assert!(fish.contains("__fish_seen_subcommand_from deny"));
        assert!(fish.contains(" -a all "));
        assert!(fish.contains("__fish_seen_subcommand_from reasoning"));
        assert!(fish.contains(" -a xhigh "));
        assert!(fish.contains("__fish_seen_subcommand_from goal"));
        assert!(fish.contains(" -a pause "));
        assert!(fish.contains("__fish_seen_subcommand_from rollback"));
        assert!(fish.contains(" -a diff "));
        assert!(fish.contains("__fish_seen_subcommand_from browser"));
        assert!(fish.contains(" -a disconnect "));
        assert!(fish.contains("__fish_seen_subcommand_from topic"));
        assert!(fish.contains(" -a off "));

        assert!(zsh.contains("'bg:Alias for background'"));
        assert!(zsh.contains("'tasks:Alias for agents'"));
        assert!(zsh.contains("'fork:Alias for branch'"));
        assert!(zsh.contains("'q:Alias for queue'"));
        assert!(zsh.contains("'reset:Alias for new'"));
        assert!(zsh.contains("'set-home:Alias for sethome'"));
        assert!(zsh.contains("'gquota:"));
        assert!(zsh.contains("'reload_mcp:Alias for reload-mcp'"));
        assert!(zsh.contains("_describe 'approve command'"));
        assert!(zsh.contains("'always:"));
        assert!(zsh.contains("_describe 'deny command'"));
        assert!(zsh.contains("_describe 'reasoning command'"));
        assert!(zsh.contains("'xhigh:"));
        assert!(zsh.contains("_describe 'goal command'"));
        assert!(zsh.contains("'pause:"));
        assert!(zsh.contains("_describe 'rollback command'"));
        assert!(zsh.contains("'diff:"));
        assert!(zsh.contains("_describe 'browser command'"));
        assert!(zsh.contains("_describe 'topic command'"));
        assert!(zsh.contains("'off:"));
    }
}
