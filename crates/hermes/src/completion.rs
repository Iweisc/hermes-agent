use std::collections::BTreeMap;
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
    help: String,
    flags: Vec<String>,
    subcommands: BTreeMap<String, CommandInfo>,
}

pub fn print_completion(args: CompletionArgs) -> Result<(), Box<dyn Error>> {
    let tree = walk_command(&mut Cli::command());
    let script = match args.shell {
        CompletionShell::Bash => generate_bash(&tree),
        CompletionShell::Zsh => generate_zsh(&tree),
        CompletionShell::Fish => generate_fish(&tree),
    };
    println!("{script}");
    Ok(())
}

fn walk_command(command: &mut clap::Command) -> CommandInfo {
    command.build();
    let mut info = CommandInfo {
        help: clean(
            command
                .get_about()
                .map(|value| value.to_string())
                .unwrap_or_default(),
        ),
        flags: command_flags(command),
        subcommands: BTreeMap::new(),
    };
    for subcommand in command.get_subcommands_mut() {
        if subcommand.is_hide_set() {
            continue;
        }
        info.subcommands
            .insert(subcommand.get_name().to_string(), walk_command(subcommand));
    }
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
            if let Some(long) = arg.get_long() {
                values.push(format!("--{long}"));
            }
            values
        })
        .collect::<Vec<_>>();
    flags.sort();
    flags.dedup();
    flags
}

fn clean(text: String) -> String {
    text.replace(['\'', '"', '\\'], "")
        .chars()
        .take(60)
        .collect()
}

fn generate_bash(tree: &CommandInfo) -> String {
    let top_cmds = tree.subcommands.keys().cloned().collect::<Vec<_>>();
    let top_cmds_str = top_cmds.join(" ");
    let top_flags = tree.flags.join(" ");
    let mut cases = Vec::new();
    for (name, info) in &tree.subcommands {
        cases.push(build_bash_case(name, info));
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

fn build_bash_case(name: &str, info: &CommandInfo) -> String {
    let flags = info.flags.join(" ");
    if info.subcommands.is_empty() {
        return format!(
            "        {name})\n\
             \t\tCOMPREPLY=($(compgen -W \"{flags}\" -- \"$cur\"))\n\
             \t\treturn\n\
             \t\t;;"
        );
    }
    let subcommands = info.subcommands.keys().cloned().collect::<Vec<_>>();
    let subcommand_str = subcommands.join(" ");
    let mut nested = Vec::new();
    for (sub_name, sub_info) in &info.subcommands {
        let sub_flags = sub_info.flags.join(" ");
        nested.push(format!(
            "                {sub_name})\n\
             \t\t\t\tCOMPREPLY=($(compgen -W \"{sub_flags}\" -- \"$cur\"))\n\
             \t\t\t\treturn\n\
             \t\t\t\t;;"
        ));
    }
    format!(
        "        {name})\n\
         \t\tif [[ $COMP_CWORD -eq 2 ]]; then\n\
         \t\t\tCOMPREPLY=($(compgen -W \"{subcommand_str} {flags}\" -- \"$cur\"))\n\
         \t\t\treturn\n\
         \t\tfi\n\
         \t\tcase \"$subcmd\" in\n\
         {nested}\n\
         \t\tesac\n\
         \t\t;;",
        nested = nested.join("\n")
    )
}

fn generate_zsh(tree: &CommandInfo) -> String {
    let mut top_cmds = Vec::new();
    let mut sub_cases = Vec::new();
    for (name, info) in &tree.subcommands {
        top_cmds.push(format!("                '{}:{}'", name, info.help));
        if !info.subcommands.is_empty() {
            let mut nested = Vec::new();
            for (sub_name, sub_info) in &info.subcommands {
                nested.push(format!(
                    "                        '{}:{}'",
                    sub_name, sub_info.help
                ));
            }
            sub_cases.push(format!(
                "                {name})\n\
                 \t\t\tlocal -a {safe}_cmds\n\
                 \t\t\t{safe}_cmds=(\n\
                 {nested}\n\
                 \t\t\t)\n\
                 \t\t\t_describe '{name} command' {safe}_cmds\n\
                 \t\t\t;;",
                safe = name.replace('-', "_"),
                nested = nested.join("\n")
            ));
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
    let top_cmds = tree.subcommands.keys().cloned().collect::<Vec<_>>();
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
    for (name, info) in &tree.subcommands {
        lines.push(format!(
            "complete -c hermes -f -n 'not __fish_seen_subcommand_from {top_cmds_str}' -a {name} -d '{}'",
            info.help
        ));
    }
    lines.push("".to_string());
    lines.push("# Subcommand completions".to_string());
    for (name, info) in &tree.subcommands {
        if info.subcommands.is_empty() {
            continue;
        }
        lines.push(format!("# {name}"));
        for (sub_name, sub_info) in &info.subcommands {
            lines.push(format!(
                "complete -c hermes -f -n '__fish_seen_subcommand_from {name}' -a {sub_name} -d '{}'",
                sub_info.help
            ));
        }
    }
    lines.push(
        "complete -c hermes -f -n '__fish_seen_subcommand_from use path; and __fish_seen_subcommand_from profile' -a '(__hermes_profiles)' -d 'Profile name'"
            .to_string(),
    );
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bash_completion_lists_top_level_commands() {
        let tree = walk_command(&mut Cli::command());
        let script = generate_bash(&tree);
        assert!(script.contains("debug"));
        assert!(script.contains("completion"));
        assert!(script.contains("_hermes_profiles"));
    }

    #[test]
    fn zsh_completion_includes_profile_helper() {
        let tree = walk_command(&mut Cli::command());
        let script = generate_zsh(&tree);
        assert!(script.contains("_hermes_profiles"));
        assert!(script.contains("profile command"));
    }

    #[test]
    fn fish_completion_contains_subcommand_entries() {
        let tree = walk_command(&mut Cli::command());
        let script = generate_fish(&tree);
        assert!(script.contains("complete -c hermes"));
        assert!(script.contains("sessions"));
    }
}
