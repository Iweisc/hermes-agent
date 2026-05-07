use std::error::Error;

use clap::Args;

use crate::python_bridge::launch_python_main_command;

#[derive(Args, Debug, Clone)]
pub struct CompatArgs {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub args: Vec<String>,
}

pub fn print_setup(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("setup", args)
}

pub fn print_gateway(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("gateway", args)
}

pub fn print_skills(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("skills", args)
}

pub fn print_snapshot(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("snapshot", args)
}

pub fn print_plugins(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("plugins", args)
}

pub fn print_curator(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("curator", args)
}

pub fn print_memory(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("memory", args)
}

pub fn print_mcp(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("mcp", args)
}

pub fn print_insights(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("insights", args)
}

pub fn print_claw(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("claw", args)
}

pub fn print_migrate(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("migrate", args)
}

pub fn print_cleanup(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("cleanup", args)
}

pub fn print_acp(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("acp", args)
}

pub fn print_whatsapp(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("whatsapp", args)
}

pub fn print_update(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("update", args)
}

pub fn print_uninstall(args: CompatArgs) -> Result<(), Box<dyn Error>> {
    print_passthrough("uninstall", args)
}

fn print_passthrough(command_name: &str, args: CompatArgs) -> Result<(), Box<dyn Error>> {
    launch_python_main_command(command_name, &args.args, Some("HERMES_COMPAT_PYTHON"), &[])
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct CompatHarness {
        #[command(flatten)]
        args: CompatArgs,
    }

    #[test]
    fn compat_args_preserve_hyphenated_values() {
        let parsed =
            CompatHarness::try_parse_from(["gateway", "status", "--deep", "--full", "--system"])
                .unwrap();
        assert_eq!(
            parsed.args.args,
            vec![
                "status".to_string(),
                "--deep".to_string(),
                "--full".to_string(),
                "--system".to_string()
            ]
        );
    }
}
