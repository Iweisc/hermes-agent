use std::error::Error;

use clap::{Args, ValueEnum};

use crate::python_bridge::launch_python_main_command;

#[derive(Args, Debug, Clone)]
pub struct SetupArgs {
    #[arg(value_enum)]
    pub section: Option<SetupSection>,
    #[arg(long = "non-interactive", default_value_t = false)]
    pub non_interactive: bool,
    #[arg(long, default_value_t = false)]
    pub reset: bool,
    #[arg(long, default_value_t = false)]
    pub reconfigure: bool,
    #[arg(long, default_value_t = false)]
    pub quick: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum SetupSection {
    #[value(name = "model")]
    Model,
    #[value(name = "tts")]
    Tts,
    #[value(name = "terminal")]
    Terminal,
    #[value(name = "gateway")]
    Gateway,
    #[value(name = "tools")]
    Tools,
    #[value(name = "agent")]
    Agent,
}

pub fn print_setup(args: SetupArgs) -> Result<(), Box<dyn Error>> {
    let argv = bridge_setup_args(&args);
    launch_python_main_command("setup", &argv, Some("HERMES_SETUP_PYTHON"), &[])
}

fn bridge_setup_args(args: &SetupArgs) -> Vec<String> {
    let mut argv = Vec::new();
    if let Some(section) = args.section {
        argv.push(section.as_str().to_string());
    }
    if args.non_interactive {
        argv.push(String::from("--non-interactive"));
    }
    if args.reset {
        argv.push(String::from("--reset"));
    }
    if args.reconfigure {
        argv.push(String::from("--reconfigure"));
    }
    if args.quick {
        argv.push(String::from("--quick"));
    }
    argv
}

impl SetupSection {
    fn as_str(self) -> &'static str {
        match self {
            SetupSection::Model => "model",
            SetupSection::Tts => "tts",
            SetupSection::Terminal => "terminal",
            SetupSection::Gateway => "gateway",
            SetupSection::Tools => "tools",
            SetupSection::Agent => "agent",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct SetupHarness {
        #[command(flatten)]
        args: SetupArgs,
    }

    #[test]
    fn setup_args_preserve_section_and_flags() {
        let parsed = SetupHarness::try_parse_from([
            "setup",
            "gateway",
            "--non-interactive",
            "--reset",
            "--reconfigure",
            "--quick",
        ])
        .unwrap();
        assert_eq!(parsed.args.section, Some(SetupSection::Gateway));
        assert!(parsed.args.non_interactive);
        assert!(parsed.args.reset);
        assert!(parsed.args.reconfigure);
        assert!(parsed.args.quick);
    }

    #[test]
    fn bridge_setup_args_preserve_python_order() {
        let args = SetupArgs {
            section: Some(SetupSection::Model),
            non_interactive: true,
            reset: false,
            reconfigure: true,
            quick: false,
        };
        assert_eq!(
            bridge_setup_args(&args),
            vec![
                String::from("model"),
                String::from("--non-interactive"),
                String::from("--reconfigure"),
            ]
        );
    }
}
