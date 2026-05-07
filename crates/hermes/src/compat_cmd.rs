use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct CompatArgs {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub args: Vec<String>,
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
