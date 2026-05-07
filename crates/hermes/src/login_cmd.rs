use std::error::Error;

use clap::Args;

#[derive(Args, Debug)]
pub struct LoginArgs {
    #[arg(long, value_parser = ["nous", "openai-codex"])]
    pub provider: Option<String>,
    #[arg(long = "portal-url")]
    pub portal_url: Option<String>,
    #[arg(long = "inference-url")]
    pub inference_url: Option<String>,
    #[arg(long = "client-id")]
    pub client_id: Option<String>,
    #[arg(long)]
    pub scope: Option<String>,
    #[arg(long = "no-browser")]
    pub no_browser: bool,
    #[arg(long, default_value_t = 15.0)]
    pub timeout: f64,
    #[arg(long = "ca-bundle")]
    pub ca_bundle: Option<String>,
    #[arg(long)]
    pub insecure: bool,
}

pub fn print_login(_args: LoginArgs) -> Result<(), Box<dyn Error>> {
    println!("{}", login_removed_message());
    Ok(())
}

fn login_removed_message() -> &'static str {
    "The 'hermes login' command has been removed.\nUse 'hermes auth' to manage credentials,\n'hermes model' to select a provider, or 'hermes setup' for full setup."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn login_message_matches_python_guidance() {
        let message = login_removed_message();
        assert!(message.contains("hermes login"));
        assert!(message.contains("hermes auth"));
        assert!(message.contains("hermes model"));
        assert!(message.contains("hermes setup"));
    }
}
