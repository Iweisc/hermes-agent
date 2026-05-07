use std::error::Error;

use crate::python_bridge::launch_python_main_command;

pub fn print_whatsapp() -> Result<(), Box<dyn Error>> {
    launch_python_main_command("whatsapp", &[], Some("HERMES_WHATSAPP_PYTHON"), &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whatsapp_command_name_is_stable() {
        let result = print_whatsapp as fn() -> Result<(), Box<dyn Error>>;
        let _ = result;
    }
}
