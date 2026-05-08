use std::cell::RefCell;
use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Args, ValueEnum};
use hermes_core::{
    HermesContext, get_auth_status_summary, list_provider_profiles, normalize_model_for_provider,
    resolve_provider_api_mode,
};
use serde_json::Value as JsonValue;
use serde_yaml::{Mapping, Value};

use crate::auth_cmd::{self, AuthAddArgs};
use crate::config_cmd::{read_raw_yaml_mapping, save_env_value, write_yaml_mapping};
use crate::gateway_cmd;
use crate::python_bridge::{project_root, resolve_repo_python};
use crate::tools_cmd;

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

pub fn print_setup(context: &HermesContext, args: SetupArgs) -> Result<(), Box<dyn Error>> {
    if should_use_native_model_setup(&args)
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        let mut ui = TerminalUi;
        return run_native_model_setup(context, &mut ui);
    }
    if should_use_native_agent_setup(&args)
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        let mut ui = TerminalUi;
        return run_native_agent_setup(context, &mut ui);
    }
    if should_use_native_tts_setup(&args) && io::stdin().is_terminal() && io::stdout().is_terminal()
    {
        let mut ui = TerminalUi;
        return run_native_tts_setup(context, &mut ui);
    }
    if should_use_native_terminal_setup(&args)
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        let mut ui = TerminalUi;
        return run_native_terminal_setup(context, &mut ui);
    }
    if should_use_native_gateway_setup(&args)
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        return run_native_gateway_setup(context);
    }
    if should_use_native_tools_setup(&args)
        && io::stdin().is_terminal()
        && io::stdout().is_terminal()
    {
        return run_native_tools_setup(context);
    }
    if let Some(section) = direct_python_setup_section(&args) {
        return print_direct_setup_python(section);
    }
    print_setup_python(args)
}

fn print_setup_python(args: SetupArgs) -> Result<(), Box<dyn Error>> {
    print_setup_python_bootstrap(SETUP_BOOTSTRAP, args)
}

fn print_setup_python_bootstrap(bootstrap: &str, args: SetupArgs) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_SETUP_PYTHON"))
        .ok_or("could not find a Python interpreter for setup")?;

    let mut command = Command::new(&python);
    command
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env(
            "HERMES_SETUP_NON_INTERACTIVE",
            if args.non_interactive { "1" } else { "0" },
        )
        .env("HERMES_SETUP_RESET", if args.reset { "1" } else { "0" })
        .env(
            "HERMES_SETUP_RECONFIGURE",
            if args.reconfigure { "1" } else { "0" },
        )
        .env("HERMES_SETUP_QUICK", if args.quick { "1" } else { "0" });
    if let Some(section) = args.section {
        command.env("HERMES_SETUP_SECTION", section.as_str());
    }
    command.arg("-c").arg(bootstrap);

    let status = command.status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message("setup", status).into())
}

fn print_direct_setup_python(section: SetupSection) -> Result<(), Box<dyn Error>> {
    let bootstrap = match section {
        SetupSection::Model => SETUP_MODEL_BOOTSTRAP,
        SetupSection::Gateway
        | SetupSection::Tools
        | SetupSection::Tts
        | SetupSection::Terminal
        | SetupSection::Agent => {
            return Err(format!("unsupported direct setup section: {}", section.as_str()).into());
        }
    };
    print_setup_python_bootstrap(
        bootstrap,
        SetupArgs {
            section: Some(section),
            non_interactive: false,
            reset: false,
            reconfigure: false,
            quick: false,
        },
    )
}

fn should_use_native_agent_setup(args: &SetupArgs) -> bool {
    matches!(args.section, Some(SetupSection::Agent))
        && !args.non_interactive
        && !args.reset
        && !args.reconfigure
        && !args.quick
}

fn should_use_native_model_setup(args: &SetupArgs) -> bool {
    matches!(args.section, Some(SetupSection::Model))
        && !args.non_interactive
        && !args.reset
        && !args.reconfigure
        && !args.quick
}

fn should_use_native_tts_setup(args: &SetupArgs) -> bool {
    matches!(args.section, Some(SetupSection::Tts))
        && !args.non_interactive
        && !args.reset
        && !args.reconfigure
        && !args.quick
}

fn should_use_native_terminal_setup(args: &SetupArgs) -> bool {
    matches!(args.section, Some(SetupSection::Terminal))
        && !args.non_interactive
        && !args.reset
        && !args.reconfigure
        && !args.quick
}

fn should_use_native_gateway_setup(args: &SetupArgs) -> bool {
    matches!(args.section, Some(SetupSection::Gateway))
        && !args.non_interactive
        && !args.reset
        && !args.reconfigure
        && !args.quick
}

fn should_use_native_tools_setup(args: &SetupArgs) -> bool {
    matches!(args.section, Some(SetupSection::Tools))
        && !args.non_interactive
        && !args.reset
        && !args.reconfigure
        && !args.quick
}

fn direct_python_setup_section(args: &SetupArgs) -> Option<SetupSection> {
    if args.non_interactive || args.reset || args.reconfigure || args.quick {
        return None;
    }
    match args.section {
        Some(SetupSection::Model) => args.section,
        _ => None,
    }
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

const SETUP_BOOTSTRAP: &str = concat!(
    "import argparse\n",
    "import os\n",
    "from hermes_cli.setup import run_setup_wizard\n",
    "args = argparse.Namespace(\n",
    "    section=(os.environ.get('HERMES_SETUP_SECTION') or None),\n",
    "    non_interactive=(os.environ.get('HERMES_SETUP_NON_INTERACTIVE') == '1'),\n",
    "    reset=(os.environ.get('HERMES_SETUP_RESET') == '1'),\n",
    "    reconfigure=(os.environ.get('HERMES_SETUP_RECONFIGURE') == '1'),\n",
    "    quick=(os.environ.get('HERMES_SETUP_QUICK') == '1'),\n",
    ")\n",
    "run_setup_wizard(args)\n",
);

const SETUP_MODEL_BOOTSTRAP: &str = concat!(
    "from hermes_cli.config import ensure_hermes_home, is_managed, managed_error, load_config, save_config\n",
    "from hermes_cli.setup import is_interactive_stdin, print_noninteractive_setup_guidance\n",
    "if is_managed():\n",
    "    managed_error('run setup wizard')\n",
    "else:\n",
    "    ensure_hermes_home()\n",
    "    if not is_interactive_stdin():\n",
    "        print_noninteractive_setup_guidance('Running in a non-interactive environment (no TTY detected).')\n",
    "    else:\n",
    "        config = load_config()\n",
    "        from hermes_cli.setup import setup_model_provider\n",
    "        setup_model_provider(config)\n",
    "        save_config(config)\n",
);

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

fn run_native_tools_setup(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    if let Some(system) = setup_managed_system(context) {
        eprintln!(
            "{}",
            format_setup_managed_message(&system, "run setup wizard")
        );
        return Ok(());
    }
    tools_cmd::run_native_tools_interactive(context)
}

fn run_native_model_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
) -> Result<(), Box<dyn Error>> {
    if let Some(system) = setup_managed_system(context) {
        ui.line(&format_setup_managed_message(&system, "run setup wizard"))?;
        return Ok(());
    }

    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let current_provider =
        get_nested_string(&root, &["model", "provider"]).unwrap_or_else(|| "auto".to_string());
    let current_model = get_nested_string(&root, &["model", "default"]).unwrap_or_default();
    let current_base_url = get_nested_string(&root, &["model", "base_url"]).unwrap_or_default();

    let providers = native_setup_model_providers(context);
    let saved_custom_providers = collect_saved_custom_model_providers(&root);
    let current_label = model_provider_label(&current_provider);

    ui.blank()?;
    ui.line("⚕ Hermes Setup — Inference Provider")?;
    ui.line("Choose a provider and default model.")?;
    ui.line(
        "Some fresh OAuth login, Copilot ACP, and the advanced model picker still use the compatibility flow.",
    )?;
    ui.blank()?;
    ui.line(&format!("Current provider: {current_label}"))?;
    ui.line(&format!(
        "Current model: {}",
        if current_model.trim().is_empty() {
            "(not set)"
        } else {
            current_model.trim()
        }
    ))?;
    ui.blank()?;

    for (index, provider) in providers.iter().enumerate() {
        ui.line(&format!("  {}. {}", index + 1, provider.label))?;
    }
    let saved_offset = providers.len();
    for (index, provider) in saved_custom_providers.iter().enumerate() {
        let saved_model = provider.model.as_deref().unwrap_or("");
        let model_hint = if saved_model.is_empty() {
            String::new()
        } else {
            format!(" — {saved_model}")
        };
        ui.line(&format!(
            "  {}. {} ({}){}",
            saved_offset + index + 1,
            provider.name,
            format_saved_custom_provider_url(&provider.base_url),
            model_hint
        ))?;
    }
    let custom_endpoint_choice = providers.len() + saved_custom_providers.len() + 1;
    let remove_custom_choice =
        (!saved_custom_providers.is_empty()).then_some(custom_endpoint_choice + 1);
    let auxiliary_choice = remove_custom_choice.unwrap_or(custom_endpoint_choice) + 1;
    let compatibility_choice = auxiliary_choice + 1;
    let keep_choice = compatibility_choice + 1;
    ui.line(&format!(
        "  {}. Custom endpoint (enter URL manually)",
        custom_endpoint_choice
    ))?;
    if let Some(remove_custom_choice) = remove_custom_choice {
        ui.line(&format!(
            "  {}. Remove a saved custom provider",
            remove_custom_choice
        ))?;
    }
    ui.line(&format!(
        "  {}. Configure auxiliary models...",
        auxiliary_choice
    ))?;
    ui.line(&format!(
        "  {}. Use compatibility flow (OAuth, advanced picker)",
        compatibility_choice
    ))?;
    ui.line(&format!("  {}. Keep current", keep_choice))?;

    let mut default_choice = providers
        .iter()
        .position(|provider| provider.name == current_provider)
        .map(|index| index + 1)
        .unwrap_or(keep_choice);
    if current_provider == "custom" && !current_base_url.trim().is_empty() {
        if let Some(index) = saved_custom_providers
            .iter()
            .position(|provider| provider.base_url == current_base_url.trim_end_matches('/'))
        {
            default_choice = saved_offset + index + 1;
        } else {
            default_choice = custom_endpoint_choice;
        }
    }
    let selection = prompt_menu_choice(ui, "Select provider: ", keep_choice, default_choice)?;

    if selection == keep_choice {
        ui.line(&format!(
            "Keeping current model provider: {}",
            if current_label.is_empty() {
                "auto"
            } else {
                current_label.as_str()
            }
        ))?;
        return Ok(());
    }

    if selection == compatibility_choice {
        return print_direct_setup_python(SetupSection::Model);
    }

    if selection == custom_endpoint_choice {
        let current_api_key = get_nested_string(&root, &["model", "api_key"])
            .or_else(|| env_value_for_context(context, "OPENAI_API_KEY"))
            .unwrap_or_default();
        return run_native_custom_model_setup(
            context,
            ui,
            &mut root,
            &current_provider,
            &current_model,
            &current_base_url,
            &current_api_key,
        );
    }

    if Some(selection) == remove_custom_choice {
        return remove_saved_custom_model_provider(context, ui, &mut root, &saved_custom_providers);
    }

    if selection == auxiliary_choice {
        return run_native_auxiliary_model_setup(context, ui, &mut root);
    }

    if selection > providers.len() {
        let selected = &saved_custom_providers[selection - providers.len() - 1];
        let current_model_for_provider = if current_provider == "custom"
            && selected.base_url == current_base_url.trim_end_matches('/')
        {
            current_model.clone()
        } else {
            selected.model.clone().unwrap_or_default()
        };
        let model_name = prompt_model_name(ui, "custom", &current_model_for_provider)?;
        apply_saved_custom_model_provider_choice(context, &mut root, selected, &model_name)?;
        ui.line(&format!(
            "Default model set to: {} (via {})",
            normalize_model_for_provider(&model_name, "custom"),
            selected.name
        ))?;
        return Ok(());
    }

    let selected = &providers[selection - 1];

    if selected.auth_type == "api_key" {
        ensure_model_provider_secret(context, ui, selected)?;
    } else {
        ensure_runtime_model_provider_credentials(context, ui, selected)?;
    }
    let selected_base_url = resolved_native_model_provider_base_url(
        context,
        selected,
        &current_provider,
        &current_base_url,
    )?;
    let base_url = if selected.auth_type == "api_key" {
        prompt_model_base_url(ui, selected, &selected_base_url)?
    } else {
        selected_base_url
    };
    let current_model_for_provider = if current_provider == selected.name {
        current_model.clone()
    } else {
        String::new()
    };
    let model_name = prompt_model_name(ui, &selected.name, &current_model_for_provider)?;
    apply_model_provider_choice(context, &mut root, selected, &model_name, &base_url)?;

    ui.line(&format!(
        "Default model set to: {} (via {})",
        normalize_model_for_provider(&model_name, &selected.name),
        selected.label
    ))?;
    Ok(())
}

pub(crate) fn run_native_model_setup_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut ui = StreamUi { input, output };
    run_native_model_setup(context, &mut ui)
}

fn run_native_auxiliary_model_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    root: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    let reset_choice = AUXILIARY_MODEL_TASKS.len() + 1;
    let back_choice = reset_choice + 1;
    loop {
        ui.blank()?;
        ui.line("Auxiliary models — side-task routing")?;
        ui.blank()?;
        ui.line("Override individual side tasks without changing your main chat model.")?;
        ui.line(
            "Use auto to inherit the main provider, or pin a task to a specific provider/model.",
        )?;
        ui.blank()?;

        for (index, task) in AUXILIARY_MODEL_TASKS.iter().enumerate() {
            ui.line(&format!(
                "  {}. {} ({}) — {}",
                index + 1,
                task.name,
                task.description,
                format_auxiliary_task_current(root, task.key)
            ))?;
        }
        ui.line(&format!("  {}. Reset all to auto", reset_choice))?;
        ui.line(&format!("  {}. Back", back_choice))?;

        let selection = prompt_menu_choice(ui, "Select task: ", back_choice, 1)?;
        if selection == back_choice {
            return Ok(());
        }
        if selection == reset_choice {
            let count = reset_all_auxiliary_tasks(root);
            write_yaml_mapping(&context.config_path(), root)?;
            if count == 0 {
                ui.line("All auxiliary tasks were already set to auto.")?;
            } else {
                ui.line(&format!("Reset {count} auxiliary task(s) to auto."))?;
            }
            continue;
        }

        configure_auxiliary_model_task(context, ui, root, AUXILIARY_MODEL_TASKS[selection - 1])?;
    }
}

fn configure_auxiliary_model_task(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    root: &mut Mapping,
    task: AuxiliaryModelTask,
) -> Result<(), Box<dyn Error>> {
    let current_provider = get_nested_string(root, &["auxiliary", task.key, "provider"])
        .unwrap_or_else(|| "auto".to_string());
    let current_model =
        get_nested_string(root, &["auxiliary", task.key, "model"]).unwrap_or_default();
    let current_base_url =
        get_nested_string(root, &["auxiliary", task.key, "base_url"]).unwrap_or_default();
    let current_api_key =
        get_nested_string(root, &["auxiliary", task.key, "api_key"]).unwrap_or_default();
    let providers = native_auxiliary_model_providers(context, &current_provider, &current_base_url);

    ui.blank()?;
    ui.line(&format!(
        "Configure {} — current: {}",
        task.name,
        format_auxiliary_task_current(root, task.key)
    ))?;
    ui.blank()?;
    ui.line("  1. auto (recommended)")?;
    for (index, provider) in providers.iter().enumerate() {
        ui.line(&format!("  {}. {}", index + 2, provider.label))?;
    }
    let custom_choice = providers.len() + 2;
    let back_choice = custom_choice + 1;
    ui.line(&format!(
        "  {}. Custom endpoint (direct URL)",
        custom_choice
    ))?;
    ui.line(&format!("  {}. Back", back_choice))?;

    let default_choice = if !current_base_url.trim().is_empty() {
        custom_choice
    } else if current_provider.trim().eq_ignore_ascii_case("auto") {
        1
    } else {
        providers
            .iter()
            .position(|provider| provider.name == current_provider)
            .map(|index| index + 2)
            .unwrap_or(1)
    };
    let selection = prompt_menu_choice(ui, "Select provider: ", back_choice, default_choice)?;
    if selection == back_choice {
        ui.line("No change.")?;
        return Ok(());
    }
    if selection == 1 {
        save_auxiliary_task_choice(root, task.key, "auto", "", "", "");
        write_yaml_mapping(&context.config_path(), root)?;
        ui.line(&format!("{}: reset to auto.", task.name))?;
        return Ok(());
    }
    if selection == custom_choice {
        return run_native_auxiliary_custom_endpoint_setup(
            context,
            ui,
            root,
            task,
            &current_base_url,
            &current_model,
            &current_api_key,
        );
    }

    let provider = &providers[selection - 2];
    let model_name = prompt_auxiliary_model_name(ui, &current_model)?;
    save_auxiliary_task_choice(root, task.key, &provider.name, &model_name, "", "");
    write_yaml_mapping(&context.config_path(), root)?;
    if model_name.trim().is_empty() {
        ui.line(&format!(
            "{}: {} (provider default model)",
            task.name, provider.label
        ))?;
    } else {
        ui.line(&format!(
            "{}: {} · {}",
            task.name,
            provider.label,
            model_name.trim()
        ))?;
    }
    Ok(())
}

fn run_native_auxiliary_custom_endpoint_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    root: &mut Mapping,
    task: AuxiliaryModelTask,
    current_base_url: &str,
    current_model: &str,
    current_api_key: &str,
) -> Result<(), Box<dyn Error>> {
    ui.blank()?;
    ui.line(&format!("Custom endpoint for {}", task.name))?;
    ui.line("Provide an OpenAI-compatible base URL for this side task.")?;
    ui.blank()?;

    let mut base_url = match prompt_custom_endpoint_base_url(ui, current_base_url)? {
        Some(value) => value,
        None => {
            ui.line("No URL provided. No change.")?;
            return Ok(());
        }
    };
    base_url = maybe_add_local_v1_suffix(ui, &base_url)?;
    let model_name = prompt_auxiliary_model_name(ui, current_model)?;
    let api_key = prompt_custom_endpoint_api_key(ui, current_api_key)?;

    save_auxiliary_task_choice(root, task.key, "custom", &model_name, &base_url, &api_key);
    write_yaml_mapping(&context.config_path(), root)?;
    let short = format_saved_custom_provider_url(&base_url);
    if model_name.trim().is_empty() {
        ui.line(&format!("{}: custom ({short})", task.name))?;
    } else {
        ui.line(&format!(
            "{}: custom ({short}) · {}",
            task.name,
            model_name.trim()
        ))?;
    }
    Ok(())
}

fn prompt_auxiliary_model_name(
    ui: &mut dyn SetupUi,
    current: &str,
) -> Result<String, Box<dyn Error>> {
    let prompt = if current.trim().is_empty() {
        "Model slug [optional, '-' for provider default]: "
    } else {
        "Model slug [press Enter to keep current, '-' for provider default]: "
    };
    let input = ui.prompt(prompt)?;
    let trimmed = input.trim();
    if trimmed == "-" {
        return Ok(String::new());
    }
    if trimmed.is_empty() {
        return Ok(current.trim().to_string());
    }
    Ok(trimmed.to_string())
}

fn native_auxiliary_model_providers(
    context: &HermesContext,
    current_provider: &str,
    current_base_url: &str,
) -> Vec<NativeAuxiliaryProvider> {
    let current_provider = canonical_provider_name(current_provider);
    let current_is_custom = !current_base_url.trim().is_empty();
    let mut providers = list_provider_profiles()
        .into_iter()
        .filter(|profile| profile.name != "custom")
        .filter(|profile| {
            (!current_is_custom && current_provider.as_deref() == Some(profile.name))
                || provider_available_for_auxiliary(context, profile.name)
        })
        .map(|profile| NativeAuxiliaryProvider {
            name: profile.name.to_string(),
            label: model_provider_label(profile.name),
        })
        .collect::<Vec<_>>();
    providers.sort_by_key(|provider| model_provider_order(&provider.name));
    providers.dedup_by(|left, right| left.name == right.name);
    providers
}

fn provider_available_for_auxiliary(context: &HermesContext, provider: &str) -> bool {
    get_auth_status_summary(context.hermes_home().as_path(), provider)
        .map(|status| status.configured || status.logged_in)
        .unwrap_or(false)
}

fn canonical_provider_name(provider: &str) -> Option<String> {
    let normalized = provider.trim();
    if normalized.is_empty() {
        return None;
    }
    list_provider_profiles()
        .into_iter()
        .find(|profile| profile.matches(normalized))
        .map(|profile| profile.name.to_string())
        .or_else(|| Some(normalized.to_ascii_lowercase()))
}

fn format_auxiliary_task_current(root: &Mapping, task: &str) -> String {
    let provider = get_nested_string(root, &["auxiliary", task, "provider"])
        .unwrap_or_else(|| "auto".to_string());
    let model = get_nested_string(root, &["auxiliary", task, "model"]).unwrap_or_default();
    let base_url = get_nested_string(root, &["auxiliary", task, "base_url"]).unwrap_or_default();
    if !base_url.trim().is_empty() {
        let short = format_saved_custom_provider_url(&base_url);
        if model.trim().is_empty() {
            return format!("custom ({short})");
        }
        return format!("custom ({short}) · {}", model.trim());
    }
    if provider.trim().is_empty() || provider.trim().eq_ignore_ascii_case("auto") {
        if model.trim().is_empty() {
            return "auto".to_string();
        }
        return format!("auto · {}", model.trim());
    }
    let label = model_provider_label(provider.trim());
    if model.trim().is_empty() {
        return label;
    }
    format!("{label} · {}", model.trim())
}

fn save_auxiliary_task_choice(
    root: &mut Mapping,
    task: &str,
    provider: &str,
    model: &str,
    base_url: &str,
    api_key: &str,
) {
    let auxiliary = ensure_mapping(root, "auxiliary");
    let entry = ensure_mapping(auxiliary, task);
    entry.insert(
        yaml_key("provider"),
        Value::String(provider.trim().to_string()),
    );
    entry.insert(yaml_key("model"), Value::String(model.trim().to_string()));
    entry.insert(
        yaml_key("base_url"),
        Value::String(base_url.trim_end_matches('/').to_string()),
    );
    entry.insert(
        yaml_key("api_key"),
        Value::String(api_key.trim().to_string()),
    );
}

fn reset_all_auxiliary_tasks(root: &mut Mapping) -> usize {
    let auxiliary = ensure_mapping(root, "auxiliary");
    let mut reset = 0usize;
    for task in AUXILIARY_MODEL_TASKS {
        let entry = ensure_mapping(auxiliary, task.key);
        let changed = mapping_string(entry, "provider")
            .is_some_and(|value| !value.trim().is_empty() && !value.eq_ignore_ascii_case("auto"))
            || mapping_string(entry, "model").is_some_and(|value| !value.trim().is_empty())
            || mapping_string(entry, "base_url").is_some_and(|value| !value.trim().is_empty())
            || mapping_string(entry, "api_key").is_some_and(|value| !value.trim().is_empty());
        entry.insert(yaml_key("provider"), Value::String("auto".to_string()));
        entry.insert(yaml_key("model"), Value::String(String::new()));
        entry.insert(yaml_key("base_url"), Value::String(String::new()));
        entry.insert(yaml_key("api_key"), Value::String(String::new()));
        if changed {
            reset += 1;
        }
    }
    reset
}

fn run_native_gateway_setup(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    if let Some(system) = setup_managed_system(context) {
        eprintln!(
            "{}",
            format_setup_managed_message(&system, "run setup wizard")
        );
        return Ok(());
    }
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    run_native_gateway_setup_with_io(context, &mut input, &mut output)
}

fn run_native_gateway_setup_with_io(
    context: &HermesContext,
    input: &mut dyn io::BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    if let Some(system) = setup_managed_system(context) {
        writeln!(
            output,
            "{}",
            format_setup_managed_message(&system, "run setup wizard")
        )?;
        return Ok(());
    }
    gateway_cmd::run_gateway_setup_with_io(context, input, output, false)
}

fn run_native_tools_setup_with_io(
    context: &HermesContext,
    input: &mut dyn io::BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    if let Some(system) = setup_managed_system(context) {
        writeln!(
            output,
            "{}",
            format_setup_managed_message(&system, "run setup wizard")
        )?;
        return Ok(());
    }
    tools_cmd::run_native_tools_interactive_with_io(context, input, output)
}

fn setup_managed_system(context: &HermesContext) -> Option<String> {
    if let Ok(raw) = std::env::var("HERMES_MANAGED") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let normalized = trimmed.to_ascii_lowercase();
            return Some(match normalized.as_str() {
                "true" | "1" | "yes" | "nix" | "nixos" => String::from("NixOS"),
                "brew" | "homebrew" => String::from("Homebrew"),
                _ => trimmed.to_string(),
            });
        }
    }
    context
        .hermes_home()
        .join(".managed")
        .exists()
        .then_some(String::from("NixOS"))
}

fn format_setup_managed_message(system: &str, action: &str) -> String {
    let raw = std::env::var("HERMES_MANAGED").unwrap_or_default();
    let normalized = raw.trim().to_ascii_lowercase();
    if system == "NixOS" {
        let env_hint = if matches!(normalized.as_str(), "true" | "1" | "yes") {
            "true"
        } else if raw.trim().is_empty() {
            "true"
        } else {
            raw.trim()
        };
        return format!(
            "Cannot {action}: this Hermes installation is managed by NixOS (HERMES_MANAGED={env_hint}).\nEdit services.hermes-agent.settings in your configuration.nix and run:\n  sudo nixos-rebuild switch"
        );
    }
    if system == "Homebrew" {
        let env_hint = if raw.trim().is_empty() {
            "homebrew"
        } else {
            raw.trim()
        };
        return format!(
            "Cannot {action}: this Hermes installation is managed by Homebrew (HERMES_MANAGED={env_hint}).\nUse:\n  brew upgrade hermes-agent"
        );
    }
    format!(
        "Cannot {action}: this Hermes installation is managed by {system}.\nUse your package manager to upgrade or reinstall Hermes."
    )
}

trait SetupUi {
    fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>>;
    fn prompt(&mut self, prompt: &str) -> Result<String, Box<dyn Error>>;
    fn prompt_secret(&mut self, prompt: &str) -> Result<String, Box<dyn Error>>;

    fn blank(&mut self) -> Result<(), Box<dyn Error>> {
        self.line("")
    }
}

struct TerminalUi;

impl SetupUi for TerminalUi {
    fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>> {
        println!("{text}");
        Ok(())
    }

    fn prompt(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
        let mut stdout = io::stdout();
        stdout.write_all(prompt.as_bytes())?;
        stdout.flush()?;
        let mut input = String::new();
        let read = io::stdin().read_line(&mut input)?;
        if read == 0 {
            return Err("setup cancelled".into());
        }
        Ok(input.trim().to_string())
    }

    fn prompt_secret(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
        let mut stdout = io::stdout();
        stdout.write_all(prompt.as_bytes())?;
        stdout.flush()?;

        let echo_disabled = if cfg!(unix) && io::stdin().is_terminal() {
            Command::new("stty")
                .arg("-echo")
                .status()
                .ok()
                .is_some_and(|status| status.success())
        } else {
            false
        };

        let mut input = String::new();
        let read = io::stdin().read_line(&mut input)?;

        if echo_disabled {
            let _ = Command::new("stty").arg("echo").status();
            println!();
        }

        if read == 0 {
            return Err("setup cancelled".into());
        }
        Ok(input.trim().to_string())
    }
}

struct StreamUi<'a> {
    input: &'a mut dyn BufRead,
    output: &'a mut dyn Write,
}

struct SetupUiWriteAdapter<'a> {
    ui: &'a mut dyn SetupUi,
    buffer: String,
}

impl<'a> SetupUiWriteAdapter<'a> {
    fn new(ui: &'a mut dyn SetupUi) -> Self {
        Self {
            ui,
            buffer: String::new(),
        }
    }

    fn flush_buffered_line(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let line = std::mem::take(&mut self.buffer);
        self.ui
            .line(&line)
            .map_err(|error| io::Error::other(error.to_string()))
    }
}

impl Write for SetupUiWriteAdapter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        for segment in text.split_inclusive('\n') {
            if let Some(line) = segment.strip_suffix('\n') {
                self.buffer
                    .push_str(line.strip_suffix('\r').unwrap_or(line));
                self.flush_buffered_line()?;
            } else {
                self.buffer.push_str(segment);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffered_line()
    }
}

struct SharedSetupUiWriteAdapter<'a> {
    ui: Rc<RefCell<&'a mut dyn SetupUi>>,
    buffer: String,
}

impl<'a> SharedSetupUiWriteAdapter<'a> {
    fn new(ui: Rc<RefCell<&'a mut dyn SetupUi>>) -> Self {
        Self {
            ui,
            buffer: String::new(),
        }
    }

    fn flush_buffered_line(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let line = std::mem::take(&mut self.buffer);
        self.ui
            .borrow_mut()
            .line(&line)
            .map_err(|error| io::Error::other(error.to_string()))
    }
}

impl Write for SharedSetupUiWriteAdapter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        for segment in text.split_inclusive('\n') {
            if let Some(line) = segment.strip_suffix('\n') {
                self.buffer
                    .push_str(line.strip_suffix('\r').unwrap_or(line));
                self.flush_buffered_line()?;
            } else {
                self.buffer.push_str(segment);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffered_line()
    }
}

#[derive(Debug, Clone)]
struct NativeModelProvider {
    name: String,
    label: String,
    base_url: &'static str,
    api_mode: &'static str,
    auth_type: &'static str,
    api_key_env_var: Option<String>,
    base_url_env_var: Option<String>,
}

const NATIVE_MODEL_RUNTIME_PROVIDERS: &[&str] = &[
    "nous",
    "openai-codex",
    "google-gemini-cli",
    "bedrock",
    "copilot",
    "qwen-oauth",
    "minimax-oauth",
];

#[derive(Debug, Clone)]
struct SavedCustomModelProvider {
    name: String,
    base_url: String,
    api_key_config_value: Option<String>,
    api_mode: Option<String>,
    model: Option<String>,
    source: SavedCustomModelProviderSource,
}

#[derive(Debug, Clone)]
enum SavedCustomModelProviderSource {
    LegacyCustomProviders {
        index: usize,
    },
    ProvidersMap {
        key: String,
        model_field_key: String,
    },
}

impl SavedCustomModelProvider {
    fn saved_provider_key(&self) -> Option<String> {
        match &self.source {
            SavedCustomModelProviderSource::LegacyCustomProviders { .. } => None,
            SavedCustomModelProviderSource::ProvidersMap { key, .. } => {
                Some(key.trim().to_ascii_lowercase())
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct AuxiliaryModelTask {
    key: &'static str,
    name: &'static str,
    description: &'static str,
}

const AUXILIARY_MODEL_TASKS: &[AuxiliaryModelTask] = &[
    AuxiliaryModelTask {
        key: "vision",
        name: "Vision",
        description: "image/screenshot analysis",
    },
    AuxiliaryModelTask {
        key: "compression",
        name: "Compression",
        description: "context summarization",
    },
    AuxiliaryModelTask {
        key: "web_extract",
        name: "Web extract",
        description: "web page summarization",
    },
    AuxiliaryModelTask {
        key: "session_search",
        name: "Session search",
        description: "past-conversation recall",
    },
    AuxiliaryModelTask {
        key: "approval",
        name: "Approval",
        description: "smart command approval",
    },
    AuxiliaryModelTask {
        key: "mcp",
        name: "MCP",
        description: "MCP tool reasoning",
    },
    AuxiliaryModelTask {
        key: "title_generation",
        name: "Title generation",
        description: "session titles",
    },
    AuxiliaryModelTask {
        key: "skills_hub",
        name: "Skills hub",
        description: "skills search/install",
    },
    AuxiliaryModelTask {
        key: "curator",
        name: "Curator",
        description: "skill-usage review pass",
    },
];

#[derive(Debug, Clone)]
struct NativeAuxiliaryProvider {
    name: String,
    label: String,
}

impl SetupUi for StreamUi<'_> {
    fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>> {
        writeln!(self.output, "{text}")?;
        Ok(())
    }

    fn prompt(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
        write!(self.output, "{prompt}")?;
        self.output.flush()?;
        let mut line = String::new();
        if self.input.read_line(&mut line)? == 0 {
            return Err("setup cancelled".into());
        }
        Ok(line.trim_end_matches(['\r', '\n']).trim().to_string())
    }

    fn prompt_secret(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
        self.prompt(prompt)
    }
}

fn run_native_agent_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let current_max = get_nested_i64(&root, &["agent", "max_turns"]).unwrap_or(90);
    let current_mode = get_nested_string(&root, &["display", "tool_progress"])
        .unwrap_or_else(|| "all".to_string());
    let current_threshold = get_nested_f64(&root, &["compression", "threshold"]).unwrap_or(0.50);
    let current_reset_mode =
        get_nested_string(&root, &["session_reset", "mode"]).unwrap_or_else(|| "both".to_string());
    let current_idle = get_nested_i64(&root, &["session_reset", "idle_minutes"]).unwrap_or(1440);
    let current_hour = get_nested_i64(&root, &["session_reset", "at_hour"]).unwrap_or(4);

    ui.blank()?;
    ui.line("⚕ Hermes Setup — Agent Settings")?;
    ui.line("Configure max turns, tool progress, compression, and session resets.")?;
    ui.blank()?;

    let max_turns = prompt_positive_i64(
        ui,
        "Max iterations",
        current_max,
        "Enter a positive integer.",
    )?;
    ensure_mapping(&mut root, "agent")
        .insert(yaml_key("max_turns"), serde_yaml::to_value(max_turns)?);
    root.remove(yaml_key("max_turns"));
    let _ = remove_env_key(&context.env_path(), "HERMES_MAX_ITERATIONS")?;
    ui.line(&format!("Max iterations set to {max_turns}"))?;

    ui.blank()?;
    ui.line("Tool progress mode: off, new, all, verbose")?;
    let tool_progress = prompt_enum_choice(
        ui,
        "Tool progress mode",
        &["off", "new", "all", "verbose"],
        &current_mode,
    )?;
    ensure_mapping(&mut root, "display").insert(
        yaml_key("tool_progress"),
        Value::String(tool_progress.clone()),
    );
    ui.line(&format!("Tool progress set to: {tool_progress}"))?;

    ui.blank()?;
    let compression_threshold = prompt_f64_range(
        ui,
        "Compression threshold (0.5-0.95)",
        current_threshold,
        0.5,
        0.95,
    )?;
    let compression = ensure_mapping(&mut root, "compression");
    compression.insert(yaml_key("enabled"), Value::Bool(true));
    compression.insert(
        yaml_key("threshold"),
        serde_yaml::to_value(compression_threshold)?,
    );
    ui.line(&format!(
        "Context compression threshold set to {compression_threshold}"
    ))?;

    ui.blank()?;
    ui.line("Session reset mode:")?;
    ui.line("  1. Inactivity + daily reset")?;
    ui.line("  2. Inactivity only")?;
    ui.line("  3. Daily only")?;
    ui.line("  4. Never auto-reset")?;
    let default_reset = match current_reset_mode.as_str() {
        "idle" => 2,
        "daily" => 3,
        "none" => 4,
        _ => 1,
    };
    let reset_choice = prompt_menu_choice(ui, "Select [1]: ", 4, default_reset)?;
    let session_reset = ensure_mapping(&mut root, "session_reset");
    match reset_choice {
        1 => {
            let idle = prompt_positive_i64(
                ui,
                "  Inactivity timeout (minutes)",
                current_idle,
                "Enter a positive integer.",
            )?;
            let hour = prompt_i64_range(
                ui,
                "  Daily reset hour (0-23, local time)",
                current_hour,
                0,
                23,
            )?;
            session_reset.insert(yaml_key("mode"), Value::String("both".to_string()));
            session_reset.insert(yaml_key("idle_minutes"), serde_yaml::to_value(idle)?);
            session_reset.insert(yaml_key("at_hour"), serde_yaml::to_value(hour)?);
            ui.line(&format!(
                "Sessions reset after {idle} min idle or daily at {hour}:00"
            ))?;
        }
        2 => {
            let idle = prompt_positive_i64(
                ui,
                "  Inactivity timeout (minutes)",
                current_idle,
                "Enter a positive integer.",
            )?;
            session_reset.insert(yaml_key("mode"), Value::String("idle".to_string()));
            session_reset.insert(yaml_key("idle_minutes"), serde_yaml::to_value(idle)?);
            ui.line(&format!("Sessions reset after {idle} min of inactivity"))?;
        }
        3 => {
            let hour = prompt_i64_range(
                ui,
                "  Daily reset hour (0-23, local time)",
                current_hour,
                0,
                23,
            )?;
            session_reset.insert(yaml_key("mode"), Value::String("daily".to_string()));
            session_reset.insert(yaml_key("at_hour"), serde_yaml::to_value(hour)?);
            ui.line(&format!("Sessions reset daily at {hour}:00"))?;
        }
        4 => {
            session_reset.insert(yaml_key("mode"), Value::String("none".to_string()));
            ui.line("Sessions will never auto-reset. Use /reset manually when needed.")?;
        }
        _ => unreachable!(),
    }

    write_yaml_mapping(&context.config_path(), &root)?;
    ui.blank()?;
    ui.line("Agent Settings configuration complete!")?;
    Ok(())
}

fn native_setup_model_providers(context: &HermesContext) -> Vec<NativeModelProvider> {
    let mut providers = list_provider_profiles()
        .into_iter()
        .filter(|profile| profile.name != "custom")
        .filter_map(|profile| {
            if profile.auth_type == "api_key" {
                return Some(NativeModelProvider {
                    name: profile.name.to_string(),
                    label: model_provider_label(profile.name),
                    base_url: profile.base_url,
                    api_mode: profile.api_mode,
                    auth_type: profile.auth_type,
                    api_key_env_var: profile.api_key_env_vars().next().map(ToOwned::to_owned),
                    base_url_env_var: profile.base_url_env_var().map(ToOwned::to_owned),
                });
            }
            if native_runtime_model_provider_supported(profile.name)
                && provider_visible_for_native_model_setup(context, profile.name)
            {
                return Some(NativeModelProvider {
                    name: profile.name.to_string(),
                    label: model_provider_label(profile.name),
                    base_url: profile.base_url,
                    api_mode: profile.api_mode,
                    auth_type: profile.auth_type,
                    api_key_env_var: None,
                    base_url_env_var: None,
                });
            }
            None
        })
        .collect::<Vec<_>>();
    providers.sort_by_key(|provider| model_provider_order(&provider.name));
    providers
}

fn native_runtime_model_provider_supported(provider: &str) -> bool {
    NATIVE_MODEL_RUNTIME_PROVIDERS.contains(&provider)
}

fn provider_visible_for_native_model_setup(context: &HermesContext, provider: &str) -> bool {
    provider_available_for_native_model_setup(context, provider)
        || provider_supports_fresh_native_model_setup(provider)
}

fn provider_available_for_native_model_setup(context: &HermesContext, provider: &str) -> bool {
    get_auth_status_summary(context.hermes_home().as_path(), provider)
        .map(|status| status.configured || status.logged_in)
        .unwrap_or(false)
}

fn provider_supports_fresh_native_model_setup(provider: &str) -> bool {
    matches!(
        provider,
        "openai-codex" | "minimax-oauth" | "nous" | "google-gemini-cli"
    )
}

fn ensure_runtime_model_provider_credentials(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    provider: &NativeModelProvider,
) -> Result<(), Box<dyn Error>> {
    if provider_available_for_native_model_setup(context, &provider.name) {
        ui.line(&format!(
            "{} credentials: already configured",
            provider.label
        ))?;
        return Ok(());
    }

    match provider.name.as_str() {
        "openai-codex" => {
            let mut output = SetupUiWriteAdapter::new(ui);
            auth_cmd::native_auth_add_openai_codex_oauth_with_io(
                context,
                &AuthAddArgs {
                    provider: provider.name.clone(),
                    auth_type: Some("oauth".to_string()),
                    ..AuthAddArgs::default()
                },
                &mut output,
            )?;
            output.flush()?;
        }
        "minimax-oauth" => {
            let mut output = SetupUiWriteAdapter::new(ui);
            auth_cmd::native_auth_add_minimax_oauth_with_io(
                context,
                &AuthAddArgs {
                    provider: provider.name.clone(),
                    auth_type: Some("oauth".to_string()),
                    ..AuthAddArgs::default()
                },
                &mut output,
            )?;
            output.flush()?;
        }
        "nous" => {
            let mut output = SetupUiWriteAdapter::new(ui);
            auth_cmd::native_auth_add_nous_oauth_with_io(
                context,
                &AuthAddArgs {
                    provider: provider.name.clone(),
                    auth_type: Some("oauth".to_string()),
                    ..AuthAddArgs::default()
                },
                &mut output,
            )?;
            output.flush()?;
        }
        "google-gemini-cli" => {
            let shared_ui = Rc::new(RefCell::new(ui));
            let mut output = SharedSetupUiWriteAdapter::new(Rc::clone(&shared_ui));
            auth_cmd::native_auth_add_google_gemini_oauth_with_prompt(
                context,
                &AuthAddArgs {
                    provider: provider.name.clone(),
                    auth_type: Some("oauth".to_string()),
                    ..AuthAddArgs::default()
                },
                &mut output,
                |_output| shared_ui.borrow_mut().prompt("Callback URL or code"),
            )?;
            output.flush()?;
        }
        _ => {
            return Err(
                format!("{} is not available for fresh native setup", provider.label).into(),
            );
        }
    }

    if !provider_available_for_native_model_setup(context, &provider.name) {
        return Err(format!(
            "{} did not persist usable auth state after native setup",
            provider.label
        )
        .into());
    }
    Ok(())
}

fn resolved_native_model_provider_base_url(
    context: &HermesContext,
    provider: &NativeModelProvider,
    current_provider: &str,
    current_base_url: &str,
) -> Result<String, Box<dyn Error>> {
    if provider.name == current_provider && !current_base_url.trim().is_empty() {
        return Ok(current_base_url.trim_end_matches('/').to_string());
    }

    let auth_path = context.hermes_home().join("auth.json");
    let auth_store = if auth_path.exists() {
        serde_json::from_str::<JsonValue>(&fs::read_to_string(&auth_path)?)?
    } else {
        JsonValue::Null
    };

    let from_state = match provider.name.as_str() {
        "nous" | "minimax-oauth" => auth_store
            .get("providers")
            .and_then(|providers| providers.get(&provider.name))
            .and_then(|state| state.get("inference_base_url"))
            .and_then(JsonValue::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.trim_end_matches('/').to_string()),
        _ => None,
    };
    if let Some(base_url) = from_state {
        return Ok(base_url);
    }

    if provider.name == "openai-codex"
        && let Some(base_url) = std::env::var("HERMES_CODEX_BASE_URL")
            .ok()
            .map(|value| value.trim().trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty())
    {
        return Ok(base_url);
    }

    if provider.name == "qwen-oauth"
        && let Some(base_url) = std::env::var("HERMES_QWEN_BASE_URL")
            .ok()
            .map(|value| value.trim().trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty())
    {
        return Ok(base_url);
    }

    if provider.name == "bedrock"
        && let Some(region) = ["AWS_REGION", "AWS_DEFAULT_REGION"]
            .into_iter()
            .find_map(|env_var| {
                std::env::var(env_var)
                    .ok()
                    .map(|value| value.trim().to_ascii_lowercase())
                    .filter(|value| {
                        !value.is_empty()
                            && value.chars().all(|ch| {
                                ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-'
                            })
                    })
            })
    {
        return Ok(format!("https://bedrock-runtime.{region}.amazonaws.com"));
    }

    if provider.base_url.trim().is_empty() {
        return Err(format!("{} has no default base URL", provider.label).into());
    }
    Ok(provider.base_url.trim_end_matches('/').to_string())
}

fn collect_saved_custom_model_providers(root: &Mapping) -> Vec<SavedCustomModelProvider> {
    let mut providers = Vec::new();
    let mut seen_provider_keys = HashSet::new();
    let mut seen_name_url_model = HashSet::new();

    if let Some(entries) = root.get(yaml_key("providers")).and_then(Value::as_mapping) {
        for (key, value) in entries {
            let Some(provider_key) = key.as_str().map(str::trim).filter(|key| !key.is_empty())
            else {
                continue;
            };
            let Some(mapping) = value.as_mapping() else {
                continue;
            };
            if let Some(provider) = saved_custom_model_provider_from_mapping(
                mapping,
                Some(provider_key),
                SavedCustomModelProviderSource::ProvidersMap {
                    key: provider_key.to_string(),
                    model_field_key: preferred_saved_provider_model_key(mapping),
                },
            ) {
                append_saved_custom_provider(
                    &mut providers,
                    &mut seen_provider_keys,
                    &mut seen_name_url_model,
                    provider,
                );
            }
        }
    }

    if let Some(entries) = root
        .get(yaml_key("custom_providers"))
        .and_then(Value::as_sequence)
    {
        for (index, value) in entries.iter().enumerate() {
            let Some(mapping) = value.as_mapping() else {
                continue;
            };
            if let Some(provider) = saved_custom_model_provider_from_mapping(
                mapping,
                None,
                SavedCustomModelProviderSource::LegacyCustomProviders { index },
            ) {
                append_saved_custom_provider(
                    &mut providers,
                    &mut seen_provider_keys,
                    &mut seen_name_url_model,
                    provider,
                );
            }
        }
    }

    providers
}

fn append_saved_custom_provider(
    providers: &mut Vec<SavedCustomModelProvider>,
    seen_provider_keys: &mut HashSet<String>,
    seen_name_url_model: &mut HashSet<(String, String, String)>,
    provider: SavedCustomModelProvider,
) {
    let provider_key = provider.saved_provider_key();
    let name = provider.name.trim().to_ascii_lowercase();
    let base_url = provider.base_url.trim_end_matches('/').to_ascii_lowercase();
    let model = provider
        .model
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();

    if let Some(provider_key) = provider_key.as_deref() {
        if seen_provider_keys.contains(provider_key) {
            return;
        }
    }
    if !name.is_empty()
        && !base_url.is_empty()
        && seen_name_url_model.contains(&(name.clone(), base_url.clone(), model.clone()))
    {
        return;
    }

    if let Some(provider_key) = provider_key {
        seen_provider_keys.insert(provider_key);
    }
    if !name.is_empty() && !base_url.is_empty() {
        seen_name_url_model.insert((name, base_url, model));
    }
    providers.push(provider);
}

fn saved_custom_model_provider_from_mapping(
    mapping: &Mapping,
    provider_key: Option<&str>,
    source: SavedCustomModelProviderSource,
) -> Option<SavedCustomModelProvider> {
    let base_url = mapping_string_alias(mapping, &["base_url", "url", "api", "baseUrl"])?;
    if !looks_like_http_url(&base_url) {
        return None;
    }

    let name = mapping_string_alias(mapping, &["name"])
        .or_else(|| provider_key.map(|value| value.trim().to_string()))
        .filter(|value| !value.trim().is_empty())?;
    let raw_api_key = mapping_string_alias(mapping, &["api_key", "apiKey"]);
    let key_env = mapping_string_alias(mapping, &["key_env", "api_key_env", "keyEnv", "apiKeyEnv"]);
    let api_key_config_value = raw_api_key.or_else(|| key_env.map(|key| format!("${{{key}}}")));

    Some(SavedCustomModelProvider {
        name,
        base_url: base_url.trim_end_matches('/').to_string(),
        api_key_config_value,
        api_mode: mapping_string_alias(mapping, &["api_mode", "transport", "apiMode"]),
        model: mapping_string_alias(mapping, &["model", "default_model", "defaultModel"]),
        source,
    })
}

fn preferred_saved_provider_model_key(mapping: &Mapping) -> String {
    if mapping.contains_key(yaml_key("model")) {
        return String::from("model");
    }
    if mapping.contains_key(yaml_key("default_model")) {
        return String::from("default_model");
    }
    if mapping.contains_key(yaml_key("defaultModel")) {
        return String::from("defaultModel");
    }
    String::from("default_model")
}

fn format_saved_custom_provider_url(base_url: &str) -> String {
    base_url
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string()
}

fn run_native_custom_model_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    root: &mut Mapping,
    current_provider: &str,
    current_model: &str,
    current_base_url: &str,
    current_api_key: &str,
) -> Result<(), Box<dyn Error>> {
    ui.blank()?;
    ui.line("Custom OpenAI-compatible endpoint configuration:")?;
    if !current_base_url.trim().is_empty() {
        ui.line(&format!("  Current URL: {}", current_base_url.trim()))?;
    }
    if !current_api_key.trim().is_empty() {
        ui.line("  API key: already configured")?;
    }
    ui.blank()?;

    let mut base_url = match prompt_custom_endpoint_base_url(ui, current_base_url)? {
        Some(value) => value,
        None => {
            ui.line("No URL provided. Cancelled.")?;
            return Ok(());
        }
    };
    let api_key = prompt_custom_endpoint_api_key(ui, current_api_key)?;
    base_url = maybe_add_local_v1_suffix(ui, &base_url)?;

    let suggested_model = if current_provider == "custom" {
        current_model
    } else {
        ""
    };
    let model_name = prompt_optional_model_name(ui, suggested_model)?;
    let context_length = prompt_optional_context_length(ui)?;
    let display_name =
        prompt_with_default(ui, "Display name", &auto_custom_provider_name(&base_url))?;

    apply_custom_model_provider_choice(
        root,
        &base_url,
        if api_key.trim().is_empty() {
            None
        } else {
            Some(api_key.trim())
        },
        if model_name.trim().is_empty() {
            None
        } else {
            Some(model_name.trim())
        },
    )?;
    save_custom_model_provider(
        root,
        &base_url,
        if api_key.trim().is_empty() {
            None
        } else {
            Some(api_key.trim())
        },
        if model_name.trim().is_empty() {
            None
        } else {
            Some(model_name.trim())
        },
        context_length,
        &display_name,
    );
    clear_active_provider_marker(context.hermes_home().as_path())?;
    write_yaml_mapping(&context.config_path(), root)?;

    if model_name.trim().is_empty() {
        ui.line("Endpoint saved. Use `hermes model` to set a model.")?;
    } else {
        ui.line(&format!(
            "Default model set to: {} (via {})",
            normalize_model_for_provider(model_name.trim(), "custom"),
            base_url
        ))?;
    }
    Ok(())
}

fn prompt_custom_endpoint_base_url(
    ui: &mut dyn SetupUi,
    current: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    loop {
        let prompt = if current.trim().is_empty() {
            "API base URL [e.g. https://api.example.com/v1]: "
        } else {
            "API base URL [press Enter to keep current]: "
        };
        let input = ui.prompt(prompt)?;
        let value = if input.trim().is_empty() {
            current.trim().to_string()
        } else {
            input.trim().to_string()
        };
        if value.is_empty() {
            return Ok(None);
        }
        if !looks_like_http_url(&value) {
            ui.line("Base URL must start with http:// or https://")?;
            continue;
        }
        return Ok(Some(value.trim_end_matches('/').to_string()));
    }
}

fn prompt_custom_endpoint_api_key(
    ui: &mut dyn SetupUi,
    current: &str,
) -> Result<String, Box<dyn Error>> {
    let prompt = if current.trim().is_empty() {
        "API key [optional]: "
    } else {
        "API key [optional, press Enter to keep current]: "
    };
    let input = ui.prompt_secret(prompt)?;
    if input.trim().is_empty() {
        return Ok(current.trim().to_string());
    }
    Ok(input.trim().to_string())
}

fn maybe_add_local_v1_suffix(
    ui: &mut dyn SetupUi,
    base_url: &str,
) -> Result<String, Box<dyn Error>> {
    let normalized = base_url.trim_end_matches('/').to_ascii_lowercase();
    let looks_local = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        ":11434",
        ":8080",
        ":5000",
    ]
    .iter()
    .any(|marker| normalized.contains(marker));
    if !looks_local || normalized.ends_with("/v1") {
        return Ok(base_url.trim_end_matches('/').to_string());
    }

    ui.blank()?;
    ui.line("Hint: most local model servers require /v1 in the base URL.")?;
    ui.line(&format!(
        "  Suggested URL: {}/v1",
        base_url.trim_end_matches('/')
    ))?;
    if prompt_yes_no(ui, "Add /v1? [Y/n]: ", true)? {
        return Ok(format!("{}/v1", base_url.trim_end_matches('/')));
    }
    Ok(base_url.trim_end_matches('/').to_string())
}

fn prompt_optional_model_name(
    ui: &mut dyn SetupUi,
    current: &str,
) -> Result<String, Box<dyn Error>> {
    if current.trim().is_empty() {
        return Ok(ui
            .prompt("Default model [leave blank to skip]: ")?
            .trim()
            .to_string());
    }
    let input = ui.prompt(&format!(
        "Default model [{current}] (leave blank to keep current): "
    ))?;
    if input.trim().is_empty() {
        return Ok(current.trim().to_string());
    }
    Ok(input.trim().to_string())
}

fn prompt_optional_context_length(ui: &mut dyn SetupUi) -> Result<Option<i64>, Box<dyn Error>> {
    loop {
        let input = ui.prompt("Context length in tokens [leave blank for auto-detect]: ")?;
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let normalized = trimmed.replace(',', "");
        let parsed = if normalized.ends_with('k') || normalized.ends_with('K') {
            normalized[..normalized.len() - 1]
                .parse::<i64>()
                .ok()
                .and_then(|value| value.checked_mul(1000))
        } else {
            normalized.parse::<i64>().ok()
        };
        match parsed {
            Some(value) if value > 0 => return Ok(Some(value)),
            _ => ui.line("Enter a positive integer, or leave it blank.")?,
        }
    }
}

fn auto_custom_provider_name(base_url: &str) -> String {
    let mut clean = base_url
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string();
    if clean.ends_with("/v1") {
        clean.truncate(clean.len() - 3);
        clean = clean.trim_end_matches('/').to_string();
    }
    let host = clean.split('/').next().unwrap_or(clean.as_str()).trim();
    if host.contains("localhost") || host.contains("127.0.0.1") {
        return format!("Local ({host})");
    }
    if host.to_ascii_lowercase().contains("runpod") {
        return format!("RunPod ({host})");
    }
    let mut chars = host.chars();
    match chars.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::from("Custom endpoint"),
    }
}

fn apply_custom_model_provider_choice(
    root: &mut Mapping,
    base_url: &str,
    api_key: Option<&str>,
    model_name: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let model = ensure_mapping(root, "model");
    model.insert(yaml_key("provider"), Value::String(String::from("custom")));
    model.insert(
        yaml_key("base_url"),
        Value::String(base_url.trim_end_matches('/').to_string()),
    );
    if let Some(model_name) = model_name {
        model.insert(
            yaml_key("default"),
            Value::String(normalize_model_for_provider(model_name, "custom")),
        );
    }
    if let Some(api_key) = api_key {
        model.insert(yaml_key("api_key"), Value::String(api_key.to_string()));
    } else {
        model.remove(yaml_key("api_key"));
    }
    model.remove(yaml_key("api_mode"));
    Ok(())
}

fn save_custom_model_provider(
    root: &mut Mapping,
    base_url: &str,
    api_key: Option<&str>,
    model_name: Option<&str>,
    context_length: Option<i64>,
    display_name: &str,
) {
    let providers = ensure_sequence(root, "custom_providers");
    let normalized_base_url = base_url.trim_end_matches('/');
    for entry in providers.iter_mut() {
        let Some(mapping) = entry.as_mapping_mut() else {
            continue;
        };
        let Some(existing_base_url) =
            mapping_string_alias(mapping, &["base_url", "url", "api", "baseUrl"])
        else {
            continue;
        };
        if existing_base_url.trim_end_matches('/') != normalized_base_url {
            continue;
        }
        if let Some(model_name) = model_name {
            if mapping_string(mapping, "model").as_deref() != Some(model_name) {
                mapping.insert(yaml_key("model"), Value::String(model_name.to_string()));
            }
            if let Some(context_length) = context_length {
                let models = ensure_mapping(mapping, "models");
                let entry = ensure_mapping(models, model_name);
                entry.insert(
                    yaml_key("context_length"),
                    serde_yaml::to_value(context_length).unwrap_or(Value::Null),
                );
            }
        }
        return;
    }

    let mut entry = Mapping::new();
    entry.insert(
        yaml_key("name"),
        Value::String(display_name.trim().to_string()),
    );
    entry.insert(
        yaml_key("base_url"),
        Value::String(normalized_base_url.to_string()),
    );
    if let Some(api_key) = api_key {
        entry.insert(yaml_key("api_key"), Value::String(api_key.to_string()));
    }
    if let Some(model_name) = model_name {
        entry.insert(yaml_key("model"), Value::String(model_name.to_string()));
        if let Some(context_length) = context_length {
            let mut models = Mapping::new();
            let mut model_settings = Mapping::new();
            model_settings.insert(
                yaml_key("context_length"),
                serde_yaml::to_value(context_length).unwrap_or(Value::Null),
            );
            models.insert(yaml_key(model_name), Value::Mapping(model_settings));
            entry.insert(yaml_key("models"), Value::Mapping(models));
        }
    }
    providers.push(Value::Mapping(entry));
}

fn remove_saved_custom_model_provider(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    root: &mut Mapping,
    providers: &[SavedCustomModelProvider],
) -> Result<(), Box<dyn Error>> {
    if providers.is_empty() {
        ui.line("No saved custom providers configured.")?;
        return Ok(());
    }

    ui.blank()?;
    ui.line("Remove a saved custom provider:")?;
    ui.blank()?;
    for (index, provider) in providers.iter().enumerate() {
        ui.line(&format!(
            "  {}. {} ({})",
            index + 1,
            provider.name,
            format_saved_custom_provider_url(&provider.base_url)
        ))?;
    }
    let cancel_choice = providers.len() + 1;
    ui.line(&format!("  {}. Cancel", cancel_choice))?;

    let selection = prompt_menu_choice(ui, "Select provider to remove: ", cancel_choice, 1)?;
    if selection == cancel_choice {
        ui.line("No change.")?;
        return Ok(());
    }

    let selected = &providers[selection - 1];
    remove_saved_custom_model_provider_from_root(root, &selected.source)?;
    write_yaml_mapping(&context.config_path(), root)?;
    ui.line(&format!(
        "Removed \"{}\" from saved custom providers.",
        selected.name
    ))?;
    Ok(())
}

fn remove_saved_custom_model_provider_from_root(
    root: &mut Mapping,
    source: &SavedCustomModelProviderSource,
) -> Result<(), Box<dyn Error>> {
    match source {
        SavedCustomModelProviderSource::LegacyCustomProviders { index } => {
            let Some(entries) = root
                .get_mut(yaml_key("custom_providers"))
                .and_then(Value::as_sequence_mut)
            else {
                return Err("custom_providers list is missing".into());
            };
            if *index >= entries.len() {
                return Err("saved custom provider entry no longer exists".into());
            }
            entries.remove(*index);
        }
        SavedCustomModelProviderSource::ProvidersMap { key, .. } => {
            let Some(entries) = root
                .get_mut(yaml_key("providers"))
                .and_then(Value::as_mapping_mut)
            else {
                return Err("providers mapping is missing".into());
            };
            if entries.remove(yaml_key(key)).is_none() {
                return Err("saved custom provider entry no longer exists".into());
            }
        }
    }
    Ok(())
}

fn model_provider_order(provider: &str) -> (usize, String) {
    let priority = match provider {
        "openrouter" => 0,
        "nous" => 1,
        "openai" => 2,
        "openai-codex" => 3,
        "anthropic" => 4,
        "bedrock" => 5,
        "copilot" => 6,
        "deepseek" => 7,
        "gemini" => 8,
        "google-gemini-cli" => 9,
        "qwen-oauth" => 10,
        "minimax" => 11,
        "minimax-oauth" => 12,
        "xai" => 13,
        "zai" => 14,
        "ai-gateway" => 15,
        _ => 100,
    };
    (priority, provider.to_string())
}

fn model_provider_label(provider: &str) -> String {
    match provider {
        "openrouter" => "OpenRouter".to_string(),
        "nous" => "Nous Portal".to_string(),
        "openai" => "OpenAI".to_string(),
        "openai-codex" => "OpenAI Codex".to_string(),
        "anthropic" => "Anthropic".to_string(),
        "bedrock" => "AWS Bedrock".to_string(),
        "copilot" => "GitHub Copilot".to_string(),
        "deepseek" => "DeepSeek".to_string(),
        "gemini" => "Google Gemini API".to_string(),
        "google-gemini-cli" => "Google Gemini (OAuth)".to_string(),
        "qwen-oauth" => "Qwen OAuth".to_string(),
        "xai" => "xAI".to_string(),
        "zai" => "Z.AI / GLM".to_string(),
        "minimax" => "MiniMax".to_string(),
        "minimax-oauth" => "MiniMax (OAuth)".to_string(),
        "minimax-cn" => "MiniMax CN".to_string(),
        "kimi-coding" => "Kimi Coding".to_string(),
        "kimi-coding-cn" => "Kimi Coding (China)".to_string(),
        "ai-gateway" => "Vercel AI Gateway".to_string(),
        "opencode-zen" => "OpenCode Zen".to_string(),
        "opencode-go" => "OpenCode Go".to_string(),
        "lmstudio" => "LM Studio".to_string(),
        other => other
            .split('-')
            .filter(|segment| !segment.is_empty())
            .map(|segment| {
                let mut chars = segment.chars();
                match chars.next() {
                    Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn ensure_model_provider_secret(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    provider: &NativeModelProvider,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key_env_var) = provider.api_key_env_var.as_deref() else {
        return Ok(());
    };
    let current = env_value_for_context(context, api_key_env_var);
    if current.is_some() {
        ui.line(&format!("{} API key: already configured", provider.label))?;
        if !prompt_yes_no(ui, "  Update API key? [y/N]: ", false)? {
            return Ok(());
        }
    } else {
        ui.line(&format!(
            "{} requires an API key in {}.",
            provider.label, api_key_env_var
        ))?;
    }

    let api_key = ui.prompt_secret(&format!("{} API key: ", provider.label))?;
    if api_key.trim().is_empty() {
        return Err("No API key provided.".into());
    }
    save_env_value(context.env_path(), api_key_env_var, api_key.trim())?;
    ui.line(&format!("Saved {}", api_key_env_var))?;
    Ok(())
}

fn prompt_model_base_url(
    ui: &mut dyn SetupUi,
    provider: &NativeModelProvider,
    current: &str,
) -> Result<String, Box<dyn Error>> {
    let fallback = if current.trim().is_empty() {
        provider.base_url
    } else {
        current
    };
    loop {
        let value = prompt_with_default(ui, "Base URL", fallback)?;
        if value.trim().is_empty() {
            return Err("Base URL must not be empty.".into());
        }
        if !looks_like_http_url(&value) {
            ui.line("Base URL must start with http:// or https://")?;
            continue;
        }
        return Ok(value.trim_end_matches('/').to_string());
    }
}

fn prompt_model_name(
    ui: &mut dyn SetupUi,
    provider: &str,
    current: &str,
) -> Result<String, Box<dyn Error>> {
    let suggested = if current.trim().is_empty() {
        default_model_for_provider(provider)
    } else {
        current.to_string()
    };
    loop {
        let value = if suggested.trim().is_empty() {
            ui.prompt("Default model: ")?
        } else {
            prompt_with_default(ui, "Default model", &suggested)?
        };
        if !value.trim().is_empty() {
            return Ok(value.trim().to_string());
        }
        ui.line("Model name must not be empty.")?;
    }
}

fn default_model_for_provider(provider: &str) -> String {
    match provider {
        "openrouter" => "openai/gpt-5.4".to_string(),
        "nous" => "moonshotai/kimi-k2.6".to_string(),
        "openai" => "gpt-5.4".to_string(),
        "openai-codex" => "gpt-5.5".to_string(),
        "anthropic" => "claude-sonnet-4.6".to_string(),
        "bedrock" => "anthropic.claude-sonnet-4-6-20250514-v1:0".to_string(),
        "copilot" => "gpt-5.4".to_string(),
        "deepseek" => "deepseek-chat".to_string(),
        "gemini" => "gemini-2.5-flash".to_string(),
        "google-gemini-cli" => "gemini-3.1-pro-preview".to_string(),
        "xai" => "grok-4-fast-reasoning".to_string(),
        "zai" => "glm-5".to_string(),
        "minimax" | "minimax-cn" => "MiniMax-M2.7".to_string(),
        "minimax-oauth" => "MiniMax-M2.7".to_string(),
        "qwen-oauth" => "qwen3-coder-plus".to_string(),
        "ai-gateway" => "anthropic/claude-sonnet-4.6".to_string(),
        "kimi-coding" | "kimi-coding-cn" => "kimi-k2.6".to_string(),
        "alibaba" => "qwen-plus".to_string(),
        "alibaba-coding-plan" => "qwen-plus".to_string(),
        "arcee" => "trinity-mini".to_string(),
        "gmi" => "gpt-4.1-mini".to_string(),
        "huggingface" => "Qwen/Qwen3.5-397B-A17B".to_string(),
        "kilocode" => "openai/gpt-5.4".to_string(),
        "lmstudio" => "local-model".to_string(),
        "nvidia" => "meta/llama-3.1-70b-instruct".to_string(),
        "ollama-cloud" => "deepseek-r1".to_string(),
        "opencode-go" => "kimi-k2.6".to_string(),
        "opencode-zen" => "gpt-5.4".to_string(),
        "stepfun" => "step-3.5-flash".to_string(),
        "tencent-tokenhub" => "deepseek-v3.1".to_string(),
        "xiaomi" => "mimo-v2.5".to_string(),
        _ => String::new(),
    }
}

fn apply_model_provider_choice(
    context: &HermesContext,
    root: &mut Mapping,
    provider: &NativeModelProvider,
    model_name: &str,
    base_url: &str,
) -> Result<(), Box<dyn Error>> {
    let normalized_model = normalize_model_for_provider(model_name, &provider.name);
    let model = ensure_mapping(root, "model");
    model.insert(yaml_key("default"), Value::String(normalized_model));
    model.insert(yaml_key("provider"), Value::String(provider.name.clone()));
    model.insert(yaml_key("base_url"), Value::String(base_url.to_string()));
    if let Some(api_mode) = resolve_provider_api_mode(&provider.name, model_name) {
        model.insert(yaml_key("api_mode"), Value::String(api_mode.to_string()));
    } else if !provider.api_mode.trim().is_empty() {
        model.insert(
            yaml_key("api_mode"),
            Value::String(provider.api_mode.to_string()),
        );
    } else {
        model.remove(yaml_key("api_mode"));
    }

    if let Some(base_url_env_var) = provider.base_url_env_var.as_deref() {
        save_env_value(context.env_path(), base_url_env_var, base_url)?;
    }
    if provider.name != "custom" {
        let _ = remove_env_key(&context.env_path(), "OPENAI_BASE_URL")?;
    }
    clear_active_provider_marker(context.hermes_home().as_path())?;
    write_yaml_mapping(&context.config_path(), root)?;
    Ok(())
}

fn apply_saved_custom_model_provider_choice(
    context: &HermesContext,
    root: &mut Mapping,
    provider: &SavedCustomModelProvider,
    model_name: &str,
) -> Result<(), Box<dyn Error>> {
    let model = ensure_mapping(root, "model");
    model.insert(
        yaml_key("default"),
        Value::String(normalize_model_for_provider(model_name, "custom")),
    );
    model.insert(yaml_key("provider"), Value::String(String::from("custom")));
    model.insert(
        yaml_key("base_url"),
        Value::String(provider.base_url.trim_end_matches('/').to_string()),
    );
    if let Some(api_key) = provider.api_key_config_value.as_deref() {
        model.insert(yaml_key("api_key"), Value::String(api_key.to_string()));
    } else {
        model.remove(yaml_key("api_key"));
    }
    if let Some(api_mode) = provider.api_mode.as_deref() {
        model.insert(yaml_key("api_mode"), Value::String(api_mode.to_string()));
    } else {
        model.remove(yaml_key("api_mode"));
    }

    persist_saved_custom_provider_model(root, &provider.source, model_name);
    clear_active_provider_marker(context.hermes_home().as_path())?;
    write_yaml_mapping(&context.config_path(), root)?;
    Ok(())
}

fn persist_saved_custom_provider_model(
    root: &mut Mapping,
    source: &SavedCustomModelProviderSource,
    model_name: &str,
) {
    match source {
        SavedCustomModelProviderSource::LegacyCustomProviders { index } => {
            let Some(entries) = root
                .get_mut(yaml_key("custom_providers"))
                .and_then(Value::as_sequence_mut)
            else {
                return;
            };
            let Some(entry) = entries.get_mut(*index).and_then(Value::as_mapping_mut) else {
                return;
            };
            entry.insert(yaml_key("model"), Value::String(model_name.to_string()));
        }
        SavedCustomModelProviderSource::ProvidersMap {
            key,
            model_field_key,
        } => {
            let Some(entries) = root
                .get_mut(yaml_key("providers"))
                .and_then(Value::as_mapping_mut)
            else {
                return;
            };
            let Some(entry) = entries
                .get_mut(yaml_key(key))
                .and_then(Value::as_mapping_mut)
            else {
                return;
            };
            entry.insert(
                yaml_key(model_field_key),
                Value::String(model_name.to_string()),
            );
        }
    }
}

fn clear_active_provider_marker(hermes_home: &Path) -> Result<(), Box<dyn Error>> {
    let auth_path = hermes_home.join("auth.json");
    if !auth_path.exists() {
        return Ok(());
    }
    let mut payload = serde_json::from_str::<JsonValue>(&fs::read_to_string(&auth_path)?)?;
    let Some(root) = payload.as_object_mut() else {
        return Ok(());
    };
    if root.get("active_provider").is_some() {
        root.insert("active_provider".to_string(), JsonValue::Null);
        fs::write(&auth_path, serde_json::to_string_pretty(&payload)?)?;
    }
    Ok(())
}

fn run_native_tts_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let managed_available = nous_auth_present(context);
    let current_provider =
        get_nested_string(&root, &["tts", "provider"]).unwrap_or_else(|| "edge".to_string());
    let current_use_gateway = root
        .get(yaml_key("tts"))
        .and_then(Value::as_mapping)
        .and_then(|tts| mapping_bool(tts, "use_gateway"))
        .unwrap_or(false);

    let mut providers = Vec::new();
    if managed_available {
        providers.push(("nous_managed", "Nous Subscription"));
    }
    providers.extend([
        ("edge", "Edge TTS"),
        ("elevenlabs", "ElevenLabs"),
        ("openai", "OpenAI TTS"),
        ("xai", "xAI TTS"),
        ("minimax", "MiniMax TTS"),
        ("mistral", "Mistral Voxtral TTS"),
        ("gemini", "Google Gemini TTS"),
        ("neutts", "NeuTTS"),
        ("kittentts", "KittenTTS"),
        ("piper", "Piper"),
    ]);
    let current_label = if current_use_gateway && current_provider == "openai" && managed_available
    {
        "Nous Subscription"
    } else {
        provider_label(&current_provider, &providers)
    };

    ui.blank()?;
    ui.line("⚕ Hermes Setup — Text-to-Speech")?;
    ui.line(&format!("Current: {current_label}"))?;
    ui.blank()?;
    for (index, (_, label)) in providers.iter().enumerate() {
        ui.line(&format!("  {}. {}", index + 1, label))?;
    }
    ui.line(&format!(
        "  {}. Keep current ({})",
        providers.len() + 1,
        current_label
    ))?;

    let selection = prompt_menu_choice(
        ui,
        "Select TTS provider: ",
        providers.len() + 1,
        providers.len() + 1,
    )?;
    if selection == providers.len() + 1 {
        ui.line(&format!("Keeping current TTS provider: {current_label}"))?;
        return Ok(());
    }

    let mut selected = providers[selection - 1].0.to_string();
    let mut resolved_provider = selected.clone();
    let mut use_gateway = false;
    match selected.as_str() {
        "nous_managed" => {
            resolved_provider = "openai".to_string();
            use_gateway = true;
            ui.line(
                "TTS requests will use the managed Nous gateway and bill to your subscription.",
            )?;
        }
        "elevenlabs" => {
            if env_value("ELEVENLABS_API_KEY").is_none() {
                let api_key = ui.prompt_secret("ElevenLabs API key: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                    resolved_provider = selected.clone();
                } else {
                    save_env_value(context.env_path(), "ELEVENLABS_API_KEY", api_key.trim())?;
                    ui.line("ElevenLabs API key saved")?;
                }
            }
        }
        "openai" => {
            if env_value("VOICE_TOOLS_OPENAI_KEY").is_none()
                && env_value("OPENAI_API_KEY").is_none()
            {
                let api_key = ui.prompt_secret("OpenAI API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                    resolved_provider = selected.clone();
                } else {
                    save_env_value(context.env_path(), "VOICE_TOOLS_OPENAI_KEY", api_key.trim())?;
                    ui.line("OpenAI TTS API key saved")?;
                }
            }
        }
        "xai" => {
            if env_value("XAI_API_KEY").is_none() {
                let api_key = ui.prompt_secret("xAI API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                    resolved_provider = selected.clone();
                } else {
                    save_env_value(context.env_path(), "XAI_API_KEY", api_key.trim())?;
                    ui.line("xAI TTS API key saved")?;
                }
            }
            if selected == "xai" {
                let voice_id = ui.prompt("xAI voice_id (Enter for default eve): ")?;
                if !voice_id.trim().is_empty() {
                    ensure_mapping(ensure_mapping(&mut root, "tts"), "xai").insert(
                        yaml_key("voice_id"),
                        Value::String(voice_id.trim().to_string()),
                    );
                    ui.line(&format!("xAI voice_id set to: {}", voice_id.trim()))?;
                }
            }
        }
        "minimax" => {
            if env_value("MINIMAX_API_KEY").is_none() {
                let api_key = ui.prompt_secret("MiniMax API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                    resolved_provider = selected.clone();
                } else {
                    save_env_value(context.env_path(), "MINIMAX_API_KEY", api_key.trim())?;
                    ui.line("MiniMax TTS API key saved")?;
                }
            }
        }
        "mistral" => {
            if env_value("MISTRAL_API_KEY").is_none() {
                let api_key = ui.prompt_secret("Mistral API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                    resolved_provider = selected.clone();
                } else {
                    save_env_value(context.env_path(), "MISTRAL_API_KEY", api_key.trim())?;
                    ui.line("Mistral TTS API key saved")?;
                }
            }
        }
        "gemini" => {
            if env_value("GEMINI_API_KEY").is_none() && env_value("GOOGLE_API_KEY").is_none() {
                ui.line("Get a free API key at https://aistudio.google.com/app/apikey")?;
                let api_key = ui.prompt_secret("Gemini API key for TTS: ")?;
                if api_key.trim().is_empty() {
                    ui.line("No API key provided. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                    resolved_provider = selected.clone();
                } else {
                    save_env_value(context.env_path(), "GEMINI_API_KEY", api_key.trim())?;
                    ui.line("Gemini TTS API key saved")?;
                }
            }
        }
        "neutts" => {
            if !python_module_installed("neutts")? {
                ui.line("NeuTTS requires a Python package and espeak-ng.")?;
                if prompt_yes_no(ui, "Install NeuTTS dependencies now? [Y/n]: ", true)? {
                    if !install_neutts_deps(ui)? {
                        ui.line("NeuTTS installation incomplete. Falling back to Edge TTS.")?;
                        selected = "edge".to_string();
                        resolved_provider = selected.clone();
                    }
                } else {
                    ui.line("Skipping install. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                    resolved_provider = selected.clone();
                }
            } else {
                ui.line("NeuTTS is already installed")?;
            }
        }
        "kittentts" => {
            if !python_module_installed("kittentts")? {
                ui.line("KittenTTS is lightweight and requires no API key.")?;
                if prompt_yes_no(ui, "Install KittenTTS now? [Y/n]: ", true)? {
                    if !install_kittentts_deps(ui)? {
                        ui.line("KittenTTS installation incomplete. Falling back to Edge TTS.")?;
                        selected = "edge".to_string();
                        resolved_provider = selected.clone();
                    }
                } else {
                    ui.line("Skipping install. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                    resolved_provider = selected.clone();
                }
            } else {
                ui.line("KittenTTS is already installed")?;
            }
        }
        "piper" => {
            if !python_module_installed("piper")? {
                ui.line("Piper is local neural TTS with voices downloaded on first use.")?;
                if prompt_yes_no(ui, "Install Piper now? [Y/n]: ", true)? {
                    if !install_piper_deps(ui)? {
                        ui.line("Piper installation incomplete. Falling back to Edge TTS.")?;
                        selected = "edge".to_string();
                        resolved_provider = selected.clone();
                    }
                } else {
                    ui.line("Skipping install. Falling back to Edge TTS.")?;
                    selected = "edge".to_string();
                    resolved_provider = selected.clone();
                }
            } else {
                ui.line("Piper is already installed")?;
            }
        }
        _ => {}
    }

    let tts = ensure_mapping(&mut root, "tts");
    tts.insert(
        yaml_key("provider"),
        Value::String(resolved_provider.clone()),
    );
    tts.insert(yaml_key("use_gateway"), Value::Bool(use_gateway));
    write_yaml_mapping(&context.config_path(), &root)?;
    ui.line(&format!(
        "TTS provider set to: {}",
        provider_label(&selected, &providers)
    ))?;
    Ok(())
}

pub(crate) fn run_native_tts_setup_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut ui = StreamUi { input, output };
    run_native_tts_setup(context, &mut ui)?;
    Ok(())
}

fn run_native_terminal_setup(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let current_backend =
        get_nested_string(&root, &["terminal", "backend"]).unwrap_or_else(|| "local".to_string());

    let mut choices = vec![
        ("local", "Local - run directly on this machine (default)"),
        (
            "docker",
            "Docker - isolated container with configurable resources",
        ),
        ("modal", "Modal - serverless cloud sandbox"),
        ("ssh", "SSH - run on a remote machine"),
        (
            "daytona",
            "Daytona - persistent cloud development environment",
        ),
        (
            "vercel_sandbox",
            "Vercel Sandbox - cloud microVM with snapshot filesystem persistence",
        ),
    ];
    if cfg!(target_os = "linux") {
        choices.push((
            "singularity",
            "Singularity/Apptainer - HPC-friendly container",
        ));
    }

    ui.blank()?;
    ui.line("⚕ Hermes Setup — Terminal Backend")?;
    ui.line("Choose where Hermes runs shell commands and code.")?;
    ui.blank()?;
    for (index, (_, label)) in choices.iter().enumerate() {
        ui.line(&format!("  {}. {}", index + 1, label))?;
    }
    ui.line(&format!(
        "  {}. Keep current ({})",
        choices.len() + 1,
        current_backend
    ))?;

    let selected_index = prompt_menu_choice(
        ui,
        "Select terminal backend: ",
        choices.len() + 1,
        choices.len() + 1,
    )?;
    if selected_index == choices.len() + 1 {
        ui.line(&format!("Keeping current backend: {current_backend}"))?;
        return Ok(());
    }

    let selected_backend = choices[selected_index - 1].0;
    let terminal = ensure_mapping(&mut root, "terminal");
    terminal.insert(
        yaml_key("backend"),
        Value::String(selected_backend.to_string()),
    );

    match selected_backend {
        "local" => configure_local_terminal(context, ui, terminal)?,
        "docker" => configure_docker_terminal(context, ui, terminal)?,
        "modal" => configure_modal_terminal(context, ui, terminal)?,
        "singularity" => configure_singularity_terminal(context, ui, terminal)?,
        "ssh" => configure_ssh_terminal(context, ui)?,
        "daytona" => configure_daytona_terminal(context, ui, terminal)?,
        "vercel_sandbox" => configure_vercel_terminal(ui, terminal)?,
        _ => unreachable!(),
    }

    save_env_value(context.env_path(), "TERMINAL_ENV", selected_backend)?;
    if selected_backend == "vercel_sandbox" {
        if let Some(runtime) = mapping_string(terminal, "vercel_runtime") {
            save_env_value(context.env_path(), "TERMINAL_VERCEL_RUNTIME", &runtime)?;
        }
    }
    if selected_backend == "daytona" {
        if let Some(image) = mapping_string(terminal, "daytona_image") {
            save_env_value(context.env_path(), "TERMINAL_DAYTONA_IMAGE", &image)?;
        }
    }
    if selected_backend == "modal" {
        if let Some(mode) = mapping_string(terminal, "modal_mode") {
            save_env_value(context.env_path(), "TERMINAL_MODAL_MODE", &mode)?;
        }
    }
    write_yaml_mapping(&context.config_path(), &root)?;
    ui.blank()?;
    ui.line(&format!("Terminal backend set to: {selected_backend}"))?;
    Ok(())
}

fn configure_local_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: Local")?;
    let current_cwd =
        mapping_string(terminal, "cwd").unwrap_or_else(|| context.home_dir().display().to_string());
    let cwd = ui.prompt(&format!("  Gateway working directory [{}]: ", current_cwd))?;
    if !cwd.trim().is_empty() {
        terminal.insert(yaml_key("cwd"), Value::String(cwd.trim().to_string()));
    }

    if env_value("SUDO_PASSWORD").is_some() {
        ui.line("Sudo password: configured")?;
    } else if prompt_yes_no(
        ui,
        "Enable sudo support? (stores password for apt install, etc.) [y/N]: ",
        false,
    )? {
        let sudo_password = ui.prompt_secret("  Sudo password: ")?;
        if !sudo_password.trim().is_empty() {
            save_env_value(context.env_path(), "SUDO_PASSWORD", sudo_password.trim())?;
            ui.line("Sudo password saved")?;
        }
    }
    Ok(())
}

fn configure_docker_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: Docker")?;
    if !command_exists("docker") {
        ui.line("Docker not found in PATH. Install Docker: https://docs.docker.com/get-docker/")?;
    }
    let current_image = mapping_string(terminal, "docker_image")
        .unwrap_or_else(|| "nikolaik/python-nodejs:python3.11-nodejs20".to_string());
    let image = prompt_with_default(ui, "  Docker image", &current_image)?;
    terminal.insert(yaml_key("docker_image"), Value::String(image.clone()));
    save_env_value(context.env_path(), "TERMINAL_DOCKER_IMAGE", &image)?;
    prompt_container_resources(ui, terminal)
}

fn configure_modal_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: Modal")?;
    ui.line("Serverless cloud sandboxes. Each session gets its own container.")?;

    let current_mode = normalize_modal_mode(
        mapping_string(terminal, "modal_mode")
            .or_else(|| env_value("TERMINAL_MODAL_MODE"))
            .as_deref(),
    );
    let managed_available = managed_modal_available(context);
    let use_managed = if managed_available {
        ui.line("Choose how Modal execution should be billed.")?;
        let default_choice = match current_mode.as_str() {
            "managed" => 1,
            "direct" => 2,
            _ => {
                if env_value("MODAL_TOKEN_ID").is_some() {
                    2
                } else {
                    1
                }
            }
        };
        prompt_menu_choice(
            ui,
            "Select how Modal execution should be billed: ",
            2,
            default_choice,
        )? == 1
    } else {
        false
    };

    if use_managed {
        terminal.insert(yaml_key("modal_mode"), Value::String("managed".to_string()));
        ui.line(
            "Modal execution will use the managed Nous gateway and bill to your subscription.",
        )?;
        if env_value("MODAL_TOKEN_ID").is_some() || env_value("MODAL_TOKEN_SECRET").is_some() {
            ui.line("Direct Modal credentials are still configured, but this backend is pinned to managed mode.")?;
        }
    } else {
        terminal.insert(yaml_key("modal_mode"), Value::String("direct".to_string()));
        ui.line("Requires a Modal account: https://modal.com")?;
        if !python_module_installed("modal")? {
            ui.line("modal SDK not detected; attempting install...")?;
            if install_python_package(&["modal"])? {
                ui.line("modal SDK installed")?;
            } else {
                ui.line("Install failed — run manually: pip install modal")?;
            }
        }

        ui.blank()?;
        ui.line("Modal authentication:")?;
        ui.line("  Get your token at: https://modal.com/settings")?;
        if env_value("MODAL_TOKEN_ID").is_some() {
            ui.line("  Modal token: already configured")?;
            if prompt_yes_no(ui, "  Update Modal credentials? [y/N]: ", false)? {
                let token_id = ui.prompt_secret("    Modal Token ID: ")?;
                let token_secret = ui.prompt_secret("    Modal Token Secret: ")?;
                if !token_id.trim().is_empty() {
                    save_env_value(context.env_path(), "MODAL_TOKEN_ID", token_id.trim())?;
                }
                if !token_secret.trim().is_empty() {
                    save_env_value(
                        context.env_path(),
                        "MODAL_TOKEN_SECRET",
                        token_secret.trim(),
                    )?;
                }
            }
        } else {
            let token_id = ui.prompt_secret("    Modal Token ID: ")?;
            let token_secret = ui.prompt_secret("    Modal Token Secret: ")?;
            if !token_id.trim().is_empty() {
                save_env_value(context.env_path(), "MODAL_TOKEN_ID", token_id.trim())?;
            }
            if !token_secret.trim().is_empty() {
                save_env_value(
                    context.env_path(),
                    "MODAL_TOKEN_SECRET",
                    token_secret.trim(),
                )?;
            }
        }
    }

    prompt_container_resources(ui, terminal)
}

fn configure_singularity_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: Singularity/Apptainer")?;
    if !command_exists("apptainer") && !command_exists("singularity") {
        ui.line(
            "Singularity/Apptainer not found in PATH. Install: https://apptainer.org/docs/admin/main/installation.html",
        )?;
    }
    let current_image = mapping_string(terminal, "singularity_image")
        .unwrap_or_else(|| "docker://nikolaik/python-nodejs:python3.11-nodejs20".to_string());
    let image = prompt_with_default(ui, "  Container image", &current_image)?;
    terminal.insert(yaml_key("singularity_image"), Value::String(image.clone()));
    save_env_value(context.env_path(), "TERMINAL_SINGULARITY_IMAGE", &image)?;
    prompt_container_resources(ui, terminal)
}

fn configure_ssh_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: SSH")?;
    let current_host = env_value("TERMINAL_SSH_HOST").unwrap_or_default();
    let host = prompt_with_default(ui, "  SSH host (hostname or IP)", &current_host)?;
    if !host.trim().is_empty() {
        save_env_value(context.env_path(), "TERMINAL_SSH_HOST", host.trim())?;
    }

    let default_user = std::env::var("USER").unwrap_or_default();
    let current_user = env_value("TERMINAL_SSH_USER").unwrap_or_else(|| default_user.clone());
    let user = prompt_with_default(ui, "  SSH user", &current_user)?;
    if !user.trim().is_empty() {
        save_env_value(context.env_path(), "TERMINAL_SSH_USER", user.trim())?;
    }

    let current_port = env_value("TERMINAL_SSH_PORT").unwrap_or_else(|| "22".to_string());
    let port = prompt_with_default(ui, "  SSH port", &current_port)?;
    if !port.trim().is_empty() && port.trim() != "22" {
        save_env_value(context.env_path(), "TERMINAL_SSH_PORT", port.trim())?;
    }

    let default_key = context.home_dir().join(".ssh").join("id_rsa");
    let current_key =
        env_value("TERMINAL_SSH_KEY").unwrap_or_else(|| default_key.display().to_string());
    let ssh_key = prompt_with_default(ui, "  SSH private key path", &current_key)?;
    if !ssh_key.trim().is_empty() {
        save_env_value(context.env_path(), "TERMINAL_SSH_KEY", ssh_key.trim())?;
    }

    if !host.trim().is_empty() && prompt_yes_no(ui, "  Test SSH connection? [Y/n]: ", true)? {
        let mut ssh = Command::new("ssh");
        ssh.args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]);
        if !ssh_key.trim().is_empty() {
            ssh.args(["-i", ssh_key.trim()]);
        }
        if !port.trim().is_empty() && port.trim() != "22" {
            ssh.args(["-p", port.trim()]);
        }
        let destination = if !user.trim().is_empty() {
            format!("{}@{}", user.trim(), host.trim())
        } else {
            host.trim().to_string()
        };
        ssh.arg(destination).arg("echo ok");
        match ssh.output() {
            Ok(output) if output.status.success() => ui.line("  SSH connection successful!")?,
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let detail = stderr.trim();
                if detail.is_empty() {
                    ui.line("  SSH connection failed.")?;
                } else {
                    ui.line(&format!("  SSH connection failed: {detail}"))?;
                }
            }
            Err(error) => ui.line(&format!("  SSH connection failed: {error}"))?,
        }
    }

    Ok(())
}

fn configure_vercel_terminal(
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    const SUPPORTED_RUNTIMES: &[&str] = &["node24", "node22", "python3.13"];

    ui.line("Terminal backend: Vercel Sandbox")?;
    ui.line("Cloud microVM sandboxes with snapshot-backed filesystem persistence.")?;
    ui.line("Requires the optional SDK: pip install 'hermes-agent[vercel]'")?;
    if !python_module_installed("vercel")? {
        ui.line("vercel SDK not detected; install it before using this backend.")?;
    }

    let current_runtime =
        mapping_string(terminal, "vercel_runtime").unwrap_or_else(|| "node24".to_string());
    let runtime = prompt_enum_choice(ui, "  Runtime", SUPPORTED_RUNTIMES, &current_runtime)?;
    terminal.insert(yaml_key("vercel_runtime"), Value::String(runtime.clone()));

    let current_persist = mapping_bool(terminal, "container_persistent").unwrap_or(true);
    let persist_default = if current_persist { "yes" } else { "no" };
    let persist = prompt_yes_no(
        ui,
        &format!("  Persist filesystem with snapshots? (yes/no) [{persist_default}]: "),
        current_persist,
    )?;
    terminal.insert(yaml_key("container_persistent"), Value::Bool(persist));

    let current_cpu = mapping_f64(terminal, "container_cpu").unwrap_or(1.0);
    let cpu = prompt_f64_range(ui, "  CPU cores", current_cpu, 0.1, 512.0)?;
    terminal.insert(yaml_key("container_cpu"), serde_yaml::to_value(cpu)?);

    let current_memory = mapping_i64(terminal, "container_memory").unwrap_or(5120);
    let memory = prompt_positive_i64(
        ui,
        "  Memory in MB (5120 = 5GB)",
        current_memory,
        "Enter a positive integer.",
    )?;
    terminal.insert(yaml_key("container_memory"), serde_yaml::to_value(memory)?);

    terminal.insert(yaml_key("container_disk"), serde_yaml::to_value(51200_i64)?);
    Ok(())
}

fn configure_daytona_terminal(
    context: &HermesContext,
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    ui.line("Terminal backend: Daytona")?;
    ui.line("Persistent cloud development environments.")?;
    ui.line("Sign up at: https://daytona.io")?;
    if !python_module_installed("daytona")? {
        ui.line("daytona SDK not detected; attempting install...")?;
        if install_python_package(&["daytona"])? {
            ui.line("daytona SDK installed")?;
        } else {
            ui.line("Install failed — run manually: pip install daytona")?;
        }
    }

    let existing_key = env_value("DAYTONA_API_KEY");
    if let Some(current) = existing_key {
        ui.line("  Daytona API key: already configured")?;
        if prompt_yes_no(ui, "  Update API key? [y/N]: ", false)? {
            let api_key = ui.prompt_secret("    Daytona API key: ")?;
            if !api_key.trim().is_empty() {
                save_env_value(context.env_path(), "DAYTONA_API_KEY", api_key.trim())?;
                ui.line("    Updated")?;
            } else {
                let _ = current;
            }
        }
    } else {
        let api_key = ui.prompt_secret("    Daytona API key: ")?;
        if !api_key.trim().is_empty() {
            save_env_value(context.env_path(), "DAYTONA_API_KEY", api_key.trim())?;
            ui.line("    Configured")?;
        }
    }

    let current_image = mapping_string(terminal, "daytona_image")
        .unwrap_or_else(|| "nikolaik/python-nodejs:python3.11-nodejs20".to_string());
    let image = prompt_with_default(ui, "  Sandbox image", &current_image)?;
    terminal.insert(yaml_key("daytona_image"), Value::String(image));
    prompt_container_resources(ui, terminal)
}

fn prompt_container_resources(
    ui: &mut dyn SetupUi,
    terminal: &mut Mapping,
) -> Result<(), Box<dyn Error>> {
    let current_persist = mapping_bool(terminal, "container_persistent").unwrap_or(true);
    let persist_default = if current_persist { "yes" } else { "no" };
    let persist = prompt_yes_no(
        ui,
        &format!("  Persist filesystem across sessions? (yes/no) [{persist_default}]: "),
        current_persist,
    )?;
    terminal.insert(yaml_key("container_persistent"), Value::Bool(persist));

    let current_cpu = mapping_f64(terminal, "container_cpu").unwrap_or(1.0);
    let cpu = prompt_f64_range(ui, "  CPU cores", current_cpu, 0.1, 512.0)?;
    terminal.insert(yaml_key("container_cpu"), serde_yaml::to_value(cpu)?);

    let current_memory = mapping_i64(terminal, "container_memory").unwrap_or(5120);
    let memory = prompt_positive_i64(
        ui,
        "  Memory in MB (5120 = 5GB)",
        current_memory,
        "Enter a positive integer.",
    )?;
    terminal.insert(yaml_key("container_memory"), serde_yaml::to_value(memory)?);

    let current_disk = mapping_i64(terminal, "container_disk").unwrap_or(51200);
    let disk = prompt_positive_i64(
        ui,
        "  Disk in MB (51200 = 50GB)",
        current_disk,
        "Enter a positive integer.",
    )?;
    terminal.insert(yaml_key("container_disk"), serde_yaml::to_value(disk)?);
    Ok(())
}

fn prompt_with_default(
    ui: &mut dyn SetupUi,
    label: &str,
    current: &str,
) -> Result<String, Box<dyn Error>> {
    let input = ui.prompt(&format!("{label} [{current}]: "))?;
    if input.trim().is_empty() {
        return Ok(current.to_string());
    }
    Ok(input.trim().to_string())
}

fn mapping_string(mapping: &Mapping, key: &str) -> Option<String> {
    mapping
        .get(yaml_key(key))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn mapping_string_alias(mapping: &Mapping, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        mapping_string(mapping, key)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn mapping_bool(mapping: &Mapping, key: &str) -> Option<bool> {
    mapping.get(yaml_key(key)).and_then(Value::as_bool)
}

fn mapping_i64(mapping: &Mapping, key: &str) -> Option<i64> {
    mapping.get(yaml_key(key)).and_then(Value::as_i64)
}

fn mapping_f64(mapping: &Mapping, key: &str) -> Option<f64> {
    mapping.get(yaml_key(key)).and_then(Value::as_f64)
}

fn provider_label<'a>(provider: &str, providers: &'a [(&str, &'a str)]) -> &'a str {
    providers
        .iter()
        .find_map(|(key, label)| (*key == provider).then_some(*label))
        .unwrap_or("Custom")
}

fn prompt_yes_no(
    ui: &mut dyn SetupUi,
    prompt: &str,
    default_yes: bool,
) -> Result<bool, Box<dyn Error>> {
    loop {
        let input = ui.prompt(prompt)?;
        let normalized = input.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Ok(default_yes);
        }
        match normalized.as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => ui.line("Please answer yes or no.")?,
        }
    }
}

fn env_value_for_context(context: &HermesContext, key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| read_env_file_value(&context.env_path(), key))
}

fn read_env_file_value(path: &PathBuf, key: &str) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((entry_key, entry_value)) = trimmed.split_once('=') else {
            continue;
        };
        if entry_key.trim() == key {
            let value = entry_value.trim();
            if value.is_empty() {
                return None;
            }
            return Some(value.to_string());
        }
    }
    None
}

fn env_value(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn looks_like_http_url(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    normalized.starts_with("http://") || normalized.starts_with("https://")
}

fn nous_auth_present(context: &HermesContext) -> bool {
    get_auth_status_summary(&context.hermes_home(), "nous")
        .map(|status| status.logged_in)
        .unwrap_or(false)
}

fn normalize_modal_mode(value: Option<&str>) -> String {
    let normalized = value.unwrap_or("auto").trim().to_ascii_lowercase();
    match normalized.as_str() {
        "direct" | "managed" | "auto" => normalized,
        _ => "auto".to_string(),
    }
}

fn managed_modal_available(context: &HermesContext) -> bool {
    if !nous_auth_present(context) {
        return false;
    }
    if env_value("TOOL_GATEWAY_USER_TOKEN").is_some() {
        return true;
    }

    let path = context.hermes_home().join("auth.json");
    let Ok(payload) = fs::read_to_string(path) else {
        return false;
    };
    serde_json::from_str::<JsonValue>(&payload)
        .ok()
        .and_then(|json| {
            json.get("providers")
                .and_then(|providers| providers.get("nous"))
                .and_then(|provider| provider.get("access_token"))
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|token| !token.is_empty())
                .map(|_| true)
        })
        .unwrap_or(false)
}

fn setup_python_interpreter() -> Option<PathBuf> {
    resolve_repo_python(&project_root(), Some("HERMES_SETUP_PYTHON"))
}

fn python_module_installed(module: &str) -> Result<bool, Box<dyn Error>> {
    let Some(python) = setup_python_interpreter() else {
        return Ok(false);
    };
    let status = Command::new(python)
        .arg("-c")
        .arg(format!(
            "import importlib.util, sys; raise SystemExit(0 if importlib.util.find_spec({module:?}) else 1)"
        ))
        .status()?;
    Ok(status.success())
}

fn install_python_package(packages: &[&str]) -> Result<bool, Box<dyn Error>> {
    let Some(python) = setup_python_interpreter() else {
        return Ok(false);
    };
    let mut command = Command::new(python);
    command.arg("-m").arg("pip").arg("install").arg("-U");
    for package in packages {
        command.arg(package);
    }
    Ok(command.status()?.success())
}

fn install_neutts_deps(ui: &mut dyn SetupUi) -> Result<bool, Box<dyn Error>> {
    if !command_exists("espeak-ng") {
        ui.line("NeuTTS requires espeak-ng for phonemization.")?;
        ui.line(&format!(
            "Install with: {}",
            if cfg!(target_os = "macos") {
                "brew install espeak-ng"
            } else if cfg!(target_os = "windows") {
                "choco install espeak-ng"
            } else {
                "sudo apt install espeak-ng"
            }
        ))?;
        if !prompt_yes_no(ui, "Install espeak-ng now? [Y/n]: ", true)? {
            return Ok(false);
        }
        let mut install = if cfg!(target_os = "macos") {
            let mut cmd = Command::new("brew");
            cmd.args(["install", "espeak-ng"]);
            cmd
        } else if cfg!(target_os = "windows") {
            let mut cmd = Command::new("choco");
            cmd.args(["install", "espeak-ng", "-y"]);
            cmd
        } else {
            let mut cmd = Command::new("sudo");
            cmd.args(["apt", "install", "-y", "espeak-ng"]);
            cmd
        };
        if !install.status().ok().is_some_and(|status| status.success()) {
            return Ok(false);
        }
        ui.line("espeak-ng installed")?;
    }

    let Some(python) = setup_python_interpreter() else {
        return Ok(false);
    };
    ui.line("Installing neutts Python package...")?;
    let status = Command::new(python)
        .args(["-m", "pip", "install", "-U", "neutts[all]", "--quiet"])
        .status()?;
    Ok(status.success())
}

fn install_kittentts_deps(ui: &mut dyn SetupUi) -> Result<bool, Box<dyn Error>> {
    let Some(python) = setup_python_interpreter() else {
        return Ok(false);
    };
    let wheel_url = "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl";
    ui.line("Installing kittentts Python package...")?;
    let status = Command::new(python)
        .args([
            "-m",
            "pip",
            "install",
            "-U",
            wheel_url,
            "soundfile",
            "--quiet",
        ])
        .status()?;
    Ok(status.success())
}

fn install_piper_deps(ui: &mut dyn SetupUi) -> Result<bool, Box<dyn Error>> {
    ui.line("Installing piper-tts Python package...")?;
    let success = install_python_package(&["piper-tts", "--quiet"])?;
    if success {
        ui.line("Piper installed. Voices download on first use.")?;
    }
    Ok(success)
}

fn command_exists(command: &str) -> bool {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .map(|path| path.join(command))
        .any(|candidate| candidate.is_file())
}

fn prompt_positive_i64(
    ui: &mut dyn SetupUi,
    label: &str,
    current: i64,
    error_message: &str,
) -> Result<i64, Box<dyn Error>> {
    loop {
        let input = ui.prompt(&format!("{label} [{current}]: "))?;
        if input.is_empty() {
            return Ok(current);
        }
        match input.parse::<i64>() {
            Ok(value) if value > 0 => return Ok(value),
            _ => ui.line(error_message)?,
        }
    }
}

fn prompt_i64_range(
    ui: &mut dyn SetupUi,
    label: &str,
    current: i64,
    min: i64,
    max: i64,
) -> Result<i64, Box<dyn Error>> {
    loop {
        let input = ui.prompt(&format!("{label} [{current}]: "))?;
        if input.is_empty() {
            return Ok(current);
        }
        match input.parse::<i64>() {
            Ok(value) if value >= min && value <= max => return Ok(value),
            _ => ui.line(&format!("Enter a whole number between {min} and {max}."))?,
        }
    }
}

fn prompt_f64_range(
    ui: &mut dyn SetupUi,
    label: &str,
    current: f64,
    min: f64,
    max: f64,
) -> Result<f64, Box<dyn Error>> {
    loop {
        let input = ui.prompt(&format!("{label} [{current}]: "))?;
        if input.is_empty() {
            return Ok(current);
        }
        match input.parse::<f64>() {
            Ok(value) if value >= min && value <= max => return Ok(value),
            _ => ui.line(&format!("Enter a number between {min} and {max}."))?,
        }
    }
}

fn prompt_enum_choice(
    ui: &mut dyn SetupUi,
    label: &str,
    options: &[&str],
    current: &str,
) -> Result<String, Box<dyn Error>> {
    loop {
        let input = ui.prompt(&format!("{label} [{current}]: "))?;
        if input.is_empty() {
            return Ok(current.to_string());
        }
        let normalized = input.to_ascii_lowercase();
        if options.iter().any(|option| *option == normalized) {
            return Ok(normalized);
        }
        ui.line(&format!("Choose one of: {}", options.join(", ")))?;
    }
}

fn prompt_menu_choice(
    ui: &mut dyn SetupUi,
    prompt: &str,
    max: usize,
    default: usize,
) -> Result<usize, Box<dyn Error>> {
    loop {
        let input = ui.prompt(prompt)?;
        if input.is_empty() {
            return Ok(default);
        }
        match input.parse::<usize>() {
            Ok(value) if (1..=max).contains(&value) => return Ok(value),
            _ => ui.line(&format!("Enter a number between 1 and {max}."))?,
        }
    }
}

fn get_nested_i64(root: &Mapping, path: &[&str]) -> Option<i64> {
    get_nested_value(root, path).and_then(Value::as_i64)
}

fn get_nested_f64(root: &Mapping, path: &[&str]) -> Option<f64> {
    get_nested_value(root, path).and_then(Value::as_f64)
}

fn get_nested_string(root: &Mapping, path: &[&str]) -> Option<String> {
    get_nested_value(root, path)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn get_nested_value<'a>(root: &'a Mapping, path: &[&str]) -> Option<&'a Value> {
    let (last, parents) = path.split_last()?;
    let mut current = root;
    for segment in parents {
        current = current.get(yaml_key(segment))?.as_mapping()?;
    }
    current.get(yaml_key(last))
}

fn ensure_mapping<'a>(root: &'a mut Mapping, key: &str) -> &'a mut Mapping {
    let key_value = yaml_key(key).clone();
    let replace = !matches!(root.get(&key_value), Some(Value::Mapping(_)));
    if replace {
        root.insert(key_value.clone(), Value::Mapping(Mapping::new()));
    }
    match root.get_mut(&key_value) {
        Some(Value::Mapping(mapping)) => mapping,
        _ => unreachable!(),
    }
}

fn ensure_sequence<'a>(root: &'a mut Mapping, key: &str) -> &'a mut Vec<Value> {
    let key_value = yaml_key(key).clone();
    let replace = !matches!(root.get(&key_value), Some(Value::Sequence(_)));
    if replace {
        root.insert(key_value.clone(), Value::Sequence(Vec::new()));
    }
    match root.get_mut(&key_value) {
        Some(Value::Sequence(sequence)) => sequence,
        _ => unreachable!(),
    }
}

fn yaml_key(key: &str) -> Value {
    Value::String(key.to_string())
}

fn remove_env_key(path: &Path, key: &str) -> Result<bool, Box<dyn Error>> {
    if !path.exists() {
        return Ok(false);
    }
    let original = fs::read_to_string(path)?;
    let mut kept = Vec::new();
    let mut removed = false;
    for line in original.lines() {
        if line
            .strip_prefix(key)
            .is_some_and(|rest| rest.starts_with('='))
        {
            removed = true;
            continue;
        }
        kept.push(format!("{line}\n"));
    }
    if !removed {
        return Ok(false);
    }
    atomic_write(path, kept.concat().as_bytes())?;
    Ok(true)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".tmp-{unique}"));
    let tmp = std::path::PathBuf::from(tmp);
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::env;
    use std::fs;
    use std::io::Read;
    use std::net::TcpListener;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(test)]
    use std::sync::{Arc, Mutex, OnceLock};
    use std::thread;
    use tempfile::TempDir;

    #[derive(Parser, Debug)]
    struct SetupHarness {
        #[command(flatten)]
        args: SetupArgs,
    }

    #[cfg(test)]
    fn test_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(test)]
    fn set_env_var(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        unsafe {
            env::set_var(key, value);
        }
    }

    #[cfg(test)]
    fn remove_env_var(key: &str) {
        unsafe {
            env::remove_var(key);
        }
    }

    fn spawn_codex_oauth_server(
        access_token: &str,
        refresh_token: &str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        let access_token = access_token.to_string();
        let refresh_token = refresh_token.to_string();
        let handle = thread::spawn(move || {
            for idx in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 8192];
                let size = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]).to_string();
                requests_clone.lock().unwrap().push(request);
                let (status_line, payload) = match idx {
                    0 => (
                        "HTTP/1.1 200 OK",
                        serde_json::json!({
                            "user_code": "USER-CODE",
                            "device_auth_id": "device-auth-123",
                            "interval": 1
                        })
                        .to_string(),
                    ),
                    1 => (
                        "HTTP/1.1 200 OK",
                        serde_json::json!({
                            "authorization_code": "auth-code-123",
                            "code_verifier": "verifier-xyz"
                        })
                        .to_string(),
                    ),
                    _ => (
                        "HTTP/1.1 200 OK",
                        serde_json::json!({
                            "access_token": access_token,
                            "refresh_token": refresh_token
                        })
                        .to_string(),
                    ),
                };
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (base_url, requests, handle)
    }

    fn spawn_minimax_oauth_server(
        access_token: &str,
        refresh_token: &str,
        state: &str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        let access_token = access_token.to_string();
        let refresh_token = refresh_token.to_string();
        let state = state.to_string();
        let base_for_thread = base_url.clone();
        let handle = thread::spawn(move || {
            for idx in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 8192];
                let size = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]).to_string();
                requests_clone.lock().unwrap().push(request);
                let payload = match idx {
                    0 => serde_json::json!({
                        "user_code": "MINIMAX-CODE",
                        "verification_uri": format!("{base_for_thread}/verify"),
                        "expired_in": 60,
                        "interval": 100,
                        "state": state
                    }),
                    _ => serde_json::json!({
                        "status": "success",
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "expired_in": 3600,
                        "token_type": "Bearer",
                        "resource_url": "group-123",
                        "notification_message": "quota synced"
                    }),
                }
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (base_url, requests, handle)
    }

    fn spawn_nous_oauth_server(
        access_token: &str,
        refresh_token: &str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        let access_token = access_token.to_string();
        let refresh_token = refresh_token.to_string();
        let base_for_thread = base_url.clone();
        let handle = thread::spawn(move || {
            for idx in 0..3 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 8192];
                let size = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]).to_string();
                requests_clone.lock().unwrap().push(request.clone());
                let payload = match idx {
                    0 => serde_json::json!({
                        "device_code": "nous-device-code",
                        "user_code": "NOUS-CODE",
                        "verification_uri": format!("{base_for_thread}/verify"),
                        "verification_uri_complete": format!("{base_for_thread}/verify?code=NOUS-CODE"),
                        "expires_in": 60,
                        "interval": 1
                    }),
                    1 => serde_json::json!({
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "token_type": "Bearer",
                        "scope": "inference:mint_agent_key offline_access",
                        "expires_in": 3600,
                        "inference_base_url": format!("{base_for_thread}/portal-inference/v1")
                    }),
                    _ => serde_json::json!({
                        "api_key": "nous-agent-key",
                        "key_id": "agent-key-123",
                        "expires_at": "2999-01-02T00:00:00Z",
                        "expires_in": 86400,
                        "inference_base_url": format!("{base_for_thread}/runtime-inference/v1"),
                        "reused": false
                    }),
                }
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (base_url, requests, handle)
    }

    fn spawn_google_oauth_server(
        access_token: &str,
        refresh_token: &str,
        email: &str,
    ) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_clone = Arc::clone(&requests);
        let access_token = access_token.to_string();
        let refresh_token = refresh_token.to_string();
        let email = email.to_string();
        let handle = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 8192];
                let size = stream.read(&mut buffer).unwrap();
                let request = String::from_utf8_lossy(&buffer[..size]).to_string();
                let first_line = request.lines().next().unwrap_or_default().to_string();
                requests_clone.lock().unwrap().push(request.clone());
                let (status_line, payload) = if first_line.starts_with("POST /token ") {
                    (
                        "HTTP/1.1 200 OK",
                        serde_json::json!({
                            "access_token": access_token,
                            "refresh_token": refresh_token,
                            "expires_in": 3600,
                            "token_type": "Bearer"
                        })
                        .to_string(),
                    )
                } else if first_line.starts_with("GET /userinfo?alt=json ") {
                    (
                        "HTTP/1.1 200 OK",
                        serde_json::json!({
                            "email": email
                        })
                        .to_string(),
                    )
                } else {
                    (
                        "HTTP/1.1 404 Not Found",
                        serde_json::json!({
                            "error": "not_found"
                        })
                        .to_string(),
                    )
                };
                let response = format!(
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        (base_url, requests, handle)
    }

    #[derive(Default)]
    struct TestUi {
        answers: Vec<String>,
        cursor: usize,
        output: String,
    }

    impl TestUi {
        fn new(answers: &[&str]) -> Self {
            Self {
                answers: answers.iter().map(|value| (*value).to_string()).collect(),
                cursor: 0,
                output: String::new(),
            }
        }
    }

    impl SetupUi for TestUi {
        fn line(&mut self, text: &str) -> Result<(), Box<dyn Error>> {
            self.output.push_str(text);
            self.output.push('\n');
            Ok(())
        }

        fn prompt(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
            self.output.push_str(prompt);
            let answer = self
                .answers
                .get(self.cursor)
                .cloned()
                .ok_or("missing test answer")?;
            self.cursor += 1;
            Ok(answer)
        }

        fn prompt_secret(&mut self, prompt: &str) -> Result<String, Box<dyn Error>> {
            self.prompt(prompt)
        }
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
    fn direct_python_setup_section_only_allows_plain_model() {
        assert_eq!(
            direct_python_setup_section(&SetupArgs {
                section: Some(SetupSection::Model),
                non_interactive: false,
                reset: false,
                reconfigure: false,
                quick: false,
            }),
            Some(SetupSection::Model)
        );
        assert_eq!(
            direct_python_setup_section(&SetupArgs {
                section: Some(SetupSection::Gateway),
                non_interactive: false,
                reset: false,
                reconfigure: false,
                quick: false,
            }),
            None
        );
        assert_eq!(
            direct_python_setup_section(&SetupArgs {
                section: Some(SetupSection::Tools),
                non_interactive: false,
                reset: false,
                reconfigure: false,
                quick: false,
            }),
            None
        );
        assert_eq!(
            direct_python_setup_section(&SetupArgs {
                section: Some(SetupSection::Tools),
                non_interactive: false,
                reset: false,
                reconfigure: false,
                quick: true,
            }),
            None
        );
        assert_eq!(
            direct_python_setup_section(&SetupArgs {
                section: Some(SetupSection::Tts),
                non_interactive: false,
                reset: false,
                reconfigure: false,
                quick: false,
            }),
            None
        );
    }

    #[test]
    fn native_tools_setup_only_allows_plain_tools() {
        assert!(should_use_native_tools_setup(&SetupArgs {
            section: Some(SetupSection::Tools),
            non_interactive: false,
            reset: false,
            reconfigure: false,
            quick: false,
        }));
        assert!(!should_use_native_tools_setup(&SetupArgs {
            section: Some(SetupSection::Tools),
            non_interactive: false,
            reset: false,
            reconfigure: true,
            quick: false,
        }));
        assert!(!should_use_native_tools_setup(&SetupArgs {
            section: Some(SetupSection::Gateway),
            non_interactive: false,
            reset: false,
            reconfigure: false,
            quick: false,
        }));
    }

    #[test]
    fn native_model_setup_only_allows_plain_model() {
        assert!(should_use_native_model_setup(&SetupArgs {
            section: Some(SetupSection::Model),
            non_interactive: false,
            reset: false,
            reconfigure: false,
            quick: false,
        }));
        assert!(!should_use_native_model_setup(&SetupArgs {
            section: Some(SetupSection::Model),
            non_interactive: true,
            reset: false,
            reconfigure: false,
            quick: false,
        }));
        assert!(!should_use_native_model_setup(&SetupArgs {
            section: Some(SetupSection::Gateway),
            non_interactive: false,
            reset: false,
            reconfigure: false,
            quick: false,
        }));
    }

    #[test]
    fn native_gateway_setup_only_allows_plain_gateway() {
        assert!(should_use_native_gateway_setup(&SetupArgs {
            section: Some(SetupSection::Gateway),
            non_interactive: false,
            reset: false,
            reconfigure: false,
            quick: false,
        }));
        assert!(!should_use_native_gateway_setup(&SetupArgs {
            section: Some(SetupSection::Gateway),
            non_interactive: false,
            reset: false,
            reconfigure: true,
            quick: false,
        }));
        assert!(!should_use_native_gateway_setup(&SetupArgs {
            section: Some(SetupSection::Model),
            non_interactive: false,
            reset: false,
            reconfigure: false,
            quick: false,
        }));
    }

    #[test]
    fn setup_section_names_are_stable() {
        assert_eq!(SetupSection::Model.as_str(), "model");
        assert_eq!(SetupSection::Tts.as_str(), "tts");
        assert_eq!(SetupSection::Terminal.as_str(), "terminal");
        assert_eq!(SetupSection::Gateway.as_str(), "gateway");
        assert_eq!(SetupSection::Tools.as_str(), "tools");
        assert_eq!(SetupSection::Agent.as_str(), "agent");
    }

    #[test]
    #[cfg(unix)]
    fn setup_uses_python_override_and_env_flags() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'section=%s non_interactive=%s reset=%s reconfigure=%s quick=%s\\n' \\\n\
    \"$HERMES_SETUP_SECTION\" \"$HERMES_SETUP_NON_INTERACTIVE\" \"$HERMES_SETUP_RESET\" \\\n\
    \"$HERMES_SETUP_RECONFIGURE\" \"$HERMES_SETUP_QUICK\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_SETUP_PYTHON", &fake_python);
        let context = HermesContext::new(temp.path());
        print_setup(
            &context,
            SetupArgs {
                section: Some(SetupSection::Gateway),
                non_interactive: true,
                reset: true,
                reconfigure: true,
                quick: true,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("section=gateway non_interactive=1 reset=1 reconfigure=1 quick=1"));

        remove_env_var("HERMES_SETUP_PYTHON");
    }

    #[test]
    #[cfg(unix)]
    fn setup_model_section_uses_direct_python_bootstrap() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  case \"$2\" in\n\
    *\"setup_model_provider\"*) echo model >> '{}';;\n\
    *\"run_setup_wizard\"*) echo wizard >> '{}';;\n\
  esac\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display(),
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_SETUP_PYTHON", &fake_python);
        let context = HermesContext::new(temp.path());
        print_setup(
            &context,
            SetupArgs {
                section: Some(SetupSection::Model),
                non_interactive: false,
                reset: false,
                reconfigure: false,
                quick: false,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("model"));
        assert!(!output.contains("wizard"));
        remove_env_var("HERMES_SETUP_PYTHON");
    }

    #[test]
    fn setup_model_native_openrouter_updates_env_config_and_auth_marker() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "model:\n  provider: custom\n  default: llama3\n  base_url: http://old.example/v1\n  api_mode: chat_completions\n",
        )
        .unwrap();
        fs::write(home.join(".env"), "OPENAI_BASE_URL=http://old.example/v1\n").unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "version": 1,
                "active_provider": "nous",
                "providers": {
                    "nous": {
                        "access_token": "token"
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let openrouter_choice = providers
            .iter()
            .position(|provider| provider.name == "openrouter")
            .map(|index| index + 1)
            .unwrap();
        let mut input = io::Cursor::new(
            format!("{openrouter_choice}\nsk-openrouter\n\nopenai/gpt-5.4\n").into_bytes(),
        );
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: openrouter"));
        assert!(config_text.contains("default: openai/gpt-5.4"));
        assert!(config_text.contains("base_url: https://openrouter.ai/api/v1"));
        assert!(config_text.contains("api_mode: chat_completions"));

        let env_text = fs::read_to_string(home.join(".env")).unwrap();
        assert!(env_text.contains("OPENROUTER_API_KEY=sk-openrouter"));
        assert!(!env_text.contains("OPENAI_BASE_URL="));

        let auth_text = fs::read_to_string(home.join("auth.json")).unwrap();
        let auth_json: JsonValue = serde_json::from_str(&auth_text).unwrap();
        assert!(auth_json["active_provider"].is_null());

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Choose a provider and default model."));
        assert!(rendered.contains("Default model set to: openai/gpt-5.4 (via OpenRouter)"));
    }

    #[test]
    fn setup_model_saved_legacy_custom_provider_stays_native() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "custom_providers:\n  - name: Local Ollama\n    base_url: http://localhost:11434/v1\n    api_key: ${LOCAL_LLM_KEY}\n    model: llama3.1:8b\n    api_mode: chat_completions\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let saved_choice = providers.len() + 1;
        let mut input = io::Cursor::new(format!("{saved_choice}\nllama3.3:70b\n").into_bytes());
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: custom"));
        assert!(config_text.contains("default: llama3.3:70b"));
        assert!(config_text.contains("base_url: http://localhost:11434/v1"));
        assert!(config_text.contains("api_key: ${LOCAL_LLM_KEY}"));
        assert!(config_text.contains("api_mode: chat_completions"));
        assert!(config_text.contains("- name: Local Ollama"));
        assert!(config_text.contains("model: llama3.3:70b"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Local Ollama (localhost:11434/v1)"));
        assert!(rendered.contains("Default model set to: llama3.3:70b (via Local Ollama)"));
    }

    #[test]
    fn setup_model_saved_keyed_custom_provider_stays_native() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "providers:\n  demo-endpoint:\n    name: Demo Endpoint\n    api: https://demo.example/v1/\n    key_env: DEMO_API_KEY\n    default_model: demo-chat-v1\n    api_mode: chat_completions\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let saved_choice = providers.len() + 1;
        let mut input = io::Cursor::new(format!("{saved_choice}\ndemo-chat-v2\n").into_bytes());
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: custom"));
        assert!(config_text.contains("default: demo-chat-v2"));
        assert!(config_text.contains("base_url: https://demo.example/v1"));
        assert!(config_text.contains("api_key: ${DEMO_API_KEY}"));
        assert!(config_text.contains("default_model: demo-chat-v2"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Demo Endpoint (demo.example/v1)"));
        assert!(rendered.contains("Default model set to: demo-chat-v2 (via Demo Endpoint)"));
    }

    #[test]
    fn setup_model_custom_endpoint_stays_native() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let custom_choice = providers.len() + 1;
        let mut input = io::Cursor::new(
            format!("{custom_choice}\nhttp://localhost:11434\nsk-local\n\nllama3.1:8b\n64k\n\n")
                .into_bytes(),
        );
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: custom"));
        assert!(config_text.contains("default: llama3.1:8b"));
        assert!(config_text.contains("base_url: http://localhost:11434/v1"));
        assert!(config_text.contains("api_key: sk-local"));
        assert!(config_text.contains("name: Local (localhost:11434)"));
        assert!(config_text.contains("model: llama3.1:8b"));
        assert!(config_text.contains("context_length: 64000"));
        assert!(!config_text.contains("api_mode:"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Custom endpoint (enter URL manually)"));
        assert!(rendered.contains("Suggested URL: http://localhost:11434/v1"));
        assert!(
            rendered.contains("Default model set to: llama3.1:8b (via http://localhost:11434/v1)")
        );
    }

    #[test]
    fn setup_model_remove_saved_legacy_custom_provider_stays_native() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "custom_providers:\n  - name: Local Ollama\n    base_url: http://localhost:11434/v1\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let remove_choice = providers.len() + 3;
        let mut input = io::Cursor::new(format!("{remove_choice}\n1\n").into_bytes());
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(!config_text.contains("Local Ollama"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Remove a saved custom provider"));
        assert!(rendered.contains("Removed \"Local Ollama\" from saved custom providers."));
    }

    #[test]
    fn setup_model_remove_saved_keyed_custom_provider_stays_native() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "providers:\n  demo-endpoint:\n    name: Demo Endpoint\n    api: https://demo.example/v1/\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let remove_choice = providers.len() + 3;
        let mut input = io::Cursor::new(format!("{remove_choice}\n1\n").into_bytes());
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(!config_text.contains("demo-endpoint"));
        assert!(!config_text.contains("Demo Endpoint"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Removed \"Demo Endpoint\" from saved custom providers."));
    }

    #[test]
    fn setup_model_openai_codex_fresh_login_stays_native() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        let (issuer, requests, server) = spawn_codex_oauth_server(
            "header.eyJlbWFpbCI6ImNvZGV4LXNldHVwQGV4YW1wbGUuY29tIn0.sig",
            "codex-refresh-setup",
        );
        set_env_var("HERMES_AUTH_CODEX_ISSUER", &issuer);
        set_env_var(
            "HERMES_AUTH_CODEX_TOKEN_URL",
            format!("{issuer}/oauth/token"),
        );
        set_env_var("HERMES_AUTH_CODEX_MAX_WAIT_SECONDS", "5");
        set_env_var(
            "HERMES_CODEX_BASE_URL",
            "https://codex.example.test/backend-api/codex",
        );

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let codex_choice = providers
            .iter()
            .position(|provider| provider.name == "openai-codex")
            .map(|index| index + 1)
            .unwrap();
        let mut input = io::Cursor::new(format!("{codex_choice}\ngpt-5.5\n").into_bytes());
        let mut output = Vec::new();

        let result = run_native_model_setup_with_io(&context, &mut input, &mut output);
        remove_env_var("HERMES_AUTH_CODEX_ISSUER");
        remove_env_var("HERMES_AUTH_CODEX_TOKEN_URL");
        remove_env_var("HERMES_AUTH_CODEX_MAX_WAIT_SECONDS");
        remove_env_var("HERMES_CODEX_BASE_URL");

        result.unwrap();
        server.join().unwrap();
        let captured = requests.lock().unwrap().clone();
        assert_eq!(captured.len(), 3);

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: openai-codex"));
        assert!(config_text.contains("default: gpt-5.5"));
        assert!(config_text.contains("base_url: https://codex.example.test/backend-api/codex"));
        assert!(config_text.contains("api_mode: codex_responses"));

        let auth_text = fs::read_to_string(home.join("auth.json")).unwrap();
        let auth_json: JsonValue = serde_json::from_str(&auth_text).unwrap();
        assert_eq!(
            auth_json["providers"]["openai-codex"]["tokens"]["access_token"],
            "header.eyJlbWFpbCI6ImNvZGV4LXNldHVwQGV4YW1wbGUuY29tIn0.sig"
        );
        assert_eq!(
            auth_json["providers"]["openai-codex"]["tokens"]["refresh_token"],
            "codex-refresh-setup"
        );
        assert_eq!(
            auth_json["credential_pool"]["openai-codex"][0]["label"],
            "codex-setup@example.com"
        );

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Signing in to OpenAI Codex..."));
        assert!(rendered.contains("Added openai-codex OAuth credential #1"));
        assert!(rendered.contains("Default model set to: gpt-5.5 (via OpenAI Codex)"));
    }

    #[test]
    fn setup_model_minimax_oauth_fresh_login_stays_native() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        let (portal_base_url, requests, server) = spawn_minimax_oauth_server(
            "header.eyJwcmVmZXJyZWRfdXNlcm5hbWUiOiJtaW5pbWF4LXNldHVwQGV4YW1wbGUuY29tIn0.sig",
            "minimax-refresh-setup",
            "minimax-state-setup",
        );
        set_env_var("HERMES_AUTH_MINIMAX_PORTAL_URL", &portal_base_url);
        set_env_var(
            "HERMES_AUTH_MINIMAX_TEST_VERIFIER",
            "minimax-verifier-setup",
        );
        set_env_var("HERMES_AUTH_MINIMAX_TEST_STATE", "minimax-state-setup");

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let minimax_choice = providers
            .iter()
            .position(|provider| provider.name == "minimax-oauth")
            .map(|index| index + 1)
            .unwrap();
        let mut input = io::Cursor::new(format!("{minimax_choice}\nMiniMax-M2.7\n").into_bytes());
        let mut output = Vec::new();

        let result = run_native_model_setup_with_io(&context, &mut input, &mut output);
        remove_env_var("HERMES_AUTH_MINIMAX_PORTAL_URL");
        remove_env_var("HERMES_AUTH_MINIMAX_TEST_VERIFIER");
        remove_env_var("HERMES_AUTH_MINIMAX_TEST_STATE");

        result.unwrap();
        server.join().unwrap();
        let captured = requests.lock().unwrap().clone();
        assert_eq!(captured.len(), 2);

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: minimax-oauth"));
        assert!(config_text.contains("default: MiniMax-M2.7"));
        assert!(config_text.contains(&format!("base_url: {portal_base_url}/anthropic")));
        assert!(config_text.contains("api_mode: anthropic_messages"));

        let auth_text = fs::read_to_string(home.join("auth.json")).unwrap();
        let auth_json: JsonValue = serde_json::from_str(&auth_text).unwrap();
        assert_eq!(
            auth_json["providers"]["minimax-oauth"]["access_token"],
            "header.eyJwcmVmZXJyZWRfdXNlcm5hbWUiOiJtaW5pbWF4LXNldHVwQGV4YW1wbGUuY29tIn0.sig"
        );
        assert_eq!(
            auth_json["providers"]["minimax-oauth"]["refresh_token"],
            "minimax-refresh-setup"
        );
        assert_eq!(
            auth_json["credential_pool"]["minimax-oauth"][0]["label"],
            "minimax-setup@example.com"
        );

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Starting Hermes login via MiniMax OAuth..."));
        assert!(rendered.contains(&format!("Portal: {portal_base_url}")));
        assert!(rendered.contains("Added minimax-oauth OAuth credential #1"));
        assert!(rendered.contains("Default model set to: MiniMax-M2.7 (via MiniMax (OAuth))"));
    }

    #[test]
    fn setup_model_nous_fresh_login_stays_native() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        let shared = temp.path().join("shared");
        fs::create_dir_all(&home).unwrap();
        let (portal_base_url, requests, server) = spawn_nous_oauth_server(
            "header.eyJlbWFpbCI6Im5vdXMtc2V0dXBAZXhhbXBsZS5jb20ifQ.sig",
            "nous-refresh-setup",
        );
        set_env_var("HERMES_SHARED_AUTH_DIR", &shared);
        set_env_var("HERMES_PORTAL_BASE_URL", &portal_base_url);
        set_env_var(
            "NOUS_INFERENCE_BASE_URL",
            format!("{portal_base_url}/requested-inference/v1"),
        );

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let nous_choice = providers
            .iter()
            .position(|provider| provider.name == "nous")
            .map(|index| index + 1)
            .unwrap();
        let mut input =
            io::Cursor::new(format!("{nous_choice}\nmoonshotai/kimi-k2.6\n").into_bytes());
        let mut output = Vec::new();

        let result = run_native_model_setup_with_io(&context, &mut input, &mut output);
        remove_env_var("HERMES_SHARED_AUTH_DIR");
        remove_env_var("HERMES_PORTAL_BASE_URL");
        remove_env_var("NOUS_INFERENCE_BASE_URL");

        result.unwrap();
        server.join().unwrap();
        let captured = requests.lock().unwrap().clone();
        assert_eq!(captured.len(), 3);

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: nous"));
        assert!(config_text.contains("default: moonshotai/kimi-k2.6"));
        assert!(config_text.contains(&format!("base_url: {portal_base_url}/runtime-inference/v1")));
        assert!(config_text.contains("api_mode: chat_completions"));

        let auth_text = fs::read_to_string(home.join("auth.json")).unwrap();
        let auth_json: JsonValue = serde_json::from_str(&auth_text).unwrap();
        assert_eq!(
            auth_json["providers"]["nous"]["refresh_token"],
            "nous-refresh-setup"
        );
        assert_eq!(
            auth_json["providers"]["nous"]["agent_key"],
            "nous-agent-key"
        );
        assert_eq!(
            auth_json["credential_pool"]["nous"][0]["label"],
            "nous-setup@example.com"
        );

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Starting Hermes login via Nous..."));
        assert!(rendered.contains(&format!("Portal: {portal_base_url}")));
        assert!(rendered.contains("Added nous OAuth credential #1"));
        assert!(rendered.contains("Default model set to: moonshotai/kimi-k2.6 (via Nous Portal)"));
    }

    #[test]
    fn setup_model_google_gemini_fresh_login_stays_native() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        let (base_url, requests, server) = spawn_google_oauth_server(
            "google-access-setup",
            "google-refresh-setup",
            "gemini-setup@example.com",
        );
        set_env_var(
            "HERMES_AUTH_GOOGLE_AUTH_URL",
            format!("{base_url}/authorize"),
        );
        set_env_var("HERMES_AUTH_GOOGLE_TOKEN_URL", format!("{base_url}/token"));
        set_env_var(
            "HERMES_AUTH_GOOGLE_USERINFO_URL",
            format!("{base_url}/userinfo"),
        );
        set_env_var("HERMES_AUTH_GOOGLE_TEST_STATE", "google-state-setup");
        set_env_var(
            "HERMES_AUTH_GOOGLE_TEST_VERIFIER",
            "fixed-google-verifier-setup",
        );
        set_env_var("HERMES_GEMINI_CLIENT_ID", "test-google-client");
        set_env_var("HERMES_GEMINI_CLIENT_SECRET", "test-google-secret");
        set_env_var("SSH_TTY", "1");

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let google_choice = providers
            .iter()
            .position(|provider| provider.name == "google-gemini-cli")
            .map(|index| index + 1)
            .unwrap();
        let mut input = io::Cursor::new(
            format!(
                "{google_choice}\nhttp://127.0.0.1:8085/oauth2callback?code=google-code-setup&state=google-state-setup\ngemini-3.1-pro-preview\n"
            )
            .into_bytes(),
        );
        let mut output = Vec::new();

        let result = run_native_model_setup_with_io(&context, &mut input, &mut output);
        remove_env_var("HERMES_AUTH_GOOGLE_AUTH_URL");
        remove_env_var("HERMES_AUTH_GOOGLE_TOKEN_URL");
        remove_env_var("HERMES_AUTH_GOOGLE_USERINFO_URL");
        remove_env_var("HERMES_AUTH_GOOGLE_TEST_STATE");
        remove_env_var("HERMES_AUTH_GOOGLE_TEST_VERIFIER");
        remove_env_var("HERMES_GEMINI_CLIENT_ID");
        remove_env_var("HERMES_GEMINI_CLIENT_SECRET");
        remove_env_var("SSH_TTY");

        result.unwrap();
        server.join().unwrap();
        let captured = requests.lock().unwrap().clone();
        assert_eq!(captured.len(), 2);

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: google-gemini-cli"));
        assert!(config_text.contains("default: gemini-3.1-pro-preview"));
        assert!(config_text.contains("base_url: cloudcode-pa://google"));
        assert!(config_text.contains("api_mode: chat_completions"));

        let oauth_state: JsonValue = serde_json::from_str(
            &fs::read_to_string(home.join("auth").join("google_oauth.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(oauth_state["access"], "google-access-setup");
        assert_eq!(oauth_state["refresh"], "google-refresh-setup");
        assert_eq!(oauth_state["email"], "gemini-setup@example.com");

        let auth_text = fs::read_to_string(home.join("auth.json")).unwrap();
        let auth_json: JsonValue = serde_json::from_str(&auth_text).unwrap();
        assert_eq!(
            auth_json["credential_pool"]["google-gemini-cli"][0]["label"],
            "gemini-setup@example.com"
        );

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Open this URL to authorize Hermes with Google Gemini CLI:"));
        assert!(
            rendered.contains(
                "Default model set to: gemini-3.1-pro-preview (via Google Gemini (OAuth))"
            )
        );
    }

    #[test]
    fn setup_model_openai_codex_stays_native_when_logged_in() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": "codex-access-token"
                        },
                        "auth_mode": "chatgpt"
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let codex_choice = providers
            .iter()
            .position(|provider| provider.name == "openai-codex")
            .map(|index| index + 1)
            .unwrap();
        let mut input = io::Cursor::new(format!("{codex_choice}\ngpt-5.4\n").into_bytes());
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: openai-codex"));
        assert!(config_text.contains("default: gpt-5.4"));
        assert!(config_text.contains("base_url: https://chatgpt.com/backend-api/codex"));
        assert!(config_text.contains("api_mode: codex_responses"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("OpenAI Codex credentials: already configured"));
        assert!(rendered.contains("Default model set to: gpt-5.4 (via OpenAI Codex)"));
    }

    #[test]
    fn setup_model_nous_stays_native_when_logged_in() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("auth.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "version": 1,
                "providers": {
                    "nous": {
                        "access_token": "nous-access-token",
                        "portal_base_url": "https://portal.nous.test",
                        "inference_base_url": "https://inference.nous.test/v1"
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let nous_choice = providers
            .iter()
            .position(|provider| provider.name == "nous")
            .map(|index| index + 1)
            .unwrap();
        let mut input =
            io::Cursor::new(format!("{nous_choice}\nmoonshotai/kimi-k2.6\n").into_bytes());
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: nous"));
        assert!(config_text.contains("default: moonshotai/kimi-k2.6"));
        assert!(config_text.contains("base_url: https://inference.nous.test/v1"));
        assert!(config_text.contains("api_mode: chat_completions"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Nous Portal credentials: already configured"));
        assert!(rendered.contains("Default model set to: moonshotai/kimi-k2.6 (via Nous Portal)"));
    }

    #[test]
    fn setup_model_bedrock_stays_native_when_aws_configured() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        set_env_var("AWS_PROFILE", "test-bedrock");
        set_env_var("AWS_REGION", "us-west-2");

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let bedrock_choice = providers
            .iter()
            .position(|provider| provider.name == "bedrock")
            .map(|index| index + 1)
            .unwrap();
        let mut input = io::Cursor::new(
            format!("{bedrock_choice}\nanthropic.claude-sonnet-4-6-20250514-v1:0\n").into_bytes(),
        );
        let mut output = Vec::new();

        let result = run_native_model_setup_with_io(&context, &mut input, &mut output);
        remove_env_var("AWS_PROFILE");
        remove_env_var("AWS_REGION");

        result.unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: bedrock"));
        assert!(config_text.contains("default: anthropic.claude-sonnet-4-6-20250514-v1:0"));
        assert!(config_text.contains("base_url: https://bedrock-runtime.us-west-2.amazonaws.com"));
        assert!(config_text.contains("api_mode: bedrock_converse"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("AWS Bedrock credentials: already configured"));
        assert!(rendered.contains(
            "Default model set to: anthropic.claude-sonnet-4-6-20250514-v1:0 (via AWS Bedrock)"
        ));
    }

    #[test]
    fn setup_model_copilot_stays_native_when_token_present() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        set_env_var("COPILOT_GITHUB_TOKEN", "github_pat_test_token");

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let copilot_choice = providers
            .iter()
            .position(|provider| provider.name == "copilot")
            .map(|index| index + 1)
            .unwrap();
        let mut input = io::Cursor::new(format!("{copilot_choice}\ngpt-5.4\n").into_bytes());
        let mut output = Vec::new();

        let result = run_native_model_setup_with_io(&context, &mut input, &mut output);
        remove_env_var("COPILOT_GITHUB_TOKEN");

        result.unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: copilot"));
        assert!(config_text.contains("default: gpt-5.4"));
        assert!(config_text.contains("base_url: https://api.githubcopilot.com"));
        assert!(config_text.contains("api_mode: codex_responses"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("GitHub Copilot credentials: already configured"));
        assert!(rendered.contains("Default model set to: gpt-5.4 (via GitHub Copilot)"));
    }

    #[test]
    fn setup_model_auxiliary_custom_endpoint_stays_native() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let auxiliary_choice = providers.len() + 2;
        let aux_providers = native_auxiliary_model_providers(&context, "auto", "");
        let custom_choice = aux_providers.len() + 2;
        let back_choice = AUXILIARY_MODEL_TASKS.len() + 2;
        let mut input = io::Cursor::new(
            format!(
                "{auxiliary_choice}\n1\n{custom_choice}\nhttp://localhost:11434\n\nllama3.1:8b\nsk-local\n{back_choice}\n"
            )
            .into_bytes(),
        );
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("vision:"));
        assert!(config_text.contains("provider: custom"));
        assert!(config_text.contains("model: llama3.1:8b"));
        assert!(config_text.contains("base_url: http://localhost:11434/v1"));
        assert!(config_text.contains("api_key: sk-local"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Configure auxiliary models..."));
        assert!(rendered.contains("Vision: custom (localhost:11434/v1) · llama3.1:8b"));
    }

    #[test]
    fn setup_model_auxiliary_provider_pin_stays_native() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "auxiliary:\n  vision:\n    provider: openrouter\n    model: openai/gpt-5.4\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let auxiliary_choice = providers.len() + 2;
        let openrouter_choice = native_auxiliary_model_providers(&context, "openrouter", "")
            .iter()
            .position(|provider| provider.name == "openrouter")
            .map(|index| index + 2)
            .unwrap();
        let back_choice = AUXILIARY_MODEL_TASKS.len() + 2;
        let mut input = io::Cursor::new(
            format!(
                "{auxiliary_choice}\n1\n{openrouter_choice}\nopenai/gpt-5.4-mini\n{back_choice}\n"
            )
            .into_bytes(),
        );
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: openrouter"));
        assert!(config_text.contains("model: openai/gpt-5.4-mini"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Vision: OpenRouter · openai/gpt-5.4-mini"));
    }

    #[test]
    fn setup_model_auxiliary_reset_stays_native() {
        let temp = TempDir::new().unwrap();
        let home = temp.path().join("home");
        fs::create_dir_all(&home).unwrap();
        fs::write(
            home.join("config.yaml"),
            "auxiliary:\n  vision:\n    provider: openrouter\n    model: openai/gpt-5.4\n    base_url: https://old.example/v1\n    api_key: old-secret\n    timeout: 77\n",
        )
        .unwrap();
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let providers = native_setup_model_providers(&context);
        let auxiliary_choice = providers.len() + 2;
        let reset_choice = AUXILIARY_MODEL_TASKS.len() + 1;
        let back_choice = AUXILIARY_MODEL_TASKS.len() + 2;
        let mut input = io::Cursor::new(
            format!("{auxiliary_choice}\n{reset_choice}\n{back_choice}\n").into_bytes(),
        );
        let mut output = Vec::new();

        run_native_model_setup_with_io(&context, &mut input, &mut output).unwrap();

        let config_text = fs::read_to_string(home.join("config.yaml")).unwrap();
        assert!(config_text.contains("provider: auto"));
        assert!(config_text.contains("timeout: 77"));
        assert!(!config_text.contains("https://old.example/v1"));
        assert!(!config_text.contains("old-secret"));

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Reset 1 auxiliary task(s) to auto."));
    }

    #[test]
    #[cfg(unix)]
    fn setup_gateway_section_runs_native_gateway_flow() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        let fake_python = temp.path().join("gateway-python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'accept=%s key=%s\\n' \"$HERMES_ACCEPT_HOOKS\" \"$HERMES_GATEWAY_SETUP_PLATFORM\" >> '{}'\n\
  cat <<'JSON'\n\
[]\n\
JSON\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_GATEWAY_PYTHON", &fake_python);
        let mut input = io::Cursor::new(b"1\n".to_vec());
        let mut output = Vec::new();
        run_native_gateway_setup_with_io(&context, &mut input, &mut output).unwrap();

        let log_text = fs::read_to_string(&log).unwrap();
        assert!(log_text.contains("accept= key="));
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Gateway Setup"));
        assert!(rendered.contains("Messaging Platforms"));
        assert!(rendered.contains("No platforms configured"));
        remove_env_var("HERMES_GATEWAY_PYTHON");
    }

    #[test]
    fn setup_tools_section_runs_native_tools_flow() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        if let Some(parent) = context.config_path().parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(
            context.config_path(),
            "platform_toolsets:\n  cli:\n    - file\n",
        )
        .unwrap();
        let mut input = io::Cursor::new(b"1\n1,4\n3\n".to_vec());
        let mut output = Vec::new();

        run_native_tools_setup_with_io(&context, &mut input, &mut output).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("- web"));
        assert!(saved.contains("- file"));
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Hermes Tool Configuration"));
        assert!(rendered.contains("Configure CLI"));
    }

    #[test]
    fn setup_tools_managed_install_reports_error() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        let mut input = io::Cursor::new(Vec::<u8>::new());
        let mut output = Vec::new();

        set_env_var("HERMES_MANAGED", "homebrew");
        run_native_tools_setup_with_io(&context, &mut input, &mut output).unwrap();
        remove_env_var("HERMES_MANAGED");

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Cannot run setup wizard"));
        assert!(rendered.contains("Homebrew"));
    }

    #[test]
    fn native_agent_setup_updates_config_and_removes_legacy_env() {
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(
            context.config_path(),
            "agent:\n  max_turns: 90\ndisplay:\n  tool_progress: all\ncompression:\n  threshold: 0.6\nsession_reset:\n  mode: both\n  idle_minutes: 120\n  at_hour: 6\nmax_turns: 12\n",
        )
        .unwrap();
        fs::write(
            context.env_path(),
            "OPENAI_API_KEY=test\nHERMES_MAX_ITERATIONS=77\n",
        )
        .unwrap();

        let mut ui = TestUi::new(&["120", "verbose", "0.75", "2", "300"]);
        run_native_agent_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("max_turns: 120"));
        assert!(saved.contains("tool_progress: verbose"));
        assert!(saved.contains("enabled: true"));
        assert!(saved.contains("threshold: 0.75"));
        assert!(saved.contains("mode: idle"));
        assert!(saved.contains("idle_minutes: 300"));
        assert!(!saved.contains("\nmax_turns: 12\n"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("OPENAI_API_KEY=test"));
        assert!(!env_text.contains("HERMES_MAX_ITERATIONS"));
    }

    #[test]
    fn native_tts_setup_writes_provider_and_secret() {
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(context.config_path(), "tts:\n  provider: edge\n").unwrap();

        let mut ui = TestUi::new(&["3", "sk-tts-test"]);
        run_native_tts_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("provider: openai"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("VOICE_TOOLS_OPENAI_KEY=sk-tts-test"));
    }

    #[test]
    #[cfg(unix)]
    fn native_tts_setup_writes_managed_nous_provider() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            r#"{"providers":{"nous":{"access_token":"nous-access-token"}}}"#,
        )
        .unwrap();

        let mut ui = TestUi::new(&["1"]);
        run_native_tts_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("provider: openai"));
        assert!(saved.contains("use_gateway: true"));
    }

    #[test]
    #[cfg(unix)]
    fn native_tts_setup_installs_piper() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  exit 1\n\
fi\n\
if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ] && [ \"$3\" = \"install\" ]; then\n\
  printf '%s %s %s %s %s %s\\n' \"$1\" \"$2\" \"$3\" \"$4\" \"$5\" \"$6\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        set_env_var("HERMES_SETUP_PYTHON", &fake_python);

        let mut ui = TestUi::new(&["10", "y"]);
        run_native_tts_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("provider: piper"));
        let logged = fs::read_to_string(&log).unwrap();
        assert!(logged.contains("-m pip install -U piper-tts --quiet"));
        remove_env_var("HERMES_SETUP_PYTHON");
    }

    #[test]
    fn native_terminal_setup_writes_docker_backend() {
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(context.config_path(), "terminal:\n  backend: local\n").unwrap();

        let mut ui = TestUi::new(&["2", "my-image:latest", "yes", "2", "4096", "20480"]);
        run_native_terminal_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("backend: docker"));
        assert!(saved.contains("docker_image: my-image:latest"));
        assert!(saved.contains("container_persistent: true"));
        assert!(saved.contains("container_cpu: 2.0"));
        assert!(saved.contains("container_memory: 4096"));
        assert!(saved.contains("container_disk: 20480"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("TERMINAL_ENV=docker"));
        assert!(env_text.contains("TERMINAL_DOCKER_IMAGE=my-image:latest"));
    }

    #[test]
    #[cfg(unix)]
    fn native_terminal_setup_writes_modal_direct_backend() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(context.config_path(), "terminal:\n  backend: local\n").unwrap();

        let fake_python = temp.path().join("python3");
        fs::write(
            &fake_python,
            "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then exit 0; fi\nif [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ]; then exit 0; fi\nexit 9\n",
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();
        set_env_var("HERMES_SETUP_PYTHON", &fake_python);

        let mut ui = TestUi::new(&["3", "modal-id", "modal-secret", "yes", "2", "4096", "20480"]);
        run_native_terminal_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("backend: modal"));
        assert!(saved.contains("modal_mode: direct"));
        assert!(saved.contains("container_persistent: true"));
        assert!(saved.contains("container_cpu: 2.0"));
        assert!(saved.contains("container_memory: 4096"));
        assert!(saved.contains("container_disk: 20480"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("TERMINAL_ENV=modal"));
        assert!(env_text.contains("TERMINAL_MODAL_MODE=direct"));
        assert!(env_text.contains("MODAL_TOKEN_ID=modal-id"));
        assert!(env_text.contains("MODAL_TOKEN_SECRET=modal-secret"));
        remove_env_var("HERMES_SETUP_PYTHON");
    }

    #[test]
    fn native_terminal_setup_writes_modal_managed_backend() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(context.config_path(), "terminal:\n  backend: local\n").unwrap();
        fs::write(
            context.hermes_home().join("auth.json"),
            r#"{"providers":{"nous":{"access_token":"nous-access-token"}}}"#,
        )
        .unwrap();

        let mut ui = TestUi::new(&["3", "1", "no", "1.5", "2048", "51200"]);
        run_native_terminal_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("backend: modal"));
        assert!(saved.contains("modal_mode: managed"));
        assert!(saved.contains("container_persistent: false"));
        assert!(saved.contains("container_cpu: 1.5"));
        assert!(saved.contains("container_memory: 2048"));
        assert!(saved.contains("container_disk: 51200"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("TERMINAL_ENV=modal"));
        assert!(env_text.contains("TERMINAL_MODAL_MODE=managed"));
        assert!(!env_text.contains("MODAL_TOKEN_ID="));
        assert!(!env_text.contains("MODAL_TOKEN_SECRET="));
    }

    #[test]
    fn native_terminal_setup_writes_vercel_backend() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(context.config_path(), "terminal:\n  backend: local\n").unwrap();

        let mut ui = TestUi::new(&["6", "node22", "no", "1.5", "2048"]);
        run_native_terminal_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("backend: vercel_sandbox"));
        assert!(saved.contains("vercel_runtime: node22"));
        assert!(saved.contains("container_persistent: false"));
        assert!(saved.contains("container_cpu: 1.5"));
        assert!(saved.contains("container_memory: 2048"));
        assert!(saved.contains("container_disk: 51200"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("TERMINAL_ENV=vercel_sandbox"));
        assert!(env_text.contains("TERMINAL_VERCEL_RUNTIME=node22"));
    }

    #[test]
    #[cfg(unix)]
    fn native_terminal_setup_writes_daytona_backend() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let context = HermesContext::new(temp.path());
        fs::create_dir_all(context.hermes_home()).unwrap();
        fs::write(context.config_path(), "terminal:\n  backend: local\n").unwrap();

        let fake_python = temp.path().join("python3");
        fs::write(
            &fake_python,
            "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then exit 0; fi\nif [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ]; then exit 0; fi\nexit 9\n",
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();
        set_env_var("HERMES_SETUP_PYTHON", &fake_python);

        let mut ui = TestUi::new(&[
            "5",
            "daytona-key",
            "sandbox:latest",
            "yes",
            "2",
            "4096",
            "20480",
        ]);
        run_native_terminal_setup(&context, &mut ui).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("backend: daytona"));
        assert!(saved.contains("daytona_image: sandbox:latest"));
        assert!(saved.contains("container_persistent: true"));
        assert!(saved.contains("container_cpu: 2.0"));
        assert!(saved.contains("container_memory: 4096"));
        assert!(saved.contains("container_disk: 20480"));

        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("TERMINAL_ENV=daytona"));
        assert!(env_text.contains("DAYTONA_API_KEY=daytona-key"));
        assert!(env_text.contains("TERMINAL_DAYTONA_IMAGE=sandbox:latest"));
        remove_env_var("HERMES_SETUP_PYTHON");
    }

    #[test]
    fn agent_setup_with_extra_flags_stays_on_python_path() {
        let _guard = test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let fake_python = temp.path().join("python3");
        let log = temp.path().join("python.log");
        fs::write(
            &fake_python,
            format!(
                "#!/bin/sh\n\
if [ \"$1\" = \"-c\" ]; then\n\
  printf 'section=%s quick=%s\\n' \"$HERMES_SETUP_SECTION\" \"$HERMES_SETUP_QUICK\" >> '{}'\n\
  exit 0\n\
fi\n\
exit 9\n",
                log.display()
            ),
        )
        .unwrap();
        let mut perms = fs::metadata(&fake_python).unwrap().permissions();
        #[cfg(unix)]
        perms.set_mode(0o755);
        fs::set_permissions(&fake_python, perms).unwrap();

        set_env_var("HERMES_SETUP_PYTHON", &fake_python);
        let context = HermesContext::new(temp.path());
        print_setup(
            &context,
            SetupArgs {
                section: Some(SetupSection::Agent),
                non_interactive: false,
                reset: false,
                reconfigure: false,
                quick: true,
            },
        )
        .unwrap();

        let output = fs::read_to_string(&log).unwrap();
        assert!(output.contains("section=agent quick=1"));
        remove_env_var("HERMES_SETUP_PYTHON");
    }
}
