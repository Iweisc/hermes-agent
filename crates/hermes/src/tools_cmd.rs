use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::{Command, ExitStatus};

use clap::{Args, Subcommand};
use hermes_core::{
    DelegateExecutor, HermesContext, LoadedConfig, ModelOverrides, ToolRuntime, dispatch_tool,
    get_tool_definitions, resolve_toolset, validate_toolset,
};
use serde_yaml::{Mapping, Value};

use crate::config_cmd::{read_raw_yaml_mapping, save_env_value, write_yaml_mapping};
use crate::mcp_cmd;
use crate::python_bridge::{project_root, resolve_repo_python};
use crate::setup_cmd;
use crate::{disabled_memory_toolsets, run_clarify_prompt};

#[derive(Args, Debug, Clone)]
pub struct ToolsArgs {
    #[arg(long, default_value_t = false)]
    pub summary: bool,
    #[command(subcommand)]
    pub command: Option<ToolsCommand>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum ToolsCommand {
    #[command(alias = "ls")]
    List {
        #[arg(long, default_value = "cli")]
        platform: String,
        #[arg(long)]
        toolset: Option<String>,
    },
    Enable {
        names: Vec<String>,
        #[arg(long, default_value = "cli")]
        platform: String,
    },
    Disable {
        names: Vec<String>,
        #[arg(long, default_value = "cli")]
        platform: String,
    },
    Run {
        name: String,
        #[arg(long, default_value = "{}")]
        args: String,
        #[arg(long)]
        cwd: Option<PathBuf>,
    },
}

#[derive(Copy, Clone)]
struct ConfigurableToolset {
    name: &'static str,
    label: &'static str,
}

#[derive(Copy, Clone)]
struct PlatformDef {
    name: &'static str,
    label: &'static str,
    default_toolset: &'static str,
}

const CONFIGURABLE_TOOLSETS: &[ConfigurableToolset] = &[
    ConfigurableToolset {
        name: "web",
        label: "Web Search & Scraping",
    },
    ConfigurableToolset {
        name: "browser",
        label: "Browser Automation",
    },
    ConfigurableToolset {
        name: "terminal",
        label: "Terminal & Processes",
    },
    ConfigurableToolset {
        name: "file",
        label: "File Operations",
    },
    ConfigurableToolset {
        name: "code_execution",
        label: "Code Execution",
    },
    ConfigurableToolset {
        name: "vision",
        label: "Vision / Image Analysis",
    },
    ConfigurableToolset {
        name: "video",
        label: "Video Analysis",
    },
    ConfigurableToolset {
        name: "image_gen",
        label: "Image Generation",
    },
    ConfigurableToolset {
        name: "moa",
        label: "Mixture of Agents",
    },
    ConfigurableToolset {
        name: "tts",
        label: "Text-to-Speech",
    },
    ConfigurableToolset {
        name: "skills",
        label: "Skills",
    },
    ConfigurableToolset {
        name: "todo",
        label: "Task Planning",
    },
    ConfigurableToolset {
        name: "memory",
        label: "Memory",
    },
    ConfigurableToolset {
        name: "session_search",
        label: "Session Search",
    },
    ConfigurableToolset {
        name: "clarify",
        label: "Clarifying Questions",
    },
    ConfigurableToolset {
        name: "delegation",
        label: "Task Delegation",
    },
    ConfigurableToolset {
        name: "cronjob",
        label: "Cron Jobs",
    },
    ConfigurableToolset {
        name: "messaging",
        label: "Cross-Platform Messaging",
    },
    ConfigurableToolset {
        name: "rl",
        label: "RL Training",
    },
    ConfigurableToolset {
        name: "homeassistant",
        label: "Home Assistant",
    },
    ConfigurableToolset {
        name: "spotify",
        label: "Spotify",
    },
    ConfigurableToolset {
        name: "discord",
        label: "Discord (read/participate)",
    },
    ConfigurableToolset {
        name: "discord_admin",
        label: "Discord Server Admin",
    },
    ConfigurableToolset {
        name: "yuanbao",
        label: "Yuanbao",
    },
];

const DEFAULT_OFF_TOOLSETS: &[&str] = &[
    "moa",
    "homeassistant",
    "rl",
    "spotify",
    "discord",
    "discord_admin",
    "video",
];

const PLATFORM_SCOPED_TOOLSETS: &[(&str, &[&str])] =
    &[("discord", &["discord"]), ("discord_admin", &["discord"])];

const PLATFORMS: &[PlatformDef] = &[
    PlatformDef {
        name: "cli",
        label: "CLI",
        default_toolset: "hermes-cli",
    },
    PlatformDef {
        name: "telegram",
        label: "Telegram",
        default_toolset: "hermes-telegram",
    },
    PlatformDef {
        name: "discord",
        label: "Discord",
        default_toolset: "hermes-discord",
    },
    PlatformDef {
        name: "slack",
        label: "Slack",
        default_toolset: "hermes-slack",
    },
    PlatformDef {
        name: "whatsapp",
        label: "WhatsApp",
        default_toolset: "hermes-whatsapp",
    },
    PlatformDef {
        name: "signal",
        label: "Signal",
        default_toolset: "hermes-signal",
    },
    PlatformDef {
        name: "bluebubbles",
        label: "BlueBubbles",
        default_toolset: "hermes-bluebubbles",
    },
    PlatformDef {
        name: "email",
        label: "Email",
        default_toolset: "hermes-email",
    },
    PlatformDef {
        name: "homeassistant",
        label: "Home Assistant",
        default_toolset: "hermes-homeassistant",
    },
    PlatformDef {
        name: "mattermost",
        label: "Mattermost",
        default_toolset: "hermes-mattermost",
    },
    PlatformDef {
        name: "matrix",
        label: "Matrix",
        default_toolset: "hermes-matrix",
    },
    PlatformDef {
        name: "dingtalk",
        label: "DingTalk",
        default_toolset: "hermes-dingtalk",
    },
    PlatformDef {
        name: "feishu",
        label: "Feishu",
        default_toolset: "hermes-feishu",
    },
    PlatformDef {
        name: "wecom",
        label: "WeCom",
        default_toolset: "hermes-wecom",
    },
    PlatformDef {
        name: "wecom_callback",
        label: "WeCom Callback",
        default_toolset: "hermes-wecom-callback",
    },
    PlatformDef {
        name: "weixin",
        label: "Weixin",
        default_toolset: "hermes-weixin",
    },
    PlatformDef {
        name: "qqbot",
        label: "QQBot",
        default_toolset: "hermes-qqbot",
    },
    PlatformDef {
        name: "yuanbao",
        label: "Yuanbao",
        default_toolset: "hermes-yuanbao",
    },
    PlatformDef {
        name: "webhook",
        label: "Webhook",
        default_toolset: "hermes-webhook",
    },
    PlatformDef {
        name: "api_server",
        label: "API Server",
        default_toolset: "hermes-api-server",
    },
    PlatformDef {
        name: "cron",
        label: "Cron",
        default_toolset: "hermes-cron",
    },
];

const TOOLS_RECONFIGURE_TOOLSET_BOOTSTRAP: &str = concat!(
    "import os\n",
    "from hermes_cli.config import load_config, save_config\n",
    "from hermes_cli.tools_config import TOOL_CATEGORIES, _configure_tool_category_for_reconfig, _reconfigure_simple_requirements\n",
    "config = load_config()\n",
    "toolset = os.environ['HERMES_TOOLS_RECONFIGURE_TOOLSET']\n",
    "category = TOOL_CATEGORIES.get(toolset)\n",
    "if category:\n",
    "    _configure_tool_category_for_reconfig(toolset, category, config)\n",
    "else:\n",
    "    _reconfigure_simple_requirements(toolset)\n",
    "save_config(config)\n",
);

const RECONFIGURABLE_TOOLSETS: &[&str] = &[
    "web",
    "browser",
    "vision",
    "image_gen",
    "moa",
    "tts",
    "rl",
    "homeassistant",
    "spotify",
];

const NATIVE_RECONFIGURABLE_TOOLSETS: &[&str] =
    &["web", "vision", "moa", "homeassistant", "tts", "image_gen"];

const DEFAULT_HASS_URL: &str = "http://homeassistant.local:8123";
const DEFAULT_FIRECRAWL_URL: &str = "http://localhost:3002";
const DEFAULT_SEARXNG_URL: &str = "http://localhost:8080";
const DEFAULT_FAL_IMAGE_MODEL: &str = "fal-ai/flux-2/klein/9b";
const DEFAULT_OPENAI_IMAGE_MODEL: &str = "gpt-image-2-medium";
const DEFAULT_XAI_IMAGE_MODEL: &str = "grok-imagine-image";

pub fn print_tools(
    context: &HermesContext,
    config: &LoadedConfig,
    args: ToolsArgs,
) -> Result<(), Box<dyn Error>> {
    if args.summary {
        if args.command.is_some() {
            return Err("tools --summary cannot be combined with a subcommand".into());
        }
        print!(
            "{}",
            render_tools_summary(&read_raw_yaml_mapping(&context.config_path())?)
        );
        return Ok(());
    }

    match args.command {
        None => {
            if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
                return Err("'hermes tools' requires an interactive terminal".into());
            }
            run_native_tools_interactive(context)
        }
        Some(ToolsCommand::List { platform, toolset }) => {
            if let Some(toolset_name) = toolset.as_deref() {
                return print_tool_definitions(config, toolset_name);
            }
            print!(
                "{}",
                render_tools_list(
                    &read_raw_yaml_mapping(&context.config_path())?,
                    normalize_platform(&platform)?,
                )?
            );
            Ok(())
        }
        Some(ToolsCommand::Enable { names, platform }) => update_tools_config(
            context,
            normalize_platform(&platform)?,
            &names,
            ToolAction::Enable,
        ),
        Some(ToolsCommand::Disable { names, platform }) => update_tools_config(
            context,
            normalize_platform(&platform)?,
            &names,
            ToolAction::Disable,
        ),
        Some(ToolsCommand::Run { name, args, cwd }) => run_tool(context, config, &name, &args, cwd),
    }
}

fn print_tool_definitions(config: &LoadedConfig, toolset_name: &str) -> Result<(), Box<dyn Error>> {
    let trimmed = toolset_name.trim();
    if trimmed.is_empty() {
        return Err("toolset must not be empty".into());
    }
    if !validate_toolset(trimmed) {
        return Err(format!("unknown toolset '{trimmed}'").into());
    }
    let enabled = vec![trimmed.to_string()];
    let disabled = disabled_memory_toolsets(&config.config.memory);
    let tools = get_tool_definitions(Some(enabled.as_slice()), disabled.as_deref());
    for tool in tools {
        println!(
            "{}\ttoolset={}\temoji={}\tdescription={}",
            tool.name, tool.toolset, tool.emoji, tool.description
        );
    }
    Ok(())
}

fn run_tool(
    context: &HermesContext,
    config: &LoadedConfig,
    name: &str,
    raw_args: &str,
    cwd: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let trimmed_name = name.trim();
    if trimmed_name.is_empty() {
        return Err("tool name cannot be empty".into());
    }
    let parsed_args: serde_json::Value = serde_json::from_str(raw_args)?;
    if !parsed_args.is_object() {
        return Err("tools run --args must be a JSON object".into());
    }
    let disabled = disabled_memory_toolsets(&config.config.memory);
    let tool_names = get_tool_definitions(Some(&config.config.toolsets), disabled.as_deref())
        .into_iter()
        .map(|tool| tool.name)
        .collect::<Vec<_>>();
    let delegate = DelegateExecutor::new(
        context.clone(),
        config.clone(),
        "rust-delegate",
        config.config.toolsets.clone(),
        ModelOverrides::default(),
        cwd.clone()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
    );
    let runtime = match cwd {
        Some(path) => ToolRuntime::new(path),
        None => ToolRuntime::default(),
    }
    .with_hermes_home(context.hermes_home())
    .with_available_tool_names(tool_names)
    .with_clarify_callback(run_clarify_prompt)
    .with_delegate_callback(move |request| delegate.execute(request));
    let mut runtime = runtime;
    let _ = runtime.load_memory_store(&config.config.memory);
    println!("{}", dispatch_tool(trimmed_name, parsed_args, &runtime));
    Ok(())
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum InteractiveToolsChoice {
    Platform(usize),
    AllPlatforms,
    Reconfigure,
    ConfigureMcp,
    Done,
}

pub(crate) fn run_native_tools_interactive(context: &HermesContext) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    run_native_tools_interactive_with_io(context, &mut input, &mut output)
}

pub(crate) fn run_native_tools_interactive_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "Hermes Tool Configuration")?;
    writeln!(output, "  Enable or disable built-in tools per platform.")?;
    writeln!(
        output,
        "  Some provider/API-key reconfiguration paths still use the narrowed compatibility path."
    )?;
    writeln!(output)?;

    loop {
        let root = read_raw_yaml_mapping(&context.config_path())?;
        let platforms = enabled_platforms();
        let options = build_interactive_tools_options(&root, &platforms);
        let choice = prompt_menu_choice(
            input,
            output,
            "Select an option",
            &options
                .iter()
                .map(|(_, label)| label.as_str())
                .collect::<Vec<_>>(),
        )?;
        match options[choice].0 {
            InteractiveToolsChoice::Platform(index) => {
                let platform = platforms
                    .get(index)
                    .ok_or("selected platform is no longer available")?;
                let current = enabled_builtin_toolsets(&root, platform.name);
                let selected = prompt_toolset_selection(
                    input,
                    output,
                    platform.label,
                    platform.name,
                    &current,
                )?;
                if let Some(enabled) = selected {
                    apply_platform_selection(output, context, platform.name, &current, &enabled)?;
                } else {
                    writeln!(output, "No changes.")?;
                }
                writeln!(output)?;
            }
            InteractiveToolsChoice::AllPlatforms => {
                let mut union = BTreeSet::new();
                for platform in &platforms {
                    union.extend(enabled_builtin_toolsets(&root, platform.name));
                }
                let selected =
                    prompt_toolset_selection(input, output, "All platforms", "cli", &union)?;
                if let Some(enabled) = selected {
                    apply_global_selection(output, context, &platforms, &root, &enabled)?;
                } else {
                    writeln!(output, "No changes.")?;
                }
                writeln!(output)?;
            }
            InteractiveToolsChoice::Reconfigure => {
                run_tools_reconfigure_with_io(context, input, output)?;
                writeln!(output)?;
            }
            InteractiveToolsChoice::ConfigureMcp => {
                configure_mcp_tools_interactive_with_io(context, input, output)?;
                writeln!(output)?;
            }
            InteractiveToolsChoice::Done => break,
        }
    }

    Ok(())
}

fn run_tools_reconfigure_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let root = read_raw_yaml_mapping(&context.config_path())?;
    let configurable = CONFIGURABLE_TOOLSETS
        .iter()
        .filter(|toolset| {
            reconfigurable_toolset(toolset.name)
                && (toolset_enabled_for_reconfigure(&root, toolset.name)
                    || toolset_has_known_configuration(context, &root, toolset.name))
        })
        .collect::<Vec<_>>();

    if configurable.is_empty() {
        writeln!(output, "No configured tools to reconfigure.")?;
        return Ok(());
    }

    let mut labels = configurable
        .iter()
        .map(|toolset| toolset.label)
        .collect::<Vec<_>>();
    labels.push("Cancel");
    let choice = prompt_menu_choice(
        input,
        output,
        "Which tool would you like to reconfigure?",
        &labels,
    )?;
    if choice >= configurable.len() {
        return Ok(());
    }

    let toolset = configurable[choice].name;
    if native_reconfigurable_toolset(toolset) {
        run_native_tool_reconfigure_with_io(context, toolset, input, output)
    } else {
        run_python_tools_reconfigure_toolset(toolset)
    }
}

fn run_native_tool_reconfigure_with_io(
    context: &HermesContext,
    toolset: &str,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    match toolset {
        "web" => reconfigure_web_with_io(context, input, output),
        "vision" => reconfigure_simple_env_tool_with_io(
            context,
            input,
            output,
            "Vision / Image Analysis",
            "OPENROUTER_API_KEY",
            Some("https://openrouter.ai/keys"),
        ),
        "moa" => reconfigure_simple_env_tool_with_io(
            context,
            input,
            output,
            "Mixture of Agents",
            "OPENROUTER_API_KEY",
            Some("https://openrouter.ai/keys"),
        ),
        "homeassistant" => reconfigure_homeassistant_with_io(context, input, output),
        "tts" => reconfigure_tts_with_io(context, input, output),
        "image_gen" => reconfigure_image_gen_with_io(context, input, output),
        other => run_python_tools_reconfigure_toolset(other),
    }
}

fn reconfigure_web_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let current_backend = nested_string(&root, &["web", "backend"]).unwrap_or_default();
    let current_use_gateway = nested_bool(&root, &["web", "use_gateway"]).unwrap_or(false);
    let current_firecrawl_url = env_value_for_context(context, "FIRECRAWL_API_URL");
    let compatibility_recommended = current_use_gateway
        || !matches!(
            current_backend.as_str(),
            "" | "firecrawl" | "exa" | "parallel" | "tavily" | "searxng"
        );

    writeln!(output)?;
    writeln!(output, "Web Search & Scraping")?;
    let choices = [
        "Firecrawl Cloud",
        "Exa",
        "Parallel",
        "Tavily",
        "Firecrawl Self-Hosted",
        "SearXNG",
        if compatibility_recommended {
            "Use compatibility flow (recommended)"
        } else {
            "Use compatibility flow"
        },
    ];
    let selection = prompt_menu_choice(input, output, "Select provider", &choices)?;
    if selection == choices.len() - 1 {
        return run_python_tools_reconfigure_toolset("web");
    }

    {
        let web = ensure_mapping(&mut root, "web");
        web.insert(
            yaml_key("backend"),
            Value::String(
                match selection {
                    0 | 4 => "firecrawl",
                    1 => "exa",
                    2 => "parallel",
                    3 => "tavily",
                    5 => "searxng",
                    _ => unreachable!(),
                }
                .to_string(),
            ),
        );
        web.insert(yaml_key("use_gateway"), Value::Bool(false));
    }

    match selection {
        0 => {
            writeln!(output, "Get key at: https://firecrawl.dev")?;
            prompt_secret_env_update(
                context,
                input,
                output,
                "FIRECRAWL_API_KEY",
                "Firecrawl API key",
            )?;
        }
        1 => {
            writeln!(output, "Get key at: https://exa.ai")?;
            prompt_secret_env_update(context, input, output, "EXA_API_KEY", "Exa API key")?;
        }
        2 => {
            writeln!(output, "Get key at: https://parallel.ai")?;
            prompt_secret_env_update(
                context,
                input,
                output,
                "PARALLEL_API_KEY",
                "Parallel API key",
            )?;
        }
        3 => {
            writeln!(output, "Get key at: https://app.tavily.com/home")?;
            prompt_secret_env_update(context, input, output, "TAVILY_API_KEY", "Tavily API key")?;
        }
        4 => {
            prompt_url_env_update(
                context,
                input,
                output,
                "FIRECRAWL_API_URL",
                "Firecrawl instance URL",
                current_firecrawl_url
                    .as_deref()
                    .unwrap_or(DEFAULT_FIRECRAWL_URL),
            )?;
        }
        5 => {
            let current = env_value_for_context(context, "SEARXNG_URL");
            prompt_url_env_update(
                context,
                input,
                output,
                "SEARXNG_URL",
                "SearXNG instance URL",
                current.as_deref().unwrap_or(DEFAULT_SEARXNG_URL),
            )?;
        }
        _ => unreachable!(),
    }

    write_yaml_mapping(&context.config_path(), &root)?;
    writeln!(
        output,
        "Saved web backend: {}",
        nested_string(&root, &["web", "backend"]).unwrap_or_default()
    )?;
    Ok(())
}

fn reconfigure_homeassistant_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "Smart Home")?;
    prompt_secret_env_update(
        context,
        input,
        output,
        "HASS_TOKEN",
        "Home Assistant Long-Lived Access Token",
    )?;
    let current =
        env_value_for_context(context, "HASS_URL").unwrap_or_else(|| DEFAULT_HASS_URL.to_string());
    prompt_url_env_update(
        context,
        input,
        output,
        "HASS_URL",
        "Home Assistant URL",
        &current,
    )?;
    writeln!(output, "Home Assistant settings updated.")?;
    Ok(())
}

fn reconfigure_tts_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    if setup_cmd::run_native_tts_setup_with_io(context, input, output)? {
        return Ok(());
    }
    run_python_tools_reconfigure_toolset("tts")
}

fn reconfigure_image_gen_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let current_provider = nested_string(&root, &["image_gen", "provider"]).unwrap_or_default();
    let current_use_gateway = nested_bool(&root, &["image_gen", "use_gateway"]).unwrap_or(false);
    let compatibility_recommended =
        current_use_gateway || !matches!(current_provider.as_str(), "" | "fal" | "openai" | "xai");

    writeln!(output)?;
    writeln!(output, "Image Generation")?;
    let choices = [
        "FAL.ai",
        "OpenAI Images",
        "xAI Images",
        if compatibility_recommended {
            "Use compatibility flow (recommended)"
        } else {
            "Use compatibility flow"
        },
    ];
    let selection = prompt_menu_choice(input, output, "Select provider", &choices)?;
    if selection == choices.len() - 1 {
        return run_python_tools_reconfigure_toolset("image_gen");
    }

    {
        let image_gen = ensure_mapping(&mut root, "image_gen");
        image_gen.insert(yaml_key("use_gateway"), Value::Bool(false));
    }

    match selection {
        0 => {
            let current_model = nested_string(&root, &["image_gen", "model"]);
            let image_gen = ensure_mapping(&mut root, "image_gen");
            image_gen.insert(yaml_key("provider"), Value::String("fal".to_string()));
            if !current_model.as_deref().is_some_and(is_known_fal_model) {
                image_gen.insert(
                    yaml_key("model"),
                    Value::String(DEFAULT_FAL_IMAGE_MODEL.to_string()),
                );
            }
            writeln!(output, "Get key at: https://fal.ai/dashboard/keys")?;
            prompt_secret_env_update(context, input, output, "FAL_KEY", "FAL API key")?;
            writeln!(
                output,
                "Image generation provider set to FAL.ai ({})",
                nested_string(&root, &["image_gen", "model"])
                    .unwrap_or_else(|| DEFAULT_FAL_IMAGE_MODEL.to_string())
            )?;
        }
        1 => {
            let openai_model = nested_string(&root, &["image_gen", "openai", "model"]);
            let image_gen = ensure_mapping(&mut root, "image_gen");
            image_gen.insert(yaml_key("provider"), Value::String("openai".to_string()));
            let openai = ensure_nested_mapping(image_gen, "openai");
            if !openai_model
                .as_deref()
                .is_some_and(is_known_openai_image_model)
            {
                openai.insert(
                    yaml_key("model"),
                    Value::String(DEFAULT_OPENAI_IMAGE_MODEL.to_string()),
                );
            }
            prompt_secret_env_update(context, input, output, "OPENAI_API_KEY", "OpenAI API key")?;
            writeln!(
                output,
                "Image generation provider set to OpenAI ({})",
                nested_string(&root, &["image_gen", "openai", "model"])
                    .unwrap_or_else(|| DEFAULT_OPENAI_IMAGE_MODEL.to_string())
            )?;
        }
        2 => {
            let xai_model = nested_string(&root, &["image_gen", "xai", "model"]);
            let image_gen = ensure_mapping(&mut root, "image_gen");
            image_gen.insert(yaml_key("provider"), Value::String("xai".to_string()));
            let xai = ensure_nested_mapping(image_gen, "xai");
            if !xai_model.as_deref().is_some_and(is_known_xai_image_model) {
                xai.insert(
                    yaml_key("model"),
                    Value::String(DEFAULT_XAI_IMAGE_MODEL.to_string()),
                );
            }
            prompt_secret_env_update(context, input, output, "XAI_API_KEY", "xAI API key")?;
            writeln!(
                output,
                "Image generation provider set to xAI ({})",
                nested_string(&root, &["image_gen", "xai", "model"])
                    .unwrap_or_else(|| DEFAULT_XAI_IMAGE_MODEL.to_string())
            )?;
        }
        _ => unreachable!(),
    }

    write_yaml_mapping(&context.config_path(), &root)?;
    Ok(())
}

fn reconfigure_simple_env_tool_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    label: &str,
    key: &str,
    url: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    writeln!(output)?;
    writeln!(output, "{label}")?;
    if let Some(url) = url {
        writeln!(output, "Get key at: {url}")?;
    }
    prompt_secret_env_update(context, input, output, key, key)?;
    writeln!(output, "{label} updated.")?;
    Ok(())
}

fn prompt_secret_env_update(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    key: &str,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    if let Some(existing) = env_value_for_context(context, key) {
        writeln!(
            output,
            "{key}: configured ({})",
            masked_secret_preview(&existing)
        )?;
    }
    let value = prompt_line(input, output, &format!("{label} (Enter to keep current)"))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        writeln!(output, "Kept current.")?;
        return Ok(());
    }
    save_env_value(context.env_path(), key, trimmed)?;
    writeln!(output, "Updated {key}.")?;
    Ok(())
}

fn prompt_url_env_update(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    key: &str,
    label: &str,
    current: &str,
) -> Result<(), Box<dyn Error>> {
    loop {
        let value = prompt_line(input, output, &format!("{label} [{current}]"))?;
        let trimmed = value.trim();
        if trimmed.is_empty() {
            writeln!(output, "Kept current.")?;
            return Ok(());
        }
        if !looks_like_http_url(trimmed) {
            writeln!(output, "{key} must start with http:// or https://.")?;
            continue;
        }
        save_env_value(context.env_path(), key, trimmed)?;
        writeln!(output, "Updated {key}.")?;
        return Ok(());
    }
}

fn configure_mcp_tools_interactive_with_io(
    context: &HermesContext,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let root = read_raw_yaml_mapping(&context.config_path())?;
    let servers = configured_mcp_server_names(&root);
    if servers.is_empty() {
        writeln!(output, "No enabled MCP servers configured.")?;
        return Ok(());
    }

    loop {
        let mut labels = servers
            .iter()
            .map(|name| format!("Configure MCP server: {name}"))
            .collect::<Vec<_>>();
        labels.push(String::from("Done"));
        let choice = prompt_menu_choice(
            &mut *input,
            &mut *output,
            "Select an MCP server",
            &labels.iter().map(String::as_str).collect::<Vec<_>>(),
        )?;
        if choice == servers.len() {
            return Ok(());
        }
        mcp_cmd::configure_server_name_with_io(
            context,
            &servers[choice],
            &mut *input,
            &mut *output,
        )?;
        writeln!(output)?;
    }
}

fn build_interactive_tools_options(
    root: &Mapping,
    platforms: &[&'static PlatformDef],
) -> Vec<(InteractiveToolsChoice, String)> {
    let mut options = Vec::new();
    let total = CONFIGURABLE_TOOLSETS.len();
    for (index, platform) in platforms.iter().enumerate() {
        let enabled = enabled_builtin_toolsets(root, platform.name);
        options.push((
            InteractiveToolsChoice::Platform(index),
            format!("Configure {} ({}/{})", platform.label, enabled.len(), total),
        ));
    }
    if platforms.len() > 1 {
        options.push((
            InteractiveToolsChoice::AllPlatforms,
            String::from("Configure all platforms (global)"),
        ));
    }
    options.push((
        InteractiveToolsChoice::Reconfigure,
        String::from("Reconfigure an existing tool's provider or API key"),
    ));
    if !collect_mcp_server_filters(root).is_empty() {
        options.push((
            InteractiveToolsChoice::ConfigureMcp,
            String::from("Configure MCP server tools"),
        ));
    }
    options.push((InteractiveToolsChoice::Done, String::from("Done")));
    options
}

fn prompt_menu_choice(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    title: &str,
    choices: &[&str],
) -> Result<usize, Box<dyn Error>> {
    loop {
        writeln!(output, "{title}:")?;
        for (index, choice) in choices.iter().enumerate() {
            writeln!(output, "  {}. {}", index + 1, choice)?;
        }
        let response = prompt_line(input, output, "Enter a number")?;
        let trimmed = response.trim();
        if trimmed.is_empty() {
            writeln!(output, "Please enter a selection.")?;
            continue;
        }
        let Ok(index) = trimmed.parse::<usize>() else {
            writeln!(output, "Invalid selection: '{trimmed}'.")?;
            continue;
        };
        if !(1..=choices.len()).contains(&index) {
            writeln!(output, "Selection must be between 1 and {}.", choices.len())?;
            continue;
        }
        return Ok(index - 1);
    }
}

fn prompt_toolset_selection(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    label: &str,
    platform: &str,
    current: &BTreeSet<String>,
) -> Result<Option<BTreeSet<String>>, Box<dyn Error>> {
    let available = CONFIGURABLE_TOOLSETS
        .iter()
        .filter(|toolset| toolset_allowed_for_platform(toolset.name, platform))
        .collect::<Vec<_>>();
    loop {
        writeln!(output, "Tools for {label}:")?;
        for (index, toolset) in available.iter().enumerate() {
            let status = if current.contains(toolset.name) {
                "[x]"
            } else {
                "[ ]"
            };
            writeln!(output, "  {}. {} {}", index + 1, status, toolset.label)?;
        }
        writeln!(
            output,
            "Enter enabled tool numbers as comma-separated values or ranges."
        )?;
        writeln!(output, "Use 'all', 'none', or press Enter to keep current.")?;
        let response = prompt_line(input, output, "Selection")?;
        match parse_toolset_selection(&response, available.len()) {
            Ok(parsed) => {
                return Ok(parsed.map(|indexes| {
                    indexes
                        .into_iter()
                        .map(|index| available[index].name.to_string())
                        .collect()
                }));
            }
            Err(error) => {
                writeln!(output, "{error}")?;
            }
        }
    }
}

fn parse_toolset_selection(
    raw: &str,
    total: usize,
) -> Result<Option<BTreeSet<usize>>, Box<dyn Error>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok(Some((0..total).collect()));
    }
    if trimmed.eq_ignore_ascii_case("none") {
        return Ok(Some(BTreeSet::new()));
    }

    let mut selected = BTreeSet::new();
    for part in trimmed.split(',') {
        let token = part.trim();
        if token.is_empty() {
            return Err("selection contains an empty item".into());
        }
        if let Some((start_raw, end_raw)) = token.split_once('-') {
            let start = parse_selection_index(start_raw.trim(), total)?;
            let end = parse_selection_index(end_raw.trim(), total)?;
            if start > end {
                return Err(format!("invalid range '{token}'").into());
            }
            selected.extend(start..=end);
            continue;
        }
        selected.insert(parse_selection_index(token, total)?);
    }
    Ok(Some(selected))
}

fn parse_selection_index(raw: &str, total: usize) -> Result<usize, Box<dyn Error>> {
    let number = raw
        .parse::<usize>()
        .map_err(|_| format!("invalid selection '{raw}'"))?;
    if !(1..=total).contains(&number) {
        return Err(format!("selection '{raw}' is out of range").into());
    }
    Ok(number - 1)
}

fn prompt_line(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    prompt: &str,
) -> Result<String, Box<dyn Error>> {
    write!(output, "{prompt}: ")?;
    output.flush()?;
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Err("interactive input closed".into());
    }
    Ok(line.trim_end_matches(['\r', '\n']).to_string())
}

fn apply_platform_selection(
    output: &mut dyn Write,
    context: &HermesContext,
    platform: &str,
    current: &BTreeSet<String>,
    enabled: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    save_platform_toolsets(&mut root, platform, enabled)?;
    write_yaml_mapping(&context.config_path(), &root)?;
    print_toolset_delta(
        output,
        platform_definition(platform).map_or(platform, |value| value.label),
        current,
        enabled,
    )?;
    Ok(())
}

fn apply_global_selection(
    output: &mut dyn Write,
    context: &HermesContext,
    platforms: &[&'static PlatformDef],
    root: &Mapping,
    enabled: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    let mut updated_root = root.clone();
    let mut changed = false;
    for platform in platforms {
        let previous = enabled_builtin_toolsets(root, platform.name);
        let next = enabled
            .iter()
            .filter(|name| toolset_allowed_for_platform(name, platform.name))
            .cloned()
            .collect::<BTreeSet<_>>();
        if previous != next {
            save_platform_toolsets(&mut updated_root, platform.name, &next)?;
            print_toolset_delta(output, platform.label, &previous, &next)?;
            changed = true;
        }
    }
    if changed {
        write_yaml_mapping(&context.config_path(), &updated_root)?;
    } else {
        writeln!(output, "No changes.")?;
    }
    Ok(())
}

fn print_toolset_delta(
    output: &mut dyn Write,
    label: &str,
    previous: &BTreeSet<String>,
    next: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    writeln!(output, "{label}:")?;
    let added = next
        .difference(previous)
        .filter_map(|name| configurable_toolset(name))
        .collect::<Vec<_>>();
    let removed = previous
        .difference(next)
        .filter_map(|name| configurable_toolset(name))
        .collect::<Vec<_>>();
    if added.is_empty() && removed.is_empty() {
        writeln!(output, "  No changes.")?;
        return Ok(());
    }
    for toolset in added {
        writeln!(output, "  + {}", toolset.label)?;
    }
    for toolset in removed {
        writeln!(output, "  - {}", toolset.label)?;
    }
    writeln!(output, "  Saved.")?;
    Ok(())
}

fn run_python_tools_reconfigure_toolset(toolset: &str) -> Result<(), Box<dyn Error>> {
    let root = project_root();
    let python = resolve_repo_python(&root, Some("HERMES_TOOLS_PYTHON"))
        .ok_or("could not find a Python interpreter for tools")?;
    let status = Command::new(&python)
        .current_dir(&root)
        .env("PYTHONPATH", root.display().to_string())
        .env("HERMES_TOOLS_RECONFIGURE_TOOLSET", toolset)
        .arg("-c")
        .arg(TOOLS_RECONFIGURE_TOOLSET_BOOTSTRAP)
        .status()?;
    if status.success() {
        return Ok(());
    }
    Err(exit_status_message(&format!("tools reconfigure {toolset}"), status).into())
}

fn render_tools_summary(root: &Mapping) -> String {
    let mut lines = Vec::new();
    let total = CONFIGURABLE_TOOLSETS.len();
    lines.push(String::from("Tool Summary"));
    lines.push(String::new());
    for platform in enabled_platforms() {
        let enabled = enabled_builtin_toolsets(root, platform.name);
        lines.push(format!(
            "  {} ({}/{})",
            platform.label,
            enabled.len(),
            total
        ));
        if enabled.is_empty() {
            lines.push(String::from("    (none enabled)"));
            continue;
        }
        for toolset in CONFIGURABLE_TOOLSETS {
            if enabled.contains(toolset.name) {
                lines.push(format!("    enabled  {}", toolset.label));
            }
        }
    }
    lines.push(String::new());
    lines.join("\n")
}

fn render_tools_list(root: &Mapping, platform: &str) -> Result<String, Box<dyn Error>> {
    let enabled = enabled_builtin_toolsets(root, platform);
    let mut lines = Vec::new();
    lines.push(format!("Built-in toolsets ({platform}):"));
    for toolset in CONFIGURABLE_TOOLSETS {
        if !toolset_allowed_for_platform(toolset.name, platform) {
            continue;
        }
        let status = if enabled.contains(toolset.name) {
            "enabled"
        } else {
            "disabled"
        };
        lines.push(format!(
            "  {status:<8}  {}  {}",
            toolset.name, toolset.label
        ));
    }

    let mcp_servers = collect_mcp_server_filters(root);
    if !mcp_servers.is_empty() {
        lines.push(String::new());
        lines.push(String::from("MCP servers:"));
        for (name, filter) in mcp_servers {
            if !filter.include.is_empty() {
                lines.push(format!(
                    "  {name}  [include only: {}]",
                    filter.include.join(", ")
                ));
            } else if !filter.exclude.is_empty() {
                lines.push(format!(
                    "  {name}  [excluded: {}]",
                    filter.exclude.join(", ")
                ));
            } else {
                lines.push(format!("  {name}  all tools enabled"));
            }
        }
    }

    lines.push(String::new());
    Ok(lines.join("\n"))
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum ToolAction {
    Enable,
    Disable,
}

fn update_tools_config(
    context: &HermesContext,
    platform: &str,
    names: &[String],
    action: ToolAction,
) -> Result<(), Box<dyn Error>> {
    if names.is_empty() {
        return Err("at least one tool target is required".into());
    }
    let mut root = read_raw_yaml_mapping(&context.config_path())?;
    let mut toolset_targets = Vec::new();
    let mut mcp_targets = Vec::new();

    for raw in names {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("tool target must not be empty".into());
        }
        if trimmed.contains(':') {
            let Some((server, tool)) = trimmed.split_once(':') else {
                return Err(format!("invalid MCP tool target '{trimmed}'").into());
            };
            if server.trim().is_empty() || tool.trim().is_empty() {
                return Err(format!("invalid MCP tool target '{trimmed}'").into());
            }
            mcp_targets.push((server.trim().to_string(), tool.trim().to_string()));
            continue;
        }
        if configurable_toolset(trimmed).is_none() {
            return Err(format!("unknown toolset '{trimmed}'").into());
        }
        if !toolset_allowed_for_platform(trimmed, platform) {
            let allowed = allowed_platforms_for_toolset(trimmed);
            return Err(format!(
                "toolset '{trimmed}' is not available on platform '{platform}' (only: {})",
                allowed.join(", ")
            )
            .into());
        }
        toolset_targets.push(trimmed.to_string());
    }

    if !toolset_targets.is_empty() {
        let mut enabled = enabled_builtin_toolsets(&root, platform);
        match action {
            ToolAction::Enable => enabled.extend(toolset_targets.iter().cloned()),
            ToolAction::Disable => {
                for name in &toolset_targets {
                    enabled.remove(name);
                }
            }
        }
        save_platform_toolsets(&mut root, platform, &enabled)?;
    }

    if !mcp_targets.is_empty() {
        apply_mcp_changes(&mut root, &mcp_targets, action)?;
    }

    write_yaml_mapping(&context.config_path(), &root)?;
    let verb = match action {
        ToolAction::Enable => "Enabled",
        ToolAction::Disable => "Disabled",
    };
    println!("{verb}: {}", names.join(", "));
    Ok(())
}

fn save_platform_toolsets(
    root: &mut Mapping,
    platform: &str,
    enabled_toolsets: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    let platform_def =
        platform_definition(platform).ok_or_else(|| format!("unknown platform '{platform}'"))?;
    let configurable = configurable_toolset_names()
        .into_iter()
        .map(String::from)
        .collect::<HashSet<_>>();
    let platform_defaults = platform_default_toolset_names()
        .into_iter()
        .map(String::from)
        .collect::<HashSet<_>>();

    let existing = load_explicit_platform_toolsets(root, platform_def.name);
    let mut preserved = BTreeSet::new();
    for entry in existing {
        if !configurable.contains(&entry) && !platform_defaults.contains(&entry) {
            preserved.insert(entry);
        }
    }
    preserved.remove("no_mcp");

    let platform_toolsets = ensure_mapping(root, "platform_toolsets");
    let saved = enabled_toolsets
        .iter()
        .filter(|name| toolset_allowed_for_platform(name, platform_def.name))
        .cloned()
        .chain(preserved)
        .map(Value::String)
        .collect::<Vec<_>>();
    platform_toolsets.insert(yaml_key(platform_def.name), Value::Sequence(saved));
    Ok(())
}

fn apply_mcp_changes(
    root: &mut Mapping,
    targets: &[(String, String)],
    action: ToolAction,
) -> Result<(), Box<dyn Error>> {
    let servers = root
        .get(yaml_key("mcp_servers"))
        .and_then(Value::as_mapping)
        .cloned()
        .unwrap_or_default();
    for (server_name, _) in targets {
        if !servers.contains_key(yaml_key(server_name)) {
            return Err(format!("MCP server '{server_name}' not found in config").into());
        }
    }

    let mcp_servers = ensure_mapping(root, "mcp_servers");
    for (server_name, tool_name) in targets {
        let server_value = mcp_servers
            .get_mut(yaml_key(server_name))
            .ok_or_else(|| format!("MCP server '{server_name}' not found in config"))?;
        let server_mapping = server_value
            .as_mapping_mut()
            .ok_or_else(|| format!("MCP server '{server_name}' must be a mapping"))?;
        let tools_mapping = ensure_nested_mapping(server_mapping, "tools");
        let exclude = tools_mapping
            .get(yaml_key("exclude"))
            .and_then(Value::as_sequence)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|value| value.as_str().map(ToOwned::to_owned))
            .collect::<Vec<_>>();
        let updated = match action {
            ToolAction::Disable => {
                let mut next = exclude;
                if !next.iter().any(|entry| entry == tool_name) {
                    next.push(tool_name.clone());
                }
                next
            }
            ToolAction::Enable => exclude
                .into_iter()
                .filter(|entry| entry != tool_name)
                .collect::<Vec<_>>(),
        };
        tools_mapping.insert(
            yaml_key("exclude"),
            Value::Sequence(updated.into_iter().map(Value::String).collect()),
        );
    }
    Ok(())
}

#[derive(Default)]
struct McpToolFilter {
    include: Vec<String>,
    exclude: Vec<String>,
}

fn collect_mcp_server_filters(root: &Mapping) -> BTreeMap<String, McpToolFilter> {
    let mut result = BTreeMap::new();
    let Some(servers) = root
        .get(yaml_key("mcp_servers"))
        .and_then(Value::as_mapping)
    else {
        return result;
    };
    for (name, config) in servers {
        let Some(server_name) = name.as_str() else {
            continue;
        };
        let Some(mapping) = config.as_mapping() else {
            continue;
        };
        let Some(tools) = mapping.get(yaml_key("tools")).and_then(Value::as_mapping) else {
            result.insert(server_name.to_string(), McpToolFilter::default());
            continue;
        };
        let include = yaml_sequence_strings(tools, "include");
        let exclude = yaml_sequence_strings(tools, "exclude");
        result.insert(server_name.to_string(), McpToolFilter { include, exclude });
    }
    result
}

fn configured_mcp_server_names(root: &Mapping) -> Vec<String> {
    let Some(servers) = root
        .get(yaml_key("mcp_servers"))
        .and_then(Value::as_mapping)
    else {
        return Vec::new();
    };
    servers
        .iter()
        .filter_map(|(name, config)| {
            let server_name = name.as_str()?.trim();
            if server_name.is_empty() {
                return None;
            }
            let mapping = config.as_mapping()?;
            mcp_server_enabled(mapping).then_some(server_name.to_string())
        })
        .collect()
}

fn reconfigurable_toolset(toolset: &str) -> bool {
    RECONFIGURABLE_TOOLSETS
        .iter()
        .any(|value| value == &toolset)
}

fn native_reconfigurable_toolset(toolset: &str) -> bool {
    NATIVE_RECONFIGURABLE_TOOLSETS
        .iter()
        .any(|value| value == &toolset)
}

fn toolset_enabled_for_reconfigure(root: &Mapping, toolset: &str) -> bool {
    enabled_platforms().iter().any(|platform| {
        toolset_allowed_for_platform(toolset, platform.name)
            && enabled_builtin_toolsets(root, platform.name).contains(toolset)
    })
}

fn toolset_has_known_configuration(context: &HermesContext, root: &Mapping, toolset: &str) -> bool {
    match toolset {
        "web" => {
            nested_mapping(root, &["web"]).is_some()
                || [
                    "FIRECRAWL_API_KEY",
                    "FIRECRAWL_API_URL",
                    "EXA_API_KEY",
                    "PARALLEL_API_KEY",
                    "TAVILY_API_KEY",
                    "SEARXNG_URL",
                ]
                .into_iter()
                .any(|key| env_value_for_context(context, key).is_some())
        }
        "browser" => {
            nested_mapping(root, &["browser"]).is_some()
                || [
                    "BROWSER_USE_API_KEY",
                    "BROWSERBASE_API_KEY",
                    "BROWSERBASE_PROJECT_ID",
                    "FIRECRAWL_API_KEY",
                    "CAMOFOX_URL",
                ]
                .into_iter()
                .any(|key| env_value_for_context(context, key).is_some())
        }
        "tts" => nested_mapping(root, &["tts"]).is_some(),
        "image_gen" => {
            nested_mapping(root, &["image_gen"]).is_some()
                || env_value_for_context(context, "FAL_KEY").is_some()
                || env_value_for_context(context, "OPENAI_API_KEY").is_some()
                || env_value_for_context(context, "XAI_API_KEY").is_some()
        }
        "homeassistant" => {
            env_value_for_context(context, "HASS_TOKEN").is_some()
                || env_value_for_context(context, "HASS_URL").is_some()
        }
        "vision" | "moa" => env_value_for_context(context, "OPENROUTER_API_KEY").is_some(),
        "spotify" => nested_string(root, &["auth", "type"]).is_some(),
        "rl" => {
            env_value_for_context(context, "TINKER_API_KEY").is_some()
                || env_value_for_context(context, "WANDB_API_KEY").is_some()
        }
        _ => false,
    }
}

fn mcp_server_enabled(config: &Mapping) -> bool {
    match config.get(yaml_key("enabled")) {
        None => true,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::String(text)) => !matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Some(Value::Number(number)) => number.as_i64().unwrap_or(1) != 0,
        _ => true,
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

fn masked_secret_preview(value: &str) -> String {
    let preview = value.chars().take(8).collect::<String>();
    if value.chars().count() > 8 {
        format!("{preview}...")
    } else {
        preview
    }
}

fn looks_like_http_url(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    normalized.starts_with("http://") || normalized.starts_with("https://")
}

fn is_known_fal_model(model: &str) -> bool {
    matches!(
        model,
        "fal-ai/flux-2/klein/9b"
            | "fal-ai/flux-2-pro"
            | "fal-ai/z-image/turbo"
            | "fal-ai/nano-banana-pro"
            | "fal-ai/gpt-image-1.5"
            | "fal-ai/gpt-image-2"
            | "fal-ai/ideogram/v3"
            | "fal-ai/recraft/v4/pro/text-to-image"
            | "fal-ai/qwen-image"
    )
}

fn is_known_openai_image_model(model: &str) -> bool {
    matches!(
        model,
        "gpt-image-2-low" | "gpt-image-2-medium" | "gpt-image-2-high"
    )
}

fn is_known_xai_image_model(model: &str) -> bool {
    model == DEFAULT_XAI_IMAGE_MODEL
}

fn nested_mapping<'a>(root: &'a Mapping, path: &[&str]) -> Option<&'a Mapping> {
    let mut current = root;
    for segment in path {
        current = current.get(yaml_key(segment))?.as_mapping()?;
    }
    Some(current)
}

fn nested_string(root: &Mapping, path: &[&str]) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let (parents, last) = path.split_at(path.len() - 1);
    nested_mapping(root, parents)?
        .get(yaml_key(last[0]))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn nested_bool(root: &Mapping, path: &[&str]) -> Option<bool> {
    if path.is_empty() {
        return None;
    }
    let (parents, last) = path.split_at(path.len() - 1);
    let value = nested_mapping(root, parents)?.get(yaml_key(last[0]))?;
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn enabled_builtin_toolsets(root: &Mapping, platform: &str) -> BTreeSet<String> {
    let Some(platform_def) = platform_definition(platform) else {
        return BTreeSet::new();
    };
    let toolset_names = load_platform_toolsets(root, platform_def.name);
    let configurable = configurable_toolset_names();
    let has_explicit_config = toolset_names
        .iter()
        .any(|name| configurable.contains(&name.as_str()));

    let mut enabled = BTreeSet::new();
    if has_explicit_config {
        for name in &toolset_names {
            if configurable.contains(&name.as_str()) && toolset_allowed_for_platform(name, platform)
            {
                enabled.insert(name.clone());
            }
        }
    } else {
        let platform_tools = resolve_toolset(platform_def.default_toolset)
            .into_iter()
            .collect::<HashSet<_>>();
        for toolset in CONFIGURABLE_TOOLSETS {
            if !toolset_allowed_for_platform(toolset.name, platform) {
                continue;
            }
            let resolved = resolve_toolset(toolset.name);
            if !resolved.is_empty() && resolved.iter().all(|tool| platform_tools.contains(tool)) {
                enabled.insert(toolset.name.to_string());
            }
        }
        let mut default_off = DEFAULT_OFF_TOOLSETS
            .iter()
            .map(|value| (*value).to_string())
            .collect::<HashSet<_>>();
        if default_off.contains(platform) && allowed_platforms_for_toolset(platform).is_empty() {
            default_off.remove(platform);
        }
        if default_off.contains("homeassistant")
            && std::env::var("HASS_TOKEN")
                .ok()
                .is_some_and(|value| !value.trim().is_empty())
        {
            default_off.remove("homeassistant");
        }
        enabled.retain(|name| !default_off.contains(name));
    }

    for disabled in agent_disabled_toolsets(root) {
        enabled.remove(&disabled);
    }
    enabled
}

fn load_platform_toolsets(root: &Mapping, platform: &str) -> Vec<String> {
    let Some(platform_def) = platform_definition(platform) else {
        return Vec::new();
    };
    let explicit = load_explicit_platform_toolsets(root, platform_def.name);
    if explicit.is_empty() {
        vec![platform_def.default_toolset.to_string()]
    } else {
        explicit
    }
}

fn load_explicit_platform_toolsets(root: &Mapping, platform: &str) -> Vec<String> {
    root.get(yaml_key("platform_toolsets"))
        .and_then(Value::as_mapping)
        .and_then(|mapping| mapping.get(yaml_key(platform)))
        .and_then(Value::as_sequence)
        .map(|sequence| {
            sequence
                .iter()
                .map(yaml_scalar_to_string)
                .filter(|value| !value.trim().is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn agent_disabled_toolsets(root: &Mapping) -> BTreeSet<String> {
    root.get(yaml_key("agent"))
        .and_then(Value::as_mapping)
        .and_then(|mapping| mapping.get(yaml_key("disabled_toolsets")))
        .and_then(Value::as_sequence)
        .map(|sequence| {
            sequence
                .iter()
                .filter_map(|value| value.as_str().map(ToOwned::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn enabled_platforms() -> Vec<&'static PlatformDef> {
    let mut platforms = vec![platform_definition("cli").expect("cli platform")];
    for (platform, env_key) in [
        ("telegram", "TELEGRAM_BOT_TOKEN"),
        ("discord", "DISCORD_BOT_TOKEN"),
        ("slack", "SLACK_BOT_TOKEN"),
        ("whatsapp", "WHATSAPP_ENABLED"),
        ("qqbot", "QQ_APP_ID"),
    ] {
        if env_var_present(env_key) {
            if let Some(definition) = platform_definition(platform) {
                platforms.push(definition);
            }
        }
    }
    platforms
}

fn env_var_present(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

fn normalize_platform(raw: &str) -> Result<&str, Box<dyn Error>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("platform must not be empty".into());
    }
    if platform_definition(trimmed).is_none() {
        let available = PLATFORMS
            .iter()
            .map(|platform| platform.name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!("unknown platform '{trimmed}'. Valid: {available}").into());
    }
    Ok(trimmed)
}

fn platform_definition(name: &str) -> Option<&'static PlatformDef> {
    PLATFORMS.iter().find(|platform| platform.name == name)
}

fn configurable_toolset(name: &str) -> Option<&'static ConfigurableToolset> {
    CONFIGURABLE_TOOLSETS
        .iter()
        .find(|toolset| toolset.name == name)
}

fn configurable_toolset_names() -> HashSet<&'static str> {
    CONFIGURABLE_TOOLSETS
        .iter()
        .map(|toolset| toolset.name)
        .collect()
}

fn platform_default_toolset_names() -> HashSet<&'static str> {
    PLATFORMS
        .iter()
        .map(|platform| platform.default_toolset)
        .collect()
}

fn toolset_allowed_for_platform(toolset: &str, platform: &str) -> bool {
    allowed_platforms_for_toolset(toolset).is_empty()
        || allowed_platforms_for_toolset(toolset)
            .iter()
            .any(|allowed| allowed == &platform)
}

fn allowed_platforms_for_toolset(toolset: &str) -> Vec<&'static str> {
    PLATFORM_SCOPED_TOOLSETS
        .iter()
        .find(|(name, _)| *name == toolset)
        .map(|(_, platforms)| platforms.to_vec())
        .unwrap_or_default()
}

fn yaml_scalar_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        _ => String::new(),
    }
}

fn yaml_sequence_strings(mapping: &Mapping, key: &str) -> Vec<String> {
    mapping
        .get(yaml_key(key))
        .and_then(Value::as_sequence)
        .map(|sequence| {
            sequence
                .iter()
                .filter_map(|value| value.as_str().map(ToOwned::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn ensure_mapping<'a>(root: &'a mut Mapping, key: &str) -> &'a mut Mapping {
    let value = root
        .entry(yaml_key(key))
        .or_insert_with(|| Value::Mapping(Mapping::new()));
    if !matches!(value, Value::Mapping(_)) {
        *value = Value::Mapping(Mapping::new());
    }
    value.as_mapping_mut().expect("mapping inserted above")
}

fn ensure_nested_mapping<'a>(root: &'a mut Mapping, key: &str) -> &'a mut Mapping {
    let value = root
        .entry(yaml_key(key))
        .or_insert_with(|| Value::Mapping(Mapping::new()));
    if !matches!(value, Value::Mapping(_)) {
        *value = Value::Mapping(Mapping::new());
    }
    value.as_mapping_mut().expect("mapping inserted above")
}

fn yaml_key(key: &str) -> Value {
    Value::String(key.to_string())
}

fn exit_status_message(command: &str, status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("{command} exited with status {code}"),
        None => format!("{command} terminated by signal"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::fs;
    use std::io::Cursor;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Parser, Debug)]
    struct ToolsHarness {
        #[command(subcommand)]
        command: ToolsCommand,
    }

    fn temp_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("hermes-rs-tools-{label}-{unique}"))
    }

    fn write_config(path: &Path, body: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }

    #[cfg(unix)]
    fn write_stdio_test_server(path: &Path) {
        fs::write(
            path,
            "#!/bin/sh\n\
printf '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"protocolVersion\":\"2025-03-26\"}}\\n'\n\
read line\n\
read line\n\
printf '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"alpha\",\"description\":\"first\"},{\"name\":\"beta\",\"description\":\"second\"}]}}\\n'\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).unwrap();
    }

    #[test]
    fn list_subcommand_parses_platform_and_toolset() {
        let parsed = ToolsHarness::try_parse_from([
            "tools",
            "list",
            "--platform",
            "discord",
            "--toolset",
            "web",
        ])
        .unwrap();
        match parsed.command {
            ToolsCommand::List { platform, toolset } => {
                assert_eq!(platform, "discord");
                assert_eq!(toolset.as_deref(), Some("web"));
            }
            _ => panic!("expected list subcommand"),
        }
    }

    #[test]
    fn enable_subcommand_parses_multiple_targets() {
        let parsed = ToolsHarness::try_parse_from([
            "tools",
            "enable",
            "web",
            "github:create_issue",
            "--platform",
            "cli",
        ])
        .unwrap();
        match parsed.command {
            ToolsCommand::Enable { names, platform } => {
                assert_eq!(platform, "cli");
                assert_eq!(
                    names,
                    vec![String::from("web"), String::from("github:create_issue"),]
                );
            }
            _ => panic!("expected enable subcommand"),
        }
    }

    #[test]
    fn render_summary_reports_enabled_toolsets() {
        let root = serde_yaml::from_str::<Value>(
            "platform_toolsets:\n  cli:\n    - web\n    - file\n    - memory\n",
        )
        .unwrap();
        let text = render_tools_summary(root.as_mapping().unwrap());
        assert!(text.contains("Tool Summary"));
        assert!(text.contains("CLI (3/24)"));
        assert!(text.contains("enabled  Web Search & Scraping"));
        assert!(text.contains("enabled  File Operations"));
    }

    #[test]
    fn enable_builtin_toolset_writes_platform_config() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("enable");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        write_config(
            &context.config_path(),
            "platform_toolsets:\n  cli:\n    - file\nmcp_servers:\n  github:\n    url: https://example.com/mcp\n",
        );

        update_tools_config(&context, "cli", &[String::from("web")], ToolAction::Enable).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("platform_toolsets:"));
        assert!(saved.contains("- file"));
        assert!(saved.contains("- web"));
        assert!(saved.contains("mcp_servers:"));
    }

    #[test]
    fn disable_mcp_tool_updates_exclude_list() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("disable-mcp");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        write_config(
            &context.config_path(),
            "mcp_servers:\n  github:\n    url: https://example.com/mcp\n",
        );

        update_tools_config(
            &context,
            "cli",
            &[String::from("github:create_issue")],
            ToolAction::Disable,
        )
        .unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("exclude:"));
        assert!(saved.contains("- create_issue"));
    }

    #[test]
    fn parse_toolset_selection_supports_ranges_and_keywords() {
        assert_eq!(parse_toolset_selection("", 4).unwrap(), None);
        assert_eq!(
            parse_toolset_selection("all", 3).unwrap(),
            Some(BTreeSet::from([0, 1, 2]))
        );
        assert_eq!(
            parse_toolset_selection("none", 3).unwrap(),
            Some(BTreeSet::new())
        );
        assert_eq!(
            parse_toolset_selection("1,3-4", 4).unwrap(),
            Some(BTreeSet::from([0, 2, 3]))
        );
        assert!(parse_toolset_selection("0", 4).is_err());
        assert!(parse_toolset_selection("4-2", 4).is_err());
    }

    #[test]
    fn interactive_tools_menu_updates_cli_toolsets_natively() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("interactive");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        write_config(
            &context.config_path(),
            "platform_toolsets:\n  cli:\n    - file\n",
        );
        let mut input = Cursor::new(b"1\n1,4\n3\n".to_vec());
        let mut output = Vec::new();

        run_native_tools_interactive_with_io(&context, &mut input, &mut output).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("- web"));
        assert!(saved.contains("- file"));
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Hermes Tool Configuration"));
        assert!(rendered.contains("Configure CLI"));
        assert!(rendered.contains("CLI:"));
        assert!(rendered.contains("+ Web Search & Scraping"));
    }

    #[test]
    fn tools_reconfigure_updates_web_backend_natively() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("reconfigure-web");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        write_config(
            &context.config_path(),
            "platform_toolsets:\n  cli:\n    - web\nweb:\n  backend: exa\n",
        );
        fs::write(context.env_path(), "EXA_API_KEY=old-exa-key\n").unwrap();

        let mut input = Cursor::new(b"1\n1\nnew-firecrawl-key\n".to_vec());
        let mut output = Vec::new();
        run_tools_reconfigure_with_io(&context, &mut input, &mut output).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("backend: firecrawl"));
        assert!(saved.contains("use_gateway: false"));
        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("EXA_API_KEY=old-exa-key"));
        assert!(env_text.contains("FIRECRAWL_API_KEY=new-firecrawl-key"));
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Which tool would you like to reconfigure?"));
        assert!(rendered.contains("Saved web backend: firecrawl"));
    }

    #[test]
    fn tools_reconfigure_tts_uses_native_setup_flow() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("reconfigure-tts");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        write_config(&context.config_path(), "tts:\n  provider: edge\n");

        let mut input = Cursor::new(b"3\nsk-tts-test\n".to_vec());
        let mut output = Vec::new();
        reconfigure_tts_with_io(&context, &mut input, &mut output).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("provider: openai"));
        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("VOICE_TOOLS_OPENAI_KEY=sk-tts-test"));
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Hermes Setup"));
        assert!(rendered.contains("TTS provider set to: OpenAI TTS"));
    }

    #[test]
    fn tools_reconfigure_image_gen_sets_openai_provider_natively() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("reconfigure-image-gen");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        write_config(&context.config_path(), "image_gen:\n  provider: fal\n");

        let mut input = Cursor::new(b"2\nsk-openai-image\n".to_vec());
        let mut output = Vec::new();
        reconfigure_image_gen_with_io(&context, &mut input, &mut output).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("provider: openai"));
        assert!(saved.contains("model: gpt-image-2-medium"));
        assert!(saved.contains("use_gateway: false"));
        let env_text = fs::read_to_string(context.env_path()).unwrap();
        assert!(env_text.contains("OPENAI_API_KEY=sk-openai-image"));
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Image Generation"));
        assert!(rendered.contains("Image generation provider set to OpenAI"));
    }

    #[test]
    fn tools_reconfigure_browser_uses_python_override() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let root = project_root();
        let temp = temp_path("bridge");
        let log_path = temp.join("tools-bridge.log");
        let python = temp.join("python3");
        let home = temp_path("reconfigure-browser");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        write_config(
            &context.config_path(),
            "platform_toolsets:\n  cli:\n    - browser\n",
        );
        fs::create_dir_all(&temp).unwrap();
        fs::write(
            &python,
            format!(
                "#!/bin/sh\nprintf 'toolset=%s\\n' \"$HERMES_TOOLS_RECONFIGURE_TOOLSET\" > \"{}\"\nprintf '%s\\n' \"$@\" >> \"{}\"\nexit 0\n",
                log_path.display(),
                log_path.display()
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&python).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&python, perms).unwrap();
        }

        unsafe { std::env::set_var("HERMES_TOOLS_PYTHON", &python) };
        let mut input = Cursor::new(b"1\n".to_vec());
        let mut output = Vec::new();
        let result = run_tools_reconfigure_with_io(&context, &mut input, &mut output);
        unsafe { std::env::remove_var("HERMES_TOOLS_PYTHON") };

        result.unwrap();
        let logged = fs::read_to_string(log_path).unwrap();
        assert!(logged.contains("toolset=browser"));
        assert!(logged.contains("-c"));
        assert!(logged.contains("_configure_tool_category_for_reconfig"));
        assert!(root.exists());
    }

    #[test]
    #[cfg(unix)]
    fn interactive_tools_menu_configures_mcp_servers_natively() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let home = temp_path("interactive-mcp");
        let context = HermesContext::new("/tmp").with_hermes_home_env(Some(home.clone()));
        fs::create_dir_all(&home).unwrap();
        let temp = tempfile::TempDir::new().unwrap();
        let script = temp.path().join("server.sh");
        write_stdio_test_server(&script);
        write_config(
            &context.config_path(),
            &format!(
                "mcp_servers:\n  alpha:\n    command: {}\n",
                serde_yaml::to_string(&script.display().to_string())
                    .unwrap()
                    .trim()
            ),
        );

        let mut input = Cursor::new(b"3\n1\n2\n2\n4\n".to_vec());
        let mut output = Vec::new();
        run_native_tools_interactive_with_io(&context, &mut input, &mut output).unwrap();

        let saved = fs::read_to_string(context.config_path()).unwrap();
        assert!(saved.contains("include:"));
        assert!(saved.contains("- beta"));
        assert!(!saved.contains("- alpha"));
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Select an MCP server"));
        assert!(rendered.contains("Connecting to 'alpha'"));
        assert!(rendered.contains("Updated config: 1/2 tools enabled"));

        let _ = fs::remove_dir_all(home);
    }
}
