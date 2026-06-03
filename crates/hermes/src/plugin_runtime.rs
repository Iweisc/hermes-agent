use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

pub(crate) use hermes_core::{
    DashboardSurface, DiscoveredPlugin, PlatformSurface, PluginCliCommand, PluginSource,
};
use hermes_core::{
    HermesContext, ToolDefinition, ToolRuntime, discover_context_engine_plugins,
    discover_dashboard_surfaces,
    discover_enabled_general_plugins as core_discover_enabled_general_plugins,
    discover_enabled_plugin_cli_commands, discover_enabled_plugin_platforms,
    discover_general_plugins as core_discover_general_plugins,
    discover_hook_registrations_from_source,
    discover_memory_provider_plugins as core_discover_memory_provider_plugins,
    discover_platform_surfaces_from_source, discover_plugin_cli_commands_from_source,
    discover_scanned_plugins, discover_tool_definitions_from_source,
};
#[cfg(test)]
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProviderOption {
    pub(crate) name: String,
    pub(crate) description: String,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct PluginCatalog {
    pub(crate) plugins: Vec<DiscoveredPlugin>,
    pub(crate) dashboards: Vec<DashboardSurface>,
    pub(crate) memory_providers: Vec<ProviderOption>,
    pub(crate) context_engines: Vec<ProviderOption>,
}

pub(crate) fn attach_python_plugin_runtime(
    context: &HermesContext,
    runtime: ToolRuntime,
) -> Result<ToolRuntime, Box<dyn Error>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let has_general_plugins =
        !core_discover_enabled_general_plugins(&context.hermes_home(), &cwd).is_empty();
    if !has_general_plugins && selected_context_engine(context).is_none() {
        return Ok(runtime);
    }
    hermes_core::attach_python_plugin_runtime(&context.hermes_home(), runtime)
        .map_err(|error| -> Box<dyn Error> { error.into() })
}

fn selected_context_engine(context: &HermesContext) -> Option<String> {
    let text = fs::read_to_string(context.config_path()).ok()?;
    let yaml = serde_yaml::from_str::<serde_yaml::Value>(&text).ok()?;
    let root = yaml.as_mapping()?;
    let context_section = root
        .get(&serde_yaml::Value::String("context".to_string()))?
        .as_mapping()?;
    let engine = context_section
        .get(&serde_yaml::Value::String("engine".to_string()))?
        .as_str()?
        .trim();
    (!engine.is_empty() && engine != "compressor").then(|| engine.to_string())
}

pub(crate) fn discover_plugin_catalog(
    context: &HermesContext,
) -> Result<PluginCatalog, Box<dyn Error>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let winners = discover_scanned_plugins(&context.hermes_home(), &cwd)
        .into_iter()
        .map(|plugin| (plugin.key.clone(), plugin))
        .collect::<BTreeMap<_, _>>();
    let dashboards = discover_dashboard_surfaces(&context.hermes_home(), &cwd);

    let mut memory = BTreeMap::<String, ProviderOption>::new();
    for plugin in discover_memory_provider_plugins(context)? {
        memory.entry(plugin.name.clone()).or_insert(ProviderOption {
            name: plugin.name,
            description: plugin.description,
        });
    }
    let context_engines = discover_context_engine_plugins(&context.hermes_home(), &cwd)
        .into_iter()
        .map(|plugin| ProviderOption {
            name: plugin.name,
            description: plugin.description,
        })
        .collect::<Vec<_>>();

    let mut plugins = winners.into_values().collect::<Vec<_>>();
    plugins.sort_by(|left, right| left.name.cmp(&right.name).then(left.key.cmp(&right.key)));

    Ok(PluginCatalog {
        plugins,
        dashboards,
        memory_providers: memory.into_values().collect(),
        context_engines,
    })
}

pub(crate) fn discover_general_plugins(
    context: &HermesContext,
) -> Result<Vec<DiscoveredPlugin>, Box<dyn Error>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    Ok(core_discover_general_plugins(&context.hermes_home(), &cwd))
}

pub(crate) fn discover_enabled_platform_plugins(
    context: &HermesContext,
) -> Result<Vec<PlatformSurface>, Box<dyn Error>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    Ok(discover_enabled_plugin_platforms(
        &context.hermes_home(),
        &cwd,
    ))
}

pub(crate) fn discover_enabled_cli_commands(
    context: &HermesContext,
) -> Result<Vec<PluginCliCommand>, Box<dyn Error>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    Ok(discover_enabled_plugin_cli_commands(
        &context.hermes_home(),
        &cwd,
    ))
}

pub(crate) fn discover_memory_providers(
    context: &HermesContext,
) -> Result<Vec<ProviderOption>, Box<dyn Error>> {
    let mut providers = discover_memory_provider_plugins(context)?
        .into_iter()
        .map(|provider| ProviderOption {
            name: provider.name,
            description: provider.description,
        })
        .collect::<Vec<_>>();
    providers.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(providers)
}

pub(crate) fn discover_memory_provider_plugins(
    context: &HermesContext,
) -> Result<Vec<DiscoveredPlugin>, Box<dyn Error>> {
    Ok(core_discover_memory_provider_plugins(
        &context.hermes_home(),
    ))
}

pub(crate) fn find_memory_provider_plugin(
    context: &HermesContext,
    provider_name: &str,
) -> Result<Option<DiscoveredPlugin>, Box<dyn Error>> {
    let name = provider_name.trim();
    if name.is_empty() {
        return Ok(None);
    }
    Ok(discover_memory_provider_plugins(context)?
        .into_iter()
        .find(|plugin| plugin.name == name))
}

pub(crate) fn discover_context_engines(
    context: &HermesContext,
) -> Result<Vec<ProviderOption>, Box<dyn Error>> {
    let mut providers = discover_plugin_catalog(context)?.context_engines;
    providers.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(providers)
}

fn discover_cli_commands(source_text: &str, plugin_name: &str) -> Vec<PluginCliCommand> {
    discover_plugin_cli_commands_from_source(source_text, plugin_name)
}

fn discover_hook_registrations(source_text: &str) -> Vec<String> {
    discover_hook_registrations_from_source(source_text)
}

fn discover_tool_definitions(source_text: &str) -> Vec<ToolDefinition> {
    discover_tool_definitions_from_source(source_text)
}

fn discover_platform_surfaces(source_text: &str, plugin_name: &str) -> Vec<PlatformSurface> {
    discover_platform_surfaces_from_source(source_text, plugin_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    use hermes_core::{dispatch_tool, get_tool_definitions_with_runtime};
    use tempfile::TempDir;

    fn write_plugin(dir: &Path, manifest: &str, source: &str) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("plugin.yaml"), manifest).unwrap();
        fs::write(dir.join("__init__.py"), source).unwrap();
    }

    #[test]
    fn extract_cli_and_platform_surfaces_from_source() {
        let source = r#"
def register(ctx):
    ctx.register_cli_command(
        name="meet",
        help="Join a meeting",
        description="Meet tools",
    )
    ctx.register_platform(
        name="irc",
        label="IRC",
        required_env=["IRC_SERVER", "IRC_CHANNEL"],
        install_hint="stdlib only",
        setup_fn=interactive_setup,
        emoji="💬",
    )
"#;
        let commands = discover_cli_commands(source, "demo");
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].name, "meet");
        assert_eq!(commands[0].help, "Join a meeting");

        let platforms = discover_platform_surfaces(source, "demo");
        assert_eq!(platforms.len(), 1);
        assert_eq!(platforms[0].key, "irc");
        assert_eq!(platforms[0].label, "IRC");
        assert_eq!(platforms[0].required_env, vec!["IRC_SERVER", "IRC_CHANNEL"]);
        assert!(platforms[0].has_setup_fn);
    }

    #[test]
    fn extract_tool_and_hook_surfaces_from_source() {
        let source = r#"
DEMO_SCHEMA = {
    "name": "demo_tool",
    "description": "Demo tool",
    "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
}

def register(ctx):
    ctx.register_tool(
        name="demo_tool",
        toolset="demo",
        schema=DEMO_SCHEMA,
        handler=handle_demo,
        emoji="plug",
    )
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
"#;
        let tools = discover_tool_definitions(source);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "demo_tool");
        assert_eq!(tools[0].toolset, "demo");
        assert_eq!(tools[0].description, "Demo tool");
        assert_eq!(tools[0].emoji, "plug");
        assert_eq!(
            tools[0].schema["parameters"]["type"],
            serde_json::json!("object")
        );

        let hooks = discover_hook_registrations(source);
        assert_eq!(hooks, vec![String::from("pre_llm_call")]);
    }

    #[test]
    fn extract_tuple_registered_tools_from_source() {
        let source = r#"
MEET_JOIN_SCHEMA = {
    "name": "meet_join",
    "description": "Join a meeting",
    "parameters": {"type": "object", "properties": {}},
}
MEET_STATUS_SCHEMA = {
    "name": "meet_status",
    "description": "Report meeting status",
    "parameters": {"type": "object", "properties": {}},
}
_TOOLS = (
    ("meet_join", MEET_JOIN_SCHEMA, handle_meet_join, "📞"),
    ("meet_status", MEET_STATUS_SCHEMA, handle_meet_status, "🟢"),
)

def register(ctx):
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="google_meet",
            schema=schema,
            handler=handler,
            emoji=emoji,
        )
"#;
        let tools = discover_tool_definitions(source);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "meet_join");
        assert_eq!(tools[0].toolset, "google_meet");
        assert_eq!(tools[0].description, "Join a meeting");
        assert_eq!(tools[1].name, "meet_status");
        assert_eq!(tools[1].description, "Report meeting status");
    }

    #[test]
    fn catalog_discovers_nested_plugins_and_dashboard_manifests() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled = temp.path().join("bundled");
        let home = temp.path().join(".hermes");
        let user_plugins = home.join("plugins");
        fs::create_dir_all(&user_plugins).unwrap();

        write_plugin(
            &bundled.join("platforms").join("irc"),
            "name: irc-platform\nkind: platform\ndescription: IRC\n",
            "from .adapter import register\n",
        );
        fs::write(
            bundled.join("platforms").join("irc").join("adapter.py"),
            r#"
def register(ctx):
    ctx.register_platform(name="irc", label="IRC", required_env=["IRC_SERVER"], emoji="💬")
"#,
        )
        .unwrap();
        write_plugin(
            &bundled.join("image_gen").join("openai"),
            "name: openai\nkind: backend\ndescription: OpenAI images\n",
            "def register(ctx):\n    pass\n",
        );
        write_plugin(
            &bundled.join("memory").join("honcho"),
            "name: honcho\ndescription: Memory\n",
            "class MemoryProvider:\n    pass\n",
        );
        write_plugin(
            &bundled.join("context_engine").join("custom-engine"),
            "name: custom-engine\ndescription: Custom context engine\n",
            "class DemoContext(ContextEngine):\n    pass\n",
        );
        fs::create_dir_all(bundled.join("example-dashboard").join("dashboard")).unwrap();
        fs::write(
            bundled
                .join("example-dashboard")
                .join("dashboard")
                .join("manifest.json"),
            r#"{"name":"example","label":"Example","entry":"dist/index.js","slots":["sidebar"]}"#,
        )
        .unwrap();

        unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", &bundled) };
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));

        let catalog = discover_plugin_catalog(&context).unwrap();
        assert!(
            catalog
                .plugins
                .iter()
                .any(|plugin| plugin.name == "irc-platform")
        );
        assert!(catalog.plugins.iter().any(|plugin| plugin.name == "openai"));
        assert!(
            catalog
                .memory_providers
                .iter()
                .any(|provider| provider.name == "honcho")
        );
        assert!(
            catalog
                .context_engines
                .iter()
                .any(|provider| provider.name == "custom-engine")
        );
        assert!(
            catalog
                .dashboards
                .iter()
                .any(|dashboard| dashboard.name == "example")
        );

        unsafe { std::env::remove_var("HERMES_BUNDLED_PLUGINS") };
    }

    #[test]
    fn memory_provider_discovery_prefers_bundled_and_accepts_manifestless_user_plugins() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let bundled = temp.path().join("bundled");
        let home = temp.path().join(".hermes");
        let user_plugins = home.join("plugins");
        fs::create_dir_all(bundled.join("memory").join("honcho")).unwrap();
        fs::create_dir_all(user_plugins.join("honcho")).unwrap();
        fs::create_dir_all(user_plugins.join("custom")).unwrap();

        fs::write(
            bundled.join("memory").join("honcho").join("__init__.py"),
            "class Honcho(MemoryProvider):\n    pass\n",
        )
        .unwrap();
        fs::write(
            bundled.join("memory").join("honcho").join("plugin.yaml"),
            "description: Bundled honcho\n",
        )
        .unwrap();
        fs::write(
            user_plugins.join("honcho").join("__init__.py"),
            "class Honcho(MemoryProvider):\n    pass\n",
        )
        .unwrap();
        fs::write(
            user_plugins.join("custom").join("__init__.py"),
            "def register_memory_provider(ctx):\n    pass\n",
        )
        .unwrap();

        unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", &bundled) };
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        let providers = discover_memory_provider_plugins(&context).unwrap();
        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0].name, "custom");
        assert_eq!(providers[0].description, "");
        assert_eq!(providers[1].name, "honcho");
        assert_eq!(providers[1].description, "Bundled honcho");
        unsafe { std::env::remove_var("HERMES_BUNDLED_PLUGINS") };
    }

    #[test]
    fn enabled_cli_commands_include_active_memory_provider() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let bundled = temp.path().join("bundled");
        fs::create_dir_all(home.join("plugins")).unwrap();
        fs::create_dir_all(bundled.join("memory").join("honcho")).unwrap();
        fs::write(
            bundled.join("memory").join("honcho").join("plugin.yaml"),
            "name: honcho\ndescription: Honcho provider\n",
        )
        .unwrap();
        fs::write(
            bundled.join("memory").join("honcho").join("cli.py"),
            "def register_cli(subparser):\n    pass\n",
        )
        .unwrap();
        fs::write(home.join("config.yaml"), "memory:\n  provider: honcho\n").unwrap();

        unsafe { std::env::set_var("HERMES_BUNDLED_PLUGINS", &bundled) };
        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home));
        let commands = discover_enabled_cli_commands(&context).unwrap();
        assert!(commands.iter().any(|command| command.name == "honcho"));
        unsafe { std::env::remove_var("HERMES_BUNDLED_PLUGINS") };
    }

    #[test]
    fn attach_python_plugin_runtime_bridges_tools_and_hooks() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let plugin_dir = home.join("plugins").join("demo");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("plugin.yaml"),
            "name: demo\ndescription: Demo plugin\nkind: standalone\n",
        )
        .unwrap();
        fs::write(
            plugin_dir.join("__init__.py"),
            r#"
import json


def register(ctx):
    ctx.register_tool(
        name="demo_tool",
        toolset="demo_tools",
        schema={
            "name": "demo_tool",
            "description": "Demo tool",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
        handler=lambda args, **kwargs: json.dumps(
            {
                "success": True,
                "echo": args.get("message", ""),
                "task_id": kwargs.get("task_id", ""),
            }
        ),
        description="Demo tool",
        emoji="plug",
    )

    def pre_llm_call(**kwargs):
        return {"context": "dynamic context"}

    def pre_tool_call(**kwargs):
        args = kwargs.get("args") or {}
        if kwargs.get("tool_name") == "demo_tool" and args.get("message") == "blocked":
            return {"action": "block", "message": "blocked by plugin"}
        return None

    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("pre_tool_call", pre_tool_call)
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - demo\n",
        )
        .unwrap();

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let runtime = attach_python_plugin_runtime(
            &context,
            ToolRuntime::new(temp.path()).with_hermes_home(home),
        )
        .unwrap();

        let enabled_toolsets = vec![String::from("demo_tools")];
        let definitions =
            get_tool_definitions_with_runtime(Some(&enabled_toolsets), None, Some(&runtime));
        assert!(definitions.iter().any(|tool| tool.name == "demo_tool"));

        let hook_results = runtime.invoke_hook(
            "pre_llm_call",
            &serde_json::json!({
                "session_id": "session-1",
                "user_message": "hi",
                "conversation_history": [],
                "is_first_turn": true,
                "model": "test-model",
                "platform": "",
                "sender_id": "",
            }),
        );
        assert!(hook_results.iter().any(|result| {
            result
                .get("context")
                .and_then(JsonValue::as_str)
                .is_some_and(|value| value == "dynamic context")
        }));

        let result = dispatch_tool(
            "demo_tool",
            serde_json::json!({"message": "hello"}),
            &runtime,
        );
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        assert_eq!(
            parsed.get("echo").and_then(JsonValue::as_str),
            Some("hello")
        );

        let blocked = dispatch_tool(
            "demo_tool",
            serde_json::json!({"message": "blocked"}),
            &runtime,
        );
        let parsed_blocked: JsonValue = serde_json::from_str(&blocked).unwrap();
        assert_eq!(
            parsed_blocked.get("error").and_then(JsonValue::as_str),
            Some("blocked by plugin")
        );
    }

    #[test]
    fn attach_python_plugin_runtime_keeps_hook_only_plugins_active() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let plugin_dir = home.join("plugins").join("observer");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("plugin.yaml"),
            "name: observer\ndescription: Hook only plugin\nkind: standalone\n",
        )
        .unwrap();
        fs::write(
            plugin_dir.join("__init__.py"),
            r#"
def register(ctx):
    def pre_llm_call(**kwargs):
        return {"context": "hook-only context"}
    ctx.register_hook("pre_llm_call", pre_llm_call)
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - observer\n",
        )
        .unwrap();

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let runtime = attach_python_plugin_runtime(
            &context,
            ToolRuntime::new(temp.path()).with_hermes_home(home),
        )
        .unwrap();

        let hook_results = runtime.invoke_hook(
            "pre_llm_call",
            &serde_json::json!({
                "session_id": "session-1",
                "user_message": "hi",
                "conversation_history": [],
                "is_first_turn": true,
                "model": "test-model",
                "platform": "",
                "sender_id": "",
            }),
        );
        assert!(hook_results.iter().any(|result| {
            result
                .get("context")
                .and_then(JsonValue::as_str)
                .is_some_and(|value| value == "hook-only context")
        }));
    }

    #[test]
    fn attach_python_plugin_runtime_applies_terminal_output_transform_hook() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let temp = TempDir::new().unwrap();
        let home = temp.path().join(".hermes");
        let plugin_dir = home.join("plugins").join("normalizer");
        fs::create_dir_all(&plugin_dir).unwrap();
        fs::write(
            plugin_dir.join("plugin.yaml"),
            "name: normalizer\ndescription: Terminal output normalizer\nkind: standalone\n",
        )
        .unwrap();
        fs::write(
            plugin_dir.join("__init__.py"),
            r#"
def register(ctx):
    def transform_terminal_output(**kwargs):
        if kwargs.get("command") == "printf 'hi from shell\\n'":
            return "normalized output"
        return None

    ctx.register_hook("transform_terminal_output", transform_terminal_output)
"#,
        )
        .unwrap();
        fs::write(
            home.join("config.yaml"),
            "plugins:\n  enabled:\n    - normalizer\n",
        )
        .unwrap();

        let context = HermesContext::new(temp.path()).with_hermes_home_env(Some(home.clone()));
        let runtime = attach_python_plugin_runtime(
            &context,
            ToolRuntime::new(temp.path()).with_hermes_home(home),
        )
        .unwrap();

        let result = dispatch_tool(
            "terminal",
            serde_json::json!({"command": "printf 'hi from shell\\n'"}),
            &runtime,
        );
        let parsed: JsonValue = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["exit_code"], serde_json::json!(0));
        assert_eq!(parsed["output"], serde_json::json!("normalized output"));
    }
}
