use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Subcommand;
use hermes_core::{HermesContext, LoadedConfig};
use hmac::{Hmac, Mac};
use reqwest::blocking::Client;
use serde_json::{Map as JsonMap, Value as JsonValue, json};
use serde_yaml::{Mapping, Value as YamlValue};
use sha2::{Digest, Sha256};

const SUBSCRIPTIONS_FILENAME: &str = "webhook_subscriptions.json";
const DEFAULT_WEBHOOK_PORT: u16 = 8644;
const DEFAULT_WEBHOOK_HOST: &str = "0.0.0.0";
const DEFAULT_TEST_PAYLOAD: &str =
    r#"{"test": true, "event_type": "test", "message": "Hello from hermes webhook test"}"#;

type HmacSha256 = Hmac<Sha256>;

#[derive(Subcommand, Debug)]
pub enum WebhookCommand {
    #[command(alias = "add")]
    Subscribe {
        name: String,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value = "")]
        events: String,
        #[arg(long, default_value = "")]
        description: String,
        #[arg(long, default_value = "")]
        skills: String,
        #[arg(long, default_value = "log")]
        deliver: String,
        #[arg(long = "deliver-chat-id", default_value = "")]
        deliver_chat_id: String,
        #[arg(long, default_value = "")]
        secret: String,
        #[arg(long = "deliver-only")]
        deliver_only: bool,
    },
    #[command(alias = "ls")]
    List,
    #[command(alias = "rm")]
    Remove { name: String },
    Test {
        name: String,
        #[arg(long, default_value = "")]
        payload: String,
    },
}

pub fn print_webhook(
    context: &HermesContext,
    loaded: &LoadedConfig,
    command: Option<WebhookCommand>,
) -> Result<(), Box<dyn Error>> {
    let Some(command) = command else {
        println!("Usage: hermes webhook {{subscribe|list|remove|test}}");
        println!("Run 'hermes webhook --help' for details.");
        return Ok(());
    };

    if !is_webhook_enabled(loaded) {
        println!("{}", setup_hint(context));
        return Ok(());
    }

    match command {
        WebhookCommand::Subscribe {
            name,
            prompt,
            events,
            description,
            skills,
            deliver,
            deliver_chat_id,
            secret,
            deliver_only,
        } => subscribe(
            context,
            loaded,
            SubscribeArgs {
                name,
                prompt,
                events,
                description,
                skills,
                deliver,
                deliver_chat_id,
                secret,
                deliver_only,
            },
        )?,
        WebhookCommand::List => list_subscriptions(context, loaded)?,
        WebhookCommand::Remove { name } => remove_subscription(context, &name)?,
        WebhookCommand::Test { name, payload } => {
            test_subscription(context, loaded, &name, &payload)?
        }
    }
    Ok(())
}

#[derive(Debug)]
struct SubscribeArgs {
    name: String,
    prompt: String,
    events: String,
    description: String,
    skills: String,
    deliver: String,
    deliver_chat_id: String,
    secret: String,
    deliver_only: bool,
}

fn subscribe(
    context: &HermesContext,
    loaded: &LoadedConfig,
    args: SubscribeArgs,
) -> Result<(), Box<dyn Error>> {
    let name = normalize_subscription_name(&args.name)?;
    let mut subscriptions = load_subscriptions(context)?;
    let is_update = subscriptions.contains_key(&name);
    let secret = if args.secret.trim().is_empty() {
        generate_secret()
    } else {
        args.secret.trim().to_string()
    };
    let events = split_csv(&args.events);
    let skills = split_csv(&args.skills);
    let deliver = normalize_non_empty(&args.deliver).unwrap_or_else(|| "log".to_string());
    if args.deliver_only && deliver == "log" {
        return Err("--deliver-only requires --deliver to be a real target, not 'log'".into());
    }

    let mut route = JsonMap::new();
    route.insert(
        "description".to_string(),
        JsonValue::String(
            normalize_non_empty(&args.description)
                .unwrap_or_else(|| format!("Agent-created subscription: {name}")),
        ),
    );
    route.insert(
        "events".to_string(),
        JsonValue::Array(events.iter().cloned().map(JsonValue::String).collect()),
    );
    route.insert("secret".to_string(), JsonValue::String(secret.clone()));
    route.insert(
        "prompt".to_string(),
        JsonValue::String(normalize_non_empty(&args.prompt).unwrap_or_default()),
    );
    route.insert(
        "skills".to_string(),
        JsonValue::Array(skills.iter().cloned().map(JsonValue::String).collect()),
    );
    route.insert("deliver".to_string(), JsonValue::String(deliver.clone()));
    route.insert("created_at".to_string(), JsonValue::String(now_iso()));
    if args.deliver_only {
        route.insert("deliver_only".to_string(), JsonValue::Bool(true));
    }
    if let Some(chat_id) = normalize_non_empty(&args.deliver_chat_id) {
        route.insert("deliver_extra".to_string(), json!({"chat_id": chat_id}));
    }

    subscriptions.insert(name.clone(), JsonValue::Object(route));
    save_subscriptions(context, &subscriptions)?;

    let status = if is_update { "Updated" } else { "Created" };
    let base_url = webhook_base_url(loaded);
    println!("\n  {status} webhook subscription: {name}");
    println!("  URL:    {base_url}/webhooks/{name}");
    println!("  Secret: {secret}");
    if events.is_empty() {
        println!("  Events: (all)");
    } else {
        println!("  Events: {}", events.join(", "));
    }
    println!("  Deliver: {deliver}");
    if args.deliver_only {
        println!("  Mode: direct delivery (no agent, zero LLM cost)");
    }
    if let Some(prompt) = normalize_non_empty(&args.prompt) {
        let preview = truncate(&prompt, 80);
        let label = if args.deliver_only {
            "Message"
        } else {
            "Prompt"
        };
        println!("  {label}: {preview}");
    }
    println!("\n  Configure your service to POST to the URL above.");
    println!("  Use the secret for HMAC-SHA256 signature validation.");
    println!("  The gateway must be running to receive events (hermes gateway run).\n");
    Ok(())
}

fn list_subscriptions(
    context: &HermesContext,
    loaded: &LoadedConfig,
) -> Result<(), Box<dyn Error>> {
    let subscriptions = load_subscriptions(context)?;
    if subscriptions.is_empty() {
        println!("  No dynamic webhook subscriptions.");
        println!("  Create one with: hermes webhook subscribe <name>");
        return Ok(());
    }

    let base_url = webhook_base_url(loaded);
    println!("\n  {} webhook subscription(s):\n", subscriptions.len());
    for (name, route) in subscriptions {
        let events = route
            .get("events")
            .and_then(JsonValue::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(JsonValue::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "(all)".to_string());
        let mut deliver = route
            .get("deliver")
            .and_then(JsonValue::as_str)
            .unwrap_or("log")
            .to_string();
        if route
            .get("deliver_only")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false)
        {
            deliver.push_str(" (direct - no agent)");
        }
        println!("  * {name}");
        if let Some(description) = route.get("description").and_then(JsonValue::as_str) {
            if !description.trim().is_empty() {
                println!("    {description}");
            }
        }
        println!("    URL:     {base_url}/webhooks/{name}");
        println!("    Events:  {events}");
        println!("    Deliver: {deliver}");
        println!();
    }
    Ok(())
}

fn remove_subscription(context: &HermesContext, name: &str) -> Result<(), Box<dyn Error>> {
    let key = normalize_subscription_name(name)?;
    let mut subscriptions = load_subscriptions(context)?;
    if subscriptions.remove(&key).is_none() {
        println!("  No subscription named '{key}'.");
        println!("  Note: Static routes from config.yaml cannot be removed here.");
        return Ok(());
    }
    save_subscriptions(context, &subscriptions)?;
    println!("  Removed webhook subscription: {key}");
    Ok(())
}

fn test_subscription(
    context: &HermesContext,
    loaded: &LoadedConfig,
    name: &str,
    payload: &str,
) -> Result<(), Box<dyn Error>> {
    let key = normalize_subscription_name(name)?;
    let subscriptions = load_subscriptions(context)?;
    let Some(route) = subscriptions.get(&key).and_then(JsonValue::as_object) else {
        println!("  No subscription named '{key}'.");
        return Ok(());
    };

    let secret = route
        .get("secret")
        .and_then(JsonValue::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or("subscription secret is missing")?;
    let payload = normalize_non_empty(payload).unwrap_or_else(|| DEFAULT_TEST_PAYLOAD.to_string());
    let url = format!("{}/webhooks/{}", webhook_base_url(loaded), key);
    let signature = compute_signature(secret, &payload)?;

    println!("  Sending test POST to {url}");
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("X-Hub-Signature-256", signature)
        .header("X-GitHub-Event", "test")
        .body(payload.clone())
        .send();

    match response {
        Ok(response) => {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            println!("  Response ({}): {}", status.as_u16(), body);
        }
        Err(error) => {
            println!("  Error: {error}");
            println!("  Is the gateway running? (hermes gateway run)");
        }
    }
    Ok(())
}

fn load_subscriptions(
    context: &HermesContext,
) -> Result<BTreeMap<String, JsonValue>, Box<dyn Error>> {
    let path = subscriptions_path(context);
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = fs::read_to_string(path)?;
    let parsed = serde_json::from_str::<JsonValue>(&raw).unwrap_or(JsonValue::Null);
    let Some(object) = parsed.as_object() else {
        return Ok(BTreeMap::new());
    };
    Ok(object
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

fn save_subscriptions(
    context: &HermesContext,
    subscriptions: &BTreeMap<String, JsonValue>,
) -> Result<(), Box<dyn Error>> {
    let path = subscriptions_path(context);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(subscriptions)?;
    atomic_write(&path, payload.as_bytes())
}

fn subscriptions_path(context: &HermesContext) -> PathBuf {
    context.hermes_home().join(SUBSCRIPTIONS_FILENAME)
}

fn is_webhook_enabled(loaded: &LoadedConfig) -> bool {
    platform_webhook_mapping(loaded)
        .and_then(|mapping| mapping_value(mapping, "enabled"))
        .and_then(yaml_bool)
        .unwrap_or_else(|| env_truthy("WEBHOOK_ENABLED"))
}

fn webhook_base_url(loaded: &LoadedConfig) -> String {
    let extra = platform_webhook_mapping(loaded)
        .and_then(|mapping| mapping_value(mapping, "extra").and_then(YamlValue::as_mapping));
    let host = extra
        .and_then(|mapping| mapping_value(mapping, "host"))
        .and_then(YamlValue::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::trim)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            env_string("WEBHOOK_HOST").unwrap_or_else(|| DEFAULT_WEBHOOK_HOST.to_string())
        });
    let port = extra
        .and_then(|mapping| mapping_value(mapping, "port"))
        .and_then(yaml_u16)
        .or_else(|| env_string("WEBHOOK_PORT").and_then(|value| value.parse::<u16>().ok()))
        .unwrap_or(DEFAULT_WEBHOOK_PORT);
    let display_host = if host == "0.0.0.0" {
        "localhost".to_string()
    } else {
        host
    };
    format!("http://{display_host}:{port}")
}

fn setup_hint(context: &HermesContext) -> String {
    format!(
        "\n  Webhook platform is not enabled. To set it up:\n\n  \
1. Run the gateway setup wizard:\n     hermes gateway setup\n\n  \
2. Or add to {}/config.yaml:\n     platforms:\n       webhook:\n         enabled: true\n         extra:\n           host: \"0.0.0.0\"\n           port: 8644\n           secret: \"your-global-hmac-secret\"\n\n  \
3. Or set environment variables in {}/.env:\n     WEBHOOK_ENABLED=true\n     WEBHOOK_PORT=8644\n     WEBHOOK_SECRET=your-global-secret\n\n  \
Then start the gateway: hermes gateway run\n",
        context.display_hermes_home(),
        context.display_hermes_home()
    )
}

fn platform_webhook_mapping<'a>(loaded: &'a LoadedConfig) -> Option<&'a Mapping> {
    let root = loaded.raw.as_mapping()?;
    let platforms = mapping_value(root, "platforms")?.as_mapping()?;
    mapping_value(platforms, "webhook")?.as_mapping()
}

fn normalize_subscription_name(value: &str) -> Result<String, Box<dyn Error>> {
    let normalized = value.trim().to_ascii_lowercase().replace(' ', "-");
    let valid = !normalized.is_empty()
        && normalized.chars().enumerate().all(|(index, ch)| match ch {
            'a'..='z' | '0'..='9' => true,
            '-' | '_' => index > 0,
            _ => false,
        });
    if !valid
        || !normalized
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
    {
        return Err(format!(
            "Invalid name '{}'. Use lowercase alphanumeric with hyphens/underscores.",
            normalized
        )
        .into());
    }
    Ok(normalized)
}

fn split_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn normalize_non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn truncate(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        value.to_string()
    } else {
        value
            .chars()
            .take(max.saturating_sub(3))
            .collect::<String>()
            + "..."
    }
}

fn now_iso() -> String {
    chrono::DateTime::<chrono::Utc>::from(SystemTime::now())
        .to_rfc3339()
        .replace("+00:00", "Z")
}

fn generate_secret() -> String {
    let mut random_bytes = [0_u8; 32];
    if let Ok(mut file) = File::open("/dev/urandom") {
        if file.read_exact(&mut random_bytes).is_ok() {
            return hex_bytes(&random_bytes);
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos().to_le_bytes().to_vec())
            .unwrap_or_default(),
    );
    hasher.update(std::process::id().to_le_bytes());
    hex_bytes(&hasher.finalize())
}

fn compute_signature(secret: &str, payload: &str) -> Result<String, Box<dyn Error>> {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())?;
    mac.update(payload.as_bytes());
    let bytes = mac.finalize().into_bytes();
    Ok(format!("sha256={}", hex_bytes(&bytes)))
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn env_truthy(key: &str) -> bool {
    env_string(key)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn mapping_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_string()))
}

fn yaml_bool(value: &YamlValue) -> Option<bool> {
    match value {
        YamlValue::Bool(boolean) => Some(*boolean),
        YamlValue::String(text) => Some(matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )),
        _ => None,
    }
}

fn yaml_u16(value: &YamlValue) -> Option<u16> {
    match value {
        YamlValue::Number(number) => number
            .as_u64()
            .and_then(|value| u16::try_from(value).ok())
            .or_else(|| number.as_i64().and_then(|value| u16::try_from(value).ok())),
        YamlValue::String(text) => text.trim().parse::<u16>().ok(),
        _ => None,
    }
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("tmp-{unique}"));
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::thread;
    use tempfile::TempDir;

    fn temp_context(config_text: &str) -> (TempDir, HermesContext, LoadedConfig) {
        let dir = TempDir::new().unwrap();
        let context =
            HermesContext::new(dir.path()).with_hermes_home_env(Some(dir.path().join(".hermes")));
        fs::create_dir_all(context.hermes_home()).unwrap();
        let loaded = LoadedConfig {
            path: context.config_path(),
            raw: serde_yaml::from_str::<YamlValue>(config_text).unwrap(),
            config: hermes_core::HermesConfig::default(),
            warnings: Vec::new(),
        };
        (dir, context, loaded)
    }

    #[test]
    fn subscribe_persists_route() {
        let (_dir, context, loaded) = temp_context(
            "platforms:\n  webhook:\n    enabled: true\n    extra:\n      host: 0.0.0.0\n      port: 8644\n",
        );
        subscribe(
            &context,
            &loaded,
            SubscribeArgs {
                name: "my-hook".to_string(),
                prompt: "hello".to_string(),
                events: "push,issue".to_string(),
                description: "".to_string(),
                skills: "skill-a,skill-b".to_string(),
                deliver: "telegram".to_string(),
                deliver_chat_id: "123".to_string(),
                secret: "abc".to_string(),
                deliver_only: true,
            },
        )
        .unwrap();
        let subs = load_subscriptions(&context).unwrap();
        let route = subs.get("my-hook").and_then(JsonValue::as_object).unwrap();
        assert_eq!(
            route.get("secret"),
            Some(&JsonValue::String("abc".to_string()))
        );
        assert_eq!(route.get("deliver_only"), Some(&JsonValue::Bool(true)));
    }

    #[test]
    fn webhook_enablement_and_base_url_read_raw_config() {
        let (_dir, _context, loaded) = temp_context(
            "platforms:\n  webhook:\n    enabled: true\n    extra:\n      host: 0.0.0.0\n      port: 9001\n",
        );
        assert!(is_webhook_enabled(&loaded));
        assert_eq!(webhook_base_url(&loaded), "http://localhost:9001");
    }

    #[test]
    fn test_command_sends_signed_request() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut headers = Vec::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" || line.is_empty() {
                    break;
                }
                headers.push(line.trim().to_string());
            }
            let signature = headers.join("\n");
            let response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
            stream.write_all(response.as_bytes()).unwrap();
            signature
        });

        let (_dir, context, loaded) = temp_context(&format!(
            "platforms:\n  webhook:\n    enabled: true\n    extra:\n      host: 127.0.0.1\n      port: {}\n",
            addr.port()
        ));
        let mut subscriptions = BTreeMap::new();
        subscriptions.insert(
            "demo".to_string(),
            json!({"secret":"topsecret","deliver":"log"}),
        );
        save_subscriptions(&context, &subscriptions).unwrap();
        test_subscription(&context, &loaded, "demo", DEFAULT_TEST_PAYLOAD).unwrap();
        let signature = handle.join().unwrap();
        assert!(
            signature
                .to_ascii_lowercase()
                .contains("x-hub-signature-256: sha256=")
        );
    }
}
