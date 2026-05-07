use std::env;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::blocking::multipart::{Form, Part};
use serde_json::{Value, json};

use crate::tools::{ToolRuntime, tool_error, tool_result};
use crate::yuanbao::{send_yuanbao_message, send_yuanbao_message_with_media};

const REQUEST_TIMEOUT_SECS: u64 = 30;
const TELEGRAM_DEFAULT_BASE_URL: &str = "https://api.telegram.org";
const DISCORD_DEFAULT_BASE_URL: &str = "https://discord.com/api/v10";
const SLACK_DEFAULT_BASE_URL: &str = "https://slack.com/api";
const FEISHU_DEFAULT_BASE_URL: &str = "https://open.feishu.cn";
const LARK_DEFAULT_BASE_URL: &str = "https://open.larksuite.com";
const MATRIX_DEFAULT_BASE_URL: &str = "";
const YUANBAO_DEFAULT_BASE_URL: &str = "https://bot.yuanbao.tencent.com";
const SIGNAL_MAX_ATTACHMENTS_PER_MSG: usize = 32;

const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp"];
const VIDEO_EXTS: &[&str] = &["mp4", "mov", "avi", "mkv", "webm"];
const TELEGRAM_SEND_AUDIO_EXTS: &[&str] = &["mp3", "m4a"];
const TELEGRAM_VOICE_EXTS: &[&str] = &["ogg", "opus"];
const FEISHU_AUDIO_EXTS: &[&str] = &["ogg", "mp3", "wav", "m4a", "aac", "flac", "opus", "webm"];
const FEISHU_OPUS_EXTS: &[&str] = &["ogg", "opus"];
const FEISHU_MEDIA_EXTS: &[&str] = &["mp4", "mov", "avi", "m4v"];
const MATRIX_VIDEO_EXTS: &[&str] = &["mp4", "mov", "avi", "mkv", "3gp"];
const MATRIX_AUDIO_EXTS: &[&str] = &["ogg", "opus", "mp3", "wav", "m4a", "flac"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlatformKind {
    Telegram,
    Discord,
    Slack,
    Feishu,
    Matrix,
    Signal,
    Yuanbao,
}

impl PlatformKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Telegram => "telegram",
            Self::Discord => "discord",
            Self::Slack => "slack",
            Self::Feishu => "feishu",
            Self::Matrix => "matrix",
            Self::Signal => "signal",
            Self::Yuanbao => "yuanbao",
        }
    }

    fn from_name(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "telegram" => Some(Self::Telegram),
            "discord" => Some(Self::Discord),
            "slack" => Some(Self::Slack),
            "feishu" => Some(Self::Feishu),
            "matrix" => Some(Self::Matrix),
            "signal" => Some(Self::Signal),
            "yuanbao" => Some(Self::Yuanbao),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct HomeTarget {
    chat_id: String,
    thread_id: Option<String>,
    name: String,
}

#[derive(Debug, Clone)]
struct TelegramConfig {
    token: String,
    base_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct DiscordConfig {
    token: String,
    base_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct SlackConfig {
    token: String,
    base_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct FeishuConfig {
    app_id: String,
    app_secret: String,
    base_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct MatrixConfig {
    token: String,
    homeserver: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct SignalConfig {
    http_url: String,
    account: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct YuanbaoConfig {
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct MediaAttachment {
    path: PathBuf,
    is_voice: bool,
}

#[derive(Debug, Clone)]
struct ResolvedTarget {
    chat_id: String,
    thread_id: Option<String>,
    used_home_channel: bool,
}

#[derive(Debug, Clone)]
struct SentMessage {
    platform: PlatformKind,
    chat_id: String,
    message_id: Option<String>,
    thread_id: Option<String>,
    note: Option<String>,
}

#[derive(Debug, Clone)]
struct FeishuTokenEntry {
    key: String,
    token: String,
    expires_at: std::time::Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeishuMediaRoute {
    Image,
    File {
        upload_type: &'static str,
        msg_type: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixMediaKind {
    Image,
    Video,
    Audio,
    File,
}

#[derive(Debug, Clone)]
struct SendError(String);

impl Display for SendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

static FEISHU_TOKEN_CACHE: OnceLock<Mutex<Option<FeishuTokenEntry>>> = OnceLock::new();

pub fn send_message_available() -> bool {
    load_configs().has_any()
}

pub fn send_message_schema() -> Value {
    json!({
        "name": "send_message",
        "description": "Send a message to a connected messaging platform, or list the configured delivery targets. Supported in the Rust runtime: Telegram, Discord, Slack, Feishu, Matrix, Signal, and Yuanbao. When the user asks to send to a specific destination, call send_message with action='list' first if you need to inspect configured home targets or cached channel-directory entries. Human-friendly targets like slack:#engineering and discord:Guild/channel are resolved from the cached channel directory when available.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["send", "list"],
                    "description": "Use 'send' to deliver a message or 'list' to inspect configured messaging targets."
                },
                "target": {
                    "type": "string",
                    "description": "Delivery target. Format: 'platform' to use the configured home target, 'platform:chat_id' with optional 'platform:chat_id:thread_id' for thread-aware platforms, 'signal:+15551234567' or 'signal:group:<group_id>' for Signal, 'matrix:!roomid:server.org' or 'matrix:@user:server.org' for Matrix, or 'yuanbao:group:<group_code>' / 'yuanbao:direct:<account_id>' for Yuanbao."
                },
                "message": {
                    "type": "string",
                    "description": "Text to send. MEDIA:/absolute/or/relative/path tags are supported for Telegram, Discord, Slack, Feishu, Matrix, Signal, and Yuanbao. [[audio_as_voice]] marks OGG or Opus media for Telegram voice delivery."
                }
            },
            "required": []
        }
    })
}

pub fn handle_send_message(args: &Value, runtime: &ToolRuntime) -> String {
    let action = match optional_non_empty_string(args, "action") {
        Ok(value) => value.unwrap_or_else(|| "send".to_string()),
        Err(error) => return tool_error(error),
    };
    match action.as_str() {
        "list" => handle_list_targets(runtime),
        "send" => handle_send(args, runtime),
        _ => tool_error(format!("Unsupported action: {action}")),
    }
}

fn handle_list_targets(runtime: &ToolRuntime) -> String {
    let configs = load_configs();
    if !configs.has_any() {
        return tool_error(
            "No supported messaging platform is configured. Set TELEGRAM_BOT_TOKEN, DISCORD_BOT_TOKEN, SLACK_BOT_TOKEN, FEISHU_APP_ID/FEISHU_APP_SECRET, MATRIX_ACCESS_TOKEN/MATRIX_HOMESERVER, SIGNAL_HTTP_URL/SIGNAL_ACCOUNT, or YUANBAO_APP_ID/YUANBAO_APP_SECRET first.",
        );
    }

    let mut targets = Vec::new();
    let mut platforms = Vec::new();
    for kind in [
        PlatformKind::Telegram,
        PlatformKind::Discord,
        PlatformKind::Slack,
        PlatformKind::Feishu,
        PlatformKind::Matrix,
        PlatformKind::Signal,
        PlatformKind::Yuanbao,
    ] {
        if let Some(target) = configs.home_target(kind) {
            targets.push(json!({
                "platform": kind.as_str(),
                "target": format_target(kind, &target.chat_id, target.thread_id.as_deref()),
                "chat_id": target.chat_id,
                "thread_id": target.thread_id,
                "name": target.name,
                "home": true,
            }));
        }
        if configs.is_configured(kind) {
            platforms.push(json!({
                "platform": kind.as_str(),
                "has_home_channel": configs.home_target(kind).is_some(),
            }));
        }
    }

    let directory = load_channel_directory(runtime.hermes_home());
    let directory_text = format_channel_directory(&directory);

    tool_result(json!({
        "success": true,
        "targets": targets,
        "platforms": platforms,
        "directory": directory_text,
        "directory_updated_at": directory.get("updated_at").cloned().unwrap_or(Value::Null),
    }))
}

fn handle_send(args: &Value, runtime: &ToolRuntime) -> String {
    let target = match required_non_empty_string(args, "target") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let message = match required_non_empty_string(args, "message") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };

    let configs = load_configs();
    if !configs.has_any() {
        return tool_error(
            "No supported messaging platform is configured. Set TELEGRAM_BOT_TOKEN, DISCORD_BOT_TOKEN, SLACK_BOT_TOKEN, FEISHU_APP_ID/FEISHU_APP_SECRET, MATRIX_ACCESS_TOKEN/MATRIX_HOMESERVER, SIGNAL_HTTP_URL/SIGNAL_ACCOUNT, or YUANBAO_APP_ID/YUANBAO_APP_SECRET first.",
        );
    }

    let (kind, explicit_ref) = match split_platform_target(&target) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let resolved = match resolve_target(&configs, runtime, kind, explicit_ref.as_deref()) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let (media, cleaned_message) = match extract_media(&message, runtime) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if cleaned_message.is_empty() && media.is_empty() {
        return tool_error("No deliverable text or media remained after processing MEDIA tags");
    }

    let sent = match kind {
        PlatformKind::Telegram => configs
            .telegram
            .as_ref()
            .ok_or_else(|| SendError("Telegram is not configured".to_string()))
            .and_then(|config| send_telegram(config, &resolved, &cleaned_message, &media)),
        PlatformKind::Discord => configs
            .discord
            .as_ref()
            .ok_or_else(|| SendError("Discord is not configured".to_string()))
            .and_then(|config| send_discord(config, &resolved, &cleaned_message, &media)),
        PlatformKind::Slack => configs
            .slack
            .as_ref()
            .ok_or_else(|| SendError("Slack is not configured".to_string()))
            .and_then(|config| send_slack(config, &resolved, &cleaned_message, &media)),
        PlatformKind::Feishu => configs
            .feishu
            .as_ref()
            .ok_or_else(|| SendError("Feishu is not configured".to_string()))
            .and_then(|config| send_feishu(config, &resolved, &cleaned_message, &media)),
        PlatformKind::Matrix => configs
            .matrix
            .as_ref()
            .ok_or_else(|| SendError("Matrix is not configured".to_string()))
            .and_then(|config| send_matrix(config, &resolved, &cleaned_message, &media)),
        PlatformKind::Signal => configs
            .signal
            .as_ref()
            .ok_or_else(|| SendError("Signal is not configured".to_string()))
            .and_then(|config| send_signal(config, &resolved, &cleaned_message, &media)),
        PlatformKind::Yuanbao => configs
            .yuanbao
            .as_ref()
            .ok_or_else(|| SendError("Yuanbao is not configured".to_string()))
            .and_then(|config| send_yuanbao(config, &resolved, &cleaned_message, &media)),
    };

    match sent {
        Ok(sent) => tool_result(json!({
            "success": true,
            "platform": sent.platform.as_str(),
            "chat_id": sent.chat_id,
            "message_id": sent.message_id,
            "thread_id": sent.thread_id,
            "note": sent.note,
        })),
        Err(error) => tool_error(error.to_string()),
    }
}

#[derive(Debug, Clone, Default)]
struct ConfigSet {
    telegram: Option<TelegramConfig>,
    discord: Option<DiscordConfig>,
    slack: Option<SlackConfig>,
    feishu: Option<FeishuConfig>,
    matrix: Option<MatrixConfig>,
    signal: Option<SignalConfig>,
    yuanbao: Option<YuanbaoConfig>,
}

impl ConfigSet {
    fn has_any(&self) -> bool {
        self.telegram.is_some()
            || self.discord.is_some()
            || self.slack.is_some()
            || self.feishu.is_some()
            || self.matrix.is_some()
            || self.signal.is_some()
            || self.yuanbao.is_some()
    }

    fn is_configured(&self, kind: PlatformKind) -> bool {
        match kind {
            PlatformKind::Telegram => self.telegram.is_some(),
            PlatformKind::Discord => self.discord.is_some(),
            PlatformKind::Slack => self.slack.is_some(),
            PlatformKind::Feishu => self.feishu.is_some(),
            PlatformKind::Matrix => self.matrix.is_some(),
            PlatformKind::Signal => self.signal.is_some(),
            PlatformKind::Yuanbao => self.yuanbao.is_some(),
        }
    }

    fn home_target(&self, kind: PlatformKind) -> Option<HomeTarget> {
        match kind {
            PlatformKind::Telegram => self
                .telegram
                .as_ref()
                .and_then(|config| config.home.clone()),
            PlatformKind::Discord => self.discord.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::Slack => self.slack.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::Feishu => self.feishu.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::Matrix => self.matrix.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::Signal => self.signal.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::Yuanbao => self.yuanbao.as_ref().and_then(|config| config.home.clone()),
        }
    }
}

fn load_configs() -> ConfigSet {
    ConfigSet {
        telegram: load_telegram_config(),
        discord: load_discord_config(),
        slack: load_slack_config(),
        feishu: load_feishu_config(),
        matrix: load_matrix_config(),
        signal: load_signal_config(),
        yuanbao: load_yuanbao_config(),
    }
}

fn load_telegram_config() -> Option<TelegramConfig> {
    let token = env_trimmed("TELEGRAM_BOT_TOKEN")?;
    Some(TelegramConfig {
        token,
        base_url: normalize_base_url(
            env_trimmed("TELEGRAM_API_BASE_URL").as_deref(),
            TELEGRAM_DEFAULT_BASE_URL,
        ),
        home: load_home_target(
            "TELEGRAM_HOME_CHANNEL",
            "TELEGRAM_HOME_CHANNEL_THREAD_ID",
            "TELEGRAM_HOME_CHANNEL_NAME",
        ),
    })
}

fn load_discord_config() -> Option<DiscordConfig> {
    let token = env_trimmed("DISCORD_BOT_TOKEN")?;
    Some(DiscordConfig {
        token,
        base_url: normalize_base_url(
            env_trimmed("DISCORD_API_BASE_URL").as_deref(),
            DISCORD_DEFAULT_BASE_URL,
        ),
        home: load_home_target(
            "DISCORD_HOME_CHANNEL",
            "DISCORD_HOME_CHANNEL_THREAD_ID",
            "DISCORD_HOME_CHANNEL_NAME",
        ),
    })
}

fn load_slack_config() -> Option<SlackConfig> {
    let token = env_trimmed("SLACK_BOT_TOKEN")?;
    Some(SlackConfig {
        token,
        base_url: normalize_base_url(
            env_trimmed("SLACK_API_BASE_URL").as_deref(),
            SLACK_DEFAULT_BASE_URL,
        ),
        home: load_home_target(
            "SLACK_HOME_CHANNEL",
            "SLACK_HOME_CHANNEL_THREAD_ID",
            "SLACK_HOME_CHANNEL_NAME",
        ),
    })
}

fn load_feishu_config() -> Option<FeishuConfig> {
    let app_id = env_trimmed("FEISHU_APP_ID")?;
    let app_secret = env_trimmed("FEISHU_APP_SECRET")?;
    let domain = env::var("FEISHU_DOMAIN").unwrap_or_else(|_| "feishu".to_string());
    let base_url = normalize_feishu_base_url(&domain).ok()?;
    Some(FeishuConfig {
        app_id,
        app_secret,
        base_url,
        home: load_home_target(
            "FEISHU_HOME_CHANNEL",
            "FEISHU_HOME_CHANNEL_THREAD_ID",
            "FEISHU_HOME_CHANNEL_NAME",
        ),
    })
}

fn load_matrix_config() -> Option<MatrixConfig> {
    let token = env_trimmed("MATRIX_ACCESS_TOKEN")?;
    let homeserver = normalize_base_url(
        env_trimmed("MATRIX_HOMESERVER").as_deref(),
        MATRIX_DEFAULT_BASE_URL,
    );
    if homeserver.is_empty() {
        return None;
    }
    Some(MatrixConfig {
        token,
        homeserver,
        home: load_home_target(
            "MATRIX_HOME_ROOM",
            "MATRIX_HOME_ROOM_THREAD_ID",
            "MATRIX_HOME_ROOM_NAME",
        ),
    })
}

fn load_signal_config() -> Option<SignalConfig> {
    let http_url = env_trimmed("SIGNAL_HTTP_URL")?;
    let account = env_trimmed("SIGNAL_ACCOUNT")?;
    Some(SignalConfig {
        http_url: normalize_base_url(Some(&http_url), ""),
        account,
        home: env_trimmed("SIGNAL_HOME_CHANNEL").map(|chat_id| HomeTarget {
            chat_id,
            thread_id: None,
            name: env_trimmed("SIGNAL_HOME_CHANNEL_NAME").unwrap_or_else(|| "Home".to_string()),
        }),
    })
}

fn load_yuanbao_config() -> Option<YuanbaoConfig> {
    let _app_key = env_trimmed("YUANBAO_APP_ID").or_else(|| env_trimmed("YUANBAO_APP_KEY"))?;
    let _app_secret = env_trimmed("YUANBAO_APP_SECRET")?;
    let _base_url = normalize_base_url(
        env_trimmed("YUANBAO_API_DOMAIN").as_deref(),
        YUANBAO_DEFAULT_BASE_URL,
    );
    Some(YuanbaoConfig {
        home: env_trimmed("YUANBAO_HOME_CHANNEL").and_then(|chat_id| {
            if validate_yuanbao_target(&chat_id).is_err() {
                return None;
            }
            Some(HomeTarget {
                chat_id,
                thread_id: None,
                name: env_trimmed("YUANBAO_HOME_CHANNEL_NAME")
                    .unwrap_or_else(|| "Home".to_string()),
            })
        }),
    })
}

fn load_home_target(chat_key: &str, thread_key: &str, name_key: &str) -> Option<HomeTarget> {
    let chat_id = env_trimmed(chat_key)?;
    Some(HomeTarget {
        chat_id,
        thread_id: env_trimmed(thread_key),
        name: env_trimmed(name_key).unwrap_or_else(|| "Home".to_string()),
    })
}

fn env_trimmed(key: &str) -> Option<String> {
    env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn normalize_base_url(raw: Option<&str>, default: &str) -> String {
    let value = raw.unwrap_or(default).trim();
    value.trim_end_matches('/').to_string()
}

fn normalize_feishu_base_url(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("feishu") {
        return Ok(FEISHU_DEFAULT_BASE_URL.to_string());
    }
    if trimmed.eq_ignore_ascii_case("lark") {
        return Ok(LARK_DEFAULT_BASE_URL.to_string());
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let url =
            Url::parse(trimmed).map_err(|error| format!("invalid FEISHU_DOMAIN URL: {error}"))?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err("FEISHU_DOMAIN URL must include http or https and a host".to_string());
        }
        return Ok(url.to_string().trim_end_matches('/').to_string());
    }
    Err("FEISHU_DOMAIN must be 'feishu', 'lark', or a full http(s) base URL".to_string())
}

fn split_platform_target(raw: &str) -> Result<(PlatformKind, Option<String>), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("target must not be empty".to_string());
    }
    let mut parts = trimmed.splitn(2, ':');
    let platform = parts.next().unwrap_or_default();
    let kind = PlatformKind::from_name(platform)
        .ok_or_else(|| format!("Unsupported platform '{platform}'. Supported Rust send_message platforms: telegram, discord, slack, feishu, matrix, signal, yuanbao."))?;
    Ok((kind, parts.next().map(str::to_string)))
}

fn resolve_target(
    configs: &ConfigSet,
    runtime: &ToolRuntime,
    kind: PlatformKind,
    explicit_ref: Option<&str>,
) -> Result<ResolvedTarget, String> {
    let configured = configs.is_configured(kind);
    if !configured {
        return Err(format!("Platform '{}' is not configured.", kind.as_str()));
    }
    let home = configs.home_target(kind);
    match explicit_ref {
        None => {
            let Some(home) = home else {
                return Err(format!(
                    "No home channel is configured for {}. Use an explicit target like '{}:<chat_id>' or set the *_HOME_CHANNEL env var.",
                    kind.as_str(),
                    kind.as_str()
                ));
            };
            Ok(ResolvedTarget {
                chat_id: home.chat_id,
                thread_id: home.thread_id,
                used_home_channel: true,
            })
        }
        Some(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Err("target reference must not be empty".to_string());
            }
            let resolved = if let Some(value) = parse_explicit_target(kind, trimmed)? {
                value
            } else {
                let Some(resolved_ref) =
                    resolve_channel_name(runtime.hermes_home(), kind.as_str(), trimmed)
                else {
                    return Err(format!(
                        "Could not resolve '{trimmed}' on {}. Use send_message(action='list') to inspect cached targets or provide an explicit id.",
                        kind.as_str()
                    ));
                };
                parse_explicit_target(kind, &resolved_ref)?.ok_or_else(|| {
                    format!(
                        "Resolved '{}' on {} to an unsupported target '{}'.",
                        trimmed,
                        kind.as_str(),
                        resolved_ref
                    )
                })?
            };
            if kind == PlatformKind::Signal && resolved.thread_id.is_some() {
                return Err(
                    "Signal thread targets are not supported in the Rust send_message runtime."
                        .to_string(),
                );
            }
            Ok(resolved)
        }
    }
}

fn parse_explicit_target(kind: PlatformKind, raw: &str) -> Result<Option<ResolvedTarget>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("target reference must not be empty".to_string());
    }
    if kind == PlatformKind::Yuanbao {
        if trimmed.starts_with("group:") || trimmed.starts_with("direct:") {
            validate_yuanbao_target(trimmed)?;
            return Ok(Some(ResolvedTarget {
                chat_id: trimmed.to_string(),
                thread_id: None,
                used_home_channel: false,
            }));
        }
        if trimmed.chars().all(|ch| ch.is_ascii_digit()) {
            let chat_id = format!("group:{trimmed}");
            validate_yuanbao_target(&chat_id)?;
            return Ok(Some(ResolvedTarget {
                chat_id,
                thread_id: None,
                used_home_channel: false,
            }));
        }
        return Ok(None);
    }

    if kind == PlatformKind::Matrix {
        if (trimmed.starts_with('!') || trimmed.starts_with('@')) && trimmed.contains(':') {
            validate_target_component("chat_id", trimmed)?;
            return Ok(Some(ResolvedTarget {
                chat_id: trimmed.to_string(),
                thread_id: None,
                used_home_channel: false,
            }));
        }
        return Ok(None);
    }

    if kind == PlatformKind::Signal {
        if let Some(group_id) = trimmed.strip_prefix("group:") {
            validate_target_component("group_id", group_id)?;
            return Ok(Some(ResolvedTarget {
                chat_id: trimmed.to_string(),
                thread_id: None,
                used_home_channel: false,
            }));
        }
        if is_signal_e164(trimmed) {
            validate_target_component("chat_id", trimmed)?;
            return Ok(Some(ResolvedTarget {
                chat_id: trimmed.to_string(),
                thread_id: None,
                used_home_channel: false,
            }));
        }
        return Ok(None);
    }

    let segments = trimmed.split(':').collect::<Vec<_>>();
    if segments.is_empty() || segments[0].trim().is_empty() || segments.len() > 2 {
        return Ok(None);
    }
    let chat_id = segments[0].trim();
    let thread_id = segments
        .get(1)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty());
    let explicit = match kind {
        PlatformKind::Telegram | PlatformKind::Discord => {
            is_numeric_chat_id(chat_id) && thread_id.is_none_or(is_positive_numeric_id)
        }
        PlatformKind::Slack => is_slack_chat_id(chat_id) && thread_id.is_none_or(is_thread_id),
        PlatformKind::Feishu => is_feishu_chat_id(chat_id) && thread_id.is_none_or(is_thread_id),
        PlatformKind::Matrix | PlatformKind::Signal | PlatformKind::Yuanbao => false,
    };
    if !explicit {
        return Ok(None);
    }
    validate_target_component("chat_id", chat_id)?;
    if let Some(thread_id) = thread_id {
        validate_target_component("thread_id", thread_id)?;
    }
    Ok(Some(ResolvedTarget {
        chat_id: chat_id.to_string(),
        thread_id: thread_id.map(ToOwned::to_owned),
        used_home_channel: false,
    }))
}

fn validate_target_component(key: &str, raw: &str) -> Result<(), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    if trimmed.len() > 256
        || trimmed
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace())
    {
        return Err(format!("{key} contains unsupported characters"));
    }
    Ok(())
}

fn is_numeric_chat_id(raw: &str) -> bool {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix('-') {
        !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit())
    } else {
        !trimmed.is_empty() && trimmed.chars().all(|ch| ch.is_ascii_digit())
    }
}

fn is_positive_numeric_id(raw: &str) -> bool {
    !raw.is_empty() && raw.chars().all(|ch| ch.is_ascii_digit())
}

fn is_slack_chat_id(raw: &str) -> bool {
    let mut chars = raw.chars();
    let Some(prefix) = chars.next() else {
        return false;
    };
    matches!(prefix, 'C' | 'G' | 'D')
        && raw.len() >= 9
        && chars.all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit())
}

fn is_feishu_chat_id(raw: &str) -> bool {
    ["oc_", "ou_", "on_", "chat_", "open_"]
        .into_iter()
        .any(|prefix| raw.starts_with(prefix))
}

fn is_signal_e164(raw: &str) -> bool {
    let Some(rest) = raw.strip_prefix('+') else {
        return false;
    };
    let len = rest.len();
    (7..=15).contains(&len) && rest.chars().all(|ch| ch.is_ascii_digit())
}

fn is_thread_id(raw: &str) -> bool {
    !raw.is_empty()
        && raw
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

fn validate_yuanbao_target(raw: &str) -> Result<(), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("chat_id must not be empty".to_string());
    }
    if let Some(value) = trimmed.strip_prefix("group:") {
        return validate_target_component("group_code", value);
    }
    if let Some(value) = trimmed.strip_prefix("direct:") {
        return validate_target_component("account_id", value);
    }
    Err("Yuanbao targets must be 'group:<group_code>' or 'direct:<account_id>'".to_string())
}

fn load_channel_directory(hermes_home: &Path) -> Value {
    let path = hermes_home.join("channel_directory.json");
    let Ok(content) = fs::read_to_string(path) else {
        return json!({ "updated_at": null, "platforms": {} });
    };
    serde_json::from_str(&content)
        .unwrap_or_else(|_| json!({ "updated_at": null, "platforms": {} }))
}

fn format_channel_directory(directory: &Value) -> String {
    let Some(platforms) = directory.get("platforms").and_then(Value::as_object) else {
        return "No messaging platforms connected or no channels discovered yet.".to_string();
    };
    if !platforms
        .values()
        .any(|value| value.as_array().is_some_and(|items| !items.is_empty()))
    {
        return "No messaging platforms connected or no channels discovered yet.".to_string();
    }
    let mut lines = vec!["Available messaging targets:".to_string(), String::new()];
    let mut keys = platforms.keys().cloned().collect::<Vec<_>>();
    keys.sort();
    for platform in keys {
        let Some(channels) = platforms.get(&platform).and_then(Value::as_array) else {
            continue;
        };
        if channels.is_empty() {
            continue;
        }
        lines.push(format!("{}:", title_case(&platform)));
        for channel in channels {
            let name = channel_display_name(&platform, channel);
            lines.push(format!("  {}:{}", platform, name));
        }
        lines.push(String::new());
    }
    lines.push(r#"Use these as the "target" parameter when sending."#.to_string());
    lines.push(r#"Bare platform name (e.g. "telegram") sends to home channel."#.to_string());
    lines.join("\n")
}

fn title_case(raw: &str) -> String {
    let mut chars = raw.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    format!("{}{}", first.to_ascii_uppercase(), chars.as_str())
}

fn resolve_channel_name(hermes_home: &Path, platform_name: &str, name: &str) -> Option<String> {
    let directory = load_channel_directory(hermes_home);
    let channels = directory
        .get("platforms")
        .and_then(Value::as_object)?
        .get(platform_name)?
        .as_array()?;
    if channels.is_empty() {
        return None;
    }

    let raw = name.trim();
    for channel in channels {
        if channel.get("id").and_then(Value::as_str) == Some(raw) {
            return Some(raw.to_string());
        }
    }

    let query = normalize_channel_query(name);
    for channel in channels {
        if normalize_channel_query(
            channel
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        ) == query
        {
            return channel
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
        }
        if normalize_channel_query(&channel_display_name(platform_name, channel)) == query {
            return channel
                .get("id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
        }
    }

    if let Some((guild_part, channel_part)) = query.rsplit_once('/') {
        for channel in channels {
            let guild = channel
                .get("guild")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();
            if guild == guild_part
                && normalize_channel_query(
                    channel
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                ) == channel_part
            {
                return channel
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
            }
        }
    }

    let matches = channels
        .iter()
        .filter(|channel| {
            normalize_channel_query(
                channel
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
            .starts_with(&query)
        })
        .collect::<Vec<_>>();
    if matches.len() == 1 {
        return matches[0]
            .get("id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
    }
    None
}

fn normalize_channel_query(raw: &str) -> String {
    raw.trim_start_matches('#').trim().to_ascii_lowercase()
}

fn channel_display_name(platform_name: &str, channel: &Value) -> String {
    let name = channel
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if platform_name == "discord" && channel.get("guild").and_then(Value::as_str).is_some() {
        return format!("#{name}");
    }
    if platform_name != "discord"
        && let Some(kind) = channel.get("type").and_then(Value::as_str)
        && !kind.trim().is_empty()
    {
        return format!("{name} ({kind})");
    }
    name
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{key} is required"))
}

fn optional_non_empty_string(args: &Value, key: &str) -> Result<Option<String>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Some(_) => Err(format!("{key} must be a string")),
    }
}

fn extract_media(
    content: &str,
    runtime: &ToolRuntime,
) -> Result<(Vec<MediaAttachment>, String), String> {
    let has_voice_tag = content.contains("[[audio_as_voice]]");
    let mut media = Vec::new();
    let mut cleaned_lines = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[[audio_as_voice]]" {
            continue;
        }
        if let Some(path) = parse_media_line(trimmed) {
            let resolved = runtime
                .resolve_path(&path)
                .map_err(|error| format!("Invalid MEDIA path '{path}': {error}"))?;
            if !resolved.is_file() {
                return Err(format!("Media file not found: {}", resolved.display()));
            }
            media.push(MediaAttachment {
                path: resolved,
                is_voice: has_voice_tag,
            });
            continue;
        }
        cleaned_lines.push(line);
    }

    let cleaned = cleaned_lines
        .join("\n")
        .replace("[[audio_as_voice]]", "")
        .trim()
        .to_string();

    Ok((media, cleaned))
}

fn parse_media_line(line: &str) -> Option<String> {
    let mut trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(stripped) = strip_matching_wrapper(trimmed) {
        trimmed = stripped;
    }
    let rest = trimmed.strip_prefix("MEDIA:")?.trim();
    if rest.is_empty() {
        return None;
    }
    let mut path = rest
        .trim_matches(|ch| matches!(ch, '`' | '"' | '\''))
        .trim()
        .to_string();
    while path.ends_with(|ch: char| matches!(ch, ',' | ';' | ':' | ')' | '}' | ']')) {
        path.pop();
    }
    (!path.is_empty()).then_some(path)
}

fn strip_matching_wrapper(value: &str) -> Option<&str> {
    let bytes = value.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let first = bytes[0] as char;
    let last = bytes[bytes.len() - 1] as char;
    if matches!(first, '`' | '"' | '\'') && first == last {
        Some(&value[1..value.len() - 1])
    } else {
        None
    }
}

fn send_telegram(
    config: &TelegramConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    let client = http_client()?;
    let mut last_message_id = None;

    if !message.is_empty() {
        let url = format!("{}/bot{}/sendMessage", config.base_url, config.token);
        let mut payload = serde_json::Map::new();
        payload.insert("chat_id".to_string(), Value::String(target.chat_id.clone()));
        payload.insert("text".to_string(), Value::String(message.to_string()));
        if let Some(thread_id) = target.thread_id.as_ref() {
            payload.insert(
                "message_thread_id".to_string(),
                Value::String(thread_id.clone()),
            );
        }
        let response = client
            .post(url)
            .json(&Value::Object(payload))
            .send()
            .map_err(|error| SendError(format!("Telegram send failed: {error}")))?;
        let body = parse_json_response(response, "Telegram send failed")?;
        if !body.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            return Err(SendError(format!(
                "Telegram send failed: {}",
                body.get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
            )));
        }
        last_message_id = body.pointer("/result/message_id").and_then(value_as_string);
    }

    for attachment in media {
        let (method, field) = telegram_upload_endpoint(attachment);
        let url = format!("{}/bot{}/{}", config.base_url, config.token, method);
        let file_name = attachment
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("attachment.bin")
            .to_string();
        let bytes = fs::read(&attachment.path).map_err(|error| {
            SendError(format!(
                "Reading media {} failed: {error}",
                attachment.path.display()
            ))
        })?;
        let part = Part::bytes(bytes).file_name(file_name);
        let mut form = Form::new()
            .text("chat_id", target.chat_id.clone())
            .part(field.to_string(), part);
        if let Some(thread_id) = target.thread_id.as_ref() {
            form = form.text("message_thread_id", thread_id.clone());
        }

        let response = client
            .post(url)
            .multipart(form)
            .send()
            .map_err(|error| SendError(format!("Telegram media send failed: {error}")))?;
        let body = parse_json_response(response, "Telegram media send failed")?;
        if !body.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            return Err(SendError(format!(
                "Telegram media send failed: {}",
                body.get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
            )));
        }
        last_message_id = body.pointer("/result/message_id").and_then(value_as_string);
    }

    Ok(SentMessage {
        platform: PlatformKind::Telegram,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: target.thread_id.clone(),
        note: target.used_home_channel.then(|| {
            format!(
                "Sent to telegram home channel (chat_id: {})",
                target.chat_id
            )
        }),
    })
}

fn telegram_upload_endpoint(attachment: &MediaAttachment) -> (&'static str, &'static str) {
    let ext = file_extension(&attachment.path);
    if IMAGE_EXTS.contains(&ext.as_str()) {
        return ("sendPhoto", "photo");
    }
    if VIDEO_EXTS.contains(&ext.as_str()) {
        return ("sendVideo", "video");
    }
    if TELEGRAM_VOICE_EXTS.contains(&ext.as_str()) && attachment.is_voice {
        return ("sendVoice", "voice");
    }
    if TELEGRAM_SEND_AUDIO_EXTS.contains(&ext.as_str()) {
        return ("sendAudio", "audio");
    }
    ("sendDocument", "document")
}

fn send_discord(
    config: &DiscordConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    let client = http_client()?;
    let destination = target
        .thread_id
        .as_deref()
        .unwrap_or(target.chat_id.as_str());
    let url = format!("{}/channels/{destination}/messages", config.base_url);
    let mut last_message_id = None;

    if !message.is_empty() {
        let response = client
            .post(&url)
            .header("Authorization", format!("Bot {}", config.token))
            .header("Content-Type", "application/json")
            .json(&json!({ "content": message }))
            .send()
            .map_err(|error| SendError(format!("Discord send failed: {error}")))?;
        let body = parse_json_response(response, "Discord send failed")?;
        last_message_id = body.get("id").and_then(value_as_string);
    }

    for attachment in media {
        let file_name = attachment
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("attachment.bin")
            .to_string();
        let bytes = fs::read(&attachment.path).map_err(|error| {
            SendError(format!(
                "Reading media {} failed: {error}",
                attachment.path.display()
            ))
        })?;
        let form = Form::new().part("files[0]", Part::bytes(bytes).file_name(file_name));
        let response = client
            .post(&url)
            .header("Authorization", format!("Bot {}", config.token))
            .multipart(form)
            .send()
            .map_err(|error| SendError(format!("Discord media send failed: {error}")))?;
        let body = parse_json_response(response, "Discord media send failed")?;
        last_message_id = body.get("id").and_then(value_as_string);
    }

    Ok(SentMessage {
        platform: PlatformKind::Discord,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: target.thread_id.clone(),
        note: target
            .used_home_channel
            .then(|| format!("Sent to discord home channel (chat_id: {})", target.chat_id)),
    })
}

fn send_slack(
    config: &SlackConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    let client = http_client()?;
    let mut last_message_id = None;

    if !message.is_empty() {
        let url = format!("{}/chat.postMessage", config.base_url);
        let mut payload = serde_json::Map::new();
        payload.insert("channel".to_string(), Value::String(target.chat_id.clone()));
        payload.insert("text".to_string(), Value::String(message.to_string()));
        payload.insert("mrkdwn".to_string(), Value::Bool(true));
        if let Some(thread_id) = target.thread_id.as_ref() {
            payload.insert("thread_ts".to_string(), Value::String(thread_id.clone()));
        }
        let response = client
            .post(url)
            .bearer_auth(&config.token)
            .header("Content-Type", "application/json")
            .json(&Value::Object(payload))
            .send()
            .map_err(|error| SendError(format!("Slack send failed: {error}")))?;
        let body = parse_json_response(response, "Slack send failed")?;
        if !body.get("ok").and_then(Value::as_bool).unwrap_or(false) {
            return Err(SendError(format!(
                "Slack send failed: {}",
                body.get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
            )));
        }
        last_message_id = body.get("ts").and_then(value_as_string);
    }

    for attachment in media {
        let file_name = attachment
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("attachment.bin")
            .to_string();
        let bytes = fs::read(&attachment.path).map_err(|error| {
            SendError(format!(
                "Reading media {} failed: {error}",
                attachment.path.display()
            ))
        })?;
        last_message_id = Some(slack_upload_file(
            &client, config, target, &file_name, &bytes,
        )?);
    }

    Ok(SentMessage {
        platform: PlatformKind::Slack,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: target.thread_id.clone(),
        note: target
            .used_home_channel
            .then(|| format!("Sent to slack home channel (chat_id: {})", target.chat_id)),
    })
}

fn slack_upload_file(
    client: &Client,
    config: &SlackConfig,
    target: &ResolvedTarget,
    file_name: &str,
    bytes: &[u8],
) -> Result<String, SendError> {
    let init_response = client
        .post(format!("{}/files.getUploadURLExternal", config.base_url))
        .bearer_auth(&config.token)
        .form(&vec![
            ("filename".to_string(), file_name.to_string()),
            ("length".to_string(), bytes.len().to_string()),
        ])
        .send()
        .map_err(|error| SendError(format!("Slack media upload init failed: {error}")))?;
    let init_body = parse_json_response(init_response, "Slack media upload init failed")?;
    if !init_body
        .get("ok")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(SendError(format!(
            "Slack media upload init failed: {}",
            init_body
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        )));
    }
    let upload_url = init_body
        .get("upload_url")
        .and_then(value_as_string)
        .ok_or_else(|| {
            SendError("Slack media upload init failed: missing upload_url".to_string())
        })?;
    let file_id = init_body
        .get("file_id")
        .and_then(value_as_string)
        .ok_or_else(|| SendError("Slack media upload init failed: missing file_id".to_string()))?;

    let upload_response = client
        .post(&upload_url)
        .body(bytes.to_vec())
        .send()
        .map_err(|error| SendError(format!("Slack media upload failed: {error}")))?;
    if !upload_response.status().is_success() {
        let status = upload_response.status();
        let body = upload_response.text().unwrap_or_default();
        return Err(SendError(format!(
            "Slack media upload failed: status={} body={}",
            status.as_u16(),
            body
        )));
    }

    let mut params = vec![
        (
            "files".to_string(),
            json!([{ "id": file_id, "title": file_name }]).to_string(),
        ),
        ("channel_id".to_string(), target.chat_id.clone()),
    ];
    if let Some(thread_id) = target.thread_id.as_ref() {
        params.push(("thread_ts".to_string(), thread_id.clone()));
    }
    let complete_response = client
        .post(format!("{}/files.completeUploadExternal", config.base_url))
        .bearer_auth(&config.token)
        .form(&params)
        .send()
        .map_err(|error| SendError(format!("Slack media finalize failed: {error}")))?;
    let complete_body = parse_json_response(complete_response, "Slack media finalize failed")?;
    if !complete_body
        .get("ok")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(SendError(format!(
            "Slack media finalize failed: {}",
            complete_body
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        )));
    }
    Ok(complete_body
        .pointer("/files/0/id")
        .and_then(value_as_string)
        .unwrap_or(file_id))
}

fn send_feishu(
    config: &FeishuConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    let token = feishu_access_token(config)?;
    let client = http_client()?;
    let mut last_message_id = None;

    if !message.is_empty() {
        last_message_id = Some(feishu_send_message(
            &client,
            config,
            &token,
            target,
            "text",
            json!({ "text": message }).to_string(),
            "Feishu send failed",
        )?);
    }

    for attachment in media {
        let route = feishu_media_route(attachment);
        let file_name = attachment
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("attachment.bin")
            .to_string();
        let bytes = fs::read(&attachment.path).map_err(|error| {
            SendError(format!(
                "Reading media {} failed: {error}",
                attachment.path.display()
            ))
        })?;
        let message_id = match route {
            FeishuMediaRoute::Image => {
                let image_key = feishu_upload_image(&client, config, &token, &file_name, &bytes)?;
                feishu_send_message(
                    &client,
                    config,
                    &token,
                    target,
                    "image",
                    json!({ "image_key": image_key }).to_string(),
                    "Feishu media send failed",
                )?
            }
            FeishuMediaRoute::File {
                upload_type,
                msg_type,
            } => {
                let file_key =
                    feishu_upload_file(&client, config, &token, upload_type, &file_name, &bytes)?;
                feishu_send_message(
                    &client,
                    config,
                    &token,
                    target,
                    msg_type,
                    json!({ "file_key": file_key }).to_string(),
                    "Feishu media send failed",
                )?
            }
        };
        last_message_id = Some(message_id);
    }

    Ok(SentMessage {
        platform: PlatformKind::Feishu,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: target.thread_id.clone(),
        note: target
            .used_home_channel
            .then(|| format!("Sent to feishu home channel (chat_id: {})", target.chat_id)),
    })
}

fn send_matrix(
    config: &MatrixConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    let client = http_client()?;
    let mut last_message_id = None;

    if !message.is_empty() {
        last_message_id = Some(send_matrix_message(
            &client,
            config,
            target,
            json!({
                "msgtype": "m.text",
                "body": message,
            }),
            "Matrix send failed",
        )?);
    }

    for attachment in media {
        let file_name = attachment
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("attachment.bin")
            .to_string();
        let bytes = fs::read(&attachment.path).map_err(|error| {
            SendError(format!(
                "Reading media {} failed: {error}",
                attachment.path.display()
            ))
        })?;
        let mime_type = guess_matrix_mime_type(&attachment.path, &bytes);
        let content_uri = matrix_upload_media(&client, config, &file_name, &mime_type, &bytes)?;
        let message_id = send_matrix_message(
            &client,
            config,
            target,
            matrix_media_payload(
                matrix_media_kind(attachment),
                &file_name,
                &mime_type,
                bytes.len(),
                &content_uri,
            ),
            "Matrix media send failed",
        )?;
        last_message_id = Some(message_id);
    }

    Ok(SentMessage {
        platform: PlatformKind::Matrix,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: target.thread_id.clone(),
        note: target
            .used_home_channel
            .then(|| format!("Sent to matrix home room (chat_id: {})", target.chat_id)),
    })
}

fn send_matrix_message(
    client: &Client,
    config: &MatrixConfig,
    target: &ResolvedTarget,
    mut payload: Value,
    context: &str,
) -> Result<String, SendError> {
    if let Some(thread_id) = target.thread_id.as_ref() {
        payload["m.relates_to"] = json!({
            "rel_type": "m.thread",
            "event_id": thread_id,
            "is_falling_back": true,
        });
    }
    let txn_id = format!("rust-send-{}", unique_suffix());
    let encoded_room: String =
        url::form_urlencoded::byte_serialize(target.chat_id.as_bytes()).collect();
    let url = format!(
        "{}/_matrix/client/v3/rooms/{}/send/m.room.message/{}",
        config.homeserver, encoded_room, txn_id
    );
    let response = client
        .put(url)
        .bearer_auth(&config.token)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|error| SendError(format!("{context}: {error}")))?;
    let body = parse_json_response(response, context)?;
    body.get("event_id")
        .and_then(value_as_string)
        .ok_or_else(|| SendError(format!("{context}: missing event_id")))
}

fn matrix_upload_media(
    client: &Client,
    config: &MatrixConfig,
    file_name: &str,
    mime_type: &str,
    bytes: &[u8],
) -> Result<String, SendError> {
    let mut url =
        Url::parse(&format!("{}/_matrix/media/v3/upload", config.homeserver)).map_err(|error| {
            SendError(format!(
                "Matrix media upload failed: invalid homeserver URL: {error}"
            ))
        })?;
    url.query_pairs_mut().append_pair("filename", file_name);
    let response = client
        .post(url)
        .bearer_auth(&config.token)
        .header("Content-Type", mime_type)
        .body(bytes.to_vec())
        .send()
        .map_err(|error| SendError(format!("Matrix media upload failed: {error}")))?;
    let body = parse_json_response(response, "Matrix media upload failed")?;
    body.get("content_uri")
        .and_then(value_as_string)
        .ok_or_else(|| SendError("Matrix media upload failed: missing content_uri".to_string()))
}

fn matrix_media_payload(
    kind: MatrixMediaKind,
    file_name: &str,
    mime_type: &str,
    size: usize,
    content_uri: &str,
) -> Value {
    let msgtype = match kind {
        MatrixMediaKind::Image => "m.image",
        MatrixMediaKind::Video => "m.video",
        MatrixMediaKind::Audio => "m.audio",
        MatrixMediaKind::File => "m.file",
    };
    let mut payload = json!({
        "msgtype": msgtype,
        "body": file_name,
        "url": content_uri,
        "info": {
            "mimetype": mime_type,
            "size": size,
        },
    });
    if kind == MatrixMediaKind::Audio {
        payload["org.matrix.msc3245.voice"] = json!({});
    }
    payload
}

fn matrix_media_kind(attachment: &MediaAttachment) -> MatrixMediaKind {
    let ext = file_extension(&attachment.path);
    if IMAGE_EXTS.contains(&ext.as_str()) {
        return MatrixMediaKind::Image;
    }
    if MATRIX_VIDEO_EXTS.contains(&ext.as_str()) {
        return MatrixMediaKind::Video;
    }
    if MATRIX_AUDIO_EXTS.contains(&ext.as_str()) {
        return MatrixMediaKind::Audio;
    }
    MatrixMediaKind::File
}

fn guess_matrix_mime_type(path: &Path, bytes: &[u8]) -> String {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return "image/png".to_string();
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return "image/jpeg".to_string();
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return "image/gif".to_string();
    }
    if bytes.starts_with(b"BM") {
        return "image/bmp".to_string();
    }
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return "image/webp".to_string();
    }
    match file_extension(path).as_str() {
        "jpg" | "jpeg" => "image/jpeg".to_string(),
        "png" => "image/png".to_string(),
        "gif" => "image/gif".to_string(),
        "webp" => "image/webp".to_string(),
        "bmp" => "image/bmp".to_string(),
        "mp4" => "video/mp4".to_string(),
        "mov" => "video/quicktime".to_string(),
        "avi" => "video/x-msvideo".to_string(),
        "mkv" => "video/x-matroska".to_string(),
        "3gp" => "video/3gpp".to_string(),
        "ogg" | "opus" => "audio/ogg".to_string(),
        "mp3" => "audio/mpeg".to_string(),
        "wav" => "audio/wav".to_string(),
        "m4a" => "audio/mp4".to_string(),
        "flac" => "audio/flac".to_string(),
        "pdf" => "application/pdf".to_string(),
        "doc" => "application/msword".to_string(),
        "docx" => {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string()
        }
        "xls" => "application/vnd.ms-excel".to_string(),
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string(),
        "ppt" => "application/vnd.ms-powerpoint".to_string(),
        "pptx" => {
            "application/vnd.openxmlformats-officedocument.presentationml.presentation".to_string()
        }
        "txt" => "text/plain".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

fn send_signal(
    config: &SignalConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    if target.thread_id.is_some() {
        return Err(SendError(
            "Signal thread targets are not supported in the Rust send_message runtime.".to_string(),
        ));
    }

    let attachment_paths = media
        .iter()
        .map(|attachment| attachment.path.display().to_string())
        .collect::<Vec<_>>();
    let attachment_batches = if attachment_paths.is_empty() {
        vec![Vec::new()]
    } else {
        attachment_paths
            .chunks(SIGNAL_MAX_ATTACHMENTS_PER_MSG)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>()
    };

    for (index, batch) in attachment_batches.iter().enumerate() {
        let batch_message = if index == 0 { message } else { "" };
        let client =
            http_client_with_timeout(Duration::from_secs(signal_batch_timeout_secs(batch.len())))?;
        let body = signal_send_batch(
            &client,
            config,
            target,
            batch_message,
            batch,
            index + 1,
            attachment_batches.len(),
        )?;
        if let Some(error) = body.get("error").filter(|value| !value.is_null()) {
            return Err(SendError(format!(
                "Signal RPC error on batch {}/{}: {}",
                index + 1,
                attachment_batches.len(),
                signal_error_text(error)
            )));
        }
    }

    Ok(SentMessage {
        platform: PlatformKind::Signal,
        chat_id: target.chat_id.clone(),
        message_id: None,
        thread_id: None,
        note: target
            .used_home_channel
            .then(|| format!("Sent to signal home channel (chat_id: {})", target.chat_id)),
    })
}

fn signal_send_batch(
    client: &Client,
    config: &SignalConfig,
    target: &ResolvedTarget,
    message: &str,
    attachments: &[String],
    batch_index: usize,
    batch_total: usize,
) -> Result<Value, SendError> {
    let mut params = json!({
        "account": config.account,
        "message": message,
    });
    if target.chat_id.starts_with("group:") {
        params["groupId"] = json!(target.chat_id.trim_start_matches("group:"));
    } else {
        params["recipient"] = json!([target.chat_id]);
    }
    if !attachments.is_empty() {
        params["attachments"] = json!(attachments);
    }
    let payload = json!({
        "jsonrpc": "2.0",
        "method": "send",
        "params": params,
        "id": format!("send_{}_{}", unique_suffix(), batch_index),
    });
    let response = client
        .post(format!("{}/api/v1/rpc", config.http_url))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|error| {
            SendError(format!(
                "Signal send failed on batch {batch_index}/{batch_total}: {error}"
            ))
        })?;
    parse_json_response(
        response,
        &format!("Signal send failed on batch {batch_index}/{batch_total}"),
    )
}

fn signal_batch_timeout_secs(attachment_count: usize) -> u64 {
    if attachment_count == 0 {
        return REQUEST_TIMEOUT_SECS;
    }
    (attachment_count as u64 * 5).max(60)
}

fn signal_error_text(error: &Value) -> String {
    if let Some(message) = error.get("message").and_then(Value::as_str) {
        let code = error.get("code").and_then(value_as_i64);
        return match code {
            Some(code) => format!("code={code} message={message}"),
            None => message.to_string(),
        };
    }
    if let Some(text) = error.as_str() {
        return text.to_string();
    }
    error.to_string()
}

fn feishu_send_message(
    client: &Client,
    config: &FeishuConfig,
    token: &str,
    target: &ResolvedTarget,
    msg_type: &str,
    content: String,
    context: &str,
) -> Result<String, SendError> {
    let response = if let Some(thread_id) = target.thread_id.as_ref() {
        let url = format!(
            "{}/open-apis/im/v1/messages/{}/reply",
            config.base_url, thread_id
        );
        client
            .post(url)
            .bearer_auth(token)
            .header("Content-Type", "application/json")
            .json(&json!({
                "content": content,
                "msg_type": msg_type,
                "reply_in_thread": true,
                "uuid": format!("rust-send-{}", unique_suffix()),
            }))
            .send()
            .map_err(|error| SendError(format!("{context}: {error}")))?
    } else {
        let url = format!(
            "{}/open-apis/im/v1/messages?receive_id_type={}",
            config.base_url,
            feishu_receive_id_type(&target.chat_id),
        );
        client
            .post(url)
            .bearer_auth(token)
            .header("Content-Type", "application/json")
            .json(&json!({
                "receive_id": target.chat_id,
                "msg_type": msg_type,
                "content": content,
                "uuid": format!("rust-send-{}", unique_suffix()),
            }))
            .send()
            .map_err(|error| SendError(format!("{context}: {error}")))?
    };
    let body = parse_json_response(response, context)?;
    let code = body.get("code").and_then(value_as_i64).unwrap_or(0);
    if code != 0 {
        return Err(SendError(format!(
            "{context}: code={code} msg={}",
            body.get("msg")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        )));
    }
    body.pointer("/data/message_id")
        .and_then(value_as_string)
        .ok_or_else(|| SendError(format!("{context}: missing message_id")))
}

fn feishu_upload_image(
    client: &Client,
    config: &FeishuConfig,
    token: &str,
    file_name: &str,
    bytes: &[u8],
) -> Result<String, SendError> {
    let part = Part::bytes(bytes.to_vec()).file_name(file_name.to_string());
    let response = client
        .post(format!("{}/open-apis/im/v1/images", config.base_url))
        .bearer_auth(token)
        .multipart(
            Form::new()
                .text("image_type", "message")
                .part("image", part),
        )
        .send()
        .map_err(|error| SendError(format!("Feishu image upload failed: {error}")))?;
    let body = parse_json_response(response, "Feishu image upload failed")?;
    let code = body.get("code").and_then(value_as_i64).unwrap_or(0);
    if code != 0 {
        return Err(SendError(format!(
            "Feishu image upload failed: code={code} msg={}",
            body.get("msg")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        )));
    }
    body.pointer("/data/image_key")
        .and_then(value_as_string)
        .ok_or_else(|| SendError("Feishu image upload failed: missing image_key".to_string()))
}

fn feishu_upload_file(
    client: &Client,
    config: &FeishuConfig,
    token: &str,
    file_type: &str,
    file_name: &str,
    bytes: &[u8],
) -> Result<String, SendError> {
    let part = Part::bytes(bytes.to_vec()).file_name(file_name.to_string());
    let response = client
        .post(format!("{}/open-apis/im/v1/files", config.base_url))
        .bearer_auth(token)
        .multipart(
            Form::new()
                .text("file_type", file_type.to_string())
                .text("file_name", file_name.to_string())
                .part("file", part),
        )
        .send()
        .map_err(|error| SendError(format!("Feishu file upload failed: {error}")))?;
    let body = parse_json_response(response, "Feishu file upload failed")?;
    let code = body.get("code").and_then(value_as_i64).unwrap_or(0);
    if code != 0 {
        return Err(SendError(format!(
            "Feishu file upload failed: code={code} msg={}",
            body.get("msg")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        )));
    }
    body.pointer("/data/file_key")
        .and_then(value_as_string)
        .ok_or_else(|| SendError("Feishu file upload failed: missing file_key".to_string()))
}

fn feishu_receive_id_type(chat_id: &str) -> &'static str {
    if chat_id.starts_with("ou_") {
        "open_id"
    } else {
        "chat_id"
    }
}

fn feishu_media_route(attachment: &MediaAttachment) -> FeishuMediaRoute {
    let ext = file_extension(&attachment.path);
    if IMAGE_EXTS.contains(&ext.as_str()) {
        return FeishuMediaRoute::Image;
    }
    if FEISHU_OPUS_EXTS.contains(&ext.as_str()) {
        return FeishuMediaRoute::File {
            upload_type: "opus",
            msg_type: "audio",
        };
    }
    if FEISHU_MEDIA_EXTS.contains(&ext.as_str()) {
        return FeishuMediaRoute::File {
            upload_type: "mp4",
            msg_type: "media",
        };
    }
    let upload_type = match ext.as_str() {
        "pdf" => "pdf",
        "doc" | "docx" => "doc",
        "xls" | "xlsx" => "xls",
        "ppt" | "pptx" => "ppt",
        _ => "stream",
    };
    let msg_type = if FEISHU_AUDIO_EXTS.contains(&ext.as_str()) && attachment.is_voice {
        "audio"
    } else {
        "file"
    };
    FeishuMediaRoute::File {
        upload_type,
        msg_type: if upload_type == "stream" && msg_type == "audio" {
            "file"
        } else {
            msg_type
        },
    }
}

fn send_yuanbao(
    _config: &YuanbaoConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    let message_id = if media.is_empty() {
        send_yuanbao_message(&target.chat_id, message)
    } else {
        send_yuanbao_message_with_media(
            &target.chat_id,
            message,
            &media
                .iter()
                .map(|item| item.path.clone())
                .collect::<Vec<_>>(),
            None,
        )
    }
    .map_err(|error| SendError(format!("Yuanbao send failed: {error}")))?;
    Ok(SentMessage {
        platform: PlatformKind::Yuanbao,
        chat_id: target.chat_id.clone(),
        message_id: Some(message_id),
        thread_id: None,
        note: target
            .used_home_channel
            .then(|| format!("Sent to yuanbao home channel (chat_id: {})", target.chat_id)),
    })
}

fn feishu_access_token(config: &FeishuConfig) -> Result<String, SendError> {
    let cache_key = format!(
        "{}|{}|{}",
        config.base_url, config.app_id, config.app_secret
    );
    let cache = FEISHU_TOKEN_CACHE.get_or_init(|| Mutex::new(None));
    if let Some(entry) = cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
        && entry.key == cache_key
        && std::time::Instant::now() < entry.expires_at
    {
        return Ok(entry.token);
    }

    let client = http_client()?;
    let response = client
        .post(format!(
            "{}/open-apis/auth/v3/tenant_access_token/internal",
            config.base_url
        ))
        .json(&json!({
            "app_id": config.app_id,
            "app_secret": config.app_secret,
        }))
        .send()
        .map_err(|error| SendError(format!("Feishu token request failed: {error}")))?;
    let body = parse_json_response(response, "Feishu token request failed")?;
    let code = body.get("code").and_then(value_as_i64).unwrap_or(0);
    if code != 0 {
        return Err(SendError(format!(
            "Feishu token request failed: code={code} msg={}",
            body.get("msg")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        )));
    }
    let token = body
        .get("tenant_access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            SendError("Feishu token response did not contain tenant_access_token".to_string())
        })?
        .to_string();
    let ttl = body
        .get("expire")
        .and_then(value_as_i64)
        .or_else(|| body.get("expires_in").and_then(value_as_i64))
        .unwrap_or(7200)
        .max(0) as u64;
    let entry = FeishuTokenEntry {
        key: cache_key,
        token: token.clone(),
        expires_at: std::time::Instant::now() + Duration::from_secs(ttl.saturating_sub(60).max(1)),
    };
    *cache.lock().unwrap_or_else(|error| error.into_inner()) = Some(entry);
    Ok(token)
}

fn http_client() -> Result<Client, SendError> {
    http_client_with_timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
}

fn http_client_with_timeout(timeout: Duration) -> Result<Client, SendError> {
    Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|error| SendError(format!("failed to build HTTP client: {error}")))
}

fn parse_json_response(
    response: reqwest::blocking::Response,
    context: &str,
) -> Result<Value, SendError> {
    let status = response.status();
    let raw = response
        .text()
        .map_err(|error| SendError(format!("{context}: failed to read response body: {error}")))?;
    if !status.is_success() {
        return Err(SendError(format!(
            "{context}: HTTP {}: {}",
            status.as_u16(),
            raw
        )));
    }
    serde_json::from_str(&raw)
        .map_err(|error| SendError(format!("{context}: invalid JSON response: {error}")))
}

fn file_extension(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn format_target(kind: PlatformKind, chat_id: &str, thread_id: Option<&str>) -> String {
    if kind == PlatformKind::Matrix {
        return format!("{}:{}", kind.as_str(), chat_id);
    }
    match thread_id {
        Some(thread_id) if !thread_id.trim().is_empty() => {
            format!("{}:{}:{}", kind.as_str(), chat_id, thread_id)
        }
        _ => format!("{}:{}", kind.as_str(), chat_id),
    }
}

fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then_some(trimmed.to_string())
        }
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn value_as_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number.as_i64(),
        Value::String(text) => text.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use tempfile::TempDir;
    use tungstenite::Message;

    static TEST_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn test_env_lock() -> &'static Mutex<()> {
        TEST_ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
        test_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn with_env_var(key: &str, value: Option<&str>) {
        match value {
            Some(value) => unsafe { env::set_var(key, value) },
            None => unsafe { env::remove_var(key) },
        }
    }

    fn clear_feishu_cache() {
        if let Some(cache) = FEISHU_TOKEN_CACHE.get() {
            *cache.lock().unwrap_or_else(|error| error.into_inner()) = None;
        }
    }

    fn runtime_for(temp: &TempDir) -> ToolRuntime {
        ToolRuntime::new(temp.path()).with_hermes_home(temp.path())
    }

    fn mock_server<F>(request_count: usize, handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: Fn(usize, String, String) -> (u16, String) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            for index in 0..request_count {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 4096];
                loop {
                    let read = stream.read(&mut buffer).unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let header_end = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|value| value + 4)
                    .unwrap_or(request.len());
                let headers = String::from_utf8_lossy(&request[..header_end]).to_string();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let mut body_bytes = request[header_end..].to_vec();
                while body_bytes.len() < content_length {
                    let read = stream.read(&mut buffer).unwrap();
                    if read == 0 {
                        break;
                    }
                    body_bytes.extend_from_slice(&buffer[..read]);
                }
                let body = String::from_utf8_lossy(&body_bytes[..content_length]).to_string();
                let (status, response_body) = handler(index, headers, body);
                let response = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });
        (format!("http://{}", addr), join)
    }

    fn mock_ws_server<F>(handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: FnOnce(tungstenite::WebSocket<std::net::TcpStream>) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let websocket = tungstenite::accept(stream).unwrap();
            handler(websocket);
        });
        (format!("ws://{}", addr), join)
    }

    fn encode_varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                return out;
            }
        }
    }

    fn push_varint_field(buffer: &mut Vec<u8>, number: u32, value: u64) {
        buffer.extend_from_slice(&encode_varint((u64::from(number) << 3) | 0));
        buffer.extend_from_slice(&encode_varint(value));
    }

    fn push_bytes_field(buffer: &mut Vec<u8>, number: u32, value: &[u8]) {
        buffer.extend_from_slice(&encode_varint((u64::from(number) << 3) | 2));
        buffer.extend_from_slice(&encode_varint(value.len() as u64));
        buffer.extend_from_slice(value);
    }

    fn push_string_field(buffer: &mut Vec<u8>, number: u32, value: &str) {
        push_bytes_field(buffer, number, value.as_bytes());
    }

    fn decode_varint(bytes: &[u8], cursor: &mut usize) -> u64 {
        let mut shift = 0u32;
        let mut value = 0u64;
        loop {
            let byte = bytes[*cursor];
            *cursor += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return value;
            }
            shift += 7;
        }
    }

    fn field_bytes(bytes: &[u8], wanted: u32) -> Option<Vec<u8>> {
        let mut cursor = 0usize;
        while cursor < bytes.len() {
            let key = decode_varint(bytes, &mut cursor);
            let number = (key >> 3) as u32;
            let wire = (key & 0x07) as u8;
            match wire {
                0 => {
                    let _ = decode_varint(bytes, &mut cursor);
                }
                2 => {
                    let len = decode_varint(bytes, &mut cursor) as usize;
                    let value = bytes[cursor..cursor + len].to_vec();
                    cursor += len;
                    if number == wanted {
                        return Some(value);
                    }
                }
                _ => return None,
            }
        }
        None
    }

    fn field_string(bytes: &[u8], wanted: u32) -> Option<String> {
        let value = field_bytes(bytes, wanted)?;
        String::from_utf8(value).ok()
    }

    fn frame_cmd_and_msg_id(frame: &[u8]) -> (String, String) {
        let head = field_bytes(frame, 1).unwrap();
        (
            field_string(&head, 2).unwrap_or_default(),
            field_string(&head, 4).unwrap_or_default(),
        )
    }

    fn auth_bind_response(msg_id: &str) -> Vec<u8> {
        let mut data = Vec::new();
        push_varint_field(&mut data, 1, 0);
        push_string_field(&mut data, 3, "conn-1");

        let mut head = Vec::new();
        push_varint_field(&mut head, 1, 1);
        push_string_field(&mut head, 2, "auth-bind");
        push_varint_field(&mut head, 3, 1);
        push_string_field(&mut head, 4, msg_id);
        push_string_field(&mut head, 5, "conn_access");

        let mut out = Vec::new();
        push_bytes_field(&mut out, 1, &head);
        push_bytes_field(&mut out, 2, &data);
        out
    }

    fn ok_group_send_response(msg_id: &str) -> Vec<u8> {
        let mut head = Vec::new();
        push_varint_field(&mut head, 1, 1);
        push_string_field(&mut head, 2, "send_group_message");
        push_varint_field(&mut head, 3, 2);
        push_string_field(&mut head, 4, msg_id);
        push_string_field(&mut head, 5, "yuanbao_openclaw_proxy");

        let mut out = Vec::new();
        push_bytes_field(&mut out, 1, &head);
        out
    }

    #[test]
    fn list_targets_reports_configured_platforms() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        with_env_var("DISCORD_BOT_TOKEN", Some("discord-token"));
        with_env_var("DISCORD_HOME_CHANNEL", Some("12345"));
        with_env_var("DISCORD_HOME_CHANNEL_NAME", Some("Ops"));
        with_env_var("SLACK_BOT_TOKEN", Some("slack-token"));
        with_env_var("SLACK_HOME_CHANNEL", None);
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["targets"][0]["target"], json!("discord:12345"));
        assert!(
            parsed["platforms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["platform"] == "slack")
        );
    }

    #[test]
    fn list_targets_reports_yuanbao_home_channel() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        with_env_var("YUANBAO_APP_ID", Some("yb-app"));
        with_env_var("YUANBAO_APP_SECRET", Some("yb-secret"));
        with_env_var("YUANBAO_HOME_CHANNEL", Some("group:home123"));
        with_env_var("YUANBAO_HOME_CHANNEL_NAME", Some("Pai Home"));
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["targets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["target"] == "yuanbao:group:home123")
        );
        assert!(
            parsed["platforms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["platform"] == "yuanbao")
        );
    }

    #[test]
    fn list_targets_includes_cached_directory_text() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        fs::write(
            temp.path().join("channel_directory.json"),
            json!({
                "updated_at": "2026-05-07T12:00:00",
                "platforms": {
                    "slack": [
                        { "id": "CENG12345", "name": "engineering", "type": "channel" }
                    ],
                    "discord": [
                        { "id": "123456789", "name": "bot-home", "guild": "Nous", "type": "channel" }
                    ]
                }
            })
            .to_string(),
        )
        .unwrap();
        with_env_var("SLACK_BOT_TOKEN", Some("slack-token"));
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        let directory = parsed["directory"].as_str().unwrap();
        assert!(directory.contains("Slack:"));
        assert!(directory.contains("slack:engineering (channel)"));
        assert!(directory.contains("discord:#bot-home"));
        assert_eq!(parsed["directory_updated_at"], json!("2026-05-07T12:00:00"));
    }

    #[test]
    fn list_targets_preserves_matrix_home_thread_separately() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        with_env_var("MATRIX_ACCESS_TOKEN", Some("matrix-token"));
        with_env_var("MATRIX_HOMESERVER", Some("https://matrix.example"));
        with_env_var("MATRIX_HOME_ROOM", Some("!roomid:example.org"));
        with_env_var(
            "MATRIX_HOME_ROOM_THREAD_ID",
            Some("$thread_root:example.org"),
        );
        with_env_var("MATRIX_HOME_ROOM_NAME", Some("Ops Thread"));
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        let matrix = parsed["targets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["platform"] == "matrix")
            .unwrap();
        assert_eq!(matrix["target"], json!("matrix:!roomid:example.org"));
        assert_eq!(matrix["chat_id"], json!("!roomid:example.org"));
        assert_eq!(matrix["thread_id"], json!("$thread_root:example.org"));
        assert_eq!(matrix["name"], json!("Ops Thread"));
    }

    #[test]
    fn resolves_slack_named_target_from_channel_directory() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        fs::write(
            temp.path().join("channel_directory.json"),
            json!({
                "updated_at": "2026-05-07T12:00:00",
                "platforms": {
                    "slack": [
                        { "id": "CENG12345", "name": "engineering", "type": "channel" }
                    ]
                }
            })
            .to_string(),
        )
        .unwrap();
        let (base_url, join) = mock_server(1, move |_index, headers, body| {
            assert!(headers.starts_with("POST /chat.postMessage "));
            let payload: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(payload["channel"], json!("CENG12345"));
            assert_eq!(payload["text"], json!("deploy now"));
            (
                200,
                json!({ "ok": true, "ts": "1710000000.000100" }).to_string(),
            )
        });
        with_env_var("SLACK_BOT_TOKEN", Some("slack-token"));
        with_env_var("SLACK_API_BASE_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "slack:#engineering",
                "message": "deploy now",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("slack"));
        assert_eq!(parsed["chat_id"], json!("CENG12345"));
        join.join().unwrap();
    }

    #[test]
    fn extracts_media_and_voice_tag() {
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let file_path = temp.path().join("clip.ogg");
        fs::write(&file_path, b"voice").unwrap();
        let content = format!("[[audio_as_voice]]\nMEDIA:{}\nhello", file_path.display());
        let (media, cleaned) = extract_media(&content, &runtime).unwrap();
        assert_eq!(media.len(), 1);
        assert!(media[0].is_voice);
        assert_eq!(cleaned, "hello");
    }

    #[test]
    fn sends_discord_text_and_media() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("image.png");
        fs::write(&media_path, b"png-bytes").unwrap();
        let (base_url, join) = mock_server(2, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /channels/123/messages "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["content"], json!("hello"));
                (200, json!({ "id": "msg-1" }).to_string())
            }
            1 => {
                assert!(headers.starts_with("POST /channels/123/messages "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("content-type: multipart/form-data;")
                );
                (200, json!({ "id": "msg-2" }).to_string())
            }
            _ => unreachable!(),
        });

        with_env_var("DISCORD_BOT_TOKEN", Some("discord-token"));
        with_env_var("DISCORD_API_BASE_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "discord:123",
                "message": format!("hello\nMEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("discord"));
        assert_eq!(parsed["message_id"], json!("msg-2"));
        join.join().unwrap();
    }

    #[test]
    fn sends_telegram_voice_media() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("clip.ogg");
        fs::write(&media_path, b"ogg-bytes").unwrap();
        let (base_url, join) = mock_server(2, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /bottelegram-token/sendMessage "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["text"], json!("hello"));
                (
                    200,
                    json!({ "ok": true, "result": { "message_id": 41 } }).to_string(),
                )
            }
            1 => {
                assert!(headers.starts_with("POST /bottelegram-token/sendVoice "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("content-type: multipart/form-data;")
                );
                (
                    200,
                    json!({ "ok": true, "result": { "message_id": 42 } }).to_string(),
                )
            }
            _ => unreachable!(),
        });

        with_env_var("TELEGRAM_BOT_TOKEN", Some("telegram-token"));
        with_env_var("TELEGRAM_API_BASE_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "telegram:98765:12",
                "message": format!("hello\n[[audio_as_voice]]\nMEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["thread_id"], json!("12"));
        assert_eq!(parsed["message_id"], json!("42"));
        join.join().unwrap();
    }

    #[test]
    fn sends_feishu_text_via_token_flow() {
        let _guard = acquire_test_lock();
        clear_feishu_cache();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (base_url, join) = mock_server(2, move |index, headers, body| match index {
            0 => {
                assert!(
                    headers.starts_with("POST /open-apis/auth/v3/tenant_access_token/internal ")
                );
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["app_id"], json!("cli_test"));
                (
                    200,
                    json!({ "code": 0, "tenant_access_token": "tenant-token", "expire": 7200 })
                        .to_string(),
                )
            }
            1 => {
                assert!(
                    headers.starts_with("POST /open-apis/im/v1/messages?receive_id_type=chat_id ")
                );
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer tenant-token")
                );
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["receive_id"], json!("oc_home"));
                let inner: Value =
                    serde_json::from_str(payload["content"].as_str().unwrap()).unwrap();
                assert_eq!(inner["text"], json!("hi there"));
                (
                    200,
                    json!({ "code": 0, "data": { "message_id": "om_1" } }).to_string(),
                )
            }
            _ => unreachable!(),
        });

        with_env_var("FEISHU_APP_ID", Some("cli_test"));
        with_env_var("FEISHU_APP_SECRET", Some("secret_test"));
        with_env_var("FEISHU_DOMAIN", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "feishu:oc_home",
                "message": "hi there",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["message_id"], json!("om_1"));
        join.join().unwrap();
    }

    #[test]
    fn feishu_media_routing_matches_python_adapter() {
        let audio = MediaAttachment {
            path: PathBuf::from("voice.ogg"),
            is_voice: true,
        };
        assert_eq!(
            feishu_media_route(&audio),
            FeishuMediaRoute::File {
                upload_type: "opus",
                msg_type: "audio",
            }
        );

        let video = MediaAttachment {
            path: PathBuf::from("clip.mp4"),
            is_voice: false,
        };
        assert_eq!(
            feishu_media_route(&video),
            FeishuMediaRoute::File {
                upload_type: "mp4",
                msg_type: "media",
            }
        );

        let doc = MediaAttachment {
            path: PathBuf::from("spec.pdf"),
            is_voice: false,
        };
        assert_eq!(
            feishu_media_route(&doc),
            FeishuMediaRoute::File {
                upload_type: "pdf",
                msg_type: "file",
            }
        );

        let song = MediaAttachment {
            path: PathBuf::from("song.mp3"),
            is_voice: false,
        };
        assert_eq!(
            feishu_media_route(&song),
            FeishuMediaRoute::File {
                upload_type: "stream",
                msg_type: "file",
            }
        );
    }

    #[test]
    fn sends_feishu_media_via_upload_and_message_flow() {
        let _guard = acquire_test_lock();
        clear_feishu_cache();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("proof.png");
        fs::write(&media_path, b"png-bytes").unwrap();
        let (base_url, join) = mock_server(3, move |index, headers, body| match index {
            0 => {
                assert!(
                    headers.starts_with("POST /open-apis/auth/v3/tenant_access_token/internal ")
                );
                (
                    200,
                    json!({ "code": 0, "tenant_access_token": "tenant-token", "expire": 7200 })
                        .to_string(),
                )
            }
            1 => {
                assert!(headers.starts_with("POST /open-apis/im/v1/images "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer tenant-token")
                );
                assert!(body.contains("name=\"image_type\""));
                assert!(body.contains("name=\"image\""));
                (
                    200,
                    json!({ "code": 0, "data": { "image_key": "img_123" } }).to_string(),
                )
            }
            2 => {
                assert!(
                    headers.starts_with("POST /open-apis/im/v1/messages?receive_id_type=chat_id ")
                );
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["receive_id"], json!("oc_media"));
                assert_eq!(payload["msg_type"], json!("image"));
                let inner: Value =
                    serde_json::from_str(payload["content"].as_str().unwrap()).unwrap();
                assert_eq!(inner["image_key"], json!("img_123"));
                (
                    200,
                    json!({ "code": 0, "data": { "message_id": "om_media_1" } }).to_string(),
                )
            }
            _ => unreachable!(),
        });

        with_env_var("FEISHU_APP_ID", Some("cli_test"));
        with_env_var("FEISHU_APP_SECRET", Some("secret_test"));
        with_env_var("FEISHU_DOMAIN", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "feishu:oc_media",
                "message": format!("MEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["message_id"], json!("om_media_1"));
        join.join().unwrap();
    }

    #[test]
    fn sends_matrix_text_via_client_server_api() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (base_url, join) = mock_server(1, move |_index, headers, body| {
            assert!(headers.starts_with(
                "PUT /_matrix/client/v3/rooms/%21roomid%3Aexample.org/send/m.room.message/"
            ));
            assert!(
                headers
                    .to_ascii_lowercase()
                    .contains("authorization: bearer matrix-token")
            );
            let payload: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(payload["msgtype"], json!("m.text"));
            assert_eq!(payload["body"], json!("matrix hello"));
            (200, json!({ "event_id": "$event123" }).to_string())
        });

        with_env_var("MATRIX_ACCESS_TOKEN", Some("matrix-token"));
        with_env_var("MATRIX_HOMESERVER", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "matrix:!roomid:example.org",
                "message": "matrix hello",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("matrix"));
        assert_eq!(parsed["chat_id"], json!("!roomid:example.org"));
        assert_eq!(parsed["message_id"], json!("$event123"));
        join.join().unwrap();
    }

    #[test]
    fn matrix_media_routing_matches_python_adapter() {
        let image = MediaAttachment {
            path: PathBuf::from("proof.png"),
            is_voice: false,
        };
        assert_eq!(matrix_media_kind(&image), MatrixMediaKind::Image);

        let video = MediaAttachment {
            path: PathBuf::from("clip.3gp"),
            is_voice: false,
        };
        assert_eq!(matrix_media_kind(&video), MatrixMediaKind::Video);

        let audio = MediaAttachment {
            path: PathBuf::from("voice.mp3"),
            is_voice: false,
        };
        assert_eq!(matrix_media_kind(&audio), MatrixMediaKind::Audio);

        let file = MediaAttachment {
            path: PathBuf::from("notes.pdf"),
            is_voice: false,
        };
        assert_eq!(matrix_media_kind(&file), MatrixMediaKind::File);
    }

    #[test]
    fn sends_matrix_media_via_upload_and_message_flow() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("proof.png");
        fs::write(&media_path, b"png-bytes").unwrap();
        let (base_url, join) = mock_server(3, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with(
                    "PUT /_matrix/client/v3/rooms/%21roomid%3Aexample.org/send/m.room.message/"
                ));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["msgtype"], json!("m.text"));
                assert_eq!(payload["body"], json!("matrix hello"));
                (200, json!({ "event_id": "$event-text" }).to_string())
            }
            1 => {
                assert!(headers.starts_with("POST /_matrix/media/v3/upload?filename=proof.png "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer matrix-token")
                );
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("content-type: image/png")
                );
                assert_eq!(body, "png-bytes");
                (
                    200,
                    json!({ "content_uri": "mxc://example.org/media123" }).to_string(),
                )
            }
            2 => {
                assert!(headers.starts_with(
                    "PUT /_matrix/client/v3/rooms/%21roomid%3Aexample.org/send/m.room.message/"
                ));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["msgtype"], json!("m.image"));
                assert_eq!(payload["body"], json!("proof.png"));
                assert_eq!(payload["url"], json!("mxc://example.org/media123"));
                assert_eq!(payload["info"]["mimetype"], json!("image/png"));
                assert_eq!(payload["info"]["size"], json!(9));
                (200, json!({ "event_id": "$event-media" }).to_string())
            }
            _ => unreachable!(),
        });
        with_env_var("MATRIX_ACCESS_TOKEN", Some("matrix-token"));
        with_env_var("MATRIX_HOMESERVER", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "matrix:!roomid:example.org",
                "message": format!("matrix hello\nMEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("matrix"));
        assert_eq!(parsed["message_id"], json!("$event-media"));
        join.join().unwrap();
    }

    #[test]
    fn sends_matrix_audio_as_voice_event() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("voice.ogg");
        fs::write(&media_path, b"voice-bytes").unwrap();
        let (base_url, join) = mock_server(2, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /_matrix/media/v3/upload?filename=voice.ogg "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("content-type: audio/ogg")
                );
                assert_eq!(body, "voice-bytes");
                (
                    200,
                    json!({ "content_uri": "mxc://example.org/voice456" }).to_string(),
                )
            }
            1 => {
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["msgtype"], json!("m.audio"));
                assert_eq!(payload["body"], json!("voice.ogg"));
                assert_eq!(payload["url"], json!("mxc://example.org/voice456"));
                assert_eq!(payload["info"]["mimetype"], json!("audio/ogg"));
                assert_eq!(payload["info"]["size"], json!(11));
                assert_eq!(payload["org.matrix.msc3245.voice"], json!({}));
                (200, json!({ "event_id": "$event-voice" }).to_string())
            }
            _ => unreachable!(),
        });
        with_env_var("MATRIX_ACCESS_TOKEN", Some("matrix-token"));
        with_env_var("MATRIX_HOMESERVER", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "matrix:!roomid:example.org",
                "message": format!("MEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["message_id"], json!("$event-voice"));
        join.join().unwrap();
    }

    #[test]
    fn sends_matrix_home_thread_for_text_and_media() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("proof.png");
        fs::write(&media_path, b"png-bytes").unwrap();
        let (base_url, join) = mock_server(3, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with(
                    "PUT /_matrix/client/v3/rooms/%21roomid%3Aexample.org/send/m.room.message/"
                ));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["msgtype"], json!("m.text"));
                assert_eq!(payload["body"], json!("thread hello"));
                assert_eq!(payload["m.relates_to"]["rel_type"], json!("m.thread"));
                assert_eq!(
                    payload["m.relates_to"]["event_id"],
                    json!("$thread_root:example.org")
                );
                assert_eq!(payload["m.relates_to"]["is_falling_back"], json!(true));
                (200, json!({ "event_id": "$event-thread-text" }).to_string())
            }
            1 => {
                assert!(headers.starts_with("POST /_matrix/media/v3/upload?filename=proof.png "));
                (
                    200,
                    json!({ "content_uri": "mxc://example.org/media-threaded" }).to_string(),
                )
            }
            2 => {
                assert!(headers.starts_with(
                    "PUT /_matrix/client/v3/rooms/%21roomid%3Aexample.org/send/m.room.message/"
                ));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["msgtype"], json!("m.image"));
                assert_eq!(payload["url"], json!("mxc://example.org/media-threaded"));
                assert_eq!(payload["m.relates_to"]["rel_type"], json!("m.thread"));
                assert_eq!(
                    payload["m.relates_to"]["event_id"],
                    json!("$thread_root:example.org")
                );
                assert_eq!(payload["m.relates_to"]["is_falling_back"], json!(true));
                (
                    200,
                    json!({ "event_id": "$event-thread-media" }).to_string(),
                )
            }
            _ => unreachable!(),
        });

        with_env_var("MATRIX_ACCESS_TOKEN", Some("matrix-token"));
        with_env_var("MATRIX_HOMESERVER", Some(&base_url));
        with_env_var("MATRIX_HOME_ROOM", Some("!roomid:example.org"));
        with_env_var(
            "MATRIX_HOME_ROOM_THREAD_ID",
            Some("$thread_root:example.org"),
        );
        let result = handle_send(
            &json!({
                "target": "matrix",
                "message": format!("thread hello\nMEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("matrix"));
        assert_eq!(parsed["chat_id"], json!("!roomid:example.org"));
        assert_eq!(parsed["thread_id"], json!("$thread_root:example.org"));
        assert_eq!(parsed["message_id"], json!("$event-thread-media"));
        join.join().unwrap();
    }

    #[test]
    fn sends_signal_text_via_json_rpc() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (base_url, join) = mock_server(1, move |_index, headers, body| {
            assert!(headers.starts_with("POST /api/v1/rpc "));
            let payload: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(payload["jsonrpc"], json!("2.0"));
            assert_eq!(payload["method"], json!("send"));
            assert_eq!(payload["params"]["account"], json!("+15550000000"));
            assert_eq!(payload["params"]["message"], json!("signal hello"));
            assert_eq!(payload["params"]["recipient"], json!(["+15551234567"]));
            assert!(payload["params"].get("attachments").is_none());
            (
                200,
                json!({ "jsonrpc": "2.0", "result": { "timestamp": 1710000000 } }).to_string(),
            )
        });

        with_env_var("SIGNAL_HTTP_URL", Some(&base_url));
        with_env_var("SIGNAL_ACCOUNT", Some("+15550000000"));
        let result = handle_send(
            &json!({
                "target": "signal:+15551234567",
                "message": "signal hello",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("signal"));
        assert_eq!(parsed["chat_id"], json!("+15551234567"));
        join.join().unwrap();
    }

    #[test]
    fn sends_signal_group_media_in_batches() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_paths = (0..33)
            .map(|index| {
                let path = temp.path().join(format!("image-{index}.png"));
                fs::write(&path, format!("png-{index}")).unwrap();
                path
            })
            .collect::<Vec<_>>();
        let (base_url, join) = mock_server(2, move |index, headers, body| {
            assert!(headers.starts_with("POST /api/v1/rpc "));
            let payload: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(payload["method"], json!("send"));
            assert_eq!(payload["params"]["account"], json!("+15550000000"));
            assert_eq!(payload["params"]["groupId"], json!("group-123"));
            let attachments = payload["params"]["attachments"].as_array().unwrap();
            match index {
                0 => {
                    assert_eq!(payload["params"]["message"], json!("caption"));
                    assert_eq!(attachments.len(), 32);
                }
                1 => {
                    assert_eq!(payload["params"]["message"], json!(""));
                    assert_eq!(attachments.len(), 1);
                }
                _ => unreachable!(),
            }
            (
                200,
                json!({ "jsonrpc": "2.0", "result": { "timestamp": 1710000000 + index } })
                    .to_string(),
            )
        });

        with_env_var("SIGNAL_HTTP_URL", Some(&base_url));
        with_env_var("SIGNAL_ACCOUNT", Some("+15550000000"));
        let message = format!(
            "caption\n{}",
            media_paths
                .iter()
                .map(|path| format!("MEDIA:{}", path.display()))
                .collect::<Vec<_>>()
                .join("\n")
        );
        let result = handle_send(
            &json!({
                "target": "signal:group:group-123",
                "message": message,
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("signal"));
        assert_eq!(parsed["chat_id"], json!("group:group-123"));
        join.join().unwrap();
    }

    #[test]
    fn sends_yuanbao_text_via_native_client() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (http_base, http_join) = mock_server(1, |_index, _headers, _body| {
            (
                200,
                json!({
                    "code": 0,
                    "data": {
                        "token": "token-1",
                        "bot_id": "bot-1",
                        "duration": 3600
                    }
                })
                .to_string(),
            )
        });
        let (ws_url, ws_join) = mock_ws_server(|mut websocket| {
            let auth = websocket.read().unwrap().into_data();
            let (auth_cmd, auth_msg_id) = frame_cmd_and_msg_id(&auth);
            assert_eq!(auth_cmd, "auth-bind");
            websocket
                .send(Message::Binary(auth_bind_response(&auth_msg_id).into()))
                .unwrap();

            let send = websocket.read().unwrap().into_data();
            let (send_cmd, send_msg_id) = frame_cmd_and_msg_id(&send);
            assert_eq!(send_cmd, "send_group_message");
            websocket
                .send(Message::Binary(ok_group_send_response(&send_msg_id).into()))
                .unwrap();
        });

        with_env_var("YUANBAO_APP_ID", Some("yb-app"));
        with_env_var("YUANBAO_APP_SECRET", Some("yb-secret"));
        with_env_var("YUANBAO_API_DOMAIN", Some(&http_base));
        with_env_var("YUANBAO_WS_URL", Some(&ws_url));

        let result = handle_send(
            &json!({
                "target": "yuanbao:group:123456",
                "message": "hello pai",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("yuanbao"));
        assert_eq!(parsed["chat_id"], json!("group:123456"));
        assert!(parsed["message_id"].as_str().unwrap().starts_with("grp_"));

        http_join.join().unwrap();
        ws_join.join().unwrap();
    }

    #[test]
    fn sends_slack_media_via_external_upload_flow() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("proof.txt");
        fs::write(&media_path, b"slack-bytes").unwrap();
        let (upload_base, upload_join) = mock_server(1, move |_index, headers, body| {
            assert!(headers.starts_with("POST /upload/1 "));
            assert_eq!(body, "slack-bytes");
            (200, "{}".to_string())
        });
        let upload_url = format!("{upload_base}/upload/1");
        let (base_url, join) = mock_server(3, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /chat.postMessage "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer slack-token")
                );
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["channel"], json!("C12345678"));
                assert_eq!(payload["text"], json!("hello slack"));
                assert_eq!(payload["thread_ts"], json!("1710000000.000100"));
                (
                    200,
                    json!({ "ok": true, "ts": "1710000000.000101" }).to_string(),
                )
            }
            1 => {
                assert!(headers.starts_with("POST /files.getUploadURLExternal "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer slack-token")
                );
                let params = url::form_urlencoded::parse(body.as_bytes())
                    .into_owned()
                    .collect::<std::collections::HashMap<_, _>>();
                assert_eq!(params.get("filename"), Some(&"proof.txt".to_string()));
                assert_eq!(params.get("length"), Some(&"11".to_string()));
                (
                    200,
                    json!({
                        "ok": true,
                        "upload_url": upload_url,
                        "file_id": "F123"
                    })
                    .to_string(),
                )
            }
            2 => {
                assert!(headers.starts_with("POST /files.completeUploadExternal "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("authorization: bearer slack-token")
                );
                let params = url::form_urlencoded::parse(body.as_bytes())
                    .into_owned()
                    .collect::<std::collections::HashMap<_, _>>();
                assert_eq!(params.get("channel_id"), Some(&"C12345678".to_string()));
                assert_eq!(
                    params.get("thread_ts"),
                    Some(&"1710000000.000100".to_string())
                );
                let files: Value = serde_json::from_str(params.get("files").unwrap()).unwrap();
                assert_eq!(files[0]["id"], json!("F123"));
                assert_eq!(files[0]["title"], json!("proof.txt"));
                (
                    200,
                    json!({
                        "ok": true,
                        "files": [{ "id": "F123" }]
                    })
                    .to_string(),
                )
            }
            _ => unreachable!(),
        });
        with_env_var("SLACK_BOT_TOKEN", Some("slack-token"));
        with_env_var("SLACK_API_BASE_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "slack:C12345678:1710000000.000100",
                "message": format!("hello slack\nMEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("slack"));
        assert_eq!(parsed["chat_id"], json!("C12345678"));
        assert_eq!(parsed["thread_id"], json!("1710000000.000100"));
        assert_eq!(parsed["message_id"], json!("F123"));
        join.join().unwrap();
        upload_join.join().unwrap();
    }

    #[test]
    fn sends_yuanbao_media_via_native_client() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("image.png");
        fs::write(
            &media_path,
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01",
        )
        .unwrap();
        let (upload_base, upload_join) = mock_server(1, move |_index, headers, body| {
            assert!(headers.starts_with("PUT /uploaded/test.png "));
            assert!(
                headers
                    .to_ascii_lowercase()
                    .contains("authorization: q-sign-algorithm=sha1")
            );
            assert!(!body.is_empty());
            (200, "{}".to_string())
        });
        let upload_url = format!("{upload_base}/uploaded/test.png");
        let (http_base, http_join) = mock_server(2, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /api/v5/robotLogic/sign-token "));
                (
                    200,
                    json!({
                        "code": 0,
                        "data": {
                            "token": "token-1",
                            "bot_id": "bot-1",
                            "duration": 3600
                        }
                    })
                    .to_string(),
                )
            }
            1 => {
                assert!(headers.starts_with("POST /api/resource/genUploadInfo "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["fileName"], json!("image.png"));
                (
                    200,
                    json!({
                        "code": 0,
                        "data": {
                            "bucketName": "bucket-1",
                            "region": "ap-guangzhou",
                            "location": "/uploaded/test.png",
                            "encryptTmpSecretId": "tmp-id",
                            "encryptTmpSecretKey": "tmp-secret",
                            "encryptToken": "session-token",
                            "startTime": 100,
                            "expiredTime": 4000,
                            "resourceUrl": upload_url
                        }
                    })
                    .to_string(),
                )
            }
            _ => unreachable!(),
        });
        let (ws_url, ws_join) = mock_ws_server(|mut websocket| {
            let auth = websocket.read().unwrap().into_data();
            let (auth_cmd, auth_msg_id) = frame_cmd_and_msg_id(&auth);
            assert_eq!(auth_cmd, "auth-bind");
            websocket
                .send(Message::Binary(auth_bind_response(&auth_msg_id).into()))
                .unwrap();

            let send = websocket.read().unwrap().into_data();
            let (send_cmd, send_msg_id) = frame_cmd_and_msg_id(&send);
            assert_eq!(send_cmd, "send_group_message");
            websocket
                .send(Message::Binary(ok_group_send_response(&send_msg_id).into()))
                .unwrap();
        });

        with_env_var("YUANBAO_APP_ID", Some("yb-app"));
        with_env_var("YUANBAO_APP_SECRET", Some("yb-secret"));
        with_env_var("YUANBAO_API_DOMAIN", Some(&http_base));
        with_env_var("YUANBAO_WS_URL", Some(&ws_url));
        let result = handle_send(
            &json!({
                "target": "yuanbao:group:123456",
                "message": format!("MEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("yuanbao"));
        assert_eq!(parsed["chat_id"], json!("group:123456"));
        assert!(parsed["message_id"].as_str().unwrap().starts_with("grp_"));
        http_join.join().unwrap();
        upload_join.join().unwrap();
        ws_join.join().unwrap();
    }
}
