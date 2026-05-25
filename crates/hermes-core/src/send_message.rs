use std::env;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use aes::Aes128;
use aes::cipher::{Array, BlockCipherEncrypt, KeyInit};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use getrandom::fill as fill_random;
use lettre::message::{
    Attachment as EmailAttachment, Mailbox, Message as EmailMessage, MultiPart, SinglePart,
    header::ContentType,
};
use lettre::transport::smtp::authentication::{Credentials, Mechanism};
use lettre::transport::smtp::client::Tls;
use lettre::{Address, SmtpTransport, Transport};
use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::blocking::multipart::{Form, Part};
use serde_json::{Value, json};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message as WsMessage, WebSocket, connect as ws_connect};

use crate::tools::{ToolRuntime, tool_error, tool_result};
use crate::yuanbao::{send_yuanbao_message, send_yuanbao_message_with_media};

const REQUEST_TIMEOUT_SECS: u64 = 30;
const TELEGRAM_DEFAULT_BASE_URL: &str = "https://api.telegram.org";
const DISCORD_DEFAULT_BASE_URL: &str = "https://discord.com/api/v10";
const SLACK_DEFAULT_BASE_URL: &str = "https://slack.com/api";
const FEISHU_DEFAULT_BASE_URL: &str = "https://open.feishu.cn";
const LARK_DEFAULT_BASE_URL: &str = "https://open.larksuite.com";
const MATRIX_DEFAULT_BASE_URL: &str = "";
const WHATSAPP_DEFAULT_BRIDGE_URL: &str = "http://127.0.0.1:3000";
const QQBOT_DEFAULT_BASE_URL: &str = "https://api.sgroup.qq.com";
const QQBOT_DEFAULT_TOKEN_URL: &str = "https://bots.qq.com/app/getAppAccessToken";
const MATTERMOST_DEFAULT_BASE_URL: &str = "";
const WECOM_DEFAULT_WS_URL: &str = "wss://openws.work.weixin.qq.com";
const WEIXIN_DEFAULT_BASE_URL: &str = "https://ilinkai.weixin.qq.com";
const WEIXIN_DEFAULT_CDN_BASE_URL: &str = "https://novac2c.cdn.weixin.qq.com/c2c";
const EMAIL_DEFAULT_SMTP_PORT_STARTTLS: u16 = 587;
const EMAIL_DEFAULT_SMTP_PORT_TLS: u16 = 465;
const EMAIL_DEFAULT_SMTP_PORT_NONE: u16 = 25;
const TWILIO_DEFAULT_BASE_URL: &str = "https://api.twilio.com/2010-04-01/Accounts";
const HOMEASSISTANT_DEFAULT_BASE_URL: &str = "http://homeassistant.local:8123";
const BLUEBUBBLES_DEFAULT_BASE_URL: &str = "";
const YUANBAO_DEFAULT_BASE_URL: &str = "https://bot.yuanbao.tencent.com";
const SIGNAL_MAX_ATTACHMENTS_PER_MSG: usize = 32;
const WECOM_IMAGE_MAX_BYTES: usize = 10 * 1024 * 1024;
const WECOM_VIDEO_MAX_BYTES: usize = 10 * 1024 * 1024;
const WECOM_VOICE_MAX_BYTES: usize = 2 * 1024 * 1024;
const WECOM_FILE_MAX_BYTES: usize = 20 * 1024 * 1024;
const WECOM_UPLOAD_CHUNK_SIZE: usize = 512 * 1024;
const WECOM_MAX_UPLOAD_CHUNKS: usize = 100;
const WEIXIN_MAX_MESSAGE_LENGTH: usize = 2000;
const WEIXIN_CDN_TIMEOUT_SECS: u64 = 120;
const WEIXIN_CHANNEL_VERSION: &str = "2.2.0";
const WEIXIN_APP_ID: &str = "bot";
const WEIXIN_APP_CLIENT_VERSION: u32 = (2 << 16) | (2 << 8);
const WEIXIN_MEDIA_IMAGE: i64 = 1;
const WEIXIN_MEDIA_VIDEO: i64 = 2;
const WEIXIN_MEDIA_FILE: i64 = 3;
const WEIXIN_MEDIA_VOICE: i64 = 4;
const WEIXIN_ITEM_TEXT: i64 = 1;
const WEIXIN_ITEM_IMAGE: i64 = 2;
const WEIXIN_ITEM_VOICE: i64 = 3;
const WEIXIN_ITEM_FILE: i64 = 4;
const WEIXIN_ITEM_VIDEO: i64 = 5;

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
    WhatsApp,
    DingTalk,
    QqBot,
    Mattermost,
    WeCom,
    Weixin,
    Email,
    Sms,
    HomeAssistant,
    BlueBubbles,
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
            Self::WhatsApp => "whatsapp",
            Self::DingTalk => "dingtalk",
            Self::QqBot => "qqbot",
            Self::Mattermost => "mattermost",
            Self::WeCom => "wecom",
            Self::Weixin => "weixin",
            Self::Email => "email",
            Self::Sms => "sms",
            Self::HomeAssistant => "homeassistant",
            Self::BlueBubbles => "bluebubbles",
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
            "whatsapp" => Some(Self::WhatsApp),
            "dingtalk" => Some(Self::DingTalk),
            "qqbot" => Some(Self::QqBot),
            "mattermost" => Some(Self::Mattermost),
            "wecom" => Some(Self::WeCom),
            "weixin" => Some(Self::Weixin),
            "email" => Some(Self::Email),
            "sms" => Some(Self::Sms),
            "homeassistant" => Some(Self::HomeAssistant),
            "bluebubbles" => Some(Self::BlueBubbles),
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
struct WhatsAppConfig {
    bridge_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct DingTalkConfig {
    webhook_url: Option<String>,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct QqBotConfig {
    app_id: String,
    client_secret: String,
    base_url: String,
    token_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct MattermostConfig {
    token: String,
    base_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct WeComConfig {
    bot_id: String,
    secret: String,
    ws_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct WeixinConfig {
    token: String,
    account_id: String,
    base_url: String,
    cdn_base_url: String,
    split_multiline_messages: bool,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmailSecurity {
    StartTls,
    Tls,
    None,
}

#[derive(Debug, Clone)]
struct EmailConfig {
    address: String,
    password: String,
    smtp_host: String,
    smtp_port: u16,
    smtp_security: EmailSecurity,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct SmsConfig {
    account_sid: String,
    auth_token: String,
    from_number: String,
    base_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct HomeAssistantConfig {
    token: String,
    base_url: String,
    home: Option<HomeTarget>,
}

#[derive(Debug, Clone)]
struct BlueBubblesConfig {
    server_url: String,
    password: String,
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
        "description": "Send a message to a connected messaging platform, or list the configured delivery targets. Supported in the Rust runtime: Telegram, Discord, Slack, Feishu, Matrix, WhatsApp, DingTalk, QQBot, Mattermost, WeCom, Weixin, Email, SMS, Home Assistant, BlueBubbles, Signal, and Yuanbao. When the user asks to send to a specific destination, call send_message with action='list' first if you need to inspect configured home targets or cached channel-directory entries. Human-friendly targets like slack:#engineering and discord:Guild/channel are resolved from the cached channel directory when available.",
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
                    "description": "Delivery target. Format: 'platform' to use the configured home target, 'platform:chat_id' with optional 'platform:chat_id:thread_id' for thread-aware platforms, 'signal:+15551234567' or 'signal:group:<group_id>' for Signal, 'sms:+15551234567' for Twilio SMS, 'email:user@example.com' for SMTP delivery, 'dingtalk:<chat_id>' for DingTalk home or logical chat ids, 'wecom:<chat_id>' for Enterprise WeChat bot delivery, 'weixin:<wxid|gh|v*_...|...@chatroom|filehelper>' for Weixin iLink delivery, 'matrix:!roomid:server.org' or 'matrix:@user:server.org' for Matrix, 'whatsapp:+15551234567' or 'whatsapp:<jid>' for WhatsApp, 'qqbot:<openid>', 'qqbot:group:<group_openid>', or 'qqbot:channel:<channel_id>' for QQBot, 'homeassistant:<notification_id>' for Home Assistant notifications, 'bluebubbles:<chat_guid|phone|email>' for BlueBubbles/iMessage, or 'yuanbao:group:<group_code>' / 'yuanbao:direct:<account_id>' for Yuanbao."
                },
                "message": {
                    "type": "string",
                    "description": "Text to send. MEDIA:/absolute/or/relative/path tags are supported for Telegram, Discord, Slack, Feishu, Matrix, WhatsApp, QQBot, Mattermost, WeCom, Weixin, Email, BlueBubbles, Signal, and Yuanbao. [[audio_as_voice]] marks OGG or Opus media for Telegram voice delivery."
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
            "No supported messaging platform is configured. Set TELEGRAM_BOT_TOKEN, DISCORD_BOT_TOKEN, SLACK_BOT_TOKEN, FEISHU_APP_ID/FEISHU_APP_SECRET, MATRIX_ACCESS_TOKEN/MATRIX_HOMESERVER, WHATSAPP_ENABLED, DINGTALK_WEBHOOK_URL, QQ_APP_ID/QQ_CLIENT_SECRET, MATTERMOST_TOKEN/MATTERMOST_URL, WECOM_BOT_ID/WECOM_SECRET, WEIXIN_TOKEN/WEIXIN_ACCOUNT_ID, EMAIL_ADDRESS/EMAIL_PASSWORD/EMAIL_SMTP_HOST, TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_PHONE_NUMBER, HASS_TOKEN, BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD, SIGNAL_HTTP_URL/SIGNAL_ACCOUNT, or YUANBAO_APP_ID/YUANBAO_APP_SECRET first.",
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
        PlatformKind::WhatsApp,
        PlatformKind::DingTalk,
        PlatformKind::QqBot,
        PlatformKind::Mattermost,
        PlatformKind::WeCom,
        PlatformKind::Weixin,
        PlatformKind::Email,
        PlatformKind::Sms,
        PlatformKind::HomeAssistant,
        PlatformKind::BlueBubbles,
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
            "No supported messaging platform is configured. Set TELEGRAM_BOT_TOKEN, DISCORD_BOT_TOKEN, SLACK_BOT_TOKEN, FEISHU_APP_ID/FEISHU_APP_SECRET, MATRIX_ACCESS_TOKEN/MATRIX_HOMESERVER, WHATSAPP_ENABLED, DINGTALK_WEBHOOK_URL, QQ_APP_ID/QQ_CLIENT_SECRET, MATTERMOST_TOKEN/MATTERMOST_URL, WECOM_BOT_ID/WECOM_SECRET, WEIXIN_TOKEN/WEIXIN_ACCOUNT_ID, EMAIL_ADDRESS/EMAIL_PASSWORD/EMAIL_SMTP_HOST, TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_PHONE_NUMBER, HASS_TOKEN, BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD, SIGNAL_HTTP_URL/SIGNAL_ACCOUNT, or YUANBAO_APP_ID/YUANBAO_APP_SECRET first.",
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
        PlatformKind::WhatsApp => configs
            .whatsapp
            .as_ref()
            .ok_or_else(|| SendError("WhatsApp is not configured".to_string()))
            .and_then(|config| send_whatsapp(config, &resolved, &cleaned_message, &media)),
        PlatformKind::DingTalk => configs
            .dingtalk
            .as_ref()
            .ok_or_else(|| SendError("DingTalk is not configured".to_string()))
            .and_then(|config| send_dingtalk(config, &resolved, &cleaned_message, &media)),
        PlatformKind::QqBot => configs
            .qqbot
            .as_ref()
            .ok_or_else(|| SendError("QQBot is not configured".to_string()))
            .and_then(|config| send_qqbot(config, &resolved, &cleaned_message, &media)),
        PlatformKind::Mattermost => configs
            .mattermost
            .as_ref()
            .ok_or_else(|| SendError("Mattermost is not configured".to_string()))
            .and_then(|config| send_mattermost(config, &resolved, &cleaned_message, &media)),
        PlatformKind::WeCom => configs
            .wecom
            .as_ref()
            .ok_or_else(|| SendError("WeCom is not configured".to_string()))
            .and_then(|config| send_wecom(config, &resolved, &cleaned_message, &media)),
        PlatformKind::Weixin => configs
            .weixin
            .as_ref()
            .ok_or_else(|| SendError("Weixin is not configured".to_string()))
            .and_then(|config| send_weixin(config, runtime, &resolved, &cleaned_message, &media)),
        PlatformKind::Email => configs
            .email
            .as_ref()
            .ok_or_else(|| SendError("Email is not configured".to_string()))
            .and_then(|config| send_email(config, &resolved, &cleaned_message, &media)),
        PlatformKind::Sms => configs
            .sms
            .as_ref()
            .ok_or_else(|| SendError("SMS is not configured".to_string()))
            .and_then(|config| send_sms(config, &resolved, &cleaned_message, &media)),
        PlatformKind::HomeAssistant => configs
            .homeassistant
            .as_ref()
            .ok_or_else(|| SendError("Home Assistant is not configured".to_string()))
            .and_then(|config| send_homeassistant(config, &resolved, &cleaned_message, &media)),
        PlatformKind::BlueBubbles => configs
            .bluebubbles
            .as_ref()
            .ok_or_else(|| SendError("BlueBubbles is not configured".to_string()))
            .and_then(|config| send_bluebubbles(config, &resolved, &cleaned_message, &media)),
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
    whatsapp: Option<WhatsAppConfig>,
    dingtalk: Option<DingTalkConfig>,
    qqbot: Option<QqBotConfig>,
    mattermost: Option<MattermostConfig>,
    wecom: Option<WeComConfig>,
    weixin: Option<WeixinConfig>,
    email: Option<EmailConfig>,
    sms: Option<SmsConfig>,
    homeassistant: Option<HomeAssistantConfig>,
    bluebubbles: Option<BlueBubblesConfig>,
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
            || self.whatsapp.is_some()
            || self.dingtalk.is_some()
            || self.qqbot.is_some()
            || self.mattermost.is_some()
            || self.wecom.is_some()
            || self.weixin.is_some()
            || self.email.is_some()
            || self.sms.is_some()
            || self.homeassistant.is_some()
            || self.bluebubbles.is_some()
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
            PlatformKind::WhatsApp => self.whatsapp.is_some(),
            PlatformKind::DingTalk => self.dingtalk.is_some(),
            PlatformKind::QqBot => self.qqbot.is_some(),
            PlatformKind::Mattermost => self.mattermost.is_some(),
            PlatformKind::WeCom => self.wecom.is_some(),
            PlatformKind::Weixin => self.weixin.is_some(),
            PlatformKind::Email => self.email.is_some(),
            PlatformKind::Sms => self.sms.is_some(),
            PlatformKind::HomeAssistant => self.homeassistant.is_some(),
            PlatformKind::BlueBubbles => self.bluebubbles.is_some(),
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
            PlatformKind::WhatsApp => self
                .whatsapp
                .as_ref()
                .and_then(|config| config.home.clone()),
            PlatformKind::DingTalk => self
                .dingtalk
                .as_ref()
                .and_then(|config| config.home.clone()),
            PlatformKind::QqBot => self.qqbot.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::Mattermost => self
                .mattermost
                .as_ref()
                .and_then(|config| config.home.clone()),
            PlatformKind::WeCom => self.wecom.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::Weixin => self.weixin.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::Email => self.email.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::Sms => self.sms.as_ref().and_then(|config| config.home.clone()),
            PlatformKind::HomeAssistant => self
                .homeassistant
                .as_ref()
                .and_then(|config| config.home.clone()),
            PlatformKind::BlueBubbles => self
                .bluebubbles
                .as_ref()
                .and_then(|config| config.home.clone()),
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
        whatsapp: load_whatsapp_config(),
        dingtalk: load_dingtalk_config(),
        qqbot: load_qqbot_config(),
        mattermost: load_mattermost_config(),
        wecom: load_wecom_config(),
        weixin: load_weixin_config(),
        email: load_email_config(),
        sms: load_sms_config(),
        homeassistant: load_homeassistant_config(),
        bluebubbles: load_bluebubbles_config(),
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

fn load_whatsapp_config() -> Option<WhatsAppConfig> {
    let enabled = env_truthy("WHATSAPP_ENABLED");
    let home = load_home_target(
        "WHATSAPP_HOME_CHANNEL",
        "WHATSAPP_HOME_CHANNEL_THREAD_ID",
        "WHATSAPP_HOME_CHANNEL_NAME",
    );
    let has_bridge_override = env_trimmed("WHATSAPP_BRIDGE_URL").is_some()
        || env_trimmed("WHATSAPP_BRIDGE_PORT").is_some();
    if !enabled && home.is_none() && !has_bridge_override {
        return None;
    }
    Some(WhatsAppConfig {
        bridge_url: whatsapp_bridge_base_url().ok()?,
        home,
    })
}

fn load_dingtalk_config() -> Option<DingTalkConfig> {
    let webhook_url = env_trimmed("DINGTALK_WEBHOOK_URL")
        .map(|raw| normalize_dingtalk_webhook_url(&raw))
        .transpose()
        .ok()?;
    let has_gateway_credentials = env_trimmed("DINGTALK_CLIENT_ID").is_some()
        && env_trimmed("DINGTALK_CLIENT_SECRET").is_some();
    let home = env_trimmed("DINGTALK_HOME_CHANNEL").and_then(|chat_id| {
        is_dingtalk_chat_id(&chat_id).then(|| HomeTarget {
            chat_id,
            thread_id: None,
            name: env_trimmed("DINGTALK_HOME_CHANNEL_NAME").unwrap_or_else(|| "Home".to_string()),
        })
    });
    if webhook_url.is_none() && !has_gateway_credentials && home.is_none() {
        return None;
    }
    Some(DingTalkConfig { webhook_url, home })
}

fn load_qqbot_config() -> Option<QqBotConfig> {
    let app_id = env_trimmed("QQ_APP_ID")?;
    let client_secret = env_trimmed("QQ_CLIENT_SECRET")?;
    Some(QqBotConfig {
        app_id,
        client_secret,
        base_url: normalize_base_url(
            env_trimmed("QQBOT_API_BASE_URL").as_deref(),
            QQBOT_DEFAULT_BASE_URL,
        ),
        token_url: normalize_base_url(
            env_trimmed("QQBOT_TOKEN_URL").as_deref(),
            QQBOT_DEFAULT_TOKEN_URL,
        ),
        home: env_trimmed("QQBOT_HOME_CHANNEL")
            .or_else(|| env_trimmed("QQ_HOME_CHANNEL"))
            .and_then(|chat_id| {
                parse_qqbot_target(&chat_id).ok().map(|_| HomeTarget {
                    chat_id,
                    thread_id: None,
                    name: env_trimmed("QQBOT_HOME_CHANNEL_NAME")
                        .or_else(|| env_trimmed("QQ_HOME_CHANNEL_NAME"))
                        .unwrap_or_else(|| "Home".to_string()),
                })
            }),
    })
}

fn load_mattermost_config() -> Option<MattermostConfig> {
    let token = env_trimmed("MATTERMOST_TOKEN")?;
    let base_url = normalize_base_url(
        env_trimmed("MATTERMOST_URL").as_deref(),
        MATTERMOST_DEFAULT_BASE_URL,
    );
    if base_url.is_empty() {
        return None;
    }
    Some(MattermostConfig {
        token,
        base_url,
        home: load_home_target(
            "MATTERMOST_HOME_CHANNEL",
            "MATTERMOST_HOME_CHANNEL_THREAD_ID",
            "MATTERMOST_HOME_CHANNEL_NAME",
        ),
    })
}

fn load_wecom_config() -> Option<WeComConfig> {
    let bot_id = env_trimmed("WECOM_BOT_ID")?;
    let secret = env_trimmed("WECOM_SECRET")?;
    Some(WeComConfig {
        bot_id,
        secret,
        ws_url: normalize_ws_url(
            env_trimmed("WECOM_WEBSOCKET_URL").as_deref(),
            WECOM_DEFAULT_WS_URL,
            "WECOM_WEBSOCKET_URL",
        )
        .ok()?,
        home: env_trimmed("WECOM_HOME_CHANNEL").map(|chat_id| HomeTarget {
            chat_id,
            thread_id: None,
            name: env_trimmed("WECOM_HOME_CHANNEL_NAME").unwrap_or_else(|| "Home".to_string()),
        }),
    })
}

fn load_weixin_config() -> Option<WeixinConfig> {
    let token = env_trimmed("WEIXIN_TOKEN")?;
    let account_id = env_trimmed("WEIXIN_ACCOUNT_ID")?;
    Some(WeixinConfig {
        token,
        account_id,
        base_url: normalize_base_url(
            env_trimmed("WEIXIN_BASE_URL").as_deref(),
            WEIXIN_DEFAULT_BASE_URL,
        ),
        cdn_base_url: normalize_base_url(
            env_trimmed("WEIXIN_CDN_BASE_URL").as_deref(),
            WEIXIN_DEFAULT_CDN_BASE_URL,
        ),
        split_multiline_messages: env_truthy("WEIXIN_SPLIT_MULTILINE_MESSAGES"),
        home: env_trimmed("WEIXIN_HOME_CHANNEL").and_then(|chat_id| {
            is_weixin_target(&chat_id).then(|| HomeTarget {
                chat_id,
                thread_id: None,
                name: env_trimmed("WEIXIN_HOME_CHANNEL_NAME").unwrap_or_else(|| "Home".to_string()),
            })
        }),
    })
}

fn load_email_config() -> Option<EmailConfig> {
    let address = env_trimmed("EMAIL_ADDRESS")?;
    parse_email_address(&address).ok()?;
    let password = env_trimmed("EMAIL_PASSWORD")?;
    let smtp_host = env_trimmed("EMAIL_SMTP_HOST")?;
    let smtp_security = email_security_mode(env_trimmed("EMAIL_SMTP_SECURITY").as_deref()).ok()?;
    let smtp_port =
        email_smtp_port(env_trimmed("EMAIL_SMTP_PORT").as_deref(), smtp_security).ok()?;
    Some(EmailConfig {
        address,
        password,
        smtp_host,
        smtp_port,
        smtp_security,
        home: env_trimmed("EMAIL_HOME_ADDRESS").and_then(|chat_id| {
            parse_email_address(&chat_id).ok().map(|_| HomeTarget {
                chat_id,
                thread_id: None,
                name: env_trimmed("EMAIL_HOME_ADDRESS_NAME").unwrap_or_else(|| "Home".to_string()),
            })
        }),
    })
}

fn load_sms_config() -> Option<SmsConfig> {
    let account_sid = env_trimmed("TWILIO_ACCOUNT_SID")?;
    let auth_token = env_trimmed("TWILIO_AUTH_TOKEN")?;
    let from_number = env_trimmed("TWILIO_PHONE_NUMBER")?;
    Some(SmsConfig {
        account_sid,
        auth_token,
        from_number,
        base_url: normalize_base_url(
            env_trimmed("TWILIO_API_BASE_URL").as_deref(),
            TWILIO_DEFAULT_BASE_URL,
        ),
        home: env_trimmed("SMS_HOME_CHANNEL").map(|chat_id| HomeTarget {
            chat_id,
            thread_id: None,
            name: env_trimmed("SMS_HOME_CHANNEL_NAME").unwrap_or_else(|| "Home".to_string()),
        }),
    })
}

fn load_homeassistant_config() -> Option<HomeAssistantConfig> {
    let token = env_trimmed("HASS_TOKEN")?;
    Some(HomeAssistantConfig {
        token,
        base_url: normalize_base_url(
            env_trimmed("HASS_URL").as_deref(),
            HOMEASSISTANT_DEFAULT_BASE_URL,
        ),
        home: env_trimmed("HASS_NOTIFICATION_ID").map(|chat_id| HomeTarget {
            chat_id,
            thread_id: None,
            name: env_trimmed("HASS_NOTIFICATION_NAME").unwrap_or_else(|| "Home".to_string()),
        }),
    })
}

fn load_bluebubbles_config() -> Option<BlueBubblesConfig> {
    let server_url = normalize_base_url(
        env_trimmed("BLUEBUBBLES_SERVER_URL").as_deref(),
        BLUEBUBBLES_DEFAULT_BASE_URL,
    );
    let password = env_trimmed("BLUEBUBBLES_PASSWORD")?;
    if server_url.is_empty() {
        return None;
    }
    Some(BlueBubblesConfig {
        server_url,
        password,
        home: load_home_target(
            "BLUEBUBBLES_HOME_CHANNEL",
            "BLUEBUBBLES_HOME_CHANNEL_THREAD_ID",
            "BLUEBUBBLES_HOME_CHANNEL_NAME",
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

fn env_truthy(key: &str) -> bool {
    env::var(key).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn email_security_mode(raw: Option<&str>) -> Result<EmailSecurity, String> {
    match raw
        .unwrap_or("starttls")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "" | "starttls" | "start_tls" | "tls-required" => Ok(EmailSecurity::StartTls),
        "tls" | "ssl" | "smtps" | "wrapper" => Ok(EmailSecurity::Tls),
        "none" | "plain" | "insecure" => Ok(EmailSecurity::None),
        other => Err(format!(
            "EMAIL_SMTP_SECURITY must be one of: starttls, tls, or none (got '{other}')"
        )),
    }
}

fn email_smtp_port(raw: Option<&str>, security: EmailSecurity) -> Result<u16, String> {
    let default = match security {
        EmailSecurity::StartTls => EMAIL_DEFAULT_SMTP_PORT_STARTTLS,
        EmailSecurity::Tls => EMAIL_DEFAULT_SMTP_PORT_TLS,
        EmailSecurity::None => EMAIL_DEFAULT_SMTP_PORT_NONE,
    };
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(default);
    };
    let port = raw
        .parse::<u16>()
        .map_err(|_| "EMAIL_SMTP_PORT must be a valid TCP port".to_string())?;
    if port == 0 {
        return Err("EMAIL_SMTP_PORT must be between 1 and 65535".to_string());
    }
    Ok(port)
}

fn normalize_http_base_url(raw: &str, key: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    let url = Url::parse(trimmed).map_err(|error| format!("invalid {key}: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(format!("{key} must include http or https and a host"));
    }
    Ok(url.to_string().trim_end_matches('/').to_string())
}

fn normalize_dingtalk_webhook_url(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    let url =
        Url::parse(trimmed).map_err(|error| format!("invalid DINGTALK_WEBHOOK_URL: {error}"))?;
    let Some(host) = url.host_str() else {
        return Err("DINGTALK_WEBHOOK_URL must include a host".to_string());
    };
    let scheme = url.scheme();
    let is_loopback = matches!(host, "127.0.0.1" | "localhost" | "::1");
    let is_dingtalk_host = matches!(host, "api.dingtalk.com" | "oapi.dingtalk.com");
    if !(scheme == "https" && is_dingtalk_host || matches!(scheme, "http" | "https") && is_loopback)
    {
        return Err(
            "DINGTALK_WEBHOOK_URL must use https://api.dingtalk.com/... or https://oapi.dingtalk.com/...".to_string(),
        );
    }
    if !url.path().starts_with("/robot/send") {
        return Err("DINGTALK_WEBHOOK_URL must point to /robot/send".to_string());
    }
    let has_access_token = url
        .query_pairs()
        .any(|(key, value)| key == "access_token" && !value.trim().is_empty());
    if !has_access_token {
        return Err(
            "DINGTALK_WEBHOOK_URL must include a non-empty access_token query parameter"
                .to_string(),
        );
    }
    Ok(trimmed.to_string())
}

fn normalize_ws_url(raw: Option<&str>, default: &str, key: &str) -> Result<String, String> {
    let trimmed = raw.unwrap_or(default).trim();
    let url = Url::parse(trimmed).map_err(|error| format!("invalid {key}: {error}"))?;
    if !matches!(url.scheme(), "ws" | "wss") || url.host_str().is_none() {
        return Err(format!("{key} must include ws or wss and a host"));
    }
    Ok(trimmed.trim_end_matches('/').to_string())
}

fn whatsapp_bridge_base_url() -> Result<String, String> {
    if let Some(raw) = env_trimmed("WHATSAPP_BRIDGE_URL") {
        return normalize_http_base_url(&raw, "WHATSAPP_BRIDGE_URL");
    }
    if let Some(raw_port) = env_trimmed("WHATSAPP_BRIDGE_PORT") {
        let port = raw_port
            .parse::<u16>()
            .map_err(|_| "WHATSAPP_BRIDGE_PORT must be a valid TCP port".to_string())?;
        if port == 0 {
            return Err("WHATSAPP_BRIDGE_PORT must be between 1 and 65535".to_string());
        }
        return Ok(format!("http://127.0.0.1:{port}"));
    }
    Ok(WHATSAPP_DEFAULT_BRIDGE_URL.to_string())
}

fn split_platform_target(raw: &str) -> Result<(PlatformKind, Option<String>), String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("target must not be empty".to_string());
    }
    let mut parts = trimmed.splitn(2, ':');
    let platform = parts.next().unwrap_or_default();
    let kind = PlatformKind::from_name(platform)
        .ok_or_else(|| format!("Unsupported platform '{platform}'. Supported Rust send_message platforms: telegram, discord, slack, feishu, matrix, whatsapp, dingtalk, qqbot, mattermost, wecom, weixin, email, sms, homeassistant, bluebubbles, signal, yuanbao."))?;
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
            if resolved.thread_id.is_some() {
                match kind {
                    PlatformKind::WeCom => {
                        return Err(
                            "WeCom thread targets are not supported in the Rust send_message runtime."
                                .to_string(),
                        );
                    }
                    PlatformKind::Weixin => {
                        return Err(
                            "Weixin thread targets are not supported in the Rust send_message runtime."
                                .to_string(),
                        );
                    }
                    PlatformKind::Email => {
                        return Err(
                            "Email thread targets are not supported in the Rust send_message runtime."
                                .to_string(),
                        );
                    }
                    PlatformKind::Signal => {
                        return Err(
                            "Signal thread targets are not supported in the Rust send_message runtime."
                                .to_string(),
                        );
                    }
                    PlatformKind::WhatsApp => {
                        return Err(
                            "WhatsApp thread targets are not supported in the Rust send_message runtime."
                                .to_string(),
                        );
                    }
                    _ => {}
                }
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

    if kind == PlatformKind::WhatsApp {
        return Ok(Some(ResolvedTarget {
            chat_id: normalize_whatsapp_target(trimmed)?,
            thread_id: None,
            used_home_channel: false,
        }));
    }

    if kind == PlatformKind::DingTalk {
        if trimmed.contains(':') {
            return Err(
                "DingTalk targets must be a bare chat id like dingtalk:cidXXXX==".to_string(),
            );
        }
        if !is_dingtalk_chat_id(trimmed) {
            return Ok(None);
        }
        return Ok(Some(ResolvedTarget {
            chat_id: trimmed.to_string(),
            thread_id: None,
            used_home_channel: false,
        }));
    }

    if kind == PlatformKind::QqBot {
        return Ok(Some(ResolvedTarget {
            chat_id: parse_qqbot_target(trimmed)?,
            thread_id: None,
            used_home_channel: false,
        }));
    }

    if kind == PlatformKind::WeCom {
        validate_target_component("chat_id", trimmed)?;
        return Ok(Some(ResolvedTarget {
            chat_id: trimmed.to_string(),
            thread_id: None,
            used_home_channel: false,
        }));
    }

    if kind == PlatformKind::Weixin {
        if !is_weixin_target(trimmed) {
            return Ok(None);
        }
        return Ok(Some(ResolvedTarget {
            chat_id: trimmed.to_string(),
            thread_id: None,
            used_home_channel: false,
        }));
    }

    if kind == PlatformKind::Email {
        if trimmed.contains(':') {
            return Err(
                "Email targets must be a bare address like email:user@example.com".to_string(),
            );
        }
        return Ok(Some(ResolvedTarget {
            chat_id: parse_email_address(trimmed)?.to_string(),
            thread_id: None,
            used_home_channel: false,
        }));
    }

    if kind == PlatformKind::Sms {
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

    if kind == PlatformKind::HomeAssistant {
        return Ok(Some(ResolvedTarget {
            chat_id: trimmed.to_string(),
            thread_id: None,
            used_home_channel: false,
        }));
    }

    if kind == PlatformKind::BlueBubbles {
        return Ok(Some(ResolvedTarget {
            chat_id: trimmed.to_string(),
            thread_id: None,
            used_home_channel: false,
        }));
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
        PlatformKind::Mattermost => {
            validate_target_component("chat_id", chat_id).is_ok()
                && thread_id
                    .is_none_or(|value| validate_target_component("thread_id", value).is_ok())
        }
        PlatformKind::Matrix
        | PlatformKind::WhatsApp
        | PlatformKind::DingTalk
        | PlatformKind::QqBot
        | PlatformKind::WeCom
        | PlatformKind::Weixin
        | PlatformKind::Email
        | PlatformKind::Sms
        | PlatformKind::HomeAssistant
        | PlatformKind::BlueBubbles
        | PlatformKind::Signal
        | PlatformKind::Yuanbao => false,
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

fn is_dingtalk_chat_id(raw: &str) -> bool {
    let trimmed = raw.trim();
    !trimmed.is_empty()
        && !trimmed.contains(':')
        && validate_target_component("chat_id", trimmed).is_ok()
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

fn normalize_whatsapp_target(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("chat_id must not be empty".to_string());
    }
    if is_whatsapp_jid(trimmed) {
        validate_target_component("chat_id", trimmed)?;
        return Ok(trimmed.to_string());
    }
    let digits = normalize_phone_digits(trimmed)
        .ok_or_else(|| "WhatsApp targets must be a phone number or WhatsApp JID".to_string())?;
    validate_target_component("chat_id", &digits)?;
    Ok(format!("{digits}@s.whatsapp.net"))
}

fn parse_qqbot_target(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("chat_id must not be empty".to_string());
    }
    if let Some(value) = trimmed.strip_prefix("group:") {
        validate_target_component("group_openid", value)?;
        return Ok(trimmed.to_string());
    }
    if let Some(value) = trimmed.strip_prefix("channel:") {
        validate_target_component("channel_id", value)?;
        return Ok(trimmed.to_string());
    }
    if let Some(value) = trimmed.strip_prefix("user:") {
        validate_target_component("user_openid", value)?;
        return Ok(trimmed.to_string());
    }
    validate_target_component("chat_id", trimmed)?;
    Ok(trimmed.to_string())
}

fn is_whatsapp_jid(raw: &str) -> bool {
    let trimmed = raw.trim();
    trimmed.ends_with("@s.whatsapp.net") || trimmed.ends_with("@g.us") || trimmed.ends_with("@lid")
}

fn normalize_phone_digits(raw: &str) -> Option<String> {
    let mut digits = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else if !matches!(ch, '+' | '-' | '(' | ')' | ' ') {
            return None;
        }
    }
    ((7..=20).contains(&digits.len())).then_some(digits)
}

fn is_weixin_target(raw: &str) -> bool {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("filehelper") {
        return true;
    }
    if trimmed.ends_with("@chatroom")
        && !trimmed.starts_with('@')
        && trimmed
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-' | '@'))
    {
        return true;
    }
    ["wxid_", "gh_", "wm_", "wb_"].into_iter().any(|prefix| {
        trimmed.starts_with(prefix)
            && trimmed[prefix.len()..]
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
    }) || trimmed
        .strip_prefix('v')
        .and_then(|rest| rest.split_once('_'))
        .is_some_and(|(digits, suffix)| {
            !digits.is_empty()
                && digits.chars().all(|ch| ch.is_ascii_digit())
                && !suffix.is_empty()
                && suffix
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
        })
}

fn parse_email_address(raw: &str) -> Result<Address, String> {
    raw.trim()
        .parse::<Address>()
        .map_err(|_| "Email targets must be valid email addresses".to_string())
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
        let mime_type = guess_mime_type(&attachment.path, &bytes);
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

fn send_mattermost(
    config: &MattermostConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    let client = http_client()?;
    let mut last_message_id = None;

    if !message.is_empty() {
        let mut payload = serde_json::Map::new();
        payload.insert(
            "channel_id".to_string(),
            Value::String(target.chat_id.clone()),
        );
        payload.insert("message".to_string(), Value::String(message.to_string()));
        if let Some(thread_id) = target.thread_id.as_ref() {
            payload.insert("root_id".to_string(), Value::String(thread_id.clone()));
        }
        let response = client
            .post(format!("{}/api/v4/posts", config.base_url))
            .bearer_auth(&config.token)
            .header("Content-Type", "application/json")
            .json(&Value::Object(payload))
            .send()
            .map_err(|error| SendError(format!("Mattermost send failed: {error}")))?;
        let body = parse_json_response(response, "Mattermost send failed")?;
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
        let mime_type = guess_mime_type(&attachment.path, &bytes);
        let file_id = mattermost_upload_file(
            &client,
            config,
            &target.chat_id,
            &file_name,
            &mime_type,
            &bytes,
        )?;
        let mut payload = serde_json::Map::new();
        payload.insert(
            "channel_id".to_string(),
            Value::String(target.chat_id.clone()),
        );
        payload.insert("message".to_string(), Value::String(String::new()));
        payload.insert("file_ids".to_string(), json!([file_id]));
        if let Some(thread_id) = target.thread_id.as_ref() {
            payload.insert("root_id".to_string(), Value::String(thread_id.clone()));
        }
        let response = client
            .post(format!("{}/api/v4/posts", config.base_url))
            .bearer_auth(&config.token)
            .header("Content-Type", "application/json")
            .json(&Value::Object(payload))
            .send()
            .map_err(|error| SendError(format!("Mattermost media send failed: {error}")))?;
        let body = parse_json_response(response, "Mattermost media send failed")?;
        last_message_id = body.get("id").and_then(value_as_string);
    }

    Ok(SentMessage {
        platform: PlatformKind::Mattermost,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: target.thread_id.clone(),
        note: target.used_home_channel.then(|| {
            format!(
                "Sent to mattermost home channel (chat_id: {})",
                target.chat_id
            )
        }),
    })
}

fn mattermost_upload_file(
    client: &Client,
    config: &MattermostConfig,
    channel_id: &str,
    file_name: &str,
    mime_type: &str,
    bytes: &[u8],
) -> Result<String, SendError> {
    let part = Part::bytes(bytes.to_vec())
        .file_name(file_name.to_string())
        .mime_str(mime_type)
        .map_err(|error| SendError(format!("Mattermost media upload failed: {error}")))?;
    let form = Form::new()
        .text("channel_id", channel_id.to_string())
        .part("files", part);
    let response = client
        .post(format!("{}/api/v4/files", config.base_url))
        .bearer_auth(&config.token)
        .multipart(form)
        .send()
        .map_err(|error| SendError(format!("Mattermost media upload failed: {error}")))?;
    let body = parse_json_response(response, "Mattermost media upload failed")?;
    body.pointer("/file_infos/0/id")
        .and_then(value_as_string)
        .ok_or_else(|| SendError("Mattermost media upload failed: missing file id".to_string()))
}

fn send_wecom(
    config: &WeComConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    if target.thread_id.is_some() {
        return Err(SendError(
            "WeCom thread targets are not supported in the Rust send_message runtime.".to_string(),
        ));
    }

    let mut session = WeComSession::connect(config)?;
    let mut last_message_id = None;

    if !message.is_empty() {
        last_message_id = Some(session.send_markdown(&target.chat_id, message)?);
    }

    for attachment in media {
        let bytes = fs::read(&attachment.path).map_err(|error| {
            SendError(format!(
                "Reading media {} failed: {error}",
                attachment.path.display()
            ))
        })?;
        let file_name = attachment
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("attachment.bin")
            .to_string();
        let mime_type = guess_mime_type(&attachment.path, &bytes);
        let media_type = wecom_media_type(&mime_type, bytes.len())?;
        let media_id = session.upload_media(media_type, &file_name, &bytes)?;
        last_message_id = Some(session.send_media(&target.chat_id, media_type, &media_id)?);
    }

    Ok(SentMessage {
        platform: PlatformKind::WeCom,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: None,
        note: target
            .used_home_channel
            .then(|| format!("Sent to wecom home channel (chat_id: {})", target.chat_id)),
    })
}

fn wecom_media_type(mime_type: &str, size: usize) -> Result<&'static str, SendError> {
    if size > WECOM_FILE_MAX_BYTES {
        return Err(SendError(
            "WeCom media send failed: file exceeds the 20MB platform limit".to_string(),
        ));
    }
    let normalized = mime_type.trim().to_ascii_lowercase();
    if normalized.starts_with("image/") {
        return Ok(if size > WECOM_IMAGE_MAX_BYTES {
            "file"
        } else {
            "image"
        });
    }
    if normalized.starts_with("video/") {
        return Ok(if size > WECOM_VIDEO_MAX_BYTES {
            "file"
        } else {
            "video"
        });
    }
    if normalized.starts_with("audio/") || normalized == "application/ogg" {
        if normalized != "audio/amr" || size > WECOM_VOICE_MAX_BYTES {
            return Ok("file");
        }
        return Ok("voice");
    }
    Ok("file")
}

struct WeComSession {
    websocket: WebSocket<MaybeTlsStream<TcpStream>>,
}

impl WeComSession {
    fn connect(config: &WeComConfig) -> Result<Self, SendError> {
        let (mut websocket, _) = ws_connect(config.ws_url.as_str())
            .map_err(|error| SendError(format!("WeCom websocket connect failed: {error}")))?;
        let subscribe_req_id = format!("subscribe-{}", unique_suffix());
        let subscribe = json!({
            "cmd": "aibot_subscribe",
            "headers": { "req_id": subscribe_req_id },
            "body": {
                "bot_id": config.bot_id,
                "secret": config.secret,
                "device_id": format!("rust-send-{}", unique_suffix()),
            }
        });
        Self::send_payload(&mut websocket, &subscribe)?;
        let response = Self::wait_for_response(&mut websocket, &subscribe_req_id)?;
        if let Some(error) = wecom_response_error(&response) {
            return Err(SendError(format!("WeCom subscribe failed: {error}")));
        }
        Ok(Self { websocket })
    }

    fn send_markdown(&mut self, chat_id: &str, message: &str) -> Result<String, SendError> {
        let response = self.request(
            "aibot_send_msg",
            json!({
                "chatid": chat_id,
                "msgtype": "markdown",
                "markdown": { "content": message.chars().take(4000).collect::<String>() },
            }),
        )?;
        if let Some(error) = wecom_response_error(&response) {
            return Err(SendError(format!("WeCom send failed: {error}")));
        }
        Ok(wecom_payload_req_id(&response)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("wecom-{}", unique_suffix())))
    }

    fn upload_media(
        &mut self,
        media_type: &str,
        file_name: &str,
        bytes: &[u8],
    ) -> Result<String, SendError> {
        let total_chunks = bytes.len().div_ceil(WECOM_UPLOAD_CHUNK_SIZE);
        if total_chunks > WECOM_MAX_UPLOAD_CHUNKS {
            return Err(SendError(
                "WeCom media send failed: file exceeds the maximum upload chunk count".to_string(),
            ));
        }
        let init = self.request(
            "aibot_upload_media_init",
            json!({
                "type": media_type,
                "filename": file_name,
                "total_size": bytes.len(),
                "total_chunks": total_chunks,
                "md5": format!("{:x}", md5::compute(bytes)),
            }),
        )?;
        if let Some(error) = wecom_response_error(&init) {
            return Err(SendError(format!(
                "WeCom media upload init failed: {error}"
            )));
        }
        let upload_id = init
            .pointer("/body/upload_id")
            .and_then(value_as_string)
            .ok_or_else(|| {
                SendError("WeCom media upload init failed: missing upload_id".to_string())
            })?;
        for (index, chunk) in bytes.chunks(WECOM_UPLOAD_CHUNK_SIZE).enumerate() {
            let response = self.request(
                "aibot_upload_media_chunk",
                json!({
                    "upload_id": upload_id,
                    "chunk_index": index,
                    "base64_data": BASE64_STANDARD.encode(chunk),
                }),
            )?;
            if let Some(error) = wecom_response_error(&response) {
                return Err(SendError(format!(
                    "WeCom media upload chunk {} failed: {error}",
                    index
                )));
            }
        }
        let finish = self.request(
            "aibot_upload_media_finish",
            json!({ "upload_id": upload_id }),
        )?;
        if let Some(error) = wecom_response_error(&finish) {
            return Err(SendError(format!(
                "WeCom media upload finish failed: {error}"
            )));
        }
        finish
            .pointer("/body/media_id")
            .and_then(value_as_string)
            .ok_or_else(|| {
                SendError("WeCom media upload finish failed: missing media_id".to_string())
            })
    }

    fn send_media(
        &mut self,
        chat_id: &str,
        media_type: &str,
        media_id: &str,
    ) -> Result<String, SendError> {
        let response = self.request(
            "aibot_send_msg",
            json!({
                "chatid": chat_id,
                "msgtype": media_type,
                media_type: { "media_id": media_id },
            }),
        )?;
        if let Some(error) = wecom_response_error(&response) {
            return Err(SendError(format!("WeCom media send failed: {error}")));
        }
        Ok(wecom_payload_req_id(&response)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("wecom-{}", unique_suffix())))
    }

    fn request(&mut self, cmd: &str, body: Value) -> Result<Value, SendError> {
        let req_id = format!("{cmd}-{}", unique_suffix());
        let payload = json!({
            "cmd": cmd,
            "headers": { "req_id": req_id },
            "body": body,
        });
        Self::send_payload(&mut self.websocket, &payload)?;
        Self::wait_for_response(&mut self.websocket, &req_id)
    }

    fn send_payload(
        websocket: &mut WebSocket<MaybeTlsStream<TcpStream>>,
        payload: &Value,
    ) -> Result<(), SendError> {
        websocket
            .send(WsMessage::Text(payload.to_string().into()))
            .map_err(|error| SendError(format!("WeCom websocket send failed: {error}")))
    }

    fn wait_for_response(
        websocket: &mut WebSocket<MaybeTlsStream<TcpStream>>,
        expected_req_id: &str,
    ) -> Result<Value, SendError> {
        loop {
            let frame = websocket
                .read()
                .map_err(|error| SendError(format!("WeCom websocket read failed: {error}")))?;
            match frame {
                WsMessage::Text(text) => {
                    let payload: Value = serde_json::from_str(text.as_ref()).map_err(|error| {
                        SendError(format!("WeCom websocket returned invalid JSON: {error}"))
                    })?;
                    if wecom_payload_req_id(&payload).as_deref() == Some(expected_req_id) {
                        return Ok(payload);
                    }
                }
                WsMessage::Binary(_) | WsMessage::Ping(_) | WsMessage::Pong(_) => {}
                WsMessage::Close(_) => {
                    return Err(SendError(
                        "WeCom websocket closed before responding".to_string(),
                    ));
                }
                WsMessage::Frame(_) => {}
            }
        }
    }
}

fn wecom_payload_req_id(payload: &Value) -> Option<String> {
    payload
        .get("headers")
        .and_then(Value::as_object)
        .and_then(|headers| headers.get("req_id"))
        .and_then(value_as_string)
}

fn wecom_response_error(response: &Value) -> Option<String> {
    let errcode = response.get("errcode").and_then(value_as_i64).unwrap_or(0);
    if errcode == 0 {
        return None;
    }
    Some(format!(
        "errcode {}: {}",
        errcode,
        response
            .get("errmsg")
            .and_then(Value::as_str)
            .unwrap_or("unknown error")
    ))
}

fn send_weixin(
    config: &WeixinConfig,
    runtime: &ToolRuntime,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    if target.thread_id.is_some() {
        return Err(SendError(
            "Weixin thread targets are not supported in the Rust send_message runtime.".to_string(),
        ));
    }

    let client = http_client_with_timeout(Duration::from_secs(WEIXIN_CDN_TIMEOUT_SECS))?;
    let context_token =
        load_weixin_context_token(runtime.hermes_home(), &config.account_id, &target.chat_id);
    let mut last_message_id = None;

    if !message.is_empty() {
        for chunk in weixin_split_text(message, config.split_multiline_messages) {
            let client_id = format!("hermes-weixin-{}", unique_suffix());
            weixin_send_text_chunk(
                &client,
                config,
                &target.chat_id,
                &chunk,
                context_token.as_deref(),
                &client_id,
            )?;
            last_message_id = Some(client_id);
        }
    }

    for attachment in media {
        last_message_id = Some(weixin_send_media(
            &client,
            config,
            &target.chat_id,
            attachment,
            context_token.as_deref(),
        )?);
    }

    Ok(SentMessage {
        platform: PlatformKind::Weixin,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: None,
        note: target
            .used_home_channel
            .then(|| format!("Sent to weixin home channel (chat_id: {})", target.chat_id)),
    })
}

fn load_weixin_context_token(
    hermes_home: &Path,
    account_id: &str,
    chat_id: &str,
) -> Option<String> {
    let path = hermes_home
        .join("weixin")
        .join("accounts")
        .join(format!("{account_id}.context-tokens.json"));
    let content = fs::read_to_string(path).ok()?;
    let data: Value = serde_json::from_str(&content).ok()?;
    data.get(chat_id)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn weixin_send_text_chunk(
    client: &Client,
    config: &WeixinConfig,
    chat_id: &str,
    chunk: &str,
    context_token: Option<&str>,
    client_id: &str,
) -> Result<(), SendError> {
    let mut active_context_token = context_token.map(ToOwned::to_owned);
    for _attempt in 0..2 {
        let response = weixin_send_text_message(
            client,
            config,
            chat_id,
            chunk,
            active_context_token.as_deref(),
            client_id,
        )?;
        if weixin_message_succeeded(&response) {
            return Ok(());
        }
        if active_context_token.is_some() && weixin_is_stale_session(&response) {
            active_context_token = None;
            continue;
        }
        return Err(SendError(format!(
            "Weixin send failed: {}",
            weixin_response_error(&response)
        )));
    }
    Err(SendError(
        "Weixin send failed after retrying without context_token".to_string(),
    ))
}

fn weixin_send_text_message(
    client: &Client,
    config: &WeixinConfig,
    chat_id: &str,
    text: &str,
    context_token: Option<&str>,
    client_id: &str,
) -> Result<Value, SendError> {
    let mut message = serde_json::Map::new();
    message.insert("from_user_id".to_string(), json!(""));
    message.insert("to_user_id".to_string(), json!(chat_id));
    message.insert("client_id".to_string(), json!(client_id));
    message.insert("message_type".to_string(), json!(2));
    message.insert("message_state".to_string(), json!(2));
    message.insert(
        "item_list".to_string(),
        json!([{
            "type": WEIXIN_ITEM_TEXT,
            "text_item": { "text": text },
        }]),
    );
    if let Some(context_token) = context_token {
        message.insert("context_token".to_string(), json!(context_token));
    }
    weixin_api_post(
        client,
        &config.base_url,
        "ilink/bot/sendmessage",
        json!({ "msg": Value::Object(message) }),
        Some(&config.token),
        "Weixin send failed",
    )
}

fn weixin_send_media(
    client: &Client,
    config: &WeixinConfig,
    chat_id: &str,
    attachment: &MediaAttachment,
    context_token: Option<&str>,
) -> Result<String, SendError> {
    let path = &attachment.path;
    let plaintext = fs::read(path)
        .map_err(|error| SendError(format!("Reading media {} failed: {error}", path.display())))?;
    let mime_type = guess_mime_type(path, &plaintext);
    let (media_type, item) =
        weixin_build_media_item(client, config, chat_id, path, &mime_type, &plaintext)?;
    let client_id = format!("hermes-weixin-{}", unique_suffix());
    let mut message = serde_json::Map::new();
    message.insert("from_user_id".to_string(), json!(""));
    message.insert("to_user_id".to_string(), json!(chat_id));
    message.insert("client_id".to_string(), json!(&client_id));
    message.insert("message_type".to_string(), json!(2));
    message.insert("message_state".to_string(), json!(2));
    message.insert("item_list".to_string(), json!([item]));
    if let Some(context_token) = context_token {
        message.insert("context_token".to_string(), json!(context_token));
    }
    let response = weixin_api_post(
        client,
        &config.base_url,
        "ilink/bot/sendmessage",
        json!({ "msg": Value::Object(message) }),
        Some(&config.token),
        "Weixin media send failed",
    )?;
    if !weixin_message_succeeded(&response) {
        return Err(SendError(format!(
            "Weixin media send failed: {}",
            weixin_response_error(&response)
        )));
    }
    let _ = media_type;
    Ok(client_id)
}

fn weixin_build_media_item(
    client: &Client,
    config: &WeixinConfig,
    chat_id: &str,
    path: &Path,
    mime_type: &str,
    plaintext: &[u8],
) -> Result<(i64, Value), SendError> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("attachment.bin")
        .to_string();
    let path_text = path.display().to_string();
    let (media_type, item_type) = if mime_type.starts_with("image/") {
        (WEIXIN_MEDIA_IMAGE, WEIXIN_ITEM_IMAGE)
    } else if mime_type.starts_with("video/") {
        (WEIXIN_MEDIA_VIDEO, WEIXIN_ITEM_VIDEO)
    } else if path_text.ends_with(".silk") {
        (WEIXIN_MEDIA_VOICE, WEIXIN_ITEM_VOICE)
    } else {
        (WEIXIN_MEDIA_FILE, WEIXIN_ITEM_FILE)
    };

    let mut aes_key = [0_u8; 16];
    fill_random(&mut aes_key)
        .map_err(|error| SendError(format!("Weixin media encryption setup failed: {error}")))?;
    let filekey = random_hex(16)?;
    let ciphertext = weixin_encrypt_ecb(plaintext, &aes_key)?;
    let upload = weixin_api_post(
        client,
        &config.base_url,
        "ilink/bot/getuploadurl",
        json!({
            "filekey": filekey,
            "media_type": media_type,
            "to_user_id": chat_id,
            "rawsize": plaintext.len(),
            "rawfilemd5": format!("{:x}", md5::compute(plaintext)),
            "filesize": ciphertext.len(),
            "no_need_thumb": true,
            "aeskey": hex_string(&aes_key),
        }),
        Some(&config.token),
        "Weixin upload URL request failed",
    )?;
    if !weixin_message_succeeded(&upload) {
        return Err(SendError(format!(
            "Weixin upload URL request failed: {}",
            weixin_response_error(&upload)
        )));
    }
    let upload_url = upload
        .get("upload_full_url")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            upload
                .get("upload_param")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| {
                    format!(
                        "{}/upload?encrypted_query_param={}&filekey={}",
                        config.cdn_base_url,
                        url::form_urlencoded::byte_serialize(value.as_bytes()).collect::<String>(),
                        filekey
                    )
                })
        })
        .ok_or_else(|| {
            SendError(
                "Weixin upload URL request failed: missing upload_full_url or upload_param"
                    .to_string(),
            )
        })?;
    let encrypted_query_param = weixin_upload_ciphertext(client, &upload_url, &ciphertext)?;
    let aes_key_for_api = BASE64_STANDARD.encode(hex_string(&aes_key).as_bytes());

    let media_ref = json!({
        "encrypt_query_param": encrypted_query_param,
        "aes_key": aes_key_for_api,
        "encrypt_type": 1,
    });
    let item = match item_type {
        WEIXIN_ITEM_IMAGE => json!({
            "type": WEIXIN_ITEM_IMAGE,
            "image_item": {
                "media": media_ref,
                "mid_size": ciphertext.len(),
            }
        }),
        WEIXIN_ITEM_VIDEO => json!({
            "type": WEIXIN_ITEM_VIDEO,
            "video_item": {
                "media": media_ref,
                "video_size": ciphertext.len(),
                "play_length": 0,
                "video_md5": format!("{:x}", md5::compute(plaintext)),
            }
        }),
        WEIXIN_ITEM_VOICE => json!({
            "type": WEIXIN_ITEM_VOICE,
            "voice_item": {
                "media": media_ref,
                "encode_type": 6,
                "bits_per_sample": 16,
                "sample_rate": 24000,
                "playtime": 0,
            }
        }),
        _ => json!({
            "type": WEIXIN_ITEM_FILE,
            "file_item": {
                "media": media_ref,
                "file_name": file_name,
                "len": plaintext.len().to_string(),
            }
        }),
    };
    Ok((media_type, item))
}

fn weixin_upload_ciphertext(
    client: &Client,
    upload_url: &str,
    ciphertext: &[u8],
) -> Result<String, SendError> {
    let response = client
        .post(upload_url)
        .header("Content-Type", "application/octet-stream")
        .body(ciphertext.to_vec())
        .send()
        .map_err(|error| SendError(format!("Weixin CDN upload failed: {error}")))?;
    let status = response.status();
    let encrypted_param = response
        .headers()
        .get("x-encrypted-param")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let body = response
        .text()
        .map_err(|error| SendError(format!("Weixin CDN upload failed: {error}")))?;
    if !status.is_success() {
        return Err(SendError(format!(
            "Weixin CDN upload failed: HTTP {}: {}",
            status.as_u16(),
            body
        )));
    }
    encrypted_param.ok_or_else(|| {
        SendError(format!(
            "Weixin CDN upload failed: missing x-encrypted-param header: {body}"
        ))
    })
}

fn weixin_api_post(
    client: &Client,
    base_url: &str,
    endpoint: &str,
    payload: Value,
    token: Option<&str>,
    context: &str,
) -> Result<Value, SendError> {
    let mut request_body = payload.as_object().cloned().unwrap_or_default();
    request_body.insert(
        "base_info".to_string(),
        json!({ "channel_version": WEIXIN_CHANNEL_VERSION }),
    );
    let body = serde_json::to_string(&Value::Object(request_body))
        .map_err(|error| SendError(format!("{context}: failed to encode request body: {error}")))?;
    let mut request = client
        .post(format!("{}/{}", base_url.trim_end_matches('/'), endpoint))
        .header("Content-Type", "application/json")
        .header("AuthorizationType", "ilink_bot_token")
        .header("Content-Length", body.as_bytes().len().to_string())
        .header("X-WECHAT-UIN", random_wechat_uin()?)
        .header("iLink-App-Id", WEIXIN_APP_ID)
        .header(
            "iLink-App-ClientVersion",
            WEIXIN_APP_CLIENT_VERSION.to_string(),
        )
        .body(body);
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let response = request
        .send()
        .map_err(|error| SendError(format!("{context}: {error}")))?;
    parse_json_response(response, context)
}

fn weixin_split_text(content: &str, split_multiline: bool) -> Vec<String> {
    let formatted = weixin_format_message(content);
    if formatted.is_empty() {
        return Vec::new();
    }
    if !split_multiline {
        return split_text_fixed_width(&formatted, WEIXIN_MAX_MESSAGE_LENGTH);
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    for line in formatted.lines() {
        let candidate = if current.is_empty() {
            line.to_string()
        } else {
            format!("{current}\n{line}")
        };
        if candidate.chars().count() <= WEIXIN_MAX_MESSAGE_LENGTH {
            current = candidate;
            continue;
        }
        if !current.is_empty() {
            chunks.push(current);
            current = String::new();
        }
        let line_chunks = split_text_fixed_width(line, WEIXIN_MAX_MESSAGE_LENGTH);
        if let Some((last, rest)) = line_chunks.split_last() {
            chunks.extend(rest.iter().cloned());
            current = last.clone();
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn split_text_fixed_width(content: &str, max_chars: usize) -> Vec<String> {
    if content.chars().count() <= max_chars {
        return vec![content.trim().to_string()];
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in content.chars() {
        if current.chars().count() >= max_chars {
            out.push(current.trim().to_string());
            current.clear();
        }
        current.push(ch);
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out
}

fn weixin_format_message(content: &str) -> String {
    let mut lines = Vec::new();
    let mut blank_run = false;
    for raw_line in content.lines() {
        let trimmed = raw_line.trim_end();
        if trimmed.is_empty() {
            if !blank_run {
                lines.push(String::new());
            }
            blank_run = true;
            continue;
        }
        blank_run = false;
        lines.push(rewrite_weixin_heading(trimmed));
    }
    lines.join("\n").trim().to_string()
}

fn rewrite_weixin_heading(line: &str) -> String {
    let trimmed = line.trim();
    let level = trimmed.chars().take_while(|ch| *ch == '#').count();
    if level == 0 {
        return trimmed.to_string();
    }
    let title = trimmed[level..].trim();
    if title.is_empty() {
        return trimmed.to_string();
    }
    if level == 1 {
        format!("【{title}】")
    } else {
        format!("**{title}**")
    }
}

fn weixin_message_succeeded(response: &Value) -> bool {
    response.get("ret").and_then(value_as_i64).unwrap_or(0) == 0
        && response.get("errcode").and_then(value_as_i64).unwrap_or(0) == 0
}

fn weixin_is_stale_session(response: &Value) -> bool {
    let ret = response.get("ret").and_then(value_as_i64);
    let errcode = response.get("errcode").and_then(value_as_i64);
    if ret == Some(-14) || errcode == Some(-14) {
        return true;
    }
    (ret == Some(-2) || errcode == Some(-2))
        && response
            .get("errmsg")
            .and_then(Value::as_str)
            .is_some_and(|value| value.eq_ignore_ascii_case("unknown error"))
}

fn weixin_response_error(response: &Value) -> String {
    let ret = response.get("ret").and_then(value_as_i64).unwrap_or(0);
    let errcode = response.get("errcode").and_then(value_as_i64).unwrap_or(0);
    let errmsg = response
        .get("errmsg")
        .or_else(|| response.get("msg"))
        .and_then(Value::as_str)
        .unwrap_or("unknown error");
    format!("ret={ret} errcode={errcode} errmsg={errmsg}")
}

fn random_wechat_uin() -> Result<String, SendError> {
    let mut bytes = [0_u8; 4];
    fill_random(&mut bytes)
        .map_err(|error| SendError(format!("Weixin random header generation failed: {error}")))?;
    let value = u32::from_be_bytes(bytes);
    Ok(BASE64_STANDARD.encode(value.to_string()))
}

fn random_hex(len_bytes: usize) -> Result<String, SendError> {
    let mut bytes = vec![0_u8; len_bytes];
    fill_random(&mut bytes)
        .map_err(|error| SendError(format!("Weixin random generation failed: {error}")))?;
    Ok(hex_string(&bytes))
}

fn hex_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn weixin_encrypt_ecb(plaintext: &[u8], key: &[u8; 16]) -> Result<Vec<u8>, SendError> {
    let mut padded = plaintext.to_vec();
    let pad_len = 16 - (padded.len() % 16);
    padded.extend(std::iter::repeat_n(pad_len as u8, pad_len));
    let cipher = Aes128::new(&Array::from(*key));
    for chunk in padded.chunks_mut(16) {
        let block = <&mut aes::Block>::try_from(chunk).map_err(|_| {
            SendError("Failed to map Weixin upload payload into AES blocks.".to_string())
        })?;
        cipher.encrypt_block(block);
    }
    Ok(padded)
}

fn send_email(
    config: &EmailConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    if target.thread_id.is_some() {
        return Err(SendError(
            "Email thread targets are not supported in the Rust send_message runtime.".to_string(),
        ));
    }

    let email = build_email_message(config, &target.chat_id, message, media)?;
    let transport = build_email_transport(config)?;
    let response = transport
        .send(&email)
        .map_err(|error| SendError(format!("Email send failed: {error}")))?;
    let message_id = response.first_line().map(str::to_string);

    Ok(SentMessage {
        platform: PlatformKind::Email,
        chat_id: target.chat_id.clone(),
        message_id,
        thread_id: None,
        note: target
            .used_home_channel
            .then(|| format!("Sent to email home channel (chat_id: {})", target.chat_id)),
    })
}

fn build_email_message(
    config: &EmailConfig,
    to_address: &str,
    message: &str,
    media: &[MediaAttachment],
) -> Result<EmailMessage, SendError> {
    let from = email_mailbox(&config.address, "EMAIL_ADDRESS")?;
    let to = email_mailbox(to_address, "target")?;
    let builder = EmailMessage::builder()
        .from(from)
        .to(to)
        .subject("Hermes Agent");

    if media.is_empty() {
        return builder
            .header(ContentType::TEXT_PLAIN)
            .body(message.to_string())
            .map_err(|error| SendError(format!("Email send failed: {error}")));
    }

    let mut multipart = MultiPart::mixed().build();
    if !message.is_empty() {
        multipart = multipart.singlepart(SinglePart::plain(message.to_string()));
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
        let mime_type = ContentType::parse(&guess_mime_type(&attachment.path, &bytes))
            .map_err(|error| SendError(format!("Email attachment type is invalid: {error}")))?;
        multipart = multipart.singlepart(EmailAttachment::new(file_name).body(bytes, mime_type));
    }

    builder
        .multipart(multipart)
        .map_err(|error| SendError(format!("Email send failed: {error}")))
}

fn build_email_transport(config: &EmailConfig) -> Result<SmtpTransport, SendError> {
    let credentials = Credentials::new(config.address.clone(), config.password.clone());
    let builder = match config.smtp_security {
        EmailSecurity::StartTls => SmtpTransport::starttls_relay(&config.smtp_host)
            .map_err(|error| SendError(format!("Email SMTP setup failed: {error}")))?,
        EmailSecurity::Tls => SmtpTransport::relay(&config.smtp_host)
            .map_err(|error| SendError(format!("Email SMTP setup failed: {error}")))?,
        EmailSecurity::None => SmtpTransport::builder_dangerous(&config.smtp_host).tls(Tls::None),
    };
    Ok(builder
        .port(config.smtp_port)
        .timeout(Some(Duration::from_secs(REQUEST_TIMEOUT_SECS)))
        .credentials(credentials)
        .authentication(vec![Mechanism::Plain])
        .build())
}

fn email_mailbox(raw: &str, key: &str) -> Result<Mailbox, SendError> {
    raw.trim()
        .parse::<Mailbox>()
        .map_err(|error| SendError(format!("{key} is not a valid email address: {error}")))
}

fn send_whatsapp(
    config: &WhatsAppConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    if target.thread_id.is_some() {
        return Err(SendError(
            "WhatsApp thread targets are not supported in the Rust send_message runtime."
                .to_string(),
        ));
    }

    let client = http_client_with_timeout(Duration::from_secs(120))?;
    let mut last_message_id = None;

    if !message.is_empty() {
        let response = client
            .post(whatsapp_bridge_url(config, "/send")?)
            .json(&json!({
                "chatId": target.chat_id,
                "message": message,
            }))
            .send()
            .map_err(|error| SendError(format!("WhatsApp send failed: {error}")))?;
        let body = parse_json_response(response, "WhatsApp send failed")?;
        last_message_id = body.get("messageId").and_then(value_as_string);
    }

    for attachment in media {
        let file_name = attachment
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("attachment.bin")
            .to_string();
        let response = client
            .post(whatsapp_bridge_url(config, "/send-media")?)
            .json(&json!({
                "chatId": target.chat_id,
                "filePath": attachment.path.display().to_string(),
                "fileName": file_name,
                "mediaType": whatsapp_media_type(attachment),
            }))
            .send()
            .map_err(|error| SendError(format!("WhatsApp media send failed: {error}")))?;
        let body = parse_json_response(response, "WhatsApp media send failed")?;
        last_message_id = body.get("messageId").and_then(value_as_string);
    }

    Ok(SentMessage {
        platform: PlatformKind::WhatsApp,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: None,
        note: target.used_home_channel.then(|| {
            format!(
                "Sent to whatsapp home channel (chat_id: {})",
                target.chat_id
            )
        }),
    })
}

fn whatsapp_bridge_url(config: &WhatsAppConfig, path: &str) -> Result<Url, SendError> {
    config
        .bridge_url
        .parse::<Url>()
        .map_err(|error| SendError(format!("WhatsApp bridge URL is invalid: {error}")))?
        .join(path)
        .map_err(|error| SendError(format!("WhatsApp bridge path is invalid: {error}")))
}

fn whatsapp_media_type(attachment: &MediaAttachment) -> &'static str {
    let ext = file_extension(&attachment.path);
    if IMAGE_EXTS.contains(&ext.as_str()) {
        return "image";
    }
    if VIDEO_EXTS.contains(&ext.as_str()) {
        return "video";
    }
    if matches!(ext.as_str(), "ogg" | "opus" | "mp3" | "wav" | "m4a") {
        return "audio";
    }
    "document"
}

fn send_dingtalk(
    config: &DingTalkConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    if !media.is_empty() {
        return Err(SendError(
            "DingTalk media attachments are not supported in the Rust send_message runtime."
                .to_string(),
        ));
    }
    if target.thread_id.is_some() {
        return Err(SendError(
            "DingTalk thread targets are not supported in the Rust send_message runtime."
                .to_string(),
        ));
    }
    let webhook_url = config.webhook_url.as_deref().ok_or_else(|| {
        SendError(
            "DingTalk send_message requires DINGTALK_WEBHOOK_URL for static robot delivery."
                .to_string(),
        )
    })?;
    let response = http_client()?
        .post(webhook_url)
        .json(&json!({
            "msgtype": "text",
            "text": { "content": message },
        }))
        .send()
        .map_err(|error| {
            SendError(format!(
                "DingTalk send failed: {}",
                redact_dingtalk_error(webhook_url, &error.to_string())
            ))
        })?;
    let body = parse_json_response(response, "DingTalk send failed")?;
    let errcode = body.get("errcode").and_then(value_as_i64).unwrap_or(0);
    if errcode != 0 {
        let errmsg = body
            .get("errmsg")
            .and_then(value_as_string)
            .unwrap_or_else(|| format!("errcode {errcode}"));
        return Err(SendError(format!("DingTalk API error: {errmsg}")));
    }
    Ok(SentMessage {
        platform: PlatformKind::DingTalk,
        chat_id: target.chat_id.clone(),
        message_id: None,
        thread_id: None,
        note: target.used_home_channel.then(|| {
            format!(
                "Sent to dingtalk home channel (chat_id: {})",
                target.chat_id
            )
        }),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QqBotTargetKind {
    Auto,
    User,
    Group,
    Channel,
}

fn send_qqbot(
    config: &QqBotConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    if target.thread_id.is_some() {
        return Err(SendError(
            "QQBot thread targets are not supported in the Rust send_message runtime.".to_string(),
        ));
    }

    let client = http_client_with_timeout(Duration::from_secs(120))?;
    let token = qqbot_access_token(&client, config)?;
    let (kind, chat_id) = qqbot_target_parts(&target.chat_id);
    let mut last_message_id = None;

    if media.is_empty() && !message.is_empty() {
        last_message_id = Some(send_qqbot_text(
            &client, config, &token, kind, chat_id, message,
        )?);
    }
    if !media.is_empty() {
        last_message_id = Some(send_qqbot_media(
            &client, config, &token, kind, chat_id, message, media,
        )?);
    }

    Ok(SentMessage {
        platform: PlatformKind::QqBot,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: None,
        note: target
            .used_home_channel
            .then(|| format!("Sent to qqbot home channel (chat_id: {})", target.chat_id)),
    })
}

fn qqbot_target_parts(chat_id: &str) -> (QqBotTargetKind, &str) {
    if let Some(value) = chat_id.strip_prefix("user:") {
        return (QqBotTargetKind::User, value);
    }
    if let Some(value) = chat_id.strip_prefix("group:") {
        return (QqBotTargetKind::Group, value);
    }
    if let Some(value) = chat_id.strip_prefix("channel:") {
        return (QqBotTargetKind::Channel, value);
    }
    (QqBotTargetKind::Auto, chat_id)
}

fn qqbot_access_token(client: &Client, config: &QqBotConfig) -> Result<String, SendError> {
    let response = client
        .post(&config.token_url)
        .json(&json!({
            "appId": config.app_id,
            "clientSecret": config.client_secret,
        }))
        .send()
        .map_err(|error| SendError(format!("QQBot token request failed: {error}")))?;
    let body = parse_json_response(response, "QQBot token request failed")?;
    body.get("access_token")
        .and_then(value_as_string)
        .ok_or_else(|| SendError("QQBot token request failed: missing access_token".to_string()))
}

fn send_qqbot_text(
    client: &Client,
    config: &QqBotConfig,
    token: &str,
    kind: QqBotTargetKind,
    chat_id: &str,
    message: &str,
) -> Result<String, SendError> {
    let payload = json!({
        "content": message.chars().take(4000).collect::<String>(),
        "msg_type": 0,
    });
    let attempts = match kind {
        QqBotTargetKind::Channel => vec![format!("/channels/{chat_id}/messages")],
        QqBotTargetKind::User => vec![format!("/v2/users/{chat_id}/messages")],
        QqBotTargetKind::Group => vec![format!("/v2/groups/{chat_id}/messages")],
        QqBotTargetKind::Auto => vec![
            format!("/channels/{chat_id}/messages"),
            format!("/v2/users/{chat_id}/messages"),
            format!("/v2/groups/{chat_id}/messages"),
        ],
    };
    qqbot_try_send(
        client,
        config,
        token,
        attempts,
        payload,
        "QQBot send failed",
    )
}

fn send_qqbot_media(
    client: &Client,
    config: &QqBotConfig,
    token: &str,
    kind: QqBotTargetKind,
    chat_id: &str,
    message: &str,
    media: &[MediaAttachment],
) -> Result<String, SendError> {
    if kind == QqBotTargetKind::Channel {
        return Err(SendError(
            "QQBot channel targets do not support native media in the Rust send_message runtime."
                .to_string(),
        ));
    }

    let attempts = match kind {
        QqBotTargetKind::User => vec![(
            format!("/v2/users/{chat_id}/files"),
            format!("/v2/users/{chat_id}/messages"),
        )],
        QqBotTargetKind::Group => vec![(
            format!("/v2/groups/{chat_id}/files"),
            format!("/v2/groups/{chat_id}/messages"),
        )],
        QqBotTargetKind::Auto => vec![
            (
                format!("/v2/users/{chat_id}/files"),
                format!("/v2/users/{chat_id}/messages"),
            ),
            (
                format!("/v2/groups/{chat_id}/files"),
                format!("/v2/groups/{chat_id}/messages"),
            ),
        ],
        QqBotTargetKind::Channel => Vec::new(),
    };

    let mut last_message_id = None;
    for attachment in media {
        let bytes = fs::read(&attachment.path).map_err(|error| {
            SendError(format!(
                "Reading media {} failed: {error}",
                attachment.path.display()
            ))
        })?;
        let (file_type, label) = qqbot_media_type(attachment);
        let file_name = attachment
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("attachment.bin")
            .to_string();
        let mut upload_payload = serde_json::Map::new();
        upload_payload.insert("file_type".to_string(), json!(file_type));
        upload_payload.insert("srv_send_msg".to_string(), Value::Bool(false));
        upload_payload.insert(
            "file_data".to_string(),
            Value::String(BASE64_STANDARD.encode(bytes)),
        );
        if file_type == 4 {
            upload_payload.insert("file_name".to_string(), Value::String(file_name));
        }

        let mut failures = Vec::new();
        let mut sent_message_id = None;
        for (upload_path, send_path) in &attempts {
            let upload_response = match qqbot_try_json_request(
                client,
                config,
                token,
                std::slice::from_ref(upload_path),
                Value::Object(upload_payload.clone()),
                &format!("QQBot {label} upload failed"),
            ) {
                Ok(value) => value,
                Err(error) => {
                    failures.push(error.to_string());
                    continue;
                }
            };
            let Some(file_info) = upload_response.get("file_info").cloned() else {
                failures.push(format!("QQBot {label} upload failed: missing file_info"));
                continue;
            };
            let mut send_payload = serde_json::Map::new();
            send_payload.insert("msg_type".to_string(), json!(7));
            send_payload.insert("media".to_string(), json!({ "file_info": file_info }));
            if !message.is_empty() {
                send_payload.insert(
                    "content".to_string(),
                    Value::String(message.chars().take(4000).collect()),
                );
            }
            match qqbot_try_send(
                client,
                config,
                token,
                vec![send_path.clone()],
                Value::Object(send_payload),
                &format!("QQBot {label} send failed"),
            ) {
                Ok(message_id) => {
                    sent_message_id = Some(message_id);
                    break;
                }
                Err(error) => failures.push(error.to_string()),
            }
        }
        last_message_id = Some(sent_message_id.ok_or_else(|| {
            SendError(format!(
                "QQBot {label} send failed: {}",
                failures.join("; ")
            ))
        })?);
    }

    last_message_id.ok_or_else(|| SendError("QQBot media send failed: no attachments".to_string()))
}

fn qqbot_media_type(attachment: &MediaAttachment) -> (u8, &'static str) {
    let ext = file_extension(&attachment.path);
    if IMAGE_EXTS.contains(&ext.as_str()) {
        return (1, "image");
    }
    if VIDEO_EXTS.contains(&ext.as_str()) {
        return (2, "video");
    }
    if matches!(
        ext.as_str(),
        "ogg" | "opus" | "mp3" | "wav" | "m4a" | "aac" | "flac" | "amr" | "silk"
    ) {
        return (3, "voice");
    }
    (4, "file")
}

fn qqbot_try_send(
    client: &Client,
    config: &QqBotConfig,
    token: &str,
    paths: Vec<String>,
    payload: Value,
    context: &str,
) -> Result<String, SendError> {
    let body = qqbot_try_json_request(client, config, token, &paths, payload, context)?;
    Ok(body
        .get("id")
        .and_then(value_as_string)
        .unwrap_or_else(|| unique_suffix().to_string()))
}

fn qqbot_try_json_request(
    client: &Client,
    config: &QqBotConfig,
    token: &str,
    paths: &[String],
    payload: Value,
    context: &str,
) -> Result<Value, SendError> {
    let mut failures = Vec::new();
    for path in paths {
        let response = client
            .post(format!("{}{}", config.base_url, path))
            .header("Authorization", format!("QQBot {token}"))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .map_err(|error| SendError(format!("{context}: {error}")))?;
        let status = response.status();
        let raw = response.text().map_err(|error| {
            SendError(format!("{context}: failed to read response body: {error}"))
        })?;
        if status.is_success() {
            return serde_json::from_str(&raw)
                .map_err(|error| SendError(format!("{context}: invalid JSON response: {error}")));
        }
        failures.push(format!("{} {}", path, status.as_u16()));
    }
    Err(SendError(format!("{context}: {}", failures.join(", "))))
}

fn send_sms(
    config: &SmsConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    if !media.is_empty() {
        return Err(SendError(
            "SMS media attachments are not supported in the Rust send_message runtime.".to_string(),
        ));
    }
    if target.thread_id.is_some() {
        return Err(SendError(
            "SMS thread targets are not supported in the Rust send_message runtime.".to_string(),
        ));
    }

    let client = http_client()?;
    let url = format!("{}/{}/Messages.json", config.base_url, config.account_sid);
    let response = client
        .post(url)
        .basic_auth(&config.account_sid, Some(&config.auth_token))
        .form(&[
            ("From", config.from_number.as_str()),
            ("To", target.chat_id.as_str()),
            ("Body", message),
        ])
        .send()
        .map_err(|error| SendError(format!("SMS send failed: {error}")))?;
    let body = parse_json_response(response, "SMS send failed")?;
    let message_id = body
        .get("sid")
        .and_then(value_as_string)
        .ok_or_else(|| SendError("SMS send failed: missing sid".to_string()))?;

    Ok(SentMessage {
        platform: PlatformKind::Sms,
        chat_id: target.chat_id.clone(),
        message_id: Some(message_id),
        thread_id: None,
        note: target
            .used_home_channel
            .then(|| format!("Sent to sms home channel (chat_id: {})", target.chat_id)),
    })
}

fn send_homeassistant(
    config: &HomeAssistantConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    if !media.is_empty() {
        return Err(SendError(
            "Home Assistant media attachments are not supported in the Rust send_message runtime."
                .to_string(),
        ));
    }
    if target.thread_id.is_some() {
        return Err(SendError(
            "Home Assistant thread targets are not supported in the Rust send_message runtime."
                .to_string(),
        ));
    }

    let client = http_client_with_timeout(Duration::from_secs(10))?;
    let mut payload = serde_json::Map::new();
    payload.insert(
        "title".to_string(),
        Value::String("Hermes Agent".to_string()),
    );
    payload.insert(
        "message".to_string(),
        Value::String(message.chars().take(4096).collect()),
    );
    if !target.chat_id.trim().is_empty() {
        payload.insert(
            "notification_id".to_string(),
            Value::String(target.chat_id.clone()),
        );
    }

    let response = client
        .post(format!(
            "{}/api/services/persistent_notification/create",
            config.base_url
        ))
        .bearer_auth(&config.token)
        .header("Content-Type", "application/json")
        .json(&Value::Object(payload))
        .send()
        .map_err(|error| SendError(format!("Home Assistant send failed: {error}")))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        return Err(SendError(format!(
            "Home Assistant send failed: HTTP {}: {}",
            status.as_u16(),
            body
        )));
    }

    Ok(SentMessage {
        platform: PlatformKind::HomeAssistant,
        chat_id: target.chat_id.clone(),
        message_id: Some(format!("ha-{}", unique_suffix())),
        thread_id: None,
        note: target.used_home_channel.then(|| {
            format!(
                "Sent to homeassistant notification target (chat_id: {})",
                target.chat_id
            )
        }),
    })
}

fn send_bluebubbles(
    config: &BlueBubblesConfig,
    target: &ResolvedTarget,
    message: &str,
    media: &[MediaAttachment],
) -> Result<SentMessage, SendError> {
    let client = http_client_with_timeout(Duration::from_secs(120))?;
    let guid = if let Some(guid) = lookup_bluebubbles_chat_guid(&client, config, &target.chat_id)? {
        guid
    } else if is_bluebubbles_address(&target.chat_id) && !message.is_empty() && media.is_empty() {
        let message_id =
            create_bluebubbles_chat_for_handle(&client, config, &target.chat_id, message)?;
        return Ok(SentMessage {
            platform: PlatformKind::BlueBubbles,
            chat_id: target.chat_id.clone(),
            message_id: Some(message_id),
            thread_id: None,
            note: target.used_home_channel.then(|| {
                format!(
                    "Sent to bluebubbles home channel (chat_id: {})",
                    target.chat_id
                )
            }),
        });
    } else {
        return Err(SendError(format!(
            "BlueBubbles chat not found for target: {}",
            target.chat_id
        )));
    };
    let mut last_message_id = None;

    if !message.is_empty() {
        last_message_id = Some(send_bluebubbles_text(&client, config, &guid, message)?);
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
        last_message_id = Some(send_bluebubbles_attachment(
            &client,
            config,
            &guid,
            &file_name,
            &bytes,
            attachment.is_voice,
        )?);
    }

    Ok(SentMessage {
        platform: PlatformKind::BlueBubbles,
        chat_id: target.chat_id.clone(),
        message_id: last_message_id,
        thread_id: None,
        note: target.used_home_channel.then(|| {
            format!(
                "Sent to bluebubbles home channel (chat_id: {})",
                target.chat_id
            )
        }),
    })
}

fn bluebubbles_api_url(config: &BlueBubblesConfig, path: &str) -> Result<Url, SendError> {
    let mut url = Url::parse(&format!("{}{}", config.server_url, path))
        .map_err(|error| SendError(format!("BlueBubbles URL build failed: {error}")))?;
    url.query_pairs_mut()
        .append_pair("password", &config.password);
    Ok(url)
}

fn lookup_bluebubbles_chat_guid(
    client: &Client,
    config: &BlueBubblesConfig,
    target: &str,
) -> Result<Option<String>, SendError> {
    let trimmed = target.trim();
    if trimmed.is_empty() {
        return Err(SendError(
            "BlueBubbles target must not be empty".to_string(),
        ));
    }
    if trimmed.contains(';') {
        return Ok(Some(trimmed.to_string()));
    }

    let response = client
        .post(bluebubbles_api_url(config, "/api/v1/chat/query")?)
        .json(&json!({
            "limit": 100,
            "offset": 0,
            "with": ["participants"],
        }))
        .send()
        .map_err(|error| SendError(format!("BlueBubbles chat query failed: {error}")))?;
    let body = parse_json_response(response, "BlueBubbles chat query failed")?;
    if let Some(chats) = body.get("data").and_then(Value::as_array) {
        for chat in chats {
            let guid = chat
                .get("guid")
                .and_then(value_as_string)
                .or_else(|| chat.get("chatGuid").and_then(value_as_string));
            let identifier = chat
                .get("chatIdentifier")
                .and_then(value_as_string)
                .or_else(|| chat.get("identifier").and_then(value_as_string));
            if identifier.as_deref() == Some(trimmed)
                && let Some(guid) = guid.clone()
            {
                return Ok(Some(guid));
            }
            if let Some(participants) = chat.get("participants").and_then(Value::as_array) {
                for participant in participants {
                    if participant
                        .get("address")
                        .and_then(value_as_string)
                        .as_deref()
                        == Some(trimmed)
                        && let Some(guid) = guid.clone()
                    {
                        return Ok(Some(guid));
                    }
                }
            }
        }
    }
    Ok(None)
}

fn create_bluebubbles_chat_for_handle(
    client: &Client,
    config: &BlueBubblesConfig,
    address: &str,
    message: &str,
) -> Result<String, SendError> {
    let response = client
        .post(bluebubbles_api_url(config, "/api/v1/chat/new")?)
        .json(&json!({
            "addresses": [address],
            "message": message,
            "tempGuid": format!("temp-{}", unique_suffix()),
        }))
        .send()
        .map_err(|error| SendError(format!("BlueBubbles chat create failed: {error}")))?;
    let body = parse_json_response(response, "BlueBubbles chat create failed")?;
    body.pointer("/data/guid")
        .and_then(value_as_string)
        .or_else(|| body.pointer("/data/messageGuid").and_then(value_as_string))
        .ok_or_else(|| SendError("BlueBubbles chat create failed: missing message id".to_string()))
}

fn send_bluebubbles_text(
    client: &Client,
    config: &BlueBubblesConfig,
    guid: &str,
    message: &str,
) -> Result<String, SendError> {
    let response = client
        .post(bluebubbles_api_url(config, "/api/v1/message/text")?)
        .json(&json!({
            "chatGuid": guid,
            "tempGuid": format!("temp-{}", unique_suffix()),
            "message": message,
        }))
        .send()
        .map_err(|error| SendError(format!("BlueBubbles send failed: {error}")))?;
    let body = parse_json_response(response, "BlueBubbles send failed")?;
    Ok(body
        .pointer("/data/guid")
        .and_then(value_as_string)
        .or_else(|| body.pointer("/data/messageGuid").and_then(value_as_string))
        .unwrap_or_else(|| "ok".to_string()))
}

fn send_bluebubbles_attachment(
    client: &Client,
    config: &BlueBubblesConfig,
    guid: &str,
    file_name: &str,
    bytes: &[u8],
    is_audio_message: bool,
) -> Result<String, SendError> {
    let mut form = Form::new()
        .text("chatGuid", guid.to_string())
        .text("name", file_name.to_string())
        .text("tempGuid", format!("temp-{}", unique_suffix()))
        .part(
            "attachment",
            Part::bytes(bytes.to_vec()).file_name(file_name.to_string()),
        );
    if is_audio_message {
        form = form.text("isAudioMessage", "true");
    }
    let response = client
        .post(bluebubbles_api_url(config, "/api/v1/message/attachment")?)
        .multipart(form)
        .send()
        .map_err(|error| SendError(format!("BlueBubbles attachment send failed: {error}")))?;
    let body = parse_json_response(response, "BlueBubbles attachment send failed")?;
    Ok(body
        .pointer("/data/guid")
        .and_then(value_as_string)
        .or_else(|| body.pointer("/data/messageGuid").and_then(value_as_string))
        .unwrap_or_else(|| "ok".to_string()))
}

fn is_bluebubbles_address(value: &str) -> bool {
    value.contains('@')
        || value
            .strip_prefix('+')
            .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
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

fn guess_mime_type(path: &Path, bytes: &[u8]) -> String {
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
        "amr" => "audio/amr".to_string(),
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

fn redact_dingtalk_error(webhook_url: &str, message: &str) -> String {
    message.replace(webhook_url, &redact_dingtalk_webhook_url(webhook_url))
}

fn redact_dingtalk_webhook_url(webhook_url: &str) -> String {
    let Ok(mut url) = Url::parse(webhook_url) else {
        return webhook_url.to_string();
    };
    let query = url
        .query_pairs()
        .map(|(key, value)| {
            if key == "access_token" {
                (key.into_owned(), "***".to_string())
            } else {
                (key.into_owned(), value.into_owned())
            }
        })
        .collect::<Vec<_>>();
    {
        let mut pairs = url.query_pairs_mut();
        pairs.clear();
        for (key, value) in query {
            pairs.append_pair(&key, &value);
        }
    }
    url.to_string()
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

    use std::io::{BufRead, BufReader, Read, Write};
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

    fn mock_raw_http_server<F>(request_count: usize, handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: Fn(usize, String, Vec<u8>) -> (u16, Vec<(String, String)>, Vec<u8>) + Send + 'static,
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
                let (status, response_headers, response_body) =
                    handler(index, headers, body_bytes[..content_length].to_vec());
                let mut response = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nConnection: close\r\n",
                    response_body.len()
                );
                for (name, value) in response_headers {
                    response.push_str(&format!("{name}: {value}\r\n"));
                }
                response.push_str("\r\n");
                stream.write_all(response.as_bytes()).unwrap();
                stream.write_all(&response_body).unwrap();
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

    fn mock_smtp_server<F>(handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: FnOnce(Vec<String>, String) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let reader_stream = stream.try_clone().unwrap();
            let mut reader = BufReader::new(reader_stream);
            let mut commands = Vec::new();
            let mut data = String::new();

            stream
                .write_all(b"220 localhost ESMTP Hermes Test\r\n")
                .unwrap();
            stream.flush().unwrap();

            loop {
                let mut line = String::new();
                let read = reader.read_line(&mut line).unwrap();
                if read == 0 {
                    break;
                }
                let trimmed = line.trim_end_matches(['\r', '\n']).to_string();
                if trimmed.is_empty() {
                    continue;
                }
                commands.push(trimmed.clone());
                if trimmed.starts_with("EHLO ") || trimmed.starts_with("HELO ") {
                    stream
                        .write_all(b"250-localhost\r\n250-AUTH PLAIN\r\n250 OK\r\n")
                        .unwrap();
                } else if trimmed == "AUTH PLAIN" {
                    stream.write_all(b"334 \r\n").unwrap();
                } else if trimmed.starts_with("AUTH PLAIN ") {
                    stream
                        .write_all(b"235 2.7.0 Authentication successful\r\n")
                        .unwrap();
                } else if trimmed.starts_with("MAIL FROM:") || trimmed.starts_with("RCPT TO:") {
                    stream.write_all(b"250 2.1.5 OK\r\n").unwrap();
                } else if trimmed == "DATA" {
                    stream
                        .write_all(b"354 End data with <CR><LF>.<CR><LF>\r\n")
                        .unwrap();
                    loop {
                        let mut data_line = String::new();
                        let read = reader.read_line(&mut data_line).unwrap();
                        if read == 0 {
                            break;
                        }
                        if data_line == ".\r\n" || data_line == ".\n" {
                            break;
                        }
                        if let Some(rest) = data_line.strip_prefix("..") {
                            data.push('.');
                            data.push_str(rest);
                        } else {
                            data.push_str(&data_line);
                        }
                    }
                    stream
                        .write_all(b"250 2.0.0 queued as test-123\r\n")
                        .unwrap();
                } else if trimmed == "QUIT" {
                    stream.write_all(b"221 2.0.0 Bye\r\n").unwrap();
                    break;
                } else {
                    stream.write_all(b"250 OK\r\n").unwrap();
                }
                stream.flush().unwrap();
            }

            handler(commands, data);
        });
        (format!("127.0.0.1:{}", addr.port()), join)
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

    fn websocket_text_payload(message: Message) -> Value {
        match message {
            Message::Text(text) => serde_json::from_str(text.as_ref()).unwrap(),
            other => panic!("expected text websocket frame, got {other:?}"),
        }
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
    fn list_targets_reports_whatsapp_home_channel() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        with_env_var("WHATSAPP_ENABLED", Some("true"));
        with_env_var("WHATSAPP_HOME_CHANNEL", Some("1234567890@lid"));
        with_env_var("WHATSAPP_HOME_CHANNEL_NAME", Some("Owner DM"));
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["targets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["target"] == "whatsapp:1234567890@lid")
        );
        assert!(
            parsed["platforms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["platform"] == "whatsapp")
        );
    }

    #[test]
    fn list_targets_reports_dingtalk_home_channel() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        with_env_var("DINGTALK_HOME_CHANNEL", Some("cidhome=="));
        with_env_var("DINGTALK_HOME_CHANNEL_NAME", Some("DingTalk Home"));
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["targets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["target"] == "dingtalk:cidhome==")
        );
        assert!(
            parsed["platforms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["platform"] == "dingtalk")
        );
    }

    #[test]
    fn list_targets_reports_qqbot_home_channel() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        with_env_var("QQ_APP_ID", Some("qq-app"));
        with_env_var("QQ_CLIENT_SECRET", Some("qq-secret"));
        with_env_var("QQBOT_HOME_CHANNEL", Some("user-open"));
        with_env_var("QQBOT_HOME_CHANNEL_NAME", Some("QQ Home"));
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["targets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["target"] == "qqbot:user-open")
        );
        assert!(
            parsed["platforms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["platform"] == "qqbot")
        );
    }

    #[test]
    fn list_targets_reports_wecom_home_channel() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        with_env_var("WECOM_BOT_ID", Some("wecom-bot"));
        with_env_var("WECOM_SECRET", Some("wecom-secret"));
        with_env_var("WECOM_HOME_CHANNEL", Some("chat-home"));
        with_env_var("WECOM_HOME_CHANNEL_NAME", Some("WeCom Home"));
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["targets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["target"] == "wecom:chat-home")
        );
        assert!(
            parsed["platforms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["platform"] == "wecom")
        );
    }

    #[test]
    fn list_targets_reports_weixin_home_channel() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        with_env_var("WEIXIN_TOKEN", Some("wx-token"));
        with_env_var("WEIXIN_ACCOUNT_ID", Some("wx-account"));
        with_env_var("WEIXIN_HOME_CHANNEL", Some("wxid_home123"));
        with_env_var("WEIXIN_HOME_CHANNEL_NAME", Some("Weixin Home"));
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["targets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["target"] == "weixin:wxid_home123")
        );
        assert!(
            parsed["platforms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["platform"] == "weixin")
        );
    }

    #[test]
    fn list_targets_reports_email_home_channel() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        with_env_var("EMAIL_ADDRESS", Some("bot@example.com"));
        with_env_var("EMAIL_PASSWORD", Some("app-pass"));
        with_env_var("EMAIL_SMTP_HOST", Some("smtp.example.com"));
        with_env_var("EMAIL_HOME_ADDRESS", Some("ops@example.com"));
        with_env_var("EMAIL_HOME_ADDRESS_NAME", Some("Ops Inbox"));
        let result = handle_list_targets(&runtime);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed["targets"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["target"] == "email:ops@example.com")
        );
        assert!(
            parsed["platforms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["platform"] == "email")
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
    fn sends_mattermost_text_and_media() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("proof.png");
        fs::write(&media_path, b"\x89PNG\r\n\x1a\nmattermost").unwrap();
        let (base_url, join) = mock_server(3, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /api/v4/posts "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["channel_id"], json!("channel123"));
                assert_eq!(payload["message"], json!("hello mattermost"));
                assert_eq!(payload["root_id"], json!("thread456"));
                (200, json!({ "id": "post-1" }).to_string())
            }
            1 => {
                assert!(headers.starts_with("POST /api/v4/files "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("content-type: multipart/form-data;")
                );
                (
                    200,
                    json!({ "file_infos": [{ "id": "file-1" }] }).to_string(),
                )
            }
            2 => {
                assert!(headers.starts_with("POST /api/v4/posts "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["channel_id"], json!("channel123"));
                assert_eq!(payload["message"], json!(""));
                assert_eq!(payload["file_ids"], json!(["file-1"]));
                assert_eq!(payload["root_id"], json!("thread456"));
                (200, json!({ "id": "post-2" }).to_string())
            }
            _ => unreachable!(),
        });

        with_env_var("MATTERMOST_TOKEN", Some("mm-token"));
        with_env_var("MATTERMOST_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "mattermost:channel123:thread456",
                "message": format!("hello mattermost\nMEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("mattermost"));
        assert_eq!(parsed["chat_id"], json!("channel123"));
        assert_eq!(parsed["thread_id"], json!("thread456"));
        assert_eq!(parsed["message_id"], json!("post-2"));
        join.join().unwrap();
    }

    #[test]
    fn sends_wecom_text_via_websocket() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (ws_url, join) = mock_ws_server(move |mut websocket| {
            let subscribe = websocket_text_payload(websocket.read().unwrap());
            assert_eq!(subscribe["cmd"], json!("aibot_subscribe"));
            let subscribe_req_id = subscribe["headers"]["req_id"].as_str().unwrap();
            assert_eq!(subscribe["body"]["bot_id"], json!("wecom-bot"));
            assert_eq!(subscribe["body"]["secret"], json!("wecom-secret"));
            websocket
                .send(Message::Text(
                    json!({
                        "headers": { "req_id": subscribe_req_id },
                        "errcode": 0,
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();

            let send = websocket_text_payload(websocket.read().unwrap());
            assert_eq!(send["cmd"], json!("aibot_send_msg"));
            assert_eq!(send["body"]["chatid"], json!("chat-123"));
            assert_eq!(send["body"]["msgtype"], json!("markdown"));
            assert_eq!(send["body"]["markdown"]["content"], json!("hello wecom"));
            let send_req_id = send["headers"]["req_id"].as_str().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "headers": { "req_id": send_req_id },
                        "errcode": 0,
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();
        });
        with_env_var("WECOM_BOT_ID", Some("wecom-bot"));
        with_env_var("WECOM_SECRET", Some("wecom-secret"));
        with_env_var("WECOM_WEBSOCKET_URL", Some(&ws_url));
        let result = handle_send(
            &json!({
                "target": "wecom:chat-123",
                "message": "hello wecom",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("wecom"));
        assert_eq!(parsed["chat_id"], json!("chat-123"));
        join.join().unwrap();
    }

    #[test]
    fn sends_wecom_media_via_websocket_upload_flow() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("wecom.png");
        let media_bytes = b"\x89PNG\r\n\x1a\nwecom-media";
        fs::write(&media_path, media_bytes).unwrap();
        let expected_chunk = BASE64_STANDARD.encode(media_bytes);
        let (ws_url, join) = mock_ws_server(move |mut websocket| {
            let subscribe = websocket_text_payload(websocket.read().unwrap());
            let subscribe_req_id = subscribe["headers"]["req_id"].as_str().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "headers": { "req_id": subscribe_req_id },
                        "errcode": 0,
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();

            let upload_init = websocket_text_payload(websocket.read().unwrap());
            assert_eq!(upload_init["cmd"], json!("aibot_upload_media_init"));
            assert_eq!(upload_init["body"]["type"], json!("image"));
            assert_eq!(upload_init["body"]["filename"], json!("wecom.png"));
            assert_eq!(upload_init["body"]["total_chunks"], json!(1));
            let init_req_id = upload_init["headers"]["req_id"].as_str().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "headers": { "req_id": init_req_id },
                        "errcode": 0,
                        "body": { "upload_id": "upload-1" },
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();

            let upload_chunk = websocket_text_payload(websocket.read().unwrap());
            assert_eq!(upload_chunk["cmd"], json!("aibot_upload_media_chunk"));
            assert_eq!(upload_chunk["body"]["upload_id"], json!("upload-1"));
            assert_eq!(upload_chunk["body"]["chunk_index"], json!(0));
            assert_eq!(upload_chunk["body"]["base64_data"], json!(expected_chunk));
            let chunk_req_id = upload_chunk["headers"]["req_id"].as_str().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "headers": { "req_id": chunk_req_id },
                        "errcode": 0,
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();

            let upload_finish = websocket_text_payload(websocket.read().unwrap());
            assert_eq!(upload_finish["cmd"], json!("aibot_upload_media_finish"));
            assert_eq!(upload_finish["body"]["upload_id"], json!("upload-1"));
            let finish_req_id = upload_finish["headers"]["req_id"].as_str().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "headers": { "req_id": finish_req_id },
                        "errcode": 0,
                        "body": { "media_id": "media-1" },
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();

            let send_media = websocket_text_payload(websocket.read().unwrap());
            assert_eq!(send_media["cmd"], json!("aibot_send_msg"));
            assert_eq!(send_media["body"]["chatid"], json!("chat-123"));
            assert_eq!(send_media["body"]["msgtype"], json!("image"));
            assert_eq!(send_media["body"]["image"]["media_id"], json!("media-1"));
            let send_req_id = send_media["headers"]["req_id"].as_str().unwrap();
            websocket
                .send(Message::Text(
                    json!({
                        "headers": { "req_id": send_req_id },
                        "errcode": 0,
                    })
                    .to_string()
                    .into(),
                ))
                .unwrap();
        });
        with_env_var("WECOM_BOT_ID", Some("wecom-bot"));
        with_env_var("WECOM_SECRET", Some("wecom-secret"));
        with_env_var("WECOM_WEBSOCKET_URL", Some(&ws_url));
        let result = handle_send(
            &json!({
                "target": "wecom:chat-123",
                "message": format!("MEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("wecom"));
        assert_eq!(parsed["chat_id"], json!("chat-123"));
        join.join().unwrap();
    }

    #[test]
    fn sends_weixin_text_with_context_token_retry() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let account_dir = temp.path().join("weixin").join("accounts");
        fs::create_dir_all(&account_dir).unwrap();
        fs::write(
            account_dir.join("wx-account.context-tokens.json"),
            json!({ "wxid_target": "ctx-123" }).to_string(),
        )
        .unwrap();

        let (base_url, join) = mock_raw_http_server(2, move |index, headers, body| {
            assert!(headers.starts_with("POST /ilink/bot/sendmessage "));
            let normalized_headers = headers.to_ascii_lowercase();
            assert!(normalized_headers.contains("authorizationtype: ilink_bot_token"));
            assert!(normalized_headers.contains("authorization: bearer wx-token"));
            let payload: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(payload["base_info"]["channel_version"], json!("2.2.0"));
            assert_eq!(payload["msg"]["to_user_id"], json!("wxid_target"));
            assert_eq!(payload["msg"]["item_list"][0]["type"], json!(1));
            assert_eq!(
                payload["msg"]["item_list"][0]["text_item"]["text"],
                json!("hello weixin")
            );
            match index {
                0 => {
                    assert_eq!(payload["msg"]["context_token"], json!("ctx-123"));
                    (
                        200,
                        vec![("Content-Type".to_string(), "application/json".to_string())],
                        json!({ "ret": -14, "errcode": -14, "errmsg": "session expired" })
                            .to_string()
                            .into_bytes(),
                    )
                }
                1 => {
                    assert!(payload["msg"].get("context_token").is_none());
                    (
                        200,
                        vec![("Content-Type".to_string(), "application/json".to_string())],
                        json!({ "ret": 0, "errcode": 0 }).to_string().into_bytes(),
                    )
                }
                _ => unreachable!(),
            }
        });

        with_env_var("WEIXIN_TOKEN", Some("wx-token"));
        with_env_var("WEIXIN_ACCOUNT_ID", Some("wx-account"));
        with_env_var("WEIXIN_BASE_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "weixin:wxid_target",
                "message": "hello weixin",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("weixin"));
        assert_eq!(parsed["chat_id"], json!("wxid_target"));
        join.join().unwrap();
    }

    #[test]
    fn sends_weixin_media_via_ilink_and_cdn_upload() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("proof.png");
        let media_bytes = b"\x89PNG\r\n\x1a\nweixin-media";
        fs::write(&media_path, media_bytes).unwrap();

        let (upload_base_url, upload_join) =
            mock_raw_http_server(1, move |_index, headers, body| {
                assert!(headers.starts_with("POST /upload?upload=1 "));
                assert!(!body.is_empty());
                (
                    200,
                    vec![("x-encrypted-param".to_string(), "enc-param-1".to_string())],
                    Vec::new(),
                )
            });

        let upload_url = format!("{upload_base_url}/upload?upload=1");
        let (base_url, join) = mock_raw_http_server(2, move |index, headers, body| {
            let payload: Value = serde_json::from_slice(&body).unwrap();
            match index {
                0 => {
                    assert!(headers.starts_with("POST /ilink/bot/getuploadurl "));
                    assert_eq!(payload["to_user_id"], json!("wxid_target"));
                    assert_eq!(payload["media_type"], json!(1));
                    assert_eq!(payload["rawsize"], json!(media_bytes.len()));
                    (
                        200,
                        vec![("Content-Type".to_string(), "application/json".to_string())],
                        json!({
                            "ret": 0,
                            "errcode": 0,
                            "upload_full_url": upload_url,
                        })
                        .to_string()
                        .into_bytes(),
                    )
                }
                1 => {
                    assert!(headers.starts_with("POST /ilink/bot/sendmessage "));
                    assert_eq!(payload["msg"]["to_user_id"], json!("wxid_target"));
                    assert_eq!(payload["msg"]["item_list"][0]["type"], json!(2));
                    assert_eq!(
                        payload["msg"]["item_list"][0]["image_item"]["media"]["encrypt_query_param"],
                        json!("enc-param-1")
                    );
                    assert!(
                        payload["msg"]["item_list"][0]["image_item"]["media"]["aes_key"]
                            .as_str()
                            .unwrap()
                            .len()
                            > 10
                    );
                    (
                        200,
                        vec![("Content-Type".to_string(), "application/json".to_string())],
                        json!({ "ret": 0, "errcode": 0 }).to_string().into_bytes(),
                    )
                }
                _ => unreachable!(),
            }
        });

        with_env_var("WEIXIN_TOKEN", Some("wx-token"));
        with_env_var("WEIXIN_ACCOUNT_ID", Some("wx-account"));
        with_env_var("WEIXIN_BASE_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "weixin:wxid_target",
                "message": format!("MEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("weixin"));
        assert_eq!(parsed["chat_id"], json!("wxid_target"));
        join.join().unwrap();
        upload_join.join().unwrap();
    }

    #[test]
    fn sends_email_text_via_smtp() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (smtp_addr, join) = mock_smtp_server(move |commands, data| {
            assert!(commands.iter().any(|line| line.starts_with("EHLO ")));
            assert!(commands.iter().any(|line| line.starts_with("AUTH PLAIN ")));
            assert!(
                commands
                    .iter()
                    .any(|line| line == "MAIL FROM:<bot@example.com>")
            );
            assert!(
                commands
                    .iter()
                    .any(|line| line == "RCPT TO:<user@example.com>")
            );
            assert!(data.contains("Subject: Hermes Agent"));
            assert!(data.contains("From: bot@example.com"));
            assert!(data.contains("To: user@example.com"));
            assert!(data.contains("hello over smtp"));
        });
        let (smtp_host, smtp_port) = smtp_addr.split_once(':').unwrap();
        with_env_var("EMAIL_ADDRESS", Some("bot@example.com"));
        with_env_var("EMAIL_PASSWORD", Some("app-pass"));
        with_env_var("EMAIL_SMTP_HOST", Some(smtp_host));
        with_env_var("EMAIL_SMTP_PORT", Some(smtp_port));
        with_env_var("EMAIL_SMTP_SECURITY", Some("none"));
        let result = handle_send(
            &json!({
                "target": "email:user@example.com",
                "message": "hello over smtp",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("email"));
        assert_eq!(parsed["chat_id"], json!("user@example.com"));
        assert!(
            parsed["message_id"]
                .as_str()
                .unwrap_or_default()
                .contains("queued as test-123")
        );
        join.join().unwrap();
    }

    #[test]
    fn sends_email_with_attachment_via_smtp() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let attachment_path = temp.path().join("report.bin");
        let attachment_bytes = b"\x89PNG\r\n\x1a\nattachment-body";
        fs::write(&attachment_path, attachment_bytes).unwrap();
        let expected_base64 = BASE64_STANDARD.encode(attachment_bytes);
        let (smtp_addr, join) = mock_smtp_server(move |commands, data| {
            assert!(commands.iter().any(|line| line.starts_with("AUTH PLAIN ")));
            assert!(data.contains("Subject: Hermes Agent"));
            assert!(data.contains("multipart/mixed"));
            assert!(data.contains("message with file"));
            assert!(data.contains("filename=\"report.bin\""));
            assert!(data.contains("Content-Type: image/png"));
            assert!(data.contains(&expected_base64));
        });
        let (smtp_host, smtp_port) = smtp_addr.split_once(':').unwrap();
        with_env_var("EMAIL_ADDRESS", Some("bot@example.com"));
        with_env_var("EMAIL_PASSWORD", Some("app-pass"));
        with_env_var("EMAIL_SMTP_HOST", Some(smtp_host));
        with_env_var("EMAIL_SMTP_PORT", Some(smtp_port));
        with_env_var("EMAIL_SMTP_SECURITY", Some("none"));
        let result = handle_send(
            &json!({
                "target": "email:user@example.com",
                "message": format!("message with file\nMEDIA:{}", attachment_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("email"));
        join.join().unwrap();
    }

    #[test]
    fn sends_sms_text_via_twilio() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (base_url, join) = mock_server(1, move |_index, headers, body| {
            assert!(headers.starts_with("POST /AC123/Messages.json "));
            assert!(
                headers
                    .to_ascii_lowercase()
                    .contains("authorization: basic ")
            );
            let params = url::form_urlencoded::parse(body.as_bytes())
                .into_owned()
                .collect::<std::collections::HashMap<_, _>>();
            assert_eq!(params.get("From"), Some(&"+15550000000".to_string()));
            assert_eq!(params.get("To"), Some(&"+15551234567".to_string()));
            assert_eq!(params.get("Body"), Some(&"sms hello".to_string()));
            (200, json!({ "sid": "SM123" }).to_string())
        });

        with_env_var("TWILIO_ACCOUNT_SID", Some("AC123"));
        with_env_var("TWILIO_AUTH_TOKEN", Some("twilio-secret"));
        with_env_var("TWILIO_PHONE_NUMBER", Some("+15550000000"));
        with_env_var("TWILIO_API_BASE_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "sms:+15551234567",
                "message": "sms hello",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("sms"));
        assert_eq!(parsed["chat_id"], json!("+15551234567"));
        assert_eq!(parsed["message_id"], json!("SM123"));
        join.join().unwrap();
    }

    #[test]
    fn sends_homeassistant_notification() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (base_url, join) = mock_server(1, move |_index, headers, body| {
            assert!(headers.starts_with("POST /api/services/persistent_notification/create "));
            let payload: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(payload["title"], json!("Hermes Agent"));
            assert_eq!(payload["message"], json!("lights are on"));
            assert_eq!(payload["notification_id"], json!("kitchen-alert"));
            (200, "[]".to_string())
        });

        with_env_var("HASS_TOKEN", Some("ha-token"));
        with_env_var("HASS_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "homeassistant:kitchen-alert",
                "message": "lights are on",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("homeassistant"));
        assert_eq!(parsed["chat_id"], json!("kitchen-alert"));
        assert!(parsed["message_id"].as_str().unwrap().starts_with("ha-"));
        join.join().unwrap();
    }

    #[test]
    fn sends_whatsapp_text_and_media_via_bridge() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("voice.ogg");
        fs::write(&media_path, b"ogg-bytes").unwrap();
        let expected_media_path = media_path.display().to_string();
        let (base_url, join) = mock_server(2, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /send "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["chatId"], json!("15551234567@s.whatsapp.net"));
                assert_eq!(payload["message"], json!("hello whatsapp"));
                (
                    200,
                    json!({ "success": true, "messageId": "wa-msg-1" }).to_string(),
                )
            }
            1 => {
                assert!(headers.starts_with("POST /send-media "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["chatId"], json!("15551234567@s.whatsapp.net"));
                assert_eq!(payload["filePath"], json!(expected_media_path));
                assert_eq!(payload["fileName"], json!("voice.ogg"));
                assert_eq!(payload["mediaType"], json!("audio"));
                (
                    200,
                    json!({ "success": true, "messageId": "wa-msg-2" }).to_string(),
                )
            }
            _ => unreachable!(),
        });

        with_env_var("WHATSAPP_ENABLED", Some("true"));
        with_env_var("WHATSAPP_BRIDGE_URL", Some(&base_url));
        let result = handle_send(
            &json!({
                "target": "whatsapp:+15551234567",
                "message": format!("hello whatsapp\nMEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("whatsapp"));
        assert_eq!(parsed["chat_id"], json!("15551234567@s.whatsapp.net"));
        assert_eq!(parsed["message_id"], json!("wa-msg-2"));
        join.join().unwrap();
    }

    #[test]
    fn sends_dingtalk_text_via_webhook() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (base_url, join) = mock_server(1, move |_index, headers, body| {
            assert!(headers.starts_with("POST /robot/send?access_token=test-token "));
            let payload: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(payload["msgtype"], json!("text"));
            assert_eq!(payload["text"]["content"], json!("hello dingtalk"));
            (200, json!({ "errcode": 0, "errmsg": "ok" }).to_string())
        });

        let webhook_url = format!("{base_url}/robot/send?access_token=test-token");
        with_env_var("DINGTALK_WEBHOOK_URL", Some(&webhook_url));
        let result = handle_send(
            &json!({
                "target": "dingtalk:cidding==",
                "message": "hello dingtalk",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("dingtalk"));
        assert_eq!(parsed["chat_id"], json!("cidding=="));
        assert_eq!(parsed["message_id"], Value::Null);
        join.join().unwrap();
    }

    #[test]
    fn sends_qqbot_text_with_auto_target_fallback() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (base_url, join) = mock_server(3, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /app/getAppAccessToken "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["appId"], json!("qq-app"));
                assert_eq!(payload["clientSecret"], json!("qq-secret"));
                (200, json!({ "access_token": "qq-token" }).to_string())
            }
            1 => {
                assert!(headers.starts_with("POST /channels/user-open/messages "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["content"], json!("hello qq"));
                assert_eq!(payload["msg_type"], json!(0));
                (404, json!({ "message": "channel not found" }).to_string())
            }
            2 => {
                assert!(headers.starts_with("POST /v2/users/user-open/messages "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["content"], json!("hello qq"));
                assert_eq!(payload["msg_type"], json!(0));
                (200, json!({ "id": "qq-msg-1" }).to_string())
            }
            _ => unreachable!(),
        });

        with_env_var("QQ_APP_ID", Some("qq-app"));
        with_env_var("QQ_CLIENT_SECRET", Some("qq-secret"));
        with_env_var("QQBOT_API_BASE_URL", Some(&base_url));
        with_env_var(
            "QQBOT_TOKEN_URL",
            Some(&format!("{base_url}/app/getAppAccessToken")),
        );
        let result = handle_send(
            &json!({
                "target": "qqbot:user-open",
                "message": "hello qq",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("qqbot"));
        assert_eq!(parsed["chat_id"], json!("user-open"));
        assert_eq!(parsed["message_id"], json!("qq-msg-1"));
        join.join().unwrap();
    }

    #[test]
    fn sends_qqbot_group_media() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("proof.png");
        fs::write(&media_path, b"\x89PNG\r\n\x1a\nqqbot").unwrap();
        let (base_url, join) = mock_server(3, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /app/getAppAccessToken "));
                (200, json!({ "access_token": "qq-token" }).to_string())
            }
            1 => {
                assert!(headers.starts_with("POST /v2/groups/group-open/files "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["file_type"], json!(1));
                assert_eq!(payload["srv_send_msg"], json!(false));
                assert!(
                    payload["file_data"]
                        .as_str()
                        .unwrap()
                        .starts_with("iVBORw0K")
                );
                (200, json!({ "file_info": "file-token-1" }).to_string())
            }
            2 => {
                assert!(headers.starts_with("POST /v2/groups/group-open/messages "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["msg_type"], json!(7));
                assert_eq!(payload["media"]["file_info"], json!("file-token-1"));
                assert_eq!(payload["content"], json!("see attachment"));
                (200, json!({ "id": "qq-media-1" }).to_string())
            }
            _ => unreachable!(),
        });

        with_env_var("QQ_APP_ID", Some("qq-app"));
        with_env_var("QQ_CLIENT_SECRET", Some("qq-secret"));
        with_env_var("QQBOT_API_BASE_URL", Some(&base_url));
        with_env_var(
            "QQBOT_TOKEN_URL",
            Some(&format!("{base_url}/app/getAppAccessToken")),
        );
        let result = handle_send(
            &json!({
                "target": "qqbot:group:group-open",
                "message": format!("see attachment\nMEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("qqbot"));
        assert_eq!(parsed["chat_id"], json!("group:group-open"));
        assert_eq!(parsed["message_id"], json!("qq-media-1"));
        join.join().unwrap();
    }

    #[test]
    fn sends_bluebubbles_text_and_media_for_existing_chat() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("voice.m4a");
        fs::write(&media_path, b"m4a-bytes").unwrap();
        let (base_url, join) = mock_server(3, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /api/v1/chat/query?password=bb-secret "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["limit"], json!(100));
                (
                    200,
                    json!({
                        "data": [
                            {
                                "guid": "iMessage;-;+15551234567",
                                "chatIdentifier": "+15551234567",
                                "participants": [{"address": "+15551234567"}]
                            }
                        ]
                    })
                    .to_string(),
                )
            }
            1 => {
                assert!(headers.starts_with("POST /api/v1/message/text?password=bb-secret "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["chatGuid"], json!("iMessage;-;+15551234567"));
                assert_eq!(payload["message"], json!("hello bubble"));
                (200, json!({ "data": { "guid": "msg-1" } }).to_string())
            }
            2 => {
                assert!(headers.starts_with("POST /api/v1/message/attachment?password=bb-secret "));
                assert!(
                    headers
                        .to_ascii_lowercase()
                        .contains("content-type: multipart/form-data;")
                );
                (200, json!({ "data": { "guid": "msg-2" } }).to_string())
            }
            _ => unreachable!(),
        });

        with_env_var("BLUEBUBBLES_SERVER_URL", Some(&base_url));
        with_env_var("BLUEBUBBLES_PASSWORD", Some("bb-secret"));
        let result = handle_send(
            &json!({
                "target": "bluebubbles:+15551234567",
                "message": format!("hello bubble\nMEDIA:{}", media_path.display()),
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("bluebubbles"));
        assert_eq!(parsed["chat_id"], json!("+15551234567"));
        assert_eq!(parsed["message_id"], json!("msg-2"));
        join.join().unwrap();
    }

    #[test]
    fn sends_bluebubbles_first_message_by_creating_chat() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let (base_url, join) = mock_server(2, move |index, headers, body| match index {
            0 => {
                assert!(headers.starts_with("POST /api/v1/chat/query?password=bb-secret "));
                (
                    200,
                    json!({
                        "data": []
                    })
                    .to_string(),
                )
            }
            1 => {
                assert!(headers.starts_with("POST /api/v1/chat/new?password=bb-secret "));
                let payload: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(payload["addresses"], json!(["user@example.com"]));
                assert_eq!(payload["message"], json!("hello first thread"));
                (200, json!({ "data": { "guid": "new-msg-1" } }).to_string())
            }
            _ => unreachable!(),
        });

        with_env_var("BLUEBUBBLES_SERVER_URL", Some(&base_url));
        with_env_var("BLUEBUBBLES_PASSWORD", Some("bb-secret"));
        let result = handle_send(
            &json!({
                "target": "bluebubbles:user@example.com",
                "message": "hello first thread",
            }),
            &runtime,
        );
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["success"], json!(true));
        assert_eq!(parsed["platform"], json!("bluebubbles"));
        assert_eq!(parsed["chat_id"], json!("user@example.com"));
        assert_eq!(parsed["message_id"], json!("new-msg-1"));
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
