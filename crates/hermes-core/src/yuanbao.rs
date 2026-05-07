use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use chrono::{FixedOffset, Utc};
use hmac::{Hmac, Mac};
use md5::compute as md5_compute;
use reqwest::blocking::Client;
use reqwest::header::CONTENT_TYPE;
use serde::Serialize;
use serde_json::{Value, json};
use sha1::{Digest as Sha1Digest, Sha1};
use sha2::Sha256;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket, connect};
use unicode_normalization::UnicodeNormalization;
use url::Url;
use url::form_urlencoded;

use crate::tools::{ToolRuntime, tool_error, tool_result};

const DEFAULT_LIMIT: usize = 10;
const MAX_LIMIT: usize = 50;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
struct Sticker {
    sticker_id: &'static str,
    package_id: &'static str,
    name: &'static str,
    description: &'static str,
    width: u16,
    height: u16,
    formats: &'static str,
}

const STICKERS: &[Sticker] = &[
    Sticker {
        sticker_id: "278",
        package_id: "1003",
        name: "六六六",
        description: "666 厉害 牛 棒 绝了 好强 awesome",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "262",
        package_id: "1003",
        name: "我想开了",
        description: "想开 佛系 释怀 顿悟 看淡了 无所谓",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "130",
        package_id: "1003",
        name: "害羞",
        description: "腼腆 不好意思 脸红 娇羞 羞涩 捂脸",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "252",
        package_id: "1003",
        name: "比心",
        description: "笔芯 爱你 爱心手势 love heart 喜欢你",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "125",
        package_id: "1003",
        name: "委屈",
        description: "难过 想哭 可怜巴巴 瘪嘴 受伤 被欺负",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "146",
        package_id: "1003",
        name: "亲亲",
        description: "么么 mua 亲一下 kiss 飞吻 啵",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "131",
        package_id: "1003",
        name: "酷",
        description: "帅 墨镜 cool 高冷 有型 swagger",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "145",
        package_id: "1003",
        name: "睡",
        description: "睡觉 困 zzZ 打盹 躺平 休眠 sleepy",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "152",
        package_id: "1003",
        name: "发呆",
        description: "懵 愣住 放空 呆滞 出神 脑子空白",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "157",
        package_id: "1003",
        name: "可怜",
        description: "卖萌 求饶 委屈巴巴 弱小 拜托 眼巴巴",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "200",
        package_id: "1003",
        name: "摊手",
        description: "无奈 没办法 耸肩 随便 那咋整 whatever",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "213",
        package_id: "1003",
        name: "头大",
        description: "头疼 烦恼 郁闷 难搞 崩溃 一团乱",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "256",
        package_id: "1003",
        name: "吓",
        description: "害怕 惊恐 震惊 吓一跳 恐怖 怂",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "203",
        package_id: "1003",
        name: "吐血",
        description: "无语 崩溃 被雷 内伤 一口老血 屮",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "185",
        package_id: "1003",
        name: "哼",
        description: "傲娇 生气 不满 撇嘴 不理 赌气",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "220",
        package_id: "1003",
        name: "嘿嘿",
        description: "坏笑 猥琐笑 偷笑 憨笑 得意 你懂的",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "218",
        package_id: "1003",
        name: "头秃",
        description: "程序员 加班 焦虑 没头发 秃了 肝爆",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "221",
        package_id: "1003",
        name: "暗中观察",
        description: "窥屏 潜水 偷偷看 角落 围观 屏住呼吸",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "224",
        package_id: "1003",
        name: "我酸了",
        description: "嫉妒 柠檬精 羡慕 吃柠檬 眼红 恰柠檬",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "246",
        package_id: "1003",
        name: "打call",
        description: "应援 加油 支持 喝彩 助威 call",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "251",
        package_id: "1003",
        name: "庆祝",
        description: "祝贺 开心 耶 party 胜利 干杯",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "151",
        package_id: "1003",
        name: "奋斗",
        description: "努力 加油 拼搏 冲 干劲 卷起来",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "143",
        package_id: "1003",
        name: "惊讶",
        description: "震惊 哇 不敢相信 OMG 居然 这么离谱",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "144",
        package_id: "1003",
        name: "疑问",
        description: "问号 不懂 啥 为什么 啥情况 懵逼问",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "248",
        package_id: "1003",
        name: "仔细分析",
        description: "思考 推敲 认真 研究 琢磨 让我想想",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "184",
        package_id: "1003",
        name: "撅嘴",
        description: "嘟嘴 卖萌 不高兴 撒娇 嘴翘",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "199",
        package_id: "1003",
        name: "泪奔",
        description: "大哭 伤心 破防 感动哭 泪流满面 呜呜",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "276",
        package_id: "1003",
        name: "尊嘟假嘟",
        description: "真的假的 真假 可爱问 你骗我 是不是",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "113",
        package_id: "1003",
        name: "略略略",
        description: "调皮 吐舌 不服 略 气死你 鬼脸",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "180",
        package_id: "1003",
        name: "困",
        description: "想睡 倦 打哈欠 睁不开眼 好困啊 sleepy",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "181",
        package_id: "1003",
        name: "折磨",
        description: "难受 痛苦 煎熬 蚌埠住了 受不了 要命",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "182",
        package_id: "1003",
        name: "抠鼻",
        description: "不屑 无聊 淡定 无所谓 鄙视 挖鼻",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "183",
        package_id: "1003",
        name: "鼓掌",
        description: "拍手 叫好 赞同 666 喝彩 掌声",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "204",
        package_id: "1003",
        name: "斜眼笑",
        description: "滑稽 坏笑 doge 意味深长 阴阳怪气 嘿嘿嘿",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "216",
        package_id: "1003",
        name: "辣眼睛",
        description: "看不下去 cringe 毁三观 太丑了 瞎了",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "217",
        package_id: "1003",
        name: "哦哟",
        description: "惊讶 起哄 哇哦 有戏 不简单 哟",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "222",
        package_id: "1003",
        name: "吃瓜",
        description: "围观 看戏 八卦 路人 看热闹 板凳",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "225",
        package_id: "1003",
        name: "狗头",
        description: "doge 保命 开玩笑 滑稽 反讽 懂的都懂",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "227",
        package_id: "1003",
        name: "敬礼",
        description: "salute 尊重 收到 遵命 致敬 报告",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "231",
        package_id: "1003",
        name: "哦",
        description: "知道了 明白 敷衍 嗯 这样啊 收到",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "236",
        package_id: "1003",
        name: "拿到红包",
        description: "红包 谢谢老板 发财 开心 抢到了 欧气",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "239",
        package_id: "1003",
        name: "牛吖",
        description: "牛 厉害 强 666 佩服 大佬",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "272",
        package_id: "1003",
        name: "贴贴",
        description: "抱抱 亲昵 蹭蹭 亲密 靠靠 撒娇贴",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "138",
        package_id: "1003",
        name: "爱心",
        description: "心 love 喜欢你 红心 示爱 么么哒",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "170",
        package_id: "1003",
        name: "晚安",
        description: "好梦 睡了 night 早点休息 安啦 moon",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "176",
        package_id: "1003",
        name: "太阳",
        description: "晴天 早上好 阳光 morning 好天气 日",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "266",
        package_id: "1003",
        name: "柠檬",
        description: "酸 嫉妒 柠檬精 羡慕 我酸 恰柠檬",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "267",
        package_id: "1003",
        name: "大冤种",
        description: "倒霉 吃亏 自嘲 好心没好报 背锅 工具人",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "132",
        package_id: "1003",
        name: "吐了",
        description: "恶心 yue 受不了 嫌弃 想吐 生理不适",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "134",
        package_id: "1003",
        name: "怒",
        description: "生气 愤怒 火大 暴躁 气炸 怼",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "165",
        package_id: "1003",
        name: "玫瑰",
        description: "花 示爱 表白 浪漫 送你花 情人节",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "119",
        package_id: "1003",
        name: "凋谢",
        description: "花谢 失恋 难过 枯萎 心碎 凉了",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "159",
        package_id: "1003",
        name: "点赞",
        description: "赞 认同 好棒 good like 大拇指 顶",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "164",
        package_id: "1003",
        name: "握手",
        description: "合作 你好 商务 hello deal 成交 友好",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "163",
        package_id: "1003",
        name: "抱拳",
        description: "谢谢 失敬 江湖 承让 拜托 有礼",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "169",
        package_id: "1003",
        name: "ok",
        description: "好的 收到 没问题 okay 行 可以 懂了",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "174",
        package_id: "1003",
        name: "拳头",
        description: "加油 干 冲 fight 力量 击拳 硬气",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "191",
        package_id: "1003",
        name: "鞭炮",
        description: "过年 喜庆 爆竹 春节 噼里啪啦 红",
        width: 128,
        height: 128,
        formats: "png",
    },
    Sticker {
        sticker_id: "258",
        package_id: "1003",
        name: "烟花",
        description: "庆典 漂亮 新年 嘭 绽放 节日快乐",
        width: 128,
        height: 128,
        formats: "png",
    },
];

const MENTION_HINT: &str = "To @mention a user, you MUST use the format: space + @ + nickname + space (e.g. \" @Alice \").";
const DEFAULT_WS_GATEWAY_URL: &str = "wss://bot-wss.yuanbao.tencent.com/wss/connection";
const DEFAULT_API_DOMAIN: &str = "https://bot.yuanbao.tencent.com";
const YUANBAO_TOOL_TIMEOUT_SECS: u64 = 15;
const YUANBAO_MAX_MEDIA_BYTES: usize = 50 * 1024 * 1024;
const YUANBAO_UPLOAD_INFO_PATH: &str = "/api/resource/genUploadInfo";
const YUANBAO_INSTANCE_ID: &str = "17";
const BIZ_PACKAGE: &str = "yuanbao_openclaw_proxy";
const MODULE_CONN_ACCESS: &str = "conn_access";
const CMD_AUTH_BIND: &str = "auth-bind";
const CMD_QUERY_GROUP_INFO: &str = "query_group_info";
const CMD_GET_GROUP_MEMBER_LIST: &str = "get_group_member_list";
const CMD_SEND_C2C_MESSAGE: &str = "send_c2c_message";
const CMD_SEND_GROUP_MESSAGE: &str = "send_group_message";
const CMD_TYPE_REQUEST: u64 = 0;
const CMD_TYPE_RESPONSE: u64 = 1;
const WT_VARINT: u8 = 0;
const WT_LEN: u8 = 2;

static YUANBAO_SEQ: AtomicU32 = AtomicU32::new(0);

type HmacSha256 = Hmac<Sha256>;
type HmacSha1 = Hmac<Sha1>;

#[derive(Debug, Clone)]
struct YuanbaoConfig {
    app_key: String,
    app_secret: String,
    api_domain: String,
    ws_url: String,
    route_env: String,
    app_version: String,
    operation_system: String,
    bot_version: String,
}

#[derive(Debug, Clone)]
struct YuanbaoToken {
    token: String,
    bot_id: String,
}

#[derive(Debug, Clone)]
struct YuanbaoGroupInfo {
    group_name: String,
    owner_id: String,
    owner_nickname: String,
    member_count: u64,
}

#[derive(Debug, Clone)]
struct YuanbaoMember {
    user_id: String,
    nickname: String,
    role: u64,
}

#[derive(Debug, Clone)]
struct ConnHead {
    cmd_type: u64,
    cmd: String,
    msg_id: String,
    status: u64,
}

#[derive(Debug, Clone)]
struct ConnMsg {
    head: ConnHead,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
enum ProtoValue {
    Varint(u64),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone)]
struct ProtoField {
    number: u32,
    value: ProtoValue,
}

#[derive(Debug, Clone)]
struct MsgBodyElement {
    msg_type: String,
    msg_content: MsgContent,
}

#[derive(Debug, Clone)]
struct ImageInfo {
    kind: u64,
    size: u64,
    width: u64,
    height: u64,
    url: String,
}

#[derive(Debug, Clone, Default)]
struct MsgContent {
    text: Option<String>,
    uuid: Option<String>,
    image_format: Option<u64>,
    index: Option<u64>,
    data: Option<String>,
    image_info_array: Vec<ImageInfo>,
    url: Option<String>,
    file_size: Option<u64>,
    file_name: Option<String>,
}

#[derive(Debug, Clone)]
struct CosCredentials {
    bucket_name: String,
    region: String,
    location: String,
    secret_id: String,
    secret_key: String,
    session_token: String,
    start_time: Option<i64>,
    expired_time: Option<i64>,
    resource_url: Option<String>,
}

#[derive(Debug, Clone)]
struct UploadedMedia {
    url: String,
    uuid: String,
    size: u64,
    file_name: String,
    mime_type: String,
    width: Option<u64>,
    height: Option<u64>,
}

struct YuanbaoClient {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    bot_id: String,
    sign_token: String,
    config: YuanbaoConfig,
}

pub fn yuanbao_available() -> bool {
    load_yuanbao_config().is_ok()
}

pub fn send_yuanbao_message(chat_id: &str, message: &str) -> Result<String, String> {
    let target = chat_id.trim();
    if target.is_empty() {
        return Err("chat_id is required".to_string());
    }
    let text = message.trim();
    if text.is_empty() {
        return Err("message is required".to_string());
    }
    with_yuanbao_client(|client| client.send_chat_message(target, text))
}

pub fn send_yuanbao_message_with_media(
    chat_id: &str,
    message: &str,
    media_paths: &[PathBuf],
    group_code: Option<&str>,
) -> Result<String, String> {
    let target = chat_id.trim();
    if target.is_empty() {
        return Err("chat_id is required".to_string());
    }
    if message.trim().is_empty() && media_paths.is_empty() {
        return Err("message or media_files is required".to_string());
    }
    let group_context = group_code.map(str::trim).filter(|value| !value.is_empty());
    with_yuanbao_client(|client| {
        client.send_chat_with_media(target, message.trim(), media_paths, group_context)
    })
}

pub fn yb_query_group_info_schema() -> Value {
    json!({
        "name": "yb_query_group_info",
        "description": "Query basic info about a Yuanbao group (Pai), including group name, owner, and member count.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_code": {
                    "type": "string",
                    "description": "The unique group identifier."
                }
            },
            "required": ["group_code"]
        }
    })
}

pub fn yb_query_group_members_schema() -> Value {
    json!({
        "name": "yb_query_group_members",
        "description": "Query Yuanbao group members. Use this before @mentioning anyone, or when you need to find a user, list bots, or inspect the current group roster.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_code": {
                    "type": "string",
                    "description": "The unique group identifier."
                },
                "action": {
                    "type": "string",
                    "enum": ["find", "list_bots", "list_all"],
                    "description": "find searches by nickname, list_bots returns Yuanbao AI and bot entries, list_all returns the full member list."
                },
                "name": {
                    "type": "string",
                    "description": "Optional nickname fragment for action='find'."
                },
                "mention": {
                    "type": "boolean",
                    "description": "Set true when you need an exact @mention hint in the response."
                }
            },
            "required": ["group_code", "action"]
        }
    })
}

pub fn yb_send_dm_schema() -> Value {
    json!({
        "name": "yb_send_dm",
        "description": "Send a private/direct message to a Yuanbao group member. Provide user_id directly if known, otherwise the tool resolves it from the group member list by nickname.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_code": {
                    "type": "string",
                    "description": "The source group code. Required when user_id is not provided. Defaults to the current session group when available."
                },
                "name": {
                    "type": "string",
                    "description": "Target nickname fragment. Required when user_id is not provided."
                },
                "message": {
                    "type": "string",
                    "description": "Text to send. MEDIA:/path tags are supported for local files."
                },
                "user_id": {
                    "type": "string",
                    "description": "Direct Yuanbao account id. If provided, skips the member lookup."
                },
                "media_files": {
                    "type": "array",
                    "description": "Optional local files to send after the text message. Each item must provide a local path.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string"
                            },
                            "is_voice": {
                                "type": "boolean"
                            }
                        },
                        "required": ["path"]
                    }
                }
            }
        }
    })
}

pub fn yb_send_sticker_schema() -> Value {
    json!({
        "name": "yb_send_sticker",
        "description": "Send a built-in Yuanbao sticker (TIMFaceElem) to the current or specified chat. Call yb_search_sticker first if you do not know the right sticker name or sticker_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "sticker": {
                    "type": "string",
                    "description": "Sticker name or numeric sticker_id. Empty sends a random built-in sticker."
                },
                "chat_id": {
                    "type": "string",
                    "description": "Target chat id. Defaults to HERMES_SESSION_CHAT_ID. Format: group:<group_code>, direct:<account_id>, or bare account id."
                },
                "reply_to": {
                    "type": "string",
                    "description": "Optional ref_msg_id for group reply threads."
                }
            }
        }
    })
}

pub fn handle_yb_query_group_info(args: &Value, _runtime: &ToolRuntime) -> String {
    let group_code = match required_non_empty_string(args, "group_code") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let result = with_yuanbao_client(|client| client.query_group_info(&group_code));
    match result {
        Ok(info) => tool_result(json!({
            "success": true,
            "group_code": group_code,
            "group_name": info.group_name,
            "member_count": info.member_count,
            "owner": {
                "user_id": info.owner_id,
                "nickname": info.owner_nickname,
            },
            "note": "The group is called \"派 (Pai)\" in the app.",
        })),
        Err(error) => tool_error(error),
    }
}

pub fn handle_yb_query_group_members(args: &Value, _runtime: &ToolRuntime) -> String {
    let group_code = match required_non_empty_string(args, "group_code") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let action = match required_non_empty_string(args, "action") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if !matches!(action.as_str(), "find" | "list_bots" | "list_all") {
        return tool_error("action must be one of: find, list_bots, list_all");
    }
    let name = match optional_string(args, "name") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let mention = match optional_bool(args, "mention") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let result = with_yuanbao_client(|client| client.get_group_member_list(&group_code));
    let raw_members = match result {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if raw_members.is_empty() {
        return tool_error("No members found in this group.");
    }
    let all_members = raw_members_to_json(raw_members);
    let mut members = all_members.clone();

    let hint = mention.then(|| json!({ "mention_hint": MENTION_HINT }));
    match action.as_str() {
        "list_bots" => {
            members.retain(|member| {
                matches!(
                    member.get("role").and_then(Value::as_str),
                    Some("yuanbao_ai" | "bot")
                )
            });
            if members.is_empty() {
                return tool_error("No bots found in this group.");
            }
            let mut payload = json!({
                "success": true,
                "msg": format!("Found {} bot(s).", members.len()),
                "members": members,
            });
            if let Some(hint) = hint {
                payload
                    .as_object_mut()
                    .expect("payload object")
                    .extend(hint.as_object().cloned().unwrap_or_default());
            }
            tool_result(payload)
        }
        "find" => {
            if !name.is_empty() {
                let needle = name.to_lowercase();
                let matched = members
                    .into_iter()
                    .filter(|member| {
                        member
                            .get("nickname")
                            .and_then(Value::as_str)
                            .map(|value| value.to_lowercase().contains(&needle))
                            .unwrap_or(false)
                    })
                    .collect::<Vec<_>>();
                if matched.is_empty() {
                    let mut payload = json!({
                        "success": false,
                        "msg": format!("No match for \"{name}\". All members listed below."),
                        "members": all_members,
                    });
                    if let Some(hint) = hint {
                        payload
                            .as_object_mut()
                            .expect("payload object")
                            .extend(hint.as_object().cloned().unwrap_or_default());
                    }
                    return tool_result(payload);
                }
                let mut payload = json!({
                    "success": true,
                    "msg": format!("Found {} member(s) matching \"{name}\".", matched.len()),
                    "members": matched,
                });
                if let Some(hint) = hint {
                    payload
                        .as_object_mut()
                        .expect("payload object")
                        .extend(hint.as_object().cloned().unwrap_or_default());
                }
                return tool_result(payload);
            }
            let mut payload = json!({
                "success": true,
                "msg": format!("Found {} member(s).", members.len()),
                "members": members,
            });
            if let Some(hint) = hint {
                payload
                    .as_object_mut()
                    .expect("payload object")
                    .extend(hint.as_object().cloned().unwrap_or_default());
            }
            tool_result(payload)
        }
        _ => {
            let mut payload = json!({
                "success": true,
                "msg": format!("Found {} member(s).", members.len()),
                "members": members,
            });
            if let Some(hint) = hint {
                payload
                    .as_object_mut()
                    .expect("payload object")
                    .extend(hint.as_object().cloned().unwrap_or_default());
            }
            tool_result(payload)
        }
    }
}

pub fn handle_yb_send_dm(args: &Value, runtime: &ToolRuntime) -> String {
    let user_id = match optional_string(args, "user_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let name = match optional_string(args, "name") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let group_code_arg = match optional_string(args, "group_code") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let message = match optional_string(args, "message") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let (cleaned_message, media_paths) = match extract_message_media(args, runtime, &message) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    if cleaned_message.is_empty() && media_paths.is_empty() {
        return tool_error("message or media_files is required");
    }

    let group_code = if group_code_arg.is_empty() {
        session_group_code().unwrap_or_default()
    } else {
        group_code_arg
    };

    let outcome = with_yuanbao_client(|client| {
        let resolved = if !user_id.is_empty() {
            (user_id.clone(), name.clone())
        } else {
            if group_code.is_empty() {
                return Err("group_code is required when user_id is not provided".to_string());
            }
            if name.is_empty() {
                return Err("name is required when user_id is not provided".to_string());
            }
            let members = client.get_group_member_list(&group_code)?;
            let needle = name.to_lowercase();
            let matched = members
                .into_iter()
                .filter(|member| member.nickname.to_lowercase().contains(&needle))
                .collect::<Vec<_>>();
            if matched.is_empty() {
                return Err(format!(
                    "No member matching \"{name}\" found in group {group_code}."
                ));
            }
            if matched.len() > 1 {
                let candidates = matched
                    .into_iter()
                    .map(|member| {
                        json!({
                            "user_id": member.user_id,
                            "nickname": member.nickname,
                        })
                    })
                    .collect::<Vec<_>>();
                return Ok(json!({
                    "success": false,
                    "error": format!("Multiple members match \"{name}\". Please specify which one."),
                    "candidates": candidates,
                }));
            }
            let member = matched.into_iter().next().expect("single match");
            (member.user_id, member.nickname)
        };

        let message_id = client.send_chat_with_media(
            &resolved.0,
            &cleaned_message,
            &media_paths,
            Some(&group_code),
        )?;
        Ok(json!({
            "success": true,
            "user_id": resolved.0,
            "nickname": resolved.1,
            "message_id": message_id,
            "note": format!("DM sent to \"{}\" successfully.", resolved.1),
        }))
    });

    match outcome {
        Ok(value) => tool_result(value),
        Err(error) => tool_error(error),
    }
}

pub fn handle_yb_send_sticker(args: &Value, _runtime: &ToolRuntime) -> String {
    let sticker = match optional_string(args, "sticker") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let reply_to = match optional_string(args, "reply_to") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let chat_id = match optional_string(args, "chat_id") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let target = if chat_id.is_empty() {
        match env::var("HERMES_SESSION_CHAT_ID") {
            Ok(value) if !value.trim().is_empty() => value.trim().to_string(),
            _ => {
                return tool_error("chat_id is required (no active yuanbao session detected)");
            }
        }
    } else {
        chat_id
    };
    let Some(sticker_obj) = resolve_sticker(&sticker) else {
        return tool_error(format!(
            "Sticker not found: {sticker:?}. Use yb_search_sticker first to discover available stickers."
        ));
    };
    let sent = with_yuanbao_client(|client| client.send_sticker(&target, sticker_obj, &reply_to));
    match sent {
        Ok(message_id) => tool_result(json!({
            "success": true,
            "chat_id": target,
            "sticker": {
                "sticker_id": sticker_obj.sticker_id,
                "name": sticker_obj.name,
            },
            "message_id": message_id,
            "note": "Sticker delivered to the chat. If you have additional text to say, reply now; otherwise end your turn without generating text.",
        })),
        Err(error) => tool_error(error),
    }
}

pub fn yb_search_sticker_schema() -> Value {
    json!({
        "name": "yb_search_sticker",
        "description": "Search Yuanbao's built-in sticker catalogue by name, sticker id, or related keywords. Empty queries return the first items in catalogue order. Use this to pick a sticker candidate before any future Yuanbao send flow.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search text. This can be a sticker name, sticker id, Chinese keyword, or related English synonym."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Must be between 1 and 50. Defaults to 10."
                }
            }
        }
    })
}

pub fn handle_yb_search_sticker(args: &Value, _runtime: &ToolRuntime) -> String {
    let query = match optional_string(args, "query") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let limit = match optional_limit(args, "limit") {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let matches = search_stickers(&query, limit);
    tool_result(json!({
        "success": true,
        "query": query,
        "count": matches.len(),
        "results": matches.into_iter().map(sticker_result).collect::<Vec<_>>(),
    }))
}

fn optional_string(args: &Value, key: &str) -> Result<String, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(value)) => Ok(value.trim().to_string()),
        _ => Err(format!("{key} must be a string")),
    }
}

fn optional_limit(args: &Value, key: &str) -> Result<usize, String> {
    let Some(value) = args.get(key) else {
        return Ok(DEFAULT_LIMIT);
    };
    match value {
        Value::Null => Ok(DEFAULT_LIMIT),
        Value::Number(number) => {
            let Some(raw) = number.as_u64() else {
                return Err(format!(
                    "{key} must be an integer between 1 and {MAX_LIMIT}"
                ));
            };
            let parsed = usize::try_from(raw)
                .map_err(|_| format!("{key} must be an integer between 1 and {MAX_LIMIT}"))?;
            validate_limit(parsed, key)
        }
        Value::String(text) => {
            let parsed = text
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("{key} must be an integer between 1 and {MAX_LIMIT}"))?;
            validate_limit(parsed, key)
        }
        _ => Err(format!(
            "{key} must be an integer between 1 and {MAX_LIMIT}"
        )),
    }
}

fn validate_limit(limit: usize, key: &str) -> Result<usize, String> {
    if (1..=MAX_LIMIT).contains(&limit) {
        Ok(limit)
    } else {
        Err(format!("{key} must be between 1 and {MAX_LIMIT}"))
    }
}

fn sticker_result(sticker: &Sticker) -> Value {
    json!({
        "sticker_id": sticker.sticker_id,
        "name": sticker.name,
        "description": sticker.description,
        "package_id": sticker.package_id,
    })
}

fn search_stickers(query: &str, limit: usize) -> Vec<&'static Sticker> {
    let safe_limit = limit.clamp(1, MAX_LIMIT);
    if query.is_empty() || normalize_text(query).is_empty() {
        return STICKERS.iter().take(safe_limit).collect();
    }

    let query_norm = normalize_text(query);
    let mut scored = STICKERS
        .iter()
        .map(|sticker| {
            let name_score = score_field(sticker.name, query);
            let desc_score = score_field(sticker.description, query) * 0.88;
            let id_score = score_sticker_id(sticker.sticker_id, &query_norm);
            (name_score.max(desc_score).max(id_score), sticker)
        })
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    let top = scored.first().map(|(score, _)| *score).unwrap_or(0.0);
    if top <= 0.0 {
        return scored
            .into_iter()
            .take(safe_limit)
            .map(|(_, sticker)| sticker)
            .collect();
    }

    let floor = if top >= 22.0 {
        18.0
    } else if top >= 12.0 {
        (top * 0.5).max(10.0)
    } else {
        (top * 0.35).max(6.0)
    };

    let filtered = scored
        .iter()
        .filter(|(score, _)| *score >= floor)
        .map(|(_, sticker)| *sticker)
        .collect::<Vec<_>>();
    let output = if filtered.is_empty() {
        scored
            .into_iter()
            .map(|(_, sticker)| sticker)
            .collect::<Vec<_>>()
    } else {
        filtered
    };
    output.into_iter().take(safe_limit).collect()
}

fn score_sticker_id(sticker_id: &str, query_norm: &str) -> f64 {
    if sticker_id.is_empty() || query_norm.is_empty() {
        return 0.0;
    }
    let sticker_norm = normalize_text(sticker_id);
    if sticker_norm == query_norm {
        100.0
    } else if sticker_norm.contains(query_norm) {
        84.0
    } else {
        0.0
    }
}

fn normalize_text(raw: &str) -> String {
    raw.nfkc().collect::<String>().trim().to_lowercase()
}

fn compact_text(raw: &str) -> String {
    normalize_text(raw)
        .chars()
        .filter(|ch| !is_compact_separator(*ch))
        .collect()
}

fn is_compact_separator(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '\u{3000}'
                | '-'
                | '_'
                | '·'
                | '.'
                | ','
                | '，'
                | '。'
                | '!'
                | '！'
                | '?'
                | '？'
                | '"'
                | '“'
                | '”'
                | '\''
                | '‘'
                | '’'
                | '、'
                | '/'
                | '\\'
        )
}

fn multiset_char_hit_ratio(needle: &str, haystack: &str) -> f64 {
    if needle.is_empty() {
        return 0.0;
    }
    let mut bag = HashMap::<char, usize>::new();
    for ch in haystack.chars() {
        *bag.entry(ch).or_insert(0) += 1;
    }
    let mut hits = 0usize;
    for ch in needle.chars() {
        if let Some(count) = bag.get_mut(&ch)
            && *count > 0
        {
            hits += 1;
            *count -= 1;
        }
    }
    hits as f64 / needle.chars().count() as f64
}

fn bigram_jaccard(left: &str, right: &str) -> f64 {
    let left_bigrams = string_bigrams(left);
    let right_bigrams = string_bigrams(right);
    if left_bigrams.is_empty() || right_bigrams.is_empty() {
        return 0.0;
    }
    let intersection = left_bigrams.intersection(&right_bigrams).count();
    let union = left_bigrams.len() + right_bigrams.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn string_bigrams(raw: &str) -> HashSet<String> {
    let chars = raw.chars().collect::<Vec<_>>();
    if chars.len() < 2 {
        return HashSet::new();
    }
    let mut out = HashSet::new();
    for window in chars.windows(2) {
        out.insert(window.iter().collect::<String>());
    }
    out
}

fn longest_subsequence_ratio(needle: &str, haystack: &str) -> f64 {
    let needle_chars = needle.chars().collect::<Vec<_>>();
    if needle_chars.is_empty() {
        return 0.0;
    }
    let mut cursor = 0usize;
    for ch in haystack.chars() {
        if cursor >= needle_chars.len() {
            break;
        }
        if ch == needle_chars[cursor] {
            cursor += 1;
        }
    }
    cursor as f64 / needle_chars.len() as f64
}

fn score_field(haystack: &str, query: &str) -> f64 {
    let hay = normalize_text(haystack);
    let q = normalize_text(query);
    if hay.is_empty() || q.is_empty() {
        return 0.0;
    }
    let hay_compact = compact_text(haystack);
    let query_compact = compact_text(query);
    let mut best: f64 = 0.0;
    if hay == q {
        best = best.max(100.0);
    }
    if hay.contains(&q) {
        best = best.max(92.0 + q.chars().count().min(6) as f64);
    }
    if q.chars().count() >= 2 && hay.starts_with(&q) {
        best = best.max(88.0);
    }
    if !query_compact.is_empty() && hay_compact.contains(&query_compact) {
        best = best.max(86.0);
    }
    best = best.max(multiset_char_hit_ratio(&query_compact, &hay_compact) * 62.0);
    best = best.max(bigram_jaccard(&query_compact, &hay_compact) * 58.0);
    best = best.max(longest_subsequence_ratio(&query_compact, &hay_compact) * 52.0);
    if q.chars().count() == 1 && hay.contains(&q) {
        best = best.max(68.0);
    }
    best
}

fn required_non_empty_string(args: &Value, key: &str) -> Result<String, String> {
    let value = optional_string(args, key)?;
    if value.is_empty() {
        Err(format!("{key} is required"))
    } else {
        Ok(value)
    }
}

fn optional_bool(args: &Value, key: &str) -> Result<bool, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        _ => Err(format!("{key} must be a boolean")),
    }
}

fn resolve_sticker(raw: &str) -> Option<&'static Sticker> {
    if raw.trim().is_empty() {
        return Some(random_sticker());
    }
    let trimmed = raw.trim();
    if trimmed.chars().all(|ch| ch.is_ascii_digit())
        && let Some(found) = STICKERS
            .iter()
            .find(|sticker| sticker.sticker_id == trimmed)
    {
        return Some(found);
    }
    let normalized = normalize_text(trimmed);
    if let Some(found) = STICKERS
        .iter()
        .find(|sticker| normalize_text(sticker.name) == normalized)
    {
        return Some(found);
    }
    if let Some(found) = STICKERS
        .iter()
        .find(|sticker| normalize_text(sticker.description).contains(&normalized))
    {
        return Some(found);
    }
    search_stickers(trimmed, 1).into_iter().next()
}

fn random_sticker() -> &'static Sticker {
    let index = (unix_ts_nanos() as usize) % STICKERS.len();
    &STICKERS[index]
}

fn extract_message_media(
    args: &Value,
    runtime: &ToolRuntime,
    message: &str,
) -> Result<(String, Vec<PathBuf>), String> {
    let mut media_paths = parse_media_files_arg(args, runtime)?;
    let mut cleaned_lines = Vec::new();
    for line in message.lines() {
        let trimmed = line.trim();
        if trimmed == "[[audio_as_voice]]" {
            continue;
        }
        if let Some(path) = parse_media_line(trimmed) {
            media_paths.push(resolve_media_path(runtime, &path)?);
            continue;
        }
        cleaned_lines.push(line);
    }
    Ok((cleaned_lines.join("\n").trim().to_string(), media_paths))
}

fn parse_media_files_arg(args: &Value, runtime: &ToolRuntime) -> Result<Vec<PathBuf>, String> {
    let Some(items) = args.get("media_files") else {
        return Ok(Vec::new());
    };
    let Some(array) = items.as_array() else {
        return Err("media_files must be an array".to_string());
    };
    let mut paths = Vec::new();
    for (index, item) in array.iter().enumerate() {
        let path = match item {
            Value::String(text) => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    return Err(format!("media_files[{index}] path must not be empty"));
                }
                trimmed.to_string()
            }
            Value::Object(map) => {
                let Some(path) = map.get("path").and_then(Value::as_str).map(str::trim) else {
                    return Err(format!("media_files[{index}].path is required"));
                };
                if path.is_empty() {
                    return Err(format!("media_files[{index}].path must not be empty"));
                }
                if let Some(is_voice) = map.get("is_voice")
                    && !is_voice.is_boolean()
                {
                    return Err(format!("media_files[{index}].is_voice must be a boolean"));
                }
                path.to_string()
            }
            _ => return Err(format!("media_files[{index}] must be an object or string")),
        };
        paths.push(resolve_media_path(runtime, &path)?);
    }
    Ok(paths)
}

fn resolve_media_path(runtime: &ToolRuntime, raw: &str) -> Result<PathBuf, String> {
    let resolved = runtime
        .resolve_path(raw)
        .map_err(|error| format!("Invalid media path '{raw}': {error}"))?;
    if !resolved.is_file() {
        return Err(format!("Media file not found: {}", resolved.display()));
    }
    Ok(resolved)
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

fn session_group_code() -> Option<String> {
    let chat_id = env::var("HERMES_SESSION_CHAT_ID").ok()?;
    let trimmed = chat_id.trim();
    trimmed
        .strip_prefix("group:")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn raw_members_to_json(members: Vec<YuanbaoMember>) -> Vec<Value> {
    members
        .into_iter()
        .map(|member| {
            json!({
                "user_id": member.user_id,
                "nickname": member.nickname,
                "role": role_label(member.role),
            })
        })
        .collect()
}

fn role_label(role: u64) -> &'static str {
    match role {
        1 => "user",
        2 => "yuanbao_ai",
        3 => "bot",
        _ => "unknown",
    }
}

fn with_yuanbao_client<T, F>(callback: F) -> Result<T, String>
where
    F: FnOnce(&mut YuanbaoClient) -> Result<T, String>,
{
    let config = load_yuanbao_config()?;
    let mut client = YuanbaoClient::connect_with_config(&config)?;
    callback(&mut client)
}

fn load_yuanbao_config() -> Result<YuanbaoConfig, String> {
    let app_key = first_non_empty_env(&["YUANBAO_APP_ID", "YUANBAO_APP_KEY"])
        .ok_or_else(|| "YUANBAO_APP_ID or YUANBAO_APP_KEY is required".to_string())?;
    let app_secret = required_env("YUANBAO_APP_SECRET")?;
    let api_domain = first_non_empty_env(&["YUANBAO_API_DOMAIN"])
        .unwrap_or_else(|| DEFAULT_API_DOMAIN.to_string());
    let ws_url = first_non_empty_env(&["YUANBAO_WS_URL"])
        .unwrap_or_else(|| DEFAULT_WS_GATEWAY_URL.to_string());
    let route_env = first_non_empty_env(&["YUANBAO_ROUTE_ENV"]).unwrap_or_default();
    Ok(YuanbaoConfig {
        app_key,
        app_secret,
        api_domain: api_domain.trim_end_matches('/').to_string(),
        ws_url,
        route_env,
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        operation_system: env::consts::OS.to_string(),
        bot_version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

fn first_non_empty_env(keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| env::var(key).ok())
        .map(|value| value.trim().to_string())
        .find(|value| !value.is_empty())
}

fn required_env(key: &str) -> Result<String, String> {
    let value = env::var(key).map_err(|_| format!("{key} is required"))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(format!("{key} is required"))
    } else {
        Ok(trimmed.to_string())
    }
}

impl YuanbaoClient {
    fn connect_with_config(config: &YuanbaoConfig) -> Result<Self, String> {
        let token = fetch_sign_token(config)?;
        Url::parse(&config.ws_url).map_err(|error| format!("Invalid YUANBAO_WS_URL: {error}"))?;
        let (mut socket, _) = connect(config.ws_url.as_str())
            .map_err(|error| format!("Yuanbao websocket connect failed: {error}"))?;
        set_yuanbao_timeouts(&mut socket, Duration::from_secs(YUANBAO_TOOL_TIMEOUT_SECS));

        let auth_msg_id = format!("auth_{:x}", unix_ts_nanos());
        let payload = encode_auth_bind(config, &token, &auth_msg_id);
        socket
            .send(Message::Binary(payload.into()))
            .map_err(|error| format!("Yuanbao auth send failed: {error}"))?;

        loop {
            let frame = read_yuanbao_frame(&mut socket)?;
            if frame.head.cmd_type == CMD_TYPE_RESPONSE
                && frame.head.cmd == CMD_AUTH_BIND
                && frame.head.msg_id == auth_msg_id
            {
                let _ = decode_auth_bind_response(&frame.data)?;
                return Ok(Self {
                    socket,
                    bot_id: token.bot_id,
                    sign_token: token.token,
                    config: config.clone(),
                });
            }
        }
    }

    fn query_group_info(&mut self, group_code: &str) -> Result<YuanbaoGroupInfo, String> {
        let req_id = format!("qgi_{}", next_seq_no());
        let response = self.send_request(&req_id, encode_query_group_info(&req_id, group_code))?;
        ensure_status_zero(&response.head, "query_group_info")?;
        decode_query_group_info_response(&response.data)
    }

    fn get_group_member_list(&mut self, group_code: &str) -> Result<Vec<YuanbaoMember>, String> {
        let req_id = format!("gml_{}", next_seq_no());
        let response = self.send_request(
            &req_id,
            encode_get_group_member_list(&req_id, group_code, 0, 200),
        )?;
        ensure_status_zero(&response.head, "get_group_member_list")?;
        decode_group_member_list_response(&response.data)
    }

    fn send_sticker(
        &mut self,
        chat_id: &str,
        sticker: &Sticker,
        reply_to: &str,
    ) -> Result<String, String> {
        self.send_chat_body(chat_id, &build_sticker_message_body(sticker), "", reply_to)
    }

    fn send_chat_message(&mut self, chat_id: &str, message: &str) -> Result<String, String> {
        self.send_chat_body(chat_id, &build_text_message_body(message), "", "")
    }

    fn send_chat_with_media(
        &mut self,
        chat_id: &str,
        message: &str,
        media_paths: &[PathBuf],
        group_context: Option<&str>,
    ) -> Result<String, String> {
        if message.trim().is_empty() && media_paths.is_empty() {
            return Err("message or media_files is required".to_string());
        }
        let mut last_message_id = None;
        if !message.trim().is_empty() {
            last_message_id = Some(self.send_chat_message_with_group_context(
                chat_id,
                message.trim(),
                group_context.unwrap_or_default(),
            )?);
        }
        for media_path in media_paths {
            let msg_body = self.prepare_media_message(media_path)?;
            last_message_id = Some(self.send_chat_body(
                chat_id,
                &msg_body,
                group_context.unwrap_or_default(),
                "",
            )?);
        }
        last_message_id.ok_or_else(|| "message or media_files is required".to_string())
    }

    fn send_chat_message_with_group_context(
        &mut self,
        chat_id: &str,
        message: &str,
        group_code: &str,
    ) -> Result<String, String> {
        self.send_chat_body(chat_id, &build_text_message_body(message), group_code, "")
    }

    fn send_chat_body(
        &mut self,
        chat_id: &str,
        msg_body: &[MsgBodyElement],
        group_context: &str,
        reply_to: &str,
    ) -> Result<String, String> {
        if let Some(group_code) = group_code_for_chat(chat_id) {
            self.send_group_message_body(group_code, msg_body, reply_to)
        } else {
            let account_id = direct_account_for_chat(chat_id).ok_or_else(|| {
                "chat_id must be group:<group_code>, direct:<account_id>, or bare account id"
                    .to_string()
            })?;
            self.send_c2c_message_body(account_id, msg_body, group_context)
        }
    }

    fn send_group_message_body(
        &mut self,
        group_code: &str,
        msg_body: &[MsgBodyElement],
        reply_to: &str,
    ) -> Result<String, String> {
        let req_id = format!("grp_{}", next_seq_no());
        let response = self.send_request(
            &req_id,
            encode_send_group_message(&req_id, group_code, &self.bot_id, msg_body, reply_to),
        )?;
        ensure_status_zero(&response.head, "send_group_message")?;
        Ok(response.head.msg_id)
    }

    fn send_c2c_message_body(
        &mut self,
        account_id: &str,
        msg_body: &[MsgBodyElement],
        group_code: &str,
    ) -> Result<String, String> {
        let req_id = format!("c2c_{}", next_seq_no());
        let response = self.send_request(
            &req_id,
            encode_send_c2c_message(&req_id, account_id, &self.bot_id, msg_body, group_code),
        )?;
        ensure_status_zero(&response.head, "send_c2c_message")?;
        Ok(response.head.msg_id)
    }

    fn prepare_media_message(&self, path: &Path) -> Result<Vec<MsgBodyElement>, String> {
        let file_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("invalid media file name: {}", path.display()))?
            .to_string();
        let bytes = fs::read(path)
            .map_err(|error| format!("reading {} failed: {error}", path.display()))?;
        if bytes.is_empty() {
            return Err(format!("media file is empty: {}", path.display()));
        }
        if bytes.len() > YUANBAO_MAX_MEDIA_BYTES {
            return Err(format!(
                "media file exceeds Yuanbao's 50 MB limit: {}",
                path.display()
            ));
        }
        let mime_type = guess_mime_type(&file_name, &bytes);
        let credentials = self.fetch_upload_credentials(&file_name)?;
        let uploaded = upload_media_to_cos(&credentials, &file_name, &mime_type, &bytes)?;
        if is_image_type(&file_name, &mime_type) {
            Ok(build_image_message_body(&uploaded))
        } else {
            Ok(build_file_message_body(&uploaded))
        }
    }

    fn fetch_upload_credentials(&self, file_name: &str) -> Result<CosCredentials, String> {
        let url = format!("{}{}", self.config.api_domain, YUANBAO_UPLOAD_INFO_PATH);
        let client = Client::builder()
            .timeout(Duration::from_secs(YUANBAO_TOOL_TIMEOUT_SECS))
            .build()
            .map_err(|error| format!("Creating Yuanbao upload client failed: {error}"))?;
        let mut request = client
            .post(url)
            .header(CONTENT_TYPE, "application/json")
            .header("X-Token", &self.sign_token)
            .header("X-ID", &self.bot_id)
            .header("X-Source", "web")
            .json(&json!({
                "fileName": file_name,
                "fileId": nonce_hex(),
                "docFrom": "localDoc",
                "docOpenId": "",
            }));
        if !self.config.route_env.is_empty() {
            request = request.header("X-Route-Env", &self.config.route_env);
        }
        let response = request
            .send()
            .map_err(|error| format!("Yuanbao upload info request failed: {error}"))?;
        let status = response.status();
        let body = response
            .text()
            .map_err(|error| format!("Reading Yuanbao upload info response failed: {error}"))?;
        if !status.is_success() {
            return Err(format!(
                "Yuanbao upload info request failed with status {}: {}",
                status.as_u16(),
                body
            ));
        }
        let payload = serde_json::from_str::<Value>(&body)
            .map_err(|error| format!("Parsing Yuanbao upload info response failed: {error}"))?;
        let code = payload.get("code").and_then(Value::as_i64).unwrap_or(0);
        if code != 0 {
            let message = payload
                .get("msg")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(format!(
                "Yuanbao upload info error: code={code}, msg={message}"
            ));
        }
        let data = payload.get("data").unwrap_or(&payload);
        Ok(CosCredentials {
            bucket_name: json_string_field(data, &["bucketName"])
                .ok_or_else(|| "Yuanbao upload info response was missing bucketName".to_string())?,
            region: json_string_field(data, &["region"]).unwrap_or_default(),
            location: json_string_field(data, &["location"])
                .ok_or_else(|| "Yuanbao upload info response was missing location".to_string())?,
            secret_id: json_string_field(data, &["encryptTmpSecretId", "tmpSecretId"])
                .unwrap_or_default(),
            secret_key: json_string_field(data, &["encryptTmpSecretKey", "tmpSecretKey"])
                .unwrap_or_default(),
            session_token: json_string_field(data, &["encryptToken", "sessionToken"])
                .unwrap_or_default(),
            start_time: json_i64_field(data, &["startTime"]),
            expired_time: json_i64_field(data, &["expiredTime"]),
            resource_url: json_string_field(data, &["resourceUrl"]),
        })
    }

    fn send_request(&mut self, req_id: &str, payload: Vec<u8>) -> Result<ConnMsg, String> {
        self.socket
            .send(Message::Binary(payload.into()))
            .map_err(|error| format!("Yuanbao request send failed: {error}"))?;
        loop {
            let frame = read_yuanbao_frame(&mut self.socket)?;
            if frame.head.cmd_type == CMD_TYPE_RESPONSE && frame.head.msg_id == req_id {
                return Ok(frame);
            }
        }
    }
}

fn ensure_status_zero(head: &ConnHead, action: &str) -> Result<(), String> {
    if head.status == 0 {
        Ok(())
    } else {
        Err(format!("{action} failed with status {}", head.status))
    }
}

fn read_yuanbao_frame(
    socket: &mut WebSocket<MaybeTlsStream<TcpStream>>,
) -> Result<ConnMsg, String> {
    loop {
        let message = socket
            .read()
            .map_err(|error| format!("Reading Yuanbao response failed: {error}"))?;
        match message {
            Message::Binary(bytes) => return decode_conn_msg(&bytes),
            Message::Ping(payload) => {
                socket
                    .send(Message::Pong(payload))
                    .map_err(|error| format!("Responding to Yuanbao ping failed: {error}"))?;
            }
            Message::Close(_) => {
                return Err("Yuanbao connection closed before a response arrived".to_string());
            }
            _ => {}
        }
    }
}

fn set_yuanbao_timeouts(socket: &mut WebSocket<MaybeTlsStream<TcpStream>>, timeout: Duration) {
    match socket.get_mut() {
        MaybeTlsStream::Plain(stream) => {
            let _ = stream.set_read_timeout(Some(timeout));
            let _ = stream.set_write_timeout(Some(timeout));
        }
        MaybeTlsStream::Rustls(stream) => {
            let tcp = stream.get_mut();
            let _ = tcp.set_read_timeout(Some(timeout));
            let _ = tcp.set_write_timeout(Some(timeout));
        }
        _ => {}
    }
}

fn fetch_sign_token(config: &YuanbaoConfig) -> Result<YuanbaoToken, String> {
    let timestamp = beijing_timestamp();
    let nonce = nonce_hex();
    let signature = compute_signature(&nonce, &timestamp, &config.app_key, &config.app_secret)?;
    let url = format!("{}/api/v5/robotLogic/sign-token", config.api_domain);
    let client = Client::builder()
        .timeout(Duration::from_secs(YUANBAO_TOOL_TIMEOUT_SECS))
        .build()
        .map_err(|error| format!("Creating Yuanbao HTTP client failed: {error}"))?;
    let mut request = client
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .header("X-AppVersion", &config.app_version)
        .header("X-OperationSystem", &config.operation_system)
        .header("X-Instance-Id", YUANBAO_INSTANCE_ID)
        .header("X-Bot-Version", &config.bot_version)
        .json(&json!({
            "app_key": config.app_key,
            "nonce": nonce,
            "signature": signature,
            "timestamp": timestamp,
        }));
    if !config.route_env.is_empty() {
        request = request.header("X-Route-Env", &config.route_env);
    }
    let response = request
        .send()
        .map_err(|error| format!("Yuanbao sign token request failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Yuanbao sign token request failed with status {}",
            response.status()
        ));
    }
    let payload = response
        .json::<Value>()
        .map_err(|error| format!("Parsing Yuanbao sign token response failed: {error}"))?;
    let code = payload.get("code").and_then(Value::as_i64).unwrap_or(0);
    if code != 0 {
        let message = payload
            .get("msg")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        return Err(format!(
            "Yuanbao sign token error: code={code}, msg={message}"
        ));
    }
    let data = payload.get("data").unwrap_or(&payload);
    let token = data
        .get("token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Yuanbao sign token response was missing token".to_string())?
        .to_string();
    let bot_id = data
        .get("bot_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Yuanbao sign token response was missing bot_id".to_string())?
        .to_string();
    Ok(YuanbaoToken { token, bot_id })
}

fn compute_signature(
    nonce: &str,
    timestamp: &str,
    app_key: &str,
    app_secret: &str,
) -> Result<String, String> {
    let plain = format!("{nonce}{timestamp}{app_key}{app_secret}");
    let mut mac =
        HmacSha256::new_from_slice(app_secret.as_bytes()).map_err(|error| error.to_string())?;
    mac.update(plain.as_bytes());
    let output = mac.finalize().into_bytes();
    Ok(hex_string(output.as_slice()))
}

fn hex_string(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(hex_digit(byte >> 4));
        out.push(hex_digit(byte & 0x0f));
    }
    out
}

fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        _ => (b'a' + (value - 10)) as char,
    }
}

fn nonce_hex() -> String {
    format!("{:032x}", unix_ts_nanos())
}

fn beijing_timestamp() -> String {
    let offset = FixedOffset::east_opt(8 * 3600).expect("valid offset");
    Utc::now()
        .with_timezone(&offset)
        .format("%Y-%m-%dT%H:%M:%S+08:00")
        .to_string()
}

fn unix_ts_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn json_string_field(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(|field| match field {
            Value::String(text) => {
                let trimmed = text.trim();
                (!trimmed.is_empty()).then_some(trimmed.to_string())
            }
            Value::Number(number) => Some(number.to_string()),
            _ => None,
        })
}

fn json_i64_field(value: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter()
        .filter_map(|key| value.get(*key))
        .find_map(|field| match field {
            Value::Number(number) => number.as_i64(),
            Value::String(text) => text.trim().parse::<i64>().ok(),
            _ => None,
        })
}

fn guess_mime_type(file_name: &str, bytes: &[u8]) -> String {
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
    match file_name
        .rsplit('.')
        .next()
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => "image/jpeg".to_string(),
        Some("png") => "image/png".to_string(),
        Some("gif") => "image/gif".to_string(),
        Some("webp") => "image/webp".to_string(),
        Some("bmp") => "image/bmp".to_string(),
        Some("heic") => "image/heic".to_string(),
        Some("tiff") => "image/tiff".to_string(),
        Some("ico") => "image/x-icon".to_string(),
        Some("pdf") => "application/pdf".to_string(),
        Some("doc") => "application/msword".to_string(),
        Some("docx") => {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string()
        }
        Some("xls") => "application/vnd.ms-excel".to_string(),
        Some("xlsx") => {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string()
        }
        Some("ppt") => "application/vnd.ms-powerpoint".to_string(),
        Some("pptx") => {
            "application/vnd.openxmlformats-officedocument.presentationml.presentation".to_string()
        }
        Some("txt") => "text/plain".to_string(),
        Some("zip") => "application/zip".to_string(),
        Some("tar") => "application/x-tar".to_string(),
        Some("gz") => "application/gzip".to_string(),
        Some("mp3") => "audio/mpeg".to_string(),
        Some("mp4") => "video/mp4".to_string(),
        Some("wav") => "audio/wav".to_string(),
        Some("ogg") => "audio/ogg".to_string(),
        Some("webm") => "video/webm".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

fn is_image_type(file_name: &str, mime_type: &str) -> bool {
    mime_type.starts_with("image/")
        || matches!(
            file_name
                .rsplit('.')
                .next()
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("jpg" | "jpeg" | "png" | "gif" | "webp" | "bmp" | "heic" | "tiff" | "ico")
        )
}

fn image_format_for_mime(mime_type: &str) -> u64 {
    match mime_type.to_ascii_lowercase().as_str() {
        "image/jpeg" | "image/jpg" => 1,
        "image/gif" => 2,
        "image/png" => 3,
        "image/bmp" => 4,
        _ => 255,
    }
}

fn md5_hex(bytes: &[u8]) -> String {
    format!("{:x}", md5_compute(bytes))
}

fn parse_image_size(bytes: &[u8]) -> Option<(u64, u64)> {
    parse_png_size(bytes)
        .or_else(|| parse_jpeg_size(bytes))
        .or_else(|| parse_gif_size(bytes))
        .or_else(|| parse_webp_size(bytes))
}

fn parse_png_size(bytes: &[u8]) -> Option<(u64, u64)> {
    if bytes.len() < 24 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return None;
    }
    Some((
        u32::from_be_bytes(bytes[16..20].try_into().ok()?) as u64,
        u32::from_be_bytes(bytes[20..24].try_into().ok()?) as u64,
    ))
}

fn parse_jpeg_size(bytes: &[u8]) -> Option<(u64, u64)> {
    if bytes.len() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8 {
        return None;
    }
    let mut cursor = 2usize;
    while cursor + 9 < bytes.len() {
        if bytes[cursor] != 0xff {
            cursor += 1;
            continue;
        }
        let marker = bytes[cursor + 1];
        if matches!(marker, 0xc0 | 0xc2) {
            let height = u16::from_be_bytes(bytes[cursor + 5..cursor + 7].try_into().ok()?) as u64;
            let width = u16::from_be_bytes(bytes[cursor + 7..cursor + 9].try_into().ok()?) as u64;
            return Some((width, height));
        }
        if cursor + 4 > bytes.len() {
            break;
        }
        let segment_len =
            u16::from_be_bytes(bytes[cursor + 2..cursor + 4].try_into().ok()?) as usize;
        if segment_len < 2 {
            break;
        }
        cursor += 2 + segment_len;
    }
    None
}

fn parse_gif_size(bytes: &[u8]) -> Option<(u64, u64)> {
    if bytes.len() < 10 || !(bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        return None;
    }
    Some((
        u16::from_le_bytes(bytes[6..8].try_into().ok()?) as u64,
        u16::from_le_bytes(bytes[8..10].try_into().ok()?) as u64,
    ))
}

fn parse_webp_size(bytes: &[u8]) -> Option<(u64, u64)> {
    if bytes.len() < 16 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return None;
    }
    match &bytes[12..16] {
        b"VP8 "
            if bytes.len() >= 30 && bytes[23] == 0x9d && bytes[24] == 0x01 && bytes[25] == 0x2a =>
        {
            Some((
                (u16::from_le_bytes(bytes[26..28].try_into().ok()?) & 0x3fff) as u64,
                (u16::from_le_bytes(bytes[28..30].try_into().ok()?) & 0x3fff) as u64,
            ))
        }
        b"VP8L" if bytes.len() >= 25 && bytes[20] == 0x2f => {
            let bits = u32::from_le_bytes(bytes[21..25].try_into().ok()?);
            Some((
                ((bits & 0x3fff) + 1) as u64,
                (((bits >> 14) & 0x3fff) + 1) as u64,
            ))
        }
        b"VP8X" if bytes.len() >= 30 => {
            let width =
                (u32::from(bytes[24]) | (u32::from(bytes[25]) << 8) | (u32::from(bytes[26]) << 16))
                    + 1;
            let height =
                (u32::from(bytes[27]) | (u32::from(bytes[28]) << 8) | (u32::from(bytes[29]) << 16))
                    + 1;
            Some((width as u64, height as u64))
        }
        _ => None,
    }
}

fn percent_encode_component(value: &str) -> String {
    form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

fn percent_encode_path(value: &str) -> String {
    percent_encode_component(value).replace("%2F", "/")
}

fn hmac_sha1_hex(key: &[u8], data: &str) -> Result<String, String> {
    let mut mac = HmacSha1::new_from_slice(key).map_err(|error| error.to_string())?;
    mac.update(data.as_bytes());
    Ok(hex_string(mac.finalize().into_bytes().as_slice()))
}

fn cos_authorization(
    path: &str,
    headers: &[(String, String)],
    secret_id: &str,
    secret_key: &str,
    start_time: i64,
    expire_seconds: i64,
) -> Result<String, String> {
    let q_sign_time = format!("{start_time};{}", start_time + expire_seconds.max(1));
    let sign_key = hmac_sha1_hex(secret_key.as_bytes(), &q_sign_time)?;
    let mut sorted_headers = headers
        .iter()
        .filter(|(_, value)| !value.is_empty())
        .map(|(key, value)| (key.to_ascii_lowercase(), percent_encode_component(value)))
        .collect::<Vec<_>>();
    sorted_headers.sort_by(|left, right| left.0.cmp(&right.0));
    let header_list = sorted_headers
        .iter()
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let header_str = sorted_headers
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&");
    let http_string = format!("put\n{path}\n\n{header_str}\n");
    let http_sha = hex_string(Sha1::digest(http_string.as_bytes()).as_slice());
    let string_to_sign = format!("sha1\n{q_sign_time}\n{http_sha}\n");
    let signature = hmac_sha1_hex(sign_key.as_bytes(), &string_to_sign)?;
    Ok(format!(
        "q-sign-algorithm=sha1&q-ak={secret_id}&q-sign-time={q_sign_time}&q-key-time={q_sign_time}&q-header-list={header_list}&q-url-param-list=&q-signature={signature}"
    ))
}

fn upload_target(credentials: &CosCredentials) -> Result<(String, String, String), String> {
    if let Some(resource_url) = credentials
        .resource_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let parsed = Url::parse(resource_url)
            .map_err(|error| format!("Invalid Yuanbao resourceUrl: {error}"))?;
        let host = parsed
            .host_str()
            .map(str::to_string)
            .ok_or_else(|| "Yuanbao resourceUrl was missing a host".to_string())?;
        let path = if parsed.path().is_empty() {
            "/".to_string()
        } else {
            parsed.path().to_string()
        };
        return Ok((resource_url.to_string(), host, path));
    }
    if credentials.bucket_name.trim().is_empty() {
        return Err("Yuanbao upload credentials were missing bucketName".to_string());
    }
    let encoded_key = percent_encode_path(&credentials.location);
    let host = if credentials.region.trim().is_empty() {
        format!("{}.cos.accelerate.myqcloud.com", credentials.bucket_name)
    } else {
        format!(
            "{}.cos.{}.myqcloud.com",
            credentials.bucket_name, credentials.region
        )
    };
    let path = format!("/{}", encoded_key.trim_start_matches('/'));
    Ok((format!("https://{host}{path}"), host, path))
}

fn upload_media_to_cos(
    credentials: &CosCredentials,
    file_name: &str,
    mime_type: &str,
    bytes: &[u8],
) -> Result<UploadedMedia, String> {
    if credentials.secret_id.trim().is_empty()
        || credentials.secret_key.trim().is_empty()
        || credentials.location.trim().is_empty()
    {
        return Err("Yuanbao upload credentials were incomplete".to_string());
    }
    let (upload_url, host, path) = upload_target(credentials)?;
    let now = i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
    .unwrap_or(0);
    let sign_start = credentials.start_time.unwrap_or(now);
    let expire_seconds = credentials
        .expired_time
        .map(|value| value.saturating_sub(now))
        .filter(|value| *value > 0)
        .unwrap_or(3600);
    let authorization = cos_authorization(
        &path,
        &[
            ("content-type".to_string(), mime_type.to_string()),
            ("host".to_string(), host.clone()),
            (
                "x-cos-security-token".to_string(),
                credentials.session_token.clone(),
            ),
        ],
        &credentials.secret_id,
        &credentials.secret_key,
        sign_start,
        expire_seconds,
    )?;
    let client = Client::builder()
        .timeout(Duration::from_secs(YUANBAO_TOOL_TIMEOUT_SECS * 4))
        .build()
        .map_err(|error| format!("Creating Yuanbao COS client failed: {error}"))?;
    let response = client
        .put(&upload_url)
        .header("Authorization", authorization)
        .header(CONTENT_TYPE, mime_type)
        .header("x-cos-security-token", &credentials.session_token)
        .body(bytes.to_vec())
        .send()
        .map_err(|error| format!("Yuanbao COS upload failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .unwrap_or_else(|_| "<unreadable body>".to_string());
        return Err(format!(
            "Yuanbao COS upload failed with status {}: {}",
            status.as_u16(),
            body
        ));
    }
    let (width, height) = if is_image_type(file_name, mime_type) {
        parse_image_size(bytes)
            .map(|(w, h)| (Some(w), Some(h)))
            .unwrap_or((None, None))
    } else {
        (None, None)
    };
    Ok(UploadedMedia {
        url: credentials
            .resource_url
            .clone()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(upload_url),
        uuid: md5_hex(bytes),
        size: bytes.len() as u64,
        file_name: file_name.to_string(),
        mime_type: mime_type.to_string(),
        width,
        height,
    })
}

fn build_image_message_body(uploaded: &UploadedMedia) -> Vec<MsgBodyElement> {
    let mut image_info_array = Vec::new();
    let mut info = ImageInfo {
        kind: 1,
        size: uploaded.size,
        width: uploaded.width.unwrap_or(0),
        height: uploaded.height.unwrap_or(0),
        url: uploaded.url.clone(),
    };
    if info.width == 0 {
        info.width = 0;
    }
    if info.height == 0 {
        info.height = 0;
    }
    image_info_array.push(info);
    vec![MsgBodyElement {
        msg_type: "TIMImageElem".to_string(),
        msg_content: MsgContent {
            uuid: Some(uploaded.uuid.clone()),
            image_format: Some(image_format_for_mime(&uploaded.mime_type)),
            url: Some(uploaded.url.clone()),
            image_info_array,
            ..MsgContent::default()
        },
    }]
}

fn build_file_message_body(uploaded: &UploadedMedia) -> Vec<MsgBodyElement> {
    vec![MsgBodyElement {
        msg_type: "TIMFileElem".to_string(),
        msg_content: MsgContent {
            uuid: Some(uploaded.uuid.clone()),
            url: Some(uploaded.url.clone()),
            file_size: Some(uploaded.size),
            file_name: Some(uploaded.file_name.clone()),
            ..MsgContent::default()
        },
    }]
}

fn next_seq_no() -> u32 {
    YUANBAO_SEQ.fetch_add(1, Ordering::Relaxed)
}

fn group_code_for_chat(chat_id: &str) -> Option<&str> {
    chat_id
        .trim()
        .strip_prefix("group:")
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn direct_account_for_chat(chat_id: &str) -> Option<&str> {
    let trimmed = chat_id.trim();
    if let Some(value) = trimmed.strip_prefix("direct:") {
        let value = value.trim();
        (!value.is_empty()).then_some(value)
    } else if !trimmed.is_empty() {
        Some(trimmed)
    } else {
        None
    }
}

fn build_text_message_body(message: &str) -> Vec<MsgBodyElement> {
    vec![MsgBodyElement {
        msg_type: "TIMTextElem".to_string(),
        msg_content: MsgContent {
            text: Some(message.to_string()),
            ..MsgContent::default()
        },
    }]
}

fn build_sticker_message_body(sticker: &Sticker) -> Vec<MsgBodyElement> {
    let data = json!({
        "sticker_id": sticker.sticker_id,
        "package_id": sticker.package_id,
        "width": sticker.width,
        "height": sticker.height,
        "formats": sticker.formats,
        "name": sticker.name,
    })
    .to_string();
    vec![MsgBodyElement {
        msg_type: "TIMFaceElem".to_string(),
        msg_content: MsgContent {
            index: Some(0),
            data: Some(data),
            ..MsgContent::default()
        },
    }]
}

fn encode_auth_bind(config: &YuanbaoConfig, token: &YuanbaoToken, msg_id: &str) -> Vec<u8> {
    let mut auth = Vec::new();
    push_string_field(&mut auth, 1, &token.bot_id);
    push_string_field(&mut auth, 2, "web");
    push_string_field(&mut auth, 3, &token.token);

    let mut device = Vec::new();
    push_string_field(&mut device, 1, &config.app_version);
    push_string_field(&mut device, 2, &config.operation_system);
    push_string_field(&mut device, 10, YUANBAO_INSTANCE_ID);
    push_string_field(&mut device, 24, &config.bot_version);

    let mut req = Vec::new();
    push_string_field(&mut req, 1, &config.app_key);
    push_bytes_field(&mut req, 2, &auth);
    push_bytes_field(&mut req, 3, &device);
    if !config.route_env.is_empty() {
        push_string_field(&mut req, 5, &config.route_env);
    }

    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        CMD_AUTH_BIND,
        u64::from(next_seq_no()),
        msg_id,
        MODULE_CONN_ACCESS,
        &req,
        false,
    )
}

fn encode_query_group_info(req_id: &str, group_code: &str) -> Vec<u8> {
    let mut body = Vec::new();
    push_string_field(&mut body, 1, group_code);
    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        CMD_QUERY_GROUP_INFO,
        u64::from(next_seq_no()),
        req_id,
        BIZ_PACKAGE,
        &body,
        false,
    )
}

fn encode_get_group_member_list(
    req_id: &str,
    group_code: &str,
    offset: u64,
    limit: u64,
) -> Vec<u8> {
    let mut body = Vec::new();
    push_string_field(&mut body, 1, group_code);
    if offset > 0 {
        push_varint_field(&mut body, 2, offset);
    }
    push_varint_field(&mut body, 3, limit);
    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        CMD_GET_GROUP_MEMBER_LIST,
        u64::from(next_seq_no()),
        req_id,
        BIZ_PACKAGE,
        &body,
        false,
    )
}

fn encode_send_c2c_message(
    req_id: &str,
    to_account: &str,
    from_account: &str,
    msg_body: &[MsgBodyElement],
    group_code: &str,
) -> Vec<u8> {
    let mut body = Vec::new();
    push_string_field(&mut body, 1, req_id);
    push_string_field(&mut body, 2, to_account);
    if !from_account.is_empty() {
        push_string_field(&mut body, 3, from_account);
    }
    for element in msg_body {
        push_bytes_field(&mut body, 5, &encode_msg_body_element(element));
    }
    if !group_code.is_empty() {
        push_string_field(&mut body, 6, group_code);
    }
    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        CMD_SEND_C2C_MESSAGE,
        u64::from(next_seq_no()),
        req_id,
        BIZ_PACKAGE,
        &body,
        false,
    )
}

fn encode_send_group_message(
    req_id: &str,
    group_code: &str,
    from_account: &str,
    msg_body: &[MsgBodyElement],
    reply_to: &str,
) -> Vec<u8> {
    let mut body = Vec::new();
    push_string_field(&mut body, 1, req_id);
    push_string_field(&mut body, 2, group_code);
    if !from_account.is_empty() {
        push_string_field(&mut body, 3, from_account);
    }
    for element in msg_body {
        push_bytes_field(&mut body, 6, &encode_msg_body_element(element));
    }
    if !reply_to.is_empty() {
        push_string_field(&mut body, 7, reply_to);
    }
    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        CMD_SEND_GROUP_MESSAGE,
        u64::from(next_seq_no()),
        req_id,
        BIZ_PACKAGE,
        &body,
        false,
    )
}

fn encode_msg_body_element(element: &MsgBodyElement) -> Vec<u8> {
    let mut body = Vec::new();
    push_string_field(&mut body, 1, &element.msg_type);
    push_bytes_field(&mut body, 2, &encode_msg_content(&element.msg_content));
    body
}

fn encode_msg_content(content: &MsgContent) -> Vec<u8> {
    let mut body = Vec::new();
    if let Some(text) = &content.text {
        push_string_field(&mut body, 1, text);
    }
    if let Some(uuid) = &content.uuid {
        push_string_field(&mut body, 2, uuid);
    }
    if let Some(image_format) = content.image_format {
        push_varint_field(&mut body, 3, image_format);
    }
    if let Some(data) = &content.data {
        push_string_field(&mut body, 4, data);
    }
    if let Some(url) = &content.url {
        push_string_field(&mut body, 10, url);
    }
    if let Some(file_size) = content.file_size {
        push_varint_field(&mut body, 11, file_size);
    }
    if let Some(file_name) = &content.file_name {
        push_string_field(&mut body, 12, file_name);
    }
    if let Some(index) = content.index {
        push_varint_field(&mut body, 9, index);
    }
    for image in &content.image_info_array {
        let mut image_body = Vec::new();
        if image.kind > 0 {
            push_varint_field(&mut image_body, 1, image.kind);
        }
        if image.size > 0 {
            push_varint_field(&mut image_body, 2, image.size);
        }
        if image.width > 0 {
            push_varint_field(&mut image_body, 3, image.width);
        }
        if image.height > 0 {
            push_varint_field(&mut image_body, 4, image.height);
        }
        if !image.url.is_empty() {
            push_string_field(&mut image_body, 5, &image.url);
        }
        push_bytes_field(&mut body, 8, &image_body);
    }
    body
}

fn encode_conn_msg_full(
    cmd_type: u64,
    cmd: &str,
    seq_no: u64,
    msg_id: &str,
    module: &str,
    data: &[u8],
    need_ack: bool,
) -> Vec<u8> {
    let mut head = Vec::new();
    if cmd_type != 0 {
        push_varint_field(&mut head, 1, cmd_type);
    }
    if !cmd.is_empty() {
        push_string_field(&mut head, 2, cmd);
    }
    if seq_no != 0 {
        push_varint_field(&mut head, 3, seq_no);
    }
    if !msg_id.is_empty() {
        push_string_field(&mut head, 4, msg_id);
    }
    if !module.is_empty() {
        push_string_field(&mut head, 5, module);
    }
    if need_ack {
        push_varint_field(&mut head, 6, 1);
    }

    let mut out = Vec::new();
    push_bytes_field(&mut out, 1, &head);
    if !data.is_empty() {
        push_bytes_field(&mut out, 2, data);
    }
    out
}

fn decode_conn_msg(data: &[u8]) -> Result<ConnMsg, String> {
    let fields = parse_fields(data)?;
    let head_bytes =
        field_bytes(&fields, 1).ok_or_else(|| "Yuanbao frame was missing head".to_string())?;
    let head_fields = parse_fields(head_bytes)?;
    let head = ConnHead {
        cmd_type: field_varint(&head_fields, 1).unwrap_or(0),
        cmd: field_string(&head_fields, 2).unwrap_or_default(),
        msg_id: field_string(&head_fields, 4).unwrap_or_default(),
        status: field_varint(&head_fields, 10).unwrap_or(0),
    };
    Ok(ConnMsg {
        head,
        data: field_bytes(&fields, 2)
            .map(ToOwned::to_owned)
            .unwrap_or_default(),
    })
}

fn decode_auth_bind_response(data: &[u8]) -> Result<String, String> {
    let fields = parse_fields(data)?;
    let code = field_varint(&fields, 1).unwrap_or(0);
    if code != 0 {
        let message = field_string(&fields, 2).unwrap_or_else(|| "unknown error".to_string());
        return Err(format!("Yuanbao auth failed: code={code}, msg={message}"));
    }
    let connect_id = field_string(&fields, 3)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "Yuanbao auth response was missing connect_id".to_string())?;
    Ok(connect_id)
}

fn decode_query_group_info_response(data: &[u8]) -> Result<YuanbaoGroupInfo, String> {
    let fields = parse_fields(data)?;
    let code = field_varint(&fields, 1).unwrap_or(0);
    if code != 0 {
        let message = field_string(&fields, 2).unwrap_or_else(|| "unknown error".to_string());
        return Err(format!("query_group_info returned code={code}: {message}"));
    }
    let nested = field_bytes(&fields, 3)
        .ok_or_else(|| "query_group_info response was missing group info".to_string())?;
    let info_fields = parse_fields(nested)?;
    Ok(YuanbaoGroupInfo {
        group_name: field_string(&info_fields, 1).unwrap_or_default(),
        owner_id: field_string(&info_fields, 2).unwrap_or_default(),
        owner_nickname: field_string(&info_fields, 3).unwrap_or_default(),
        member_count: field_varint(&info_fields, 4).unwrap_or(0),
    })
}

fn decode_group_member_list_response(data: &[u8]) -> Result<Vec<YuanbaoMember>, String> {
    let fields = parse_fields(data)?;
    let code = field_varint(&fields, 1).unwrap_or(0);
    if code != 0 {
        let message = field_string(&fields, 2).unwrap_or_else(|| "unknown error".to_string());
        return Err(format!(
            "get_group_member_list returned code={code}: {message}"
        ));
    }
    let members = repeated_bytes(&fields, 3)
        .into_iter()
        .map(|bytes| {
            let member_fields = parse_fields(bytes)?;
            Ok(YuanbaoMember {
                user_id: field_string(&member_fields, 1).unwrap_or_default(),
                nickname: field_string(&member_fields, 2).unwrap_or_default(),
                role: field_varint(&member_fields, 3).unwrap_or(0),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(members)
}

fn parse_fields(data: &[u8]) -> Result<Vec<ProtoField>, String> {
    let mut cursor = 0usize;
    let mut out = Vec::new();
    while cursor < data.len() {
        let key = decode_varint(data, &mut cursor)?;
        let number =
            u32::try_from(key >> 3).map_err(|_| "Invalid protobuf field number".to_string())?;
        let wire_type =
            u8::try_from(key & 0x07).map_err(|_| "Invalid protobuf wire type".to_string())?;
        let value = match wire_type {
            WT_VARINT => ProtoValue::Varint(decode_varint(data, &mut cursor)?),
            WT_LEN => {
                let len = usize::try_from(decode_varint(data, &mut cursor)?)
                    .map_err(|_| "Invalid protobuf length".to_string())?;
                if cursor + len > data.len() {
                    return Err("Invalid protobuf length-delimited field".to_string());
                }
                let bytes = data[cursor..cursor + len].to_vec();
                cursor += len;
                ProtoValue::Bytes(bytes)
            }
            _ => return Err(format!("Unsupported protobuf wire type: {wire_type}")),
        };
        out.push(ProtoField { number, value });
    }
    Ok(out)
}

fn decode_varint(data: &[u8], cursor: &mut usize) -> Result<u64, String> {
    let mut shift = 0u32;
    let mut value = 0u64;
    while *cursor < data.len() {
        let byte = data[*cursor];
        *cursor += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            return Err("Invalid protobuf varint".to_string());
        }
    }
    Err("Unexpected EOF while reading protobuf varint".to_string())
}

fn field_varint(fields: &[ProtoField], number: u32) -> Option<u64> {
    fields
        .iter()
        .find_map(|field| match (&field.value, field.number == number) {
            (ProtoValue::Varint(value), true) => Some(*value),
            _ => None,
        })
}

fn field_bytes(fields: &[ProtoField], number: u32) -> Option<&[u8]> {
    fields
        .iter()
        .find_map(|field| match (&field.value, field.number == number) {
            (ProtoValue::Bytes(bytes), true) => Some(bytes.as_slice()),
            _ => None,
        })
}

fn repeated_bytes(fields: &[ProtoField], number: u32) -> Vec<&[u8]> {
    fields
        .iter()
        .filter_map(|field| match (&field.value, field.number == number) {
            (ProtoValue::Bytes(bytes), true) => Some(bytes.as_slice()),
            _ => None,
        })
        .collect()
}

fn field_string(fields: &[ProtoField], number: u32) -> Option<String> {
    let bytes = field_bytes(fields, number)?;
    String::from_utf8(bytes.to_vec()).ok()
}

fn push_varint_field(buffer: &mut Vec<u8>, number: u32, value: u64) {
    buffer.extend_from_slice(&encode_varint(
        (u64::from(number) << 3) | u64::from(WT_VARINT),
    ));
    buffer.extend_from_slice(&encode_varint(value));
}

fn push_string_field(buffer: &mut Vec<u8>, number: u32, value: &str) {
    if value.is_empty() {
        return;
    }
    push_bytes_field(buffer, number, value.as_bytes());
}

fn push_bytes_field(buffer: &mut Vec<u8>, number: u32, value: &[u8]) {
    buffer.extend_from_slice(&encode_varint((u64::from(number) << 3) | u64::from(WT_LEN)));
    buffer.extend_from_slice(&encode_varint(value.len() as u64));
    buffer.extend_from_slice(value);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Mutex, OnceLock};
    use std::thread;
    use tempfile::TempDir;

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

    fn runtime_for(temp: &TempDir) -> ToolRuntime {
        ToolRuntime::new(temp.path()).with_hermes_home(temp.path())
    }

    fn mock_http_server<F>(request_count: usize, handler: F) -> (String, thread::JoinHandle<()>)
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
        F: FnOnce(tungstenite::WebSocket<TcpStream>) + Send + 'static,
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

    fn auth_bind_response(msg_id: &str, connect_id: &str) -> Vec<u8> {
        let mut data = Vec::new();
        push_varint_field(&mut data, 1, 0);
        push_string_field(&mut data, 3, connect_id);
        encode_conn_msg_full(
            CMD_TYPE_RESPONSE,
            CMD_AUTH_BIND,
            1,
            msg_id,
            MODULE_CONN_ACCESS,
            &data,
            false,
        )
    }

    fn group_info_response(msg_id: &str) -> Vec<u8> {
        let mut group = Vec::new();
        push_string_field(&mut group, 1, "Pai Alpha");
        push_string_field(&mut group, 2, "owner-1");
        push_string_field(&mut group, 3, "Owner");
        push_varint_field(&mut group, 4, 42);

        let mut data = Vec::new();
        push_varint_field(&mut data, 1, 0);
        push_bytes_field(&mut data, 3, &group);
        encode_conn_msg_full(
            CMD_TYPE_RESPONSE,
            CMD_QUERY_GROUP_INFO,
            2,
            msg_id,
            BIZ_PACKAGE,
            &data,
            false,
        )
    }

    fn group_members_response(msg_id: &str) -> Vec<u8> {
        let mut alice = Vec::new();
        push_string_field(&mut alice, 1, "u1");
        push_string_field(&mut alice, 2, "Alice");
        push_varint_field(&mut alice, 3, 1);

        let mut bot = Vec::new();
        push_string_field(&mut bot, 1, "u2");
        push_string_field(&mut bot, 2, "YB Bot");
        push_varint_field(&mut bot, 3, 3);

        let mut data = Vec::new();
        push_varint_field(&mut data, 1, 0);
        push_bytes_field(&mut data, 3, &alice);
        push_bytes_field(&mut data, 3, &bot);
        push_varint_field(&mut data, 5, 1);
        encode_conn_msg_full(
            CMD_TYPE_RESPONSE,
            CMD_GET_GROUP_MEMBER_LIST,
            2,
            msg_id,
            BIZ_PACKAGE,
            &data,
            false,
        )
    }

    fn ok_send_response(cmd: &str, msg_id: &str) -> Vec<u8> {
        encode_conn_msg_full(CMD_TYPE_RESPONSE, cmd, 2, msg_id, BIZ_PACKAGE, &[], false)
    }

    #[test]
    fn empty_query_returns_first_items() {
        let matches = search_stickers("", 3);
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].name, "六六六");
        assert_eq!(matches[1].name, "我想开了");
        assert_eq!(matches[2].name, "害羞");
    }

    #[test]
    fn exact_sticker_id_match_ranks_first() {
        let matches = search_stickers("252", 3);
        assert_eq!(matches.first().map(|sticker| sticker.name), Some("比心"));
    }

    #[test]
    fn fuzzy_name_match_uses_compacted_text() {
        let matches = search_stickers("暗 中 观 察", 3);
        assert_eq!(
            matches.first().map(|sticker| sticker.name),
            Some("暗中观察")
        );
    }

    #[test]
    fn english_description_match_works() {
        let matches = search_stickers("awesome", 3);
        assert_eq!(matches.first().map(|sticker| sticker.name), Some("六六六"));
    }

    #[test]
    fn fullwidth_query_is_normalized() {
        let matches = search_stickers("ＯＫ", 3);
        assert_eq!(matches.first().map(|sticker| sticker.name), Some("ok"));
    }

    #[test]
    fn limit_validation_rejects_out_of_range_values() {
        assert_eq!(
            optional_limit(&json!({ "limit": 0 }), "limit"),
            Err("limit must be between 1 and 50".to_string())
        );
        assert_eq!(
            optional_limit(&json!({ "limit": 51 }), "limit"),
            Err("limit must be between 1 and 50".to_string())
        );
    }

    #[test]
    fn handler_returns_json_payload() {
        let output = handle_yb_search_sticker(
            &json!({ "query": "比心", "limit": 1 }),
            &ToolRuntime::default(),
        );
        let value: Value = serde_json::from_str(&output).expect("valid json");
        assert_eq!(value.get("success").and_then(Value::as_bool), Some(true));
        assert_eq!(value.get("count").and_then(Value::as_u64), Some(1));
        assert_eq!(
            value
                .get("results")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|item| item.get("name"))
                .and_then(Value::as_str),
            Some("比心")
        );
    }

    #[test]
    fn group_info_handler_uses_mock_yuanbao_transport() {
        let _guard = acquire_test_lock();
        let (http_base, http_join) = mock_http_server(1, |_index, _headers, _body| {
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
            let auth = decode_conn_msg(&websocket.read().unwrap().into_data()).unwrap();
            assert_eq!(auth.head.cmd, CMD_AUTH_BIND);
            websocket
                .send(Message::Binary(
                    auth_bind_response(&auth.head.msg_id, "conn-1").into(),
                ))
                .unwrap();

            let request = decode_conn_msg(&websocket.read().unwrap().into_data()).unwrap();
            assert_eq!(request.head.cmd, CMD_QUERY_GROUP_INFO);
            websocket
                .send(Message::Binary(
                    group_info_response(&request.head.msg_id).into(),
                ))
                .unwrap();
        });

        with_env_var("YUANBAO_APP_ID", Some("app-key"));
        with_env_var("YUANBAO_APP_SECRET", Some("secret-key"));
        with_env_var("YUANBAO_API_DOMAIN", Some(&http_base));
        with_env_var("YUANBAO_WS_URL", Some(&ws_url));

        let output =
            handle_yb_query_group_info(&json!({ "group_code": "g-1" }), &ToolRuntime::default());
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["success"], json!(true));
        assert_eq!(value["group_name"], json!("Pai Alpha"));
        assert_eq!(value["member_count"], json!(42));
        assert_eq!(value["owner"]["user_id"], json!("owner-1"));

        http_join.join().unwrap();
        ws_join.join().unwrap();
    }

    #[test]
    fn group_members_handler_filters_bots() {
        let _guard = acquire_test_lock();
        let (http_base, http_join) = mock_http_server(1, |_index, _headers, _body| {
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
            let auth = decode_conn_msg(&websocket.read().unwrap().into_data()).unwrap();
            websocket
                .send(Message::Binary(
                    auth_bind_response(&auth.head.msg_id, "conn-2").into(),
                ))
                .unwrap();

            let request = decode_conn_msg(&websocket.read().unwrap().into_data()).unwrap();
            assert_eq!(request.head.cmd, CMD_GET_GROUP_MEMBER_LIST);
            websocket
                .send(Message::Binary(
                    group_members_response(&request.head.msg_id).into(),
                ))
                .unwrap();
        });

        with_env_var("YUANBAO_APP_ID", Some("app-key"));
        with_env_var("YUANBAO_APP_SECRET", Some("secret-key"));
        with_env_var("YUANBAO_API_DOMAIN", Some(&http_base));
        with_env_var("YUANBAO_WS_URL", Some(&ws_url));

        let output = handle_yb_query_group_members(
            &json!({ "group_code": "g-1", "action": "list_bots", "mention": true }),
            &ToolRuntime::default(),
        );
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["success"], json!(true));
        assert_eq!(value["members"].as_array().unwrap().len(), 1);
        assert_eq!(value["members"][0]["nickname"], json!("YB Bot"));
        assert_eq!(value["mention_hint"], json!(MENTION_HINT));

        http_join.join().unwrap();
        ws_join.join().unwrap();
    }

    #[test]
    fn send_sticker_handler_uses_session_chat_id() {
        let _guard = acquire_test_lock();
        let (http_base, http_join) = mock_http_server(1, |_index, _headers, _body| {
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
            let auth = decode_conn_msg(&websocket.read().unwrap().into_data()).unwrap();
            websocket
                .send(Message::Binary(
                    auth_bind_response(&auth.head.msg_id, "conn-3").into(),
                ))
                .unwrap();

            let request = decode_conn_msg(&websocket.read().unwrap().into_data()).unwrap();
            assert_eq!(request.head.cmd, CMD_SEND_GROUP_MESSAGE);
            let body_fields = parse_fields(&request.data).unwrap();
            let elements = repeated_bytes(&body_fields, 6);
            let first = parse_fields(elements[0]).unwrap();
            assert_eq!(field_string(&first, 1).as_deref(), Some("TIMFaceElem"));
            websocket
                .send(Message::Binary(
                    ok_send_response(CMD_SEND_GROUP_MESSAGE, &request.head.msg_id).into(),
                ))
                .unwrap();
        });

        with_env_var("YUANBAO_APP_ID", Some("app-key"));
        with_env_var("YUANBAO_APP_SECRET", Some("secret-key"));
        with_env_var("YUANBAO_API_DOMAIN", Some(&http_base));
        with_env_var("YUANBAO_WS_URL", Some(&ws_url));
        with_env_var("HERMES_SESSION_CHAT_ID", Some("group:g-1"));

        let output = handle_yb_send_sticker(&json!({ "sticker": "比心" }), &ToolRuntime::default());
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["success"], json!(true));
        assert_eq!(value["sticker"]["name"], json!("比心"));
        assert_eq!(value["chat_id"], json!("group:g-1"));

        http_join.join().unwrap();
        ws_join.join().unwrap();
    }

    #[test]
    fn send_dm_supports_media_request() {
        let _guard = acquire_test_lock();
        let temp = TempDir::new().unwrap();
        let runtime = runtime_for(&temp);
        let media_path = temp.path().join("image.png");
        fs::write(
            &media_path,
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01",
        )
        .unwrap();
        let (upload_base, upload_join) = mock_http_server(1, move |_index, headers, body| {
            assert!(headers.starts_with("PUT /uploaded/dm.png "));
            assert!(
                headers
                    .to_ascii_lowercase()
                    .contains("authorization: q-sign-algorithm=sha1")
            );
            assert!(!body.is_empty());
            (200, "{}".to_string())
        });
        let upload_url = format!("{upload_base}/uploaded/dm.png");
        let (http_base, http_join) = mock_http_server(2, move |index, headers, body| match index {
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
                            "location": "/uploaded/dm.png",
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
            let auth = decode_conn_msg(&websocket.read().unwrap().into_data()).unwrap();
            websocket
                .send(Message::Binary(
                    auth_bind_response(&auth.head.msg_id, "conn-4").into(),
                ))
                .unwrap();

            let members = decode_conn_msg(&websocket.read().unwrap().into_data()).unwrap();
            assert_eq!(members.head.cmd, CMD_GET_GROUP_MEMBER_LIST);
            websocket
                .send(Message::Binary(
                    group_members_response(&members.head.msg_id).into(),
                ))
                .unwrap();

            let send = decode_conn_msg(&websocket.read().unwrap().into_data()).unwrap();
            assert_eq!(send.head.cmd, CMD_SEND_C2C_MESSAGE);
            let body_fields = parse_fields(&send.data).unwrap();
            let elements = repeated_bytes(&body_fields, 5);
            let first = parse_fields(elements[0]).unwrap();
            assert_eq!(field_string(&first, 1).as_deref(), Some("TIMImageElem"));
            websocket
                .send(Message::Binary(
                    ok_send_response(CMD_SEND_C2C_MESSAGE, &send.head.msg_id).into(),
                ))
                .unwrap();
        });

        with_env_var("YUANBAO_APP_ID", Some("app-key"));
        with_env_var("YUANBAO_APP_SECRET", Some("secret-key"));
        with_env_var("YUANBAO_API_DOMAIN", Some(&http_base));
        with_env_var("YUANBAO_WS_URL", Some(&ws_url));

        let output = handle_yb_send_dm(
            &json!({
                "group_code": "g-1",
                "name": "Alice",
                "message": format!("MEDIA:{}", media_path.display())
            }),
            &runtime,
        );
        let value: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(value["success"], json!(true));
        assert_eq!(value["user_id"], json!("u1"));
        assert_eq!(value["nickname"], json!("Alice"));
        assert!(value["message_id"].as_str().unwrap().starts_with("c2c_"));

        http_join.join().unwrap();
        upload_join.join().unwrap();
        ws_join.join().unwrap();
    }
}
