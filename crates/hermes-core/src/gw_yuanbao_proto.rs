//! gw_yuanbao_proto - Yuanbao WebSocket protocol codec (native Rust port).
//!
//! Faithful port of `gateway/platforms/yuanbao_proto.py`.
//!
//! Protocol layering:
//!   WebSocket frame
//!     └── ConnMsg (protobuf: trpc.yuanbao.conn_common.ConnMsg)
//!           ├── head: Head  (cmd_type, cmd, seq_no, msg_id, module, ...)
//!           └── data: bytes  (business payload, standard protobuf)
//!                 └── InboundMessagePush / SendC2CMessageReq / ...
//!
//! The conn layer (ConnMsg) is itself standard protobuf; over WebSocket each
//! frame == one ConnMsg protobuf bytes blob (no framing magic).
//!
//! Implementation: hand-written varint / protobuf wire-format codec, no
//! external protobuf dependency (matching the Python original).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================
// Constants
// ============================================================

/// conn-layer message type names mapped to their fully-qualified proto names.
pub const PB_MSG_TYPES: &[(&str, &str)] = &[
    ("ConnMsg", "trpc.yuanbao.conn_common.ConnMsg"),
    ("AuthBindReq", "trpc.yuanbao.conn_common.AuthBindReq"),
    ("AuthBindRsp", "trpc.yuanbao.conn_common.AuthBindRsp"),
    ("PingReq", "trpc.yuanbao.conn_common.PingReq"),
    ("PingRsp", "trpc.yuanbao.conn_common.PingRsp"),
    ("KickoutMsg", "trpc.yuanbao.conn_common.KickoutMsg"),
    ("DirectedPush", "trpc.yuanbao.conn_common.DirectedPush"),
    ("PushMsg", "trpc.yuanbao.conn_common.PushMsg"),
];

// cmd_type enum (ConnMsg.Head.cmd_type)
pub const CMD_TYPE_REQUEST: u64 = 0; // upstream request
pub const CMD_TYPE_RESPONSE: u64 = 1; // response to an upstream request
pub const CMD_TYPE_PUSH: u64 = 2; // downstream push
pub const CMD_TYPE_PUSH_ACK: u64 = 3; // ACK of a downstream push

// built-in command words
pub const CMD_AUTH_BIND: &str = "auth-bind";
pub const CMD_PING: &str = "ping";
pub const CMD_KICKOUT: &str = "kickout";
pub const CMD_UPDATE_META: &str = "update-meta";

// built-in module names
pub const MODULE_CONN_ACCESS: &str = "conn_access";

/// biz package short name (matches the TS client).
pub const BIZ_PKG: &str = "yuanbao_openclaw_proxy";

/// biz-layer service/method names mapped to their package-qualified form.
pub const BIZ_SERVICES: &[(&str, &str)] = &[
    ("InboundMessagePush", "yuanbao_openclaw_proxy.InboundMessagePush"),
    ("SendC2CMessageReq", "yuanbao_openclaw_proxy.SendC2CMessageReq"),
    ("SendC2CMessageRsp", "yuanbao_openclaw_proxy.SendC2CMessageRsp"),
    ("SendGroupMessageReq", "yuanbao_openclaw_proxy.SendGroupMessageReq"),
    ("SendGroupMessageRsp", "yuanbao_openclaw_proxy.SendGroupMessageRsp"),
    ("QueryGroupInfoReq", "yuanbao_openclaw_proxy.QueryGroupInfoReq"),
    ("QueryGroupInfoRsp", "yuanbao_openclaw_proxy.QueryGroupInfoRsp"),
    ("GetGroupMemberListReq", "yuanbao_openclaw_proxy.GetGroupMemberListReq"),
    ("GetGroupMemberListRsp", "yuanbao_openclaw_proxy.GetGroupMemberListRsp"),
    ("SendPrivateHeartbeatReq", "yuanbao_openclaw_proxy.SendPrivateHeartbeatReq"),
    ("SendPrivateHeartbeatRsp", "yuanbao_openclaw_proxy.SendPrivateHeartbeatRsp"),
    ("SendGroupHeartbeatReq", "yuanbao_openclaw_proxy.SendGroupHeartbeatReq"),
    ("SendGroupHeartbeatRsp", "yuanbao_openclaw_proxy.SendGroupHeartbeatRsp"),
];

/// openclaw instance_id (fixed value 17).
pub const HERMES_INSTANCE_ID: u32 = 17;

// Reply heartbeat status constants
pub const WS_HEARTBEAT_RUNNING: u64 = 1;
pub const WS_HEARTBEAT_FINISH: u64 = 2;

// ============================================================
// Sequence number generation
// ============================================================

static SEQ_COUNTER: AtomicU64 = AtomicU64::new(0);
const SEQ_MAX: u64 = u32::MAX as u64; // uint32 ceiling (2**32 - 1)

/// Generate an incrementing sequence number (thread-safe, wraps to 0 on overflow).
///
/// Returns the *current* value then increments, exactly like the Python original
/// which reads `_seq_counter` before bumping it.
pub fn next_seq_no() -> u64 {
    // fetch_update gives us the pre-update value while masking the stored next.
    SEQ_COUNTER
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |cur| {
            Some((cur + 1) & SEQ_MAX)
        })
        .unwrap()
}

// ============================================================
// Protobuf wire-format primitives (hand-written)
// ============================================================

pub const WT_VARINT: u64 = 0;
pub const WT_64BIT: u64 = 1;
pub const WT_LEN: u64 = 2;
pub const WT_32BIT: u64 = 5;

/// Encode a (possibly negative, interpreted as 64-bit two's complement) integer
/// as a protobuf varint.
pub fn encode_varint(value: i64) -> Vec<u8> {
    let mut v: u64 = value as u64; // two's-complement reinterpretation
    let mut out = Vec::new();
    loop {
        let bits = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            out.push(bits | 0x80);
        } else {
            out.push(bits);
            break;
        }
    }
    out
}

/// Encode an unsigned 64-bit integer as a protobuf varint.
pub fn encode_varint_u64(value: u64) -> Vec<u8> {
    let mut v = value;
    let mut out = Vec::new();
    loop {
        let bits = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            out.push(bits | 0x80);
        } else {
            out.push(bits);
            break;
        }
    }
    out
}

/// Decode a varint from `data[pos..]`, returning `(value, new_pos)`.
pub fn decode_varint(data: &[u8], mut pos: usize) -> Result<(u64, usize), String> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    while pos < data.len() {
        let b = data[pos];
        pos += 1;
        result |= ((b & 0x7F) as u64) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            return Ok((result, pos));
        }
        if shift >= 64 {
            return Err("varint too long".to_string());
        }
    }
    // Python returns whatever it accumulated when input is truncated.
    Ok((result, pos))
}

/// Encode a protobuf field tag (field number + wire type) followed by `value`.
pub fn encode_field(field_number: u64, wire_type: u64, value: &[u8]) -> Vec<u8> {
    let tag = (field_number << 3) | wire_type;
    let mut out = encode_varint_u64(tag);
    out.extend_from_slice(value);
    out
}

/// Encode the value portion of a length-prefixed UTF-8 string field.
pub fn encode_string(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut out = encode_varint_u64(bytes.len() as u64);
    out.extend_from_slice(bytes);
    out
}

/// Encode the value portion of a length-prefixed bytes field.
pub fn encode_bytes(b: &[u8]) -> Vec<u8> {
    let mut out = encode_varint_u64(b.len() as u64);
    out.extend_from_slice(b);
    out
}

/// Encode a nested message (length-prefixed). Identical to `encode_bytes`.
pub fn encode_message(b: &[u8]) -> Vec<u8> {
    encode_bytes(b)
}

/// A decoded protobuf field value.
#[derive(Debug, Clone, PartialEq)]
pub enum ProtoValue {
    Varint(u64),
    Bytes(Vec<u8>),
}

impl ProtoValue {
    pub fn as_varint(&self) -> Option<u64> {
        match self {
            ProtoValue::Varint(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            ProtoValue::Bytes(b) => Some(b),
            _ => None,
        }
    }
}

/// A single parsed field: (field_number, wire_type, value).
#[derive(Debug, Clone)]
pub struct ProtoField {
    pub field_number: u64,
    pub wire_type: u64,
    pub value: ProtoValue,
}

/// Parse all fields of a protobuf message.
pub fn parse_fields(data: &[u8]) -> Result<Vec<ProtoField>, String> {
    let mut fields = Vec::new();
    let mut pos = 0usize;
    let n = data.len();
    while pos < n {
        let (tag, np) = decode_varint(data, pos)?;
        pos = np;
        let field_number = tag >> 3;
        let wire_type = tag & 0x07;
        match wire_type {
            WT_VARINT => {
                let (val, np) = decode_varint(data, pos)?;
                pos = np;
                fields.push(ProtoField {
                    field_number,
                    wire_type,
                    value: ProtoValue::Varint(val),
                });
            }
            WT_LEN => {
                let (length, np) = decode_varint(data, pos)?;
                pos = np;
                let length = length as usize;
                let end = pos.saturating_add(length).min(data.len());
                let val = data[pos..end].to_vec();
                pos = end;
                fields.push(ProtoField {
                    field_number,
                    wire_type,
                    value: ProtoValue::Bytes(val),
                });
            }
            WT_64BIT => {
                let end = (pos + 8).min(data.len());
                let val = data[pos..end].to_vec();
                pos = end;
                fields.push(ProtoField {
                    field_number,
                    wire_type,
                    value: ProtoValue::Bytes(val),
                });
            }
            WT_32BIT => {
                let end = (pos + 4).min(data.len());
                let val = data[pos..end].to_vec();
                pos = end;
                fields.push(ProtoField {
                    field_number,
                    wire_type,
                    value: ProtoValue::Bytes(val),
                });
            }
            other => {
                return Err(format!("unknown wire type {} at pos {}", other, pos - 1));
            }
        }
    }
    Ok(fields)
}

/// Convert a parsed-field list into `{field_number: [(wire_type, value), ...]}`.
///
/// `BTreeMap` keeps deterministic ordering; repeated fields collect in encounter order.
pub fn fields_to_dict(fields: &[ProtoField]) -> BTreeMap<u64, Vec<(u64, ProtoValue)>> {
    let mut d: BTreeMap<u64, Vec<(u64, ProtoValue)>> = BTreeMap::new();
    for f in fields {
        d.entry(f.field_number)
            .or_default()
            .push((f.wire_type, f.value.clone()));
    }
    d
}

type FieldDict = BTreeMap<u64, Vec<(u64, ProtoValue)>>;

/// Take the first string-typed value for field `fn_`, or `default`.
pub fn get_string(fdict: &FieldDict, fn_: u64, default: &str) -> String {
    if let Some(entries) = fdict.get(&fn_) {
        if let Some((wt, val)) = entries.first() {
            if *wt == WT_LEN {
                if let ProtoValue::Bytes(b) = val {
                    return String::from_utf8_lossy(b).into_owned();
                }
            }
        }
    }
    default.to_string()
}

/// Take the first varint value for field `fn_`, or `default`.
pub fn get_varint(fdict: &FieldDict, fn_: u64, default: u64) -> u64 {
    if let Some(entries) = fdict.get(&fn_) {
        if let Some((wt, val)) = entries.first() {
            if *wt == WT_VARINT {
                if let ProtoValue::Varint(v) = val {
                    return *v;
                }
            }
        }
    }
    default
}

/// Take the first bytes/message value for field `fn_`, or empty.
pub fn get_bytes(fdict: &FieldDict, fn_: u64) -> Vec<u8> {
    if let Some(entries) = fdict.get(&fn_) {
        if let Some((wt, val)) = entries.first() {
            if *wt == WT_LEN {
                if let ProtoValue::Bytes(b) = val {
                    return b.clone();
                }
            }
        }
    }
    Vec::new()
}

/// Collect every repeated bytes/message value for field `fn_`.
pub fn get_repeated_bytes(fdict: &FieldDict, fn_: u64) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if let Some(entries) = fdict.get(&fn_) {
        for (wt, val) in entries {
            if *wt == WT_LEN {
                if let ProtoValue::Bytes(b) = val {
                    out.push(b.clone());
                }
            }
        }
    }
    out
}

// ============================================================
// ConnMsg layer codec
// ============================================================

/// Decoded ConnMsg.Head.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Head {
    pub cmd_type: u64,
    pub cmd: String,
    pub seq_no: u64,
    pub msg_id: String,
    pub module: String,
    pub need_ack: bool,
    pub status: u64,
}

#[allow(clippy::too_many_arguments)]
fn encode_head(
    cmd_type: u64,
    cmd: &str,
    seq_no: u64,
    msg_id: &str,
    module: &str,
    need_ack: bool,
    status: u64,
) -> Vec<u8> {
    let mut buf = Vec::new();
    if cmd_type != 0 {
        buf.extend(encode_field(1, WT_VARINT, &encode_varint_u64(cmd_type)));
    }
    if !cmd.is_empty() {
        buf.extend(encode_field(2, WT_LEN, &encode_string(cmd)));
    }
    if seq_no != 0 {
        buf.extend(encode_field(3, WT_VARINT, &encode_varint_u64(seq_no)));
    }
    if !msg_id.is_empty() {
        buf.extend(encode_field(4, WT_LEN, &encode_string(msg_id)));
    }
    if !module.is_empty() {
        buf.extend(encode_field(5, WT_LEN, &encode_string(module)));
    }
    if need_ack {
        buf.extend(encode_field(6, WT_VARINT, &encode_varint_u64(1)));
    }
    if status != 0 {
        buf.extend(encode_field(10, WT_VARINT, &encode_varint_u64(status)));
    }
    buf
}

fn decode_head(data: &[u8]) -> Head {
    let fields = parse_fields(data).unwrap_or_default();
    let fdict = fields_to_dict(&fields);
    Head {
        cmd_type: get_varint(&fdict, 1, 0),
        cmd: get_string(&fdict, 2, ""),
        seq_no: get_varint(&fdict, 3, 0),
        msg_id: get_string(&fdict, 4, ""),
        module: get_string(&fdict, 5, ""),
        need_ack: get_varint(&fdict, 6, 0) != 0,
        status: get_varint(&fdict, 10, 0),
    }
}

/// A decoded ConnMsg.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConnMsg {
    pub msg_type: u64,
    pub seq_no: u64,
    pub data: Vec<u8>,
    pub head: Head,
}

/// Encode a ConnMsg with the simplified signature (cmd_type, seq_no, data).
pub fn encode_conn_msg(msg_type: u64, seq_no: u64, data: &[u8]) -> Vec<u8> {
    let head_bytes = encode_head(msg_type, "", seq_no, "", "", false, 0);
    let mut buf = encode_field(1, WT_LEN, &encode_message(&head_bytes));
    if !data.is_empty() {
        buf.extend(encode_field(2, WT_LEN, &encode_bytes(data)));
    }
    buf
}

/// Decode a ConnMsg into `{msg_type, seq_no, data, head}`.
pub fn decode_conn_msg(data: &[u8]) -> ConnMsg {
    let fields = parse_fields(data).unwrap_or_default();
    let fdict = fields_to_dict(&fields);
    let head_bytes = get_bytes(&fdict, 1);
    let payload = get_bytes(&fdict, 2);
    let head = if !head_bytes.is_empty() {
        decode_head(&head_bytes)
    } else {
        Head::default()
    };
    ConnMsg {
        msg_type: head.cmd_type,
        seq_no: head.seq_no,
        data: payload,
        head,
    }
}

/// Encode a full ConnMsg, exposing all head fields.
#[allow(clippy::too_many_arguments)]
pub fn encode_conn_msg_full(
    cmd_type: u64,
    cmd: &str,
    seq_no: u64,
    msg_id: &str,
    module: &str,
    data: &[u8],
    need_ack: bool,
) -> Vec<u8> {
    let head_bytes = encode_head(cmd_type, cmd, seq_no, msg_id, module, need_ack, 0);
    let mut buf = encode_field(1, WT_LEN, &encode_message(&head_bytes));
    if !data.is_empty() {
        buf.extend(encode_field(2, WT_LEN, &encode_bytes(data)));
    }
    buf
}

// ============================================================
// BizMsg layer codec
// ============================================================

/// Decoded biz-layer view of a ConnMsg.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BizMsg {
    pub service: String,
    pub method: String,
    pub req_id: String,
    pub body: Vec<u8>,
    pub is_response: bool,
    pub head: Head,
}

/// Wrap a business payload into ConnMsg bytes.
pub fn encode_biz_msg(service: &str, method: &str, req_id: &str, body: &[u8]) -> Vec<u8> {
    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        method,
        next_seq_no(),
        req_id,
        service,
        body,
        false,
    )
}

/// Decode ConnMsg bytes into the biz-layer view.
pub fn decode_biz_msg(data: &[u8]) -> BizMsg {
    let result = decode_conn_msg(data);
    let head = result.head.clone();
    BizMsg {
        service: head.module.clone(),
        method: head.cmd.clone(),
        req_id: head.msg_id.clone(),
        body: result.data,
        is_response: head.cmd_type == CMD_TYPE_RESPONSE,
        head,
    }
}

// ============================================================
// Business protobuf message codec (biz payload)
// ============================================================

/// One image-info entry inside a MsgContent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageInfo {
    pub kind: u64, // field 1 "type"
    pub size: u64,
    pub width: u64,
    pub height: u64,
    pub url: String,
}

/// MsgContent body.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MsgContent {
    pub text: String,
    pub uuid: String,
    pub data: String,
    pub desc: String,
    pub ext: String,
    pub sound: String,
    pub url: String,
    pub file_name: String,
    pub image_format: u64,
    pub index: u64,
    pub file_size: u64,
    pub image_info_array: Vec<ImageInfo>,
}

fn encode_msg_content(content: &MsgContent) -> Vec<u8> {
    let mut buf = Vec::new();
    // string fields in proto field-number order
    for (fn_, v) in [
        (1u64, &content.text),
        (2, &content.uuid),
        (4, &content.data),
        (5, &content.desc),
        (6, &content.ext),
        (7, &content.sound),
        (10, &content.url),
        (12, &content.file_name),
    ] {
        if !v.is_empty() {
            buf.extend(encode_field(fn_, WT_LEN, &encode_string(v)));
        }
    }
    for (fn_, v) in [
        (3u64, content.image_format),
        (9, content.index),
        (11, content.file_size),
    ] {
        if v != 0 {
            buf.extend(encode_field(fn_, WT_VARINT, &encode_varint_u64(v)));
        }
    }
    for img in &content.image_info_array {
        let mut img_buf = Vec::new();
        for (ifn, iv) in [
            (1u64, img.kind),
            (2, img.size),
            (3, img.width),
            (4, img.height),
        ] {
            if iv != 0 {
                img_buf.extend(encode_field(ifn, WT_VARINT, &encode_varint_u64(iv)));
            }
        }
        if !img.url.is_empty() {
            img_buf.extend(encode_field(5, WT_LEN, &encode_string(&img.url)));
        }
        buf.extend(encode_field(8, WT_LEN, &encode_message(&img_buf)));
    }
    buf
}

fn decode_msg_content(data: &[u8]) -> MsgContent {
    let fields = parse_fields(data).unwrap_or_default();
    let fdict = fields_to_dict(&fields);
    let mut content = MsgContent {
        text: get_string(&fdict, 1, ""),
        uuid: get_string(&fdict, 2, ""),
        data: get_string(&fdict, 4, ""),
        desc: get_string(&fdict, 5, ""),
        ext: get_string(&fdict, 6, ""),
        sound: get_string(&fdict, 7, ""),
        url: get_string(&fdict, 10, ""),
        file_name: get_string(&fdict, 12, ""),
        image_format: get_varint(&fdict, 3, 0),
        index: get_varint(&fdict, 9, 0),
        file_size: get_varint(&fdict, 11, 0),
        image_info_array: Vec::new(),
    };
    for img_bytes in get_repeated_bytes(&fdict, 8) {
        let ifields = parse_fields(&img_bytes).unwrap_or_default();
        let ifdict = fields_to_dict(&ifields);
        let img = ImageInfo {
            kind: get_varint(&ifdict, 1, 0),
            size: get_varint(&ifdict, 2, 0),
            width: get_varint(&ifdict, 3, 0),
            height: get_varint(&ifdict, 4, 0),
            url: get_string(&ifdict, 5, ""),
        };
        // Python keeps the image only if it had any non-empty field.
        if img != ImageInfo::default() {
            content.image_info_array.push(img);
        }
    }
    content
}

/// One MsgBodyElement: a typed content element.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MsgBodyElement {
    pub msg_type: String, // e.g. "TIMTextElem"
    pub msg_content: MsgContent,
}

fn encode_msg_body_element(element: &MsgBodyElement) -> Vec<u8> {
    let mut buf = Vec::new();
    if !element.msg_type.is_empty() {
        buf.extend(encode_field(1, WT_LEN, &encode_string(&element.msg_type)));
    }
    // Python only encodes msg_content when the dict is truthy (non-empty).
    if element.msg_content != MsgContent::default() {
        let content_bytes = encode_msg_content(&element.msg_content);
        buf.extend(encode_field(2, WT_LEN, &encode_message(&content_bytes)));
    }
    buf
}

fn decode_msg_body_element(data: &[u8]) -> MsgBodyElement {
    let fields = parse_fields(data).unwrap_or_default();
    let fdict = fields_to_dict(&fields);
    let msg_type = get_string(&fdict, 1, "");
    let content_bytes = get_bytes(&fdict, 2);
    let msg_content = if !content_bytes.is_empty() {
        decode_msg_content(&content_bytes)
    } else {
        MsgContent::default()
    };
    MsgBodyElement {
        msg_type,
        msg_content,
    }
}

// ---------- LogInfoExt ----------

fn encode_log_ext(trace_id: &str) -> Vec<u8> {
    if trace_id.is_empty() {
        return Vec::new();
    }
    encode_field(1, WT_LEN, &encode_string(trace_id))
}

/// Decoded ImMsgSeq sub-message (recall list entry).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImMsgSeq {
    pub msg_seq: u64,
    pub msg_id: String,
}

fn decode_im_msg_seq(data: &[u8]) -> ImMsgSeq {
    let fields = parse_fields(data).unwrap_or_default();
    let fdict = fields_to_dict(&fields);
    ImMsgSeq {
        msg_seq: get_varint(&fdict, 1, 0),
        msg_id: get_string(&fdict, 2, ""),
    }
}

fn decode_log_ext(data: &[u8]) -> String {
    let fields = parse_fields(data).unwrap_or_default();
    let fdict = fields_to_dict(&fields);
    get_string(&fdict, 1, "")
}

// ============================================================
// Inbound message parsing
// ============================================================

/// Parsed InboundMessagePush.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InboundPush {
    pub callback_command: String,
    pub from_account: String,
    pub to_account: String,
    pub sender_nickname: String,
    pub group_id: String,
    pub group_code: String,
    pub group_name: String,
    pub msg_seq: u64,
    pub msg_random: u64,
    pub msg_time: u64,
    pub msg_key: String,
    pub msg_id: String,
    pub msg_body: Vec<MsgBodyElement>,
    pub cloud_custom_data: String,
    pub event_time: u64,
    pub bot_owner_id: String,
    pub recall_msg_seq_list: Option<Vec<ImMsgSeq>>,
    pub claw_msg_type: u64,
    pub private_from_group_code: String,
    pub trace_id: String,
}

/// Decode an InboundMessagePush biz payload. Returns `None` on parse failure.
///
/// Note: the Python version filters empty values from the returned dict (except
/// `msg_body` and `msg_seq`). The strongly-typed struct here keeps every field;
/// callers inspect emptiness directly. `recall_msg_seq_list` stays `None` when
/// there were no entries, matching the original.
pub fn decode_inbound_push(data: &[u8]) -> Option<InboundPush> {
    let fields = parse_fields(data).ok()?;
    let fdict = fields_to_dict(&fields);

    let mut msg_body = Vec::new();
    for el_bytes in get_repeated_bytes(&fdict, 13) {
        msg_body.push(decode_msg_body_element(&el_bytes));
    }

    let log_ext_bytes = get_bytes(&fdict, 20);
    let trace_id = if !log_ext_bytes.is_empty() {
        decode_log_ext(&log_ext_bytes)
    } else {
        String::new()
    };

    let recall_seq_raw = get_repeated_bytes(&fdict, 17);
    let recall_msg_seq_list = if recall_seq_raw.is_empty() {
        None
    } else {
        Some(recall_seq_raw.iter().map(|b| decode_im_msg_seq(b)).collect())
    };

    Some(InboundPush {
        callback_command: get_string(&fdict, 1, ""),
        from_account: get_string(&fdict, 2, ""),
        to_account: get_string(&fdict, 3, ""),
        sender_nickname: get_string(&fdict, 4, ""),
        group_id: get_string(&fdict, 5, ""),
        group_code: get_string(&fdict, 6, ""),
        group_name: get_string(&fdict, 7, ""),
        msg_seq: get_varint(&fdict, 8, 0),
        msg_random: get_varint(&fdict, 9, 0),
        msg_time: get_varint(&fdict, 10, 0),
        msg_key: get_string(&fdict, 11, ""),
        msg_id: get_string(&fdict, 12, ""),
        msg_body,
        cloud_custom_data: get_string(&fdict, 14, ""),
        event_time: get_varint(&fdict, 15, 0),
        bot_owner_id: get_string(&fdict, 16, ""),
        recall_msg_seq_list,
        claw_msg_type: get_varint(&fdict, 18, 0),
        private_from_group_code: get_string(&fdict, 19, ""),
        trace_id,
    })
}

// ============================================================
// Outbound message encoding
// ============================================================

#[allow(clippy::too_many_arguments)]
fn encode_send_c2c_req(
    to_account: &str,
    from_account: &str,
    msg_body: &[MsgBodyElement],
    msg_id: &str,
    msg_random: u64,
    msg_seq: Option<u64>,
    group_code: &str,
    trace_id: &str,
) -> Vec<u8> {
    let mut buf = Vec::new();
    if !msg_id.is_empty() {
        buf.extend(encode_field(1, WT_LEN, &encode_string(msg_id)));
    }
    buf.extend(encode_field(2, WT_LEN, &encode_string(to_account)));
    if !from_account.is_empty() {
        buf.extend(encode_field(3, WT_LEN, &encode_string(from_account)));
    }
    if msg_random != 0 {
        buf.extend(encode_field(4, WT_VARINT, &encode_varint_u64(msg_random)));
    }
    for el in msg_body {
        let el_bytes = encode_msg_body_element(el);
        buf.extend(encode_field(5, WT_LEN, &encode_message(&el_bytes)));
    }
    if !group_code.is_empty() {
        buf.extend(encode_field(6, WT_LEN, &encode_string(group_code)));
    }
    if let Some(seq) = msg_seq {
        buf.extend(encode_field(7, WT_VARINT, &encode_varint_u64(seq)));
    }
    if !trace_id.is_empty() {
        let log_bytes = encode_log_ext(trace_id);
        buf.extend(encode_field(8, WT_LEN, &encode_message(&log_bytes)));
    }
    buf
}

#[allow(clippy::too_many_arguments)]
fn encode_send_group_req(
    group_code: &str,
    from_account: &str,
    msg_body: &[MsgBodyElement],
    msg_id: &str,
    to_account: &str,
    random: &str,
    msg_seq: Option<u64>,
    ref_msg_id: &str,
    trace_id: &str,
) -> Vec<u8> {
    let mut buf = Vec::new();
    if !msg_id.is_empty() {
        buf.extend(encode_field(1, WT_LEN, &encode_string(msg_id)));
    }
    buf.extend(encode_field(2, WT_LEN, &encode_string(group_code)));
    if !from_account.is_empty() {
        buf.extend(encode_field(3, WT_LEN, &encode_string(from_account)));
    }
    if !to_account.is_empty() {
        buf.extend(encode_field(4, WT_LEN, &encode_string(to_account)));
    }
    if !random.is_empty() {
        buf.extend(encode_field(5, WT_LEN, &encode_string(random)));
    }
    for el in msg_body {
        let el_bytes = encode_msg_body_element(el);
        buf.extend(encode_field(6, WT_LEN, &encode_message(&el_bytes)));
    }
    if !ref_msg_id.is_empty() {
        buf.extend(encode_field(7, WT_LEN, &encode_string(ref_msg_id)));
    }
    if let Some(seq) = msg_seq {
        buf.extend(encode_field(8, WT_VARINT, &encode_varint_u64(seq)));
    }
    if !trace_id.is_empty() {
        let log_bytes = encode_log_ext(trace_id);
        buf.extend(encode_field(9, WT_LEN, &encode_message(&log_bytes)));
    }
    buf
}

/// Encode a C2C send-message request; returns full ConnMsg bytes.
#[allow(clippy::too_many_arguments)]
pub fn encode_send_c2c_message(
    to_account: &str,
    msg_body: &[MsgBodyElement],
    from_account: &str,
    msg_id: &str,
    msg_random: u64,
    msg_seq: Option<u64>,
    group_code: &str,
    trace_id: &str,
) -> Vec<u8> {
    let biz_bytes = encode_send_c2c_req(
        to_account,
        from_account,
        msg_body,
        msg_id,
        msg_random,
        msg_seq,
        group_code,
        trace_id,
    );
    let req_id = if msg_id.is_empty() {
        format!("c2c_{}", next_seq_no())
    } else {
        msg_id.to_string()
    };
    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        "send_c2c_message",
        next_seq_no(),
        &req_id,
        BIZ_PKG,
        &biz_bytes,
        false,
    )
}

/// Encode a group send-message request; returns full ConnMsg bytes.
#[allow(clippy::too_many_arguments)]
pub fn encode_send_group_message(
    group_code: &str,
    msg_body: &[MsgBodyElement],
    from_account: &str,
    msg_id: &str,
    to_account: &str,
    random: &str,
    msg_seq: Option<u64>,
    ref_msg_id: &str,
    trace_id: &str,
) -> Vec<u8> {
    let biz_bytes = encode_send_group_req(
        group_code,
        from_account,
        msg_body,
        msg_id,
        to_account,
        random,
        msg_seq,
        ref_msg_id,
        trace_id,
    );
    let req_id = if msg_id.is_empty() {
        format!("grp_{}", next_seq_no())
    } else {
        msg_id.to_string()
    };
    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        "send_group_message",
        next_seq_no(),
        &req_id,
        BIZ_PKG,
        &biz_bytes,
        false,
    )
}

// ============================================================
// AuthBind / Ping helpers
// ============================================================

/// Construct an auth-bind request ConnMsg bytes.
#[allow(clippy::too_many_arguments)]
pub fn encode_auth_bind(
    biz_id: &str,
    uid: &str,
    source: &str,
    token: &str,
    msg_id: &str,
    app_version: &str,
    operation_system: &str,
    bot_version: &str,
    route_env: &str,
) -> Vec<u8> {
    // AuthInfo
    let mut auth_buf = encode_field(1, WT_LEN, &encode_string(uid));
    auth_buf.extend(encode_field(2, WT_LEN, &encode_string(source)));
    auth_buf.extend(encode_field(3, WT_LEN, &encode_string(token)));

    // DeviceInfo
    let mut dev_buf = Vec::new();
    if !app_version.is_empty() {
        dev_buf.extend(encode_field(1, WT_LEN, &encode_string(app_version)));
    }
    if !operation_system.is_empty() {
        dev_buf.extend(encode_field(2, WT_LEN, &encode_string(operation_system)));
    }
    dev_buf.extend(encode_field(
        10,
        WT_LEN,
        &encode_string(&HERMES_INSTANCE_ID.to_string()),
    ));
    if !bot_version.is_empty() {
        dev_buf.extend(encode_field(24, WT_LEN, &encode_string(bot_version)));
    }

    let mut req_buf = encode_field(1, WT_LEN, &encode_string(biz_id));
    req_buf.extend(encode_field(2, WT_LEN, &encode_message(&auth_buf)));
    req_buf.extend(encode_field(3, WT_LEN, &encode_message(&dev_buf)));
    if !route_env.is_empty() {
        req_buf.extend(encode_field(5, WT_LEN, &encode_string(route_env)));
    }

    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        CMD_AUTH_BIND,
        next_seq_no(),
        msg_id,
        MODULE_CONN_ACCESS,
        &req_buf,
        false,
    )
}

/// Construct a ping request ConnMsg bytes (PingReq is an empty message).
pub fn encode_ping(msg_id: &str) -> Vec<u8> {
    encode_conn_msg_full(
        CMD_TYPE_REQUEST,
        CMD_PING,
        next_seq_no(),
        msg_id,
        MODULE_CONN_ACCESS,
        &[],
        false,
    )
}

/// Construct a push-ACK reply ConnMsg bytes, echoing the original head's cmd/msg_id/module.
pub fn encode_push_ack(original_head: &Head) -> Vec<u8> {
    encode_conn_msg_full(
        CMD_TYPE_PUSH_ACK,
        &original_head.cmd,
        next_seq_no(),
        &original_head.msg_id,
        &original_head.module,
        &[],
        false,
    )
}

// ============================================================
// Heartbeat encoding
// ============================================================

/// Encode a SendPrivateHeartbeatReq; returns full ConnMsg bytes.
pub fn encode_send_private_heartbeat(
    from_account: &str,
    to_account: &str,
    heartbeat: u64,
) -> Vec<u8> {
    let mut buf = encode_field(1, WT_LEN, &encode_string(from_account));
    buf.extend(encode_field(2, WT_LEN, &encode_string(to_account)));
    buf.extend(encode_field(3, WT_VARINT, &encode_varint_u64(heartbeat)));
    let req_id = format!("hb_priv_{}", next_seq_no());
    encode_biz_msg(BIZ_PKG, "send_private_heartbeat", &req_id, &buf)
}

/// Encode a SendGroupHeartbeatReq; returns full ConnMsg bytes.
///
/// When `send_time` is 0, the current epoch milliseconds are used.
pub fn encode_send_group_heartbeat(
    from_account: &str,
    group_code: &str,
    heartbeat: u64,
    send_time: u64,
) -> Vec<u8> {
    let ts = if send_time != 0 {
        send_time
    } else {
        now_millis()
    };
    let mut buf = encode_field(1, WT_LEN, &encode_string(from_account));
    buf.extend(encode_field(2, WT_LEN, &encode_string(""))); // to_account empty for group
    buf.extend(encode_field(3, WT_LEN, &encode_string(group_code)));
    buf.extend(encode_field(4, WT_VARINT, &encode_varint_u64(ts)));
    buf.extend(encode_field(5, WT_VARINT, &encode_varint_u64(heartbeat)));
    let req_id = format!("hb_grp_{}", next_seq_no());
    encode_biz_msg(BIZ_PKG, "send_group_heartbeat", &req_id, &buf)
}

fn now_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ============================================================
// Group info query
// ============================================================

/// Encode a QueryGroupInfoReq; returns full ConnMsg bytes.
pub fn encode_query_group_info(group_code: &str) -> Vec<u8> {
    let buf = encode_field(1, WT_LEN, &encode_string(group_code));
    let req_id = format!("qgi_{}", next_seq_no());
    encode_biz_msg(BIZ_PKG, "query_group_info", &req_id, &buf)
}

/// Decoded QueryGroupInfoRsp.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QueryGroupInfoRsp {
    pub code: u64,
    pub message: String,
    pub group_name: String,
    pub owner_id: String,
    pub owner_nickname: String,
    pub member_count: u64,
}

/// Decode a QueryGroupInfoRsp biz payload. Returns `None` on parse failure.
pub fn decode_query_group_info_rsp(data: &[u8]) -> Option<QueryGroupInfoRsp> {
    let fields = parse_fields(data).ok()?;
    let fdict = fields_to_dict(&fields);
    let code = get_varint(&fdict, 1, 0);
    let message = get_string(&fdict, 2, "");

    let mut result = QueryGroupInfoRsp {
        code,
        message,
        ..Default::default()
    };

    // field 3 = nested GroupInfo message
    let gi_bytes = fdict
        .get(&3)
        .and_then(|entries| entries.first())
        .and_then(|(_, v)| v.as_bytes())
        .map(|b| b.to_vec())
        .unwrap_or_default();
    if !gi_bytes.is_empty() {
        let gi_fields = parse_fields(&gi_bytes).unwrap_or_default();
        let gi = fields_to_dict(&gi_fields);
        result.group_name = get_string(&gi, 1, "");
        result.owner_id = get_string(&gi, 2, "");
        result.owner_nickname = get_string(&gi, 3, "");
        result.member_count = get_varint(&gi, 4, 0);
    }
    Some(result)
}

// ============================================================
// Group member-list query
// ============================================================

/// Encode a GetGroupMemberListReq; returns full ConnMsg bytes.
pub fn encode_get_group_member_list(group_code: &str, offset: u64, limit: u64) -> Vec<u8> {
    let mut buf = encode_field(1, WT_LEN, &encode_string(group_code));
    if offset != 0 {
        buf.extend(encode_field(2, WT_VARINT, &encode_varint_u64(offset)));
    }
    buf.extend(encode_field(3, WT_VARINT, &encode_varint_u64(limit)));
    let req_id = format!("gml_{}", next_seq_no());
    encode_biz_msg(BIZ_PKG, "get_group_member_list", &req_id, &buf)
}

/// One member entry inside a GetGroupMemberListRsp.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemberInfo {
    pub user_id: String,
    pub nickname: String,
    pub role: u64, // 0=member, 1=admin, 2=owner
    pub join_time: u64,
    pub name_card: String,
}

/// Decoded GetGroupMemberListRsp.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GetGroupMemberListRsp {
    pub code: u64,
    pub message: String,
    pub members: Vec<MemberInfo>,
    pub next_offset: u64,
    pub is_complete: bool,
}

/// Decode a GetGroupMemberListRsp biz payload. Returns `None` on parse failure.
pub fn decode_get_group_member_list_rsp(data: &[u8]) -> Option<GetGroupMemberListRsp> {
    let fields = parse_fields(data).ok()?;
    let fdict = fields_to_dict(&fields);
    let code = get_varint(&fdict, 1, 0);

    let mut members = Vec::new();
    for member_bytes in get_repeated_bytes(&fdict, 3) {
        let mfields = parse_fields(&member_bytes).unwrap_or_default();
        let mdict = fields_to_dict(&mfields);
        members.push(MemberInfo {
            user_id: get_string(&mdict, 1, ""),
            nickname: get_string(&mdict, 2, ""),
            role: get_varint(&mdict, 3, 0),
            join_time: get_varint(&mdict, 4, 0),
            name_card: get_string(&mdict, 5, ""),
        });
    }

    Some(GetGroupMemberListRsp {
        code,
        message: get_string(&fdict, 2, ""),
        members,
        next_offset: get_varint(&fdict, 4, 0),
        is_complete: get_varint(&fdict, 5, 0) != 0,
    })
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 300, 16384, u32::MAX as u64, u64::MAX] {
            let enc = encode_varint_u64(v);
            let (dec, pos) = decode_varint(&enc, 0).unwrap();
            assert_eq!(dec, v);
            assert_eq!(pos, enc.len());
        }
    }

    #[test]
    fn test_varint_known_encoding() {
        assert_eq!(encode_varint_u64(0), vec![0x00]);
        assert_eq!(encode_varint_u64(1), vec![0x01]);
        assert_eq!(encode_varint_u64(300), vec![0xAC, 0x02]);
    }

    #[test]
    fn test_negative_varint_twos_complement() {
        // -1 should become 10 bytes of 0xFF.. matching 64-bit two's complement.
        let enc = encode_varint(-1);
        let (dec, _) = decode_varint(&enc, 0).unwrap();
        assert_eq!(dec, u64::MAX);
    }

    #[test]
    fn test_conn_msg_roundtrip() {
        let payload = b"hello-payload";
        let encoded = encode_conn_msg(CMD_TYPE_REQUEST, 42, payload);
        let decoded = decode_conn_msg(&encoded);
        // CMD_TYPE_REQUEST is 0, so head.cmd_type omitted -> decodes to 0.
        assert_eq!(decoded.msg_type, 0);
        assert_eq!(decoded.seq_no, 42);
        assert_eq!(decoded.data, payload);
    }

    #[test]
    fn test_conn_msg_full_roundtrip() {
        let encoded = encode_conn_msg_full(
            CMD_TYPE_PUSH,
            "send_c2c_message",
            7,
            "req-123",
            BIZ_PKG,
            b"body",
            true,
        );
        let decoded = decode_conn_msg(&encoded);
        assert_eq!(decoded.msg_type, CMD_TYPE_PUSH);
        assert_eq!(decoded.seq_no, 7);
        assert_eq!(decoded.data, b"body");
        assert_eq!(decoded.head.cmd, "send_c2c_message");
        assert_eq!(decoded.head.msg_id, "req-123");
        assert_eq!(decoded.head.module, BIZ_PKG);
        assert!(decoded.head.need_ack);
    }

    #[test]
    fn test_biz_msg_roundtrip() {
        let body = b"biz-body";
        let encoded = encode_biz_msg(BIZ_PKG, "query_group_info", "qgi-1", body);
        let decoded = decode_biz_msg(&encoded);
        assert_eq!(decoded.service, BIZ_PKG);
        assert_eq!(decoded.method, "query_group_info");
        assert_eq!(decoded.req_id, "qgi-1");
        assert_eq!(decoded.body, body);
        assert!(!decoded.is_response); // cmd_type Request, not Response
    }

    #[test]
    fn test_msg_content_roundtrip() {
        let content = MsgContent {
            text: "hi there".to_string(),
            uuid: "u-1".to_string(),
            image_format: 2,
            image_info_array: vec![ImageInfo {
                kind: 1,
                size: 1024,
                width: 100,
                height: 200,
                url: "https://x/y.png".to_string(),
            }],
            ..Default::default()
        };
        let encoded = encode_msg_content(&content);
        let decoded = decode_msg_content(&encoded);
        assert_eq!(decoded, content);
    }

    #[test]
    fn test_msg_body_element_roundtrip() {
        let el = MsgBodyElement {
            msg_type: "TIMTextElem".to_string(),
            msg_content: MsgContent {
                text: "hello".to_string(),
                ..Default::default()
            },
        };
        let encoded = encode_msg_body_element(&el);
        let decoded = decode_msg_body_element(&encoded);
        assert_eq!(decoded, el);
    }

    #[test]
    fn test_send_c2c_decodes_as_inbound_fields() {
        // Build a C2C send and decode it back through the biz + inbound layers
        // to confirm field numbers line up where they overlap.
        let body = vec![MsgBodyElement {
            msg_type: "TIMTextElem".to_string(),
            msg_content: MsgContent {
                text: "ping".to_string(),
                ..Default::default()
            },
        }];
        let conn = encode_send_c2c_message(
            "to-acct", &body, "from-acct", "mid-1", 999, Some(5), "", "trace-x",
        );
        let biz = decode_biz_msg(&conn);
        assert_eq!(biz.method, "send_c2c_message");
        assert_eq!(biz.req_id, "mid-1");
        assert_eq!(biz.service, BIZ_PKG);

        // The biz body is a SendC2CMessageReq; parse field 2 (to_account) etc.
        let fields = parse_fields(&biz.body).unwrap();
        let fdict = fields_to_dict(&fields);
        assert_eq!(get_string(&fdict, 1, ""), "mid-1");
        assert_eq!(get_string(&fdict, 2, ""), "to-acct");
        assert_eq!(get_string(&fdict, 3, ""), "from-acct");
        assert_eq!(get_varint(&fdict, 4, 0), 999);
        assert_eq!(get_varint(&fdict, 7, 0), 5);
    }

    #[test]
    fn test_inbound_push_roundtrip() {
        // Manually build an InboundMessagePush payload.
        let el = MsgBodyElement {
            msg_type: "TIMTextElem".to_string(),
            msg_content: MsgContent {
                text: "hi".to_string(),
                ..Default::default()
            },
        };
        let el_bytes = encode_msg_body_element(&el);

        let mut buf = encode_field(1, WT_LEN, &encode_string("OnSendMsgCallback"));
        buf.extend(encode_field(2, WT_LEN, &encode_string("sender@x")));
        buf.extend(encode_field(4, WT_LEN, &encode_string("Nick")));
        buf.extend(encode_field(6, WT_LEN, &encode_string("grp-code"))); // group_code
        buf.extend(encode_field(8, WT_VARINT, &encode_varint_u64(123))); // msg_seq
        buf.extend(encode_field(12, WT_LEN, &encode_string("msg-id-9")));
        buf.extend(encode_field(13, WT_LEN, &encode_message(&el_bytes)));
        // log_ext (field 20) with trace_id
        let log = encode_log_ext("trace-42");
        buf.extend(encode_field(20, WT_LEN, &encode_message(&log)));

        let push = decode_inbound_push(&buf).unwrap();
        assert_eq!(push.callback_command, "OnSendMsgCallback");
        assert_eq!(push.from_account, "sender@x");
        assert_eq!(push.sender_nickname, "Nick");
        assert_eq!(push.group_code, "grp-code");
        assert_eq!(push.msg_seq, 123);
        assert_eq!(push.msg_id, "msg-id-9");
        assert_eq!(push.msg_body.len(), 1);
        assert_eq!(push.msg_body[0].msg_content.text, "hi");
        assert_eq!(push.trace_id, "trace-42");
        assert!(push.recall_msg_seq_list.is_none());
    }

    #[test]
    fn test_query_group_info_rsp() {
        // GroupInfo nested
        let mut gi = encode_field(1, WT_LEN, &encode_string("My Group"));
        gi.extend(encode_field(2, WT_LEN, &encode_string("owner-1")));
        gi.extend(encode_field(3, WT_LEN, &encode_string("OwnerNick")));
        gi.extend(encode_field(4, WT_VARINT, &encode_varint_u64(50)));

        let mut buf = encode_field(1, WT_VARINT, &encode_varint_u64(0));
        buf.extend(encode_field(2, WT_LEN, &encode_string("ok")));
        buf.extend(encode_field(3, WT_LEN, &encode_message(&gi)));

        let rsp = decode_query_group_info_rsp(&buf).unwrap();
        assert_eq!(rsp.code, 0);
        assert_eq!(rsp.message, "ok");
        assert_eq!(rsp.group_name, "My Group");
        assert_eq!(rsp.owner_id, "owner-1");
        assert_eq!(rsp.owner_nickname, "OwnerNick");
        assert_eq!(rsp.member_count, 50);
    }

    #[test]
    fn test_get_group_member_list_rsp() {
        let mut m1 = encode_field(1, WT_LEN, &encode_string("u1"));
        m1.extend(encode_field(2, WT_LEN, &encode_string("Alice")));
        m1.extend(encode_field(3, WT_VARINT, &encode_varint_u64(2)));

        let mut buf = encode_field(1, WT_VARINT, &encode_varint_u64(0));
        buf.extend(encode_field(2, WT_LEN, &encode_string("ok")));
        buf.extend(encode_field(3, WT_LEN, &encode_message(&m1)));
        buf.extend(encode_field(4, WT_VARINT, &encode_varint_u64(200)));
        buf.extend(encode_field(5, WT_VARINT, &encode_varint_u64(1)));

        let rsp = decode_get_group_member_list_rsp(&buf).unwrap();
        assert_eq!(rsp.code, 0);
        assert_eq!(rsp.message, "ok");
        assert_eq!(rsp.members.len(), 1);
        assert_eq!(rsp.members[0].user_id, "u1");
        assert_eq!(rsp.members[0].nickname, "Alice");
        assert_eq!(rsp.members[0].role, 2);
        assert_eq!(rsp.next_offset, 200);
        assert!(rsp.is_complete);
    }

    #[test]
    fn test_encode_auth_bind_decodes() {
        let conn = encode_auth_bind(
            "biz-99", "uid-1", "src-1", "tok-1", "auth-mid", "1.0.0", "linux", "bot-2", "prod",
        );
        let decoded = decode_conn_msg(&conn);
        assert_eq!(decoded.head.cmd, CMD_AUTH_BIND);
        assert_eq!(decoded.head.module, MODULE_CONN_ACCESS);
        assert_eq!(decoded.head.msg_id, "auth-mid");

        // Decode AuthBindReq body.
        let fields = parse_fields(&decoded.data).unwrap();
        let fdict = fields_to_dict(&fields);
        assert_eq!(get_string(&fdict, 1, ""), "biz-99");
        let auth = get_bytes(&fdict, 2);
        let auth_fields = parse_fields(&auth).unwrap();
        let adict = fields_to_dict(&auth_fields);
        assert_eq!(get_string(&adict, 1, ""), "uid-1");
        assert_eq!(get_string(&adict, 2, ""), "src-1");
        assert_eq!(get_string(&adict, 3, ""), "tok-1");

        let dev = get_bytes(&fdict, 3);
        let ddict = fields_to_dict(&parse_fields(&dev).unwrap());
        assert_eq!(get_string(&ddict, 10, ""), "17"); // HERMES_INSTANCE_ID
        assert_eq!(get_string(&fdict, 5, ""), "prod");
    }

    #[test]
    fn test_encode_ping_and_push_ack() {
        let ping = encode_ping("ping-1");
        let d = decode_conn_msg(&ping);
        assert_eq!(d.head.cmd, CMD_PING);
        assert_eq!(d.head.module, MODULE_CONN_ACCESS);
        assert!(d.data.is_empty());

        let head = Head {
            cmd: "some-cmd".to_string(),
            msg_id: "m-1".to_string(),
            module: "mod-1".to_string(),
            ..Default::default()
        };
        let ack = encode_push_ack(&head);
        let d2 = decode_conn_msg(&ack);
        assert_eq!(d2.msg_type, CMD_TYPE_PUSH_ACK);
        assert_eq!(d2.head.cmd, "some-cmd");
        assert_eq!(d2.head.msg_id, "m-1");
        assert_eq!(d2.head.module, "mod-1");
    }

    #[test]
    fn test_heartbeats_encode() {
        let priv_hb = encode_send_private_heartbeat("from", "to", WS_HEARTBEAT_RUNNING);
        let bp = decode_biz_msg(&priv_hb);
        assert_eq!(bp.method, "send_private_heartbeat");
        let pf = fields_to_dict(&parse_fields(&bp.body).unwrap());
        assert_eq!(get_string(&pf, 1, ""), "from");
        assert_eq!(get_string(&pf, 2, ""), "to");
        assert_eq!(get_varint(&pf, 3, 0), WS_HEARTBEAT_RUNNING);

        let grp_hb = encode_send_group_heartbeat("from", "grp", WS_HEARTBEAT_FINISH, 12345);
        let bg = decode_biz_msg(&grp_hb);
        assert_eq!(bg.method, "send_group_heartbeat");
        let gf = fields_to_dict(&parse_fields(&bg.body).unwrap());
        assert_eq!(get_string(&gf, 1, ""), "from");
        assert_eq!(get_string(&gf, 3, ""), "grp");
        assert_eq!(get_varint(&gf, 4, 0), 12345);
        assert_eq!(get_varint(&gf, 5, 0), WS_HEARTBEAT_FINISH);
    }

    #[test]
    fn test_next_seq_no_increments_and_wraps() {
        SEQ_COUNTER.store(SEQ_MAX, Ordering::SeqCst);
        let a = next_seq_no();
        let b = next_seq_no();
        assert_eq!(a, SEQ_MAX);
        assert_eq!(b, 0); // wrapped
    }
}
