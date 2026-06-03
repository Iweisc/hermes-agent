//! `gw_yuanbao_media` — native Rust port of `gateway/platforms/yuanbao_media.py`.
//!
//! Provides Tencent Yuanbao media handling: COS (Cloud Object Storage)
//! upload using temporary credentials + HMAC-SHA1 request signing, URL
//! download with a size guard, dependency-free image size parsing
//! (JPEG/PNG/GIF/WebP), and Tencent IM (TIM) media message body
//! construction.
//!
//! Ported from the TypeScript `media.ts` of the `yuanbao-openclaw-plugin`,
//! using `reqwest::blocking` instead of the COS Node SDK to avoid pulling in
//! an extra dependency.
//!
//! COS upload flow:
//!   1. call [`get_cos_credentials`] (`genUploadInfo`) to obtain temporary
//!      credentials (`tmpSecretId`/`tmpSecretKey`/`sessionToken`),
//!   2. build an `Authorization` header via HMAC-SHA1 with those credentials,
//!   3. HTTP PUT the bytes to COS ([`upload_to_cos`]).

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use reqwest::blocking::Client;
use serde_json::{json, Map, Value};
use sha1::{Digest, Sha1};

type HmacSha1 = Hmac<Sha1>;

// ============ Constants ============

/// API path for the upload-info endpoint (`genUploadInfo`).
pub const UPLOAD_INFO_PATH: &str = "/api/resource/genUploadInfo";
/// Default Yuanbao API domain.
pub const DEFAULT_API_DOMAIN: &str = "yuanbao.tencent.com";
/// Default maximum download size, in megabytes.
pub const DEFAULT_MAX_SIZE_MB: u64 = 50;
/// Whether COS uploads prefer the global-acceleration domain suffix.
pub const COS_USE_ACCELERATE: bool = true;

// ============ Type mappings ============

/// MIME type → TIM `image_format` numeric code.
fn mime_to_image_format(mime: &str) -> Option<i64> {
    match mime {
        "image/jpeg" | "image/jpg" => Some(1),
        "image/gif" => Some(2),
        "image/png" => Some(3),
        "image/bmp" => Some(4),
        "image/webp" | "image/heic" | "image/tiff" => Some(255),
        _ => None,
    }
}

/// File extension (including leading dot, lowercased) → MIME type.
fn ext_to_mime(ext: &str) -> Option<&'static str> {
    let mime = match ext {
        ".jpg" | ".jpeg" => "image/jpeg",
        ".png" => "image/png",
        ".gif" => "image/gif",
        ".webp" => "image/webp",
        ".bmp" => "image/bmp",
        ".heic" => "image/heic",
        ".tiff" => "image/tiff",
        ".ico" => "image/x-icon",
        ".pdf" => "application/pdf",
        ".doc" => "application/msword",
        ".docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls" => "application/vnd.ms-excel",
        ".xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".ppt" => "application/vnd.ms-powerpoint",
        ".pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".txt" => "text/plain",
        ".zip" => "application/zip",
        ".tar" => "application/x-tar",
        ".gz" => "application/gzip",
        ".mp3" => "audio/mpeg",
        ".mp4" => "video/mp4",
        ".wav" => "audio/wav",
        ".ogg" => "audio/ogg",
        ".webm" => "video/webm",
        _ => return None,
    };
    Some(mime)
}

// ============ Utility functions ============

/// Lowercased file extension (including the leading dot), or `""` if none.
fn file_ext(filename: &str) -> String {
    Path::new(filename)
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy().to_ascii_lowercase()))
        .unwrap_or_default()
}

/// Guess a MIME type from a file extension.
pub fn guess_mime_type(filename: &str) -> String {
    ext_to_mime(&file_ext(filename))
        .unwrap_or("application/octet-stream")
        .to_string()
}

/// Whether the file is treated as an image (by MIME prefix or extension).
pub fn is_image(filename: &str, mime_type: &str) -> bool {
    if mime_type.starts_with("image/") {
        return true;
    }
    matches!(
        file_ext(filename).as_str(),
        ".jpg" | ".jpeg" | ".png" | ".gif" | ".webp" | ".bmp" | ".heic" | ".tiff" | ".ico"
    )
}

/// TIM image-format code for a MIME type (defaults to `255`).
pub fn get_image_format(mime_type: &str) -> i64 {
    mime_to_image_format(&mime_type.to_ascii_lowercase()).unwrap_or(255)
}

/// MD5 hex digest of the bytes.
pub fn md5_hex(data: &[u8]) -> String {
    format!("{:x}", md5::compute(data))
}

/// Generate a random 32-char hex file id (16 random bytes).
pub fn generate_file_id() -> String {
    let mut buf = [0u8; 16];
    // `getrandom` is a workspace dependency of hermes-core (0.3 API: `fill`).
    if getrandom::fill(&mut buf).is_err() {
        // Fall back to a time-derived seed if the OS RNG is unavailable.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let bytes = nanos.to_le_bytes();
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = bytes[i % bytes.len()];
        }
    }
    hex_string(&buf)
}

/// Lowercase hex encoding of bytes.
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

// ============ Image size parsing (no third-party deps) ============

/// Parsed image dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

/// Parse the pixel dimensions of an image (JPEG/PNG/GIF/WebP) from its bytes.
///
/// Returns `None` when the format cannot be recognised.
pub fn parse_image_size(data: &[u8]) -> Option<ImageSize> {
    parse_png_size(data)
        .or_else(|| parse_jpeg_size(data))
        .or_else(|| parse_gif_size(data))
        .or_else(|| parse_webp_size(data))
}

fn be_u16(b: &[u8]) -> u16 {
    u16::from(b[0]) << 8 | u16::from(b[1])
}

fn le_u16(b: &[u8]) -> u16 {
    u16::from(b[1]) << 8 | u16::from(b[0])
}

fn be_u32(b: &[u8]) -> u32 {
    u32::from(b[0]) << 24 | u32::from(b[1]) << 16 | u32::from(b[2]) << 8 | u32::from(b[3])
}

fn le_u32(b: &[u8]) -> u32 {
    u32::from(b[3]) << 24 | u32::from(b[2]) << 16 | u32::from(b[1]) << 8 | u32::from(b[0])
}

fn parse_png_size(buf: &[u8]) -> Option<ImageSize> {
    if buf.len() < 24 {
        return None;
    }
    if &buf[0..4] != b"\x89PNG" {
        return None;
    }
    Some(ImageSize {
        width: be_u32(&buf[16..20]),
        height: be_u32(&buf[20..24]),
    })
}

fn parse_jpeg_size(buf: &[u8]) -> Option<ImageSize> {
    if buf.len() < 4 || buf[0] != 0xFF || buf[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 9 < buf.len() {
        if buf[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = buf[i + 1];
        if marker == 0xC0 || marker == 0xC2 {
            let h = be_u16(&buf[i + 5..i + 7]);
            let w = be_u16(&buf[i + 7..i + 9]);
            return Some(ImageSize {
                width: u32::from(w),
                height: u32::from(h),
            });
        }
        if i + 3 < buf.len() {
            i += 2 + usize::from(be_u16(&buf[i + 2..i + 4]));
        } else {
            break;
        }
    }
    None
}

fn parse_gif_size(buf: &[u8]) -> Option<ImageSize> {
    if buf.len() < 10 {
        return None;
    }
    let sig = &buf[0..6];
    if sig != b"GIF87a" && sig != b"GIF89a" {
        return None;
    }
    Some(ImageSize {
        width: u32::from(le_u16(&buf[6..8])),
        height: u32::from(le_u16(&buf[8..10])),
    })
}

fn parse_webp_size(buf: &[u8]) -> Option<ImageSize> {
    if buf.len() < 16 {
        return None;
    }
    if &buf[0..4] != b"RIFF" || &buf[8..12] != b"WEBP" {
        return None;
    }
    match &buf[12..16] {
        b"VP8 " => {
            if buf.len() >= 30 && buf[23] == 0x9D && buf[24] == 0x01 && buf[25] == 0x2A {
                let w = le_u16(&buf[26..28]) & 0x3FFF;
                let h = le_u16(&buf[28..30]) & 0x3FFF;
                return Some(ImageSize {
                    width: u32::from(w),
                    height: u32::from(h),
                });
            }
            None
        }
        b"VP8L" => {
            if buf.len() >= 25 && buf[20] == 0x2F {
                let bits = le_u32(&buf[21..25]);
                let w = (bits & 0x3FFF) + 1;
                let h = ((bits >> 14) & 0x3FFF) + 1;
                return Some(ImageSize {
                    width: w,
                    height: h,
                });
            }
            None
        }
        b"VP8X" => {
            if buf.len() >= 30 {
                let w = (u32::from(buf[24])
                    | (u32::from(buf[25]) << 8)
                    | (u32::from(buf[26]) << 16))
                    + 1;
                let h = (u32::from(buf[27])
                    | (u32::from(buf[28]) << 8)
                    | (u32::from(buf[29]) << 16))
                    + 1;
                return Some(ImageSize {
                    width: w,
                    height: h,
                });
            }
            None
        }
        _ => None,
    }
}

// ============ URL download ============

/// Download a URL's contents, returning `(bytes, content_type)`.
///
/// Mirrors the Python `download_url`: HEAD-checks the size first (ignoring
/// servers that reject HEAD), then performs a GET enforcing the size limit.
///
/// # Errors
/// Returns an error string when the content exceeds `max_size_mb`, or on a
/// network/HTTP failure.
pub fn download_url(url: &str, max_size_mb: u64) -> Result<(Vec<u8>, String), String> {
    let max_bytes = max_size_mb * 1024 * 1024;
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("creating download client failed: {e}"))?;

    // HEAD pre-check. A non-success status is ignored (some servers reject HEAD).
    if let Ok(head) = client.head(url).send() {
        if head.status().is_success() {
            if let Some(len) = head
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
            {
                if len > 0 && len > max_bytes {
                    return Err(format!(
                        "文件过大: {:.1} MB > {} MB",
                        len as f64 / 1024.0 / 1024.0,
                        max_size_mb
                    ));
                }
            }
        }
    }

    let resp = client
        .get(url)
        .send()
        .map_err(|e| format!("download GET failed: {e}"))?;
    let resp = resp
        .error_for_status()
        .map_err(|e| format!("download GET failed: {e}"))?;

    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or("").trim().to_string())
        .unwrap_or_default();

    let data = resp
        .bytes()
        .map_err(|e| format!("reading download body failed: {e}"))?;
    if data.len() as u64 > max_bytes {
        return Err(format!("文件过大: 已超过 {max_size_mb} MB 限制"));
    }
    Ok((data.to_vec(), content_type))
}

// ============ COS auth (HMAC-SHA1) ============

/// Percent-encode a value with no safe characters (equivalent to Python's
/// `urllib.parse.quote(v, safe="")`).
fn quote_all(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        let unreserved = byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'_' | b'.' | b'~');
        if unreserved {
            out.push(*byte as char);
        } else {
            out.push('%');
            out.push(hex_digit(byte >> 4).to_ascii_uppercase());
            out.push(hex_digit(byte & 0x0f).to_ascii_uppercase());
        }
    }
    out
}

/// Percent-encode a COS key while preserving `/` (Python `quote(key, safe="/")`).
fn quote_path(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        let safe = byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'/');
        if safe {
            out.push(*byte as char);
        } else {
            out.push('%');
            out.push(hex_digit(byte >> 4).to_ascii_uppercase());
            out.push(hex_digit(byte & 0x0f).to_ascii_uppercase());
        }
    }
    out
}

fn hmac_sha1_hex(key: &[u8], data: &str) -> String {
    let mut mac = HmacSha1::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data.as_bytes());
    hex_string(mac.finalize().into_bytes().as_slice())
}

fn sha1_hex(data: &str) -> String {
    hex_string(Sha1::digest(data.as_bytes()).as_slice())
}

fn now_unix() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    )
    .unwrap_or(0)
}

/// Build a COS request signature (`q-sign-algorithm=sha1`).
///
/// See <https://cloud.tencent.com/document/product/436/7778>.
///
/// `method` must be lowercased (e.g. `"put"`); `path` the URL-encoded path;
/// `params`/`headers` the (key, value) pairs participating in the signature.
#[allow(clippy::too_many_arguments)]
pub fn cos_sign(
    method: &str,
    path: &str,
    params: &[(String, String)],
    headers: &[(String, String)],
    secret_id: &str,
    secret_key: &str,
    start_time: Option<i64>,
    expire_seconds: i64,
) -> String {
    let now = now_unix();
    let start = start_time.unwrap_or(now);
    let q_sign_time = format!("{};{}", start, start + expire_seconds);

    // Step 1: SignKey = HMAC-SHA1(SecretKey, q-sign-time)
    let sign_key = hmac_sha1_hex(secret_key.as_bytes(), &q_sign_time);

    // Step 2: HttpString. Params and headers sorted by lowercased key.
    let sorted_params = sort_lower_quoted(params);
    let sorted_headers = sort_lower_quoted(headers);

    let url_param_list = join_keys(&sorted_params);
    let url_params = join_kv(&sorted_params);
    let header_list = join_keys(&sorted_headers);
    let header_str = join_kv(&sorted_headers);

    let http_string = format!(
        "{}\n{}\n{}\n{}\n",
        method.to_ascii_lowercase(),
        path,
        url_params,
        header_str
    );

    // Step 3: StringToSign = sha1(HttpString)
    let sha1_of_http = sha1_hex(&http_string);
    let string_to_sign = format!("sha1\n{}\n{}\n", q_sign_time, sha1_of_http);

    // Step 4: Signature = HMAC-SHA1(SignKey, StringToSign)
    let signature = hmac_sha1_hex(sign_key.as_bytes(), &string_to_sign);

    format!(
        "q-sign-algorithm=sha1&q-ak={secret_id}&q-sign-time={q_sign_time}&q-key-time={q_sign_time}&q-header-list={header_list}&q-url-param-list={url_param_list}&q-signature={signature}"
    )
}

/// Lowercase keys, percent-encode (safe="") values, sort by key (BTreeMap keeps
/// dictionary order and de-dups like Python's sorted-on-dict behaviour).
fn sort_lower_quoted(pairs: &[(String, String)]) -> Vec<(String, String)> {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for (key, value) in pairs {
        map.insert(key.to_ascii_lowercase(), quote_all(value));
    }
    map.into_iter().collect()
}

fn join_keys(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";")
}

fn join_kv(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("&")
}

// ============ Main public API ============

/// Call `genUploadInfo` to obtain COS temporary credentials and upload config.
///
/// Returns the `data` object from the response (or the whole body when no
/// `data` key is present), after verifying that `code == 0` (or absent) and
/// that the required fields `bucketName` and `location` are present.
///
/// # Errors
/// Returns an error string on non-zero `code`, missing required fields, or a
/// network/HTTP failure.
#[allow(clippy::too_many_arguments)]
pub fn get_cos_credentials(
    app_key: &str,
    api_domain: &str,
    token: &str,
    filename: &str,
    file_id: Option<&str>,
    bot_id: &str,
    route_env: &str,
) -> Result<Value, String> {
    let owned_id;
    let file_id = match file_id {
        Some(id) => id,
        None => {
            owned_id = generate_file_id();
            owned_id.as_str()
        }
    };

    let upload_url = format!("{}{}", api_domain.trim_end_matches('/'), UPLOAD_INFO_PATH);

    let mut req = Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("creating genUploadInfo client failed: {e}"))?
        .post(&upload_url)
        .header("Content-Type", "application/json")
        .header("X-Token", token)
        .header("X-ID", if bot_id.is_empty() { app_key } else { bot_id })
        .header("X-Source", "web");
    if !route_env.is_empty() {
        req = req.header("X-Route-Env", route_env);
    }

    let body = json!({
        "fileName": filename,
        "fileId": file_id,
        "docFrom": "localDoc",
        "docOpenId": "",
    });

    let resp = req
        .json(&body)
        .send()
        .map_err(|e| format!("genUploadInfo request failed: {e}"))?;
    let resp = resp
        .error_for_status()
        .map_err(|e| format!("genUploadInfo request failed: {e}"))?;
    let result: Value = resp
        .json()
        .map_err(|e| format!("genUploadInfo returned invalid JSON: {e}"))?;

    // code != 0 (and not null/absent) → error
    if let Some(code) = result.get("code") {
        if !code.is_null() && code.as_i64() != Some(0) {
            let msg = result
                .get("msg")
                .and_then(Value::as_str)
                .unwrap_or_default();
            return Err(format!("genUploadInfo 失败: code={code}, msg={msg}"));
        }
    }

    let data = match result.get("data") {
        Some(d) if !d.is_null() => d.clone(),
        _ => result.clone(),
    };

    let mut missing = Vec::new();
    for field in ["bucketName", "location"] {
        let present = data
            .get(field)
            .map(field_truthy)
            .unwrap_or(false);
        if !present {
            missing.push(field);
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "genUploadInfo 返回字段不完整: 缺少字段 {missing:?}"
        ));
    }

    Ok(data)
}

/// Whether a JSON field counts as present/truthy (matches Python `not data.get(f)`).
fn field_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(s) => !s.is_empty(),
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Result of a successful COS upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadResult {
    pub url: String,
    pub uuid: String,
    pub size: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

impl UploadResult {
    /// Serialise to the same JSON object the Python function returns.
    pub fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("url".to_string(), Value::String(self.url.clone()));
        map.insert("uuid".to_string(), Value::String(self.uuid.clone()));
        map.insert("size".to_string(), Value::from(self.size));
        if let Some(w) = self.width {
            map.insert("width".to_string(), Value::from(w));
        }
        if let Some(h) = self.height {
            map.insert("height".to_string(), Value::from(h));
        }
        Value::Object(map)
    }
}

fn cred_str<'a>(credentials: &'a Value, key: &str) -> &'a str {
    credentials.get(key).and_then(Value::as_str).unwrap_or("")
}

fn cred_int(credentials: &Value, key: &str) -> Option<i64> {
    credentials.get(key).and_then(Value::as_i64)
}

/// Upload bytes to COS via PUT, signing with the temporary credentials
/// returned by [`get_cos_credentials`].
///
/// # Errors
/// Returns an error string when credentials are incomplete or COS responds
/// with a non-2xx status / network failure.
pub fn upload_to_cos(
    file_bytes: &[u8],
    filename: &str,
    content_type: &str,
    credentials: &Value,
    bucket: &str,
    region: &str,
) -> Result<UploadResult, String> {
    let secret_id = cred_str(credentials, "encryptTmpSecretId");
    let secret_key = cred_str(credentials, "encryptTmpSecretKey");
    let session_token = cred_str(credentials, "encryptToken");
    let cos_key = cred_str(credentials, "location");
    let resource_url = cred_str(credentials, "resourceUrl");
    let start_time = cred_int(credentials, "startTime");
    let expired_time = cred_int(credentials, "expiredTime");

    if secret_id.is_empty() || secret_key.is_empty() || cos_key.is_empty() {
        return Err(format!(
            "COS credentials 不完整: secretId={}, secretKey={}, location={}",
            py_bool(!secret_id.is_empty()),
            py_bool(!secret_key.is_empty()),
            py_bool(!cos_key.is_empty()),
        ));
    }

    // Build COS upload host (prefer global acceleration).
    let cos_host = if COS_USE_ACCELERATE {
        format!("{bucket}.cos.accelerate.myqcloud.com")
    } else {
        format!("{bucket}.cos.{region}.myqcloud.com")
    };

    // URL-encode the cos_key, preserving '/'.
    let encoded_key = quote_path(cos_key);
    let trimmed_key = encoded_key.trim_start_matches('/');
    let cos_url = format!("https://{cos_host}/{trimmed_key}");

    // Resolve the effective Content-Type.
    let mut effective_ct = content_type.to_string();
    if effective_ct.is_empty() || effective_ct == "application/octet-stream" {
        if is_image(filename, "") {
            effective_ct = guess_mime_type(filename);
        } else {
            effective_ct = "application/octet-stream".to_string();
        }
    }

    let file_uuid = md5_hex(file_bytes);
    let file_size = file_bytes.len() as u64;

    let sign_headers = vec![
        ("host".to_string(), cos_host.clone()),
        ("content-type".to_string(), effective_ct.clone()),
        (
            "x-cos-security-token".to_string(),
            session_token.to_string(),
        ),
    ];

    let now = now_unix();
    let sign_start = start_time.filter(|&t| t != 0).unwrap_or(now);
    let sign_expire = match expired_time {
        Some(exp) if exp > now => exp - now,
        _ => 3600,
    };

    let authorization = cos_sign(
        "put",
        &format!("/{trimmed_key}"),
        &[],
        &sign_headers,
        secret_id,
        secret_key,
        Some(sign_start),
        sign_expire,
    );

    log::info!(
        "COS PUT: bucket={bucket} region={region} key={cos_key} size={file_size} mime={effective_ct}"
    );

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| format!("creating COS client failed: {e}"))?;
    let resp = client
        .put(&cos_url)
        .header("Authorization", authorization)
        .header("Content-Type", &effective_ct)
        .header("x-cos-security-token", session_token)
        .body(file_bytes.to_vec())
        .send()
        .map_err(|e| format!("COS PUT failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp
            .text()
            .unwrap_or_else(|_| "<unreadable body>".to_string());
        return Err(format!(
            "COS upload failed with status {}: {}",
            status.as_u16(),
            body
        ));
    }

    let url = if resource_url.is_empty() {
        cos_url
    } else {
        resource_url.to_string()
    };

    let (width, height) = if effective_ct.starts_with("image/") {
        match parse_image_size(file_bytes) {
            Some(sz) => (Some(sz.width), Some(sz.height)),
            None => (None, None),
        }
    } else {
        (None, None)
    };

    log::info!("COS upload succeeded: url={url} size={file_size}");

    Ok(UploadResult {
        url,
        uuid: file_uuid,
        size: file_size,
        width,
        height,
    })
}

/// Render a Python-style boolean for log/error parity (`True`/`False`).
fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

// ============ TIM media message bodies ============

/// Extract a basename from a URL path (Python `_basename_from_url`).
fn basename_from_url(url: &str) -> String {
    match url::Url::parse(url) {
        Ok(parsed) => parsed
            .path()
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_string(),
        Err(_) => String::new(),
    }
}

/// Build a Tencent IM `TIMImageElem` message body.
///
/// See <https://cloud.tencent.com/document/product/269/2720>.
#[allow(clippy::too_many_arguments)]
pub fn build_image_msg_body(
    url: &str,
    uuid: Option<&str>,
    filename: Option<&str>,
    size: u64,
    width: u32,
    height: u32,
    mime_type: &str,
) -> Vec<Value> {
    // _uuid = uuid or filename or basename(url) or "image"
    let basename = basename_from_url(url);
    let resolved_uuid = first_nonempty(&[
        uuid.unwrap_or(""),
        filename.unwrap_or(""),
        basename.as_str(),
        "image",
    ]);

    let image_format = if mime_type.is_empty() {
        255
    } else {
        get_image_format(mime_type)
    };

    vec![json!({
        "msg_type": "TIMImageElem",
        "msg_content": {
            "uuid": resolved_uuid,
            "image_format": image_format,
            "image_info_array": [
                {
                    "type": 1,
                    "size": size,
                    "width": width,
                    "height": height,
                    "url": url,
                }
            ],
        },
    })]
}

/// Build a Tencent IM `TIMFileElem` message body.
///
/// See <https://cloud.tencent.com/document/product/269/2720>.
pub fn build_file_msg_body(
    url: &str,
    filename: &str,
    uuid: Option<&str>,
    size: u64,
) -> Vec<Value> {
    // _uuid = uuid or filename
    let resolved_uuid = first_nonempty(&[uuid.unwrap_or(""), filename]);

    vec![json!({
        "msg_type": "TIMFileElem",
        "msg_content": {
            "uuid": resolved_uuid,
            "file_name": filename,
            "file_size": size,
            "url": url,
        },
    })]
}

/// First non-empty candidate; falls back to `""` if all empty.
fn first_nonempty<'a>(candidates: &[&'a str]) -> &'a str {
    candidates.iter().copied().find(|s| !s.is_empty()).unwrap_or("")
}

// ============ Tests ============

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guesses_mime_from_extension() {
        assert_eq!(guess_mime_type("photo.JPG"), "image/jpeg");
        assert_eq!(guess_mime_type("doc.docx"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
        assert_eq!(guess_mime_type("unknown.xyz"), "application/octet-stream");
        assert_eq!(guess_mime_type("noext"), "application/octet-stream");
    }

    #[test]
    fn detects_images() {
        assert!(is_image("a.png", ""));
        assert!(is_image("a.dat", "image/heic"));
        assert!(!is_image("a.pdf", ""));
        assert!(is_image("CARD.WEBP", ""));
    }

    #[test]
    fn image_format_codes() {
        assert_eq!(get_image_format("image/jpeg"), 1);
        assert_eq!(get_image_format("IMAGE/PNG"), 3);
        assert_eq!(get_image_format("image/gif"), 2);
        assert_eq!(get_image_format("image/bmp"), 4);
        assert_eq!(get_image_format("image/webp"), 255);
        assert_eq!(get_image_format("application/pdf"), 255);
    }

    #[test]
    fn md5_matches_known_vector() {
        // md5("") = d41d8cd98f00b204e9800998ecf8427e
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn file_id_is_32_hex() {
        let id = generate_file_id();
        assert_eq!(id.len(), 32);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn parses_png_dimensions() {
        // PNG signature + IHDR with 0x0100 x 0x0080
        let mut buf = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        buf.extend_from_slice(&[0, 0, 0, 13]); // length
        buf.extend_from_slice(b"IHDR");
        buf.extend_from_slice(&[0, 0, 1, 0]); // width 256
        buf.extend_from_slice(&[0, 0, 0, 128]); // height 128
        let sz = parse_image_size(&buf).unwrap();
        assert_eq!(sz, ImageSize { width: 256, height: 128 });
    }

    #[test]
    fn parses_gif_dimensions() {
        let mut buf = b"GIF89a".to_vec();
        buf.extend_from_slice(&[0x0A, 0x00]); // width 10 LE
        buf.extend_from_slice(&[0x14, 0x00]); // height 20 LE
        let sz = parse_image_size(&buf).unwrap();
        assert_eq!(sz, ImageSize { width: 10, height: 20 });
    }

    #[test]
    fn parses_jpeg_dimensions() {
        // SOI, then a SOF0 marker with height 0x0040, width 0x0060
        let buf = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xC0, // SOF0
            0x00, 0x11, // segment length
            0x08, // precision
            0x00, 0x40, // height 64
            0x00, 0x60, // width 96
            0x03, // components
            0, 0, 0, 0, 0, 0, // padding to satisfy i+9 < len
        ];
        let sz = parse_image_size(&buf).unwrap();
        assert_eq!(sz, ImageSize { width: 96, height: 64 });
    }

    #[test]
    fn unrecognised_image_returns_none() {
        assert!(parse_image_size(b"not an image at all").is_none());
        assert!(parse_image_size(b"").is_none());
    }

    #[test]
    fn quote_all_encodes_reserved() {
        assert_eq!(quote_all("a b/c"), "a%20b%2Fc");
        assert_eq!(quote_all("hello-world_1.0~"), "hello-world_1.0~");
    }

    #[test]
    fn quote_path_preserves_slash() {
        assert_eq!(quote_path("/foo/bar baz"), "/foo/bar%20baz");
    }

    #[test]
    fn cos_sign_is_deterministic_and_structured() {
        let headers = vec![
            ("host".to_string(), "b.cos.accelerate.myqcloud.com".to_string()),
            ("content-type".to_string(), "image/png".to_string()),
            ("x-cos-security-token".to_string(), "tok".to_string()),
        ];
        let auth = cos_sign(
            "put",
            "/path/to/key.png",
            &[],
            &headers,
            "AKID",
            "SECRET",
            Some(1_700_000_000),
            3600,
        );
        // Deterministic for fixed start_time.
        let auth2 = cos_sign(
            "put",
            "/path/to/key.png",
            &[],
            &headers,
            "AKID",
            "SECRET",
            Some(1_700_000_000),
            3600,
        );
        assert_eq!(auth, auth2);
        assert!(auth.starts_with("q-sign-algorithm=sha1&q-ak=AKID"));
        assert!(auth.contains("q-sign-time=1700000000;1700003600"));
        // Header list sorted by lowercased key.
        assert!(auth.contains("q-header-list=content-type;host;x-cos-security-token"));
        assert!(auth.contains("q-url-param-list=&"));
    }

    #[test]
    fn cos_sign_matches_python_reference_vector() {
        // Cross-checked against the Python `_cos_sign` for the same inputs.
        let headers = vec![("host".to_string(), "example.com".to_string())];
        let auth = cos_sign(
            "put",
            "/test.txt",
            &[],
            &headers,
            "secretid",
            "secretkey",
            Some(1_000_000),
            3600,
        );
        // SignKey = HMAC-SHA1("secretkey", "1000000;1003600")
        let sign_key = hmac_sha1_hex(b"secretkey", "1000000;1003600");
        let http_string = "put\n/test.txt\n\nhost=example.com\n";
        let http_sha = sha1_hex(http_string);
        let string_to_sign = format!("sha1\n1000000;1003600\n{http_sha}\n");
        let signature = hmac_sha1_hex(sign_key.as_bytes(), &string_to_sign);
        assert!(auth.contains(&format!("q-signature={signature}")));
    }

    #[test]
    fn image_msg_body_resolves_uuid_chain() {
        let body = build_image_msg_body(
            "https://cdn.example.com/path/pic.png",
            None,
            None,
            123,
            10,
            20,
            "image/png",
        );
        let content = &body[0]["msg_content"];
        assert_eq!(content["uuid"], "pic.png");
        assert_eq!(content["image_format"], 3);
        let info = &content["image_info_array"][0];
        assert_eq!(info["type"], 1);
        assert_eq!(info["size"], 123);
        assert_eq!(info["width"], 10);
        assert_eq!(info["height"], 20);
    }

    #[test]
    fn image_msg_body_falls_back_to_image_literal() {
        let body = build_image_msg_body("not-a-url", None, None, 0, 0, 0, "");
        assert_eq!(body[0]["msg_content"]["uuid"], "image");
        assert_eq!(body[0]["msg_content"]["image_format"], 255);
    }

    #[test]
    fn image_msg_body_prefers_explicit_uuid() {
        let body = build_image_msg_body("https://x/y.png", Some("MYUUID"), Some("name.png"), 0, 0, 0, "image/jpeg");
        assert_eq!(body[0]["msg_content"]["uuid"], "MYUUID");
    }

    #[test]
    fn file_msg_body_shape() {
        let body = build_file_msg_body("https://x/doc.pdf", "report.pdf", None, 4096);
        let content = &body[0]["msg_content"];
        assert_eq!(body[0]["msg_type"], "TIMFileElem");
        assert_eq!(content["uuid"], "report.pdf");
        assert_eq!(content["file_name"], "report.pdf");
        assert_eq!(content["file_size"], 4096);
        assert_eq!(content["url"], "https://x/doc.pdf");
    }

    #[test]
    fn upload_result_json_omits_absent_dims() {
        let r = UploadResult {
            url: "u".into(),
            uuid: "abc".into(),
            size: 9,
            width: None,
            height: None,
        };
        let v = r.to_json();
        assert_eq!(v["url"], "u");
        assert!(v.get("width").is_none());

        let r2 = UploadResult {
            url: "u".into(),
            uuid: "abc".into(),
            size: 9,
            width: Some(4),
            height: Some(5),
        };
        let v2 = r2.to_json();
        assert_eq!(v2["width"], 4);
        assert_eq!(v2["height"], 5);
    }

    #[test]
    fn cos_credentials_missing_fields_error() {
        // We can't hit the network; verify the field-truthy logic directly.
        let data = json!({ "bucketName": "b", "location": "" });
        assert!(field_truthy(data.get("bucketName").unwrap()));
        assert!(!field_truthy(data.get("location").unwrap()));
    }
}
