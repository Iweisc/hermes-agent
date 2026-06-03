//! QQBot package-level constants shared across adapter, onboard, and other modules.
//!
//! Faithful native Rust port of `gateway/platforms/qqbot/constants.py`.

// ---------------------------------------------------------------------------
// QQBot adapter version — bump on functional changes to the adapter package.
// ---------------------------------------------------------------------------

/// QQBot adapter version.
pub const QQBOT_VERSION: &str = "1.1.0";

// ---------------------------------------------------------------------------
// API endpoints
// ---------------------------------------------------------------------------

/// Default portal host (used when `QQ_PORTAL_HOST` is unset).
pub const DEFAULT_PORTAL_HOST: &str = "q.qq.com";

/// The portal domain is configurable via `QQ_PORTAL_HOST` for corporate proxies
/// or test environments. Default: `q.qq.com` (production).
///
/// In Python this is a module-level constant evaluated once at import time:
/// `PORTAL_HOST = os.getenv("QQ_PORTAL_HOST", "q.qq.com")`. Here we expose a
/// function so the environment is read at call time, which is the idiomatic
/// way to reproduce the behaviour without relying on lazy static init.
pub fn portal_host() -> String {
    std::env::var("QQ_PORTAL_HOST").unwrap_or_else(|_| DEFAULT_PORTAL_HOST.to_string())
}

pub const API_BASE: &str = "https://api.sgroup.qq.com";
pub const TOKEN_URL: &str = "https://bots.qq.com/app/getAppAccessToken";
pub const GATEWAY_URL_PATH: &str = "/gateway";

// QR-code onboard endpoints (on the portal host)
pub const ONBOARD_CREATE_PATH: &str = "/lite/create_bind_task";
pub const ONBOARD_POLL_PATH: &str = "/lite/poll_bind_result";

/// QR URL template. Mirrors the Python f-style template; use [`qr_url`] to
/// substitute the `task_id`.
pub const QR_URL_TEMPLATE: &str =
    "https://q.qq.com/qqbot/openclaw/connect.html?task_id={task_id}&_wv=2&source=hermes";

/// Build the QR onboard URL for the given `task_id`.
pub fn qr_url(task_id: &str) -> String {
    QR_URL_TEMPLATE.replace("{task_id}", task_id)
}

// ---------------------------------------------------------------------------
// Timeouts & retry
// ---------------------------------------------------------------------------

pub const DEFAULT_API_TIMEOUT: f64 = 30.0;
pub const FILE_UPLOAD_TIMEOUT: f64 = 120.0;
pub const CONNECT_TIMEOUT_SECONDS: f64 = 20.0;

/// Reconnect backoff schedule (seconds).
pub const RECONNECT_BACKOFF: [u64; 5] = [2, 5, 10, 30, 60];
pub const MAX_RECONNECT_ATTEMPTS: u32 = 100;
pub const RATE_LIMIT_DELAY: u64 = 60; // seconds
pub const QUICK_DISCONNECT_THRESHOLD: f64 = 5.0; // seconds
pub const MAX_QUICK_DISCONNECT_COUNT: u32 = 3;

pub const ONBOARD_POLL_INTERVAL: f64 = 2.0; // seconds between poll_bind_result calls
pub const ONBOARD_API_TIMEOUT: f64 = 10.0;

// ---------------------------------------------------------------------------
// Message limits
// ---------------------------------------------------------------------------

pub const MAX_MESSAGE_LENGTH: usize = 4000;
pub const DEDUP_WINDOW_SECONDS: u64 = 300;
pub const DEDUP_MAX_SIZE: usize = 1000;

// ---------------------------------------------------------------------------
// QQ Bot message types
// ---------------------------------------------------------------------------

pub const MSG_TYPE_TEXT: i64 = 0;
pub const MSG_TYPE_MARKDOWN: i64 = 2;
pub const MSG_TYPE_MEDIA: i64 = 7;
pub const MSG_TYPE_INPUT_NOTIFY: i64 = 6;

// ---------------------------------------------------------------------------
// QQ Bot file media types
// ---------------------------------------------------------------------------

pub const MEDIA_TYPE_IMAGE: i64 = 1;
pub const MEDIA_TYPE_VIDEO: i64 = 2;
pub const MEDIA_TYPE_VOICE: i64 = 3;
pub const MEDIA_TYPE_FILE: i64 = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(QQBOT_VERSION, "1.1.0");
    }

    #[test]
    fn test_portal_host_default() {
        // Ensure the env var is unset for the default branch.
        unsafe {
            std::env::remove_var("QQ_PORTAL_HOST");
        }
        assert_eq!(portal_host(), "q.qq.com");
    }

    #[test]
    fn test_portal_host_override() {
        unsafe {
            std::env::set_var("QQ_PORTAL_HOST", "proxy.example.com");
        }
        assert_eq!(portal_host(), "proxy.example.com");
        unsafe {
            std::env::remove_var("QQ_PORTAL_HOST");
        }
    }

    #[test]
    fn test_qr_url() {
        assert_eq!(
            qr_url("abc123"),
            "https://q.qq.com/qqbot/openclaw/connect.html?task_id=abc123&_wv=2&source=hermes"
        );
    }

    #[test]
    fn test_endpoints() {
        assert_eq!(API_BASE, "https://api.sgroup.qq.com");
        assert_eq!(TOKEN_URL, "https://bots.qq.com/app/getAppAccessToken");
        assert_eq!(GATEWAY_URL_PATH, "/gateway");
        assert_eq!(ONBOARD_CREATE_PATH, "/lite/create_bind_task");
        assert_eq!(ONBOARD_POLL_PATH, "/lite/poll_bind_result");
    }

    #[test]
    fn test_retry_and_limits() {
        assert_eq!(RECONNECT_BACKOFF, [2, 5, 10, 30, 60]);
        assert_eq!(MAX_RECONNECT_ATTEMPTS, 100);
        assert_eq!(RATE_LIMIT_DELAY, 60);
        assert_eq!(QUICK_DISCONNECT_THRESHOLD, 5.0);
        assert_eq!(MAX_QUICK_DISCONNECT_COUNT, 3);
        assert_eq!(MAX_MESSAGE_LENGTH, 4000);
        assert_eq!(DEDUP_WINDOW_SECONDS, 300);
        assert_eq!(DEDUP_MAX_SIZE, 1000);
    }

    #[test]
    fn test_message_and_media_types() {
        assert_eq!(MSG_TYPE_TEXT, 0);
        assert_eq!(MSG_TYPE_MARKDOWN, 2);
        assert_eq!(MSG_TYPE_MEDIA, 7);
        assert_eq!(MSG_TYPE_INPUT_NOTIFY, 6);
        assert_eq!(MEDIA_TYPE_IMAGE, 1);
        assert_eq!(MEDIA_TYPE_VIDEO, 2);
        assert_eq!(MEDIA_TYPE_VOICE, 3);
        assert_eq!(MEDIA_TYPE_FILE, 4);
    }
}
