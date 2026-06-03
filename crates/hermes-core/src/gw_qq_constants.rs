//! QQBot package-level constants shared across adapter, onboard, and other modules.
//!
//! Native Rust port of `gateway/platforms/qqbot/constants.py`.

use std::env;

// ---------------------------------------------------------------------------
// QQBot adapter version — bump on functional changes to the adapter package.
// ---------------------------------------------------------------------------

/// QQBot adapter version.
pub const QQBOT_VERSION: &str = "1.1.0";

// ---------------------------------------------------------------------------
// API endpoints
// ---------------------------------------------------------------------------

/// Default portal host when `QQ_PORTAL_HOST` is unset.
pub const DEFAULT_PORTAL_HOST: &str = "q.qq.com";

/// The portal domain, configurable via `QQ_PORTAL_HOST` for corporate proxies
/// or test environments. Default: `q.qq.com` (production).
///
/// This is resolved at call time (rather than at module load) so it reflects
/// the current process environment, matching the Python module which reads the
/// env var once at import. Callers that need import-time semantics should cache
/// the result themselves.
pub fn portal_host() -> String {
    env::var("QQ_PORTAL_HOST").unwrap_or_else(|_| DEFAULT_PORTAL_HOST.to_string())
}

/// Base URL for the QQ bot REST API.
pub const API_BASE: &str = "https://api.sgroup.qq.com";

/// URL used to obtain an app access token.
pub const TOKEN_URL: &str = "https://bots.qq.com/app/getAppAccessToken";

/// Path component appended to obtain the websocket gateway URL.
pub const GATEWAY_URL_PATH: &str = "/gateway";

// QR-code onboard endpoints (on the portal host)

/// Path to create a bind task during QR onboarding.
pub const ONBOARD_CREATE_PATH: &str = "/lite/create_bind_task";

/// Path to poll for bind results during QR onboarding.
pub const ONBOARD_POLL_PATH: &str = "/lite/poll_bind_result";

/// Format the QR connect URL for the given onboarding task id.
///
/// Mirrors the Python `QR_URL_TEMPLATE.format(task_id=...)`.
pub fn qr_url(task_id: &str) -> String {
    format!(
        "https://q.qq.com/qqbot/openclaw/connect.html?task_id={task_id}&_wv=2&source=hermes"
    )
}

// ---------------------------------------------------------------------------
// Timeouts & retry
// ---------------------------------------------------------------------------

/// Default REST API timeout, in seconds.
pub const DEFAULT_API_TIMEOUT: f64 = 30.0;

/// Timeout for file upload requests, in seconds.
pub const FILE_UPLOAD_TIMEOUT: f64 = 120.0;

/// Connection-establishment timeout, in seconds.
pub const CONNECT_TIMEOUT_SECONDS: f64 = 20.0;

/// Reconnect backoff schedule, in seconds.
pub const RECONNECT_BACKOFF: &[u64] = &[2, 5, 10, 30, 60];

/// Maximum number of reconnect attempts.
pub const MAX_RECONNECT_ATTEMPTS: u32 = 100;

/// Delay applied after hitting a rate limit, in seconds.
pub const RATE_LIMIT_DELAY: u64 = 60;

/// A disconnect occurring sooner than this (in seconds) counts as "quick".
pub const QUICK_DISCONNECT_THRESHOLD: f64 = 5.0;

/// Maximum number of quick disconnects tolerated.
pub const MAX_QUICK_DISCONNECT_COUNT: u32 = 3;

/// Interval between `poll_bind_result` calls, in seconds.
pub const ONBOARD_POLL_INTERVAL: f64 = 2.0;

/// Timeout for onboarding API calls, in seconds.
pub const ONBOARD_API_TIMEOUT: f64 = 10.0;

// ---------------------------------------------------------------------------
// Message limits
// ---------------------------------------------------------------------------

/// Maximum outbound message length, in characters.
pub const MAX_MESSAGE_LENGTH: usize = 4000;

/// Deduplication window, in seconds.
pub const DEDUP_WINDOW_SECONDS: u64 = 300;

/// Maximum number of entries retained in the dedup cache.
pub const DEDUP_MAX_SIZE: usize = 1000;

// ---------------------------------------------------------------------------
// QQ Bot message types
// ---------------------------------------------------------------------------

/// Plain-text message.
pub const MSG_TYPE_TEXT: i32 = 0;
/// Markdown message.
pub const MSG_TYPE_MARKDOWN: i32 = 2;
/// Media (rich) message.
pub const MSG_TYPE_MEDIA: i32 = 7;
/// Input-notify (typing indicator) message.
pub const MSG_TYPE_INPUT_NOTIFY: i32 = 6;

// ---------------------------------------------------------------------------
// QQ Bot file media types
// ---------------------------------------------------------------------------

/// Image media.
pub const MEDIA_TYPE_IMAGE: i32 = 1;
/// Video media.
pub const MEDIA_TYPE_VIDEO: i32 = 2;
/// Voice media.
pub const MEDIA_TYPE_VOICE: i32 = 3;
/// File media.
pub const MEDIA_TYPE_FILE: i32 = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches() {
        assert_eq!(QQBOT_VERSION, "1.1.0");
    }

    #[test]
    fn endpoints_match() {
        assert_eq!(API_BASE, "https://api.sgroup.qq.com");
        assert_eq!(TOKEN_URL, "https://bots.qq.com/app/getAppAccessToken");
        assert_eq!(GATEWAY_URL_PATH, "/gateway");
        assert_eq!(ONBOARD_CREATE_PATH, "/lite/create_bind_task");
        assert_eq!(ONBOARD_POLL_PATH, "/lite/poll_bind_result");
    }

    #[test]
    fn qr_url_formats_task_id() {
        assert_eq!(
            qr_url("abc123"),
            "https://q.qq.com/qqbot/openclaw/connect.html?task_id=abc123&_wv=2&source=hermes"
        );
    }

    #[test]
    fn portal_host_default_when_unset() {
        // Use a unique key removal to avoid clobbering parallel tests; the env
        // var is process-global, so we read whatever the current state is and
        // assert the default behaviour by removing it first.
        unsafe { env::remove_var("QQ_PORTAL_HOST"); }
        assert_eq!(portal_host(), "q.qq.com");
        assert_eq!(portal_host(), DEFAULT_PORTAL_HOST);
    }

    #[test]
    fn portal_host_reads_env() {
        unsafe { env::set_var("QQ_PORTAL_HOST", "proxy.example.com"); }
        assert_eq!(portal_host(), "proxy.example.com");
        unsafe { env::remove_var("QQ_PORTAL_HOST"); }
    }

    #[test]
    fn timeouts_match() {
        assert_eq!(DEFAULT_API_TIMEOUT, 30.0);
        assert_eq!(FILE_UPLOAD_TIMEOUT, 120.0);
        assert_eq!(CONNECT_TIMEOUT_SECONDS, 20.0);
        assert_eq!(ONBOARD_POLL_INTERVAL, 2.0);
        assert_eq!(ONBOARD_API_TIMEOUT, 10.0);
    }

    #[test]
    fn retry_constants_match() {
        assert_eq!(RECONNECT_BACKOFF, &[2, 5, 10, 30, 60]);
        assert_eq!(MAX_RECONNECT_ATTEMPTS, 100);
        assert_eq!(RATE_LIMIT_DELAY, 60);
        assert_eq!(QUICK_DISCONNECT_THRESHOLD, 5.0);
        assert_eq!(MAX_QUICK_DISCONNECT_COUNT, 3);
    }

    #[test]
    fn message_limits_match() {
        assert_eq!(MAX_MESSAGE_LENGTH, 4000);
        assert_eq!(DEDUP_WINDOW_SECONDS, 300);
        assert_eq!(DEDUP_MAX_SIZE, 1000);
    }

    #[test]
    fn message_types_match() {
        assert_eq!(MSG_TYPE_TEXT, 0);
        assert_eq!(MSG_TYPE_MARKDOWN, 2);
        assert_eq!(MSG_TYPE_MEDIA, 7);
        assert_eq!(MSG_TYPE_INPUT_NOTIFY, 6);
    }

    #[test]
    fn media_types_match() {
        assert_eq!(MEDIA_TYPE_IMAGE, 1);
        assert_eq!(MEDIA_TYPE_VIDEO, 2);
        assert_eq!(MEDIA_TYPE_VOICE, 3);
        assert_eq!(MEDIA_TYPE_FILE, 4);
    }
}
