//! Shared OpenRouter API client for Hermes tools.
//!
//! Native Rust port of `tools/openrouter_client.py`.
//!
//! Provides a single lazily-initialized OpenRouter-compatible client that all
//! tool modules can share. Routes through the centralized provider router
//! conventions (auth, headers, API format) handled in
//! [`crate::ag_auxiliary_client`] so behaviour stays consistent.
//!
//! Python semantics being preserved:
//!   * `get_async_client()` lazily constructs the client on first call and
//!     reuses it thereafter (process-global singleton).
//!   * The client is `None` (i.e. construction fails) when `OPENROUTER_API_KEY`
//!     is unset, in which case the Python raised `ValueError`. Here we return
//!     an `Err`.
//!   * `check_api_key()` returns whether `OPENROUTER_API_KEY` is present.

use std::sync::{Arc, OnceLock};

use crate::ag_auxiliary_client::{
    openrouter_headers_base, OPENROUTER_BASE_URL, OPENROUTER_MODEL,
};

/// Environment variable holding the OpenRouter API key.
pub const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";

/// Error returned when the shared client cannot be constructed.
///
/// Mirrors the `ValueError("OPENROUTER_API_KEY environment variable not set")`
/// raised by the Python `get_async_client`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenRouterClientError {
    /// `OPENROUTER_API_KEY` was not set (or empty).
    MissingApiKey,
}

impl std::fmt::Display for OpenRouterClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenRouterClientError::MissingApiKey => {
                write!(f, "OPENROUTER_API_KEY environment variable not set")
            }
        }
    }
}

impl std::error::Error for OpenRouterClientError {}

/// A shared OpenRouter-compatible client.
///
/// This is the Rust analogue of the `AsyncOpenAI` instance the Python module
/// caches. It captures the resolved auth (`api_key`), the base URL, the default
/// model, and the attribution headers that the centralized router would attach.
///
/// The fields are exposed so other tool modules can build concrete HTTP
/// requests (e.g. via `reqwest::blocking`) using consistent auth/headers.
#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    /// Resolved API key (already verified non-empty at construction).
    pub api_key: String,
    /// API base URL, e.g. `https://openrouter.ai/api/v1`.
    pub base_url: String,
    /// Default OpenRouter model for auxiliary calls.
    pub default_model: String,
    /// Attribution headers (HTTP-Referer / X-Title / categories).
    pub headers: Vec<(String, String)>,
}

impl OpenRouterClient {
    /// Construct a client from explicit parameters. The provided `base_url`
    /// and `default_model` fall back to the package defaults when empty.
    pub fn new(api_key: impl Into<String>, base_url: Option<&str>, default_model: Option<&str>) -> Self {
        let base_url = match base_url {
            Some(b) if !b.trim().is_empty() => b.trim().to_string(),
            _ => OPENROUTER_BASE_URL.to_string(),
        };
        let default_model = match default_model {
            Some(m) if !m.trim().is_empty() => m.trim().to_string(),
            _ => OPENROUTER_MODEL.to_string(),
        };
        let headers = openrouter_headers_base()
            .into_iter()
            .map(|(k, v)| (k, v))
            .collect();
        OpenRouterClient {
            api_key: api_key.into(),
            base_url,
            default_model,
            headers,
        }
    }

    /// Resolve a client from the environment. Returns
    /// [`OpenRouterClientError::MissingApiKey`] when `OPENROUTER_API_KEY` is
    /// unset or empty (the Rust analogue of the Python `client is None` ->
    /// `ValueError` branch).
    pub fn from_env() -> Result<Self, OpenRouterClientError> {
        match resolve_api_key() {
            Some(key) => Ok(OpenRouterClient::new(key, None, None)),
            None => Err(OpenRouterClientError::MissingApiKey),
        }
    }

    /// The `Authorization` header value (`Bearer <key>`).
    pub fn authorization_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }

    /// Full list of request headers including `Authorization`.
    pub fn request_headers(&self) -> Vec<(String, String)> {
        let mut h = self.headers.clone();
        h.push(("Authorization".to_string(), self.authorization_header()));
        h
    }
}

/// Read and trim the OpenRouter API key from the environment, returning `None`
/// when it is unset or blank.
fn resolve_api_key() -> Option<String> {
    match std::env::var(OPENROUTER_API_KEY_ENV) {
        Ok(v) if !v.trim().is_empty() => Some(v.trim().to_string()),
        _ => None,
    }
}

/// Process-global cache mirroring the Python module-level `_client`.
static SHARED_CLIENT: OnceLock<Arc<OpenRouterClient>> = OnceLock::new();

/// Return a shared OpenRouter-compatible client.
///
/// The client is created lazily on first successful call and reused thereafter
/// (matching the Python `get_async_client`). Returns
/// [`OpenRouterClientError::MissingApiKey`] when `OPENROUTER_API_KEY` is not
/// set.
///
/// Note: like the Python global, once a client is successfully cached it is
/// reused for the lifetime of the process even if the environment later
/// changes. Failed attempts are *not* cached, so a later call after the key is
/// set will succeed.
pub fn get_async_client() -> Result<Arc<OpenRouterClient>, OpenRouterClientError> {
    if let Some(existing) = SHARED_CLIENT.get() {
        return Ok(existing.clone());
    }
    let client = Arc::new(OpenRouterClient::from_env()?);
    // If two threads race, the first one to `set` wins; we then return the
    // stored value regardless to keep a single shared instance.
    let _ = SHARED_CLIENT.set(client);
    Ok(SHARED_CLIENT
        .get()
        .expect("SHARED_CLIENT set above")
        .clone())
}

/// Check whether the OpenRouter API key is present.
///
/// Mirrors Python `check_api_key()` (`bool(os.getenv("OPENROUTER_API_KEY"))`),
/// which is falsy for both unset and empty-string values.
pub fn check_api_key() -> bool {
    resolve_api_key().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize env-mutating tests since the process environment is global.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_key<F: FnOnce()>(value: Option<&str>, f: F) {
        let _g = ENV_LOCK.lock().unwrap();
        let prev = std::env::var(OPENROUTER_API_KEY_ENV).ok();
        unsafe {
            match value {
                Some(v) => std::env::set_var(OPENROUTER_API_KEY_ENV, v),
                None => std::env::remove_var(OPENROUTER_API_KEY_ENV),
            }
        }
        f();
        unsafe {
            match prev {
                Some(v) => std::env::set_var(OPENROUTER_API_KEY_ENV, v),
                None => std::env::remove_var(OPENROUTER_API_KEY_ENV),
            }
        }
    }

    #[test]
    fn check_api_key_unset_is_false() {
        with_key(None, || {
            assert!(!check_api_key());
        });
    }

    #[test]
    fn check_api_key_empty_is_false() {
        with_key(Some("   "), || {
            assert!(!check_api_key());
        });
    }

    #[test]
    fn check_api_key_present_is_true() {
        with_key(Some("sk-or-abc"), || {
            assert!(check_api_key());
        });
    }

    #[test]
    fn from_env_missing_key_errors() {
        with_key(None, || {
            let err = OpenRouterClient::from_env().unwrap_err();
            assert_eq!(err, OpenRouterClientError::MissingApiKey);
            assert_eq!(
                err.to_string(),
                "OPENROUTER_API_KEY environment variable not set"
            );
        });
    }

    #[test]
    fn from_env_present_key_builds_client() {
        with_key(Some("sk-or-xyz"), || {
            let c = OpenRouterClient::from_env().unwrap();
            assert_eq!(c.api_key, "sk-or-xyz");
            assert_eq!(c.base_url, OPENROUTER_BASE_URL);
            assert_eq!(c.default_model, OPENROUTER_MODEL);
            assert_eq!(c.authorization_header(), "Bearer sk-or-xyz");
        });
    }

    #[test]
    fn key_is_trimmed() {
        with_key(Some("  sk-or-trim  "), || {
            let c = OpenRouterClient::from_env().unwrap();
            assert_eq!(c.api_key, "sk-or-trim");
        });
    }

    #[test]
    fn request_headers_include_auth_and_attribution() {
        let c = OpenRouterClient::new("sk-test", None, None);
        let headers = c.request_headers();
        assert!(headers
            .iter()
            .any(|(k, v)| k == "Authorization" && v == "Bearer sk-test"));
        assert!(headers.iter().any(|(k, _)| k == "X-Title"));
        assert!(headers.iter().any(|(k, _)| k == "HTTP-Referer"));
    }

    #[test]
    fn new_overrides_base_and_model() {
        let c = OpenRouterClient::new(
            "k",
            Some("https://example.test/v1"),
            Some("custom/model"),
        );
        assert_eq!(c.base_url, "https://example.test/v1");
        assert_eq!(c.default_model, "custom/model");
    }

    #[test]
    fn get_async_client_caches() {
        // Best-effort: only meaningful if a key is set in the environment.
        with_key(Some("sk-or-cache"), || {
            if let Ok(a) = get_async_client() {
                let b = get_async_client().expect("second call");
                assert!(Arc::ptr_eq(&a, &b));
            }
        });
    }
}
