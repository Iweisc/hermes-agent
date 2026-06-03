//! Provider profile base type.
//!
//! A [`ProviderProfile`] declares everything about an inference provider in one
//! place: auth, endpoints, client quirks, request-time quirks. The transport
//! reads this instead of receiving 20+ boolean flags.
//!
//! Provider profiles are DECLARATIVE — they describe the provider's behavior.
//! They do NOT own client construction, credential rotation, or streaming.
//! Those stay on the agent.
//!
//! This is a native Rust port of `providers/base.py`. The Python module models
//! the profile as a `@dataclass` with overridable method hooks. In Rust we keep
//! the data on a plain owned struct and expose the hooks as methods that match
//! the default (base-class) behavior. Subclass-style customization is expressed
//! by callers overriding the relevant fields or wrapping the struct.

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::Value;

/// Temperature handling for a provider request.
///
/// Mirrors the Python tri-state: the dataclass field `fixed_temperature` can be
/// `None` (use caller default), the `OMIT_TEMPERATURE` sentinel (send nothing),
/// or a concrete float.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Temperature {
    /// `None` — use the caller's default temperature.
    Default,
    /// `OMIT_TEMPERATURE` sentinel — do not send a temperature at all
    /// (e.g. Kimi: the server manages it).
    Omit,
    /// A concrete fixed temperature value.
    Fixed(f64),
}

impl Default for Temperature {
    fn default() -> Self {
        Temperature::Default
    }
}

/// Declarative description of an inference provider.
///
/// Construct with [`ProviderProfile::new`] (which seeds the same defaults as the
/// Python dataclass) and override fields as needed, or build the struct directly.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderProfile {
    // ── Identity ─────────────────────────────────────────────
    pub name: String,
    pub api_mode: String,
    pub aliases: Vec<String>,

    // ── Human-readable metadata ───────────────────────────────
    /// e.g. "GMI Cloud" — shown in picker/labels.
    pub display_name: String,
    /// e.g. "GMI Cloud (multi-model direct API)" — picker subtitle.
    pub description: String,
    /// e.g. "https://www.gmicloud.ai/" — shown during setup.
    pub signup_url: String,

    // ── Auth & endpoints ─────────────────────────────────────
    pub env_vars: Vec<String>,
    pub base_url: String,
    /// Explicit models endpoint; falls back to `{base_url}/models`.
    pub models_url: String,
    /// `api_key` | `oauth_device_code` | `oauth_external` | `copilot` | `aws_sdk`.
    pub auth_type: String,

    // ── Model catalog ─────────────────────────────────────────
    /// Curated list shown in the `/model` picker when live fetch fails.
    /// Only agentic models that support tool calling should appear here.
    pub fallback_models: Vec<String>,

    /// Base hostname for URL→provider reverse-mapping. Derived from `base_url`
    /// when empty.
    pub hostname: String,

    // ── Client-level quirks (set once at client construction) ─
    pub default_headers: BTreeMap<String, String>,

    // ── Request-level quirks ─────────────────────────────────
    /// Temperature handling. See [`Temperature`].
    pub fixed_temperature: Temperature,
    pub default_max_tokens: Option<i64>,
    /// Cheap model for auxiliary tasks (compression, vision, etc.).
    /// Empty = use main model.
    pub default_aux_model: String,
}

impl Default for ProviderProfile {
    fn default() -> Self {
        Self::new("")
    }
}

impl ProviderProfile {
    /// Build a profile with the same field defaults as the Python dataclass.
    ///
    /// Only `name` is required; everything else mirrors the dataclass defaults.
    pub fn new(name: impl Into<String>) -> Self {
        ProviderProfile {
            name: name.into(),
            api_mode: "chat_completions".to_string(),
            aliases: Vec::new(),
            display_name: String::new(),
            description: String::new(),
            signup_url: String::new(),
            env_vars: Vec::new(),
            base_url: String::new(),
            models_url: String::new(),
            auth_type: "api_key".to_string(),
            fallback_models: Vec::new(),
            hostname: String::new(),
            default_headers: BTreeMap::new(),
            fixed_temperature: Temperature::Default,
            default_max_tokens: None,
            default_aux_model: String::new(),
        }
    }

    /// Return the provider's base hostname for URL-based detection.
    ///
    /// Uses `self.hostname` if set explicitly, otherwise derives it from
    /// `base_url`. e.g. `https://api.gmi-serving.com/v1` → `api.gmi-serving.com`.
    pub fn get_hostname(&self) -> String {
        if !self.hostname.is_empty() {
            return self.hostname.clone();
        }
        if !self.base_url.is_empty() {
            if let Ok(parsed) = url::Url::parse(&self.base_url) {
                return parsed.host_str().unwrap_or("").to_string();
            }
            return String::new();
        }
        String::new()
    }

    /// Provider-specific message preprocessing.
    ///
    /// Called AFTER codex field sanitization, BEFORE the developer role swap.
    /// Default: pass-through.
    pub fn prepare_messages(&self, messages: Vec<Value>) -> Vec<Value> {
        messages
    }

    /// Provider-specific `extra_body` fields.
    ///
    /// Merged into the API kwargs `extra_body`. Default: empty object.
    ///
    /// The Python signature accepts `session_id` and arbitrary `**context`;
    /// callers needing those should override. The default ignores them.
    pub fn build_extra_body(&self, _session_id: Option<&str>) -> serde_json::Map<String, Value> {
        serde_json::Map::new()
    }

    /// Provider-specific kwargs split between `extra_body` and top-level
    /// `api_kwargs`.
    ///
    /// Returns `(extra_body_additions, top_level_kwargs)`. The transport merges
    /// `extra_body_additions` into `extra_body`, and `top_level_kwargs` directly
    /// into `api_kwargs`.
    ///
    /// This split exists because some providers put reasoning config in
    /// `extra_body` (OpenRouter: `extra_body.reasoning`) while others put it as
    /// top-level `api_kwargs` (Kimi: `api_kwargs.reasoning_effort`).
    ///
    /// Default: `({}, {})`.
    pub fn build_api_kwargs_extras(
        &self,
        _reasoning_config: Option<&Value>,
    ) -> (serde_json::Map<String, Value>, serde_json::Map<String, Value>) {
        (serde_json::Map::new(), serde_json::Map::new())
    }

    /// Resolve the models endpoint URL using the Python resolution order:
    ///   1. `self.models_url` (explicit override)
    ///   2. `self.base_url + "/models"` (standard OpenAI-compat fallback)
    ///
    /// Returns `None` when neither is available (matches the early `return None`
    /// in `fetch_models`).
    pub fn models_endpoint(&self) -> Option<String> {
        let url = self.models_url.trim();
        if !url.is_empty() {
            return Some(url.to_string());
        }
        if self.base_url.is_empty() {
            return None;
        }
        Some(format!("{}/models", self.base_url.trim_end_matches('/')))
    }

    /// Fetch the live model list from the provider's models endpoint.
    ///
    /// Returns a list of model ID strings, or `None` if the fetch failed or the
    /// provider does not support live model listing.
    ///
    /// The default implementation sends Bearer auth when `api_key` is given and
    /// forwards `self.default_headers`.
    ///
    /// Callers must always fall back to the static fallback list when this
    /// returns `None`.
    pub fn fetch_models(&self, api_key: Option<&str>, timeout: Duration) -> Option<Vec<String>> {
        let url = self.models_endpoint()?;

        let client = match reqwest::blocking::Client::builder().timeout(timeout).build() {
            Ok(c) => c,
            Err(exc) => {
                log::debug!("fetch_models({}): {}", self.name, exc);
                return None;
            }
        };

        let mut req = client.get(&url).header("Accept", "application/json");
        if let Some(key) = api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }
        for (k, v) in &self.default_headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let result = (|| -> Result<Vec<String>, Box<dyn std::error::Error>> {
            let resp = req.send()?;
            let data: Value = resp.json()?;
            Ok(parse_model_ids(&data))
        })();

        match result {
            Ok(ids) => Some(ids),
            Err(exc) => {
                log::debug!("fetch_models({}): {}", self.name, exc);
                None
            }
        }
    }
}

/// Parse model IDs out of a `/models` response body.
///
/// Mirrors the Python:
/// `items = data if isinstance(data, list) else data.get("data", [])`
/// `return [m["id"] for m in items if isinstance(m, dict) and "id" in m]`
pub fn parse_model_ids(data: &Value) -> Vec<String> {
    let items: &[Value] = match data {
        Value::Array(arr) => arr.as_slice(),
        Value::Object(obj) => match obj.get("data") {
            Some(Value::Array(arr)) => arr.as_slice(),
            _ => &[],
        },
        _ => &[],
    };

    items
        .iter()
        .filter_map(|m| {
            m.as_object()
                .and_then(|obj| obj.get("id"))
                .and_then(|id| id.as_str())
                .map(|s| s.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_match_python_dataclass() {
        let p = ProviderProfile::new("gmi");
        assert_eq!(p.name, "gmi");
        assert_eq!(p.api_mode, "chat_completions");
        assert_eq!(p.auth_type, "api_key");
        assert!(p.aliases.is_empty());
        assert!(p.env_vars.is_empty());
        assert!(p.fallback_models.is_empty());
        assert_eq!(p.default_max_tokens, None);
        assert_eq!(p.fixed_temperature, Temperature::Default);
        assert!(p.default_aux_model.is_empty());
        assert!(p.default_headers.is_empty());
    }

    #[test]
    fn hostname_explicit_wins() {
        let mut p = ProviderProfile::new("x");
        p.hostname = "explicit.example.com".into();
        p.base_url = "https://other.example.com/v1".into();
        assert_eq!(p.get_hostname(), "explicit.example.com");
    }

    #[test]
    fn hostname_derived_from_base_url() {
        let mut p = ProviderProfile::new("gmi");
        p.base_url = "https://api.gmi-serving.com/v1".into();
        assert_eq!(p.get_hostname(), "api.gmi-serving.com");
    }

    #[test]
    fn hostname_empty_when_nothing_set() {
        let p = ProviderProfile::new("x");
        assert_eq!(p.get_hostname(), "");
    }

    #[test]
    fn models_endpoint_uses_explicit_override() {
        let mut p = ProviderProfile::new("openrouter");
        p.models_url = "https://openrouter.ai/api/v1/models".into();
        p.base_url = "https://openrouter.ai/api/v1".into();
        assert_eq!(
            p.models_endpoint().as_deref(),
            Some("https://openrouter.ai/api/v1/models")
        );
    }

    #[test]
    fn models_endpoint_falls_back_to_base_url() {
        let mut p = ProviderProfile::new("openai");
        p.base_url = "https://api.openai.com/v1/".into();
        // trailing slash stripped before appending /models
        assert_eq!(
            p.models_endpoint().as_deref(),
            Some("https://api.openai.com/v1/models")
        );
    }

    #[test]
    fn models_endpoint_none_without_urls() {
        let p = ProviderProfile::new("noendpoint");
        assert_eq!(p.models_endpoint(), None);
    }

    #[test]
    fn models_endpoint_blank_models_url_treated_as_empty() {
        let mut p = ProviderProfile::new("x");
        p.models_url = "   ".into();
        p.base_url = "https://h.example/v2".into();
        assert_eq!(
            p.models_endpoint().as_deref(),
            Some("https://h.example/v2/models")
        );
    }

    #[test]
    fn parse_model_ids_from_data_object() {
        let body = json!({
            "data": [
                {"id": "model-a", "object": "model"},
                {"id": "model-b"},
                {"object": "model"},          // no id → skipped
                "not-a-dict",                  // not an object → skipped
                {"id": 123}                    // non-string id → skipped
            ]
        });
        assert_eq!(parse_model_ids(&body), vec!["model-a", "model-b"]);
    }

    #[test]
    fn parse_model_ids_from_top_level_array() {
        let body = json!([
            {"id": "x"},
            {"id": "y"}
        ]);
        assert_eq!(parse_model_ids(&body), vec!["x", "y"]);
    }

    #[test]
    fn parse_model_ids_missing_data_key_is_empty() {
        let body = json!({"object": "list"});
        assert!(parse_model_ids(&body).is_empty());
    }

    #[test]
    fn parse_model_ids_scalar_body_is_empty() {
        assert!(parse_model_ids(&json!("nope")).is_empty());
        assert!(parse_model_ids(&json!(42)).is_empty());
    }

    #[test]
    fn hooks_default_passthrough_and_empty() {
        let p = ProviderProfile::new("x");
        let msgs = vec![json!({"role": "user", "content": "hi"})];
        assert_eq!(p.prepare_messages(msgs.clone()), msgs);
        assert!(p.build_extra_body(Some("sess")).is_empty());
        let (eb, top) = p.build_api_kwargs_extras(None);
        assert!(eb.is_empty());
        assert!(top.is_empty());
    }

    #[test]
    fn temperature_variants() {
        assert_eq!(Temperature::default(), Temperature::Default);
        assert_eq!(Temperature::Fixed(0.7), Temperature::Fixed(0.7));
        assert_ne!(Temperature::Omit, Temperature::Default);
    }
}
