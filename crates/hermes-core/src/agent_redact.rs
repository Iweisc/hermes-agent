//! Regex-based secret redaction for logs and tool output.
//!
//! Applies pattern matching to mask API keys, tokens, and credentials
//! before they reach log files, verbose output, or gateway logs.
//!
//! Short tokens (< 18 chars) are fully masked. Longer tokens preserve
//! the first 6 and last 4 characters for debuggability.
//!
//! Native Rust port of `agent/redact.py`. The `regex` crate does not support
//! lookaround, so the two patterns that relied on lookbehind/lookahead in the
//! Python source (`_PREFIX_RE` word-boundary guards and the phone-number
//! trailing guard) are implemented with manual boundary checks here.

use std::sync::OnceLock;

use regex::{Captures, Regex};

// ---------------------------------------------------------------------------
// Sensitive key name sets
// ---------------------------------------------------------------------------

/// Sensitive query-string parameter names (case-insensitive exact match).
/// Catches tokens whose values don't match any known vendor prefix regex
/// (e.g. opaque tokens, short OAuth codes).
const SENSITIVE_QUERY_PARAMS: &[&str] = &[
    "access_token",
    "refresh_token",
    "id_token",
    "token",
    "api_key",
    "apikey",
    "client_secret",
    "password",
    "auth",
    "jwt",
    "session",
    "secret",
    "key",
    "code",            // OAuth authorization codes
    "signature",       // pre-signed URL signatures
    "x-amz-signature",
];

fn is_sensitive_query_param(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    SENSITIVE_QUERY_PARAMS.iter().any(|&p| p == lower)
}

// ---------------------------------------------------------------------------
// Redaction enable flag (snapshotted once at first use)
// ---------------------------------------------------------------------------

/// Snapshot the `HERMES_REDACT_SECRETS` env var on first access so runtime env
/// mutations (e.g. an LLM-generated `export HERMES_REDACT_SECRETS=true`) cannot
/// enable/disable redaction mid-session. OFF by default — the user must opt in.
fn redact_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        let v = std::env::var("HERMES_REDACT_SECRETS")
            .unwrap_or_default()
            .to_ascii_lowercase();
        matches!(v.as_str(), "1" | "true" | "yes" | "on")
    })
}

// ---------------------------------------------------------------------------
// Known API key prefix patterns
// ---------------------------------------------------------------------------

/// Known API key prefixes -- match the prefix + contiguous token chars.
const PREFIX_PATTERNS: &[&str] = &[
    r"sk-[A-Za-z0-9_-]{10,}",          // OpenAI / OpenRouter / Anthropic (sk-ant-*)
    r"ghp_[A-Za-z0-9]{10,}",           // GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",   // GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",           // GitHub OAuth access token
    r"ghu_[A-Za-z0-9]{10,}",           // GitHub user-to-server token
    r"ghs_[A-Za-z0-9]{10,}",           // GitHub server-to-server token
    r"ghr_[A-Za-z0-9]{10,}",           // GitHub refresh token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",   // Slack tokens
    r"AIza[A-Za-z0-9_-]{30,}",         // Google API keys
    r"pplx-[A-Za-z0-9]{10,}",          // Perplexity
    r"fal_[A-Za-z0-9_-]{10,}",         // Fal.ai
    r"fc-[A-Za-z0-9]{10,}",            // Firecrawl
    r"bb_live_[A-Za-z0-9_-]{10,}",     // BrowserBase
    r"gAAAA[A-Za-z0-9_=-]{20,}",       // Codex encrypted tokens
    r"AKIA[A-Z0-9]{16}",               // AWS Access Key ID
    r"sk_live_[A-Za-z0-9]{10,}",       // Stripe secret key (live)
    r"sk_test_[A-Za-z0-9]{10,}",       // Stripe secret key (test)
    r"rk_live_[A-Za-z0-9]{10,}",       // Stripe restricted key
    r"SG\.[A-Za-z0-9_-]{10,}",         // SendGrid API key
    r"hf_[A-Za-z0-9]{10,}",            // HuggingFace token
    r"r8_[A-Za-z0-9]{10,}",            // Replicate API token
    r"npm_[A-Za-z0-9]{10,}",           // npm access token
    r"pypi-[A-Za-z0-9_-]{10,}",        // PyPI API token
    r"dop_v1_[A-Za-z0-9]{10,}",        // DigitalOcean PAT
    r"doo_v1_[A-Za-z0-9]{10,}",        // DigitalOcean OAuth
    r"am_[A-Za-z0-9_-]{10,}",          // AgentMail API key
    r"sk_[A-Za-z0-9_]{10,}",           // ElevenLabs TTS key (sk_ underscore, not sk- dash)
    r"tvly-[A-Za-z0-9]{10,}",          // Tavily search API key
    r"exa_[A-Za-z0-9]{10,}",           // Exa search API key
    r"gsk_[A-Za-z0-9]{10,}",           // Groq Cloud API key
    r"syt_[A-Za-z0-9]{10,}",           // Matrix access token
    r"retaindb_[A-Za-z0-9]{10,}",      // RetainDB API key
    r"hsk-[A-Za-z0-9]{10,}",           // Hindsight API key
    r"mem0_[A-Za-z0-9]{10,}",          // Mem0 Platform API key
    r"brv_[A-Za-z0-9]{10,}",           // ByteRover API key
];

// ---------------------------------------------------------------------------
// Compiled regexes (lazily initialised once)
// ---------------------------------------------------------------------------

struct Patterns {
    prefix: Regex,
    env_assign: Regex,
    json_field: Regex,
    auth_header: Regex,
    telegram: Regex,
    private_key: Regex,
    db_connstr: Regex,
    jwt: Regex,
    discord_mention: Regex,
    signal_phone: Regex,
    url_with_query: Regex,
    url_userinfo: Regex,
    form_body: Regex,
}

fn patterns() -> &'static Patterns {
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(|| {
        // Prefix alternation. The Python wraps this in lookbehind/lookahead
        // word-boundary guards; we drop them from the regex and enforce the
        // boundary manually in `redact_prefixes`.
        let prefix_alt = PREFIX_PATTERNS.join("|");
        let prefix = Regex::new(&format!("({prefix_alt})")).unwrap();

        // ENV assignment: KEY=value where KEY contains a secret-like name.
        //
        // The Python uses `(['"]?)(\S+)\2` — an optional opening quote with a
        // backreference forcing the closing quote to match. The `regex` crate
        // lacks backreferences, so we emulate it with an ordered alternation:
        // double-quoted | single-quoted | unquoted. Group indices:
        //   1 = name, 2 = double-quoted value, 3 = single-quoted value,
        //   4 = unquoted value. Exactly one of 2/3/4 matches per hit. This
        //   reproduces the original's match spans and captures exactly.
        let secret_env_names = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)";
        let env_assign = Regex::new(&format!(
            r#"([A-Z0-9_]{{0,50}}{secret_env_names}[A-Z0-9_]{{0,50}})\s*=\s*(?:"(\S+)"|'(\S+)'|(\S+))"#
        ))
        .unwrap();

        // JSON field: "apiKey": "value", "token": "value", etc. Case-insensitive.
        let json_key_names = r"(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|auth_token|bearer|secret_value|raw_secret|secret_input|key_material)";
        let json_field = Regex::new(&format!(
            r#"(?i)("{json_key_names}")\s*:\s*"([^"]+)""#
        ))
        .unwrap();

        let auth_header =
            Regex::new(r"(?i)(Authorization:\s*Bearer\s+)(\S+)").unwrap();

        // Telegram bot tokens: bot<digits>:<token> or <digits>:<token>.
        let telegram =
            Regex::new(r"(bot)?(\d{8,}):([-A-Za-z0-9_]{30,})").unwrap();

        // Private key blocks. `[\s\S]` (Python) == `(?s).` here.
        let private_key = Regex::new(
            r"(?s)-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----",
        )
        .unwrap();

        // Database connection strings: protocol://user:PASSWORD@host
        let db_connstr = Regex::new(
            r"(?i)((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:]+:)([^@]+)(@)",
        )
        .unwrap();

        // JWT tokens: header.payload[.signature] — always start with "eyJ".
        let jwt =
            Regex::new(r"eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}").unwrap();

        // Discord user/role mentions: <@123...> or <@!123...>.
        let discord_mention = Regex::new(r"<@!?(\d{17,20})>").unwrap();

        // E.164 phone numbers. The Python uses a `(?![A-Za-z0-9])` trailing
        // guard; we match the core and enforce the guard manually.
        let signal_phone = Regex::new(r"\+[1-9]\d{6,14}").unwrap();

        // URLs containing query strings: scheme://authority path ?query [#frag].
        let url_with_query = Regex::new(
            r"(https?|wss?|ftp)://([^\s/?#]+)([^\s?#]*)\?([^\s#]+)(#\S*)?",
        )
        .unwrap();

        // URLs containing userinfo: scheme://user:password@host (any scheme).
        let url_userinfo =
            Regex::new(r"(https?|wss?|ftp)://([^/\s:@]+):([^/\s@]+)@").unwrap();

        // Form-urlencoded body detection: entire text is k=v&k=v with no newlines.
        let form_body = Regex::new(
            r"^[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*)+$",
        )
        .unwrap();

        Patterns {
            prefix,
            env_assign,
            json_field,
            auth_header,
            telegram,
            private_key,
            db_connstr,
            jwt,
            discord_mention,
            signal_phone,
            url_with_query,
            url_userinfo,
            form_body,
        }
    })
}

// ---------------------------------------------------------------------------
// Masking helpers
// ---------------------------------------------------------------------------

/// Options for [`mask_secret`]. Mirrors the keyword args of the Python helper.
pub struct MaskOptions<'a> {
    /// Leading characters to preserve.
    pub head: usize,
    /// Trailing characters to preserve.
    pub tail: usize,
    /// Values shorter than this length are fully masked (returns `placeholder`).
    pub floor: usize,
    /// Value returned for too-short inputs.
    pub placeholder: &'a str,
    /// Value returned when `value` is empty.
    pub empty: &'a str,
}

impl Default for MaskOptions<'_> {
    fn default() -> Self {
        MaskOptions {
            head: 4,
            tail: 4,
            floor: 12,
            placeholder: "***",
            empty: "",
        }
    }
}

/// Mask a secret for display, preserving `head` and `tail` characters.
///
/// Canonical helper for display-time redaction across Hermes. Empty input
/// returns `opts.empty`; values shorter than `opts.floor` return
/// `opts.placeholder`; otherwise returns `<head>...<tail>`.
///
/// Length and slicing are byte-based to match Python's behaviour on ASCII
/// secrets (the only realistic input here). Use [`mask_secret_default`] for
/// the common case.
pub fn mask_secret(value: &str, opts: &MaskOptions) -> String {
    if value.is_empty() {
        return opts.empty.to_string();
    }
    let len = value.chars().count();
    if len < opts.floor {
        return opts.placeholder.to_string();
    }
    let head: String = value.chars().take(opts.head).collect();
    let tail: String = {
        let chars: Vec<char> = value.chars().collect();
        let start = chars.len().saturating_sub(opts.tail);
        chars[start..].iter().collect()
    };
    format!("{head}...{tail}")
}

/// [`mask_secret`] with the default options (head=4, tail=4, floor=12).
pub fn mask_secret_default(value: &str) -> String {
    mask_secret(value, &MaskOptions::default())
}

/// Mask a log token — conservative 18-char floor, preserves 6 prefix / 4 suffix.
///
/// Empty input returns `"***"` (historical behaviour, distinct from
/// [`mask_secret`] which returns the empty default).
fn mask_token(token: &str) -> String {
    if token.is_empty() {
        return "***".to_string();
    }
    mask_secret(
        token,
        &MaskOptions {
            head: 6,
            tail: 4,
            floor: 18,
            placeholder: "***",
            empty: "",
        },
    )
}

// ---------------------------------------------------------------------------
// Query / URL / form helpers
// ---------------------------------------------------------------------------

/// Redact sensitive parameter values in a URL query string (`k=v&k=v`).
fn redact_query_string(query: &str) -> String {
    if query.is_empty() {
        return query.to_string();
    }
    let parts: Vec<String> = query
        .split('&')
        .map(|pair| {
            if !pair.contains('=') {
                return pair.to_string();
            }
            // Python's str.partition splits on the FIRST '='.
            let (key, value) = pair.split_once('=').unwrap();
            if is_sensitive_query_param(key) {
                let _ = value;
                format!("{key}=***")
            } else {
                pair.to_string()
            }
        })
        .collect();
    parts.join("&")
}

/// Scan text for URLs with query strings and redact sensitive params.
fn redact_url_query_params(text: &str) -> String {
    let p = patterns();
    p.url_with_query
        .replace_all(text, |m: &Captures| {
            let scheme = &m[1];
            let authority = &m[2];
            let path = &m[3];
            let query = redact_query_string(&m[4]);
            let fragment = m.get(5).map(|x| x.as_str()).unwrap_or("");
            format!("{scheme}://{authority}{path}?{query}{fragment}")
        })
        .into_owned()
}

/// Strip `user:password@` from HTTP/WS/FTP URLs (DB schemes handled elsewhere).
fn redact_url_userinfo(text: &str) -> String {
    let p = patterns();
    p.url_userinfo
        .replace_all(text, |m: &Captures| {
            format!("{}://{}:***@", &m[1], &m[2])
        })
        .into_owned()
}

/// Redact sensitive values in a form-urlencoded body. Only applies when the
/// entire input looks like a pure form body (k=v&k=v, no newlines, no other
/// text).
fn redact_form_body(text: &str) -> String {
    if text.is_empty() || text.contains('\n') || !text.contains('&') {
        return text.to_string();
    }
    let trimmed = text.trim();
    let p = patterns();
    if !p.form_body.is_match(trimmed) {
        return text.to_string();
    }
    redact_query_string(trimmed)
}

// ---------------------------------------------------------------------------
// Prefix + phone helpers (manual boundary handling)
// ---------------------------------------------------------------------------

fn is_prefix_boundary_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Apply the known-prefix patterns. Mirrors the Python `_PREFIX_RE`, including
/// its `(?<![A-Za-z0-9_-])(...)(?![A-Za-z0-9_-])` word-boundary guards which we
/// enforce manually (the `regex` crate lacks lookaround).
fn redact_prefixes(text: &str) -> String {
    let p = patterns();
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut last = 0usize;
    for m in p.prefix.find_iter(text) {
        let start = m.start();
        let end = m.end();

        // Lookbehind: char immediately before the match must not be a
        // boundary char.
        let preceded_ok = if start == 0 {
            true
        } else {
            let prev = text[..start].chars().next_back().unwrap();
            !is_prefix_boundary_char(prev)
        };
        // Lookahead: char immediately after the match must not be a boundary
        // char.
        let followed_ok = if end >= bytes.len() {
            true
        } else {
            let next = text[end..].chars().next().unwrap();
            !is_prefix_boundary_char(next)
        };

        if preceded_ok && followed_ok {
            out.push_str(&text[last..start]);
            out.push_str(&mask_token(m.as_str()));
            last = end;
        }
        // If guards fail, leave the span untouched; `last` stays put so the
        // text is copied verbatim on the next accepted match (or at the end).
    }
    out.push_str(&text[last..]);
    out
}

/// Apply the E.164 phone-number redaction with the manual `(?![A-Za-z0-9])`
/// trailing guard.
fn redact_phones(text: &str) -> String {
    let p = patterns();
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for m in p.signal_phone.find_iter(text) {
        let start = m.start();
        let end = m.end();
        // Trailing guard: next char must not be ASCII alphanumeric.
        let followed_ok = if end >= bytes.len() {
            true
        } else {
            let next = text[end..].chars().next().unwrap();
            !next.is_ascii_alphanumeric()
        };
        if followed_ok {
            out.push_str(&text[last..start]);
            out.push_str(&mask_phone(m.as_str()));
            last = end;
        }
    }
    out.push_str(&text[last..]);
    out
}

fn mask_phone(phone: &str) -> String {
    // Phone is ASCII; byte length == char length.
    if phone.len() <= 8 {
        format!("{}****{}", &phone[..2], &phone[phone.len() - 2..])
    } else {
        format!("{}****{}", &phone[..4], &phone[phone.len() - 4..])
    }
}

// ---------------------------------------------------------------------------
// Top-level redaction
// ---------------------------------------------------------------------------

/// Apply all redaction patterns to a block of text.
///
/// Safe to call on any string — non-matching text passes through unchanged.
/// Disabled by default; enable via `HERMES_REDACT_SECRETS=true` (snapshotted at
/// first use). Set `force = true` for safety boundaries that must never return
/// raw secrets regardless of the global preference.
///
/// Set `code_file = true` to skip the ENV-assignment and JSON-field patterns
/// when the text is known to be source code (e.g. `MAX_TOKENS=...` constants,
/// `"apiKey": "test"` fixtures). All other patterns still apply.
pub fn redact_sensitive_text(text: &str, force: bool, code_file: bool) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    if !(force || redact_enabled()) {
        return text.to_string();
    }

    let p = patterns();
    let mut text = text.to_string();

    // Known prefixes (sk-, ghp_, etc.) — manual word-boundary handling.
    text = redact_prefixes(&text);

    if !code_file {
        // ENV assignments: OPENAI_API_KEY=***
        text = p
            .env_assign
            .replace_all(&text, |m: &Captures| {
                let name = &m[1];
                // Determine which value alternative matched and the surrounding
                // quote (preserved verbatim like the Python `\2` group).
                let (quote, value) = if let Some(v) = m.get(2) {
                    ("\"", v.as_str())
                } else if let Some(v) = m.get(3) {
                    ("'", v.as_str())
                } else {
                    ("", &m[4])
                };
                format!("{name}={quote}{}{quote}", mask_token(value))
            })
            .into_owned();

        // JSON fields: "apiKey": "***"
        text = p
            .json_field
            .replace_all(&text, |m: &Captures| {
                let key = &m[1];
                let value = &m[2];
                format!("{key}: \"{}\"", mask_token(value))
            })
            .into_owned();
    }

    // Authorization headers.
    text = p
        .auth_header
        .replace_all(&text, |m: &Captures| {
            format!("{}{}", &m[1], mask_token(&m[2]))
        })
        .into_owned();

    // Telegram bot tokens.
    text = p
        .telegram
        .replace_all(&text, |m: &Captures| {
            let prefix = m.get(1).map(|x| x.as_str()).unwrap_or("");
            let digits = &m[2];
            format!("{prefix}{digits}:***")
        })
        .into_owned();

    // Private key blocks.
    text = p
        .private_key
        .replace_all(&text, "[REDACTED PRIVATE KEY]")
        .into_owned();

    // Database connection string passwords.
    text = p
        .db_connstr
        .replace_all(&text, |m: &Captures| {
            format!("{}***{}", &m[1], &m[3])
        })
        .into_owned();

    // JWT tokens (eyJ... — base64-encoded JSON headers).
    text = p
        .jwt
        .replace_all(&text, |m: &Captures| mask_token(&m[0]))
        .into_owned();

    // URL userinfo (http(s)://user:pass@host) — non-DB schemes.
    text = redact_url_userinfo(&text);

    // URL query params containing opaque tokens (?access_token=…&code=…).
    text = redact_url_query_params(&text);

    // Form-urlencoded bodies (only triggers on clean k=v&k=v inputs).
    text = redact_form_body(&text);

    // Discord user/role mentions (<@snowflake_id>).
    text = p
        .discord_mention
        .replace_all(&text, |m: &Captures| {
            let bang = if m[0].contains('!') { "!" } else { "" };
            format!("<@{bang}***>")
        })
        .into_owned();

    // E.164 phone numbers (Signal, WhatsApp) — manual trailing guard.
    text = redact_phones(&text);

    text
}

/// Convenience wrapper matching the most common call site: redaction governed
/// by the global flag, treating the input as ordinary (non-code) text.
pub fn redact_log(text: &str) -> String {
    redact_sensitive_text(text, false, false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Force-redact convenience for tests (independent of the env flag).
    fn r(text: &str) -> String {
        redact_sensitive_text(text, true, false)
    }
    fn r_code(text: &str) -> String {
        redact_sensitive_text(text, true, true)
    }

    #[test]
    fn mask_secret_examples() {
        assert_eq!(mask_secret_default("sk-proj-abcdef1234567890"), "sk-p...7890");
        assert_eq!(mask_secret_default("short"), "***");
        assert_eq!(mask_secret_default(""), "");
        let opts = MaskOptions {
            empty: "(not set)",
            ..Default::default()
        };
        assert_eq!(mask_secret("", &opts), "(not set)");
        let opts = MaskOptions {
            head: 6,
            tail: 4,
            floor: 18,
            ..Default::default()
        };
        assert_eq!(mask_secret("long-token", &opts), "***");
    }

    #[test]
    fn prefix_tokens() {
        assert_eq!(r("my key sk-abcdefghij1234567890 done"), "my key sk-abc...7890 done");
        assert_eq!(r("token=ghp_abcdefghij1234567890"), "token=ghp_ab...7890");
        // Too short to satisfy {10,} -> no match.
        assert_eq!(r("foo sk-short"), "foo sk-short");
    }

    #[test]
    fn prefix_word_boundary_guards() {
        // Preceded by a word char -> lookbehind fails -> no redaction.
        assert_eq!(r("xsk-abcdefghij1234567890"), "xsk-abcdefghij1234567890");
        // Trailing char is part of the token run, so it is included & masked.
        assert_eq!(r("sk-abcdefghij1234567890x"), "sk-abc...890x");
    }

    #[test]
    fn auth_header() {
        assert_eq!(
            r("Authorization: Bearer abcdefghijklmnopqrst"),
            "Authorization: Bearer abcdef...qrst"
        );
    }

    #[test]
    fn url_query_and_userinfo() {
        assert_eq!(
            r("https://example.com/cb?code=ABC123&state=xyz"),
            "https://example.com/cb?code=***&state=xyz"
        );
        assert_eq!(
            r("https://user:tokenval@api.example.com/v1/foo"),
            "https://user:***@api.example.com/v1/foo"
        );
    }

    #[test]
    fn db_connstr() {
        assert_eq!(r("postgres://u:pass@host/db"), "postgres://u:***@host/db");
    }

    #[test]
    fn telegram_tokens() {
        assert_eq!(
            r("bot123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"),
            "bot123456789:***"
        );
        assert_eq!(
            r("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"),
            "123456789:***"
        );
    }

    #[test]
    fn phone_numbers() {
        assert_eq!(r("call +14155552671 now"), "call +141****2671 now");
        // Trailing alphanumeric -> guard fails -> no redaction.
        assert_eq!(r("+14155552671abc"), "+14155552671abc");
        // Short phone (<= 8 chars) uses the 2/2 mask.
        assert_eq!(r("+1234567 x"), "+1****67 x");
    }

    #[test]
    fn discord_mentions() {
        assert_eq!(r("ping <@!123456789012345678> hi"), "ping <@!***> hi");
        assert_eq!(r("<@123456789012345678>"), "<@***>");
    }

    #[test]
    fn jwt() {
        assert_eq!(
            r("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcd"),
            "eyJhbG...abcd"
        );
    }

    #[test]
    fn env_and_json_fields() {
        assert_eq!(r("OPENAI_API_KEY=verysecretvalue123"), "OPENAI_API_KEY=veryse...e123");
        assert_eq!(r("{\"apiKey\": \"verysecretvalue123\"}"), "{\"apiKey\": \"veryse...e123\"}");
    }

    #[test]
    fn code_file_skips_env_and_json() {
        // ENV-assignment pattern is skipped for source code.
        assert_eq!(r_code("MAX_TOKENS=secretvalue123456"), "MAX_TOKENS=secretvalue123456");
    }

    #[test]
    fn form_body() {
        assert_eq!(r("a=1&b=2&token=secret&c=3"), "a=1&b=2&token=***&c=3");
        assert_eq!(r("a=1&password=secret"), "a=1&password=***");
    }

    #[test]
    fn private_key_block() {
        let input = "-----BEGIN RSA PRIVATE KEY-----\nabc\ndef\n-----END RSA PRIVATE KEY-----";
        assert_eq!(r(input), "[REDACTED PRIVATE KEY]");
    }

    #[test]
    fn disabled_passthrough() {
        // force=false and env flag off by default in the test process.
        assert_eq!(redact_sensitive_text("sk-abcdefghij1234567890", false, false), "sk-abcdefghij1234567890");
    }

    #[test]
    fn empty_passthrough() {
        assert_eq!(r(""), "");
    }
}
