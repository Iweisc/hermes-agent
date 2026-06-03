//! Feishu document comment access-control rules.
//!
//! 3-tier rule resolution: exact doc > wildcard "*" > top-level > code defaults.
//! Each field (enabled/policy/allow_from) falls back independently.
//!
//! Config:  `~/.hermes/feishu_comment_rules.json`   (mtime-cached, hot-reload)
//! Pairing: `~/.hermes/feishu_comment_pairing.json`
//!
//! Faithful native Rust port of `gateway/platforms/feishu_comment_rules.py`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// `~/.hermes` honouring `HERMES_HOME`. Mirrors Python `get_hermes_home()`.
pub fn get_hermes_home() -> PathBuf {
    if let Ok(val) = env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/"))
        .join(".hermes")
}

/// Path to the comment-rules config file.
pub fn rules_file() -> PathBuf {
    get_hermes_home().join("feishu_comment_rules.json")
}

/// Path to the pairing-store file.
pub fn pairing_file() -> PathBuf {
    get_hermes_home().join("feishu_comment_pairing.json")
}

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

/// Valid policy values. Anything else is treated as absent / invalid.
pub const VALID_POLICIES: [&str; 2] = ["allowlist", "pairing"];

fn is_valid_policy(p: &str) -> bool {
    VALID_POLICIES.contains(&p)
}

/// Per-document rule. `None` means "inherit from lower tier".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommentDocumentRule {
    pub enabled: Option<bool>,
    pub policy: Option<String>,
    /// `None` means the `allow_from` key was absent (inherit). `Some(set)`
    /// means it was present (even if empty).
    pub allow_from: Option<HashSet<String>>,
}

/// Top-level comment access config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommentsConfig {
    pub enabled: bool,
    pub policy: String,
    pub allow_from: HashSet<String>,
    pub documents: HashMap<String, CommentDocumentRule>,
}

impl Default for CommentsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            policy: "pairing".to_string(),
            allow_from: HashSet::new(),
            documents: HashMap::new(),
        }
    }
}

/// Fully resolved rule after field-by-field fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCommentRule {
    pub enabled: bool,
    pub policy: String,
    pub allow_from: HashSet<String>,
    /// e.g. `"exact:docx:xxx"` | `"wildcard"` | `"top"` | `"default"`.
    pub match_source: String,
}

// ---------------------------------------------------------------------------
// Mtime-cached file loading
// ---------------------------------------------------------------------------

/// Generic mtime-based file cache: `stat()` per access, re-read only on change.
///
/// Returns a JSON object (`Map`). Missing file / parse error => empty object.
pub struct MtimeCache {
    path: PathBuf,
    mtime: u64,
    data: Option<Map<String, Value>>,
}

impl MtimeCache {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            mtime: 0,
            data: None,
        }
    }

    /// Returns the file's modification time as a nanosecond integer, or `None`
    /// if the file does not exist / cannot be stat'd.
    fn current_mtime(&self) -> Option<u64> {
        let meta = fs::metadata(&self.path).ok()?;
        let modified = meta.modified().ok()?;
        let dur = modified.duration_since(UNIX_EPOCH).ok()?;
        Some(dur.as_nanos() as u64)
    }

    pub fn load(&mut self) -> Map<String, Value> {
        let mtime = match self.current_mtime() {
            Some(m) => m,
            None => {
                // FileNotFoundError equivalent.
                self.mtime = 0;
                self.data = Some(Map::new());
                return Map::new();
            }
        };

        if mtime == self.mtime {
            if let Some(data) = &self.data {
                return data.clone();
            }
        }

        let data = match fs::read_to_string(&self.path) {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(Value::Object(obj)) => obj,
                Ok(_) => Map::new(),
                Err(_) => {
                    log::warn!(
                        "[Feishu-Rules] Failed to read {}, using empty config",
                        self.path.display()
                    );
                    Map::new()
                }
            },
            Err(_) => {
                log::warn!(
                    "[Feishu-Rules] Failed to read {}, using empty config",
                    self.path.display()
                );
                Map::new()
            }
        };

        self.mtime = mtime;
        self.data = Some(data.clone());
        data
    }

    /// Invalidate cache so the next `load()` re-reads from disk.
    pub fn invalidate(&mut self) {
        self.mtime = 0;
        self.data = None;
    }
}

// Process-wide caches matching the Python module-level singletons.
static RULES_CACHE: Mutex<Option<MtimeCache>> = Mutex::new(None);
static PAIRING_CACHE: Mutex<Option<MtimeCache>> = Mutex::new(None);

fn with_rules_cache<R>(f: impl FnOnce(&mut MtimeCache) -> R) -> R {
    let mut guard = RULES_CACHE.lock().unwrap();
    if guard.is_none() {
        *guard = Some(MtimeCache::new(rules_file()));
    }
    f(guard.as_mut().unwrap())
}

fn with_pairing_cache<R>(f: impl FnOnce(&mut MtimeCache) -> R) -> R {
    let mut guard = PAIRING_CACHE.lock().unwrap();
    if guard.is_none() {
        *guard = Some(MtimeCache::new(pairing_file()));
    }
    f(guard.as_mut().unwrap())
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// Parse a list of strings into a set; return `None` if key absent (or not a list).
///
/// Mirrors `_parse_frozenset`: only lists/tuples produce a set; everything
/// else (including absent) yields `None`. Each element is stringified, trimmed,
/// and dropped if empty.
fn parse_set(raw: Option<&Value>) -> Option<HashSet<String>> {
    match raw {
        None | Some(Value::Null) => None,
        Some(Value::Array(items)) => {
            let mut set = HashSet::new();
            for item in items {
                let s = value_to_str(item);
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    set.insert(trimmed.to_string());
                }
            }
            Some(set)
        }
        Some(_) => None,
    }
}

/// Stringify a JSON value the way Python's `str()` would for the values that
/// realistically appear in an `allow_from` list (strings, numbers, bools).
fn value_to_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Null => "None".to_string(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

/// Coerce a JSON value to bool the way Python `bool()` does for config values.
fn value_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn parse_document_rule(raw: &Map<String, Value>) -> CommentDocumentRule {
    let enabled = match raw.get("enabled") {
        None | Some(Value::Null) => None,
        Some(v) => Some(value_truthy(v)),
    };

    let policy = match raw.get("policy") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let p = value_to_str(v).trim().to_lowercase();
            if is_valid_policy(&p) {
                Some(p)
            } else {
                None
            }
        }
    };

    let allow_from = parse_set(raw.get("allow_from"));

    CommentDocumentRule {
        enabled,
        policy,
        allow_from,
    }
}

/// Load comment rules from disk (mtime-cached).
pub fn load_config() -> CommentsConfig {
    let raw = with_rules_cache(|c| c.load());
    if raw.is_empty() {
        return CommentsConfig::default();
    }

    let mut documents: HashMap<String, CommentDocumentRule> = HashMap::new();
    if let Some(Value::Object(raw_docs)) = raw.get("documents") {
        for (key, rule_raw) in raw_docs {
            if let Value::Object(rule_obj) = rule_raw {
                documents.insert(key.clone(), parse_document_rule(rule_obj));
            }
        }
    }

    let policy = match raw.get("policy") {
        Some(v) => value_to_str(v).trim().to_lowercase(),
        None => "pairing".to_string(),
    };
    let policy = if is_valid_policy(&policy) {
        policy
    } else {
        "pairing".to_string()
    };

    let enabled = match raw.get("enabled") {
        Some(v) => value_truthy(v),
        None => true,
    };

    let allow_from = parse_set(raw.get("allow_from")).unwrap_or_default();

    CommentsConfig {
        enabled,
        policy,
        allow_from,
        documents,
    }
}

// ---------------------------------------------------------------------------
// Rule resolution (field-by-field fallback)
// ---------------------------------------------------------------------------

/// Check if any document rule key starts with `"wiki:"`.
pub fn has_wiki_keys(cfg: &CommentsConfig) -> bool {
    cfg.documents.keys().any(|k| k.starts_with("wiki:"))
}

/// Resolve effective rule: exact doc -> wiki key -> wildcard -> top-level -> defaults.
pub fn resolve_rule(
    cfg: &CommentsConfig,
    file_type: &str,
    file_token: &str,
    wiki_token: &str,
) -> ResolvedCommentRule {
    let exact_key = format!("{file_type}:{file_token}");

    let mut exact = cfg.documents.get(&exact_key);
    let mut exact_src = format!("exact:{exact_key}");
    if exact.is_none() && !wiki_token.is_empty() {
        let wiki_key = format!("wiki:{wiki_token}");
        exact = cfg.documents.get(&wiki_key);
        exact_src = format!("exact:{wiki_key}");
    }

    let wildcard = cfg.documents.get("*");

    // (rule, source) layers in priority order.
    let mut layers: Vec<(&CommentDocumentRule, String)> = Vec::new();
    if let Some(e) = exact {
        layers.push((e, exact_src));
    }
    if let Some(w) = wildcard {
        layers.push((w, "wildcard".to_string()));
    }

    // `_pick`: walk layers; first non-None value wins, else fall to top-level.
    let pick_enabled = || -> (bool, String) {
        for (layer, source) in &layers {
            if let Some(v) = layer.enabled {
                return (v, source.clone());
            }
        }
        (cfg.enabled, "top".to_string())
    };
    let pick_policy = || -> (String, String) {
        for (layer, source) in &layers {
            if let Some(v) = &layer.policy {
                return (v.clone(), source.clone());
            }
        }
        (cfg.policy.clone(), "top".to_string())
    };
    let pick_allow = || -> (HashSet<String>, String) {
        for (layer, source) in &layers {
            if let Some(v) = &layer.allow_from {
                return (v.clone(), source.clone());
            }
        }
        (cfg.allow_from.clone(), "top".to_string())
    };

    let (enabled, en_src) = pick_enabled();
    let (policy, pol_src) = pick_policy();
    let (allow_from, _) = pick_allow();

    // match_source = highest-priority tier that contributed enabled/policy.
    let best_src = min_priority_source(&en_src, &pol_src);

    ResolvedCommentRule {
        enabled,
        policy,
        allow_from,
        match_source: best_src,
    }
}

/// Return the source with the highest priority (lowest rank).
/// rank: exact=0, wildcard=1, top=2, anything else=3.
/// Ties keep the first argument (matching Python `min`'s stable behaviour).
fn min_priority_source(a: &str, b: &str) -> String {
    fn rank(s: &str) -> u8 {
        let prefix = s.split(':').next().unwrap_or("");
        match prefix {
            "exact" => 0,
            "wildcard" => 1,
            "top" => 2,
            _ => 3,
        }
    }
    if rank(b) < rank(a) {
        b.to_string()
    } else {
        a.to_string()
    }
}

// ---------------------------------------------------------------------------
// Pairing store
// ---------------------------------------------------------------------------

/// Return set of approved user open_ids (mtime-cached).
fn load_pairing_approved() -> HashSet<String> {
    let data = with_pairing_cache(|c| c.load());
    match data.get("approved") {
        Some(Value::Object(obj)) => obj.keys().cloned().collect(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter(|v| value_truthy(v))
            .map(value_to_str)
            .collect(),
        _ => HashSet::new(),
    }
}

fn save_pairing(data: &Map<String, Value>) -> std::io::Result<()> {
    let path = pairing_file();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    let serialized = serde_json::to_string_pretty(&Value::Object(data.clone()))
        .unwrap_or_else(|_| "{}".to_string());
    fs::write(&tmp, serialized)?;
    fs::rename(&tmp, &path)?;
    with_pairing_cache(|c| c.invalidate());
    Ok(())
}

fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Add a user to the pairing-approved list. Returns `true` if newly added.
pub fn pairing_add(user_open_id: &str) -> bool {
    let mut data = with_pairing_cache(|c| c.load());
    let mut approved = match data.get("approved") {
        Some(Value::Object(obj)) => obj.clone(),
        _ => Map::new(),
    };
    if approved.contains_key(user_open_id) {
        return false;
    }
    let mut meta = Map::new();
    meta.insert(
        "approved_at".to_string(),
        Value::from(now_unix_secs()),
    );
    approved.insert(user_open_id.to_string(), Value::Object(meta));
    data.insert("approved".to_string(), Value::Object(approved));
    let _ = save_pairing(&data);
    true
}

/// Remove a user from the pairing-approved list. Returns `true` if removed.
pub fn pairing_remove(user_open_id: &str) -> bool {
    let mut data = with_pairing_cache(|c| c.load());
    let mut approved = match data.get("approved") {
        Some(Value::Object(obj)) => obj.clone(),
        _ => return false,
    };
    if !approved.contains_key(user_open_id) {
        return false;
    }
    approved.remove(user_open_id);
    data.insert("approved".to_string(), Value::Object(approved));
    let _ = save_pairing(&data);
    true
}

/// Return the approved map `{user_open_id: {approved_at: ...}}`.
pub fn pairing_list() -> Map<String, Value> {
    let data = with_pairing_cache(|c| c.load());
    match data.get("approved") {
        Some(Value::Object(obj)) => obj.clone(),
        _ => Map::new(),
    }
}

// ---------------------------------------------------------------------------
// Access check (public API)
// ---------------------------------------------------------------------------

/// Check if user passes the resolved rule's policy gate.
pub fn is_user_allowed(rule: &ResolvedCommentRule, user_open_id: &str) -> bool {
    if rule.allow_from.contains(user_open_id) {
        return true;
    }
    if rule.policy == "pairing" {
        return load_pairing_approved().contains(user_open_id);
    }
    false
}

// ---------------------------------------------------------------------------
// CLI helpers
// ---------------------------------------------------------------------------

fn sorted_vec(set: &HashSet<String>) -> Vec<String> {
    let mut v: Vec<String> = set.iter().cloned().collect();
    v.sort();
    v
}

/// Render the `status` command output as a String.
pub fn render_status() -> String {
    let cfg = load_config();
    let rf = rules_file();
    let pf = pairing_file();
    let mut out = String::new();

    out.push_str(&format!("Rules file: {}\n", rf.display()));
    out.push_str(&format!("  exists: {}\n", py_bool(rf.exists())));
    out.push_str(&format!("Pairing file: {}\n", pf.display()));
    out.push_str(&format!("  exists: {}\n", py_bool(pf.exists())));
    out.push('\n');
    out.push_str("Top-level:\n");
    out.push_str(&format!("  enabled:    {}\n", py_bool(cfg.enabled)));
    out.push_str(&format!("  policy:     {}\n", cfg.policy));
    let af = if cfg.allow_from.is_empty() {
        "[]".to_string()
    } else {
        format!("{:?}", sorted_vec(&cfg.allow_from))
    };
    out.push_str(&format!("  allow_from: {af}\n"));
    out.push('\n');

    if !cfg.documents.is_empty() {
        out.push_str(&format!("Document rules ({}):\n", cfg.documents.len()));
        let mut keys: Vec<&String> = cfg.documents.keys().collect();
        keys.sort();
        for key in keys {
            let rule = &cfg.documents[key];
            let mut parts: Vec<String> = Vec::new();
            if let Some(e) = rule.enabled {
                parts.push(format!("enabled={}", py_bool(e)));
            }
            if let Some(p) = &rule.policy {
                parts.push(format!("policy={p}"));
            }
            if let Some(a) = &rule.allow_from {
                parts.push(format!("allow_from={:?}", sorted_vec(a)));
            }
            let body = if parts.is_empty() {
                "(empty — inherits all)".to_string()
            } else {
                parts.join(", ")
            };
            out.push_str(&format!("  [{key}] {body}\n"));
        }
    } else {
        out.push_str("Document rules: (none)\n");
    }
    out.push('\n');

    let approved = pairing_list();
    out.push_str(&format!("Pairing approved ({}):\n", approved.len()));
    let sorted: BTreeMap<&String, &Value> = approved.iter().collect();
    for (uid, meta) in sorted {
        let ts = meta
            .get("approved_at")
            .map(format_number)
            .unwrap_or_else(|| "0".to_string());
        out.push_str(&format!("  {uid}  (approved_at={ts})\n"));
    }
    out
}

/// Render the `check <fileType:token> <user>` command output as a String.
pub fn render_check(doc_key: &str, user_open_id: &str) -> String {
    let cfg = load_config();
    let parts: Vec<&str> = doc_key.splitn(2, ':').collect();
    if parts.len() != 2 {
        return format!(
            "Error: doc_key must be 'fileType:fileToken', got '{doc_key}'\n"
        );
    }
    let file_type = parts[0];
    let file_token = parts[1];
    let rule = resolve_rule(&cfg, file_type, file_token, "");
    let allowed = is_user_allowed(&rule, user_open_id);

    let af = if rule.allow_from.is_empty() {
        "[]".to_string()
    } else {
        format!("{:?}", sorted_vec(&rule.allow_from))
    };

    let mut out = String::new();
    out.push_str(&format!("Document:     {doc_key}\n"));
    out.push_str(&format!("User:         {user_open_id}\n"));
    out.push_str("Resolved rule:\n");
    out.push_str(&format!("  enabled:      {}\n", py_bool(rule.enabled)));
    out.push_str(&format!("  policy:       {}\n", rule.policy));
    out.push_str(&format!("  allow_from:   {af}\n"));
    out.push_str(&format!("  match_source: {}\n", rule.match_source));
    out.push_str(&format!(
        "Result:       {}\n",
        if allowed { "ALLOWED" } else { "DENIED" }
    ));
    out
}

fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// Render a JSON number like Python would (integers without a trailing `.0`
/// only when they came in as ints; floats keep their representation).
fn format_number(v: &Value) -> String {
    match v {
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(
        enabled: Option<bool>,
        policy: Option<&str>,
        allow_from: Option<&[&str]>,
    ) -> CommentDocumentRule {
        CommentDocumentRule {
            enabled,
            policy: policy.map(|s| s.to_string()),
            allow_from: allow_from
                .map(|a| a.iter().map(|s| s.to_string()).collect()),
        }
    }

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_set_handles_lists_and_trims() {
        let v = serde_json::json!(["a", " b ", "", "c"]);
        let got = parse_set(Some(&v)).unwrap();
        assert_eq!(got, set(&["a", "b", "c"]));

        assert!(parse_set(None).is_none());
        assert!(parse_set(Some(&Value::Null)).is_none());
        assert!(parse_set(Some(&serde_json::json!("notalist"))).is_none());
    }

    #[test]
    fn parse_document_rule_validates_policy() {
        let raw: Map<String, Value> = serde_json::from_value(serde_json::json!({
            "enabled": false,
            "policy": "ALLOWLIST",
            "allow_from": ["u1"]
        }))
        .unwrap();
        let rule = parse_document_rule(&raw);
        assert_eq!(rule.enabled, Some(false));
        assert_eq!(rule.policy.as_deref(), Some("allowlist"));
        assert_eq!(rule.allow_from, Some(set(&["u1"])));

        let raw2: Map<String, Value> =
            serde_json::from_value(serde_json::json!({ "policy": "bogus" }))
                .unwrap();
        let rule2 = parse_document_rule(&raw2);
        assert_eq!(rule2.policy, None);
        assert_eq!(rule2.enabled, None);
        assert_eq!(rule2.allow_from, None);
    }

    #[test]
    fn resolve_defaults_to_top_level() {
        let cfg = CommentsConfig::default();
        let rule = resolve_rule(&cfg, "docx", "abc", "");
        assert!(rule.enabled);
        assert_eq!(rule.policy, "pairing");
        assert!(rule.allow_from.is_empty());
        assert_eq!(rule.match_source, "top");
    }

    #[test]
    fn resolve_exact_overrides_wildcard_and_top() {
        let mut cfg = CommentsConfig::default();
        cfg.documents.insert(
            "*".to_string(),
            doc(Some(false), Some("allowlist"), Some(&["w1"])),
        );
        cfg.documents.insert(
            "docx:abc".to_string(),
            doc(Some(true), None, Some(&["e1"])),
        );
        let rule = resolve_rule(&cfg, "docx", "abc", "");
        // enabled from exact, policy falls to wildcard, allow_from from exact.
        assert!(rule.enabled);
        assert_eq!(rule.policy, "allowlist");
        assert_eq!(rule.allow_from, set(&["e1"]));
        assert_eq!(rule.match_source, "exact:docx:abc");
    }

    #[test]
    fn resolve_field_by_field_fallback() {
        let mut cfg = CommentsConfig::default();
        cfg.enabled = true;
        cfg.policy = "pairing".to_string();
        // exact rule only specifies enabled; policy must fall through.
        cfg.documents.insert(
            "docx:abc".to_string(),
            doc(Some(false), None, None),
        );
        let rule = resolve_rule(&cfg, "docx", "abc", "");
        assert!(!rule.enabled);
        assert_eq!(rule.policy, "pairing");
        // match_source picks highest priority tier contributing enabled/policy.
        // enabled comes from exact (rank 0), policy from top (rank 2) -> exact.
        assert_eq!(rule.match_source, "exact:docx:abc");
    }

    #[test]
    fn resolve_uses_wiki_token_when_no_exact() {
        let mut cfg = CommentsConfig::default();
        cfg.documents.insert(
            "wiki:wt1".to_string(),
            doc(Some(false), Some("allowlist"), None),
        );
        let rule = resolve_rule(&cfg, "docx", "abc", "wt1");
        assert!(!rule.enabled);
        assert_eq!(rule.policy, "allowlist");
        assert_eq!(rule.match_source, "exact:wiki:wt1");
    }

    #[test]
    fn resolve_exact_beats_wiki() {
        let mut cfg = CommentsConfig::default();
        cfg.documents
            .insert("docx:abc".to_string(), doc(Some(true), None, None));
        cfg.documents
            .insert("wiki:wt1".to_string(), doc(Some(false), None, None));
        let rule = resolve_rule(&cfg, "docx", "abc", "wt1");
        assert!(rule.enabled);
        assert_eq!(rule.match_source, "exact:docx:abc");
    }

    #[test]
    fn has_wiki_keys_detects_prefix() {
        let mut cfg = CommentsConfig::default();
        assert!(!has_wiki_keys(&cfg));
        cfg.documents
            .insert("wiki:x".to_string(), CommentDocumentRule::default());
        assert!(has_wiki_keys(&cfg));
    }

    #[test]
    fn is_user_allowed_allow_from_short_circuit() {
        let rule = ResolvedCommentRule {
            enabled: true,
            policy: "allowlist".to_string(),
            allow_from: set(&["u1"]),
            match_source: "top".to_string(),
        };
        assert!(is_user_allowed(&rule, "u1"));
        assert!(!is_user_allowed(&rule, "u2"));
    }

    #[test]
    fn min_priority_source_picks_highest() {
        assert_eq!(min_priority_source("exact:x", "top"), "exact:x");
        assert_eq!(min_priority_source("top", "wildcard"), "wildcard");
        assert_eq!(min_priority_source("wildcard", "wildcard"), "wildcard");
        // tie keeps first arg
        assert_eq!(min_priority_source("top", "top"), "top");
    }

    #[test]
    fn pairing_store_roundtrip() {
        // Isolate HERMES_HOME to a temp dir for this test.
        let dir = std::env::temp_dir().join(format!(
            "hermes_feishu_rules_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // SAFETY: single-threaded test context for env mutation.
        unsafe { env::set_var("HERMES_HOME", &dir); }
        with_pairing_cache(|c| {
            *c = MtimeCache::new(pairing_file());
        });

        assert!(pairing_list().is_empty());
        assert!(pairing_add("user-a"));
        assert!(!pairing_add("user-a")); // already present
        assert!(pairing_list().contains_key("user-a"));

        let approved = load_pairing_approved();
        assert!(approved.contains("user-a"));

        assert!(pairing_remove("user-a"));
        assert!(!pairing_remove("user-a"));
        assert!(pairing_list().is_empty());

        let _ = fs::remove_dir_all(&dir);
        unsafe { env::remove_var("HERMES_HOME"); }
    }

    #[test]
    fn value_truthy_matches_python() {
        assert!(!value_truthy(&Value::Null));
        assert!(!value_truthy(&serde_json::json!(0)));
        assert!(value_truthy(&serde_json::json!(1)));
        assert!(!value_truthy(&serde_json::json!("")));
        assert!(value_truthy(&serde_json::json!("x")));
        assert!(!value_truthy(&serde_json::json!([])));
    }
}
