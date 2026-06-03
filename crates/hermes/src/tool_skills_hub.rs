//! Skills Hub — source adapters and hub state management.
//!
//! Native Rust port of `tools/skills_hub.py`.
//!
//! Provides:
//!   - [`GitHubAuth`]: shared GitHub API authentication (PAT, gh CLI, GitHub App)
//!   - [`SkillSource`] trait: interface for all skill registry adapters
//!   - [`OptionalSkillSource`], [`GitHubSource`], [`WellKnownSkillSource`],
//!     [`UrlSource`], [`SkillsShSource`], [`ClawHubSource`],
//!     [`ClaudeMarketplaceSource`], [`LobeHubSource`], [`HermesIndexSource`]
//!   - [`HubLockFile`]: tracks provenance of installed hub skills
//!   - Hub state directory management (quarantine, audit log, taps, index cache)
//!
//! Network calls use `reqwest::blocking`; request construction and response
//! parsing mirror the Python `httpx` usage exactly.
//!
//! This module is intentionally self-contained: the few cross-module
//! dependencies in the Python original (`hermes_constants.get_hermes_home`,
//! `tools.skills_guard.{ScanResult, content_hash, TRUSTED_REPOS}`) are
//! reproduced as local helpers so the module compiles standalone. When the
//! shared guard module is exposed cross-crate it can be swapped in.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use regex::Regex;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Mirror of `hermes_constants.get_hermes_home()` (HERMES_HOME env or ~/.hermes).
pub fn get_hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let val = val.trim();
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
    home.join(".hermes")
}

/// `optional-skills/` directory, honouring `HERMES_OPTIONAL_SKILLS`.
pub fn get_optional_skills_dir(default: PathBuf) -> PathBuf {
    if let Ok(v) = std::env::var("HERMES_OPTIONAL_SKILLS") {
        let v = v.trim();
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    default
}

pub fn skills_dir() -> PathBuf {
    get_hermes_home().join("skills")
}
pub fn hub_dir() -> PathBuf {
    skills_dir().join(".hub")
}
pub fn lock_file_path() -> PathBuf {
    hub_dir().join("lock.json")
}
pub fn quarantine_dir() -> PathBuf {
    hub_dir().join("quarantine")
}
pub fn audit_log_path() -> PathBuf {
    hub_dir().join("audit.log")
}
pub fn taps_file_path() -> PathBuf {
    hub_dir().join("taps.json")
}
pub fn index_cache_dir() -> PathBuf {
    hub_dir().join("index-cache")
}

/// Cache duration for remote index fetches (1 hour).
pub const INDEX_CACHE_TTL: u64 = 3600;

/// Repositories that are considered "trusted". Mirror of
/// `tools.skills_guard.TRUSTED_REPOS`.
pub const TRUSTED_REPOS: &[&str] = &["openai/skills", "anthropics/skills"];

// ---------------------------------------------------------------------------
// Minimal ScanResult (mirror of tools.skills_guard.ScanResult, verdict-only use)
// ---------------------------------------------------------------------------

/// Minimal view of a scan result; only `verdict` is consumed by this module.
#[derive(Debug, Clone, Default)]
pub struct ScanResult {
    pub verdict: String,
}

/// Deterministic content hash of an installed skill directory. Mirror of
/// `tools.skills_guard.content_hash`: sha256 over sorted relative file paths
/// and their bytes, returned as `sha256:<first 16 hex chars>`.
pub fn content_hash(skill_path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    collect_files(skill_path, skill_path, &mut entries);
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha256::new();
    for (rel, abs) in entries {
        hasher.update(rel.as_bytes());
        if let Ok(bytes) = std::fs::read(&abs) {
            hasher.update(&bytes);
        }
    }
    let digest = hasher.finalize();
    let hex = digest.iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!("sha256:{}", &hex[..16])
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, out);
        } else if path.is_file() {
            if let Ok(rel) = path.strip_prefix(root) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                out.push((rel_str, path));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

/// File content in a bundle — either text or raw bytes (binary).
#[derive(Debug, Clone)]
pub enum FileContent {
    Text(String),
    Bytes(Vec<u8>),
}

impl FileContent {
    pub fn as_bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            FileContent::Text(s) => std::borrow::Cow::Borrowed(s.as_bytes()),
            FileContent::Bytes(b) => std::borrow::Cow::Borrowed(b),
        }
    }
}

/// Minimal metadata returned by search results.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    /// "official", "github", "clawhub", "claude-marketplace", "lobehub", ...
    pub source: String,
    /// source-specific id (e.g. "openai/skills/skill-creator")
    pub identifier: String,
    /// "builtin" | "trusted" | "community"
    pub trust_level: String,
    pub repo: Option<String>,
    pub path: Option<String>,
    pub tags: Vec<String>,
    pub extra: BTreeMap<String, Value>,
}

/// A downloaded skill ready for quarantine/scanning/installation.
#[derive(Debug, Clone)]
pub struct SkillBundle {
    pub name: String,
    /// relative_path -> file content
    pub files: BTreeMap<String, FileContent>,
    pub source: String,
    pub identifier: String,
    pub trust_level: String,
    pub metadata: BTreeMap<String, Value>,
}

impl SkillBundle {
    pub fn new(
        name: impl Into<String>,
        files: BTreeMap<String, FileContent>,
        source: impl Into<String>,
        identifier: impl Into<String>,
        trust_level: impl Into<String>,
    ) -> Self {
        SkillBundle {
            name: name.into(),
            files,
            source: source.into(),
            identifier: identifier.into(),
            trust_level: trust_level.into(),
            metadata: BTreeMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Path validation
// ---------------------------------------------------------------------------

/// Normalize and validate bundle-controlled paths before touching disk.
/// Returns the cleaned forward-slash path or an error string.
pub fn normalize_bundle_path(
    path_value: &str,
    field_name: &str,
    allow_nested: bool,
) -> Result<String, String> {
    let raw = path_value.trim();
    if raw.is_empty() {
        return Err(format!("Unsafe {field_name}: empty path"));
    }
    let normalized = raw.replace('\\', "/");
    let parts: Vec<&str> = normalized
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();

    let is_absolute = normalized.starts_with('/');
    if is_absolute {
        return Err(format!("Unsafe {field_name}: {path_value}"));
    }
    if parts.is_empty() || parts.iter().any(|p| *p == "..") {
        return Err(format!("Unsafe {field_name}: {path_value}"));
    }
    // Windows drive letter check (e.g. "C:")
    let drive_re = drive_letter_re();
    if drive_re.is_match(parts[0]) {
        return Err(format!("Unsafe {field_name}: {path_value}"));
    }
    if !allow_nested && parts.len() != 1 {
        return Err(format!("Unsafe {field_name}: {path_value}"));
    }
    Ok(parts.join("/"))
}

fn drive_letter_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z]:$").unwrap())
}

pub fn validate_skill_name(name: &str) -> Result<String, String> {
    normalize_bundle_path(name, "skill name", false)
}

pub fn validate_category_name(category: &str) -> Result<String, String> {
    normalize_bundle_path(category, "category", false)
}

pub fn validate_bundle_rel_path(rel_path: &str) -> Result<String, String> {
    normalize_bundle_path(rel_path, "bundle file path", true)
}

// ---------------------------------------------------------------------------
// HTTP helpers (reqwest::blocking)
// ---------------------------------------------------------------------------

fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new())
}

// ---------------------------------------------------------------------------
// GitHub Authentication
// ---------------------------------------------------------------------------

/// GitHub API authentication. Tries methods in priority order:
///   1. GITHUB_TOKEN / GH_TOKEN env var (PAT)
///   2. `gh auth token` subprocess
///   3. GitHub App JWT + installation token
///   4. Unauthenticated
pub struct GitHubAuth {
    cached_token: std::cell::RefCell<Option<String>>,
    cached_method: std::cell::RefCell<Option<String>>,
    app_token_expiry: std::cell::Cell<f64>,
}

impl Default for GitHubAuth {
    fn default() -> Self {
        Self::new()
    }
}

impl GitHubAuth {
    pub fn new() -> Self {
        GitHubAuth {
            cached_token: std::cell::RefCell::new(None),
            cached_method: std::cell::RefCell::new(None),
            app_token_expiry: std::cell::Cell::new(0.0),
        }
    }

    /// Return authorization headers for GitHub API requests.
    pub fn get_headers(&self) -> Vec<(String, String)> {
        let token = self.resolve_token();
        let mut headers = vec![(
            "Accept".to_string(),
            "application/vnd.github.v3+json".to_string(),
        )];
        if let Some(t) = token {
            headers.push(("Authorization".to_string(), format!("token {t}")));
        }
        headers
    }

    pub fn is_authenticated(&self) -> bool {
        self.resolve_token().is_some()
    }

    /// Return which auth method is active: 'pat', 'gh-cli', 'github-app', or 'anonymous'.
    pub fn auth_method(&self) -> String {
        self.resolve_token();
        self.cached_method
            .borrow()
            .clone()
            .unwrap_or_else(|| "anonymous".to_string())
    }

    fn resolve_token(&self) -> Option<String> {
        // Return cached token if still valid
        if let Some(t) = self.cached_token.borrow().clone() {
            let method = self.cached_method.borrow().clone();
            if method.as_deref() != Some("github-app") || now_secs() < self.app_token_expiry.get() {
                return Some(t);
            }
        }

        // 1. Environment variable
        let env_token = std::env::var("GITHUB_TOKEN")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("GH_TOKEN").ok().filter(|s| !s.is_empty()));
        if let Some(token) = env_token {
            *self.cached_token.borrow_mut() = Some(token.clone());
            *self.cached_method.borrow_mut() = Some("pat".to_string());
            return Some(token);
        }

        // 2. gh CLI
        if let Some(token) = self.try_gh_cli() {
            *self.cached_token.borrow_mut() = Some(token.clone());
            *self.cached_method.borrow_mut() = Some("gh-cli".to_string());
            return Some(token);
        }

        // 3. GitHub App
        if let Some(token) = self.try_github_app() {
            *self.cached_token.borrow_mut() = Some(token.clone());
            *self.cached_method.borrow_mut() = Some("github-app".to_string());
            self.app_token_expiry.set(now_secs() + 3500.0);
            return Some(token);
        }

        *self.cached_method.borrow_mut() = Some("anonymous".to_string());
        None
    }

    fn try_gh_cli(&self) -> Option<String> {
        let output = std::process::Command::new("gh")
            .args(["auth", "token"])
            .output()
            .ok()?;
        if output.status.success() {
            let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !s.is_empty() {
                return Some(s);
            }
        }
        None
    }

    fn try_github_app(&self) -> Option<String> {
        let app_id = std::env::var("GITHUB_APP_ID").ok()?;
        let key_path = std::env::var("GITHUB_APP_PRIVATE_KEY_PATH").ok()?;
        let installation_id = std::env::var("GITHUB_APP_INSTALLATION_ID").ok()?;
        if app_id.is_empty() || key_path.is_empty() || installation_id.is_empty() {
            return None;
        }
        // JWT signing (RS256) requires a dedicated crate (jsonwebtoken); not in
        // the allowed set. Mirror Python's "PyJWT not installed → skip" branch.
        log::debug!("GitHub App JWT signing unavailable (no jwt crate), skipping GitHub App auth");
        let _ = (app_id, key_path, installation_id);
        None
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Frontmatter parsing (shared)
// ---------------------------------------------------------------------------

/// Parse YAML frontmatter from SKILL.md content. Mirror of
/// `GitHubSource._parse_frontmatter_quick`.
pub fn parse_frontmatter_quick(content: &str) -> serde_yaml::Mapping {
    if !content.starts_with("---") {
        return serde_yaml::Mapping::new();
    }
    let rest = &content[3..];
    let re = frontmatter_re();
    let m = match re.find(rest) {
        Some(m) => m,
        None => return serde_yaml::Mapping::new(),
    };
    // yaml_text = content[3 : m.start() + 3]
    let yaml_text = &content[3..(m.start() + 3)];
    match serde_yaml::from_str::<serde_yaml::Value>(yaml_text) {
        Ok(serde_yaml::Value::Mapping(m)) => m,
        _ => serde_yaml::Mapping::new(),
    }
}

fn frontmatter_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\n---\s*\n").unwrap())
}

fn yaml_get<'a>(m: &'a serde_yaml::Mapping, key: &str) -> Option<&'a serde_yaml::Value> {
    m.get(serde_yaml::Value::String(key.to_string()))
}

fn yaml_str(v: Option<&serde_yaml::Value>) -> String {
    match v {
        Some(serde_yaml::Value::String(s)) => s.clone(),
        Some(serde_yaml::Value::Null) | None => String::new(),
        Some(other) => serde_yaml::to_string(other)
            .unwrap_or_default()
            .trim_end()
            .to_string(),
    }
}

/// Extract `tags` from frontmatter, preferring `metadata.hermes.tags`.
fn extract_tags_from_fm(fm: &serde_yaml::Mapping) -> Vec<String> {
    if let Some(serde_yaml::Value::Mapping(meta)) = yaml_get(fm, "metadata") {
        if let Some(serde_yaml::Value::Mapping(hermes)) = yaml_get(meta, "hermes") {
            if let Some(serde_yaml::Value::Sequence(seq)) = yaml_get(hermes, "tags") {
                return seq.iter().map(yaml_value_to_string).collect();
            }
        }
    }
    if let Some(serde_yaml::Value::Sequence(seq)) = yaml_get(fm, "tags") {
        return seq.iter().map(yaml_value_to_string).collect();
    }
    Vec::new()
}

fn yaml_value_to_string(v: &serde_yaml::Value) -> String {
    match v {
        serde_yaml::Value::String(s) => s.clone(),
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        _ => serde_yaml::to_string(v).unwrap_or_default().trim_end().to_string(),
    }
}

// ---------------------------------------------------------------------------
// Source adapter interface
// ---------------------------------------------------------------------------

/// Abstract base for all skill registry adapters.
pub trait SkillSource {
    fn search(&self, query: &str, limit: usize) -> Vec<SkillMeta>;
    fn fetch(&self, identifier: &str) -> Option<SkillBundle>;
    fn inspect(&self, identifier: &str) -> Option<SkillMeta>;
    fn source_id(&self) -> String;
    fn trust_level_for(&self, _identifier: &str) -> String {
        "community".to_string()
    }
    /// Mirror of Python's `getattr(src, "is_available", False)` — only the
    /// Hermes index source overrides this.
    fn is_available(&self) -> bool {
        false
    }
}

fn trust_rank(level: &str) -> i32 {
    match level {
        "builtin" => 2,
        "trusted" => 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Index cache helpers
// ---------------------------------------------------------------------------

fn mtime_age_secs(path: &Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    let mtime_secs = mtime.duration_since(UNIX_EPOCH).ok()?.as_secs_f64();
    let now = now_secs();
    if now < mtime_secs {
        Some(0)
    } else {
        Some((now - mtime_secs) as u64)
    }
}

/// Read cached data if not expired (TTL = [`INDEX_CACHE_TTL`]).
pub fn read_index_cache(key: &str) -> Option<Value> {
    let cache_file = index_cache_dir().join(format!("{key}.json"));
    if !cache_file.exists() {
        return None;
    }
    let age = mtime_age_secs(&cache_file)?;
    if age > INDEX_CACHE_TTL {
        return None;
    }
    let text = std::fs::read_to_string(&cache_file).ok()?;
    serde_json::from_str(&text).ok()
}

/// Write data to cache, ensuring a `.ignore` exists in the hub dir.
pub fn write_index_cache(key: &str, data: &Value) {
    let _ = std::fs::create_dir_all(index_cache_dir());
    let ignore_file = hub_dir().join(".ignore");
    if !ignore_file.exists() {
        let _ = std::fs::write(&ignore_file, "# Exclude hub internals from search tools\n*\n");
    }
    let cache_file = index_cache_dir().join(format!("{key}.json"));
    if let Ok(s) = serde_json::to_string(data) {
        if let Err(e) = std::fs::write(&cache_file, s) {
            log::debug!("Could not write cache: {e}");
        }
    }
}

fn md5_hex(input: &str) -> String {
    format!("{:x}", md5::compute(input.as_bytes()))
}

// ---------------------------------------------------------------------------
// SkillMeta <-> JSON serialization for caches
// ---------------------------------------------------------------------------

/// Convert a [`SkillMeta`] to the JSON shape used by caches (mirror of
/// `_skill_meta_to_dict`).
pub fn skill_meta_to_dict(meta: &SkillMeta) -> Value {
    serde_json::json!({
        "name": meta.name,
        "description": meta.description,
        "source": meta.source,
        "identifier": meta.identifier,
        "trust_level": meta.trust_level,
        "repo": meta.repo,
        "path": meta.path,
        "tags": meta.tags,
        "extra": meta.extra,
    })
}

/// Reconstruct a [`SkillMeta`] from a cached JSON object (mirror of
/// `SkillMeta(**item)`). Returns None when required fields are missing.
pub fn skill_meta_from_value(v: &Value) -> Option<SkillMeta> {
    let obj = v.as_object()?;
    let get_str = |k: &str| obj.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let opt_str = |k: &str| {
        obj.get(k)
            .and_then(|x| if x.is_null() { None } else { x.as_str() })
            .map(|s| s.to_string())
    };
    let tags = obj
        .get("tags")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .map(|t| t.as_str().map(|s| s.to_string()).unwrap_or_default())
                .collect()
        })
        .unwrap_or_default();
    let extra = obj
        .get("extra")
        .and_then(|x| x.as_object())
        .map(|o| o.iter().map(|(k, val)| (k.clone(), val.clone())).collect())
        .unwrap_or_default();
    Some(SkillMeta {
        name: get_str("name"),
        description: get_str("description"),
        source: get_str("source"),
        identifier: get_str("identifier"),
        trust_level: get_str("trust_level"),
        repo: opt_str("repo"),
        path: opt_str("path"),
        tags,
        extra,
    })
}

fn metas_from_cache(v: &Value) -> Vec<SkillMeta> {
    v.as_array()
        .map(|a| a.iter().filter_map(skill_meta_from_value).collect())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// GitHub source adapter
// ---------------------------------------------------------------------------

/// A configured tap (GitHub repo + path).
#[derive(Debug, Clone)]
pub struct Tap {
    pub repo: String,
    pub path: String,
}

/// Fetch skills from GitHub repos via the Contents API.
pub struct GitHubSource {
    pub auth: std::rc::Rc<GitHubAuth>,
    pub taps: Vec<Tap>,
    /// repo -> (default_branch, tree_entries)
    tree_cache: std::cell::RefCell<HashMap<String, (String, Vec<Value>)>>,
    rate_limited: std::cell::Cell<bool>,
}

impl GitHubSource {
    pub fn default_taps() -> Vec<Tap> {
        vec![
            Tap { repo: "openai/skills".into(), path: "skills/".into() },
            Tap { repo: "anthropics/skills".into(), path: "skills/".into() },
            Tap { repo: "VoltAgent/awesome-agent-skills".into(), path: "skills/".into() },
            Tap { repo: "garrytan/gstack".into(), path: "".into() },
            Tap { repo: "MiniMax-AI/cli".into(), path: "skill/".into() },
        ]
    }

    pub fn new(auth: std::rc::Rc<GitHubAuth>, extra_taps: Option<Vec<Tap>>) -> Self {
        let mut taps = Self::default_taps();
        if let Some(extra) = extra_taps {
            taps.extend(extra);
        }
        GitHubSource {
            auth,
            taps,
            tree_cache: std::cell::RefCell::new(HashMap::new()),
            rate_limited: std::cell::Cell::new(false),
        }
    }

    pub fn is_rate_limited(&self) -> bool {
        self.rate_limited.get()
    }

    pub fn trust_level_for_impl(identifier: &str) -> String {
        let parts: Vec<&str> = identifier.splitn(3, '/').collect();
        if parts.len() >= 2 {
            let repo = format!("{}/{}", parts[0], parts[1]);
            if TRUSTED_REPOS.contains(&repo.as_str()) {
                return "trusted".to_string();
            }
        }
        "community".to_string()
    }

    fn get_headers_with_accept(&self, accept: &str) -> Vec<(String, String)> {
        let mut h = self.auth.get_headers();
        // Override/add Accept (Python merges with **headers then sets Accept).
        h.retain(|(k, _)| k != "Accept");
        h.push(("Accept".to_string(), accept.to_string()));
        h
    }

    fn apply_headers(
        mut req: reqwest::blocking::RequestBuilder,
        headers: &[(String, String)],
    ) -> reqwest::blocking::RequestBuilder {
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
        req
    }

    fn list_skills_in_repo(&self, repo: &str, path: &str) -> Vec<SkillMeta> {
        let cache_key = format!("{repo}_{path}").replace('/', "_").replace(' ', "_");
        if let Some(cached) = read_index_cache(&cache_key) {
            return metas_from_cache(&cached);
        }

        let url = format!(
            "https://api.github.com/repos/{repo}/contents/{}",
            path.trim_end_matches('/')
        );
        let client = http_client();
        let req = Self::apply_headers(
            client.get(&url).timeout(Duration::from_secs(15)),
            &self.auth.get_headers(),
        )
        .header(reqwest::header::USER_AGENT, "hermes-skills-hub");
        let resp = match req.send() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        if resp.status().as_u16() != 200 {
            return Vec::new();
        }
        let entries: Value = match resp.json() {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let entries = match entries.as_array() {
            Some(a) => a.clone(),
            None => return Vec::new(),
        };

        let mut skills: Vec<SkillMeta> = Vec::new();
        for entry in &entries {
            if entry.get("type").and_then(|v| v.as_str()) != Some("dir") {
                continue;
            }
            let dir_name = entry.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if dir_name.starts_with('.') || dir_name.starts_with('_') {
                continue;
            }
            let prefix = path.trim_end_matches('/');
            let skill_identifier = if !prefix.is_empty() {
                format!("{repo}/{prefix}/{dir_name}")
            } else {
                format!("{repo}/{dir_name}")
            };
            if let Some(meta) = self.inspect(&skill_identifier) {
                skills.push(meta);
            }
        }

        let cache_data = Value::Array(skills.iter().map(github_meta_to_dict).collect());
        write_index_cache(&cache_key, &cache_data);
        skills
    }

    fn get_repo_tree(&self, repo: &str) -> Option<(String, Vec<Value>)> {
        if let Some(v) = self.tree_cache.borrow().get(repo) {
            return Some(v.clone());
        }
        let client = http_client();
        let headers = self.auth.get_headers();

        // Resolve default branch
        let req = Self::apply_headers(
            client
                .get(format!("https://api.github.com/repos/{repo}"))
                .timeout(Duration::from_secs(15)),
            &headers,
        )
        .header(reqwest::header::USER_AGENT, "hermes-skills-hub");
        let resp = req.send().ok()?;
        if resp.status().as_u16() != 200 {
            self.check_rate_limit_response(resp.status().as_u16(), &resp);
            return None;
        }
        let repo_json: Value = resp.json().ok()?;
        let default_branch = repo_json
            .get("default_branch")
            .and_then(|v| v.as_str())
            .unwrap_or("main")
            .to_string();

        // Fetch recursive tree
        let req = Self::apply_headers(
            client
                .get(format!(
                    "https://api.github.com/repos/{repo}/git/trees/{default_branch}"
                ))
                .query(&[("recursive", "1")])
                .timeout(Duration::from_secs(30)),
            &headers,
        )
        .header(reqwest::header::USER_AGENT, "hermes-skills-hub");
        let resp = req.send().ok()?;
        if resp.status().as_u16() != 200 {
            self.check_rate_limit_response(resp.status().as_u16(), &resp);
            return None;
        }
        let tree_data: Value = resp.json().ok()?;
        if tree_data.get("truncated").and_then(|v| v.as_bool()) == Some(true) {
            log::debug!("Git tree truncated for {repo}, cannot cache");
            return None;
        }
        let entries = tree_data
            .get("tree")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        self.tree_cache
            .borrow_mut()
            .insert(repo.to_string(), (default_branch.clone(), entries.clone()));
        Some((default_branch, entries))
    }

    fn check_rate_limit_response(&self, status: u16, resp: &reqwest::blocking::Response) {
        if status == 403 {
            let remaining = resp
                .headers()
                .get("X-RateLimit-Remaining")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            if remaining == "0" {
                self.rate_limited.set(true);
                log::warn!(
                    "GitHub API rate limit exhausted (unauthenticated: 60 req/hr). \
                     Set GITHUB_TOKEN or install the gh CLI to raise the limit to 5,000/hr."
                );
            }
        }
    }

    fn download_directory(&self, repo: &str, path: &str) -> BTreeMap<String, FileContent> {
        if let Some(files) = self.download_directory_via_tree(repo, path) {
            return files;
        }
        log::debug!("Tree API unavailable for {repo}/{path}, falling back to Contents API");
        self.download_directory_recursive(repo, path)
    }

    fn download_directory_via_tree(
        &self,
        repo: &str,
        path: &str,
    ) -> Option<BTreeMap<String, FileContent>> {
        let path = path.trim_end_matches('/');
        let (_branch, tree_entries) = self.get_repo_tree(repo)?;

        let prefix = format!("{path}/");
        let has_entries = tree_entries.iter().any(|item| {
            item.get("path")
                .and_then(|v| v.as_str())
                .map(|p| p.starts_with(&prefix))
                .unwrap_or(false)
        });
        if !has_entries {
            // Path definitively doesn't exist; return empty (not None) to skip fallback.
            return Some(BTreeMap::new());
        }

        let mut files: BTreeMap<String, FileContent> = BTreeMap::new();
        for item in &tree_entries {
            if item.get("type").and_then(|v| v.as_str()) != Some("blob") {
                continue;
            }
            let item_path = item.get("path").and_then(|v| v.as_str()).unwrap_or("");
            if !item_path.starts_with(&prefix) {
                continue;
            }
            let rel_path = &item_path[prefix.len()..];
            match self.fetch_file_content(repo, item_path) {
                Some(content) => {
                    files.insert(rel_path.to_string(), FileContent::Text(content));
                }
                None => {
                    log::debug!("Skipped file (fetch failed): {repo}/{item_path}");
                }
            }
        }

        if files.is_empty() {
            None
        } else {
            Some(files)
        }
    }

    fn download_directory_recursive(
        &self,
        repo: &str,
        path: &str,
    ) -> BTreeMap<String, FileContent> {
        let mut files: BTreeMap<String, FileContent> = BTreeMap::new();
        let url = format!(
            "https://api.github.com/repos/{repo}/contents/{}",
            path.trim_end_matches('/')
        );
        let client = http_client();
        let req = Self::apply_headers(
            client.get(&url).timeout(Duration::from_secs(15)),
            &self.auth.get_headers(),
        )
        .header(reqwest::header::USER_AGENT, "hermes-skills-hub");
        let resp = match req.send() {
            Ok(r) => r,
            Err(_) => return files,
        };
        if resp.status().as_u16() != 200 {
            log::debug!(
                "Contents API returned {} for {repo}/{path}",
                resp.status().as_u16()
            );
            return files;
        }
        let entries: Value = match resp.json() {
            Ok(v) => v,
            Err(_) => return files,
        };
        let entries = match entries.as_array() {
            Some(a) => a.clone(),
            None => return files,
        };

        for entry in &entries {
            let name = entry.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let entry_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let entry_path = entry.get("path").and_then(|v| v.as_str()).unwrap_or("");
            if entry_type == "file" {
                if let Some(content) = self.fetch_file_content(repo, entry_path) {
                    files.insert(name.to_string(), FileContent::Text(content));
                }
            } else if entry_type == "dir" {
                let sub_files = self.download_directory_recursive(repo, entry_path);
                if sub_files.is_empty() {
                    log::debug!("Empty or failed subdirectory: {repo}/{entry_path}");
                }
                for (sub_name, sub_content) in sub_files {
                    files.insert(format!("{name}/{sub_name}"), sub_content);
                }
            }
        }
        files
    }

    fn find_skill_in_repo_tree(&self, repo: &str, skill_name: &str) -> Option<String> {
        let (_branch, tree_entries) = self.get_repo_tree(repo)?;
        let suffix = format!("/{skill_name}/SKILL.md");
        let exact = format!("{skill_name}/SKILL.md");
        for entry in &tree_entries {
            if entry.get("type").and_then(|v| v.as_str()) != Some("blob") {
                continue;
            }
            let path = entry.get("path").and_then(|v| v.as_str()).unwrap_or("");
            if path.ends_with(&suffix) || path == exact {
                let skill_dir = &path[..path.len() - "/SKILL.md".len()];
                return Some(format!("{repo}/{skill_dir}"));
            }
        }
        None
    }

    fn fetch_file_content(&self, repo: &str, path: &str) -> Option<String> {
        let url = format!("https://api.github.com/repos/{repo}/contents/{path}");
        let client = http_client();
        let headers = self.get_headers_with_accept("application/vnd.github.v3.raw");
        let req = Self::apply_headers(
            client.get(&url).timeout(Duration::from_secs(15)),
            &headers,
        )
        .header(reqwest::header::USER_AGENT, "hermes-skills-hub");
        match req.send() {
            Ok(resp) => {
                let status = resp.status().as_u16();
                if status == 200 {
                    return resp.text().ok();
                }
                self.check_rate_limit_response(status, &resp);
            }
            Err(e) => log::debug!("GitHub contents API fetch failed: {e}"),
        }
        None
    }
}

fn github_meta_to_dict(meta: &SkillMeta) -> Value {
    serde_json::json!({
        "name": meta.name,
        "description": meta.description,
        "source": meta.source,
        "identifier": meta.identifier,
        "trust_level": meta.trust_level,
        "repo": meta.repo,
        "path": meta.path,
        "tags": meta.tags,
    })
}

impl SkillSource for GitHubSource {
    fn source_id(&self) -> String {
        "github".to_string()
    }

    fn trust_level_for(&self, identifier: &str) -> String {
        Self::trust_level_for_impl(identifier)
    }

    fn search(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        let mut results: Vec<SkillMeta> = Vec::new();
        let query_lower = query.to_lowercase();

        for tap in &self.taps {
            let skills = self.list_skills_in_repo(&tap.repo, &tap.path);
            for skill in skills {
                let searchable = format!(
                    "{} {} {}",
                    skill.name,
                    skill.description,
                    skill.tags.join(" ")
                )
                .to_lowercase();
                if searchable.contains(&query_lower) {
                    results.push(skill);
                }
            }
        }

        // Deduplicate by name, preferring higher trust levels.
        let mut seen: HashMap<String, SkillMeta> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for r in results {
            match seen.get(&r.name) {
                None => {
                    order.push(r.name.clone());
                    seen.insert(r.name.clone(), r);
                }
                Some(existing) => {
                    if trust_rank(&r.trust_level) > trust_rank(&existing.trust_level) {
                        seen.insert(r.name.clone(), r);
                    }
                }
            }
        }
        let deduped: Vec<SkillMeta> = order.into_iter().filter_map(|n| seen.remove(&n)).collect();
        deduped.into_iter().take(limit).collect()
    }

    fn fetch(&self, identifier: &str) -> Option<SkillBundle> {
        let parts: Vec<&str> = identifier.splitn(3, '/').collect();
        if parts.len() < 3 {
            return None;
        }
        let repo = format!("{}/{}", parts[0], parts[1]);
        let skill_path = parts[2];

        let files = self.download_directory(&repo, skill_path);
        if files.is_empty() || !files.contains_key("SKILL.md") {
            return None;
        }
        let skill_name = skill_path
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(skill_path)
            .to_string();
        let trust = Self::trust_level_for_impl(identifier);
        Some(SkillBundle::new(
            skill_name, files, "github", identifier, trust,
        ))
    }

    fn inspect(&self, identifier: &str) -> Option<SkillMeta> {
        let parts: Vec<&str> = identifier.splitn(3, '/').collect();
        if parts.len() < 3 {
            return None;
        }
        let repo = format!("{}/{}", parts[0], parts[1]);
        let skill_path = parts[2].trim_end_matches('/');
        let skill_md_path = format!("{skill_path}/SKILL.md");

        let content = self.fetch_file_content(&repo, &skill_md_path)?;
        let fm = parse_frontmatter_quick(&content);
        let skill_name = {
            let n = yaml_str(yaml_get(&fm, "name"));
            if n.is_empty() {
                skill_path.rsplit('/').next().unwrap_or(skill_path).to_string()
            } else {
                n
            }
        };
        let description = yaml_str(yaml_get(&fm, "description"));
        let tags = extract_tags_from_fm(&fm);

        Some(SkillMeta {
            name: skill_name,
            description,
            source: "github".to_string(),
            identifier: identifier.to_string(),
            trust_level: Self::trust_level_for_impl(identifier),
            repo: Some(repo),
            path: Some(skill_path.to_string()),
            tags,
            extra: BTreeMap::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// URL parsing helper (path component of an HTTP(S) URL)
// ---------------------------------------------------------------------------

fn url_path(raw: &str) -> Option<String> {
    url::Url::parse(raw).ok().map(|u| u.path().to_string())
}

fn fetch_text_simple(url: &str, timeout_secs: u64) -> Option<String> {
    let client = http_client();
    match client
        .get(url)
        .timeout(Duration::from_secs(timeout_secs))
        .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
        .send()
    {
        Ok(resp) if resp.status().as_u16() == 200 => resp.text().ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Well-known Agent Skills endpoint source adapter
// ---------------------------------------------------------------------------

/// Read skills from a domain exposing /.well-known/skills/index.json.
pub struct WellKnownSkillSource;

const WK_BASE_PATH: &str = "/.well-known/skills";

struct WkParsedIdent {
    index_url: String,
    base_url: String,
    skill_name: String,
    skill_url: String,
}

struct WkParsedIndex {
    index_url: String,
    base_url: String,
    skills: Vec<Value>,
}

impl WellKnownSkillSource {
    fn query_to_index_url(query: &str) -> Option<String> {
        let query = query.trim();
        if !(query.starts_with("http://") || query.starts_with("https://")) {
            return None;
        }
        if query.ends_with("/index.json") {
            return Some(query.to_string());
        }
        let marker = format!("{WK_BASE_PATH}/");
        if query.contains(&marker) {
            let base = query.split(&marker).next().unwrap_or("");
            let base_url = format!("{base}{WK_BASE_PATH}");
            return Some(format!("{base_url}/index.json"));
        }
        Some(format!("{}{WK_BASE_PATH}/index.json", query.trim_end_matches('/')))
    }

    fn parse_identifier(identifier: &str) -> Option<WkParsedIdent> {
        let raw = identifier
            .strip_prefix("well-known:")
            .unwrap_or(identifier);
        if !(raw.starts_with("http://") || raw.starts_with("https://")) {
            return None;
        }
        // Drop fragment (mirror urlparse._replace(fragment="")).
        let (clean_url, fragment) = match raw.split_once('#') {
            Some((u, f)) => (u.to_string(), f.to_string()),
            None => (raw.to_string(), String::new()),
        };

        if clean_url.ends_with("/index.json") {
            if fragment.is_empty() {
                return None;
            }
            let base_url = clean_url[..clean_url.len() - "/index.json".len()].to_string();
            let skill_name = fragment;
            let skill_url = format!("{base_url}/{skill_name}");
            return Some(WkParsedIdent {
                index_url: clean_url,
                base_url,
                skill_name,
                skill_url,
            });
        }

        let skill_url = if clean_url.ends_with("/SKILL.md") {
            clean_url[..clean_url.len() - "/SKILL.md".len()].to_string()
        } else {
            clean_url.trim_end_matches('/').to_string()
        };

        let marker = format!("{WK_BASE_PATH}/");
        if !skill_url.contains(&marker) {
            return None;
        }
        let (base_url, skill_name) = skill_url.rsplit_once('/')?;
        Some(WkParsedIdent {
            index_url: format!("{base_url}/index.json"),
            base_url: base_url.to_string(),
            skill_name: skill_name.to_string(),
            skill_url: skill_url.clone(),
        })
    }

    fn parse_index(&self, index_url: &str) -> Option<WkParsedIndex> {
        let cache_key = format!("well_known_index_{}", md5_hex(index_url));
        if let Some(cached) = read_index_cache(&cache_key) {
            if cached.is_object() && cached.get("skills").map(|s| s.is_array()).unwrap_or(false) {
                let skills = cached
                    .get("skills")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                return Some(WkParsedIndex {
                    index_url: cached
                        .get("index_url")
                        .and_then(|v| v.as_str())
                        .unwrap_or(index_url)
                        .to_string(),
                    base_url: cached
                        .get("base_url")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    skills,
                });
            }
        }

        let client = http_client();
        let resp = client
            .get(index_url)
            .timeout(Duration::from_secs(20))
            .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
            .send()
            .ok()?;
        if resp.status().as_u16() != 200 {
            return None;
        }
        let data: Value = resp.json().ok()?;
        let skills = if data.is_object() {
            data.get("skills").and_then(|v| v.as_array()).cloned().unwrap_or_default()
        } else {
            return None;
        };
        let base_url = index_url[..index_url.len() - "/index.json".len()].to_string();
        let cache = serde_json::json!({
            "index_url": index_url,
            "base_url": base_url,
            "skills": skills,
        });
        write_index_cache(&cache_key, &cache);
        Some(WkParsedIndex {
            index_url: index_url.to_string(),
            base_url,
            skills,
        })
    }

    fn index_entry(&self, index_url: &str, skill_name: &str) -> Option<Value> {
        let parsed = self.parse_index(index_url)?;
        for entry in &parsed.skills {
            if entry.is_object() && entry.get("name").and_then(|v| v.as_str()) == Some(skill_name) {
                return Some(entry.clone());
            }
        }
        None
    }

    fn wrap_identifier(base_url: &str, skill_name: &str) -> String {
        format!("well-known:{}/{}", base_url.trim_end_matches('/'), skill_name)
    }
}

fn json_files_list(entry: &Value) -> Vec<String> {
    match entry.get("files").and_then(|v| v.as_array()) {
        Some(arr) if !arr.is_empty() => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => vec!["SKILL.md".to_string()],
    }
}

impl SkillSource for WellKnownSkillSource {
    fn source_id(&self) -> String {
        "well-known".to_string()
    }

    fn search(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        let index_url = match Self::query_to_index_url(query) {
            Some(u) => u,
            None => return Vec::new(),
        };
        let parsed = match self.parse_index(&index_url) {
            Some(p) => p,
            None => return Vec::new(),
        };
        let mut results = Vec::new();
        for entry in parsed.skills.iter().take(limit) {
            let name = match entry.get("name").and_then(|v| v.as_str()) {
                Some(n) if !n.is_empty() => n.to_string(),
                _ => continue,
            };
            let description = entry
                .get("description")
                .map(yaml_json_str)
                .unwrap_or_default();
            let files = json_files_list(entry);
            let mut extra = BTreeMap::new();
            extra.insert("index_url".to_string(), Value::String(parsed.index_url.clone()));
            extra.insert("base_url".to_string(), Value::String(parsed.base_url.clone()));
            extra.insert(
                "files".to_string(),
                Value::Array(files.iter().map(|f| Value::String(f.clone())).collect()),
            );
            results.push(SkillMeta {
                name: name.clone(),
                description,
                source: "well-known".to_string(),
                identifier: Self::wrap_identifier(&parsed.base_url, &name),
                trust_level: "community".to_string(),
                repo: None,
                path: Some(name),
                tags: Vec::new(),
                extra,
            });
        }
        results
    }

    fn inspect(&self, identifier: &str) -> Option<SkillMeta> {
        let parsed = Self::parse_identifier(identifier)?;
        let entry = self.index_entry(&parsed.index_url, &parsed.skill_name)?;
        let skill_md = fetch_text_simple(&format!("{}/SKILL.md", parsed.skill_url), 20)?;
        let fm = parse_frontmatter_quick(&skill_md);
        let fm_desc = yaml_str(yaml_get(&fm, "description"));
        let entry_desc = entry.get("description").map(yaml_json_str).unwrap_or_default();
        let description = if !fm_desc.is_empty() {
            fm_desc
        } else {
            entry_desc
        };
        let fm_name = yaml_str(yaml_get(&fm, "name"));
        let name = if !fm_name.is_empty() { fm_name } else { parsed.skill_name.clone() };
        let files = json_files_list(&entry);
        let mut extra = BTreeMap::new();
        extra.insert("index_url".to_string(), Value::String(parsed.index_url.clone()));
        extra.insert("base_url".to_string(), Value::String(parsed.base_url.clone()));
        extra.insert(
            "files".to_string(),
            Value::Array(files.iter().map(|f| Value::String(f.clone())).collect()),
        );
        extra.insert("endpoint".to_string(), Value::String(parsed.skill_url.clone()));
        Some(SkillMeta {
            name,
            description,
            source: "well-known".to_string(),
            identifier: Self::wrap_identifier(&parsed.base_url, &parsed.skill_name),
            trust_level: "community".to_string(),
            repo: None,
            path: Some(parsed.skill_name),
            tags: Vec::new(),
            extra,
        })
    }

    fn fetch(&self, identifier: &str) -> Option<SkillBundle> {
        let parsed = Self::parse_identifier(identifier)?;
        let skill_name = match validate_skill_name(&parsed.skill_name) {
            Ok(n) => n,
            Err(_) => {
                log::warn!("Well-known skill identifier contained unsafe skill name: {identifier}");
                return None;
            }
        };
        let entry = self.index_entry(&parsed.index_url, &parsed.skill_name)?;
        let files = json_files_list(&entry);

        let mut downloaded: BTreeMap<String, FileContent> = BTreeMap::new();
        for rel_path in &files {
            if rel_path.is_empty() {
                continue;
            }
            let safe_rel = match validate_bundle_rel_path(rel_path) {
                Ok(s) => s,
                Err(_) => {
                    log::warn!(
                        "Well-known skill {identifier} advertised unsafe file path: {rel_path:?}"
                    );
                    return None;
                }
            };
            let text = fetch_text_simple(&format!("{}/{}", parsed.skill_url, safe_rel), 20)?;
            downloaded.insert(safe_rel, FileContent::Text(text));
        }
        if !downloaded.contains_key("SKILL.md") {
            return None;
        }
        let mut bundle = SkillBundle::new(
            skill_name.clone(),
            downloaded,
            "well-known",
            Self::wrap_identifier(&parsed.base_url, &skill_name),
            "community",
        );
        bundle.metadata.insert("index_url".into(), Value::String(parsed.index_url));
        bundle.metadata.insert("base_url".into(), Value::String(parsed.base_url));
        bundle.metadata.insert("endpoint".into(), Value::String(parsed.skill_url));
        bundle.metadata.insert(
            "files".into(),
            Value::Array(files.iter().map(|f| Value::String(f.clone())).collect()),
        );
        Some(bundle)
    }
}

fn yaml_json_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Direct URL source adapter
// ---------------------------------------------------------------------------

/// Fetch a single-file SKILL.md skill directly from an HTTP(S) URL.
pub struct UrlSource;

impl UrlSource {
    fn matches(identifier: &str) -> bool {
        let ident = identifier.trim();
        let lower = ident.to_lowercase();
        if !(lower.starts_with("http://") || lower.starts_with("https://")) {
            return false;
        }
        if ident.contains("/.well-known/skills/") || ident.trim_end_matches('/').ends_with("/index.json") {
            return false;
        }
        match url_path(ident) {
            Some(p) => p.to_lowercase().ends_with(".md"),
            None => false,
        }
    }

    fn is_valid_skill_name(name: Option<&str>) -> bool {
        let name = match name {
            Some(n) => n,
            None => return false,
        };
        let candidate = name.trim().to_lowercase();
        if candidate.is_empty()
            || matches!(candidate.as_str(), "skill" | "readme" | "index" | "unnamed-skill")
        {
            return false;
        }
        valid_name_re().is_match(&candidate)
    }

    fn resolve_skill_name(fm: &serde_yaml::Mapping, url: &str) -> Option<String> {
        // 1. Frontmatter name.
        let fm_name = yaml_str(yaml_get(fm, "name"));
        if !fm_name.is_empty() && Self::is_valid_skill_name(Some(&fm_name)) {
            return Some(fm_name.trim().to_string());
        }
        // 2. URL-slug heuristic.
        let path = url_path(url)?;
        let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        if let Some(last) = parts.last() {
            if last.to_lowercase() == "skill.md" && parts.len() >= 2 {
                let candidate = parts[parts.len() - 2];
                if Self::is_valid_skill_name(Some(candidate)) {
                    return Some(candidate.to_string());
                }
            }
        }
        if let Some(last) = parts.last() {
            let candidate = strip_md_suffix(last);
            if Self::is_valid_skill_name(Some(&candidate)) {
                return Some(candidate);
            }
        }
        None
    }
}

fn valid_name_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[a-z][a-z0-9_-]*$").unwrap())
}

fn strip_md_suffix(s: &str) -> String {
    let re = md_suffix_re();
    re.replace(s, "").to_string()
}

fn md_suffix_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\.md$").unwrap())
}

impl SkillSource for UrlSource {
    fn source_id(&self) -> String {
        "url".to_string()
    }

    fn search(&self, _query: &str, _limit: usize) -> Vec<SkillMeta> {
        Vec::new()
    }

    fn inspect(&self, identifier: &str) -> Option<SkillMeta> {
        if !Self::matches(identifier) {
            return None;
        }
        let url = identifier.trim();
        let text = fetch_text_simple(url, 20)?;
        let fm = parse_frontmatter_quick(&text);
        let name = Self::resolve_skill_name(&fm, url);
        let description = yaml_str(yaml_get(&fm, "description"));
        let tags = {
            // metadata.hermes.tags only (matches UrlSource.inspect).
            if let Some(serde_yaml::Value::Mapping(meta)) = yaml_get(&fm, "metadata") {
                if let Some(serde_yaml::Value::Mapping(hermes)) = yaml_get(meta, "hermes") {
                    if let Some(serde_yaml::Value::Sequence(seq)) = yaml_get(hermes, "tags") {
                        seq.iter().map(yaml_value_to_string).collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        };
        let awaiting = name.is_none();
        let resolved_name = name.unwrap_or_default();
        let mut extra = BTreeMap::new();
        extra.insert("url".to_string(), Value::String(url.to_string()));
        extra.insert("awaiting_name".to_string(), Value::Bool(awaiting));
        Some(SkillMeta {
            name: resolved_name.clone(),
            description,
            source: "url".to_string(),
            identifier: url.to_string(),
            trust_level: "community".to_string(),
            repo: None,
            path: Some(resolved_name),
            tags,
            extra,
        })
    }

    fn fetch(&self, identifier: &str) -> Option<SkillBundle> {
        if !Self::matches(identifier) {
            return None;
        }
        let url = identifier.trim();
        let text = fetch_text_simple(url, 20)?;
        let fm = parse_frontmatter_quick(&text);
        let name = Self::resolve_skill_name(&fm, url);

        let skill_name = match &name {
            Some(n) => match validate_skill_name(n) {
                Ok(s) => s,
                Err(_) => {
                    log::warn!("URL skill {url} produced unsafe skill name: {n:?}");
                    return None;
                }
            },
            None => String::new(),
        };
        let mut files = BTreeMap::new();
        files.insert("SKILL.md".to_string(), FileContent::Text(text));
        let mut bundle = SkillBundle::new(skill_name.clone(), files, "url", url, "community");
        bundle.metadata.insert("url".into(), Value::String(url.to_string()));
        bundle.metadata.insert("awaiting_name".into(), Value::Bool(skill_name.is_empty()));
        Some(bundle)
    }
}

// ---------------------------------------------------------------------------
// Lock file management
// ---------------------------------------------------------------------------

/// Manages skills/.hub/lock.json — tracks provenance of installed hub skills.
pub struct HubLockFile {
    pub path: PathBuf,
}

impl Default for HubLockFile {
    fn default() -> Self {
        HubLockFile { path: lock_file_path() }
    }
}

impl HubLockFile {
    pub fn new(path: PathBuf) -> Self {
        HubLockFile { path }
    }

    pub fn load(&self) -> Value {
        if !self.path.exists() {
            return serde_json::json!({"version": 1, "installed": {}});
        }
        match std::fs::read_to_string(&self.path) {
            Ok(s) => serde_json::from_str(&s)
                .unwrap_or_else(|_| serde_json::json!({"version": 1, "installed": {}})),
            Err(_) => serde_json::json!({"version": 1, "installed": {}}),
        }
    }

    pub fn save(&self, data: &Value) {
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut s) = serde_json::to_string_pretty(data) {
            s.push('\n');
            let _ = std::fs::write(&self.path, s);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_install(
        &self,
        name: &str,
        source: &str,
        identifier: &str,
        trust_level: &str,
        scan_verdict: &str,
        skill_hash: &str,
        install_path: &str,
        files: &[String],
        metadata: Option<&BTreeMap<String, Value>>,
    ) {
        let mut data = self.load();
        let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false);
        let meta_value = match metadata {
            Some(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            None => Value::Object(Default::default()),
        };
        let entry = serde_json::json!({
            "source": source,
            "identifier": identifier,
            "trust_level": trust_level,
            "scan_verdict": scan_verdict,
            "content_hash": skill_hash,
            "install_path": install_path,
            "files": files,
            "metadata": meta_value,
            "installed_at": now,
            "updated_at": now,
        });
        if let Some(installed) = data
            .get_mut("installed")
            .and_then(|v| v.as_object_mut())
        {
            installed.insert(name.to_string(), entry);
        }
        self.save(&data);
    }

    pub fn record_uninstall(&self, name: &str) {
        let mut data = self.load();
        if let Some(installed) = data.get_mut("installed").and_then(|v| v.as_object_mut()) {
            installed.remove(name);
        }
        self.save(&data);
    }

    pub fn get_installed(&self, name: &str) -> Option<Value> {
        let data = self.load();
        data.get("installed")
            .and_then(|v| v.as_object())
            .and_then(|o| o.get(name))
            .cloned()
    }

    pub fn list_installed(&self) -> Vec<Value> {
        let data = self.load();
        let mut result = Vec::new();
        if let Some(installed) = data.get("installed").and_then(|v| v.as_object()) {
            for (name, entry) in installed {
                if let Some(obj) = entry.as_object() {
                    let mut merged = serde_json::Map::new();
                    merged.insert("name".to_string(), Value::String(name.clone()));
                    for (k, v) in obj {
                        merged.insert(k.clone(), v.clone());
                    }
                    result.push(Value::Object(merged));
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Taps management
// ---------------------------------------------------------------------------

/// Manages the taps.json file — custom GitHub repo sources.
pub struct TapsManager {
    pub path: PathBuf,
}

impl Default for TapsManager {
    fn default() -> Self {
        TapsManager { path: taps_file_path() }
    }
}

impl TapsManager {
    pub fn new(path: PathBuf) -> Self {
        TapsManager { path }
    }

    pub fn load(&self) -> Vec<Value> {
        if !self.path.exists() {
            return Vec::new();
        }
        let text = match std::fs::read_to_string(&self.path) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let data: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        data.get("taps")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
    }

    pub fn save(&self, taps: &[Value]) {
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let wrapped = serde_json::json!({ "taps": taps });
        if let Ok(mut s) = serde_json::to_string_pretty(&wrapped) {
            s.push('\n');
            let _ = std::fs::write(&self.path, s);
        }
    }

    /// Add a tap. Returns false if it already exists.
    pub fn add(&self, repo: &str, path: &str) -> bool {
        let mut taps = self.load();
        if taps
            .iter()
            .any(|t| t.get("repo").and_then(|v| v.as_str()) == Some(repo))
        {
            return false;
        }
        taps.push(serde_json::json!({"repo": repo, "path": path}));
        self.save(&taps);
        true
    }

    /// Remove a tap by repo name. Returns false if not found.
    pub fn remove(&self, repo: &str) -> bool {
        let taps = self.load();
        let new_taps: Vec<Value> = taps
            .iter()
            .filter(|t| t.get("repo").and_then(|v| v.as_str()) != Some(repo))
            .cloned()
            .collect();
        if new_taps.len() == taps.len() {
            return false;
        }
        self.save(&new_taps);
        true
    }

    pub fn list_taps(&self) -> Vec<Value> {
        self.load()
    }

    /// As [`Tap`] structs for use by [`GitHubSource`].
    pub fn list_taps_typed(&self) -> Vec<Tap> {
        self.load()
            .iter()
            .filter_map(|t| {
                let repo = t.get("repo").and_then(|v| v.as_str())?.to_string();
                let path = t
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("skills/")
                    .to_string();
                Some(Tap { repo, path })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Audit log
// ---------------------------------------------------------------------------

/// Append a line to the audit log.
pub fn append_audit_log(
    action: &str,
    skill_name: &str,
    source: &str,
    trust_level: &str,
    verdict: &str,
    extra: &str,
) {
    let log_path = audit_log_path();
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let mut parts = vec![
        timestamp,
        action.to_string(),
        skill_name.to_string(),
        format!("{source}:{trust_level}"),
        verdict.to_string(),
    ];
    if !extra.is_empty() {
        parts.push(extra.to_string());
    }
    let line = parts.join(" ") + "\n";
    use std::io::Write;
    match std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(mut f) => {
            if let Err(e) = f.write_all(line.as_bytes()) {
                log::debug!("Could not write audit log: {e}");
            }
        }
        Err(e) => log::debug!("Could not write audit log: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Hub operations (high-level)
// ---------------------------------------------------------------------------

/// Create the .hub directory structure if it doesn't exist.
pub fn ensure_hub_dirs() {
    let _ = std::fs::create_dir_all(hub_dir());
    let _ = std::fs::create_dir_all(quarantine_dir());
    let _ = std::fs::create_dir_all(index_cache_dir());
    if !lock_file_path().exists() {
        let _ = std::fs::write(lock_file_path(), "{\"version\": 1, \"installed\": {}}\n");
    }
    if !audit_log_path().exists() {
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(audit_log_path());
    }
    if !taps_file_path().exists() {
        let _ = std::fs::write(taps_file_path(), "{\"taps\": []}\n");
    }
}

/// Write a skill bundle to the quarantine directory for scanning.
pub fn quarantine_bundle(bundle: &SkillBundle) -> Result<PathBuf, String> {
    ensure_hub_dirs();
    let skill_name = validate_skill_name(&bundle.name)?;
    let mut validated: Vec<(String, &FileContent)> = Vec::new();
    for (rel_path, content) in &bundle.files {
        let safe = validate_bundle_rel_path(rel_path)?;
        validated.push((safe, content));
    }
    let dest = quarantine_dir().join(&skill_name);
    if dest.exists() {
        let _ = std::fs::remove_dir_all(&dest);
    }
    std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;

    for (rel_path, content) in validated {
        let mut file_dest = dest.clone();
        for seg in rel_path.split('/') {
            file_dest.push(seg);
        }
        if let Some(parent) = file_dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        match content {
            FileContent::Bytes(b) => {
                std::fs::write(&file_dest, b).map_err(|e| e.to_string())?
            }
            FileContent::Text(s) => {
                std::fs::write(&file_dest, s.as_bytes()).map_err(|e| e.to_string())?
            }
        }
    }
    Ok(dest)
}

/// Move a scanned skill from quarantine into the skills directory.
pub fn install_from_quarantine(
    quarantine_path: &Path,
    skill_name: &str,
    category: &str,
    bundle: &SkillBundle,
    scan_result: &ScanResult,
) -> Result<PathBuf, String> {
    let safe_skill_name = validate_skill_name(skill_name)?;
    let safe_category = if !category.is_empty() {
        validate_category_name(category)?
    } else {
        String::new()
    };
    let quarantine_resolved = quarantine_path
        .canonicalize()
        .unwrap_or_else(|_| quarantine_path.to_path_buf());
    let quarantine_root = quarantine_dir()
        .canonicalize()
        .unwrap_or_else(|_| quarantine_dir());
    if !quarantine_resolved.starts_with(&quarantine_root) {
        return Err(format!("Unsafe quarantine path: {}", quarantine_path.display()));
    }

    let install_dir = if !safe_category.is_empty() {
        skills_dir().join(&safe_category).join(&safe_skill_name)
    } else {
        skills_dir().join(&safe_skill_name)
    };

    if install_dir.exists() {
        let _ = std::fs::remove_dir_all(&install_dir);
    }

    // Warn (but don't block) if SKILL.md is very large.
    let skill_md = quarantine_path.join("SKILL.md");
    if let Ok(meta) = std::fs::metadata(&skill_md) {
        let size = meta.len();
        if size > 100_000 {
            log::warn!(
                "Skill '{safe_skill_name}' has a large SKILL.md ({size} chars). \
                 Large skills consume significant context when loaded. \
                 Consider asking the author to split it into smaller files."
            );
        }
    }

    if let Some(parent) = install_dir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    move_dir(quarantine_path, &install_dir).map_err(|e| e.to_string())?;

    let lock = HubLockFile::default();
    let install_rel = install_dir
        .strip_prefix(skills_dir())
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| install_dir.to_string_lossy().to_string());
    let files: Vec<String> = bundle.files.keys().cloned().collect();
    let chash = content_hash(&install_dir);
    lock.record_install(
        &safe_skill_name,
        &bundle.source,
        &bundle.identifier,
        &bundle.trust_level,
        &scan_result.verdict,
        &chash,
        &install_rel,
        &files,
        Some(&bundle.metadata),
    );

    append_audit_log(
        "INSTALL",
        &safe_skill_name,
        &bundle.source,
        &bundle.trust_level,
        &scan_result.verdict,
        &chash,
    );

    Ok(install_dir)
}

fn move_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    match std::fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            copy_dir_recursive(src, dst)?;
            std::fs::remove_dir_all(src)
        }
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target)?;
        } else {
            std::fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

/// Remove a hub-installed skill. Refuses to remove builtins.
pub fn uninstall_skill(skill_name: &str) -> (bool, String) {
    let lock = HubLockFile::default();
    let entry = match lock.get_installed(skill_name) {
        Some(e) => e,
        None => {
            return (
                false,
                format!("'{skill_name}' is not a hub-installed skill (may be a builtin)"),
            )
        }
    };
    let install_path_rel = entry.get("install_path").and_then(|v| v.as_str()).unwrap_or("");
    let install_path = skills_dir().join(install_path_rel);
    if install_path.exists() {
        let _ = std::fs::remove_dir_all(&install_path);
    }
    lock.record_uninstall(skill_name);
    let source = entry.get("source").and_then(|v| v.as_str()).unwrap_or("");
    let trust = entry.get("trust_level").and_then(|v| v.as_str()).unwrap_or("");
    append_audit_log("UNINSTALL", skill_name, source, trust, "n/a", "user_request");
    (true, format!("Uninstalled '{skill_name}' from {install_path_rel}"))
}

/// Compute a deterministic hash for an in-memory skill bundle.
pub fn bundle_content_hash(bundle: &SkillBundle) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    // BTreeMap iterates in sorted key order, matching Python's sorted().
    for content in bundle.files.values() {
        hasher.update(content.as_bytes().as_ref());
    }
    let digest = hasher.finalize();
    let hex = digest.iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!("sha256:{}", &hex[..16])
}

pub fn source_matches(source: &dyn SkillSource, source_name: &str) -> bool {
    let normalized = if source_name == "skills.sh" {
        "skills-sh"
    } else {
        source_name
    };
    source.source_id() == normalized
}

/// Check installed hub skills for upstream changes.
pub fn check_for_skill_updates(
    name: Option<&str>,
    lock: Option<HubLockFile>,
    sources: &[Box<dyn SkillSource>],
) -> Vec<Value> {
    let lock = lock.unwrap_or_default();
    let mut installed = lock.list_installed();
    if let Some(n) = name {
        installed.retain(|e| e.get("name").and_then(|v| v.as_str()) == Some(n));
    }

    let mut results: Vec<Value> = Vec::new();
    for entry in &installed {
        let identifier = entry.get("identifier").and_then(|v| v.as_str()).unwrap_or("");
        let source_name = entry.get("source").and_then(|v| v.as_str()).unwrap_or("");
        let entry_name = entry.get("name").and_then(|v| v.as_str()).unwrap_or("");

        let matching: Vec<&Box<dyn SkillSource>> = sources
            .iter()
            .filter(|s| source_matches(s.as_ref(), source_name))
            .collect();
        let candidate_sources: Vec<&Box<dyn SkillSource>> = if matching.is_empty() {
            sources.iter().collect()
        } else {
            matching
        };

        let mut bundle: Option<SkillBundle> = None;
        for src in candidate_sources {
            bundle = src.fetch(identifier);
            if bundle.is_some() {
                break;
            }
        }

        let bundle = match bundle {
            Some(b) => b,
            None => {
                results.push(serde_json::json!({
                    "name": entry_name,
                    "identifier": identifier,
                    "source": source_name,
                    "status": "unavailable",
                }));
                continue;
            }
        };

        let current_hash = entry.get("content_hash").and_then(|v| v.as_str()).unwrap_or("");
        let latest_hash = bundle_content_hash(&bundle);
        let status = if current_hash == latest_hash {
            "up_to_date"
        } else {
            "update_available"
        };
        results.push(serde_json::json!({
            "name": entry_name,
            "identifier": identifier,
            "source": source_name,
            "status": status,
            "current_hash": current_hash,
            "latest_hash": latest_hash,
        }));
    }
    results
}

// ---------------------------------------------------------------------------
// HTML / regex helpers for skills.sh
// ---------------------------------------------------------------------------

fn strip_html(value: &str) -> String {
    strip_html_re().replace_all(value, "").to_string()
}

fn strip_html_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"<[^>]+>").unwrap())
}

// ---------------------------------------------------------------------------
// skills.sh source adapter
// ---------------------------------------------------------------------------

/// Discover skills via skills.sh and fetch content from the underlying GitHub repo.
pub struct SkillsShSource {
    pub auth: std::rc::Rc<GitHubAuth>,
    pub github: GitHubSource,
}

const SKILLS_SH_BASE_URL: &str = "https://skills.sh";

impl SkillsShSource {
    pub fn new(auth: std::rc::Rc<GitHubAuth>) -> Self {
        let github = GitHubSource::new(auth.clone(), None);
        SkillsShSource { auth, github }
    }

    fn search_url() -> String {
        format!("{SKILLS_SH_BASE_URL}/api/search")
    }

    fn skill_link_re() -> &'static Regex {
        static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r#"href=["']/(?P<id>(?:(?!agents/|_next/|api/)[^"'/]+/[^"'/]+/[^"'/]+))["']"#)
                .unwrap()
        })
    }

    fn install_cmd_re() -> &'static Regex {
        static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(
                r"(?i)npx\s+skills\s+add\s+(?P<repo>https?://github\.com/[^\s<]+|[^\s<]+)(?:\s+--skill\s+(?P<skill>[^\s<]+))?",
            )
            .unwrap()
        })
    }

    fn page_h1_re() -> &'static Regex {
        static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| Regex::new(r"(?is)<h1[^>]*>(?P<title>.*?)</h1>").unwrap())
    }

    fn prose_h1_re() -> &'static Regex {
        static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r#"(?is)<div[^>]*class=["'][^"']*prose[^"']*["'][^>]*>.*?<h1[^>]*>(?P<title>.*?)</h1>"#).unwrap()
        })
    }

    fn prose_p_re() -> &'static Regex {
        static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r#"(?is)<div[^>]*class=["'][^"']*prose[^"']*["'][^>]*>.*?<p[^>]*>(?P<body>.*?)</p>"#).unwrap()
        })
    }

    fn weekly_installs_re() -> &'static Regex {
        static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| {
            Regex::new(r#"(?s)Weekly Installs.*?children\\":\\"(?P<count>[0-9.,Kk]+)\\""#).unwrap()
        })
    }

    fn normalize_identifier(identifier: &str) -> String {
        for prefix in ["skills-sh/", "skills.sh/", "skils-sh/", "skils.sh/"] {
            if let Some(rest) = identifier.strip_prefix(prefix) {
                return rest.to_string();
            }
        }
        identifier.to_string()
    }

    fn wrap_identifier(identifier: &str) -> String {
        format!("skills-sh/{identifier}")
    }

    fn candidate_identifiers(identifier: &str) -> Vec<String> {
        let parts: Vec<&str> = identifier.splitn(3, '/').collect();
        if parts.len() < 3 {
            return vec![identifier.to_string()];
        }
        let repo = format!("{}/{}", parts[0], parts[1]);
        let skill_path = parts[2].trim_start_matches('/');
        let candidates = vec![
            format!("{repo}/{skill_path}"),
            format!("{repo}/skills/{skill_path}"),
            format!("{repo}/.agents/skills/{skill_path}"),
            format!("{repo}/.claude/skills/{skill_path}"),
        ];
        let mut seen = HashSet::new();
        let mut deduped = Vec::new();
        for c in candidates {
            if seen.insert(c.clone()) {
                deduped.push(c);
            }
        }
        deduped
    }

    fn extract_repo_slug(repo_value: &str) -> Option<String> {
        let mut v = repo_value.trim().to_string();
        if let Some(rest) = v.strip_prefix("https://github.com/") {
            v = rest.to_string();
        }
        let v = v.trim_matches('/');
        let parts: Vec<&str> = v.split('/').collect();
        if parts.len() >= 2 {
            Some(format!("{}/{}", parts[0], parts[1]))
        } else {
            None
        }
    }

    fn extract_first_match(re: &Regex, text: &str) -> Option<String> {
        let caps = re.captures(text)?;
        // First non-empty captured group (skip group 0).
        let mut value: Option<&str> = None;
        for i in 1..caps.len() {
            if let Some(m) = caps.get(i) {
                if !m.as_str().is_empty() {
                    value = Some(m.as_str());
                    break;
                }
            }
        }
        let value = value?;
        let stripped = strip_html(value);
        let trimmed = stripped.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    fn extract_weekly_installs(html: &str) -> Option<String> {
        Self::weekly_installs_re()
            .captures(html)
            .and_then(|c| c.name("count").map(|m| m.as_str().to_string()))
    }

    fn extract_security_audits(html: &str, _identifier: &str) -> BTreeMap<String, String> {
        let mut audits = BTreeMap::new();
        let verdict_re = Regex::new(r"(?i)(Pass|Warn|Fail)").unwrap();
        for audit in ["agent-trust-hub", "socket", "snyk"] {
            let needle = format!("/security/{audit}");
            if let Some(idx) = html.find(&needle) {
                let end = (idx + 500).min(html.len());
                let window = &html[idx..end];
                if let Some(m) = verdict_re.captures(window).and_then(|c| c.get(1)) {
                    audits.insert(audit.to_string(), title_case(m.as_str()));
                }
            }
        }
        audits
    }

    fn token_variants(value: Option<&str>) -> HashSet<String> {
        let value = match value {
            Some(v) if !v.is_empty() => v,
            _ => return HashSet::new(),
        };
        let plain = strip_html(value).trim().trim_matches('/').to_lowercase();
        if plain.is_empty() {
            return HashSet::new();
        }
        let base = plain.rsplit('/').next().unwrap_or("").to_string();
        let sanitize_re = Regex::new(r"[^a-z0-9/_-]+").unwrap();
        let sanitized = sanitize_re.replace_all(&plain, "-").trim_matches('-').to_string();
        let sanitized_base = if !sanitized.is_empty() {
            sanitized.rsplit('/').next().unwrap_or("").to_string()
        } else {
            String::new()
        };
        let slash_tail = plain.rsplit('/').next().unwrap_or("").to_string();
        let slash_tail_clean = {
            let t = slash_tail.trim_start_matches('@');
            t.rsplit('/').next().unwrap_or("").to_string()
        };
        let mut variants: HashSet<String> = HashSet::new();
        let candidates = [
            plain.clone(),
            plain.replace('_', "-"),
            plain.replace('/', "-"),
            base.clone(),
            base.replace('_', "-"),
            base.replace('/', "-"),
            sanitized.clone(),
            if !sanitized.is_empty() { sanitized.replace('/', "-") } else { String::new() },
            sanitized_base,
            slash_tail_clean.clone(),
            slash_tail_clean.replace('_', "-"),
        ];
        for c in candidates {
            if !c.is_empty() {
                variants.insert(c);
            }
        }
        variants
    }

    fn matches_skill_tokens(meta: &SkillMeta, skill_tokens: &[String]) -> bool {
        let mut candidates: HashSet<String> = HashSet::new();
        candidates.extend(Self::token_variants(Some(&meta.name)));
        candidates.extend(Self::token_variants(meta.path.as_deref()));
        let ident_tail = if !meta.identifier.is_empty() {
            meta.identifier.splitn(3, '/').last().map(|s| s.to_string())
        } else {
            None
        };
        candidates.extend(Self::token_variants(ident_tail.as_deref()));

        for token in skill_tokens {
            let variants = Self::token_variants(Some(token));
            if variants.intersection(&candidates).next().is_some() {
                return true;
            }
        }
        false
    }

    fn detail_to_metadata(&self, canonical: &str, detail: Option<&Value>) -> BTreeMap<String, Value> {
        let parts: Vec<&str> = canonical.splitn(3, '/').collect();
        let repo = if parts.len() >= 2 {
            format!("{}/{}", parts[0], parts[1])
        } else {
            String::new()
        };
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "detail_url".to_string(),
            Value::String(format!("{SKILLS_SH_BASE_URL}/{canonical}")),
        );
        if !repo.is_empty() {
            metadata.insert(
                "repo_url".to_string(),
                Value::String(format!("https://github.com/{repo}")),
            );
        }
        if let Some(d) = detail {
            for key in ["weekly_installs", "install_command", "repo_url", "detail_url", "security_audits"] {
                if let Some(v) = d.get(key) {
                    if !value_is_empty(v) {
                        metadata.insert(key.to_string(), v.clone());
                    }
                }
            }
        }
        metadata
    }

    fn meta_from_search_item(&self, item: &Value) -> Option<SkillMeta> {
        let item = item.as_object()?;
        let mut canonical = item.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
        let repo_field = item.get("source").and_then(|v| v.as_str());
        let skill_path_field = item.get("skillId").and_then(|v| v.as_str());

        let canonical_ok = canonical
            .as_ref()
            .map(|c| c.matches('/').count() >= 2)
            .unwrap_or(false);
        if !canonical_ok {
            match (repo_field, skill_path_field) {
                (Some(r), Some(sp)) => canonical = Some(format!("{r}/{sp}")),
                _ => return None,
            }
        }
        let canonical = canonical?;
        let parts: Vec<&str> = canonical.splitn(3, '/').collect();
        if parts.len() < 3 {
            return None;
        }
        let repo = format!("{}/{}", parts[0], parts[1]);
        let skill_path = parts[2].to_string();
        let installs = item.get("installs").and_then(|v| v.as_i64());
        let installs_label = match installs {
            Some(n) => format!(" · {} installs", group_thousands(n)),
            None => String::new(),
        };
        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| skill_path.rsplit('/').next().unwrap_or("").to_string());

        let mut extra = BTreeMap::new();
        extra.insert(
            "installs".to_string(),
            item.get("installs").cloned().unwrap_or(Value::Null),
        );
        extra.insert(
            "detail_url".to_string(),
            Value::String(format!("{SKILLS_SH_BASE_URL}/{canonical}")),
        );
        extra.insert(
            "repo_url".to_string(),
            Value::String(format!("https://github.com/{repo}")),
        );

        Some(SkillMeta {
            name,
            description: format!("Indexed by skills.sh from {repo}{installs_label}"),
            source: "skills.sh".to_string(),
            identifier: Self::wrap_identifier(&canonical),
            trust_level: GitHubSource::trust_level_for_impl(&canonical),
            repo: Some(repo),
            path: Some(skill_path),
            tags: Vec::new(),
            extra,
        })
    }

    fn fetch_detail_page(&self, identifier: &str) -> Option<Value> {
        let cache_key = format!("skills_sh_detail_{}", md5_hex(identifier));
        if let Some(cached) = read_index_cache(&cache_key) {
            if cached.is_object() {
                return Some(cached);
            }
        }
        let client = http_client();
        let resp = client
            .get(format!("{SKILLS_SH_BASE_URL}/{identifier}"))
            .timeout(Duration::from_secs(20))
            .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
            .send()
            .ok()?;
        if resp.status().as_u16() != 200 {
            return None;
        }
        let html = resp.text().ok()?;
        let detail = self.parse_detail_page(identifier, &html)?;
        write_index_cache(&cache_key, &detail);
        Some(detail)
    }

    fn parse_detail_page(&self, identifier: &str, html: &str) -> Option<Value> {
        let parts: Vec<&str> = identifier.splitn(3, '/').collect();
        if parts.len() < 3 {
            return None;
        }
        let default_repo = format!("{}/{}", parts[0], parts[1]);
        let skill_token = parts[2].to_string();
        let mut repo = default_repo.clone();
        let mut install_skill = skill_token.clone();

        let mut install_command: Option<String> = None;
        if let Some(caps) = Self::install_cmd_re().captures(html) {
            install_command = Some(caps.get(0).map(|m| m.as_str().trim().to_string()).unwrap_or_default());
            let repo_value = caps.name("repo").map(|m| m.as_str().trim()).unwrap_or("");
            if let Some(sk) = caps.name("skill") {
                install_skill = sk.as_str().trim().to_string();
            }
            if let Some(slug) = Self::extract_repo_slug(repo_value) {
                repo = slug;
            }
        }

        let page_title = Self::extract_first_match(Self::page_h1_re(), html);
        let body_title = Self::extract_first_match(Self::prose_h1_re(), html);
        let body_summary = Self::extract_first_match(Self::prose_p_re(), html);
        let weekly_installs = Self::extract_weekly_installs(html);
        let security_audits = Self::extract_security_audits(html, identifier);

        Some(serde_json::json!({
            "repo": repo,
            "install_skill": install_skill,
            "page_title": page_title,
            "body_title": body_title,
            "body_summary": body_summary,
            "weekly_installs": weekly_installs,
            "install_command": install_command,
            "repo_url": format!("https://github.com/{repo}"),
            "detail_url": format!("{SKILLS_SH_BASE_URL}/{identifier}"),
            "security_audits": security_audits,
        }))
    }

    fn discover_identifier(&self, identifier: &str, detail: Option<&Value>) -> Option<String> {
        let parts: Vec<&str> = identifier.splitn(3, '/').collect();
        if parts.len() < 3 {
            return None;
        }
        let default_repo = format!("{}/{}", parts[0], parts[1]);
        let repo = detail
            .and_then(|d| d.get("repo").and_then(|v| v.as_str()))
            .unwrap_or(&default_repo)
            .to_string();
        let skill_token = parts[2].rsplit('/').next().unwrap_or("").to_string();
        let mut tokens = vec![skill_token.clone()];
        if let Some(d) = detail {
            for k in ["install_skill", "page_title", "body_title"] {
                tokens.push(d.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string());
            }
        }

        let base_paths = ["skills/", ".agents/skills/", ".claude/skills/"];
        for base_path in base_paths {
            let skills = self.github.list_skills_in_repo(&repo, base_path);
            for meta in &skills {
                if Self::matches_skill_tokens(meta, &tokens) {
                    return Some(meta.identifier.clone());
                }
            }
        }

        if let Some(tree_result) = self.github.find_skill_in_repo_tree(&repo, &skill_token) {
            return Some(tree_result);
        }

        // Fallback: scan repo root.
        let client = http_client();
        let root_url = format!("https://api.github.com/repos/{repo}/contents/");
        let req = GitHubSource::apply_headers(
            client.get(&root_url).timeout(Duration::from_secs(15)),
            &self.github.auth.get_headers(),
        )
        .header(reqwest::header::USER_AGENT, "hermes-skills-hub");
        if let Ok(resp) = req.send() {
            if resp.status().as_u16() == 200 {
                if let Ok(Value::Array(entries)) = resp.json::<Value>() {
                    for entry in &entries {
                        if entry.get("type").and_then(|v| v.as_str()) != Some("dir") {
                            continue;
                        }
                        let dir_name = entry.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        if dir_name.starts_with('.') || dir_name.starts_with('_') {
                            continue;
                        }
                        if matches!(dir_name, "skills" | ".agents" | ".claude") {
                            continue;
                        }
                        let direct_id = format!("{repo}/{dir_name}/{skill_token}");
                        if let Some(meta) = self.github.inspect(&direct_id) {
                            return Some(meta.identifier);
                        }
                        let skills = self.github.list_skills_in_repo(&repo, &format!("{dir_name}/"));
                        for meta in &skills {
                            if Self::matches_skill_tokens(meta, &tokens) {
                                return Some(meta.identifier.clone());
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn resolve_github_meta(&self, identifier: &str, detail: Option<&Value>) -> Option<SkillMeta> {
        for candidate in Self::candidate_identifiers(identifier) {
            if let Some(meta) = self.github.inspect(&candidate) {
                return Some(meta);
            }
        }
        let resolved = self.discover_identifier(identifier, detail)?;
        self.github.inspect(&resolved)
    }

    fn finalize_inspect_meta(&self, mut meta: SkillMeta, canonical: &str, detail: Option<&Value>) -> SkillMeta {
        meta.source = "skills.sh".to_string();
        meta.identifier = Self::wrap_identifier(canonical);
        meta.trust_level = self.trust_level_for(canonical);
        let mut merged = meta.extra.clone();
        for (k, v) in self.detail_to_metadata(canonical, detail) {
            merged.insert(k, v);
        }
        meta.extra = merged;

        if let Some(d) = detail {
            let body_summary = d.get("body_summary").and_then(|v| v.as_str()).unwrap_or("");
            let weekly = d.get("weekly_installs").and_then(|v| v.as_str()).unwrap_or("");
            if !body_summary.is_empty() {
                meta.description = body_summary.to_string();
            } else if !meta.description.is_empty() && !weekly.is_empty() {
                meta.description = format!("{} · {} weekly installs on skills.sh", meta.description, weekly);
            }
        }
        meta
    }

    fn featured_skills(&self, limit: usize) -> Vec<SkillMeta> {
        let cache_key = "skills_sh_featured";
        if let Some(cached) = read_index_cache(cache_key) {
            return metas_from_cache(&cached).into_iter().take(limit).collect();
        }
        let client = http_client();
        let resp = match client
            .get(SKILLS_SH_BASE_URL)
            .timeout(Duration::from_secs(20))
            .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
            .send()
        {
            Ok(r) if r.status().as_u16() == 200 => r,
            _ => return Vec::new(),
        };
        let text = match resp.text() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let mut seen: HashSet<String> = HashSet::new();
        let mut results: Vec<SkillMeta> = Vec::new();
        for caps in Self::skill_link_re().captures_iter(&text) {
            let canonical = caps.name("id").map(|m| m.as_str().to_string()).unwrap_or_default();
            if seen.contains(&canonical) {
                continue;
            }
            seen.insert(canonical.clone());
            let parts: Vec<&str> = canonical.splitn(3, '/').collect();
            if parts.len() < 3 {
                continue;
            }
            let repo = format!("{}/{}", parts[0], parts[1]);
            let skill_path = parts[2].to_string();
            results.push(SkillMeta {
                name: skill_path.rsplit('/').next().unwrap_or("").to_string(),
                description: format!("Featured on skills.sh from {repo}"),
                source: "skills.sh".to_string(),
                identifier: Self::wrap_identifier(&canonical),
                trust_level: GitHubSource::trust_level_for_impl(&canonical),
                repo: Some(repo),
                path: Some(skill_path),
                tags: Vec::new(),
                extra: BTreeMap::new(),
            });
            if results.len() >= limit {
                break;
            }
        }
        let cache_data = Value::Array(results.iter().map(skill_meta_to_dict).collect());
        write_index_cache(cache_key, &cache_data);
        results
    }
}

fn value_is_empty(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::Bool(b) => !*b,
        Value::Number(n) => n.as_f64() == Some(0.0),
    }
}

fn group_thousands(n: i64) -> String {
    let s = n.abs().to_string();
    let bytes = s.as_bytes();
    let mut out = String::new();
    let len = bytes.len();
    for (i, c) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*c as char);
    }
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

fn title_case(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
        None => String::new(),
    }
}

impl SkillSource for SkillsShSource {
    fn source_id(&self) -> String {
        "skills-sh".to_string()
    }

    fn trust_level_for(&self, identifier: &str) -> String {
        self.github.trust_level_for(&Self::normalize_identifier(identifier))
    }

    fn search(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        if query.trim().is_empty() {
            return self.featured_skills(limit);
        }
        let cache_key = format!("skills_sh_search_{}", md5_hex(&format!("{query}|{limit}")));
        if let Some(cached) = read_index_cache(&cache_key) {
            return metas_from_cache(&cached).into_iter().take(limit).collect();
        }
        let client = http_client();
        let resp = match client
            .get(Self::search_url())
            .query(&[("q", query), ("limit", &limit.to_string())])
            .timeout(Duration::from_secs(20))
            .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
            .send()
        {
            Ok(r) if r.status().as_u16() == 200 => r,
            _ => return Vec::new(),
        };
        let data: Value = match resp.json() {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let items = if data.is_object() {
            data.get("skills").and_then(|v| v.as_array()).cloned().unwrap_or_default()
        } else {
            return Vec::new();
        };
        let mut results = Vec::new();
        for item in items.iter().take(limit) {
            if let Some(meta) = self.meta_from_search_item(item) {
                results.push(meta);
            }
        }
        let cache_data = Value::Array(results.iter().map(skill_meta_to_dict).collect());
        write_index_cache(&cache_key, &cache_data);
        results
    }

    fn fetch(&self, identifier: &str) -> Option<SkillBundle> {
        let canonical = Self::normalize_identifier(identifier);
        let detail = self.fetch_detail_page(&canonical);
        for candidate in Self::candidate_identifiers(&canonical) {
            if let Some(mut bundle) = self.github.fetch(&candidate) {
                bundle.source = "skills.sh".to_string();
                bundle.identifier = Self::wrap_identifier(&canonical);
                for (k, v) in self.detail_to_metadata(&canonical, detail.as_ref()) {
                    bundle.metadata.insert(k, v);
                }
                return Some(bundle);
            }
        }
        if let Some(resolved) = self.discover_identifier(&canonical, detail.as_ref()) {
            if let Some(mut bundle) = self.github.fetch(&resolved) {
                bundle.source = "skills.sh".to_string();
                bundle.identifier = Self::wrap_identifier(&canonical);
                for (k, v) in self.detail_to_metadata(&canonical, detail.as_ref()) {
                    bundle.metadata.insert(k, v);
                }
                return Some(bundle);
            }
        }
        None
    }

    fn inspect(&self, identifier: &str) -> Option<SkillMeta> {
        let canonical = Self::normalize_identifier(identifier);
        let detail = self.fetch_detail_page(&canonical);
        let meta = self.resolve_github_meta(&canonical, detail.as_ref())?;
        Some(self.finalize_inspect_meta(meta, &canonical, detail.as_ref()))
    }
}

// ---------------------------------------------------------------------------
// ClawHub source adapter
// ---------------------------------------------------------------------------

/// Fetch skills from ClawHub (clawhub.ai) via their HTTP API.
pub struct ClawHubSource;

const CLAWHUB_BASE_URL: &str = "https://clawhub.ai/api/v1";

impl ClawHubSource {
    fn normalize_tags(tags: &Value) -> Vec<String> {
        match tags {
            Value::Array(a) => a.iter().map(yaml_json_str).collect(),
            Value::Object(o) => o
                .keys()
                .filter(|k| k.as_str() != "latest")
                .cloned()
                .collect(),
            _ => Vec::new(),
        }
    }

    fn coerce_skill_payload(data: &Value) -> Option<Value> {
        let obj = data.as_object()?;
        if let Some(Value::Object(nested)) = obj.get("skill") {
            let mut merged = nested.clone();
            if let Some(lv) = obj.get("latestVersion") {
                if !lv.is_null() && !merged.contains_key("latestVersion") {
                    merged.insert("latestVersion".to_string(), lv.clone());
                }
            }
            return Some(Value::Object(merged));
        }
        Some(data.clone())
    }

    fn query_terms(query: &str) -> Vec<String> {
        let re = Regex::new(r"[^a-z0-9]+").unwrap();
        re.split(&query.to_lowercase())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .collect()
    }

    fn search_score(query: &str, meta: &SkillMeta) -> i32 {
        let query_norm = query.trim().to_lowercase();
        if query_norm.is_empty() {
            return 1;
        }
        let identifier = meta.identifier.to_lowercase();
        let name = meta.name.to_lowercase();
        let description = meta.description.to_lowercase();
        let normalized_identifier = Self::query_terms(&identifier).join(" ");
        let normalized_name = Self::query_terms(&name).join(" ");
        let query_terms = Self::query_terms(&query_norm);
        let identifier_terms = Self::query_terms(&identifier);
        let name_terms = Self::query_terms(&name);
        let mut score = 0;

        if query_norm == identifier {
            score += 140;
        }
        if query_norm == name {
            score += 130;
        }
        if normalized_identifier == query_norm {
            score += 125;
        }
        if normalized_name == query_norm {
            score += 120;
        }
        if normalized_identifier.starts_with(&query_norm) {
            score += 95;
        }
        if normalized_name.starts_with(&query_norm) {
            score += 90;
        }
        if !query_terms.is_empty()
            && identifier_terms.len() >= query_terms.len()
            && identifier_terms[..query_terms.len()] == query_terms[..]
        {
            score += 70;
        }
        if !query_terms.is_empty()
            && name_terms.len() >= query_terms.len()
            && name_terms[..query_terms.len()] == query_terms[..]
        {
            score += 65;
        }
        if identifier.contains(&query_norm) {
            score += 40;
        }
        if name.contains(&query_norm) {
            score += 35;
        }
        if description.contains(&query_norm) {
            score += 10;
        }
        for term in &query_terms {
            if identifier_terms.contains(term) {
                score += 15;
            }
            if name_terms.contains(term) {
                score += 12;
            }
            if description.contains(term) {
                score += 3;
            }
        }
        score
    }

    fn dedupe_results(results: Vec<SkillMeta>) -> Vec<SkillMeta> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut deduped = Vec::new();
        for result in results {
            let key = if !result.identifier.is_empty() {
                result.identifier.to_lowercase()
            } else {
                result.name.to_lowercase()
            };
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);
            deduped.push(result);
        }
        deduped
    }

    fn exact_slug_meta(&self, query: &str) -> Option<SkillMeta> {
        let slug = query.trim().rsplit('/').next().unwrap_or("").to_string();
        let query_terms = Self::query_terms(query);
        let mut candidates: Vec<String> = Vec::new();

        let slug_re = Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._-]*$").unwrap();
        if !slug.is_empty() && slug_re.is_match(&slug) {
            candidates.push(slug.clone());
        }
        if !query_terms.is_empty() {
            let base_slug = query_terms.join("-");
            if query_terms.len() >= 2 {
                candidates.extend([
                    format!("{base_slug}-agent"),
                    format!("{base_slug}-skill"),
                    format!("{base_slug}-tool"),
                    format!("{base_slug}-assistant"),
                    format!("{base_slug}-playbook"),
                    base_slug,
                ]);
            } else {
                candidates.push(base_slug);
            }
        }
        let mut seen: HashSet<String> = HashSet::new();
        for candidate in candidates {
            if seen.contains(&candidate) {
                continue;
            }
            seen.insert(candidate.clone());
            if let Some(meta) = self.inspect(&candidate) {
                return Some(meta);
            }
        }
        None
    }

    fn finalize_search_results(&self, query: &str, results: Vec<SkillMeta>, limit: usize) -> Vec<SkillMeta> {
        let query_norm = query.trim().to_string();
        if query_norm.is_empty() {
            return Self::dedupe_results(results).into_iter().take(limit).collect();
        }
        let mut filtered: Vec<SkillMeta> = results
            .iter()
            .filter(|m| Self::search_score(&query_norm, m) > 0)
            .cloned()
            .collect();
        filtered.sort_by(|a, b| {
            let sa = Self::search_score(&query_norm, a);
            let sb = Self::search_score(&query_norm, b);
            sb.cmp(&sa)
                .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
                .then_with(|| a.identifier.to_lowercase().cmp(&b.identifier.to_lowercase()))
        });
        filtered = Self::dedupe_results(filtered);

        if let Some(exact) = self.exact_slug_meta(&query_norm) {
            filtered.retain(|m| Self::search_score(&query_norm, m) >= 20);
            let mut combined = vec![exact];
            combined.extend(filtered);
            filtered = Self::dedupe_results(combined);
        }

        if !filtered.is_empty() {
            return filtered.into_iter().take(limit).collect();
        }

        let slug_query_re = Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$").unwrap();
        if slug_query_re.is_match(&query_norm) {
            return Vec::new();
        }
        Self::dedupe_results(results).into_iter().take(limit).collect()
    }

    fn search_catalog(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        let cache_key = format!("clawhub_search_catalog_v1_{}", md5_hex(&format!("{query}|{limit}")));
        if let Some(cached) = read_index_cache(&cache_key) {
            return metas_from_cache(&cached).into_iter().take(limit).collect();
        }
        let catalog = self.load_catalog_index();
        if catalog.is_empty() {
            return Vec::new();
        }
        let results = self.finalize_search_results(query, catalog, limit);
        let cache_data = Value::Array(results.iter().map(skill_meta_to_dict).collect());
        write_index_cache(&cache_key, &cache_data);
        results
    }

    fn load_catalog_index(&self) -> Vec<SkillMeta> {
        let cache_key = "clawhub_catalog_v1";
        if let Some(cached) = read_index_cache(cache_key) {
            return metas_from_cache(&cached);
        }
        let mut cursor: Option<String> = None;
        let mut results: Vec<SkillMeta> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let max_pages = 50;
        let client = http_client();

        for _ in 0..max_pages {
            let mut params: Vec<(String, String)> = vec![("limit".to_string(), "200".to_string())];
            if let Some(c) = &cursor {
                params.push(("cursor".to_string(), c.clone()));
            }
            let resp = match client
                .get(format!("{CLAWHUB_BASE_URL}/skills"))
                .query(&params)
                .timeout(Duration::from_secs(30))
                .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
                .send()
            {
                Ok(r) if r.status().as_u16() == 200 => r,
                _ => break,
            };
            let data: Value = match resp.json() {
                Ok(v) => v,
                Err(_) => break,
            };
            let items = if data.is_object() {
                data.get("items").and_then(|v| v.as_array()).cloned().unwrap_or_default()
            } else {
                Vec::new()
            };
            if items.is_empty() {
                break;
            }
            for item in &items {
                let slug = match item.get("slug").and_then(|v| v.as_str()) {
                    Some(s) if !s.is_empty() => s.to_string(),
                    _ => continue,
                };
                if seen.contains(&slug) {
                    continue;
                }
                seen.insert(slug.clone());
                let display_name = item
                    .get("displayName")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("name").and_then(|v| v.as_str()))
                    .filter(|s| !s.is_empty())
                    .unwrap_or(&slug)
                    .to_string();
                let summary = item
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("description").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();
                let tags = Self::normalize_tags(item.get("tags").unwrap_or(&Value::Null));
                results.push(SkillMeta {
                    name: display_name,
                    description: summary,
                    source: "clawhub".to_string(),
                    identifier: slug,
                    trust_level: "community".to_string(),
                    repo: None,
                    path: None,
                    tags,
                    extra: BTreeMap::new(),
                });
            }
            cursor = data.get("nextCursor").and_then(|v| v.as_str()).map(|s| s.to_string());
            if cursor.as_deref().map(|c| c.is_empty()).unwrap_or(true) {
                break;
            }
        }
        let cache_data = Value::Array(results.iter().map(skill_meta_to_dict).collect());
        write_index_cache(cache_key, &cache_data);
        results
    }

    fn get_json(&self, url: &str, timeout: u64) -> Option<Value> {
        let client = http_client();
        let resp = client
            .get(url)
            .timeout(Duration::from_secs(timeout))
            .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
            .send()
            .ok()?;
        if resp.status().as_u16() != 200 {
            return None;
        }
        resp.json().ok()
    }

    fn resolve_latest_version(&self, slug: &str, skill_data: &Value) -> Option<String> {
        if let Some(Value::Object(latest)) = skill_data.get("latestVersion").map(|v| v.clone()).as_ref() {
            if let Some(v) = latest.get("version").and_then(|v| v.as_str()) {
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
        if let Some(Value::Object(tags)) = skill_data.get("tags") {
            if let Some(v) = tags.get("latest").and_then(|v| v.as_str()) {
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
        let versions = self.get_json(&format!("{CLAWHUB_BASE_URL}/skills/{slug}/versions"), 20)?;
        if let Some(arr) = versions.as_array() {
            if let Some(Value::Object(first)) = arr.first() {
                if let Some(v) = first.get("version").and_then(|v| v.as_str()) {
                    if !v.is_empty() {
                        return Some(v.to_string());
                    }
                }
            }
        }
        None
    }

    fn extract_files(&self, version_data: &Value) -> BTreeMap<String, FileContent> {
        let mut files: BTreeMap<String, FileContent> = BTreeMap::new();
        let file_list = version_data.get("files");

        if let Some(Value::Object(map)) = file_list {
            for (k, v) in map {
                if let Some(s) = v.as_str() {
                    files.insert(k.clone(), FileContent::Text(s.to_string()));
                }
            }
            return files;
        }
        let arr = match file_list.and_then(|v| v.as_array()) {
            Some(a) => a,
            None => return files,
        };
        for file_meta in arr {
            let obj = match file_meta.as_object() {
                Some(o) => o,
                None => continue,
            };
            let fname = obj
                .get("path")
                .and_then(|v| v.as_str())
                .or_else(|| obj.get("name").and_then(|v| v.as_str()));
            let fname = match fname {
                Some(f) if !f.is_empty() => f.to_string(),
                _ => continue,
            };
            if let Some(content) = obj.get("content").and_then(|v| v.as_str()) {
                files.insert(fname, FileContent::Text(content.to_string()));
                continue;
            }
            let raw_url = obj
                .get("rawUrl")
                .and_then(|v| v.as_str())
                .or_else(|| obj.get("downloadUrl").and_then(|v| v.as_str()))
                .or_else(|| obj.get("url").and_then(|v| v.as_str()));
            if let Some(u) = raw_url {
                if u.starts_with("http") {
                    if let Some(content) = self.fetch_text(u) {
                        files.insert(fname, FileContent::Text(content));
                    }
                }
            }
        }
        files
    }

    fn download_zip(&self, slug: &str, version: &str) -> BTreeMap<String, FileContent> {
        let mut files: BTreeMap<String, FileContent> = BTreeMap::new();
        let max_retries = 3;
        let client = http_client();
        for attempt in 0..max_retries {
            let resp = match client
                .get(format!("{CLAWHUB_BASE_URL}/download"))
                .query(&[("slug", slug), ("version", version)])
                .timeout(Duration::from_secs(30))
                .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
                .send()
            {
                Ok(r) => r,
                Err(e) => {
                    log::debug!("ClawHub ZIP download failed for {slug} v{version}: {e}");
                    return files;
                }
            };
            let status = resp.status().as_u16();
            if status == 429 {
                let retry_after = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(5);
                let retry_after = retry_after.min(15);
                log::debug!(
                    "ClawHub download rate-limited for {slug}, retrying in {retry_after}s (attempt {}/{max_retries})",
                    attempt + 1
                );
                std::thread::sleep(Duration::from_secs(retry_after));
                continue;
            }
            if status != 200 {
                log::debug!("ClawHub ZIP download for {slug} v{version} returned {status}");
                return files;
            }
            let bytes = match resp.bytes() {
                Ok(b) => b,
                Err(e) => {
                    log::debug!("ClawHub ZIP download failed for {slug} v{version}: {e}");
                    return files;
                }
            };
            let reader = std::io::Cursor::new(bytes);
            let mut archive = match zip::ZipArchive::new(reader) {
                Ok(a) => a,
                Err(_) => {
                    log::warn!("ClawHub returned invalid ZIP for {slug} v{version}");
                    return files;
                }
            };
            for i in 0..archive.len() {
                let mut entry = match archive.by_index(i) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                if entry.is_dir() {
                    continue;
                }
                let raw_name = entry.name().to_string();
                let name = match validate_bundle_rel_path(&raw_name) {
                    Ok(n) => n,
                    Err(_) => {
                        log::debug!("Skipping unsafe ZIP member path: {raw_name}");
                        continue;
                    }
                };
                if entry.size() > 500_000 {
                    log::debug!("Skipping large file in ZIP: {name} ({} bytes)", entry.size());
                    continue;
                }
                let mut buf = Vec::new();
                use std::io::Read;
                if entry.read_to_end(&mut buf).is_err() {
                    continue;
                }
                match String::from_utf8(buf) {
                    Ok(s) => {
                        files.insert(name, FileContent::Text(s));
                    }
                    Err(_) => {
                        log::debug!("Skipping non-text file in ZIP: {name}");
                    }
                }
            }
            return files;
        }
        log::debug!("ClawHub ZIP download exhausted retries for {slug} v{version}");
        files
    }

    fn fetch_text(&self, url: &str) -> Option<String> {
        fetch_text_simple(url, 20)
    }
}

impl SkillSource for ClawHubSource {
    fn source_id(&self) -> String {
        "clawhub".to_string()
    }

    fn search(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        let query = query.trim().to_string();

        if !query.is_empty() {
            let query_terms = Self::query_terms(&query);
            if query_terms.len() >= 2 {
                if let Some(direct) = self.exact_slug_meta(&query) {
                    return vec![direct];
                }
            }
            let results = self.search_catalog(&query, limit);
            if !results.is_empty() {
                return results;
            }
        }

        let cache_key = format!(
            "clawhub_search_listing_v1_{}_{limit}",
            md5_hex(&query)
        );
        if let Some(cached) = read_index_cache(&cache_key) {
            return self.finalize_search_results(&query, metas_from_cache(&cached), limit);
        }

        let client = http_client();
        let resp = match client
            .get(format!("{CLAWHUB_BASE_URL}/skills"))
            .query(&[("search", query.as_str()), ("limit", &limit.to_string())])
            .timeout(Duration::from_secs(15))
            .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
            .send()
        {
            Ok(r) if r.status().as_u16() == 200 => r,
            _ => return Vec::new(),
        };
        let data: Value = match resp.json() {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let skills_data = if data.is_object() {
            data.get("items").cloned().unwrap_or(data)
        } else {
            data
        };
        let arr = match skills_data.as_array() {
            Some(a) => a.clone(),
            None => return Vec::new(),
        };
        let mut results = Vec::new();
        for item in arr.iter().take(limit) {
            let slug = match item.get("slug").and_then(|v| v.as_str()) {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => continue,
            };
            let display_name = item
                .get("displayName")
                .and_then(|v| v.as_str())
                .or_else(|| item.get("name").and_then(|v| v.as_str()))
                .filter(|s| !s.is_empty())
                .unwrap_or(&slug)
                .to_string();
            let summary = item
                .get("summary")
                .and_then(|v| v.as_str())
                .or_else(|| item.get("description").and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string();
            let tags = Self::normalize_tags(item.get("tags").unwrap_or(&Value::Null));
            results.push(SkillMeta {
                name: display_name,
                description: summary,
                source: "clawhub".to_string(),
                identifier: slug,
                trust_level: "community".to_string(),
                repo: None,
                path: None,
                tags,
                extra: BTreeMap::new(),
            });
        }
        let final_results = self.finalize_search_results(&query, results, limit);
        let cache_data = Value::Array(final_results.iter().map(skill_meta_to_dict).collect());
        write_index_cache(&cache_key, &cache_data);
        final_results
    }

    fn fetch(&self, identifier: &str) -> Option<SkillBundle> {
        let slug = identifier.rsplit('/').next().unwrap_or(identifier).to_string();
        let skill_data = self.get_json(&format!("{CLAWHUB_BASE_URL}/skills/{slug}"), 20)?;
        if !skill_data.is_object() {
            return None;
        }
        let latest_version = match self.resolve_latest_version(&slug, &skill_data) {
            Some(v) => v,
            None => {
                log::warn!("ClawHub fetch failed for {slug}: could not resolve latest version");
                return None;
            }
        };

        let mut files = self.download_zip(&slug, &latest_version);

        if !files.contains_key("SKILL.md") {
            if let Some(version_data) =
                self.get_json(&format!("{CLAWHUB_BASE_URL}/skills/{slug}/versions/{latest_version}"), 20)
            {
                if version_data.is_object() {
                    let extracted = self.extract_files(&version_data);
                    if !extracted.is_empty() {
                        files = extracted;
                    }
                    if !files.contains_key("SKILL.md") {
                        if let Some(Value::Object(_)) = version_data.get("version") {
                            let nested = version_data.get("version").unwrap();
                            let nested_files = self.extract_files(nested);
                            if !nested_files.is_empty() {
                                files = nested_files;
                            }
                        }
                    }
                }
            }
        }

        if !files.contains_key("SKILL.md") {
            log::warn!(
                "ClawHub fetch for {slug} resolved version {latest_version} but could not retrieve file content"
            );
            return None;
        }
        Some(SkillBundle::new(slug.clone(), files, "clawhub", slug, "community"))
    }

    fn inspect(&self, identifier: &str) -> Option<SkillMeta> {
        let slug = identifier.rsplit('/').next().unwrap_or(identifier).to_string();
        let raw = self.get_json(&format!("{CLAWHUB_BASE_URL}/skills/{slug}"), 20)?;
        let data = Self::coerce_skill_payload(&raw)?;
        let obj = data.as_object()?;
        let tags = Self::normalize_tags(obj.get("tags").unwrap_or(&Value::Null));
        let name = obj
            .get("displayName")
            .and_then(|v| v.as_str())
            .or_else(|| obj.get("name").and_then(|v| v.as_str()))
            .or_else(|| obj.get("slug").and_then(|v| v.as_str()))
            .filter(|s| !s.is_empty())
            .unwrap_or(&slug)
            .to_string();
        let description = obj
            .get("summary")
            .and_then(|v| v.as_str())
            .or_else(|| obj.get("description").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string();
        let identifier = obj
            .get("slug")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(&slug)
            .to_string();
        Some(SkillMeta {
            name,
            description,
            source: "clawhub".to_string(),
            identifier,
            trust_level: "community".to_string(),
            repo: None,
            path: None,
            tags,
            extra: BTreeMap::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Claude Code marketplace source adapter
// ---------------------------------------------------------------------------

/// Discover skills from Claude Code marketplace repos.
pub struct ClaudeMarketplaceSource {
    pub auth: std::rc::Rc<GitHubAuth>,
}

const KNOWN_MARKETPLACES: &[&str] = &["anthropics/skills", "aiskillstore/marketplace"];

impl ClaudeMarketplaceSource {
    pub fn new(auth: std::rc::Rc<GitHubAuth>) -> Self {
        ClaudeMarketplaceSource { auth }
    }

    fn trust_level_for_impl(identifier: &str) -> String {
        GitHubSource::trust_level_for_impl(identifier)
    }

    fn fetch_marketplace_index(&self, repo: &str) -> Vec<Value> {
        let cache_key = format!("claude_marketplace_{}", repo.replace('/', "_"));
        if let Some(cached) = read_index_cache(&cache_key) {
            if let Some(arr) = cached.as_array() {
                return arr.clone();
            }
        }
        let url = format!("https://api.github.com/repos/{repo}/contents/.claude-plugin/marketplace.json");
        let client = http_client();
        let mut h = self.auth.get_headers();
        h.retain(|(k, _)| k != "Accept");
        h.push(("Accept".to_string(), "application/vnd.github.v3.raw".to_string()));
        let req = GitHubSource::apply_headers(
            client.get(&url).timeout(Duration::from_secs(15)),
            &h,
        )
        .header(reqwest::header::USER_AGENT, "hermes-skills-hub");
        let resp = match req.send() {
            Ok(r) if r.status().as_u16() == 200 => r,
            _ => return Vec::new(),
        };
        let text = match resp.text() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let data: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let plugins = data.get("plugins").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        write_index_cache(&cache_key, &Value::Array(plugins.clone()));
        plugins
    }
}

impl SkillSource for ClaudeMarketplaceSource {
    fn source_id(&self) -> String {
        "claude-marketplace".to_string()
    }

    fn trust_level_for(&self, identifier: &str) -> String {
        Self::trust_level_for_impl(identifier)
    }

    fn search(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        let mut results: Vec<SkillMeta> = Vec::new();
        let query_lower = query.to_lowercase();
        for marketplace_repo in KNOWN_MARKETPLACES {
            let plugins = self.fetch_marketplace_index(marketplace_repo);
            for plugin in &plugins {
                let pname = plugin.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let pdesc = plugin.get("description").and_then(|v| v.as_str()).unwrap_or("");
                let searchable = format!("{pname} {pdesc}").to_lowercase();
                if searchable.contains(&query_lower) {
                    let source_path = plugin.get("source").and_then(|v| v.as_str()).unwrap_or("");
                    let identifier = if let Some(rest) = source_path.strip_prefix("./") {
                        format!("{marketplace_repo}/{rest}")
                    } else if source_path.contains('/') {
                        source_path.to_string()
                    } else {
                        format!("{marketplace_repo}/{source_path}")
                    };
                    results.push(SkillMeta {
                        name: pname.to_string(),
                        description: pdesc.to_string(),
                        source: "claude-marketplace".to_string(),
                        identifier: identifier.clone(),
                        trust_level: Self::trust_level_for_impl(&identifier),
                        repo: Some(marketplace_repo.to_string()),
                        path: None,
                        tags: Vec::new(),
                        extra: BTreeMap::new(),
                    });
                }
            }
        }
        results.into_iter().take(limit).collect()
    }

    fn fetch(&self, identifier: &str) -> Option<SkillBundle> {
        let gh = GitHubSource::new(self.auth.clone(), None);
        let mut bundle = gh.fetch(identifier)?;
        bundle.source = "claude-marketplace".to_string();
        Some(bundle)
    }

    fn inspect(&self, identifier: &str) -> Option<SkillMeta> {
        let gh = GitHubSource::new(self.auth.clone(), None);
        let mut meta = gh.inspect(identifier)?;
        meta.source = "claude-marketplace".to_string();
        meta.trust_level = Self::trust_level_for_impl(identifier);
        Some(meta)
    }
}

// ---------------------------------------------------------------------------
// LobeHub source adapter
// ---------------------------------------------------------------------------

/// Fetch skills from LobeHub's agent marketplace.
pub struct LobeHubSource;

const LOBEHUB_INDEX_URL: &str = "https://chat-agents.lobehub.com/index.json";

impl LobeHubSource {
    fn fetch_index(&self) -> Option<Value> {
        let cache_key = "lobehub_index";
        if let Some(cached) = read_index_cache(cache_key) {
            return Some(cached);
        }
        let client = http_client();
        let resp = client
            .get(LOBEHUB_INDEX_URL)
            .timeout(Duration::from_secs(30))
            .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
            .send()
            .ok()?;
        if resp.status().as_u16() != 200 {
            return None;
        }
        let data: Value = resp.json().ok()?;
        write_index_cache(cache_key, &data);
        Some(data)
    }

    fn fetch_agent(&self, agent_id: &str) -> Option<Value> {
        let url = format!("https://chat-agents.lobehub.com/{agent_id}.json");
        let client = http_client();
        match client
            .get(&url)
            .timeout(Duration::from_secs(15))
            .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
            .send()
        {
            Ok(resp) if resp.status().as_u16() == 200 => resp.json().ok(),
            _ => None,
        }
    }

    fn agents_list(index: &Value) -> Option<Vec<Value>> {
        if index.is_object() {
            index.get("agents").and_then(|v| v.as_array()).cloned()
        } else {
            index.as_array().cloned()
        }
    }

    fn convert_to_skill_md(agent_data: &Value) -> String {
        let meta = agent_data.get("meta").unwrap_or(agent_data);
        let identifier = agent_data
            .get("identifier")
            .and_then(|v| v.as_str())
            .unwrap_or("lobehub-agent");
        let title = meta.get("title").and_then(|v| v.as_str()).unwrap_or(identifier);
        let description = meta.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let tag_list: Vec<String> = meta
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(yaml_json_str).collect())
            .unwrap_or_default();
        let system_role = agent_data
            .get("config")
            .and_then(|c| c.get("systemRole"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let desc_trunc: String = description.chars().take(500).collect();
        let fm_lines = vec![
            "---".to_string(),
            format!("name: {identifier}"),
            format!("description: {desc_trunc}"),
            "metadata:".to_string(),
            "  hermes:".to_string(),
            format!("    tags: [{}]", tag_list.join(", ")),
            "  lobehub:".to_string(),
            "    source: lobehub".to_string(),
            "---".to_string(),
        ];
        let body_lines = vec![
            format!("# {title}"),
            String::new(),
            description.to_string(),
            String::new(),
            "## Instructions".to_string(),
            String::new(),
            if system_role.is_empty() {
                "(No system role defined)".to_string()
            } else {
                system_role.to_string()
            },
        ];
        format!("{}\n\n{}\n", fm_lines.join("\n"), body_lines.join("\n"))
    }
}

impl SkillSource for LobeHubSource {
    fn source_id(&self) -> String {
        "lobehub".to_string()
    }

    fn search(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        let index = match self.fetch_index() {
            Some(i) => i,
            None => return Vec::new(),
        };
        let query_lower = query.to_lowercase();
        let agents = match Self::agents_list(&index) {
            Some(a) => a,
            None => return Vec::new(),
        };
        let mut results = Vec::new();
        for agent in &agents {
            let meta = agent.get("meta").unwrap_or(agent);
            let agent_ident = agent.get("identifier").and_then(|v| v.as_str()).unwrap_or("");
            let title = meta
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or(agent_ident);
            let desc = meta.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let tags: Vec<String> = meta
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(yaml_json_str).collect())
                .unwrap_or_default();
            let tags_joined = tags.join(" ");
            let searchable = format!("{title} {desc} {tags_joined}").to_lowercase();
            if searchable.contains(&query_lower) {
                let identifier = if !agent_ident.is_empty() {
                    agent_ident.to_string()
                } else {
                    title.to_lowercase().replace(' ', "-")
                };
                let desc_trunc: String = desc.chars().take(200).collect();
                results.push(SkillMeta {
                    name: identifier.clone(),
                    description: desc_trunc,
                    source: "lobehub".to_string(),
                    identifier: format!("lobehub/{identifier}"),
                    trust_level: "community".to_string(),
                    repo: None,
                    path: None,
                    tags,
                    extra: BTreeMap::new(),
                });
            }
            if results.len() >= limit {
                break;
            }
        }
        results
    }

    fn fetch(&self, identifier: &str) -> Option<SkillBundle> {
        let agent_id = if let Some(rest) = identifier.strip_prefix("lobehub/") {
            rest.to_string()
        } else {
            identifier.to_string()
        };
        let agent_data = self.fetch_agent(&agent_id)?;
        let skill_md = Self::convert_to_skill_md(&agent_data);
        let mut files = BTreeMap::new();
        files.insert("SKILL.md".to_string(), FileContent::Text(skill_md));
        Some(SkillBundle::new(
            agent_id.clone(),
            files,
            "lobehub",
            format!("lobehub/{agent_id}"),
            "community",
        ))
    }

    fn inspect(&self, identifier: &str) -> Option<SkillMeta> {
        let agent_id = if let Some(rest) = identifier.strip_prefix("lobehub/") {
            rest.to_string()
        } else {
            identifier.to_string()
        };
        let index = self.fetch_index()?;
        let agents = Self::agents_list(&index)?;
        for agent in &agents {
            if agent.get("identifier").and_then(|v| v.as_str()) == Some(agent_id.as_str()) {
                let meta = agent.get("meta").unwrap_or(agent);
                let tags: Vec<String> = meta
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().map(yaml_json_str).collect())
                    .unwrap_or_default();
                return Some(SkillMeta {
                    name: agent_id.clone(),
                    description: meta.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    source: "lobehub".to_string(),
                    identifier: format!("lobehub/{agent_id}"),
                    trust_level: "community".to_string(),
                    repo: None,
                    path: None,
                    tags,
                    extra: BTreeMap::new(),
                });
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Official optional skills source adapter
// ---------------------------------------------------------------------------

/// Fetch skills from the optional-skills/ directory shipped with the repo.
pub struct OptionalSkillSource {
    pub optional_dir: PathBuf,
}

impl Default for OptionalSkillSource {
    fn default() -> Self {
        Self::new()
    }
}

impl OptionalSkillSource {
    pub fn new() -> Self {
        // Mirror Path(__file__).parent.parent / "optional-skills"; the native
        // default is the repo root's optional-skills next to the binary's CWD.
        let default = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("optional-skills");
        let optional_dir = get_optional_skills_dir(default);
        OptionalSkillSource { optional_dir }
    }

    pub fn with_dir(dir: PathBuf) -> Self {
        OptionalSkillSource { optional_dir: dir }
    }

    fn parse_frontmatter(content: &str) -> serde_yaml::Mapping {
        parse_frontmatter_quick(content)
    }

    fn find_skill_dir(&self, name: &str) -> Option<PathBuf> {
        if !self.optional_dir.is_dir() {
            return None;
        }
        let mut found = None;
        Self::walk_skill_md(&self.optional_dir, &mut |skill_md| {
            if found.is_none() {
                if let Some(parent) = skill_md.parent() {
                    if parent.file_name().and_then(|n| n.to_str()) == Some(name) {
                        found = Some(parent.to_path_buf());
                    }
                }
            }
        });
        found
    }

    fn walk_skill_md(dir: &Path, cb: &mut dyn FnMut(&Path)) {
        let read = match std::fs::read_dir(dir) {
            Ok(r) => r,
            Err(_) => return,
        };
        for entry in read.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::walk_skill_md(&path, cb);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("SKILL.md") {
                cb(&path);
            }
        }
    }

    fn scan_all(&self) -> Vec<SkillMeta> {
        if !self.optional_dir.is_dir() {
            return Vec::new();
        }
        let mut skill_mds: Vec<PathBuf> = Vec::new();
        Self::walk_skill_md(&self.optional_dir, &mut |p| skill_mds.push(p.to_path_buf()));
        skill_mds.sort();

        let mut results = Vec::new();
        for skill_md in &skill_mds {
            let parent = match skill_md.parent() {
                Some(p) => p,
                None => continue,
            };
            let rel = match parent.strip_prefix(&self.optional_dir) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if rel
                .components()
                .any(|c| c.as_os_str().to_str().map(|s| s.starts_with('.')).unwrap_or(false))
            {
                continue;
            }
            let content = match std::fs::read_to_string(skill_md) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let fm = Self::parse_frontmatter(&content);
            let name = {
                let n = yaml_str(yaml_get(&fm, "name"));
                if n.is_empty() {
                    parent.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string()
                } else {
                    n
                }
            };
            let desc = yaml_str(yaml_get(&fm, "description"));
            let desc_trunc: String = desc.chars().take(200).collect();
            let tags = {
                if let Some(serde_yaml::Value::Mapping(meta)) = yaml_get(&fm, "metadata") {
                    if let Some(serde_yaml::Value::Mapping(hermes)) = yaml_get(meta, "hermes") {
                        if let Some(serde_yaml::Value::Sequence(seq)) = yaml_get(hermes, "tags") {
                            seq.iter().map(yaml_value_to_string).collect()
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };
            let rel_path = rel.to_string_lossy().replace('\\', "/");
            results.push(SkillMeta {
                name,
                description: desc_trunc,
                source: "official".to_string(),
                identifier: format!("official/{rel_path}"),
                trust_level: "builtin".to_string(),
                repo: None,
                path: Some(rel_path),
                tags,
                extra: BTreeMap::new(),
            });
        }
        results
    }
}

impl SkillSource for OptionalSkillSource {
    fn source_id(&self) -> String {
        "official".to_string()
    }

    fn trust_level_for(&self, _identifier: &str) -> String {
        "builtin".to_string()
    }

    fn search(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        let mut results = Vec::new();
        let query_lower = query.to_lowercase();
        for meta in self.scan_all() {
            let searchable = format!("{} {} {}", meta.name, meta.description, meta.tags.join(" "))
                .to_lowercase();
            if searchable.contains(&query_lower) {
                results.push(meta);
            }
            if results.len() >= limit {
                break;
            }
        }
        results
    }

    fn fetch(&self, identifier: &str) -> Option<SkillBundle> {
        let rel = if let Some(rest) = identifier.strip_prefix("official/") {
            rest.to_string()
        } else {
            identifier.to_string()
        };
        let skill_dir = self.optional_dir.join(&rel);

        // Guard against path traversal.
        let optional_resolved = self
            .optional_dir
            .canonicalize()
            .unwrap_or_else(|_| self.optional_dir.clone());
        let resolved = match skill_dir.canonicalize() {
            Ok(r) => r,
            Err(_) => {
                // Path doesn't exist; try by name only (last segment).
                let skill_name = rel.rsplit('/').next().unwrap_or(&rel);
                match self.find_skill_dir(skill_name) {
                    Some(d) => d,
                    None => return None,
                }
            }
        };
        let skill_dir = if resolved.starts_with(&optional_resolved) {
            if resolved.is_dir() {
                resolved
            } else {
                let skill_name = rel.rsplit('/').next().unwrap_or(&rel);
                match self.find_skill_dir(skill_name) {
                    Some(d) => d,
                    None => return None,
                }
            }
        } else {
            return None;
        };

        let mut files: BTreeMap<String, FileContent> = BTreeMap::new();
        let mut walk: Vec<PathBuf> = vec![skill_dir.clone()];
        while let Some(dir) = walk.pop() {
            let read = match std::fs::read_dir(&dir) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for entry in read.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk.push(path);
                    continue;
                }
                let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if fname.starts_with('.') {
                    continue;
                }
                if path.components().any(|c| c.as_os_str() == "__pycache__") {
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) == Some("pyc") {
                    continue;
                }
                if let Ok(rel_path) = path.strip_prefix(&skill_dir) {
                    let rel_str = rel_path.to_string_lossy().replace('\\', "/");
                    match std::fs::read(&path) {
                        Ok(bytes) => {
                            files.insert(rel_str, FileContent::Bytes(bytes));
                        }
                        Err(_) => continue,
                    }
                }
            }
        }

        if files.is_empty() {
            return None;
        }
        let name = skill_dir.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let rel_to_optional = skill_dir
            .strip_prefix(&self.optional_dir)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| name.clone());
        Some(SkillBundle::new(
            name,
            files,
            "official",
            format!("official/{rel_to_optional}"),
            "builtin",
        ))
    }

    fn inspect(&self, identifier: &str) -> Option<SkillMeta> {
        let rel = if let Some(rest) = identifier.strip_prefix("official/") {
            rest.to_string()
        } else {
            identifier.to_string()
        };
        let skill_name = rel.rsplit('/').next().unwrap_or(&rel).to_string();
        for meta in self.scan_all() {
            if meta.name == skill_name {
                return Some(meta);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Hermes centralized index source
// ---------------------------------------------------------------------------

pub const HERMES_INDEX_URL: &str =
    "https://hermes-agent.nousresearch.com/docs/api/skills-index.json";
pub const HERMES_INDEX_TTL: u64 = 6 * 3600;

fn hermes_index_cache_file() -> PathBuf {
    index_cache_dir().join("hermes-index.json")
}

fn load_stale_index_cache() -> Option<Value> {
    let path = hermes_index_cache_file();
    if path.exists() {
        if let Ok(text) = std::fs::read_to_string(&path) {
            return serde_json::from_str(&text).ok();
        }
    }
    None
}

/// Fetch the centralized skills index, with local cache.
pub fn load_hermes_index() -> Option<Value> {
    let cache_file = hermes_index_cache_file();
    if cache_file.exists() {
        if let Some(age) = mtime_age_secs(&cache_file) {
            if age < HERMES_INDEX_TTL {
                if let Ok(text) = std::fs::read_to_string(&cache_file) {
                    if let Ok(v) = serde_json::from_str(&text) {
                        return Some(v);
                    }
                }
            }
        }
    }

    let client = http_client();
    let resp = match client
        .get(HERMES_INDEX_URL)
        .timeout(Duration::from_secs(15))
        .header(reqwest::header::USER_AGENT, "hermes-skills-hub")
        .send()
    {
        Ok(r) => r,
        Err(e) => {
            log::debug!("Hermes index fetch failed: {e}");
            return load_stale_index_cache();
        }
    };
    if resp.status().as_u16() != 200 {
        log::debug!("Hermes index fetch returned {}", resp.status().as_u16());
        return load_stale_index_cache();
    }
    let data: Value = match resp.json() {
        Ok(v) => v,
        Err(e) => {
            log::debug!("Hermes index fetch failed: {e}");
            return load_stale_index_cache();
        }
    };
    if !data.is_object() || data.get("skills").is_none() {
        return load_stale_index_cache();
    }
    if let Some(parent) = cache_file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(s) = serde_json::to_string(&data) {
        let _ = std::fs::write(&cache_file, s);
    }
    Some(data)
}

/// Skill source backed by the centralized Hermes Skills Index.
pub struct HermesIndexSource {
    pub auth: std::rc::Rc<GitHubAuth>,
    index: std::cell::RefCell<Option<Value>>,
    loaded: std::cell::Cell<bool>,
    github: std::cell::RefCell<Option<GitHubSource>>,
}

impl HermesIndexSource {
    pub fn new(auth: std::rc::Rc<GitHubAuth>) -> Self {
        HermesIndexSource {
            auth,
            index: std::cell::RefCell::new(None),
            loaded: std::cell::Cell::new(false),
            github: std::cell::RefCell::new(None),
        }
    }

    fn ensure_loaded(&self) -> Value {
        if !self.loaded.get() {
            *self.index.borrow_mut() = load_hermes_index();
            self.loaded.set(true);
        }
        self.index
            .borrow()
            .clone()
            .unwrap_or_else(|| serde_json::json!({}))
    }

    fn skills_of<'a>(index: &'a Value) -> Vec<Value> {
        index
            .get("skills")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
    }

    fn find_entry(&self, identifier: &str, index: &Value) -> Option<Value> {
        let skills = Self::skills_of(index);

        for s in &skills {
            if s.get("identifier").and_then(|v| v.as_str()) == Some(identifier) {
                return Some(s.clone());
            }
        }

        let prefixes = ["skills-sh/", "skills.sh/", "official/", "github/", "clawhub/"];
        let mut normalized = identifier.to_string();
        for prefix in prefixes {
            if let Some(rest) = identifier.strip_prefix(prefix) {
                normalized = rest.to_string();
                break;
            }
        }

        for s in &skills {
            let sid = s.get("identifier").and_then(|v| v.as_str()).unwrap_or("");
            let mut stored_normalized = sid.to_string();
            for prefix in prefixes {
                if let Some(rest) = sid.strip_prefix(prefix) {
                    stored_normalized = rest.to_string();
                    break;
                }
            }
            if stored_normalized == normalized {
                return Some(s.clone());
            }
        }
        None
    }

    fn to_meta(entry: &Value) -> SkillMeta {
        let get_str = |k: &str| entry.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let tags = entry
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(yaml_json_str).collect())
            .unwrap_or_default();
        let extra = entry
            .get("extra")
            .and_then(|v| v.as_object())
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        SkillMeta {
            name: get_str("name"),
            description: get_str("description"),
            source: {
                let s = get_str("source");
                if s.is_empty() {
                    "hermes-index".to_string()
                } else {
                    s
                }
            },
            identifier: get_str("identifier"),
            trust_level: {
                let t = get_str("trust_level");
                if t.is_empty() {
                    "community".to_string()
                } else {
                    t
                }
            },
            repo: entry.get("repo").and_then(|v| v.as_str()).map(|s| s.to_string()),
            path: entry.get("path").and_then(|v| v.as_str()).map(|s| s.to_string()),
            tags,
            extra,
        }
    }

    fn with_github<R>(&self, f: impl FnOnce(&GitHubSource) -> R) -> R {
        if self.github.borrow().is_none() {
            *self.github.borrow_mut() = Some(GitHubSource::new(self.auth.clone(), None));
        }
        let borrow = self.github.borrow();
        f(borrow.as_ref().unwrap())
    }
}

impl SkillSource for HermesIndexSource {
    fn source_id(&self) -> String {
        "hermes-index".to_string()
    }

    fn is_available(&self) -> bool {
        let index = self.ensure_loaded();
        !Self::skills_of(&index).is_empty()
    }

    fn trust_level_for(&self, identifier: &str) -> String {
        let index = self.ensure_loaded();
        for skill in Self::skills_of(&index) {
            if skill.get("identifier").and_then(|v| v.as_str()) == Some(identifier) {
                return skill
                    .get("trust_level")
                    .and_then(|v| v.as_str())
                    .unwrap_or("community")
                    .to_string();
            }
        }
        "community".to_string()
    }

    fn search(&self, query: &str, limit: usize) -> Vec<SkillMeta> {
        let index = self.ensure_loaded();
        let skills = Self::skills_of(&index);
        if skills.is_empty() {
            return Vec::new();
        }
        if query.trim().is_empty() {
            return skills.iter().take(limit).map(Self::to_meta).collect();
        }
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();
        for s in &skills {
            let name = s.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let desc = s.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let tags: Vec<String> = s
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(yaml_json_str).collect())
                .unwrap_or_default();
            let searchable = format!("{name} {desc} {}", tags.join(" ")).to_lowercase();
            if searchable.contains(&query_lower) {
                results.push(Self::to_meta(s));
                if results.len() >= limit {
                    break;
                }
            }
        }
        results
    }

    fn fetch(&self, identifier: &str) -> Option<SkillBundle> {
        let index = self.ensure_loaded();
        let entry = self.find_entry(identifier, &index)?;

        if let Some(resolved) = entry.get("resolved_github_id").and_then(|v| v.as_str()) {
            if !resolved.is_empty() {
                if let Some(mut bundle) = self.with_github(|gh| gh.fetch(resolved)) {
                    bundle.source = entry
                        .get("source")
                        .and_then(|v| v.as_str())
                        .unwrap_or("hermes-index")
                        .to_string();
                    bundle.identifier = identifier.to_string();
                    return Some(bundle);
                }
            }
        }

        let repo = entry.get("repo").and_then(|v| v.as_str()).unwrap_or("");
        let path = entry.get("path").and_then(|v| v.as_str()).unwrap_or("");
        if !repo.is_empty() && !path.is_empty() {
            let github_id = format!("{repo}/{path}");
            if let Some(mut bundle) = self.with_github(|gh| gh.fetch(&github_id)) {
                bundle.source = entry
                    .get("source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("hermes-index")
                    .to_string();
                bundle.identifier = identifier.to_string();
                return Some(bundle);
            }
        }
        None
    }

    fn inspect(&self, identifier: &str) -> Option<SkillMeta> {
        let index = self.ensure_loaded();
        self.find_entry(identifier, &index).map(|e| Self::to_meta(&e))
    }
}

// ---------------------------------------------------------------------------
// Source router + unified search
// ---------------------------------------------------------------------------

/// Create all configured source adapters. Returns a list of active sources
/// for search/fetch operations.
pub fn create_source_router(auth: Option<std::rc::Rc<GitHubAuth>>) -> Vec<Box<dyn SkillSource>> {
    let auth = auth.unwrap_or_else(|| std::rc::Rc::new(GitHubAuth::new()));
    let taps_mgr = TapsManager::default();
    let extra_taps = taps_mgr.list_taps_typed();

    let sources: Vec<Box<dyn SkillSource>> = vec![
        Box::new(OptionalSkillSource::new()),
        Box::new(HermesIndexSource::new(auth.clone())),
        Box::new(SkillsShSource::new(auth.clone())),
        Box::new(WellKnownSkillSource),
        Box::new(UrlSource),
        Box::new(GitHubSource::new(auth.clone(), Some(extra_taps))),
        Box::new(ClawHubSource),
        Box::new(ClaudeMarketplaceSource::new(auth.clone())),
        Box::new(LobeHubSource),
    ];
    sources
}

fn search_one_source(src: &dyn SkillSource, query: &str, limit: usize) -> (String, Vec<SkillMeta>) {
    let sid = src.source_id();
    let results = src.search(query, limit);
    (sid, results)
}

/// Search all sources with per-source limits and an index-availability skip.
///
/// Returns `(all_results, source_counts, timed_out_ids)`. The Rust port runs
/// searches sequentially (no thread-pool); `timed_out_ids` is therefore always
/// empty. The index-availability fast path that skips external API sources is
/// preserved.
pub fn parallel_search_sources(
    sources: &[Box<dyn SkillSource>],
    query: &str,
    per_source_limits: Option<&HashMap<String, usize>>,
    source_filter: &str,
) -> (Vec<SkillMeta>, BTreeMap<String, usize>, Vec<String>) {
    let empty_limits = HashMap::new();
    let per_source_limits = per_source_limits.unwrap_or(&empty_limits);

    let api_source_ids: HashSet<&str> = [
        "github",
        "skills-sh",
        "clawhub",
        "claude-marketplace",
        "lobehub",
        "well-known",
    ]
    .into_iter()
    .collect();

    let mut index_available = false;
    if source_filter == "all" {
        for src in sources {
            if src.source_id() == "hermes-index" && src.is_available() {
                index_available = true;
                break;
            }
        }
    }

    let mut active: Vec<&Box<dyn SkillSource>> = Vec::new();
    for src in sources {
        let sid = src.source_id();
        if source_filter != "all" && sid != source_filter && sid != "official" {
            continue;
        }
        if index_available && api_source_ids.contains(sid.as_str()) {
            continue;
        }
        active.push(src);
    }

    let mut all_results: Vec<SkillMeta> = Vec::new();
    let mut source_counts: BTreeMap<String, usize> = BTreeMap::new();
    let timed_out_ids: Vec<String> = Vec::new();

    for src in active {
        let lim = per_source_limits
            .get(&src.source_id())
            .copied()
            .unwrap_or(50);
        let (sid, results) = search_one_source(src.as_ref(), query, lim);
        source_counts.insert(sid, results.len());
        all_results.extend(results);
    }

    (all_results, source_counts, timed_out_ids)
}

/// Search all sources and merge results (dedupe by name, prefer higher trust).
pub fn unified_search(
    query: &str,
    sources: &[Box<dyn SkillSource>],
    source_filter: &str,
    limit: usize,
) -> Vec<SkillMeta> {
    let (all_results, _, _) = parallel_search_sources(sources, query, None, source_filter);

    let mut seen: HashMap<String, SkillMeta> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for r in all_results {
        match seen.get(&r.name) {
            None => {
                order.push(r.name.clone());
                seen.insert(r.name.clone(), r);
            }
            Some(existing) => {
                if trust_rank(&r.trust_level) > trust_rank(&existing.trust_level) {
                    seen.insert(r.name.clone(), r);
                }
            }
        }
    }
    let deduped: Vec<SkillMeta> = order.into_iter().filter_map(|n| seen.remove(&n)).collect();
    deduped.into_iter().take(limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_bundle_path_basic() {
        assert_eq!(normalize_bundle_path("a/b/c", "f", true).unwrap(), "a/b/c");
        assert_eq!(normalize_bundle_path("./a/./b", "f", true).unwrap(), "a/b");
        assert_eq!(normalize_bundle_path("a\\b", "f", true).unwrap(), "a/b");
    }

    #[test]
    fn test_normalize_bundle_path_rejects_traversal() {
        assert!(normalize_bundle_path("../evil", "f", true).is_err());
        assert!(normalize_bundle_path("/abs", "f", true).is_err());
        assert!(normalize_bundle_path("", "f", true).is_err());
        assert!(normalize_bundle_path("   ", "f", true).is_err());
        assert!(normalize_bundle_path("C:/win", "f", true).is_err());
    }

    #[test]
    fn test_validate_skill_name_no_nesting() {
        assert_eq!(validate_skill_name("my-skill").unwrap(), "my-skill");
        assert!(validate_skill_name("a/b").is_err());
    }

    #[test]
    fn test_trust_level_for() {
        assert_eq!(GitHubSource::trust_level_for_impl("openai/skills/foo"), "trusted");
        assert_eq!(GitHubSource::trust_level_for_impl("anthropics/skills/bar"), "trusted");
        assert_eq!(GitHubSource::trust_level_for_impl("random/repo/baz"), "community");
        assert_eq!(GitHubSource::trust_level_for_impl("single"), "community");
    }

    #[test]
    fn test_parse_frontmatter_quick() {
        let content = "---\nname: my-skill\ndescription: A test skill\ntags:\n  - a\n  - b\n---\n\n# Body\n";
        let fm = parse_frontmatter_quick(content);
        assert_eq!(yaml_str(yaml_get(&fm, "name")), "my-skill");
        assert_eq!(yaml_str(yaml_get(&fm, "description")), "A test skill");
        let tags = extract_tags_from_fm(&fm);
        assert_eq!(tags, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_parse_frontmatter_quick_metadata_hermes_tags() {
        let content = "---\nname: x\nmetadata:\n  hermes:\n    tags:\n      - one\n      - two\n---\nbody\n";
        let fm = parse_frontmatter_quick(content);
        assert_eq!(extract_tags_from_fm(&fm), vec!["one".to_string(), "two".to_string()]);
    }

    #[test]
    fn test_parse_frontmatter_quick_no_frontmatter() {
        assert!(parse_frontmatter_quick("# just a heading\n").is_empty());
        assert!(parse_frontmatter_quick("---\nno closing").is_empty());
    }

    #[test]
    fn test_skill_meta_roundtrip() {
        let mut extra = BTreeMap::new();
        extra.insert("url".to_string(), Value::String("http://x".to_string()));
        let meta = SkillMeta {
            name: "n".into(),
            description: "d".into(),
            source: "github".into(),
            identifier: "a/b/c".into(),
            trust_level: "community".into(),
            repo: Some("a/b".into()),
            path: Some("c".into()),
            tags: vec!["t1".into()],
            extra,
        };
        let v = skill_meta_to_dict(&meta);
        let back = skill_meta_from_value(&v).unwrap();
        assert_eq!(back.name, "n");
        assert_eq!(back.repo, Some("a/b".to_string()));
        assert_eq!(back.tags, vec!["t1".to_string()]);
        assert_eq!(back.extra.get("url").unwrap().as_str(), Some("http://x"));
    }

    #[test]
    fn test_bundle_content_hash_deterministic() {
        let mut files = BTreeMap::new();
        files.insert("SKILL.md".to_string(), FileContent::Text("hello".to_string()));
        files.insert("a.txt".to_string(), FileContent::Bytes(b"world".to_vec()));
        let bundle = SkillBundle::new("x", files, "github", "a/b/c", "community");
        let h1 = bundle_content_hash(&bundle);
        let h2 = bundle_content_hash(&bundle);
        assert_eq!(h1, h2);
        assert!(h1.starts_with("sha256:"));
        assert_eq!(h1.len(), "sha256:".len() + 16);
    }

    #[test]
    fn test_source_matches_alias() {
        struct S;
        impl SkillSource for S {
            fn search(&self, _: &str, _: usize) -> Vec<SkillMeta> { vec![] }
            fn fetch(&self, _: &str) -> Option<SkillBundle> { None }
            fn inspect(&self, _: &str) -> Option<SkillMeta> { None }
            fn source_id(&self) -> String { "skills-sh".to_string() }
        }
        let s = S;
        assert!(source_matches(&s, "skills.sh"));
        assert!(source_matches(&s, "skills-sh"));
        assert!(!source_matches(&s, "github"));
    }

    #[test]
    fn test_md5_hex() {
        assert_eq!(md5_hex("abc"), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn test_url_source_matches() {
        assert!(UrlSource::matches("https://example.com/path/SKILL.md"));
        assert!(UrlSource::matches("http://x.io/a.md"));
        assert!(!UrlSource::matches("https://x.io/a.txt"));
        assert!(!UrlSource::matches("https://x.io/.well-known/skills/foo.md"));
        assert!(!UrlSource::matches("https://x.io/index.json"));
        assert!(!UrlSource::matches("github:openai/skills/foo"));
    }

    #[test]
    fn test_url_resolve_skill_name() {
        let fm = serde_yaml::Mapping::new();
        assert_eq!(
            UrlSource::resolve_skill_name(&fm, "https://x.io/my-skill/SKILL.md"),
            Some("my-skill".to_string())
        );
        assert_eq!(
            UrlSource::resolve_skill_name(&fm, "https://x.io/cool-tool.md"),
            Some("cool-tool".to_string())
        );
        // "skill.md" alone is rejected as useless.
        assert_eq!(UrlSource::resolve_skill_name(&fm, "https://x.io/SKILL.md"), None);
    }

    #[test]
    fn test_url_is_valid_skill_name() {
        assert!(UrlSource::is_valid_skill_name(Some("my-skill")));
        assert!(UrlSource::is_valid_skill_name(Some("a_b-c1")));
        assert!(!UrlSource::is_valid_skill_name(Some("SKILL")));
        assert!(!UrlSource::is_valid_skill_name(Some("readme")));
        assert!(!UrlSource::is_valid_skill_name(Some("1abc")));
        assert!(!UrlSource::is_valid_skill_name(None));
    }

    #[test]
    fn test_wellknown_query_to_index_url() {
        assert_eq!(
            WellKnownSkillSource::query_to_index_url("https://x.io/foo/index.json"),
            Some("https://x.io/foo/index.json".to_string())
        );
        assert_eq!(
            WellKnownSkillSource::query_to_index_url("https://x.io"),
            Some("https://x.io/.well-known/skills/index.json".to_string())
        );
        assert_eq!(
            WellKnownSkillSource::query_to_index_url("https://x.io/.well-known/skills/abc"),
            Some("https://x.io/.well-known/skills/index.json".to_string())
        );
        assert_eq!(WellKnownSkillSource::query_to_index_url("not-a-url"), None);
    }

    #[test]
    fn test_wellknown_parse_identifier() {
        let p = WellKnownSkillSource::parse_identifier(
            "well-known:https://x.io/.well-known/skills/index.json#foo",
        )
        .unwrap();
        assert_eq!(p.skill_name, "foo");
        assert_eq!(p.base_url, "https://x.io/.well-known/skills");
        assert_eq!(p.skill_url, "https://x.io/.well-known/skills/foo");

        let p2 = WellKnownSkillSource::parse_identifier(
            "https://x.io/.well-known/skills/bar/SKILL.md",
        )
        .unwrap();
        assert_eq!(p2.skill_name, "bar");
    }

    #[test]
    fn test_skillssh_normalize_and_candidates() {
        assert_eq!(SkillsShSource::normalize_identifier("skills-sh/a/b/c"), "a/b/c");
        assert_eq!(SkillsShSource::normalize_identifier("skills.sh/a/b/c"), "a/b/c");
        assert_eq!(SkillsShSource::normalize_identifier("a/b/c"), "a/b/c");

        let cands = SkillsShSource::candidate_identifiers("owner/repo/myskill");
        assert_eq!(cands[0], "owner/repo/myskill");
        assert_eq!(cands[1], "owner/repo/skills/myskill");
        assert!(cands.contains(&"owner/repo/.claude/skills/myskill".to_string()));
    }

    #[test]
    fn test_skillssh_token_variants() {
        let v = SkillsShSource::token_variants(Some("My_Cool/Skill"));
        assert!(v.contains("my_cool/skill"));
        assert!(v.contains("skill"));
        assert!(v.contains("my-cool/skill"));
    }

    #[test]
    fn test_skillssh_extract_repo_slug() {
        assert_eq!(
            SkillsShSource::extract_repo_slug("https://github.com/owner/repo/extra"),
            Some("owner/repo".to_string())
        );
        assert_eq!(
            SkillsShSource::extract_repo_slug("owner/repo"),
            Some("owner/repo".to_string())
        );
        assert_eq!(SkillsShSource::extract_repo_slug("solo"), None);
    }

    #[test]
    fn test_clawhub_query_terms_and_score() {
        let terms = ClawHubSource::query_terms("Hello, World!");
        assert_eq!(terms, vec!["hello".to_string(), "world".to_string()]);

        let meta = SkillMeta {
            name: "hello-world".into(),
            identifier: "hello-world".into(),
            description: "a greeting".into(),
            ..Default::default()
        };
        let exact = ClawHubSource::search_score("hello-world", &meta);
        let partial = ClawHubSource::search_score("greeting", &meta);
        assert!(exact > partial);
        assert!(partial > 0);
    }

    #[test]
    fn test_clawhub_normalize_tags() {
        let v = serde_json::json!(["a", "b"]);
        assert_eq!(ClawHubSource::normalize_tags(&v), vec!["a", "b"]);
        let v = serde_json::json!({"latest": "1.0", "stable": "0.9"});
        let tags = ClawHubSource::normalize_tags(&v);
        assert_eq!(tags, vec!["stable".to_string()]);
        assert_eq!(ClawHubSource::normalize_tags(&Value::Null), Vec::<String>::new());
    }

    #[test]
    fn test_clawhub_coerce_payload() {
        let nested = serde_json::json!({
            "skill": {"slug": "x", "name": "X"},
            "latestVersion": "1.2.3"
        });
        let coerced = ClawHubSource::coerce_skill_payload(&nested).unwrap();
        assert_eq!(coerced.get("slug").unwrap().as_str(), Some("x"));
        assert_eq!(coerced.get("latestVersion").unwrap().as_str(), Some("1.2.3"));
    }

    #[test]
    fn test_lobehub_convert_to_skill_md() {
        let agent = serde_json::json!({
            "identifier": "translator",
            "meta": {"title": "Translator", "description": "Translates text", "tags": ["lang", "tools"]},
            "config": {"systemRole": "You translate."}
        });
        let md = LobeHubSource::convert_to_skill_md(&agent);
        assert!(md.starts_with("---\nname: translator\n"));
        assert!(md.contains("description: Translates text"));
        assert!(md.contains("tags: [lang, tools]"));
        assert!(md.contains("# Translator"));
        assert!(md.contains("You translate."));
    }

    #[test]
    fn test_group_thousands() {
        assert_eq!(group_thousands(1234567), "1,234,567");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1000), "1,000");
    }

    #[test]
    fn test_title_case() {
        assert_eq!(title_case("pass"), "Pass");
        assert_eq!(title_case("WARN"), "Warn");
        assert_eq!(title_case(""), "");
    }

    #[test]
    fn test_claude_marketplace_search_identifier_resolution() {
        // Test the identifier-building logic via search would need network;
        // instead test trust_level mapping for known repos.
        assert_eq!(
            ClaudeMarketplaceSource::trust_level_for_impl("anthropics/skills/x"),
            "trusted"
        );
        assert_eq!(
            ClaudeMarketplaceSource::trust_level_for_impl("other/repo/x"),
            "community"
        );
    }

    #[test]
    fn test_lock_file_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("hermes_lock_test_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let lock = HubLockFile::new(tmp.clone());
        assert!(lock.get_installed("foo").is_none());
        lock.record_install(
            "foo",
            "github",
            "a/b/c",
            "community",
            "safe",
            "sha256:abc",
            "foo",
            &["SKILL.md".to_string()],
            None,
        );
        let entry = lock.get_installed("foo").unwrap();
        assert_eq!(entry.get("source").unwrap().as_str(), Some("github"));
        assert_eq!(entry.get("scan_verdict").unwrap().as_str(), Some("safe"));
        let listed = lock.list_installed();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].get("name").unwrap().as_str(), Some("foo"));
        lock.record_uninstall("foo");
        assert!(lock.get_installed("foo").is_none());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_taps_manager() {
        let tmp = std::env::temp_dir().join(format!("hermes_taps_test_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let taps = TapsManager::new(tmp.clone());
        assert!(taps.load().is_empty());
        assert!(taps.add("owner/repo", "skills/"));
        assert!(!taps.add("owner/repo", "skills/")); // duplicate
        assert_eq!(taps.list_taps().len(), 1);
        let typed = taps.list_taps_typed();
        assert_eq!(typed[0].repo, "owner/repo");
        assert_eq!(typed[0].path, "skills/");
        assert!(taps.remove("owner/repo"));
        assert!(!taps.remove("owner/repo"));
        assert!(taps.load().is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_optional_skill_source_scan() {
        let tmp = std::env::temp_dir().join(format!("hermes_opt_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let skill_dir = tmp.join("category").join("my-skill");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: my-skill\ndescription: A test skill\nmetadata:\n  hermes:\n    tags:\n      - x\n---\nbody\n",
        )
        .unwrap();
        let src = OptionalSkillSource::with_dir(tmp.clone());
        let all = src.scan_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "my-skill");
        assert_eq!(all[0].source, "official");
        assert_eq!(all[0].trust_level, "builtin");
        assert_eq!(all[0].identifier, "official/category/my-skill");
        assert_eq!(all[0].tags, vec!["x".to_string()]);

        let bundle = src.fetch("official/category/my-skill").unwrap();
        assert_eq!(bundle.name, "my-skill");
        assert!(bundle.files.contains_key("SKILL.md"));

        let meta = src.inspect("official/my-skill").unwrap();
        assert_eq!(meta.name, "my-skill");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
