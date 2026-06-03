//! WebResearchEnv — RL Environment for Multi-Step Web Research
//! ============================================================
//!
//! Native Rust port of `environments/web_research_env.py`.
//!
//! Trains models to do accurate, efficient, multi-source web research.
//!
//! Reward signals:
//!   - Answer correctness  (LLM judge, 0.0–1.0)
//!   - Source diversity    (used >=2 distinct domains)
//!   - Efficiency          (penalizes excessive tool calls)
//!   - Tool usage          (bonus for actually using web tools)
//!
//! Dataset: FRAMES benchmark (Google, 2024) — multi-hop factual questions
//!   HuggingFace: google/frames-benchmark
//!   Fallback:    built-in sample questions (no HF token needed)
//!
//! The Python original subclasses a large async agent-loop framework
//! (`HermesAgentBaseEnv`) that is not ported to Rust. This module ports the
//! self-contained, pure-logic core that the Python class implements directly:
//!
//!   * configuration defaults / weights ([`WebResearchEnvConfig`])
//!   * the built-in fallback sample dataset ([`sample_questions`])
//!   * dataset setup + train/eval split ([`WebResearchEnv::setup_from_items`])
//!   * round-robin item iteration ([`WebResearchEnv::get_next_item`])
//!   * prompt formatting ([`WebResearchEnv::format_prompt`])
//!   * the multi-signal reward computation ([`WebResearchEnv::compute_reward`])
//!   * the LLM-judge JSON parser + heuristic fallback scorer
//!   * domain extraction
//!   * wandb-style aggregate metric reporting
//!
//! The LLM-judge network call is represented by a pluggable closure so the
//! caller can supply a real `chat_completion` transport; the pure parsing and
//! heuristic fallback are ported faithfully.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Data shapes (minimal local mirrors of the Python framework types)
// ---------------------------------------------------------------------------

/// A single research dataset item: a factual question + reference answer.
///
/// Mirrors the dict shape used throughout the Python env:
/// `{"question", "answer", "difficulty", "hops"}`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResearchItem {
    pub question: String,
    pub answer: String,
    #[serde(default = "default_difficulty")]
    pub difficulty: String,
    #[serde(default = "default_hops")]
    pub hops: u32,
}

fn default_difficulty() -> String {
    "unknown".to_string()
}
fn default_hops() -> u32 {
    2
}

impl ResearchItem {
    pub fn new(
        question: impl Into<String>,
        answer: impl Into<String>,
        difficulty: impl Into<String>,
        hops: u32,
    ) -> Self {
        ResearchItem {
            question: question.into(),
            answer: answer.into(),
            difficulty: difficulty.into(),
            hops,
        }
    }
}

/// A chat message as seen in an [`AgentResult`].
///
/// Only the fields the reward logic inspects are modelled. `tool_calls`
/// follows the OpenAI shape: a list of objects each carrying a
/// `{"function": {"name": ...}}`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
        }
    }
}

/// Result of running the agent loop on a single item.
///
/// Mirrors `environments.agent_loop.AgentResult` (only the fields the reward
/// function reads).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentResult {
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    /// Number of agent turns used. The Python code falls back to the number of
    /// tool-call names collected when this is `0`/falsey.
    #[serde(default)]
    pub turns_used: u32,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the web research RL environment.
///
/// Field defaults match `WebResearchEnvConfig` in the Python source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebResearchEnvConfig {
    /// Weight for answer correctness in reward (LLM judge score).
    pub correctness_weight: f64,
    /// Weight for tool usage signal (did the model actually use web tools?).
    pub tool_usage_weight: f64,
    /// Weight for efficiency signal (penalizes excessive tool calls).
    pub efficiency_weight: f64,
    /// Bonus reward for citing >=2 distinct domains.
    pub diversity_bonus: f64,

    /// Maximum tool calls before efficiency penalty begins.
    pub efficient_max_calls: i64,
    /// Tool call count where efficiency penalty steepens.
    pub heavy_penalty_calls: i64,

    /// Number of held-out items for evaluation.
    pub eval_size: usize,
    /// Fraction of dataset to hold out for evaluation (0.0–1.0).
    pub eval_split_ratio: f64,

    /// HuggingFace dataset name for research questions.
    pub dataset_name: String,

    /// System prompt used when building agent messages.
    pub system_prompt: String,
    /// Maximum agent turns.
    pub max_agent_turns: u32,
}

impl Default for WebResearchEnvConfig {
    fn default() -> Self {
        WebResearchEnvConfig {
            correctness_weight: 0.6,
            tool_usage_weight: 0.2,
            efficiency_weight: 0.2,
            diversity_bonus: 0.1,
            efficient_max_calls: 5,
            heavy_penalty_calls: 10,
            eval_size: 20,
            eval_split_ratio: 0.1,
            dataset_name: "google/frames-benchmark".to_string(),
            system_prompt: "You are a highly capable research agent. When asked a factual question, \
                always use web_search to find current, accurate information before answering. \
                Cite at least 2 sources. Be concise and accurate."
                .to_string(),
            max_agent_turns: 15,
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback sample dataset (used when HuggingFace is unavailable)
// Multi-hop questions requiring real web search to answer.
// ---------------------------------------------------------------------------

/// Built-in fallback dataset. Matches `SAMPLE_QUESTIONS` exactly.
pub fn sample_questions() -> Vec<ResearchItem> {
    vec![
        ResearchItem::new(
            "What is the current population of the capital city of the country that won the 2022 FIFA World Cup?",
            "Buenos Aires has approximately 3 million people in the city proper, or around 15 million in the greater metro area.",
            "medium",
            2,
        ),
        ResearchItem::new(
            "Who is the CEO of the company that makes the most widely used open-source container orchestration platform?",
            "The Linux Foundation oversees Kubernetes. CNCF (Cloud Native Computing Foundation) is the specific body — it does not have a traditional CEO but has an executive director.",
            "medium",
            2,
        ),
        ResearchItem::new(
            "What programming language was used to write the original version of the web framework used by Instagram?",
            "Django, which Instagram was built on, is written in Python.",
            "easy",
            2,
        ),
        ResearchItem::new(
            "In what year was the university founded where the inventor of the World Wide Web currently holds a professorship?",
            "Tim Berners-Lee holds a professorship at MIT (founded 1861) and the University of Southampton (founded 1952).",
            "hard",
            3,
        ),
        ResearchItem::new(
            "What is the latest stable version of the programming language that ranks #1 on the TIOBE index as of this year?",
            "Python is currently #1 on TIOBE. The latest stable version should be verified via the official python.org site.",
            "medium",
            2,
        ),
        ResearchItem::new(
            "How many employees does the parent company of Instagram have?",
            "Meta Platforms (parent of Instagram) employs approximately 70,000+ people as of recent reports.",
            "medium",
            2,
        ),
        ResearchItem::new(
            "What is the current interest rate set by the central bank of the country where the Eiffel Tower is located?",
            "The European Central Bank sets rates for France/eurozone. The current rate should be verified — it has changed frequently in 2023-2025.",
            "hard",
            2,
        ),
        ResearchItem::new(
            "Which company acquired the startup founded by the creator of Oculus VR?",
            "Palmer Luckey founded Oculus VR, which was acquired by Facebook (now Meta). He later founded Anduril Industries.",
            "medium",
            2,
        ),
        ResearchItem::new(
            "What is the market cap of the company that owns the most popular search engine in Russia?",
            "Yandex (now split into separate entities after 2024 restructuring). Current market cap should be verified via financial sources.",
            "hard",
            2,
        ),
        ResearchItem::new(
            "What was the GDP growth rate of the country that hosted the most recent Summer Olympics?",
            "Paris, France hosted the 2024 Summer Olympics. France's recent GDP growth should be verified via World Bank or IMF data.",
            "hard",
            2,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Reward breakdown
// ---------------------------------------------------------------------------

/// Decomposed reward signals returned by [`WebResearchEnv::compute_reward`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardBreakdown {
    pub correctness: f64,
    pub tool_used: f64,
    pub efficiency: f64,
    pub diversity: f64,
    pub total: f64,
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// RL environment for training multi-step web research skills.
///
/// Holds the dataset state + metric buffers; the heavy agent-loop / transport
/// wiring lives in the (not-yet-ported) base framework. The pure scoring and
/// dataset logic is fully ported here.
pub struct WebResearchEnv {
    pub config: WebResearchEnvConfig,

    items: Vec<ResearchItem>,
    eval_items: Vec<ResearchItem>,
    index: usize,

    // Metrics tracking for wandb
    reward_buffer: Vec<f64>,
    correctness_buffer: Vec<f64>,
    tool_usage_buffer: Vec<f64>,
    efficiency_buffer: Vec<f64>,
    diversity_buffer: Vec<f64>,
}

impl WebResearchEnv {
    pub const NAME: &'static str = "web-research";
    /// Default toolsets for this environment — web + file for saving notes.
    pub const DEFAULT_TOOLSETS: [&'static str; 2] = ["web", "file"];

    pub fn new(config: WebResearchEnvConfig) -> Self {
        WebResearchEnv {
            config,
            items: Vec::new(),
            eval_items: Vec::new(),
            index: 0,
            reward_buffer: Vec::new(),
            correctness_buffer: Vec::new(),
            tool_usage_buffer: Vec::new(),
            efficiency_buffer: Vec::new(),
            diversity_buffer: Vec::new(),
        }
    }

    pub fn train_items(&self) -> &[ResearchItem] {
        &self.items
    }

    pub fn eval_items(&self) -> &[ResearchItem] {
        &self.eval_items
    }

    // ------------------------------------------------------------------
    // 1. Setup — load dataset
    // ------------------------------------------------------------------

    /// Set up from a fully-loaded item list (e.g. the FRAMES benchmark
    /// converted to [`ResearchItem`]s), holding out an eval split.
    ///
    /// Mirrors the HuggingFace branch of `setup()`:
    ///   eval_size = max(config.eval_size, len(items) * eval_split_ratio)
    /// then shuffle and split `eval_items = items[:eval_size]`,
    /// `items = items[eval_size:]`.
    ///
    /// `shuffle` lets callers inject a deterministic permutation (e.g. for
    /// tests); pass the identity to skip shuffling.
    pub fn setup_from_items<F>(&mut self, mut items: Vec<ResearchItem>, shuffle: F)
    where
        F: FnOnce(&mut Vec<ResearchItem>),
    {
        let ratio_size = (items.len() as f64 * self.config.eval_split_ratio) as usize;
        let eval_size = self.config.eval_size.max(ratio_size).min(items.len());
        shuffle(&mut items);
        let train = items.split_off(eval_size);
        self.eval_items = items;
        self.items = train;
    }

    /// Set up using the built-in fallback sample dataset.
    ///
    /// Mirrors the fallback branch of `setup()`:
    ///   shuffle; split = max(1, len * 8 // 10);
    ///   items = SAMPLE_QUESTIONS[:split]; eval = SAMPLE_QUESTIONS[split:].
    pub fn setup_fallback<F>(&mut self, shuffle: F)
    where
        F: FnOnce(&mut Vec<ResearchItem>),
    {
        let mut questions = sample_questions();
        shuffle(&mut questions);
        let split = std::cmp::max(1, questions.len() * 8 / 10);
        let split = split.min(questions.len());
        let eval = questions.split_off(split);
        self.items = questions;
        self.eval_items = eval;
    }

    // ------------------------------------------------------------------
    // 2. get_next_item — return the next question
    // ------------------------------------------------------------------

    /// Return the next item, cycling through the dataset.
    ///
    /// Returns `Err` when the dataset is empty (mirrors the `RuntimeError`).
    pub fn get_next_item(&mut self) -> Result<ResearchItem, String> {
        if self.items.is_empty() {
            return Err("Dataset is empty. Did you call setup()?".to_string());
        }
        let item = self.items[self.index % self.items.len()].clone();
        self.index += 1;
        Ok(item)
    }

    // ------------------------------------------------------------------
    // 3. format_prompt — build the user-facing prompt
    // ------------------------------------------------------------------

    /// Format the research question as a task prompt.
    pub fn format_prompt(&self, item: &ResearchItem) -> String {
        format!(
            "Research the following question thoroughly using web search. \
You MUST search the web to find current, accurate information — \
do not rely solely on your training data.\n\n\
Question: {question}\n\n\
Requirements:\n\
- Use web_search and/or web_extract tools to find information\n\
- Search at least 2 different sources\n\
- Provide a concise, accurate answer (2-4 sentences)\n\
- Cite the sources you used",
            question = item.question
        )
    }

    /// Build the agent message list for an item (system + user prompt).
    pub fn build_messages(&self, item: &ResearchItem) -> Vec<ChatMessage> {
        let mut messages = Vec::new();
        if !self.config.system_prompt.is_empty() {
            messages.push(ChatMessage::new("system", self.config.system_prompt.clone()));
        }
        messages.push(ChatMessage::new("user", self.format_prompt(item)));
        messages
    }

    // ------------------------------------------------------------------
    // 4. compute_reward — multi-signal scoring
    // ------------------------------------------------------------------

    /// Extract the final assistant response + collected tool names from a
    /// result, mirroring the reversed-iteration logic in `compute_reward`.
    ///
    /// Returns `(final_response, tools_used)`.
    pub fn extract_response_and_tools(result: &AgentResult) -> (String, Vec<String>) {
        let mut final_response = String::new();
        let mut tools_used: Vec<String> = Vec::new();
        for msg in result.messages.iter().rev() {
            let has_content = msg.content.as_deref().map(|c| !c.is_empty()).unwrap_or(false);
            if msg.role == "assistant" && has_content && final_response.is_empty() {
                final_response = msg.content.clone().unwrap_or_default();
            }
            if msg.role == "assistant" {
                if let Some(tcs) = &msg.tool_calls {
                    for tc in tcs {
                        // fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("");
                        if !name.is_empty() {
                            tools_used.push(name.to_string());
                        }
                    }
                }
            }
        }
        (final_response, tools_used)
    }

    /// Compute the multi-signal reward for a single item/result.
    ///
    ///   correctness_weight * correctness  — judged via `judge`
    ///   tool_usage_weight  * tool_used    — binary: web tool used?
    ///   efficiency_weight  * efficiency   — penalizes wasteful tool usage
    ///   + diversity_bonus                 — source diversity (>=2 domains)
    ///
    /// `judge` is invoked as `judge(question, expected, model_answer)` and must
    /// return a score in `[0,1]`; supply [`WebResearchEnv::heuristic_score`] (or
    /// a closure wrapping it) for the offline/no-LLM path.
    ///
    /// Records the breakdown into the metric buffers (as the Python code does).
    pub fn compute_reward<J>(
        &mut self,
        item: &ResearchItem,
        result: &AgentResult,
        mut judge: J,
    ) -> RewardBreakdown
    where
        J: FnMut(&str, &str, &str) -> f64,
    {
        let (final_response, tools_used) = Self::extract_response_and_tools(result);

        // tool_call_count = result.turns_used or len(tools_used)
        let tool_call_count: i64 = if result.turns_used != 0 {
            result.turns_used as i64
        } else {
            tools_used.len() as i64
        };

        let cfg = &self.config;

        // ---- Signal 1: Answer correctness (LLM judge) ----------------
        let correctness = judge(&item.question, &item.answer, &final_response);

        // ---- Signal 2: Web tool usage --------------------------------
        const WEB_TOOLS: [&str; 4] = ["web_search", "web_extract", "search", "firecrawl"];
        let tool_used = if tools_used.iter().any(|t| WEB_TOOLS.contains(&t.as_str())) {
            1.0
        } else {
            0.0
        };

        // ---- Signal 3: Efficiency ------------------------------------
        let efficiency = if tool_call_count <= cfg.efficient_max_calls {
            1.0
        } else if tool_call_count <= cfg.heavy_penalty_calls {
            1.0 - (tool_call_count - cfg.efficient_max_calls) as f64 * 0.08
        } else {
            (1.0 - (tool_call_count - cfg.efficient_max_calls) as f64 * 0.12).max(0.0)
        };

        // ---- Bonus: Source diversity ---------------------------------
        let domains = Self::extract_domains(&final_response);
        let diversity = if domains.len() >= 2 {
            cfg.diversity_bonus
        } else {
            0.0
        };

        // ---- Combine ------------------------------------------------
        let mut reward = cfg.correctness_weight * correctness
            + cfg.tool_usage_weight * tool_used
            + cfg.efficiency_weight * efficiency
            + diversity;
        reward = reward.clamp(0.0, 1.0);

        // Track for wandb
        self.reward_buffer.push(reward);
        self.correctness_buffer.push(correctness);
        self.tool_usage_buffer.push(tool_used);
        self.efficiency_buffer.push(efficiency);
        self.diversity_buffer.push(diversity);

        RewardBreakdown {
            correctness,
            tool_used,
            efficiency,
            diversity,
            total: reward,
        }
    }

    /// Pop the last entry from each metric buffer (down to `target_len`).
    ///
    /// Used by `evaluate()` to roll back the buffers so eval rollouts don't
    /// pollute training metrics. Mirrors the buffer rollback loop.
    pub fn rollback_buffers_to(&mut self, target_len: usize) {
        for buf in [
            &mut self.reward_buffer,
            &mut self.correctness_buffer,
            &mut self.tool_usage_buffer,
            &mut self.efficiency_buffer,
            &mut self.diversity_buffer,
        ] {
            if buf.len() > target_len {
                buf.pop();
            }
        }
    }

    pub fn correctness_buffer_len(&self) -> usize {
        self.correctness_buffer.len()
    }

    /// Read the correctness value recorded at `idx` (used by eval to recover
    /// the score `compute_reward` appended), defaulting to 0.0.
    pub fn correctness_at(&self, idx: usize) -> f64 {
        self.correctness_buffer.get(idx).copied().unwrap_or(0.0)
    }

    // ------------------------------------------------------------------
    // 6. wandb_log — custom metrics
    // ------------------------------------------------------------------

    /// Produce the wandb metric map from the current buffers and clear them.
    ///
    /// Mirrors `wandb_log`: only emits when the reward buffer is non-empty,
    /// then clears all buffers. Returns the metrics merged into `wandb_metrics`.
    pub fn wandb_log(&mut self, mut wandb_metrics: serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
        if !self.reward_buffer.is_empty() {
            let n = self.reward_buffer.len();
            let nf = n as f64;
            let mean = |b: &[f64]| -> f64 { b.iter().sum::<f64>() / nf };

            wandb_metrics.insert("train/mean_reward".into(), mean(&self.reward_buffer).into());
            wandb_metrics.insert("train/mean_correctness".into(), mean(&self.correctness_buffer).into());
            wandb_metrics.insert("train/mean_tool_usage".into(), mean(&self.tool_usage_buffer).into());
            wandb_metrics.insert("train/mean_efficiency".into(), mean(&self.efficiency_buffer).into());
            wandb_metrics.insert("train/mean_diversity".into(), mean(&self.diversity_buffer).into());
            wandb_metrics.insert("train/total_rollouts".into(), n.into());

            // Accuracy buckets
            let correct_rate =
                self.correctness_buffer.iter().filter(|&&c| c >= 0.7).count() as f64 / nf;
            let tool_usage_rate =
                self.tool_usage_buffer.iter().filter(|&&t| t > 0.0).count() as f64 / nf;
            wandb_metrics.insert("train/correct_rate".into(), correct_rate.into());
            wandb_metrics.insert("train/tool_usage_rate".into(), tool_usage_rate.into());

            // Clear buffers
            self.reward_buffer.clear();
            self.correctness_buffer.clear();
            self.tool_usage_buffer.clear();
            self.efficiency_buffer.clear();
            self.diversity_buffer.clear();
        }
        wandb_metrics
    }

    // ------------------------------------------------------------------
    // Private helpers (ported as associated fns)
    // ------------------------------------------------------------------

    /// Extract the score float from an LLM judge JSON response.
    ///
    /// Strips ``` / ```json fences and parses JSON, reading `score`; on failure
    /// falls back to a regex for `"score": <num>`. Returns `Some` only when the
    /// score lies in `[0,1]`.
    pub fn parse_judge_json(text: &str) -> Option<f64> {
        // clean = re.sub(r"```(?:json)?|```", "", text).strip()
        let fence_re = regex::Regex::new(r"```(?:json)?|```").unwrap();
        let clean = fence_re.replace_all(text, "");
        let clean = clean.trim();

        if let Ok(data) = serde_json::from_str::<Value>(clean) {
            // float(data.get("score", -1))
            let score = match data.get("score") {
                Some(Value::Number(n)) => n.as_f64(),
                Some(Value::String(s)) => s.trim().parse::<f64>().ok(),
                Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
                _ => Some(-1.0),
            };
            if let Some(score) = score {
                if (0.0..=1.0).contains(&score) {
                    return Some(score);
                }
            }
            // Python: if json parsed but score out of range / non-numeric raising,
            // it would fall through to the regex only on exception. A successfully
            // parsed JSON with an out-of-range score returns None below.
            return None;
        }

        // except: regex fallback
        let score_re = regex::Regex::new(r#""score"\s*:\s*([0-9.]+)"#).unwrap();
        if let Some(caps) = score_re.captures(text) {
            if let Ok(score) = caps[1].parse::<f64>() {
                if (0.0..=1.0).contains(&score) {
                    return Some(score);
                }
            }
        }
        None
    }

    /// Lightweight keyword-overlap score used as a fallback when no LLM judge
    /// is available. Mirrors `_heuristic_score`.
    pub fn heuristic_score(expected: &str, model_answer: &str) -> f64 {
        // Exactly the distinct words in the Python `stopwords` set.
        const STOPWORDS: [&str; 28] = [
            "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "for",
            "with", "and", "or", "but", "it", "its", "this", "that", "as", "by", "from", "be",
            "has", "have", "had",
        ];
        let stopwords: HashSet<&str> = STOPWORDS.iter().copied().collect();

        let tokenize = |text: &str| -> HashSet<String> {
            let word_re = regex::Regex::new(r"\b\w+\b").unwrap();
            word_re
                .find_iter(&text.to_lowercase())
                .map(|m| m.as_str().to_string())
                .filter(|t| !stopwords.contains(t.as_str()) && t.chars().count() > 2)
                .collect()
        };

        let expected_tokens = tokenize(expected);
        let answer_tokens = tokenize(model_answer);

        if expected_tokens.is_empty() {
            return 0.5;
        }

        let overlap = expected_tokens.intersection(&answer_tokens).count();
        let union = expected_tokens.union(&answer_tokens).count();

        let jaccard = if union > 0 {
            overlap as f64 / union as f64
        } else {
            0.0
        };
        let recall = overlap as f64 / expected_tokens.len() as f64;
        (0.4 * jaccard + 0.6 * recall).min(1.0)
    }

    /// Extract unique domains from URLs cited in the response.
    ///
    /// Mirrors `_extract_domains`: finds `https?://...` URLs, parses the host,
    /// lowercases and strips a leading `www.` prefix.
    pub fn extract_domains(text: &str) -> HashSet<String> {
        let url_re = regex::Regex::new(r#"https?://[^\s\)>\]"']+"#).unwrap();
        let mut domains: HashSet<String> = HashSet::new();
        for m in url_re.find_iter(text) {
            let url = m.as_str();
            if let Ok(parsed) = url::Url::parse(url) {
                if let Some(host) = parsed.host_str() {
                    // domain = parsed.netloc.lower().lstrip("www.")
                    // Note: Python's lstrip("www.") strips ANY leading chars in
                    // the set {'w','.'}, not the literal prefix. Reproduce that.
                    let lowered = host.to_lowercase();
                    let domain = lowered.trim_start_matches(['w', '.']);
                    if !domain.is_empty() {
                        domains.insert(domain.to_string());
                    }
                }
            }
        }
        domains
    }
}

/// Compute the aggregate eval metrics from a set of per-sample records,
/// mirroring the `eval_metrics` dict built in `evaluate()`.
///
/// Each tuple is `(correctness, reward, tool_calls)`.
pub fn eval_metrics(samples: &[(f64, f64, i64)]) -> serde_json::Map<String, Value> {
    let n = samples.len();
    let nf = n as f64;
    let mut m = serde_json::Map::new();
    let mean_correctness = if n > 0 {
        samples.iter().map(|s| s.0).sum::<f64>() / nf
    } else {
        0.0
    };
    let mean_reward = if n > 0 {
        samples.iter().map(|s| s.1).sum::<f64>() / nf
    } else {
        0.0
    };
    let mean_tool_calls = if n > 0 {
        samples.iter().map(|s| s.2 as f64).sum::<f64>() / nf
    } else {
        0.0
    };
    let tool_usage_rate = if n > 0 {
        samples.iter().filter(|s| s.2 > 0).count() as f64 / nf
    } else {
        0.0
    };
    m.insert("eval/mean_correctness".into(), mean_correctness.into());
    m.insert("eval/mean_reward".into(), mean_reward.into());
    m.insert("eval/mean_tool_calls".into(), mean_tool_calls.into());
    m.insert("eval/tool_usage_rate".into(), tool_usage_rate.into());
    m.insert("eval/n_items".into(), n.into());
    m
}

/// Build the LLM-judge prompt for an answer (mirrors `_llm_judge`'s prompt).
pub fn judge_prompt(question: &str, expected: &str, model_answer: &str) -> String {
    format!(
        "You are an impartial judge evaluating the quality of an AI research answer.\n\n\
Question: {question}\n\n\
Reference answer: {expected}\n\n\
Model answer: {model_answer}\n\n\
Score the model answer on a scale from 0.0 to 1.0 where:\n  \
1.0 = fully correct and complete\n  \
0.7 = mostly correct with minor gaps\n  \
0.4 = partially correct\n  \
0.1 = mentions relevant topic but wrong or very incomplete\n  \
0.0 = completely wrong or no answer\n\n\
Consider: factual accuracy, completeness, and relevance.\n\
Respond with ONLY a JSON object: {{\"score\": <float>, \"reason\": \"<one sentence>\"}}",
        question = question,
        expected = expected,
        model_answer = model_answer
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn no_shuffle(_: &mut Vec<ResearchItem>) {}

    #[test]
    fn sample_dataset_has_ten_items() {
        assert_eq!(sample_questions().len(), 10);
    }

    #[test]
    fn setup_fallback_splits_8_2() {
        let mut env = WebResearchEnv::new(WebResearchEnvConfig::default());
        env.setup_fallback(no_shuffle);
        // split = max(1, 10 * 8 // 10) = 8
        assert_eq!(env.train_items().len(), 8);
        assert_eq!(env.eval_items().len(), 2);
        // order preserved with no_shuffle
        assert_eq!(env.train_items()[0], sample_questions()[0]);
    }

    #[test]
    fn setup_from_items_holds_out_eval_size() {
        let mut env = WebResearchEnv::new(WebResearchEnvConfig::default());
        // 100 items: eval_size = max(20, 100*0.1=10) = 20
        let items: Vec<ResearchItem> = (0..100)
            .map(|i| ResearchItem::new(format!("q{i}"), format!("a{i}"), "medium", 2))
            .collect();
        env.setup_from_items(items, no_shuffle);
        assert_eq!(env.eval_items().len(), 20);
        assert_eq!(env.train_items().len(), 80);
        // split: eval = items[:20], train = items[20:]
        assert_eq!(env.eval_items()[0].question, "q0");
        assert_eq!(env.train_items()[0].question, "q20");
    }

    #[test]
    fn get_next_item_cycles() {
        let mut env = WebResearchEnv::new(WebResearchEnvConfig::default());
        env.setup_fallback(no_shuffle);
        let n = env.train_items().len();
        let first = env.get_next_item().unwrap();
        for _ in 0..(n - 1) {
            env.get_next_item().unwrap();
        }
        // wrapped around
        let wrapped = env.get_next_item().unwrap();
        assert_eq!(first, wrapped);
    }

    #[test]
    fn get_next_item_empty_errors() {
        let mut env = WebResearchEnv::new(WebResearchEnvConfig::default());
        assert!(env.get_next_item().is_err());
    }

    #[test]
    fn parse_judge_json_plain() {
        assert_eq!(
            WebResearchEnv::parse_judge_json(r#"{"score": 0.7, "reason": "ok"}"#),
            Some(0.7)
        );
    }

    #[test]
    fn parse_judge_json_fenced() {
        let t = "```json\n{\"score\": 0.4, \"reason\": \"meh\"}\n```";
        assert_eq!(WebResearchEnv::parse_judge_json(t), Some(0.4));
    }

    #[test]
    fn parse_judge_json_out_of_range() {
        assert_eq!(WebResearchEnv::parse_judge_json(r#"{"score": 1.5}"#), None);
    }

    #[test]
    fn parse_judge_json_regex_fallback() {
        // Not valid JSON, but contains a score field.
        let t = "Here is my verdict \"score\": 0.9 and some trailing prose";
        assert_eq!(WebResearchEnv::parse_judge_json(t), Some(0.9));
    }

    #[test]
    fn parse_judge_json_no_score() {
        assert_eq!(WebResearchEnv::parse_judge_json("no score here"), None);
    }

    #[test]
    fn heuristic_score_empty_answer() {
        // empty model answer -> tokens empty, overlap 0, recall 0
        assert_eq!(WebResearchEnv::heuristic_score("Paris France", ""), 0.0);
    }

    #[test]
    fn heuristic_score_empty_expected_returns_half() {
        assert_eq!(WebResearchEnv::heuristic_score("the a is", "anything"), 0.5);
    }

    #[test]
    fn heuristic_score_perfect_overlap() {
        let s = WebResearchEnv::heuristic_score("Python Django Instagram", "Python Django Instagram");
        // jaccard=1, recall=1 -> 1.0
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn extract_domains_strips_www() {
        let text = "See https://www.example.com/page and http://docs.rust-lang.org/book";
        let domains = WebResearchEnv::extract_domains(text);
        assert!(domains.contains("example.com"));
        assert!(domains.contains("docs.rust-lang.org"));
        assert_eq!(domains.len(), 2);
    }

    #[test]
    fn extract_domains_none() {
        assert!(WebResearchEnv::extract_domains("no urls here").is_empty());
    }

    #[test]
    fn extract_response_and_tools_works() {
        let result = AgentResult {
            messages: vec![
                ChatMessage::new("user", "q"),
                ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: Some(vec![json!({"function": {"name": "web_search"}})]),
                },
                ChatMessage::new("assistant", "Final answer with https://a.com"),
            ],
            turns_used: 0,
        };
        let (resp, tools) = WebResearchEnv::extract_response_and_tools(&result);
        assert_eq!(resp, "Final answer with https://a.com");
        assert_eq!(tools, vec!["web_search".to_string()]);
    }

    #[test]
    fn compute_reward_full_signals() {
        let mut env = WebResearchEnv::new(WebResearchEnvConfig::default());
        let item = ResearchItem::new("q?", "Paris France capital", "medium", 2);
        let result = AgentResult {
            messages: vec![
                ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: Some(vec![
                        json!({"function": {"name": "web_search"}}),
                        json!({"function": {"name": "web_extract"}}),
                    ]),
                },
                ChatMessage::new(
                    "assistant",
                    "Answer: Paris France capital. Sources: https://www.a.com https://b.org",
                ),
            ],
            turns_used: 2, // <= efficient_max_calls=5 -> efficiency 1.0
        };
        // Judge returns perfect correctness.
        let bd = env.compute_reward(&item, &result, |_, _, _| 1.0);
        assert_eq!(bd.tool_used, 1.0);
        assert_eq!(bd.efficiency, 1.0);
        assert_eq!(bd.diversity, 0.1); // 2 distinct domains
        // reward = 0.6*1 + 0.2*1 + 0.2*1 + 0.1 = 1.1 clamped to 1.0
        assert_eq!(bd.total, 1.0);
        assert_eq!(env.correctness_buffer_len(), 1);
    }

    #[test]
    fn compute_reward_efficiency_tiers() {
        let cfg = WebResearchEnvConfig::default();
        let mut env = WebResearchEnv::new(cfg);
        let item = ResearchItem::new("q?", "expected", "medium", 2);

        // 7 turns: between efficient(5) and heavy(10): 1 - (7-5)*0.08 = 0.84
        let result = AgentResult {
            messages: vec![ChatMessage::new("assistant", "no urls")],
            turns_used: 7,
        };
        let bd = env.compute_reward(&item, &result, |_, _, _| 0.0);
        assert!((bd.efficiency - 0.84).abs() < 1e-9);

        // 20 turns: max(0, 1 - (20-5)*0.12) = max(0, 1-1.8)=0
        let result2 = AgentResult {
            messages: vec![ChatMessage::new("assistant", "x")],
            turns_used: 20,
        };
        let bd2 = env.compute_reward(&item, &result2, |_, _, _| 0.0);
        assert_eq!(bd2.efficiency, 0.0);
    }

    #[test]
    fn compute_reward_falls_back_to_tool_count_when_turns_zero() {
        let mut env = WebResearchEnv::new(WebResearchEnvConfig::default());
        let item = ResearchItem::new("q?", "expected", "medium", 2);
        // 6 tool calls, turns_used=0 -> tool_call_count=6 -> efficiency 1-(6-5)*.08=0.92
        let tcs: Vec<Value> = (0..6)
            .map(|_| json!({"function": {"name": "search"}}))
            .collect();
        let result = AgentResult {
            messages: vec![ChatMessage {
                role: "assistant".into(),
                content: Some("ans".into()),
                tool_calls: Some(tcs),
            }],
            turns_used: 0,
        };
        let bd = env.compute_reward(&item, &result, |_, _, _| 0.0);
        assert!((bd.efficiency - 0.92).abs() < 1e-9);
        assert_eq!(bd.tool_used, 1.0); // "search" is a web tool
    }

    #[test]
    fn wandb_log_emits_and_clears() {
        let mut env = WebResearchEnv::new(WebResearchEnvConfig::default());
        let item = ResearchItem::new("q?", "expected", "medium", 2);
        let result = AgentResult {
            messages: vec![ChatMessage::new("assistant", "ans")],
            turns_used: 1,
        };
        env.compute_reward(&item, &result, |_, _, _| 1.0);
        let m = env.wandb_log(serde_json::Map::new());
        assert!(m.contains_key("train/mean_reward"));
        assert_eq!(m.get("train/total_rollouts").unwrap().as_u64(), Some(1));
        // buffers cleared
        assert_eq!(env.correctness_buffer_len(), 0);
        // second call: empty buffers -> no metrics
        let m2 = env.wandb_log(serde_json::Map::new());
        assert!(m2.is_empty());
    }

    #[test]
    fn rollback_buffers() {
        let mut env = WebResearchEnv::new(WebResearchEnvConfig::default());
        let item = ResearchItem::new("q?", "expected", "medium", 2);
        let result = AgentResult {
            messages: vec![ChatMessage::new("assistant", "ans")],
            turns_used: 1,
        };
        env.compute_reward(&item, &result, |_, _, _| 0.5);
        let before = env.correctness_buffer_len();
        env.compute_reward(&item, &result, |_, _, _| 0.9);
        assert_eq!(env.correctness_at(before), 0.9);
        env.rollback_buffers_to(before);
        assert_eq!(env.correctness_buffer_len(), before);
    }

    #[test]
    fn eval_metrics_aggregates() {
        let samples = vec![(1.0, 0.9, 3i64), (0.0, 0.1, 0)];
        let m = eval_metrics(&samples);
        assert_eq!(m.get("eval/n_items").unwrap().as_u64(), Some(2));
        assert!((m.get("eval/mean_correctness").unwrap().as_f64().unwrap() - 0.5).abs() < 1e-9);
        assert!((m.get("eval/tool_usage_rate").unwrap().as_f64().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn format_prompt_contains_question() {
        let env = WebResearchEnv::new(WebResearchEnvConfig::default());
        let item = ResearchItem::new("What is X?", "X is Y", "easy", 1);
        let p = env.format_prompt(&item);
        assert!(p.contains("Question: What is X?"));
        assert!(p.contains("web_search"));
    }

    #[test]
    fn build_messages_includes_system() {
        let env = WebResearchEnv::new(WebResearchEnvConfig::default());
        let item = ResearchItem::new("q", "a", "easy", 1);
        let msgs = env.build_messages(&item);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
    }
}
