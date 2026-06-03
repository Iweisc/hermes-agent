//! AgenticOPDEnv — On-Policy Distillation for Agentic Tool-Calling Tasks
//! (native Rust port of `environments/agentic_opd_env.py`).
//!
//! This is the first Atropos environment to populate the
//! `distill_token_ids` / `distill_logprobs` fields on a `ScoredDataGroup`,
//! enabling on-policy distillation (OPD) training.
//!
//! Key idea (from OpenClaw-RL, Princeton 2026): every time an agent receives a
//! next-state signal (tool result, error trace, test verdict), that signal
//! contains hindsight information about how the agent's PREVIOUS response could
//! have been better. This environment:
//!
//!   1. Runs standard agentic rollouts (tool-calling agent loop)
//!   2. Walks the conversation to find (assistant_turn, next_state) pairs
//!   3. Uses an LLM judge to extract "hints" from next-state signals
//!   4. Builds an enhanced prompt (original context + hint)
//!   5. Scores the student's response tokens under the enhanced distribution
//!      using VLLM's prompt_logprobs
//!   6. Packages the teacher's top-K predictions as `distill_token_ids` /
//!      `distill_logprobs` on the `ScoredDataGroup`
//!
//! The Python module is an Atropos environment subclass that wires into a large
//! async framework (HuggingFace `datasets`, an inference `server`, a tokenizer,
//! wandb, etc.). That framework is not yet ported to Rust, so this module ports
//! the *self-contained, deterministic logic* faithfully:
//!
//!   - the built-in coding-task fallback set,
//!   - the hint-extraction prompt construction & parsing,
//!   - majority-voted hint selection,
//!   - hint-appending to a message list,
//!   - turn-pair extraction from a conversation,
//!   - token-span search,
//!   - reward computation,
//!   - prompt formatting,
//!   - config defaults,
//!   - the per-position distill array assembly.
//!
//! Network / tokenizer / inference pieces are expressed as traits/parameters so
//! callers can plug in their concrete server + tokenizer once those are ported.

use std::collections::BTreeSet;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

// ════════════════════════════════════════════════════════════════════════
// Built-in coding tasks (fallback when no HF dataset is configured)
// ════════════════════════════════════════════════════════════════════════

/// A single coding task item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodingTask {
    pub task: String,
    pub test_code: String,
    pub difficulty: String,
}

impl CodingTask {
    pub fn new(task: impl Into<String>, test_code: impl Into<String>, difficulty: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            test_code: test_code.into(),
            difficulty: difficulty.into(),
        }
    }
}

/// The built-in coding tasks, mirroring `BUILTIN_CODING_TASKS` in Python.
pub fn builtin_coding_tasks() -> Vec<CodingTask> {
    vec![
        CodingTask::new(
            "Write a Python function `fizzbuzz(n)` that returns a list of strings from 1 to n. \
For multiples of 3 return 'Fizz', for multiples of 5 return 'Buzz', \
for multiples of both return 'FizzBuzz', otherwise the number as a string.",
            "from solution import fizzbuzz\n\
assert fizzbuzz(15) == ['1','2','Fizz','4','Buzz','Fizz','7','8','Fizz','Buzz','11','Fizz','13','14','FizzBuzz']\n\
assert fizzbuzz(1) == ['1']\n\
assert fizzbuzz(0) == []\n\
print('All tests passed!')\n",
            "easy",
        ),
        CodingTask::new(
            "Write a Python function `is_palindrome(s)` that checks if a string is a palindrome, \
ignoring case and non-alphanumeric characters. Return True or False.",
            "from solution import is_palindrome\n\
assert is_palindrome('A man, a plan, a canal: Panama') == True\n\
assert is_palindrome('race a car') == False\n\
assert is_palindrome('') == True\n\
assert is_palindrome('Was it a car or a cat I saw?') == True\n\
print('All tests passed!')\n",
            "easy",
        ),
        CodingTask::new(
            "Write a Python function `two_sum(nums, target)` that returns the indices of the two \
numbers in `nums` that add up to `target`. Assume exactly one solution exists. \
Return a list of two indices [i, j] where i < j.",
            "from solution import two_sum\n\
assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n\
assert two_sum([3, 2, 4], 6) == [1, 2]\n\
assert two_sum([3, 3], 6) == [0, 1]\n\
print('All tests passed!')\n",
            "easy",
        ),
        CodingTask::new(
            "Write a Python function `flatten(lst)` that takes an arbitrarily nested list and \
returns a flat list of all elements. For example, flatten([1, [2, [3, 4], 5]]) \
should return [1, 2, 3, 4, 5].",
            "from solution import flatten\n\
assert flatten([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5]\n\
assert flatten([]) == []\n\
assert flatten([1, 2, 3]) == [1, 2, 3]\n\
assert flatten([[[[1]]]]) == [1]\n\
assert flatten([1, [2], [[3]], [[[4]]]]) == [1, 2, 3, 4]\n\
print('All tests passed!')\n",
            "medium",
        ),
        CodingTask::new(
            "Write a Python function `longest_common_prefix(strs)` that finds the longest \
common prefix string amongst a list of strings. If there is no common prefix, \
return an empty string.",
            "from solution import longest_common_prefix\n\
assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'\n\
assert longest_common_prefix(['dog', 'racecar', 'car']) == ''\n\
assert longest_common_prefix(['interspecies', 'interstellar', 'interstate']) == 'inters'\n\
assert longest_common_prefix(['a']) == 'a'\n\
assert longest_common_prefix([]) == ''\n\
print('All tests passed!')\n",
            "easy",
        ),
        CodingTask::new(
            "Write a Python function `group_anagrams(strs)` that groups anagrams together. \
Return a list of lists, where each inner list contains strings that are anagrams of \
each other. The order of groups and strings within groups does not matter.",
            "from solution import group_anagrams\n\
result = group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat'])\n\
result_sorted = sorted([sorted(g) for g in result])\n\
assert result_sorted == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]\n\
assert group_anagrams([]) == []\n\
assert group_anagrams(['a']) == [['a']]\n\
print('All tests passed!')\n",
            "medium",
        ),
        CodingTask::new(
            "Write a Python function `valid_parentheses(s)` that determines if a string \
containing just '(', ')', '{', '}', '[' and ']' is valid. A string is valid if \
open brackets are closed by the same type and in the correct order.",
            "from solution import valid_parentheses\n\
assert valid_parentheses('()') == True\n\
assert valid_parentheses('()[]{}') == True\n\
assert valid_parentheses('(]') == False\n\
assert valid_parentheses('([)]') == False\n\
assert valid_parentheses('{[]}') == True\n\
assert valid_parentheses('') == True\n\
print('All tests passed!')\n",
            "easy",
        ),
        CodingTask::new(
            "Write a Python function `merge_intervals(intervals)` that merges overlapping \
intervals. Each interval is a list [start, end]. Return the merged intervals sorted \
by start time.",
            "from solution import merge_intervals\n\
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n\
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n\
assert merge_intervals([[1,4],[0,4]]) == [[0,4]]\n\
assert merge_intervals([]) == []\n\
assert merge_intervals([[1,2]]) == [[1,2]]\n\
print('All tests passed!')\n",
            "medium",
        ),
    ]
}

// ════════════════════════════════════════════════════════════════════════
// Hint extraction prompts (adapted from OpenClaw-RL)
// ════════════════════════════════════════════════════════════════════════

/// System prompt for the hindsight-hint extraction judge.
pub const HINT_JUDGE_SYSTEM: &str = "You are a process reward model used for hindsight hint extraction.\n\
You are given:\n\
1) The assistant response at turn t.\n\
2) The next state at turn t+1, along with its **role**.\n\n\
## Understanding the next state's role\n\
- role='user': A reply from the user (follow-up, correction, new request, etc.).\n\
- role='tool': The return value of a tool the assistant invoked. \
This content was NOT available before the assistant's action — \
it exists BECAUSE the assistant called the tool. \
A successful, non-error tool output generally means the assistant's \
action was appropriate; do NOT treat it as information the assistant \
should have already known.\n\n\
Your goal is to decide whether the next state reveals useful hindsight information\n\
that could have helped improve the assistant response at turn t.\n\n\
Output format rules (strict):\n\
- You MUST include exactly one final decision token: \\boxed{1} or \\boxed{-1}.\n\
- If and only if decision is \\boxed{1}, provide a concise, information-dense hint in 1-3 sentences,\n\
  wrapped between [HINT_START] and [HINT_END].\n\
- If decision is \\boxed{-1}, do not provide a hint block.\n\
- Hint must be concrete and actionable for improving the previous response.";

fn boxed_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\\boxed\{(-?\d+)\}").unwrap())
}

fn hint_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    // (?s) = DOTALL so `.` matches newlines.
    RE.get_or_init(|| Regex::new(r"(?s)\[HINT_START\](.*?)\[HINT_END\]").unwrap())
}

/// A chat message `{role, content}` as used in the conversation lists.
pub type ChatMessage = Map<String, Value>;

/// Build the messages for the hint extraction judge.
///
/// Mirrors `_build_hint_judge_messages`.
pub fn build_hint_judge_messages(
    response_text: &str,
    next_state_text: &str,
    next_state_role: &str,
) -> Vec<ChatMessage> {
    let user = format!(
        "## Assistant response (turn t)\n{response_text}\n\n\
## Next state (turn t+1) [role: {next_state_role}]\n{next_state_text}\n\n\
Now output your decision and (if positive) the hint in the required format."
    );
    let mut sys = Map::new();
    sys.insert("role".to_string(), json!("system"));
    sys.insert("content".to_string(), json!(HINT_JUDGE_SYSTEM));
    let mut usr = Map::new();
    usr.insert("role".to_string(), json!("user"));
    usr.insert("content".to_string(), json!(user));
    vec![sys, usr]
}

/// Parse the judge's boxed decision and hint text.
///
/// Mirrors `_parse_hint_result`. Returns `(score, hint)` where `score` is
/// `Some(1)`, `Some(-1)`, or `None` (any other / missing value is `None`).
pub fn parse_hint_result(text: &str) -> (Option<i64>, String) {
    let mut score: Option<i64> = None;
    if let Some(last) = boxed_re().captures_iter(text).last() {
        if let Some(m) = last.get(1) {
            score = m.as_str().parse::<i64>().ok();
        }
    }
    if score != Some(1) && score != Some(-1) {
        score = None;
    }
    let hint = hint_re()
        .captures_iter(text)
        .last()
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().trim().to_string())
        .unwrap_or_default();
    (score, hint)
}

/// A single judge vote.
#[derive(Debug, Clone, PartialEq)]
pub struct HintVote {
    pub score: Option<i64>,
    pub hint: String,
}

/// Select the best hint from majority-voted judge results.
///
/// Mirrors `_select_best_hint`: keep votes with `score == 1` and a trimmed hint
/// longer than 10 chars, then return the one with the longest trimmed hint.
/// On ties, Python's `max` keeps the first-seen maximum — we replicate that by
/// using `>` (strictly greater) when scanning in order.
pub fn select_best_hint(votes: &[HintVote]) -> Option<&HintVote> {
    let mut best: Option<&HintVote> = None;
    let mut best_len: usize = 0;
    for v in votes {
        if v.score == Some(1) && v.hint.trim().chars().count() > 10 {
            let l = v.hint.trim().chars().count();
            if best.is_none() || l > best_len {
                best = Some(v);
                best_len = l;
            }
        }
    }
    best
}

/// Coerce a message `content` value into a single string, joining list parts.
///
/// Mirrors the Python handling where content may be a list of `{text: ...}`
/// dicts or plain strings.
fn content_to_string(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .map(|c| match c {
                Value::Object(o) => o
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                other => value_str(other),
            })
            .collect::<Vec<_>>()
            .join(" "),
        Value::Null => String::new(),
        other => value_str(other),
    }
}

/// Python `str(x)`-ish rendering for non-string scalar content.
fn value_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "None".to_string(),
        Value::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        other => other.to_string(),
    }
}

/// Clone messages and append the hint to the last user message.
///
/// Mirrors `_append_hint_to_messages`.
pub fn append_hint_to_messages(messages: &[ChatMessage], hint: &str) -> Vec<ChatMessage> {
    if messages.is_empty() {
        let mut m = Map::new();
        m.insert("role".to_string(), json!("user"));
        m.insert(
            "content".to_string(),
            json!(format!("[user's hint / instruction]\n{hint}")),
        );
        return vec![m];
    }

    let mut cloned: Vec<ChatMessage> = messages.to_vec();

    // Find last user message.
    let mut target_idx: Option<usize> = None;
    for i in (0..cloned.len()).rev() {
        if cloned[i].get("role").and_then(|r| r.as_str()) == Some("user") {
            target_idx = Some(i);
            break;
        }
    }
    let target_idx = target_idx.unwrap_or(cloned.len() - 1);

    let content = cloned[target_idx]
        .get("content")
        .map(content_to_string)
        .unwrap_or_default();
    let suffix = format!("\n\n[user's hint / instruction]\n{}", hint.trim());
    let new_content = format!("{content}{suffix}");
    let new_content = new_content.trim().to_string();
    cloned[target_idx].insert("content".to_string(), json!(new_content));
    cloned
}

// ════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════

/// Configuration for the agentic OPD environment.
///
/// Mirrors `AgenticOPDConfig` (the OPD-specific subset plus the shared agent
/// fields this module actually reads). Defaults match `config_init` where they
/// differ from the field-level defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgenticOPDConfig {
    // --- OPD settings ---
    pub opd_enabled: bool,
    pub distill_topk: usize,
    pub prm_votes: usize,
    pub hint_max_next_state_chars: usize,

    // --- Reward settings ---
    pub correctness_weight: f64,
    pub efficiency_weight: f64,
    pub tool_usage_weight: f64,

    // --- Dataset ---
    pub dataset_name: Option<String>,

    // --- Eval ---
    pub eval_size: usize,
    pub eval_split_ratio: f64,

    // --- Agent loop (shared fields consumed here) ---
    pub max_agent_turns: usize,
    pub system_prompt: String,
}

impl Default for AgenticOPDConfig {
    /// Field-level defaults, matching `Field(default=...)` in the Python config.
    fn default() -> Self {
        Self {
            opd_enabled: true,
            distill_topk: 50,
            prm_votes: 3,
            hint_max_next_state_chars: 4000,
            correctness_weight: 0.7,
            efficiency_weight: 0.15,
            tool_usage_weight: 0.15,
            dataset_name: None,
            eval_size: 10,
            eval_split_ratio: 0.15,
            max_agent_turns: 15,
            system_prompt: String::new(),
        }
    }
}

/// The default system prompt used by `config_init`.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a skilled Python programmer. When given a coding task:\n\
1. Write the solution to a file called 'solution.py'\n\
2. Write the test code to a file called 'test_solution.py'\n\
3. Run the tests with: python test_solution.py\n\
4. If tests fail, read the error output carefully, fix your code, and re-run\n\
5. Once all tests pass, report success\n\n\
Be efficient — write clean code and fix errors methodically.";

impl AgenticOPDConfig {
    /// Mirrors `AgenticOPDEnv.config_init`'s environment defaults (the subset
    /// represented in this struct).
    pub fn config_init() -> Self {
        Self {
            opd_enabled: true,
            distill_topk: 50,
            prm_votes: 3,
            max_agent_turns: 15,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            ..Self::default()
        }
    }
}

/// Default toolsets for this environment, matching `default_toolsets`.
pub const DEFAULT_TOOLSETS: &[&str] = &["terminal", "file"];

/// The environment name, matching `name = "agentic-opd"`.
pub const ENV_NAME: &str = "agentic-opd";

// ════════════════════════════════════════════════════════════════════════
// Prompt formatting
// ════════════════════════════════════════════════════════════════════════

/// Format the coding task as a user prompt.
///
/// Mirrors `AgenticOPDEnv.format_prompt`.
pub fn format_prompt(item: &CodingTask) -> String {
    let mut prompt = format!("Solve the following coding task.\n\n## Task\n{}\n\n", item.task);
    if !item.test_code.is_empty() {
        prompt.push_str(&format!(
            "## Tests\nThe following test code will be used to verify your solution:\n\
```python\n{}```\n\n",
            item.test_code
        ));
    }
    prompt.push_str(
        "## Instructions\n\
1. Write your solution to `solution.py`\n\
2. Write the test code to `test_solution.py`\n\
3. Run `python test_solution.py` to verify\n\
4. Fix any failures and re-run until all tests pass\n",
    );
    prompt
}

// ════════════════════════════════════════════════════════════════════════
// Reward computation
// ════════════════════════════════════════════════════════════════════════

/// The result of running the agent loop (minimal port of `AgentResult`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentResult {
    pub turns_used: usize,
    /// Conversation messages produced by the agent loop.
    pub messages: Vec<ChatMessage>,
}

/// Result of executing a test command in the sandbox.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TestExecResult {
    pub output: String,
    pub exit_code: i64,
}

/// Breakdown of a computed reward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardBreakdown {
    pub correctness: f64,
    pub efficiency: f64,
    pub tool_usage: f64,
    pub reward: f64,
}

/// Score the test correctness signal from a (possibly absent) test execution.
///
/// Mirrors the `correctness` branch of `compute_reward`. `None` represents the
/// case where running the test raised an exception (correctness = 0.0).
pub fn score_correctness(test_result: Option<&TestExecResult>) -> f64 {
    let Some(tr) = test_result else {
        return 0.0;
    };
    let output_lower = tr.output.to_lowercase();
    if tr.exit_code == 0 && output_lower.contains("passed") {
        1.0
    } else if tr.exit_code == 0 {
        0.8
    } else if output_lower.contains("assert") && output_lower.contains("error") {
        0.2
    } else {
        0.1
    }
}

/// Score the efficiency signal from the number of turns used.
///
/// Mirrors the `efficiency` branch of `compute_reward`.
pub fn score_efficiency(turns_used: usize, max_turns: usize) -> f64 {
    if turns_used <= 3 {
        1.0
    } else if turns_used <= max_turns / 2 {
        0.8
    } else if turns_used <= max_turns * 3 / 4 {
        0.5
    } else {
        0.2
    }
}

/// Collect the set of tool names invoked by assistant messages.
///
/// Mirrors the `tools_used` accumulation in `compute_reward`.
pub fn tools_used(messages: &[ChatMessage]) -> BTreeSet<String> {
    let mut used = BTreeSet::new();
    for msg in messages {
        if msg.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }
        let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) else {
            continue;
        };
        // Python truthiness: empty list is falsy → skipped.
        if tool_calls.is_empty() {
            continue;
        }
        for tc in tool_calls {
            let name = tc
                .as_object()
                .and_then(|o| o.get("function"))
                .and_then(|f| f.as_object())
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            if !name.is_empty() {
                used.insert(name.to_string());
            }
        }
    }
    used
}

/// Score the tool-usage signal.
///
/// Mirrors the `tool_usage` branch of `compute_reward`.
pub fn score_tool_usage(tools: &BTreeSet<String>) -> f64 {
    let has_terminal = tools.contains("terminal");
    let has_file = tools.contains("write_file") || tools.contains("patch");
    if has_terminal && has_file {
        1.0
    } else if has_terminal {
        0.6
    } else if !tools.is_empty() {
        0.3
    } else {
        0.0
    }
}

/// Combine the three reward signals.
///
/// Mirrors the full `compute_reward` (sans buffer side-effects). The caller
/// supplies the test execution result (or `None` if running it raised).
pub fn compute_reward(
    cfg: &AgenticOPDConfig,
    result: &AgentResult,
    test_result: Option<&TestExecResult>,
) -> RewardBreakdown {
    let correctness = score_correctness(test_result);
    let efficiency = score_efficiency(result.turns_used, cfg.max_agent_turns);
    let tools = tools_used(&result.messages);
    let tool_usage = score_tool_usage(&tools);

    let mut reward = cfg.correctness_weight * correctness
        + cfg.efficiency_weight * efficiency
        + cfg.tool_usage_weight * tool_usage;
    reward = reward.min(1.0).max(0.0);

    RewardBreakdown {
        correctness,
        efficiency,
        tool_usage,
        reward,
    }
}

// ════════════════════════════════════════════════════════════════════════
// Turn-pair extraction
// ════════════════════════════════════════════════════════════════════════

/// An (assistant, next_state) turn pair extracted from a conversation.
///
/// Mirrors the dicts produced by `_extract_turn_pairs`.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnPair {
    /// Messages up to (not including) the assistant turn.
    pub context_messages: Vec<ChatMessage>,
    pub assistant_text: String,
    pub next_state_text: String,
    pub next_state_role: String,
}

/// Walk conversation messages to find (assistant, next_state) pairs.
///
/// Mirrors `_extract_turn_pairs`. `max_next_state_chars` corresponds to
/// `config.hint_max_next_state_chars`.
pub fn extract_turn_pairs(messages: &[ChatMessage], max_next_state_chars: usize) -> Vec<TurnPair> {
    let mut pairs = Vec::new();
    let n = messages.len();
    let mut i = 0usize;
    while i < n {
        let msg = &messages[i];
        let is_assistant = msg.get("role").and_then(|r| r.as_str()) == Some("assistant");
        let content = msg.get("content");
        // Python: `msg.get("content")` truthiness — non-empty string only.
        let assistant_text = content.and_then(|c| c.as_str()).unwrap_or("");
        if is_assistant && !assistant_text.is_empty() {
            let context: Vec<ChatMessage> = messages[..i].to_vec();

            // Look ahead for next state.
            let mut j = i + 1;
            let mut next_states: Vec<&ChatMessage> = Vec::new();
            while j < n {
                let next_msg = &messages[j];
                match next_msg.get("role").and_then(|r| r.as_str()) {
                    Some("tool") => {
                        next_states.push(next_msg);
                        j += 1;
                    }
                    Some("user") => {
                        next_states.push(next_msg);
                        break;
                    }
                    _ => break,
                }
            }

            if !next_states.is_empty() {
                let next_role = next_states[0]
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("tool")
                    .to_string();
                let mut next_text_parts: Vec<String> = Vec::new();
                for ns in &next_states {
                    // Python `ns.get("content", "")` then truthiness check.
                    let raw = ns.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    if raw.is_empty() {
                        continue;
                    }
                    let truncated = truncate_chars(raw, max_next_state_chars);
                    next_text_parts.push(truncated);
                }
                let next_text = next_text_parts.join("\n---\n");
                if !next_text.trim().is_empty() {
                    pairs.push(TurnPair {
                        context_messages: context,
                        assistant_text: assistant_text.to_string(),
                        next_state_text: next_text,
                        next_state_role: next_role,
                    });
                }
            }
        }
        i += 1;
    }
    pairs
}

/// Truncate text to `max_chars` characters (char-based, like Python slicing),
/// appending the truncation marker if it was longer.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() > max_chars {
        let head: String = s.chars().take(max_chars).collect();
        format!("{head}\n...[truncated]")
    } else {
        s.to_string()
    }
}

// ════════════════════════════════════════════════════════════════════════
// Token-span search
// ════════════════════════════════════════════════════════════════════════

/// Find where `sub_tokens` appears in `full_tokens`, searching from the end.
///
/// Mirrors `_find_token_span`. Returns the start index, or `None` if not found
/// (including the empty-input and oversize-sub cases).
pub fn find_token_span(full_tokens: &[i64], sub_tokens: &[i64]) -> Option<usize> {
    if sub_tokens.is_empty() || full_tokens.is_empty() {
        return None;
    }
    let sub_len = sub_tokens.len();
    let full_len = full_tokens.len();
    if sub_len > full_len {
        return None;
    }
    // Search backwards: i from (full_len - sub_len) down to 0 inclusive.
    let mut i = full_len - sub_len;
    loop {
        if &full_tokens[i..i + sub_len] == sub_tokens {
            return Some(i);
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    None
}

// ════════════════════════════════════════════════════════════════════════
// Distill array assembly
// ════════════════════════════════════════════════════════════════════════

/// A single position's top-K teacher prediction.
pub type TopK<T> = Vec<T>;

/// Build the zero-initialised distill arrays for a sequence.
///
/// Mirrors the initialisation in `_opd_for_sequence`: each of `seq_len`
/// positions holds a `k`-length vector of zeros.
pub fn init_distill_arrays(seq_len: usize, k: usize) -> (Vec<TopK<i64>>, Vec<TopK<f64>>) {
    let ids = vec![vec![0i64; k]; seq_len];
    let lps = vec![vec![0.0f64; k]; seq_len];
    (ids, lps)
}

/// Align teacher top-K results to exactly `response_len` positions, padding from
/// the left with zero vectors when the teacher returned fewer positions.
///
/// Mirrors the right-slice / left-pad block in `_opd_for_sequence`.
pub fn align_response_topk(
    teacher_topk_ids: &[TopK<i64>],
    teacher_topk_lps: &[TopK<f64>],
    response_len: usize,
    k: usize,
) -> (Vec<TopK<i64>>, Vec<TopK<f64>>) {
    if teacher_topk_ids.len() >= response_len {
        let start = teacher_topk_ids.len() - response_len;
        (
            teacher_topk_ids[start..].to_vec(),
            teacher_topk_lps[start..].to_vec(),
        )
    } else {
        let pad_len = response_len - teacher_topk_ids.len();
        let mut ids: Vec<TopK<i64>> = vec![vec![0i64; k]; pad_len];
        ids.extend_from_slice(teacher_topk_ids);
        let mut lps: Vec<TopK<f64>> = vec![vec![0.0f64; k]; pad_len];
        lps.extend_from_slice(teacher_topk_lps);
        (ids, lps)
    }
}

/// Splice aligned response top-K predictions into the full-sequence distill
/// arrays at `turn_start`, padding/truncating each position to exactly `k`.
///
/// Mirrors the inner mapping loop of `_opd_for_sequence`. Returns `true` if the
/// turn was scored (always true when this is reached, since Python increments
/// `turns_scored` once `turn_start` is found).
pub fn splice_turn_topk(
    distill_token_ids: &mut [TopK<i64>],
    distill_logprobs: &mut [TopK<f64>],
    resp_topk_ids: &[TopK<i64>],
    resp_topk_lps: &[TopK<f64>],
    turn_start: usize,
    response_len: usize,
    k: usize,
) {
    let seq_len = distill_token_ids.len();
    if turn_start >= seq_len {
        return;
    }
    let limit = response_len.min(seq_len - turn_start);
    for j in 0..limit {
        let pos = turn_start + j;
        if pos < seq_len && j < resp_topk_ids.len() {
            let mut ids: Vec<i64> = resp_topk_ids[j].iter().take(k).copied().collect();
            let mut lps: Vec<f64> = resp_topk_lps[j].iter().take(k).copied().collect();
            while ids.len() < k {
                ids.push(0);
                lps.push(0.0);
            }
            distill_token_ids[pos] = ids;
            distill_logprobs[pos] = lps;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Eval / wandb metric helpers
// ════════════════════════════════════════════════════════════════════════

/// A single evaluation sample, mirroring the dicts collected in `evaluate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalSample {
    pub prompt: String,
    pub response: String,
    pub correctness: f64,
    pub reward: f64,
    pub turns: usize,
}

/// Compute the eval metrics block from a list of samples.
///
/// Mirrors the `eval_metrics` dict in `evaluate`.
pub fn eval_metrics(samples: &[EvalSample]) -> Map<String, Value> {
    let n = samples.len();
    let mut m = Map::new();
    let (mean_correctness, mean_reward, pass_rate) = if n == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let sum_c: f64 = samples.iter().map(|s| s.correctness).sum();
        let sum_r: f64 = samples.iter().map(|s| s.reward).sum();
        let passes = samples.iter().filter(|s| s.correctness >= 0.8).count();
        (
            sum_c / n as f64,
            sum_r / n as f64,
            passes as f64 / n as f64,
        )
    };
    m.insert("eval/mean_correctness".to_string(), json!(mean_correctness));
    m.insert("eval/mean_reward".to_string(), json!(mean_reward));
    m.insert("eval/pass_rate".to_string(), json!(pass_rate));
    m.insert("eval/n_items".to_string(), json!(n));
    m
}

/// Truncate a string to at most `n` characters (char-based, like Python `[:n]`).
pub fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The metric buffers used to aggregate training-time signals.
///
/// Mirrors the `_*_buffer` lists on the environment instance.
#[derive(Debug, Clone, Default)]
pub struct MetricBuffers {
    pub reward: Vec<f64>,
    pub correctness: Vec<f64>,
    pub efficiency: Vec<f64>,
    pub tool_usage: Vec<f64>,
    pub hints_extracted: Vec<i64>,
    pub opd_turns_scored: Vec<i64>,
}

impl MetricBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a reward breakdown into the training buffers.
    pub fn record_reward(&mut self, b: &RewardBreakdown) {
        self.reward.push(b.reward);
        self.correctness.push(b.correctness);
        self.efficiency.push(b.efficiency);
        self.tool_usage.push(b.tool_usage);
    }

    /// Build the wandb metrics dict, clearing the buffers afterwards.
    ///
    /// Mirrors `wandb_log` (without the `super().wandb_log` chaining).
    pub fn drain_wandb_metrics(&mut self) -> Map<String, Value> {
        let mut m = Map::new();

        if !self.reward.is_empty() {
            let n = self.reward.len() as f64;
            m.insert(
                "train/mean_reward".to_string(),
                json!(self.reward.iter().sum::<f64>() / n),
            );
            m.insert(
                "train/mean_correctness".to_string(),
                json!(self.correctness.iter().sum::<f64>() / n),
            );
            m.insert(
                "train/mean_efficiency".to_string(),
                json!(self.efficiency.iter().sum::<f64>() / n),
            );
            m.insert(
                "train/mean_tool_usage".to_string(),
                json!(self.tool_usage.iter().sum::<f64>() / n),
            );
            let passes = self.correctness.iter().filter(|&&c| c >= 0.8).count();
            m.insert("train/pass_rate".to_string(), json!(passes as f64 / n));
            m.insert("train/total_rollouts".to_string(), json!(self.reward.len()));

            self.reward.clear();
            self.correctness.clear();
            self.efficiency.clear();
            self.tool_usage.clear();
        }

        if !self.hints_extracted.is_empty() {
            let n = self.hints_extracted.len() as f64;
            m.insert(
                "opd/mean_hints_per_rollout".to_string(),
                json!(self.hints_extracted.iter().sum::<i64>() as f64 / n),
            );
            m.insert(
                "opd/mean_turns_scored".to_string(),
                json!(self.opd_turns_scored.iter().sum::<i64>() as f64 / n),
            );
            let with_hints = self.hints_extracted.iter().filter(|&&h| h > 0).count();
            m.insert("opd/hint_rate".to_string(), json!(with_hints as f64 / n));
            m.insert(
                "opd/total_hints".to_string(),
                json!(self.hints_extracted.iter().sum::<i64>()),
            );
            m.insert(
                "opd/total_scored_turns".to_string(),
                json!(self.opd_turns_scored.iter().sum::<i64>()),
            );

            self.hints_extracted.clear();
            self.opd_turns_scored.clear();
        }

        m
    }
}

// ════════════════════════════════════════════════════════════════════════
// Dataset split helper
// ════════════════════════════════════════════════════════════════════════

/// Split the built-in tasks into (train, eval) using the 85/15 rule from
/// `setup`'s fallback branch. `items` should already be shuffled by the caller.
///
/// Mirrors `split = max(1, len(items) * 85 // 100)`.
pub fn split_builtin_tasks(items: Vec<CodingTask>) -> (Vec<CodingTask>, Vec<CodingTask>) {
    let split = std::cmp::max(1, items.len() * 85 / 100);
    let split = split.min(items.len());
    let train = items[..split].to_vec();
    let eval = items[split..].to_vec();
    (train, eval)
}

/// Compute the eval-holdout size for a HuggingFace-loaded dataset.
///
/// Mirrors `eval_size = max(config.eval_size, int(len * eval_split_ratio))`.
pub fn dataset_eval_size(n_items: usize, cfg: &AgenticOPDConfig) -> usize {
    let ratio_size = (n_items as f64 * cfg.eval_split_ratio) as usize;
    std::cmp::max(cfg.eval_size, ratio_size)
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        let mut m = Map::new();
        m.insert("role".to_string(), json!(role));
        m.insert("content".to_string(), json!(content));
        m
    }

    #[test]
    fn builtin_tasks_count_and_fields() {
        let tasks = builtin_coding_tasks();
        assert_eq!(tasks.len(), 8);
        assert_eq!(tasks[0].difficulty, "easy");
        assert!(tasks[0].task.contains("fizzbuzz"));
        assert!(tasks[3].difficulty == "medium");
    }

    #[test]
    fn parse_hint_positive() {
        let text = "I think yes. \\boxed{1} [HINT_START]Use a dict for O(n).[HINT_END]";
        let (score, hint) = parse_hint_result(text);
        assert_eq!(score, Some(1));
        assert_eq!(hint, "Use a dict for O(n).");
    }

    #[test]
    fn parse_hint_negative_no_hint() {
        let text = "No useful info. \\boxed{-1}";
        let (score, hint) = parse_hint_result(text);
        assert_eq!(score, Some(-1));
        assert_eq!(hint, "");
    }

    #[test]
    fn parse_hint_invalid_score_becomes_none() {
        let (score, _) = parse_hint_result("\\boxed{2}");
        assert_eq!(score, None);
        let (score2, _) = parse_hint_result("no box here");
        assert_eq!(score2, None);
    }

    #[test]
    fn parse_hint_uses_last_box_and_hint() {
        let text = "\\boxed{1} [HINT_START]first[HINT_END] then \\boxed{-1} \
            [HINT_START]second hint longer[HINT_END]";
        let (score, hint) = parse_hint_result(text);
        assert_eq!(score, Some(-1));
        assert_eq!(hint, "second hint longer");
    }

    #[test]
    fn parse_hint_multiline_dotall() {
        let text = "\\boxed{1}\n[HINT_START]line one\nline two[HINT_END]";
        let (_, hint) = parse_hint_result(text);
        assert_eq!(hint, "line one\nline two");
    }

    #[test]
    fn select_best_hint_longest() {
        let votes = vec![
            HintVote { score: Some(1), hint: "short hint xx".to_string() },
            HintVote { score: Some(-1), hint: "ignored because score -1".to_string() },
            HintVote { score: Some(1), hint: "a much longer and better hint here".to_string() },
            HintVote { score: Some(1), hint: "tiny".to_string() }, // <= 10 chars trimmed -> excluded
        ];
        let best = select_best_hint(&votes).unwrap();
        assert_eq!(best.hint, "a much longer and better hint here");
    }

    #[test]
    fn select_best_hint_none() {
        let votes = vec![
            HintVote { score: None, hint: "x".to_string() },
            HintVote { score: Some(-1), hint: "this would qualify by length".to_string() },
            HintVote { score: Some(1), hint: "too short".to_string() },
        ];
        assert!(select_best_hint(&votes).is_none());
    }

    #[test]
    fn select_best_hint_keeps_first_on_tie() {
        let votes = vec![
            HintVote { score: Some(1), hint: "AAAAAAAAAAAA".to_string() },
            HintVote { score: Some(1), hint: "BBBBBBBBBBBB".to_string() },
        ];
        let best = select_best_hint(&votes).unwrap();
        assert_eq!(best.hint, "AAAAAAAAAAAA");
    }

    #[test]
    fn append_hint_to_empty() {
        let out = append_hint_to_messages(&[], "do better");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], json!("user"));
        assert_eq!(out[0]["content"], json!("[user's hint / instruction]\ndo better"));
    }

    #[test]
    fn append_hint_to_last_user() {
        let messages = vec![
            msg("system", "sys"),
            msg("user", "original question"),
            msg("assistant", "answer"),
        ];
        let out = append_hint_to_messages(&messages, "  consider edge cases  ");
        // last user is index 1
        assert_eq!(
            out[1]["content"],
            json!("original question\n\n[user's hint / instruction]\nconsider edge cases")
        );
        // assistant untouched
        assert_eq!(out[2]["content"], json!("answer"));
    }

    #[test]
    fn append_hint_no_user_uses_last() {
        let messages = vec![msg("system", "sys"), msg("assistant", "a")];
        let out = append_hint_to_messages(&messages, "hint text");
        assert_eq!(
            out[1]["content"],
            json!("a\n\n[user's hint / instruction]\nhint text")
        );
    }

    #[test]
    fn append_hint_list_content() {
        let mut m = Map::new();
        m.insert("role".to_string(), json!("user"));
        m.insert(
            "content".to_string(),
            json!([{"text": "part1"}, {"text": "part2"}]),
        );
        let out = append_hint_to_messages(&[m], "extra");
        assert_eq!(
            out[0]["content"],
            json!("part1 part2\n\n[user's hint / instruction]\nextra")
        );
    }

    #[test]
    fn build_judge_messages_shape() {
        let msgs = build_hint_judge_messages("resp", "state", "tool");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], json!("system"));
        assert_eq!(msgs[1]["role"], json!("user"));
        let user = msgs[1]["content"].as_str().unwrap();
        assert!(user.contains("## Assistant response (turn t)\nresp"));
        assert!(user.contains("[role: tool]\nstate"));
    }

    #[test]
    fn format_prompt_with_tests() {
        let item = CodingTask::new("do thing", "assert True\n", "easy");
        let p = format_prompt(&item);
        assert!(p.contains("## Task\ndo thing"));
        assert!(p.contains("```python\nassert True\n```"));
        assert!(p.contains("## Instructions"));
    }

    #[test]
    fn format_prompt_no_tests() {
        let item = CodingTask::new("do thing", "", "easy");
        let p = format_prompt(&item);
        assert!(!p.contains("## Tests"));
        assert!(p.contains("## Instructions"));
    }

    #[test]
    fn correctness_branches() {
        assert_eq!(
            score_correctness(Some(&TestExecResult { output: "All tests Passed!".into(), exit_code: 0 })),
            1.0
        );
        assert_eq!(
            score_correctness(Some(&TestExecResult { output: "ran ok".into(), exit_code: 0 })),
            0.8
        );
        assert_eq!(
            score_correctness(Some(&TestExecResult { output: "AssertionError: ...".into(), exit_code: 1 })),
            0.2
        );
        assert_eq!(
            score_correctness(Some(&TestExecResult { output: "SyntaxError".into(), exit_code: 1 })),
            0.1
        );
        assert_eq!(score_correctness(None), 0.0);
    }

    #[test]
    fn efficiency_branches() {
        // max_turns = 15 → //2 = 7, *3//4 = 11
        assert_eq!(score_efficiency(2, 15), 1.0);
        assert_eq!(score_efficiency(3, 15), 1.0);
        assert_eq!(score_efficiency(7, 15), 0.8);
        assert_eq!(score_efficiency(11, 15), 0.5);
        assert_eq!(score_efficiency(15, 15), 0.2);
    }

    #[test]
    fn tool_usage_branches() {
        let both: BTreeSet<String> =
            ["terminal", "write_file"].iter().map(|s| s.to_string()).collect();
        assert_eq!(score_tool_usage(&both), 1.0);
        let term: BTreeSet<String> = ["terminal"].iter().map(|s| s.to_string()).collect();
        assert_eq!(score_tool_usage(&term), 0.6);
        let other: BTreeSet<String> = ["search"].iter().map(|s| s.to_string()).collect();
        assert_eq!(score_tool_usage(&other), 0.3);
        let none: BTreeSet<String> = BTreeSet::new();
        assert_eq!(score_tool_usage(&none), 0.0);
    }

    #[test]
    fn tools_used_extraction() {
        let mut assistant = Map::new();
        assistant.insert("role".to_string(), json!("assistant"));
        assistant.insert(
            "tool_calls".to_string(),
            json!([
                {"function": {"name": "terminal"}},
                {"function": {"name": "write_file"}},
                {"function": {"name": ""}},
            ]),
        );
        let msgs = vec![assistant];
        let used = tools_used(&msgs);
        assert!(used.contains("terminal"));
        assert!(used.contains("write_file"));
        assert_eq!(used.len(), 2);
    }

    #[test]
    fn compute_reward_full() {
        let cfg = AgenticOPDConfig::config_init();
        let mut assistant = Map::new();
        assistant.insert("role".to_string(), json!("assistant"));
        assistant.insert(
            "tool_calls".to_string(),
            json!([{"function": {"name": "terminal"}}, {"function": {"name": "write_file"}}]),
        );
        let result = AgentResult { turns_used: 2, messages: vec![assistant] };
        let tr = TestExecResult { output: "All tests passed!".into(), exit_code: 0 };
        let b = compute_reward(&cfg, &result, Some(&tr));
        // 0.7*1 + 0.15*1 + 0.15*1 = 1.0
        assert!((b.reward - 1.0).abs() < 1e-9);
        assert_eq!(b.correctness, 1.0);
        assert_eq!(b.efficiency, 1.0);
        assert_eq!(b.tool_usage, 1.0);
    }

    #[test]
    fn extract_turn_pairs_basic() {
        let mut tool = msg("tool", "Test failed: AssertionError");
        tool.insert("role".to_string(), json!("tool"));
        let messages = vec![
            msg("system", "sys"),
            msg("user", "task"),
            msg("assistant", "here is my code"),
            tool,
        ];
        let pairs = extract_turn_pairs(&messages, 4000);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].assistant_text, "here is my code");
        assert_eq!(pairs[0].next_state_text, "Test failed: AssertionError");
        assert_eq!(pairs[0].next_state_role, "tool");
        assert_eq!(pairs[0].context_messages.len(), 2);
    }

    #[test]
    fn extract_turn_pairs_truncation() {
        let long = "x".repeat(5000);
        let messages = vec![
            msg("assistant", "code"),
            msg("tool", &long),
        ];
        let pairs = extract_turn_pairs(&messages, 4000);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].next_state_text.ends_with("\n...[truncated]"));
        // 4000 chars + marker
        assert_eq!(pairs[0].next_state_text.chars().count(), 4000 + "\n...[truncated]".chars().count());
    }

    #[test]
    fn extract_turn_pairs_multiple_tools_then_user() {
        let messages = vec![
            msg("assistant", "first"),
            msg("tool", "t1"),
            msg("tool", "t2"),
            msg("user", "u1"),
            msg("assistant", "second"),
        ];
        let pairs = extract_turn_pairs(&messages, 4000);
        // first assistant -> tools+user; second assistant has no following state
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].next_state_text, "t1\n---\nt2\n---\nu1");
        assert_eq!(pairs[0].next_state_role, "tool");
    }

    #[test]
    fn extract_turn_pairs_empty_content_skipped() {
        let messages = vec![
            msg("assistant", ""),
            msg("tool", "result"),
        ];
        let pairs = extract_turn_pairs(&messages, 4000);
        assert!(pairs.is_empty());
    }

    #[test]
    fn find_span_backwards() {
        let full = vec![1, 2, 3, 2, 3, 4];
        // [2,3] appears at index 1 and 3; backwards search returns 3.
        assert_eq!(find_token_span(&full, &[2, 3]), Some(3));
    }

    #[test]
    fn find_span_not_found_and_edge() {
        assert_eq!(find_token_span(&[1, 2], &[]), None);
        assert_eq!(find_token_span(&[], &[1]), None);
        assert_eq!(find_token_span(&[1, 2], &[1, 2, 3]), None);
        assert_eq!(find_token_span(&[5, 6, 7], &[6]), Some(1));
        assert_eq!(find_token_span(&[5, 6, 7], &[5, 6, 7]), Some(0));
    }

    #[test]
    fn align_topk_right_slice() {
        let ids = vec![vec![1], vec![2], vec![3], vec![4]];
        let lps = vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]];
        let (rid, rlp) = align_response_topk(&ids, &lps, 2, 1);
        assert_eq!(rid, vec![vec![3], vec![4]]);
        assert_eq!(rlp, vec![vec![0.3], vec![0.4]]);
    }

    #[test]
    fn align_topk_left_pad() {
        let ids = vec![vec![7, 8]];
        let lps = vec![vec![0.7, 0.8]];
        let (rid, rlp) = align_response_topk(&ids, &lps, 3, 2);
        assert_eq!(rid, vec![vec![0, 0], vec![0, 0], vec![7, 8]]);
        assert_eq!(rlp, vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.7, 0.8]]);
    }

    #[test]
    fn splice_turn_into_sequence() {
        let (mut ids, mut lps) = init_distill_arrays(5, 2);
        let resp_ids = vec![vec![11, 12, 99], vec![21]]; // first truncated to k=2, second padded
        let resp_lps = vec![vec![1.1, 1.2, 9.9], vec![2.1]];
        splice_turn_topk(&mut ids, &mut lps, &resp_ids, &resp_lps, 2, 2, 2);
        assert_eq!(ids[0], vec![0, 0]);
        assert_eq!(ids[2], vec![11, 12]); // truncated to k
        assert_eq!(ids[3], vec![21, 0]); // padded to k
        assert_eq!(lps[2], vec![1.1, 1.2]);
        assert_eq!(lps[3], vec![2.1, 0.0]);
    }

    #[test]
    fn init_distill_shapes() {
        let (ids, lps) = init_distill_arrays(3, 4);
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0].len(), 4);
        assert_eq!(lps[2], vec![0.0; 4]);
    }

    #[test]
    fn eval_metrics_compute() {
        let samples = vec![
            EvalSample { prompt: "p".into(), response: "r".into(), correctness: 1.0, reward: 0.9, turns: 2 },
            EvalSample { prompt: "p".into(), response: "r".into(), correctness: 0.2, reward: 0.3, turns: 5 },
        ];
        let m = eval_metrics(&samples);
        assert_eq!(m["eval/n_items"], json!(2));
        assert!((m["eval/mean_correctness"].as_f64().unwrap() - 0.6).abs() < 1e-9);
        assert!((m["eval/pass_rate"].as_f64().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn eval_metrics_empty() {
        let m = eval_metrics(&[]);
        assert_eq!(m["eval/mean_correctness"], json!(0.0));
        assert_eq!(m["eval/n_items"], json!(0));
    }

    #[test]
    fn buffers_wandb_metrics() {
        let mut b = MetricBuffers::new();
        b.record_reward(&RewardBreakdown { correctness: 1.0, efficiency: 0.8, tool_usage: 1.0, reward: 0.9 });
        b.record_reward(&RewardBreakdown { correctness: 0.2, efficiency: 0.5, tool_usage: 0.6, reward: 0.3 });
        b.hints_extracted.push(2);
        b.hints_extracted.push(0);
        b.opd_turns_scored.push(1);
        b.opd_turns_scored.push(0);
        let m = b.drain_wandb_metrics();
        assert!((m["train/mean_reward"].as_f64().unwrap() - 0.6).abs() < 1e-9);
        assert_eq!(m["train/total_rollouts"], json!(2));
        assert!((m["train/pass_rate"].as_f64().unwrap() - 0.5).abs() < 1e-9);
        assert!((m["opd/mean_hints_per_rollout"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert!((m["opd/hint_rate"].as_f64().unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(m["opd/total_hints"], json!(2));
        // buffers cleared
        assert!(b.reward.is_empty());
        assert!(b.hints_extracted.is_empty());
    }

    #[test]
    fn split_builtin() {
        let tasks = builtin_coding_tasks();
        let (train, eval) = split_builtin_tasks(tasks);
        // 8 * 85 // 100 = 6
        assert_eq!(train.len(), 6);
        assert_eq!(eval.len(), 2);
    }

    #[test]
    fn dataset_eval_size_calc() {
        let cfg = AgenticOPDConfig::default();
        // 100 * 0.15 = 15 > eval_size 10
        assert_eq!(dataset_eval_size(100, &cfg), 15);
        // 20 * 0.15 = 3 < eval_size 10
        assert_eq!(dataset_eval_size(20, &cfg), 10);
    }

    #[test]
    fn config_defaults_match() {
        let c = AgenticOPDConfig::config_init();
        assert!(c.opd_enabled);
        assert_eq!(c.distill_topk, 50);
        assert_eq!(c.prm_votes, 3);
        assert_eq!(c.max_agent_turns, 15);
        assert!((c.correctness_weight - 0.7).abs() < 1e-9);
        assert!(c.system_prompt.contains("skilled Python programmer"));
    }

    #[test]
    fn take_chars_helper() {
        assert_eq!(take_chars("hello", 3), "hel");
        assert_eq!(take_chars("hi", 10), "hi");
    }
}
