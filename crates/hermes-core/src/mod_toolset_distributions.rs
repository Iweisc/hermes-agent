//! Toolset Distributions Module
//!
//! This module defines distributions of toolsets for data generation runs.
//! Each distribution specifies which toolsets should be used and their probability
//! of being selected for any given prompt during batch processing.
//!
//! A distribution maps toolset names to their selection probability (%).
//! Probabilities should sum to 100, but the system tolerates other sums.
//!
//! Faithful port of the Python module `toolset_distributions.py`.
//!
//! Key behavioral notes preserved from the Python original:
//! - Both the distribution map and each distribution's `toolsets` map preserve
//!   *insertion order* (Python `dict`). This matters for `sample_*`'s tie-break
//!   on the highest-probability toolset (Python `max` returns the first maximum
//!   encountered when iterating insertion order).
//! - Sampling uses `random() * 100 < probability`, where `random()` is in `[0, 1)`.
//!   A probability of `100` therefore *always* selects (since `random()*100 < 100`),
//!   and a probability of `0` never selects.
//! - Toolset validity is checked via [`crate::mod_toolsets::validate_toolset`].

use std::collections::BTreeMap;

/// A single toolset distribution definition: a human-readable description plus
/// an ordered list of `(toolset_name, probability_percentage)` pairs.
///
/// The toolsets are stored as an ordered `Vec` (not a map) to faithfully
/// reproduce Python's insertion-ordered `dict` semantics, which the sampling
/// tie-break logic depends on.
#[derive(Debug, Clone, PartialEq)]
pub struct Distribution {
    pub description: &'static str,
    /// Ordered `(toolset_name, probability%)` pairs.
    pub toolsets: Vec<(&'static str, i32)>,
}

impl Distribution {
    fn new(description: &'static str, toolsets: &[(&'static str, i32)]) -> Self {
        Distribution {
            description,
            toolsets: toolsets.to_vec(),
        }
    }
}

/// Validation hook abstraction so callers can inject their own toolset validator
/// (e.g. for tests or alternative registries). The default uses
/// [`crate::mod_toolsets::validate_toolset`].
pub trait ToolsetValidator {
    fn validate_toolset(&self, name: &str) -> bool;
}

/// Default validator delegating to the ported toolsets module.
pub struct DefaultValidator;

impl ToolsetValidator for DefaultValidator {
    fn validate_toolset(&self, name: &str) -> bool {
        crate::mod_toolsets::validate_toolset(name)
    }
}

/// Source of randomness for sampling, abstracted so tests can supply a
/// deterministic sequence. Returns a value in `[0.0, 1.0)`, matching Python's
/// `random.random()`.
pub trait RandomSource {
    fn random(&mut self) -> f64;
}

/// Build the full ordered list of distributions, preserving the exact order and
/// values from the Python source.
fn build_distributions() -> Vec<(&'static str, Distribution)> {
    vec![
        (
            "default",
            Distribution::new(
                "All available tools, all the time",
                &[
                    ("web", 100),
                    ("vision", 100),
                    ("image_gen", 100),
                    ("terminal", 100),
                    ("file", 100),
                    ("moa", 100),
                    ("browser", 100),
                ],
            ),
        ),
        (
            "image_gen",
            Distribution::new(
                "Heavy focus on image generation with vision and web support",
                &[
                    ("image_gen", 90),
                    ("vision", 90),
                    ("web", 55),
                    ("terminal", 45),
                    ("moa", 10),
                ],
            ),
        ),
        (
            "research",
            Distribution::new(
                "Web research with vision analysis and reasoning",
                &[
                    ("web", 90),
                    ("browser", 70),
                    ("vision", 50),
                    ("moa", 40),
                    ("terminal", 10),
                ],
            ),
        ),
        (
            "science",
            Distribution::new(
                "Scientific research with web, terminal, file, and browser capabilities",
                &[
                    ("web", 94),
                    ("terminal", 94),
                    ("file", 94),
                    ("vision", 65),
                    ("browser", 50),
                    ("image_gen", 15),
                    ("moa", 10),
                ],
            ),
        ),
        (
            "development",
            Distribution::new(
                "Terminal, file tools, and reasoning with occasional web lookup",
                &[
                    ("terminal", 80),
                    ("file", 80),
                    ("moa", 60),
                    ("web", 30),
                    ("vision", 10),
                ],
            ),
        ),
        (
            "safe",
            Distribution::new(
                "All tools except terminal for safety",
                &[
                    ("web", 80),
                    ("browser", 70),
                    ("vision", 60),
                    ("image_gen", 60),
                    ("moa", 50),
                ],
            ),
        ),
        (
            "balanced",
            Distribution::new(
                "Equal probability of all toolsets",
                &[
                    ("web", 50),
                    ("vision", 50),
                    ("image_gen", 50),
                    ("terminal", 50),
                    ("file", 50),
                    ("moa", 50),
                    ("browser", 50),
                ],
            ),
        ),
        (
            "minimal",
            Distribution::new("Only web tools for basic research", &[("web", 100)]),
        ),
        (
            "terminal_only",
            Distribution::new(
                "Terminal and file tools for code execution tasks",
                &[("terminal", 100), ("file", 100)],
            ),
        ),
        (
            "terminal_web",
            Distribution::new(
                "Terminal and file tools with web search for documentation lookup",
                &[("terminal", 100), ("file", 100), ("web", 100)],
            ),
        ),
        (
            "creative",
            Distribution::new(
                "Image generation and vision analysis focus",
                &[("image_gen", 90), ("vision", 90), ("web", 30)],
            ),
        ),
        (
            "reasoning",
            Distribution::new(
                "Heavy mixture of agents usage with minimal other tools",
                &[("moa", 90), ("web", 30), ("terminal", 20)],
            ),
        ),
        (
            "browser_use",
            Distribution::new(
                "Full browser-based web interaction with search, vision, and page control",
                &[("browser", 100), ("web", 80), ("vision", 70)],
            ),
        ),
        (
            "browser_only",
            Distribution::new(
                "Only browser automation tools for pure web interaction tasks",
                &[("browser", 100)],
            ),
        ),
        (
            "browser_tasks",
            Distribution::new(
                "Browser-focused distribution (browser toolset includes web_search for finding URLs since Google blocks direct browser searches)",
                &[("browser", 97), ("vision", 12), ("terminal", 15)],
            ),
        ),
        (
            "terminal_tasks",
            Distribution::new(
                "Terminal-focused distribution with high terminal/file availability, occasional other tools",
                &[
                    ("terminal", 97),
                    ("file", 97),
                    ("web", 97),
                    ("browser", 75),
                    ("vision", 50),
                    ("image_gen", 10),
                ],
            ),
        ),
        (
            "mixed_tasks",
            Distribution::new(
                "Mixed distribution with high browser, terminal, and file availability for complex tasks",
                &[
                    ("browser", 92),
                    ("terminal", 92),
                    ("file", 92),
                    ("web", 35),
                    ("vision", 15),
                    ("image_gen", 15),
                ],
            ),
        ),
    ]
}

/// Return the ordered list of `(name, Distribution)` pairs, mirroring the
/// insertion order of the Python `DISTRIBUTIONS` dict.
pub fn distributions() -> Vec<(&'static str, Distribution)> {
    build_distributions()
}

/// Get a toolset distribution by name.
///
/// Returns `None` if the distribution is not found (mirrors Python's
/// `DISTRIBUTIONS.get(name)`).
pub fn get_distribution(name: &str) -> Option<Distribution> {
    build_distributions()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, d)| d)
}

/// List all available distributions as an ordered `Vec` of `(name, Distribution)`.
///
/// This is the order-preserving analogue of Python's `list_distributions()`
/// (which returns a `dict` copy). Use [`list_distributions_map`] if you need a
/// `BTreeMap` and don't care about insertion order.
pub fn list_distributions() -> Vec<(&'static str, Distribution)> {
    build_distributions()
}

/// List all available distributions as a `BTreeMap` keyed by name.
///
/// Note: a `BTreeMap` is *sorted* by key, unlike Python's insertion-ordered
/// dict. Prefer [`list_distributions`] when order matters.
pub fn list_distributions_map() -> BTreeMap<&'static str, Distribution> {
    build_distributions().into_iter().collect()
}

/// Check if a distribution name is valid (i.e. present in the registry).
pub fn validate_distribution(distribution_name: &str) -> bool {
    build_distributions()
        .iter()
        .any(|(n, _)| *n == distribution_name)
}

/// Sample toolsets from a distribution using the default validator and a
/// real random source backed by the `rand`-free `getrandom`-style approach.
///
/// This convenience wrapper uses a simple thread-local-free PRNG seeded from the
/// system clock; for deterministic behavior or production randomness, prefer
/// [`sample_toolsets_from_distribution_with`].
///
/// # Errors
/// Returns `Err` with a message if the distribution name is unknown (mirrors
/// Python's `ValueError`).
pub fn sample_toolsets_from_distribution(
    distribution_name: &str,
) -> Result<Vec<String>, String> {
    let mut rng = SystemRandom::new();
    let validator = DefaultValidator;
    sample_toolsets_from_distribution_with(distribution_name, &mut rng, &validator)
}

/// Core sampling routine, parameterised over the random source and validator so
/// it can be tested deterministically.
///
/// Each toolset in the distribution has a `%` chance of being included; multiple
/// toolsets may be active simultaneously. If no toolset is selected (possible
/// with low probabilities), the highest-probability *valid* toolset is selected
/// as a fallback (first-encountered on ties, matching Python `max`).
///
/// # Errors
/// Returns `Err` if the distribution name is unknown.
pub fn sample_toolsets_from_distribution_with<R, V>(
    distribution_name: &str,
    rng: &mut R,
    validator: &V,
) -> Result<Vec<String>, String>
where
    R: RandomSource,
    V: ToolsetValidator,
{
    let dist = get_distribution(distribution_name)
        .ok_or_else(|| format!("Unknown distribution: {distribution_name}"))?;

    let mut selected: Vec<String> = Vec::new();

    for &(toolset_name, probability) in &dist.toolsets {
        // Validate toolset exists.
        if !validator.validate_toolset(toolset_name) {
            // Mirrors the Python warning print; we route through `log` so callers
            // can capture or suppress it.
            log::warn!(
                "Toolset '{toolset_name}' in distribution '{distribution_name}' is not valid"
            );
            continue;
        }

        // Roll the dice - include if random*100 < probability.
        if rng.random() * 100.0 < probability as f64 {
            selected.push(toolset_name.to_string());
        }
    }

    // Fallback: ensure at least one toolset when the distribution is non-empty.
    if selected.is_empty() && !dist.toolsets.is_empty() {
        // Find the toolset with the highest probability; on ties keep the first
        // encountered (matching Python `max(..., key=...)`).
        let mut best: Option<(&str, i32)> = None;
        for &(name, prob) in &dist.toolsets {
            match best {
                Some((_, best_prob)) if best_prob >= prob => {}
                _ => best = Some((name, prob)),
            }
        }
        if let Some((name, _)) = best {
            if validator.validate_toolset(name) {
                selected.push(name.to_string());
            }
        }
    }

    Ok(selected)
}

/// Render detailed, human-readable information about a distribution as a
/// `String` (the Python original printed to stdout; returning a string is more
/// idiomatic and testable). Returns an error-prefixed string for unknown names.
pub fn print_distribution_info(distribution_name: &str) -> String {
    match get_distribution(distribution_name) {
        None => format!("\u{274c} Unknown distribution: {distribution_name}"),
        Some(dist) => {
            let mut out = String::new();
            out.push_str(&format!("\n\u{1f4ca} Distribution: {distribution_name}\n"));
            out.push_str(&format!("   Description: {}\n", dist.description));
            out.push_str("   Toolsets:\n");

            // Sort by probability descending; Python's `sorted(..., reverse=True)`
            // is stable, so equal probabilities retain insertion order.
            let mut ordered: Vec<(&str, i32)> = dist.toolsets.clone();
            ordered.sort_by(|a, b| b.1.cmp(&a.1));
            // `sort_by` in Rust is stable, matching Python's stable sort.

            for (toolset, prob) in ordered {
                // Python: f"     • {toolset:15} : {prob:3}% chance"
                out.push_str(&format!("     \u{2022} {toolset:<15} : {prob:>3}% chance\n"));
            }
            out
        }
    }
}

/// A small, dependency-free linear-congruential PRNG used as the default random
/// source. It is *not* cryptographically secure and is only intended to mirror
/// the statistical behavior of Python's `random.random()` for sampling.
pub struct SystemRandom {
    state: u64,
}

impl SystemRandom {
    pub fn new() -> Self {
        // Seed from the system clock, mixed to avoid trivially-correlated seeds.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        let seed = nanos
            ^ (std::process::id() as u64).wrapping_mul(0x2545_F491_4F6C_DD1D)
            ^ 0xD1B5_4A32_D192_ED03;
        SystemRandom {
            state: seed | 1, // ensure non-zero
        }
    }

    /// Create a PRNG with an explicit seed (useful for reproducible runs).
    pub fn from_seed(seed: u64) -> Self {
        SystemRandom { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Default for SystemRandom {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomSource for SystemRandom {
    fn random(&mut self) -> f64 {
        // Use the top 53 bits to form a double in [0, 1), like Python.
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic random source that replays a fixed sequence, repeating
    /// the last value once exhausted.
    struct ScriptedRandom {
        values: Vec<f64>,
        idx: usize,
    }

    impl ScriptedRandom {
        fn new(values: Vec<f64>) -> Self {
            ScriptedRandom { values, idx: 0 }
        }
    }

    impl RandomSource for ScriptedRandom {
        fn random(&mut self) -> f64 {
            let v = if self.idx < self.values.len() {
                self.values[self.idx]
            } else {
                *self.values.last().unwrap_or(&0.0)
            };
            self.idx += 1;
            v
        }
    }

    /// Validator accepting an explicit allow-list.
    struct AllowList(Vec<&'static str>);
    impl ToolsetValidator for AllowList {
        fn validate_toolset(&self, name: &str) -> bool {
            self.0.contains(&name)
        }
    }

    /// Validator accepting everything.
    struct AcceptAll;
    impl ToolsetValidator for AcceptAll {
        fn validate_toolset(&self, _name: &str) -> bool {
            true
        }
    }

    #[test]
    fn get_distribution_known_and_unknown() {
        let d = get_distribution("research").expect("research exists");
        assert_eq!(d.description, "Web research with vision analysis and reasoning");
        assert_eq!(d.toolsets.first(), Some(&("web", 90)));
        assert!(get_distribution("does_not_exist").is_none());
    }

    #[test]
    fn validate_distribution_works() {
        assert!(validate_distribution("default"));
        assert!(validate_distribution("mixed_tasks"));
        assert!(!validate_distribution("nope"));
    }

    #[test]
    fn list_distributions_preserves_order() {
        let list = list_distributions();
        assert_eq!(list.first().map(|(n, _)| *n), Some("default"));
        assert_eq!(list.get(1).map(|(n, _)| *n), Some("image_gen"));
        assert_eq!(list.last().map(|(n, _)| *n), Some("mixed_tasks"));
        assert_eq!(list.len(), 17);
    }

    #[test]
    fn sample_unknown_distribution_errors() {
        let mut rng = ScriptedRandom::new(vec![0.0]);
        let v = AcceptAll;
        let err = sample_toolsets_from_distribution_with("ghost", &mut rng, &v);
        assert_eq!(err, Err("Unknown distribution: ghost".to_string()));
    }

    #[test]
    fn probability_100_always_selects() {
        // random()*100 < 100 is always true for random in [0,1).
        // Use a value near the top of the range to be safe.
        let mut rng = ScriptedRandom::new(vec![0.999_999_999]);
        let v = AcceptAll;
        let out = sample_toolsets_from_distribution_with("minimal", &mut rng, &v).unwrap();
        assert_eq!(out, vec!["web".to_string()]);
    }

    #[test]
    fn probability_check_boundary() {
        // research first toolset is web@90. random()=0.9 -> 90.0 < 90 == false.
        // Subsequent: browser@70 with random=0.0 -> 0<70 true, etc.
        let v = AcceptAll;
        // Sequence per toolset: web, browser, vision, moa, terminal.
        let mut rng = ScriptedRandom::new(vec![0.9, 0.0, 0.0, 0.0, 0.0]);
        let out = sample_toolsets_from_distribution_with("research", &mut rng, &v).unwrap();
        // web excluded (0.9*100=90 not < 90), the rest included.
        assert_eq!(out, vec!["browser", "vision", "moa", "terminal"]);
    }

    #[test]
    fn fallback_picks_highest_probability_valid() {
        // All rolls high so nothing selected -> fallback to highest prob.
        // research: web@90 is the highest, first-encountered.
        let v = AcceptAll;
        let mut rng = ScriptedRandom::new(vec![0.9999]);
        let out = sample_toolsets_from_distribution_with("research", &mut rng, &v).unwrap();
        assert_eq!(out, vec!["web".to_string()]);
    }

    #[test]
    fn fallback_skips_invalid_highest() {
        // Highest prob toolset invalid -> fallback yields empty (Python behavior:
        // it only appends if the single highest-prob toolset validates).
        // research highest is web@90; disallow web.
        let v = AllowList(vec!["browser", "vision", "moa", "terminal"]);
        let mut rng = ScriptedRandom::new(vec![0.9999]);
        let out = sample_toolsets_from_distribution_with("research", &mut rng, &v).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn invalid_toolsets_are_skipped_during_sampling() {
        // Only allow "web"; with low roll all that validate get included.
        let v = AllowList(vec!["web"]);
        let mut rng = ScriptedRandom::new(vec![0.0]);
        let out = sample_toolsets_from_distribution_with("default", &mut rng, &v).unwrap();
        assert_eq!(out, vec!["web".to_string()]);
    }

    #[test]
    fn tie_break_uses_first_encountered() {
        // balanced: all 50. Fallback should pick the first one ("web").
        let v = AcceptAll;
        let mut rng = ScriptedRandom::new(vec![0.9999]);
        let out = sample_toolsets_from_distribution_with("balanced", &mut rng, &v).unwrap();
        assert_eq!(out, vec!["web".to_string()]);
    }

    #[test]
    fn print_info_unknown() {
        let s = print_distribution_info("missing");
        assert!(s.contains("Unknown distribution: missing"));
    }

    #[test]
    fn print_info_sorts_desc_stable() {
        let s = print_distribution_info("research");
        let web_pos = s.find("web ").unwrap();
        let browser_pos = s.find("browser").unwrap();
        let terminal_pos = s.find("terminal").unwrap();
        // web@90 before browser@70 before terminal@10
        assert!(web_pos < browser_pos);
        assert!(browser_pos < terminal_pos);
        assert!(s.contains(" 90% chance"));
        assert!(s.contains(" 10% chance"));
    }

    #[test]
    fn system_random_in_range() {
        let mut rng = SystemRandom::from_seed(42);
        for _ in 0..1000 {
            let v = rng.random();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
