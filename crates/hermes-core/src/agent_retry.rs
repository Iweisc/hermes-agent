//! Retry utilities — jittered backoff for decorrelated retries.
//!
//! Replaces fixed exponential backoff with jittered delays to prevent
//! thundering-herd retry spikes when multiple sessions hit the same
//! rate-limited provider concurrently.
//!
//! Port of `agent/retry_utils.py`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Monotonic counter for jitter seed uniqueness within the same process.
///
/// The Python original protects an `int` counter with a `threading.Lock`;
/// here an `AtomicU64` provides the same "each call gets a unique, strictly
/// increasing tick" guarantee without a separate mutex, and is safe under
/// concurrent retry paths (e.g. multiple gateway sessions retrying at once).
static JITTER_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Default base delay in seconds for attempt 1.
pub const DEFAULT_BASE_DELAY: f64 = 5.0;
/// Default maximum delay cap in seconds.
pub const DEFAULT_MAX_DELAY: f64 = 120.0;
/// Default jitter ratio.
pub const DEFAULT_JITTER_RATIO: f64 = 0.5;

/// Compute the deterministic (pre-jitter) backoff delay.
///
/// Returns `min(base * 2^(attempt-1), max_delay)`, matching the Python's
/// exponent/overflow handling: a non-positive `base_delay` or an exponent of
/// 63 or more collapses straight to `max_delay`.
pub fn backoff_delay(attempt: i64, base_delay: f64, max_delay: f64) -> f64 {
    // exponent = max(0, attempt - 1)
    let exponent = (attempt - 1).max(0);

    if exponent >= 63 || base_delay <= 0.0 {
        max_delay
    } else {
        // base_delay * (2 ** exponent), capped at max_delay.
        // 2 ** exponent for exponent < 63 fits comfortably in f64.
        let scaled = base_delay * (2.0_f64).powi(exponent as i32);
        scaled.min(max_delay)
    }
}

/// Current monotonic-ish seed source: nanoseconds since the Unix epoch.
///
/// Mirrors Python's `time.time_ns()`. Falls back to 0 if the clock is set
/// before the epoch (should never happen in practice).
fn time_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Compute the 32-bit jitter seed for a given clock reading and tick.
///
/// `seed = (time_ns ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF`
///
/// Extracted so tests can verify the seed derivation deterministically.
fn jitter_seed(time_ns: u128, tick: u64) -> u32 {
    // 0x9E3779B9 is the 32-bit golden-ratio constant. Python computes
    // `tick * 0x9E3779B9` in arbitrary precision then masks to 32 bits after
    // XOR; we reproduce that by doing the XOR in u128 and masking at the end.
    let mixed = time_ns ^ ((tick as u128).wrapping_mul(0x9E37_79B9));
    (mixed & 0xFFFF_FFFF) as u32
}

/// A tiny deterministic PRNG (SplitMix64) seeded from a 32-bit seed.
///
/// The Python original uses `random.Random(seed)` (a Mersenne Twister) to draw
/// a single `uniform(0, hi)`. We cannot reproduce CPython's MT19937 stream
/// bit-for-bit without porting the whole generator, and the orchestrating loop
/// only cares that the jitter is (a) deterministic for a given seed and (b)
/// uniformly distributed within `[0, hi]`. SplitMix64 satisfies both. This is
/// the one intentional deviation from the Python; see `behavior_notes`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u32) -> Self {
        // Spread the 32-bit seed across 64 bits so low-entropy seeds still
        // produce well-mixed first outputs.
        let s = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
        SplitMix64 { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits for a uniform double in [0, 1), the standard
        // construction also used by CPython's `random.random()`.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform float in [0, hi].
    fn uniform_0(&mut self, hi: f64) -> f64 {
        // Python's `random.uniform(a, b)` returns a + (b - a) * random();
        // with a == 0 that is hi * random(). The endpoint `hi` is reachable
        // only via rounding, exactly as in Python.
        hi * self.next_f64()
    }
}

/// Compute a jittered exponential backoff delay using default tuning.
///
/// `attempt` is the 1-based retry attempt number. Equivalent to calling
/// [`jittered_backoff_with`] with the `DEFAULT_*` constants.
pub fn jittered_backoff(attempt: i64) -> f64 {
    jittered_backoff_with(
        attempt,
        DEFAULT_BASE_DELAY,
        DEFAULT_MAX_DELAY,
        DEFAULT_JITTER_RATIO,
    )
}

/// Compute a jittered exponential backoff delay.
///
/// # Arguments
/// * `attempt` - 1-based retry attempt number.
/// * `base_delay` - Base delay in seconds for attempt 1.
/// * `max_delay` - Maximum delay cap in seconds.
/// * `jitter_ratio` - Fraction of the computed delay to use as the random
///   jitter range. `0.5` means jitter is uniform in `[0, 0.5 * delay]`.
///
/// # Returns
/// Delay in seconds: `min(base * 2^(attempt-1), max_delay) + jitter`.
///
/// The jitter decorrelates concurrent retries so multiple sessions hitting the
/// same provider don't all retry at the same instant.
pub fn jittered_backoff_with(
    attempt: i64,
    base_delay: f64,
    max_delay: f64,
    jitter_ratio: f64,
) -> f64 {
    // Atomically bump the counter and capture our unique tick. fetch_add
    // returns the previous value, so add 1 to match Python's "increment then
    // read" semantics (first tick == 1).
    let tick = JITTER_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;

    let delay = backoff_delay(attempt, base_delay, max_delay);

    let seed = jitter_seed(time_ns(), tick);
    let mut rng = SplitMix64::new(seed);
    let jitter = rng.uniform_0(jitter_ratio * delay);

    delay + jitter
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_exponentially_until_cap() {
        // attempt 1 -> base
        assert_eq!(backoff_delay(1, 5.0, 120.0), 5.0);
        // attempt 2 -> base * 2
        assert_eq!(backoff_delay(2, 5.0, 120.0), 10.0);
        // attempt 3 -> base * 4
        assert_eq!(backoff_delay(3, 5.0, 120.0), 20.0);
        // attempt 4 -> base * 8
        assert_eq!(backoff_delay(4, 5.0, 120.0), 40.0);
        // attempt 5 -> base * 16 = 80
        assert_eq!(backoff_delay(5, 5.0, 120.0), 80.0);
        // attempt 6 -> base * 32 = 160 -> capped at 120
        assert_eq!(backoff_delay(6, 5.0, 120.0), 120.0);
        // far beyond cap stays capped
        assert_eq!(backoff_delay(50, 5.0, 120.0), 120.0);
    }

    #[test]
    fn attempt_zero_and_negative_treated_as_attempt_one() {
        // exponent = max(0, attempt - 1)
        assert_eq!(backoff_delay(0, 5.0, 120.0), 5.0);
        assert_eq!(backoff_delay(-5, 5.0, 120.0), 5.0);
    }

    #[test]
    fn large_exponent_collapses_to_max() {
        // exponent >= 63 -> max_delay directly
        assert_eq!(backoff_delay(64, 5.0, 120.0), 120.0);
        assert_eq!(backoff_delay(1000, 5.0, 120.0), 120.0);
    }

    #[test]
    fn nonpositive_base_collapses_to_max() {
        assert_eq!(backoff_delay(1, 0.0, 120.0), 120.0);
        assert_eq!(backoff_delay(3, -1.0, 120.0), 120.0);
    }

    #[test]
    fn jitter_seed_matches_python_formula() {
        // seed = (time_ns ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
        let t: u128 = 0;
        let tick: u64 = 1;
        let expected = ((0u128 ^ (1u128 * 0x9E37_79B9)) & 0xFFFF_FFFF) as u32;
        assert_eq!(jitter_seed(t, tick), expected);
        assert_eq!(jitter_seed(t, tick), 0x9E37_79B9);

        // A non-zero clock XORs in.
        let t2: u128 = 0xFFFF_FFFF_FFFF;
        let s = jitter_seed(t2, 2);
        let expected2 = ((t2 ^ (2u128 * 0x9E37_79B9)) & 0xFFFF_FFFF) as u32;
        assert_eq!(s, expected2);
    }

    #[test]
    fn jitter_seed_is_masked_to_32_bits() {
        // No matter the inputs, the seed fits in u32 (the mask).
        let s = jitter_seed(u128::MAX, u64::MAX);
        assert!(u64::from(s) <= 0xFFFF_FFFF);
    }

    #[test]
    fn full_delay_within_expected_bounds() {
        // With defaults, attempt 1: delay = 5, jitter in [0, 0.5*5] = [0, 2.5].
        for _ in 0..1000 {
            let d = jittered_backoff(1);
            assert!(d >= 5.0, "delay {} below base", d);
            assert!(d <= 5.0 + 2.5 + 1e-9, "delay {} above max", d);
        }
    }

    #[test]
    fn jitter_ratio_zero_yields_exact_delay() {
        // jitter_ratio 0 -> jitter range is [0, 0], so result == delay exactly.
        for attempt in 1..=8 {
            let d = jittered_backoff_with(attempt, 5.0, 120.0, 0.0);
            let expected = backoff_delay(attempt, 5.0, 120.0);
            assert_eq!(d, expected);
        }
    }

    #[test]
    fn counter_is_monotonic_and_unique_across_calls() {
        // Each call bumps the shared counter; ticks are strictly increasing.
        let a = JITTER_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        let b = JITTER_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        assert!(b > a);
    }

    #[test]
    fn rng_is_deterministic_for_a_seed() {
        let mut r1 = SplitMix64::new(42);
        let mut r2 = SplitMix64::new(42);
        for _ in 0..16 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn rng_uniform_stays_in_range() {
        let mut r = SplitMix64::new(7);
        for _ in 0..10_000 {
            let v = r.uniform_0(10.0);
            assert!(v >= 0.0 && v <= 10.0);
        }
    }

    #[test]
    fn rng_uniform_roughly_centered() {
        // Sanity: the mean of uniform(0, hi) should be near hi/2.
        let mut r = SplitMix64::new(123_456);
        let n = 50_000;
        let hi = 4.0;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += r.uniform_0(hi);
        }
        let mean = sum / n as f64;
        assert!((mean - hi / 2.0).abs() < 0.1, "mean {} off-center", mean);
    }
}
