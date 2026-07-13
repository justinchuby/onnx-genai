//! Token sampling from processed logits.

use crate::config::GenerateOptions;
use crate::logits::{ProcessorContext, TokenId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Per-request random state used only by categorical sampling.
pub(crate) struct SamplingRng {
    rng: StdRng,
}

impl SamplingRng {
    pub(crate) fn new(seed: Option<u64>) -> Self {
        let rng = seed.map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);
        Self { rng }
    }

    pub(crate) fn for_row(seed: Option<u64>, row_index: usize) -> Self {
        Self::new(seed.map(|seed| seed.wrapping_add(row_index as u64)))
    }

    pub(crate) fn value_for(&mut self, options: &GenerateOptions) -> f32 {
        if options.greedy || options.temperature == 0.0 {
            0.0
        } else {
            self.rng.random()
        }
    }
}

/// Final token selector used after logit processors have run.
///
/// The built-in generation path constructs one of the built-in samplers from
/// [`GenerateOptions`]. New decode paths can provide another implementation
/// without changing token commit logic.
pub trait Sampler: Send {
    fn sample(&mut self, logits: &[f32], context: &ProcessorContext) -> TokenId;
    fn name(&self) -> &str;
}

/// Argmax sampler. Ties keep the lowest token id.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32], _context: &ProcessorContext) -> TokenId {
        sample_greedy(logits)
    }

    fn name(&self) -> &str {
        "greedy"
    }
}

/// Categorical sampler using the deterministic RNG value supplied by the caller.
#[derive(Debug, Clone, Copy)]
pub struct CategoricalSampler {
    rng_value: f32,
}

impl CategoricalSampler {
    pub fn new(rng_value: f32) -> Self {
        Self { rng_value }
    }
}

impl Sampler for CategoricalSampler {
    fn sample(&mut self, logits: &[f32], _context: &ProcessorContext) -> TokenId {
        sample_categorical(logits, self.rng_value)
    }

    fn name(&self) -> &str {
        "categorical"
    }
}

pub(crate) enum DefaultSampler {
    Greedy(GreedySampler),
    Categorical(CategoricalSampler),
}

impl Sampler for DefaultSampler {
    fn sample(&mut self, logits: &[f32], context: &ProcessorContext) -> TokenId {
        match self {
            Self::Greedy(sampler) => sampler.sample(logits, context),
            Self::Categorical(sampler) => sampler.sample(logits, context),
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Greedy(sampler) => sampler.name(),
            Self::Categorical(sampler) => sampler.name(),
        }
    }
}

pub(crate) fn default_sampler_for_options(
    options: &GenerateOptions,
    rng_value: f32,
) -> DefaultSampler {
    if options.greedy || options.temperature == 0.0 {
        DefaultSampler::Greedy(GreedySampler)
    } else {
        DefaultSampler::Categorical(CategoricalSampler::new(rng_value))
    }
}

/// Sample a token from logits using argmax. Ties keep the lowest token id.
///
/// Implemented as two vectorizable passes — a horizontal max followed by a
/// first-match search — instead of a single branchy scan. Both passes are
/// free of per-element data-dependent branches, so the compiler autovectorizes
/// them; over a ~150k-entry vocabulary this is several times faster than the
/// scalar running-best loop, which measurably reduces per-token decode latency.
/// `f32::max` propagates the non-NaN operand, so NaNs are ignored exactly as in
/// the scalar version; an all-NaN (or empty) input yields token 0.
pub fn sample_greedy(logits: &[f32]) -> u32 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_logit == f32::NEG_INFINITY {
        // Empty input, all-NaN, or all -inf: token 0 (matches the first
        // element the scalar running-best scan would have selected).
        return 0;
    }
    logits
        .iter()
        .position(|&logit| logit == max_logit)
        .unwrap_or(0) as u32
}

/// Sample categorically from logits using `rng_value` in [0, 1).
pub fn sample_categorical(logits: &[f32], rng_value: f32) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    let max_logit = logits
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max_logit.is_finite() {
        return sample_greedy(logits);
    }

    let weights: Vec<f32> = logits
        .iter()
        .map(|&logit| {
            if logit.is_nan() {
                0.0
            } else {
                (logit - max_logit).exp()
            }
        })
        .collect();
    let exp_sum: f32 = weights.iter().sum();
    if !exp_sum.is_finite() || exp_sum <= 0.0 {
        return sample_greedy(logits);
    }

    let target = rng_value.clamp(0.0, 1.0);
    let mut cumulative = 0.0;
    for (i, weight) in weights.iter().enumerate() {
        cumulative += *weight / exp_sum;
        if target < cumulative {
            return i as u32;
        }
    }

    (logits.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_chooses_lowest_id_for_ties() {
        assert_eq!(sample_greedy(&[1.0, 3.0, 3.0]), 1);
    }

    #[test]
    fn categorical_respects_rng_value() {
        assert_eq!(sample_categorical(&[10.0, 0.0], 0.99), 0);
    }

    #[test]
    fn sampler_trait_dispatch_matches_free_functions() {
        let context = ProcessorContext::default();
        let mut greedy = GreedySampler;
        assert_eq!(greedy.sample(&[1.0, 2.0, 2.0], &context), 1);

        let mut categorical = CategoricalSampler::new(0.75);
        assert_eq!(categorical.sample(&[0.0, 0.0], &context), 1);
    }

    #[test]
    fn seeded_sampling_is_reproducible_and_seed_sensitive() {
        let options = GenerateOptions {
            greedy: false,
            seed: Some(42),
            ..Default::default()
        };
        let draw = |seed| {
            let mut rng = SamplingRng::new(Some(seed));
            (0..64)
                .map(|_| sample_categorical(&[0.0, 0.0, 0.0], rng.value_for(&options)))
                .collect::<Vec<_>>()
        };

        assert_eq!(draw(42), draw(42));
        assert_ne!(draw(42), draw(43));
    }

    #[test]
    fn categorical_sampling_matches_softmax_distribution() {
        let logits = [0.0_f32, 1.0, 2.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = logits.map(|logit| (logit - max_logit).exp());
        let total: f32 = weights.iter().sum();
        let expected = weights.map(|weight| weight / total);
        let options = GenerateOptions {
            greedy: false,
            ..Default::default()
        };
        let mut rng = SamplingRng::new(Some(7));
        let draws = 100_000;
        let mut counts = [0_usize; 3];
        for _ in 0..draws {
            counts[sample_categorical(&logits, rng.value_for(&options)) as usize] += 1;
        }

        for (count, probability) in counts.into_iter().zip(expected) {
            let observed = count as f32 / draws as f32;
            assert!(
                (observed - probability).abs() < 0.01,
                "observed {observed}, expected {probability}"
            );
        }
        assert!(counts.into_iter().all(|count| count > 0));
    }

    #[test]
    fn greedy_is_seed_independent_and_does_not_advance_rng() {
        let greedy = GenerateOptions {
            greedy: true,
            seed: Some(11),
            ..Default::default()
        };
        let sampled = GenerateOptions {
            greedy: false,
            seed: Some(11),
            ..Default::default()
        };
        let mut after_greedy = SamplingRng::new(greedy.seed);
        for _ in 0..32 {
            assert_eq!(sample_greedy(&[1.0, 3.0, 2.0]), 1);
            assert_eq!(after_greedy.value_for(&greedy), 0.0);
        }
        let mut untouched = SamplingRng::new(greedy.seed);
        assert_eq!(
            after_greedy.value_for(&sampled),
            untouched.value_for(&sampled)
        );
    }
}
