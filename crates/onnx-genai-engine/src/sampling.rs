//! Token sampling from processed logits.

use crate::config::GenerateOptions;
use crate::logits::{ProcessorContext, TokenId};

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
pub fn sample_greedy(logits: &[f32]) -> u32 {
    let mut best: Option<(usize, f32)> = None;
    for (idx, &logit) in logits.iter().enumerate() {
        if logit.is_nan() {
            continue;
        }
        match best {
            None => best = Some((idx, logit)),
            Some((_, best_logit)) if logit > best_logit => best = Some((idx, logit)),
            _ => {}
        }
    }
    best.map(|(idx, _)| idx as u32).unwrap_or(0)
}

/// Sample categorically from logits using `rng_value` in [0, 1].
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
        if target <= cumulative {
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
}
