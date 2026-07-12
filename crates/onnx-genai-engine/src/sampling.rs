//! Token sampling from processed logits.

/// Sample a token from logits using the specified method.
pub fn sample_greedy(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Sample with temperature (logits should already be temperature-scaled).
pub fn sample_categorical(logits: &[f32], rng_value: f32) -> u32 {
    // Softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();

    // Sample
    let mut cumulative = 0.0;
    for (i, &logit) in logits.iter().enumerate() {
        cumulative += (logit - max_logit).exp() / exp_sum;
        if rng_value <= cumulative {
            return i as u32;
        }
    }

    (logits.len() - 1) as u32
}
