//! Token sampling from processed logits.

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
}
