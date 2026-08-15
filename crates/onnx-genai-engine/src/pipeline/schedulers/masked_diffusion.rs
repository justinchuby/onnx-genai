//! Masked (discrete) language diffusion scheduler.
//!
//! LLaDA confidence-ranked and MDLM-style ancestral unmasking over a discrete
//! token sequence. Extracted verbatim from `pipeline.rs`.

use anyhow::Context;
use onnx_genai_ort::Value;
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::Scheduler;

/// Masked (discrete) language diffusion with two unmasking strategies, selected
/// by the scheduler config's `remasking` field.
///
/// The loop-carried tensor is an int64 token sequence `[B, S]` (prompt tokens
/// plus a masked generation region), the denoiser emits `[B, S, V]` logits, and
/// each step unmasks a growing subset of the still-masked positions until all
/// are filled by the final step.
///
/// **`remasking = "low_confidence"` (default)** — faithful to `ML-GSAI/LLaDA`'s
/// `generate` (`cfg_scale = 0`):
///   * the chosen token per position is the argmax of the (optionally
///     Gumbel-noised) logits (`add_gumbel_noise`; identity at `temperature = 0`);
///   * the confidence that ranks positions for remasking is the clean-softmax
///     probability of that chosen token (`remasking = "low_confidence"`);
///   * each step commits the highest-confidence still-masked positions, split
///     evenly across steps (`ceil(remaining / remaining_steps)`).
///
/// **`remasking = "random"`** — MDLM-style ancestral sampling (Sahoo et al.):
///   * each still-masked position unmasks *independently* with the schedule
///     probability `1/(steps_remaining_in_block)` (so the expected unmasked
///     fraction matches the log-linear absorbing schedule, and the final step
///     unmasks everything);
///   * on unmasking, the token is *sampled* from the model's categorical
///     distribution via the Gumbel-max trick (`temperature = 1.0` is a true
///     categorical sample; `0` is greedy argmax). The mask token id is never
///     emitted (SUBS parameterization). This per-position stochastic unmasking
///     avoids the degenerate repetition confidence-ranked greedy decoding
///     produces on non-LLaDA checkpoints such as MDLM.
///
/// With `block_length` set, the generation region is split into contiguous
/// left-to-right blocks; the total `num_steps` is divided evenly across the
/// `num_blocks`, and each step only unmasks tokens inside the current block
/// (semi-autoregressive remasking). A single block (the default) spans the
/// whole masked region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Remasking {
    /// LLaDA confidence-ranked commit (default).
    LowConfidence,
    /// MDLM-style per-position stochastic ancestral unmasking.
    Random,
}

#[derive(Debug)]
pub(super) struct MaskedDiffusion {
    pub(super) mask_token_id: i64,
    pub(super) temperature: f32,
    pub(super) block_length: Option<usize>,
    /// Unmasking strategy (see [`Remasking`]).
    pub(super) remasking: Remasking,
    /// Per-sequence generation-region start (prompt length), captured on the
    /// first step of a loop and cleared by [`Scheduler::reset`]. This lets the
    /// semi-autoregressive block boundaries be derived without threading the
    /// prompt length through the [`Scheduler`] trait.
    pub(super) generation_start: Mutex<Option<Vec<usize>>>,
}

impl MaskedDiffusion {
    fn lock_generation_start(&self) -> MutexGuard<'_, Option<Vec<usize>>> {
        self.generation_start
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Capture each sequence's generation-region start (prompt length) on the
    /// first use of a loop — the index of its first mask token. Cleared by
    /// [`Scheduler::reset`]. Called from both `step` and `cfg_uncond_sample`, so
    /// whichever runs first in a loop iteration records it from the seed.
    fn ensure_generation_start(&self, tokens: &[i64], batch: usize, sequence_length: usize) {
        let mut guard = self.lock_generation_start();
        if guard.is_some() {
            return;
        }
        let mut starts = Vec::with_capacity(batch);
        for row_index in 0..batch {
            let start = row_index * sequence_length;
            let first_mask = tokens[start..start + sequence_length]
                .iter()
                .position(|&token| token == self.mask_token_id)
                .unwrap_or(sequence_length);
            starts.push(first_mask);
        }
        *guard = Some(starts);
    }

    /// Argmax token id and its clean-softmax confidence for one logit row.
    ///
    /// `gumbel` supplies one uniform sample in `(0, 1)` per vocab entry when
    /// `temperature > 0`; it is ignored (and may be empty) at `temperature = 0`.
    fn predict_row(&self, row: &[f32], gumbel: &[f64]) -> (i64, f32) {
        // Clean-softmax denominator (numerically stable) for the confidence.
        let max_logit = row.iter().copied().fold(f32::MIN, f32::max);
        let sum_exp: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum();

        // Chosen token: argmax of the (optionally Gumbel-noised) logits.
        // LLaDA: logits.exp() / (-log u)^temperature, i.e. argmax of
        // `logit - temperature * ln(-ln u)`.
        let mut best_index = 0usize;
        let mut best_score = f32::MIN;
        for (j, &logit) in row.iter().enumerate() {
            let score = if self.temperature > 0.0 {
                let u = gumbel[j];
                logit - self.temperature * (-u.ln()).ln() as f32
            } else {
                logit
            };
            if score > best_score {
                best_score = score;
                best_index = j;
            }
        }
        let confidence = (row[best_index] - max_logit).exp() / sum_exp;
        (best_index as i64, confidence)
    }

    /// Sample one token from a logit row for MDLM-style ancestral unmasking.
    ///
    /// Uses the Gumbel-max trick so `temperature = 1.0` draws a true categorical
    /// sample from `softmax(logits)`, while `temperature = 0` is greedy argmax
    /// (matching [`predict_row`]'s token choice). The mask token id is excluded
    /// (SUBS parameterization: an unmasked position is never re-set to mask).
    fn sample_token(&self, row: &[f32], step: usize, position: usize, vocab: usize) -> i64 {
        let gumbel = if self.temperature > 0.0 {
            gumbel_uniforms(step, position, vocab)
        } else {
            Vec::new()
        };
        let mut best_index: Option<usize> = None;
        let mut best_score = f32::MIN;
        for (j, &logit) in row.iter().enumerate() {
            if j as i64 == self.mask_token_id {
                continue;
            }
            let score = if self.temperature > 0.0 {
                logit - self.temperature * (-gumbel[j].ln()).ln() as f32
            } else {
                logit
            };
            if best_index.is_none() || score > best_score {
                best_score = score;
                best_index = Some(j);
            }
        }
        best_index.unwrap_or(0) as i64
    }
}

impl Scheduler for MaskedDiffusion {
    fn reset(&self) {
        *self.lock_generation_start() = None;
    }

    /// LLaDA unconditional pass: re-mask the prompt tokens of the current
    /// sequence (`un_x[prompt] = mask_id`), leaving the generation region as-is.
    fn cfg_uncond_sample(&self, sample: &Value) -> anyhow::Result<Option<Value>> {
        let shape = sample.shape().to_vec();
        let tokens = sample.to_vec_i64()?;
        let count = tokens.len();
        let sequence_length = *shape.last().unwrap_or(&(count as i64)) as usize;
        if sequence_length == 0 {
            return Ok(None);
        }
        let batch = count.checked_div(sequence_length).unwrap_or(0).max(1);
        self.ensure_generation_start(&tokens, batch, sequence_length);
        let generation_start = self
            .lock_generation_start()
            .clone()
            .context("masked-diffusion generation start was not initialized")?;

        let mut output = tokens;
        for (row_index, &prompt_length) in generation_start.iter().enumerate() {
            let row_start = row_index * sequence_length;
            for offset in 0..prompt_length.min(sequence_length) {
                output[row_start + offset] = self.mask_token_id;
            }
        }
        Value::from_slice_i64(&output, &shape)
            .map(Some)
            .map_err(Into::into)
    }

    fn step(
        &self,
        step: usize,
        num_steps: usize,
        tokens: &Value,
        logits: &Value,
    ) -> anyhow::Result<Value> {
        let token_shape = tokens.shape().to_vec();
        let tokens = tokens.to_vec_i64()?;
        let sequence_count = tokens.len();
        let logit_shape = logits.shape();
        let vocab = *logit_shape
            .last()
            .context("masked_diffusion logits must be rank >= 1")? as usize;
        if vocab == 0 || sequence_count == 0 || logits.numel() != sequence_count * vocab {
            anyhow::bail!(
                "masked_diffusion shape mismatch: tokens {token_shape:?}, logits {logit_shape:?}"
            );
        }
        // Split the flat token buffer into per-sequence rows so top-k selection
        // and the transfer schedule are computed independently per sequence
        // (matching LLaDA's per-batch-row `topk`). Rank-1 inputs are one row.
        let sequence_length = *token_shape.last().unwrap_or(&(sequence_count as i64)) as usize;
        let batch = sequence_count
            .checked_div(sequence_length)
            .unwrap_or(0)
            .max(1);

        // Capture each sequence's generation-region start on the first step.
        self.ensure_generation_start(&tokens, batch, sequence_length);
        let generation_start = self
            .lock_generation_start()
            .clone()
            .context("masked-diffusion generation start was not initialized")?;

        let all_logits = logits.to_vec_f32()?;
        let mut output = tokens.clone();

        for (row_index, &prompt_length) in generation_start.iter().enumerate() {
            let row_start = row_index * sequence_length;
            let generation_length = sequence_length.saturating_sub(prompt_length);
            if generation_length == 0 {
                continue;
            }
            let block_length = self
                .block_length
                .unwrap_or(generation_length)
                .min(generation_length)
                .max(1);
            if !generation_length.is_multiple_of(block_length) {
                anyhow::bail!(
                    "masked_diffusion: generation length {generation_length} is not divisible \
                     by block_length {block_length}"
                );
            }
            let num_blocks = generation_length / block_length;
            if !num_steps.is_multiple_of(num_blocks) {
                anyhow::bail!(
                    "masked_diffusion: num_steps {num_steps} is not divisible by num_blocks \
                     {num_blocks} (generation_length {generation_length} / block_length \
                     {block_length})"
                );
            }
            let steps_per_block = num_steps / num_blocks;
            let block_index = (step / steps_per_block).min(num_blocks - 1);
            let step_in_block = step % steps_per_block;
            let block_start = prompt_length + block_index * block_length;
            let block_end = (block_start + block_length).min(sequence_length);
            let remaining_steps_in_block = steps_per_block - step_in_block;

            match self.remasking {
                Remasking::LowConfidence => {
                    // Predicted token + confidence for every still-masked position
                    // inside the current block (only these are candidates).
                    let mut candidates: Vec<(usize, i64, f32)> = Vec::new();
                    for offset in block_start..block_end {
                        let position = row_start + offset;
                        if tokens[position] != self.mask_token_id {
                            continue;
                        }
                        let logit_row = &all_logits[position * vocab..(position + 1) * vocab];
                        let gumbel = if self.temperature > 0.0 {
                            gumbel_uniforms(step, position, vocab)
                        } else {
                            Vec::new()
                        };
                        let (predicted, confidence) = self.predict_row(logit_row, &gumbel);
                        candidates.push((position, predicted, confidence));
                    }
                    if candidates.is_empty() {
                        continue;
                    }
                    // Commit the highest-confidence subset for this block-step. The
                    // even split of the block's masked count across its remaining
                    // steps equals ceil(remaining / remaining_steps_in_block).
                    candidates
                        .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    let commit = candidates.len().div_ceil(remaining_steps_in_block);
                    for &(position, predicted, _) in candidates.iter().take(commit) {
                        output[position] = predicted;
                    }
                }
                Remasking::Random => {
                    // MDLM ancestral update: each still-masked position in the block
                    // unmasks independently with probability 1/(steps remaining), so
                    // the expected unmasked fraction follows the log-linear schedule
                    // and the block's final step unmasks everything. The token is
                    // sampled from the model's categorical distribution.
                    let last_step_in_block = remaining_steps_in_block <= 1;
                    let unmask_prob = 1.0f64 / remaining_steps_in_block as f64;
                    for offset in block_start..block_end {
                        let position = row_start + offset;
                        if tokens[position] != self.mask_token_id {
                            continue;
                        }
                        if last_step_in_block || unmask_uniform(step, position) < unmask_prob {
                            let logit_row = &all_logits[position * vocab..(position + 1) * vocab];
                            output[position] = self.sample_token(logit_row, step, position, vocab);
                        }
                    }
                }
            }
        }
        Value::from_slice_i64(&output, &token_shape).map_err(Into::into)
    }
}

/// One uniform sample in `(0, 1)` per vocab entry for Gumbel-max sampling,
/// seeded deterministically from `(step, position)` so a run is reproducible.
///
/// Note: this is reproducible across onnx-genai runs but is NOT bit-identical to
/// LLaDA's `torch.rand`-based sampling; parity tests exercise `temperature = 0`.
fn gumbel_uniforms(step: usize, position: usize, vocab: usize) -> Vec<f64> {
    use rand::{Rng, SeedableRng};
    let seed = (step as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(position as u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..vocab)
        // Clamp away from 0 and 1 to keep -ln(-ln u) finite.
        .map(|_| rng.random::<f64>().clamp(1e-9, 1.0 - 1e-9))
        .collect()
}

/// One uniform sample in `[0, 1)` per `(step, position)` for the MDLM ancestral
/// per-position unmask decision, seeded deterministically (so a run is
/// reproducible) but with a distinct mix from [`gumbel_uniforms`] so the unmask
/// decision is independent of the token-sampling noise at the same position.
fn unmask_uniform(step: usize, position: usize) -> f64 {
    use rand::{Rng, SeedableRng};
    let seed = (step as u64)
        .wrapping_mul(0x2545_F491_4F6C_DD1D)
        .wrapping_add((position as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(0xA076_1D64_78BD_642F);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    rng.random::<f64>()
}

#[cfg(test)]
mod tests {
    use super::super::SchedulerRegistry;
    use super::*;
    use onnx_genai_metadata::SchedulerSpec;

    #[test]
    fn masked_diffusion_recovers_poisoned_generation_start_lock() {
        let scheduler = MaskedDiffusion {
            mask_token_id: 4,
            temperature: 0.0,
            block_length: None,
            remasking: Remasking::LowConfidence,
            generation_start: Mutex::new(None),
        };

        std::thread::scope(|scope| {
            let result = scope
                .spawn(|| {
                    let mut guard = scheduler.lock_generation_start();
                    *guard = Some(vec![2]);
                    panic!("poison generation-start lock");
                })
                .join();
            assert!(result.is_err());
        });

        assert_eq!(*scheduler.lock_generation_start(), Some(vec![2]));
    }

    #[test]
    fn masked_diffusion_random_unmasks_all_and_never_emits_mask() {
        // MDLM-style ancestral unmasking: by the final step every masked position
        // must be filled, the mask token must never be emitted (even though it has
        // the largest raw logit here), the prompt prefix is preserved, and the run
        // is deterministic.
        let vocab = 5usize;
        let mask_id = 4i64;
        let seq = 6usize;
        let prompt_len = 2usize;
        let num_steps = 4usize;

        // Sharp logits: the mask token has the highest logit (must be excluded),
        // and each position's highest *non-mask* logit is a distinct token.
        let mut logits = vec![0f32; seq * vocab];
        for pos in 0..seq {
            logits[pos * vocab + mask_id as usize] = 100.0;
            logits[pos * vocab + (pos % 4)] = 10.0;
        }
        let logits_value =
            Value::from_slice_f32(&logits, &[1, seq as i64, vocab as i64]).expect("logits");

        let sched = MaskedDiffusion {
            mask_token_id: mask_id,
            temperature: 0.0, // greedy token choice => deterministic, random unmask order
            block_length: None,
            remasking: Remasking::Random,
            generation_start: Mutex::new(None),
        };

        let seed = vec![1i64, 2, mask_id, mask_id, mask_id, mask_id];
        let run = |sched: &MaskedDiffusion| -> Vec<i64> {
            sched.reset();
            let mut value = Value::from_slice_i64(&seed, &[1, seq as i64]).expect("seed");
            for step in 0..num_steps {
                value = sched
                    .step(step, num_steps, &value, &logits_value)
                    .expect("step");
            }
            value.to_vec_i64().expect("tokens")
        };

        let out = run(&sched);
        assert_eq!(&out[..prompt_len], &[1, 2], "prompt prefix preserved");
        for (pos, &tok) in out.iter().enumerate() {
            assert_ne!(
                tok, mask_id,
                "position {pos} still masked / emitted the mask token"
            );
        }
        for (offset, &token) in out[prompt_len..seq].iter().enumerate() {
            let pos = prompt_len + offset;
            assert_eq!(token, (pos % 4) as i64, "position {pos} token");
        }
        assert_eq!(run(&sched), out, "ancestral sampling is deterministic");
    }

    #[test]
    fn masked_diffusion_rejects_unknown_remasking() {
        let registry = SchedulerRegistry::default();
        let spec = SchedulerSpec {
            kind: "masked_diffusion".to_string(),
            mask_token_id: Some(4),
            remasking: Some("nonsense".to_string()),
            ..SchedulerSpec::default()
        };
        assert!(registry.build(&spec, 4).is_err());
    }
}
