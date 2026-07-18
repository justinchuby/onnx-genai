//! Shared token-generation loop used by direct, session, priority, pipeline, and speculative paths.

use crate::config::{
    FinishReason, GenerateOptions, GenerateResult, GenerateToken, GenerateTokenCallback,
    TokenLogprob,
};
use crate::logits::{ProcessorChain, ProcessorContext, TokenId};
use crate::processors::{
    ensure_constrained_finish, finish_reason_after_token, is_device_portable_chain,
    select_next_token, select_next_token_with_rng, select_next_token_with_sampler,
};
use crate::sampling::{Sampler, SamplingRng};
use onnx_genai_ort::Tokenizer;

pub(crate) struct DecodeLoopState {
    pub(crate) generated_tokens: Vec<TokenId>,
    pub(crate) generated_text: String,
    pub(crate) step: usize,
    pub(crate) prefix_cache_hit_len: usize,
    pub(crate) logprobs: Option<Vec<TokenLogprob>>,
    pub(crate) rng: SamplingRng,
    /// Optional caller-supplied final token selector. When set it replaces the
    /// default greedy/categorical [`Sampler`] (the logit-processor chain still
    /// runs first) and disables the device greedy fast path. This is how the
    /// engine's public `generate_*_with_sampler` methods — and, through them,
    /// the C ABI — inject a foreign sampler into the shared decode loop.
    pub(crate) custom_sampler: Option<Box<dyn Sampler>>,
}

impl DecodeLoopState {
    pub(crate) fn new(
        prefix_cache_hit_len: usize,
        seed: Option<u64>,
        top_logprobs: Option<usize>,
    ) -> Self {
        Self::with_rng(prefix_cache_hit_len, SamplingRng::new(seed), top_logprobs)
    }

    pub(crate) fn with_rng(
        prefix_cache_hit_len: usize,
        rng: SamplingRng,
        top_logprobs: Option<usize>,
    ) -> Self {
        Self {
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            step: 0,
            prefix_cache_hit_len,
            logprobs: top_logprobs.map(|_| Vec::new()),
            rng,
            custom_sampler: None,
        }
    }
}

pub(crate) trait DecodeLoopBackend {
    fn context_len(&self) -> usize;
    fn processor_prompt_tokens(&self) -> Vec<TokenId>;
    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>>;
    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()>;
    /// Whether the backend can select the greedy (argmax) token internally,
    /// skipping host logits materialization. Only consulted for greedy
    /// decoding with no logit processors and no logprobs.
    fn greedy_fastpath_supported(&self) -> bool {
        false
    }
    /// Run one decode step and return only the argmax token id, selecting it on
    /// the device/buffer without copying the full vocabulary to the host. Only
    /// called when [`Self::greedy_fastpath_supported`] is true.
    fn next_token_greedy(&mut self) -> anyhow::Result<TokenId> {
        anyhow::bail!("greedy fast path is not supported by this decode backend")
    }
    /// Whether the backend can apply device-portable sampling without materializing
    /// host logits.
    fn sampled_fastpath_supported(&self) -> bool {
        false
    }
    /// Run one decode step and sample on the device.
    ///
    /// Returns `Ok(Some(token))` when the device sampler selected a token,
    /// `Ok(None)` when the fast path does not apply to this step (e.g. the
    /// multi-token prompt-prefill step, which has no captured graph) so the
    /// caller should fall back to host sampling *without* disabling the fast
    /// path for subsequent single-token decode steps, and `Err` on a genuine
    /// failure that should latch the fast path off.
    fn next_token_sampled(
        &mut self,
        _params: &onnx_genai_ort::DeviceSampleParams,
    ) -> anyhow::Result<Option<TokenId>> {
        anyhow::bail!("device sampled fast path is not supported by this decode backend")
    }
    /// Record that a device sampling attempt was unavailable for this backend.
    fn sampled_fastpath_failed(&mut self) {}
}

pub(crate) fn run_decode_loop<B: DecodeLoopBackend>(
    backend: &mut B,
    state: &mut DecodeLoopState,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    tokenizer: &Tokenizer,
    max_context: Option<usize>,
    mut callback: Option<&mut GenerateTokenCallback<'_>>,
) -> anyhow::Result<GenerateResult> {
    while state.generated_tokens.len() < options.max_new_tokens {
        if let Some(result) = step_decode_loop(
            backend,
            state,
            options,
            chain,
            tokenizer,
            max_context,
            callback.as_deref_mut(),
        )? {
            return Ok(result);
        }
    }

    ensure_constrained_finish(options, &state.generated_text, FinishReason::MaxTokens)?;
    finish_result(
        tokenizer,
        &state.generated_tokens,
        FinishReason::MaxTokens,
        state.prefix_cache_hit_len,
        state.logprobs.as_deref(),
    )
}

pub(crate) fn step_decode_loop<B: DecodeLoopBackend>(
    backend: &mut B,
    state: &mut DecodeLoopState,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    tokenizer: &Tokenizer,
    max_context: Option<usize>,
    callback: Option<&mut GenerateTokenCallback<'_>>,
) -> anyhow::Result<Option<GenerateResult>> {
    if reached_context_limit(backend.context_len(), max_context) {
        ensure_constrained_finish(options, &state.generated_text, FinishReason::Length)?;
        return finish_result(
            tokenizer,
            &state.generated_tokens,
            FinishReason::Length,
            state.prefix_cache_hit_len,
            state.logprobs.as_deref(),
        )
        .map(Some);
    }

    // A custom sampler replaces the default greedy/categorical selection, so the
    // device fast paths must be bypassed to give the sampler processed logits.
    let greedy_fastpath = chain.is_empty()
        && options.top_logprobs.is_none()
        && (options.greedy || options.temperature == 0.0)
        && state.custom_sampler.is_none()
        && backend.greedy_fastpath_supported();
    // Keep greedy behavior on its existing argmax path. The sampled path is only
    // for categorical decoding whose processor chain the device sampler supports.
    let sampled_fastpath = !greedy_fastpath
        && !options.greedy
        && options.temperature != 0.0
        && options.top_logprobs.is_none()
        && state.custom_sampler.is_none()
        && is_device_portable_chain(chain)
        && backend.sampled_fastpath_supported();

    let sampled_params = sampled_fastpath.then(|| onnx_genai_ort::DeviceSampleParams {
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        min_p: options.min_p,
        greedy: false,
        // Draw from the same request RNG as host categorical sampling. If the
        // device call fails, this value is reused by the host fallback.
        rng_value: state.rng.value_for(options),
    });

    let sampled_result = sampled_params.as_ref().map(|params| {
        let _span = onnx_genai_ort::prof_span!("loop.next_logits");
        backend.next_token_sampled(params)
    });

    // Only a hard error latches the device fast path off. `Ok(None)` means the
    // fast path did not apply to this step (the multi-token prefill has no
    // captured graph and returns host logits); it must keep the fast path armed
    // for the subsequent single-token decode steps that *can* device-sample.
    if sampled_result.as_ref().is_some_and(Result::is_err) {
        backend.sampled_fastpath_failed();
    }

    let mut host_token = |rng_value: Option<f32>| -> anyhow::Result<TokenId> {
        let context = ProcessorContext {
            prompt_tokens: backend.processor_prompt_tokens(),
            generated_tokens: state.generated_tokens.clone(),
            generated_text: state.generated_text.clone(),
            step: state.step,
        };
        let mut logits = {
            let _span = onnx_genai_ort::prof_span!("loop.next_logits");
            backend.next_logits()?
        };
        let token_id = {
            let _span = onnx_genai_ort::prof_span!("loop.sampling");
            if let Some(sampler) = state.custom_sampler.as_deref_mut() {
                select_next_token_with_sampler(&mut logits, &context, chain, sampler)
            } else if let Some(rng_value) = rng_value {
                select_next_token(&mut logits, &context, options, chain, rng_value)
            } else {
                select_next_token_with_rng(&mut logits, &context, options, chain, &mut state.rng)
            }
        };
        if let (Some(top_logprobs), Some(logprobs)) =
            (options.top_logprobs, state.logprobs.as_mut())
        {
            logprobs.push(logprob_for_token(&logits, token_id, top_logprobs));
        }
        Ok(token_id)
    };

    let token_id = if greedy_fastpath {
        let _span = onnx_genai_ort::prof_span!("loop.next_logits");
        backend.next_token_greedy()?
    } else if let Some(Ok(Some(token_id))) = sampled_result {
        token_id
    } else {
        // `step_sampled` is allowed to report that the device sampler is not
        // available. Reuse its draw to preserve seeded host sampling behavior.
        host_token(sampled_params.as_ref().map(|params| params.rng_value))?
    };
    {
        let _span = onnx_genai_ort::prof_span!("loop.commit_token");
        backend.commit_token(token_id)?;
    }

    let _commit_span = onnx_genai_ort::prof_span!("loop.commit_selected");
    if let Some(finish_reason) = commit_selected_token(
        state,
        backend.processor_prompt_tokens(),
        token_id,
        options,
        chain,
        tokenizer,
        callback,
    )? {
        return finish_result(
            tokenizer,
            &state.generated_tokens,
            finish_reason,
            state.prefix_cache_hit_len,
            state.logprobs.as_deref(),
        )
        .map(Some);
    }
    Ok(None)
}

pub(crate) fn commit_selected_token(
    state: &mut DecodeLoopState,
    prompt_tokens: Vec<TokenId>,
    token_id: TokenId,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    tokenizer: &Tokenizer,
    callback: Option<&mut GenerateTokenCallback<'_>>,
) -> anyhow::Result<Option<FinishReason>> {
    state.generated_tokens.push(token_id);
    let token_text = {
        let _span = onnx_genai_ort::prof_span!("loop.detokenize");
        tokenizer
            .decode(&[token_id])
            .map_err(|e| anyhow::anyhow!("Failed to detokenize token {token_id}: {}", e))?
    };
    state.generated_text.push_str(&token_text);
    let context = ProcessorContext {
        prompt_tokens,
        generated_tokens: state.generated_tokens.clone(),
        generated_text: state.generated_text.clone(),
        step: state.step,
    };
    let finish_reason = finish_reason_after_token(token_id, options, chain, &context);
    if let Some(callback) = callback {
        callback(GenerateToken {
            token_id,
            text: token_text,
            finish_reason: finish_reason.clone(),
        })?;
    }
    state.step += 1;
    Ok(finish_reason)
}

pub(crate) fn finish_result(
    tokenizer: &Tokenizer,
    generated_tokens: &[TokenId],
    finish_reason: FinishReason,
    prefix_cache_hit_len: usize,
    logprobs: Option<&[TokenLogprob]>,
) -> anyhow::Result<GenerateResult> {
    Ok(GenerateResult {
        text: tokenizer
            .decode(generated_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to detokenize generated tokens: {}", e))?,
        token_ids: generated_tokens.to_vec(),
        finish_reason,
        prefix_cache_hit_len,
        logprobs: logprobs.map(<[TokenLogprob]>::to_vec),
    })
}

pub(crate) fn logprob_for_token(
    logits: &[f32],
    token_id: TokenId,
    top_logprobs: usize,
) -> TokenLogprob {
    let max_logit = logits
        .iter()
        .copied()
        .filter(|logit| logit.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let logsumexp = max_logit
        + logits
            .iter()
            .copied()
            .filter(|logit| logit.is_finite())
            .map(|logit| (logit - max_logit).exp())
            .sum::<f32>()
            .ln();
    let logprob = logits
        .get(token_id as usize)
        .copied()
        .filter(|logit| logit.is_finite())
        .map_or(f32::NEG_INFINITY, |logit| logit - logsumexp);

    let mut top = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, logit)| logit.is_finite())
        .map(|(id, logit)| (id as TokenId, logit - logsumexp))
        .collect::<Vec<_>>();
    top.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    top.truncate(top_logprobs);
    if !top.iter().any(|(id, _)| *id == token_id) {
        top.push((token_id, logprob));
        top.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    }

    TokenLogprob {
        token_id,
        logprob,
        top,
    }
}

pub(crate) fn reached_context_limit(
    current_context_len: usize,
    max_context: Option<usize>,
) -> bool {
    max_context.is_some_and(|limit| current_context_len >= limit)
}

pub(crate) fn exceeded_context_limit(
    current_context_len: usize,
    max_context: Option<usize>,
) -> bool {
    max_context.is_some_and(|limit| current_context_len > limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::build_processor_chain;
    use std::path::Path;

    #[derive(Clone, Copy)]
    enum SampledOutcome {
        HardError,
        NotApplicable,
        #[allow(dead_code)]
        Token(TokenId),
    }

    struct MockBackend {
        logits: Vec<Vec<f32>>,
        next_logits: usize,
        sampled_attempts: usize,
        sampled_supported: bool,
        sampled_failed: bool,
        sampled_outcome: SampledOutcome,
        committed: Vec<TokenId>,
    }

    impl MockBackend {
        fn new(sampled_supported: bool) -> Self {
            Self::with_outcome(sampled_supported, SampledOutcome::HardError)
        }

        fn with_outcome(sampled_supported: bool, sampled_outcome: SampledOutcome) -> Self {
            Self {
                logits: vec![
                    vec![0.0, 0.4, 1.0],
                    vec![0.5, 1.0, 0.0],
                    vec![1.0, 0.0, 0.5],
                ],
                next_logits: 0,
                sampled_attempts: 0,
                sampled_supported,
                sampled_failed: false,
                sampled_outcome,
                committed: Vec::new(),
            }
        }
    }

    impl DecodeLoopBackend for MockBackend {
        fn context_len(&self) -> usize {
            self.committed.len() + 1
        }

        fn processor_prompt_tokens(&self) -> Vec<TokenId> {
            vec![0]
        }

        fn next_logits(&mut self) -> anyhow::Result<Vec<f32>> {
            let logits = self.logits[self.next_logits].clone();
            self.next_logits += 1;
            Ok(logits)
        }

        fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()> {
            self.committed.push(token_id);
            Ok(())
        }

        fn sampled_fastpath_supported(&self) -> bool {
            self.sampled_supported && !self.sampled_failed
        }

        fn next_token_sampled(
            &mut self,
            _params: &onnx_genai_ort::DeviceSampleParams,
        ) -> anyhow::Result<Option<TokenId>> {
            self.sampled_attempts += 1;
            match self.sampled_outcome {
                SampledOutcome::HardError => anyhow::bail!("device sampler unavailable"),
                SampledOutcome::NotApplicable => Ok(None),
                SampledOutcome::Token(token) => Ok(Some(token)),
            }
        }

        fn sampled_fastpath_failed(&mut self) {
            self.sampled_failed = true;
        }
    }

    fn tokenizer() -> anyhow::Result<Tokenizer> {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json")
            .canonicalize()?;
        Tokenizer::from_file(&fixture).map_err(Into::into)
    }

    #[test]
    fn sampled_fastpath_error_falls_back_to_seeded_host_sampling() -> anyhow::Result<()> {
        let options = GenerateOptions {
            max_new_tokens: 3,
            greedy: false,
            temperature: 0.8,
            top_k: 2,
            top_p: 0.95,
            min_p: 0.1,
            seed: Some(17),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None)?;
        let tokenizer = tokenizer()?;

        let mut fallback_backend = MockBackend::new(true);
        let mut fallback_state = DecodeLoopState::new(0, options.seed, None);
        let fallback = run_decode_loop(
            &mut fallback_backend,
            &mut fallback_state,
            &options,
            &chain,
            &tokenizer,
            None,
            None,
        )?;

        let mut host_backend = MockBackend::new(false);
        let mut host_state = DecodeLoopState::new(0, options.seed, None);
        let host = run_decode_loop(
            &mut host_backend,
            &mut host_state,
            &options,
            &chain,
            &tokenizer,
            None,
            None,
        )?;

        assert_eq!(fallback.token_ids, host.token_ids);
        assert_eq!(fallback_backend.committed, host_backend.committed);
        assert_eq!(fallback_backend.sampled_attempts, 1);
        Ok(())
    }

    #[test]
    fn sampled_fastpath_not_applicable_does_not_latch_off() -> anyhow::Result<()> {
        // `Ok(None)` (e.g. the multi-token prefill step) must fall back to host
        // sampling for that step WITHOUT disabling the device fast path, so the
        // backend keeps being asked on every subsequent step.
        let options = GenerateOptions {
            max_new_tokens: 3,
            greedy: false,
            temperature: 0.8,
            top_k: 2,
            top_p: 0.95,
            min_p: 0.1,
            seed: Some(17),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None)?;
        let tokenizer = tokenizer()?;

        let mut na_backend = MockBackend::with_outcome(true, SampledOutcome::NotApplicable);
        let mut na_state = DecodeLoopState::new(0, options.seed, None);
        let na = run_decode_loop(
            &mut na_backend,
            &mut na_state,
            &options,
            &chain,
            &tokenizer,
            None,
            None,
        )?;

        let mut host_backend = MockBackend::new(false);
        let mut host_state = DecodeLoopState::new(0, options.seed, None);
        let host = run_decode_loop(
            &mut host_backend,
            &mut host_state,
            &options,
            &chain,
            &tokenizer,
            None,
            None,
        )?;

        // Same tokens as the pure host path (device never selected a token) ...
        assert_eq!(na.token_ids, host.token_ids);
        assert_eq!(na_backend.committed, host_backend.committed);
        // ... but the fast path stayed armed and was retried every step.
        assert_eq!(na_backend.sampled_attempts, host_backend.committed.len());
        assert!(!na_backend.sampled_failed);
        Ok(())
    }
}
