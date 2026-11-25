//! Shared token-generation loop used by direct, session, priority, pipeline, and speculative paths.

use anyhow::Context as _;

use crate::config::{
    FinishReason, GenerateOptions, GenerateResult, GenerateToken, GenerateTokenCallback,
    TokenLogprob,
};
use crate::logits::{ProcessorChain, ProcessorContext, TokenId};
use crate::processors::{
    finish_reason_after_token, select_next_token, select_next_token_with_rng,
    select_next_token_with_sampler,
};
use crate::sampling::{Sampler, SamplingRng};
use onnx_genai_ort::Tokenizer;

/// A physical prefix can come from an older request or be evaluated during
/// this request to capture a semantic boundary. Only the former is cache reuse.
#[derive(Clone, Copy)]
// Only Native constructs prepared prefixes; the shared clock also builds in
// reference-only ORT configurations.
#[cfg_attr(not(feature = "native-backend"), allow(dead_code))]
pub(crate) enum PreparedPrompt {
    Reused {
        tokens: usize,
    },
    Evaluated {
        tokens: usize,
        elapsed: std::time::Duration,
    },
}
impl PreparedPrompt {
    pub(crate) fn tokens(self) -> usize {
        match self {
            Self::Reused { tokens } | Self::Evaluated { tokens, .. } => tokens,
        }
    }
    #[cfg(any(feature = "native-backend", test))]
    pub(crate) fn cached_tokens(self) -> usize {
        match self {
            Self::Reused { tokens } => tokens,
            Self::Evaluated { .. } => 0,
        }
    }
    fn elapsed(self) -> std::time::Duration {
        match self {
            Self::Reused { .. } => std::time::Duration::ZERO,
            Self::Evaluated { elapsed, .. } => elapsed,
        }
    }
}

/// Carries only active decoder work across scheduler turns. Every step closes
/// before invoking the caller's callback, so delayed publication cannot inflate
/// model throughput measurements.
#[derive(Clone, Default)]
pub(crate) struct GenerationClock {
    prepared: Option<PreparedPrompt>,
    step_started: Option<std::time::Instant>,
    sample: Option<crate::config::GenerationTimings>,
    prompt_finished: bool,
}
impl GenerationClock {
    #[cfg(any(feature = "native-backend", test))]
    pub(crate) fn prepare_prompt(&mut self, prefix: PreparedPrompt) {
        self.prepared = Some(prefix);
    }
    pub(crate) fn has_finished_prompt(&self) -> bool {
        self.prompt_finished
    }
    pub(crate) fn initial_progress(
        &self,
        total: usize,
        cache: usize,
    ) -> crate::pipeline::PromptProgress {
        crate::pipeline::PromptProgress {
            total,
            cache,
            processed: self.prepared.map_or(cache, PreparedPrompt::tokens),
            elapsed: self
                .prepared
                .map_or(std::time::Duration::ZERO, PreparedPrompt::elapsed),
        }
    }

    fn begin_forward(&mut self, prompt_tokens: usize, cached_prompt_tokens: usize) {
        self.begin_forward_at(
            std::time::Instant::now(),
            prompt_tokens,
            cached_prompt_tokens,
        );
    }
    fn begin_forward_at(
        &mut self,
        now: std::time::Instant,
        prompt_tokens: usize,
        cached_prompt_tokens: usize,
    ) {
        self.step_started = Some(now);
        self.sample.get_or_insert(crate::config::GenerationTimings {
            prompt_tokens: prompt_tokens.saturating_sub(cached_prompt_tokens),
            cached_prompt_tokens,
            generated_tokens: 0,
            prompt_elapsed: self
                .prepared
                .map_or(std::time::Duration::ZERO, PreparedPrompt::elapsed),
            generation_elapsed: std::time::Duration::ZERO,
        });
    }
    fn finish_forward(&mut self) {
        self.finish_forward_at(std::time::Instant::now());
    }
    fn finish_forward_at(&mut self, now: std::time::Instant) {
        if !self.prompt_finished {
            if let (Some(start), Some(sample)) = (self.step_started, self.sample.as_mut()) {
                sample.prompt_elapsed += now.duration_since(start);
                self.step_started = Some(now);
            }
            self.prompt_finished = true;
        }
    }
    fn token_selected(&mut self, generated_tokens: usize) {
        self.token_selected_at(std::time::Instant::now(), generated_tokens);
    }
    fn token_selected_at(&mut self, now: std::time::Instant, generated_tokens: usize) {
        if let (Some(start), Some(sample)) = (self.step_started.take(), self.sample.as_mut()) {
            sample.generation_elapsed += now.duration_since(start);
            sample.generated_tokens = generated_tokens;
        }
    }
    pub(crate) fn snapshot(&self) -> Option<crate::config::GenerationTimings> {
        self.sample
    }
}

pub(crate) struct DecodeLoopState {
    pub(crate) clock: GenerationClock,
    pub(crate) generated_tokens: Vec<TokenId>,
    pub(crate) generated_text: String,
    pub(crate) text_decoder: crate::tokenization::TokenTextDecoder,
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
            clock: GenerationClock::default(),
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            text_decoder: Default::default(),
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
    fn processor_prompt_tokens(&self) -> &[TokenId];
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

/// What one decoder forward pass produced.
///
/// Either the backend selected the token itself on the device (the greedy-argmax
/// and device-sampling fast paths, which never materialize host logits) or it
/// returned host logits for the policy to score. Splitting the step here is what
/// lets the workflow interpreter own the token loop while
/// [`select_and_commit_step`] stays the single implementation of sampling,
/// stopping, and commit — a second copy of that policy is exactly what this
/// split exists to avoid.
pub(crate) struct ForwardOutcome {
    /// Token the device picked, when a fast path applied.
    pub(crate) device_token: Option<TokenId>,
    /// Host logits, when no fast path applied. Moved into the caller.
    pub(crate) logits: Option<Vec<f32>>,
    /// RNG draw already consumed by an attempted device sample, reused by the
    /// host fallback so a seeded run samples identically either way.
    pub(crate) rng_value: Option<f32>,
}

/// Run one decoder forward pass, taking the device fast path when it applies.
pub(crate) fn forward_step<B: DecodeLoopBackend + ?Sized>(
    backend: &mut B,
    state: &mut DecodeLoopState,
    options: &GenerateOptions,
    chain: &ProcessorChain,
) -> anyhow::Result<ForwardOutcome> {
    state.clock.begin_forward(
        backend
            .context_len()
            .saturating_sub(state.generated_tokens.len()),
        state.prefix_cache_hit_len,
    );
    let outcome = forward_step_inner(backend, state, options, chain)?;
    state.clock.finish_forward();
    Ok(outcome)
}

fn forward_step_inner<B: DecodeLoopBackend + ?Sized>(
    backend: &mut B,
    state: &mut DecodeLoopState,
    options: &GenerateOptions,
    chain: &ProcessorChain,
) -> anyhow::Result<ForwardOutcome> {
    let plan = crate::processors::device_sampling_plan(
        chain,
        options,
        state.custom_sampler.is_some(),
        backend.greedy_fastpath_supported(),
        backend.sampled_fastpath_supported(),
    );
    if matches!(plan, crate::processors::DeviceSamplingPlan::Greedy) {
        let _span = onnx_genai_ort::prof_span!("loop.next_logits");
        return Ok(ForwardOutcome {
            device_token: Some(backend.next_token_greedy()?),
            logits: None,
            rng_value: None,
        });
    }
    let sampled_params =
        matches!(plan, crate::processors::DeviceSamplingPlan::Sampled).then(|| {
            onnx_genai_ort::DeviceSampleParams {
                temperature: options.temperature,
                top_k: options.top_k,
                top_p: options.top_p,
                min_p: options.min_p,
                greedy: false,
                rng_value: state.rng.value_for(options),
            }
        });
    if let Some(params) = sampled_params.as_ref() {
        let result = {
            let _span = onnx_genai_ort::prof_span!("loop.next_logits");
            backend.next_token_sampled(params)
        };
        // Only a hard error latches the fast path off. `Ok(None)` means it did
        // not apply to this step (multi-token prefill has no captured graph) and
        // must stay armed for the single-token steps that can device-sample.
        match result {
            Ok(Some(token)) => {
                return Ok(ForwardOutcome {
                    device_token: Some(token),
                    logits: None,
                    rng_value: Some(params.rng_value),
                });
            }
            Ok(None) => {}
            Err(_) => backend.sampled_fastpath_failed(),
        }
    }
    let logits = {
        let _span = onnx_genai_ort::prof_span!("loop.next_logits");
        backend.next_logits()?
    };
    Ok(ForwardOutcome {
        device_token: None,
        logits: Some(logits),
        rng_value: sampled_params.as_ref().map(|params| params.rng_value),
    })
}

/// Score, commit, and stop-check one step's forward outcome.
///
/// The single implementation of the next-token policy: the logit-processor
/// chain, the sampler, logprob capture, KV commit, and stop/EOS detection. Both
/// the run-to-completion canonical loop and the scheduler's per-step drive call
/// it, so there is one policy no matter who owns the iteration.
pub(crate) fn select_and_commit_step<B: DecodeLoopBackend + ?Sized>(
    backend: &mut B,
    state: &mut DecodeLoopState,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    tokenizer: &Tokenizer,
    forward: ForwardOutcome,
    callback: Option<&mut GenerateTokenCallback<'_>>,
) -> anyhow::Result<(TokenId, Option<FinishReason>)> {
    anyhow::ensure!(
        state.custom_sampler.is_none()
            || options.top_logprobs.is_none()
            || options.logprob_source != crate::config::LogprobSource::Sampling,
        "logprob_source=Sampling cannot inspect a caller-supplied sampler's distribution; request Model or Processed probabilities instead"
    );
    let token_id = match forward.device_token {
        Some(token_id) => token_id,
        None => {
            let context = if chain.is_empty() && state.custom_sampler.is_none() {
                ProcessorContext::default()
            } else {
                ProcessorContext {
                    prompt_tokens: backend.processor_prompt_tokens().to_vec(),
                    generated_tokens: state.generated_tokens.clone(),
                    generated_text: state.generated_text.clone(),
                    step: state.step,
                }
            };
            let mut logits = forward
                .logits
                .context("a forward pass produced neither a device token nor host logits")?;
            let probabilities = ProbabilitySnapshot::capture(options, &logits);
            let token_id = {
                let _span = onnx_genai_ort::prof_span!("loop.sampling");
                if let Some(sampler) = state.custom_sampler.as_deref_mut() {
                    select_next_token_with_sampler(&mut logits, &context, chain, sampler)?
                } else if let Some(rng_value) = forward.rng_value {
                    select_next_token(&mut logits, &context, options, chain, rng_value)?
                } else {
                    select_next_token_with_rng(
                        &mut logits,
                        &context,
                        options,
                        chain,
                        &mut state.rng,
                    )?
                }
            };
            if let (Some(top_logprobs), Some(logprobs)) =
                (options.top_logprobs, state.logprobs.as_mut())
            {
                logprobs.push(probabilities.for_token(&logits, token_id, top_logprobs));
            }
            token_id
        }
    };
    {
        let _span = onnx_genai_ort::prof_span!("loop.commit_token");
        backend.commit_token(token_id)?;
    }
    let _commit_span = onnx_genai_ort::prof_span!("loop.commit_selected");
    let finish = commit_selected_token(
        state,
        backend.processor_prompt_tokens(),
        token_id,
        options,
        chain,
        tokenizer,
        callback,
    )?;
    Ok((token_id, finish))
}

pub(crate) fn commit_selected_token(
    state: &mut DecodeLoopState,
    prompt_tokens: &[TokenId],
    token_id: TokenId,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    tokenizer: &Tokenizer,
    callback: Option<&mut GenerateTokenCallback<'_>>,
) -> anyhow::Result<Option<FinishReason>> {
    if state.step == 0 {
        state.text_decoder = crate::tokenization::TokenTextDecoder::with_preserved_tokens(
            &options.effective_preserved_tokens(),
        );
    }
    state.generated_tokens.push(token_id);
    // Which token this step produced. Without it a timeline shows that a step
    // happened but not what came out of it, so a stall cannot be tied to the
    // text being generated at the time. The text is whatever the loop already
    // decoded — tracing does not run the tokenizer itself, so a traced run
    // decodes exactly as often as an untraced one.
    let mut commit_span = onnx_genai_ort::prof_span!("loop.commit_selected_token");
    commit_span.set_arg("token_id", token_id);
    let decode_text = callback.is_some() || !chain.is_empty() || state.custom_sampler.is_some();
    let mut token_text = if decode_text {
        let _span = onnx_genai_ort::prof_span!("loop.detokenize");
        state.text_decoder.step(tokenizer, token_id)?
    } else {
        String::new()
    };
    if !token_text.is_empty() {
        commit_span.set_arg("token", token_text.clone());
    }
    state.generated_text.push_str(&token_text);
    let mut context = if chain.is_empty() {
        ProcessorContext::default()
    } else {
        ProcessorContext {
            prompt_tokens: prompt_tokens.to_vec(),
            generated_tokens: state.generated_tokens.clone(),
            generated_text: state.generated_text.clone(),
            step: state.step,
        }
    };
    let mut finish_reason = finish_reason_after_token(token_id, options, chain, &context);
    if decode_text && (finish_reason.is_some() || state.step + 1 >= options.max_new_tokens) {
        let tail = state.text_decoder.finish(tokenizer)?;
        state.generated_text.push_str(&tail);
        token_text.push_str(&tail);
        if !tail.is_empty() && finish_reason.is_none() && !chain.is_empty() {
            context.generated_text.push_str(&tail);
            finish_reason = finish_reason_after_token(token_id, options, chain, &context);
        }
    }
    state.clock.token_selected(state.generated_tokens.len());
    if let Some(callback) = callback {
        callback(GenerateToken {
            timings: state.clock.snapshot(),
            logprob: state
                .logprobs
                .as_ref()
                .and_then(|values| values.last())
                .cloned(),
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
    preserved_tokens: &std::collections::BTreeSet<TokenId>,
) -> anyhow::Result<GenerateResult> {
    Ok(GenerateResult {
        timings: None,
        retained_cache_tokens: 0,
        text: crate::tokenization::decode_output(tokenizer, generated_tokens, preserved_tokens)?,
        preserved_tokens: preserved_tokens.clone(),
        token_ids: generated_tokens.to_vec(),
        finish_reason,
        tool_calls: Vec::new(),
        prefix_cache_hit_len,
        logprobs: logprobs.map(<[TokenLogprob]>::to_vec),
        budget_cap: None,
    })
}

/// Retain model logits only when probability reporting requests them. Sampling
/// mutates its own logits; the selected token is scored against this snapshot.
pub(crate) struct ProbabilitySnapshot {
    model: Option<Vec<f32>>,
    greedy: bool,
}

impl ProbabilitySnapshot {
    pub(crate) fn capture(options: &GenerateOptions, logits: &[f32]) -> Self {
        Self {
            model: (options.top_logprobs.is_some()
                && options.logprob_source == crate::config::LogprobSource::Model)
                .then(|| logits.to_vec()),
            greedy: options.logprob_source == crate::config::LogprobSource::Sampling
                && options.selects_greedily(),
        }
    }

    pub(crate) fn for_token(&self, processed: &[f32], token: TokenId, top: usize) -> TokenLogprob {
        if self.greedy {
            return TokenLogprob {
                token_id: token,
                logprob: 0.0,
                top: vec![(token, 0.0)],
            };
        }
        logprob_for_token(self.model.as_deref().unwrap_or(processed), token, top)
    }
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
        .filter(|(_, logit)| logit.is_finite() || *logit == f32::NEG_INFINITY)
        .map(|(id, logit)| {
            (
                id as TokenId,
                if logit.is_finite() {
                    logit - logsumexp
                } else {
                    f32::NEG_INFINITY
                },
            )
        })
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
    #[test]
    fn model_probability_snapshot_precedes_bias_and_preserves_selected_metadata() {
        let model = vec![1.0, 2.0, 3.0];
        let options = crate::GenerateOptions {
            top_logprobs: Some(1),
            logprob_source: crate::LogprobSource::Model,
            ..Default::default()
        };
        let snapshot = super::ProbabilitySnapshot::capture(&options, &model);
        let processed = vec![1000.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let actual = snapshot.for_token(&processed, 0, 1);
        assert_eq!(actual, super::logprob_for_token(&model, 0, 1));
        assert_eq!(actual.top.len(), 2); // selected token remains distinct from top-1
        let zero = super::logprob_for_token(&[0.0, f32::NEG_INFINITY], 0, 2);
        assert_eq!(zero.top, vec![(0, 0.0), (1, f32::NEG_INFINITY)]);
        let processed_options = crate::GenerateOptions {
            logprob_source: crate::LogprobSource::Processed,
            ..options.clone()
        };
        let snapshot = super::ProbabilitySnapshot::capture(&processed_options, &model);
        assert!(snapshot.model.is_none());
        assert_eq!(snapshot.for_token(&processed, 0, 1).logprob, 0.0);
        let disabled = crate::GenerateOptions {
            top_logprobs: None,
            ..options
        };
        assert!(
            super::ProbabilitySnapshot::capture(&disabled, &model)
                .model
                .is_none()
        );
    }

    /// The interpreter these loop tests drive.
    ///
    /// The loop is the workflow's, so a unit test must supply one; using the
    /// same emitter a minimal real decoder goes through keeps the test on the
    /// production path instead of a bespoke shape.
    fn canonical_runtime() -> crate::pipeline::WorkflowRuntime {
        crate::pipeline::generation::test_decoder_runtime()
            .expect("a minimal decoder workflow builds a hosted interpreter")
    }

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
            Self::with_logits(
                sampled_supported,
                sampled_outcome,
                vec![
                    vec![0.0, 0.4, 1.0],
                    vec![0.5, 1.0, 0.0],
                    vec![1.0, 0.0, 0.5],
                ],
            )
        }

        fn with_logits(
            sampled_supported: bool,
            sampled_outcome: SampledOutcome,
            logits: Vec<Vec<f32>>,
        ) -> Self {
            Self {
                logits,
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

        fn processor_prompt_tokens(&self) -> &[TokenId] {
            &[0]
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

    fn tool_protocol_tokenizer() -> anyhow::Result<Tokenizer> {
        Tokenizer::from_file(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tiny-tool-call/tokenizer.json"),
        )
        .map_err(Into::into)
    }

    fn generated_tokens(tokenizer: &Tokenizer, text: &str) -> anyhow::Result<Vec<TokenId>> {
        let tokens = tokenizer.encode(text)?;
        Ok(tokens
            .into_iter()
            .filter(|token| {
                tokenizer
                    .decode(&[*token])
                    .is_ok_and(|piece| !piece.is_empty())
            })
            .collect())
    }

    #[test]
    fn terminal_token_projection_matches_committed_history_and_decoder_policy() -> anyhow::Result<()>
    {
        use crate::tokenization::{SpecialTokenHandling, TokenTextDecoder};
        use serde_json::{Value, json};
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json");
        let original: Value = serde_json::from_slice(&std::fs::read(source)?)?;
        for registered_special in [false, true] {
            let mut value = original.clone();
            if registered_special {
                value["added_tokens"] = json!([{
                    "id":3,"content":"<eos>","single_word":false,"lstrip":false,
                    "rstrip":false,"normalized":false,"special":true
                }]);
            }
            let scratch = tempfile::tempdir()?;
            let path = scratch.path().join("tokenizer.json");
            std::fs::write(&path, serde_json::to_vec(&value)?)?;
            let tokenizer = Tokenizer::from_file(path)?;
            for stop_on_eos in [false, true] {
                let options = GenerateOptions {
                    eos_token_id: Some(3),
                    stop_on_eos,
                    max_new_tokens: 2,
                    ..Default::default()
                };
                let mut state = DecodeLoopState::new(0, None, None);
                let mut events = Vec::new();
                let mut callback = |token| {
                    events.push(token);
                    Ok(())
                };
                let finish = commit_selected_token(
                    &mut state,
                    &[],
                    3,
                    &options,
                    &ProcessorChain::new(),
                    &tokenizer,
                    Some(&mut callback),
                )?;
                assert_eq!(finish, stop_on_eos.then_some(FinishReason::EosToken));
                assert_eq!(state.generated_tokens, [3]);
                let text = if registered_special { "" } else { "<eos>" };
                assert_eq!(state.generated_text, text);
                assert_eq!(events[0].text, text);
                assert_eq!(events[0].finish_reason, finish);
                let mut projection = TokenTextDecoder::new(SpecialTokenHandling::Preserve);
                let mut projected = projection.step(&tokenizer, 3)?;
                projected.push_str(&projection.finish(&tokenizer)?);
                assert_eq!(
                    projected,
                    tokenizer.decode_with_special_tokens(&state.generated_tokens)?
                );
                assert_eq!(projected, "<eos>");
                if let Some(finish) = finish {
                    let result = finish_result(
                        &tokenizer,
                        &state.generated_tokens,
                        finish,
                        0,
                        None,
                        &options.effective_preserved_tokens(),
                    )?;
                    assert_eq!(result.token_ids, [3]);
                    assert_eq!(result.text, text);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn text_stop_uses_incremental_sequence_text_instead_of_isolated_tokens() -> anyhow::Result<()> {
        let tokenizer = tokenizer()?;
        let options = GenerateOptions {
            max_new_tokens: 8,
            stop_sequences: vec![crate::logits::StopSequence::Text("tok22 tok22".into())],
            ..Default::default()
        };
        let chain = build_processor_chain(&options, Some(&tokenizer), false)?;
        let mut state = DecodeLoopState::new(0, None, None);
        let mut chunks = Vec::new();
        let mut callback = |token: GenerateToken| {
            chunks.push(token.text);
            Ok(())
        };
        assert_eq!(
            commit_selected_token(
                &mut state,
                &[],
                22,
                &options,
                &chain,
                &tokenizer,
                Some(&mut callback)
            )?,
            None
        );
        assert_eq!(
            commit_selected_token(
                &mut state,
                &[],
                22,
                &options,
                &chain,
                &tokenizer,
                Some(&mut callback)
            )?,
            Some(FinishReason::StopSequence { index: 0 })
        );
        assert_eq!(state.generated_text, "tok22 tok22");
        assert_eq!(chunks.concat(), state.generated_text);
        Ok(())
    }
    #[test]
    fn sampled_fastpath_error_falls_back_to_seeded_host_sampling() -> anyhow::Result<()> {
        let canonical = canonical_runtime();
        let options = GenerateOptions {
            max_new_tokens: 3,
            greedy: false,
            temperature: 0.8,
            top_k: 2,
            top_p: 0.95,
            min_p: 0.1,
            sampler_order: Some(
                crate::SamplerSequence::new(vec![
                    crate::SamplerStage::Temperature,
                    crate::SamplerStage::TopK,
                    crate::SamplerStage::TopP,
                    crate::SamplerStage::MinP,
                ])
                .unwrap(),
            ),
            seed: Some(17),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false)?;
        let tokenizer = tokenizer()?;

        let mut fallback_backend = MockBackend::new(true);
        let mut fallback_state = DecodeLoopState::new(0, options.seed, None);
        let fallback = crate::pipeline::generation::generate_with_decode_core(
            &canonical,
            &mut fallback_backend,
            &mut fallback_state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
            None,
        )?;

        let mut host_backend = MockBackend::new(false);
        let mut host_state = DecodeLoopState::new(0, options.seed, None);
        let host = crate::pipeline::generation::generate_with_decode_core(
            &canonical,
            &mut host_backend,
            &mut host_state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
            None,
        )?;

        assert_eq!(fallback.token_ids, host.token_ids);
        assert_eq!(fallback_backend.committed, host_backend.committed);
        assert_eq!(fallback_backend.sampled_attempts, 1);
        Ok(())
    }

    #[test]
    fn seeded_generation_reproduces_the_full_multi_token_stream() -> anyhow::Result<()> {
        let tokenizer = tokenizer()?;
        let generate = |seed| -> anyhow::Result<Vec<TokenId>> {
            let options = GenerateOptions {
                max_new_tokens: 24,
                greedy: false,
                stop_on_eos: false,
                seed: Some(seed),
                ..Default::default()
            };
            let chain = build_processor_chain(&options, None, false)?;
            let mut backend = MockBackend::with_logits(
                false,
                SampledOutcome::HardError,
                vec![vec![0.0; 8]; options.max_new_tokens],
            );
            let mut state = DecodeLoopState::new(0, options.seed, None);
            Ok(crate::pipeline::generation::generate_with_decode_core(
                &canonical_runtime(),
                &mut backend,
                &mut state,
                &[],
                crate::pipeline::generation::GenerationRequest {
                    control: None,
                    options: &options,
                    chain: &chain,
                    tokenizer: &tokenizer,
                    max_context: None,
                },
                None,
            )?
            .token_ids)
        };

        let first = generate(42)?;
        assert_eq!(first.len(), 24);
        assert_eq!(first, generate(42)?);
        assert_ne!(first, generate(43)?);
        Ok(())
    }

    #[test]
    fn sampled_fastpath_not_applicable_does_not_latch_off() -> anyhow::Result<()> {
        let canonical = canonical_runtime();
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
            sampler_order: Some(
                crate::SamplerSequence::new(vec![
                    crate::SamplerStage::Temperature,
                    crate::SamplerStage::TopK,
                    crate::SamplerStage::TopP,
                    crate::SamplerStage::MinP,
                ])
                .unwrap(),
            ),
            seed: Some(17),
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false)?;
        let tokenizer = tokenizer()?;

        let mut na_backend = MockBackend::with_outcome(true, SampledOutcome::NotApplicable);
        let mut na_state = DecodeLoopState::new(0, options.seed, None);
        let na = crate::pipeline::generation::generate_with_decode_core(
            &canonical,
            &mut na_backend,
            &mut na_state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
            None,
        )?;

        let mut host_backend = MockBackend::new(false);
        let mut host_state = DecodeLoopState::new(0, options.seed, None);
        let host = crate::pipeline::generation::generate_with_decode_core(
            &canonical,
            &mut host_backend,
            &mut host_state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
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

    #[test]
    fn unsupported_sampled_runner_never_runs_device_step() -> anyhow::Result<()> {
        let canonical = canonical_runtime();
        // A backend that cannot sample on the device (`sampled_fastpath_supported`
        // is false) must never have `next_token_sampled` invoked. That method runs
        // a model step which advances KV state; if it were called and then failed,
        // the host fallback would re-run the step and double-advance the KV cache,
        // corrupting the first decode token. Guard that the device step is skipped
        // entirely and the pure host path is taken.
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
        let chain = build_processor_chain(&options, None, false)?;
        let tokenizer = tokenizer()?;

        // `HardError` outcome would panic-latch if ever reached; assert it isn't.
        let mut backend = MockBackend::with_outcome(false, SampledOutcome::HardError);
        let mut state = DecodeLoopState::new(0, options.seed, None);
        let result = crate::pipeline::generation::generate_with_decode_core(
            &canonical,
            &mut backend,
            &mut state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
            None,
        )?;

        assert!(!result.token_ids.is_empty());
        assert_eq!(backend.sampled_attempts, 0);
        assert!(!backend.sampled_failed);
        Ok(())
    }

    #[test]
    fn empty_processor_chain_preserves_final_and_streamed_text() -> anyhow::Result<()> {
        let options = GenerateOptions {
            max_new_tokens: 3,
            greedy: true,
            temperature: 0.0,
            stop_on_eos: false,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false)?;
        assert!(chain.is_empty());
        let tokenizer = tokenizer()?;

        let canonical = canonical_runtime();
        let mut plain_backend = MockBackend::new(false);
        let mut plain_state = DecodeLoopState::new(0, options.seed, None);
        let plain = crate::pipeline::generation::generate_with_decode_core(
            &canonical,
            &mut plain_backend,
            &mut plain_state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
            None,
        )?;

        let mut streamed_ids = Vec::new();
        let mut callback = |token: GenerateToken| {
            streamed_ids.push(token.token_id);
            assert!(!token.text.is_empty());
            Ok(())
        };
        let canonical = canonical_runtime();
        let mut streamed_backend = MockBackend::new(false);
        let mut streamed_state = DecodeLoopState::new(0, options.seed, None);
        let streamed = crate::pipeline::generation::generate_with_decode_core(
            &canonical,
            &mut streamed_backend,
            &mut streamed_state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
            Some(&mut callback),
        )?;

        assert_eq!(plain.token_ids, streamed.token_ids);
        assert_eq!(plain.text, streamed.text);
        assert_eq!(streamed_ids, streamed.token_ids);
        Ok(())
    }

    fn logits_for_tokens(tokens: &[TokenId]) -> Vec<Vec<f32>> {
        let width = tokens.iter().copied().max().unwrap_or_default() as usize + 1;
        tokens
            .iter()
            .map(|token| {
                let mut logits = vec![f32::NEG_INFINITY; width];
                logits[*token as usize] = 0.0;
                logits
            })
            .collect()
    }

    fn tagged_json_observer(
        policy: crate::ToolCallPolicy,
    ) -> crate::pipeline::ToolCallStagedOutputObserver {
        crate::pipeline::ToolCallStagedOutputObserver::new(
            onnx_genai_metadata::ToolProtocol::from_declaration(
                &onnx_genai_metadata::ToolProtocolDeclaration {
                    identity: "tagged-json".to_string(),
                    version: "v1".to_string(),
                },
            )
            .expect("supported protocol"),
            policy,
            Default::default(),
        )
    }

    #[test]
    fn production_generation_host_waits_for_boundary_and_commits_every_adjacent_call()
    -> anyhow::Result<()> {
        let tokenizer = tool_protocol_tokenizer()?;
        let envelope = r#"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#;
        let one_call = generated_tokens(&tokenizer, envelope)?;
        let tokens = [one_call.as_slice(), one_call.as_slice()].concat();
        assert_eq!(tokenizer.decode(&tokens)?, format!("{envelope} {envelope}"));
        let options = GenerateOptions {
            max_new_tokens: tokens.len(),
            greedy: true,
            temperature: 0.0,
            stop_on_eos: false,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false)?;
        let mut backend =
            MockBackend::with_logits(false, SampledOutcome::HardError, logits_for_tokens(&tokens));
        let mut state = DecodeLoopState::new(0, options.seed, None);
        let mut observer = tagged_json_observer(crate::ToolCallPolicy::Auto);
        let mut delivered = Vec::new();
        let mut callback = |token: GenerateToken| {
            delivered.push(token.token_id);
            Ok(())
        };

        let result = crate::pipeline::generation::generate_with_decode_core_and_staged_observer(
            &canonical_runtime(),
            &mut backend,
            &mut state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
            Some(&mut observer),
            Some(&mut callback),
        )?;

        assert_eq!(result.finish_reason, FinishReason::ToolCalls);
        assert_eq!(result.token_ids, tokens);
        assert_eq!(backend.next_logits, 2);
        assert_eq!(delivered, result.token_ids);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "weather");
        assert_eq!(result.tool_calls[1].name, "weather");
        assert_eq!(result.tool_calls[0].id, "call_0");
        assert_eq!(result.tool_calls[1].id, "call_1");
        assert_eq!(result.tool_calls[0].arguments, r#"{"city":"Paris"}"#);
        Ok(())
    }

    #[test]
    fn production_generation_host_aborts_policy_failure_without_callback() -> anyhow::Result<()> {
        let tokenizer = tool_protocol_tokenizer()?;
        let tokens = generated_tokens(&tokenizer, "ordinary assistant text")?;
        assert_eq!(tokenizer.decode(&tokens)?, "ordinary assistant text");
        let options = GenerateOptions {
            max_new_tokens: tokens.len(),
            greedy: true,
            temperature: 0.0,
            stop_on_eos: false,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false)?;
        let mut backend =
            MockBackend::with_logits(false, SampledOutcome::HardError, logits_for_tokens(&tokens));
        let mut state = DecodeLoopState::new(0, options.seed, None);
        let mut observer = tagged_json_observer(crate::ToolCallPolicy::Required);
        let mut delivered = Vec::new();
        let mut callback = |token: GenerateToken| {
            delivered.push(token.token_id);
            Ok(())
        };

        let error = crate::pipeline::generation::generate_with_decode_core_and_staged_observer(
            &canonical_runtime(),
            &mut backend,
            &mut state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
            Some(&mut observer),
            Some(&mut callback),
        )
        .expect_err("required no-call output must fail before publication");
        let error = format!("{error:#}");

        assert!(error.contains("at least one was required"), "{error}");
        assert!(delivered.is_empty());
        Ok(())
    }

    #[test]
    fn tool_call_callback_failure_is_post_commit_and_not_replayed() -> anyhow::Result<()> {
        let tokenizer = tool_protocol_tokenizer()?;
        let envelope = r#"<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>"#;
        let tokens = generated_tokens(&tokenizer, envelope)?;
        let options = GenerateOptions {
            max_new_tokens: tokens.len(),
            greedy: true,
            temperature: 0.0,
            stop_on_eos: false,
            ..Default::default()
        };
        let chain = build_processor_chain(&options, None, false)?;
        let mut backend =
            MockBackend::with_logits(false, SampledOutcome::HardError, logits_for_tokens(&tokens));
        let mut state = DecodeLoopState::new(0, options.seed, None);
        let mut observer = tagged_json_observer(crate::ToolCallPolicy::Auto);
        let mut deliveries = 0;
        let mut callback = |_token: GenerateToken| {
            deliveries += 1;
            anyhow::bail!("receiver disconnected")
        };

        let error = crate::pipeline::generation::generate_with_decode_core_and_staged_observer(
            &canonical_runtime(),
            &mut backend,
            &mut state,
            &[],
            crate::pipeline::generation::GenerationRequest {
                control: None,
                options: &options,
                chain: &chain,
                tokenizer: &tokenizer,
                max_context: None,
            },
            Some(&mut observer),
            Some(&mut callback),
        )
        .expect_err("post-commit delivery failure remains visible");

        assert!(format!("{error:#}").contains("receiver disconnected"));
        assert_eq!(deliveries, 1);
        assert_eq!(backend.committed, tokens);
        assert_eq!(observer.committed_calls().len(), 1);
        Ok(())
    }
}

#[cfg(test)]
mod generation_clock_tests {
    use super::GenerationClock;
    use std::time::{Duration, Instant};
    #[test]
    fn generation_clock_attributes_prepared_prefix_work_to_its_actual_provenance() {
        use super::PreparedPrompt;
        for prefix in [
            PreparedPrompt::Reused { tokens: 3 },
            PreparedPrompt::Evaluated {
                tokens: 3,
                elapsed: Duration::from_millis(40),
            },
        ] {
            let mut clock = GenerationClock::default();
            clock.prepare_prompt(prefix);
            let initial = clock.initial_progress(7, prefix.cached_tokens());
            assert_eq!(initial.total, 7);
            assert_eq!(initial.processed, 3);
            assert_eq!(initial.cache, prefix.cached_tokens());
            assert_eq!(initial.elapsed, prefix.elapsed());
            let start = Instant::now();
            clock.begin_forward_at(start, 7, prefix.cached_tokens());
            clock.finish_forward_at(start + Duration::from_millis(10));
            clock.token_selected_at(start + Duration::from_millis(12), 1);
            let sample = clock.snapshot().unwrap();
            assert_eq!(sample.prompt_tokens, 7 - prefix.cached_tokens());
            assert_eq!(sample.cached_prompt_tokens, prefix.cached_tokens());
            assert_eq!(
                sample.prompt_elapsed,
                prefix.elapsed() + Duration::from_millis(10)
            );
            assert_eq!(sample.generation_elapsed, Duration::from_millis(2));
        }
    }

    #[test]
    fn generation_clock_excludes_delivery_and_scheduler_gaps_from_decode_work() {
        let mut clock = GenerationClock::default();
        assert_eq!(clock.snapshot(), None);
        let start = Instant::now();
        clock.begin_forward_at(start, 7, 4);
        clock.finish_forward_at(start + Duration::from_millis(10));
        clock.token_selected_at(start + Duration::from_millis(12), 1);
        let first = clock.snapshot().unwrap();
        assert_eq!(first.prompt_tokens, 3);
        assert_eq!(first.cached_prompt_tokens, 4);
        assert_eq!(first.generated_tokens, 1);
        assert_eq!(first.prompt_elapsed, Duration::from_millis(10));
        assert_eq!(first.generation_elapsed, Duration::from_millis(2));
        // Clone exactly as the prioritized drive suspends/restores its clock.
        let mut resumed = clock.clone();
        resumed.begin_forward_at(start + Duration::from_millis(100), 7, 4);
        resumed.finish_forward_at(start + Duration::from_millis(103));
        resumed.token_selected_at(start + Duration::from_millis(105), 2);
        let second = resumed.snapshot().unwrap();
        assert_eq!(second.prompt_elapsed, first.prompt_elapsed);
        assert_eq!(second.generation_elapsed, Duration::from_millis(7));
        assert_eq!(second.generated_tokens, 2);
        assert_eq!(
            clock.snapshot(),
            Some(first),
            "an earlier published snapshot is immutable"
        );
    }
}
