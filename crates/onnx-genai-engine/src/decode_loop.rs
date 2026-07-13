//! Shared token-generation loop used by direct, session, priority, pipeline, and speculative paths.

use crate::config::{
    FinishReason, GenerateOptions, GenerateResult, GenerateToken, GenerateTokenCallback,
    TokenLogprob,
};
use crate::logits::{ProcessorChain, ProcessorContext, TokenId};
use crate::processors::{
    ensure_constrained_finish, finish_reason_after_token, select_next_token_with_rng,
};
use crate::sampling::SamplingRng;
use onnx_genai_ort::Tokenizer;

pub(crate) struct DecodeLoopState {
    pub(crate) generated_tokens: Vec<TokenId>,
    pub(crate) generated_text: String,
    pub(crate) step: usize,
    pub(crate) prefix_cache_hit_len: usize,
    pub(crate) logprobs: Option<Vec<TokenLogprob>>,
    pub(crate) rng: SamplingRng,
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
        }
    }
}

pub(crate) trait DecodeLoopBackend {
    fn context_len(&self) -> usize;
    fn processor_prompt_tokens(&self) -> Vec<TokenId>;
    fn next_logits(&mut self) -> anyhow::Result<Vec<f32>>;
    fn commit_token(&mut self, token_id: TokenId) -> anyhow::Result<()>;
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
        select_next_token_with_rng(&mut logits, &context, options, chain, &mut state.rng)
    };
    if let (Some(top_logprobs), Some(logprobs)) = (options.top_logprobs, state.logprobs.as_mut()) {
        logprobs.push(logprob_for_token(&logits, token_id, top_logprobs));
    }
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
