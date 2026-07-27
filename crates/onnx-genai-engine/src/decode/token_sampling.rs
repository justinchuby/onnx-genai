//! Token selection over decode-step outputs (argmax / sampled / logits)
//! and speculative draft proposal.
//!
//! Pure code motion from `decode.rs`.

use super::logits::{extract_last_hidden, extract_next_token_logits_from_outputs};
use super::metadata::{draft_decode_input_tokens, session_decode_input_tokens};
use super::step::{
    run_decode_session_argmax, run_decode_session_logits, run_decode_session_sampled,
    run_decode_step,
};
use super::*;

/// Greedy fast-path sibling of [`next_session_token_logits`] for the optimized
/// decode runner.
///
/// Returns `Some(token)` when the shared-buffer runner selected the argmax token
/// internally (no host logits materialized), or `None` when the fast path does
/// not apply (no runner, or a runner that cannot select internally) so the
/// caller falls back to [`next_session_token_logits`] plus host sampling.
///
/// The capability check happens before any windowed-prefix consumption or KV
/// advancement, so returning `None` leaves session state untouched and safe for
/// the fallback to re-drive.
pub(crate) fn next_session_token_argmax(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
) -> anyhow::Result<Option<u32>> {
    if !state.decode_state.has_runner() || !state.decode_state.runner_supports_argmax() {
        return Ok(None);
    }
    let (mut input_tokens, mut past_len) = session_decode_input_tokens(state)?;
    consume_windowed_prefix(
        session,
        kv_model,
        kv_cache,
        seq,
        state,
        &mut input_tokens,
        &mut past_len,
    )?;
    let input_len = input_tokens.len();
    let token = run_decode_session_argmax(&mut state.decode_state, &input_tokens, past_len)?
        .context("argmax-capable decode runner returned no token")?;
    let _kv_span = onnx_genai_ort::prof_span!("engine.kv_bookkeeping");
    kv_cache
        .append(seq, input_len)
        .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {e}"))?;
    state.kv_token_count += input_len;
    Ok(Some(token))
}

/// Device-sampled fast-path sibling of [`next_session_token_logits`].
///
/// The caller falls back to host logits and sampling when this returns an error.
pub(crate) fn next_session_token_sampled(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
    params: &DeviceSampleParams,
) -> anyhow::Result<Option<u32>> {
    if !state.decode_state.has_runner() || !state.decode_state.runner_supports_sampled() {
        return Ok(None);
    }
    let (mut input_tokens, mut past_len) = session_decode_input_tokens(state)?;
    // The device sampler only applies to captured single-token decode steps. The
    // prompt-prefill (multi-token) step has no captured graph and returns host
    // logits, so signal "not applicable this step" (`Ok(None)`) *without*
    // running the model or advancing KV state — the caller re-drives via the
    // host logits path. Crucially this is not a device-sampler failure, so the
    // fast path stays armed for the single-token decode steps that follow.
    // `session_decode_input_tokens` is a pure read, so returning here leaves all
    // session state untouched for the host fallback to re-drive.
    if input_tokens.len() != 1 {
        return Ok(None);
    }
    consume_windowed_prefix(
        session,
        kv_model,
        kv_cache,
        seq,
        state,
        &mut input_tokens,
        &mut past_len,
    )?;
    let input_len = input_tokens.len();
    let token =
        run_decode_session_sampled(&mut state.decode_state, &input_tokens, past_len, params)?
            .context("sample-capable decode runner returned no token")?;
    kv_cache
        .append(seq, input_len)
        .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {e}"))?;
    state.kv_token_count += input_len;
    Ok(Some(token))
}

pub(crate) fn next_session_token_logits(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
) -> anyhow::Result<Vec<f32>> {
    let (mut input_tokens, mut past_len) = session_decode_input_tokens(state)?;
    consume_windowed_prefix(
        session,
        kv_model,
        kv_cache,
        seq,
        state,
        &mut input_tokens,
        &mut past_len,
    )?;
    let input_len = input_tokens.len();
    if state.decode_state.has_runner() {
        let logits = run_decode_session_logits(&mut state.decode_state, &input_tokens, past_len)?;
        kv_cache
            .append(seq, input_len)
            .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {e}"))?;
        state.kv_token_count += input_len;
        return logits
            .into_iter()
            .last()
            .context("decode session produced no logits");
    }
    let retained_past_len = state.decode_state.retained_kv_len(past_len);
    let outputs = run_decode_step(session, &mut state.decode_state, &input_tokens, past_len)?;
    if state.decode_state.use_kv {
        if let Some(kv_model) = kv_model {
            mirror_present_kv_to_pages(
                session,
                kv_model,
                kv_cache,
                seq,
                &outputs,
                retained_past_len,
                input_len,
            )?;
        } else {
            kv_cache
                .append(seq, input_len)
                .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {e}"))?;
        }
        state.kv_token_count += input_len;
        apply_paged_sliding_window(
            kv_cache,
            seq,
            state.decode_state.sliding_window(),
            state.decode_state.sink_tokens(),
        )?;
    }
    extract_next_token_logits_from_outputs(
        session,
        &outputs,
        state.decode_state.io.logits_output.as_deref(),
    )
}

pub(crate) fn next_session_token_logits_and_hidden(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
    hidden_output: &str,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let (logits, mut hidden) = next_session_token_logits_and_hiddens(
        session,
        kv_model,
        kv_cache,
        seq,
        state,
        &[hidden_output.to_string()],
    )?;
    Ok((
        logits,
        hidden
            .pop()
            .context("target model did not produce the requested hidden state")?,
    ))
}

pub(crate) fn next_session_token_logits_and_hiddens(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
    hidden_outputs: &[String],
) -> anyhow::Result<(Vec<f32>, Vec<Vec<f32>>)> {
    if state.decode_state.has_runner() {
        anyhow::bail!(
            "speculative hidden-state outputs {hidden_outputs:?} are not exposed by the optimized decode runner; initialize the target with the legacy output-preserving decode path"
        );
    }
    let (mut input_tokens, mut past_len) = session_decode_input_tokens(state)?;
    consume_windowed_prefix(
        session,
        kv_model,
        kv_cache,
        seq,
        state,
        &mut input_tokens,
        &mut past_len,
    )?;
    let input_len = input_tokens.len();
    let retained_past_len = state.decode_state.retained_kv_len(past_len);
    let outputs = run_decode_step(session, &mut state.decode_state, &input_tokens, past_len)?;
    if state.decode_state.use_kv {
        if let Some(kv_model) = kv_model {
            mirror_present_kv_to_pages(
                session,
                kv_model,
                kv_cache,
                seq,
                &outputs,
                retained_past_len,
                input_len,
            )?;
        } else {
            kv_cache
                .append(seq, input_len)
                .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {e}"))?;
        }
        state.kv_token_count += input_len;
        apply_paged_sliding_window(
            kv_cache,
            seq,
            state.decode_state.sliding_window(),
            state.decode_state.sink_tokens(),
        )?;
    }
    let logits = extract_next_token_logits_from_outputs(
        session,
        &outputs,
        state.decode_state.io.logits_output.as_deref(),
    )?;
    let hidden = hidden_outputs
        .iter()
        .map(|output| extract_last_hidden(session, &outputs, output))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((logits, hidden))
}

pub(crate) fn next_draft_token_logits(
    draft_model: &mut DraftModel,
    draft_state: &mut DraftSession,
) -> anyhow::Result<Vec<f32>> {
    let (input_tokens, past_len) = draft_decode_input_tokens(draft_state)?;
    let input_len = input_tokens.len();
    if draft_state.decode_state.has_runner() {
        let logits =
            run_decode_session_logits(&mut draft_state.decode_state, &input_tokens, past_len)?;
        draft_model
            .kv_cache
            .append(draft_state.seq, input_len)
            .map_err(|e| anyhow::anyhow!("Failed to advance draft KV sequence: {e}"))?;
        draft_state.kv_token_count += input_len;
        return logits
            .into_iter()
            .last()
            .context("draft decode session produced no logits");
    }
    let retained_past_len = draft_state.decode_state.retained_kv_len(past_len);
    let outputs = run_decode_step(
        &draft_model.session,
        &mut draft_state.decode_state,
        &input_tokens,
        past_len,
    )?;
    if draft_state.decode_state.use_kv {
        if let Some(kv_model) = &draft_model.kv_model {
            mirror_present_kv_to_pages(
                &draft_model.session,
                kv_model,
                &mut draft_model.kv_cache,
                draft_state.seq,
                &outputs,
                retained_past_len,
                input_len,
            )?;
        } else {
            draft_model
                .kv_cache
                .append(draft_state.seq, input_len)
                .map_err(|e| anyhow::anyhow!("Failed to advance draft KV sequence: {e}"))?;
        }
        draft_state.kv_token_count += input_len;
        apply_paged_sliding_window(
            &mut draft_model.kv_cache,
            draft_state.seq,
            draft_state.decode_state.sliding_window(),
            draft_state.decode_state.sink_tokens(),
        )?;
    }

    extract_next_token_logits_from_outputs(
        &draft_model.session,
        &outputs,
        draft_state.decode_state.io.logits_output.as_deref(),
    )
}

pub(crate) fn apply_paged_sliding_window(
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    sliding_window: Option<usize>,
    sink_tokens: usize,
) -> anyhow::Result<()> {
    if let Some(window_size) = sliding_window {
        kv_cache
            .apply_sliding_window_with_sinks(seq, window_size, sink_tokens)
            .map_err(|error| {
                anyhow::anyhow!("Failed to apply KV sliding window for sequence {seq}: {error}")
            })?;
    }
    Ok(())
}

fn consume_windowed_prefix(
    session: &Session,
    kv_model: Option<&KvModelInfo>,
    kv_cache: &mut PagedKvCache,
    seq: SessionId,
    state: &mut EngineSession,
    input_tokens: &mut Vec<TokenId>,
    past_len: &mut usize,
) -> anyhow::Result<()> {
    let Some(window_size) = state.decode_state.sliding_window() else {
        return Ok(());
    };
    let mut consumed = 0;
    while input_tokens.len() - consumed > 1 {
        let retained_past_len = state.decode_state.retained_kv_len(*past_len);
        let chunk_capacity = window_size;
        let remaining = input_tokens.len() - consumed;
        if remaining <= chunk_capacity {
            break;
        }
        let chunk_len = chunk_capacity.min(remaining - 1);
        let chunk = input_tokens[consumed..consumed + chunk_len].to_vec();
        let outputs = run_decode_step(session, &mut state.decode_state, &chunk, *past_len)?;
        if let Some(kv_model) = kv_model {
            mirror_present_kv_to_pages(
                session,
                kv_model,
                kv_cache,
                seq,
                &outputs,
                retained_past_len,
                chunk_len,
            )?;
        } else {
            kv_cache
                .append(seq, chunk_len)
                .map_err(|error| anyhow::anyhow!("Failed to advance KV sequence {seq}: {error}"))?;
        }
        state.kv_token_count += chunk_len;
        *past_len += chunk_len;
        apply_paged_sliding_window(
            kv_cache,
            seq,
            Some(window_size),
            state.decode_state.sink_tokens(),
        )?;
        consumed += chunk_len;
    }
    if consumed > 0 {
        input_tokens.drain(..consumed);
    }
    Ok(())
}

/// Bundled inputs for [`propose_draft_tokens`]: the draft model and its session,
/// the decode context (generated tokens/text and starting step), and the
/// sampling configuration (options, processor chain, and RNG) used to
/// speculatively propose a linear chain of draft tokens.
pub(crate) struct DraftProposalRequest<'a> {
    pub(crate) draft_model: &'a mut DraftModel,
    pub(crate) draft_state: &'a mut DraftSession,
    pub(crate) width: usize,
    pub(crate) generated_tokens: &'a [TokenId],
    pub(crate) generated_text: &'a str,
    pub(crate) first_step: usize,
    pub(crate) options: &'a GenerateOptions,
    pub(crate) chain: &'a ProcessorChain,
    pub(crate) rng: &'a mut SamplingRng,
}

pub(crate) fn propose_draft_tokens(
    request: DraftProposalRequest<'_>,
) -> anyhow::Result<Vec<TokenId>> {
    let DraftProposalRequest {
        draft_model,
        draft_state,
        width,
        generated_tokens,
        generated_text,
        first_step,
        options,
        chain,
        rng,
    } = request;
    let prompt_len = draft_state
        .tokens
        .len()
        .saturating_sub(generated_tokens.len());
    let mut proposed = Vec::with_capacity(width);
    let mut draft_generated = generated_tokens.to_vec();
    let mut draft_text = generated_text.to_string();

    for offset in 0..width {
        let mut logits = next_draft_token_logits(draft_model, draft_state)?;
        let context = ProcessorContext {
            prompt_tokens: draft_state.tokens[..prompt_len.min(draft_state.tokens.len())].to_vec(),
            generated_tokens: draft_generated.clone(),
            generated_text: draft_text.clone(),
            step: first_step + offset,
        };
        let token = select_next_token_with_rng(&mut logits, &context, options, chain, rng);
        proposed.push(token);
        draft_generated.push(token);
        draft_state.tokens.push(token);
        draft_text.clear();
    }

    Ok(proposed)
}
