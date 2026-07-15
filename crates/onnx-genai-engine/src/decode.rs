//! Engine-side decode policy and ORT decode-step adapters.
//!
//! The ORT crate owns a single forward pass and its runtime KV buffers. This
//! module converts engine token context into those low-level calls and exposes
//! [`DecodeBackend`] as the seam used by engine generation policy.
//! [`ModelDecodePath`] is only the model-I/O selection enum; despite the older
//! issue wording, it is not the boundary trait. Multi-step generation, token
//! selection, stopping, constraints, and KV-management policy remain in the
//! engine.

use crate::config::{GenerateOptions, SessionId};
use crate::kv_bridge::{KvModelInfo, mirror_present_kv_to_pages};
use crate::logits::{ProcessorChain, ProcessorContext, TokenId};
use crate::processors::select_next_token_with_rng;
use crate::sampling::SamplingRng;
use crate::session::{DraftModel, DraftSession, EngineSession};
use anyhow::Context;
use onnx_genai_kv::{KvCacheOps, PagedKvCache};
use onnx_genai_metadata::InferenceMetadata;
use onnx_genai_ort::{
    DataType, DecodeKvMode, DecodeSession, DecodeSessionOptions, Session, StaticCacheDecodeOptions,
    StaticCacheDecodeSession, TensorInfo, Value,
};
use std::collections::HashMap;

#[derive(Debug, Clone)]
/// Model-I/O strategy used to construct the appropriate [`DecodeBackend`].
pub(crate) enum ModelDecodePath {
    StaticCache {
        max_len: usize,
    },
    PastPresent {
        shared_buffer: bool,
        max_len: Option<usize>,
        sliding_window: Option<usize>,
        /// Number of pinned leading attention-sink tokens (StreamingLLM), kept
        /// alongside the sliding window. `None`/`0` disables sink retention.
        sink_tokens: Option<usize>,
    },
    Legacy,
}

#[allow(dead_code)]
/// Engine-facing boundary over low-level ORT forward-pass/KV-buffer sessions.
///
/// Implementations produce logits and maintain or rewind their local KV buffer
/// cursor. Callers decide which tokens to feed, when to stop, and how logical
/// KV state participates in generation.
pub(crate) trait DecodeBackend {
    fn current_len(&self) -> usize;
    fn max_context(&self) -> Option<usize> {
        None
    }
    fn decode(&mut self, token_ids: &[TokenId], past_len: usize) -> anyhow::Result<Vec<Vec<f32>>>;
    fn rewind(&mut self, target_len: usize) -> anyhow::Result<()>;
    fn reset(&mut self) -> anyhow::Result<()> {
        self.rewind(0)
    }
}

#[allow(clippy::large_enum_variant)]
#[allow(dead_code)]
enum DecodeRunner {
    StaticCache(StaticCacheDecodeSession<'static>),
    PastPresent(DecodeSession<'static>),
    #[cfg(feature = "native-backend")]
    Native(crate::native_decode::NativeDecodeSession),
}

impl DecodeRunner {
    fn as_backend(&mut self) -> &mut dyn DecodeBackend {
        match self {
            DecodeRunner::StaticCache(runner) => runner,
            DecodeRunner::PastPresent(runner) => runner,
            #[cfg(feature = "native-backend")]
            DecodeRunner::Native(runner) => runner,
        }
    }
}

impl DecodeBackend for DecodeSession<'static> {
    fn current_len(&self) -> usize {
        self.past_len()
    }

    fn decode(&mut self, token_ids: &[TokenId], past_len: usize) -> anyhow::Result<Vec<Vec<f32>>> {
        let total_len = past_len + token_ids.len();
        let input_ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let attention_mask = vec![1_i64; total_len];
        let position_ids = (past_len..total_len)
            .map(|pos| i64::try_from(pos).context("position id exceeds i64 range"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let logits = self.step(&input_ids, &attention_mask, &position_ids)?;
        let _extract = onnx_genai_ort::prof_span!("engine.logits_to_vec");
        extract_logits_value_sequence(&logits)
    }

    fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        DecodeSession::rewind(self, target_len)?;
        Ok(())
    }
}

impl DecodeBackend for StaticCacheDecodeSession<'static> {
    fn current_len(&self) -> usize {
        StaticCacheDecodeSession::current_len(self)
    }

    fn max_context(&self) -> Option<usize> {
        Some(self.max_len())
    }

    fn decode(&mut self, token_ids: &[TokenId], _past_len: usize) -> anyhow::Result<Vec<Vec<f32>>> {
        let input_ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        if self.current_len() == 0 {
            let position_ids = (0..input_ids.len())
                .map(|pos| i64::try_from(pos).context("position id exceeds i64 range"))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let logits = self.prefill(&input_ids, &position_ids)?;
            extract_logits_value_sequence(&logits)
        } else {
            let mut logits = Vec::with_capacity(input_ids.len());
            for &token in &input_ids {
                let pos =
                    i64::try_from(self.current_len()).context("position id exceeds i64 range")?;
                let value = self.step(&[token], &[pos])?;
                logits.push(extract_logits_value_next(&value)?);
            }
            Ok(logits)
        }
    }

    fn rewind(&mut self, target_len: usize) -> anyhow::Result<()> {
        StaticCacheDecodeSession::rewind(self, target_len)?;
        Ok(())
    }
}

pub(crate) struct DecodeState {
    pub(crate) use_kv: bool,
    pub(crate) past: HashMap<String, Value>,
    pub(crate) present_to_past: HashMap<String, String>,
    pub(crate) kv_inputs: Vec<String>,
    sliding_window: Option<usize>,
    sink_tokens: usize,
    retained_kv_len: usize,
    runner: Option<DecodeRunner>,
}

impl DecodeState {
    pub(crate) fn new(session: &Session) -> anyhow::Result<Self> {
        let kv_inputs = session
            .inputs()
            .iter()
            .filter(|info| is_kv_input(&info.name))
            .map(|info| info.name.clone())
            .collect::<Vec<_>>();
        let present_outputs = session
            .outputs()
            .iter()
            .filter(|info| is_present_output(&info.name))
            .map(|info| info.name.clone())
            .collect::<Vec<_>>();

        if kv_inputs.is_empty() && present_outputs.is_empty() {
            return Ok(Self {
                use_kv: false,
                past: HashMap::new(),
                present_to_past: HashMap::new(),
                kv_inputs,
                sliding_window: None,
                sink_tokens: 0,
                retained_kv_len: 0,
                runner: None,
            });
        }

        let mut present_to_past = HashMap::new();
        for output in &present_outputs {
            if let Some(input) = matching_past_input(output, &kv_inputs) {
                present_to_past.insert(output.clone(), input.clone());
            }
        }

        if kv_inputs.is_empty()
            || present_outputs.is_empty()
            || present_to_past.len() != present_outputs.len()
        {
            anyhow::bail!(
                "model exposes incomplete KV I/O; past inputs: {:?}, present outputs: {:?}",
                kv_inputs,
                present_outputs
            );
        }

        Ok(Self {
            use_kv: true,
            past: HashMap::new(),
            present_to_past,
            kv_inputs,
            sliding_window: None,
            sink_tokens: 0,
            retained_kv_len: 0,
            runner: None,
        })
    }

    pub(crate) fn new_for_path(session: &Session, path: &ModelDecodePath) -> anyhow::Result<Self> {
        match path {
            ModelDecodePath::Legacy => Self::new(session),
            ModelDecodePath::StaticCache { .. } => Ok(Self {
                use_kv: true,
                past: HashMap::new(),
                present_to_past: HashMap::new(),
                kv_inputs: Vec::new(),
                sliding_window: None,
                sink_tokens: 0,
                retained_kv_len: 0,
                runner: Some(DecodeRunner::StaticCache(StaticCacheDecodeSession::new(
                    stable_session_ref(session),
                    StaticCacheDecodeOptions { batch_size: 1 },
                )?)),
            }),
            ModelDecodePath::PastPresent {
                shared_buffer,
                max_len,
                sliding_window,
                sink_tokens,
            } => {
                let mut state = Self::new(session)?;
                state.sliding_window = *sliding_window;
                state.sink_tokens = sink_tokens.unwrap_or(0);
                if state.use_kv && sliding_window.is_none() {
                    state.runner = Some(DecodeRunner::PastPresent(DecodeSession::new(
                        stable_session_ref(session),
                        DecodeSessionOptions {
                            batch_size: 1,
                            max_length: *max_len,
                            past_present_share_buffer: Some(*shared_buffer),
                        },
                    )?));
                }
                Ok(state)
            }
        }
    }

    pub(crate) fn has_runner(&self) -> bool {
        self.runner.is_some()
    }

    pub(crate) fn is_windowed(&self) -> bool {
        self.sliding_window.is_some()
    }

    pub(crate) fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// Number of pinned leading attention-sink tokens (0 if disabled).
    pub(crate) fn sink_tokens(&self) -> usize {
        self.sink_tokens
    }

    pub(crate) fn uses_token_prefix_cache(&self) -> bool {
        self.has_runner() || self.is_windowed()
    }

    pub(crate) fn retained_kv_len(&self, absolute_past_len: usize) -> usize {
        if self.is_windowed() {
            self.retained_kv_len
        } else {
            absolute_past_len
        }
    }

    pub(crate) fn runner_len(&self) -> usize {
        match &self.runner {
            Some(DecodeRunner::StaticCache(session)) => session.current_len(),
            Some(DecodeRunner::PastPresent(session)) => session.past_len(),
            #[cfg(feature = "native-backend")]
            Some(DecodeRunner::Native(session)) => session.current_len(),
            None => 0,
        }
    }

    pub(crate) fn rewind_runner(&mut self, target_len: usize) -> anyhow::Result<()> {
        match &mut self.runner {
            Some(DecodeRunner::StaticCache(session)) => session.rewind(target_len)?,
            Some(DecodeRunner::PastPresent(session)) => session.rewind(target_len)?,
            #[cfg(feature = "native-backend")]
            Some(DecodeRunner::Native(session)) => session.rewind(target_len)?,
            None => {
                self.past.clear();
            }
        }
        Ok(())
    }

    /// Whether this state's runner can hand off its KV cache as owned host
    /// tensors (export/import) — true only for a `PastPresent` runner in
    /// [`DecodeKvMode::ZeroCopyRebind`]. Shared-buffer / static-cache runners own
    /// fixed device buffers that are not portable across sessions, so the
    /// connector cannot extract or inject their KV.
    pub(crate) fn runner_supports_kv_handoff(&self) -> bool {
        matches!(
            &self.runner,
            Some(DecodeRunner::PastPresent(session))
                if session.mode() == DecodeKvMode::ZeroCopyRebind
        )
    }

    /// Export the runner's current KV as owned `(past_key_values.* name, Value)`
    /// pairs covering `[0, runner_len())`. Only valid when
    /// [`runner_supports_kv_handoff`](Self::runner_supports_kv_handoff) is true.
    pub(crate) fn export_runner_kv(&self) -> anyhow::Result<Vec<(String, Value)>> {
        match &self.runner {
            Some(DecodeRunner::PastPresent(session)) => Ok(session.export_kv()?),
            _ => anyhow::bail!("no ZeroCopyRebind PastPresent runner to export KV from"),
        }
    }

    /// Replace the runner's KV with `kv` (covering `len` tokens) so the next
    /// decode step continues from `len` tokens of context. Only valid when
    /// [`runner_supports_kv_handoff`](Self::runner_supports_kv_handoff) is true.
    pub(crate) fn import_runner_kv(
        &mut self,
        len: usize,
        kv: Vec<(String, Value)>,
    ) -> anyhow::Result<()> {
        match &mut self.runner {
            Some(DecodeRunner::PastPresent(session)) => {
                session.import_kv(len, kv)?;
                Ok(())
            }
            _ => anyhow::bail!("no ZeroCopyRebind PastPresent runner to import KV into"),
        }
    }

    pub(crate) fn apply_window_after_step(
        &mut self,
        session: &Session,
        _absolute_total_len: usize,
        present_len: usize,
    ) -> anyhow::Result<()> {
        let Some(window_size) = self.sliding_window else {
            return Ok(());
        };
        let sink = self.sink_tokens.min(present_len);
        let window_start = present_len.saturating_sub(window_size);
        // The sink prefix and the window cover the whole present buffer: keep it.
        if window_start <= sink {
            self.retained_kv_len = present_len;
            return Ok(());
        }
        let window_len = present_len - window_start;
        for input_name in &self.kv_inputs {
            let info = session
                .inputs()
                .iter()
                .find(|info| info.name == *input_name)
                .with_context(|| format!("missing KV input metadata for '{input_name}'"))?;
            let seq_axis = info
                .shape
                .len()
                .checked_sub(2)
                .context("KV input rank must be at least 2")?;
            let value = self
                .past
                .get(input_name)
                .with_context(|| format!("missing cached KV tensor for '{input_name}'"))?;
            let trimmed = if sink == 0 {
                slice_value_axis(value, seq_axis, window_start, window_len)?
            } else {
                // StreamingLLM: pin sink rows, then the trailing window rows.
                let head = slice_value_axis(value, seq_axis, 0, sink)?;
                let tail = slice_value_axis(value, seq_axis, window_start, window_len)?;
                concat_value_axis(&head, &tail, seq_axis)?
            };
            self.past.insert(input_name.clone(), trimmed);
        }
        self.retained_kv_len = sink + window_len;
        Ok(())
    }

    pub(crate) fn rewind_windowed(
        &mut self,
        absolute_current_len: usize,
        target_len: usize,
    ) -> anyhow::Result<()> {
        let window_size = self
            .sliding_window
            .context("windowed rewind requires sliding-window state")?;

        if self.sink_tokens == 0 {
            let retained_start = absolute_current_len.saturating_sub(self.retained_kv_len);
            if target_len < retained_start {
                anyhow::bail!(
                    "cannot rewind sliding-window KV to absolute position {target_len}; positions before {retained_start} were evicted"
                );
            }
            let target_retained_len = target_len - retained_start;
            if target_retained_len < self.retained_kv_len {
                for value in self.past.values_mut() {
                    let seq_axis = value
                        .shape()
                        .len()
                        .checked_sub(2)
                        .context("KV tensor rank must be at least 2")?;
                    *value = slice_value_axis(value, seq_axis, 0, target_retained_len)?;
                }
            }
            self.retained_kv_len = target_retained_len.min(window_size);
            return Ok(());
        }

        // Sink-aware layout: the buffer holds `sink` pinned rows followed by the
        // window rows, so the absolute retained set is `[0, sink) ∪ [ws, len)`.
        let sink = self.sink_tokens.min(self.retained_kv_len);
        let window_len = self.retained_kv_len - sink;
        let window_abs_start = absolute_current_len.saturating_sub(window_len);
        let new_retained = if target_len >= window_abs_start {
            sink + (target_len - window_abs_start)
        } else if target_len <= sink {
            target_len
        } else {
            anyhow::bail!(
                "cannot rewind sliding-window KV to absolute position {target_len}; positions in the evicted gap [{sink}, {window_abs_start}) are unavailable"
            );
        };
        if new_retained < self.retained_kv_len {
            for value in self.past.values_mut() {
                let seq_axis = value
                    .shape()
                    .len()
                    .checked_sub(2)
                    .context("KV tensor rank must be at least 2")?;
                *value = slice_value_axis(value, seq_axis, 0, new_retained)?;
            }
        }
        self.retained_kv_len = new_retained;
        Ok(())
    }
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
            .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {}", e))?;
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
                .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {}", e))?;
        }
        state.kv_token_count += input_len;
        apply_paged_sliding_window(
            kv_cache,
            seq,
            state.decode_state.sliding_window(),
            state.decode_state.sink_tokens(),
        )?;
    }
    extract_next_token_logits(session, outputs)
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
            "speculative hidden-state outputs {:?} are not exposed by the optimized decode runner; initialize the target with the legacy output-preserving decode path",
            hidden_outputs
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
                .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {}", e))?;
        }
        state.kv_token_count += input_len;
        apply_paged_sliding_window(
            kv_cache,
            seq,
            state.decode_state.sliding_window(),
            state.decode_state.sink_tokens(),
        )?;
    }
    let logits = extract_next_token_logits_from_outputs(session, &outputs)?;
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
            .map_err(|e| anyhow::anyhow!("Failed to advance draft KV sequence: {}", e))?;
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
                .map_err(|e| anyhow::anyhow!("Failed to advance draft KV sequence: {}", e))?;
        }
        draft_state.kv_token_count += input_len;
        apply_paged_sliding_window(
            &mut draft_model.kv_cache,
            draft_state.seq,
            draft_state.decode_state.sliding_window(),
            draft_state.decode_state.sink_tokens(),
        )?;
    }

    extract_next_token_logits(&draft_model.session, outputs)
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn propose_draft_tokens(
    draft_model: &mut DraftModel,
    draft_state: &mut DraftSession,
    width: usize,
    generated_tokens: &[TokenId],
    generated_text: &str,
    first_step: usize,
    options: &GenerateOptions,
    chain: &ProcessorChain,
    rng: &mut SamplingRng,
) -> anyhow::Result<Vec<TokenId>> {
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

pub(crate) fn session_decode_input_tokens(
    state: &EngineSession,
) -> anyhow::Result<(Vec<TokenId>, usize)> {
    if state.decode_state.use_kv {
        if state.kv_token_count > state.tokens.len() {
            anyhow::bail!(
                "session KV token count {} exceeds logical context length {}",
                state.kv_token_count,
                state.tokens.len()
            );
        }
        let input_tokens = state.tokens[state.kv_token_count..].to_vec();
        if input_tokens.is_empty() {
            anyhow::bail!("session decode step has no new token to feed");
        }
        Ok((input_tokens, state.kv_token_count))
    } else {
        if state.tokens.is_empty() {
            anyhow::bail!("decode step requires at least one context token");
        }
        Ok((state.tokens.clone(), 0))
    }
}

pub(crate) fn draft_decode_input_tokens(
    state: &DraftSession,
) -> anyhow::Result<(Vec<TokenId>, usize)> {
    if state.decode_state.use_kv {
        if state.kv_token_count > state.tokens.len() {
            anyhow::bail!(
                "draft KV token count {} exceeds logical context length {}",
                state.kv_token_count,
                state.tokens.len()
            );
        }
        let input_tokens = state.tokens[state.kv_token_count..].to_vec();
        if input_tokens.is_empty() {
            anyhow::bail!("draft decode step has no new token to feed");
        }
        Ok((input_tokens, state.kv_token_count))
    } else {
        if state.tokens.is_empty() {
            anyhow::bail!("draft decode step requires at least one context token");
        }
        Ok((state.tokens.clone(), 0))
    }
}

pub(crate) fn detect_model_decode_path(
    session: &Session,
    metadata_max_context: Option<usize>,
    shared_kv_max_len: Option<usize>,
    sliding_window: Option<usize>,
    sink_tokens: usize,
) -> anyhow::Result<ModelDecodePath> {
    if let Some(signature) = StaticCacheDecodeSession::detect(session)? {
        if sliding_window.is_some() {
            anyhow::bail!(
                "sliding-window attention is not supported by the static-cache decode path; Mobius must emit a rotating/circular static cache contract"
            );
        }
        return Ok(ModelDecodePath::StaticCache {
            max_len: signature.max_len,
        });
    }

    let has_kv_inputs = session.inputs().iter().any(|info| is_kv_input(&info.name));
    let has_present_outputs = session
        .outputs()
        .iter()
        .any(|info| is_present_output(&info.name));
    if has_kv_inputs || has_present_outputs {
        if sliding_window.is_some() {
            // Sliding-window models take the bounded paged past/present path
            // (`shared_buffer: false`); the graph remains responsible for
            // local-attention masking while the runtime applies windowed KV
            // eviction on the paged cache. A declared share-buffer-eligible KV
            // dtype (`shared_kv_max_len`) only enables the append-only single
            // shared buffer, which cannot express windowed eviction, so it is
            // intentionally skipped here in favor of the windowed paged path
            // rather than refused — this keeps every fp16/fp32 GQA SWA model
            // (Gemma/Mistral-style) on a supported decode path.
            if shared_kv_max_len.is_some() {
                tracing::debug!(
                    "model declares both sliding_window and a share-buffer KV dtype; using the bounded paged sliding-window path and skipping the append-only shared KV buffer"
                );
            }
            // This path bounds the runtime-owned past tensors and preserves
            // absolute position_ids while the graph applies its trained window.
            return Ok(ModelDecodePath::PastPresent {
                shared_buffer: false,
                max_len: None,
                sliding_window,
                sink_tokens: (sink_tokens > 0).then_some(sink_tokens),
            });
        }
        // Our own `InferenceMetadata` (from `inference_metadata.yaml`) can declare
        // that the runtime owns a single max-length KV buffer that is aliased
        // `present.*` -> `past_key_values.*` across decode steps (share-buffer) —
        // for example the fp16 GroupQueryAttention WebGPU export or the fp32
        // GroupQueryAttention CPU recipe. We honor that here in place of
        // onnxruntime-genai's `genai_config.json`: the GQA op computes attention
        // while the runtime manages the KV buffer itself, giving O(1)/token KV
        // instead of the growing `ZeroCopyRebind` path whose per-token cost
        // scales with context. `shared_kv_max_len` pre-sizes that buffer.
        //
        // The SharedBuffer path is taken only when the metadata requests it AND
        // the session's EP declares it can accept the runtime-owned
        // fixed-capacity present buffer as a pre-bound output. This capability
        // predicate (not `is_metal()`) is the sole gate: the Metal plugin
        // declares no such support by default, so it stays on `ZeroCopyRebind`
        // until opted in — see `Session::supports_fixed_capacity_present_binding`.
        let supports_present_binding = session.supports_fixed_capacity_present_binding();
        if let (DecodeKvMode::SharedBuffer, Some(max_len)) = (
            decode_kv_mode_from_shared_buffer_len(shared_kv_max_len, supports_present_binding),
            shared_kv_max_len,
        ) {
            return Ok(ModelDecodePath::PastPresent {
                shared_buffer: true,
                max_len: Some(max_len),
                sliding_window: None,
                sink_tokens: None,
            });
        }

        let shared_buffer = supports_present_binding
            && session.past_present_share_buffer_supported()
            && metadata_max_context.is_some();
        return Ok(ModelDecodePath::PastPresent {
            shared_buffer,
            max_len: metadata_max_context.filter(|_| shared_buffer),
            sliding_window: None,
            sink_tokens: None,
        });
    }

    Ok(ModelDecodePath::Legacy)
}

/// Sliding-window size declared by the model, if present and valid.
pub(crate) fn sliding_window_from_metadata(
    metadata: &InferenceMetadata,
) -> anyhow::Result<Option<usize>> {
    let window = metadata
        .model
        .as_ref()
        .and_then(|model| model.attention.as_ref())
        .and_then(|attention| attention.sliding_window);
    if window == Some(0) {
        anyhow::bail!("model.attention.sliding_window must be greater than zero");
    }
    Ok(window)
}

/// Number of pinned attention-sink tokens declared by the model (StreamingLLM,
/// DESIGN §40.4). Only meaningful when `sliding_window` is set; defaults to 0.
pub(crate) fn sink_tokens_from_metadata(metadata: &InferenceMetadata) -> usize {
    metadata
        .model
        .as_ref()
        .and_then(|model| model.attention.as_ref())
        .and_then(|attention| attention.sink_tokens)
        .unwrap_or(0)
}

/// Decide whether our `InferenceMetadata` requests the runtime to own a single
/// max-length device-resident KV buffer with `present.*` -> `past_key_values.*`
/// aliasing (share-buffer), returning that buffer's token capacity.
///
/// This replaces onnxruntime-genai's `genai_config.json` `past_present_share_buffer`
/// hint: we derive the same intent from the model's own inference metadata. The
/// runtime always owns/manages the KV cache; the GQA op is used only for
/// on-device attention compute. We infer runtime-owned share-buffer KV from:
///   * `model.attention.type` == group-query attention, plus
///   * a group-query-attention (GQA) `model.attention.type`, plus
///   * a share-buffer-compatible native KV dtype — float16, bfloat16, or
///     float32 — via `kv_cache.native_dtype` or
///     `model.runtime_configurable.kv_cache.dtype`, plus
///   * a declared `model.max_sequence_length` (used to pre-size the buffer).
///
/// Non-GQA / static-cache / unsupported-dtype models return `None` and keep
/// their existing decode paths unchanged. fp32 GQA (the CPU recipe) previously
/// fell through to the growing `ZeroCopyRebind` path, which reprocessed the KV
/// each step and made per-token cost scale with context; it now shares one
/// max-length buffer for O(1)/token KV, matching the fp16 GQA path.
pub(crate) fn shared_kv_buffer_len_from_metadata(metadata: &InferenceMetadata) -> Option<usize> {
    let model = metadata.model.as_ref()?;
    let attention = model.attention.as_ref()?;
    if !is_group_query_attention(&attention.attention_type) {
        return None;
    }
    if !metadata_kv_is_share_buffer_dtype(metadata) {
        return None;
    }
    model.max_sequence_length
}

/// Resolve the low-level decode KV mode from native inference metadata and the
/// session's present-binding capability.
///
/// This deliberately takes only two orthogonal inputs — the metadata's
/// share-buffer request (`shared_kv_buffer_len`) and a single semantic
/// capability bool (`supports_fixed_capacity_present_binding`) — rather than an
/// execution-provider identity. Metadata describes the model's past/present
/// aliasing contract (identical for every provider); the capability describes
/// whether the active EP can accept the runtime-owned fixed-capacity present
/// buffer as a pre-bound output. `SharedBuffer` is selected only when the
/// metadata requests it AND the session declares the capability; otherwise the
/// growing `ZeroCopyRebind` path is used. Keeping this pure keeps it testable
/// without an ORT session and keeps EP-identity knowledge out of decode logic.
pub(crate) fn decode_kv_mode_from_shared_buffer_len(
    shared_kv_buffer_len: Option<usize>,
    supports_fixed_capacity_present_binding: bool,
) -> DecodeKvMode {
    if shared_kv_buffer_len.is_some() && supports_fixed_capacity_present_binding {
        DecodeKvMode::SharedBuffer
    } else {
        DecodeKvMode::ZeroCopyRebind
    }
}

/// Whether an `attention.type` string denotes group-query attention (GQA).
fn is_group_query_attention(attention_type: &str) -> bool {
    let normalized = attention_type.to_ascii_lowercase().replace(['-', ' '], "_");
    matches!(
        normalized.as_str(),
        "group_query_attention" | "grouped_query_attention" | "gqa"
    )
}

/// Whether the model declares a share-buffer-compatible native KV cache dtype,
/// via either `kv_cache.native_dtype` or
/// `model.runtime_configurable.kv_cache.dtype`. The ORT GroupQueryAttention
/// kernel supports `past_present_share_buffer` for float16, bfloat16, and
/// float32 KV, so all three dtypes are eligible for the shared KV buffer.
fn metadata_kv_is_share_buffer_dtype(metadata: &InferenceMetadata) -> bool {
    let native = metadata
        .kv_cache
        .as_ref()
        .and_then(|kv| kv.native_dtype.as_deref())
        .is_some_and(is_share_buffer_kv_dtype);
    let runtime = metadata
        .model
        .as_ref()
        .and_then(|model| model.runtime_configurable.as_ref())
        .and_then(|runtime| runtime.kv_cache.as_ref())
        .is_some_and(|kv| kv.dtype.iter().any(|dtype| is_share_buffer_kv_dtype(dtype)));
    native || runtime
}

/// Whether a dtype string denotes a KV dtype the share-buffer GQA path supports
/// (16- or 32-bit floating point).
fn is_share_buffer_kv_dtype(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_lowercase().as_str(),
        "float16" | "fp16" | "half" | "bfloat16" | "bf16" | "float32" | "fp32" | "float"
    )
}

fn stable_session_ref(session: &Session) -> &'static Session {
    // SAFETY: This lifetime extension is sound only because the referenced
    // `Session` is owned by a `Box<Session>` stored in `Engine.session` or
    // `DraftModel.session`, while all `DecodeRunner`s that receive the returned
    // reference stay inside `EngineSession`s owned by the same `Engine` (or are
    // short-lived locals under `&mut Engine`). `Engine.sessions` is declared
    // before `_environment`, `session`, and `draft`, so persistent runners are
    // dropped before the boxed sessions and ORT environment; moving `Engine` does
    // not move the boxed allocation. This would become unsound if runners escaped
    // their owning `Engine`, were sent to background tasks, or if field/drop order
    // changed so the target/draft sessions could be dropped before sessions.
    unsafe { std::mem::transmute::<&Session, &'static Session>(session) }
}

pub(crate) fn run_decode_session_logits(
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    if token_ids.is_empty() {
        anyhow::bail!("decode session step requires at least one input token");
    }
    let current_len = decode_state.runner_len();
    if current_len > past_len {
        decode_state.rewind_runner(past_len)?;
    } else if current_len < past_len {
        anyhow::bail!(
            "decode session cursor {} is behind requested past length {}; replay is required",
            current_len,
            past_len
        );
    }

    decode_state
        .runner
        .as_mut()
        .context("decode session runner not initialized")?
        .as_backend()
        .decode(token_ids, past_len)
    .map_err(|error| {
        let message = error.to_string();
        if is_gather_out_of_bounds(&message) {
            anyhow::anyhow!(
                "model context length exceeded during ORT decode; configure inference metadata `model.max_sequence_length` or GenerateOptions::max_context to stop cleanly before the context window is exceeded: {}",
                error
            )
        } else {
            error
        }
    })
}

pub(crate) fn run_decode_step(
    session: &Session,
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
) -> anyhow::Result<Vec<Value>> {
    run_decode_step_with_extra(session, decode_state, token_ids, past_len, &[])
}

pub(crate) fn run_decode_step_with_extra(
    session: &Session,
    decode_state: &mut DecodeState,
    token_ids: &[TokenId],
    past_len: usize,
    extra_inputs: &[(String, Value)],
) -> anyhow::Result<Vec<Value>> {
    if token_ids.is_empty() {
        anyhow::bail!("decode step requires at least one input token");
    }

    let seq_len = token_ids.len();
    let retained_past_len = decode_state.retained_kv_len(past_len);
    let (total_len, position_ids) = decode_step_layout(past_len, retained_past_len, seq_len)?;
    let input_ids = token_ids
        .iter()
        .map(|&id| i64::from(id))
        .collect::<Vec<_>>();
    let attention_mask = vec![1_i64; total_len];

    let mut owned_inputs: Vec<(String, Value)> = Vec::new();
    for info in session.inputs() {
        let lower = info.name.to_ascii_lowercase();
        if is_token_input_name(&lower) {
            ensure_i64(info)?;
            owned_inputs.push((
                info.name.clone(),
                Value::from_slice_i64(&input_ids, &[1, seq_len as i64])?,
            ));
        } else if lower == "attention_mask" || lower.ends_with(".attention_mask") {
            ensure_i64(info)?;
            owned_inputs.push((
                info.name.clone(),
                Value::from_slice_i64(&attention_mask, &[1, total_len as i64])?,
            ));
        } else if lower == "position_ids" || lower.ends_with(".position_ids") {
            ensure_i64(info)?;
            owned_inputs.push((
                info.name.clone(),
                Value::from_slice_i64(&position_ids, &[1, seq_len as i64])?,
            ));
        } else if decode_state.use_kv && decode_state.kv_inputs.contains(&info.name) {
            let value = if retained_past_len == 0 {
                empty_past_value(info)?
            } else {
                clone_value(decode_state.past.get(&info.name).with_context(|| {
                    format!("missing cached KV tensor for input '{}'", info.name)
                })?)?
            };
            owned_inputs.push((info.name.clone(), value));
        } else if let Some((_, value)) = extra_inputs.iter().find(|(name, _)| name == &info.name) {
            owned_inputs.push((info.name.clone(), clone_value(value)?));
        } else {
            anyhow::bail!(
                "unsupported model input '{}' with shape {:?}; supported inputs are input_ids, attention_mask, position_ids, past key-values, and pipeline-routed extra inputs",
                info.name,
                info.shape
            );
        }
    }

    let input_refs = owned_inputs
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<Vec<_>>();
    let outputs = session.run(&input_refs).map_err(|e| {
        let message = e.to_string();
        if is_gather_out_of_bounds(&message) {
            anyhow::anyhow!(
                "model context length exceeded during ORT decode; configure inference metadata `model.max_sequence_length` or GenerateOptions::max_context to stop cleanly before the context window is exceeded: {}",
                e
            )
        } else {
            anyhow::anyhow!("ORT session run failed: {}", e)
        }
    })?;

    if decode_state.use_kv {
        decode_state.past.clear();
        for (name, value) in session.output_names().iter().zip(outputs.iter()) {
            if let Some(past_name) = decode_state.present_to_past.get(name) {
                decode_state
                    .past
                    .insert(past_name.clone(), clone_value(value)?);
            }
        }
        decode_state.apply_window_after_step(session, past_len + seq_len, total_len)?;
    }

    Ok(outputs)
}

pub(crate) fn extract_next_token_logits(
    session: &Session,
    outputs: Vec<Value>,
) -> anyhow::Result<Vec<f32>> {
    extract_next_token_logits_from_outputs(session, &outputs)
}

fn extract_next_token_logits_from_outputs(
    session: &Session,
    outputs: &[Value],
) -> anyhow::Result<Vec<f32>> {
    let logits_index = session
        .output_names()
        .iter()
        .position(|name| name == "logits")
        .or_else(|| {
            session
                .output_names()
                .iter()
                .position(|name| name.to_ascii_lowercase().contains("logits"))
        })
        .context("model did not expose a logits output")?;
    let logits = outputs
        .get(logits_index)
        .context("logits output index was out of range")?;
    let shape = logits.shape();
    let data = logits
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {}", e))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(data),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            let start = (*seq as usize - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }

        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            let start = (*seq as usize - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {:?}", other),
    }
}

fn extract_last_hidden(
    session: &Session,
    outputs: &[Value],
    output_name: &str,
) -> anyhow::Result<Vec<f32>> {
    let index = session
        .output_names()
        .iter()
        .position(|name| name == output_name)
        .with_context(|| {
            format!("target model did not expose hidden-state output '{output_name}'")
        })?;
    let value = outputs
        .get(index)
        .context("hidden-state output index was out of range")?;
    let shape = value.shape();
    let data = value
        .to_vec_f32_lossy()
        .map_err(|error| anyhow::anyhow!("Failed to read target hidden-state tensor: {error}"))?;
    match shape {
        [hidden] if *hidden > 0 => Ok(data),
        [seq, hidden] if *seq > 0 && *hidden > 0 => {
            let hidden = *hidden as usize;
            let start = (*seq as usize - 1) * hidden;
            Ok(data[start..start + hidden].to_vec())
        }
        [batch, seq, hidden] if *batch == 1 && *seq > 0 && *hidden > 0 => {
            let hidden = *hidden as usize;
            let start = (*seq as usize - 1) * hidden;
            Ok(data[start..start + hidden].to_vec())
        }
        other => anyhow::bail!(
            "unsupported target hidden-state tensor shape for '{output_name}': {:?}",
            other
        ),
    }
}

pub(crate) fn extract_logits_sequence(
    session: &Session,
    outputs: Vec<Value>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let logits_index = session
        .output_names()
        .iter()
        .position(|name| name == "logits")
        .or_else(|| {
            session
                .output_names()
                .iter()
                .position(|name| name.to_ascii_lowercase().contains("logits"))
        })
        .context("model did not expose a logits output")?;
    let logits = outputs
        .get(logits_index)
        .context("logits output index was out of range")?;
    let shape = logits.shape();
    let data = logits
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {}", e))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(vec![data]),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {:?}", other),
    }
}

fn extract_logits_value_next(logits: &Value) -> anyhow::Result<Vec<f32>> {
    let sequence = extract_logits_value_sequence(logits)?;
    sequence
        .into_iter()
        .last()
        .context("logits tensor did not contain any sequence rows")
}

fn decode_step_layout(
    absolute_past_len: usize,
    retained_past_len: usize,
    input_len: usize,
) -> anyhow::Result<(usize, Vec<i64>)> {
    let attended_len = retained_past_len
        .checked_add(input_len)
        .context("attention length overflow")?;
    let absolute_total_len = absolute_past_len
        .checked_add(input_len)
        .context("absolute position overflow")?;
    let position_ids = (absolute_past_len..absolute_total_len)
        .map(|position| i64::try_from(position).context("position id exceeds i64 range"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((attended_len, position_ids))
}

fn extract_logits_value_sequence(logits: &Value) -> anyhow::Result<Vec<Vec<f32>>> {
    let shape = logits.shape();
    let data = logits
        .to_vec_f32_lossy()
        .map_err(|e| anyhow::anyhow!("Failed to read logits tensor: {}", e))?;

    match shape {
        [vocab] if *vocab > 0 => Ok(vec![data]),
        [seq, vocab] if *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        [batch, seq, vocab] if *batch > 0 && *seq > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            Ok(data
                .chunks(vocab)
                .take(*seq as usize)
                .map(|chunk| chunk.to_vec())
                .collect())
        }
        other => anyhow::bail!("unsupported logits tensor shape: {:?}", other),
    }
}

fn ensure_i64(info: &TensorInfo) -> anyhow::Result<()> {
    if info.dtype != DataType::Int64 {
        anyhow::bail!("input '{}' must be Int64, got {:?}", info.name, info.dtype);
    }
    Ok(())
}

fn is_token_input_name(lower_name: &str) -> bool {
    lower_name == "input_ids"
        || lower_name == "decoder_input_ids"
        || lower_name.ends_with(".input_ids")
        || lower_name.ends_with(".decoder_input_ids")
}

fn empty_past_value(info: &TensorInfo) -> anyhow::Result<Value> {
    if !matches!(
        info.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        anyhow::bail!(
            "KV input '{}' must be Float32, Float16, or BFloat16, got {:?}",
            info.name,
            info.dtype
        );
    }
    if info.shape.len() < 3 {
        anyhow::bail!(
            "KV input '{}' has unsupported shape {:?}",
            info.name,
            info.shape
        );
    }
    let seq_axis = info.shape.len() - 2;
    let mut shape = Vec::with_capacity(info.shape.len());
    for (axis, &dim) in info.shape.iter().enumerate() {
        let value = if axis == 0 {
            1
        } else if axis == seq_axis {
            0
        } else if dim > 0 {
            dim
        } else {
            anyhow::bail!(
                "cannot infer static dimension {} for empty KV input '{}' shape {:?}",
                axis,
                info.name,
                info.shape
            );
        };
        shape.push(value);
    }
    match info.dtype {
        DataType::Float32 => Value::from_slice_f32(&[], &shape),
        DataType::Float16 => Value::from_vec_f16_bits(Vec::new(), &shape),
        DataType::BFloat16 => Value::from_vec_bf16_bits(Vec::new(), &shape),
        _ => unreachable!("dtype checked above"),
    }
    .map_err(|e| anyhow::anyhow!("Failed to create empty KV input '{}': {}", info.name, e))
}

pub(crate) fn clone_value(value: &Value) -> anyhow::Result<Value> {
    match value.dtype() {
        DataType::Float32 => Value::from_slice_f32(&value.to_vec_f32()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Float32 ORT value: {}", e)),
        DataType::Float16 => Value::from_vec_f16_bits(value.to_vec_f16_bits()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Float16 ORT value: {}", e)),
        DataType::BFloat16 => Value::from_vec_bf16_bits(value.to_vec_bf16_bits()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone BFloat16 ORT value: {}", e)),
        DataType::Int64 => Value::from_slice_i64(&value.to_vec_i64()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Int64 ORT value: {}", e)),
        dtype => anyhow::bail!("unsupported cached ORT value dtype: {:?}", dtype),
    }
}

fn slice_value_axis(value: &Value, axis: usize, start: usize, len: usize) -> anyhow::Result<Value> {
    let shape = value.shape();
    let axis_len = *shape.get(axis).context("KV slice axis is out of bounds")?;
    let axis_len = usize::try_from(axis_len).context("KV slice axis length is negative")?;
    if start > axis_len || len > axis_len - start {
        anyhow::bail!(
            "KV slice [{start}..{}) exceeds axis length {axis_len}",
            start + len
        );
    }
    let mut output_shape = shape.to_vec();
    output_shape[axis] = i64::try_from(len).context("KV slice length exceeds i64")?;

    fn copy_axis_slice<T: Copy>(
        input: &[T],
        shape: &[i64],
        axis: usize,
        start: usize,
        len: usize,
    ) -> Vec<T> {
        let inner = shape[axis + 1..]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let outer = shape[..axis]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let axis_len = shape[axis] as usize;
        let mut output = Vec::with_capacity(outer * len * inner);
        for outer_idx in 0..outer {
            let base = outer_idx * axis_len * inner + start * inner;
            output.extend_from_slice(&input[base..base + len * inner]);
        }
        output
    }

    match value.dtype() {
        DataType::Float32 => Value::from_vec_f32(
            copy_axis_slice(&value.to_vec_f32()?, shape, axis, start, len),
            &output_shape,
        ),
        DataType::Float16 => Value::from_vec_f16_bits(
            copy_axis_slice(&value.to_vec_f16_bits()?, shape, axis, start, len),
            &output_shape,
        ),
        DataType::BFloat16 => Value::from_vec_bf16_bits(
            copy_axis_slice(&value.to_vec_bf16_bits()?, shape, axis, start, len),
            &output_shape,
        ),
        dtype => anyhow::bail!("cannot slice cached KV tensor with dtype {dtype:?}"),
    }
    .map_err(|error| anyhow::anyhow!("Failed to slice cached KV tensor: {error}"))
}

/// Concatenate two KV tensors of identical shape (except along `axis`) into a
/// single tensor along that axis. Used to splice the pinned attention-sink rows
/// in front of the sliding-window rows.
fn concat_value_axis(first: &Value, second: &Value, axis: usize) -> anyhow::Result<Value> {
    let first_shape = first.shape();
    let second_shape = second.shape();
    if first_shape.len() != second_shape.len() {
        anyhow::bail!("cannot concatenate KV tensors of differing rank");
    }
    for (dim, (a, b)) in first_shape.iter().zip(second_shape.iter()).enumerate() {
        if dim != axis && a != b {
            anyhow::bail!("cannot concatenate KV tensors: mismatched shape on axis {dim}");
        }
    }
    let mut output_shape = first_shape.to_vec();
    output_shape[axis] = first_shape[axis] + second_shape[axis];

    fn interleave<T: Copy>(
        first: &[T],
        second: &[T],
        shape_a: &[i64],
        shape_b: &[i64],
        axis: usize,
    ) -> Vec<T> {
        let inner = shape_a[axis + 1..]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let outer = shape_a[..axis]
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>();
        let a_axis = shape_a[axis] as usize;
        let b_axis = shape_b[axis] as usize;
        let mut output = Vec::with_capacity(outer * (a_axis + b_axis) * inner);
        for outer_idx in 0..outer {
            let a_base = outer_idx * a_axis * inner;
            output.extend_from_slice(&first[a_base..a_base + a_axis * inner]);
            let b_base = outer_idx * b_axis * inner;
            output.extend_from_slice(&second[b_base..b_base + b_axis * inner]);
        }
        output
    }

    if first.dtype() != second.dtype() {
        anyhow::bail!("cannot concatenate KV tensors of differing dtype");
    }
    match first.dtype() {
        DataType::Float32 => Value::from_vec_f32(
            interleave(
                &first.to_vec_f32()?,
                &second.to_vec_f32()?,
                first_shape,
                second_shape,
                axis,
            ),
            &output_shape,
        ),
        DataType::Float16 => Value::from_vec_f16_bits(
            interleave(
                &first.to_vec_f16_bits()?,
                &second.to_vec_f16_bits()?,
                first_shape,
                second_shape,
                axis,
            ),
            &output_shape,
        ),
        DataType::BFloat16 => Value::from_vec_bf16_bits(
            interleave(
                &first.to_vec_bf16_bits()?,
                &second.to_vec_bf16_bits()?,
                first_shape,
                second_shape,
                axis,
            ),
            &output_shape,
        ),
        dtype => anyhow::bail!("cannot concatenate cached KV tensor with dtype {dtype:?}"),
    }
    .map_err(|error| anyhow::anyhow!("Failed to concatenate cached KV tensor: {error}"))
}

pub(crate) fn is_kv_input(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("past") && (lower.contains("key") || lower.contains("value"))
}

pub(crate) fn is_present_output(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("present") && (lower.contains("key") || lower.contains("value"))
}

pub(crate) fn matching_past_input<'a>(
    present_name: &str,
    inputs: &'a [String],
) -> Option<&'a String> {
    let present_suffix = kv_suffix(present_name)?;
    inputs
        .iter()
        .find(|input| kv_suffix(input).as_deref() == Some(present_suffix.as_str()))
}

fn kv_suffix(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    for prefix in [
        "past_key_values.",
        "present_key_values.",
        "past.",
        "present.",
    ] {
        if let Some(suffix) = lower.strip_prefix(prefix) {
            return Some(suffix.to_string());
        }
    }
    None
}

pub(crate) fn is_gather_out_of_bounds(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("gather")
        && (lower.contains("indices element out of data bounds")
            || lower.contains("idx=") && lower.contains("out of"))
}

#[cfg(test)]
mod tests {
    use super::{
        decode_kv_mode_from_shared_buffer_len, decode_step_layout, is_group_query_attention,
        is_share_buffer_kv_dtype, is_token_input_name, shared_kv_buffer_len_from_metadata,
        slice_value_axis, sliding_window_from_metadata,
    };
    use onnx_genai_genai_config::GenAiConfig;
    use onnx_genai_metadata::{
        AttentionConfig, InferenceMetadata, KvCacheSpec, ModelCapabilities, RuntimeConfigurable,
        RuntimeKvConfig,
    };
    use onnx_genai_ort::{DecodeKvMode, Value};

    #[test]
    fn recognizes_causal_and_seq2seq_token_input_names() {
        assert!(is_token_input_name("input_ids"));
        assert!(is_token_input_name("decoder_input_ids"));
        assert!(is_token_input_name("model.input_ids"));
        assert!(is_token_input_name("model.decoder_input_ids"));
        assert!(!is_token_input_name("encoder_input_ids"));
    }

    #[test]
    fn recognizes_group_query_attention_variants() {
        assert!(is_group_query_attention("group_query_attention"));
        assert!(is_group_query_attention("group-query-attention"));
        assert!(is_group_query_attention("Group Query Attention"));
        assert!(is_group_query_attention("GQA"));
        assert!(is_group_query_attention("grouped_query_attention"));
        assert!(!is_group_query_attention("multi_head_attention"));
        assert!(!is_group_query_attention("attention"));
    }

    #[test]
    fn recognizes_share_buffer_kv_dtype_variants() {
        assert!(is_share_buffer_kv_dtype("float16"));
        assert!(is_share_buffer_kv_dtype("FP16"));
        assert!(is_share_buffer_kv_dtype("half"));
        assert!(is_share_buffer_kv_dtype("float32"));
        assert!(is_share_buffer_kv_dtype("FP32"));
        assert!(is_share_buffer_kv_dtype("float"));
        assert!(is_share_buffer_kv_dtype("bfloat16"));
        assert!(is_share_buffer_kv_dtype("BF16"));
        assert!(!is_share_buffer_kv_dtype("int8"));
    }

    fn empty_metadata() -> InferenceMetadata {
        InferenceMetadata {
            required_capabilities: vec![],
            model: None,
            kv_cache: None,
            quantization: None,
            pipeline: None,
            strategy: None,
            speculative: None,
            structured_output: None,
            hardware_requirements: None,
        }
    }

    fn gqa_attention() -> AttentionConfig {
        AttentionConfig {
            attention_type: "group_query_attention".to_string(),
            num_kv_heads: Some(2),
            num_attention_heads: Some(14),
            head_dim: Some(64),
            sliding_window: None,
            sink_tokens: None,
            fallback_behavior: None,
        }
    }

    #[test]
    fn shared_kv_from_gqa_fp16_native_dtype() {
        let metadata = InferenceMetadata {
            model: Some(ModelCapabilities {
                attention: Some(gqa_attention()),
                max_sequence_length: Some(4096),
                speculative: None,
                runtime_configurable: None,
            }),
            kv_cache: Some(KvCacheSpec {
                native_dtype: Some("float16".to_string()),
                quantization_tolerance: None,
                sensitive_layers: None,
                operations: None,
            }),
            ..empty_metadata()
        };
        assert_eq!(shared_kv_buffer_len_from_metadata(&metadata), Some(4096));
    }

    #[test]
    fn genai_share_buffer_metadata_resolves_shared_mode_for_mlx_without_ep_gate() {
        let config: GenAiConfig = serde_json::from_str(
            r#"{
                "model": {
                    "context_length": 4096,
                    "decoder": {
                        "head_size": 64,
                        "num_attention_heads": 14,
                        "num_key_value_heads": 2,
                        "num_hidden_layers": 24
                    }
                },
                "search": { "past_present_share_buffer": true }
            }"#,
        )
        .expect("valid share-buffer genai_config");
        let metadata = config
            .to_inference_metadata(Some("float16"))
            .expect("share-buffer metadata");

        // The metadata contract is provider-independent: given a capable session
        // (CPU/CUDA/WebGPU, or an opted-in Metal), this share-buffer metadata
        // resolves to the SharedBuffer mode.
        assert_eq!(
            decode_kv_mode_from_shared_buffer_len(
                shared_kv_buffer_len_from_metadata(&metadata),
                true,
            ),
            DecodeKvMode::SharedBuffer
        );
    }

    #[test]
    fn decode_kv_mode_gates_shared_buffer_on_present_binding_capability() {
        // Share-buffer requested by metadata (Some(max_len)).
        let requested = Some(4096usize);
        // Metadata does NOT request share-buffer.
        let not_requested: Option<usize> = None;

        // Capable session (CPU/CUDA/WebGPU, or opted-in Metal) ⇒ SharedBuffer.
        assert_eq!(
            decode_kv_mode_from_shared_buffer_len(requested, true),
            DecodeKvMode::SharedBuffer
        );

        // Metal-without-opt-in (capability FALSE) ⇒ ZeroCopyRebind, even though
        // the metadata requested the shared buffer. This preserves today's Metal
        // behavior and keeps `is_metal()` out of decode logic.
        assert_eq!(
            decode_kv_mode_from_shared_buffer_len(requested, false),
            DecodeKvMode::ZeroCopyRebind
        );

        // No share-buffer request ⇒ ZeroCopyRebind regardless of capability.
        assert_eq!(
            decode_kv_mode_from_shared_buffer_len(not_requested, true),
            DecodeKvMode::ZeroCopyRebind
        );
        assert_eq!(
            decode_kv_mode_from_shared_buffer_len(not_requested, false),
            DecodeKvMode::ZeroCopyRebind
        );
    }

    #[test]
    fn shared_kv_from_gqa_fp16_runtime_configurable_dtype() {
        let metadata = InferenceMetadata {
            model: Some(ModelCapabilities {
                attention: Some(gqa_attention()),
                max_sequence_length: Some(2048),
                speculative: None,
                runtime_configurable: Some(RuntimeConfigurable {
                    kv_cache: Some(RuntimeKvConfig {
                        dtype: vec!["float32".to_string(), "float16".to_string()],
                    }),
                    prefix_cache: None,
                    continuous_batching: None,
                    chunked_prefill: None,
                }),
            }),
            kv_cache: None,
            ..empty_metadata()
        };
        assert_eq!(shared_kv_buffer_len_from_metadata(&metadata), Some(2048));
    }

    #[test]
    fn no_shared_kv_when_not_gqa() {
        let metadata = InferenceMetadata {
            model: Some(ModelCapabilities {
                attention: Some(AttentionConfig {
                    attention_type: "multi_head_attention".to_string(),
                    ..gqa_attention()
                }),
                max_sequence_length: Some(4096),
                speculative: None,
                runtime_configurable: None,
            }),
            kv_cache: Some(KvCacheSpec {
                native_dtype: Some("float16".to_string()),
                quantization_tolerance: None,
                sensitive_layers: None,
                operations: None,
            }),
            ..empty_metadata()
        };
        assert_eq!(shared_kv_buffer_len_from_metadata(&metadata), None);
    }

    #[test]
    fn shared_kv_from_gqa_fp32_native_dtype() {
        // The CPU recipe declares fp32 GQA KV; it must take the shared-buffer
        // path (O(1)/token) rather than the growing ZeroCopyRebind path.
        let metadata = InferenceMetadata {
            model: Some(ModelCapabilities {
                attention: Some(gqa_attention()),
                max_sequence_length: Some(4096),
                speculative: None,
                runtime_configurable: None,
            }),
            kv_cache: Some(KvCacheSpec {
                native_dtype: Some("float32".to_string()),
                quantization_tolerance: None,
                sensitive_layers: None,
                operations: None,
            }),
            ..empty_metadata()
        };
        assert_eq!(shared_kv_buffer_len_from_metadata(&metadata), Some(4096));
    }

    #[test]
    fn shared_kv_from_gqa_bf16_native_dtype() {
        let metadata = InferenceMetadata {
            model: Some(ModelCapabilities {
                attention: Some(gqa_attention()),
                max_sequence_length: Some(4096),
                speculative: None,
                runtime_configurable: None,
            }),
            kv_cache: Some(KvCacheSpec {
                native_dtype: Some("bfloat16".to_string()),
                quantization_tolerance: None,
                sensitive_layers: None,
                operations: None,
            }),
            ..empty_metadata()
        };
        assert_eq!(shared_kv_buffer_len_from_metadata(&metadata), Some(4096));
    }

    #[test]
    fn no_shared_kv_when_unsupported_kv_dtype() {
        let metadata = InferenceMetadata {
            model: Some(ModelCapabilities {
                attention: Some(gqa_attention()),
                max_sequence_length: Some(4096),
                speculative: None,
                runtime_configurable: None,
            }),
            kv_cache: Some(KvCacheSpec {
                native_dtype: Some("int8".to_string()),
                quantization_tolerance: None,
                sensitive_layers: None,
                operations: None,
            }),
            ..empty_metadata()
        };
        assert_eq!(shared_kv_buffer_len_from_metadata(&metadata), None);
    }

    #[test]
    fn no_shared_kv_when_max_sequence_length_absent() {
        let metadata = InferenceMetadata {
            model: Some(ModelCapabilities {
                attention: Some(gqa_attention()),
                max_sequence_length: None,
                speculative: None,
                runtime_configurable: None,
            }),
            kv_cache: Some(KvCacheSpec {
                native_dtype: Some("float16".to_string()),
                quantization_tolerance: None,
                sensitive_layers: None,
                operations: None,
            }),
            ..empty_metadata()
        };
        assert_eq!(shared_kv_buffer_len_from_metadata(&metadata), None);
    }

    #[test]
    fn no_shared_kv_when_metadata_empty() {
        assert_eq!(shared_kv_buffer_len_from_metadata(&empty_metadata()), None);
    }

    #[test]
    fn sliding_window_metadata_is_consumed_and_validated() {
        let mut attention = gqa_attention();
        attention.sliding_window = Some(4096);
        let metadata = InferenceMetadata {
            model: Some(ModelCapabilities {
                attention: Some(attention),
                max_sequence_length: Some(131_072),
                speculative: None,
                runtime_configurable: None,
            }),
            ..empty_metadata()
        };
        assert_eq!(sliding_window_from_metadata(&metadata).unwrap(), Some(4096));

        let mut invalid = metadata.clone();
        invalid
            .model
            .as_mut()
            .unwrap()
            .attention
            .as_mut()
            .unwrap()
            .sliding_window = Some(0);
        assert!(sliding_window_from_metadata(&invalid).is_err());
        assert_eq!(
            sliding_window_from_metadata(&empty_metadata()).unwrap(),
            None
        );
    }

    #[test]
    fn windowed_layout_keeps_absolute_positions_with_bounded_attention_length() {
        let (attended_len, position_ids) = decode_step_layout(10_000, 4096, 3).unwrap();
        assert_eq!(attended_len, 4099);
        assert_eq!(position_ids, vec![10_000, 10_001, 10_002]);

        let (full_len, full_positions) = decode_step_layout(7, 7, 2).unwrap();
        assert_eq!(full_len, 9);
        assert_eq!(full_positions, vec![7, 8]);
    }

    #[test]
    fn kv_axis_slicing_keeps_requested_suffix_in_order() {
        let value = Value::from_vec_f32(
            vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
            &[1, 1, 5, 2],
        )
        .unwrap();
        let suffix = slice_value_axis(&value, 2, 2, 3).unwrap();

        assert_eq!(suffix.shape(), &[1, 1, 3, 2]);
        assert_eq!(
            suffix.to_vec_f32().unwrap(),
            vec![20.0, 21.0, 30.0, 31.0, 40.0, 41.0]
        );
    }
}
