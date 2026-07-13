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
use onnx_genai_ort::{
    DataType, DecodeSession, DecodeSessionOptions, Session, StaticCacheDecodeOptions,
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

enum DecodeRunner {
    StaticCache(StaticCacheDecodeSession<'static>),
    PastPresent(DecodeSession<'static>),
}

impl DecodeRunner {
    fn as_backend(&mut self) -> &mut dyn DecodeBackend {
        match self {
            DecodeRunner::StaticCache(runner) => runner,
            DecodeRunner::PastPresent(runner) => runner,
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
                runner: Some(DecodeRunner::StaticCache(StaticCacheDecodeSession::new(
                    stable_session_ref(session),
                    StaticCacheDecodeOptions { batch_size: 1 },
                )?)),
            }),
            ModelDecodePath::PastPresent {
                shared_buffer,
                max_len,
            } => {
                let mut state = Self::new(session)?;
                if state.use_kv {
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

    pub(crate) fn runner_len(&self) -> usize {
        match &self.runner {
            Some(DecodeRunner::StaticCache(session)) => session.current_len(),
            Some(DecodeRunner::PastPresent(session)) => session.past_len(),
            None => 0,
        }
    }

    pub(crate) fn rewind_runner(&mut self, target_len: usize) -> anyhow::Result<()> {
        match &mut self.runner {
            Some(DecodeRunner::StaticCache(session)) => session.rewind(target_len)?,
            Some(DecodeRunner::PastPresent(session)) => session.rewind(target_len)?,
            None => {
                self.past.clear();
            }
        }
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
    let (input_tokens, past_len) = session_decode_input_tokens(state)?;
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
    let outputs = run_decode_step(session, &mut state.decode_state, &input_tokens, past_len)?;
    if state.decode_state.use_kv {
        if let Some(kv_model) = kv_model {
            mirror_present_kv_to_pages(
                session, kv_model, kv_cache, seq, &outputs, past_len, input_len,
            )?;
        } else {
            kv_cache
                .append(seq, input_len)
                .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {}", e))?;
        }
        state.kv_token_count += input_len;
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
    let (input_tokens, past_len) = session_decode_input_tokens(state)?;
    let input_len = input_tokens.len();
    let outputs = run_decode_step(session, &mut state.decode_state, &input_tokens, past_len)?;
    if state.decode_state.use_kv {
        if let Some(kv_model) = kv_model {
            mirror_present_kv_to_pages(
                session, kv_model, kv_cache, seq, &outputs, past_len, input_len,
            )?;
        } else {
            kv_cache
                .append(seq, input_len)
                .map_err(|e| anyhow::anyhow!("Failed to advance KV sequence {seq}: {}", e))?;
        }
        state.kv_token_count += input_len;
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
                past_len,
                input_len,
            )?;
        } else {
            draft_model
                .kv_cache
                .append(draft_state.seq, input_len)
                .map_err(|e| anyhow::anyhow!("Failed to advance draft KV sequence: {}", e))?;
        }
        draft_state.kv_token_count += input_len;
    }
    extract_next_token_logits(&draft_model.session, outputs)
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
    genai_config: Option<crate::genai_config::GenaiRuntimeConfig>,
) -> anyhow::Result<ModelDecodePath> {
    if let Some(signature) = StaticCacheDecodeSession::detect(session)? {
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
        // A `genai_config.json` `past_present_share_buffer` declaration (e.g. the
        // fp16 GroupQueryAttention WebGPU export) is authoritative: the model
        // owns a single max-length KV buffer on-device that must be aliased
        // present->past across steps. Its `max_length` (falling back to
        // `context_length` or inference-metadata context) pre-sizes that buffer.
        let config_share_buffer = genai_config
            .and_then(|config| config.past_present_share_buffer)
            .unwrap_or(false);
        let config_max_len = genai_config.and_then(|config| config.effective_max_length());
        if config_share_buffer {
            let max_len = config_max_len.or(metadata_max_context).ok_or_else(|| {
                anyhow::anyhow!(
                    "genai_config.json declares past_present_share_buffer but no max_length, context_length, or metadata max_sequence_length to size the shared KV buffer"
                )
            })?;
            return Ok(ModelDecodePath::PastPresent {
                shared_buffer: true,
                max_len: Some(max_len),
            });
        }

        let shared_buffer =
            session.past_present_share_buffer_supported() && metadata_max_context.is_some();
        return Ok(ModelDecodePath::PastPresent {
            shared_buffer,
            max_len: metadata_max_context.filter(|_| shared_buffer),
        });
    }

    Ok(ModelDecodePath::Legacy)
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
    let total_len = past_len + seq_len;
    let input_ids = token_ids
        .iter()
        .map(|&id| i64::from(id))
        .collect::<Vec<_>>();
    let attention_mask = vec![1_i64; total_len];
    let position_ids = (past_len..total_len)
        .map(|pos| i64::try_from(pos).context("position id exceeds i64 range"))
        .collect::<anyhow::Result<Vec<_>>>()?;

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
            let value = if past_len == 0 {
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
    if info.dtype != DataType::Float32 {
        anyhow::bail!(
            "KV input '{}' must be Float32 for Phase 1, got {:?}",
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
    Value::from_slice_f32(&[], &shape)
        .map_err(|e| anyhow::anyhow!("Failed to create empty KV input '{}': {}", info.name, e))
}

pub(crate) fn clone_value(value: &Value) -> anyhow::Result<Value> {
    match value.dtype() {
        DataType::Float32 => Value::from_slice_f32(&value.to_vec_f32()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Float32 ORT value: {}", e)),
        DataType::Int64 => Value::from_slice_i64(&value.to_vec_i64()?, value.shape())
            .map_err(|e| anyhow::anyhow!("Failed to clone Int64 ORT value: {}", e)),
        dtype => anyhow::bail!("unsupported cached ORT value dtype: {:?}", dtype),
    }
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
    use super::is_token_input_name;

    #[test]
    fn recognizes_causal_and_seq2seq_token_input_names() {
        assert!(is_token_input_name("input_ids"));
        assert!(is_token_input_name("decoder_input_ids"));
        assert!(is_token_input_name("model.input_ids"));
        assert!(is_token_input_name("model.decoder_input_ids"));
        assert!(!is_token_input_name("encoder_input_ids"));
    }
}
