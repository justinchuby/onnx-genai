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
use onnx_genai_metadata::{InferenceMetadata, LoopStatePair, PositionProgram};
use onnx_genai_ort::decode_contract::name_contains_past_key_value;
use onnx_genai_ort::{
    DataType, DecodeKvMode, DecodeSession, DecodeSessionOptions, DeviceSampleParams, GraphIo,
    Session, StaticCacheDecodeOptions, StaticCacheDecodeSession, TensorInfo, Value,
};
use std::collections::{HashMap, HashSet};

mod logits;
mod metadata;
mod resolved_io;
mod state;
mod step;
mod token_sampling;
mod values;

#[cfg(test)]
mod tests;

pub(crate) use logits::{extract_logits_sequence_with_io, extract_next_token_logits_from_outputs};
#[cfg(feature = "native-backend")]
pub(crate) use metadata::{KeySequenceLengthsPolicy, key_sequence_lengths_policy};
pub(crate) use metadata::{
    detect_model_decode_path, shared_kv_buffer_len_from_metadata, sink_tokens_from_metadata,
    sliding_window_from_metadata,
};
pub(crate) use state::DecodeState;
#[cfg(feature = "native-backend")]
pub(crate) use step::position_ids_from_starts;
pub(crate) use step::{run_decode_session_logits, run_decode_step, run_decode_step_with_extra};
pub(crate) use token_sampling::{
    DraftProposalRequest, apply_paged_sliding_window, next_session_token_argmax,
    next_session_token_logits, next_session_token_logits_and_hidden,
    next_session_token_logits_and_hiddens, next_session_token_sampled, propose_draft_tokens,
};
pub(crate) use values::{clone_value, is_kv_input};

use logits::{extract_logits_value_next, extract_logits_value_sequence};

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

/// Engine-facing boundary over low-level ORT forward-pass/KV-buffer sessions.
///
/// Implementations produce logits and maintain or rewind their local KV buffer
/// cursor. Callers decide which tokens to feed, when to stop, and how logical
/// KV state participates in generation.
pub(crate) trait DecodeBackend {
    fn decode(&mut self, token_ids: &[TokenId], past_len: usize) -> anyhow::Result<Vec<Vec<f32>>>;
    /// Greedy fast path: run the decode step and return only the argmax token
    /// id of the final position, or `None` when this backend cannot select the
    /// token internally (the caller then falls back to [`Self::decode`] plus
    /// host-side sampling). Only valid when no logit processors run and greedy
    /// sampling is requested — the caller must enforce those preconditions.
    fn decode_argmax(
        &mut self,
        _token_ids: &[TokenId],
        _past_len: usize,
    ) -> anyhow::Result<Option<u32>> {
        Ok(None)
    }
    /// Whether [`Self::decode_argmax`] can select the token internally. Backends
    /// return `false` unless they support the fast path so callers can decide
    /// without triggering the step's side effects.
    fn supports_argmax(&self) -> bool {
        false
    }
    fn decode_sampled(
        &mut self,
        _token_ids: &[TokenId],
        _past_len: usize,
        _params: &DeviceSampleParams,
    ) -> anyhow::Result<Option<u32>> {
        Ok(None)
    }
    fn supports_sampled(&self) -> bool {
        false
    }
}

#[allow(clippy::large_enum_variant)]
enum DecodeRunner {
    StaticCache(StaticCacheDecodeSession<'static>),
    PastPresent(DecodeSession<'static>),
    // Kept for the planned native DecodeState runner construction path.
    #[cfg_attr(feature = "native-backend", allow(dead_code))]
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

    fn supports_argmax(&self) -> bool {
        match self {
            DecodeRunner::StaticCache(runner) => runner.supports_argmax(),
            DecodeRunner::PastPresent(runner) => runner.supports_argmax(),
            #[cfg(feature = "native-backend")]
            DecodeRunner::Native(runner) => runner.supports_argmax(),
        }
    }

    fn supports_sampled(&self) -> bool {
        match self {
            DecodeRunner::PastPresent(runner) => runner.supports_sampled(),
            _ => false,
        }
    }
}

impl DecodeBackend for DecodeSession<'static> {
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

    fn decode_argmax(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
    ) -> anyhow::Result<Option<u32>> {
        let prepare_span = onnx_genai_ort::prof_span!("engine.ort_prepare_inputs");
        let total_len = past_len + token_ids.len();
        let input_ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let attention_mask = vec![1_i64; total_len];
        let position_ids = (past_len..total_len)
            .map(|pos| i64::try_from(pos).context("position id exceeds i64 range"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        drop(prepare_span);
        let token = self.step_argmax(&input_ids, &attention_mask, &position_ids)?;
        Ok(Some(token))
    }

    fn supports_argmax(&self) -> bool {
        true
    }

    fn decode_sampled(
        &mut self,
        token_ids: &[TokenId],
        past_len: usize,
        params: &DeviceSampleParams,
    ) -> anyhow::Result<Option<u32>> {
        let total_len = past_len + token_ids.len();
        let input_ids = token_ids
            .iter()
            .map(|&id| i64::from(id))
            .collect::<Vec<_>>();
        let attention_mask = vec![1_i64; total_len];
        let position_ids = (past_len..total_len)
            .map(|pos| i64::try_from(pos).context("position id exceeds i64 range"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Some(self.step_sampled(
            &input_ids,
            &attention_mask,
            &position_ids,
            params,
        )?))
    }

    fn supports_sampled(&self) -> bool {
        self.will_sample_on_device()
    }
}

impl DecodeBackend for StaticCacheDecodeSession<'static> {
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
}
