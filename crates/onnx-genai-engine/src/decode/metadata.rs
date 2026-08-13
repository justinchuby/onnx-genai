//! Decode-path and KV-mode selection from inference metadata.
//!
//! Pure code motion from `decode.rs`.

use super::*;

#[cfg(any(test, feature = "native-backend"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KeySequenceLengthsPolicy {
    Canonical,
    UnitBatchScalar,
}

#[cfg(any(test, feature = "native-backend"))]
/// Resolve the representation contract independently of an attention op name.
pub(crate) fn key_sequence_lengths_policy(
    metadata: &InferenceMetadata,
) -> KeySequenceLengthsPolicy {
    match metadata
        .model
        .as_ref()
        .and_then(|model| model.attention.as_ref())
        .and_then(|attention| attention.key_sequence_lengths.as_ref())
        .and_then(|lengths| lengths.scalar_broadcast)
    {
        Some(onnx_genai_metadata::SequenceLengthScalarBroadcast::UnitBatch) => {
            KeySequenceLengthsPolicy::UnitBatchScalar
        }
        None => KeySequenceLengthsPolicy::Canonical,
    }
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
    io: Option<&onnx_genai_metadata::ModelIoSpec>,
    sliding_window: Option<usize>,
    sink_tokens: usize,
) -> anyhow::Result<ModelDecodePath> {
    let has_kv_inputs = io
        .and_then(|io| io.kv_inputs.as_ref())
        .is_some_and(|ports| !ports.is_empty());
    let has_present_outputs = io
        .and_then(|io| io.kv_outputs.as_ref())
        .is_some_and(|ports| !ports.is_empty());
    if has_kv_inputs || has_present_outputs {
        if sliding_window.is_some() {
            // Sliding-window models take the bounded paged past/present path
            // (`shared_buffer: false`); the graph remains responsible for
            // local-attention masking while the runtime applies windowed KV
            // eviction on the paged cache.
            // This path bounds the runtime-owned past tensors and preserves
            // absolute position_ids while the graph applies its trained window.
            return Ok(ModelDecodePath::PastPresent {
                shared_buffer: false,
                max_len: None,
                sliding_window,
                sink_tokens: (sink_tokens > 0).then_some(sink_tokens),
            });
        }
        return Ok(ModelDecodePath::PastPresent {
            shared_buffer: false,
            max_len: None,
            sliding_window: None,
            sink_tokens: None,
        });
    }

    Ok(ModelDecodePath::Generic)
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
