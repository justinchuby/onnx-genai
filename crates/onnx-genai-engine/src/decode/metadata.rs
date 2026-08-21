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

/// Whether the decoder graph's attention op can consume a **padded** past KV
/// buffer, i.e. one whose sequence extent is the runtime-owned capacity rather
/// than the number of valid tokens.
///
/// This is the property the share-buffer path actually requires, and it is a
/// property of the *operator*, not of the execution provider. `GroupQueryAttention`
/// and friends take an explicit valid-length input (`seqlens_k` / `total_sequence_length`)
/// and write the new step in place at that offset, so a fixed-capacity past is
/// exactly what they expect. The standard opset `Attention` has no such input:
/// it derives `total_sequence_length` from the past tensor's own sequence
/// dimension and cross-checks it against the attention mask, so a capacity-padded
/// past makes the two disagree and ORT rejects the run with
/// `inconsistent total_sequence_length (between attn_mask and past_key and past_value)`.
///
/// Returning `false` routes the model to `ZeroCopyRebind`, which feeds the past
/// at its exact logical length — correct for both operator families, at the cost
/// of the growing-KV rebind the share-buffer path avoids.
///
/// When no graph is available for inspection this returns `true`, preserving the
/// previous behaviour for callers that cannot supply one.
pub(crate) fn graph_accepts_padded_past(graph: &onnx_runtime_ir::Graph) -> bool {
    fn node_rejects_padded_past(node: &onnx_runtime_ir::Node) -> bool {
        // The standard opset `Attention` (domain "" / "ai.onnx") is the case that
        // cross-checks. The `com.microsoft` attention ops take an explicit valid
        // length and are fine with a padded past.
        if node.op_type == "Attention" && matches!(node.domain.as_str(), "" | "ai.onnx") {
            return true;
        }
        node.attributes.values().any(|attr| match attr {
            onnx_runtime_ir::Attribute::Graph(subgraph) => !graph_accepts_padded_past(subgraph),
            onnx_runtime_ir::Attribute::Graphs(subgraphs) => {
                subgraphs.iter().any(|sub| !graph_accepts_padded_past(sub))
            }
            _ => false,
        })
    }
    !graph.nodes.values().any(node_rejects_padded_past)
}

/// Whether the decoder graph positively uses an attention operator that takes an
/// **explicit valid length** and writes each step in place at that offset —
/// `GroupQueryAttention` and its relatives.
///
/// This is the affirmative counterpart to [`graph_accepts_padded_past`]. The two
/// are deliberately not inverses: the veto is permissive (anything that is not a
/// standard `Attention` is allowed through, so an unrecognised op cannot silently
/// break a model), whereas enabling the shared KV buffer needs *positive*
/// evidence, so an unrecognised op must not switch it on.
///
/// This exists because a `genai_config.json` exported for CPU carries
/// `search.past_present_share_buffer: false` — correct for CPU, but not a fact
/// about the model. Taking that declaration as the last word left CUDA runs of
/// GQA models on the growing-KV rebind path, where ORT CUDA-graph capture is
/// unreachable no matter what `enable_cuda_graph` says.
pub(crate) fn graph_uses_explicit_kv_length_attention(graph: &onnx_runtime_ir::Graph) -> bool {
    fn node_takes_explicit_kv_length(node: &onnx_runtime_ir::Node) -> bool {
        if matches!(
            node.op_type.as_str(),
            "GroupQueryAttention" | "SparseAttention" | "PagedAttention"
        ) {
            return true;
        }
        node.attributes.values().any(|attr| match attr {
            onnx_runtime_ir::Attribute::Graph(subgraph) => {
                graph_uses_explicit_kv_length_attention(subgraph)
            }
            onnx_runtime_ir::Attribute::Graphs(subgraphs) => subgraphs
                .iter()
                .any(graph_uses_explicit_kv_length_attention),
            _ => false,
        })
    }
    graph.nodes.values().any(node_takes_explicit_kv_length)
}

pub(crate) fn detect_model_decode_path(
    session: &Session,
    io: Option<&onnx_genai_metadata::ModelIoSpec>,
    metadata_max_context: Option<usize>,
    shared_kv_max_len: Option<usize>,
    sliding_window: Option<usize>,
    sliding_window_graph: Option<&onnx_runtime_ir::Graph>,
    sink_tokens: usize,
) -> anyhow::Result<ModelDecodePath> {
    // A `sliding_window` declared in `inference_metadata.yaml` is only *active*
    // when the exported decoder graph actually enforces a local-attention window
    // on its attention ops (an ORT `GroupQueryAttention` with a positive
    // `local_window_size`). If the window is declared in metadata but the graph
    // computes GLOBAL attention, the runtime cannot conjure the mask back, so the
    // model is routed as global attention rather than onto the capture-unstable
    // growing/paged KV path.
    //
    // This is a fallback, not an endorsement: a graph that drops a window its
    // architecture actually trains with is an export defect, and it silently
    // corrupts generation once a prompt outgrows the window. Muse-Glimmer-30B is
    // the known example (52 GQA ops, none carrying `local_window_size`, while the
    // architecture uses a 2048-token window on 39 of its 52 layers), so the
    // mismatch is logged loudly enough to be diagnosed.
    //
    // When no graph is supplied (best-effort inspection unavailable) the declared
    // window is kept, preserving the prior behavior for real SWA models
    // (Gemma/Mistral-style) whose graph we could not read.
    let sliding_window = effective_sliding_window(sliding_window, sliding_window_graph);

    if let Some(signature) = StaticCacheDecodeSession::detect(session, io)? {
        if sliding_window.is_some() {
            anyhow::bail!(
                "sliding-window attention is not supported by the static-cache decode path; Mobius must emit a rotating/circular static cache contract"
            );
        }
        return Ok(ModelDecodePath::StaticCache {
            max_len: signature.max_len,
        });
    }

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
        //
        // The EP predicate is necessary but not sufficient: the *operator* must
        // also accept a capacity-padded past. The standard opset `Attention`
        // does not (it derives `total_sequence_length` from the past tensor and
        // cross-checks it against the mask), so such a graph is routed to
        // `ZeroCopyRebind` regardless of what the EP or the metadata allow.
        // See [`graph_accepts_padded_past`].
        let accepts_padded_past = sliding_window_graph.is_none_or(graph_accepts_padded_past);
        if !accepts_padded_past {
            tracing::debug!(
                "decoder graph uses the standard opset Attention op, which cross-checks the \
                 attention mask against the past KV extent; using ZeroCopyRebind instead of the \
                 shared KV buffer"
            );
        }
        let supports_present_binding =
            session.supports_fixed_capacity_present_binding() && accepts_padded_past;
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

        // A model that never advertised share-buffer can still be eligible: a
        // `genai_config.json` exported for CPU carries
        // `search.past_present_share_buffer: false`, which is right for CPU and
        // says nothing about the model. Accept positive graph evidence instead —
        // an attention op that takes an explicit valid KV length is exactly the
        // operator contract the shared buffer needs. Without this, CUDA runs of
        // GQA models stayed on the growing-KV rebind path, where ORT CUDA-graph
        // capture is unreachable regardless of `enable_cuda_graph`.
        let graph_declares_explicit_kv_length =
            sliding_window_graph.is_some_and(graph_uses_explicit_kv_length_attention);
        let advertises_share_buffer =
            session.past_present_share_buffer_supported() || graph_declares_explicit_kv_length;
        let shared_buffer =
            supports_present_binding && advertises_share_buffer && metadata_max_context.is_some();
        return Ok(ModelDecodePath::PastPresent {
            shared_buffer,
            max_len: metadata_max_context.filter(|_| shared_buffer),
            sliding_window: None,
            sink_tokens: None,
        });
    }

    Ok(ModelDecodePath::Legacy)
}

/// Resolve the *effective* sliding window used for decode-path selection.
///
/// A `sliding_window` value originates from our own `inference_metadata.yaml`
/// (`model.attention.sliding_window`); it is not necessarily what the exported
/// decoder graph computes. This returns the declared window only when the graph
/// truly enforces a local-attention window (see [`graph_enforces_sliding_window`]).
/// When the window is declared in metadata but the graph computes global
/// attention, the window is dropped (`None`) so the model is not forced onto the
/// capture-unstable growing/paged KV path, and the mismatch is warned about: the
/// runtime cannot re-apply a mask the export left out. When no graph is available
/// to inspect, the declared window is kept unchanged to avoid regressing real
/// sliding-window models whose graph we could not read.
pub(crate) fn effective_sliding_window(
    declared: Option<usize>,
    graph: Option<&onnx_runtime_ir::Graph>,
) -> Option<usize> {
    let window = declared?;
    match graph {
        Some(graph) if !graph_enforces_sliding_window(graph) => {
            tracing::warn!(
                sliding_window = window,
                "inference metadata declares a sliding_window but the decoder graph carries no local-attention window (no GQA local_window_size), so attention runs globally and is routed to the shared-buffer/fixed-capacity KV path; if the architecture really is windowed this is an export defect and generation will diverge once a prompt exceeds the window"
            );
            None
        }
        _ => Some(window),
    }
}

/// Op types whose attributes can carry a trained local-attention window.
fn is_windowable_attention_op(op_type: &str) -> bool {
    matches!(
        op_type,
        "GroupQueryAttention" | "MultiHeadAttention" | "Attention" | "SparseAttention"
    )
}

/// Whether the decoder graph actually enforces a local (sliding) attention
/// window on any of its attention operators.
///
/// Graph-truth basis for SWA classification: an ORT `GroupQueryAttention` (or a
/// related attention op) enforces a window only when it carries a positive
/// `local_window_size` attribute (ORT's default is `-1`, meaning full/global
/// attention). Real sliding-window exports (Gemma/Mistral-style) set this
/// attribute; a model with a metadata-only, graph-unenforced window
/// (Muse-Glimmer-30B) does not. Control-flow subgraph bodies are traversed so a
/// window buried in an `If`/`Loop`/`Scan` body is still detected.
pub(crate) fn graph_enforces_sliding_window(graph: &onnx_runtime_ir::Graph) -> bool {
    graph.nodes.values().any(node_enforces_local_window)
}

fn node_enforces_local_window(node: &onnx_runtime_ir::Node) -> bool {
    if is_windowable_attention_op(&node.op_type)
        && node
            .attr("local_window_size")
            .and_then(|attr| attr.as_int())
            .is_some_and(|window| window > 0)
    {
        return true;
    }
    node.attributes.values().any(|attr| match attr {
        onnx_runtime_ir::Attribute::Graph(subgraph) => graph_enforces_sliding_window(subgraph),
        onnx_runtime_ir::Attribute::Graphs(subgraphs) => {
            subgraphs.iter().any(graph_enforces_sliding_window)
        }
        _ => false,
    })
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
    let declared_shared_buffer =
        model.io.as_ref().and_then(|io| io.kv_update.as_deref()) == Some("shared_buffer");
    let legacy_group_query_attention = model
        .attention
        .as_ref()
        .is_some_and(|attention| is_group_query_attention(&attention.attention_type));
    if !declared_shared_buffer && !legacy_group_query_attention {
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
pub(super) fn is_group_query_attention(attention_type: &str) -> bool {
    let normalized = attention_type.to_ascii_lowercase().replace(['-', ' '], "_");
    matches!(
        normalized.as_str(),
        "group_query"
            | "grouped_query"
            | "group_query_attention"
            | "grouped_query_attention"
            | "gqa"
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
pub(super) fn is_share_buffer_kv_dtype(dtype: &str) -> bool {
    matches!(
        dtype.to_ascii_lowercase().as_str(),
        "float16" | "fp16" | "half" | "bfloat16" | "bf16" | "float32" | "fp32" | "float"
    )
}
