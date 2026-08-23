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

/// What the runtime is able to offer for a single aliased ("shared") KV buffer.
///
/// Deployment policy, resolved by the runtime and never serialized into a
/// package: whether this execution provider consumes a fixed-capacity present
/// binding, and what sequence capacity the buffer may be sized to.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SharedKvOffer {
    /// Whether the execution provider accepts a pre-bound, fixed-capacity
    /// `present` output (the mechanism a shared KV buffer needs).
    pub(crate) present_binding_supported: bool,
    /// Sequence capacity the runtime is willing to reserve, if bounded.
    pub(crate) max_len: Option<usize>,
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
    io: Option<&onnx_genai_metadata::DecoderAbi>,
    sliding_window: Option<usize>,
    sink_tokens: usize,
    shared_kv: SharedKvOffer,
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
        // Two independent facts must agree before a single aliased KV buffer is
        // used. The PACKAGE states whether aliasing `present` onto `past` is
        // legal for its graph (`io.aliasing`); the RUNTIME states whether this
        // deployment can exploit it (execution-provider fixed-capacity present
        // binding, plus a capacity to size the buffer to). Neither may decide
        // alone: a package cannot demand a deployment, and a runtime must not
        // alias a graph that never said it was safe.
        let aliasing = io
            .and_then(|io| io.aliasing)
            .unwrap_or(onnx_genai_metadata::StateAliasing::Forbidden);
        if aliasing != onnx_genai_metadata::StateAliasing::Forbidden
            && shared_kv.present_binding_supported
            && let Some(max_len) = shared_kv.max_len
        {
            return Ok(ModelDecodePath::PastPresent {
                shared_buffer: true,
                max_len: Some(max_len),
                sliding_window: None,
                sink_tokens: None,
            });
        }
        if aliasing == onnx_genai_metadata::StateAliasing::Required {
            anyhow::bail!(
                "aliasing is 'required', but this deployment cannot alias present onto \
                 past: execution-provider fixed-capacity present binding is {}, and the resolved \
                 KV capacity is {}. Lower the requirement to 'permitted', or run on a provider \
                 that supports the shared KV buffer with a bounded max context",
                if shared_kv.present_binding_supported {
                    "available"
                } else {
                    "unavailable"
                },
                shared_kv
                    .max_len
                    .map_or_else(|| "unknown".to_owned(), |len| len.to_string()),
            );
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

#[cfg(test)]
mod aliasing_tests {
    use onnx_genai_metadata::{DecoderAbi, StateAliasing};

    use super::{ModelDecodePath, SharedKvOffer, detect_model_decode_path};

    /// A minimal past/present port contract, optionally declaring alias legality.
    fn kv_io(aliasing: Option<StateAliasing>) -> DecoderAbi {
        let mut declared = serde_json::json!({
            "kv_inputs": ["past_key_values.0.key"],
            "kv_outputs": ["present.0.key"],
        });
        if let Some(aliasing) = aliasing {
            declared["aliasing"] = serde_json::to_value(aliasing).expect("aliasing");
        }
        serde_json::from_value(declared).expect("io spec")
    }

    fn capable() -> SharedKvOffer {
        SharedKvOffer {
            present_binding_supported: true,
            max_len: Some(4096),
        }
    }

    #[test]
    fn permitted_aliasing_plus_a_capable_deployment_shares_the_buffer() {
        let path = detect_model_decode_path(
            Some(&kv_io(Some(StateAliasing::Permitted))),
            None,
            0,
            capable(),
        )
        .expect("decode path");
        assert!(matches!(
            path,
            ModelDecodePath::PastPresent {
                shared_buffer: true,
                max_len: Some(4096),
                sliding_window: None,
                sink_tokens: None,
            }
        ));
    }

    #[test]
    fn an_undeclared_package_never_gets_an_aliased_buffer() {
        // Silence means forbidden. A capable deployment must not alias a graph
        // that never stated the aliasing is safe -- doing so would corrupt KV for
        // any graph that reads `past` after writing `present`.
        let path =
            detect_model_decode_path(Some(&kv_io(None)), None, 0, capable()).expect("decode path");
        assert!(matches!(
            path,
            ModelDecodePath::PastPresent {
                shared_buffer: false,
                ..
            }
        ));
    }

    #[test]
    fn a_permitting_package_on_an_incapable_deployment_falls_back() {
        // `permitted` is permission, not a demand: both an unsupported provider
        // and an unknown capacity fall back silently to the unshared path.
        for offer in [
            SharedKvOffer {
                present_binding_supported: false,
                max_len: Some(4096),
            },
            SharedKvOffer {
                present_binding_supported: true,
                max_len: None,
            },
        ] {
            let path = detect_model_decode_path(
                Some(&kv_io(Some(StateAliasing::Permitted))),
                None,
                0,
                offer,
            )
            .expect("decode path");
            assert!(
                matches!(
                    path,
                    ModelDecodePath::PastPresent {
                        shared_buffer: false,
                        ..
                    }
                ),
                "offer {offer:?} must not alias"
            );
        }
    }

    #[test]
    fn a_requiring_package_fails_loudly_on_an_incapable_deployment() {
        // `required` means the graph is only correct with aliasing, so quietly
        // taking the unshared path would produce wrong output. Refuse instead.
        let error = detect_model_decode_path(
            Some(&kv_io(Some(StateAliasing::Required))),
            None,
            0,
            SharedKvOffer {
                present_binding_supported: false,
                max_len: Some(4096),
            },
        )
        .expect_err("required aliasing must not be silently downgraded");
        let message = error.to_string();
        assert!(message.contains("aliasing"), "{message}");
        assert!(message.contains("unavailable"), "{message}");
    }

    #[test]
    fn sliding_window_keeps_the_bounded_paged_path_regardless_of_aliasing() {
        // Windowed models evict KV, which the aliased fixed buffer cannot express.
        let path = detect_model_decode_path(
            Some(&kv_io(Some(StateAliasing::Permitted))),
            Some(2),
            0,
            capable(),
        )
        .expect("decode path");
        assert!(matches!(
            path,
            ModelDecodePath::PastPresent {
                shared_buffer: false,
                max_len: None,
                sliding_window: Some(2),
                sink_tokens: None,
            }
        ));
    }

    #[test]
    fn a_standard_attention_graph_withdraws_the_offer_even_when_the_package_permits_aliasing() {
        // Three facts must agree before a buffer is aliased, not two: the PACKAGE
        // permits it, the EP can bind a fixed-capacity present, and the graph's
        // attention OPERATOR can read a capacity-padded past. The third is the one
        // that is invisible in metadata and in EP capabilities alike, so it is
        // checked against the graph itself when the offer is built.
        //
        // This reproduces the composition `shared_kv_offer` performs: a permitting
        // package on a fully capable provider must still fall back to the exact
        // length rebind path when the graph carries the standard opset
        // `Attention`, which cross-checks `total_sequence_length` against the mask
        // and would otherwise fail at the first decode step.
        use super::graph_accepts_padded_past;
        use onnx_runtime_ir::{Attribute, Graph, Node};

        let standard_attention = {
            let mut graph = Graph::default();
            graph.nodes.insert_with(|id| {
                let mut node = Node::new(id, "Attention", vec![], vec![]);
                node.domain = String::new();
                node.attributes
                    .insert("is_causal".to_string(), Attribute::Int(1));
                node
            });
            graph
        };
        assert!(
            !graph_accepts_padded_past(&standard_attention),
            "standard opset Attention must reject a padded past"
        );

        let ep_is_capable = true;
        let offer = SharedKvOffer {
            present_binding_supported: ep_is_capable
                && graph_accepts_padded_past(&standard_attention),
            max_len: Some(4096),
        };
        let path =
            detect_model_decode_path(Some(&kv_io(Some(StateAliasing::Permitted))), None, 0, offer)
                .expect("decode path");
        assert!(
            matches!(
                path,
                ModelDecodePath::PastPresent {
                    shared_buffer: false,
                    ..
                }
            ),
            "an operator that cannot read a padded past must not be given one"
        );
    }
}
