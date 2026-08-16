//! Unit tests for the decode submodules.
//!
//! Pure code motion from `decode.rs`.

use super::metadata::{
    KeySequenceLengthsPolicy, decode_kv_mode_from_shared_buffer_len, effective_sliding_window,
    graph_enforces_sliding_window, is_group_query_attention, is_share_buffer_kv_dtype,
    key_sequence_lengths_policy, shared_kv_buffer_len_from_metadata, sliding_window_from_metadata,
};
use super::step::{build_position_step, decode_step_layout};
use super::values::{slice_value_axis, zero_state_value};
use onnx_genai_genai_config::GenAiConfig;
use onnx_genai_metadata::{
    AttentionConfig, InferenceMetadata, KvCacheSpec, ModelCapabilities, RuntimeConfigurable,
    RuntimeKvConfig,
};
use onnx_genai_ort::{DataType, DecodeKvMode, TensorInfo, Value};

#[test]
fn is_group_query_attention_recognizes_variants() {
    for attention_type in [
        "grouped_query",
        "group_query",
        "grouped_query_attention",
        "group_query_attention",
        "gqa",
        "Grouped-Query",
        "GROUPED QUERY",
        "group-query-attention",
        "Group Query Attention",
        "GQA",
    ] {
        assert!(
            is_group_query_attention(attention_type),
            "{attention_type:?} should be recognized as GQA"
        );
    }

    for attention_type in ["multi_head_attention", "mha", ""] {
        assert!(
            !is_group_query_attention(attention_type),
            "{attention_type:?} should not be recognized as GQA"
        );
    }
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

#[test]
fn metadata_free_multiaxis_positions_are_rejected() {
    let info = TensorInfo {
        name: "position_ids".to_string(),
        dtype: DataType::Int64,
        shape: vec![3, 1, -1],
    };
    let error = build_position_step(&info, None, None, 0, 1, &[0], &[])
        .err()
        .expect("rank-3 positions without metadata must fail");
    assert!(
        error
            .to_string()
            .contains("require pipeline.positions metadata"),
        "{error}"
    );
}

#[test]
fn fixed_state_zero_initialization_is_fallible_and_supports_half_types() {
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let info = TensorInfo {
            name: format!("{dtype:?}_state"),
            dtype,
            shape: vec![2, 3],
        };
        let value = zero_state_value(&info).expect("small half state should initialize");
        assert_eq!(value.dtype(), dtype);
        assert_eq!(value.shape(), [2, 3]);
    }

    let hostile = TensorInfo {
        name: "hostile_state".to_string(),
        dtype: DataType::Float32,
        shape: vec![i64::MAX / 4 + 1],
    };
    let error = zero_state_value(&hostile)
        .err()
        .expect("an unallocatable state must return an error");
    assert!(error.to_string().contains("failed to allocate"), "{error}");
}

#[test]
fn fixed_state_zero_initialization_resolves_symbolic_batch_axis() {
    // A hybrid recurrent state such as `conv_state [batch, 6144, 3]` exports its
    // leading batch axis as symbolic (-1). Single-sequence decode resolves it to
    // 1, mirroring the empty-KV convention, and preserves the concrete extents.
    let conv_state = TensorInfo {
        name: "past_key_values.0.conv_state".to_string(),
        dtype: DataType::Float32,
        shape: vec![-1, 6144, 3],
    };
    let value = zero_state_value(&conv_state).expect("symbolic-batch state initializes");
    assert_eq!(value.shape(), [1, 6144, 3]);

    // A symbolic NON-batch dimension cannot be zero-initialized without guessing
    // model data, so it is refused loudly.
    let inner_symbolic = TensorInfo {
        name: "past_key_values.0.recurrent_state".to_string(),
        dtype: DataType::Float32,
        shape: vec![-1, 16, -1, 128],
    };
    let error = zero_state_value(&inner_symbolic)
        .err()
        .expect("symbolic non-batch dimension must fail");
    assert!(error.to_string().contains("non-batch"), "{error}");
}

fn empty_metadata() -> InferenceMetadata {
    InferenceMetadata {
        required_capabilities: vec![],
        schema_version: None,
        model: None,
        kv_cache: None,
        quantization: None,
        preprocessing: None,
        pipeline: None,
        strategy: None,
        speculative: None,
        structured_output: None,
        hardware_requirements: None,
        generation: None,
        tokens: None,
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
        key_sequence_lengths: None,
        fallback_behavior: None,
    }
}

fn model_capabilities(
    attention: AttentionConfig,
    max_sequence_length: Option<usize>,
    runtime_configurable: Option<RuntimeConfigurable>,
) -> ModelCapabilities {
    ModelCapabilities {
        vocab_size: None,
        io: None,
        attention: Some(attention),
        max_sequence_length,
        speculative: None,
        runtime_configurable,
        mixture_of_experts: None,
    }
}

#[test]
fn shared_kv_from_gqa_fp16_native_dtype() {
    let metadata = InferenceMetadata {
        model: Some(model_capabilities(gqa_attention(), Some(4096), None)),
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
fn declared_shared_buffer_is_attention_type_agnostic() {
    let metadata: InferenceMetadata = serde_yaml::from_str(
        r#"
model:
  max_sequence_length: 1024
  attention:
    type: multi_head_attention
  io:
    kv_update: shared_buffer
kv_cache:
  native_dtype: float32
"#,
    )
    .expect("valid declarative shared-buffer metadata");
    assert_eq!(shared_kv_buffer_len_from_metadata(&metadata), Some(1024));
}

#[test]
fn key_sequence_lengths_policy_is_generic_and_strict_by_default() {
    let absent: InferenceMetadata =
        serde_yaml::from_str("model:\n  attention:\n    type: future_attention\n")
            .expect("valid attention metadata");
    assert_eq!(
        key_sequence_lengths_policy(&absent),
        KeySequenceLengthsPolicy::Canonical
    );

    let present: InferenceMetadata = serde_yaml::from_str(
        "model:\n  attention:\n    type: future_attention\n    key_sequence_lengths:\n      scalar_broadcast: unit_batch\n",
    )
    .expect("valid generalized attention metadata");
    assert_eq!(
        key_sequence_lengths_policy(&present),
        KeySequenceLengthsPolicy::UnitBatchScalar
    );
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
        decode_kv_mode_from_shared_buffer_len(shared_kv_buffer_len_from_metadata(&metadata), true,),
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
        model: Some(model_capabilities(
            gqa_attention(),
            Some(2048),
            Some(RuntimeConfigurable {
                kv_cache: Some(RuntimeKvConfig {
                    dtype: vec!["float32".to_string(), "float16".to_string()],
                }),
                prefix_cache: None,
                continuous_batching: None,
                chunked_prefill: None,
            }),
        )),
        kv_cache: None,
        ..empty_metadata()
    };
    assert_eq!(shared_kv_buffer_len_from_metadata(&metadata), Some(2048));
}

#[test]
fn no_shared_kv_when_not_gqa() {
    let metadata = InferenceMetadata {
        model: Some(model_capabilities(
            AttentionConfig {
                attention_type: "multi_head_attention".to_string(),
                ..gqa_attention()
            },
            Some(4096),
            None,
        )),
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
        model: Some(model_capabilities(gqa_attention(), Some(4096), None)),
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
        model: Some(model_capabilities(gqa_attention(), Some(4096), None)),
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
        model: Some(model_capabilities(gqa_attention(), Some(4096), None)),
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
        model: Some(model_capabilities(gqa_attention(), None, None)),
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
        model: Some(model_capabilities(attention, Some(131_072), None)),
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

/// Build a single-node decoder graph with one `GroupQueryAttention` op. When
/// `local_window_size` is `Some(w)`, the op carries that attribute (a real,
/// graph-enforced sliding window); when `None`, the op has no window attribute
/// (global attention), mirroring a vestigial metadata-only window.
fn gqa_graph(local_window_size: Option<i64>) -> onnx_runtime_ir::Graph {
    use onnx_runtime_ir::{Attribute, Graph, Node};
    let mut graph = Graph::default();
    graph.nodes.insert_with(|id| {
        let mut node = Node::new(id, "GroupQueryAttention", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        node.attributes
            .insert("num_heads".to_string(), Attribute::Int(32));
        node.attributes
            .insert("kv_num_heads".to_string(), Attribute::Int(2));
        node.attributes
            .insert("do_rotary".to_string(), Attribute::Int(1));
        if let Some(window) = local_window_size {
            node.attributes
                .insert("local_window_size".to_string(), Attribute::Int(window));
        }
        node
    });
    graph
}

#[test]
fn graph_enforces_sliding_window_detects_gqa_local_window() {
    // Real SWA export (Gemma/Mistral-style): GQA op carries local_window_size > 0.
    assert!(graph_enforces_sliding_window(&gqa_graph(Some(4096))));
}

#[test]
fn graph_without_local_window_is_global_attention() {
    // Muse-Glimmer-style: GQA ops carry NO local_window_size => global attention.
    assert!(!graph_enforces_sliding_window(&gqa_graph(None)));
    // ORT's disabled sentinel (-1) and 0 both mean "no window".
    assert!(!graph_enforces_sliding_window(&gqa_graph(Some(-1))));
    assert!(!graph_enforces_sliding_window(&gqa_graph(Some(0))));
}

#[test]
fn graph_enforces_sliding_window_recurses_into_subgraphs() {
    use onnx_runtime_ir::{Attribute, Graph, Node};
    let mut outer = Graph::default();
    outer.nodes.insert_with(|id| {
        let mut node = Node::new(id, "If", vec![], vec![]);
        node.attributes.insert(
            "then_branch".to_string(),
            Attribute::Graph(Box::new(gqa_graph(Some(2048)))),
        );
        node
    });
    assert!(graph_enforces_sliding_window(&outer));
}

#[test]
fn effective_sliding_window_drops_vestigial_metadata_window() {
    // A window declared in inference_metadata.yaml but NOT enforced by the graph
    // (Muse-Glimmer): treated as global attention so it routes to shared-buffer.
    let global_graph = gqa_graph(None);
    assert_eq!(
        effective_sliding_window(Some(2048), Some(&global_graph)),
        None
    );
}

#[test]
fn effective_sliding_window_preserves_real_swa_window() {
    // A window the graph actually enforces (real SWA) stays active => windowed path.
    let windowed_graph = gqa_graph(Some(4096));
    assert_eq!(
        effective_sliding_window(Some(4096), Some(&windowed_graph)),
        Some(4096)
    );
}

#[test]
fn effective_sliding_window_is_conservative_without_graph() {
    // If the graph cannot be inspected, keep the declared window so real SWA
    // models are never regressed onto a global-attention path.
    assert_eq!(effective_sliding_window(Some(4096), None), Some(4096));
    // No declared window is always None regardless of graph.
    assert_eq!(
        effective_sliding_window(None, Some(&gqa_graph(Some(4096)))),
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
