//! Unit tests for the decode submodules.
//!
//! Pure code motion from `decode.rs`.

use super::metadata::{
    KeySequenceLengthsPolicy, graph_accepts_padded_past, graph_uses_explicit_kv_length_attention,
    key_sequence_lengths_policy, sliding_window_from_metadata,
};
use super::step::{build_position_step, decode_step_layout};
use super::values::{slice_value_axis, zero_state_value};
use onnx_genai_metadata::{
    AttentionConfig, InferenceMetadata, ModelCapabilities, RuntimeConfigurable,
};
use onnx_genai_ort::{DataType, TensorInfo, Value};

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
    InferenceMetadata::default()
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
        attention: Some(attention),
        max_sequence_length,
        runtime_configurable,
        sharding: None,
        mixture_of_experts: None,
    }
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
fn sliding_window_metadata_is_consumed_and_validated() {
    let mut attention = gqa_attention();
    attention.sliding_window = Some(4096);
    let mut metadata = empty_metadata();
    metadata.model = Some(model_capabilities(attention, Some(131_072), None));
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

/// Build a single-node decoder graph with one `GroupQueryAttention` op. When
/// `local_window_size` is `Some(w)`, the op carries that attribute; when `None`,
/// the op has no window attribute (global attention).
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

/// A decoder graph whose attention is the **standard opset** `Attention`
/// (domain `""`), the operator that derives `total_sequence_length` from the
/// past KV tensor and cross-checks it against the attention mask.
fn standard_attention_graph(domain: &str) -> onnx_runtime_ir::Graph {
    use onnx_runtime_ir::{Attribute, Graph, Node};
    let mut graph = Graph::default();
    graph.nodes.insert_with(|id| {
        let mut node = Node::new(id, "Attention", vec![], vec![]);
        node.domain = domain.to_string();
        node.attributes
            .insert("is_causal".to_string(), Attribute::Int(1));
        node.attributes
            .insert("kv_num_heads".to_string(), Attribute::Int(8));
        node.attributes
            .insert("q_num_heads".to_string(), Attribute::Int(16));
        node
    });
    graph
}

#[test]
fn standard_attention_graph_rejects_a_capacity_padded_past() {
    // The share-buffer path binds `past_key_values.*` at the runtime-owned
    // capacity, not at the valid length. The standard opset `Attention` has no
    // valid-length input, so ORT computes total_sequence_length from that padded
    // extent and rejects the run when it disagrees with the mask:
    //   "inconsistent total_sequence_length (between attn_mask and past_key ...)"
    // Both spellings of the default domain must be caught.
    assert!(!graph_accepts_padded_past(&standard_attention_graph("")));
    assert!(!graph_accepts_padded_past(&standard_attention_graph(
        "ai.onnx"
    )));
}

#[test]
fn group_query_attention_graph_accepts_a_capacity_padded_past() {
    // GQA takes an explicit valid length (`seqlens_k`) and writes the new step in
    // place at that offset, so a fixed-capacity past is exactly its contract.
    // This is the case the share-buffer path was designed for and must keep.
    assert!(graph_accepts_padded_past(&gqa_graph(None)));
    assert!(graph_accepts_padded_past(&gqa_graph(Some(4096))));
}

#[test]
fn a_single_standard_attention_node_disqualifies_a_mixed_graph() {
    // One cross-checking op is enough to break the whole decode step, so the
    // predicate must be "all ops accept" rather than "any op accepts".
    use onnx_runtime_ir::{Attribute, Node};
    let mut graph = gqa_graph(None);
    graph.nodes.insert_with(|id| {
        let mut node = Node::new(id, "Attention", vec![], vec![]);
        node.domain = String::new();
        node.attributes
            .insert("is_causal".to_string(), Attribute::Int(1));
        node
    });
    assert!(!graph_accepts_padded_past(&graph));
}

#[test]
fn gqa_graph_is_positive_evidence_for_the_shared_kv_buffer() {
    // A `genai_config.json` exported for CPU carries
    // `search.past_present_share_buffer: false`, which is correct for CPU and is
    // not a fact about the model. Positive graph evidence — an attention op that
    // takes an explicit valid KV length — is what actually qualifies the shared
    // buffer, so a GQA graph must count even when nothing advertised it.
    assert!(graph_uses_explicit_kv_length_attention(&gqa_graph(None)));
    assert!(graph_uses_explicit_kv_length_attention(&gqa_graph(Some(
        4096
    ))));
}

#[test]
fn standard_attention_graph_is_not_evidence_for_the_shared_kv_buffer() {
    // The standard opset `Attention` derives the KV extent from the past tensor
    // itself, so a capacity-padded past is exactly what it cannot take.
    assert!(!graph_uses_explicit_kv_length_attention(
        &standard_attention_graph("")
    ));
    assert!(!graph_uses_explicit_kv_length_attention(
        &standard_attention_graph("ai.onnx")
    ));
}

#[test]
fn the_share_buffer_predicates_are_not_inverses_of_each_other() {
    // Deliberately asymmetric: the veto is permissive so an unrecognised op
    // cannot silently break a model, while enabling needs positive evidence so
    // an unrecognised op cannot silently switch the shared buffer on. A graph of
    // ops we do not classify must therefore be allowed by one and rejected by
    // the other.
    use onnx_runtime_ir::{Graph, Node};
    let mut graph = Graph::default();
    graph
        .nodes
        .insert_with(|id| Node::new(id, "MatMul", vec![], vec![]));
    assert!(
        graph_accepts_padded_past(&graph),
        "an unrecognised op must not trip the veto"
    );
    assert!(
        !graph_uses_explicit_kv_length_attention(&graph),
        "an unrecognised op must not count as positive evidence"
    );
}
