use super::*;
#[cfg(feature = "cuda")]
use onnx_genai_metadata::LoopStatePair;
use onnx_genai_metadata::{KvOwnership, ModelIoSpec, SequenceInputKind};
use onnx_runtime_ir::{Attribute, Graph, Node, NodeId, Shape, SymbolId, TensorData};
use prost::Message;
use std::collections::BTreeMap;

#[cfg(feature = "cuda")]
fn qwen_cuda_smoke_model_dir() -> Option<std::path::PathBuf> {
    let model_dir = std::env::var_os("ONNX_GENAI_QWEN_CUDA_SMOKE_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/home/justinchu/qwen2.5-0.5b-int4-onnx"));
    if !model_dir.join("model.onnx").is_file() {
        eprintln!(
            "skipping CUDA smoke; target model is not installed (set ONNX_GENAI_QWEN_CUDA_SMOKE_MODEL to its directory)"
        );
        return None;
    }
    Some(model_dir)
}

#[test]
fn tensor_argmax_reads_only_the_final_logits_row_and_keeps_first_tie() {
    let tensor = Tensor::from_f32(&[1, 2, 4], &[100.0, 0.0, 0.0, 0.0, 1.0, 7.0, 7.0, 2.0]).unwrap();
    assert_eq!(argmax_logits_tensor(&tensor).unwrap(), 1);
}

#[test]
fn tensor_argmax_matches_across_supported_logits_dtypes() {
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        let tensor = tensor_from_f32_as(dtype, &[1, 4], &[-2.0, 3.0, 1.0, 0.0]).unwrap();
        assert_eq!(argmax_logits_tensor(&tensor).unwrap(), 1, "{dtype:?}");
    }
}

#[test]
fn tensor_argmax_rejects_non_finite_logits_like_full_extraction() {
    let tensor = Tensor::from_f32(&[1, 3], &[0.0, f32::NAN, 1.0]).unwrap();
    let error = argmax_logits_tensor(&tensor).unwrap_err().to_string();
    assert!(error.contains("non-finite logits"), "{error}");
}

#[test]
fn graph_capture_auto_enables_for_owned_cuda_kv() {
    let structural = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Owned,
    };
    assert!(structural.is_capture_safe());
    // Env unset, no programmatic override, offload disabled -> auto-enable from
    // structure.
    assert!(resolve_graph_capture_enabled(
        None, false, false, structural, false
    ));
}

#[test]
fn weight_offload_forces_graph_capture_off() {
    let safe = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Owned,
    };
    // Offload wins over every other signal: auto-safe structure, an explicit
    // env=1, and an explicit programmatic request all still resolve to OFF.
    assert!(!resolve_graph_capture_enabled(
        None, false, false, safe, true
    ));
    assert!(!resolve_graph_capture_enabled(None, true, true, safe, true));
    assert!(!resolve_graph_capture_enabled(
        Some(true),
        true,
        true,
        safe,
        true
    ));
    // Sanity: with offload disabled the same safe structure enables capture, so
    // the exclusion above is genuinely caused by offload.
    assert!(resolve_graph_capture_enabled(
        None, false, false, safe, false
    ));
}

#[test]
fn cuda_kv_capacity_uses_metadata_as_default_fact() {
    let capacity = resolve_cuda_kv_capacity(
        None,
        None,
        Some(4096),
        10,
        Some(CudaDeviceMemorySnapshot {
            free_bytes: 20_000,
            total_bytes: 40_000,
        }),
    )
    .unwrap();
    assert_eq!(capacity.max_len, 4096);
    assert_eq!(capacity.source, "model.max_sequence_length");
}

#[test]
fn cuda_kv_capacity_without_metadata_is_unbounded_until_growth_fails() {
    let capacity = resolve_cuda_kv_capacity(
        None,
        None,
        None,
        28_680,
        Some(CudaDeviceMemorySnapshot {
            free_bytes: 5_925_502_976,
            total_bytes: 8_585_281_536,
        }),
    )
    .unwrap();
    assert_eq!(capacity.max_len, usize::MAX);
    assert_eq!(
        capacity.source,
        "unbounded (model.max_sequence_length unavailable)"
    );
}

#[test]
fn cuda_kv_capacity_honors_env_before_metadata() {
    let capacity = resolve_cuda_kv_capacity(
        None,
        Some(8192),
        Some(16_384),
        10,
        Some(CudaDeviceMemorySnapshot {
            free_bytes: 20_000,
            total_bytes: 40_000,
        }),
    )
    .unwrap();
    assert_eq!(capacity.max_len, 8192);
    assert_eq!(capacity.source, "ONNX_GENAI_CUDA_KV_MAX_LEN");
}

#[test]
fn cuda_kv_capacity_metadata_caps_oversized_explicit_override() {
    let capacity = resolve_cuda_kv_capacity(None, Some(8192), Some(4096), 10, None).unwrap();
    assert_eq!(capacity.max_len, 4096);
    assert!(
        capacity
            .source
            .contains("ONNX_GENAI_CUDA_KV_MAX_LEN clamped by model.max_sequence_length"),
        "{}",
        capacity.source
    );
}

#[test]
fn cuda_kv_capacity_error_explains_source_and_device_memory() {
    let capacity = CudaKvCapacity {
        max_len: 4096,
        source: "model.max_sequence_length".to_owned(),
        metadata_max_len: Some(4096),
        device_memory: Some(CudaDeviceMemorySnapshot {
            free_bytes: 20_000,
            total_bytes: 40_000,
        }),
        bytes_per_token: 10,
    };
    let message = cuda_kv_capacity_exceeded_message(4097, &capacity);
    assert!(
        message.contains("requested context length 4097"),
        "{message}"
    );
    assert!(message.contains("configured max_len 4096"), "{message}");
    assert!(
        message.contains("source: model.max_sequence_length"),
        "{message}"
    );
    assert!(
        message.contains("model.max_sequence_length: 4096"),
        "{message}"
    );
    assert!(message.contains("CUDA free=20000 bytes"), "{message}");
    assert!(message.contains("ONNX_GENAI_CUDA_KV_MAX_LEN"), "{message}");
}

#[test]
fn graph_capture_auto_declines_for_non_owned_or_non_cuda() {
    let shared = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Shared,
    };
    assert!(!shared.is_capture_safe());
    assert!(!resolve_graph_capture_enabled(
        None, false, false, shared, false
    ));

    let cpu_owned = GraphCaptureStructuralSafety {
        device_is_cuda: false,
        kv_ownership: KvOwnership::Owned,
    };
    assert!(!cpu_owned.is_capture_safe());
    assert!(!resolve_graph_capture_enabled(
        None, false, false, cpu_owned, false
    ));
}

#[test]
fn graph_capture_env_explicit_overrides_auto_decision() {
    let safe = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Owned,
    };
    let unsafe_structural = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Shared,
    };
    // ONNX_GENAI_CUDA_GRAPH=0 forces OFF even when structurally safe.
    assert!(!resolve_graph_capture_enabled(
        None, true, false, safe, false
    ));
    // ONNX_GENAI_CUDA_GRAPH=1 forces ON even when structure would decline
    // (the runtime decline machinery is still the final safety net).
    assert!(resolve_graph_capture_enabled(
        None,
        true,
        true,
        unsafe_structural,
        false
    ));
}

#[test]
fn graph_capture_programmatic_override_wins_over_env_and_structure() {
    let safe = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Owned,
    };
    // Programmatic Some(false) beats explicit env=1 and safe structure.
    assert!(!resolve_graph_capture_enabled(
        Some(false),
        true,
        true,
        safe,
        false
    ));
    // Programmatic Some(true) beats explicit env=0.
    assert!(resolve_graph_capture_enabled(
        Some(true),
        true,
        false,
        safe,
        false
    ));
}

#[test]
fn capture_fallback_emits_each_structured_decline_to_tracer() {
    let report = CaptureDeclineReport {
        entries: vec![onnx_runtime_session::CaptureDecline {
            node_id: Some(12),
            op_type: "GroupQueryAttention".to_string(),
            domain: "com.microsoft".to_string(),
            reason: "requires warmed f32 q_seq==1 k_seq==1 fixed-capacity device-KV reference path"
                .to_string(),
            seam_reason: Some(onnx_runtime_session::SeamReason::KernelCaptureUnsupported),
        }],
    };
    let (trace, events) = TraceContext::in_memory();

    trace_capture_declines(&trace, &report);

    let events = events.events();
    assert_eq!(events.len(), 1);
    let args = events[0].args.as_ref().unwrap();
    assert_eq!(args[onnx_runtime_tracer::ARG_CAPTURE_REJECTED_NODE], 12);
    assert_eq!(
        args[onnx_runtime_tracer::ARG_CAPTURE_REJECTED_OP],
        "GroupQueryAttention"
    );
    assert_eq!(
        args[onnx_runtime_tracer::ARG_CAPTURE_REJECTED_REASON],
        report.entries[0].reason
    );
}

#[test]
fn graph_uses_decode_pool_detects_quantized_ops_including_subgraphs() {
    let f32_vec =
        |graph: &mut Graph| graph.create_value(DataType::Float32, vec![1.into(), 8.into()]);

    // Dense-f32 graph (only MatMul) gains nothing from the SPMD decode pool.
    let mut dense = Graph::new();
    let a = f32_vec(&mut dense);
    let b = dense.create_value(DataType::Float32, vec![8.into(), 8.into()]);
    let out = f32_vec(&mut dense);
    insert_op(&mut dense, "MatMul", vec![a, b], out, &[]);
    assert!(!graph_uses_decode_pool(&dense));

    // Quantized graph (MatMulNBits) dispatches through the SPMD pool.
    let mut quant = Graph::new();
    let qa = f32_vec(&mut quant);
    let qw = quant.create_value(DataType::Uint8, vec![8.into(), 2.into()]);
    let qout = f32_vec(&mut quant);
    insert_op(&mut quant, "MatMulNBits", vec![qa, qw], qout, &[]);
    assert!(graph_uses_decode_pool(&quant));

    // A quantized op nested in an `If` subgraph is still detected.
    let mut branch = Graph::new();
    let ba = f32_vec(&mut branch);
    let bw = branch.create_value(DataType::Uint8, vec![8.into(), 2.into()]);
    let bout = f32_vec(&mut branch);
    insert_op(&mut branch, "MatMulNBits", vec![ba, bw], bout, &[]);
    let mut outer = Graph::new();
    let cond = outer.create_value(DataType::Bool, vec![1.into()]);
    let if_out = f32_vec(&mut outer);
    insert_op(
        &mut outer,
        "If",
        vec![cond],
        if_out,
        &[("then_branch", Attribute::Graph(Box::new(branch)))],
    );
    assert!(graph_uses_decode_pool(&outer));
}

fn insert_op(
    graph: &mut Graph,
    op_type: &str,
    inputs: Vec<onnx_runtime_ir::ValueId>,
    output: onnx_runtime_ir::ValueId,
    attributes: &[(&str, Attribute)],
) {
    let mut node = Node::new(
        NodeId(0),
        op_type,
        inputs.into_iter().map(Some).collect(),
        vec![output],
    );
    for (name, value) in attributes {
        node.attributes.insert((*name).to_string(), value.clone());
    }
    graph.insert_node(node);
}

fn tiny_decoder(last_token_logits: bool) -> InferenceSession {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 11);
    let batch = graph.intern_symbol("batch");
    let sequence = graph.intern_symbol("sequence");
    let total = graph.intern_symbol("total");
    let past = graph.intern_symbol("past");
    let shape = |dims: &[Dim]| -> Shape { dims.to_vec() };

    let input_ids = graph.create_named_value(
        "input_ids",
        DataType::Int64,
        shape(&[batch.into(), sequence.into()]),
    );
    let attention_mask = graph.create_named_value(
        "attention_mask",
        DataType::Int64,
        shape(&[batch.into(), total.into()]),
    );
    let position_ids = graph.create_named_value(
        "position_ids",
        DataType::Int64,
        shape(&[batch.into(), sequence.into()]),
    );
    let past_key = graph.create_named_value(
        "past_key_values.0.key",
        DataType::Float32,
        shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
    );
    let past_value = graph.create_named_value(
        "past_key_values.0.value",
        DataType::Float32,
        shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
    );
    for input in [
        input_ids,
        attention_mask,
        position_ids,
        past_key,
        past_value,
    ] {
        graph.add_input(input);
    }

    let cast = graph.create_value(DataType::Float32, shape(&[batch.into(), sequence.into()]));
    insert_op(
        &mut graph,
        "Cast",
        vec![input_ids],
        cast,
        &[("to", Attribute::Int(1))],
    );
    let current_kv = graph.create_value(
        DataType::Float32,
        shape(&[batch.into(), 1.into(), sequence.into(), 1.into()]),
    );
    insert_op(
        &mut graph,
        "Unsqueeze",
        vec![cast],
        current_kv,
        &[("axes", Attribute::Ints(vec![1, 3]))],
    );

    let logits = if last_token_logits {
        let logits = graph.create_named_value(
            "logits",
            DataType::Float32,
            shape(&[1.into(), 1.into(), 2.into()]),
        );
        let data = [10.0f32, 20.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        insert_op(
            &mut graph,
            "Constant",
            vec![],
            logits,
            &[(
                "value",
                Attribute::Tensor(TensorData::from_raw(DataType::Float32, vec![1, 1, 2], data)),
            )],
        );
        logits
    } else {
        let logits = graph.create_named_value(
            "logits",
            DataType::Float32,
            shape(&[batch.into(), sequence.into(), 1.into()]),
        );
        insert_op(
            &mut graph,
            "Unsqueeze",
            vec![cast],
            logits,
            &[("axes", Attribute::Ints(vec![2]))],
        );
        logits
    };
    let present_key = graph.create_named_value(
        "present.0.key",
        DataType::Float32,
        shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
    );
    insert_op(
        &mut graph,
        "Concat",
        vec![past_key, current_kv],
        present_key,
        &[("axis", Attribute::Int(2))],
    );
    let present_value = graph.create_named_value(
        "present.0.value",
        DataType::Float32,
        shape(&[batch.into(), 1.into(), total.into(), 1.into()]),
    );
    insert_op(
        &mut graph,
        "Concat",
        vec![past_value, current_kv],
        present_value,
        &[("axis", Attribute::Int(2))],
    );
    for output in [logits, present_key, present_value] {
        graph.add_output(output);
    }
    InferenceSession::from_graph(graph).expect("build tiny decoder")
}

fn target_io(sequence_source: SequenceInputKind) -> ModelIoSpec {
    ModelIoSpec {
        sequence_source: Some(sequence_source),
        kv_ownership: Some(KvOwnership::Owned),
        token_input: (sequence_source == SequenceInputKind::TokenIds).then(|| "input_ids".into()),
        inputs_embeds_input: (sequence_source == SequenceInputKind::InputsEmbeds)
            .then(|| "embedded_sequence".into()),
        attention_mask_input: Some("attention_mask".into()),
        position_ids_input: None,
        logits_output: Some("logits".into()),
        hidden_output: None,
        kv_inputs: Some(vec!["cache_key".into()]),
        kv_outputs: Some(vec!["next_key".into()]),
        encoder_hidden_states_input: None,
        audio_features_input: None,
        cross_kv_inputs: None,
        cross_kv_outputs: None,
        kv_update: None,
        state_pairs: None,
        optional_inputs: BTreeMap::new(),
        static_cache: None,
    }
}

fn tiny_decoder_io() -> ModelIoSpec {
    ModelIoSpec {
        sequence_source: Some(SequenceInputKind::TokenIds),
        kv_ownership: Some(KvOwnership::Owned),
        token_input: Some("input_ids".into()),
        inputs_embeds_input: None,
        attention_mask_input: Some("attention_mask".into()),
        position_ids_input: Some("position_ids".into()),
        logits_output: Some("logits".into()),
        hidden_output: None,
        kv_inputs: Some(vec![
            "past_key_values.0.key".into(),
            "past_key_values.0.value".into(),
        ]),
        kv_outputs: Some(vec!["present.0.key".into(), "present.0.value".into()]),
        encoder_hidden_states_input: None,
        audio_features_input: None,
        cross_kv_inputs: None,
        cross_kv_outputs: None,
        kv_update: None,
        state_pairs: None,
        optional_inputs: BTreeMap::new(),
        static_cache: None,
    }
}

#[test]
fn native_decoder_requires_explicit_ambiguous_io() {
    let error = match NativeDecodeSession::from_session(tiny_decoder(false)) {
        Ok(_) => panic!("ambiguous decoder roles must require metadata"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("model.io.token_input"),
        "{error:#}"
    );
}

fn tiny_embedding_target(with_routed_input: bool) -> InferenceSession {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 11);
    let batch = graph.intern_symbol("batch");
    let sequence = graph.intern_symbol("sequence");
    let total = graph.intern_symbol("total");
    let past = graph.intern_symbol("past");
    let embedded_sequence = graph.create_named_value(
        "embedded_sequence",
        DataType::Float32,
        vec![batch.into(), sequence.into(), 1.into()],
    );
    let attention_mask = graph.create_named_value(
        "attention_mask",
        DataType::Int64,
        vec![batch.into(), total.into()],
    );
    let routed = with_routed_input.then(|| {
        graph.create_named_value(
            "routed_features",
            DataType::Float32,
            vec![batch.into(), sequence.into(), 1.into()],
        )
    });
    let cache_key = graph.create_named_value(
        "cache_key",
        DataType::Float32,
        vec![batch.into(), 1.into(), past.into(), 1.into()],
    );
    graph.add_input(embedded_sequence);
    graph.add_input(attention_mask);
    if let Some(routed) = routed {
        graph.add_input(routed);
    }
    graph.add_input(cache_key);

    let logits = graph.create_named_value(
        "logits",
        DataType::Float32,
        vec![batch.into(), sequence.into(), 1.into()],
    );
    if let Some(routed) = routed {
        insert_op(
            &mut graph,
            "Add",
            vec![embedded_sequence, routed],
            logits,
            &[],
        );
    } else {
        insert_op(&mut graph, "Identity", vec![embedded_sequence], logits, &[]);
    }
    let next_key = graph.create_named_value(
        "next_key",
        DataType::Float32,
        vec![1.into(), 1.into(), 2.into(), 1.into()],
    );
    insert_op(
        &mut graph,
        "Constant",
        vec![],
        next_key,
        &[(
            "value",
            Attribute::Tensor(TensorData::from_raw(
                DataType::Float32,
                vec![1, 1, 2, 1],
                vec![0; 2 * std::mem::size_of::<f32>()],
            )),
        )],
    );
    for output in [logits, next_key] {
        graph.add_output(output);
    }
    session_from_graph(graph)
}

fn tiny_step_producer() -> InferenceSession {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let input = graph.create_named_value(
        "producer_input",
        DataType::Float32,
        vec![1.into(), 2.into(), 1.into()],
    );
    let output = graph.create_named_value(
        "producer_output",
        DataType::Float32,
        vec![1.into(), 2.into(), 1.into()],
    );
    graph.add_input(input);
    insert_op(&mut graph, "Identity", vec![input], output, &[]);
    graph.add_output(output);
    session_from_graph(graph)
}

fn session_from_graph(graph: Graph) -> InferenceSession {
    let bytes = onnx_std::Model::new(graph)
        .to_proto()
        .expect("serialize ONNX model")
        .encode_to_vec();
    InferenceSession::builder()
        .model_bytes(&bytes)
        .build()
        .expect("load ONNX model")
}

fn proposer_io(sequence_source: SequenceInputKind, kv_ownership: KvOwnership) -> ModelIoSpec {
    ModelIoSpec {
        sequence_source: Some(sequence_source),
        kv_ownership: Some(kv_ownership),
        token_input: (sequence_source == SequenceInputKind::TokenIds).then(|| "input_ids".into()),
        inputs_embeds_input: (sequence_source == SequenceInputKind::InputsEmbeds)
            .then(|| "embeddings".into()),
        attention_mask_input: Some("mask".into()),
        position_ids_input: Some("positions".into()),
        logits_output: Some("draft_scores".into()),
        hidden_output: (sequence_source == SequenceInputKind::InputsEmbeds)
            .then(|| "next_state".into()),
        kv_inputs: (kv_ownership == KvOwnership::Owned)
            .then(|| vec!["cache_key".into(), "cache_value".into()]),
        kv_outputs: (kv_ownership == KvOwnership::Owned)
            .then(|| vec!["next_key".into(), "next_value".into()]),
        encoder_hidden_states_input: None,
        audio_features_input: None,
        cross_kv_inputs: None,
        cross_kv_outputs: None,
        kv_update: None,
        state_pairs: None,
        optional_inputs: std::collections::BTreeMap::new(),
        static_cache: None,
    }
}

fn tiny_owned_kv_proposer() -> InferenceSession {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 11);
    let batch = graph.intern_symbol("batch");
    let sequence = graph.intern_symbol("sequence");
    let total = graph.intern_symbol("total");
    let past = graph.intern_symbol("past");
    let input_ids = graph.create_named_value(
        "input_ids",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    let mask = graph.create_named_value("mask", DataType::Int64, vec![batch.into(), total.into()]);
    let positions = graph.create_named_value(
        "positions",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    let cache_key = graph.create_named_value(
        "cache_key",
        DataType::Float32,
        vec![batch.into(), 1.into(), past.into(), 1.into()],
    );
    let cache_value = graph.create_named_value(
        "cache_value",
        DataType::Float32,
        vec![batch.into(), 1.into(), past.into(), 1.into()],
    );
    for input in [input_ids, mask, positions, cache_key, cache_value] {
        graph.add_input(input);
    }
    let cast = graph.create_named_value(
        "token_values",
        DataType::Float32,
        vec![batch.into(), sequence.into()],
    );
    insert_op(
        &mut graph,
        "Cast",
        vec![input_ids],
        cast,
        &[("to", Attribute::Int(DataType::Float32 as i64))],
    );
    let draft_scores = graph.create_named_value(
        "draft_scores",
        DataType::Float32,
        vec![batch.into(), sequence.into(), 1.into()],
    );
    insert_op(
        &mut graph,
        "Unsqueeze",
        vec![cast],
        draft_scores,
        &[("axes", Attribute::Ints(vec![2]))],
    );
    let current = graph.create_named_value(
        "current_cache",
        DataType::Float32,
        vec![batch.into(), 1.into(), sequence.into(), 1.into()],
    );
    insert_op(
        &mut graph,
        "Unsqueeze",
        vec![cast],
        current,
        &[("axes", Attribute::Ints(vec![1, 3]))],
    );
    let next_key = graph.create_named_value(
        "next_key",
        DataType::Float32,
        vec![batch.into(), 1.into(), total.into(), 1.into()],
    );
    insert_op(
        &mut graph,
        "Concat",
        vec![cache_key, current],
        next_key,
        &[("axis", Attribute::Int(2))],
    );
    let next_value = graph.create_named_value(
        "next_value",
        DataType::Float32,
        vec![batch.into(), 1.into(), total.into(), 1.into()],
    );
    insert_op(
        &mut graph,
        "Concat",
        vec![cache_value, current],
        next_value,
        &[("axis", Attribute::Int(2))],
    );
    for output in [draft_scores, next_key, next_value] {
        graph.add_output(output);
    }
    session_from_graph(graph)
}

fn tiny_shared_kv_embed_proposer(width: usize) -> InferenceSession {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let batch = graph.intern_symbol("batch");
    let sequence = graph.intern_symbol("sequence");
    let kv_len = graph.intern_symbol("kv_len");
    let embeddings = graph.create_named_value(
        "embeddings",
        DataType::Float32,
        vec![batch.into(), sequence.into(), width.into()],
    );
    let mask = graph.create_named_value("mask", DataType::Int64, vec![batch.into(), kv_len.into()]);
    let positions = graph.create_named_value(
        "positions",
        DataType::Int64,
        vec![batch.into(), sequence.into()],
    );
    let shared_key = graph.create_named_value(
        "external.key",
        DataType::Float32,
        vec![batch.into(), 1.into(), kv_len.into(), width.into()],
    );
    let shared_value = graph.create_named_value(
        "external.value",
        DataType::Float32,
        vec![batch.into(), 1.into(), kv_len.into(), width.into()],
    );
    for input in [embeddings, mask, positions, shared_key, shared_value] {
        graph.add_input(input);
    }
    let draft_scores = graph.create_named_value(
        "draft_scores",
        DataType::Float32,
        vec![batch.into(), sequence.into(), width.into()],
    );
    insert_op(&mut graph, "Identity", vec![embeddings], draft_scores, &[]);
    let next_state = graph.create_named_value(
        "next_state",
        DataType::Float32,
        vec![batch.into(), sequence.into(), width.into()],
    );
    insert_op(&mut graph, "Identity", vec![embeddings], next_state, &[]);
    graph.add_output(draft_scores);
    graph.add_output(next_state);
    session_from_graph(graph)
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum AuxOutput {
    /// Static `[1, 1]` auxiliary output produced by `Cast(input_ids)`. The
    /// original capture-safe smoke case: no symbolic dims to reason about.
    StaticUnit,
    /// `[batch, 1]` auxiliary output whose leading dim is a *genuine*
    /// symbolic `batch` dim shared with `input_ids`. It resolves to `1` at
    /// decode, so it is structurally a decode unit and capture must still
    /// succeed with the output persistently bound (collapsed to `[1, 1]`).
    SymbolicBatch,
    /// `[1, total_seq]` auxiliary output produced by `Cast(attention_mask)`,
    /// whose trailing dim is the symbolic `total_seq` dim. That dim grows
    /// with the sequence and is NOT batch/query-seq, so F1 must decline to
    /// persistently bind it (collapsing to `[1, 1]` would under-allocate)
    /// and fall back to eager, where the executor JIT-sizes it each step.
    SymbolicTotalSeq,
}

#[cfg(feature = "cuda")]
fn capture_safe_cuda_decoder(
    graph_capture: bool,
    max_len: usize,
) -> anyhow::Result<NativeDecodeSession> {
    build_cuda_decoder(graph_capture, max_len, AuxOutput::StaticUnit)
}

#[cfg(feature = "cuda")]
fn build_cuda_decoder(
    graph_capture: bool,
    max_len: usize,
    aux: AuxOutput,
) -> anyhow::Result<NativeDecodeSession> {
    build_cuda_decoder_with_fixed_state(graph_capture, max_len, aux, false)
}

#[cfg(feature = "cuda")]
fn build_cuda_decoder_with_fixed_state(
    graph_capture: bool,
    max_len: usize,
    aux: AuxOutput,
    fixed_state: bool,
) -> anyhow::Result<NativeDecodeSession> {
    use prost::Message;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let batch = graph.intern_symbol("batch");
    let total = graph.intern_symbol("total");
    let past = graph.intern_symbol("past");

    // input_ids declares a symbolic batch dim for the SymbolicBatch case so
    // the aux output can legitimately *share* it; it is bound to `[1, 1]` at
    // decode regardless, so the other cases are unaffected.
    let input_ids_shape = match aux {
        AuxOutput::SymbolicBatch => vec![batch.into(), 1.into()],
        _ => vec![1.into(), 1.into()],
    };
    let input_ids = graph.create_named_value("input_ids", DataType::Int64, input_ids_shape);
    let attention_mask = graph.create_named_value(
        "attention_mask",
        DataType::Int64,
        vec![1.into(), total.into()],
    );
    let position_ids =
        graph.create_named_value("position_ids", DataType::Int64, vec![1.into(), 1.into()]);
    let past_key = graph.create_named_value(
        "past_key_values.0.key",
        DataType::Float32,
        vec![1.into(), 1.into(), past.into(), 1.into()],
    );
    let past_value = graph.create_named_value(
        "past_key_values.0.value",
        DataType::Float32,
        vec![1.into(), 1.into(), past.into(), 1.into()],
    );
    for input in [
        input_ids,
        attention_mask,
        position_ids,
        past_key,
        past_value,
    ] {
        graph.add_input(input);
    }
    let conv_state = fixed_state.then(|| {
        let value = graph.create_named_value(
            "past_key_values.0.conv_state",
            DataType::Float16,
            vec![1.into(), 4.into(), 3.into()],
        );
        graph.add_input(value);
        value
    });

    let logits = graph.create_named_value("logits", DataType::Float32, vec![1.into(), 1.into()]);
    insert_op(
        &mut graph,
        "Cast",
        vec![input_ids],
        logits,
        &[("to", Attribute::Int(DataType::Float32 as i64))],
    );
    let present_key = graph.create_named_value(
        "present.0.key",
        DataType::Float32,
        vec![1.into(), 1.into(), past.into(), 1.into()],
    );
    insert_op(
        &mut graph,
        "Cast",
        vec![past_key],
        present_key,
        &[("to", Attribute::Int(DataType::Float32 as i64))],
    );
    let present_value = graph.create_named_value(
        "present.0.value",
        DataType::Float32,
        vec![1.into(), 1.into(), past.into(), 1.into()],
    );
    insert_op(
        &mut graph,
        "Cast",
        vec![past_value],
        present_value,
        &[("to", Attribute::Int(DataType::Float32 as i64))],
    );
    // Auxiliary output geometry drives the F1 structural analysis.
    let (aux_shape, aux_source): (Vec<Dim>, _) = match aux {
        AuxOutput::StaticUnit => (vec![1.into(), 1.into()], input_ids),
        AuxOutput::SymbolicBatch => (vec![batch.into(), 1.into()], input_ids),
        AuxOutput::SymbolicTotalSeq => (vec![1.into(), total.into()], attention_mask),
    };
    let auxiliary = graph.create_named_value("auxiliary_state", DataType::Float32, aux_shape);
    insert_op(
        &mut graph,
        "Cast",
        vec![aux_source],
        auxiliary,
        &[("to", Attribute::Int(DataType::Float32 as i64))],
    );
    for output in [logits, present_key, present_value, auxiliary] {
        graph.add_output(output);
    }
    if let Some(conv_state) = conv_state {
        let present_conv_state = graph.create_named_value(
            "present.0.conv_state",
            DataType::Float16,
            vec![1.into(), 4.into(), 3.into()],
        );
        insert_op(
            &mut graph,
            "Identity",
            vec![conv_state],
            present_conv_state,
            &[],
        );
        graph.add_output(present_conv_state);
    }

    let model = onnx_std::Model::new(graph).to_proto()?.encode_to_vec();
    let session = InferenceSession::builder()
        .model_bytes(&model)
        .device(DevicePreference::Gpu { index: Some(0) })
        .build()
        .context("build capture-safe CUDA decoder")?;
    if fixed_state {
        let mut io = tiny_decoder_io();
        io.state_pairs = Some(vec![LoopStatePair {
            input: "past_key_values.0.conv_state".into(),
            output: "present.0.conv_state".into(),
            init: Some("zeros".into()),
            update: Some("replace".into()),
        }]);
        NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
            session,
            Some(max_len),
            Some(&io),
        )
    } else {
        NativeDecodeSession::from_session_with_cuda_options(
            session,
            NativeDecodeCudaOptions {
                kv_max_len: Some(max_len),
                metadata_max_len: None,
                graph_capture: Some(graph_capture),
            },
        )
    }
}

#[cfg(feature = "cuda")]
fn binding_addresses(session: &NativeDecodeSession) -> Vec<usize> {
    session
        .cuda
        .as_ref()
        .expect("CUDA state")
        .bindings
        .iter()
        .map(|binding| binding.device_ptr() as usize)
        .collect()
}

#[cfg(feature = "cuda")]
fn input_update_stats(session: &NativeDecodeSession) -> [DeviceBindingTransferStats; 3] {
    let state = session.cuda.as_ref().expect("CUDA state");
    [
        state.bindings[0].transfer_stats(),
        state.bindings[state.input_ids_binding].transfer_stats(),
        state.bindings[state.position_ids_binding.expect("position_ids binding")].transfer_stats(),
    ]
}

#[cfg(feature = "cuda")]
fn assert_single_value_uploads(
    before: [DeviceBindingTransferStats; 3],
    after: [DeviceBindingTransferStats; 3],
) {
    for (before, after) in before.into_iter().zip(after) {
        assert_eq!(after.host_upload_calls, before.host_upload_calls + 1);
        assert_eq!(
            after.host_upload_bytes,
            before.host_upload_bytes + std::mem::size_of::<i64>() as u64
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_decode_bindings(
    session: &mut NativeDecodeSession,
    addresses: &[usize],
    token: TokenId,
    position: usize,
    max_len: usize,
) -> anyhow::Result<()> {
    assert_eq!(binding_addresses(session), addresses);
    let state = session.cuda.as_mut().expect("CUDA state");

    let input = state.bindings[state.input_ids_binding].read_bytes()?;
    assert_eq!(
        i64::from_le_bytes(input.try_into().expect("one input id")),
        i64::from(token)
    );

    let position_bytes =
        state.bindings[state.position_ids_binding.expect("position_ids binding")].read_bytes()?;
    assert_eq!(
        i64::from_le_bytes(position_bytes.try_into().expect("one position id")),
        position as i64
    );

    let mask = state.bindings[0]
        .read_bytes()?
        .chunks_exact(std::mem::size_of::<i64>())
        .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(mask.len(), max_len);
    assert!(mask[..=position].iter().all(|&value| value == 1));
    assert!(mask[position + 1..].iter().all(|&value| value == 0));
    assert_eq!(state.bindings[0].logical_shape(), &[1, position + 1]);
    for binding in &state.bindings[state.kv_binding_range.clone()] {
        assert_eq!(binding.logical_shape()[2], position + 1);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn run_capture_safe_decode(
    session: &mut NativeDecodeSession,
    tokens: &[TokenId],
    addresses: &[usize],
    max_len: usize,
) -> anyhow::Result<Vec<Vec<u32>>> {
    let mut logits = Vec::with_capacity(tokens.len());
    for (position, &token) in tokens.iter().enumerate() {
        let before = input_update_stats(session);
        let step = session.decode(&[token], position)?;
        let after = input_update_stats(session);
        assert_single_value_uploads(before, after);
        assert_decode_bindings(session, addresses, token, position, max_len)?;
        logits.push(
            step.into_iter()
                .flatten()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
        );
    }
    Ok(logits)
}

#[test]
fn native_decode_advances_kv_and_rewinds() {
    let mut session = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_decoder(false),
        None,
        Some(&tiny_decoder_io()),
    )
    .expect("load decoder");
    let logits = session.decode(&[1, 2, 3], 0).expect("prefill");
    assert_eq!(logits.len(), 3);
    assert_eq!(logits[0].len(), 1);
    assert_eq!(session.current_len(), 3);

    let logits = session.decode(&[4], 3).expect("decode");
    assert_eq!(logits.len(), 1);
    assert_eq!(logits[0].len(), 1);
    assert_eq!(session.current_len(), 4);

    session.rewind(2).expect("rewind");
    assert_eq!(session.current_len(), 2);
    session.decode(&[5], 2).expect("decode after rewind");
    assert_eq!(session.current_len(), 3);
}

#[test]
fn native_target_step_preserves_token_driven_binding() {
    let mut session = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_decoder(false),
        None,
        Some(&tiny_decoder_io()),
    )
    .expect("load token target");
    assert!(session.step_inputs.iter().any(|binding| {
        binding.name == "input_ids" && binding.source == NativeStepInputSource::TokenIds
    }));
    assert!(
        !session
            .step_inputs
            .iter_mut()
            .any(|binding| binding.source == NativeStepInputSource::Routed)
    );

    let logits = session.decode(&[3, 5], 0).expect("token target step");
    assert_eq!(logits, vec![vec![3.0], vec![5.0]]);
}

#[test]
fn native_target_step_binds_declared_inputs_embeds_instead_of_tokens() {
    let io = target_io(SequenceInputKind::InputsEmbeds);
    let mut session = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_embedding_target(false),
        None,
        Some(&io),
    )
    .expect("load embedding target");
    assert!(session.step_inputs.iter().any(|binding| {
        binding.name == "embedded_sequence" && binding.source == NativeStepInputSource::InputsEmbeds
    }));
    assert!(
        !session
            .step_inputs
            .iter()
            .any(|binding| binding.source == NativeStepInputSource::TokenIds)
    );

    let embeddings = Tensor::from_f32(&[1, 2, 1], &[1.25, 2.5]).expect("embedding tensor");
    let logits = session
        .decode_with_step_inputs(&[101, 102], 0, &[("embedded_sequence".into(), embeddings)])
        .expect("embedding target step");
    assert_eq!(logits, vec![vec![1.25], vec![2.5]]);
}

#[test]
fn native_target_step_resolves_routed_component_output_by_declared_port() {
    let mut producer = tiny_step_producer();
    let producer_input = Tensor::from_f32(&[1, 2, 1], &[10.0, 20.0]).expect("producer input");
    let mut producer_outputs = producer
        .run(&[("producer_input", &producer_input)])
        .expect("producer step");
    let routed = producer_outputs.remove(0);

    let io = target_io(SequenceInputKind::InputsEmbeds);
    let mut target = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_embedding_target(true),
        None,
        Some(&io),
    )
    .expect("load routed target");
    assert!(target.step_inputs.iter().any(|binding| {
        binding.name == "routed_features" && binding.source == NativeStepInputSource::Routed
    }));
    let embeddings = Tensor::from_f32(&[1, 2, 1], &[1.0, 2.0]).expect("embedding tensor");
    let logits = target
        .decode_with_step_inputs(
            &[7, 8],
            0,
            &[
                ("embedded_sequence".into(), embeddings),
                ("routed_features".into(), routed),
            ],
        )
        .expect("routed target step");
    assert_eq!(logits, vec![vec![11.0], vec![22.0]]);
}

#[test]
fn native_proposer_defaults_preserve_token_ids_and_owned_kv() {
    let mut io = proposer_io(SequenceInputKind::TokenIds, KvOwnership::Owned);
    io.sequence_source = None;
    io.kv_ownership = None;
    let mut proposer = NativeProposerSession::from_session(tiny_owned_kv_proposer(), Some(&io))
        .expect("load token proposer");
    let first = proposer.step_token_ids(&[2, 4]).expect("first proposal");
    assert_eq!(first.logits, Some(vec![vec![2.0], vec![4.0]]));
    assert_eq!(first.projected_state, None);
    let second = proposer.step_token_ids(&[7]).expect("second proposal");
    assert_eq!(second.logits, Some(vec![vec![7.0]]));
    assert_eq!(proposer.current_len, 3);
}

#[test]
fn native_proposer_runs_inputs_embeds_with_shared_kv_and_output_roles() {
    let width = 3;
    let io = proposer_io(SequenceInputKind::InputsEmbeds, KvOwnership::Shared);
    let mut proposer =
        NativeProposerSession::from_session(tiny_shared_kv_embed_proposer(width), Some(&io))
            .expect("load embedding proposer");
    let key = Tensor::from_f32(&[1, 1, 2, width], &[0.0; 6]).expect("shared key");
    let value = Tensor::from_f32(&[1, 1, 2, width], &[1.0; 6]).expect("shared value");
    let inputs = vec![
        ("external.key".to_string(), key),
        ("external.value".to_string(), value),
    ];
    let embeddings = [1.0, 2.0, 3.0];
    let output = proposer
        .step_inputs_embeds(&embeddings, 5, &inputs)
        .expect("shared-KV proposal");
    assert_eq!(output.logits, Some(vec![embeddings.to_vec()]));
    assert_eq!(output.projected_state, Some(embeddings.to_vec()));
    assert_eq!(proposer.current_len, 0, "shared KV is target-owned");
}

#[test]
fn native_decode_verify_then_rewind_matches_fresh_decode() {
    // WP1 exit criterion (CPU logic coverage): verify K tokens, rewind to the
    // committed length, and prove a subsequent decode is bit-identical to a
    // fresh decode from the same committed prefix (no KV corruption). The
    // device-KV bit-identity variant is `native_cuda_verify_rewind_no_kv_corruption`.
    let mut session = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_decoder(false),
        None,
        Some(&tiny_decoder_io()),
    )
    .expect("load decoder");
    let prompt = [1, 2, 3];
    session.decode(&prompt, 0).expect("prefill");
    let past = session.current_len();
    assert_eq!(past, prompt.len());

    // Verify a K-token draft via the verify primitive: returns [K, vocab].
    let draft = [4, 5, 6];
    let rows = session.decode_verify(&draft, past).expect("verify");
    assert_eq!(rows.len(), draft.len());
    assert_eq!(rows[0].len(), 1);
    assert_eq!(session.current_len(), past + draft.len());

    // Accept j of the draft, rewind device/host KV to the committed length.
    let j = 1;
    session.rewind(past + j).expect("rewind");
    assert_eq!(session.current_len(), past + j);

    // Subsequent decode from the committed prefix.
    let feed = 9;
    let after = session
        .decode(&[feed], past + j)
        .expect("decode after rewind");

    // Fresh session decoded over the committed prefix prompt ++ draft[..j].
    let mut fresh = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_decoder(false),
        None,
        Some(&tiny_decoder_io()),
    )
    .expect("fresh decoder");
    let mut committed = prompt.to_vec();
    committed.extend_from_slice(&draft[..j]);
    fresh.decode(&committed, 0).expect("fresh prefill");
    let fresh_after = fresh
        .decode(&[feed], committed.len())
        .expect("fresh decode");

    let after_bits = after
        .iter()
        .flatten()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    let fresh_bits = fresh_after
        .iter()
        .flatten()
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(
        after_bits, fresh_bits,
        "verify+rewind diverged from fresh decode"
    );
}

#[test]
fn native_decode_verify_requires_matching_past_and_nonempty_draft() {
    let mut session = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_decoder(false),
        None,
        Some(&tiny_decoder_io()),
    )
    .expect("load decoder");
    session.decode(&[1, 2], 0).expect("prefill");
    assert!(
        session
            .decode_verify(&[], 2)
            .expect_err("empty draft must fail")
            .to_string()
            .contains("at least one draft token")
    );
    assert!(
        session
            .decode_verify(&[3], 5)
            .expect_err("past mismatch must fail")
            .to_string()
            .contains("past length mismatch")
    );
}

#[test]
fn native_decode_option_c_scaffolding_is_dormant_by_default() {
    // The padded M=maxK capture + retain-graph-on-rewind switches (option (c))
    // must stay dormant. On a CPU session (no CUDA state) the controls are
    // inert no-ops and the capacity stays `None`.
    let mut session = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_decoder(false),
        None,
        Some(&tiny_decoder_io()),
    )
    .expect("load decoder");
    assert_eq!(session.padded_query_capacity(), None);
    session.set_retain_graph_on_rewind(true);
    session.configure_padded_verify_capture(8);
    assert_eq!(session.padded_query_capacity(), None);
}

#[test]
fn persistent_auxiliary_output_shape_is_fixed_and_rejects_strings() {
    let shape = DecodeCudaState::persistent_output_shape(
        "auxiliary_state",
        DataType::Float16,
        &[Dim::Symbolic(SymbolId(0)), Dim::Static(1536)],
    )
    .expect("numeric auxiliary output must be bindable");
    assert_eq!(shape, [1, 1536]);

    let error = DecodeCudaState::persistent_output_shape(
        "auxiliary_text",
        DataType::String,
        &[Dim::Static(1)],
    )
    .expect_err("variable-width auxiliary output must fail explicitly");
    let message = error.to_string();
    assert!(message.contains("auxiliary_text"));
    assert!(message.contains("fixed-size device tensor storage"));
    assert!(message.contains("export this output as a numeric tensor"));
}

#[test]
fn cuda_persistent_state_shapes_preserve_growing_kv_and_fixed_recurrent_geometry() {
    let batch = SymbolId(0);
    let past = SymbolId(1);
    let (kv_physical, kv_logical) = DecodeCudaState::persistent_state_shapes(
        "past_key_values.0.key",
        DataType::Float16,
        &[
            Dim::Symbolic(batch),
            Dim::Static(4),
            Dim::Symbolic(past),
            Dim::Static(256),
        ],
        512,
        false,
    )
    .expect("rank-4 growing KV");
    assert_eq!(kv_physical, [1, 4, 512, 256]);
    assert_eq!(kv_logical, [1, 4, 0, 256]);

    let (conv_physical, conv_logical) = DecodeCudaState::persistent_state_shapes(
        "past_key_values.0.conv_state",
        DataType::Float16,
        &[Dim::Symbolic(batch), Dim::Static(10_240), Dim::Static(3)],
        512,
        true,
    )
    .expect("rank-3 fixed convolution state");
    assert_eq!(conv_physical, [1, 10_240, 3]);
    assert_eq!(conv_logical, conv_physical);

    let (recurrent_physical, recurrent_logical) = DecodeCudaState::persistent_state_shapes(
        "past_key_values.0.recurrent_state",
        DataType::Float16,
        &[
            Dim::Symbolic(batch),
            Dim::Static(48),
            Dim::Static(128),
            Dim::Static(128),
        ],
        512,
        true,
    )
    .expect("rank-4 fixed recurrent state");
    assert_eq!(recurrent_physical, [1, 48, 128, 128]);
    assert_eq!(recurrent_logical, recurrent_physical);
}

#[test]
fn cuda_fixed_state_shapes_reject_unbounded_non_batch_dimensions() {
    let error = DecodeCudaState::persistent_state_shapes(
        "state",
        DataType::Float16,
        &[Dim::Symbolic(SymbolId(0)), Dim::Symbolic(SymbolId(1))],
        128,
        true,
    )
    .expect_err("non-batch fixed-state dimensions require static bounds");
    assert!(error.to_string().contains(
        "dimension 1 in shape [Symbolic(SymbolId(0)), Symbolic(SymbolId(1))] is symbolic"
    ));
}

#[test]
fn unit_symbol_collection_is_structural_and_batch_aware() {
    // input_ids / position_ids are bound to `[1, 1]` at decode, so *every*
    // symbolic axis is a decode-unit (batch or query-seq). attention_mask
    // and past-KV grow along their sequence axis, so only axis 0 (batch) is
    // a unit; the total_seq / past symbols must NOT be collected.
    let batch = SymbolId(0);
    let query_seq = SymbolId(1);
    let total = SymbolId(2);
    let past = SymbolId(3);

    let mut unit = HashSet::new();
    // input_ids: [batch, query_seq]
    DecodeCudaState::collect_unit_symbols(
        &[Dim::Symbolic(batch), Dim::Symbolic(query_seq)],
        false,
        &mut unit,
    );
    // attention_mask: [batch, total] — batch only.
    DecodeCudaState::collect_unit_symbols(
        &[Dim::Symbolic(batch), Dim::Symbolic(total)],
        true,
        &mut unit,
    );
    // past-KV: [batch, heads, past, head_dim] — batch only.
    DecodeCudaState::collect_unit_symbols(
        &[
            Dim::Symbolic(batch),
            Dim::Static(4),
            Dim::Symbolic(past),
            Dim::Static(8),
        ],
        true,
        &mut unit,
    );

    assert!(unit.contains(&batch));
    assert!(unit.contains(&query_seq));
    assert!(
        !unit.contains(&total),
        "total_seq must not be a decode unit"
    );
    assert!(!unit.contains(&past), "past must not be a decode unit");
}

#[test]
fn unresolved_symbolic_axis_flags_only_non_unit_symbols() {
    let batch = SymbolId(0);
    let query_seq = SymbolId(1);
    let total = SymbolId(2);
    let unit = HashSet::from([batch, query_seq]);

    // Fully static aux output: always bindable.
    assert_eq!(
        DecodeCudaState::unresolved_symbolic_axis(&[Dim::Static(1), Dim::Static(1536)], &unit),
        None
    );
    // Symbolic dim that IS a decode unit (batch): safe to collapse to 1.
    assert_eq!(
        DecodeCudaState::unresolved_symbolic_axis(
            &[Dim::Symbolic(batch), Dim::Static(1536)],
            &unit
        ),
        None
    );
    // Symbolic dim that is NOT batch/query-seq (an accumulator indexed by
    // total_seq): flagged with its axis and symbol so decode declines to
    // persistently bind it.
    assert_eq!(
        DecodeCudaState::unresolved_symbolic_axis(
            &[Dim::Static(1), Dim::Symbolic(total), Dim::Static(64)],
            &unit
        ),
        Some((1, total))
    );
}

#[cfg(feature = "cuda")]
#[test]
fn native_cuda_binds_rank3_fixed_state_without_changing_growing_kv() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }

    let mut session = build_cuda_decoder_with_fixed_state(false, 16, AuxOutput::StaticUnit, true)?;
    let state = session.cuda.as_mut().expect("CUDA state");
    assert_eq!(state.kv_binding_range.len(), 2);
    let fixed = state
        .bindings
        .iter_mut()
        .find(|binding| binding.input_name() == "past_key_values.0.conv_state")
        .expect("rank-3 fixed state binding");
    assert_eq!(fixed.physical_shape(), &[1, 4, 3]);
    assert_eq!(fixed.logical_shape(), &[1, 4, 3]);
    assert!(fixed.read_bytes()?.iter().all(|byte| *byte == 0));

    session.decode(&[7], 0)?;
    let state = session.cuda.as_mut().expect("CUDA state");
    for binding in &state.bindings[state.kv_binding_range.clone()] {
        assert_eq!(binding.logical_shape()[2], 1);
    }
    let fixed = state
        .bindings
        .iter_mut()
        .find(|binding| binding.input_name() == "past_key_values.0.conv_state")
        .expect("rank-3 fixed state binding");
    assert_eq!(fixed.logical_shape(), &[1, 4, 3]);
    assert!(fixed.read_bytes()?.iter().all(|byte| *byte == 0));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn native_cuda_capture_replay_is_bit_exact_and_refreshes_decode_inputs() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }

    const MAX_LEN: usize = 16;
    const TOKENS: [TokenId; 10] = [3, 17, 5, 29, 11, 23, 7, 31, 13, 2];

    let mut eager = capture_safe_cuda_decoder(false, MAX_LEN)?;
    let eager_addresses = binding_addresses(&eager);
    let eager_first = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;
    let eager_stats = eager.cuda_kv_debug_stats().expect("CUDA stats");
    assert!(!eager_stats.graph.enabled);
    assert_eq!(eager_stats.graph.captures, 0);
    assert_eq!(eager_stats.graph.replays, 0);
    assert_eq!(eager_stats.graph.fallbacks, 0);
    assert!(eager.cuda_graph_fallback_reason().is_none());

    let mut captured = capture_safe_cuda_decoder(true, MAX_LEN)?;
    {
        let state = captured.cuda.as_ref().expect("CUDA state");
        assert_eq!(state.auxiliary_binding_range.len(), 1);
        assert_eq!(
            state.bindings[state.auxiliary_binding_range.start].output_name(),
            Some("auxiliary_state")
        );
        assert!(state.auxiliary_binding_range.end <= state.base_binding_count);
    }
    let captured_addresses = binding_addresses(&captured);
    let captured_first =
        run_capture_safe_decode(&mut captured, &TOKENS, &captured_addresses, MAX_LEN)?;
    let first_stats = captured.cuda_kv_debug_stats().expect("CUDA stats");
    assert!(first_stats.graph.enabled);
    assert_eq!(first_stats.graph.captures, 1);
    assert_eq!(first_stats.graph.replays, TOKENS.len() as u64 - 2);
    assert_eq!(first_stats.graph.fallbacks, 0);
    assert!(captured.cuda_graph_fallback_reason().is_none());
    assert_eq!(captured_first, eager_first);
    assert_eq!(
        captured_first,
        TOKENS
            .iter()
            .map(|&token| vec![(token as f32).to_bits()])
            .collect::<Vec<_>>()
    );
    assert_eq!(captured_addresses, binding_addresses(&captured));
    assert_eq!(
        first_stats.kv_transfers,
        DeviceBindingTransferStats::default()
    );

    eager.reset()?;
    captured.reset()?;
    let eager_second = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;
    let captured_second =
        run_capture_safe_decode(&mut captured, &TOKENS, &captured_addresses, MAX_LEN)?;
    let second_stats = captured.cuda_kv_debug_stats().expect("CUDA stats");
    assert_eq!(captured_second, eager_second);
    assert_eq!(captured_second, captured_first);
    assert_eq!(second_stats.graph.captures, 2);
    assert_eq!(second_stats.graph.replays, 2 * (TOKENS.len() as u64 - 2));
    assert_eq!(second_stats.graph.fallbacks, 0);
    assert_eq!(captured_addresses, binding_addresses(&captured));
    assert_eq!(
        second_stats.kv_transfers,
        DeviceBindingTransferStats::default()
    );

    eprintln!(
        "native CUDA capture-safe decode parity: captures={} replays={} fallbacks={} steps_per_generation={}",
        second_stats.graph.captures,
        second_stats.graph.replays,
        second_stats.graph.fallbacks,
        TOKENS.len()
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn native_cuda_symbolic_batch_aux_captures_bit_exact() -> anyhow::Result<()> {
    // F2 positive case: an auxiliary output with a *genuinely symbolic* dim
    // (`batch`) that resolves to 1 at decode must remain persistently
    // bindable and fully capturable — capture succeeds, no fallback, and
    // replay is bit-exact with the eager device path.
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }

    const MAX_LEN: usize = 16;
    const TOKENS: [TokenId; 8] = [3, 17, 5, 29, 11, 23, 7, 31];

    let mut eager = build_cuda_decoder(false, MAX_LEN, AuxOutput::SymbolicBatch)?;
    let eager_addresses = binding_addresses(&eager);
    let eager_logits = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;

    let mut captured = build_cuda_decoder(true, MAX_LEN, AuxOutput::SymbolicBatch)?;
    {
        let state = captured.cuda.as_ref().expect("CUDA state");
        // The symbolic-batch aux output is structurally a decode unit, so it
        // is persistently bound (collapsed to [1, 1]) — F1 does NOT decline.
        assert!(state.graph_enabled);
        assert!(captured.cuda_auxiliary_bind_declines().is_empty());
        let state = captured.cuda.as_ref().unwrap();
        assert_eq!(state.auxiliary_binding_range.len(), 1);
        assert_eq!(
            state.bindings[state.auxiliary_binding_range.start].output_name(),
            Some("auxiliary_state")
        );
    }
    let captured_addresses = binding_addresses(&captured);
    let captured_logits =
        run_capture_safe_decode(&mut captured, &TOKENS, &captured_addresses, MAX_LEN)?;
    let stats = captured.cuda_kv_debug_stats().expect("CUDA stats");
    assert!(stats.graph.enabled);
    assert_eq!(stats.graph.captures, 1);
    assert_eq!(stats.graph.replays, TOKENS.len() as u64 - 2);
    assert_eq!(stats.graph.fallbacks, 0);
    assert!(captured.cuda_graph_fallback_reason().is_none());
    assert_eq!(captured_logits, eager_logits);
    assert_eq!(
        captured_logits,
        TOKENS
            .iter()
            .map(|&token| vec![(token as f32).to_bits()])
            .collect::<Vec<_>>()
    );
    assert_eq!(captured_addresses, binding_addresses(&captured));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn native_cuda_symbolic_total_seq_aux_declines_capture_but_decodes_eagerly() -> anyhow::Result<()> {
    // F2 negative case (the F1 path): an auxiliary output whose symbolic dim
    // is `total_seq` — NOT batch/query-seq — cannot be collapsed to a fixed
    // persistent binding without under-allocating. F1 must decline capture
    // at binding time (leaving the output unbound), yet decode MUST still
    // work via the eager device path, producing correct output.
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }

    const MAX_LEN: usize = 16;
    const TOKENS: [TokenId; 8] = [3, 17, 5, 29, 11, 23, 7, 31];

    // Request graph capture; F1 must decline it structurally.
    let mut declined = build_cuda_decoder(true, MAX_LEN, AuxOutput::SymbolicTotalSeq)?;
    {
        let state = declined.cuda.as_ref().expect("CUDA state");
        assert!(
            !state.graph_enabled,
            "F1 must disable capture for an unresolved-symbolic aux output"
        );
        // The unbindable aux output is left out of the persistent bindings.
        assert_eq!(state.auxiliary_binding_range.len(), 0);
    }
    let declines = declined.cuda_auxiliary_bind_declines();
    assert_eq!(declines.len(), 1);
    assert!(declines[0].contains("auxiliary_state"));
    assert!(declines[0].contains("total"));
    assert!(declines[0].contains("not structurally batch or query-seq"));

    let addresses = binding_addresses(&declined);
    let declined_logits = run_capture_safe_decode(&mut declined, &TOKENS, &addresses, MAX_LEN)?;
    let stats = declined.cuda_kv_debug_stats().expect("CUDA stats");
    assert!(!stats.graph.enabled);
    assert_eq!(stats.graph.captures, 0);
    assert_eq!(stats.graph.replays, 0);
    assert_eq!(stats.graph.fallbacks, 0);

    // Decode is bit-exact with a plain eager (graph_capture=false) run — the
    // unresolved aux output changes nothing about the decode result.
    let mut eager = build_cuda_decoder(false, MAX_LEN, AuxOutput::SymbolicTotalSeq)?;
    let eager_addresses = binding_addresses(&eager);
    let eager_logits = run_capture_safe_decode(&mut eager, &TOKENS, &eager_addresses, MAX_LEN)?;
    assert_eq!(declined_logits, eager_logits);
    assert_eq!(
        declined_logits,
        TOKENS
            .iter()
            .map(|&token| vec![(token as f32).to_bits()])
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn native_decode_accepts_last_token_only_logits_and_advances_kv() {
    let mut session = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_decoder(true),
        None,
        Some(&tiny_decoder_io()),
    )
    .expect("load decoder");

    let logits = session.decode(&[1, 2, 3], 0).expect("prefill");
    assert_eq!(logits, vec![vec![10.0, 20.0]]);
    assert_eq!(session.current_len(), 3);

    let logits = session.decode(&[4], 3).expect("decode");
    assert_eq!(logits, vec![vec![10.0, 20.0]]);
    assert_eq!(session.current_len(), 4);
}

#[cfg(feature = "cuda")]
#[test]
fn native_cuda_qwen_decode_matches_cpu_tokens() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }

    let Some(model_dir) = qwen_cuda_smoke_model_dir() else {
        return Ok(());
    };
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
    let prompt = tokenizer.encode("Hello")?;
    const HORIZON: usize = 64;
    let generate = |session: &mut NativeDecodeSession| -> anyhow::Result<(Vec<TokenId>, u128)> {
        let mut logits = session
            .decode(&prompt, 0)?
            .pop()
            .context("prefill must produce logits")?;
        let mut tokens = Vec::with_capacity(HORIZON);
        let mut decode_nanos = 0u128;
        for step in 0..HORIZON {
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index as TokenId)
                .context("logits must not be empty")?;
            tokens.push(token);
            if step + 1 == HORIZON {
                break;
            }
            let start = std::time::Instant::now();
            logits = session
                .decode(&[token], session.current_len())?
                .pop()
                .context("decode must produce logits")?;
            decode_nanos += start.elapsed().as_nanos();
        }
        Ok((tokens, decode_nanos))
    };

    let mut cpu = NativeDecodeSession::load(model_dir.join("model.onnx"), NativeDecodeDevice::Cpu)?;
    let (cpu_tokens, _) = generate(&mut cpu)?;
    drop(cpu);

    let mut eager = NativeDecodeSession::load_with_cuda_options(
        model_dir.join("model.onnx"),
        NativeDecodeDevice::Cuda { index: Some(0) },
        NativeDecodeCudaOptions {
            kv_max_len: Some(128),
            metadata_max_len: None,
            graph_capture: Some(false),
        },
    )?;
    let eager_before = eager
        .cuda_kv_debug_stats()
        .context("CUDA session must expose device KV stats")?;
    let (eager_tokens, eager_nanos) = generate(&mut eager)?;
    let eager_after = eager.cuda_kv_debug_stats().unwrap();
    assert!(!eager_after.graph.enabled);
    assert_eq!(eager_after.graph.captures, 0);
    assert_eq!(eager_after.graph.replays, 0);
    drop(eager);

    let mut captured = NativeDecodeSession::load_with_cuda_options(
        model_dir.join("model.onnx"),
        NativeDecodeDevice::Cuda { index: Some(0) },
        NativeDecodeCudaOptions {
            kv_max_len: Some(128),
            metadata_max_len: None,
            graph_capture: Some(true),
        },
    )?;
    let captured_before = captured.cuda_kv_debug_stats().unwrap();
    let (captured_tokens, captured_nanos) = generate(&mut captured)?;
    let captured_after = captured.cuda_kv_debug_stats().unwrap();

    assert_eq!(cpu_tokens.len(), HORIZON);
    assert_eq!(eager_tokens, cpu_tokens);
    assert_eq!(captured_tokens, eager_tokens);
    assert_eq!(
        &cpu_tokens[..8],
        &[11576, 42740, 11, 358, 614, 264, 3405, 911]
    );
    assert_eq!(eager_before.device_ptrs, eager_after.device_ptrs);
    assert_eq!(captured_before.device_ptrs, captured_after.device_ptrs);
    assert_eq!(
        captured_after.kv_transfers,
        DeviceBindingTransferStats::default()
    );
    assert!(captured_after.graph.enabled);
    assert_eq!(captured_after.graph.captures, 1);
    assert_eq!(captured_after.graph.replays, HORIZON as u64 - 2);
    assert_eq!(captured_after.graph.fallbacks, 0);
    assert!(captured.cuda_graph_fallback_reason().is_none());

    let eager_us = eager_nanos as f64 / (HORIZON - 1) as f64 / 1000.0;
    let captured_us = captured_nanos as f64 / (HORIZON - 1) as f64 / 1000.0;
    eprintln!(
        "native CUDA decode wall-time: eager={eager_us:.1} us/token, graph-flag={captured_us:.1} us/token, delta={:.1}%",
        (captured_us / eager_us - 1.0) * 100.0
    );

    captured.rewind(captured_after.logical_len - 2)?;
    let rewound = captured.cuda_kv_debug_stats().unwrap();
    assert_eq!(rewound.logical_len, captured_after.logical_len - 2);
    assert_eq!(rewound.device_ptrs, captured_before.device_ptrs);
    assert_eq!(rewound.kv_transfers, DeviceBindingTransferStats::default());

    captured.reset()?;
    let (second_tokens, _) = generate(&mut captured)?;
    assert_eq!(second_tokens, captured_tokens);

    captured.reset()?;
    let error = captured
        .decode(&vec![0; 129], 0)
        .expect_err("decode beyond configured KV capacity must fail");
    assert!(error.to_string().contains("CUDA KV capacity exceeded"));
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn native_cuda_verify_rewind_no_kv_corruption() -> anyhow::Result<()> {
    // WP1 exit criterion on real device KV: decode K draft tokens through the
    // eager M=K verify primitive, rewind to the committed length (past+j), and
    // prove a subsequent M=1 decode is BIT-IDENTICAL to a fresh M=1 decode from
    // the same committed prefix. Bit-identity proves the rewind left no stale
    // KV columns attended. Both rewind regimes are exercised: option (b)
    // (invalidate-on-rewind, the default) and the dormant option (c) guard
    // (retain-graph-on-rewind), which must be equally KV-correct.
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    let Some(model_dir) = qwen_cuda_smoke_model_dir() else {
        return Ok(());
    };
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
    let prompt = tokenizer.encode("The quick brown fox")?;

    let argmax = |row: &[f32]| -> TokenId {
        row.iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as TokenId)
            .expect("logits row must not be empty")
    };
    let make = |graph: bool| -> anyhow::Result<NativeDecodeSession> {
        NativeDecodeSession::load_with_cuda_options(
            model_dir.join("model.onnx"),
            NativeDecodeDevice::Cuda { index: Some(0) },
            NativeDecodeCudaOptions {
                kv_max_len: Some(128),
                metadata_max_len: None,
                graph_capture: Some(graph),
            },
        )
    };

    // Oracle: greedy-continue the prompt to obtain a deterministic draft.
    let mut oracle = make(true)?;
    let mut logits = oracle.decode(&prompt, 0)?.pop().context("prefill logits")?;
    let mut cont = Vec::new();
    for _ in 0..6 {
        let token = argmax(&logits);
        cont.push(token);
        logits = oracle
            .decode(&[token], oracle.current_len())?
            .pop()
            .context("oracle decode logits")?;
    }
    drop(oracle);

    let past = prompt.len();
    let draft = &cont[..4];
    let j = 2usize; // pretend the driver accepted 2 of the 4 draft tokens
    let feed = cont[j]; // deterministic next token fed after the committed prefix

    for retain in [false, true] {
        let mut verify_sess = make(true)?;
        verify_sess.decode(&prompt, 0)?;
        if retain {
            verify_sess.set_retain_graph_on_rewind(true);
        }
        assert_eq!(verify_sess.current_len(), past);

        // Eager M=K verify pass returns one logits row per draft position.
        let rows = verify_sess.decode_verify(draft, past)?;
        assert_eq!(rows.len(), draft.len());
        assert_eq!(verify_sess.current_len(), past + draft.len());

        // Rewind to the committed length; mask/KV logical shapes must follow.
        verify_sess.rewind(past + j)?;
        assert_eq!(verify_sess.current_len(), past + j);
        let stats = verify_sess.cuda_kv_debug_stats().unwrap();
        assert_eq!(stats.logical_len, past + j);

        let after = verify_sess.decode(&[feed], past + j)?;

        // Fresh M=1 reference from the committed prefix prompt ++ draft[..j].
        let mut fresh = make(true)?;
        let mut committed = prompt.clone();
        committed.extend_from_slice(&draft[..j]);
        fresh.decode(&committed, 0)?;
        let fresh_after = fresh.decode(&[feed], committed.len())?;

        let after_bits = after
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        let fresh_bits = fresh_after
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            after_bits, fresh_bits,
            "verify+rewind (retain_graph_on_rewind={retain}) corrupted device KV vs fresh M=1 decode"
        );
    }
    Ok(())
}

#[test]
fn native_logits_shapes_match_ort_semantics() {
    let cases = [
        (vec![3], 1),
        (vec![1, 3], 1),
        (vec![2, 3], 2),
        (vec![1, 1, 3], 1),
        (vec![1, 2, 3], 2),
        (vec![2, 2, 3], 2),
    ];
    for (shape, expected_rows) in cases {
        let values = (0..shape.iter().product::<usize>())
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let tensor = Tensor::from_f32(&shape, &values).expect("create logits");
        let logits = extract_logits(&tensor).expect("extract logits");
        assert_eq!(logits.len(), expected_rows, "shape {shape:?}");
        assert_eq!(logits[0].len(), 3, "shape {shape:?}");
    }

    let tensor = Tensor::from_f32(&[1, 1, 1, 3], &[0.0; 3]).expect("create logits");
    let error = extract_logits(&tensor).expect_err("rank-four logits must be rejected");
    assert!(
        error
            .to_string()
            .contains("unsupported logits tensor shape")
    );
}

/// Build a rank-4 growable KV input consumed by a `GroupQueryAttention`
/// node. The node is never executed — these tests only inspect graph
/// topology through [`all_pasts_consumed_by_gqa`].
fn graph_with_gqa_consuming_past(past_name: &str) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 11);
    let batch = graph.intern_symbol("batch");
    let past = graph.intern_symbol("past");
    let past_key = graph.create_named_value(
        past_name,
        DataType::Float32,
        vec![batch.into(), 1.into(), past.into(), 1.into()],
    );
    graph.add_input(past_key);
    let present = graph.create_named_value(
        "present.0.key",
        DataType::Float32,
        vec![batch.into(), 1.into(), past.into(), 1.into()],
    );
    insert_op(
        &mut graph,
        "GroupQueryAttention",
        vec![past_key],
        present,
        &[],
    );
    graph.add_output(present);
    graph
}

#[test]
fn all_pasts_consumed_by_gqa_true_only_when_every_past_feeds_gqa() {
    let past = "past_key_values.0.key".to_string();
    let gqa_graph = graph_with_gqa_consuming_past(&past);
    assert!(all_pasts_consumed_by_gqa(
        &gqa_graph,
        std::slice::from_ref(&past)
    ));
    // A past that no GQA node consumes must not enable the in-place path.
    assert!(!all_pasts_consumed_by_gqa(
        &gqa_graph,
        &["past_key_values.0.value".to_string()]
    ));
    // Empty pair sets are never eligible.
    assert!(!all_pasts_consumed_by_gqa(&gqa_graph, &[]));
}

#[test]
fn all_pasts_consumed_by_gqa_false_for_concat_producer() {
    // The `tiny_decoder` builds present KV with a plain `Concat`, which has
    // no append-aware in-place path, so the gate must decline it.
    let session = tiny_decoder(false);
    let pasts = vec![
        "past_key_values.0.key".to_string(),
        "past_key_values.0.value".to_string(),
    ];
    assert!(!all_pasts_consumed_by_gqa(session.graph(), &pasts));
}

#[test]
fn decode_cpu_kv_state_declines_non_gqa_model() {
    // End-to-end gate: on a Concat-based decoder the persistent CPU KV state
    // must refuse to bind (returns `Ok(None)`), keeping the safe copy path.
    let mut session = tiny_decoder(false);
    let mut present_to_past = HashMap::new();
    present_to_past.insert(
        "present.0.key".to_string(),
        "past_key_values.0.key".to_string(),
    );
    present_to_past.insert(
        "present.0.value".to_string(),
        "past_key_values.0.value".to_string(),
    );
    let state =
        DecodeCpuKvState::new(&mut session, &present_to_past, 128).expect("gate must not error");
    assert!(
        state.is_none(),
        "Concat producer must not take in-place path"
    );
}

#[test]
fn tiny_decoder_matches_across_inplace_env_toggle() {
    // The Concat model falls back to the copy path regardless of the opt-in
    // env var, so greedy decoding is identical whether the flag is on or off
    // — proving the gate prevents any behavioural change on ineligible models.
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let run = |value: &str| -> Vec<Vec<f32>> {
        // SAFETY: serialized by `env_lock`; restored below.
        unsafe { std::env::set_var("ONNX_GENAI_CPU_INPLACE_KV", value) };
        let mut session = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
            tiny_decoder(false),
            None,
            Some(&tiny_decoder_io()),
        )
        .expect("load decoder");
        let prefill = session.decode(&[1, 2, 3], 0).expect("prefill");
        let step = session.decode(&[4], 3).expect("decode step");
        unsafe { std::env::remove_var("ONNX_GENAI_CPU_INPLACE_KV") };
        let mut rows = prefill;
        rows.extend(step);
        rows
    };
    let on = run("1");
    let off = run("0");
    assert_eq!(on, off, "in-place env toggle must not change Concat decode");
}

#[test]
fn cpu_inplace_kv_max_len_env_parsing() {
    let _guard = env_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    // SAFETY: serialized by `env_lock`; each case restores the environment.
    unsafe { std::env::remove_var("ONNX_GENAI_CPU_INPLACE_KV") };
    unsafe { std::env::remove_var("ONNX_GENAI_CPU_KV_MAX_LEN") };
    assert_eq!(
        cpu_inplace_kv_max_len_from_env().expect("default"),
        Some(DEFAULT_CPU_KV_MAX_LEN)
    );

    unsafe { std::env::set_var("ONNX_GENAI_CPU_INPLACE_KV", "0") };
    assert_eq!(cpu_inplace_kv_max_len_from_env().expect("disabled"), None);
    unsafe { std::env::set_var("ONNX_GENAI_CPU_INPLACE_KV", "1") };

    unsafe { std::env::set_var("ONNX_GENAI_CPU_KV_MAX_LEN", "2048") };
    assert_eq!(
        cpu_inplace_kv_max_len_from_env().expect("custom"),
        Some(2048)
    );

    unsafe { std::env::set_var("ONNX_GENAI_CPU_KV_MAX_LEN", "0") };
    assert!(cpu_inplace_kv_max_len_from_env().is_err(), "zero rejected");

    unsafe { std::env::set_var("ONNX_GENAI_CPU_KV_MAX_LEN", "notanumber") };
    assert!(
        cpu_inplace_kv_max_len_from_env().is_err(),
        "garbage rejected"
    );

    unsafe { std::env::remove_var("ONNX_GENAI_CPU_INPLACE_KV") };
    unsafe { std::env::remove_var("ONNX_GENAI_CPU_KV_MAX_LEN") };
}

/// Serializes the environment-variable-mutating tests so their `set_var`
/// calls do not race under the parallel test runner.
fn env_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// The native position rank is derived from the graph's declared `position_ids`
/// shape: a conventional rank-2 `[batch, seq]` input maps to 1 coordinate axis,
/// while a rank-3 mrope `[N, batch, seq]` input maps to its static leading dim.
/// A non-static leading dim (or an absent input) is handled without inventing an
/// axis count.
#[test]
fn declared_position_rank_maps_graph_shape() {
    use onnx_genai_ort::{DataType, TensorInfo};

    let inputs = vec![
        TensorInfo {
            name: "position_ids_2d".to_string(),
            dtype: DataType::Int64,
            shape: vec![1, -1],
        },
        TensorInfo {
            name: "position_ids_mrope".to_string(),
            dtype: DataType::Int64,
            shape: vec![3, -1, -1],
        },
        TensorInfo {
            name: "position_ids_dynamic_lead".to_string(),
            dtype: DataType::Int64,
            shape: vec![-1, -1, -1],
        },
    ];

    // Rank-2 conventional → 1 axis (the legacy `[1, S]` layout).
    assert_eq!(
        declared_position_rank(&inputs, Some("position_ids_2d")).unwrap(),
        1
    );
    // Rank-3 mrope with a static leading dim → that dim (3 coordinate streams).
    assert_eq!(
        declared_position_rank(&inputs, Some("position_ids_mrope")).unwrap(),
        3
    );
    // No declared position input → 1 (unused, harmless).
    assert_eq!(declared_position_rank(&inputs, None).unwrap(), 1);
    // A rank-3 input with a non-static leading dim cannot be resolved.
    assert!(declared_position_rank(&inputs, Some("position_ids_dynamic_lead")).is_err());
}
