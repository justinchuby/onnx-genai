use super::kv_commit::KvCommitLayout;
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

#[cfg(feature = "cuda")]
struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

#[cfg(feature = "cuda")]
impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: callers must hold `env_lock()` while this guard is live, and
        // the guard restores the value on drop before releasing that lock.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

#[cfg(feature = "cuda")]
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: paired with `EnvVarGuard::set`; this test-only guard restores
        // process environment before the next CUDA smoke test observes it.
        unsafe {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
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
        None, false, false, structural, false, false
    ));
}

#[test]
fn every_decode_level_capture_decline_names_its_predicate() {
    let owned_cuda = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Owned,
    };
    let cases = [
        resolve_graph_capture_decision(None, false, false, owned_cuda, true, false),
        resolve_graph_capture_decision(Some(false), false, false, owned_cuda, false, false),
        resolve_graph_capture_decision(None, true, false, owned_cuda, false, false),
        resolve_graph_capture_decision(
            None,
            false,
            false,
            GraphCaptureStructuralSafety {
                device_is_cuda: false,
                kv_ownership: KvOwnership::Owned,
            },
            false,
            false,
        ),
        resolve_graph_capture_decision(
            None,
            false,
            false,
            GraphCaptureStructuralSafety {
                device_is_cuda: true,
                kv_ownership: KvOwnership::Shared,
            },
            false,
            false,
        ),
    ];

    for decision in cases {
        assert!(!decision.is_enabled());
        let reason = decision
            .decline_reason()
            .expect("a disabled capture decision must always carry a reason");
        assert!(
            reason.starts_with("predicate `"),
            "decline must name its predicate: {reason}"
        );
    }

    let enabled = resolve_graph_capture_decision(None, false, false, owned_cuda, false, false);
    assert!(enabled.is_enabled());
    assert!(enabled.decline_reason().is_none());
}

#[test]
fn weight_offload_forces_graph_capture_off() {
    let safe = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Owned,
    };
    // On the pointer-unstable (non stable-VA) paging path, offload wins over
    // every other signal: auto-safe structure, an explicit env=1, and an
    // explicit programmatic request all still resolve to OFF.
    assert!(!resolve_graph_capture_enabled(
        None, false, false, safe, true, false
    ));
    assert!(!resolve_graph_capture_enabled(
        None, true, true, safe, true, false
    ));
    assert!(!resolve_graph_capture_enabled(
        Some(true),
        true,
        true,
        safe,
        true,
        false
    ));
    // Sanity: with offload disabled the same safe structure enables capture, so
    // the exclusion above is genuinely caused by offload.
    assert!(resolve_graph_capture_enabled(
        None, false, false, safe, false, false
    ));
}

#[test]
fn weight_offload_on_stable_va_path_keeps_graph_capture() {
    // Issue #716: when offload runs on the stable virtual-address VMM paging
    // path, weight page-ins reuse a reserved-once device VA, which is
    // capture-safe. Offload no longer forces capture OFF; the normal precedence
    // (programmatic > env > structural) applies exactly as with offload off.
    let safe = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Owned,
    };
    // Auto-decision from a safe structure now enables capture with offload on.
    assert!(resolve_graph_capture_enabled(
        None, false, false, safe, true, true
    ));
    // An explicit env=1 is honored.
    assert!(resolve_graph_capture_enabled(
        None, true, true, safe, true, true
    ));
    // A programmatic request is honored in both directions.
    assert!(resolve_graph_capture_enabled(
        Some(true),
        false,
        false,
        safe,
        true,
        true
    ));
    assert!(!resolve_graph_capture_enabled(
        Some(false),
        false,
        false,
        safe,
        true,
        true
    ));
    // An unsafe structure still declines, exactly as it would with offload off.
    let unsafe_structure = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Shared,
    };
    assert!(!resolve_graph_capture_enabled(
        None,
        false,
        false,
        unsafe_structure,
        true,
        true
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

#[cfg(feature = "cuda")]
#[test]
fn non_power_of_two_kv_growth_routes_overlapping_copies_through_scratch() {
    let inner = 32usize;
    let elem = 2usize;
    let old_stride = 2048 * inner * elem;
    let new_stride = 3000 * inner * elem;
    let segment = 2048 * inner * elem;

    assert_eq!(
        in_place_copy_route(old_stride, new_stride, segment),
        InPlaceCopyRoute::Scratch,
        "clamped non-power-of-two growth can overlap adjacent KV blocks and must not use cudaMemcpy device-to-device"
    );
    assert_eq!(
        in_place_copy_route(old_stride, old_stride * 2, segment),
        InPlaceCopyRoute::DeviceToDevice,
        "doubling growth keeps adjacent KV block copies disjoint"
    );
}

#[test]
fn graph_capture_auto_declines_for_non_owned_or_non_cuda() {
    let shared = GraphCaptureStructuralSafety {
        device_is_cuda: true,
        kv_ownership: KvOwnership::Shared,
    };
    assert!(!shared.is_capture_safe());
    assert!(!resolve_graph_capture_enabled(
        None, false, false, shared, false, false
    ));

    let cpu_owned = GraphCaptureStructuralSafety {
        device_is_cuda: false,
        kv_ownership: KvOwnership::Owned,
    };
    assert!(!cpu_owned.is_capture_safe());
    assert!(!resolve_graph_capture_enabled(
        None, false, false, cpu_owned, false, false
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
        None, true, false, safe, false, false
    ));
    // ONNX_GENAI_CUDA_GRAPH=1 forces ON even when structure would decline
    // (the runtime decline machinery is still the final safety net).
    assert!(resolve_graph_capture_enabled(
        None,
        true,
        true,
        unsafe_structural,
        false,
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
        false,
        false
    ));
    // Programmatic Some(true) beats explicit env=0.
    assert!(resolve_graph_capture_enabled(
        Some(true),
        true,
        false,
        safe,
        false,
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

/// A tiny hybrid decoder mirroring `tiny_decoder` but adding a recurrent
/// linear-attention layer: layer 0 carries `conv_state`/`recurrent_state`
/// (fixed loop-carried state, threaded past→present by `Identity`), layer 1 is
/// dense GQA (`key`/`value`). All ports use the conventional
/// `past_key_values.%d.*` / `present.%d.*` names, so the loader's graph-derived
/// I/O fallback can classify them without a declared `io:` block.
fn tiny_hybrid_decoder() -> InferenceSession {
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
    // Layer 0: recurrent state ports (fixed-extent, replaced each step).
    let conv_state = graph.create_named_value(
        "past_key_values.0.conv_state",
        DataType::Float32,
        shape(&[batch.into(), 4.into(), 3.into()]),
    );
    let recurrent_state = graph.create_named_value(
        "past_key_values.0.recurrent_state",
        DataType::Float32,
        shape(&[batch.into(), 2.into(), 4.into(), 4.into()]),
    );
    // Layer 1: dense GQA KV ports (growable along seq).
    let past_key = graph.create_named_value(
        "past_key_values.1.key",
        DataType::Float32,
        shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
    );
    let past_value = graph.create_named_value(
        "past_key_values.1.value",
        DataType::Float32,
        shape(&[batch.into(), 1.into(), past.into(), 1.into()]),
    );
    for input in [
        input_ids,
        attention_mask,
        position_ids,
        conv_state,
        recurrent_state,
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

    // Recurrent state: replaced wholesale (Identity pass-through here).
    let present_conv = graph.create_named_value(
        "present.0.conv_state",
        DataType::Float32,
        shape(&[batch.into(), 4.into(), 3.into()]),
    );
    insert_op(&mut graph, "Identity", vec![conv_state], present_conv, &[]);
    let present_recurrent = graph.create_named_value(
        "present.0.recurrent_state",
        DataType::Float32,
        shape(&[batch.into(), 2.into(), 4.into(), 4.into()]),
    );
    insert_op(
        &mut graph,
        "Identity",
        vec![recurrent_state],
        present_recurrent,
        &[],
    );
    // Dense KV: appended along the sequence axis.
    let present_key = graph.create_named_value(
        "present.1.key",
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
        "present.1.value",
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
    for output in [
        logits,
        present_conv,
        present_recurrent,
        present_key,
        present_value,
    ] {
        graph.add_output(output);
    }
    InferenceSession::from_graph(graph).expect("build tiny hybrid decoder")
}

#[test]
fn native_decoder_auto_derives_hybrid_state_io() {
    // A hybrid decoder with NO declared `io:` must load via the graph-derived
    // fallback: token/mask/position/logits bound by conventional name, dense KV
    // for layer 1, and the layer-0 recurrent ports classified as loop-carried
    // state pairs. This is the exact gap that blocked qwen3.x hybrids (#384).
    NativeDecodeSession::from_session(tiny_hybrid_decoder())
        .expect("hybrid decoder auto-derives its decoder-state io from the graph");
}

#[test]
fn native_kv_mirror_gate_excludes_hybrid_recurrent_decoders() {
    // #695: prefix/KV-mirror reuse restores only attention KV. A hybrid
    // recurrent decoder's unmasked conv/recurrent state is NOT reconstructed
    // from a reused prefix (only re-zeroed on a full reset), so a mirrored
    // continuation would run a fresh-zero recurrent state against a reused
    // attention prefix and silently emit wrong logits. The mirror-support gate
    // must therefore decline (forcing a full recompute) whenever recurrent
    // state is present — detected generically from graph metadata, no model
    // name. Single-shot generation never consults these gates, so it is
    // unaffected.
    let hybrid = NativeDecodeSession::from_session(tiny_hybrid_decoder())
        .expect("hybrid decoder loads via graph-derived io");
    assert!(
        hybrid.has_recurrent_state(),
        "tiny_hybrid_decoder carries conv/recurrent state"
    );
    assert!(
        !hybrid.supports_host_kv_mirror(),
        "host KV-mirror reuse must be disabled for hybrid recurrent decoders (#695)"
    );
    assert!(
        !hybrid.supports_device_kv_mirror(),
        "device KV-mirror reuse must be disabled for hybrid recurrent decoders (#695)"
    );

    // Control: a pure-dense rank-4 f32 decoder has no recurrent state, so the
    // host mirror path stays enabled — the gate fires ONLY on recurrent models.
    let dense = NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(
        tiny_decoder(true),
        None,
        Some(&tiny_decoder_io()),
    )
    .expect("dense decoder loads with explicit io");
    assert!(
        !dense.has_recurrent_state(),
        "tiny_decoder is pure-dense attention KV"
    );
    assert!(
        dense.supports_host_kv_mirror(),
        "pure-dense host decoders keep KV-mirror reuse enabled"
    );
}

#[test]
fn native_decoder_auto_derive_skips_dense_ambiguous_decoder() {
    // The fallback's safety gate: a pure-dense decoder derives ZERO state pairs,
    // so auto-derive declines and the model keeps its existing shape-inference
    // path — which still fails on the ambiguous token input exactly as before.
    // This proves the fallback never over-fires on models that load today.
    let error = match NativeDecodeSession::from_session(tiny_decoder(false)) {
        Ok(_) => panic!("dense ambiguous decoder must still require explicit token_input"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("model.io.token_input"),
        "{error:#}"
    );
}

fn target_io(sequence_source: SequenceInputKind) -> ModelIoSpec {
    ModelIoSpec {
        sequence_source: Some(sequence_source),
        kv_ownership: Some(KvOwnership::Owned),
        kv_layout: None,
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
        kv_layout: None,
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
        kv_layout: None,
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
                weight_offload_enabled: None,
                weight_offload_stable_va: None,
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

/// Build a synthetic single-layer CUDA decoder whose `logits` are a direct
/// function of the *incoming* recurrent `conv_state`, and whose next state
/// accumulates the decoded token id. Unlike `build_cuda_decoder_with_fixed_state`
/// (where `conv_state` only passes through `Identity` and never reaches the
/// logits, so a stale state is invisible), here a non-reset recurrent state
/// changes the emitted logits on the next generation — the exact signature of
/// the session-reuse corruption bug. Used by the regression test that a reused
/// `NativeDecodeSession` re-zeros recurrent state on `reset()`.
#[cfg(feature = "cuda")]
fn build_cuda_recurrent_logits_decoder(max_len: usize) -> anyhow::Result<NativeDecodeSession> {
    use prost::Message;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let total = graph.intern_symbol("total");
    let past = graph.intern_symbol("past");

    let input_ids =
        graph.create_named_value("input_ids", DataType::Int64, vec![1.into(), 1.into()]);
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
    let conv_state = graph.create_named_value(
        "past_key_values.0.conv_state",
        DataType::Float16,
        vec![1.into(), 4.into(), 3.into()],
    );
    for input in [
        input_ids,
        attention_mask,
        position_ids,
        past_key,
        past_value,
        conv_state,
    ] {
        graph.add_input(input);
    }

    // logits = ReduceSum(Cast(conv_state)) — depends on the INCOMING recurrent
    // state, so a stale (non-zeroed) conv_state changes the emitted logits. Cast
    // to f32 first: cuDNN reduction does not support Float16 inputs.
    let conv_f32 = graph.create_named_value(
        "conv_state_f32",
        DataType::Float32,
        vec![1.into(), 4.into(), 3.into()],
    );
    insert_op(
        &mut graph,
        "Cast",
        vec![conv_state],
        conv_f32,
        &[("to", Attribute::Int(DataType::Float32 as i64))],
    );
    let logits = graph.create_named_value(
        "logits",
        DataType::Float32,
        vec![1.into(), 1.into(), 1.into()],
    );
    insert_op(
        &mut graph,
        "ReduceSum",
        vec![conv_f32],
        logits,
        &[("keepdims", Attribute::Int(1))],
    );

    // present.0.conv_state = conv_state + Cast(input_ids) — accumulate the token
    // id into every recurrent slot so the state grows deterministically per step.
    let token_f16 =
        graph.create_named_value("token_f16", DataType::Float16, vec![1.into(), 1.into()]);
    insert_op(
        &mut graph,
        "Cast",
        vec![input_ids],
        token_f16,
        &[("to", Attribute::Int(DataType::Float16 as i64))],
    );
    let present_conv_state = graph.create_named_value(
        "present.0.conv_state",
        DataType::Float16,
        vec![1.into(), 4.into(), 3.into()],
    );
    insert_op(
        &mut graph,
        "Add",
        vec![conv_state, token_f16],
        present_conv_state,
        &[],
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

    for output in [logits, present_key, present_value, present_conv_state] {
        graph.add_output(output);
    }

    let model = onnx_std::Model::new(graph).to_proto()?.encode_to_vec();
    let session = InferenceSession::builder()
        .model_bytes(&model)
        .device(DevicePreference::Gpu { index: Some(0) })
        .build()
        .context("build recurrent-logits CUDA decoder")?;
    let mut io = tiny_decoder_io();
    io.state_pairs = Some(vec![LoopStatePair {
        input: "past_key_values.0.conv_state".into(),
        output: "present.0.conv_state".into(),
        init: Some("zeros".into()),
        update: Some("replace".into()),
    }]);
    NativeDecodeSession::from_session_with_cuda_kv_max_len_and_io(session, Some(max_len), Some(&io))
}

/// Regression guard for the native-CUDA session-reuse corruption bug: a reused
/// `NativeDecodeSession` must re-zero fixed recurrent/conv state on `reset()`,
/// so generation #2 starts from the declared `init: zeros` — not generation #1's
/// terminal state. Before the fix, `rewind(0)` left the recurrent binding stale
/// and the second decode sequence produced different logits (non-deterministic
/// degenerate output on hybrid LinearAttention models). This test is
/// non-vacuous: the logits are a direct function of the incoming recurrent
/// state, so a stale state changes the asserted values.
#[cfg(feature = "cuda")]
#[test]
fn native_cuda_reused_session_rezeros_recurrent_state() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }

    const TOKENS: [TokenId; 3] = [5, 7, 9];
    let mut session = build_cuda_recurrent_logits_decoder(16)?;

    let run_once = |session: &mut NativeDecodeSession| -> anyhow::Result<Vec<f32>> {
        session.reset()?;
        let mut logits = Vec::new();
        for (position, token) in TOKENS.iter().enumerate() {
            let step = session.decode(&[*token], position)?;
            logits.push(step[0][0]);
        }
        Ok(logits)
    };

    let first = run_once(&mut session)?;
    // The incoming recurrent state accumulates the decoded token ids, so the
    // per-step logits form a strictly growing sequence — proving the state does
    // flow into the logits (guards against a vacuous test).
    assert_eq!(
        first[0], 0.0,
        "generation must start from zeroed recurrent state"
    );
    assert!(
        first[1] > first[0] && first[2] > first[1],
        "recurrent state must accumulate into logits: {first:?}"
    );

    let second = run_once(&mut session)?;
    assert_eq!(
        first, second,
        "reused session must re-zero recurrent state on reset(): gen#1 {first:?} != gen#2 {second:?}"
    );
    Ok(())
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
        1,
    )
    .expect("numeric auxiliary output must be bindable");
    assert_eq!(shape, [1, 1536]);

    let error = DecodeCudaState::persistent_output_shape(
        "auxiliary_text",
        DataType::String,
        &[Dim::Static(1)],
        1,
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
        1,
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
        1,
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
        1,
        512,
        true,
    )
    .expect("rank-4 fixed recurrent state");
    assert_eq!(recurrent_physical, [1, 48, 128, 128]);
    assert_eq!(recurrent_logical, recurrent_physical);
}

#[test]
fn cuda_persistent_state_shapes_thread_batch_axis_0() {
    // Stage 2b-impl-1 (#750): the batch extent is threaded to axis 0 rather than
    // hard-coded to 1. The BNSH axis order must not move: axis 0 is batch, axis
    // 2 is the grown seq capacity, regardless of `batch`.
    let batch_sym = SymbolId(0);
    let past = SymbolId(1);
    let (kv_physical, kv_logical) = DecodeCudaState::persistent_state_shapes(
        "past_key_values.0.key",
        DataType::Float16,
        &[
            Dim::Symbolic(batch_sym),
            Dim::Static(4),
            Dim::Symbolic(past),
            Dim::Static(256),
        ],
        3,
        512,
        false,
    )
    .expect("rank-4 growing KV threads batch");
    assert_eq!(kv_physical, [3, 4, 512, 256]);
    assert_eq!(kv_logical, [3, 4, 0, 256]);

    // Fixed recurrent state threads batch on axis 0 too.
    let (recurrent_physical, recurrent_logical) = DecodeCudaState::persistent_state_shapes(
        "past_key_values.0.recurrent_state",
        DataType::Float16,
        &[
            Dim::Symbolic(batch_sym),
            Dim::Static(48),
            Dim::Static(128),
            Dim::Static(128),
        ],
        3,
        512,
        true,
    )
    .expect("rank-4 fixed recurrent state threads batch");
    assert_eq!(recurrent_physical, [3, 48, 128, 128]);
    assert_eq!(recurrent_logical, recurrent_physical);
}

#[test]
fn persistent_output_shape_threads_batch_only_on_axis_0() {
    // The batch axis (0) takes the threaded extent; every other symbolic axis is
    // a query-seq unit collapsed to 1. Static axes (e.g. vocab) are preserved.
    let batch = SymbolId(0);
    let seq = SymbolId(1);
    let shape = DecodeCudaState::persistent_output_shape(
        "logits",
        DataType::Float32,
        &[
            Dim::Symbolic(batch),
            Dim::Symbolic(seq),
            Dim::Static(151_936),
        ],
        3,
    )
    .expect("logits shape threads batch");
    assert_eq!(shape, [3, 1, 151_936]);

    // At batch 1 (every current caller) it is byte-identical to the historical
    // collapse-every-symbolic-to-1 behavior.
    let shape_batch_1 = DecodeCudaState::persistent_output_shape(
        "logits",
        DataType::Float32,
        &[
            Dim::Symbolic(batch),
            Dim::Symbolic(seq),
            Dim::Static(151_936),
        ],
        1,
    )
    .expect("logits shape at batch 1");
    assert_eq!(shape_batch_1, [1, 1, 151_936]);
}

#[test]
fn kv_growth_byte_layout_keeps_head_major_bnsh_and_permutes_seq_major_to_bsnh() {
    // BNSH declared shape [batch, kv_heads, capacity, head_dim].
    let bnsh = [1usize, 8, 512, 128];

    // Head-major must be byte-identical to the historical behavior: the byte
    // layout equals the declared shape and the grow axis is 2.
    let (head_bytes, head_axis) =
        kv_growth_byte_layout(&bnsh, KvCommitLayout::HeadMajor).expect("head-major layout");
    assert_eq!(head_bytes, bnsh.to_vec());
    assert_eq!(head_axis, 2);

    // Seq-major stores bytes as BSNH [batch, capacity, kv_heads, head_dim] and
    // grows on axis 1.
    let (seq_bytes, seq_axis) =
        kv_growth_byte_layout(&bnsh, KvCommitLayout::SeqMajor).expect("seq-major layout");
    assert_eq!(seq_bytes, vec![1, 512, 8, 128]);
    assert_eq!(seq_axis, 1);

    // Rank must be four.
    assert!(kv_growth_byte_layout(&[1, 8, 512], KvCommitLayout::SeqMajor).is_err());
}

// The defining property of the seq-major buffer: the per-token stride is
// `kv_heads * head_dim` and is independent of the sequence capacity, so growing
// the capacity does not move any live byte. Head-major's per-head stripe stride
// scales with capacity, so growth re-strides every head but head 0.
#[test]
fn seq_major_growth_is_fixed_stride_and_moves_no_bytes() {
    // Mirror the block/stride arithmetic of `copy_kv_prefix_device_to_device*`:
    // for a byte-layout shape and grow axis, block `b` starts at
    // `b * capacity * inner` elements, where `inner` is the product of the axes
    // below the grow axis.
    fn block_bases(bnsh: &[usize], layout: KvCommitLayout) -> (Vec<usize>, usize) {
        let (bytes, axis) = kv_growth_byte_layout(bnsh, layout).expect("byte layout");
        let capacity = bytes[axis];
        let blocks: usize = bytes[..axis].iter().product();
        let inner: usize = bytes[axis + 1..].iter().product();
        let bases = (0..blocks).map(|b| b * capacity * inner).collect();
        (bases, inner)
    }

    let old = [1usize, 8, 256, 128];
    let new = [1usize, 8, 512, 128];

    // Seq-major: one block (batch), fixed inner stride kv_heads*head_dim, and the
    // single block base is 0 for both capacities, so the live prefix keeps its
    // byte offsets — the in-place copy is a no-op.
    let (seq_old_bases, seq_inner) = block_bases(&old, KvCommitLayout::SeqMajor);
    let (seq_new_bases, seq_inner_new) = block_bases(&new, KvCommitLayout::SeqMajor);
    assert_eq!(seq_inner, 8 * 128);
    assert_eq!(
        seq_inner_new,
        8 * 128,
        "per-token stride is capacity-independent"
    );
    assert_eq!(
        seq_old_bases, seq_new_bases,
        "no seq-major block base moves on growth"
    );
    assert_eq!(seq_old_bases, vec![0]);

    // Head-major: one block per head, and every head but head 0 moves to a wider
    // stride when the capacity grows.
    let (head_old_bases, head_inner) = block_bases(&old, KvCommitLayout::HeadMajor);
    let (head_new_bases, _) = block_bases(&new, KvCommitLayout::HeadMajor);
    assert_eq!(head_inner, 128);
    assert_ne!(
        head_old_bases, head_new_bases,
        "head-major re-strides on growth"
    );
    for head in 1..8 {
        assert!(head_new_bases[head] > head_old_bases[head]);
    }
}

// Batch-N control (stage 2b-impl-2, #750): head-major growing-bucket growth is
// batch-general, seq-major growing-bucket growth at batch>1 is an explicit named
// refusal (it would relocate every sequence b>0). This asserts the geometry math
// at batch>1 even though runtime batch stays fixed to 1 — a transposed axis or a
// silently wrong capacity-dependent stride could not surface at batch=1.
#[test]
fn kv_growth_byte_layout_refuses_seq_major_batch_n_and_allows_head_major_batch_n() {
    // Head-major: batch axis outermost, byte layout == declared shape, grow axis
    // 2, block count = batch * kv_heads — each (batch, head) stripe re-strides
    // independently and correctly, so batch-N growth is exact.
    let head_batch_n = [3usize, 8, 512, 128];
    let (bytes, axis) = kv_growth_byte_layout(&head_batch_n, KvCommitLayout::HeadMajor)
        .expect("head-major batch-N growing bucket is batch-general");
    assert_eq!(bytes, head_batch_n.to_vec());
    assert_eq!(axis, 2);
    let blocks: usize = bytes[..axis].iter().product();
    assert_eq!(blocks, 3 * 8, "head-major blocks = batch * kv_heads");

    // Seq-major growing bucket at batch>1 is refused with a named error rather
    // than a silently wrong capacity-dependent stride.
    let seq_batch_n = [3usize, 8, 512, 128];
    let error = kv_growth_byte_layout(&seq_batch_n, KvCommitLayout::SeqMajor)
        .expect_err("seq-major batch-N growing bucket must be refused");
    let message = error.to_string();
    assert!(message.contains("seq-major"), "names the layout: {message}");
    assert!(message.contains("batch 3"), "names the batch: {message}");
    assert!(
        message.contains("relocat"),
        "names the constraint: {message}"
    );

    // Batch-1 seq-major is unaffected: byte-identical BSNH permutation, grow
    // axis 1.
    let seq_batch_1 = [1usize, 8, 512, 128];
    let (seq_bytes, seq_axis) = kv_growth_byte_layout(&seq_batch_1, KvCommitLayout::SeqMajor)
        .expect("seq-major batch-1 unaffected");
    assert_eq!(seq_bytes, vec![1, 512, 8, 128]);
    assert_eq!(seq_axis, 1);
}

#[test]
fn cuda_fixed_state_shapes_reject_unbounded_non_batch_dimensions() {
    let error = DecodeCudaState::persistent_state_shapes(
        "state",
        DataType::Float16,
        &[Dim::Symbolic(SymbolId(0)), Dim::Symbolic(SymbolId(1))],
        1,
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
    let reason = stats
        .graph
        .decline_reason
        .as_deref()
        .expect("a binding-time capture decline must be observable");
    assert!(
        reason.contains("predicate `auxiliary_outputs_have_fixed_persistent_shapes`"),
        "{reason}"
    );

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
            weight_offload_enabled: None,
            weight_offload_stable_va: None,
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
            weight_offload_enabled: None,
            weight_offload_stable_va: None,
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

/// VMM-backed CUDA KV must reserve full context while committing only the
/// buckets the sequence can touch.
///
/// This is the wiring test the allocator-only committed-range test cannot be.
/// It reads committed bytes from the KV bindings themselves, not from global VMM
/// counters that unrelated workspaces can move. Sabotaging the initial committed
/// range or dropping `committed_ranges` in the plumbing makes the load-time KV
/// commitment equal the full-context reservation and this test fails; sabotaging
/// `commits_on_demand()` makes growth replace the bindings and this test fails.
#[cfg(feature = "cuda")]
#[test]
fn native_cuda_vmm_kv_grows_in_place_and_commits_more_granules() -> anyhow::Result<()> {
    if std::env::var_os("ONNX_GENAI_RUN_CUDA_SMOKE").is_none() {
        eprintln!("skipping CUDA smoke; set ONNX_GENAI_RUN_CUDA_SMOKE=1 to run");
        return Ok(());
    }
    let _guard = env_lock()
        .lock()
        .expect("CUDA smoke env mutex should not be poisoned");

    let Some(model_dir) = qwen_cuda_smoke_model_dir() else {
        return Ok(());
    };
    let _vmm = EnvVarGuard::set("ONNX_GENAI_CUDA_VMM", "1");
    let _graph = EnvVarGuard::set("ONNX_GENAI_CUDA_GRAPH", "0");
    let _kv_max = EnvVarGuard::set("ONNX_GENAI_CUDA_KV_MAX_LEN", "32768");
    onnx_runtime_ep_cuda::vmm_allocator::reset_global_vmm_stats();

    let mut session = NativeDecodeSession::load_with_resolved_io(
        model_dir.join("model.onnx"),
        NativeDecodeDevice::Cuda { index: Some(0) },
    )?;
    let before = session
        .cuda_kv_debug_stats()
        .context("CUDA session must expose KV stats")?;
    let full_kv = usize::try_from(session.kv_reservation(before.hard_max_len)?.0)
        .context("full KV reservation fits in usize")?;
    let binding_count = before.device_ptrs.len();
    let granule = before
        .kv_committed_bytes
        .checked_div(binding_count)
        .context("CUDA KV stats must include at least one binding")?;
    let expected_committed = |stats: &CudaKvDebugStats| -> usize {
        stats
            .device_ptrs
            .iter()
            .zip(stats.kv_physical_bytes_by_binding.iter())
            .map(|(&ptr, &bytes)| {
                if bytes == 0 {
                    0
                } else {
                    ((ptr % granule) + bytes).div_ceil(granule) * granule
                }
            })
            .sum()
    };
    let expected_physical_bytes =
        |stats: &CudaKvDebugStats| -> usize { stats.kv_physical_bytes_by_binding.iter().sum() };
    assert_eq!(
        before.kv_committed_bytes,
        expected_committed(&before),
        "load must commit exactly the rounded physical KV bucket, not the full reserved context"
    );
    assert!(
        expected_physical_bytes(&before) < full_kv / 2,
        "VMM KV should expose only the initial physical bucket at load: physical={} full_context={full_kv}",
        expected_physical_bytes(&before)
    );
    assert!(
        before.kv_committed_bytes < full_kv,
        "VMM KV should commit far less than the full reserved context at load: committed={} full_context={full_kv}",
        before.kv_committed_bytes
    );

    let required = 8193;
    assert!(
        before.hard_max_len >= required,
        "CUDA smoke fixture must allow growth to {required} tokens; hard max is {}",
        before.hard_max_len
    );
    session
        .cuda
        .as_mut()
        .context("CUDA state must be present")?
        .ensure_capacity(&mut session.session, required)?;

    let after = session.cuda_kv_debug_stats().unwrap();
    let expected_delta = expected_committed(&after) - expected_committed(&before);
    assert_eq!(
        before.device_ptrs, after.device_ptrs,
        "VMM KV growth must commit the existing virtual range, not replace bindings"
    );
    assert_eq!(
        after.kv_committed_bytes - before.kv_committed_bytes,
        expected_delta,
        "growth must commit exactly the next KV bucket delta, not global workspace traffic"
    );
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
                weight_offload_enabled: None,
                weight_offload_stable_va: None,
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

// ---------------------------------------------------------------------------
// Inc-1b PR-2: decode-inline routing decision (Harry guard #2 — runtime
// scan-axis extent==1 fallback). The plan is graph-property gated (an
// inlineable single-trip recurrent `Scan`), not a user flag; the with-Scan →
// sibling-built and dense → no-sibling gate is covered directly by the session
// crate's `decode_inline_sibling_folds_body_into_captured_graph_byte_exact` and
// `decode_inline_sibling_none_for_dense_graph` executor tests.
// ---------------------------------------------------------------------------

#[test]
fn decode_inline_routes_only_single_token_when_enabled() {
    // Enabled + sibling ready + single token + no eager step inputs → route to
    // the decode-inline exec.
    assert!(route_decode_inline_decision(
        DecodeInlineState::Enabled,
        true,
        1,
        false
    ));
    // Guard #2: a multi-token (extent≠1) step must fall back to the main Scan
    // executor even when the sibling is ready — never run a collapsed graph.
    assert!(!route_decode_inline_decision(
        DecodeInlineState::Enabled,
        true,
        2,
        false
    ));
    assert!(!route_decode_inline_decision(
        DecodeInlineState::Enabled,
        true,
        8,
        false
    ));
    // A zero-token step is not a decode step.
    assert!(!route_decode_inline_decision(
        DecodeInlineState::Enabled,
        true,
        0,
        false
    ));
}

/// PR-3 scope-lock (Harry #588 rec #4): a decoder that declares
/// `inputs_embeds`/routed step-input ports is NEVER routed to the decode-inline
/// sibling, even when it is otherwise a perfect candidate (Enabled + ready +
/// single token). This locks the author's stated scope against future dispatch
/// drift on both `decode_cuda` and the `decode_cuda_greedy` fast path.
#[test]
fn decode_inline_never_routes_when_decoder_has_eager_step_inputs() {
    // The exact single-token candidate that WOULD route (mutation baseline)...
    assert!(route_decode_inline_decision(
        DecodeInlineState::Enabled,
        true,
        1,
        false
    ));
    // ...is refused the instant the decoder declares eager step inputs.
    assert!(
        !route_decode_inline_decision(DecodeInlineState::Enabled, true, 1, true),
        "inputs_embeds/routed decoders must never reach the decode-inline sibling"
    );
    // Belt-and-suspenders across token counts.
    for count in [1usize, 2, 8] {
        assert!(!route_decode_inline_decision(
            DecodeInlineState::Enabled,
            true,
            count,
            true
        ));
    }
}

#[test]
fn decode_inline_never_routes_when_disabled_or_unbuilt() {
    // Untried / Disabled never route, regardless of token count.
    for state in [DecodeInlineState::Untried, DecodeInlineState::Disabled] {
        assert!(!route_decode_inline_decision(state, true, 1, false));
        assert!(!route_decode_inline_decision(state, false, 1, false));
    }
    // Enabled but no sibling actually built (defensive): never route.
    assert!(!route_decode_inline_decision(
        DecodeInlineState::Enabled,
        false,
        1,
        false
    ));
}

/// The native CUDA KV-cache layout is resolved from the model's declared
/// `kv_layout` descriptor, with an environment override for residency
/// diagnostics. Absent metadata means head-major — exactly the historical
/// behavior — so no currently-loadable model changes its committed geometry.
#[cfg(feature = "cuda")]
#[test]
fn cuda_kv_layout_resolves_from_metadata_with_env_override() {
    use super::cuda::resolve_cuda_kv_layout;
    use super::kv_commit::KvCommitLayout;
    use onnx_genai_metadata::KvCacheLayout;

    let _guard = env_lock().lock().unwrap();

    // Absent metadata → head-major (the historical default, byte-identical).
    {
        let _env = EnvVarGuard::set("ONNX_GENAI_CUDA_KV_LAYOUT", "");
        // A truly-empty value is not one of the recognized tokens, so it is
        // ignored and the metadata (here `None`) decides.
        unsafe {
            std::env::remove_var("ONNX_GENAI_CUDA_KV_LAYOUT");
        }
        assert_eq!(resolve_cuda_kv_layout(None), KvCommitLayout::HeadMajor);
        assert_eq!(
            resolve_cuda_kv_layout(Some(&KvCacheLayout::head_major_bnsh())),
            KvCommitLayout::HeadMajor
        );
        // A declared seq-major descriptor selects seq-major.
        assert_eq!(
            resolve_cuda_kv_layout(Some(&KvCacheLayout::seq_major_bsnh())),
            KvCommitLayout::SeqMajor
        );
    }

    // The env override wins over the descriptor, in both directions, so a
    // residency measurement can pin the layout regardless of the model.
    {
        let _env = EnvVarGuard::set("ONNX_GENAI_CUDA_KV_LAYOUT", "seq_major");
        assert_eq!(
            resolve_cuda_kv_layout(Some(&KvCacheLayout::head_major_bnsh())),
            KvCommitLayout::SeqMajor
        );
    }
    {
        let _env = EnvVarGuard::set("ONNX_GENAI_CUDA_KV_LAYOUT", "head_major");
        assert_eq!(
            resolve_cuda_kv_layout(Some(&KvCacheLayout::seq_major_bsnh())),
            KvCommitLayout::HeadMajor
        );
    }
    // An unrecognized override token is ignored (falls back to the descriptor).
    {
        let _env = EnvVarGuard::set("ONNX_GENAI_CUDA_KV_LAYOUT", "nonsense");
        assert_eq!(
            resolve_cuda_kv_layout(Some(&KvCacheLayout::seq_major_bsnh())),
            KvCommitLayout::SeqMajor
        );
    }
}

/// Fixed recurrent state is sized from the graph, per sequence.
///
/// It is a per-sequence cost that scales exactly the way KV does, and it was
/// invisible to the thing deciding how many sequences fit -- so the governor
/// could admit a batch that does not fit, and the failure landed as an
/// allocation error mid-generation rather than a refusal at admission.
///
/// The fixture's layer 0 carries `conv_state [batch, 4, 3]` and
/// `recurrent_state [batch, 2, 4, 4]`: 12 + 32 = 44 f32 elements, 176 bytes,
/// with the batch axis counted as one because the figure is per sequence and
/// the scheduler multiplies.
#[test]
fn recurrent_state_is_sized_per_sequence_from_the_graph() {
    let session = tiny_hybrid_decoder();
    // The decoder declares layer 0's state pair; layer 1 is dense KV.
    let declared = std::collections::HashMap::from([
        (
            "present.0.conv_state".to_string(),
            "past_key_values.0.conv_state".to_string(),
        ),
        (
            "present.0.recurrent_state".to_string(),
            "past_key_values.0.recurrent_state".to_string(),
        ),
    ]);
    let bytes =
        crate::native_decode::tensor::recurrent_state_bytes_per_sequence(&session, &declared)
            .expect("the fixture pins every non-batch axis");
    assert_eq!(bytes, 176, "12 + 32 f32 elements is 176 bytes");
}

/// A decoder with no recurrent layers asks for nothing.
///
/// Most decoders are this, and it must not read as a failure or as a
/// reservation of zero that implies something was measured.
#[test]
fn a_decoder_without_recurrent_layers_needs_no_recurrent_state() {
    let session = tiny_decoder(false);
    let declared = std::collections::HashMap::new();
    let bytes =
        crate::native_decode::tensor::recurrent_state_bytes_per_sequence(&session, &declared)
            .expect("a dense decoder is sizeable");
    assert_eq!(bytes, 0);
}

/// KV is sized at full context, per sequence, from the declared pairs.
///
/// The native path's page table carries no storage, so nothing else leases
/// this: without it the ledger was missing the largest per-sequence cost the
/// decoder has, and every tier total read low by that amount.
///
/// The fixture's layer 1 carries `key` and `value`, each `[batch, 1, past, 1]`
/// f32. At 128 tokens that is 128 elements each, 512 bytes each, 1024 together,
/// with the batch axis counted as one because the figure is per sequence.
#[test]
fn kv_is_sized_at_full_context_per_sequence() {
    let session = tiny_hybrid_decoder();
    let declared = std::collections::HashMap::from([
        (
            "present.1.key".to_string(),
            "past_key_values.1.key".to_string(),
        ),
        (
            "present.1.value".to_string(),
            "past_key_values.1.value".to_string(),
        ),
    ]);
    let bytes = crate::native_decode::tensor::kv_cache_bytes_per_sequence(&session, &declared, 128)
        .expect("the fixture pins every axis but the growable one");
    assert_eq!(bytes, 1024, "2 tensors x 128 f32 elements is 1024 bytes");
}

/// Recurrent state is not charged again as KV.
///
/// Both are loop-carried and both appear in `present_to_past`, so a helper that
/// merely walked the declared pairs would charge the recurrent tensors twice --
/// once at their real fixed size and once multiplied by the context length,
/// which for a long context is wrong by orders of magnitude and in the
/// direction that refuses models that fit.
#[test]
fn recurrent_state_is_not_charged_as_kv() {
    let session = tiny_hybrid_decoder();
    // Everything the hybrid decoder declares: layer 0's state pair *and*
    // layer 1's KV.
    let all = std::collections::HashMap::from([
        (
            "present.0.conv_state".to_string(),
            "past_key_values.0.conv_state".to_string(),
        ),
        (
            "present.0.recurrent_state".to_string(),
            "past_key_values.0.recurrent_state".to_string(),
        ),
        (
            "present.1.key".to_string(),
            "past_key_values.1.key".to_string(),
        ),
        (
            "present.1.value".to_string(),
            "past_key_values.1.value".to_string(),
        ),
    ]);
    let bytes = crate::native_decode::tensor::kv_cache_bytes_per_sequence(&session, &all, 128)
        .expect("the fixture is sizeable");
    assert_eq!(
        bytes, 1024,
        "only layer 1's KV counts; layer 0's state is charged at its own fixed size"
    );
}

/// A decoder that declares no loop-carried tensors asks for no KV.
///
/// Zero here is a fact about the graph, not a failure and not an unmeasured
/// value standing in for one.
#[test]
fn a_decoder_declaring_no_pairs_needs_no_kv_reservation() {
    let session = tiny_hybrid_decoder();
    let none = std::collections::HashMap::new();
    let bytes = crate::native_decode::tensor::kv_cache_bytes_per_sequence(&session, &none, 4096)
        .expect("an empty declaration is sizeable");
    assert_eq!(bytes, 0);
}

/// An input that merely looks like recurrent state is not charged as it.
///
/// `is_recurrent_state_shape` only asks whether the penultimate axis is static,
/// which is also true of a fixed-length KV input and of any unrelated
/// fixed-shape input. Discovering state by that test alone would charge memory
/// the decoder never keeps -- and this reservation refuses a load when it does
/// not fit, so an over-count rejects models that work.
///
/// The declared pairs are the authority; the shape test only classifies what
/// they name.
#[test]
fn an_undeclared_input_of_the_same_shape_is_not_charged() {
    let session = tiny_hybrid_decoder();
    let declared_none = std::collections::HashMap::new();
    assert_eq!(
        crate::native_decode::tensor::recurrent_state_bytes_per_sequence(&session, &declared_none)
            .expect("sizeable"),
        0,
        "nothing is declared, so nothing is state, whatever the shapes look like"
    );

    // Declaring only the conv pair charges only the conv pair: 4 * 3 f32 = 48.
    let conv_only = std::collections::HashMap::from([(
        "present.0.conv_state".to_string(),
        "past_key_values.0.conv_state".to_string(),
    )]);
    assert_eq!(
        crate::native_decode::tensor::recurrent_state_bytes_per_sequence(&session, &conv_only)
            .expect("sizeable"),
        48,
        "the recurrent_state input is present in the graph but was not declared"
    );
}
