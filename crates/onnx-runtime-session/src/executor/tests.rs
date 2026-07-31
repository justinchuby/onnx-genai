use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use onnx_runtime_ep_api::{
    CaptureSupport, Cost, EpConfig, EpError, ExecutionProviderCapabilities, Fence, Kernel,
    NegotiatedWeight,
};

use super::*;

#[test]
fn phase_profile_gating_and_accumulation() {
    // Single test (not two) so the process-global enable flag is never
    // toggled concurrently by a sibling test under the parallel runner.

    // Disabled: a span records nothing and never captures a timestamp.
    phase_profile::force_enabled(false);
    let disabled_phase = "test.phase.disabled";
    let before = phase_profile::snapshot(disabled_phase);
    {
        let _s = phase_span!(disabled_phase);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        phase_profile::snapshot(disabled_phase),
        before,
        "a disabled phase span must not accumulate any samples"
    );

    // Enabled: a span accumulates exactly one positive-duration sample.
    phase_profile::force_enabled(true);
    let enabled_phase = "test.phase.enabled";
    let (base_ns, base_count) = phase_profile::snapshot(enabled_phase).unwrap_or((0, 0));
    {
        let _s = phase_span!(enabled_phase);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let (after_ns, after_count) =
        phase_profile::snapshot(enabled_phase).expect("enabled span must record a sample");
    assert_eq!(after_count, base_count + 1, "one span => one sample");
    assert!(
        after_ns > base_ns,
        "an enabled span must accumulate a positive duration"
    );

    // Restore the default (disabled) state so other tests stay inert.
    phase_profile::force_enabled(false);
}

// A produced top-level output is handed to the caller by moving its host
// buffer out of the executor (zero-copy), while a producer-less output (an
// initializer routed straight to a graph output) stays on the copy path so
// its borrowed/shared storage is never freed. Repeated runs must keep
// producing correct bytes even though the produced buffer was moved out and
// its `buffer_shapes` entry cleared (forcing a fresh allocation each run).
#[test]
fn zero_copy_output_move_reallocates_and_preserves_producer_less_output() {
    use onnx_runtime_ir::TensorData;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let a = graph.create_named_value("a", DataType::Float32, static_shape([3]));
    let b = graph.create_named_value("b", DataType::Float32, static_shape([3]));
    graph.add_input(a);
    graph.add_input(b);

    // Producer-less graph output: an initializer wired straight to an output.
    let k = graph.create_named_value("k", DataType::Float32, static_shape([3]));
    graph.set_initializer(
        k,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![3],
            [100.0f32, 200.0, 300.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );

    // Produced output: Add(a, b). This is the movable, owned, host output.
    let sum = graph.create_named_value("sum", DataType::Float32, static_shape([3]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(a), Some(b)],
        vec![sum],
    ));
    graph.add_output(sum);
    graph.add_output(k);

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let a_val = Tensor::from_f32(&[3], &[1.0, 2.0, 3.0]).unwrap();
    let b_val = Tensor::from_f32(&[3], &[10.0, 20.0, 30.0]).unwrap();

    // Three runs prove the move-out survives buffer reallocation: run 1 moves
    // `sum`'s buffer out, so runs 2 and 3 must reallocate it from scratch.
    for _ in 0..3 {
        let outputs = executor
            .run(&[("a", &a_val), ("b", &b_val)])
            .expect("run must succeed after a prior output buffer was moved out");
        assert_eq!(outputs[0].to_vec_f32(), vec![11.0, 22.0, 33.0]);
        assert_eq!(
            outputs[1].to_vec_f32(),
            vec![100.0, 200.0, 300.0],
            "producer-less initializer output must stay intact across runs"
        );
        // The produced output was handed off zero-copy: its buffer is gone
        // from the executor. The initializer stays resident (copy path).
        assert!(
            !executor.buffers.contains_key(&sum),
            "produced output buffer must be moved out, not copied"
        );
        assert!(
            executor.buffers.contains_key(&k),
            "producer-less output must not have its buffer stolen"
        );
    }
}

fn inplace_chain_graph(keep_input_output: bool, keep_input_live: bool) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("input", DataType::Float32, static_shape([4]));
    graph.add_input(input);
    let first = graph.create_named_value("first", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Tanh", vec![Some(input)], vec![first]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([4]));
    if keep_input_live {
        graph.insert_node(Node::new(
            NodeId(1),
            "Add",
            vec![Some(input), Some(first)],
            vec![output],
        ));
    } else {
        graph.insert_node(Node::new(
            NodeId(1),
            "Tanh",
            vec![Some(first)],
            vec![output],
        ));
    }
    if keep_input_output {
        graph.add_output(input);
        graph.add_output(first);
    }
    graph.add_output(output);
    graph
}

#[test]
fn compute_in_place_chain_is_byte_identical_and_fires() {
    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    let weights = Arc::new(WeightStore::new());
    let ep = auto_detect_cpu_ep().unwrap();
    let mut enabled = Executor::build(
        inplace_chain_graph(false, false),
        Arc::clone(&weights),
        Arc::clone(&ep),
    )
    .unwrap();
    let enabled_output = enabled.run(&[("input", &values)]).unwrap()[0]
        .as_bytes()
        .to_vec();
    assert_eq!(enabled.compute_in_place_alias_count, 2);

    let mut disabled = Executor::build(inplace_chain_graph(false, false), weights, ep).unwrap();
    disabled.compute_in_place_enabled = false;
    let disabled_output = disabled.run(&[("input", &values)]).unwrap()[0]
        .as_bytes()
        .to_vec();
    assert_eq!(disabled.compute_in_place_alias_count, 0);
    assert_eq!(enabled_output, disabled_output);
}

#[test]
fn compute_in_place_refuses_live_and_graph_output_inputs() {
    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    for (keep_input_output, keep_input_live) in [(true, false), (false, true)] {
        let mut executor = Executor::build(
            inplace_chain_graph(keep_input_output, keep_input_live),
            Arc::new(WeightStore::new()),
            auto_detect_cpu_ep().unwrap(),
        )
        .unwrap();
        let outputs = executor.run(&[("input", &values)]).unwrap();
        assert_eq!(executor.compute_in_place_alias_count, 0);
        if keep_input_output {
            assert_eq!(outputs[0].as_bytes(), values.as_bytes());
        }
    }
}

/// An `If` branch body of `branch_out = Identity(h)`, capturing the outer value
/// `h` by name (a producer-less named value bound from the enclosing scope).
fn identity_capture_body() -> Graph {
    let mut body = Graph::new();
    body.opset_imports.insert(String::new(), 17);
    let captured = body.create_named_value("h", DataType::Float32, static_shape([4]));
    let out = body.create_named_value("branch_out", DataType::Float32, static_shape([4]));
    body.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(captured)],
        vec![out],
    ));
    body.add_output(out);
    body
}

/// `h = Relu(x); t = Tanh(h); y = If(cond){ Identity(h) }`. The intermediate `h`
/// has exactly one *formal* consumer (the in-place-eligible `Tanh`, whose output
/// `t` is dead), but both `If` branches capture `h` by name afterwards. Liveness
/// that ignores control-flow captures would mark `h` dead at `Tanh` and alias its
/// buffer away, so the later capture would read freed memory.
fn inplace_capture_if_graph() -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);
    let cond = graph.create_named_value("cond", DataType::Bool, static_shape([1]));
    graph.add_input(cond);
    let h = graph.create_named_value("h", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![h]));
    let t = graph.create_named_value("t", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(1), "Tanh", vec![Some(h)], vec![t]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(2), "If", vec![Some(cond)], vec![y]));
    graph.subgraphs.insert(
        (NodeId(2), "then_branch".to_string()),
        identity_capture_body(),
    );
    graph.subgraphs.insert(
        (NodeId(2), "else_branch".to_string()),
        identity_capture_body(),
    );
    graph.add_output(y);
    graph
}

/// Regression (issue #85): a value captured by an `If`/`Loop`/`Scan` body must
/// never be aliased away by an earlier in-place op, even though the capture is
/// implicit (by name) and absent from every plan node's formal `inputs`. Asserts
/// the default (enabled) path succeeds, returns `Relu(x)`, and is byte-identical
/// to the fully out-of-place reference path.
#[test]
fn compute_in_place_preserves_control_flow_captures() {
    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    let cond = Tensor::from_raw(DataType::Bool, vec![1], &[1]).unwrap();
    let weights = Arc::new(WeightStore::new());
    let ep = auto_detect_cpu_ep().unwrap();

    let mut enabled = Executor::build(
        inplace_capture_if_graph(),
        Arc::clone(&weights),
        Arc::clone(&ep),
    )
    .unwrap();
    let enabled_output = enabled.run(&[("x", &values), ("cond", &cond)]).unwrap()[0]
        .as_bytes()
        .to_vec();

    let mut disabled = Executor::build(inplace_capture_if_graph(), weights, ep).unwrap();
    disabled.compute_in_place_enabled = false;
    let disabled_output = disabled.run(&[("x", &values), ("cond", &cond)]).unwrap()[0]
        .as_bytes()
        .to_vec();

    let expected = Tensor::from_f32(&[4], &[0.0, 0.0, 0.5, 2.0]).unwrap();
    assert_eq!(enabled_output, expected.as_bytes());
    assert_eq!(enabled_output, disabled_output);
}

/// A multi-layer, decode-shaped graph over a length-4 activation vector,
/// mixing the two hazards a real transformer decode step combines and that a
/// single-op unit graph cannot exercise together:
///
///   * **residual reuse** — each layer computes `n = Tanh(prev)` then
///     `r = Add(prev, n)`, so `prev` is read *twice* and stays live across the
///     activation; and
///   * **a long-lived carry** (`k = Relu(x)`, a stand-in for a KV/cache value)
///     that is produced first and consumed only at the very end, staying live
///     across every intervening layer.
///
/// A pure activation chain (`t = Tanh(a); u = Tanh(t); ...`) is layered on the
/// tail so compute-in-place *does* fire (each link's input is genuinely dead),
/// while the residual and carry values above must be recognised as still-live
/// and left un-aliased. If liveness ever mis-marks a residual input or the carry
/// as dead and aliases its buffer away — the regression class issue #85 guards —
/// the enabled path reads clobbered memory and diverges from the out-of-place
/// reference, so the byte-identical assertion below fails.
fn decode_shaped_residual_graph() -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", DataType::Float32, static_shape([4]));
    graph.add_input(x);

    // Long-lived carry produced first, consumed last (KV/cache stand-in).
    let k = graph.create_named_value("k", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![k]));

    // Three residual layers: `prev` feeds both the activation and the add, so it
    // must stay live across `Tanh` — a naive "dead at its next single use"
    // liveness would wrongly alias it away.
    let mut prev = x;
    let mut nid = 1u32;
    for layer in 0..3 {
        let n = graph.create_named_value(format!("n{layer}"), DataType::Float32, static_shape([4]));
        graph.insert_node(Node::new(NodeId(nid), "Tanh", vec![Some(prev)], vec![n]));
        nid += 1;
        let r = graph.create_named_value(format!("r{layer}"), DataType::Float32, static_shape([4]));
        graph.insert_node(Node::new(
            NodeId(nid),
            "Add",
            vec![Some(prev), Some(n)],
            vec![r],
        ));
        nid += 1;
        prev = r;
    }

    // Merge the carry back in — this is `k`'s only (and therefore last) use.
    let merged = graph.create_named_value("merged", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(nid),
        "Add",
        vec![Some(prev), Some(k)],
        vec![merged],
    ));
    nid += 1;

    // Pure activation tail: each link's input is genuinely dead, so
    // compute-in-place is *expected* to fire here (proving the optimization is
    // active in this graph, not silently disabled).
    let mut cur = merged;
    for tail in 0..3 {
        let out =
            graph.create_named_value(format!("tail{tail}"), DataType::Float32, static_shape([4]));
        graph.insert_node(Node::new(NodeId(nid), "Tanh", vec![Some(cur)], vec![out]));
        nid += 1;
        cur = out;
    }
    graph.add_output(cur);
    graph
}

/// Regression guard for the issue #85 class the reported native/CUDA decode
/// scare pointed at: a still-live value (a residual input or a long-lived
/// KV/cache carry) must never be aliased away by an earlier in-place op on a
/// *real, multi-layer* decode-shaped graph — the shape a single-op unit test
/// cannot reproduce. Asserts the default (enabled) path (a) actually aliases
/// (so the guard is exercised, not vacuously satisfied) and (b) is byte-for-byte
/// identical to the fully out-of-place reference. Weakening the liveness guard
/// so a residual/carry input is aliased away makes the enabled output diverge
/// and this test fail (verified by mutation).
#[test]
fn compute_in_place_multilayer_decode_residual_is_byte_identical_and_fires() {
    let values = Tensor::from_f32(&[4], &[-2.0, -0.5, 0.5, 2.0]).unwrap();
    let weights = Arc::new(WeightStore::new());
    let ep = auto_detect_cpu_ep().unwrap();

    let mut enabled = Executor::build(
        decode_shaped_residual_graph(),
        Arc::clone(&weights),
        Arc::clone(&ep),
    )
    .unwrap();
    let enabled_output = enabled.run(&[("x", &values)]).unwrap()[0]
        .as_bytes()
        .to_vec();
    assert!(
        enabled.compute_in_place_alias_count >= 3,
        "compute-in-place must fire on the decode-shaped graph's activation tail \
         (expected >= 3 aliases, got {}); a zero count would make this guard vacuous",
        enabled.compute_in_place_alias_count,
    );

    let mut disabled = Executor::build(decode_shaped_residual_graph(), weights, ep).unwrap();
    disabled.compute_in_place_enabled = false;
    let disabled_output = disabled.run(&[("x", &values)]).unwrap()[0]
        .as_bytes()
        .to_vec();
    assert_eq!(disabled.compute_in_place_alias_count, 0);
    assert_eq!(
        enabled_output, disabled_output,
        "compute-in-place aliased a still-live residual/carry value on a multi-layer \
         decode-shaped graph — the exact corruption class issue #85 must prevent",
    );
}

struct CaptureDecliningKernel;

impl Kernel for CaptureDecliningKernel {
    fn execute(
        &self,
        _inputs: &[TensorView],
        _outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "requires M==1 decode GEMV without group_indices; got a prefill signature",
        )
    }
}

#[test]
fn kernel_capture_reason_propagates_into_structured_report() {
    let mut node = Node::new(NodeId(9), "MatMulNBits", vec![], vec![]);
    node.domain = "com.microsoft".to_string();
    let decline = kernel_capture_decline(node.id, &node, &CaptureDecliningKernel).expect("decline");
    let report = CaptureDeclineReport::one(decline);

    assert_eq!(
        report.entries,
        vec![CaptureDecline {
            node_id: Some(9),
            op_type: "MatMulNBits".to_string(),
            domain: "com.microsoft".to_string(),
            reason: "requires M==1 decode GEMV without group_indices; got a prefill signature"
                .to_string(),
            seam_reason: Some(SeamReason::KernelCaptureUnsupported),
        }]
    );
    assert!(report.to_string().contains("node 9"));
    assert!(
        report
            .to_string()
            .contains("requires M==1 decode GEMV without group_indices")
    );
}

#[test]
fn seam_reasons_map_to_structural_capture_paths() {
    let cases = [
        (
            SeamReason::HostControlFlowOrSequence,
            CapturePathKind::HostSeam,
            "host-seam",
        ),
        (
            SeamReason::UnresolvedOutputShape,
            CapturePathKind::EagerDeviceSeam,
            "eager-device-seam",
        ),
        (
            SeamReason::UnresolvedInputShape,
            CapturePathKind::EagerDeviceSeam,
            "eager-device-seam",
        ),
        (
            SeamReason::KernelNotWarmed,
            CapturePathKind::EagerDeviceSeam,
            "eager-device-seam",
        ),
        (
            SeamReason::KernelCaptureUnsupported,
            CapturePathKind::EagerDeviceSeam,
            "eager-device-seam",
        ),
    ];

    for (reason, expected_kind, expected_label) in cases {
        assert_eq!(reason.path_kind(), expected_kind);
        assert_eq!(reason.label(), expected_label);
    }
    assert_eq!(CapturePathKind::CaptureRegion.label(), "capture-region");
}

#[test]
fn ep_structural_plan_plus_executor_kernel_checks_matches_legacy_declines() {
    use onnx_runtime_ir::static_shape;

    fn legacy_node_capture_reason(
        executor: &Executor,
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Option<CaptureDecline> {
        let node = executor.graph.node(plan.node_id);
        if is_control_flow_op(&node.op_type, &node.domain)
            || is_sequence_op(&node.op_type, &node.domain)
        {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::HostControlFlowOrSequence,
                "control-flow and sequence nodes are not device-graph capturable",
            ));
        }
        if plan
            .outputs
            .iter()
            .any(|output| !resolved.contains_key(output))
        {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::UnresolvedOutputShape,
                "data-dependent output shape was unresolved before capture",
            ));
        }
        let Some(input_shapes) = plan
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| resolved.get(&value).cloned())
                    .unwrap_or(Some(Vec::new()))
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::UnresolvedInputShape,
                "data-dependent input shape was unresolved before capture",
            ));
        };
        let key = KernelKey {
            node: plan.node_id.0,
            shapes: input_shapes,
        };
        let Some(kernel) = executor.cache.entries.get(&key) else {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::KernelNotWarmed,
                "kernel has not been warmed for the requested capture shape",
            ));
        };
        kernel_capture_decline(plan.node_id, node, kernel.as_ref())
    }

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    for index in 0..6 {
        let input = graph.create_named_value(
            format!("input_{index}"),
            DataType::Float32,
            static_shape([1]),
        );
        let output = graph.create_named_value(
            format!("output_{index}"),
            DataType::Float32,
            static_shape([1]),
        );
        graph.add_input(input);
        graph.add_output(output);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![output],
        ));
    }

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().expect("CPU EP"),
    )
    .expect("representative static graph");
    let mut resolved = executor
        .value_shapes
        .iter()
        .filter_map(|(&value, shape)| as_static_shape(shape).map(|shape| (value, shape)))
        .collect::<HashMap<_, _>>();
    let keys = executor
        .plan
        .iter()
        .map(|plan| KernelKey {
            node: plan.node_id.0,
            shapes: plan
                .inputs
                .iter()
                .map(|input| {
                    input
                        .map(|value| resolved[&value].clone())
                        .unwrap_or_default()
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    executor.graph.node_mut(executor.plan[0].node_id).op_type = "If".to_string();
    resolved.remove(&executor.plan[0].outputs[0]);
    resolved.remove(&executor.plan[0].inputs[0].expect("present input"));
    resolved.remove(&executor.plan[1].outputs[0]);
    resolved.remove(&executor.plan[1].inputs[0].expect("present input"));
    resolved.remove(&executor.plan[2].inputs[0].expect("present input"));
    executor.cache.entries.remove(&keys[3]);
    executor
        .cache
        .entries
        .insert(keys[4].clone(), Box::new(CaptureDecliningKernel));

    let legacy = executor
        .plan
        .iter()
        .map(|plan| legacy_node_capture_reason(&executor, plan, &resolved))
        .collect::<Vec<_>>();
    let refactored = executor
        .plan
        .iter()
        .map(|plan| executor.node_capture_reason(plan, &resolved))
        .collect::<Vec<_>>();

    assert_eq!(refactored, legacy);
    assert_eq!(
        refactored
            .iter()
            .map(|decline| decline.as_ref().and_then(|decline| decline.seam_reason))
            .collect::<Vec<_>>(),
        vec![
            Some(SeamReason::HostControlFlowOrSequence),
            Some(SeamReason::UnresolvedOutputShape),
            Some(SeamReason::UnresolvedInputShape),
            Some(SeamReason::KernelNotWarmed),
            Some(SeamReason::KernelCaptureUnsupported),
            None,
        ]
    );
}

#[test]
fn capture_shapes_seed_unresolved_external_values_without_overwriting_resolved_shapes() {
    let external_value = |shape| ExternalValue {
        dtype: DataType::Float32,
        shape,
        accepts_subshape: false,
        ptr: std::ptr::null_mut(),
        len: 0,
        alignment: 1,
        device: onnx_runtime_ir::DeviceId::cpu(),
    };
    let mut external = ExternalBindings::default();
    external
        .inputs
        .insert(ValueId(0), external_value(vec![1, 2]));
    external
        .outputs
        .insert(ValueId(1), external_value(vec![1, 4, 128, 64]));
    external
        .outputs
        .insert(ValueId(2), external_value(vec![1, 4, 128, 64]));

    let mut resolved = HashMap::from([(ValueId(0), vec![1, 1])]);
    external.seed_capture_shapes(&mut resolved);

    assert_eq!(resolved[&ValueId(0)], vec![1, 1]);
    assert_eq!(resolved[&ValueId(1)], vec![1, 4, 128, 64]);
    assert_eq!(resolved[&ValueId(2)], vec![1, 4, 128, 64]);
}

#[test]
fn only_gqa_cache_inputs_use_physical_capacity_as_kernel_geometry() {
    let mut gqa = Node::new(NodeId(0), "GroupQueryAttention", vec![], vec![]);
    gqa.domain = "com.microsoft".to_string();
    let attention = Node::new(NodeId(1), "Attention", vec![], vec![]);

    assert!(kernel_input_uses_physical_capacity(&gqa, 3));
    assert!(kernel_input_uses_physical_capacity(&gqa, 4));
    assert!(!kernel_input_uses_physical_capacity(&gqa, 0));
    assert!(!kernel_input_uses_physical_capacity(&attention, 4));
}

#[test]
fn only_capacity_aware_inputs_keep_physical_capacity() {
    let shape = Node::new(NodeId(0), "Shape", vec![], vec![]);
    let reduce_sum = Node::new(NodeId(1), "ReduceSum", vec![], vec![]);
    let cumsum = Node::new(NodeId(2), "CumSum", vec![], vec![]);
    let unsqueeze = Node::new(NodeId(3), "Unsqueeze", vec![], vec![]);

    assert!(kernel_input_uses_padded_capacity(&shape, 0));
    assert!(kernel_input_uses_padded_capacity(&reduce_sum, 0));
    assert!(!kernel_input_uses_padded_capacity(&cumsum, 0));
    assert!(!kernel_input_uses_padded_capacity(&unsqueeze, 0));
    assert!(!kernel_input_uses_padded_capacity(&shape, 1));

    // GLM-5.2's `indexer` attention branch consumes the attention mask
    // through elementwise arithmetic (a `Cast`→`Add` that combines a
    // logical-width indexer score with the mask). Such a consumer is NOT
    // padded-capacity-safe: it must observe the logical valid length, or the
    // padded physical capacity (`max_len`) leaks into the `Add` and fails to
    // broadcast against the logical-width score. Because it is not in the
    // Shape/ReduceSum allowlist, the binding is forced to expose its logical
    // prefix — which is exactly what fixes the GLM-5.2 decode broadcast bug.
    let indexer_add = Node::new(NodeId(4), "Add", vec![], vec![]);
    let indexer_cast = Node::new(NodeId(5), "Cast", vec![], vec![]);
    assert!(!kernel_input_uses_padded_capacity(&indexer_add, 0));
    assert!(!kernel_input_uses_padded_capacity(&indexer_cast, 0));
}

struct WeightDeliveryKernel {
    deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
}

impl WeightDeliveryKernel {
    fn copy_bytes(bytes: &[u8], output: &mut TensorMut<'_>) -> onnx_runtime_ep_api::Result<()> {
        if bytes.len() != output.byte_size() {
            return Err(EpError::KernelFailed(
                "test output byte count mismatch".into(),
            ));
        }
        // SAFETY: the executor bounds-checked and exclusively borrowed the
        // output allocation, which is exactly `output.byte_size()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.data.0.cast::<u8>(), bytes.len());
        }
        Ok(())
    }
}

impl Kernel for WeightDeliveryKernel {
    fn execute(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        self.deliveries.lock().unwrap().push("resident");
        let bytes = unsafe {
            std::slice::from_raw_parts(inputs[0].data_ptr::<u8>(), inputs[0].byte_size())
        };
        Self::copy_bytes(bytes, &mut outputs[0])
    }

    fn execute_with_inputs(
        &self,
        inputs: &[KernelInput<'_>],
        outputs: &mut [TensorMut],
    ) -> onnx_runtime_ep_api::Result<()> {
        match &inputs[0] {
            KernelInput::Tensor(view) => self.execute(std::slice::from_ref(view), outputs),
            KernelInput::Weight(handle) => {
                self.deliveries.lock().unwrap().push("lazy");
                let NegotiatedWeight::Lazy(lazy) =
                    handle.negotiate(&ExecutionProviderCapabilities::nxrt_weight_paging())?
                else {
                    return Err(EpError::KernelFailed(
                        "nxrt test EP expected a lazy WeightHandle".into(),
                    ));
                };
                let resident = lazy.materialize()?;
                Self::copy_bytes(resident.bytes(), &mut outputs[0])
            }
        }
    }
}

struct WeightDeliveryEp {
    cpu: CpuExecutionProvider,
    lazy: bool,
    optional_input_contract: bool,
    deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
    device: onnx_runtime_ir::DeviceId,
    allocations: Arc<AtomicUsize>,
    host_uploads: Arc<AtomicUsize>,
}

impl WeightDeliveryEp {
    fn new(lazy: bool, deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>) -> Self {
        Self::with_device(
            lazy,
            deliveries,
            onnx_runtime_ir::DeviceId::cpu(),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )
    }

    fn non_host(
        lazy: bool,
        deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
        allocations: Arc<AtomicUsize>,
        host_uploads: Arc<AtomicUsize>,
    ) -> Self {
        Self::with_device(
            lazy,
            deliveries,
            onnx_runtime_ir::DeviceId::new(onnx_runtime_ir::DeviceType::Custom(7), 0),
            allocations,
            host_uploads,
        )
    }

    fn with_device(
        lazy: bool,
        deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
        device: onnx_runtime_ir::DeviceId,
        allocations: Arc<AtomicUsize>,
        host_uploads: Arc<AtomicUsize>,
    ) -> Self {
        let mut cpu = CpuExecutionProvider::new();
        cpu.initialize(&EpConfig::default()).unwrap();
        Self {
            cpu,
            lazy,
            optional_input_contract: false,
            deliveries,
            device,
            allocations,
            host_uploads,
        }
    }

    fn copy_bytes(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        if size != 0 {
            // The test EP tags host allocations as a non-host custom device
            // so executor placement is realistic while bytes stay inspectable.
            unsafe { std::ptr::copy_nonoverlapping(src, dst, size) };
        }
        Ok(())
    }
}

impl ExecutionProvider for WeightDeliveryEp {
    fn name(&self) -> &str {
        if self.lazy {
            "nxrt_test_ep"
        } else {
            "stock_test_ep"
        }
    }

    fn device_type(&self) -> onnx_runtime_ir::DeviceType {
        self.device.device_type
    }

    fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        self.device
    }

    fn capabilities(&self) -> ExecutionProviderCapabilities {
        if self.lazy {
            ExecutionProviderCapabilities::nxrt_weight_paging()
        } else {
            ExecutionProviderCapabilities::stock()
        }
    }

    fn initialize(&mut self, _config: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        _shapes: &[Shape],
        input_dtypes: &[DataType],
        _layouts: &[TensorLayout],
    ) -> KernelMatch {
        if self.optional_input_contract && op.op_type == "OptionalContract" {
            if input_dtypes == [DataType::Float32, DataType::Undefined, DataType::Bool] {
                return KernelMatch::Supported {
                    cost: Cost::ZERO,
                    required_input_layouts: None,
                    output_layouts: vec![TensorLayout::contiguous()],
                };
            }
            return KernelMatch::unsupported(format!(
                "OptionalContract requires [Float32, Undefined, Bool] input dtypes, got {input_dtypes:?}"
            ));
        }
        if LazyWeightBoundary::BlockQuantizedMoe.matches(&op.domain, &op.op_type)
            || (op.is_default_domain() && op.op_type == "Identity")
        {
            KernelMatch::Supported {
                cost: Cost::ZERO,
                required_input_layouts: None,
                output_layouts: vec![TensorLayout::contiguous()],
            }
        } else {
            KernelMatch::unsupported(format!(
                "no handler for {}::{} at opset {opset} — test EP intentionally declines this op",
                canonical_domain(op),
                op.op_type
            ))
        }
    }

    fn get_kernel(
        &self,
        _op: &Node,
        _shapes: &[Vec<usize>],
        _opset: u64,
    ) -> onnx_runtime_ep_api::Result<Box<dyn Kernel>> {
        Ok(Box::new(WeightDeliveryKernel {
            deliveries: Arc::clone(&self.deliveries),
        }))
    }

    fn allocate(&self, size: usize, alignment: usize) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        if self.device.is_host_accessible() {
            return self.cpu.allocate(size, alignment);
        }
        let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
            .map_err(|_| EpError::AlignmentError)?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(EpError::OutOfMemory {
                requested: size,
                available: 0,
            });
        }
        Ok(unsafe { DeviceBuffer::from_raw_parts(ptr.cast(), self.device, size, alignment) })
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
        if self.device.is_host_accessible() {
            return self.cpu.deallocate(buffer);
        }
        let size = buffer.len();
        let alignment = buffer.alignment();
        let ptr = buffer.into_raw().cast::<u8>();
        let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
            .expect("test EP allocated this layout");
        unsafe { std::alloc::dealloc(ptr, layout) };
        Ok(())
    }

    fn copy(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        if size > src.len() || size > dst.len() {
            return Err(EpError::KernelFailed("test EP copy out of bounds".into()));
        }
        self.copy_bytes(src.as_ptr().cast(), dst.as_mut_ptr().cast(), size)
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<Fence> {
        self.copy(src, dst, size)?;
        Ok(Fence::default())
    }

    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn copy_from_host(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer,
    ) -> onnx_runtime_ep_api::Result<()> {
        if src.len() > dst.len() {
            return Err(EpError::KernelFailed(
                "test EP host upload out of bounds".into(),
            ));
        }
        self.host_uploads.fetch_add(1, Ordering::Relaxed);
        self.copy_bytes(src.as_ptr(), dst.as_mut_ptr().cast(), src.len())
    }

    fn copy_to_host(&self, src: &DeviceBuffer, dst: &mut [u8]) -> onnx_runtime_ep_api::Result<()> {
        if dst.len() > src.len() {
            return Err(EpError::KernelFailed(
                "test EP host download out of bounds".into(),
            ));
        }
        self.copy_bytes(src.as_ptr().cast(), dst.as_mut_ptr(), dst.len())
    }
}

fn weight_delivery_fixture() -> (Graph, Arc<WeightStore>, std::path::PathBuf) {
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    let root = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap().join("target"))
        .join("weight-handle-tests");
    std::fs::create_dir_all(&root).unwrap();
    let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path = root.join(format!(
        "block-quantized-moe-{}-{id}.bin",
        std::process::id()
    ));
    std::fs::write(&path, [1u8, 2, 3, 4]).unwrap();

    let mut graph = Graph::new();
    graph.opset_imports.insert("pkg.nxrt".into(), 1);
    let weight = graph.create_named_value("weight", DataType::Uint8, static_shape([4]));
    graph.set_initializer(
        weight,
        WeightRef::External {
            path: path.clone(),
            offset: 0,
            length: 4,
            dtype: DataType::Uint8,
            dims: vec![4],
        },
    );
    let output = graph.create_named_value("output", DataType::Uint8, static_shape([4]));
    let mut node = Node::new(
        NodeId(0),
        "BlockQuantizedMoE",
        vec![Some(weight)],
        vec![output],
    );
    node.domain = "pkg.nxrt".into();
    graph.insert_node(node);
    graph.add_output(output);

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();
    (graph, Arc::new(store), path)
}

#[test]
fn claim_time_optional_input_dtype_is_undefined_not_silently_float32() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 1);
    let data = graph.create_named_value("data", DataType::Float32, static_shape([1]));
    let training_mode = graph.create_named_value("training_mode", DataType::Bool, static_shape([]));
    let output = graph.create_named_value("output", DataType::Float32, static_shape([1]));
    graph.add_input(data);
    graph.add_input(training_mode);
    graph.add_output(output);
    graph.insert_node(Node::new(
        NodeId(0),
        "OptionalContract",
        vec![Some(data), None, Some(training_mode)],
        vec![output],
    ));

    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut ep = WeightDeliveryEp::new(false, deliveries);
    ep.optional_input_contract = true;
    let executor = Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(ep));
    assert!(
        executor.is_ok(),
        "an omitted optional input must reach supports_op as DataType::Undefined"
    );
}

#[test]
fn executor_opens_per_op_span_only_when_tracing_enabled() {
    use onnx_runtime_tracer::TraceContext;

    // Disabled (default noop): no spans recorded, hot path stays quiet.
    {
        let (graph, weights, path) = weight_delivery_fixture();
        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ep: Arc<dyn ExecutionProvider> =
            Arc::new(WeightDeliveryEp::new(false, Arc::clone(&deliveries)));
        let mut executor = Executor::build(graph, weights, ep).unwrap();
        let (trace, events) = TraceContext::in_memory();
        trace.set_enabled(false);
        executor.set_trace_context(trace);
        let _ = executor.run(&[]).unwrap();
        drop(executor);
        std::fs::remove_file(path).unwrap();
        assert!(
            events.events().is_empty(),
            "a disabled trace context must not open op spans"
        );
    }

    // Enabled: exactly one op span per executed node, named by op type.
    {
        let (graph, weights, path) = weight_delivery_fixture();
        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ep: Arc<dyn ExecutionProvider> =
            Arc::new(WeightDeliveryEp::new(false, Arc::clone(&deliveries)));
        let mut executor = Executor::build(graph, weights, ep).unwrap();
        let (trace, events) = TraceContext::in_memory();
        executor.set_trace_context(trace);
        let _ = executor.run(&[]).unwrap();
        drop(executor);
        std::fs::remove_file(path).unwrap();
        let spans = events.events();
        assert_eq!(spans.len(), 1, "one op span per executed node");
        assert_eq!(spans[0].name, "BlockQuantizedMoE");
        assert_eq!(spans[0].cat, "op");
    }
}

#[test]
fn op_capture_trace_annotates_span_with_status_and_reason() {
    use onnx_runtime_tracer::TraceContext;

    // Rejected: both a status and the actionable why-not reason land on the
    // active op-span. This is the branch a production model that declines
    // capture would hit; the qwen fixture captures cleanly so it is proven
    // here directly rather than in the live trace.
    {
        let (trace, events) = TraceContext::in_memory();
        {
            let _span = trace.span("MatMulNBits", "op");
            OpCaptureTrace::Rejected(
                "kernel declares CaptureSupport::Unsupported: per-call workspace alloc",
            )
            .annotate();
        }
        let recorded = events.events();
        assert_eq!(recorded.len(), 1);
        let args = recorded[0].args.as_ref().unwrap();
        assert_eq!(args[ARG_CAPTURE_STATUS], "rejected");
        assert!(
            args[ARG_CAPTURE_REASON]
                .as_str()
                .unwrap()
                .contains("CaptureSupport::Unsupported")
        );
    }

    // Captured: status only, no reason (nothing was declined).
    {
        let (trace, events) = TraceContext::in_memory();
        {
            let _span = trace.span("MatMulNBits", "op");
            OpCaptureTrace::Captured.annotate();
        }
        let recorded = events.events();
        let args = recorded[0].args.as_ref().unwrap();
        assert_eq!(args[ARG_CAPTURE_STATUS], "captured");
    }

    // Eager: no capture attempt, so no capture annotation at all.
    {
        let (trace, events) = TraceContext::in_memory();
        {
            let _span = trace.span("MatMulNBits", "op");
            OpCaptureTrace::Eager.annotate();
        }
        let recorded = events.events();
        assert!(
            recorded[0]
                .args
                .as_ref()
                .map(|a| a.get(ARG_CAPTURE_STATUS).is_none())
                .unwrap_or(true),
            "eager ops carry no capture status"
        );
    }
}

#[test]
fn executor_selects_lazy_or_resident_weight_delivery_from_ep_capability() {
    for (lazy, expected) in [(true, "lazy"), (false, "resident")] {
        let (graph, weights, path) = weight_delivery_fixture();
        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ep: Arc<dyn ExecutionProvider> =
            Arc::new(WeightDeliveryEp::new(lazy, Arc::clone(&deliveries)));
        let mut executor = Executor::build(graph, weights, ep).unwrap();
        let outputs = executor.run(&[]).unwrap();

        assert_eq!(outputs[0].as_bytes(), &[1, 2, 3, 4]);
        assert_eq!(&*deliveries.lock().unwrap(), &[expected]);
        drop(executor);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn non_host_lazy_only_initializer_skips_eager_device_residency() {
    for (lazy, expected_allocations, expected_uploads, expected_delivery) in
        [(true, 1, 0, "lazy"), (false, 2, 1, "resident")]
    {
        let (graph, weights, path) = weight_delivery_fixture();
        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let allocations = Arc::new(AtomicUsize::new(0));
        let host_uploads = Arc::new(AtomicUsize::new(0));
        let ep: Arc<dyn ExecutionProvider> = Arc::new(WeightDeliveryEp::non_host(
            lazy,
            Arc::clone(&deliveries),
            Arc::clone(&allocations),
            Arc::clone(&host_uploads),
        ));
        let mut executor = Executor::build(graph, weights, ep).unwrap();

        assert_eq!(
            allocations.load(Ordering::Relaxed),
            expected_allocations,
            "lazy nxrt builds only the output; stock EPs also allocate the initializer"
        );
        assert_eq!(
            host_uploads.load(Ordering::Relaxed),
            expected_uploads,
            "lazy nxrt must not upload the initializer during build"
        );

        let outputs = executor.run(&[]).unwrap();
        assert_eq!(outputs[0].as_bytes(), &[1, 2, 3, 4]);
        assert_eq!(&*deliveries.lock().unwrap(), &[expected_delivery]);
        assert_eq!(
            host_uploads.load(Ordering::Relaxed),
            expected_uploads,
            "dispatch must not introduce a second EP upload"
        );
        drop(executor);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn initializer_shared_with_resident_consumer_uses_one_device_copy() {
    let (mut graph, weights, path) = weight_delivery_fixture();
    graph.opset_imports.insert(String::new(), 17);
    let weight = graph
        .values
        .iter()
        .find_map(|(vid, value)| (value.name.as_deref() == Some("weight")).then_some(vid))
        .unwrap();
    let resident_output =
        graph.create_named_value("resident_output", DataType::Uint8, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(1),
        "Identity",
        vec![Some(weight)],
        vec![resident_output],
    ));
    graph.add_output(resident_output);

    let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let allocations = Arc::new(AtomicUsize::new(0));
    let host_uploads = Arc::new(AtomicUsize::new(0));
    let ep: Arc<dyn ExecutionProvider> = Arc::new(WeightDeliveryEp::non_host(
        true,
        Arc::clone(&deliveries),
        Arc::clone(&allocations),
        Arc::clone(&host_uploads),
    ));
    let mut executor = Executor::build(graph, weights, ep).unwrap();

    assert!(
        !executor.weight_handles.contains_key(&weight),
        "a resident consumer makes the single eager device copy authoritative"
    );
    assert_eq!(allocations.load(Ordering::Relaxed), 3);
    assert_eq!(host_uploads.load(Ordering::Relaxed), 1);

    let outputs = executor.run(&[]).unwrap();
    assert_eq!(outputs[0].as_bytes(), &[1, 2, 3, 4]);
    assert_eq!(outputs[1].as_bytes(), &[1, 2, 3, 4]);
    assert_eq!(&*deliveries.lock().unwrap(), &["resident", "resident"]);
    assert_eq!(
        host_uploads.load(Ordering::Relaxed),
        1,
        "both consumers must share the one resident initializer"
    );
    drop(executor);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn coverage_collector_surfaces_ep_decline_reason() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("x", DataType::Float32, vec![Dim::Static(1)]);
    let output = graph.create_named_value("y", DataType::Float32, vec![Dim::Static(1)]);
    graph.insert_node(Node::new(
        NodeId(0),
        "NotRegistered",
        vec![Some(input)],
        vec![output],
    ));

    let ep = CpuExecutionProvider::new();
    let mut issues = Vec::new();
    collect_cuda_coverage_issues(&graph, &graph, &ep, "graph", &mut issues);

    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].op_type, "NotRegistered");
    assert_eq!(issues[0].domain, "ai.onnx");
    assert!(
        issues[0]
            .reason
            .contains("no handler for ai.onnx::NotRegistered at opset 17"),
        "{}",
        issues[0].reason
    );
    assert!(
        !issues[0].reason.contains("unsupported by"),
        "{}",
        issues[0].reason
    );
}

#[test]
fn cuda_coverage_report_groups_all_distinct_failure_classes_deterministically() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("x", DataType::Float32, vec![Dim::Static(1)]);

    let op_types = [
        "RepeatedMissing",
        "Missing08",
        "RepeatedMissing",
        "Missing07",
        "Missing06",
        "RepeatedMissing",
        "Missing05",
        "Missing04",
        "Missing03",
        "Missing02",
        "Missing01",
        "Missing00",
        "RepeatedMissing",
    ];
    for (index, op_type) in op_types.into_iter().enumerate() {
        let output = graph.create_named_value(
            format!("output_{index}"),
            DataType::Float32,
            vec![Dim::Static(1)],
        );
        graph.insert_node(Node::new(
            NodeId(index as u32),
            op_type,
            vec![Some(input)],
            vec![output],
        ));
    }

    let ep = WeightDeliveryEp::with_device(
        false,
        Arc::new(std::sync::Mutex::new(Vec::new())),
        onnx_runtime_ir::DeviceId::cuda(0),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let report = || {
        cuda_fallback_report(&graph, &ep)
            .expect("CUDA declines must produce a fallback report")
            .to_string()
    };
    let first = report();
    let second = report();

    assert_eq!(first, second);
    assert!(first.contains("13 nodes assigned to CPU"));
    assert!(first.contains("GPU EP stock_test_ep did not claim 13 node(s)"));
    assert!(first.contains("the whole session uses cpu_ep"));
    assert_eq!(first.matches("ai.onnx::RepeatedMissing:").count(), 1);
    assert!(first.contains("ai.onnx::RepeatedMissing: no handler"));
    assert!(first.contains("[count=4; examples: graph/node#0, graph/node#12, graph/node#2]"));
    assert!(!first.contains("graph/node#5"));

    for op_type in [
        "Missing00",
        "Missing01",
        "Missing02",
        "Missing03",
        "Missing04",
        "Missing05",
        "Missing06",
        "Missing07",
        "Missing08",
    ] {
        assert_eq!(
            first.matches(&format!("ai.onnx::{op_type}:")).count(),
            1,
            "{first}"
        );
        assert!(
            first.contains(&format!("ai.onnx::{op_type}: no handler")),
            "{first}"
        );
    }
    assert!(!first.contains("more unsupported node"));
}

#[test]
fn cuda_decline_warns_and_falls_back_to_cpu_unless_strict() {
    let graph = || {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let input = graph.create_named_value("input", DataType::Float32, vec![Dim::Static(1)]);
        let output = graph.create_named_value("output", DataType::Float32, vec![Dim::Static(1)]);
        graph.add_input(input);
        graph.add_output(output);
        graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![output],
        ));
        graph
    };
    let cuda_ep = || {
        Arc::new(WeightDeliveryEp::with_device(
            false,
            Arc::new(std::sync::Mutex::new(Vec::new())),
            onnx_runtime_ir::DeviceId::cuda(0),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )) as Arc<dyn ExecutionProvider>
    };

    let exec = Executor::build_with_cuda_requirement(
        graph(),
        Arc::new(WeightStore::new()),
        cuda_ep(),
        false,
    )
    .expect("default CUDA decline must use the CPU fallback");
    assert_eq!(exec.device_id().device_type, DeviceType::Cpu);
    let report = exec
        .execution_provider_fallback_report()
        .expect("fallback must remain observable");
    assert_eq!(report.assigned_node_count, 1);
    assert_eq!(report.assigned_ops, ["ai.onnx::Relu"]);
    assert_eq!(report.declines.len(), 1);
    assert_eq!(report.declines[0].op_type, "Relu");
    assert!(report.declines[0].reason.contains("intentionally declines"));

    let strict = Executor::build_with_cuda_requirement(
        graph(),
        Arc::new(WeightStore::new()),
        cuda_ep(),
        true,
    )
    .err()
    .expect("strict CUDA must reject CPU fallback");
    assert!(strict.to_string().contains("ONNX_GENAI_REQUIRE_CUDA=1"));
}

#[test]
fn sequence_executor_preserves_element_arc_identity() {
    use onnx_runtime_ir::{TensorData, WeightRef, static_shape};

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
    graph.set_initializer(
        input,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![2],
            [7.0f32, 8.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let zero = graph.create_named_value("zero", DataType::Int64, static_shape([]));
    graph.set_initializer(
        zero,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![],
            0i64.to_le_bytes().to_vec(),
        )),
    );
    let one = graph.create_named_value("one", DataType::Int64, static_shape([]));
    graph.set_initializer(
        one,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![],
            1i64.to_le_bytes().to_vec(),
        )),
    );

    let first_sequence = graph.create_value(DataType::Float32, static_shape([]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceConstruct",
        vec![Some(input)],
        vec![first_sequence],
    ));
    let first_at = graph.create_value(DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceAt",
        vec![Some(first_sequence), Some(zero)],
        vec![first_at],
    ));
    let inserted_sequence = graph.create_value(DataType::Float32, static_shape([]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceInsert",
        vec![Some(first_sequence), Some(first_at)],
        vec![inserted_sequence],
    ));
    let second_at = graph.create_value(DataType::Float32, static_shape([2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "SequenceAt",
        vec![Some(inserted_sequence), Some(one)],
        vec![second_at],
    ));
    graph.add_output(second_at);

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    let output = executor.run(&[]).unwrap();
    assert_eq!(output[0].to_vec_f32(), vec![7.0, 8.0]);

    let original = &executor.sequences[&first_sequence].elements()[0];
    let first_at_arc = &executor.seq_elem_values[&first_at];
    let inserted = &executor.sequences[&inserted_sequence].elements()[1];
    let second_at_arc = &executor.seq_elem_values[&second_at];
    assert!(original.shares_storage_with(first_at_arc));
    assert!(original.shares_storage_with(inserted));
    assert!(original.shares_storage_with(second_at_arc));
    assert_eq!(original.as_ptr(), executor.buffers[&input].as_ptr());
}

#[test]
fn view_bounds_rejects_out_of_bounds_view() {
    // A [2, 3] f32 view needs 24 bytes; give it a 16-byte backing length.
    let shape = [2usize, 3];
    let strides = compute_contiguous_strides(&shape);
    let err = view_bounds(&shape, &strides, 0, DataType::Float32, 16);
    assert!(err.is_err(), "gate must reject an oversized view");

    // Exactly-fitting length is accepted.
    assert!(view_bounds(&shape, &strides, 0, DataType::Float32, 24).is_ok());
}

/// A negative byte offset region (via a byte_offset that pushes the origin
/// past the buffer) is also rejected.
#[test]
fn view_bounds_rejects_offset_overrun() {
    let shape = [4usize];
    let strides = compute_contiguous_strides(&shape);
    // 4 f32 = 16 bytes; origin at byte 8 leaves only 8 bytes → overrun.
    assert!(view_bounds(&shape, &strides, 8, DataType::Float32, 16).is_err());
    assert!(view_bounds(&shape, &strides, 0, DataType::Float32, 16).is_ok());
}

#[test]
fn sub_byte_view_bounds_rejects_geometry_overflow() {
    let shape = [usize::MAX, 2];
    let strides = compute_contiguous_strides(&shape);
    let error = view_bounds(&shape, &strides, 0, DataType::Int4, usize::MAX);
    assert!(matches!(error, Err(SessionError::ShapeOverflow { .. })));
}

#[test]
fn sub_byte_view_bounds_rejects_offset_overflow() {
    let shape = [1usize];
    let strides = compute_contiguous_strides(&shape);
    let error = view_bounds(&shape, &strides, usize::MAX, DataType::Int4, usize::MAX);
    assert!(matches!(error, Err(SessionError::ShapeOverflow { .. })));
}

#[test]
fn device_binding_validation_rejects_geometry_overflow() {
    let element_count = usize::MAX / 4;
    let error = bindings::required_binding_bytes(DataType::Float64, &[element_count], "huge");
    assert!(matches!(error, Err(SessionError::ShapeOverflow { .. })));
}

/// Symbol substitution: static dims pass through, bound symbols resolve, an
/// unbound symbol yields `None` (the uninferred-shape signal).
#[test]
fn substitute_resolves_bound_symbols_only() {
    let mut bindings = HashMap::new();
    bindings.insert(SymbolId(0), 7usize);
    let shape = vec![Dim::Symbolic(SymbolId(0)), Dim::Static(4)];
    assert_eq!(substitute(&shape, &bindings), Some(vec![7, 4]));

    let unbound = vec![Dim::Symbolic(SymbolId(1)), Dim::Static(4)];
    assert_eq!(substitute(&unbound, &bindings), None);
}

/// H-D1: element-count multiplication must be overflow-checked so a huge or
/// malicious shape reports `ShapeOverflow` instead of wrapping `usize` and
/// under-sizing the buffer.
#[test]
fn checked_numel_detects_overflow() {
    // Well-formed shapes multiply normally.
    assert_eq!(checked_numel(&[2, 3, 4], || "v".into()).unwrap(), 24);
    assert_eq!(checked_numel(&[], || "v".into()).unwrap(), 1);

    // A product past usize::MAX overflows.
    let huge = [usize::MAX, 2];
    let err = checked_numel(&huge, || "value#9".into());
    assert!(matches!(err, Err(SessionError::ShapeOverflow { .. })));
}

/// H-D1 (byte layer): even when the element *count* fits in `usize`, the
/// count → bytes multiply can wrap for a fixed-width dtype. The allocation
/// path must report `ShapeOverflow` rather than under-allocating.
#[test]
fn checked_storage_bytes_detects_byte_overflow() {
    // `usize::MAX / 4` elements fit in usize (pass checked_numel) but
    // `* 8` bytes for Float64 wraps — this is the exploited under-alloc.
    let numel = usize::MAX / 4;
    let err = checked_storage_bytes(DataType::Float64, numel, || "value#9".into(), &[numel]);
    assert!(matches!(err, Err(SessionError::ShapeOverflow { .. })));

    // A well-formed size passes through unchanged.
    assert_eq!(
        checked_storage_bytes(DataType::Float32, 4, || "v".into(), &[4]).unwrap(),
        16
    );
}

/// The data-dependent shape sizer must return exactly one shape per output
/// so the run loop's `out_shapes[oi]` indexing can never misindex. Slice is
/// single-output, so it returns a 1-element Vec; the run loop additionally
/// guards the count (see `OutputShapeCountMismatch`).
#[test]
fn dynamic_output_shapes_slice_is_single_output() {
    let node = Node::new(NodeId(0), "Slice", vec![], vec![]);
    let input_shapes = vec![vec![4usize, 2]];
    let input_values = vec![
        None,          // data (unused by sizer)
        Some(vec![1]), // starts
        Some(vec![3]), // ends
        Some(vec![0]), // axes
        Some(vec![1]), // steps
    ];
    let input_dtypes = vec![
        DataType::Float32,
        DataType::Int64,
        DataType::Int64,
        DataType::Int64,
        DataType::Int64,
    ];
    let out =
        dynamic_output_shapes(&node, &input_shapes, &input_dtypes, &input_values, &[], 17).unwrap();
    assert_eq!(out.len(), 1, "Slice must resolve exactly one output shape");
    assert_eq!(out[0], vec![2, 2]);

    let mut custom_slice = Node::new(NodeId(1), "Slice", vec![], vec![ValueId(0)]);
    custom_slice.domain = "example.custom".into();
    assert!(
        dynamic_output_shapes(
            &custom_slice,
            &input_shapes,
            &input_dtypes,
            &input_values,
            &[],
            17
        )
        .is_none(),
        "ONNX Slice semantics must not be applied to an unrelated custom-domain op"
    );

    // An op the sizer cannot resolve returns None (surfaces as UnresolvedShape).
    let other = Node::new(
        NodeId(2),
        "NxrtNeverRegisteredSentinelOp",
        vec![],
        vec![ValueId(0)],
    );
    assert!(
        dynamic_output_shapes(&other, &input_shapes, &input_dtypes, &input_values, &[], 17)
            .is_none()
    );
}

#[test]
fn dynamic_output_shapes_unsqueeze_supports_input_and_attribute_axes() {
    use onnx_runtime_ir::Attribute;

    let input_axes = Node::new(
        NodeId(0),
        "Unsqueeze",
        vec![Some(ValueId(0)), Some(ValueId(1))],
        vec![ValueId(2)],
    );
    assert_eq!(
        dynamic_output_shapes(
            &input_axes,
            &[vec![2, 3], vec![2]],
            &[DataType::Float32, DataType::Int64],
            &[None, Some(vec![0, -1])],
            &[],
            17,
        ),
        Some(vec![vec![1, 2, 3, 1]])
    );

    let mut attribute_axes = Node::new(
        NodeId(1),
        "Unsqueeze",
        vec![Some(ValueId(0))],
        vec![ValueId(1)],
    );
    attribute_axes
        .attributes
        .insert("axes".into(), Attribute::Ints(vec![1, -1]));
    assert_eq!(
        dynamic_output_shapes(
            &attribute_axes,
            &[vec![2, 3]],
            &[DataType::Float32],
            &[None],
            &[],
            11,
        ),
        Some(vec![vec![2, 1, 3, 1]])
    );
}

#[test]
fn dynamic_output_shapes_resize_reads_runtime_scales() {
    let node = Node::new(
        NodeId(0),
        "Resize",
        vec![Some(ValueId(0)), Some(ValueId(1)), Some(ValueId(2))],
        vec![ValueId(3)],
    );
    assert_eq!(
        dynamic_output_shapes(
            &node,
            &[vec![1, 128, 13, 13], vec![8], vec![4]],
            &[DataType::Float32, DataType::Float32, DataType::Float32],
            &[None, None, None],
            &[None, None, Some(vec![1.0, 1.0, 2.0, 2.0])],
            11,
        ),
        Some(vec![vec![1, 128, 26, 26]])
    );
}

#[test]
fn dynamic_output_shapes_non_max_suppression_counts_selected_boxes() {
    let node = Node::new(
        NodeId(0),
        "NonMaxSuppression",
        (0..5).map(|index| Some(ValueId(index))).collect(),
        vec![ValueId(5)],
    );
    let shapes = vec![vec![1, 3, 4], vec![1, 1, 3], vec![], vec![], vec![]];
    let dtypes = vec![
        DataType::Float32,
        DataType::Float32,
        DataType::Int64,
        DataType::Float32,
        DataType::Float32,
    ];
    let ints = vec![None, None, Some(vec![2]), None, None];
    let floats = vec![
        Some(vec![0., 0., 1., 1., 0., 0., 0.9, 0.9, 2., 2., 3., 3.]),
        Some(vec![0.9, 0.8, 0.7]),
        None,
        Some(vec![0.5]),
        Some(vec![0.0]),
    ];
    assert_eq!(
        dynamic_output_shapes(&node, &shapes, &dtypes, &ints, &floats, 11),
        Some(vec![vec![2, 3]])
    );
}

#[test]
fn dynamic_output_shapes_gqa_supports_packed_qkv() {
    use onnx_runtime_ir::{Attribute, ValueId};

    let mut node = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        vec![
            Some(ValueId(0)),
            None,
            None,
            Some(ValueId(3)),
            Some(ValueId(4)),
            Some(ValueId(5)),
            Some(ValueId(6)),
        ],
        vec![ValueId(7), ValueId(8), ValueId(9)],
    );
    node.domain = "com.microsoft".into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(14));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(2));
    let input_shapes = vec![
        vec![1, 1, 1152],
        vec![],
        vec![],
        vec![1, 2, 16, 64],
        vec![1, 2, 16, 64],
        vec![1],
        vec![],
    ];
    let input_values = vec![None, None, None, None, None, None, Some(vec![17])];

    assert_eq!(
        dynamic_output_shapes(
            &node,
            &input_shapes,
            &[
                DataType::Float32,
                DataType::Undefined,
                DataType::Undefined,
                DataType::Float32,
                DataType::Float32,
                DataType::Int32,
                DataType::Int32,
            ],
            &input_values,
            &[],
            1,
        ),
        Some(vec![
            vec![1, 1, 896],
            vec![1, 2, 17, 64],
            vec![1, 2, 17, 64],
        ])
    );
}

/// The effective opset is read from the graph's import for the op's domain,
/// with the default and `ai.onnx` spellings treated as one.
#[test]
fn effective_opset_reads_graph_import() {
    let mut graph = Graph::default();
    graph.opset_imports.insert(String::new(), 12);
    let node = Node::new(NodeId(0), "Softmax", vec![], vec![]);
    assert_eq!(effective_opset(&graph, &node), 12);

    graph.opset_imports.insert(String::new(), 0);
    assert_eq!(effective_opset(&graph, &node), 0);
}

#[test]
#[should_panic(expected = "internal invariant violated")]
fn effective_opset_requires_validated_import() {
    effective_opset(
        &Graph::default(),
        &Node::new(NodeId(0), "Softmax", vec![], vec![]),
    );
}

#[test]
fn child_executor_binds_formals_captures_and_inline_initializers_in_output_order() {
    use onnx_runtime_ir::{TensorData, WeightRef, static_shape};

    let mut body = Graph::new();
    let formal = body.create_named_value("formal", DataType::Float32, static_shape([2]));
    body.add_input(formal);
    let captured = body.create_named_value("captured", DataType::Float32, static_shape([2]));
    let one = body.create_named_value("one", DataType::Float32, static_shape([2]));
    body.set_initializer(
        one,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![2],
            [1.0f32, 1.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let sum = body.create_named_value("sum", DataType::Float32, static_shape([2]));
    body.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(formal), Some(captured)],
        vec![sum],
    ));
    let adjusted = body.create_named_value("adjusted", DataType::Float32, static_shape([2]));
    body.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(sum), Some(one)],
        vec![adjusted],
    ));
    // Deliberately reverse production order to prove formal output ordering.
    body.add_output(adjusted);
    body.add_output(sum);

    let mut opsets = HashMap::new();
    opsets.insert(String::new(), 17);
    let mut child = ChildExecutor::new(
        "direct-test",
        body,
        opsets,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    let mut outer_scope = HashMap::new();
    outer_scope.insert(
        "captured".to_string(),
        Tensor::from_f32(&[2], &[10.0, 20.0]).unwrap(),
    );

    let first = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();
    let outputs = child.run(&[&first], &outer_scope).unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].to_vec_f32(), vec![13.0, 24.0]);
    assert_eq!(outputs[1].to_vec_f32(), vec![12.0, 23.0]);
    assert_eq!(child.stats(), ChildExecutorStats { builds: 1, runs: 1 });

    let second = Tensor::from_f32(&[2], &[-1.0, 4.0]).unwrap();
    let outputs = child.run(&[&second], &outer_scope).unwrap();
    assert_eq!(outputs[0].to_vec_f32(), vec![10.0, 25.0]);
    assert_eq!(outputs[1].to_vec_f32(), vec![9.0, 24.0]);
    assert_eq!(
        child.stats(),
        ChildExecutorStats { builds: 1, runs: 2 },
        "matching input signatures must reuse the compiled child plan"
    );
}

fn unary_child(name: &str) -> ChildExecutor {
    let mut body = Graph::new();
    let input = body.create_named_value("input", DataType::Float32, Vec::new());
    body.add_input(input);
    let output = body.create_named_value("output", DataType::Float32, Vec::new());
    body.insert_node(Node::new(
        NodeId(0),
        "Relu",
        vec![Some(input)],
        vec![output],
    ));
    body.add_output(output);

    let mut opsets = HashMap::new();
    opsets.insert(String::new(), 17);
    ChildExecutor::new(
        name,
        body,
        opsets,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap()
}

#[test]
fn child_executor_reuses_a_signature_after_an_intervening_signature() {
    let mut child = unary_child("a-b-a");
    let outer_scope = HashMap::new();
    let a = Tensor::from_f32(&[1], &[-1.0]).unwrap();
    let b = Tensor::from_f32(&[2], &[-2.0, 3.0]).unwrap();

    assert_eq!(
        child.run(&[&a], &outer_scope).unwrap()[0].to_vec_f32(),
        vec![0.0]
    );
    assert_eq!(
        child.run(&[&b], &outer_scope).unwrap()[0].to_vec_f32(),
        vec![0.0, 3.0]
    );
    assert_eq!(
        child.run(&[&a], &outer_scope).unwrap()[0].to_vec_f32(),
        vec![0.0]
    );
    assert_eq!(child.stats(), ChildExecutorStats { builds: 2, runs: 3 });
}

#[test]
fn child_executor_lru_evicts_oldest_signature_only() {
    let mut child = unary_child("lru-eviction");
    let outer_scope = HashMap::new();
    let inputs = (1..=CHILD_EXECUTOR_CACHE_CAPACITY + 1)
        .map(|len| Tensor::from_f32(&[len], &vec![len as f32; len]).unwrap())
        .collect::<Vec<_>>();

    for input in &inputs {
        child.run(&[input], &outer_scope).unwrap();
    }
    assert_eq!(
        child.stats(),
        ChildExecutorStats {
            builds: (CHILD_EXECUTOR_CACHE_CAPACITY + 1) as u64,
            runs: (CHILD_EXECUTOR_CACHE_CAPACITY + 1) as u64,
        }
    );

    child.run(&[&inputs[0]], &outer_scope).unwrap();
    child.run(&[inputs.last().unwrap()], &outer_scope).unwrap();
    assert_eq!(
        child.stats(),
        ChildExecutorStats {
            builds: (CHILD_EXECUTOR_CACHE_CAPACITY + 2) as u64,
            runs: (CHILD_EXECUTOR_CACHE_CAPACITY + 3) as u64,
        },
        "the evicted oldest signature must rebuild while a recent entry remains cached"
    );
}

fn captured_add_child(name: &str) -> ChildExecutor {
    let mut body = Graph::new();
    let input = body.create_named_value("input", DataType::Float32, Vec::new());
    body.add_input(input);
    let captured = body.create_named_value("captured", DataType::Float32, Vec::new());
    let output = body.create_named_value("output", DataType::Float32, Vec::new());
    body.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(input), Some(captured)],
        vec![output],
    ));
    body.add_output(output);

    let mut opsets = HashMap::new();
    opsets.insert(String::new(), 17);
    ChildExecutor::new(
        name,
        body,
        opsets,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap()
}

#[test]
fn child_executor_cached_plan_rebinds_captures_without_stale_state() {
    let mut child = captured_add_child("capture-shadowing");
    let a_input = Tensor::from_f32(&[1], &[1.0]).unwrap();
    let b_input = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();

    let mut scope = HashMap::new();
    scope.insert(
        "captured".to_string(),
        Tensor::from_f32(&[1], &[10.0]).unwrap(),
    );
    assert_eq!(
        child.run(&[&a_input], &scope).unwrap()[0].to_vec_f32(),
        vec![11.0]
    );

    scope.insert(
        "captured".to_string(),
        Tensor::from_f32(&[2], &[20.0, 30.0]).unwrap(),
    );
    assert_eq!(
        child.run(&[&b_input], &scope).unwrap()[0].to_vec_f32(),
        vec![22.0, 33.0]
    );

    scope.insert(
        "captured".to_string(),
        Tensor::from_f32(&[1], &[40.0]).unwrap(),
    );
    let cached = child.run(&[&a_input], &scope).unwrap()[0].to_vec_f32();
    let mut fresh = captured_add_child("capture-shadowing-fresh");
    let freshly_compiled = fresh.run(&[&a_input], &scope).unwrap()[0].to_vec_f32();

    assert_eq!(cached, vec![41.0]);
    assert_eq!(cached, freshly_compiled);
    assert_eq!(child.stats(), ChildExecutorStats { builds: 2, runs: 3 });
}

// --- weight-streaming: zero-copy borrowed initializer buffers -----------

use onnx_runtime_ir::{WeightRef, static_shape};
use std::path::PathBuf;

/// A writable scratch dir under the workspace `target/` (never `/tmp`).
fn weightstream_tmp_dir() -> PathBuf {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/weightstream_test"
    ));
    std::fs::create_dir_all(&dir).expect("create weight-streaming test dir");
    dir
}

fn f32_le(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// (b) An aligned external-data initializer is backed **zero-copy** by a
/// borrowed buffer whose data pointer EQUALS the WeightStore's mmap slice —
/// no allocation, no copy. A model larger than RAM relies on this.
#[test]
fn aligned_external_initializer_is_borrowed_zero_copy() {
    let align = TensorLayout::contiguous().alignment;
    let path = weightstream_tmp_dir().join("aligned_init.bin");
    let w_data = [1.0f32, 2.0, 3.0, 4.0];
    std::fs::write(&path, f32_le(&w_data)).unwrap();

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();

    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
    g.set_initializer(
        w,
        WeightRef::External {
            path: path.clone(),
            offset: 0, // mmap base is page-aligned -> 0 is `align`-aligned
            length: 16,
            dtype: DataType::Float32,
            dims: vec![4],
        },
    );
    let y = g.create_value(DataType::Float32, static_shape([4]));
    g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(w)], vec![y]));
    g.add_output(y);

    let ep = auto_detect_cpu_ep().unwrap();
    let exec = Executor::build(g, Arc::new(store), ep).unwrap();

    let weight = &exec.graph.initializers[&w];
    let src = exec.weights().bytes(weight).unwrap();
    assert!(
        (src.as_ptr() as usize).is_multiple_of(align),
        "mmap window must be aligned for this test to exercise the zero-copy path"
    );
    let buf = &exec.buffers[&w];
    assert!(
        buf.is_borrowed(),
        "aligned initializer must be borrowed, not copied"
    );
    assert_eq!(
        buf.as_ptr() as *const u8,
        src.as_ptr(),
        "zero-copy: the buffer must alias the mmap bytes (no copy)"
    );

    let _ = std::fs::remove_file(&path);
}

/// (c) An external-data initializer that is dtype-aligned but not 64-byte
/// aligned remains a zero-copy mmap borrow and is numerically correct.
#[test]
fn device_unaligned_external_initializer_is_borrowed_at_dtype_alignment() {
    let align = TensorLayout::contiguous().alignment;
    let path = weightstream_tmp_dir().join("unaligned_init.bin");
    // Prefix the weight window with 8 bytes so it starts at offset 8, which
    // is f32-aligned but not a multiple of the EP allocation alignment (64).
    let offset = 8usize;
    let w_data = [5.0f32, 6.0, 7.0, 8.0];
    let mut file = vec![0u8; offset];
    file.extend_from_slice(&f32_le(&w_data));
    std::fs::write(&path, &file).unwrap();

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();

    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
    g.set_initializer(
        w,
        WeightRef::External {
            path: path.clone(),
            offset,
            length: 16,
            dtype: DataType::Float32,
            dims: vec![4],
        },
    );
    let x = g.create_named_value("X", DataType::Float32, static_shape([4]));
    g.add_input(x);
    let y = g.create_value(DataType::Float32, static_shape([4]));
    g.insert_node(Node::new(NodeId(0), "Add", vec![Some(x), Some(w)], vec![y]));
    g.add_output(y);

    let ep = auto_detect_cpu_ep().unwrap();
    let mut exec = Executor::build(g, Arc::new(store), ep).unwrap();

    let weight = &exec.graph.initializers[&w];
    let src = exec.weights().bytes(weight).unwrap();
    assert!(
        !(src.as_ptr() as usize).is_multiple_of(align),
        "window must be unaligned for this test to exercise the fallback"
    );
    let buf = &exec.buffers[&w];
    assert!(
        buf.is_borrowed(),
        "dtype-aligned mmap initializer must remain borrowed"
    );
    assert_eq!(
        buf.as_ptr() as *const u8,
        src.as_ptr(),
        "zero-copy buffer must alias the mmap window"
    );
    assert_eq!(buf.alignment(), std::mem::align_of::<f32>());

    // The copy is numerically correct: Y = X + W.
    let x_tensor = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let out = exec.run(&[("X", &x_tensor)]).unwrap();
    assert_eq!(out.len(), 1);
    let got = out[0].to_vec_f32();
    let want = [15.0f32, 26.0, 37.0, 48.0];
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn unaligned_external_qmoe_keeps_route_first_enabled_and_matches_legacy() {
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _env_guard = ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("weight-offload env lock");

    struct RestoreEnv(Option<OsString>);
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            if let Some(value) = self.0.take() {
                // SAFETY: this test serializes all mutations it performs.
                unsafe { std::env::set_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV, value) };
            } else {
                // SAFETY: this test serializes all mutations it performs.
                unsafe { std::env::remove_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV) };
            }
        }
    }

    let _restore = RestoreEnv(std::env::var_os(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV));
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-runtime-ep-cpu/tests/fixtures/qmoe_weight_offload/model.onnx");
    let input_values: Vec<f32> = (0..64).map(|index| index as f32 * 0.03125 - 1.0).collect();
    let router_values = vec![
        9.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 9.0,
    ];
    let input = Tensor::from_f32(&[4, 16], &input_values).unwrap();
    let router = Tensor::from_f32(&[4, 4], &router_values).unwrap();

    // SAFETY: guarded above; both executors compile synchronously here.
    unsafe { std::env::set_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV, "0") };
    let (legacy_graph, legacy_weights) =
        onnx_runtime_loader::load_model_with_weights(&fixture).unwrap();
    let mut legacy =
        Executor::build(legacy_graph, legacy_weights, auto_detect_cpu_ep().unwrap()).unwrap();
    let legacy_output = legacy.run(&[("X", &input), ("router", &router)]).unwrap();

    // SAFETY: guarded above; the offload kernel captures the flag at build.
    unsafe { std::env::set_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV, "1") };
    let before = onnx_runtime_ep_cpu::weight_offload_stats();
    let (offload_graph, offload_weights) =
        onnx_runtime_loader::load_model_with_weights(&fixture).unwrap();
    let mut offload = Executor::build(
        offload_graph,
        offload_weights,
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    for (&value, weight) in &offload.graph.initializers {
        let WeightRef::External { .. } = weight else {
            continue;
        };
        let source = offload.weights.bytes(weight).unwrap();
        assert!(!(source.as_ptr() as usize).is_multiple_of(TensorLayout::contiguous().alignment));
        let buffer = &offload.buffers[&value];
        assert!(buffer.is_borrowed());
        assert_eq!(buffer.as_ptr() as *const u8, source.as_ptr());
    }
    let offload_output = offload.run(&[("X", &input), ("router", &router)]).unwrap();
    let after = onnx_runtime_ep_cpu::weight_offload_stats();

    assert_eq!(
        offload_output[0].to_vec_f32(),
        legacy_output[0].to_vec_f32()
    );
    assert!(
        after.layer_executions
            >= before
                .layer_executions
                .checked_add(1)
                .expect("layer execution counter overflow")
    );
    assert!(after.bytes_read_from_mmap > before.bytes_read_from_mmap);
}

/// (d) Soundness guard: even when an initializer's mmap bytes are aligned
/// (so the zero-copy path would otherwise fire), the executor must NOT
/// borrow them if the value also has a producer — i.e. a malformed graph
/// reused the initializer's `ValueId` as a node output. Borrowing yields a
/// read-only buffer; a kernel writing that output would write through the
/// mmap (SIGSEGV / aliasing UB). The build must fall back to an owned,
/// writable copy instead.
#[test]
fn producer_backed_initializer_is_not_borrowed() {
    let align = TensorLayout::contiguous().alignment;
    let path = weightstream_tmp_dir().join("producer_backed_init.bin");
    let w_data = [1.0f32, 2.0, 3.0, 4.0];
    std::fs::write(&path, f32_le(&w_data)).unwrap();

    let mut store = WeightStore::new();
    store.map_external(&path).unwrap();

    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    let x = g.create_named_value("X", DataType::Float32, static_shape([4]));
    g.add_input(x);
    let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
    g.set_initializer(
        w,
        WeightRef::External {
            path: path.clone(),
            offset: 0, // aligned: without the producer guard this would borrow
            length: 16,
            dtype: DataType::Float32,
            dims: vec![4],
        },
    );
    // Reuse the initializer's ValueId as a node output -> gives `w` a
    // producer, exactly the malformed shape the loader also rejects.
    g.insert_node(Node::new(NodeId(0), "Identity", vec![Some(x)], vec![w]));
    let y = g.create_value(DataType::Float32, static_shape([4]));
    g.insert_node(Node::new(NodeId(1), "Add", vec![Some(x), Some(w)], vec![y]));
    g.add_output(y);

    assert!(
        g.value(w).producer.is_some(),
        "test setup: initializer value must have a producer",
    );

    let ep = auto_detect_cpu_ep().unwrap();
    let exec = Executor::build(g, Arc::new(store), ep).unwrap();

    let weight = &exec.graph.initializers[&w];
    let src = exec.weights().bytes(weight).unwrap();
    assert!(
        (src.as_ptr() as usize).is_multiple_of(align),
        "mmap window must be aligned so only the producer guard prevents borrowing",
    );
    let buf = &exec.buffers[&w];
    assert!(
        !buf.is_borrowed(),
        "producer-backed initializer must fall back to an owned writable copy",
    );
    assert_ne!(
        buf.as_ptr() as *const u8,
        src.as_ptr(),
        "producer-backed initializer must not alias read-only mmap bytes",
    );

    let _ = std::fs::remove_file(&path);
}

/// Stage-0 capture prerequisite: a data-dependent decode shape that
/// `resolve_soft` omits (here `Range`'s runtime-length output feeding a
/// capture-safe `Cast`) forms an *unresolved-shape* eager seam. After one
/// eager warmup, [`Executor::seed_warm_decode_capture_shapes`] seeds that
/// value's exact just-in-time shape for the identical decode binding
/// signature, and the node no longer reports an unresolved-shape seam — the
/// executor now *admits* the already-capture-safe node instead of rejecting
/// it before consulting its kernel. Non-tautological: it asserts the
/// before/after seam transition and the concrete seeded extent.
#[test]
fn warm_decode_seeding_admits_previously_unresolved_capture_safe_node() {
    use onnx_runtime_ir::{Attribute, static_shape};

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    // Range(start, limit, delta) with all three supplied at run time, so
    // static shape inference cannot pin the output length: it stays symbolic
    // (data-dependent) and `resolve_soft` omits it.
    let start = graph.create_named_value("start", DataType::Int64, static_shape([]));
    let limit = graph.create_named_value("limit", DataType::Int64, static_shape([]));
    let delta = graph.create_named_value("delta", DataType::Int64, static_shape([]));
    graph.add_input(start);
    graph.add_input(limit);
    graph.add_input(delta);
    let len_sym = graph.intern_symbol("range_len");
    let r = graph.create_named_value("r", DataType::Int64, vec![len_sym.into()]);
    graph.insert_node(Node::new(
        NodeId(0),
        "Range",
        vec![Some(start), Some(limit), Some(delta)],
        vec![r],
    ));
    let y = graph.create_named_value("y", DataType::Float32, vec![len_sym.into()]);
    let mut cast = Node::new(NodeId(0), "Cast", vec![Some(r)], vec![y]);
    cast.attributes
        .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
    graph.insert_node(cast);
    graph.add_output(y);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    // Warm-decode capture-shape seeding is a device-graph-capture concern
    // (capture-capable EPs only), where the decode-plan memo is disabled by
    // construction (`device_type() != Cuda` gate). The memo-eligible CPU
    // eager path deliberately skips recording `capture_warm_shapes` (it never
    // captures). Disable the now-default-ON memo so this executor records the
    // warm shapes exactly as a capture-capable EP would at runtime.
    exec.set_decode_memo_enabled(false);

    let zero = Tensor::from_raw(DataType::Int64, vec![], &0i64.to_le_bytes()).unwrap();
    let four = Tensor::from_raw(DataType::Int64, vec![], &4i64.to_le_bytes()).unwrap();
    let one = Tensor::from_raw(DataType::Int64, vec![], &1i64.to_le_bytes()).unwrap();
    let inputs = [("start", &zero), ("limit", &four), ("delta", &one)];

    let cast_pi = exec
        .plan
        .iter()
        .position(|p| exec.graph.node(p.node_id).op_type == "Cast")
        .expect("plan contains the Cast node");

    // --- Before any warmup: the Cast is an unresolved-shape eager seam. ---
    let bindings = exec
        .bind_symbols(&inputs, &ExternalBindings::default())
        .unwrap();
    let pre = exec.resolve_soft(&bindings);
    assert!(
        !pre.contains_key(&r) && !pre.contains_key(&y),
        "Range's runtime-length output (and its Cast) must be data-dependent (unresolved)"
    );
    let pre_seam = exec
        .node_capture_reason(&exec.plan[cast_pi], &pre)
        .and_then(|decline| decline.seam_reason);
    assert!(
        matches!(
            pre_seam,
            Some(SeamReason::UnresolvedInputShape) | Some(SeamReason::UnresolvedOutputShape)
        ),
        "without seeding the Cast must be an unresolved-shape seam; got {pre_seam:?}"
    );

    // --- One eager warmup records the exact just-in-time shapes. ----------
    let out = exec.run(&inputs).unwrap();
    assert_eq!(out[0].to_vec_f32(), vec![0.0, 1.0, 2.0, 3.0]);

    // --- After warmup: the identical signature seeds the warm shapes. -----
    let bindings2 = exec
        .bind_symbols(&inputs, &ExternalBindings::default())
        .unwrap();
    let mut post = exec.resolve_soft(&bindings2);
    assert!(
        !post.contains_key(&r),
        "resolve_soft alone still omits the data-dependent value"
    );
    exec.seed_warm_decode_capture_shapes(&mut post, &ExternalBindings::default());
    assert_eq!(
        post.get(&r),
        Some(&vec![4usize]),
        "warm seeding must restore Range's exact eager-resolved output shape"
    );
    assert_eq!(post.get(&y), Some(&vec![4usize]));

    // The unresolved-shape seam is gone: the executor admits the node to the
    // shape gate (any remaining decline is a kernel-capability decision, not
    // a missing-shape rejection).
    let post_seam = exec
        .node_capture_reason(&exec.plan[cast_pi], &post)
        .and_then(|decline| decline.seam_reason);
    assert!(
        !matches!(
            post_seam,
            Some(SeamReason::UnresolvedInputShape) | Some(SeamReason::UnresolvedOutputShape)
        ),
        "warm-seeded decode shapes must clear the unresolved-shape seam; got {post_seam:?}"
    );

    // A changed decode signature must NOT seed (pointer/capacity instability):
    // a phantom persistent input binding makes the current signature differ
    // from the warmup's, so the warm shapes are withheld and the value stays
    // unresolved rather than risk baking a stale shape into a captured graph.
    let mut mismatched = exec.resolve_soft(&bindings2);
    let mut other = ExternalBindings::default();
    other.inputs.insert(
        start,
        ExternalValue {
            dtype: DataType::Int64,
            shape: vec![],
            accepts_subshape: false,
            ptr: 0x1000 as *mut std::ffi::c_void,
            len: 8,
            alignment: 8,
            device: onnx_runtime_ir::DeviceId::cpu(),
        },
    );
    exec.seed_warm_decode_capture_shapes(&mut mismatched, &other);
    assert!(
        !mismatched.contains_key(&r),
        "a changed persistent-binding signature must withhold the warm seed"
    );
}

/// A kernel that aborts device-graph *recording* (e.g. it advertises capture
/// support but synchronizes, which CUDA rejects mid-capture) is quarantined
/// by op-type so a single mislabeled kernel cannot abort the whole segmented
/// capture. Once its `(domain, op_type)` is quarantined,
/// [`Executor::node_capture_reason`] must force that node to a
/// `CaptureRecordingFailed` eager seam even when its shapes are fully
/// resolved and it would otherwise reach the kernel gate. Non-tautological:
/// it asserts the *transition* from "not an unresolved-shape seam" to a
/// forced-seam classification caused solely by the quarantine.
#[test]
fn quarantined_op_type_is_forced_to_a_capture_recording_failed_seam() {
    use onnx_runtime_ir::{Attribute, static_shape};

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 13);
    let x = graph.create_named_value("x", DataType::Int64, static_shape([4]));
    graph.add_input(x);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    let mut cast = Node::new(NodeId(0), "Cast", vec![Some(x)], vec![y]);
    cast.attributes
        .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
    graph.insert_node(cast);
    graph.add_output(y);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let cast_pi = exec
        .plan
        .iter()
        .position(|p| exec.graph.node(p.node_id).op_type == "Cast")
        .expect("plan contains the Cast node");

    // Statically-shaped Cast: its I/O shapes resolve without any seeding, so
    // it is NOT an unresolved-shape seam and reaches the kernel gate.
    let xt = Tensor::from_raw(
        DataType::Int64,
        vec![4],
        &[0i64, 1, 2, 3]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    )
    .unwrap();
    let bindings = exec
        .bind_symbols(&[("x", &xt)], &ExternalBindings::default())
        .unwrap();
    let resolved = exec.resolve_soft(&bindings);
    let pre_seam = exec
        .node_capture_reason(&exec.plan[cast_pi], &resolved)
        .and_then(|decline| decline.seam_reason);
    assert!(
        !matches!(pre_seam, Some(SeamReason::CaptureRecordingFailed)),
        "a non-quarantined statically-shaped node must not be a recording-failed seam; \
             got {pre_seam:?}"
    );

    // Quarantine the Cast op-type (as the capture retry loop does after a
    // kernel aborts recording) and re-check: it is now a forced eager seam
    // regardless of its resolved shapes or kernel capability.
    exec.capture_quarantine_ops
        .insert(("ai.onnx".to_string(), "Cast".to_string()));
    let post = exec.node_capture_reason(&exec.plan[cast_pi], &resolved);
    assert_eq!(
        post.and_then(|decline| decline.seam_reason),
        Some(SeamReason::CaptureRecordingFailed),
        "a quarantined op-type must be forced to a CaptureRecordingFailed eager seam"
    );
}

// ===================================================================
// F5 Stage 1 — steady-state decode-plan memo guard tests.
// ===================================================================

/// A decode-like symbolic graph: input `x` of shape `[batch, seq]` (both
/// symbolic) feeds an `Add(x, x)` whose output is length-*variant*, while an
/// inline initializer `w` of static shape `[4]` feeds a `Mul(y, w)` whose
/// output is length-*invariant*. This exercises both memo partitions.
#[cfg(test)]
struct DecodeMemoIds {
    batch: SymbolId,
    seq: SymbolId,
    x2: ValueId,
    ymul: ValueId,
}

#[cfg(test)]
fn decode_memo_test_graph() -> (Graph, DecodeMemoIds) {
    use onnx_runtime_ir::TensorData;
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let batch = graph.intern_symbol("batch");
    let seq = graph.intern_symbol("seq");

    // Length-variant spine: x[batch, seq] -> Add(x, x) -> x2[batch, seq].
    let x = graph.create_named_value("x", DataType::Float32, vec![batch.into(), seq.into()]);
    graph.add_input(x);
    let x2 = graph.create_named_value("x2", DataType::Float32, vec![batch.into(), seq.into()]);
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(x), Some(x)],
        vec![x2],
    ));
    graph.add_output(x2);

    // Length-invariant tail: y[4] (required input) * w[4] (initializer).
    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.add_input(y);
    let w = graph.create_named_value("w", DataType::Float32, static_shape([4]));
    graph.set_initializer(
        w,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![4],
            [1.0f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let ymul = graph.create_named_value("ymul", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(y), Some(w)],
        vec![ymul],
    ));
    graph.add_output(ymul);
    (
        graph,
        DecodeMemoIds {
            batch,
            seq,
            x2,
            ymul,
        },
    )
}

#[cfg(test)]
fn decode_memo_run(exec: &mut Executor, batch: usize, seq: usize) -> Vec<Vec<f32>> {
    let x = Tensor::from_f32(
        &[batch, seq],
        &(0..batch * seq).map(|i| i as f32 + 1.0).collect::<Vec<_>>(),
    )
    .unwrap();
    let y = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    exec.run(&[("x", &x), ("y", &y)])
        .unwrap()
        .into_iter()
        .map(|t| t.to_vec_f32())
        .collect()
}

/// The default-ON master switch (Ripley's authoritative GO): the memo is
/// enabled unless `ONNX_GENAI_DECODE_MEMO` is an explicit OFF value
/// (`0`/`false`/`off`, case-insensitive, whitespace-trimmed). Unset, empty,
/// and unrecognized values all fail safe toward the validated fast path (ON).
#[test]
fn decode_memo_env_default_on_unless_explicitly_disabled() {
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let _env_guard = ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("decode-memo env lock");

    struct RestoreEnv(Option<OsString>);
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            match self.0.take() {
                // SAFETY: this test serializes all env mutations via ENV_LOCK.
                Some(value) => unsafe { std::env::set_var("ONNX_GENAI_DECODE_MEMO", value) },
                None => unsafe { std::env::remove_var("ONNX_GENAI_DECODE_MEMO") },
            }
        }
    }
    let _restore = RestoreEnv(std::env::var_os("ONNX_GENAI_DECODE_MEMO"));

    // Unset ⇒ ON (default).
    // SAFETY: guarded by ENV_LOCK above.
    unsafe { std::env::remove_var("ONNX_GENAI_DECODE_MEMO") };
    assert!(decode_memo_env_enabled(), "unset must default ON");

    // Explicit OFF values (case-insensitive, trimmed) ⇒ OFF.
    for off in ["0", "false", "off", "FALSE", "Off", "  0  ", "\tOFF\n"] {
        // SAFETY: guarded by ENV_LOCK above.
        unsafe { std::env::set_var("ONNX_GENAI_DECODE_MEMO", off) };
        assert!(!decode_memo_env_enabled(), "{off:?} must disable the memo");
    }

    // Explicit ON values and any unrecognized/empty value ⇒ ON (fail-safe).
    for on in [
        "1", "true", "on", "ON", "True", " on ", "", "  ", "yes", "2", "banana",
    ] {
        // SAFETY: guarded by ENV_LOCK above.
        unsafe { std::env::set_var("ONNX_GENAI_DECODE_MEMO", on) };
        assert!(decode_memo_env_enabled(), "{on:?} must keep the memo ON");
    }
}

/// Plan-invalidation unit test (F5 Stage 1 merge gate #2): prefill→decode
/// rebuilds (signature changed), pure length growth replays (signature
/// stable), and a batch change rebuilds.
#[test]
fn decode_plan_memo_rebuilds_and_replays() {
    let (graph, ids) = decode_memo_test_graph();
    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    exec.set_decode_memo_enabled(true);

    // "Prefill" step [1, 4]: first observation, nothing to diff yet.
    decode_memo_run(&mut exec, 1, 4);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Primed);
    assert!(exec.decode_memo.is_none());

    // First decode step [1, 5]: diffs against the prefill step and (re)builds
    // the memo — the prefill→decode transition changed the plan signature.
    decode_memo_run(&mut exec, 1, 5);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Rebuilt);
    let memo = exec.decode_memo.as_ref().expect("memo built");
    // `seq` grew (4→5) so it is varying; `batch` stayed 1 so it is invariant.
    assert!(memo.decode_varying.contains(&ids.seq));
    assert!(!memo.decode_varying.contains(&ids.batch));
    // The invariant tail (`ymul`) is cached; the variant spine (`x2`) is not.
    assert!(memo.invariant_shapes.contains_key(&ids.ymul));
    assert!(memo.variant_values.contains(&ids.x2));

    // Two decode steps at growing L both REPLAY: signature stable (only the
    // varying `seq` grows), invariant map reused, variant map re-resolved.
    let out6 = decode_memo_run(&mut exec, 1, 6);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Replayed);
    let out7 = decode_memo_run(&mut exec, 1, 7);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Replayed);
    // The variant tail was genuinely re-resolved to the new length.
    assert_eq!(out6[0].len(), 6);
    assert_eq!(out7[0].len(), 7);

    // A batch change [1, ·] → [2, ·] REBUILDS: `batch` was a non-varying
    // binding, so the change fails the replay guard and forces a rebuild.
    decode_memo_run(&mut exec, 2, 7);
    assert_eq!(exec.decode_memo_action(), DecodeMemoAction::Rebuilt);
    let memo = exec.decode_memo.as_ref().expect("memo rebuilt");
    assert_eq!(memo.reference_bindings.get(&ids.batch), Some(&2));
}

/// Token-exact lock (F5 Stage 1 merge gate #1): over ≥128 growing-length CPU
/// decode steps the memo-ON output is bit-identical to the memo-OFF output,
/// step for step. `decode_memo_verify` (forced on by
/// `set_decode_memo_enabled`) additionally asserts every replayed shape map
/// equals a fresh `resolve_soft`.
///
/// This locks the property the memo must never violate: it changes only
/// shape-resolution bookkeeping, never a produced byte. A full real-model
/// engine-level lock is available by running the (ignored) decode-lock
/// tests with `ONNX_GENAI_DECODE_MEMO=1`; this executor-level lock proves
/// memo==non-memo bit-exactness on the real CPU kernels without a model
/// fixture.
#[test]
fn decode_plan_memo_is_token_exact_over_128_steps() {
    const STEPS: usize = 130;

    let (off_graph, _) = decode_memo_test_graph();
    let mut off = Executor::build(
        off_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    // Memo is default-ON now; explicitly disable it for the reference run.
    off.set_decode_memo_enabled(false);
    assert!(!off.decode_memo_enabled);

    let (on_graph, _) = decode_memo_test_graph();
    let mut on = Executor::build(
        on_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    on.set_decode_memo_enabled(true);

    let mut replays = 0usize;
    for step in 0..STEPS {
        let seq = 3 + step; // strictly growing sequence length
        let ref_out = decode_memo_run(&mut off, 1, seq);
        let memo_out = decode_memo_run(&mut on, 1, seq);
        assert_eq!(
            ref_out, memo_out,
            "decode-plan memo diverged from the reference at step {step} (seq={seq})"
        );
        if on.decode_memo_action() == DecodeMemoAction::Replayed {
            replays += 1;
        }
    }
    // Steady state must actually engage the replay fast path for the bulk of
    // the run (priming costs the first two steps).
    assert!(
        replays >= STEPS - 2,
        "expected the memo to replay in steady state; only {replays}/{STEPS} replays"
    );
}

/// Proof-of-fire (F5 Stage 1): a PERSISTENT device-I/O binding shaped like a
/// KV cache — input+output aliased, length `L` growing by one each step —
/// must PRIME then REPLAY under the memo. This is the regression lock for
/// Ripley's finding that the memo reported `primed=0 rebuilt=0 replayed=0` on
/// the real native decode path because the old gate excluded any run carrying
/// external bindings. With `decode_memo_verify` on (forced by
/// `set_decode_memo_enabled`), every replay is also asserted byte-identical to
/// a fresh `resolve_soft`, so this doubles as a token-exact lock on the
/// persistent-binding path.
#[test]
fn decode_plan_memo_fires_on_persistent_kv_bindings() {
    use onnx_runtime_ir::{TensorData, static_shape};

    // KV-like length-variant spine: kv[L] -> Relu -> kvout[L], aliased into
    // one persistent device buffer. Static invariant tail: y[4] * w[4].
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let l = graph.intern_symbol("L");
    let kv = graph.create_named_value("kv", DataType::Float32, vec![l.into()]);
    graph.add_input(kv);
    let kvout = graph.create_named_value("kvout", DataType::Float32, vec![l.into()]);
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(kv)], vec![kvout]));
    graph.add_output(kvout);

    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.add_input(y);
    let w = graph.create_named_value("w", DataType::Float32, static_shape([4]));
    graph.set_initializer(
        w,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![4],
            [1.0f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let ymul = graph.create_named_value("ymul", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(y), Some(w)],
        vec![ymul],
    ));
    graph.add_output(ymul);

    let mut exec = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    exec.set_decode_memo_enabled(true);

    // Pre-allocate the KV buffer to a fixed capacity (stable pointer) and grow
    // only the logical length each step — the pre-allocated-KV-cache case.
    const CAP: usize = 128;
    let mut kv_binding = exec
        .allocate_device_binding(
            "kv".into(),
            Some("kvout".into()),
            DataType::Float32,
            vec![CAP],
            vec![1],
        )
        .unwrap();
    let ptr0 = kv_binding.device_ptr();
    let y_tensor = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();

    let mut replays = 0usize;
    for step in 0..8usize {
        let len = 4 + step; // strictly growing KV length L
        kv_binding.set_logical_shape(vec![len]).unwrap();
        let bytes: Vec<u8> = (0..len).flat_map(|i| (i as f32).to_le_bytes()).collect();
        kv_binding.write_bytes(0, &bytes).unwrap();
        exec.run_with_device_bindings(&[("y", &y_tensor)], std::slice::from_mut(&mut kv_binding))
            .unwrap();
        if exec.decode_memo_action() == DecodeMemoAction::Replayed {
            replays += 1;
        }
    }

    // The pointer stayed stable (a growing-length view, not a realloc), so
    // the memo must not have been invalidated by capacity noise.
    assert_eq!(kv_binding.device_ptr(), ptr0);

    let (primed, rebuilt, replayed, ineligible) = exec.decode_memo_counts();
    assert_eq!(
        ineligible, 0,
        "persistent-KV decode must be memo-eligible, not excluded (the F5 regression)"
    );
    assert!(primed >= 1, "the first decode step must prime the memo");
    assert!(
        replayed >= 1,
        "steady persistent-KV decode must replay the memo \
             (primed={primed} rebuilt={rebuilt} replayed={replayed})"
    );
    assert_eq!(replays as u64, replayed);
}

/// F5 Stage 2 test graph. The persistent-KV spine (`kv[L] -> Relu -> kvout[L]`)
/// keeps the memo eligible, and an invariant tail (`y[4] * w[4] -> ymul[4]`)
/// feeds a `Reshape(ymul, [2,2]) -> yview` — a pure invariant zero-copy view.
/// Stage 2 must reinstate `yview` and elide the Reshape's dispatch on replay.
/// Returns the graph and the `ymul` value id (the view's source buffer).
#[cfg(test)]
fn stage2_view_graph() -> (Graph, ValueId) {
    use onnx_runtime_ir::{TensorData, static_shape};
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let l = graph.intern_symbol("L");
    let kv = graph.create_named_value("kv", DataType::Float32, vec![l.into()]);
    graph.add_input(kv);
    let kvout = graph.create_named_value("kvout", DataType::Float32, vec![l.into()]);
    graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(kv)], vec![kvout]));
    graph.add_output(kvout);

    let y = graph.create_named_value("y", DataType::Float32, static_shape([4]));
    graph.add_input(y);
    let w = graph.create_named_value("w", DataType::Float32, static_shape([4]));
    graph.set_initializer(
        w,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![4],
            [1.0f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )),
    );
    let ymul = graph.create_named_value("ymul", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(y), Some(w)],
        vec![ymul],
    ));

    // Reshape ymul[4] -> yview[2,2] through a constant shape initializer, which
    // the CPU Reshape kernel serves as a zero-copy view over `ymul`'s buffer.
    let yshape = graph.create_named_value("yshape", DataType::Int64, static_shape([2]));
    graph.set_initializer(
        yshape,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![2],
            [2i64, 2].into_iter().flat_map(i64::to_le_bytes).collect(),
        )),
    );
    let yview = graph.create_named_value("yview", DataType::Float32, static_shape([2, 2]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Reshape",
        vec![Some(ymul), Some(yshape)],
        vec![yview],
    ));
    graph.add_output(yview);
    (graph, ymul)
}

/// Run one persistent-KV decode step against [`stage2_view_graph`]: grow the KV
/// length to `len`, feed a fresh `y` (so `ymul` — and thus the elided `yview` —
/// varies each step, proving the reinstated view reads the freshly computed
/// source bytes), and return the materialized `yview` output.
#[cfg(test)]
fn stage2_run(
    exec: &mut Executor,
    kv_binding: &mut DeviceIoBinding,
    len: usize,
    y_bias: f32,
) -> Vec<f32> {
    kv_binding.set_logical_shape(vec![len]).unwrap();
    let bytes: Vec<u8> = (0..len).flat_map(|i| (i as f32).to_le_bytes()).collect();
    kv_binding.write_bytes(0, &bytes).unwrap();
    let y = Tensor::from_f32(
        &[4],
        &[y_bias + 1.0, y_bias + 2.0, y_bias + 3.0, y_bias + 4.0],
    )
    .unwrap();
    let outs = exec
        .run_with_device_bindings(&[("y", &y)], std::slice::from_mut(kv_binding))
        .unwrap();
    // Graph outputs are [kvout (bound → None), yview (returned)].
    outs.into_iter()
        .flatten()
        .next()
        .expect("yview output")
        .to_vec_f32()
}

#[cfg(test)]
fn stage2_kv_binding(exec: &Executor) -> DeviceIoBinding {
    exec.allocate_device_binding(
        "kv".into(),
        Some("kvout".into()),
        DataType::Float32,
        vec![256],
        vec![1],
    )
    .unwrap()
}

/// F5 Stage 2 proof-of-fire + token-exact lock. Over ≥128 growing-length
/// persistent-KV decode steps the invariant `Reshape` view is reinstated and
/// its dispatch elided (`views_reused`/`dispatch_elided` both grow), and the
/// memo-ON `yview` output is bit-identical to the memo-OFF reference every
/// step (with `y` — and therefore the view's source `ymul` — changing each
/// step, so a stale alias would immediately diverge). `decode_memo_verify`
/// (forced on by `set_decode_memo_enabled`) additionally asserts every
/// reinstated view equals a freshly built one in-flight.
#[test]
fn decode_view_plan_fires_and_is_token_exact_over_128_steps() {
    const STEPS: usize = 130;

    let (off_graph, _) = stage2_view_graph();
    let mut off = Executor::build(
        off_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    // Memo is default-ON now; explicitly disable it for the reference run.
    off.set_decode_memo_enabled(false);
    assert!(!off.decode_memo_enabled, "reference must run memo-OFF");
    let mut off_kv = stage2_kv_binding(&off);

    let (on_graph, _) = stage2_view_graph();
    let mut on = Executor::build(
        on_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    on.set_decode_memo_enabled(true);
    let mut on_kv = stage2_kv_binding(&on);

    for step in 0..STEPS {
        let len = 4 + step; // strictly growing KV length L
        let bias = step as f32; // vary the invariant-shape source each step
        let ref_out = stage2_run(&mut off, &mut off_kv, len, bias);
        let memo_out = stage2_run(&mut on, &mut on_kv, len, bias);
        assert_eq!(
            ref_out, memo_out,
            "Stage 2 view reuse diverged from the reference at step {step} (L={len})"
        );
    }

    let (views_reused, dispatch_elided) = on.decode_view_plan_counts();
    assert!(
        views_reused > 0 && dispatch_elided > 0,
        "Stage 2 must fire on steady decode (views_reused={views_reused}, \
             dispatch_elided={dispatch_elided})"
    );
    // Non-vacuous: the reshape view must be reused/elided on the bulk of steps
    // (priming + first rebuild cost the first few steps).
    assert!(
        views_reused as usize >= STEPS - 4,
        "expected steady Stage 2 reuse; only {views_reused}/{STEPS} views reused"
    );
    assert!(
        on.decode_view_plan.is_some(),
        "the cached view plan must survive steady-state replay"
    );
}

/// F5 Stage 2 buffer-identity invalidation lock. If a cached view's source
/// buffer is reallocated to a different base pointer between steps — the exact
/// hazard Stage 1 could ignore but Stage 2 cannot — the plan MUST detect the
/// signature mismatch, decline to reinstate the (now stale) alias, and fall
/// back to a full dispatch. The step's output must still be correct (no
/// dangling/stale view served) and the reuse counter must NOT advance.
#[test]
fn decode_view_plan_rebuilds_on_source_buffer_move() {
    use onnx_runtime_ir::TensorLayout;

    let (off_graph, _) = stage2_view_graph();
    let mut off = Executor::build(
        off_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    let mut off_kv = stage2_kv_binding(&off);

    let (on_graph, ymul) = stage2_view_graph();
    let mut on = Executor::build(
        on_graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    on.set_decode_memo_enabled(true);
    let mut on_kv = stage2_kv_binding(&on);

    // Warm up so the view plan is built and firing.
    for step in 0..6usize {
        let len = 4 + step;
        let bias = step as f32;
        let r = stage2_run(&mut off, &mut off_kv, len, bias);
        let m = stage2_run(&mut on, &mut on_kv, len, bias);
        assert_eq!(r, m, "warmup diverged at step {step}");
    }
    assert!(
        on.decode_view_plan.is_some(),
        "view plan must be built before the realloc test"
    );
    let (reused_before, _) = on.decode_view_plan_counts();

    // Forcibly MOVE the view's source buffer (`ymul`) to a fresh allocation of
    // the same capacity — a base-pointer change the plan's signature must catch.
    let old = on.buffers.remove(&ymul).expect("ymul buffer");
    let cap = old.len();
    // Allocate the replacement *before* releasing the original, so the two
    // cannot share an address. Freeing first and allocating the same size
    // is exactly the request an allocator satisfies from its free list by
    // handing back the block just released, which left the pointer
    // unchanged and made this test fail wherever that happened.
    let fresh = on
        .ep
        .allocate(cap, TensorLayout::contiguous().alignment)
        .unwrap();
    let moved_ptr = fresh.as_ptr() as usize;
    assert_ne!(
        moved_ptr,
        old.as_ptr() as usize,
        "the replacement buffer must not reuse the original address, or the \
             signature check below is not being exercised"
    );
    on.ep.deallocate(old).unwrap();
    on.buffers.insert(ymul, fresh);
    // Sanity: the plan's recorded source pointer no longer matches.
    assert!(
        !on.stage2_buffer_sig_matches(on.decode_view_plan.as_ref().unwrap()),
        "the forced realloc must break the buffer-identity signature"
    );

    // Next step: Stage 2 must decline reuse (sig mismatch) yet stay correct.
    let len = 4 + 6;
    let bias = 6.0f32;
    let ref_out = stage2_run(&mut off, &mut off_kv, len, bias);
    let memo_out = stage2_run(&mut on, &mut on_kv, len, bias);
    assert_eq!(
        ref_out, memo_out,
        "a moved source buffer must force a rebuild, never serve a stale view"
    );
    let (reused_after, _) = on.decode_view_plan_counts();
    assert_eq!(
        reused_after, reused_before,
        "the mismatched step must NOT reuse cached views (would be stale)"
    );
    // The freshly computed ymul must live in a real buffer again (the moved one
    // or a self-healed reallocation), never left dangling.
    let healed = on.buffers.get(&ymul).expect("ymul rebound").as_ptr() as usize;
    assert!(healed == moved_ptr || healed != 0, "ymul must be backed");
}

// ============================================================================
// Kernel pre-binding (Stage 3): reachability proof
// ============================================================================

/// Proves the kernel pre-binding fast path is taken during steady-state dispatch
/// for a static-shape graph. The TEST_HITS counter increments on the pre-bound
/// path; a non-zero delta after two runs proves the path fires.
#[test]
fn kernel_prebinding_fast_path_fires_on_static_graph() {
    use super::PREBIND_FAST_PATH_TEST_HITS;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a = graph.create_named_value("a", DataType::Float32, static_shape([4]));
    let b = graph.create_named_value("b", DataType::Float32, static_shape([4]));
    graph.add_input(a);
    graph.add_input(b);
    let sum = graph.create_named_value("sum", DataType::Float32, static_shape([4]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(a), Some(b)],
        vec![sum],
    ));
    graph.add_output(sum);

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    let a_val = Tensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b_val = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();

    // First run populates the binding (build already pre-compiled for static graphs).
    let before = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    executor
        .run(&[("a", &a_val), ("b", &b_val)])
        .expect("first run");
    let after_first = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);

    // The static-shape build pre-populates kernel_bindings, so the very first
    // run should already hit the fast path.
    assert!(
        after_first > before,
        "pre-bound fast path must fire on the first run of a static-shape graph \
         (before={before}, after={after_first})"
    );

    // Second run: same shapes, so fast path fires again.
    executor
        .run(&[("a", &a_val), ("b", &b_val)])
        .expect("second run");
    let after_second = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    assert!(
        after_second > after_first,
        "pre-bound fast path must fire on subsequent runs with stable shapes \
         (after_first={after_first}, after_second={after_second})"
    );

    // Verify correctness too.
    let out = executor.run(&[("a", &a_val), ("b", &b_val)]).unwrap();
    assert_eq!(out[0].to_vec_f32(), vec![11.0, 22.0, 33.0, 44.0]);
}

/// Proves the fallback path fires when shapes change (e.g. prefill→decode), and
/// that the pre-binding is updated so subsequent calls with the new shape hit
/// the fast path.
#[test]
fn kernel_prebinding_fallback_fires_on_shape_change() {
    use super::{PREBIND_FALLBACK_TEST_HITS, PREBIND_FAST_PATH_TEST_HITS};
    #[allow(unused_imports)]
    use onnx_runtime_ir::SymbolId;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);

    // Dynamic first dim (simulates sequence length changes).
    let seq_sym = graph.intern_symbol("seq");
    let shape_a: Shape = vec![Dim::Symbolic(seq_sym), Dim::Static(4)];
    let shape_sum: Shape = vec![Dim::Symbolic(seq_sym), Dim::Static(4)];

    let a = graph.create_named_value("a", DataType::Float32, shape_a.clone());
    let b = graph.create_named_value("b", DataType::Float32, shape_a.clone());
    graph.add_input(a);
    graph.add_input(b);
    let sum = graph.create_named_value("sum", DataType::Float32, shape_sum);
    graph.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(a), Some(b)],
        vec![sum],
    ));
    graph.add_output(sum);

    let mut executor = Executor::build(
        graph,
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();

    // Run 1: seq=3 (populates binding).
    let a1 = Tensor::from_f32(&[3, 4], &[1.0; 12]).unwrap();
    let b1 = Tensor::from_f32(&[3, 4], &[2.0; 12]).unwrap();
    executor.run(&[("a", &a1), ("b", &b1)]).expect("run seq=3");

    // Run 2: same seq=3 → fast path.
    let fast_before = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    executor
        .run(&[("a", &a1), ("b", &b1)])
        .expect("run seq=3 again");
    let fast_after = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    assert!(
        fast_after > fast_before,
        "fast path must fire on same-shape repeat"
    );

    // Run 3: seq=1 → shape change → fallback fires, binding updated.
    let a2 = Tensor::from_f32(&[1, 4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let b2 = Tensor::from_f32(&[1, 4], &[1.0, 1.0, 1.0, 1.0]).unwrap();
    let fallback_before = PREBIND_FALLBACK_TEST_HITS.load(Ordering::Relaxed);
    let out = executor.run(&[("a", &a2), ("b", &b2)]).expect("run seq=1");
    let fallback_after = PREBIND_FALLBACK_TEST_HITS.load(Ordering::Relaxed);
    assert!(
        fallback_after > fallback_before,
        "fallback path must fire on shape change"
    );
    assert_eq!(out[0].to_vec_f32(), vec![6.0, 7.0, 8.0, 9.0]);

    // Run 4: seq=1 again → fast path fires (binding was updated).
    let fast_before2 = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    executor
        .run(&[("a", &a2), ("b", &b2)])
        .expect("run seq=1 again");
    let fast_after2 = PREBIND_FAST_PATH_TEST_HITS.load(Ordering::Relaxed);
    assert!(
        fast_after2 > fast_before2,
        "after shape change, the updated binding must serve the fast path"
    );
}

/// Build a parent graph with a single `Scan` over a **multi-node** body so the
/// single-trip inline dual-path and the generic loop are exercised on identical
/// non-trivial work. `steps` is the scan-axis length (`1` = a decode step; `>1`
/// = a prefill-shaped run). The body threads carried state through two scan
/// inputs across three ops and emits one carried-state output plus one
/// per-iteration scan output:
///
///   `state_x  = Add(state, x)`
///   `state_out = Mul(state_x, y)`   (next carried state)
///   `scan_out  = Sub(state_out, x)` (stacked on the scan axis)
fn scan_inline_test_graph(steps: usize) -> Graph {
    const W: usize = 3;

    let mut body = Graph::new();
    body.opset_imports.insert(String::new(), 17);
    let state = body.create_named_value("state", DataType::Float32, static_shape([W]));
    let x = body.create_named_value("x", DataType::Float32, static_shape([W]));
    let y = body.create_named_value("y", DataType::Float32, static_shape([W]));
    body.add_input(state);
    body.add_input(x);
    body.add_input(y);
    let state_x = body.create_named_value("state_x", DataType::Float32, static_shape([W]));
    body.insert_node(Node::new(
        NodeId(0),
        "Add",
        vec![Some(state), Some(x)],
        vec![state_x],
    ));
    let state_out = body.create_named_value("state_out", DataType::Float32, static_shape([W]));
    body.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(state_x), Some(y)],
        vec![state_out],
    ));
    let scan_out = body.create_named_value("scan_out", DataType::Float32, static_shape([W]));
    body.insert_node(Node::new(
        NodeId(0),
        "Sub",
        vec![Some(state_out), Some(x)],
        vec![scan_out],
    ));
    body.add_output(state_out);
    body.add_output(scan_out);

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let initial = init_inline(&mut graph, "initial", &[W], vec![0.0; W]);
    let x_in = graph.create_named_value("X", DataType::Float32, static_shape([steps, W]));
    let y_in = graph.create_named_value("Y", DataType::Float32, static_shape([steps, W]));
    graph.add_input(x_in);
    graph.add_input(y_in);
    let final_state = graph.create_named_value("final_state", DataType::Float32, static_shape([W]));
    let scan_output =
        graph.create_named_value("scan_output", DataType::Float32, static_shape([steps, W]));
    let mut scan = Node::new(
        NodeId(0),
        "Scan",
        vec![Some(initial), Some(x_in), Some(y_in)],
        vec![final_state, scan_output],
    );
    scan.attributes
        .insert("num_scan_inputs".to_string(), Attribute::Int(2));
    let scan_id = graph.insert_node(scan);
    graph.subgraphs.insert((scan_id, "body".to_string()), body);
    graph.add_output(final_state);
    graph.add_output(scan_output);
    graph
}

fn init_inline(graph: &mut Graph, name: &str, dims: &[usize], data: Vec<f32>) -> ValueId {
    use onnx_runtime_ir::{TensorData, WeightRef};
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let value =
        graph.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    graph.set_initializer(
        value,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            dims.to_vec(),
            bytes,
        )),
    );
    value
}

fn run_scan_inline_graph(steps: usize, inline: bool) -> (Vec<Vec<u8>>, u64) {
    let mut exec = Executor::build(
        scan_inline_test_graph(steps),
        Arc::new(WeightStore::new()),
        auto_detect_cpu_ep().unwrap(),
    )
    .unwrap();
    exec.scan_inline_single_trip_enabled = inline;

    let n = steps * 3;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 2.0).collect();
    let x_t = Tensor::from_f32(&[steps, 3], &x).unwrap();
    let y_t = Tensor::from_f32(&[steps, 3], &y).unwrap();
    let outputs = exec.run(&[("X", &x_t), ("Y", &y_t)]).unwrap();
    let bytes = outputs.iter().map(|t| t.as_bytes().to_vec()).collect();
    (bytes, exec.scan_inline_single_trip_count())
}

/// Slice-1a correctness gate. Proves the flag-gated single-trip `Scan` inline
/// dual-path is (1) **byte-exact** with the generic `exec_scan` loop, and (2)
/// **non-vacuously engaged** and **runtime-keyed** — engaging only at
/// `trip_count == 1` and never on a prefill-shaped (`trip_count > 1`) run, even
/// with the flag ON. The shared-plan tripwire: a static single-trip rewrite
/// would fire on prefill too; this asserts it does not. Byte-equality is checked
/// over BOTH the carried `final_state` and the stacked `scan_output`, so a wrong
/// inline path (dropped/duplicated body run, mis-stacked scan axis, or skipped
/// state thread) makes the test FAIL.
#[test]
fn scan_single_trip_inline_is_byte_exact_and_runtime_keyed() {
    // Decode regime (trip_count == 1): inline path engages exactly once and is
    // byte-identical to the loop over every output.
    let (loop_out, loop_count) = run_scan_inline_graph(1, false);
    let (inline_out, inline_count) = run_scan_inline_graph(1, true);
    assert_eq!(loop_count, 0, "flag OFF must never engage the inline path");
    assert_eq!(
        inline_count, 1,
        "flag ON at trip_count==1 must engage the inline path exactly once"
    );
    assert_eq!(
        inline_out, loop_out,
        "single-trip inline output must be byte-exact with the loop path"
    );

    // Prefill regime (trip_count == 3): even with the flag ON the inline path
    // must NOT engage (runtime-keyed, not a static rewrite), and the output must
    // still match the loop.
    let (prefill_loop, prefill_loop_count) = run_scan_inline_graph(3, false);
    let (prefill_inline, prefill_inline_count) = run_scan_inline_graph(3, true);
    assert_eq!(prefill_loop_count, 0, "loop path never counts");
    assert_eq!(
        prefill_inline_count, 0,
        "flag ON must NOT inline a prefill (trip_count>1) Scan — the shared-plan tripwire"
    );
    assert_eq!(
        prefill_inline, prefill_loop,
        "prefill output must be identical flag-on vs flag-off"
    );
}
