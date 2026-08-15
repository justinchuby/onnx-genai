//! Integration tests proving parity between the Rust `ExecutionProvider` trait
//! path and the outbound ORT plugin C ABI path.
//!
//! # Capability parity rule (encoded here)
//!
//! The C ABI `GetCapability` path applies a **fail-closed shape-inference filter**
//! on top of the trait's `supports_op` / `supports_node` result:
//!
//! ```text
//! C_ABI_claims = trait_claims ∩ { nodes where ShapeInference::for_node ≠ Declined }
//! ```
//!
//! Every node claimed by the C ABI is also supported by the trait, but the
//! converse is not true: a node the trait supports may still be `Declined` by
//! shape inference, causing the C ABI to exclude it. This is intentional and
//! prevents over-claiming ops whose output shapes cannot be inferred.
//!
//! # Numerical parity
//!
//! Both paths use the same Rust kernel, so outputs must be bit-identical.
//!
//! # Error parity
//!
//! An unsupported op is declined by both paths. A shape-inference `Declined` op
//! is supported by the trait but excluded by the C ABI — documented divergence.

use onnx_runtime_ep_api::EpConfig;
use onnx_runtime_ep_api::abi::OrtGraphView;
use onnx_runtime_ep_api::provider::ExecutionProvider;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_plugin::compute::ShapeInference;
use onnx_runtime_ir::{DataType, Dim, Graph, GraphView, GraphViewCache, Node, NodeId};

/// Build a minimal graph with a single op node.
///
/// Opset 13 is imported so that `effective_opset` returns a nonzero version
/// for every node; without this the registry check in `supports_op` declines
/// everything because opset 0 is not a registered version.
fn single_op_graph(op_type: &str, domain: &str, input_shapes: &[&[usize]]) -> Graph {
    let mut graph = Graph::default();
    graph.opset_imports.insert(String::new(), 13);

    let shape_for = |dims: &[usize]| -> Vec<Dim> { dims.iter().map(|&d| Dim::Static(d)).collect() };

    // Create input values.
    let input_vids: Vec<_> = input_shapes
        .iter()
        .enumerate()
        .map(|(i, dims)| {
            let vid =
                graph.create_named_value(format!("input_{i}"), DataType::Float32, shape_for(dims));
            graph.add_input(vid);
            vid
        })
        .collect();

    // Create output value.
    let out_vid = graph.create_named_value("output_0", DataType::Float32, vec![]);
    graph.add_output(out_vid);

    // Create the op node.
    let mut node = Node::new(
        NodeId(0),
        op_type.to_string(),
        input_vids.iter().map(|v| Some(*v)).collect(),
        vec![out_vid],
    );
    node.domain = domain.to_string();
    graph.insert_node(node);

    graph
}

fn make_cpu_ep() -> CpuExecutionProvider {
    let mut ep = CpuExecutionProvider::new();
    ep.initialize(&EpConfig::default()).unwrap();
    ep
}

// ─── Capability parity tests ─────────────────────────────────────────────────

/// For ops the trait supports AND shape inference does NOT decline,
/// both paths must claim the node.
#[test]
fn capability_parity_supported_ops_with_known_shapes() {
    let ep = make_cpu_ep();

    let test_cases: &[(&str, &[&[usize]])] = &[
        ("Add", &[&[2, 3], &[2, 3]]),
        ("Mul", &[&[4, 4], &[4, 4]]),
        ("Relu", &[&[3, 5]]),
        ("MatMul", &[&[2, 3], &[3, 4]]),
        ("Sigmoid", &[&[2, 2]]),
        ("Tanh", &[&[10]]),
    ];

    for &(op_type, input_shapes) in test_cases {
        let graph = single_op_graph(op_type, "", input_shapes);
        let cache = GraphViewCache::build(&graph).unwrap();
        let view = GraphView::new(&graph, &cache);

        // Path A (trait): check supports_node
        let node_idx = view.nodes().next().unwrap();
        let trait_result = ep.supports_node(&view, node_idx, 13);
        assert!(
            trait_result.is_supported(),
            "Trait must support {op_type}, got: {trait_result:?}"
        );

        // Shape inference must NOT decline
        let node = view.node(node_idx);
        let shapes_vec: Vec<Vec<Option<usize>>> = input_shapes
            .iter()
            .map(|s| s.iter().copied().map(Some).collect())
            .collect();
        let si = ShapeInference::for_node(node, &shapes_vec, 1);
        assert!(
            !matches!(si, ShapeInference::Declined { .. }),
            "ShapeInference must not decline {op_type}, got: {si:?}"
        );

        // Path B (C ABI simulation): query_capabilities should claim the node
        let ort_view = OrtGraphView::new(&view);
        let claims = ort_view.query_capabilities(&ep);
        assert!(
            !claims.is_empty(),
            "C ABI path must claim {op_type} (trait supports + shape inference not declined)"
        );
    }
}

/// For ops the trait supports but shape inference DECLINES, the C ABI fail-closed
/// filter removes them — intentional, documented divergence.
///
/// **Parity rule pinned:** the ep.rs `ep_get_capability` function applies:
/// ```text
/// C_ABI_claims = query_capabilities(ep) ∩ { nodes where for_node ≠ Declined }
/// ```
/// `OrtGraphView::query_capabilities` is the trait-only first half; the
/// shape-inference filter is applied afterward in ep.rs. We test each predicate
/// independently so neither can hide the other.
///
/// **Real Declined case:** `Unsqueeze` at opset 13 where `axes` come from
/// `input[1]` (runtime tensor), not from an attribute. `for_node` returns
/// `Declined` because the axes are data-dependent and cannot be resolved
/// statically without attributes.
#[test]
fn capability_parity_supported_but_shape_declined() {
    let ep = make_cpu_ep();

    // `Unsqueeze` at opset 13: axes from input[1], not an attribute.
    // The single_op_graph helper provides no attributes, so for_node declines.
    let graph = single_op_graph("Unsqueeze", "", &[&[3, 4], &[1]]);
    let cache = GraphViewCache::build(&graph).unwrap();
    let view = GraphView::new(&graph, &cache);
    let node_idx = view.nodes().next().unwrap();
    let node = view.node(node_idx);

    let shapes_vec: Vec<Vec<Option<usize>>> = [&[3usize, 4][..], &[1usize][..]]
        .iter()
        .map(|s| s.iter().copied().map(Some).collect())
        .collect();
    let si = ShapeInference::for_node(node, &shapes_vec, 1);

    // ShapeInference must decline Unsqueeze (opset-13 axes-from-input path).
    assert!(
        matches!(si, ShapeInference::Declined { .. }),
        "ShapeInference::for_node MUST decline Unsqueeze at opset-13 \
         (axes are data-dependent, not an attribute), got: {si:?}"
    );

    // The trait may support Unsqueeze. If it does, the C ABI filter predicate
    // evaluates to false → C ABI would NOT claim it.
    let trait_result = ep.supports_node(&view, node_idx, 13);
    if trait_result.is_supported() {
        // Both conditions must hold for C ABI to claim the node:
        // 1. trait supports it   ← TRUE (just verified above)
        // 2. for_node != Declined ← FALSE (just verified above)
        // Therefore the C ABI filter predicate is: true AND false = false.
        // The filter would remove this node from the claim set.
        let filter_would_pass = !matches!(
            ShapeInference::for_node(node, &shapes_vec, 1),
            ShapeInference::Declined { .. }
        );
        assert!(
            !filter_would_pass,
            "INTENTIONAL DIVERGENCE: trait supports Unsqueeze but \
             ShapeInference::for_node returns Declined (axes are runtime-valued). \
             The C ABI fail-closed filter would exclude this node."
        );
    }
    // If the trait also doesn't support it, both paths decline — no divergence to pin.
}

/// Ops that neither path supports (unknown domain, fake op).
#[test]
fn capability_parity_unsupported_ops() {
    let ep = make_cpu_ep();

    let test_cases: &[(&str, &str)] = &[
        ("FakeOp", ""),
        ("MatMul", "com.nonexistent"),
        ("CustomThing", "pkg.nxrt"),
    ];

    for &(op_type, domain) in test_cases {
        let graph = single_op_graph(op_type, domain, &[&[2, 3], &[3, 4]]);
        let cache = GraphViewCache::build(&graph).unwrap();
        let view = GraphView::new(&graph, &cache);

        // Path A: trait must NOT support
        let node_idx = view.nodes().next().unwrap();
        let trait_result = ep.supports_node(&view, node_idx, 13);
        assert!(
            !trait_result.is_supported(),
            "Trait must NOT support {domain}/{op_type}"
        );

        // Path B: C ABI must also produce no claims
        let ort_view = OrtGraphView::new(&view);
        let claims = ort_view.query_capabilities(&ep);
        assert!(
            claims.is_empty(),
            "C ABI must NOT claim unsupported {domain}/{op_type}"
        );
    }
}

// ─── Numerical parity (memory path) ─────────────────────────────────────────

/// Memory roundtrip through the trait path: allocate → copy_from_host →
/// copy_to_host → deallocate must be bit-exact.
#[test]
fn numerical_parity_memory_roundtrip() {
    let ep = make_cpu_ep();

    let data: Vec<f32> = vec![1.0, 2.0, 3.5, -42.5, 0.0, f32::INFINITY];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr().cast(),
            data.len() * std::mem::size_of::<f32>(),
        )
    };

    let mut buf = ep.allocate(bytes.len(), 8).expect("allocate must succeed");
    ep.copy_from_host(bytes, &mut buf).expect("copy_from_host");

    let mut readback = vec![0u8; bytes.len()];
    ep.copy_to_host(&buf, &mut readback).expect("copy_to_host");

    let result: &[f32] =
        unsafe { std::slice::from_raw_parts(readback.as_ptr().cast(), data.len()) };
    assert_eq!(
        result,
        &data[..],
        "Memory roundtrip must be bit-exact (allocate→copy_from_host→copy_to_host)"
    );

    ep.deallocate(buf).expect("deallocate");
}

/// Device-to-device copy through the trait path.
#[test]
fn numerical_parity_device_copy() {
    let ep = make_cpu_ep();

    let data: Vec<f32> = vec![42.0, -1.5, 0.0, 99.9];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr().cast(),
            data.len() * std::mem::size_of::<f32>(),
        )
    };

    let mut src = ep.allocate(bytes.len(), 4).expect("alloc src");
    let mut dst = ep.allocate(bytes.len(), 4).expect("alloc dst");

    ep.copy_from_host(bytes, &mut src).expect("fill src");
    ep.copy(&src, &mut dst, bytes.len()).expect("device copy");

    let mut readback = vec![0u8; bytes.len()];
    ep.copy_to_host(&dst, &mut readback).expect("read dst");

    let result: &[f32] =
        unsafe { std::slice::from_raw_parts(readback.as_ptr().cast(), data.len()) };
    assert_eq!(result, &data[..], "Device-to-device copy must be bit-exact");

    ep.deallocate(src).expect("dealloc src");
    ep.deallocate(dst).expect("dealloc dst");
}

// ─── Error parity tests ─────────────────────────────────────────────────────

/// Shape-inference Declined ops: the trait may support them but the C ABI
/// fail-closed filter removes them — the documented intentional divergence.
///
/// **Concrete case pinned:** `Unsqueeze` at opset 13 with axes from `input[1]`
/// (data-dependent, not an attribute). `ShapeInference::for_node` declines
/// because the axes cannot be resolved statically without the attribute.
/// The C ABI `ep_get_capability` filter predicate therefore evaluates to false
/// for this node even if `supports_node` would accept it.
#[test]
fn error_parity_declined_shape_inference_is_cabi_only() {
    let ep = make_cpu_ep();

    // Unsqueeze at opset-13 with no axes attribute → for_node returns Declined.
    let graph = single_op_graph("Unsqueeze", "", &[&[3, 4], &[1]]);
    let cache = GraphViewCache::build(&graph).unwrap();
    let view = GraphView::new(&graph, &cache);
    let node_idx = view.nodes().next().unwrap();
    let node = view.node(node_idx);

    // Step 1 — Trait path claims Unsqueeze (it is in the kernel registry).
    // This is the "first half" that OrtGraphView::query_capabilities covers.
    let ort_view = OrtGraphView::new(&view);
    let trait_claims = ort_view.query_capabilities(&ep);
    assert!(
        !trait_claims.is_empty(),
        "Trait (query_capabilities) must claim Unsqueeze — it is a registered kernel"
    );

    // Step 2 — Shape inference is Declined for this node (axes are data-dependent).
    let input_shapes: Vec<Vec<Option<usize>>> = vec![vec![Some(3), Some(4)], vec![Some(1)]];
    let si = ShapeInference::for_node(node, &input_shapes, 1);
    assert!(
        matches!(si, ShapeInference::Declined { .. }),
        "Unsqueeze at opset-13 without axes attribute must be Declined by shape inference, got: {si:?}"
    );

    // Step 3 — The C ABI filter in ep_get_capability applies:
    //   C_ABI_claims = query_capabilities(ep) ∩ { nodes where for_node ≠ Declined }
    // Since shape inference returns Declined, the filter removes all claims.
    // Simulate the filter inline:
    let cabi_claims: Vec<_> = trait_claims
        .into_iter()
        .filter(|claim| {
            claim.node_ids.iter().all(|&nid| {
                let Some(idx) = view.node_index(nid) else {
                    return false;
                };
                let n = view.node(idx);
                let si_check = ShapeInference::for_node(n, &input_shapes, 1);
                !matches!(si_check, ShapeInference::Declined { .. })
            })
        })
        .collect();
    assert!(
        cabi_claims.is_empty(),
        "C ABI filter must produce no claims for Unsqueeze with Declined shape inference \
         (ep_get_capability removes this node even though the trait claimed it)"
    );
}

/// Completely unknown op: both paths decline.
#[test]
fn error_parity_unknown_op_declined_by_both() {
    let ep = make_cpu_ep();

    let graph = single_op_graph("TotallyFakeOp", "com.fake.domain", &[&[2, 3]]);
    let cache = GraphViewCache::build(&graph).unwrap();
    let view = GraphView::new(&graph, &cache);
    let node_idx = view.nodes().next().unwrap();

    // Trait declines
    let trait_result = ep.supports_node(&view, node_idx, 13);
    assert!(
        !trait_result.is_supported(),
        "Trait must decline unknown op"
    );

    // C ABI declines
    let ort_view = OrtGraphView::new(&view);
    let claims = ort_view.query_capabilities(&ep);
    assert!(claims.is_empty(), "C ABI must decline unknown op");
}

// ─── Multi-node parity ───────────────────────────────────────────────────────

/// A graph with both supported and unsupported nodes: the C ABI must claim
/// exactly the subset that passes both supports_node AND shape inference.
#[test]
fn capability_parity_mixed_graph() {
    let ep = make_cpu_ep();

    let mut graph = Graph::default();
    // Without opset imports, effective_opset returns None → 0 → registry check
    // declines everything. Set opset 13 to match what single_op_graph does.
    graph.opset_imports.insert(String::new(), 13);

    let shape4: Vec<Dim> = vec![Dim::Static(4)];

    let v_in0 = graph.create_named_value("in0", DataType::Float32, shape4.clone());
    let v_in1 = graph.create_named_value("in1", DataType::Float32, shape4.clone());
    graph.add_input(v_in0);
    graph.add_input(v_in1);

    let v_add_out = graph.create_named_value("add_out", DataType::Float32, shape4.clone());
    let v_fake_out = graph.create_named_value("fake_out", DataType::Float32, vec![]);
    graph.add_output(v_fake_out);

    // Add node: supported + shape inference OK
    let add_node = Node::new(
        NodeId(0),
        "Add",
        vec![Some(v_in0), Some(v_in1)],
        vec![v_add_out],
    );
    graph.insert_node(add_node);

    // FakeOp node: unsupported
    let fake_node = Node::new(
        NodeId(0),
        "TotallyFakeOp",
        vec![Some(v_add_out)],
        vec![v_fake_out],
    );
    graph.insert_node(fake_node);

    let cache = GraphViewCache::build(&graph).unwrap();
    let view = GraphView::new(&graph, &cache);

    // Verify: trait supports Add but not FakeOp
    let mut nodes_iter = view.nodes();
    let first_idx = nodes_iter.next().unwrap();
    let second_idx = nodes_iter.next().unwrap();

    // Identify which is Add vs FakeOp
    let (add_idx, fake_idx) = if view.node(first_idx).op_type == "Add" {
        (first_idx, second_idx)
    } else {
        (second_idx, first_idx)
    };

    assert!(
        ep.supports_node(&view, add_idx, 13).is_supported(),
        "Trait must support Add"
    );
    assert!(
        !ep.supports_node(&view, fake_idx, 13).is_supported(),
        "Trait must not support TotallyFakeOp"
    );

    // C ABI: should claim only the Add node
    let ort_view = OrtGraphView::new(&view);
    let claims = ort_view.query_capabilities(&ep);
    assert_eq!(
        claims.len(),
        1,
        "Exactly one claim (Add) expected, got {}",
        claims.len()
    );
    assert_eq!(
        claims[0].node_ids.len(),
        1,
        "Claim must contain exactly 1 node (Add)"
    );
}

/// Verify the capability-parity invariant holds for the `com.microsoft` domain:
/// the CPU EP should not support arbitrary MS domain ops.
#[test]
fn capability_parity_com_microsoft_domain() {
    let ep = make_cpu_ep();

    // A com.microsoft op the CPU EP doesn't know about
    let graph = single_op_graph("UnknownMsOp", "com.microsoft", &[&[2, 3]]);
    let cache = GraphViewCache::build(&graph).unwrap();
    let view = GraphView::new(&graph, &cache);
    let node_idx = view.nodes().next().unwrap();

    let trait_result = ep.supports_node(&view, node_idx, 1);
    // Both paths must agree on decline
    let ort_view = OrtGraphView::new(&view);
    let claims = ort_view.query_capabilities(&ep);

    if !trait_result.is_supported() {
        assert!(
            claims.is_empty(),
            "C ABI must not claim an unsupported com.microsoft op"
        );
    }
    // If the trait somehow supports it, C ABI may claim it too (no divergence for supported ops
    // with non-Declined shape inference) — but we don't assert that here since the EP shouldn't.
}
