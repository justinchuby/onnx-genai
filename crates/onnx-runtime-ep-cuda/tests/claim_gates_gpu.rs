#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! CUDA placement regressions for constrained GLM standard operators.

use onnx_runtime_ep_api::{ExecutionProvider, KernelMatch};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};

fn node(
    op_type: &str,
    input_dtypes: &[DataType],
    outputs: usize,
    omitted_input: Option<usize>,
    attrs: &[(&str, Attribute)],
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    let inputs = input_dtypes
        .iter()
        .enumerate()
        .map(|(index, &dtype)| {
            let value =
                graph.create_named_value(format!("input_{index}"), dtype, static_shape([1]));
            graph.add_input(value);
            if omitted_input == Some(index) {
                None
            } else {
                Some(value)
            }
        })
        .collect();
    let outputs = (0..outputs)
        .map(|index| {
            graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                static_shape([1]),
            )
        })
        .collect();
    let mut node = Node::new(NodeId(0), op_type, inputs, outputs);
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let id = graph.insert_node(node);
    (graph, id)
}

fn assert_rejected(
    ep: &CudaExecutionProvider,
    op_type: &str,
    opset: u64,
    input_dtypes: &[DataType],
    outputs: usize,
) {
    let (graph, id) = node(op_type, input_dtypes, outputs, None, &[]);
    assert!(
        matches!(
            ep.supports_op(graph.node(id), opset, &[], input_dtypes, &[]),
            KernelMatch::Unsupported { .. }
        ),
        "{op_type} must reject its unsupported input dtype at claim time"
    );
}

fn assert_supported(
    ep: &CudaExecutionProvider,
    op_type: &str,
    opset: u64,
    input_dtypes: &[DataType],
    outputs: usize,
) {
    let (graph, id) = node(op_type, input_dtypes, outputs, None, &[]);
    assert!(
        ep.supports_op(graph.node(id), opset, &[], input_dtypes, &[])
            .is_supported(),
        "{op_type} must claim its supported input dtype combination"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn glm_standard_claim_gates_reject_runtime_unsupported_input_dtypes() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");

    assert_supported(
        &ep,
        "RMSNormalization",
        23,
        &[DataType::BFloat16, DataType::Float32],
        1,
    );
    assert_rejected(
        &ep,
        "RMSNormalization",
        23,
        &[DataType::Float64, DataType::Float32],
        1,
    );
    let fp16_dtypes = [DataType::Float16, DataType::Float32];
    let (fp16_graph, fp16_id) = node("RMSNormalization", &fp16_dtypes, 1, None, &[]);
    assert!(
        ep.supports_op(fp16_graph.node(fp16_id), 23, &[], &fp16_dtypes, &[])
            .is_supported(),
        "RMSNormalization must claim fp16 activations with f32 scale"
    );
    assert_rejected(
        &ep,
        "RotaryEmbedding",
        23,
        &[
            DataType::Float32,
            DataType::Float32,
            DataType::Float32,
            DataType::Int32,
        ],
        1,
    );
    assert_rejected(&ep, "TopK", 24, &[DataType::Float32, DataType::Int32], 2);
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let dtypes = [dtype, DataType::Int64];
        let (graph, id) = node("TopK", &dtypes, 2, None, &[]);
        assert!(
            ep.supports_op(graph.node(id), 24, &[], &dtypes, &[])
                .is_supported(),
            "TopK must claim {dtype:?} router values"
        );
    }
    assert_rejected(&ep, "CumSum", 24, &[DataType::Float32, DataType::Int32], 1);
    assert_rejected(
        &ep,
        "Gather",
        24,
        &[DataType::Float32, DataType::Float32],
        1,
    );
    assert_rejected(
        &ep,
        "GatherElements",
        24,
        &[DataType::Float32, DataType::Int32],
        1,
    );
    for data in [
        DataType::Float16,
        DataType::Float32,
        DataType::BFloat16,
        DataType::Int64,
    ] {
        for indices in [DataType::Int32, DataType::Int64] {
            assert_supported(&ep, "ScatterElements", 24, &[data, indices, data], 1);
        }
    }
    assert_rejected(
        &ep,
        "Where",
        24,
        &[DataType::Int64, DataType::Float32, DataType::Float32],
        1,
    );
    assert_rejected(&ep, "Expand", 24, &[DataType::Float32, DataType::Int32], 1);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn optional_glm_inputs_distinguish_omission_from_wrong_dtype() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    let dtypes = [
        DataType::Float32,
        DataType::Float32,
        DataType::Float32,
        DataType::Undefined,
    ];
    let (omitted_graph, omitted_id) = node("RotaryEmbedding", &dtypes, 1, Some(3), &[]);
    assert!(
        ep.supports_op(omitted_graph.node(omitted_id), 23, &[], &dtypes, &[])
            .is_supported(),
        "an omitted RotaryEmbedding position_ids must be claimed"
    );

    let present_dtypes = [
        DataType::Float32,
        DataType::Float32,
        DataType::Float32,
        DataType::Int32,
    ];
    let (present_graph, present_id) = node("RotaryEmbedding", &present_dtypes, 1, None, &[]);
    assert!(matches!(
        ep.supports_op(
            present_graph.node(present_id),
            23,
            &[],
            &present_dtypes,
            &[]
        ),
        KernelMatch::Unsupported { ref reason } if reason.contains("position_ids")
    ));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn claim_gates_reject_attributes_cuda_would_otherwise_silently_coerce() {
    let ep = CudaExecutionProvider::new_default().expect("CUDA runtime must be available");
    for (op_type, opset, input_dtypes, outputs, attrs) in [
        (
            "RMSNormalization",
            23,
            vec![DataType::Float32, DataType::Float32],
            1,
            vec![("stash_type", Attribute::Int(16))],
        ),
        (
            "RotaryEmbedding",
            23,
            vec![DataType::Float32, DataType::Float32, DataType::Float32],
            1,
            vec![("num_heads", Attribute::Int(-1))],
        ),
        (
            "TopK",
            24,
            vec![DataType::Float32, DataType::Int64],
            2,
            vec![("largest", Attribute::Int(2))],
        ),
        (
            "CumSum",
            24,
            vec![DataType::Float32, DataType::Int64],
            1,
            vec![("exclusive", Attribute::Int(-1))],
        ),
        (
            "ScatterElements",
            24,
            vec![DataType::Float32, DataType::Int64, DataType::Float32],
            1,
            vec![("reduction", Attribute::String(b"overwrite".to_vec()))],
        ),
    ] {
        let (graph, id) = node(op_type, &input_dtypes, outputs, None, &attrs);
        assert!(
            matches!(
                ep.supports_op(graph.node(id), opset, &[], &input_dtypes, &[]),
                KernelMatch::Unsupported { .. }
            ),
            "{op_type} must reject an invalid attribute at claim time"
        );
    }
}

/// `com.microsoft::GatherBlockQuantized` with sub-byte zero points only matches
/// CPU/ORT numerics for an EVEN number of blocks per row (the CUDA kernel uses
/// global nibble addressing for zero points; CPU/ORT pack per row). The claim
/// gate must decline an ODD blocks-per-row layout explicitly, and must still
/// claim the even layout.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn gather_block_quantized_odd_blocks_per_row_with_zero_points_declines() {
    let Ok(ep) = CudaExecutionProvider::new_default() else {
        eprintln!("skip: no CUDA GPU available");
        panic!(
            "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
        );
    };

    // Build a GBQ node: data, indices, scales, zero_points (bits=4, block_size=16).
    let build = |data_last: usize| -> (Graph, NodeId, Vec<onnx_runtime_ir::Shape>) {
        let mut graph = Graph::new();
        let data = graph.create_named_value("data", DataType::Uint8, static_shape([4, data_last]));
        let indices = graph.create_named_value("indices", DataType::Int64, static_shape([2usize]));
        let scales = graph.create_named_value("scales", DataType::Float32, static_shape([1usize]));
        let zero_points =
            graph.create_named_value("zero_points", DataType::Uint8, static_shape([1usize]));
        for v in [data, indices, scales, zero_points] {
            graph.add_input(v);
        }
        let out = graph.create_named_value("out", DataType::Float32, static_shape([1usize]));
        let mut n = Node::new(
            NodeId(0),
            "GatherBlockQuantized",
            vec![Some(data), Some(indices), Some(scales), Some(zero_points)],
            vec![out],
        );
        n.domain = "com.microsoft".into();
        n.attributes.insert("bits".into(), Attribute::Int(4));
        n.attributes.insert("block_size".into(), Attribute::Int(16));
        n.attributes.insert("gather_axis".into(), Attribute::Int(0));
        n.attributes
            .insert("quantize_axis".into(), Attribute::Int(1));
        let id = graph.insert_node(n);
        let shapes = vec![
            static_shape([4usize, data_last]),
            static_shape([2usize]),
            static_shape([1usize]),
            static_shape([1usize]),
        ];
        (graph, id, shapes)
    };

    let dtypes = [
        DataType::Uint8,
        DataType::Int64,
        DataType::Float32,
        DataType::Uint8,
    ];

    // data_last=8 → after_gather_dim = 8*2 = 16, block_size 16 → 1 block/row (ODD).
    let (odd_graph, odd_id, odd_shapes) = build(8);
    let odd = ep.supports_op(odd_graph.node(odd_id), 1, &odd_shapes, &dtypes, &[]);
    assert!(
        matches!(&odd, KernelMatch::Unsupported { reason } if reason.contains("blocks per row")),
        "odd blocks-per-row GBQ with zero points must decline loudly"
    );

    // data_last=16 → after_gather_dim = 16*2 = 32, block_size 16 → 2 blocks/row (EVEN).
    let (even_graph, even_id, even_shapes) = build(16);
    assert!(
        ep.supports_op(even_graph.node(even_id), 1, &even_shapes, &dtypes, &[])
            .is_supported(),
        "even blocks-per-row GBQ with zero points must still be claimed"
    );
}
