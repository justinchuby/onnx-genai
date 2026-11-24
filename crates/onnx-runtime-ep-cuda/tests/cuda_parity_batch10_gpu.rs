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
//! GPU parity for issue #67 CUDA coverage batch 10.

mod common;

use common::{
    assert_close, build_graph, decode_floats, float_input, input, require_cuda, run_cpu, run_cuda,
};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Model, static_shape};

fn tolerance(dtype: DataType) -> f32 {
    match dtype {
        DataType::Float32 => 2e-5,
        DataType::Float16 => 4e-3,
        DataType::BFloat16 => 4e-2,
        _ => 0.0,
    }
}

fn assert_float_outputs(label: &str, dtype: DataType, cuda: &[Vec<u8>], cpu: &[Vec<u8>]) {
    assert_close(
        label,
        dtype,
        &decode_floats(&cuda[0], dtype),
        &decode_floats(&cpu[0], dtype),
        tolerance(dtype),
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn affine_grid_matches_cpu_for_two_and_three_dimensional_grids() {
    let ep = require_cuda();
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for (theta_shape, theta, size, output_shape, align_corners) in [
            (
                vec![1, 2, 3],
                vec![1.0, 0.25, -0.5, 0.1, 1.0, 0.75],
                vec![1i64, 1, 1, 3],
                vec![1, 1, 3, 2],
                1i64,
            ),
            (
                vec![1, 3, 4],
                vec![
                    1.0, 0.0, 0.25, -0.5, 0.1, 1.0, 0.0, 0.75, 0.0, 0.2, 1.0, -0.25,
                ],
                vec![1, 1, 2, 2, 3],
                vec![1, 2, 2, 3, 3],
                0,
            ),
        ] {
            let inputs = vec![
                float_input(dtype, &theta_shape, &theta),
                input(DataType::Int64, &[size.len()], &size),
            ];
            let outputs = vec![(dtype, output_shape)];
            let attrs = vec![("align_corners", Attribute::Int(align_corners))];
            let cuda = run_cuda(&ep, "AffineGrid", "", 20, &inputs, &outputs, &attrs);
            let cpu = run_cpu("AffineGrid", "", 20, &inputs, &outputs, &attrs);
            assert_float_outputs("AffineGrid", dtype, &cuda, &cpu);
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn batch_normalization_matches_cpu_across_float_storage_types() {
    let ep = require_cuda();
    let x = [-3.0, -1.0, 2.0, 4.0, 0.5, 1.5, -2.5, 8.0];
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        let inputs = vec![
            float_input(dtype, &[1, 2, 2, 2], &x),
            float_input(dtype, &[2], &[1.5, 0.25]),
            float_input(dtype, &[2], &[-0.5, 2.0]),
            float_input(dtype, &[2], &[0.25, -1.0]),
            float_input(dtype, &[2], &[2.0, 4.0]),
        ];
        let outputs = vec![(dtype, vec![1, 2, 2, 2])];
        let attrs = vec![("epsilon", Attribute::Float(1e-3))];
        let cuda = run_cuda(&ep, "BatchNormalization", "", 15, &inputs, &outputs, &attrs);
        let cpu = run_cpu("BatchNormalization", "", 15, &inputs, &outputs, &attrs);
        assert_float_outputs("BatchNormalization", dtype, &cuda, &cpu);
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn compress_matches_cpu_for_negative_axis_and_flattened_selection() {
    let ep = require_cuda();
    let axis_inputs = vec![
        input(DataType::Int32, &[2, 4], &[1i32, 2, 3, 4, 5, 6, 7, 8]),
        input(DataType::Bool, &[3], &[false, true, true]),
    ];
    let axis_outputs = vec![(DataType::Int32, vec![2, 2])];
    let attrs = vec![("axis", Attribute::Int(-1))];
    assert_eq!(
        run_cuda(&ep, "Compress", "", 11, &axis_inputs, &axis_outputs, &attrs),
        run_cpu("Compress", "", 11, &axis_inputs, &axis_outputs, &attrs)
    );

    let flat_inputs = vec![
        float_input(
            DataType::BFloat16,
            &[2, 3],
            &[1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
        ),
        input(
            DataType::Bool,
            &[6],
            &[true, false, false, true, false, true],
        ),
    ];
    let flat_outputs = vec![(DataType::BFloat16, vec![3])];
    assert_eq!(
        run_cuda(&ep, "Compress", "", 11, &flat_inputs, &flat_outputs, &[]),
        run_cpu("Compress", "", 11, &flat_inputs, &flat_outputs, &[])
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn dynamic_quantize_linear_matches_cpu_for_ranges_and_constants() {
    let ep = require_cuda();
    for values in [
        vec![-5.0f32, -1.25, 0.0, 0.5, 4.0, 12.0],
        vec![3.25f32; 7],
        Vec::new(),
    ] {
        let inputs = vec![input(DataType::Float32, &[values.len()], &values)];
        let outputs = vec![
            (DataType::Uint8, vec![values.len()]),
            (DataType::Float32, vec![]),
            (DataType::Uint8, vec![]),
        ];
        let cuda = run_cuda(&ep, "DynamicQuantizeLinear", "", 11, &inputs, &outputs, &[]);
        let cpu = run_cpu("DynamicQuantizeLinear", "", 11, &inputs, &outputs, &[]);
        assert_eq!(cuda[0], cpu[0], "DynamicQuantizeLinear quantized data");
        assert_close(
            "DynamicQuantizeLinear scale",
            DataType::Float32,
            &decode_floats(&cuda[1], DataType::Float32),
            &decode_floats(&cpu[1], DataType::Float32),
            1e-7,
        );
        assert_eq!(cuda[2], cpu[2], "DynamicQuantizeLinear zero point");
    }
}

fn global_pool_case(op: &str, opset: u64, attributes: &[(&str, Attribute)]) {
    let ep = require_cuda();
    let values = [
        -3.0f32, 1.0, 2.0, 8.0, -4.0, 0.5, 3.0, 7.0, 6.0, -2.0, 1.5, 0.25,
    ];
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        let inputs = vec![float_input(dtype, &[1, 2, 2, 3], &values)];
        let outputs = vec![(dtype, vec![1, 2, 1, 1])];
        let cuda = run_cuda(&ep, op, "", opset, &inputs, &outputs, attributes);
        let cpu = run_cpu(op, "", opset, &inputs, &outputs, attributes);
        assert_float_outputs(op, dtype, &cuda, &cpu);
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn global_average_pool_matches_cpu() {
    global_pool_case("GlobalAveragePool", 1, &[]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn global_max_pool_matches_cpu() {
    global_pool_case("GlobalMaxPool", 1, &[]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn global_lp_pool_matches_cpu_for_nondefault_p() {
    global_pool_case("GlobalLpPool", 2, &[("p", Attribute::Int(3))]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn lp_normalization_dtype_admission_changes_at_opset22_gpu() {
    let ep = require_cuda();
    for (dtype, opset, expected) in [
        (DataType::Float16, 21, true),
        (DataType::Float32, 21, true),
        (DataType::BFloat16, 21, false),
        (DataType::BFloat16, 22, true),
    ] {
        let inputs = vec![float_input(dtype, &[2, 3], &[3.0, 4.0, 0.0, 0.0, 0.0, 5.0])];
        let outputs = vec![(dtype, vec![2, 3])];
        let (graph, node_id) = build_graph("LpNormalization", "", opset, &inputs, &outputs, &[]);
        let model = Model::new(&graph);
        let claim = ep.supports_op(
            model.graph.node(node_id),
            opset,
            &[static_shape([2, 3])],
            &[dtype],
            &[],
        );
        assert_eq!(
            claim.is_supported(),
            expected,
            "{dtype:?} at opset {opset}: {:?}",
            claim.reason()
        );
        if !expected {
            let reason = claim.reason().unwrap_or_default();
            assert!(reason.contains("BFloat16"), "{reason}");
            assert!(reason.contains("opset 22"), "{reason}");
        }
    }

    let inputs = vec![float_input(
        DataType::BFloat16,
        &[2, 3],
        &[3.0, 4.0, 0.0, 0.0, 0.0, 5.0],
    )];
    let outputs = vec![(DataType::BFloat16, vec![2, 3])];
    let attrs = vec![("fused_reduce_chain", Attribute::Int(1))];
    let (graph, node_id) = build_graph("LpNormalization", "", 21, &inputs, &outputs, &attrs);
    let model = Model::new(&graph);
    let claim = ep.supports_op(
        model.graph.node(node_id),
        21,
        &[static_shape([2, 3])],
        &[DataType::BFloat16],
        &[],
    );
    assert!(
        claim.is_supported(),
        "optimizer-created BF16 LpNormalization must remain claimable: {:?}",
        claim.reason()
    );
    let cuda = run_cuda(
        &ep,
        "LpNormalization",
        "",
        21,
        &inputs,
        &outputs,
        &[
            ("axis", Attribute::Int(-1)),
            ("p", Attribute::Int(2)),
            ("fused_reduce_chain", Attribute::Int(1)),
        ],
    );
    assert_close(
        "optimizer-created LpNormalization independent golden",
        DataType::BFloat16,
        &decode_floats(&cuda[0], DataType::BFloat16),
        &[0.6, 0.8, 0.0, 0.0, 0.0, 1.0],
        tolerance(DataType::BFloat16),
    );
}

fn lp_normalization_reference(values: &[f32], shape: [usize; 3], axis: usize, p: i64) -> Vec<f32> {
    let outer = shape[..axis].iter().product::<usize>();
    let axis_len = shape[axis];
    let inner = shape[axis + 1..].iter().product::<usize>();
    let mut output = vec![0.0; values.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut norm = 0.0;
            for axis_index in 0..axis_len {
                let index = (outer_index * axis_len + axis_index) * inner + inner_index;
                let value = values[index].abs();
                norm += if p == 1 { value } else { value * value };
            }
            if p == 2 {
                norm = norm.sqrt();
            }
            for axis_index in 0..axis_len {
                let index = (outer_index * axis_len + axis_index) * inner + inner_index;
                output[index] = values[index] / norm;
            }
        }
    }
    output
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn lp_normalization_legacy_dtypes_match_independent_golden_gpu() {
    let ep = require_cuda();
    let values = [
        1.0f32, -2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 9.0, -10.0, 11.0, 12.0,
    ];
    for dtype in [DataType::Float16, DataType::Float32] {
        for (axis, normalized_axis, p) in [(-1i64, 2usize, 1i64), (1, 1, 2)] {
            let inputs = vec![float_input(dtype, &[2, 3, 2], &values)];
            let outputs = vec![(dtype, vec![2, 3, 2])];
            let attrs = vec![("axis", Attribute::Int(axis)), ("p", Attribute::Int(p))];
            let cuda = run_cuda(&ep, "LpNormalization", "", 1, &inputs, &outputs, &attrs);
            let expected = lp_normalization_reference(&values, [2, 3, 2], normalized_axis, p);
            assert_close(
                "LpNormalization legacy independent golden",
                dtype,
                &decode_floats(&cuda[0], dtype),
                &expected,
                tolerance(dtype),
            );
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn lp_normalization_opset22_bf16_matches_independent_golden_gpu() {
    let ep = require_cuda();
    let values = [
        1.0f32, -2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0, 9.0, -10.0, 11.0, 12.0,
    ];
    for (axis, normalized_axis, p) in [(-1i64, 2usize, 1i64), (1, 1, 2)] {
        let inputs = vec![float_input(DataType::BFloat16, &[2, 3, 2], &values)];
        let outputs = vec![(DataType::BFloat16, vec![2, 3, 2])];
        let attrs = vec![("axis", Attribute::Int(axis)), ("p", Attribute::Int(p))];
        let cuda = run_cuda(&ep, "LpNormalization", "", 22, &inputs, &outputs, &attrs);
        let expected = lp_normalization_reference(&values, [2, 3, 2], normalized_axis, p);
        assert_close(
            "LpNormalization independent golden",
            DataType::BFloat16,
            &decode_floats(&cuda[0], DataType::BFloat16),
            &expected,
            tolerance(DataType::BFloat16),
        );
    }
}
