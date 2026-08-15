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
//! GPU parity tests for the issue #67 CUDA op-coverage batch: the
//! trigonometric/hyperbolic unary-math family (`Tan`, `Sinh`, `Cosh`, `Asin`,
//! `Acos`, `Atan`, `Asinh`, `Acosh`, `Atanh`), the metadata/movement ops
//! `Identity`, `Flatten`, `Size`, and the triangular mask `Trilu`.
//!
//! Every check runs the real CUDA kernel and compares its result against the CPU
//! execution provider running the same ONNX node (the reference path), asserting
//! concrete values. The suite skips cleanly when no CUDA runtime is present so a
//! host without a GPU still passes.

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceId, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

struct Tensor {
    dtype: DataType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn raw<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: primitive test values are plain old data with no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    }
}

fn input<T: Copy>(dtype: DataType, shape: &[usize], values: &[T]) -> Tensor {
    Tensor {
        dtype,
        shape: shape.to_vec(),
        bytes: raw(values),
    }
}

fn encode_floats(values: &[f32], dtype: DataType) -> Vec<u8> {
    match dtype {
        DataType::Float32 => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        DataType::Float16 => values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
        DataType::BFloat16 => values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
            .collect(),
        other => panic!("unsupported float dtype {other:?}"),
    }
}

fn decode_floats(bytes: &[u8], dtype: DataType) -> Vec<f32> {
    match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_ne_bytes(c.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unsupported float dtype {other:?}"),
    }
}

fn build_graph(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    let input_values = inputs
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            let value = graph.create_named_value(
                format!("input_{index}"),
                tensor.dtype,
                static_shape(tensor.shape.iter().copied()),
            );
            graph.add_input(value);
            value
        })
        .collect::<Vec<_>>();
    let output_values = outputs
        .iter()
        .enumerate()
        .map(|(index, (dtype, shape))| {
            graph.create_named_value(
                format!("output_{index}"),
                *dtype,
                static_shape(shape.iter().copied()),
            )
        })
        .collect::<Vec<_>>();
    let mut node = Node::new(
        NodeId(0),
        op,
        input_values.into_iter().map(Some).collect(),
        output_values.clone(),
    );
    for (name, value) in attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let node_id = graph.insert_node(node);
    for output in output_values {
        graph.add_output(output);
    }
    (graph, node_id)
}

fn run_cuda(
    ep: &CudaExecutionProvider,
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let (graph, node_id) = build_graph(op, opset, inputs, outputs, attrs);
    let model = Model::new(&graph);
    let concrete_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape.clone())
        .collect::<Vec<_>>();
    // The claim gate (`supports_op`) must accept exactly the nodes we run so a
    // graph partitioner keeps them on CUDA rather than falling back to the CPU EP.
    let claim_shapes = inputs
        .iter()
        .map(|tensor| static_shape(tensor.shape.iter().copied()))
        .collect::<Vec<_>>();
    let claim_dtypes = inputs.iter().map(|tensor| tensor.dtype).collect::<Vec<_>>();
    let claim = ep.supports_op(
        model.graph.node(node_id),
        opset,
        &claim_shapes,
        &claim_dtypes,
        &[],
    );
    assert!(
        claim.is_supported(),
        "{op} (opset {opset}) must be claimed by the CUDA EP, got: {:?}",
        claim.reason()
    );
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, opset)
        .unwrap();

    let input_buffers = inputs
        .iter()
        .map(|tensor| {
            let buffer = ep.allocate(tensor.bytes.len().max(1), 256).unwrap();
            if !tensor.bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .htod(&tensor.bytes, cuptr(buffer.as_ptr()))
                        .unwrap()
                };
            }
            buffer
        })
        .collect::<Vec<_>>();
    let input_strides = inputs
        .iter()
        .map(|tensor| compute_contiguous_strides(&tensor.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_buffers)
        .zip(&input_strides)
        .map(|((tensor, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                tensor.dtype,
                &tensor.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let mut output_buffers = outputs
        .iter()
        .map(|(dtype, shape)| {
            ep.allocate(dtype.storage_bytes(shape.iter().product()).max(1), 256)
                .unwrap()
        })
        .collect::<Vec<DeviceBuffer>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_views = outputs
        .iter()
        .zip(output_buffers.iter_mut())
        .zip(&output_strides)
        .map(|(((dtype, shape), buffer), strides)| {
            TensorMut::new(
                DevicePtrMut(buffer.as_mut_ptr()),
                *dtype,
                shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    kernel.execute(&input_views, &mut output_views).unwrap();

    let result = outputs
        .iter()
        .zip(&output_buffers)
        .map(|((dtype, shape), buffer)| {
            let mut bytes = vec![0; dtype.storage_bytes(shape.iter().product())];
            if !bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .dtoh(&mut bytes, cuptr(buffer.as_ptr()))
                        .unwrap()
                };
            }
            bytes
        })
        .collect();
    for buffer in input_buffers {
        ep.deallocate(buffer).unwrap();
    }
    for buffer in output_buffers {
        ep.deallocate(buffer).unwrap();
    }
    result
}

fn run_cpu(
    op: &str,
    opset: u64,
    inputs: &[Tensor],
    outputs: &[(DataType, Vec<usize>)],
    attrs: &[(&str, Attribute)],
) -> Vec<Vec<u8>> {
    let ep = CpuExecutionProvider::new();
    let (graph, node_id) = build_graph(op, opset, inputs, outputs, attrs);
    let model = Model::new(&graph);
    let concrete_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape.clone())
        .collect::<Vec<_>>();
    let kernel = ep
        .get_kernel(model.graph.node(node_id), &concrete_shapes, opset)
        .unwrap();
    let input_strides = inputs
        .iter()
        .map(|tensor| compute_contiguous_strides(&tensor.shape))
        .collect::<Vec<_>>();
    let input_views = inputs
        .iter()
        .zip(&input_strides)
        .map(|(tensor, strides)| {
            TensorView::new(
                DevicePtr(tensor.bytes.as_ptr().cast()),
                tensor.dtype,
                &tensor.shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect::<Vec<_>>();
    let output_strides = outputs
        .iter()
        .map(|(_, shape)| compute_contiguous_strides(shape))
        .collect::<Vec<_>>();
    let mut output_bytes = outputs
        .iter()
        .map(|(dtype, shape)| vec![0_u8; dtype.storage_bytes(shape.iter().product())])
        .collect::<Vec<_>>();
    let mut output_views = outputs
        .iter()
        .zip(&output_strides)
        .zip(output_bytes.iter_mut())
        .map(|(((dtype, shape), strides), bytes)| {
            TensorMut::new(
                DevicePtrMut(bytes.as_mut_ptr().cast()),
                *dtype,
                shape,
                strides,
                DeviceId::cpu(),
            )
        })
        .collect::<Vec<_>>();
    kernel.execute(&input_views, &mut output_views).unwrap();
    drop(output_views);
    output_bytes
}

fn require_cuda() -> CudaExecutionProvider {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => panic!(
            "CUDA test requires CUDA device/runtime; CPU-only runs must leave this test ignored: {error}"
        ),
        Err(_) => panic!(
            "CUDA test requires CUDA runtime libraries; CPU-only runs must leave this test ignored"
        ),
    }
}

fn assert_close(op: &str, dtype: DataType, got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len(), "{op} {dtype:?}: length mismatch");
    for (index, (&got, &want)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "{op} {dtype:?} index {index}: got {got}, expected {want}, tolerance {tolerance}"
        );
    }
}

/// Trigonometric / hyperbolic unary math parity across f32/f16/bf16, each op fed
/// domain-safe values, compared element-wise against the CPU EP.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn trig_hyperbolic_unary_matches_cpu() {
    let ep = require_cuda();
    // (op, domain-safe inputs).
    let cases: &[(&str, &[f32])] = &[
        ("Tan", &[-1.2, -0.4, 0.0, 0.5, 1.1]),
        ("Sinh", &[-2.0, -0.7, 0.0, 0.9, 2.5]),
        ("Cosh", &[-2.0, -0.7, 0.0, 0.9, 2.5]),
        ("Asin", &[-0.95, -0.4, 0.0, 0.5, 0.9]),
        ("Acos", &[-0.95, -0.4, 0.0, 0.5, 0.9]),
        ("Atan", &[-5.0, -0.7, 0.0, 1.3, 4.0]),
        ("Asinh", &[-5.0, -0.7, 0.0, 1.3, 4.0]),
        ("Acosh", &[1.0, 1.4, 2.0, 3.5, 6.0]),
        ("Atanh", &[-0.9, -0.4, 0.0, 0.5, 0.85]),
    ];
    for &(op, values) in cases {
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            let shape = vec![values.len()];
            let inputs = [Tensor {
                dtype,
                shape: shape.clone(),
                bytes: encode_floats(values, dtype),
            }];
            let outputs = [(dtype, shape.clone())];
            let cuda = run_cuda(&ep, op, 17, &inputs, &outputs, &[]);
            let cpu = run_cpu(op, 17, &inputs, &outputs, &[]);
            let cuda_f = decode_floats(&cuda[0], dtype);
            let cpu_f = decode_floats(&cpu[0], dtype);
            // f32 device intrinsics vs host libm agree to a few ulp; the half
            // formats are dominated by their own rounding step.
            let tolerance = match dtype {
                DataType::Float32 => 1e-4,
                DataType::Float16 => 3e-3,
                DataType::BFloat16 => 3e-2,
                _ => unreachable!(),
            };
            assert_close(op, dtype, &cuda_f, &cpu_f, tolerance);
        }
        eprintln!("{op}: CUDA matches CPU EP across f32/f16/bf16");
    }
}

/// `Identity` and `Flatten` are dtype-agnostic byte copies; assert exact byte
/// equality with the CPU EP for float and integer payloads.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn identity_and_flatten_match_cpu() {
    let ep = require_cuda();
    let f32_values: Vec<f32> = (0..12).map(|v| v as f32 * 0.5 - 3.0).collect();
    let i64_values: Vec<i64> = (0..12).map(|v| (v as i64) * 7 - 5).collect();

    // Identity: same shape in and out.
    for (dtype, bytes) in [
        (DataType::Float32, raw(&f32_values)),
        (DataType::Int64, raw(&i64_values)),
    ] {
        let inputs = [Tensor {
            dtype,
            shape: vec![3, 4],
            bytes,
        }];
        let outputs = [(dtype, vec![3, 4])];
        let cuda = run_cuda(&ep, "Identity", 16, &inputs, &outputs, &[]);
        let cpu = run_cpu("Identity", 16, &inputs, &outputs, &[]);
        assert_eq!(cuda, cpu, "Identity {dtype:?} bytes mismatch");
    }

    // Flatten: axis=2 collapses [2,2,3] -> [4,3] but preserves row-major order.
    let inputs = [input(DataType::Float32, &[2, 2, 3], &f32_values)];
    let outputs = [(DataType::Float32, vec![4, 3])];
    let attrs = [("axis", Attribute::Int(2))];
    let cuda = run_cuda(&ep, "Flatten", 13, &inputs, &outputs, &attrs);
    let cpu = run_cpu("Flatten", 13, &inputs, &outputs, &attrs);
    assert_eq!(cuda, cpu, "Flatten bytes mismatch");
    eprintln!("Identity/Flatten: CUDA matches CPU EP");
}

/// `Size` yields the input's element count as an Int64 scalar.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn size_matches_cpu() {
    let ep = require_cuda();
    let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let inputs = [input(DataType::Float32, &[2, 3, 4], &values)];
    let outputs = [(DataType::Int64, vec![])];
    let cuda = run_cuda(&ep, "Size", 13, &inputs, &outputs, &[]);
    let cpu = run_cpu("Size", 13, &inputs, &outputs, &[]);
    assert_eq!(cuda, cpu, "Size bytes mismatch");
    let scalar = i64::from_ne_bytes(cuda[0].as_slice().try_into().unwrap());
    assert_eq!(scalar, 24, "Size scalar value");
    eprintln!("Size: CUDA matches CPU EP (= {scalar})");
}

/// `Trilu` upper/lower masks with several diagonal offsets and dtypes, compared
/// byte-for-byte against the CPU EP over batched trailing 2-D matrices. `k` is a
/// runtime Int64 scalar so the CUDA path exercises its device-scalar read.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn trilu_matches_cpu() {
    let ep = require_cuda();
    // A batch of two 3x4 matrices with distinct values.
    let f32_values: Vec<f32> = (0..24).map(|v| v as f32 + 1.0).collect();
    let i64_values: Vec<i64> = (0..24).map(|v| (v as i64) + 1).collect();
    let shape = [2usize, 3, 4];

    for (upper, k) in [
        (1_i64, None),
        (0, None),
        (1, Some(1_i64)),
        (1, Some(-1)),
        (0, Some(2)),
        (0, Some(-2)),
    ] {
        for (dtype, bytes) in [
            (DataType::Float32, raw(&f32_values)),
            (DataType::Int64, raw(&i64_values)),
        ] {
            let mut inputs = vec![Tensor {
                dtype,
                shape: shape.to_vec(),
                bytes,
            }];
            if let Some(k) = k {
                inputs.push(input(DataType::Int64, &[], &[k]));
            }
            let outputs = [(dtype, shape.to_vec())];
            let attrs = [("upper", Attribute::Int(upper))];
            let cuda = run_cuda(&ep, "Trilu", 14, &inputs, &outputs, &attrs);
            let cpu = run_cpu("Trilu", 14, &inputs, &outputs, &attrs);
            assert_eq!(
                cuda, cpu,
                "Trilu upper={upper} k={k:?} {dtype:?} bytes mismatch"
            );
        }
    }
    eprintln!("Trilu: CUDA matches CPU EP across upper/lower, k offsets, f32/i64");
}

// ────────────────────────────────────────────────────────────────────────────
// Issue #67 coverage batch 2: extended reductions (`ReduceProd`,
// `ReduceSumSquare`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`),
// extended activations (`Swish`, `ThresholdedRelu`), the variadic elementwise
// ops (`Sum`, `Mean`), and `Mod`. Every case compares the real CUDA kernel
// against the CPU EP running the identical node.
// ────────────────────────────────────────────────────────────────────────────

/// Output shape for a reduction over `axes` (negative axes allowed) of an input
/// with shape `in_shape`, honouring `keepdims`.
fn reduce_out_shape(in_shape: &[usize], axes: &[i64], keepdims: bool) -> Vec<usize> {
    let rank = in_shape.len() as i64;
    let reduced: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (a + rank) as usize
            } else {
                a as usize
            }
        })
        .collect();
    let mut out = Vec::new();
    for (axis, &dim) in in_shape.iter().enumerate() {
        if reduced.contains(&axis) {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }
    out
}

/// Extended f32 reductions across several axes / keepdims combinations, compared
/// against the CPU EP element-wise. Inputs are strictly positive so `ReduceLogSum`
/// (`log(sum(x))`) stays well defined.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn extended_reductions_match_cpu() {
    let ep = require_cuda();
    let in_shape = vec![2usize, 3, 4];
    let count: usize = in_shape.iter().product();
    // Positive, mildly varied magnitudes keep every reduction well conditioned.
    let values: Vec<f32> = (0..count)
        .map(|v| 0.5 + (v % 7) as f32 * 0.3 + (v % 3) as f32 * 0.1)
        .collect();

    let axes_cases: &[(&[i64], bool)] = &[
        (&[1], true),
        (&[1], false),
        (&[0, 2], false),
        (&[-1], false),
        (&[0, 1, 2], true),
    ];
    for op in [
        "ReduceProd",
        "ReduceSumSquare",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
    ] {
        for &(axes, keepdims) in axes_cases {
            let out_shape = reduce_out_shape(&in_shape, axes, keepdims);
            let inputs = [input(DataType::Float32, &in_shape, &values)];
            let outputs = [(DataType::Float32, out_shape.clone())];
            let attrs = [
                ("axes", Attribute::Ints(axes.to_vec())),
                ("keepdims", Attribute::Int(keepdims as i64)),
            ];
            let cuda = run_cuda(&ep, op, 13, &inputs, &outputs, &attrs);
            let cpu = run_cpu(op, 13, &inputs, &outputs, &attrs);
            let cuda_f = decode_floats(&cuda[0], DataType::Float32);
            let cpu_f = decode_floats(&cpu[0], DataType::Float32);
            // Reductions accumulate in f32; the surviving difference is a few ulp
            // scaled by the reduced magnitude (products/log-sum-exp are largest).
            assert_close(op, DataType::Float32, &cuda_f, &cpu_f, 2e-2);
        }
        eprintln!("{op}: CUDA matches CPU EP across axes/keepdims");
    }
}

/// `ReduceLogSumExp` with the opset-18 axes-*input* form (rather than an
/// attribute) exercises the kernel's device-side axes read.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn reduce_log_sum_exp_axes_input_matches_cpu() {
    let ep = require_cuda();
    let in_shape = vec![2usize, 3, 4];
    let count: usize = in_shape.iter().product();
    let values: Vec<f32> = (0..count).map(|v| (v as f32 * 0.17) - 2.0).collect();
    let axes: Vec<i64> = vec![2];
    let out_shape = reduce_out_shape(&in_shape, &axes, false);
    let inputs = [
        input(DataType::Float32, &in_shape, &values),
        input(DataType::Int64, &[axes.len()], &axes),
    ];
    let outputs = [(DataType::Float32, out_shape)];
    let attrs = [("keepdims", Attribute::Int(0))];
    let cuda = run_cuda(&ep, "ReduceLogSumExp", 18, &inputs, &outputs, &attrs);
    let cpu = run_cpu("ReduceLogSumExp", 18, &inputs, &outputs, &attrs);
    let cuda_f = decode_floats(&cuda[0], DataType::Float32);
    let cpu_f = decode_floats(&cpu[0], DataType::Float32);
    assert_close("ReduceLogSumExp", DataType::Float32, &cuda_f, &cpu_f, 2e-3);
    eprintln!("ReduceLogSumExp (axes-input, opset 18): CUDA matches CPU EP");
}

/// Regression guard for the numerical-stability defect fixed in PR #266: a naive
/// `log(sum(exp(x)))` overflows to `+inf` for large-magnitude inputs, while the
/// CPU EP (and now the CUDA kernel) stabilizes as `m + log(sum(exp(x - m)))`.
/// These inputs (≈90 and a wide negative-to-positive spread) return `+inf` under
/// the old naive kernel, so this test fails on it and passes on the stable one.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn reduce_log_sum_exp_large_values_match_cpu() {
    let ep = require_cuda();
    // Row 0: the classic overflow case `[90, 91, 92, 93]` (naive exp -> +inf).
    // Row 1: a wide spread mixing large negatives and positives; exp(120) also
    // overflows f32 (~3.4e38) without max-subtraction.
    let in_shape = vec![2usize, 4];
    let values: Vec<f32> = vec![90.0, 91.0, 92.0, 93.0, -100.0, 5.0, 120.0, -3.0];
    let axes: Vec<i64> = vec![1];
    let out_shape = reduce_out_shape(&in_shape, &axes, false);
    let inputs = [input(DataType::Float32, &in_shape, &values)];
    let outputs = [(DataType::Float32, out_shape)];
    let attrs = [
        ("axes", Attribute::Ints(axes.clone())),
        ("keepdims", Attribute::Int(0)),
    ];
    let cuda = run_cuda(&ep, "ReduceLogSumExp", 13, &inputs, &outputs, &attrs);
    let cpu = run_cpu("ReduceLogSumExp", 13, &inputs, &outputs, &attrs);
    let cuda_f = decode_floats(&cuda[0], DataType::Float32);
    let cpu_f = decode_floats(&cpu[0], DataType::Float32);
    // The CPU reference must stay finite — proving the stabilization is the point
    // of this test (a naive kernel would emit `+inf` here).
    assert!(
        cpu_f.iter().all(|v| v.is_finite()),
        "CPU reference should be finite, got {cpu_f:?}"
    );
    assert!(
        cuda_f.iter().all(|v| v.is_finite()),
        "CUDA output overflowed (naive log(sum(exp))?): {cuda_f:?}"
    );
    assert_close("ReduceLogSumExp", DataType::Float32, &cuda_f, &cpu_f, 2e-3);
    eprintln!("ReduceLogSumExp (large values): CUDA matches CPU EP, no overflow");
}

/// `Swish` and `ThresholdedRelu` across f32/f16/bf16 with default and explicit
/// `alpha`, compared against the CPU EP.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn extended_activations_match_cpu() {
    let ep = require_cuda();
    let values: &[f32] = &[-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0];
    type ActivationCase = (&'static str, u64, Vec<(&'static str, Attribute)>);
    let cases: Vec<ActivationCase> = vec![
        ("Swish", 24, vec![]),
        ("Swish", 24, vec![("alpha", Attribute::Float(0.5))]),
        ("ThresholdedRelu", 10, vec![]),
        (
            "ThresholdedRelu",
            10,
            vec![("alpha", Attribute::Float(1.0))],
        ),
    ];
    for (op, opset, attrs) in &cases {
        let (op, opset) = (*op, *opset);
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            let shape = vec![values.len()];
            let inputs = [Tensor {
                dtype,
                shape: shape.clone(),
                bytes: encode_floats(values, dtype),
            }];
            let outputs = [(dtype, shape.clone())];
            let cuda = run_cuda(&ep, op, opset, &inputs, &outputs, attrs);
            let cpu = run_cpu(op, opset, &inputs, &outputs, attrs);
            let cuda_f = decode_floats(&cuda[0], dtype);
            let cpu_f = decode_floats(&cpu[0], dtype);
            let tolerance = match dtype {
                DataType::Float32 => 1e-4,
                DataType::Float16 => 3e-3,
                DataType::BFloat16 => 3e-2,
                _ => unreachable!(),
            };
            assert_close(op, dtype, &cuda_f, &cpu_f, tolerance);
        }
        eprintln!("{op} {attrs:?}: CUDA matches CPU EP across f32/f16/bf16");
    }
}

/// `Sum` and `Mean` over a variadic, broadcasting input list across f32/f16/bf16,
/// compared against the CPU EP.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn variadic_sum_mean_match_cpu() {
    let ep = require_cuda();
    // Three operands with distinct broadcastable shapes -> output [2,3].
    let a: &[f32] = &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let b: &[f32] = &[10.0, 20.0, 30.0];
    let c: &[f32] = &[100.0, 200.0];
    for op in ["Sum", "Mean"] {
        for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            let inputs = [
                Tensor {
                    dtype,
                    shape: vec![2, 3],
                    bytes: encode_floats(a, dtype),
                },
                Tensor {
                    dtype,
                    shape: vec![3],
                    bytes: encode_floats(b, dtype),
                },
                Tensor {
                    dtype,
                    shape: vec![2, 1],
                    bytes: encode_floats(c, dtype),
                },
            ];
            let outputs = [(dtype, vec![2usize, 3])];
            let cuda = run_cuda(&ep, op, 13, &inputs, &outputs, &[]);
            let cpu = run_cpu(op, 13, &inputs, &outputs, &[]);
            let cuda_f = decode_floats(&cuda[0], dtype);
            let cpu_f = decode_floats(&cpu[0], dtype);
            let tolerance = match dtype {
                DataType::Float32 => 1e-4,
                DataType::Float16 => 5e-1,
                DataType::BFloat16 => 4.0,
                _ => unreachable!(),
            };
            assert_close(op, dtype, &cuda_f, &cpu_f, tolerance);
        }
        eprintln!("{op}: CUDA matches CPU EP (variadic broadcasting)");
    }
}

fn decode_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn decode_i64(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// `Mod` parity against the CPU EP: f32 (`fmod=1`), plus i32/i64 in both the
/// C-truncated (`fmod=1`) and Python floor (`fmod=0`) modes, including negative
/// dividends and divisors and a divide-by-zero (defined to yield 0).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn mod_matches_cpu() {
    let ep = require_cuda();

    // Float Mod requires fmod=1 per ONNX.
    let fa: &[f32] = &[5.3, -5.3, 7.5, -7.5, 2.0, -2.0];
    let fb: &[f32] = &[2.0, 2.0, -2.5, -2.5, 3.0, 3.0];
    let inputs = [
        input(DataType::Float32, &[6], fa),
        input(DataType::Float32, &[6], fb),
    ];
    let outputs = [(DataType::Float32, vec![6usize])];
    let attrs = [("fmod", Attribute::Int(1))];
    let cuda = run_cuda(&ep, "Mod", 13, &inputs, &outputs, &attrs);
    let cpu = run_cpu("Mod", 13, &inputs, &outputs, &attrs);
    assert_close(
        "Mod",
        DataType::Float32,
        &decode_floats(&cuda[0], DataType::Float32),
        &decode_floats(&cpu[0], DataType::Float32),
        1e-5,
    );

    // Integer Mod: both modes, negative operands, and a zero divisor.
    let ia: &[i64] = &[7, -7, 7, -7, 5, -5, 9];
    let ib: &[i64] = &[3, 3, -3, -3, -2, 2, 0];
    for fmod in [0_i64, 1] {
        // i64
        let inputs = [
            input(DataType::Int64, &[7], ia),
            input(DataType::Int64, &[7], ib),
        ];
        let outputs = [(DataType::Int64, vec![7usize])];
        let attrs = [("fmod", Attribute::Int(fmod))];
        let cuda = run_cuda(&ep, "Mod", 13, &inputs, &outputs, &attrs);
        let cpu = run_cpu("Mod", 13, &inputs, &outputs, &attrs);
        assert_eq!(
            decode_i64(&cuda[0]),
            decode_i64(&cpu[0]),
            "Mod i64 fmod={fmod} mismatch"
        );

        // i32
        let ia32: Vec<i32> = ia.iter().map(|&v| v as i32).collect();
        let ib32: Vec<i32> = ib.iter().map(|&v| v as i32).collect();
        let inputs = [
            input(DataType::Int32, &[7], &ia32),
            input(DataType::Int32, &[7], &ib32),
        ];
        let outputs = [(DataType::Int32, vec![7usize])];
        let cuda = run_cuda(&ep, "Mod", 13, &inputs, &outputs, &attrs);
        let cpu = run_cpu("Mod", 13, &inputs, &outputs, &attrs);
        assert_eq!(
            decode_i32(&cuda[0]),
            decode_i32(&cpu[0]),
            "Mod i32 fmod={fmod} mismatch"
        );
    }
    eprintln!("Mod: CUDA matches CPU EP (f32 fmod=1; i32/i64 truncated + floor)");
}

// ────────────────────────────────────────────────────────────────────────────
// Issue #67 coverage batch 3: unary float predicates (`IsInf` with the
// detect_positive/detect_negative attributes, `IsNaN`) and `PRelu` (parametric
// ReLU with a NumPy-broadcastable slope). Every case runs the real CUDA kernel
// and compares it byte-for-byte (bool) or element-wise (PRelu) against the CPU
// EP running the identical node.
// ────────────────────────────────────────────────────────────────────────────

/// `IsInf` across f32/f16/bf16 with every detect_positive/detect_negative
/// combination (plus the attribute defaults), fed +inf/-inf/NaN and finite
/// values. Bool output is compared byte-for-byte with the CPU EP.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn is_inf_matches_cpu() {
    let ep = require_cuda();
    // +inf, -inf, NaN, finite positive, finite negative, zero, -0.0.
    let values: &[f32] = &[
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        3.5,
        -2.25,
        0.0,
        -0.0,
    ];
    // `None` exercises the attribute defaults (both detections on).
    let attr_cases: &[Option<(i64, i64)>] =
        &[None, Some((1, 1)), Some((1, 0)), Some((0, 1)), Some((0, 0))];
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for &flags in attr_cases {
            let shape = vec![values.len()];
            let inputs = [Tensor {
                dtype,
                shape: shape.clone(),
                bytes: encode_floats(values, dtype),
            }];
            let outputs = [(DataType::Bool, shape.clone())];
            let attrs: Vec<(&str, Attribute)> = match flags {
                None => vec![],
                Some((dp, dn)) => vec![
                    ("detect_positive", Attribute::Int(dp)),
                    ("detect_negative", Attribute::Int(dn)),
                ],
            };
            let cuda = run_cuda(&ep, "IsInf", 10, &inputs, &outputs, &attrs);
            let cpu = run_cpu("IsInf", 10, &inputs, &outputs, &attrs);
            assert_eq!(
                cuda, cpu,
                "IsInf {dtype:?} flags {flags:?} bool bytes mismatch"
            );
        }
    }
    eprintln!("IsInf: CUDA matches CPU EP across f32/f16/bf16 and every detect flag combo");
}

/// `IsNaN` across f32/f16/bf16, fed NaN/inf/finite values. Bool output compared
/// byte-for-byte with the CPU EP. Also covers the empty-tensor edge case.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn is_nan_matches_cpu() {
    let ep = require_cuda();
    let values: &[f32] = &[
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -4.5,
        1.0,
        f32::NAN,
    ];
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        let shape = vec![values.len()];
        let inputs = [Tensor {
            dtype,
            shape: shape.clone(),
            bytes: encode_floats(values, dtype),
        }];
        let outputs = [(DataType::Bool, shape.clone())];
        let cuda = run_cuda(&ep, "IsNaN", 13, &inputs, &outputs, &[]);
        let cpu = run_cpu("IsNaN", 13, &inputs, &outputs, &[]);
        assert_eq!(cuda, cpu, "IsNaN {dtype:?} bool bytes mismatch");

        // Empty input: zero elements, empty bool output on both EPs.
        let empty = [Tensor {
            dtype,
            shape: vec![0],
            bytes: vec![],
        }];
        let empty_out = [(DataType::Bool, vec![0usize])];
        let cuda_empty = run_cuda(&ep, "IsNaN", 13, &empty, &empty_out, &[]);
        let cpu_empty = run_cpu("IsNaN", 13, &empty, &empty_out, &[]);
        assert_eq!(cuda_empty, cpu_empty, "IsNaN {dtype:?} empty mismatch");
    }
    eprintln!("IsNaN: CUDA matches CPU EP across f32/f16/bf16 (incl. empty)");
}

/// `PRelu` across f32/f16/bf16 with several slope broadcast shapes (scalar,
/// per-channel, full, rank-0 scalar), compared element-wise against the CPU EP.
/// Negative-input lanes exercise the `x * slope` branch; the slope itself is
/// negative to make the two branches unambiguous.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn prelu_matches_cpu() {
    let ep = require_cuda();

    // X shape [2,3,4] with a mix of negative, zero, and positive lanes.
    let x_shape = vec![2usize, 3, 4];
    let count: usize = x_shape.iter().product();
    let x_values: Vec<f32> = (0..count)
        .map(|v| (v as f32) - (count as f32) / 2.0)
        .collect();

    // (slope shape, slope values) — each unidirectionally broadcastable to X.
    let per_channel: Vec<f32> = vec![-0.25, 0.10, -0.50]; // one per channel (dim 1)
    let full: Vec<f32> = (0..count).map(|v| -0.05 * (v as f32 + 1.0)).collect();
    let slope_cases: Vec<(Vec<usize>, Vec<f32>)> = vec![
        (vec![1], vec![-0.30]),            // scalar broadcast
        (vec![3, 1], per_channel.clone()), // per-channel over dims 1..
        (x_shape.clone(), full.clone()),   // exact-shape slope
    ];

    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for (slope_shape, slope_values) in &slope_cases {
            let inputs = [
                Tensor {
                    dtype,
                    shape: x_shape.clone(),
                    bytes: encode_floats(&x_values, dtype),
                },
                Tensor {
                    dtype,
                    shape: slope_shape.clone(),
                    bytes: encode_floats(slope_values, dtype),
                },
            ];
            let outputs = [(dtype, x_shape.clone())];
            let cuda = run_cuda(&ep, "PRelu", 16, &inputs, &outputs, &[]);
            let cpu = run_cpu("PRelu", 16, &inputs, &outputs, &[]);
            let cuda_f = decode_floats(&cuda[0], dtype);
            let cpu_f = decode_floats(&cpu[0], dtype);
            // Both EPs compute in f32 and round once on store, so results are
            // bit-identical up to the storage rounding step; a tight tolerance
            // guards against any accidental precision divergence.
            let tolerance = match dtype {
                DataType::Float32 => 0.0,
                DataType::Float16 => 3e-3,
                DataType::BFloat16 => 3e-2,
                _ => unreachable!(),
            };
            assert_close("PRelu", dtype, &cuda_f, &cpu_f, tolerance);
        }

        // Rank-0 scalar X with a rank-0 scalar slope (negative lane).
        let inputs = [
            Tensor {
                dtype,
                shape: vec![],
                bytes: encode_floats(&[-1.5], dtype),
            },
            Tensor {
                dtype,
                shape: vec![],
                bytes: encode_floats(&[-0.20], dtype),
            },
        ];
        let outputs = [(dtype, vec![])];
        let cuda = run_cuda(&ep, "PRelu", 16, &inputs, &outputs, &[]);
        let cpu = run_cpu("PRelu", 16, &inputs, &outputs, &[]);
        let cuda_f = decode_floats(&cuda[0], dtype);
        let cpu_f = decode_floats(&cpu[0], dtype);
        let tolerance = match dtype {
            DataType::Float32 => 0.0,
            DataType::Float16 => 3e-3,
            DataType::BFloat16 => 3e-2,
            _ => unreachable!(),
        };
        assert_close("PRelu(scalar)", dtype, &cuda_f, &cpu_f, tolerance);
    }
    eprintln!("PRelu: CUDA matches CPU EP across f32/f16/bf16 and scalar/per-channel/full slopes");
}
