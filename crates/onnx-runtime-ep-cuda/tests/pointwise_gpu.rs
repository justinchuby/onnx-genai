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
//! GPU-vs-CPU checks for floating pointwise dtype and broadcasting coverage.

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    CaptureSupport, DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

fn encode(values: &[f32], dtype: DataType) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| match dtype {
            DataType::Float16 => f16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
            DataType::BFloat16 => bf16::from_f32(value).to_bits().to_ne_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect()
}

fn decode(bytes: &[u8], dtype: DataType) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_ne_bytes([chunk[0], chunk[1]]);
            match dtype {
                DataType::Float16 => f16::from_bits(bits).to_f32(),
                DataType::BFloat16 => bf16::from_bits(bits).to_f32(),
                _ => unreachable!(),
            }
        })
        .collect()
}

fn quantize(values: &[f32], dtype: DataType) -> Vec<f32> {
    decode(&encode(values, dtype), dtype)
}

fn run_binary(
    ep: &CudaExecutionProvider,
    op: &str,
    dtype: DataType,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<f32> {
    let rt = ep.runtime();
    let a_bytes = encode(a, dtype);
    let b_bytes = encode(b, dtype);
    let a_buf = ep.allocate(a_bytes.len(), 256).unwrap();
    let b_buf = ep.allocate(b_bytes.len(), 256).unwrap();
    let mut y_buf = ep
        .allocate(out_shape.iter().product::<usize>() * 2, 256)
        .unwrap();
    unsafe {
        rt.htod(&a_bytes, cuptr(a_buf.as_ptr())).unwrap();
        rt.htod(&b_bytes, cuptr(b_buf.as_ptr())).unwrap();
    }

    let a_strides = compute_contiguous_strides(a_shape);
    let b_strides = compute_contiguous_strides(b_shape);
    let y_strides = compute_contiguous_strides(out_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(a_buf.as_ptr()),
            dtype,
            a_shape,
            &a_strides,
            ep.device_id(),
        ),
        TensorView::new(
            DevicePtr(b_buf.as_ptr()),
            dtype,
            b_shape,
            &b_strides,
            ep.device_id(),
        ),
    ];
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        dtype,
        out_shape,
        &y_strides,
        ep.device_id(),
    );
    let kernel = ep
        .get_kernel(&Node::new(NodeId(0), op, vec![], vec![]), &[], 17)
        .unwrap();
    kernel.execute(&inputs, &mut [output]).unwrap();

    let mut out = vec![0u8; out_shape.iter().product::<usize>() * 2];
    unsafe { rt.dtoh(&mut out, cuptr(y_buf.as_ptr())).unwrap() };
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    decode(&out, dtype)
}

fn run_unary(ep: &CudaExecutionProvider, op: &str, dtype: DataType, x: &[f32]) -> Vec<f32> {
    let rt = ep.runtime();
    let bytes = encode(x, dtype);
    let x_buf = ep.allocate(bytes.len(), 256).unwrap();
    let mut y_buf = ep.allocate(bytes.len(), 256).unwrap();
    unsafe { rt.htod(&bytes, cuptr(x_buf.as_ptr())).unwrap() };
    let shape = [x.len()];
    let strides = compute_contiguous_strides(&shape);
    let input = TensorView::new(
        DevicePtr(x_buf.as_ptr()),
        dtype,
        &shape,
        &strides,
        ep.device_id(),
    );
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        dtype,
        &shape,
        &strides,
        ep.device_id(),
    );
    let kernel = ep
        .get_kernel(&Node::new(NodeId(0), op, vec![], vec![]), &[], 17)
        .unwrap();
    kernel.execute(&[input], &mut [output]).unwrap();
    let mut out = vec![0u8; bytes.len()];
    unsafe { rt.dtoh(&mut out, cuptr(y_buf.as_ptr())).unwrap() };
    ep.deallocate(x_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    decode(&out, dtype)
}

/// Run the fused `rsqrt` path: a standard-domain `Reciprocal` node carrying the
/// `_cuda_rsqrt` marker (as `CudaRsqrtFusion` produces), which the CUDA
/// unary-math factory dispatches to the single-kernel `rsqrt_{dtype}`.
fn run_rsqrt(ep: &CudaExecutionProvider, dtype: DataType, x: &[f32]) -> Vec<f32> {
    let rt = ep.runtime();
    let bytes = encode(x, dtype);
    let x_buf = ep.allocate(bytes.len(), 256).unwrap();
    let mut y_buf = ep.allocate(bytes.len(), 256).unwrap();
    unsafe { rt.htod(&bytes, cuptr(x_buf.as_ptr())).unwrap() };
    let shape = [x.len()];
    let strides = compute_contiguous_strides(&shape);
    let input = TensorView::new(
        DevicePtr(x_buf.as_ptr()),
        dtype,
        &shape,
        &strides,
        ep.device_id(),
    );
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        dtype,
        &shape,
        &strides,
        ep.device_id(),
    );
    let mut node = Node::new(NodeId(0), "Reciprocal", vec![], vec![]);
    node.attributes
        .insert("_cuda_rsqrt".into(), Attribute::Int(1));
    let kernel = ep.get_kernel(&node, &[], 17).unwrap();
    kernel.execute(&[input], &mut [output]).unwrap();
    let mut out = vec![0u8; bytes.len()];
    unsafe { rt.dtoh(&mut out, cuptr(y_buf.as_ptr())).unwrap() };
    ep.deallocate(x_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    decode(&out, dtype)
}

fn run_predicate(
    ep: &CudaExecutionProvider,
    op: &str,
    dtype: DataType,
    a_bytes: &[u8],
    a_shape: &[usize],
    b_bytes: &[u8],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<u8> {
    let rt = ep.runtime();
    let a_buf = ep.allocate(a_bytes.len(), 256).unwrap();
    let b_buf = ep.allocate(b_bytes.len(), 256).unwrap();
    let mut y_buf = ep.allocate(out_shape.iter().product(), 256).unwrap();
    unsafe {
        rt.htod(a_bytes, cuptr(a_buf.as_ptr())).unwrap();
        rt.htod(b_bytes, cuptr(b_buf.as_ptr())).unwrap();
    }
    let a_strides = compute_contiguous_strides(a_shape);
    let b_strides = compute_contiguous_strides(b_shape);
    let y_strides = compute_contiguous_strides(out_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(a_buf.as_ptr()),
            dtype,
            a_shape,
            &a_strides,
            ep.device_id(),
        ),
        TensorView::new(
            DevicePtr(b_buf.as_ptr()),
            dtype,
            b_shape,
            &b_strides,
            ep.device_id(),
        ),
    ];
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Bool,
        out_shape,
        &y_strides,
        ep.device_id(),
    );
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let a_value = graph.create_named_value("a", dtype, static_shape(a_shape.iter().copied()));
    let b_value = graph.create_named_value("b", dtype, static_shape(b_shape.iter().copied()));
    let y_value =
        graph.create_named_value("y", DataType::Bool, static_shape(out_shape.iter().copied()));
    graph.add_input(a_value);
    graph.add_input(b_value);
    let node_id = graph.insert_node(Node::new(
        NodeId(0),
        op,
        vec![Some(a_value), Some(b_value)],
        vec![y_value],
    ));
    graph.add_output(y_value);
    let model = Model::new(&graph);
    let kernel = ep.get_kernel(model.graph.node(node_id), &[], 17).unwrap();
    kernel.execute(&inputs, &mut [output]).unwrap();
    let mut out = vec![0; out_shape.iter().product()];
    unsafe { rt.dtoh(&mut out, cuptr(y_buf.as_ptr())).unwrap() };
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    out
}

fn bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: `u8` has alignment 1 and the returned slice covers exactly the
    // original initialized values for the duration of `values`.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn cpu_predicate<T: Copy>(
    a: &[T],
    a_shape: &[usize],
    b: &[T],
    b_shape: &[usize],
    out_shape: &[usize],
    op: impl Fn(T, T) -> bool,
) -> Vec<u8> {
    let strides = |shape: &[usize]| {
        let contiguous = compute_contiguous_strides(shape);
        let leading = out_shape.len() - shape.len();
        (0..out_shape.len())
            .map(|axis| {
                if axis < leading || shape[axis - leading] == 1 {
                    0
                } else {
                    contiguous[axis - leading] as usize
                }
            })
            .collect::<Vec<_>>()
    };
    let a_strides = strides(a_shape);
    let b_strides = strides(b_shape);
    (0..out_shape.iter().product())
        .map(|mut linear| {
            let (mut ai, mut bi) = (0, 0);
            for axis in (0..out_shape.len()).rev() {
                let coord = linear % out_shape[axis];
                linear /= out_shape[axis];
                ai += coord * a_strides[axis];
                bi += coord * b_strides[axis];
            }
            u8::from(op(a[ai], b[bi]))
        })
        .collect()
}

fn cpu_broadcast(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
    op: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let strides = |shape: &[usize]| {
        let contiguous = compute_contiguous_strides(shape);
        let leading = out_shape.len() - shape.len();
        (0..out_shape.len())
            .map(|axis| {
                if axis < leading || shape[axis - leading] == 1 {
                    0
                } else {
                    contiguous[axis - leading] as usize
                }
            })
            .collect::<Vec<_>>()
    };
    let a_strides = strides(a_shape);
    let b_strides = strides(b_shape);
    let n = out_shape.iter().product();
    (0..n)
        .map(|index| {
            let mut linear = index;
            let mut ai = 0;
            let mut bi = 0;
            for axis in (0..out_shape.len()).rev() {
                let coord = linear % out_shape[axis];
                linear /= out_shape[axis];
                ai += coord * a_strides[axis];
                bi += coord * b_strides[axis];
            }
            op(a[ai], b[bi])
        })
        .collect()
}

fn assert_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "index {index}: got {got}, expected {expected}, tolerance {tolerance}"
        );
    }
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn f16_bf16_arithmetic_matches_cpu_compute_domain() {
    let ep = require_cuda();
    let a = [-3.0, -1.5, 0.5, 2.0, 5.0, 8.0];
    let b = [0.5, 2.0, -4.0, 0.25, 2.0, -2.0];
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let aq = quantize(&a, dtype);
        let bq = quantize(&b, dtype);
        let tolerance = if dtype == DataType::Float16 {
            0.002
        } else {
            0.04
        };
        for (op, f) in [
            ("Add", (|x, y| x + y) as fn(f32, f32) -> f32),
            ("Sub", (|x, y| x - y) as fn(f32, f32) -> f32),
            ("Mul", (|x, y| x * y) as fn(f32, f32) -> f32),
            ("Div", (|x, y| x / y) as fn(f32, f32) -> f32),
        ] {
            let expected = quantize(
                &aq.iter()
                    .zip(&bq)
                    .map(|(&x, &y)| f(x, y))
                    .collect::<Vec<_>>(),
                dtype,
            );
            let got = run_binary(&ep, op, dtype, &a, &[6], &b, &[6], &[6]);
            assert_close(&got, &expected, tolerance);
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn f16_bf16_numpy_broadcast_matches_cpu_reference() {
    let ep = require_cuda();
    let a = (0..12).map(|i| i as f32 * 0.25 - 1.0).collect::<Vec<_>>();
    let b = (0..15).map(|i| i as f32 * 0.1 + 0.5).collect::<Vec<_>>();
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let aq = quantize(&a, dtype);
        let bq = quantize(&b, dtype);
        let expected = quantize(
            &cpu_broadcast(&aq, &[4, 1, 3], &bq, &[1, 5, 3], &[4, 5, 3], |x, y| x + y),
            dtype,
        );
        let got = run_binary(
            &ep,
            "Add",
            dtype,
            &a,
            &[4, 1, 3],
            &b,
            &[1, 5, 3],
            &[4, 5, 3],
        );
        assert_close(
            &got,
            &expected,
            if dtype == DataType::Float16 {
                0.002
            } else {
                0.04
            },
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn half_unary_and_activation_families_match_cpu_reference() {
    let ep = require_cuda();
    let x = [-3.0, -1.0, -0.0, 0.5, 2.0];
    for dtype in [DataType::Float16, DataType::BFloat16] {
        let xq = quantize(&x, dtype);
        let tolerance = if dtype == DataType::Float16 {
            0.003
        } else {
            0.04
        };
        for (op, f) in [
            (
                "Relu",
                (|v: f32| if v.is_nan() { v } else { v.max(0.0) }) as fn(f32) -> f32,
            ),
            ("Exp", (|v: f32| v.exp()) as fn(f32) -> f32),
            (
                "LeakyRelu",
                (|v: f32| if v >= 0.0 { v } else { 0.01 * v }) as fn(f32) -> f32,
            ),
        ] {
            let expected = quantize(&xq.iter().copied().map(f).collect::<Vec<_>>(), dtype);
            assert_close(&run_unary(&ep, op, dtype, &x), &expected, tolerance);
        }
    }
}

/// The fused `rsqrt` kernel (from `CudaRsqrtFusion`) must be **byte-identical**
/// to the unfused `Reciprocal(Sqrt(x))` two-kernel path — that is the parity
/// contract that keeps native decode greedy tokens unchanged. It reproduces the
/// intermediate narrow-dtype rounding (`Sqrt` rounds `sqrtf(x)` to the storage
/// dtype, then `Reciprocal` reads that back), so an exact-equality check (not a
/// tolerance) is required.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn fused_rsqrt_is_byte_identical_to_sqrt_then_reciprocal() {
    let ep = require_cuda();
    // Positive inputs across a wide magnitude range (rsqrt domain); avoids NaN
    // so exact equality is well-defined.
    let x = [
        1.0e-4, 0.01, 0.25, 0.5, 1.0, 2.0, 3.0, 7.5, 42.0, 123.5, 1000.0, 65504.0,
    ];
    for dtype in [DataType::Float16, DataType::BFloat16] {
        // Two-kernel reference: Sqrt then Reciprocal, each rounding to `dtype`.
        let sqrted = run_unary(&ep, "Sqrt", dtype, &x);
        let reference = run_unary(&ep, "Reciprocal", dtype, &sqrted);
        let fused = run_rsqrt(&ep, dtype, &x);
        assert_eq!(
            fused.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "fused rsqrt must be bit-identical to Reciprocal(Sqrt(x)) for {dtype:?}"
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn logical_family_numpy_broadcast_matches_cpu_reference() {
    let ep = require_cuda();
    let a = [0_u8, 1];
    let b = [0_u8, 1, 1];
    let expected = [
        ("And", vec![0, 0, 0, 0, 1, 1]),
        ("Or", vec![0, 1, 1, 1, 1, 1]),
        ("Xor", vec![0, 1, 1, 1, 0, 0]),
    ];
    for (op, expected) in expected {
        assert_eq!(
            run_predicate(&ep, op, DataType::Bool, &a, &[2, 1], &b, &[1, 3], &[2, 3]),
            expected,
            "{op}"
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn comparison_family_numpy_broadcast_matches_cpu_reference() {
    let ep = require_cuda();
    let a = [1.0_f32, 3.0];
    let b = [2.0_f32, 3.0, 4.0];
    let a_bytes =
        unsafe { std::slice::from_raw_parts(a.as_ptr().cast::<u8>(), std::mem::size_of_val(&a)) };
    let b_bytes =
        unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u8>(), std::mem::size_of_val(&b)) };
    let expected = [
        ("Equal", vec![0, 0, 0, 0, 1, 0]),
        ("Greater", vec![0, 0, 0, 1, 0, 0]),
        ("Less", vec![1, 1, 1, 0, 0, 1]),
        ("GreaterOrEqual", vec![0, 0, 0, 1, 1, 0]),
        ("LessOrEqual", vec![1, 1, 1, 0, 1, 1]),
    ];
    for (op, expected) in expected {
        assert_eq!(
            run_predicate(
                &ep,
                op,
                DataType::Float32,
                a_bytes,
                &[2, 1],
                b_bytes,
                &[1, 3],
                &[2, 3],
            ),
            expected,
            "{op}"
        );
    }
}

fn assert_integer_comparisons<T>(ep: &CudaExecutionProvider, dtype: DataType, a: &[T], b: &[T])
where
    T: Copy + PartialEq + PartialOrd,
{
    let a_shape = [2, 1];
    let b_shape = [1, 3];
    let out_shape = [2, 3];
    for (op, predicate) in [
        ("Equal", (|x: T, y: T| x == y) as fn(T, T) -> bool),
        ("Greater", (|x: T, y: T| x > y) as fn(T, T) -> bool),
        ("Less", (|x: T, y: T| x < y) as fn(T, T) -> bool),
        ("GreaterOrEqual", (|x: T, y: T| x >= y) as fn(T, T) -> bool),
        ("LessOrEqual", (|x: T, y: T| x <= y) as fn(T, T) -> bool),
    ] {
        assert_eq!(
            run_predicate(
                ep,
                op,
                dtype,
                bytes(a),
                &a_shape,
                bytes(b),
                &b_shape,
                &out_shape
            ),
            cpu_predicate(a, &a_shape, b, &b_shape, &out_shape, predicate),
            "{dtype:?} {op}"
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn integer_comparisons_broadcast_match_cpu_oracle() {
    let ep = require_cuda();
    assert_integer_comparisons(&ep, DataType::Int64, &[1_i64, 3], &[2_i64, 3, 4]);
    assert_integer_comparisons(&ep, DataType::Int32, &[1_i32, 3], &[2_i32, 3, 4]);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn integer_comparisons_cover_glm_like_masks_and_are_deterministic() {
    let ep = require_cuda();
    let position_ids = [-2_i64, -1, 0, 1, 2, 3];
    let zero = [0_i64];
    let expected = cpu_predicate(&position_ids, &[2, 3], &zero, &[], &[2, 3], |a, b| a >= b);
    for _ in 0..5 {
        assert_eq!(
            run_predicate(
                &ep,
                "GreaterOrEqual",
                DataType::Int64,
                bytes(&position_ids),
                &[2, 3],
                bytes(&zero),
                &[],
                &[2, 3],
            ),
            expected
        );
    }

    let token_ids = [1_i32, 0, 42, 0];
    let pad_id = [0_i32];
    assert_eq!(
        run_predicate(
            &ep,
            "Equal",
            DataType::Int32,
            bytes(&token_ids),
            &[2, 2],
            bytes(&pad_id),
            &[],
            &[2, 2],
        ),
        vec![0, 1, 0, 1]
    );

    assert_eq!(
        run_predicate(
            &ep,
            "Equal",
            DataType::Bool,
            &[0, 1],
            &[2, 1],
            &[0, 1, 1],
            &[1, 3],
            &[2, 3],
        ),
        vec![1, 0, 0, 0, 1, 1]
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn comparison_claims_integer_dtypes_and_rejects_unsupported_dtype() {
    let ep = require_cuda();
    let node = Node::new(NodeId(0), "Equal", vec![], vec![]);
    let shapes = [static_shape([2]), static_shape([2])];
    for dtype in [DataType::Int64, DataType::Int32] {
        assert!(matches!(
            ep.supports_op(&node, 17, &shapes, &[dtype, dtype], &[]),
            KernelMatch::Supported { .. }
        ));
    }
    assert!(matches!(
        ep.supports_op(
            &node,
            17,
            &shapes,
            &[DataType::Float64, DataType::Float64],
            &[]
        ),
        KernelMatch::Unsupported { ref reason }
            if reason.contains("Equal: operand dtype Float64 not supported on CUDA EP")
    ));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn binary_static_shape_metadata_captures_replays_and_tracks_signatures() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    let decode_shape = [1, 1, 4];
    let bias_shape = [4];
    let decode_strides = compute_contiguous_strides(&decode_shape);
    let bias_strides = compute_contiguous_strides(&bias_shape);
    let a_values = [1.0_f32, 2.0, 3.0, 4.0];
    let b_values = [0.5_f32, -1.0, 2.0, 3.0];
    let expected = [1.5_f32, 1.0, 5.0, 7.0];

    let a_buf = ep.allocate(std::mem::size_of_val(&a_values), 256).unwrap();
    let b_buf = ep.allocate(std::mem::size_of_val(&b_values), 256).unwrap();
    let mut y_buf = ep.allocate(std::mem::size_of_val(&expected), 256).unwrap();
    unsafe {
        runtime
            .htod(bytes(&a_values), cuptr(a_buf.as_ptr()))
            .unwrap();
        runtime
            .htod(bytes(&b_values), cuptr(b_buf.as_ptr()))
            .unwrap();
    }
    let inputs = [
        TensorView::new(
            DevicePtr(a_buf.as_ptr()),
            DataType::Float32,
            &decode_shape,
            &decode_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(b_buf.as_ptr()),
            DataType::Float32,
            &bias_shape,
            &bias_strides,
            device,
        ),
    ];
    let mut kernel = ep
        .get_kernel(&Node::new(NodeId(0), "Add", vec![], vec![]), &[], 17)
        .unwrap();
    kernel.set_capture_seq_independent(true);
    assert!(!kernel.cuda_graph_compatible());

    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Float32,
        &decode_shape,
        &decode_strides,
        device,
    );
    kernel.execute(&inputs, &mut [output]).unwrap();
    assert!(kernel.cuda_graph_compatible());
    let mut eager_bytes = vec![0_u8; std::mem::size_of_val(&expected)];
    unsafe {
        runtime
            .dtoh(&mut eager_bytes, cuptr(y_buf.as_ptr()))
            .unwrap()
    };
    let eager = eager_bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(eager, expected);
    let after_warmup = runtime.allocation_counts();

    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Float32,
        &decode_shape,
        &decode_strides,
        device,
    );
    kernel.execute(&inputs, &mut [output]).unwrap();
    assert_eq!(runtime.allocation_counts(), after_warmup);

    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Float32,
        &decode_shape,
        &decode_strides,
        device,
    );
    kernel.execute(&inputs, &mut [output]).unwrap();
    runtime.end_graph_capture().unwrap();
    runtime.replay_graph().unwrap();
    let mut replay_bytes = vec![0_u8; std::mem::size_of_val(&expected)];
    unsafe {
        runtime
            .dtoh(&mut replay_bytes, cuptr(y_buf.as_ptr()))
            .unwrap()
    };
    let replay = replay_bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(replay, eager);
    assert_eq!(runtime.allocation_counts(), after_warmup);
    assert!(runtime.reset_graph().unwrap());

    let i64_values = [1_i64, 2, 3, 4];
    let i64_bias = [1_i64; 4];
    let i64_a = ep
        .allocate(std::mem::size_of_val(&i64_values), 256)
        .unwrap();
    let i64_b = ep.allocate(std::mem::size_of_val(&i64_bias), 256).unwrap();
    let mut i64_y = ep
        .allocate(std::mem::size_of_val(&i64_values), 256)
        .unwrap();
    unsafe {
        runtime
            .htod(bytes(&i64_values), cuptr(i64_a.as_ptr()))
            .unwrap();
        runtime
            .htod(bytes(&i64_bias), cuptr(i64_b.as_ptr()))
            .unwrap();
    }
    let i64_inputs = [
        TensorView::new(
            DevicePtr(i64_a.as_ptr()),
            DataType::Int64,
            &decode_shape,
            &decode_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(i64_b.as_ptr()),
            DataType::Int64,
            &bias_shape,
            &bias_strides,
            device,
        ),
    ];
    assert!(kernel.cuda_graph_compatible());
    let i64_output = TensorMut::new(
        DevicePtrMut(i64_y.as_mut_ptr()),
        DataType::Int64,
        &decode_shape,
        &decode_strides,
        device,
    );
    kernel.execute(&i64_inputs, &mut [i64_output]).unwrap();
    assert!(kernel.cuda_graph_compatible());
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    let i64_output = TensorMut::new(
        DevicePtrMut(i64_y.as_mut_ptr()),
        DataType::Int64,
        &decode_shape,
        &decode_strides,
        device,
    );
    kernel.execute(&i64_inputs, &mut [i64_output]).unwrap();
    runtime.end_graph_capture().unwrap();
    runtime.replay_graph().unwrap();
    let mut i64_replay = vec![0_u8; std::mem::size_of_val(&i64_values)];
    unsafe {
        runtime
            .dtoh(&mut i64_replay, cuptr(i64_y.as_ptr()))
            .unwrap()
    };
    assert_eq!(
        i64_replay
            .chunks_exact(8)
            .map(|value| i64::from_ne_bytes(value.try_into().unwrap()))
            .collect::<Vec<_>>(),
        vec![2, 3, 4, 5]
    );
    assert!(runtime.reset_graph().unwrap());

    let prefill_shape = [1, 2, 4];
    let prefill_strides = compute_contiguous_strides(&prefill_shape);
    let prefill_values = [1.0_f32; 8];
    let prefill_a = ep
        .allocate(std::mem::size_of_val(&prefill_values), 256)
        .unwrap();
    let mut prefill_y = ep
        .allocate(std::mem::size_of_val(&prefill_values), 256)
        .unwrap();
    unsafe {
        runtime
            .htod(bytes(&prefill_values), cuptr(prefill_a.as_ptr()))
            .unwrap()
    };
    let prefill_inputs = [
        TensorView::new(
            DevicePtr(prefill_a.as_ptr()),
            DataType::Float32,
            &prefill_shape,
            &prefill_strides,
            device,
        ),
        inputs[1],
    ];
    let prefill_output = TensorMut::new(
        DevicePtrMut(prefill_y.as_mut_ptr()),
        DataType::Float32,
        &prefill_shape,
        &prefill_strides,
        device,
    );
    let before_shape_change = runtime.allocation_counts();
    kernel
        .execute(&prefill_inputs, &mut [prefill_output])
        .unwrap();
    assert!(kernel.cuda_graph_compatible());
    let after_shape_change = runtime.allocation_counts();
    assert_eq!(
        after_shape_change.allocations,
        before_shape_change.allocations + 1
    );
    assert_eq!(after_shape_change.frees, before_shape_change.frees);
    assert!(
        ep.release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "replaced broadcast metadata must eventually be released: {:?}",
        ep.deferred_release_stats()
    );

    drop(kernel);
    assert!(
        ep.release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "live broadcast metadata must be released with the kernel: {:?}",
        ep.deferred_release_stats()
    );
    ep.deallocate(prefill_a).unwrap();
    ep.deallocate(prefill_y).unwrap();
    ep.deallocate(i64_a).unwrap();
    ep.deallocate(i64_b).unwrap();
    ep.deallocate(i64_y).unwrap();
    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
}

/// C1: a fully-static (sequence-independent) shape whose leading dims do **not**
/// collapse to 1 — e.g. a head-major `[heads=32, feat=128]` decode tensor — is
/// rejected by the runtime-extent heuristic (`is_fixed_decode_shape`) but must be
/// admitted to capture once the executor reports the node's IR output shape is
/// fully static via `set_capture_seq_independent(true)`. The same shape with the
/// flag left `false` (as the executor does for a symbolic/growing seq axis) must
/// stay eager. This is the metadata-driven gate that distinguishes
/// `[heads=32, feat=128]` (capturable) from `[tokens=32, feat=128]` (unsafe),
/// which runtime extents alone cannot — with no per-model hardcoding.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn seq_independent_static_shape_is_capture_eligible_via_metadata_flag() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();

    // Fully static, but leading dims are 32 (do not collapse to 1): the
    // runtime-extent heuristic alone rejects this shape for capture.
    let shape = [32_usize, 128];
    let bias_shape = [128_usize];
    let strides = compute_contiguous_strides(&shape);
    let bias_strides = compute_contiguous_strides(&bias_shape);
    let n: usize = shape.iter().product();
    let a_values = vec![1.0_f32; n];
    let b_values = vec![0.5_f32; bias_shape[0]];

    let a_buf = ep.allocate(n * 4, 256).unwrap();
    let b_buf = ep.allocate(bias_shape[0] * 4, 256).unwrap();
    let mut y_buf = ep.allocate(n * 4, 256).unwrap();
    unsafe {
        runtime
            .htod(bytes(&a_values), cuptr(a_buf.as_ptr()))
            .unwrap();
        runtime
            .htod(bytes(&b_values), cuptr(b_buf.as_ptr()))
            .unwrap();
    }
    let inputs = [
        TensorView::new(
            DevicePtr(a_buf.as_ptr()),
            DataType::Float32,
            &shape,
            &strides,
            device,
        ),
        TensorView::new(
            DevicePtr(b_buf.as_ptr()),
            DataType::Float32,
            &bias_shape,
            &bias_strides,
            device,
        ),
    ];

    // Flag left false (executor default for a symbolic/growing seq axis): the
    // static-but-non-collapsing shape must NOT become capture-eligible.
    let eager_kernel = ep
        .get_kernel(&Node::new(NodeId(0), "Add", vec![], vec![]), &[], 17)
        .unwrap();
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        device,
    );
    eager_kernel.execute(&inputs, &mut [output]).unwrap();
    assert!(
        matches!(
            eager_kernel.capture_support(),
            CaptureSupport::Unsupported { .. }
        ),
        "static [32,128] must stay eager without the seq-independent metadata flag"
    );

    // Flag true (executor derived all output dims are Dim::Static): the same shape
    // is now capture-eligible and a captured replay reproduces the eager result.
    let mut capture_kernel = ep
        .get_kernel(&Node::new(NodeId(0), "Add", vec![], vec![]), &[], 17)
        .unwrap();
    capture_kernel.set_capture_seq_independent(true);
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        device,
    );
    capture_kernel.execute(&inputs, &mut [output]).unwrap();
    assert_eq!(
        capture_kernel.capture_support(),
        CaptureSupport::Supported,
        "static [32,128] must be capture-eligible once flagged seq-independent"
    );

    let mut eager_bytes = vec![0_u8; n * 4];
    unsafe {
        runtime
            .dtoh(&mut eager_bytes, cuptr(y_buf.as_ptr()))
            .unwrap();
    }
    let eager = eager_bytes
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();

    runtime
        .begin_graph_capture(&[capture_kernel.as_ref()])
        .unwrap();
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        device,
    );
    capture_kernel.execute(&inputs, &mut [output]).unwrap();
    runtime.end_graph_capture().unwrap();
    runtime.replay_graph().unwrap();
    let mut replay_bytes = vec![0_u8; n * 4];
    unsafe {
        runtime
            .dtoh(&mut replay_bytes, cuptr(y_buf.as_ptr()))
            .unwrap();
    }
    let replay = replay_bytes
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(replay, eager, "captured replay must match the eager result");
    assert!(runtime.reset_graph().unwrap());

    ep.deallocate(a_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
}
