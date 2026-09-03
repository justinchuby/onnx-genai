#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

mod common;

use std::ffi::c_void;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use common::{Tensor, decode_floats, float_input, input, run_cuda};
use cudarc::driver::result::event;
use cudarc::driver::sys::CUevent_flags;
use half::{bf16, f16};
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{
    CudaExecutionProvider, EinsumRouteOverride, build_cuda_registry_descriptors,
    cuda_supported_dtypes_for_op, einsum_execution_stats, execute_einsum_with_route,
    execute_einsum_with_route_and_memory_ceiling, reset_einsum_execution_stats,
};
use onnx_runtime_ir::{
    Attribute, DataType, TensorLayout, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

fn suite_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn require_cuda() -> CudaExecutionProvider {
    CudaExecutionProvider::new_default().expect("CUDA runtime must be available")
}

fn quantize(values: &[f32], dtype: DataType) -> Vec<f64> {
    match dtype {
        DataType::Float32 => values.iter().map(|&value| value as f64).collect(),
        DataType::Float16 => values
            .iter()
            .map(|&value| f16::from_f32(value).to_f32() as f64)
            .collect(),
        DataType::BFloat16 => values
            .iter()
            .map(|&value| bf16::from_f32(value).to_f32() as f64)
            .collect(),
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn f64_gemm_reference(
    a: &[f64],
    b: &[f64],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    transpose_a: bool,
    transpose_b: bool,
    a_batch_stride: usize,
    b_batch_stride: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; batch * m * n];
    for batch_index in 0..batch {
        let a_base = batch_index * a_batch_stride;
        let b_base = batch_index * b_batch_stride;
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0;
                for reduction in 0..k {
                    let a_index = if transpose_a {
                        a_base + reduction * m + row
                    } else {
                        a_base + row * k + reduction
                    };
                    let b_index = if transpose_b {
                        b_base + column * k + reduction
                    } else {
                        b_base + reduction * n + column
                    };
                    sum += a[a_index] * b[b_index];
                }
                output[batch_index * m * n + row * n + column] = sum;
            }
        }
    }
    output
}

fn tolerance(dtype: DataType) -> (f64, f64) {
    match dtype {
        DataType::Float32 => (3e-5, 3e-5),
        DataType::Float16 => (2e-2, 3e-3),
        DataType::BFloat16 => (8e-2, 1e-2),
        _ => unreachable!(),
    }
}

fn assert_close(got: &[f32], expected: &[f64], dtype: DataType, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: output length");
    let (atol, rtol) = tolerance(dtype);
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let error = (got as f64 - expected).abs();
        let allowed = atol + rtol * expected.abs();
        assert!(
            error <= allowed,
            "{label}: element {index}: got {got}, expected {expected}, \
             error {error} > {allowed} for {dtype:?}"
        );
    }
}

fn values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let integer = (index.wrapping_mul(37 + salt) + 11 * salt + 3) % 97;
            (integer as f32 - 48.0) / 17.0
        })
        .collect()
}

fn run_gemm_case(
    ep: &CudaExecutionProvider,
    equation: &str,
    dtype: DataType,
    a_shape: &[usize],
    b_shape: &[usize],
    output_shape: &[usize],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    transpose_a: bool,
    transpose_b: bool,
    a_batch_stride: usize,
    b_batch_stride: usize,
) {
    let a_values = values(a_shape.iter().product(), 1);
    let b_values = values(b_shape.iter().product(), 5);
    let outputs = run_cuda(
        ep,
        "Einsum",
        "",
        12,
        &[
            float_input(dtype, a_shape, &a_values),
            float_input(dtype, b_shape, &b_values),
        ],
        &[(dtype, output_shape.to_vec())],
        &[("equation", Attribute::String(equation.as_bytes().to_vec()))],
    );
    let got = decode_floats(&outputs[0], dtype);
    let expected = f64_gemm_reference(
        &quantize(&a_values, dtype),
        &quantize(&b_values, dtype),
        batch,
        m,
        k,
        n,
        transpose_a,
        transpose_b,
        a_batch_stride,
        b_batch_stride,
    );
    assert_close(&got, &expected, dtype, equation);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn registry_reachability_and_claim_declines_are_intentional() {
    let _lock = suite_lock();
    let ep = require_cuda();
    assert!(
        onnx_runtime_ep_cuda::CUDA_COVERED_OPS.contains(&"Einsum"),
        "coverage census must expose Einsum"
    );
    let descriptors = build_cuda_registry_descriptors(ep.runtime().clone());
    let entries = descriptors
        .iter()
        .filter(|descriptor| descriptor.domain.is_empty() && descriptor.op_type == "Einsum")
        .collect::<Vec<_>>();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].since_version, 12);
    assert_eq!(
        cuda_supported_dtypes_for_op("Einsum", ""),
        &[
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::BFloat16,
            DataType::Uint8,
            DataType::Uint16,
            DataType::Uint32,
            DataType::Uint64,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
        ]
    );

    let valid_inputs = [
        float_input(DataType::Float32, &[2, 3], &[0.0; 6]),
        float_input(DataType::Float32, &[3, 4], &[0.0; 12]),
    ];
    let (graph, node) = common::build_graph(
        "Einsum",
        "",
        12,
        &valid_inputs,
        &[(DataType::Float32, vec![2, 4])],
        &[("equation", Attribute::String(b"ik,kj->ij".to_vec()))],
    );
    let model = Model::new(&graph);
    let shapes = [static_shape([2, 3]), static_shape([3, 4])];
    let dtypes = [DataType::Float32, DataType::Float32];
    let supported = ep.supports_op(model.graph.node(node), 12, &shapes, &dtypes, &[]);
    assert!(supported.is_supported(), "{:?}", supported.reason());
    assert!(
        !ep.supports_op(model.graph.node(node), 11, &shapes, &dtypes, &[])
            .is_supported(),
        "Einsum is introduced in opset 12"
    );
    let cpu = CpuExecutionProvider::new();
    assert!(
        cpu.supports_op(model.graph.node(node), 12, &shapes, &dtypes, &[])
            .is_supported(),
        "CPU and CUDA must both consume the shared canonical Einsum plan after CPU support landed"
    );

    reset_einsum_execution_stats();
    for (equation, shapes) in [
        (
            "ik,kj->ji",
            vec![static_shape([2, 3]), static_shape([3, 4])],
        ),
        (
            "...mk,...kn->...mn",
            vec![static_shape([2, 1, 3, 4]), static_shape([2, 5, 4, 6])],
        ),
        ("ij->i", vec![static_shape([2, 3])]),
    ] {
        let inputs = shapes
            .iter()
            .map(|shape| {
                let concrete = onnx_runtime_ir::as_static_shape(shape).unwrap();
                float_input(
                    DataType::Float32,
                    &concrete,
                    &vec![0.0; concrete.iter().product()],
                )
            })
            .collect::<Vec<_>>();
        let (graph, node) = common::build_graph(
            "Einsum",
            "",
            12,
            &inputs,
            &[(DataType::Float32, vec![1])],
            &[("equation", Attribute::String(equation.as_bytes().to_vec()))],
        );
        let model = Model::new(&graph);
        let dtypes = vec![DataType::Float32; shapes.len()];
        let claim = ep.supports_op(model.graph.node(node), 12, &shapes, &dtypes, &[]);
        assert!(claim.is_supported(), "{equation}: {:?}", claim.reason());
    }
    let fallback_stats = einsum_execution_stats();
    assert_eq!(fallback_stats.claim_fallbacks, 0);
    assert_eq!(fallback_stats.last_fallback_reason, None);

    let strided = [
        TensorLayout::strided(vec![1, 2]),
        TensorLayout::contiguous(),
    ];
    let claim = ep.supports_op(model.graph.node(node), 12, &shapes, &dtypes, &strided);
    assert!(claim.is_supported(), "{:?}", claim.reason());
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn descriptor_transpose_contractions_match_f64_reference() {
    let _lock = suite_lock();
    let ep = require_cuda();
    for (equation, a_shape, b_shape, transpose_a, transpose_b) in [
        ("ik,kj->ij", &[3, 5][..], &[5, 4][..], false, false),
        ("ki,kj->ij", &[5, 3][..], &[5, 4][..], true, false),
        ("ik,jk->ij", &[3, 5][..], &[4, 5][..], false, true),
        ("ki,jk->ij", &[5, 3][..], &[4, 5][..], true, true),
    ] {
        reset_einsum_execution_stats();
        run_gemm_case(
            &ep,
            equation,
            DataType::Float32,
            a_shape,
            b_shape,
            &[3, 4],
            1,
            3,
            5,
            4,
            transpose_a,
            transpose_b,
            0,
            0,
        );
        let stats = einsum_execution_stats();
        assert_eq!(stats.gemm_launches, 1, "{equation}");
        if transpose_a || transpose_b {
            assert_eq!(stats.descriptor_transpose_gemm_launches, 1, "{equation}");
            assert_eq!(stats.canonical_gemm_launches, 0, "{equation}");
        } else {
            assert_eq!(stats.descriptor_transpose_gemm_launches, 0, "{equation}");
            assert_eq!(stats.canonical_gemm_launches, 1, "{equation}");
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn dtypes_bmm_stride_zero_and_flattened_groups_match_f64_reference() {
    let _lock = suite_lock();
    let ep = require_cuda();
    for dtype in [DataType::Float32, DataType::Float16] {
        run_gemm_case(
            &ep,
            "bij,bjk->bik",
            dtype,
            &[3, 2, 5],
            &[3, 5, 4],
            &[3, 2, 4],
            3,
            2,
            5,
            4,
            false,
            false,
            10,
            20,
        );
        run_gemm_case(
            &ep,
            "mk,...kn->...mn",
            dtype,
            &[2, 5],
            &[3, 5, 4],
            &[3, 2, 4],
            3,
            2,
            5,
            4,
            false,
            false,
            0,
            20,
        );
        run_gemm_case(
            &ep,
            "abxy,xycd->abcd",
            dtype,
            &[2, 3, 2, 2],
            &[2, 2, 2, 3],
            &[2, 3, 2, 3],
            1,
            6,
            4,
            6,
            false,
            false,
            0,
            0,
        );
    }

    let bf16_inputs = [
        float_input(DataType::BFloat16, &[2, 3], &[0.0; 6]),
        float_input(DataType::BFloat16, &[3, 4], &[0.0; 12]),
    ];
    let (graph, node) = common::build_graph(
        "Einsum",
        "",
        12,
        &bf16_inputs,
        &[(DataType::BFloat16, vec![2, 4])],
        &[("equation", Attribute::String(b"ik,kj->ij".to_vec()))],
    );
    let model = Model::new(&graph);
    let claim = ep.supports_op(
        model.graph.node(node),
        12,
        &[static_shape([2, 3]), static_shape([3, 4])],
        &[DataType::BFloat16, DataType::BFloat16],
        &[],
    );
    assert!(
        claim
            .reason()
            .is_some_and(|reason| reason.contains("not admitted by Einsum-12")),
        "bf16 decline must name the schema blocker: {:?}",
        claim.reason()
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn dot_and_zero_dimensions_have_exact_metadata_and_values() {
    let _lock = suite_lock();
    let ep = require_cuda();
    run_gemm_case(
        &ep,
        "i,i->",
        DataType::Float32,
        &[7],
        &[7],
        &[],
        1,
        1,
        7,
        1,
        false,
        false,
        0,
        0,
    );
    run_gemm_case(
        &ep,
        "i,ij->j",
        DataType::Float32,
        &[7],
        &[7, 5],
        &[5],
        1,
        1,
        7,
        5,
        false,
        false,
        0,
        0,
    );
    run_gemm_case(
        &ep,
        "ij,j->i",
        DataType::Float32,
        &[5, 7],
        &[7],
        &[5],
        1,
        5,
        7,
        1,
        false,
        false,
        0,
        0,
    );

    reset_einsum_execution_stats();
    let zero_k = run_cuda(
        &ep,
        "Einsum",
        "",
        12,
        &[
            float_input(DataType::Float32, &[2, 0], &[]),
            float_input(DataType::Float32, &[0, 3], &[]),
        ],
        &[(DataType::Float32, vec![2, 3])],
        &[("equation", Attribute::String(b"ik,kj->ij".to_vec()))],
    );
    assert_eq!(decode_floats(&zero_k[0], DataType::Float32), vec![0.0; 6]);
    assert_eq!(einsum_execution_stats().zero_fill_launches, 1);

    reset_einsum_execution_stats();
    let zero_output = run_cuda(
        &ep,
        "Einsum",
        "",
        12,
        &[
            float_input(DataType::Float32, &[0, 4], &[]),
            float_input(DataType::Float32, &[4, 3], &[1.0; 12]),
        ],
        &[(DataType::Float32, vec![0, 3])],
        &[("equation", Attribute::String(b"ik,kj->ij".to_vec()))],
    );
    assert!(zero_output[0].is_empty());
    assert_eq!(einsum_execution_stats().gemm_launches, 0);
    assert_eq!(einsum_execution_stats().zero_fill_launches, 0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn generic_reduction_multilinear_and_integer_routes_execute_natively() {
    let _lock = suite_lock();
    let ep = require_cuda();

    reset_einsum_execution_stats();
    let reduced = run_cuda(
        &ep,
        "Einsum",
        "",
        12,
        &[float_input(
            DataType::Float32,
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )],
        &[(DataType::Float32, vec![2])],
        &[("equation", Attribute::String(b"ij->i".to_vec()))],
    );
    assert_eq!(
        decode_floats(&reduced[0], DataType::Float32),
        vec![6.0, 15.0]
    );
    assert_eq!(
        einsum_execution_stats().last_route,
        Some(onnx_runtime_ep_cuda::CudaEinsumRoute::GenericNative)
    );

    reset_einsum_execution_stats();
    let multilinear = run_cuda(
        &ep,
        "Einsum",
        "",
        12,
        &[
            float_input(DataType::Float32, &[2], &[2.0, 3.0]),
            float_input(DataType::Float32, &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            float_input(DataType::Float32, &[2], &[5.0, 7.0]),
        ],
        &[(DataType::Float32, vec![])],
        &[("equation", Attribute::String(b"i,ij,j->".to_vec()))],
    );
    assert_eq!(
        decode_floats(&multilinear[0], DataType::Float32),
        vec![167.0]
    );
    assert_eq!(
        einsum_execution_stats().last_route,
        Some(onnx_runtime_ep_cuda::CudaEinsumRoute::OptimizedDp)
    );
    assert!(einsum_execution_stats().optimized_step_launches >= 2);
    assert!(
        einsum_execution_stats().optimized_cublas_launches > 0,
        "the f32 multilinear tree must contain a real cuBLASLt contraction step"
    );

    reset_einsum_execution_stats();
    let integer = run_cuda(
        &ep,
        "Einsum",
        "",
        12,
        &[
            input(DataType::Int8, &[4], &[127i8, -128, -1, 2]),
            input(DataType::Int8, &[4], &[2i8, 2, -1, 100]),
        ],
        &[(DataType::Int8, vec![4])],
        &[("equation", Attribute::String(b"i,i->i".to_vec()))],
    );
    assert_eq!(
        integer[0],
        common::raw(&[-2i8, 0, 1, -56]),
        "integer arithmetic is modular at the declared width"
    );
    assert_eq!(
        einsum_execution_stats().last_route,
        Some(onnx_runtime_ep_cuda::CudaEinsumRoute::OptimizedDp)
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn optimizer_memory_ceiling_selects_generic_native_without_intermediates() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let shapes = [vec![2], vec![2, 2], vec![2]];
    let output_shape: [usize; 0] = [];
    let (kernel, inputs, buffers, mut output) =
        make_direct_kernel(&ep, "i,ij,j->", &shapes, &output_shape);
    let strides = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect::<Vec<_>>();
    let output_strides = compute_contiguous_strides(&output_shape);
    let views = inputs
        .iter()
        .zip(&buffers)
        .zip(&strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    reset_einsum_execution_stats();
    execute_einsum_with_route_and_memory_ceiling(
        kernel.as_ref(),
        &views,
        &mut [TensorMut::new(
            DevicePtrMut(output.as_mut_ptr()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            ep.device_id(),
        )],
        EinsumRouteOverride::Auto,
        0,
    )
    .unwrap();
    let stats = einsum_execution_stats();
    assert_eq!(
        stats.last_route,
        Some(onnx_runtime_ep_cuda::CudaEinsumRoute::GenericNative)
    );
    assert_eq!(stats.workspace_bytes, 0);
    assert_eq!(stats.generic_native_launches, 1);
    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn generic_native_consumes_strided_and_negative_stride_views() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();

    let vector_shape = [3usize];
    let vector_output_shape = [3usize];
    let (reverse, reverse_inputs, reverse_buffers, mut reverse_output) =
        make_direct_kernel(&ep, "i->i", &[vector_shape.to_vec()], &vector_output_shape);
    let reverse_stride = [-1i64];
    let output_strides = compute_contiguous_strides(&vector_output_shape);
    let reverse_view = TensorView::new(
        DevicePtr(reverse_buffers[0].as_ptr()),
        DataType::Float32,
        &vector_shape,
        &reverse_stride,
        ep.device_id(),
    )
    .with_byte_offset(2 * std::mem::size_of::<f32>());
    execute_einsum_with_route(
        reverse.as_ref(),
        std::slice::from_ref(&reverse_view),
        &mut [TensorMut::new(
            DevicePtrMut(reverse_output.as_mut_ptr()),
            DataType::Float32,
            &vector_output_shape,
            &output_strides,
            ep.device_id(),
        )],
        EinsumRouteOverride::GenericNative,
    )
    .unwrap();
    runtime.synchronize().unwrap();
    let mut reverse_bytes = vec![0u8; 3 * 4];
    unsafe {
        runtime
            .dtoh(&mut reverse_bytes, cuptr(reverse_output.as_ptr()))
            .unwrap()
    };
    let source = decode_floats(&reverse_inputs[0].bytes, DataType::Float32);
    assert_eq!(
        decode_floats(&reverse_bytes, DataType::Float32),
        [source[2], source[1], source[0]]
    );

    let diagonal_shape = [2usize, 2];
    let diagonal_output_shape = [2usize];
    let diagonal_inputs = [float_input(DataType::Float32, &diagonal_shape, &[0.0; 4])];
    let (graph, node) = common::build_graph(
        "Einsum",
        "",
        12,
        &diagonal_inputs,
        &[(DataType::Float32, diagonal_output_shape.to_vec())],
        &[("equation", Attribute::String(b"ii->i".to_vec()))],
    );
    let model = Model::new(&graph);
    let diagonal = ep
        .get_kernel(model.graph.node(node), &[diagonal_shape.to_vec()], 12)
        .unwrap();
    let diagonal_values = [1.0f32, 2.0, 3.0, 99.0, 4.0];
    let diagonal_bytes = common::raw(&diagonal_values);
    let diagonal_buffer = ep.allocate(diagonal_bytes.len(), 256).unwrap();
    let mut diagonal_output = ep.allocate(2 * 4, 256).unwrap();
    unsafe {
        runtime
            .htod(&diagonal_bytes, cuptr(diagonal_buffer.as_ptr()))
            .unwrap()
    };
    let diagonal_strides = [3i64, 1];
    let diagonal_output_strides = compute_contiguous_strides(&diagonal_output_shape);
    execute_einsum_with_route(
        diagonal.as_ref(),
        &[TensorView::new(
            DevicePtr(diagonal_buffer.as_ptr()),
            DataType::Float32,
            &diagonal_shape,
            &diagonal_strides,
            ep.device_id(),
        )],
        &mut [TensorMut::new(
            DevicePtrMut(diagonal_output.as_mut_ptr()),
            DataType::Float32,
            &diagonal_output_shape,
            &diagonal_output_strides,
            ep.device_id(),
        )],
        EinsumRouteOverride::GenericNative,
    )
    .unwrap();
    runtime.synchronize().unwrap();
    let mut diagonal_result = vec![0u8; 8];
    unsafe {
        runtime
            .dtoh(&mut diagonal_result, cuptr(diagonal_output.as_ptr()))
            .unwrap()
    };
    assert_eq!(
        decode_floats(&diagonal_result, DataType::Float32),
        [1.0, 4.0]
    );

    for buffer in reverse_buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(reverse_output).unwrap();
    ep.deallocate(diagonal_buffer).unwrap();
    ep.deallocate(diagonal_output).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn permutation_and_diagonal_use_zero_copy_views_and_materialize_correctly() {
    let _lock = suite_lock();
    let ep = require_cuda();
    reset_einsum_execution_stats();

    let input = float_input(DataType::Float32, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let (graph, node) = common::build_graph(
        "Einsum",
        "",
        12,
        std::slice::from_ref(&input),
        &[(DataType::Float32, vec![3, 2])],
        &[("equation", Attribute::String(b"ij->ji".to_vec()))],
    );
    let model = Model::new(&graph);
    let kernel = ep
        .get_kernel(model.graph.node(node), &[vec![2, 3]], 12)
        .unwrap();
    let buffer = ep.allocate(input.bytes.len(), 256).unwrap();
    unsafe {
        ep.runtime()
            .htod(&input.bytes, cuptr(buffer.as_ptr()))
            .unwrap()
    };
    let input_strides = compute_contiguous_strides(&input.shape);
    let view = TensorView::new(
        DevicePtr(buffer.as_ptr()),
        input.dtype,
        &input.shape,
        &input_strides,
        ep.device_id(),
    );
    let views = kernel
        .view_outputs(std::slice::from_ref(&view), &[vec![3, 2]], 1)
        .expect("permutation must be a zero-copy view");
    assert_eq!(views[0].shape, vec![3, 2]);
    assert_eq!(views[0].strides, vec![1, 3]);
    assert_eq!(views[0].byte_offset, 0);
    assert!(kernel.capture_support().is_supported());
    assert_eq!(einsum_execution_stats().view_aliases, 1);
    assert_eq!(einsum_execution_stats().materialization_bytes, 0);
    ep.deallocate(buffer).unwrap();

    let transpose = run_cuda(
        &ep,
        "Einsum",
        "",
        12,
        &[input],
        &[(DataType::Float32, vec![3, 2])],
        &[("equation", Attribute::String(b"ij->ji".to_vec()))],
    );
    assert_eq!(
        decode_floats(&transpose[0], DataType::Float32),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let diagonal = run_cuda(
        &ep,
        "Einsum",
        "",
        12,
        &[float_input(
            DataType::Float32,
            &[3, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )],
        &[(DataType::Float32, vec![3])],
        &[("equation", Attribute::String(b"ii->i".to_vec()))],
    );
    assert_eq!(
        decode_floats(&diagonal[0], DataType::Float32),
        vec![1.0, 5.0, 9.0]
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn explicit_transpose_reports_production_materialization_counters() {
    let _lock = suite_lock();
    let ep = require_cuda();
    onnx_runtime_ep_cuda::reset_movement_execution_stats();
    let output = run_cuda(
        &ep,
        "Transpose",
        "",
        13,
        &[float_input(
            DataType::Float32,
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )],
        &[(DataType::Float32, vec![3, 2])],
        &[("perm", Attribute::Ints(vec![1, 0]))],
    );
    assert_eq!(
        decode_floats(&output[0], DataType::Float32),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
    let stats = onnx_runtime_ep_cuda::movement_execution_stats();
    assert_eq!(stats.transpose_launches, 1);
    assert_eq!(stats.capture_recordings, 0);
    assert_eq!(
        stats.persistent_metadata_bytes,
        4 * std::mem::size_of::<u64>() as u64
    );
    assert_eq!(
        stats.materialization_bytes,
        6 * std::mem::size_of::<f32>() as u64
    );
}

fn make_direct_kernel(
    ep: &CudaExecutionProvider,
    equation: &str,
    input_shapes: &[Vec<usize>],
    output_shape: &[usize],
) -> (
    Box<dyn onnx_runtime_ep_api::Kernel>,
    Vec<Tensor>,
    Vec<onnx_runtime_ep_api::DeviceBuffer>,
    onnx_runtime_ep_api::DeviceBuffer,
) {
    let inputs = input_shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            let data = values(shape.iter().product(), index + 2);
            float_input(DataType::Float32, shape, &data)
        })
        .collect::<Vec<_>>();
    let (graph, node) = common::build_graph(
        "Einsum",
        "",
        12,
        &inputs,
        &[(DataType::Float32, output_shape.to_vec())],
        &[("equation", Attribute::String(equation.as_bytes().to_vec()))],
    );
    let model = Model::new(&graph);
    let kernel = ep
        .get_kernel(model.graph.node(node), input_shapes, 12)
        .unwrap();
    let buffers = inputs
        .iter()
        .map(|input| {
            let buffer = ep.allocate(input.bytes.len().max(1), 256).unwrap();
            if !input.bytes.is_empty() {
                unsafe {
                    ep.runtime()
                        .htod(&input.bytes, cuptr(buffer.as_ptr()))
                        .unwrap()
                };
            }
            buffer
        })
        .collect::<Vec<_>>();
    let output = ep
        .allocate(
            DataType::Float32
                .storage_bytes(output_shape.iter().product())
                .max(1),
            256,
        )
        .unwrap();
    (kernel, inputs, buffers, output)
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn warm_plan_is_allocation_free_with_stable_workspace_through_capture_and_replay() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();
    let input_shapes = [vec![32, 64], vec![4, 64, 48]];
    let output_shape = [4, 32, 48];
    let (kernel, inputs, buffers, mut output) =
        make_direct_kernel(&ep, "mk,...kn->...mn", &input_shapes, &output_shape);
    assert!(!kernel.capture_support().is_supported());
    assert!(kernel.capture_support().reason().unwrap().contains("warm"));
    let input_strides = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect::<Vec<_>>();
    let output_strides = compute_contiguous_strides(&output_shape);
    let views = inputs
        .iter()
        .zip(&buffers)
        .zip(&input_strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                DataType::Float32,
                &input.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let execute = |output: &mut onnx_runtime_ep_api::DeviceBuffer| {
        kernel
            .execute(
                &views,
                &mut [TensorMut::new(
                    DevicePtrMut(output.as_mut_ptr()),
                    DataType::Float32,
                    &output_shape,
                    &output_strides,
                    ep.device_id(),
                )],
            )
            .unwrap();
    };

    reset_einsum_execution_stats();
    execute(&mut output);
    let warm = einsum_execution_stats();
    assert_eq!(warm.plan_builds, 1);
    assert_eq!(warm.gemm_launches, 1);
    assert_eq!(warm.materialization_bytes, 0);
    assert!(warm.setup_ns > 0);
    assert!(kernel.capture_support().is_supported());
    let allocations = runtime.allocation_counts();
    let workspace = (warm.workspace_ptr, warm.workspace_bytes);

    execute(&mut output);
    let repeated = einsum_execution_stats();
    assert_eq!(repeated.plan_builds, 1);
    assert_eq!(repeated.plan_cache_hits, 1);
    assert_eq!(
        (repeated.workspace_ptr, repeated.workspace_bytes),
        workspace
    );
    assert_eq!(runtime.allocation_counts(), allocations);

    let graph_before = runtime.graph_execution_counts();
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute(&mut output);
    assert!(runtime.is_capturing().unwrap());
    assert_eq!(runtime.allocation_counts(), allocations);
    let recorded = einsum_execution_stats();
    assert_eq!(recorded.capture_recordings, 1);
    assert_eq!(
        (recorded.workspace_ptr, recorded.workspace_bytes),
        workspace
    );
    runtime.end_graph_capture().unwrap();
    for _ in 0..5 {
        runtime.replay_graph().unwrap();
    }
    runtime.synchronize().unwrap();
    let graph_after = runtime.graph_execution_counts();
    assert_eq!(graph_after.captures - graph_before.captures, 1);
    assert_eq!(graph_after.replays - graph_before.replays, 5);
    assert_eq!(runtime.allocation_counts(), allocations);
    assert!(runtime.reset_graph().unwrap());

    let mut bytes = vec![0u8; output_shape.iter().product::<usize>() * 4];
    unsafe { runtime.dtoh(&mut bytes, cuptr(output.as_ptr())).unwrap() };
    let got = decode_floats(&bytes, DataType::Float32);
    let a = quantize(
        &decode_floats(&inputs[0].bytes, DataType::Float32),
        DataType::Float32,
    );
    let b = quantize(
        &decode_floats(&inputs[1].bytes, DataType::Float32),
        DataType::Float32,
    );
    let expected = f64_gemm_reference(&a, &b, 4, 32, 64, 48, false, false, 0, 64 * 48);
    assert_close(&got, &expected, DataType::Float32, "captured replay");

    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn captured_private_resources_outlive_dropped_einsum_kernels() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();

    let gemm_shapes = [vec![32, 64], vec![4, 64, 48]];
    let gemm_output_shape = [4, 32, 48];
    let (gemm, gemm_inputs, gemm_buffers, mut gemm_output) =
        make_direct_kernel(&ep, "mk,...kn->...mn", &gemm_shapes, &gemm_output_shape);
    let gemm_input_strides = gemm_inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect::<Vec<_>>();
    let gemm_output_strides = compute_contiguous_strides(&gemm_output_shape);
    let gemm_views = gemm_inputs
        .iter()
        .zip(&gemm_buffers)
        .zip(&gemm_input_strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let execute_gemm = |kernel: &dyn onnx_runtime_ep_api::Kernel,
                        output: &mut onnx_runtime_ep_api::DeviceBuffer| {
        kernel
            .execute(
                &gemm_views,
                &mut [TensorMut::new(
                    DevicePtrMut(output.as_mut_ptr()),
                    DataType::Float32,
                    &gemm_output_shape,
                    &gemm_output_strides,
                    ep.device_id(),
                )],
            )
            .unwrap();
    };

    let view_shape = [2, 3];
    let view_output_shape = [3, 2];
    let (view, view_inputs, view_buffers, mut view_output) =
        make_direct_kernel(&ep, "ij->ji", &[view_shape.to_vec()], &view_output_shape);
    let view_input_strides = compute_contiguous_strides(&view_shape);
    let view_output_strides = compute_contiguous_strides(&view_output_shape);
    let view_input = TensorView::new(
        DevicePtr(view_buffers[0].as_ptr()),
        DataType::Float32,
        &view_shape,
        &view_input_strides,
        ep.device_id(),
    );
    let execute_view = |kernel: &dyn onnx_runtime_ep_api::Kernel,
                        output: &mut onnx_runtime_ep_api::DeviceBuffer| {
        kernel
            .execute(
                std::slice::from_ref(&view_input),
                &mut [TensorMut::new(
                    DevicePtrMut(output.as_mut_ptr()),
                    DataType::Float32,
                    &view_output_shape,
                    &view_output_strides,
                    ep.device_id(),
                )],
            )
            .unwrap();
    };

    reset_einsum_execution_stats();
    execute_gemm(gemm.as_ref(), &mut gemm_output);
    execute_view(view.as_ref(), &mut view_output);
    let workspace_bytes = einsum_execution_stats().workspace_bytes;
    assert_eq!(
        gemm.device_graph_resources().len(),
        usize::from(workspace_bytes != 0),
        "only an allocated cuBLASLt workspace needs graph ownership"
    );
    assert_eq!(
        view.device_graph_resources().len(),
        1,
        "materialized view metadata must expose one immutable graph owner"
    );
    assert!(gemm.capture_support().is_supported());
    assert!(view.capture_support().is_supported());

    runtime
        .begin_graph_capture(&[gemm.as_ref(), view.as_ref()])
        .unwrap();
    execute_gemm(gemm.as_ref(), &mut gemm_output);
    execute_view(view.as_ref(), &mut view_output);
    runtime.end_graph_capture().unwrap();

    let counts_before_drop = runtime.allocation_counts();
    let pooled_before_drop = runtime.raw_pool_retained_bytes();
    drop(gemm);
    drop(view);
    assert_eq!(
        runtime.allocation_counts(),
        counts_before_drop,
        "dropping kernels must not free addresses embedded in a live graph"
    );
    assert_eq!(
        runtime.raw_pool_retained_bytes(),
        pooled_before_drop,
        "dropping kernels must not return graph-owned addresses to the raw pool"
    );

    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();

    let mut gemm_bytes = vec![0u8; gemm_output_shape.iter().product::<usize>() * 4];
    unsafe {
        runtime
            .dtoh(&mut gemm_bytes, cuptr(gemm_output.as_ptr()))
            .unwrap()
    };
    let gemm_actual = decode_floats(&gemm_bytes, DataType::Float32);
    let gemm_a = quantize(
        &decode_floats(&gemm_inputs[0].bytes, DataType::Float32),
        DataType::Float32,
    );
    let gemm_b = quantize(
        &decode_floats(&gemm_inputs[1].bytes, DataType::Float32),
        DataType::Float32,
    );
    let gemm_expected =
        f64_gemm_reference(&gemm_a, &gemm_b, 4, 32, 64, 48, false, false, 0, 64 * 48);
    assert_close(
        &gemm_actual,
        &gemm_expected,
        DataType::Float32,
        "replay after GEMM kernel drop",
    );

    let mut view_bytes = vec![0u8; view_output_shape.iter().product::<usize>() * 4];
    unsafe {
        runtime
            .dtoh(&mut view_bytes, cuptr(view_output.as_ptr()))
            .unwrap()
    };
    let source = decode_floats(&view_inputs[0].bytes, DataType::Float32);
    assert_eq!(
        decode_floats(&view_bytes, DataType::Float32),
        vec![
            source[0], source[3], source[1], source[4], source[2], source[5]
        ]
    );

    let counts_before_reset = runtime.allocation_counts();
    let pooled_before_reset = runtime.raw_pool_retained_bytes();
    assert!(runtime.reset_graph().unwrap());
    let counts_after_reset = runtime.allocation_counts();
    let pooled_after_reset = runtime.raw_pool_retained_bytes();
    assert!(
        counts_after_reset.frees > counts_before_reset.frees
            || pooled_after_reset > pooled_before_reset,
        "destroying the graph must release its private workspace/metadata owners"
    );

    for buffer in gemm_buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(gemm_output).unwrap();
    for buffer in view_buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(view_output).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn materialized_view_failed_rewarm_preserves_capture_ready_snapshot() {
    #[cfg(feature = "gpu-tests")]
    materialized_view_failed_rewarm_preserves_capture_ready_snapshot_gpu();
}

#[cfg(feature = "gpu-tests")]
fn materialized_view_failed_rewarm_preserves_capture_ready_snapshot_gpu() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();
    let shape = [2usize, 3];
    let output_shape = [3usize, 2];
    let (kernel, inputs, buffers, mut a_output) =
        make_direct_kernel(&ep, "ij->ji", &[shape.to_vec()], &output_shape);
    let a_strides = compute_contiguous_strides(&shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let a_view = TensorView::new(
        DevicePtr(buffers[0].as_ptr()),
        DataType::Float32,
        &shape,
        &a_strides,
        ep.device_id(),
    );
    let run_a = |output: &mut onnx_runtime_ep_api::DeviceBuffer| {
        kernel.execute(
            std::slice::from_ref(&a_view),
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                ep.device_id(),
            )],
        )
    };
    run_a(&mut a_output).unwrap();
    let a_resource_id = kernel.device_graph_resources()[0].identity();

    let b_values = [10_f32, 11., 12., 0., 20., 21., 22., 0.];
    let b_bytes = b_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let b_buffer = ep.allocate(b_bytes.len(), 256).unwrap();
    let mut b_output = ep
        .allocate(output_shape.iter().product::<usize>() * 4, 256)
        .unwrap();
    unsafe {
        runtime.htod(&b_bytes, cuptr(b_buffer.as_ptr())).unwrap();
    }
    let b_strides = [4_i64, 1];
    let b_view = TensorView::new(
        DevicePtr(b_buffer.as_ptr()),
        DataType::Float32,
        &shape,
        &b_strides,
        ep.device_id(),
    );
    let run_b = |output: &mut onnx_runtime_ep_api::DeviceBuffer| {
        kernel.execute(
            std::slice::from_ref(&b_view),
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                ep.device_id(),
            )],
        )
    };

    let counts_before_failure = runtime.allocation_counts();
    let pooled_before_failure = runtime.raw_pool_retained_bytes();
    runtime.fail_warm_transaction_at_for_test(1);
    let failure = run_b(&mut b_output)
        .expect_err("valid strided view B must fail after staging new metadata");
    assert!(
        failure
            .to_string()
            .contains("injected staged warm-cache failure after Einsum view metadata"),
        "{failure}"
    );
    assert_eq!(
        kernel.device_graph_resources()[0].identity(),
        a_resource_id,
        "failed Einsum view rewarm must retain A's metadata owner"
    );
    let counts_after_failure = runtime.allocation_counts();
    assert!(
        counts_after_failure.frees > counts_before_failure.frees
            || runtime.raw_pool_retained_bytes() > pooled_before_failure,
        "the rejected Einsum candidate must return its staged allocation exactly once"
    );

    run_a(&mut a_output).unwrap();
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    run_a(&mut a_output).unwrap();
    runtime.end_graph_capture().unwrap();

    run_b(&mut b_output).unwrap();
    assert_ne!(
        kernel.device_graph_resources()[0].identity(),
        a_resource_id,
        "successful eager B must publish its own metadata owner"
    );
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    let mut replay = vec![0u8; inputs[0].bytes.len()];
    unsafe {
        runtime.dtoh(&mut replay, cuptr(a_output.as_ptr())).unwrap();
    }
    let source = decode_floats(&inputs[0].bytes, DataType::Float32);
    assert_eq!(
        decode_floats(&replay, DataType::Float32),
        vec![
            source[0], source[3], source[1], source[4], source[2], source[5]
        ]
    );
    assert!(runtime.reset_graph().unwrap());

    ep.deallocate(buffers.into_iter().next().unwrap()).unwrap();
    ep.deallocate(a_output).unwrap();
    ep.deallocate(b_buffer).unwrap();
    ep.deallocate(b_output).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn generic_failed_rewarm_preserves_capture_ready_snapshot() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();
    let shape = [2usize, 3];
    let output_shape = [2usize];
    let (kernel, inputs, buffers, mut a_output) =
        make_direct_kernel(&ep, "ij->i", &[shape.to_vec()], &output_shape);
    let strides = compute_contiguous_strides(&shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let a_view = TensorView::new(
        DevicePtr(buffers[0].as_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        ep.device_id(),
    );
    let run = |input: &TensorView, output: &mut onnx_runtime_ep_api::DeviceBuffer| {
        execute_einsum_with_route(
            kernel.as_ref(),
            std::slice::from_ref(input),
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                ep.device_id(),
            )],
            EinsumRouteOverride::GenericNative,
        )
    };
    run(&a_view, &mut a_output).unwrap();
    let resource = kernel.device_graph_resources()[0].identity();

    let b_values = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let b_bytes = common::raw(&b_values);
    let b_buffer = ep.allocate(b_bytes.len(), 256).unwrap();
    let mut b_output = ep.allocate(2 * 4, 256).unwrap();
    unsafe {
        runtime.htod(&b_bytes, cuptr(b_buffer.as_ptr())).unwrap();
        runtime
            .htod(&common::raw(&[123.0f32, 456.0]), cuptr(b_output.as_ptr()))
            .unwrap();
    }
    let b_view = TensorView::new(
        DevicePtr(b_buffer.as_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        ep.device_id(),
    );
    #[cfg(feature = "gpu-tests")]
    runtime.fail_warm_transaction_at_for_test(2);
    let counts_before_failure = runtime.allocation_counts();
    let failure = run(&b_view, &mut b_output).expect_err("rewarm fault must fail");
    assert!(
        failure
            .to_string()
            .contains("injected staged warm-cache failure after Einsum metadata"),
        "{failure}"
    );
    assert_eq!(kernel.device_graph_resources()[0].identity(), resource);
    assert!(
        runtime.allocation_counts().frees > counts_before_failure.frees
            || runtime.raw_pool_retained_bytes() > 0,
        "the failed candidate must release or pool its private metadata allocation"
    );
    let mut untouched = vec![0u8; 8];
    unsafe {
        runtime
            .dtoh(&mut untouched, cuptr(b_output.as_ptr()))
            .unwrap()
    };
    assert_eq!(decode_floats(&untouched, DataType::Float32), [123.0, 456.0]);

    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    run(&a_view, &mut a_output).unwrap();
    runtime.end_graph_capture().unwrap();
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    assert!(runtime.reset_graph().unwrap());
    let mut replay = vec![0u8; 8];
    unsafe { runtime.dtoh(&mut replay, cuptr(a_output.as_ptr())).unwrap() };
    let source = decode_floats(&inputs[0].bytes, DataType::Float32);
    assert_eq!(
        decode_floats(&replay, DataType::Float32),
        [
            source[..3].iter().sum::<f32>(),
            source[3..].iter().sum::<f32>(),
        ]
    );

    ep.deallocate(buffers.into_iter().next().unwrap()).unwrap();
    ep.deallocate(a_output).unwrap();
    ep.deallocate(b_buffer).unwrap();
    ep.deallocate(b_output).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn captured_generic_resources_outlive_dropped_kernel() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();
    let shape = [2usize, 3];
    let output_shape = [2usize];
    let (kernel, inputs, buffers, mut output) =
        make_direct_kernel(&ep, "ij->i", &[shape.to_vec()], &output_shape);
    let strides = compute_contiguous_strides(&shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let input = TensorView::new(
        DevicePtr(buffers[0].as_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        ep.device_id(),
    );
    let execute = |kernel: &dyn onnx_runtime_ep_api::Kernel,
                   output: &mut onnx_runtime_ep_api::DeviceBuffer| {
        execute_einsum_with_route(
            kernel,
            std::slice::from_ref(&input),
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                ep.device_id(),
            )],
            EinsumRouteOverride::GenericNative,
        )
        .unwrap();
    };
    execute(kernel.as_ref(), &mut output);
    assert_eq!(kernel.device_graph_resources().len(), 1);
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute(kernel.as_ref(), &mut output);
    runtime.end_graph_capture().unwrap();
    let counts_before_drop = runtime.allocation_counts();
    drop(kernel);
    assert_eq!(runtime.allocation_counts(), counts_before_drop);
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    let mut bytes = vec![0u8; 8];
    unsafe { runtime.dtoh(&mut bytes, cuptr(output.as_ptr())).unwrap() };
    let source = decode_floats(&inputs[0].bytes, DataType::Float32);
    assert_eq!(
        decode_floats(&bytes, DataType::Float32),
        [
            source[..3].iter().sum::<f32>(),
            source[3..].iter().sum::<f32>(),
        ]
    );
    assert!(runtime.reset_graph().unwrap());
    ep.deallocate(buffers.into_iter().next().unwrap()).unwrap();
    ep.deallocate(output).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn captured_optimized_plan_resources_outlive_dropped_kernel() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();
    let shapes = [vec![2], vec![2, 2], vec![2]];
    let output_shape: [usize; 0] = [];
    let (kernel, inputs, buffers, mut output) =
        make_direct_kernel(&ep, "i,ij,j->", &shapes, &output_shape);
    let strides = inputs
        .iter()
        .map(|input| compute_contiguous_strides(&input.shape))
        .collect::<Vec<_>>();
    let output_strides = compute_contiguous_strides(&output_shape);
    let views = inputs
        .iter()
        .zip(&buffers)
        .zip(&strides)
        .map(|((input, buffer), strides)| {
            TensorView::new(
                DevicePtr(buffer.as_ptr()),
                input.dtype,
                &input.shape,
                strides,
                ep.device_id(),
            )
        })
        .collect::<Vec<_>>();
    let execute = |kernel: &dyn onnx_runtime_ep_api::Kernel,
                   output: &mut onnx_runtime_ep_api::DeviceBuffer| {
        execute_einsum_with_route(
            kernel,
            &views,
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                ep.device_id(),
            )],
            EinsumRouteOverride::Optimized,
        )
        .unwrap();
    };
    reset_einsum_execution_stats();
    execute(kernel.as_ref(), &mut output);
    let stats = einsum_execution_stats();
    assert_eq!(
        stats.last_route,
        Some(onnx_runtime_ep_cuda::CudaEinsumRoute::OptimizedDp)
    );
    assert!(stats.optimized_cublas_launches > 0);
    assert!(!kernel.device_graph_resources().is_empty());
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute(kernel.as_ref(), &mut output);
    runtime.end_graph_capture().unwrap();

    let counts_before_drop = runtime.allocation_counts();
    drop(kernel);
    assert_eq!(runtime.allocation_counts(), counts_before_drop);
    runtime.replay_graph().unwrap();
    runtime.synchronize().unwrap();
    let mut bytes = vec![0u8; 4];
    unsafe { runtime.dtoh(&mut bytes, cuptr(output.as_ptr())).unwrap() };
    let values = inputs
        .iter()
        .map(|input| decode_floats(&input.bytes, DataType::Float32))
        .collect::<Vec<_>>();
    let expected = (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| values[0][i] * values[1][i * 2 + j] * values[2][j])
        .sum::<f32>();
    assert_close(
        &decode_floats(&bytes, DataType::Float32),
        &[expected as f64],
        DataType::Float32,
        "optimized replay after kernel drop",
    );
    assert!(runtime.reset_graph().unwrap());
    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn contraction_rejects_output_alias_before_launch() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();
    let input_shapes = [vec![2, 2], vec![2, 2]];
    let output_shape = [2, 2];
    let (kernel, inputs, mut buffers, output) =
        make_direct_kernel(&ep, "ik,kj->ij", &input_shapes, &output_shape);
    ep.deallocate(output).unwrap();
    let strides = compute_contiguous_strides(&[2, 2]);
    let a = TensorView::new(
        DevicePtr(buffers[0].as_ptr()),
        DataType::Float32,
        &[2, 2],
        &strides,
        ep.device_id(),
    );
    let b = TensorView::new(
        DevicePtr(buffers[1].as_ptr()),
        DataType::Float32,
        &[2, 2],
        &strides,
        ep.device_id(),
    );
    let before = einsum_execution_stats().gemm_launches;
    let error = kernel
        .execute(
            &[a, b],
            &mut [TensorMut::new(
                DevicePtrMut(buffers[0].as_mut_ptr()),
                DataType::Float32,
                &[2, 2],
                &strides,
                ep.device_id(),
            )],
        )
        .unwrap_err();
    assert!(error.to_string().contains("overlaps input #0"));
    assert_eq!(einsum_execution_stats().gemm_launches, before);
    runtime.synchronize().unwrap();
    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }
    drop(inputs);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn materialized_view_rejects_partial_offset_overlap_before_mutation() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();
    let input_shape = [2, 3];
    let output_shape = [3, 2];
    let (kernel, inputs, buffers, output) =
        make_direct_kernel(&ep, "ij->ji", &[input_shape.to_vec()], &output_shape);
    ep.deallocate(output).unwrap();
    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }

    let mut shared = ep.allocate(64, 256).unwrap();
    let input_offset = 8u64;
    unsafe {
        runtime
            .htod(
                &inputs[0].bytes,
                cuptr(shared.as_ptr()).checked_add(input_offset).unwrap(),
            )
            .unwrap()
    };
    let input_strides = compute_contiguous_strides(&input_shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let input = TensorView::new(
        DevicePtr(shared.as_ptr()),
        DataType::Float32,
        &input_shape,
        &input_strides,
        ep.device_id(),
    )
    .with_byte_offset(input_offset as usize);

    reset_einsum_execution_stats();
    let allocations_before = runtime.allocation_counts();
    let error = kernel
        .execute(
            std::slice::from_ref(&input),
            &mut [TensorMut::new(
                DevicePtrMut(shared.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                ep.device_id(),
            )
            .with_byte_offset(12)],
        )
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("materialized permutation/diagonal"));
    assert!(message.contains("overlaps input byte range"));
    assert_eq!(
        runtime.allocation_counts(),
        allocations_before,
        "overlap rejection must happen before persistent metadata allocation"
    );
    assert_eq!(einsum_execution_stats().view_materializations, 0);
    assert_eq!(einsum_execution_stats().materialization_bytes, 0);

    for (label, output_offset) in [("adjacent", 32usize), ("gapped", 40usize)] {
        kernel
            .execute(
                std::slice::from_ref(&input),
                &mut [TensorMut::new(
                    DevicePtrMut(shared.as_mut_ptr()),
                    DataType::Float32,
                    &output_shape,
                    &output_strides,
                    ep.device_id(),
                )
                .with_byte_offset(output_offset)],
            )
            .unwrap_or_else(|error| panic!("{label} output must be legal: {error}"));
        runtime.synchronize().unwrap();
        let mut bytes = vec![0u8; 24];
        unsafe {
            runtime
                .dtoh(
                    &mut bytes,
                    cuptr(shared.as_ptr())
                        .checked_add(output_offset as u64)
                        .unwrap(),
                )
                .unwrap()
        };
        assert_eq!(
            decode_floats(&bytes, DataType::Float32),
            vec![
                decode_floats(&inputs[0].bytes, DataType::Float32)[0],
                decode_floats(&inputs[0].bytes, DataType::Float32)[3],
                decode_floats(&inputs[0].bytes, DataType::Float32)[1],
                decode_floats(&inputs[0].bytes, DataType::Float32)[4],
                decode_floats(&inputs[0].bytes, DataType::Float32)[2],
                decode_floats(&inputs[0].bytes, DataType::Float32)[5],
            ],
            "{label}"
        );
    }

    ep.deallocate(shared).unwrap();
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn materialized_view_address_overflow_is_actionable_and_pre_mutation() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let runtime = ep.runtime();
    let input_shape = [2, 3];
    let output_shape = [3, 2];
    let (kernel, _inputs, buffers, mut output) =
        make_direct_kernel(&ep, "ij->ji", &[input_shape.to_vec()], &output_shape);
    let input_strides = compute_contiguous_strides(&input_shape);
    let output_strides = compute_contiguous_strides(&output_shape);
    let invalid_offset = usize::MAX - (usize::MAX % DataType::Float32.byte_size());
    let input = TensorView::new(
        DevicePtr(buffers[0].as_ptr()),
        DataType::Float32,
        &input_shape,
        &input_strides,
        ep.device_id(),
    )
    .with_byte_offset(invalid_offset);

    reset_einsum_execution_stats();
    let allocations_before = runtime.allocation_counts();
    let error = kernel
        .execute(
            std::slice::from_ref(&input),
            &mut [TensorMut::new(
                DevicePtrMut(output.as_mut_ptr()),
                DataType::Float32,
                &output_shape,
                &output_strides,
                ep.device_id(),
            )],
        )
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("materialized permutation/diagonal input"));
    assert!(message.contains("address range overflows u64"));
    assert!(message.contains(&format!("byte_offset {invalid_offset}")));
    assert!(message.contains("use a view"));
    assert_eq!(
        runtime.allocation_counts(),
        allocations_before,
        "overflow rejection must happen before persistent metadata allocation"
    );
    assert_eq!(einsum_execution_stats().view_materializations, 0);

    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }
    ep.deallocate(output).unwrap();
}

enum BenchMode {
    Einsum(Box<dyn onnx_runtime_ep_api::Kernel>),
    Materialized {
        transpose: Box<dyn onnx_runtime_ep_api::Kernel>,
        einsum: Box<dyn onnx_runtime_ep_api::Kernel>,
    },
}

struct BenchFixture {
    mode: BenchMode,
    a: Tensor,
    b: Tensor,
    a_buffer: onnx_runtime_ep_api::DeviceBuffer,
    b_buffer: onnx_runtime_ep_api::DeviceBuffer,
    temporary: Option<onnx_runtime_ep_api::DeviceBuffer>,
    output: onnx_runtime_ep_api::DeviceBuffer,
    output_shape: Vec<usize>,
    dtype: DataType,
}

const BENCH_M: usize = 256;
const BENCH_K: usize = 512;
const BENCH_N: usize = 384;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchArm {
    Descriptor,
    Materialized,
    Control,
}

impl BenchArm {
    fn label(self) -> &'static str {
        match self {
            Self::Descriptor => "descriptor-transpose",
            Self::Materialized => "explicit-materialization",
            Self::Control => "canonical-control",
        }
    }

    fn expected_route(self) -> &'static str {
        match self {
            Self::Descriptor => "descriptor-transpose",
            Self::Materialized => "explicit-materialization",
            Self::Control => "canonical",
        }
    }
}

impl BenchFixture {
    fn kernel_refs(&self) -> Vec<&dyn onnx_runtime_ep_api::Kernel> {
        match &self.mode {
            BenchMode::Einsum(kernel) => vec![kernel.as_ref()],
            BenchMode::Materialized { transpose, einsum } => {
                vec![transpose.as_ref(), einsum.as_ref()]
            }
        }
    }

    fn execute(&self, ep: &CudaExecutionProvider) {
        let a_strides = compute_contiguous_strides(&self.a.shape);
        let b_strides = compute_contiguous_strides(&self.b.shape);
        let output_strides = compute_contiguous_strides(&self.output_shape);
        let a = TensorView::new(
            DevicePtr(self.a_buffer.as_ptr()),
            self.a.dtype,
            &self.a.shape,
            &a_strides,
            ep.device_id(),
        );
        let b = TensorView::new(
            DevicePtr(self.b_buffer.as_ptr()),
            self.b.dtype,
            &self.b.shape,
            &b_strides,
            ep.device_id(),
        );
        let output = || {
            TensorMut::new(
                DevicePtrMut(self.output.as_ptr() as *mut c_void),
                self.dtype,
                &self.output_shape,
                &output_strides,
                ep.device_id(),
            )
        };
        match &self.mode {
            BenchMode::Einsum(kernel) => kernel.execute(&[a, b], &mut [output()]).unwrap(),
            BenchMode::Materialized { transpose, einsum } => {
                let temporary = self.temporary.as_ref().unwrap();
                let temporary_shape = [self.a.shape[1], self.a.shape[0]];
                let temporary_strides = compute_contiguous_strides(&temporary_shape);
                transpose
                    .execute(
                        std::slice::from_ref(&a),
                        &mut [TensorMut::new(
                            DevicePtrMut(temporary.as_ptr() as *mut c_void),
                            self.dtype,
                            &temporary_shape,
                            &temporary_strides,
                            ep.device_id(),
                        )],
                    )
                    .unwrap();
                let temporary_view = TensorView::new(
                    DevicePtr(temporary.as_ptr()),
                    self.dtype,
                    &temporary_shape,
                    &temporary_strides,
                    ep.device_id(),
                );
                einsum
                    .execute(&[temporary_view, b], &mut [output()])
                    .unwrap();
            }
        }
    }

    fn finish(self, ep: &CudaExecutionProvider) {
        let Self {
            mode,
            a_buffer,
            b_buffer,
            temporary,
            output,
            ..
        } = self;
        drop(mode);
        ep.deallocate(a_buffer).unwrap();
        ep.deallocate(b_buffer).unwrap();
        if let Some(temporary) = temporary {
            ep.deallocate(temporary).unwrap();
        }
        ep.deallocate(output).unwrap();
    }

    fn output_bytes(&self, ep: &CudaExecutionProvider) -> Vec<u8> {
        ep.runtime().synchronize().unwrap();
        let mut bytes = vec![0; self.dtype.storage_bytes(self.output_shape.iter().product())];
        unsafe {
            ep.runtime()
                .dtoh(&mut bytes, cuptr(self.output.as_ptr()))
                .unwrap()
        };
        bytes
    }
}

fn bench_kernel(
    ep: &CudaExecutionProvider,
    op: &str,
    inputs: &[Tensor],
    output_shape: &[usize],
    attrs: &[(&str, Attribute)],
) -> Box<dyn onnx_runtime_ep_api::Kernel> {
    let (graph, node) = common::build_graph(
        op,
        "",
        if op == "Einsum" { 12 } else { 13 },
        inputs,
        &[(inputs[0].dtype, output_shape.to_vec())],
        attrs,
    );
    let model = Model::new(&graph);
    ep.get_kernel(
        model.graph.node(node),
        &inputs
            .iter()
            .map(|input| input.shape.clone())
            .collect::<Vec<_>>(),
        if op == "Einsum" { 12 } else { 13 },
    )
    .unwrap()
}

fn bench_fixture(
    ep: &CudaExecutionProvider,
    arm: BenchArm,
    dtype: DataType,
    logical_a: &[f32],
    b_values: &[f32],
) -> BenchFixture {
    let canonical = arm == BenchArm::Control;
    let a_shape = if canonical {
        vec![BENCH_M, BENCH_K]
    } else {
        vec![BENCH_K, BENCH_M]
    };
    let physical_a = if canonical {
        logical_a.to_vec()
    } else {
        let mut transposed = vec![0.0; logical_a.len()];
        for row in 0..BENCH_M {
            for reduction in 0..BENCH_K {
                transposed[reduction * BENCH_M + row] = logical_a[row * BENCH_K + reduction];
            }
        }
        transposed
    };
    let a = float_input(dtype, &a_shape, &physical_a);
    let b = float_input(dtype, &[BENCH_K, BENCH_N], b_values);
    let output_shape = vec![BENCH_M, BENCH_N];
    let a_buffer = ep.allocate(a.bytes.len(), 256).unwrap();
    let b_buffer = ep.allocate(b.bytes.len(), 256).unwrap();
    let output = ep
        .allocate(dtype.storage_bytes(BENCH_M * BENCH_N), 256)
        .unwrap();
    unsafe {
        ep.runtime()
            .htod(&a.bytes, cuptr(a_buffer.as_ptr()))
            .unwrap();
        ep.runtime()
            .htod(&b.bytes, cuptr(b_buffer.as_ptr()))
            .unwrap();
    }

    if arm == BenchArm::Materialized {
        let temporary = ep
            .allocate(dtype.storage_bytes(BENCH_M * BENCH_K), 256)
            .unwrap();
        let transpose = bench_kernel(
            ep,
            "Transpose",
            std::slice::from_ref(&a),
            &[BENCH_M, BENCH_K],
            &[("perm", Attribute::Ints(vec![1, 0]))],
        );
        let temporary_tensor =
            float_input(dtype, &[BENCH_M, BENCH_K], &vec![0.0; BENCH_M * BENCH_K]);
        let einsum = bench_kernel(
            ep,
            "Einsum",
            &[temporary_tensor, b.clone()],
            &output_shape,
            &[("equation", Attribute::String(b"ik,kj->ij".to_vec()))],
        );
        BenchFixture {
            mode: BenchMode::Materialized { transpose, einsum },
            a,
            b,
            a_buffer,
            b_buffer,
            temporary: Some(temporary),
            output,
            output_shape,
            dtype,
        }
    } else {
        let equation = if canonical {
            b"ik,kj->ij".as_slice()
        } else {
            b"ki,kj->ij".as_slice()
        };
        let einsum = bench_kernel(
            ep,
            "Einsum",
            &[a.clone(), b.clone()],
            &output_shape,
            &[("equation", Attribute::String(equation.to_vec()))],
        );
        BenchFixture {
            mode: BenchMode::Einsum(einsum),
            a,
            b,
            a_buffer,
            b_buffer,
            temporary: None,
            output,
            output_shape,
            dtype,
        }
    }
}

fn physical_gpu() -> String {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES")
        .expect("benchmark requires CUDA_VISIBLE_DEVICES=<one physical index or UUID>");
    assert!(
        !visible.is_empty() && !visible.contains(','),
        "benchmark requires exactly one CUDA_VISIBLE_DEVICES selector, got `{visible}`"
    );
    let physical =
        std::env::var("ONNX_GENAI_CUDA_PHYSICAL_DEVICE").unwrap_or_else(|_| visible.clone());
    assert_eq!(
        physical, visible,
        "ONNX_GENAI_CUDA_PHYSICAL_DEVICE must identify the one CUDA_VISIBLE_DEVICES device"
    );
    assert_eq!(
        std::env::var("ONNX_GENAI_CUDA_DEVICE").as_deref(),
        Ok("0"),
        "logical CUDA mapping must be pinned with ONNX_GENAI_CUDA_DEVICE=0"
    );
    physical
}

#[derive(Clone, Debug)]
struct GpuState {
    utilization: u32,
    clock_mhz: u32,
    max_clock_mhz: u32,
    power_w: f64,
    memory_mib: u64,
    foreign_processes: Vec<String>,
}

fn gpu_state(label: &str, phase: &str) -> GpuState {
    let physical = physical_gpu();
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "-i",
            &physical,
            "--query-gpu=utilization.gpu,clocks.sm,clocks.max.sm,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("nvidia-smi");
    assert!(output.status.success(), "nvidia-smi failed");
    let state = String::from_utf8(output.stdout).unwrap();
    let fields = state.trim().split(',').map(str::trim).collect::<Vec<_>>();
    assert_eq!(fields.len(), 5, "unexpected nvidia-smi state row: {state}");
    let processes = std::process::Command::new("nvidia-smi")
        .args([
            "-i",
            &physical,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("nvidia-smi compute process query");
    assert!(
        processes.status.success(),
        "nvidia-smi compute process query failed"
    );
    let own_pid = std::process::id();
    let foreign_processes = String::from_utf8(processes.stdout)
        .unwrap()
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(',').map(str::trim);
            let pid = fields.next()?.parse::<u32>().ok()?;
            let memory = fields.next().unwrap_or("unknown");
            (pid != own_pid).then(|| format!("{pid}:{memory}MiB"))
        })
        .collect::<Vec<_>>();
    let state = GpuState {
        utilization: fields[0].parse().unwrap(),
        clock_mhz: fields[1].parse().unwrap(),
        max_clock_mhz: fields[2].parse().unwrap(),
        power_w: fields[3].parse().unwrap(),
        memory_mib: fields[4].parse().unwrap(),
        foreign_processes,
    };
    println!(
        "GPU_STATE,label={label},phase={phase},util_pct={},clock_mhz={},max_clock_mhz={},power_w={:.2},memory_mib={},foreign_processes={}",
        state.utilization,
        state.clock_mhz,
        state.max_clock_mhz,
        state.power_w,
        state.memory_mib,
        if state.foreign_processes.is_empty() {
            "none".to_string()
        } else {
            state.foreign_processes.join(";")
        }
    );
    state
}

fn hostlock_provenance(label: &str, expect_runnable: u32) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = std::process::Command::new(root.join("scripts/hostlock.sh"))
        .args([
            "provenance",
            "--oneline",
            "--expect-runnable",
            &expect_runnable.to_string(),
        ])
        .current_dir(&root)
        .output()
        .expect("hostlock provenance");
    assert!(
        output.status.success(),
        "hostlock provenance failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let row = String::from_utf8(output.stdout).unwrap();
    let field = |key: &str| {
        row.split_whitespace()
            .find_map(|entry| entry.strip_prefix(&format!("{key}=")))
    };
    assert_eq!(field("hostlock_state"), Some("HELD"), "{row}");
    assert_eq!(field("declared"), Some("yes"), "{row}");
    assert_eq!(field("held_owner_source"), Some("flag"), "{row}");
    assert_eq!(
        field("held_by"),
        std::env::var("HOSTLOCK_OWNER").ok().as_deref(),
        "{row}"
    );
    assert!(
        field("gate").is_some_and(|gate| gate.starts_with("satisfied:")),
        "{row}"
    );
    assert_eq!(field("contended"), Some("no"), "{row}");
    assert_eq!(field("lock_scope"), Some("box"), "{row}");
    println!("HOSTLOCK,label={label},{}", row.trim());
}

fn wait_for_idle_gpu(label: &str) {
    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    let mut consecutive = 0;
    loop {
        let state = gpu_state(label, "idle-precheck");
        if state.utilization <= 2 && state.foreign_processes.is_empty() {
            consecutive += 1;
            if consecutive == 2 {
                return;
            }
        } else {
            consecutive = 0;
        }
        assert!(
            Instant::now() < deadline,
            "{label}: GPU did not become idle and foreign-process-free within 60 seconds"
        );
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

struct BenchResult {
    arm: BenchArm,
    batch: usize,
    slot: usize,
    route: &'static str,
    kernel_us: f32,
    fixture_setup_us: f64,
    warm_execute_us: f64,
    plan_setup_us: f64,
    capture_us: f64,
    ramp_ms: f64,
    captures: u64,
    ramp_replays: u64,
    timed_replays: u64,
    captured_kernel_launches: u64,
    workspace_bytes: u64,
    persistent_metadata_bytes: u64,
    materialization_bytes: u64,
    gemm_launches: u64,
    canonical_gemm_launches: u64,
    descriptor_gemm_launches: u64,
    transpose_launches: u64,
    fallback_count: u64,
    fallback_reason: String,
    allocations_before: onnx_runtime_ep_cuda::CudaAllocationCounts,
    allocations_after_warm: onnx_runtime_ep_cuda::CudaAllocationCounts,
    allocations_after_capture: onnx_runtime_ep_cuda::CudaAllocationCounts,
    allocations_after_timed: onnx_runtime_ep_cuda::CudaAllocationCounts,
    allocations_after_finish: onnx_runtime_ep_cuda::CudaAllocationCounts,
    clock_pre_mhz: u32,
    clock_post_mhz: u32,
    power_pre_w: f64,
    power_post_w: f64,
    oracle_max_abs: f64,
    oracle_max_rel: f64,
    output_digest: u64,
}

fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn oracle_metrics(got: &[f32], expected: &[f64], dtype: DataType) -> (f64, f64) {
    assert_close(got, expected, dtype, "256x512x384 captured benchmark");
    got.iter().zip(expected).fold(
        (0.0_f64, 0.0_f64),
        |(max_abs, max_rel), (&got, &expected)| {
            let error = (f64::from(got) - expected).abs();
            (
                max_abs.max(error),
                max_rel.max(error / expected.abs().max(1e-12)),
            )
        },
    )
}

fn delta_graph_counts(
    after: onnx_runtime_ep_cuda::CudaGraphExecutionCounts,
    before: onnx_runtime_ep_cuda::CudaGraphExecutionCounts,
) -> onnx_runtime_ep_cuda::CudaGraphExecutionCounts {
    onnx_runtime_ep_cuda::CudaGraphExecutionCounts {
        captures: after.captures - before.captures,
        replays: after.replays - before.replays,
    }
}

fn ramp_graph(
    runtime: &onnx_runtime_ep_cuda::runtime::CudaRuntime,
    label: &str,
    ramp_seconds: u64,
) -> (f64, u64) {
    let stop = Arc::new(AtomicBool::new(false));
    let witness_stop = Arc::clone(&stop);
    let witness_label = label.to_string();
    let witness = std::thread::spawn(move || {
        while !witness_stop.load(Ordering::Relaxed) {
            gpu_state(&witness_label, "warm-sample");
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });
    let graph_before = runtime.graph_execution_counts();
    let start = Instant::now();
    while start.elapsed().as_secs() < ramp_seconds {
        for _ in 0..1024 {
            runtime.replay_graph().unwrap();
        }
        runtime.synchronize().unwrap();
    }
    let elapsed = start.elapsed();
    stop.store(true, Ordering::Relaxed);
    witness.join().unwrap();
    let graph_after = runtime.graph_execution_counts();
    assert!(
        elapsed >= std::time::Duration::from_secs(ramp_seconds),
        "{label}: warmup ended before its wall-clock floor"
    );
    (
        elapsed.as_secs_f64() * 1000.0,
        graph_after.replays - graph_before.replays,
    )
}

fn measure_bench_arm(
    ep: &CudaExecutionProvider,
    arm: BenchArm,
    dtype: DataType,
    batch_index: usize,
    slot: usize,
    logical_a: &[f32],
    b_values: &[f32],
    expected: &[f64],
    replay_batch: usize,
    ramp_seconds: u64,
    expect_runnable: u32,
) -> BenchResult {
    let label = format!("{}-{dtype:?}-b{batch_index}-s{slot}", arm.label());
    hostlock_provenance(&label, expect_runnable);
    wait_for_idle_gpu(&label);
    reset_einsum_execution_stats();
    onnx_runtime_ep_cuda::reset_movement_execution_stats();
    let runtime = ep.runtime();
    let allocations_before = runtime.allocation_counts();
    let fixture_setup = Instant::now();
    let fixture = bench_fixture(ep, arm, dtype, logical_a, b_values);
    let fixture_setup_us = fixture_setup.elapsed().as_secs_f64() * 1e6;

    let warm_execute = Instant::now();
    fixture.execute(ep);
    runtime.synchronize().unwrap();
    let warm_execute_us = warm_execute.elapsed().as_secs_f64() * 1e6;
    assert!(
        fixture
            .kernel_refs()
            .iter()
            .all(|kernel| kernel.capture_support().is_supported()),
        "{label}: every kernel must be warmed and capturable"
    );
    let allocations_after_warm = runtime.allocation_counts();
    let graph_before_capture = runtime.graph_execution_counts();
    let capture = Instant::now();
    runtime.begin_graph_capture(&fixture.kernel_refs()).unwrap();
    fixture.execute(ep);
    runtime.end_graph_capture().unwrap();
    let capture_us = capture.elapsed().as_secs_f64() * 1e6;
    let graph_after_capture = runtime.graph_execution_counts();
    let capture_counts = delta_graph_counts(graph_after_capture, graph_before_capture);
    assert_eq!(capture_counts.captures, 1, "{label}: capture count");
    assert_eq!(capture_counts.replays, 0, "{label}: capture is not replay");
    let allocations_after_capture = runtime.allocation_counts();
    assert_eq!(
        allocations_after_capture, allocations_after_warm,
        "{label}: capture allocated after warmup"
    );

    let (ramp_ms, ramp_replays) = ramp_graph(runtime, &label, ramp_seconds);
    let clock_pre = gpu_state(&label, "timed-pre");
    assert!(
        clock_pre.foreign_processes.is_empty(),
        "{label}: foreign GPU process appeared before the timed region: {clock_pre:?}"
    );
    assert!(
        clock_pre.clock_mhz as f64 >= f64::from(clock_pre.max_clock_mhz) * 0.90,
        "{label}: timed region started below 90% of maximum SM clock: {clock_pre:?}"
    );

    let start = event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
    let end = event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
    let graph_before_timed = runtime.graph_execution_counts();
    unsafe { event::record(start, runtime.stream_ptr()) }.unwrap();
    for _ in 0..replay_batch {
        runtime.replay_graph().unwrap();
    }
    unsafe { event::record(end, runtime.stream_ptr()) }.unwrap();
    unsafe { event::synchronize(end) }.unwrap();
    let kernel_us = unsafe { event::elapsed(start, end) }.unwrap() * 1000.0 / replay_batch as f32;
    unsafe {
        event::destroy(start).unwrap();
        event::destroy(end).unwrap();
    }
    let graph_after_timed = runtime.graph_execution_counts();
    let timed_replays = graph_after_timed.replays - graph_before_timed.replays;
    assert_eq!(
        timed_replays, replay_batch as u64,
        "{label}: timed replay count"
    );
    let allocations_after_timed = runtime.allocation_counts();
    assert_eq!(
        allocations_after_timed, allocations_after_warm,
        "{label}: replay allocated after warmup"
    );
    let clock_post = gpu_state(&label, "timed-post");
    assert!(
        clock_post.foreign_processes.is_empty(),
        "{label}: foreign GPU process appeared during the timed region: {clock_post:?}"
    );
    hostlock_provenance(&format!("{label}-timed-post"), expect_runnable);
    assert!(
        clock_post.clock_mhz as f64 >= f64::from(clock_post.max_clock_mhz) * 0.90,
        "{label}: timed region ended below 90% of maximum SM clock: {clock_post:?}"
    );
    let clock_drift =
        (f64::from(clock_post.clock_mhz) / f64::from(clock_pre.clock_mhz) - 1.0).abs();
    assert!(
        clock_drift <= 0.05,
        "{label}: SM clock moved {:.2}% across timed region",
        clock_drift * 100.0
    );

    let output_bytes = fixture.output_bytes(ep);
    let output = decode_floats(&output_bytes, dtype);
    let (oracle_max_abs, oracle_max_rel) = oracle_metrics(&output, expected, dtype);
    let output_digest = digest(&output_bytes);
    let stats = einsum_execution_stats();
    let movement = onnx_runtime_ep_cuda::movement_execution_stats();
    let route = if movement.capture_recordings != 0 {
        "explicit-materialization"
    } else if stats.descriptor_transpose_gemm_launches != 0 {
        "descriptor-transpose"
    } else if stats.canonical_gemm_launches != 0 {
        "canonical"
    } else {
        "unknown"
    };
    assert_eq!(route, arm.expected_route(), "{label}: actual route");
    assert_eq!(
        stats.claim_fallbacks, 0,
        "{label}: benchmark arm fell back: {:?}",
        stats.last_fallback_reason
    );
    let captured_kernel_launches = stats.capture_recordings + movement.capture_recordings;
    assert_eq!(
        captured_kernel_launches,
        if arm == BenchArm::Materialized { 2 } else { 1 },
        "{label}: captured kernel launch count"
    );
    let persistent_metadata_bytes =
        stats.persistent_metadata_bytes + movement.persistent_metadata_bytes;
    let materialization_bytes = stats.materialization_bytes + movement.materialization_bytes;

    assert!(runtime.reset_graph().unwrap());
    fixture.finish(ep);
    let allocations_after_finish = runtime.allocation_counts();
    BenchResult {
        arm,
        batch: batch_index,
        slot,
        route,
        kernel_us,
        fixture_setup_us,
        warm_execute_us,
        plan_setup_us: stats.setup_ns as f64 / 1000.0,
        capture_us,
        ramp_ms,
        captures: capture_counts.captures,
        ramp_replays,
        timed_replays,
        captured_kernel_launches,
        workspace_bytes: stats.workspace_bytes,
        persistent_metadata_bytes,
        materialization_bytes,
        gemm_launches: stats.gemm_launches,
        canonical_gemm_launches: stats.canonical_gemm_launches,
        descriptor_gemm_launches: stats.descriptor_transpose_gemm_launches,
        transpose_launches: movement.transpose_launches,
        fallback_count: stats.claim_fallbacks,
        fallback_reason: stats
            .last_fallback_reason
            .unwrap_or_else(|| "none".to_string())
            .replace(',', ";")
            .replace('\n', " "),
        allocations_before,
        allocations_after_warm,
        allocations_after_capture,
        allocations_after_timed,
        allocations_after_finish,
        clock_pre_mhz: clock_pre.clock_mhz,
        clock_post_mhz: clock_post.clock_mhz,
        power_pre_w: clock_pre.power_w,
        power_post_w: clock_post.power_w,
        oracle_max_abs,
        oracle_max_rel,
        output_digest,
    }
}

fn print_bench_result(dtype: DataType, result: &BenchResult) {
    println!(
        "BENCH,{dtype:?},{},{},{},{},{:.6},{:.1},{:.1},{:.1},{:.1},{:.1},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.2},{:.2},{:.8e},{:.8e},{:016x}",
        result.batch,
        result.slot,
        result.arm.label(),
        result.route,
        result.kernel_us,
        result.fixture_setup_us,
        result.warm_execute_us,
        result.plan_setup_us,
        result.capture_us,
        result.ramp_ms,
        result.captures,
        result.ramp_replays,
        result.timed_replays,
        result.captured_kernel_launches,
        result.gemm_launches,
        result.canonical_gemm_launches,
        result.descriptor_gemm_launches,
        result.transpose_launches,
        result.fallback_count,
        result.fallback_reason,
        result.workspace_bytes,
        result.persistent_metadata_bytes,
        result.materialization_bytes,
        result.allocations_before.allocations,
        result.allocations_before.frees,
        result.allocations_after_warm.allocations,
        result.allocations_after_warm.frees,
        result.allocations_after_capture.allocations,
        result.allocations_after_capture.frees,
        result.allocations_after_timed.allocations,
        result.allocations_after_timed.frees,
        result.allocations_after_finish.allocations,
        result.allocations_after_finish.frees,
        result.clock_pre_mhz,
        result.clock_post_mhz,
        result.power_pre_w,
        result.power_post_w,
        result.oracle_max_abs,
        result.oracle_max_rel,
        result.output_digest,
    );
}

fn median_range(values: &[f32]) -> (f32, f32, f32) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    (median, sorted[0], sorted[sorted.len() - 1])
}

fn gpu_identity(ep: &CudaExecutionProvider) {
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap();
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "-i",
            &physical_gpu(),
            "--query-gpu=index,uuid,name,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("nvidia-smi identity");
    assert!(output.status.success(), "nvidia-smi identity query failed");
    assert_eq!(ep.runtime().ordinal(), 0);
    println!(
        "GPU_IDENTITY,logical_ordinal=0,cuda_visible_devices={},physical={}",
        visible,
        String::from_utf8(output.stdout).unwrap().trim()
    );
}

#[test]
#[ignore = "requires a pinned CUDA GPU; exact 256x512x384 f64 oracle is intentionally expensive"]
fn einsum_benchmark_arms_match_exact_f64_oracle() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let logical_a = values(BENCH_M * BENCH_K, 13);
    let b_values = values(BENCH_K * BENCH_N, 17);
    for dtype in [DataType::Float16, DataType::Float32] {
        let expected = f64_gemm_reference(
            &quantize(&logical_a, dtype),
            &quantize(&b_values, dtype),
            1,
            BENCH_M,
            BENCH_K,
            BENCH_N,
            false,
            false,
            0,
            0,
        );
        for arm in [
            BenchArm::Descriptor,
            BenchArm::Materialized,
            BenchArm::Control,
        ] {
            reset_einsum_execution_stats();
            onnx_runtime_ep_cuda::reset_movement_execution_stats();
            let fixture = bench_fixture(&ep, arm, dtype, &logical_a, &b_values);
            fixture.execute(&ep);
            let allocations = ep.runtime().allocation_counts();
            ep.runtime()
                .begin_graph_capture(&fixture.kernel_refs())
                .unwrap();
            fixture.execute(&ep);
            ep.runtime().end_graph_capture().unwrap();
            ep.runtime().replay_graph().unwrap();
            let bytes = fixture.output_bytes(&ep);
            let output = decode_floats(&bytes, dtype);
            oracle_metrics(&output, &expected, dtype);
            assert_eq!(ep.runtime().allocation_counts(), allocations);
            assert!(ep.runtime().reset_graph().unwrap());
            fixture.finish(&ep);
            let einsum = einsum_execution_stats();
            let movement = onnx_runtime_ep_cuda::movement_execution_stats();
            let route = if movement.capture_recordings != 0 {
                "explicit-materialization"
            } else if einsum.descriptor_transpose_gemm_launches != 0 {
                "descriptor-transpose"
            } else {
                "canonical"
            };
            assert_eq!(route, arm.expected_route());
            assert_eq!(einsum.claim_fallbacks, 0);
        }
    }
}

/// Host-locked, captured CUDA-event benchmark for the synthetic 256x512x384
/// transpose contraction. `scripts/bench_cuda_einsum.sh` is the only supported
/// entry point because it proves the lock, build, tree, and device mapping.
#[test]
#[ignore = "requires an idle pinned CUDA GPU and the host lock"]
fn einsum_captured_descriptor_benchmark() {
    let _lock = suite_lock();
    let replay_batch = std::env::var("EINSUM_BENCH_REPLAYS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2048);
    let batches = std::env::var("EINSUM_BENCH_BATCHES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    let ramp_seconds = std::env::var("EINSUM_BENCH_RAMP_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let expect_runnable = std::env::var("EINSUM_BENCH_EXPECT_RUNNABLE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let max_drift_percent = std::env::var("EINSUM_BENCH_MAX_DRIFT_PERCENT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5.0);
    let max_control_spread_percent = std::env::var("EINSUM_BENCH_MAX_CONTROL_SPREAD_PERCENT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5.0);
    assert!(batches >= 3, "benchmark evidence requires n >= 3 batches");
    assert!(
        ramp_seconds >= 8,
        "benchmark evidence requires at least 8 seconds of continuous warmup per arm"
    );
    hostlock_provenance("sweep-start", expect_runnable);
    let ep = require_cuda();
    gpu_identity(&ep);
    println!(
        "BENCH_SCOPE,synthetic_captured_contraction_only=true,model_or_end_to_end_claim=false,M={BENCH_M},K={BENCH_K},N={BENCH_N}"
    );
    println!(
        "BENCH_CONFIG,gpu={},dtypes=Float16|Float32,replay_batch={},batches={},abba=true,ramp_seconds={},expect_runnable={},max_drift_percent={:.3},max_control_spread_percent={:.3}",
        physical_gpu(),
        replay_batch,
        batches,
        ramp_seconds,
        expect_runnable,
        max_drift_percent,
        max_control_spread_percent,
    );
    println!(
        "BENCH_HEADER,dtype,batch,slot,arm,route,kernel_us,fixture_setup_us,warm_execute_us,plan_setup_us,capture_us,ramp_ms,captures,ramp_replays,timed_replays,captured_kernel_launches,gemm_launches,canonical_gemm_launches,descriptor_gemm_launches,transpose_launches,fallback_count,fallback_reason,workspace_bytes,persistent_metadata_bytes,materialization_bytes,alloc_before,free_before,alloc_after_warm,free_after_warm,alloc_after_capture,free_after_capture,alloc_after_timed,free_after_timed,alloc_after_finish,free_after_finish,clock_pre_mhz,clock_post_mhz,power_pre_w,power_post_w,oracle_max_abs,oracle_max_rel,output_digest"
    );

    let logical_a = values(BENCH_M * BENCH_K, 13);
    let b_values = values(BENCH_K * BENCH_N, 17);
    for dtype in [DataType::Float16, DataType::Float32] {
        let expected = f64_gemm_reference(
            &quantize(&logical_a, dtype),
            &quantize(&b_values, dtype),
            1,
            BENCH_M,
            BENCH_K,
            BENCH_N,
            false,
            false,
            0,
            0,
        );
        let logical_a_bytes = float_input(dtype, &[BENCH_M, BENCH_K], &logical_a).bytes;
        let b_bytes = float_input(dtype, &[BENCH_K, BENCH_N], &b_values).bytes;
        let math_digest = digest(&[logical_a_bytes, b_bytes].concat());
        println!(
            "ORACLE_CONFIG,dtype={dtype:?},kind=independent-f64-cpu,input_math_digest={math_digest:016x},output_elements={}",
            expected.len()
        );

        let mut results = Vec::new();
        for batch_index in 0..batches {
            let order = if batch_index % 2 == 0 {
                [
                    BenchArm::Descriptor,
                    BenchArm::Materialized,
                    BenchArm::Control,
                    BenchArm::Materialized,
                    BenchArm::Descriptor,
                ]
            } else {
                [
                    BenchArm::Materialized,
                    BenchArm::Descriptor,
                    BenchArm::Control,
                    BenchArm::Descriptor,
                    BenchArm::Materialized,
                ]
            };
            let block_start = results.len();
            for (slot, arm) in order.into_iter().enumerate() {
                let result = measure_bench_arm(
                    &ep,
                    arm,
                    dtype,
                    batch_index,
                    slot,
                    &logical_a,
                    &b_values,
                    &expected,
                    replay_batch,
                    ramp_seconds,
                    expect_runnable,
                );
                println!(
                    "ORACLE_PASS,dtype={dtype:?},batch={batch_index},slot={slot},arm={},math_digest={math_digest:016x},max_abs={:.8e},max_rel={:.8e},output_digest={:016x}",
                    arm.label(),
                    result.oracle_max_abs,
                    result.oracle_max_rel,
                    result.output_digest
                );
                print_bench_result(dtype, &result);
                results.push(result);
            }
            for (pair, (left, right)) in [(0, (0, 1)), (1, (3, 4))] {
                let pair_results = [&results[block_start + left], &results[block_start + right]];
                let descriptor = pair_results
                    .iter()
                    .find(|result| result.arm == BenchArm::Descriptor)
                    .unwrap();
                let materialized = pair_results
                    .iter()
                    .find(|result| result.arm == BenchArm::Materialized)
                    .unwrap();
                println!(
                    "PAIR_RATIO,dtype={dtype:?},batch={batch_index},pair={pair},descriptor_us={:.6},materialized_us={:.6},materialized_over_descriptor={:.6}",
                    descriptor.kernel_us,
                    materialized.kernel_us,
                    materialized.kernel_us / descriptor.kernel_us
                );
            }
        }

        let first_descriptor = results
            .iter()
            .find(|result| result.arm == BenchArm::Descriptor)
            .unwrap()
            .kernel_us;
        let drift = measure_bench_arm(
            &ep,
            BenchArm::Descriptor,
            dtype,
            batches,
            0,
            &logical_a,
            &b_values,
            &expected,
            replay_batch,
            ramp_seconds,
            expect_runnable,
        );
        print_bench_result(dtype, &drift);
        let drift_percent = (drift.kernel_us / first_descriptor - 1.0) * 100.0;
        println!(
            "BENCH_DRIFT,dtype={dtype:?},descriptor_first_us={first_descriptor:.6},descriptor_last_us={:.6},percent={drift_percent:.3}",
            drift.kernel_us
        );
        assert!(
            drift_percent.abs() <= max_drift_percent,
            "{dtype:?}: first/last descriptor drift {drift_percent:.3}% exceeds \
             {max_drift_percent:.3}%"
        );

        for arm in [
            BenchArm::Descriptor,
            BenchArm::Materialized,
            BenchArm::Control,
        ] {
            let values = results
                .iter()
                .filter(|result| result.arm == arm)
                .map(|result| result.kernel_us)
                .collect::<Vec<_>>();
            let (median, min, max) = median_range(&values);
            println!(
                "BENCH_SUMMARY,dtype={dtype:?},arm={},n={},median_us={median:.6},min_us={min:.6},max_us={max:.6}",
                arm.label(),
                values.len()
            );
        }
        let controls = results
            .iter()
            .filter(|result| result.arm == BenchArm::Control)
            .map(|result| result.kernel_us)
            .collect::<Vec<_>>();
        let (_, control_min, control_max) = median_range(&controls);
        let control_spread_percent = (control_max / control_min - 1.0) * 100.0;
        println!(
            "CONTROL_GATE,dtype={dtype:?},n={},min_us={control_min:.6},max_us={control_max:.6},spread_percent={control_spread_percent:.3}",
            controls.len()
        );
        assert!(
            control_spread_percent <= max_control_spread_percent,
            "{dtype:?}: unaffected control spread {control_spread_percent:.3}% exceeds \
             {max_control_spread_percent:.3}%"
        );

        let descriptor = results
            .iter()
            .filter(|result| result.arm == BenchArm::Descriptor)
            .map(|result| result.kernel_us)
            .collect::<Vec<_>>();
        let materialized = results
            .iter()
            .filter(|result| result.arm == BenchArm::Materialized)
            .map(|result| result.kernel_us)
            .collect::<Vec<_>>();
        let (descriptor_median, _, _) = median_range(&descriptor);
        let (materialized_median, _, _) = median_range(&materialized);
        println!(
            "BENCH_CONCLUSION,dtype={dtype:?},scope=captured-synthetic-256x512x384-only,descriptor_median_us={descriptor_median:.6},materialized_median_us={materialized_median:.6},materialized_over_descriptor={:.6}",
            materialized_median / descriptor_median
        );
    }
    hostlock_provenance("sweep-end", expect_runnable);
}
