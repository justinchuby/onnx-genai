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

use common::{Tensor, decode_floats, float_input, run_cuda};
use cudarc::driver::result::event;
use cudarc::driver::sys::CUevent_flags;
use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, KernelMatch, TensorMut, TensorView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{
    CudaExecutionProvider, build_cuda_registry_descriptors, cuda_supported_dtypes_for_op,
    einsum_execution_stats, reset_einsum_execution_stats,
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
        &[DataType::Float32, DataType::Float16]
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
        !cpu.supports_op(model.graph.node(node), 12, &shapes, &dtypes, &[])
            .is_supported(),
        "CPU/CUDA registration difference is deliberate in this CUDA-only change"
    );

    for (equation, shapes, expected_reason) in [
        (
            "ik,kj->ji",
            vec![static_shape([2, 3]), static_shape([3, 4])],
            "output permutation",
        ),
        (
            "...mk,...kn->...mn",
            vec![static_shape([2, 1, 3, 4]), static_shape([2, 5, 4, 6])],
            "partial multi-axis batch",
        ),
        (
            "ij->i",
            vec![static_shape([2, 3])],
            "reductions/elementwise",
        ),
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
        assert!(
            matches!(claim, KernelMatch::Unsupported { .. }),
            "{equation}"
        );
        assert!(
            claim.reason().unwrap().contains(expected_reason),
            "{equation}: {:?}",
            claim.reason()
        );
    }

    let strided = [
        TensorLayout::strided(vec![1, 2]),
        TensorLayout::contiguous(),
    ];
    let claim = ep.supports_op(model.graph.node(node), 12, &shapes, &dtypes, &strided);
    assert!(!claim.is_supported());
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
            .is_some_and(|reason| reason.contains("not an opset-12 Einsum type")),
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
    assert!(error.to_string().contains("must not alias"));
    assert_eq!(einsum_execution_stats().gemm_launches, before);
    runtime.synchronize().unwrap();
    for buffer in buffers {
        ep.deallocate(buffer).unwrap();
    }
    drop(inputs);
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
    materialized: bool,
    canonical: bool,
    dtype: DataType,
) -> BenchFixture {
    const M: usize = 256;
    const K: usize = 512;
    const N: usize = 384;
    let a_shape = if canonical { vec![M, K] } else { vec![K, M] };
    let a = float_input(dtype, &a_shape, &values(M * K, 13));
    let b = float_input(dtype, &[K, N], &values(K * N, 17));
    let output_shape = vec![M, N];
    let a_buffer = ep.allocate(a.bytes.len(), 256).unwrap();
    let b_buffer = ep.allocate(b.bytes.len(), 256).unwrap();
    let output = ep.allocate(dtype.storage_bytes(M * N), 256).unwrap();
    unsafe {
        ep.runtime()
            .htod(&a.bytes, cuptr(a_buffer.as_ptr()))
            .unwrap();
        ep.runtime()
            .htod(&b.bytes, cuptr(b_buffer.as_ptr()))
            .unwrap();
    }

    if materialized {
        let temporary = ep.allocate(dtype.storage_bytes(M * K), 256).unwrap();
        let transpose = bench_kernel(
            ep,
            "Transpose",
            std::slice::from_ref(&a),
            &[M, K],
            &[("perm", Attribute::Ints(vec![1, 0]))],
        );
        let temporary_tensor = float_input(dtype, &[M, K], &vec![0.0; M * K]);
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
    std::env::var("ONNX_GENAI_CUDA_PHYSICAL_DEVICE")
        .ok()
        .or_else(|| {
            std::env::var("CUDA_VISIBLE_DEVICES")
                .ok()
                .and_then(|visible| visible.split(',').next().map(str::to_owned))
        })
        .unwrap_or_else(|| "0".into())
}

fn gpu_state(label: &str, require_idle: bool) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "-i",
            &physical_gpu(),
            "--query-gpu=utilization.gpu,clocks.sm,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("nvidia-smi");
    assert!(output.status.success(), "nvidia-smi failed");
    let state = String::from_utf8(output.stdout).unwrap();
    println!("GPU_STATE,{label},{}", state.trim());
    if require_idle {
        let utilization = state
            .split(',')
            .next()
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();
        assert!(
            utilization <= 2,
            "GPU must be idle before every benchmark arm, observed {utilization}%"
        );
    }
}

struct BenchResult {
    label: &'static str,
    kernel_us: f32,
    setup_us: f64,
    plan_setup_us: f64,
    workspace_bytes: u64,
    launches: u64,
    materialization_bytes: u64,
}

fn measure_bench_arm(
    ep: &CudaExecutionProvider,
    label: &'static str,
    materialized: bool,
    canonical: bool,
    dtype: DataType,
    batch: usize,
    ramp_seconds: u64,
) -> BenchResult {
    gpu_state(label, true);
    reset_einsum_execution_stats();
    let runtime = ep.runtime();
    let setup = Instant::now();
    let fixture = bench_fixture(ep, materialized, canonical, dtype);
    fixture.execute(ep);
    assert!(
        fixture
            .kernel_refs()
            .iter()
            .all(|kernel| kernel.capture_support().is_supported()),
        "{label}: every kernel must be warmed and capturable"
    );
    let allocations = runtime.allocation_counts();
    runtime.begin_graph_capture(&fixture.kernel_refs()).unwrap();
    fixture.execute(ep);
    runtime.end_graph_capture().unwrap();
    let setup_us = setup.elapsed().as_secs_f64() * 1e6;
    assert_eq!(runtime.allocation_counts(), allocations);

    if ramp_seconds != 0 {
        let stop = Arc::new(AtomicBool::new(false));
        let witness_stop = stop.clone();
        let witness = std::thread::spawn(move || {
            while !witness_stop.load(Ordering::Relaxed) {
                gpu_state("ramp", false);
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        });
        let start = Instant::now();
        while start.elapsed().as_secs() < ramp_seconds {
            for _ in 0..64 {
                runtime.replay_graph().unwrap();
            }
            runtime.synchronize().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        witness.join().unwrap();
    }

    let start = event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
    let end = event::create(CUevent_flags::CU_EVENT_DEFAULT).unwrap();
    unsafe { event::record(start, runtime.stream_ptr()) }.unwrap();
    for _ in 0..batch {
        runtime.replay_graph().unwrap();
    }
    unsafe { event::record(end, runtime.stream_ptr()) }.unwrap();
    unsafe { event::synchronize(end) }.unwrap();
    let kernel_us = unsafe { event::elapsed(start, end) }.unwrap() * 1000.0 / batch as f32;
    unsafe {
        event::destroy(start).unwrap();
        event::destroy(end).unwrap();
    }
    assert_eq!(runtime.allocation_counts(), allocations);
    runtime.reset_graph().unwrap();
    let stats = einsum_execution_stats();
    fixture.finish(ep);
    BenchResult {
        label,
        kernel_us,
        setup_us,
        plan_setup_us: stats.setup_ns as f64 / 1000.0,
        workspace_bytes: stats.workspace_bytes,
        launches: if materialized { 2 } else { 1 },
        materialization_bytes: if materialized {
            (256 * 512 * dtype.byte_size()) as u64
        } else {
            0
        },
    }
}

/// Host-locked captured replay benchmark used for the PR evidence table.
///
/// Run only on an idle, pinned GPU:
/// `scripts/hostlock.sh run --owner batty --reason "CUDA Einsum captured sweep" -- \
///  cargo test ... einsum_captured_descriptor_benchmark -- --ignored --exact --nocapture`
#[test]
#[ignore = "requires an idle pinned CUDA GPU and the host lock"]
fn einsum_captured_descriptor_benchmark() {
    let _lock = suite_lock();
    let ep = require_cuda();
    let batch = std::env::var("EINSUM_BENCH_BATCH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2048);
    let repeats = std::env::var("EINSUM_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    let ramp_seconds = std::env::var("EINSUM_BENCH_RAMP_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let dtype = match std::env::var("EINSUM_BENCH_DTYPE")
        .unwrap_or_else(|_| "f16".into())
        .as_str()
    {
        "f16" => DataType::Float16,
        "f32" => DataType::Float32,
        other => panic!("EINSUM_BENCH_DTYPE must be f16 or f32, got {other}"),
    };
    assert!(repeats >= 3, "benchmark evidence requires n >= 3");
    println!(
        "BENCH_CONFIG,gpu={},dtype={dtype:?},M=256,K=512,N=384,batch={},repeats={},ramp_seconds={}",
        physical_gpu(),
        batch,
        repeats,
        ramp_seconds
    );
    println!(
        "BENCH_HEADER,label,rep,kernel_us,setup_us,plan_setup_us,workspace_bytes,launches,materialization_bytes"
    );

    let mut first_descriptor = None;
    for rep in 0..repeats {
        for (label, materialized, canonical) in [
            ("descriptor", false, false),
            ("materialized", true, false),
            ("control-a", false, true),
            ("control-b", false, true),
        ] {
            let result = measure_bench_arm(
                &ep,
                label,
                materialized,
                canonical,
                dtype,
                batch,
                if rep == 0 && label == "descriptor" {
                    ramp_seconds
                } else {
                    0
                },
            );
            if first_descriptor.is_none() && label == "descriptor" {
                first_descriptor = Some(result.kernel_us);
            }
            println!(
                "BENCH,{},{},{:.4},{:.1},{:.1},{},{},{}",
                result.label,
                rep,
                result.kernel_us,
                result.setup_us,
                result.plan_setup_us,
                result.workspace_bytes,
                result.launches,
                result.materialization_bytes
            );
        }
    }
    let last = measure_bench_arm(&ep, "descriptor-last", false, false, dtype, batch, 0);
    println!(
        "BENCH,{},{},{:.4},{:.1},{:.1},{},{},{}",
        last.label,
        repeats,
        last.kernel_us,
        last.setup_us,
        last.plan_setup_us,
        last.workspace_bytes,
        last.launches,
        last.materialization_bytes
    );
    let first = first_descriptor.unwrap();
    println!(
        "BENCH_DRIFT,descriptor_first_us={first:.4},descriptor_last_us={:.4},percent={:.3}",
        last.kernel_us,
        (last.kernel_us / first - 1.0) * 100.0
    );
}
