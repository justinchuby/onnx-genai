mod common;

use std::sync::{Mutex, MutexGuard};

use common::{
    Tensor, build_graph, decode_floats, float_input, input, require_cuda, run_cpu, run_cuda,
};
use onnx_runtime_ep_api::{ExecutionProvider, KernelMatch};
use onnx_runtime_ep_cuda::{CUDA_COVERED_OPS, cufft_plan_cache_stats};
use onnx_runtime_ir::{Attribute, DataType, TensorLayout, static_shape};
use onnx_runtime_loader::Model;

static DFT_GPU_LOCK: Mutex<()> = Mutex::new(());

fn lock_dft_gpu() -> MutexGuard<'static, ()> {
    DFT_GPU_LOCK.lock().unwrap_or_else(|poisoned| {
        eprintln!(
            "WARNING: DFT_GPU_LOCK was poisoned by a prior test panic — recovering. \
             Investigate the original failure above."
        );
        poisoned.into_inner()
    })
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn dft(
    ep: &onnx_runtime_ep_cuda::CudaExecutionProvider,
    input_tensor: Tensor,
    output_shape: &[usize],
    attrs: &[(&str, Attribute)],
    optional: &[Tensor],
) -> Vec<f32> {
    dft_at(ep, 20, input_tensor, output_shape, attrs, optional)
}

fn dft_at(
    ep: &onnx_runtime_ep_cuda::CudaExecutionProvider,
    opset: u64,
    input_tensor: Tensor,
    output_shape: &[usize],
    attrs: &[(&str, Attribute)],
    optional: &[Tensor],
) -> Vec<f32> {
    let mut inputs = vec![input_tensor];
    inputs.extend_from_slice(optional);
    let outputs = [(DataType::Float32, output_shape.to_vec())];
    decode_floats(
        &run_cuda(ep, "DFT", "", opset, &inputs, &outputs, attrs)[0],
        DataType::Float32,
    )
}

fn cpu_dft(
    input_tensor: Tensor,
    output_shape: &[usize],
    attrs: &[(&str, Attribute)],
    optional: &[Tensor],
) -> Vec<f32> {
    cpu_dft_at(20, input_tensor, output_shape, attrs, optional)
}

fn cpu_dft_at(
    opset: u64,
    input_tensor: Tensor,
    output_shape: &[usize],
    attrs: &[(&str, Attribute)],
    optional: &[Tensor],
) -> Vec<f32> {
    let mut inputs = vec![input_tensor];
    inputs.extend_from_slice(optional);
    let outputs = [(DataType::Float32, output_shape.to_vec())];
    decode_floats(
        &run_cpu("DFT", "", opset, &inputs, &outputs, attrs)[0],
        DataType::Float32,
    )
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn opset_defaults_and_explicit_axes_match_cpu_on_rank4_input() {
    let ep = require_cuda();
    let tensor = float_input(DataType::Float32, &[1, 8, 6, 1], &[0.0; 48]);
    let onesided = [("onesided", Attribute::Int(1))];

    let opset17 = dft_at(&ep, 17, tensor.clone(), &[1, 5, 6, 2], &onesided, &[]);
    assert_close(
        &opset17,
        &cpu_dft_at(17, tensor.clone(), &[1, 5, 6, 2], &onesided, &[]),
        1e-5,
    );

    let opset20 = dft_at(&ep, 20, tensor.clone(), &[1, 8, 4, 2], &onesided, &[]);
    assert_close(
        &opset20,
        &cpu_dft_at(20, tensor.clone(), &[1, 8, 4, 2], &onesided, &[]),
        1e-5,
    );

    let axis_attr = [("onesided", Attribute::Int(1)), ("axis", Attribute::Int(2))];
    let explicit_attr = dft_at(&ep, 17, tensor.clone(), &[1, 8, 4, 2], &axis_attr, &[]);
    assert_close(
        &explicit_attr,
        &cpu_dft_at(17, tensor.clone(), &[1, 8, 4, 2], &axis_attr, &[]),
        1e-5,
    );

    let optional = [
        common::absent_input(DataType::Undefined),
        input(DataType::Int64, &[], &[1_i64]),
    ];
    let explicit_input = dft_at(&ep, 20, tensor.clone(), &[1, 5, 6, 2], &onesided, &optional);
    assert_close(
        &explicit_input,
        &cpu_dft_at(20, tensor, &[1, 5, 6, 2], &onesided, &optional),
        1e-5,
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn forward_complex_n4_pins_sign_convention() {
    let _suite_lock = lock_dft_gpu();
    let ep = require_cuda();
    let input = float_input(
        DataType::Float32,
        &[1, 4, 2],
        &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
    );
    let actual = dft(&ep, input.clone(), &[1, 4, 2], &[], &[]);
    let expected = [10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
    assert_close(&actual, &expected, 1e-5);
    assert_close(&actual, &cpu_dft(input, &[1, 4, 2], &[], &[]), 1e-5);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn inverse_complex_n4_applies_exactly_one_over_n() {
    let _suite_lock = lock_dft_gpu();
    let ep = require_cuda();
    let spectrum = float_input(
        DataType::Float32,
        &[1, 4, 2],
        &[10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0],
    );
    let attrs = [("inverse", Attribute::Int(1))];
    let actual = dft(&ep, spectrum.clone(), &[1, 4, 2], &attrs, &[]);
    assert_close(&actual, &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0], 1e-5);
    assert_close(&actual, &cpu_dft(spectrum, &[1, 4, 2], &attrs, &[]), 1e-5);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn real_full_has_conjugate_tail_and_onesided_has_n_over_two_plus_one() {
    let _suite_lock = lock_dft_gpu();
    let ep = require_cuda();
    let input = float_input(DataType::Float32, &[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]);
    let full = dft(&ep, input.clone(), &[1, 4, 2], &[], &[]);
    assert_close(&full, &[10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0], 1e-5);
    assert_close(&full[2..4], &[full[6], -full[7]], 1e-5);

    let attrs = [("onesided", Attribute::Int(1))];
    let half = dft(&ep, input.clone(), &[1, 3, 2], &attrs, &[]);
    assert_eq!(
        half.len(),
        3 * 2,
        "N=4 onesided output must contain N/2+1 bins"
    );
    assert_close(&half, &full[..6], 1e-5);
    assert_close(&half, &cpu_dft(input, &[1, 3, 2], &attrs, &[]), 1e-5);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn dft_length_truncates_and_zero_pads() {
    let _suite_lock = lock_dft_gpu();
    let ep = require_cuda();
    let input_tensor = float_input(DataType::Float32, &[1, 4, 1], &[1.0, -2.0, 0.5, 3.0]);
    for length in [2_i64, 7] {
        let length_input = input(DataType::Int64, &[], &[length]);
        let output_shape = [1, length as usize, 2];
        let actual = dft(
            &ep,
            input_tensor.clone(),
            &output_shape,
            &[],
            std::slice::from_ref(&length_input),
        );
        let expected = cpu_dft(
            input_tensor.clone(),
            &output_shape,
            &[],
            std::slice::from_ref(&length_input),
        );
        assert_close(&actual, &expected, 2e-5);
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn batched_non_default_axis_matches_cpu() {
    let _suite_lock = lock_dft_gpu();
    let ep = require_cuda();
    let values = (0..2 * 3 * 4)
        .map(|index| (index as f32 - 7.0) * 0.25)
        .collect::<Vec<_>>();
    let input_tensor = float_input(DataType::Float32, &[2, 3, 4, 1], &values);
    let absent_length = common::absent_input(DataType::Undefined);
    let axis = input(DataType::Int64, &[], &[1_i64]);
    let optional = [absent_length, axis];
    let actual = dft(&ep, input_tensor.clone(), &[2, 3, 4, 2], &[], &optional);
    let expected = cpu_dft(input_tensor, &[2, 3, 4, 2], &[], &optional);
    assert_close(&actual, &expected, 2e-5);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn claim_gates_decline_unsupported_dtype_layout_and_complex_onesided() {
    let _suite_lock = lock_dft_gpu();
    let ep = require_cuda();
    assert!(CUDA_COVERED_OPS.contains(&"DFT"));

    for dtype in [DataType::Float16, DataType::BFloat16, DataType::Float64] {
        let tensor = Tensor {
            dtype,
            shape: vec![1, 4, 1],
            bytes: vec![0; dtype.storage_bytes(4)],
            absent: false,
        };
        let (graph, id) = build_graph("DFT", "", 20, &[tensor], &[(dtype, vec![1, 4, 2])], &[]);
        let model = Model::new(&graph);
        let claim = ep.supports_op(
            model.graph.node(id),
            20,
            &[static_shape([1, 4, 1])],
            &[dtype],
            &[TensorLayout::contiguous()],
        );
        assert!(
            matches!(claim, KernelMatch::Unsupported { .. }),
            "{dtype:?} must decline at claim time"
        );
    }

    let complex = float_input(DataType::Float32, &[1, 4, 2], &[0.0; 8]);
    let attrs = [("onesided", Attribute::Int(1))];
    let (graph, id) = build_graph(
        "DFT",
        "",
        20,
        &[complex],
        &[(DataType::Float32, vec![1, 3, 2])],
        &attrs,
    );
    let model = Model::new(&graph);
    assert!(matches!(
        ep.supports_op(
            model.graph.node(id),
            20,
            &[static_shape([1, 4, 2])],
            &[DataType::Float32],
            &[TensorLayout::contiguous()],
        ),
        KernelMatch::Unsupported { ref reason } if reason.contains("only for real input")
    ));

    let real = float_input(DataType::Float32, &[1, 4, 1], &[0.0; 4]);
    let (graph, id) = build_graph(
        "DFT",
        "",
        20,
        &[real],
        &[(DataType::Float32, vec![1, 4, 2])],
        &[],
    );
    let model = Model::new(&graph);
    assert!(matches!(
        ep.supports_op(
            model.graph.node(id),
            20,
            &[static_shape([1, 4, 1])],
            &[DataType::Float32],
            &[TensorLayout::strided(vec![8, 2, 1])],
        ),
        KernelMatch::Unsupported { ref reason } if reason.contains("contiguous")
    ));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn repeated_geometry_reuses_bounded_cufft_plan_and_declares_capture_unsupported() {
    let _suite_lock = lock_dft_gpu();
    let ep = require_cuda();
    let values = (0..13)
        .map(|index| index as f32 * 0.125)
        .collect::<Vec<_>>();
    let tensor = float_input(DataType::Float32, &[1, 13, 1], &values);
    let before = cufft_plan_cache_stats();
    let first = dft(&ep, tensor.clone(), &[1, 13, 2], &[], &[]);
    let middle = cufft_plan_cache_stats();
    let second = dft(&ep, tensor, &[1, 13, 2], &[], &[]);
    let after = cufft_plan_cache_stats();
    assert_close(&first, &second, 1e-5);
    assert_eq!(
        middle.creations - before.creations,
        1,
        "first unseen geometry must create one cuFFT plan"
    );
    assert!(
        after.hits > middle.hits,
        "second invocation must reuse the cached cuFFT plan"
    );
    for length in 2..=18usize {
        let values = (0..length)
            .map(|index| index as f32 / length as f32)
            .collect::<Vec<_>>();
        let tensor = float_input(DataType::Float32, &[1, length, 1], &values);
        let _ = dft(&ep, tensor, &[1, length, 2], &[], &[]);
    }
    let after_pressure = cufft_plan_cache_stats();
    assert!(
        after_pressure.evictions > after.evictions,
        "more than 16 unique geometries must evict an old cuFFT plan"
    );

    let input_tensor = float_input(DataType::Float32, &[1, 13, 1], &values);
    let (graph, id) = build_graph(
        "DFT",
        "",
        20,
        &[input_tensor],
        &[(DataType::Float32, vec![1, 13, 2])],
        &[],
    );
    let model = Model::new(&graph);
    let kernel = ep
        .get_kernel(model.graph.node(id), &[vec![1, 13, 1]], 20)
        .unwrap();
    let capture = kernel.capture_support();
    assert!(!capture.is_supported());
    assert!(capture.reason().unwrap().contains("plan selection"));
}
