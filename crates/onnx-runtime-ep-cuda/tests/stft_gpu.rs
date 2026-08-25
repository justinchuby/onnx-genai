mod common;

use common::{
    Tensor, absent_input, build_graph, decode_floats, float_input, input, require_cuda, run_cpu,
    run_cuda,
};
use onnx_runtime_ep_api::{ExecutionProvider, KernelMatch};
use onnx_runtime_ep_cuda::{CUDA_COVERED_OPS, cufft_plan_cache_stats, stft_last_execution_stats};
use onnx_runtime_ir::{Attribute, DataType, TensorLayout, static_shape};
use onnx_runtime_loader::Model;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

fn stft_inputs(
    signal: Tensor,
    frame_step: Tensor,
    window: Option<Tensor>,
    frame_length: Option<Tensor>,
) -> Vec<Tensor> {
    vec![
        signal,
        frame_step,
        window.unwrap_or_else(|| absent_input(DataType::Undefined)),
        frame_length.unwrap_or_else(|| absent_input(DataType::Undefined)),
    ]
}

fn stft(
    ep: &onnx_runtime_ep_cuda::CudaExecutionProvider,
    inputs: &[Tensor],
    output_shape: &[usize],
    attrs: &[(&str, Attribute)],
) -> Vec<f32> {
    decode_floats(
        &run_cuda(
            ep,
            "STFT",
            "",
            17,
            inputs,
            &[(DataType::Float32, output_shape.to_vec())],
            attrs,
        )[0],
        DataType::Float32,
    )
}

fn cpu_stft(inputs: &[Tensor], output_shape: &[usize], attrs: &[(&str, Attribute)]) -> Vec<f32> {
    decode_floats(
        &run_cpu(
            "STFT",
            "",
            17,
            inputs,
            &[(DataType::Float32, output_shape.to_vec())],
            attrs,
        )[0],
        DataType::Float32,
    )
}

fn sequence(length: usize) -> Vec<f32> {
    (0..length).map(|index| index as f32 + 1.0).collect()
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "non-string panic".to_string()
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn nontrivial_window_is_applied_and_matches_cpu() {
    let ep = require_cuda();
    let inputs = stft_inputs(
        float_input(DataType::Float32, &[1, 8, 1], &sequence(8)),
        input(DataType::Int64, &[], &[2_i64]),
        Some(float_input(DataType::Float32, &[4], &[0.25, 0.5, 1.5, 2.0])),
        None,
    );
    let attrs = [("onesided", Attribute::Int(0))];
    let actual = stft(&ep, &inputs, &[1, 3, 4, 2], &attrs);
    let expected = cpu_stft(&inputs, &[1, 3, 4, 2], &attrs);
    assert_close(&actual, &expected, 2e-5);
    assert_eq!(actual[0], 13.75, "first-frame DC must include the window");
    assert_ne!(actual[0], 10.0, "ignoring the window must be observable");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn overlapping_step_selects_the_middle_frame() {
    let ep = require_cuda();
    let inputs = stft_inputs(
        float_input(DataType::Float32, &[1, 8, 1], &sequence(8)),
        input(DataType::Int64, &[], &[2_i64]),
        None,
        Some(input(DataType::Int64, &[], &[4_i64])),
    );
    let attrs = [("onesided", Attribute::Int(0))];
    let actual = stft(&ep, &inputs, &[1, 3, 4, 2], &attrs);
    assert_eq!(actual[8], 18.0, "middle frame must start at sample 2");
    assert_ne!(
        actual[8], 26.0,
        "using frame_length as a non-overlapping step must fail"
    );
    assert_close(&actual, &cpu_stft(&inputs, &[1, 3, 4, 2], &attrs), 2e-5);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn final_complete_frame_is_not_dropped() {
    let ep = require_cuda();
    let inputs = stft_inputs(
        float_input(DataType::Float32, &[1, 8, 1], &sequence(8)),
        input(DataType::Int64, &[], &[2_i64]),
        None,
        Some(input(DataType::Int64, &[], &[4_i64])),
    );
    let attrs = [("onesided", Attribute::Int(0))];
    let actual = stft(&ep, &inputs, &[1, 3, 4, 2], &attrs);
    assert_eq!(actual.len(), 3 * 4 * 2);
    assert_eq!(
        actual[16], 26.0,
        "the final complete frame [5,6,7,8] must be transformed"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn complex_exact_frame_produces_full_spectrum() {
    let ep = require_cuda();
    let inputs = stft_inputs(
        float_input(
            DataType::Float32,
            &[1, 4, 2],
            &[1.0, 0.5, 2.0, -1.0, 0.0, 3.0, -2.0, 0.25],
        ),
        input(DataType::Int32, &[], &[4_i32]),
        None,
        Some(input(DataType::Int64, &[], &[4_i64])),
    );
    let attrs = [("onesided", Attribute::Int(0))];
    let actual = stft(&ep, &inputs, &[1, 1, 4, 2], &attrs);
    assert_close(&actual, &cpu_stft(&inputs, &[1, 1, 4, 2], &attrs), 2e-5);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn onesided_keeps_n_over_two_plus_one_and_matches_full_prefix() {
    let ep = require_cuda();
    let inputs = stft_inputs(
        float_input(
            DataType::Float32,
            &[1, 8, 1],
            &[0.5, 1.0, -0.25, 2.0, 0.0, -1.0, 0.75, 0.25],
        ),
        input(DataType::Int64, &[], &[8_i64]),
        None,
        Some(input(DataType::Int64, &[], &[8_i64])),
    );
    let one = stft(&ep, &inputs, &[1, 1, 5, 2], &[]);
    let full_attrs = [("onesided", Attribute::Int(0))];
    let full = stft(&ep, &inputs, &[1, 1, 8, 2], &full_attrs);
    assert_eq!(one.len(), 5 * 2, "N=8 requires N/2+1 bins");
    assert_close(&one, &full[..10], 2e-5);
    assert!(
        one[8].abs() > 1e-3,
        "the Nyquist bin must be present rather than returning N/2 bins"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn non_power_of_two_dynamic_lengths_and_batched_signals_match_cpu() {
    let ep = require_cuda();
    let values = (0..12)
        .map(|index| (index as f32 - 3.0) * 0.2)
        .collect::<Vec<_>>();
    let inputs = stft_inputs(
        float_input(DataType::Float32, &[2, 6, 1], &values),
        input(DataType::Int32, &[], &[1_i32]),
        None,
        Some(input(DataType::Int64, &[], &[3_i64])),
    );
    let actual = stft(&ep, &inputs, &[2, 4, 2, 2], &[]);
    let expected = cpu_stft(&inputs, &[2, 4, 2, 2], &[]);
    assert_close(&actual, &expected, 3e-5);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn zero_step_and_short_signal_fail_with_cpu_consistent_errors() {
    let ep = require_cuda();
    let zero_step_inputs = stft_inputs(
        float_input(DataType::Float32, &[1, 4, 1], &sequence(4)),
        input(DataType::Int64, &[], &[0_i64]),
        None,
        Some(input(DataType::Int64, &[], &[4_i64])),
    );
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stft(&ep, &zero_step_inputs, &[1, 1, 3, 2], &[])
    }))
    .expect_err("frame_step=0 must fail");
    assert!(panic_message(panic).contains("must be greater than zero"));

    let short_inputs = stft_inputs(
        float_input(DataType::Float32, &[1, 3, 1], &sequence(3)),
        input(DataType::Int64, &[], &[1_i64]),
        None,
        Some(input(DataType::Int64, &[], &[4_i64])),
    );
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stft(&ep, &short_inputs, &[1, 0, 3, 2], &[])
    }))
    .expect_err("a signal shorter than frame_length must fail");
    assert!(panic_message(panic).contains("complete unpadded frames"));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn claim_gates_decline_unsupported_dtype_layout_and_complex_onesided() {
    let ep = require_cuda();
    assert!(CUDA_COVERED_OPS.contains(&"STFT"));

    for dtype in [DataType::Float16, DataType::BFloat16, DataType::Float64] {
        let signal = Tensor {
            dtype,
            shape: vec![1, 4, 1],
            bytes: vec![0; dtype.storage_bytes(4)],
            absent: false,
        };
        let inputs = stft_inputs(
            signal,
            input(DataType::Int64, &[], &[2_i64]),
            None,
            Some(input(DataType::Int64, &[], &[4_i64])),
        );
        let (graph, id) = build_graph("STFT", "", 17, &inputs, &[(dtype, vec![1, 1, 3, 2])], &[]);
        let model = Model::new(&graph);
        let claim = ep.supports_op(
            model.graph.node(id),
            17,
            &[
                static_shape([1, 4, 1]),
                static_shape([]),
                static_shape([]),
                static_shape([]),
            ],
            &[dtype, DataType::Int64, DataType::Undefined, DataType::Int64],
            &[],
        );
        assert!(
            matches!(claim, KernelMatch::Unsupported { .. }),
            "{dtype:?} signal must decline at claim time"
        );
    }

    let complex_inputs = stft_inputs(
        float_input(DataType::Float32, &[1, 4, 2], &[0.0; 8]),
        input(DataType::Int64, &[], &[4_i64]),
        None,
        Some(input(DataType::Int64, &[], &[4_i64])),
    );
    let (graph, id) = build_graph(
        "STFT",
        "",
        17,
        &complex_inputs,
        &[(DataType::Float32, vec![1, 1, 3, 2])],
        &[],
    );
    let model = Model::new(&graph);
    assert!(matches!(
        ep.supports_op(
            model.graph.node(id),
            17,
            &[
                static_shape([1, 4, 2]),
                static_shape([]),
                static_shape([]),
                static_shape([]),
            ],
            &[
                DataType::Float32,
                DataType::Int64,
                DataType::Undefined,
                DataType::Int64,
            ],
            &[],
        ),
        KernelMatch::Unsupported { ref reason } if reason.contains("requires a real signal")
    ));

    let real_inputs = stft_inputs(
        float_input(DataType::Float32, &[1, 4, 1], &[0.0; 4]),
        input(DataType::Int64, &[], &[2_i64]),
        Some(float_input(DataType::Float32, &[4], &[1.0; 4])),
        None,
    );
    let (graph, id) = build_graph(
        "STFT",
        "",
        17,
        &real_inputs,
        &[(DataType::Float32, vec![1, 1, 3, 2])],
        &[],
    );
    let model = Model::new(&graph);
    assert!(matches!(
        ep.supports_op(
            model.graph.node(id),
            17,
            &[
                static_shape([1, 4, 1]),
                static_shape([]),
                static_shape([4]),
                static_shape([]),
            ],
            &[
                DataType::Float32,
                DataType::Int64,
                DataType::Float32,
                DataType::Undefined,
            ],
            &[
                TensorLayout::strided(vec![8, 2, 1]),
                TensorLayout::contiguous(),
                TensorLayout::contiguous(),
                TensorLayout::contiguous(),
            ],
        ),
        KernelMatch::Unsupported { ref reason } if reason.contains("contiguous")
    ));

    assert!(matches!(
        ep.supports_op(
            model.graph.node(id),
            17,
            &[
                static_shape([1, 4, 1]),
                static_shape([]),
                static_shape([4]),
                static_shape([]),
            ],
            &[
                DataType::Float32,
                DataType::Int64,
                DataType::Float32,
                DataType::Undefined,
            ],
            &[
                TensorLayout::contiguous(),
                TensorLayout::contiguous(),
                TensorLayout::strided(vec![2]),
                TensorLayout::contiguous(),
            ],
        ),
        KernelMatch::Unsupported { ref reason } if reason.contains("input 2")
    ));

    let zero_signal = stft_inputs(
        float_input(DataType::Float32, &[1, 0, 1], &[]),
        input(DataType::Int64, &[], &[1_i64]),
        None,
        Some(input(DataType::Int64, &[], &[1_i64])),
    );
    let (graph, id) = build_graph(
        "STFT",
        "",
        17,
        &zero_signal,
        &[(DataType::Float32, vec![1, 0, 1, 2])],
        &[],
    );
    let model = Model::new(&graph);
    assert!(matches!(
        ep.supports_op(
            model.graph.node(id),
            17,
            &[
                static_shape([1, 0, 1]),
                static_shape([]),
                static_shape([]),
                static_shape([]),
            ],
            &[
                DataType::Float32,
                DataType::Int64,
                DataType::Undefined,
                DataType::Int64,
            ],
            &[],
        ),
        KernelMatch::Unsupported { ref reason } if reason.contains("greater than zero")
    ));
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn shared_plan_cache_reuses_exact_key_and_capture_fails_closed() {
    let ep = require_cuda();
    let values = (0..32)
        .map(|index| (index as f32 - 8.0) * 0.125)
        .collect::<Vec<_>>();
    let inputs = stft_inputs(
        float_input(DataType::Float32, &[2, 16, 1], &values),
        input(DataType::Int64, &[], &[3_i64]),
        None,
        Some(input(DataType::Int64, &[], &[5_i64])),
    );
    let before = cufft_plan_cache_stats();
    let first = stft(&ep, &inputs, &[2, 4, 3, 2], &[]);
    let middle = cufft_plan_cache_stats();
    let second = stft(&ep, &inputs, &[2, 4, 3, 2], &[]);
    let after = cufft_plan_cache_stats();
    assert_close(&first, &second, 2e-5);
    assert_eq!(middle.creations - before.creations, 1);
    assert!(after.hits > middle.hits);

    let stats = stft_last_execution_stats();
    assert_eq!(stats.frames_per_signal, 4);
    assert_eq!(stats.fft_batch, 8);
    assert_eq!(stats.pack_unpack_launches, 2);
    assert_eq!(stats.cufft_executions, 1);
    assert!(stats.workspace_bytes >= 8 * 5 * 2 * 4);
    eprintln!(
        "STFT_METRICS frames={} fft_batch={} pack_unpack_launches={} cufft_executions={} \
         workspace_bytes={}",
        stats.frames_per_signal,
        stats.fft_batch,
        stats.pack_unpack_launches,
        stats.cufft_executions,
        stats.workspace_bytes
    );

    let smaller_inputs = stft_inputs(
        float_input(DataType::Float32, &[1, 16, 1], &values[..16]),
        input(DataType::Int64, &[], &[3_i64]),
        None,
        Some(input(DataType::Int64, &[], &[5_i64])),
    );
    let _ = stft(&ep, &smaller_inputs, &[1, 4, 3, 2], &[]);
    assert!(
        cufft_plan_cache_stats().creations > after.creations,
        "a different FFT batch must use a distinct plan-cache key"
    );

    let (graph, id) = build_graph(
        "STFT",
        "",
        17,
        &inputs,
        &[(DataType::Float32, vec![2, 4, 3, 2])],
        &[],
    );
    let model = Model::new(&graph);
    let kernel = ep
        .get_kernel(
            model.graph.node(id),
            &[vec![2, 16, 1], vec![], vec![], vec![]],
            17,
        )
        .unwrap();
    let capture = kernel.capture_support();
    assert!(!capture.is_supported());
    assert!(capture.reason().unwrap().contains("frame_step"));
}
