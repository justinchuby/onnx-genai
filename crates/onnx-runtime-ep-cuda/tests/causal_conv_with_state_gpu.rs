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
//! GPU parity for `com.microsoft::CausalConvWithState` (issue #67): the CUDA EP
//! depthwise causal short-conv with rolling state, checked byte/tol-exact
//! against the CPU EP oracle across fp32/fp16/bf16, decode (`L=1`) and prefill
//! (`L>1`), with/without `bias`, with/without `past_state`, and both `none` and
//! `silu` activations.

mod common;

use common::{assert_close, decode_floats, float_input, require_cuda, run_cpu, run_cuda};
use onnx_runtime_ir::{Attribute, DataType};

const OP: &str = "CausalConvWithState";
const DOMAIN: &str = "com.microsoft";
const OPSET: u64 = 1;

fn tolerance(dtype: DataType, silu: bool) -> f32 {
    match (dtype, silu) {
        (DataType::Float32, false) => 2e-5,
        (DataType::Float32, true) => 1e-4,
        (DataType::Float16, _) => 4e-3,
        (DataType::BFloat16, _) => 4e-2,
        _ => 0.0,
    }
}

#[allow(clippy::too_many_arguments)]
fn check(
    ep: &onnx_runtime_ep_cuda::CudaExecutionProvider,
    dtype: DataType,
    batch: usize,
    channels: usize,
    length: usize,
    kernel_size: usize,
    x: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    state: Option<&[f32]>,
    silu: bool,
    want_present: bool,
) {
    let pad = kernel_size - 1;
    let mut inputs = vec![
        float_input(dtype, &[batch, channels, length], x),
        float_input(dtype, &[channels, 1, kernel_size], weight),
    ];
    if let Some(bias) = bias {
        inputs.push(float_input(dtype, &[channels], bias));
    }
    if let Some(state) = state {
        assert!(
            bias.is_some(),
            "past_state is positional input 3; needs bias"
        );
        inputs.push(float_input(dtype, &[batch, channels, pad], state));
    }
    let mut outputs = vec![(dtype, vec![batch, channels, length])];
    if want_present {
        outputs.push((dtype, vec![batch, channels, pad]));
    }
    let attrs = vec![
        ("ndim", Attribute::Int(1)),
        (
            "activation",
            Attribute::String(if silu {
                b"silu".to_vec()
            } else {
                b"none".to_vec()
            }),
        ),
    ];
    let cuda = run_cuda(ep, OP, DOMAIN, OPSET, &inputs, &outputs, &attrs);
    let cpu = run_cpu(OP, DOMAIN, OPSET, &inputs, &outputs, &attrs);
    let tol = tolerance(dtype, silu);
    for (idx, (c, r)) in cuda.iter().zip(&cpu).enumerate() {
        assert_close(
            &format!("{OP}[out{idx}]"),
            dtype,
            &decode_floats(c, dtype),
            &decode_floats(r, dtype),
            tol,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn causal_conv_with_state_matches_cpu_across_configs() {
    let ep = require_cuda();
    // B=1, C=3, K=4. Weights per channel (depthwise [C,1,K]).
    let weight = [
        0.1f32, 0.2, -0.3, 0.5, // c0
        -0.4, 0.25, 0.6, -0.1, // c1
        0.7, -0.2, 0.15, 0.3, // c2
    ];
    let bias = [0.05f32, -0.1, 0.2];
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for silu in [false, true] {
            // Decode step: L=1, with past_state (K-1=3 frames per channel).
            let x1 = [1.0f32, -2.0, 0.5]; // [1,3,1]
            let state = [
                0.5f32, -0.5, 1.0, // c0 (3 = K-1 frames)
                0.2, 0.4, -0.6, // c1
                -1.0, 0.3, 0.8, // c2
            ];
            check(
                &ep,
                dtype,
                1,
                3,
                1,
                4,
                &x1,
                &weight,
                Some(&bias),
                Some(&state),
                silu,
                true,
            );

            // Prefill: L=4, no state (absent -> zeros), no present output.
            let x4 = [
                0.1f32, 0.2, 0.3, 0.4, // c0
                -0.5, 0.6, -0.7, 0.8, // c1
                0.9, -1.0, 1.1, -1.2, // c2
            ];
            check(
                &ep,
                dtype,
                1,
                3,
                4,
                4,
                &x4,
                &weight,
                Some(&bias),
                None,
                silu,
                false,
            );

            // Prefill with state + present, and no bias (bias absent).
            check(&ep, dtype, 1, 3, 4, 4, &x4, &weight, None, None, silu, true);
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn causal_conv_with_state_matches_cpu_for_batched_decode() {
    let ep = require_cuda();
    // B=2, C=2, K=3.
    let weight = [0.3f32, -0.6, 0.9, 0.2, 0.5, -0.4];
    let bias = [0.1f32, -0.2];
    let x = [1.0f32, -1.0, 0.25, 0.75]; // [2,2,1]
    let state = [
        0.5f32, -0.5, // b0 c0 (K-1=2)
        0.2, 0.4, // b0 c1
        -1.0, 0.3, // b1 c0
        0.8, -0.9, // b1 c1
    ];
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        for silu in [false, true] {
            check(
                &ep,
                dtype,
                2,
                2,
                1,
                3,
                &x,
                &weight,
                Some(&bias),
                Some(&state),
                silu,
                true,
            );
        }
    }
}

/// The standard `ai.onnx` opset-27 spelling reaches the same fused kernel.
///
/// ONNX standardized this op in opset 27 with a contract identical to the
/// `com.microsoft` one we already served, so it is registered against the same
/// factory rather than growing a second implementation to keep in step. This
/// checks the claim two ways: the standard-domain CUDA result matches the
/// standard-domain CPU oracle, and it matches the contrib-domain CUDA result
/// bit for bit — if the registration were missing, `run_cuda` would find no
/// kernel and the node would never reach this EP at all.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn the_standard_domain_spelling_reaches_the_same_kernel() {
    let ep = require_cuda();
    // Prefill with a carry state and a bias: the shape that exercises the
    // concatenation, the per-channel bias, and the emitted present state.
    let x: Vec<f32> = vec![0.5, -1.0, 2.0, -0.25, 1.5, 0.75, -0.5, 0.25];
    let weight: Vec<f32> = vec![0.5, 0.25, 1.0, -0.75, 0.5, 0.125];
    let bias: Vec<f32> = vec![0.25, -0.5];
    let state: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4];

    let inputs = vec![
        float_input(DataType::Float32, &[1, 2, 4], &x),
        float_input(DataType::Float32, &[2, 1, 3], &weight),
        float_input(DataType::Float32, &[2], &bias),
        float_input(DataType::Float32, &[1, 2, 2], &state),
    ];
    let outputs = vec![
        (DataType::Float32, vec![1, 2, 4]),
        (DataType::Float32, vec![1, 2, 2]),
    ];
    // The standard op defines only `activation`; it has no `ndim`.
    let standard_attrs = vec![("activation", Attribute::String(b"silu".to_vec()))];
    let contrib_attrs = vec![
        ("ndim", Attribute::Int(1)),
        ("activation", Attribute::String(b"silu".to_vec())),
    ];

    let standard_cuda = run_cuda(&ep, OP, "", 27, &inputs, &outputs, &standard_attrs);
    let standard_cpu = run_cpu(OP, "", 27, &inputs, &outputs, &standard_attrs);
    let contrib_cuda = run_cuda(&ep, OP, DOMAIN, OPSET, &inputs, &outputs, &contrib_attrs);

    for slot in 0..outputs.len() {
        let got = decode_floats(&standard_cuda[slot], DataType::Float32);
        let expected = decode_floats(&standard_cpu[slot], DataType::Float32);
        assert_close(
            &format!("standard-domain CUDA vs CPU, output {slot}"),
            DataType::Float32,
            &got,
            &expected,
            2e-5,
        );
        // Same kernel, same inputs: the two spellings must not merely be close.
        assert_eq!(
            standard_cuda[slot], contrib_cuda[slot],
            "the standard and contrib spellings must resolve to one kernel, output {slot}"
        );
    }
}
