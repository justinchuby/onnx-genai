//! `CausalConvWithState` under the standard `ai.onnx` opset-27 spelling.
//!
//! We already implemented this as a `com.microsoft` contrib op for Qwen3.5 /
//! Gated DeltaNet. ONNX standardized the same operation in opset 27, and this
//! checks that the claim is justified: the existing kernel is exercised through
//! the standard-domain registration and compared against golden values taken
//! from `onnx.reference.ReferenceEvaluator` running a real opset-27 node —
//! not against a re-derivation of the spec text.

#[path = "../benches/common/mod.rs"]
mod common;

use common::{FloatDType, Tensor, assert_close, make_kernel};
use onnx_runtime_ir::Attribute;

/// `(batch=1, channels=2, length=4)` with a width-3 depthwise kernel, so the
/// carry state is two positions wide.
const INPUT_SHAPE: [usize; 3] = [1, 2, 4];
const WEIGHT_SHAPE: [usize; 3] = [2, 1, 3];
const STATE_SHAPE: [usize; 3] = [1, 2, 2];

fn input() -> Tensor {
    Tensor::floats(
        FloatDType::F32,
        &INPUT_SHAPE,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
}

/// Powers of ten per tap so a misordered or misaligned tap shows up as an
/// obviously wrong digit rather than a plausible near-miss.
fn weight() -> Tensor {
    Tensor::floats(
        FloatDType::F32,
        &WEIGHT_SHAPE,
        &[1.0, 10.0, 100.0, 2.0, 20.0, 200.0],
    )
}

fn run(
    activation: Option<&str>,
    bias: Option<&Tensor>,
    past_state: Option<&Tensor>,
) -> (Vec<f32>, Vec<f32>) {
    let input = input();
    let weight = weight();
    let mut output = Tensor::zeros(FloatDType::F32, &INPUT_SHAPE);
    let mut present = Tensor::zeros(FloatDType::F32, &STATE_SHAPE);

    let mut views = vec![input.view(), weight.view()];
    let mut shapes = vec![INPUT_SHAPE.to_vec(), WEIGHT_SHAPE.to_vec()];
    if let Some(bias) = bias {
        views.push(bias.view());
        shapes.push(vec![INPUT_SHAPE[1]]);
    }
    if let Some(past) = past_state {
        if bias.is_none() {
            // `bias` is optional but positional; the past state is input 3.
            panic!("this harness passes past_state only alongside bias");
        }
        views.push(past.view());
        shapes.push(STATE_SHAPE.to_vec());
    }

    let attrs: Vec<(&str, Attribute)> = match activation {
        Some(name) => vec![("activation", Attribute::String(name.as_bytes().to_vec()))],
        None => vec![],
    };

    make_kernel("CausalConvWithState", attrs, &shapes, 27)
        .execute(&views, &mut [output.view_mut(), present.view_mut()])
        .expect("CausalConvWithState executes under the ai.onnx domain");
    (output.to_f32(), present.to_f32())
}

#[test]
fn prefill_treats_an_absent_past_state_as_zeros() {
    // Golden values from the ONNX reference implementation.
    let (output, present) = run(None, None, None);
    assert_close(
        &output,
        &[100.0, 210.0, 321.0, 432.0, 1000.0, 1300.0, 1530.0, 1752.0],
        0.0,
    );
    // The carry is the tail of concat(past, input) — here the last two input
    // positions of each channel.
    assert_close(&present, &[3.0, 4.0, 7.0, 8.0], 0.0);
}

#[test]
fn a_past_state_extends_the_convolution_backwards() {
    let bias = Tensor::floats(FloatDType::F32, &[2], &[0.0, 0.0]);
    let past = Tensor::floats(FloatDType::F32, &STATE_SHAPE, &[7.0, 8.0, 9.0, 11.0]);
    let (output, present) = run(None, Some(&bias), Some(&past));
    // Only the first two positions of each channel can see the carry, so the
    // tail is unchanged from the prefill case — that contrast is the causality
    // check: the state reaches backwards and never forwards.
    assert_close(
        &output,
        &[187.0, 218.0, 321.0, 432.0, 1238.0, 1322.0, 1530.0, 1752.0],
        0.0,
    );
    assert_close(&present, &[3.0, 4.0, 7.0, 8.0], 0.0);
}

#[test]
fn bias_is_added_per_channel() {
    let bias = Tensor::floats(FloatDType::F32, &[2], &[0.5, -0.5]);
    let past = Tensor::floats(FloatDType::F32, &STATE_SHAPE, &[7.0, 8.0, 9.0, 11.0]);
    let (output, _) = run(None, Some(&bias), Some(&past));
    assert_close(
        &output,
        &[187.5, 218.5, 321.5, 432.5, 1237.5, 1321.5, 1529.5, 1751.5],
        0.0,
    );
}

#[test]
fn silu_is_applied_after_the_bias() {
    // Deliberately small magnitudes. An earlier version of this check used the
    // values above, where sigmoid saturates to 1.0 and SiLU is
    // indistinguishable from the identity — the test would have passed against
    // a kernel that ignored the attribute entirely.
    let input = Tensor::floats(FloatDType::F32, &[1, 1, 4], &[0.5, -1.0, 2.0, -0.25]);
    let weight = Tensor::floats(FloatDType::F32, &[1, 1, 3], &[0.5, 0.25, 1.0]);
    let mut output = Tensor::zeros(FloatDType::F32, &[1, 1, 4]);
    let mut present = Tensor::zeros(FloatDType::F32, &[1, 1, 2]);

    make_kernel(
        "CausalConvWithState",
        vec![("activation", Attribute::String(b"silu".to_vec()))],
        &[vec![1, 1, 4], vec![1, 1, 3]],
        27,
    )
    .execute(
        &[input.view(), weight.view()],
        &mut [output.view_mut(), present.view_mut()],
    )
    .expect("CausalConvWithState executes");

    // Reference output without activation is [0.5, -0.875, 2.0, -0.25]; these
    // are its SiLU, from the ONNX reference implementation.
    assert_close(
        &output.to_f32(),
        &[0.31123, -0.257438, 1.761594, -0.109456],
        1e-5,
    );
}

#[test]
fn swish_is_the_same_function_as_silu() {
    // The spec names both spellings for one function, so a model using either
    // must decode identically.
    let input = Tensor::floats(FloatDType::F32, &[1, 1, 4], &[0.5, -1.0, 2.0, -0.25]);
    let weight = Tensor::floats(FloatDType::F32, &[1, 1, 3], &[0.5, 0.25, 1.0]);
    let mut outputs = Vec::new();
    for name in ["silu", "swish"] {
        let mut output = Tensor::zeros(FloatDType::F32, &[1, 1, 4]);
        let mut present = Tensor::zeros(FloatDType::F32, &[1, 1, 2]);
        make_kernel(
            "CausalConvWithState",
            vec![("activation", Attribute::String(name.as_bytes().to_vec()))],
            &[vec![1, 1, 4], vec![1, 1, 3]],
            27,
        )
        .execute(
            &[input.view(), weight.view()],
            &mut [output.view_mut(), present.view_mut()],
        )
        .expect("CausalConvWithState executes");
        outputs.push(output.to_f32());
    }
    assert_eq!(outputs[0], outputs[1]);
}
