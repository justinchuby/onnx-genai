//! Arithmetic-independent checks of the actual production projection stages.
//! Decoding uses the separately tested canonical f32 decoder, widened to f64.

use super::*;
use crate::kernels::moe::{Activation, MoeActivationAttributes, numerical_trace::Projection};
use crate::kernels::testutil::Owned;

const ACTIVATION_ULP_BUDGET: u32 = 8;

type ActivationSample = (u32, usize, usize, usize, (f64, f64), f32, f64);

/// Fixture-specific: all experts receive every row, supplied route weights
/// sum to one, and expert contributions are accumulated in ascending order.
pub(super) struct Oracle {
    input: Vec<f32>,
    router: Vec<f32>,
    weights: [Option<Vec<Vec<f64>>>; 3],
    biases: [Option<Vec<f32>>; 3],
    layouts: [ProjectionLayout; 3],
    attributes: MoeActivationAttributes,
    rows: usize,
}
impl Oracle {
    pub(super) fn new(
        input: &Owned,
        router: &Owned,
        weights: [Option<&Owned>; 3],
        biases: [Option<&Owned>; 3],
        layouts: [ProjectionLayout; 3],
        attributes: MoeActivationAttributes,
    ) -> Self {
        let weights = std::array::from_fn(|slot| {
            weights[slot].map(|bank| {
                let layout = layouts[slot];
                (0..layout.experts)
                    .map(|expert| {
                        let bytes = &bank.bytes[layout.expert_byte_range(expert).unwrap()];
                        dequantize_expert_slice(
                            layout.format,
                            bytes,
                            layout.out_features,
                            layout.in_features,
                        )
                        .unwrap()
                        .into_iter()
                        .map(widen_normal)
                        .collect()
                    })
                    .collect()
            })
        });
        Self {
            rows: input.view().numel() / layouts[0].in_features,
            input: input.to_f32(),
            router: router.to_f32(),
            weights,
            biases: biases.map(|bias| bias.map(Owned::to_f32)),
            layouts,
            attributes,
        }
    }

    pub(super) fn check(&self, label: &str, records: &[Projection], result: &Owned) {
        let experts = self.layouts[0].experts;
        let stages = if self.weights[2].is_some() { 3 } else { 2 };
        assert_eq!(
            records.len(),
            experts * stages,
            "{label}: all expert projections must be observed"
        );
        let hidden = self.layouts[0].in_features;
        let inter = self.layouts[1].in_features;
        let mut scatter = vec![0.0f32; self.rows * hidden];
        let mut worst_ratio = 0.0f64;
        let mut worst_activation: Option<ActivationSample> = None;
        for (expert, events) in records.chunks_exact(stages).enumerate() {
            let first = &events[0];
            let last = &events[stages - 1];
            assert_eq!(first.name, "fc1");
            assert_eq!(last.name, "fc2");
            assert_eq!(
                first.input, self.input,
                "{label}: expert {expert} gather order"
            );
            if stages == 3 {
                assert_eq!(events[1].name, "fc3");
                assert_eq!(events[1].input, self.input);
            }
            for (event, slot) in events.iter().zip(if stages == 3 {
                vec![0, 2, 1]
            } else {
                vec![0, 1]
            }) {
                let layout = self.layouts[slot];
                assert_eq!(event.rows, self.rows);
                assert_eq!(event.input.len(), self.rows * layout.in_features);
                assert_eq!(event.output.len(), self.rows * layout.out_features);
                let bias = self.biases[slot]
                    .as_ref()
                    .map(|b| &b[expert * layout.out_features..(expert + 1) * layout.out_features]);
                assert_eq!(
                    event.bias.as_deref(),
                    bias,
                    "{label}: expert {expert} {} bias slice",
                    event.name
                );
                let weights = &self.weights[slot].as_ref().unwrap()[expert];
                for row in 0..self.rows {
                    for feature in 0..layout.out_features {
                        let x =
                            &event.input[row * layout.in_features..(row + 1) * layout.in_features];
                        let w = &weights
                            [feature * layout.in_features..(feature + 1) * layout.in_features];
                        let (want, budget, scale) =
                            projection_reference(x, w, bias.map_or(0., |b| f64::from(b[feature])));
                        let got = f64::from(event.output[row * layout.out_features + feature]);
                        let error = (got - want).abs();
                        worst_ratio = worst_ratio.max(error / budget.max(f64::MIN_POSITIVE));
                        assert!(
                            got.is_finite() && error <= budget,
                            "{label}: expert {expert} {} row {row} feature {feature}: {got} vs f64 {want}; error={error}, sum_abs={scale}, budget={budget}",
                            event.name
                        );
                    }
                }
            }
            for row in 0..self.rows {
                for feature in 0..inter {
                    let separate = stages == 3;
                    let first_width = self.layouts[0].out_features;
                    let first_at = |i| f64::from(first.output[row * first_width + i]);
                    let gated = self.attributes.activation == Activation::Swiglu
                        || (self.attributes.activation == Activation::Silu && separate);
                    let (want, operands) = if gated {
                        let (g, l) = if separate {
                            (
                                first_at(feature),
                                f64::from(events[1].output[row * inter + feature]),
                            )
                        } else if self.attributes.swiglu_fusion == 1 {
                            (first_at(2 * feature), first_at(2 * feature + 1))
                        } else {
                            (first_at(feature), first_at(inter + feature))
                        };
                        let limit = f64::from(self.attributes.swiglu_limit);
                        let g = g.min(limit);
                        let l = l.clamp(-limit, limit);
                        (
                            g * sigmoid(f64::from(self.attributes.activation_alpha) * g)
                                * (l + f64::from(self.attributes.activation_beta)),
                            (g, l),
                        )
                    } else {
                        let x = first_at(feature);
                        (
                            match self.attributes.activation {
                                Activation::Identity => x,
                                Activation::Relu => x.max(0.),
                                Activation::Gelu => {
                                    0.5 * x
                                        * (1.
                                            + ((2. / std::f64::consts::PI).sqrt()
                                                * (x + 0.044715 * x * x * x))
                                                .tanh())
                                }
                                Activation::Silu => x * sigmoid(x),
                                Activation::Swiglu => unreachable!(),
                            },
                            (x, 0.),
                        )
                    };
                    let got = last.input[row * inter + feature];
                    let distance = ulps(got, want as f32);
                    if worst_activation
                        .as_ref()
                        .is_none_or(|(old, ..)| distance > *old)
                    {
                        worst_activation =
                            Some((distance, expert, row, feature, operands, got, want));
                    }
                    if matches!(
                        self.attributes.activation,
                        Activation::Identity | Activation::Relu | Activation::Gelu
                    ) {
                        assert_eq!(
                            got.to_bits(),
                            (want as f32).to_bits(),
                            "{label}: exact activation expert {expert} row {row} feature {feature}"
                        );
                    } else {
                        // Fixture-local scalar activation contract, separate from
                        // GEMM's roundoff bound. No global libm guarantee is inferred.
                        let target = want as f32;
                        assert!(
                            got.is_finite() && ulps(got, target) <= ACTIVATION_ULP_BUDGET,
                            "{label}: activation expert {expert} row {row} feature {feature}: {got} vs f64 {want} ({} ULP)",
                            ulps(got, target)
                        );
                    }
                }
            }
            // This fixture selects every expert and supplies already-normalized
            // routing weights. Reconstruct the actual expert-order f32 scatter.
            for row in 0..self.rows {
                assert_eq!(
                    self.router[row * experts..(row + 1) * experts]
                        .iter()
                        .sum::<f32>(),
                    1.
                );
                let weight = self.router[row * experts + expert];
                for feature in 0..hidden {
                    scatter[row * hidden + feature] += weight * last.output[row * hidden + feature];
                }
            }
        }
        let got = result.to_f32();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            scatter.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{label}: exact expert-order routing scatter"
        );
        if self.attributes.activation == Activation::Identity {
            self.check_affine_composite(label, &got);
        }
        eprintln!(
            "[{label}] all FC1/FC3/activation/FC2/scatter checks passed; max GEMM error/budget={worst_ratio:.6}; worst activation (ULP, expert, row, feature, operands, actual, f64)={worst_activation:?}"
        );
    }
    /// The identity case permits propagation of the standard linear error
    /// envelope through both projections and routing, without assumptions about
    /// nonlinear sensitivities. This directly covers the cancellation regression.
    fn check_affine_composite(&self, label: &str, result: &[f32]) {
        let [first, last, _] = self.layouts;
        let mut worst = (0.0f64, 0usize, 0.0f64, Reference::default());
        for row in 0..self.rows {
            let expert_values = (0..first.experts)
                .map(|expert| {
                    let first_weights = &self.weights[0].as_ref().unwrap()[expert];
                    let first_values = (0..first.out_features)
                        .map(|feature| {
                            affine_reference(
                                self.input[row * first.in_features..(row + 1) * first.in_features]
                                    .iter()
                                    .map(|&x| Reference {
                                        value: widen_normal(x),
                                        error: 0.,
                                    }),
                                &first_weights[feature * first.in_features
                                    ..(feature + 1) * first.in_features],
                                f64::from(
                                    self.biases[0].as_ref().unwrap()
                                        [expert * first.out_features + feature],
                                ),
                            )
                            .0
                        })
                        .collect::<Vec<_>>();
                    let last_weights = &self.weights[1].as_ref().unwrap()[expert];
                    (0..last.out_features)
                        .map(|feature| {
                            affine_reference(
                                first_values.iter().copied(),
                                &last_weights
                                    [feature * last.in_features..(feature + 1) * last.in_features],
                                f64::from(
                                    self.biases[1].as_ref().unwrap()
                                        [expert * last.out_features + feature],
                                ),
                            )
                            .0
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let weights = self.router[row * first.experts..(row + 1) * first.experts]
                .iter()
                .copied()
                .map(widen_normal)
                .collect::<Vec<_>>();
            for feature in 0..last.out_features {
                let (reference, _) = affine_reference(
                    expert_values.iter().map(|values| values[feature]),
                    &weights,
                    0.,
                );
                let index = row * last.out_features + feature;
                let actual = f64::from(result[index]);
                let error = (actual - reference.value).abs();
                assert!(
                    error <= reference.error,
                    "{label}: complete affine composite row {row} feature {feature}: {actual} vs f64 {}, error {error} exceeds {}",
                    reference.value,
                    reference.error
                );
                let ratio = error / reference.error.max(f64::MIN_POSITIVE);
                if ratio > worst.0 {
                    worst = (ratio, index, actual, reference);
                }
            }
        }
        eprintln!(
            "[{label}] complete affine composite (error/budget, index, actual, f64+budget): {worst:?}"
        );
    }

    /// Deliberate corruptions must fail at the intended checker, not an
    /// unrelated shape assertion. FC2 is recomputed after corrupting activation
    /// so the activation check cannot hide behind a downstream GEMM failure.
    pub(super) fn check_negative_controls(&self, records: &[Projection], result: &Owned) {
        let mut bad = records.to_vec();
        let width = self.layouts[0].in_features;
        let weights = &self.weights[0].as_ref().unwrap()[0][..width];
        let bias = self.biases[0].as_ref().unwrap()[0];
        let (_, budget, _) = projection_reference(&bad[0].input[..width], weights, f64::from(bias));
        bad[0].output[0] = (f64::from(bad[0].output[0]) + 4. * budget + 1.) as f32;
        rejects(
            || self.check("corrupt projection", &bad, result),
            "fc1 row 0 feature 0",
        );

        let mut bad = records.to_vec();
        bad[0].input.rotate_left(width);
        rejects(
            || self.check("corrupt gather", &bad, result),
            "gather order",
        );
        let mut bad = records.to_vec();
        let bias = &mut bad[0].bias.as_mut().unwrap()[0];
        *bias = bias.next_up();
        if !bias.is_normal() {
            *bias = 2. * f32::MIN_POSITIVE;
        }
        rejects(|| self.check("corrupt bias", &bad, result), "bias slice");

        let mut bad = records.to_vec();
        let last = if self.weights[2].is_some() { 2 } else { 1 };
        for _ in 0..2 * ACTIVATION_ULP_BUDGET + 1 {
            bad[last].input[0] = bad[last].input[0].next_up();
        }
        if !bad[last].input[0].is_normal() {
            bad[last].input[0] = if bad[last].input[0].is_sign_negative() {
                -2. * f32::MIN_POSITIVE
            } else {
                2. * f32::MIN_POSITIVE
            };
        }
        let layout = self.layouts[1];
        let weights = &self.weights[1].as_ref().unwrap()[0];
        for feature in 0..layout.out_features {
            let (value, _, _) = projection_reference(
                &bad[last].input[..layout.in_features],
                &weights[feature * layout.in_features..(feature + 1) * layout.in_features],
                f64::from(bad[last].bias.as_ref().unwrap()[feature]),
            );
            bad[last].output[feature] = value as f32;
        }
        rejects(
            || self.check("corrupt activation", &bad, result),
            "activation expert 0 row 0 feature 0",
        );

        let mut bad = result.to_f32();
        bad[0] = bad[0].next_up();
        let bad = Owned::f32(result.view().shape, &bad);
        rejects(
            || self.check("corrupt scatter", records, &bad),
            "exact expert-order routing scatter",
        );
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        let e = x.exp();
        e / (1. + e)
    }
}
fn ulps(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    let ordered = |x: f32| {
        let bits = x.to_bits();
        if bits >> 31 != 0 {
            !bits
        } else {
            bits | 0x80000000
        }
    };
    ordered(a).abs_diff(ordered(b))
}

#[derive(Clone, Copy, Debug, Default)]
struct Reference {
    value: f64,
    error: f64,
}

/// Standard dot-product forward-error scale: gamma_(2K+1) * sum_abs.
/// Counting separate multiply/add plus bias covers fused and unfused orders.
/// Propagated input uncertainty, f64 reference rounding and an absolute
/// underflow allowance are accounted separately from that scale. This fixture
/// checks normal/zero operands explicitly; it does not certify arbitrary DAZ
/// modes, where flushing an input can be amplified by its weight.
fn affine_reference(
    x: impl ExactSizeIterator<Item = Reference>,
    w: &[f64],
    bias: f64,
) -> (Reference, f64) {
    assert_eq!(x.len(), w.len());
    let operations = (2 * x.len() + 1) as f64;
    let mut value = bias;
    let mut scale = bias.abs();
    let mut propagated = 0.;
    for (x, &w) in x.zip(w) {
        assert!(
            x.value == 0. || x.value.abs() >= f64::from(f32::MIN_POSITIVE),
            "fixture reference requires normal/zero operands"
        );
        assert!(
            w == 0. || w.abs() >= f64::from(f32::MIN_POSITIVE),
            "fixture reference requires normal/zero operands"
        );
        let term = x.value * w;
        assert!(term.is_finite());
        value += term;
        scale += term.abs();
        propagated += x.error * w.abs();
    }
    assert!(
        scale.is_finite() && scale + propagated < f64::from(f32::MAX),
        "fixture dot products must not overflow f32"
    );
    let gamma = |u: f64| {
        let n = operations * u;
        assert!(n < 1.);
        n / (1. - n)
    };
    let error = propagated
        + gamma(f64::from(f32::EPSILON) / 2.) * (scale + propagated)
        + gamma(f64::EPSILON / 2.) * scale
        + operations * f64::from(f32::MIN_POSITIVE);
    (Reference { value, error }, scale)
}

fn widen_normal(value: f32) -> f64 {
    assert!(
        value.to_bits() & 0x7fff_ffff == 0 || value.is_normal(),
        "fixture reference requires normal/zero operands"
    );
    f64::from(value)
}

fn projection_reference(x: &[f32], w: &[f64], bias: f64) -> (f64, f64, f64) {
    let (reference, scale) = affine_reference(
        x.iter().map(|&x| Reference {
            value: widen_normal(x),
            error: 0.,
        }),
        w,
        bias,
    );
    (reference.value, reference.error, scale)
}

#[test]
fn cancellation_budget_has_independent_scale_and_rejects_wrong_values() {
    let (want, budget, scale) = projection_reference(&[1.; 4], &[10000., 1., -10000., -1.], 0.);
    assert_eq!(want, 0.);
    assert_eq!(scale, 20002.);
    assert!(
        2e-5 < budget,
        "a small residual does not make the dot product well-conditioned"
    );
    assert!(0.1 > budget, "wrong data/bias must still fail");
    assert_eq!(ulps(1., f32::from_bits(1f32.to_bits() + 9)), 9);
}

fn rejects(check: impl FnOnce() + std::panic::UnwindSafe, reason: &str) {
    let failure = std::panic::catch_unwind(check)
        .expect_err("the numerical checker must reject deliberate corruption");
    let message = failure
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| failure.downcast_ref::<&str>().copied())
        .expect("assertion message");
    assert!(
        message.contains(reason),
        "expected refusal at {reason:?}, got {message}"
    );
}

#[test]
fn composed_projection_carries_input_uncertainty() {
    let (reference, _) = affine_reference(
        [
            Reference {
                value: 1.,
                error: 0.01,
            },
            Reference {
                value: 1.,
                error: 0.01,
            },
        ]
        .into_iter(),
        &[100., -100.],
        0.,
    );
    assert_eq!(reference.value, 0.);
    assert!(reference.error >= 2.);
    assert!((100. * 1.01 - 100. * 0.99 - reference.value).abs() <= reference.error);
    assert!(
        4. > reference.error,
        "propagated bounds must still reject an excessive error"
    );
}

#[test]
fn subnormal_operands_are_outside_the_fixture_roundoff_model() {
    rejects(
        || {
            projection_reference(&[f32::from_bits(1)], &[1e30], 0.);
        },
        "normal/zero operands",
    );
}
