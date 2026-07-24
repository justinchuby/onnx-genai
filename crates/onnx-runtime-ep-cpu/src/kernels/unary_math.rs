//! Additional unary floating-point elementwise math kernels (`docs/ORT2.md` §4.4).
//!
//! These mirror the [`elementwise`](super::elementwise) unary family (a straight
//! per-element map through [`to_dense_f32_widen`]/[`write_dense_f32_narrow`]) but
//! live in their own module to keep the shared `elementwise.rs` low-conflict
//! while the dtype-coverage pass proceeds there in parallel.
//!
//! Every op widens `f16`/`bf16` to `f32`, computes using ONNX/NumPy numerics,
//! and narrows back to the requested output float type:
//!
//! * `Abs`, `Neg`, `Reciprocal` — trivial arithmetic.
//! * `Exp`, `Log`, `Sin`, `Cos` — `std` intrinsics.
//! * `Floor`, `Ceil` — `std` intrinsics.
//! * `Round` — ONNX "round half to even" (banker's rounding), not Rust's
//!   round-half-away-from-zero `f32::round`; uses [`f32::round_ties_even`].
//! * `Sign` — `-1`/`0`/`+1` with `sign(0) = 0` and `sign(NaN) = NaN`
//!   (Rust's `f32::signum` returns `±1` for zero, so it cannot be used here).
//! * `Sigmoid` — logistic `1/(1+e^-x)`, evaluated in the numerically stable
//!   branch form so large-magnitude inputs neither overflow nor underflow.
//! * `Softplus` — `log(1 + e^x)`, evaluated as `max(x,0) + log1p(e^-|x|)` for
//!   the same stability reason.
//! * `Softsign` — `x / (1 + |x|)`.

use crate::dtype::{to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow};
use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::check_arity;

/// The per-element operation for a unary math kernel.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MathOp {
    Abs,
    Neg,
    Reciprocal,
    Exp,
    Log,
    Sign,
    Floor,
    Ceil,
    Round,
    Sin,
    Cos,
    Sigmoid,
    Softplus,
    Softsign,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Cosh,
    Sinh,
    Tan,
}

impl MathOp {
    fn name(self) -> &'static str {
        match self {
            MathOp::Abs => "Abs",
            MathOp::Neg => "Neg",
            MathOp::Reciprocal => "Reciprocal",
            MathOp::Exp => "Exp",
            MathOp::Log => "Log",
            MathOp::Sign => "Sign",
            MathOp::Floor => "Floor",
            MathOp::Ceil => "Ceil",
            MathOp::Round => "Round",
            MathOp::Sin => "Sin",
            MathOp::Cos => "Cos",
            MathOp::Sigmoid => "Sigmoid",
            MathOp::Softplus => "Softplus",
            MathOp::Softsign => "Softsign",
            MathOp::Acos => "Acos",
            MathOp::Acosh => "Acosh",
            MathOp::Asin => "Asin",
            MathOp::Asinh => "Asinh",
            MathOp::Atan => "Atan",
            MathOp::Atanh => "Atanh",
            MathOp::Cosh => "Cosh",
            MathOp::Sinh => "Sinh",
            MathOp::Tan => "Tan",
        }
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            MathOp::Abs => x.abs(),
            MathOp::Neg => -x,
            MathOp::Reciprocal => 1.0 / x,
            MathOp::Exp => x.exp(),
            MathOp::Log => x.ln(),
            MathOp::Sign => sign(x),
            MathOp::Floor => x.floor(),
            MathOp::Ceil => x.ceil(),
            // ONNX Round is round-half-to-even, unlike Rust's `round` which
            // rounds half away from zero.
            MathOp::Round => x.round_ties_even(),
            MathOp::Sin => x.sin(),
            MathOp::Cos => x.cos(),
            MathOp::Sigmoid => sigmoid(x),
            MathOp::Softplus => softplus(x),
            MathOp::Softsign => x / (1.0 + x.abs()),
            MathOp::Acos => x.acos(),
            MathOp::Acosh => x.acosh(),
            MathOp::Asin => x.asin(),
            MathOp::Asinh => x.asinh(),
            MathOp::Atan => x.atan(),
            MathOp::Atanh => x.atanh(),
            MathOp::Cosh => x.cosh(),
            MathOp::Sinh => x.sinh(),
            MathOp::Tan => x.tan(),
        }
    }
}

/// ONNX `Sign`: `-1`/`0`/`+1`, with `sign(0) = 0` and `sign(NaN) = NaN`.
fn sign(x: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Numerically stable logistic sigmoid `1/(1+e^-x)`.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable `Softplus`: `log(1 + e^x) = max(x,0) + log1p(e^-|x|)`.
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// A stateless unary math kernel.
pub struct UnaryMathKernel {
    op: MathOp,
}

macro_rules! math_factory {
    ($factory:ident, $variant:expr) => {
        /// Factory (no attributes).
        pub struct $factory;
        impl KernelFactory for $factory {
            fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                Ok(Box::new(UnaryMathKernel { op: $variant }))
            }
        }
    };
}

math_factory!(AbsFactory, MathOp::Abs);
math_factory!(NegFactory, MathOp::Neg);
math_factory!(ReciprocalFactory, MathOp::Reciprocal);
math_factory!(ExpFactory, MathOp::Exp);
math_factory!(LogFactory, MathOp::Log);
math_factory!(SignFactory, MathOp::Sign);
math_factory!(FloorFactory, MathOp::Floor);
math_factory!(CeilFactory, MathOp::Ceil);
math_factory!(RoundFactory, MathOp::Round);
math_factory!(SinFactory, MathOp::Sin);
math_factory!(CosFactory, MathOp::Cos);
math_factory!(SigmoidFactory, MathOp::Sigmoid);
math_factory!(SoftplusFactory, MathOp::Softplus);
math_factory!(SoftsignFactory, MathOp::Softsign);
math_factory!(AcosFactory, MathOp::Acos);
math_factory!(AcoshFactory, MathOp::Acosh);
math_factory!(AsinFactory, MathOp::Asin);
math_factory!(AsinhFactory, MathOp::Asinh);
math_factory!(AtanFactory, MathOp::Atan);
math_factory!(AtanhFactory, MathOp::Atanh);
math_factory!(CoshFactory, MathOp::Cosh);
math_factory!(SinhFactory, MathOp::Sinh);
math_factory!(TanFactory, MathOp::Tan);

impl UnaryMathKernel {
    fn execute_f32(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let x = to_dense_f32_widen(self.op.name(), &inputs[0])?;
        let y: Vec<f32> = x.iter().map(|&v| self.op.apply(v)).collect();
        write_dense_f32_narrow(self.op.name(), &mut outputs[0], &y)
    }
}

impl Kernel for UnaryMathKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(self.op.name(), inputs, outputs, 1, 1, 1)?;
        if self.op == MathOp::Neg {
            return match inputs[0].dtype {
                DataType::Int32 => {
                    let values = to_dense::<i32>(&inputs[0])?
                        .into_iter()
                        .map(i32::wrapping_neg)
                        .collect::<Vec<_>>();
                    write_dense::<i32>(&mut outputs[0], &values)
                }
                DataType::Int64 => {
                    let values = to_dense::<i64>(&inputs[0])?
                        .into_iter()
                        .map(i64::wrapping_neg)
                        .collect::<Vec<_>>();
                    write_dense::<i64>(&mut outputs[0], &values)
                }
                _ => self.execute_f32(inputs, outputs),
            };
        }
        self.execute_f32(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn run(op: MathOp, x: &Owned, out: &mut Owned) {
        UnaryMathKernel { op }
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
    }

    #[test]
    fn abs_neg_reciprocal() {
        let x = Owned::f32(&[4], &[-2., 0., 3., -4.]);
        let mut out = Owned::zeros_f32(&[4]);
        run(MathOp::Abs, &x, &mut out);
        assert_eq!(out.to_f32(), vec![2., 0., 3., 4.]);
        run(MathOp::Neg, &x, &mut out);
        assert_eq!(out.to_f32(), vec![2., -0., -3., 4.]);
        let x2 = Owned::f32(&[3], &[2., 4., -0.5]);
        let mut o3 = Owned::zeros_f32(&[3]);
        run(MathOp::Reciprocal, &x2, &mut o3);
        assert_eq!(o3.to_f32(), vec![0.5, 0.25, -2.0]);
    }

    #[test]
    fn neg_supports_signed_integer_tensors() {
        let i32_input = Owned::i32(&[3], &[i32::MIN, -7, 3]);
        let mut i32_output = Owned::zeros(DataType::Int32, &[3]);
        run(MathOp::Neg, &i32_input, &mut i32_output);
        assert_eq!(
            i32_output
                .bytes
                .chunks_exact(4)
                .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![i32::MIN, 7, -3]
        );

        let i64_input = Owned::i64(&[3], &[i64::MIN, -7, 3]);
        let mut i64_output = Owned::zeros(DataType::Int64, &[3]);
        run(MathOp::Neg, &i64_input, &mut i64_output);
        assert_eq!(
            i64_output
                .bytes
                .chunks_exact(8)
                .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>(),
            vec![i64::MIN, 7, -3]
        );
    }

    #[test]
    fn exp_log_roundtrip() {
        let x = Owned::f32(&[3], &[0., 1., 2.]);
        let mut out = Owned::zeros_f32(&[3]);
        run(MathOp::Exp, &x, &mut out);
        let r = out.to_f32();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - std::f32::consts::E).abs() < 1e-5);
        let lx = Owned::f32(&[2], &[1.0, std::f32::consts::E]);
        let mut lo = Owned::zeros_f32(&[2]);
        run(MathOp::Log, &lx, &mut lo);
        let lr = lo.to_f32();
        assert!(lr[0].abs() < 1e-6);
        assert!((lr[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn floor_ceil_round_ties_even() {
        let x = Owned::f32(&[6], &[1.4, 1.6, -1.4, 0.5, 1.5, 2.5]);
        let mut out = Owned::zeros_f32(&[6]);
        run(MathOp::Floor, &x, &mut out);
        assert_eq!(out.to_f32(), vec![1., 1., -2., 0., 1., 2.]);
        run(MathOp::Ceil, &x, &mut out);
        assert_eq!(out.to_f32(), vec![2., 2., -1., 1., 2., 3.]);
        run(MathOp::Round, &x, &mut out);
        // 0.5 -> 0, 1.5 -> 2, 2.5 -> 2 (round half to even).
        assert_eq!(out.to_f32(), vec![1., 2., -1., 0., 2., 2.]);
    }

    #[test]
    fn sign_zero_and_nan() {
        let x = Owned::f32(&[5], &[-3., -0.0, 0.0, 5., f32::NAN]);
        let mut out = Owned::zeros_f32(&[5]);
        run(MathOp::Sign, &x, &mut out);
        let r = out.to_f32();
        assert_eq!(r[0], -1.0);
        assert_eq!(r[1], 0.0);
        assert_eq!(r[2], 0.0);
        assert_eq!(r[3], 1.0);
        assert!(r[4].is_nan());
    }

    #[test]
    fn sin_cos_known_values() {
        let x = Owned::f32(&[2], &[0.0, std::f32::consts::FRAC_PI_2]);
        let mut out = Owned::zeros_f32(&[2]);
        run(MathOp::Sin, &x, &mut out);
        let r = out.to_f32();
        assert!(r[0].abs() < 1e-6);
        assert!((r[1] - 1.0).abs() < 1e-6);
        run(MathOp::Cos, &x, &mut out);
        let c = out.to_f32();
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!(c[1].abs() < 1e-6);
    }

    #[test]
    fn sigmoid_symmetry_and_extremes() {
        let x = Owned::f32(&[3], &[0.0, 100.0, -100.0]);
        let mut out = Owned::zeros_f32(&[3]);
        run(MathOp::Sigmoid, &x, &mut out);
        let r = out.to_f32();
        assert!((r[0] - 0.5).abs() < 1e-6);
        assert!((r[1] - 1.0).abs() < 1e-6);
        assert!(r[2].abs() < 1e-6);
    }

    #[test]
    fn softplus_matches_reference() {
        let x = Owned::f32(&[3], &[0.0, 1.0, -1.0]);
        let mut out = Owned::zeros_f32(&[3]);
        run(MathOp::Softplus, &x, &mut out);
        let r = out.to_f32();
        // log(2), log(1+e), log(1+1/e)
        assert!((r[0] - std::f32::consts::LN_2).abs() < 1e-6);
        assert!((r[1] - (1.0f32 + 1.0f32.exp()).ln()).abs() < 1e-6);
        assert!((r[2] - (1.0f32 + (-1.0f32).exp()).ln()).abs() < 1e-6);
    }

    #[test]
    fn softsign_matches_reference_and_bounds() {
        let x = Owned::f32(&[5], &[-100.0, -1.0, 0.0, 1.0, 100.0]);
        let mut out = Owned::zeros_f32(&[5]);
        run(MathOp::Softsign, &x, &mut out);
        let actual = out.to_f32();
        let expected = [-100.0 / 101.0, -0.5, 0.0, 0.5, 100.0 / 101.0];
        for (got, want) in actual.into_iter().zip(expected) {
            assert!((got - want).abs() < 1e-6);
            assert!(got > -1.0 && got < 1.0);
        }
    }
    #[test]
    fn unary_math_bf16_matches_widened_f32_reference() {
        let x = Owned::bf16(&[4], &[-1., 0., 1., 80.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[4]);
        run(MathOp::Softplus, &x, &mut out);
        // Golden BF16 values generated independently from log(1 + exp(x)) in
        // f64, then rounded to nearest-even BF16.
        assert_eq!(out.to_u16_bits(), vec![0x3ea0, 0x3f31, 0x3fa8, 0x42a0]);
    }

    #[test]
    fn every_unary_math_op_bf16_handles_special_values() {
        const NAN: u16 = 0x7fc0;
        // Inputs are [-inf, -0, +0, +inf, NaN]. Expected values below follow
        // each ONNX operation's definition, not MathOp::apply.
        let cases: &[(MathOp, [u16; 5])] = &[
            (MathOp::Abs, [0x7f80, 0, 0, 0x7f80, NAN]),
            (MathOp::Neg, [0x7f80, 0, 0x8000, 0xff80, NAN]),
            (MathOp::Reciprocal, [0x8000, 0xff80, 0x7f80, 0, NAN]),
            (MathOp::Exp, [0, 0x3f80, 0x3f80, 0x7f80, NAN]),
            (MathOp::Log, [NAN, 0xff80, 0xff80, 0x7f80, NAN]),
            (MathOp::Sign, [0xbf80, 0, 0, 0x3f80, NAN]),
            (MathOp::Floor, [0xff80, 0x8000, 0, 0x7f80, NAN]),
            (MathOp::Ceil, [0xff80, 0x8000, 0, 0x7f80, NAN]),
            (MathOp::Round, [0xff80, 0x8000, 0, 0x7f80, NAN]),
            (MathOp::Sin, [NAN, 0x8000, 0, NAN, NAN]),
            (MathOp::Cos, [NAN, 0x3f80, 0x3f80, NAN, NAN]),
            (MathOp::Sigmoid, [0, 0x3f00, 0x3f00, 0x3f80, NAN]),
            (MathOp::Softplus, [0, 0x3f31, 0x3f31, 0x7f80, NAN]),
            (MathOp::Softsign, [NAN, 0x8000, 0, NAN, NAN]),
            (MathOp::Acos, [NAN, 0x3fc9, 0x3fc9, NAN, NAN]),
            (MathOp::Acosh, [NAN, NAN, NAN, 0x7f80, NAN]),
            (MathOp::Asin, [NAN, 0x8000, 0, NAN, NAN]),
            (MathOp::Asinh, [0xff80, 0x8000, 0, 0x7f80, NAN]),
            (MathOp::Atan, [0xbfc9, 0x8000, 0, 0x3fc9, NAN]),
            (MathOp::Atanh, [NAN, 0x8000, 0, NAN, NAN]),
            (MathOp::Cosh, [0x7f80, 0x3f80, 0x3f80, 0x7f80, NAN]),
            (MathOp::Sinh, [0xff80, 0x8000, 0, 0x7f80, NAN]),
            (MathOp::Tan, [NAN, 0x8000, 0, NAN, NAN]),
        ];
        let values = [f32::NEG_INFINITY, -0.0, 0.0, f32::INFINITY, f32::NAN];

        for &(op, expected) in cases {
            let input = Owned::bf16(&[values.len()], &values);
            let mut output = Owned::zeros(DataType::BFloat16, &[values.len()]);
            run(op, &input, &mut output);
            for (got, expected) in output.to_u16_bits().into_iter().zip(expected) {
                if expected == NAN {
                    assert!(
                        got & 0x7f80 == 0x7f80 && got & 0x007f != 0,
                        "{}: expected NaN, got 0x{got:04x}",
                        op.name()
                    );
                } else {
                    assert_eq!(got, expected, "{}: expected 0x{expected:04x}", op.name());
                }
            }
        }
    }
}
