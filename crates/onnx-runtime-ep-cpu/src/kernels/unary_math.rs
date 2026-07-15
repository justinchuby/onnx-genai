//! Additional unary f32 elementwise math kernels (`docs/ORT2.md` §4.4).
//!
//! These mirror the [`elementwise`](super::elementwise) unary family (a straight
//! per-element `f32 -> f32` map through [`to_dense_f32`]/[`write_dense_f32`]) but
//! live in their own module to keep the shared `elementwise.rs` low-conflict
//! while the dtype-coverage pass proceeds there in parallel.
//!
//! Every op targets ONNX/NumPy numerics on `f32`:
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

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{check_arity, to_dense_f32, write_dense_f32};

/// The per-element operation for a unary math kernel.
#[derive(Clone, Copy)]
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

impl Kernel for UnaryMathKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(self.op.name(), inputs, outputs, 1, 1, 1)?;
        let x = to_dense_f32(&inputs[0])?;
        let y: Vec<f32> = x.iter().map(|&v| self.op.apply(v)).collect();
        write_dense_f32(&mut outputs[0], &y)
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
}
