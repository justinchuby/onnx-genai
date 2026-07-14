//! Elementwise f32 kernels (`docs/ORT2.md` §4.4).
//!
//! Two tight families share this module because they share the same dense-f32
//! read/write plumbing:
//!
//! * **Binary broadcasting** — `Sub`, `Mul`, `Div`, `Pow`, and the variadic
//!   `Min`. Each reuses [`broadcast_apply`](super::add::broadcast_apply) (the
//!   numpy right-alignment / size-1 broadcast machinery Add already defines) so
//!   broadcasting semantics stay identical across every binary op.
//! * **Unary** — `Sqrt`, `Erf`, `Tanh`: a straight per-element map.
//!
//! Numerics target ONNX/NumPy exactly. `Erf` has no `std` intrinsic, so it uses
//! the pure-Rust `libm::erf` (the correctly-rounded Sun/FreeBSD algorithm),
//! keeping the crate FFI-free (libm is pure Rust, no `cc`) while matching the
//! ONNX reference to < 1 ulp near zero.

use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::add::broadcast_apply;
use super::{check_arity, to_dense_f32, write_dense_f32};
use crate::strided::numel;

/// The combining operation for a binary elementwise kernel.
#[derive(Clone, Copy)]
enum BinOp {
    Sub,
    Mul,
    Div,
    Pow,
    /// Variadic minimum (ONNX `Min` accepts 1..N inputs).
    Min,
    /// Variadic maximum (ONNX `Max` accepts 1..N inputs).
    Max,
}

impl BinOp {
    fn name(self) -> &'static str {
        match self {
            BinOp::Sub => "Sub",
            BinOp::Mul => "Mul",
            BinOp::Div => "Div",
            BinOp::Pow => "Pow",
            BinOp::Min => "Min",
            BinOp::Max => "Max",
        }
    }

    /// Fold `acc` (accumulated left value) with a new operand `v`.
    fn apply(self, acc: f32, v: f32) -> f32 {
        match self {
            BinOp::Sub => acc - v,
            BinOp::Mul => acc * v,
            BinOp::Div => acc / v,
            BinOp::Pow => acc.powf(v),
            // ONNX Min/Max propagate NaN (numpy `minimum`/`maximum` semantics).
            // Rust's `f32::min`/`f32::max` SUPPRESS NaN (return the non-NaN
            // operand), so guard explicitly before delegating.
            BinOp::Min => {
                if acc.is_nan() || v.is_nan() {
                    f32::NAN
                } else {
                    acc.min(v)
                }
            }
            BinOp::Max => {
                if acc.is_nan() || v.is_nan() {
                    f32::NAN
                } else {
                    acc.max(v)
                }
            }
        }
    }
}

/// A stateless binary/variadic elementwise kernel.
pub struct BinaryKernel {
    op: BinOp,
}

macro_rules! binary_factory {
    ($factory:ident, $variant:expr) => {
        /// Factory (no attributes).
        pub struct $factory;
        impl KernelFactory for $factory {
            fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                Ok(Box::new(BinaryKernel { op: $variant }))
            }
        }
    };
}

binary_factory!(SubFactory, BinOp::Sub);
binary_factory!(MulFactory, BinOp::Mul);
binary_factory!(DivFactory, BinOp::Div);
binary_factory!(PowFactory, BinOp::Pow);
binary_factory!(MinFactory, BinOp::Min);
binary_factory!(MaxFactory, BinOp::Max);

impl Kernel for BinaryKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Min/Max are variadic (1..N); the rest are strictly binary.
        let (min_in, max_in) = match self.op {
            BinOp::Min | BinOp::Max => (1, usize::MAX),
            _ => (2, 2),
        };
        check_arity(self.op.name(), inputs, outputs, min_in, max_in, 1)?;

        let out_shape = outputs[0].shape.to_vec();
        let n = numel(&out_shape);
        let mut out = vec![0.0f32; n];

        // Seed the accumulator from the first operand (broadcast to the output).
        let first = to_dense_f32(&inputs[0])?;
        broadcast_apply(&first, inputs[0].shape, &out_shape, |i, v| out[i] = v)?;

        // Fold each remaining operand with the op's combiner.
        for input in &inputs[1..] {
            let rhs = to_dense_f32(input)?;
            let op = self.op;
            broadcast_apply(&rhs, input.shape, &out_shape, |i, v| {
                out[i] = op.apply(out[i], v)
            })?;
        }

        write_dense_f32(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// The per-element operation for a unary elementwise kernel.
#[derive(Clone, Copy)]
enum UnOp {
    Sqrt,
    Erf,
    Tanh,
}

impl UnOp {
    fn name(self) -> &'static str {
        match self {
            UnOp::Sqrt => "Sqrt",
            UnOp::Erf => "Erf",
            UnOp::Tanh => "Tanh",
        }
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            UnOp::Sqrt => x.sqrt(),
            UnOp::Erf => erf(x as f64) as f32,
            UnOp::Tanh => x.tanh(),
        }
    }
}

/// A stateless unary elementwise kernel.
pub struct UnaryKernel {
    op: UnOp,
}

macro_rules! unary_factory {
    ($factory:ident, $variant:expr) => {
        /// Factory (no attributes).
        pub struct $factory;
        impl KernelFactory for $factory {
            fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                Ok(Box::new(UnaryKernel { op: $variant }))
            }
        }
    };
}

unary_factory!(SqrtFactory, UnOp::Sqrt);
unary_factory!(ErfFactory, UnOp::Erf);
unary_factory!(TanhFactory, UnOp::Tanh);

impl Kernel for UnaryKernel {
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

/// Gauss error function. Delegates to the pure-Rust `libm::erf`, which is the
/// correctly-rounded (< 1 ulp) Sun/FreeBSD algorithm — the same one the C
/// standard library and ONNX reference runtimes use. An earlier polynomial
/// (Abramowitz & Stegun 7.1.26) was ~1e-9 off near zero and failed the upstream
/// conformance suite's tight (`atol=0`) tolerance. NaN propagates.
///
/// Shared with the fused `Gelu` kernel (`kernels::gelu`) so both the standalone
/// `Erf` op and the fused GELU compute bit-identical error-function values.
pub(crate) fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    libm::erf(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn run_bin(f: BinOp, a: &Owned, b: &Owned, out: &mut Owned) {
        BinaryKernel { op: f }
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
    }

    #[test]
    fn sub_same_shape() {
        let a = Owned::f32(&[2, 2], &[10., 20., 30., 40.]);
        let b = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run_bin(BinOp::Sub, &a, &b, &mut out);
        assert_eq!(out.to_f32(), vec![9., 18., 27., 36.]);
    }

    #[test]
    fn mul_broadcasts_2d_with_2d() {
        // [3,1] * [1,4] -> [3,4] full outer product.
        let a = Owned::f32(&[3, 1], &[1., 2., 3.]);
        let b = Owned::f32(&[1, 4], &[10., 20., 30., 40.]);
        let mut out = Owned::zeros_f32(&[3, 4]);
        run_bin(BinOp::Mul, &a, &b, &mut out);
        assert_eq!(
            out.to_f32(),
            vec![
                10., 20., 30., 40., // 1 * row
                20., 40., 60., 80., // 2 * row
                30., 60., 90., 120., // 3 * row
            ]
        );
    }

    #[test]
    fn div_broadcasts_scalar() {
        let a = Owned::f32(&[2, 2], &[2., 4., 6., 8.]);
        let b = Owned::f32(&[], &[2.]); // scalar
        let mut out = Owned::zeros_f32(&[2, 2]);
        run_bin(BinOp::Div, &a, &b, &mut out);
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn div_by_zero_is_inf_and_nan() {
        let a = Owned::f32(&[2], &[1., 0.]);
        let b = Owned::f32(&[2], &[0., 0.]);
        let mut out = Owned::zeros_f32(&[2]);
        run_bin(BinOp::Div, &a, &b, &mut out);
        let r = out.to_f32();
        assert!(r[0].is_infinite() && r[0] > 0.0);
        assert!(r[1].is_nan());
    }

    #[test]
    fn pow_square() {
        let a = Owned::f32(&[3], &[2., 3., 4.]);
        let b = Owned::f32(&[], &[2.]);
        let mut out = Owned::zeros_f32(&[3]);
        run_bin(BinOp::Pow, &a, &b, &mut out);
        assert_eq!(out.to_f32(), vec![4., 9., 16.]);
    }

    #[test]
    fn min_variadic_three_inputs_with_broadcast() {
        let a = Owned::f32(&[2, 2], &[5., 1., 8., 2.]);
        let b = Owned::f32(&[2, 2], &[3., 3., 3., 3.]);
        let c = Owned::f32(&[1], &[4.]); // broadcast scalar-ish
        let mut out = Owned::zeros_f32(&[2, 2]);
        BinaryKernel { op: BinOp::Min }
            .execute(
                &[a.view(), b.view(), c.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        // min(a,3,4) elementwise: min(5,3,4)=3, min(1,3,4)=1, min(8,3,4)=3, min(2,3,4)=2
        assert_eq!(out.to_f32(), vec![3., 1., 3., 2.]);
    }

    #[test]
    fn min_propagates_nan() {
        // ONNX Min propagates NaN (numpy semantics) — unlike Rust's f32::min
        // which would return the non-NaN operand. NaN in ANY position wins.
        let a = Owned::f32(&[3], &[f32::NAN, 2.0, 5.0]);
        let b = Owned::f32(&[3], &[1.0, f32::NAN, 3.0]);
        let mut out = Owned::zeros_f32(&[3]);
        run_bin(BinOp::Min, &a, &b, &mut out);
        let r = out.to_f32();
        assert!(r[0].is_nan(), "NaN in lhs must propagate");
        assert!(r[1].is_nan(), "NaN in rhs must propagate");
        assert_eq!(r[2], 3.0);
    }

    #[test]
    fn max_propagates_nan_and_reduces() {
        // Max mirrors Min: elementwise maximum, NaN-propagating, variadic.
        let a = Owned::f32(&[3], &[f32::NAN, 2.0, 5.0]);
        let b = Owned::f32(&[3], &[1.0, f32::NAN, 3.0]);
        let mut out = Owned::zeros_f32(&[3]);
        run_bin(BinOp::Max, &a, &b, &mut out);
        let r = out.to_f32();
        assert!(r[0].is_nan(), "NaN in lhs must propagate");
        assert!(r[1].is_nan(), "NaN in rhs must propagate");
        assert_eq!(r[2], 5.0);
    }

    #[test]
    fn max_variadic_three_inputs() {
        let a = Owned::f32(&[2, 2], &[5., 1., 8., 2.]);
        let b = Owned::f32(&[2, 2], &[3., 3., 3., 3.]);
        let c = Owned::f32(&[1], &[4.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        BinaryKernel { op: BinOp::Max }
            .execute(&[a.view(), b.view(), c.view()], &mut [out.view_mut()])
            .unwrap();
        // max(a,3,4): max(5,3,4)=5, max(1,3,4)=4, max(8,3,4)=8, max(2,3,4)=4
        assert_eq!(out.to_f32(), vec![5., 4., 8., 4.]);
    }

    #[test]
    fn sqrt_unary() {
        let a = Owned::f32(&[3], &[4., 9., 16.]);
        let mut out = Owned::zeros_f32(&[3]);
        UnaryKernel { op: UnOp::Sqrt }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![2., 3., 4.]);
    }

    #[test]
    fn tanh_known_values() {
        let a = Owned::f32(&[3], &[0., 1., -1.]);
        let mut out = Owned::zeros_f32(&[3]);
        UnaryKernel { op: UnOp::Tanh }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        let r = out.to_f32();
        assert!((r[0] - 0.0).abs() < 1e-6);
        assert!((r[1] - 0.761_594_2).abs() < 1e-6);
        assert!((r[2] + 0.761_594_2).abs() < 1e-6);
    }

    #[test]
    fn erf_known_values() {
        // erf(0)=0, erf(1)=0.8427007929, erf(-1)=-0.8427007929, erf(2)=0.9953222650
        let a = Owned::f32(&[4], &[0., 1., -1., 2.]);
        let mut out = Owned::zeros_f32(&[4]);
        UnaryKernel { op: UnOp::Erf }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        let r = out.to_f32();
        assert!((r[0] - 0.0).abs() < 1e-6);
        assert!((r[1] - 0.842_700_8).abs() < 1e-6);
        assert!((r[2] + 0.842_700_8).abs() < 1e-6);
        assert!((r[3] - 0.995_322_3).abs() < 1e-6);
    }

    #[test]
    fn erf_odd_symmetry_and_limits() {
        assert!((erf(0.0)).abs() < 1e-6);
        assert!((erf(6.0) - 1.0).abs() < 1e-6);
        assert!((erf(-6.0) + 1.0).abs() < 1e-6);
        assert!(erf(f64::NAN).is_nan());
    }

    #[test]
    fn erf_near_zero_high_accuracy() {
        // The A&S 7.1.26 approximation was ~1e-9 off near zero; libm::erf is
        // correctly rounded. Check tight agreement against reference values
        // (erf(x) ≈ 2/√π · x for tiny x).
        let two_over_sqrt_pi = 1.128_379_167_095_512_6_f64;
        for &x in &[1e-3_f64, 1e-4, 1e-5, 1e-6, 1e-7, 1e-9] {
            let expected = two_over_sqrt_pi * x - two_over_sqrt_pi * x * x * x / 3.0;
            assert!(
                (erf(x) - expected).abs() < 1e-15,
                "erf({x}) = {}, expected ≈ {expected}",
                erf(x)
            );
        }
        // A few tabulated exact values to full f64 precision.
        assert!((erf(0.5) - 0.520_499_877_813_046_5).abs() < 1e-12);
        assert!((erf(1.0) - 0.842_700_792_949_714_9).abs() < 1e-12);
        assert!((erf(2.0) - 0.995_322_265_018_952_7).abs() < 1e-12);
    }
}
