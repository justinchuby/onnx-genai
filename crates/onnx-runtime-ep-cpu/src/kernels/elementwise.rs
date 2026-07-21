//! Elementwise f32 kernels (`docs/ORT2.md` §4.4).
//!
//! Two tight families share this module because they share the same dense-f32
//! read/write plumbing:
//!
//! * **Binary broadcasting** — `Sub`, `Mul`, `Div`, `Mod`, `Pow`, and the variadic
//!   `Min`. Each reuses [`broadcast_apply`](super::add::broadcast_apply) (the
//!   numpy right-alignment / size-1 broadcast machinery Add already defines) so
//!   broadcasting semantics stay identical across every binary op.
//! * **Unary** — `Sqrt`, `Erf`, `Tanh`: a straight per-element map.
//!
//! Numerics target ONNX/NumPy exactly. `Erf` has no `std` intrinsic, so it uses
//! the pure-Rust `libm::erf` (the correctly-rounded Sun/FreeBSD algorithm),
//! keeping the crate FFI-free (libm is pure Rust, no `cc`) while matching the
//! ONNX reference to < 1 ulp near zero.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::add::{broadcast_apply, require_same_dtype};
use super::check_arity;
use crate::dtype::{
    ComputeDomain, FloatElem, NumericElem, to_dense, to_dense_float, write_dense, write_dense_float,
};
use crate::strided::numel;
use crate::{dispatch_arith, dispatch_float};

/// The combining operation for a binary elementwise kernel.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Sub,
    Mul,
    Div,
    Pow,
    /// Variadic minimum (ONNX `Min` accepts 1..N inputs).
    Min,
    /// Variadic maximum (ONNX `Max` accepts 1..N inputs).
    Max,
    /// Variadic sum (ONNX `Sum` accepts 1..N inputs).
    Sum,
    /// Variadic arithmetic mean (ONNX `Mean` accepts 1..N inputs).
    Mean,
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
            BinOp::Sum => "Sum",
            BinOp::Mean => "Mean",
        }
    }

    /// Fold `acc` (accumulated left value) with a new operand `v`, in the
    /// element's compute domain. NaN-propagation for `Min`/`Max` and integer
    /// wrapping/divide semantics live in [`ComputeDomain`], so this stays a thin
    /// dtype-generic dispatch.
    fn apply<C: ComputeDomain>(self, acc: C, v: C) -> C {
        match self {
            BinOp::Sub => acc.c_sub(v),
            BinOp::Mul => acc.c_mul(v),
            BinOp::Div => acc.c_div(v),
            BinOp::Pow => acc.c_pow(v),
            BinOp::Min => acc.c_min(v),
            BinOp::Max => acc.c_max(v),
            BinOp::Sum | BinOp::Mean => acc.c_add(v),
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
binary_factory!(SumFactory, BinOp::Sum);
binary_factory!(MeanFactory, BinOp::Mean);

/// Factory for ONNX `Mod`, carrying its `fmod` semantic selector.
pub struct ModFactory;

impl KernelFactory for ModFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let fmod = match node.attr("fmod") {
            None | Some(Attribute::Int(0)) => false,
            Some(Attribute::Int(1)) => true,
            Some(Attribute::Int(value)) => {
                return Err(EpError::KernelFailed(format!(
                    "Mod: `fmod` must be 0 or 1, got {value}"
                )));
            }
            Some(_) => {
                return Err(EpError::KernelFailed(
                    "Mod: `fmod` must be an integer attribute".into(),
                ));
            }
        };
        Ok(Box::new(ModKernel { fmod }))
    }
}

/// ONNX `Mod`: integer floor-mod when `fmod=0`, C-style remainder when `fmod=1`.
pub struct ModKernel {
    fmod: bool,
}

impl Kernel for ModKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Mod", inputs, outputs, 2, 2, 1)?;
        if self.fmod {
            dispatch_arith!(inputs[0].dtype, "Mod", T => {
                mod_typed::<T>(true, inputs, outputs)
            })
        } else {
            match inputs[0].dtype {
                DataType::Int8 => mod_typed::<i8>(false, inputs, outputs),
                DataType::Int16 => mod_typed::<i16>(false, inputs, outputs),
                DataType::Int32 => mod_typed::<i32>(false, inputs, outputs),
                DataType::Int64 => mod_typed::<i64>(false, inputs, outputs),
                DataType::Uint8 => mod_typed::<u8>(false, inputs, outputs),
                DataType::Uint16 => mod_typed::<u16>(false, inputs, outputs),
                DataType::Uint32 => mod_typed::<u32>(false, inputs, outputs),
                DataType::Uint64 => mod_typed::<u64>(false, inputs, outputs),
                dtype => Err(EpError::KernelFailed(format!(
                    "Mod: fmod=0 requires an integer dtype, got {dtype:?}; \
                     floating-point Mod requires fmod=1"
                ))),
            }
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

trait ModDomain: ComputeDomain {
    fn c_mod(self, divisor: Self, fmod: bool) -> Self;
}

macro_rules! impl_float_mod {
    ($($t:ty),*) => {$(
        impl ModDomain for $t {
            #[inline]
            fn c_mod(self, divisor: Self, _fmod: bool) -> Self {
                self % divisor
            }
        }
    )*};
}

macro_rules! impl_signed_mod {
    ($($t:ty),*) => {$(
        impl ModDomain for $t {
            #[inline]
            fn c_mod(self, divisor: Self, fmod: bool) -> Self {
                if divisor == 0 {
                    return 0;
                }
                let remainder = self.wrapping_rem(divisor);
                if !fmod
                    && remainder != 0
                    && (remainder < 0) != (divisor < 0)
                {
                    remainder.wrapping_add(divisor)
                } else {
                    remainder
                }
            }
        }
    )*};
}

macro_rules! impl_unsigned_mod {
    ($($t:ty),*) => {$(
        impl ModDomain for $t {
            #[inline]
            fn c_mod(self, divisor: Self, _fmod: bool) -> Self {
                if divisor == 0 { 0 } else { self % divisor }
            }
        }
    )*};
}

impl_float_mod!(f32, f64);
impl_signed_mod!(i8, i16, i32, i64);
impl_unsigned_mod!(u8, u16, u32, u64);

fn mod_typed<T: NumericElem>(
    fmod: bool,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()>
where
    T::Acc: ModDomain,
{
    require_same_dtype("Mod", &inputs[1], T::DTYPE)?;
    let lhs = to_dense::<T>(&inputs[0])?;
    let rhs = to_dense::<T>(&inputs[1])?;
    let out_shape = outputs[0].shape.to_vec();
    let mut acc = vec![T::Acc::default(); numel(&out_shape)];
    broadcast_apply(&lhs, inputs[0].shape, &out_shape, |i, value| {
        acc[i] = value.to_acc()
    })?;
    broadcast_apply(&rhs, inputs[1].shape, &out_shape, |i, divisor| {
        acc[i] = acc[i].c_mod(divisor.to_acc(), fmod)
    })?;
    let out = acc.into_iter().map(T::from_acc).collect::<Vec<_>>();
    write_dense::<T>(&mut outputs[0], &out)
}

impl Kernel for BinaryKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        // Min/Max/Sum/Mean are variadic (1..N); the rest are strictly binary.
        let (min_in, max_in) = match self.op {
            BinOp::Min | BinOp::Max | BinOp::Sum | BinOp::Mean => (1, usize::MAX),
            _ => (2, 2),
        };
        check_arity(self.op.name(), inputs, outputs, min_in, max_in, 1)?;
        let op = self.op;
        if op == BinOp::Mul {
            crate::trace::record_kernel_metrics(inputs, outputs, || {
                (outputs[0].numel() as u64).saturating_mul(inputs.len().saturating_sub(1) as u64)
            });
        }
        if op == BinOp::Mul && multiply_contiguous_f32(inputs, &mut outputs[0]) {
            return Ok(());
        }
        match op {
            BinOp::Pow => {
                dispatch_arith!(inputs[0].dtype, op.name(), T => pow_typed::<T>(inputs, outputs))
            }
            BinOp::Sum | BinOp::Mean => {
                dispatch_float!(inputs[0].dtype, op.name(), T => binary_typed::<T>(op, inputs, outputs))
            }
            _ => {
                dispatch_arith!(inputs[0].dtype, op.name(), T => binary_typed::<T>(op, inputs, outputs))
            }
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn multiply_contiguous_f32(inputs: &[TensorView], output: &mut TensorMut) -> bool {
    if inputs.len() != 2
        || inputs[0].dtype != DataType::Float32
        || inputs[1].dtype != DataType::Float32
        || output.dtype != DataType::Float32
        || inputs[0].shape != output.shape
        || inputs[1].shape != output.shape
        || !inputs[0].is_contiguous()
        || !inputs[1].is_contiguous()
        || !output.is_contiguous()
    {
        return false;
    }

    let n = output.numel();
    let bytes = n.saturating_mul(std::mem::size_of::<f32>());
    let output_start = output.data_ptr_mut::<f32>() as usize;
    let output_end = output_start.saturating_add(bytes);
    if inputs.iter().any(|input| {
        let input_start = input.data_ptr::<f32>() as usize;
        let input_end = input_start.saturating_add(bytes);
        output_start < input_end && input_start < output_end
    }) {
        return false;
    }
    // SAFETY: the executor bounds-checks every view before dispatch. The dtype,
    // equal shapes, and contiguous layouts above prove each pointer spans n f32s;
    // the range check proves the mutable output does not alias either input.
    let lhs = unsafe { std::slice::from_raw_parts(inputs[0].data_ptr::<f32>(), n) };
    let rhs = unsafe { std::slice::from_raw_parts(inputs[1].data_ptr::<f32>(), n) };
    let output = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), n) };
    for ((output, &lhs), &rhs) in output.iter_mut().zip(lhs).zip(rhs) {
        *output = lhs * rhs;
    }
    true
}

/// Base-storage behavior for ONNX Pow.  The exponent is allowed to have a
/// different numeric storage type, while the result always uses this base type.
trait PowBase: NumericElem {
    fn pow_exponent(self, exponent: f64) -> Self;
}

// Implement explicitly so f16/bf16 retain their normal f32 compute-and-round path.
impl PowBase for f32 {
    fn pow_exponent(self, exponent: f64) -> Self {
        self.powf(exponent as f32)
    }
}
impl PowBase for f64 {
    fn pow_exponent(self, exponent: f64) -> Self {
        self.powf(exponent)
    }
}
impl PowBase for half::f16 {
    fn pow_exponent(self, exponent: f64) -> Self {
        half::f16::from_f32(self.to_f32().powf(exponent as f32))
    }
}
impl PowBase for half::bf16 {
    fn pow_exponent(self, exponent: f64) -> Self {
        half::bf16::from_f32(self.to_f32().powf(exponent as f32))
    }
}
macro_rules! impl_pow_int {
    ($($t:ty),* $(,)?) => {$(
        impl PowBase for $t {
            fn pow_exponent(self, exponent: f64) -> Self { (self as f64).powf(exponent) as Self }
        }
    )*};
}
impl_pow_int!(i8, i16, i32, i64, u8, u16, u32, u64);

fn exponents_as_f64(input: &TensorView) -> Result<Vec<f64>> {
    macro_rules! dense {
        ($t:ty) => {
            to_dense::<$t>(input)?
                .into_iter()
                .map(|v| v as f64)
                .collect()
        };
    }
    match input.dtype {
        DataType::Float32 => Ok(dense!(f32)),
        DataType::Float64 => Ok(dense!(f64)),
        DataType::Float16 => Ok(to_dense::<half::f16>(input)?
            .into_iter()
            .map(|v| v.to_f32() as f64)
            .collect()),
        DataType::BFloat16 => Ok(to_dense::<half::bf16>(input)?
            .into_iter()
            .map(|v| v.to_f32() as f64)
            .collect()),
        DataType::Int8 => Ok(dense!(i8)),
        DataType::Int16 => Ok(dense!(i16)),
        DataType::Int32 => Ok(dense!(i32)),
        DataType::Int64 => Ok(dense!(i64)),
        DataType::Uint8 => Ok(dense!(u8)),
        DataType::Uint16 => Ok(dense!(u16)),
        DataType::Uint32 => Ok(dense!(u32)),
        DataType::Uint64 => Ok(dense!(u64)),
        dtype => Err(EpError::KernelFailed(format!(
            "Pow: unsupported exponent dtype {dtype:?}"
        ))),
    }
}

fn pow_typed<T: PowBase>(inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
    if outputs[0].dtype != T::DTYPE {
        return Err(EpError::KernelFailed(
            "Pow: output dtype must match base dtype".into(),
        ));
    }
    let base = to_dense::<T>(&inputs[0])?;
    let exponent = exponents_as_f64(&inputs[1])?;
    let out_shape = outputs[0].shape.to_vec();
    let mut values = vec![T::from_f32_scalar(0.0); numel(&out_shape)];
    broadcast_apply(&base, inputs[0].shape, &out_shape, |i, value| {
        values[i] = value
    })?;
    broadcast_apply(&exponent, inputs[1].shape, &out_shape, |i, value| {
        values[i] = values[i].pow_exponent(value)
    })?;
    write_dense::<T>(&mut outputs[0], &values)
}

/// Dtype-generic binary/variadic fold: seed from the first operand, then fold
/// each remaining operand with the op's combiner, all in `T`'s compute domain.
fn binary_typed<T: NumericElem>(
    op: BinOp,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    let out_shape = outputs[0].shape.to_vec();
    let n = numel(&out_shape);
    let mut acc = vec![T::Acc::default(); n];

    // Seed the accumulator from the first operand (broadcast to the output).
    let first = to_dense::<T>(&inputs[0])?;
    broadcast_apply(&first, inputs[0].shape, &out_shape, |i, v| {
        acc[i] = v.to_acc()
    })?;

    // Fold each remaining operand with the op's combiner.
    for input in &inputs[1..] {
        require_same_dtype(op.name(), input, T::DTYPE)?;
        let rhs = to_dense::<T>(input)?;
        broadcast_apply(&rhs, input.shape, &out_shape, |i, v| {
            acc[i] = op.apply(acc[i], v.to_acc())
        })?;
    }

    let out: Vec<T> = acc
        .into_iter()
        .map(|v| {
            T::from_acc(if matches!(op, BinOp::Mean) {
                v.c_div_usize(inputs.len())
            } else {
                v
            })
        })
        .collect();
    write_dense::<T>(&mut outputs[0], &out)
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
        let op = self.op;
        dispatch_float!(inputs[0].dtype, op.name(), T => unary_typed::<T>(op, inputs, outputs))
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Dtype-generic unary map: widen each element to `f32`, apply the (unchanged)
/// f32 transcendental, narrow back. Float dtypes only (ONNX defines `Sqrt`,
/// `Erf`, `Tanh` over float16/float/double/bfloat16).
fn unary_typed<T: FloatElem>(
    op: UnOp,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    let x = to_dense_float::<T>(&inputs[0])?;
    let y: Vec<T> = x
        .iter()
        .map(|&v| T::from_f32(op.apply(v.to_f32())))
        .collect();
    write_dense_float::<T>(&mut outputs[0], &y)
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
    fn mul_same_shape_contiguous() {
        let a = Owned::f32(&[2, 3], &[1., -2., 3., 4., 0., f32::NAN]);
        let b = Owned::f32(&[2, 3], &[5., 6., -7., 0.5, 8., 9.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        run_bin(BinOp::Mul, &a, &b, &mut out);
        let values = out.to_f32();
        assert_eq!(&values[..5], &[5., -12., -21., 2., 0.]);
        assert!(values[5].is_nan());
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
    fn mod_integer_floor_semantics_follow_divisor_sign_i32() {
        let a = Owned::i32(&[4], &[-5, 5, -5, 5]);
        let b = Owned::i32(&[4], &[3, 3, -3, -3]);
        let mut out = Owned::zeros(DataType::Int32, &[4]);
        ModKernel { fmod: false }
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![1, 2, -2, -1]);
    }

    #[test]
    fn mod_integer_floor_semantics_broadcast_i64() {
        let a = Owned::i64(&[3, 1], &[5, -5, 8]);
        let b = Owned::i64(&[1, 4], &[3, -3, 4, -4]);
        let mut out = Owned::zeros(DataType::Int64, &[3, 4]);
        ModKernel { fmod: false }
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![2, -1, 1, -3, 1, -2, 3, -1, 2, -1, 0, 0]);
    }

    #[test]
    fn mod_fmod_float_follows_dividend_sign() {
        let a = Owned::f32(&[4], &[-5.5, 5.5, -5.5, 5.5]);
        let b = Owned::f32(&[4], &[3.0, 3.0, -3.0, -3.0]);
        let mut out = Owned::zeros_f32(&[4]);
        let mut node = Node::new(onnx_runtime_ir::NodeId(0), "Mod", vec![], vec![]);
        node.attributes.insert("fmod".into(), Attribute::Int(1));
        ModFactory
            .create(&node, &[])
            .unwrap()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![-2.5, 2.5, -2.5, 2.5]);
    }

    #[test]
    fn mod_fmod_bf16_computes_in_f32_and_preserves_dividend_sign() {
        let dividends = [-5.5, 7.3];
        let divisors = [3.0, 2.2];
        let a = Owned::bf16(&[2], &dividends);
        let b = Owned::bf16(&[2], &divisors);
        let mut out = Owned::zeros(DataType::BFloat16, &[2]);

        ModKernel { fmod: true }
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        let expected = dividends
            .into_iter()
            .zip(divisors)
            .map(|(dividend, divisor)| {
                let dividend = half::bf16::from_f32(dividend).to_f32();
                let divisor = half::bf16::from_f32(divisor).to_f32();
                half::bf16::from_f32(dividend % divisor).to_f32()
            })
            .collect::<Vec<_>>();
        assert_eq!(out.to_bf16_as_f32(), expected);
        assert!(out.to_bf16_as_f32()[0].is_sign_negative());
    }

    #[test]
    fn mod_integer_zero_divisor_matches_div_convention() {
        let a = Owned::i32(&[2], &[5, -5]);
        let b = Owned::i32(&[2], &[0, 0]);
        let mut out = Owned::zeros(DataType::Int32, &[2]);
        ModKernel { fmod: false }
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![0, 0]);
    }

    #[test]
    fn mod_default_mode_rejects_float() {
        let a = Owned::f32(&[1], &[5.5]);
        let b = Owned::f32(&[1], &[3.0]);
        let mut out = Owned::zeros_f32(&[1]);
        let error = ModKernel { fmod: false }
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("fmod=0 requires an integer dtype")
        );
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
    fn pow_accepts_mixed_base_and_exponent_types() {
        let base = Owned::f32(&[2], &[2., 3.]);
        let exponent = Owned::i64(&[2], &[3, 2]);
        let mut out = Owned::zeros_f32(&[2]);
        BinaryKernel { op: BinOp::Pow }
            .execute(&[base.view(), exponent.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![8., 9.]);

        let base = Owned::i32(&[2], &[2, 3]);
        let exponent = Owned::f32(&[2], &[3., 2.]);
        let mut out = Owned::zeros(DataType::Int32, &[2]);
        BinaryKernel { op: BinOp::Pow }
            .execute(&[base.view(), exponent.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![8, 9]);
    }

    #[test]
    fn pow_covers_integer_base_and_exponent_combinations() {
        let base = Owned::i32(&[2], &[2, 3]);
        let exponent = Owned::i32(&[2], &[3, 2]);
        let mut out = Owned::zeros(DataType::Int32, &[2]);
        BinaryKernel { op: BinOp::Pow }
            .execute(&[base.view(), exponent.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![8, 9]);

        let base = Owned::i64(&[3], &[2, -3, 4]);
        let exponent = Owned::i64(&[3], &[3, 2, 0]);
        let mut out = Owned::zeros(DataType::Int64, &[3]);
        BinaryKernel { op: BinOp::Pow }
            .execute(&[base.view(), exponent.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![8, 9, 1]);
    }

    #[test]
    fn pow_accepts_float_exponents_for_i64_base() {
        let base = Owned::i64(&[2], &[2, 3]);
        let exponent = Owned::f32(&[2], &[3., 2.]);
        let mut out = Owned::zeros(DataType::Int64, &[2]);
        BinaryKernel { op: BinOp::Pow }
            .execute(&[base.view(), exponent.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i64(), vec![8, 9]);
    }

    #[test]
    fn min_variadic_three_inputs_with_broadcast() {
        let a = Owned::f32(&[2, 2], &[5., 1., 8., 2.]);
        let b = Owned::f32(&[2, 2], &[3., 3., 3., 3.]);
        let c = Owned::f32(&[1], &[4.]); // broadcast scalar-ish
        let mut out = Owned::zeros_f32(&[2, 2]);
        BinaryKernel { op: BinOp::Min }
            .execute(&[a.view(), b.view(), c.view()], &mut [out.view_mut()])
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
    fn sum_variadic_broadcasts_matrix_vector_and_scalar() {
        let matrix = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let vector = Owned::f32(&[3], &[10., 20., 30.]);
        let scalar = Owned::f32(&[], &[100.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        BinaryKernel { op: BinOp::Sum }
            .execute(
                &[matrix.view(), vector.view(), scalar.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(out.to_f32(), vec![111., 122., 133., 114., 125., 136.]);
    }

    #[test]
    fn mean_variadic_broadcasts_matrix_vector_and_scalar() {
        let matrix = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let vector = Owned::f32(&[3], &[10., 20., 30.]);
        let scalar = Owned::f32(&[], &[100.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        BinaryKernel { op: BinOp::Mean }
            .execute(
                &[matrix.view(), vector.view(), scalar.view()],
                &mut [out.view_mut()],
            )
            .unwrap();
        assert_eq!(
            out.to_f32(),
            vec![37., 40.666_668, 44.333_332, 38., 41.666_668, 45.333_332]
        );
    }

    #[test]
    fn sum_rejects_integer_input() {
        let input = Owned::i32(&[2], &[1, 2]);
        let mut out = Owned::zeros(DataType::Int32, &[2]);
        let error = BinaryKernel { op: BinOp::Sum }
            .execute(&[input.view()], &mut [out.view_mut()])
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Sum: unsupported element type Int32")
        );
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

    // --- dtype coverage ----------------------------------------------------

    use onnx_runtime_ir::DataType;

    #[test]
    fn mul_f16_computes_in_f32() {
        let a = Owned::f16(&[3, 1], &[1., 2., 3.]);
        let b = Owned::f16(&[1, 4], &[10., 20., 30., 40.]);
        let mut out = Owned::zeros(DataType::Float16, &[3, 4]);
        BinaryKernel { op: BinOp::Mul }
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_f16_as_f32(),
            vec![10., 20., 30., 40., 20., 40., 60., 80., 30., 60., 90., 120.]
        );
    }

    #[test]
    fn sub_bf16() {
        let a = Owned::bf16(&[2, 2], &[10., 20., 30., 40.]);
        let b = Owned::bf16(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros(DataType::BFloat16, &[2, 2]);
        BinaryKernel { op: BinOp::Sub }
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_bf16_as_f32(), vec![9., 18., 27., 36.]);
    }

    #[test]
    fn div_int32_truncates_and_guards_zero() {
        // Integer Div is truncating; divide-by-zero yields 0 (not a panic).
        let a = Owned::i32(&[3], &[7, -7, 5]);
        let b = Owned::i32(&[3], &[2, 2, 0]);
        let mut out = Owned::zeros(DataType::Int32, &[3]);
        BinaryKernel { op: BinOp::Div }
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_i32(), vec![3, -3, 0]);
    }

    #[test]
    fn min_max_f16_propagate_nan() {
        // NaN pattern 0x7E00 in f16; Min/Max must propagate it.
        let a = Owned::f16_bits(&[2], &[0x7E00, 0x4000 /* 2.0 */]);
        let b = Owned::f16(&[2], &[1.0, 5.0]);
        let mut mn = Owned::zeros(DataType::Float16, &[2]);
        let mut mx = Owned::zeros(DataType::Float16, &[2]);
        BinaryKernel { op: BinOp::Min }
            .execute(&[a.view(), b.view()], &mut [mn.view_mut()])
            .unwrap();
        BinaryKernel { op: BinOp::Max }
            .execute(&[a.view(), b.view()], &mut [mx.view_mut()])
            .unwrap();
        // Position 0 is NaN in both.
        assert_eq!(mn.to_u16_bits()[0] & 0x7C00, 0x7C00);
        assert_ne!(mn.to_u16_bits()[0] & 0x03FF, 0);
        assert_eq!(mn.to_f16_as_f32()[1], 2.0);
        assert_eq!(mx.to_f16_as_f32()[1], 5.0);
    }

    #[test]
    fn sqrt_f16_and_bf16() {
        let a16 = Owned::f16(&[3], &[4., 9., 16.]);
        let mut o16 = Owned::zeros(DataType::Float16, &[3]);
        UnaryKernel { op: UnOp::Sqrt }
            .execute(&[a16.view()], &mut [o16.view_mut()])
            .unwrap();
        assert_eq!(o16.to_f16_as_f32(), vec![2., 3., 4.]);

        let ab = Owned::bf16(&[3], &[4., 9., 16.]);
        let mut ob = Owned::zeros(DataType::BFloat16, &[3]);
        UnaryKernel { op: UnOp::Sqrt }
            .execute(&[ab.view()], &mut [ob.view_mut()])
            .unwrap();
        assert_eq!(ob.to_bf16_as_f32(), vec![2., 3., 4.]);
    }

    #[test]
    fn tanh_f16_matches_f32_within_tolerance() {
        let a = Owned::f16(&[3], &[0., 1., -1.]);
        let mut out = Owned::zeros(DataType::Float16, &[3]);
        UnaryKernel { op: UnOp::Tanh }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        let r = out.to_f16_as_f32();
        assert!(r[0].abs() < 1e-2);
        assert!((r[1] - 0.7616).abs() < 1e-2);
        assert!((r[2] + 0.7616).abs() < 1e-2);
    }

    #[test]
    fn erf_bf16_reaches_dtype_without_touching_formula() {
        // Erf's numeric formula is unchanged; the dtype dispatch simply widens.
        let a = Owned::bf16(&[2], &[0., 1.]);
        let mut out = Owned::zeros(DataType::BFloat16, &[2]);
        UnaryKernel { op: UnOp::Erf }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap();
        let r = out.to_bf16_as_f32();
        assert!(r[0].abs() < 1e-2);
        assert!((r[1] - 0.8427).abs() < 5e-2); // bf16 has ~3 significant digits
    }

    #[test]
    fn sqrt_rejects_integer_dtype_with_rule1() {
        let a = Owned::i32(&[2], &[4, 9]);
        let mut out = Owned::zeros(DataType::Int32, &[2]);
        let err = UnaryKernel { op: UnOp::Sqrt }
            .execute(&[a.view()], &mut [out.view_mut()])
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("WHAT") && msg.contains("HOW"));
    }
}
