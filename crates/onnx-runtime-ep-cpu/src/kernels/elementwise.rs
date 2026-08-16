//! Elementwise f32 kernels (`docs/architecture/ORT2.md` §4.4).
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
use super::simd_activations;
use crate::dtype::{
    ComputeDomain, FloatElem, NumericElem, to_dense, to_dense_f32_widen, to_dense_float,
    write_dense, write_dense_float,
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

/// Resolve a [`BinOp`] to a monomorphised combiner **once**, outside the hot
/// loop. Every arm binds `$fold` to a closure over `C: ComputeDomain` and
/// evaluates `$body`.
macro_rules! dispatch_binop {
    ($op:expr, $fold:ident => $body:expr) => {
        match $op {
            BinOp::Sub => {
                let $fold = |a: <T as NumericElem>::Acc, b| a.c_sub(b);
                $body
            }
            BinOp::Mul => {
                let $fold = |a: <T as NumericElem>::Acc, b| a.c_mul(b);
                $body
            }
            BinOp::Div => {
                let $fold = |a: <T as NumericElem>::Acc, b| a.c_div(b);
                $body
            }
            BinOp::Pow => {
                let $fold = |a: <T as NumericElem>::Acc, b| a.c_pow(b);
                $body
            }
            BinOp::Min => {
                let $fold = |a: <T as NumericElem>::Acc, b| a.c_min(b);
                $body
            }
            BinOp::Max => {
                let $fold = |a: <T as NumericElem>::Acc, b| a.c_max(b);
                $body
            }
            BinOp::Sum | BinOp::Mean => {
                let $fold = |a: <T as NumericElem>::Acc, b| a.c_add(b);
                $body
            }
        }
    };
}

/// A stateless binary/variadic elementwise kernel.
pub struct BinaryKernel {
    op: BinOp,
    /// Structural FLOPs (one op per broadcast output element) when the input
    /// shapes were static at build time; `None` otherwise (issue #995 — never
    /// fabricated).
    flops: Option<u64>,
}

macro_rules! binary_factory {
    ($factory:ident, $variant:expr) => {
        /// Factory (no attributes).
        pub struct $factory;
        impl KernelFactory for $factory {
            fn create(&self, _node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                Ok(Box::new(BinaryKernel {
                    op: $variant,
                    flops: super::flops::elementwise_flops(shapes),
                }))
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
        if matches!(op, BinOp::Sub | BinOp::Mul | BinOp::Div)
            && (binary_contiguous(op, inputs, &mut outputs[0])
                || binary_broadcast_contiguous(op, inputs, &mut outputs[0]))
        {
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

    fn estimated_flops(&self) -> Option<u64> {
        self.flops
    }
}

/// Fast path for strictly-binary `Sub`/`Mul`/`Div` when both operands and the
/// output share one contiguous shape (no broadcasting) and do not alias. This
/// is the common decode-time case (SwiGLU `gate * up`, residual `Sub`, etc.) and
/// runs a single tight per-element loop instead of the general
/// [`broadcast_apply`] path, which recomputes a multi-axis source index for
/// every element and allocates an accumulator plus dense staging buffers.
///
/// Byte-identical to the general path: each element uses the same
/// `to_acc`/`from_acc` rounding and the same [`BinOp::apply`] combiner. Covers
/// every numeric dtype (f16/bf16 previously fell to the slow path since the old
/// fast path was f32-only).
fn binary_contiguous(op: BinOp, inputs: &[TensorView], output: &mut TensorMut) -> bool {
    if inputs.len() != 2
        || inputs[0].dtype != output.dtype
        || inputs[1].dtype != output.dtype
        || inputs[0].shape != output.shape
        || inputs[1].shape != output.shape
        || inputs[0].strides != output.strides
        || inputs[1].strides != output.strides
        || !onnx_runtime_ir::is_dense(output.shape, output.strides)
    {
        return false;
    }

    let n = output.numel();
    let bytes = output.byte_size();
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(bytes);
    if inputs.iter().any(|input| {
        let input_start = input.data_ptr::<u8>() as usize;
        let input_end = input_start.saturating_add(bytes);
        output_start < input_end && input_start < output_end
    }) {
        return false;
    }

    match output.dtype {
        DataType::Float32 => binary_contiguous_typed::<f32>(op, inputs, output, n),
        DataType::Float64 => binary_contiguous_typed::<f64>(op, inputs, output, n),
        DataType::Float16 => binary_contiguous_typed::<half::f16>(op, inputs, output, n),
        DataType::BFloat16 => binary_contiguous_typed::<half::bf16>(op, inputs, output, n),
        _ => return false,
    }
    true
}

fn binary_contiguous_typed<T: NumericElem>(
    op: BinOp,
    inputs: &[TensorView],
    output: &mut TensorMut,
    n: usize,
) {
    // SAFETY: the caller proved equal contiguous shapes (each pointer spans n
    // Ts), matching dtypes, and no output/input aliasing.
    let lhs = unsafe { std::slice::from_raw_parts(inputs[0].data_ptr::<T>(), n) };
    let rhs = unsafe { std::slice::from_raw_parts(inputs[1].data_ptr::<T>(), n) };
    let out = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<T>(), n) };
    if try_half_binary::<T>(op, lhs, rhs, out) {
        return;
    }
    // Same reason as `binary_broadcast_typed`: resolve the combiner once so the
    // element loop is a single arithmetic op and can vectorise.
    dispatch_binop!(op, fold => {
        for ((out, &lhs), &rhs) in out.iter_mut().zip(lhs).zip(rhs) {
            *out = T::from_acc(fold(lhs.to_acc(), rhs.to_acc()));
        }
    });
}

/// Elements converted per staged pass when a 16-bit float binary op runs
/// through the `f32` compute domain. The two `f32` staging buffers of this
/// length are 8 KiB total, so a pass stays resident in L1 next to the 16-bit
/// chunks. Matches `F16_STAGE_CHUNK` in `dense_elementwise`, which stages the
/// unary paths the same way.
///
/// Only the x86 staging path consumes this; on other targets `try_half_binary`
/// declines unconditionally and the constant would be dead code.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const HALF_STAGE_CHUNK: usize = 1024;

/// The two 16-bit float storages whose `NumericElem::Acc` is `f32`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum HalfKind {
    F16,
    Bf16,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl HalfKind {
    fn available(self) -> bool {
        match self {
            HalfKind::F16 => crate::dtype::f16c::available(),
            HalfKind::Bf16 => crate::dtype::bf16x::available(),
        }
    }

    /// Bulk equivalent of `NumericElem::to_acc` (`<half>::to_f32`).
    ///
    /// # Safety
    /// [`HalfKind::available`] must hold for `self`; `src.len() == dst.len()`.
    #[inline]
    unsafe fn widen(self, src: &[u16], dst: &mut [f32]) {
        debug_assert_eq!(src.len(), dst.len());
        unsafe {
            match self {
                HalfKind::F16 => crate::dtype::f16c::widen(src, dst),
                // `widen_quieting`, not `widen`: `to_acc` is `bf16::to_f32`,
                // which quiets a signalling NaN, and the fallback this replaces
                // would too.
                HalfKind::Bf16 => crate::dtype::bf16x::widen_quieting(src, dst),
            }
        }
    }

    /// Bulk equivalent of `NumericElem::from_acc` (`<half>::from_f32`).
    ///
    /// # Safety
    /// [`HalfKind::available`] must hold for `self`; `src.len() == dst.len()`.
    #[inline]
    unsafe fn narrow(self, src: &[f32], dst: &mut [u16]) {
        debug_assert_eq!(src.len(), dst.len());
        unsafe {
            match self {
                HalfKind::F16 => crate::dtype::f16c::narrow(src, dst),
                HalfKind::Bf16 => crate::dtype::bf16x::narrow(src, dst),
            }
        }
    }
}

/// How one operand of a staged 16-bit binary op maps onto the output.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
enum HalfSide<'a> {
    /// One element per output element, in output order; widened per chunk.
    Full(&'a [u16]),
    /// A single value, widened once and splatted.
    Splat(f32),
    /// A block repeated over the output, widened once up front so the repeat
    /// costs an `f32` copy per pass instead of a re-conversion.
    Block(Vec<f32>),
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl HalfSide<'_> {
    /// # Safety
    /// [`HalfKind::available`] must hold for `kind`.
    unsafe fn new(kind: HalfKind, src: &[u16], n: usize) -> HalfSide<'_> {
        if src.len() == n {
            HalfSide::Full(src)
        } else if src.len() == 1 {
            let mut v = [0.0f32];
            unsafe { kind.widen(src, &mut v) };
            HalfSide::Splat(v[0])
        } else {
            let mut block = vec![0.0f32; src.len()];
            unsafe { kind.widen(src, &mut block) };
            HalfSide::Block(block)
        }
    }

    /// Materialize `dst.len()` `f32` operand values starting at output element
    /// `offset`. Mirrors the `i % len` indexing of [`broadcast_loops`].
    ///
    /// # Safety
    /// [`HalfKind::available`] must hold for `kind`.
    #[inline]
    unsafe fn stage(&self, kind: HalfKind, offset: usize, dst: &mut [f32]) {
        match self {
            HalfSide::Full(src) => unsafe { kind.widen(&src[offset..offset + dst.len()], dst) },
            HalfSide::Splat(v) => dst.fill(*v),
            HalfSide::Block(block) => {
                let mut pos = offset % block.len();
                let mut written = 0;
                while written < dst.len() {
                    let take = (block.len() - pos).min(dst.len() - written);
                    dst[written..written + take].copy_from_slice(&block[pos..pos + take]);
                    written += take;
                    pos = 0;
                }
            }
        }
    }
}

/// Run a 16-bit float binary op by widening both operands to `f32` in bulk,
/// folding in the `f32` domain, and narrowing back.
///
/// This is not an approximation: `NumericElem::Acc` for `f16`/`bf16` is already
/// `f32`, so the scalar path this replaces computes exactly the same
/// `from_acc(fold(to_acc(a), to_acc(b)))`. The only change is that the two
/// conversions become vector instructions instead of the `half` crate's
/// software conversion, which is what made every 16-bit binary op ~12x slower
/// than ORT even after the dense walk was in place.
///
/// # Safety
/// [`HalfKind::available`] must hold for `kind`, and each of `lhs`/`rhs` must
/// have length `out.len()`, `1`, or a length that divides the output the same
/// way [`broadcast_loops`] assumes.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn binary_half<T: NumericElem<Acc = f32>>(
    op: BinOp,
    kind: HalfKind,
    lhs: &[u16],
    rhs: &[u16],
    out: &mut [u16],
) {
    let n = out.len();
    let (lhs, rhs) = unsafe { (HalfSide::new(kind, lhs, n), HalfSide::new(kind, rhs, n)) };
    // Two buffers, not three: the fold writes back over the left staging
    // buffer, so a pass touches 8 KiB instead of 12 KiB.
    let mut lbuf = [0.0f32; HALF_STAGE_CHUNK];
    let mut rbuf = [0.0f32; HALF_STAGE_CHUNK];
    dispatch_binop!(op, fold => {
        let mut offset = 0;
        while offset < n {
            let len = HALF_STAGE_CHUNK.min(n - offset);
            let lbuf = &mut lbuf[..len];
            match (&lhs, &rhs) {
                // `x op scalar` and `scalar op x`: the operand is a register
                // constant, so the pass is one widen, one arithmetic op and one
                // narrow with no second staging buffer touched at all. This is
                // the shape an inlined ONNX activation function emits.
                (HalfSide::Full(src), HalfSide::Splat(v)) => {
                    unsafe { kind.widen(&src[offset..offset + len], lbuf) };
                    let v = *v;
                    for a in lbuf.iter_mut() {
                        *a = fold(*a, v);
                    }
                }
                (HalfSide::Splat(v), HalfSide::Full(src)) => {
                    unsafe { kind.widen(&src[offset..offset + len], lbuf) };
                    let v = *v;
                    for b in lbuf.iter_mut() {
                        *b = fold(v, *b);
                    }
                }
                _ => {
                    let rbuf = &mut rbuf[..len];
                    unsafe {
                        lhs.stage(kind, offset, lbuf);
                        rhs.stage(kind, offset, rbuf);
                    }
                    for (a, &b) in lbuf.iter_mut().zip(rbuf.iter()) {
                        *a = fold(*a, b);
                    }
                }
            }
            unsafe { kind.narrow(lbuf, &mut out[offset..offset + len]) };
            offset += len;
        }
    });
}

/// If `T` is a 16-bit float and the host has the bulk converters, run the op
/// through [`binary_half`] and report that it was handled.
///
/// Declines (returning `false`, leaving `out` untouched) for every other dtype
/// and on hosts without `f16c`/`avx2`, so the scalar loops below stay the
/// portable fallback.
#[inline]
fn try_half_binary<T: NumericElem>(op: BinOp, lhs: &[T], rhs: &[T], out: &mut [T]) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        try_half_binary_x86::<T>(op, lhs, rhs, out)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = (op, lhs, rhs, out);
        false
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn try_half_binary_x86<T: NumericElem>(op: BinOp, lhs: &[T], rhs: &[T], out: &mut [T]) -> bool {
    {
        let kind = match T::DTYPE {
            DataType::Float16 => HalfKind::F16,
            DataType::BFloat16 => HalfKind::Bf16,
            _ => return false,
        };
        if !kind.available() {
            return false;
        }
        // A zero-length operand against a non-empty output would make the
        // repeat modulo in `HalfSide::stage` divide by zero. `dense_operand`
        // cannot produce that (a zero extent in an operand is a zero extent in
        // the output), but decline rather than rely on it.
        if (lhs.is_empty() || rhs.is_empty()) && !out.is_empty() {
            return false;
        }
        debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<u16>());
        debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<u16>());
        // SAFETY: `T::DTYPE` pinned `T` to `half::f16`/`half::bf16`, each a
        // `#[repr(transparent)]` newtype over `u16`, so the reinterpretation is
        // layout-preserving and length-preserving. The three slices came from
        // the caller and keep their provenance and exclusivity here.
        let (lhs, rhs, out) = unsafe {
            (
                std::slice::from_raw_parts(lhs.as_ptr().cast::<u16>(), lhs.len()),
                std::slice::from_raw_parts(rhs.as_ptr().cast::<u16>(), rhs.len()),
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u16>(), out.len()),
            )
        };
        // SAFETY: `available()` was just checked, and the operand lengths are
        // whatever the caller already proved valid for the scalar loops.
        unsafe {
            match kind {
                HalfKind::F16 => binary_half::<half::f16>(op, kind, lhs, rhs, out),
                HalfKind::Bf16 => binary_half::<half::bf16>(op, kind, lhs, rhs, out),
            }
        }
        true
    }
}

/// How a dense operand maps onto the output in [`binary_broadcast_contiguous`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum DenseOperand {
    /// One element per output element, in the same order.
    Full,
    /// A contiguous block of `n` elements repeated over the output, i.e. the
    /// operand's shape is a right-aligned suffix of the output shape. Covers a
    /// scalar (`n == 1`) and the row-bias case `[B, S, C] op [C]`.
    Repeat(usize),
}

/// Classify `view` against a **row-major contiguous** output shape, or `None`
/// when the broadcast is not a simple suffix repeat (e.g. an interior size-1
/// axis such as `[B, 1, C]`, which needs the general strided walk).
fn dense_operand(view: &TensorView, out_shape: &[usize]) -> Option<DenseOperand> {
    if !view.is_contiguous() {
        return None;
    }
    if view.shape == out_shape {
        return Some(DenseOperand::Full);
    }
    // Leading unit axes carry no elements, so strip them before matching.
    let first = view
        .shape
        .iter()
        .position(|&d| d != 1)
        .unwrap_or(view.shape.len());
    let tail = &view.shape[first..];
    if tail.len() > out_shape.len() || tail != &out_shape[out_shape.len() - tail.len()..] {
        return None;
    }
    Some(DenseOperand::Repeat(tail.iter().product()))
}

/// Fast path for strictly-binary `Sub`/`Mul`/`Div` where the operands broadcast
/// onto the output as a **suffix repeat** — most importantly a scalar operand,
/// which is what every inlined ONNX activation function multiplies and adds by.
///
/// [`binary_contiguous`] only fires when all three shapes are identical, so
/// `x * 0.5` fell all the way to [`broadcast_apply`], which per element does a
/// dot product over the rank, a `next_index` carry chain and a closure call,
/// and additionally allocates a whole-tensor accumulator that it walks three
/// times. That measured ~7.7 ns/element, about **60x slower than ORT**.
///
/// Byte-identical to the general path: same `to_acc`/`from_acc` rounding and the
/// same [`BinOp::apply`] combiner, just with the index arithmetic hoisted.
fn binary_broadcast_contiguous(op: BinOp, inputs: &[TensorView], output: &mut TensorMut) -> bool {
    if inputs.len() != 2
        || inputs[0].dtype != output.dtype
        || inputs[1].dtype != output.dtype
        || !output.is_contiguous()
    {
        return false;
    }
    let out_shape = output.shape.to_vec();
    let (Some(lhs_kind), Some(rhs_kind)) = (
        dense_operand(&inputs[0], &out_shape),
        dense_operand(&inputs[1], &out_shape),
    ) else {
        return false;
    };
    // Nothing to gain over `binary_contiguous`, which already handles this and
    // also accepts dense-but-permuted layouts.
    if lhs_kind == DenseOperand::Full && rhs_kind == DenseOperand::Full {
        return false;
    }

    let n = output.numel();
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(output.byte_size());
    if inputs.iter().any(|input| {
        let start = input.data_ptr::<u8>() as usize;
        let end = start.saturating_add(input.byte_size());
        output_start < end && start < output_end
    }) {
        return false;
    }

    let lens = (kind_len(lhs_kind, n), kind_len(rhs_kind, n));
    match output.dtype {
        DataType::Float32 => binary_broadcast_typed::<f32>(op, inputs, output, n, lens),
        DataType::Float64 => binary_broadcast_typed::<f64>(op, inputs, output, n, lens),
        DataType::Float16 => binary_broadcast_typed::<half::f16>(op, inputs, output, n, lens),
        DataType::BFloat16 => binary_broadcast_typed::<half::bf16>(op, inputs, output, n, lens),
        _ => return false,
    }
    true
}

fn kind_len(kind: DenseOperand, n: usize) -> usize {
    match kind {
        DenseOperand::Full => n,
        DenseOperand::Repeat(m) => m,
    }
}

fn binary_broadcast_typed<T: NumericElem>(
    op: BinOp,
    inputs: &[TensorView],
    output: &mut TensorMut,
    n: usize,
    (lhs_len, rhs_len): (usize, usize),
) {
    // SAFETY: the caller proved both inputs are contiguous with a shape that is
    // a right-aligned suffix of the output shape, so each pointer spans exactly
    // `lhs_len`/`rhs_len` `T`s; the output is contiguous over `n` `T`s; dtypes
    // match; and no input overlaps the output.
    let lhs = unsafe { std::slice::from_raw_parts(inputs[0].data_ptr::<T>(), lhs_len) };
    let rhs = unsafe { std::slice::from_raw_parts(inputs[1].data_ptr::<T>(), rhs_len) };
    let out = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<T>(), n) };
    if try_half_binary::<T>(op, lhs, rhs, out) {
        return;
    }

    // `BinOp::apply` matches on a runtime value. Left inside the element loop
    // that match is re-evaluated per element and blocks auto-vectorisation
    // entirely (measured: 1.1 ns/element for an f32 multiply, ~8x off ORT).
    // Resolving it once here lets each arm monomorphise to a straight-line
    // loop over a single arithmetic op.
    dispatch_binop!(op, fold => broadcast_loops::<T, _>(fold, lhs, rhs, out));
}

/// The four broadcast shapes, generic over an already-resolved combiner.
#[inline(always)]
fn broadcast_loops<T: NumericElem, F: Fn(T::Acc, T::Acc) -> T::Acc>(
    fold: F,
    lhs: &[T],
    rhs: &[T],
    out: &mut [T],
) {
    let n = out.len();
    match (lhs.len(), rhs.len()) {
        // `x op scalar` -- the dominant case; hoisting the operand out of the
        // loop is what lets this vectorise.
        (l, 1) if l == n => {
            let r = rhs[0].to_acc();
            for (out, &lhs) in out.iter_mut().zip(lhs) {
                *out = T::from_acc(fold(lhs.to_acc(), r));
            }
        }
        (1, r) if r == n => {
            let l = lhs[0].to_acc();
            for (out, &rhs) in out.iter_mut().zip(rhs) {
                *out = T::from_acc(fold(l, rhs.to_acc()));
            }
        }
        // `[.., C] op [C]` -- walk whole rows so the repeated operand stays in
        // L1 and neither index needs a division.
        (l, r) if l == n && r != 0 => {
            for (row, out) in out.chunks_mut(r).enumerate() {
                let lhs = &lhs[row * r..row * r + out.len()];
                for ((out, &lhs), &rhs) in out.iter_mut().zip(lhs).zip(rhs) {
                    *out = T::from_acc(fold(lhs.to_acc(), rhs.to_acc()));
                }
            }
        }
        (l, r) if r == n && l != 0 => {
            for (row, out) in out.chunks_mut(l).enumerate() {
                let rhs = &rhs[row * l..row * l + out.len()];
                for ((out, &lhs), &rhs) in out.iter_mut().zip(lhs).zip(rhs) {
                    *out = T::from_acc(fold(lhs.to_acc(), rhs.to_acc()));
                }
            }
        }
        // Both operands repeat (e.g. `[4, 3] <- [3] op [3]`). Rare; still far
        // cheaper than the strided walk.
        (l, r) => {
            for (i, out) in out.iter_mut().enumerate() {
                *out = T::from_acc(fold(lhs[i % l].to_acc(), rhs[i % r].to_acc()));
            }
        }
    }
}

/// `Add` shares this primitive: `BinOp::Sum` folds with `c_add`, which is
/// exactly what `add_typed` computes, with the same `to_acc`/`from_acc`
/// rounding. Exposed so `add.rs` does not grow a second copy of the walk.
pub(super) fn add_dense_fast_path(inputs: &[TensorView], output: &mut TensorMut) -> bool {
    binary_contiguous(BinOp::Sum, inputs, output)
        || binary_broadcast_contiguous(BinOp::Sum, inputs, output)
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
    /// Structural FLOPs (one op per element) when the input shape was static at
    /// build time; `None` otherwise (issue #995 — never fabricated).
    flops: Option<u64>,
}

macro_rules! unary_factory {
    ($factory:ident, $variant:expr) => {
        /// Factory (no attributes).
        pub struct $factory;
        impl KernelFactory for $factory {
            fn create(&self, _node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
                Ok(Box::new(UnaryKernel {
                    op: $variant,
                    flops: super::flops::elementwise_flops(shapes),
                }))
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
        // Vectorised f32-domain fast path. `Float64` is excluded because it
        // would round-trip through f32 and lose precision; `Erf` keeps its
        // exact scalar form (see `kernels::simd_activations` for why `Erf` is
        // deliberately not approximated).
        //
        // `Sqrt` joins `Tanh` here for a different reason than the usual
        // speed-vs-accuracy trade: `vsqrtps` is the same correctly-rounded IEEE
        // square root as the `sqrtss` it replaces, so this is pure throughput.
        // The old `dispatch_float!`/`unary_typed` arm was the cost — it widened
        // element by element through the `FloatElem` trait, `collect()`ed a
        // whole `Vec<T>`, then copied that into the output, and the per-element
        // `match` on the runtime `UnOp` blocked vectorisation of even the plain
        // f32 case.
        if matches!(op, UnOp::Tanh | UnOp::Sqrt) && inputs[0].dtype != DataType::Float64 {
            let x = to_dense_f32_widen(op.name(), &inputs[0])?;
            return simd_activations::write_mapped(
                op.name(),
                &mut outputs[0],
                &x,
                |x, y| match op {
                    UnOp::Tanh => simd_activations::tanh_f32_slice(x, y),
                    UnOp::Sqrt => simd_activations::sqrt_f32_slice(x, y),
                    // Unreachable today (the gate above only admits the two
                    // arms), but a scalar fallback beats a panic if the gate
                    // and this match ever drift apart.
                    UnOp::Erf => {
                        for (o, v) in y.iter_mut().zip(x) {
                            *o = op.apply(*v);
                        }
                    }
                },
            );
        }
        dispatch_float!(inputs[0].dtype, op.name(), T => unary_typed::<T>(op, inputs, outputs))
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn can_run_in_place(&self, input_index: usize) -> bool {
        input_index == 0
    }

    fn estimated_flops(&self) -> Option<u64> {
        self.flops
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
        BinaryKernel { op: f, flops: None }
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
        BinaryKernel {
            op: BinOp::Pow,
            flops: None,
        }
        .execute(&[base.view(), exponent.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![8., 9.]);

        let base = Owned::i32(&[2], &[2, 3]);
        let exponent = Owned::f32(&[2], &[3., 2.]);
        let mut out = Owned::zeros(DataType::Int32, &[2]);
        BinaryKernel {
            op: BinOp::Pow,
            flops: None,
        }
        .execute(&[base.view(), exponent.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_i32(), vec![8, 9]);
    }

    #[test]
    fn pow_covers_integer_base_and_exponent_combinations() {
        let base = Owned::i32(&[2], &[2, 3]);
        let exponent = Owned::i32(&[2], &[3, 2]);
        let mut out = Owned::zeros(DataType::Int32, &[2]);
        BinaryKernel {
            op: BinOp::Pow,
            flops: None,
        }
        .execute(&[base.view(), exponent.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_i32(), vec![8, 9]);

        let base = Owned::i64(&[3], &[2, -3, 4]);
        let exponent = Owned::i64(&[3], &[3, 2, 0]);
        let mut out = Owned::zeros(DataType::Int64, &[3]);
        BinaryKernel {
            op: BinOp::Pow,
            flops: None,
        }
        .execute(&[base.view(), exponent.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_i64(), vec![8, 9, 1]);
    }

    #[test]
    fn pow_accepts_float_exponents_for_i64_base() {
        let base = Owned::i64(&[2], &[2, 3]);
        let exponent = Owned::f32(&[2], &[3., 2.]);
        let mut out = Owned::zeros(DataType::Int64, &[2]);
        BinaryKernel {
            op: BinOp::Pow,
            flops: None,
        }
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
        BinaryKernel {
            op: BinOp::Min,
            flops: None,
        }
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
        BinaryKernel {
            op: BinOp::Max,
            flops: None,
        }
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
        BinaryKernel {
            op: BinOp::Sum,
            flops: None,
        }
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
        BinaryKernel {
            op: BinOp::Mean,
            flops: None,
        }
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
        let error = BinaryKernel {
            op: BinOp::Sum,
            flops: None,
        }
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
        UnaryKernel {
            op: UnOp::Sqrt,
            flops: None,
        }
        .execute(&[a.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![2., 3., 4.]);
    }

    #[test]
    fn unary_advertises_safe_in_place_input() {
        let kernel = UnaryKernel {
            op: UnOp::Tanh,
            flops: None,
        };
        assert!(kernel.can_run_in_place(0));
        assert!(!kernel.can_run_in_place(1));
    }

    #[test]
    fn tanh_known_values() {
        let a = Owned::f32(&[3], &[0., 1., -1.]);
        let mut out = Owned::zeros_f32(&[3]);
        UnaryKernel {
            op: UnOp::Tanh,
            flops: None,
        }
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
        UnaryKernel {
            op: UnOp::Erf,
            flops: None,
        }
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
        let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
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
        BinaryKernel {
            op: BinOp::Mul,
            flops: None,
        }
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
        BinaryKernel {
            op: BinOp::Sub,
            flops: None,
        }
        .execute(&[a.view(), b.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_bf16_as_f32(), vec![9., 18., 27., 36.]);
    }

    #[test]
    fn mul_f16_contiguous_matches_broadcast_path() {
        // Same-shape contiguous inputs take the `binary_contiguous` fast path,
        // which must be byte-identical to the general broadcast_apply path (same
        // to_acc/from_acc f16->f32->f16 rounding). Values with non-trivial f16
        // rounding exercise the round-trip.
        // 61 deliberately leaves a remainder for any common SIMD lane width.
        let lhs: Vec<f32> = (0..61).map(|i| (i as f32) * 0.3 - 5.0).collect();
        let rhs: Vec<f32> = (0..61).map(|i| 1.0 / (i as f32 + 1.7)).collect();
        let a = Owned::f16(&[61], &lhs);
        let b = Owned::f16(&[61], &rhs);
        let mut fast = Owned::zeros(DataType::Float16, &[61]);
        BinaryKernel {
            op: BinOp::Mul,
            flops: None,
        }
        .execute(&[a.view(), b.view()], &mut [fast.view_mut()])
        .unwrap();
        // Reference: reproduce the general per-element f16 compute directly.
        let want: Vec<f32> = lhs
            .iter()
            .zip(&rhs)
            .map(|(&x, &y)| {
                half::f16::from_f32(
                    half::f16::from_f32(x).to_f32() * half::f16::from_f32(y).to_f32(),
                )
                .to_f32()
            })
            .collect();
        assert_eq!(fast.to_f16_as_f32(), want);

        // Force the general broadcast fallback by reshaping one operand so the
        // contiguous fast path is bypassed (shape [61,1] x [1,1] broadcasts a
        // scalar over the column). The fast path and broadcast path must agree
        // bit-for-bit on the equal-shape region, so compare RAW f16 bits.
        let a_col = Owned::f16(&[61, 1], &lhs);
        let b_scalar = Owned::f16(&[1, 1], &[rhs[0]]);
        let mut broadcast = Owned::zeros(DataType::Float16, &[61, 1]);
        BinaryKernel {
            op: BinOp::Mul,
            flops: None,
        }
        .execute(
            &[a_col.view(), b_scalar.view()],
            &mut [broadcast.view_mut()],
        )
        .unwrap();
        // Recompute the fast path with the same scalar multiplier as a full
        // [61] contiguous op, then compare raw bits against the broadcast result.
        let rhs_scalar = vec![rhs[0]; 61];
        let a_flat = Owned::f16(&[61], &lhs);
        let b_flat = Owned::f16(&[61], &rhs_scalar);
        let mut fast_scalar = Owned::zeros(DataType::Float16, &[61]);
        BinaryKernel {
            op: BinOp::Mul,
            flops: None,
        }
        .execute(
            &[a_flat.view(), b_flat.view()],
            &mut [fast_scalar.view_mut()],
        )
        .unwrap();
        assert_eq!(
            fast_scalar.to_u16_bits(),
            broadcast.to_u16_bits(),
            "contiguous fast path and broadcast fallback must be bit-identical"
        );
    }

    #[test]
    fn sub_div_f16_contiguous_matches_broadcast_path() {
        // Cover the generalized contiguous fast path for Sub and Div in f16,
        // asserting bit-identity with the broadcast fallback (Gaff nit).
        // 53 exercises the contiguous loop's remainder path.
        let lhs: Vec<f32> = (0..53).map(|i| (i as f32) * 0.25 - 6.0).collect();
        let rhs: Vec<f32> = (0..53).map(|i| (i as f32) * 0.1 + 1.3).collect();
        for op in [BinOp::Sub, BinOp::Div] {
            let a = Owned::f16(&[53], &lhs);
            let b = Owned::f16(&[53], &rhs);
            let mut fast = Owned::zeros(DataType::Float16, &[53]);
            BinaryKernel { op, flops: None }
                .execute(&[a.view(), b.view()], &mut [fast.view_mut()])
                .unwrap();

            // Broadcast fallback over [53,1] x [1,1] with the first rhs value.
            let a_col = Owned::f16(&[53, 1], &lhs);
            let b_scalar = Owned::f16(&[1, 1], &[rhs[0]]);
            let mut broadcast = Owned::zeros(DataType::Float16, &[53, 1]);
            BinaryKernel { op, flops: None }
                .execute(
                    &[a_col.view(), b_scalar.view()],
                    &mut [broadcast.view_mut()],
                )
                .unwrap();
            let rhs_scalar = vec![rhs[0]; 53];
            let b_flat = Owned::f16(&[53], &rhs_scalar);
            let a_flat = Owned::f16(&[53], &lhs);
            let mut fast_scalar = Owned::zeros(DataType::Float16, &[53]);
            BinaryKernel { op, flops: None }
                .execute(
                    &[a_flat.view(), b_flat.view()],
                    &mut [fast_scalar.view_mut()],
                )
                .unwrap();
            assert_eq!(
                fast_scalar.to_u16_bits(),
                broadcast.to_u16_bits(),
                "contiguous and broadcast f16 paths must be bit-identical"
            );
        }
    }

    #[test]
    fn div_int32_truncates_and_guards_zero() {
        // Integer Div is truncating; divide-by-zero yields 0 (not a panic).
        let a = Owned::i32(&[3], &[7, -7, 5]);
        let b = Owned::i32(&[3], &[2, 2, 0]);
        let mut out = Owned::zeros(DataType::Int32, &[3]);
        BinaryKernel {
            op: BinOp::Div,
            flops: None,
        }
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
        BinaryKernel {
            op: BinOp::Min,
            flops: None,
        }
        .execute(&[a.view(), b.view()], &mut [mn.view_mut()])
        .unwrap();
        BinaryKernel {
            op: BinOp::Max,
            flops: None,
        }
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
        UnaryKernel {
            op: UnOp::Sqrt,
            flops: None,
        }
        .execute(&[a16.view()], &mut [o16.view_mut()])
        .unwrap();
        assert_eq!(o16.to_f16_as_f32(), vec![2., 3., 4.]);

        let ab = Owned::bf16(&[3], &[4., 9., 16.]);
        let mut ob = Owned::zeros(DataType::BFloat16, &[3]);
        UnaryKernel {
            op: UnOp::Sqrt,
            flops: None,
        }
        .execute(&[ab.view()], &mut [ob.view_mut()])
        .unwrap();
        assert_eq!(ob.to_bf16_as_f32(), vec![2., 3., 4.]);
    }

    #[test]
    fn tanh_f16_matches_f32_within_tolerance() {
        let a = Owned::f16(&[3], &[0., 1., -1.]);
        let mut out = Owned::zeros(DataType::Float16, &[3]);
        UnaryKernel {
            op: UnOp::Tanh,
            flops: None,
        }
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
        UnaryKernel {
            op: UnOp::Erf,
            flops: None,
        }
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
        let err = UnaryKernel {
            op: UnOp::Sqrt,
            flops: None,
        }
        .execute(&[a.view()], &mut [out.view_mut()])
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("WHAT") && msg.contains("HOW"));
    }
}

/// The dense broadcast fast path must be a *pure* optimisation of the general
/// [`broadcast_apply`] walk.
///
/// Each case is run twice over the **same logical values**: once with a
/// contiguous broadcast operand (which takes the fast path) and once with a
/// strided view of padded storage holding the same elements (which
/// `dense_operand` rejects, so it falls through to the general path). The two
/// results are compared **bitwise**, so a difference in `to_acc`/`from_acc`
/// rounding, in NaN payload, or in signed zero cannot be hidden by `==` on
/// floats.
#[cfg(test)]
mod dense_broadcast_equivalence_tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    /// Interleave `data` with padding and expose it through stride 2, so the
    /// view holds exactly `data` but is not contiguous.
    fn strided_f32(shape: &[usize], data: &[f32]) -> Owned {
        let mut padded = Vec::with_capacity(data.len() * 2);
        for &v in data {
            padded.push(v);
            padded.push(f32::NAN);
        }
        let strides: Vec<i64> = {
            let mut s = onnx_runtime_ir::compute_contiguous_strides(shape);
            for v in &mut s {
                *v *= 2;
            }
            s
        };
        Owned::f32(&[data.len() * 2], &padded).with_view(shape, &strides)
    }

    fn strided_f16(shape: &[usize], data: &[f32]) -> Owned {
        let mut padded = Vec::with_capacity(data.len() * 2);
        for &v in data {
            padded.push(v);
            padded.push(f32::NAN);
        }
        let strides: Vec<i64> = {
            let mut s = onnx_runtime_ir::compute_contiguous_strides(shape);
            for v in &mut s {
                *v *= 2;
            }
            s
        };
        Owned::f16(&[data.len() * 2], &padded).with_view(shape, &strides)
    }

    fn run(op: BinOp, l: &Owned, r: &Owned, out_shape: &[usize], dtype: DataType) -> Vec<u8> {
        let mut out = Owned::zeros(dtype, out_shape);
        BinaryKernel { op, flops: None }
            .execute(&[l.view(), r.view()], &mut [out.view_mut()])
            .unwrap();
        out.bytes
    }

    /// Adversarial values: signed zeros, infinities, a NaN, subnormals and
    /// magnitudes that make `Div` overflow and underflow.
    fn payload(n: usize) -> Vec<f32> {
        let special = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            1e-30,
            -1e-30,
        ];
        (0..n)
            .map(|i| {
                if i < special.len() {
                    special[i]
                } else {
                    (i as f32 - (n as f32) / 2.0) * 0.75
                }
            })
            .collect()
    }

    #[test]
    fn f32_suffix_and_scalar_broadcast_match_the_strided_path_bitwise() {
        for op in [BinOp::Sub, BinOp::Mul, BinOp::Div] {
            for (out_shape, rhs_shape) in [
                (vec![64usize], vec![] as Vec<usize>),
                (vec![64], vec![1]),
                (vec![8, 8], vec![8]),
                (vec![4, 2, 8], vec![8]),
                (vec![4, 2, 8], vec![2, 8]),
                (vec![4, 2, 8], vec![1, 1, 8]),
                (vec![3, 5], vec![3, 5]),
            ] {
                let n: usize = out_shape.iter().product();
                let m: usize = rhs_shape.iter().product();
                let lhs = payload(n);
                let rhs = payload(m.max(1));
                let l = Owned::f32(&out_shape, &lhs);
                let fast = run(
                    op,
                    &l,
                    &Owned::f32(&rhs_shape, &rhs),
                    &out_shape,
                    DataType::Float32,
                );
                let slow = run(
                    op,
                    &l,
                    &strided_f32(&rhs_shape, &rhs),
                    &out_shape,
                    DataType::Float32,
                );
                assert_eq!(
                    fast,
                    slow,
                    "{:?} out={out_shape:?} rhs={rhs_shape:?}",
                    op.name()
                );

                // ... and with the broadcast operand on the left, which matters
                // because Sub and Div are not commutative.
                let fast = run(
                    op,
                    &Owned::f32(&rhs_shape, &rhs),
                    &l,
                    &out_shape,
                    DataType::Float32,
                );
                let slow = run(
                    op,
                    &strided_f32(&rhs_shape, &rhs),
                    &l,
                    &out_shape,
                    DataType::Float32,
                );
                assert_eq!(
                    fast,
                    slow,
                    "{:?} (lhs broadcast) out={out_shape:?} lhs={rhs_shape:?}",
                    op.name()
                );
            }
        }
    }

    /// f16 goes through `to_acc`/`from_acc`, so this pins that the fast path
    /// rounds through f32 exactly like the general walk, comparing raw bits.
    #[test]
    fn f16_broadcast_matches_the_strided_path_bitwise() {
        for op in [BinOp::Sub, BinOp::Mul, BinOp::Div] {
            for (out_shape, rhs_shape) in [
                (vec![64usize], vec![] as Vec<usize>),
                (vec![8, 8], vec![8]),
                (vec![4, 2, 8], vec![2, 8]),
            ] {
                let n: usize = out_shape.iter().product();
                let m: usize = rhs_shape.iter().product();
                let lhs = payload(n);
                let rhs = payload(m.max(1));
                let l = Owned::f16(&out_shape, &lhs);
                let fast = run(
                    op,
                    &l,
                    &Owned::f16(&rhs_shape, &rhs),
                    &out_shape,
                    DataType::Float16,
                );
                let slow = run(
                    op,
                    &l,
                    &strided_f16(&rhs_shape, &rhs),
                    &out_shape,
                    DataType::Float16,
                );
                assert_eq!(fast, slow, "f16 {:?} out={out_shape:?}", op.name());
            }
        }
    }

    /// An interior size-1 axis is not a suffix repeat, so it must decline to the
    /// general walk -- otherwise the fast path would silently produce the wrong
    /// broadcast.
    #[test]
    fn interior_unit_axis_is_declined_and_still_broadcasts_correctly() {
        let l = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let r = Owned::f32(&[2, 1], &[10., 20.]);
        assert!(dense_operand(&r.view(), &[2, 3]).is_none());
        let mut out = Owned::zeros_f32(&[2, 3]);
        BinaryKernel {
            op: BinOp::Mul,
            flops: None,
        }
        .execute(&[l.view(), r.view()], &mut [out.view_mut()])
        .unwrap();
        assert_eq!(out.to_f32(), vec![10., 20., 30., 80., 100., 120.]);
    }

    /// A mismatched trailing extent must not be mistaken for a repeat.
    #[test]
    fn non_suffix_shapes_are_declined() {
        let r = Owned::f32(&[4], &[1., 2., 3., 4.]);
        assert!(dense_operand(&r.view(), &[2, 3]).is_none());
        let r = Owned::f32(&[2, 3, 4], &[0.; 24]);
        assert!(dense_operand(&r.view(), &[3, 4]).is_none());
    }
}

/// The staged f16/bf16 binary path must be a pure speed-up of the scalar loop
/// it replaces: same `to_acc`, same combiner, same `from_acc`, therefore the
/// same *bits*. These compare against that scalar formula directly rather than
/// against approximate expectations, and use adversarial inputs (signed zero,
/// ±inf, quiet and signalling NaN, both subnormal edges, f16 overflow, and
/// values engineered to land on an f16 rounding tie) so a rounding or
/// NaN-handling difference cannot hide behind float equality.
#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod half_binary_staging_tests {
    use super::*;

    const OPS: [BinOp; 6] = [
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Min,
        BinOp::Max,
        BinOp::Sum,
    ];

    /// Deterministic xorshift so the sweep is reproducible in CI.
    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
    }

    /// f32 values chosen to stress every part of the 16-bit round trip.
    fn adversarial_f32(count: usize) -> Vec<f32> {
        let mut v = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001), // signalling NaN
            f32::from_bits(0xff80_0001), // negative signalling NaN
            65504.0,                     // f16::MAX
            -65504.0,
            65520.0,                     // rounds to f16 infinity
            6.097e-5,                    // f16 smallest normal
            f32::from_bits(0x3380_0000), // 2^-24: f16 smallest subnormal
            f32::from_bits(0x3300_0000), // 2^-25: an exact tie to zero
            f32::from_bits(0x33c0_0000), // 1.5 * 2^-24: an exact tie away from zero
            3.4028235e38,
            -3.4028235e38,
            1.1754944e-38, // f32 smallest normal
            1e-45,         // f32 subnormal
        ];
        let mut rng = Rng(0x9e37_79b9);
        while v.len() < count {
            // Mostly in-range magnitudes so the arithmetic itself is meaningful,
            // with a slice of raw bit patterns for the pathological tail.
            let r = rng.next();
            v.push(if r.is_multiple_of(8) {
                f32::from_bits(rng.next())
            } else {
                ((r as f64 / u32::MAX as f64) as f32 - 0.5) * 128.0
            });
        }
        v.truncate(count);
        v
    }

    /// Bit-exact reference: exactly what `binary_contiguous_typed` /
    /// `broadcast_loops` compute element by element.
    fn reference<T: NumericElem<Acc = f32>>(op: BinOp, lhs: &[T], rhs: &[T], n: usize) -> Vec<u16> {
        (0..n)
            .map(|i| {
                let a = lhs[i % lhs.len()].to_acc();
                let b = rhs[i % rhs.len()].to_acc();
                T::from_acc(op.apply(a, b))
            })
            .map(|t| {
                // `T` is a `repr(transparent)` newtype over `u16`.
                let mut bits = 0u16;
                // SAFETY: `size_of::<T>() == size_of::<u16>()`, asserted below.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        std::ptr::from_ref(&t).cast::<u16>(),
                        &mut bits,
                        1,
                    )
                };
                bits
            })
            .collect()
    }

    fn bits_of<T>(v: &[T]) -> &[u16] {
        assert_eq!(std::mem::size_of::<T>(), 2);
        // SAFETY: caller only passes `half::f16`/`half::bf16`.
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u16>(), v.len()) }
    }

    fn check<T: NumericElem<Acc = f32> + Copy>(label: &str, lhs: &[T], rhs: &[T], n: usize) {
        for op in OPS {
            let want = reference::<T>(op, lhs, rhs, n);
            let mut got = vec![T::from_acc(0.0); n];
            assert!(
                try_half_binary::<T>(op, lhs, rhs, &mut got),
                "{label}: staged 16-bit path declined {}",
                op.name()
            );
            let got_bits = bits_of(&got);
            for i in 0..n {
                let (a, b) = (lhs[i % lhs.len()].to_acc(), rhs[i % rhs.len()].to_acc());
                if a.is_nan() && b.is_nan() {
                    // Which NaN payload survives is not fixed: see
                    // `nan_payload_choice_only_differs_when_both_operands_are_nan`.
                    assert!(
                        got[i].to_acc().is_nan(),
                        "{label}: {} lost NaN at i={i}",
                        op.name()
                    );
                    continue;
                }
                assert_eq!(
                    got_bits[i],
                    want[i],
                    "{label}: {} diverged at i={i} (n={n}, |lhs|={}, |rhs|={}): \
                     lhs={:#06x} rhs={:#06x} got={:#06x} want={:#06x}",
                    op.name(),
                    lhs.len(),
                    rhs.len(),
                    bits_of(lhs)[i % lhs.len()],
                    bits_of(rhs)[i % rhs.len()],
                    got_bits[i],
                    want[i]
                );
            }
        }
    }

    /// Lengths that straddle `HALF_STAGE_CHUNK` in both directions, plus repeat
    /// blocks that do not divide the chunk so `HalfSide::stage`'s wrap-around
    /// is exercised on a boundary rather than only at offset 0.
    const LENS: [usize; 9] = [1, 2, 7, 1023, 1024, 1025, 2048, 3000, 4097];
    const REPEATS: [usize; 6] = [1, 3, 7, 512, 1000, 1024];

    #[test]
    fn f16_staged_binary_matches_the_scalar_path_bitwise() {
        if !HalfKind::F16.available() {
            return;
        }
        for n in LENS {
            let src = adversarial_f32(n * 2);
            let a: Vec<half::f16> = src[..n].iter().map(|&f| half::f16::from_f32(f)).collect();
            let b: Vec<half::f16> = src[n..].iter().map(|&f| half::f16::from_f32(f)).collect();
            check("f16 same-shape", &a, &b, n);
            check("f16 scalar rhs", &a, &b[..1], n);
            check("f16 scalar lhs", &a[..1], &b, n);
            for r in REPEATS {
                if r > n {
                    continue;
                }
                check("f16 suffix rhs", &a, &b[..r], n);
                check("f16 suffix lhs", &a[..r], &b, n);
            }
            // Both operands repeat: the modulo arm of `broadcast_loops`.
            if n >= 7 {
                check("f16 both repeat", &a[..7], &b[..3], n);
            }
        }
    }

    #[test]
    fn bf16_staged_binary_matches_the_scalar_path_bitwise() {
        if !HalfKind::Bf16.available() {
            return;
        }
        for n in LENS {
            let src = adversarial_f32(n * 2);
            let a: Vec<half::bf16> = src[..n].iter().map(|&f| half::bf16::from_f32(f)).collect();
            let b: Vec<half::bf16> = src[n..].iter().map(|&f| half::bf16::from_f32(f)).collect();
            check("bf16 same-shape", &a, &b, n);
            check("bf16 scalar rhs", &a, &b[..1], n);
            check("bf16 scalar lhs", &a[..1], &b, n);
            for r in REPEATS {
                if r > n {
                    continue;
                }
                check("bf16 suffix rhs", &a, &b[..r], n);
            }
            if n >= 7 {
                check("bf16 both repeat", &a[..7], &b[..3], n);
            }
        }
    }

    /// Every bf16 bit pattern on the left against a fixed adversarial right
    /// operand — including signalling NaNs, which is what pins
    /// `widen_quieting` rather than the raw shift widen.
    #[test]
    fn bf16_exhaustive_left_operand_matches_the_scalar_path() {
        if !HalfKind::Bf16.available() {
            return;
        }
        let a: Vec<half::bf16> = (0..=u16::MAX).map(half::bf16::from_bits).collect();
        let n = a.len();
        let src = adversarial_f32(n);
        let b: Vec<half::bf16> = src.iter().map(|&f| half::bf16::from_f32(f)).collect();
        check("bf16 exhaustive", &a, &b, n);
        check("bf16 exhaustive scalar rhs", &a, &b[..1], n);
    }

    /// Same sweep for f16.
    #[test]
    fn f16_exhaustive_left_operand_matches_the_scalar_path() {
        if !HalfKind::F16.available() {
            return;
        }
        let a: Vec<half::f16> = (0..=u16::MAX).map(half::f16::from_bits).collect();
        let n = a.len();
        let src = adversarial_f32(n);
        let b: Vec<half::f16> = src.iter().map(|&f| half::f16::from_f32(f)).collect();
        check("f16 exhaustive", &a, &b, n);
        check("f16 exhaustive scalar rhs", &a, &b[..1], n);
    }

    /// Non-16-bit dtypes must not be diverted, and the output must be left
    /// untouched when the path declines.
    #[test]
    fn wider_dtypes_are_declined_untouched() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let mut out = [f32::NAN; 3];
        assert!(!try_half_binary::<f32>(BinOp::Mul, &a, &b, &mut out));
        assert!(out.iter().all(|v| v.is_nan()));

        let a = [1.0f64, 2.0];
        let b = [4.0f64, 5.0];
        let mut out = [f64::NAN; 2];
        assert!(!try_half_binary::<f64>(BinOp::Mul, &a, &b, &mut out));
        assert!(out.iter().all(|v| v.is_nan()));
    }

    /// A zero-length operand against a non-empty output is declined rather than
    /// dividing by zero in the repeat modulo.
    #[test]
    fn empty_operand_against_non_empty_output_is_declined() {
        let a: [half::f16; 0] = [];
        let b = [half::f16::from_f32(1.0)];
        let mut out = [half::f16::ZERO; 4];
        assert!(!try_half_binary::<half::f16>(BinOp::Mul, &a, &b, &mut out));
        // Empty output is fine either way; it must not panic.
        let mut empty: [half::f16; 0] = [];
        let _ = try_half_binary::<half::f16>(BinOp::Mul, &a, &a, &mut empty);
    }

    /// End-to-end through the kernel, so the staged path is proven reachable
    /// from `BinaryKernel`/`AddKernel` and not just callable in isolation.
    #[test]
    fn kernel_level_f16_broadcast_matches_the_reference() {
        use crate::kernels::testutil::Owned;
        let n = 4096usize;
        let src = adversarial_f32(n);
        let a = Owned::f16(&[n], &src);
        let k = Owned::f16(&[], &[0.75]);
        let a_h: Vec<half::f16> = src.iter().map(|&f| half::f16::from_f32(f)).collect();
        let k_h = [half::f16::from_f32(0.75)];

        for op in [BinOp::Mul, BinOp::Sub, BinOp::Div] {
            let kernel = BinaryKernel { op, flops: None };
            let mut out = Owned::zeros(DataType::Float16, &[n]);
            kernel
                .execute(&[a.view(), k.view()], &mut [out.view_mut()])
                .unwrap();
            assert_eq!(
                out.to_u16_bits(),
                reference::<half::f16>(op, &a_h, &k_h, n),
                "{} f16 scalar broadcast diverged end to end",
                op.name()
            );
        }
    }

    /// The one place the staged path is *not* bit-identical to a naive scalar
    /// loop, pinned rather than papered over.
    ///
    /// Vectorising a commutative fold lets the compiler pick either operand
    /// order, and x86 multiply/add returns the **first** NaN operand quieted, so
    /// when *both* operands are NaN the surviving payload can be the right one
    /// instead of the left one. IEEE-754 and ONNX both leave NaN payload
    /// propagation unspecified, and the already-merged `f32` dense loop has
    /// exactly the same property, so this is shared behaviour of the dense
    /// paths, not something the 16-bit staging introduces.
    ///
    /// The test asserts the two halves of that claim: (1) for `f16` the
    /// divergence happens **only** when both operands are NaN, for the two
    /// commutative ops, and the result is always still NaN; (2) the `f32` dense
    /// path already commutes NaN operands the same way.
    #[test]
    fn nan_payload_choice_only_differs_when_both_operands_are_nan() {
        if !HalfKind::F16.available() {
            return;
        }
        let lhs: Vec<half::f16> = (0..=u16::MAX).map(half::f16::from_bits).collect();
        let mut rng = Rng(0x1234_5678);
        let rhs: Vec<half::f16> = (0..=u16::MAX)
            .map(|_| half::f16::from_bits((rng.next() >> 8) as u16))
            .collect();
        let n = lhs.len();
        for op in OPS {
            let mut got = vec![half::f16::ZERO; n];
            assert!(try_half_binary::<half::f16>(op, &lhs, &rhs, &mut got));
            let want = reference::<half::f16>(op, &lhs, &rhs, n);
            let mut both_nan_diffs = 0;
            for i in 0..n {
                if got[i].to_bits() == want[i] {
                    continue;
                }
                assert!(
                    lhs[i].is_nan() && rhs[i].is_nan(),
                    "{} diverged at i={i} without two NaN operands: lhs={:#06x} rhs={:#06x}",
                    op.name(),
                    lhs[i].to_bits(),
                    rhs[i].to_bits()
                );
                assert!(
                    got[i].is_nan(),
                    "{} turned NaN op NaN into a number",
                    op.name()
                );
                both_nan_diffs += 1;
            }
            // Only the commutative folds can be reassociated this way.
            if !matches!(op, BinOp::Mul | BinOp::Sum) {
                assert_eq!(
                    both_nan_diffs,
                    0,
                    "{} is not commutative and must be bit-exact everywhere",
                    op.name()
                );
            }
        }
    }

    /// Half of the claim above: the `f32` dense path, which is already on
    /// `main`, is subject to exactly the same NaN-payload freedom. Which
    /// payload survives depends on whether the loop was vectorised (it is in
    /// release, is not in debug), so the assertion is the *contract* — the
    /// result is a NaN carrying one of the two operand payloads, quieted —
    /// rather than a specific bit pattern.
    #[test]
    fn f32_dense_path_propagates_one_of_the_two_nan_payloads() {
        use crate::kernels::testutil::Owned;
        let x = f32::from_bits(0xffc0_0003);
        let y = f32::from_bits(0xff80_0007);
        let a = Owned::f32(&[8], &[x; 8]);
        let b = Owned::f32(&[8], &[y; 8]);
        let mut out = Owned::zeros_f32(&[8]);
        BinaryKernel {
            op: BinOp::Mul,
            flops: None,
        }
        .execute(&[a.view(), b.view()], &mut [out.view_mut()])
        .unwrap();
        for got in out.to_f32() {
            assert!(got.is_nan(), "f32 dense Mul turned NaN * NaN into a number");
            let bits = got.to_bits();
            // Quieted x, or quieted y.
            assert!(
                bits == x.to_bits() | 0x0040_0000 || bits == y.to_bits() | 0x0040_0000,
                "f32 dense Mul invented a NaN payload: {bits:#010x}"
            );
        }
    }
}
