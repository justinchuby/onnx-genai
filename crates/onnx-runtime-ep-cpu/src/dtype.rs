//! Reusable dtype-generic machinery for the arithmetic CPU kernels
//! (`docs/ORT2.md` §4.4; project preference "不同的dtype，是不是可以用 template").
//!
//! The Phase-1 kernels were originally f32-only, which turned every ONNX
//! `float16` / `bfloat16` / integer conformance case into a spurious failure —
//! the numeric op was correct, the kernel simply refused the element type. This
//! module supplies the *one* dtype-dispatch mechanism the arithmetic kernels
//! share so multi-dtype support is written once, not copy-pasted per dtype.
//!
//! ## The three moving parts
//!
//! * [`ComputeDomain`] — the numeric domain arithmetic is *evaluated* in. Low-
//!   and medium-precision floats (`f16`/`bf16`/`f32`) compute in `f32`; `f64`
//!   computes in `f64`; each integer width computes in itself (wrapping,
//!   C-style). This is where the actual `+`/`-`/`min`/NaN-propagation semantics
//!   live, once.
//! * [`NumericElem`] — a *storage* element type (what a tensor view addresses)
//!   plus how it widens to / narrows from its [`ComputeDomain`]. `f16`/`bf16`
//!   store as 2-byte LE and round-trip through `f32` for compute (standard),
//!   never reinterpreted as `f32` bits.
//! * [`FloatElem`] — the float-only subset used by unary transcendental kernels
//!   (`Sqrt`, `Tanh`, `Erf`, …) and by the MatMul/Gemm f32-accumulate path.
//!
//! ## The dispatch macros
//!
//! [`dispatch_arith`] and [`dispatch_float`] map a runtime [`DataType`] to a
//! monomorphized generic body over the matching Rust element type, and emit a
//! RULE #1 (`WHAT`/`WHY`/`HOW`) error for any dtype the op is not defined over —
//! we never fabricate support for a type ONNX does not define the op on.
//!
//! New kernels get multi-dtype for free: read with [`to_dense`], compute in the
//! element's [`NumericElem::Acc`], write with [`write_dense`], and wrap the body
//! in the appropriate `dispatch_*` macro.

use onnx_runtime_ep_api::{EpError, Result, TensorMut, TensorView};
use onnx_runtime_ir::DataType;

use crate::strided::{elem_offset, next_index, numel};

/// The numeric domain a kernel evaluates arithmetic in.
///
/// Kept separate from the storage element type so several storage widths can
/// share a single arithmetic implementation (all of `f16`/`bf16`/`f32` fold in
/// `f32`) and so the delicate semantics — NaN-propagating `min`/`max`, integer
/// wrapping, integer divide-by-zero → 0 — are written exactly once.
pub trait ComputeDomain: Copy + Default {
    fn c_add(self, o: Self) -> Self;
    fn c_sub(self, o: Self) -> Self;
    fn c_mul(self, o: Self) -> Self;
    fn c_div(self, o: Self) -> Self;
    fn c_pow(self, o: Self) -> Self;
    fn c_div_usize(self, divisor: usize) -> Self;
    /// ONNX/numpy `Min`: NaN-propagating for floats, `Ord::min` for integers.
    fn c_min(self, o: Self) -> Self;
    /// ONNX/numpy `Max`: NaN-propagating for floats, `Ord::max` for integers.
    fn c_max(self, o: Self) -> Self;
}

macro_rules! impl_float_compute {
    ($($t:ty),*) => {$(
        impl ComputeDomain for $t {
            #[inline] fn c_add(self, o: Self) -> Self { self + o }
            #[inline] fn c_sub(self, o: Self) -> Self { self - o }
            #[inline] fn c_mul(self, o: Self) -> Self { self * o }
            #[inline] fn c_div(self, o: Self) -> Self { self / o }
            #[inline] fn c_pow(self, o: Self) -> Self { self.powf(o) }
            #[inline] fn c_div_usize(self, divisor: usize) -> Self { self / divisor as $t }
            // Rust's `min`/`max` SUPPRESS NaN (return the non-NaN operand); ONNX
            // `Min`/`Max` PROPAGATE it (numpy semantics), so guard explicitly.
            #[inline] fn c_min(self, o: Self) -> Self {
                if self.is_nan() || o.is_nan() { <$t>::NAN } else { self.min(o) }
            }
            #[inline] fn c_max(self, o: Self) -> Self {
                if self.is_nan() || o.is_nan() { <$t>::NAN } else { self.max(o) }
            }
        }
    )*};
}
impl_float_compute!(f32, f64);

macro_rules! impl_int_compute {
    ($($t:ty),*) => {$(
        impl ComputeDomain for $t {
            // C-style wrapping arithmetic, matching ONNX Runtime's integer ops.
            #[inline] fn c_add(self, o: Self) -> Self { self.wrapping_add(o) }
            #[inline] fn c_sub(self, o: Self) -> Self { self.wrapping_sub(o) }
            #[inline] fn c_mul(self, o: Self) -> Self { self.wrapping_mul(o) }
            // Integer divide-by-zero is UB in ONNX; return 0 (numpy's result)
            // rather than panicking. `wrapping_div` also absorbs INT_MIN / -1.
            #[inline] fn c_div(self, o: Self) -> Self {
                if o == 0 { 0 } else { self.wrapping_div(o) }
            }
            // Integer Pow via f64 (exact for the magnitudes ONNX exercises);
            // negative exponents (fractional result) truncate toward zero.
            #[inline] fn c_pow(self, o: Self) -> Self { (self as f64).powf(o as f64) as $t }
            #[inline] fn c_div_usize(self, divisor: usize) -> Self {
                ((self as i128) / divisor as i128) as $t
            }
            #[inline] fn c_min(self, o: Self) -> Self { core::cmp::min(self, o) }
            #[inline] fn c_max(self, o: Self) -> Self { core::cmp::max(self, o) }
        }
    )*};
}
impl_int_compute!(i8, i16, i32, i64, u8, u16, u32, u64);

/// A tensor *storage* element type plus its widen/narrow to a [`ComputeDomain`].
///
/// # Safety-adjacent contract
/// [`DTYPE`](Self::DTYPE) MUST equal the [`DataType`] whose in-memory layout is
/// exactly `Self` (same `size_of`, native-endian bit pattern). The dispatch
/// macros bind the generic type to the matched dtype, upholding this so
/// [`to_dense`]/[`write_dense`] read/write the correct number of bytes.
pub trait NumericElem: Copy {
    /// The tensor dtype whose storage layout is exactly `Self`.
    const DTYPE: DataType;
    /// The domain this element's arithmetic is evaluated in.
    type Acc: ComputeDomain;
    fn to_acc(self) -> Self::Acc;
    fn from_acc(a: Self::Acc) -> Self;
}

/// The float-only subset (widens to / narrows from `f32`), used by unary
/// transcendental kernels and the MatMul/Gemm f32-accumulate path.
pub trait FloatElem: Copy {
    const DTYPE: DataType;
    fn to_f32(self) -> f32;
    fn from_f32(f: f32) -> Self;
}

// --- f32 -------------------------------------------------------------------
impl NumericElem for f32 {
    const DTYPE: DataType = DataType::Float32;
    type Acc = f32;
    #[inline]
    fn to_acc(self) -> f32 {
        self
    }
    #[inline]
    fn from_acc(a: f32) -> Self {
        a
    }
}
impl FloatElem for f32 {
    const DTYPE: DataType = DataType::Float32;
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn from_f32(f: f32) -> Self {
        f
    }
}

// --- f64 -------------------------------------------------------------------
impl NumericElem for f64 {
    const DTYPE: DataType = DataType::Float64;
    type Acc = f64;
    #[inline]
    fn to_acc(self) -> f64 {
        self
    }
    #[inline]
    fn from_acc(a: f64) -> Self {
        a
    }
}
impl FloatElem for f64 {
    const DTYPE: DataType = DataType::Float64;
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f32(f: f32) -> Self {
        f as f64
    }
}

// --- f16 / bf16 (2-byte LE storage; compute in f32) ------------------------
impl NumericElem for half::f16 {
    const DTYPE: DataType = DataType::Float16;
    type Acc = f32;
    #[inline]
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn from_acc(a: f32) -> Self {
        half::f16::from_f32(a)
    }
}
impl FloatElem for half::f16 {
    const DTYPE: DataType = DataType::Float16;
    #[inline]
    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }
    #[inline]
    fn from_f32(f: f32) -> Self {
        half::f16::from_f32(f)
    }
}
impl NumericElem for half::bf16 {
    const DTYPE: DataType = DataType::BFloat16;
    type Acc = f32;
    #[inline]
    fn to_acc(self) -> f32 {
        self.to_f32()
    }
    #[inline]
    fn from_acc(a: f32) -> Self {
        half::bf16::from_f32(a)
    }
}
impl FloatElem for half::bf16 {
    const DTYPE: DataType = DataType::BFloat16;
    #[inline]
    fn to_f32(self) -> f32 {
        half::bf16::to_f32(self)
    }
    #[inline]
    fn from_f32(f: f32) -> Self {
        half::bf16::from_f32(f)
    }
}

// --- integers (compute in themselves) --------------------------------------
macro_rules! impl_int_elem {
    ($($t:ty => $dt:expr),* $(,)?) => {$(
        impl NumericElem for $t {
            const DTYPE: DataType = $dt;
            type Acc = $t;
            #[inline] fn to_acc(self) -> $t { self }
            #[inline] fn from_acc(a: $t) -> Self { a }
        }
    )*};
}
impl_int_elem!(
    i8 => DataType::Int8,
    i16 => DataType::Int16,
    i32 => DataType::Int32,
    i64 => DataType::Int64,
    u8 => DataType::Uint8,
    u16 => DataType::Uint16,
    u32 => DataType::Uint32,
    u64 => DataType::Uint64,
);

/// Materialize a strided view of element type `T` into a dense, row-major
/// `Vec<T>`, applying the view's strides and byte offset.
///
/// `T::DTYPE` must match `view.dtype` (the dispatch macros guarantee this); the
/// debug assertion catches a mis-wired call site before it can read the wrong
/// element width.
pub fn to_dense<T: NumericElem>(view: &TensorView) -> Result<Vec<T>> {
    read_strided::<T>(view, T::DTYPE)
}

/// [`to_dense`] for the float-only [`FloatElem`] subset.
pub fn to_dense_float<T: FloatElem>(view: &TensorView) -> Result<Vec<T>> {
    read_strided::<T>(view, T::DTYPE)
}

fn read_strided<T: Copy>(view: &TensorView, want: DataType) -> Result<Vec<T>> {
    view.validate()?;
    debug_assert_eq!(
        std::mem::size_of::<T>(),
        want.byte_size(),
        "read_strided element width must match dtype byte size"
    );
    if view.dtype != want {
        return Err(EpError::InvalidTensorView {
            reason: format!("expected {want:?} view, got {:?}", view.dtype),
        });
    }
    let n = numel(view.shape);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    let origin = view.data_ptr::<T>();
    let mut idx = vec![0usize; view.shape.len()];
    loop {
        let off = elem_offset(view.strides, &idx);
        // SAFETY: `origin` is the element origin of a validated view; `off` is an
        // in-shape element offset (each component `< shape[d]`), so the address
        // lies within the extent the view describes — bounds-checked against the
        // backing allocation by the owning EP (ep-api invariant #1). `T` is a
        // plain numeric/`half` type with no invalid bit patterns.
        out.push(unsafe { *origin.offset(off) });
        if !next_index(view.shape, &mut idx) {
            break;
        }
    }
    Ok(out)
}

/// Write a dense, row-major `&[T]` into `out`, applying the output view's
/// strides and byte offset. `data.len()` must equal the output element count
/// and `out.dtype` must equal `T::DTYPE`.
pub fn write_dense<T: NumericElem>(out: &mut TensorMut, data: &[T]) -> Result<()> {
    write_strided::<T>(out, data, T::DTYPE)
}

/// [`write_dense`] for the float-only [`FloatElem`] subset.
pub fn write_dense_float<T: FloatElem>(out: &mut TensorMut, data: &[T]) -> Result<()> {
    write_strided::<T>(out, data, T::DTYPE)
}

fn write_strided<T: Copy>(out: &mut TensorMut, data: &[T], want: DataType) -> Result<()> {
    out.validate()?;
    if out.dtype != want {
        return Err(EpError::InvalidTensorView {
            reason: format!("expected {want:?} output, got {:?}", out.dtype),
        });
    }
    let n = numel(out.shape);
    if data.len() != n {
        return Err(EpError::KernelFailed(format!(
            "output element count {n} does not match produced {}",
            data.len()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let origin = out.data_ptr_mut::<T>();
    let strides = out.strides;
    let shape = out.shape;
    let mut idx = vec![0usize; shape.len()];
    let mut i = 0usize;
    loop {
        let off = elem_offset(strides, &idx);
        // SAFETY: `origin` is the element origin of a validated output view; `off`
        // is an in-shape offset within the extent the view describes (bounds-
        // checked by the EP per invariant #1). The row-major walk visits every
        // logical index exactly once, so each address is written once.
        unsafe {
            *origin.offset(off) = data[i];
        }
        i += 1;
        if !next_index(shape, &mut idx) {
            break;
        }
    }
    Ok(())
}

/// RULE #1 error for a dtype an op is not defined over: WHAT is unsupported,
/// WHY, and HOW to proceed.
pub fn unsupported_dtype(op: &str, dtype: DataType) -> EpError {
    EpError::KernelFailed(format!(
        "{op}: unsupported element type {dtype:?} (WHAT: this CPU kernel was asked \
         to run {op} on a {dtype:?} tensor). WHY: ONNX does not define {op} for \
         {dtype:?}, or arithmetic on it is not implemented by this execution \
         provider. HOW: insert a `Cast` to a supported numeric dtype (e.g. \
         Float32) before {op}, or run the op on an EP that implements {dtype:?}."
    ))
}

/// Map a runtime [`DataType`] to a monomorphized body over the matching Rust
/// element type, across the full ONNX numeric set (floats + signed/unsigned
/// integers). Binds `$T` via a local `type` alias; unsupported dtypes yield a
/// RULE #1 error. The body must evaluate to `Result<()>`.
///
/// ```ignore
/// dispatch_arith!(inputs[0].dtype, "Add", T => run::<T>(inputs, outputs))
/// ```
#[macro_export]
macro_rules! dispatch_arith {
    ($dtype:expr, $op:expr, $T:ident => $body:expr) => {{
        match $dtype {
            ::onnx_runtime_ir::DataType::Float32 => {
                type $T = f32;
                $body
            }
            ::onnx_runtime_ir::DataType::Float16 => {
                type $T = half::f16;
                $body
            }
            ::onnx_runtime_ir::DataType::BFloat16 => {
                type $T = half::bf16;
                $body
            }
            ::onnx_runtime_ir::DataType::Float64 => {
                type $T = f64;
                $body
            }
            ::onnx_runtime_ir::DataType::Int8 => {
                type $T = i8;
                $body
            }
            ::onnx_runtime_ir::DataType::Int16 => {
                type $T = i16;
                $body
            }
            ::onnx_runtime_ir::DataType::Int32 => {
                type $T = i32;
                $body
            }
            ::onnx_runtime_ir::DataType::Int64 => {
                type $T = i64;
                $body
            }
            ::onnx_runtime_ir::DataType::Uint8 => {
                type $T = u8;
                $body
            }
            ::onnx_runtime_ir::DataType::Uint16 => {
                type $T = u16;
                $body
            }
            ::onnx_runtime_ir::DataType::Uint32 => {
                type $T = u32;
                $body
            }
            ::onnx_runtime_ir::DataType::Uint64 => {
                type $T = u64;
                $body
            }
            other => Err($crate::dtype::unsupported_dtype($op, other)),
        }
    }};
}

/// Like [`dispatch_arith`] but restricted to the floating-point dtypes ONNX
/// defines transcendental / accumulate ops over (`f32`, `f16`, `bf16`, `f64`).
#[macro_export]
macro_rules! dispatch_float {
    ($dtype:expr, $op:expr, $T:ident => $body:expr) => {{
        match $dtype {
            ::onnx_runtime_ir::DataType::Float32 => {
                type $T = f32;
                $body
            }
            ::onnx_runtime_ir::DataType::Float16 => {
                type $T = half::f16;
                $body
            }
            ::onnx_runtime_ir::DataType::BFloat16 => {
                type $T = half::bf16;
                $body
            }
            ::onnx_runtime_ir::DataType::Float64 => {
                type $T = f64;
                $body
            }
            other => Err($crate::dtype::unsupported_dtype($op, other)),
        }
    }};
}

/// Widen any supported float view (`f32`/`f16`/`bf16`/`f64`) to a dense
/// `Vec<f32>` for the MatMul/Gemm f32-accumulate path. Rejects non-float dtypes
/// with a RULE #1 error.
pub fn to_dense_f32_widen(op: &str, view: &TensorView) -> Result<Vec<f32>> {
    dispatch_float!(view.dtype, op, T => {
        let raw = to_dense_float::<T>(view)?;
        Ok(raw.into_iter().map(|v| v.to_f32()).collect())
    })
}

/// Narrow a dense `Vec<f32>` result into `out`, rounding to `out`'s float dtype
/// (`f32`/`f16`/`bf16`/`f64`). Counterpart to [`to_dense_f32_widen`].
pub fn write_dense_f32_narrow(op: &str, out: &mut TensorMut, data: &[f32]) -> Result<()> {
    dispatch_float!(out.dtype, op, T => {
        let narrowed: Vec<T> = data.iter().map(|&v| T::from_f32(v)).collect();
        write_dense_float::<T>(out, &narrowed)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrips_through_f32_without_bit_reinterpret() {
        // 1.0 in f16 is 0x3C00; reinterpreting those 2 bytes as an f32 would be
        // a denormal ~1.7e-41, not 1.0 — assert we widen, not bit-cast.
        let h = half::f16::from_f32(1.0);
        assert_eq!(h.to_bits(), 0x3C00);
        assert_eq!(NumericElem::to_acc(h), 1.0f32);
        assert_eq!(half::f16::from_acc(1.0f32).to_bits(), 0x3C00);
    }

    #[test]
    fn int_div_by_zero_is_zero_not_panic() {
        assert_eq!(5i32.c_div(0), 0);
        assert_eq!(i32::MIN.c_div(-1), i32::MIN); // no overflow panic
    }

    #[test]
    fn float_min_max_propagate_nan() {
        assert!(f32::NAN.c_min(1.0).is_nan());
        assert!(1.0f32.c_max(f32::NAN).is_nan());
        assert_eq!(2.0f32.c_min(3.0), 2.0);
        assert_eq!(2.0f32.c_max(3.0), 3.0);
    }

    #[test]
    fn int_ops_wrap() {
        assert_eq!(i8::MAX.c_add(1), i8::MIN);
        assert_eq!(200u8.c_mul(2), 144); // 400 mod 256
    }

    #[test]
    fn unsupported_dtype_message_has_what_why_how() {
        let e = unsupported_dtype("Add", DataType::Bool);
        let s = format!("{e}");
        assert!(s.contains("WHAT"));
        assert!(s.contains("WHY"));
        assert!(s.contains("HOW"));
    }
}
