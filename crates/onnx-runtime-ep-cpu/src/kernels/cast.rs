//! `Cast`: convert a tensor between element types (`docs/ORT2.md` §4.4).
//!
//! Numeric semantics follow ONNX / C++ `static_cast`:
//! * float → integer truncates toward zero and **saturates** out-of-range
//!   values to the target integer's bounds (ONNX Cast semantics; NaN → 0),
//!   converting straight to the target type so narrow targets clamp rather than
//!   wrap;
//! * any numeric → `bool` is `x != 0` (NaN casts to `true`);
//! * integer → integer is a width-narrowing/widening reinterpret (`as`);
//! * float ↔ float rounds to the nearest representable value.
//!
//! The BERT target only needs f32 ↔ i64 ↔ i32 ↔ bool, but the conversion table
//! is written generically over the fixed-width numeric dtypes so it stays
//! model-agnostic.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, elem_size};
use crate::strided::{next_index, numel};

/// A single element read from the source, kept in whichever lane preserves its
/// value exactly: floats in `F`, integers/bools in `I`.
#[derive(Clone, Copy)]
enum Num {
    F(f64),
    I(i64),
}

impl Num {
    fn to_f64(self) -> f64 {
        match self {
            Num::F(f) => f,
            Num::I(i) => i as f64,
        }
    }

    /// Truncate toward zero into an `i64` lane (float) or pass through (int).
    /// Rust's `f as i64` saturates out-of-range floats to `i64::MIN/MAX` and
    /// maps NaN to 0 — exactly ONNX Cast's float→int saturation.
    fn to_i64(self) -> i64 {
        match self {
            Num::F(f) => f as i64,
            Num::I(i) => i,
        }
    }

    fn is_nonzero(self) -> bool {
        match self {
            Num::F(f) => f != 0.0,
            Num::I(i) => i != 0,
        }
    }
}

/// Convert a [`Num`] to a narrower integer target with ONNX Cast semantics:
/// a **float** source saturates directly to the *target* type (Rust `as`
/// clamps out-of-range floats and maps NaN to 0), while an **integer** source
/// wraps (two's-complement `static_cast`, matching ORT's int→int Cast).
///
/// The distinction matters for out-of-range floats: routing them through an
/// `i64` intermediate first would saturate to the i64 range and then *wrap*
/// into the narrow type, yielding a garbage value instead of the saturated one.
macro_rules! num_to_int {
    ($name:ident, $ty:ty) => {
        impl Num {
            fn $name(self) -> $ty {
                match self {
                    Num::F(f) => f as $ty,
                    Num::I(i) => i as $ty,
                }
            }
        }
    };
}

num_to_int!(to_i32, i32);
num_to_int!(to_i16, i16);
num_to_int!(to_i8, i8);
num_to_int!(to_u8, u8);
num_to_int!(to_u16, u16);
num_to_int!(to_u32, u32);

/// Cast kernel carrying the target dtype (`None` until the `to` attribute is
/// resolved; execution errors if it was absent).
pub struct CastKernel {
    to: Option<DataType>,
}

/// Factory reading the ONNX `to` attribute (a `TensorProto.DataType` integer).
pub struct CastFactory;

impl KernelFactory for CastFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let to = node
            .attr("to")
            .and_then(|a| a.as_int())
            .and_then(|raw| DataType::from_onnx(raw as i32));
        Ok(Box::new(CastKernel { to }))
    }
}

impl Kernel for CastKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Cast", inputs, outputs, 1, 1, 1)?;
        let to = self.to.ok_or_else(|| {
            EpError::KernelFailed("Cast: missing or unsupported `to` attribute".into())
        })?;
        if outputs[0].dtype != to {
            return Err(EpError::KernelFailed(format!(
                "Cast: output dtype {:?} does not match `to` {to:?}",
                outputs[0].dtype
            )));
        }

        let src = read_src(&inputs[0])?;
        let out_esize = elem_size(to)?;
        let mut bytes = Vec::with_capacity(src.len() * out_esize);
        for &n in &src {
            write_num(&mut bytes, n, to)?;
        }
        super::write_dense_bytes(&mut outputs[0], &bytes)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Read a (possibly strided) view into a dense row-major `Vec<Num>`.
fn read_src(view: &TensorView) -> Result<Vec<Num>> {
    view.validate()?;
    let esize = elem_size(view.dtype)?;
    let n = numel(view.shape);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    let origin = view.data_ptr::<u8>();
    let mut idx = vec![0usize; view.shape.len()];
    loop {
        let byte_off = crate::strided::elem_offset(view.strides, &idx) * esize as isize;
        // SAFETY: `origin` is the byte origin of a validated view; `byte_off ..
        // byte_off + esize` is an in-shape offset within the extent the view
        // describes (bounds-checked by the EP per invariant #1). We copy `esize`
        // bytes into a fresh stack buffer before interpreting them.
        let mut buf = [0u8; 8];
        unsafe {
            std::ptr::copy_nonoverlapping(origin.offset(byte_off), buf.as_mut_ptr(), esize);
        }
        out.push(decode(view.dtype, &buf)?);
        if !next_index(view.shape, &mut idx) {
            break;
        }
    }
    Ok(out)
}

/// Interpret the little-endian element bytes of `dtype`.
fn decode(dtype: DataType, buf: &[u8; 8]) -> Result<Num> {
    Ok(match dtype {
        DataType::Float32 => Num::F(f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as f64),
        DataType::Float64 => Num::F(f64::from_le_bytes(*buf)),
        DataType::Int64 => Num::I(i64::from_le_bytes(*buf)),
        DataType::Int32 => Num::I(i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as i64),
        DataType::Int16 => Num::I(i16::from_le_bytes([buf[0], buf[1]]) as i64),
        DataType::Int8 => Num::I(buf[0] as i8 as i64),
        DataType::Uint8 => Num::I(buf[0] as i64),
        DataType::Uint16 => Num::I(u16::from_le_bytes([buf[0], buf[1]]) as i64),
        DataType::Uint32 => Num::I(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as i64),
        DataType::Bool => Num::I((buf[0] != 0) as i64),
        other => {
            return Err(EpError::KernelFailed(format!(
                "Cast: unsupported source dtype {other:?}"
            )));
        }
    })
}

/// Append the little-endian bytes of `n` converted to `dtype`.
fn write_num(out: &mut Vec<u8>, n: Num, dtype: DataType) -> Result<()> {
    match dtype {
        DataType::Float32 => out.extend_from_slice(&(n.to_f64() as f32).to_le_bytes()),
        DataType::Float64 => out.extend_from_slice(&n.to_f64().to_le_bytes()),
        DataType::Int64 => out.extend_from_slice(&n.to_i64().to_le_bytes()),
        DataType::Int32 => out.extend_from_slice(&n.to_i32().to_le_bytes()),
        DataType::Int16 => out.extend_from_slice(&n.to_i16().to_le_bytes()),
        DataType::Int8 => out.extend_from_slice(&n.to_i8().to_le_bytes()),
        DataType::Uint8 => out.push(n.to_u8()),
        DataType::Uint16 => out.extend_from_slice(&n.to_u16().to_le_bytes()),
        DataType::Uint32 => out.extend_from_slice(&n.to_u32().to_le_bytes()),
        DataType::Bool => out.push(n.is_nonzero() as u8),
        other => {
            return Err(EpError::KernelFailed(format!(
                "Cast: unsupported target dtype {other:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn cast(to: DataType, input: &Owned, out: &mut Owned) {
        CastKernel { to: Some(to) }
            .execute(&[input.view()], &mut [out.view_mut()])
            .unwrap();
    }

    #[test]
    fn f32_to_i64_truncates_toward_zero() {
        let a = Owned::f32(&[4], &[1.9, -1.9, 2.5, -2.5]);
        let mut out = Owned::zeros(DataType::Int64, &[4]);
        cast(DataType::Int64, &a, &mut out);
        assert_eq!(out.to_i64(), vec![1, -1, 2, -2]);
    }

    #[test]
    fn i64_to_f32_roundtrip() {
        let a = Owned::i64(&[3], &[0, 7, -13]);
        let mut out = Owned::zeros(DataType::Float32, &[3]);
        cast(DataType::Float32, &a, &mut out);
        assert_eq!(out.to_f32(), vec![0.0, 7.0, -13.0]);
    }

    #[test]
    fn i64_to_i32_and_back() {
        let a = Owned::i64(&[2], &[123456, -7]);
        let mut i32out = Owned::zeros(DataType::Int32, &[2]);
        cast(DataType::Int32, &a, &mut i32out);
        assert_eq!(i32out.to_i32(), vec![123456, -7]);
        let mut back = Owned::zeros(DataType::Int64, &[2]);
        cast(DataType::Int64, &i32out, &mut back);
        assert_eq!(back.to_i64(), vec![123456, -7]);
    }

    #[test]
    fn f32_to_bool_nonzero() {
        let a = Owned::f32(&[4], &[0.0, 1.0, -2.5, 0.0]);
        let mut out = Owned::zeros(DataType::Bool, &[4]);
        cast(DataType::Bool, &a, &mut out);
        assert_eq!(out.to_bool(), vec![false, true, true, false]);
    }

    #[test]
    fn bool_to_f32() {
        let a = Owned::bool_(&[3], &[true, false, true]);
        let mut out = Owned::zeros(DataType::Float32, &[3]);
        cast(DataType::Float32, &a, &mut out);
        assert_eq!(out.to_f32(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn nan_casts_to_true_bool() {
        let a = Owned::f32(&[1], &[f32::NAN]);
        let mut out = Owned::zeros(DataType::Bool, &[1]);
        cast(DataType::Bool, &a, &mut out);
        assert_eq!(out.to_bool(), vec![true]);
    }

    #[test]
    fn i32_input_to_f32() {
        let a = Owned::i32(&[3], &[-4, 0, 11]);
        let mut out = Owned::zeros(DataType::Float32, &[3]);
        cast(DataType::Float32, &a, &mut out);
        assert_eq!(out.to_f32(), vec![-4.0, 0.0, 11.0]);
    }

    #[test]
    fn f32_out_of_range_saturates_to_target_int() {
        // A float far outside i32/i16/i8 range must SATURATE to the target's
        // bound, not wrap. The old i64-intermediate path wrapped narrow targets.
        let big = 1.0e20_f32;
        let neg = -1.0e20_f32;

        let a = Owned::f32(&[2], &[big, neg]);
        let mut i32out = Owned::zeros(DataType::Int32, &[2]);
        cast(DataType::Int32, &a, &mut i32out);
        assert_eq!(i32out.to_i32(), vec![i32::MAX, i32::MIN]);

        let mut i64out = Owned::zeros(DataType::Int64, &[2]);
        cast(DataType::Int64, &a, &mut i64out);
        assert_eq!(i64out.to_i64(), vec![i64::MAX, i64::MIN]);
    }

    #[test]
    fn f32_out_of_range_saturates_unsigned() {
        // Negative and over-range floats clamp to [0, u8::MAX] for uint8.
        let a = Owned::f32(&[3], &[-5.0, 300.0, 42.0]);
        let mut out = Owned::zeros(DataType::Uint8, &[3]);
        cast(DataType::Uint8, &a, &mut out);
        // uint8 lane holds the saturated values one byte each.
        assert_eq!(out.bytes, vec![0u8, 255u8, 42u8]);
    }

    #[test]
    fn nan_casts_to_zero_int() {
        // ONNX Cast maps NaN → 0 for integer targets (Rust `as` does the same).
        let a = Owned::f32(&[1], &[f32::NAN]);
        let mut out = Owned::zeros(DataType::Int32, &[1]);
        cast(DataType::Int32, &a, &mut out);
        assert_eq!(out.to_i32(), vec![0]);
    }

    #[test]
    fn missing_to_attribute_errors() {
        let a = Owned::f32(&[1], &[1.0]);
        let mut out = Owned::zeros(DataType::Int64, &[1]);
        let err = CastKernel { to: None }.execute(&[a.view()], &mut [out.view_mut()]);
        assert!(err.is_err());
    }
}
