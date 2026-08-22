//! `Cast`: convert a tensor between element types (`docs/architecture/ORT2.md` §4.4).
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

/// `CastLike` kernel. Its target dtype is supplied by input 1 rather than an
/// attribute, but conversion itself is exactly the `Cast` conversion table.
pub struct CastLikeKernel;

/// Factory for `CastLike`.
pub struct CastLikeFactory;

impl KernelFactory for CastLikeFactory {
    fn create(&self, _node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(CastLikeKernel))
    }
}

impl Kernel for CastKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Cast", inputs, outputs, 1, 1, 1)?;
        let to = self.to.ok_or_else(|| {
            EpError::KernelFailed("Cast: missing or unsupported `to` attribute".into())
        })?;
        cast_to("Cast", &inputs[0], &mut outputs[0], to)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl Kernel for CastLikeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("CastLike", inputs, outputs, 2, 2, 1)?;
        inputs[1].validate()?;
        cast_to("CastLike", &inputs[0], &mut outputs[0], inputs[1].dtype)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn cast_to(op: &str, input: &TensorView, output: &mut TensorMut, to: DataType) -> Result<()> {
    // Cast to/from STRING is a real ONNX feature, but the ep-cpu stores
    // string tensors out-of-band (String has no fixed-width raw byte
    // layout), so the byte-oriented conversion table below cannot produce or
    // consume them. Reject explicitly (RULE #1) instead of emitting garbage.
    if to == DataType::String || input.dtype == DataType::String {
        return Err(EpError::KernelFailed(format!(
            "{op}: string conversion (from {:?} to {to:?}) is unsupported on the CPU \
                 execution provider. WHY: ep-cpu stores string tensors out-of-band and Cast \
                 here operates on fixed-width numeric bytes, so a string source or target has \
                 no raw layout to read or write. HOW: keep Cast between numeric/bool dtypes on \
                 ep-cpu, or perform string (de)serialization outside the graph before running \
                 it on this provider.",
            input.dtype
        )));
    }
    if output.dtype != to {
        return Err(EpError::KernelFailed(format!(
            "{op}: output dtype {:?} does not match target dtype {to:?}",
            output.dtype
        )));
    }

    // Fast path first: both views row-major contiguous and non-overlapping.
    // The generic path below costs an `elem_offset` dot product, a `next_index`
    // carry chain, a 16-byte `Num` pushed to a heap `Vec`, a per-element
    // `DataType` match and a per-element `Vec<u8>` extend -- about 9 ns per
    // element, which made `Cast` two orders of magnitude slower than ORT and
    // dominated every mixed-precision graph. See `cast_contiguous`.
    if cast_contiguous(input, output, to)? {
        return Ok(());
    }

    let src = read_src(input)?;
    let out_esize = elem_size(to)?;
    let mut bytes = Vec::with_capacity(src.len() * out_esize);
    for &n in &src {
        write_num(&mut bytes, n, to)?;
    }
    super::write_dense_bytes(output, &bytes)
}

/// Number of elements converted per staging batch. Sized so the `Num` staging
/// buffer (16 B/element) stays inside L1 alongside the source and destination
/// windows, the same shape of trade-off as `dense_elementwise::F16_STAGE_CHUNK`.
const CAST_STAGE_CHUNK: usize = 1024;

/// Marker types naming an ONNX element type for the monomorphised conversion
/// loops. Distinct markers are needed because `Bool` and `Uint8` share `u8` as
/// their storage type but decode differently (`b != 0` vs `b as i64`).
mod lane {
    pub struct F32;
    pub struct F64;
    pub struct F16;
    pub struct BF16;
    pub struct I64;
    pub struct I32;
    pub struct I16;
    pub struct I8;
    pub struct U8;
    pub struct U16;
    pub struct U32;
    pub struct Bool;
}

/// Decode a contiguous run of one element type into the `Num` lane.
///
/// Every implementation must be **exactly** the corresponding arm of
/// [`decode`]; `contiguous_cast_matches_strided_reference` pins that.
trait DecodeLane {
    const SIZE: usize;
    fn decode(bytes: &[u8]) -> Num;
}

/// Encode the `Num` lane into a contiguous run of one element type.
///
/// Every implementation must be **exactly** the corresponding arm of
/// [`write_num`].
trait EncodeLane {
    const SIZE: usize;
    fn encode(n: Num, out: &mut [u8]);
}

macro_rules! impl_lane {
    ($marker:ty, $size:literal, $dec:expr, $enc:expr) => {
        impl DecodeLane for $marker {
            const SIZE: usize = $size;
            #[inline(always)]
            fn decode(bytes: &[u8]) -> Num {
                let f: fn(&[u8]) -> Num = $dec;
                f(bytes)
            }
        }
        impl EncodeLane for $marker {
            const SIZE: usize = $size;
            #[inline(always)]
            fn encode(n: Num, out: &mut [u8]) {
                let f: fn(Num, &mut [u8]) = $enc;
                f(n, out)
            }
        }
    };
}

impl_lane!(
    lane::F32,
    4,
    |b| Num::F(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64),
    |n, o| o.copy_from_slice(&(n.to_f64() as f32).to_le_bytes())
);
impl_lane!(
    lane::F64,
    8,
    |b| Num::F(f64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    ])),
    |n, o| o.copy_from_slice(&n.to_f64().to_le_bytes())
);
impl_lane!(
    lane::F16,
    2,
    |b| Num::F(half::f16::from_le_bytes([b[0], b[1]]).to_f64()),
    |n, o| o.copy_from_slice(&half::f16::from_f32(n.to_f64() as f32).to_le_bytes())
);
impl_lane!(
    lane::BF16,
    2,
    |b| Num::F(half::bf16::from_le_bytes([b[0], b[1]]).to_f64()),
    |n, o| o.copy_from_slice(&half::bf16::from_f32(n.to_f64() as f32).to_le_bytes())
);
impl_lane!(
    lane::I64,
    8,
    |b| Num::I(i64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    ])),
    |n, o| o.copy_from_slice(&n.to_i64().to_le_bytes())
);
impl_lane!(
    lane::I32,
    4,
    |b| Num::I(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64),
    |n, o| o.copy_from_slice(&n.to_i32().to_le_bytes())
);
impl_lane!(
    lane::I16,
    2,
    |b| Num::I(i16::from_le_bytes([b[0], b[1]]) as i64),
    |n, o| o.copy_from_slice(&n.to_i16().to_le_bytes())
);
impl_lane!(lane::I8, 1, |b| Num::I(b[0] as i8 as i64), |n, o| o[0] =
    n.to_i8() as u8);
impl_lane!(lane::U8, 1, |b| Num::I(b[0] as i64), |n, o| o[0] =
    n.to_u8());
impl_lane!(
    lane::U16,
    2,
    |b| Num::I(u16::from_le_bytes([b[0], b[1]]) as i64),
    |n, o| o.copy_from_slice(&n.to_u16().to_le_bytes())
);
impl_lane!(
    lane::U32,
    4,
    |b| Num::I(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64),
    |n, o| o.copy_from_slice(&n.to_u32().to_le_bytes())
);
impl_lane!(lane::Bool, 1, |b| Num::I((b[0] != 0) as i64), |n, o| o[0] =
    n.is_nonzero() as u8);

/// Run `$body` with `$L` bound to the marker for `$dt`, or evaluate to `None`
/// when the dtype has no fixed-width lane (`String`, sub-byte types, ...).
macro_rules! with_lane {
    ($dt:expr, $L:ident, $body:expr) => {
        match $dt {
            DataType::Float32 => {
                type $L = lane::F32;
                Some($body)
            }
            DataType::Float64 => {
                type $L = lane::F64;
                Some($body)
            }
            DataType::Float16 => {
                type $L = lane::F16;
                Some($body)
            }
            DataType::BFloat16 => {
                type $L = lane::BF16;
                Some($body)
            }
            DataType::Int64 => {
                type $L = lane::I64;
                Some($body)
            }
            DataType::Int32 => {
                type $L = lane::I32;
                Some($body)
            }
            DataType::Int16 => {
                type $L = lane::I16;
                Some($body)
            }
            DataType::Int8 => {
                type $L = lane::I8;
                Some($body)
            }
            DataType::Uint8 => {
                type $L = lane::U8;
                Some($body)
            }
            DataType::Uint16 => {
                type $L = lane::U16;
                Some($body)
            }
            DataType::Uint32 => {
                type $L = lane::U32;
                Some($body)
            }
            DataType::Bool => {
                type $L = lane::Bool;
                Some($body)
            }
            _ => None,
        }
    };
}

/// Decode `n` contiguous `S` elements into `stage`, hoisting the source dtype
/// match out of the loop.
fn decode_run<S: DecodeLane>(src: &[u8], stage: &mut [Num], n: usize) {
    for (k, slot) in stage.iter_mut().enumerate().take(n) {
        *slot = S::decode(&src[k * S::SIZE..(k + 1) * S::SIZE]);
    }
}

/// Encode `n` staged `Num`s into contiguous `D` elements.
fn encode_run<D: EncodeLane>(stage: &[Num], dst: &mut [u8], n: usize) {
    for (k, &v) in stage.iter().enumerate().take(n) {
        D::encode(v, &mut dst[k * D::SIZE..(k + 1) * D::SIZE]);
    }
}

/// Contiguous, non-aliasing `Cast`. Returns `Ok(false)` when the layout or
/// dtype pair is not eligible so the caller runs the generic strided path.
///
/// Staging through a fixed-size `Num` batch keeps this at 12 + 12 = 24
/// monomorphised loops instead of the 144 a fully fused `S x D` loop would
/// need, while still paying the `DataType` match once per batch rather than
/// once per element. The staging buffer is 16 KiB and L1-resident.
fn cast_contiguous(input: &TensorView, output: &mut TensorMut, to: DataType) -> Result<bool> {
    if !input.is_contiguous() || !output.is_contiguous() {
        return Ok(false);
    }
    let n = numel(input.shape);
    if n != numel(output.shape) {
        return Ok(false);
    }
    if n == 0 {
        return Ok(true);
    }
    let (Ok(src_esize), Ok(dst_esize)) = (elem_size(input.dtype), elem_size(to)) else {
        return Ok(false);
    };

    // Resolve lane eligibility once, before any work. `elem_size` succeeds for
    // several dtypes this path has no lane for (`Uint64`, `Complex64/128`, the
    // `Float8*` family), so without this the loop below would decode a whole
    // batch only to discover the target is unsupported. Hoisting it also makes
    // the invariant explicit: past this point both `with_lane!` sites are
    // `Some`, so the loop cannot bail after writing to `dst`.
    let src_lane = with_lane!(input.dtype, S, <S as DecodeLane>::SIZE);
    let dst_lane = with_lane!(to, D, <D as EncodeLane>::SIZE);
    let (Some(src_lane), Some(dst_lane)) = (src_lane, dst_lane) else {
        return Ok(false);
    };
    // The staging loop slices by `elem_size` but each lane decodes/encodes
    // `SIZE` bytes; they must agree or the walk would shear.
    debug_assert_eq!(src_lane, src_esize);
    debug_assert_eq!(dst_lane, dst_esize);

    // The generic path materialises the whole result before touching the
    // output, so it tolerates an aliasing in-place Cast; this one writes as it
    // reads. No current caller aliases (`can_run_in_place` is false for Cast),
    // but fall back rather than depend on that.
    let src_start = input.data_ptr::<u8>() as usize;
    let dst_start = output.data_ptr_mut::<u8>() as usize;
    if src_start < dst_start.saturating_add(n * dst_esize)
        && dst_start < src_start.saturating_add(n * src_esize)
    {
        return Ok(false);
    }

    // SAFETY: both views are validated and row-major contiguous, so they
    // describe exactly `n * esize` readable/writable bytes from their origins
    // (ep-api safety invariant #1). The overlap check above proves the two byte
    // ranges are disjoint, so the shared borrow and the unique borrow cannot
    // alias.
    let src = unsafe { std::slice::from_raw_parts(input.data_ptr::<u8>(), n * src_esize) };
    let dst = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<u8>(), n * dst_esize) };

    if cast_contiguous_simd(input.dtype, to, src, dst, n) {
        return Ok(true);
    }

    let mut stage = [Num::I(0); CAST_STAGE_CHUNK];
    let mut done = 0usize;
    while done < n {
        let len = CAST_STAGE_CHUNK.min(n - done);
        // Both unwraps are guarded by the eligibility check above; the dtype
        // pair is invariant for the whole call.
        with_lane!(
            input.dtype,
            S,
            decode_run::<S>(
                &src[done * src_esize..(done + len) * src_esize],
                &mut stage,
                len
            )
        )
        .expect("source lane eligibility is checked before the staging loop");
        with_lane!(
            to,
            D,
            encode_run::<D>(
                &stage,
                &mut dst[done * dst_esize..(done + len) * dst_esize],
                len
            )
        )
        .expect("target lane eligibility is checked before the staging loop");
        done += len;
    }
    Ok(true)
}

/// Vector conversions for the float-to-float pairs that dominate transformer
/// graphs, where an inlined mixed-precision function turns into a `Cast` on
/// either side of every arithmetic node.
///
/// Each is bit-identical to the [`decode`]/[`write_num`] pair it replaces:
/// `_mm256_cvtph_ps` is exact for every f16, `_mm256_cvtps_ph` under
/// `_MM_FROUND_TO_NEAREST_INT` matches `half::f16::from_f32`, and the bf16
/// helpers match `half::bf16`'s round-to-nearest-even and sNaN quieting. The
/// exhaustive tests below pin all four against the generic path.
fn cast_contiguous_simd(
    from: DataType,
    to: DataType,
    src: &[u8],
    dst: &mut [u8],
    n: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use crate::dtype::{bf16x, f16c};
        // SAFETY (all four arms): `src`/`dst` were built above from validated
        // contiguous views holding exactly `n` elements of the stated dtype, so
        // reinterpreting their bytes as `n` `u16`/`f32` lanes is in bounds.
        //
        // On alignment: the helpers themselves issue *unaligned* loads and
        // stores, but `from_raw_parts` requires the slice *reference* to be
        // aligned regardless. `validate_view` only pins `byte_offset % esize`,
        // so this rests on the tensor base pointer being at least 4-byte
        // aligned -- which every allocator in the path provides (ORT's arena
        // and the system allocator both return >= 16-byte-aligned blocks). The
        // `debug_assert`s below trip in debug builds if that ever stops holding.
        // This is the same assumption already made by `dense_elementwise`,
        // `add`, `elementwise` and `activations`; it is not introduced here.
        match (from, to) {
            (DataType::Float16, DataType::Float32) if f16c::available() => {
                debug_assert_eq!(src.as_ptr().align_offset(align_of::<u16>()), 0);
                debug_assert_eq!(dst.as_ptr().align_offset(align_of::<f32>()), 0);
                debug_assert_eq!(src.as_ptr().align_offset(align_of::<u16>()), 0);
                debug_assert_eq!(dst.as_ptr().align_offset(align_of::<f32>()), 0);
                let s = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u16>(), n) };
                let d =
                    unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f32>(), n) };
                unsafe { f16c::widen(s, d) };
                true
            }
            (DataType::Float32, DataType::Float16) if f16c::available() => {
                debug_assert_eq!(src.as_ptr().align_offset(align_of::<f32>()), 0);
                debug_assert_eq!(dst.as_ptr().align_offset(align_of::<u16>()), 0);
                debug_assert_eq!(src.as_ptr().align_offset(align_of::<f32>()), 0);
                debug_assert_eq!(dst.as_ptr().align_offset(align_of::<u16>()), 0);
                let s = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<f32>(), n) };
                let d =
                    unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u16>(), n) };
                unsafe { f16c::narrow(s, d) };
                true
            }
            (DataType::BFloat16, DataType::Float32) if bf16x::available() => {
                let s = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<u16>(), n) };
                let d =
                    unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f32>(), n) };
                unsafe { bf16x::widen_quieting(s, d) };
                true
            }
            (DataType::Float32, DataType::BFloat16) if bf16x::available() => {
                let s = unsafe { std::slice::from_raw_parts(src.as_ptr().cast::<f32>(), n) };
                let d =
                    unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u16>(), n) };
                unsafe { bf16x::narrow(s, d) };
                true
            }
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (from, to, src, dst, n);
        false
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
        DataType::Float16 => Num::F(half::f16::from_le_bytes([buf[0], buf[1]]).to_f64()),
        DataType::BFloat16 => Num::F(half::bf16::from_le_bytes([buf[0], buf[1]]).to_f64()),
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
        DataType::Float16 => {
            out.extend_from_slice(&half::f16::from_f32(n.to_f64() as f32).to_le_bytes())
        }
        DataType::BFloat16 => {
            out.extend_from_slice(&half::bf16::from_f32(n.to_f64() as f32).to_le_bytes())
        }
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
    fn cast_like_uses_target_dtype_for_i32_and_f16() {
        let input = Owned::f32(&[2], &[1.9, -2.5]);
        let target_i32 = Owned::i32(&[], &[0]);
        let mut i32out = Owned::zeros(DataType::Int32, &[2]);
        CastLikeKernel
            .execute(&[input.view(), target_i32.view()], &mut [i32out.view_mut()])
            .unwrap();
        assert_eq!(i32out.to_i32(), vec![1, -2]);

        let target_f16 = Owned::f16(&[], &[0.]);
        let mut f16out = Owned::zeros(DataType::Float16, &[2]);
        CastLikeKernel
            .execute(&[input.view(), target_f16.view()], &mut [f16out.view_mut()])
            .unwrap();
        assert_eq!(f16out.dtype, DataType::Float16);
        assert!((f16out.to_f16_as_f32()[0] - 1.9).abs() < 1e-3);
        assert_eq!(f16out.to_f16_as_f32()[1], -2.5);
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

    #[test]
    fn cast_to_string_is_rejected_with_actionable_error() {
        // ep-cpu has no raw string layout: Cast to String must fail loudly
        // (RULE #1) rather than produce garbage.
        let a = Owned::f32(&[1], &[1.0]);
        let mut out = Owned::zeros(DataType::String, &[1]);
        let err = CastKernel {
            to: Some(DataType::String),
        }
        .execute(&[a.view()], &mut [out.view_mut()]);
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("string conversion"), "message was: {msg}");
        assert!(msg.contains("HOW:"), "message must be actionable: {msg}");
    }

    #[test]
    fn cast_from_string_is_rejected_with_actionable_error() {
        // Casting a String source is equally unsupported on ep-cpu.
        let a = Owned::zeros(DataType::String, &[1]);
        let mut out = Owned::zeros(DataType::Float32, &[1]);
        let err = CastKernel {
            to: Some(DataType::Float32),
        }
        .execute(&[a.view()], &mut [out.view_mut()]);
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("string conversion"), "message was: {msg}");
    }
}

/// Bit-exactness of the contiguous fast path against the generic strided path.
///
/// `cast_contiguous` (and its SIMD arms) must be a pure optimisation: for every
/// input bit pattern it has to produce byte-for-byte what `read_src` +
/// `write_num` produced before it existed. NaN payloads, signed zeros,
/// subnormals and float->int saturation all round-trip through raw bytes here,
/// so a payload or rounding-mode difference cannot hide behind `==` on floats.
#[cfg(test)]
mod contiguous_fast_path_tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    /// The per-element computation the generic path performs, extracted so the
    /// fast path can be diffed against it directly.
    fn reference_bytes(from: DataType, to: DataType, src: &[u8]) -> Vec<u8> {
        let esize = elem_size(from).unwrap();
        let mut out = Vec::with_capacity(src.len() / esize * elem_size(to).unwrap());
        for chunk in src.chunks_exact(esize) {
            let mut buf = [0u8; 8];
            buf[..esize].copy_from_slice(chunk);
            write_num(&mut out, decode(from, &buf).unwrap(), to).unwrap();
        }
        out
    }

    fn run_kernel(from: DataType, to: DataType, src: &[u8]) -> Vec<u8> {
        let n = src.len() / elem_size(from).unwrap();
        let mut input = Owned::zeros(from, &[n]);
        input.bytes.copy_from_slice(src);
        let mut output = Owned::zeros(to, &[n]);
        CastKernel { to: Some(to) }
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        std::mem::take(&mut output.bytes)
    }

    const TARGETS: [DataType; 12] = [
        DataType::Float32,
        DataType::Float64,
        DataType::Float16,
        DataType::BFloat16,
        DataType::Int64,
        DataType::Int32,
        DataType::Int16,
        DataType::Int8,
        DataType::Uint8,
        DataType::Uint16,
        DataType::Uint32,
        DataType::Bool,
    ];

    fn all_16_bit_patterns() -> Vec<u8> {
        let mut v = Vec::with_capacity(1 << 17);
        for bits in 0..=u16::MAX {
            v.extend_from_slice(&bits.to_le_bytes());
        }
        v
    }

    /// Every one of the 65536 f16 encodings -- both zeros, all subnormals, both
    /// infinities and every quiet and **signalling** NaN payload -- converted to
    /// all 12 fixed-width targets. This is what pins the F16C `widen` arm and
    /// the float->int saturation rules.
    #[test]
    fn every_float16_pattern_matches_the_generic_path() {
        let src = all_16_bit_patterns();
        for to in TARGETS {
            let got = run_kernel(DataType::Float16, to, &src);
            let want = reference_bytes(DataType::Float16, to, &src);
            assert_eq!(
                got, want,
                "float16 -> {to:?} diverged from the generic path"
            );
        }
    }

    /// Same exhaustive sweep for bfloat16, pinning the AVX2 `widen_quieting`
    /// arm including its signalling-NaN quieting.
    #[test]
    fn every_bfloat16_pattern_matches_the_generic_path() {
        let src = all_16_bit_patterns();
        for to in TARGETS {
            let got = run_kernel(DataType::BFloat16, to, &src);
            let want = reference_bytes(DataType::BFloat16, to, &src);
            assert_eq!(
                got, want,
                "bfloat16 -> {to:?} diverged from the generic path"
            );
        }
    }

    /// f32 sources over an adversarial value set, into every target. Covers the
    /// `narrow` arms (round-to-nearest-even, overflow to infinity, subnormal
    /// flush behaviour) and float->integer saturation at every width.
    #[test]
    fn adversarial_float32_values_match_the_generic_path() {
        let mut vals: Vec<f32> = vec![
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            1.0,
            -1.0,
            0.5,
            -0.5,
            1.5,
            2.5,
            -1.5,
            -2.5,
            65504.0,
            -65504.0,
            65520.0,
            65536.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
            i8::MAX as f32,
            i8::MIN as f32,
            u8::MAX as f32,
            i16::MAX as f32,
            i16::MIN as f32,
            u16::MAX as f32,
            i32::MAX as f32,
            i32::MIN as f32,
            u32::MAX as f32,
            i64::MAX as f32,
            i64::MIN as f32,
            1e30,
            -1e30,
            6.1e-5,
            -6.1e-5,
            5.96e-8,
        ];
        // Quiet and signalling NaNs of both signs, plus a few payloads.
        for bits in [
            0x7FC0_0000u32,
            0xFFC0_0000,
            0x7F80_0001,
            0xFF80_0001,
            0x7FFF_FFFF,
        ] {
            vals.push(f32::from_bits(bits));
        }
        // A dense ramp so ordinary values are covered too, and so the run spans
        // several staging chunks and ends on a partial one.
        for i in 0..5000 {
            vals.push((i as f32 - 2500.0) * 0.37);
        }
        let mut src = Vec::with_capacity(vals.len() * 4);
        for v in &vals {
            src.extend_from_slice(&v.to_le_bytes());
        }
        for to in TARGETS {
            let got = run_kernel(DataType::Float32, to, &src);
            let want = reference_bytes(DataType::Float32, to, &src);
            assert_eq!(
                got, want,
                "float32 -> {to:?} diverged from the generic path"
            );
        }
    }

    /// Integer and bool sources, including the values where float->int and
    /// int->int differ (wrap vs saturate) and where i64 -> f32 double-rounds
    /// through f64.
    #[test]
    fn integer_sources_match_the_generic_path() {
        let ints: Vec<i64> = vec![
            0,
            1,
            -1,
            127,
            128,
            -128,
            255,
            256,
            32767,
            -32768,
            65535,
            i32::MAX as i64,
            i32::MIN as i64,
            u32::MAX as i64,
            i64::MAX,
            i64::MIN,
            (1i64 << 53) + 1,
            -((1i64 << 53) + 1),
            (1i64 << 62) + 12345,
        ];
        for (from, esize) in [
            (DataType::Int64, 8usize),
            (DataType::Int32, 4),
            (DataType::Int16, 2),
            (DataType::Int8, 1),
            (DataType::Uint8, 1),
            (DataType::Uint16, 2),
            (DataType::Uint32, 4),
            (DataType::Bool, 1),
        ] {
            let mut src = Vec::new();
            for v in &ints {
                src.extend_from_slice(&v.to_le_bytes()[..esize]);
            }
            for to in TARGETS {
                let got = run_kernel(from, to, &src);
                let want = reference_bytes(from, to, &src);
                assert_eq!(
                    got, want,
                    "{from:?} -> {to:?} diverged from the generic path"
                );
            }
        }
    }

    /// Lengths around the staging-chunk boundary and the 8-wide vector width,
    /// so a tail bug in either the staged loop or the SIMD arms cannot hide
    /// behind a large aligned run.
    #[test]
    fn chunk_and_vector_boundary_lengths_match_the_generic_path() {
        let mut lengths: Vec<usize> = (0..20).collect();
        for base in [CAST_STAGE_CHUNK, 2 * CAST_STAGE_CHUNK] {
            for delta in 0..=8 {
                lengths.push(base + delta);
                lengths.push(base - delta);
            }
        }
        for n in lengths {
            for (from, to) in [
                (DataType::Float16, DataType::Float32),
                (DataType::Float32, DataType::Float16),
                (DataType::BFloat16, DataType::Float32),
                (DataType::Float32, DataType::BFloat16),
                (DataType::Float32, DataType::Uint8),
                (DataType::Int64, DataType::Float32),
            ] {
                let esize = elem_size(from).unwrap();
                let src: Vec<u8> = (0..n * esize).map(|i| (i * 37 + 11) as u8).collect();
                let got = run_kernel(from, to, &src);
                let want = reference_bytes(from, to, &src);
                assert_eq!(got, want, "{from:?} -> {to:?} diverged at n={n}");
            }
        }
    }

    /// A non-row-major input must still go through the generic strided walk and
    /// produce the transposed reading, so the fast path cannot be taken
    /// unconditionally.
    #[test]
    fn strided_input_still_uses_the_generic_path() {
        let input =
            Owned::f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).with_view(&[3, 2], &[1, 3]);
        assert!(!input.view().is_contiguous());
        let mut output = Owned::zeros(DataType::Float32, &[3, 2]);
        CastKernel {
            to: Some(DataType::Float32),
        }
        .execute(&[input.view()], &mut [output.view_mut()])
        .unwrap();
        assert_eq!(output.to_f32(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
