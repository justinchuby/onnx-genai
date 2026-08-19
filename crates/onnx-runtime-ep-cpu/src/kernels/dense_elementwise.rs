//! Shared dense-elementwise SIMD infrastructure for per-element unary ops.
//!
//! ## Design
//!
//! A per-element operation (Relu, Clip, Abs, Neg, …) does not depend on logical
//! axis order — only on visiting every element exactly once. The traditional
//! "contiguity guard" (`is_contiguous()`) unnecessarily rejects dense-but-permuted
//! tensors (e.g. NHWC layout in an NCHW model). This module provides dispatch
//! that accepts any tensor whose backing memory is **dense** (all elements packed
//! without holes) and whose input/output strides match, regardless of logical
//! layout.
//!
//! Three tiers:
//! 1. **Dense SIMD** — input & output are both dense with matching strides.
//!    Process the entire backing buffer with NEON/scalar bulk loops.
//! 2. **Strided** — non-dense strides. Falls to the caller's generic path
//!    (widen/narrow). Counted for observability.
//! 3. **Scalar fallback** — last resort (covered by caller's widen path).
//!
//! ## Dtype coverage
//!
//! - **f32**: NEON `vmaxq_f32` / `vminq_f32` on aarch64, scalar elsewhere.
//! - **f16**: NEON widen-to-f32 (`vcvt_f32_f16`), process via f32 SIMD, narrow
//!   back (`vcvt_f16_f32`) — available on ALL aarch64 (no `target_feature = "fp16"`
//!   required). This gives SIMD throughput for f16 on every Apple Silicon chip.
//! - **bf16**: Widen-to-f32, compute, narrow-back via scalar (no native bf16
//!   comparison on NEON).
//!
//! ## NaN semantics
//!
//! NaN **propagates** on all paths. On NEON: `vmaxq_f32` / `vminq_f32` lower
//! to `FMAX` / `FMIN` (propagating). On scalar: PartialOrd comparisons (NaN
//! compares false, passing through unchanged). The f16 path inherits f32
//! NaN-propagation since it widens through `vcvt_f32_f16` (NaN f16 → NaN f32).
//!
//! ## Signed zero
//!
//! NEON `FMAX(+0, -0)` returns `+0`; scalar comparison preserves `-0`. This is
//! an accepted divergence (matches ONNX spec and ORT/MLAS behaviour).

use std::sync::atomic::{AtomicU64, Ordering};

use onnx_runtime_ep_api::{Result, TensorMut, TensorView};
use onnx_runtime_ir::DataType;

use crate::strided::numel;

// ─── Dispatch counters ──────────────────────────────────────────────────────

/// Counter: dense SIMD path fired for f32.
#[doc(hidden)]
pub static DENSE_ELEM_F32_HITS: AtomicU64 = AtomicU64::new(0);

/// Counter: dense SIMD path fired for f16 (via widen/narrow NEON).
#[doc(hidden)]
pub static DENSE_ELEM_F16_HITS: AtomicU64 = AtomicU64::new(0);

/// Counter: dense SIMD path fired for bf16 (via widen/narrow scalar).
#[doc(hidden)]
pub static DENSE_ELEM_BF16_HITS: AtomicU64 = AtomicU64::new(0);

/// Counter: non-dense input fell through to caller's generic path.
#[doc(hidden)]
pub static DENSE_ELEM_NON_DENSE_FALLBACK_HITS: AtomicU64 = AtomicU64::new(0);

// ─── Op trait ───────────────────────────────────────────────────────────────

/// A per-element unary operation that can be applied via SIMD bulk or scalar.
pub trait ElementwiseOp {
    /// Apply to a contiguous f32 buffer: `dst[i] = op(src[i])`.
    fn apply_f32_bulk(&self, src: &[f32], dst: &mut [f32]);

    /// Scalar f32 application for tail/fallback.
    fn apply_f32_scalar(&self, x: f32) -> f32;
}

// ─── Dispatch ───────────────────────────────────────────────────────────────

/// Attempt the dense elementwise fast path. Returns `Ok(true)` if handled.
///
/// Accepts when:
/// - Input and output have the same shape.
/// - Input and output strides match (same memory traversal visits
///   corresponding elements).
/// - Both are **dense** (backing memory has no holes).
/// - Dtype is f32, f16, or bf16.
/// - No pointer aliasing between input and output (or handled via alloc).
pub fn try_dense_elementwise(
    op: &dyn ElementwiseOp,
    input: &TensorView,
    output: &mut TensorMut,
) -> Result<bool> {
    // Shape must match.
    if input.shape != output.shape {
        return Ok(false);
    }

    // Strides must match (same memory traversal order).
    if input.strides != output.strides {
        return Ok(false);
    }

    // Both must be dense (elements packed without holes).
    if !onnx_runtime_ir::is_dense(input.shape, input.strides) {
        DENSE_ELEM_NON_DENSE_FALLBACK_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(false);
    }

    // Dtype dispatch.
    match (input.dtype, output.dtype) {
        (DataType::Float32, DataType::Float32) => {
            dispatch_dense_f32(op, input, output)?;
            DENSE_ELEM_F32_HITS.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        }
        (DataType::Float16, DataType::Float16) => {
            dispatch_dense_f16(op, input, output)?;
            DENSE_ELEM_F16_HITS.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        }
        (DataType::BFloat16, DataType::BFloat16) => {
            dispatch_dense_bf16(op, input, output)?;
            DENSE_ELEM_BF16_HITS.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        }
        _ => Ok(false),
    }
}

// ─── f32 dispatch ───────────────────────────────────────────────────────────

fn dispatch_dense_f32(
    op: &dyn ElementwiseOp,
    input: &TensorView,
    output: &mut TensorMut,
) -> Result<()> {
    let len = numel(input.shape);

    // Overlap guard.
    let input_start = input.data_ptr::<u8>() as usize;
    let input_end = input_start.saturating_add(input.byte_size());
    let output_start = output.data_ptr_mut::<u8>() as usize;
    let output_end = output_start.saturating_add(output.byte_size());
    if output_start < input_end && input_start < output_end {
        let src = unsafe { std::slice::from_raw_parts(input.data_ptr::<f32>(), len) }.to_vec();
        let dst = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), len) };
        op.apply_f32_bulk(&src, dst);
        return Ok(());
    }

    let src = unsafe { std::slice::from_raw_parts(input.data_ptr::<f32>(), len) };
    let dst = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<f32>(), len) };
    op.apply_f32_bulk(src, dst);
    Ok(())
}

// ─── f16 dispatch (widen→f32 SIMD→narrow) ───────────────────────────────────

fn dispatch_dense_f16(
    op: &dyn ElementwiseOp,
    input: &TensorView,
    output: &mut TensorMut,
) -> Result<()> {
    let len = numel(input.shape);

    let src_u16 = unsafe { std::slice::from_raw_parts(input.data_ptr::<u16>(), len) };
    let dst_u16 = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<u16>(), len) };

    #[cfg(target_arch = "aarch64")]
    {
        // NEON widen/narrow path: process 4 f16 values at a time through f32 SIMD.
        // SAFETY: NEON is always available on aarch64.
        unsafe { elementwise_f16_neon(op, src_u16, dst_u16) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if crate::dtype::f16c::available() {
            // SAFETY: `available()` confirmed `f16c` + `avx2`.
            unsafe { elementwise_f16_f16c(op, src_u16, dst_u16) };
            return Ok(());
        }
        // Scalar widen/narrow fallback.
        for (d, &s) in dst_u16.iter_mut().zip(src_u16.iter()) {
            let v = half::f16::from_bits(s).to_f32();
            let r = op.apply_f32_scalar(v);
            *d = half::f16::from_f32(r).to_bits();
        }
    }

    Ok(())
}

/// Number of elements converted per staging pass.
///
/// Two `f32` staging buffers of this length are stack-allocated per call
/// (8 KiB total at 1024), which stays inside L1 and inside the smaller stacks
/// ORT's intra-op worker threads run on. Anything larger stops paying for
/// itself: the conversion is memory-bound and the buffers must survive a call
/// through `apply_f32_bulk`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const F16_STAGE_CHUNK: usize = 1024;

/// x86 f16 path: F16C widen 8 f16→f32, apply the op's f32 bulk kernel, F16C
/// narrow back, in `F16_STAGE_CHUNK`-element passes.
///
/// This replaces a scalar loop that called `half::f16::from_bits(..).to_f32()`
/// and `half::f16::from_f32(..)` per element — software conversion that made
/// every f16 elementwise op an order of magnitude slower than ORT's, even for
/// ops whose arithmetic is a single comparison. aarch64 already had the NEON
/// equivalent; x86 had the F16C primitives in `dtype` but nothing called them
/// from here.
///
/// Numerically identical to the scalar loop it replaces: `_mm256_cvtph_ps` is
/// exact for every f16, and `_mm256_cvtps_ph` with `_MM_FROUND_TO_NEAREST_INT`
/// matches `half::f16::from_f32`'s round-to-nearest-even. The op itself sees
/// the same `f32` values either way — but note it now sees them through
/// `apply_f32_bulk` rather than `apply_f32_scalar`, so the two must agree
/// (they do; the f32 dense path already relies on this).
///
/// # Safety
/// The running CPU must support `f16c` + `avx2`; `src.len() == dst.len()`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn elementwise_f16_f16c(op: &dyn ElementwiseOp, src: &[u16], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len(), "f16 staging needs equal lengths");
    let mut widened = [0.0f32; F16_STAGE_CHUNK];
    let mut applied = [0.0f32; F16_STAGE_CHUNK];
    for (src_chunk, dst_chunk) in src
        .chunks(F16_STAGE_CHUNK)
        .zip(dst.chunks_mut(F16_STAGE_CHUNK))
    {
        let len = src_chunk.len();
        // SAFETY: caller guarantees f16c+avx2; the slices are equal length.
        unsafe { crate::dtype::f16c::widen(src_chunk, &mut widened[..len]) };
        op.apply_f32_bulk(&widened[..len], &mut applied[..len]);
        // SAFETY: as above.
        unsafe { crate::dtype::f16c::narrow(&applied[..len], dst_chunk) };
    }
}

/// NEON f16 path: widen 4 f16→f32, apply op, narrow 4 f32→f16.
/// Uses `vcvt_f32_f16` / `vcvt_f16_f32` which are available on ALL aarch64.
///
/// # Safety
/// Caller must ensure `src` and `dst` have the same length and do not overlap.
#[cfg(target_arch = "aarch64")]
unsafe fn elementwise_f16_neon(op: &dyn ElementwiseOp, src: &[u16], dst: &mut [u16]) {
    use std::arch::aarch64::*;

    let n = src.len();
    let mut i = 0usize;

    // Process 16 f16 values per iteration (4×4 f32 vectors).
    let bulk_end = n & !15;
    while i < bulk_end {
        unsafe {
            let h0 = vld1_u16(src.as_ptr().add(i));
            let h1 = vld1_u16(src.as_ptr().add(i + 4));
            let h2 = vld1_u16(src.as_ptr().add(i + 8));
            let h3 = vld1_u16(src.as_ptr().add(i + 12));

            let f0 = vcvt_f32_f16(vreinterpret_f16_u16(h0));
            let f1 = vcvt_f32_f16(vreinterpret_f16_u16(h1));
            let f2 = vcvt_f32_f16(vreinterpret_f16_u16(h2));
            let f3 = vcvt_f32_f16(vreinterpret_f16_u16(h3));

            let mut src_buf = [0.0f32; 16];
            let mut dst_buf = [0.0f32; 16];
            vst1q_f32(src_buf.as_mut_ptr(), f0);
            vst1q_f32(src_buf.as_mut_ptr().add(4), f1);
            vst1q_f32(src_buf.as_mut_ptr().add(8), f2);
            vst1q_f32(src_buf.as_mut_ptr().add(12), f3);

            op.apply_f32_bulk(&src_buf, &mut dst_buf);

            let r0 = vcvt_f16_f32(vld1q_f32(dst_buf.as_ptr()));
            let r1 = vcvt_f16_f32(vld1q_f32(dst_buf.as_ptr().add(4)));
            let r2 = vcvt_f16_f32(vld1q_f32(dst_buf.as_ptr().add(8)));
            let r3 = vcvt_f16_f32(vld1q_f32(dst_buf.as_ptr().add(12)));

            vst1_u16(dst.as_mut_ptr().add(i), vreinterpret_u16_f16(r0));
            vst1_u16(dst.as_mut_ptr().add(i + 4), vreinterpret_u16_f16(r1));
            vst1_u16(dst.as_mut_ptr().add(i + 8), vreinterpret_u16_f16(r2));
            vst1_u16(dst.as_mut_ptr().add(i + 12), vreinterpret_u16_f16(r3));
        }

        i += 16;
    }

    // Tail: 4 at a time.
    while i + 4 <= n {
        unsafe {
            let h = vld1_u16(src.as_ptr().add(i));
            let f = vcvt_f32_f16(vreinterpret_f16_u16(h));
            let mut src_buf = [0.0f32; 4];
            let mut dst_buf = [0.0f32; 4];
            vst1q_f32(src_buf.as_mut_ptr(), f);
            op.apply_f32_bulk(&src_buf, &mut dst_buf);
            let r = vcvt_f16_f32(vld1q_f32(dst_buf.as_ptr()));
            vst1_u16(dst.as_mut_ptr().add(i), vreinterpret_u16_f16(r));
        }
        i += 4;
    }

    // Scalar tail.
    while i < n {
        unsafe {
            let v = half::f16::from_bits(*src.get_unchecked(i)).to_f32();
            let r = op.apply_f32_scalar(v);
            *dst.get_unchecked_mut(i) = half::f16::from_f32(r).to_bits();
        }
        i += 1;
    }
}

// ─── bf16 dispatch (widen→f32→narrow, scalar on all platforms) ──────────────

fn dispatch_dense_bf16(
    op: &dyn ElementwiseOp,
    input: &TensorView,
    output: &mut TensorMut,
) -> Result<()> {
    let len = numel(input.shape);

    let src_u16 = unsafe { std::slice::from_raw_parts(input.data_ptr::<u16>(), len) };
    let dst_u16 = unsafe { std::slice::from_raw_parts_mut(output.data_ptr_mut::<u16>(), len) };

    // bf16 has no native NEON comparison ops on current Apple Silicon.
    // Widen to f32, process in bulk, narrow back.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if crate::dtype::bf16x::available() {
        // SAFETY: `available()` confirmed `avx2`.
        unsafe { elementwise_bf16_avx2(op, src_u16, dst_u16) };
        return Ok(());
    }

    let mut f32_buf: Vec<f32> = src_u16
        .iter()
        .map(|&bits| half::bf16::from_bits(bits).to_f32())
        .collect();

    op.apply_f32_bulk(&f32_buf.clone(), &mut f32_buf);

    for (d, &v) in dst_u16.iter_mut().zip(f32_buf.iter()) {
        *d = half::bf16::from_f32(v).to_bits();
    }

    Ok(())
}

/// x86 bf16 path: AVX2 widen, apply, narrow, in staged chunks.
///
/// Besides vectorising the conversion this drops two whole-tensor heap
/// allocations — the scalar path built a `Vec<f32>` of the entire tensor and
/// then `clone()`d it, because `apply_f32_bulk` needs disjoint source and
/// destination.
///
/// `widen_quieting`, not `widen`: it replaces `half::bf16::to_f32`, which
/// quiets a signalling NaN, so this keeps the staged `f32` identical to what
/// the scalar path produced. For `Relu` and `Clip` the choice is not observable
/// — they propagate NaN unchanged and `narrow` re-quiets on the way out — but
/// an op that inspected the payload, or a future caller that read the staging
/// buffer, would see the difference. `narrow` matches `half::bf16::from_f32`'s
/// round-to-nearest-even.
///
/// # Safety
/// The running CPU must support `avx2`; `src.len() == dst.len()`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn elementwise_bf16_avx2(op: &dyn ElementwiseOp, src: &[u16], dst: &mut [u16]) {
    debug_assert_eq!(src.len(), dst.len(), "bf16 staging needs equal lengths");
    let mut widened = [0.0f32; F16_STAGE_CHUNK];
    let mut applied = [0.0f32; F16_STAGE_CHUNK];
    for (src_chunk, dst_chunk) in src
        .chunks(F16_STAGE_CHUNK)
        .zip(dst.chunks_mut(F16_STAGE_CHUNK))
    {
        let len = src_chunk.len();
        // SAFETY: caller guarantees avx2; the slices are equal length.
        unsafe { crate::dtype::bf16x::widen_quieting(src_chunk, &mut widened[..len]) };
        op.apply_f32_bulk(&widened[..len], &mut applied[..len]);
        // SAFETY: as above.
        unsafe { crate::dtype::bf16x::narrow(&applied[..len], dst_chunk) };
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Relu operation
// ═══════════════════════════════════════════════════════════════════════════════

/// Relu: `max(0, x)` — NaN-propagating, uses `vmaxq_f32` (FMAX) on NEON.
pub struct ReluOp;

impl ElementwiseOp for ReluOp {
    fn apply_f32_bulk(&self, src: &[f32], dst: &mut [f32]) {
        relu_f32_simd(src, dst);
    }

    fn apply_f32_scalar(&self, x: f32) -> f32 {
        if x < 0.0 { 0.0 } else { x }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Clip operation
// ═══════════════════════════════════════════════════════════════════════════════

/// Clip: `clamp(x, min, max)` — NaN-propagating, uses `vmaxq_f32`/`vminq_f32`.
pub struct ClipOp {
    pub minimum: f32,
    pub maximum: f32,
}

impl ElementwiseOp for ClipOp {
    fn apply_f32_bulk(&self, src: &[f32], dst: &mut [f32]) {
        clip_f32_simd(src, dst, self.minimum, self.maximum);
    }

    fn apply_f32_scalar(&self, x: f32) -> f32 {
        clamp_nan_propagating(x, self.minimum, self.maximum)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// f32 SIMD implementations
// ═══════════════════════════════════════════════════════════════════════════════

fn relu_f32_simd(src: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        relu_f32_neon(src, dst);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if avx2_available() {
            // SAFETY: `avx2_available()` confirmed AVX2; the slices are equal
            // length (`dispatch_dense_f32` derives both from the same `numel`).
            unsafe { relu_f32_avx2(src, dst) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = if s < 0.0 { 0.0 } else { s };
        }
    }
}

/// Relu over eight lanes at a time.
///
/// `vmaxps` is not a *general* `max`: Intel defines it as
/// `IF SRC1 > SRC2 THEN SRC1 ELSE SRC2`, and it returns `SRC2` whenever either
/// operand is NaN. So the operand order here is load-bearing, not stylistic —
/// with the zero vector as `SRC1` and the data as `SRC2` the instruction
/// reproduces `if x < 0.0 { 0.0 } else { x }` exactly, including the two cases
/// that make Relu awkward:
///
/// * `x = NaN` — the comparison is unordered, `SRC2` wins, the NaN propagates
///   with its payload intact. The scalar form agrees because `NaN < 0.0` is
///   false.
/// * `x = -0.0` — `0.0 > -0.0` is false, so `SRC2` wins and the sign of zero
///   survives. The scalar form agrees for the same reason.
///
/// Writing `_mm256_max_ps(x, zero)` instead would return `+0.0` for both, which
/// is a different function. `relu_f32_avx2_is_bit_identical_to_the_scalar_form`
/// pins all of this against the scalar reference over the whole special-value
/// set, bit pattern for bit pattern.
///
/// # Safety
///
/// The running CPU must support `avx2`; `src.len() == dst.len()`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_f32_avx2(src: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    unsafe {
        let zero = _mm256_setzero_ps();
        let mut i = 0usize;
        // Four vectors per iteration: the loop is pure load/max/store with no
        // cross-lane dependency, so unrolling is what keeps the two load ports
        // and the store port busy while the loop counter retires.
        let bulk_end = n & !31;
        while i < bulk_end {
            let a0 = _mm256_loadu_ps(src.as_ptr().add(i));
            let a1 = _mm256_loadu_ps(src.as_ptr().add(i + 8));
            let a2 = _mm256_loadu_ps(src.as_ptr().add(i + 16));
            let a3 = _mm256_loadu_ps(src.as_ptr().add(i + 24));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_max_ps(zero, a0));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_max_ps(zero, a1));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), _mm256_max_ps(zero, a2));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), _mm256_max_ps(zero, a3));
            i += 32;
        }
        while i + 8 <= n {
            let a = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_max_ps(zero, a));
            i += 8;
        }
        while i < n {
            let x = *src.get_unchecked(i);
            *dst.get_unchecked_mut(i) = if x < 0.0 { 0.0 } else { x };
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn relu_f32_neon(src: &[f32], dst: &mut [f32]) {
    use std::arch::aarch64::*;
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    unsafe {
        let vzero = vdupq_n_f32(0.0);
        let mut i = 0usize;
        let bulk_end = n & !15;
        while i < bulk_end {
            let a0 = vld1q_f32(src.as_ptr().add(i));
            let a1 = vld1q_f32(src.as_ptr().add(i + 4));
            let a2 = vld1q_f32(src.as_ptr().add(i + 8));
            let a3 = vld1q_f32(src.as_ptr().add(i + 12));
            let r0 = vmaxq_f32(a0, vzero);
            let r1 = vmaxq_f32(a1, vzero);
            let r2 = vmaxq_f32(a2, vzero);
            let r3 = vmaxq_f32(a3, vzero);
            vst1q_f32(dst.as_mut_ptr().add(i), r0);
            vst1q_f32(dst.as_mut_ptr().add(i + 4), r1);
            vst1q_f32(dst.as_mut_ptr().add(i + 8), r2);
            vst1q_f32(dst.as_mut_ptr().add(i + 12), r3);
            i += 16;
        }
        while i + 4 <= n {
            let a = vld1q_f32(src.as_ptr().add(i));
            let r = vmaxq_f32(a, vzero);
            vst1q_f32(dst.as_mut_ptr().add(i), r);
            i += 4;
        }
        while i < n {
            let v = *src.get_unchecked(i);
            *dst.get_unchecked_mut(i) = if v < 0.0 { 0.0 } else { v };
            i += 1;
        }
    }
}

fn clip_f32_simd(src: &[f32], dst: &mut [f32], minimum: f32, maximum: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        clip_f32_neon(src, dst, minimum, maximum);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if avx2_available() {
            // SAFETY: `avx2_available()` confirmed AVX2; the slices are equal
            // length (`dispatch_dense_f32` derives both from the same `numel`).
            unsafe { clip_f32_avx2(src, dst, minimum, maximum) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = clamp_nan_propagating(s, minimum, maximum);
        }
    }
}

/// Clip over eight lanes at a time.
///
/// The same operand-order argument as [`relu_f32_avx2`], twice.
/// [`clamp_nan_propagating`] is `max` then `min`, each written so a NaN input
/// falls through untouched, and `vmaxps`/`vminps` both return `SRC2` on an
/// unordered compare — so the bound goes in `SRC1` and the data in `SRC2`:
///
/// * `_mm256_max_ps(lo, x)` is `if x < lo { lo } else { x }`.
/// * `_mm256_min_ps(hi, x)` is `if x > hi { hi } else { x }`
///   (`MINPS: IF SRC1 < SRC2 THEN SRC1 ELSE SRC2`).
///
/// Reversing either operand pair turns a NaN into the bound and `-0.0` into
/// `+0.0`. `clip_f32_avx2_is_bit_identical_to_the_scalar_form` holds this to
/// the scalar reference bit for bit.
///
/// # Safety
///
/// The running CPU must support `avx2`; `src.len() == dst.len()`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn clip_f32_avx2(src: &[f32], dst: &mut [f32], minimum: f32, maximum: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    unsafe {
        let lo = _mm256_set1_ps(minimum);
        let hi = _mm256_set1_ps(maximum);
        let mut i = 0usize;
        let bulk_end = n & !31;
        while i < bulk_end {
            let a0 = _mm256_loadu_ps(src.as_ptr().add(i));
            let a1 = _mm256_loadu_ps(src.as_ptr().add(i + 8));
            let a2 = _mm256_loadu_ps(src.as_ptr().add(i + 16));
            let a3 = _mm256_loadu_ps(src.as_ptr().add(i + 24));
            let r0 = _mm256_min_ps(hi, _mm256_max_ps(lo, a0));
            let r1 = _mm256_min_ps(hi, _mm256_max_ps(lo, a1));
            let r2 = _mm256_min_ps(hi, _mm256_max_ps(lo, a2));
            let r3 = _mm256_min_ps(hi, _mm256_max_ps(lo, a3));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), r0);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), r1);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), r2);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), r3);
            i += 32;
        }
        while i + 8 <= n {
            let a = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i),
                _mm256_min_ps(hi, _mm256_max_ps(lo, a)),
            );
            i += 8;
        }
        while i < n {
            *dst.get_unchecked_mut(i) =
                clamp_nan_propagating(*src.get_unchecked(i), minimum, maximum);
            i += 1;
        }
    }
}

/// Whether the AVX2 arms above are live on this CPU.
///
/// Answers on every architecture so tests can branch on it portably; only the
/// `x86_64` dispatch arms call it.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    allow(dead_code)
)]
#[inline]
fn avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(target_arch = "aarch64")]
fn clip_f32_neon(src: &[f32], dst: &mut [f32], minimum: f32, maximum: f32) {
    use std::arch::aarch64::*;
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    unsafe {
        let vmin = vdupq_n_f32(minimum);
        let vmax = vdupq_n_f32(maximum);
        let mut i = 0usize;
        let bulk_end = n & !15;
        while i < bulk_end {
            let a0 = vld1q_f32(src.as_ptr().add(i));
            let a1 = vld1q_f32(src.as_ptr().add(i + 4));
            let a2 = vld1q_f32(src.as_ptr().add(i + 8));
            let a3 = vld1q_f32(src.as_ptr().add(i + 12));
            let c0 = vminq_f32(vmaxq_f32(a0, vmin), vmax);
            let c1 = vminq_f32(vmaxq_f32(a1, vmin), vmax);
            let c2 = vminq_f32(vmaxq_f32(a2, vmin), vmax);
            let c3 = vminq_f32(vmaxq_f32(a3, vmin), vmax);
            vst1q_f32(dst.as_mut_ptr().add(i), c0);
            vst1q_f32(dst.as_mut_ptr().add(i + 4), c1);
            vst1q_f32(dst.as_mut_ptr().add(i + 8), c2);
            vst1q_f32(dst.as_mut_ptr().add(i + 12), c3);
            i += 16;
        }
        while i + 4 <= n {
            let a = vld1q_f32(src.as_ptr().add(i));
            let c = vminq_f32(vmaxq_f32(a, vmin), vmax);
            vst1q_f32(dst.as_mut_ptr().add(i), c);
            i += 4;
        }
        while i < n {
            *dst.get_unchecked_mut(i) =
                clamp_nan_propagating(*src.get_unchecked(i), minimum, maximum);
            i += 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scalar helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// NaN-propagating clamp. If `x` is NaN, both comparisons are false and NaN
/// passes through unchanged. Matches NEON FMAX/FMIN semantics.
#[inline(always)]
pub fn clamp_nan_propagating(x: f32, minimum: f32, maximum: f32) -> f32 {
    let x = if x < minimum { minimum } else { x };
    if x > maximum { maximum } else { x }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_dense_basic_cases() {
        // Contiguous is always dense.
        assert!(onnx_runtime_ir::is_dense(&[2, 3, 4], &[12, 4, 1]));
        // Column-major [4,3] with strides [1,4] is dense.
        assert!(onnx_runtime_ir::is_dense(&[4, 3], &[1, 4]));
        // Row-major [4,3] with strides [3,1] is dense.
        assert!(onnx_runtime_ir::is_dense(&[4, 3], &[3, 1]));
        // Strided [4,3] with strides [4,1] has holes — NOT dense.
        assert!(!onnx_runtime_ir::is_dense(&[4, 3], &[4, 1]));
        // Scalar is dense.
        assert!(onnx_runtime_ir::is_dense(&[], &[]));
        // Size-1 dims have unconstrained strides.
        assert!(onnx_runtime_ir::is_dense(&[1, 4, 3], &[999, 3, 1]));
        // Transposed [2,4,3] from [2,3,4] with strides [12,1,4] is dense.
        assert!(onnx_runtime_ir::is_dense(&[2, 4, 3], &[12, 1, 4]));
        // Broadcast stride 0 for size > 1 is not dense (repeats elements).
        assert!(!onnx_runtime_ir::is_dense(&[4, 3], &[0, 1]));
    }

    #[test]
    fn relu_f32_nan_propagates() {
        let op = ReluOp;
        let src = [f32::NAN, -1.0, 0.0, 1.0];
        let mut dst = [0.0f32; 4];
        op.apply_f32_bulk(&src, &mut dst);
        assert!(dst[0].is_nan());
        assert_eq!(dst[1], 0.0);
        assert_eq!(dst[2], 0.0);
        assert_eq!(dst[3], 1.0);
    }

    #[test]
    fn clip_f32_nan_propagates() {
        let op = ClipOp {
            minimum: -0.5,
            maximum: 0.5,
        };
        let src = [f32::NAN, -1.0, 0.0, 1.0];
        let mut dst = [0.0f32; 4];
        op.apply_f32_bulk(&src, &mut dst);
        assert!(dst[0].is_nan());
        assert_eq!(dst[1], -0.5);
        assert_eq!(dst[2], 0.0);
        assert_eq!(dst[3], 0.5);
    }

    #[test]
    fn relu_f32_boundary_lengths() {
        let op = ReluOp;
        for len in [1, 3, 4, 15, 16, 17, 63, 64, 65, 1023] {
            let src: Vec<f32> = (0..len).map(|i| (i as f32) - (len as f32 / 2.0)).collect();
            let mut dst = vec![0.0f32; len];
            op.apply_f32_bulk(&src, &mut dst);
            for (idx, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
                let expected = if s < 0.0 { 0.0 } else { s };
                assert_eq!(expected.to_bits(), d.to_bits(), "len={len} idx={idx}");
            }
        }
    }

    #[test]
    fn clip_f32_boundary_lengths() {
        let op = ClipOp {
            minimum: -2.0,
            maximum: 2.0,
        };
        for len in [1, 3, 4, 15, 16, 17, 63, 64, 65] {
            let src: Vec<f32> = (0..len).map(|i| (i as f32) - (len as f32 / 2.0)).collect();
            let mut dst = vec![0.0f32; len];
            op.apply_f32_bulk(&src, &mut dst);
            for (idx, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
                let expected = clamp_nan_propagating(s, -2.0, 2.0);
                assert_eq!(expected.to_bits(), d.to_bits(), "len={len} idx={idx}");
            }
        }
    }

    #[test]
    fn relu_f16_via_widen_narrow() {
        // Test the full dispatch path for f16.
        let values_f32 = [f32::NAN, -1.0, 0.0, 1.0, -0.5, 2.0, -3.0, 0.5];
        let src: Vec<u16> = values_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let mut dst = vec![0u16; src.len()];

        let op = ReluOp;

        #[cfg(target_arch = "aarch64")]
        unsafe {
            elementwise_f16_neon(&op, &src, &mut dst);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                let v = half::f16::from_bits(s).to_f32();
                *d = half::f16::from_f32(op.apply_f32_scalar(v)).to_bits();
            }
        }

        let results: Vec<f32> = dst
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();
        assert!(results[0].is_nan(), "NaN must propagate through f16 path");
        assert_eq!(results[1], 0.0);
        assert_eq!(results[2], 0.0);
        assert_eq!(results[3], 1.0);
        assert_eq!(results[4], 0.0);
        assert_eq!(results[5], 2.0);
        assert_eq!(results[6], 0.0);
        assert_eq!(results[7], 0.5);
    }

    #[test]
    fn clip_f16_via_widen_narrow() {
        let values_f32 = [f32::NAN, -1.0, 0.0, 1.0, -0.5, 2.0, -3.0, 0.5];
        let src: Vec<u16> = values_f32
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let mut dst = vec![0u16; src.len()];

        let op = ClipOp {
            minimum: -0.5,
            maximum: 0.5,
        };

        #[cfg(target_arch = "aarch64")]
        unsafe {
            elementwise_f16_neon(&op, &src, &mut dst);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                let v = half::f16::from_bits(s).to_f32();
                *d = half::f16::from_f32(op.apply_f32_scalar(v)).to_bits();
            }
        }

        let results: Vec<f32> = dst
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();
        assert!(results[0].is_nan(), "NaN must propagate through f16 Clip");
        assert_eq!(results[1], -0.5);
        assert_eq!(results[2], 0.0);
        assert_eq!(results[3], 0.5);
        assert_eq!(results[4], -0.5);
        assert_eq!(results[5], 0.5);
        assert_eq!(results[6], -0.5);
        assert_eq!(results[7], 0.5);
    }

    #[test]
    fn relu_bf16_via_widen_narrow() {
        let values_f32 = [f32::NAN, -1.0, 0.0, 1.0];
        let src: Vec<u16> = values_f32
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect();
        let mut dst = vec![0u16; src.len()];

        let op = ReluOp;
        // Use dispatch_dense_bf16 logic manually.
        let mut f32_buf: Vec<f32> = src
            .iter()
            .map(|&bits| half::bf16::from_bits(bits).to_f32())
            .collect();
        let f32_src = f32_buf.clone();
        op.apply_f32_bulk(&f32_src, &mut f32_buf);
        for (d, &v) in dst.iter_mut().zip(f32_buf.iter()) {
            *d = half::bf16::from_f32(v).to_bits();
        }

        let results: Vec<f32> = dst
            .iter()
            .map(|&bits| half::bf16::from_bits(bits).to_f32())
            .collect();
        assert!(results[0].is_nan());
        assert_eq!(results[1], 0.0);
        assert_eq!(results[2], 0.0);
        assert_eq!(results[3], 1.0);
    }
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod x86_vector_parity_tests {
    use super::*;

    /// Every f32 shape that has ever made a `max`/`min` kernel disagree with
    /// the scalar form it claims to reproduce: both zeros, both infinities, a
    /// quiet and a signalling NaN with distinct payloads, the subnormal
    /// boundary, and the extremes.
    fn special_values() -> Vec<f32> {
        let mut v = vec![
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(0x0000_0001), // smallest subnormal
            f32::from_bits(0x8000_0001), // smallest negative subnormal
            f32::from_bits(0x7FC0_0000), // quiet NaN
            f32::from_bits(0x7FC0_DEAD), // quiet NaN, distinct payload
            f32::from_bits(0xFFC0_0000), // negative quiet NaN
            f32::from_bits(0x7F80_0001), // signalling NaN
            1e-30,
            -1e-30,
            1e30,
            -1e30,
            0.5,
            -0.5,
            3.0,
            -3.0,
        ];
        // Pad past the 32-wide unrolled body so the bulk loop, the 8-wide
        // remainder and the scalar tail are all exercised at several lengths.
        for i in 0..90 {
            v.push((i as f32) * 0.37 - 16.0);
        }
        v
    }

    /// Compare by **bits**, not by value: `-0.0 == 0.0` and `NaN != NaN` under
    /// `f32` equality, which are exactly the two cases where a wrong operand
    /// order hides.
    fn assert_bit_identical(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: lane {i} is {g:?} ({:#010x}), scalar says {w:?} ({:#010x})",
                g.to_bits(),
                w.to_bits()
            );
        }
    }

    /// The AVX2 Relu must be the scalar `if x < 0.0 { 0.0 } else { x }`, not
    /// merely "a relu". Swapping the `_mm256_max_ps` operands — the natural
    /// way to write it — turns every NaN into `+0.0` and `-0.0` into `+0.0`,
    /// and this fails on both (verified).
    #[test]
    fn relu_f32_avx2_is_bit_identical_to_the_scalar_form() {
        if !avx2_available() {
            eprintln!("skipping: host lacks avx2");
            return;
        }
        let src = special_values();
        // Sweep every length so the bulk / 8-wide / tail boundaries all land
        // on special values rather than only on the padding.
        for len in 0..=src.len() {
            let input = &src[..len];
            let want: Vec<f32> = input.iter().map(|&x| ReluOp.apply_f32_scalar(x)).collect();
            let mut got = vec![f32::from_bits(0xDEAD_BEEF); len];
            unsafe { relu_f32_avx2(input, &mut got) };
            assert_bit_identical(&got, &want, &format!("relu len {len}"));
        }
    }

    /// Same contract for Clip, over bound pairs that put the special values on
    /// both sides of both bounds — including a NaN bound, which ORT permits and
    /// which `vminps`/`vmaxps` resolve by their `SRC2` rule.
    #[test]
    fn clip_f32_avx2_is_bit_identical_to_the_scalar_form() {
        if !avx2_available() {
            eprintln!("skipping: host lacks avx2");
            return;
        }
        let src = special_values();
        let bounds = [
            (0.0f32, 6.0f32),
            (-1.0, 1.0),
            (-0.0, 0.0),
            (f32::NEG_INFINITY, f32::INFINITY),
            (-3.0, -1.0),
            (1.0, -1.0), // inverted: min > max, ONNX leaves this to the impl
            (f32::NAN, 1.0),
            (-1.0, f32::NAN),
        ];
        for (lo, hi) in bounds {
            for len in [0usize, 1, 7, 8, 9, 31, 32, 33, 64, 100, src.len()] {
                let input = &src[..len.min(src.len())];
                let want: Vec<f32> = input
                    .iter()
                    .map(|&x| clamp_nan_propagating(x, lo, hi))
                    .collect();
                let mut got = vec![f32::from_bits(0xDEAD_BEEF); input.len()];
                unsafe { clip_f32_avx2(input, &mut got, lo, hi) };
                assert_bit_identical(&got, &want, &format!("clip [{lo:?},{hi:?}] len {len}"));
            }
        }
    }

    /// The dispatcher, not just the kernel: `relu_f32_simd` is what `ReluOp`
    /// actually calls, and it must agree with the scalar form too. This is the
    /// test that would catch a wrong `#[cfg]` arm or a missing `return`.
    #[test]
    fn the_dispatched_relu_and_clip_agree_with_their_scalar_forms() {
        let src = special_values();
        let want_relu: Vec<f32> = src.iter().map(|&x| ReluOp.apply_f32_scalar(x)).collect();
        let mut got_relu = vec![0.0f32; src.len()];
        relu_f32_simd(&src, &mut got_relu);
        assert_bit_identical(&got_relu, &want_relu, "dispatched relu");

        let op = ClipOp {
            minimum: -2.0,
            maximum: 4.0,
        };
        let want_clip: Vec<f32> = src.iter().map(|&x| op.apply_f32_scalar(x)).collect();
        let mut got_clip = vec![0.0f32; src.len()];
        clip_f32_simd(&src, &mut got_clip, op.minimum, op.maximum);
        assert_bit_identical(&got_clip, &want_clip, "dispatched clip");
    }
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod x86_stage_tests {
    use super::*;

    /// Reference: the scalar `half` loop the F16C path replaced.
    fn scalar_f16(op: &dyn ElementwiseOp, src: &[u16]) -> Vec<u16> {
        src.iter()
            .map(|&s| {
                let v = half::f16::from_bits(s).to_f32();
                half::f16::from_f32(op.apply_f32_scalar(v)).to_bits()
            })
            .collect()
    }

    /// Reference: the scalar `half` loop the AVX2 bf16 path replaced.
    fn scalar_bf16(op: &dyn ElementwiseOp, src: &[u16]) -> Vec<u16> {
        src.iter()
            .map(|&s| {
                let v = half::bf16::from_bits(s).to_f32();
                half::bf16::from_f32(op.apply_f32_scalar(v)).to_bits()
            })
            .collect()
    }

    fn vector_f16(op: &dyn ElementwiseOp, src: &[u16]) -> Vec<u16> {
        let mut dst = vec![0u16; src.len()];
        // SAFETY: guarded by the caller's `available()` check.
        unsafe { elementwise_f16_f16c(op, src, &mut dst) };
        dst
    }

    fn vector_bf16(op: &dyn ElementwiseOp, src: &[u16]) -> Vec<u16> {
        let mut dst = vec![0u16; src.len()];
        // SAFETY: guarded by the caller's `available()` check.
        unsafe { elementwise_bf16_avx2(op, src, &mut dst) };
        dst
    }

    /// Every one of the 65536 f16 bit patterns — normals, subnormals, ±0, ±Inf,
    /// and every NaN payload — must come back bit-identical to the scalar path.
    /// Run over the full space at once so the input spans many staging chunks.
    /// 65536 is a whole number of chunks and of vectors, so this test says
    /// nothing about tails; `..._handle_every_tail_length` below covers those.
    #[test]
    fn f16_vector_path_is_bit_identical_over_the_whole_domain() {
        if !crate::dtype::f16c::available() {
            eprintln!("skipping: host lacks f16c/avx2");
            return;
        }
        let all: Vec<u16> = (0..=u16::MAX).collect();
        for (name, op) in ops() {
            let want = scalar_f16(op.as_ref(), &all);
            let got = vector_f16(op.as_ref(), &all);
            for i in 0..all.len() {
                assert_eq!(
                    got[i], want[i],
                    "{name}: f16 input {:#06x} gave {:#06x}, scalar gives {:#06x}",
                    all[i], got[i], want[i]
                );
            }
        }
    }

    /// Same, for bf16.
    #[test]
    fn bf16_vector_path_is_bit_identical_over_the_whole_domain() {
        if !crate::dtype::bf16x::available() {
            eprintln!("skipping: host lacks avx2");
            return;
        }
        let all: Vec<u16> = (0..=u16::MAX).collect();
        for (name, op) in ops() {
            let want = scalar_bf16(op.as_ref(), &all);
            let got = vector_bf16(op.as_ref(), &all);
            for i in 0..all.len() {
                assert_eq!(
                    got[i], want[i],
                    "{name}: bf16 input {:#06x} gave {:#06x}, scalar gives {:#06x}",
                    all[i], got[i], want[i]
                );
            }
        }
    }

    /// Lengths around the staging-chunk boundary and the 8-wide vector width,
    /// so a tail-handling bug cannot hide behind a 65536-element run.
    #[test]
    fn f16_and_bf16_vector_paths_handle_every_tail_length() {
        if !crate::dtype::f16c::available() {
            eprintln!("skipping: host lacks f16c/avx2");
            return;
        }
        let pattern: Vec<u16> = (0..=u16::MAX).step_by(7).collect();
        let mut lengths: Vec<usize> = (0..=17).collect();
        for base in [F16_STAGE_CHUNK, 2 * F16_STAGE_CHUNK] {
            for d in 0..=8 {
                lengths.push(base - d);
                lengths.push(base + d);
            }
        }
        for (name, op) in ops() {
            for &len in &lengths {
                let src = &pattern[..len.min(pattern.len())];
                assert_eq!(
                    vector_f16(op.as_ref(), src),
                    scalar_f16(op.as_ref(), src),
                    "{name}: f16 mismatch at length {len}"
                );
                assert_eq!(
                    vector_bf16(op.as_ref(), src),
                    scalar_bf16(op.as_ref(), src),
                    "{name}: bf16 mismatch at length {len}"
                );
            }
        }
    }

    fn ops() -> Vec<(&'static str, Box<dyn ElementwiseOp>)> {
        vec![
            ("Relu", Box::new(ReluOp)),
            (
                "Clip(-1,1)",
                Box::new(ClipOp {
                    minimum: -1.0,
                    maximum: 1.0,
                }),
            ),
            (
                "Clip(0,inf)",
                Box::new(ClipOp {
                    minimum: 0.0,
                    maximum: f32::INFINITY,
                }),
            ),
        ]
    }
}
