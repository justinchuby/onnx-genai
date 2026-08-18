//! `Softmax`: numerically stable softmax for f32 (`docs/architecture/ORT2.md` §4.4).
//!
//! ## Two opset semantics (both implemented, selected by opset)
//!
//! ONNX changed `Softmax`'s definition at opset 13:
//!
//! * **opset ≥ 13** ([`SoftmaxKernel`] with `coerce_2d = false`): `axis` is the
//!   single reduction axis — softmax is normalized along that one axis.
//! * **opset ≤ 12** ([`SoftmaxKernel`] with `coerce_2d = true`): the input is
//!   coerced to a 2D matrix `[d_0·…·d_{axis-1}, d_axis·…·d_{n-1}]` and softmax
//!   is taken over each row (the *entire* flattened trailing block), not just
//!   the `axis` dimension.
//!
//! The two definitions coincide exactly when `axis` is the last dimension
//! (every trailing block is then a single axis). They diverge for `axis != last`,
//! so applying the opset-13 kernel to an opset-12 node silently produced wrong
//! results — the advisory this kernel now closes. The registry keys the two
//! factories at `since_version` 1 (legacy) and 13 (per-axis); the provider's
//! opset-aware lookup selects the correct one.
//!
//! Stability: each reduction slice subtracts its max before `exp`, so large
//! logits (e.g. masked-attention `-inf`/`1e9` fills) never overflow.

use crate::dtype::{
    output_direct_write_eligible, slice_byte_range, to_dense_f32_widen, write_dense_f32_narrow,
};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::check_arity;
use crate::strided::numel;

/// f32 Softmax kernel carrying the raw `axis` attribute and the opset semantics.
pub struct SoftmaxKernel {
    axis: i64,
    /// `true` for opset ≤ 12 (coerce-to-2D over the flattened trailing block);
    /// `false` for opset ≥ 13 (normalize over the single `axis`).
    coerce_2d: bool,
}

/// Factory for the opset ≥ 13 per-axis `Softmax` (`axis` default -1).
pub struct SoftmaxFactory;

/// Factory for the legacy opset ≤ 12 coerce-to-2D `Softmax` (`axis` default 1).
pub struct SoftmaxLegacyFactory;

impl KernelFactory for SoftmaxFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
        Ok(Box::new(SoftmaxKernel {
            axis,
            coerce_2d: false,
        }))
    }
}

impl KernelFactory for SoftmaxLegacyFactory {
    fn create(&self, node: &Node, _shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let axis = node.attr("axis").and_then(|a| a.as_int()).unwrap_or(1);
        Ok(Box::new(SoftmaxKernel {
            axis,
            coerce_2d: true,
        }))
    }
}

/// Rows below which the row-major softmax stays on one thread. A row is at
/// least a few hundred bytes of streaming work, so the crossover is set by the
/// pool's wake-up cost rather than by cache residency; 64 rows of any realistic
/// width clears it comfortably.
const MIN_PARALLEL_SOFTMAX_ROWS: usize = 64;

/// Elements below which the row-major softmax stays on one thread, so that a
/// tall-and-very-thin tensor (many rows of 2) does not pay for a fan-out that
/// buys nothing.
const MIN_PARALLEL_SOFTMAX_ELEMENTS: usize = 16 * 1024;

/// Numerically stable row-major softmax of `n` contiguous rows of `d` elements,
/// in place.
///
/// This is the shape every attention score matrix has, and the reason the
/// operator is worth vectorising at all: the scalar form costs one `f32::exp`
/// libm call per element. With `mlas` this hands each row to `MlasComputeSoftmax`
/// — the *same* primitive ONNX Runtime's own `Softmax` and attention kernels
/// use, which finds the row max, evaluates `exp` and normalizes in one pass
/// over the row.
///
/// The `exp` is only vectorised when the optional `mlas` feature is on, which
/// it is **not** by default. The shipping build evaluates `f32::exp` one
/// element at a time, and that is worth about 9x against ORT on a standalone
/// `Softmax`. Fixing it is tracked separately; this comment previously claimed
/// an 8-lane polynomial unconditionally, which was only ever true of the MLAS
/// arm.
///
/// Rows are independent, so the outer loop fans out across the shared Rayon
/// pool once the tensor is large enough to pay for it. ORT parallelizes the
/// identical loop; leaving it serial is what made this kernel lose by a factor
/// that *grew* with the thread count.
#[cfg(test)]
pub(crate) fn softmax_rows_in_place(data: &mut [f32], n: usize, d: usize) {
    scale_mask_softmax_rows(data, n, d, 1.0, None);
}

/// Broadcast plan for the additive attention mask, resolved once per run.
///
/// Holds the dense mask together with the effective stride of every score axis
/// into it (0 on broadcast axes), so a worker can locate any score row's mask
/// row from the row index alone — no shared cursor, no serial walk.
#[derive(Clone, Copy)]
pub(crate) struct MaskPlan<'a> {
    pub(crate) values: &'a [f32],
    pub(crate) eff: &'a [i64],
    pub(crate) scores_shape: &'a [usize],
}

impl MaskPlan<'_> {
    /// `(base, step)`: flat index of score row `row`'s first mask element, and
    /// the mask step per score column (0 when the mask broadcasts along the
    /// key axis, 1 when it is contiguous there).
    fn row_offset(&self, row: usize) -> (usize, usize) {
        let rank = self.scores_shape.len();
        if rank == 0 {
            return (0, 0);
        }
        let mut base = 0i64;
        let mut rem = row;
        // The row index is row-major over every axis but the last, so peel the
        // innermost leading axis first and work outward.
        for axis in (0..rank - 1).rev() {
            let dim = self.scores_shape[axis].max(1);
            base += self.eff[axis] * (rem % dim) as i64;
            rem /= dim;
        }
        (base as usize, self.eff[rank - 1].max(0) as usize)
    }
}

/// Rows are processed in tiles of about this many bytes so a tile stays in a
/// core's private cache between the scale/mask write and the softmax's two
/// reads, while still handing MLAS several rows per call.
const ROW_TILE_BYTES: usize = 32 * 1024;

/// `softmax(scores * scale + mask)` over the last axis, in place and in one
/// parallel pass.
pub(crate) fn scale_mask_softmax_rows(
    data: &mut [f32],
    outer: usize,
    d: usize,
    scale: f32,
    mask: Option<MaskPlan<'_>>,
) {
    if d == 0 || outer == 0 {
        return;
    }
    debug_assert_eq!(data.len(), outer * d);
    match parallel_rows_per_task(outer, d) {
        Some(rows_per_task) => {
            use rayon::prelude::*;
            data.par_chunks_mut(rows_per_task * d)
                .enumerate()
                .for_each(|(chunk, rows)| {
                    scale_mask_softmax_serial(rows, chunk * rows_per_task, d, scale, mask);
                });
        }
        None => scale_mask_softmax_serial(data, 0, d, scale, mask),
    }
}

/// Serial worker for one contiguous run of rows starting at global row
/// `first_row` (needed only to index the mask).
fn scale_mask_softmax_serial(
    data: &mut [f32],
    first_row: usize,
    d: usize,
    scale: f32,
    mask: Option<MaskPlan<'_>>,
) {
    let rows = data.len() / d;
    if rows == 0 {
        return;
    }
    let rows_per_tile = (ROW_TILE_BYTES / (d * size_of::<f32>())).clamp(1, rows);
    let mut row0 = 0usize;
    while row0 < rows {
        let tile_rows = rows_per_tile.min(rows - row0);
        let tile = &mut data[row0 * d..(row0 + tile_rows) * d];
        match mask {
            Some(plan) => {
                for (i, row) in tile.chunks_mut(d).enumerate() {
                    let (base, step) = plan.row_offset(first_row + row0 + i);
                    match step {
                        // Mask broadcasts along the key axis: one value per row.
                        0 => {
                            let m = plan.values[base];
                            for v in row.iter_mut() {
                                *v = *v * scale + m;
                            }
                        }
                        // The common case — a contiguous mask row.
                        //
                        // `*v * scale + m` replaces a `*= scale` pass followed
                        // by a `+= m` pass, and is bit-identical to it only
                        // because rustc does not contract to an FMA without an
                        // explicit `f32::mul_add`. That is what lets the
                        // attention parity tests assert exact equality against
                        // the two-pass reference; enabling contraction
                        // globally would relax it to a tolerance.
                        1 => {
                            for (v, &m) in row.iter_mut().zip(&plan.values[base..base + d]) {
                                *v = *v * scale + m;
                            }
                        }
                        // Defensive: a dense row-major mask always yields a
                        // last-axis stride of 0 or 1, so this is unreachable
                        // today. It is kept correct rather than asserted away
                        // so a future non-contiguous mask source cannot
                        // silently produce wrong offsets.
                        step => {
                            for (j, v) in row.iter_mut().enumerate() {
                                *v = *v * scale + plan.values[base + j * step];
                            }
                        }
                    }
                }
            }
            None if scale != 1.0 => {
                for v in tile.iter_mut() {
                    *v *= scale;
                }
            }
            None => {}
        }
        softmax_rows_serial(tile, tile_rows, d);
        row0 += tile_rows;
    }
}

/// Out-of-place row-major softmax: `src` and `dst` are `n × d` and disjoint.
///
/// `dst` is written straight from `src` in a single traversal - the row max,
/// `exp(row - max)` and the normalization all read `src` and write `dst`, so no
/// element of `src` is ever copied into `dst` first. That copy was a whole extra
/// read+write pass over the tensor - 66 MiB of traffic per inference on a 33 MiB
/// prefill logit block - and it bought nothing: ORT never pays it because it
/// hands `MlasComputeSoftmax` the graph input and output buffers directly, and
/// this path now does the same. The result is bit-identical to a
/// `copy(src -> dst)` followed by the in-place reducer - the existing
/// `the_parallel_fan_out_is_bit_identical_to_the_serial_path` test asserts that
/// equality on both the MLAS and the pure-Rust builds.
pub(crate) fn softmax_rows(src: &[f32], dst: &mut [f32], n: usize, d: usize) {
    debug_assert_eq!(src.len(), n * d);
    debug_assert_eq!(dst.len(), n * d);
    if d == 0 || n == 0 {
        return;
    }
    if let Some(rows_per_task) = parallel_rows_per_task(n, d) {
        use rayon::prelude::*;
        let block = rows_per_task * d;
        dst.par_chunks_mut(block)
            .zip(src.par_chunks(block))
            .for_each(|(out, inp)| {
                softmax_rows_serial_out(inp, out, out.len() / d, d);
            });
    } else {
        softmax_rows_serial_out(src, dst, n, d);
    }
}

/// `Some(rows_per_task)` when a fan-out is worth it, `None` to stay serial.
fn parallel_rows_per_task(n: usize, d: usize) -> Option<usize> {
    if n < MIN_PARALLEL_SOFTMAX_ROWS || n.saturating_mul(d) < MIN_PARALLEL_SOFTMAX_ELEMENTS {
        return None;
    }
    let workers = rayon::current_num_threads().max(1);
    if workers < 2 {
        return None;
    }
    // Whole rows per task, so each chunk is a valid `n' × d` sub-matrix and
    // MLAS is invoked once per chunk rather than once per row.
    Some(n.div_ceil(workers).max(1))
}

#[cfg(feature = "mlas")]
fn softmax_rows_serial(data: &mut [f32], n: usize, d: usize) {
    mlas_sys::compute_softmax_in_place(data, n, d);
}

/// Out-of-place counterpart of [`softmax_rows_serial`]: read `src`, write `dst`,
/// no copy. MLAS's non-log softmax streams `Input -> Output` and then rescales
/// `Output` alone, so this is bit-identical to `dst.copy_from_slice(src)`
/// followed by the in-place form.
#[cfg(feature = "mlas")]
fn softmax_rows_serial_out(src: &[f32], dst: &mut [f32], n: usize, d: usize) {
    mlas_sys::compute_softmax(src, dst, n, d);
}

/// Portable fallback with the same semantics: subtract the row max, `exp`,
/// normalize. Kept exact against the MLAS path by the parity tests below.
///
/// The scalar `f32::exp` this used to call was one libm invocation per element
/// and was the entire gap against ORT's vectorised CPU softmax (9-11x on the
/// attention shapes). Both the in-place and the out-of-place forms now route
/// every row through the *one* [`softmax_row_core`] routine, which evaluates
/// `exp` eight lanes at a time on AVX2+FMA (with a scalar fallback). Sharing a
/// single core is also what keeps the two forms bit-identical - see
/// `out_of_place_softmax_is_bit_identical_on_pathological_rows`.
#[cfg(not(feature = "mlas"))]
fn softmax_rows_serial(data: &mut [f32], n: usize, d: usize) {
    for row in data.chunks_mut(d) {
        let ptr = row.as_mut_ptr();
        // SAFETY: `row` is exactly `d` elements, so `ptr` is valid for `d` f32
        // reads and writes. This is the in-place case (`src == dst`), which
        // `softmax_row_core` supports because it loads each block from the
        // source before storing the corresponding destination block, so no
        // lane is ever read back after it is written.
        unsafe { softmax_row_core(ptr as *const f32, ptr, d) };
    }
    let _ = n;
}

/// Portable out-of-place counterpart of [`softmax_rows_serial`]: read `src`,
/// write `dst`, no copy. Bit-identical to `dst.copy_from_slice(src)` followed by
/// the in-place form because it calls the *same* [`softmax_row_core`] per row,
/// reading each `src` element in place of the copied `dst` element it would
/// otherwise have read.
#[cfg(not(feature = "mlas"))]
fn softmax_rows_serial_out(src: &[f32], dst: &mut [f32], n: usize, d: usize) {
    for (srow, drow) in src.chunks(d).zip(dst.chunks_mut(d)) {
        debug_assert_eq!(srow.len(), d);
        debug_assert_eq!(drow.len(), d);
        // SAFETY: `srow` and `drow` are each exactly `d` elements. `softmax_rows`
        // documents `src`/`dst` as disjoint, so these do not overlap; the core
        // reads `d` floats from the first pointer and writes `d` to the second.
        unsafe { softmax_row_core(srow.as_ptr(), drow.as_mut_ptr(), d) };
    }
    let _ = n;
}

/// The single row reducer shared by the in-place and out-of-place non-MLAS
/// softmax paths: find the row max, write `exp(src - max)` into `dst`, and
/// normalize by the row sum. Reading from `src` and writing to `dst` through
/// one routine is what makes those two paths bit-identical (including on the
/// non-finite rows the parity tests pin), because the max/`exp`/normalize
/// arithmetic is literally the same instructions in both.
///
/// On x86 with AVX2+FMA it evaluates `exp` eight lanes at a time; elsewhere,
/// and on the sub-8 tail of every row, it uses the scalar libm `exp`.
///
/// # Safety
/// `src` must be valid for `d` f32 reads and `dst` for `d` f32 writes. The two
/// regions must be either identical (`src == dst`, the in-place case) or fully
/// disjoint - never partially overlapping - because the AVX2 kernel loads a
/// whole 8-lane block from `src` before storing the matching block to `dst`.
#[cfg(not(feature = "mlas"))]
#[inline]
unsafe fn softmax_row_core(src: *const f32, dst: *mut f32, d: usize) {
    if d == 0 {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if softmax_avx2::available() {
            // SAFETY: the branch proves the running CPU has AVX2+FMA, and the
            // caller's contract on `src`/`dst`/`d` is forwarded unchanged.
            unsafe { softmax_avx2::softmax_row(src, dst, d) };
            return;
        }
    }
    // SAFETY: same pointer/length contract; the scalar path performs the
    // identical reads and writes without any SIMD.
    unsafe { softmax_row_scalar(src, dst, d) };
}

/// Scalar reference reducer: one libm `exp` per element. This is the fallback
/// for non-x86 targets, for x86 without AVX2+FMA, and (in the AVX2 kernel) for
/// the tail of a row whose length is not a multiple of 8.
///
/// # Safety
/// Same contract as [`softmax_row_core`].
#[cfg(not(feature = "mlas"))]
unsafe fn softmax_row_scalar(src: *const f32, dst: *mut f32, d: usize) {
    let mut max = f32::NEG_INFINITY;
    for i in 0..d {
        // SAFETY: `i < d` and `src` is valid for `d` reads.
        let v = unsafe { *src.add(i) };
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for i in 0..d {
        // SAFETY: `i < d`; `src`/`dst` are valid for `d` reads/writes and the
        // `src` lane is read before the `dst` lane is written, so an exact
        // in-place alias is fine.
        let e = (unsafe { *src.add(i) } - max).exp();
        unsafe { *dst.add(i) = e };
        sum += e;
    }
    let inv = 1.0 / sum;
    for i in 0..d {
        // SAFETY: `i < d` and `dst` is valid for `d` writes.
        unsafe { *dst.add(i) *= inv };
    }
}

/// AVX2+FMA row softmax: the same max / `exp(x - max)` / normalize the scalar
/// path performs, but with `exp` evaluated on eight f32 lanes at once.
///
/// The polynomial is Cephes' single-precision `expf` (range-reduce
/// `x = k·ln2 + r`, a degree-6 minimax on `r`, then scale by `2^k`), which is
/// accurate to ~1 ULP - comfortably inside the `1e-6` softmax tolerance the
/// tests assert against libm. Because softmax only ever evaluates `exp(v - max)`
/// with `v - max <= 0`, the reduced argument never overflows; the non-finite
/// lanes (`-inf` -> `0`, `NaN` -> `NaN`) are patched explicitly so a fully
/// masked row still normalizes to `NaN` and a partially masked row's masked
/// positions are exactly `0`, matching the scalar reference bit for bit through
/// the shared normalization.
#[cfg(all(
    not(feature = "mlas"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod softmax_avx2 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Runtime detection: both AVX2 (256-bit integer/float ops for the `2^k`
    /// exponent build) and FMA (the polynomial's fused multiply-adds).
    #[inline]
    pub fn available() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    /// `exp` on eight lanes for the softmax domain (`x <= 0`, plus `-inf`/`NaN`).
    ///
    /// # Safety
    /// The caller must run on a CPU with AVX2+FMA (guaranteed by [`available`]).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    #[allow(clippy::excessive_precision)]
    unsafe fn exp8(x: __m256) -> __m256 {
        let log2e = _mm256_set1_ps(core::f32::consts::LOG2_E);
        // ln2 split into a high part that is exact in f32 and a low
        // correction (Cody-Waite), so `x - k*ln2` keeps its low bits.
        let ln2_hi = _mm256_set1_ps(0.693_359_375_f32);
        let ln2_lo = _mm256_set1_ps(-2.121_944_40e-4_f32);
        let one = _mm256_set1_ps(1.0);

        // Cephes `expf` polynomial coefficients for `exp(r)` on
        // `|r| <= ln2/2`.
        let c0 = _mm256_set1_ps(1.987_569_15e-4_f32);
        let c1 = _mm256_set1_ps(1.398_199_95e-3_f32);
        let c2 = _mm256_set1_ps(8.333_451_9e-3_f32);
        let c3 = _mm256_set1_ps(4.166_579_6e-2_f32);
        let c4 = _mm256_set1_ps(1.666_666_5e-1_f32);
        let c5 = _mm256_set1_ps(5.000_000_1e-1_f32);

        // Below this, f32 `exp` has underflowed to (sub)normals we flush to
        // 0; `-inf` lands here too and must become exactly 0.
        let underflow = _mm256_set1_ps(-87.336_544_f32);

        let is_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x);
        // Ordered `<=` so `NaN` reports false here (it is handled above).
        let is_zero = _mm256_cmp_ps::<_CMP_LE_OQ>(x, underflow);

        // Clamp so `-inf`/`NaN` cannot poison the exponent build; those
        // lanes are overwritten below.
        let xc = _mm256_max_ps(x, underflow);

        // k = round(x * log2e); r = x - k*ln2 (two-part).
        let k = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm256_mul_ps(xc, log2e),
        );
        let r = _mm256_fnmadd_ps(k, ln2_hi, xc);
        let r = _mm256_fnmadd_ps(k, ln2_lo, r);

        // Horner: exp(r) = ((((((c0*r+c1)*r+c2)*r+c3)*r+c4)*r+c5)*r^2 + r + 1.
        let mut p = _mm256_fmadd_ps(c0, r, c1);
        p = _mm256_fmadd_ps(p, r, c2);
        p = _mm256_fmadd_ps(p, r, c3);
        p = _mm256_fmadd_ps(p, r, c4);
        p = _mm256_fmadd_ps(p, r, c5);
        let r2 = _mm256_mul_ps(r, r);
        p = _mm256_fmadd_ps(p, r2, r);
        p = _mm256_add_ps(p, one);

        // 2^k by constructing the IEEE-754 exponent: (k + 127) << 23.
        let ki = _mm256_cvtps_epi32(k);
        let biased = _mm256_add_epi32(ki, _mm256_set1_epi32(127));
        let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(biased));
        let mut y = _mm256_mul_ps(p, pow2);

        // Underflow / -inf -> exactly 0 (andnot zeroes the masked lanes).
        y = _mm256_andnot_ps(is_zero, y);
        // NaN input -> NaN output, so a poisoned row sum stays NaN.
        y = _mm256_blendv_ps(y, _mm256_set1_ps(f32::NAN), is_nan);
        y
    }

    /// Horizontal maximum of the eight lanes.
    ///
    /// # Safety
    /// AVX must be available (implied by AVX2 on the calling path).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn hmax(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let m = _mm_max_ps(lo, hi);
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        let m = _mm_max_ss(m, _mm_shuffle_ps::<0x55>(m, m));
        _mm_cvtss_f32(m)
    }

    /// Horizontal sum of the eight lanes (NaN-propagating).
    ///
    /// # Safety
    /// AVX must be available (implied by AVX2 on the calling path).
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn hsum(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0x55>(s, s));
        _mm_cvtss_f32(s)
    }

    /// One row: max, then `exp(src - max)` into `dst`, then normalize.
    ///
    /// # Safety
    /// AVX2+FMA must be available. `src` is valid for `d` f32 reads and `dst`
    /// for `d` f32 writes; the two regions are identical or disjoint. Every
    /// 8-lane block is loaded from `src` before the matching block is stored to
    /// `dst`, so an exact in-place alias is sound.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax_row(src: *const f32, dst: *mut f32, d: usize) {
        unsafe {
            // Pass 1: row maximum. Feeding `v` as the first `max_ps` operand
            // makes a `NaN` lane return the running accumulator, so `NaN`
            // inputs are ignored exactly as `if v > max` ignores them.
            let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
            let mut i = 0usize;
            while i + 8 <= d {
                let v = _mm256_loadu_ps(src.add(i));
                vmax = _mm256_max_ps(v, vmax);
                i += 8;
            }
            let mut max = hmax(vmax);
            while i < d {
                let v = *src.add(i);
                if v > max {
                    max = v;
                }
                i += 1;
            }

            // Pass 2: exp(v - max) into `dst`, accumulating the row sum. A NaN
            // lane propagates into the sum, so a poisoned row normalizes to NaN.
            let vmaxb = _mm256_set1_ps(max);
            let mut vsum = _mm256_setzero_ps();
            i = 0;
            while i + 8 <= d {
                let v = _mm256_loadu_ps(src.add(i));
                let e = exp8(_mm256_sub_ps(v, vmaxb));
                _mm256_storeu_ps(dst.add(i), e);
                vsum = _mm256_add_ps(vsum, e);
                i += 8;
            }
            let mut sum = hsum(vsum);
            while i < d {
                let e = (*src.add(i) - max).exp();
                *dst.add(i) = e;
                sum += e;
                i += 1;
            }

            // Pass 3: normalize by the reciprocal row sum.
            let inv = 1.0f32 / sum;
            let vinv = _mm256_set1_ps(inv);
            i = 0;
            while i + 8 <= d {
                let v = _mm256_loadu_ps(dst.add(i));
                _mm256_storeu_ps(dst.add(i), _mm256_mul_ps(v, vinv));
                i += 8;
            }
            while i < d {
                *dst.add(i) *= inv;
                i += 1;
            }
        }
    }
}

/// Softmax `n` independent contiguous rows of `axis_dim` elements each over the
/// stride-`inner` interleaving: element `a` of slice `(o, i)` lives at
/// `o·axis_dim·inner + a·inner + i`. With `inner == 1` this is a plain
/// row-major softmax; with `inner > 1` it reduces along an interior axis.
///
/// The `FusedAttention` kernel (`kernels::fused_attention`) shares the same
/// numerically-stable reducer through [`scale_mask_softmax_rows`], which folds
/// its scale and mask into this module's parallel row driver instead of
/// duplicating the max-subtract/exp/normalize loop.
///
/// `inner == 1` is the overwhelmingly common case (softmax over the last axis)
/// and is routed to [`softmax_rows`]; only a genuine interior-axis
/// reduction takes the strided loop, where the gather makes vectorisation
/// unprofitable anyway.
pub(crate) fn softmax_slices(
    x: &[f32],
    out: &mut [f32],
    outer: usize,
    axis_dim: usize,
    inner: usize,
) {
    if inner == 1 {
        let len = outer * axis_dim;
        softmax_rows(&x[..len], &mut out[..len], outer, axis_dim);
        return;
    }
    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis_dim * inner + i;
            let mut max = f32::NEG_INFINITY;
            for a in 0..axis_dim {
                let v = x[base + a * inner];
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0f32;
            for a in 0..axis_dim {
                let e = (x[base + a * inner] - max).exp();
                out[base + a * inner] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for a in 0..axis_dim {
                out[base + a * inner] *= inv;
            }
        }
    }
}

impl Kernel for SoftmaxKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("Softmax", inputs, outputs, 1, 1, 1)?;
        let x = to_dense_f32_widen("Softmax", &inputs[0])?;
        let shape = inputs[0].shape;
        let rank = shape.len();
        if rank == 0 {
            return Err(EpError::KernelFailed(
                "Softmax: input must have rank >= 1".into(),
            ));
        }
        let axis = if self.axis < 0 {
            self.axis + rank as i64
        } else {
            self.axis
        };
        if axis < 0 || axis as usize >= rank {
            return Err(EpError::KernelFailed(format!(
                "Softmax: axis {} out of range for rank {rank}",
                self.axis
            )));
        }
        let axis = axis as usize;
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let slices = if self.coerce_2d {
                crate::trace::product(shape[..axis].iter().copied())
            } else {
                crate::trace::product(shape[..axis].iter().chain(&shape[axis + 1..]).copied())
            };
            // Per element: subtract, exp, sum and normalization multiply; one
            // reciprocal per reduction slice. Max comparisons are not FLOPs.
            (inputs[0].numel() as u64)
                .saturating_mul(4)
                .saturating_add(slices)
        });

        let len = numel(shape);
        if len == 0 {
            // Nothing to write, and `from_raw_parts_mut` below requires a
            // non-null pointer even for a zero-length slice.
            return Ok(());
        }
        // Softmax reads every input element before writing the corresponding
        // output, so a f32 output buffer that does not alias the widened input
        // can be written straight through - saving a full-tensor scratch
        // allocation and copy on the dominant f32 path. When `x` borrows the
        // input (no widening happened) the ranges can genuinely overlap, which
        // is why the disjointness is checked rather than assumed.
        let read_ranges = [slice_byte_range(x.as_ref())];
        let direct = output_direct_write_eligible(&mut outputs[0], len, &read_ranges);
        let mut owned;
        let out: &mut [f32] = if direct {
            // SAFETY: `output_direct_write_eligible` proved the buffer is a
            // host-accessible contiguous Float32 tensor of exactly `len`
            // elements whose byte range is disjoint from the input we read.
            unsafe { std::slice::from_raw_parts_mut(outputs[0].data_ptr_mut::<f32>(), len) }
        } else {
            owned = vec![0.0f32; len];
            &mut owned
        };

        if self.coerce_2d {
            // opset ≤ 12: coerce to 2D `[d_0·…·d_{axis-1}, d_axis·…·d_{n-1}]`
            // and softmax each row over the whole flattened trailing block.
            let rows: usize = shape[..axis].iter().product();
            let cols: usize = shape[axis..].iter().product();
            // Trailing block is contiguous, so `inner == 1`.
            softmax_slices(&x, out, rows, cols, 1);
        } else {
            // opset ≥ 13: normalize over the single `axis`, viewing the tensor
            // as `[outer, axis_dim, inner]`.
            let axis_dim = shape[axis];
            let outer: usize = shape[..axis].iter().product();
            let inner: usize = shape[axis + 1..].iter().product();
            softmax_slices(&x, out, outer, axis_dim, inner);
        }
        if direct {
            Ok(())
        } else {
            write_dense_f32_narrow("Softmax", &mut outputs[0], out)
        }
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    fn run(axis: i64, x: &Owned, out: &mut Owned) {
        SoftmaxKernel {
            axis,
            coerce_2d: false,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
    }

    fn run_legacy(axis: i64, x: &Owned, out: &mut Owned) {
        SoftmaxKernel {
            axis,
            coerce_2d: true,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
    }

    fn approx(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-6, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn softmax_last_axis_2d() {
        // [2,3], axis 1. Row [1,2,3]: softmax = [0.09003, 0.24473, 0.66524].
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 1., 2., 3.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        run(1, &x, &mut out);
        let e = [0.090_030_57, 0.244_728_47, 0.665_240_96];
        let mut want = e.to_vec();
        want.extend_from_slice(&e);
        approx(&out.to_f32(), &want);
        // Each row sums to 1.
        let r = out.to_f32();
        assert!((r[0] + r[1] + r[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_axis0() {
        // [2,2], axis 0 reduces over rows (column-wise softmax).
        let x = Owned::f32(&[2, 2], &[1., 2., 1., 2.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run(0, &x, &mut out);
        // Each column has equal entries -> 0.5 each.
        approx(&out.to_f32(), &[0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn softmax_negative_axis() {
        let x = Owned::f32(&[1, 3], &[0., 0., 0.]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        run(-1, &x, &mut out);
        approx(&out.to_f32(), &[1. / 3., 1. / 3., 1. / 3.]);
    }

    #[test]
    fn softmax_numerically_stable_large_values() {
        // Without max-subtraction these overflow to inf; the result must stay
        // finite and (nearly) one-hot on the largest logit.
        let x = Owned::f32(&[1, 3], &[1000.0, 1001.0, 1002.0]);
        let mut out = Owned::zeros_f32(&[1, 3]);
        run(1, &x, &mut out);
        let r = out.to_f32();
        assert!(r.iter().all(|v| v.is_finite()));
        assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Same gaps as [0,1,2] -> [0.09003, 0.24473, 0.66524].
        approx(&r, &[0.090_030_57, 0.244_728_47, 0.665_240_96]);
    }

    #[test]
    fn softmax_batched_last_axis_4d() {
        // [1,1,2,2] with axis -1 — the BERT attention shape pattern.
        let x = Owned::f32(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[1, 1, 2, 2]);
        run(-1, &x, &mut out);
        let r = out.to_f32();
        // row [1,2] and row [3,4] both softmax to [0.26894, 0.73106].
        approx(&r, &[0.268_941_43, 0.731_058_6, 0.268_941_43, 0.731_058_6]);
    }

    #[test]
    fn softmax_opset13_default_axis_is_last_dimension() {
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        SoftmaxFactory
            .create(
                &Node::new(
                    onnx_runtime_ir::NodeId(0),
                    "Softmax",
                    Vec::new(),
                    Vec::new(),
                ),
                &[],
            )
            .unwrap()
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        approx(
            &out.to_f32(),
            &[0.268_941_43, 0.731_058_6, 0.268_941_43, 0.731_058_6],
        );
    }

    #[test]
    fn softmax_opset12_axis0_coerces_to_single_row() {
        // [2,2], axis 0. opset≤12 coerces to `[1, 4]` (rows before axis 0 = 1)
        // and softmaxes the ENTIRE flattened tensor as one row — unlike the
        // opset-13 per-axis (column-wise) definition.
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        run_legacy(0, &x, &mut out);
        let r = out.to_f32();
        approx(&r, &[0.032_058_6, 0.087_144_32, 0.236_882_82, 0.643_914_2]);
        // The whole tensor is one softmax row → all elements sum to 1.
        assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_opset12_differs_from_opset13_when_axis_not_last() {
        // Same [2,2] axis-0 input: the opset-13 per-axis kernel normalizes each
        // column independently, so the two definitions must disagree here.
        let x = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let mut per_axis = Owned::zeros_f32(&[2, 2]);
        let mut legacy = Owned::zeros_f32(&[2, 2]);
        run(0, &x, &mut per_axis);
        run_legacy(0, &x, &mut legacy);
        // opset-13: each column [1,3] and [2,4] → [0.11920, 0.88080].
        approx(
            &per_axis.to_f32(),
            &[0.119_202_92, 0.119_202_92, 0.880_797_1, 0.880_797_1],
        );
        // The two kernels genuinely diverge (the bug this fix closes).
        let (a, b) = (per_axis.to_f32(), legacy.to_f32());
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3));
    }

    #[test]
    fn softmax_opset12_matches_opset13_on_last_axis() {
        // When axis == last dim, the coerce-to-2D and per-axis definitions
        // coincide exactly — the BERT-attention case.
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut per_axis = Owned::zeros_f32(&[2, 3]);
        let mut legacy = Owned::zeros_f32(&[2, 3]);
        run(-1, &x, &mut per_axis);
        run_legacy(-1, &x, &mut legacy);
        approx(&per_axis.to_f32(), &legacy.to_f32());
    }
    #[test]
    fn softmax_bf16_matches_widened_f32_reference() {
        let values = [-10.0, 0.0, 1.0, 80.0, -80.0, f32::INFINITY];
        let x = Owned::bf16(&[2, 3], &values);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 3]);
        run(-1, &x, &mut out);
        let rounded = x.to_bf16_as_f32();
        let mut reference = vec![0.0; rounded.len()];
        softmax_slices(&rounded, &mut reference, 2, 3, 1);
        let expected: Vec<_> = reference
            .into_iter()
            .map(half::bf16::from_f32)
            .map(half::bf16::to_f32)
            .collect();
        for (got, want) in out.to_bf16_as_f32().into_iter().zip(expected) {
            assert!(got == want || (got.is_nan() && want.is_nan()));
        }
    }
}

#[cfg(test)]
mod vectorized_tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    /// Independent scalar oracle, deliberately written the naive way: the
    /// implementation under test is allowed to use MLAS, a fan-out, or both.
    fn reference_rows(src: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * d];
        for r in 0..n {
            let row = &src[r * d..(r + 1) * d];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (i, &v) in row.iter().enumerate() {
                let e = (v - max).exp();
                out[r * d + i] = e;
                sum += e;
            }
            for i in 0..d {
                out[r * d + i] /= sum;
            }
        }
        out
    }

    fn synthetic(n: usize, d: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n * d)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) * 20.0 - 10.0
            })
            .collect()
    }

    /// The fan-out must not change a single bit relative to the serial path:
    /// rows are independent, so chunking cannot legitimately alter any result.
    /// Sizes straddle `MIN_PARALLEL_SOFTMAX_ROWS` and
    /// `MIN_PARALLEL_SOFTMAX_ELEMENTS` in both directions.
    #[test]
    fn the_parallel_fan_out_is_bit_identical_to_the_serial_path() {
        for (n, d) in [
            (1usize, 1usize),
            (1, 4096),
            (63, 512),   // below the row floor
            (64, 8),     // clears rows, below the element floor
            (64, 256),   // clears both
            (4096, 128), // comfortably parallel
            (129, 1000), // rows not divisible by any worker count
        ] {
            let src = synthetic(n, d, (n * 31 + d) as u32);
            let mut serial = src.clone();
            softmax_rows_serial(&mut serial, n, d);

            let mut fanned = src.clone();
            softmax_rows_in_place(&mut fanned, n, d);
            assert_eq!(serial, fanned, "in-place n={n} d={d}");

            let mut out_of_place = vec![f32::NAN; n * d];
            softmax_rows(&src, &mut out_of_place, n, d);
            assert_eq!(serial, out_of_place, "out-of-place n={n} d={d}");
        }
    }

    /// The out-of-place softmax derives every output element from `src` alone.
    /// It writes `dst` without first copying `src` into it, so if it ever read
    /// `dst` (the copy this kernel deliberately no longer makes) pre-existing
    /// garbage there would leak into the result. Poisoning `dst` with values
    /// that would wreck a softmax if mistaken for logits - `+inf` would win
    /// every row max and `NaN` would poison every sum - and asserting the
    /// result is bit-for-bit identical to running over a clean buffer pins the
    /// destination-independence down exactly, on both the MLAS and pure-Rust
    /// builds.
    #[test]
    fn out_of_place_softmax_never_reads_the_destination() {
        for (n, d) in [(1usize, 1usize), (3, 7), (64, 256), (129, 500)] {
            let src = synthetic(n, d, (7 * n + d) as u32);

            let mut clean = vec![0.0f32; n * d];
            softmax_rows(&src, &mut clean, n, d);

            let poison = [f32::INFINITY, f32::NAN, f32::NEG_INFINITY, 1e30];
            let mut garbage: Vec<f32> = (0..n * d).map(|i| poison[i % poison.len()]).collect();
            softmax_rows(&src, &mut garbage, n, d);

            assert_eq!(clean, garbage, "destination garbage leaked at n={n} d={d}");
        }
    }

    /// ...and the no-copy path must still be a softmax, not merely
    /// deterministic. Checks it against an independent scalar oracle with the
    /// destination pre-poisoned with `NaN`, so a regression that reads the
    /// poison or miscomputes the max/`exp`/normalize reduction surfaces here
    /// rather than only under the A/B harness.
    #[test]
    fn out_of_place_softmax_matches_the_reference() {
        for (n, d) in [(1usize, 1usize), (3, 7), (64, 256), (129, 500)] {
            let src = synthetic(n, d, (5 * n + 3 * d) as u32);
            let expect = reference_rows(&src, n, d);
            let mut dst: Vec<f32> = vec![f32::NAN; n * d];
            softmax_rows(&src, &mut dst, n, d);
            for (i, (g, e)) in dst.iter().zip(&expect).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-6,
                    "n={n} d={d} idx={i}: {g} vs {e} (out-of-place softmax wrong)"
                );
                assert!(g.is_finite(), "n={n} d={d} idx={i}: non-finite {g}");
            }
        }
    }
    /// The out-of-place form must be *bit*-identical to the copy-then-in-place
    /// form it replaced, not merely close, and the interesting inputs are the
    /// ones a tolerance-based check cannot see, because the in-place reducer is
    /// vectorised and this path is not, so their `exp` implementations have to
    /// agree on the non-finite lanes bit for bit: a fully masked row, where the
    /// row max is `-inf` and `exp(-inf - -inf)` is `exp(NaN)`, so a NaN whose
    /// payload and sign must match propagates through the whole row; a row whose
    /// max is `+inf`, likewise NaN; and rows carrying quiet, signalling and
    /// negative-payload NaNs, plus denormals. This repo has already shipped one
    /// bug of exactly this shape - a bf16 widen that raw-shifted instead of
    /// quieting, diverging from the reference on 126 of 65536 patterns - and the
    /// tolerance-based tests above would not catch its analogue here, so this
    /// compares raw bits.
    ///
    /// Not Miri-tractable, for two independent reasons, both measured rather
    /// than assumed. Miri reports `is_x86_feature_detected!("avx2") == false`,
    /// so it only ever runs the scalar fallback and cannot validate the AVX2
    /// kernel this test exists to police. And Miri deliberately returns a
    /// nondeterministic approximation for libm calls - two evaluations of
    /// `(-0.5f32).exp()` in one Miri process gave `0x3f1b4595` and
    /// `0x3f1b4599` - plus a nondeterministic NaN sign/payload per operation,
    /// so *any* bit-identity assertion over `exp` fails there by construction,
    /// with a seed-dependent lane. Hardware `exp` and hardware NaN propagation
    /// are both deterministic, which is why this holds off Miri.
    #[test]
    #[cfg_attr(miri, ignore = "Miri: no AVX2, and nondeterministic libm/NaN results")]
    fn out_of_place_softmax_is_bit_identical_on_pathological_rows() {
        // 8 is a whole vector body with no tail; 11 is a vector body plus a
        // 3-lane scalar tail; 6 is a pure tail with no vector body at all.
        // The tail widths matter: a NaN in a vectorized lane is forced to a
        // canonical NaN by the mask, while a NaN in a tail lane is whatever
        // libm returns, so a future change that stopped both forms sharing one
        // core would diverge on exactly these rows and nowhere else.
        for d in [8usize, 11, 6] {
            check_pathological_rows_bit_identical(d);
        }
    }

    fn check_pathological_rows_bit_identical(d: usize) {
        assert!(d >= 6, "the row patterns below index up to lane 5");
        let big = d - 1;
        let rows: Vec<Vec<f32>> = vec![
            // Fully masked: sum of exponentials is 0, so normalization is 0/0.
            vec![f32::NEG_INFINITY; d],
            // Row max is +inf, so exp(inf - inf) is NaN.
            {
                let mut r = vec![1.0f32; d];
                r[3] = f32::INFINITY;
                r
            },
            // Both infinities in one row.
            {
                let mut r = vec![0.5f32; d];
                r[0] = f32::INFINITY;
                r[big] = f32::NEG_INFINITY;
                r
            },
            // Quiet NaN.
            {
                let mut r = vec![2.0f32; d];
                r[2] = f32::NAN;
                r
            },
            // Signalling NaN: the pattern class that broke the bf16 widen.
            {
                let mut r = vec![-1.0f32; d];
                r[5] = f32::from_bits(0x7f80_0001);
                r
            },
            // Negative-payload NaN.
            {
                let mut r = vec![3.0f32; d];
                r[1] = f32::from_bits(0xffc0_0003);
                r
            },
            // Denormals.
            (0..d).map(|i| f32::from_bits(1 + i as u32)).collect(),
            // Ordinary finite logits, as a control.
            (0..d).map(|i| i as f32 * 0.25 - 1.0).collect(),
        ];
        let n = rows.len();
        let src: Vec<f32> = rows.concat();

        // The form this kernel replaced: copy the whole tensor, then reduce in
        // place over the copy.
        let mut expect = src.clone();
        softmax_rows_in_place(&mut expect, n, d);

        let mut got = vec![0.0f32; n * d];
        softmax_rows(&src, &mut got, n, d);

        for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "d={d} row {} lane {}: out-of-place {:#010x} != copy+in-place {:#010x}",
                i / d,
                i % d,
                g.to_bits(),
                e.to_bits()
            );
        }

        // Guard against the whole comparison going vacuous: these rows must
        // actually produce the non-finite results the test claims to pin down.
        // Without this, a change that quietly made every row finite would leave
        // the bit comparison trivially true.
        assert!(
            expect[..d].iter().all(|v| v.is_nan()),
            "d={d}: the fully masked row stopped producing NaN; this test no longer covers it"
        );
        assert!(
            expect.iter().filter(|v| v.is_nan()).count() >= 5 * d,
            "d={d}: the pathological rows stopped producing NaN; the bit comparison is now vacuous"
        );
    }

    #[test]
    fn vectorized_rows_match_the_scalar_reference() {
        for (n, d) in [(1usize, 1usize), (3, 7), (64, 256), (200, 33)] {
            let src = synthetic(n, d, (n + d) as u32);
            let expect = reference_rows(&src, n, d);
            let mut got = src.clone();
            softmax_rows_in_place(&mut got, n, d);
            for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-6,
                    "n={n} d={d} idx={i}: {g} vs {e} (vectorized exp differs from libm)"
                );
            }
            // Every row must still sum to one.
            for r in 0..n {
                let sum: f32 = got[r * d..(r + 1) * d].iter().sum();
                assert!(
                    (sum - 1.0).abs() <= 1e-5,
                    "n={n} d={d} row {r} sums to {sum}"
                );
            }
        }
    }

    /// Degenerate extents must be no-ops rather than panics or divisions by
    /// zero - a zero-length axis reaches the kernel through dynamic shapes.
    #[test]
    fn empty_extents_are_a_no_op() {
        let mut empty: Vec<f32> = Vec::new();
        softmax_rows_in_place(&mut empty, 0, 8);
        softmax_rows_in_place(&mut empty, 8, 0);
        softmax_rows(&[], &mut [], 0, 8);
        softmax_rows(&[], &mut [], 8, 0);
        assert!(empty.is_empty());
    }

    /// A row of very large logits must not overflow (the max is subtracted
    /// first), and `-inf` entries - how masked attention positions arrive -
    /// must produce exact zeros rather than NaN, as long as the row is not
    /// entirely masked.
    #[test]
    fn large_and_masked_logits_stay_finite() {
        let n = 96; // over the row floor, so this runs through the fan-out
        let d = 256;
        let mut src = vec![0.0f32; n * d];
        for r in 0..n {
            for c in 0..d {
                src[r * d + c] = if c % 3 == 0 { f32::NEG_INFINITY } else { 1e30 };
            }
        }
        let mut got = src.clone();
        softmax_rows_in_place(&mut got, n, d);
        for r in 0..n {
            let row = &got[r * d..(r + 1) * d];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() <= 1e-5, "row {r} sums to {sum}");
            for (c, &v) in row.iter().enumerate() {
                assert!(v.is_finite(), "row {r} col {c} is {v}");
                if c % 3 == 0 {
                    assert_eq!(v, 0.0, "masked position must be exactly zero");
                }
            }
        }
    }

    /// A *fully* masked row has no finite maximum, so `x - max` is `-inf -
    /// -inf = NaN` for every element. That is what the scalar loop produced
    /// before this change and what ORT's own MLAS-backed Softmax produces, so
    /// the behaviour is pinned rather than "fixed": silently substituting a
    /// uniform distribution here would diverge from the reference runtime.
    #[test]
    fn a_fully_masked_row_reproduces_the_reference_runtimes_nan() {
        for n in [1usize, 96] {
            let d = 256;
            let src = vec![f32::NEG_INFINITY; n * d];
            let mut got = src.clone();
            softmax_rows_in_place(&mut got, n, d);
            assert!(
                got.iter().all(|v| v.is_nan()),
                "n={n}: fully masked row must stay NaN, not become uniform"
            );
            // The scalar oracle agrees, so this is not an MLAS artefact.
            assert!(reference_rows(&src, n, d).iter().all(|v| v.is_nan()));
        }
    }

    /// The direct-write path and the owned-scratch path must agree exactly.
    /// The kernel picks between them on an aliasing check the caller cannot
    /// see, so both have to be exercised: a strided input forces the widen to
    /// allocate, which is the owned arm.
    #[test]
    fn direct_write_and_owned_scratch_agree() {
        let rows = 80;
        let cols = 256;
        let src = synthetic(rows, cols, 7);

        let x = Owned::f32(&[rows, cols], &src);
        let mut direct = Owned::zeros_f32(&[rows, cols]);
        SoftmaxKernel {
            axis: -1,
            coerce_2d: false,
        }
        .execute(&[x.view()], &mut [direct.view_mut()])
        .unwrap();

        // A stride-2 view over a doubled buffer is not contiguous, so
        // `to_dense_f32_widen` must copy and the output check still holds -
        // this is the arm that also covers non-f32 outputs.
        let mut doubled = vec![0.0f32; rows * cols * 2];
        for (i, v) in src.iter().enumerate() {
            doubled[i * 2] = *v;
        }
        let strided = Owned::f32(&[rows, cols * 2], &doubled)
            .with_view(&[rows, cols], &[(cols * 2) as i64, 2]);
        let mut owned = Owned::zeros_f32(&[rows, cols]);
        SoftmaxKernel {
            axis: -1,
            coerce_2d: false,
        }
        .execute(&[strided.view()], &mut [owned.view_mut()])
        .unwrap();

        assert_eq!(direct.to_f32(), owned.to_f32());
        let expect = reference_rows(&src, rows, cols);
        for (g, e) in direct.to_f32().iter().zip(&expect) {
            assert!((g - e).abs() <= 1e-6, "{g} vs {e}");
        }
    }

    /// An interior-axis reduction (`inner > 1`) must not be routed to the
    /// row-major fast path, which would silently softmax the wrong elements.
    #[test]
    fn interior_axis_still_reduces_along_that_axis() {
        // [2, 3, 4]: axis 1 has inner = 4.
        let data: Vec<f32> = (0..24).map(|i| (i % 5) as f32 - 2.0).collect();
        let x = Owned::f32(&[2, 3, 4], &data);
        let mut out = Owned::zeros_f32(&[2, 3, 4]);
        SoftmaxKernel {
            axis: 1,
            coerce_2d: false,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        // Each (outer, inner) column of 3 must sum to one.
        for o in 0..2 {
            for i in 0..4 {
                let got = out.to_f32();
                let sum: f32 = (0..3).map(|a| got[o * 12 + a * 4 + i]).sum();
                assert!((sum - 1.0).abs() <= 1e-6, "o={o} i={i} sums to {sum}");
            }
        }
    }

    /// The output binding may legitimately alias the input buffer (in-place
    /// execution via device I/O bindings). Softmax rewrites each element after
    /// reading it, so the direct-write path must decline and fall back to the
    /// owned scratch; otherwise the row max/sum would be computed over
    /// already-normalized values.
    #[test]
    fn an_output_aliasing_its_input_matches_the_disjoint_result() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::{DataType, DeviceId, compute_contiguous_strides};

        let (rows, cols) = (96usize, 128usize);
        let src = synthetic(rows, cols, 21);

        let disjoint = {
            let x = Owned::f32(&[rows, cols], &src);
            let mut out = Owned::zeros_f32(&[rows, cols]);
            SoftmaxKernel {
                axis: -1,
                coerce_2d: false,
            }
            .execute(&[x.view()], &mut [out.view_mut()])
            .unwrap();
            out.to_f32()
        };

        let mut shared = src.clone();
        let in_ptr = shared.as_ptr() as *const std::ffi::c_void;
        let out_ptr = shared.as_mut_ptr() as *mut std::ffi::c_void;
        let shape = [rows, cols];
        let strides = compute_contiguous_strides(&shape);
        let f32c = DataType::Float32;
        let cpu = DeviceId::cpu();
        let x = TensorView::new(DevicePtr(in_ptr), f32c, &shape, &strides, cpu);
        let out = TensorMut::new(DevicePtrMut(out_ptr), f32c, &shape, &strides, cpu);
        SoftmaxKernel {
            axis: -1,
            coerce_2d: false,
        }
        .execute(&[x], &mut [out])
        .unwrap();

        assert_eq!(shared, disjoint);
    }

    /// A zero-element tensor must not reach `from_raw_parts_mut`.
    #[test]
    fn a_zero_element_tensor_is_a_no_op() {
        let x = Owned::zeros_f32(&[0, 8]);
        let mut out = Owned::zeros_f32(&[0, 8]);
        SoftmaxKernel {
            axis: -1,
            coerce_2d: false,
        }
        .execute(&[x.view()], &mut [out.view_mut()])
        .unwrap();
        assert!(out.to_f32().is_empty());
    }
}
