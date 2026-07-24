//! Native BF16×BF16→FP32 GEMM for x86-64 hosts with `avx512_bf16`.
//!
//! ONNX Runtime's CPU EP has no bf16 compute; our MatMul supports bf16 by
//! widening every element to f32 and running the f32 SGEMM (correct, but it
//! doubles the operand bandwidth and does no native bf16 work). On a
//! Sapphire-Rapids-class core with `avx512_bf16`, `_mm512_dpbf16_ps` multiplies
//! 32 bf16 pairs and horizontally accumulates them into 16 f32 lanes in a single
//! instruction. This module routes bf16×bf16 MatMul through that instruction,
//! keeping the operands in bf16 (half the bytes) and accumulating in **f32**.
//!
//! ## Numerics
//!
//! `_mm512_dpbf16_ps` rounds each bf16×bf16 product to f32, then accumulates in
//! f32. It is therefore NOT bit-identical to the widen-then-`fma` reference
//! (which widens each bf16 operand to an exact f32 before multiplying), but both
//! carry an f32 accumulator, so the error stays within bf16 tolerance — never
//! worse than the upcast path for realistic magnitudes (verified in the tests).
//! The accumulator is f32, never bf16: a bf16/f16 accumulator loses 1–25 % at
//! realistic reduction lengths.
//!
//! ## Layout & strategy
//!
//! `a` is `m*k` row-major bf16, `b` is `k*n` row-major bf16, `c` is `m*n`
//! row-major f32 (overwrite). `_mm512_dpbf16_ps` reduces along the contiguous
//! lane axis, so B is transposed once into `k`-contiguous `n*k` panels
//! (`b_t[j*k + p] = b[p*n + j]`); A rows are already `k`-contiguous. A `MR×NR`
//! register tile of dot-product accumulators reuses each loaded A/B chunk across
//! the tile; K tails (K not a multiple of 32) use a masked epi16 load. Rayon
//! parallelizes over disjoint row blocks of C.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use rayon::prelude::*;

/// Whether the running CPU can execute the native bf16 GEMM. Requires
/// `avx512_bf16` (the `_mm512_dpbf16_ps` dot product), `avx512bw` (masked
/// 16-bit loads for the K tail) and `avx512f` (f32 accumulate + reduce).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn native_available() -> bool {
    std::arch::is_x86_feature_detected!("avx512bf16")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512f")
}

#[cfg(not(target_arch = "x86_64"))]
// Portable tests call this stub, while all non-test library callers are x86_64-gated.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[inline]
pub(crate) fn native_available() -> bool {
    false
}

/// Register-tile rows (independent C rows accumulated together).
#[cfg(target_arch = "x86_64")]
const MR: usize = 4;
/// Register-tile columns (independent C columns accumulated together).
#[cfg(target_arch = "x86_64")]
const NR: usize = 4;
/// Row-block granularity for Rayon; each block owns a disjoint C row range.
#[cfg(target_arch = "x86_64")]
const MC: usize = 64;

/// Compute `c[m,n] = a[m,k] @ b[k,n]` natively in bf16 (overwrite), accumulating
/// in f32 via `_mm512_dpbf16_ps`. `a`/`b` are raw bf16 bit patterns.
///
/// The caller MUST have confirmed [`native_available`]; `a.len() == m*k`,
/// `b.len() == k*n`, `c.len() == m*n`.
#[cfg(target_arch = "x86_64")]
pub(crate) fn gemm(a: &[u16], b: &[u16], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    // Transpose B into k-contiguous panels so `_mm512_dpbf16_ps` reduces along
    // the unit-stride axis: b_t[j*k + p] = b[p*n + j].
    let b_t = transpose_b(b, k, n);

    // Parallelize over disjoint row blocks of C. Each block reads shared,
    // immutable A/B_t and writes its own C rows, so there is no aliasing.
    c.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(blk, c_blk)| {
            let i0 = blk * MC;
            let rows = c_blk.len() / n;
            let a_blk = &a[i0 * k..i0 * k + rows * k];
            // SAFETY: `native_available()` (the caller's precondition) confirmed
            // avx512bf16 + avx512bw + avx512f, which every intrinsic used below
            // requires. The slice lengths satisfy the microkernel's index bounds.
            unsafe { gemm_block(a_blk, &b_t, c_blk, rows, k, n) };
        });
}

/// Transpose `b[k,n]` (row-major) into `b_t[n,k]` (row-major), preserving the
/// bf16 bit patterns. Parallelized over destination rows (columns of B).
#[cfg(target_arch = "x86_64")]
fn transpose_b(b: &[u16], k: usize, n: usize) -> Vec<u16> {
    let mut b_t = vec![0u16; n * k];
    b_t.par_chunks_mut(k).enumerate().for_each(|(j, dst)| {
        for (p, d) in dst.iter_mut().enumerate() {
            *d = b[p * n + j];
        }
    });
    b_t
}

/// Compute one row block: `c[rows,n] = a[rows,k] @ b_t[n,k]^T` (overwrite),
/// register-tiling `MR×NR` dot products.
///
/// # Safety
/// The CPU must support avx512bf16 + avx512bw + avx512f; slice lengths must
/// match `rows`, `k`, `n` as documented.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bf16,avx512bw,avx512f")]
unsafe fn gemm_block(a: &[u16], b_t: &[u16], c: &mut [f32], rows: usize, k: usize, n: usize) {
    let mut i = 0;
    while i < rows {
        let mr = MR.min(rows - i);
        let mut j = 0;
        while j < n {
            let nr = NR.min(n - j);
            // SAFETY: propagated from the caller's feature/bounds contract.
            unsafe { micro_kernel(a, b_t, c, k, n, i, j, mr, nr) };
            j += NR;
        }
        i += MR;
    }
}

/// Accumulate the `mr×nr` (≤ `MR×NR`) tile of C at rows `[i, i+mr)` × cols
/// `[j, j+nr)` as f32 dot products of A rows and transposed-B rows via
/// `_mm512_dpbf16_ps`, then store the reduced scalars (overwrite).
///
/// # Safety
/// Requires avx512bf16 + avx512bw + avx512f; the tile indices must stay in
/// bounds of `a` (`rows×k`), `b_t` (`n×k`) and `c` (`rows×n`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bf16,avx512bw,avx512f")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
unsafe fn micro_kernel(
    a: &[u16],
    b_t: &[u16],
    c: &mut [f32],
    k: usize,
    n: usize,
    i: usize,
    j: usize,
    mr: usize,
    nr: usize,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // SAFETY: the caller guarantees avx512bf16 + avx512bw + avx512f are present
    // (every intrinsic below requires them) and that the tile indices stay in
    // bounds of `a` (`rows×k`), `b_t` (`n×k`) and `c` (`rows×n`).
    unsafe {
        // MR×NR f32 accumulators (16 zmm at the MR=NR=4 tile).
        let mut acc = [[_mm512_setzero_ps(); NR]; MR];
        let mut a_ch = [_mm512_setzero_si512(); MR];
        let mut b_ch = [_mm512_setzero_si512(); NR];

        let a_ptr = a.as_ptr();
        let b_ptr = b_t.as_ptr();

        let mut p = 0;
        while p < k {
            let chunk = 32.min(k - p);
            if chunk == 32 {
                for ii in 0..mr {
                    a_ch[ii] = _mm512_loadu_si512(a_ptr.add((i + ii) * k + p) as *const __m512i);
                }
                for jj in 0..nr {
                    b_ch[jj] = _mm512_loadu_si512(b_ptr.add((j + jj) * k + p) as *const __m512i);
                }
            } else {
                // Masked K tail: load only the `chunk` valid bf16 lanes, zeroing
                // the rest so their bf16 products add nothing to the accumulator.
                let mask: __mmask32 = (1u32 << chunk) - 1;
                for ii in 0..mr {
                    a_ch[ii] =
                        _mm512_maskz_loadu_epi16(mask, a_ptr.add((i + ii) * k + p) as *const i16);
                }
                for jj in 0..nr {
                    b_ch[jj] =
                        _mm512_maskz_loadu_epi16(mask, b_ptr.add((j + jj) * k + p) as *const i16);
                }
            }
            for ii in 0..mr {
                let av: __m512bh = core::mem::transmute::<__m512i, __m512bh>(a_ch[ii]);
                for jj in 0..nr {
                    let bv: __m512bh = core::mem::transmute::<__m512i, __m512bh>(b_ch[jj]);
                    acc[ii][jj] = _mm512_dpbf16_ps(acc[ii][jj], av, bv);
                }
            }
            p += 32;
        }

        for ii in 0..mr {
            for jj in 0..nr {
                *c.get_unchecked_mut((i + ii) * n + (j + jj)) = _mm512_reduce_add_ps(acc[ii][jj]);
            }
        }
    }
}
