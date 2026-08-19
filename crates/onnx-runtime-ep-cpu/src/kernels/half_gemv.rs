//! Decode-shaped (`M == 1`) GEMV against an f16 weight held in 16-bit storage.
//!
//! # Why this exists
//!
//! `matmul::try_matmul_half` packs both operands into cache-sized panels and
//! runs a blocked `MR x NR` micro-kernel.  That layout is the right one for
//! prefill, where every packed panel of `B` is reused across `MR` rows of `A`.
//!
//! At `M == 1` there is no reuse: each weight element is touched exactly once,
//! so there is no arithmetic to amortise the packing against and every cycle
//! spent packing is wasted.  Apple hosts already dodge this — `matmul` checks a
//! NEON f16 GEMV *before* `try_matmul_half`, and the comment there records that
//! the blocked GEMM is "~4x slower than the bandwidth-optimal NEON GEMV at M=1
//! decode shapes".  x86 had no such path and fell into the blocked kernel.
//!
//! Touching each weight once is what makes memory speed the *ceiling*; it does
//! not on its own put the kernel anywhere near it.  This one originally ran at
//! 12-47 GB/s against a measured 75.8 GB/s host bandwidth, because accumulating
//! straight into `acc` cost two more memory operations per FMA than the weight
//! load itself.  Reaching 79-86% of the roofline took the register tiling
//! described on [`TILE`].  See
//! `docs/benchmarks/2026-08-19-f16-decode-gemv-register-tile.md`.
//!
//! # Why it reads `B` in `[K, N]` order rather than transposing
//!
//! The obvious GEMV walks `B^T` so each output column is one contiguous dot
//! product.  That needs a transposed copy of the weight — a one-time
//! `2 * K * N` byte allocation, which is 272 MB for a 896x151936 lm_head.
//! `try_matmul_half` reads `B` straight from the mapped weight with no such
//! copy, so transposing would trade a real, permanent memory cost for speed.
//!
//! Instead this kernel keeps `B` exactly as stored and blocks over *columns*:
//! a worker owns a stripe `[j0, j0 + W)` of the output, walks `p = 0..k`, and
//! at each step reads the `W` contiguous weights at `b[p * n + j0]` and does
//! `acc[0..W] += a[p] * widen(...)`.  Every cache line fetched is fully
//! consumed, the weight is still read exactly once end to end, and nothing is
//! allocated or cached.  It also works for a *non-constant* `B`, which a
//! prepacked transpose cannot.
//!
//! # Numerics
//!
//! `f16 -> f32` is exact for every `f16` value, and `_mm256_cvtph_ps` is
//! bit-identical to [`half::f16::to_f32`] (see `dtype::f16c`).  Accumulation is
//! `f32`, matching the blocked kernel's accumulator width.
//!
//! Because each output element accumulates over `p` in strictly increasing
//! order, the summation order is *exactly* that of a naive triple loop —
//! independent of the stripe width, the lane count, and the thread count.  The
//! SIMD path, the scalar fallback and a naive reference therefore all agree bit
//! for bit, which the tests assert directly rather than within a tolerance.

/// Output columns whose accumulators are held in `ymm` registers across the
/// whole `k` loop.
///
/// The point of the tile is what the inner loop *stops* doing. Accumulating
/// straight into `acc` costs three memory operations per 8-lane FMA — load the
/// weight, load the accumulator, store it back — so the loop is limited by the
/// load/store ports rather than by the FMA units, and every FMA also carries a
/// store-to-load forwarding round trip from the previous `p`. Keeping the
/// accumulators in registers leaves exactly one memory operation per FMA (the
/// weight), which is the minimum this problem can have: the weight is read
/// once and nothing else is touched.
///
/// 64 columns is 8 `ymm` accumulators, plus the broadcast activation and the
/// widened weight, so 10 of 16 architectural registers. It is also 128 bytes,
/// exactly two cache lines, and `STRIPE` is a multiple of it, so a tile never
/// straddles a line it does not fully consume.
///
/// Tiling does not re-read anything: a stripe's weight is swept once per tile
/// over `STRIPE / TILE` disjoint column ranges, which is the same `k * STRIPE`
/// elements in total and the same `n`-element stride between consecutive `p`.
const TILE: usize = 64;

/// Output columns owned by one worker at a time.
///
/// Must be at least 32 (one 64-byte line holds 32 `f16`) so no fetched line is
/// partially wasted; 512 keeps the `f32` accumulators at 2 KiB, comfortably
/// inside L1, while amortising the row-pointer advance over 8 lines of weight.
const STRIPE: usize = 512;

/// Minimum `k * n` before the stripe loop is handed to rayon.  Below this the
/// fork/join costs more than the work; measured on a 32-vCPU AVX2 host.
const PARALLEL_MIN_WORK: usize = 1 << 16;

/// Is the hardware-accelerated f16 GEMV available on this host?
pub(crate) fn simd_available() -> bool {
    std::arch::is_x86_feature_detected!("f16c")
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
}

/// `out[j] = sum_p a[p] * b[p * n + j]`, with `b` the `[K, N]` row-major f16
/// weight exactly as stored — no transpose, no copy, no cache.
///
/// # Panics
/// When `b` is shorter than `k * n`. This is a real check, not a
/// `debug_assert`: [`stripe_simd`] does unchecked pointer arithmetic derived
/// from `k`, `n` and `j0`, so a short `b` would be a memory-safety bug rather
/// than a wrong answer. `k * n` is computed with `checked_mul` so an
/// overflowing geometry fails closed instead of wrapping to a small bound.
pub(crate) fn gemv_f16_kn(a: &[f32], b: &[u16], out: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(a.len(), k);
    debug_assert_eq!(out.len(), n);
    let weights = k.checked_mul(n).expect("f16 GEMV geometry overflows usize");
    assert!(
        b.len() >= weights && a.len() >= k,
        "f16 GEMV operands are too small for k={k} n={n}: a={} b={}",
        a.len(),
        b.len()
    );
    if n == 0 {
        return;
    }
    // An empty contraction sums nothing: every output is zero. Short-circuited
    // so the stripe loop never runs with `k == 0` and reads no weight at all.
    if k == 0 {
        out.fill(0.0);
        return;
    }

    let simd = simd_available();
    let stripe = |j0: usize, acc: &mut [f32]| {
        if simd {
            // SAFETY: `simd_available()` confirmed f16c + avx2 + fma. `j0` and
            // `acc.len()` come from a `chunks_mut(STRIPE)` over an `n`-element
            // slice, so `j0 + acc.len() <= n`; `a.len() == k` and
            // `b.len() >= k * n` was asserted above, so every read below is
            // in bounds.
            unsafe {
                stripe_simd(a, b, acc, j0, k, n);
            }
            return;
        }
        stripe_scalar(a, b, acc, j0, k, n);
    };

    let work = weights;
    if work < PARALLEL_MIN_WORK || rayon::current_num_threads() <= 1 {
        for (tile, acc) in out.chunks_mut(STRIPE).enumerate() {
            stripe(tile * STRIPE, acc);
        }
        return;
    }

    use rayon::prelude::*;
    out.par_chunks_mut(STRIPE)
        .enumerate()
        .for_each(|(tile, acc)| {
            stripe(tile * STRIPE, acc);
        });
}

/// Reference stripe: `acc[j] = sum_p a[p] * b[p * n + j0 + j]`.
///
/// Accumulates over `p` in increasing order, one element at a time — the same
/// order [`stripe_simd`] uses within each lane.
fn stripe_scalar(a: &[f32], b: &[u16], acc: &mut [f32], j0: usize, k: usize, n: usize) {
    acc.fill(0.0);
    let w = acc.len();
    for (p, &av) in a.iter().enumerate().take(k) {
        let row = &b[p * n + j0..p * n + j0 + w];
        for (slot, &bits) in acc.iter_mut().zip(row.iter()) {
            *slot = half::f16::from_bits(bits).to_f32().mul_add(av, *slot);
        }
    }
}

/// F16C + FMA stripe kernel.
///
/// Column-tiled: each [`TILE`] of output columns keeps its accumulators in
/// `ymm` for the whole `k` loop and is stored exactly once, so the inner loop
/// issues one load per FMA instead of a load/FMA/store triple. Within any one
/// output element the `p` order is still strictly increasing, so this is
/// bit-identical to the untiled form and to [`stripe_scalar`] — the tiling
/// changes which registers hold a partial sum, never the order it is built in.
///
/// # Safety
/// The running CPU must support `f16c`, `avx2` and `fma` (see
/// [`simd_available`]); `a.len() >= k`; and `(k - 1) * n + j0 + acc.len() <=
/// b.len()`, i.e. every `b[p * n + j0 .. p * n + j0 + acc.len()]` for `p < k`
/// must be in bounds.
#[target_feature(enable = "f16c,avx2,fma")]
unsafe fn stripe_simd(a: &[f32], b: &[u16], acc: &mut [f32], j0: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let w = acc.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = acc.as_mut_ptr();
    unsafe {
        let mut j = 0;
        // Register-resident tiles: `TILE / 8` accumulators live in `ymm`
        // across the whole contraction and reach memory once, at the end.
        while j + TILE <= w {
            let mut sums = [_mm256_setzero_ps(); TILE / 8];
            let base = bp.add(j0 + j);
            for p in 0..k {
                let av = _mm256_set1_ps(*ap.add(p));
                let row = base.add(p * n);
                for (lane, sum) in sums.iter_mut().enumerate() {
                    let h = _mm_loadu_si128(row.add(lane * 8) as *const __m128i);
                    *sum = _mm256_fmadd_ps(_mm256_cvtph_ps(h), av, *sum);
                }
            }
            for (lane, sum) in sums.iter().enumerate() {
                _mm256_storeu_ps(cp.add(j + lane * 8), *sum);
            }
            j += TILE;
        }
        // Sub-tile remainder: one register accumulator per 8 columns, same
        // shape as above, so it is bit-identical to the tiled path too.
        while j + 8 <= w {
            let mut sum = _mm256_setzero_ps();
            let base = bp.add(j0 + j);
            for p in 0..k {
                let av = _mm256_set1_ps(*ap.add(p));
                let h = _mm_loadu_si128(base.add(p * n) as *const __m128i);
                sum = _mm256_fmadd_ps(_mm256_cvtph_ps(h), av, sum);
            }
            _mm256_storeu_ps(cp.add(j), sum);
            j += 8;
        }
        // Scalar tail, using the same fused multiply-add so the last lanes
        // round identically to the vectorised ones.
        while j < w {
            let mut sum = 0.0f32;
            let base = bp.add(j0 + j);
            for p in 0..k {
                let bv = half::f16::from_bits(*base.add(p * n)).to_f32();
                sum = bv.mul_add(*ap.add(p), sum);
            }
            *cp.add(j) = sum;
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16v(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|v| half::f16::from_f32(*v).to_bits())
            .collect()
    }

    /// Straight-line reference: no striping, no lanes, no unrolling.
    ///
    /// Accumulates over `p` in increasing order, which is exactly what the
    /// kernel does per output element — so this is a *bitwise* oracle, not an
    /// approximate one.
    fn naive(a: &[f32], b: &[u16], k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for (p, &av) in a.iter().enumerate().take(k) {
            for (j, slot) in out.iter_mut().enumerate() {
                *slot = half::f16::from_bits(b[p * n + j])
                    .to_f32()
                    .mul_add(av, *slot);
            }
        }
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Pseudo-random but reproducible; deliberately *not* f16-exact, so a
    /// kernel that reassociated the sum would show up as a bit mismatch.
    fn sample(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn gemv_is_bit_identical_to_a_naive_reference() {
        // Not a tolerance check: the kernel accumulates in the same order as
        // the reference, so anything other than equality is a real defect.
        let k = 40;
        let n = 37;
        let a = sample(k, 7);
        let b = f16v(&sample(k * n, 11));
        let mut out = vec![0.0f32; n];
        gemv_f16_kn(&a, &b, &mut out, k, n);
        assert_eq!(bits(&out), bits(&naive(&a, &b, k, n)));
    }

    #[test]
    fn simd_and_scalar_stripes_agree_bit_for_bit() {
        let k = 33;
        let n = 91;
        let a = sample(k, 3);
        let b = f16v(&sample(k * n, 5));
        for j0 in [0usize, 8, 17, 83] {
            let w = (n - j0).min(19);
            let mut scalar = vec![f32::NAN; w];
            stripe_scalar(&a, &b, &mut scalar, j0, k, n);
            if simd_available() {
                let mut simd = vec![f32::NAN; w];
                // SAFETY: guarded by `simd_available()`; `j0 + w <= n` and `b`
                // holds `k * n` elements.
                unsafe { stripe_simd(&a, &b, &mut simd, j0, k, n) };
                assert_eq!(bits(&simd), bits(&scalar), "stripe at j0={j0} w={w}");
            }
        }
    }

    #[test]
    fn widths_below_across_and_beyond_the_stripe_are_exact() {
        // Cover an empty contraction, widths under and over the 8-lane SIMD
        // step, and an `n` past STRIPE so more than one stripe runs.
        for (k, n) in [
            (0usize, 5usize),
            (1, 1),
            (3, 7),
            (5, 8),
            (5, 9),
            (2, 512),
            (2, 513),
            (3, 1100),
        ] {
            let a = sample(k, 13);
            let b = f16v(&sample(k * n, 17));
            let mut out = vec![f32::NAN; n];
            gemv_f16_kn(&a, &b, &mut out, k, n);
            assert_eq!(bits(&out), bits(&naive(&a, &b, k, n)), "k={k} n={n}");
        }
    }

    #[test]
    fn zero_width_output_writes_nothing() {
        let mut out: Vec<f32> = Vec::new();
        gemv_f16_kn(&[1.0, 2.0], &[], &mut out, 2, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn stripe_widths_around_the_tile_boundary_are_exact() {
        // The register-tiled path runs in `TILE`-column blocks, then 8-lane
        // blocks, then scalars, so every width class and every boundary
        // between them has to land on the same bits. Sweeping a contiguous
        // range rather than picked widths is what makes an off-by-one in the
        // tile loop's `j + TILE <= w` bound impossible to miss. The top width
        // is two whole tiles plus one 8-lane block plus one scalar, so the
        // sweep ends having exercised all three in a single call.
        if !simd_available() {
            return;
        }
        let k = 11;
        let n = 300;
        let a = sample(k, 41);
        let b = f16v(&sample(k * n, 43));
        let mut checked = 0usize;
        for w in 1..=(2 * TILE + 9) {
            for j0 in [0usize, 8, 64, 67] {
                assert!(j0 + w <= n, "the sweep must not skip j0={j0} w={w}");
                let mut scalar = vec![f32::NAN; w];
                stripe_scalar(&a, &b, &mut scalar, j0, k, n);
                let mut simd = vec![f32::NAN; w];
                // SAFETY: `simd_available()` checked above; `j0 + w <= n` is
                // asserted, and `b` holds `k * n` elements.
                unsafe { stripe_simd(&a, &b, &mut simd, j0, k, n) };
                assert_eq!(bits(&simd), bits(&scalar), "stripe j0={j0} w={w}");
                checked += 1;
            }
        }
        assert_eq!(checked, 4 * (2 * TILE + 9), "the sweep silently shrank");
    }

    #[test]
    fn a_tile_never_straddles_a_stripe() {
        // `TILE`'s doc comment rests on this: a tile is two whole cache lines
        // and a stripe is a whole number of tiles, so no tile boundary can
        // split a line the kernel has already paid to fetch.
        assert_eq!(
            STRIPE % TILE,
            0,
            "STRIPE={STRIPE} must be a whole number of TILE={TILE} blocks"
        );
        assert_eq!(TILE % 8, 0, "a tile must be a whole number of 8-lane steps");
        assert_eq!(
            (TILE * size_of::<u16>()) % 64,
            0,
            "a tile must be a whole number of 64-byte cache lines"
        );
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        // Above PARALLEL_MIN_WORK the stripes are split across rayon workers.
        // Each output element is still owned by exactly one worker and summed
        // in `p` order, so the pool size must not change a single bit.
        let k = 300;
        let n = 1100;
        assert!(
            k * n >= PARALLEL_MIN_WORK,
            "test must exercise the parallel split"
        );
        let a = sample(k, 23);
        let b = f16v(&sample(k * n, 29));
        let reference = naive(&a, &b, k, n);
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool");
            let mut out = vec![f32::NAN; n];
            pool.install(|| gemv_f16_kn(&a, &b, &mut out, k, n));
            assert_eq!(bits(&out), bits(&reference), "{threads} threads changed it");
        }
    }

    #[test]
    fn large_magnitudes_accumulate_in_f32_not_f16() {
        // 2048 + 1 is not representable in f16; an f16 accumulator would drop
        // every term after the first. Widening to f32 first keeps them.
        let k = 64;
        let n = 1;
        let a = vec![1.0f32; k];
        let mut values = vec![1.0f32; k];
        values[0] = 2048.0;
        let b = f16v(&values);
        let mut out = vec![0.0f32; n];
        gemv_f16_kn(&a, &b, &mut out, k, n);
        assert_eq!(out[0], 2048.0 + (k as f32 - 1.0));
    }

    #[test]
    #[should_panic(expected = "operands are too small")]
    fn a_short_weight_panics_rather_than_reading_out_of_bounds() {
        // `stripe_simd` derives raw pointers from `k`, `n` and `j0`, so a
        // short `b` must fail closed in release too -- not just under
        // `debug_assert`.
        let mut out = vec![0.0f32; 4];
        gemv_f16_kn(&[1.0, 2.0], &[0u16; 5], &mut out, 2, 4);
    }

    #[test]
    fn the_weight_is_read_in_row_major_k_by_n_order() {
        // Pins the layout contract: `b` is [K, N] as stored, NOT a transpose.
        // A square symmetric case would pass either way, so use a non-square
        // asymmetric one.
        //
        // B = [[1, 2], [3, 4], [5, 6]], a = [1, 10, 100]
        // out = [1 + 30 + 500, 2 + 40 + 600] = [531, 642]
        let k = 3;
        let n = 2;
        let b = f16v(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let a = [1.0f32, 10.0, 100.0];
        let mut out = vec![0.0f32; n];
        gemv_f16_kn(&a, &b, &mut out, k, n);
        assert_eq!(out, vec![531.0, 642.0]);
    }
}
