//! Decode-shaped (`M == 1`) GEMV against an f16 or bf16 weight held in 16-bit
//! storage.
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
//! bit-identical to [`half::f16::to_f32`] (see `dtype::f16c`).  `bf16 -> f32`
//! is a left shift by 16: exact for every finite value, since `bf16` *is* the
//! top half of an `f32`.  Accumulation is `f32`, matching the blocked kernel's
//! accumulator width.
//!
//! The one place the shift and [`half::bf16::to_f32`] disagree is a signalling
//! NaN, whose payload the shift keeps and the `half` crate canonicalizes to a
//! quiet NaN — 126 of the 65536 `bf16` patterns, all NaN-encoding only.  The
//! blocked half GEMM this replaces widens by the same shift, so a decode that
//! switches to this kernel sees no change; `bf16_widening_matches_the_half_crate_over_the_whole_domain`
//! pins the count.
//!
//! Because each output element accumulates over `p` in strictly increasing
//! order, the summation order is *exactly* that of a naive triple loop —
//! independent of the stripe width, the lane count, and the thread count.  The
//! SIMD path, the scalar fallback and a naive reference therefore all agree bit
//! for bit, which the tests assert directly rather than within a tolerance.

use super::half_gemm::HalfFormat;

#[cfg(target_arch = "x86")]
use std::arch::x86::__m256;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__m256;

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

/// Is the hardware-accelerated GEMV available on this host for `format`?
///
/// `bf16` deliberately does *not* ask for `f16c`: it widens with an AVX2
/// shift, so requiring the `f16` conversion unit would disable the kernel on
/// hosts that can run it perfectly well.
pub(crate) fn simd_available(format: HalfFormat) -> bool {
    let base =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    match format {
        HalfFormat::F16 => base && std::arch::is_x86_feature_detected!("f16c"),
        HalfFormat::Bf16 => base,
    }
}

/// `out[j] = sum_p a[p] * b[p * n + j]`, with `b` the `[K, N]` row-major
/// `format` weight exactly as stored — no transpose, no copy, no cache.
///
/// # Panics
/// When `b` is shorter than `k * n`. This is a real check, not a
/// `debug_assert`: [`stripe_simd`] does unchecked pointer arithmetic derived
/// from `k`, `n` and `j0`, so a short `b` would be a memory-safety bug rather
/// than a wrong answer. `k * n` is computed with `checked_mul` so an
/// overflowing geometry fails closed instead of wrapping to a small bound.
pub(crate) fn gemv_half_kn(
    format: HalfFormat,
    a: &[f32],
    b: &[u16],
    out: &mut [f32],
    k: usize,
    n: usize,
) {
    debug_assert_eq!(a.len(), k);
    debug_assert_eq!(out.len(), n);
    let weights = k
        .checked_mul(n)
        .expect("half GEMV geometry overflows usize");
    assert!(
        b.len() >= weights && a.len() >= k,
        "half GEMV operands are too small for k={k} n={n}: a={} b={}",
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

    let simd = simd_available(format);
    let stripe = |j0: usize, acc: &mut [f32]| {
        // SAFETY (both arms): `simd_available(format)` confirmed exactly the
        // features the selected kernel declares. `j0` and `acc.len()` come
        // from a `chunks_mut(STRIPE)` over an `n`-element slice, so
        // `j0 + acc.len() <= n`; `a.len() == k` and `b.len() >= k * n` was
        // asserted above, so every read below is in bounds.
        match (simd, format) {
            (true, HalfFormat::F16) => unsafe { stripe_simd_f16(a, b, acc, j0, k, n) },
            (true, HalfFormat::Bf16) => unsafe { stripe_simd_bf16(a, b, acc, j0, k, n) },
            (false, _) => stripe_scalar(format, a, b, acc, j0, k, n),
        }
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
fn stripe_scalar(
    format: HalfFormat,
    a: &[f32],
    b: &[u16],
    acc: &mut [f32],
    j0: usize,
    k: usize,
    n: usize,
) {
    acc.fill(0.0);
    let w = acc.len();
    for (p, &av) in a.iter().enumerate().take(k) {
        let row = &b[p * n + j0..p * n + j0 + w];
        for (slot, &bits) in acc.iter_mut().zip(row.iter()) {
            *slot = widen_scalar(format, bits).mul_add(av, *slot);
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Counts [`gemv_f16_nk`] entries so a caller's test can assert *which*
    /// route served a call. Values alone cannot: the blocked fallback computes
    /// the same thing, only two orders of magnitude slower, which is the
    /// entire point.
    ///
    /// Thread-*local* on purpose. The count is taken at entry, on whichever
    /// thread called in, before any rayon dispatch — so a per-thread counter
    /// is both exact and immune to the rest of a threaded test harness running
    /// other GEMVs concurrently. A global would need every caller in the crate
    /// to agree on a lock, which is precisely the kind of coupling that rots.
    pub(crate) static NK_GEMV_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Output rows owned by one task in the `[N, K]` kernel.
///
/// Each row there is an *independent, contiguous* dot product, so unlike
/// [`STRIPE`] this can be as fine as we like without splitting a cache line.
/// 8 rows is 8 sequential streams -- enough to keep the prefetchers busy while
/// still giving a 3584-row projection 448 tasks to spread over the pool.
const ROW_STRIPE: usize = 8;

/// `out[j] = sum_p a[p] * b[j * k + p]`, with `b` the `[N, K]` row-major f16
/// weight exactly as stored — no transpose, no copy, no cache.
///
/// This is the layout `Gemm`'s `transB = 1` holds, which is what essentially
/// every `nn.Linear` export produces. It is a *better* layout for a GEMV than
/// the `[K, N]` one [`gemv_half_kn`] serves: each output element is one
/// contiguous `k`-element dot product, so the weight streams front to back,
/// every row is an independent task, and the partition can be arbitrarily fine.
/// The reason to have both is simply that the caller does not get to choose how
/// the weight was stored.
///
/// # Numerics
///
/// Unlike [`gemv_half_kn`], this is **not** the summation order of a naive
/// sequential loop, because the contraction runs along the vector lanes rather
/// than across them. The order is still fixed and fully specified, and in
/// particular does not depend on the thread count or on [`ROW_STRIPE`]:
///
/// 1. Lane `l` of accumulator `c` sums `p = 32*i + 8*c + l` for `i = 0, 1, ...`
///    in increasing `i`, over the largest multiple of 32 at or below `k`.
/// 2. Any remaining whole groups of 8 accumulate into `acc0` the same way.
/// 3. The four accumulators combine as `(acc0 + acc1) + (acc2 + acc3)`, lanewise.
/// 4. Those 8 lanes are summed low to high.
/// 5. The final `k % 8` elements are added on in increasing `p`.
///
/// Splitting one sum into 32 partial sums is, if anything, *more* accurate than
/// accumulating sequentially — but it is a different result, so the tests below
/// pin it against a reference that reproduces exactly these five steps rather
/// than against [`naive`](tests::naive).
///
/// # Panics
/// When `b` is shorter than `k * n`, for the same memory-safety reason
/// [`gemv_half_kn`] panics.
pub(crate) fn gemv_f16_nk(a: &[f32], b: &[u16], out: &mut [f32], k: usize, n: usize) {
    #[cfg(test)]
    NK_GEMV_CALLS.with(|calls| calls.set(calls.get() + 1));
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
    if k == 0 {
        out.fill(0.0);
        return;
    }

    let simd = simd_available(HalfFormat::F16);
    let rows = |j0: usize, acc: &mut [f32]| {
        for (offset, slot) in acc.iter_mut().enumerate() {
            let row = &b[(j0 + offset) * k..(j0 + offset) * k + k];
            *slot = if simd {
                // SAFETY: `simd_available(F16)` confirmed f16c + avx2 + fma, and
                // `row` was just sliced to exactly `k` elements.
                unsafe { dot_row_simd(a, row, k) }
            } else {
                dot_row_scalar(a, row, k)
            };
        }
    };

    if weights < PARALLEL_MIN_WORK || rayon::current_num_threads() <= 1 {
        for (tile, acc) in out.chunks_mut(ROW_STRIPE).enumerate() {
            rows(tile * ROW_STRIPE, acc);
        }
        return;
    }

    use rayon::prelude::*;
    out.par_chunks_mut(ROW_STRIPE)
        .enumerate()
        .for_each(|(tile, acc)| {
            rows(tile * ROW_STRIPE, acc);
        });
}

/// Reference for one `[N, K]` row, in the exact order [`gemv_f16_nk`] documents.
fn dot_row_scalar(a: &[f32], row: &[u16], k: usize) -> f32 {
    let widen = |p: usize| half::f16::from_bits(row[p]).to_f32();
    let mut acc = [[0.0f32; 8]; 4];
    // Groups of 8 consumed by the four-chain loop; whatever is left over runs
    // into chain 0, exactly as the 8-wide SIMD loop does.
    let quads = (k / 32) * 4;
    let groups = k / 8;
    for g in 0..groups {
        let chain = if g < quads { g % 4 } else { 0 };
        for (lane, cell) in acc[chain].iter_mut().enumerate() {
            let p = g * 8 + lane;
            *cell = widen(p).mul_add(a[p], *cell);
        }
    }
    // Mirrors the vector combine and the store-then-sum that follows it.
    let mut lanes = [0.0f32; 8];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        *slot = (acc[0][lane] + acc[1][lane]) + (acc[2][lane] + acc[3][lane]);
    }
    let mut total = 0.0f32;
    for lane in lanes {
        total += lane;
    }
    for (p, &av) in a.iter().enumerate().take(k).skip(groups * 8) {
        total = widen(p).mul_add(av, total);
    }
    total
}

/// F16C + FMA dot product of one contiguous `[N, K]` row against `a`.
///
/// # Safety
/// The running CPU must support `f16c`, `avx2` and `fma` (see
/// [`simd_available`]); `a.len() >= k` and `row.len() >= k`.
#[target_feature(enable = "f16c,avx2,fma")]
unsafe fn dot_row_simd(a: &[f32], row: &[u16], k: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let ap = a.as_ptr();
    let bp = row.as_ptr();
    unsafe {
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut p = 0;
        // Four independent chains so the 4-cycle FMA latency does not serialise
        // a kernel whose real limit is weight bandwidth.
        while p + 32 <= k {
            for (c, slot) in acc.iter_mut().enumerate() {
                let h = _mm_loadu_si128(bp.add(p + c * 8) as *const __m128i);
                let bw = _mm256_cvtph_ps(h);
                let av = _mm256_loadu_ps(ap.add(p + c * 8));
                *slot = _mm256_fmadd_ps(bw, av, *slot);
            }
            p += 32;
        }
        while p + 8 <= k {
            let h = _mm_loadu_si128(bp.add(p) as *const __m128i);
            let bw = _mm256_cvtph_ps(h);
            let av = _mm256_loadu_ps(ap.add(p));
            acc[0] = _mm256_fmadd_ps(bw, av, acc[0]);
            p += 8;
        }

        let combined = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), combined);
        let mut total = 0.0f32;
        for lane in lanes {
            total += lane;
        }
        while p < k {
            let bv = half::f16::from_bits(*bp.add(p)).to_f32();
            total = bv.mul_add(*ap.add(p), total);
            p += 1;
        }
        total
    }
}

/// F16C + FMA stripe kernel.
/// One element widened the way each format's vector kernel widens it.
///
/// `bf16` shifts rather than calling [`half::bf16::to_f32`] so the fallback
/// reproduces the vector path exactly, including on a signalling NaN.
#[inline(always)]
fn widen_scalar(format: HalfFormat, bits: u16) -> f32 {
    match format {
        HalfFormat::F16 => half::f16::from_bits(bits).to_f32(),
        HalfFormat::Bf16 => f32::from_bits((bits as u32) << 16),
    }
}

/// The vectorised stripe, written once and instantiated per format.
///
/// A macro rather than a generic because `#[target_feature]` is an attribute
/// on a concrete function: `f16` needs `f16c` for `_mm256_cvtph_ps` and `bf16`
/// must *not* ask for it, and a generic would either demand the union of both
/// feature sets or lose the inlining that makes the intrinsics worth using.
///
/// `$widen8` widens 8 contiguous 16-bit weights to `__m256`; the scalar tail
/// uses `widen_scalar($format, ..)` with the same fused multiply-add, so the
/// last lanes round identically to the vectorised ones.
///
/// Column-tiled: each [`TILE`] of output columns keeps its accumulators in
/// `ymm` for the whole `k` loop and is stored exactly once, so the inner loop
/// issues one load per FMA instead of a load/FMA/store triple. Within any one
/// output element the `p` order is still strictly increasing, so this stays
/// bit-identical to the untiled form and to [`stripe_scalar`] — the tiling
/// changes which registers hold a partial sum, never the order it is built in.
/// Every element of `acc` is written exactly once, so unlike [`stripe_scalar`]
/// there is no zeroing pass.
macro_rules! stripe_simd_fn {
    ($name:ident, $features:literal, $format:expr, $widen8:expr) => {
        /// # Safety
        /// The running CPU must support every feature named in the
        /// `target_feature` attribute (see [`simd_available`]);
        /// `a.len() >= k`; and `(k - 1) * n + j0 + acc.len() <= b.len()`, i.e.
        /// every `b[p * n + j0 .. p * n + j0 + acc.len()]` for `p < k` must be
        /// in bounds.
        #[target_feature(enable = $features)]
        unsafe fn $name(a: &[f32], b: &[u16], acc: &mut [f32], j0: usize, k: usize, n: usize) {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;

            let widen8: unsafe fn(*const u16) -> __m256 = $widen8;
            let w = acc.len();
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let cp = acc.as_mut_ptr();
            unsafe {
                let mut j = 0;
                // Register-resident tiles: `TILE / 8` accumulators live in
                // `ymm` across the whole contraction and reach memory once, at
                // the end.
                while j + TILE <= w {
                    let mut sums = [_mm256_setzero_ps(); TILE / 8];
                    let base = bp.add(j0 + j);
                    for p in 0..k {
                        let av = _mm256_set1_ps(*ap.add(p));
                        let row = base.add(p * n);
                        for (lane, sum) in sums.iter_mut().enumerate() {
                            *sum = _mm256_fmadd_ps(widen8(row.add(lane * 8)), av, *sum);
                        }
                    }
                    for (lane, sum) in sums.iter().enumerate() {
                        _mm256_storeu_ps(cp.add(j + lane * 8), *sum);
                    }
                    j += TILE;
                }
                // Sub-tile remainder: one register accumulator per 8 columns,
                // same shape as above, so it is bit-identical to the tiled path.
                while j + 8 <= w {
                    let mut sum = _mm256_setzero_ps();
                    let base = bp.add(j0 + j);
                    for p in 0..k {
                        let av = _mm256_set1_ps(*ap.add(p));
                        sum = _mm256_fmadd_ps(widen8(base.add(p * n)), av, sum);
                    }
                    _mm256_storeu_ps(cp.add(j), sum);
                    j += 8;
                }
                // Scalar tail, using the same fused multiply-add so the last
                // lanes round identically to the vectorised ones.
                while j < w {
                    let mut sum = 0.0f32;
                    let base = bp.add(j0 + j);
                    for p in 0..k {
                        sum = widen_scalar($format, *base.add(p * n)).mul_add(*ap.add(p), sum);
                    }
                    *cp.add(j) = sum;
                    j += 1;
                }
            }
        }
    };
}

/// # Safety
/// `f16c` + `avx2`; `p .. p + 8` must be a readable run of 8 `u16`.
#[target_feature(enable = "f16c,avx2")]
unsafe fn widen8_f16(p: *const u16) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    unsafe { _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i)) }
}

/// `bf16` is the top 16 bits of the `f32`, so widening is a zero-extend and a
/// shift — no conversion unit, and exact for every finite value.
///
/// # Safety
/// `avx2`; `p .. p + 8` must be a readable run of 8 `u16`.
#[target_feature(enable = "avx2")]
unsafe fn widen8_bf16(p: *const u16) -> __m256 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    unsafe {
        let raw = _mm_loadu_si128(p as *const __m128i);
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(raw)))
    }
}

stripe_simd_fn!(
    stripe_simd_f16,
    "f16c,avx2,fma",
    HalfFormat::F16,
    widen8_f16
);
stripe_simd_fn!(stripe_simd_bf16, "avx2,fma", HalfFormat::Bf16, widen8_bf16);

#[cfg(test)]
mod tests {
    use super::*;

    const FORMATS: [HalfFormat; 2] = [HalfFormat::F16, HalfFormat::Bf16];

    fn halfv(format: HalfFormat, values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|v| match format {
                HalfFormat::F16 => half::f16::from_f32(*v).to_bits(),
                HalfFormat::Bf16 => half::bf16::from_f32(*v).to_bits(),
            })
            .collect()
    }

    /// Straight-line reference: no striping, no lanes, no unrolling.
    ///
    /// Accumulates over `p` in increasing order, which is exactly what the
    /// kernel does per output element — so this is a *bitwise* oracle, not an
    /// approximate one.
    fn naive(format: HalfFormat, a: &[f32], b: &[u16], k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for (p, &av) in a.iter().enumerate().take(k) {
            for (j, slot) in out.iter_mut().enumerate() {
                *slot = widen_scalar(format, b[p * n + j]).mul_add(av, *slot);
            }
        }
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Pseudo-random but reproducible; deliberately *not* half-exact, so a
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
        for format in FORMATS {
            let b = halfv(format, &sample(k * n, 11));
            let mut out = vec![0.0f32; n];
            gemv_half_kn(format, &a, &b, &mut out, k, n);
            assert_eq!(bits(&out), bits(&naive(format, &a, &b, k, n)), "{format:?}");
        }
    }

    #[test]
    fn simd_and_scalar_stripes_agree_bit_for_bit() {
        let k = 33;
        let n = 91;
        let a = sample(k, 3);
        for format in FORMATS {
            let b = halfv(format, &sample(k * n, 5));
            for j0 in [0usize, 8, 17, 83] {
                let w = (n - j0).min(19);
                let mut scalar = vec![f32::NAN; w];
                stripe_scalar(format, &a, &b, &mut scalar, j0, k, n);
                if !simd_available(format) {
                    continue;
                }
                let mut simd = vec![f32::NAN; w];
                // SAFETY: guarded by `simd_available(format)`; `j0 + w <= n`
                // and `b` holds `k * n` elements.
                unsafe {
                    match format {
                        HalfFormat::F16 => stripe_simd_f16(&a, &b, &mut simd, j0, k, n),
                        HalfFormat::Bf16 => stripe_simd_bf16(&a, &b, &mut simd, j0, k, n),
                    }
                }
                assert_eq!(
                    bits(&simd),
                    bits(&scalar),
                    "{format:?} stripe at j0={j0} w={w}"
                );
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
            for format in FORMATS {
                let b = halfv(format, &sample(k * n, 17));
                let mut out = vec![f32::NAN; n];
                gemv_half_kn(format, &a, &b, &mut out, k, n);
                assert_eq!(
                    bits(&out),
                    bits(&naive(format, &a, &b, k, n)),
                    "{format:?} k={k} n={n}"
                );
            }
        }
    }

    #[test]
    fn zero_width_output_writes_nothing() {
        let mut out: Vec<f32> = Vec::new();
        gemv_half_kn(HalfFormat::F16, &[1.0, 2.0], &[], &mut out, 2, 0);
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
        //
        // Both formats: the tiling lives in `stripe_simd_fn!`, so it is
        // instantiated once per widening kernel and each instance has to be
        // checked against its own scalar reference.
        let k = 11;
        let n = 300;
        let a = sample(k, 41);
        let raw = sample(k * n, 43);
        let mut checked = 0usize;
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            if !simd_available(format) {
                continue;
            }
            let b = halfv(format, &raw);
            for w in 1..=(2 * TILE + 9) {
                for j0 in [0usize, 8, 64, 67] {
                    assert!(j0 + w <= n, "the sweep must not skip j0={j0} w={w}");
                    let mut scalar = vec![f32::NAN; w];
                    stripe_scalar(format, &a, &b, &mut scalar, j0, k, n);
                    let mut simd = vec![f32::NAN; w];
                    // SAFETY: `simd_available(format)` checked above, and it
                    // confirms exactly the features the selected kernel
                    // declares; `j0 + w <= n` is asserted, and `b` holds
                    // `k * n` elements.
                    unsafe {
                        match format {
                            HalfFormat::F16 => stripe_simd_f16(&a, &b, &mut simd, j0, k, n),
                            HalfFormat::Bf16 => stripe_simd_bf16(&a, &b, &mut simd, j0, k, n),
                        }
                    };
                    assert_eq!(bits(&simd), bits(&scalar), "{format:?} stripe j0={j0} w={w}");
                    checked += 1;
                }
            }
        }
        if simd_available(HalfFormat::F16) {
            assert_eq!(checked, 8 * (2 * TILE + 9), "the sweep silently shrank");
        }
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
        for format in FORMATS {
            let b = halfv(format, &sample(k * n, 29));
            let reference = naive(format, &a, &b, k, n);
            for threads in [1usize, 2, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("rayon pool");
                let mut out = vec![f32::NAN; n];
                pool.install(|| gemv_half_kn(format, &a, &b, &mut out, k, n));
                assert_eq!(
                    bits(&out),
                    bits(&reference),
                    "{format:?}: {threads} threads changed it"
                );
            }
        }
    }

    #[test]
    fn large_magnitudes_accumulate_in_f32_not_the_storage_format() {
        // 2048 + 1 is not representable in f16 (and 256 + 1 is not in bf16);
        // an accumulator in the storage format would drop every term after the
        // first. Widening to f32 first keeps them.
        let k = 64;
        let n = 1;
        let a = vec![1.0f32; k];
        for (format, big) in [(HalfFormat::F16, 2048.0f32), (HalfFormat::Bf16, 256.0f32)] {
            let mut values = vec![1.0f32; k];
            values[0] = big;
            let b = halfv(format, &values);
            let mut out = vec![0.0f32; n];
            gemv_half_kn(format, &a, &b, &mut out, k, n);
            assert_eq!(out[0], big + (k as f32 - 1.0), "{format:?}");
        }
    }

    #[test]
    #[should_panic(expected = "operands are too small")]
    fn a_short_weight_panics_rather_than_reading_out_of_bounds() {
        // `stripe_simd` derives raw pointers from `k`, `n` and `j0`, so a
        // short `b` must fail closed in release too -- not just under
        // `debug_assert`.
        let mut out = vec![0.0f32; 4];
        gemv_half_kn(HalfFormat::F16, &[1.0, 2.0], &[0u16; 5], &mut out, 2, 4);
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
        let a = [1.0f32, 10.0, 100.0];
        for format in FORMATS {
            let b = halfv(format, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let mut out = vec![0.0f32; n];
            gemv_half_kn(format, &a, &b, &mut out, k, n);
            assert_eq!(out, vec![531.0, 642.0], "{format:?}");
        }
    }

    /// Every one of the 65536 `bf16` patterns, widened by the shift the kernel
    /// uses and compared against the `half` crate.
    ///
    /// Agreement must be total except on signalling NaN, where the shift keeps
    /// the payload and `half` canonicalizes to a quiet NaN: 126 patterns, all
    /// NaN-encoding only. The blocked half GEMM this kernel replaces widens by
    /// the same shift, so a decode that switches to this path sees no change.
    #[test]
    fn bf16_widening_matches_the_half_crate_over_the_whole_domain() {
        let mut divergent = 0usize;
        for pattern in 0..=u16::MAX {
            let got = widen_scalar(HalfFormat::Bf16, pattern);
            let want = half::bf16::from_bits(pattern).to_f32();
            if got.to_bits() == want.to_bits() {
                continue;
            }
            assert!(
                got.is_nan() && want.is_nan(),
                "bf16 {pattern:#06x} widened to {got} not {want}"
            );
            divergent += 1;
        }
        // sign x (payload != 0) with the quiet bit clear: 2 * (2^6 - 1).
        assert_eq!(divergent, 126, "the NaN-encoding divergence count moved");
    }

    /// The vector widen must agree with the scalar one it is documented to
    /// reproduce, across the whole 16-bit domain, for both formats.
    #[test]
    fn vector_and_scalar_widening_agree_over_the_whole_domain() {
        let n = 8;
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let a = [1.0f32];
        for format in FORMATS {
            if !simd_available(format) {
                continue;
            }
            for row in all.chunks(n) {
                let mut simd = vec![f32::NAN; n];
                let mut scalar = vec![f32::NAN; n];
                stripe_scalar(format, &a, row, &mut scalar, 0, 1, n);
                // SAFETY: guarded by `simd_available(format)`; `row` holds
                // `1 * n` elements and `j0 = 0`.
                unsafe {
                    match format {
                        HalfFormat::F16 => stripe_simd_f16(&a, row, &mut simd, 0, 1, n),
                        HalfFormat::Bf16 => stripe_simd_bf16(&a, row, &mut simd, 0, 1, n),
                    }
                }
                for (lane, (got, want)) in simd.iter().zip(&scalar).enumerate() {
                    assert!(
                        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                        "{format:?} lane {lane}: {got} != {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_nk_weight_is_read_in_row_major_n_by_k_order() {
        // The mirror of the test above, and the reason this kernel exists: the
        // same logical matrix stored transposed must give the same answer.
        //
        // B^T = [[1, 3, 5], [2, 4, 6]], a = [1, 10, 100]
        // out = [1 + 30 + 500, 2 + 40 + 600] = [531, 642]
        let k = 3;
        let n = 2;
        let b = halfv(HalfFormat::F16, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        let a = [1.0f32, 10.0, 100.0];
        let mut out = vec![0.0f32; n];
        gemv_f16_nk(&a, &b, &mut out, k, n);
        assert_eq!(out, vec![531.0, 642.0]);
    }

    #[test]
    fn nk_and_kn_agree_on_the_same_logical_matrix() {
        // Transposing the storage must not change the value. It legitimately
        // may change the last bits -- the two kernels contract along different
        // axes -- so this is the one place a tolerance is right, and it is
        // scaled by the magnitude the k-length sum can reach.
        for (k, n) in [
            (1usize, 1usize),
            (7, 3),
            (32, 8),
            (33, 9),
            (64, 16),
            (129, 37),
        ] {
            let a = sample(k, 21);
            let kn = halfv(HalfFormat::F16, &sample(k * n, 23));
            let mut nk = vec![0u16; k * n];
            for p in 0..k {
                for j in 0..n {
                    nk[j * k + p] = kn[p * n + j];
                }
            }
            let mut from_kn = vec![0.0f32; n];
            gemv_half_kn(HalfFormat::F16, &a, &kn, &mut from_kn, k, n);
            let mut from_nk = vec![0.0f32; n];
            gemv_f16_nk(&a, &nk, &mut from_nk, k, n);
            for (lhs, rhs) in from_kn.iter().zip(&from_nk) {
                assert!(
                    (lhs - rhs).abs() <= 1e-5 * (1.0 + lhs.abs()),
                    "k={k} n={n}: {lhs} vs {rhs}"
                );
            }
        }
    }

    #[test]
    fn nk_simd_and_scalar_rows_agree_bit_for_bit() {
        // The documented reduction order is the contract; `dot_row_scalar` is
        // its executable statement, so the vector path must match it exactly at
        // every `k` around the 32- and 8-element loop boundaries.
        if !simd_available(HalfFormat::F16) {
            return;
        }
        for k in [
            1usize, 7, 8, 9, 31, 32, 33, 39, 40, 64, 65, 96, 127, 128, 3584,
        ] {
            let a = sample(k, 31);
            let row = halfv(HalfFormat::F16, &sample(k, 37));
            let scalar = dot_row_scalar(&a, &row, k);
            // SAFETY: guarded by `simd_available(F16)`; `a` and `row` both hold
            // exactly `k` elements.
            let simd = unsafe { dot_row_simd(&a, &row, k) };
            assert_eq!(simd.to_bits(), scalar.to_bits(), "k={k}");
        }
    }

    #[test]
    fn nk_is_independent_of_the_thread_count() {
        // `ROW_STRIPE` partitions the output, never the contraction, so the
        // answer must not move when the pool width does.
        let k = 300;
        let n = 133;
        let a = sample(k, 41);
        let b = halfv(HalfFormat::F16, &sample(k * n, 43));
        let mut reference = vec![0.0f32; n];
        gemv_f16_nk(&a, &b, &mut reference, k, n);
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let mut out = vec![0.0f32; n];
            pool.install(|| gemv_f16_nk(&a, &b, &mut out, k, n));
            assert_eq!(bits(&out), bits(&reference), "threads={threads}");
        }
    }

    #[test]
    #[should_panic(expected = "operands are too small")]
    fn a_short_nk_weight_panics_rather_than_reading_out_of_bounds() {
        let mut out = vec![0.0f32; 4];
        gemv_f16_nk(&[1.0, 2.0], &[0u16; 5], &mut out, 2, 4);
    }

    #[test]
    fn nk_handles_an_empty_contraction_and_a_partial_final_stripe() {
        // k = 0 writes zeros without reading any weight; n = 13 leaves the last
        // ROW_STRIPE chunk partial.
        let mut out = vec![f32::NAN; 13];
        gemv_f16_nk(&[], &[], &mut out, 0, 13);
        assert_eq!(out, vec![0.0f32; 13]);

        let k = 5;
        let n = 13;
        let a = sample(k, 47);
        let b = halfv(HalfFormat::F16, &sample(k * n, 53));
        let mut out = vec![0.0f32; n];
        gemv_f16_nk(&a, &b, &mut out, k, n);
        for (j, got) in out.iter().enumerate() {
            let want = dot_row_scalar(&a, &b[j * k..j * k + k], k);
            assert_eq!(got.to_bits(), want.to_bits(), "row {j}");
        }
    }
}
