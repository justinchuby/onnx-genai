//! Native integer GEMM for `QLinearMatMul`, on the operand bytes.
//!
//! # Why this exists
//!
//! `QLinearMatMul`'s MLAS route is a research-build-only path (`--features
//! mlas`). The build this project ships had no integer GEMM at all: it widened
//! **both** operands to `Vec<i32>` per call and ran a scalar rank-1 update. For
//! `1x2048x2048` that is a 16 MiB materialisation of `B` on every call, and the
//! inner loop then streams those 16 MiB once per row of `A` instead of the
//! 4 MiB the weight actually occupies. Measured through an ORT session at one
//! thread, `ours/ORT` was **11.8x at M=1 and 12.0x at M=128** — by a wide margin
//! the largest loss in the matmul family.
//!
//! # What it computes
//!
//! Exactly what the scalar loop computed, which is the zero-point expansion
//!
//! ```text
//! sum_k (a_ik - za_i) * (b_kj - zb_j)
//!   == sum_k (a_ik - za_i) * b_kj  -  zb_j * sum_k (a_ik - za_i)
//! ```
//!
//! an identity over the integers. Every accumulation here is `i32` wrapping
//! arithmetic, i.e. exactly arithmetic mod 2^32, which is **associative and
//! commutative**. Blocking, re-ordering and vectorising the sum therefore
//! cannot change a single output bit, including on overflow — the property the
//! tests in this module assert directly rather than assume.
//!
//! # How it goes fast
//!
//! `vpmaddwd` (`_mm256_madd_epi16`) multiplies eight *pairs* of `i16` and sums
//! each pair into an `i32`: 16 multiply-accumulates per instruction, against
//! four for the widen-and-`vpmulld` shape LLVM produces from the scalar loop.
//! It is exact here rather than merely close: a centred `a` is in `[-255, 255]`
//! and a raw `b` in `[-128, 255]`, so a product is at most `65025` in magnitude
//! and the pairwise sum at most `130050` — nowhere near `i32` overflow, and
//! `vpmaddwd` does not saturate the way `vpmaddubsw` does. (That saturation is
//! precisely why the MLAS route has to translate operands into the `u8 x u8`
//! domain first; this kernel needs no such translation.)
//!
//! Using it requires `B`'s two `k` neighbours to sit next to each other, which
//! `B[k][n]` row-major does not give, so `B` is packed into 16-column,
//! `k`-pair-interleaved tiles. The pack is SIMD (four instructions per 16
//! columns per `k` pair) and costs about 1% of the GEMM it feeds; it is blocked
//! to a 256 KiB scratch panel that stays in L2 while every row of `A` sweeps
//! it, so `B` is read from memory once rather than once per row.

use rayon::prelude::*;

/// One quantized operand: its dense bytes and whether they are `i8` or `u8`.
#[derive(Clone, Copy)]
pub(crate) struct Operand<'a> {
    pub bytes: &'a [u8],
    pub signed: bool,
}

impl Operand<'_> {
    /// The element at `index`, widened to `i32` in its own sign domain.
    #[inline]
    fn at(&self, index: usize) -> i32 {
        let byte = self.bytes[index];
        if self.signed {
            byte as i8 as i32
        } else {
            i32::from(byte)
        }
    }
}

/// Columns per register tile: two `__m256i` accumulators per row.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const NR: usize = 16;
/// Rows per register tile. Four rows share every packed `B` load.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MR: usize = 4;
/// Columns per packed panel.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const NC: usize = 256;
/// `k` rows per packed panel. Even, so `k` pairs never straddle two panels.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const KC: usize = 512;

/// Smallest `m * k * n` worth splitting across the pool.
///
/// Inherited unchanged from the scalar loop this replaces: the fork is not free
/// and below this it costs more than the split saves.
const PARALLEL_MIN_WORK: usize = 1 << 16;

/// `products[i][j] += sum_k (a[i][k] - za[i]) * b[k][j] - zb[j] * sum_k (a[i][k] - za[i])`.
///
/// `products` must be `m * n` and is accumulated into, so the caller decides
/// whether it starts zeroed. `a_zero_points` is per row of `A`, `b_zero_points`
/// per column of `B`; both are already widened.
///
/// Deterministic at every thread count: tasks own disjoint column blocks, and
/// within a block the accumulation order is fixed by the blocking constants,
/// not by the pool.
pub(crate) fn qgemm(
    a: Operand<'_>,
    b: Operand<'_>,
    a_zero_points: &[i32],
    b_zero_points: &[i32],
    m: usize,
    k: usize,
    n: usize,
    products: &mut [i32],
) {
    debug_assert_eq!(products.len(), m * n);
    debug_assert_eq!(a_zero_points.len(), m);
    debug_assert_eq!(b_zero_points.len(), n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        return;
    }

    // `sum_k (a_ik - za_i)`, the multiplier on the column zero point. Computed
    // once per row here rather than inside the `k` loop, which is the only
    // reason the row loop below can be blocked at all.
    let mut a_sums = vec![0i32; m];
    for (row, sum) in a_sums.iter_mut().enumerate() {
        let zero = a_zero_points[row];
        let mut acc = 0i32;
        for index in row * k..row * k + k {
            acc = acc.wrapping_add(a.at(index).wrapping_sub(zero));
        }
        *sum = acc;
    }

    accumulate_products(a, b, a_zero_points, m, k, n, products);

    // The `- zb_j * sum_k(...)` half of the expansion, applied once per output
    // rather than once per `k`.
    for (row, out) in products.chunks_mut(n).enumerate() {
        let sum = a_sums[row];
        if sum == 0 {
            continue;
        }
        for (value, &zero) in out.iter_mut().zip(b_zero_points) {
            *value = value.wrapping_sub(sum.wrapping_mul(zero));
        }
    }
}

/// The `sum_k (a_ik - za_i) * b_kj` half. Dispatches to the AVX2 kernel when
/// the host has it and falls back to the portable loop otherwise.
fn accumulate_products(
    a: Operand<'_>,
    b: Operand<'_>,
    a_zero_points: &[i32],
    m: usize,
    k: usize,
    n: usize,
    products: &mut [i32],
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: guarded by the runtime feature probe immediately above.
        unsafe { accumulate_products_avx2(a, b, a_zero_points, m, k, n, products) };
        return;
    }
    accumulate_products_scalar(a, b, a_zero_points, m, k, n, products);
}

/// Portable reference: the loop shape this module replaces, on bytes instead of
/// widened `i32`. Also the oracle the AVX2 kernel is proven bit-equal against.
fn accumulate_products_scalar(
    a: Operand<'_>,
    b: Operand<'_>,
    a_zero_points: &[i32],
    m: usize,
    k: usize,
    n: usize,
    products: &mut [i32],
) {
    let row_work = |row: usize, out: &mut [i32]| {
        let zero = a_zero_points[row];
        for inner in 0..k {
            let centered = a.at(row * k + inner).wrapping_sub(zero);
            if centered == 0 {
                continue;
            }
            let base = inner * n;
            for (value, column) in out.iter_mut().zip(0..n) {
                *value = value.wrapping_add(centered.wrapping_mul(b.at(base + column)));
            }
        }
    };
    if m <= 1 || rayon::current_num_threads() <= 1 || m * n * k < PARALLEL_MIN_WORK {
        for (row, out) in products.chunks_mut(n).enumerate() {
            row_work(row, out);
        }
    } else {
        products
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, out)| row_work(row, out));
    }
}

/// `products` as a pointer that the block tasks may share.
///
/// A task owns the rectangle `[m0, m0 + mc) x [n0, n0 + nc)` and writes nothing
/// outside it. Column starts are multiples of `block_width` and row starts are
/// multiples of `block_height`, with `nc <= block_width` and `mc <=
/// block_height`, so the rectangles tile the output and no two tasks address
/// the same `i32`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
struct ProductsPtr(*mut i32);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl ProductsPtr {
    /// # Safety
    ///
    /// `offset` must be within the `products` slice this was built from.
    ///
    /// Taking the pointer through a method rather than reading `.0` also keeps
    /// closure capture on the whole struct, which is what carries `Send`.
    #[inline]
    unsafe fn at(&self, offset: usize) -> *mut i32 {
        // SAFETY: the caller keeps `offset` in bounds.
        unsafe { self.0.add(offset) }
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// SAFETY: the pointer is only ever offset into disjoint column ranges by the
// task that owns them; see `accumulate_products_avx2`.
unsafe impl Send for ProductsPtr {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// SAFETY: as above.
unsafe impl Sync for ProductsPtr {}

/// AVX2 driver: pack `B` panel by panel, sweep `A` over each packed panel.
///
/// # Safety
///
/// The host must support AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_products_avx2(
    a: Operand<'_>,
    b: Operand<'_>,
    a_zero_points: &[i32],
    m: usize,
    k: usize,
    n: usize,
    products: &mut [i32],
) {
    // `A` centred and paired along `k`, once for the whole call: lane `p` of
    // row `i` holds `(a[i][2p] - za_i, a[i][2p+1] - za_i)` as two `i16` in one
    // `i32`, which is what `vpbroadcastd` + `vpmaddwd` consume. `2 * m * k`
    // bytes, against the `4 * k * n` this kernel stops materialising.
    let pairs = k.div_ceil(2);
    let mut a_pairs = vec![0i32; m * pairs];
    for row in 0..m {
        let zero = a_zero_points[row];
        for pair in 0..pairs {
            let lo = a.at(row * k + 2 * pair).wrapping_sub(zero);
            let hi = if 2 * pair + 1 < k {
                a.at(row * k + 2 * pair + 1).wrapping_sub(zero)
            } else {
                0
            };
            a_pairs[row * pairs + pair] = ((lo as u16 as u32) | ((hi as u16 as u32) << 16)) as i32;
        }
    }

    let parallel = rayon::current_num_threads() > 1 && m * n * k >= PARALLEL_MIN_WORK;
    // Packing `B` pays for itself by being re-read once per row block. At
    // `m <= MR` there is exactly one row block, so the panel would be written
    // and read once each and the pack would be pure overhead -- and it is not
    // small overhead: it is `2 * k * n` bytes of stores against a GEMV that
    // only reads `k * n`. Decode is `m == 1`, so this is the hot shape, not a
    // corner. The fused kernel below does the same interleave in registers and
    // feeds `vpmaddwd` directly, never spilling a panel.
    let fused = m <= MR;
    // The packed path re-reads its panel per row block, so a panel that fits L2
    // is what matters and `NC` is fixed. The fused path reads every byte of its
    // block exactly once, so splitting the columns further only re-walks `B`:
    // serially it takes the whole row at once, and in parallel it takes the
    // widest block that still gives every worker something to do. Blocks stay a
    // multiple of a cache line so no two workers share one.
    let block_width = match (fused, parallel) {
        (false, _) => NC,
        (true, false) => n,
        (true, true) => n
            .div_ceil(rayon::current_num_threads())
            .next_multiple_of(64)
            .max(64),
    };
    let column_blocks = n.div_ceil(block_width);
    // A column block owns a packed panel, so splitting the columns further to
    // reach every worker would shrink the panel and re-walk `B`. Splitting the
    // *rows* instead duplicates only the pack, which is around a percent of the
    // GEMM it feeds -- so the row split is what grows to fill the pool, and it
    // grows only as far as the pool needs. Left at one block, an `n` of 2048
    // offers eight tasks and a sixteen-worker pool leaves half of itself idle
    // and spinning, which measured *slower* than running the whole GEMM on one
    // thread.
    let row_blocks = if fused || !parallel {
        1
    } else {
        rayon::current_num_threads()
            .div_ceil(column_blocks)
            .clamp(1, m.div_ceil(MR))
    };
    let block_height = m.div_ceil(row_blocks).next_multiple_of(MR);
    let blocks = column_blocks * row_blocks;
    let base = ProductsPtr(products.as_mut_ptr());
    let run_block = |block: usize| {
        let n0 = (block % column_blocks) * block_width;
        let nc = block_width.min(n - n0);
        let m0 = (block / column_blocks) * block_height;
        if m0 >= m {
            return;
        }
        let mc = block_height.min(m - m0);
        if fused {
            // SAFETY: AVX2 is guaranteed by the caller. The kernel reads
            // `b[row * n + n0 .. + nc]` for `row < k` and `a_pairs[r * pairs +
            // p]` for `r < m <= MR` and `p < pairs`, and writes `products[r * n
            // + n0 .. + nc]`, which is this task's column block.
            //
            // The tile count per row falls as the row count rises so that the
            // `2 * R * T` accumulators keep fitting the register file.
            unsafe {
                let a_pairs = a_pairs.as_ptr();
                let out = base.at(n0);
                match (m, b.signed) {
                    (1, false) => {
                        accumulate_fused::<1, 4, false>(b, a_pairs, pairs, n, n0, nc, k, out)
                    }
                    (1, true) => {
                        accumulate_fused::<1, 4, true>(b, a_pairs, pairs, n, n0, nc, k, out)
                    }
                    (2, false) => {
                        accumulate_fused::<2, 2, false>(b, a_pairs, pairs, n, n0, nc, k, out)
                    }
                    (2, true) => {
                        accumulate_fused::<2, 2, true>(b, a_pairs, pairs, n, n0, nc, k, out)
                    }
                    (3, false) => {
                        accumulate_fused::<3, 1, false>(b, a_pairs, pairs, n, n0, nc, k, out)
                    }
                    (3, true) => {
                        accumulate_fused::<3, 1, true>(b, a_pairs, pairs, n, n0, nc, k, out)
                    }
                    (_, false) => {
                        accumulate_fused::<4, 1, false>(b, a_pairs, pairs, n, n0, nc, k, out)
                    }
                    (_, true) => {
                        accumulate_fused::<4, 1, true>(b, a_pairs, pairs, n, n0, nc, k, out)
                    }
                }
            }
            return;
        }
        // One panel per task, reused across every `k` block: `KC/2` pairs of
        // `NC` columns is 256 KiB, which stays in L2 while `A` sweeps it.
        let tiles = nc.div_ceil(NR);
        let mut panel = vec![0i16; (KC / 2) * tiles * NR * 2];
        let mut k0 = 0;
        while k0 < k {
            let kc = KC.min(k - k0);
            let panel_pairs = kc.div_ceil(2);
            // SAFETY: AVX2 is guaranteed by the caller. `panel` is sized for
            // `KC/2 >= panel_pairs` pairs of `tiles * NR` columns, and the pack
            // reads `b[(k0 + row) * n + n0 + column]` only for `row < kc` and
            // `column < nc`, both in bounds.
            unsafe {
                if b.signed {
                    pack_panel::<true>(b, &mut panel, n, n0, nc, k0, kc, tiles);
                } else {
                    pack_panel::<false>(b, &mut panel, n, n0, nc, k0, kc, tiles);
                }
            };
            let mut r0 = m0;
            while r0 < m0 + mc {
                let rows = MR.min(m0 + mc - r0);
                for tile in 0..tiles {
                    let c0 = n0 + tile * NR;
                    // From `nc`, as in `pack_panel`, so a `block_width` that is
                    // not a multiple of `NR` cannot make the two disagree.
                    let width = NR.min(nc - tile * NR);
                    // SAFETY: AVX2 as above. The tile reads `panel_pairs`
                    // pairs from tile `tile`, which `pack_panel` just wrote,
                    // and `a_pairs[(r0 + r) * pairs + p]` for `r < rows <= MR`
                    // and `p < panel_pairs`, in bounds because `k0 / 2 +
                    // panel_pairs <= pairs`. It writes `products[(r0 + r) * n +
                    // c0 .. + width]`, which lies inside this task's column
                    // block and inside `products`.
                    unsafe {
                        accumulate_tile(
                            panel.as_ptr().add(tile * (KC / 2) * NR * 2),
                            a_pairs.as_ptr().add(r0 * pairs + k0 / 2),
                            pairs,
                            panel_pairs,
                            rows,
                            width,
                            base.at(r0 * n + c0),
                            n,
                        );
                    }
                }
                r0 += MR;
            }
            k0 += KC;
        }
    };

    if blocks <= 1 || !parallel {
        for block in 0..blocks {
            run_block(block);
        }
    } else {
        (0..blocks).into_par_iter().for_each(run_block);
    }
}

/// Packs `B[k0..k0+kc][n0..n0+nc]` into `k`-pair-interleaved 16-column tiles.
///
/// Tile `t`, pair `p` occupies `panel[t * (KC/2) * NR * 2 + p * 32 ..][..32]`
/// as `(b[2p][c], b[2p+1][c])` for the tile's 16 columns in order. Missing
/// columns and a missing odd `k` partner are packed as zero, which contributes
/// a zero product rather than a special case in the inner loop.
///
/// # Safety
///
/// AVX2 required. `panel` must hold `tiles * (KC/2) * NR * 2` `i16`, and
/// `b` must address `(k0 + kc) * n` elements.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_panel<const SIGNED: bool>(
    b: Operand<'_>,
    panel: &mut [i16],
    n: usize,
    n0: usize,
    nc: usize,
    k0: usize,
    kc: usize,
    tiles: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        for tile in 0..tiles {
            let c0 = tile * NR;
            let width = NR.min(nc - c0);
            let out = panel.as_mut_ptr().add(tile * (KC / 2) * NR * 2);
            let mut pair = 0;
            while 2 * pair < kc {
                let dst = out.add(pair * NR * 2);
                if width == NR {
                    let lo_row = b.bytes.as_ptr().add((k0 + 2 * pair) * n + n0 + c0);
                    let w0 = widen16::<SIGNED>(lo_row);
                    let w1 = if 2 * pair + 1 < kc {
                        widen16::<SIGNED>(lo_row.add(n))
                    } else {
                        _mm256_setzero_si256()
                    };
                    // `vpmovzx` leaves the 16 columns as four qwords
                    // `c0-3 | c4-7 | c8-11 | c12-15`, and `vpunpck*wd`
                    // interleaves *within* each 128-bit lane. Swapping the two
                    // middle qwords first makes `unpacklo` yield columns 0-7
                    // and `unpackhi` columns 8-15, both in order, so a lane of
                    // the accumulator maps to one column with no final permute.
                    let w0 = _mm256_permute4x64_epi64(w0, 0b1101_1000);
                    let w1 = _mm256_permute4x64_epi64(w1, 0b1101_1000);
                    _mm256_storeu_si256(dst.cast(), _mm256_unpacklo_epi16(w0, w1));
                    _mm256_storeu_si256(dst.add(16).cast(), _mm256_unpackhi_epi16(w0, w1));
                } else {
                    for column in 0..NR {
                        let (lo, hi) = if column < width {
                            let index = (k0 + 2 * pair) * n + n0 + c0 + column;
                            let lo = b.at(index) as i16;
                            let hi = if 2 * pair + 1 < kc {
                                b.at(index + n) as i16
                            } else {
                                0
                            };
                            (lo, hi)
                        } else {
                            (0, 0)
                        };
                        *dst.add(column * 2) = lo;
                        *dst.add(column * 2 + 1) = hi;
                    }
                }
                pair += 1;
            }
        }
    }
}

/// The pack-free kernel for `m <= MR`: interleaves `B`'s `k` pairs in registers
/// and feeds them straight to `vpmaddwd`.
///
/// `R` is the row count and `T` the number of 16-column tiles held in
/// registers, chosen by the caller so `2 * R * T` accumulators fit the sixteen
/// architectural `ymm` registers.
///
/// Two things keep this near the memory bound rather than the issue bound.
/// First the accumulators stay in registers for a whole `k` block, so the only
/// traffic in the inner loop is `B` itself. Second the column permutation
/// `vpunpck*wd` implies is *not* undone per iteration: the accumulators are
/// simply held in the permuted order `c0-3 c8-11 | c4-7 c12-15` and put back in
/// order once, when the block is flushed. That removes two `vperm` from every
/// sixteen columns of every `k` pair and leaves eight instructions per
/// thirty-two multiply-accumulates.
///
/// `B` is walked down a fixed column strip, so a `k` block's footprint is
/// `FUSED_KC` cache lines -- L1-resident -- and every line is read once.
///
/// # Safety
///
/// AVX2 required. `b` must address `k * n` bytes; `a_pairs` must address `R`
/// rows of `ceil(k/2)` `i32` at stride `pairs`; `out` must address `R` rows of
/// at least `column + 16 * T` `i32` at stride `n`, and this call writes only
/// `out[r * n + column .. + 16 * T]` for `r < R`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_strip<const R: usize, const T: usize, const SIGNED: bool>(
    b: Operand<'_>,
    a_pairs: *const i32,
    pairs: usize,
    n: usize,
    column: usize,
    k0: usize,
    kc: usize,
    k: usize,
    out: *mut i32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        let mut acc = [[[_mm256_setzero_si256(); 2]; T]; R];
        let rows = b.bytes.as_ptr().add(k0 * n + column);
        for pair in 0..kc.div_ceil(2) {
            let lo_row = rows.add(2 * pair * n);
            let has_hi = k0 + 2 * pair + 1 < k;
            let mut av = [_mm256_setzero_si256(); R];
            for (row, av) in av.iter_mut().enumerate() {
                *av = _mm256_set1_epi32(*a_pairs.add(row * pairs + k0 / 2 + pair));
            }
            for tile in 0..T {
                let w0 = widen16::<SIGNED>(lo_row.add(tile * NR));
                let w1 = if has_hi {
                    widen16::<SIGNED>(lo_row.add(n + tile * NR))
                } else {
                    _mm256_setzero_si256()
                };
                let lo = _mm256_unpacklo_epi16(w0, w1);
                let hi = _mm256_unpackhi_epi16(w0, w1);
                for (av, acc) in av.iter().zip(acc.iter_mut()) {
                    acc[tile][0] = _mm256_add_epi32(acc[tile][0], _mm256_madd_epi16(*av, lo));
                    acc[tile][1] = _mm256_add_epi32(acc[tile][1], _mm256_madd_epi16(*av, hi));
                }
            }
        }
        // `vpunpcklwd` pairs the low four `i16` of *each* 128-bit lane, so the
        // first accumulator holds columns 0-3 and 8-11 and the second 4-7 and
        // 12-15. One `vperm2i128` each puts them back in order.
        for (row, acc) in acc.iter().enumerate() {
            for (tile, acc) in acc.iter().enumerate() {
                let dst = out.add(row * n + tile * NR);
                let low = _mm256_permute2x128_si256(acc[0], acc[1], 0x20);
                let high = _mm256_permute2x128_si256(acc[0], acc[1], 0x31);
                _mm256_storeu_si256(
                    dst.cast(),
                    _mm256_add_epi32(_mm256_loadu_si256(dst.cast()), low),
                );
                _mm256_storeu_si256(
                    dst.add(8).cast(),
                    _mm256_add_epi32(_mm256_loadu_si256(dst.add(8).cast()), high),
                );
            }
        }
    }
}

/// `k` rows a [`fused_strip`] call holds accumulators across. Even, so `k`
/// pairs never straddle two blocks. At 256 a strip's footprint is 256 cache
/// lines, which stays in L1 while the neighbouring strips of the same block are
/// walked.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const FUSED_KC: usize = 256;

/// Drives [`fused_strip`] across a column block: the widest strip the register
/// file allows, then a single tile, then a scalar remainder.
///
/// # Safety
///
/// AVX2 required. `b` must address `k * n` bytes, `a_pairs` `R` rows of
/// `ceil(k/2)` `i32` at stride `pairs`, and `out` `R` rows of `nc` `i32` at
/// stride `n`; only those are written.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate_fused<const R: usize, const T: usize, const SIGNED: bool>(
    b: Operand<'_>,
    a_pairs: *const i32,
    pairs: usize,
    n: usize,
    n0: usize,
    nc: usize,
    k: usize,
    out: *mut i32,
) {
    unsafe {
        let mut k0 = 0;
        while k0 < k {
            let kc = FUSED_KC.min(k - k0);
            let mut column = 0;
            while column + NR * T <= nc {
                fused_strip::<R, T, SIGNED>(
                    b,
                    a_pairs,
                    pairs,
                    n,
                    n0 + column,
                    k0,
                    kc,
                    k,
                    out.add(column),
                );
                column += NR * T;
            }
            while column + NR <= nc {
                fused_strip::<R, 1, SIGNED>(
                    b,
                    a_pairs,
                    pairs,
                    n,
                    n0 + column,
                    k0,
                    kc,
                    k,
                    out.add(column),
                );
                column += NR;
            }
            // Fewer than sixteen columns left. A vector strip would have to
            // mask both the `B` load and the accumulator store, so the plain
            // loop is both simpler and, over at most fifteen columns, no
            // slower.
            for pair in 0..kc.div_ceil(2) {
                let index = (k0 + 2 * pair) * n + n0;
                let has_hi = k0 + 2 * pair + 1 < k;
                for (row, column) in (0..R).flat_map(|row| (column..nc).map(move |c| (row, c))) {
                    let packed = *a_pairs.add(row * pairs + k0 / 2 + pair) as u32;
                    let left = i32::from(packed as u16 as i16);
                    let right = i32::from((packed >> 16) as u16 as i16);
                    let mut sum = left.wrapping_mul(b.at(index + column));
                    if has_hi {
                        sum = sum.wrapping_add(right.wrapping_mul(b.at(index + n + column)));
                    }
                    let dst = out.add(row * n + column);
                    *dst = (*dst).wrapping_add(sum);
                }
            }
            k0 += FUSED_KC;
        }
    }
}

/// Sixteen operand bytes widened to `i16`, in the operand's own sign domain.
///
/// # Safety
///
/// AVX2 required; `src` must address 16 readable bytes.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn widen16<const SIGNED: bool>(src: *const u8) -> std::arch::x86_64::__m256i {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        let raw = _mm_loadu_si128(src.cast());
        if SIGNED {
            _mm256_cvtepi8_epi16(raw)
        } else {
            _mm256_cvtepu8_epi16(raw)
        }
    }
}

/// Accumulates one `rows x width` register tile over `panel_pairs` `k` pairs.
///
/// # Safety
///
/// AVX2 required. `panel` must address `panel_pairs * 32` `i16`; `a_pairs` must
/// address `rows` rows of at least `panel_pairs` `i32` at stride `a_stride`;
/// `out` must address `rows` rows of at least `width` `i32` at stride
/// `out_stride`, and this call writes only those.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate_tile(
    panel: *const i16,
    a_pairs: *const i32,
    a_stride: usize,
    panel_pairs: usize,
    rows: usize,
    width: usize,
    out: *mut i32,
    out_stride: usize,
) {
    unsafe {
        match rows {
            1 => tile::<1>(
                panel,
                a_pairs,
                a_stride,
                panel_pairs,
                width,
                out,
                out_stride,
            ),
            2 => tile::<2>(
                panel,
                a_pairs,
                a_stride,
                panel_pairs,
                width,
                out,
                out_stride,
            ),
            3 => tile::<3>(
                panel,
                a_pairs,
                a_stride,
                panel_pairs,
                width,
                out,
                out_stride,
            ),
            _ => tile::<4>(
                panel,
                a_pairs,
                a_stride,
                panel_pairs,
                width,
                out,
                out_stride,
            ),
        }
    }
}

/// The `R x 16` microkernel. See [`accumulate_tile`] for the safety contract.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile<const R: usize>(
    panel: *const i16,
    a_pairs: *const i32,
    a_stride: usize,
    panel_pairs: usize,
    width: usize,
    out: *mut i32,
    out_stride: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        let mut acc = [[_mm256_setzero_si256(); 2]; R];
        for pair in 0..panel_pairs {
            let base = panel.add(pair * NR * 2);
            let b0 = _mm256_loadu_si256(base.cast());
            let b1 = _mm256_loadu_si256(base.add(16).cast());
            for (row, acc) in acc.iter_mut().enumerate() {
                let av = _mm256_set1_epi32(*a_pairs.add(row * a_stride + pair));
                acc[0] = _mm256_add_epi32(acc[0], _mm256_madd_epi16(av, b0));
                acc[1] = _mm256_add_epi32(acc[1], _mm256_madd_epi16(av, b1));
            }
        }
        if width == NR {
            for (row, acc) in acc.iter().enumerate() {
                let dst = out.add(row * out_stride);
                _mm256_storeu_si256(
                    dst.cast(),
                    _mm256_add_epi32(_mm256_loadu_si256(dst.cast()), acc[0]),
                );
                _mm256_storeu_si256(
                    dst.add(8).cast(),
                    _mm256_add_epi32(_mm256_loadu_si256(dst.add(8).cast()), acc[1]),
                );
            }
        } else {
            // Ragged right edge: the tile computed 16 columns because the panel
            // was zero-padded to 16, but only `width` of them exist in the
            // output.
            let mut lanes = [0i32; NR];
            for (row, acc) in acc.iter().enumerate() {
                _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc[0]);
                _mm256_storeu_si256(lanes.as_mut_ptr().add(8).cast(), acc[1]);
                let dst = out.add(row * out_stride);
                for (column, lane) in lanes.iter().enumerate().take(width) {
                    *dst.add(column) = (*dst.add(column)).wrapping_add(*lane);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition, written the slowest and most obvious way there is.
    fn oracle(
        a: Operand<'_>,
        b: Operand<'_>,
        a_zero_points: &[i32],
        b_zero_points: &[i32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<i32> {
        let mut out = vec![0i32; m * n];
        for row in 0..m {
            for column in 0..n {
                let mut acc = 0i32;
                for inner in 0..k {
                    let left = a.at(row * k + inner).wrapping_sub(a_zero_points[row]);
                    let right = b.at(inner * n + column).wrapping_sub(b_zero_points[column]);
                    acc = acc.wrapping_add(left.wrapping_mul(right));
                }
                out[row * n + column] = acc;
            }
        }
        out
    }

    fn bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as u8
            })
            .collect()
    }

    fn zero_points(len: usize, signed: bool, seed: u64) -> Vec<i32> {
        bytes(len, seed)
            .into_iter()
            .map(|byte| {
                if signed {
                    byte as i8 as i32
                } else {
                    i32::from(byte)
                }
            })
            .collect()
    }

    /// `qgemm` with the AVX2 arm forced off, so the portable path is covered on
    /// a host that has AVX2.
    fn portable(
        a: Operand<'_>,
        b: Operand<'_>,
        a_zero_points: &[i32],
        b_zero_points: &[i32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<i32> {
        let mut out = vec![0i32; m * n];
        accumulate_products_scalar(a, b, a_zero_points, m, k, n, &mut out);
        for (row, chunk) in out.chunks_mut(n).enumerate() {
            let sum = (0..k).fold(0i32, |acc, inner| {
                acc.wrapping_add(a.at(row * k + inner).wrapping_sub(a_zero_points[row]))
            });
            for (value, &zero) in chunk.iter_mut().zip(b_zero_points) {
                *value = value.wrapping_sub(sum.wrapping_mul(zero));
            }
        }
        out
    }

    fn check(m: usize, k: usize, n: usize, a_signed: bool, b_signed: bool) {
        let a_bytes = bytes(m * k, 0x51ed_2701 ^ (m as u64));
        let b_bytes = bytes(k * n, 0x9e37_79b9 ^ (n as u64));
        let a = Operand {
            bytes: &a_bytes,
            signed: a_signed,
        };
        let b = Operand {
            bytes: &b_bytes,
            signed: b_signed,
        };
        let az = zero_points(m, a_signed, 0x1234_5677);
        let bz = zero_points(n, b_signed, 0x7654_3210);

        let expect = oracle(a, b, &az, &bz, m, k, n);
        let mut got = vec![0i32; m * n];
        qgemm(a, b, &az, &bz, m, k, n, &mut got);
        let label = format!("{m}x{k}x{n} a_signed={a_signed} b_signed={b_signed}");
        assert_eq!(got, expect, "{label}: qgemm disagrees with the oracle");
        assert_eq!(
            portable(a, b, &az, &bz, m, k, n),
            expect,
            "{label}: the portable arm disagrees with the oracle"
        );
    }

    #[test]
    fn matches_the_integer_oracle_across_shapes_and_signedness() {
        for &(m, k, n) in &[
            (1usize, 1usize, 1usize),
            (1, 3, 5),
            (1, 2048, 33),
            (1, 600, 200),
            (1, 513, 64),
            (2, 7, 16),
            (2, 300, 100),
            (3, 65, 17),
            (3, 257, 79),
            (4, 64, 64),
            (4, 259, 143),
            (5, 33, 47),
            (7, 129, 131),
            (8, 512, 256),
            (17, 31, 15),
            (33, 8, 259),
        ] {
            for a_signed in [false, true] {
                for b_signed in [false, true] {
                    check(m, k, n, a_signed, b_signed);
                }
            }
        }
    }

    /// The SIMD kernel and the portable loop must agree **bit for bit**, not
    /// approximately: they are two orderings of the same wrapping `i32` sum, and
    /// wrapping addition is associative. Driven at a shape wide enough to use
    /// every path — full tiles, a ragged right edge, several `k` panels and an
    /// odd `k` so the final pair has no partner.
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn the_simd_kernel_is_bit_identical_to_the_portable_loop() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: host has no AVX2");
            return;
        }
        let (m, k, n) = (9usize, 1029usize, 279usize);
        let a_bytes = bytes(m * k, 0xabcd_1234);
        let b_bytes = bytes(k * n, 0x1357_9bdf);
        for (a_signed, b_signed) in [(false, false), (false, true), (true, false), (true, true)] {
            let a = Operand {
                bytes: &a_bytes,
                signed: a_signed,
            };
            let b = Operand {
                bytes: &b_bytes,
                signed: b_signed,
            };
            let az = zero_points(m, a_signed, 0x2222_3333);

            let mut simd = vec![0i32; m * n];
            // SAFETY: guarded by the feature probe above.
            unsafe { accumulate_products_avx2(a, b, &az, m, k, n, &mut simd) };
            let mut portable = vec![0i32; m * n];
            accumulate_products_scalar(a, b, &az, m, k, n, &mut portable);
            assert_eq!(
                simd, portable,
                "a_signed={a_signed} b_signed={b_signed}: SIMD and portable disagree"
            );
        }
    }

    /// The two SIMD kernels are selected by `m` alone, so the bit-identity
    /// claim has to be made on both sides of that boundary. This is the
    /// pack-free side: `m <= MR`, the decode shape, driven at a `k` long enough
    /// to span several `FUSED_KC` blocks and an `n` that leaves a wide strip, a
    /// single tile and a scalar remainder all non-empty.
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn the_pack_free_kernel_is_bit_identical_to_the_portable_loop() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: host has no AVX2");
            return;
        }
        let (k, n) = (1031usize, 279usize);
        for m in 1..=MR {
            let a_bytes = bytes(m * k, 0x5150_4948);
            let b_bytes = bytes(k * n, 0x0f1e_2d3c);
            for (a_signed, b_signed) in [(false, false), (true, true), (false, true)] {
                let a = Operand {
                    bytes: &a_bytes,
                    signed: a_signed,
                };
                let b = Operand {
                    bytes: &b_bytes,
                    signed: b_signed,
                };
                let az = zero_points(m, a_signed, 0x9999_1111);

                let mut simd = vec![0i32; m * n];
                // SAFETY: guarded by the feature probe above.
                unsafe { accumulate_products_avx2(a, b, &az, m, k, n, &mut simd) };
                let mut portable = vec![0i32; m * n];
                accumulate_products_scalar(a, b, &az, m, k, n, &mut portable);
                assert_eq!(
                    simd, portable,
                    "m={m} a_signed={a_signed} b_signed={b_signed}: pack-free and portable disagree"
                );
            }
        }
    }

    /// Overflow is not an edge case to avoid, it is behaviour to pin: the
    /// kernel is specified as wrapping `i32`, and re-ordering a wrapping sum
    /// cannot change it. A `k` this long with extremal operands wraps many
    /// times over.
    #[test]
    fn wrapping_overflow_is_reordering_invariant() {
        let (m, k, n) = (2usize, 40_960usize, 19usize);
        let a_bytes = vec![0xffu8; m * k];
        let b_bytes = vec![0x7fu8; k * n];
        let a = Operand {
            bytes: &a_bytes,
            signed: false,
        };
        let b = Operand {
            bytes: &b_bytes,
            signed: false,
        };
        let az = vec![0i32; m];
        let bz = vec![-128i32; n];
        let expect = oracle(a, b, &az, &bz, m, k, n);
        let mut got = vec![0i32; m * n];
        qgemm(a, b, &az, &bz, m, k, n, &mut got);
        assert_eq!(got, expect, "wrapping accumulation must survive blocking");
        assert!(
            expect.iter().any(|&value| value < 0),
            "the fixture must actually overflow, or it proves nothing"
        );
    }

    /// `products` is accumulated into, not overwritten: the caller owns the
    /// initial value and the batch loop relies on it being zero.
    #[test]
    fn the_result_is_accumulated_into_the_caller_s_buffer() {
        let (m, k, n) = (3usize, 40usize, 21usize);
        let a_bytes = bytes(m * k, 7);
        let b_bytes = bytes(k * n, 11);
        let a = Operand {
            bytes: &a_bytes,
            signed: false,
        };
        let b = Operand {
            bytes: &b_bytes,
            signed: false,
        };
        let az = vec![3i32; m];
        let bz = vec![9i32; n];
        let mut fresh = vec![0i32; m * n];
        qgemm(a, b, &az, &bz, m, k, n, &mut fresh);
        let mut seeded = vec![100i32; m * n];
        qgemm(a, b, &az, &bz, m, k, n, &mut seeded);
        for (seeded, fresh) in seeded.iter().zip(&fresh) {
            assert_eq!(*seeded, fresh.wrapping_add(100));
        }
    }

    /// Splitting across the pool must not move a bit. The column blocks are
    /// disjoint, so this is a claim about the code, not about floating point.
    #[test]
    fn the_thread_count_cannot_change_the_result() {
        let (m, k, n) = (6usize, 300usize, 1100usize);
        let a_bytes = bytes(m * k, 0x4242);
        let b_bytes = bytes(k * n, 0x2424);
        let a = Operand {
            bytes: &a_bytes,
            signed: true,
        };
        let b = Operand {
            bytes: &b_bytes,
            signed: false,
        };
        let az = zero_points(m, true, 5);
        let bz = zero_points(n, false, 6);
        let mut reference = vec![0i32; m * n];
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| qgemm(a, b, &az, &bz, m, k, n, &mut reference));
        for threads in [2usize, 3, 8] {
            let mut got = vec![0i32; m * n];
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| qgemm(a, b, &az, &bz, m, k, n, &mut got));
            assert_eq!(got, reference, "{threads} threads changed the result");
        }
    }

    /// `k == 0` contributes nothing, and `m == 0` / `n == 0` must not index.
    ///
    /// The zero-point slices are sized to `m` and `n` even in the cases that
    /// return immediately. [`qgemm`] documents them as one entry per row of `A`
    /// and per column of `B`, and asserts exactly that on entry; a caller whose
    /// `m` is zero still has `n` columns and still knows their zero points.
    /// Passing `&[]` for a non-zero `n` here tested the early return against a
    /// call the function does not accept — it passed only because
    /// `debug_assert` compiles out under `--release`, and failed the moment the
    /// same test ran in a debug profile.
    #[test]
    fn degenerate_extents_do_nothing() {
        let empty: Vec<u8> = Vec::new();
        let a = Operand {
            bytes: &empty,
            signed: false,
        };
        let mut out: Vec<i32> = Vec::new();
        qgemm(a, a, &[], &[0, 0, 0, 0], 0, 4, 4, &mut out);
        qgemm(a, a, &[1, 2], &[], 2, 4, 0, &mut out);
        let mut kept = vec![7i32; 6];
        qgemm(a, a, &[1, 2], &[1, 2, 3], 2, 0, 3, &mut kept);
        assert_eq!(kept, vec![7i32; 6], "k == 0 must leave the buffer alone");
    }

    /// Kernel-level A/B for the integer GEMM, with a built-in control.
    ///
    /// The session-level harness in `plugin_ort_e2e.rs` is the number that
    /// counts, but on a contended box its thread sweep is unusable: ORT's own
    /// pool spins while ours runs, so both sides move together and neither
    /// ratio means anything. This measures the kernel alone, which is what a
    /// blocking or threshold change actually moves.
    ///
    /// `portable` is the control arm. It is the same arithmetic with none of
    /// the blocking, so it must not move when a SIMD constant is retuned; if it
    /// does, the box was busy and the run says nothing.
    ///
    /// Run: `cargo test -p onnx-runtime-ep-cpu --lib --release bench_qgemm_ab \
    ///   -- --ignored --nocapture`. Knobs: `QGEMM_AB_ITERS`, `QGEMM_AB_THREADS`
    /// (comma-separated), `QGEMM_AB_SHAPES` (`mxkxn`, comma-separated).
    #[test]
    #[ignore = "manual perf harness; run explicitly"]
    fn bench_qgemm_ab() {
        use std::time::Instant;

        let parse = |name: &str, fallback: &str| -> Vec<String> {
            std::env::var(name)
                .ok()
                .filter(|spec| !spec.trim().is_empty())
                .unwrap_or_else(|| fallback.to_string())
                .split(',')
                .map(|piece| piece.trim().to_string())
                .collect()
        };
        let iters: usize = std::env::var("QGEMM_AB_ITERS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(21);
        let threads: Vec<usize> = parse("QGEMM_AB_THREADS", "1,2,4,8,16")
            .iter()
            .filter_map(|value| value.parse().ok())
            .collect();
        // Qwen2.5-14B-style quantized shapes: decode (m=1), a short prefill
        // (m=4, the last shape the pack-free kernel takes) and prefill (m=128).
        let shapes: Vec<(usize, usize, usize)> = parse(
            "QGEMM_AB_SHAPES",
            "1x2048x2048,1x5120x5120,4x2048x2048,128x2048x2048,128x5120x5120",
        )
        .iter()
        .filter_map(|spec| {
            let mut parts = spec.split('x').filter_map(|value| value.parse().ok());
            Some((parts.next()?, parts.next()?, parts.next()?))
        })
        .collect();

        println!("shape,threads,arm,p50_ms,gmacs");
        for &(m, k, n) in &shapes {
            let a_bytes = bytes(m * k, 0x1111_2222);
            let b_bytes = bytes(k * n, 0x3333_4444);
            let a = Operand {
                bytes: &a_bytes,
                signed: false,
            };
            let b = Operand {
                bytes: &b_bytes,
                signed: false,
            };
            let az = zero_points(m, false, 1);
            let bz = zero_points(n, false, 2);
            let mut products = vec![0i32; m * n];
            for &count in &threads {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(count)
                    .build()
                    .unwrap();
                for arm in ["native", "portable"] {
                    // The control is serial by construction and quadratically
                    // slower, so it is measured once per shape, not per thread.
                    if arm == "portable" && (count != threads[0] || m * k * n > 1 << 26) {
                        continue;
                    }
                    let mut samples = Vec::with_capacity(iters);
                    for _ in 0..iters {
                        products.iter_mut().for_each(|value| *value = 0);
                        let start = Instant::now();
                        match arm {
                            "native" => {
                                pool.install(|| qgemm(a, b, &az, &bz, m, k, n, &mut products))
                            }
                            _ => {
                                accumulate_products_scalar(a, b, &az, m, k, n, &mut products);
                            }
                        }
                        samples.push(start.elapsed().as_secs_f64() * 1e3);
                    }
                    samples.sort_by(f64::total_cmp);
                    let p50 = samples[samples.len() / 2];
                    let gmacs = (m * k * n) as f64 / (p50 * 1e-3) / 1e9;
                    println!("{m}x{k}x{n},{count},{arm},{p50:.4},{gmacs:.2}");
                }
            }
        }
    }
}
