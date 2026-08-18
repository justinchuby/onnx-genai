//! Runtime-dispatched blocked `f16`/`bf16` GEMM with `f32` accumulation.
//!
//! Operands stay in their 16-bit storage format until they are packed into
//! cache-sized `f32` panels. The shared register-tiled kernel then accumulates
//! in `f32`; callers perform one final narrowing step to the requested output
//! dtype. AVX2 widens bf16 directly and uses F16C when available for f16; NEON
//! widens bf16 directly and uses FP16 conversion when available. Both SIMD
//! paths share the same packed-panel structure and fall back to scalar widening
//! for unsupported layouts or conversion features. A scalar micro-kernel
//! remains the correctness path on every other CPU. When the host also has FMA
//! (the common AVX2-class case, including CI runners) the x86 inner loop uses a
//! fused `_mm256_fmadd_ps`; aarch64 uses baseline `vfmaq_f32`.

use rayon::prelude::*;

const MR: usize = 4;
const NR: usize = 8;
const KC: usize = 128;
const NC: usize = 64;
const MAX_MC: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExecutionPath {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86Avx2,
    /// AVX2 packing plus a fused-multiply-add (`_mm256_fmadd_ps`) inner loop.
    /// Selected only when the host also has FMA (the common AVX2-class case,
    /// including CI runners); halves the inner-loop op count versus the plain
    /// AVX2 mul+add microkernel on the prefill (M>1) hot path.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    X86Avx2Fma,
    #[cfg(target_arch = "aarch64")]
    Aarch64Neon,
}

/// Whether `path` uses the AVX2 widening/packing helpers (both the mul+add and
/// the FMA microkernels share identical F16C/AVX2 conversion).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn is_x86_avx2(path: ExecutionPath) -> bool {
    matches!(path, ExecutionPath::X86Avx2 | ExecutionPath::X86Avx2Fma)
}

#[inline]
fn selected_execution_path() -> ExecutionPath {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        if std::arch::is_x86_feature_detected!("fma") {
            return ExecutionPath::X86Avx2Fma;
        }
        return ExecutionPath::X86Avx2;
    }

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return ExecutionPath::Aarch64Neon;
    }

    ExecutionPath::Scalar
}

/// The 16-bit floating-point storage format used by a GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HalfFormat {
    F16,
    Bf16,
}

/// Logical matrix strides, measured in 16-bit elements.
///
/// A contiguous row-major matrix uses `(columns, 1)`. Swapping the two strides
/// provides a transposed logical view without materializing another buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatrixLayout {
    pub(crate) row_stride: usize,
    pub(crate) column_stride: usize,
}

impl MatrixLayout {
    pub(crate) const fn row_major(columns: usize) -> Self {
        Self {
            row_stride: columns,
            column_stride: 1,
        }
    }

    pub(crate) const fn transposed(stored_columns: usize) -> Self {
        Self {
            row_stride: 1,
            column_stride: stored_columns,
        }
    }
}

trait HalfElement: Send + Sync {
    fn to_f32(bits: u16) -> f32;
    fn pack_contiguous(source: &[u16], destination: &mut [f32], path: ExecutionPath);
}

struct F16;

impl HalfElement for F16 {
    #[inline]
    fn to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    #[inline]
    fn pack_contiguous(source: &[u16], destination: &mut [f32], path: ExecutionPath) {
        debug_assert_eq!(source.len(), destination.len());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_avx2(path) && std::arch::is_x86_feature_detected!("f16c") {
            // SAFETY: both features required by the conversion routine were
            // runtime-detected, and its slices have equal lengths.
            unsafe { widen_f16_x86(source, destination) };
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if path == ExecutionPath::Aarch64Neon && std::arch::is_aarch64_feature_detected!("fp16") {
            // SAFETY: NEON and FP16 were runtime-detected, and the slices have
            // equal lengths.
            unsafe { widen_f16_neon(source, destination) };
            return;
        }

        widen_scalar::<Self>(source, destination);
    }
}

struct Bf16;

impl HalfElement for Bf16 {
    #[inline]
    fn to_f32(bits: u16) -> f32 {
        half::bf16::from_bits(bits).to_f32()
    }

    #[inline]
    fn pack_contiguous(source: &[u16], destination: &mut [f32], path: ExecutionPath) {
        debug_assert_eq!(source.len(), destination.len());

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_avx2(path) {
            // SAFETY: AVX2 was runtime-detected, and the slices have equal
            // lengths.
            unsafe { widen_bf16_x86(source, destination) };
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if path == ExecutionPath::Aarch64Neon {
            // SAFETY: NEON was runtime-detected, and the slices have equal
            // lengths.
            unsafe { widen_bf16_neon(source, destination) };
            return;
        }

        widen_scalar::<Self>(source, destination);
    }
}

#[inline]
fn widen_scalar<T: HalfElement>(source: &[u16], destination: &mut [f32]) {
    for (output, &bits) in destination.iter_mut().zip(source) {
        *output = T::to_f32(bits);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,f16c")]
unsafe fn widen_f16_x86(source: &[u16], destination: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let vectorized_length = source.len() / 8 * 8;
    let mut index = 0;
    // SAFETY: the caller guarantees AVX2/F16C. Each vector iteration reads and
    // writes eight elements inside equal-length slices.
    unsafe {
        while index < vectorized_length {
            let packed = _mm_loadu_si128(source.as_ptr().add(index).cast());
            let wide = _mm256_cvtph_ps(packed);
            _mm256_storeu_ps(destination.as_mut_ptr().add(index), wide);
            index += 8;
        }
    }
    widen_scalar::<F16>(&source[index..], &mut destination[index..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn widen_bf16_x86(source: &[u16], destination: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let vectorized_length = source.len() / 8 * 8;
    let mut index = 0;
    // SAFETY: the caller guarantees AVX2. Each vector iteration reads and
    // writes eight elements inside equal-length slices.
    unsafe {
        while index < vectorized_length {
            let packed = _mm_loadu_si128(source.as_ptr().add(index).cast());
            let wide = _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16);
            _mm256_storeu_ps(
                destination.as_mut_ptr().add(index),
                _mm256_castsi256_ps(wide),
            );
            index += 8;
        }
    }
    widen_scalar::<Bf16>(&source[index..], &mut destination[index..]);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn widen_f16_neon(source: &[u16], destination: &mut [f32]) {
    use std::arch::aarch64::*;

    let vectorized_length = source.len() / 4 * 4;
    let mut index = 0;
    // SAFETY: the caller guarantees NEON/FP16. Each vector iteration reads and
    // writes four elements inside equal-length slices.
    unsafe {
        while index < vectorized_length {
            let packed = vld1_u16(source.as_ptr().add(index));
            let wide = vcvt_f32_f16(vreinterpret_f16_u16(packed));
            vst1q_f32(destination.as_mut_ptr().add(index), wide);
            index += 4;
        }
    }
    widen_scalar::<F16>(&source[index..], &mut destination[index..]);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn widen_bf16_neon(source: &[u16], destination: &mut [f32]) {
    use std::arch::aarch64::*;

    let vectorized_length = source.len() / 4 * 4;
    let mut index = 0;
    // SAFETY: the caller guarantees NEON. Each vector iteration reads and
    // writes four elements inside equal-length slices.
    unsafe {
        while index < vectorized_length {
            let packed = vld1_u16(source.as_ptr().add(index));
            let wide_bits = vshlq_n_u32::<16>(vmovl_u16(packed));
            vst1q_f32(
                destination.as_mut_ptr().add(index),
                vreinterpretq_f32_u32(wide_bits),
            );
            index += 4;
        }
    }
    widen_scalar::<Bf16>(&source[index..], &mut destination[index..]);
}

/// Compute `c[m,n] = a[m,k] @ b[k,n]` with `f32` accumulation.
///
/// `a` and `b` contain raw `f16` or `bf16` bit patterns selected by `format`.
/// Their logical layouts may be row-major or transposed. `c` is overwritten.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    format: HalfFormat,
    a: &[u16],
    a_layout: MatrixLayout,
    b: &[u16],
    b_layout: MatrixLayout,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    gemm_with_path(
        format,
        a,
        a_layout,
        b,
        b_layout,
        c,
        m,
        k,
        n,
        selected_execution_path(),
    );
}

#[allow(clippy::too_many_arguments)]
fn gemm_with_path(
    format: HalfFormat,
    a: &[u16],
    a_layout: MatrixLayout,
    b: &[u16],
    b_layout: MatrixLayout,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    path: ExecutionPath,
) {
    debug_assert_eq!(c.len(), m * n);
    match format {
        HalfFormat::F16 => gemm_impl::<F16>(a, a_layout, b, b_layout, c, m, k, n, path),
        HalfFormat::Bf16 => gemm_impl::<Bf16>(a, a_layout, b, b_layout, c, m, k, n, path),
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_impl<T: HalfElement>(
    a: &[u16],
    a_layout: MatrixLayout,
    b: &[u16],
    b_layout: MatrixLayout,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    path: ExecutionPath,
) {
    if m == 0 || n == 0 {
        return;
    }
    c.fill(0.0);

    let threads = rayon::current_num_threads();
    // Splitting is not free: the fork costs, and the workers keep spinning
    // after the region ends, which competes with whatever runs next. Below
    // `PARALLEL_MIN_WORK` that costs more than the split saves, so stay serial
    // -- the same guard `half_gemv` and `accelerate_gemm` already apply.
    let split_mc = {
        let rows = m.div_ceil(threads.saturating_mul(2)).clamp(1, MAX_MC);
        if rows == 1 {
            1
        } else {
            rows.div_ceil(MR).saturating_mul(MR).min(MAX_MC)
        }
    };
    let parallel = forced_route().unwrap_or_else(|| {
        threads > 1
            && m.saturating_mul(k).saturating_mul(n) >= PARALLEL_MIN_WORK
            // A split into a single block cannot use more than one thread, so
            // it is pure fork overhead. `m == 1` always lands here.
            && m.div_ceil(split_mc) > 1
    });
    let mc = if parallel { split_mc } else { MAX_MC.min(m) };

    let block = |block_index: usize, c_block: &mut [f32]| {
        let first_row = block_index * mc;
        let rows = c_block.len() / n;
        gemm_block::<T>(
            a, a_layout, b, b_layout, c_block, first_row, rows, k, n, path,
        );
    };

    if !parallel {
        count_serial_gemm();
        for (block_index, c_block) in c.chunks_mut(mc * n).enumerate() {
            block(block_index, c_block);
        }
        return;
    }

    c.par_chunks_mut(mc * n)
        .enumerate()
        .for_each(|(block_index, c_block)| block(block_index, c_block));
}

/// Minimum multiply-accumulate count (`m*k*n`) before splitting a half GEMM
/// across the pool.
///
/// `gemm_impl` used to fork unconditionally, so it spent 0.14 ms of pool
/// overhead to multiply an 8x64 by a 64x64 -- work a single core finishes in
/// 0.046 ms. That is the common case now: since the f16 prefill gate landed,
/// this kernel only ever sees the *small* shapes that decline widening, so the
/// unguarded fork was mis-sized for every shape it still serves.
///
/// Measured `serial/parallel` (>1 = splitting wins) at `m = 8`, interleaved
/// rep-by-rep, `p50` of 9, pinned to 16 physical cores, two independent runs
/// per thread count (`bench_half_gemm_parallel_threshold`):
///
/// | m*k*n     |       T=2 |       T=4 |       T=8 |      T=16 |      T=32 |
/// |-----------|-----------|-----------|-----------|-----------|-----------|
/// |    32_768 | 0.88/0.86 | 0.51/0.55 | 0.63/0.75 | 0.36/0.31 | 0.21/0.19 |
/// |    65_536 | 0.79/0.78 | 0.63/0.63 | 0.85/0.88 | 0.53/0.53 | 0.41/0.33 |
/// |   131_072 | 1.02/1.02 | 0.79/0.80 | 1.14/1.18 | 0.76/0.60 | 0.63/0.54 |
/// |   262_144 | 1.20/1.20 | 0.92/0.92 | 1.16/1.51 | 1.03/1.02 | 0.86/0.80 |
/// |   393_216 | 1.27/1.28 | 0.91/0.97 | 1.27/1.64 | 1.04/1.14 | 0.99/1.04 |
/// |   524_288 | 1.34/1.30 | 1.01/1.02 | 1.29/1.76 | 1.20/1.08 | 1.19/1.12 |
/// | 1_048_576 | 1.41/1.37 | 1.08/1.04 | 1.47/1.95 | 1.34/1.30 | 1.31/1.31 |
///
/// Re-measured after `codegen-units = 1` was pinned for this crate, which makes
/// the *serial* route materially faster and therefore moves the crossover up.
/// Same harness, two runs per thread count:
///
/// | m*k*n     |       T=2 |       T=4 |       T=8 |      T=16 |
/// |-----------|-----------|-----------|-----------|-----------|
/// |   262_144 |      1.20 | 0.92/0.91 | 1.52      | 0.64/0.95 |
/// |   393_216 |      1.28 | 0.99/0.96 | 1.66      | 0.80/1.10 |
/// |   524_288 |      1.32 | 0.99/0.98 | 1.81      | 1.03/1.19 |
/// |   786_432 |      1.33 | 1.03/0.92 | 1.35      | 0.97/1.25 |
/// | 1_048_576 |      1.37 | 1.05/1.05 | 1.46      | 1.31/1.34 |
///
/// `1_048_576` is now the smallest size that wins at every measured thread count
/// in every run (worst case 1.05x). `524_288` no longer qualifies: it is a wash
/// at T=4 (0.99/0.98) and `786_432` dips to 0.92 there and 0.97 at T=16. The
/// rule is unchanged -- smallest size that wins everywhere -- only the build it
/// is measured against is faster, so the answer moved.
///
/// Below the threshold the picture is not merely mixed but bad: 0.32-0.37 at
/// T=16 for the smallest shapes, which is the 3x the guard exists to stop.
///
/// Deliberately a single constant rather than a function of thread count: the
/// measured crossover is *not* monotone in threads (T=4 wants the highest
/// threshold, T=8 the lowest), because for small `m` the block count is pinned
/// by `m` rather than by the pool, so a thread-scaled formula would be fitting
/// noise.
///
/// `m*k*n` is a proxy, not a cost model. The split is over *rows of C*, so the
/// achievable speedup is bounded by the block count, and the parallel route
/// re-packs `B` per row-block -- for small `m` (`mc == 1`) that is `m`
/// redundant packs. That is why the ratios plateau near 1.1x at T=4 rather than
/// scaling with the pool, and why some shapes above the threshold still gain
/// little. Sibling kernels apply the same kind of guard
/// (`half_gemv::PARALLEL_MIN_WORK`, `accelerate_gemm`'s `k*n` bound).
const PARALLEL_MIN_WORK: usize = 1_048_576;

#[cfg(test)]
thread_local! {
    /// Test-only override letting a benchmark time both routes in one process.
    static FORCED_ROUTE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[inline]
fn forced_route() -> Option<bool> {
    #[cfg(test)]
    {
        FORCED_ROUTE.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        None
    }
}

#[cfg(test)]
fn with_forced_route<R>(parallel: bool, op: impl FnOnce() -> R) -> R {
    FORCED_ROUTE.with(|c| c.set(Some(parallel)));
    let out = op();
    FORCED_ROUTE.with(|c| c.set(None));
    out
}

#[cfg(test)]
thread_local! {
    /// Counts half GEMMs that declined to split, so a test can prove the guard
    /// fired rather than inferring it from a timing.
    static SERIAL_GEMMS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline]
fn count_serial_gemm() {
    #[cfg(test)]
    SERIAL_GEMMS.with(|c| c.set(c.get() + 1));
}

/// Number of unsplit half GEMMs on this thread since the last reset.
#[cfg(test)]
fn serial_gemms() -> usize {
    SERIAL_GEMMS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn reset_serial_gemms() {
    SERIAL_GEMMS.with(|c| c.set(0));
}

#[allow(clippy::too_many_arguments)]
fn gemm_block<T: HalfElement>(
    a: &[u16],
    a_layout: MatrixLayout,
    b: &[u16],
    b_layout: MatrixLayout,
    c: &mut [f32],
    first_row: usize,
    rows: usize,
    k: usize,
    n: usize,
    path: ExecutionPath,
) {
    let mut a_panel = Vec::with_capacity(rows * KC);
    let mut b_panel = Vec::with_capacity(KC * NC);

    for depth_start in (0..k).step_by(KC) {
        let panel_depth = KC.min(k - depth_start);
        pack_a::<T>(
            a,
            a_layout,
            first_row,
            rows,
            depth_start,
            panel_depth,
            &mut a_panel,
            path,
        );

        for column_start in (0..n).step_by(NC) {
            let panel_columns = NC.min(n - column_start);
            pack_b::<T>(
                b,
                b_layout,
                depth_start,
                panel_depth,
                column_start,
                panel_columns,
                &mut b_panel,
                path,
            );

            for row_start in (0..rows).step_by(MR) {
                let tile_rows = MR.min(rows - row_start);
                for panel_column_start in (0..panel_columns).step_by(NR) {
                    let tile_columns = NR.min(panel_columns - panel_column_start);
                    micro_kernel(
                        path,
                        &a_panel,
                        &b_panel,
                        c,
                        n,
                        row_start,
                        panel_column_start,
                        column_start,
                        panel_depth,
                        panel_columns,
                        tile_rows,
                        tile_columns,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_a<T: HalfElement>(
    source: &[u16],
    layout: MatrixLayout,
    first_row: usize,
    rows: usize,
    depth_start: usize,
    panel_depth: usize,
    packed: &mut Vec<f32>,
    path: ExecutionPath,
) {
    packed.clear();
    packed.resize(rows * panel_depth, 0.0);
    for row in 0..rows {
        let destination = &mut packed[row * panel_depth..(row + 1) * panel_depth];
        if layout.column_stride == 1 {
            let source_start = (first_row + row) * layout.row_stride + depth_start;
            T::pack_contiguous(
                &source[source_start..source_start + panel_depth],
                destination,
                path,
            );
        } else {
            for (depth, output) in destination.iter_mut().enumerate() {
                let source_index = (first_row + row) * layout.row_stride
                    + (depth_start + depth) * layout.column_stride;
                *output = T::to_f32(source[source_index]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_b<T: HalfElement>(
    source: &[u16],
    layout: MatrixLayout,
    depth_start: usize,
    panel_depth: usize,
    column_start: usize,
    panel_columns: usize,
    packed: &mut Vec<f32>,
    path: ExecutionPath,
) {
    packed.clear();
    packed.resize(panel_depth * panel_columns, 0.0);
    for depth in 0..panel_depth {
        let destination = &mut packed[depth * panel_columns..(depth + 1) * panel_columns];
        if layout.column_stride == 1 {
            let source_start = (depth_start + depth) * layout.row_stride + column_start;
            T::pack_contiguous(
                &source[source_start..source_start + panel_columns],
                destination,
                path,
            );
        } else {
            for (column, output) in destination.iter_mut().enumerate() {
                let source_index = (depth_start + depth) * layout.row_stride
                    + (column_start + column) * layout.column_stride;
                *output = T::to_f32(source[source_index]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn micro_kernel(
    path: ExecutionPath,
    a_panel: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    c_columns: usize,
    row_start: usize,
    panel_column_start: usize,
    column_start: usize,
    panel_depth: usize,
    panel_columns: usize,
    tile_rows: usize,
    tile_columns: usize,
) {
    if tile_columns == NR {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if path == ExecutionPath::X86Avx2Fma {
            // SAFETY: AVX2+FMA were runtime-detected when the path was selected,
            // and the tile/slice bounds are established by the blocked driver.
            unsafe {
                micro_kernel_avx2_fma(
                    a_panel,
                    b_panel,
                    c,
                    c_columns,
                    row_start,
                    panel_column_start,
                    column_start,
                    panel_depth,
                    panel_columns,
                    tile_rows,
                )
            };
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if path == ExecutionPath::X86Avx2 {
            // SAFETY: AVX2 was runtime-detected when the path was selected, and
            // the tile/slice bounds are established by the blocked driver.
            unsafe {
                micro_kernel_avx2(
                    a_panel,
                    b_panel,
                    c,
                    c_columns,
                    row_start,
                    panel_column_start,
                    column_start,
                    panel_depth,
                    panel_columns,
                    tile_rows,
                )
            };
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if path == ExecutionPath::Aarch64Neon {
            // SAFETY: NEON was runtime-detected when the path was selected, and
            // the tile/slice bounds are established by the blocked driver.
            unsafe {
                micro_kernel_neon(
                    a_panel,
                    b_panel,
                    c,
                    c_columns,
                    row_start,
                    panel_column_start,
                    column_start,
                    panel_depth,
                    panel_columns,
                    tile_rows,
                )
            };
            return;
        }
    }

    micro_kernel_scalar(
        a_panel,
        b_panel,
        c,
        c_columns,
        row_start,
        panel_column_start,
        column_start,
        panel_depth,
        panel_columns,
        tile_rows,
        tile_columns,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn micro_kernel_scalar(
    a_panel: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    c_columns: usize,
    row_start: usize,
    panel_column_start: usize,
    column_start: usize,
    panel_depth: usize,
    panel_columns: usize,
    tile_rows: usize,
    tile_columns: usize,
) {
    let mut accumulators = [[0.0f32; NR]; MR];
    for depth in 0..panel_depth {
        let b_row = &b_panel[depth * panel_columns + panel_column_start
            ..depth * panel_columns + panel_column_start + tile_columns];
        for (tile_row, accumulator_row) in accumulators.iter_mut().enumerate().take(tile_rows) {
            let a_value = a_panel[tile_row.saturating_add(row_start) * panel_depth + depth];
            for (tile_column, accumulator) in
                accumulator_row.iter_mut().enumerate().take(tile_columns)
            {
                *accumulator += a_value * b_row[tile_column];
            }
        }
    }

    for (tile_row, accumulator_row) in accumulators.iter().enumerate().take(tile_rows) {
        let output_start = (row_start + tile_row) * c_columns + column_start + panel_column_start;
        let output_row = &mut c[output_start..output_start + tile_columns];
        for (output, accumulator) in output_row.iter_mut().zip(accumulator_row) {
            *output += accumulator;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_kernel_avx2(
    a_panel: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    c_columns: usize,
    row_start: usize,
    panel_column_start: usize,
    column_start: usize,
    panel_depth: usize,
    panel_columns: usize,
    tile_rows: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // SAFETY: the caller guarantees AVX2 and a full NR-wide tile. The driver
    // establishes all A/B/C bounds.
    unsafe {
        let mut accumulators = [_mm256_setzero_ps(); MR];
        for depth in 0..panel_depth {
            let b_row = _mm256_loadu_ps(
                b_panel
                    .as_ptr()
                    .add(depth * panel_columns + panel_column_start),
            );
            for (tile_row, accumulator) in accumulators.iter_mut().enumerate().take(tile_rows) {
                let a_value = a_panel[(row_start + tile_row) * panel_depth + depth];
                *accumulator =
                    _mm256_add_ps(*accumulator, _mm256_mul_ps(_mm256_set1_ps(a_value), b_row));
            }
        }

        for (tile_row, accumulator) in accumulators.iter().enumerate().take(tile_rows) {
            let output_start =
                (row_start + tile_row) * c_columns + column_start + panel_column_start;
            let output = c.as_mut_ptr().add(output_start);
            let previous = _mm256_loadu_ps(output);
            _mm256_storeu_ps(output, _mm256_add_ps(previous, *accumulator));
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_kernel_avx2_fma(
    a_panel: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    c_columns: usize,
    row_start: usize,
    panel_column_start: usize,
    column_start: usize,
    panel_depth: usize,
    panel_columns: usize,
    tile_rows: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // SAFETY: the caller guarantees AVX2+FMA and a full NR-wide tile. The driver
    // establishes all A/B/C bounds.
    unsafe {
        let mut accumulators = [_mm256_setzero_ps(); MR];
        for depth in 0..panel_depth {
            let b_row = _mm256_loadu_ps(
                b_panel
                    .as_ptr()
                    .add(depth * panel_columns + panel_column_start),
            );
            for (tile_row, accumulator) in accumulators.iter_mut().enumerate().take(tile_rows) {
                let a_value = a_panel[(row_start + tile_row) * panel_depth + depth];
                // Fused multiply-add: `acc = a*b + acc` in one instruction,
                // halving the inner-loop op count versus separate mul + add.
                *accumulator = _mm256_fmadd_ps(_mm256_set1_ps(a_value), b_row, *accumulator);
            }
        }

        for (tile_row, accumulator) in accumulators.iter().enumerate().take(tile_rows) {
            let output_start =
                (row_start + tile_row) * c_columns + column_start + panel_column_start;
            let output = c.as_mut_ptr().add(output_start);
            let previous = _mm256_loadu_ps(output);
            _mm256_storeu_ps(output, _mm256_add_ps(previous, *accumulator));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_kernel_neon(
    a_panel: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    c_columns: usize,
    row_start: usize,
    panel_column_start: usize,
    column_start: usize,
    panel_depth: usize,
    panel_columns: usize,
    tile_rows: usize,
) {
    use std::arch::aarch64::*;

    // SAFETY: the caller guarantees NEON and a full NR-wide tile. The driver
    // establishes all A/B/C bounds.
    unsafe {
        let mut accumulator_low = [vdupq_n_f32(0.0); MR];
        let mut accumulator_high = [vdupq_n_f32(0.0); MR];
        for depth in 0..panel_depth {
            let b_row = b_panel
                .as_ptr()
                .add(depth * panel_columns + panel_column_start);
            let b_low = vld1q_f32(b_row);
            let b_high = vld1q_f32(b_row.add(4));
            for tile_row in 0..tile_rows {
                let a_value = a_panel[(row_start + tile_row) * panel_depth + depth];
                let a_vector = vdupq_n_f32(a_value);
                // Fused multiply-add (`vfmaq_f32` is baseline aarch64 NEON):
                // `acc = a*b + acc` in one instruction.
                accumulator_low[tile_row] = vfmaq_f32(accumulator_low[tile_row], a_vector, b_low);
                accumulator_high[tile_row] =
                    vfmaq_f32(accumulator_high[tile_row], a_vector, b_high);
            }
        }

        for tile_row in 0..tile_rows {
            let output_start =
                (row_start + tile_row) * c_columns + column_start + panel_column_start;
            let output = c.as_mut_ptr().add(output_start);
            vst1q_f32(
                output,
                vaddq_f32(vld1q_f32(output), accumulator_low[tile_row]),
            );
            vst1q_f32(
                output.add(4),
                vaddq_f32(vld1q_f32(output.add(4)), accumulator_high[tile_row]),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0; m * n];
        for row in 0..m {
            for column in 0..n {
                for depth in 0..k {
                    output[row * n + column] += a[row * k + depth] * b[depth * n + column];
                }
            }
        }
        output
    }

    /// Ignored perf probe: single-thread f16 GEMM at a representative prefill
    /// shape, comparing the plain AVX2 (mul+add) microkernel against the AVX2
    /// FMA microkernel. Iterations are interleaved so shared-machine load drift
    /// cancels out; the reported speedup is load-independent. Run pinned:
    /// `taskset -c 1 cargo test -p onnx-runtime-ep-cpu --release half_gemm_prefill_gflops -- --ignored --nocapture`
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[ignore = "perf probe; run manually pinned to one core"]
    fn half_gemm_prefill_gflops() {
        use std::time::Instant;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool");
        let (m, k, n) = (512usize, 4096usize, 4096usize);
        let format = HalfFormat::F16;
        let a: Vec<u16> = (0..m * k)
            .map(|i| half_bits(format, ((i as f32 * 0.017).sin()) * 0.25))
            .collect();
        let b: Vec<u16> = (0..k * n)
            .map(|i| half_bits(format, ((i as f32 * 0.013).cos()) * 0.25))
            .collect();
        let flop = 2.0 * m as f64 * k as f64 * n as f64;

        let paths: &[(&str, ExecutionPath)] = &[
            ("avx2 (mul+add)", ExecutionPath::X86Avx2),
            ("avx2+fma", ExecutionPath::X86Avx2Fma),
        ];

        pool.install(|| {
            let mut c = vec![0.0f32; m * n];
            let run = |path: ExecutionPath, c: &mut [f32]| {
                gemm_with_path(
                    format,
                    &a,
                    MatrixLayout::row_major(k),
                    &b,
                    MatrixLayout::row_major(n),
                    c,
                    m,
                    k,
                    n,
                    path,
                );
            };
            // Warm up both.
            for &(_, path) in paths {
                run(path, &mut c);
            }
            let mut times = vec![Vec::new(); paths.len()];
            for _ in 0..7 {
                for (slot, &(_, path)) in paths.iter().enumerate() {
                    let start = Instant::now();
                    run(path, &mut c);
                    times[slot].push(start.elapsed().as_secs_f64());
                }
            }
            let mut medians = Vec::new();
            for (slot, &(label, _)) in paths.iter().enumerate() {
                times[slot].sort_by(|x, y| x.partial_cmp(y).unwrap());
                let median = times[slot][times[slot].len() / 2];
                medians.push(median);
                println!(
                    "half_gemm f16 {m}x{k}x{n} 1-thread {label}: median {:.3} ms, {:.2} GFLOPS",
                    median * 1e3,
                    flop / median / 1e9,
                );
            }
            println!(
                "half_gemm f16 FMA speedup vs mul+add: {:.2}x",
                medians[0] / medians[1]
            );
        });
    }

    /// All hardware-accelerated paths available on this host, most-preferred
    /// first. Each is validated against the scalar path so both the plain AVX2
    /// (mul+add) and AVX2+FMA microkernels stay honest on machines that have
    /// FMA, while CI without FMA still exercises the plain AVX2 path.
    fn detected_simd_paths() -> Vec<ExecutionPath> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let mut paths = Vec::new();
            if std::arch::is_x86_feature_detected!("avx2") {
                paths.push(ExecutionPath::X86Avx2);
                if std::arch::is_x86_feature_detected!("fma") {
                    paths.push(ExecutionPath::X86Avx2Fma);
                }
            }
            paths
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                vec![ExecutionPath::Aarch64Neon]
            } else {
                Vec::new()
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Vec::new()
        }
    }

    fn half_bits(format: HalfFormat, value: f32) -> u16 {
        match format {
            HalfFormat::F16 => half::f16::from_f32(value).to_bits(),
            HalfFormat::Bf16 => half::bf16::from_f32(value).to_bits(),
        }
    }

    #[test]
    fn runtime_simd_half_gemm_matches_scalar_for_square_skinny_and_tail_shapes() {
        let simd_paths = detected_simd_paths();
        if simd_paths.is_empty() {
            return;
        }
        // The auto-selected path must be one of the accelerated paths we test.
        assert!(simd_paths.contains(&selected_execution_path()));

        const SHAPES: &[(usize, usize, usize)] = &[
            (8, 16, 16),
            (1, 33, 17),
            (3, 129, 13),
            (7, 15, 24),
            (9, 257, 70),
        ];

        for &simd_path in &simd_paths {
            for format in [HalfFormat::F16, HalfFormat::Bf16] {
                for &(m, k, n) in SHAPES {
                    let a: Vec<u16> = (0..m * k)
                        .map(|index| {
                            half_bits(format, ((index as f32 * 0.071 + 0.13).sin()) * 0.375)
                        })
                        .collect();
                    let b: Vec<u16> = (0..k * n)
                        .map(|index| {
                            half_bits(format, ((index as f32 * 0.053 - 0.29).cos()) * 0.375)
                        })
                        .collect();
                    let mut scalar = vec![0.0; m * n];
                    let mut simd = vec![0.0; m * n];

                    gemm_with_path(
                        format,
                        &a,
                        MatrixLayout::row_major(k),
                        &b,
                        MatrixLayout::row_major(n),
                        &mut scalar,
                        m,
                        k,
                        n,
                        ExecutionPath::Scalar,
                    );
                    gemm_with_path(
                        format,
                        &a,
                        MatrixLayout::row_major(k),
                        &b,
                        MatrixLayout::row_major(n),
                        &mut simd,
                        m,
                        k,
                        n,
                        simd_path,
                    );

                    let max_error = scalar
                        .iter()
                        .zip(&simd)
                        .map(|(scalar, simd)| (scalar - simd).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_error <= 1e-6,
                        "{format:?} {simd_path:?} {m}x{k}x{n} differs from scalar by {max_error}"
                    );
                }
            }
        }
    }

    #[test]
    fn portable_half_gemm_matches_widened_f32_reference_and_is_deterministic() {
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 127, 65),
            (3, 5, 7),
            (17, 130, 11),
            (5, 257, 70),
            (33, 9, 2),
        ];
        let mut state = 0xC001_D00D_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.5
        };

        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            for &(m, k, n) in SHAPES {
                let a_source: Vec<f32> = (0..m * k).map(|_| next()).collect();
                let b_source: Vec<f32> = (0..k * n).map(|_| next()).collect();
                let (a_bits, a_wide): (Vec<_>, Vec<_>) = a_source
                    .iter()
                    .map(|&value| match format {
                        HalfFormat::F16 => {
                            let value = half::f16::from_f32(value);
                            (value.to_bits(), value.to_f32())
                        }
                        HalfFormat::Bf16 => {
                            let value = half::bf16::from_f32(value);
                            (value.to_bits(), value.to_f32())
                        }
                    })
                    .unzip();
                let (b_bits, b_wide): (Vec<_>, Vec<_>) = b_source
                    .iter()
                    .map(|&value| match format {
                        HalfFormat::F16 => {
                            let value = half::f16::from_f32(value);
                            (value.to_bits(), value.to_f32())
                        }
                        HalfFormat::Bf16 => {
                            let value = half::bf16::from_f32(value);
                            (value.to_bits(), value.to_f32())
                        }
                    })
                    .unzip();
                let expected = reference(&a_wide, &b_wide, m, k, n);
                let mut first = vec![0.0; m * n];
                let mut second = vec![0.0; m * n];
                gemm(
                    format,
                    &a_bits,
                    MatrixLayout::row_major(k),
                    &b_bits,
                    MatrixLayout::row_major(n),
                    &mut first,
                    m,
                    k,
                    n,
                );
                gemm(
                    format,
                    &a_bits,
                    MatrixLayout::row_major(k),
                    &b_bits,
                    MatrixLayout::row_major(n),
                    &mut second,
                    m,
                    k,
                    n,
                );

                assert_eq!(
                    first, second,
                    "{format:?} {m}x{k}x{n} was not deterministic"
                );
                let max_error = first
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_error <= 2e-5,
                    "{format:?} {m}x{k}x{n}: max f32 accumulation error {max_error}"
                );
            }
        }
    }

    #[test]
    fn portable_half_gemm_supports_transposed_layouts() {
        let logical_a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let logical_b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let stored_a = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let stored_b = [7.0f32, 9.0, 11.0, 8.0, 10.0, 12.0];
        let expected = reference(&logical_a, &logical_b, 2, 3, 2);

        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            let convert = |value: f32| match format {
                HalfFormat::F16 => half::f16::from_f32(value).to_bits(),
                HalfFormat::Bf16 => half::bf16::from_f32(value).to_bits(),
            };
            let a: Vec<u16> = stored_a.iter().copied().map(convert).collect();
            let b: Vec<u16> = stored_b.iter().copied().map(convert).collect();
            let mut actual = vec![0.0; 4];
            gemm(
                format,
                &a,
                MatrixLayout::transposed(2),
                &b,
                MatrixLayout::transposed(3),
                &mut actual,
                2,
                3,
                2,
            );
            assert_eq!(actual, expected);
        }
    }
}

#[cfg(test)]
mod half_gemm_par_bench {
    use super::*;
    use std::time::Instant;

    fn operands(m: usize, k: usize, n: usize) -> (Vec<u16>, Vec<u16>, Vec<f32>) {
        let a = (0..m * k)
            .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.1).to_bits())
            .collect();
        let b = (0..k * n)
            .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.1).to_bits())
            .collect();
        (a, b, vec![0.0f32; m * n])
    }

    fn run(a: &[u16], b: &[u16], c: &mut [f32], m: usize, k: usize, n: usize, parallel: bool) {
        with_forced_route(parallel, || {
            gemm_impl::<F16>(
                a,
                MatrixLayout::row_major(k),
                b,
                MatrixLayout::row_major(n),
                c,
                m,
                k,
                n,
                ExecutionPath::Scalar,
            )
        });
    }

    /// Sites [`PARALLEL_MIN_WORK`]: serial vs pool-split half GEMM, interleaved
    /// rep-by-rep so both routes see the same machine load. Pin the run
    /// (`taskset -c 0-15`) and set `RAYON_NUM_THREADS`.
    #[test]
    #[ignore = "microbench: run explicitly with --ignored --nocapture"]
    fn bench_half_gemm_parallel_threshold() {
        println!("threads={}", rayon::current_num_threads());
        println!("m,k,n,work,serial_ms,par_ms,serial/par");
        for &(m, k, n) in &[
            (8usize, 64usize, 64usize), //    32_768
            (8, 128, 64),               //    65_536
            (8, 128, 128),              //   131_072
            (8, 192, 128),              //   196_608
            (8, 256, 128),              //   262_144
            (8, 256, 192),              //   393_216
            (8, 256, 256),              //   524_288
            (8, 384, 256),              //   786_432
            (8, 512, 256),              // 1_048_576
            (8, 512, 384),              // 1_572_864
            (8, 512, 512),              // 2_097_152
            (8, 768, 512),              // 3_145_728
        ] {
            let (a, b, mut c) = operands(m, k, n);
            run(&a, &b, &mut c, m, k, n, false);
            let expected = c.clone();
            run(&a, &b, &mut c, m, k, n, true);
            assert_eq!(c, expected, "{m}x{k}x{n}: split changed the result");

            let (mut sv, mut pv) = (Vec::new(), Vec::new());
            for _ in 0..9 {
                let t = Instant::now();
                run(&a, &b, &mut c, m, k, n, false);
                sv.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                run(&a, &b, &mut c, m, k, n, true);
                pv.push(t.elapsed().as_secs_f64() * 1e3);
            }
            sv.sort_by(f64::total_cmp);
            pv.sort_by(f64::total_cmp);
            println!(
                "{m},{k},{n},{},{:.4},{:.4},{:.2}",
                m * k * n,
                sv[4],
                pv[4],
                sv[4] / pv[4]
            );
        }
    }
}

#[cfg(test)]
mod half_gemm_guard_tests {
    use super::*;

    fn operands(m: usize, k: usize, n: usize) -> (Vec<u16>, Vec<u16>, Vec<f32>) {
        let a = (0..m * k)
            .map(|i| half::f16::from_f32(((i % 23) as f32 - 11.0) * 0.07).to_bits())
            .collect();
        let b = (0..k * n)
            .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.05).to_bits())
            .collect();
        (a, b, vec![0.0f32; m * n])
    }

    fn gemm_as<T: HalfElement>(a: &[u16], b: &[u16], c: &mut [f32], m: usize, k: usize, n: usize) {
        gemm_impl::<T>(
            a,
            MatrixLayout::row_major(k),
            b,
            MatrixLayout::row_major(n),
            c,
            m,
            k,
            n,
            ExecutionPath::Scalar,
        );
    }

    fn gemm(a: &[u16], b: &[u16], c: &mut [f32], m: usize, k: usize, n: usize) {
        gemm_impl::<F16>(
            a,
            MatrixLayout::row_major(k),
            b,
            MatrixLayout::row_major(n),
            c,
            m,
            k,
            n,
            ExecutionPath::Scalar,
        );
    }

    /// The threshold is a measured crossover, not a guess -- an earlier guess
    /// sat 8x too low, inside the range the sweep shows losing 0.3x-0.9x. Pin
    /// the value so a silent edit has to come back through the sweep, which
    /// spans five thread counts precisely because the crossover moves with the
    /// pool.
    #[test]
    fn half_gemm_parallel_threshold_matches_the_measured_crossover() {
        assert_eq!(
            PARALLEL_MIN_WORK, 1_048_576,
            "smallest m*k*n measured to win at every thread count; re-run \
             bench_half_gemm_parallel_threshold before changing it"
        );
    }

    /// Falsifies the guard itself: work below the crossover must stay on one
    /// thread, and work above it must still split. Without the first half, an
    /// 8x64x64 GEMM forks and takes 3.1x longer than doing it serially.
    #[test]
    fn half_gemm_declines_to_split_work_below_the_crossover() {
        if rayon::current_num_threads() < 2 {
            eprintln!("skipping: needs a multi-thread pool");
            return;
        }
        // 8*64*64 = 32_768, the shape measured at 0.32x when split.
        let (a, b, mut c) = operands(8, 64, 64);
        reset_serial_gemms();
        gemm(&a, &b, &mut c, 8, 64, 64);
        assert_eq!(
            serial_gemms(),
            1,
            "work below PARALLEL_MIN_WORK must not fork the pool"
        );

        // 8*512*384 = 1_572_864, measured 1.33x-1.51x in favour of splitting.
        let (a, b, mut c) = operands(8, 512, 384);
        reset_serial_gemms();
        gemm(&a, &b, &mut c, 8, 512, 384);
        assert_eq!(
            serial_gemms(),
            0,
            "work above PARALLEL_MIN_WORK must still be split"
        );
    }

    /// Pins the comparison at *exactly* the threshold. Without this, swapping
    /// `>=` for `>` in the guard changes behaviour at the boundary and every
    /// other test still passes.
    #[test]
    fn the_crossover_itself_splits_and_one_mac_below_it_does_not() {
        if rayon::current_num_threads() < 2 {
            eprintln!("skipping: needs a multi-thread pool");
            return;
        }
        // 8 * 512 * 256 == 1_048_576 == PARALLEL_MIN_WORK exactly.
        let (m, k, n) = (8usize, 512usize, 256usize);
        assert_eq!(
            m * k * n,
            PARALLEL_MIN_WORK,
            "shape must sit on the boundary"
        );
        let (a, b, mut c) = operands(m, k, n);
        reset_serial_gemms();
        gemm(&a, &b, &mut c, m, k, n);
        assert_eq!(
            serial_gemms(),
            0,
            "work equal to PARALLEL_MIN_WORK must split (the bound is inclusive)"
        );

        // One MAC below the threshold: k*n is one column short.
        let n1 = n - 1;
        assert!(m * k * n1 < PARALLEL_MIN_WORK);
        let (a, b, mut c) = operands(m, k, n1);
        reset_serial_gemms();
        gemm(&a, &b, &mut c, m, k, n1);
        assert_eq!(
            serial_gemms(),
            1,
            "work below PARALLEL_MIN_WORK must stay serial"
        );
    }

    /// A split into one block cannot use more than one thread, so forking for
    /// it is pure overhead however large the operand is. `m == 1` is the case
    /// that matters: a 1x2048 by 2048x2048 GEMV is 4.2M MACs, far above the
    /// threshold, yet yields exactly one block.
    #[test]
    fn a_single_block_is_never_split_however_large_the_operand() {
        if rayon::current_num_threads() < 2 {
            eprintln!("skipping: needs a multi-thread pool");
            return;
        }
        let (m, k, n) = (1usize, 1024usize, 2048usize);
        assert!(m * k * n > PARALLEL_MIN_WORK, "must clear the work bound");
        let (a, b, mut c) = operands(m, k, n);
        reset_serial_gemms();
        gemm(&a, &b, &mut c, m, k, n);
        assert_eq!(
            serial_gemms(),
            1,
            "a one-block split is pure fork overhead and must be declined"
        );
    }

    /// The routing is element-type agnostic, so bf16 must get the same
    /// bit-for-bit guarantee as f16 rather than inheriting it by assumption.
    #[test]
    fn bf16_routes_agree_bit_for_bit() {
        for &(m, k, n) in &[(8usize, 64usize, 64usize), (13, 129, 67), (130, 65, 67)] {
            let a: Vec<u16> = (0..m * k)
                .map(|i| half::bf16::from_f32(((i % 23) as f32 - 11.0) * 0.07).to_bits())
                .collect();
            let b: Vec<u16> = (0..k * n)
                .map(|i| half::bf16::from_f32(((i % 17) as f32 - 8.0) * 0.05).to_bits())
                .collect();
            let mut c = vec![0.0f32; m * n];
            with_forced_route(false, || gemm_as::<Bf16>(&a, &b, &mut c, m, k, n));
            let serial = c.clone();
            with_forced_route(true, || gemm_as::<Bf16>(&a, &b, &mut c, m, k, n));
            let sb: Vec<u32> = serial.iter().map(|v| v.to_bits()).collect();
            let pb: Vec<u32> = c.iter().map(|v| v.to_bits()).collect();
            assert!(
                sb == pb,
                "{m}x{k}x{n}: split and serial bf16 half GEMM disagree bit-for-bit"
            );
        }
    }

    /// The guard changes *scheduling*, never arithmetic: both routes must agree
    /// bit-for-bit, including on shapes whose row count is not a multiple of
    /// the register-block height so the final block is a tail.
    ///
    /// Deliberately spans `m > MAX_MC`: the serial route blocks at
    /// `MAX_MC.min(m)`, so every shape with `m <= 64` yields a *single* block
    /// and never exercises the loop's block indexing. Without a `m > 64` shape
    /// here, corrupting the serial route's `first_row` goes undetected.
    #[test]
    fn both_routes_agree_bit_for_bit() {
        assert_eq!(MAX_MC, 64, "shape list below is chosen to straddle MAX_MC");
        for &(m, k, n) in &[
            (1usize, 64usize, 64usize),
            (3, 65, 33),
            (8, 64, 64),
            (13, 129, 67),
            (15, 256, 256),
            (16, 257, 129),
            (65, 33, 49),  // just over MAX_MC: two serial blocks, second a tail
            (130, 65, 67), // several serial blocks with a tail
            (192, 17, 23), // exact multiple of MAX_MC: no tail
        ] {
            let (a, b, mut c) = operands(m, k, n);
            with_forced_route(false, || gemm(&a, &b, &mut c, m, k, n));
            let serial = c.clone();
            with_forced_route(true, || gemm(&a, &b, &mut c, m, k, n));
            let sb: Vec<u32> = serial.iter().map(|v| v.to_bits()).collect();
            let pb: Vec<u32> = c.iter().map(|v| v.to_bits()).collect();
            assert!(
                sb == pb,
                "{m}x{k}x{n}: split and serial half GEMM disagree bit-for-bit"
            );
        }
    }
}
