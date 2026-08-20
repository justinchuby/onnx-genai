//! MLAS-style packed SIMD f32 SGEMM for x86-64 (AVX2 + FMA).
//!
//! This is a from-scratch Rust port of the *algorithm* used by ONNX Runtime's
//! MLAS SGEMM (MIT-licensed): panel packing + a register-blocked SIMD
//! microkernel + cache blocking. No MLAS C++ source is copied; only the
//! well-known GEBP/GEPP blocking strategy (Goto/van de Geijn) that MLAS itself
//! is built on is reproduced here in idiomatic Rust.
//!
//! Layout: `a` is `m*k` row-major, `b` is `k*n` row-major, `c` is `m*n`
//! row-major. The kernel computes `c = a @ b` (overwrite), accumulating in f32,
//! identical numerics to the generic path within f32 tolerance.
//!
//! ## Design
//!
//! * **Microkernel**: a `MR x NR` = `6 x 16` tile of C held in 12 YMM
//!   accumulators (16 f32 lanes = two `__m256` per row, six rows). Two more
//!   registers hold the B row (`2 x __m256`) and one broadcasts an A element,
//!   fitting the 16 YMM register file. Accumulation uses `_mm256_fmadd_ps`.
//! * **Packing**: A is packed per `MR`-row panel as `[k][MR]` (unit-stride
//!   broadcast source); B is packed per `NR`-column panel as `[k][NR]`
//!   (unit-stride vector loads). Edge panels are zero-padded to full `MR`/`NR`
//!   so the microkernel never needs masking; only the valid `mr x nr` corner of
//!   C is written back.
//! * **Cache blocking**: K is blocked in `KC` panels so a packed B panel
//!   (`KC x NR`) stays L1-resident while a C strip accumulates across it.
//!   Columns are blocked in `NC`-wide strips that also form the unit of Rayon
//!   parallelism.
//! * **Threading**: Rayon parallelizes over disjoint column strips of C. A is
//!   packed once and shared read-only; each strip packs its own B panels. Writes
//!   target disjoint columns, so a small `unsafe` `Send`/`Sync` pointer wrapper
//!   hands each task its output region without aliasing.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use rayon::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::half_gemm::HalfFormat;

/// Register-tile rows.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MR: usize = 6;
/// Register-tile columns (two `__m256` lanes).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const NR: usize = 16;
/// K-panel width kept L1-resident.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const KC: usize = 256;

/// Raw pointer to C wrapped so Rayon can hand disjoint column strips to worker
/// threads. Each strip writes a disjoint set of columns, so no two tasks ever
/// touch the same element — the `Send`/`Sync` impls are sound given that
/// invariant, which the driver upholds.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
struct CPtr(*mut f32);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for CPtr {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for CPtr {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl CPtr {
    /// Take `self` by value so the whole wrapper (not its raw field) is captured
    /// by a closure — preserving the `Send`/`Sync` guarantees.
    #[inline]
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Entry point: `c[m,n] = a[m,k] @ b[k,n]` (overwrite) using the AVX2/FMA
/// microkernel. The caller must ensure the host supports AVX2 + FMA (checked by
/// [`crate::backend::has_simd_x86`]); callers without it must use the generic
/// fallback instead.
///
/// `m == 1` takes the native GEMV ([`sgemm_simd_m1`]); every other `m` takes
/// the packed GEBP path. The dispatch itself lives in [`sgemm_simd_variant`],
/// which still accepts the route as an argument so the A/B harness can drive
/// both in one process — but production no longer has a choice to make, and
/// there is no env probe on this call.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn sgemm_simd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    sgemm_simd_variant(a, b, c, m, k, n, true);
}

/// Backend body with the M==1 route decision passed in explicitly rather than
/// read from env. `use_m1_gemv` only affects `m == 1`; for `m >= 2` the packed
/// GEBP path is taken unconditionally, which is *why* the M=128 rows in the A/B
/// harness are a valid control: an M==1-only route cannot move them.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm_simd_variant(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    use_m1_gemv: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    // #1091: dedicated M=1 GEMV. MLAS routes M==1 away from its packed GEBP for
    // exactly this reason (`sgemm.cpp`): "The data from matrix B is not
    // referenced multiple times, so using a local packed buffer is a wasted
    // memory copy." The packed path always calls `pack_b`, so at M=1 it pays a
    // full read+write copy of B (K*N f32) that is reused zero times — the whole
    // 2.2–4.6x decode gap measured against MLAS on AVX2. This GEMV streams B in
    // place: no pack, no resident buffer, strictly less memory traffic.
    //
    // Dispatch boundary (measured on this host, AVX2, process CPU time, control
    // -gated so the M=128 prefill rows confirm the run was quiet — see
    // `bench_f32_gemm_ab`). Per M=1 shape, gemv / packed CPU time:
    //   1x5120x7168   0.50    1x5120x5120   0.34    1x5120x13824  0.43
    //   1x13824x5120  0.42    1x5120x152064 0.34
    // The GEMV strictly dominates the packed default on *every* decode shape
    // (2.0-2.9x), so there is deliberately no fall-back to `sgemm_simd_packed`
    // here: it would be uniformly slower. The only residual gap is versus MLAS
    // (not our default), and only on the two largest shapes -- down_proj
    // (K=13824, gemv/mlas 1.22) and lm_head (N=152064, gemv/mlas 1.18) -- where
    // sequential B streaming trails MLAS's blocked M=1 asm; the other three win
    // outright (0.68-0.93). That MLAS-only boundary is a known limit, not a
    // regression against the code this build actually ships.
    //
    // #1091 landed this behind a default-off env toggle "until the win is
    // measured". It has now been measured end to end, through an ORT session
    // rather than a kernel driver: at `1x2048x2048`, f32, one thread on each
    // side, `ours/ORT` p50 goes 7.57 -> 1.18 with this route on, while the
    // M=128 prefill row -- which cannot reach it -- stays at 1.03. So the route
    // is the default and `use_m1_gemv` survives only as the A/B harness's
    // handle on the path it replaced.
    if m == 1 && use_m1_gemv {
        // SAFETY: `SimdX86` is only selected when the host has AVX2+FMA (see
        // `crate::backend::has_simd_x86`), the same guarantee `micro_6x16`
        // relies on; slice lengths are validated by the caller (`a.len()==k`,
        // `b.len()==k*n`, `c.len()==n`).
        unsafe {
            sgemm_simd_m1(a, b, c, k, n);
        }
        return;
    }

    sgemm_simd_packed(a, b, c, m, k, n);
}

/// Packed GEBP path: pack A into `MR`-row panels and B into `NR`-column L1
/// panels, then drive the `6x16` microkernel over Rayon column strips. This is
/// the correctness baseline for every `m` and the sole path for `m >= 2`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn sgemm_simd_packed(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Pack A once: one contiguous [k][MR] panel per MR-row block, zero-padded.
    let m_panels = m.div_ceil(MR);
    let mut apack = vec![0.0f32; m_panels * k * MR];
    pack_a(a, &mut apack, m, k);

    // Choose an NR-aligned column strip width: enough strips for load balance
    // across the Rayon pool while keeping each packed B panel cache-friendly.
    let n_panels = n.div_ceil(NR);
    let threads = rayon::current_num_threads().max(1);
    let target_tasks = threads.saturating_mul(8).max(1);
    let panels_per_strip = n_panels.div_ceil(target_tasks).clamp(1, 16);
    let strip_cols = panels_per_strip * NR;
    let strip_count = n.div_ceil(strip_cols);

    let cptr = CPtr(c.as_mut_ptr());
    let apack = &apack;

    (0..strip_count).into_par_iter().for_each(|s| {
        let j0 = s * strip_cols;
        let nc = strip_cols.min(n - j0);
        // Scratch for this strip's packed B panels: [KC][NR] per NR sub-panel.
        let strip_panels = nc.div_ceil(NR);
        let mut bpack = vec![0.0f32; KC * strip_panels * NR];
        // SAFETY: `cptr` addresses the caller's `c`; this task only writes
        // columns [j0, j0+nc), disjoint from every other strip.
        let c_base = cptr.get();

        let mut pc = 0usize;
        while pc < k {
            let kc = KC.min(k - pc);
            pack_b(b, &mut bpack, k, n, pc, kc, j0, nc);
            let first = pc == 0;

            for ip in 0..m_panels {
                let i0 = ip * MR;
                let mr = MR.min(m - i0);
                let apanel = &apack[ip * k * MR + pc * MR..ip * k * MR + pc * MR + kc * MR];
                let mut jr = 0usize;
                let mut jp = 0usize;
                while jr < nc {
                    let nr = NR.min(nc - jr);
                    let bpanel = &bpack[jp * KC * NR..jp * KC * NR + kc * NR];
                    // SAFETY: AVX2/FMA verified by the caller; `c_base` points at
                    // valid `m*n` storage and (i0,j0+jr) with (mr,nr) stays in
                    // bounds; this strip owns these columns exclusively.
                    unsafe {
                        micro_6x16(
                            apanel.as_ptr(),
                            bpanel.as_ptr(),
                            c_base.add(i0 * n + j0 + jr),
                            n,
                            kc,
                            mr,
                            nr,
                            first,
                        );
                    }
                    jr += NR;
                    jp += 1;
                }
            }
            pc += KC;
        }
    });
}

/// Transposed-B ("NT") entry point: `c[m,n] = a[m,k] @ b_nk[n,k]^T`, where
/// `b_nk` is `[n, k]` row-major (output row `n` at offset `n*k`, unit stride
/// over `k`). This is the layout the int4 `MatMulNBits` decode path already
/// caches (`weight_nk`), so routing prefill through here lets the batched GEMM
/// reuse that one contiguous dequant instead of materializing a second,
/// transposed `[k, n]` copy — the strided-scatter `Kn` pass that dominated
/// large-model time-to-first-token (#959).
///
/// **Bit-identical to `sgemm_simd(a, b_kn, c, m, k, n)`** on the `[k, n]`
/// transpose `b_kn` of `b_nk` (i.e. `b_kn[p*n + j] == b_nk[j*k + p]`): the
/// NT B-packer ([`pack_b_nt`]) produces the exact same packed panels the NN
/// packer ([`pack_b`]) would from `b_kn`, and the same [`pack_a`],
/// [`micro_6x16`] and K-panel order run over them, so every output element's
/// f32 accumulation sequence is unchanged. The only thing that moves is *how*
/// B reaches the pack buffer: MLAS's trick, ported natively — a contiguous
/// read of one `b_nk` row into an L1-resident tile at pack time, not a
/// full-array stride-`n` scatter.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn sgemm_simd_nt(a: &[f32], b_nk: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    sgemm_simd_nt_packed(a, b_nk, c, m, k, n);
}

/// Packed GEBP path for the transposed-B GEMM. Mirrors [`sgemm_simd_packed`]
/// exactly — same A packing, same column-strip / K-panel blocking, same
/// microkernel — except each strip packs its B panels from the `[n, k]`
/// operand via [`pack_b_nt`] rather than from a `[k, n]` operand via
/// [`pack_b`]. Because the packed panels are byte-for-byte identical to the NN
/// path's, so is the result.
///
/// Column strips are the unit of parallelism. When this runs inside an ORT
/// session an intra-op host pool is installed ([`onnx_runtime_ep_api::host_parallel`]);
/// the strips are dispatched onto *that* pool instead of forking a second
/// (rayon) pool beside ORT's spinning workers (#1143). With no host installed
/// — the native executor, which owns the machine — the strips run on rayon as
/// before. The strip decomposition does not affect any output element's
/// reduction order, so the choice of pool is numerically transparent.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn sgemm_simd_nt_packed(a: &[f32], b_nk: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Pack A once: one contiguous [k][MR] panel per MR-row block, zero-padded.
    let m_panels = m.div_ceil(MR);
    let mut apack = vec![0.0f32; m_panels * k * MR];
    pack_a(a, &mut apack, m, k);

    // Choose an NR-aligned column strip width: enough strips for load balance
    // while keeping each packed B panel cache-friendly (same policy as the NN
    // packed path).
    let n_panels = n.div_ceil(NR);
    let threads = rayon::current_num_threads().max(1);
    let target_tasks = threads.saturating_mul(8).max(1);
    let panels_per_strip = n_panels.div_ceil(target_tasks).clamp(1, 16);
    let strip_cols = panels_per_strip * NR;
    let strip_count = n.div_ceil(strip_cols);

    let cptr = CPtr(c.as_mut_ptr());
    let apack = &apack;

    // One column strip `[j0, j0+nc)` of C: pack its B panels from `b_nk` and
    // drive the microkernel across every K-panel. Each strip writes a disjoint
    // set of columns, so tasks never alias.
    let run_strip = move |s: usize| {
        let j0 = s * strip_cols;
        let nc = strip_cols.min(n - j0);
        let strip_panels = nc.div_ceil(NR);
        let mut bpack = vec![0.0f32; KC * strip_panels * NR];
        // SAFETY: `cptr` addresses the caller's `c`; this task only writes
        // columns [j0, j0+nc), disjoint from every other strip.
        let c_base = cptr.get();

        let mut pc = 0usize;
        while pc < k {
            let kc = KC.min(k - pc);
            pack_b_nt(b_nk, &mut bpack, k, pc, kc, j0, nc);
            let first = pc == 0;

            for ip in 0..m_panels {
                let i0 = ip * MR;
                let mr = MR.min(m - i0);
                let apanel = &apack[ip * k * MR + pc * MR..ip * k * MR + pc * MR + kc * MR];
                let mut jr = 0usize;
                let mut jp = 0usize;
                while jr < nc {
                    let nr = NR.min(nc - jr);
                    let bpanel = &bpack[jp * KC * NR..jp * KC * NR + kc * NR];
                    // SAFETY: AVX2/FMA verified by the caller; `c_base` points at
                    // valid `m*n` storage and (i0,j0+jr) with (mr,nr) stays in
                    // bounds; this strip owns these columns exclusively.
                    unsafe {
                        micro_6x16(
                            apanel.as_ptr(),
                            bpanel.as_ptr(),
                            c_base.add(i0 * n + j0 + jr),
                            n,
                            kc,
                            mr,
                            nr,
                            first,
                        );
                    }
                    jr += NR;
                    jp += 1;
                }
            }
            pc += KC;
        }
    };

    dispatch_strips(strip_count, &run_strip);
}

/// Dispatch `strip_count` independent column strips across a thread pool.
///
/// Prefers the host runtime's own pool when one is installed and has been seen
/// to help (an ORT intra-op pool — #1143): forking a rayon pool beside ORT's
/// spinning workers oversubscribes the cores. Falls back to rayon when there is
/// no host (the native executor) or the host's pool has no workers. A strip
/// reached from inside a host task keeps running on that task's thread rather
/// than nesting a second split.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn dispatch_strips(strip_count: usize, run_strip: &(dyn Fn(usize) + Sync)) {
    use onnx_runtime_ep_api::host_parallel;

    if !host_parallel::in_host_task()
        && let Some(host) = host_parallel::current()
        && host.prefer_host()
    {
        host.run(strip_count, run_strip);
        return;
    }
    (0..strip_count).into_par_iter().for_each(run_strip);
}

/// Pack a `kc x nc` block of the transposed operand `b_nk` (`[n, k]`
/// row-major) into `NR`-column panels, producing the *same* bytes
/// [`pack_b`] would from the `[k, n]` transpose. For output column
/// `col = j0 + jp*NR + c`, `b_nk` row `col` is contiguous over `k`, so the
/// `kc` depths for that column are read with unit stride (`b_nk[col*k + pc ..]`)
/// and scattered into the L1-resident pack tile at stride `NR`. That is the
/// whole transpose, done as a pack-time reshape of a tile already in cache
/// rather than a stride-`n` scatter across the full array.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn pack_b_nt(
    b_nk: &[f32],
    bpack: &mut [f32],
    k: usize,
    pc: usize,
    kc: usize,
    j0: usize,
    nc: usize,
) {
    let n_panels = nc.div_ceil(NR);
    for jp in 0..n_panels {
        let jcol = j0 + jp * NR;
        let nr = NR.min(nc - jp * NR);
        let dst = &mut bpack[jp * KC * NR..jp * KC * NR + kc * NR];
        for c in 0..nr {
            // `b_nk` row `jcol + c` is contiguous over the K dimension.
            let src = &b_nk[(jcol + c) * k + pc..(jcol + c) * k + pc + kc];
            for (p, &value) in src.iter().enumerate() {
                dst[p * NR + c] = value;
            }
        }
        // Zero-pad columns [nr, NR) so the microkernel needs no masking.
        if nr < NR {
            for p in 0..kc {
                dst[p * NR + nr..p * NR + NR].fill(0.0);
            }
        }
    }
}

/// Native M=1 SGEMV: `c[1,n] = a[1,k] @ b[k,n]` (overwrite), reading B in place.
///
/// Port of the *mechanism* behind MLAS `SgemmKernelM1Avx` (not its code): for a
/// single output row, matrix B is streamed exactly once, so MLAS deliberately
/// skips the packed-panel copy the general kernel uses. We mirror that — no
/// `pack_b`, no scratch, no resident buffer. Columns are tiled 32-wide so four
/// independent `__m256` accumulators hide the FMA latency chain (MLAS unrolls
/// K by 4 for the same reason); each 64-byte B cache line is read once and
/// fully consumed by a 16-lane tile. Rayon parallelizes over disjoint column
/// strips, so each task overwrites its own region of `c` with no aliasing.
///
/// # Safety
/// The host must support AVX2 + FMA (guaranteed by `SimdX86` selection). `a`
/// must address `k` f32, `b` `k*n` f32, `c` `n` f32.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sgemm_simd_m1(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    // Column strips are the unit of parallelism; width is a multiple of 32 so
    // every task starts on a 32-lane group boundary. A minimum of 8 groups
    // (256 columns) keeps the sequential B runs long enough to prefetch while
    // still giving ~2 tasks per worker for load balance.
    let threads = rayon::current_num_threads().max(1);
    let groups = n.div_ceil(32);
    let target_tasks = threads.saturating_mul(2).max(1);
    let groups_per_strip = groups.div_ceil(target_tasks).max(8);
    let strip_cols = groups_per_strip * 32;
    let strip_count = n.div_ceil(strip_cols);

    let cptr = CPtr(c.as_mut_ptr());

    (0..strip_count).into_par_iter().for_each(|s| {
        let j0 = s * strip_cols;
        let nc = strip_cols.min(n - j0);
        // SAFETY: this strip writes only columns [j0, j0+nc) of `c`, disjoint
        // from every other strip; AVX2+FMA verified by the caller; the B reads
        // `b[p*n + col + lane]` stay within `k*n` because `col+lane < n`. `a`
        // and `b` are shared read-only (`&[f32]` is `Sync`).
        unsafe {
            gemv_m1_strip(a.as_ptr(), b.as_ptr(), cptr.get(), k, n, j0, nc);
        }
    });
}

/// One column strip `[j0, j0+nc)` of the M=1 GEMV. See [`sgemm_simd_m1`].
///
/// K-outer / N-inner with **sequential** B streaming, mirroring MLAS
/// `SgemmKernelM1Avx` (`ProcessRowLoop4` over K unrolled ×4, `ProcessColumnLoop`
/// over N): for a group of up to 4 K-rows we sweep the whole strip contiguously,
/// so each B row is read front-to-back (hardware-prefetch friendly) exactly
/// once. C is accumulated in place across the `ceil(K/4)` groups and stays in
/// cache because a strip is sized to the pool, not to N. This is the layout
/// that matters for wide outputs (e.g. the 152064-wide lm_head), where a
/// column-major GEMV would stride B by `N` and thrash the TLB.
///
/// # Safety
/// AVX2+FMA required. `a` addresses `k` f32; `b` addresses `k*n` f32; `c`
/// addresses `n` f32 and this call writes only `[j0, j0+nc)`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_m1_strip(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    n: usize,
    j0: usize,
    nc: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        // Zero this strip's output; we then accumulate into it across K-groups.
        for j in 0..nc {
            *c.add(j0 + j) = 0.0;
        }

        let mut p = 0usize;
        // K unrolled by 4: one contiguous sweep of the strip per 4 B-rows,
        // amortising the C load/store over four FMAs (MLAS's ProcessRowLoop4).
        while p + 4 <= k {
            let a0 = _mm256_broadcast_ss(&*a.add(p));
            let a1 = _mm256_broadcast_ss(&*a.add(p + 1));
            let a2 = _mm256_broadcast_ss(&*a.add(p + 2));
            let a3 = _mm256_broadcast_ss(&*a.add(p + 3));
            let r0 = b.add(p * n + j0);
            let r1 = b.add((p + 1) * n + j0);
            let r2 = b.add((p + 2) * n + j0);
            let r3 = b.add((p + 3) * n + j0);
            let cbase = c.add(j0);
            let mut jj = 0usize;
            while jj + 8 <= nc {
                let mut acc = _mm256_loadu_ps(cbase.add(jj));
                acc = _mm256_fmadd_ps(a0, _mm256_loadu_ps(r0.add(jj)), acc);
                acc = _mm256_fmadd_ps(a1, _mm256_loadu_ps(r1.add(jj)), acc);
                acc = _mm256_fmadd_ps(a2, _mm256_loadu_ps(r2.add(jj)), acc);
                acc = _mm256_fmadd_ps(a3, _mm256_loadu_ps(r3.add(jj)), acc);
                _mm256_storeu_ps(cbase.add(jj), acc);
                jj += 8;
            }
            while jj < nc {
                let col = j0 + jj;
                *c.add(col) += *a.add(p) * *r0.add(jj)
                    + *a.add(p + 1) * *r1.add(jj)
                    + *a.add(p + 2) * *r2.add(jj)
                    + *a.add(p + 3) * *r3.add(jj);
                jj += 1;
            }
            p += 4;
        }
        // K remainder (0..3 rows): one sequential sweep each.
        while p < k {
            let av = _mm256_broadcast_ss(&*a.add(p));
            let rp = b.add(p * n + j0);
            let cbase = c.add(j0);
            let mut jj = 0usize;
            while jj + 8 <= nc {
                let acc = _mm256_fmadd_ps(
                    av,
                    _mm256_loadu_ps(rp.add(jj)),
                    _mm256_loadu_ps(cbase.add(jj)),
                );
                _mm256_storeu_ps(cbase.add(jj), acc);
                jj += 8;
            }
            while jj < nc {
                *c.add(j0 + jj) += *a.add(p) * *rp.add(jj);
                jj += 1;
            }
            p += 1;
        }
    }
}

/// Pack A into `MR`-row panels: `apack[panel][p*MR + r] = a[(panel*MR+r)*k + p]`,
/// zero-padding rows past `m`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn pack_a(a: &[f32], apack: &mut [f32], m: usize, k: usize) {
    let m_panels = m.div_ceil(MR);
    for ip in 0..m_panels {
        let i0 = ip * MR;
        let mr = MR.min(m - i0);
        let dst = &mut apack[ip * k * MR..ip * k * MR + k * MR];
        for p in 0..k {
            let out = &mut dst[p * MR..p * MR + MR];
            for r in 0..mr {
                out[r] = a[(i0 + r) * k + p];
            }
            // rows [mr, MR) remain zero (pre-zeroed buffer).
        }
    }
}

/// Pack a `kc x nc` block of B (rows `[pc,pc+kc)`, cols `[j0,j0+nc)`) into
/// `NR`-column panels: `bpack[panel][p*NR + c] = b[(pc+p)*n + j0+panel*NR+c]`,
/// zero-padding columns past `nc`. Uses the full `KC` panel stride so unused
/// tail rows of the scratch simply stay zero.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn pack_b(
    b: &[f32],
    bpack: &mut [f32],
    _k: usize,
    n: usize,
    pc: usize,
    kc: usize,
    j0: usize,
    nc: usize,
) {
    let n_panels = nc.div_ceil(NR);
    for jp in 0..n_panels {
        let jcol = j0 + jp * NR;
        let nr = NR.min(nc - jp * NR);
        let dst = &mut bpack[jp * KC * NR..jp * KC * NR + kc * NR];
        for p in 0..kc {
            let src = &b[(pc + p) * n + jcol..(pc + p) * n + jcol + nr];
            let out = &mut dst[p * NR..p * NR + NR];
            out[..nr].copy_from_slice(src);
            out[nr..NR].fill(0.0);
        }
    }
}

/// AVX2/FMA `6 x 16` microkernel. Accumulates `apack (kc x MR)` times
/// `bpack (kc x NR)` into the `mr x nr` corner of C at `c` (row stride `n`).
/// When `first` is true C is overwritten; otherwise the tile is added into the
/// running C (used across K-panels).
///
/// # Safety
/// The host must support AVX2 + FMA. `apack`/`bpack` must each address at least
/// `kc*MR` / `kc*NR` valid f32. `c` must address a valid `mr x nr` tile with row
/// stride `n`. `mr <= MR`, `nr <= NR`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_6x16(
    apack: *const f32,
    bpack: *const f32,
    c: *mut f32,
    n: usize,
    kc: usize,
    mr: usize,
    nr: usize,
    first: bool,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // Edition 2024 requires an explicit unsafe block even inside an unsafe fn.
    // SAFETY conditions are documented on the function signature above.
    unsafe {
        // 12 accumulators: two 8-wide lanes per row, six rows.
        let mut c0 = [_mm256_setzero_ps(); MR];
        let mut c1 = [_mm256_setzero_ps(); MR];

        let mut p = 0usize;
        while p < kc {
            let b0 = _mm256_loadu_ps(bpack.add(p * NR));
            let b1 = _mm256_loadu_ps(bpack.add(p * NR + 8));
            let arow = apack.add(p * MR);
            // Unrolled over the six A rows; padded rows broadcast 0 and are no-ops.
            let a0 = _mm256_broadcast_ss(&*arow.add(0));
            c0[0] = _mm256_fmadd_ps(a0, b0, c0[0]);
            c1[0] = _mm256_fmadd_ps(a0, b1, c1[0]);
            let a1 = _mm256_broadcast_ss(&*arow.add(1));
            c0[1] = _mm256_fmadd_ps(a1, b0, c0[1]);
            c1[1] = _mm256_fmadd_ps(a1, b1, c1[1]);
            let a2 = _mm256_broadcast_ss(&*arow.add(2));
            c0[2] = _mm256_fmadd_ps(a2, b0, c0[2]);
            c1[2] = _mm256_fmadd_ps(a2, b1, c1[2]);
            let a3 = _mm256_broadcast_ss(&*arow.add(3));
            c0[3] = _mm256_fmadd_ps(a3, b0, c0[3]);
            c1[3] = _mm256_fmadd_ps(a3, b1, c1[3]);
            let a4 = _mm256_broadcast_ss(&*arow.add(4));
            c0[4] = _mm256_fmadd_ps(a4, b0, c0[4]);
            c1[4] = _mm256_fmadd_ps(a4, b1, c1[4]);
            let a5 = _mm256_broadcast_ss(&*arow.add(5));
            c0[5] = _mm256_fmadd_ps(a5, b0, c0[5]);
            c1[5] = _mm256_fmadd_ps(a5, b1, c1[5]);
            p += 1;
        }

        if nr == NR {
            // Full-width store: two vector lanes per valid row.
            for r in 0..mr {
                let dst = c.add(r * n);
                if first {
                    _mm256_storeu_ps(dst, c0[r]);
                    _mm256_storeu_ps(dst.add(8), c1[r]);
                } else {
                    let old0 = _mm256_loadu_ps(dst);
                    let old1 = _mm256_loadu_ps(dst.add(8));
                    _mm256_storeu_ps(dst, _mm256_add_ps(old0, c0[r]));
                    _mm256_storeu_ps(dst.add(8), _mm256_add_ps(old1, c1[r]));
                }
            }
        } else {
            // Edge tile: spill each row to a scratch line and copy valid columns.
            let mut tmp = [0.0f32; NR];
            for r in 0..mr {
                _mm256_storeu_ps(tmp.as_mut_ptr(), c0[r]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), c1[r]);
                let dst = c.add(r * n);
                for (col, &val) in tmp[..nr].iter().enumerate() {
                    if first {
                        *dst.add(col) = val;
                    } else {
                        *dst.add(col) += val;
                    }
                }
            }
        }
    }
}

/// A block-quantized weight in its on-disk `MatMulNBits` layout, borrowed
/// rather than materialized, that can dequantize a run of one column's depths
/// straight into a packed panel.
///
/// The bit width is the only thing that differs between the implementations:
/// the strip policy, the panel layout and the microkernel are shared, so the
/// GEBP driver below is written once against this trait. `dequant_column` is
/// the innermost loop of the whole route, so it is a generic method rather than
/// an object-safe one -- every call monomorphizes down to a shift, a subtract
/// and a multiply with no indirect call.
#[cfg(target_arch = "x86_64")]
pub(crate) trait BlockQuantWeight: Sync {
    /// Dequantize depths `[pc, pc + kc)` of column `col` into
    /// `dst[p * NR + slot]` for `p` in `0..kc`.
    ///
    /// `scale_at(col * block_count + block)` supplies the scale; it is called
    /// once per block, not once per element.
    fn dequant_column<S: Fn(usize) -> f32>(
        &self,
        scale_at: &S,
        col: usize,
        pc: usize,
        kc: usize,
        slot: usize,
        dst: &mut [f32],
    );

    /// Fill a whole `kc x NR` panel: columns `[jcol, jcol + nr)` of depths
    /// `[pc, pc + kc)` into `dst[p * NR + slot]`.
    ///
    /// The default is exactly the loop it replaces, so an implementation that
    /// does not override it keeps its previous behaviour byte for byte. It
    /// exists so an implementation *can* see the whole panel at once: filling
    /// one column at a time writes `dst[p * NR + slot]` for consecutive `p`,
    /// one f32 every 64 bytes, where a group of columns transposed in registers
    /// can be written contiguously.
    ///
    /// Do not reach for this on the strength of the store pattern alone. The
    /// panel is only `KC * NR * 4` bytes and stays in L1, so the strided store
    /// is cheap: measured directly on the 8-bit weight -- where the store is
    /// the *only* thing an override can remove -- it is worth **0.07 ms of a
    /// 2.58 ms pack, inside the noise**. What paid for [`Int4Weight`]'s
    /// override was its scalar nibble unpack, not its stores. An override is
    /// worth it when the per-element *arithmetic* vectorizes.
    fn dequant_panel<S: Fn(usize) -> f32>(
        &self,
        scale_at: &S,
        jcol: usize,
        pc: usize,
        kc: usize,
        nr: usize,
        dst: &mut [f32],
    ) {
        for slot in 0..nr {
            self.dequant_column(scale_at, jcol + slot, pc, kc, slot, dst);
        }
    }
}

/// Block-quantized int4 weight described in its on-disk `MatMulNBits` layout,
/// borrowed rather than materialized.
///
/// Row `col` of the weight occupies `block_count * (block_size / 2)` packed
/// bytes; two 4-bit values per byte, low nibble first. `zero_points`, when
/// present, is one nibble per block in the same two-per-byte packing; absent
/// means symmetric quantization with the implicit midpoint 8.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Int4Weight<'a> {
    pub packed: &'a [u8],
    pub zero_points: Option<&'a [u8]>,
    pub block_size: usize,
    pub block_count: usize,
}

#[cfg(target_arch = "x86_64")]
impl Int4Weight<'_> {
    #[inline]
    fn packed_row_len(&self) -> usize {
        self.block_count * (self.block_size / 2)
    }

    #[inline]
    fn zero_point_row_len(&self) -> usize {
        self.block_count.div_ceil(2)
    }

    /// Zero point for `(col, block)`, or the symmetric midpoint.
    #[inline]
    fn zero_point(&self, col: usize, block: usize) -> f32 {
        match self.zero_points {
            None => 8.0,
            Some(zp) => {
                let byte = zp[col * self.zero_point_row_len() + block / 2];
                f32::from((byte >> (4 * (block % 2))) & 0x0f)
            }
        }
    }

    /// Dequantize depths `[pc, pc + kc)` of the [`DEQUANT_GROUP`] columns
    /// starting at `jcol + slot` into `dst[p * NR + slot ..][..DEQUANT_GROUP]`.
    ///
    /// The arithmetic is the same as [`Int4Weight::dequant_column`] -- widen
    /// the nibble, subtract the block's zero point, multiply by the block's
    /// scale, in that order -- done eight lanes at a time. The subtract and the
    /// multiply stay separate `_mm256_sub_ps` / `_mm256_mul_ps`, never
    /// contracted into an FMA, so every value is bit-identical to the scalar
    /// path rather than merely close.
    ///
    /// Columns are grouped because the nibble unpack has to be vectorized and
    /// a vector of eight dequantized values is eight *depths of one column*,
    /// which is the wrong orientation for a `[depth][NR]` panel. Transposing
    /// eight such vectors in registers fixes the orientation, and makes the
    /// stores contiguous as a side effect.
    ///
    /// The unpack is the part that pays. Measured at `4096x11008`, this
    /// override takes the pack's fixed cost from 4.92 ms to 2.09 ms; the same
    /// transpose applied to the 8-bit weight, which has no unpack to remove,
    /// is worth 0.07 ms and does not survive an interleaved A/B. So of the
    /// 2.83 ms, roughly 0.07 ms is the store and the rest is the nibble
    /// unpack: the index arithmetic, the bounds-checked byte load, the
    /// variable shift and the mask. The widen, subtract and multiply are in
    /// both scalar loops, so they cancel into the 0.07 ms.
    ///
    /// # Safety
    /// AVX2 must be available. `dst` must be at least `kc * NR` long,
    /// `slot + DEQUANT_GROUP` must not exceed `NR`, and `self.block_size` must
    /// be a multiple of [`DEQUANT_GROUP`].
    #[target_feature(enable = "avx2")]
    unsafe fn dequant_panel_avx2<S: Fn(usize) -> f32>(
        &self,
        scale_at: &S,
        jcol: usize,
        pc: usize,
        kc: usize,
        slot: usize,
        dst: &mut [f32],
    ) {
        use std::arch::x86_64::*;

        let block_size = self.block_size;
        let blob = block_size / 2;
        let row_len = self.packed_row_len();
        let whole = kc / DEQUANT_GROUP * DEQUANT_GROUP;

        let mut p = 0usize;
        while p < whole {
            let depth = pc + p;
            let block = depth / block_size;
            // Scale and zero point are constant across a whole block, but the
            // group is only eight depths, so looking them up per group repeats
            // each one `block_size / 8` times -- and for int4 the zero point
            // lookup is itself a nibble extract. Hoist them to the block.
            let mut scales = [0.0f32; DEQUANT_GROUP];
            let mut zeros = [0.0f32; DEQUANT_GROUP];
            for lane in 0..DEQUANT_GROUP {
                let col = jcol + slot + lane;
                scales[lane] = scale_at(col * self.block_count + block);
                zeros[lane] = self.zero_point(col, block);
            }
            let run = (block_size - depth % block_size).min(whole - p);
            let mut q = 0usize;
            while q < run {
                let offset_in_block = (depth + q) % block_size;
                let mut vecs = [_mm256_setzero_ps(); DEQUANT_GROUP];
                for (lane, vec) in vecs.iter_mut().enumerate() {
                    let col = jcol + slot + lane;
                    let scale = scales[lane];
                    let zero_point = zeros[lane];
                    let byte_at = col * row_len + block * blob + offset_in_block / 2;
                    // Eight nibbles of one column are four whole packed bytes,
                    // because the block size is a multiple of the group. Taken as
                    // one slice rather than four indexes: four separate
                    // bounds-checked byte loads and the shifts to reassemble them
                    // cost 18% of this pack (fixed 2.16 ms -> 1.78 ms at
                    // 4096x11008), where the slice compiles to a single unaligned
                    // 32-bit load with one bounds check.
                    let raw =
                        u32::from_le_bytes(self.packed[byte_at..byte_at + 4].try_into().unwrap());
                    let packed4 = _mm_cvtsi32_si128(raw as i32);
                    let low = _mm_and_si128(packed4, _mm_set1_epi8(0x0f));
                    // A 16-bit shift drags the neighbouring byte's low nibble into
                    // the top half of each lane; the mask drops it again.
                    let high = _mm_and_si128(_mm_srli_epi16(packed4, 4), _mm_set1_epi8(0x0f));
                    // Back into depth order: low nibble of each byte comes first.
                    let ordered = _mm_unpacklo_epi8(low, high);
                    let widened = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(ordered));
                    *vec = _mm256_mul_ps(
                        _mm256_sub_ps(widened, _mm256_set1_ps(zero_point)),
                        _mm256_set1_ps(scale),
                    );
                }
                // SAFETY: AVX2 is this function's own contract.
                unsafe { transpose8x8_ps(&mut vecs) };
                for (step, vec) in vecs.iter().enumerate() {
                    // SAFETY: `(p + q + step) * NR + slot + DEQUANT_GROUP <= kc * NR`,
                    // since `p + q + step < kc` and `slot + DEQUANT_GROUP <= NR`.
                    unsafe {
                        _mm256_storeu_ps(dst.as_mut_ptr().add((p + q + step) * NR + slot), *vec)
                    };
                }
                q += DEQUANT_GROUP;
            }
            p += run;
        }

        // Depths past the last whole group, when `kc` is not a multiple of
        // eight. Only the final `k` panel can reach this.
        for lane in 0..DEQUANT_GROUP {
            let col = jcol + slot + lane;
            let mut q = whole;
            while q < kc {
                let depth = pc + q;
                let block = depth / block_size;
                let offset_in_block = depth % block_size;
                let scale = scale_at(col * self.block_count + block);
                let zero_point = self.zero_point(col, block);
                let run = (block_size - offset_in_block).min(kc - q);
                for step in 0..run {
                    let index = offset_in_block + step;
                    let byte = self.packed[col * row_len + block * blob + index / 2];
                    let nibble = (byte >> (4 * (index % 2))) & 0x0f;
                    dst[(q + step) * NR + slot + lane] = (f32::from(nibble) - zero_point) * scale;
                }
                q += run;
            }
        }
    }
}

/// Columns dequantized together by [`Int4Weight::dequant_panel_avx2`].
///
/// Not a tuning parameter: it is the f32 lane count of an `__m256`, and the
/// transpose that makes the stores contiguous is an 8x8 one.
#[cfg(target_arch = "x86_64")]
const DEQUANT_GROUP: usize = 8;

/// Transpose eight `__m256` of f32 in registers: on entry `v[c]` holds eight
/// consecutive depths of column `c`; on exit `v[p]` holds one depth of all
/// eight columns.
///
/// The standard unpack / shuffle / permute sequence -- 24 shuffles, no memory
/// round trip.
///
/// # Safety
/// AVX2 must be available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose8x8_ps(v: &mut [std::arch::x86_64::__m256; 8]) {
    use std::arch::x86_64::*;
    let a0 = _mm256_unpacklo_ps(v[0], v[1]);
    let a1 = _mm256_unpackhi_ps(v[0], v[1]);
    let a2 = _mm256_unpacklo_ps(v[2], v[3]);
    let a3 = _mm256_unpackhi_ps(v[2], v[3]);
    let a4 = _mm256_unpacklo_ps(v[4], v[5]);
    let a5 = _mm256_unpackhi_ps(v[4], v[5]);
    let a6 = _mm256_unpacklo_ps(v[6], v[7]);
    let a7 = _mm256_unpackhi_ps(v[6], v[7]);
    let b0 = _mm256_shuffle_ps(a0, a2, 0b01_00_01_00);
    let b1 = _mm256_shuffle_ps(a0, a2, 0b11_10_11_10);
    let b2 = _mm256_shuffle_ps(a1, a3, 0b01_00_01_00);
    let b3 = _mm256_shuffle_ps(a1, a3, 0b11_10_11_10);
    let b4 = _mm256_shuffle_ps(a4, a6, 0b01_00_01_00);
    let b5 = _mm256_shuffle_ps(a4, a6, 0b11_10_11_10);
    let b6 = _mm256_shuffle_ps(a5, a7, 0b01_00_01_00);
    let b7 = _mm256_shuffle_ps(a5, a7, 0b11_10_11_10);
    v[0] = _mm256_permute2f128_ps(b0, b4, 0x20);
    v[1] = _mm256_permute2f128_ps(b1, b5, 0x20);
    v[2] = _mm256_permute2f128_ps(b2, b6, 0x20);
    v[3] = _mm256_permute2f128_ps(b3, b7, 0x20);
    v[4] = _mm256_permute2f128_ps(b0, b4, 0x31);
    v[5] = _mm256_permute2f128_ps(b1, b5, 0x31);
    v[6] = _mm256_permute2f128_ps(b2, b6, 0x31);
    v[7] = _mm256_permute2f128_ps(b3, b7, 0x31);
}

#[cfg(target_arch = "x86_64")]
impl BlockQuantWeight for Int4Weight<'_> {
    fn dequant_column<S: Fn(usize) -> f32>(
        &self,
        scale_at: &S,
        col: usize,
        pc: usize,
        kc: usize,
        slot: usize,
        dst: &mut [f32],
    ) {
        let block_size = self.block_size;
        let blob = block_size / 2;
        let packed_row =
            &self.packed[col * self.packed_row_len()..(col + 1) * self.packed_row_len()];
        let mut p = 0usize;
        while p < kc {
            let depth = pc + p;
            let block = depth / block_size;
            let offset_in_block = depth % block_size;
            let scale = scale_at(col * self.block_count + block);
            let zero_point = self.zero_point(col, block);
            // Depths left in this block, clipped to the panel.
            let run = (block_size - offset_in_block).min(kc - p);
            let block_bytes = &packed_row[block * blob..block * blob + blob];
            for step in 0..run {
                let index = offset_in_block + step;
                let byte = block_bytes[index / 2];
                let nibble = (byte >> (4 * (index % 2))) & 0x0f;
                dst[(p + step) * NR + slot] = (f32::from(nibble) - zero_point) * scale;
            }
            p += run;
        }
    }

    /// Fill the panel a [`DEQUANT_GROUP`]-wide column group at a time.
    ///
    /// Falls back to the per-column default whenever the vector path's
    /// preconditions do not hold, so behaviour is unchanged, only faster.
    fn dequant_panel<S: Fn(usize) -> f32>(
        &self,
        scale_at: &S,
        jcol: usize,
        pc: usize,
        kc: usize,
        nr: usize,
        dst: &mut [f32],
    ) {
        // The vector path needs each eight-depth run to sit inside one block
        // (so scale and zero point are loop-invariant across it) and to start
        // on a byte boundary (so eight nibbles are four whole bytes). `pc` is a
        // multiple of `KC`, so both follow from the block being a multiple of
        // the group.
        if self.block_size.is_multiple_of(DEQUANT_GROUP) && crate::backend::has_simd_x86() {
            let groups = nr / DEQUANT_GROUP;
            for g in 0..groups {
                // SAFETY: `has_simd_x86()` is the AVX2 check the callee
                // requires. `dst` is the caller's `kc * NR` panel, and the
                // callee writes only `[p * NR + slot, p * NR + slot + 8)` for
                // `p < kc` with `slot + 8 <= nr <= NR`.
                unsafe {
                    self.dequant_panel_avx2(scale_at, jcol, pc, kc, g * DEQUANT_GROUP, dst);
                }
            }
            for slot in groups * DEQUANT_GROUP..nr {
                self.dequant_column(scale_at, jcol + slot, pc, kc, slot, dst);
            }
            return;
        }
        for slot in 0..nr {
            self.dequant_column(scale_at, jcol + slot, pc, kc, slot, dst);
        }
    }
}

/// Block-quantized **8-bit** weight in the same borrowed `MatMulNBits` layout.
///
/// One byte per element, so row `col` occupies `block_count * block_size`
/// packed bytes and `zero_points`, when present, is one whole byte per block.
/// Absent means the implicit midpoint 128 -- `1 << (bits - 1)`, the same rule
/// the 4-bit case applies at 8.
///
/// The 8-bit case is the one where fusing matters most in absolute terms: the
/// route it replaces materializes a full `k * n` f32 weight *per call*, which
/// is four bytes out for every one byte in.
#[cfg(target_arch = "x86_64")]
pub(crate) struct Int8Weight<'a> {
    pub packed: &'a [u8],
    pub zero_points: Option<&'a [u8]>,
    pub block_size: usize,
    pub block_count: usize,
}

#[cfg(target_arch = "x86_64")]
impl Int8Weight<'_> {
    #[inline]
    fn packed_row_len(&self) -> usize {
        self.block_count * self.block_size
    }

    /// Zero point for `(col, block)`, or the symmetric midpoint.
    #[inline]
    fn zero_point(&self, col: usize, block: usize) -> f32 {
        match self.zero_points {
            None => 128.0,
            Some(zp) => f32::from(zp[col * self.block_count + block]),
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl BlockQuantWeight for Int8Weight<'_> {
    fn dequant_column<S: Fn(usize) -> f32>(
        &self,
        scale_at: &S,
        col: usize,
        pc: usize,
        kc: usize,
        slot: usize,
        dst: &mut [f32],
    ) {
        let block_size = self.block_size;
        let packed_row_len = self.packed_row_len();
        let packed_row = &self.packed[col * packed_row_len..(col + 1) * packed_row_len];
        let mut p = 0usize;
        while p < kc {
            let depth = pc + p;
            let block = depth / block_size;
            let offset_in_block = depth % block_size;
            let scale = scale_at(col * self.block_count + block);
            let zero_point = self.zero_point(col, block);
            let run = (block_size - offset_in_block).min(kc - p);
            let block_bytes = &packed_row[block * block_size + offset_in_block..][..run];
            for (step, &byte) in block_bytes.iter().enumerate() {
                dst[(p + step) * NR + slot] = (f32::from(byte) - zero_point) * scale;
            }
            p += run;
        }
    }
}

/// Prefill GEMM for borrowed block-quantized weights: `c[m,n] = a[m,k] @
/// dequant(B)[k,n]`, computed with the packed GEBP machinery above but with the
/// dequantization **fused into B's pack step**.
///
/// The point is arithmetic intensity. The row-serial borrowed kernel (#1117)
/// re-streams every packed weight byte once per activation row, so an `m`-token
/// prefill does the memory traffic of `m` decodes and gains nothing from
/// batching. Dequantizing the whole weight into a resident f32 `[n, k]` cache
/// fixes the traffic but costs 8x the weight in RAM, which is exactly what #979
/// removed. Fusing dequant into `pack_b` gets both: each packed byte is read
/// once per call, expanded into an L1-resident `KC x NR` f32 panel, and then
/// reused by every one of the `m` rows through the same `6x16` microkernel the
/// f32 SGEMM uses. The only extra memory is the per-strip panel scratch —
/// tens of KB, not gigabytes.
///
/// `scale_at(col * block_count + block)` supplies the scale, keeping the
/// caller's scale dtype (f32/f16/bf16) out of this module; it is called once
/// per block per column, not per element.
///
/// Generic over [`BlockQuantWeight`], so 4-bit and 8-bit share everything but
/// the unpacking of a byte.
///
/// # Safety
/// The caller must have verified AVX2 + FMA (i.e. [`crate::backend::has_simd_x86`]).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn quant_prefill_gebp<W, S>(
    a: &[f32],
    weight: &W,
    scale_at: S,
    bias: Option<&[f32]>,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) where
    W: BlockQuantWeight,
    S: Fn(usize) -> f32 + Sync,
{
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for (i, v) in c.iter_mut().enumerate() {
            *v = bias.map_or(0.0, |b| b[i % n]);
        }
        return;
    }

    let m_panels = m.div_ceil(MR);
    let mut apack = vec![0.0f32; m_panels * k * MR];
    pack_a(a, &mut apack, m, k);

    // Same strip policy as `sgemm_simd_packed`: NR-aligned, enough strips to
    // balance the pool, narrow enough that a strip's packed panels stay warm.
    let n_panels = n.div_ceil(NR);
    let threads = rayon::current_num_threads().max(1);
    let target_tasks = threads.saturating_mul(8).max(1);
    let panels_per_strip = n_panels.div_ceil(target_tasks).clamp(1, 16);
    let strip_cols = panels_per_strip * NR;
    let strip_count = n.div_ceil(strip_cols);

    let cptr = CPtr(c.as_mut_ptr());
    let apack = &apack;
    let scale_at = &scale_at;

    (0..strip_count).into_par_iter().for_each(|s| {
        let j0 = s * strip_cols;
        let nc = strip_cols.min(n - j0);
        let strip_panels = nc.div_ceil(NR);
        let mut bpack = vec![0.0f32; KC * strip_panels * NR];
        // SAFETY: `cptr` addresses the caller's `c`; this task writes only
        // columns [j0, j0+nc), disjoint from every other strip.
        let c_base = cptr.get();

        let mut pc = 0usize;
        while pc < k {
            let kc = KC.min(k - pc);
            pack_b_quant(weight, scale_at, &mut bpack, pc, kc, j0, nc);
            let first = pc == 0;

            for ip in 0..m_panels {
                let i0 = ip * MR;
                let mr = MR.min(m - i0);
                let apanel = &apack[ip * k * MR + pc * MR..ip * k * MR + pc * MR + kc * MR];
                let mut jr = 0usize;
                let mut jp = 0usize;
                while jr < nc {
                    let nr = NR.min(nc - jr);
                    let bpanel = &bpack[jp * KC * NR..jp * KC * NR + kc * NR];
                    // SAFETY: AVX2/FMA verified by the caller; `c_base` points
                    // at valid `m*n` storage and (i0, j0+jr) with (mr, nr)
                    // stays in bounds; this strip owns these columns.
                    unsafe {
                        micro_6x16(
                            apanel.as_ptr(),
                            bpanel.as_ptr(),
                            c_base.add(i0 * n + j0 + jr),
                            n,
                            kc,
                            mr,
                            nr,
                            first,
                        );
                    }
                    jr += NR;
                    jp += 1;
                }
            }
            pc += KC;
        }

        if let Some(bias) = bias {
            let bias_strip = &bias[j0..j0 + nc];
            for i in 0..m {
                // SAFETY: same disjoint-column ownership as the GEMM above.
                let row = unsafe { std::slice::from_raw_parts_mut(c_base.add(i * n + j0), nc) };
                for (v, b) in row.iter_mut().zip(bias_strip) {
                    *v += b;
                }
            }
        }
    });
}

/// Dequantize the `kc x nc` block of a block-quantized weight (depths
/// `[pc, pc+kc)`, columns `[j0, j0+nc)`) straight into `NR`-column f32 panels,
/// producing the exact bytes [`pack_b`] would produce from a materialized
/// `[k, n]` f32 weight.
///
/// The weight is stored column-major (`[n, block_count, blob]`), so one output
/// column's depths are contiguous — the same property that makes the transposed
/// pack cheap. Scale and zero point are hoisted per block by the
/// [`BlockQuantWeight`] implementation, so the inner loop is an unpack, a
/// subtract and a multiply.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn pack_b_quant<W, S>(
    weight: &W,
    scale_at: &S,
    bpack: &mut [f32],
    pc: usize,
    kc: usize,
    j0: usize,
    nc: usize,
) where
    W: BlockQuantWeight,
    S: Fn(usize) -> f32,
{
    let n_panels = nc.div_ceil(NR);
    for jp in 0..n_panels {
        let jcol = j0 + jp * NR;
        let nr = NR.min(nc - jp * NR);
        let dst = &mut bpack[jp * KC * NR..jp * KC * NR + kc * NR];
        weight.dequant_panel(scale_at, jcol, pc, kc, nr, dst);
        // Zero-pad columns [nr, NR) so the microkernel needs no masking.
        if nr < NR {
            for p in 0..kc {
                dst[p * NR + nr..p * NR + NR].fill(0.0);
            }
        }
    }
}

/// Fused widen→pack GEBP for a contiguous `f16`/`bf16` prefill GEMM:
/// `c[m,n] = a[m,k] @ b[k,n]` in `f32` (overwrite), with both operands still in
/// 16-bit storage.
///
/// The blocked half GEMM ([`super::half_gemm`]) splits only over rows of C, so
/// every row block re-widens and re-packs the whole of `B`: at `m = 64` on a
/// 32-thread host its own block size collapses to one row, i.e. 64 full passes
/// over the weight. This kernel keeps the packed-panel structure of
/// [`sgemm_simd_packed`] instead — `B` is widened *directly into* the L1
/// `KC x NR` panel the existing [`micro_6x16`] consumes, so the weight is
/// traversed once per column strip regardless of `m`, and the tuned microkernel
/// is reused unchanged.
///
/// `A` is widened once into a dense `m*k` f32 buffer and packed by the shared
/// [`pack_a`]; that transient is bounded by the activation, not the weight
/// (`m*k*4` bytes, ~4 MB at `m = 256, k = 4096`), and is freed when the call
/// returns. No f32 copy of `B` is ever materialized or retained, so this adds
/// no weight-derived cache.
///
/// Because the panels produced here are element-for-element what [`pack_a`] and
/// [`pack_b`] produce from the widened operands, results are **bit-identical**
/// to `sgemm_simd(widen(a), widen(b))` for every finite operand.
///
/// The one documented exception is a `bf16` *signalling* NaN: widening by shift
/// keeps its payload, while `half::bf16::to_f32` canonicalizes it to a quiet
/// NaN, so 126 of the 65536 `bf16` patterns widen to a different NaN encoding
/// here (`widening_matches_the_half_crate_over_the_whole_domain` pins exactly
/// which). No finite value differs, NaN still propagates as NaN, and the same
/// shift is what [`super::half_gemm`] already uses -- so this is inherited
/// behaviour, not new. `f16` has no such case: the hardware conversion is
/// bit-identical to `half::f16::to_f32` across the entire domain.
///
/// The caller must ensure the host supports AVX2 + FMA
/// ([`crate::backend::has_simd_x86`]).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn half_prefill_gebp(
    format: HalfFormat,
    a: &[u16],
    b: &[u16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c.fill(0.0);
        return;
    }

    let mut a_wide = vec![0.0f32; m * k];
    widen_half(format, &a[..m * k], &mut a_wide);

    let m_panels = m.div_ceil(MR);
    let mut apack = vec![0.0f32; m_panels * k * MR];
    pack_a(&a_wide, &mut apack, m, k);
    drop(a_wide);

    let widen = select_half_widen(format);
    let n_panels = n.div_ceil(NR);
    let threads = rayon::current_num_threads().max(1);
    let target_tasks = threads.saturating_mul(8).max(1);
    let panels_per_strip = n_panels.div_ceil(target_tasks).clamp(1, 16);
    let strip_cols = panels_per_strip * NR;
    let strip_count = n.div_ceil(strip_cols);

    let cptr = CPtr(c.as_mut_ptr());
    let apack = &apack;

    (0..strip_count).into_par_iter().for_each(|s| {
        let j0 = s * strip_cols;
        let nc = strip_cols.min(n - j0);
        let strip_panels = nc.div_ceil(NR);
        let mut bpack = vec![0.0f32; KC * strip_panels * NR];
        // SAFETY: `cptr` addresses the caller's `c`; this task only writes
        // columns [j0, j0+nc), disjoint from every other strip.
        let c_base = cptr.get();

        let mut pc = 0usize;
        while pc < k {
            let kc = KC.min(k - pc);
            pack_b_half(widen, b, &mut bpack, n, pc, kc, j0, nc);
            let first = pc == 0;

            for ip in 0..m_panels {
                let i0 = ip * MR;
                let mr = MR.min(m - i0);
                let apanel = &apack[ip * k * MR + pc * MR..ip * k * MR + pc * MR + kc * MR];
                let mut jr = 0usize;
                let mut jp = 0usize;
                while jr < nc {
                    let nr = NR.min(nc - jr);
                    let bpanel = &bpack[jp * KC * NR..jp * KC * NR + kc * NR];
                    // SAFETY: AVX2/FMA verified by the caller; `c_base` points
                    // at valid `m*n` storage and (i0, j0+jr) with (mr, nr)
                    // stays in bounds; this strip owns these columns.
                    unsafe {
                        micro_6x16(
                            apanel.as_ptr(),
                            bpanel.as_ptr(),
                            c_base.add(i0 * n + j0 + jr),
                            n,
                            kc,
                            mr,
                            nr,
                            first,
                        );
                    }
                    jr += NR;
                    jp += 1;
                }
            }
            pc += KC;
        }
    });
}

/// Widen a contiguous run of 16-bit floats into `f32`, using the vector
/// conversion when the host has it. Delegates to the same helpers the blocked
/// half GEMM packs with, so both kernels widen identically.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn widen_half(format: HalfFormat, source: &[u16], destination: &mut [f32]) {
    super::half_gemm::widen_contiguous(format, source, destination);
}

/// Widen and pack a `kc x nc` block of a 16-bit `B` (rows `[pc,pc+kc)`, columns
/// `[j0,j0+nc)`) into the `NR`-column `f32` panels [`micro_6x16`] reads,
/// zero-padding columns past `nc`. The layout matches [`pack_b`] applied to a
/// widened `B` exactly, panel stride `KC` included.
///
/// The whole block is packed under one feature dispatch: the conversion is
/// otherwise a handful of instructions per 16 columns, so a per-panel call
/// would cost more than the work it guards.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn pack_b_half(
    widen: HalfWiden,
    b: &[u16],
    bpack: &mut [f32],
    n: usize,
    pc: usize,
    kc: usize,
    j0: usize,
    nc: usize,
) {
    match widen {
        HalfWiden::F16c => {
            // SAFETY: `HalfWiden::F16c` is only selected after runtime
            // detection of AVX2 and F16C.
            unsafe { pack_b_half_f16c(b, bpack, n, pc, kc, j0, nc) }
        }
        HalfWiden::Bf16Avx2 => {
            // SAFETY: `HalfWiden::Bf16Avx2` is only selected after runtime
            // detection of AVX2.
            unsafe { pack_b_half_bf16_avx2(b, bpack, n, pc, kc, j0, nc) }
        }
        HalfWiden::Scalar(format) => pack_b_half_scalar(format, b, bpack, n, pc, kc, j0, nc),
    }
}

/// Which widening the host can use for a given 16-bit format. `f16` needs F16C
/// for the hardware conversion; `bf16` widens with a shift, so AVX2 (already
/// required by the microkernel) is enough.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HalfWiden {
    F16c,
    Bf16Avx2,
    Scalar(HalfFormat),
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn select_half_widen(format: HalfFormat) -> HalfWiden {
    // Both arms check *every* feature their target function declares, rather
    // than leaning on the AVX2 the microkernel already required: this is also
    // reached from tests, which do not go through that gate.
    match format {
        HalfFormat::F16
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("f16c") =>
        {
            HalfWiden::F16c
        }
        HalfFormat::Bf16 if std::arch::is_x86_feature_detected!("avx2") => HalfWiden::Bf16Avx2,
        other => HalfWiden::Scalar(other),
    }
}

/// Portable widen-and-pack, used when the host lacks the vector conversion.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn pack_b_half_scalar(
    format: HalfFormat,
    b: &[u16],
    bpack: &mut [f32],
    n: usize,
    pc: usize,
    kc: usize,
    j0: usize,
    nc: usize,
) {
    let n_panels = nc.div_ceil(NR);
    for jp in 0..n_panels {
        let jcol = j0 + jp * NR;
        let nr = NR.min(nc - jp * NR);
        let dst = &mut bpack[jp * KC * NR..jp * KC * NR + kc * NR];
        for p in 0..kc {
            let src = &b[(pc + p) * n + jcol..(pc + p) * n + jcol + nr];
            let out = &mut dst[p * NR..p * NR + NR];
            widen_half(format, src, &mut out[..nr]);
            out[nr..NR].fill(0.0);
        }
    }
}

/// F16C widen-and-pack: each 16-column row segment is two `_mm256_cvtph_ps`
/// conversions straight into the panel.
///
/// # Safety
/// The host must support AVX2 + F16C. Indices are the same ones
/// [`pack_b_half_scalar`] checks, so every access stays inside `b`/`bpack`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_b_half_f16c(
    b: &[u16],
    bpack: &mut [f32],
    n: usize,
    pc: usize,
    kc: usize,
    j0: usize,
    nc: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n_panels = nc.div_ceil(NR);
    for jp in 0..n_panels {
        let jcol = j0 + jp * NR;
        let nr = NR.min(nc - jp * NR);
        let dst = &mut bpack[jp * KC * NR..jp * KC * NR + kc * NR];
        for p in 0..kc {
            let src = &b[(pc + p) * n + jcol..(pc + p) * n + jcol + nr];
            let out = &mut dst[p * NR..p * NR + NR];
            if nr == NR {
                // SAFETY: `src` holds NR = 16 elements and `out` 16 f32.
                unsafe {
                    let lo = _mm_loadu_si128(src.as_ptr().cast());
                    let hi = _mm_loadu_si128(src.as_ptr().add(8).cast());
                    _mm256_storeu_ps(out.as_mut_ptr(), _mm256_cvtph_ps(lo));
                    _mm256_storeu_ps(out.as_mut_ptr().add(8), _mm256_cvtph_ps(hi));
                }
                continue;
            }
            for (slot, &bits) in out[..nr].iter_mut().zip(src) {
                *slot = half::f16::from_bits(bits).to_f32();
            }
            out[nr..NR].fill(0.0);
        }
    }
}

/// AVX2 widen-and-pack for `bf16`: widening is a 16-bit left shift, so each
/// 16-column row segment is two shifts straight into the panel.
///
/// # Safety
/// The host must support AVX2. Indices are the same ones
/// [`pack_b_half_scalar`] checks, so every access stays inside `b`/`bpack`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_b_half_bf16_avx2(
    b: &[u16],
    bpack: &mut [f32],
    n: usize,
    pc: usize,
    kc: usize,
    j0: usize,
    nc: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n_panels = nc.div_ceil(NR);
    for jp in 0..n_panels {
        let jcol = j0 + jp * NR;
        let nr = NR.min(nc - jp * NR);
        let dst = &mut bpack[jp * KC * NR..jp * KC * NR + kc * NR];
        for p in 0..kc {
            let src = &b[(pc + p) * n + jcol..(pc + p) * n + jcol + nr];
            let out = &mut dst[p * NR..p * NR + NR];
            if nr == NR {
                // SAFETY: `src` holds NR = 16 elements and `out` 16 f32.
                unsafe {
                    let lo = _mm_loadu_si128(src.as_ptr().cast());
                    let hi = _mm_loadu_si128(src.as_ptr().add(8).cast());
                    let lo = _mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(lo));
                    let hi = _mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(hi));
                    _mm256_storeu_ps(out.as_mut_ptr(), _mm256_castsi256_ps(lo));
                    _mm256_storeu_ps(out.as_mut_ptr().add(8), _mm256_castsi256_ps(hi));
                }
                continue;
            }
            for (slot, &bits) in out[..nr].iter_mut().zip(src) {
                *slot = half::bf16::from_bits(bits).to_f32();
            }
            out[nr..NR].fill(0.0);
        }
    }
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    use super::*;
    use crate::backend::has_simd_x86;

    /// Naive reference GEMM (row-major, f32 accumulate) for cross-checking.
    fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let aip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += aip * b[p * n + j];
                }
            }
        }
        c
    }

    fn fill(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i + seed) as f32 * 0.123).sin()) * 2.0 - 0.5)
            .collect()
    }

    fn check(m: usize, k: usize, n: usize) {
        if !has_simd_x86() {
            return; // No AVX2/FMA: the SIMD path is never selected here.
        }
        let a = fill(m * k, 1);
        let b = fill(k * n, 7);
        let expect = reference(&a, &b, m, k, n);
        let mut got = vec![0.0f32; m * n];
        sgemm_simd(&a, &b, &mut got, m, k, n);
        for (idx, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            let tol = 1e-3 * (1.0 + e.abs());
            assert!(
                (g - e).abs() <= tol,
                "mismatch at {idx} for {m}x{k}x{n}: got {g}, expect {e}"
            );
        }
    }

    #[test]
    fn exact_tile_multiple() {
        check(12, 64, 32);
    }

    #[test]
    fn tail_shapes() {
        // M/N/K not multiples of MR(6)/NR(16): exercises zero-padded packing.
        check(7, 33, 17);
        check(1, 5, 3);
        check(6, 16, 16);
        check(5, 1, 5);
    }

    #[test]
    fn thin_vectors() {
        check(1, 128, 1); // 1xK @ Kx1
        check(1, 512, 256); // GEMV-like row
        check(256, 512, 1); // column result
    }

    #[test]
    fn multi_kpanel() {
        // K spans several KC blocks to exercise the accumulate (non-first) path.
        check(9, KC * 2 + 13, 40);
    }

    #[test]
    fn zero_dims() {
        let mut c = vec![1.0f32; 4];
        sgemm_simd(&[], &[], &mut c, 0, 3, 4); // m=0: leaves c untouched
        sgemm_simd(&[1.0], &[], &mut c, 2, 0, 2); // k=0: zeros c
        assert_eq!(&c[..4], &[0.0, 0.0, 0.0, 0.0]);
    }

    /// #1091: the native M=1 GEMV must match the naive reference within f32
    /// tolerance across tile-exact, tail, and multi-cache-line N shapes.
    fn check_m1(k: usize, n: usize) {
        if !has_simd_x86() {
            return;
        }
        let a = fill(k, 3);
        let b = fill(k * n, 11);
        let expect = reference(&a, &b, 1, k, n);
        let mut got = vec![0.0f32; n];
        // SAFETY: has_simd_x86() confirmed AVX2+FMA; slices are sized 1*k / k*n / n.
        unsafe {
            sgemm_simd_m1(&a, &b, &mut got, k, n);
        }
        for (idx, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            let tol = 1e-3 * (1.0 + e.abs());
            assert!(
                (g - e).abs() <= tol,
                "m1 mismatch at {idx} for 1x{k}x{n}: got {g}, expect {e}"
            );
        }
    }

    #[test]
    fn m1_gemv_shapes() {
        check_m1(64, 32); // exact 32-col group
        check_m1(128, 16); // two 8-col tiles, no 32-group
        check_m1(96, 40); // one 32-group + 8-tile
        check_m1(50, 37); // 32-group + tail scalar (37 = 32 + 5)
        check_m1(1, 3); // tiny, all scalar tail
        check_m1(512, 5120); // model-scale decode width
    }

    /// The M=1 GEMV must agree with the packed path within f32 tolerance — they
    /// differ only by summation reassociation, never in which products are
    /// summed. The packed side is reached through `sgemm_simd_variant(.., false)`
    /// rather than `sgemm_simd`, because `sgemm_simd` now *is* the GEMV at
    /// `m == 1`; comparing it against itself would prove nothing.
    #[test]
    fn m1_route_matches_packed_within_tolerance() {
        if !has_simd_x86() {
            return;
        }
        let (k, n) = (300usize, 517usize);
        let a = fill(k, 5);
        let b = fill(k * n, 9);
        let mut packed = vec![0.0f32; n];
        sgemm_simd_variant(&a, &b, &mut packed, 1, k, n, false);
        let mut gemv = vec![0.0f32; n];
        // SAFETY: AVX2+FMA confirmed; sizes match 1*k / k*n / n.
        unsafe {
            sgemm_simd_m1(&a, &b, &mut gemv, k, n);
        }
        let max_ref = packed.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let max_err = gemv
            .iter()
            .zip(packed.iter())
            .fold(0.0f32, |m, (&g, &p)| m.max((g - p).abs()));
        assert!(
            max_err <= 1e-3 * (1.0 + max_ref),
            "m1 GEMV vs packed max abs error {max_err} exceeds tolerance (max_ref {max_ref})"
        );
        // ...and the two really are different code, so the tolerance check
        // above is not comparing one path with itself.
        assert_ne!(
            packed, gemv,
            "packed and GEMV produced identical bits; one of the two routes was \
             not taken and the tolerance assertion is vacuous"
        );
    }

    /// Transpose a `[k, n]` row-major matrix into the `[n, k]` row-major
    /// ("Nk") layout the NT kernel reads: `b_nk[j*k + p] == b_kn[p*n + j]`.
    fn transpose_kn_to_nk(b_kn: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut b_nk = vec![0.0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                b_nk[j * k + p] = b_kn[p * n + j];
            }
        }
        b_nk
    }

    /// The NT kernel (`c = a @ b_nk^T`) must be **bit-identical** to the NN
    /// packed path (`c = a @ b_kn`) on the transpose of `b_nk`. The NT B-packer
    /// produces the same packed panels the NN packer would, and the same
    /// microkernel and K-panel order run over them, so the f32 accumulation
    /// sequence — and therefore every bit of the result — is unchanged. This is
    /// stricter than a tolerance check and is the property `MatMulNBits` prefill
    /// relies on to reuse its cached `Nk` dequant.
    fn check_nt_bit_identical(m: usize, k: usize, n: usize) {
        if !has_simd_x86() {
            return; // No AVX2/FMA: the SIMD path is never selected here.
        }
        let a = fill(m * k, 2);
        let b_kn = fill(k * n, 13);
        let b_nk = transpose_kn_to_nk(&b_kn, k, n);

        // NN reference: force the packed path (never the M=1 GEMV) so the
        // comparison is packed-vs-packed for every m.
        let mut expect = vec![0.0f32; m * n];
        sgemm_simd_packed(&a, &b_kn, &mut expect, m, k, n);

        let mut got = vec![0.0f32; m * n];
        sgemm_simd_nt(&a, &b_nk, &mut got, m, k, n);

        for (idx, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "NT vs NN packed not bit-identical for {m}x{k}x{n} at {idx}: got {g}, expect {e}"
            );
        }
    }

    #[test]
    fn nt_matches_nn_packed_bit_identical() {
        // Exact tile multiples and a multi-KC-panel K.
        check_nt_bit_identical(12, 64, 32);
        check_nt_bit_identical(9, KC * 2 + 13, 48);
        // Tail shapes: m, k, n not multiples of MR(6)/NR(16)/8; the corners
        // where an NT packer's zero-padding and edge stores would break.
        for &m in &[1usize, 2, 7, 33] {
            for &n in &[1usize, 3, 63, 64, 65] {
                check_nt_bit_identical(m, 40, n);
            }
        }
        check_nt_bit_identical(7, 33, 17);
        check_nt_bit_identical(5, 1, 5);
    }

    /// Randomized differential sweep: NT vs NN packed over a spread of odd
    /// shapes must be bit-identical every time.
    #[test]
    fn nt_matches_nn_packed_randomized() {
        if !has_simd_x86() {
            return;
        }
        // A small LCG so the shapes are reproducible without an rng dep.
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut next = |bound: usize| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1 + (state >> 33) as usize % bound
        };
        for _ in 0..64 {
            let m = next(40);
            let k = next(300);
            let n = next(200);
            check_nt_bit_identical(m, k, n);
        }
    }

    #[test]
    fn nt_zero_dims() {
        let mut c = vec![1.0f32; 4];
        sgemm_simd_nt(&[], &[], &mut c, 0, 3, 4); // m=0: leaves c untouched
        assert_eq!(&c[..4], &[1.0, 1.0, 1.0, 1.0]);
        sgemm_simd_nt(&[1.0], &[], &mut c, 2, 0, 2); // k=0: zeros c
        assert_eq!(&c[..4], &[0.0, 0.0, 0.0, 0.0]);
    }

    /// Microbenchmark for the NT prefill route (#959, #1091). Quantifies, at
    /// real decode-layer shapes, the term this change removes: the strided
    /// `Kn` materialization (each K step written at stride N — the scatter #959
    /// measured at ~2.9x the contiguous `Nk` pass and degrading with N), plus
    /// the NN vs NT GEMM cost over the same data. The NT route reuses the
    /// contiguous `Nk` weight the decode path already caches, so the whole `Kn`
    /// pass goes to zero. Bit-identity (NT == NN on the transpose) is asserted
    /// in the same harness so the timing is never taken on a wrong result.
    ///
    /// Ignored by default (a benchmark, not a correctness gate). Run with:
    /// `cargo test --release -p onnx-runtime-ep-cpu --lib nt_prefill_bench \
    ///   -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn nt_prefill_bench() {
        use std::time::Instant;

        // Representative decode-projection shapes (K x N), prefill row count m.
        let shapes = [(5120usize, 5120usize), (5120, 13824), (13824, 5120)];
        let m = 16usize;
        let reps = 7;

        for (k, n) in shapes {
            // The natural contiguous [n, k] weight the decode path caches.
            let b_nk = fill(n * k, 21);
            let a = fill(m * k, 4);

            // Correctness: NT over b_nk must equal NN over its [k, n] transpose.
            let b_kn = transpose_kn_to_nk(&b_nk, n, k); // reuse: [n,k]->[k,n]
            let mut nn = vec![0.0f32; m * n];
            let mut nt = vec![0.0f32; m * n];
            sgemm_simd_packed(&a, &b_kn, &mut nn, m, k, n);
            sgemm_simd_nt(&a, &b_nk, &mut nt, m, k, n);
            assert_eq!(nn, nt, "NT vs NN not bit-identical at {m}x{k}x{n}");

            // Time the strided-scatter Kn materialization the NT route removes:
            // building the [k, n] transpose from [n, k] (element weight[out,depth]
            // written at index depth*n+out) — the shape of the `dequant-kn` pass.
            let mut kn_scatter = vec![0.0f32; k * n];
            let scatter = |dst: &mut [f32]| {
                for out in 0..n {
                    let row = &b_nk[out * k..out * k + k];
                    for (depth, &v) in row.iter().enumerate() {
                        dst[depth * n + out] = v;
                    }
                }
            };
            let bench = |label: &str, mut f: Box<dyn FnMut() + '_>| {
                let mut samples: Vec<f64> = (0..reps)
                    .map(|_| {
                        let t = Instant::now();
                        f();
                        t.elapsed().as_secs_f64() * 1e3
                    })
                    .collect();
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let best = samples[0];
                let worst = samples[reps - 1];
                let median = samples[reps / 2];
                eprintln!(
                    "  {label:<14} best={best:7.3}ms median={median:7.3}ms worst={worst:7.3}ms"
                );
                best
            };

            eprintln!(
                "shape m={m} k={k} n={n} (weight {} MiB f32):",
                (n * k * 4) >> 20
            );
            let kn_ms = bench(
                "dequant-kn*",
                Box::new(|| {
                    scatter(&mut kn_scatter);
                    std::hint::black_box(&kn_scatter);
                }),
            );
            let nn_ms = bench(
                "NN gemm",
                Box::new(|| {
                    sgemm_simd_packed(&a, &b_kn, &mut nn, m, k, n);
                    std::hint::black_box(&nn);
                }),
            );
            let nt_ms = bench(
                "NT gemm",
                Box::new(|| {
                    sgemm_simd_nt(&a, &b_nk, &mut nt, m, k, n);
                    std::hint::black_box(&nt);
                }),
            );
            eprintln!(
                "  old (dequant-kn* + NN) = {:.3}ms  ->  new (NT, kn removed) = {:.3}ms  \
                 [dequant-kn* is a plain f32 transpose; the int4 dequant it stands in for \
                 is ~2.9x heavier per #959]",
                kn_ms + nn_ms,
                nt_ms
            );
        }
    }
    /// What the public entry point actually does at `m == 1`.
    ///
    /// The route used to be an env toggle that defaulted to the packed path, so
    /// the shipped binary never reached the GEMV. Nothing else in this file
    /// fails if that regresses -- `m1_gemv_shapes` calls the kernel directly and
    /// `m1_route_matches_packed_within_tolerance` names both routes explicitly
    /// -- which is exactly why this test exists, and why it asserts bit
    /// equality with the GEMV rather than a tolerance against the packed path.
    #[test]
    fn the_default_entry_point_routes_m1_to_the_gemv() {
        if !has_simd_x86() {
            return;
        }
        let (k, n) = (300usize, 517usize);
        let a = fill(k, 5);
        let b = fill(k * n, 9);
        let mut shipped = vec![0.0f32; n];
        sgemm_simd(&a, &b, &mut shipped, 1, k, n);
        let mut gemv = vec![0.0f32; n];
        // SAFETY: AVX2+FMA confirmed; sizes match 1*k / k*n / n.
        unsafe {
            sgemm_simd_m1(&a, &b, &mut gemv, k, n);
        }
        assert_eq!(
            shipped, gemv,
            "sgemm_simd must take the M=1 GEMV, bit for bit"
        );

        // The M>=2 rows are the control: the GEMV cannot reach them, so the
        // shipped entry must still be the packed kernel there.
        let a2 = fill(2 * k, 5);
        let mut shipped2 = vec![0.0f32; 2 * n];
        sgemm_simd(&a2, &b, &mut shipped2, 2, k, n);
        let mut packed2 = vec![0.0f32; 2 * n];
        sgemm_simd_variant(&a2, &b, &mut packed2, 2, k, n, false);
        assert_eq!(
            shipped2, packed2,
            "M>=2 must still be the packed GEBP path, bit for bit"
        );
    }

    // The route is a compile-time constant, so no environment variable can
    // reach it. There is deliberately no test that sets
    // `ONNX_GENAI_CPU_MM_SIMD_M1_GEMV` to prove that: `set_var` racing another
    // thread's `getenv` is a data race, and this test binary runs its cases in
    // parallel next to dozens of live `env::var` readers. The bit-exact
    // comparison in `the_default_entry_point_routes_m1_to_the_gemv` pins the
    // route without touching process state.

    #[cfg(target_arch = "x86_64")]
    /// Build a deterministic int4 weight in `MatMulNBits` layout plus the f32
    /// matrix it dequantizes to, so the fused kernel can be checked against the
    /// ordinary SGEMM on the *same* numbers.
    fn int4_weight(
        k: usize,
        n: usize,
        block_size: usize,
        asymmetric: bool,
    ) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>, Vec<f32>) {
        let block_count = k.div_ceil(block_size);
        let blob = block_size / 2;
        let mut packed = vec![0u8; n * block_count * blob];
        for (i, byte) in packed.iter_mut().enumerate() {
            *byte = ((i * 37 + 11) % 256) as u8;
        }
        let scales: Vec<f32> = (0..n * block_count)
            .map(|i| 0.01 + ((i % 13) as f32) * 0.003)
            .collect();
        let zero_points = asymmetric.then(|| {
            (0..n * block_count.div_ceil(2))
                .map(|i| ((i * 53 + 7) % 256) as u8)
                .collect::<Vec<u8>>()
        });

        // Dequantize to a `[k, n]` row-major f32 matrix: the reference operand.
        let mut dense = vec![0.0f32; k * n];
        for col in 0..n {
            for depth in 0..k {
                let block = depth / block_size;
                let offset = depth % block_size;
                let byte = packed[col * block_count * blob + block * blob + offset / 2];
                let nibble = (byte >> (4 * (offset % 2))) & 0x0f;
                let zero_point = match &zero_points {
                    None => 8.0,
                    Some(zp) => {
                        let byte = zp[col * block_count.div_ceil(2) + block / 2];
                        f32::from((byte >> (4 * (block % 2))) & 0x0f)
                    }
                };
                dense[depth * n + col] =
                    (f32::from(nibble) - zero_point) * scales[col * block_count + block];
            }
        }
        (packed, scales, zero_points, dense)
    }

    #[cfg(target_arch = "x86_64")]
    fn check_int4_gebp(m: usize, k: usize, n: usize, block_size: usize, asymmetric: bool) {
        if !has_simd_x86() {
            return;
        }
        let (packed, scales, zero_points, dense) = int4_weight(k, n, block_size, asymmetric);
        let a = fill(m * k, 3);
        let bias: Vec<f32> = (0..n).map(|j| ((j % 7) as f32) * 0.25 - 0.5).collect();

        let mut expect = vec![0.0f32; m * n];
        sgemm_simd(&a, &dense, &mut expect, m, k, n);
        for (i, v) in expect.iter_mut().enumerate() {
            *v += bias[i % n];
        }

        let weight = Int4Weight {
            packed: &packed,
            zero_points: zero_points.as_deref(),
            block_size,
            block_count: k.div_ceil(block_size),
        };
        let mut got = vec![0.0f32; m * n];
        quant_prefill_gebp(
            &a,
            &weight,
            |index| scales[index],
            Some(&bias),
            &mut got,
            m,
            k,
            n,
        );
        assert_eq!(
            got, expect,
            "fused dequant must reproduce the packed SGEMM on the dequantized \
             weight bit for bit (m={m}, k={k}, n={n}, block={block_size}, asym={asymmetric})"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int4_gebp_matches_the_sgemm_on_the_dequantized_weight() {
        // Panel-aligned, then every awkward edge: m/n past the 6x16 tile, k
        // past one KC panel, and a K that is not a whole number of blocks.
        check_int4_gebp(8, 64, 16, 32, false);
        check_int4_gebp(8, 64, 16, 32, true);
        check_int4_gebp(13, 96, 37, 32, false);
        check_int4_gebp(13, 96, 37, 32, true);
        check_int4_gebp(9, 300, 33, 32, true);
        check_int4_gebp(8, 128, 48, 16, false);
        check_int4_gebp(8, 128, 48, 128, true);
    }

    /// The vectorized panel fill must reproduce the per-column scalar fill
    /// exactly. The end-to-end test above would catch a gross error, but this
    /// pins the packer itself, so a bug that happens to cancel in the GEMM --
    /// or one that only shows in a padding lane the microkernel ignores --
    /// still fails.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int4_dequant_panel_is_bit_identical_to_the_per_column_path() {
        if !has_simd_x86() {
            return;
        }
        // `nr`: two whole groups, one whole group, a group plus every scalar
        // tail width, and a panel too narrow for any group at all.
        // `kc`: multiples of eight and every remainder, including one longer
        // than a block so the run has to re-hoist scale and zero point.
        // `block_size`: 8/24/32/40/128 take the vector path; 2 and 4 are not
        // multiples of the group and must fall back to the scalar one. 24 and
        // 40 are multiples of the group that do *not* divide `KC`, which is
        // what pins the block-scoped hoist: it walks whole groups inside one
        // block, so `block_size - pc % block_size` has to stay a multiple of
        // the group even when the block does not tile `pc` evenly.
        let k = 256usize;
        let n = 32usize;
        for &block_size in &[2usize, 4, 8, 24, 32, 40, 128] {
            for &asymmetric in &[false, true] {
                let (packed, scales, zero_points, _) = int4_weight(k, n, block_size, asymmetric);
                let weight = Int4Weight {
                    packed: &packed,
                    zero_points: zero_points.as_deref(),
                    block_size,
                    block_count: k.div_ceil(block_size),
                };
                let scale_at = |index: usize| scales[index];
                for &nr in &[1usize, 5, 7, 8, 9, 13, 15, 16] {
                    for &kc in &[1usize, 7, 8, 9, 16, 33, 64, 130] {
                        // `pc = 0` and a later panel, so the block arithmetic
                        // is exercised at a non-zero depth offset too.
                        for &pc in &[0usize, 128] {
                            if pc + kc > k {
                                continue;
                            }
                            let jcol = 8usize;
                            let mut expect = vec![f32::NAN; kc * NR];
                            for slot in 0..nr {
                                weight.dequant_column(
                                    &scale_at,
                                    jcol + slot,
                                    pc,
                                    kc,
                                    slot,
                                    &mut expect,
                                );
                            }
                            let mut got = vec![f32::NAN; kc * NR];
                            weight.dequant_panel(&scale_at, jcol, pc, kc, nr, &mut got);
                            for p in 0..kc {
                                for slot in 0..nr {
                                    let (e, g) = (expect[p * NR + slot], got[p * NR + slot]);
                                    assert_eq!(
                                        e.to_bits(),
                                        g.to_bits(),
                                        "panel mismatch at depth {p} slot {slot} \
                                         (block={block_size}, asym={asymmetric}, nr={nr}, \
                                         kc={kc}, pc={pc}): {e} vs {g}"
                                    );
                                }
                                // Columns past `nr` are the caller's
                                // zero-padding, so the packer must not have
                                // touched them.
                                for slot in nr..NR {
                                    assert!(
                                        got[p * NR + slot].is_nan(),
                                        "packer wrote past nr={nr} at depth {p} slot {slot}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int4_gebp_handles_a_k_that_is_not_a_whole_block() {
        // k = 70 with block_size 32 leaves a 6-deep tail block; the packed row
        // still carries a full blob, so only the valid depths may be read.
        check_int4_gebp(8, 70, 32, 32, false);
        check_int4_gebp(8, 70, 32, 32, true);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn int4_gebp_degenerate_shapes_are_bias_only_or_empty() {
        if !has_simd_x86() {
            return;
        }
        let weight = Int4Weight {
            packed: &[],
            zero_points: None,
            block_size: 32,
            block_count: 0,
        };
        let bias = vec![1.5f32, -2.0];
        let mut c = vec![7.0f32; 2 * 2];
        quant_prefill_gebp(&[], &weight, |_| 0.0, Some(&bias), &mut c, 2, 0, 2);
        assert_eq!(c, vec![1.5, -2.0, 1.5, -2.0], "k=0 leaves only the bias");

        let mut empty: Vec<f32> = Vec::new();
        quant_prefill_gebp(&[], &weight, |_| 0.0, None, &mut empty, 0, 0, 0);
        assert!(empty.is_empty());
    }

    /// Round an f32 to the given 16-bit format and back, so a reference built
    /// from the f32 values sees exactly the operand the kernel sees.
    fn narrow(format: HalfFormat, value: f32) -> (u16, f32) {
        match format {
            HalfFormat::F16 => {
                let bits = half::f16::from_f32(value);
                (bits.to_bits(), bits.to_f32())
            }
            HalfFormat::Bf16 => {
                let bits = half::bf16::from_f32(value);
                (bits.to_bits(), bits.to_f32())
            }
        }
    }

    fn half_operand(format: HalfFormat, len: usize, seed: usize) -> (Vec<u16>, Vec<f32>) {
        fill(len, seed)
            .into_iter()
            .map(|value| narrow(format, value))
            .unzip()
    }

    /// The fused widen-pack GEBP must be *bit-identical* to the f32 kernel run
    /// on the widened operands: it builds the same panels from the same values
    /// and drives the same microkernel, so any difference is a packing bug, not
    /// float reassociation.
    fn check_half(format: HalfFormat, m: usize, k: usize, n: usize) {
        if !has_simd_x86() {
            return; // No AVX2/FMA: this path is never selected.
        }
        let (a_bits, a_wide) = half_operand(format, m * k, 1);
        let (b_bits, b_wide) = half_operand(format, k * n, 2);

        let mut expect = vec![0.0f32; m * n];
        sgemm_simd(&a_wide, &b_wide, &mut expect, m, k, n);

        let mut got = vec![0.0f32; m * n];
        half_prefill_gebp(format, &a_bits, &b_bits, &mut got, m, k, n);

        assert_eq!(
            got, expect,
            "{format:?} {m}x{k}x{n}: fused widen-pack GEBP must match the f32 \
             kernel on the widened operands bit for bit"
        );
    }

    #[test]
    fn half_gebp_matches_the_widened_f32_kernel() {
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            check_half(format, 2, 64, 32);
            check_half(format, 8, 128, 64);
            check_half(format, 64, 96, 48);
        }
    }

    #[test]
    fn half_gebp_handles_tail_shapes() {
        // M/N/K not multiples of MR(6)/NR(16)/KC(256), and a K larger than one
        // K-panel so the accumulate-across-panels path runs.
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            check_half(format, 7, 33, 17);
            check_half(format, 1, 5, 3);
            check_half(format, 5, 300, 19);
            check_half(format, 13, 260, 271);
        }
    }

    #[test]
    fn half_gebp_degenerate_shapes_write_nothing() {
        if !has_simd_x86() {
            return;
        }
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            let mut c = Vec::new();
            half_prefill_gebp(format, &[], &[], &mut c, 0, 4, 4);
            assert!(c.is_empty());

            // k == 0 is a zero-length reduction: C is all zeros, not untouched.
            let mut c = vec![7.0f32; 6];
            half_prefill_gebp(format, &[], &[], &mut c, 2, 0, 3);
            assert_eq!(c, vec![0.0f32; 6]);
        }
    }

    /// Pack `B` one element at a time through the `half` crate: the reference
    /// the vector conversion has to reproduce.
    ///
    /// Deliberately *not* `pack_b_half_scalar`, which routes through
    /// `half_gemm::widen_contiguous` and therefore vectorizes on any host that
    /// can -- comparing that against the SIMD path would compare SIMD with
    /// SIMD and prove nothing about the fallback.
    #[allow(clippy::too_many_arguments)]
    fn pack_b_half_reference(
        format: HalfFormat,
        b: &[u16],
        bpack: &mut [f32],
        n: usize,
        pc: usize,
        kc: usize,
        j0: usize,
        nc: usize,
    ) {
        for jp in 0..nc.div_ceil(NR) {
            let jcol = j0 + jp * NR;
            let nr = NR.min(nc - jp * NR);
            let dst = &mut bpack[jp * KC * NR..jp * KC * NR + kc * NR];
            for p in 0..kc {
                for c in 0..NR {
                    dst[p * NR + c] = if c < nr {
                        let bits = b[(pc + p) * n + jcol + c];
                        match format {
                            HalfFormat::F16 => half::f16::from_bits(bits).to_f32(),
                            HalfFormat::Bf16 => half::bf16::from_bits(bits).to_f32(),
                        }
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    /// The vector conversion must produce the same panels as widening one
    /// element at a time through `half` -- otherwise the bit-exactness claim
    /// only holds on the machine it was measured on, and the scalar fallback
    /// silently computes something else.
    #[test]
    fn packing_matches_the_element_at_a_time_reference() {
        let (n, kc, nc) = (48usize, 5usize, 48usize);
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            let (b_bits, _) = half_operand(format, 8 * n, 3);
            let mut reference = vec![0.0f32; KC * nc.div_ceil(NR) * NR];
            let mut scalar = reference.clone();
            let mut simd = reference.clone();
            pack_b_half_reference(format, &b_bits, &mut reference, n, 1, kc, 0, nc);
            pack_b_half_scalar(format, &b_bits, &mut scalar, n, 1, kc, 0, nc);
            pack_b_half(
                select_half_widen(format),
                &b_bits,
                &mut simd,
                n,
                1,
                kc,
                0,
                nc,
            );
            assert_eq!(reference, scalar, "{format:?} fallback packing diverged");
            assert_eq!(reference, simd, "{format:?} packing must not depend on ISA");
        }
    }

    /// Sweep the *entire* 16-bit domain -- NaN, infinity and denormals
    /// included -- through the packing the kernel actually uses, and compare
    /// against `half`.
    ///
    /// `f16` must agree on all 65536 patterns. `bf16` must agree on every
    /// pattern that is not a signalling NaN; on those 126 the shift keeps the
    /// payload where `half` canonicalizes to a quiet NaN, which changes no
    /// finite value and still propagates NaN. That divergence is inherited
    /// from `half_gemm`'s own widening, and this test is where it is written
    /// down rather than left to be discovered.
    #[test]
    fn widening_matches_the_half_crate_over_the_whole_domain() {
        let (n, nc) = (NR, NR);
        let all_bits: Vec<u16> = (0..=u16::MAX).collect();
        let rows = all_bits.len() / n;
        for format in [HalfFormat::F16, HalfFormat::Bf16] {
            let mut divergent = 0usize;
            for pc in (0..rows).step_by(KC) {
                let kc = KC.min(rows - pc);
                let mut reference = vec![0.0f32; KC * NR];
                let mut simd = reference.clone();
                pack_b_half_reference(format, &all_bits, &mut reference, n, pc, kc, 0, nc);
                pack_b_half(
                    select_half_widen(format),
                    &all_bits,
                    &mut simd,
                    n,
                    pc,
                    kc,
                    0,
                    nc,
                );
                for (got, want) in simd.iter().zip(&reference) {
                    if got.to_bits() == want.to_bits() {
                        continue;
                    }
                    assert!(
                        got.is_nan() && want.is_nan(),
                        "{format:?}: {got} != {want} on a non-NaN input"
                    );
                    divergent += 1;
                }
            }
            let expected = match format {
                HalfFormat::F16 => 0,
                // The bf16 signalling-NaN patterns: sign x (payload != 0) with
                // the quiet bit clear, i.e. 2 * (2^6 - 1) = 126.
                HalfFormat::Bf16 => 126,
            };
            assert_eq!(
                divergent, expected,
                "{format:?}: NaN-encoding divergences moved; the doc comment on \
                 `half_prefill_gebp` states this count"
            );
        }
    }
}
