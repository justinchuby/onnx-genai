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
/// M==1 selection of the native GEMV ([`sgemm_simd_m1`]) is read once from the
/// `ONNX_GENAI_CPU_MM_SIMD_M1_GEMV` toggle here; the actual dispatch lives in
/// [`sgemm_simd_variant`] so the A/B harness can drive both variants in one
/// process without touching process-global env state.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn sgemm_simd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    sgemm_simd_variant(a, b, c, m, k, n, m1_gemv_enabled());
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
    // place: no pack, no resident buffer, strictly less memory traffic. Behind
    // a same-binary A/B toggle (default off) like #1104's nblk kernel, because
    // it reassociates the f32 sum versus the packed path.
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

/// A/B toggle for the #1091 native M=1 GEMV that absorbs MLAS's `SgemmKernelM1`
/// mechanism (stream B in place, no packed buffer) into the built-in `SimdX86`
/// backend. `1`/`on` enables it; unset or `0`/`off` keeps the packed GEBP path
/// for every M. Default off so the shipped `SimdX86` path is unchanged until
/// the win is measured, exactly like the `ONNX_GENAI_CPU_MM_INT4_NBLK` (#1104)
/// and `NXRT_CPU_GEMM_BACKEND` toggles that preceded it. This is a read-only
/// env probe — production is the only writer of process state here.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn m1_gemv_enabled() -> bool {
    std::env::var("ONNX_GENAI_CPU_MM_SIMD_M1_GEMV")
        .ok()
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("off")
        })
        .unwrap_or(false)
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
    /// tolerance across tile-exact, tail, and multi-cache-line N shapes. Calls
    /// the kernel directly (no env mutation) so it does not depend on the
    /// process-global `ONNX_GENAI_CPU_MM_SIMD_M1_GEMV` toggle.
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

    /// The M=1 GEMV taken through the public `sgemm_simd` entry (with the toggle
    /// on) must agree with the packed path (toggle off) within f32 tolerance —
    /// they differ only by summation reassociation, never in which products are
    /// summed. Guarded so no other test observes the env change: the toggle is
    /// set and cleared within this test only, and `sgemm_simd` reads it once.
    #[test]
    fn m1_route_matches_packed_within_tolerance() {
        if !has_simd_x86() {
            return;
        }
        let (k, n) = (300usize, 517usize);
        let a = fill(k, 5);
        let b = fill(k * n, 9);
        let mut packed = vec![0.0f32; n];
        sgemm_simd(&a, &b, &mut packed, 1, k, n);
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
}
