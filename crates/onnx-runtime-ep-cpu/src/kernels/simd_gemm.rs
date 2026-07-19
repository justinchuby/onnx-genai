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
const MR: usize = 6;
/// Register-tile columns (two `__m256` lanes).
const NR: usize = 16;
/// K-panel width kept L1-resident.
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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn sgemm_simd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for v in c.iter_mut() {
            *v = 0.0;
        }
        return;
    }

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
}
