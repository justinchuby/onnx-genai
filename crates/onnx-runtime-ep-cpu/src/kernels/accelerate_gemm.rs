//! Apple Accelerate + NEON GEMV for the M=1 decode hot path.
//!
//! Three routes, selected per-op by matrix geometry:
//!
//! - `sgemm`: Accelerate `cblas_sgemm` for M>1 prefill (reaches AMX).
//! - `neon_gemv_col_parallel`: Column-parallel NEON GEMV on pre-transposed B
//!   for M=1 decode (65-93 GB/s with Rayon on M1 Max).
//! - `neon_gemv_parallel`: Row-parallel NEON GEMV fallback for M=1 when
//!   pre-transposed B is not available.

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// C[m,n] = A[m,k] @ B[k,n] via Accelerate `cblas_sgemm`.
pub fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

// ─── Column-parallel GEMV on pre-transposed B ───────────────────────

/// Column-parallel NEON GEMV using pre-transposed B_T[N,K] row-major.
///
/// Computes y[n] = x[k] @ B[k,n] where B_T[n,k] is the cached transpose
/// of the weight matrix. Each thread processes a distinct slice of output
/// columns via independent dot products — no partial sums, no reduction,
/// no per-call allocation.
///
/// Dispatch priority:
/// 1. Persistent SPMD pool (near-zero barrier latency, workers already hot).
/// 2. Rayon `par_chunks_mut` (fork-join fallback).
/// 3. Single-threaded for small matrices where dispatch > compute.
pub fn neon_gemv_col_parallel(x: &[f32], bt: &[f32], y: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(bt.len(), n * k);
    debug_assert_eq!(y.len(), n);

    // Single-threaded for small matrices (dispatch > compute time).
    if k * n < 500_000 || rayon::current_num_threads() <= 1 {
        neon_gemv_batch(bt, x, y, n, k);
        return;
    }

    // Prefer the persistent SPMD pool when active: workers are already hot
    // (spin-then-park), and the sense-reversing barrier is ~50 ns vs Rayon's
    // ~2-5 µs fork-join overhead per call.
    if let Some(spmd) = crate::kernels::matmul_nbits::spmd_decode_active() {
        spmd.dispatch_output_rows(y, k, &|start, outputs| {
            neon_gemv_batch(&bt[start * k..], x, outputs, outputs.len(), k);
        });
        return;
    }

    // Rayon par_chunks_mut: fork-join over contiguous output slices.
    use rayon::prelude::*;
    let threads = rayon::current_num_threads();
    let chunk = n.div_ceil(threads).max(1);
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(t, y_chunk)| {
            let n0 = t * chunk;
            neon_gemv_batch(&bt[n0 * k..], x, y_chunk, y_chunk.len(), k);
        });
}

/// 4-row batched NEON GEMV: compute 4 dot products simultaneously, sharing
/// x reads across rows to improve ILP and memory bandwidth utilization.
/// Measured 24-35% faster than 1-row-at-a-time on DRAM-bound shapes.
#[cfg(target_arch = "aarch64")]
fn neon_gemv_batch(bt: &[f32], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    use std::arch::aarch64::*;
    let mut i = 0;
    // Process 4 output rows at a time: read x once, compute 4 dot products.
    // Uses 8 independent accumulators (2 per row) to fill the FMA pipeline.
    while i + 4 <= n {
        let (r0, r1, r2, r3) = (
            &bt[i * k..],
            &bt[(i + 1) * k..],
            &bt[(i + 2) * k..],
            &bt[(i + 3) * k..],
        );
        let mut a0 = unsafe { vdupq_n_f32(0.0) };
        let mut a1 = unsafe { vdupq_n_f32(0.0) };
        let mut a2 = unsafe { vdupq_n_f32(0.0) };
        let mut a3 = unsafe { vdupq_n_f32(0.0) };
        let mut b0 = unsafe { vdupq_n_f32(0.0) };
        let mut b1 = unsafe { vdupq_n_f32(0.0) };
        let mut b2 = unsafe { vdupq_n_f32(0.0) };
        let mut b3 = unsafe { vdupq_n_f32(0.0) };
        let mut j = 0;
        while j + 8 <= k {
            unsafe {
                let x0 = vld1q_f32(x.as_ptr().add(j));
                let x1 = vld1q_f32(x.as_ptr().add(j + 4));
                a0 = vfmaq_f32(a0, vld1q_f32(r0.as_ptr().add(j)), x0);
                b0 = vfmaq_f32(b0, vld1q_f32(r0.as_ptr().add(j + 4)), x1);
                a1 = vfmaq_f32(a1, vld1q_f32(r1.as_ptr().add(j)), x0);
                b1 = vfmaq_f32(b1, vld1q_f32(r1.as_ptr().add(j + 4)), x1);
                a2 = vfmaq_f32(a2, vld1q_f32(r2.as_ptr().add(j)), x0);
                b2 = vfmaq_f32(b2, vld1q_f32(r2.as_ptr().add(j + 4)), x1);
                a3 = vfmaq_f32(a3, vld1q_f32(r3.as_ptr().add(j)), x0);
                b3 = vfmaq_f32(b3, vld1q_f32(r3.as_ptr().add(j + 4)), x1);
            }
            j += 8;
        }
        // Reduce NEON accumulators and add scalar tail for remainder elements.
        let mut s0 = unsafe { vaddvq_f32(vaddq_f32(a0, b0)) };
        let mut s1 = unsafe { vaddvq_f32(vaddq_f32(a1, b1)) };
        let mut s2 = unsafe { vaddvq_f32(vaddq_f32(a2, b2)) };
        let mut s3 = unsafe { vaddvq_f32(vaddq_f32(a3, b3)) };
        while j < k {
            let xv = x[j];
            s0 += r0[j] * xv;
            s1 += r1[j] * xv;
            s2 += r2[j] * xv;
            s3 += r3[j] * xv;
            j += 1;
        }
        y[i] = s0;
        y[i + 1] = s1;
        y[i + 2] = s2;
        y[i + 3] = s3;
        i += 4;
    }
    // Remainder rows (fewer than 4)
    while i < n {
        y[i] = neon_dot(&bt[i * k..(i + 1) * k], x);
        i += 1;
    }
}

/// Scalar fallback for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
fn neon_gemv_batch(bt: &[f32], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    for i in 0..n {
        y[i] = neon_dot(&bt[i * k..(i + 1) * k], x);
    }
}

/// NEON 4×-unrolled dot product: `sum(a[i] * b[i])`.
#[cfg(target_arch = "aarch64")]
fn neon_dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(a.len(), b.len());
    let k = a.len();
    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };
    let mut j = 0;
    while j + 16 <= k {
        unsafe {
            acc0 = vfmaq_f32(
                acc0,
                vld1q_f32(a.as_ptr().add(j)),
                vld1q_f32(b.as_ptr().add(j)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vld1q_f32(a.as_ptr().add(j + 4)),
                vld1q_f32(b.as_ptr().add(j + 4)),
            );
            acc2 = vfmaq_f32(
                acc2,
                vld1q_f32(a.as_ptr().add(j + 8)),
                vld1q_f32(b.as_ptr().add(j + 8)),
            );
            acc3 = vfmaq_f32(
                acc3,
                vld1q_f32(a.as_ptr().add(j + 12)),
                vld1q_f32(b.as_ptr().add(j + 12)),
            );
        }
        j += 16;
    }
    let mut sum = unsafe {
        let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(s)
    };
    while j < k {
        sum += a[j] * b[j];
        j += 1;
    }
    sum
}

/// Scalar fallback for non-aarch64 targets (correctness reference).
#[cfg(not(target_arch = "aarch64"))]
fn neon_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

// ─── Row-parallel GEMV fallback (when pre-transposed B unavailable) ──

/// Rayon-parallel NEON GEMV: y[n] = x[k] @ B[k,n] (M=1 decode fallback).
///
/// Row-parallel: partitions K so each thread reads contiguous B rows.
/// Uses 4\u00d7 K-unrolled outer product to reduce partial-sum traffic.
pub fn neon_gemv_parallel(x: &[f32], b: &[f32], y: &mut [f32], k: usize, n: usize) {
    use rayon::prelude::*;

    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(y.len(), n);

    let threads = rayon::current_num_threads();

    // For small matrices, Accelerate sgemm has lower overhead.
    if threads <= 1 || k * n < 500_000 {
        sgemm(x, b, y, 1, k, n);
        return;
    }

    let rows_per_thread = k.div_ceil(threads);

    let tasks: Vec<(usize, usize)> = (0..threads)
        .map(|t| {
            let p0 = t * rows_per_thread;
            let p_end = ((t + 1) * rows_per_thread).min(k);
            (p0, p_end)
        })
        .filter(|&(p0, p_end)| p0 < p_end)
        .collect();

    let partials: Vec<Vec<f32>> = tasks
        .into_par_iter()
        .map(|(p0, p_end)| {
            let mut partial = vec![0.0f32; n];
            #[cfg(target_arch = "aarch64")]
            neon_outer_product_unrolled(x, b, &mut partial, n, p0, p_end);
            #[cfg(not(target_arch = "aarch64"))]
            scalar_outer_product(x, b, &mut partial, n, p0, p_end);
            partial
        })
        .collect();

    // Reduce
    for v in y.iter_mut() {
        *v = 0.0;
    }
    for partial in &partials {
        #[cfg(target_arch = "aarch64")]
        neon_vector_add(y, partial);
        #[cfg(not(target_arch = "aarch64"))]
        for (yi, pi) in y.iter_mut().zip(partial.iter()) {
            *yi += pi;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn neon_vector_add(dst: &mut [f32], src: &[f32]) {
    use std::arch::aarch64::*;
    let n = dst.len();
    let mut j = 0usize;
    while j + 16 <= n {
        unsafe {
            let d0 = vld1q_f32(dst.as_ptr().add(j));
            let d1 = vld1q_f32(dst.as_ptr().add(j + 4));
            let d2 = vld1q_f32(dst.as_ptr().add(j + 8));
            let d3 = vld1q_f32(dst.as_ptr().add(j + 12));
            let s0 = vld1q_f32(src.as_ptr().add(j));
            let s1 = vld1q_f32(src.as_ptr().add(j + 4));
            let s2 = vld1q_f32(src.as_ptr().add(j + 8));
            let s3 = vld1q_f32(src.as_ptr().add(j + 12));
            vst1q_f32(dst.as_mut_ptr().add(j), vaddq_f32(d0, s0));
            vst1q_f32(dst.as_mut_ptr().add(j + 4), vaddq_f32(d1, s1));
            vst1q_f32(dst.as_mut_ptr().add(j + 8), vaddq_f32(d2, s2));
            vst1q_f32(dst.as_mut_ptr().add(j + 12), vaddq_f32(d3, s3));
        }
        j += 16;
    }
    while j < n {
        dst[j] += src[j];
        j += 1;
    }
}

/// 4\u00d7 K-unrolled NEON outer product: partial[j] += x[row] * B[row,j].
#[cfg(target_arch = "aarch64")]
fn neon_outer_product_unrolled(
    x: &[f32],
    b: &[f32],
    partial: &mut [f32],
    n: usize,
    p0: usize,
    p_end: usize,
) {
    use std::arch::aarch64::*;
    let mut row = p0;
    // Process 4 rows at a time to reduce partial-buffer traffic.
    while row + 4 <= p_end {
        let xv0 = unsafe { vdupq_n_f32(x[row]) };
        let xv1 = unsafe { vdupq_n_f32(x[row + 1]) };
        let xv2 = unsafe { vdupq_n_f32(x[row + 2]) };
        let xv3 = unsafe { vdupq_n_f32(x[row + 3]) };
        let b0 = &b[row * n..];
        let b1 = &b[(row + 1) * n..];
        let b2 = &b[(row + 2) * n..];
        let b3 = &b[(row + 3) * n..];
        let mut j = 0usize;
        while j + 4 <= n {
            unsafe {
                let p = vld1q_f32(partial.as_ptr().add(j));
                let r = vfmaq_f32(
                    vfmaq_f32(
                        vfmaq_f32(
                            vfmaq_f32(p, vld1q_f32(b0.as_ptr().add(j)), xv0),
                            vld1q_f32(b1.as_ptr().add(j)),
                            xv1,
                        ),
                        vld1q_f32(b2.as_ptr().add(j)),
                        xv2,
                    ),
                    vld1q_f32(b3.as_ptr().add(j)),
                    xv3,
                );
                vst1q_f32(partial.as_mut_ptr().add(j), r);
            }
            j += 4;
        }
        while j < n {
            partial[j] +=
                x[row] * b0[j] + x[row + 1] * b1[j] + x[row + 2] * b2[j] + x[row + 3] * b3[j];
            j += 1;
        }
        row += 4;
    }
    // Remainder rows
    while row < p_end {
        let xval = x[row];
        let xv = unsafe { vdupq_n_f32(xval) };
        let b_row = &b[row * n..];
        let mut j = 0usize;
        while j + 4 <= n {
            unsafe {
                let p = vld1q_f32(partial.as_ptr().add(j));
                vst1q_f32(
                    partial.as_mut_ptr().add(j),
                    vfmaq_f32(p, vld1q_f32(b_row.as_ptr().add(j)), xv),
                );
            }
            j += 4;
        }
        while j < n {
            partial[j] += xval * b_row[j];
            j += 1;
        }
        row += 1;
    }
}

/// Scalar outer product fallback for non-aarch64 targets.
#[allow(dead_code)]
#[cfg(not(target_arch = "aarch64"))]
fn scalar_outer_product(
    x: &[f32],
    b: &[f32],
    partial: &mut [f32],
    n: usize,
    p0: usize,
    p_end: usize,
) {
    for row in p0..p_end {
        let xval = x[row];
        let b_row = &b[row * n..row * n + n];
        for j in 0..n {
            partial[j] += xval * b_row[j];
        }
    }
}

// ─── FP16 storage GEMV (f16 weights, f32 activations, f32 accumulate) ─

/// Load 4 × f16 (8 bytes) from `ptr` and widen to `float32x4_t`.
///
/// Uses `fcvtl` which is ARMv8 base FP — no FEAT_FP16 required.
/// `vld1_f16`/`vcvt_f32_f16` intrinsics need unstable `f16` type;
/// this inline-asm wrapper avoids the nightly dependency.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_f16x4_to_f32x4(ptr: *const u16) -> std::arch::aarch64::float32x4_t {
    let result: std::arch::aarch64::float32x4_t;
    unsafe {
        // TODO: replace this asm with `vld1_f16`/`vcvt_f32_f16` or the stable
        // f16 widening intrinsic once Rust's `f16` type and aarch64 f16
        // conversion intrinsics stabilize. Chew verified bit-exactness vs
        // scalar edge cases; see `.squad/decisions/inbox/chew-pr227-fp16-review.md`.
        std::arch::asm!(
            "ldr {v:d}, [{ptr}]",
            "fcvtl {v:v}.4s, {v:v}.4h",
            ptr = in(reg) ptr,
            v = out(vreg) result,
            options(nostack, readonly, pure),
        );
    }
    result
}

/// Column-parallel NEON GEMV with f16 weight storage and f32 accumulate.
///
/// Same dispatch priority as [`neon_gemv_col_parallel`] (SPMD pool → Rayon →
/// single-threaded) but reads pre-transposed B_T in f16 directly from the
/// mmap'd model file, halving memory bandwidth. Weights are widened to f32
/// in NEON registers via `fcvtl` (ARMv8 base, safe M1–M4) and accumulated
/// in f32, giving ~2.3e-4 max relative error (same as FP16 storage + FP32
/// accumulate reference).
///
/// `bt_f16` is `N×K` row-major with each element stored as a raw `u16`
/// (`half::f16` bit pattern). `x` is the f32 activation vector of length K.
pub fn neon_gemv_f16_col_parallel(x: &[f32], bt_f16: &[u16], y: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(bt_f16.len(), n * k);
    debug_assert_eq!(y.len(), n);

    if k * n < 500_000 || rayon::current_num_threads() <= 1 {
        neon_gemv_f16_batch(bt_f16, x, y, n, k);
        return;
    }

    if let Some(spmd) = crate::kernels::matmul_nbits::spmd_decode_active() {
        spmd.dispatch_output_rows(y, k, &|start, outputs| {
            neon_gemv_f16_batch(&bt_f16[start * k..], x, outputs, outputs.len(), k);
        });
        return;
    }

    use rayon::prelude::*;
    let threads = rayon::current_num_threads();
    let chunk = n.div_ceil(threads).max(1);
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(t, y_chunk)| {
            let n0 = t * chunk;
            neon_gemv_f16_batch(&bt_f16[n0 * k..], x, y_chunk, y_chunk.len(), k);
        });
}

/// 4-row batched NEON GEMV with f16 weights: compute 4 dot products
/// simultaneously, loading f16 weights via `fcvtl` → f32 and sharing
/// f32 activation reads across rows. F32 accumulation throughout.
#[cfg(target_arch = "aarch64")]
fn neon_gemv_f16_batch(bt_f16: &[u16], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    use std::arch::aarch64::*;
    let mut i = 0;
    while i + 4 <= n {
        let (r0, r1, r2, r3) = (
            &bt_f16[i * k..],
            &bt_f16[(i + 1) * k..],
            &bt_f16[(i + 2) * k..],
            &bt_f16[(i + 3) * k..],
        );
        let mut a0 = unsafe { vdupq_n_f32(0.0) };
        let mut a1 = unsafe { vdupq_n_f32(0.0) };
        let mut a2 = unsafe { vdupq_n_f32(0.0) };
        let mut a3 = unsafe { vdupq_n_f32(0.0) };
        let mut b0 = unsafe { vdupq_n_f32(0.0) };
        let mut b1 = unsafe { vdupq_n_f32(0.0) };
        let mut b2 = unsafe { vdupq_n_f32(0.0) };
        let mut b3 = unsafe { vdupq_n_f32(0.0) };
        let mut j = 0;
        // Main loop: 8 elements per iteration (2 × 4 f16 loads per row).
        while j + 8 <= k {
            unsafe {
                let x0 = vld1q_f32(x.as_ptr().add(j));
                let x1 = vld1q_f32(x.as_ptr().add(j + 4));
                // Row 0: load 8 f16 → 2×4 f32, FMA with activations.
                let w0a = load_f16x4_to_f32x4(r0.as_ptr().add(j));
                let w0b = load_f16x4_to_f32x4(r0.as_ptr().add(j + 4));
                a0 = vfmaq_f32(a0, w0a, x0);
                b0 = vfmaq_f32(b0, w0b, x1);
                // Row 1
                let w1a = load_f16x4_to_f32x4(r1.as_ptr().add(j));
                let w1b = load_f16x4_to_f32x4(r1.as_ptr().add(j + 4));
                a1 = vfmaq_f32(a1, w1a, x0);
                b1 = vfmaq_f32(b1, w1b, x1);
                // Row 2
                let w2a = load_f16x4_to_f32x4(r2.as_ptr().add(j));
                let w2b = load_f16x4_to_f32x4(r2.as_ptr().add(j + 4));
                a2 = vfmaq_f32(a2, w2a, x0);
                b2 = vfmaq_f32(b2, w2b, x1);
                // Row 3
                let w3a = load_f16x4_to_f32x4(r3.as_ptr().add(j));
                let w3b = load_f16x4_to_f32x4(r3.as_ptr().add(j + 4));
                a3 = vfmaq_f32(a3, w3a, x0);
                b3 = vfmaq_f32(b3, w3b, x1);
            }
            j += 8;
        }
        let mut s0 = unsafe { vaddvq_f32(vaddq_f32(a0, b0)) };
        let mut s1 = unsafe { vaddvq_f32(vaddq_f32(a1, b1)) };
        let mut s2 = unsafe { vaddvq_f32(vaddq_f32(a2, b2)) };
        let mut s3 = unsafe { vaddvq_f32(vaddq_f32(a3, b3)) };
        // Scalar tail: widen remaining f16 elements individually.
        while j < k {
            let xv = x[j];
            s0 += half::f16::from_bits(r0[j]).to_f32() * xv;
            s1 += half::f16::from_bits(r1[j]).to_f32() * xv;
            s2 += half::f16::from_bits(r2[j]).to_f32() * xv;
            s3 += half::f16::from_bits(r3[j]).to_f32() * xv;
            j += 1;
        }
        y[i] = s0;
        y[i + 1] = s1;
        y[i + 2] = s2;
        y[i + 3] = s3;
        i += 4;
    }
    while i < n {
        y[i] = neon_dot_f16(&bt_f16[i * k..(i + 1) * k], x);
        i += 1;
    }
}

/// Scalar fallback for non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
fn neon_gemv_f16_batch(bt_f16: &[u16], x: &[f32], y: &mut [f32], n: usize, k: usize) {
    for i in 0..n {
        y[i] = neon_dot_f16(&bt_f16[i * k..(i + 1) * k], x);
    }
}

/// NEON dot product: f16 weights × f32 activations → f32 scalar.
#[cfg(target_arch = "aarch64")]
fn neon_dot_f16(a_f16: &[u16], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(a_f16.len(), b.len());
    let k = a_f16.len();
    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };
    let mut j = 0;
    while j + 16 <= k {
        unsafe {
            acc0 = vfmaq_f32(
                acc0,
                load_f16x4_to_f32x4(a_f16.as_ptr().add(j)),
                vld1q_f32(b.as_ptr().add(j)),
            );
            acc1 = vfmaq_f32(
                acc1,
                load_f16x4_to_f32x4(a_f16.as_ptr().add(j + 4)),
                vld1q_f32(b.as_ptr().add(j + 4)),
            );
            acc2 = vfmaq_f32(
                acc2,
                load_f16x4_to_f32x4(a_f16.as_ptr().add(j + 8)),
                vld1q_f32(b.as_ptr().add(j + 8)),
            );
            acc3 = vfmaq_f32(
                acc3,
                load_f16x4_to_f32x4(a_f16.as_ptr().add(j + 12)),
                vld1q_f32(b.as_ptr().add(j + 12)),
            );
        }
        j += 16;
    }
    let mut sum = unsafe {
        let s = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        vaddvq_f32(s)
    };
    while j < k {
        sum += half::f16::from_bits(a_f16[j]).to_f32() * b[j];
        j += 1;
    }
    sum
}

/// Scalar dot fallback for non-aarch64.
#[cfg(not(target_arch = "aarch64"))]
fn neon_dot_f16(a_f16: &[u16], b: &[f32]) -> f32 {
    a_f16
        .iter()
        .zip(b)
        .map(|(&a, &b)| half::f16::from_bits(a).to_f32() * b)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_gemv(x: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; n];
        for j in 0..n {
            for i in 0..k {
                y[j] += x[i] * b[i * n + j];
            }
        }
        y
    }

    #[test]
    fn sgemm_matches_reference() {
        let (m, k, n) = (4, 8, 6);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05).collect();
        let mut c = vec![0.0f32; m * n];
        sgemm(&a, &b, &mut c, m, k, n);
        // Verify against naive
        for i in 0..m {
            for j in 0..n {
                let expected: f32 = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-3,
                    "sgemm[{i},{j}]: got {}, expected {expected}",
                    c[i * n + j]
                );
            }
        }
    }

    #[test]
    fn col_parallel_gemv_matches_reference() {
        let (k, n) = (64, 128);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 100) as f32) * 0.001).collect();
        // Transpose b
        let mut bt = vec![0.0f32; n * k];
        for i in 0..k {
            for j in 0..n {
                bt[j * k + i] = b[i * n + j];
            }
        }
        let mut y = vec![0.0f32; n];
        neon_gemv_col_parallel(&x, &bt, &mut y, k, n);
        let ref_y = reference_gemv(&x, &b, k, n);
        let max_err = y
            .iter()
            .zip(&ref_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "col_parallel_gemv max error {max_err} exceeds 1e-4"
        );
    }

    #[test]
    fn row_parallel_gemv_matches_reference() {
        let (k, n) = (64, 128);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 100) as f32) * 0.001).collect();
        let mut y = vec![0.0f32; n];
        neon_gemv_parallel(&x, &b, &mut y, k, n);
        let ref_y = reference_gemv(&x, &b, k, n);
        let max_err = y
            .iter()
            .zip(&ref_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "row_parallel_gemv max error {max_err} exceeds 1e-4"
        );
    }

    #[test]
    fn col_parallel_matches_at_model_scale() {
        let (k, n) = (896, 4864);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 1000) as f32) * 0.0001).collect();
        let mut bt = vec![0.0f32; n * k];
        for i in 0..k {
            for j in 0..n {
                bt[j * k + i] = b[i * n + j];
            }
        }
        let mut y_col = vec![0.0f32; n];
        neon_gemv_col_parallel(&x, &bt, &mut y_col, k, n);
        let ref_y = reference_gemv(&x, &b, k, n);
        let max_rel = y_col
            .iter()
            .zip(&ref_y)
            .filter(|(_, r)| r.abs() > 1e-6)
            .map(|(a, r)| ((a - r) / r).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_rel < 0.02,
            "col_parallel model-scale max relative error {max_rel} exceeds 2%"
        );
    }

    // ─── FP16 GEMV tests ────────────────────────────────────────────

    /// Convert f32 values to f16 bit patterns (u16).
    fn f32_to_f16_bits(vals: &[f32]) -> Vec<u16> {
        vals.iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect()
    }

    /// Reference GEMV using f64 accumulation for the gold standard,
    /// with weights rounded through f16 to match the kernel's input.
    fn reference_gemv_f16(x: &[f32], b_f16: &[u16], k: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0.0f64; n];
        for j in 0..n {
            for i in 0..k {
                let w = half::f16::from_bits(b_f16[i * n + j]).to_f32() as f64;
                y[j] += (x[i] as f64) * w;
            }
        }
        y.iter().map(|&v| v as f32).collect()
    }

    // Chew's PR #227 review measured 2.38e-7 max relative drift vs an f64
    // reference and 1.73e-6 FP16-vs-F32 GEMV parity. Use 1e-4 relative
    // (>50× measured parity) and 1e-5 absolute (>40× the odd-tail measured
    // 2.28e-7) so M1 Air/M1 Max/M4 Max worker-count differences can move rows
    // between batched and scalar-tail code paths without hiding real regressions.
    const F16_GEMV_MAX_REL_TOLERANCE: f32 = 1e-4;
    const F16_GEMV_MAX_ABS_TOLERANCE: f32 = 1e-5;

    #[test]
    fn f16_col_parallel_gemv_matches_reference() {
        let (k, n) = (64, 128);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 100) as f32) * 0.001).collect();
        let b_f16 = f32_to_f16_bits(&b);
        // Transpose b_f16[K,N] → bt_f16[N,K]
        let mut bt_f16 = vec![0u16; n * k];
        for i in 0..k {
            for j in 0..n {
                bt_f16[j * k + i] = b_f16[i * n + j];
            }
        }
        let mut y = vec![0.0f32; n];
        neon_gemv_f16_col_parallel(&x, &bt_f16, &mut y, k, n);
        let ref_y = reference_gemv_f16(&x, &b_f16, k, n);
        let max_err = y
            .iter()
            .zip(&ref_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < F16_GEMV_MAX_ABS_TOLERANCE,
            "f16 col_parallel_gemv max error {max_err} exceeds {F16_GEMV_MAX_ABS_TOLERANCE}"
        );
    }

    #[test]
    fn f16_col_parallel_matches_at_model_scale() {
        let (k, n) = (896, 4864);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 1000) as f32) * 0.0001).collect();
        let b_f16 = f32_to_f16_bits(&b);
        let mut bt_f16 = vec![0u16; n * k];
        for i in 0..k {
            for j in 0..n {
                bt_f16[j * k + i] = b_f16[i * n + j];
            }
        }
        let ref_y = reference_gemv_f16(&x, &b_f16, k, n);
        for workers in [1, 3, 7, 11] {
            let max_rel = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    let mut y = vec![0.0f32; n];
                    neon_gemv_f16_col_parallel(&x, &bt_f16, &mut y, k, n);
                    y.iter()
                        .zip(&ref_y)
                        .filter(|(_, r)| r.abs() > 1e-6)
                        .map(|(a, r)| ((a - r) / r).abs())
                        .fold(0.0f32, f32::max)
                });
            assert!(
                max_rel < F16_GEMV_MAX_REL_TOLERANCE,
                "f16 col_parallel model-scale max relative error {max_rel} with {workers} workers exceeds {F16_GEMV_MAX_REL_TOLERANCE}"
            );
        }
    }

    #[test]
    fn f16_gemv_odd_k_tail() {
        // Test with K not divisible by 8 or 16 to exercise scalar tail.
        let (k, n) = (67, 9);
        let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 50) as f32) * 0.01).collect();
        let b_f16 = f32_to_f16_bits(&b);
        let mut bt_f16 = vec![0u16; n * k];
        for i in 0..k {
            for j in 0..n {
                bt_f16[j * k + i] = b_f16[i * n + j];
            }
        }
        let mut y = vec![0.0f32; n];
        neon_gemv_f16_col_parallel(&x, &bt_f16, &mut y, k, n);
        let ref_y = reference_gemv_f16(&x, &b_f16, k, n);
        let max_err = y
            .iter()
            .zip(&ref_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < F16_GEMV_MAX_ABS_TOLERANCE,
            "f16 gemv odd tail max error {max_err} exceeds {F16_GEMV_MAX_ABS_TOLERANCE}"
        );
    }
}
