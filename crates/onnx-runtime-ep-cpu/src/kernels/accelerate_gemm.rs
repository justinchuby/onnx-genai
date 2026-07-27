//! Apple Accelerate + NEON GEMV for the M=1 decode hot path.
//!
//! Three routes:
//! - `sgemm`: Accelerate `cblas_sgemm` for M>1 prefill (reaches AMX).
//! - `neon_gemv_col_parallel`: Column-parallel NEON GEMV on pre-transposed B
//!   (primary M=1 path, 100-117 GB/s on M1 Max).
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
/// of the weight matrix. Each Rayon thread processes a distinct slice of
/// output columns via independent dot products — no partial sums, no
/// reduction, no per-call allocation.
///
/// For small matrices (k*n < 500K), runs single-threaded to avoid Rayon
/// dispatch overhead.
pub fn neon_gemv_col_parallel(x: &[f32], bt: &[f32], y: &mut [f32], k: usize, n: usize) {
    use rayon::prelude::*;

    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(bt.len(), n * k);
    debug_assert_eq!(y.len(), n);

    // Single-threaded for small matrices (Rayon dispatch > compute time).
    if k * n < 500_000 || rayon::current_num_threads() <= 1 {
        for i in 0..n {
            y[i] = neon_dot(&bt[i * k..(i + 1) * k], x);
        }
        return;
    }

    let threads = rayon::current_num_threads();
    let chunk = n.div_ceil(threads).max(1);
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(t, y_chunk)| {
            let n0 = t * chunk;
            for (li, yi) in y_chunk.iter_mut().enumerate() {
                let i = n0 + li;
                *yi = neon_dot(&bt[i * k..(i + 1) * k], x);
            }
        });
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
}
