//! Apple Accelerate + NEON GEMV for the M=1 decode hot path,
//! plus BNNS fp16→f32 MatMul for M≥2 prefill/batch-decode (reaches AMX).
//!
//! Four routes, selected per-op by matrix geometry:
//!
//! - `sgemm`: Accelerate `cblas_sgemm` for M>1 f32 prefill (reaches AMX).
//! - `bnns_matmul_f16`: BNNS `BNNSFilterCreateLayerBroadcastMatMul` for M≥2
//!   fp16→f32 prefill/batch-decode (~2451 GFLOPS on M1 Max via AMX).
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

// ─── BNNS fp16→f32 MatMul (prefill / batch-decode, M≥2) ─────────────

/// BNNS NDArray descriptor — matches Apple's C struct layout exactly (176 bytes).
/// Row-major convention: `size[0]` = columns (N), `size[1]` = rows (M).
#[repr(C)]
struct BNNSNDArrayDescriptor {
    flags: u32,
    layout: u32,
    size: [usize; 8],
    stride: [usize; 8],
    data: *mut std::ffi::c_void,
    data_type: u32,
    _pad0: u32,
    table_data: *mut std::ffi::c_void,
    table_data_type: u32,
    data_scale: f32,
    data_bias: f32,
    _pad1: u32,
}

/// BNNS broadcast matmul parameters — matches Apple's C struct (544 bytes).
#[repr(C)]
struct BNNSLayerParametersBroadcastMatMul {
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
    quadratic: bool,
    a_is_weights: bool,
    b_is_weights: bool,
    _pad: [u8; 3],
    i_a_desc: BNNSNDArrayDescriptor,
    i_b_desc: BNNSNDArrayDescriptor,
    o_desc: BNNSNDArrayDescriptor,
}

/// Opaque BNNS filter handle.
type BNNSFilter = *mut std::ffi::c_void;

/// BNNSFilterParameters — pass null for defaults.
#[repr(C)]
struct BNNSFilterParameters {
    _opaque: [u8; 0],
}

const BNNS_DATA_TYPE_FLOAT16: u32 = 0x10010;
const BNNS_DATA_TYPE_FLOAT32: u32 = 0x10020;
const BNNS_DATA_LAYOUT_ROW_MAJOR_MATRIX: u32 = 0x20000;

unsafe extern "C" {
    fn BNNSFilterCreateLayerBroadcastMatMul(
        params: *const BNNSLayerParametersBroadcastMatMul,
        filter_params: *const BNNSFilterParameters,
    ) -> BNNSFilter;

    fn BNNSFilterApplyTwoInput(
        filter: BNNSFilter,
        input1: *const std::ffi::c_void,
        input2: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
    ) -> i32;

    fn BNNSFilterDestroy(filter: BNNSFilter);
}

fn make_nd_desc(
    rows: usize,
    cols: usize,
    data_type: u32,
    data: *mut std::ffi::c_void,
) -> BNNSNDArrayDescriptor {
    let mut size = [0usize; 8];
    let mut stride = [0usize; 8];
    size[0] = cols;
    size[1] = rows;
    stride[0] = 1;
    stride[1] = cols;
    BNNSNDArrayDescriptor {
        flags: 0,
        layout: BNNS_DATA_LAYOUT_ROW_MAJOR_MATRIX,
        size,
        stride,
        data,
        data_type,
        _pad0: 0,
        table_data: std::ptr::null_mut(),
        table_data_type: 0,
        data_scale: 0.0,
        data_bias: 0.0,
        _pad1: 0,
    }
}

/// Returns `true` if BNNS fp16→f32 matmul is available on this system.
/// Result is cached after the first probe.
pub fn bnns_matmul_available() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    // 0 = unchecked, 1 = available, 2 = unavailable
    static CACHED: AtomicU8 = AtomicU8::new(0);
    let v = CACHED.load(Ordering::Relaxed);
    if v != 0 {
        return v == 1;
    }
    let a: [u16; 1] = [0x3C00]; // fp16 1.0
    let b: [u16; 1] = [0x3C00];
    let mut c: [f32; 1] = [0.0];
    let ok = bnns_matmul_f16(&a, &b, &mut c, 1, 1, 1) && (c[0] - 1.0).abs() < 1e-3;
    CACHED.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
    ok
}

/// C[m,n] = A[m,k] @ B[k,n] via BNNS, fp16 inputs → f32 output.
/// Returns `true` on success, `false` if the filter could not be created.
/// Thread-local BNNS filter cache.  Creating a `BNNSFilter` involves GCD
/// dispatch setup and possibly AMX micro-code compilation — measured at 3–19 ms
/// cold and ~50 µs warm.  Caching by (M, K, N) amortises this to zero for the
/// second and subsequent calls at each shape (a typical 24-layer model has only
/// 4–5 unique weight shapes, so the cache stays tiny).
///
/// Filters are destroyed when the thread exits via the `Drop` impl on
/// `FilterCache`.
struct FilterCache {
    map: std::collections::HashMap<(usize, usize, usize, bool), BNNSFilter>,
}

impl FilterCache {
    fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
        }
    }

    fn get_or_create(&mut self, m: usize, k: usize, n: usize, trans_b: bool) -> BNNSFilter {
        *self.map.entry((m, k, n, trans_b)).or_insert_with(|| {
            // When trans_b, B input is [N,K] row-major (B^T), BNNS transposes it.
            let b_desc = if trans_b {
                make_nd_desc(n, k, BNNS_DATA_TYPE_FLOAT16, std::ptr::null_mut())
            } else {
                make_nd_desc(k, n, BNNS_DATA_TYPE_FLOAT16, std::ptr::null_mut())
            };
            let params = BNNSLayerParametersBroadcastMatMul {
                alpha: 1.0,
                beta: 0.0,
                trans_a: false,
                trans_b,
                quadratic: false,
                a_is_weights: false,
                b_is_weights: false,
                _pad: [0; 3],
                i_a_desc: make_nd_desc(m, k, BNNS_DATA_TYPE_FLOAT16, std::ptr::null_mut()),
                i_b_desc: b_desc,
                o_desc: make_nd_desc(m, n, BNNS_DATA_TYPE_FLOAT32, std::ptr::null_mut()),
            };
            unsafe { BNNSFilterCreateLayerBroadcastMatMul(&params, std::ptr::null()) }
        })
    }
}

impl Drop for FilterCache {
    fn drop(&mut self) {
        for (_, filter) in self.map.drain() {
            if !filter.is_null() {
                unsafe {
                    BNNSFilterDestroy(filter);
                }
            }
        }
    }
}

std::thread_local! {
    static BNNS_FILTER_CACHE: std::cell::RefCell<FilterCache> =
        std::cell::RefCell::new(FilterCache::new());
}

pub fn bnns_matmul_f16(a: &[u16], b: &[u16], c: &mut [f32], m: usize, k: usize, n: usize) -> bool {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    BNNS_FILTER_CACHE.with(|cache| {
        let filter = cache.borrow_mut().get_or_create(m, k, n, false);
        if filter.is_null() {
            return false;
        }
        let rc = unsafe {
            BNNSFilterApplyTwoInput(
                filter,
                a.as_ptr() as *const std::ffi::c_void,
                b.as_ptr() as *const std::ffi::c_void,
                c.as_mut_ptr() as *mut std::ffi::c_void,
            )
        };
        rc == 0
    })
}

/// C[m,n] = A[m,k] @ B^T[n,k]^T via BNNS with transposed B.
///
/// B is passed as a row-major [N,K] matrix (i.e. B^T stored row-major).
/// BNNS applies the transpose internally, so the caller does NOT need to
/// materialise a contiguous row-major [K,N] copy.  This is critical for
/// column-major weights (e.g. lm_head vocab) where the column-major
/// [K,N] storage IS a row-major [N,K] B^T — eliminating a 272 MB copy.
pub fn bnns_matmul_f16_trans_b(
    a: &[u16],
    bt: &[u16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(bt.len(), n * k); // B^T is [N, K]
    debug_assert_eq!(c.len(), m * n);

    BNNS_FILTER_CACHE.with(|cache| {
        let filter = cache.borrow_mut().get_or_create(m, k, n, true);
        if filter.is_null() {
            return false;
        }
        let rc = unsafe {
            BNNSFilterApplyTwoInput(
                filter,
                a.as_ptr() as *const std::ffi::c_void,
                bt.as_ptr() as *const std::ffi::c_void,
                c.as_mut_ptr() as *mut std::ffi::c_void,
            )
        };
        rc == 0
    })
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

// ─── Thin-M GEMM: column-parallel batched-GEMV for M=2..16, large N ──

/// Minimum K×N element count for the thin-M path to activate. Below this,
/// B fits in cache and `cblas_sgemm` is faster.
///
/// **Fitted constant.** Measured crossover on M1 Max (48 MB SLC) at
/// K×N ≈ 2–4M elements. We use 4M (16 MB at f32) as the portable floor
/// because it exceeds the smallest SLC on any Apple Silicon Mac (8 MB on
/// base M1/M2/M3; source: Notebookcheck, Wikipedia die-shot analyses).
/// At 16 MB the weight matrix cannot reside in SLC, so every `cblas_sgemm`
/// panel re-read hits DRAM. The column-parallel NEON path streams B_T once,
/// which is superior at thin M under these conditions.
///
/// The 16 MB figure also exceeds the per-P-cluster L2 (12 MB on M1, 16 MB
/// on M2/M3), reinforcing that B is DRAM-resident for all parts.
///
/// **Bracket:** measured [2M, 4M] on M1 Max; 4M chosen as conservative
/// upper bound so the path only activates when the streaming advantage is
/// clear across all SLC sizes.
pub const THIN_M_LARGE_B_THRESHOLD: usize = 4_000_000;

/// Maximum M for the thin-M GEMM path. Above this, `cblas_sgemm` amortizes
/// its panel overhead efficiently and matches or beats column-parallel NEON.
///
/// **Mechanism (general):** As M grows, arithmetic intensity increases and
/// the compute-bound regime favors `cblas_sgemm`'s AMX-backed tiling.
/// **Coefficient (fitted):** Measured on M1 Max — NEON wins by ≥1.2× at
/// M≤16, break-even at M≈20, cblas wins at M≥24. Bracket: [16, 24].
const THIN_M_MAX: usize = 16;

/// Column-parallel thin-M GEMM using pre-transposed B_T\[N,K\] row-major.
///
/// Computes C\[M,N\] = A\[M,K\] @ B\[K,N\] by treating each output column j as
/// M independent dot products: C\[i,j\] = dot(A\[i,:\], B_T\[j,:\]).
///
/// The column-first loop order ensures that for each group of 4 B_T rows
/// (12–48 KB depending on K), all M dot products are computed while the data
/// is L1-hot. Total DRAM traffic is dominated by a single pass of B_T.
///
/// Parallelized across Rayon threads (or the persistent SPMD pool when active)
/// over disjoint column ranges — each thread writes to non-overlapping elements
/// of `c`, so no reduction or synchronization is needed.
pub fn neon_thin_m_gemm_col_parallel(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(bt.len(), n * k);
    debug_assert_eq!(c.len(), m * n);

    let threads = rayon::current_num_threads();

    // Single-threaded fallback for small matrices or single-threaded pools.
    if threads <= 1 || k * n < 500_000 {
        neon_thin_m_tile(a, bt, c, m, k, n, 0, n);
        return;
    }

    // Parallelize over column strips via Rayon. Each strip owns a disjoint
    // column range of C: strip t writes c[i*n + j0 .. i*n + j0+jn] for all
    // rows i, which does not overlap any other strip.
    use rayon::prelude::*;
    let chunk = n.div_ceil(threads).max(1);
    let num_strips = n.div_ceil(chunk);
    // SAFETY: each task writes to disjoint elements of `c` (non-overlapping
    // column ranges). The raw pointer send is necessary because `par_chunks_mut`
    // cannot partition `c` by columns (it's row-major).
    let c_ptr = c.as_mut_ptr() as usize;
    (0..num_strips).into_par_iter().for_each(|t| {
        let j0 = t * chunk;
        let jn = chunk.min(n - j0);
        let c_base = c_ptr as *mut f32;
        // SAFETY: writes only to c[i*n + j0 .. i*n + j0+jn] for i in 0..m,
        // which is disjoint from any other strip's range.
        unsafe {
            neon_thin_m_tile_raw(a, bt, c_base, m, k, n, j0, jn);
        }
    });
}

/// Returns `true` if the thin-M GEMM path should be used for the given shape.
/// Checks: M in 2..=THIN_M_MAX and K×N exceeds the streaming threshold.
#[inline]
pub fn thin_m_gemm_eligible(m: usize, k: usize, n: usize) -> bool {
    (2..=THIN_M_MAX).contains(&m) && k.saturating_mul(n) > THIN_M_LARGE_B_THRESHOLD
}

/// Compute a tile of C columns [j0..j0+jn] for all M rows, writing into a
/// contiguous output slice (used for single-threaded path and testing).
fn neon_thin_m_tile(
    a: &[f32],
    bt: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    j0: usize,
    jn: usize,
) {
    // Column-first: process 4 output columns at a time, computing all M rows
    // per group while B_T data is L1-hot.
    let mut j = 0;
    while j + 4 <= jn {
        let col = j0 + j;
        for i in 0..m {
            let a_row = &a[i * k..i * k + k];
            let s = neon_dot4_bt(a_row, bt, col, k);
            c[i * n + col] = s[0];
            c[i * n + col + 1] = s[1];
            c[i * n + col + 2] = s[2];
            c[i * n + col + 3] = s[3];
        }
        j += 4;
    }
    // Remainder columns
    while j < jn {
        let col = j0 + j;
        for i in 0..m {
            c[i * n + col] = neon_dot(&bt[col * k..(col + 1) * k], &a[i * k..i * k + k]);
        }
        j += 1;
    }
}

/// Same as `neon_thin_m_tile` but writes via raw pointer (for parallel path).
///
/// # Safety
/// Caller must ensure `c_base[i * n + j0 .. i * n + j0 + jn]` for `i` in
/// `0..m` are valid, non-overlapping with other concurrent writes.
unsafe fn neon_thin_m_tile_raw(
    a: &[f32],
    bt: &[f32],
    c_base: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    j0: usize,
    jn: usize,
) {
    let mut j = 0;
    while j + 4 <= jn {
        let col = j0 + j;
        for i in 0..m {
            let a_row = &a[i * k..i * k + k];
            let s = neon_dot4_bt(a_row, bt, col, k);
            unsafe {
                *c_base.add(i * n + col) = s[0];
                *c_base.add(i * n + col + 1) = s[1];
                *c_base.add(i * n + col + 2) = s[2];
                *c_base.add(i * n + col + 3) = s[3];
            }
        }
        j += 4;
    }
    while j < jn {
        let col = j0 + j;
        for i in 0..m {
            let val = neon_dot(&bt[col * k..(col + 1) * k], &a[i * k..i * k + k]);
            unsafe {
                *c_base.add(i * n + col) = val;
            }
        }
        j += 1;
    }
}

/// Compute 4 dot products simultaneously: dot(x, B_T[col+0]), dot(x, B_T[col+1]),
/// dot(x, B_T[col+2]), dot(x, B_T[col+3]) — sharing x loads across 4 B_T rows.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_dot4_bt(x: &[f32], bt: &[f32], col: usize, k: usize) -> [f32; 4] {
    use std::arch::aarch64::*;
    let r0 = &bt[col * k..];
    let r1 = &bt[(col + 1) * k..];
    let r2 = &bt[(col + 2) * k..];
    let r3 = &bt[(col + 3) * k..];
    let mut a0 = unsafe { vdupq_n_f32(0.0) };
    let mut a1 = unsafe { vdupq_n_f32(0.0) };
    let mut a2 = unsafe { vdupq_n_f32(0.0) };
    let mut a3 = unsafe { vdupq_n_f32(0.0) };
    let mut b0 = unsafe { vdupq_n_f32(0.0) };
    let mut b1 = unsafe { vdupq_n_f32(0.0) };
    let mut b2 = unsafe { vdupq_n_f32(0.0) };
    let mut b3 = unsafe { vdupq_n_f32(0.0) };
    let mut p = 0;
    while p + 8 <= k {
        unsafe {
            let x0 = vld1q_f32(x.as_ptr().add(p));
            let x1 = vld1q_f32(x.as_ptr().add(p + 4));
            a0 = vfmaq_f32(a0, vld1q_f32(r0.as_ptr().add(p)), x0);
            b0 = vfmaq_f32(b0, vld1q_f32(r0.as_ptr().add(p + 4)), x1);
            a1 = vfmaq_f32(a1, vld1q_f32(r1.as_ptr().add(p)), x0);
            b1 = vfmaq_f32(b1, vld1q_f32(r1.as_ptr().add(p + 4)), x1);
            a2 = vfmaq_f32(a2, vld1q_f32(r2.as_ptr().add(p)), x0);
            b2 = vfmaq_f32(b2, vld1q_f32(r2.as_ptr().add(p + 4)), x1);
            a3 = vfmaq_f32(a3, vld1q_f32(r3.as_ptr().add(p)), x0);
            b3 = vfmaq_f32(b3, vld1q_f32(r3.as_ptr().add(p + 4)), x1);
        }
        p += 8;
    }
    let mut s0 = unsafe { vaddvq_f32(vaddq_f32(a0, b0)) };
    let mut s1 = unsafe { vaddvq_f32(vaddq_f32(a1, b1)) };
    let mut s2 = unsafe { vaddvq_f32(vaddq_f32(a2, b2)) };
    let mut s3 = unsafe { vaddvq_f32(vaddq_f32(a3, b3)) };
    while p < k {
        let xv = x[p];
        s0 += r0[p] * xv;
        s1 += r1[p] * xv;
        s2 += r2[p] * xv;
        s3 += r3[p] * xv;
        p += 1;
    }
    [s0, s1, s2, s3]
}

/// Scalar fallback for non-aarch64 targets.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn neon_dot4_bt(x: &[f32], bt: &[f32], col: usize, k: usize) -> [f32; 4] {
    let mut s = [0.0f32; 4];
    for c in 0..4 {
        let row = &bt[(col + c) * k..(col + c + 1) * k];
        s[c] = x.iter().zip(row).map(|(&a, &b)| a * b).sum();
    }
    s
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

    // ─── BNNS fp16→f32 tests ───────────────────────────────────

    #[test]
    fn bnns_availability_probe_returns_consistent_result() {
        let first = bnns_matmul_available();
        let second = bnns_matmul_available();
        assert_eq!(
            first, second,
            "BNNS availability probe must be deterministic"
        );
    }

    /// Numerics parity: BNNS fp16→f32 vs f64 reference at multiple shapes.
    /// Tolerance scales with sqrt(K) to account for fp16 mantissa errors
    /// accumulating during the K-dimension reduction.
    #[test]
    fn bnns_matmul_f16_matches_f64_reference() {
        if !bnns_matmul_available() {
            eprintln!("BNNS not available, skipping");
            return;
        }
        let shapes: &[(usize, usize, usize)] = &[
            (2, 4, 3),
            (4, 8, 6),
            (16, 64, 32),
            (128, 896, 4864), // model-scale prefill
        ];
        for &(m, k, n) in shapes {
            let a_f32: Vec<f32> = (0..m * k)
                .map(|i| ((i % 997) as f32) * 0.001 - 0.5)
                .collect();
            let b_f32: Vec<f32> = (0..k * n)
                .map(|i| ((i % 991) as f32) * 0.001 - 0.5)
                .collect();
            let a_f16 = f32_to_f16_bits(&a_f32);
            let b_f16 = f32_to_f16_bits(&b_f32);
            let a_f64: Vec<f64> = a_f16
                .iter()
                .map(|&v| half::f16::from_bits(v).to_f64())
                .collect();
            let b_f64: Vec<f64> = b_f16
                .iter()
                .map(|&v| half::f16::from_bits(v).to_f64())
                .collect();
            let mut ref_c = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for p in 0..k {
                        sum += a_f64[i * k + p] * b_f64[p * n + j];
                    }
                    ref_c[i * n + j] = sum;
                }
            }
            let mut c = vec![0.0f32; m * n];
            assert!(
                bnns_matmul_f16(&a_f16, &b_f16, &mut c, m, k, n),
                "BNNS filter creation failed for ({m},{k},{n})"
            );
            let max_rel = c
                .iter()
                .zip(&ref_c)
                .filter(|(_, r)| r.abs() > 1e-6)
                .map(|(a, r)| (((*a as f64) - r) / r).abs())
                .fold(0.0f64, f64::max);
            // fp16 has ~9.77e-4 relative precision; f32 accumulation over K
            // elements introduces ~sqrt(K) * eps_fp16 expected error.
            let tol = 0.005 + (k as f64).sqrt() * 2e-3;
            assert!(
                max_rel < tol,
                "BNNS f16 matmul ({m},{k},{n}) max relative error {max_rel:.6} \
                 exceeds tolerance {tol:.4}"
            );
        }
    }

    /// BNNS trans_b correctness: C = A[m,k] @ B[k,n] using B^T[n,k] with trans_b.
    /// Verifies that `bnns_matmul_f16_trans_b` produces the same result as the
    /// non-transposed path, which is critical for the column-major weight rescue.
    #[test]
    fn bnns_matmul_f16_trans_b_matches_normal() {
        if !bnns_matmul_available() {
            eprintln!("BNNS not available, skipping");
            return;
        }
        let shapes: &[(usize, usize, usize)] = &[(2, 3, 4), (11, 64, 32), (40, 896, 4864)];
        for &(m, k, n) in shapes {
            let a_f32: Vec<f32> = (0..m * k)
                .map(|i| ((i % 997) as f32) * 0.001 - 0.5)
                .collect();
            let b_f32: Vec<f32> = (0..k * n)
                .map(|i| ((i % 991) as f32) * 0.001 - 0.5)
                .collect();
            let a_f16 = f32_to_f16_bits(&a_f32);
            let b_f16 = f32_to_f16_bits(&b_f32);

            // Normal path: A[m,k] @ B[k,n] row-major
            let mut c_normal = vec![0.0f32; m * n];
            assert!(bnns_matmul_f16(&a_f16, &b_f16, &mut c_normal, m, k, n));

            // Trans_b path: A[m,k] @ B^T[n,k]^T — create B^T as row-major [n,k]
            let mut bt_f16 = vec![0u16; n * k];
            for i in 0..k {
                for j in 0..n {
                    bt_f16[j * k + i] = b_f16[i * n + j];
                }
            }
            let mut c_trans = vec![0.0f32; m * n];
            assert!(
                bnns_matmul_f16_trans_b(&a_f16, &bt_f16, &mut c_trans, m, k, n),
                "bnns_matmul_f16_trans_b failed at ({m},{k},{n})"
            );

            // Compare: should be bitwise identical (same BNNS accumulation)
            // or at least within f32 epsilon
            let max_diff = c_normal
                .iter()
                .zip(&c_trans)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = c_normal
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-6);
            assert!(
                max_diff / scale < 1e-5,
                "trans_b vs normal mismatch at ({m},{k},{n}): max_diff={max_diff}, scale={scale}"
            );
        }
    }

    /// Edge values: fp16 max, denormals, NaN, zero.
    #[test]
    fn bnns_matmul_f16_handles_edge_values() {
        if !bnns_matmul_available() {
            eprintln!("BNNS not available, skipping");
            return;
        }
        let max_f16: u16 = 0x7BFF; // 65504.0
        let one_f16: u16 = 0x3C00; // 1.0
        let mut c = [0.0f32; 1];
        assert!(bnns_matmul_f16(&[max_f16], &[one_f16], &mut c, 1, 1, 1));
        assert!(
            (c[0] - 65504.0).abs() < 1.0,
            "fp16 max * 1.0 = {} (expected ~65504.0)",
            c[0]
        );

        let zero: u16 = 0x0000;
        c[0] = 999.0;
        assert!(bnns_matmul_f16(&[zero], &[max_f16], &mut c, 1, 1, 1));
        assert!(c[0].abs() < 1e-6, "zero * max = {} (expected 0.0)", c[0]);

        let nan_f16: u16 = 0x7E00;
        c[0] = 0.0;
        assert!(bnns_matmul_f16(&[nan_f16], &[one_f16], &mut c, 1, 1, 1));
        assert!(c[0].is_nan(), "NaN * 1.0 should be NaN, got {}", c[0]);

        let denorm: u16 = 0x0001;
        c[0] = 999.0;
        assert!(bnns_matmul_f16(&[denorm], &[one_f16], &mut c, 1, 1, 1));
        assert!(
            c[0].abs() < 1e-4,
            "denorm * 1.0 = {} (expected near zero)",
            c[0]
        );
    }

    /// Bitwise determinism: same inputs produce identical output bytes.
    #[test]
    fn bnns_matmul_f16_deterministic() {
        if !bnns_matmul_available() {
            eprintln!("BNNS not available, skipping");
            return;
        }
        let (m, k, n) = (16, 64, 32);
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();
        let a = f32_to_f16_bits(&a_f32);
        let b = f32_to_f16_bits(&b_f32);
        let mut c1 = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        assert!(bnns_matmul_f16(&a, &b, &mut c1, m, k, n));
        assert!(bnns_matmul_f16(&a, &b, &mut c2, m, k, n));
        assert_eq!(
            c1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            c2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "BNNS must be bitwise deterministic"
        );
    }
}
