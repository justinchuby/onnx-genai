//! `MatMul`: numpy-style matrix multiplication for floating-point tensors,
//! including batched and broadcast leading dimensions and 1-D vector operands
//! (`docs/ORT2.md` §4.4).
//!
//! ## Perf seam (Phase-1.5)
//!
//! The 2-D tile GEMM ([`gemm`]) dispatches on [`CpuBackend::auto_detect`]
//! (`docs/ORT2.md` §25.2):
//!
//! * **Generic** (default fallback, always compiled, offline): a blocked,
//!   register-tiled, rayon-parallelized pure-Rust f32 GEMM ([`gemm_generic`]).
//!   It is the correctness baseline and contains no `unsafe`.
//! * **`SimdX86`** (default on AVX2/FMA x86-64, runtime-detected): an
//!   MLAS-style packed SIMD f32 SGEMM ([`simd_gemm`]) — panel packing + a
//!   `6×16` AVX2/FMA register microkernel + K/N cache blocking, parallelized
//!   over column strips. Selected automatically with no cargo feature; falls
//!   back to Generic when AVX2/FMA is absent.
//! * **`Mlas`** (opt-in `mlas` feature on x86-64): vendored MLAS f32 SGEMM,
//!   selected only with `NXRT_CPU_GEMM_BACKEND=mlas`. Multi-threaded — MLAS
//!   partitions the GEMM and runs the tiles across the process Rayon pool — but
//!   kept opt-in (not an automatic default) pending a later slice.
//!
//! The batched / broadcast / 1-D-vector handling in [`matmul_dense`] is
//! backend-agnostic; only the inner 2-D tile GEMM changes. The session also
//! marks graph-initializer inputs so this kernel can safely prepack constants.

use std::borrow::Cow;
use std::sync::OnceLock;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Node, broadcast_shapes, compute_contiguous_strides};
use rayon::prelude::*;

use super::check_arity;
use super::half_gemm::{self, HalfFormat, MatrixLayout};
use crate::backend::CpuBackend;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::{next_index, numel};

// MLAS-style packed SIMD f32 GEMM (the `SimdX86` backend). Kept in a sibling
// file but included here so `kernels/mod.rs` needs no edit; it is an internal
// perf detail of the MatMul hot path, not a new op.
#[path = "simd_gemm.rs"]
mod simd_gemm;

// Native BF16×BF16→FP32 GEMM (`_mm512_dpbf16_ps`) for avx512_bf16 hosts. It is
// runtime-detected and otherwise falls back to the portable blocked half GEMM.
#[path = "bf16_gemm.rs"]
mod bf16_gemm;

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[path = "accelerate_gemm.rs"]
mod accelerate_gemm;

/// Re-export the f16 GEMV for use by sibling kernels (FusedMatMulBias).
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) use accelerate_gemm::neon_gemv_f16_col_parallel;

/// Test-only counter: incremented each time the FP16 GEMV decode path is reached
/// inside [`MatMulKernel::execute_with_backend`]. Guards against dispatch
/// regressions where a broader half-precision GEMM intercepts M=1 decode before
/// the bandwidth-optimal GEMV.
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "ios")
))]
static GEMV_F16_TEST_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Per-kernel cache for immutable MatMul operands that require materialization.
///
/// Contiguous f32 constants already have the ideal representation, so they stay
/// zero-copy and need no owned cache entry.
#[derive(Default)]
pub(crate) struct MatMulPrepack {
    constant_inputs: [bool; 2],
    dense: [OnceLock<Vec<f32>>; 2],
    #[cfg(feature = "mlas")]
    packed_b: OnceLock<mlas_sys::PackedB>,
    /// Lazily-computed transpose of the B weight matrix for the Accelerate
    /// column-parallel GEMV path. Only populated for constant (model weight)
    /// inputs on macOS/iOS.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    transposed_b: OnceLock<Vec<f32>>,
    /// Lazily-computed f16 transpose of the B weight matrix. Stores the raw
    /// u16 bit patterns of half::f16 in N×K layout, read directly from the
    /// mmap'd model file without widening to f32. Only populated when B is a
    /// constant Float16 input on macOS/iOS.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    transposed_b_f16: OnceLock<Vec<u16>>,
}

impl MatMulPrepack {
    pub(crate) fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in self.constant_inputs.iter_mut().enumerate() {
            *is_constant = constant_inputs.get(index).copied().unwrap_or(false);
        }
    }

    pub(crate) fn dense<'a>(
        &'a self,
        index: usize,
        view: &'a TensorView<'_>,
    ) -> Result<Cow<'a, [f32]>> {
        if !self.constant_inputs[index] {
            return to_dense_f32_widen("MatMul", view);
        }
        if let Some(cached) = self.dense[index].get() {
            return Ok(Cow::Borrowed(cached));
        }

        match to_dense_f32_widen("MatMul", view)? {
            Cow::Borrowed(dense) => Ok(Cow::Borrowed(dense)),
            Cow::Owned(dense) => {
                let _ = self.dense[index].set(dense);
                Ok(Cow::Borrowed(
                    self.dense[index]
                        .get()
                        .expect("constant MatMul prepack was just initialized"),
                ))
            }
        }
    }

    /// Returns a cached transpose of B[K,N] -> B_T[N,K] row-major.
    /// Only transposes constant (model weight) inputs. Returns `None` for
    /// activations. Uses Rayon + cache-blocking to hide the one-time cost.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn transposed_b(&self, b: &[f32], k: usize, n: usize) -> Option<&[f32]> {
        if !self.constant_inputs[1] {
            return None;
        }
        Some(
            self.transposed_b
                .get_or_init(|| {
                    use rayon::prelude::*;
                    let mut bt = vec![0.0f32; n * k];
                    // Parallel tiled transpose: each Rayon task handles a strip
                    // of output rows (columns of B), using a tile size that keeps
                    // both read and write working sets in L1 cache.
                    let threads = rayon::current_num_threads();
                    let rows_per_thread = n.div_ceil(threads).max(1);
                    bt.par_chunks_mut(rows_per_thread * k)
                        .enumerate()
                        .for_each(|(t, bt_chunk)| {
                            let j0 = t * rows_per_thread;
                            let j_end = (j0 + rows_per_thread).min(n);
                            let chunk_n = j_end - j0;
                            const TILE: usize = 64;
                            for i0 in (0..k).step_by(TILE) {
                                let ie = (i0 + TILE).min(k);
                                for jj in 0..chunk_n {
                                    let j = j0 + jj;
                                    for i in i0..ie {
                                        bt_chunk[jj * k + i] = b[i * n + j];
                                    }
                                }
                            }
                        });
                    bt
                })
                .as_slice(),
        )
    }

    /// Returns a cached f16 transpose of B[K,N] → B_T[N,K] row-major.
    ///
    /// Like [`transposed_b`] but preserves the original f16 storage format
    /// (as raw u16 bit patterns), reading directly from the mmap'd model
    /// buffer. Only populated for constant Float16 inputs on macOS/iOS.
    /// Returns `None` for non-constant inputs or non-Float16 dtypes.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) fn transposed_b_f16(
        &self,
        b_view: &TensorView,
        k: usize,
        n: usize,
    ) -> Option<&[u16]> {
        use onnx_runtime_ir::DataType;
        if !self.constant_inputs[1] || b_view.dtype != DataType::Float16 || !b_view.is_contiguous()
        {
            return None;
        }
        Some(
            self.transposed_b_f16
                .get_or_init(|| {
                    let numel = k * n;
                    // SAFETY: validated contiguous Float16 view → exactly `numel`
                    // 2-byte elements at `data_ptr`; `half::f16` is
                    // `repr(transparent)` over `u16`.
                    let src =
                        unsafe { std::slice::from_raw_parts(b_view.data_ptr::<u16>(), numel) };
                    use rayon::prelude::*;
                    let mut bt = vec![0u16; n * k];
                    let threads = rayon::current_num_threads();
                    let rows_per_thread = n.div_ceil(threads).max(1);
                    bt.par_chunks_mut(rows_per_thread * k)
                        .enumerate()
                        .for_each(|(t, bt_chunk)| {
                            let j0 = t * rows_per_thread;
                            let j_end = (j0 + rows_per_thread).min(n);
                            let chunk_n = j_end - j0;
                            const TILE: usize = 64;
                            for i0 in (0..k).step_by(TILE) {
                                let ie = (i0 + TILE).min(k);
                                for jj in 0..chunk_n {
                                    let j = j0 + jj;
                                    for i in i0..ie {
                                        bt_chunk[jj * k + i] = src[i * n + j];
                                    }
                                }
                            }
                        });
                    bt
                })
                .as_slice(),
        )
    }

    #[cfg(feature = "mlas")]
    fn packed_b(&self, b: &[f32], k: usize, n: usize) -> Option<&mlas_sys::PackedB> {
        self.constant_inputs[1].then(|| {
            self.packed_b
                .get_or_init(|| mlas_sys::PackedB::new(n, k, b))
        })
    }
}

/// f32 MatMul kernel with initializer-only operand prepacking.
#[derive(Default)]
pub struct MatMulKernel {
    prepack: MatMulPrepack,
}

/// Factory for [`MatMulKernel`] (no attributes).
pub struct MatMulFactory;

impl KernelFactory for MatMulFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MatMulKernel::default()))
    }
}

/// 2-D tile GEMM dispatch: `c[m,n] = sum_k a[m,k] * b[k,n]` (overwrite).
///
/// `a` is `m*k` row-major, `b` is `k*n` row-major, `c` is `m*n` row-major.
/// Picks the backend via [`CpuBackend::auto_detect`] (`docs/ORT2.md` §25.2):
/// `SimdX86` when supported by the host, otherwise the pure-Rust blocked GEMM.
/// The result is bit-plausible across backends within f32 tolerance.
pub(crate) fn gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    gemm_with_backend(CpuBackend::auto_detect(), a, b, c, m, k, n)
}

#[cfg(feature = "mlas")]
fn gemm_packed(
    a: &[f32],
    packed: &mlas_sys::PackedB,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    assert_eq!(packed.dimensions(), (k, n));
    mlas_sys::sgemm_nn_packed(m, a, packed, c);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemm_with_backend(
    backend: CpuBackend,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    match backend {
        #[cfg(feature = "mlas")]
        CpuBackend::Mlas => {
            mlas_sys::sgemm_nn(m, n, k, a, b, c);
            Ok(())
        }
        // Built-in MLAS-style packed SIMD backend for AVX2/FMA x86-64 hosts.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        CpuBackend::SimdX86 => {
            simd_gemm::sgemm_simd(a, b, c, m, k, n);
            Ok(())
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        CpuBackend::Accelerate => {
            if m == 1 {
                accelerate_gemm::neon_gemv_parallel(a, b, c, k, n);
            } else {
                accelerate_gemm::sgemm(a, b, c, m, k, n);
            }
            Ok(())
        }
        _ => {
            gemm_generic(a, b, c, m, k, n);
            Ok(())
        }
    }
}

// Register microkernel tile: MR rows x NR cols of C accumulated in registers.
const MR: usize = 4;
const NR: usize = 4;
// Cache block over the K dimension so a panel of B stays resident in L1/L2
// while a strip of C accumulates across it.
const KC: usize = 256;
const MAX_MC: usize = 64;

/// Pure-Rust blocked, register-tiled, rayon-parallelized f32 GEMM (the Generic
/// backend). Overwrites `c` with `a @ b`.
///
/// Strategy: the outer M dimension is split into pool-sized row blocks
/// distributed across Rayon. Each task blocks over K in `KC`-wide panels and
/// walks N in `NR`-wide strips, accumulating an `MR x NR` register tile so the
/// hot inner loop over the N strip autovectorizes. Contains no `unsafe`.
fn gemm_generic(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let threads = rayon::current_num_threads();
    if threads > 1 && m < threads && n > 1 {
        gemm_generic_col_parallel(a, b, c, m, k, n, threads);
        return;
    }
    let mc = if threads <= 1 {
        MAX_MC.min(m)
    } else {
        let target_tasks = threads.saturating_mul(2);
        let rows = m.div_ceil(target_tasks).clamp(1, MAX_MC);
        if rows == 1 {
            1
        } else {
            rows.div_ceil(MR).saturating_mul(MR).min(MAX_MC)
        }
    };
    // Parallelize over row blocks of C; each block owns a disjoint slice of `c`
    // and reads shared, immutable `a`/`b`, so there is no aliasing. Size the
    // blocks from the Rayon pool: prefill commonly has fewer rows than cores,
    // while large matrices retain MR-row reuse and bounded task counts.
    // `for_each_init` rather than `for_each` so a worker-lane span covers this
    // worker's whole row block instead of being reopened per block.
    c.par_chunks_mut(mc * n).enumerate().for_each_init(
        || crate::trace::worker_span("MatMul.row_block"),
        |_span, (blk, c_block)| {
            let i0 = blk * mc;
            let rows = c_block.len() / n; // last block may be short
            let a_block = &a[i0 * k..i0 * k + rows * k];
            gemm_block(a_block, b, c_block, rows, k, n);
        },
    );
}

fn gemm_generic_col_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) {
    use rayon::prelude::*;
    let cols_per_strip = n.div_ceil(threads).max(1);
    let strips: Vec<(usize, usize)> = (0..n)
        .step_by(cols_per_strip)
        .map(|j0| (j0, cols_per_strip.min(n - j0)))
        .collect();
    let results: Vec<(usize, usize, Vec<f32>)> = strips
        .into_par_iter()
        .map(|(j0, strip_n)| {
            let mut b_strip = vec![0.0f32; k * strip_n];
            for row in 0..k {
                b_strip[row * strip_n..row * strip_n + strip_n]
                    .copy_from_slice(&b[row * n + j0..row * n + j0 + strip_n]);
            }
            let mut c_strip = vec![0.0f32; m * strip_n];
            gemm_block(a, &b_strip, &mut c_strip, m, k, strip_n);
            (j0, strip_n, c_strip)
        })
        .collect();
    for (j0, strip_n, c_strip) in results {
        for row in 0..m {
            c[row * n + j0..row * n + j0 + strip_n]
                .copy_from_slice(&c_strip[row * strip_n..row * strip_n + strip_n]);
        }
    }
}

/// Compute `c_block[rows,n] = a_block[rows,k] @ b[k,n]` (overwrite) for one row
/// block, blocking over K and register-tiling MR x NR.
fn gemm_block(a: &[f32], b: &[f32], c: &mut [f32], rows: usize, k: usize, n: usize) {
    for v in c.iter_mut() {
        *v = 0.0;
    }
    let mut kk = 0;
    while kk < k {
        let kc = KC.min(k - kk);
        let mut i = 0;
        while i < rows {
            let mr = MR.min(rows - i);
            let mut j = 0;
            while j < n {
                let nr = NR.min(n - j);
                micro_kernel(a, b, c, k, n, i, j, kk, kc, mr, nr);
                j += NR;
            }
            i += MR;
        }
        kk += KC;
    }
}

/// Accumulate an `mr x nr` (≤ `MR x NR`) tile of C over the K-panel
/// `[kk, kk+kc)`, adding into the existing `c` contents.
#[inline]
#[allow(clippy::too_many_arguments)]
fn micro_kernel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    k: usize,
    n: usize,
    i: usize,
    j: usize,
    kk: usize,
    kc: usize,
    mr: usize,
    nr: usize,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for p in kk..kk + kc {
        let brow = &b[p * n + j..p * n + j + nr];
        for (ii, acc_row) in acc.iter_mut().enumerate().take(mr) {
            let aik = a[(i + ii) * k + p];
            for (jj, acc_v) in acc_row.iter_mut().enumerate().take(nr) {
                *acc_v += aik * brow[jj];
            }
        }
    }
    for (ii, acc_row) in acc.iter().enumerate().take(mr) {
        let c_row = &mut c[(i + ii) * n + j..(i + ii) * n + j + nr];
        for (jj, cv) in c_row.iter_mut().enumerate().take(nr) {
            *cv += acc_row[jj];
        }
    }
}

impl Kernel for MatMulKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.prepack.set_constant_inputs(constant_inputs);
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.execute_with_backend(inputs, outputs, CpuBackend::auto_detect())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn estimated_flops(&self) -> Option<u64> {
        None
    }
}

impl MatMulKernel {
    fn execute_with_backend(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        backend: CpuBackend,
    ) -> Result<()> {
        check_arity("MatMul", inputs, outputs, 2, 2, 1)?;
        let geom = matmul_geometry(&inputs[0], &inputs[1])?;
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            (numel(&geom.batch_shape) as u64)
                .saturating_mul(geom.m as u64)
                .saturating_mul(geom.n as u64)
                .saturating_mul(geom.k as u64)
                .saturating_mul(2)
        });

        // FP16 storage GEMV: when B is a constant Float16 weight and M=1
        // (decode), GEMV directly from the f16 mmap'd data — reading 2 bytes
        // per weight instead of 4, halving memory bandwidth. This MUST be
        // checked before try_matmul_half: the blocked GEMM is ~4× slower than
        // the bandwidth-optimal NEON GEMV at M=1 decode shapes.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if backend == CpuBackend::Accelerate
            && geom.m == 1
            && numel(&geom.batch_shape) <= 1
            && geom.b_promoted_rank == 2
            && inputs[1].dtype == onnx_runtime_ir::DataType::Float16
            && let Some(bt_f16) = self.prepack.transposed_b_f16(&inputs[1], geom.k, geom.n)
        {
            #[cfg(all(test, target_arch = "aarch64"))]
            GEMV_F16_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let a_dense = self.prepack.dense(0, &inputs[0])?;
            let mut result = vec![0.0f32; geom.n];
            accelerate_gemm::neon_gemv_f16_col_parallel(
                &a_dense,
                bt_f16,
                &mut result,
                geom.k,
                geom.n,
            );
            return write_dense_f32_narrow("MatMul", &mut outputs[0], &result);
        }

        // Dedicated half-precision path: contiguous f16/bf16 operands stay in
        // 16-bit storage and are packed in cache-sized panels for f32
        // accumulation. Bf16 may use the runtime-gated AVX-512 BF16 kernel;
        // every other host uses the portable blocked implementation.
        if let Some(result) = try_matmul_half(&inputs[0], &inputs[1], &geom)? {
            return write_dense_f32_narrow("MatMul", &mut outputs[0], &result);
        }

        // Direct f32 output fast path: when the output is a contiguous Float32
        // CPU tensor that does not alias either input, GEMM writes straight into
        // its backing buffer, skipping both the intermediate result `Vec<f32>`
        // and the narrowing copy performed by `write_dense_f32_narrow`. Every
        // other case (f16/bf16/f64, strided/non-contiguous, or a possibly
        // aliasing output) uses the owned-buffer fallback below unchanged.
        if output_is_direct_f32_eligible(&inputs[0], &inputs[1], &outputs[0]) {
            let out = &mut outputs[0];
            // `validate()` confirms rank/dtype/offset invariants; it does NOT
            // prove backing-buffer bounds — that is the executor's `view_bounds`
            // contract for this SSA output. We still gate the pointer-slice on a
            // logical length match against the computed result length so a
            // mismatched shape errors BEFORE any GEMM write.
            out.validate()?;
            let numel = out.numel();
            if numel != geom.result_len {
                return Err(EpError::KernelFailed(format!(
                    "MatMul: output element count {numel} does not match result length {}",
                    geom.result_len
                )));
            }
            // A zero-sized result writes nothing; return before forming a slice
            // from a possibly-dangling zero-length output pointer.
            if numel == 0 {
                return Ok(());
            }
            let ptr = out.data_ptr_mut::<f32>();
            // SAFETY: the eligibility check proved `out` is a CPU, Float32,
            // row-major-contiguous view that does not alias either input's byte
            // range, so no live input slice overlaps this buffer. `data_ptr_mut`
            // applies `byte_offset` to select the element origin, and the
            // executor's bounds contract guarantees `numel` initialized f32 slots
            // exist there; `numel == geom.result_len` was just verified, so the
            // GEMM writes exactly within bounds. The slice is the sole mutable
            // borrow of this storage for the duration of the call.
            let out_slice = unsafe { std::slice::from_raw_parts_mut(ptr, numel) };
            return matmul_dense_into_with_backend(
                &self.prepack.dense(0, &inputs[0])?,
                &self.prepack.dense(1, &inputs[1])?,
                &geom,
                backend,
                Some(&self.prepack),
                out_slice,
            );
        }

        let out =
            matmul_dense_prepacked_with_backend(&inputs[0], &inputs[1], &self.prepack, backend)?;
        // If either operand was 1-D, the corresponding size-1 axis is squeezed
        // out of the result; the narrowing writer uses the output view's own
        // shape and dtype (f32/f16/bf16/f64), so the buffer matches element for
        // element and rounds to the requested precision.
        write_dense_f32_narrow("MatMul", &mut outputs[0], &out)
    }
}

/// Whether `MatMulKernel::execute` may GEMM directly into `out`'s backing
/// buffer instead of the intermediate-vector + narrowing fallback.
///
/// Requires a CPU, Float32, row-major-contiguous output whose byte range does
/// not overlap either input. The overlap check mirrors the in-place fast-path
/// convention used elsewhere in this crate (`kernels/activations.rs`,
/// `kernels/elementwise.rs`): a same-device pointer-range test on the element
/// origins. It is sound here because a zero-copy input operand
/// (`to_dense_f32_widen` borrows only contiguous Float32 views) is read
/// straight from its own contiguous `numel * 4` byte range, which this test
/// covers exactly; any other input dtype/layout is materialized into a fresh
/// owned buffer before the GEMM, so it cannot alias the output at all.
/// Whether an output tensor is eligible for direct f32 GEMM writes: must be
/// contiguous Float32 on CPU, not aliasing either input.
pub(crate) fn output_is_direct_f32_eligible(
    a: &TensorView,
    b: &TensorView,
    out: &TensorMut,
) -> bool {
    use onnx_runtime_ir::DataType;
    use onnx_runtime_ir::DeviceType;

    if out.device.device_type != DeviceType::Cpu
        || out.dtype != DataType::Float32
        || !out.is_contiguous()
    {
        return false;
    }

    // Row-major-contiguous Float32: the whole logical extent is one dense range
    // starting at the element origin.
    let out_origin = (out.data.0 as *const u8).wrapping_add(out.byte_offset) as usize;
    let out_end = out_origin.saturating_add(out.byte_size());

    !std::iter::once(a)
        .chain(std::iter::once(b))
        .any(|input| output_overlaps_input(out_origin, out_end, input, out.device))
}

/// Pointer-range overlap test between the output byte range `[out_origin,
/// out_end)` and one input's element-origin byte range, on the same device.
/// Absent (null) inputs never overlap.
fn output_overlaps_input(
    out_origin: usize,
    out_end: usize,
    input: &TensorView,
    out_device: onnx_runtime_ir::DeviceId,
) -> bool {
    if input.is_absent() || input.device != out_device {
        return false;
    }
    let in_start = input.data_ptr::<u8>() as usize;
    let in_end = in_start.saturating_add(input.byte_size());
    out_origin < in_end && in_start < out_end
}

/// Attempt the dedicated portable half GEMM path. Both operands must be
/// contiguous and have the same `Float16` or `BFloat16` dtype. The operands stay
/// in 16-bit storage until cache-panel packing, accumulation is always `f32`,
/// and the caller narrows once into the requested output dtype.
fn try_matmul_half(
    a: &TensorView,
    b: &TensorView,
    geom: &MatMulGeometry,
) -> Result<Option<Vec<f32>>> {
    use onnx_runtime_ir::DataType;

    let format = match (a.dtype, b.dtype) {
        (DataType::Float16, DataType::Float16) => HalfFormat::F16,
        (DataType::BFloat16, DataType::BFloat16) => HalfFormat::Bf16,
        _ => return Ok(None),
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Ok(None);
    }
    a.validate()?;
    b.validate()?;
    let mut out = vec![0.0f32; geom.result_len];
    if out.is_empty() || geom.k == 0 {
        return Ok(Some(out));
    }
    let a_len = a.numel();
    let b_len = b.numel();
    // SAFETY: validated contiguous Float16/BFloat16 views address exactly
    // `a_len`/`b_len` two-byte elements. Both half types are transparent over
    // `u16`, so reading their storage as raw bit patterns is sound.
    let a_bits = unsafe { std::slice::from_raw_parts(a.data_ptr::<u16>(), a_len) };
    let b_bits = unsafe { std::slice::from_raw_parts(b.data_ptr::<u16>(), b_len) };
    half_dense_into(format, a_bits, b_bits, geom, &mut out);
    Ok(Some(out))
}

/// Drive the half GEMM over a single, batched, or broadcast operand pair into
/// `out` (row-major f32), reusing the shared batch-broadcast walk.
fn half_dense_into(
    format: HalfFormat,
    a: &[u16],
    b: &[u16],
    geom: &MatMulGeometry,
    out: &mut [f32],
) {
    let (m, k, n) = (geom.m, geom.k, geom.n);
    let (a_mat, b_mat, c_mat) = (geom.a_mat, geom.b_mat, geom.c_mat);
    if geom.batch_shape.is_empty() {
        half_gemm_tile(format, a, b, out, m, k, n);
        return;
    }
    let mut bidx = vec![0usize; geom.batch_shape.len()];
    let mut b_out = 0usize;
    loop {
        let a_off = broadcast_offset(&bidx, &geom.a_batch, &geom.a_batch_strides) * a_mat;
        let b_off = broadcast_offset(&bidx, &geom.b_batch, &geom.b_batch_strides) * b_mat;
        half_gemm_tile(
            format,
            &a[a_off..a_off + a_mat],
            &b[b_off..b_off + b_mat],
            &mut out[b_out * c_mat..b_out * c_mat + c_mat],
            m,
            k,
            n,
        );
        b_out += 1;
        if !next_index(&geom.batch_shape, &mut bidx) {
            break;
        }
    }
}

/// Select the runtime-gated AVX-512 BF16 microkernel when available; otherwise
/// use the portable blocked half GEMM. The f16 path is always portable today.
fn half_gemm_tile(
    format: HalfFormat,
    a: &[u16],
    b: &[u16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if format == HalfFormat::Bf16 && bf16_gemm::native_available() {
        bf16_gemm::gemm(a, b, c, m, k, n);
        return;
    }

    half_gemm::gemm(
        format,
        a,
        MatrixLayout::row_major(k),
        b,
        MatrixLayout::row_major(n),
        c,
        m,
        k,
        n,
    );
}

/// Compute `A @ B` (numpy semantics: batched, broadcast leading dims, 1-D
/// operand promotion) into a dense row-major `Vec<f32>`.
///
/// Operands may be any float dtype (`f32`/`f16`/`bf16`/`f64`). Contiguous half
/// inputs use the blocked half GEMM; other low/medium precision layouts widen to
/// dense `f32`. Both routes accumulate in `f32`. Shared by [`MatMulKernel`] and
/// the fused `FusedMatMulBias` kernel.
pub(crate) fn matmul_dense(a: &TensorView, b: &TensorView) -> Result<Vec<f32>> {
    let geom = matmul_geometry(a, b)?;
    if let Some(result) = try_matmul_half(a, b, &geom)? {
        return Ok(result);
    }
    matmul_dense_impl_with_geom(
        to_dense_f32_widen("MatMul", a)?,
        to_dense_f32_widen("MatMul", b)?,
        &geom,
        CpuBackend::auto_detect(),
        None,
    )
}

pub(crate) fn matmul_dense_prepacked(
    a: &TensorView,
    b: &TensorView,
    prepack: &MatMulPrepack,
) -> Result<Vec<f32>> {
    matmul_dense_prepacked_with_backend(a, b, prepack, CpuBackend::auto_detect())
}

/// Compute `MatMul(A, B)` directly into a caller-supplied output slice,
/// skipping the intermediate `Vec<f32>` allocation. Returns the number of
/// elements written (equal to the result length from matmul_geometry).
/// Used by FusedMatMulBias's direct-output path.
pub(crate) fn matmul_dense_prepacked_into(
    a: &TensorView,
    b: &TensorView,
    prepack: &MatMulPrepack,
    out: &mut [f32],
) -> Result<usize> {
    let backend = CpuBackend::auto_detect();
    let geom = matmul_geometry(a, b)?;
    if out.len() < geom.result_len {
        return Err(EpError::KernelFailed(format!(
            "FusedMatMulBias direct: output buffer {} < result length {}",
            out.len(),
            geom.result_len
        )));
    }
    let out_slice = &mut out[..geom.result_len];
    matmul_dense_into_with_backend(
        &prepack.dense(0, a)?,
        &prepack.dense(1, b)?,
        &geom,
        backend,
        Some(prepack),
        out_slice,
    )?;
    Ok(geom.result_len)
}

fn matmul_dense_prepacked_with_backend(
    a: &TensorView,
    b: &TensorView,
    prepack: &MatMulPrepack,
    backend: CpuBackend,
) -> Result<Vec<f32>> {
    let geom = matmul_geometry(a, b)?;
    if let Some(result) = try_matmul_half(a, b, &geom)? {
        return Ok(result);
    }
    matmul_dense_impl_with_geom(
        prepack.dense(0, a)?,
        prepack.dense(1, b)?,
        &geom,
        backend,
        Some(prepack),
    )
}

/// Shared owned-vector GEMM: allocate the result buffer and GEMM into it.
/// Geometry is pre-computed by the caller to avoid redundant derivation.
fn matmul_dense_impl_with_geom(
    a_dense: Cow<'_, [f32]>,
    b_dense: Cow<'_, [f32]>,
    geom: &MatMulGeometry,
    backend: CpuBackend,
    prepack: Option<&MatMulPrepack>,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; geom.result_len];
    matmul_dense_into_with_backend(&a_dense, &b_dense, geom, backend, prepack, &mut out)?;
    Ok(out)
}

/// Precomputed MatMul dimensions: 1-D promotion, inner-dim agreement, batch
/// broadcast, and per-tile element counts. Computed once so both the owned and
/// direct-output paths share exactly one geometry derivation.
struct MatMulGeometry {
    m: usize,
    k: usize,
    n: usize,
    a_mat: usize,
    b_mat: usize,
    c_mat: usize,
    a_batch: Vec<usize>,
    b_batch: Vec<usize>,
    a_batch_strides: Vec<i64>,
    b_batch_strides: Vec<i64>,
    batch_shape: Vec<usize>,
    /// Promoted rank of B; the MLAS PackedB path applies only to a 2-D B.
    #[cfg_attr(not(feature = "mlas"), allow(dead_code))]
    b_promoted_rank: usize,
    /// Total elements in the result: `batch_count * c_mat`.
    result_len: usize,
}

/// Derive [`MatMulGeometry`] from the two operand views (numpy matmul
/// semantics: 1-D promotion, inner-dim check, broadcast leading dims).
fn matmul_geometry(a: &TensorView, b: &TensorView) -> Result<MatMulGeometry> {
    // Promote 1-D operands per numpy matmul: a [K] -> [1,K] (drop row after),
    // b [K] -> [K,1] (drop col after).
    let a_raw = a.shape;
    let b_raw = b.shape;
    let a_1d = a_raw.len() == 1;
    let b_1d = b_raw.len() == 1;
    let a_shape: Vec<usize> = if a_1d {
        vec![1, a_raw[0]]
    } else {
        a_raw.to_vec()
    };
    let b_shape: Vec<usize> = if b_1d {
        vec![b_raw[0], 1]
    } else {
        b_raw.to_vec()
    };

    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(EpError::KernelFailed(
            "MatMul: operands must be at least 1-D".into(),
        ));
    }

    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    if k != k2 {
        return Err(EpError::KernelFailed(format!(
            "MatMul: inner dims disagree ({k} vs {k2})"
        )));
    }

    // Broadcast the batch (leading) dimensions.
    let a_batch = a_shape[..a_shape.len() - 2].to_vec();
    let b_batch = b_shape[..b_shape.len() - 2].to_vec();
    let batch_shape = broadcast_shapes(&a_batch, &b_batch)?;
    let batch_count = numel(&batch_shape);

    let a_batch_strides = compute_contiguous_strides(&a_batch);
    let b_batch_strides = compute_contiguous_strides(&b_batch);
    let a_mat = m * k;
    let b_mat = k * n;
    let c_mat = m * n;

    Ok(MatMulGeometry {
        m,
        k,
        n,
        a_mat,
        b_mat,
        c_mat,
        a_batch,
        b_batch,
        a_batch_strides,
        b_batch_strides,
        batch_shape,
        b_promoted_rank: b_shape.len(),
        result_len: batch_count * c_mat,
    })
}

/// Run the GEMM (single, batched, or broadcast) into the caller-supplied
/// row-major `out` slice. `out.len()` MUST equal `geom.result_len`; a mismatch
/// returns an `EpError` before any write. This is the single code path shared by
/// the owned-vector wrapper and the direct-output fast path.
fn matmul_dense_into_with_backend(
    a_dense: &[f32],
    b_dense: &[f32],
    geom: &MatMulGeometry,
    backend: CpuBackend,
    prepack: Option<&MatMulPrepack>,
    out: &mut [f32],
) -> Result<()> {
    if out.len() != geom.result_len {
        return Err(EpError::KernelFailed(format!(
            "MatMul: output buffer length {} does not match result length {}",
            out.len(),
            geom.result_len
        )));
    }

    // Any zero dimension (batch, M, or N) yields an empty result — matching
    // numpy/ONNX reference semantics. Return before the compute loop, which
    // otherwise runs once even for a zero-sized batch (a `loop { … } while`) and
    // would index into empty operand slices.
    if out.is_empty() {
        return Ok(());
    }

    let (m, k, n) = (geom.m, geom.k, geom.n);
    let (a_mat, b_mat, c_mat) = (geom.a_mat, geom.b_mat, geom.c_mat);

    // `prepack` is consumed on macOS/iOS (Accelerate transposed_b) and with the
    // `mlas` feature (packed_b). On other platforms it is passed through for API
    // consistency but not yet consumed.
    #[cfg(not(any(feature = "mlas", target_os = "macos", target_os = "ios")))]
    let _ = &prepack;

    #[cfg(feature = "mlas")]
    let packed_b = if backend == CpuBackend::Mlas && geom.b_promoted_rank == 2 {
        prepack.and_then(|prepack| prepack.packed_b(b_dense, k, n))
    } else {
        None
    };

    // Accelerate hybrid M=1 fast path (macOS/iOS):
    //
    // Column-parallel NEON GEMV on pre-transposed B_T, parallelized via the
    // Rayon decode pool. Single-threaded for small matrices (dispatch > compute).
    // Accelerate sgemv was measured at ~30-50 µs GCD wake-up overhead per call,
    // making it SLOWER than single-threaded NEON for L2-resident matrices (the
    // wake-up dominates the compute saving). Accelerate sgemm is retained for
    // M>1 prefill where AMX provides genuine compute benefit.
    //
    // Treat batch_shape with product ≤ 1 as non-batched: during decode the
    // activation is typically [1, 1, K] against a [K, N] weight, giving
    // batch_shape = [1]. The single-element batch is equivalent to no batch
    // and must not skip this optimized path.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if backend == CpuBackend::Accelerate
        && m == 1
        && numel(&geom.batch_shape) <= 1
        && geom.b_promoted_rank == 2
    {
        if let Some(bt) = prepack.and_then(|p| p.transposed_b(b_dense, k, n)) {
            accelerate_gemm::neon_gemv_col_parallel(a_dense, bt, out, k, n);
        } else {
            accelerate_gemm::neon_gemv_parallel(a_dense, b_dense, out, k, n);
        }
        return Ok(());
    }

    // A batch product of 1 (e.g. batch_shape = [1] from a [1,1,K] activation
    // against a [K,N] weight) is equivalent to no batch. Use the non-batched
    // path to pick up packed_b, avoid the per-batch iteration overhead, and
    // stay consistent with the Accelerate M=1 fast path above.
    if numel(&geom.batch_shape) <= 1 {
        // No effective batch dims: a single matmul.
        #[cfg(feature = "mlas")]
        if let Some(packed_b) = packed_b {
            gemm_packed(a_dense, packed_b, out, m, k, n)?;
        } else {
            gemm_with_backend(backend, a_dense, b_dense, out, m, k, n)?;
        }
        #[cfg(not(feature = "mlas"))]
        gemm_with_backend(backend, a_dense, b_dense, out, m, k, n)?;
    } else {
        let mut bidx = vec![0usize; geom.batch_shape.len()];
        let mut b_out = 0usize;
        loop {
            let a_off = broadcast_offset(&bidx, &geom.a_batch, &geom.a_batch_strides) * a_mat;
            let b_off = broadcast_offset(&bidx, &geom.b_batch, &geom.b_batch_strides) * b_mat;
            let a_tile = &a_dense[a_off..a_off + a_mat];
            let c_tile = &mut out[b_out * c_mat..b_out * c_mat + c_mat];
            #[cfg(feature = "mlas")]
            if let Some(packed_b) = packed_b {
                gemm_packed(a_tile, packed_b, c_tile, m, k, n)?;
            } else {
                gemm_with_backend(
                    backend,
                    a_tile,
                    &b_dense[b_off..b_off + b_mat],
                    c_tile,
                    m,
                    k,
                    n,
                )?;
            }
            #[cfg(not(feature = "mlas"))]
            gemm_with_backend(
                backend,
                a_tile,
                &b_dense[b_off..b_off + b_mat],
                c_tile,
                m,
                k,
                n,
            )?;
            b_out += 1;
            if !next_index(&geom.batch_shape, &mut bidx) {
                break;
            }
        }
    }

    Ok(())
}

/// Element offset of batch index `bidx` into a batch of shape `batch`,
/// broadcasting any size-1 axis (stride 0). `bidx` is indexed over the
/// broadcast (output) batch shape, right-aligned onto `batch`.
fn broadcast_offset(bidx: &[usize], batch: &[usize], batch_strides: &[i64]) -> usize {
    let out_rank = bidx.len();
    let mut off = 0i64;
    for axis in 0..batch.len() {
        let out_axis = axis + (out_rank - batch.len());
        let i = if batch[axis] == 1 { 0 } else { bidx[out_axis] };
        off += batch_strides[axis] * i as i64;
    }
    off as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn matmul_zero_batch_returns_empty_without_panicking() {
        // Regression: a zero-sized batch dim (broadcast to a 0-length result)
        // used to run the compute loop once and index empty operand slices,
        // panicking. It must return an empty buffer instead (numpy/ONNX
        // reference semantics).
        let a = Owned::f32(&[0, 1, 1], &[]);
        let b = Owned::f32(&[0, 1, 1], &[]);
        let out = matmul_dense(&a.view(), &b.view()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn matmul_2x3_times_3x2() {
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        // C = [[58,64],[139,154]]
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![58., 64., 139., 154.]);
    }

    #[cfg(feature = "tracing")]
    #[test]
    fn matmul_populates_active_trace_span_metrics() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        let (trace, events) = onnx_runtime_tracer::TraceContext::in_memory();
        {
            let _span = trace.span("MatMul", "compute");
            MatMulKernel::default()
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();
        }

        let events = events.events();
        let args = events[0].args.as_ref().expect("MatMul trace args");
        assert_eq!(args["bytes"], 64);
        assert_eq!(args["flops"], 24);
    }

    #[test]
    fn matmul_with_transposed_b_view() {
        // B stored as [2,3] row-major, exposed transposed as [3,2] strides [1,3].
        // A[2,3] @ Bt[3,2] where Bt = B.T.
        // B = [[7,9,11],[8,10,12]] stored; Bt = [[7,8],[9,10],[11,12]].
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[2, 3], &[7., 9., 11., 8., 10., 12.]).with_view(&[3, 2], &[1, 3]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // Same as the contiguous case above.
        assert_eq!(out.to_f32(), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn matmul_batched() {
        // Two independent [2,2] matmuls.
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2, 2], &[1., 0., 0., 1., 2., 0., 0., 2.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // batch0: A@I = A; batch1: [[5,6],[7,8]]*2 = [[10,12],[14,16]]
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 10., 12., 14., 16.]);
    }

    #[test]
    fn matmul_broadcast_batch() {
        // A [2,2,2] @ B [2,2] (broadcast B over batch)
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2], &[1., 0., 0., 1.]); // identity
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn matmul_vector_times_matrix() {
        // a [3] @ B [3,2] -> [2]
        let a = Owned::f32(&[3], &[1., 2., 3.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        assert_eq!(out.to_f32(), vec![58., 64.]);
    }

    #[test]
    fn matmul_f16_accumulates_in_f32() {
        // A[2,3] @ B[3,2] in f16; compute widens to f32, result rounds to f16.
        let a = Owned::f16(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f16(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Float16, &[2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f16_as_f32(), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn matmul_half_dispatch_matches_widened_reference_across_irregular_shapes() {
        use onnx_runtime_ir::DataType;

        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 127, 65),
            (3, 5, 7),
            (17, 130, 11),
            (5, 257, 2),
            (2, 0, 3),
        ];
        let mut state = 0x51A7_CAFE_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.5
        };

        for dtype in [DataType::Float16, DataType::BFloat16] {
            for &(m, k, n) in SHAPES {
                let a_source: Vec<f32> = (0..m * k).map(|_| next()).collect();
                let b_source: Vec<f32> = (0..k * n).map(|_| next()).collect();
                let round = |value: f32| match dtype {
                    DataType::Float16 => half::f16::from_f32(value).to_f32(),
                    DataType::BFloat16 => half::bf16::from_f32(value).to_f32(),
                    _ => unreachable!(),
                };
                let a_wide: Vec<f32> = a_source.iter().copied().map(round).collect();
                let b_wide: Vec<f32> = b_source.iter().copied().map(round).collect();
                let a = match dtype {
                    DataType::Float16 => Owned::f16(&[m, k], &a_source),
                    DataType::BFloat16 => Owned::bf16(&[m, k], &a_source),
                    _ => unreachable!(),
                };
                let b = match dtype {
                    DataType::Float16 => Owned::f16(&[k, n], &b_source),
                    DataType::BFloat16 => Owned::bf16(&[k, n], &b_source),
                    _ => unreachable!(),
                };
                let geometry = matmul_geometry(&a.view(), &b.view()).unwrap();
                assert!(
                    try_matmul_half(&a.view(), &b.view(), &geometry)
                        .unwrap()
                        .is_some(),
                    "{dtype:?} should select the dedicated half GEMM"
                );

                let mut expected = vec![0.0f32; m * n];
                for row in 0..m {
                    for column in 0..n {
                        for depth in 0..k {
                            expected[row * n + column] +=
                                a_wide[row * k + depth] * b_wide[depth * n + column];
                        }
                        expected[row * n + column] = round(expected[row * n + column]);
                    }
                }

                let mut first = Owned::zeros(dtype, &[m, n]);
                let mut second = Owned::zeros(dtype, &[m, n]);
                MatMulKernel::default()
                    .execute(&[a.view(), b.view()], &mut [first.view_mut()])
                    .unwrap();
                MatMulKernel::default()
                    .execute(&[a.view(), b.view()], &mut [second.view_mut()])
                    .unwrap();
                let first = match dtype {
                    DataType::Float16 => first.to_f16_as_f32(),
                    DataType::BFloat16 => first.to_bf16_as_f32(),
                    _ => unreachable!(),
                };
                let second = match dtype {
                    DataType::Float16 => second.to_f16_as_f32(),
                    DataType::BFloat16 => second.to_bf16_as_f32(),
                    _ => unreachable!(),
                };
                assert_eq!(first, second, "{dtype:?} {m}x{k}x{n} was not deterministic");

                let tolerance = match dtype {
                    DataType::Float16 => 2e-3,
                    DataType::BFloat16 => 3e-2,
                    _ => unreachable!(),
                };
                let max_error = first
                    .iter()
                    .zip(expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_error <= tolerance,
                    "{dtype:?} {m}x{k}x{n}: max error {max_error} exceeds {tolerance}"
                );
            }
        }
    }

    #[test]
    fn matmul_f16_preserves_near_tie_argmax_after_f32_accumulation() {
        // The f32 reference margin (0.01171875) is below one f16 ULP at 20.
        // Widened accumulation must still retain the correct first-column winner
        // before the single final f16 narrowing.
        let a = Owned::f16(&[1, 4], &[1., 1., 1., 1.]);
        // Exact f16 representable input used to exercise the near-tie boundary.
        #[allow(clippy::excessive_precision)]
        let b = Owned::f16(
            &[4, 2],
            &[5.00390625, 5.0, 5.00390625, 5.0, 5.00390625, 5.0, 5.0, 5.0],
        );
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Float16, &[1, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f16_as_f32(), vec![20.015625, 20.0]);
    }

    #[test]
    fn matmul_bf16_batched() {
        let a = Owned::bf16(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::bf16(&[2, 2, 2], &[1., 0., 0., 1., 2., 0., 0., 2.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_bf16_as_f32(),
            vec![1., 2., 3., 4., 10., 12., 14., 16.]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_bf16_native_matches_f64_reference_within_bf16_tolerance() {
        // Native `_mm512_dpbf16_ps` GEMM must (a) match an f64 reference over the
        // SAME bf16-rounded operands within bf16 tolerance, and (b) be no worse
        // than the widen-to-f32 upcast path — bf16-input products are exact in
        // f32, so both paths differ from f64 only by f32 summation rounding.
        if !bf16_gemm::native_available() {
            eprintln!("skipping: host lacks avx512_bf16");
            return;
        }
        // Shapes exercise m=1 decode, general prefill, and K/N-32 tails.
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 2048, 512), // decode GEMV
            (1, 100, 40),   // decode, K & N not multiples of 32
            (32, 256, 64),  // prefill, aligned
            (17, 130, 50),  // prefill, ragged M/K/N
            (4, 33, 3),     // tiny K tail (33 = 32 + 1)
        ];

        let mut state = 0x9E37_79B9_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 // ~[-1, 1]
        };

        let mut worst_native_rel = 0.0f64;
        let mut worst_ratio = 0.0f64;
        for &(m, k, n) in SHAPES {
            let a_f32: Vec<f32> = (0..m * k).map(|_| next()).collect();
            let b_f32: Vec<f32> = (0..k * n).map(|_| next()).collect();

            // Round operands to bf16; both compute paths and the reference share
            // these exact bf16 values so the only difference is accumulation.
            let a_bf: Vec<half::bf16> = a_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let b_bf: Vec<half::bf16> = b_f32.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let a_bits: Vec<u16> = a_bf.iter().map(|v| v.to_bits()).collect();
            let b_bits: Vec<u16> = b_bf.iter().map(|v| v.to_bits()).collect();
            let a_wide: Vec<f32> = a_bf.iter().map(|v| v.to_f32()).collect();
            let b_wide: Vec<f32> = b_bf.iter().map(|v| v.to_f32()).collect();

            // f64 reference over the bf16-rounded values.
            let mut reference = vec![0.0f64; m * n];
            // Upcast reference: widen bf16 -> f32, accumulate in f32.
            let mut upcast = vec![0.0f32; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut acc64 = 0.0f64;
                    let mut acc32 = 0.0f32;
                    for depth in 0..k {
                        acc64 += a_wide[row * k + depth] as f64 * b_wide[depth * n + col] as f64;
                        acc32 += a_wide[row * k + depth] * b_wide[depth * n + col];
                    }
                    reference[row * n + col] = acc64;
                    upcast[row * n + col] = acc32;
                }
            }

            // Native bf16 path.
            let mut native = vec![0.0f32; m * n];
            bf16_gemm::gemm(&a_bits, &b_bits, &mut native, m, k, n);

            let rel = |got: f32, want: f64| -> f64 {
                let denom = want.abs().max(1.0);
                (got as f64 - want).abs() / denom
            };
            let mut max_native = 0.0f64;
            let mut max_upcast = 0.0f64;
            for idx in 0..m * n {
                max_native = max_native.max(rel(native[idx], reference[idx]));
                max_upcast = max_upcast.max(rel(upcast[idx], reference[idx]));
            }
            worst_native_rel = worst_native_rel.max(max_native);
            // Native must not be materially worse than the upcast reference.
            let ratio = max_native / max_upcast.max(1e-9);
            worst_ratio = worst_ratio.max(ratio);
            assert!(
                max_native <= max_upcast * 4.0 + 1e-4,
                "{m}x{k}@{k}x{n}: native rel {max_native} worse than upcast {max_upcast}"
            );
            // bf16 accumulation over K stays within a loose bf16 tolerance.
            assert!(
                max_native <= 5e-2,
                "{m}x{k}@{k}x{n}: native rel {max_native} exceeds bf16 tolerance"
            );
        }
        println!(
            "native bf16 GEMM: worst native-vs-f64 rel {worst_native_rel:.3e}, \
             worst native/upcast ratio {worst_ratio:.3}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn matmul_bf16_native_handles_k_tail_and_kernel_matches_kernel_path() {
        // The public MatMul kernel (which auto-routes to the native path on this
        // host) must produce a bf16 result matching a direct f64 reference for a
        // K that is not a multiple of the 32-lane bf16 width.
        if !bf16_gemm::native_available() {
            eprintln!("skipping: host lacks avx512_bf16");
            return;
        }
        let (m, k, n) = (3usize, 70usize, 5usize); // 70 = 64 + 6 tail
        let mut state = 0x1357_9BDF_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 1.0
        };
        let a_f32: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let a = Owned::bf16(&[m, k], &a_f32);
        let b = Owned::bf16(&[k, n], &b_f32);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::BFloat16, &[m, n]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();

        // f64 reference over bf16-rounded operands, then rounded to bf16 output.
        let a_w: Vec<f32> = a_f32
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let b_w: Vec<f32> = b_f32
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let got = out.to_bf16_as_f32();
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for depth in 0..k {
                    acc += a_w[row * k + depth] as f64 * b_w[depth * n + col] as f64;
                }
                let want = half::bf16::from_f32(acc as f32).to_f32();
                let denom = want.abs().max(1.0);
                let rel = (got[row * n + col] - want).abs() / denom;
                assert!(rel <= 3e-2, "K-tail mismatch at ({row},{col}): rel {rel}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[ignore = "microbench: run explicitly with --ignored --nocapture"]
    fn bench_bf16_native_vs_upcast() {
        use std::time::Instant;
        if !bf16_gemm::native_available() {
            eprintln!("skipping bench: host lacks avx512_bf16");
            return;
        }
        // Decode (m=1 GEMV) and prefill (MxKxN) shapes, LLM-representative.
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 4096, 4096),   // decode GEMV
            (1, 4096, 11008),  // decode MLP up-proj
            (128, 4096, 4096), // prefill attention proj
            (256, 2048, 8192), // prefill MLP
        ];
        let mut state = 0x5DEE_CE66_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0
        };
        let median3 = |mut f: Box<dyn FnMut() -> f64>| {
            let mut t = [f(), f(), f()];
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[1]
        };

        println!("bf16 GEMM microbench (native _mm512_dpbf16_ps vs widen-to-f32 + SGEMM)");
        for &(m, k, n) in SHAPES {
            let a_bits: Vec<u16> = (0..m * k)
                .map(|_| half::bf16::from_f32(next()).to_bits())
                .collect();
            let b_bits: Vec<u16> = (0..k * n)
                .map(|_| half::bf16::from_f32(next()).to_bits())
                .collect();
            let flops = 2.0 * m as f64 * k as f64 * n as f64;

            // Native bf16 path.
            let native_ms = {
                let (a, b) = (a_bits.clone(), b_bits.clone());
                median3(Box::new(move || {
                    let mut c = vec![0.0f32; m * n];
                    let t = Instant::now();
                    bf16_gemm::gemm(&a, &b, &mut c, m, k, n);
                    std::hint::black_box(&c);
                    t.elapsed().as_secs_f64() * 1e3
                }))
            };

            // Upcast reference: widen bf16 -> f32, then use the crate's f32
            // SGEMM.
            let upcast_ms = {
                let (a, b) = (a_bits.clone(), b_bits.clone());
                median3(Box::new(move || {
                    let t = Instant::now();
                    let a_f: Vec<f32> = a
                        .iter()
                        .map(|&x| half::bf16::from_bits(x).to_f32())
                        .collect();
                    let b_f: Vec<f32> = b
                        .iter()
                        .map(|&x| half::bf16::from_bits(x).to_f32())
                        .collect();
                    let mut c = vec![0.0f32; m * n];
                    gemm(&a_f, &b_f, &mut c, m, k, n).unwrap();
                    std::hint::black_box(&c);
                    t.elapsed().as_secs_f64() * 1e3
                }))
            };

            let g = |ms: f64| flops / (ms * 1e-3) / 1e9;
            println!(
                "  {m:>4}x{k}x{n}: native {native_ms:>8.3} ms ({:>7.1} GFLOP/s)  \
                 upcast {upcast_ms:>8.3} ms ({:>7.1} GFLOP/s)  speedup {:.2}x",
                g(native_ms),
                g(upcast_ms),
                upcast_ms / native_ms,
            );
        }
    }

    #[test]
    fn matmul_rejects_integer_dtype_with_rule1() {
        let a = Owned::i32(&[2, 2], &[1, 2, 3, 4]);
        let b = Owned::i32(&[2, 2], &[1, 0, 0, 1]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Int32, &[2, 2]);
        let err = MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap_err();
        assert!(format!("{err}").contains("WHAT"));
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn matmul_generic_block_boundaries_match_naive_reference() {
        const SHAPES: &[(usize, usize, usize)] = &[
            (65, 257, 70),
            (128, 300, 200),
            (100, 64, 4),
            (4, 256, 4),
            (1, 512, 1),
            (200, 1, 200),
        ];
        const ABS_TOLERANCE: f32 = 1e-3;

        let mut state = 0x1234_5678_u32;
        let mut next_f32 = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.25
        };

        let mut overall_max_abs_error = 0.0f32;
        for &(m, k, n) in SHAPES {
            let a_data: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();
            let b_data: Vec<f32> = (0..k * n).map(|_| next_f32()).collect();

            let mut reference = vec![0.0f32; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0f32;
                    for depth in 0..k {
                        sum += a_data[row * k + depth] * b_data[depth * n + col];
                    }
                    reference[row * n + col] = sum;
                }
            }

            let a = Owned::f32(&[m, k], &a_data);
            let b = Owned::f32(&[k, n], &b_data);
            let mut out = Owned::zeros_f32(&[m, n]);
            MatMulKernel::default()
                .execute(&[a.view(), b.view()], &mut [out.view_mut()])
                .unwrap();

            let actual = out.to_f32();
            let max_abs_error = actual
                .iter()
                .zip(&reference)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            overall_max_abs_error = overall_max_abs_error.max(max_abs_error);
            assert!(
                max_abs_error <= ABS_TOLERANCE,
                "{m}x{k} @ {k}x{n}: max abs error {max_abs_error} exceeds {ABS_TOLERANCE}"
            );
        }

        println!("generic MatMul max abs error: {overall_max_abs_error}");
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn default_gemm_backend_matches_generic_reference() {
        // Locks the auto-detected default GEMM path: after flipping the f32
        // default to MLAS, `gemm()` (which dispatches on `CpuBackend::auto_detect`)
        // must still match the generic reference within f32 tolerance, so the
        // faster default never changes decode outputs (token-ID parity).
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 2048, 2048), // decode GEMV (m=1), the dense hot path
            (1, 2304, 9216), // GeGLU up/gate projection tile
            (1, 9216, 2304), // GeGLU down projection tile
            (5, 128, 256),
            (32, 512, 512),
        ];
        assert_eq!(CpuBackend::auto_detect(), CpuBackend::Mlas);
        let mut state = 0x1234_abcd_u32;
        let mut next_f32 = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.25
        };
        for &(m, k, n) in SHAPES {
            let a: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| next_f32()).collect();
            let mut expected = vec![0.0; m * n];
            let mut actual = vec![0.0; m * n];
            gemm_generic(&a, &b, &mut expected, m, k, n);
            gemm(&a, &b, &mut actual, m, k, n).unwrap();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 1e-3,
                "{m}x{k} @ {k}x{n}: default-backend max error {max_error} exceeds tolerance"
            );
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_gemm_matches_generic_for_matrix_and_batched_vector_tiles() {
        const SHAPES: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (7, 13, 5),
            (32, 512, 512),
            (97, 11, 3),
            // Each tile below is how batched and vector MatMul route through gemm.
            (1, 13, 5),
            (3, 13, 1),
        ];
        let mut state = 0x5eed_1234_u32;
        let mut next_f32 = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.25
        };

        for &(m, k, n) in SHAPES {
            let a: Vec<f32> = (0..m * k).map(|_| next_f32()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| next_f32()).collect();
            let mut expected = vec![0.0; m * n];
            let mut actual = vec![0.0; m * n];
            gemm_generic(&a, &b, &mut expected, m, k, n);
            gemm_with_backend(CpuBackend::Mlas, &a, &b, &mut actual, m, k, n).unwrap();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error <= 1e-3,
                "{m}x{k} @ {k}x{n}: MLAS max error {max_error} exceeds tolerance"
            );
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_constant_b_packed_kernel_matches_unpacked_and_generic() {
        for (m, k, n) in [(5usize, 17usize, 9usize), (33, 64, 48)] {
            let a_data: Vec<f32> = (0..m * k)
                .map(|i| ((i as f32 * 0.037).sin()) * 0.25)
                .collect();
            let b_data: Vec<f32> = (0..k * n)
                .map(|i| ((i as f32 * 0.021 + 0.3).cos()) * 0.25)
                .collect();
            let a = Owned::f32(&[m, k], &a_data);
            let b = Owned::f32(&[k, n], &b_data);
            let mut out = Owned::zeros_f32(&[m, n]);
            let mut kernel = MatMulKernel::default();
            kernel.set_constant_inputs(&[false, true]);
            kernel
                .execute_with_backend(
                    &[a.view(), b.view()],
                    &mut [out.view_mut()],
                    CpuBackend::Mlas,
                )
                .unwrap();

            let mut unpacked = vec![0.0; m * n];
            let mut generic = vec![0.0; m * n];
            gemm_with_backend(CpuBackend::Mlas, &a_data, &b_data, &mut unpacked, m, k, n).unwrap();
            gemm_with_backend(CpuBackend::Generic, &a_data, &b_data, &mut generic, m, k, n)
                .unwrap();

            let packed = out.to_f32();
            for (index, ((packed, unpacked), generic)) in
                packed.iter().zip(&unpacked).zip(&generic).enumerate()
            {
                assert!(
                    (packed - unpacked).abs() <= 1e-4,
                    "{m}x{k}x{n} packed/unpacked mismatch at {index}: {packed} vs {unpacked}"
                );
                assert!(
                    (packed - generic).abs() <= 1e-3,
                    "{m}x{k}x{n} packed/generic mismatch at {index}: {packed} vs {generic}"
                );
            }
            assert!(kernel.prepack.packed_b.get().is_some());
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_constant_b_packed_buffer_is_reused() {
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        let weight_data: Vec<f32> = (0..17 * 9)
            .map(|i| ((i as f32 * 0.031).sin()) * 0.5)
            .collect();
        let weight = Owned::f16(&[17, 9], &weight_data);

        let a1_data: Vec<f32> = (0..5 * 17).map(|i| i as f32 * 0.01).collect();
        let a1 = Owned::f32(&[5, 17], &a1_data);
        let mut out1 = Owned::zeros_f32(&[5, 9]);
        kernel
            .execute_with_backend(
                &[a1.view(), weight.view()],
                &mut [out1.view_mut()],
                CpuBackend::Mlas,
            )
            .unwrap();
        let packed_ptr = kernel.prepack.packed_b.get().unwrap() as *const mlas_sys::PackedB;
        let dense_ptr = kernel.prepack.dense[1].get().unwrap().as_ptr();

        let a2_data: Vec<f32> = (0..5 * 17)
            .map(|i| ((i as f32 * 0.07).cos()) * 0.2)
            .collect();
        let a2 = Owned::f32(&[5, 17], &a2_data);
        let mut out2 = Owned::zeros_f32(&[5, 9]);
        kernel
            .execute_with_backend(
                &[a2.view(), weight.view()],
                &mut [out2.view_mut()],
                CpuBackend::Mlas,
            )
            .unwrap();

        assert_eq!(
            kernel.prepack.packed_b.get().unwrap() as *const mlas_sys::PackedB,
            packed_ptr
        );
        assert_eq!(kernel.prepack.dense[1].get().unwrap().as_ptr(), dense_ptr);
        assert!(kernel.prepack.dense[0].get().is_none());
        assert_ne!(out1.to_f32(), out2.to_f32());
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_packed_cache_requires_mlas_constant_unbatched_b() {
        let (m, k, n) = (5usize, 17usize, 9usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 * 0.02).sin()) * 0.1)
            .collect();
        let a = Owned::f32(&[m, k], &a_data);
        let b = Owned::f32(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[m, n]);
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, false]);
        kernel
            .execute_with_backend(
                &[a.view(), b.view()],
                &mut [out.view_mut()],
                CpuBackend::Mlas,
            )
            .unwrap();

        let mut expected = vec![0.0; m * n];
        gemm_generic(&a_data, &b_data, &mut expected, m, k, n);
        assert!(kernel.prepack.packed_b.get().is_none());
        for (actual, expected) in out.to_f32().iter().zip(&expected) {
            assert!((actual - expected).abs() <= 1e-3);
        }

        let mut generic_kernel = MatMulKernel::default();
        generic_kernel.set_constant_inputs(&[false, true]);
        let mut generic_out = Owned::zeros_f32(&[m, n]);
        generic_kernel
            .execute_with_backend(
                &[a.view(), b.view()],
                &mut [generic_out.view_mut()],
                CpuBackend::Generic,
            )
            .unwrap();
        assert!(generic_kernel.prepack.packed_b.get().is_none());
        assert_eq!(generic_out.to_f32(), expected);

        let batched_b_data = [b_data.clone(), b_data].concat();
        let batched_a_data = [a_data.clone(), a_data].concat();
        let batched_a = Owned::f32(&[2, m, k], &batched_a_data);
        let batched_b = Owned::f32(&[2, k, n], &batched_b_data);
        let mut batched_out = Owned::zeros_f32(&[2, m, n]);
        let mut batched_kernel = MatMulKernel::default();
        batched_kernel.set_constant_inputs(&[false, true]);
        batched_kernel
            .execute_with_backend(
                &[batched_a.view(), batched_b.view()],
                &mut [batched_out.view_mut()],
                CpuBackend::Mlas,
            )
            .unwrap();
        assert!(batched_kernel.prepack.packed_b.get().is_none());
        for (actual, expected) in batched_out.to_f32().iter().zip(expected.iter().cycle()) {
            assert!((actual - expected).abs() <= 1e-3);
        }
    }

    #[cfg(feature = "mlas")]
    #[test]
    fn mlas_selects_a_float_kernel_on_x86_64() {
        assert_ne!(mlas_sys::selected_float_kernel(), 0);
    }

    #[test]
    fn constant_weight_prepack_reuses_weight_and_keeps_activation_live() {
        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);
        let weight = Owned::f16(&[2, 2], &[2., 0., 0., 3.]);

        let a1 = Owned::f32(&[1, 2], &[1., 2.]);
        let mut out1 = Owned::zeros_f32(&[1, 2]);
        kernel
            .execute(&[a1.view(), weight.view()], &mut [out1.view_mut()])
            .unwrap();
        assert_eq!(out1.to_f32(), vec![2., 6.]);
        // On macOS with M=1 and f16 weights, the f16 GEMV path reads weights
        // directly as f16 (via transposed_b_f16) and never widens to f32 —
        // so dense[1] is NOT populated. On other platforms, dense[1] IS populated.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert!(kernel.prepack.transposed_b_f16.get().is_some());
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        assert!(kernel.prepack.dense[1].get().is_some());
        assert!(kernel.prepack.dense[0].get().is_none());

        // Capture the cache pointer *before* the second execute so the
        // comparison below is a real guard: it proves the first call populated
        // the cache and the second reused it (rather than repopulating).
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let ptr_before = kernel.prepack.transposed_b_f16.get().unwrap().as_ptr();
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let ptr_before = kernel.prepack.dense[1].get().unwrap().as_ptr();

        let a2 = Owned::f32(&[1, 2], &[4., 5.]);
        let mut out2 = Owned::zeros_f32(&[1, 2]);
        kernel
            .execute(&[a2.view(), weight.view()], &mut [out2.view_mut()])
            .unwrap();
        assert_eq!(out2.to_f32(), vec![8., 15.]);

        // The pointer must be unchanged — the OnceLock was already populated.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert_eq!(
            kernel.prepack.transposed_b_f16.get().unwrap().as_ptr(),
            ptr_before,
            "transposed_b_f16 cache was reallocated on the second execute"
        );
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        assert_eq!(
            kernel.prepack.dense[1].get().unwrap().as_ptr(),
            ptr_before,
            "dense[1] cache was reallocated on the second execute"
        );
    }

    // --- Direct f32 output path (Option A) --------------------------------

    /// Reference row-major GEMM for verifying the direct path numerically.
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

    #[test]
    fn direct_f32_eligible_for_contiguous_cpu_output() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        assert!(output_is_direct_f32_eligible(
            &a.view(),
            &b.view(),
            &out.view_mut()
        ));
    }

    #[test]
    fn direct_f32_rejects_non_f32_output() {
        // f16 output must fall back to the narrowing writer.
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros(onnx_runtime_ir::DataType::Float16, &[2, 2]);
        assert!(!output_is_direct_f32_eligible(
            &a.view(),
            &b.view(),
            &out.view_mut()
        ));
    }

    #[test]
    fn direct_f32_2d_nonsquare_matches_reference() {
        // A[2,3] @ B[3,4] contiguous f32: the direct path writes into `out`.
        let a_data = [1., 2., 3., 4., 5., 6.];
        let b_data = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let a = Owned::f32(&[2, 3], &a_data);
        let b = Owned::f32(&[3, 4], &b_data);
        let mut out = Owned::zeros_f32(&[2, 4]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), naive_matmul(&a_data, &b_data, 2, 3, 4));
    }

    #[test]
    fn direct_f32_batched_and_broadcast_match_reference() {
        // Batched: two independent [2,2] matmuls into one contiguous output.
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2, 2], &[1., 0., 0., 1., 2., 0., 0., 2.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 10., 12., 14., 16.]);

        // Broadcast B over the batch dim.
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2], &[1., 0., 0., 1.]);
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn direct_f32_matrix_times_vector() {
        // A[2,3] @ b[3] -> [2] (b promoted to [3,1], result col squeezed).
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3], &[7., 9., 11.]);
        let mut out = Owned::zeros_f32(&[2]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // [1*7+2*9+3*11, 4*7+5*9+6*11] = [58, 139]
        assert_eq!(out.to_f32(), vec![58., 139.]);
    }

    #[test]
    fn direct_f32_vector_times_vector_scalar_result() {
        // a[3] @ b[3] -> scalar (shape []), a promoted [1,3], b promoted [3,1].
        let a = Owned::f32(&[3], &[1., 2., 3.]);
        let b = Owned::f32(&[3], &[4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[]);
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(out.to_f32(), vec![32.]);
    }

    #[test]
    fn direct_f32_zero_sized_result_writes_nothing() {
        // A zero batch dim yields an empty result; the direct path must return
        // before any GEMM write.
        let a = Owned::f32(&[0, 2, 3], &[]);
        let b = Owned::f32(&[0, 3, 2], &[]);
        let mut out = Owned::zeros_f32(&[0, 2, 2]);
        assert!(output_is_direct_f32_eligible(
            &a.view(),
            &b.view(),
            &out.view_mut()
        ));
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert!(out.to_f32().is_empty());
    }

    #[test]
    fn strided_f32_output_takes_fallback_and_is_correct() {
        // A[2,2] @ B[2,2] into a NON-contiguous [2,2] output: row stride 3 over a
        // [2,3] backing buffer. It must NOT take the direct path; the strided
        // writer scatters into positions 0,1,3,4.
        let a = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let b = Owned::f32(&[2, 2], &[5., 6., 7., 8.]);
        let mut out = Owned::zeros_f32(&[2, 3]).with_view(&[2, 2], &[3, 1]);
        assert!(!out.view_mut().is_contiguous());
        assert!(!output_is_direct_f32_eligible(
            &a.view(),
            &b.view(),
            &out.view_mut()
        ));
        MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // C = [[19,22],[43,50]] scattered: buf[0]=19, buf[1]=22, buf[3]=43, buf[4]=50.
        assert_eq!(out.to_f32(), vec![19., 22., 0., 43., 50., 0.]);
    }

    #[test]
    fn mismatched_output_length_errors_before_write() {
        // A[2,3] @ B[3,2] -> [2,2] (4 elems), but the output view has 6 elems.
        // The direct path must error on the length check before any GEMM write.
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        let err = MatMulKernel::default()
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap_err();
        assert!(format!("{err}").contains("does not match result length"));
        // Nothing was written.
        assert_eq!(out.to_f32(), vec![0.; 6]);
    }

    #[test]
    fn output_overlaps_input_helper_detects_ranges() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::{DataType, DeviceId, DeviceType};

        let buf = [0.0f32; 8];
        let shape = [2usize, 2];
        let strides = compute_contiguous_strides(&shape);
        let base = buf.as_ptr() as usize;
        let bytes = 4 * 4; // 4 f32

        // Input covering buf[0..4].
        let input = TensorView::new(
            DevicePtr(buf.as_ptr() as *const std::ffi::c_void),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );

        // Overlapping output starting inside the input range.
        assert!(output_overlaps_input(
            base + 8,
            base + 8 + bytes,
            &input,
            DeviceId::cpu()
        ));
        // Disjoint output entirely past the input range.
        assert!(!output_overlaps_input(
            base + bytes,
            base + 2 * bytes,
            &input,
            DeviceId::cpu()
        ));
        // Absent input never overlaps.
        assert!(!output_overlaps_input(
            base,
            base + bytes,
            &TensorView::absent(DataType::Float32),
            DeviceId::cpu()
        ));
        // Different device is treated as non-overlapping (distinct address space).
        assert!(!output_overlaps_input(
            base,
            base + bytes,
            &input,
            DeviceId::new(DeviceType::Cuda, 0)
        ));
        let _ = DevicePtrMut(std::ptr::null_mut());
    }

    #[test]
    fn aliasing_output_takes_fallback_and_is_correct() {
        // DeviceIoBinding permits input/output aliasing. Construct an output that
        // shares A's backing buffer; the direct path must be rejected and the
        // owned-buffer fallback must still produce the correct result even though
        // A is read while C is written to the same memory.
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
        use onnx_runtime_ir::{DataType, DeviceId};

        // A = [[1,2],[3,4]] shared with C; B = column swap so C = [[2,1],[4,3]].
        let mut shared = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_buf = [0.0f32, 1.0, 1.0, 0.0];
        let shape = vec![2usize, 2];
        let strides = compute_contiguous_strides(&shape);
        let a_ptr = shared.as_ptr() as *const std::ffi::c_void;
        let c_ptr = shared.as_mut_ptr() as *mut std::ffi::c_void;

        let a = TensorView::new(
            DevicePtr(a_ptr),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let b = TensorView::new(
            DevicePtr(b_buf.as_ptr() as *const std::ffi::c_void),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let c = TensorMut::new(
            DevicePtrMut(c_ptr),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );

        // Output aliases input A: direct path must be rejected.
        assert!(!output_is_direct_f32_eligible(&a, &b, &c));

        MatMulKernel::default().execute(&[a, b], &mut [c]).unwrap();
        // Fallback computed the full result into a temp buffer before writing.
        assert_eq!(shared, vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn accelerate_sgemm_matches_generic_for_small_shapes() {
        let shapes = [(1, 4, 4), (2, 3, 5), (4, 8, 4), (1, 16, 16), (8, 8, 8)];
        for (m, k, n) in shapes {
            let a: Vec<f32> = (0..m * k)
                .map(|i| (i as f32 * 0.7 + 0.3).sin() * 2.0)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| (i as f32 * 1.3 + 0.7).cos() * 2.0)
                .collect();
            let mut generic_out = vec![0.0f32; m * n];
            let mut accel = vec![0.0f32; m * n];
            gemm_with_backend(CpuBackend::Generic, &a, &b, &mut generic_out, m, k, n).unwrap();
            gemm_with_backend(CpuBackend::Accelerate, &a, &b, &mut accel, m, k, n).unwrap();
            for idx in 0..m * n {
                let diff = (generic_out[idx] - accel[idx]).abs();
                assert!(
                    diff < 1e-4,
                    "[{m},{k},{n}][{idx}]: g={} a={} d={}",
                    generic_out[idx],
                    accel[idx],
                    diff
                );
            }
        }
    }

    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn accelerate_decode_gemv_matches_generic_at_model_scale() {
        let shapes = [(1, 896, 896), (1, 896, 4864), (1, 4864, 896)];
        for (m, k, n) in shapes {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.31 + 0.17).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.13 + 0.71).cos()).collect();
            let mut generic_out = vec![0.0f32; m * n];
            let mut accel = vec![0.0f32; m * n];
            gemm_with_backend(CpuBackend::Generic, &a, &b, &mut generic_out, m, k, n).unwrap();
            gemm_with_backend(CpuBackend::Accelerate, &a, &b, &mut accel, m, k, n).unwrap();
            let max_rel = generic_out
                .iter()
                .zip(accel.iter())
                .map(|(g, a)| (g - a).abs() / g.abs().max(1e-8))
                .fold(0.0f32, f32::max);
            // Chew measured the worst model-scale accumulation-order drift at
            // 1.57%; 1.8% keeps modest cross-machine headroom without letting a
            // real GEMV regression hide behind the old 2% envelope.
            assert!(max_rel < 1.8e-2, "[{m},{k},{n}]: max_rel={max_rel}");
        }
    }

    #[test]
    fn col_parallel_gemm_matches_reference_for_m1() {
        let (m, k, n) = (1, 256, 512);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.41 + 0.13).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.23 + 0.57).cos()).collect();
        let mut reference = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                reference[i * n + j] = s;
            }
        }
        let mut col_par = vec![0.0f32; m * n];
        gemm_generic_col_parallel(&a, &b, &mut col_par, m, k, n, 8);
        for idx in 0..m * n {
            let diff = (reference[idx] - col_par[idx]).abs();
            assert!(
                diff < 1e-4,
                "[{m},{k},{n}][{idx}]: ref={} col={} d={}",
                reference[idx],
                col_par[idx],
                diff
            );
        }
    }

    /// Guard: an FP16 weight MatMul at M=1 (decode shape) must reach the NEON
    /// GEMV path on Apple Silicon, not the half-precision blocked GEMM.  If the
    /// dispatch order changes so `try_matmul_half` intercepts M=1 before the
    /// GEMV, decode throughput drops ~4×.  This test would have caught the
    /// `half_gemm.rs` regression that took native FP16 from 60→13 tok/s.
    ///
    /// Both A and B are Float16 to exercise the real fp16-model dispatch: the
    /// regression occurs precisely when `try_matmul_half` matches (f16, f16) at
    /// M=1 before the GEMV path can claim it.
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn fp16_m1_decode_reaches_neon_gemv_not_half_gemm() {
        use std::sync::atomic::Ordering;

        let (k, n) = (64, 32);
        let a_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        let a = Owned::f16(&[1, k], &a_data);
        let b = Owned::f16(&[k, n], &b_data);
        let mut out = Owned::zeros_f32(&[1, n]);

        let mut kernel = MatMulKernel::default();
        kernel.set_constant_inputs(&[false, true]);

        let before = GEMV_F16_TEST_HITS.load(Ordering::Relaxed);
        kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        let after = GEMV_F16_TEST_HITS.load(Ordering::Relaxed);

        assert!(
            after > before,
            "FP16 M=1 decode did not reach neon_gemv_f16_col_parallel — \
             half_gemm.rs is likely intercepting M=1 before the GEMV path, \
             which causes a ~4× decode throughput regression"
        );
        // Sanity: output should be finite
        assert!(
            out.to_f32().iter().all(|v| v.is_finite()),
            "GEMV produced non-finite output"
        );
    }
}
