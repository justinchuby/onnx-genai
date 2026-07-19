//! `MatMul`: numpy-style matrix multiplication for f32, including batched and
//! broadcast leading dimensions and 1-D vector operands (`docs/ORT2.md` §4.4).
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
use crate::backend::CpuBackend;
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::{next_index, numel};

// MLAS-style packed SIMD f32 GEMM (the `SimdX86` backend). Kept in a sibling
// file but included here so `kernels/mod.rs` needs no edit; it is an internal
// perf detail of the MatMul hot path, not a new op.
#[path = "simd_gemm.rs"]
mod simd_gemm;

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
}

impl MatMulPrepack {
    pub(crate) fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in self.constant_inputs.iter_mut().enumerate() {
            *is_constant = constant_inputs.get(index).copied().unwrap_or(false);
        }
    }

    fn dense<'a>(&'a self, index: usize, view: &'a TensorView<'_>) -> Result<Cow<'a, [f32]>> {
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
        // Xnnpack / Accelerate placeholders (and Generic) share the pure-Rust
        // kernel until their native backends are wired.
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
    c.par_chunks_mut(mc * n)
        .enumerate()
        .for_each(|(blk, c_block)| {
            let i0 = blk * mc;
            let rows = c_block.len() / n; // last block may be short
            let a_block = &a[i0 * k..i0 * k + rows * k];
            gemm_block(a_block, b, c_block, rows, k, n);
        });
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
        let out =
            matmul_dense_prepacked_with_backend(&inputs[0], &inputs[1], &self.prepack, backend)?;
        // If either operand was 1-D, the corresponding size-1 axis is squeezed
        // out of the result; the narrowing writer uses the output view's own
        // shape and dtype (f32/f16/bf16/f64), so the buffer matches element for
        // element and rounds to the requested precision.
        write_dense_f32_narrow("MatMul", &mut outputs[0], &out)
    }
}

/// Compute `A @ B` (numpy semantics: batched, broadcast leading dims, 1-D
/// operand promotion) into a dense row-major `Vec<f32>`.
///
/// Operands may be any float dtype (`f32`/`f16`/`bf16`/`f64`); low/medium
/// precision inputs are widened to `f32` and the GEMM accumulates in `f32`
/// (standard mixed-precision matmul). Shared by [`MatMulKernel`] and the fused
/// `FusedMatMulBias` kernel so both go through exactly one GEMM implementation.
pub(crate) fn matmul_dense(a: &TensorView, b: &TensorView) -> Result<Vec<f32>> {
    matmul_dense_impl_with_backend(
        a,
        b,
        to_dense_f32_widen("MatMul", a)?,
        to_dense_f32_widen("MatMul", b)?,
        CpuBackend::auto_detect(),
        #[cfg(feature = "mlas")]
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

fn matmul_dense_prepacked_with_backend(
    a: &TensorView,
    b: &TensorView,
    prepack: &MatMulPrepack,
    backend: CpuBackend,
) -> Result<Vec<f32>> {
    matmul_dense_impl_with_backend(
        a,
        b,
        prepack.dense(0, a)?,
        prepack.dense(1, b)?,
        backend,
        #[cfg(feature = "mlas")]
        Some(prepack),
    )
}

fn matmul_dense_impl_with_backend(
    a: &TensorView,
    b: &TensorView,
    a_dense: Cow<'_, [f32]>,
    b_dense: Cow<'_, [f32]>,
    backend: CpuBackend,
    #[cfg(feature = "mlas")] prepack: Option<&MatMulPrepack>,
) -> Result<Vec<f32>> {
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
    let a_batch = &a_shape[..a_shape.len() - 2];
    let b_batch = &b_shape[..b_shape.len() - 2];
    let batch_shape = broadcast_shapes(a_batch, b_batch)?;
    let batch_count = numel(&batch_shape);

    let a_batch_strides = compute_contiguous_strides(a_batch);
    let b_batch_strides = compute_contiguous_strides(b_batch);
    let a_mat = m * k;
    let b_mat = k * n;
    let c_mat = m * n;

    let mut out = vec![0.0f32; batch_count * c_mat];

    // Any zero dimension (batch, M, or N) yields an empty result — matching
    // numpy/ONNX reference semantics. Return before the compute loop, which
    // otherwise runs once even for a zero-sized batch (a `loop { … } while`) and
    // would index into empty operand slices.
    if out.is_empty() {
        return Ok(out);
    }

    #[cfg(feature = "mlas")]
    let packed_b = if backend == CpuBackend::Mlas && b_shape.len() == 2 {
        prepack.and_then(|prepack| prepack.packed_b(&b_dense, k, n))
    } else {
        None
    };

    if batch_shape.is_empty() {
        // No batch dims: a single matmul.
        #[cfg(feature = "mlas")]
        if let Some(packed_b) = packed_b {
            gemm_packed(&a_dense, packed_b, &mut out, m, k, n)?;
        } else {
            gemm_with_backend(backend, &a_dense, &b_dense, &mut out, m, k, n)?;
        }
        #[cfg(not(feature = "mlas"))]
        gemm_with_backend(backend, &a_dense, &b_dense, &mut out, m, k, n)?;
    } else {
        let mut bidx = vec![0usize; batch_shape.len()];
        let mut b_out = 0usize;
        loop {
            let a_off = broadcast_offset(&bidx, a_batch, &a_batch_strides) * a_mat;
            let b_off = broadcast_offset(&bidx, b_batch, &b_batch_strides) * b_mat;
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
            if !next_index(&batch_shape, &mut bidx) {
                break;
            }
        }
    }

    Ok(out)
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
        assert!(kernel.prepack.dense[1].get().is_some());
        assert!(kernel.prepack.dense[0].get().is_none());

        let cached_weight = kernel.prepack.dense[1].get().unwrap().as_ptr();
        let a2 = Owned::f32(&[1, 2], &[4., 5.]);
        let mut out2 = Owned::zeros_f32(&[1, 2]);
        kernel
            .execute(&[a2.view(), weight.view()], &mut [out2.view_mut()])
            .unwrap();
        assert_eq!(out2.to_f32(), vec![8., 15.]);
        assert_eq!(
            kernel.prepack.dense[1].get().unwrap().as_ptr(),
            cached_weight
        );
    }
}
