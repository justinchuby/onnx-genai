//! `MatMul`: numpy-style matrix multiplication for f32, including batched and
//! broadcast leading dimensions and 1-D vector operands (`docs/ORT2.md` §4.4).
//!
//! ## Perf seam (Phase-1.5)
//!
//! The inner loop here is a **naive** triple-loop GEMM — correct, not fast.
//! This is the single hottest kernel for transformer inference, so it is the
//! prime target for the Phase-1.5 perf pass: replace [`gemm`] with a blocked /
//! SIMD implementation (oneDNN via FFI, or a Rust BLAS such as `matrixmultiply`
//! or `gemm`) behind this same [`Kernel`] impl. Nothing above the [`Kernel`]
//! trait — not the provider, not the session — needs to change.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{broadcast_shapes, compute_contiguous_strides, Node};

use super::{check_arity, to_dense_f32, write_dense_f32};
use crate::strided::{next_index, numel};

/// Stateless f32 MatMul kernel.
pub struct MatMulKernel;

/// Factory for [`MatMulKernel`] (no attributes).
pub struct MatMulFactory;

impl KernelFactory for MatMulFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MatMulKernel))
    }
}

/// Naive row-major GEMM: `c[m,n] = sum_k a[m,k] * b[k,n]` for a single 2-D tile.
/// `a` is `m*k` row-major, `b` is `k*n` row-major, `c` is `m*n` row-major.
fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for p in 0..k {
            let aik = a[i * k + p];
            if aik == 0.0 {
                continue;
            }
            let a_row = &b[p * n..p * n + n];
            let c_row = &mut c[i * n..i * n + n];
            for j in 0..n {
                c_row[j] += aik * a_row[j];
            }
        }
    }
}

impl Kernel for MatMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("MatMul", inputs, outputs, 2, 2, 1)?;
        let a_dense = to_dense_f32(&inputs[0])?;
        let b_dense = to_dense_f32(&inputs[1])?;

        // Promote 1-D operands per numpy matmul: a [K] -> [1,K] (drop row after),
        // b [K] -> [K,1] (drop col after).
        let a_raw = inputs[0].shape;
        let b_raw = inputs[1].shape;
        let a_1d = a_raw.len() == 1;
        let b_1d = b_raw.len() == 1;
        let a_shape: Vec<usize> = if a_1d { vec![1, a_raw[0]] } else { a_raw.to_vec() };
        let b_shape: Vec<usize> = if b_1d { vec![b_raw[0], 1] } else { b_raw.to_vec() };

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

        let mut out = vec![0.0f32; batch_count.max(1) * c_mat];

        if batch_shape.is_empty() {
            // No batch dims: a single matmul.
            gemm(&a_dense, &b_dense, &mut out, m, k, n);
        } else {
            let mut bidx = vec![0usize; batch_shape.len()];
            let mut b_out = 0usize;
            loop {
                let a_off = broadcast_offset(&bidx, a_batch, &a_batch_strides) * a_mat;
                let b_off = broadcast_offset(&bidx, b_batch, &b_batch_strides) * b_mat;
                gemm(
                    &a_dense[a_off..a_off + a_mat],
                    &b_dense[b_off..b_off + b_mat],
                    &mut out[b_out * c_mat..b_out * c_mat + c_mat],
                    m,
                    k,
                    n,
                );
                b_out += 1;
                if !next_index(&batch_shape, &mut bidx) {
                    break;
                }
            }
        }

        // If either operand was 1-D, the corresponding size-1 axis is squeezed
        // out of the result; `write_dense_f32` uses the output view's own shape,
        // so the dense buffer already matches element-for-element.
        write_dense_f32(&mut outputs[0], &out)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn estimated_flops(&self) -> Option<u64> {
        None
    }
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
    fn matmul_2x3_times_3x2() {
        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        // C = [[58,64],[139,154]]
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        MatMulKernel
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
        MatMulKernel
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
        MatMulKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // batch0: A@I = A; batch1: [[5,6],[7,8]]*2 = [[10,12],[14,16]]
        assert_eq!(
            out.to_f32(),
            vec![1., 2., 3., 4., 10., 12., 14., 16.]
        );
    }

    #[test]
    fn matmul_broadcast_batch() {
        // A [2,2,2] @ B [2,2] (broadcast B over batch)
        let a = Owned::f32(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = Owned::f32(&[2, 2], &[1., 0., 0., 1.]); // identity
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        MatMulKernel
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
        MatMulKernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        assert_eq!(out.to_f32(), vec![58., 64.]);
    }
}
