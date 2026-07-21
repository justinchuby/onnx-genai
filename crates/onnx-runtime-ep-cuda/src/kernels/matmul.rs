//! `MatMul` on the GPU via cuBLASLt (`docs/ORT2.md` §15.3).
//!
//! Supports dense rank >= 2 operands with NumPy/ONNX broadcasting across all
//! leading batch dimensions for f32 / f16 / bf16, all in true fp32
//! accumulation. Broadcast runs are expressed as cuBLASLt strided batches,
//! including stride-zero operands. The row-major → column-major mapping lives
//! in [`crate::blas`].
//!
//! ## Limits (all reported as actionable errors, never panics)
//!
//! * rank-1 operand promotion is not implemented yet
//! * non-contiguous (strided) device inputs are not implemented yet
//! * dtypes other than f32 / f16 / bf16 are not implemented yet
//! * mismatched inner dims / dtypes → a plain kernel error (a real mistake, not
//!   a missing feature)

use std::sync::Arc;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{self, GemmDtype, GemmParams, WORKSPACE_BYTES};
use crate::error::not_implemented;
use crate::runtime::{CudaRuntime, cuptr};

/// Factory for [`MatMulKernel`]; carries the shared CUDA runtime.
pub struct MatMulFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for MatMulFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MatMulKernel {
            runtime: self.runtime.clone(),
        }))
    }
}

/// cuBLASLt-backed f32/f16/bf16 MatMul kernel.
pub struct MatMulKernel {
    runtime: Arc<CudaRuntime>,
}

/// Map an ONNX element type to a cuBLASLt GEMM dtype.
fn gemm_dtype(dt: DataType) -> Result<GemmDtype> {
    match dt {
        DataType::Float32 => Ok(GemmDtype::F32),
        DataType::Float16 => Ok(GemmDtype::F16),
        DataType::BFloat16 => Ok(GemmDtype::Bf16),
        other => Err(not_implemented(format!("MatMul with dtype {other:?}"))),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct MatMulPlan {
    batch_shape: Vec<usize>,
    a_batch_strides: Vec<usize>,
    b_batch_strides: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct BatchRun {
    a_matrix: usize,
    b_matrix: usize,
    c_matrix: usize,
    batch: usize,
    a_stride: usize,
    b_stride: usize,
}

fn broadcast_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    let mut stride = 1;
    for i in (0..dims.len()).rev() {
        strides[i] = if dims[i] == 1 { 0 } else { stride };
        stride *= dims[i];
    }
    strides
}

fn matmul_plan(a: &[usize], b: &[usize]) -> Result<MatMulPlan> {
    if a.len() < 2 || b.len() < 2 {
        return Err(not_implemented(format!(
            "MatMul with operand ranks {}D x {}D (rank-1 promotion is not supported yet)",
            a.len(),
            b.len()
        )));
    }
    let (m, k, n) = (a[a.len() - 2], a[a.len() - 1], b[b.len() - 1]);
    if b[b.len() - 2] != k {
        return Err(inner_mismatch(a, b));
    }

    let batch_rank = (a.len() - 2).max(b.len() - 2);
    let mut a_batch_dims = vec![1; batch_rank];
    let mut b_batch_dims = vec![1; batch_rank];
    a_batch_dims[batch_rank - (a.len() - 2)..].copy_from_slice(&a[..a.len() - 2]);
    b_batch_dims[batch_rank - (b.len() - 2)..].copy_from_slice(&b[..b.len() - 2]);

    let mut batch_shape = Vec::with_capacity(batch_rank);
    for (&ad, &bd) in a_batch_dims.iter().zip(&b_batch_dims) {
        if ad != bd && ad != 1 && bd != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: batch dimensions do not broadcast between A {a:?} and B {b:?}"
            )));
        }
        batch_shape.push(ad.max(bd));
    }

    Ok(MatMulPlan {
        a_batch_strides: broadcast_strides(&a_batch_dims),
        b_batch_strides: broadcast_strides(&b_batch_dims),
        batch_shape,
        m,
        k,
        n,
    })
}

impl MatMulPlan {
    fn output_shape(&self) -> Vec<usize> {
        let mut shape = self.batch_shape.clone();
        shape.extend([self.m, self.n]);
        shape
    }

    fn batch_runs(&self) -> Vec<BatchRun> {
        if self.batch_shape.is_empty() {
            return vec![BatchRun {
                a_matrix: 0,
                b_matrix: 0,
                c_matrix: 0,
                batch: 1,
                a_stride: 0,
                b_stride: 0,
            }];
        }

        let inner = *self.batch_shape.last().unwrap();
        let outer: usize = self.batch_shape[..self.batch_shape.len() - 1]
            .iter()
            .product();
        let mut runs = Vec::with_capacity(outer);
        for outer_index in 0..outer {
            let mut remaining = outer_index;
            let mut a_matrix = 0;
            let mut b_matrix = 0;
            for axis in (0..self.batch_shape.len() - 1).rev() {
                let coord = remaining % self.batch_shape[axis];
                remaining /= self.batch_shape[axis];
                a_matrix += coord * self.a_batch_strides[axis];
                b_matrix += coord * self.b_batch_strides[axis];
            }
            let last = self.batch_shape.len() - 1;
            runs.push(BatchRun {
                a_matrix,
                b_matrix,
                c_matrix: outer_index * inner,
                batch: inner,
                a_stride: self.a_batch_strides[last],
                b_stride: self.b_batch_strides[last],
            });
        }
        runs
    }
}

fn inner_mismatch(a: &[usize], b: &[usize]) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep MatMul: inner dimensions disagree between A {a:?} and B {b:?}"
    ))
}

impl MatMulKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let a = &inputs[0];
        let b = &inputs[1];

        // All operands must share one supported element type.
        let dtype = gemm_dtype(a.dtype)?;
        if b.dtype != a.dtype || outputs[0].dtype != a.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: mixed dtypes A={:?} B={:?} C={:?} (all must match)",
                a.dtype, b.dtype, outputs[0].dtype
            )));
        }

        // Dense, row-major device buffers are required. Strided views (e.g. a
        // transposed input) must be materialised by the graph.
        if !a.is_contiguous() || !b.is_contiguous() {
            return Err(not_implemented(
                "MatMul with a non-contiguous (strided) input; \
                 insert an explicit copy/transpose before the MatMul",
            ));
        }
        if !outputs[0].is_contiguous() {
            return Err(not_implemented("MatMul with a non-contiguous output"));
        }

        let plan = matmul_plan(a.shape, b.shape)?;

        let expected_shape = plan.output_shape();
        if outputs[0].shape != expected_shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep MatMul: output shape {:?}, expected {expected_shape:?}",
                outputs[0].shape
            )));
        }

        // Device pointers (byte_offset applied). These are opaque CUDA
        // addresses, never dereferenced on the host.
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const std::ffi::c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const std::ffi::c_void);
        let c_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const std::ffi::c_void);

        let workspace = self.runtime.alloc_raw(WORKSPACE_BYTES)?;
        let elem_bytes = a.dtype.byte_size();
        let a_matrix_bytes = plan.m * plan.k * elem_bytes;
        let b_matrix_bytes = plan.k * plan.n * elem_bytes;
        let c_matrix_bytes = plan.m * plan.n * elem_bytes;

        let result = plan
            .batch_runs()
            .into_iter()
            .try_for_each(|run| {
                let params = GemmParams {
                    dtype,
                    a: a_ptr + (run.a_matrix * a_matrix_bytes) as u64,
                    b: b_ptr + (run.b_matrix * b_matrix_bytes) as u64,
                    c: c_ptr + (run.c_matrix * c_matrix_bytes) as u64,
                    m: plan.m,
                    k: plan.k,
                    n: plan.n,
                    batch: run.batch,
                    a_batch_stride: run.a_stride * plan.m * plan.k,
                    b_batch_stride: run.b_stride * plan.k * plan.n,
                    epilogue: None,
                };
                // SAFETY: the plan's broadcast offsets address complete matrices
                // inside A/B/Y; workspace and stream remain live for every run.
                unsafe {
                    blas::gemm(
                        self.runtime.blas(),
                        self.runtime.stream_ptr(),
                        &params,
                        workspace,
                        WORKSPACE_BYTES,
                    )
                }
            })
            .and_then(|()| self.runtime.synchronize());

        // Always release the workspace, even on failure.
        // SAFETY: `workspace` came from the `alloc_raw` above and is freed once.
        let free = unsafe { self.runtime.free_raw(workspace) };
        result.and(free)
    }
}

impl Kernel for MatMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        // Dense inputs only (see `run`).
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "per-call workspace allocation/free and heuristic query are not capturable",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_2d_ok() {
        let p = matmul_plan(&[2, 3], &[3, 4]).unwrap();
        assert_eq!((p.m, p.k, p.n), (2, 3, 4));
        assert_eq!(p.output_shape(), [2, 4]);
        assert_eq!(p.batch_runs()[0].batch, 1);
    }

    #[test]
    fn plan_3d_equal_batch_ok() {
        let p = matmul_plan(&[5, 2, 3], &[5, 3, 4]).unwrap();
        assert_eq!(p.output_shape(), [5, 2, 4]);
        assert_eq!(p.batch_runs()[0].batch, 5);
    }

    #[test]
    fn plan_inner_mismatch_is_plain_error() {
        let e = matmul_plan(&[2, 3], &[4, 5]).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("inner dimensions disagree"), "{msg}");
        // A genuine mistake, not a deferred feature.
        assert!(!msg.contains("not yet implemented"), "{msg}");
    }

    #[test]
    fn plan_broadcast_batch() {
        let p = matmul_plan(&[3, 1, 2, 4], &[1, 5, 4, 6]).unwrap();
        assert_eq!(p.output_shape(), [3, 5, 2, 6]);
        assert_eq!(
            p.batch_runs(),
            [
                BatchRun {
                    a_matrix: 0,
                    b_matrix: 0,
                    c_matrix: 0,
                    batch: 5,
                    a_stride: 0,
                    b_stride: 1
                },
                BatchRun {
                    a_matrix: 1,
                    b_matrix: 0,
                    c_matrix: 5,
                    batch: 5,
                    a_stride: 0,
                    b_stride: 1
                },
                BatchRun {
                    a_matrix: 2,
                    b_matrix: 0,
                    c_matrix: 10,
                    batch: 5,
                    a_stride: 0,
                    b_stride: 1
                },
            ]
        );
    }

    #[test]
    fn plan_high_rank_equal_batch() {
        let p = matmul_plan(&[2, 3, 4, 5], &[2, 3, 5, 6]).unwrap();
        assert_eq!(p.output_shape(), [2, 3, 4, 6]);
        assert_eq!(p.batch_runs().len(), 2);
        assert!(p.batch_runs().iter().all(|run| run.batch == 3));
    }

    #[test]
    fn plan_2d_broadcast_across_4d() {
        let p = matmul_plan(&[4, 5], &[2, 3, 5, 6]).unwrap();
        assert_eq!(p.output_shape(), [2, 3, 4, 6]);
        assert!(p.batch_runs().iter().all(|run| run.a_stride == 0));
    }

    #[test]
    fn plan_rejects_rank_1_with_clear_error() {
        let e = matmul_plan(&[5], &[5, 6]).unwrap_err();
        assert!(format!("{e}").contains("rank-1 promotion"), "{e}");
    }

    #[test]
    fn dtype_mapping_and_unsupported() {
        assert_eq!(gemm_dtype(DataType::Float32).unwrap(), GemmDtype::F32);
        assert_eq!(gemm_dtype(DataType::Float16).unwrap(), GemmDtype::F16);
        assert_eq!(gemm_dtype(DataType::BFloat16).unwrap(), GemmDtype::Bf16);
        let e = gemm_dtype(DataType::Int64).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("dtype Int64"), "{msg}");
        assert!(msg.contains("not yet implemented"), "{msg}");
    }
}
