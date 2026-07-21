//! `Gemm` (ONNX standard domain) on the GPU: `Y = alpha·A'·B' + beta·C`, where
//! `A'`/`B'` are `A`/`B` optionally transposed per the `transA`/`transB`
//! attributes and the optional bias `C` is unidirectionally broadcast to
//! `[M, N]` (`docs/ORT2.md` §15.3; RULES.md #4 — GEMM stays on cuBLASLt).
//!
//! ## Design — reuse the proven cuBLASLt path
//!
//! The matmul core is [`crate::blas::gemm_ex`] (the same column-major-aware
//! entry the attention GEMMs use), so the delicate row-major↔column-major
//! mapping lives in exactly one place. This kernel only computes the transpose
//! flags / leading dimensions for the `Gemm` attribute combinations and folds
//! `alpha` into the GEMM. The `beta·C` bias is a fused NVRTC epilogue add
//! (broadcasting `C` over rows and/or columns), avoiding a full-size bias
//! materialisation.
//!
//! ## This slice's limits (all actionable errors, never panics)
//!
//! * **dtype:** f32 only (the bias epilogue is f32; f16/bf16 land with a
//!   templated epilogue next).
//! * **rank:** `A`/`B` must be 2-D; `C` (if present) must broadcast to `[M,N]`
//!   (rank 0/1/2 with each dim `1`, `M`, or `N` as ONNX requires).
//! * **layout:** dense row-major inputs/outputs (a strided view returns a
//!   "materialise first" error).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{GemmDtype, GemmEx, WORKSPACE_BYTES, gemm_ex};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC source for the fused `beta·C` bias epilogue. `C` is addressed through
/// two broadcast strides (`0` for a size-1 dim), so one kernel covers scalar,
/// per-row, per-column, and full `[M,N]` bias without materialising it.
const BIAS_SRC: &str = r#"
extern "C" __global__ void gemm_bias_f32(
    float*       y,             // [m, n] row-major, in/out
    const float* c,             // broadcastable bias
    const int    m,
    const int    n,
    const int    c_row_stride,  // 0 if C broadcasts over rows, else n_c
    const int    c_col_stride,  // 0 if C broadcasts over cols, else 1
    const float  beta)
{
    const long total = (long)m * n;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        const int row = (int)(idx / n);
        const int col = (int)(idx % n);
        const float cv = c[(long)row * c_row_stride + (long)col * c_col_stride];
        y[idx] += beta * cv;
    }
}
"#;

const BIAS_MODULE: &str = "gemm_bias_f32";
const BIAS_ENTRY: &str = "gemm_bias_f32";
const BIAS_BLOCK: u32 = 256;

/// Factory for [`GemmKernel`]; reads `alpha`/`beta` (default 1.0) and
/// `transA`/`transB` (default 0) — all model-agnostic runtime attributes.
pub struct GemmFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for GemmFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let alpha = node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0);
        let beta = node.attr("beta").and_then(|a| a.as_float()).unwrap_or(1.0);
        let trans_a = node.attr("transA").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let trans_b = node.attr("transB").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        Ok(Box::new(GemmKernel {
            runtime: self.runtime.clone(),
            alpha,
            beta,
            trans_a,
            trans_b,
        }))
    }
}

/// cuBLASLt-backed f32 `Gemm` kernel with a fused NVRTC bias epilogue.
#[derive(Debug)]
pub struct GemmKernel {
    runtime: Arc<CudaRuntime>,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

/// Resolved `Gemm` problem geometry and the cuBLASLt leading dims / transpose
/// flags that realise the row-major `Y = alpha·A'·B'` (see [`GemmKernel::run`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GemmPlan {
    pub(super) m: usize,
    pub(super) k: usize,
    pub(super) n: usize,
    /// cuBLASLt `transa` (applied to the ONNX **B** operand — see the mapping).
    pub(super) transa: bool,
    /// cuBLASLt `transb` (applied to the ONNX **A** operand).
    pub(super) transb: bool,
    /// Leading dim of the ONNX B operand as stored (`= B.shape[1]`).
    pub(super) ldb_operand: usize,
    /// Leading dim of the ONNX A operand as stored (`= A.shape[1]`).
    pub(super) lda_operand: usize,
}

/// Compute the GEMM plan from the raw (row-major) operand shapes and the
/// transpose flags. Mirrors the row-major→column-major identity documented in
/// [`crate::blas`]: to get row-major `Y[M,N]` we ask cuBLASLt for the
/// column-major `Yᵀ = alpha·(B')ᵀ·(A')ᵀ`, feeding the B operand as cuBLASLt's
/// first matrix and A as its second.
pub(super) fn plan_gemm(
    a: &[usize],
    b: &[usize],
    trans_a: bool,
    trans_b: bool,
) -> Result<GemmPlan> {
    if a.len() != 2 || b.len() != 2 {
        return Err(not_implemented(format!(
            "Gemm with operand ranks {}D x {}D (Gemm requires 2-D A and B)",
            a.len(),
            b.len()
        )));
    }
    let (ra, ca) = (a[0], a[1]);
    let (rb, cb) = (b[0], b[1]);

    // A' is [M,K]; B' is [K,N]. Resolve from the transpose flags.
    let (m, ka) = if trans_a { (ca, ra) } else { (ra, ca) };
    let (kb, n) = if trans_b { (cb, rb) } else { (rb, cb) };
    if ka != kb {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Gemm: inner dimensions disagree — A' is [{m},{ka}] but B' is [{kb},{n}] \
             (A {a:?} transA={trans_a}, B {b:?} transB={trans_b})"
        )));
    }
    Ok(GemmPlan {
        m,
        k: ka,
        n,
        // cuBLASLt first matrix = ONNX B, transposed iff transB; second = ONNX A,
        // transposed iff transA. Leading dims are the stored column counts.
        transa: trans_b,
        transb: trans_a,
        ldb_operand: cb,
        lda_operand: ca,
    })
}

/// Broadcast strides (`c_row_stride`, `c_col_stride`) for a bias `C` of the
/// given shape onto an `[M,N]` output, or an actionable error if `C` does not
/// unidirectionally broadcast to `[M,N]` (ONNX Gemm requirement).
fn bias_strides(c: &[usize], m: usize, n: usize) -> Result<(i32, i32)> {
    // Normalise C to a 2-D [cr, cc] with trailing-dim alignment.
    let (cr, cc) = match c.len() {
        0 => (1usize, 1usize),
        1 => (1usize, c[0]),
        2 => (c[0], c[1]),
        _ => {
            return Err(not_implemented(format!(
                "Gemm bias C rank {} (bias must broadcast to [M,N]; rank <= 2)",
                c.len()
            )));
        }
    };
    if (cr != 1 && cr != m) || (cc != 1 && cc != n) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep Gemm: bias C {c:?} does not broadcast to [M={m}, N={n}] \
             (each C dim must be 1, M, or N)"
        )));
    }
    let row_stride = if cr == 1 { 0 } else { cc as i32 };
    let col_stride = if cc == 1 { 0 } else { 1 };
    Ok((row_stride, col_stride))
}

impl GemmKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(2..=3).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Gemm: expected 2 inputs (A,B) or 3 (A,B,C) and 1 output, \
                 got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        let a = &inputs[0];
        let b = &inputs[1];
        let bias = inputs.get(2).filter(|c| !c.is_absent());

        for (name, dt) in [("A", a.dtype), ("B", b.dtype), ("Y", outputs[0].dtype)] {
            if dt != DataType::Float32 {
                return Err(not_implemented(format!(
                    "Gemm with {name} dtype {dt:?} (this slice is f32-only; f16/bf16 pending)"
                )));
            }
        }
        for (name, contiguous) in [
            ("A", a.is_contiguous()),
            ("B", b.is_contiguous()),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(not_implemented(format!(
                    "Gemm with a non-contiguous (strided) {name}; materialise it (insert a copy) \
                     before the Gemm"
                )));
            }
        }

        let plan = plan_gemm(a.shape, b.shape, self.trans_a, self.trans_b)?;
        let expected_out = plan.m * plan.n;
        if outputs[0].numel() != expected_out {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Gemm: output has {} elements, expected {} ([M={},N={}])",
                outputs[0].numel(),
                expected_out,
                plan.m,
                plan.n
            )));
        }

        // Resolve the bias plan (strides) before touching the device.
        let bias_plan = match bias {
            None => None,
            Some(c) => {
                if c.dtype != DataType::Float32 {
                    return Err(not_implemented(format!(
                        "Gemm bias C dtype {:?} (this slice is f32-only)",
                        c.dtype
                    )));
                }
                if !c.is_contiguous() {
                    return Err(not_implemented(
                        "Gemm with a non-contiguous (strided) bias C; materialise it first",
                    ));
                }
                let (rs, cs) = bias_strides(c.shape, plan.m, plan.n)?;
                Some((c, rs, cs))
            }
        };

        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        let workspace = self.runtime.alloc_raw(WORKSPACE_BYTES)?;

        // Column-major Yᵀ[N,M] = alpha·(B')ᵀ·(A')ᵀ → row-major Y[M,N].
        let params = GemmEx {
            dtype: GemmDtype::F32,
            transa: plan.transa,
            transb: plan.transb,
            m: plan.n,
            n: plan.m,
            k: plan.k,
            alpha: self.alpha,
            beta: 0.0,
            a: b_ptr,
            lda: plan.ldb_operand,
            b: a_ptr,
            ldb: plan.lda_operand,
            c: y_ptr,
            ldc: plan.n,
            epilogue: None,
        };

        // SAFETY: A/B/Y are validated dense f32 device buffers sized for the
        // planned shapes; `workspace` is live for the call; Y aliases neither A
        // nor B (distinct output buffer); the context is bound by `alloc_raw`.
        let gemm_res = unsafe {
            gemm_ex(
                self.runtime.blas(),
                self.runtime.stream_ptr(),
                &params,
                workspace,
                WORKSPACE_BYTES,
            )
        };

        // Fused bias epilogue: Y += beta·C (broadcast). Skipped when beta==0.
        let bias_res = gemm_res.and_then(|()| {
            if let Some((c, rs, cs)) = bias_plan {
                if self.beta != 0.0 {
                    self.apply_bias(y_ptr, c, plan.m, plan.n, rs, cs)
                } else {
                    Ok(())
                }
            } else {
                Ok(())
            }
        });

        let synced = bias_res.and_then(|()| self.runtime.synchronize());

        // Always release the workspace, even on failure.
        // SAFETY: `workspace` came from the `alloc_raw` above and is freed once.
        let free = unsafe { self.runtime.free_raw(workspace) };
        synced.and(free)
    }

    /// Launch the fused `Y += beta·C` broadcast bias epilogue.
    fn apply_bias(
        &self,
        y_ptr: cudarc::driver::sys::CUdeviceptr,
        c: &TensorView,
        m: usize,
        n: usize,
        row_stride: i32,
        col_stride: i32,
    ) -> Result<()> {
        let c_ptr = cuptr(c.data_ptr::<u8>() as *const c_void);
        let total = m * n;
        let (m_i, n_i) = (
            i32::try_from(m)
                .map_err(|_| EpError::KernelFailed(format!("cuda_ep Gemm: M={m} exceeds i32")))?,
            i32::try_from(n)
                .map_err(|_| EpError::KernelFailed(format!("cuda_ep Gemm: N={n} exceeds i32")))?,
        );
        let beta = self.beta;
        let func = self
            .runtime
            .nvrtc_function(BIAS_MODULE, BIAS_SRC, BIAS_ENTRY)?;
        let blocks = total.div_ceil(BIAS_BLOCK as usize).clamp(1, 65_535) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BIAS_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&y_ptr)
            .arg(&c_ptr)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&row_stride)
            .arg(&col_stride)
            .arg(&beta);
        // SAFETY: `func` is the compiled bias entry; the argument list matches
        // its (float*, const float*, int, int, int, int, float) signature; `y`
        // covers [m,n] f32 and `c` is addressable at every broadcast index the
        // strides produce (validated by `bias_strides`).
        unsafe { builder.launch(cfg) }
            .map(|_| ())
            .map_err(|e| driver_err("launch gemm_bias_f32", e))
    }
}

impl Kernel for GemmKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "per-call workspace allocation/free is not capturable",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_no_transpose() {
        let p = plan_gemm(&[2, 3], &[3, 4], false, false).unwrap();
        assert_eq!((p.m, p.k, p.n), (2, 3, 4));
        assert!(!p.transa && !p.transb);
        // cuBLASLt first matrix is B (ld = B cols), second is A (ld = A cols).
        assert_eq!(p.ldb_operand, 4);
        assert_eq!(p.lda_operand, 3);
    }

    #[test]
    fn plan_trans_a() {
        // A stored [K,M] = [3,2], transA -> A' = [M,K] = [2,3].
        let p = plan_gemm(&[3, 2], &[3, 4], true, false).unwrap();
        assert_eq!((p.m, p.k, p.n), (2, 3, 4));
        assert!(p.transb, "transA maps to cuBLASLt transb");
        assert!(!p.transa);
    }

    #[test]
    fn plan_trans_b() {
        // B stored [N,K] = [4,3], transB -> B' = [K,N] = [3,4].
        let p = plan_gemm(&[2, 3], &[4, 3], false, true).unwrap();
        assert_eq!((p.m, p.k, p.n), (2, 3, 4));
        assert!(p.transa, "transB maps to cuBLASLt transa");
        assert!(!p.transb);
    }

    #[test]
    fn plan_inner_mismatch_is_plain_error() {
        let e = plan_gemm(&[2, 3], &[5, 4], false, false).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("inner dimensions disagree"), "{msg}");
        assert!(!msg.contains("not implemented"), "{msg}");
    }

    #[test]
    fn plan_rejects_non_2d() {
        let e = plan_gemm(&[2, 3, 4], &[4, 5], false, false).unwrap_err();
        assert!(format!("{e}").contains("requires 2-D"), "{e}");
    }

    #[test]
    fn bias_strides_scalar_and_vectors() {
        // scalar -> broadcast over both axes.
        assert_eq!(bias_strides(&[], 2, 4).unwrap(), (0, 0));
        // 1-D [N] -> per-column, broadcast over rows.
        assert_eq!(bias_strides(&[4], 2, 4).unwrap(), (0, 1));
        // 1-D [1] -> scalar.
        assert_eq!(bias_strides(&[1], 2, 4).unwrap(), (0, 0));
        // full [M,N].
        assert_eq!(bias_strides(&[2, 4], 2, 4).unwrap(), (4, 1));
        // [M,1] -> per-row (stride = C cols = 1), broadcast over columns.
        assert_eq!(bias_strides(&[2, 1], 2, 4).unwrap(), (1, 0));
        // [1,N] -> per-column.
        assert_eq!(bias_strides(&[1, 4], 2, 4).unwrap(), (0, 1));
    }

    #[test]
    fn bias_strides_rejects_non_broadcastable() {
        let e = bias_strides(&[3, 4], 2, 4).unwrap_err();
        assert!(format!("{e}").contains("does not broadcast"), "{e}");
    }
}
