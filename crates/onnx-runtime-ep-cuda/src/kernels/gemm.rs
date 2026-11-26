//! `Gemm` (ONNX standard domain) on the GPU: `Y = alpha·A'·B' + beta·C`, where
//! `A'`/`B'` are `A`/`B` optionally transposed per the `transA`/`transB`
//! attributes and the optional bias `C` is unidirectionally broadcast to
//! the output (`docs/architecture/ORT2.md` §15.3; GEMM stays on cuBLASLt).
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
//! * **rank:** B is always rank 2. Gemm-29 also accepts vector and N-D A.
//! * **layout:** dense row-major inputs/outputs (a strided view returns a
//!   "materialise first" error).

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::gemm::{GemmGeometry, ND_GEMM_OPSET, validate_bias};
use onnx_runtime_ir::{DataType, Node};

use super::elementwise::BroadcastMetadataCache;
use crate::blas::{self, GemmDtype, GemmEx, StridedBatchedGemmParams};
use crate::device_metadata::publish_metadata_candidate;
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC source for the fused `beta·C` bias epilogue. `C` is addressed through
/// two broadcast strides (`0` for a size-1 dim), so one kernel covers scalar,
/// per-row, per-column, and full `[M,N]` bias without materialising it.
const BIAS_SRC: &str = r#"
extern "C" __global__ void gemm_bias_f32(
    float*       y,             // [m, n] row-major, in/out
    const float* c,             // broadcastable bias
    const unsigned long long m,
    const unsigned long long n,
    const unsigned long long c_row_stride,  // 0 if C broadcasts over rows, else n_c
    const unsigned long long c_col_stride,  // 0 if C broadcasts over cols, else 1
    const float  beta)
{
    const unsigned long long total = m * n;
    for (unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long offset;
        if (c_col_stride == 1 && c_row_stride == n) {
            offset = idx;
        } else if (total <= 0xffffffffULL) {
            // Keep the normal-sized broadcast epilogue off 64-bit division.
            const auto i = (unsigned int)idx;
            const auto width = (unsigned int)n;
            offset = (c_row_stride ? (i / width) * (unsigned int)c_row_stride : 0)
                   + (c_col_stride ? i % width : 0);
        } else {
            offset = (c_row_stride ? (idx / n) * c_row_stride : 0)
                   + (c_col_stride ? idx % n : 0);
        }
        const float cv = c[offset];
        y[idx] += beta * cv;
    }
}

extern "C" __global__ void gemm_bias_nd_f32(
    float* y, const float* c, const unsigned long long* metadata,
    int rank, unsigned long long total, float beta, float empty_product, int add_product)
{
    const auto* strides = metadata + rank;
    for (unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (unsigned long long)gridDim.x * blockDim.x) {
        auto remaining = idx;
        unsigned long long offset = 0;
        for (int d = rank - 1; d >= 0; --d) {
            if (strides[d]) offset += (remaining % metadata[d]) * strides[d];
            remaining /= metadata[d];
        }
        float v = add_product ? y[idx] : empty_product;
        if (c) v += beta * c[offset];
        y[idx] = v;
    }
}
"#;

const BIAS_MODULE: &str = "gemm_bias_f32";
const BIAS_ENTRY: &str = "gemm_bias_f32";
const BIAS_BLOCK: u32 = 256;
pub(crate) const DTYPES: &[DataType] = &[DataType::Float32];

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
            nd: node
                .local_opset()
                .is_some_and(|version| version >= ND_GEMM_OPSET),
            bias_metadata: Mutex::new(BroadcastMetadataCache::new(self.runtime.clone())),
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
    nd: bool,
    bias_metadata: Mutex<BroadcastMetadataCache>,
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
    let geometry =
        GemmGeometry::new(a, b, trans_a, trans_b, false).map_err(EpError::KernelFailed)?;
    Ok(GemmPlan {
        m: geometry.m,
        k: geometry.k,
        n: geometry.n,
        // cuBLASLt first matrix = ONNX B, transposed iff transB; second = ONNX A,
        // transposed iff transA. Leading dims are the stored column counts.
        transa: trans_b,
        transb: trans_a,
        ldb_operand: b[1],
        lda_operand: a[1],
    })
}

/// Broadcast strides (`c_row_stride`, `c_col_stride`) for a bias `C` of the
/// given shape onto an `[M,N]` output, or an actionable error if `C` does not
/// unidirectionally broadcast to `[M,N]` (ONNX Gemm requirement).
fn bias_strides(c: &[usize], m: usize, n: usize) -> Result<(usize, usize)> {
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
    let row_stride = if cr == 1 { 0 } else { cc };
    let col_stride = if cc == 1 { 0 } else { 1 };
    Ok((row_stride, col_stride))
}

impl GemmKernel {
    fn geometry(&self, a: &[usize], b: &[usize]) -> Result<GemmGeometry> {
        GemmGeometry::new(a, b, self.trans_a, self.trans_b, self.nd).map_err(EpError::KernelFailed)
    }

    fn single_params(&self, g: GemmGeometry, a_rank: usize, a: u64, b: u64, y: u64) -> GemmEx {
        let trans_a = Self::transpose_a(g, a_rank);
        GemmEx {
            dtype: GemmDtype::F32,
            transa: self.trans_b,
            transb: trans_a,
            m: g.n,
            n: g.rows,
            k: g.k,
            alpha: self.alpha,
            beta: 0.0,
            a: b,
            lda: if self.trans_b { g.k } else { g.n },
            b: a,
            ldb: if trans_a { g.m } else { g.k },
            c: y,
            ldc: g.n,
            epilogue: None,
        }
    }

    fn transpose_a(g: GemmGeometry, rank: usize) -> bool {
        if rank == 2 {
            g.axes.trans_a
        } else {
            g.transposes_a_storage()
        }
    }

    fn batched_params(&self, g: GemmGeometry, a: u64, b: u64, y: u64) -> StridedBatchedGemmParams {
        StridedBatchedGemmParams {
            dtype: GemmDtype::F32,
            a,
            b,
            c: y,
            m: g.m,
            k: g.k,
            n: g.n,
            batch: g.batch,
            transpose_a: true,
            transpose_b: self.trans_b,
            a_batch_stride: g.m * g.k,
            b_batch_stride: 0,
        }
    }

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
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

        onnx_runtime_ep_api::gemm::validate_dtypes(
            DTYPES,
            self.nd,
            [a.dtype, b.dtype, outputs[0].dtype]
                .into_iter()
                .chain(bias.map(|c| c.dtype)),
        )
        .map_err(EpError::KernelFailed)?;
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

        let g = self.geometry(a.shape, b.shape)?;
        let y_shape = outputs[0].shape;
        g.validate_output(a.shape, b.shape, y_shape)
            .map_err(EpError::KernelFailed)?;

        // Resolve the bias plan (strides) before touching the device.
        let bias_plan = match bias {
            None => None,
            Some(c) => {
                if !c.is_contiguous() {
                    return Err(not_implemented(
                        "Gemm with a non-contiguous (strided) bias C; materialise it first",
                    ));
                }
                validate_bias(c.shape, y_shape).map_err(EpError::KernelFailed)?;
                Some(c)
            }
        };
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let mut flops = (g.rows as u64)
                .saturating_mul(g.n as u64)
                .saturating_mul(g.k as u64)
                .saturating_mul(2);
            if bias_plan.is_some() && self.beta != 0.0 {
                flops = flops
                    .saturating_add((g.rows as u64).saturating_mul(g.n as u64).saturating_mul(2));
            }
            flops
        });

        if g.elements == 0 {
            return Ok(());
        }
        let bias = bias_plan.filter(|_| self.beta != 0.0);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        if g.k == 0 {
            self.apply_bias_nd(y_ptr, bias, y_shape, g.elements, false)?;
            return self.runtime.synchronize();
        }
        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);

        // SAFETY: The executor provides distinct dense device buffers covering
        // the validated shapes and the live workspace on this stream.
        unsafe {
            if g.batch > 1 && g.transposes_a_storage() {
                let params = self.batched_params(g, a_ptr, b_ptr, y_ptr);
                blas::governed_strided_batched_gemm(
                    self.runtime.blas(),
                    self.runtime.stream_ptr(),
                    &params,
                    self.alpha,
                    workspace,
                    "Gemm",
                )?;
            } else {
                let params = self.single_params(g, a.shape.len(), a_ptr, b_ptr, y_ptr);
                blas::governed_gemm_ex(
                    self.runtime.blas(),
                    self.runtime.stream_ptr(),
                    &params,
                    workspace,
                    "Gemm",
                )?;
            }
        }

        if let Some(c) = bias {
            if y_shape.len() == 2 {
                let (rs, cs) = bias_strides(c.shape, g.rows, g.n)?;
                self.apply_bias(y_ptr, c, g.rows, g.n, rs, cs)?;
            } else if c.numel() == 1 {
                self.apply_bias(y_ptr, c, g.rows, g.n, 0, 0)?;
            } else if c.numel() == g.elements {
                self.apply_bias(y_ptr, c, g.rows, g.n, g.n, 1)?;
            } else if c.numel() == g.n && c.shape.last() == Some(&g.n) {
                self.apply_bias(y_ptr, c, g.rows, g.n, 0, 1)?;
            } else {
                self.apply_bias_nd(y_ptr, Some(c), y_shape, g.elements, true)?;
            }
        }
        self.runtime.synchronize()
    }

    fn workspace_requirement_for(
        &self,
        inputs: &[TensorMetadata<'_>],
    ) -> Result<WorkspaceRequirement> {
        if !(2..=3).contains(&inputs.len()) {
            return Ok(WorkspaceRequirement::NONE);
        }
        let a = &inputs[0];
        let b = &inputs[1];
        if a.dtype != DataType::Float32 || b.dtype != DataType::Float32 {
            return Ok(WorkspaceRequirement::NONE);
        }
        let g = self.geometry(a.shape, b.shape)?;
        if g.elements == 0 || g.k == 0 {
            return Ok(WorkspaceRequirement::NONE);
        }
        let bytes = if g.batch > 1 && g.transposes_a_storage() {
            let params = self.batched_params(g, 1, 1, 1);
            blas::strided_batched_gemm_workspace_bytes(self.runtime.blas(), &params, "Gemm")?
        } else {
            let params = self.single_params(g, a.shape.len(), 1, 1, 1);
            blas::gemm_ex_workspace_bytes(self.runtime.blas(), &params)?
        };
        Ok(blas::governed_workspace_requirement(bytes))
    }

    /// Launch the fused `Y += beta·C` broadcast bias epilogue.
    fn apply_bias(
        &self,
        y_ptr: cudarc::driver::sys::CUdeviceptr,
        c: &TensorView,
        m: usize,
        n: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<()> {
        let c_ptr = cuptr(c.data_ptr::<u8>() as *const c_void);
        let total = m * n;
        let (m, n, row_stride, col_stride) =
            (m as u64, n as u64, row_stride as u64, col_stride as u64);
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
            .arg(&m)
            .arg(&n)
            .arg(&row_stride)
            .arg(&col_stride)
            .arg(&beta);
        // SAFETY: `func` is the compiled bias entry; the argument list matches
        // its signature; `y`
        // covers [m,n] f32 and `c` is addressable at every broadcast index the
        // strides produce (validated by `bias_strides`).
        unsafe { builder.launch(cfg) }
            .map(|_| ())
            .map_err(|e| driver_err("launch gemm_bias_f32", e))
    }

    fn apply_bias_nd(
        &self,
        y: u64,
        bias: Option<&TensorView>,
        shape: &[usize],
        elements: usize,
        add_product: bool,
    ) -> Result<()> {
        let func = self
            .runtime
            .nvrtc_function(BIAS_MODULE, BIAS_SRC, "gemm_bias_nd_f32")?;
        let (c, metadata, rank, candidate) = if let Some(c) = bias {
            let rank = i32::try_from(shape.len()).map_err(|_| {
                EpError::KernelFailed(format!(
                    "cuda_ep Gemm: output rank {} exceeds i32; reduce tensor rank",
                    shape.len()
                ))
            })?;
            let mut candidate = self
                .bias_metadata
                .lock()
                .map_err(|_| {
                    EpError::KernelFailed(
                    "cuda_ep Gemm: bias metadata state is poisoned; recreate the execution session"
                        .into(),
                )
                })?
                .clone();
            let metadata = candidate.prepare(c.shape, &[], shape)?;
            (
                cuptr(c.data_ptr::<u8>().cast()),
                metadata,
                rank,
                Some(candidate),
            )
        } else {
            // Empty products without an active bias do not read descriptors.
            // In particular, beta=0 must not depend on cached bias state.
            (0, 0, 0, None)
        };
        let total = elements as u64;
        let empty_product = 0.0 * self.alpha;
        let add_product = i32::from(add_product);
        let cfg = LaunchConfig {
            grid_dim: (
                elements.div_ceil(BIAS_BLOCK as usize).clamp(1, 65_535) as u32,
                1,
                1,
            ),
            block_dim: (BIAS_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&y)
            .arg(&c)
            .arg(&metadata)
            .arg(&rank)
            .arg(&total)
            .arg(&self.beta)
            .arg(&empty_product)
            .arg(&add_product);
        // SAFETY: All pointers cover their validated shapes. The private
        // candidate retains immutable metadata through asynchronous submission;
        // native retirement preserves pending readers if submission fails.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err("launch gemm_bias_nd_f32", e))?;
        if let Some(candidate) = candidate {
            publish_metadata_candidate(&self.bias_metadata, candidate, "Gemm bias")?;
        }
        Ok(())
    }
}

impl Kernel for GemmKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        self.workspace_requirement_for(inputs)
            .map(onnx_runtime_ep_api::WorkspacePreparation::Ready)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Gemm has no warmed capture contract for cuBLASLt and bias metadata; execute eagerly",
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
        assert!(format!("{e}").contains("rank 2 before opset 29"), "{e}");
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

    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable gpu-tests"
    )]
    fn nd_bias_metadata_rewarm_is_transactional_and_inactive_bias_bypasses_cache() {
        use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider};
        use onnx_runtime_ir::compute_contiguous_strides;

        let ep =
            crate::CudaExecutionProvider::new_default().expect("CUDA device must be available");
        let runtime = ep.runtime();
        runtime.set_defer_eager_sync(true);
        let bias = ep.allocate(12, 256).unwrap();
        let output = ep.allocate(32, 256).unwrap();
        let values = [2_f32, 4., 8.];
        // SAFETY: the live bias allocation covers these three f32 values.
        unsafe {
            runtime
                .htod(
                    std::slice::from_raw_parts(
                        values.as_ptr().cast(),
                        std::mem::size_of_val(&values),
                    ),
                    cuptr(bias.as_ptr()),
                )
                .unwrap();
        }
        let mut kernel = GemmKernel {
            runtime: runtime.clone(),
            alpha: 1.,
            beta: 1.,
            trans_a: false,
            trans_b: false,
            nd: true,
            bias_metadata: Mutex::new(BroadcastMetadataCache::new(runtime.clone())),
        };
        // K=0 isolates descriptor preparation and the real Gemm empty-product
        // epilogue without cuBLAS workspace/admission obscuring its counters.
        let execute = |kernel: &GemmKernel, c_shape: Option<&[usize]>, empty: bool| {
            let a_shape = if empty { [0, 2, 0] } else { [2, 2, 0] };
            let b_shape = [0, 2];
            let y_shape = if empty { [0, 2, 2] } else { [2, 2, 2] };
            let a_strides = compute_contiguous_strides(&a_shape)
                .expect("fixture shape has representable contiguous strides");
            let b_strides = compute_contiguous_strides(&b_shape)
                .expect("fixture shape has representable contiguous strides");
            let y_strides = compute_contiguous_strides(&y_shape)
                .expect("fixture shape has representable contiguous strides");
            // A and B contain no elements and are never read. Their pointer
            // borrows live backing rather than inventing a device address.
            let mut inputs = vec![
                TensorView::new(
                    DevicePtr(output.as_ptr()),
                    DataType::Float32,
                    &a_shape,
                    &a_strides,
                    ep.device_id(),
                ),
                TensorView::new(
                    DevicePtr(output.as_ptr()),
                    DataType::Float32,
                    &b_shape,
                    &b_strides,
                    ep.device_id(),
                ),
            ];
            let c_strides = c_shape.map(|shape| {
                compute_contiguous_strides(shape)
                    .expect("fixture shape has representable contiguous strides")
            });
            if let Some(shape) = c_shape {
                inputs.push(TensorView::new(
                    DevicePtr(bias.as_ptr()),
                    DataType::Float32,
                    shape,
                    c_strides.as_ref().unwrap(),
                    ep.device_id(),
                ));
            }
            kernel.execute(
                &inputs,
                &mut [TensorMut::new(
                    DevicePtrMut(output.as_ptr().cast_mut()),
                    DataType::Float32,
                    &y_shape,
                    &y_strides,
                    ep.device_id(),
                )],
            )
        };
        let read = || {
            let mut bytes = [0_u8; 32];
            // SAFETY: the live output covers eight f32 values; dtoh drains its producer.
            unsafe {
                runtime.dtoh(&mut bytes, cuptr(output.as_ptr())).unwrap();
            }
            bytes
                .chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let counts = || {
            (
                runtime.allocation_counts(),
                runtime.transfer_counts(),
                runtime.forced_synchronization_count(),
            )
        };
        let a = [2_usize, 1, 1];
        let b = [1_usize, 2, 1];
        execute(&kernel, Some(&a), false).unwrap();
        assert_eq!(read(), [2., 2., 2., 2., 4., 4., 4., 4.]);
        let identity_a = kernel
            .bias_metadata
            .lock()
            .unwrap()
            .device_graph_resource()
            .unwrap()
            .identity();
        let baseline = counts();
        for _ in 0..8 {
            execute(&kernel, Some(&a), false).unwrap();
        }
        assert_eq!(
            counts(),
            baseline,
            "warm N-D bias must reuse its immutable descriptor"
        );

        let baseline = counts();
        let error = execute(&kernel, Some(&[3]), false).unwrap_err().to_string();
        assert!(error.contains("does not broadcast"), "{error}");
        assert_eq!(
            counts(),
            baseline,
            "invalid bias must fail before device mutation"
        );
        #[cfg(feature = "gpu-tests")]
        {
            runtime.fail_warm_transaction_at_for_test(1);
            let error = execute(&kernel, Some(&b), false).unwrap_err().to_string();
            assert!(
                error.contains("injected staged warm-cache failure"),
                "{error}"
            );
            assert_eq!(
                kernel
                    .bias_metadata
                    .lock()
                    .unwrap()
                    .device_graph_resource()
                    .unwrap()
                    .identity(),
                identity_a
            );
        }
        ep.wait_for_deferred_releases().unwrap();
        let baseline = counts();
        execute(&kernel, Some(&a), false).unwrap();
        assert_eq!(
            counts(),
            baseline,
            "failed replacement must preserve warm A"
        );
        assert_eq!(read(), [2., 2., 2., 2., 4., 4., 4., 4.]);
        execute(&kernel, Some(&b), false).unwrap();
        assert_ne!(
            kernel
                .bias_metadata
                .lock()
                .unwrap()
                .device_graph_resource()
                .unwrap()
                .identity(),
            identity_a
        );
        assert_eq!(read(), [2., 2., 4., 4., 2., 2., 4., 4.]);
        ep.wait_for_deferred_releases().unwrap();
        let baseline = counts();
        for _ in 0..8 {
            execute(&kernel, Some(&b), false).unwrap();
        }
        assert_eq!(
            counts(),
            baseline,
            "warm B must not upload or admit another descriptor"
        );
        let decline = kernel.capture_support();
        assert!(!kernel.cuda_graph_compatible());
        assert!(format!("{decline:?}").contains("execute eagerly"));
        let error = runtime
            .begin_graph_capture(&[&kernel])
            .unwrap_err()
            .to_string();
        assert!(error.contains("rejected before begin_capture"), "{error}");
        assert!(!runtime.is_capturing().unwrap());

        // Poisoned inactive metadata must not make a no-bias/zero-beta Gemm
        // depend on a descriptor it will never read.
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = kernel.bias_metadata.lock().unwrap();
                panic!("fixture: poison bias metadata state");
            }))
            .is_err()
        );
        execute(&kernel, None, false).unwrap();
        assert_eq!(read(), [0.; 8]);
        kernel.beta = 0.;
        execute(&kernel, Some(&a), false).unwrap();
        assert_eq!(read(), [0.; 8]);
        let baseline = counts();
        execute(&kernel, None, true).unwrap();
        assert_eq!(
            counts(),
            baseline,
            "empty output must not touch metadata or device work"
        );
        kernel.beta = 1.;
        let baseline = counts();
        let error = execute(&kernel, Some(&a), false).unwrap_err().to_string();
        assert!(
            error.contains("bias metadata state is poisoned")
                && error.contains("recreate the execution session"),
            "{error}"
        );
        assert_eq!(
            counts(),
            baseline,
            "active poisoned metadata must fail before preparation"
        );
        drop(kernel);
        ep.wait_for_deferred_releases().unwrap();
        ep.deallocate(bias).unwrap();
        ep.deallocate(output).unwrap();
    }
}
