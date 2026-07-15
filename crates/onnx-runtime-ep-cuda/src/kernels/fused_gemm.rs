//! cuBLASLt fused-epilogue GEMMs for `com.microsoft::FusedMatMulBias` and
//! `com.microsoft::FusedGemm`.
//!
//! Both kernels require a dense per-N bias vector. cuBLASLt sees the row-major
//! output as column-major `Yᵀ[N,M]`, so its bias-vector length is exactly `N`,
//! matching the ONNX output-channel axis without a transpose or materialization.

use std::ffi::c_void;
use std::sync::Arc;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{GemmDtype, GemmEpilogue, GemmEpilogueKind, GemmEx, WORKSPACE_BYTES, gemm_ex};
use crate::error::not_implemented;
use crate::runtime::{CudaRuntime, cuptr};

use super::gemm::plan_gemm;

pub struct FusedMatMulBiasFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for FusedMatMulBiasFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(FusedEpilogueKernel {
            runtime: self.runtime.clone(),
            op_name: "FusedMatMulBias",
            epilogue: GemmEpilogueKind::Bias,
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
        }))
    }
}

pub struct FusedGemmFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for FusedGemmFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let activation = match node.attr("activation") {
            Some(attr) => attr.as_str().ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep FusedGemm: activation must be a valid UTF-8 string".into(),
                )
            })?,
            None => "Relu",
        };
        let epilogue = parse_activation(activation)?;
        for attr in ["activation_alpha", "activation_beta", "activation_gamma"] {
            if node.attr(attr).is_some() {
                return Err(not_implemented(format!(
                    "FusedGemm attribute {attr}; cuBLASLt Relu/Gelu epilogues have no \
                     parameterized activation form"
                )));
            }
        }
        Ok(Box::new(FusedEpilogueKernel {
            runtime: self.runtime.clone(),
            op_name: "FusedGemm",
            epilogue,
            alpha: node
                .attr("alpha")
                .and_then(|attr| attr.as_float())
                .unwrap_or(1.0),
            beta: node
                .attr("beta")
                .and_then(|attr| attr.as_float())
                .unwrap_or(1.0),
            trans_a: node
                .attr("transA")
                .and_then(|attr| attr.as_int())
                .unwrap_or(0)
                != 0,
            trans_b: node
                .attr("transB")
                .and_then(|attr| attr.as_int())
                .unwrap_or(0)
                != 0,
        }))
    }
}

fn parse_activation(activation: &str) -> Result<GemmEpilogueKind> {
    match activation.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "identity" => Ok(GemmEpilogueKind::Bias),
        "relu" => Ok(GemmEpilogueKind::ReluBias),
        "gelu" => Ok(GemmEpilogueKind::GeluBias),
        other => {
            return Err(not_implemented(format!(
                "FusedGemm activation {other:?}; supported activations are Relu, Gelu, \
                     and an empty/Identity activation"
            )));
        }
    }
}

struct FusedEpilogueKernel {
    runtime: Arc<CudaRuntime>,
    op_name: &'static str,
    epilogue: GemmEpilogueKind,
    alpha: f32,
    beta: f32,
    trans_a: bool,
    trans_b: bool,
}

fn gemm_dtype(dtype: DataType, op_name: &str) -> Result<GemmDtype> {
    match dtype {
        DataType::Float32 => Ok(GemmDtype::F32),
        DataType::Float16 => Ok(GemmDtype::F16),
        DataType::BFloat16 => Ok(GemmDtype::Bf16),
        other => Err(not_implemented(format!(
            "{op_name} dtype {other:?}; supported dtypes are f32, f16, and bf16"
        ))),
    }
}

impl FusedEpilogueKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != 3 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {}: expected A, B, per-N bias and one output, got {} inputs and {} outputs",
                self.op_name,
                inputs.len(),
                outputs.len()
            )));
        }
        let (a, b, bias) = (&inputs[0], &inputs[1], &inputs[2]);
        let dtype = gemm_dtype(a.dtype, self.op_name)?;
        if b.dtype != a.dtype || bias.dtype != a.dtype || outputs[0].dtype != a.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {}: mixed dtypes A={:?} B={:?} bias={:?} Y={:?}; all must match",
                self.op_name, a.dtype, b.dtype, bias.dtype, outputs[0].dtype
            )));
        }
        for (name, contiguous) in [
            ("A", a.is_contiguous()),
            ("B", b.is_contiguous()),
            ("bias", bias.is_contiguous()),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(not_implemented(format!(
                    "{} with non-contiguous {name}; materialise it first",
                    self.op_name
                )));
            }
        }

        let plan = plan_gemm(a.shape, b.shape, self.trans_a, self.trans_b)?;
        if outputs[0].shape != [plan.m, plan.n] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {}: output shape {:?}, expected [{}, {}]",
                self.op_name, outputs[0].shape, plan.m, plan.n
            )));
        }
        if bias.shape != [plan.n] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {}: bias shape {:?}; cuBLASLt fused bias must be the per-N vector [{}]",
                self.op_name, bias.shape, plan.n
            )));
        }
        if self.beta != 1.0 {
            return Err(not_implemented(format!(
                "{} beta={}; cuBLASLt BIAS_POINTER adds an unscaled bias, so fused execution \
                 currently requires beta=1",
                self.op_name, self.beta
            )));
        }

        let a_ptr = cuptr(a.data_ptr::<u8>() as *const c_void);
        let b_ptr = cuptr(b.data_ptr::<u8>() as *const c_void);
        let bias_ptr = cuptr(bias.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let workspace = self.runtime.alloc_raw(WORKSPACE_BYTES)?;
        let params = GemmEx {
            dtype,
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
            epilogue: Some(GemmEpilogue {
                kind: self.epilogue,
                bias: bias_ptr,
            }),
        };
        // SAFETY: dense device buffers and the per-N bias were validated above;
        // the row-major/column-major dimensions and leading dimensions come from
        // the shared Gemm plan; workspace and stream are live for the call.
        let result = unsafe {
            gemm_ex(
                self.runtime.blas(),
                self.runtime.stream_ptr(),
                &params,
                workspace,
                WORKSPACE_BYTES,
            )
        }
        .and_then(|()| self.runtime.synchronize());
        // SAFETY: `workspace` was allocated above and is released exactly once.
        let free = unsafe { self.runtime.free_raw(workspace) };
        result.and(free)
    }
}

impl Kernel for FusedEpilogueKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_mapping_and_rejection() {
        assert_eq!(parse_activation("").unwrap(), GemmEpilogueKind::Bias);
        assert_eq!(
            parse_activation("Relu").unwrap(),
            GemmEpilogueKind::ReluBias
        );
        assert_eq!(
            parse_activation("GELU").unwrap(),
            GemmEpilogueKind::GeluBias
        );
        let error = parse_activation("LeakyRelu").unwrap_err();
        assert!(format!("{error}").contains("supported activations"));
    }
}
