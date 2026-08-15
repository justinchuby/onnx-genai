//! `com.microsoft` fused GELU activations — `BiasGelu`, `FastGelu`, and
//! `QuickGelu` — as f32/f16/bf16 NVRTC kernels.
//!
//! These mirror the CPU EP (`contrib_fused.rs`): `BiasGelu` and `FastGelu`
//! add a broadcast per-last-dimension `bias` before the GELU, and select the
//! exact (error-function) or tanh approximation respectively; `QuickGelu`
//! computes `x · sigmoid(alpha · x)`. Like the CPU kernels, half/bfloat inputs
//! are widened to compute and narrowed back on store, and the exact/tanh GELU
//! is evaluated in `double` so it is numerically identical to the CPU oracle
//! (which computes GELU in `f64`).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const MODULE: &str = "fused_gelu_float_v1";

// `kind` discriminants shared with the NVRTC source below.
const KIND_BIAS: i32 = 0;
const KIND_FAST: i32 = 1;
const KIND_QUICK: i32 = 2;

const SRC: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define NXRT_HAS_CUDA_HALF_HEADERS 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif
template <typename T> __device__ float load_float(T value);
template <> __device__ float load_float<float>(float value) { return value; }
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
template <> __device__ float load_float<__half>(__half value) { return __half2float(value); }
template <> __device__ float load_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }
#endif
template <typename T> __device__ T store_float(float value);
template <> __device__ float store_float<float>(float value) { return value; }
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
template <> __device__ __half store_float<__half>(float value) { return __float2half_rn(value); }
template <> __device__ __nv_bfloat16 store_float<__nv_bfloat16>(float value) { return __float2bfloat16_rn(value); }
#endif

__device__ float fused_gelu_scalar(float summed, int kind) {
    // Match the CPU EP's `-inf -> 0` guard on the (post-bias) GELU argument.
    if (isinf(summed) && summed < 0.0f) return 0.0f;
    const double a = (double)summed;
    double g;
    if (kind == 0) {
        // Exact (erf) GELU: 0.5*a*(1 + erf(a / sqrt(2))).
        g = 0.5 * a * (1.0 + erf(a * 0.7071067811865476));
    } else {
        // Tanh-approximation GELU: 0.5*a*(1 + tanh(sqrt(2/pi)*(a + 0.044715*a^3))).
        const double inner = 0.7978845608028654 * (a + 0.044715 * a * a * a);
        g = 0.5 * a * (1.0 + tanh(inner));
    }
    return (float)g;
}

__device__ float quick_gelu_scalar(float x, float alpha) {
    if (isinf(x) && x < 0.0f) return 0.0f;
    const float z = alpha * x;
    float s;
    if (z >= 0.0f) {
        s = 1.0f / (1.0f + (float)exp((double)-z));
    } else {
        const float e = (float)exp((double)z);
        s = e / (1.0f + e);
    }
    return x * s;
}

#define DEFINE_FUSED_GELU(TYPE, SUFFIX) \
extern "C" __global__ void fused_gelu_##SUFFIX( \
    const TYPE* x, const TYPE* bias, TYPE* y, \
    const unsigned long long n, const unsigned long long width, \
    const int kind, const int has_bias, const float alpha) { \
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) { \
        const float xv = load_float<TYPE>(x[i]); \
        float out; \
        if (kind == 2) { \
            out = quick_gelu_scalar(xv, alpha); \
        } else { \
            const float b = has_bias ? load_float<TYPE>(bias[i % width]) : 0.0f; \
            out = fused_gelu_scalar(xv + b, kind); \
        } \
        y[i] = store_float<TYPE>(out); \
    } \
}

DEFINE_FUSED_GELU(float, f32)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_FUSED_GELU(__half, f16)
DEFINE_FUSED_GELU(__nv_bfloat16, bf16)
#endif
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedGeluOp {
    /// `com.microsoft::BiasGelu` — exact GELU of `x + bias` (bias required).
    Bias,
    /// `com.microsoft::FastGelu` — tanh GELU of `x + bias` (bias optional).
    Fast,
    /// `com.microsoft::QuickGelu` — `x · sigmoid(alpha · x)`.
    Quick,
}

impl FusedGeluOp {
    fn name(self) -> &'static str {
        match self {
            Self::Bias => "BiasGelu",
            Self::Fast => "FastGelu",
            Self::Quick => "QuickGelu",
        }
    }

    fn kind(self) -> i32 {
        match self {
            Self::Bias => KIND_BIAS,
            Self::Fast => KIND_FAST,
            Self::Quick => KIND_QUICK,
        }
    }
}

fn dtype_suffix(op: &str, dtype: DataType) -> Result<&'static str> {
    match dtype {
        DataType::Float32 => Ok("f32"),
        DataType::Float16 => Ok("f16"),
        DataType::BFloat16 => Ok("bf16"),
        other => Err(not_implemented(format!(
            "{op} with dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

pub struct FusedGeluFactory {
    pub op: FusedGeluOp,
    pub runtime: Arc<CudaRuntime>,
}

/// Resolve `QuickGelu`'s `alpha` attribute, defaulting to the `com.microsoft`
/// reference value (`1.702`) the CPU EP uses (`contrib_fused.rs`).
fn quickgelu_alpha(node: &Node) -> f32 {
    node.attr("alpha")
        .and_then(|a| a.as_float())
        .unwrap_or(1.702)
}

impl KernelFactory for FusedGeluFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let alpha = if self.op == FusedGeluOp::Quick {
            quickgelu_alpha(node)
        } else {
            0.0
        };
        Ok(Box::new(FusedGeluKernel {
            op: self.op,
            alpha,
            runtime: self.runtime.clone(),
        }))
    }
}

struct FusedGeluKernel {
    op: FusedGeluOp,
    alpha: f32,
    runtime: Arc<CudaRuntime>,
}

impl FusedGeluKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.op.name();
        // Arity: BiasGelu requires (x, bias); FastGelu accepts (x) or (x, bias);
        // QuickGelu takes (x) only.
        let has_bias_input = match self.op {
            FusedGeluOp::Bias => {
                if inputs.len() != 2 {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: expected 2 inputs (X, bias), got {}",
                        inputs.len()
                    )));
                }
                true
            }
            FusedGeluOp::Fast => match inputs.len() {
                1 => false,
                2 => !inputs[1].is_absent(),
                other => {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: expected 1 or 2 inputs (X[, bias]), got {other}"
                    )));
                }
            },
            FusedGeluOp::Quick => {
                if inputs.len() != 1 {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep {op}: expected 1 input (X), got {}",
                        inputs.len()
                    )));
                }
                false
            }
        };
        if outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 1 output, got {}",
                outputs.len()
            )));
        }

        let x = &inputs[0];
        let suffix = dtype_suffix(op, x.dtype)?;
        if x.dtype != DataType::Float32 {
            self.runtime.require_nvrtc_half_headers(op)?;
        }
        require_contiguous(op, "input", x.is_contiguous())?;
        require_contiguous(op, "output", outputs[0].is_contiguous())?;
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} must equal input shape {:?}",
                outputs[0].shape, x.shape
            )));
        }

        // Resolve the broadcast width (last dimension) for the bias path.
        let width = if has_bias_input {
            let bias = &inputs[1];
            if bias.dtype != x.dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: bias dtype {:?} must equal input dtype {:?}",
                    bias.dtype, x.dtype
                )));
            }
            require_contiguous(op, "bias", bias.is_contiguous())?;
            let Some(&last) = x.shape.last() else {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: X must have rank at least 1 to add a bias"
                )));
            };
            if bias.numel() != last {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: bias has {} elements, expected last dimension {last}",
                    bias.numel()
                )));
            }
            last
        } else {
            1
        };

        let n = x.numel();
        if n == 0 {
            return Ok(());
        }

        let entry = format!("fused_gelu_{suffix}");
        let func = self.runtime.nvrtc_function(MODULE, SRC, &entry)?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let bias_ptr = if has_bias_input {
            cuptr(inputs[1].data_ptr::<u8>() as *const c_void)
        } else {
            cuptr(std::ptr::null())
        };
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let n_u = n as u64;
        let width_u = width as u64;
        let kind = self.op.kind();
        let has_bias = i32::from(has_bias_input);
        let alpha = self.alpha;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            // Roughly the transcendental cost per element (erf/tanh/exp).
            (n as u64).saturating_mul(8)
        });
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&bias_ptr)
            .arg(&y_ptr)
            .arg(&n_u)
            .arg(&width_u)
            .arg(&kind)
            .arg(&has_bias)
            .arg(&alpha);
        // SAFETY: x/y cover n contiguous elements of `suffix` dtype; `bias` (when
        // has_bias) covers `width` elements and is only indexed as i % width.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }
}

impl Kernel for FusedGeluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }
}

fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

fn require_contiguous(op: &str, name: &str, contiguous: bool) -> Result<()> {
    if !contiguous {
        return Err(not_implemented(format!(
            "{op} with a non-contiguous (strided) {name}; materialise it before the op"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvrtc_source_defines_every_dtype_entry() {
        for suffix in ["f32", "f16", "bf16"] {
            assert!(
                SRC.contains(&format!(
                    "DEFINE_FUSED_GELU({}",
                    match suffix {
                        "f32" => "float",
                        "f16" => "__half",
                        _ => "__nv_bfloat16",
                    }
                )),
                "missing fused GELU definition for {suffix}"
            );
        }
    }

    #[test]
    fn quickgelu_alpha_defaults_to_msft_reference() {
        use onnx_runtime_ir::{Attribute, NodeId};
        let node = Node::new(NodeId(0), "QuickGelu", vec![], vec![]);
        assert_eq!(quickgelu_alpha(&node), 1.702);
        let mut with_attr = Node::new(NodeId(0), "QuickGelu", vec![], vec![]);
        with_attr
            .attributes
            .insert("alpha".into(), Attribute::Float(1.5));
        assert_eq!(quickgelu_alpha(&with_attr), 1.5);
    }
}
