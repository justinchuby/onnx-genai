//! Attribute-driven f32/f16/bf16 activation kernels (CUDA Wave 4).

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const MODULE: &str = "wave4_activations_float_v4";

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

__device__ float op_leaky_relu(float v, float alpha, float unused) { return v >= 0.0f ? v : alpha * v; }
__device__ float op_elu(float v, float alpha, float unused) { return v >= 0.0f ? v : alpha * expm1f(v); }
__device__ float op_hard_sigmoid(float v, float alpha, float beta) {
    v = alpha * v + beta;
    return isnan(v) ? v : (v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v));
}
__device__ float op_clip(float v, float min_value, float max_value) {
    return isnan(v) ? v : fminf(fmaxf(v, min_value), max_value);
}
__device__ float op_softsign(float v, float unused0, float unused1) { return v / (1.0f + fabsf(v)); }
__device__ float op_selu(float v, float alpha, float gamma) {
    return gamma * (v >= 0.0f ? v : alpha * expm1f(v));
}
__device__ float op_thresholded_relu(float v, float alpha, float unused) { return v > alpha ? v : 0.0f; }
// Celu(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1)).  NaN is propagated
// explicitly because CUDA's fmaxf/fminf (like Rust's f32::max/min) return the
// non-NaN operand when one argument is NaN, so both min/max terms would
// collapse to 0 and Celu(NaN) would come out 0.  The CPU scalar does the same
// isnan guard for the same reason.
__device__ float op_celu(float v, float alpha, float unused) {
    if (isnan(v)) return v;
    return fmaxf(v, 0.0f) + fminf(0.0f, alpha * (expf(v / alpha) - 1.0f));
}
__device__ float op_swish(float v, float alpha, float unused) {
    const float z = alpha * v;
    float s;
    if (z >= 0.0f) {
        s = 1.0f / (1.0f + (float)exp((double)-z));
    } else {
        const float e = (float)exp((double)z);
        s = e / (1.0f + e);
    }
    return v * s;
}

#define DEFINE_ACT(NAME, TYPE, SUFFIX) \
extern "C" __global__ void NAME##_##SUFFIX( \
    const TYPE* x, TYPE* y, const int n, const float p0, const float p1) { \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) \
        y[i] = store_float<TYPE>(op_##NAME(load_float<TYPE>(x[i]), p0, p1)); \
}
#define DEFINE_FOR_TYPE(TYPE, SUFFIX) \
DEFINE_ACT(leaky_relu, TYPE, SUFFIX) \
DEFINE_ACT(elu, TYPE, SUFFIX) \
DEFINE_ACT(hard_sigmoid, TYPE, SUFFIX) \
DEFINE_ACT(clip, TYPE, SUFFIX) \
DEFINE_ACT(softsign, TYPE, SUFFIX) \
DEFINE_ACT(selu, TYPE, SUFFIX) \
DEFINE_ACT(thresholded_relu, TYPE, SUFFIX) \
DEFINE_ACT(celu, TYPE, SUFFIX) \
DEFINE_ACT(swish, TYPE, SUFFIX)
DEFINE_FOR_TYPE(float, f32)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_FOR_TYPE(__half, f16)
DEFINE_FOR_TYPE(__nv_bfloat16, bf16)
#endif
"#;

// Integer Clip (opset-11 min/max arrive as inputs). Kept in a separate module
// from the float activation kernels because the clamp is exact in the integer
// domain — routing int32/int64 through the float `op_clip` above would lose
// precision for magnitudes beyond 2^24.
const INT_MODULE: &str = "clip_integer_v1";
const INT_SRC: &str = r#"
extern "C" __global__ void clip_i32(
    const int* x, int* y, const int n, const long long lo, const long long hi) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        long long v = (long long)x[i];
        v = v < lo ? lo : (v > hi ? hi : v);
        y[i] = (int)v;
    }
}
extern "C" __global__ void clip_i64(
    const long long* x, long long* y, const int n, const long long lo, const long long hi) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        long long v = x[i];
        y[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatDtype {
    F32,
    F16,
    Bf16,
}

impl FloatDtype {
    fn from_onnx(op: &str, name: &str, dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::Float32 => Ok(Self::F32),
            DataType::Float16 => Ok(Self::F16),
            DataType::BFloat16 => Ok(Self::Bf16),
            other => Err(not_implemented(format!(
                "{op} with {name} dtype {other:?} (supported: Float32, Float16, BFloat16)"
            ))),
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationOp {
    LeakyRelu { alpha: f32 },
    Elu { alpha: f32 },
    HardSigmoid { alpha: f32, beta: f32 },
    Clip { min: f32, max: f32 },
    Softsign,
    Selu { alpha: f32, gamma: f32 },
    ThresholdedRelu { alpha: f32 },
    Celu { alpha: f32 },
    Swish { alpha: f32 },
}

impl ActivationOp {
    fn stem(self) -> &'static str {
        match self {
            Self::LeakyRelu { .. } => "leaky_relu",
            Self::Elu { .. } => "elu",
            Self::HardSigmoid { .. } => "hard_sigmoid",
            Self::Clip { .. } => "clip",
            Self::Softsign => "softsign",
            Self::Selu { .. } => "selu",
            Self::ThresholdedRelu { .. } => "thresholded_relu",
            Self::Celu { .. } => "celu",
            Self::Swish { .. } => "swish",
        }
    }

    fn entry(self, dtype: FloatDtype) -> String {
        format!("{}_{}", self.stem(), dtype.suffix())
    }

    fn name(self) -> &'static str {
        match self {
            Self::LeakyRelu { .. } => "LeakyRelu",
            Self::Elu { .. } => "Elu",
            Self::HardSigmoid { .. } => "HardSigmoid",
            Self::Clip { .. } => "Clip",
            Self::Softsign => "Softsign",
            Self::Selu { .. } => "Selu",
            Self::ThresholdedRelu { .. } => "ThresholdedRelu",
            Self::Celu { .. } => "Celu",
            Self::Swish { .. } => "Swish",
        }
    }

    fn params(self) -> (f32, f32) {
        match self {
            Self::LeakyRelu { alpha }
            | Self::Elu { alpha }
            | Self::ThresholdedRelu { alpha }
            | Self::Celu { alpha }
            | Self::Swish { alpha } => (alpha, 0.0),
            Self::HardSigmoid { alpha, beta } => (alpha, beta),
            Self::Clip { min, max } => (min, max),
            Self::Softsign => (0.0, 0.0),
            Self::Selu { alpha, gamma } => (alpha, gamma),
        }
    }
}

pub struct ActivationFactory {
    pub name: &'static str,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for ActivationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let op = activation_from_node(self.name, node)?;
        Ok(Box::new(ActivationKernel {
            op,
            runtime: self.runtime.clone(),
        }))
    }
}

fn activation_from_node(name: &str, node: &Node) -> Result<ActivationOp> {
    Ok(match name {
        "LeakyRelu" => ActivationOp::LeakyRelu {
            alpha: node
                .attr("alpha")
                .and_then(|a| a.as_float())
                .unwrap_or(0.01),
        },
        "Elu" => ActivationOp::Elu {
            alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0),
        },
        "HardSigmoid" => ActivationOp::HardSigmoid {
            alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(0.2),
            beta: node.attr("beta").and_then(|a| a.as_float()).unwrap_or(0.5),
        },
        "Clip" => ActivationOp::Clip {
            min: node
                .attr("min")
                .and_then(|a| a.as_float())
                .unwrap_or(f32::MIN),
            max: node
                .attr("max")
                .and_then(|a| a.as_float())
                .unwrap_or(f32::MAX),
        },
        "Softsign" => ActivationOp::Softsign,
        "Selu" => ActivationOp::Selu {
            alpha: node
                .attr("alpha")
                .and_then(|a| a.as_float())
                .unwrap_or(1.67326),
            gamma: node
                .attr("gamma")
                .and_then(|a| a.as_float())
                .unwrap_or(1.0507),
        },
        "ThresholdedRelu" => ActivationOp::ThresholdedRelu {
            alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0),
        },
        "Celu" => ActivationOp::Celu {
            alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0),
        },
        "Swish" => ActivationOp::Swish {
            alpha: node.attr("alpha").and_then(|a| a.as_float()).unwrap_or(1.0),
        },
        other => {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: unknown activation factory {other}"
            )));
        }
    })
}

#[derive(Debug)]
struct ActivationKernel {
    op: ActivationOp,
    runtime: Arc<CudaRuntime>,
}

impl ActivationKernel {
    fn read_scalar(&self, name: &str, input: &TensorView) -> Result<f32> {
        FloatDtype::from_onnx(self.op.name(), name, input.dtype)?;
        require_contiguous(self.op.name(), name, input.is_contiguous())?;
        if input.numel() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Clip: {name} must contain one floating-point value, got {}",
                input.numel()
            )));
        }
        let mut bytes = [0u8; 4];
        let width = input.dtype.byte_size();
        // SAFETY: dtype and numel checks prove the allocation covers one element.
        unsafe {
            self.runtime.dtoh(
                &mut bytes[..width],
                cuptr(input.data_ptr::<u8>() as *const c_void),
            )?
        };
        Ok(match input.dtype {
            DataType::Float32 => f32::from_ne_bytes(bytes),
            DataType::Float16 => {
                half::f16::from_bits(u16::from_ne_bytes([bytes[0], bytes[1]])).to_f32()
            }
            DataType::BFloat16 => {
                half::bf16::from_bits(u16::from_ne_bytes([bytes[0], bytes[1]])).to_f32()
            }
            _ => unreachable!("validated floating dtype"),
        })
    }

    /// Read a single Int32/Int64 clip bound scalar back to the host as i64.
    fn read_int_scalar(&self, name: &str, input: &TensorView) -> Result<i64> {
        require_contiguous("Clip", name, input.is_contiguous())?;
        if input.numel() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Clip: {name} must contain one integer value, got {}",
                input.numel()
            )));
        }
        let mut bytes = [0u8; 8];
        let width = input.dtype.byte_size();
        // SAFETY: dtype and numel checks prove the allocation covers one element.
        unsafe {
            self.runtime.dtoh(
                &mut bytes[..width],
                cuptr(input.data_ptr::<u8>() as *const c_void),
            )?
        };
        Ok(match input.dtype {
            DataType::Int32 => i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64,
            DataType::Int64 => i64::from_ne_bytes(bytes),
            other => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Clip: {name} dtype {other:?} is not an integer"
                )));
            }
        })
    }

    /// Exact integer clamp for Int32/Int64 Clip (opset-11 min/max inputs).
    fn run_int_clip(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let x = &inputs[0];
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Clip: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        require_contiguous("Clip", "input", x.is_contiguous())?;
        require_contiguous("Clip", "output", outputs[0].is_contiguous())?;
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Clip: output shape {:?} must equal input shape {:?}",
                outputs[0].shape, x.shape
            )));
        }

        let (mut lo, mut hi) = (i64::MIN, i64::MAX);
        if let Some(min) = inputs.get(1).filter(|input| !input.is_absent()) {
            if min.dtype != x.dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Clip: min dtype {:?} must equal input dtype {:?}",
                    min.dtype, x.dtype
                )));
            }
            lo = self.read_int_scalar("min", min)?;
        }
        if let Some(max) = inputs.get(2).filter(|input| !input.is_absent()) {
            if max.dtype != x.dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep Clip: max dtype {:?} must equal input dtype {:?}",
                    max.dtype, x.dtype
                )));
            }
            hi = self.read_int_scalar("max", max)?;
        }
        if lo > hi {
            return Err(EpError::KernelFailed(
                "cuda_ep Clip: min must not exceed max".into(),
            ));
        }

        let n = x.numel();
        let n_i = i32::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep Clip: {n} elements exceed i32")))?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let entry = match x.dtype {
            DataType::Int32 => "clip_i32",
            DataType::Int64 => "clip_i64",
            _ => unreachable!("run_int_clip only reached for integer dtypes"),
        };
        let func = self.runtime.nvrtc_function(INT_MODULE, INT_SRC, entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        crate::trace::record_kernel_metrics(inputs, outputs, || n as u64);
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder.arg(&x_ptr).arg(&y_ptr).arg(&n_i).arg(&lo).arg(&hi);
        // SAFETY: clip_i32/clip_i64 share the (x, y, n, lo, hi) signature; x/y
        // cover n contiguous elements of the validated integer dtype.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }

    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.op.name();
        let valid_arity = if matches!(self.op, ActivationOp::Clip { .. }) {
            (1..=3).contains(&inputs.len())
        } else {
            inputs.len() == 1
        };
        if !valid_arity || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected {} input(s) and 1 output, got {} and {}",
                if op == "Clip" { "1-3" } else { "1" },
                inputs.len(),
                outputs.len()
            )));
        }

        // Integer Clip runs an exact integer clamp; every other activation (and
        // float Clip) uses the float kernels below.
        if matches!(self.op, ActivationOp::Clip { .. })
            && matches!(inputs[0].dtype, DataType::Int32 | DataType::Int64)
        {
            return self.run_int_clip(inputs, outputs);
        }

        let x = &inputs[0];
        let dtype = FloatDtype::from_onnx(op, "input", x.dtype)?;
        if dtype != FloatDtype::F32 {
            self.runtime.require_nvrtc_half_headers(op)?;
        }
        if outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, x.dtype
            )));
        }
        require_contiguous(op, "input", x.is_contiguous())?;
        require_contiguous(op, "output", outputs[0].is_contiguous())?;
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} must equal input shape {:?}",
                outputs[0].shape, x.shape
            )));
        }

        let (mut p0, mut p1) = self.op.params();
        if matches!(self.op, ActivationOp::Clip { .. }) {
            // Optional inputs retain their positional slots, so max may be
            // present at index 2 while the min placeholder at index 1 is absent.
            if let Some(min) = inputs.get(1).filter(|input| !input.is_absent()) {
                if min.dtype != x.dtype {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Clip: min dtype {:?} must equal input dtype {:?}",
                        min.dtype, x.dtype
                    )));
                }
                p0 = self.read_scalar("min", min)?;
            }
            if let Some(max) = inputs.get(2).filter(|input| !input.is_absent()) {
                if max.dtype != x.dtype {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Clip: max dtype {:?} must equal input dtype {:?}",
                        max.dtype, x.dtype
                    )));
                }
                p1 = self.read_scalar("max", max)?;
            }
            if p0 > p1 {
                return Err(EpError::KernelFailed(
                    "cuda_ep Clip: min must not exceed max".into(),
                ));
            }
        }

        let n = x.numel();
        let n_i = i32::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {n} elements exceed i32")))?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let entry = self.op.entry(dtype);
        let func = self.runtime.nvrtc_function(MODULE, SRC, &entry)?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let per_element = match self.op {
                ActivationOp::LeakyRelu { .. }
                | ActivationOp::Clip { .. }
                | ActivationOp::ThresholdedRelu { .. } => 1,
                ActivationOp::HardSigmoid { .. } | ActivationOp::Softsign => 3,
                ActivationOp::Elu { .. } | ActivationOp::Selu { .. } => 4,
                // Celu: div + exp + sub + mul + max + min + add ≈ 5 flops
                ActivationOp::Celu { .. } | ActivationOp::Swish { .. } => 5,
            };
            (n as u64).saturating_mul(per_element)
        });
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder.arg(&x_ptr).arg(&y_ptr).arg(&n_i).arg(&p0).arg(&p1);
        // SAFETY: every entry in SRC has the same (x, y, n, p0, p1) signature;
        // x/y cover n contiguous f32 elements, validated above.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }
}

impl Kernel for ActivationKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        if matches!(self.op, ActivationOp::Clip { .. }) {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Clip reads optional min/max scalars back to the host before launch",
            )
        } else {
            onnx_runtime_ep_api::CaptureSupport::Supported
        }
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
    use onnx_runtime_ir::{Attribute, NodeId};

    fn node(op: &str) -> Node {
        Node::new(NodeId(0), op, vec![], vec![])
    }

    #[test]
    fn entries_are_present_in_nvrtc_source() {
        for op in [
            ActivationOp::LeakyRelu { alpha: 0.01 },
            ActivationOp::Elu { alpha: 1.0 },
            ActivationOp::HardSigmoid {
                alpha: 0.2,
                beta: 0.5,
            },
            ActivationOp::Clip {
                min: f32::MIN,
                max: f32::MAX,
            },
            ActivationOp::Softsign,
            ActivationOp::Selu {
                alpha: 1.67326,
                gamma: 1.0507,
            },
            ActivationOp::ThresholdedRelu { alpha: 1.0 },
            ActivationOp::Celu { alpha: 1.0 },
            ActivationOp::Swish { alpha: 1.0 },
        ] {
            assert!(
                SRC.contains(&format!("DEFINE_ACT({},", op.stem())),
                "missing {}",
                op.stem()
            );
        }
    }

    #[test]
    fn defaults_and_attributes_match_cpu_references() {
        assert_eq!(
            activation_from_node("LeakyRelu", &node("LeakyRelu")).unwrap(),
            ActivationOp::LeakyRelu { alpha: 0.01 }
        );
        assert_eq!(
            activation_from_node("Elu", &node("Elu")).unwrap(),
            ActivationOp::Elu { alpha: 1.0 }
        );
        assert_eq!(
            activation_from_node("HardSigmoid", &node("HardSigmoid")).unwrap(),
            ActivationOp::HardSigmoid {
                alpha: 0.2,
                beta: 0.5
            }
        );
        assert_eq!(
            activation_from_node("Clip", &node("Clip")).unwrap(),
            ActivationOp::Clip {
                min: f32::MIN,
                max: f32::MAX
            }
        );
        assert_eq!(
            activation_from_node("Selu", &node("Selu")).unwrap(),
            ActivationOp::Selu {
                alpha: 1.67326,
                gamma: 1.0507
            }
        );
        let mut leaky = node("LeakyRelu");
        leaky
            .attributes
            .insert("alpha".into(), Attribute::Float(0.25));
        assert_eq!(
            activation_from_node("LeakyRelu", &leaky).unwrap(),
            ActivationOp::LeakyRelu { alpha: 0.25 }
        );

        let mut elu = node("Elu");
        elu.attributes
            .insert("alpha".into(), Attribute::Float(0.75));
        assert_eq!(
            activation_from_node("Elu", &elu).unwrap(),
            ActivationOp::Elu { alpha: 0.75 }
        );

        let mut hard = node("HardSigmoid");
        hard.attributes
            .insert("alpha".into(), Attribute::Float(0.3));
        hard.attributes.insert("beta".into(), Attribute::Float(0.4));
        assert_eq!(
            activation_from_node("HardSigmoid", &hard).unwrap(),
            ActivationOp::HardSigmoid {
                alpha: 0.3,
                beta: 0.4
            }
        );

        let mut clip = node("Clip");
        clip.attributes.insert("min".into(), Attribute::Float(-2.0));
        clip.attributes.insert("max".into(), Attribute::Float(3.0));
        assert_eq!(
            activation_from_node("Clip", &clip).unwrap(),
            ActivationOp::Clip {
                min: -2.0,
                max: 3.0
            }
        );

        let mut selu = node("Selu");
        selu.attributes
            .insert("alpha".into(), Attribute::Float(1.5));
        selu.attributes
            .insert("gamma".into(), Attribute::Float(1.1));
        assert_eq!(
            activation_from_node("Selu", &selu).unwrap(),
            ActivationOp::Selu {
                alpha: 1.5,
                gamma: 1.1
            }
        );

        assert_eq!(
            activation_from_node("ThresholdedRelu", &node("ThresholdedRelu")).unwrap(),
            ActivationOp::ThresholdedRelu { alpha: 1.0 }
        );
        let mut thresh = node("ThresholdedRelu");
        thresh
            .attributes
            .insert("alpha".into(), Attribute::Float(2.5));
        assert_eq!(
            activation_from_node("ThresholdedRelu", &thresh).unwrap(),
            ActivationOp::ThresholdedRelu { alpha: 2.5 }
        );

        assert_eq!(
            activation_from_node("Swish", &node("Swish")).unwrap(),
            ActivationOp::Swish { alpha: 1.0 }
        );
        let mut swish = node("Swish");
        swish
            .attributes
            .insert("alpha".into(), Attribute::Float(0.5));
        assert_eq!(
            activation_from_node("Swish", &swish).unwrap(),
            ActivationOp::Swish { alpha: 0.5 }
        );

        assert_eq!(
            activation_from_node("Celu", &node("Celu")).unwrap(),
            ActivationOp::Celu { alpha: 1.0 }
        );
        let mut celu = node("Celu");
        celu.attributes
            .insert("alpha".into(), Attribute::Float(2.0));
        assert_eq!(
            activation_from_node("Celu", &celu).unwrap(),
            ActivationOp::Celu { alpha: 2.0 }
        );
    }
}
