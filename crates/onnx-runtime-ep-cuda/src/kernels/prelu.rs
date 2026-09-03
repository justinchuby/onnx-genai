//! ONNX `PRelu` — parametric ReLU with a NumPy-broadcastable `slope` — on the
//! GPU via a runtime-compiled (NVRTC) `extern "C"` kernel. Matches the CPU EP
//! (`crates/onnx-runtime-ep-cpu/src/kernels/norm_ops.rs::prelu_typed`): the
//! output is `x` where `x >= 0`, else `x * slope`, computed in f32 for f16/bf16
//! storage (the same widen-compute-narrow convention as the pointwise slice).
//!
//! ## Scope (all limits are actionable errors, never panics)
//!
//! * **dtype:** f32/f16/bf16; `X`, `slope`, and the output share one dtype.
//! * **broadcasting:** `slope` is unidirectionally broadcast to `X` (right
//!   aligned, size-one/missing axes use a zero stride). The result shape must
//!   equal `X`'s shape — a `slope` axis larger than `X`'s is rejected, matching
//!   the CPU EP's unidirectional contract.
//! * strided views are rejected with a "materialise first" error.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{
    DeviceGraphResource, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

use super::elementwise::{
    BroadcastMetadataCache, BroadcastMetadataKey, capture_shape_eligible,
    require_matching_capture_signature,
};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

/// Threads per block for the 1-D pointwise grid (a full warp-multiple block).
const BLOCK: u32 = 256;

fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

/// The NVRTC entry-point suffix for a supported floating dtype.
fn float_suffix(op: &str, dtype: DataType) -> Result<&'static str> {
    match dtype {
        DataType::Float32 => Ok("f32"),
        DataType::Float16 => Ok("f16"),
        DataType::BFloat16 => Ok("bf16"),
        other => Err(not_implemented(format!(
            "{op} with dtype {other:?} (supported: Float32, Float16, BFloat16)"
        ))),
    }
}

/// NVRTC source: dtype-templated `PRelu`. `slope` is indexed through the shared
/// right-aligned broadcast metadata (`m`: out dims, X strides, slope strides).
const PRELU_SRC: &str = r#"
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

__device__ __forceinline__ void broadcast_indices(unsigned long long out, const unsigned long long* m, int rank, unsigned long long* xi, unsigned long long* si) {
    *xi = 0; *si = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
        unsigned long long coord = out % m[axis]; out /= m[axis];
        *xi += coord * m[rank + axis]; *si += coord * m[2 * rank + axis];
    }
}

#define DEFINE_PRELU(TYPE, SUFFIX) \
extern "C" __global__ void prelu_##SUFFIX(const TYPE* x, const TYPE* slope, TYPE* y, const unsigned long long* m, int rank, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long xi, si; broadcast_indices(i, m, rank, &xi, &si); \
        float xv = load_float<TYPE>(x[xi]); \
        float sv = load_float<TYPE>(slope[si]); \
        float r = (xv >= 0.0f) ? xv : (xv * sv); \
        y[i] = store_float<TYPE>(r); \
    } \
}
DEFINE_PRELU(float, f32)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_PRELU(__half, f16)
DEFINE_PRELU(__nv_bfloat16, bf16)
#endif
"#;

const PRELU_MODULE: &str = "prelu_float_v1";

/// The dtype + operand/broadcast shapes a captured `PRelu` launch is pinned to.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PReluCaptureSignature {
    dtype: DataType,
    shapes: BroadcastMetadataKey,
}

/// Factory for [`PReluKernel`] (no attributes).
pub struct PReluFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for PReluFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(PReluKernel {
            runtime: self.runtime.clone(),
            metadata: Mutex::new(BroadcastMetadataCache::new(self.runtime.clone())),
            last_capture_safe_signature: Mutex::new(None),
            capture_seq_independent: false,
        }))
    }
}

/// NVRTC-backed f32/f16/bf16 `PRelu` kernel with a broadcast `slope`.
#[derive(Debug)]
pub struct PReluKernel {
    runtime: Arc<CudaRuntime>,
    /// Persistent broadcast metadata so a captured launch performs no per-step
    /// host allocation/upload/free/synchronize.
    metadata: Mutex<BroadcastMetadataCache>,
    /// The exact dtype/shape signature recorded by the most recent successful
    /// fixed-decode call. `Some` iff the op is currently capture-safe.
    last_capture_safe_signature: Mutex<Option<PReluCaptureSignature>>,
    /// Metadata-derived seq-independence: `true` iff all IR output dims are
    /// statically known (no growing seq axis), making the op capture-eligible
    /// regardless of the runtime row count.
    capture_seq_independent: bool,
}

impl PReluKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        const OP: &str = "PRelu";
        let mut last_signature = self.last_capture_safe_signature.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep PRelu capture signature lock was poisoned".into())
        })?;
        let warmed_signature = last_signature.clone();
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {OP}: expected 2 inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }
        let x = &inputs[0];
        let slope = &inputs[1];
        let suffix = float_suffix(OP, x.dtype)?;
        if x.dtype != DataType::Float32 {
            self.runtime.require_nvrtc_half_headers(OP)?;
        }
        if slope.dtype != x.dtype || outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {OP}: X/slope/output dtypes must match, got {:?}/{:?}/{:?}",
                x.dtype, slope.dtype, outputs[0].dtype
            )));
        }
        if !x.is_contiguous() || !slope.is_contiguous() || !outputs[0].is_contiguous() {
            return Err(not_implemented(format!(
                "{OP} with a non-contiguous (strided) tensor; \
                 insert an explicit copy to materialise it before the op"
            )));
        }
        if slope.shape.len() > x.shape.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {OP}: slope rank {} exceeds X rank {}",
                slope.shape.len(),
                x.shape.len()
            )));
        }
        // `slope` must be unidirectionally broadcastable *to* X: the broadcast of
        // X and slope has to equal X's shape (a slope axis wider than X's would
        // grow the result and is rejected, matching the CPU EP).
        let broadcast =
            onnx_runtime_ir::broadcast_shapes(x.shape, slope.shape).map_err(EpError::Ir)?;
        if broadcast != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {OP}: slope shape {:?} is not unidirectionally broadcastable to X shape {:?}",
                slope.shape, x.shape
            )));
        }
        if outputs[0].shape != x.shape {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {OP}: output shape {:?} must equal X shape {:?}",
                outputs[0].shape, x.shape
            )));
        }

        let out_shape = x.shape.to_vec();
        let n = outputs[0].numel();
        let n_u64 = u64::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {OP}: {n} elements exceed u64")))?;
        let rank = i32::try_from(out_shape.len())
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {OP}: rank exceeds i32")))?;

        let current_signature = capture_shape_eligible(self.capture_seq_independent, &out_shape)
            .then(|| PReluCaptureSignature {
                dtype: x.dtype,
                shapes: BroadcastMetadataKey {
                    a_shape: x.shape.to_vec(),
                    b_shape: slope.shape.to_vec(),
                    out_shape: out_shape.clone(),
                },
            });
        require_matching_capture_signature(
            &self.runtime,
            OP,
            warmed_signature.as_ref(),
            current_signature.as_ref(),
        )?;

        let entry = format!("prelu_{suffix}");
        let func = self
            .runtime
            .nvrtc_function(PRELU_MODULE, PRELU_SRC, &entry)?;
        let mut metadata = self.metadata.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep PRelu metadata lock was poisoned".into())
        })?;
        let mut metadata_candidate = metadata.clone();
        let metadata_ptr = metadata_candidate.prepare(x.shape, slope.shape, &out_shape)?;
        let x_ptr = cuptr(x.data_ptr::<u8>() as *const c_void);
        let slope_ptr = cuptr(slope.data_ptr::<u8>() as *const c_void);
        let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&x_ptr)
            .arg(&slope_ptr)
            .arg(&y_ptr)
            .arg(&metadata_ptr)
            .arg(&rank)
            .arg(&n_u64);
        // SAFETY: `func` is the compiled `prelu_*` entry; its argument list is
        // (const T*, const T*, T*, metadata, rank, count), where T matches the
        // validated common dtype. All pointers cover their allocations, the
        // metadata holds three rank-length u64 arrays, and the broadcast strides
        // keep every read in bounds. The metadata pointer is the persistent cache
        // buffer, valid across replays.
        unsafe { builder.launch(cfg) }.map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        if !self.runtime.is_capturing()? {
            *metadata = metadata_candidate;
        }
        *last_signature = current_signature;
        Ok(())
    }
}

impl Kernel for PReluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        self.metadata
            .lock()
            .ok()
            .and_then(|metadata| metadata.device_graph_resource())
            .into_iter()
            .collect()
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.last_capture_safe_signature.lock() {
            Ok(signature) if signature.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "PRelu broadcast shape/dtype signature does not match the warmed capture signature",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "PRelu capture signature is unavailable because its state lock was poisoned",
            ),
        }
    }

    fn set_capture_seq_independent(&mut self, seq_independent: bool) {
        self.capture_seq_independent = seq_independent;
    }
}

/// Claim-time contract for `PRelu`: two f32/f16/bf16 operands sharing one dtype,
/// one output. Shape broadcastability is validated at execute (the claim path
/// has no operand shapes), matching the other CUDA broadcasting ops.
pub(crate) fn unsupported_reason(node: &Node, input_dtypes: &[DataType]) -> Option<String> {
    if node.inputs.len() != 2
        || node.outputs.len() != 1
        || node.inputs.iter().any(Option::is_none)
        || input_dtypes.len() != 2
    {
        return Some(format!(
            "PRelu: requires 2 present inputs and 1 output, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    if !matches!(
        input_dtypes[0],
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Some(format!(
            "PRelu: X dtype {:?} unsupported on CUDA EP; expected Float32, Float16, or BFloat16",
            input_dtypes[0]
        ));
    }
    if input_dtypes[1] != input_dtypes[0] {
        return Some(format!(
            "PRelu: slope dtype {:?} must match X dtype {:?} on CUDA EP",
            input_dtypes[1], input_dtypes[0]
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelu_entry_points_are_present_in_source() {
        for expansion in [
            "DEFINE_PRELU(float, f32)",
            "DEFINE_PRELU(__half, f16)",
            "DEFINE_PRELU(__nv_bfloat16, bf16)",
        ] {
            assert!(
                PRELU_SRC.contains(expansion),
                "missing NVRTC generator {expansion}"
            );
        }
    }

    #[test]
    fn claim_rejects_mismatched_slope_dtype() {
        let node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "PRelu",
            vec![
                Some(onnx_runtime_ir::ValueId(0)),
                Some(onnx_runtime_ir::ValueId(1)),
            ],
            vec![onnx_runtime_ir::ValueId(2)],
        );
        let reason = unsupported_reason(&node, &[DataType::Float16, DataType::Float32]);
        assert!(reason.is_some());
    }

    #[test]
    fn claim_accepts_matching_float_dtypes() {
        let node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "PRelu",
            vec![
                Some(onnx_runtime_ir::ValueId(0)),
                Some(onnx_runtime_ir::ValueId(1)),
            ],
            vec![onnx_runtime_ir::ValueId(2)],
        );
        assert!(unsupported_reason(&node, &[DataType::BFloat16, DataType::BFloat16]).is_none());
    }
}
