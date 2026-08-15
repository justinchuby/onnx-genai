//! Variadic elementwise `Sum` and `Mean` (`docs/execution/CUDA_COVERAGE.md`).
//!
//! Both ops accept `1..N` inputs and produce their NumPy multidirectional
//! broadcast shape. Each input is added into an f32 accumulator scratch buffer —
//! matching the CPU EP's f32 compute domain (`elementwise.rs`) so half inputs
//! never lose precision across successive adds — and a final store kernel writes
//! the result to the output dtype, scaling by `1/N` for `Mean`.
//!
//! The kernels are grid-stride, thread-per-output-element, and size their launch
//! from the element count; they carry no model-specific constants.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const MODULE: &str = "nary_sum_mean_v1";

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
template <> __device__ __nv_bfloat16 store_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}
#endif

#define DEFINE_ACCUM(TYPE, SUFFIX) \
extern "C" __global__ void nary_accum_##SUFFIX( \
    const TYPE* a, float* acc, const unsigned long long* metadata, \
    const int rank, const unsigned long long n, const int init) { \
    const unsigned long long* shape = metadata; \
    const unsigned long long* a_strides = metadata + rank; \
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) { \
        unsigned long long linear = i, ai = 0; \
        for (int d = rank - 1; d >= 0; --d) { \
            unsigned long long coord = linear % shape[d]; \
            linear /= shape[d]; \
            ai += coord * a_strides[d]; \
        } \
        const float v = load_float<TYPE>(a[ai]); \
        acc[i] = init ? v : acc[i] + v; \
    } \
}

#define DEFINE_STORE(TYPE, SUFFIX) \
extern "C" __global__ void nary_store_##SUFFIX( \
    const float* acc, TYPE* y, const float scale, const unsigned long long n) { \
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += (unsigned long long)gridDim.x * blockDim.x) \
        y[i] = store_float<TYPE>(acc[i] * scale); \
}

#define DEFINE_FOR_TYPE(TYPE, SUFFIX) \
DEFINE_ACCUM(TYPE, SUFFIX) \
DEFINE_STORE(TYPE, SUFFIX)

DEFINE_FOR_TYPE(float, f32)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_FOR_TYPE(__half, f16)
DEFINE_FOR_TYPE(__nv_bfloat16, bf16)
#endif
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatDtype {
    F32,
    F16,
    Bf16,
}

impl FloatDtype {
    fn from_onnx(op: &str, dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::Float32 => Ok(Self::F32),
            DataType::Float16 => Ok(Self::F16),
            DataType::BFloat16 => Ok(Self::Bf16),
            other => Err(not_implemented(format!(
                "{op} with dtype {other:?} (supported: Float32, Float16, BFloat16)"
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

/// Right-aligned broadcast metadata for one input against `out_shape`: the
/// output shape followed by the input's strides (zero where the input axis is
/// size-one or absent). Returned as `u64` values, ready for an H2D upload.
fn broadcast_metadata(input_shape: &[usize], out_shape: &[usize]) -> Vec<u64> {
    let rank = out_shape.len();
    let offset = rank - input_shape.len();

    let mut input_strides = vec![0u64; input_shape.len()];
    let mut acc = 1u64;
    for d in (0..input_shape.len()).rev() {
        input_strides[d] = acc;
        acc *= input_shape[d] as u64;
    }

    let mut metadata = Vec::with_capacity(rank * 2);
    metadata.extend(out_shape.iter().map(|&dim| dim as u64));
    for d in 0..rank {
        if d < offset {
            metadata.push(0);
        } else {
            let axis = d - offset;
            let stride = if input_shape[axis] == 1 {
                0
            } else {
                input_strides[axis]
            };
            metadata.push(stride);
        }
    }
    metadata
}

fn u64_bytes(values: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &value in values {
        out.extend_from_slice(&value.to_ne_bytes());
    }
    out
}

fn grid_for(n: usize) -> u32 {
    const MAX_BLOCKS: usize = 65_535;
    n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS) as u32
}

/// Factory for [`NaryKernel`]; `is_mean` selects `Mean` (scale by `1/N`) over
/// `Sum`.
pub struct NaryFactory {
    pub is_mean: bool,
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for NaryFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(NaryKernel {
            is_mean: self.is_mean,
            runtime: self.runtime.clone(),
        }))
    }
}

/// NVRTC-backed variadic `Sum`/`Mean` with multidirectional broadcasting and an
/// f32 accumulation domain.
#[derive(Debug)]
pub struct NaryKernel {
    is_mean: bool,
    runtime: Arc<CudaRuntime>,
}

impl NaryKernel {
    fn op_name(&self) -> &'static str {
        if self.is_mean { "Mean" } else { "Sum" }
    }

    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = self.op_name();
        if inputs.is_empty() || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: expected 1+ inputs and 1 output, got {} and {}",
                inputs.len(),
                outputs.len()
            )));
        }

        let dtype = FloatDtype::from_onnx(op, inputs[0].dtype)?;
        if dtype != FloatDtype::F32 {
            self.runtime.require_nvrtc_half_headers(op)?;
        }

        let mut out_shape: Vec<usize> = inputs[0].shape.to_vec();
        for input in inputs {
            if input.dtype != inputs[0].dtype {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: all inputs must share one dtype, got {:?} and {:?}",
                    inputs[0].dtype, input.dtype
                )));
            }
            if !input.is_contiguous() {
                return Err(not_implemented(format!(
                    "{op} with a non-contiguous (strided) input; materialise it before the op"
                )));
            }
            out_shape =
                onnx_runtime_ir::broadcast_shapes(&out_shape, input.shape).map_err(EpError::Ir)?;
        }
        if outputs[0].dtype != inputs[0].dtype {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output dtype {:?} must equal input dtype {:?}",
                outputs[0].dtype, inputs[0].dtype
            )));
        }
        if !outputs[0].is_contiguous() {
            return Err(not_implemented(format!(
                "{op} with a non-contiguous (strided) output; materialise it before the op"
            )));
        }
        if outputs[0].shape != out_shape.as_slice() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {op}: output shape {:?} must equal broadcast shape {out_shape:?}",
                outputs[0].shape
            )));
        }

        let n = outputs[0].numel();
        if n == 0 {
            return Ok(());
        }
        let n_u64 = u64::try_from(n)
            .map_err(|_| EpError::KernelFailed(format!("cuda_ep {op}: {n} elements exceed u64")))?;
        let rank = i32::try_from(out_shape.len()).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep {op}: rank {} exceeds i32",
                out_shape.len()
            ))
        })?;

        let accum =
            self.runtime
                .nvrtc_function(MODULE, SRC, &format!("nary_accum_{}", dtype.suffix()))?;
        let store =
            self.runtime
                .nvrtc_function(MODULE, SRC, &format!("nary_store_{}", dtype.suffix()))?;
        let cfg = LaunchConfig {
            grid_dim: (grid_for(n), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };

        let scratch = self.runtime.alloc_raw(n * std::mem::size_of::<f32>())?;
        let mut metadata_buffers: Vec<CUdeviceptr> = Vec::with_capacity(inputs.len());

        let result = (|| {
            for (index, input) in inputs.iter().enumerate() {
                let metadata = broadcast_metadata(input.shape, &out_shape);
                let metadata_bytes = u64_bytes(&metadata);
                let metadata_ptr = self.runtime.alloc_raw(metadata_bytes.len().max(1))?;
                metadata_buffers.push(metadata_ptr);
                // SAFETY: the fresh allocation exactly covers the metadata slice.
                unsafe { self.runtime.htod(&metadata_bytes, metadata_ptr) }?;

                let a_ptr = cuptr(input.data_ptr::<u8>() as *const c_void);
                let init: i32 = i32::from(index == 0);
                let mut builder = self.runtime.stream().launch_builder(&accum);
                builder
                    .arg(&a_ptr)
                    .arg(&scratch)
                    .arg(&metadata_ptr)
                    .arg(&rank)
                    .arg(&n_u64)
                    .arg(&init);
                // SAFETY: `a_ptr` covers the input's contiguous elements; the
                // scratch buffer holds `n` f32 accumulators; the metadata buffer
                // holds `rank*2` u64 shape/stride words matching the kernel ABI.
                unsafe { builder.launch(cfg) }
                    .map_err(|error| driver_err("launch nary_accum", error))?;
            }

            let scale: f32 = if self.is_mean {
                1.0 / inputs.len() as f32
            } else {
                1.0
            };
            let y_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
            let mut builder = self.runtime.stream().launch_builder(&store);
            builder.arg(&scratch).arg(&y_ptr).arg(&scale).arg(&n_u64);
            // SAFETY: `scratch` holds `n` accumulated f32 values and `y_ptr`
            // covers `n` output elements of the validated dtype.
            unsafe { builder.launch(cfg) }
                .map_err(|error| driver_err("launch nary_store", error))?;
            self.runtime.synchronize()
        })();

        // The synchronize above (or a failed launch) means no kernel is still
        // reading the scratch or metadata buffers, so releasing them is safe.
        // SAFETY: every pointer came from this runtime's `alloc_raw` and is
        // freed exactly once here.
        let mut cleanup = unsafe { self.runtime.free_raw(scratch) };
        for metadata_ptr in metadata_buffers {
            // SAFETY: as above; each metadata pointer is freed exactly once.
            let free = unsafe { self.runtime.free_raw(metadata_ptr) };
            cleanup = cleanup.and(free);
        }
        result.and(cleanup)
    }
}

impl Kernel for NaryKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "Sum/Mean allocate and upload per-input broadcast metadata on every call",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_present_in_source() {
        assert!(SRC.contains("DEFINE_FOR_TYPE(float, f32)"));
        assert!(SRC.contains("DEFINE_FOR_TYPE(__half, f16)"));
        assert!(SRC.contains("DEFINE_FOR_TYPE(__nv_bfloat16, bf16)"));
        assert!(SRC.contains("nary_accum_##SUFFIX"));
        assert!(SRC.contains("nary_store_##SUFFIX"));
    }

    #[test]
    fn broadcast_metadata_right_aligns_and_zeroes_broadcast_axes() {
        // Input [3] broadcast into [2,3]: leading axis absent (stride 0), last
        // axis contiguous (stride 1).
        assert_eq!(broadcast_metadata(&[3], &[2, 3]), vec![2, 3, 0, 1]);
        // Input [2,1] broadcast into [2,3]: size-one axis has stride 0.
        assert_eq!(broadcast_metadata(&[2, 1], &[2, 3]), vec![2, 3, 1, 0]);
        // Equal shape [2,3]: row-major strides.
        assert_eq!(broadcast_metadata(&[2, 3], &[2, 3]), vec![2, 3, 3, 1]);
        // Scalar output.
        assert_eq!(broadcast_metadata(&[], &[]), Vec::<u64>::new());
    }
}
