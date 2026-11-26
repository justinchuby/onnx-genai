//! Fused ONNX MeanVarianceNormalization-29 without device intermediates.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::mean_variance_normalization as mvn;
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
// A nonempty dense shape with a checked usize element count cannot have 64
// non-unit dimensions. Singleton dimensions need no metadata, regardless of rank.
const MAX_RUNS: usize = 63;

#[repr(C)]
struct Params {
    groups: u64,
    width: u64,
    extents: [u64; MAX_RUNS],
    strides: [u64; MAX_RUNS],
    kept: u32,
    total: u32,
}

// SAFETY: integer-only repr(C) layout matches the NVRTC Params.
unsafe impl DeviceRepr for Params {}

const HALF_PREFIX: &str = r#"
#define NXRT_MVN_HALF
#include <cuda_fp16.h>
#include <cuda_bf16.h>
"#;

const SOURCE: &str = r#"
using Index = unsigned long long;
struct Params {
    Index groups, width;
    Index extents[63], strides[63];
    unsigned int kept, total;
};

__device__ Index offset(Index i, const Params& p, unsigned int first, unsigned int end) {
    if (first == end) return 0;
    Index result = 0;
    for (unsigned int j = first; j + 1 < end; ++j) {
        result += (i % p.extents[j]) * p.strides[j];
        i /= p.extents[j];
    }
    return result + i * p.strides[end - 1];
}

template <typename T, typename Acc> __device__ T store(Acc x) { return static_cast<T>(x); }
template <typename T> __device__ T load(T x) { return x; }
#ifdef NXRT_MVN_HALF
template <> __device__ __half store<__half, float>(float x) { return __float2half_rn(x); }
template <> __device__ __nv_bfloat16 store<__nv_bfloat16, float>(float x) { return __float2bfloat16_rn(x); }
__device__ float load(__half x) { return __half2float(x); }
__device__ float load(__nv_bfloat16 x) { return __bfloat162float(x); }
#endif
__device__ float root(float x) { return sqrtf(x); }
__device__ double root(double x) { return sqrt(x); }

template <typename Acc>
__device__ Acc block_sum(Acc x, Acc* scratch) {
    for (int step = 16; step > 0; step /= 2)
        x += __shfl_down_sync(0xffffffff, x, step);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x / 32] = x;
    __syncthreads();
    if (threadIdx.x < 32) {
        x = threadIdx.x < blockDim.x / 32 ? scratch[threadIdx.x] : Acc(0);
        for (int step = 16; step > 0; step /= 2)
            x += __shfl_down_sync(0xffffffff, x, step);
    }
    if (threadIdx.x == 0) scratch[8] = x;
    __syncthreads();
    x = scratch[8];
    __syncthreads();
    return x;
}

template <typename T, typename Acc>
__device__ void normalize(const T* x, T* y, const Params& p, float epsilon) {
    __shared__ Acc scratch[9];
    for (Index group = blockIdx.x; group < p.groups; group += gridDim.x) {
        const Index base = offset(group, p, 0, p.kept);
        Acc sum = 0;
        for (Index i = threadIdx.x; i < p.width; i += blockDim.x)
            sum += load(x[base + offset(i, p, p.kept, p.total)]);
        const Acc mean = block_sum(sum, scratch) / Acc(p.width);
        Acc squared = 0;
        for (Index i = threadIdx.x; i < p.width; i += blockDim.x) {
            const Acc d = load(x[base + offset(i, p, p.kept, p.total)]) - mean;
            squared += d * d;
        }
        const Acc denominator = root(block_sum(squared, scratch) / Acc(p.width)) + Acc(epsilon);
        for (Index i = threadIdx.x; i < p.width; i += blockDim.x) {
            const Index j = base + offset(i, p, p.kept, p.total);
            y[j] = store<T>((load(x[j]) - mean) / denominator);
        }
    }
}

#if __CUDA_ARCH__ >= 700
#define MVN_GRID_CONSTANT __grid_constant__
#else
#define MVN_GRID_CONSTANT
#endif
#define DEFINE_MVN(TYPE, ACC, TAG) \
extern "C" __global__ void mvn_##TAG(const TYPE* x, TYPE* y, const MVN_GRID_CONSTANT Params p, float eps) { \
    normalize<TYPE, ACC>(x, y, p, eps); \
}
DEFINE_MVN(float, float, f32)
DEFINE_MVN(double, double, f64)
#ifdef NXRT_MVN_HALF
DEFINE_MVN(__half, float, f16)
DEFINE_MVN(__nv_bfloat16, float, bf16)
#endif
"#;

pub struct MeanVarianceNormalizationFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for MeanVarianceNormalizationFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MeanVarianceNormalization {
            runtime: self.runtime.clone(),
            attrs: mvn::Attributes::parse(node)?,
        }))
    }
}

struct MeanVarianceNormalization {
    runtime: Arc<CudaRuntime>,
    attrs: mvn::Attributes,
}

impl Kernel for MeanVarianceNormalization {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let op = mvn::OP;
        let ([x], [y]) = (inputs, outputs) else {
            return Err(EpError::KernelFailed(format!(
                "{op}: expected one input and one output"
            )));
        };
        if x.shape != y.shape || x.dtype != y.dtype {
            return Err(EpError::KernelFailed(format!(
                "{op}: output {:?} {:?} must match X {:?} {:?}",
                y.dtype, y.shape, x.dtype, x.shape
            )));
        }
        let entry = match x.dtype {
            DataType::Float32 => "mvn_f32",
            DataType::Float64 => "mvn_f64",
            DataType::Float16 => "mvn_f16",
            DataType::BFloat16 => "mvn_bf16",
            dtype => {
                return Err(not_implemented(format!(
                    "{op} input dtype {dtype:?}; expected a floating tensor"
                )));
            }
        };
        let layout = self.attrs.layout(x.shape)?;
        if layout.groups == 0 {
            return Ok(());
        }
        if !x.is_contiguous() || !y.is_contiguous() {
            return Err(not_implemented(format!(
                "{op} requires contiguous tensors on CUDA"
            )));
        }
        let mut params = Params {
            groups: layout.groups as u64,
            width: layout.width as u64,
            extents: [0; MAX_RUNS],
            strides: [0; MAX_RUNS],
            kept: layout.kept.len() as u32,
            total: (layout.kept.len() + layout.reduced.len()) as u32,
        };
        for (i, run) in layout.kept.iter().chain(&layout.reduced).enumerate() {
            params.extents[i] = run.extent as u64;
            params.strides[i] = run.stride as u64;
        }
        let function = if matches!(x.dtype, DataType::Float16 | DataType::BFloat16) {
            self.runtime.require_nvrtc_half_headers(op)?;
            static HALF_SOURCE: std::sync::LazyLock<String> =
                std::sync::LazyLock::new(|| format!("{HALF_PREFIX}{SOURCE}"));
            self.runtime.nvrtc_function(
                "mean_variance_normalization_half_v1",
                &HALF_SOURCE,
                entry,
            )?
        } else {
            self.runtime
                .nvrtc_function("mean_variance_normalization_v1", SOURCE, entry)?
        };
        let x_pointer = cuptr(x.data_ptr::<u8>() as *const c_void);
        let y_pointer = cuptr(y.data_ptr_mut::<u8>() as *const c_void);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_pointer)
            .arg(&y_pointer)
            .arg(&params)
            .arg(&self.attrs.epsilon);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (layout.groups.min(i32::MAX as usize) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err(&format!("launch {op}"), error))
    }
}
