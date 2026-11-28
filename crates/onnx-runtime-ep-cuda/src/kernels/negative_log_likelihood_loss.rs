use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::negative_log_likelihood_loss::{Loss, OP, Reduction, weight};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#if __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#define NXRT_HAS_CUDA_HALF_HEADERS 1
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

using U64 = unsigned long long;
using I64 = long long;

template <typename T> struct Compute {
  using Acc = T;
  __device__ static T load(T x) { return x; }
  __device__ static T store(T x) { return x; }
};
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
template <> struct Compute<__half> {
  using Acc = float;
  __device__ static float load(__half x) { return __half2float(x); }
  __device__ static __half store(float x) { return __float2half_rn(x); }
};
template <> struct Compute<__nv_bfloat16> {
  using Acc = float;
  __device__ static float load(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static __nv_bfloat16 store(float x) { return __float2bfloat16_rn(x); }
};
#endif

template <typename T, typename I>
__device__ void loss(const T* x, const I* target, const T* weight, T* out,
                     U64 n, U64 spatial, U64 classes, I64 ignore,
                     int has_ignore, int reduction) {
  using C = Compute<T>;
  using A = typename C::Acc;
  A sum = 0, denominator = 0;
  for (U64 i = (U64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (U64)gridDim.x * blockDim.x) {
    const I64 label = target[i];
    A value = 0;
    // Targets are checked on the host. Keep invalid offsets out of device loads
    // as well, even when the kernel is invoked directly rather than by a session.
    if ((!has_ignore || label != ignore) && label >= 0 && (U64)label < classes) {
      const U64 offset = (i / spatial * classes + (U64)label) * spatial + i % spatial;
      const A w = weight ? C::load(weight[label]) : (A)1;
      value = -C::load(x[offset]) * w;
      if (reduction == 2) denominator += w;
    }
    if (reduction == 0) out[i] = C::store(value);
    else sum += value;
  }
  if (reduction == 0) return;

  // Reduced outputs use one block so no workspace or atomics are needed.
  __shared__ A sums[256], denominators[256];
  sums[threadIdx.x] = sum;
  denominators[threadIdx.x] = denominator;
  __syncthreads();
  for (unsigned int step = blockDim.x / 2; step; step >>= 1) {
    if (threadIdx.x < step) {
      sums[threadIdx.x] += sums[threadIdx.x + step];
      denominators[threadIdx.x] += denominators[threadIdx.x + step];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[0] = C::store(reduction == 2 ? sums[0] / denominators[0] : sums[0]);
}

#define DEFINE_LOSS(T, I, SUFFIX, INDEX) \
extern "C" __global__ void loss_##SUFFIX##_##INDEX( \
    const T* x, const I* target, const T* weight, T* out, \
    U64 n, U64 spatial, U64 classes, I64 ignore, int has_ignore, int reduction) { \
  loss<T, I>(x, target, weight, out, n, spatial, classes, ignore, has_ignore, reduction); \
}
#define DEFINE_TYPES(T, SUFFIX) \
DEFINE_LOSS(T, int, SUFFIX, i32) \
DEFINE_LOSS(T, I64, SUFFIX, i64)
DEFINE_TYPES(float, f32)
DEFINE_TYPES(double, f64)
#ifdef NXRT_HAS_CUDA_HALF_HEADERS
DEFINE_TYPES(__half, f16)
DEFINE_TYPES(__nv_bfloat16, bf16)
#endif
"#;

pub struct Factory(pub Arc<CudaRuntime>, pub u64);

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(NativeLoss {
            runtime: self.0.clone(),
            loss: Loss::from_node(node, self.1)?,
        }))
    }
}

struct NativeLoss {
    runtime: Arc<CudaRuntime>,
    loss: Loss,
}

impl Kernel for NativeLoss {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let dims = self.loss.validate(inputs, outputs)?;
        if inputs.iter().any(|v| !v.is_absent() && !v.is_contiguous())
            || !outputs[0].is_contiguous()
        {
            return Err(not_implemented(format!(
                "{OP} requires contiguous CUDA tensors"
            )));
        }
        if dims.elements == 0 && self.loss.reduction == Reduction::None {
            return Ok(());
        }
        if self.runtime.is_capturing()? {
            return Err(not_implemented(format!(
                "{OP} target validation requires execution outside CUDA graph capture"
            )));
        }

        let suffix = match inputs[0].dtype {
            DataType::Float32 => "f32",
            DataType::Float64 => "f64",
            DataType::Float16 | DataType::BFloat16 => {
                self.runtime.require_nvrtc_half_headers(OP)?;
                if inputs[0].dtype == DataType::Float16 {
                    "f16"
                } else {
                    "bf16"
                }
            }
            _ => unreachable!("validated loss dtype"),
        };
        let target = &inputs[1];
        let width = target.dtype.byte_size();
        let size = dims.elements.checked_mul(width).ok_or_else(|| {
            EpError::KernelFailed(format!("{OP}: target byte count exceeds usize"))
        })?;
        let index = if target.dtype == DataType::Int32 {
            "i32"
        } else {
            "i64"
        };
        let x_ptr = cuptr(inputs[0].data_ptr::<u8>().cast::<c_void>());
        let target_ptr = cuptr(target.data_ptr::<u8>().cast::<c_void>());
        let weight_ptr = weight(inputs).map_or(0, |w| cuptr(w.data_ptr::<u8>().cast::<c_void>()));
        let out_ptr = cuptr(outputs[0].data_ptr_mut::<u8>().cast::<c_void>());

        // Scores, weights and results stay on device. Read only the targets to
        // report the original invalid class and its position synchronously.
        if size != 0 {
            let mut labels = vec![0; size];
            // SAFETY: the contiguous target contains exactly `size` bytes.
            unsafe { self.runtime.dtoh(&mut labels, target_ptr)? };
            for (pos, raw) in labels.chunks_exact(width).enumerate() {
                let class = if width == 4 {
                    i64::from(i32::from_ne_bytes(raw.try_into().unwrap()))
                } else {
                    i64::from_ne_bytes(raw.try_into().unwrap())
                };
                self.loss.class(class, pos, dims.classes)?;
            }
        }
        let entry = format!("loss_{suffix}_{index}");
        let function =
            self.runtime
                .nvrtc_function("negative_log_likelihood_loss_v1", SOURCE, &entry)?;
        let grid = if self.loss.reduction == Reduction::None {
            dims.elements.div_ceil(BLOCK as usize).min(65535) as u32
        } else {
            1
        };
        let (n, spatial, classes) = (
            dims.elements as u64,
            dims.spatial as u64,
            dims.classes as u64,
        );
        let ignore = self.loss.ignore_index.unwrap_or_default();
        let has_ignore = i32::from(self.loss.ignore_index.is_some());
        let reduction = self.loss.reduction as i32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_ptr)
            .arg(&target_ptr)
            .arg(&weight_ptr)
            .arg(&out_ptr)
            .arg(&n)
            .arg(&spatial)
            .arg(&classes)
            .arg(&ignore)
            .arg(&has_ignore)
            .arg(&reduction);
        // SAFETY: validated tensor shapes and dtypes match the selected entry.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| driver_err(&format!("launch {OP}"), error))
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "NegativeLogLikelihoodLoss reads targets to validate class indices before launching",
        )
    }
}
