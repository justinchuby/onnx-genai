//! Allocation-free greedy argmax for native CUDA decode.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{DeviceBuffer, EpError, Result};
use onnx_runtime_ir::DataType;

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const RESULT_BYTES: usize = 2 * std::mem::size_of::<u32>();

const SOURCE: &str = r#"
#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ float argmax_load(T value);

template <>
__device__ __forceinline__ float argmax_load<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ float argmax_load<__half>(__half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ void greedy_argmax_impl(
    const T* logits,
    unsigned long long elements,
    const unsigned int* capture_error,
    unsigned int* result) {
  extern __shared__ unsigned char shared_bytes[];
  float* best_values = reinterpret_cast<float*>(shared_bytes);
  unsigned int* best_indices =
      reinterpret_cast<unsigned int*>(best_values + blockDim.x);

  float best = -1.0f / 0.0f;
  unsigned int best_index = 0;
  for (unsigned long long i = threadIdx.x; i < elements; i += blockDim.x) {
    float value = argmax_load<T>(logits[i]);
    if (isnan(value)) continue;
    unsigned int index = static_cast<unsigned int>(i);
    if (value > best || (value == best && index < best_index)) {
      best = value;
      best_index = index;
    }
  }

  best_values[threadIdx.x] = best;
  best_indices[threadIdx.x] = best_index;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      float candidate = best_values[threadIdx.x + stride];
      unsigned int candidate_index = best_indices[threadIdx.x + stride];
      if (candidate > best ||
          (candidate == best && candidate_index < best_index)) {
        best = candidate;
        best_index = candidate_index;
        best_values[threadIdx.x] = candidate;
        best_indices[threadIdx.x] = candidate_index;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    result[0] = best_indices[0];
    result[1] = *capture_error;
  }
}

extern "C" __global__ void greedy_argmax_f32(
    const float* logits,
    unsigned long long elements,
    const unsigned int* capture_error,
    unsigned int* result) {
  greedy_argmax_impl<float>(logits, elements, capture_error, result);
}

extern "C" __global__ void greedy_argmax_f16(
    const __half* logits,
    unsigned long long elements,
    const unsigned int* capture_error,
    unsigned int* result) {
  greedy_argmax_impl<__half>(logits, elements, capture_error, result);
}
"#;

pub(crate) fn launch(
    runtime: &CudaRuntime,
    logits: &DeviceBuffer,
    elements: usize,
    dtype: DataType,
    result: &mut DeviceBuffer,
) -> Result<()> {
    if elements == 0 {
        return Err(EpError::KernelFailed(
            "cuda_ep device argmax: logits must not be empty".into(),
        ));
    }
    if elements > u32::MAX as usize {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device argmax: {elements} elements exceed the u32 token-id range"
        )));
    }
    let (entry, elem_size) = match dtype {
        DataType::Float32 => ("greedy_argmax_f32", std::mem::size_of::<f32>()),
        DataType::Float16 => ("greedy_argmax_f16", std::mem::size_of::<u16>()),
        other => {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep device argmax: unsupported logits dtype {other:?}; expected Float32 or Float16"
            )));
        }
    };
    let logits_bytes = elements.checked_mul(elem_size).ok_or_else(|| {
        EpError::KernelFailed("cuda_ep device argmax: logits byte size overflows".into())
    })?;
    if logits_bytes > logits.len() {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device argmax: {elements} values require {logits_bytes} bytes, buffer has {}",
            logits.len()
        )));
    }
    if result.len() < RESULT_BYTES {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device argmax: result buffer has {} bytes, need {RESULT_BYTES}",
            result.len()
        )));
    }
    if logits.device() != result.device() {
        return Err(EpError::KernelFailed(
            "cuda_ep device argmax: logits and result are on different devices".into(),
        ));
    }

    if dtype == DataType::Float16 {
        runtime.require_nvrtc_half_headers("device argmax")?;
    }
    let function = runtime.nvrtc_function("native_device_argmax", SOURCE, entry)?;
    let logits_ptr = cuptr(logits.as_ptr());
    let elements = elements as u64;
    let capture_error_ptr = runtime.capture_error_ptr();
    let result_ptr = cuptr(result.as_mut_ptr());
    let mut builder = runtime.stream().launch_builder(&function);
    builder
        .arg(&logits_ptr)
        .arg(&elements)
        .arg(&capture_error_ptr)
        .arg(&result_ptr);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK
                * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>()) as u32,
        })
    }
    .map(|_| ())
    .map_err(|error| driver_err("launch native device argmax", error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::{EpConfig, ExecutionProvider};

    use crate::CudaExecutionProvider;

    fn gpu() -> Option<CudaExecutionProvider> {
        let mut ep = CudaExecutionProvider::new_default().ok()?;
        ep.initialize(&EpConfig::default()).ok()?;
        Some(ep)
    }

    fn host_argmax(logits: &[f32]) -> u32 {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if max_logit == f32::NEG_INFINITY {
            return 0;
        }
        logits
            .iter()
            .position(|&value| value == max_logit)
            .unwrap_or(0) as u32
    }

    fn run_case(ep: &CudaExecutionProvider, logits: &[f32]) -> [u32; 2] {
        let bytes = logits
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let mut input = ep.allocate(bytes.len(), 256).unwrap();
        let mut output = ep.allocate(RESULT_BYTES, 256).unwrap();
        ep.copy_from_host(&bytes, &mut input).unwrap();
        ep.device_argmax(&input, logits.len(), DataType::Float32, &mut output)
            .unwrap();
        let mut result = [0_u8; RESULT_BYTES];
        ep.copy_to_host(&output, &mut result).unwrap();
        let values = [
            u32::from_ne_bytes(result[..4].try_into().unwrap()),
            u32::from_ne_bytes(result[4..].try_into().unwrap()),
        ];
        ep.deallocate(input).unwrap();
        ep.deallocate(output).unwrap();
        values
    }

    fn run_case_f16(ep: &CudaExecutionProvider, logits: &[f32]) -> [u32; 2] {
        let bytes = logits
            .iter()
            .flat_map(|&value| half::f16::from_f32(value).to_bits().to_ne_bytes())
            .collect::<Vec<_>>();
        let mut input = ep.allocate(bytes.len(), 256).unwrap();
        let mut output = ep.allocate(RESULT_BYTES, 256).unwrap();
        ep.copy_from_host(&bytes, &mut input).unwrap();
        ep.device_argmax(&input, logits.len(), DataType::Float16, &mut output)
            .unwrap();
        let mut result = [0_u8; RESULT_BYTES];
        ep.copy_to_host(&output, &mut result).unwrap();
        let values = [
            u32::from_ne_bytes(result[..4].try_into().unwrap()),
            u32::from_ne_bytes(result[4..].try_into().unwrap()),
        ];
        ep.deallocate(input).unwrap();
        ep.deallocate(output).unwrap();
        values
    }

    #[test]
    fn device_argmax_matches_host_for_random_ties_nan_and_odd_width() {
        let Some(ep) = gpu() else { return };
        let mut seed = 0x1234_5678_u32;
        let mut logits = (0..151_937)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed as i32) as f32 / i32::MAX as f32
            })
            .collect::<Vec<_>>();
        logits[17] = 9.0;
        logits[93_001] = 9.0;
        logits[77] = f32::NAN;
        let result = run_case(&ep, &logits);
        assert_eq!(result, [host_argmax(&logits), 0]);

        let all_non_finite = [f32::NAN, f32::NEG_INFINITY, f32::NAN];
        let result = run_case(&ep, &all_non_finite);
        assert_eq!(result, [host_argmax(&all_non_finite), 0]);

        let capture_error = 0x40_u32;
        unsafe {
            ep.runtime()
                .htod(
                    &capture_error.to_ne_bytes(),
                    ep.runtime().capture_error_ptr(),
                )
                .unwrap();
        }
        let result = run_case(&ep, &[1.0, 5.0, 3.0]);
        assert_eq!(result, [1, capture_error]);
        ep.runtime().reset_capture_error().unwrap();
    }

    #[test]
    fn device_argmax_f16_matches_host_for_finite_and_non_finite() {
        let Some(ep) = gpu() else { return };
        let mut logits = (0..4096)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.25)
            .collect::<Vec<_>>();
        logits[1234] = 9.5;
        logits[77] = f32::NAN;
        // Reference argmax over the fp16-rounded values, matching kernel input.
        let rounded = logits
            .iter()
            .map(|&value| half::f16::from_f32(value).to_f32())
            .collect::<Vec<_>>();
        let result = run_case_f16(&ep, &logits);
        assert_eq!(result, [host_argmax(&rounded), 0]);

        let all_non_finite = [f32::NAN, f32::NEG_INFINITY, f32::NAN];
        let result = run_case_f16(&ep, &all_non_finite);
        assert_eq!(result, [host_argmax(&all_non_finite), 0]);
    }
}
