//! Allocation-free greedy argmax for native CUDA decode.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{DeviceBuffer, EpError, Result};
use onnx_runtime_ir::DataType;

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const VALUES_PER_THREAD: usize = 4;
const MAX_PARTIALS: usize = 256;
const RESULT_BYTES: usize = 2 * std::mem::size_of::<u32>();

pub(crate) fn partial_count(elements: usize) -> usize {
    elements
        .div_ceil(BLOCK as usize * VALUES_PER_THREAD)
        .clamp(1, MAX_PARTIALS)
}

/// Scratch u32 words the device argmax result buffer needs *beyond* its
/// `2 × batch` header words, for `batch` sequences of `elements` logits each.
///
/// The scratch holds, per sequence, `partial_count(elements)` partial values
/// (f32, one u32 word each) followed by `partial_count(elements)` partial
/// indices (u32), i.e. `2 × partial_count` words per sequence. At `batch == 1`
/// this is byte-identical to the previous single-sequence `2 × partial_count`
/// scratch (stage 2b-impl-3, #750).
pub(crate) fn scratch_words(elements: usize, batch: usize) -> usize {
    2 * partial_count(elements) * batch
}

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

__device__ __forceinline__ void argmax_update(
    float candidate,
    unsigned int candidate_index,
    float& best,
    unsigned int& best_index) {
  if (candidate > best ||
      (candidate == best && candidate_index < best_index)) {
    best = candidate;
    best_index = candidate_index;
  }
}

__device__ __forceinline__ void warp_argmax(
    float& best,
    unsigned int& best_index) {
  for (unsigned int offset = 16; offset > 0; offset >>= 1) {
    float candidate = __shfl_down_sync(0xffffffffu, best, offset);
    unsigned int candidate_index =
        __shfl_down_sync(0xffffffffu, best_index, offset);
    argmax_update(candidate, candidate_index, best, best_index);
  }
}

template <typename T>
__device__ __forceinline__ void greedy_argmax_partials_impl(
    const T* logits,
    unsigned long long elements,
    float* partial_values,
    unsigned int* partial_indices) {
  // One sequence per grid row (blockIdx.y); `gridDim.x` blocks cooperatively
  // reduce that sequence's `elements` contiguous logits. Sequence `s` reads
  // `logits[s*elements + ..]` and writes its `gridDim.x` partials at
  // `partial_{values,indices}[s*gridDim.x + ..]`. At batch 1 (gridDim.y == 1)
  // the sequence offset is 0 and the launch is byte-identical to the previous
  // single-sequence kernel (stage 2b-impl-3, #750).
  unsigned long long sequence = blockIdx.y;
  const T* seq_logits = logits + sequence * elements;
  float* seq_values = partial_values + sequence * gridDim.x;
  unsigned int* seq_indices = partial_indices + sequence * gridDim.x;
  float best = -1.0f / 0.0f;
  unsigned int best_index = 0;
  unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long stride =
      static_cast<unsigned long long>(blockDim.x) * gridDim.x;
  for (; i < elements; i += stride) {
    float value = argmax_load<T>(seq_logits[i]);
    if (isnan(value)) continue;
    unsigned int index = static_cast<unsigned int>(i);
    argmax_update(value, index, best, best_index);
  }

  warp_argmax(best, best_index);
  __shared__ float warp_values[32];
  __shared__ unsigned int warp_indices[32];
  unsigned int lane = threadIdx.x & 31;
  unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_values[warp] = best;
    warp_indices[warp] = best_index;
  }
  __syncthreads();

  if (warp == 0) {
    unsigned int warp_count = (blockDim.x + 31) >> 5;
    best = lane < warp_count ? warp_values[lane] : -1.0f / 0.0f;
    best_index = lane < warp_count ? warp_indices[lane] : 0;
    warp_argmax(best, best_index);
    if (lane == 0) {
      seq_values[blockIdx.x] = best;
      seq_indices[blockIdx.x] = best_index;
    }
  }
}

extern "C" __global__ void greedy_argmax_partials_f32(
    const float* logits,
    unsigned long long elements,
    float* partial_values,
    unsigned int* partial_indices) {
  greedy_argmax_partials_impl<float>(
      logits, elements, partial_values, partial_indices);
}

extern "C" __global__ void greedy_argmax_partials_f16(
    const __half* logits,
    unsigned long long elements,
    float* partial_values,
    unsigned int* partial_indices) {
  greedy_argmax_partials_impl<__half>(
      logits, elements, partial_values, partial_indices);
}

extern "C" __global__ void greedy_argmax_finalize(
    const float* partial_values,
    const unsigned int* partial_indices,
    unsigned int partial_count,
    const unsigned int* capture_error,
    unsigned int* result) {
  // One block per sequence (blockIdx.x). Sequence `s` reduces its own
  // `partial_count` partials at `partial_{values,indices}[s*partial_count + ..]`
  // and writes its token id / capture-error pair at `result[2*s .. 2*s+2]`. At
  // batch 1 (gridDim.x == 1) this writes result[0]/result[1] from the base
  // partial region, byte-identical to the previous single-sequence finalize
  // (stage 2b-impl-3, #750).
  unsigned int sequence = blockIdx.x;
  const float* seq_values = partial_values + sequence * partial_count;
  const unsigned int* seq_indices = partial_indices + sequence * partial_count;
  float best = -1.0f / 0.0f;
  unsigned int best_index = 0;
  for (unsigned int i = threadIdx.x; i < partial_count; i += blockDim.x) {
    argmax_update(seq_values[i], seq_indices[i], best, best_index);
  }
  warp_argmax(best, best_index);
  __shared__ float warp_values[32];
  __shared__ unsigned int warp_indices[32];
  unsigned int lane = threadIdx.x & 31;
  unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_values[warp] = best;
    warp_indices[warp] = best_index;
  }
  __syncthreads();
  if (warp == 0) {
    unsigned int warp_count = (blockDim.x + 31) >> 5;
    best = lane < warp_count ? warp_values[lane] : -1.0f / 0.0f;
    best_index = lane < warp_count ? warp_indices[lane] : 0;
    warp_argmax(best, best_index);
    if (lane == 0) {
      result[2 * sequence] = best_index;
      result[2 * sequence + 1] = *capture_error;
    }
  }
}
"#;

pub(crate) fn launch(
    runtime: &CudaRuntime,
    logits: &DeviceBuffer,
    elements: usize,
    batch: usize,
    dtype: DataType,
    result: &mut DeviceBuffer,
) -> Result<()> {
    if elements == 0 {
        return Err(EpError::KernelFailed(
            "cuda_ep device argmax: logits must not be empty".into(),
        ));
    }
    if batch == 0 {
        return Err(EpError::KernelFailed(
            "cuda_ep device argmax: batch must not be zero".into(),
        ));
    }
    if elements > u32::MAX as usize {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device argmax: {elements} elements exceed the u32 token-id range"
        )));
    }
    // The grid dedicates one row (blockIdx.y for partials, blockIdx.x for
    // finalize) to each of the `batch` sequences, so the batch count must fit
    // the u32 grid dimension. At batch 1 the grid is 1-deep and byte-identical
    // to the previous single-sequence launch (stage 2b-impl-3, #750).
    if batch > u32::MAX as usize {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device argmax: batch {batch} exceeds the u32 grid range"
        )));
    }
    let (entry, elem_size) = match dtype {
        DataType::Float32 => ("greedy_argmax_partials_f32", std::mem::size_of::<f32>()),
        DataType::Float16 => ("greedy_argmax_partials_f16", std::mem::size_of::<u16>()),
        other => {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep device argmax: unsupported logits dtype {other:?}; expected Float32 or Float16"
            )));
        }
    };
    let logits_bytes = elements
        .checked_mul(elem_size)
        .and_then(|per_seq| per_seq.checked_mul(batch))
        .ok_or_else(|| {
            EpError::KernelFailed("cuda_ep device argmax: logits byte size overflows".into())
        })?;
    if logits_bytes > logits.len() {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device argmax: {batch}×{elements} values require {logits_bytes} bytes, buffer has {}",
            logits.len()
        )));
    }
    let partial_count = partial_count(elements);
    // Header is `2 × batch` u32 words (a token id + capture-error pair per
    // sequence, laid out contiguously so sequence `s` owns `result[2*s..2*s+2]`),
    // followed by `batch × 2 × partial_count` scratch words. At batch 1 this is
    // `RESULT_BYTES` header + the previous single-sequence scratch — byte-identical.
    let header_bytes = batch.checked_mul(RESULT_BYTES).ok_or_else(|| {
        EpError::KernelFailed("cuda_ep device argmax: result header size overflows".into())
    })?;
    let required_result_bytes = header_bytes
        + scratch_words(elements, batch)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                EpError::KernelFailed("cuda_ep device argmax: scratch byte size overflows".into())
            })?;
    if result.len() < required_result_bytes {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep device argmax: result buffer has {} bytes, need {required_result_bytes}",
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
    let partial_function = runtime.nvrtc_function("native_device_argmax", SOURCE, entry)?;
    let final_function =
        runtime.nvrtc_function("native_device_argmax", SOURCE, "greedy_argmax_finalize")?;
    let logits_ptr = cuptr(logits.as_ptr());
    let elements = elements as u64;
    // Scratch begins after the `batch`-wide header. Per-sequence partials are
    // laid out values-then-indices, `batch × partial_count` of each.
    let scratch_ptr = unsafe { result.as_mut_ptr().add(header_bytes) };
    let partial_values_ptr = cuptr(scratch_ptr);
    let partial_indices_ptr =
        cuptr(unsafe { scratch_ptr.add(batch * partial_count * std::mem::size_of::<f32>()) });
    let capture_error_ptr = runtime.capture_error_ptr();
    let result_ptr = cuptr(result.as_mut_ptr());
    let mut builder = runtime.stream().launch_builder(&partial_function);
    builder
        .arg(&logits_ptr)
        .arg(&elements)
        .arg(&partial_values_ptr)
        .arg(&partial_indices_ptr);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (partial_count as u32, batch as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|error| driver_err("launch native device argmax partials", error))?;

    let partial_count = partial_count as u32;
    let mut builder = runtime.stream().launch_builder(&final_function);
    builder
        .arg(&partial_values_ptr)
        .arg(&partial_indices_ptr)
        .arg(&partial_count)
        .arg(&capture_error_ptr)
        .arg(&result_ptr);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (batch as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| driver_err("launch native device argmax finalize", error))
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

    fn result_bytes(elements: usize, batch: usize) -> usize {
        (batch * RESULT_BYTES) + scratch_words(elements, batch) * std::mem::size_of::<u32>()
    }

    fn run_case(ep: &CudaExecutionProvider, logits: &[f32]) -> [u32; 2] {
        let bytes = logits
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let mut input = ep.allocate(bytes.len(), 256).unwrap();
        let mut output = ep.allocate(result_bytes(logits.len(), 1), 256).unwrap();
        ep.copy_from_host(&bytes, &mut input).unwrap();
        ep.device_argmax(&input, logits.len(), 1, DataType::Float32, &mut output)
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

    /// Run the argmax over `batch` sequences of `vocab` f32 logits laid out
    /// row-major (`rows.concat()`), returning the `(token_id, capture_error)`
    /// pair per sequence read back from the `2 × batch`-word header.
    fn run_batch_case(ep: &CudaExecutionProvider, rows: &[Vec<f32>]) -> Vec<[u32; 2]> {
        let batch = rows.len();
        let vocab = rows[0].len();
        assert!(rows.iter().all(|row| row.len() == vocab));
        let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
        let bytes = flat
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let mut input = ep.allocate(bytes.len(), 256).unwrap();
        let mut output = ep.allocate(result_bytes(vocab, batch), 256).unwrap();
        ep.copy_from_host(&bytes, &mut input).unwrap();
        ep.device_argmax(&input, vocab, batch, DataType::Float32, &mut output)
            .unwrap();
        let mut header = vec![0_u8; batch * RESULT_BYTES];
        ep.copy_to_host(&output, &mut header).unwrap();
        let out = (0..batch)
            .map(|s| {
                let base = s * RESULT_BYTES;
                [
                    u32::from_ne_bytes(header[base..base + 4].try_into().unwrap()),
                    u32::from_ne_bytes(header[base + 4..base + 8].try_into().unwrap()),
                ]
            })
            .collect();
        ep.deallocate(input).unwrap();
        ep.deallocate(output).unwrap();
        out
    }

    fn run_case_f16(ep: &CudaExecutionProvider, logits: &[f32]) -> [u32; 2] {
        let bytes = logits
            .iter()
            .flat_map(|&value| half::f16::from_f32(value).to_bits().to_ne_bytes())
            .collect::<Vec<_>>();
        let mut input = ep.allocate(bytes.len(), 256).unwrap();
        let mut output = ep.allocate(result_bytes(logits.len(), 1), 256).unwrap();
        ep.copy_from_host(&bytes, &mut input).unwrap();
        ep.device_argmax(&input, logits.len(), 1, DataType::Float16, &mut output)
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
    fn device_argmax_matches_host_for_151936_random_ties_and_nan() {
        let Some(ep) = gpu() else { return };
        let mut seed = 0x1234_5678_u32;
        let mut logits = (0..151_936)
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
    fn device_argmax_f16_matches_host_for_151936_ties_and_nan() {
        let Some(ep) = gpu() else { return };
        let mut logits = (0..151_936)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.25)
            .collect::<Vec<_>>();
        logits[1234] = 9.5;
        logits[130_001] = 9.5;
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

    #[test]
    fn device_argmax_batch_selects_per_row_max_with_deliberate_ties() {
        let Some(ep) = gpu() else { return };
        // A control that would falsify a shared reduction buffer or a wrong
        // per-sequence stride: the maximum sits at a *different* position in
        // every row, each row is larger than one block's grid-stride reach so
        // multiple partials are reduced, and two rows carry a deliberate tie so
        // the lowest-index tie-break is exercised per sequence rather than
        // globally (stage 2b-impl-3, #750).
        let vocab = 4096;
        let peaks = [17_usize, 4000, 128, 2049];
        let mut rows: Vec<Vec<f32>> = peaks
            .iter()
            .enumerate()
            .map(|(row, &peak)| {
                let mut logits = (0..vocab)
                    .map(|i| ((i + row) % 13) as f32 * 0.1)
                    .collect::<Vec<_>>();
                logits[peak] = 9.0;
                logits
            })
            .collect();
        // Row 1 gets a tie at an *earlier* index than its peak; the lower index
        // must win, proving each sequence tie-breaks against its own partials.
        rows[1][2500] = 9.0; // peak was 4000 → lower index 2500 must win
        rows[3][900] = 9.0; // peak was 2049 → lower index 900 must win
        let expected = [17_u32, 2500, 128, 900];

        let batched = run_batch_case(&ep, &rows);
        assert_eq!(batched.len(), rows.len());
        for (row, (result, &want)) in batched.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                result[0], want,
                "batched argmax row {row} selected token {} not {want}",
                result[0]
            );
            assert_eq!(result[1], 0, "row {row} unexpected capture-error flag");
            // Each batched row must equal a standalone single-sequence argmax of
            // the same logits — the byte-identity contract, row by row.
            let solo = run_case(&ep, &rows[row]);
            assert_eq!(
                *result, solo,
                "batched row {row} diverged from the single-sequence argmax"
            );
        }
    }
}
