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
  // Ties resolve to the LOWEST index, matching the canonical ONNX ArgMax
  // operator (select_last_index=false keeps the first extremal element) and the
  // host greedy references `sample_greedy` / `argmax_logits_tensor`, which keep
  // the lowest token id on ties. This is the base-decode / ORT byte-identity
  // contract. The comparison is on the global index value, so this rule holds
  // identically across the warp, block, and finalize reductions regardless of
  // lane origin (strict-improve on value, else keep the lower global index).
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
        // Reference for the device kernel, mirroring `argmax_update` exactly:
        // skip NaN, keep a running max starting at -inf, and update only on a
        // STRICT increase. Strict `>` means the first occurrence of the maximum
        // wins and later equal values never displace it — i.e. ties resolve to
        // the LOWEST index, matching the canonical ONNX ArgMax operator
        // (select_last_index=false) and the host greedy references `sample_greedy`
        // / `argmax_logits_tensor` ("ties keep the lowest token id"). An all-NaN
        // or all-(-inf) row leaves the sentinel index 0, exactly as the kernel
        // (whose accumulator initialises to (-inf, 0) and never updates).
        let mut best = f32::NEG_INFINITY;
        let mut best_index = 0u32;
        for (index, &value) in logits.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            if value > best {
                best = value;
                best_index = index as u32;
            }
        }
        best_index
    }

    fn result_bytes(elements: usize, batch: usize) -> usize {
        (batch * RESULT_BYTES) + scratch_words(elements, batch) * std::mem::size_of::<u32>()
    }

    /// Force the freshly written input buffer to be coherent before the argmax
    /// kernel reads it. In production the logits are produced on-device by the
    /// lm_head GEMV on the same compute stream, so they are already ordered and
    /// coherent. Constructed-tie tests instead feed logits via a host->device
    /// copy into pooled VMM memory; the first kernel access after a buffer-state
    /// change can otherwise observe a stale tail element (a memory-governor VMM
    /// materialization artifact, not the reduction). A device->host readback of
    /// the input synchronizes the stream and materializes those pages, giving the
    /// argmax the same coherent view production always sees.
    fn ensure_input_coherent(ep: &CudaExecutionProvider, input: &DeviceBuffer, bytes: usize) {
        let mut scratch = vec![0_u8; bytes];
        ep.copy_to_host(input, &mut scratch).unwrap();
    }

    fn run_case(ep: &CudaExecutionProvider, logits: &[f32]) -> [u32; 2] {
        let bytes = logits
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let mut input = ep.allocate(bytes.len(), 256).unwrap();
        let mut output = ep.allocate(result_bytes(logits.len(), 1), 256).unwrap();
        ep.copy_from_host(&bytes, &mut input).unwrap();
        ensure_input_coherent(ep, &input, bytes.len());
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
        ensure_input_coherent(ep, &input, bytes.len());
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
        ensure_input_coherent(ep, &input, bytes.len());
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

    /// Batched f16 sibling of [`run_batch_case`]: rounds every row to fp16, runs
    /// the whole `batch` in one launch, and reads back the per-row header. Used
    /// by the tie-break gate to exercise the f16 partials kernel over M rows.
    fn run_batch_case_f16(ep: &CudaExecutionProvider, rows: &[Vec<f32>]) -> Vec<[u32; 2]> {
        let batch = rows.len();
        let vocab = rows[0].len();
        assert!(rows.iter().all(|row| row.len() == vocab));
        let bytes = rows
            .iter()
            .flatten()
            .flat_map(|&value| half::f16::from_f32(value).to_bits().to_ne_bytes())
            .collect::<Vec<_>>();
        let mut input = ep.allocate(bytes.len(), 256).unwrap();
        let mut output = ep.allocate(result_bytes(vocab, batch), 256).unwrap();
        ep.copy_from_host(&bytes, &mut input).unwrap();
        ensure_input_coherent(ep, &input, bytes.len());
        ep.device_argmax(&input, vocab, batch, DataType::Float16, &mut output)
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
        // the LOWEST-index tie-break is exercised per sequence rather than
        // globally.
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
        // Row 1 gets a tie at an *earlier* index than its peak; the LOWER index
        // must win, proving each sequence tie-breaks against its own partials.
        rows[1][2500] = 9.0; // tie {2500, 4000} → lower index 2500 must win
        rows[3][900] = 9.0; // tie {900, 2049} → lower index 900 must win
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

    /// The decisive byte-identity gate for the lowest-index tie-break: the
    /// device argmax must equal the host lowest-index argmax (canonical ONNX
    /// ArgMax / `sample_greedy`) for every row of M∈{1,4,6,8} batches at a
    /// realistic vocab (152064), across (a) random fp16-representable logits and
    /// (b) rows with DELIBERATELY CONSTRUCTED TIES (multiple equal maxima at
    /// different indices) where the lowest tied index must win. A single row that
    /// resolved a tie to a higher index fails this test.
    ///
    /// The M rows are driven through the batched launch (one sequence per grid
    /// row) so each row is an independent argmax; the M=1 batch is the
    /// single-sequence path. Verification is against the host reference only —
    /// the batched launch is the argmax contract under test and, unlike a tight
    /// allocate/free `run_case` loop, it does not recycle a fresh per-call result
    /// buffer through the pooled allocator on every row.
    #[test]
    fn device_argmax_lowest_index_tiebreak_matches_host_for_vocab_152064() {
        let Some(ep) = gpu() else { return };
        const VOCAB: usize = 152_064;

        for &m in &[1_usize, 4, 6, 8] {
            // (a) Random rows, one distinct seed per row so a shared reduction
            // buffer or wrong per-row stride is caught. Values are quantised to
            // fp16 so the f32 and f16 kernels agree and ties occur naturally.
            let random_rows: Vec<Vec<f32>> = (0..m)
                .map(|row| {
                    let mut seed = 0x9E37_79B9_u32
                        .wrapping_mul(row as u32 + 1)
                        .wrapping_add(0x1234_5);
                    (0..VOCAB)
                        .map(|_| {
                            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            let unit = (seed >> 8) as f32 / (1_u32 << 24) as f32; // [0,1)
                            half::f16::from_f32(unit * 8.0 - 4.0).to_f32()
                        })
                        .collect()
                })
                .collect();
            let random_f32 = run_batch_case(&ep, &random_rows);
            let random_f16 = run_batch_case_f16(&ep, &random_rows);
            for (row, (f32_got, f16_got)) in random_f32.iter().zip(random_f16.iter()).enumerate() {
                let want = host_argmax(&random_rows[row]);
                assert_eq!(
                    f32_got[0], want,
                    "M={m} random row {row}: f32 device argmax {} != host lowest-index {want}",
                    f32_got[0]
                );
                assert_eq!(f32_got[1], 0, "M={m} random row {row}: unexpected capture-error");
                // The values are fp16-exact, so the f16 kernel must agree exactly.
                assert_eq!(
                    f16_got[0], want,
                    "M={m} random row {row}: f16 device argmax {} != host lowest-index {want}",
                    f16_got[0]
                );
            }

            // (b) Constructed ties: a flat low background with three EQUAL maxima
            // at ascending indices; the LOWEST must win. The tie indices differ
            // per row (and land in different grid-stride partials) so a globally
            // shared or highest-index reduction is falsified. Exercised on both
            // the f32 and f16 partials kernels.
            let tie_rows: Vec<Vec<f32>> = (0..m)
                .map(|row| {
                    let mut logits = vec![-1.0_f32; VOCAB];
                    // Spread the three ties across the vocab, offset per row.
                    let p0 = 3 + row * 7; // lowest → must be selected
                    let p1 = VOCAB / 2 + row * 101;
                    let p2 = VOCAB - 1 - row * 53;
                    for &p in &[p0, p1, p2] {
                        logits[p] = 5.0;
                    }
                    logits
                })
                .collect();
            let tie_expected: Vec<u32> = (0..m).map(|row| (3 + row * 7) as u32).collect();
            let tie_f32 = run_batch_case(&ep, &tie_rows);
            let tie_f16 = run_batch_case_f16(&ep, &tie_rows);
            for (row, (f32_got, f16_got)) in tie_f32.iter().zip(tie_f16.iter()).enumerate() {
                let want = tie_expected[row];
                assert_eq!(
                    f32_got[0], want,
                    "M={m} tie row {row}: f32 device argmax {} != lowest tied index {want}",
                    f32_got[0]
                );
                // The host reference must agree, closing the byte-identity loop.
                assert_eq!(
                    want,
                    host_argmax(&tie_rows[row]),
                    "M={m} tie row {row}: host lowest-index disagrees with expected"
                );
                assert_eq!(
                    f16_got[0], want,
                    "M={m} tie row {row}: f16 device argmax {} != lowest tied index {want}",
                    f16_got[0]
                );
            }

            // (c) Explicit partition-spanning ties. The partials kernel splits the
            // vocab into grid-stride partitions of width `stride = 256 * ceil(VOCAB
            // / 1024)`; a tie whose equal maxima land in DIFFERENT partitions is
            // only resolved correctly if every reduction stage (warp -> block ->
            // partition-partial -> finalize) breaks value ties toward the lower
            // GLOBAL index. These fixed configs place equal maxima straddling the
            // partition seams and spanning up to three partitions, including
            // Wallace's exact {3, 76032, 152063} case at vocab 152064 (→ 3). The
            // lowest global index must win in every case. One config per row
            // (cycled) so each batch exercises several boundary layouts.
            const STRIDE: usize = 256 * ((VOCAB + 1023) / 1024); // 38_144 for 152_064
            let boundary_configs: [[usize; 3]; 6] = [
                [3, VOCAB / 2, VOCAB - 1],           // Wallace's exact {3, 76032, 152063} → 3
                [STRIDE - 1, STRIDE, STRIDE + 1],    // straddles the first seam → STRIDE-1
                [3, STRIDE, 2 * STRIDE],             // 3-way across three partitions → 3
                [2 * STRIDE - 1, 2 * STRIDE, 2 * STRIDE + 1], // straddles the second seam
                [0, VOCAB / 3, VOCAB - 2],           // index 0 is the lowest
                [3 * STRIDE - 1, 3 * STRIDE, VOCAB - 1], // last seam + final index
            ];
            let boundary_rows: Vec<Vec<f32>> = (0..m)
                .map(|row| {
                    let cfg = boundary_configs[row % boundary_configs.len()];
                    let mut logits = vec![-2.0_f32; VOCAB];
                    for &p in &cfg {
                        logits[p] = 7.0;
                    }
                    logits
                })
                .collect();
            let boundary_expected: Vec<u32> = (0..m)
                .map(|row| {
                    let cfg = boundary_configs[row % boundary_configs.len()];
                    *cfg.iter().min().unwrap() as u32
                })
                .collect();
            let boundary_f32 = run_batch_case(&ep, &boundary_rows);
            let boundary_f16 = run_batch_case_f16(&ep, &boundary_rows);
            for (row, (f32_got, f16_got)) in boundary_f32.iter().zip(boundary_f16.iter()).enumerate()
            {
                let want = boundary_expected[row];
                assert_eq!(
                    f32_got[0], want,
                    "M={m} boundary row {row}: f32 device argmax {} != lowest global tied index {want} \
                     (cross-partition finalize must keep the lower index)",
                    f32_got[0]
                );
                assert_eq!(
                    want,
                    host_argmax(&boundary_rows[row]),
                    "M={m} boundary row {row}: host lowest-index disagrees with expected"
                );
                assert_eq!(
                    f16_got[0], want,
                    "M={m} boundary row {row}: f16 device argmax {} != lowest global tied index {want}",
                    f16_got[0]
                );
            }
        }
    }

    /// Base-decode byte-identity contract: on constructed ties the device argmax
    /// must select the LOWEST index, matching the host greedy sampler
    /// (`sample_greedy` / `argmax_logits_tensor`, "ties keep the lowest token
    /// id") and the canonical ONNX ArgMax operator (select_last_index=false).
    /// Covers M∈{1,4,6,8} and both the f32 and f16 partials kernels.
    #[test]
    fn device_argmax_matches_host_greedy_lowest_index_on_ties() {
        let Some(ep) = gpu() else { return };
        let vocab = 8192;
        for &m in &[1_usize, 4, 6, 8] {
            let rows: Vec<Vec<f32>> = (0..m)
                .map(|row| {
                    let mut logits = vec![0.25_f32; vocab];
                    // Three equal maxima per row at distinct indices; the lowest
                    // (`lo`) must win. Offsets per row so a shared/globally wrong
                    // reduction is falsified.
                    let lo = 5 + row * 3; // lowest → must be selected
                    let mid = vocab / 2 + row * 7;
                    let hi = vocab - 1 - row * 11;
                    for &p in &[lo, mid, hi] {
                        logits[p] = 3.0;
                    }
                    logits
                })
                .collect();
            let want: Vec<u32> = (0..m).map(|row| (5 + row * 3) as u32).collect();
            let f32_out = run_batch_case(&ep, &rows);
            let f16_out = run_batch_case_f16(&ep, &rows);
            for (row, (g32, g16)) in f32_out.iter().zip(f16_out.iter()).enumerate() {
                // The device kernel, the host greedy reference, and the expected
                // lowest index must all agree.
                assert_eq!(host_argmax(&rows[row]), want[row], "host greedy row {row}");
                assert_eq!(
                    g32[0], want[row],
                    "M={m} row {row}: f32 device argmax {} != host-greedy lowest index {}",
                    g32[0], want[row]
                );
                assert_eq!(
                    g16[0], want[row],
                    "M={m} row {row}: f16 device argmax {} != host-greedy lowest index {}",
                    g16[0], want[row]
                );
            }
        }
    }
}
