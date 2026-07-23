//! Capture-safe, split-K single-token (`Sq=1`) f32 GQA decode attention.
//!
//! The reference kernel `gqa_attention_reference_f32` computes decode attention
//! serially on thread 0 (every QK dot + `exp`) and then has all threads rescan
//! the full score row for the softmax denominator. A single warp per query row
//! fixes that serialization but still leaves small-head GQA models with only a
//! handful of resident CTAs while every warp walks the complete context. This
//! implementation mirrors the fp16 split-K kernel: up to sixteen CTAs divide
//! each query row's KV range, then a second kernel merges their online-softmax
//! states. That exposes enough parallelism to keep decode latency nearly flat
//! over practical context lengths without changing the graph contract.
//!
//! ## Capture-safety
//!
//! The launch path is legal to record inside a CUDA graph and to replay with
//! only device-buffer contents changing:
//!   * No `stream.synchronize()` or any device sync on the launch path.
//!   * No per-call `cudaMalloc`/`cudaFree`; partial states use fixed module-global
//!     scratch allocated when NVRTC loads the module.
//!   * Fixed maximum-split launch geometry. The device-resident valid length
//!     chooses the active split count, so graph replay needs no update.

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Result};

use crate::error::driver_err;
use crate::runtime::CudaRuntime;

const MODULE_KEY: &str = "gqa_decode_attention_f32_v2";
const ENTRY: &str = "gqa_decode_attention_f32";
const MERGE_ENTRY: &str = "gqa_decode_attention_f32_merge";

/// Largest `head_dim` this kernel supports. Each of the 32 warp lanes owns
/// `ceil(head_dim / 32)` output dimensions in registers, capped at 4.
pub(super) const MAX_HEAD_DIM: usize = 128;

/// Warps grouped into one CTA. Small enough to spread the (few) decode rows
/// across many SMs, large enough to amortize launch overhead.
const WARPS_PER_BLOCK: u32 = 4;
const WARP_SIZE: u32 = 32;
pub(super) const MAX_SPLITS: usize = 16;

/// Whether the warp-parallel decode kernel handles this shape. Single query
/// token (`Sq=1`) with `head_dim` within [`MAX_HEAD_DIM`].
pub(super) fn supported(query_seq: usize, head_dim: usize) -> bool {
    query_seq == 1 && (1..=MAX_HEAD_DIM).contains(&head_dim)
}

const DECODE_SRC: &str = r#"
#define GQA_WARP_SIZE 32
#define GQA_MAX_DPL 4   // dims per lane; head_dim <= 32 * GQA_MAX_DPL == 128
#define GQA_MAX_HEAD_SIZE 128
#define GQA_MAX_SPLITS 16
#define GQA_MAX_SCRATCH_ROWS 256
#define GQA_SCRATCH_STRIDE (GQA_MAX_HEAD_SIZE + 2)

__device__ __align__(16) float gqa_split_scratch[
    GQA_MAX_SCRATCH_ROWS * GQA_MAX_SPLITS * GQA_SCRATCH_STRIDE];

__device__ __forceinline__ int gqa_active_splits(const int sequence_length) {
    if (sequence_length <= 64) return 1;
    if (sequence_length <= 128) return 2;
    if (sequence_length <= 256) return 4;
    if (sequence_length <= 512) return 8;
    return GQA_MAX_SPLITS;
}

extern "C" __global__ void gqa_decode_attention_f32(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    const int* __restrict__ total_lengths,
    const int batch,
    const int query_heads,
    const int kv_heads,
    const int query_seq,
    const int head_size,
    const int cache_capacity,
    const int group_size,
    const float scale,
    const int local_window,
    const float softcap)
{
    extern __shared__ float smem[];
    const int warps_per_block = blockDim.x / GQA_WARP_SIZE;
    float* warp_max = smem;
    float* warp_sum = warp_max + warps_per_block;
    float* warp_acc = warp_sum + warps_per_block;

    const int lane = threadIdx.x % GQA_WARP_SIZE;
    const int warp_in_block = threadIdx.x / GQA_WARP_SIZE;
    const int row = blockIdx.x / GQA_MAX_SPLITS;
    const int split = blockIdx.x % GQA_MAX_SPLITS;
    const int rows = batch * query_heads * query_seq;
    if (row >= rows) return;

    const int query_pos = row % query_seq;
    const int query_head = (row / query_seq) % query_heads;
    const int batch_index = row / (query_heads * query_seq);
    const int kv_head = query_head / group_size;

    const int total = total_lengths[batch_index];
    const int causal_limit = total - query_seq + query_pos;
    const int local_start =
        (local_window > 0 && causal_limit + 1 > local_window)
            ? causal_limit + 1 - local_window
            : 0;
    const int sequence_length = max(0, causal_limit + 1 - local_start);
    const int active_splits =
        (row < GQA_MAX_SCRATCH_ROWS) ? gqa_active_splits(sequence_length) : 1;
    if (split >= active_splits) return;
    const int keys_per_split =
        (sequence_length + active_splits - 1) / active_splits;
    const int split_start = local_start + split * keys_per_split;
    const int split_end = min(causal_limit + 1, split_start + keys_per_split);

    const long q_base =
        ((long)(batch_index * query_heads + query_head) * query_seq + query_pos)
            * (long)head_size;
    const long kv_plane =
        (long)(batch_index * kv_heads + kv_head) * (long)cache_capacity * (long)head_size;

    float q_reg[GQA_MAX_DPL];
    float acc[GQA_MAX_DPL];
#pragma unroll
    for (int i = 0; i < GQA_MAX_DPL; ++i) {
        const int d = lane + i * GQA_WARP_SIZE;
        q_reg[i] = (d < head_size) ? query[q_base + d] : 0.0f;
        acc[i] = 0.0f;
    }

    const float negative_infinity = __int_as_float(0xff800000);
    float running_max = negative_infinity;
    float running_sum = 0.0f;

    for (int key_pos = split_start + warp_in_block; key_pos < split_end;
         key_pos += warps_per_block) {
        const long k_base = kv_plane + (long)key_pos * (long)head_size;
        float partial = 0.0f;
#pragma unroll
        for (int i = 0; i < GQA_MAX_DPL; ++i) {
            const int d = lane + i * GQA_WARP_SIZE;
            if (d < head_size) {
                partial += q_reg[i] * key[k_base + d];
            }
        }
        // Butterfly all-reduce: every lane ends with the full QK dot product.
#pragma unroll
        for (int offset = GQA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xffffffffu, partial, offset);
        }
        float score = partial * scale;
        if (softcap != 0.0f) {
            score = softcap * tanhf(score / softcap);
        }

        const float new_max = fmaxf(running_max, score);
        const float correction = expf(running_max - new_max);
        const float probability = expf(score - new_max);
        running_sum = running_sum * correction + probability;
#pragma unroll
        for (int i = 0; i < GQA_MAX_DPL; ++i) {
            const int d = lane + i * GQA_WARP_SIZE;
            const float v = (d < head_size) ? value[k_base + d] : 0.0f;
            acc[i] = acc[i] * correction + probability * v;
        }
        running_max = new_max;
    }

    if (lane == 0) {
        warp_max[warp_in_block] = running_max;
        warp_sum[warp_in_block] = running_sum;
    }
#pragma unroll
    for (int i = 0; i < GQA_MAX_DPL; ++i) {
        const int d = lane + i * GQA_WARP_SIZE;
        if (d < head_size) {
            warp_acc[warp_in_block * head_size + d] = acc[i];
        }
    }
    __syncthreads();

    if (warp_in_block != 0) return;
    float global_max = negative_infinity;
    for (int w = 0; w < warps_per_block; ++w) {
        global_max = fmaxf(global_max, warp_max[w]);
    }
    float denom = 0.0f;
    for (int w = 0; w < warps_per_block; ++w) {
        denom += warp_sum[w] * expf(warp_max[w] - global_max);
    }
    const bool direct_output = row >= GQA_MAX_SCRATCH_ROWS;
    float* split_state = direct_output
        ? nullptr
        : gqa_split_scratch
            + (row * GQA_MAX_SPLITS + split) * GQA_SCRATCH_STRIDE;
    if (lane == 0 && !direct_output) {
        split_state[0] = global_max;
        split_state[1] = denom;
    }
    const float inverse_sum =
        (direct_output && denom > 0.0f) ? (1.0f / denom) : 0.0f;
#pragma unroll
    for (int i = 0; i < GQA_MAX_DPL; ++i) {
        const int d = lane + i * GQA_WARP_SIZE;
        if (d < head_size) {
            float out = 0.0f;
            for (int w = 0; w < warps_per_block; ++w) {
                out += warp_acc[w * head_size + d]
                    * expf(warp_max[w] - global_max);
            }
            if (direct_output) {
                output[q_base + d] = out * inverse_sum;
            } else {
                split_state[2 + d] = out;
            }
        }
    }
}

extern "C" __global__ void gqa_decode_attention_f32_merge(
    float* __restrict__ output,
    const int* __restrict__ total_lengths,
    const int batch,
    const int query_heads,
    const int query_seq,
    const int head_size,
    const int local_window)
{
    const int row = blockIdx.x;
    const int rows = batch * query_heads * query_seq;
    if (row >= rows || row >= GQA_MAX_SCRATCH_ROWS) return;

    const int lane = threadIdx.x;
    const int query_pos = row % query_seq;
    const int batch_index = row / (query_heads * query_seq);
    const int total = total_lengths[batch_index];
    const int causal_limit = total - query_seq + query_pos;
    const int local_start =
        (local_window > 0 && causal_limit + 1 > local_window)
            ? causal_limit + 1 - local_window
            : 0;
    const int sequence_length = max(0, causal_limit + 1 - local_start);
    const int active_splits = gqa_active_splits(sequence_length);

    float global_max = __int_as_float(0xff800000);
    for (int split = 0; split < active_splits; ++split) {
        const float* state = gqa_split_scratch
            + (row * GQA_MAX_SPLITS + split) * GQA_SCRATCH_STRIDE;
        global_max = fmaxf(global_max, state[0]);
    }
    float denom = 0.0f;
    for (int split = 0; split < active_splits; ++split) {
        const float* state = gqa_split_scratch
            + (row * GQA_MAX_SPLITS + split) * GQA_SCRATCH_STRIDE;
        denom += state[1] * expf(state[0] - global_max);
    }
    const float inverse_sum = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    const long q_base = (long)row * (long)head_size;
#pragma unroll
    for (int i = 0; i < GQA_MAX_DPL; ++i) {
        const int d = lane + i * GQA_WARP_SIZE;
        if (d < head_size) {
            float out = 0.0f;
            for (int split = 0; split < active_splits; ++split) {
                const float* state = gqa_split_scratch
                    + (row * GQA_MAX_SPLITS + split) * GQA_SCRATCH_STRIDE;
                out += state[2 + d] * expf(state[0] - global_max);
            }
            output[q_base + d] = out * inverse_sum;
        }
    }
}
"#;

/// Launch the capture-safe warp-parallel decode kernel.
///
/// Present K/V live in `[batch, kv_heads, cache_capacity, head_dim]` f32 with
/// RoPE already applied to stored keys; `query`/`output` are BNSH with
/// `query_seq == 1`. The valid length per batch is read on the device from
/// `total_lengths` (never from `cache_capacity`).
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    runtime: &CudaRuntime,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    query_seq: usize,
    head_dim: usize,
    cache_capacity: usize,
    group: usize,
    scale: f32,
    query: CUdeviceptr,
    key: CUdeviceptr,
    value: CUdeviceptr,
    output: CUdeviceptr,
    total_lengths: CUdeviceptr,
    local_window: i32,
    softcap: f32,
) -> Result<()> {
    let as_i32 = |name: &str, value: usize| {
        i32::try_from(value).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep GQA decode: {name} {value} exceeds i32"))
        })
    };
    let batch_i = as_i32("batch", batch)?;
    let heads_i = as_i32("num_heads", num_heads)?;
    let kv_heads_i = as_i32("num_kv_heads", num_kv_heads)?;
    let query_seq_i = as_i32("query_seq", query_seq)?;
    let dim_i = as_i32("head_dim", head_dim)?;
    let capacity_i = as_i32("cache_capacity", cache_capacity)?;
    let group_i = as_i32("GQA group", group)?;

    let rows = batch
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(query_seq))
        .ok_or_else(|| EpError::KernelFailed("cuda_ep GQA decode: row count overflow".into()))?;
    let partial_blocks = rows
        .checked_mul(MAX_SPLITS)
        .ok_or_else(|| EpError::KernelFailed("cuda_ep GQA decode: split grid overflow".into()))?;
    let grid_x = u32::try_from(partial_blocks.max(1)).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep GQA decode: {partial_blocks} split blocks exceed CUDA grid.x"
        ))
    })?;
    let merge_grid_x = u32::try_from(rows.max(1)).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep GQA decode: {rows} rows exceed CUDA grid.x"
        ))
    })?;
    let warps = WARPS_PER_BLOCK as usize;
    let shared_floats = warps
        .checked_mul(2)
        .and_then(|base| warps.checked_mul(head_dim).map(|acc| base + acc))
        .ok_or_else(|| EpError::KernelFailed("cuda_ep GQA decode: shared-mem overflow".into()))?;
    let shared_mem_bytes =
        u32::try_from(shared_floats * std::mem::size_of::<f32>()).map_err(|_| {
            EpError::KernelFailed("cuda_ep GQA decode: shared-mem bytes exceed u32".into())
        })?;

    let function = runtime.nvrtc_function(MODULE_KEY, DECODE_SRC, ENTRY)?;
    let mut builder = runtime.stream().launch_builder(&function);
    builder
        .arg(&query)
        .arg(&key)
        .arg(&value)
        .arg(&output)
        .arg(&total_lengths)
        .arg(&batch_i)
        .arg(&heads_i)
        .arg(&kv_heads_i)
        .arg(&query_seq_i)
        .arg(&dim_i)
        .arg(&capacity_i)
        .arg(&group_i)
        .arg(&scale)
        .arg(&local_window)
        .arg(&softcap);
    // SAFETY: `ENTRY` matches this argument ABI; all buffers were sized by the
    // caller. Fixed module-global and dynamic shared scratch make the launch
    // legal to record into and replay from a CUDA graph.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (WARPS_PER_BLOCK * WARP_SIZE, 1, 1),
            shared_mem_bytes,
        })
    }
    .map_err(|error| driver_err("launch GQA f32 split-K attention", error))?;

    let merge_function = runtime.nvrtc_function(MODULE_KEY, DECODE_SRC, MERGE_ENTRY)?;
    let mut merge_builder = runtime.stream().launch_builder(&merge_function);
    merge_builder
        .arg(&output)
        .arg(&total_lengths)
        .arg(&batch_i)
        .arg(&heads_i)
        .arg(&query_seq_i)
        .arg(&dim_i)
        .arg(&local_window);
    // SAFETY: the same-stream partial launch writes every active split state
    // consumed here; both launches are graph-recordable.
    unsafe {
        merge_builder.launch(LaunchConfig {
            grid_dim: (merge_grid_x, 1, 1),
            block_dim: (WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|error| driver_err("launch GQA f32 split-K merge", error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn runtime() -> Option<Arc<CudaRuntime>> {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let runtime = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(previous_hook);
        runtime
    }

    fn as_bytes<T: Copy>(values: &[T]) -> &[u8] {
        // SAFETY: reinterpreting a POD slice as raw bytes for a host->device copy.
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    fn as_bytes_mut<T: Copy>(values: &mut [T]) -> &mut [u8] {
        // SAFETY: reinterpreting a POD slice as raw bytes for a device->host copy.
        unsafe {
            std::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(values),
            )
        }
    }

    /// Standard (non-online) softmax attention reference in f64, matching the
    /// exact math of `gqa_attention_reference_f32` for the decode shape.
    #[allow(clippy::too_many_arguments)]
    fn cpu_reference(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        total: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cache_capacity: usize,
        scale: f32,
    ) -> Vec<f32> {
        let group = num_heads / num_kv_heads;
        let mut output = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let kv_head = h / group;
            let q_base = h * head_dim;
            let mut scores = vec![0.0f64; total];
            let mut maximum = f64::NEG_INFINITY;
            for (key_pos, score_slot) in scores.iter_mut().enumerate() {
                let k_base = (kv_head * cache_capacity + key_pos) * head_dim;
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += query[q_base + d] as f64 * key[k_base + d] as f64;
                }
                let score = dot * scale as f64;
                *score_slot = score;
                maximum = maximum.max(score);
            }
            let mut denom = 0.0f64;
            for score in scores.iter_mut() {
                *score = (*score - maximum).exp();
                denom += *score;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for (key_pos, prob) in scores.iter().enumerate() {
                    let v_index = (kv_head * cache_capacity + key_pos) * head_dim + d;
                    acc += prob / denom * value[v_index] as f64;
                }
                output[q_base + d] = acc as f32;
            }
        }
        output
    }

    #[test]
    fn decode_kernel_matches_reference_softmax() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping CUDA GQA decode parity test: CUDA runtime unavailable");
            return;
        };

        let batch = 1usize;
        let num_heads = 14usize;
        let num_kv_heads = 2usize;
        let head_dim = 64usize;
        let cache_capacity = 1024usize;
        let group = num_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Deterministic LCG so the test is reproducible without extra crates.
        let mut state = 0x1234_5678u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let query: Vec<f32> = (0..num_heads * head_dim).map(|_| next()).collect();
        let key: Vec<f32> = (0..num_kv_heads * cache_capacity * head_dim)
            .map(|_| next())
            .collect();
        let value: Vec<f32> = (0..num_kv_heads * cache_capacity * head_dim)
            .map(|_| next())
            .collect();

        let query_dev = runtime.alloc_raw(query.len() * 4).unwrap();
        let key_dev = runtime.alloc_raw(key.len() * 4).unwrap();
        let value_dev = runtime.alloc_raw(value.len() * 4).unwrap();
        let output_dev = runtime.alloc_raw(num_heads * head_dim * 4).unwrap();
        let totals_dev = runtime.alloc_raw(batch * 4).unwrap();

        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime.htod(as_bytes(&query), query_dev).unwrap();
            runtime.htod(as_bytes(&key), key_dev).unwrap();
            runtime.htod(as_bytes(&value), value_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        for total in [1usize, 7, 64, 255, 1023] {
            let totals = [total as i32];
            // SAFETY: `totals_dev` holds `batch` i32 values.
            unsafe {
                runtime.htod(as_bytes(&totals), totals_dev).unwrap();
            }

            run(
                &runtime,
                batch,
                num_heads,
                num_kv_heads,
                1,
                head_dim,
                cache_capacity,
                group,
                scale,
                query_dev,
                key_dev,
                value_dev,
                output_dev,
                totals_dev,
                0,
                0.0,
            )
            .unwrap();

            let mut got = vec![0.0f32; num_heads * head_dim];
            // SAFETY: `output_dev` holds `num_heads * head_dim` f32 values.
            unsafe {
                runtime.dtoh(as_bytes_mut(&mut got), output_dev).unwrap();
            }

            let expected = cpu_reference(
                &query,
                &key,
                &value,
                total,
                num_heads,
                num_kv_heads,
                head_dim,
                cache_capacity,
                scale,
            );

            for (g, e) in got.iter().zip(expected.iter()) {
                let abs = (g - e).abs();
                let rel = abs / e.abs().max(1e-4);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
            }
        }

        // SAFETY: each pointer came from this runtime's `alloc_raw` and is freed once.
        unsafe {
            runtime.free_raw(query_dev).unwrap();
            runtime.free_raw(key_dev).unwrap();
            runtime.free_raw(value_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
            runtime.free_raw(totals_dev).unwrap();
        }

        eprintln!("GQA decode parity: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}");
        assert!(
            worst_abs < 1e-3,
            "decode kernel diverged from reference softmax: max_abs={worst_abs:.3e}"
        );
        assert!(
            worst_rel < 5e-3,
            "decode kernel diverged from reference softmax: max_rel={worst_rel:.3e}"
        );
    }

    #[test]
    fn support_gate_targets_single_token_decode() {
        assert!(supported(1, 64));
        assert!(supported(1, 128));
        assert!(!supported(1, 129));
        assert!(!supported(2, 64));
        assert!(!supported(1, 0));
    }
}
