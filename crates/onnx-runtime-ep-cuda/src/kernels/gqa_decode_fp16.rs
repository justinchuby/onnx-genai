//! Capture-safe, fp16 **flash-decode** single-token (`Sq=1`) GQA attention.
//!
//! This is the fp16 sibling of [`super::gqa_decode`]. Where the f32 kernel
//! assigns **one warp per query row** (only ~4 CTAs/layer for Qwen's 14 heads),
//! this kernel matches ORT GenAI's decode geometry: **one multi-warp CTA per
//! query head** (≈14 CTAs/layer), with the warps of a CTA cooperatively
//! **split-K**-reducing across the sequence via an online (running max/sum)
//! flash softmax. Q/K/V and the output are `__half`; every softmax statistic and
//! the value accumulator are kept in **fp32** for numerical stability (the
//! standard flash pattern). K/V are read with **vectorized `half2` loads**.
//!
//! Model shape this targets (Qwen2.5-0.5B): 14 Q heads / 2 KV heads
//! (`group_size = 7`), `head_dim = 64`, `Sq = 1`.
//!
//! ## Reduction strategy
//!
//! For a CTA that owns query head `h`:
//!   * warp `w` walks key positions `local_start + w, +warps, +2*warps, …` to the
//!     device-resident causal limit, maintaining its own running `(max, sum, acc)`
//!     over that strided subset (a partial flash softmax);
//!   * within a warp, the `QK` dot is spread across the 32 lanes over `head_dim`
//!     and finished with a `__shfl_xor_sync` butterfly (every lane owns `acc` for
//!     its `head2` slots);
//!   * warps then publish `(max, sum, acc)` to dynamic shared memory and warp 0
//!     performs the flash **merge** across warps — rescaling each warp's partial
//!     by `exp(max_w - M)` — and writes the fp16 output.
//!
//! ## RoPE
//!
//! Identical convention to [`super::gqa_decode`]: present keys are already
//! RoPE-rotated at their absolute positions when written into the KV cache, so
//! this kernel applies **no** rotary itself and reads `key`/`value` directly.
//!
//! ## Capture-safety
//!
//! The launch path is legal to record inside a CUDA graph and to replay with
//! only device-buffer contents changing:
//!   * No `stream.synchronize()` or any device sync on the launch path.
//!   * No per-call `cudaMalloc`/`cudaFree`. The only scratch is **dynamic shared
//!     memory** whose size is derived from the shape signature (`head_dim`,
//!     `WARPS_PER_BLOCK`), never allocated in global memory per call.
//!   * Fixed launch geometry: grid `= batch * query_heads * query_seq` and block
//!     `= WARPS_PER_BLOCK * 32` are sized purely from the (fixed-for-decode)
//!     shape signature, never from the runtime valid sequence length. The kernel
//!     loops to the device-resident valid length read from `total_lengths` — the
//!     same tensor the reference kernel reads — so replays observe the updated
//!     length with no relaunch or resize.

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Result};

use crate::error::driver_err;
use crate::runtime::CudaRuntime;

const MODULE_KEY: &str = "gqa_decode_attention_f16_v1";
const ENTRY: &str = "gqa_decode_attention_f16";

/// Largest `head_dim` this kernel supports. Each of the 32 warp lanes owns
/// `ceil(head_dim / 2 / 32)` `half2` slots (2 dims each) in registers, capped at
/// `GQA_MAX_H2PL == 2`, i.e. `head_dim <= 2 * 2 * 32 == 128`.
pub(super) const MAX_HEAD_DIM: usize = 128;

/// Warps grouped into one CTA. Each CTA owns one query head; its warps split-K
/// the sequence. Four warps (128 threads) is the ORT decode geometry and keeps
/// the flash merge cheap (a 4-way reduction in shared memory).
const WARPS_PER_BLOCK: u32 = 4;
const WARP_SIZE: u32 = 32;

/// Whether the fp16 flash-decode kernel handles this shape. Single query token
/// (`Sq=1`) with an **even** `head_dim` within [`MAX_HEAD_DIM`] (the `half2`
/// vectorization requires an even head size).
pub(super) fn supported(query_seq: usize, head_dim: usize) -> bool {
    query_seq == 1 && head_dim % 2 == 0 && (1..=MAX_HEAD_DIM).contains(&head_dim)
}

const DECODE_SRC: &str = r#"
#include <cuda_fp16.h>

#define GQA_WARP_SIZE 32
#define GQA_MAX_H2PL 2   // half2 slots per lane; head_dim <= 2 * 2 * 32 == 128

extern "C" __global__ void gqa_decode_attention_f16(
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    __half* __restrict__ output,
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
    // Dynamic shared layout: warp_max[warps], warp_sum[warps], then
    // warp_acc[warps * head_size] (fp32 partial value accumulators per warp).
    extern __shared__ float smem[];
    const int warps_per_block = blockDim.x / GQA_WARP_SIZE;
    float* warp_max = smem;
    float* warp_sum = warp_max + warps_per_block;
    float* warp_acc = warp_sum + warps_per_block;

    const int lane = threadIdx.x % GQA_WARP_SIZE;
    const int warp_in_block = threadIdx.x / GQA_WARP_SIZE;

    // One CTA per (batch, query_head, query_pos) row. Fixed grid == shape.
    const int row = blockIdx.x;
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

    const long q_base =
        ((long)(batch_index * query_heads + query_head) * query_seq + query_pos)
            * (long)head_size;
    const long kv_plane =
        (long)(batch_index * kv_heads + kv_head) * (long)cache_capacity * (long)head_size;

    const int h2 = head_size >> 1;   // number of half2 elements per row
    const half2* q2 = reinterpret_cast<const half2*>(query + q_base);

    float2 q_reg[GQA_MAX_H2PL];
    float2 acc[GQA_MAX_H2PL];
#pragma unroll
    for (int i = 0; i < GQA_MAX_H2PL; ++i) {
        const int j = lane + i * GQA_WARP_SIZE;
        q_reg[i] = (j < h2) ? __half22float2(q2[j]) : make_float2(0.0f, 0.0f);
        acc[i] = make_float2(0.0f, 0.0f);
    }

    const float negative_infinity = __int_as_float(0xff800000);
    float running_max = negative_infinity;
    float running_sum = 0.0f;

    // Split-K: each warp strides through a disjoint subset of key positions.
    for (int key_pos = local_start + warp_in_block; key_pos <= causal_limit;
         key_pos += warps_per_block) {
        const long kv_off = kv_plane + (long)key_pos * (long)head_size;
        const half2* k2 = reinterpret_cast<const half2*>(key + kv_off);
        float partial = 0.0f;
#pragma unroll
        for (int i = 0; i < GQA_MAX_H2PL; ++i) {
            const int j = lane + i * GQA_WARP_SIZE;
            if (j < h2) {
                const float2 k = __half22float2(k2[j]);
                partial += q_reg[i].x * k.x + q_reg[i].y * k.y;
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
        const half2* v2 = reinterpret_cast<const half2*>(value + kv_off);
#pragma unroll
        for (int i = 0; i < GQA_MAX_H2PL; ++i) {
            const int j = lane + i * GQA_WARP_SIZE;
            const float2 v = (j < h2) ? __half22float2(v2[j]) : make_float2(0.0f, 0.0f);
            acc[i].x = acc[i].x * correction + probability * v.x;
            acc[i].y = acc[i].y * correction + probability * v.y;
        }
        running_max = new_max;
    }

    // Publish each warp's partial flash state to shared memory.
    if (lane == 0) {
        warp_max[warp_in_block] = running_max;
        warp_sum[warp_in_block] = running_sum;
    }
#pragma unroll
    for (int i = 0; i < GQA_MAX_H2PL; ++i) {
        const int j = lane + i * GQA_WARP_SIZE;
        if (j < h2) {
            warp_acc[warp_in_block * head_size + 2 * j] = acc[i].x;
            warp_acc[warp_in_block * head_size + 2 * j + 1] = acc[i].y;
        }
    }
    __syncthreads();

    // Warp 0 merges all warps' partials (flash combine) and writes fp16 output.
    if (warp_in_block != 0) {
        return;
    }
    float global_max = negative_infinity;
    for (int w = 0; w < warps_per_block; ++w) {
        global_max = fmaxf(global_max, warp_max[w]);
    }
    float denom = 0.0f;
    for (int w = 0; w < warps_per_block; ++w) {
        denom += warp_sum[w] * expf(warp_max[w] - global_max);
    }
    const float inverse_sum = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

    half2* out2 = reinterpret_cast<half2*>(output + q_base);
#pragma unroll
    for (int i = 0; i < GQA_MAX_H2PL; ++i) {
        const int j = lane + i * GQA_WARP_SIZE;
        if (j < h2) {
            float ox = 0.0f;
            float oy = 0.0f;
            for (int w = 0; w < warps_per_block; ++w) {
                const float weight = expf(warp_max[w] - global_max);
                ox += warp_acc[w * head_size + 2 * j] * weight;
                oy += warp_acc[w * head_size + 2 * j + 1] * weight;
            }
            out2[j] = __floats2half2_rn(ox * inverse_sum, oy * inverse_sum);
        }
    }
}
"#;

/// Launch the capture-safe fp16 flash-decode kernel.
///
/// Present K/V live in `[batch, kv_heads, cache_capacity, head_dim]` fp16 with
/// RoPE already applied to stored keys at their absolute positions;
/// `query`/`output` are BNSH fp16 with `query_seq == 1`. The valid length per
/// batch is read on the device from `total_lengths` (never from
/// `cache_capacity`), so the launch geometry is fixed for capture/replay.
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
    runtime.require_nvrtc_half_headers("gqa_decode_attention_f16")?;

    let as_i32 = |name: &str, value: usize| {
        i32::try_from(value).map_err(|_| {
            EpError::KernelFailed(format!(
                "cuda_ep GQA fp16 decode: {name} {value} exceeds i32"
            ))
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
        .ok_or_else(|| {
            EpError::KernelFailed("cuda_ep GQA fp16 decode: row count overflow".into())
        })?;
    let grid_x = u32::try_from(rows.max(1)).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep GQA fp16 decode: {rows} rows exceed CUDA grid.x"
        ))
    })?;

    // Dynamic shared: warp_max[warps] + warp_sum[warps] + warp_acc[warps*head].
    let warps = WARPS_PER_BLOCK as usize;
    let shared_floats = warps
        .checked_mul(2)
        .and_then(|base| warps.checked_mul(head_dim).map(|acc| base + acc))
        .ok_or_else(|| {
            EpError::KernelFailed("cuda_ep GQA fp16 decode: shared-mem size overflow".into())
        })?;
    let shared_mem_bytes =
        u32::try_from(shared_floats * std::mem::size_of::<f32>()).map_err(|_| {
            EpError::KernelFailed("cuda_ep GQA fp16 decode: shared-mem bytes exceed u32".into())
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
    // caller (present K/V span `cache_capacity` rows, query/output span
    // `query_seq` tokens). The only scratch is fixed-size dynamic shared memory
    // and the kernel never syncs, so the launch is legal to record into and
    // replay from a CUDA graph.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (WARPS_PER_BLOCK * WARP_SIZE, 1, 1),
            shared_mem_bytes,
        })
    }
    .map_err(|error| driver_err("launch GQA fp16 flash-decode attention", error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use half::f16;

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

    /// fp32 (accumulated in f64) softmax attention oracle. It consumes the
    /// **fp16-rounded** inputs so the only residual error the parity test sees is
    /// the kernel's fp16 output rounding plus its fp32 (vs f64) accumulation —
    /// not the input quantization, which both sides share.
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
    fn fp16_decode_kernel_matches_reference_softmax() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping CUDA GQA fp16 decode parity test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("gqa_decode_attention_f16")
            .is_err()
        {
            eprintln!("skipping CUDA GQA fp16 decode parity test: fp16 NVRTC headers unavailable");
            return;
        }

        let batch = 1usize;
        let num_heads = 14usize;
        let num_kv_heads = 2usize;
        let head_dim = 64usize;
        let cache_capacity = 256usize;
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

        // Round to fp16 once; keep the fp16 bits (device input) and the
        // fp16-value-as-f32 (reference input) so both paths agree on inputs.
        let round = |v: f32| -> (f16, f32) {
            let h = f16::from_f32(v);
            (h, h.to_f32())
        };
        let mut q_f16 = vec![f16::ZERO; num_heads * head_dim];
        let mut q_ref = vec![0.0f32; num_heads * head_dim];
        for (dst_h, dst_f) in q_f16.iter_mut().zip(q_ref.iter_mut()) {
            let (h, f) = round(next());
            *dst_h = h;
            *dst_f = f;
        }
        let kv_len = num_kv_heads * cache_capacity * head_dim;
        let mut k_f16 = vec![f16::ZERO; kv_len];
        let mut k_ref = vec![0.0f32; kv_len];
        let mut v_f16 = vec![f16::ZERO; kv_len];
        let mut v_ref = vec![0.0f32; kv_len];
        for i in 0..kv_len {
            let (kh, kf) = round(next());
            k_f16[i] = kh;
            k_ref[i] = kf;
            let (vh, vf) = round(next());
            v_f16[i] = vh;
            v_ref[i] = vf;
        }

        let query_dev = runtime.alloc_raw(q_f16.len() * 2).unwrap();
        let key_dev = runtime.alloc_raw(k_f16.len() * 2).unwrap();
        let value_dev = runtime.alloc_raw(v_f16.len() * 2).unwrap();
        let output_dev = runtime.alloc_raw(num_heads * head_dim * 2).unwrap();
        let totals_dev = runtime.alloc_raw(batch * 4).unwrap();

        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime.htod(as_bytes(&q_f16), query_dev).unwrap();
            runtime.htod(as_bytes(&k_f16), key_dev).unwrap();
            runtime.htod(as_bytes(&v_f16), value_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut all_finite = true;
        for total in [1usize, 7, 64, 200, 255] {
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

            let mut got_f16 = vec![f16::ZERO; num_heads * head_dim];
            // SAFETY: `output_dev` holds `num_heads * head_dim` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut got_f16), output_dev)
                    .unwrap();
            }

            let expected = cpu_reference(
                &q_ref,
                &k_ref,
                &v_ref,
                total,
                num_heads,
                num_kv_heads,
                head_dim,
                cache_capacity,
                scale,
            );

            for (g16, e) in got_f16.iter().zip(expected.iter()) {
                let g = g16.to_f32();
                if !g.is_finite() {
                    all_finite = false;
                }
                let abs = (g - e).abs();
                let rel = abs / e.abs().max(1e-2);
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

        eprintln!("GQA fp16 decode parity: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}");
        assert!(
            all_finite,
            "fp16 decode kernel produced a non-finite output"
        );
        // Tolerance: with fp32 accumulation the residual is dominated by the
        // fp16 output rounding (~2^-11 * |out| for outputs in ~[-1, 1], i.e.
        // ~5e-4). 2e-3 absolute leaves headroom for the fp32-vs-f64 reduction.
        assert!(
            worst_abs < 2e-3,
            "fp16 decode kernel diverged from reference softmax: max_abs={worst_abs:.3e}"
        );
        // Relative error is measured against a 1e-2 floor so near-zero output
        // components (where fp16 rounding dominates) do not blow up the ratio.
        assert!(
            worst_rel < 5e-2,
            "fp16 decode kernel diverged from reference softmax: max_rel={worst_rel:.3e}"
        );
    }

    #[test]
    fn support_gate_targets_even_head_dim_single_token_decode() {
        assert!(supported(1, 64));
        assert!(supported(1, 128));
        assert!(!supported(1, 63)); // odd head_dim: no half2 vectorization
        assert!(!supported(1, 130)); // exceeds MAX_HEAD_DIM
        assert!(!supported(2, 64)); // prefill (Sq > 1)
        assert!(!supported(1, 0));
    }
}
