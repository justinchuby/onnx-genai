//! Standard `ai.onnx::Attention` (opset 23–26): scaled dot-product attention
//! (SDPA) with multi-head / grouped-query head sharing, an optional additive or
//! boolean attention mask, causal masking, and an in-op KV cache
//! (`past_key`/`past_value` → `present_key`/`present_value`).
//!
//! This is the *standard* ONNX operator, distinct from the private
//! `com.microsoft::FusedAttention` fusion node (see [`super::fused_attention`]),
//! which only reproduces the plain `MatMul → scale → [+mask] → Softmax →
//! MatMul` core the optimizer fuses. Standard `Attention` is a richer op: it
//! reshapes 3D `(batch, seq, hidden)` inputs into heads, supports GQA/MQA head
//! sharing, concatenates a past KV cache, offset-aware causal masking, softcap,
//! and emits up to four outputs (`Y`, `present_key`, `present_value`,
//! `qk_matmul_output`).
//!
//! ## Semantics (per the spec's applied pattern)
//!
//! ```text
//! scores = (Q·√scale) · (K·√scale)ᵀ      # √scale folded into each operand so
//!                                        # extreme magnitudes don't overflow;
//!                                        # scale defaults to 1/sqrt(head_size)
//! scores = softcap · tanh(scores/softcap)  # only when softcap != 0
//! scores = scores + attn_bias            # attn_mask (add/-inf) and causal mask
//! probs  = softmax(scores, axis=-1)      # numerically stable; fully-masked → 0
//! Y      = probs · V
//! ```
//!
//! ## GPU-native execution
//!
//! Unlike a host-staged reference that copies Q/K/V to the CPU, this kernel is
//! GPU-native: Q, K, V, the attention mask, and every bulk output stay resident
//! on the device. Two NVRTC kernels do all the heavy lifting:
//!
//! * `build_kv` gathers each K/V input — handling the 3D→4D head reshape and the
//!   `past ⧺ current` cache concatenation — into a contiguous
//!   `[batch, kv_heads, total_seq, dim]` present buffer (also the `present_key`/
//!   `present_value` outputs when requested).
//! * `attention_row` runs one CUDA block per `(batch, q_head, query)` row: it
//!   computes the scaled QK scores, softcap, the composed causal/pad/attn masks,
//!   a numerically-stable softmax, and the probs·V accumulation, writing `Y`
//!   (and the optional `qk_matmul_output`) directly to device output buffers.
//!
//! Only tiny host-side control state leaves/enters the device: the per-batch
//! causal `offset` and padding-frontier arrays are built on the host and
//! uploaded as small device arrays, and `nonpad_kv_seqlen` (a per-batch scalar
//! count) is read back to compute them. Q/K/V and the score/probability tensors
//! never round-trip through host memory.
//!
//! ## Determinism
//!
//! Each score row is reduced in a fixed order: the per-row `QK` dot products and
//! the `probs·V` accumulation each sum in ascending index order within a single
//! thread (bit-identical to the CPU reference), and the softmax max/exp/sum are
//! performed sequentially by the block's lead thread. No atomics contribute to a
//! shared accumulator, so results are byte-identical run to run.
//!
//! ## Versioning
//!
//! `Attention` was added at opset 23 and revised at opsets 24 and 25:
//!
//! * `nonpad_kv_seqlen` (7th input) — an external-cache per-batch valid-token
//!   count — is honored for v24+ and rejected for v23 (it did not exist there).
//! * Inclusive left/right sliding windows are supported from v25.
//!
//! `qk_matmul_output_mode` has the same meaning in each revision: `0` = raw QK, `1` = after
//! softcap (before mask), `2` = after mask+softcap, `3` = after softmax.
//!
//! ## Supported vs. unimplemented
//!
//! * dtype: **f32, f16, and bf16** Q/K/V, cache, additive mask, and outputs.
//!   Q/K/V, cache, and outputs use one dtype; an additive mask may independently
//!   use f32, f16, or bf16. Device f16/bf16 loads and stores are converted around
//!   fp32 score, softmax, and value accumulators.
//! * `qk_matmul_output_mode`: modes **0, 1, 2, 3** implemented per spec; any
//!   other value errors.

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::WorkspaceLayout;
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
    attention_window::{AttentionWindow, validate_attention_head_attributes},
};
use onnx_runtime_ir::{DataType, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
/// Threads per block for `attention_row` (one block services one score row).
const ROW_THREADS: u32 = 128;

/// Threads-per-block for `attention_row`, with an env override for tuning.
///
/// `attention_row` launches one block per (batch, q_head, query) row. In the
/// decode regime (q_seq == 1) the grid is only `batch * q_heads` blocks (16 for
/// a 16-head model), so the kernel is badly under-occupied and memory-latency
/// bound on the per-key K/V load chains. Giving each block more warps lets it
/// hide that latency without changing any reduction order, so the result stays
/// byte-identical. `ONNX_GENAI_ATTN_ROW_THREADS` overrides the count for tuning.
fn attention_row_threads(is_decode: bool) -> u32 {
    use std::sync::OnceLock;
    static OVERRIDE: OnceLock<Option<u32>> = OnceLock::new();
    let over = *OVERRIDE.get_or_init(|| {
        std::env::var("ONNX_GENAI_ATTN_ROW_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&n| (32..=1024).contains(&n) && n % 32 == 0)
    });
    if let Some(n) = over {
        return n;
    }
    if is_decode { 256 } else { ROW_THREADS }
}

/// Upper bound on FlashDecoding key-splits per row (keeps the combine's per-row
/// loop and the partial-output scratch small while still filling the machine).
const ATTN_SPLIT_MAX_SPLITS: u64 = 64;
/// Threads per block for the `attention_split` / `attention_combine` kernels.
const ATTN_SPLIT_THREADS: u32 = 256;
/// Adaptive split-KV target: total `attention_split` blocks
/// (`total_rows * num_splits`) to aim for so the launch fills roughly one full
/// occupancy wave of the machine (~4 resident CTAs on each of the 132 H200 SMs).
/// The decode split kernel is memory-latency bound, so an ncu + E2E chunk sweep
/// on V2-Lite showed the optimum is ~this many blocks (grid ~512 at deep
/// context), NOT one block per SM: at deep context a grid of 132 blocks
/// (`chunk`=512) leaves the SMs latency-starved and is ~5% slower than a grid of
/// 512 (`chunk`=128). Targeting a whole wave keeps enough resident CTAs per SM
/// to hide the KV-load latency.
const ATTN_SPLIT_TARGET_BLOCKS: u64 = 512;
/// Floor on keys per split for the adaptive sizing: never split so finely that a
/// split covers fewer than this many keys. Also the byte-identity margin — any
/// live context `<= ATTN_SPLIT_MIN_CHUNK` stays on the single-split bit-exact
/// fast path (the V2-Lite golden 24-tok decode lock sits well under this).
const ATTN_SPLIT_MIN_CHUNK: u64 = 128;

/// FlashDecoding split-KV configuration for one decode launch, or `None` when
/// the monolithic `attention_row` kernel should be used instead.
///
/// The decode grid is only `batch*q_heads` blocks (~16), so `attention_row`
/// leaves ~116 of 132 SMs idle and is memory-latency bound. Splitting each row's
/// key reduction across `num_splits` blocks fills the machine at wide context,
/// where `attention_row` is the #1 decode kernel (~40% of decode time).
///
/// `num_splits` is derived purely from the fixed physical KV capacity `cap` and
/// the fixed row count `total_rows` (never the live sequence length), so the
/// launch geometry — and the decision to split at all — is identical under eager
/// and CUDA-graph capture. The keys-per-split `chunk` follows from `num_splits`:
/// when the live `total_seq <= chunk` only split 0 is active and reproduces
/// `attention_row` bit-for-bit, so short-context decode stays byte-identical;
/// only genuinely wide context reorders the fp32 accumulation (validated to the
/// f64 oracle tolerance).
///
/// Adaptive sizing: aim for `ATTN_SPLIT_TARGET_BLOCKS` total blocks
/// (`total_rows * num_splits`) so the launch fills ~one occupancy wave, but never
/// split a row into pieces smaller than `ATTN_SPLIT_MIN_CHUNK` keys (which would
/// starve each split of work and lower the byte-identity threshold). For
/// V2-Lite's 16 decode rows this lands `num_splits` near 32 (grid ~512, the
/// measured optimum) at deep context and scales down gracefully at shallow
/// context, where `cap` alone caps the useful split count.
///
/// `ONNX_GENAI_ATTN_SPLITKV=0` forces the monolithic path (A/B baseline);
/// `ONNX_GENAI_ATTN_SPLIT_CHUNK` pins a fixed keys-per-split, overriding the
/// adaptive sizing (for A/B sweeps and backward compatibility).
fn attention_split_config(
    is_decode: bool,
    dev_length_eligible: bool,
    is_causal: bool,
    want_qk: bool,
    cap: u64,
    total_rows: u64,
    _v_head_size: u64,
) -> Option<(u64, u64)> {
    use std::sync::OnceLock;
    static CFG: OnceLock<(bool, Option<u64>)> = OnceLock::new();
    let (enabled, chunk_override) = *CFG.get_or_init(|| {
        let enabled = std::env::var("ONNX_GENAI_ATTN_SPLITKV")
            .ok()
            .map(|s| {
                let s = s.trim();
                !(s == "0" || s.eq_ignore_ascii_case("off") || s.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        let chunk_override = std::env::var("ONNX_GENAI_ATTN_SPLIT_CHUNK")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n >= 32);
        (enabled, chunk_override)
    });
    // Split only on the capture-safe fixed-capacity decode route, and never when
    // the raw QK scores are an observable output (the score scratch is reused as
    // per-key storage inside the split kernel).
    //
    // Also never for the causal form. The multi-split path reorders the fp32
    // key reduction (chunk-local online softmax merged by `attention_combine`)
    // versus the monolithic `attention_row` serial ascending reduction. That is
    // a legitimately different — but not bit-identical — fp32 result, and at a
    // near-tie between the top-1 and top-2 logits it can flip the greedy argmax.
    // The monolithic reduction is the reference: it matches both the parent CUDA
    // build and an independent CPU dense-prefill oracle token-for-token, whereas
    // the split reduction diverges at such a tie (measured on granite-1b: the
    // decode stream flips at the first near-tie once causal decode is routed
    // through the split kernel). The non-causal fixed-capacity decode has always
    // taken the split path, so its byte-identity baseline already includes it;
    // gating the causal form out keeps the newly capture-enabled causal decode
    // bit-for-bit equal to the pre-capture monolithic path while still capturing
    // (the split is a throughput optimization, not a capture requirement).
    if !enabled || !is_decode || !dev_length_eligible || is_causal || want_qk || cap == 0 {
        return None;
    }
    attention_split_geometry(chunk_override, cap, total_rows)
}

/// Pure split-KV geometry: map (`chunk_override`, fixed `cap`, fixed
/// `total_rows`) to `(num_splits, chunk)`, or `None` when a single split (the
/// monolithic path plus combine overhead) is all that is warranted. Split out
/// from [`attention_split_config`] so the capture-critical sizing arithmetic is
/// unit-testable without touching process env.
fn attention_split_geometry(
    chunk_override: Option<u64>,
    cap: u64,
    total_rows: u64,
) -> Option<(u64, u64)> {
    if cap == 0 {
        return None;
    }
    let rows = total_rows.max(1);
    let num_splits = match chunk_override {
        // Fixed-chunk mode: split count follows directly from the pinned chunk.
        Some(chunk) => cap.div_ceil(chunk.max(1)).clamp(1, ATTN_SPLIT_MAX_SPLITS),
        // Adaptive mode: target ~one occupancy wave of blocks, but never split a
        // row finer than ATTN_SPLIT_MIN_CHUNK keys.
        None => {
            let target_splits = ATTN_SPLIT_TARGET_BLOCKS.div_ceil(rows);
            let splits_by_cap = cap.div_ceil(ATTN_SPLIT_MIN_CHUNK);
            target_splits
                .min(splits_by_cap)
                .clamp(1, ATTN_SPLIT_MAX_SPLITS)
        }
    };
    // A single split is just the monolithic kernel plus combine overhead.
    if num_splits < 2 {
        return None;
    }
    // Keys per split; the single-split fast path triggers when total_seq <= chunk.
    let chunk = cap.div_ceil(num_splits);
    Some((num_splits, chunk))
}

/// Floats of split-KV partial scratch for `num_splits` splits over `total_rows`
/// decode rows: 2 metadata floats (running max + sum) plus `v_head_size`
/// unnormalized P·V floats per (row, split) partial.
fn attention_split_scratch_bytes(
    num_splits: u64,
    total_rows: u64,
    v_head_size: u64,
) -> Result<usize> {
    let bytes = v_head_size.checked_add(2)
        .and_then(|columns| columns.checked_mul(total_rows))
        .and_then(|elements| elements.checked_mul(num_splits))
        .and_then(|elements| elements.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| EpError::KernelFailed("Attention: split-KV workspace exceeds addressable capacity; reduce the decode geometry".into()))?;
    Ok(bytes)
}
const ATTENTION_MODULE: &str = "standard_attention_f32_f16_bf16_v4";
const ATTENTION_SOURCE: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define NEG_INF __int_as_float(0xff800000)
#define DERIVE_LEN_THREADS 256

// dtype is 0 for f32, 1 for f16, and 2 for bf16. Keep all computation in fp32;
// only the externally visible activations and cache use the requested storage
// type.
__device__ __forceinline__ float load_float(const void* data, unsigned long long index, int dtype) {
  if (dtype == 0) {
    return ((const float*)data)[index];
  }
  if (dtype == 1) {
    return __half2float(((const __half*)data)[index]);
  }
  return __bfloat162float(((const __nv_bfloat16*)data)[index]);
}

__device__ __forceinline__ void store_float(void* data, unsigned long long index, float value, int dtype) {
  if (dtype == 0) {
    ((float*)data)[index] = value;
  } else if (dtype == 1) {
    ((__half*)data)[index] = __float2half_rn(value);
  } else {
    ((__nv_bfloat16*)data)[index] = __float2bfloat16_rn(value);
  }
}

// Gather a K/V input into a contiguous [batch, heads, total_seq, dim] present
// Derive the valid attended length on-device by scanning ONE row of the additive
// attention mask bias for its first masked (large-negative) entry. The scanned
// row is the LAST query row (`row_base` = the element offset of query i=q_seq-1
// within the broadcast [.., q_seq, key_len] mask); at the final query position
// the causal+padding frontier equals the total valid key length, so this returns
// `total_seq` for a single-token decode AND `prompt_len` for a multi-token
// prefill (row 0 would wrongly report 1 under a causal mask — hence the last
// row). At fixed capacity the row is [.., max_len] with 0 bias for valid keys
// [0,total) and a large-negative bias for padding [total,max_len); the frontier
// index is the valid length. This lets both phases read their length from device
// memory (the mask the kernel already consumes) instead of host shape metadata,
// so the launch geometry stays fixed and capture-safe. Assumes a single
// contiguous right-aligned valid run (greedy decode, no interior pads).
extern "C" __global__ void derive_len(
    const void* mask, int mask_kind, unsigned long long key_len,
    unsigned long long row_base, int* out_len) {
  // Block-parallel min-index reduction over the mask row: each thread strides
  // the row and records the first padded position (v < -1000) it encounters;
  // the block-min of those indices is the overall frontier. Under the single
  // contiguous right-aligned valid run this ABI assumes, every index < the
  // frontier is valid and every index >= it is padded, so the block-min equals
  // the valid length (or `key_len` if no padding is present). This is
  // byte-identical to the prior single-thread serial scan but replaces its
  // O(key_len) latency-bound walk (which dominates decode at wide context) with
  // a coalesced parallel pass. Launched as grid(1,1,1) with DERIVE_LEN_THREADS
  // threads so the geometry stays fixed and capture-safe.
  const int nthreads = blockDim.x;
  const int tid = threadIdx.x;
  int local = (int)key_len;
  for (unsigned long long j = (unsigned long long)tid; j < key_len;
       j += (unsigned long long)nthreads) {
    const unsigned long long idx = row_base + j;
    float v;
    if (mask_kind == 1) {
      v = ((const float*)mask)[idx];
    } else if (mask_kind == 3) {
      v = __half2float(((const __half*)mask)[idx]);
    } else if (mask_kind == 4) {
      v = __bfloat162float(((const __nv_bfloat16*)mask)[idx]);
    } else {
      v = ((const unsigned char*)mask)[idx] != 0 ? 0.0f : NEG_INF;
    }
    if (v < -1000.0f) {
      // Indices are scanned in ascending order within this thread's stride, so
      // the first hit is this thread's smallest padded index.
      local = (int)j;
      break;
    }
  }
  __shared__ int red[DERIVE_LEN_THREADS];
  red[tid] = local;
  __syncthreads();
  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s && red[tid + s] < red[tid]) {
      red[tid] = red[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_len[0] = red[0];
  }
}

// buffer, applying the 3D->4D head reshape and the past ++ current concat.
extern "C" __global__ void build_kv(
    const void* past, const void* cur, void* out, int dtype,
    int has_past, int cur_is_3d, int past_is_3d,
    unsigned long long batch, unsigned long long heads,
    unsigned long long past_seq, unsigned long long cur_seq,
    unsigned long long total_seq, unsigned long long dim,
    unsigned long long out_cap, unsigned long long past_cap,
    unsigned long long write_start, unsigned long long elements,
    const int* dev_len) {
  // Capture-safe path: when `dev_len` is provided the valid length (and hence
  // the append slot) is read from device memory rather than the host-provided
  // `total_seq`/`write_start`/`past_seq`. `cur_seq` (=1 at decode) and the
  // per-head capacities stay host-constant, so grid geometry never changes.
  if (dev_len != nullptr) {
    const int total = dev_len[0];
    past_seq = (unsigned long long)(total - (long long)cur_seq);
    total_seq = (unsigned long long)total;
    write_start = past_seq;
  }
  // `out_cap`/`past_cap` are the per-head seq strides of the destination and
  // (4D) source caches. When they exceed the valid length the cache is stored
  // at a fixed physical capacity, so head h occupies a constant slot and the
  // new token is appended at row `t` without restriding the prior rows. In the
  // dense case out_cap==total_seq and past_cap==past_seq (legacy behavior).
  // `write_start` lets the fixed-slot append rebuild only rows [write_start,
  // total_seq); a full rebuild passes 0.
  const unsigned long long span = total_seq - write_start;
  for (unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements;
       idx += (unsigned long long)gridDim.x * blockDim.x) {
    unsigned long long d = idx % dim;
    unsigned long long rem = idx / dim;
    unsigned long long t = write_start + (rem % span);
    rem /= span;
    unsigned long long h = rem % heads;
    unsigned long long b = rem / heads;
    float val;
    if (has_past && t < past_seq) {
      unsigned long long off = past_is_3d
          ? (b * past_seq + t) * (heads * dim) + h * dim + d
          : ((b * heads + h) * past_cap + t) * dim + d;
      val = load_float(past, off, dtype);
    } else {
      unsigned long long c = has_past ? (t - past_seq) : t;
      unsigned long long off = cur_is_3d
          ? (b * cur_seq + c) * (heads * dim) + h * dim + d
          : ((b * heads + h) * cur_seq + c) * dim + d;
      val = load_float(cur, off, dtype);
    }
    unsigned long long out_off = ((b * heads + h) * out_cap + t) * dim + d;
    store_float(out, out_off, val, dtype);
  }
}

// Additive mask bias for logical index (b, h, i, j), broadcasting a rank<=4
// mask right-aligned against [b, h, i, j]. Mirrors the CPU reference exactly:
// a last dim shorter than total_seq pads with -inf; bool false -> -inf.
__device__ __forceinline__ float mask_bias(
    const void* mask, int mask_kind, int mask_rank,
    unsigned long long md0, unsigned long long md1,
    unsigned long long md2, unsigned long long md3,
    unsigned long long b, unsigned long long h,
    unsigned long long i, unsigned long long j,
    unsigned long long total_seq) {
  if (mask_kind == 0) {
    return 0.0f;
  }
  unsigned long long full[4] = {b, h, i, j};
  unsigned long long md[4] = {md0, md1, md2, md3};
  unsigned long long off = 0;
  for (int a = 0; a < 4; ++a) {
    unsigned long long idx = (md[a] == 1ULL) ? 0ULL : full[a];
    off = off * md[a] + idx;
  }
  if (mask_rank > 0) {
    unsigned long long last = md3;
    if (j >= last && last < total_seq) {
      return NEG_INF;
    }
  }
  if (mask_kind == 1) {
    return ((const float*)mask)[off];
  }
  if (mask_kind == 3) {
    return __half2float(((const __half*)mask)[off]);
  }
  if (mask_kind == 4) {
    return __bfloat162float(((const __nv_bfloat16*)mask)[off]);
  }
  // Bool mask: nonzero keeps (bias 0), zero masks (-inf).
  return ((const unsigned char*)mask)[off] != 0 ? 0.0f : NEG_INF;
}

// One block per (batch, q_head, query) row. Computes scaled QK scores, softcap,
// the composed causal/pad/attn masks, a stable softmax, and probs*V.
extern "C" __global__ void attention_row(
    const void* q, const void* key, const void* value,
    const void* mask, float* scores, void* y, void* qk_out,
    const long long* offsets, const long long* pad_limits, long long default_offset,
    unsigned long long batch, unsigned long long q_heads, unsigned long long q_seq,
    unsigned long long kv_heads, unsigned long long total_seq_arg,
    unsigned long long cap,
    unsigned long long head_size, unsigned long long v_head_size,
    unsigned long long group,
    int dtype, int q_is_3d, int out_is_3d, int is_causal,
    float sqrt_scale, float softcap,
    int mask_kind, int mask_rank,
    unsigned long long md0, unsigned long long md1,
    unsigned long long md2, unsigned long long md3,
    int qk_mode, int want_qk, const int* dev_len,
    long long window_left, long long window_right) {
  const unsigned long long row = blockIdx.x;
  const unsigned long long total_rows = batch * q_heads * q_seq;
  if (row >= total_rows) {
    return;
  }
  // Capture-safe path: read the growing valid length from device memory (the
  // frontier `derive_len` scanned from the mask) instead of the host-provided
  // extent, so the launch geometry stays fixed. The per-head key/value stride
  // (`cap`) is the fixed physical capacity, and the score scratch is sized for
  // `total_rows * cap`, so a device length <= cap indexes within bounds.
  const unsigned long long total_seq =
      (dev_len != nullptr) ? (unsigned long long)dev_len[0] : total_seq_arg;
  const unsigned long long i = row % q_seq;
  unsigned long long rem = row / q_seq;
  const unsigned long long qh = rem % q_heads;
  const unsigned long long b = rem / q_heads;
  const unsigned long long kvh = qh / group;
  const unsigned long long srow = row * total_seq;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // Base offset of this query row's head vector.
  const unsigned long long qoff = q_is_3d
      ? (b * q_seq + i) * (q_heads * head_size) + qh * head_size
      : ((b * q_heads + qh) * q_seq + i) * head_size;

  const long long offset =
      (dev_len != nullptr) ? (long long)total_seq - (long long)q_seq : (offsets != nullptr ? offsets[b] : default_offset);
  const long long pad_limit = pad_limits != nullptr ? pad_limits[b] : -1LL;
  const long long position = (long long)i + offset;
  unsigned long long begin = 0, end = total_seq;
  if (window_left >= 0 || window_right >= 0) {
    if (pad_limit >= 0) end = min(end, (unsigned long long)pad_limit);
    if (window_left >= 0 && position > window_left) {
      begin = min(end, (unsigned long long)(position - window_left));
    }
    const long long right = is_causal ? 0 : window_right;
    // Clip before adding: right_window_size can be as large as INT64_MAX.
    if (right >= 0 && end > 0 && position < (long long)(end - 1) - right) {
      end = (unsigned long long)max(0LL, position + right + 1);
      begin = min(begin, end);
    }
  }
  // Pre-mask diagnostics still expose every QK score.
  const bool dense_qk = want_qk && qk_mode < 2;
  const unsigned long long score_begin = dense_qk ? 0 : begin;
  const unsigned long long score_end = dense_qk ? total_seq : end;

  // Stage 1: scaled Q·Kᵀ scores (sqrt(scale) folded into each operand).
  for (unsigned long long j = score_begin + tid; j < score_end; j += nthreads) {
    const unsigned long long koff = ((b * kv_heads + kvh) * cap + j) * head_size;
    float acc = 0.0f;
    for (unsigned long long p = 0; p < head_size; ++p) {
      acc += (load_float(q, qoff + p, dtype) * sqrt_scale)
          * (load_float(key, koff + p, dtype) * sqrt_scale);
    }
    scores[srow + j] = acc;
    if (want_qk && qk_mode == 0) {
      store_float(qk_out, srow + j, acc, dtype);
    }
  }
  __syncthreads();

  // Stage 2: softcap (before mask), applied when nonzero.
  if (softcap != 0.0f) {
    for (unsigned long long j = score_begin + tid; j < score_end; j += nthreads) {
      const float s = scores[srow + j];
      scores[srow + j] = softcap * tanhf(s / softcap);
    }
    __syncthreads();
  }
  if (want_qk && qk_mode == 1) {
    for (unsigned long long j = tid; j < total_seq; j += nthreads) {
      store_float(qk_out, srow + j, scores[srow + j], dtype);
    }
    __syncthreads();
  }

  // Stage 3: attention mask + causal frontier + padding frontier.
  // When the valid length is read on-device (`dev_len`), the causal frontier is
  // derived from it too (`offset = total_seq - q_seq`, i.e. the on-device past
  // length) rather than the host `offsets[b]`, which under a frozen fixed-
  // capacity binding reports the padded capacity. Query row `i` (absolute
  // position `past + i`) attends keys `[0, past + i]`.
  for (unsigned long long j = begin + tid; j < end; j += nthreads) {
    const long long jj = (long long)j;
    if (pad_limit >= 0 && jj >= pad_limit) {
      scores[srow + j] = NEG_INF;
      continue;
    }
    if (is_causal && jj > position) {
      scores[srow + j] = NEG_INF;
      continue;
    }
    scores[srow + j] += mask_bias(mask, mask_kind, mask_rank, md0, md1, md2, md3,
                                  b, qh, i, j, total_seq);
  }
  __syncthreads();
  if (want_qk && qk_mode == 2) {
    for (unsigned long long j = tid; j < total_seq; j += nthreads) {
      store_float(qk_out, srow + j, j >= begin && j < end ? scores[srow + j] : NEG_INF, dtype);
    }
    __syncthreads();
  }

  // Stage 4: numerically-stable softmax. The lead thread performs the max,
  // exp, and sum in a fixed ascending order to match the CPU reference and be
  // reproducible; the final normalize is embarrassingly parallel.
  __shared__ float inv_sum_sh;
  __shared__ int all_masked_sh;
  if (tid == 0) {
    float m = NEG_INF;
    for (unsigned long long j = begin; j < end; ++j) {
      m = fmaxf(m, scores[srow + j]);
    }
    if (m == NEG_INF) {
      all_masked_sh = 1;
      inv_sum_sh = 0.0f;
    } else {
      all_masked_sh = 0;
      float sum = 0.0f;
      for (unsigned long long j = begin; j < end; ++j) {
        const float e = expf(scores[srow + j] - m);
        scores[srow + j] = e;
        sum += e;
      }
      inv_sum_sh = 1.0f / sum;
    }
  }
  __syncthreads();
  if (all_masked_sh) {
    for (unsigned long long j = begin + tid; j < end; j += nthreads) {
      scores[srow + j] = 0.0f;
    }
  } else {
    const float inv = inv_sum_sh;
    for (unsigned long long j = begin + tid; j < end; j += nthreads) {
      scores[srow + j] *= inv;
    }
  }
  __syncthreads();
  if (want_qk && qk_mode == 3) {
    for (unsigned long long j = tid; j < total_seq; j += nthreads) {
      store_float(qk_out, srow + j, j >= begin && j < end ? scores[srow + j] : 0.0f, dtype);
    }
    __syncthreads();
  }

  // Stage 5: Y = probs · V. Each thread owns whole output channels and sums
  // over keys in ascending order (bit-identical to the CPU reference).
  const unsigned long long ybase = out_is_3d
      ? (b * q_seq + i) * (q_heads * v_head_size) + qh * v_head_size
      : ((b * q_heads + qh) * q_seq + i) * v_head_size;
  for (unsigned long long c = tid; c < v_head_size; c += nthreads) {
    float acc = 0.0f;
    for (unsigned long long j = begin; j < end; ++j) {
      const unsigned long long voff = ((b * kv_heads + kvh) * cap + j) * v_head_size;
      acc += scores[srow + j] * load_float(value, voff + c, dtype);
    }
    store_float(y, ybase + c, acc, dtype);
  }
}

// FlashDecoding split: one block per (row, split). Each block reduces the
// contiguous key slice [split*chunk, min((split+1)*chunk, total_seq)) of one
// query row and writes a partial (max, sum, unnormalized P·V) into split_meta /
// split_out; `attention_combine` later merges the partials with a log-sum-exp
// rescale. This spreads a single row's key reduction across `num_splits` blocks
// so the decode grid (batch*q_heads rows, ~16) fills the machine instead of
// leaving ~116 SMs idle.
//
// Byte-identity fast path: when the whole row fits in one chunk
// (total_seq <= chunk => only split 0 is active) split 0 runs the EXACT
// `attention_row` reduction (global max, serial ascending softmax, ascending
// P·V) and writes the FINAL normalized output with a sentinel (max=0, sum=1) so
// the combine pass is a bit-exact identity. Only the genuinely multi-split
// (wide-context) path reorders the fp32 accumulation.
extern "C" __global__ void attention_split(
    const void* q, const void* key, const void* value,
    const void* mask, float* scores, float* split_out, float* split_meta,
    const long long* offsets, const long long* pad_limits, long long default_offset,
    unsigned long long batch, unsigned long long q_heads, unsigned long long q_seq,
    unsigned long long kv_heads, unsigned long long total_seq_arg,
    unsigned long long cap,
    unsigned long long head_size, unsigned long long v_head_size,
    unsigned long long group,
    int dtype, int q_is_3d, int is_causal,
    float sqrt_scale, float softcap,
    int mask_kind, int mask_rank,
    unsigned long long md0, unsigned long long md1,
    unsigned long long md2, unsigned long long md3,
    const int* dev_len,
    unsigned long long num_splits, unsigned long long chunk) {
  const unsigned long long row = blockIdx.x;
  const unsigned long long split = blockIdx.y;
  const unsigned long long total_rows = batch * q_heads * q_seq;
  if (row >= total_rows || split >= num_splits) {
    return;
  }
  const unsigned long long total_seq =
      (dev_len != nullptr) ? (unsigned long long)dev_len[0] : total_seq_arg;
  const unsigned long long i = row % q_seq;
  unsigned long long rem = row / q_seq;
  const unsigned long long qh = rem % q_heads;
  const unsigned long long b = rem / q_heads;
  const unsigned long long kvh = qh / group;
  const unsigned long long srow = row * total_seq;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const unsigned long long moff = (row * num_splits + split) * 2;
  const unsigned long long obase = (row * num_splits + split) * v_head_size;

  const unsigned long long chunk_start = split * chunk;
  unsigned long long chunk_end = chunk_start + chunk;
  if (chunk_end > total_seq) {
    chunk_end = total_seq;
  }

  // Empty split (no keys): emit a neutral partial that never wins the combine.
  if (chunk_start >= total_seq) {
    if (tid == 0) {
      split_meta[moff] = NEG_INF;
      split_meta[moff + 1] = 0.0f;
    }
    for (unsigned long long c = tid; c < v_head_size; c += nthreads) {
      split_out[obase + c] = 0.0f;
    }
    return;
  }

  const unsigned long long qoff = q_is_3d
      ? (b * q_seq + i) * (q_heads * head_size) + qh * head_size
      : ((b * q_heads + qh) * q_seq + i) * head_size;
  // See `attention_row`: with an on-device valid length the causal frontier is
  // derived on-device (`total_seq - q_seq`) instead of the host `offsets[b]`,
  // which reports padded capacity under a frozen fixed-capacity binding.
  const long long offset =
      (dev_len != nullptr) ? (long long)total_seq - (long long)q_seq : (offsets != nullptr ? offsets[b] : default_offset);
  const long long pad_limit = pad_limits != nullptr ? pad_limits[b] : -1LL;
  const long long causal_limit = (long long)i + offset;
  const bool single = (total_seq <= chunk);

  // Stage 1: scaled Q·Kᵀ for this chunk's keys, then softcap + mask + frontier,
  // staged into the score scratch (same values as attention_row stages 1-3).
  for (unsigned long long j = chunk_start + tid; j < chunk_end; j += nthreads) {
    const unsigned long long koff = ((b * kv_heads + kvh) * cap + j) * head_size;
    float acc = 0.0f;
    for (unsigned long long p = 0; p < head_size; ++p) {
      acc += (load_float(q, qoff + p, dtype) * sqrt_scale)
          * (load_float(key, koff + p, dtype) * sqrt_scale);
    }
    if (softcap != 0.0f) {
      acc = softcap * tanhf(acc / softcap);
    }
    const long long jj = (long long)j;
    if (pad_limit >= 0 && jj >= pad_limit) {
      acc = NEG_INF;
    } else if (is_causal && jj > causal_limit) {
      acc = NEG_INF;
    } else {
      acc += mask_bias(mask, mask_kind, mask_rank, md0, md1, md2, md3,
                       b, qh, i, j, total_seq);
    }
    scores[srow + j] = acc;
  }
  __syncthreads();

  if (single) {
    // Byte-identical path: reproduce attention_row exactly on the lead thread
    // (global max, ascending exp/sum, ascending normalize) then write the FINAL
    // normalized output with a pass-through sentinel for the combine.
    __shared__ float inv_sum_sh;
    __shared__ int all_masked_sh;
    if (tid == 0) {
      float m = NEG_INF;
      for (unsigned long long j = 0; j < total_seq; ++j) {
        m = fmaxf(m, scores[srow + j]);
      }
      if (m == NEG_INF) {
        all_masked_sh = 1;
        inv_sum_sh = 0.0f;
      } else {
        all_masked_sh = 0;
        float sum = 0.0f;
        for (unsigned long long j = 0; j < total_seq; ++j) {
          const float e = expf(scores[srow + j] - m);
          scores[srow + j] = e;
          sum += e;
        }
        inv_sum_sh = 1.0f / sum;
      }
      // Sentinel: combine computes out = split_out * exp(0-0) / 1 = split_out.
      split_meta[moff] = 0.0f;
      split_meta[moff + 1] = 1.0f;
    }
    __syncthreads();
    if (all_masked_sh) {
      for (unsigned long long j = tid; j < total_seq; j += nthreads) {
        scores[srow + j] = 0.0f;
      }
    } else {
      const float inv = inv_sum_sh;
      for (unsigned long long j = tid; j < total_seq; j += nthreads) {
        scores[srow + j] *= inv;
      }
    }
    __syncthreads();
    for (unsigned long long c = tid; c < v_head_size; c += nthreads) {
      float acc = 0.0f;
      for (unsigned long long j = 0; j < total_seq; ++j) {
        const unsigned long long voff = ((b * kv_heads + kvh) * cap + j) * v_head_size;
        acc += scores[srow + j] * load_float(value, voff + c, dtype);
      }
      split_out[obase + c] = acc;
    }
    return;
  }

  // Multi-split path (wide context): chunk-local online softmax with a
  // block-parallel max/sum reduction, unnormalized P·V, merged by the combine.
  __shared__ float red[256];
  float local_max = NEG_INF;
  for (unsigned long long j = chunk_start + tid; j < chunk_end; j += nthreads) {
    local_max = fmaxf(local_max, scores[srow + j]);
  }
  red[tid] = local_max;
  __syncthreads();
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red[tid] = fmaxf(red[tid], red[tid + stride]);
    }
    __syncthreads();
  }
  const float m = red[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (unsigned long long j = chunk_start + tid; j < chunk_end; j += nthreads) {
    const float e = (m == NEG_INF) ? 0.0f : expf(scores[srow + j] - m);
    scores[srow + j] = e;
    local_sum += e;
  }
  red[tid] = local_sum;
  __syncthreads();
  for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red[tid] += red[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    split_meta[moff] = m;
    split_meta[moff + 1] = red[0];
  }
  __syncthreads();

  // Unnormalized partial P·V over this chunk (ascending keys per channel).
  for (unsigned long long c = tid; c < v_head_size; c += nthreads) {
    float acc = 0.0f;
    for (unsigned long long j = chunk_start; j < chunk_end; ++j) {
      const unsigned long long voff = ((b * kv_heads + kvh) * cap + j) * v_head_size;
      acc += scores[srow + j] * load_float(value, voff + c, dtype);
    }
    split_out[obase + c] = acc;
  }
}

// FlashDecoding combine: one block per row merges the per-split partials with a
// numerically-stable log-sum-exp rescale. Uniform for both the single-split
// (sentinel max=0/sum=1 => bit-exact pass-through) and multi-split partials.
extern "C" __global__ void attention_combine(
    const float* split_out, const float* split_meta, void* y,
    unsigned long long batch, unsigned long long q_heads, unsigned long long q_seq,
    unsigned long long v_head_size,
    int dtype, int out_is_3d, unsigned long long num_splits) {
  const unsigned long long row = blockIdx.x;
  const unsigned long long total_rows = batch * q_heads * q_seq;
  if (row >= total_rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const unsigned long long i = row % q_seq;
  unsigned long long rem = row / q_seq;
  const unsigned long long qh = rem % q_heads;
  const unsigned long long b = rem / q_heads;

  __shared__ float gmax_sh;
  __shared__ float gsum_sh;
  if (tid == 0) {
    float gmax = NEG_INF;
    for (unsigned long long s = 0; s < num_splits; ++s) {
      gmax = fmaxf(gmax, split_meta[(row * num_splits + s) * 2]);
    }
    float gsum = 0.0f;
    if (gmax != NEG_INF) {
      for (unsigned long long s = 0; s < num_splits; ++s) {
        const float ms = split_meta[(row * num_splits + s) * 2];
        const float ls = split_meta[(row * num_splits + s) * 2 + 1];
        gsum += ls * expf(ms - gmax);
      }
    }
    gmax_sh = gmax;
    gsum_sh = gsum;
  }
  __syncthreads();
  const float gmax = gmax_sh;
  const float gsum = gsum_sh;
  const unsigned long long ybase = out_is_3d
      ? (b * q_seq + i) * (q_heads * v_head_size) + qh * v_head_size
      : ((b * q_heads + qh) * q_seq + i) * v_head_size;
  if (gmax == NEG_INF || gsum == 0.0f) {
    for (unsigned long long c = tid; c < v_head_size; c += nthreads) {
      store_float(y, ybase + c, 0.0f, dtype);
    }
    return;
  }
  const float inv = 1.0f / gsum;
  for (unsigned long long c = tid; c < v_head_size; c += nthreads) {
    float acc = 0.0f;
    for (unsigned long long s = 0; s < num_splits; ++s) {
      const float ms = split_meta[(row * num_splits + s) * 2];
      const float w = expf(ms - gmax);
      acc += split_out[(row * num_splits + s) * v_head_size + c] * w;
    }
    store_float(y, ybase + c, acc * inv, dtype);
  }
}
"#;

/// Return the claim-time denial for Attention's positional input contract.
///
/// An omitted optional input is represented by [`DataType::Undefined`]. Keep
/// that distinct from a supplied tensor so absent mask/cache/length slots are
/// accepted without weakening dtype checks for tensors that are present.
pub(crate) fn unsupported_reason(
    node: &Node,
    opset: u64,
    input_dtypes: &[DataType],
    q_rank: Option<usize>,
) -> Option<Cow<'static, str>> {
    if let Err(error) = validate_head_attributes(node, opset, q_rank) {
        return Some(Cow::Owned(error.to_string()));
    }
    if let Err(error) = AttentionWindow::from_node(node, opset) {
        return Some(Cow::Owned(error.to_string()));
    }
    let dtype_at = |index: usize| {
        input_dtypes
            .get(index)
            .copied()
            .unwrap_or(DataType::Undefined)
    };
    let floating_denial = |dtype| {
        let dtype = match dtype {
            DataType::Float16 => "f16".into(),
            DataType::BFloat16 => "bf16".into(),
            other => format!("{other:?}"),
        };
        Cow::Owned(format!(
            "Attention: dtype {dtype} not supported on CUDA (supported: f32, f16, bf16)"
        ))
    };

    for index in 0..3 {
        let dtype = dtype_at(index);
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Some(floating_denial(dtype));
        }
    }
    if dtype_at(1) != dtype_at(0) || dtype_at(2) != dtype_at(0) {
        return Some(Cow::Borrowed(
            "Attention: Q, K, and V must use the same floating dtype on CUDA",
        ));
    }

    let mask_dtype = dtype_at(3);
    if !matches!(
        mask_dtype,
        DataType::Undefined
            | DataType::Bool
            | DataType::Float32
            | DataType::Float16
            | DataType::BFloat16
    ) {
        return Some(Cow::Owned(format!(
            "Attention: attn_mask dtype {mask_dtype:?} not supported (expected bool, f32, f16, or bf16 when provided)"
        )));
    }
    let past_key_dtype = dtype_at(4);
    let past_value_dtype = dtype_at(5);
    for dtype in [past_key_dtype, past_value_dtype] {
        if dtype != DataType::Undefined
            && !matches!(
                dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            )
        {
            return Some(floating_denial(dtype));
        }
        if dtype != DataType::Undefined && dtype != dtype_at(0) {
            return Some(Cow::Borrowed(
                "Attention: Q/K/V and past_key/past_value must use the same floating dtype on CUDA",
            ));
        }
    }
    let has_past_key = past_key_dtype != DataType::Undefined;
    let has_past_value = past_value_dtype != DataType::Undefined;
    if has_past_key != has_past_value {
        return Some(Cow::Borrowed(
            "Attention: past_key and past_value must be provided together",
        ));
    }

    let nonpad_dtype = dtype_at(6);
    if !matches!(nonpad_dtype, DataType::Undefined | DataType::Int64) {
        return Some(Cow::Owned(format!(
            "Attention: nonpad_kv_seqlen dtype {nonpad_dtype:?} not supported (expected int64 when provided)"
        )));
    }
    let has_nonpad = nonpad_dtype != DataType::Undefined;
    if has_nonpad && opset < 24 {
        return Some(Cow::Borrowed(
            "Attention: nonpad_kv_seqlen was added in opset 24 and is not valid for opset 23",
        ));
    }
    if has_nonpad && has_past_key {
        return Some(Cow::Borrowed(
            "Attention: nonpad_kv_seqlen must not be used together with past_key/past_value",
        ));
    }

    None
}

/// f32/f16/bf16 standard-`Attention` kernel carrying the resolved attributes.
pub struct StandardAttentionKernel {
    runtime: Arc<CudaRuntime>,
    /// Explicit score scale; `None` → default `1/sqrt(head_size)`.
    scale: Option<f32>,
    is_causal: bool,
    window: AttentionWindow,
    q_num_heads: Option<usize>,
    kv_num_heads: Option<usize>,
    qk_matmul_output_mode: i64,
    /// Softcap value; `0.0` disables it.
    softcap: f32,
    /// Compiled positional output presence; omitted outputs need temporary KV
    /// backing and must not be confused with unresolved output shapes.
    output_slots: [bool; 4],
    /// The registered schema revision this kernel serves (23, 24 or 25).
    /// Controls `nonpad_kv_seqlen` acceptance (opset 24+ only).
    since_version: u32,
    /// Only capture readiness is retained; scratch belongs to the caller lane.
    warm_state: Mutex<Option<StdAttnCaptureSignature>>,
}

/// Decode signature warmed as capture-safe. A subsequent capture pass reuses
/// the borrowed workspace layout prepared for this shape.
#[derive(Clone, Debug, PartialEq, Eq)]
struct StdAttnCaptureSignature {
    dtype: DataType,
    inputs: Vec<(DataType, Vec<usize>, bool)>,
    outputs: Vec<(DataType, Vec<usize>)>,
    batch: usize,
    q_heads: usize,
    kv_heads: usize,
    q_seq: usize,
    key_cap: usize,
    head_size: usize,
    v_head_size: usize,
    workspace_base: usize,
    workspace_bytes: usize,
    split: Option<(u64, u64)>,
    default_offset: i64,
}

/// Alignment of the governed default-domain `Attention` score buffer (#736),
/// matching the executor's 256-byte device-allocation granularity.
const STD_SCORES_ALIGN: usize = 256;

#[cfg(test)]
const SCORES: usize = 0;
#[cfg(test)]
const DEVICE_LENGTH: usize = 1;
#[cfg(test)]
const OFFSETS: usize = 2;
#[cfg(test)]
const PAD_LIMITS: usize = 3;
#[cfg(test)]
const STAGED_KEY: usize = 4;
#[cfg(test)]
const STAGED_VALUE: usize = 5;
#[cfg(test)]
const PRESENT_KEY: usize = 6;
#[cfg(test)]
const PRESENT_VALUE: usize = 7;
#[cfg(test)]
const SPLIT_PARTIALS: usize = 8;
type StdAttentionWorkspaceLayout = WorkspaceLayout<9>;

/// Byte size of the governed default-domain `Attention` f32 score matrix for one
/// concrete geometry: `batch·q_heads·q_seq·total_seq` fp32 scores. Prepare-only
/// planning (`Kernel::workspace_requirement`) and execution (`run`) size the
/// reservation through this identical helper so the reserved and consumed byte
/// counts cannot drift; a degenerate geometry still reserves one element,
/// matching the kernel's historical `qk_expected * 4` under `bytes.max(1)`.
fn std_attention_scores_bytes(
    batch: usize,
    q_heads: usize,
    q_seq: usize,
    total_seq: usize,
) -> Result<usize> {
    let rows = batch
        .checked_mul(q_heads)
        .and_then(|value| value.checked_mul(q_seq))
        .ok_or_else(|| EpError::KernelFailed("Attention: attention row count overflow".into()))?;
    let elements = rows
        .checked_mul(total_seq)
        .ok_or_else(|| EpError::KernelFailed("Attention: score scratch size overflow".into()))?;
    elements
        .max(1)
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| EpError::KernelFailed("Attention: score scratch byte count overflow".into()))
}

fn std_attention_elements(factors: &[usize], region: &str) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, factor| {
        product.checked_mul(*factor).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Attention: {region} element count overflows for {factors:?}; reduce the shape"
            ))
        })
    })
}

fn std_attention_stage_bytes(elements: usize, element_bytes: usize) -> Result<usize> {
    elements
        .checked_mul(element_bytes)
        .map(|bytes| bytes.max(1))
        .ok_or_else(|| EpError::KernelFailed("Attention: staged KV byte count overflow".into()))
}

/// Shared layout for the default-domain `Attention` governed composite:
/// always-materialized fp32 scores followed by only the dense aliased K/V
/// staging regions this route can use. Planning and execution both call this
/// helper, so their byte formulas and offsets cannot drift.
#[allow(clippy::too_many_arguments)]
fn std_attention_workspace_layout(
    batch: usize,
    q_heads: usize,
    q_seq: usize,
    total_seq: usize,
    kv_heads: usize,
    key_seq: usize,
    head_size: usize,
    value_seq: usize,
    value_head_size: usize,
    element_bytes: usize,
    stage_key: bool,
    stage_value: bool,
    present_key_scratch: bool,
    present_value_scratch: bool,
    control_batch: usize,
    device_length: bool,
    split_bytes: usize,
) -> Result<StdAttentionWorkspaceLayout> {
    let scores = std_attention_scores_bytes(batch, q_heads, q_seq, total_seq)?;
    let key = std_attention_elements(&[batch, kv_heads, key_seq, head_size], "key workspace")?;
    let value = std_attention_elements(
        &[batch, kv_heads, value_seq, value_head_size],
        "value workspace",
    )?;
    let key_bytes = std_attention_stage_bytes(key, element_bytes)?;
    let value_bytes = std_attention_stage_bytes(value, element_bytes)?;
    let controls = control_batch.max(1).checked_mul(8).ok_or_else(|| {
        EpError::KernelFailed("Attention: control workspace size overflow; reduce the batch".into())
    })?;
    let sizes = [
        scores,
        if device_length { 4 } else { 0 },
        controls,
        controls,
        if stage_key { key_bytes } else { 0 },
        if stage_value { value_bytes } else { 0 },
        if present_key_scratch { key_bytes } else { 0 },
        if present_value_scratch {
            value_bytes
        } else {
            0
        },
        split_bytes,
    ];
    let mut regions = [std::alloc::Layout::new::<()>(); 9];
    for (index, bytes) in sizes.into_iter().enumerate() {
        regions[index] = std::alloc::Layout::from_size_align(bytes, STD_SCORES_ALIGN).map_err(|_| EpError::KernelFailed(format!("Attention: workspace region {index} exceeds addressable capacity ({bytes} bytes); reduce the geometry")))?;
    }
    WorkspaceLayout::new(regions)
}

/// Report the exact governed composite scratch a default-domain `Attention`
/// dispatch can materialize (#736), for prepare-only planning. The kernel has
/// **no** shared-memory/flash route: `attention_row` always stages the
/// `batch·q_heads·q_seq·total_seq` scores in device memory (fp32 accumulate,
/// regardless of the f32/f16/bf16 operand dtype). Dense aliased K/V growth adds
/// disjoint `present_*_expected·element_bytes` regions; fixed-capacity append,
/// no-past, and absent-present-output routes add zero staging bytes. Only
/// unresolvable or non-float input metadata returns NONE, letting `run` raise the
/// precise error.
///
/// One SessionPersistent lane retains the peak across fitting calls, including
/// deferred execution. Regions are disjoint byte ranges, not separate grants.
fn std_attention_workspace_requirement(
    inputs: &[TensorMetadata<'_>],
    q_num_heads: Option<usize>,
    kv_num_heads: Option<usize>,
    mask_can_supply_length: bool,
    is_causal: bool,
    output_slots: [bool; 4],
    outputs: &[OutputMetadata<'_>],
) -> Result<WorkspaceRequirement> {
    let (Some(q), Some(k), Some(v)) = (inputs.first(), inputs.get(1), inputs.get(2)) else {
        return Ok(WorkspaceRequirement::NONE);
    };
    if !matches!(
        q.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) || !q.present
        || !k.present
        || !v.present
        || k.dtype != q.dtype
        || v.dtype != q.dtype
    {
        return Ok(WorkspaceRequirement::NONE);
    }
    let Some((batch, q_heads, q_seq, head_size)) = bhsd_from_meta(q.shape, q_num_heads) else {
        return Ok(WorkspaceRequirement::NONE);
    };
    let Some((k_batch, kv_heads, k_seq, k_head_size)) = bhsd_from_meta(k.shape, kv_num_heads)
    else {
        return Ok(WorkspaceRequirement::NONE);
    };
    let Some((v_batch, v_heads, v_seq, value_head_size)) = bhsd_from_meta(v.shape, kv_num_heads)
    else {
        return Ok(WorkspaceRequirement::NONE);
    };
    if k_batch != batch
        || v_batch != batch
        || v_heads != kv_heads
        || k_head_size != head_size
        || k_seq != v_seq
    {
        return Ok(WorkspaceRequirement::NONE);
    }
    // Total attended length = past-cache length + current key length. The past
    // key (input 4) is optional; when a frozen fixed-capacity cache binds it at
    // physical capacity this over-estimates the valid length by the padding
    // rows, which is safe (reserve ≥ consume) — execution re-derives the exact
    // size through the same helper and refuses on any shortfall, so it never
    // silently under-allocates.
    let past_key = inputs
        .get(4)
        .filter(|past| past.present)
        .and_then(|past| bhsd_from_meta(past.shape, kv_num_heads));
    let past_value = inputs
        .get(5)
        .filter(|past| past.present)
        .and_then(|past| bhsd_from_meta(past.shape, kv_num_heads));
    if inputs.get(4).is_some_and(|past| past.present) != past_key.is_some()
        || inputs.get(5).is_some_and(|past| past.present) != past_value.is_some()
        || past_key.is_some() != past_value.is_some()
    {
        return Ok(WorkspaceRequirement::NONE);
    }
    let key_past_seq = past_key.map(|(_, _, seq, _)| seq).unwrap_or(0);
    let value_past_seq = past_value.map(|(_, _, seq, _)| seq).unwrap_or(0);
    let total_seq = key_past_seq
        .checked_add(k_seq)
        .ok_or_else(|| EpError::KernelFailed("Attention: total attended length overflow".into()))?;
    let value_total_seq = value_past_seq.checked_add(v_seq).ok_or_else(|| {
        EpError::KernelFailed("Attention: total value-cache length overflow".into())
    })?;
    if total_seq != value_total_seq {
        return Ok(WorkspaceRequirement::NONE);
    }
    validate_std_attention_outputs(output_slots, outputs, q.dtype)?;
    let output_capacity = |index: usize| match outputs.get(index) {
        Some(OutputMetadata::Tensor { shape, .. }) if shape.len() == 4 => Some(shape[2]),
        _ => None,
    };
    let key_capacity = output_capacity(1);
    let value_capacity = output_capacity(2);
    let frozen = mask_can_supply_length
        && past_key.is_some()
        && past_value.is_some()
        && key_capacity.is_some_and(|cap| cap > 0 && cap <= key_past_seq)
        && value_capacity.is_some_and(|cap| cap > 0 && cap <= value_past_seq);
    let key_cap = key_capacity
        .or(value_capacity)
        .unwrap_or(total_seq)
        .max(if frozen { 0 } else { total_seq });
    let value_cap = value_capacity
        .or(key_capacity)
        .unwrap_or(value_total_seq)
        .max(if frozen { 0 } else { value_total_seq });
    if key_cap != value_cap {
        return Err(EpError::KernelFailed(format!(
            "Attention: key/value physical capacities {key_cap}/{value_cap} differ; this kernel requires the same per-head sequence capacity for both caches"
        )));
    }
    let attended = if frozen { key_cap } else { total_seq };
    // A mask matching the past extent cannot prove an unresolved output is
    // frozen. Each declared output may still alias a growing dense cache.
    let stage_key = output_slots[1]
        && past_key.is_some()
        && !frozen
        && key_capacity.is_none_or(|cap| cap <= total_seq);
    let stage_value = output_slots[2]
        && past_value.is_some()
        && !frozen
        && value_capacity.is_none_or(|cap| cap <= value_total_seq);
    let mask_present = inputs.get(3).is_some_and(|mask| mask.present);
    // Pointer aliasing is unavailable to planning. Fund the selected split
    // whenever fixed-stride append could use it; execution may consume less.
    let device_length = mask_can_supply_length
        && past_key.is_some()
        && past_value.is_some()
        && mask_present
        && output_slots[1]
        && output_slots[2];
    let rows = batch
        .checked_mul(q_heads)
        .and_then(|rows| rows.checked_mul(q_seq))
        .ok_or_else(|| {
            EpError::KernelFailed(
                "Attention: planned row count overflow; reduce the geometry".into(),
            )
        })?;
    let split = attention_split_config(
        q_seq == 1,
        device_length,
        is_causal,
        output_slots[3],
        key_cap as u64,
        rows as u64,
        value_head_size as u64,
    );
    let split_bytes = split
        .map(|(splits, _)| {
            attention_split_scratch_bytes(splits, rows as u64, value_head_size as u64)
        })
        .transpose()?
        .unwrap_or(0);
    let present_key_scratch = !output_slots[1];
    let present_value_scratch = !output_slots[2];
    let layout = std_attention_workspace_layout(
        batch,
        q_heads,
        q_seq,
        attended,
        kv_heads,
        key_cap,
        head_size,
        value_cap,
        value_head_size,
        q.dtype.byte_size(),
        stage_key,
        stage_value,
        present_key_scratch,
        present_value_scratch,
        batch,
        device_length,
        split_bytes,
    )?;
    Ok(WorkspaceRequirement {
        bytes: u64::try_from(layout.bytes()).map_err(|_| {
            EpError::KernelFailed("Attention: composite workspace does not fit u64".into())
        })?,
        alignment: STD_SCORES_ALIGN,
        lifetime: WorkspaceLifetime::SessionPersistent,
    })
}

fn std_attention_output_slots(node: &Node) -> Result<[bool; 4]> {
    if !(1..=4).contains(&node.outputs.len()) || node.outputs[0].is_none() {
        return Err(EpError::KernelFailed(
            "Attention: declare required Y and at most three positional optional output slots"
                .into(),
        ));
    }
    Ok(std::array::from_fn(|index| {
        node.outputs.get(index).is_some_and(Option::is_some)
    }))
}

fn validate_std_attention_outputs(
    slots: [bool; 4],
    outputs: &[OutputMetadata<'_>],
    dtype: DataType,
) -> Result<()> {
    let actual = std::array::from_fn(|index| {
        outputs
            .get(index)
            .is_some_and(|output| !matches!(output, OutputMetadata::Absent))
    });
    if outputs.len() > 4 || actual != slots {
        return Err(EpError::KernelFailed(format!(
            "Attention: runtime output slots {actual:?} differ from declared {slots:?}; pass the declared positional Y/present-key/present-value/QK outputs"
        )));
    }
    for (index, output) in outputs.iter().enumerate() {
        let actual_dtype = match output {
            OutputMetadata::Absent => continue,
            OutputMetadata::Unresolved { dtype } | OutputMetadata::Tensor { dtype, .. } => *dtype,
        };
        if actual_dtype != dtype {
            return Err(EpError::KernelFailed(format!(
                "Attention: output slot {index} dtype {actual_dtype:?} differs from input {dtype:?}; use the attention input dtype"
            )));
        }
    }
    Ok(())
}

/// Factory for [`StandardAttentionKernel`], reading the standard-`Attention`
/// attributes. `since_version` selects the opset semantics.
pub struct StandardAttentionFactory {
    pub runtime: Arc<CudaRuntime>,
    pub since_version: u32,
}

fn validate_head_attributes(node: &Node, opset: u64, q_rank: Option<usize>) -> Result<()> {
    validate_attention_head_attributes(
        opset,
        q_rank,
        node.attr("q_num_heads").is_some(),
        node.attr("kv_num_heads").is_some(),
    )
}

impl KernelFactory for StandardAttentionFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        validate_head_attributes(
            node,
            u64::from(self.since_version),
            input_shapes.first().map(Vec::len),
        )?;
        let scale = node.attr("scale").and_then(|a| a.as_float());
        let is_causal = node.attr("is_causal").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let q_num_heads = node
            .attr("q_num_heads")
            .and_then(|a| a.as_int())
            .map(|v| v as usize);
        let kv_num_heads = node
            .attr("kv_num_heads")
            .and_then(|a| a.as_int())
            .map(|v| v as usize);
        let qk_matmul_output_mode = node
            .attr("qk_matmul_output_mode")
            .and_then(|a| a.as_int())
            .unwrap_or(0);
        let softcap = node
            .attr("softcap")
            .and_then(|a| a.as_float())
            .unwrap_or(0.0);
        if !(0..=3).contains(&qk_matmul_output_mode) {
            return Err(EpError::KernelFailed(format!(
                "Attention: qk_matmul_output_mode {qk_matmul_output_mode} is not supported \
                 (only 0, 1, 2, 3 are implemented)"
            )));
        }
        Ok(Box::new(StandardAttentionKernel {
            runtime: self.runtime.clone(),
            scale,
            is_causal,
            window: AttentionWindow::from_node(node, u64::from(self.since_version))?,
            q_num_heads,
            kv_num_heads,
            qk_matmul_output_mode,
            softcap,
            output_slots: std_attention_output_slots(node)?,
            since_version: self.since_version,
            warm_state: Mutex::new(None),
        }))
    }
}

fn check_arity(
    name: &str,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    min: usize,
    max: usize,
    min_outputs: usize,
) -> Result<()> {
    if !(min..=max).contains(&inputs.len()) || outputs.len() < min_outputs {
        return Err(EpError::KernelFailed(format!(
            "{name}: expected {min}..={max} inputs and at least {min_outputs} outputs"
        )));
    }
    Ok(())
}

/// Resolved `[batch, heads, seq, dim]` view of a Q/K/V input, keeping the input
/// on the device (no host copy). A 3D `(batch, seq, heads·dim)` input records
/// `is_3d` so the on-device gather can reshape it into heads.
struct BhsdDims {
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
    is_3d: bool,
}

/// Resolve a Q/K/V input's `[batch, heads, seq, dim]` dims without copying data.
///
/// A 4D input `(batch, heads, seq, dim)` is read as-is. A 3D input
/// `(batch, seq, heads·dim)` reshapes to heads via `num_heads` (from the
/// `q_num_heads`/`kv_num_heads` attributes), which is required and must divide
/// the hidden size. Non-contiguous inputs are rejected (the kernel requests
/// contiguous inputs).
fn resolve_bhsd(view: &TensorView, name: &str, num_heads: Option<usize>) -> Result<BhsdDims> {
    if !view.is_contiguous() {
        return Err(EpError::KernelFailed(
            "Attention: non-contiguous inputs are not supported".into(),
        ));
    }
    if !matches!(
        view.dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Err(EpError::KernelFailed(format!(
            "Attention: expected f32, f16, or bf16 input, got {:?}",
            view.dtype
        )));
    }
    let shape = view.shape;
    match shape.len() {
        4 => Ok(BhsdDims {
            batch: shape[0],
            heads: shape[1],
            seq: shape[2],
            dim: shape[3],
            is_3d: false,
        }),
        3 => {
            let heads = num_heads.ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Attention: 3D {name} input requires the corresponding \
                     q_num_heads/kv_num_heads attribute"
                ))
            })?;
            if heads == 0 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: {name} num_heads must be > 0"
                )));
            }
            let (batch, seq, hidden) = (shape[0], shape[1], shape[2]);
            if hidden % heads != 0 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: 3D {name} hidden size {hidden} is not divisible by num_heads \
                     {heads}"
                )));
            }
            Ok(BhsdDims {
                batch,
                heads,
                seq,
                dim: hidden / heads,
                is_3d: true,
            })
        }
        other => Err(EpError::KernelFailed(format!(
            "Attention: {name} must be rank 3 or 4, got rank {other}"
        ))),
    }
}

/// Resolve a Q/K/V input's `[batch, heads, seq, dim]` dims from static shape
/// metadata (no device access), mirroring [`resolve_bhsd`] for prepare-only
/// workspace planning. Returns `None` when the shape is not a resolvable
/// rank-3/4 attention operand, in which case the caller charges no workspace and
/// lets execution raise the precise error.
fn bhsd_from_meta(
    shape: &[usize],
    num_heads: Option<usize>,
) -> Option<(usize, usize, usize, usize)> {
    match shape.len() {
        4 => Some((shape[0], shape[1], shape[2], shape[3])),
        3 => {
            let heads = num_heads?;
            if heads == 0 || shape[2] == 0 || !shape[2].is_multiple_of(heads) {
                return None;
            }
            Some((shape[0], heads, shape[1], shape[2] / heads))
        }
        _ => None,
    }
}

fn dense_i64(runtime: &CudaRuntime, view: &TensorView) -> Result<Vec<i64>> {
    if view.dtype != DataType::Int64 {
        return Err(EpError::KernelFailed(
            "Attention: nonpad_kv_seqlen must be int64".into(),
        ));
    }
    if !view.is_contiguous() {
        return Err(EpError::KernelFailed(
            "Attention: non-contiguous inputs are not supported".into(),
        ));
    }
    let mut bytes = vec![0u8; view.dtype.storage_bytes(view.numel())];
    unsafe {
        runtime.dtoh(&mut bytes, cuptr(view.data_ptr::<u8>() as *const c_void))?;
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

/// Validate an output slot (contiguous requested dtype with the expected element count) and
/// return its device pointer.
fn output_ptr(output: &mut TensorMut, dtype: DataType, expected: usize) -> Result<CUdeviceptr> {
    if output.dtype != dtype || !output.is_contiguous() || output.numel() != expected {
        return Err(EpError::KernelFailed(
            "Attention: output must be contiguous and use the input dtype with the expected shape"
                .into(),
        ));
    }
    Ok(cuptr(output.data_ptr_mut::<u8>() as *const c_void))
}

/// Mask kind + right-aligned broadcast dims passed to the device kernel.
struct MaskMeta {
    ptr: CUdeviceptr,
    kind: i32,
    rank: i32,
    dims: [u64; 4],
}

impl StandardAttentionKernel {
    /// Launch `build_kv` to gather a K/V input (plus an optional past cache)
    /// into a contiguous `[batch, heads, total_seq, dim]` present buffer at
    /// `out_ptr`.
    #[allow(clippy::too_many_arguments)]
    fn launch_build_kv(
        &self,
        past_ptr: CUdeviceptr,
        cur_ptr: CUdeviceptr,
        out_ptr: CUdeviceptr,
        has_past: bool,
        cur_is_3d: bool,
        past_is_3d: bool,
        dtype: i32,
        batch: usize,
        heads: usize,
        past_seq: usize,
        cur_seq: usize,
        total_seq: usize,
        dim: usize,
        out_cap: usize,
        past_cap: usize,
        write_start: usize,
        dev_len: CUdeviceptr,
    ) -> Result<()> {
        // With a device length the append rebuilds exactly `cur_seq` rows into
        // their fixed slot, so the element count (and thus grid geometry) is
        // host-constant regardless of the growing valid length.
        let span = if dev_len != 0 {
            cur_seq
        } else {
            total_seq.saturating_sub(write_start)
        };
        let elements =
            std_attention_elements(&[batch, heads, span, dim], "build-KV launch")? as u64;
        if elements == 0 {
            return Ok(());
        }
        let func = self
            .runtime
            .nvrtc_function(ATTENTION_MODULE, ATTENTION_SOURCE, "build_kv")?;
        let has_past_i = i32::from(has_past);
        let cur_is_3d_i = i32::from(cur_is_3d);
        let past_is_3d_i = i32::from(past_is_3d);
        let batch = batch as u64;
        let heads = heads as u64;
        let past_seq = past_seq as u64;
        let cur_seq = cur_seq as u64;
        let total_seq = total_seq as u64;
        let dim = dim as u64;
        let out_cap = out_cap as u64;
        let past_cap = past_cap as u64;
        let write_start = write_start as u64;
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&past_ptr)
            .arg(&cur_ptr)
            .arg(&out_ptr)
            .arg(&dtype)
            .arg(&has_past_i)
            .arg(&cur_is_3d_i)
            .arg(&past_is_3d_i)
            .arg(&batch)
            .arg(&heads)
            .arg(&past_seq)
            .arg(&cur_seq)
            .arg(&total_seq)
            .arg(&dim)
            .arg(&out_cap)
            .arg(&past_cap)
            .arg(&write_start)
            .arg(&elements)
            .arg(&dev_len);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch build_kv", error))
        .map(|_| ())
    }

    /// Scan the additive attention-mask bias for its valid-length frontier and
    /// write it to `out_len` (a device `i32`). One launch of a single thread;
    /// used by the capture-safe decode path so the growing length is read from
    /// device memory rather than host shape metadata.
    fn launch_derive_len(
        &self,
        mask_ptr: CUdeviceptr,
        mask_kind: i32,
        key_len: u64,
        row_base: u64,
        out_len: CUdeviceptr,
    ) -> Result<()> {
        let func = self
            .runtime
            .nvrtc_function(ATTENTION_MODULE, ATTENTION_SOURCE, "derive_len")?;
        let mut builder = self.runtime.stream().launch_builder(&func);
        builder
            .arg(&mask_ptr)
            .arg(&mask_kind)
            .arg(&key_len)
            .arg(&row_base)
            .arg(&out_len);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch derive_len", error))
        .map(|_| ())
    }
}

impl StandardAttentionKernel {
    fn validate_capture_signature(
        state: &Option<StdAttnCaptureSignature>,
        signature: &StdAttnCaptureSignature,
    ) -> Result<()> {
        let ready = state.as_ref().ok_or_else(|| {
            EpError::KernelFailed(
                "Attention: CUDA graph capture began without a successful warmed decode \
                 signature. HOW: run the exact fixed-capacity decode call eagerly before \
                 capture."
                    .into(),
            )
        })?;
        if *ready != *signature {
            return Err(EpError::KernelFailed(format!(
                "Attention: signature changed during CUDA graph capture: warmed={ready:?}, \
                 current={signature:?}. HOW: abort capture and warm the exact replacement.",
            )));
        }
        Ok(())
    }

    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        prepared: Option<WorkspaceView>,
    ) -> Result<()> {
        let mut warm_state = self.warm_state.lock().map_err(|_| {
            EpError::KernelFailed(
                "Attention: warm-state lock poisoned; recreate the kernel before execution".into(),
            )
        })?;
        // Invalidate before validation or submission. Only a successful exact
        // invocation republishes readiness; errors cannot retain stale proof.
        let prior_signature = warm_state.take();
        let capturing = self.runtime.is_capturing()?;
        check_arity("Attention", inputs, outputs, 3, 7, 1)?;
        if outputs.len() > 4 {
            return Err(EpError::KernelFailed(format!(
                "Attention: received {} output slots, expected at most four positional outputs; use the declared Y/present-key/present-value/QK slots",
                outputs.len()
            )));
        }
        let output_metadata: [_; 4] = std::array::from_fn(|index| {
            outputs
                .get(index)
                .map(OutputMetadata::from_view)
                .unwrap_or(OutputMetadata::Absent)
        });
        validate_std_attention_outputs(self.output_slots, &output_metadata, inputs[0].dtype)?;
        // Inputs may have been uploaded asynchronously on the EP stream. During
        // CUDA-graph capture the uploads are recorded into the graph, so no host
        // synchronize is issued (and none is legal); ordering is preserved by
        // the stream the capture records.
        if !capturing {
            self.runtime.synchronize()?;
        }

        let q_rank = inputs[0].shape.len();
        validate_attention_head_attributes(
            u64::from(self.since_version),
            Some(q_rank),
            self.q_num_heads.is_some(),
            self.kv_num_heads.is_some(),
        )?;
        let q = resolve_bhsd(&inputs[0], "Q", self.q_num_heads)?;
        let k_cur = resolve_bhsd(&inputs[1], "K", self.kv_num_heads)?;
        let v_cur = resolve_bhsd(&inputs[2], "V", self.kv_num_heads)?;
        let dtype = inputs[0].dtype;
        if inputs[1].dtype != dtype || inputs[2].dtype != dtype {
            return Err(EpError::KernelFailed(
                "Attention: Q, K, and V must use the same floating dtype on CUDA".into(),
            ));
        }
        let dtype_code = match dtype {
            DataType::Float32 => 0,
            DataType::Float16 => 1,
            _ => 2,
        };
        let element_bytes = dtype.storage_bytes(1);

        // Optional past KV cache (inputs 4 and 5). They must be used together.
        // Presence is decided by input-slot binding (a null "absent" view for an
        // omitted optional input), NOT by an empty shape — a genuinely present
        // rank-0 tensor also has an empty shape but must not be treated as
        // absent.
        let has_past_key = inputs.len() > 4 && !inputs[4].is_absent();
        let has_past_value = inputs.len() > 5 && !inputs[5].is_absent();
        if has_past_key != has_past_value {
            return Err(EpError::KernelFailed(
                "Attention: past_key and past_value must be provided together".into(),
            ));
        }
        let past_key = if has_past_key {
            Some(resolve_bhsd(&inputs[4], "past_key", self.kv_num_heads)?)
        } else {
            None
        };
        let past_value = if has_past_value {
            Some(resolve_bhsd(&inputs[5], "past_value", self.kv_num_heads)?)
        } else {
            None
        };
        if has_past_key && (inputs[4].dtype != dtype || inputs[5].dtype != dtype) {
            return Err(EpError::KernelFailed(
                "Attention: Q/K/V and past_key/past_value must use the same floating dtype on CUDA"
                    .into(),
            ));
        }
        let key_past_seq = past_key.as_ref().map(|p| p.seq).unwrap_or(0);
        let value_past_seq = past_value.as_ref().map(|p| p.seq).unwrap_or(0);

        // Preserve the concat compatibility checks (past vs current dims).
        if let Some(past) = &past_key
            && (past.batch != k_cur.batch || past.heads != k_cur.heads || past.dim != k_cur.dim)
        {
            return Err(EpError::KernelFailed(format!(
                "Attention: past_key dims (b={},h={},d={}) incompatible with current \
                 (b={},h={},d={})",
                past.batch, past.heads, past.dim, k_cur.batch, k_cur.heads, k_cur.dim
            )));
        }
        if let Some(past) = &past_value
            && (past.batch != v_cur.batch || past.heads != v_cur.heads || past.dim != v_cur.dim)
        {
            return Err(EpError::KernelFailed(format!(
                "Attention: past_value dims (b={},h={},d={}) incompatible with current \
                 (b={},h={},d={})",
                past.batch, past.heads, past.dim, v_cur.batch, v_cur.heads, v_cur.dim
            )));
        }

        // `nonpad_kv_seqlen` (7th input, opset 24+): per-batch count of valid
        // (non-padding) KV tokens, used when the KV cache lives outside the op.
        // It shifts the causal frontier by `nonpad_kv_seqlen[b] - q_seq` and is
        // mutually exclusive with an in-op past cache.
        let has_nonpad = inputs.len() > 6 && !inputs[6].is_absent();
        if has_nonpad && self.since_version < 24 {
            return Err(EpError::KernelFailed(
                "Attention: the optional `nonpad_kv_seqlen` input was added in opset 24 and is \
                 not valid for opset 23"
                    .into(),
            ));
        }
        if has_nonpad && (has_past_key || has_past_value) {
            return Err(EpError::KernelFailed(
                "Attention: `nonpad_kv_seqlen` must not be used together with past_key/past_value \
                 (external vs. in-op KV cache)"
                    .into(),
            ));
        }
        let nonpad_kv_seqlen: Option<Vec<i64>> = if has_nonpad {
            if capturing {
                return Err(EpError::KernelFailed("Attention: nonpad_kv_seqlen requires host validation; execute this route outside CUDA capture".into()));
            }
            let seqlen = dense_i64(&self.runtime, &inputs[6])?;
            if seqlen.len() != q.batch {
                return Err(EpError::KernelFailed(format!(
                    "Attention: nonpad_kv_seqlen length {} must equal batch_size {}",
                    seqlen.len(),
                    q.batch
                )));
            }
            Some(seqlen)
        } else {
            None
        };

        let batch = q.batch;
        let q_heads = q.heads;
        let q_seq = q.seq;
        let head_size = q.dim;
        let kv_heads = k_cur.heads;
        let total_seq = key_past_seq.checked_add(k_cur.seq).ok_or_else(|| {
            EpError::KernelFailed(
                "Attention: total key length overflow; reduce the cache shape".into(),
            )
        })?;
        let value_total_seq = value_past_seq.checked_add(v_cur.seq).ok_or_else(|| {
            EpError::KernelFailed(
                "Attention: total value length overflow; reduce the cache shape".into(),
            )
        })?;
        let v_head_size = v_cur.dim;

        if k_cur.dim != head_size {
            return Err(EpError::KernelFailed(format!(
                "Attention: Q head_size {head_size} != K head_size {}",
                k_cur.dim
            )));
        }
        if value_total_seq != total_seq {
            return Err(EpError::KernelFailed(format!(
                "Attention: present_key seq {total_seq} != present_value seq {value_total_seq}"
            )));
        }
        if k_cur.batch != batch || v_cur.batch != batch {
            return Err(EpError::KernelFailed(
                "Attention: Q, K, V must share the batch dimension".into(),
            ));
        }
        if kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(EpError::KernelFailed(format!(
                "Attention: q_num_heads {q_heads} must be a positive multiple of kv_num_heads \
                 {kv_heads} (MHA/GQA/MQA)"
            )));
        }
        let group = q_heads / kv_heads;

        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());
        // Fold `sqrt(scale)` into each Q and K operand so the dot product is
        // `(Q·√scale)·(K·√scale)` rather than `scale·(Q·K)`. This matches the
        // spec's `Q*sqrt(scale)`, `K*sqrt(scale)` pattern and avoids overflowing
        // an intermediate `Q·Kᵀ` for extreme magnitudes.
        let sqrt_scale = scale.sqrt();

        // Resolve the attention mask (input 3), if present. Its bytes stay on
        // the device; only the broadcast metadata is read on the host. Presence
        // is decided by input-slot binding, so a rank-0 (scalar) mask is honored
        // rather than mistaken for an omitted input.
        let mask = if inputs.len() > 3 && !inputs[3].is_absent() {
            let m = &inputs[3];
            if !m.is_contiguous() {
                return Err(EpError::KernelFailed(
                    "Attention: non-contiguous inputs are not supported".into(),
                ));
            }
            let rank = m.shape.len();
            if rank > 4 {
                return Err(EpError::KernelFailed(format!(
                    "Attention: attn_mask rank {rank} is not supported (max 4)"
                )));
            }
            // Right-align the mask dims against [b, h, i, j]; missing leading
            // axes broadcast (size 1).
            let mut dims = [1u64; 4];
            for (k, &d) in m.shape.iter().enumerate() {
                dims[4 - rank + k] = d as u64;
            }
            let kind = match m.dtype {
                DataType::Bool => 2,
                DataType::Float32 => 1,
                DataType::Float16 => 3,
                DataType::BFloat16 => 4,
                other => {
                    return Err(EpError::KernelFailed(format!(
                        "Attention: attn_mask dtype {other:?} not supported (expected bool, f32, f16, or bf16)"
                    )));
                }
            };
            MaskMeta {
                ptr: cuptr(m.data_ptr::<u8>() as *const c_void),
                kind,
                rank: rank as i32,
                dims,
            }
        } else {
            MaskMeta {
                ptr: 0,
                kind: 0,
                rank: 0,
                dims: [1u64; 4],
            }
        };

        // Validate output slots up front (before any device work) so shape
        // errors surface cleanly.
        let y_expected = std_attention_elements(&[batch, q_seq, q_heads, v_head_size], "Y")?;
        std_attention_stage_bytes(y_expected, element_bytes)?;
        let y_ptr = output_ptr(&mut outputs[0], dtype, y_expected)?;
        let want_present_key = self.output_slots[1];
        let want_present_value = self.output_slots[2];
        let want_qk = self.output_slots[3];
        // Physical per-head seq capacity of the bound present K/V slots (their
        // stride when exposed at fixed capacity). Zero for dense/unbound slots.
        let present_key_phys = if want_present_key && outputs[1].shape.len() == 4 {
            outputs[1].shape[2]
        } else {
            0
        };
        let present_value_phys = if want_present_value && outputs[2].shape.len() == 4 {
            outputs[2].shape[2]
        } else {
            0
        };
        // Frozen (fixed-capacity) KV binding: the past K/V *input* is bound at its
        // physical capacity too (extent == present capacity), so its tensor
        // extent no longer reports the valid past length — that length now lives
        // on-device (the attention-mask frontier scanned by `derive_len`). This
        // is what makes the decode step carry no growing logical input shape, so
        // whole-step CUDA-graph capture stays shape-static. The eager-growing
        // path (past extent < present capacity, exercised by the unit tests with
        // explicit shapes) keeps host-derived lengths and is left untouched.
        let kv_frozen = !self.window.is_active()
            && has_past_key
            && has_past_value
            && present_key_phys > 0
            && present_value_phys > 0
            && key_past_seq >= present_key_phys
            && value_past_seq >= present_value_phys;
        // Fixed-capacity KV: a bound present K/V output may be exposed at its
        // physical capacity (seq stride > valid length) so the cache lives at a
        // constant per-head slot and the new token is appended without
        // restriding the prior rows. `*_cap` is that per-head seq stride; it
        // collapses to the valid length for dense (non-capacity) present slots,
        // preserving the legacy contiguous layout. When the binding is frozen
        // the host cannot see the valid length, so it sizes everything to the
        // physical capacity and the device length bounds the actual compute.
        let (key_cap, value_cap, total_seq, value_total_seq) = if kv_frozen {
            (
                present_key_phys,
                present_value_phys,
                present_key_phys,
                present_value_phys,
            )
        } else {
            let key_cap = if want_present_key && outputs[1].shape.len() == 4 {
                outputs[1].shape[2].max(total_seq)
            } else {
                total_seq
            };
            let value_cap = if want_present_value && outputs[2].shape.len() == 4 {
                outputs[2].shape[2].max(value_total_seq)
            } else {
                value_total_seq
            };
            (key_cap, value_cap, total_seq, value_total_seq)
        };
        if key_cap != value_cap {
            return Err(EpError::KernelFailed(format!(
                "Attention: key/value physical capacities {key_cap}/{value_cap} differ; use the same per-head sequence capacity for both caches"
            )));
        }
        let present_key_expected =
            std_attention_elements(&[batch, kv_heads, key_cap, head_size], "present key")?;
        let present_value_expected =
            std_attention_elements(&[batch, kv_heads, value_cap, v_head_size], "present value")?;
        let qk_expected = std_attention_elements(&[batch, q_heads, q_seq, total_seq], "QK output")?;

        let present_key_out = if want_present_key {
            Some(output_ptr(&mut outputs[1], dtype, present_key_expected)?)
        } else {
            None
        };
        let present_value_out = if want_present_value {
            Some(output_ptr(&mut outputs[2], dtype, present_value_expected)?)
        } else {
            None
        };
        let qk_ptr = if want_qk {
            output_ptr(&mut outputs[3], dtype, qk_expected)?
        } else {
            0
        };

        // Q/K/V device pointers (bulk data stays on the device).
        let q_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let k_cur_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let v_cur_ptr = cuptr(inputs[2].data_ptr::<u8>() as *const c_void);
        let past_key_ptr = past_key
            .as_ref()
            .map(|_| cuptr(inputs[4].data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let past_value_ptr = past_value
            .as_ref()
            .map(|_| cuptr(inputs[5].data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);

        // Per-batch causal offset and padding frontier (built on the host, then
        // uploaded as small device arrays). Query in-block index `i` attends
        // key `j` iff `j <= i + offset`. With an external cache the offset is
        // `nonpad_kv_seqlen[b] - q_seq`; with an in-op past cache it is
        // `past_seq`; otherwise 0. A negative offset fully masks leading query
        // rows (→ zero output rows). The padding frontier masks keys at
        // `j >= nonpad_kv_seqlen[b]` regardless of causal mode; `-1` disables it.
        let (offsets, pad_limits) = if let Some(seqlen) = &nonpad_kv_seqlen {
            let query = i64::try_from(q_seq).map_err(|_| EpError::KernelFailed("Attention: query length exceeds signed control capacity; reduce the query shape".into()))?;
            let offsets = seqlen.iter().map(|length| length.checked_sub(query).ok_or_else(|| EpError::KernelFailed("Attention: nonpad length minus query length overflows; supply valid external cache lengths".into()))).collect::<Result<Vec<_>>>()?;
            (offsets, seqlen.clone())
        } else {
            (Vec::new(), Vec::new())
        };
        let default_offset = i64::try_from(key_past_seq).map_err(|_| EpError::KernelFailed("Attention: past length exceeds CUDA signed offset capacity; reduce the cache shape".into()))?;

        // In-place KV growth. The decode graph binds the present K/V output
        // onto the same buffer as the past K/V input.
        //
        // Fixed-capacity present (key_cap/value_cap > valid length): the
        // cache lives at a constant per-head stride, so `build_kv` appends
        // only the new token's rows into their fixed slot and leaves the
        // prior rows (already at that stride) untouched — no restride, no
        // cross-head overlap, race-free and deterministic in place. No
        // staging needed.
        //
        // Dense present (legacy): `build_kv` rewrites the whole cache with a
        // *wider* per-head stride (total_seq > past_seq), so an aliased
        // in-place write makes head h's current-token store overlap head
        // h+1's past load across unordered threads — a data race that leaves
        // every head beyond head 0 nondeterministic. Stage the rebuild in a
        // disjoint scratch buffer (reads the pristine past), then copy the
        // fully-formed dense cache back. General: any model whose
        // default-domain Attention grows an aliased KV cache.
        let alias_key = has_past_key && present_key_out == Some(past_key_ptr);
        let alias_value = has_past_value && present_value_out == Some(past_value_ptr);
        let capacity_key = kv_frozen || key_cap > total_seq;
        let capacity_value = kv_frozen || value_cap > value_total_seq;
        let stage_key = alias_key && !capacity_key;
        let stage_value = alias_value && !capacity_value;
        // On-device valid length ABI for default-domain Attention: derive
        // the valid attended length from the attention-mask frontier so the
        // kernel reads it from device memory instead of host shape metadata
        // (whose extent is frozen when a CUDA graph is captured). Scanning
        // the LAST query row makes it correct for BOTH phases — prefill
        // returns prompt_len, decode returns total_seq (row 0 would report 1
        // under a causal prefill mask). Eligible for the fixed-capacity,
        // masked fixed-slot-append path in BOTH causal and non-causal form:
        // the standard additive causal-mask builder (`Where(And(mask, causal),
        // 0, -inf)`) is frozen to physical capacity alongside the KV, so its
        // last-row frontier is the true valid length regardless of the op's
        // `is_causal` attribute (the causal frontier and the padding frontier
        // coincide at the last query row). When `dev_len` is set the causal
        // offset and score-loop extent are both derived from it (see
        // `attention_row`), so no host length leaks into a captured causal
        // decode. Every other path passes a null pointer and keeps the
        // host-derived length, so eager/dense/GQA are bit-for-bit unchanged.
        let dev_length_eligible = !self.window.is_active()
            && has_past_key
            && has_past_value
            && mask.kind != 0
            && capacity_key
            && capacity_value
            && alias_key
            && alias_value;

        // Single-token fixed-capacity decode and the staged dense growth path
        // can be captured once warmed. Both have host-static launch geometry
        // for a given step; borrow all scratch from the caller lane so
        // capture records only device work. The staged path needs
        // dedicated K/V slots because its source must remain disjoint from the
        // aliased present/past destination.
        let staged_decode_eligible = (stage_key || stage_value) && batch == 1 && q_seq == 1;
        let capture_workspace_eligible = dev_length_eligible || staged_decode_eligible;
        let mut capture_signature = capture_workspace_eligible.then(|| StdAttnCaptureSignature {
            dtype,
            inputs: inputs
                .iter()
                .map(|input| (input.dtype, input.shape.to_vec(), input.is_absent()))
                .collect(),
            outputs: outputs
                .iter()
                .map(|output| (output.dtype, output.shape.to_vec()))
                .collect(),
            batch,
            q_heads,
            kv_heads,
            q_seq,
            key_cap,
            head_size,
            v_head_size,
            workspace_base: 0,
            workspace_bytes: 0,
            split: None,
            default_offset,
        });
        // FlashDecoding split-KV: engage on the capture-safe fixed-capacity
        // decode route. `num_splits`/`chunk` derive only from the fixed cap
        // and row count, so the choice is identical under eager and capture.
        let total_rows_u =
            std_attention_elements(&[batch, q_heads, q_seq], "attention rows")? as u64;
        let split_cfg = attention_split_config(
            q_seq == 1,
            dev_length_eligible,
            self.is_causal,
            want_qk,
            key_cap as u64,
            total_rows_u,
            v_head_size as u64,
        );
        let split_bytes = split_cfg
            .map(|(splits, _)| {
                attention_split_scratch_bytes(splits, total_rows_u, v_head_size as u64)
            })
            .transpose()?
            .unwrap_or(0);
        let workspace_layout = std_attention_workspace_layout(
            batch,
            q_heads,
            q_seq,
            total_seq,
            kv_heads,
            key_cap,
            head_size,
            value_cap,
            v_head_size,
            element_bytes,
            stage_key,
            stage_value,
            present_key_out.is_none(),
            present_value_out.is_none(),
            batch,
            dev_length_eligible,
            split_bytes,
        )?;
        let parts = workspace_layout.bind(prepared).map_err(|cause| {
            EpError::KernelFailed(format!("Attention: cannot bind caller workspace: {cause}"))
        })?;
        let prepared = WorkspaceView::new(
            prepared
                .ok_or_else(|| {
                    EpError::KernelFailed(
                        "Attention: prepare and pass caller workspace before execution".into(),
                    )
                })?
                .ptr(),
            workspace_layout.bytes(),
        );
        let [
            scores_region,
            dev_len_region,
            offsets_region,
            pad_limits_region,
            stage_key_region,
            stage_value_region,
            present_key_region,
            present_value_region,
            split_region,
        ] = parts;
        if let Some(signature) = capture_signature.as_mut() {
            signature.workspace_base = prepared.ptr().0 as usize;
            signature.workspace_bytes = prepared.bytes();
            signature.split = split_cfg;
        }
        if capturing {
            Self::validate_capture_signature(&prior_signature, capture_signature.as_ref().ok_or_else(|| EpError::KernelFailed("Attention: route is not capture eligible; warm an eligible decode before capture".into()))?)?;
            self.runtime
                .require_registered_workspace_capture(prepared, "Attention prepared workspace")?;
        }
        let pointer = |view: WorkspaceView| cuptr(view.ptr().0.cast_const());
        let scores_ptr = pointer(scores_region);
        let split_base_ptr = (split_region.bytes() != 0).then(|| pointer(split_region));
        let present_key_ptr = present_key_out.unwrap_or_else(|| pointer(present_key_region));
        let present_value_ptr = present_value_out.unwrap_or_else(|| pointer(present_value_region));
        let key_kv_ptr = if stage_key {
            pointer(stage_key_region)
        } else {
            present_key_ptr
        };
        let value_kv_ptr = if stage_value {
            pointer(stage_value_region)
        } else {
            present_value_ptr
        };
        let dev_len_ptr = if dev_length_eligible {
            pointer(dev_len_region)
        } else {
            0
        };
        // Without nonpad, controls are uniform launch arguments. No shared
        // workspace contents survive from warmup, and capture adds no upload
        // or initialization launch. Device-derived length takes precedence.
        let (offsets_ptr, pad_limits_ptr) = if nonpad_kv_seqlen.is_some() {
            let offsets_ptr = pointer(offsets_region);
            let pad_limits_ptr = pointer(pad_limits_region);
            if capturing {
                return Err(EpError::KernelFailed("Attention: nonpad_kv_seqlen requires host validation; execute this route outside CUDA capture".into()));
            }
            let offsets_bytes = unsafe {
                std::slice::from_raw_parts(
                    offsets.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(offsets.as_slice()),
                )
            };
            let pad_bytes = unsafe {
                std::slice::from_raw_parts(
                    pad_limits.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(pad_limits.as_slice()),
                )
            };
            unsafe { self.runtime.htod(offsets_bytes, offsets_ptr)? };
            unsafe { self.runtime.htod(pad_bytes, pad_limits_ptr)? };
            (offsets_ptr, pad_limits_ptr)
        } else {
            (0, 0)
        };
        if dev_length_eligible {
            let key_len = mask.dims[3];
            let last_row = mask.dims[2].saturating_sub(1);
            let row_base = last_row.checked_mul(key_len).ok_or_else(|| {
                EpError::KernelFailed(
                    "Attention: mask last-row offset overflow; reduce the mask shape".into(),
                )
            })?;
            self.launch_derive_len(mask.ptr, mask.kind, key_len, row_base, dev_len_ptr)?;
        }
        // Build present_key / present_value on the device. In capacity mode
        // the append writes only the new rows [past_seq, total_seq) into
        // their fixed slot; the dense path rebuilds all rows.
        let key_write_start = if capacity_key && alias_key {
            key_past_seq
        } else {
            0
        };
        let key_past_cap = if capacity_key { key_cap } else { key_past_seq };
        let value_write_start = if capacity_value && alias_value {
            value_past_seq
        } else {
            0
        };
        let value_past_cap = if capacity_value {
            value_cap
        } else {
            value_past_seq
        };
        self.launch_build_kv(
            past_key_ptr,
            k_cur_ptr,
            key_kv_ptr,
            has_past_key,
            k_cur.is_3d,
            past_key.as_ref().map(|p| p.is_3d).unwrap_or(false),
            dtype_code,
            batch,
            kv_heads,
            key_past_seq,
            k_cur.seq,
            total_seq,
            head_size,
            key_cap,
            key_past_cap,
            key_write_start,
            dev_len_ptr,
        )?;
        self.launch_build_kv(
            past_value_ptr,
            v_cur_ptr,
            value_kv_ptr,
            has_past_value,
            v_cur.is_3d,
            past_value.as_ref().map(|p| p.is_3d).unwrap_or(false),
            dtype_code,
            batch,
            kv_heads,
            value_past_seq,
            v_cur.seq,
            value_total_seq,
            v_head_size,
            value_cap,
            value_past_cap,
            value_write_start,
            dev_len_ptr,
        )?;

        // When staged, publish the freshly-built dense cache back into the
        // aliased present/past buffer for the next step. Source and
        // destination are disjoint, so this copy is race-free. Issue it
        // stream-ordered on the EP compute stream: `build_kv` (the producer)
        // and the next step's `build_kv` (the consumer) both run on that same
        // stream, so the copy is implicitly ordered against them without a
        // full device `synchronize()`. The execute-exit synchronize (below)
        // still drains it, preserving blocking semantics for eager callers.
        if stage_key {
            unsafe {
                self.runtime.dtod_async(
                    key_kv_ptr,
                    present_key_ptr,
                    present_key_expected * element_bytes,
                )?;
            }
        }
        if stage_value {
            unsafe {
                self.runtime.dtod_async(
                    value_kv_ptr,
                    present_value_ptr,
                    present_value_expected * element_bytes,
                )?;
            }
        }

        // Launch the attention kernel. Decode with a fixed-capacity cache
        // takes the FlashDecoding split-KV path (fills the machine at wide
        // context); every other shape takes the monolithic one-block-per-row
        // `attention_row`.
        let total_rows = total_rows_u;
        if total_rows > 0 {
            let batch_u = batch as u64;
            let q_heads_u = q_heads as u64;
            let q_seq_u = q_seq as u64;
            let kv_heads_u = kv_heads as u64;
            let total_seq_u = total_seq as u64;
            let kv_cap_u = key_cap as u64;
            let head_size_u = head_size as u64;
            let v_head_size_u = v_head_size as u64;
            let group_u = group as u64;
            let q_is_3d = i32::from(q.is_3d);
            let out_is_3d = i32::from(q_rank == 3);
            let is_causal = i32::from(self.is_causal);
            let window_left = self.window.left.unwrap_or(-1);
            let window_right = self.window.right.unwrap_or(-1);
            let mask_kind = mask.kind;
            let mask_rank = mask.rank;
            let (md0, md1, md2, md3) = (mask.dims[0], mask.dims[1], mask.dims[2], mask.dims[3]);
            let qk_mode = self.qk_matmul_output_mode as i32;
            let want_qk_i = i32::from(want_qk);
            let softcap = self.softcap;
            let grid_x = total_rows.min(u32::MAX as u64).max(1) as u32;
            if let Some((num_splits, chunk)) = split_cfg {
                let split_base = split_base_ptr.ok_or_else(|| {
                    EpError::KernelFailed(
                        "Attention: split-KV engaged but partial scratch is unallocated".into(),
                    )
                })?;
                let meta_floats = total_rows.saturating_mul(num_splits).saturating_mul(2);
                let split_meta_ptr = split_base;
                let split_out_ptr = split_base + meta_floats * std::mem::size_of::<f32>() as u64;
                let num_splits_u = num_splits;
                let chunk_u = chunk;

                let split_func = self.runtime.nvrtc_function(
                    ATTENTION_MODULE,
                    ATTENTION_SOURCE,
                    "attention_split",
                )?;
                let mut builder = self.runtime.stream().launch_builder(&split_func);
                builder
                    .arg(&q_ptr)
                    .arg(&key_kv_ptr)
                    .arg(&value_kv_ptr)
                    .arg(&mask.ptr)
                    .arg(&scores_ptr)
                    .arg(&split_out_ptr)
                    .arg(&split_meta_ptr)
                    .arg(&offsets_ptr)
                    .arg(&pad_limits_ptr)
                    .arg(&default_offset)
                    .arg(&batch_u)
                    .arg(&q_heads_u)
                    .arg(&q_seq_u)
                    .arg(&kv_heads_u)
                    .arg(&total_seq_u)
                    .arg(&kv_cap_u)
                    .arg(&head_size_u)
                    .arg(&v_head_size_u)
                    .arg(&group_u)
                    .arg(&dtype_code)
                    .arg(&q_is_3d)
                    .arg(&is_causal)
                    .arg(&sqrt_scale)
                    .arg(&softcap)
                    .arg(&mask_kind)
                    .arg(&mask_rank)
                    .arg(&md0)
                    .arg(&md1)
                    .arg(&md2)
                    .arg(&md3)
                    .arg(&dev_len_ptr)
                    .arg(&num_splits_u)
                    .arg(&chunk_u);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (grid_x, num_splits.min(u32::MAX as u64).max(1) as u32, 1),
                        block_dim: (ATTN_SPLIT_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| driver_err("launch attention_split", error))?;

                let combine_func = self.runtime.nvrtc_function(
                    ATTENTION_MODULE,
                    ATTENTION_SOURCE,
                    "attention_combine",
                )?;
                let mut cbuilder = self.runtime.stream().launch_builder(&combine_func);
                cbuilder
                    .arg(&split_out_ptr)
                    .arg(&split_meta_ptr)
                    .arg(&y_ptr)
                    .arg(&batch_u)
                    .arg(&q_heads_u)
                    .arg(&q_seq_u)
                    .arg(&v_head_size_u)
                    .arg(&dtype_code)
                    .arg(&out_is_3d)
                    .arg(&num_splits_u);
                unsafe {
                    cbuilder.launch(LaunchConfig {
                        grid_dim: (grid_x, 1, 1),
                        block_dim: (ATTN_SPLIT_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| driver_err("launch attention_combine", error))?;
            } else {
                let func = self.runtime.nvrtc_function(
                    ATTENTION_MODULE,
                    ATTENTION_SOURCE,
                    "attention_row",
                )?;
                let mut builder = self.runtime.stream().launch_builder(&func);
                builder
                    .arg(&q_ptr)
                    .arg(&key_kv_ptr)
                    .arg(&value_kv_ptr)
                    .arg(&mask.ptr)
                    .arg(&scores_ptr)
                    .arg(&y_ptr)
                    .arg(&qk_ptr)
                    .arg(&offsets_ptr)
                    .arg(&pad_limits_ptr)
                    .arg(&default_offset)
                    .arg(&batch_u)
                    .arg(&q_heads_u)
                    .arg(&q_seq_u)
                    .arg(&kv_heads_u)
                    .arg(&total_seq_u)
                    .arg(&kv_cap_u)
                    .arg(&head_size_u)
                    .arg(&v_head_size_u)
                    .arg(&group_u)
                    .arg(&dtype_code)
                    .arg(&q_is_3d)
                    .arg(&out_is_3d)
                    .arg(&is_causal)
                    .arg(&sqrt_scale)
                    .arg(&softcap)
                    .arg(&mask_kind)
                    .arg(&mask_rank)
                    .arg(&md0)
                    .arg(&md1)
                    .arg(&md2)
                    .arg(&md3)
                    .arg(&qk_mode)
                    .arg(&want_qk_i)
                    .arg(&dev_len_ptr)
                    .arg(&window_left)
                    .arg(&window_right);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (grid_x, 1, 1),
                        block_dim: (attention_row_threads(q_seq == 1), 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|error| driver_err("launch attention_row", error))?;
            }
        }
        if !capturing {
            self.runtime.synchronize()?;
        }
        *warm_state = capture_signature;
        Ok(())
    }

    /// Report the exact score + route-required staged-K/V composite this
    /// dispatch can materialize (#736). The free helper is covered by CPU-only
    /// tests so planning and execution remain aligned without a GPU.
    fn composite_workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<WorkspaceRequirement> {
        std_attention_workspace_requirement(
            inputs,
            self.q_num_heads,
            self.kv_num_heads,
            !self.window.is_active(),
            self.is_causal,
            self.output_slots,
            outputs,
        )
    }
}

impl Kernel for StandardAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<onnx_runtime_ep_api::WorkspacePreparation> {
        self.composite_workspace_requirement(inputs, outputs)
            .map(onnx_runtime_ep_api::WorkspacePreparation::Ready)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.run(inputs, outputs, workspace)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        // Eligible once a single-token decode step has been warmed with all
        // scratch borrowed from the persistent caller lane. Uniform controls
        // are launch arguments; device length is regenerated on every replay.
        match self.warm_state.lock() {
            Ok(state) if state.is_some() => onnx_runtime_ep_api::CaptureSupport::Supported,
            Ok(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires a warmed capture-eligible single-token decode step",
            ),
            Err(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "Attention capture signature is unavailable because its state lock was poisoned",
            ),
        }
    }
}

#[cfg(test)]
fn execute_prepared(
    kernel: &StandardAttentionKernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    let metadata = inputs
        .iter()
        .map(|input| TensorMetadata::new(input.dtype, input.shape, !input.is_absent()))
        .collect::<Vec<_>>();
    let out = outputs
        .iter()
        .map(OutputMetadata::from_view)
        .collect::<Vec<_>>();
    let requirement = kernel
        .workspace_requirement(&metadata, &out)
        .and_then(onnx_runtime_ep_api::WorkspacePreparation::require_ready)?;
    let owner = crate::runtime::NativeDeviceAllocation::allocate_shared(
        &kernel.runtime,
        requirement.bytes as usize,
        onnx_runtime_memory_api::MemoryRole::Workspace { step_scoped: false },
    )?;
    let view = WorkspaceView::new(
        onnx_runtime_ep_api::DevicePtrMut(owner.ptr() as *mut c_void),
        requirement.bytes as usize,
    );
    let result = kernel.execute_with_workspace(inputs, outputs, Some(view));
    let complete = kernel
        .runtime
        .stream()
        .synchronize()
        .map_err(|cause| driver_err("complete Attention workspace fixture", cause));
    result.and(complete)
}

#[cfg(test)]
mod row_threads_tests {
    //! Locks the decode-vs-prefill launch width for `attention_row`. The decode
    //! path (`is_decode == true`, i.e. `q_seq == 1`) intentionally launches wider
    //! blocks (256 threads) so each of the few per-head blocks has enough warps to
    //! hide the per-key K/V load latency; prefill keeps the base `ROW_THREADS`
    //! width. The change is byte-identical (more threads, same reduction order).
    use super::{ROW_THREADS, attention_row_threads};

    #[test]
    fn decode_uses_wider_blocks_prefill_keeps_base() {
        // No `ONNX_GENAI_ATTN_ROW_THREADS` override in the test environment, so
        // the built-in defaults apply.
        assert_eq!(attention_row_threads(false), ROW_THREADS);
        assert_eq!(attention_row_threads(true), 256);
        // Both widths must be positive multiples of a warp.
        assert_eq!(attention_row_threads(true) % 32, 0);
        assert_eq!(attention_row_threads(false) % 32, 0);
    }
}

#[cfg(test)]
mod alias_tests {
    //! Regression tests for in-place KV-cache growth in the default-domain
    //! `Attention` kernel. DeepSeek-V2-Lite (MLA) binds the `present_key` /
    //! `present_value` outputs onto the SAME device buffer as the `past_key` /
    //! `past_value` inputs. `build_kv` re-lays-out the whole cache at a WIDER
    //! per-head stride (`total_seq` vs `past_seq`) in that shared buffer, so a
    //! head's current-token write collides with the next head's past read across
    //! unordered CUDA threads — nondeterministic for every head > 0. The fix
    //! stages the rebuild into a disjoint scratch buffer and copies it back, so
    //! the aliased result must equal a non-aliased reference and be stable across
    //! repeated runs.
    use super::*;
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut};
    use onnx_runtime_ir::{DeviceId, compute_contiguous_strides};
    use std::ffi::c_void;

    fn maybe_runtime() -> Option<crate::test_support::NativeRuntimeFixture> {
        crate::test_support::maybe_native_runtime()
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn bytes_f32(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // Deterministic pseudo-random fill in [-1, 1).
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9e3779b97f4a7c15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Reproduces the DeepSeek MLA decode step: multi-head KV cache grown by one
    /// token, with `present` optionally aliasing `past` in one device buffer.
    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable gpu-tests"
    )]
    fn decode_kv_growth_alias_matches_reference_and_is_deterministic() {
        let Some((_provider, rt)) = maybe_runtime() else {
            eprintln!("skipping: no CUDA device available");
            return;
        };
        let device = DeviceId::cuda(0);

        // MLA-flavoured shape: multi-head, asymmetric K (192-style) vs V head size.
        let heads = 4usize;
        let past = 3usize;
        let qlen = 1usize;
        let total = past + qlen;
        let kdim = 6usize; // key/query head size
        let vdim = 4usize; // value head size

        // Fixed inputs.
        let q = fill(heads * qlen * kdim, 1);
        let k_cur = fill(heads * qlen * kdim, 2);
        let v_cur = fill(heads * qlen * vdim, 3);
        let past_k = fill(heads * past * kdim, 4);
        let past_v = fill(heads * past * vdim, 5);

        let kernel = StandardAttentionKernel {
            window: AttentionWindow::default(),
            runtime: rt.clone(),
            scale: None,
            is_causal: true,
            q_num_heads: Some(heads),
            kv_num_heads: Some(heads),
            qk_matmul_output_mode: 0,
            softcap: 0.0,
            output_slots: [true, true, true, false],
            since_version: 24,
            warm_state: Mutex::new(None),
        };

        // Runs one decode step; when `alias` the present KV outputs share the
        // past KV buffers. `governed` routes the score + staged K/V composite
        // through an executor-shaped prepared workspace.
        let run = |alias: bool| -> Vec<f32> {
            let q_sh = [1usize, heads, qlen, kdim];
            let kcur_sh = [1usize, heads, qlen, kdim];
            let vcur_sh = [1usize, heads, qlen, vdim];
            let pastk_sh = [1usize, heads, past, kdim];
            let pastv_sh = [1usize, heads, past, vdim];
            let presk_sh = [1usize, heads, total, kdim];
            let presv_sh = [1usize, heads, total, vdim];
            let y_sh = [1usize, heads, qlen, vdim];

            let q_st = compute_contiguous_strides(&q_sh);
            let kcur_st = compute_contiguous_strides(&kcur_sh);
            let vcur_st = compute_contiguous_strides(&vcur_sh);
            let pastk_st = compute_contiguous_strides(&pastk_sh);
            let pastv_st = compute_contiguous_strides(&pastv_sh);
            let presk_st = compute_contiguous_strides(&presk_sh);
            let presv_st = compute_contiguous_strides(&presv_sh);
            let y_st = compute_contiguous_strides(&y_sh);

            // Cache buffers sized for the grown (total) length.
            let key_cap = heads * total * kdim * 4;
            let val_cap = heads * total * vdim * 4;
            let q_bytes = f32_bytes(&q);
            let kcur_bytes = f32_bytes(&k_cur);
            let vcur_bytes = f32_bytes(&v_cur);

            unsafe {
                let key_buf = rt.alloc_raw(key_cap).unwrap();
                let val_buf = rt.alloc_raw(val_cap).unwrap();
                let q_buf = rt.alloc_raw(q_bytes.len()).unwrap();
                let kcur_buf = rt.alloc_raw(kcur_bytes.len()).unwrap();
                let vcur_buf = rt.alloc_raw(vcur_bytes.len()).unwrap();
                // Past occupies the dense [heads, past, dim] prefix of the cache buffer.
                rt.htod(&f32_bytes(&past_k), key_buf).unwrap();
                rt.htod(&f32_bytes(&past_v), val_buf).unwrap();
                rt.htod(&q_bytes, q_buf).unwrap();
                rt.htod(&kcur_bytes, kcur_buf).unwrap();
                rt.htod(&vcur_bytes, vcur_buf).unwrap();

                let (presk_buf, presv_buf) = if alias {
                    (key_buf, val_buf)
                } else {
                    (
                        rt.alloc_raw(key_cap).unwrap(),
                        rt.alloc_raw(val_cap).unwrap(),
                    )
                };
                let y_buf = rt.alloc_raw(heads * qlen * vdim * 4).unwrap();
                let dp = |p: CUdeviceptr| DevicePtr(p as *const c_void);
                let dpm = |p: CUdeviceptr| DevicePtrMut(p as *mut c_void);

                let inputs = [
                    TensorView::new(dp(q_buf), DataType::Float32, &q_sh, &q_st, device),
                    TensorView::new(dp(kcur_buf), DataType::Float32, &kcur_sh, &kcur_st, device),
                    TensorView::new(dp(vcur_buf), DataType::Float32, &vcur_sh, &vcur_st, device),
                    TensorView::absent(DataType::Float32),
                    TensorView::new(dp(key_buf), DataType::Float32, &pastk_sh, &pastk_st, device),
                    TensorView::new(dp(val_buf), DataType::Float32, &pastv_sh, &pastv_st, device),
                ];
                let mut outputs = [
                    TensorMut::new(dpm(y_buf), DataType::Float32, &y_sh, &y_st, device),
                    TensorMut::new(
                        dpm(presk_buf),
                        DataType::Float32,
                        &presk_sh,
                        &presk_st,
                        device,
                    ),
                    TensorMut::new(
                        dpm(presv_buf),
                        DataType::Float32,
                        &presv_sh,
                        &presv_st,
                        device,
                    ),
                ];

                execute_prepared(&kernel, &inputs, &mut outputs).unwrap();

                let mut y_bytes = vec![0u8; heads * qlen * vdim * 4];
                rt.dtoh(&mut y_bytes, y_buf).unwrap();

                rt.free_raw(key_buf).unwrap();
                rt.free_raw(val_buf).unwrap();
                rt.free_raw(q_buf).unwrap();
                rt.free_raw(kcur_buf).unwrap();
                rt.free_raw(vcur_buf).unwrap();
                rt.free_raw(y_buf).unwrap();
                if !alias {
                    rt.free_raw(presk_buf).unwrap();
                    rt.free_raw(presv_buf).unwrap();
                }
                bytes_f32(&y_bytes)
            }
        };

        // Non-aliased present/past buffers give the race-free ground truth.
        let reference = run(false);
        // Prepared aliased execution must reproduce the disjoint reference and
        // remain stable across repeated uses of the same kernel.
        let aliased = run(true);
        assert_eq!(
            aliased, reference,
            "in-place KV-cache growth (present aliases past) must match the non-aliased reference"
        );
        assert_eq!(
            run(true),
            reference,
            "governed staged KV growth must match the non-aliased reference"
        );
        for i in 0..4 {
            assert_eq!(
                run(true),
                aliased,
                "aliased KV-cache growth must be deterministic across runs (iteration {i})"
            );
        }
    }

    /// Fixed-capacity / fixed-slot append path (the eager perf deliverable):
    /// when the `present` KV output is bound at a *physical capacity* wider than
    /// the valid length, `build_kv` must lay the cache out at the CAPACITY
    /// per-head stride and append the new token into slot `[past_seq]`, while the
    /// attention read is bounded to the valid `[0, total_seq)` rows. Physical
    /// slots `[total_seq, cap)` hold uninitialised padding that must never be
    /// read. This guards against a regression to reading the KV tensor *extent*
    /// as the sequence length: if the kernel used `cap` (or the padded buffer) as
    /// the loop bound it would fold the non-zero padding into the scores and
    /// diverge from the dense reference.
    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable gpu-tests"
    )]
    fn decode_kv_capacity_append_matches_reference_and_ignores_padding() {
        let Some((_provider, rt)) = maybe_runtime() else {
            eprintln!("skipping: no CUDA device available");
            return;
        };
        let device = DeviceId::cuda(0);

        let heads = 4usize;
        let past = 3usize;
        let qlen = 1usize;
        let total = past + qlen;
        let kdim = 6usize;
        let vdim = 4usize;

        let q = fill(heads * qlen * kdim, 1);
        let k_cur = fill(heads * qlen * kdim, 2);
        let v_cur = fill(heads * qlen * vdim, 3);
        let past_k = fill(heads * past * kdim, 4);
        let past_v = fill(heads * past * vdim, 5);

        let kernel = StandardAttentionKernel {
            window: AttentionWindow::default(),
            runtime: rt.clone(),
            scale: None,
            is_causal: true,
            q_num_heads: Some(heads),
            kv_num_heads: Some(heads),
            qk_matmul_output_mode: 0,
            softcap: 0.0,
            output_slots: [true, true, true, false],
            since_version: 24,
            warm_state: Mutex::new(None),
        };

        // Lay a `[heads, valid, dim]` contiguous tensor into a `[heads, cap, dim]`
        // capacity-strided buffer; the padding `[valid, cap)` is filled with
        // `garbage` to prove the kernel never reads it.
        let cap_strided =
            |rows: &[f32], valid: usize, dim: usize, cap: usize, garbage: f32| -> Vec<f32> {
                let mut buf = vec![garbage; heads * cap * dim];
                for h in 0..heads {
                    for t in 0..valid {
                        for d in 0..dim {
                            buf[(h * cap + t) * dim + d] = rows[(h * valid + t) * dim + d];
                        }
                    }
                }
                buf
            };

        // `cap == total` + non-aliased is the dense, race-free ground truth.
        // `cap > total` + aliased is the fixed-slot capacity append under test.
        let run = |alias: bool, cap: usize| -> Vec<f32> {
            let q_sh = [1usize, heads, qlen, kdim];
            let kcur_sh = [1usize, heads, qlen, kdim];
            let vcur_sh = [1usize, heads, qlen, vdim];
            let pastk_sh = [1usize, heads, past, kdim];
            let pastv_sh = [1usize, heads, past, vdim];
            let presk_sh = [1usize, heads, cap, kdim];
            let presv_sh = [1usize, heads, cap, vdim];
            let y_sh = [1usize, heads, qlen, vdim];

            let q_st = compute_contiguous_strides(&q_sh);
            let kcur_st = compute_contiguous_strides(&kcur_sh);
            let vcur_st = compute_contiguous_strides(&vcur_sh);
            let pastk_st = compute_contiguous_strides(&pastk_sh);
            let pastv_st = compute_contiguous_strides(&pastv_sh);
            let presk_st = compute_contiguous_strides(&presk_sh);
            let presv_st = compute_contiguous_strides(&presv_sh);
            let y_st = compute_contiguous_strides(&y_sh);

            let key_cap_bytes = heads * cap * kdim * 4;
            let val_cap_bytes = heads * cap * vdim * 4;
            // The kernel reads `past` at the *past_cap* per-head stride: the full
            // physical capacity `cap` for the capacity/fixed-slot path (aliased,
            // cap > total), or the dense valid `past` length otherwise. Lay the
            // past buffer out at exactly that stride, with non-zero padding in the
            // physical slots beyond the valid length so a stride/bound regression
            // is caught.
            let capacity_case = alias && cap > total;
            let pcap = if capacity_case { cap } else { past };
            let key_init = cap_strided(&past_k, past, kdim, pcap, 7.5);
            let val_init = cap_strided(&past_v, past, vdim, pcap, -4.25);
            let key_past_bytes = heads * pcap * kdim * 4;
            let val_past_bytes = heads * pcap * vdim * 4;
            let q_bytes = f32_bytes(&q);
            let kcur_bytes = f32_bytes(&k_cur);
            let vcur_bytes = f32_bytes(&v_cur);

            unsafe {
                let key_buf = rt.alloc_raw(key_past_bytes.max(key_cap_bytes)).unwrap();
                let val_buf = rt.alloc_raw(val_past_bytes.max(val_cap_bytes)).unwrap();
                let q_buf = rt.alloc_raw(q_bytes.len()).unwrap();
                let kcur_buf = rt.alloc_raw(kcur_bytes.len()).unwrap();
                let vcur_buf = rt.alloc_raw(vcur_bytes.len()).unwrap();
                rt.htod(&f32_bytes(&key_init), key_buf).unwrap();
                rt.htod(&f32_bytes(&val_init), val_buf).unwrap();
                rt.htod(&q_bytes, q_buf).unwrap();
                rt.htod(&kcur_bytes, kcur_buf).unwrap();
                rt.htod(&vcur_bytes, vcur_buf).unwrap();

                let (presk_buf, presv_buf) = if alias {
                    (key_buf, val_buf)
                } else {
                    (
                        rt.alloc_raw(key_cap_bytes).unwrap(),
                        rt.alloc_raw(val_cap_bytes).unwrap(),
                    )
                };
                let y_buf = rt.alloc_raw(heads * qlen * vdim * 4).unwrap();

                let dp = |p: CUdeviceptr| DevicePtr(p as *const c_void);
                let dpm = |p: CUdeviceptr| DevicePtrMut(p as *mut c_void);

                let inputs = [
                    TensorView::new(dp(q_buf), DataType::Float32, &q_sh, &q_st, device),
                    TensorView::new(dp(kcur_buf), DataType::Float32, &kcur_sh, &kcur_st, device),
                    TensorView::new(dp(vcur_buf), DataType::Float32, &vcur_sh, &vcur_st, device),
                    TensorView::absent(DataType::Float32),
                    TensorView::new(dp(key_buf), DataType::Float32, &pastk_sh, &pastk_st, device),
                    TensorView::new(dp(val_buf), DataType::Float32, &pastv_sh, &pastv_st, device),
                ];
                let mut outputs = [
                    TensorMut::new(dpm(y_buf), DataType::Float32, &y_sh, &y_st, device),
                    TensorMut::new(
                        dpm(presk_buf),
                        DataType::Float32,
                        &presk_sh,
                        &presk_st,
                        device,
                    ),
                    TensorMut::new(
                        dpm(presv_buf),
                        DataType::Float32,
                        &presv_sh,
                        &presv_st,
                        device,
                    ),
                ];

                execute_prepared(&kernel, &inputs, &mut outputs).unwrap();

                let mut y_bytes = vec![0u8; heads * qlen * vdim * 4];
                rt.dtoh(&mut y_bytes, y_buf).unwrap();

                rt.free_raw(key_buf).unwrap();
                rt.free_raw(val_buf).unwrap();
                rt.free_raw(q_buf).unwrap();
                rt.free_raw(kcur_buf).unwrap();
                rt.free_raw(vcur_buf).unwrap();
                rt.free_raw(y_buf).unwrap();
                if !alias {
                    rt.free_raw(presk_buf).unwrap();
                    rt.free_raw(presv_buf).unwrap();
                }
                bytes_f32(&y_bytes)
            }
        };

        // Dense reference at exactly the valid length (no padding).
        let reference = run(false, total);
        // Fixed-slot append into a wider physical capacity with non-zero padding
        // must reproduce the reference exactly (padding ignored) and be stable.
        let capacity = run(true, total + 5);
        assert_eq!(
            capacity, reference,
            "capacity/fixed-slot KV append must match the dense reference and \
             ignore the non-zero physical padding beyond the valid length"
        );
        for i in 0..4 {
            assert_eq!(
                run(true, total + 5),
                capacity,
                "capacity KV append must be deterministic across runs (iteration {i})"
            );
        }
    }

    /// The on-device valid-length ABI must be correct for BOTH decode and
    /// prefill: scanning the LAST query row of the additive mask returns
    /// `total_seq` for a single-token decode and `prompt_len` for a multi-token
    /// causal prefill. This locks the last-row behavior — a row-0 (decode-only)
    /// scan would wrongly report 1 for a causal prefill mask, so this test fails
    /// if the kernel reverts to scanning row 0 or to host shape metadata.
    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable gpu-tests"
    )]
    fn derive_len_reads_valid_length_from_device_for_prefill_and_decode() {
        let Some((_provider, rt)) = maybe_runtime() else {
            eprintln!("skipping: no CUDA device available");
            return;
        };
        let kernel = StandardAttentionKernel {
            window: AttentionWindow::default(),
            runtime: rt.clone(),
            scale: None,
            is_causal: false,
            q_num_heads: Some(1),
            kv_num_heads: Some(1),
            qk_matmul_output_mode: 0,
            softcap: 0.0,
            output_slots: [true, false, false, false],
            since_version: 24,
            warm_state: Mutex::new(None),
        };
        const NEG: f32 = -65504.0;
        let mask_kind = 1i32; // f32 additive bias

        // Launch derive_len over `mask` scanning the row at `row_base` and read
        // the device-written i32 back to the host.
        let derive = |mask: &[f32], key_len: u64, row_base: u64| -> i32 {
            let mask_buf = rt.alloc_raw(mask.len() * 4).unwrap();
            unsafe { rt.htod(&f32_bytes(mask), mask_buf).unwrap() };
            let out_buf = rt.alloc_raw(std::mem::size_of::<i32>()).unwrap();
            kernel
                .launch_derive_len(mask_buf, mask_kind, key_len, row_base, out_buf)
                .unwrap();
            rt.synchronize().unwrap();
            let mut out = [0u8; 4];
            unsafe { rt.dtoh(&mut out, out_buf).unwrap() };
            unsafe { rt.free_raw(mask_buf).unwrap() };
            unsafe { rt.free_raw(out_buf).unwrap() };
            i32::from_le_bytes(out)
        };

        // Decode: mask row [1,1,1,cap] with `total` valid then padding.
        let cap = 8u64;
        let total = 5i32;
        let mut decode = vec![0.0f32; total as usize];
        decode.extend(std::iter::repeat_n(NEG, cap as usize - total as usize));
        assert_eq!(
            derive(&decode, cap, 0),
            total,
            "decode: device valid length must equal total_seq"
        );

        // Prefill: causal mask [1,1,prompt_len,cap], row i valid for keys [0,i].
        let prompt_len = 4usize;
        let mut prefill = Vec::with_capacity(prompt_len * cap as usize);
        for i in 0..prompt_len {
            for j in 0..cap as usize {
                prefill.push(if j <= i { 0.0 } else { NEG });
            }
        }
        let last_row_base = (prompt_len as u64 - 1) * cap;
        assert_eq!(
            derive(&prefill, cap, last_row_base),
            prompt_len as i32,
            "prefill: last-row scan must return prompt_len"
        );
        // Row 0 (the decode-only bug) reports 1 for a causal prefill mask, which
        // is why the ABI scans the last query row.
        assert_eq!(
            derive(&prefill, cap, 0),
            1,
            "row-0 scan reports 1 for a causal prefill mask (decode-only bug guard)"
        );

        // Wide-context decode spanning many thread strides (key_len > the 256
        // reduction threads) with an unaligned frontier: locks the block-parallel
        // min-index reduction against the prior serial scan. The valid run is
        // [0, wide_total) followed by padding; the derived length must be exactly
        // wide_total regardless of which thread stride owns the frontier.
        let wide_cap = 2600u64;
        for wide_total in [1i32, 255, 256, 257, 617, 2599, 2600] {
            let mut wide = vec![0.0f32; wide_total as usize];
            wide.extend(std::iter::repeat_n(
                NEG,
                wide_cap as usize - wide_total as usize,
            ));
            assert_eq!(
                derive(&wide, wide_cap, 0),
                wide_total,
                "wide decode: parallel frontier must equal valid length {wide_total}"
            );
        }
    }

    /// Capture eligibility of the default-domain Attention path is gated on a
    /// warmed, fixed-capacity, device-valid-length single-token decode step: the
    /// kernel only reports `Supported` after such a step records its capture
    /// signature. A fresh kernel (and every eager/dense/growing path, which never
    /// records a signature) must decline capture. This locks the gate — the test
    /// fails if `capture_support` reverts to unconditionally returning `Supported`
    /// or the device-valid-length signature requirement is dropped.
    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable gpu-tests"
    )]
    fn capture_support_gated_on_warmed_device_valid_length_signature() {
        let Some((_provider, rt)) = maybe_runtime() else {
            eprintln!("skipping: no CUDA device available");
            return;
        };
        let kernel = StandardAttentionKernel {
            window: AttentionWindow::default(),
            runtime: rt.clone(),
            scale: None,
            is_causal: false,
            q_num_heads: Some(1),
            kv_num_heads: Some(1),
            qk_matmul_output_mode: 0,
            softcap: 0.0,
            output_slots: [true, false, false, false],
            since_version: 24,
            warm_state: Mutex::new(None),
        };
        // No warmed decode step yet -> capture declined.
        assert!(
            !matches!(
                kernel.capture_support(),
                onnx_runtime_ep_api::CaptureSupport::Supported
            ),
            "fresh kernel must decline capture until a fixed-capacity device-valid-length decode step is warmed"
        );
        // Simulate a warmed single-token decode step recording its signature.
        *kernel.warm_state.lock().unwrap() = Some(StdAttnCaptureSignature {
            dtype: DataType::Float16,
            inputs: Vec::new(),
            outputs: Vec::new(),
            batch: 1,
            q_heads: 1,
            kv_heads: 1,
            q_seq: 1,
            key_cap: 4096,
            head_size: 192,
            v_head_size: 128,
            workspace_base: 256,
            workspace_bytes: 4096,
            split: None,
            default_offset: 0,
        });
        assert!(
            matches!(
                kernel.capture_support(),
                onnx_runtime_ep_api::CaptureSupport::Supported
            ),
            "capture must be Supported once a device-valid-length single-token decode step is warmed"
        );
    }
    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable gpu-tests"
    )]
    fn shared_workspace_capture_controls_survive_overwrite_and_require_real_pins() {
        use onnx_runtime_ep_api::ExecutionProvider;
        use onnx_runtime_memory_api::MemoryRole;
        let ep = crate::CudaExecutionProvider::new_default()
            .expect("Attention capture regression requires CUDA");
        let rt = ep.runtime();
        let make_kernel = || StandardAttentionKernel {
            runtime: rt.clone(),
            scale: None,
            is_causal: true,
            window: AttentionWindow::default(),
            q_num_heads: None,
            kv_num_heads: None,
            qk_matmul_output_mode: 0,
            softcap: 0.0,
            output_slots: [true, true, true, false],
            since_version: 24,
            warm_state: Mutex::new(None),
        };
        let first = make_kernel();
        let second = make_kernel();
        let q = rt.alloc_raw(8).unwrap();
        let k = rt.alloc_raw(8).unwrap();
        let v = rt.alloc_raw(8).unwrap();
        let key_a = rt.alloc_raw(16).unwrap();
        let value_a = rt.alloc_raw(16).unwrap();
        let key_b = rt.alloc_raw(24).unwrap();
        let value_b = rt.alloc_raw(24).unwrap();
        let y_a = rt.alloc_raw(8).unwrap();
        let y_b = rt.alloc_raw(8).unwrap();
        // SAFETY: each allocation covers its exact typed source bytes.
        unsafe {
            rt.htod(&f32_bytes(&[1.0, 0.5]), q).unwrap();
            rt.htod(&f32_bytes(&[0.25, 1.0]), k).unwrap();
            rt.htod(&f32_bytes(&[1.0, -1.0]), v).unwrap();
            rt.htod(&f32_bytes(&[0.5, 0.25, 0.0, 0.0]), key_a).unwrap();
            rt.htod(&f32_bytes(&[0.2, 0.7, 0.0, 0.0]), value_a).unwrap();
            rt.htod(&f32_bytes(&[0.5, 0.25, 0.75, 0.5, 0.0, 0.0]), key_b)
                .unwrap();
            rt.htod(&f32_bytes(&[0.2, 0.7, 0.8, 0.3, 0.0, 0.0]), value_b)
                .unwrap();
        }
        let requirement = |kernel: &StandardAttentionKernel, past: usize| {
            let current = [1, 1, 1, 2];
            let old = [1, 1, past, 2];
            let present = [1, 1, past + 1, 2];
            let inputs = [
                TensorMetadata::new(DataType::Float32, &current, true),
                TensorMetadata::new(DataType::Float32, &current, true),
                TensorMetadata::new(DataType::Float32, &current, true),
                TensorMetadata::new(DataType::Float32, &[], false),
                TensorMetadata::new(DataType::Float32, &old, true),
                TensorMetadata::new(DataType::Float32, &old, true),
            ];
            let outputs = [
                OutputMetadata::tensor(DataType::Float32, &current),
                OutputMetadata::tensor(DataType::Float32, &present),
                OutputMetadata::tensor(DataType::Float32, &present),
            ];
            kernel
                .workspace_requirement(&inputs, &outputs)
                .unwrap()
                .require_ready()
                .unwrap()
        };
        let bytes = requirement(&first, 1)
            .bytes
            .max(requirement(&second, 2).bytes) as usize;
        let mut workspace = ep
            .allocate_workspace(
                bytes,
                STD_SCORES_ALIGN,
                MemoryRole::Workspace { step_scoped: false },
            )
            .unwrap();
        let view = WorkspaceView::new(DevicePtrMut(workspace.as_mut_ptr()), workspace.len());
        let execute = |kernel: &StandardAttentionKernel,
                       past: usize,
                       key: CUdeviceptr,
                       value: CUdeviceptr,
                       y: CUdeviceptr,
                       view: Option<WorkspaceView>| {
            let current = [1, 1, 1, 2];
            let old = [1, 1, past, 2];
            let present = [1, 1, past + 1, 2];
            let current_st = compute_contiguous_strides(&current);
            let old_st = compute_contiguous_strides(&old);
            let present_st = compute_contiguous_strides(&present);
            let device = ep.device_id();
            let dp = |ptr| DevicePtr(ptr as *const c_void);
            let dpm = |ptr| DevicePtrMut(ptr as *mut c_void);
            let inputs = [
                TensorView::new(dp(q), DataType::Float32, &current, &current_st, device),
                TensorView::new(dp(k), DataType::Float32, &current, &current_st, device),
                TensorView::new(dp(v), DataType::Float32, &current, &current_st, device),
                TensorView::absent(DataType::Float32),
                TensorView::new(dp(key), DataType::Float32, &old, &old_st, device),
                TensorView::new(dp(value), DataType::Float32, &old, &old_st, device),
            ];
            let mut outputs = [
                TensorMut::new(dpm(y), DataType::Float32, &current, &current_st, device),
                TensorMut::new(dpm(key), DataType::Float32, &present, &present_st, device),
                TensorMut::new(dpm(value), DataType::Float32, &present, &present_st, device),
            ];
            kernel.execute_with_workspace(&inputs, &mut outputs, view)
        };
        assert!(
            execute(&first, 1, key_a, value_a, y_a, None)
                .unwrap_err()
                .to_string()
                .contains("workspace")
        );
        execute(&first, 1, key_a, value_a, y_a, Some(view)).unwrap();
        execute(&second, 2, key_b, value_b, y_b, Some(view)).unwrap();
        let read = |ptr| {
            let mut bytes = vec![0u8; 8];
            unsafe { rt.dtoh(&mut bytes, ptr).unwrap() };
            bytes
        };
        let expected = [read(y_a), read(y_b)];
        rt.begin_graph_capture(&[&first]).unwrap();
        let missing_pin = execute(&first, 1, key_a, value_a, y_a, Some(view))
            .unwrap_err()
            .to_string();
        rt.abort_graph_capture().unwrap();
        assert!(
            missing_pin.contains("real workspace owner"),
            "{missing_pin}"
        );
        execute(&first, 1, key_a, value_a, y_a, Some(view)).unwrap();
        let pin = workspace
            .device_graph_resource()
            .unwrap()
            .expect("provider workspace needs a true owner pin");
        let counts = rt.allocation_counts();
        rt.begin_graph_capture_with_resources(&[&first, &second], vec![pin])
            .unwrap();
        execute(&first, 1, key_a, value_a, y_a, Some(view)).unwrap();
        execute(&second, 2, key_b, value_b, y_b, Some(view)).unwrap();
        rt.end_graph_capture().unwrap();
        // Deliberately destroy all warmup contents between every replay. Each
        // op must reconstruct scratch or take controls from its own arguments.
        for _ in 0..3 {
            unsafe {
                rt.htod(&vec![0xff; view.bytes()], cuptr(view.ptr().0.cast_const()))
                    .unwrap()
            };
            rt.replay_graph().unwrap();
            assert_eq!([read(y_a), read(y_b)], expected);
        }
        assert_eq!(rt.allocation_counts(), counts);
        rt.reset_graph().unwrap();
        ep.deallocate_workspace(workspace).unwrap();
        for ptr in [q, k, v, key_a, value_a, key_b, value_b, y_a, y_b] {
            unsafe { rt.free_raw(ptr).unwrap() };
        }
    }
}

#[cfg(test)]
mod workspace_governance_tests {

    fn region(layout: StdAttentionWorkspaceLayout, index: usize) -> WorkspaceView {
        let backing = WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(256usize as *mut c_void),
            layout.bytes(),
        );
        layout.bind(Some(backing)).unwrap()[index]
    }
    fn slots(count: usize) -> [bool; 4] {
        std::array::from_fn(|index| index < count)
    }
    fn requirement(
        inputs: &[TensorMetadata<'_>],
        q: Option<usize>,
        kv: Option<usize>,
        mask: bool,
        count: usize,
    ) -> Result<WorkspaceRequirement> {
        let dtype = inputs
            .first()
            .map_or(DataType::Undefined, |input| input.dtype);
        let outputs = (0..count)
            .map(|_| OutputMetadata::Unresolved { dtype })
            .collect::<Vec<_>>();
        super::std_attention_workspace_requirement(
            inputs,
            q,
            kv,
            mask,
            false,
            slots(count),
            &outputs,
        )
    }
    // CPU-only unit tests (no GPU needed) for the governed default-domain
    // `Attention` score + staged-K/V composite (#736): shared sizing, route
    // split, and retained capacity.
    use super::*;
    use onnx_runtime_ep_api::TensorMetadata;
    use onnx_runtime_memory_api::MemoryRole;

    fn meta(dtype: DataType, shape: &[usize]) -> TensorMetadata<'_> {
        TensorMetadata::new(dtype, shape, true)
    }

    fn dense_inputs(
        dtype: DataType,
        batch: usize,
        q_seq: usize,
        past_seq: usize,
    ) -> Vec<TensorMetadata<'static>> {
        let q = Box::leak(Box::new([batch, q_seq, 32 * 128]));
        let kv = Box::leak(Box::new([batch, q_seq, 8 * 128]));
        let absent_mask = Box::leak(Box::new([]));
        let past = Box::leak(Box::new([batch, 8, past_seq, 128]));
        vec![
            meta(dtype, q),
            meta(dtype, kv),
            meta(dtype, kv),
            TensorMetadata::new(DataType::Undefined, absent_mask, false),
            meta(dtype, past),
            meta(dtype, past),
        ]
    }

    #[test]
    fn complete_layout_is_disjoint_and_requires_valid_caller_backing() {
        let layout = std_attention_workspace_layout(
            2, 4, 1, 8, 2, 8, 3, 8, 5, 2, true, true, true, true, 2, true, 320,
        )
        .unwrap();
        #[repr(align(256))]
        struct Backing([u8; 4096]);
        let mut backing = Backing([0; 4096]);
        let view = WorkspaceView::new(
            onnx_runtime_ep_api::DevicePtrMut(backing.0.as_mut_ptr().cast()),
            layout.bytes(),
        );
        let parts = layout.bind(Some(view)).unwrap();
        assert_eq!(parts[SCORES].bytes(), 2 * 4 * 8 * 4);
        assert_eq!(parts[DEVICE_LENGTH].bytes(), 4);
        assert_eq!(parts[OFFSETS].bytes(), 16);
        assert_eq!(parts[PAD_LIMITS].bytes(), 16);
        assert_eq!(parts[STAGED_KEY].bytes(), 2 * 2 * 8 * 3 * 2);
        assert_eq!(parts[STAGED_VALUE].bytes(), 2 * 2 * 8 * 5 * 2);
        assert_eq!(parts[PRESENT_KEY].bytes(), parts[STAGED_KEY].bytes());
        assert_eq!(parts[PRESENT_VALUE].bytes(), parts[STAGED_VALUE].bytes());
        assert_eq!(parts[SPLIT_PARTIALS].bytes(), 320);
        for pair in parts.windows(2) {
            assert!(pair[0].ptr().0 as usize + pair[0].bytes() <= pair[1].ptr().0 as usize);
        }
        for bad in [
            None,
            Some(WorkspaceView::new(view.ptr(), layout.bytes() - 1)),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut((view.ptr().0 as usize + 1) as *mut c_void),
                layout.bytes(),
            )),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut((usize::MAX - 255) as *mut c_void),
                layout.bytes(),
            )),
        ] {
            assert!(
                layout
                    .bind(bad)
                    .unwrap_err()
                    .to_string()
                    .contains("prepare and pass")
            );
        }
        assert!(attention_split_scratch_bytes(u64::MAX, 2, 8).is_err());
        assert!(attention_split_scratch_bytes(2, 2, u64::MAX).is_err());
        assert!(
            std_attention_workspace_layout(
                1,
                1,
                1,
                1,
                1,
                usize::MAX,
                2,
                1,
                2,
                4,
                true,
                false,
                false,
                false,
                1,
                false,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn output_holes_keep_their_positional_scratch_and_validate_dtypes() {
        use onnx_runtime_ir::{NodeId, ValueId};
        let node = Node::new_with_output_slots(
            NodeId(0),
            "Attention",
            vec![],
            vec![Some(ValueId(0)), None, Some(ValueId(2))],
        );
        let slots = std_attention_output_slots(&node).unwrap();
        assert_eq!(slots, [true, false, true, false]);
        let inputs = [
            meta(DataType::Float32, &[1, 2, 1, 3]),
            meta(DataType::Float32, &[1, 1, 2, 3]),
            meta(DataType::Float32, &[1, 1, 2, 5]),
        ];
        let outputs = [
            OutputMetadata::tensor(DataType::Float32, &[1, 2, 1, 5]),
            OutputMetadata::Absent,
            OutputMetadata::tensor(DataType::Float32, &[1, 1, 2, 5]),
        ];
        let req =
            std_attention_workspace_requirement(&inputs, None, None, true, false, slots, &outputs)
                .unwrap();
        let expected = std_attention_workspace_layout(
            1, 2, 1, 2, 1, 2, 3, 2, 5, 4, false, false, true, false, 1, false, 0,
        )
        .unwrap();
        assert_eq!(req.bytes, expected.bytes() as u64);
        assert!(region(expected, PRESENT_KEY).bytes() > 0);
        assert_eq!(region(expected, PRESENT_VALUE).bytes(), 0);
        assert!(
            validate_std_attention_outputs(slots, &outputs[..1], DataType::Float32)
                .unwrap_err()
                .to_string()
                .contains("output slots")
        );
        let mut wrong = outputs.clone();
        wrong[2] = OutputMetadata::Unresolved {
            dtype: DataType::Float16,
        };
        assert!(
            validate_std_attention_outputs(slots, &wrong, DataType::Float32)
                .unwrap_err()
                .to_string()
                .contains("output slot 2 dtype")
        );
    }

    #[test]
    fn output_capacity_funds_the_selected_decode_and_physical_span() {
        let inputs = [
            meta(DataType::Float16, &[1, 4, 1, 3]),
            meta(DataType::Float16, &[1, 2, 1, 3]),
            meta(DataType::Float16, &[1, 2, 1, 5]),
            meta(DataType::Float32, &[1, 1, 1, 1024]),
            meta(DataType::Float16, &[1, 2, 1024, 3]),
            meta(DataType::Float16, &[1, 2, 1024, 5]),
        ];
        let outputs = [
            OutputMetadata::tensor(DataType::Float16, &[1, 4, 1, 5]),
            OutputMetadata::tensor(DataType::Float16, &[1, 2, 1024, 3]),
            OutputMetadata::tensor(DataType::Float16, &[1, 2, 1024, 5]),
        ];
        let split = attention_split_config(true, true, false, false, 1024, 4, 5);
        let bytes = split
            .map(|(splits, _)| attention_split_scratch_bytes(splits, 4, 5).unwrap())
            .unwrap_or(0);
        let expected = std_attention_workspace_layout(
            1, 4, 1, 1024, 2, 1024, 3, 1024, 5, 2, false, false, false, false, 1, true, bytes,
        )
        .unwrap();
        let req = std_attention_workspace_requirement(
            &inputs,
            None,
            None,
            true,
            false,
            [true, true, true, false],
            &outputs,
        )
        .unwrap();
        assert_eq!(req.bytes, expected.bytes() as u64);
        assert_eq!(region(expected, DEVICE_LENGTH).bytes(), 4);
        assert_eq!(region(expected, SPLIT_PARTIALS).bytes(), bytes);
    }

    #[test]
    fn capture_signature_requires_the_exact_borrowed_address_and_layout() {
        let warmed = StdAttnCaptureSignature {
            dtype: DataType::Float32,
            inputs: vec![],
            outputs: vec![],
            batch: 1,
            q_heads: 2,
            kv_heads: 1,
            q_seq: 1,
            key_cap: 1024,
            head_size: 3,
            v_head_size: 5,
            workspace_base: 256,
            workspace_bytes: 2048,
            split: Some((4, 256)),
            default_offset: 7,
        };
        let mut state = Some(warmed.clone());
        StandardAttentionKernel::validate_capture_signature(&state, &warmed).unwrap();
        for changed in [
            StdAttnCaptureSignature {
                workspace_base: 512,
                ..warmed.clone()
            },
            StdAttnCaptureSignature {
                workspace_bytes: 4096,
                ..warmed.clone()
            },
            StdAttnCaptureSignature {
                split: None,
                ..warmed.clone()
            },
            StdAttnCaptureSignature {
                default_offset: 8,
                ..warmed.clone()
            },
        ] {
            assert!(
                StandardAttentionKernel::validate_capture_signature(&state, &changed)
                    .unwrap_err()
                    .to_string()
                    .contains("signature changed")
            );
        }
        let previous = state.take();
        assert!(
            StandardAttentionKernel::validate_capture_signature(&state, &warmed)
                .unwrap_err()
                .to_string()
                .contains("without a successful warmed")
        );
        assert!(previous.is_some());
    }

    #[test]
    fn scores_bytes_matches_score_count_formula() {
        // Planning and execution size the governed score buffer through this
        // exact helper (#736), so pin its formula: `batch·heads·q_seq·total_seq`
        // fp32 scores. A degenerate geometry still reserves one element.
        let (batch, heads, q_seq, total) = (2usize, 4usize, 8usize, 130usize);
        assert_eq!(
            std_attention_scores_bytes(batch, heads, q_seq, total).unwrap(),
            batch * heads * q_seq * total * std::mem::size_of::<f32>()
        );
        assert_eq!(
            std_attention_scores_bytes(0, heads, q_seq, total).unwrap(),
            std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn prefill_without_past_retains_the_composite() {
        // Multi-token prefill (the 512-MiB-class worst case) retains the lane
        // peak, sized to `batch·heads·q_seq·key` fp32. With no past
        // cache there is no aliased growth to stage.
        let hidden = 32 * 128;
        let q = [1usize, 2048, hidden];
        let kv = [1usize, 2048, hidden];
        let inputs = [
            meta(DataType::Float32, &q),
            meta(DataType::Float32, &kv),
            meta(DataType::Float32, &kv),
        ];
        let req = requirement(&inputs, Some(32), Some(32), true, 3).unwrap();
        // Bound present outputs (3-output node) need no served K/V scratch, so the
        // composite is the always-live scores plus the per-batch control arrays.
        let layout = std_attention_workspace_layout(
            1, 32, 2048, 2048, 32, 2048, 128, 2048, 128, 4, false, false, false, false, 1, false, 0,
        )
        .unwrap();
        assert_eq!(req.bytes, layout.bytes() as u64);
        // batch(1) x heads x q_seq x kv_seq x f32, in the order
        // std_attention_workspace_layout multiplies them; the unit batch is
        // elided so the expression is not an identity op.
        assert_eq!(region(layout, SCORES).bytes(), 32 * 2048 * 2048 * 4);
        assert!(region(layout, PRESENT_KEY).bytes() == 0);
        assert_eq!(req.lifetime, WorkspaceLifetime::SessionPersistent);
        assert!(matches!(
            req.role(),
            MemoryRole::Workspace { step_scoped: false }
        ));
        assert_eq!(req.alignment, STD_SCORES_ALIGN);
    }

    #[test]
    fn dense_single_token_growth_stages_both_as_session_persistent() {
        let inputs = dense_inputs(DataType::Float16, 1, 1, 2048);
        let layout = std_attention_workspace_layout(
            1, 32, 1, 2049, 8, 2049, 128, 2049, 128, 2, true, true, false, false, 1, false, 0,
        )
        .unwrap();
        let req = requirement(&inputs, Some(32), Some(8), true, 3).unwrap();
        assert_eq!(req.bytes, layout.bytes() as u64);
        assert_eq!(region(layout, STAGED_KEY).bytes(), 8 * 2049 * 128 * 2);
        assert_eq!(region(layout, STAGED_VALUE).bytes(), 8 * 2049 * 128 * 2);
        assert_eq!(req.lifetime, WorkspaceLifetime::SessionPersistent);
        assert!(matches!(
            req.role(),
            MemoryRole::Workspace { step_scoped: false }
        ));
    }

    #[test]
    fn dense_prefill_growth_retains_both_staging_regions() {
        let inputs = dense_inputs(DataType::Float32, 2, 8, 128);
        let layout = std_attention_workspace_layout(
            2, 32, 8, 136, 8, 136, 128, 136, 128, 4, true, true, false, false, 2, false, 0,
        )
        .unwrap();
        let req = requirement(&inputs, Some(32), Some(8), true, 3).unwrap();
        assert_eq!(req.bytes, layout.bytes() as u64);
        assert!(region(layout, STAGED_KEY).bytes() != 0);
        assert!(region(layout, STAGED_VALUE).bytes() != 0);
        assert_eq!(req.lifetime, WorkspaceLifetime::SessionPersistent);
        assert!(matches!(
            req.role(),
            MemoryRole::Workspace { step_scoped: false }
        ));
    }

    #[test]
    fn unresolved_outputs_cover_dense_aliasing_even_when_mask_matches_past() {
        let inputs = [
            meta(DataType::Float32, &[1, 2, 1, 3]),
            meta(DataType::Float32, &[1, 1, 1, 3]),
            meta(DataType::Float32, &[1, 1, 1, 5]),
            meta(DataType::Float32, &[1, 1, 1, 4]),
            meta(DataType::Float32, &[1, 1, 4, 3]),
            meta(DataType::Float32, &[1, 1, 4, 5]),
        ];
        let unresolved = [
            OutputMetadata::Unresolved {
                dtype: DataType::Float32,
            },
            OutputMetadata::Unresolved {
                dtype: DataType::Float32,
            },
            OutputMetadata::Unresolved {
                dtype: DataType::Float32,
            },
        ];
        let dense = [
            OutputMetadata::tensor(DataType::Float32, &[1, 2, 1, 5]),
            OutputMetadata::tensor(DataType::Float32, &[1, 1, 5, 3]),
            OutputMetadata::tensor(DataType::Float32, &[1, 1, 5, 5]),
        ];
        let requirement = |outputs: &[OutputMetadata<'_>]| {
            std_attention_workspace_requirement(
                &inputs,
                None,
                None,
                true,
                false,
                [true, true, true, false],
                outputs,
            )
            .unwrap()
        };
        let known = requirement(&dense);
        assert!(requirement(&unresolved).bytes >= known.bytes);
        let mut one_unknown = dense.clone();
        one_unknown[1] = unresolved[1].clone();
        assert!(requirement(&one_unknown).bytes >= known.bytes);
        one_unknown = dense.clone();
        one_unknown[2] = unresolved[2].clone();
        assert!(requirement(&one_unknown).bytes >= known.bytes);
        let frozen = [
            dense[0].clone(),
            OutputMetadata::tensor(DataType::Float32, &[1, 1, 4, 3]),
            OutputMetadata::tensor(DataType::Float32, &[1, 1, 4, 5]),
        ];
        assert!(requirement(&frozen).bytes < known.bytes);
    }

    #[test]
    fn one_known_output_capacity_bounds_the_unresolved_peer() {
        let inputs = [
            meta(DataType::Float32, &[1, 2, 1, 3]),
            meta(DataType::Float32, &[1, 1, 1, 3]),
            meta(DataType::Float32, &[1, 1, 1, 5]),
            meta(DataType::Float32, &[1, 1, 1, 4]),
            meta(DataType::Float32, &[1, 1, 4, 3]),
            meta(DataType::Float32, &[1, 1, 4, 5]),
        ];
        let outputs = [
            OutputMetadata::tensor(DataType::Float32, &[1, 2, 1, 5]),
            OutputMetadata::tensor(DataType::Float32, &[1, 1, 11, 3]),
            OutputMetadata::tensor(DataType::Float32, &[1, 1, 11, 5]),
        ];
        let requirement = |outputs: &[OutputMetadata<'_>]| {
            std_attention_workspace_requirement(
                &inputs,
                None,
                None,
                true,
                false,
                [true, true, true, false],
                outputs,
            )
        };
        let known = requirement(&outputs).unwrap();
        for index in [1, 2] {
            let mut mixed = outputs.clone();
            mixed[index] = OutputMetadata::Unresolved {
                dtype: DataType::Float32,
            };
            assert!(requirement(&mixed).unwrap().bytes >= known.bytes);
        }
        let mut mismatched = outputs.clone();
        mismatched[2] = OutputMetadata::tensor(DataType::Float32, &[1, 1, 12, 5]);
        assert!(
            requirement(&mismatched)
                .unwrap_err()
                .to_string()
                .contains("physical capacities")
        );
    }

    #[test]
    fn non_float_or_unresolvable_metadata_reserves_nothing() {
        // Non-float operand: rejected by `run`, so charge nothing.
        let q_i = [1usize, 8, 256];
        let int_inputs = [
            meta(DataType::Int32, &q_i),
            meta(DataType::Int32, &q_i),
            meta(DataType::Int32, &q_i),
        ];
        assert_eq!(
            requirement(&int_inputs, Some(4), Some(4), true, 3).unwrap(),
            WorkspaceRequirement::NONE
        );
        // Rank-2 shape is not a resolvable attention operand.
        let bad = [8usize, 256];
        let bad_inputs = [
            meta(DataType::Float32, &bad),
            meta(DataType::Float32, &bad),
            meta(DataType::Float32, &bad),
        ];
        assert_eq!(
            requirement(&bad_inputs, Some(4), Some(4), true, 3).unwrap(),
            WorkspaceRequirement::NONE
        );
    }

    #[test]
    fn adaptive_split_geometry_fills_a_wave_and_stays_capture_safe() {
        // Adaptive mode (no chunk override): target ~ATTN_SPLIT_TARGET_BLOCKS
        // blocks = total_rows * num_splits, floored so no split covers fewer than
        // ATTN_SPLIT_MIN_CHUNK keys. For V2-Lite's 16 decode rows the deep-cap
        // optimum measured on H200 is num_splits=32 (grid ~512, ~one wave).
        let rows = 16;
        // Deep cap: capped by the target-blocks wave, not by cap.
        let (n, chunk) = attention_split_geometry(None, 4096, rows).unwrap();
        assert_eq!(n, 32, "deep cap should target ~a full wave of blocks");
        assert_eq!(chunk, 128, "keys per split at the deep optimum");
        assert!(rows * n >= ATTN_SPLIT_TARGET_BLOCKS / 2);
        // Extreme cap: split count is clamped by the wave target (never explodes),
        // so chunk grows instead — keeps the combine loop + scratch bounded.
        let (n_hi, chunk_hi) = attention_split_geometry(None, 16384, rows).unwrap();
        assert_eq!(n_hi, 32);
        assert_eq!(chunk_hi, 512);
        // Shallow cap: MIN_CHUNK floor caps useful splits (cap/MIN_CHUNK), so we
        // scale down gracefully rather than over-splitting tiny rows.
        let (n_lo, chunk_lo) = attention_split_geometry(None, 512, rows).unwrap();
        assert_eq!(n_lo, 4);
        assert_eq!(chunk_lo, 128);
        // Narrow cap: below 2*MIN_CHUNK there is no useful split — stay monolithic.
        assert_eq!(attention_split_geometry(None, 128, rows), None);
        assert_eq!(attention_split_geometry(None, 0, rows), None);
        // Fixed-chunk override reproduces the pre-adaptive behaviour exactly.
        assert_eq!(
            attention_split_geometry(Some(256), 4096, rows),
            Some((16, 256))
        );
        assert_eq!(
            attention_split_geometry(Some(256), 512, rows),
            Some((2, 256))
        );
        // Same fixed cap/rows always yields the same geometry — the property the
        // CUDA-graph capture path relies on (no dependence on live seqlen).
        assert_eq!(
            attention_split_geometry(None, 4096, rows),
            attention_split_geometry(None, 4096, rows)
        );
    }
}

/// Regression guard for issue #736: the governed default-domain `Attention`
/// score + staged-K/V composite is routed through
/// `Kernel::workspace_requirement` + an executor-prepared workspace, consumed
/// via `execute_with_workspace`.
/// This CPU-only source scan fails if the governed dispatch reintroduces raw
/// staged K/V allocation, and runs on CI without a GPU.
#[cfg(test)]
mod raw_allocation_guard {
    #[test]
    fn std_attention_composite_is_governed_not_raw_allocated() {
        let source = include_str!("standard_attention.rs")
            .split("#[cfg(test)]\nfn execute_prepared")
            .next()
            .unwrap();
        for forbidden in [
            ".alloc_raw(",
            ".free_raw(",
            "StdAttnWorkspace",
            "StdWorkspaceSlot",
        ] {
            assert!(
                !source.contains(forbidden),
                "production Attention must borrow one caller lane rather than {forbidden}"
            );
        }
        assert!(source.contains("require_registered_workspace_capture"));
    }
}

#[cfg(test)]
mod claim_tests {
    use super::*;

    #[test]
    fn attention25_window_claim_is_opset_and_value_aware() {
        use onnx_runtime_ir::{Attribute, NodeId};

        let mut node = Node::new(NodeId(0), "Attention", vec![], vec![]);
        let types = [DataType::Float32; 3];
        node.attributes
            .insert("left_window_size".into(), Attribute::Int(0));
        let reason = unsupported_reason(&node, 24, &types, Some(4)).unwrap();
        assert!(reason.contains("requires opset 25"));
        assert!(unsupported_reason(&node, 25, &types, Some(4)).is_none());
        node.attributes
            .insert("kv_num_heads".into(), Attribute::Int(1));
        let reason = unsupported_reason(&node, 25, &types, Some(4)).unwrap();
        assert!(reason.contains("not valid for 4D Q"));
        assert!(unsupported_reason(&node, 25, &types, Some(3)).is_none());
        node.attributes
            .insert("right_window_size".into(), Attribute::Int(-2));
        let reason = unsupported_reason(&node, 25, &types, Some(3)).unwrap();
        assert!(reason.contains("invalid right_window_size"));
    }

    #[test]
    fn accepts_float32_additive_mask_for_half_attention() {
        for activation_dtype in [DataType::Float16, DataType::BFloat16] {
            assert!(
                unsupported_reason(
                    &Node::new(onnx_runtime_ir::NodeId(0), "Attention", vec![], vec![]),
                    24,
                    &[
                        activation_dtype,
                        activation_dtype,
                        activation_dtype,
                        DataType::Float32,
                    ],
                    None,
                )
                .is_none(),
                "{activation_dtype:?} attention should accept an f32 additive mask"
            );
        }
    }
}
