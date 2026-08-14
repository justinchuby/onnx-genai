//! Marlin-style fused fp16 × int4 tensor-core GEMM for `com.microsoft::MatMulNBits`.
//!
//! # Why this module exists
//!
//! The M>1 (prefill / speculative-verify) path of [`super::matmul_nbits`]
//! abandons the fused int4 decode GEMV and falls back to a portable 16×16
//! CUDA-core tiled GEMM with fp32 accumulation (`gemm_f16_tiled`). That fallback
//! (a) leaves the tensor cores idle, producing the ~67 ms M=1→M=2 latency cliff,
//! and (b) is not advertised as CUDA-graph capture-safe, which blocks
//! speculative-decode capture (see `docs/research/speculative-capture-feasibility.md`,
//! #957). This module provides a Marlin-style fused kernel that runs the int4
//! GEMM on the SM80+ tensor cores with a **static launch grid** (capture-safe by
//! construction) while reusing the packed weights across the M query rows.
//!
//! # Relationship to IST-DASLab/vLLM Marlin
//!
//! This is an *original* kernel that adapts Marlin's core ideas — repacked
//! weights laid out for the exact per-lane tensor-core fragment distribution, and
//! **per-group scaling applied after the tensor-core accumulate** so the fp32
//! accumulator never carries a scale that varies along K — to our concrete ONNX
//! `MatMulNBits` weight format. It is **not** a line-for-line port: upstream
//! Marlin (Apache-2.0) targets a symmetric GPTQ layout with its own column
//! permutation, and gptq_marlin depends on `<cuda_pipeline.h>` / `crt/` headers
//! that are unavailable to our NVRTC-string compilation path. Our format is
//! ONNX-native (N-major nibble packing, even-K low nibble, **asymmetric** nibble
//! zero-points, group sizes 16/32/64/128), so a native kernel is both correct and
//! simpler than translating our weights into Marlin's format. Because no upstream
//! source is copied, no third-party LICENSE vendoring is required; the design
//! lineage is credited here.
//!
//! # ONNX MatMulNBits int4 format (input to the repack)
//!
//! `Y[M,N] = A[M,K] · dequant(B)^T`, where the logical weight `B` is `[N, K]` and
//!   `dequant(B)[n,k] = (code[n,k] - zp[n, k/group]) * scale[n, k/group]`.
//! Storage:
//! * `packed`  : `[N, k_blocks, group/2]` bytes; nibble for `(n, k)` lives at
//!   `packed[(n*k_blocks + k/group) * (group/2) + (k%group)/2]`, **low** nibble
//!   for even `k`, **high** nibble for odd `k`. `code ∈ [0,15]`.
//! * `scales`  : `[N, k_blocks]`, fp16 or fp32.
//! * `zero_points` (optional, asymmetric): `[N, zp_row_bytes]` with
//!   `zp_row_bytes = ceil(k_blocks/2)`; nibble for `(n, block)` at
//!   `zp[n*zp_row_bytes + block/2]`, low nibble for even block. Default `zp = 8`.
//!
//! # Repacked weight layout (produced by [`repack_int4_weights`])
//!
//! The tensor-core kernel uses `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
//! For the B (weight) operand, a warp lane `l` (with `groupID = l>>2`,
//! `tid = l&3`) owns column `n = ncol0 + groupID` and the four K values
//! `{2·tid, 2·tid+1, 2·tid+8, 2·tid+9}` of each 16-wide K slice. We repack in
//! **8-column n-tiles** so the 32 lanes of a warp read one contiguous 64-byte
//! chunk per K slice — lane `l` reads exactly bytes `[2·l, 2·l+1]`:
//!
//! `repacked[(n_tile * (K/16) + slice) * 64 + groupID*8 + tid*2 + {0,1}]`
//!   byte+0 = code(k=2·tid) | code(k=2·tid+1) << 4
//!   byte+1 = code(k=2·tid+8) | code(k=2·tid+9) << 4
//!
//! for column `n = n_tile*8 + groupID`, where `slice = k/16` runs `0..K/16` and
//! `n_tile = 0..ceil(N/8)`. N is padded up to a multiple of 8 (tail columns hold
//! zero codes and are never stored to the output). Total size is
//! `ceil(N/8)*8 * (K/16) * 8` bytes ≈ `N*K/2` — a *reordering*, not an expansion.
//! `scales` and `zero_points` keep their original `[N, k_blocks]` indexing.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Result};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC module + entry names for the Marlin int4 tensor-core GEMM.
pub const MARLIN_MODULE: &str = "matmul_nbits_marlin_gemm";
pub const MARLIN_GEMM_ENTRY: &str = "matmul_nbits_marlin_gemm_f16";
/// Split-K GEMM entry: partitions the K/group range across `grid.z`, writing
/// fp32 per-split partials that [`MARLIN_SPLITK_REDUCE_ENTRY`] reduces. Used to
/// fill idle SMs at small M/N where the single-block kernel is occupancy-bound.
pub const MARLIN_GEMM_SPLITK_ENTRY: &str = "matmul_nbits_marlin_gemm_f16_splitk";
/// Fixed-order split-K partial reduction + fp16 bias epilogue entry.
pub const MARLIN_SPLITK_REDUCE_ENTRY: &str = "matmul_nbits_marlin_splitk_reduce";
pub const MARLIN_REPACK_ENTRY: &str = "matmul_nbits_marlin_repack";

/// N columns handled by one thread block (one warp per 8-column tensor-core tile).
pub const MARLIN_WARPS: u32 = 4;
pub const MARLIN_N_PER_BLOCK: u32 = MARLIN_WARPS * 8;
/// M rows handled by one thread block (one `m16n8k16` tensor-core tile in M).
pub const MARLIN_M_PER_BLOCK: u32 = 16;

/// Minimum compute capability for the tensor-core path (`mma.sync`/cp.async are
/// SM80+). Callers must fall back to the portable CUDA-core GEMM below this.
pub const MARLIN_MIN_SM: (u32, u32) = (8, 0);

/// Returns `true` when the device can run the Marlin tensor-core kernel.
#[must_use]
pub fn device_supports_marlin(compute_capability: (u32, u32)) -> bool {
    compute_capability >= MARLIN_MIN_SM
}

/// Opt-in gate for routing the `MatMulNBits` M>1 path through Marlin. Default
/// OFF so the tensor-core path never becomes the silent default that could
/// regress a tier before it is proven faster than the portable tiled GEMM
/// (Rule 11 / mission: Marlin is opt-in and tier-scoped). Enable with
/// `ONNX_GENAI_MARLIN_M_GT_1=1` (or `true`/`on`).
#[must_use]
pub fn marlin_m_gt_1_enabled() -> bool {
    matches!(
        std::env::var("ONNX_GENAI_MARLIN_M_GT_1").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Gate for split-K within the Marlin M>1 path. Split-K partitions the K/group
/// range across `grid.z` to fill idle SMs when M (and thus the base block count)
/// is small, at the cost of a fixed-order fp32 partial reduction that is NOT
/// byte-identical to the single-block kernel (it stays within the f64-oracle
/// tolerance and is deterministic — greedy/argmax tokens remain byte-identical,
/// validated e2e on glm-4-9b and qwen2.5-14b).
///
/// This gate lives *inside* the already-opt-in Marlin M>1 path
/// ([`marlin_m_gt_1_enabled`], default OFF), so it never affects a default /
/// consumer / edge tier; and [`choose_split_k`] only elects a split for
/// small-M / low-wave shapes (returns 1 for large-M prefill and short-K),
/// leaving those on the byte-identical direct kernel. Split-K is the measured
/// lever that collapses the speculative-verify (M=K) wall — capture B* 5.10x ->
/// 2.69x at M=8 on glm-4-9b (Sebastian's Increment-0 re-probe) — so it is ON by
/// default whenever Marlin M>1 is enabled. Opt out with
/// `ONNX_GENAI_MARLIN_SPLITK=0` (or `false`/`off`) to force the byte-identical
/// direct kernel for every M.
#[must_use]
pub fn marlin_splitk_enabled() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_MARLIN_SPLITK").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Global cache of repacked weights, keyed by the source (packed) device pointer
/// plus dims and device ordinal. Weights are immutable initializers, so the
/// device repack runs **once** per weight and every later call — including
/// captured CUDA-graph replays — reuses the cached buffer with no allocation.
/// Kept module-global (rather than a per-kernel field) so wiring touches only
/// the `matmul_nbits.rs` dispatch seam. A bounded FIFO frees the oldest entry via
/// its owning runtime when the cap is exceeded.
struct RepackEntry {
    ptr: CUdeviceptr,
    runtime: Arc<CudaRuntime>,
}

struct RepackCache {
    map: HashMap<(usize, u32, usize, usize, usize), RepackEntry>,
    order: VecDeque<(usize, u32, usize, usize, usize)>,
}

const REPACK_CACHE_CAP: usize = 4096;

static REPACK_CACHE: OnceLock<Mutex<RepackCache>> = OnceLock::new();

fn repack_cache() -> &'static Mutex<RepackCache> {
    REPACK_CACHE.get_or_init(|| {
        Mutex::new(RepackCache {
            map: HashMap::new(),
            order: VecDeque::new(),
        })
    })
}

/// Ensure the repacked tensor-core weights for `packed` exist on device, running
/// the device repack once and caching the result. Returns `(repacked_ptr,
/// warm)`, where `warm == true` means the buffer was already cached (this call
/// performed no allocation / repack / sync and is therefore CUDA-graph
/// capture-safe). A cold miss while the stream is capturing is rejected so the
/// caller can fall back rather than allocate inside a capture.
pub fn ensure_repacked(
    runtime: &Arc<CudaRuntime>,
    packed: CUdeviceptr,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<(CUdeviceptr, bool)> {
    let key = (packed as usize, runtime.ordinal(), n, k, group_size);
    {
        let cache = repack_cache().lock().expect("marlin repack cache poisoned");
        if let Some(entry) = cache.map.get(&key) {
            return Ok((entry.ptr, true));
        }
    }
    if runtime.is_capturing()? {
        return Err(EpError::KernelFailed(
            "cuda_ep: Marlin weight repack cannot allocate during CUDA-graph capture; \
             the weight must be repacked during warmup before capture"
                .into(),
        ));
    }
    let bytes = repacked_bytes(n, k);
    let out = runtime.alloc_raw(bytes)?;
    if let Err(e) = launch_marlin_repack(runtime, packed, out, n, k, group_size) {
        // SAFETY: `out` was just allocated here and is otherwise unreferenced.
        let _ = unsafe { runtime.free_raw(out) };
        return Err(e);
    }
    let mut cache = repack_cache().lock().expect("marlin repack cache poisoned");
    // Another thread may have inserted the same key while we repacked; keep one.
    if let Some(entry) = cache.map.get(&key) {
        let winner = entry.ptr;
        drop(cache);
        // SAFETY: `out` is our just-allocated duplicate; free it once.
        let _ = unsafe { runtime.free_raw(out) };
        return Ok((winner, true));
    }
    cache.map.insert(
        key,
        RepackEntry {
            ptr: out,
            runtime: runtime.clone(),
        },
    );
    cache.order.push_back(key);
    while cache.order.len() > REPACK_CACHE_CAP {
        if let Some(evict) = cache.order.pop_front()
            && let Some(entry) = cache.map.remove(&evict)
        {
            // SAFETY: exclusively owned by the cache; freed once on eviction.
            let _ = unsafe { entry.runtime.free_raw(entry.ptr) };
        }
    }
    Ok((out, false))
}

/// Module-global pool of reusable device scratch buffers for the Marlin fused
/// prefill paths (the gate projection buffer and the RMS-norm normalized-
/// activation buffer). Like the repack cache, this exists so a **warm** replay
/// performs no allocation and stays CUDA-graph capture-safe: during capture M is
/// fixed, so warmup pre-allocates the exact size and replays reuse the same
/// device pointer. Buffers are keyed by `(ordinal, slot, bytes)` — the `slot`
/// discriminator guarantees two simultaneously-live scratches of equal size
/// within one op (e.g. normalized `[M,K]` and gate `[M,N]` when `K == N`) never
/// alias. Reuse across sequential ops is safe: each op fully consumes its
/// scratch before returning and execution is serial on the stream.
struct ScratchCache {
    map: HashMap<(u32, u32, usize), RepackEntry>,
    order: VecDeque<(u32, u32, usize)>,
}

const SCRATCH_CACHE_CAP: usize = 256;

static SCRATCH_CACHE: OnceLock<Mutex<ScratchCache>> = OnceLock::new();

fn scratch_cache() -> &'static Mutex<ScratchCache> {
    SCRATCH_CACHE.get_or_init(|| {
        Mutex::new(ScratchCache {
            map: HashMap::new(),
            order: VecDeque::new(),
        })
    })
}

/// Ensure a reusable device scratch buffer of `bytes` bytes exists for `slot`,
/// returning `(ptr, warm)`. `warm == true` means the buffer was already pooled
/// (no allocation this call ⇒ capture-safe). A cold miss while capturing is
/// rejected so the caller can fall back rather than allocate inside a capture.
/// The returned buffer's contents are undefined; callers fully overwrite it.
pub fn ensure_scratch(
    runtime: &Arc<CudaRuntime>,
    slot: u32,
    bytes: usize,
) -> Result<(CUdeviceptr, bool)> {
    let key = (runtime.ordinal(), slot, bytes);
    {
        let cache = scratch_cache()
            .lock()
            .expect("marlin scratch cache poisoned");
        if let Some(entry) = cache.map.get(&key) {
            return Ok((entry.ptr, true));
        }
    }
    if runtime.is_capturing()? {
        return Err(EpError::KernelFailed(
            "cuda_ep: Marlin fused scratch cannot allocate during CUDA-graph capture; \
             the buffer must be warmed at this M before capture"
                .into(),
        ));
    }
    let ptr = runtime.alloc_raw(bytes.max(1))?;
    let mut cache = scratch_cache()
        .lock()
        .expect("marlin scratch cache poisoned");
    if let Some(entry) = cache.map.get(&key) {
        let winner = entry.ptr;
        drop(cache);
        // SAFETY: `ptr` is our just-allocated duplicate; free it once.
        let _ = unsafe { runtime.free_raw(ptr) };
        return Ok((winner, true));
    }
    cache.map.insert(
        key,
        RepackEntry {
            ptr,
            runtime: runtime.clone(),
        },
    );
    cache.order.push_back(key);
    while cache.order.len() > SCRATCH_CACHE_CAP {
        if let Some(evict) = cache.order.pop_front()
            && let Some(entry) = cache.map.remove(&evict)
        {
            // SAFETY: exclusively owned by the cache; freed once on eviction.
            let _ = unsafe { entry.runtime.free_raw(entry.ptr) };
        }
    }
    Ok((ptr, false))
}

/// Repack ONNX `MatMulNBits` int4 weights (`[N, k_blocks, group/2]`, N-major
/// nibble packing) into the per-lane tensor-core layout documented above.
///
/// `group_size` is the quantization block size (16/32/64/128). `k` must be a
/// multiple of 16 and equal to `k_blocks * group_size`.
#[must_use]
pub fn repack_int4_weights(packed: &[u8], n: usize, k: usize, group_size: usize) -> Vec<u8> {
    assert_eq!(
        k % 16,
        0,
        "K must be a multiple of 16 for the Marlin repack"
    );
    assert_eq!(k % group_size, 0, "K must be a multiple of the group size");
    let k_blocks = k / group_size;
    let blob = group_size / 2;
    let slices = k / 16;
    let n_tiles = n.div_ceil(8);
    let mut out = vec![0u8; n_tiles * slices * 64];
    let code_at = |col: usize, kk: usize| -> u8 {
        if col >= n {
            return 0;
        }
        let block = kk / group_size;
        let within = kk % group_size;
        let byte = packed[(col * k_blocks + block) * blob + within / 2];
        if within & 1 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        }
    };
    for n_tile in 0..n_tiles {
        for slice in 0..slices {
            let kbase = slice * 16;
            let tile_base = (n_tile * slices + slice) * 64;
            for group_id in 0..8usize {
                let col = n_tile * 8 + group_id;
                for tid in 0..4usize {
                    let lo =
                        code_at(col, kbase + tid * 2) | (code_at(col, kbase + tid * 2 + 1) << 4);
                    let hi = code_at(col, kbase + tid * 2 + 8)
                        | (code_at(col, kbase + tid * 2 + 9) << 4);
                    let dst = tile_base + group_id * 8 + tid * 2;
                    out[dst] = lo;
                    out[dst + 1] = hi;
                }
            }
        }
    }
    out
}

/// CUDA source for the Marlin int4 tensor-core GEMM. Uses raw `mma.sync` inline
/// PTX (no `<mma.h>` / `crt/` dependency) and `cuda_fp16.h` only.
pub const MARLIN_GEMM_SRC: &str = r#"
#include <cuda_fp16.h>

// Per-group scale applied after the tensor-core accumulate keeps the fp32
// accumulator free of a K-varying scale (Marlin's key precision move). Weights
// enter the mma centered but unscaled: (code - zp) as fp16.

__device__ __forceinline__ unsigned pack_half2(__half a, __half b) {
    __half2 h = __halves2half2(a, b);
    return *reinterpret_cast<unsigned*>(&h);
}

__device__ __forceinline__ __half load_a(
    const __half* __restrict__ a, int row, int col, int m, int k) {
    return (row < m && col < k) ? a[(long)row * k + col] : __float2half(0.0f);
}

// Fused fp16 bias epilogue mirroring matmul_nbits::fold_bias_f16:
//   bias_post_round == 0 : fp16(acc + bias)          (native MatMulNBits bias)
//   bias_post_round != 0 : fp16(fp16(acc) + bias)    (folded standalone Add)
__device__ __forceinline__ __half fold_bias(
    float value, const __half* __restrict__ bias, int column, int bias_post_round) {
    __half rounded = __float2half(value);
    if (!bias) return rounded;
    float b = __half2float(bias[column]);
    if (bias_post_round) return __float2half(__half2float(rounded) + b);
    return __float2half(value + b);
}

// grid.x = ceil(N / N_PER_BLOCK), grid.y = ceil(M / 16); blockDim.x = 32*WARPS.
// One warp owns 8 output columns; one block owns 16 M rows x (8*WARPS) columns.
extern "C" __global__ void matmul_nbits_marlin_gemm_f16(
    const __half* __restrict__ activation,   // [M, K]
    const unsigned char* __restrict__ weights, // repacked, [N * (K/16) * 8] bytes
    const void* __restrict__ scales_raw,     // [N, k_blocks], fp16 or fp32
    const unsigned char* __restrict__ zero_points, // [N, ceil(k_blocks/2)] nibbles or null
    const __half* __restrict__ bias,         // [N] or per-row residual, or null
    __half* __restrict__ output,             // [M, N]
    const int m,
    const int k,
    const int n,
    const int k_blocks,
    const int group_size,
    const int scales_fp16,
    const int bias_post_round,
    const int bias_row_stride)
{
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int group_id = lane >> 2;      // 0..7
    const int tid = lane & 3;            // 0..3

    const int n_block = (int)blockIdx.x * (8 * (int)blockDim.x / 32);
    const int ncol0 = n_block + warp * 8;   // first output column of this warp
    const int m_tile = (int)blockIdx.y * 16;

    const int slices = k / 16;
    const int slices_per_group = group_size / 16;
    const int zp_row_bytes = (k_blocks + 1) >> 1;

    // B-operand column owned by this lane, and the two C-output columns.
    const int nb_col = ncol0 + group_id;
    const int out_col0 = ncol0 + tid * 2;
    const int out_col1 = out_col0 + 1;

    // Final fp32 accumulators (scale applied per group). c0,c2 -> out_col0;
    // c1,c3 -> out_col1. c0,c1 row group_id; c2,c3 row group_id+8.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int g = 0; g < k_blocks; ++g) {
        // Per-group scale for the two output columns and zero point for the
        // B column this lane feeds.
        float scale_a = 0.0f, scale_b = 0.0f;
        if (scales_fp16) {
            const __half* s = reinterpret_cast<const __half*>(scales_raw);
            if (out_col0 < n) scale_a = __half2float(s[(long)out_col0 * k_blocks + g]);
            if (out_col1 < n) scale_b = __half2float(s[(long)out_col1 * k_blocks + g]);
        } else {
            const float* s = reinterpret_cast<const float*>(scales_raw);
            if (out_col0 < n) scale_a = s[(long)out_col0 * k_blocks + g];
            if (out_col1 < n) scale_b = s[(long)out_col1 * k_blocks + g];
        }
        int zp = 8;
        if (zero_points && nb_col < n) {
            unsigned char byte = zero_points[(long)nb_col * zp_row_bytes + (g >> 1)];
            zp = (g & 1) ? (byte >> 4) : (byte & 15);
        }

        float frag0 = 0.0f, frag1 = 0.0f, frag2 = 0.0f, frag3 = 0.0f;
        for (int s_in = 0; s_in < slices_per_group; ++s_in) {
            const int slice = g * slices_per_group + s_in;
            const int kbase = slice * 16;

            // A fragment (16x16), row-major.
            const __half a0 = load_a(activation, m_tile + group_id,     kbase + tid * 2,     m, k);
            const __half a1 = load_a(activation, m_tile + group_id,     kbase + tid * 2 + 1, m, k);
            const __half a2 = load_a(activation, m_tile + group_id + 8, kbase + tid * 2,     m, k);
            const __half a3 = load_a(activation, m_tile + group_id + 8, kbase + tid * 2 + 1, m, k);
            const __half a4 = load_a(activation, m_tile + group_id,     kbase + tid * 2 + 8, m, k);
            const __half a5 = load_a(activation, m_tile + group_id,     kbase + tid * 2 + 9, m, k);
            const __half a6 = load_a(activation, m_tile + group_id + 8, kbase + tid * 2 + 8, m, k);
            const __half a7 = load_a(activation, m_tile + group_id + 8, kbase + tid * 2 + 9, m, k);
            const unsigned ra0 = pack_half2(a0, a1);
            const unsigned ra1 = pack_half2(a2, a3);
            const unsigned ra2 = pack_half2(a4, a5);
            const unsigned ra3 = pack_half2(a6, a7);

            // B fragment (16x8), col-major. Weights centered (code - zp),
            // unscaled. Coalesced: lane l reads bytes [2l, 2l+1] of the warp's
            // 64-byte n-tile chunk for this slice.
            const int n_tile_idx = (int)blockIdx.x * ((int)blockDim.x / 32) + warp;
            unsigned char blo = 0, bhi = 0;
            {
                const long base = ((long)n_tile_idx * slices + slice) * 64
                    + group_id * 8 + tid * 2;
                blo = weights[base];
                bhi = weights[base + 1];
            }
            const __half b0 = __float2half((float)(int)(blo & 15) - (float)zp);
            const __half b1 = __float2half((float)(int)(blo >> 4) - (float)zp);
            const __half b2 = __float2half((float)(int)(bhi & 15) - (float)zp);
            const __half b3 = __float2half((float)(int)(bhi >> 4) - (float)zp);
            const unsigned rb0 = pack_half2(b0, b1);
            const unsigned rb1 = pack_half2(b2, b3);

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(frag0), "+f"(frag1), "+f"(frag2), "+f"(frag3)
                : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1));
        }
        acc0 += frag0 * scale_a;
        acc1 += frag1 * scale_b;
        acc2 += frag2 * scale_a;
        acc3 += frag3 * scale_b;
    }

    // C fragment store. Rows: group_id (c0,c1) and group_id+8 (c2,c3).
    const int row0 = m_tile + group_id;
    const int row1 = m_tile + group_id + 8;
    if (row0 < m) {
        if (out_col0 < n) {
            const __half* rb = bias ? bias + (long)row0 * bias_row_stride : bias;
            output[(long)row0 * n + out_col0] = fold_bias(acc0, rb, out_col0, bias_post_round);
        }
        if (out_col1 < n) {
            const __half* rb = bias ? bias + (long)row0 * bias_row_stride : bias;
            output[(long)row0 * n + out_col1] = fold_bias(acc1, rb, out_col1, bias_post_round);
        }
    }
    if (row1 < m) {
        if (out_col0 < n) {
            const __half* rb = bias ? bias + (long)row1 * bias_row_stride : bias;
            output[(long)row1 * n + out_col0] = fold_bias(acc2, rb, out_col0, bias_post_round);
        }
        if (out_col1 < n) {
            const __half* rb = bias ? bias + (long)row1 * bias_row_stride : bias;
            output[(long)row1 * n + out_col1] = fold_bias(acc3, rb, out_col1, bias_post_round);
        }
    }
}

// Split-K variant: partitions the K (group) range across grid.z so more thread
// blocks are resident at small M, where the single-block-per-(N,M)-tile kernel
// leaves most SMs idle (e.g. N=5120 launches only ~160 blocks on 132 SMs, so
// weight-DRAM sits near 3% of peak — an occupancy stall, not a latency one).
// Each z owns a contiguous group range and writes its fp32 partial (scale
// already applied, NO bias) to partials[z, row, col]; a separate reduce kernel
// sums the z-slices in fixed order and applies the epilogue. Fixed order keeps
// the result deterministic run-to-run (capture-stable); the cross-K reorder is
// not bit-identical to the single-block kernel but stays within the f64-oracle
// tolerance (validated by the parity test).
extern "C" __global__ void matmul_nbits_marlin_gemm_f16_splitk(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ weights,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    float* __restrict__ partials,            // [split_k, M, N]
    const int m,
    const int k,
    const int n,
    const int k_blocks,
    const int group_size,
    const int scales_fp16,
    const int groups_per_split)
{
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int group_id = lane >> 2;
    const int tid = lane & 3;

    const int z = (int)blockIdx.z;
    const int g_begin = z * groups_per_split;
    int g_end = g_begin + groups_per_split;
    if (g_end > k_blocks) g_end = k_blocks;

    const int n_block = (int)blockIdx.x * (8 * (int)blockDim.x / 32);
    const int ncol0 = n_block + warp * 8;
    const int m_tile = (int)blockIdx.y * 16;

    const int slices = k / 16;
    const int slices_per_group = group_size / 16;
    const int zp_row_bytes = (k_blocks + 1) >> 1;

    const int nb_col = ncol0 + group_id;
    const int out_col0 = ncol0 + tid * 2;
    const int out_col1 = out_col0 + 1;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int g = g_begin; g < g_end; ++g) {
        float scale_a = 0.0f, scale_b = 0.0f;
        if (scales_fp16) {
            const __half* s = reinterpret_cast<const __half*>(scales_raw);
            if (out_col0 < n) scale_a = __half2float(s[(long)out_col0 * k_blocks + g]);
            if (out_col1 < n) scale_b = __half2float(s[(long)out_col1 * k_blocks + g]);
        } else {
            const float* s = reinterpret_cast<const float*>(scales_raw);
            if (out_col0 < n) scale_a = s[(long)out_col0 * k_blocks + g];
            if (out_col1 < n) scale_b = s[(long)out_col1 * k_blocks + g];
        }
        int zp = 8;
        if (zero_points && nb_col < n) {
            unsigned char byte = zero_points[(long)nb_col * zp_row_bytes + (g >> 1)];
            zp = (g & 1) ? (byte >> 4) : (byte & 15);
        }

        float frag0 = 0.0f, frag1 = 0.0f, frag2 = 0.0f, frag3 = 0.0f;
        for (int s_in = 0; s_in < slices_per_group; ++s_in) {
            const int slice = g * slices_per_group + s_in;
            const int kbase = slice * 16;
            const __half a0 = load_a(activation, m_tile + group_id,     kbase + tid * 2,     m, k);
            const __half a1 = load_a(activation, m_tile + group_id,     kbase + tid * 2 + 1, m, k);
            const __half a2 = load_a(activation, m_tile + group_id + 8, kbase + tid * 2,     m, k);
            const __half a3 = load_a(activation, m_tile + group_id + 8, kbase + tid * 2 + 1, m, k);
            const __half a4 = load_a(activation, m_tile + group_id,     kbase + tid * 2 + 8, m, k);
            const __half a5 = load_a(activation, m_tile + group_id,     kbase + tid * 2 + 9, m, k);
            const __half a6 = load_a(activation, m_tile + group_id + 8, kbase + tid * 2 + 8, m, k);
            const __half a7 = load_a(activation, m_tile + group_id + 8, kbase + tid * 2 + 9, m, k);
            const unsigned ra0 = pack_half2(a0, a1);
            const unsigned ra1 = pack_half2(a2, a3);
            const unsigned ra2 = pack_half2(a4, a5);
            const unsigned ra3 = pack_half2(a6, a7);

            const int n_tile_idx = (int)blockIdx.x * ((int)blockDim.x / 32) + warp;
            const long base = ((long)n_tile_idx * slices + slice) * 64
                + group_id * 8 + tid * 2;
            const unsigned char blo = weights[base];
            const unsigned char bhi = weights[base + 1];
            const __half b0 = __float2half((float)(int)(blo & 15) - (float)zp);
            const __half b1 = __float2half((float)(int)(blo >> 4) - (float)zp);
            const __half b2 = __float2half((float)(int)(bhi & 15) - (float)zp);
            const __half b3 = __float2half((float)(int)(bhi >> 4) - (float)zp);
            const unsigned rb0 = pack_half2(b0, b1);
            const unsigned rb1 = pack_half2(b2, b3);

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(frag0), "+f"(frag1), "+f"(frag2), "+f"(frag3)
                : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1));
        }
        acc0 += frag0 * scale_a;
        acc1 += frag1 * scale_b;
        acc2 += frag2 * scale_a;
        acc3 += frag3 * scale_b;
    }

    const int row0 = m_tile + group_id;
    const int row1 = m_tile + group_id + 8;
    const long zbase = (long)z * m * n;
    if (row0 < m) {
        if (out_col0 < n) partials[zbase + (long)row0 * n + out_col0] = acc0;
        if (out_col1 < n) partials[zbase + (long)row0 * n + out_col1] = acc1;
    }
    if (row1 < m) {
        if (out_col0 < n) partials[zbase + (long)row1 * n + out_col0] = acc2;
        if (out_col1 < n) partials[zbase + (long)row1 * n + out_col1] = acc3;
    }
}

// Fixed-order reduction of the split-K partials plus the fp16 bias epilogue.
// One thread per output element; sums z = 0..split_k-1 deterministically.
extern "C" __global__ void matmul_nbits_marlin_splitk_reduce(
    const float* __restrict__ partials,      // [split_k, M, N]
    const __half* __restrict__ bias,
    __half* __restrict__ output,             // [M, N]
    const int m,
    const int n,
    const int split_k,
    const int bias_post_round,
    const int bias_row_stride)
{
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)m * n;
    if (idx >= total) return;
    const int row = (int)(idx / n);
    const int col = (int)(idx - (long)row * n);
    float acc = 0.0f;
    for (int z = 0; z < split_k; ++z) acc += partials[(long)z * total + idx];
    const __half* rb = bias ? bias + (long)row * bias_row_stride : bias;
    output[idx] = fold_bias(acc, rb, col, bias_post_round);
}

// Device-side repack: reorder ONNX packed int4 (N-major nibbles) into the
extern "C" __global__ void matmul_nbits_marlin_repack(
    const unsigned char* __restrict__ packed, // [N, k_blocks, group/2]
    unsigned char* __restrict__ out,          // [n_tiles * (K/16) * 64]
    const int n,
    const int k,
    const int k_blocks,
    const int group_size)
{
    const int slices = k / 16;
    const int blob = group_size / 2;
    const int n_tiles = (n + 7) / 8;
    const int total = n_tiles * 8 * slices;
    const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= total) return;

    const int slice = idx % slices;
    const int col = idx / slices;            // 0 .. n_tiles*8-1 (col == global col)
    const int n_tile = col >> 3;
    const int group_id = col & 7;
    const int kbase = slice * 16;
    const long tile_base = ((long)n_tile * slices + slice) * 64 + group_id * 8;

    // code(kk) for this column, 0 if the column is padding (col >= n).
    // packed nibble: byte[(col*k_blocks + kk/group)*blob + (kk%group)/2],
    // low nibble for even kk%group.
    #define MARLIN_CODE(kk) ( (col < n) ? ( \
        ((kk % group_size) & 1) \
            ? (packed[((long)col * k_blocks + (kk) / group_size) * blob + ((kk) % group_size) / 2] >> 4) \
            : (packed[((long)col * k_blocks + (kk) / group_size) * blob + ((kk) % group_size) / 2] & 15) \
        ) : 0 )
#pragma unroll
    for (int tid = 0; tid < 4; ++tid) {
        const unsigned char lo =
            (unsigned char)(MARLIN_CODE(kbase + tid * 2) | (MARLIN_CODE(kbase + tid * 2 + 1) << 4));
        const unsigned char hi =
            (unsigned char)(MARLIN_CODE(kbase + tid * 2 + 8) | (MARLIN_CODE(kbase + tid * 2 + 9) << 4));
        out[tile_base + tid * 2] = lo;
        out[tile_base + tid * 2 + 1] = hi;
    }
    #undef MARLIN_CODE
}
"#;

/// Size in bytes of the repacked weight buffer for the given dims.
#[must_use]
pub fn repacked_bytes(n: usize, k: usize) -> usize {
    n.div_ceil(8) * (k / 16) * 64
}

/// Launch the device-side weight repack (original ONNX packed int4 → the
/// tensor-core layout). `out` must be at least [`repacked_bytes`].
pub fn launch_marlin_repack(
    runtime: &CudaRuntime,
    packed: CUdeviceptr,
    out: CUdeviceptr,
    n: usize,
    k: usize,
    group_size: usize,
) -> Result<()> {
    let k_blocks = k / group_size;
    let function = runtime.nvrtc_function(MARLIN_MODULE, MARLIN_GEMM_SRC, MARLIN_REPACK_ENTRY)?;
    let packed_ptr = cuptr(packed as usize as *const c_void);
    let out_ptr = cuptr(out as usize as *const c_void);
    let n_i32 = i32::try_from(n).map_err(|_| overflow("N", n))?;
    let k_i32 = i32::try_from(k).map_err(|_| overflow("K", k))?;
    let k_blocks_i32 = i32::try_from(k_blocks).map_err(|_| overflow("k_blocks", k_blocks))?;
    let group_size_i32 =
        i32::try_from(group_size).map_err(|_| overflow("group_size", group_size))?;
    let total = (n.div_ceil(8) * 8 * (k / 16)) as u32;
    const THREADS: u32 = 256;
    let grid = total.div_ceil(THREADS).max(1);
    let mut builder = runtime.stream().launch_builder(&function);
    builder
        .arg(&packed_ptr)
        .arg(&out_ptr)
        .arg(&n_i32)
        .arg(&k_i32)
        .arg(&k_blocks_i32)
        .arg(&group_size_i32);
    // SAFETY: static grid; `packed` covers [N,k_blocks,group/2] bytes and `out`
    // covers repacked_bytes(n,k); the kernel bounds-checks the thread index.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|err| driver_err("launch MatMulNBits Marlin repack", err))
}

/// Launch parameters for [`launch_marlin_gemm`].
#[derive(Clone, Copy)]
pub struct MarlinGemmArgs {
    pub activation: CUdeviceptr,
    pub weights: CUdeviceptr,
    pub scales: CUdeviceptr,
    pub zero_points: Option<CUdeviceptr>,
    pub bias: Option<CUdeviceptr>,
    pub output: CUdeviceptr,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub group_size: usize,
    pub scales_fp16: bool,
    pub bias_post_round: bool,
    pub bias_row_stride: usize,
}

/// Launch the Marlin int4 tensor-core GEMM on the runtime's stream. The launch
/// grid is a pure function of `(M, N)`, so it is CUDA-graph capture-safe.
#[allow(clippy::too_many_arguments)]
pub fn launch_marlin_gemm(runtime: &CudaRuntime, args: &MarlinGemmArgs) -> Result<()> {
    if !device_supports_marlin(runtime.capabilities().compute_capability()) {
        return Err(EpError::KernelFailed(
            "cuda_ep: Marlin int4 tensor-core GEMM requires compute capability >= 8.0".into(),
        ));
    }
    runtime.require_nvrtc_half_headers("MatMulNBits Marlin int4 tensor-core GEMM")?;
    if !args.k.is_multiple_of(16) || !args.k.is_multiple_of(args.group_size) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep: Marlin GEMM requires K ({}) divisible by 16 and by group_size ({})",
            args.k, args.group_size
        )));
    }
    let k_blocks = args.k / args.group_size;
    let function = runtime.nvrtc_function(MARLIN_MODULE, MARLIN_GEMM_SRC, MARLIN_GEMM_ENTRY)?;

    let activation_ptr = cuptr(args.activation as usize as *const c_void);
    let weights_ptr = cuptr(args.weights as usize as *const c_void);
    let scales_ptr = cuptr(args.scales as usize as *const c_void);
    let zero_points_ptr = args.zero_points.unwrap_or(0);
    let bias_ptr = args.bias.unwrap_or(0);
    let output_ptr = cuptr(args.output as usize as *const c_void);

    let m_i32 = i32::try_from(args.m).map_err(|_| overflow("M", args.m))?;
    let k_i32 = i32::try_from(args.k).map_err(|_| overflow("K", args.k))?;
    let n_i32 = i32::try_from(args.n).map_err(|_| overflow("N", args.n))?;
    let k_blocks_i32 = i32::try_from(k_blocks).map_err(|_| overflow("k_blocks", k_blocks))?;
    let group_size_i32 =
        i32::try_from(args.group_size).map_err(|_| overflow("group_size", args.group_size))?;
    let scales_fp16_flag = args.scales_fp16 as i32;
    let bias_post_round_flag = args.bias_post_round as i32;
    let bias_row_stride_i32 = i32::try_from(args.bias_row_stride)
        .map_err(|_| overflow("bias_row_stride", args.bias_row_stride))?;

    let grid_x = (args.n as u32).div_ceil(MARLIN_N_PER_BLOCK).max(1);
    let grid_y = (args.m as u32).div_ceil(MARLIN_M_PER_BLOCK).max(1);

    let mut builder = runtime.stream().launch_builder(&function);
    builder
        .arg(&activation_ptr)
        .arg(&weights_ptr)
        .arg(&scales_ptr)
        .arg(&zero_points_ptr)
        .arg(&bias_ptr)
        .arg(&output_ptr)
        .arg(&m_i32)
        .arg(&k_i32)
        .arg(&n_i32)
        .arg(&k_blocks_i32)
        .arg(&group_size_i32)
        .arg(&scales_fp16_flag)
        .arg(&bias_post_round_flag)
        .arg(&bias_row_stride_i32);
    // SAFETY: static grid, all pointers are live device allocations sized by the
    // validated dims; the kernel reads only in-bounds elements (guarded loads).
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (32 * MARLIN_WARPS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|err| driver_err("launch MatMulNBits Marlin int4 tensor-core GEMM", err))
}

fn overflow(name: &str, value: usize) -> EpError {
    EpError::KernelFailed(format!(
        "cuda_ep: Marlin GEMM {name}={value} exceeds i32 range"
    ))
}

/// Number of fp32 elements in the split-K partials buffer for `(split_k, M, N)`.
#[must_use]
pub fn splitk_partials_len(split_k: usize, m: usize, n: usize) -> usize {
    split_k * m * n
}

/// Pick a split-K factor that fills the SMs at small M without over-splitting.
/// Returns 1 (no split) when the base grid already covers enough SMs or the K
/// range is too short to divide usefully. `sm_count` is the device SM count.
#[must_use]
pub fn choose_split_k(m: usize, n: usize, k_blocks: usize, sm_count: u32) -> usize {
    let base_blocks =
        (n as u32).div_ceil(MARLIN_N_PER_BLOCK) * (m as u32).div_ceil(MARLIN_M_PER_BLOCK);
    // Aim for ~2 waves of blocks. If already there, don't split.
    let target = sm_count.saturating_mul(2).max(1);
    if base_blocks >= target || k_blocks < 2 {
        return 1;
    }
    let want = target.div_ceil(base_blocks.max(1));
    // Cap so each split still owns >= 2 groups, and clamp to a small max.
    let max_by_groups = (k_blocks / 2).max(1) as u32;
    want.clamp(1, max_by_groups).min(8) as usize
}

/// Launch the split-K Marlin GEMM (fp32 partials) followed by the fixed-order
/// reduce+epilogue into fp16 `output`. `partials` must hold at least
/// [`splitk_partials_len`] fp32 elements. Both launches use a static grid, so
/// the pair is CUDA-graph capture-safe once `partials` is pre-allocated.
#[allow(clippy::too_many_arguments)]
pub fn launch_marlin_gemm_splitk(
    runtime: &CudaRuntime,
    args: &MarlinGemmArgs,
    split_k: usize,
    partials: CUdeviceptr,
) -> Result<()> {
    if !device_supports_marlin(runtime.capabilities().compute_capability()) {
        return Err(EpError::KernelFailed(
            "cuda_ep: Marlin split-K GEMM requires compute capability >= 8.0".into(),
        ));
    }
    runtime.require_nvrtc_half_headers("MatMulNBits Marlin split-K GEMM")?;
    if !args.k.is_multiple_of(16) || !args.k.is_multiple_of(args.group_size) {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep: Marlin GEMM requires K ({}) divisible by 16 and by group_size ({})",
            args.k, args.group_size
        )));
    }
    let k_blocks = args.k / args.group_size;
    let split_k = split_k.max(1);
    let groups_per_split = k_blocks.div_ceil(split_k);

    let gemm = runtime.nvrtc_function(MARLIN_MODULE, MARLIN_GEMM_SRC, MARLIN_GEMM_SPLITK_ENTRY)?;
    let reduce =
        runtime.nvrtc_function(MARLIN_MODULE, MARLIN_GEMM_SRC, MARLIN_SPLITK_REDUCE_ENTRY)?;

    let activation_ptr = cuptr(args.activation as usize as *const c_void);
    let weights_ptr = cuptr(args.weights as usize as *const c_void);
    let scales_ptr = cuptr(args.scales as usize as *const c_void);
    let zero_points_ptr = args.zero_points.unwrap_or(0);
    let bias_ptr = args.bias.unwrap_or(0);
    let output_ptr = cuptr(args.output as usize as *const c_void);
    let partials_ptr = cuptr(partials as usize as *const c_void);

    let m_i32 = i32::try_from(args.m).map_err(|_| overflow("M", args.m))?;
    let k_i32 = i32::try_from(args.k).map_err(|_| overflow("K", args.k))?;
    let n_i32 = i32::try_from(args.n).map_err(|_| overflow("N", args.n))?;
    let k_blocks_i32 = i32::try_from(k_blocks).map_err(|_| overflow("k_blocks", k_blocks))?;
    let group_size_i32 =
        i32::try_from(args.group_size).map_err(|_| overflow("group_size", args.group_size))?;
    let scales_fp16_flag = args.scales_fp16 as i32;
    let groups_per_split_i32 = i32::try_from(groups_per_split)
        .map_err(|_| overflow("groups_per_split", groups_per_split))?;
    let split_k_i32 = i32::try_from(split_k).map_err(|_| overflow("split_k", split_k))?;
    let bias_post_round_flag = args.bias_post_round as i32;
    let bias_row_stride_i32 = i32::try_from(args.bias_row_stride)
        .map_err(|_| overflow("bias_row_stride", args.bias_row_stride))?;

    let grid_x = (args.n as u32).div_ceil(MARLIN_N_PER_BLOCK).max(1);
    let grid_y = (args.m as u32).div_ceil(MARLIN_M_PER_BLOCK).max(1);

    let mut builder = runtime.stream().launch_builder(&gemm);
    builder
        .arg(&activation_ptr)
        .arg(&weights_ptr)
        .arg(&scales_ptr)
        .arg(&zero_points_ptr)
        .arg(&partials_ptr)
        .arg(&m_i32)
        .arg(&k_i32)
        .arg(&n_i32)
        .arg(&k_blocks_i32)
        .arg(&group_size_i32)
        .arg(&scales_fp16_flag)
        .arg(&groups_per_split_i32);
    // SAFETY: static grid; partials holds split_k*M*N fp32 written in-bounds.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_x, grid_y, split_k as u32),
            block_dim: (32 * MARLIN_WARPS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|err| driver_err("launch MatMulNBits Marlin split-K GEMM", err))?;

    let total = (args.m as u32).saturating_mul(args.n as u32).max(1);
    let reduce_block = 256u32;
    let reduce_grid = total.div_ceil(reduce_block);
    let mut rbuilder = runtime.stream().launch_builder(&reduce);
    rbuilder
        .arg(&partials_ptr)
        .arg(&bias_ptr)
        .arg(&output_ptr)
        .arg(&m_i32)
        .arg(&n_i32)
        .arg(&split_k_i32)
        .arg(&bias_post_round_flag)
        .arg(&bias_row_stride_i32);
    // SAFETY: static grid; one thread per in-bounds output element.
    unsafe {
        rbuilder.launch(LaunchConfig {
            grid_dim: (reduce_grid, 1, 1),
            block_dim: (reduce_block, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|err| driver_err("launch MatMulNBits Marlin split-K reduce", err))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repack must be a bijection back to the original nibble codes for every
    /// group size, and must place each lane's four nibbles where the kernel reads
    /// them (`(n*(K/16)+slice)*8 + tid*2 + {0,1}`).
    #[test]
    fn repack_roundtrip_all_group_sizes() {
        for &group in &[16usize, 32, 64, 128] {
            let n = 5;
            let k = group * 3; // k_blocks = 3
            let k_blocks = k / group;
            let blob = group / 2;
            // Deterministic codes in 0..15.
            let mut packed = vec![0u8; n * k_blocks * blob];
            let mut codes = vec![0u8; n * k];
            let mut state = 0x1234_5678u32;
            let mut next = || {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 24) & 0x0f) as u8
            };
            for col in 0..n {
                for kk in 0..k {
                    let c = next();
                    codes[col * k + kk] = c;
                }
            }
            for col in 0..n {
                for block in 0..k_blocks {
                    for pair in 0..blob {
                        let lo = codes[col * k + block * group + pair * 2] & 15;
                        let hi = codes[col * k + block * group + pair * 2 + 1] & 15;
                        packed[(col * k_blocks + block) * blob + pair] = lo | (hi << 4);
                    }
                }
            }

            let repacked = repack_int4_weights(&packed, n, k, group);
            let slices = k / 16;
            let n_tiles = n.div_ceil(8);
            assert_eq!(repacked.len(), n_tiles * slices * 64);

            for n_tile in 0..n_tiles {
                for slice in 0..slices {
                    let tile_base = (n_tile * slices + slice) * 64;
                    let kbase = slice * 16;
                    for group_id in 0..8usize {
                        let col = n_tile * 8 + group_id;
                        for tid in 0..4usize {
                            let lo = repacked[tile_base + group_id * 8 + tid * 2];
                            let hi = repacked[tile_base + group_id * 8 + tid * 2 + 1];
                            let expect = |kk: usize| -> u8 {
                                if col < n { codes[col * k + kk] & 15 } else { 0 }
                            };
                            assert_eq!(lo & 15, expect(kbase + tid * 2));
                            assert_eq!(lo >> 4, expect(kbase + tid * 2 + 1));
                            assert_eq!(hi & 15, expect(kbase + tid * 2 + 8));
                            assert_eq!(hi >> 4, expect(kbase + tid * 2 + 9));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn device_support_gate() {
        assert!(device_supports_marlin((8, 0)));
        assert!(device_supports_marlin((9, 0)));
        assert!(!device_supports_marlin((7, 5)));
        assert!(!device_supports_marlin((7, 0)));
    }

    // ---- GPU parity + microbench (require a live CUDA device; #[ignore]) ----

    use std::sync::Arc;

    use crate::runtime::CudaRuntime;

    fn runtime() -> Option<Arc<CudaRuntime>> {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let rt = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(previous);
        rt
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

    /// Result of a parity run: worst absolute error, worst relative error, and
    /// the max magnitude of the oracle output (for scaling the tolerance).
    struct Parity {
        worst_abs: f32,
        worst_rel: f32,
        max_out: f32,
        all_finite: bool,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_marlin_parity(
        rt: &Arc<CudaRuntime>,
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
        scales_fp16: bool,
        explicit_zp: bool,
        seed: u64,
    ) -> Parity {
        run_marlin_parity_impl(rt, m, k, n, group_size, scales_fp16, explicit_zp, seed, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_marlin_parity_impl(
        rt: &Arc<CudaRuntime>,
        m: usize,
        k: usize,
        n: usize,
        group_size: usize,
        scales_fp16: bool,
        explicit_zp: bool,
        seed: u64,
        split_k: usize,
    ) -> Parity {
        use half::f16;

        let k_blocks = k / group_size;
        let blob = group_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);

        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // fp16 activations + their fp16-as-f32 twin for the shared-input oracle.
        let mut activation_f16 = vec![f16::ZERO; m * k];
        let mut activation_ref = vec![0.0f32; m * k];
        for (h, f) in activation_f16.iter_mut().zip(activation_ref.iter_mut()) {
            let v = f16::from_f32(next());
            *h = v;
            *f = v.to_f32();
        }

        // int4 codes 0..15, packed in the ONNX N-major layout.
        let mut codes = vec![0u8; n * k];
        for c in codes.iter_mut() {
            *c = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob {
                    let lo = codes[col * k + block * group_size + pair * 2] & 15;
                    let hi = codes[col * k + block * group_size + pair * 2 + 1] & 15;
                    packed[(col * k_blocks + block) * blob + pair] = lo | (hi << 4);
                }
            }
        }
        let repacked = repack_int4_weights(&packed, n, k, group_size);

        // Asymmetric zero points (nibble-packed) or symmetric default 8.
        let mut zp_codes = vec![8i32; n * k_blocks];
        let mut zp_packed = vec![0u8; n * zp_row_bytes];
        if explicit_zp {
            for c in zp_codes.iter_mut() {
                *c = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as i32;
            }
            for col in 0..n {
                for block in 0..k_blocks {
                    let code = (zp_codes[col * k_blocks + block] & 15) as u8;
                    let byte = &mut zp_packed[col * zp_row_bytes + block / 2];
                    if block & 1 == 0 {
                        *byte = (*byte & 0xf0) | code;
                    } else {
                        *byte = (*byte & 0x0f) | (code << 4);
                    }
                }
            }
        }

        // Per (col, block) scales, rounded to storage dtype (shared by oracle).
        let mut scale_ref = vec![0.0f32; n * k_blocks];
        let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
        let mut scale_f32 = vec![0.0f32; n * k_blocks];
        for i in 0..n * k_blocks {
            let raw = 0.015 + 0.01 * (next() * 0.5 + 0.5);
            if scales_fp16 {
                let h = f16::from_f32(raw);
                scale_f16[i] = h;
                scale_ref[i] = h.to_f32();
            } else {
                scale_f32[i] = raw;
                scale_ref[i] = raw;
            }
        }

        // f64 dequant-and-matmul oracle: Y[r,c] = sum_k A[r,k]*(code-zp)*scale.
        let mut expected = vec![0.0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0.0f64;
                for block in 0..k_blocks {
                    let scale = scale_ref[c * k_blocks + block] as f64;
                    let zero = zp_codes[c * k_blocks + block];
                    for within in 0..group_size {
                        let depth = block * group_size + within;
                        let q = codes[c * k + depth] as i32 - zero;
                        acc += activation_ref[r * k + depth] as f64 * q as f64 * scale;
                    }
                }
                expected[r * n + c] = acc as f32;
            }
        }

        let activation_dev = rt.alloc_raw(activation_f16.len() * 2).unwrap();
        let weights_dev = rt.alloc_raw(repacked.len()).unwrap();
        let scales_dev = rt
            .alloc_raw(n * k_blocks * if scales_fp16 { 2 } else { 4 })
            .unwrap();
        let zp_dev = rt.alloc_raw(zp_packed.len().max(1)).unwrap();
        let output_dev = rt.alloc_raw(m * n * 2).unwrap();

        // SAFETY: each device buffer was sized for its source slice.
        unsafe {
            rt.htod(as_bytes(&activation_f16), activation_dev).unwrap();
            rt.htod(&repacked, weights_dev).unwrap();
            if scales_fp16 {
                rt.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            } else {
                rt.htod(as_bytes(&scale_f32), scales_dev).unwrap();
            }
            if explicit_zp {
                rt.htod(&zp_packed, zp_dev).unwrap();
            }
        }

        let args = MarlinGemmArgs {
            activation: activation_dev,
            weights: weights_dev,
            scales: scales_dev,
            zero_points: explicit_zp.then_some(zp_dev),
            bias: None,
            output: output_dev,
            m,
            k,
            n,
            group_size,
            scales_fp16,
            bias_post_round: false,
            bias_row_stride: 0,
        };
        if split_k > 1 {
            let partials_len = splitk_partials_len(split_k, m, n);
            let partials_dev = rt.alloc_raw(partials_len * 4).unwrap();
            launch_marlin_gemm_splitk(rt, &args, split_k, partials_dev).unwrap();
            rt.synchronize().unwrap();
            // SAFETY: allocated above, freed once.
            unsafe {
                rt.free_raw(partials_dev).unwrap();
            }
        } else {
            launch_marlin_gemm(rt, &args).unwrap();
            rt.synchronize().unwrap();
        }

        let mut got = vec![f16::ZERO; m * n];
        // SAFETY: output_dev holds m*n fp16 values.
        unsafe {
            rt.dtoh(as_bytes_mut(&mut got), output_dev).unwrap();
        }

        // SAFETY: each pointer came from alloc_raw above and is freed once.
        unsafe {
            rt.free_raw(activation_dev).unwrap();
            rt.free_raw(weights_dev).unwrap();
            rt.free_raw(scales_dev).unwrap();
            rt.free_raw(zp_dev).unwrap();
            rt.free_raw(output_dev).unwrap();
        }

        let mut p = Parity {
            worst_abs: 0.0,
            worst_rel: 0.0,
            max_out: 0.0,
            all_finite: true,
        };
        for (g, e) in got.iter().zip(expected.iter()) {
            let g = g.to_f32();
            if !g.is_finite() {
                p.all_finite = false;
            }
            let abs = (g - e).abs();
            let rel = abs / e.abs().max(1e-1);
            p.worst_abs = p.worst_abs.max(abs);
            p.worst_rel = p.worst_rel.max(rel);
            p.max_out = p.max_out.max(e.abs());
        }
        p
    }

    /// GPU parity vs an f64 dequant→GEMM oracle across M, group sizes, scale
    /// dtype, and symmetric/asymmetric zero points. The relayout reorders partial
    /// sums, so the contract is tolerance-based (not byte-exact): fp16 activation
    /// × fp16-centered weight with fp32 tensor-core accumulation and per-group
    /// scale-after-accumulate. Relative tolerance is generous because the oracle
    /// runs in f64; the residual reflects only fp16 input rounding + accumulation
    /// order, which is exactly what Chew's numerics gate assesses.
    #[test]
    #[ignore = "requires a live CUDA device (SM80+)"]
    fn marlin_parity_vs_f64_oracle() {
        let Some(rt) = runtime() else {
            eprintln!("skipping: CUDA runtime unavailable");
            return;
        };
        if rt.require_nvrtc_half_headers("marlin").is_err() {
            eprintln!("skipping: fp16 NVRTC headers unavailable");
            return;
        }
        if !device_supports_marlin(rt.capabilities().compute_capability()) {
            eprintln!("skipping: device is not SM80+");
            return;
        }
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        // (M, K, N) picks: M sweeps the 1->cliff region; K/N cover ragged tails.
        for &(m, k, n) in &[
            (1usize, 4096usize, 128usize),
            (2, 4096, 96),
            (7, 2048, 130),
            (16, 4096, 256),
            (33, 1024, 64),
            (64, 1536, 200),
        ] {
            for &group in &[32usize, 128, 64, 16] {
                if k % group != 0 {
                    continue;
                }
                for &scales_fp16 in &[true, false] {
                    for &zp in &[false, true] {
                        seed = seed.wrapping_add(0x1000);
                        let p = run_marlin_parity(&rt, m, k, n, group, scales_fp16, zp, seed);
                        assert!(
                            p.all_finite,
                            "non-finite output for M={m} K={k} N={n} group={group} zp={zp}"
                        );
                        // Tolerance: abs scaled by output magnitude. fp16 mantissa
                        // is ~2^-11; with K up to 4096 accumulation the relative
                        // error stays well under 2%.
                        let tol = 2e-2 * p.max_out.max(1.0);
                        assert!(
                            p.worst_abs <= tol,
                            "M={m} K={k} N={n} group={group} scales_fp16={scales_fp16} zp={zp}: \
                             worst_abs={:.4} > tol={:.4} (worst_rel={:.4}, max_out={:.2})",
                            p.worst_abs,
                            tol,
                            p.worst_rel,
                            p.max_out
                        );
                    }
                }
            }
        }
    }

    /// Split-K parity vs the same f64 oracle. Split-K sums per-K-range partials
    /// in a fixed order, so it reorders the accumulation relative to the single
    /// block kernel — still within the tolerance-based numerics contract, and
    /// deterministic (capture-stable). Sweeps a couple of split factors.
    #[test]
    #[ignore = "requires a live CUDA device (SM80+)"]
    fn marlin_splitk_parity_vs_f64_oracle() {
        let Some(rt) = runtime() else {
            eprintln!("skipping: CUDA runtime unavailable");
            return;
        };
        if rt.require_nvrtc_half_headers("marlin").is_err()
            || !device_supports_marlin(rt.capabilities().compute_capability())
        {
            eprintln!("skipping: no SM80+ device / headers");
            return;
        }
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for &(m, k, n) in &[
            (1usize, 4096usize, 128usize),
            (8, 2048, 256),
            (5, 4096, 130),
        ] {
            for &group in &[32usize, 128, 64] {
                if k % group != 0 {
                    continue;
                }
                for &split_k in &[2usize, 4, 8] {
                    for &zp in &[false, true] {
                        seed = seed.wrapping_add(0x1000);
                        let p =
                            run_marlin_parity_impl(&rt, m, k, n, group, true, zp, seed, split_k);
                        assert!(
                            p.all_finite,
                            "non-finite split-K output M={m} K={k} N={n} group={group} \
                             split_k={split_k} zp={zp}"
                        );
                        let tol = 2e-2 * p.max_out.max(1.0);
                        assert!(
                            p.worst_abs <= tol,
                            "split-K M={m} K={k} N={n} group={group} split_k={split_k} zp={zp}: \
                             worst_abs={:.4} > tol={:.4} (max_out={:.2})",
                            p.worst_abs,
                            tol,
                            p.max_out
                        );
                    }
                }
            }
        }
    }

    /// Split-K determinism: two runs of the same split-K launch must be
    /// byte-identical (fixed reduction order), which is what makes it safe under
    /// CUDA-graph capture.
    #[test]
    #[ignore = "requires a live CUDA device (SM80+)"]
    fn marlin_splitk_is_deterministic() {
        let Some(rt) = runtime() else {
            eprintln!("skipping: CUDA runtime unavailable");
            return;
        };
        if rt.require_nvrtc_half_headers("marlin").is_err()
            || !device_supports_marlin(rt.capabilities().compute_capability())
        {
            eprintln!("skipping: no SM80+ device / headers");
            return;
        }
        let a = run_marlin_parity_impl(&rt, 8, 4096, 256, 128, true, true, 0xabcd, 4);
        let b = run_marlin_parity_impl(&rt, 8, 4096, 256, 128, true, true, 0xabcd, 4);
        assert_eq!(
            a.worst_abs, b.worst_abs,
            "split-K reduction must be deterministic run-to-run"
        );
    }

    #[test]
    #[ignore = "requires a live CUDA device (SM80+); prints timing"]
    fn marlin_bandwidth_microbench() {
        use half::f16;

        let Some(rt) = runtime() else {
            eprintln!("skipping: CUDA runtime unavailable");
            return;
        };
        if rt.require_nvrtc_half_headers("marlin").is_err()
            || !device_supports_marlin(rt.capabilities().compute_capability())
        {
            eprintln!("skipping: no SM80+ device / headers");
            return;
        }
        let peak_gbps: f64 = std::env::var("MARLIN_PEAK_GBPS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(4800.0);
        let iters: usize = 100;

        // Qwen2.5-14B-ish projections: gate/up (K=5120,N=13824) and a square-ish
        // attention proj (K=5120,N=5120). Sweep M across the decode->prefill range.
        for &(k, n) in &[(5120usize, 5120usize), (5120, 13824)] {
            let group = 128usize;
            let k_blocks = k / group;
            let blob = group / 2;
            // Random weights (repacked) + fp16 scales; content is irrelevant to timing.
            let packed = vec![0x42u8; n * k_blocks * blob];
            let repacked = repack_int4_weights(&packed, n, k, group);
            let scales = vec![f16::from_f32(0.02); n * k_blocks];
            let weights_dev = rt.alloc_raw(repacked.len()).unwrap();
            let scales_dev = rt.alloc_raw(scales.len() * 2).unwrap();
            // SAFETY: buffers sized for their slices.
            unsafe {
                rt.htod(&repacked, weights_dev).unwrap();
                rt.htod(as_bytes(&scales), scales_dev).unwrap();
            }
            for &m in &[1usize, 2, 8, 32, 128] {
                let activation = vec![f16::from_f32(0.01); m * k];
                let activation_dev = rt.alloc_raw(activation.len() * 2).unwrap();
                let output_dev = rt.alloc_raw(m * n * 2).unwrap();
                // SAFETY: buffers sized for their slices.
                unsafe {
                    rt.htod(as_bytes(&activation), activation_dev).unwrap();
                }
                let args = MarlinGemmArgs {
                    activation: activation_dev,
                    weights: weights_dev,
                    scales: scales_dev,
                    zero_points: None,
                    bias: None,
                    output: output_dev,
                    m,
                    k,
                    n,
                    group_size: group,
                    scales_fp16: true,
                    bias_post_round: false,
                    bias_row_stride: 0,
                };
                // Warmup.
                for _ in 0..5 {
                    launch_marlin_gemm(&rt, &args).unwrap();
                }
                rt.synchronize().unwrap();
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    launch_marlin_gemm(&rt, &args).unwrap();
                }
                rt.synchronize().unwrap();
                let secs = start.elapsed().as_secs_f64() / iters as f64;

                let weight_bytes = (n * k / 2) as f64;
                let act_bytes = (m * k * 2) as f64;
                let out_bytes = (m * n * 2) as f64;
                let scale_bytes = (n * k_blocks * 2) as f64;
                let total_bytes = weight_bytes + act_bytes + out_bytes + scale_bytes;
                let gbps = total_bytes / secs / 1e9;
                let weight_gbps = weight_bytes / secs / 1e9;
                let gflops = (2 * m * n * k) as f64 / secs / 1e9;
                eprintln!(
                    "marlin K={k} N={n} M={m}: {us:.1} us | total {gbps:.0} GB/s ({pct:.1}% peak) \
                     | weight {wgbps:.0} GB/s ({wpct:.1}% peak) | {gflops:.0} GFLOP/s",
                    us = secs * 1e6,
                    pct = gbps / peak_gbps * 100.0,
                    wgbps = weight_gbps,
                    wpct = weight_gbps / peak_gbps * 100.0,
                );

                // Split-K sweep: fill idle SMs at small M. Report the best factor.
                let sm = rt.capabilities().multiprocessor_count();
                let auto_sk = choose_split_k(m, n, k_blocks, sm);
                let mut sk_candidates = vec![2usize, 4, 8];
                if !sk_candidates.contains(&auto_sk) && auto_sk > 1 {
                    sk_candidates.push(auto_sk);
                }
                for &sk in &sk_candidates {
                    if sk < 2 || sk > k_blocks {
                        continue;
                    }
                    let partials_dev = rt.alloc_raw(splitk_partials_len(sk, m, n) * 4).unwrap();
                    for _ in 0..5 {
                        launch_marlin_gemm_splitk(&rt, &args, sk, partials_dev).unwrap();
                    }
                    rt.synchronize().unwrap();
                    let sstart = std::time::Instant::now();
                    for _ in 0..iters {
                        launch_marlin_gemm_splitk(&rt, &args, sk, partials_dev).unwrap();
                    }
                    rt.synchronize().unwrap();
                    let ssecs = sstart.elapsed().as_secs_f64() / iters as f64;
                    let auto = if sk == auto_sk { " (auto)" } else { "" };
                    eprintln!(
                        "  split-K={sk}{auto}: {us:.1} us | {spd:.2}x vs direct | weight {wg:.0} GB/s ({wp:.1}% peak)",
                        us = ssecs * 1e6,
                        spd = secs / ssecs,
                        wg = weight_bytes / ssecs / 1e9,
                        wp = weight_bytes / ssecs / 1e9 / peak_gbps * 100.0,
                    );
                    // SAFETY: allocated just above, freed once.
                    unsafe {
                        rt.free_raw(partials_dev).unwrap();
                    }
                }

                // SAFETY: freed once each.
                unsafe {
                    rt.free_raw(activation_dev).unwrap();
                    rt.free_raw(output_dev).unwrap();
                }
            }
            // SAFETY: freed once each.
            unsafe {
                rt.free_raw(weights_dev).unwrap();
                rt.free_raw(scales_dev).unwrap();
            }
        }
    }

    /// The device repack kernel must produce byte-identical output to the host
    /// [`repack_int4_weights`] (which the GPU parity test already validates), so
    /// the wired op path and the standalone path share one layout.
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn device_repack_matches_host() {
        let Some(rt) = runtime() else {
            eprintln!("skipping: CUDA runtime unavailable");
            return;
        };
        for &(n, group) in &[(70usize, 128usize), (37, 32), (8, 16), (200, 64)] {
            let k = group * 5;
            let k_blocks = k / group;
            let blob = group / 2;
            let mut packed = vec![0u8; n * k_blocks * blob];
            let mut state = 0xdead_beefu32;
            for b in packed.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (state >> 24) as u8;
            }
            let host = repack_int4_weights(&packed, n, k, group);

            let packed_dev = rt.alloc_raw(packed.len()).unwrap();
            let out_dev = rt.alloc_raw(host.len()).unwrap();
            // SAFETY: sized for the slices.
            unsafe {
                rt.htod(&packed, packed_dev).unwrap();
            }
            launch_marlin_repack(&rt, packed_dev, out_dev, n, k, group).unwrap();
            rt.synchronize().unwrap();
            let mut got = vec![0u8; host.len()];
            // SAFETY: out_dev holds host.len() bytes.
            unsafe {
                rt.dtoh(&mut got, out_dev).unwrap();
                rt.free_raw(packed_dev).unwrap();
                rt.free_raw(out_dev).unwrap();
            }
            assert_eq!(
                got, host,
                "device repack != host repack for n={n} group={group}"
            );
        }
    }
}
