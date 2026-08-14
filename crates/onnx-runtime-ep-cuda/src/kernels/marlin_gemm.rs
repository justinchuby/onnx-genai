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
//! `{2·tid, 2·tid+1, 2·tid+8, 2·tid+9}` of each 16-wide K slice. We repack so
//! each lane reads its four nibbles as two contiguous bytes:
//!
//! `repacked[(n * (K/16) + slice) * 8 + tid*2 + {0,1}]`
//!   byte+0 = code(k=2·tid) | code(k=2·tid+1) << 4
//!   byte+1 = code(k=2·tid+8) | code(k=2·tid+9) << 4
//!
//! where `slice = k/16` runs `0..K/16`. Total size is `N*K/2` bytes — identical
//! to the source `packed`, it is a *reordering*, not an expansion. `scales` and
//! `zero_points` keep their original `[N, k_blocks]` indexing.

use std::ffi::c_void;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Result};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

/// NVRTC module + entry names for the Marlin int4 tensor-core GEMM.
pub const MARLIN_MODULE: &str = "matmul_nbits_marlin_gemm";
pub const MARLIN_GEMM_ENTRY: &str = "matmul_nbits_marlin_gemm_f16";

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
    let mut out = vec![0u8; n * slices * 8];
    let code_at = |col: usize, kk: usize| -> u8 {
        let block = kk / group_size;
        let within = kk % group_size;
        let byte = packed[(col * k_blocks + block) * blob + within / 2];
        if within & 1 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        }
    };
    for col in 0..n {
        for slice in 0..slices {
            let kbase = slice * 16;
            let dst = (col * slices + slice) * 8;
            for tid in 0..4usize {
                let lo = code_at(col, kbase + tid * 2) | (code_at(col, kbase + tid * 2 + 1) << 4);
                let hi =
                    code_at(col, kbase + tid * 2 + 8) | (code_at(col, kbase + tid * 2 + 9) << 4);
                out[dst + tid * 2] = lo;
                out[dst + tid * 2 + 1] = hi;
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

            // B fragment (16x8), col-major. Weights centered (code - zp), unscaled.
            unsigned char blo = 0, bhi = 0;
            if (nb_col < n) {
                const long base = ((long)nb_col * slices + slice) * 8 + tid * 2;
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
"#;

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
    if args.k % 16 != 0 || args.k % args.group_size != 0 {
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
            assert_eq!(repacked.len(), n * (k / 16) * 8);

            let slices = k / 16;
            for col in 0..n {
                for slice in 0..slices {
                    let dst = (col * slices + slice) * 8;
                    for tid in 0..4usize {
                        let kbase = slice * 16;
                        let lo = repacked[dst + tid * 2];
                        let hi = repacked[dst + tid * 2 + 1];
                        assert_eq!(lo & 15, codes[col * k + kbase + tid * 2] & 15);
                        assert_eq!(lo >> 4, codes[col * k + kbase + tid * 2 + 1] & 15);
                        assert_eq!(hi & 15, codes[col * k + kbase + tid * 2 + 8] & 15);
                        assert_eq!(hi >> 4, codes[col * k + kbase + tid * 2 + 9] & 15);
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
}
