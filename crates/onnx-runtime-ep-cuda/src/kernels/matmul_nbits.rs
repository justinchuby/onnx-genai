//! `com.microsoft::MatMulNBits`: decode-specialized packed INT4/INT8 GEMV plus
//! the block-wise dequantization and f32 cuBLASLt GEMM fallback used for prefill.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut,
    TensorView, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{self, GemmDtype, GemmEpilogue, GemmEpilogueKind, GemmEx, GemmParams};
use crate::error::driver_err;
use crate::kernels::marlin_gemm;
use crate::runtime::{CudaRuntime, cuptr, raw_ptr};

const DEQUANT_MODULE: &str = "matmul_nbits_dequant_f32";
const DEQUANT_ENTRY: &str = "matmul_nbits_dequant_f32";
const GEMV_MODULE: &str = "matmul_nbits_gemv";
const GEMV_F32_ENTRY: &str = "matmul_nbits_gemv_f32";
const GEMV_INT8_F32_ENTRY: &str = "matmul_nbits_gemv_int8_f32";
/// Structurally-selected int4 / `block_size == 128` / asymmetric fp32-activation
/// decode GEMV. It preserves the generic f32 GEMV's per-thread accumulation and
/// reduction order while folding block arithmetic and vectorizing packed-nibble
/// loads.
const GEMV_INT4_F32_BLOCK128_ENTRY: &str = "matmul_nbits_gemv_int4_f32_block128";
/// Structurally-selected int8 / `block_size == 128` / asymmetric fp32-activation
/// decode GEMV. Bit-for-bit identical to the generic [`GEMV_F32_ENTRY`] path
/// (same per-thread depth stride, same per-element expression, same fp32
/// reduction) but with `block_size == 128` folded into a shift (`>> 7`) so the
/// per-weight division/modulo, the runtime bit-width branch, and the repeated
/// `column * k_blocks` scale/zero-point base recomputation all drop out.
const GEMV_INT8_F32_BLOCK128_ENTRY: &str = "matmul_nbits_gemv_int8_f32_block128";
const QUANTIZE_ACCURACY4_ENTRY: &str = "matmul_nbits_quantize_accuracy4_block32";
const GEMV_ACCURACY4_ENTRY: &str = "matmul_nbits_gemv_accuracy4_block32";
const GEMV_ACCURACY4_STAGE64_ENTRY: &str = "matmul_nbits_gemv_accuracy4_block32_stage64";
// General-block-size fp32-activation accuracy_level=4 decode entries. These give
// the fp32 `run` path the same "quantize the activation to int8 once, then run a
// parallelized GEMV" treatment the block-32 path already enjoys, but for any
// power-of-two block_size and for the asymmetric (zero-point) int4 layout. They
// are bit-for-bit identical to the grid-starved `matmul_nbits_accuracy4` tiled
// GEMM (same per-K-block int8 activation quantization, same per-block integer
// dot, same sequential fp32 block accumulation) — only the parallelization
// differs, so decode tokens are unchanged.
const QUANTIZE_ACCURACY4_BLOCKWISE_ENTRY: &str = "matmul_nbits_quantize_accuracy4_blockwise";
const GEMV_ACCURACY4_BLOCKWISE_ENTRY: &str = "matmul_nbits_gemv_accuracy4_blockwise";
const ACCURACY4_MODULE: &str = "matmul_nbits_accuracy4";
const ACCURACY4_ENTRY: &str = "matmul_nbits_accuracy4";
const BLOCK_THREADS: u32 = 256;
const GEMV_ACCURACY4_THREADS: u32 = 256;
const GEMV_ACCURACY4_COLUMNS_PER_BLOCK: usize = 8;
const GEMV_ACCURACY4_SHARED_BYTES: u32 = 32 * 32;
const GEMV_ACCURACY4_STAGE64_SHARED_BYTES: u32 = 64 * 32;
const GEMV_F16_MODULE: &str = "matmul_nbits_gemv_f16";
const GEMV_F16_ENTRY: &str = "matmul_nbits_gemv_f16";
const GEMV_INT8_F16_ENTRY: &str = "matmul_nbits_gemv_int8_f16";
/// Split-K specialization of [`GEMV_INT8_F16_ENTRY`]. ncu showed the single-warp
/// standalone int8 GEMV grid-starved on Phi's int8 down projection (grid 384,
/// ~0.48 waves/SM, ~35% occupancy, 28µs) — unlike the fused RMSNorm-prologue int8
/// kernel, it has no serial prologue, so partitioning the K reduction across
/// [`GEMV_INT8_F16_SPLITK`] cooperating warps per output column multiplies the
/// grid to fill the SMs and directly attacks the memory-latency bound. The K-slice
/// partials are summed in fp32 (a new block-sum association vs the single-warp
/// kernel), so this path is near-equal — not byte-identical — to the plain entry;
/// asymmetric-zp parity is validated against a dequant reference to tolerance. Only
/// launched when `K % 256 == 0` (whole 256-wide steps, no divergent tail) and the
/// weights carry zero points; symmetric int8 keeps the byte-identical single-warp
/// kernel.
const GEMV_INT8_F16_SPLITK_ENTRY: &str = "matmul_nbits_gemv_int8_f16_splitk";
/// Warps cooperating per output column in the split-K standalone int8 GEMV. Must
/// match `K_SPLIT` in `matmul_nbits_gemv_int8_f16_splitk`. A block keeps its
/// `blockDim.x / 32` warps but now covers `warps / K_SPLIT` columns, so the launch
/// grid grows by this factor.
const GEMV_INT8_F16_SPLITK: usize = 2;
const GEMM_F16_ENTRY: &str = "matmul_nbits_gemm_f16";
/// Model-agnostic fp16 int4 decode GEMV for any power-of-two `block_size` (16,
/// 64, 128, 256, ...). The tuned [`GEMV_F16_ENTRY`]/[`GEMV_F16_SCALES_F16_ENTRY`]
/// kernels bake in the block-32 four-lane/eight-block warp layout; this entry
/// instead derives the scale/zero-point block index from the actual
/// `block_size` (`block = depth / block_size`) so a lane's contiguous 8-element
/// chunk maps to the correct block regardless of block width.
const GEMV_F16_GENERAL_BS_ENTRY: &str = "matmul_nbits_gemv_f16_general_bs";
/// Split-K counterpart of [`GEMV_F16_GENERAL_BS_ENTRY`]: `K_SPLIT` warps
/// cooperate on one output column and reduce their fp32 partials through shared
/// memory, so the launch grid is `K_SPLIT`x larger and fills the SMs on the
/// grid-starved block!=32 decode projections (medium/KV GEMVs run at ~0.5
/// waves single-warp). Near-equal (not byte-identical) to the single-warp
/// entry — the split reorders the partial-sum association, the same trade the
/// block-32 split-K entries already ship by default.
const GEMV_F16_GENERAL_BS_SPLITK_ENTRY: &str = "matmul_nbits_gemv_f16_general_bs_splitk";
/// Wide-load (128-bit `uint4` weight load, software-pipelined) counterpart of
/// [`GEMV_F16_GENERAL_BS_ENTRY`]. Each lane owns 32 contiguous nibbles per step
/// and streams them with one pipelined `uint4` load (4x fewer load instructions,
/// 2+ loads in flight) to raise memory-level parallelism / DRAM bandwidth on
/// the M=1 int4 decode GEMV toward ORT's (head-to-head: ORT 2.42 TB/s vs our
/// 0.92 TB/s at the same grid/occupancy — a pure narrow-load-issue gap). int4
/// only; requires `block_size % 32 == 0 && k % 32 == 0`. Near-equal (32-wide
/// lane interleave regroups the fp32 partials), gated greedy-token-identical.
const GEMV_F16_GENERAL_BS_WIDE_ENTRY: &str = "matmul_nbits_gemv_f16_general_bs_wide";
/// Wide-load counterpart of [`GEMV_F16_GENERAL_BS_SPLITK_ENTRY`] (see
/// [`GEMV_F16_GENERAL_BS_WIDE_ENTRY`]): `K_SPLIT` warps each walk a
/// `32 * K_SPLIT`-strided set of 32-nibble chunks with pipelined `uint4` loads.
const GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY: &str = "matmul_nbits_gemv_f16_general_bs_splitk_wide";
/// Column register-blocked wide-load GEMV
/// (`matmul_nbits_gemv_f16_general_bs_wide_multicol`): each warp emits
/// [`GEMV_F16_WIDE_MULTICOL_NC`] output columns, decoding each activation
/// sub-word once and reusing it across the columns. Attacks the L1/TEX-throughput
/// limiter of [`GEMV_F16_GENERAL_BS_WIDE_ENTRY`] (redundant per-column activation
/// re-reads) while staying byte-identical (per-column fp32 accumulation order
/// unchanged). Non-split-K only (used on the wide, occupancy-filled gate_up).
const GEMV_F16_GENERAL_BS_WIDE_MULTICOL_ENTRY: &str =
    "matmul_nbits_gemv_f16_general_bs_wide_multicol";
/// fp16 mixed-precision counterpart of
/// [`GEMV_F16_GENERAL_BS_WIDE_MULTICOL_ENTRY`]
/// (`matmul_nbits_gemv_f16_general_bs_wide_multicol_fp16`): identical
/// column-register-blocking, but the per-chunk MAC runs in fp16 `__hfma2` (two
/// fused MACs/instruction) to cut the dequant/MAC ALU that limits the fp32
/// multicol kernel — the fp16-vs-fp16 equal-conditions path against ORT's fp16
/// `MatMulFloatInt4Kernel`. The entire per-lane K reduction runs in fp16
/// `__half2` accumulators (each 32-product chunk summed in fp16, its per-block
/// scale folded in with `__hfma2`), exactly mirroring ORT's
/// `MatMulFloat4BitsKernelM1`; fp32 is used ONLY in the final cross-lane
/// warp-shuffle reduction. This is safe (no token-flipping mantissa loss)
/// because each lane strides K by 32 and folds only a handful of chunks, so the
/// fp16 accumulation is a wide, shallow tree of depth ~tens, not K. Because the
/// arithmetic mirrors ORT's it lands in ORT's own error class — NOT byte-identical
/// to the fp32 path, so it is opt-in via [`use_gemv_fp16`] and gated on accuracy
/// (error <= ORT vs an f64 oracle), not bit-identity.
const GEMV_F16_GENERAL_BS_WIDE_MULTICOL_FP16_ENTRY: &str =
    "matmul_nbits_gemv_f16_general_bs_wide_multicol_fp16";
/// Interleaved + biased (symmetric-only, OPT-IN) counterpart of
/// [`GEMV_F16_GENERAL_BS_WIDE_MULTICOL_ENTRY`]
/// (`matmul_nbits_gemv_f16_general_bs_wide_multicol_interleaved`). Same
/// column-register-blocking wide-load geometry, but consumes offline
/// nibble-interleaved weights and folds the symmetric `-8` bias into the LOP3
/// magic-removal constants (TRT-LLM `FastInterleavedAndBiasedNumericArrayConverter`
/// lever). This drops the per-block zero-point `sub.f16x2` and the `prmt.b32`
/// activation reorder from the decode inner loop (~14 -> ~9 dequant instrs / 8
/// values), attacking the instruction/issue-slot limiter of the M=1 decode GEMV.
/// Byte-identical output to the fp32 multicol kernel on symmetric weights.
/// Enabled only when [`interleave_dequant_enabled`] is set; the dispatch swaps
/// the packed pointer for the interleaved cache buffer at the same time.
const GEMV_F16_GENERAL_BS_WIDE_MULTICOL_INTERLEAVED_ENTRY: &str =
    "matmul_nbits_gemv_f16_general_bs_wide_multicol_interleaved";
/// Interleaved + biased (symmetric-only, OPT-IN) counterpart of
/// [`GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY`]
/// (`matmul_nbits_gemv_f16_general_bs_splitk_wide_interleaved`). Same split-K
/// wide-load geometry (K_SPLIT warps/column, shared-memory fp32 partial
/// reduction), but consumes offline nibble-interleaved weights and folds the
/// symmetric `-8` bias into the LOP3 converter. Captures the grid-starved
/// narrow-N projections (glm qkv/down) that take the split-K path, compounding
/// the multicol lever. Byte-identical to the non-interleaved split-K wide kernel
/// on symmetric weights. Enabled only when [`interleave_dequant_enabled`] is set.
const GEMV_F16_GENERAL_BS_SPLITK_WIDE_INTERLEAVED_ENTRY: &str =
    "matmul_nbits_gemv_f16_general_bs_splitk_wide_interleaved";
/// Multicol x split-K hybrid of [`GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY`]
/// (`matmul_nbits_gemv_f16_general_bs_splitk_wide_multicol`): keeps the split-K
/// grid-fill (K_SPLIT warps/column group, shared-memory fp32 partial reduction)
/// but register-blocks [`GEMV_F16_WIDE_MULTICOL_NC`] output columns per warp, so
/// each warp issues WIDE_NC independent 128-bit weight loads per chunk. This
/// ports the gate_up `wide_multicol` kernel's memory-level parallelism (which
/// runs at ~37% DRAM peak) to the grid-starved medium-N split-K projections
/// (down_proj / qkv / attn-out) that the single-column split-K wide kernel left
/// latency-bound at ~16% DRAM peak. A 256-thread CTA covers
/// `(8 / K_SPLIT) * GEMV_F16_WIDE_MULTICOL_NC` output columns. Byte-identical to
/// [`GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY`] (same per-lane depth0/stride, same
/// per-column accumulation order, same K_SPLIT reduction order).
const GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_ENTRY: &str =
    "matmul_nbits_gemv_f16_general_bs_splitk_wide_multicol";
/// Interleaved + biased (symmetric-only, OPT-IN) sibling of
/// [`GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_ENTRY`]
/// (`matmul_nbits_gemv_f16_general_bs_splitk_wide_multicol_interleaved`). Same
/// multicol x split-K geometry, consumes offline nibble-interleaved weights and
/// folds the symmetric -8 bias into the LOP3 converter. Byte-identical to the
/// non-interleaved multicol split-K kernel on symmetric weights; enabled only
/// when [`interleave_dequant_enabled`] is set.
const GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_INTERLEAVED_ENTRY: &str =
    "matmul_nbits_gemv_f16_general_bs_splitk_wide_multicol_interleaved";
/// Offline int4 nibble-interleave pass entry
/// (`matmul_nbits_interleave_int4`); runs once per weight into the cache buffer.
const INTERLEAVE_INT4_ENTRY: &str = "matmul_nbits_interleave_int4";
/// Output columns each warp emits in the register-blocked wide GEMV. Must match
/// `#define WIDE_NC` in the CUDA source. A 256-thread CTA (8 warps) therefore
/// covers `8 * GEMV_F16_WIDE_MULTICOL_NC` columns.
const GEMV_F16_WIDE_MULTICOL_NC: usize = 4;
/// Warps cooperating per output column in the block!=32 general_bs split-K GEMV.
/// Must match `constexpr int K_SPLIT` in `matmul_nbits_gemv_f16_general_bs_splitk`.
/// A block keeps its `blockDim.x / 32` warps but now covers `warps / K_SPLIT`
/// columns, so the launch grid grows by this factor. Tuned to 4 on glm-4-9b
/// (block-128, N=4096 medium projections): K_SPLIT=2 lifts the grid-starved
/// single-warp launch from ~0.5 to ~1 wave, K_SPLIT=4 to ~2 waves (the
/// latency-hiding target), which measured fastest (+9.7% decode vs single-warp);
/// K_SPLIT=8 saturated.
const GENERAL_BS_SPLITK: usize = 4;
/// Warps cooperating per output column GROUP in the multicol x split-K hybrid
/// GEMV (`matmul_nbits_gemv_f16_general_bs_splitk_wide_multicol`). Kept at 4
/// (independent of [`GENERAL_BS_SPLITK`] so it can be tuned separately) because
/// the hybrid register-blocks [`GEMV_F16_WIDE_MULTICOL_NC`] columns per warp
/// (WIDE_NC independent weight-load streams supply the memory-level parallelism),
/// so K_SPLIT only has to refill the grid. With WIDE_NC=4 a 256-thread CTA covers
/// `(8 / K_SPLIT) * WIDE_NC` columns; K_SPLIT=4 lands N=4096 at ~1 wave and
/// measured fastest — K_SPLIT=8 (~2 waves) regressed decode (the 8-way partial
/// reduction + halved per-warp work outweighed the extra grid-fill). Must match
/// `constexpr int K_SPLIT` in the `*_splitk_wide_multicol*` CUDA kernels.
const GENERAL_BS_SPLITK_MULTICOL: usize = 4;
/// Model-agnostic fp16 int4/int8 prefill GEMM for any power-of-two `block_size`.
/// Mirrors [`GEMM_F16_ENTRY`] but walks `K` in fixed 32-wide tiles and computes
/// `block = depth / block_size`, decoupling the tile width from the block width.
const GEMM_F16_GENERAL_BS_ENTRY: &str = "matmul_nbits_gemm_f16_general_bs";
const GEMV_F16_SCALES_F16_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16";
/// Asymmetric-zero-point specialization of [`GEMV_F16_SCALES_F16_ENTRY`]. The
/// symmetric entry above is compiled with `HasZp == false`, which
/// dead-code-eliminates the per-block zero-point global load and folds the
/// subtrahend to the constant fp16 `8.0` (byte-identical to the pre-zero-point
/// path). Weights that actually carry zero points launch this `_zp` entry so
/// only the asymmetric path pays for the extra per-block load.
const GEMV_F16_SCALES_F16_ZP_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_zp";
/// Prefetch-pipelined siblings of [`GEMV_F16_SCALES_F16_ENTRY`] /
/// [`GEMV_F16_SCALES_F16_ZP_ENTRY`]. Same lane->nibble mapping, fp16
/// accumulation, and reduction order (BYTE-IDENTICAL output), but a depth-4
/// register shift register keeps 4 weight loads in flight per lane to hide the
/// Long-Scoreboard global-load latency that dominates the single-warp kernel
/// (ncu: ~75% of warp cycles stalled on the one in-flight 32-bit load). Selected
/// by [`use_scales_f16_pipeline`] (default-on for the single-warp block-32 int4
/// path; `ONNX_GENAI_GEMV_PIPELINE=0` forces the original entry for A/B).
const GEMV_F16_SCALES_F16_PIPE_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_pipe";
const GEMV_F16_SCALES_F16_ZP_PIPE_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_zp_pipe";
const GEMV_F16_SCALES_F16_SPLITK_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_splitk";
/// Split-K specialization for standalone fp16/scales-fp16 int4 GEMV. ncu showed
/// the plain single-warp kernel can be grid-starved
/// (~0.36 waves/SM, ~64% of the SMs idle) and memory-latency bound: partitioning
/// the K reduction across [`GEMV_F16_SCALES_F16_ZP_SPLITK`] cooperating warps per
/// output column multiplies the grid, filling the machine so the extra in-flight
/// loads hide the Long-Scoreboard latency. The K-slice partials are summed in
/// fp32 (a new block-sum association vs the single-warp kernel), so this path is
/// near-equal — not byte-identical — to the plain `_zp` kernel; the asymmetric-zp
/// parity tests track a dequant reference to tolerance.
const GEMV_F16_SCALES_F16_ZP_SPLITK_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_zp_splitk";
/// Warps cooperating per output column in the split-K asymmetric int4 GEMV. Must
/// match `K_SPLIT` in `matmul_nbits_gemv_f16_scales_f16_splitk`. A block keeps
/// its `blockDim.x / 32` warps but now covers `warps / K_SPLIT` columns, so the
/// launch grid grows by this factor.
const GEMV_F16_SCALES_F16_ZP_SPLITK: usize = 2;
/// General fp16/fp16-scales GEMV with a fused RMS-normalization prologue (see
/// [`crate::optimizer::CudaSkipRmsNormMatMulFusion`]). It normalizes the input
/// activation in-kernel — byte-identically to `skip_rmsnorm_f16_warp_half4` —
/// before the standard `scales_f16` int4 dot, folding a
/// `SkipSimplifiedLayerNormalization` normalization into the following GEMV.
const GEMV_F16_SCALES_F16_RMSNORM_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_rmsnorm";
const GEMV_F16_SCALES_F16_RMSNORM_SPLITK_ENTRY: &str =
    "matmul_nbits_gemv_f16_scales_f16_rmsnorm_splitk";
/// Asymmetric-zero-point specialization of [`GEMV_F16_SCALES_F16_RMSNORM_ENTRY`]
/// (see [`GEMV_F16_SCALES_F16_ZP_ENTRY`] for the `HasZp` specialization scheme).
const GEMV_F16_SCALES_F16_RMSNORM_ZP_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_rmsnorm_zp";
/// INT8 sibling of [`GEMV_F16_SCALES_F16_RMSNORM_ENTRY`]. Shares the RMS
/// reduction and normalized-activation staging bit-for-bit and swaps in the
/// block-32 int8 dequant dot, fusing a `SkipSimplifiedLayerNormalization` into
/// the following int8 GEMV (e.g. Phi's int8 qkv projection). Compiled in the
/// same symmetric/`_zp` `HasZp` pair as the int4 sibling so a future
/// symmetric-int8 model keeps the constant-subtrahend (no per-block load) path.
const GEMV_INT8_F16_SCALES_F16_RMSNORM_ENTRY: &str =
    "matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm";
/// Asymmetric-zero-point specialization of
/// [`GEMV_INT8_F16_SCALES_F16_RMSNORM_ENTRY`] (Phi int8 qkv carries zero points).
const GEMV_INT8_F16_SCALES_F16_RMSNORM_ZP_ENTRY: &str =
    "matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_zp";
/// Standalone RMS-normalization prologue used by the fused GEMV's M>1 prefill
/// path (see [`MatMulNBitsKernel::launch_rmsnorm_prefill`]).
const RMSNORM_PREFILL_ENTRY: &str = "matmul_nbits_rmsnorm_f16_warp_half4";
/// One warp (32 lanes) normalizes one token row in the prefill prologue.
const RMSNORM_PREFILL_THREADS: u32 = 32;
const GEMV_F16_DOWN_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_down";
/// Grid-fill specializations of [`GEMV_F16_DOWN_ENTRY`]: identical numerics with
/// 4 / 2 columns per CTA instead of 8, so the launch grid grows 2x / 4x to fill
/// the multiprocessors on grid-starved (small-N) tall-skinny down projections.
/// Every output column is still reduced entirely within one CTA in the same
/// order, so the fp32 accumulation is bit-identical to the 8-column entry — only
/// the CTA count changes. Selected by [`select_down_columns`] from the device
/// multiprocessor count.
const GEMV_F16_DOWN_C4_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_down_c4";
const GEMV_F16_DOWN_C2_ENTRY: &str = "matmul_nbits_gemv_f16_scales_f16_down_c2";
const GEMM_F16_TILE: usize = 16;

/// Upper bound on batch `M` for which small-batch decode reuses the capture-safe
/// single-row decode GEMV once per row instead of the tiled prefill GEMM.
///
/// Why a bound exists: the M>1 tiled GEMM tiles the M dimension in
/// [`GEMM_F16_TILE`]-row (16) blocks with a grid of `ceil(M/16)` tile-rows, so
/// for any `M in [1, 16]` it launches the *same* `(N/16) x 1` CTA grid and pays
/// a full weight-grid dequant+MAC pass whose cost is independent of `M`. A
/// single-row decode GEMV instead reads the weights once for one row; looping it
/// `M` times reads the weights `M` times but skips the tiled GEMM's fixed
/// full-grid overhead, so it is faster while `M x gemv_step < tiled_step`.
///
/// Measured crossover on **RTX 4060 Laptop 8 GB, i7-13800H (14C/20T), CUDA 13.1**,
/// `qwen05b-q4` (native CUDA, fp16 activation, int4): the M==1 decode GEMV runs a
/// full forward step in ~2.55 ms; the tiled GEMM step is ~28.6 ms and flat for
/// M=2..16. So `M x 2.55 ms < 28.6 ms` holds up to `M ~ 11`. The default is set
/// conservatively below that measured crossover; the ratio is a kernel-efficiency
/// property (both paths are weight-bandwidth bound on the same weights) so it is
/// approximately model-independent, but it IS hardware-dependent — retune per
/// #1261 via `ONNX_GENAI_DECODE_GEMV_LOOP_MAX_M` rather than trusting this number
/// on a different GPU.
const DECODE_GEMV_LOOP_MAX_M_DEFAULT: usize = 8;

/// Resolve the looped-decode-GEMV batch bound, honoring the
/// `ONNX_GENAI_DECODE_GEMV_LOOP_MAX_M` override (read once per process). Setting
/// it to `1` disables the loop (all `M>1` go to the tiled GEMM), which is the
/// byte-identical A/B baseline for measuring the change.
fn decode_gemv_loop_max_m() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ONNX_GENAI_DECODE_GEMV_LOOP_MAX_M")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value >= 1)
            .unwrap_or(DECODE_GEMV_LOOP_MAX_M_DEFAULT)
    })
}

const GEMV_F16_SMALL_THREADS: u32 = 64;
const GEMV_F16_LARGE_THREADS: u32 = 256;
const GEMV_F16_SMALL_N_MAX: usize = 1152;
/// Block-quantization size the down-projection tiling assumes. It stages the
/// activation as `K/8` permuted half8 vectors and indexes them as 4 `uint4` per
/// K-block (`block*4 .. block*4+3`), i.e. exactly 32 activation elements per
/// block, and it has **no** partial-block tail. So the variant is only correct
/// when `block_size == 32` and `K` is a whole multiple of 32.
const GEMV_F16_DOWN_BLOCK_SIZE: usize = 32;
const GEMV_F16_DOWN_THREADS: u32 = 256;
const GEMV_F16_DOWN_COLUMNS_PER_BLOCK: usize = 8;
const GATE_UP_SWIGLU_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_swiglu";
const GATE_UP_DECOMPOSED_SWIGLU_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_decomposed_swiglu";
const GATE_UP_SWIGLU_RMSNORM_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm";
const GATE_UP_DECOMPOSED_SWIGLU_RMSNORM_ENTRY: &str =
    "matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_rmsnorm";
/// Asymmetric-zero-point specializations of the paired gate/up SwiGLU entries
/// (see [`GEMV_F16_SCALES_F16_ZP_ENTRY`] for the `HasZp` specialization scheme).
const GATE_UP_SWIGLU_ZP_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_swiglu_zp";
const GATE_UP_DECOMPOSED_SWIGLU_ZP_ENTRY: &str =
    "matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_zp";
const GATE_UP_SWIGLU_RMSNORM_ZP_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_zp";
const GATE_UP_DECOMPOSED_SWIGLU_RMSNORM_ZP_ENTRY: &str =
    "matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_rmsnorm_zp";
/// Fused-symmetric (`ONNX_GENAI_GATEUP_VEC`) specializations of the four
/// SYMMETRIC paired gate/up SwiGLU entries. Byte-identical to the entries above
/// — the dequant folds the `- 8` symmetric zero point into the magic-bias
/// constants (see `int4x8_to_half2x4_sym8`), issuing four fewer `f16x2` ops per
/// weight word to relieve the issue-bound decode GEMV. Selected by
/// [`gate_up_vec_enabled`] (default-ON, byte-identical magic-bias-fold;
/// `ONNX_GENAI_GATEUP_VEC=0` forces scalar for A/B). The asymmetric `_zp` entries
/// have no `_vec` sibling: their per-block zero point cannot fold to a constant.
const GATE_UP_SWIGLU_VEC_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_swiglu_vec";
const GATE_UP_DECOMPOSED_SWIGLU_VEC_ENTRY: &str =
    "matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_vec";
const GATE_UP_SWIGLU_RMSNORM_VEC_ENTRY: &str = "matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_vec";
const GATE_UP_DECOMPOSED_SWIGLU_RMSNORM_VEC_ENTRY: &str =
    "matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_rmsnorm_vec";
/// Occupancy-raised (`ONNX_GENAI_GATEUP_OCC`) siblings of the two SYMMETRIC
/// RMS-norm-fused `_vec` entries. Same kernel body + `__launch_bounds__(256, 8)`
/// (32 regs → 8 blocks/SM, 100% theoretical vs 75% register-limited) to hide the
/// Short-Scoreboard shared-load latency of the staged activation. Byte-identical
/// to the `_vec` entries (register-allocation hint only).
const GATE_UP_SWIGLU_RMSNORM_VEC_OCC_ENTRY: &str =
    "matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_vec_occ";
const GATE_UP_DECOMPOSED_SWIGLU_RMSNORM_VEC_OCC_ENTRY: &str =
    "matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_rmsnorm_vec_occ";
const GATE_UP_SWIGLU_THREADS: u32 = 256;

const DEQUANT_SRC: &str = r#"
extern "C" __global__ void matmul_nbits_dequant_f32(
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const int* group_indices,
    float* weight_kn,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bits)
{
    const long total = (long)k * n;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        const int depth = (int)(idx / n);
        const int output = (int)(idx % n);
        const int block = depth / block_size;
        const int within = depth - block * block_size;
        const int bit_offset = within * bits;
        const unsigned char byte =
            packed[((long)output * k_blocks + block) * blob_size + bit_offset / 8];
        const int mask = bits == 8 ? 255 : ((1 << bits) - 1);
        const int quantized = (byte >> (bit_offset & 7)) & mask;
        const int group = group_indices ? group_indices[depth] : block;
        if (group < 0 || group >= k_blocks) {
            weight_kn[idx] = 0.0f;
            continue;
        }
        int zero_point = 1 << (bits - 1);
        if (zero_points) {
            const int zp_bit_offset = group * bits;
            const unsigned char zp =
                zero_points[(long)output * zp_row_bytes + zp_bit_offset / 8];
            zero_point = (zp >> (zp_bit_offset & 7)) & mask;
        }
        weight_kn[idx] =
            ((float)quantized - (float)zero_point) * scales[(long)output * k_blocks + group];
    }
}
"#;

const DEQUANT_F16_MODULE: &str = "matmul_nbits_dequant_f16";
const DEQUANT_F16_ENTRY: &str = "matmul_nbits_dequant_f16";

/// Half-precision dequantization feeding a cuBLASLt tensor-core GEMM.
///
/// The f32 dequant feeds a `CUDA_R_32F` cuBLASLt GEMM, which on A100 runs at the
/// 19.5 TFLOP/s FP32 rate. Materializing the weights as `__half` lets the same
/// GEMM run on tensor cores with f32 accumulation instead, and halves the
/// scratch buffer.
///
/// It writes `[N, K]` rather than the f32 path's `[K, N]`, and the GEMM
/// transposes it back. `[N, K]` is the order the weights are already packed in,
/// so a thread reads one aligned 32-bit word of eight nibbles and writes the
/// eight halves they expand to as one aligned 16-byte store, with the group's
/// scale and zero point loaded once for the whole word. The `[K, N]` order would
/// instead stride every packed read by a full quantized row.
///
/// Restricted to 4-bit weights whose block size is a multiple of 8 and a power
/// of two, so a word never straddles two groups and the group index is a shift.
/// Everything else keeps the general path.
const DEQUANT_F16_SRC: &str = r#"
#include <cuda_fp16.h>

// grid.x covers K/8 eight-weight words, grid.y one output column each.
extern "C" __global__ void matmul_nbits_dequant_f16(
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    __half* __restrict__ weight_nk,
    const int k,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int block_shift,
    const int scales_fp16)
{
    const int words = k >> 3;
    const int w = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (w >= words) return;
    const int out = (int)blockIdx.y;

    const int depth0 = w << 3;
    const int block = depth0 >> block_shift;

    const long row_bytes = (long)k_blocks * blob_size;
    const unsigned codes =
        *reinterpret_cast<const unsigned*>(packed + (long)out * row_bytes + (long)w * 4);

    const long scale_idx = (long)out * k_blocks + block;
    const float scale = scales_fp16
        ? __half2float(reinterpret_cast<const __half*>(scales_raw)[scale_idx])
        : reinterpret_cast<const float*>(scales_raw)[scale_idx];

    float zero_point = 8.0f;
    if (zero_points) {
        const unsigned char byte = zero_points[(long)out * zp_row_bytes + (block >> 1)];
        zero_point = (float)((block & 1) ? (byte >> 4) : (byte & 15));
    }

    __half2 out2[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = (float)((codes >> (8 * i)) & 15) - zero_point;
        const float hi = (float)((codes >> (8 * i + 4)) & 15) - zero_point;
        out2[i] = __floats2half2_rn(lo * scale, hi * scale);
    }
    *reinterpret_cast<float4*>(weight_nk + (long)out * k + depth0) =
        *reinterpret_cast<const float4*>(out2);
}
"#;

const GEMV_SRC: &str = r#"
__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value)
{
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < ((blockDim.x + 31) >> 5) ? warp_sums[lane] : 0.0f;
    return warp == 0 ? warp_sum(value) : 0.0f;
}

// Model-agnostic fp32-activation int4/int8 decode GEMV supporting any
// power-of-two block_size. The int4 path is bit-for-bit identical to the
// original (nibble unpack, symmetric default 8, per-block-nibble zero points);
// the int8 branch reads one byte per weight, uses a symmetric default of 128,
// and reads one whole-byte zero point per block. The tuned block-32 int8 entry
// (`matmul_nbits_gemv_int8_f32`) bakes in the block-32 geometry; block sizes
// other than 32 route here so the block index derives from the real block_size.
extern "C" __global__ void matmul_nbits_gemv_f32(
    const float* activation,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bits)
{
    const int column = (int)blockIdx.x;
    if (column >= n) {
        return;
    }

    float value = 0.0f;
    for (int depth = (int)threadIdx.x; depth < k; depth += (int)blockDim.x) {
        const int block = depth / block_size;
        const int within = depth - block * block_size;
        const long blob_base = ((long)column * k_blocks + block) * blob_size;
        int quantized;
        int zero_point;
        if (bits == 8) {
            quantized = (int)packed[blob_base + within];
            zero_point =
                zero_points ? (int)zero_points[(long)column * k_blocks + block] : 128;
        } else {
            const unsigned char byte = packed[blob_base + within / 2];
            quantized = (within & 1) ? (byte >> 4) : (byte & 15);
            zero_point = 8;
            if (zero_points) {
                const unsigned char zp =
                    zero_points[(long)column * zp_row_bytes + block / 2];
                zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
            }
        }
        value += activation[depth] * ((float)quantized - (float)zero_point)
            * scales[(long)column * k_blocks + block];
    }

    value = block_sum(value);
    if (threadIdx.x == 0) {
        output[column] = value + (bias ? bias[column] : 0.0f);
    }
}

extern "C" __global__ void matmul_nbits_gemv_int8_f32(
    const float* activation,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int k_blocks)
{
    const int column = (int)blockIdx.x;
    if (column >= n) {
        return;
    }

    float value = 0.0f;
    for (int depth = (int)threadIdx.x; depth < k; depth += (int)blockDim.x) {
        const int block = depth >> 5;
        const int within = depth & 31;
        const int quantized =
            (int)packed[((long)column * k_blocks + block) * 32 + within];
        const int zero_point =
            zero_points ? (int)zero_points[(long)column * k_blocks + block] : 128;
        value += activation[depth] * ((float)quantized - (float)zero_point)
            * scales[(long)column * k_blocks + block];
    }

    value = block_sum(value);
    if (threadIdx.x == 0) {
        output[column] = value + (bias ? bias[column] : 0.0f);
    }
}

// Structurally-selected int8 / block_size == 128 / asymmetric fp32-activation
// decode GEMV. This is a specialization of `matmul_nbits_gemv_f32` for the
// generic block-128 int8 loop that dominates Qwen3-0.6B decode. It is
// BIT-FOR-BIT IDENTICAL to that kernel: each thread walks the same depth stride
// (grid-stride by blockDim), evaluates the same per-element expression
// `activation * ((float)quantized - (float)zero_point) * scale`, and reduces in
// the same block_sum order — only the address arithmetic is cheaper.
//
// With block_size == 128 the block index is a shift (`depth >> 7`) instead of an
// integer divide, and the packed-byte address collapses: for blob_size == 128,
//   (column*k_blocks + block)*128 + (depth - block*128) == column*k_blocks*128 + depth,
// so `within` (the modulo) disappears entirely. The `column * k_blocks` base is
// hoisted once into `col_kb` for the scale and zero-point rows, and the runtime
// `bits == 8` branch of the generic kernel is gone. The `zero_points ? : 128`
// selection is retained (uniform across the CTA, effectively free) so the result
// stays identical whether or not zero points are present; dispatch restricts
// this entry to the asymmetric case in practice.
extern "C" __global__ void matmul_nbits_gemv_int8_f32_block128(
    const float* activation,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int k_blocks)
{
    const int column = (int)blockIdx.x;
    if (column >= n) {
        return;
    }

    const long col_kb = (long)column * k_blocks;
    const long packed_row = col_kb << 7;  // k_blocks * 128 bytes per weight row
    const float* col_scales = scales + col_kb;
    const unsigned char* col_zp = zero_points ? zero_points + col_kb : (const unsigned char*)0;

    float value = 0.0f;
    for (int depth = (int)threadIdx.x; depth < k; depth += (int)blockDim.x) {
        const int block = depth >> 7;
        const int quantized = (int)packed[packed_row + depth];
        const int zero_point = col_zp ? (int)col_zp[block] : 128;
        value += activation[depth] * ((float)quantized - (float)zero_point)
            * col_scales[block];
    }

    value = block_sum(value);
    if (threadIdx.x == 0) {
        output[column] = value + (bias ? bias[column] : 0.0f);
    }
}

// Int4 counterpart of the block-128 int8 specialization above. Each warp owns
// 32 consecutive depths, hence 16 aligned packed bytes. Four lanes load one
// aligned 32-bit word each and shuffle it to the eight lanes consuming its
// nibbles. This removes duplicate scalar byte loads without changing any
// thread's depth sequence or fp32 arithmetic.
extern "C" __global__ void matmul_nbits_gemv_int4_f32_block128(
    const float* activation,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int k_blocks)
{
    const int column = (int)blockIdx.x;
    if (column >= n) {
        return;
    }

    const int lane = (int)threadIdx.x & 31;
    const long col_kb = (long)column * k_blocks;
    const long packed_row = col_kb << 6;  // k_blocks * 64 bytes per weight row
    const float* col_scales = scales + col_kb;
    const unsigned char* col_zp =
        zero_points + (long)column * ((k_blocks + 1) >> 1);

    float value = 0.0f;
    for (int depth = (int)threadIdx.x; depth < k; depth += (int)blockDim.x) {
        const int block = depth >> 7;
        const int warp_depth = depth - lane;
        const unsigned int* packed_words =
            (const unsigned int*)(packed + packed_row + (warp_depth >> 1));
        unsigned int packed_word = lane < 4 ? packed_words[lane] : 0;
        packed_word = __shfl_sync(__activemask(), packed_word, lane >> 3);
        const int quantized = (int)((packed_word >> ((lane & 7) << 2)) & 15);
        const unsigned char zp_byte = col_zp[block >> 1];
        const int zero_point = (block & 1) ? (zp_byte >> 4) : (zp_byte & 15);
        value += activation[depth] * ((float)quantized - (float)zero_point)
            * col_scales[block];
    }

    value = block_sum(value);
    if (threadIdx.x == 0) {
        output[column] = value + (bias ? bias[column] : 0.0f);
    }
}

// Per-K-block (block-32) int8 activation quantization. One warp (CUDA block) owns
// one K-block and emits that block's own int8 scale, matching ORT/MLAS CompInt8
// and the CPU native path. A single per-row scale is dominated by activation
// outliers and rounds small in-block magnitudes to zero, flipping argmaxes on
// outlier-heavy models (e.g. Phi-3.5); per-block scales track fp32 faithfully.
extern "C" __global__ void matmul_nbits_quantize_accuracy4_block32(
    const float* activation,
    signed char* quantized_activation,
    float* activation_scale_out,
    const int k,
    const int padded_k)
{
    (void)padded_k;
    const int block = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    const int depth = block * 32 + lane;
    const float value = (depth < k) ? activation[depth] : 0.0f;

    float max_abs = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs,
            __shfl_down_sync(0xffffffffu, max_abs, offset));
    }
    max_abs = __shfl_sync(0xffffffffu, max_abs, 0);

    const float activation_scale = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
    const float inverse_scale =
        activation_scale == 0.0f ? 0.0f : 1.0f / activation_scale;
    if (lane == 0) {
        activation_scale_out[block] = activation_scale;
    }
    int quantized = 0;
    if (depth < k && activation_scale != 0.0f) {
        quantized = (int)roundf(fminf(127.0f, fmaxf(-127.0f,
            value * inverse_scale)));
    }
    quantized_activation[depth] = (signed char)quantized;
}

__device__ __forceinline__ int unpack_int4x4(unsigned int packed, int offset)
{
    const int w0 = (int)((packed >> (offset + 0)) & 15u) - 8;
    const int w1 = (int)((packed >> (offset + 4)) & 15u) - 8;
    const int w2 = (int)((packed >> (offset + 8)) & 15u) - 8;
    const int w3 = (int)((packed >> (offset + 12)) & 15u) - 8;
    return (w0 & 255) | ((w1 & 255) << 8) | ((w2 & 255) << 16)
        | ((w3 & 255) << 24);
}

__device__ __forceinline__ int dot_int8x4(int lhs, int rhs, int accumulator)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(lhs, rhs, accumulator);
#else
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
        int lhs_value = ((unsigned int)lhs >> (byte * 8)) & 255u;
        int rhs_value = ((unsigned int)rhs >> (byte * 8)) & 255u;
        lhs_value = lhs_value >= 128 ? lhs_value - 256 : lhs_value;
        rhs_value = rhs_value >= 128 ? rhs_value - 256 : rhs_value;
        accumulator += lhs_value * rhs_value;
    }
    return accumulator;
#endif
}

extern "C" __global__ void matmul_nbits_gemv_accuracy4_block32(
    const signed char* quantized_activation,
    const float* activation_scale_ptr,
    const unsigned char* packed,
    const float* scales,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int k_blocks)
{
    extern __shared__ signed char activation_tile[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int column = (int)blockIdx.x * 8 + warp;

    float value = 0.0f;
    for (int tile_block = 0; tile_block < k_blocks; tile_block += 32) {
        const int tile_blocks = min(32, k_blocks - tile_block);
        const int tile_depths = tile_blocks * 32;
        for (int depth = tid; depth < tile_depths; depth += (int)blockDim.x) {
            activation_tile[depth] =
                quantized_activation[tile_block * 32 + depth];
        }
        __syncthreads();

        const int block = tile_block + lane;
        if (column < n && block < k_blocks) {
            const long packed_start = ((long)column * k_blocks + block) * 16;
            const uint4 packed_weights =
                *reinterpret_cast<const uint4*>(packed + packed_start);
            const unsigned int words[4] = {
                packed_weights.x, packed_weights.y, packed_weights.z, packed_weights.w
            };
            const signed char* activation_block = activation_tile + lane * 32;
            int dot = 0;
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                const int activation0 =
                    *reinterpret_cast<const int*>(activation_block + word * 8);
                const int activation1 =
                    *reinterpret_cast<const int*>(activation_block + word * 8 + 4);
                dot = dot_int8x4(activation0, unpack_int4x4(words[word], 0), dot);
                dot = dot_int8x4(activation1, unpack_int4x4(words[word], 16), dot);
            }
            // Per-block int8 activation scale times the per-block weight scale.
            const float block_scale = __fmul_rn(
                activation_scale_ptr[block],
                scales[(long)column * k_blocks + block]);
            value = __fadd_rn(value, __fmul_rn((float)dot, block_scale));
        }
        __syncthreads();
    }

    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = bias ? __fadd_rn(value, bias[column]) : value;
    }
}

extern "C" __global__ void matmul_nbits_gemv_accuracy4_block32_stage64(
    const signed char* quantized_activation,
    const float* activation_scale_ptr,
    const unsigned char* packed,
    const float* scales,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int k_blocks)
{
    extern __shared__ signed char activation_tile[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int column = (int)blockIdx.x * 8 + warp;

    float value = 0.0f;
    for (int tile_block = 0; tile_block < k_blocks; tile_block += 64) {
        const int tile_blocks = min(64, k_blocks - tile_block);
        const int tile_depths = tile_blocks * 32;
        for (int depth = tid; depth < tile_depths; depth += (int)blockDim.x) {
            activation_tile[depth] =
                quantized_activation[tile_block * 32 + depth];
        }
        __syncthreads();

        for (int tile_offset = lane; tile_offset < tile_blocks; tile_offset += 32) {
            const int block = tile_block + tile_offset;
            if (column >= n) {
                continue;
            }
            const long packed_start = ((long)column * k_blocks + block) * 16;
            const uint4 packed_weights =
                *reinterpret_cast<const uint4*>(packed + packed_start);
            const unsigned int words[4] = {
                packed_weights.x, packed_weights.y, packed_weights.z, packed_weights.w
            };
            const signed char* activation_block = activation_tile + tile_offset * 32;
            int dot = 0;
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                const int activation0 =
                    *reinterpret_cast<const int*>(activation_block + word * 8);
                const int activation1 =
                    *reinterpret_cast<const int*>(activation_block + word * 8 + 4);
                dot = dot_int8x4(activation0, unpack_int4x4(words[word], 0), dot);
                dot = dot_int8x4(activation1, unpack_int4x4(words[word], 16), dot);
            }
            const float block_scale = __fmul_rn(
                activation_scale_ptr[block],
                scales[(long)column * k_blocks + block]);
            value = __fadd_rn(value, __fmul_rn((float)dot, block_scale));
        }
        __syncthreads();
    }

    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = bias ? __fadd_rn(value, bias[column]) : value;
    }
}

// Per-K-block int8 activation quantization for ANY power-of-two block_size. One
// warp (CUDA block) owns one K-block and emits that block's own int8 scale,
// exactly matching the per-block quantization the tiled `matmul_nbits_accuracy4`
// reference performs inline (block_max / 127, symmetric round-to-nearest). The
// block-32 entry above hard-codes 32 lanes == 32 depths; this generalization
// strides each of the 32 lanes across the block so a single warp covers
// block_size (e.g. 128) depths. Padded tail depths (depth >= k in a partial
// final block) quantize to zero so they contribute nothing to the GEMV.
extern "C" __global__ void matmul_nbits_quantize_accuracy4_blockwise(
    const float* activation,
    signed char* quantized_activation,
    float* activation_scale_out,
    const int k,
    const int block_size,
    const int padded_k)
{
    (void)padded_k;
    const int block = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    const int begin = block * block_size;

    float max_abs = 0.0f;
    for (int within = lane; within < block_size; within += 32) {
        const int depth = begin + within;
        const float value = (depth < k) ? activation[depth] : 0.0f;
        max_abs = fmaxf(max_abs, fabsf(value));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs,
            __shfl_down_sync(0xffffffffu, max_abs, offset));
    }
    max_abs = __shfl_sync(0xffffffffu, max_abs, 0);

    const float activation_scale = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
    const float inverse_scale =
        activation_scale == 0.0f ? 0.0f : 1.0f / activation_scale;
    if (lane == 0) {
        activation_scale_out[block] = activation_scale;
    }
    for (int within = lane; within < block_size; within += 32) {
        const int depth = begin + within;
        int quantized = 0;
        if (depth < k && activation_scale != 0.0f) {
            quantized = (int)roundf(fminf(127.0f, fmaxf(-127.0f,
                activation[depth] * inverse_scale)));
        }
        quantized_activation[depth] = (signed char)quantized;
    }
}

// General-block-size int4 accuracy_level=4 decode GEMV over the pre-quantized
// int8 activation. One warp reduces one output column; the 32 lanes cooperate
// across the block_size depths of each K-block. The per-block integer dot is
// computed as sum(qa * qw) - zero_point * sum(qa), which is exactly the tiled
// reference's sum(qa * (qw - zero_point)). Block-128 uses two packed dp4a
// instructions per lane (one for each integer sum) on sm_61+, while older
// architectures and other block sizes retain the scalar loop. Padded tail
// activations are zero, so packed tail lanes remain exact no-ops. The fp32 block
// products are then accumulated by lane 0 in ascending block order with the same
// __fmul_rn / __fadd_rn rounding the tiled kernel uses, so the result is bit-for-bit
// identical. Symmetric int4 uses the default zero point 8; asymmetric int4
// reads the packed per-block nibble zero point. The grid width (warps per CTA)
// is chosen host-side from the device multiprocessor count so the launch fills
// consumer and datacenter GPUs alike.
extern "C" __global__ void matmul_nbits_gemv_accuracy4_blockwise(
    const signed char* quantized_activation,
    const float* activation_scale_ptr,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int k_blocks,
    const int block_size,
    const int blob_size,
    const int zp_row_bytes)
{
    (void)k;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int column = (int)blockIdx.x * (int)(blockDim.x >> 5) + warp;
    if (column >= n) {
        return;
    }

    float value = 0.0f;
    for (int block = 0; block < k_blocks; ++block) {
        const int begin = block * block_size;
        const long blob_base = ((long)column * k_blocks + block) * blob_size;
        int weighted = 0;
        int activation_sum = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
        if (block_size == 128) {
            const int within = lane * 4;
            const int activation_pack = *reinterpret_cast<const int*>(
                quantized_activation + begin + within);
            const unsigned int packed_weights =
                (unsigned int)*reinterpret_cast<const unsigned short*>(
                    packed + blob_base + (within >> 1));
            const unsigned int even_weights = packed_weights & 0x0f0fu;
            const unsigned int odd_weights = (packed_weights >> 4) & 0x0f0fu;
            const unsigned int weight_pack =
                __byte_perm(even_weights, odd_weights, 0x5140);
            weighted = dot_int8x4(
                activation_pack, (int)weight_pack, weighted);
            activation_sum = dot_int8x4(
                activation_pack, 0x01010101, activation_sum);
        } else
#endif
        {
            for (int within = lane; within < block_size; within += 32) {
                const int quantized_activation_value =
                    (int)quantized_activation[begin + within];
                const unsigned char byte = packed[blob_base + (within >> 1)];
                const int quantized_weight =
                    (within & 1) ? (byte >> 4) : (byte & 15);
                weighted += quantized_activation_value * quantized_weight;
                activation_sum += quantized_activation_value;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            weighted += __shfl_down_sync(0xffffffffu, weighted, offset);
            activation_sum +=
                __shfl_down_sync(0xffffffffu, activation_sum, offset);
        }
        if (lane == 0) {
            int zero_point = 8;
            if (zero_points) {
                const unsigned char zp =
                    zero_points[(long)column * zp_row_bytes + (block >> 1)];
                zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
            }
            const int dot = weighted - zero_point * activation_sum;
            const float combined_scale = __fmul_rn(
                activation_scale_ptr[block],
                scales[(long)column * k_blocks + block]);
            value = __fadd_rn(value, __fmul_rn((float)dot, combined_scale));
        }
    }
    if (lane == 0) {
        output[column] = bias ? __fadd_rn(value, bias[column]) : value;
    }
}
"#;

const ACCURACY4_SRC: &str = r#"
extern "C" __global__ void matmul_nbits_accuracy4(
    const float* a,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* y,
    const int m,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    const long total = (long)m * n;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        const int row = (int)(idx / n);
        const int output = (int)(idx % n);
        const float* activation = a + (long)row * k;

        // Per-K-block int8 activation quantization (block scale == the weight
        // block granularity), matching ORT/MLAS CompInt8 and the CPU native
        // path. A single per-row scale is dominated by activation outliers and
        // collapses small in-block magnitudes to zero, which flips argmaxes on
        // outlier-heavy models (e.g. Phi-3.5); per-block scales avoid that.
        float value = 0.0f;
        for (int block = 0; block < k_blocks; ++block) {
            const int begin = block * block_size;
            const int end = min(begin + block_size, k);
            float block_max = 0.0f;
            for (int depth = begin; depth < end; ++depth) {
                block_max = fmaxf(block_max, fabsf(activation[depth]));
            }
            if (block_max == 0.0f) {
                continue;
            }
            const float activation_scale = block_max / 127.0f;
            const float inverse_scale = 1.0f / activation_scale;
            int zero_point = 8;
            if (zero_points) {
                const unsigned char zp =
                    zero_points[(long)output * zp_row_bytes + block / 2];
                zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
            }
            int dot = 0;
            for (int depth = begin; depth < end; ++depth) {
                int quantized_activation =
                    (int)roundf(fminf(127.0f, fmaxf(-127.0f,
                        activation[depth] * inverse_scale)));
                const int within = depth - begin;
                const unsigned char byte =
                    packed[((long)output * k_blocks + block) * blob_size + within / 2];
                const int quantized_weight =
                    (within & 1) ? (byte >> 4) : (byte & 15);
                dot += quantized_activation * (quantized_weight - zero_point);
            }
            const float combined_scale = __fmul_rn(
                activation_scale,
                scales[(long)output * k_blocks + block]);
            value = __fadd_rn(value, __fmul_rn((float)dot, combined_scale));
        }
        y[idx] = bias ? __fadd_rn(value, bias[output]) : value;
    }
}
"#;

// Direct fp16-activation x packed-int4 GEMV (decode M=1). Unlike the
// accuracy_level=4 path this performs NO separate int8 activation-quantization
// pass. Packed nibbles are converted in registers and multiplied by fp16
// activations directly. The common fp16-scale path uses half2 accumulation,
// matching the storage precision before an fp32 warp reduction; f32 scales use
// fp32 block accumulation. Both paths round once more to fp16 on write.
const GEMV_F16_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Narrowing store shared by the GEMV epilogues that can write their result
// straight into a bf16 consumer buffer instead of an fp16 staging buffer.
//
// `out_bf16` is a launch-uniform flag, so the branch costs one predicated
// select on the single lane that stores. The bf16 arm deliberately rounds
// fp32 -> fp16 -> bf16 rather than fp32 -> bf16 directly: the staging path it
// replaces rounds to fp16 in the GEMV and then casts that fp16 to bf16, and
// reproducing the double rounding is what keeps greedy decoding bit-identical.
__device__ __forceinline__ void matmul_nbits_store_narrowed(
    void* __restrict__ output, const int index, const __half value,
    const int out_bf16)
{
    if (out_bf16) {
        ((__nv_bfloat16*)output)[index] = __float2bfloat16(__half2float(value));
    } else {
        ((__half*)output)[index] = value;
    }
}

__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Fp16 GEMV bias epilogue. The fp32 accumulator is always rounded to fp16 for
// the base output. When a bias is present:
//   * `bias_post_round == 0` (native MatMulNBits bias): add in fp32 and round
//     once — `fp16(acc + bias)` — matching an ORT-style fused epilogue.
//   * `bias_post_round != 0` (a folded standalone `Add`): round the accumulator
//     to fp16 first, then add the fp16 bias with a second fp16 round —
//     `fp16(fp16(acc) + bias)` — reproducing the original two-op path so greedy
//     tokens stay byte-identical.
__device__ __forceinline__ __half fold_bias_f16(
    const float value,
    const __half* __restrict__ bias,
    const int column,
    const int bias_post_round)
{
    const __half rounded = __float2half(value);
    if (!bias) {
        return rounded;
    }
    const float b = __half2float(bias[column]);
    if (bias_post_round) {
        return __float2half(__half2float(rounded) + b);
    }
    return __float2half(value + b);
}

// One warp per output column. Block-32 INT8 stores one unsigned quantized
// weight byte per K element and one optional uint8 zero point per block.
extern "C" __global__ void matmul_nbits_gemv_int8_f16(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int scales_fp16,
    const int bias_post_round)
{
    // Mirrors the int4 `matmul_nbits_gemv_f16` work split: four adjacent lanes
    // cooperate on one block-32 column, eight blocks are consumed per warp step.
    // Each lane issues one aligned 8-byte packed-int8 load (uint2) and one 16-byte
    // activation load (uint4), then a four-lane shuffle reduction reconstructs the
    // block dot product before its scale is applied. This replaces the previous
    // one-byte-per-lane scalar walk that only advanced 32 K per warp step.
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    float value = 0.0f;
    if (column < n) {
        const int quarter = lane & 3;
        for (int block_base = 0; block_base < k_blocks; block_base += 8) {
            const int block = block_base + (lane >> 2);
            float block_partial = 0.0f;
            if (block < k_blocks) {
                const int zero_point =
                    zero_points ? (int)zero_points[(long)column * k_blocks + block] : 128;
                const int depth = block * 32 + quarter * 8;
                const long packed_start =
                    ((long)column * k_blocks + block) * 32 + quarter * 8;
                if (depth + 8 <= k) {
                    const uint2 packed_word =
                        *reinterpret_cast<const uint2*>(packed + packed_start);
                    const unsigned char* bytes =
                        reinterpret_cast<const unsigned char*>(&packed_word);
                    const uint4 act = *reinterpret_cast<const uint4*>(activation + depth);
                    const __half* acth = reinterpret_cast<const __half*>(&act);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        block_partial += ((float)(int)bytes[i] - (float)zero_point)
                            * __half2float(acth[i]);
                    }
                } else if (depth < k) {
                    const int valid = min(8, k - depth);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int quantized = (int)packed[packed_start + i];
                            block_partial += ((float)quantized - (float)zero_point)
                                * __half2float(activation[depth + i]);
                        }
                    }
                }
            }
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 2, 4);
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 1, 4);
            if (quarter == 0 && block < k_blocks) {
                const float scale = scales_fp16
                    ? __half2float(reinterpret_cast<const __half*>(scales_raw)
                        [(long)column * k_blocks + block])
                    : reinterpret_cast<const float*>(scales_raw)
                        [(long)column * k_blocks + block];
                value += block_partial * scale;
            }
        }
    }
    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

// Split-K standalone int8 GEMV: K_SPLIT warps cooperate on one output column,
// each reducing a strided subset of the 8-block (256-wide) K steps, then summing
// their fp32 partials through shared memory. The launch grid is K_SPLIT x larger
// than the single-warp kernel, which fills the SMs on the grid-starved
// (~0.48 waves/SM) Phi int8 down-projection decode GEMV. This kernel has no
// serial prologue, so the added grid parallelism directly hides the
// Long-Scoreboard latency (unlike the fused RMSNorm-prologue int8 kernel, whose
// serial full-vector prologue caps any split-K benefit). The fp32 partial sum is
// a new block-sum association, so results are near-equal (not byte-identical) to
// the single-warp kernel; asymmetric-zp parity is validated against a dequant
// reference to tolerance. Requires K % 256 == 0 (whole steps, no divergent tail)
// — the launch only routes here in that case.
extern "C" __global__ void matmul_nbits_gemv_int8_f16_splitk(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int scales_fp16,
    const int bias_post_round)
{
    constexpr int K_SPLIT = 2;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int cols_per_block = warps_per_block / K_SPLIT;
    const int col_local = warp / K_SPLIT;
    const int ks = warp % K_SPLIT;
    const int column = (int)blockIdx.x * cols_per_block + col_local;

    __shared__ float partials[8][K_SPLIT];

    float value = 0.0f;
    if (column < n) {
        const int quarter = lane & 3;
        for (int block_base = ks * 8; block_base < k_blocks;
             block_base += K_SPLIT * 8) {
            const int block = block_base + (lane >> 2);
            float block_partial = 0.0f;
            if (block < k_blocks) {
                const int zero_point =
                    zero_points ? (int)zero_points[(long)column * k_blocks + block] : 128;
                const int depth = block * 32 + quarter * 8;
                const long packed_start =
                    ((long)column * k_blocks + block) * 32 + quarter * 8;
                if (depth + 8 <= k) {
                    const uint2 packed_word =
                        *reinterpret_cast<const uint2*>(packed + packed_start);
                    const unsigned char* bytes =
                        reinterpret_cast<const unsigned char*>(&packed_word);
                    const uint4 act = *reinterpret_cast<const uint4*>(activation + depth);
                    const __half* acth = reinterpret_cast<const __half*>(&act);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        block_partial += ((float)(int)bytes[i] - (float)zero_point)
                            * __half2float(acth[i]);
                    }
                } else if (depth < k) {
                    const int valid = min(8, k - depth);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int quantized = (int)packed[packed_start + i];
                            block_partial += ((float)quantized - (float)zero_point)
                                * __half2float(activation[depth + i]);
                        }
                    }
                }
            }
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 2, 4);
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 1, 4);
            if (quarter == 0 && block < k_blocks) {
                const float scale = scales_fp16
                    ? __half2float(reinterpret_cast<const __half*>(scales_raw)
                        [(long)column * k_blocks + block])
                    : reinterpret_cast<const float*>(scales_raw)
                        [(long)column * k_blocks + block];
                value += block_partial * scale;
            }
        }
    }
    value = warp_sum(value);
    if (lane == 0) {
        partials[col_local][ks] = (column < n) ? value : 0.0f;
    }
    __syncthreads();
    if (ks == 0 && lane == 0 && column < n) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < K_SPLIT; ++s) {
            acc += partials[col_local][s];
        }
        output[column] = fold_bias_f16(acc, bias, column, bias_post_round);
    }
}
// block-32 activation/weight tile, so each packed weight is reused by up to 16
// prompt rows and each activation by up to 16 output columns. It deliberately
// uses only ordinary shared memory, fp32 arithmetic, and __half conversion:
// no tensor-core, async-copy, or architecture-specific PTX requirement.
extern "C" __global__ void matmul_nbits_gemm_f16(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int m,
    const int k,
    const int n,
    const int k_blocks,
    const int bits,
    const int scales_fp16,
    const int bias_post_round,
    const int bias_row_stride)
{
    __shared__ float activation_tile[16][32];
    __shared__ float weight_tile[32][16];
    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;
    const int row = (int)blockIdx.y * 16 + (int)threadIdx.y;
    const int column = (int)blockIdx.x * 16 + (int)threadIdx.x;
    float value = 0.0f;

    for (int block = 0; block < k_blocks; ++block) {
#pragma unroll
        for (int load = tid; load < 16 * 32; load += 16 * 16) {
            const int tile_row = load >> 5;
            const int within = load & 31;
            const int depth = block * 32 + within;
            const int global_row = (int)blockIdx.y * 16 + tile_row;
            activation_tile[tile_row][within] =
                global_row < m && depth < k
                    ? __half2float(activation[(long)global_row * k + depth])
                    : 0.0f;
        }
#pragma unroll
        for (int load = tid; load < 32 * 16; load += 16 * 16) {
            const int tile_column = load >> 5;
            const int within = load & 31;
            const int global_column = (int)blockIdx.x * 16 + tile_column;
            const int depth = block * 32 + within;
            float weight = 0.0f;
            if (global_column < n && depth < k) {
                const long scale_index = (long)global_column * k_blocks + block;
                const float scale = scales_fp16
                    ? __half2float(
                        reinterpret_cast<const __half*>(scales_raw)[scale_index])
                    : reinterpret_cast<const float*>(scales_raw)[scale_index];
                int quantized;
                int zero_point;
                if (bits == 8) {
                    quantized = (int)packed[scale_index * 32 + within];
                    zero_point = zero_points ? (int)zero_points[scale_index] : 128;
                } else {
                    const unsigned char byte =
                        packed[scale_index * 16 + (within >> 1)];
                    quantized = (within & 1) ? (byte >> 4) : (byte & 15);
                    zero_point = 8;
                    if (zero_points) {
                        const int zp_row_bytes = (k_blocks + 1) >> 1;
                        const unsigned char zp =
                            zero_points[(long)global_column * zp_row_bytes + (block >> 1)];
                        zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
                    }
                }
                weight = ((float)quantized - (float)zero_point) * scale;
            }
            weight_tile[within][tile_column] = weight;
        }
        __syncthreads();

        if (row < m && column < n) {
#pragma unroll
            for (int within = 0; within < 32; ++within) {
                value += activation_tile[threadIdx.y][within]
                    * weight_tile[within][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (row < m && column < n) {
        // A folded residual epilogue binds a per-token residual (row stride N)
        // into the bias slot; a genuine broadcast bias keeps stride 0.
        const __half* row_bias = bias ? bias + (long)row * bias_row_stride : bias;
        output[(long)row * n + column] =
            fold_bias_f16(value, row_bias, column, bias_post_round);
    }
}

__device__ __forceinline__ void int4x8_to_half2x4_sub(
    const unsigned int packed,
    __half2* values,
    const unsigned int sub2)
{
    unsigned int* h = reinterpret_cast<unsigned int*>(values);
    constexpr unsigned int bottom_mask = 0x000f000f;
    constexpr unsigned int top_mask = 0x00f000f0;
    constexpr unsigned int fp16_magic = 0x64006400;
    constexpr unsigned int lop3_lut = (0xf0 & 0xcc) | 0xaa;
    const unsigned int top = packed >> 8;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(packed), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(packed), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));

    constexpr unsigned int fp16_1024 = 0x64006400;
    constexpr unsigned int fp16_one_sixteenth = 0x2c002c00;
    constexpr unsigned int fp16_neg64 = 0xd400d400;
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[0]) : "r"(h[0]), "r"(fp16_1024));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[1])
                 : "r"(h[1]), "r"(fp16_one_sixteenth), "r"(fp16_neg64));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[2]) : "r"(h[2]), "r"(fp16_1024));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[3])
                 : "r"(h[3]), "r"(fp16_one_sixteenth), "r"(fp16_neg64));
    // Center each nibble by subtracting the block zero point. A symmetric int4
    // weight uses the implicit `sub2 == 8` (fp16 0x48004800), which reproduces
    // the previous fixed `- 8` byte-for-byte; an asymmetric weight passes its
    // per-block zero point instead so the dequant is `(code - zp)`.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        asm volatile("sub.f16x2 %0, %1, %2;\n"
                     : "=r"(h[i]) : "r"(h[i]), "r"(sub2));
    }
}

// Symmetric int4 dequant: `(code - 8)` in fp16, byte-identical to the historical
// hard-coded `- 8` path (the `sub2` register just carries fp16 8.0).
__device__ __forceinline__ void int4x8_to_half2x4(
    const unsigned int packed,
    __half2* values)
{
    constexpr unsigned int fp16_eight = 0x48004800;
    int4x8_to_half2x4_sub(packed, values, fp16_eight);
}

// Fused symmetric int4 dequant: `(code - 8)` in fp16, emitting FOUR fewer
// `f16x2` ALU ops per packed word than [`int4x8_to_half2x4`] by folding the
// symmetric zero point (`- 8`) into the magic-bias-removal constants instead of
// issuing it as a separate trailing `sub.f16x2` per element pair.
//
// BYTE-IDENTICAL to `int4x8_to_half2x4` (the `- 8` path): every intermediate is
// an exactly-representable fp16 integer, so folding the two constant
// subtractions into one changes no rounding:
//   * bottom nibbles decode to `1024 + code` (exact in [1024, 1039]); the
//     original does `(x - 1024) - 8`, this does `x - 1032` — `1032 = 0x6408` is
//     exact and `(1024 + code) - 1032 = code - 8` with no intermediate rounding.
//   * top nibbles decode to `1024 + 16*code`; the original fma `x*(1/16) - 64`
//     yields the exact integer `code`, then `- 8`; this fuses to
//     `x*(1/16) - 72` (`72 = 0xD480` exact). `x*(1/16)` is an exact power-of-two
//     scale, so the single fma rounding lands on `code - 8` identically.
// This is a pure instruction-count reduction for the issue-bound symmetric
// paired gate/up decode GEMV — no reassociation, no math change.
__device__ __forceinline__ void int4x8_to_half2x4_sym8(
    const unsigned int packed,
    __half2* values)
{
    unsigned int* h = reinterpret_cast<unsigned int*>(values);
    constexpr unsigned int bottom_mask = 0x000f000f;
    constexpr unsigned int top_mask = 0x00f000f0;
    constexpr unsigned int fp16_magic = 0x64006400;
    constexpr unsigned int lop3_lut = (0xf0 & 0xcc) | 0xaa;
    const unsigned int top = packed >> 8;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(packed), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(packed), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));

    // 1032 = 1024 (magic bias) + 8 (symmetric zero point); -72 = -64 - 8.
    constexpr unsigned int fp16_1032 = 0x64086408;
    constexpr unsigned int fp16_one_sixteenth = 0x2c002c00;
    constexpr unsigned int fp16_neg72 = 0xd480d480;
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[0]) : "r"(h[0]), "r"(fp16_1032));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[1])
                 : "r"(h[1]), "r"(fp16_one_sixteenth), "r"(fp16_neg72));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[2]) : "r"(h[2]), "r"(fp16_1032));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[3])
                 : "r"(h[3]), "r"(fp16_one_sixteenth), "r"(fp16_neg72));
}

// Pack a scalar block zero point (nibble in [0, 15]) into an fp16x2 subtrahend
// for [`int4x8_to_half2x4_sub`].
__device__ __forceinline__ unsigned int int4_zero_point_sub2(const int zero_point)
{
    const __half zp = __float2half((float)zero_point);
    const __half2 zp2 = __halves2half2(zp, zp);
    return *reinterpret_cast<const unsigned int*>(&zp2);
}

// Load the block zero point for `column`/`block` from the packed nibble layout,
// or the symmetric default (8) when the weight carries no zero points.
__device__ __forceinline__ int int4_block_zero_point(
    const unsigned char* __restrict__ zero_points,
    const long column,
    const int block,
    const int zp_row_bytes)
{
    if (!zero_points) {
        return 8;
    }
    const unsigned char zp = zero_points[column * zp_row_bytes + (block >> 1)];
    return (block & 1) ? (zp >> 4) : (zp & 15);
}

// Compile-time-specialized per-block subtrahend for the vectorized int4 GEMVs.
// `HasZp == false` (symmetric weights) folds to the constant fp16 `8.0`
// subtrahend with no memory traffic, so the compiler emits the exact
// pre-zero-point instruction stream; `HasZp == true` reads the per-block
// asymmetric zero point. Keying off the template parameter — never the runtime
// pointer — keeps the symmetric decode path byte-identical and register-light.
template <bool HasZp>
__device__ __forceinline__ unsigned int block_sub2(
    const unsigned char* __restrict__ zero_points,
    const long column,
    const int block,
    const int zp_row_bytes)
{
    if (!HasZp) {
        return 0x48004800u;
    }
    return int4_zero_point_sub2(
        int4_block_zero_point(zero_points, column, block, zp_row_bytes));
}

// Scalar counterpart of [`block_sub2`] for the partial-block tail. `HasZp ==
// false` returns the symmetric default (8) with no load.
template <bool HasZp>
__device__ __forceinline__ int block_zp(
    const unsigned char* __restrict__ zero_points,
    const long column,
    const int block,
    const int zp_row_bytes)
{
    if (!HasZp) {
        return 8;
    }
    return int4_block_zero_point(zero_points, column, block, zp_row_bytes);
}

__device__ __forceinline__ float dot_int4x8_f16(
    const unsigned int packed,
    const __half* __restrict__ activation)
{
    const uint4 a = *reinterpret_cast<const uint4*>(activation);
    constexpr unsigned int low_halves = 0x5410;
    constexpr unsigned int high_halves = 0x7632;
    uint4 permuted;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.x) : "r"(a.x), "r"(a.z), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.y) : "r"(a.x), "r"(a.z), "r"(high_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.z) : "r"(a.y), "r"(a.w), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.w) : "r"(a.y), "r"(a.w), "r"(high_halves));

    __half2 q[4];
    int4x8_to_half2x4(packed, q);
    const float2 q04 = __half22float2(q[0]);
    const float2 q15 = __half22float2(q[1]);
    const float2 q26 = __half22float2(q[2]);
    const float2 q37 = __half22float2(q[3]);
    const float2 a04 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.x));
    const float2 a15 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.y));
    const float2 a26 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.z));
    const float2 a37 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.w));
    float dot = q04.x * a04.x;
    dot += q15.x * a15.x;
    dot += q26.x * a26.x;
    dot += q37.x * a37.x;
    dot += q04.y * a04.y;
    dot += q15.y * a15.y;
    dot += q26.y * a26.y;
    dot += q37.y * a37.y;
    return dot;
}

// Zero-point-aware [`dot_int4x8_f16`]. `sub2` is the fp16x2 subtrahend for this
// block (the packed zero point, or fp16 8.0 for symmetric weights). Because the
// centered code `(code - zp)` is an exact fp16 integer in [-15, 15], converting
// each `q` to float and accumulating the eight products in fp32 in ascending
// element order reproduces the scalar `(float)(code - zp) * __half2float(act)`
// path byte-for-byte — the LOP3 unpack only changes *how* the nibbles are
// decoded, not the arithmetic that follows. With `sub2 == 0x48004800` this is
// identical to [`dot_int4x8_f16`].
__device__ __forceinline__ float dot_int4x8_f16_sub(
    const unsigned int packed,
    const __half* __restrict__ activation,
    const unsigned int sub2)
{
    const uint4 a = *reinterpret_cast<const uint4*>(activation);
    constexpr unsigned int low_halves = 0x5410;
    constexpr unsigned int high_halves = 0x7632;
    uint4 permuted;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.x) : "r"(a.x), "r"(a.z), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.y) : "r"(a.x), "r"(a.z), "r"(high_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.z) : "r"(a.y), "r"(a.w), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.w) : "r"(a.y), "r"(a.w), "r"(high_halves));

    __half2 q[4];
    int4x8_to_half2x4_sub(packed, q, sub2);
    const float2 q04 = __half22float2(q[0]);
    const float2 q15 = __half22float2(q[1]);
    const float2 q26 = __half22float2(q[2]);
    const float2 q37 = __half22float2(q[3]);
    const float2 a04 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.x));
    const float2 a15 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.y));
    const float2 a26 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.z));
    const float2 a37 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.w));
    float dot = q04.x * a04.x;
    dot += q15.x * a15.x;
    dot += q26.x * a26.x;
    dot += q37.x * a37.x;
    dot += q04.y * a04.y;
    dot += q15.y * a15.y;
    dot += q26.y * a26.y;
    dot += q37.y * a37.y;
    return dot;
}

// Split of `dot_int4x8_f16_sub` into an activation-decode half and a
// weight-dot half so the decoded activation can be REUSED across several output
// columns (column register-blocking). `decode_activation8` converts the eight
// contiguous fp16 activations at `activation` into fp32, laid out in the exact
// summation order `dot_int4x8_f16_sub` consumes them (a04.x, a15.x, a26.x,
// a37.x, a04.y, a15.y, a26.y, a37.y). `dot_int4x8_f16_sub_act` then reproduces
// the identical 8-term fp32 dot for one weight sub-word. Because the fp16->fp32
// conversions and the add order are unchanged, the result is BIT-IDENTICAL to
// `dot_int4x8_f16_sub`; the only difference is the activation is decoded once
// and shared by all columns instead of re-loaded per column.
__device__ __forceinline__ void decode_activation8(
    const __half* __restrict__ activation,
    float* __restrict__ a /* [8] */)
{
    const uint4 av = *reinterpret_cast<const uint4*>(activation);
    constexpr unsigned int low_halves = 0x5410;
    constexpr unsigned int high_halves = 0x7632;
    uint4 permuted;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.x) : "r"(av.x), "r"(av.z), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.y) : "r"(av.x), "r"(av.z), "r"(high_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.z) : "r"(av.y), "r"(av.w), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.w) : "r"(av.y), "r"(av.w), "r"(high_halves));
    const float2 a04 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.x));
    const float2 a15 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.y));
    const float2 a26 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.z));
    const float2 a37 = __half22float2(*reinterpret_cast<const __half2*>(&permuted.w));
    a[0] = a04.x;
    a[1] = a15.x;
    a[2] = a26.x;
    a[3] = a37.x;
    a[4] = a04.y;
    a[5] = a15.y;
    a[6] = a26.y;
    a[7] = a37.y;
}

__device__ __forceinline__ float dot_int4x8_f16_sub_act(
    const unsigned int packed,
    const float* __restrict__ a /* [8] */,
    const unsigned int sub2)
{
    __half2 q[4];
    int4x8_to_half2x4_sub(packed, q, sub2);
    const float2 q04 = __half22float2(q[0]);
    const float2 q15 = __half22float2(q[1]);
    const float2 q26 = __half22float2(q[2]);
    const float2 q37 = __half22float2(q[3]);
    float dot = q04.x * a[0];
    dot += q15.x * a[1];
    dot += q26.x * a[2];
    dot += q37.x * a[3];
    dot += q04.y * a[4];
    dot += q15.y * a[5];
    dot += q26.y * a[6];
    dot += q37.y * a[7];
    return dot;
}

// ---------------------------------------------------------------------------
// TRT-LLM-style interleaved + biased int4 -> fp16 decode (OPT-IN, symmetric).
//
// This is the runtime half of the `ONNX_GENAI_INTERLEAVE_DEQUANT` lever. It
// consumes weights that were **offline-interleaved** (nibbles of each 32-bit
// word rearranged from natural [e7 e6 e5 e4 | e3 e2 e1 e0] to the TRT-LLM
// [e7 e5 e3 e1 | e6 e4 e2 e0] order — even elements in the low four nibble
// slots, odd in the high four) by the host-side interleave pass. Given that
// layout, the SAME 4x LOP3 unpack that `int4x8_to_half2x4_sub` uses now emits
// the eight fp16 codes in NATURAL element order `{e0,e1},{e2,e3},{e4,e5},
// {e6,e7}`, so no `prmt.b32` activation reorder is needed downstream (the
// activation is consumed straight as contiguous __half2 pairs).
//
// It also folds the symmetric `-8` bias directly into the LOP3 magic-removal
// constants: the even lanes subtract 1032 (0x64086408 = 1024 + 8) and the odd
// lanes fma by 1/16 then subtract 72 (0xd480 = -(64 + 8)), so the converter
// yields `(code - 8)` in fp16 with NO trailing `sub.f16x2` loop. Total is
// 1 shift + 4 LOP3 + 2 sub.f16x2 + 2 fma.f16x2 = 9 instructions / 8 values.
//
// Correctness: for symmetric weights the previous path produces `(code - 8)` in
// fp16 via magic-removal followed by a `- 8` subtract; `(code - 8)` is an exact
// fp16 integer in [-8, 7], so this single-step form yields the byte-identical
// fp16 value. Because the eight products are then accumulated in fp32 in the
// same ascending element order (e0*a0, e1*a1, ..., e7*a7), the dot is
// BIT-IDENTICAL to `dot_int4x8_f16_sub` with `sub2 == fp16 8.0`.
__device__ __forceinline__ void int4x8_to_half2x4_interleaved_biased(
    const unsigned int packed,
    __half2* values)
{
    unsigned int* h = reinterpret_cast<unsigned int*>(values);
    constexpr unsigned int bottom_mask = 0x000f000f;
    constexpr unsigned int top_mask = 0x00f000f0;
    constexpr unsigned int fp16_magic = 0x64006400;
    constexpr unsigned int lop3_lut = (0xf0 & 0xcc) | 0xaa;
    const unsigned int top = packed >> 8;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(packed), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(packed), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));

    // Fold the symmetric -8 bias into the magic-removal: even lanes hold
    // (1024 + code) and subtract 1032; odd lanes hold (1024 + 16*code) and
    // fma (*1/16 - 72), both yielding (code - 8) directly.
    constexpr unsigned int fp16_1032 = 0x64086408;
    constexpr unsigned int fp16_one_sixteenth = 0x2c002c00;
    constexpr unsigned int fp16_neg72 = 0xd480d480;
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[0]) : "r"(h[0]), "r"(fp16_1032));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[1])
                 : "r"(h[1]), "r"(fp16_one_sixteenth), "r"(fp16_neg72));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[2]) : "r"(h[2]), "r"(fp16_1032));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[3])
                 : "r"(h[3]), "r"(fp16_one_sixteenth), "r"(fp16_neg72));
}

// Natural-order activation decode for the interleaved dequant path: the eight
// contiguous fp16 activations are converted to fp32 in ascending order with no
// `prmt.b32` reorder (the interleaved converter already emits the weights in
// natural element order). `a[i] == float(activation[i])`, matching the layout
// `decode_activation8` produces for the non-interleaved path.
__device__ __forceinline__ void decode_activation8_natural(
    const __half* __restrict__ activation,
    float* __restrict__ a /* [8] */)
{
    const uint4 av = *reinterpret_cast<const uint4*>(activation);
    const float2 a01 = __half22float2(*reinterpret_cast<const __half2*>(&av.x));
    const float2 a23 = __half22float2(*reinterpret_cast<const __half2*>(&av.y));
    const float2 a45 = __half22float2(*reinterpret_cast<const __half2*>(&av.z));
    const float2 a67 = __half22float2(*reinterpret_cast<const __half2*>(&av.w));
    a[0] = a01.x;
    a[1] = a01.y;
    a[2] = a23.x;
    a[3] = a23.y;
    a[4] = a45.x;
    a[5] = a45.y;
    a[6] = a67.x;
    a[7] = a67.y;
}

// Interleaved-weight sibling of `dot_int4x8_f16_sub_act`: the pre-decoded
// natural-order activation `a[8]` is dotted with the eight `(code - 8)` fp16
// weights from the biased interleaved converter, accumulated in fp32 in
// ascending element order. Bit-identical to `dot_int4x8_f16_sub_act` with
// `sub2 == fp16 8.0` on the non-interleaved layout of the same logical weights.
__device__ __forceinline__ float dot_int4x8_f16_interleaved_act(
    const unsigned int packed,
    const float* __restrict__ a /* [8] */)
{
    __half2 q[4];
    int4x8_to_half2x4_interleaved_biased(packed, q);
    const float2 q01 = __half22float2(q[0]);
    const float2 q23 = __half22float2(q[1]);
    const float2 q45 = __half22float2(q[2]);
    const float2 q67 = __half22float2(q[3]);
    float dot = q01.x * a[0];
    dot += q01.y * a[1];
    dot += q23.x * a[2];
    dot += q23.y * a[3];
    dot += q45.x * a[4];
    dot += q45.y * a[5];
    dot += q67.x * a[6];
    dot += q67.y * a[7];
    return dot;
}

// Interleaved-weight sibling of `dot_int4x8_f16_sub` (scalar-tail helper): loads
// the eight contiguous fp16 activations itself, then dots as above.
__device__ __forceinline__ float dot_int4x8_f16_interleaved(
    const unsigned int packed,
    const __half* __restrict__ activation)
{
    float a[8];
    decode_activation8_natural(activation, a);
    return dot_int4x8_f16_interleaved_act(packed, a);
}

// ---------------------------------------------------------------------------
// High-memory-level-parallelism (wide-load) int4 decode GEMV.
//
// The single-warp `..._general_bs` loop loads ONE 32-bit weight word (8 nibbles)
// per lane per step and immediately dequant+FMAs it — a dependent
// load->dequant->FMA chain that keeps ~1 weight load in flight per lane, so at
// M=1 (where the weight stream is the entire DRAM traffic) DRAM tops out ~19%
// while the SM pipe saturates on narrow-load issue. A head-to-head ncu of ORT's
// `MatMulFloatInt4Kernel` on the identical gate_up matrix showed it streams the
// same weights at 2.42 TB/s (50% DRAM) vs our 0.92 TB/s at the SAME grid/block/
// registers/occupancy — a pure memory-level-parallelism gap, not a math or
// tiling difference.
//
// This helper closes that gap: each lane owns 32 CONTIGUOUS nibbles per step and
// loads them with ONE 128-bit `uint4` weight load (4x fewer load instructions,
// 4x bytes/instruction), and the next step's `uint4` is issued BEFORE the
// current step's dequant/FMA so >=2 wide loads are always in flight to hide the
// ~10-cycle Long-Scoreboard global-load latency. It reuses the byte-tested LOP3
// dequant (`dot_int4x8_f16_sub`) on the four 8-nibble sub-words, so the
// per-element arithmetic is unchanged. NOT cp.async (pure issue overhead at M=1,
// proven regressing) and NOT a scalar `#pragma unroll` of the LOP3 body
// (register bloat -> occupancy cliff, proven regressing) — just wider
// synchronous vector loads at ~constant register footprint, the mechanism ORT
// uses.
//
// Numerics: the 32-wide lane interleave (vs the 8-wide single-warp interleave)
// regroups the fp32 partial sums, so the result is near-equal, NOT byte-
// identical, to `..._general_bs` — exactly the K-slice reassociation the split-K
// entries already ship by default. Each 32-nibble chunk lies inside ONE block
// (guarded to `block_size % 32 == 0`, i.e. glm block-128 and qwen block-32; the
// rare block-16 export falls back to the narrow kernel), so a single scale and
// zero point cover the chunk and the `uint4` load is 16-byte aligned.
//
// Returns this lane's fp32 partial (pre warp-reduction) for the weight column
// `column`, walking chunk starts `depth0, depth0+warp_stride, ...`.
// Read logical element `i` (0..7) from an INTERLEAVED 32-bit weight word. The
// offline interleave stores physical nibble slots as [e0,e2,e4,e6,e1,e3,e5,e7],
// so logical i maps to physical slot (i even ? i/2 : 4 + i/2). Used only by the
// interleaved GEMV's scalar tail; the main loop reads whole words via the LOP3
// converter.
__device__ __forceinline__ int interleaved_nibble(const unsigned int word, const int i)
{
    const int slot = (i & 1) ? (4 + (i >> 1)) : (i >> 1);
    return (int)((word >> (slot * 4)) & 15u);
}

__device__ __forceinline__ float gemv_int4_wide_lane_dot(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const int k,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const long column,
    const int depth0,
    const int warp_stride)
{
    const long col_kb = column * (long)k_blocks;

    float value = 0.0f;
    int depth = depth0;
    bool have = (depth + 32 <= k);
    uint4 w;
    if (have) {
        const int block = depth / block_size;
        const long blob_base = (col_kb + block) * (long)blob_size;
        const int within = depth - block * block_size;
        w = *reinterpret_cast<const uint4*>(packed + blob_base + (within >> 1));
    }
    while (have) {
        const int ndepth = depth + warp_stride;
        const bool have_next = (ndepth + 32 <= k);
        uint4 wn;
        if (have_next) {
            // Issue the next wide weight load before consuming the current one,
            // so two 128-bit loads are in flight across the dequant/FMA below.
            const int nblock = ndepth / block_size;
            const long nblob = (col_kb + nblock) * (long)blob_size;
            const int nwithin = ndepth - nblock * block_size;
            wn = *reinterpret_cast<const uint4*>(packed + nblob + (nwithin >> 1));
        }
        // `depth` is 32-aligned and `block_size % 32 == 0`, so `[depth, depth+32)`
        // is inside a single block: one scale + one zero point cover the chunk.
        const int block = depth / block_size;
        float scale;
        if (scales_fp16) {
            scale = __half2float(reinterpret_cast<const __half*>(scales)[col_kb + block]);
        } else {
            scale = reinterpret_cast<const float*>(scales)[col_kb + block];
        }
        const unsigned int sub2 =
            int4_zero_point_sub2(int4_block_zero_point(zero_points, column, block, zp_row_bytes));
        value += scale * dot_int4x8_f16_sub(w.x, activation + depth, sub2);
        value += scale * dot_int4x8_f16_sub(w.y, activation + depth + 8, sub2);
        value += scale * dot_int4x8_f16_sub(w.z, activation + depth + 16, sub2);
        value += scale * dot_int4x8_f16_sub(w.w, activation + depth + 24, sub2);
        depth = ndepth;
        w = wn;
        have = have_next;
    }

    // Partial trailing chunk for this lane (`depth < k < depth + 32`): decode the
    // valid 8-nibble sub-words, matching the narrow kernel's tail arithmetic.
    if (depth < k) {
        const int block = depth / block_size;
        float scale;
        if (scales_fp16) {
            scale = __half2float(reinterpret_cast<const __half*>(scales)[col_kb + block]);
        } else {
            scale = reinterpret_cast<const float*>(scales)[col_kb + block];
        }
        const int zero_point = int4_block_zero_point(zero_points, column, block, zp_row_bytes);
        const unsigned int sub2 = int4_zero_point_sub2(zero_point);
        const long blob_base = (col_kb + block) * (long)blob_size;
        for (int off = 0; off < 32 && depth + off < k; off += 8) {
            const int d = depth + off;
            const int within = d - block * block_size;
            const int valid = min(8, k - d);
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed + blob_base + (within >> 1));
            if (valid == 8) {
                value += scale * dot_int4x8_f16_sub(packed_word, activation + d, sub2);
            } else {
                float partial = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (i < valid) {
                        const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                        partial += (float)q * __half2float(activation[d + i]);
                    }
                }
                value += partial * scale;
            }
        }
    }
    return value;
}

// Interleaved + biased (symmetric-only) sibling of `gemv_int4_wide_lane_dot`.
// Single-column wide-load lane dot that consumes offline-interleaved weights and
// folds the fixed symmetric `-8` bias inside the LOP3 converter (no per-block
// zero point, no `prmt.b32` activation reorder). The depth-2 software pipeline,
// the ascending sub-word order (w.x, w.y, w.z, w.w), and the fp32 accumulation
// order are byte-for-byte identical to `gemv_int4_wide_lane_dot` on symmetric
// weights, so each lane's fp32 partial is BIT-IDENTICAL. Used by the split-K
// wide interleaved kernel (the split-K partials therefore reduce to the same
// value as the non-interleaved split-K wide kernel).
__device__ __forceinline__ float gemv_int4_wide_lane_dot_interleaved(
   const __half* __restrict__ activation,
   const unsigned char* __restrict__ packed,
   const void* __restrict__ scales,
   const int k,
   const int block_size,
   const int k_blocks,
   const int blob_size,
   const int scales_fp16,
   const long column,
   const int depth0,
   const int warp_stride)
{
   const long col_kb = column * (long)k_blocks;

   float value = 0.0f;
   int depth = depth0;
   bool have = (depth + 32 <= k);
   uint4 w;
   if (have) {
       const int block = depth / block_size;
       const long blob_base = (col_kb + block) * (long)blob_size;
       const int within = depth - block * block_size;
       w = *reinterpret_cast<const uint4*>(packed + blob_base + (within >> 1));
   }
   while (have) {
       const int ndepth = depth + warp_stride;
       const bool have_next = (ndepth + 32 <= k);
       uint4 wn;
       if (have_next) {
           const int nblock = ndepth / block_size;
           const long nblob = (col_kb + nblock) * (long)blob_size;
           const int nwithin = ndepth - nblock * block_size;
           wn = *reinterpret_cast<const uint4*>(packed + nblob + (nwithin >> 1));
       }
       const int block = depth / block_size;
       float scale;
       if (scales_fp16) {
           scale = __half2float(reinterpret_cast<const __half*>(scales)[col_kb + block]);
       } else {
           scale = reinterpret_cast<const float*>(scales)[col_kb + block];
       }
       value += scale * dot_int4x8_f16_interleaved(w.x, activation + depth);
       value += scale * dot_int4x8_f16_interleaved(w.y, activation + depth + 8);
       value += scale * dot_int4x8_f16_interleaved(w.z, activation + depth + 16);
       value += scale * dot_int4x8_f16_interleaved(w.w, activation + depth + 24);
       depth = ndepth;
       w = wn;
       have = have_next;
   }

   if (depth < k) {
       const int block = depth / block_size;
       float scale;
       if (scales_fp16) {
           scale = __half2float(reinterpret_cast<const __half*>(scales)[col_kb + block]);
       } else {
           scale = reinterpret_cast<const float*>(scales)[col_kb + block];
       }
       const long blob_base = (col_kb + block) * (long)blob_size;
       for (int off = 0; off < 32 && depth + off < k; off += 8) {
           const int d = depth + off;
           const int within = d - block * block_size;
           const int valid = min(8, k - d);
           const unsigned int packed_word =
               *reinterpret_cast<const unsigned int*>(packed + blob_base + (within >> 1));
           if (valid == 8) {
               value += scale * dot_int4x8_f16_interleaved(packed_word, activation + d);
           } else {
               float partial = 0.0f;
#pragma unroll
               for (int i = 0; i < 8; ++i) {
                   if (i < valid) {
                       const int q = interleaved_nibble(packed_word, i) - 8;
                       partial += (float)q * __half2float(activation[d + i]);
                   }
               }
               value += partial * scale;
           }
       }
   }
   return value;
}

// Column register-blocked wide lane-dot: one warp accumulates WIDE_NC output
// columns at once. Each 8-element activation sub-word is decoded to fp32 ONCE
// (`decode_activation8`) and reused across all WIDE_NC columns, cutting the
// redundant activation L1 traffic (the head-to-head ncu limiter on the wide
// gate_up kernel was L1/TEX throughput, not DRAM) by ~WIDE_NC x, while the
// WIDE_NC independent 128-bit weight loads per chunk supply the memory-level
// parallelism that hides the Long-Scoreboard latency (replacing the depth-2
// software pipeline of the single-column `gemv_int4_wide_lane_dot`). The
// per-column `values[c] += scale * dot(sub-word)` sequence is byte-for-byte the
// same order as `gemv_int4_wide_lane_dot`, so each column's fp32 result is
// BIT-IDENTICAL to the single-column wide kernel (hence to the narrow kernel it
// already matches). `col_base` is this warp's first column; columns
// `col_base + c` beyond `n` are skipped.
#define WIDE_NC 4
__device__ __forceinline__ void gemv_int4_wide_lane_dot_multicol(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const int k,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const long col_base,
    const int n,
    const int depth0,
    const int warp_stride,
    float* __restrict__ values /* [WIDE_NC] */)
{
    bool valid[WIDE_NC];
    long col_kb[WIDE_NC];
#pragma unroll
    for (int c = 0; c < WIDE_NC; ++c) {
        values[c] = 0.0f;
        const long column = col_base + c;
        valid[c] = (column < n);
        col_kb[c] = column * (long)k_blocks;
    }

    int depth = depth0;
    while (depth + 32 <= k) {
        const int block = depth / block_size;
        const int within = depth - block * block_size;

        // Issue all WIDE_NC independent 128-bit weight loads up front so they are
        // in flight together (the load-level parallelism that hides load latency).
        uint4 w[WIDE_NC];
        float scale[WIDE_NC];
        unsigned int sub2[WIDE_NC];
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            if (!valid[c]) {
                continue;
            }
            const long base = (col_kb[c] + block) * (long)blob_size;
            w[c] = *reinterpret_cast<const uint4*>(packed + base + (within >> 1));
            if (scales_fp16) {
                scale[c] = __half2float(reinterpret_cast<const __half*>(scales)[col_kb[c] + block]);
            } else {
                scale[c] = reinterpret_cast<const float*>(scales)[col_kb[c] + block];
            }
            sub2[c] = int4_zero_point_sub2(
                int4_block_zero_point(zero_points, col_base + c, block, zp_row_bytes));
        }

        // Decode each 8-element activation sub-word once, reuse across columns.
        // Sub-word order 0..3 preserves the ascending-K accumulation of the
        // single-column kernel (w.x, w.y, w.z, w.w).
        float a8[8];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            decode_activation8(activation + depth + s * 8, a8);
#pragma unroll
            for (int c = 0; c < WIDE_NC; ++c) {
                if (!valid[c]) {
                    continue;
                }
                const unsigned int word =
                    (s == 0) ? w[c].x : (s == 1) ? w[c].y : (s == 2) ? w[c].z : w[c].w;
                values[c] += scale[c] * dot_int4x8_f16_sub_act(word, a8, sub2[c]);
            }
        }
        depth += warp_stride;
    }

    // Partial trailing chunk (`depth < k < depth + 32`): replicate the
    // single-column tail arithmetic per column.
    if (depth < k) {
        const int block = depth / block_size;
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            if (!valid[c]) {
                continue;
            }
            float scale;
            if (scales_fp16) {
                scale = __half2float(reinterpret_cast<const __half*>(scales)[col_kb[c] + block]);
            } else {
                scale = reinterpret_cast<const float*>(scales)[col_kb[c] + block];
            }
            const int zero_point =
                int4_block_zero_point(zero_points, col_base + c, block, zp_row_bytes);
            const unsigned int sub2 = int4_zero_point_sub2(zero_point);
            const long blob_base = (col_kb[c] + block) * (long)blob_size;
            for (int off = 0; off < 32 && depth + off < k; off += 8) {
                const int d = depth + off;
                const int within = d - block * block_size;
                const int valid_n = min(8, k - d);
                const unsigned int packed_word =
                    *reinterpret_cast<const unsigned int*>(packed + blob_base + (within >> 1));
                if (valid_n == 8) {
                    values[c] += scale * dot_int4x8_f16_sub(packed_word, activation + d, sub2);
                } else {
                    float partial = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid_n) {
                            const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                            partial += (float)q * __half2float(activation[d + i]);
                        }
                    }
                    values[c] += partial * scale;
                }
            }
        }
    }
}

// Interleaved + biased (symmetric-only) sibling of
// `gemv_int4_wide_lane_dot_multicol`. Consumes offline-interleaved weights and
// applies the fixed symmetric `-8` bias inside the LOP3 converter, so there is
// no per-block zero point and no `prmt.b32` activation reorder. Every column's
// fp32 accumulation order is unchanged (ascending element order, sub-word order
// w.x, w.y, w.z, w.w), so each column result is BIT-IDENTICAL to
// `gemv_int4_wide_lane_dot_multicol` on symmetric weights.
__device__ __forceinline__ void gemv_int4_wide_lane_dot_multicol_interleaved(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const int k,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int scales_fp16,
    const long col_base,
    const int n,
    const int depth0,
    const int warp_stride,
    float* __restrict__ values /* [WIDE_NC] */)
{
    bool valid[WIDE_NC];
    long col_kb[WIDE_NC];
#pragma unroll
    for (int c = 0; c < WIDE_NC; ++c) {
        values[c] = 0.0f;
        const long column = col_base + c;
        valid[c] = (column < n);
        col_kb[c] = column * (long)k_blocks;
    }

    int depth = depth0;
    while (depth + 32 <= k) {
        const int block = depth / block_size;
        const int within = depth - block * block_size;

        uint4 w[WIDE_NC];
        float scale[WIDE_NC];
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            if (!valid[c]) {
                continue;
            }
            const long base = (col_kb[c] + block) * (long)blob_size;
            w[c] = *reinterpret_cast<const uint4*>(packed + base + (within >> 1));
            if (scales_fp16) {
                scale[c] = __half2float(reinterpret_cast<const __half*>(scales)[col_kb[c] + block]);
            } else {
                scale[c] = reinterpret_cast<const float*>(scales)[col_kb[c] + block];
            }
        }

        float a8[8];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            decode_activation8_natural(activation + depth + s * 8, a8);
#pragma unroll
            for (int c = 0; c < WIDE_NC; ++c) {
                if (!valid[c]) {
                    continue;
                }
                const unsigned int word =
                    (s == 0) ? w[c].x : (s == 1) ? w[c].y : (s == 2) ? w[c].z : w[c].w;
                values[c] += scale[c] * dot_int4x8_f16_interleaved_act(word, a8);
            }
        }
        depth += warp_stride;
    }

    if (depth < k) {
        const int block = depth / block_size;
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            if (!valid[c]) {
                continue;
            }
            float scale;
            if (scales_fp16) {
                scale = __half2float(reinterpret_cast<const __half*>(scales)[col_kb[c] + block]);
            } else {
                scale = reinterpret_cast<const float*>(scales)[col_kb[c] + block];
            }
            const long blob_base = (col_kb[c] + block) * (long)blob_size;
            for (int off = 0; off < 32 && depth + off < k; off += 8) {
                const int d = depth + off;
                const int within = d - block * block_size;
                const int valid_n = min(8, k - d);
                const unsigned int packed_word =
                    *reinterpret_cast<const unsigned int*>(packed + blob_base + (within >> 1));
                if (valid_n == 8) {
                    values[c] += scale * dot_int4x8_f16_interleaved(packed_word, activation + d);
                } else {
                    float partial = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid_n) {
                            const int q = interleaved_nibble(packed_word, i) - 8;
                            partial += (float)q * __half2float(activation[d + i]);
                        }
                    }
                    values[c] += partial * scale;
                }
            }
        }
    }
}

// Decode eight contiguous fp16 activations into four `__half2` lanes laid out in
// the SAME element pairing that `int4x8_to_half2x4_sub` produces for the weights
// (`ah[i] = (a_i, a_{i+4})`), so `__hfma2(q[i], ah[i], acc)` multiplies matching
// (weight, activation) pairs. This is the fp16 sibling of `decode_activation8`:
// it stops at the permute step and keeps the halves packed (NO fp16->fp32
// conversion), because the fp16-mixed kernel consumes them directly in half2
// fused multiply-adds.
__device__ __forceinline__ void decode_activation8_h2(
    const __half* __restrict__ activation,
    __half2* __restrict__ ah /* [4] */)
{
    const uint4 av = *reinterpret_cast<const uint4*>(activation);
    constexpr unsigned int low_halves = 0x5410;
    constexpr unsigned int high_halves = 0x7632;
    uint4 permuted;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.x) : "r"(av.x), "r"(av.z), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.y) : "r"(av.x), "r"(av.z), "r"(high_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.z) : "r"(av.y), "r"(av.w), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.w) : "r"(av.y), "r"(av.w), "r"(high_halves));
    ah[0] = *reinterpret_cast<const __half2*>(&permuted.x);
    ah[1] = *reinterpret_cast<const __half2*>(&permuted.y);
    ah[2] = *reinterpret_cast<const __half2*>(&permuted.z);
    ah[3] = *reinterpret_cast<const __half2*>(&permuted.w);
}

// ---------------------------------------------------------------------------
// fp16 mixed-precision column register-blocked wide GEMV.
//
// Identical column-register-blocking + wide-load structure as
// `gemv_int4_wide_lane_dot_multicol`, but the inner multiply-accumulate runs in
// fp16 `__hfma2` (two fused MACs per instruction) instead of fp32 FFMA. This is
// the ONLY way to actually cut the dequant/MAC ALU that limits the multicol
// kernel (fp32 FFMA is already one fused op, so a "fp16-multiply then
// fp32-add-every-product" scheme is strictly MORE ops and cannot win). ORT's
// `MatMulFloatInt4Kernel` is fp16 for exactly this reason, so this is the
// fp16-vs-fp16 equal-conditions path.
//
// PRECISION CONTRACT (matches ORT's `MatMulFloat4BitsKernelM1` exactly): the
// per-lane K reduction runs entirely in fp16 __half2 accumulators — each 32-term
// chunk is summed in fp16, its per-block scale is folded in with __hfma2, and the
// result accumulates into a per-column fp16 running `total`. fp32 is used ONLY in
// the final cross-lane `warp_sum` (the 5-step shuffle). This is safe because the
// fp16 accumulation is a WIDE, SHALLOW tree: with 32 lanes striding K by 32, each
// lane folds only ~K/1024 chunks (≈4 for K=4096, ≈13 for K=13696), and inside a
// chunk the __half2 holds two 16-deep lanes — a total fp16 depth of tens, not
// thousands, so mantissa loss is negligible. (A NAIVE deep single-accumulator
// fp16 sum of all K *does* lose mantissa and flip tokens — that is the trap this
// wide-tree layout avoids, and why ORT accumulates in fp16 throughout.) Because
// the arithmetic mirrors ORT's, the f64-oracle error lands in ORT's own error
// class; it is NOT byte-identical to the fp32 path, so it ships gated on accuracy
// (error <= ORT vs the f64 oracle), not on bit-identity.
__device__ __forceinline__ void gemv_int4_fp16_lane_dot_multicol(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const int k,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const long col_base,
    const int n,
    const int depth0,
    const int warp_stride,
    float* __restrict__ values /* [WIDE_NC] */)
{
    bool valid[WIDE_NC];
    long col_kb[WIDE_NC];
    // Per-column fp16 running total across the lane's chunks (ORT-style): the
    // per-block scale is folded into this __half2 accumulator with __hfma2, so
    // the entire per-lane K reduction stays in fp16 and only the final
    // cross-lane `warp_sum` runs in fp32. Matching ORT's arithmetic exactly puts
    // this kernel in the same error class as ORT's own int4 M=1 kernel. A single
    // fp16 accumulator is a *wide, shallow* reduction tree — each lane folds only
    // a handful of chunks (K / (32 lanes * 32) ≈ 4 for K=4096, ≈13 for K=13696),
    // so almost no mantissa is lost (this is why full-fp16 accumulate is
    // production-safe here, unlike a naive deep single-accumulator fp16 sum).
    __half2 total[WIDE_NC];
#pragma unroll
    for (int c = 0; c < WIDE_NC; ++c) {
        values[c] = 0.0f;
        total[c] = __float2half2_rn(0.0f);
        const long column = col_base + c;
        valid[c] = (column < n);
        col_kb[c] = column * (long)k_blocks;
    }

    int depth = depth0;
    while (depth + 32 <= k) {
        const int block = depth / block_size;
        const int within = depth - block * block_size;

        uint4 w[WIDE_NC];
        __half2 scale2[WIDE_NC];
        unsigned int sub2[WIDE_NC];
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            if (!valid[c]) {
                continue;
            }
            const long base = (col_kb[c] + block) * (long)blob_size;
            w[c] = *reinterpret_cast<const uint4*>(packed + base + (within >> 1));
            // Splat the per-block scale into both half lanes so it can be folded
            // into the fp16 accumulator with a single __hfma2.
            if (scales_fp16) {
                scale2[c] = __half2half2(reinterpret_cast<const __half*>(scales)[col_kb[c] + block]);
            } else {
                scale2[c] = __float2half2_rn(reinterpret_cast<const float*>(scales)[col_kb[c] + block]);
            }
            sub2[c] = int4_zero_point_sub2(
                int4_block_zero_point(zero_points, col_base + c, block, zp_row_bytes));
        }

        // fp16 accumulators for this chunk's 32 (unscaled) products, one per col.
        __half2 acc[WIDE_NC];
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            acc[c] = __float2half2_rn(0.0f);
        }

        // Decode each 8-element activation sub-word to half2 once, reuse across
        // columns (the multicol L1-traffic win), and fold it into every column's
        // fp16 accumulator with __hfma2 (2 fused MACs/instruction).
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            __half2 ah[4];
            decode_activation8_h2(activation + depth + s * 8, ah);
#pragma unroll
            for (int c = 0; c < WIDE_NC; ++c) {
                if (!valid[c]) {
                    continue;
                }
                const unsigned int word =
                    (s == 0) ? w[c].x : (s == 1) ? w[c].y : (s == 2) ? w[c].z : w[c].w;
                __half2 q[4];
                int4x8_to_half2x4_sub(word, q, sub2[c]);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[c] = __hfma2(q[i], ah[i], acc[c]);
                }
            }
        }

        // Scale this chunk's fp16 partial by the block scale and fold it into the
        // fp16 running total (one __hfma2 per column) — fp32 is deferred to the
        // final cross-lane reduction.
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            if (!valid[c]) {
                continue;
            }
            total[c] = __hfma2(acc[c], scale2[c], total[c]);
        }
        depth += warp_stride;
    }

    // Collapse each column's fp16 running total to fp32 (the two half lanes are
    // the two halves of the dot product). The tail below adds into this in fp32.
#pragma unroll
    for (int c = 0; c < WIDE_NC; ++c) {
        values[c] = __low2float(total[c]) + __high2float(total[c]);
    }

    // Partial trailing chunk: compute in fp32 exactly like the fp32 multicol tail
    // so the K-tail stays precise (negligible perf, avoids fp16 edge cases).
    if (depth < k) {
        const int block = depth / block_size;
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            if (!valid[c]) {
                continue;
            }
            float scale;
            if (scales_fp16) {
                scale = __half2float(reinterpret_cast<const __half*>(scales)[col_kb[c] + block]);
            } else {
                scale = reinterpret_cast<const float*>(scales)[col_kb[c] + block];
            }
            const int zero_point =
                int4_block_zero_point(zero_points, col_base + c, block, zp_row_bytes);
            const unsigned int sub2 = int4_zero_point_sub2(zero_point);
            const long blob_base = (col_kb[c] + block) * (long)blob_size;
            for (int off = 0; off < 32 && depth + off < k; off += 8) {
                const int d = depth + off;
                const int within = d - block * block_size;
                const int valid_n = min(8, k - d);
                const unsigned int packed_word =
                    *reinterpret_cast<const unsigned int*>(packed + blob_base + (within >> 1));
                if (valid_n == 8) {
                    values[c] += scale * dot_int4x8_f16_sub(packed_word, activation + d, sub2);
                } else {
                    float partial = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid_n) {
                            const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                            partial += (float)q * __half2float(activation[d + i]);
                        }
                    }
                    values[c] += partial * scale;
                }
            }
        }
    }
}

__device__ __forceinline__ uint4 permute_activation_f16x8(
    const __half* __restrict__ activation)
{
    const uint4 a = *reinterpret_cast<const uint4*>(activation);
    constexpr unsigned int low_halves = 0x5410;
    constexpr unsigned int high_halves = 0x7632;
    uint4 permuted;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.x) : "r"(a.x), "r"(a.z), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.y) : "r"(a.x), "r"(a.z), "r"(high_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.z) : "r"(a.y), "r"(a.w), "r"(low_halves));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n"
                 : "=r"(permuted.w) : "r"(a.y), "r"(a.w), "r"(high_halves));
    return permuted;
}

__device__ __forceinline__ void accumulate_int4x8_f16_permuted(
    const unsigned int packed,
    const uint4& activation,
    const __half scale,
    __half2& sum0,
    __half2& sum1,
    __half2& sum2,
    __half2& sum3)
{
    __half2 q[4];
    int4x8_to_half2x4(packed, q);
    const __half2 scale2 = __halves2half2(scale, scale);
    sum0 = __hfma2(
        __hmul2(q[0], scale2),
        *reinterpret_cast<const __half2*>(&activation.x),
        sum0);
    sum1 = __hfma2(
        __hmul2(q[1], scale2),
        *reinterpret_cast<const __half2*>(&activation.y),
        sum1);
    sum2 = __hfma2(
        __hmul2(q[2], scale2),
        *reinterpret_cast<const __half2*>(&activation.z),
        sum2);
    sum3 = __hfma2(
        __hmul2(q[3], scale2),
        *reinterpret_cast<const __half2*>(&activation.w),
        sum3);
}

// Zero-point-aware [`accumulate_int4x8_f16_permuted`]: `sub2` centers each
// nibble by the block zero point (fp16 8.0 for symmetric weights, giving a
// byte-identical result). Used by the paired gate/up kernels, which permute the
// shared activation once and dequant each projection with its own zero point.
__device__ __forceinline__ void accumulate_int4x8_f16_permuted_zp(
    const unsigned int packed,
    const uint4& activation,
    const __half scale,
    const unsigned int sub2,
    __half2& sum0,
    __half2& sum1,
    __half2& sum2,
    __half2& sum3)
{
    __half2 q[4];
    int4x8_to_half2x4_sub(packed, q, sub2);
    const __half2 scale2 = __halves2half2(scale, scale);
    sum0 = __hfma2(__hmul2(q[0], scale2),
                   *reinterpret_cast<const __half2*>(&activation.x), sum0);
    sum1 = __hfma2(__hmul2(q[1], scale2),
                   *reinterpret_cast<const __half2*>(&activation.y), sum1);
    sum2 = __hfma2(__hmul2(q[2], scale2),
                   *reinterpret_cast<const __half2*>(&activation.z), sum2);
    sum3 = __hfma2(__hmul2(q[3], scale2),
                   *reinterpret_cast<const __half2*>(&activation.w), sum3);
}

// Fused-symmetric [`accumulate_int4x8_f16_permuted_zp`]: identical multiply-add
// (same permuted activation, same `q * scale` fp16 FMA order into `sum0..3`) but
// dequants with [`int4x8_to_half2x4_sym8`], which folds the `- 8` symmetric zero
// point into the bias constants. The `q[i]` values are byte-identical to the
// `sub2 == 8` path, so the accumulation is bit-for-bit the same while issuing
// four fewer `f16x2` ops per word — the instruction-count win for the
// issue-bound paired gate/up decode GEMV.
__device__ __forceinline__ void accumulate_int4x8_f16_permuted_sym8(
    const unsigned int packed,
    const uint4& activation,
    const __half scale,
    __half2& sum0,
    __half2& sum1,
    __half2& sum2,
    __half2& sum3)
{
    __half2 q[4];
    int4x8_to_half2x4_sym8(packed, q);
    const __half2 scale2 = __halves2half2(scale, scale);
    sum0 = __hfma2(__hmul2(q[0], scale2),
                   *reinterpret_cast<const __half2*>(&activation.x), sum0);
    sum1 = __hfma2(__hmul2(q[1], scale2),
                   *reinterpret_cast<const __half2*>(&activation.y), sum1);
    sum2 = __hfma2(__hmul2(q[2], scale2),
                   *reinterpret_cast<const __half2*>(&activation.z), sum2);
    sum3 = __hfma2(__hmul2(q[3], scale2),
                   *reinterpret_cast<const __half2*>(&activation.w), sum3);
}

__device__ __forceinline__ void accumulate_int4x8_dot_f16(
    const unsigned int packed,
    const uint4& activation,
    const __half2 scale2,
    __half2& sum)
{
    __half2 q[4];
    int4x8_to_half2x4(packed, q);
    sum = __hfma2(
        __hmul2(q[0], scale2),
        *reinterpret_cast<const __half2*>(&activation.x),
        sum);
    sum = __hfma2(
        __hmul2(q[1], scale2),
        *reinterpret_cast<const __half2*>(&activation.y),
        sum);
    sum = __hfma2(
        __hmul2(q[2], scale2),
        *reinterpret_cast<const __half2*>(&activation.z),
        sum);
    sum = __hfma2(
        __hmul2(q[3], scale2),
        *reinterpret_cast<const __half2*>(&activation.w),
        sum);
}

__device__ __forceinline__ float dot_int4x32_f16_permuted_scaled(
    const uint4& packed,
    const uint4& activation0,
    const uint4& activation1,
    const uint4& activation2,
    const uint4& activation3,
    const __half scale)
{
    const __half2 scale2 = __halves2half2(scale, scale);
    __half2 sum0 = __float2half2_rn(0.0f);
    __half2 sum1 = __float2half2_rn(0.0f);
    __half2 sum2 = __float2half2_rn(0.0f);
    __half2 sum3 = __float2half2_rn(0.0f);
    accumulate_int4x8_dot_f16(packed.x, activation0, scale2, sum0);
    accumulate_int4x8_dot_f16(packed.y, activation1, scale2, sum1);
    accumulate_int4x8_dot_f16(packed.z, activation2, scale2, sum2);
    accumulate_int4x8_dot_f16(packed.w, activation3, scale2, sum3);
    const float2 value0 = __half22float2(sum0);
    const float2 value1 = __half22float2(sum1);
    const float2 value2 = __half22float2(sum2);
    const float2 value3 = __half22float2(sum3);
    float value = value0.x;
    value += value1.x;
    value += value2.x;
    value += value3.x;
    value += value0.y;
    value += value1.y;
    value += value2.y;
    value += value3.y;
    return value;
}

__device__ __forceinline__ void accumulate_int4x8_f16(
    const unsigned int packed,
    const __half* __restrict__ activation,
    const __half scale,
    __half2& sum0,
    __half2& sum1,
    __half2& sum2,
    __half2& sum3)
{
    const uint4 permuted = permute_activation_f16x8(activation);
    accumulate_int4x8_f16_permuted(
        packed, permuted, scale, sum0, sum1, sum2, sum3);
}

// Zero-point-aware variant of [`accumulate_int4x8_f16`]: `sub2` is the fp16x2
// subtrahend for this block (the packed zero point, or fp16 8.0 for symmetric
// weights). With the symmetric default this is byte-identical to the plain
// accumulate, so callers can route both symmetric and asymmetric weights here.
__device__ __forceinline__ void accumulate_int4x8_f16_zp(
    const unsigned int packed,
    const __half* __restrict__ activation,
    const __half scale,
    const unsigned int sub2,
    __half2& sum0,
    __half2& sum1,
    __half2& sum2,
    __half2& sum3)
{
    const uint4 permuted = permute_activation_f16x8(activation);
    __half2 q[4];
    int4x8_to_half2x4_sub(packed, q, sub2);
    const __half2 scale2 = __halves2half2(scale, scale);
    sum0 = __hfma2(__hmul2(q[0], scale2),
                   *reinterpret_cast<const __half2*>(&permuted.x), sum0);
    sum1 = __hfma2(__hmul2(q[1], scale2),
                   *reinterpret_cast<const __half2*>(&permuted.y), sum1);
    sum2 = __hfma2(__hmul2(q[2], scale2),
                   *reinterpret_cast<const __half2*>(&permuted.z), sum2);
    sum3 = __hfma2(__hmul2(q[3], scale2),
                   *reinterpret_cast<const __half2*>(&permuted.w), sum3);
}

// Debias-then-scale int4 dequant (fold-scale). Recovers the exact integer code
// in fp16 (the same 1024/64 lop3-bias removal as `int4x8_to_half2x4_sub`, which
// is fp16-exact for codes 0..15 so there is no catastrophic cancellation), then
// folds the per-block scale AND zero point into ONE `fma` per pair:
// `(code - zp) * scale = fma(code, scale2, neg_zp_scale2)`. This lets the MAC
// drop its separate `__hmul2(q, scale)`, removing 4 fp16x2 multiplies per 8
// weights on the ALU-co-bound (measured 65% pipe) M=1 decode GEMV.
__device__ __forceinline__ void int4x8_to_half2x4_scaledsub(
    const unsigned int packed,
    __half2* values,
    const unsigned int scale2,
    const unsigned int neg_zp_scale2)
{
    unsigned int* h = reinterpret_cast<unsigned int*>(values);
    constexpr unsigned int bottom_mask = 0x000f000f;
    constexpr unsigned int top_mask = 0x00f000f0;
    constexpr unsigned int fp16_magic = 0x64006400;
    constexpr unsigned int lop3_lut = (0xf0 & 0xcc) | 0xaa;
    const unsigned int top = packed >> 8;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[0])
                 : "r"(packed), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[1])
                 : "r"(packed), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[2])
                 : "r"(top), "n"(bottom_mask), "n"(fp16_magic), "n"(lop3_lut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[3])
                 : "r"(top), "n"(top_mask), "n"(fp16_magic), "n"(lop3_lut));
    constexpr unsigned int fp16_1024 = 0x64006400;
    constexpr unsigned int fp16_one_sixteenth = 0x2c002c00;
    constexpr unsigned int fp16_neg64 = 0xd400d400;
    // Debias to the exact integer code first (fp16-exact for 0..15).
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[0]) : "r"(h[0]), "r"(fp16_1024));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[1])
                 : "r"(h[1]), "r"(fp16_one_sixteenth), "r"(fp16_neg64));
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(h[2]) : "r"(h[2]), "r"(fp16_1024));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(h[3])
                 : "r"(h[3]), "r"(fp16_one_sixteenth), "r"(fp16_neg64));
    // Fold `(code - zp) * scale` into one fma per pair (replaces the standalone
    // zp-subtract here and the `__hmul2(q, scale)` in the MAC).
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                     : "=r"(h[i]) : "r"(h[i]), "r"(scale2), "r"(neg_zp_scale2));
    }
}

// per CTA. Four adjacent lanes split each block-32 weight blob into aligned
// uint32 loads, so every warp issues contiguous 128-byte packed-weight
// transactions. Each lane also reads eight activations with one uint4 load.
// Register-only nibble conversion and four-lane shuffle reduction reconstruct
// each block dot product before applying its scale.
extern "C" __global__ void matmul_nbits_gemv_f16(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    float value = 0.0f;
    if (column < n) {
        const int quarter = lane & 3;
        for (int block_base = 0; block_base < k_blocks; block_base += 8) {
            const int block = block_base + (lane >> 2);
            float block_partial = 0.0f;
            if (block < k_blocks) {
                const int depth = block * block_size + quarter * 8;
                const long packed_start =
                    ((long)column * k_blocks + block) * blob_size + quarter * 4;
                const unsigned int packed_word =
                    *reinterpret_cast<const unsigned int*>(packed + packed_start);
                int zero_point = 8;
                if (zero_points) {
                    const unsigned char zp =
                        zero_points[(long)column * zp_row_bytes + block / 2];
                    zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
                }
                if (depth + 8 <= k) {
                    if (zero_points) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int q =
                                (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                            block_partial +=
                                (float)q * __half2float(activation[depth + i]);
                        }
                    } else {
                        block_partial = dot_int4x8_f16(packed_word, activation + depth);
                    }
                } else if (depth < k) {
                    const int valid = min(8, k - depth);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int q =
                                (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                            block_partial +=
                                (float)q * __half2float(activation[depth + i]);
                        }
                    }
                }
            }
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 2, 4);
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 1, 4);
            if (quarter == 0 && block < k_blocks) {
                float scale;
                if (scales_fp16) {
                    scale = __half2float(
                        reinterpret_cast<const __half*>(scales)[(long)column * k_blocks + block]);
                } else {
                    scale =
                        reinterpret_cast<const float*>(scales)[(long)column * k_blocks + block];
                }
                value += block_partial * scale;
            }
        }
    }

    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

template <bool HasZp>
__device__ __forceinline__ void matmul_nbits_gemv_f16_scales_f16_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)block_size;
    (void)scales_fp16;
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column_base = (int)blockIdx.x * columns_per_block;
    const int column = column_base + warp;

    __half2 sum0 = __float2half2_rn(0.0f);
    __half2 sum1 = __float2half2_rn(0.0f);
    __half2 sum2 = __float2half2_rn(0.0f);
    __half2 sum3 = __float2half2_rn(0.0f);
    float tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        const __half* activation_ptr = activation + lane_depth;
        const unsigned char* packed_ptr =
            packed + (long)column * k_blocks * blob_size + lane * 4;
        const __half* scale_ptr =
            scales + (long)column * k_blocks + (lane >> 2);
        int depth_base = 0;
        for (; depth_base + lane_depth + 8 <= k; depth_base += 256) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            // block == depth/32; each lane's 8 nibbles all sit in one block.
            const int block = (depth_base >> 5) + (lane >> 2);
            const unsigned int sub2 =
                block_sub2<HasZp>(zero_points, column, block, zp_row_bytes);
            accumulate_int4x8_f16_zp(
                packed_word,
                activation_ptr,
                *scale_ptr,
                sub2,
                sum0,
                sum1,
                sum2,
                sum3);
            activation_ptr += 256;
            packed_ptr += 128;
            scale_ptr += 8;
        }
        const int tail_depth = depth_base + lane_depth;
        if (tail_depth < k) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            const float scale = __half2float(*scale_ptr);
            const int tail_block = (depth_base >> 5) + (lane >> 2);
            const int zero_point =
                block_zp<HasZp>(zero_points, column, tail_block, zp_row_bytes);
            const int valid = min(8, k - tail_depth);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                    tail += (float)q * __half2float(activation_ptr[i]) * scale;
                }
            }
        }
    }
    const float2 value04 = __half22float2(sum0);
    const float2 value15 = __half22float2(sum1);
    const float2 value26 = __half22float2(sum2);
    const float2 value37 = __half22float2(sum3);
    float value = tail + value04.x;
    value += value15.x;
    value += value26.x;
    value += value37.x;
    value += value04.y;
    value += value15.y;
    value += value26.y;
    value += value37.y;
    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    matmul_nbits_gemv_f16_scales_f16_tpl<false>(activation, packed, scales_raw, zero_points, bias, output, k, n, block_size, k_blocks, blob_size, zp_row_bytes, scales_fp16, bias_post_round);
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    matmul_nbits_gemv_f16_scales_f16_tpl<true>(activation, packed, scales_raw, zero_points, bias, output, k, n, block_size, k_blocks, blob_size, zp_row_bytes, scales_fp16, bias_post_round);
}


// Prefetch-pipelined sibling of `matmul_nbits_gemv_f16_scales_f16_tpl`.
// The single-warp scales-fp16 GEMV is Long-Scoreboard bound: ncu on qwen2.5-14b
// q/o (N=5120) shows ~8.9 active warps/scheduler but only ~0.97 eligible, with
// ~75% of warp cycles stalled waiting for the one in-flight 32-bit weight load.
// The math is not the bottleneck (SM ~41%, DRAM ~24% of an H200's 4.8 TB/s).
//
// This variant keeps the EXACT same lane->nibble mapping (lane owns 8 contiguous
// nibbles at stride 256), the same fp16 `accumulate_int4x8_f16_zp` calls, and the
// same accumulation order — so its output is BYTE-IDENTICAL to the single-warp
// kernel. The only change is memory-level parallelism: a depth-PF register shift
// register holds PF prefetched weight words so PF independent global loads are in
// flight per lane, hiding the ~13-cycle load latency instead of stalling on it.
// Register-resident (manual rotation, no dynamic-indexed array -> no local spill).
template <bool HasZp>
__device__ __forceinline__ void matmul_nbits_gemv_f16_scales_f16_pipe_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)block_size;
    (void)scales_fp16;
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column_base = (int)blockIdx.x * columns_per_block;
    const int column = column_base + warp;

    __half2 sum0 = __float2half2_rn(0.0f);
    __half2 sum1 = __float2half2_rn(0.0f);
    __half2 sum2 = __float2half2_rn(0.0f);
    __half2 sum3 = __float2half2_rn(0.0f);
    float tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        const __half* activation_ptr = activation + lane_depth;
        const unsigned char* packed_ptr =
            packed + (long)column * k_blocks * blob_size + lane * 4;
        const __half* scale_ptr =
            scales + (long)column * k_blocks + (lane >> 2);
        // Number of full 8-nibble steps this lane walks (step s is valid iff
        // s * 256 + lane_depth + 8 <= k), identical to the scalar loop bound.
        const int nfull = (lane_depth + 8 <= k) ? ((k - lane_depth - 8) / 256 + 1) : 0;

        // Load the k-th step's weight word (step stride is 128 packed bytes).
        auto load_step = [&](int s) -> unsigned int {
            return *reinterpret_cast<const unsigned int*>(packed_ptr + (long)s * 128);
        };
        // Prime the shift register with the first PF steps' words (independent
        // loads -> they pipeline). Out-of-range slots are zero (never consumed).
        // The array is `#pragma unroll`-rotated with a compile-time `PF`, so it
        // stays fully in registers (no dynamic indexing -> no local spill).
        constexpr int PF = 2;  // prefetch depth (weight words in flight per lane)
        unsigned int wbuf[PF];
#pragma unroll
        for (int s = 0; s < PF; ++s) {
            wbuf[s] = (s < nfull) ? load_step(s) : 0u;
        }

        int depth_base = 0;
        for (int i = 0; i < nfull; ++i, depth_base += 256) {
            const unsigned int packed_word = wbuf[0];
            // Rotate the shift register and issue the load PF steps ahead BEFORE
            // consuming the current word, so PF loads stay in flight.
            const int pf = i + PF;
#pragma unroll
            for (int s = 0; s < PF - 1; ++s) {
                wbuf[s] = wbuf[s + 1];
            }
            wbuf[PF - 1] = (pf < nfull) ? load_step(pf) : 0u;

            const int block = (depth_base >> 5) + (lane >> 2);
            const unsigned int sub2 =
                block_sub2<HasZp>(zero_points, column, block, zp_row_bytes);
            accumulate_int4x8_f16_zp(
                packed_word, activation_ptr, *scale_ptr, sub2, sum0, sum1, sum2, sum3);
            activation_ptr += 256;
            scale_ptr += 8;
        }
        const int tail_depth = depth_base + lane_depth;
        if (tail_depth < k) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr + (long)nfull * 128);
            const float scale = __half2float(*scale_ptr);
            const int tail_block = (depth_base >> 5) + (lane >> 2);
            const int zero_point =
                block_zp<HasZp>(zero_points, column, tail_block, zp_row_bytes);
            const int valid = min(8, k - tail_depth);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                    tail += (float)q * __half2float(activation_ptr[i]) * scale;
                }
            }
        }
    }
    const float2 value04 = __half22float2(sum0);
    const float2 value15 = __half22float2(sum1);
    const float2 value26 = __half22float2(sum2);
    const float2 value37 = __half22float2(sum3);
    float value = tail + value04.x;
    value += value15.x;
    value += value26.x;
    value += value37.x;
    value += value04.y;
    value += value15.y;
    value += value26.y;
    value += value37.y;
    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_pipe(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    matmul_nbits_gemv_f16_scales_f16_pipe_tpl<false>(activation, packed, scales_raw, zero_points, bias, output, k, n, block_size, k_blocks, blob_size, zp_row_bytes, scales_fp16, bias_post_round);
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_zp_pipe(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    matmul_nbits_gemv_f16_scales_f16_pipe_tpl<true>(activation, packed, scales_raw, zero_points, bias, output, k, n, block_size, k_blocks, blob_size, zp_row_bytes, scales_fp16, bias_post_round);
}
// each reducing a strided subset of the 256-wide K steps, then summing their
// fp32 partials through shared memory. The launch grid is K_SPLIT x larger than
// the single-warp `_zp` kernel, which fills the SMs on this grid-starved,
// latency-bound decode GEMV. The fp32 partial sum is a new block-sum
// association, so results are near-equal (not byte-identical) to the
// single-warp kernel.
template <bool HasZp>
__device__ __forceinline__ void matmul_nbits_gemv_f16_scales_f16_splitk_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    void* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int out_bf16)
{
    (void)block_size;
    (void)scales_fp16;
    constexpr int K_SPLIT = 2;
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int cols_per_block = warps_per_block / K_SPLIT;
    const int col_local = warp / K_SPLIT;
    const int ks = warp % K_SPLIT;
    const int column = (int)blockIdx.x * cols_per_block + col_local;

    __shared__ float partials[8][K_SPLIT];

    __half2 sum0 = __float2half2_rn(0.0f);
    __half2 sum1 = __float2half2_rn(0.0f);
    __half2 sum2 = __float2half2_rn(0.0f);
    __half2 sum3 = __float2half2_rn(0.0f);
    float tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        int depth_base = ks * 256;
        const __half* activation_ptr = activation + depth_base + lane_depth;
        const unsigned char* packed_ptr =
            packed + (long)column * k_blocks * blob_size +
            (long)(depth_base >> 5) * blob_size + lane * 4;
        const __half* scale_ptr =
            scales + (long)column * k_blocks + (depth_base >> 5) + (lane >> 2);
        for (; depth_base < k; depth_base += K_SPLIT * 256) {
            const int depth = depth_base + lane_depth;
            if (depth >= k) {
                activation_ptr += K_SPLIT * 256;
                packed_ptr += K_SPLIT * 128;
                scale_ptr += K_SPLIT * 8;
                continue;
            }
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            const int block = (depth_base >> 5) + (lane >> 2);
            const unsigned int sub2 =
                block_sub2<HasZp>(zero_points, column, block, zp_row_bytes);
            if (depth + 8 <= k) {
                accumulate_int4x8_f16_zp(
                    packed_word, activation_ptr, *scale_ptr, sub2, sum0, sum1, sum2, sum3);
            } else {
                const float scale = __half2float(*scale_ptr);
                const int zero_point =
                    block_zp<HasZp>(zero_points, column, block, zp_row_bytes);
                const int valid = k - depth;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (i < valid) {
                        const int q =
                            (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                        tail += (float)q * __half2float(activation_ptr[i]) * scale;
                    }
                }
            }
            activation_ptr += K_SPLIT * 256;
            packed_ptr += K_SPLIT * 128;
            scale_ptr += K_SPLIT * 8;
        }
    }
    const float2 value04 = __half22float2(sum0);
    const float2 value15 = __half22float2(sum1);
    const float2 value26 = __half22float2(sum2);
    const float2 value37 = __half22float2(sum3);
    float value = tail + value04.x;
    value += value15.x;
    value += value26.x;
    value += value37.x;
    value += value04.y;
    value += value15.y;
    value += value26.y;
    value += value37.y;
    value = warp_sum(value);
    if (lane == 0) {
        partials[col_local][ks] = (column < n) ? value : 0.0f;
    }
    __syncthreads();
    if (ks == 0 && lane == 0 && column < n) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < K_SPLIT; ++s) {
            acc += partials[col_local][s];
        }
        matmul_nbits_store_narrowed(
            output, column, fold_bias_f16(acc, bias, column, bias_post_round),
            out_bf16);
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_splitk(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    void* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int out_bf16)
{
    matmul_nbits_gemv_f16_scales_f16_splitk_tpl<false>(
        activation, packed, scales_raw, zero_points, bias, output, k, n,
        block_size, k_blocks, blob_size, zp_row_bytes, scales_fp16,
        bias_post_round, out_bf16);
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_zp_splitk(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    void* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int out_bf16)
{
    matmul_nbits_gemv_f16_scales_f16_splitk_tpl<true>(
        activation, packed, scales_raw, zero_points, bias, output, k, n,
        block_size, k_blocks, blob_size, zp_row_bytes, scales_fp16,
        bias_post_round, out_bf16);
}

// Half4 view matching `skip_rmsnorm_f16_warp_half4` so the fused prologue below
// reduces the activation with the exact same chunking and rounding.
union MatMulNBitsSkipHalf4 {
    unsigned long long raw;
    __half2 pair[2];
};

// Scalar RMS-norm gamma load matching `skip_rmsnorm_f16_warp_half4`: gamma is
// only ever a final multiplicand (never part of the fp32 variance
// accumulation), so an fp32 gamma is read at full precision while an fp16 gamma
// keeps the half round-trip. This lets decoders that export gamma in fp32 (e.g.
// Phi-4-mini) take the fused RMS-norm-prologue GEMV path bit-identically to the
// standalone norm + GEMV pair.
__device__ __forceinline__ float load_rmsnorm_gamma(
    const void* __restrict__ gamma,
    const int gamma_is_half,
    const int index)
{
    return gamma_is_half
        ? __half2float(reinterpret_cast<const __half*>(gamma)[index])
        : reinterpret_cast<const float*>(gamma)[index];
}

// General fp16/fp16-scales GEMV with a fused RMS-normalization prologue. The
// preceding GEMV's residual epilogue already produced the byte-identical
// residual sum that `SkipSimplifiedLayerNormalization` would emit as its
// residual output, so this kernel only has to (1) reduce that sum exactly as
// `skip_rmsnorm_f16_warp_half4` does, (2) write the normalized activation into
// shared memory with the same rounding, and (3) run the standard `scales_f16`
// int4 dot over that staged, normalized activation. Every arithmetic step
// mirrors the standalone norm + GEMV pair, so tokens stay bit-for-bit identical
// while the separate normalization kernel is removed from the decode graph.
template <bool HasZp, bool SplitK = false>
__device__ __forceinline__ void matmul_nbits_gemv_f16_scales_f16_rmsnorm_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    // Normalized activation, staged 16-byte aligned so the dot below can reuse
    // the `scales_f16` `uint4` activation loads unchanged.
    extern __shared__ __align__(16) __half staged_activation[];
    __shared__ float shared_inv_std;
    __shared__ float partials[8][2];

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --- RMS reduction, byte-identical to `skip_rmsnorm_f16_warp_half4`. ---
    if (warp == 0) {
        const int chunks_per_lane = k / (32 * 4);
        const unsigned long long* activation4 =
            reinterpret_cast<const unsigned long long*>(activation);
        float ss0 = 0.0f;
        float ss1 = 0.0f;
        float ss2 = 0.0f;
        float ss3 = 0.0f;
        for (int item = 0; item < chunks_per_lane; ++item) {
            const int chunk = lane + item * 32;
            MatMulNBitsSkipHalf4 residual;
            residual.raw = activation4[chunk];
            const float2 rounded0 = __half22float2(residual.pair[0]);
            const float2 rounded1 = __half22float2(residual.pair[1]);
            ss0 += rounded0.x * rounded0.x;
            ss1 += rounded0.y * rounded0.y;
            ss2 += rounded1.x * rounded1.x;
            ss3 += rounded1.y * rounded1.y;
        }
        float ss = (ss0 + ss1) + (ss2 + ss3);
        for (int off = 16; off > 0; off >>= 1) {
            ss += __shfl_down_sync(0xffffffffu, ss, off);
        }
        if (lane == 0) {
            shared_inv_std = 1.0f / sqrtf(ss / (float)k + epsilon);
        }
    }
    __syncthreads();
    const float inv_std = shared_inv_std;

    // --- Normalized activation, matching the norm kernel's rounded output. ---
    for (int j = tid; j < k; j += (int)blockDim.x) {
        const float residual = __half2float(activation[j]);
        const float scale = load_rmsnorm_gamma(gamma, gamma_is_half, j);
        staged_activation[j] = __float2half((residual * inv_std) * scale);
    }
    __syncthreads();

    // --- Standard `scales_f16` int4 dot over the staged, normalized input. ---
    const int warps_per_block = (int)blockDim.x >> 5;
    const int columns_per_block = SplitK ? warps_per_block / 2 : warps_per_block;
    const int column_local = SplitK ? warp / 2 : warp;
    const int k_split = SplitK ? warp & 1 : 0;
    const int column = (int)blockIdx.x * columns_per_block + column_local;
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);

    __half2 sum0 = __float2half2_rn(0.0f);
    __half2 sum1 = __float2half2_rn(0.0f);
    __half2 sum2 = __float2half2_rn(0.0f);
    __half2 sum3 = __float2half2_rn(0.0f);
    float tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        int depth_base = k_split * 256;
        const __half* activation_ptr = staged_activation + depth_base + lane_depth;
        const unsigned char* packed_ptr =
            packed + (long)column * k_blocks * blob_size
                + (long)(depth_base >> 5) * blob_size + lane * 4;
        const __half* scale_ptr =
            scales + (long)column * k_blocks + (depth_base >> 5) + (lane >> 2);
        const int depth_step = SplitK ? 512 : 256;
        for (; depth_base + lane_depth + 8 <= k; depth_base += depth_step) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            const int block = (depth_base >> 5) + (lane >> 2);
            const unsigned int sub2 =
                block_sub2<HasZp>(zero_points, column, block, zp_row_bytes);
            accumulate_int4x8_f16_zp(
                packed_word,
                activation_ptr,
                *scale_ptr,
                sub2,
                sum0,
                sum1,
                sum2,
                sum3);
            activation_ptr += depth_step;
            packed_ptr += depth_step / 2;
            scale_ptr += depth_step / 32;
        }
        const int tail_depth = depth_base + lane_depth;
        if (tail_depth < k) {
            const unsigned int packed_word =
                *reinterpret_cast<const unsigned int*>(packed_ptr);
            const float scale = __half2float(*scale_ptr);
            const int tail_block = (depth_base >> 5) + (lane >> 2);
            const int zero_point =
                block_zp<HasZp>(zero_points, column, tail_block, zp_row_bytes);
            const int valid = min(8, k - tail_depth);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                    tail += (float)q * __half2float(activation_ptr[i]) * scale;
                }
            }
        }
    }
    const float2 value04 = __half22float2(sum0);
    const float2 value15 = __half22float2(sum1);
    const float2 value26 = __half22float2(sum2);
    const float2 value37 = __half22float2(sum3);
    float value = tail + value04.x;
    value += value15.x;
    value += value26.x;
    value += value37.x;
    value += value04.y;
    value += value15.y;
    value += value26.y;
    value += value37.y;
    value = warp_sum(value);
    if constexpr (SplitK) {
        if (lane == 0) {
            partials[column_local][k_split] = column < n ? value : 0.0f;
        }
        __syncthreads();
        if (k_split == 0 && lane == 0 && column < n) {
            output[column] = fold_bias_f16(
                partials[column_local][0] + partials[column_local][1],
                bias,
                column,
                bias_post_round);
        }
    } else if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_rmsnorm(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_scales_f16_rmsnorm_tpl<false>(activation, packed, scales_raw, zero_points, gamma, bias, output, k, n, k_blocks, blob_size, zp_row_bytes, bias_post_round, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_rmsnorm_splitk(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_scales_f16_rmsnorm_tpl<false, true>(activation, packed, scales_raw, zero_points, gamma, bias, output, k, n, k_blocks, blob_size, zp_row_bytes, bias_post_round, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_rmsnorm_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_scales_f16_rmsnorm_tpl<true>(activation, packed, scales_raw, zero_points, gamma, bias, output, k, n, k_blocks, blob_size, zp_row_bytes, bias_post_round, gamma_is_half, epsilon);
}

// Compile-time-specialized per-block int8 zero point. `HasZp == false`
// (symmetric int8) folds to the constant 128 with no load — mirroring the int4
// `block_zp` helper — so a future symmetric-int8 model keeps the constant
// subtrahend and never pays the per-block occupancy cost the int4 path shed.
template <bool HasZp>
__device__ __forceinline__ int block_zp_int8(
    const unsigned char* __restrict__ zero_points,
    const long column,
    const int block,
    const int k_blocks)
{
    if (!HasZp) {
        return 128;
    }
    return (int)zero_points[column * k_blocks + block];
}

// INT8 sibling of `matmul_nbits_gemv_f16_scales_f16_rmsnorm`. The RMS reduction
// and normalized-activation staging are byte-identical to the int4 fused kernel
// (and to the standalone `skip_rmsnorm_f16_warp_half4`); only the quantized dot
// differs, reusing the exact block-32 int8 dequant work split from
// `matmul_nbits_gemv_int8_f16` (one byte per weight, per-block uint8 zero point
// defaulting to 128, fp32 accumulation). Specialized on `HasZp` like the int4
// sibling so the symmetric case emits no per-block zero-point load.
template <bool HasZp>
__device__ __forceinline__ void matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    extern __shared__ __align__(16) __half staged_activation[];
    __shared__ float shared_inv_std;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --- RMS reduction, byte-identical to `skip_rmsnorm_f16_warp_half4`. ---
    if (warp == 0) {
        const int chunks_per_lane = k / (32 * 4);
        const unsigned long long* activation4 =
            reinterpret_cast<const unsigned long long*>(activation);
        float ss0 = 0.0f;
        float ss1 = 0.0f;
        float ss2 = 0.0f;
        float ss3 = 0.0f;
        for (int item = 0; item < chunks_per_lane; ++item) {
            const int chunk = lane + item * 32;
            MatMulNBitsSkipHalf4 residual;
            residual.raw = activation4[chunk];
            const float2 rounded0 = __half22float2(residual.pair[0]);
            const float2 rounded1 = __half22float2(residual.pair[1]);
            ss0 += rounded0.x * rounded0.x;
            ss1 += rounded0.y * rounded0.y;
            ss2 += rounded1.x * rounded1.x;
            ss3 += rounded1.y * rounded1.y;
        }
        float ss = (ss0 + ss1) + (ss2 + ss3);
        for (int off = 16; off > 0; off >>= 1) {
            ss += __shfl_down_sync(0xffffffffu, ss, off);
        }
        if (lane == 0) {
            shared_inv_std = 1.0f / sqrtf(ss / (float)k + epsilon);
        }
    }
    __syncthreads();
    const float inv_std = shared_inv_std;

    // --- Normalized activation, matching the norm kernel's rounded output. ---
    for (int j = tid; j < k; j += (int)blockDim.x) {
        const float residual = __half2float(activation[j]);
        const float scale = load_rmsnorm_gamma(gamma, gamma_is_half, j);
        staged_activation[j] = __float2half((residual * inv_std) * scale);
    }
    __syncthreads();

    // --- INT8 dot over the staged, normalized input (mirrors the non-fused
    //     `matmul_nbits_gemv_int8_f16` work split, fp32 accumulation). ---
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);

    float value = 0.0f;
    if (column < n) {
        const int quarter = lane & 3;
        for (int block_base = 0; block_base < k_blocks; block_base += 8) {
            const int block = block_base + (lane >> 2);
            float block_partial = 0.0f;
            if (block < k_blocks) {
                const int zero_point =
                    block_zp_int8<HasZp>(zero_points, column, block, k_blocks);
                const int depth = block * 32 + quarter * 8;
                const long packed_start =
                    ((long)column * k_blocks + block) * 32 + quarter * 8;
                if (depth + 8 <= k) {
                    const uint2 packed_word =
                        *reinterpret_cast<const uint2*>(packed + packed_start);
                    const unsigned char* bytes =
                        reinterpret_cast<const unsigned char*>(&packed_word);
                    const uint4 act =
                        *reinterpret_cast<const uint4*>(staged_activation + depth);
                    const __half* acth = reinterpret_cast<const __half*>(&act);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        block_partial += ((float)(int)bytes[i] - (float)zero_point)
                            * __half2float(acth[i]);
                    }
                } else if (depth < k) {
                    const int valid = min(8, k - depth);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int quantized = (int)packed[packed_start + i];
                            block_partial += ((float)quantized - (float)zero_point)
                                * __half2float(staged_activation[depth + i]);
                        }
                    }
                }
            }
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 2, 4);
            block_partial += __shfl_down_sync(0xffffffffu, block_partial, 1, 4);
            if (quarter == 0 && block < k_blocks) {
                const float scale =
                    __half2float(scales[(long)column * k_blocks + block]);
                value += block_partial * scale;
            }
        }
    }
    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

extern "C" __global__ void matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_tpl<false>(activation, packed, scales_raw, zero_points, gamma, bias, output, k, n, k_blocks, bias_post_round, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const void* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int bias_post_round,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_int8_f16_scales_f16_rmsnorm_tpl<true>(activation, packed, scales_raw, zero_points, gamma, bias, output, k, n, k_blocks, bias_post_round, gamma_is_half, epsilon);
}

// Standalone RMS-normalization prologue for the M>1 prefill path of the fused
// GEMV. It reproduces `skip_rmsnorm_f16_warp_half4` (minus the residual add,
// which the preceding GEMV's epilogue already applied) bit-for-bit: identical
// half4 chunking, identical `(ss0+ss1)+(ss2+ss3)` reduction, identical warp
// shuffle, and identical `__floats2half2_rn` output rounding. One warp
// normalizes one token row into `normalized`, which the portable tiled GEMM
// then consumes exactly as it would the standalone norm's fp16 output.
extern "C" __global__ void matmul_nbits_rmsnorm_f16_warp_half4(
    const __half* __restrict__ activation,
    const void* __restrict__ gamma,
    __half* __restrict__ normalized,
    const int norm_size,
    const int num_groups,
    const int gamma_is_half,
    const float epsilon)
{
    const int g = (int)blockIdx.x;
    if (g >= num_groups) return;
    const long base = (long)g * norm_size;
    const int lane = (int)threadIdx.x;
    const int chunks_per_lane = norm_size / (32 * 4);
    const unsigned long long* activation4 =
        reinterpret_cast<const unsigned long long*>(activation + base);
    const unsigned long long* gamma4 =
        reinterpret_cast<const unsigned long long*>(gamma);
    unsigned long long* normalized4 =
        reinterpret_cast<unsigned long long*>(normalized + base);
    float ss0 = 0.0f;
    float ss1 = 0.0f;
    float ss2 = 0.0f;
    float ss3 = 0.0f;
    for (int item = 0; item < chunks_per_lane; ++item) {
        const int chunk = lane + item * 32;
        MatMulNBitsSkipHalf4 residual;
        residual.raw = activation4[chunk];
        const float2 rounded0 = __half22float2(residual.pair[0]);
        const float2 rounded1 = __half22float2(residual.pair[1]);
        ss0 += rounded0.x * rounded0.x;
        ss1 += rounded0.y * rounded0.y;
        ss2 += rounded1.x * rounded1.x;
        ss3 += rounded1.y * rounded1.y;
    }
    float ss = (ss0 + ss1) + (ss2 + ss3);
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    float inv_std = 0.0f;
    if (lane == 0) {
        inv_std = 1.0f / sqrtf(ss / (float)norm_size + epsilon);
    }
    inv_std = __shfl_sync(0xffffffffu, inv_std, 0);
    for (int item = 0; item < chunks_per_lane; ++item) {
        const int chunk = lane + item * 32;
        MatMulNBitsSkipHalf4 residual;
        MatMulNBitsSkipHalf4 output;
        residual.raw = activation4[chunk];
        const float2 value0 = __half22float2(residual.pair[0]);
        const float2 value1 = __half22float2(residual.pair[1]);
        // gamma is only a final multiplicand: an fp16 gamma keeps the wide
        // half4 load, an fp32 gamma is read at full precision (matching the
        // standalone `skip_rmsnorm_f16_warp_half4`), so fp32-gamma decoders fuse.
        float scale0x, scale0y, scale1x, scale1y;
        if (gamma_is_half) {
            MatMulNBitsSkipHalf4 scale;
            scale.raw = gamma4[chunk];
            const float2 scale0 = __half22float2(scale.pair[0]);
            const float2 scale1 = __half22float2(scale.pair[1]);
            scale0x = scale0.x;
            scale0y = scale0.y;
            scale1x = scale1.x;
            scale1y = scale1.y;
        } else {
            const int j = chunk << 2;
            const float* gamma_f = reinterpret_cast<const float*>(gamma);
            scale0x = gamma_f[j];
            scale0y = gamma_f[j + 1];
            scale1x = gamma_f[j + 2];
            scale1y = gamma_f[j + 3];
        }
        output.pair[0] = __floats2half2_rn(
            value0.x * inv_std * scale0x,
            value0.y * inv_std * scale0y);
        output.pair[1] = __floats2half2_rn(
            value1.x * inv_std * scale1x,
            value1.y * inv_std * scale1y);
        normalized4[chunk] = output.raw;
    }
}

// SwiGLU activation, byte-identical to the standalone `op_silu` in the
// elementwise kernels: silu(x) = x * sigmoid(x), evaluated in the same
// rounding-stable form so the paired epilogue reproduces the two-op tokens.
__device__ __forceinline__ float gate_up_silu_f32(float x)
{
    if (x >= 0.0f) {
        const float denominator = __fadd_rn(1.0f, (float)exp((double)-x));
        return __fdiv_rn(x, denominator);
    }

    const float e = (float)exp((double)x);
    const float numerator = __fmul_rn(x, e);
    return __fdiv_rn(numerator, __fadd_rn(1.0f, e));
}

__device__ __forceinline__ float gate_up_decomposed_silu_f32(float x)
{
    float sigmoid;
    if (x >= 0.0f) {
        sigmoid = 1.0f / (1.0f + (float)exp((double)-x));
    } else {
        const float e = (float)exp((double)x);
        sigmoid = e / (1.0f + e);
    }
    const float sigmoid_h = __half2float(__float2half_rn(sigmoid));
    return __half2float(__float2half_rn(__fmul_rn(x, sigmoid_h)));
}

// Paired gate/up projection + SwiGLU. One warp computes column `column` of BOTH
// the gate and up projections (which share the same activation and the block-32
// fp16 layout of `matmul_nbits_gemv_f16_scales_f16`), then writes
// silu(gate)*up directly. The activation is permuted once per K-tile and reused
// by both accumulators, so the two GEMVs read the activation from registers
// exactly once. The epilogue reproduces the standalone two-op numerics
// (`fp16(gate_acc)`, `fp16(up_acc)`, then `fp16(silu(gate_h)*up_h)`) so greedy
// decoding stays byte-identical. Register-only + warp shuffles: no shared
// memory, so it is portable to sm_53+ and safe on small SMs (no >48KB opt-in).
template <bool HasZp, bool Decomposed, bool FusedSym = false>
__device__ __forceinline__ void matmul_nbits_gemv_f16_gate_up_swiglu_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    __half2 g0 = __float2half2_rn(0.0f);
    __half2 g1 = __float2half2_rn(0.0f);
    __half2 g2 = __float2half2_rn(0.0f);
    __half2 g3 = __float2half2_rn(0.0f);
    __half2 u0 = __float2half2_rn(0.0f);
    __half2 u1 = __float2half2_rn(0.0f);
    __half2 u2 = __float2half2_rn(0.0f);
    __half2 u3 = __float2half2_rn(0.0f);
    float gate_tail = 0.0f;
    float up_tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        const __half* activation_ptr = activation + lane_depth;
        const unsigned char* packed_gate_ptr =
            packed_gate + (long)column * k_blocks * blob_size + lane * 4;
        const unsigned char* packed_up_ptr =
            packed_up + (long)column * k_blocks * blob_size + lane * 4;
        const __half* scale_gate_ptr =
            scales_gate + (long)column * k_blocks + (lane >> 2);
        const __half* scale_up_ptr =
            scales_up + (long)column * k_blocks + (lane >> 2);
        int depth_base = 0;
        for (; depth_base + lane_depth + 8 <= k; depth_base += 256) {
            // Permute the shared activation once; both projections reuse it.
            const uint4 permuted = permute_activation_f16x8(activation_ptr);
            const int block = (depth_base >> 5) + (lane >> 2);
            const unsigned int gate_sub2 =
                block_sub2<HasZp>(zero_points_gate, column, block, zp_row_bytes);
            const unsigned int up_sub2 =
                block_sub2<HasZp>(zero_points_up, column, block, zp_row_bytes);
            const unsigned int gate_word =
                *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
            const unsigned int up_word =
                *reinterpret_cast<const unsigned int*>(packed_up_ptr);
            if constexpr (FusedSym && !HasZp) {
                // Symmetric fast dequant: fold `- 8` into the bias constants
                // (byte-identical, four fewer f16x2 ops/word). `gate_sub2`/
                // `up_sub2` are the constant 8 here and go unused.
                accumulate_int4x8_f16_permuted_sym8(
                    gate_word, permuted, *scale_gate_ptr, g0, g1, g2, g3);
                accumulate_int4x8_f16_permuted_sym8(
                    up_word, permuted, *scale_up_ptr, u0, u1, u2, u3);
            } else {
                accumulate_int4x8_f16_permuted_zp(
                    gate_word, permuted, *scale_gate_ptr, gate_sub2, g0, g1, g2, g3);
                accumulate_int4x8_f16_permuted_zp(
                    up_word, permuted, *scale_up_ptr, up_sub2, u0, u1, u2, u3);
            }
            activation_ptr += 256;
            packed_gate_ptr += 128;
            packed_up_ptr += 128;
            scale_gate_ptr += 8;
            scale_up_ptr += 8;
        }
        const int tail_depth = depth_base + lane_depth;
        if (tail_depth < k) {
            const unsigned int gate_word =
                *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
            const unsigned int up_word =
                *reinterpret_cast<const unsigned int*>(packed_up_ptr);
            const float gate_scale = __half2float(*scale_gate_ptr);
            const float up_scale = __half2float(*scale_up_ptr);
            const int tail_block = (depth_base >> 5) + (lane >> 2);
            const int gate_zp =
                block_zp<HasZp>(zero_points_gate, column, tail_block, zp_row_bytes);
            const int up_zp =
                block_zp<HasZp>(zero_points_up, column, tail_block, zp_row_bytes);
            const int valid = min(8, k - tail_depth);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const float a = __half2float(activation_ptr[i]);
                    const int qg = (int)((gate_word >> (i * 4)) & 15u) - gate_zp;
                    const int qu = (int)((up_word >> (i * 4)) & 15u) - up_zp;
                    gate_tail += (float)qg * a * gate_scale;
                    up_tail += (float)qu * a * up_scale;
                }
            }
        }
    }
    // Reduce each accumulator in the exact term order of the standalone
    // `matmul_nbits_gemv_f16_scales_f16` epilogue so the pre-round sums match.
    const float2 g04 = __half22float2(g0);
    const float2 g15 = __half22float2(g1);
    const float2 g26 = __half22float2(g2);
    const float2 g37 = __half22float2(g3);
    float gate_value = gate_tail + g04.x;
    gate_value += g15.x;
    gate_value += g26.x;
    gate_value += g37.x;
    gate_value += g04.y;
    gate_value += g15.y;
    gate_value += g26.y;
    gate_value += g37.y;
    gate_value = warp_sum(gate_value);

    const float2 u04 = __half22float2(u0);
    const float2 u15 = __half22float2(u1);
    const float2 u26 = __half22float2(u2);
    const float2 u37 = __half22float2(u3);
    float up_value = up_tail + u04.x;
    up_value += u15.x;
    up_value += u26.x;
    up_value += u37.x;
    up_value += u04.y;
    up_value += u15.y;
    up_value += u26.y;
    up_value += u37.y;
    up_value = warp_sum(up_value);

    if (lane == 0 && column < n) {
        // Round each projection to fp16 first (matching the separate GEMV
        // stores), then compute silu(gate)*up and round once — identical to the
        // standalone silu_mul_f16 kernel fed by the two GEMV outputs.
        const float gate_h = __half2float(__float2half(gate_value));
        const float up_h = __half2float(__float2half(up_value));
        const float silu_h = Decomposed
            ? gate_up_decomposed_silu_f32(gate_h)
            : gate_up_silu_f32(gate_h);
        output[column] = __float2half_rn(__fmul_rn(silu_h, up_h));
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_tpl<false, false>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, output, k, n, k_blocks, blob_size, zp_row_bytes);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_decomposed_swiglu(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_tpl<false, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, output, k, n, k_blocks, blob_size, zp_row_bytes);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_tpl<true, false>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, output, k, n, k_blocks, blob_size, zp_row_bytes);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_tpl<true, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, output, k, n, k_blocks, blob_size, zp_row_bytes);
}

// Fused-symmetric (`ONNX_GENAI_GATEUP_VEC`) siblings of the two SYMMETRIC
// gate/up SwiGLU entries above. Byte-identical — only the dequant folds the
// `- 8` symmetric zero point into the bias constants (see
// `int4x8_to_half2x4_sym8`), issuing four fewer f16x2 ops per weight word to
// relieve the issue-bound decode GEMV. No asymmetric `_vec` variant: the `_zp`
// entries carry a per-block zero point that cannot be folded to a constant.
extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu_vec(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_tpl<false, false, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, output, k, n, k_blocks, blob_size, zp_row_bytes);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_vec(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_tpl<false, true, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, output, k, n, k_blocks, blob_size, zp_row_bytes);
}
// This is `matmul_nbits_gemv_f16_gate_up_swiglu` preceded by the exact prologue
// of `matmul_nbits_gemv_f16_scales_f16_rmsnorm`: the block reduces the shared
// activation (the residual sum the preceding GEMV epilogue already produced)
// once, stages the normalized activation into shared memory with the same
// rounding, and then both the gate and up GEMVs read that single staged,
// normalized activation. Doing the reduction once — rather than once per
// following GEMV — is the whole point of routing the fan-out-2 post-attention
// `SkipSimplifiedLayerNormalization` through the paired kernel. Every arithmetic
// step mirrors the standalone norm followed by the two-op gate/up SwiGLU, so
// greedy tokens stay bit-for-bit identical.
template <bool HasZp, bool Decomposed, bool FusedSym = false>
__device__ __forceinline__ void matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    extern __shared__ __align__(16) __half staged_activation[];
    __shared__ float shared_inv_std;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // --- RMS reduction, byte-identical to `skip_rmsnorm_f16_warp_half4`. ---
    if (warp == 0) {
        const int chunks_per_lane = k / (32 * 4);
        const unsigned long long* activation4 =
            reinterpret_cast<const unsigned long long*>(activation);
        float ss0 = 0.0f;
        float ss1 = 0.0f;
        float ss2 = 0.0f;
        float ss3 = 0.0f;
        for (int item = 0; item < chunks_per_lane; ++item) {
            const int chunk = lane + item * 32;
            MatMulNBitsSkipHalf4 residual;
            residual.raw = activation4[chunk];
            const float2 rounded0 = __half22float2(residual.pair[0]);
            const float2 rounded1 = __half22float2(residual.pair[1]);
            ss0 += rounded0.x * rounded0.x;
            ss1 += rounded0.y * rounded0.y;
            ss2 += rounded1.x * rounded1.x;
            ss3 += rounded1.y * rounded1.y;
        }
        float ss = (ss0 + ss1) + (ss2 + ss3);
        for (int off = 16; off > 0; off >>= 1) {
            ss += __shfl_down_sync(0xffffffffu, ss, off);
        }
        if (lane == 0) {
            shared_inv_std = 1.0f / sqrtf(ss / (float)k + epsilon);
        }
    }
    __syncthreads();
    const float inv_std = shared_inv_std;

    // --- Normalized activation, matching the norm kernel's rounded output. ---
    for (int j = tid; j < k; j += (int)blockDim.x) {
        const float residual = __half2float(activation[j]);
        const float scale = load_rmsnorm_gamma(gamma, gamma_is_half, j);
        staged_activation[j] = __float2half((residual * inv_std) * scale);
    }
    __syncthreads();

    // --- Paired gate/up int4 dot over the staged, normalized activation. ---
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    __half2 g0 = __float2half2_rn(0.0f);
    __half2 g1 = __float2half2_rn(0.0f);
    __half2 g2 = __float2half2_rn(0.0f);
    __half2 g3 = __float2half2_rn(0.0f);
    __half2 u0 = __float2half2_rn(0.0f);
    __half2 u1 = __float2half2_rn(0.0f);
    __half2 u2 = __float2half2_rn(0.0f);
    __half2 u3 = __float2half2_rn(0.0f);
    float gate_tail = 0.0f;
    float up_tail = 0.0f;
    if (column < n) {
        const int lane_depth = lane * 8;
        const __half* activation_ptr = staged_activation + lane_depth;
        const unsigned char* packed_gate_ptr =
            packed_gate + (long)column * k_blocks * blob_size + lane * 4;
        const unsigned char* packed_up_ptr =
            packed_up + (long)column * k_blocks * blob_size + lane * 4;
        const __half* scale_gate_ptr =
            scales_gate + (long)column * k_blocks + (lane >> 2);
        const __half* scale_up_ptr =
            scales_up + (long)column * k_blocks + (lane >> 2);
        int depth_base = 0;
        if constexpr (HasZp) {
            // Asymmetric (zero-point) path — the dominant Phi decode kernel, which
            // ncu shows is Long-Scoreboard/global-load-latency bound. Software-
            // pipeline the int4 gate/up weight loads: issue the next iteration's
            // two 128-byte weight words before consuming the current ones so the
            // load latency overlaps this iteration's compute. Pure scheduling
            // change (identical accumulation order/ops) → bit-identical to the
            // non-prefetched loop. Only the weight words are prefetched; also
            // prefetching the small (L1/L2-resident) scales and per-block zero
            // points pushed registers 48->56 and the occupancy loss erased the
            // latency win. The symmetric (`HasZp == false`) path below keeps its
            // exact original instruction stream so Qwen stays byte-identical with
            // no register/occupancy change.
            unsigned int gate_word_next =
                *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
            unsigned int up_word_next =
                *reinterpret_cast<const unsigned int*>(packed_up_ptr);
            for (; depth_base + lane_depth + 8 <= k; depth_base += 256) {
                const uint4 permuted = permute_activation_f16x8(activation_ptr);
                const int block = (depth_base >> 5) + (lane >> 2);
                const unsigned int gate_sub2 =
                    block_sub2<HasZp>(zero_points_gate, column, block, zp_row_bytes);
                const unsigned int up_sub2 =
                    block_sub2<HasZp>(zero_points_up, column, block, zp_row_bytes);
                const unsigned int gate_word = gate_word_next;
                const unsigned int up_word = up_word_next;
                if (depth_base + 256 + lane_depth + 8 <= k) {
                    gate_word_next = *reinterpret_cast<const unsigned int*>(
                        packed_gate_ptr + 128);
                    up_word_next = *reinterpret_cast<const unsigned int*>(
                        packed_up_ptr + 128);
                }
                accumulate_int4x8_f16_permuted_zp(
                    gate_word, permuted, *scale_gate_ptr, gate_sub2, g0, g1, g2, g3);
                accumulate_int4x8_f16_permuted_zp(
                    up_word, permuted, *scale_up_ptr, up_sub2, u0, u1, u2, u3);
                activation_ptr += 256;
                packed_gate_ptr += 128;
                packed_up_ptr += 128;
                scale_gate_ptr += 8;
                scale_up_ptr += 8;
            }
        } else {
            for (; depth_base + lane_depth + 8 <= k; depth_base += 256) {
                const uint4 permuted = permute_activation_f16x8(activation_ptr);
                const int block = (depth_base >> 5) + (lane >> 2);
                const unsigned int gate_word =
                    *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
                const unsigned int up_word =
                    *reinterpret_cast<const unsigned int*>(packed_up_ptr);
                if constexpr (FusedSym) {
                    // Symmetric fast dequant: fold `- 8` into the bias constants
                    // (byte-identical, four fewer f16x2 ops/word).
                    accumulate_int4x8_f16_permuted_sym8(
                        gate_word, permuted, *scale_gate_ptr, g0, g1, g2, g3);
                    accumulate_int4x8_f16_permuted_sym8(
                        up_word, permuted, *scale_up_ptr, u0, u1, u2, u3);
                } else {
                    const unsigned int gate_sub2 = block_sub2<HasZp>(
                        zero_points_gate, column, block, zp_row_bytes);
                    const unsigned int up_sub2 = block_sub2<HasZp>(
                        zero_points_up, column, block, zp_row_bytes);
                    accumulate_int4x8_f16_permuted_zp(
                        gate_word, permuted, *scale_gate_ptr, gate_sub2, g0, g1, g2, g3);
                    accumulate_int4x8_f16_permuted_zp(
                        up_word, permuted, *scale_up_ptr, up_sub2, u0, u1, u2, u3);
                }
                activation_ptr += 256;
                packed_gate_ptr += 128;
                packed_up_ptr += 128;
                scale_gate_ptr += 8;
                scale_up_ptr += 8;
            }
        }
        const int tail_depth = depth_base + lane_depth;
        if (tail_depth < k) {
            const unsigned int gate_word =
                *reinterpret_cast<const unsigned int*>(packed_gate_ptr);
            const unsigned int up_word =
                *reinterpret_cast<const unsigned int*>(packed_up_ptr);
            const float gate_scale = __half2float(*scale_gate_ptr);
            const float up_scale = __half2float(*scale_up_ptr);
            const int tail_block = (depth_base >> 5) + (lane >> 2);
            const int gate_zp =
                block_zp<HasZp>(zero_points_gate, column, tail_block, zp_row_bytes);
            const int up_zp =
                block_zp<HasZp>(zero_points_up, column, tail_block, zp_row_bytes);
            const int valid = min(8, k - tail_depth);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < valid) {
                    const float a = __half2float(activation_ptr[i]);
                    const int qg = (int)((gate_word >> (i * 4)) & 15u) - gate_zp;
                    const int qu = (int)((up_word >> (i * 4)) & 15u) - up_zp;
                    gate_tail += (float)qg * a * gate_scale;
                    up_tail += (float)qu * a * up_scale;
                }
            }
        }
    }
    const float2 g04 = __half22float2(g0);
    const float2 g15 = __half22float2(g1);
    const float2 g26 = __half22float2(g2);
    const float2 g37 = __half22float2(g3);
    float gate_value = gate_tail + g04.x;
    gate_value += g15.x;
    gate_value += g26.x;
    gate_value += g37.x;
    gate_value += g04.y;
    gate_value += g15.y;
    gate_value += g26.y;
    gate_value += g37.y;
    gate_value = warp_sum(gate_value);

    const float2 u04 = __half22float2(u0);
    const float2 u15 = __half22float2(u1);
    const float2 u26 = __half22float2(u2);
    const float2 u37 = __half22float2(u3);
    float up_value = up_tail + u04.x;
    up_value += u15.x;
    up_value += u26.x;
    up_value += u37.x;
    up_value += u04.y;
    up_value += u15.y;
    up_value += u26.y;
    up_value += u37.y;
    up_value = warp_sum(up_value);

    if (lane == 0 && column < n) {
        const float gate_h = __half2float(__float2half(gate_value));
        const float up_h = __half2float(__float2half(up_value));
        const float silu_h = Decomposed
            ? gate_up_decomposed_silu_f32(gate_h)
            : gate_up_silu_f32(gate_h);
        output[column] = __float2half_rn(__fmul_rn(silu_h, up_h));
    }
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<false, false>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_rmsnorm(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<false, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

// Fused-symmetric (`ONNX_GENAI_GATEUP_VEC`) siblings of the two SYMMETRIC
// RMS-norm-fused gate/up SwiGLU entries — the dominant qwen2.5-14b decode
// kernel is `..decomposed_swiglu_rmsnorm`. Byte-identical (folds the `- 8`
// symmetric zero point into the dequant bias constants; see
// `int4x8_to_half2x4_sym8`), only the issued instruction count drops.
extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_vec(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<false, false, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_rmsnorm_vec(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<false, true, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

// Occupancy-raised (`ONNX_GENAI_GATEUP_OCC`) siblings of the two SYMMETRIC
// RMS-norm-fused `_vec` entries. IDENTICAL kernel body and math — the only
// change is `__launch_bounds__(256, 8)`, which caps the register allocation at
// 32 regs/thread so the SM can co-resident 8 blocks (100% theoretical vs 75%
// register-limited). The dominant qwen2.5-14b decode kernel
// (`..decomposed_swiglu_rmsnorm`) is Short-Scoreboard/shared-load-latency bound
// (~51% of stall cycles waiting on the staged-activation LDS); the extra
// resident warps hide that latency. `__launch_bounds__` only constrains
// register allocation — same instruction stream, same fp16 accumulate order,
// same RMS reduction — so it is BYTE-IDENTICAL to the `_vec` entries above.
extern "C" __global__ void __launch_bounds__(256, 8)
matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_vec_occ(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<false, false, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

extern "C" __global__ void __launch_bounds__(256, 8)
matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_rmsnorm_vec_occ(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<false, true, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<true, false>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

extern "C" __global__ void matmul_nbits_gemv_f16_gate_up_decomposed_swiglu_rmsnorm_zp(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed_gate,
    const __half* __restrict__ scales_gate,
    const unsigned char* __restrict__ packed_up,
    const __half* __restrict__ scales_up,
    const unsigned char* __restrict__ zero_points_gate,
    const unsigned char* __restrict__ zero_points_up,
    const void* __restrict__ gamma,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int gamma_is_half,
    const float epsilon)
{
    matmul_nbits_gemv_f16_gate_up_swiglu_rmsnorm_tpl<true, true>(activation, packed_gate, scales_gate, packed_up, scales_up, zero_points_gate, zero_points_up, gamma, output, k, n, k_blocks, blob_size, zp_row_bytes, gamma_is_half, epsilon);
}

// Down projection specialization: a 256-thread CTA (8 warps) computes `COLS`
// columns and parallelizes over block-32 K tiles. Each thread loads its assigned
// activation block directly into registers and reuses it across all `COLS`
// columns, then the 8 warps combine their per-column partials through shared
// memory.
//
// `COLS` is a pure grid-fill knob: every output column is still reduced
// *entirely within one CTA* by all 256 threads striding the same K tiles in the
// same order, so the fp32 accumulation is bit-identical regardless of `COLS` —
// only the CTA count (grid = ceil(N / COLS)) changes. Tall-skinny down/output
// projections have a small N, so on many-SM devices the default 8-column launch
// underfills the machine (e.g. Qwen2.5-7B down: N=3584 -> 448 CTAs, ~0.57
// waves/SM on an H200). Halving `COLS` doubles the grid to fill the idle SMs on
// this latency-bound M=1 GEMV without changing the numerics; the host picks
// `COLS` from the device multiprocessor count (see `select_down_columns`).
template <int COLS>
__device__ __forceinline__ void matmul_nbits_gemv_f16_scales_f16_down_tpl(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int n,
    const int k_blocks,
    const int blob_size,
    const int bias_post_round)
{
    __shared__ float warp_sums[8][COLS];
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int column_base = (int)blockIdx.x * COLS;

    float values[COLS];
#pragma unroll
    for (int i = 0; i < COLS; ++i) {
        values[i] = 0.0f;
    }
    for (int block = tid; block < k_blocks; block += (int)blockDim.x) {
        const __half* activation_block = activation + block * 32;
        const uint4 activation0 = permute_activation_f16x8(activation_block);
        const uint4 activation1 = permute_activation_f16x8(activation_block + 8);
        const uint4 activation2 = permute_activation_f16x8(activation_block + 16);
        const uint4 activation3 = permute_activation_f16x8(activation_block + 24);
#pragma unroll
        for (int tile_column = 0; tile_column < COLS; ++tile_column) {
            const int column = column_base + tile_column;
            if (column < n) {
                const long packed_start =
                    ((long)column * k_blocks + block) * blob_size;
                const uint4 packed_weights =
                    *reinterpret_cast<const uint4*>(packed + packed_start);
                const __half scale = scales[(long)column * k_blocks + block];
                values[tile_column] += dot_int4x32_f16_permuted_scaled(
                    packed_weights,
                    activation0,
                    activation1,
                    activation2,
                    activation3,
                    scale);
            }
        }
    }

#pragma unroll
    for (int tile_column = 0; tile_column < COLS; ++tile_column) {
        const float value = warp_sum(values[tile_column]);
        if (lane == 0) {
            warp_sums[warp][tile_column] = value;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < COLS) {
        const int column = column_base + lane;
        float value = warp_sums[0][lane];
        value += warp_sums[1][lane];
        value += warp_sums[2][lane];
        value += warp_sums[3][lane];
        value += warp_sums[4][lane];
        value += warp_sums[5][lane];
        value += warp_sums[6][lane];
        value += warp_sums[7][lane];
        if (column < n) {
            output[column] = fold_bias_f16(value, bias, column, bias_post_round);
        }
    }
}

// Default 8-column down projection (grid = ceil(N/8)).
extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_down(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)k;
    (void)block_size;
    (void)zero_points;
    (void)zp_row_bytes;
    (void)scales_fp16;
    matmul_nbits_gemv_f16_scales_f16_down_tpl<8>(
        activation, packed, scales_raw, bias, output, n, k_blocks, blob_size,
        bias_post_round);
}

// Grid-fill down projection variants: fewer columns per CTA -> proportionally
// larger grid, bit-identical output. Selected on grid-starved (small-N) down
// shapes to fill the multiprocessors on latency-bound decode.
extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_down_c4(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)k;
    (void)block_size;
    (void)zero_points;
    (void)zp_row_bytes;
    (void)scales_fp16;
    matmul_nbits_gemv_f16_scales_f16_down_tpl<4>(
        activation, packed, scales_raw, bias, output, n, k_blocks, blob_size,
        bias_post_round);
}

extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_down_c2(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)k;
    (void)block_size;
    (void)zero_points;
    (void)zp_row_bytes;
    (void)scales_fp16;
    matmul_nbits_gemv_f16_scales_f16_down_tpl<2>(
        activation, packed, scales_raw, bias, output, n, k_blocks, blob_size,
        bias_post_round);
}

// Model-agnostic fp16 int4/int8 decode GEMV supporting any power-of-two
// block_size. One warp per output column. Each lane owns contiguous 8-element K
// chunks and strides by 256 (= 32 lanes * 8) across the reduction. Unlike the
// tuned block-32 kernels, the scale / zero-point block index is derived from the
// real block_size (block = depth / block_size), so a lane's 8-element chunk
// always resolves to the block it belongs to for any block width that is a
// multiple of 8 (all supported power-of-two block sizes >= 16). The `bits`
// scalar selects the packed layout: int4 unpacks two nibbles per byte with an
// optional int4 zero point (default 8, packed two block-nibbles per byte); int8
// reads one unsigned byte per weight with an optional uint8 zero point (default
// 128, one per block) — byte-for-byte the tuned block-32 int8 layout generalized
// to any block width. fp32 accumulation is preserved; the kernel is
// register-only (capture-safe).
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    float value = 0.0f;
    if (column < n) {
        for (int depth = lane * 8; depth < k; depth += 256) {
            const int block = depth / block_size;
            const int within = depth - block * block_size;
            const long blob_base = ((long)column * k_blocks + block) * blob_size;
            float scale;
            if (scales_fp16) {
                scale = __half2float(
                    reinterpret_cast<const __half*>(scales)[(long)column * k_blocks + block]);
            } else {
                scale =
                    reinterpret_cast<const float*>(scales)[(long)column * k_blocks + block];
            }
            const int valid = min(8, k - depth);
            float partial = 0.0f;
            if (bits == 8) {
                const int zero_point =
                    zero_points ? (int)zero_points[(long)column * k_blocks + block] : 128;
                const unsigned char* block_bytes = packed + blob_base + within;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (i < valid) {
                        const int q = (int)block_bytes[i] - zero_point;
                        partial += (float)q * __half2float(activation[depth + i]);
                    }
                }
            } else {
                int zero_point = 8;
                if (zero_points) {
                    const unsigned char zp =
                        zero_points[(long)column * zp_row_bytes + (block >> 1)];
                    zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
                }
                const unsigned int packed_word =
                    *reinterpret_cast<const unsigned int*>(packed + blob_base + (within >> 1));
                if (valid == 8) {
                    // Fast path: LOP3 int4->fp16 dequant + 128-bit activation
                    // load. Byte-identical to the scalar loop below (same
                    // ascending-order fp32 products) but replaces the per-nibble
                    // shift/and/convert with 4 lop3 + f16x2 debias, cutting the
                    // dequant-ALU pressure that dominates the block!=32 GEMV.
                    const unsigned int sub2 = int4_zero_point_sub2(zero_point);
                    partial = dot_int4x8_f16_sub(packed_word, activation + depth, sub2);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                            partial += (float)q * __half2float(activation[depth + i]);
                        }
                    }
                }
            }
            value += partial * scale;
        }
    }

    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

// Split-K counterpart of `matmul_nbits_gemv_f16_general_bs`: K_SPLIT warps
// cooperate on one output column, each walking a strided subset of the 256-wide
// K steps, then summing their fp32 partials through shared memory. The launch
// grid is K_SPLIT x larger than the single-warp kernel, which fills the SMs on
// the grid-starved, latency-bound block!=32 decode GEMV (the medium
// projections run at ~0.5 waves/SM single-warp). Each warp keeps the same fp32
// LOP3 dequant loop as the single-warp kernel, so the only numeric difference
// is the new K-slice partial-sum association (near-equal, not byte-identical) —
// the same trade the block-32 split-K entries already ship by default.
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs_splitk(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    constexpr int K_SPLIT = 4;  // must match Rust GENERAL_BS_SPLITK
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int cols_per_block = warps_per_block / K_SPLIT;
    const int col_local = warp / K_SPLIT;
    const int ks = warp % K_SPLIT;
    const int column = (int)blockIdx.x * cols_per_block + col_local;

    __shared__ float partials[8][K_SPLIT];

    float value = 0.0f;
    if (column < n) {
        for (int depth = ks * 256 + lane * 8; depth < k; depth += K_SPLIT * 256) {
            const int block = depth / block_size;
            const int within = depth - block * block_size;
            const long blob_base = ((long)column * k_blocks + block) * blob_size;
            float scale;
            if (scales_fp16) {
                scale = __half2float(
                    reinterpret_cast<const __half*>(scales)[(long)column * k_blocks + block]);
            } else {
                scale =
                    reinterpret_cast<const float*>(scales)[(long)column * k_blocks + block];
            }
            const int valid = min(8, k - depth);
            float partial = 0.0f;
            if (bits == 8) {
                const int zero_point =
                    zero_points ? (int)zero_points[(long)column * k_blocks + block] : 128;
                const unsigned char* block_bytes = packed + blob_base + within;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (i < valid) {
                        const int q = (int)block_bytes[i] - zero_point;
                        partial += (float)q * __half2float(activation[depth + i]);
                    }
                }
            } else {
                int zero_point = 8;
                if (zero_points) {
                    const unsigned char zp =
                        zero_points[(long)column * zp_row_bytes + (block >> 1)];
                    zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
                }
                const unsigned int packed_word =
                    *reinterpret_cast<const unsigned int*>(packed + blob_base + (within >> 1));
                if (valid == 8) {
                    const unsigned int sub2 = int4_zero_point_sub2(zero_point);
                    partial = dot_int4x8_f16_sub(packed_word, activation + depth, sub2);
                } else {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        if (i < valid) {
                            const int q = (int)((packed_word >> (i * 4)) & 15u) - zero_point;
                            partial += (float)q * __half2float(activation[depth + i]);
                        }
                    }
                }
            }
            value += partial * scale;
        }
    }

    value = warp_sum(value);
    if (lane == 0) {
        partials[col_local][ks] = (column < n) ? value : 0.0f;
    }
    __syncthreads();
    if (ks == 0 && lane == 0 && column < n) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < K_SPLIT; ++s) {
            acc += partials[col_local][s];
        }
        output[column] = fold_bias_f16(acc, bias, column, bias_post_round);
    }
}

// Wide-load counterpart of `matmul_nbits_gemv_f16_general_bs` (see
// `gemv_int4_wide_lane_dot`). Same launch geometry (one warp per output column,
// 8 columns per 256-thread CTA) and same fp32/warp-sum reduction, but each lane
// streams 32 nibbles/step via a pipelined 128-bit weight load to lift DRAM
// throughput toward ORT's on the wide (already occupancy-filled) gate_up shape.
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs_wide(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int columns_per_block = (int)blockDim.x >> 5;
    const int column = (int)blockIdx.x * columns_per_block + warp;

    float value = 0.0f;
    if (column < n) {
        value = gemv_int4_wide_lane_dot(
            activation, packed, scales, zero_points, k, block_size, k_blocks,
            blob_size, zp_row_bytes, scales_fp16, column, lane * 32, 32 * 32);
    }
    value = warp_sum(value);
    if (lane == 0 && column < n) {
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}

// Wide-load counterpart of `matmul_nbits_gemv_f16_general_bs_splitk`: K_SPLIT
// warps cooperate on one output column, each walking a `32 * K_SPLIT`-strided
// set of 32-nibble chunks with the pipelined 128-bit loads, then summing their
// fp32 partials through shared memory (same grid-fill as the narrow split-K).
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs_splitk_wide(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    constexpr int K_SPLIT = 4;  // must match Rust GENERAL_BS_SPLITK
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int cols_per_block = warps_per_block / K_SPLIT;
    const int col_local = warp / K_SPLIT;
    const int ks = warp % K_SPLIT;
    const int column = (int)blockIdx.x * cols_per_block + col_local;

    __shared__ float partials[8][K_SPLIT];

    float value = 0.0f;
    if (column < n) {
        value = gemv_int4_wide_lane_dot(
            activation, packed, scales, zero_points, k, block_size, k_blocks,
            blob_size, zp_row_bytes, scales_fp16, column, ks * 32 * 32 + lane * 32,
            K_SPLIT * 32 * 32);
    }
    value = warp_sum(value);
    if (lane == 0) {
        partials[col_local][ks] = (column < n) ? value : 0.0f;
    }
    __syncthreads();
    if (ks == 0 && lane == 0 && column < n) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < K_SPLIT; ++s) {
            acc += partials[col_local][s];
        }
        output[column] = fold_bias_f16(acc, bias, column, bias_post_round);
    }
}

// Interleaved + biased (symmetric-only, OPT-IN) sibling of
// `matmul_nbits_gemv_f16_general_bs_splitk_wide`. Identical split-K geometry
// (K_SPLIT warps per column, shared-memory fp32 partial reduction), but consumes
// offline-interleaved weights and folds the symmetric -8 bias inside the LOP3
// converter. Because each lane's fp32 partial is bit-identical to the
// non-interleaved split-K wide kernel (see `gemv_int4_wide_lane_dot_interleaved`)
// and the K_SPLIT reduction order is unchanged, the output is byte-identical to
// `matmul_nbits_gemv_f16_general_bs_splitk_wide` on symmetric weights.
// `zero_points`/`zp_row_bytes`/`bits` are accepted for launch-signature parity
// but unused (the dispatch only routes symmetric int4 nodes here).
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs_splitk_wide_interleaved(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    (void)zero_points;
    (void)zp_row_bytes;
    (void)bits;
    constexpr int K_SPLIT = 4;  // must match Rust GENERAL_BS_SPLITK
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int cols_per_block = warps_per_block / K_SPLIT;
    const int col_local = warp / K_SPLIT;
    const int ks = warp % K_SPLIT;
    const int column = (int)blockIdx.x * cols_per_block + col_local;

    __shared__ float partials[8][K_SPLIT];

    float value = 0.0f;
    if (column < n) {
        value = gemv_int4_wide_lane_dot_interleaved(
            activation, packed, scales, k, block_size, k_blocks,
            blob_size, scales_fp16, column, ks * 32 * 32 + lane * 32,
            K_SPLIT * 32 * 32);
    }
    value = warp_sum(value);
    if (lane == 0) {
        partials[col_local][ks] = (column < n) ? value : 0.0f;
    }
    __syncthreads();
    if (ks == 0 && lane == 0 && column < n) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < K_SPLIT; ++s) {
            acc += partials[col_local][s];
        }
        output[column] = fold_bias_f16(acc, bias, column, bias_post_round);
    }
}

// Multicol x split-K hybrid of `matmul_nbits_gemv_f16_general_bs_splitk_wide`.
// Combines the split-K grid-fill (K_SPLIT warps cooperate on one column group,
// summing their fp32 partials through shared memory) with the register-blocked
// wide-load multicol dot (each warp accumulates WIDE_NC output columns, issuing
// WIDE_NC independent 128-bit weight loads per chunk). The split-K path was
// picked for the grid-starved medium-N projections (down_proj N~4096, qkv/attn-
// out) *because* single-warp multicol under-fills the SMs there (~<1 wave); this
// hybrid restores the multicol memory-level parallelism (the lever that lifts the
// gate_up `wide_multicol` kernel to ~37% DRAM peak) WHILE keeping K_SPLIT so the
// grid still fills the device. Each lane's fp32 partial for (column, ks) uses the
// SAME depth0 (`ks*32*32 + lane*32`) and stride (`K_SPLIT*32*32`) as the
// single-column split-K wide kernel, the per-column accumulation order inside
// `gemv_int4_wide_lane_dot_multicol` is unchanged, and the K_SPLIT shared-memory
// reduction order is unchanged, so the output is BYTE-IDENTICAL to
// `matmul_nbits_gemv_f16_general_bs_splitk_wide`.
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs_splitk_wide_multicol(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    constexpr int K_SPLIT = 4;  // must match Rust GENERAL_BS_SPLITK_MULTICOL
    // 256-thread CTA (8 warps) => MAX_COL_GROUPS column groups per block.
    constexpr int MAX_COL_GROUPS = 8 / K_SPLIT;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int col_groups_per_block = warps_per_block / K_SPLIT;
    const int group_local = warp / K_SPLIT;
    const int ks = warp % K_SPLIT;
    const long col_base =
        ((long)blockIdx.x * col_groups_per_block + group_local) * (long)WIDE_NC;

    __shared__ float partials[MAX_COL_GROUPS][WIDE_NC][K_SPLIT];

    float values[WIDE_NC];
    gemv_int4_wide_lane_dot_multicol(
        activation, packed, scales, zero_points, k, block_size, k_blocks,
        blob_size, zp_row_bytes, scales_fp16, col_base, n,
        ks * 32 * 32 + lane * 32, K_SPLIT * 32 * 32, values);
#pragma unroll
    for (int c = 0; c < WIDE_NC; ++c) {
        const float reduced = warp_sum(values[c]);
        if (lane == 0) {
            partials[group_local][c][ks] = reduced;
        }
    }
    __syncthreads();
    if (ks == 0 && lane == 0) {
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            const long column = col_base + c;
            if (column < n) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < K_SPLIT; ++s) {
                    acc += partials[group_local][c][s];
                }
                output[column] = fold_bias_f16(acc, bias, column, bias_post_round);
            }
        }
    }
}

// Interleaved + biased (symmetric-only, OPT-IN) sibling of
// `matmul_nbits_gemv_f16_general_bs_splitk_wide_multicol`. Same multicol x
// split-K geometry, but consumes offline nibble-interleaved weights and folds
// the symmetric -8 bias inside the LOP3 converter (dropping the per-block
// zero-point subtract and the `prmt.b32` activation reorder). Because each lane's
// fp32 partial is bit-identical to the non-interleaved multicol dot on symmetric
// weights and the K_SPLIT reduction order is unchanged, the output is
// byte-identical to `matmul_nbits_gemv_f16_general_bs_splitk_wide_multicol`.
// `zero_points`/`zp_row_bytes`/`bits` are accepted for launch-signature parity
// but unused (the dispatch only routes symmetric int4 nodes here).
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs_splitk_wide_multicol_interleaved(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    (void)zero_points;
    (void)zp_row_bytes;
    (void)bits;
    constexpr int K_SPLIT = 4;  // must match Rust GENERAL_BS_SPLITK_MULTICOL
    constexpr int MAX_COL_GROUPS = 8 / K_SPLIT;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const int col_groups_per_block = warps_per_block / K_SPLIT;
    const int group_local = warp / K_SPLIT;
    const int ks = warp % K_SPLIT;
    const long col_base =
        ((long)blockIdx.x * col_groups_per_block + group_local) * (long)WIDE_NC;

    __shared__ float partials[MAX_COL_GROUPS][WIDE_NC][K_SPLIT];

    float values[WIDE_NC];
    gemv_int4_wide_lane_dot_multicol_interleaved(
        activation, packed, scales, k, block_size, k_blocks,
        blob_size, scales_fp16, col_base, n,
        ks * 32 * 32 + lane * 32, K_SPLIT * 32 * 32, values);
#pragma unroll
    for (int c = 0; c < WIDE_NC; ++c) {
        const float reduced = warp_sum(values[c]);
        if (lane == 0) {
            partials[group_local][c][ks] = reduced;
        }
    }
    __syncthreads();
    if (ks == 0 && lane == 0) {
#pragma unroll
        for (int c = 0; c < WIDE_NC; ++c) {
            const long column = col_base + c;
            if (column < n) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < K_SPLIT; ++s) {
                    acc += partials[group_local][c][s];
                }
                output[column] = fold_bias_f16(acc, bias, column, bias_post_round);
            }
        }
    }
}

// Column register-blocked wide-load GEMV (see `gemv_int4_wide_lane_dot_multicol`).
// Same launch geometry as `matmul_nbits_gemv_f16_general_bs_wide` (256-thread
// CTA, one warp per group), but every warp emits WIDE_NC output columns, so the
// grid covers `columns_per_block = 8 * WIDE_NC` columns per block. Decoding each
// activation sub-word once and reusing it across WIDE_NC columns relieves the
// L1/TEX-throughput limiter of the single-column wide kernel while staying
// byte-identical (per-column fp32 accumulation order unchanged).
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs_wide_multicol(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const long col_base =
        ((long)blockIdx.x * warps_per_block + warp) * (long)WIDE_NC;

    float values[WIDE_NC];
    gemv_int4_wide_lane_dot_multicol(
        activation, packed, scales, zero_points, k, block_size, k_blocks,
        blob_size, zp_row_bytes, scales_fp16, col_base, n, lane * 32, 32 * 32,
        values);
#pragma unroll
    for (int c = 0; c < WIDE_NC; ++c) {
        const float reduced = warp_sum(values[c]);
        const long column = col_base + c;
        if (lane == 0 && column < n) {
            output[column] = fold_bias_f16(reduced, bias, column, bias_post_round);
        }
    }
}

// Interleaved + biased (symmetric-only, OPT-IN) sibling of
// `matmul_nbits_gemv_f16_general_bs_wide_multicol`. Identical launch geometry
// and grid; consumes offline-interleaved weights and folds the symmetric -8
// bias inside the LOP3 converter, dropping the per-block zero-point subtract and
// the `prmt.b32` activation reorder. Byte-identical output to the fp32 multicol
// kernel on symmetric weights (see `gemv_int4_wide_lane_dot_multicol_interleaved`).
// `zero_points`/`zp_row_bytes` are accepted for launch-signature parity but
// unused (the dispatch only routes symmetric nodes here).
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs_wide_multicol_interleaved(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    (void)zero_points;
    (void)zp_row_bytes;
    (void)bits;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const long col_base =
        ((long)blockIdx.x * warps_per_block + warp) * (long)WIDE_NC;

    float values[WIDE_NC];
    gemv_int4_wide_lane_dot_multicol_interleaved(
        activation, packed, scales, k, block_size, k_blocks,
        blob_size, scales_fp16, col_base, n, lane * 32, 32 * 32,
        values);
#pragma unroll
    for (int c = 0; c < WIDE_NC; ++c) {
        const float reduced = warp_sum(values[c]);
        const long column = col_base + c;
        if (lane == 0 && column < n) {
            output[column] = fold_bias_f16(reduced, bias, column, bias_post_round);
        }
    }
}

// Offline (once-per-weight) int4 nibble-interleave pass for the
// `ONNX_GENAI_INTERLEAVE_DEQUANT` lever. Reads the packed weight buffer as
// 32-bit words and rewrites each word from natural nibble order
// [e7 e6 e5 e4 | e3 e2 e1 e0] to TRT-LLM order [e7 e5 e3 e1 | e6 e4 e2 e0]
// (even elements to the low four nibble slots, odd to the high four). This is a
// pure per-word permutation, independent of block layout, so it applies to any
// int4 MatMulNBits weight whose byte count is a multiple of 4. Run once into a
// cached device buffer; the decode GEMV then reads the interleaved buffer.
extern "C" __global__ void matmul_nbits_interleave_int4(
    const unsigned int* __restrict__ src,
    unsigned int* __restrict__ dst,
    const unsigned long words)
{
    const unsigned long idx =
        (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= words) {
        return;
    }
    const unsigned int w = src[idx];
    const unsigned int e0 = (w >> 0) & 0xfu;
    const unsigned int e1 = (w >> 4) & 0xfu;
    const unsigned int e2 = (w >> 8) & 0xfu;
    const unsigned int e3 = (w >> 12) & 0xfu;
    const unsigned int e4 = (w >> 16) & 0xfu;
    const unsigned int e5 = (w >> 20) & 0xfu;
    const unsigned int e6 = (w >> 24) & 0xfu;
    const unsigned int e7 = (w >> 28) & 0xfu;
    // Physical slots [n0..n7] = [e0,e2,e4,e6,e1,e3,e5,e7].
    dst[idx] = (e0 << 0) | (e2 << 4) | (e4 << 8) | (e6 << 12)
             | (e1 << 16) | (e3 << 20) | (e5 << 24) | (e7 << 28);
}

// fp16 mixed-precision sibling of `matmul_nbits_gemv_f16_general_bs_wide_multicol`
// (see `gemv_int4_fp16_lane_dot_multicol`). Same launch geometry and grid; the
// only difference is the per-chunk fp16 __hfma2 MAC. Opt-in (gated by
// `use_gemv_fp16`) and accuracy-gated (NOT byte-identical to the fp32 path).
extern "C" __global__ void matmul_nbits_gemv_f16_general_bs_wide_multicol_fp16(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round,
    const int bits)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps_per_block = (int)blockDim.x >> 5;
    const long col_base =
        ((long)blockIdx.x * warps_per_block + warp) * (long)WIDE_NC;

    float values[WIDE_NC];
    gemv_int4_fp16_lane_dot_multicol(
        activation, packed, scales, zero_points, k, block_size, k_blocks,
        blob_size, zp_row_bytes, scales_fp16, col_base, n, lane * 32, 32 * 32,
        values);
#pragma unroll
    for (int c = 0; c < WIDE_NC; ++c) {
        const float reduced = warp_sum(values[c]);
        const long column = col_base + c;
        if (lane == 0 && column < n) {
            output[column] = fold_bias_f16(reduced, bias, column, bias_post_round);
        }
    }
}

// Model-agnostic fp16 int4/int8 prefill GEMM supporting any power-of-two
// block_size. Identical 16x16 tiling and fp32 accumulation as the tuned
// block-32 GEMM, but the reduction walks K in fixed 32-wide tiles and derives
// the block index from the real block_size (block = depth / block_size), so the
// K-tile width is decoupled from the block width. For block_size == 32 this is
// numerically identical to matmul_nbits_gemm_f16 (block == tile).
extern "C" __global__ void matmul_nbits_gemm_f16_general_bs(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int m,
    const int k,
    const int n,
    const int k_blocks,
    const int bits,
    const int scales_fp16,
    const int bias_post_round,
    const int bias_row_stride,
    const int block_size,
    const int blob_size)
{
    __shared__ float activation_tile[16][32];
    __shared__ float weight_tile[32][16];
    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;
    const int row = (int)blockIdx.y * 16 + (int)threadIdx.y;
    const int column = (int)blockIdx.x * 16 + (int)threadIdx.x;
    float value = 0.0f;

    const int k_tiles = (k + 31) / 32;
    for (int tile = 0; tile < k_tiles; ++tile) {
#pragma unroll
        for (int load = tid; load < 16 * 32; load += 16 * 16) {
            const int tile_row = load >> 5;
            const int within = load & 31;
            const int depth = tile * 32 + within;
            const int global_row = (int)blockIdx.y * 16 + tile_row;
            activation_tile[tile_row][within] =
                global_row < m && depth < k
                    ? __half2float(activation[(long)global_row * k + depth])
                    : 0.0f;
        }
#pragma unroll
        for (int load = tid; load < 32 * 16; load += 16 * 16) {
            const int tile_column = load >> 5;
            const int within = load & 31;
            const int depth = tile * 32 + within;
            const int global_column = (int)blockIdx.x * 16 + tile_column;
            float weight = 0.0f;
            if (global_column < n && depth < k) {
                const int block = depth / block_size;
                const int within_block = depth - block * block_size;
                const long scale_index = (long)global_column * k_blocks + block;
                const float scale = scales_fp16
                    ? __half2float(
                        reinterpret_cast<const __half*>(scales_raw)[scale_index])
                    : reinterpret_cast<const float*>(scales_raw)[scale_index];
                const long blob_base =
                    ((long)global_column * k_blocks + block) * blob_size;
                int quantized;
                int zero_point;
                if (bits == 8) {
                    quantized = (int)packed[blob_base + within_block];
                    zero_point = zero_points ? (int)zero_points[scale_index] : 128;
                } else {
                    const unsigned char byte =
                        packed[blob_base + (within_block >> 1)];
                    quantized = (within_block & 1) ? (byte >> 4) : (byte & 15);
                    zero_point = 8;
                    if (zero_points) {
                        const int zp_row_bytes = (k_blocks + 1) >> 1;
                        const unsigned char zp =
                            zero_points[(long)global_column * zp_row_bytes + (block >> 1)];
                        zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
                    }
                }
                weight = ((float)quantized - (float)zero_point) * scale;
            }
            weight_tile[within][tile_column] = weight;
        }
        __syncthreads();

        if (row < m && column < n) {
#pragma unroll
            for (int within = 0; within < 32; ++within) {
                value += activation_tile[threadIdx.y][within]
                    * weight_tile[within][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (row < m && column < n) {
        const __half* row_bias = bias ? bias + (long)row * bias_row_stride : bias;
        output[(long)row * n + column] =
            fold_bias_f16(value, row_bias, column, bias_post_round);
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum F16GemvVariant {
    General,
    DownProjection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct F16GemvSelection {
    variant: F16GemvVariant,
    reason: &'static str,
}

/// Choose the fp16 int4 GEMV variant by **structural shape class + capability**,
/// never by a specific model's dimensions.
///
/// The specialized `DownProjection` tiling stages the entire activation in
/// shared memory once and reuses it across the 8 columns of a CTA while the full
/// 256-thread block cooperatively reduces along `K`. That wins on the
/// **tall-skinny** class — a `K > N` GEMV (reduction depth exceeds output width,
/// e.g. an MLP down-projection or attention output-projection) — where the long
/// reduction benefits from block-parallel accumulation and each thread reuses
/// its register-held activation block across the CTA's columns. It is only
/// *correct* under the tiling's
/// hard constraints, all derived from the kernel body:
///
/// * `scales_fp16` and 4-bit weights (this fp16 GEMV path is always 4-bit),
/// * no explicit zero-points (the specialized half2/down kernels encode zp=8),
/// * `block_size == 32` and `K % 32 == 0` (full K-blocks; the kernel has no
///   partial-block tail).
///
/// Every other shape (wide `N >= K` projections, non-block-32, non-multiple-of-32
/// `K`) falls back to the general per-warp GEMV. Selection is thus generic across
/// models: any architecture's down/output projection that fits the class is
/// accelerated, and nothing keys on a magic `K`/`N`.
fn select_f16_gemv_variant(
    k: usize,
    n: usize,
    block_size: usize,
    scales_fp16: bool,
    has_zero_points: bool,
) -> F16GemvSelection {
    let down_eligible = !has_zero_points
        && scales_fp16
        && block_size == GEMV_F16_DOWN_BLOCK_SIZE
        && k.is_multiple_of(GEMV_F16_DOWN_BLOCK_SIZE)
        && k > n;
    if down_eligible {
        F16GemvSelection {
            variant: F16GemvVariant::DownProjection,
            reason: "variant=down_projection;class=tall_skinny(K>N);block_size=32;\
                     scales=fp16;K%32==0",
        }
    } else {
        F16GemvSelection {
            variant: F16GemvVariant::General,
            reason: if has_zero_points {
                "variant=general;zero_points=explicit;down_projection requires symmetric zp=8"
            } else {
                "variant=general;class=not(tall_skinny K>N & block_size=32 & \
                 scales=fp16 & K%32==0)"
            },
        }
    }
}

/// Per-SM CTA target used to size the down-projection launch. The tuned down
/// kernel is register-limited to ~6 resident CTAs/SM on sm_90, so `SM_count * 6`
/// is roughly one wave. This latency-bound M=1 GEMV keeps improving past one
/// wave — more resident CTAs hide the dependent global-load latency — so we aim
/// for ~2 waves. Measured on Qwen2.5-7B down (K=18944, N=3584) on an H200 (132
/// SMs): 8 cols/CTA = 448 CTAs (~0.57 waves) = 301.9 tok/s; 4 cols = 896
/// (~1.1 waves) = 305.8; 2 cols = 1792 (~2.3 waves) = 308.5 (+2.2%); 1 col =
/// 3584 (~4.5 waves) fell back to ~302 as 8x activation re-reads and CTA
/// oversubscription cancelled the occupancy gain. So 2 waves is the sweet spot
/// and `COLS` is floored at 2.
const DOWN_FILL_CTAS_PER_SM: usize = 12;

/// Pick the down-projection columns-per-CTA (8, 4, or 2) and matching kernel
/// entry to fill the multiprocessors. The base 8-column launch emits `ceil(N/8)`
/// CTAs; on tall-skinny down/output projections `N` is small, so a many-SM
/// device is left grid-starved (well under one wave) on this latency-bound M=1
/// GEMV. Halving the columns-per-CTA doubles the grid with bit-identical
/// numerics (each column is still reduced entirely within one CTA in the same
/// order). We keep the largest `COLS` whose grid already meets the ~2-wave
/// per-SM CTA target, so wide down projections retain the cheaper 8-column
/// launch while narrow ones split just enough to fill; `COLS` never drops below
/// 2 (a 1-column launch over-subscribes and re-reads the activation 8x, erasing
/// the gain). Keys only on `N` and the SM count — no per-model magic — and
/// returns a launch-time constant that is stable across CUDA-graph replays.
fn select_down_columns(n: usize, multiprocessor_count: u32) -> (usize, &'static str) {
    let target = (multiprocessor_count.max(1) as usize).saturating_mul(DOWN_FILL_CTAS_PER_SM);
    for (cols, entry) in [
        (GEMV_F16_DOWN_COLUMNS_PER_BLOCK, GEMV_F16_DOWN_ENTRY),
        (4usize, GEMV_F16_DOWN_C4_ENTRY),
    ] {
        if n.div_ceil(cols) >= target {
            return (cols, entry);
        }
    }
    (2usize, GEMV_F16_DOWN_C2_ENTRY)
}

/// Per-SM CTA target for the fp32-activation accuracy_level=4 blockwise GEMV.
/// This is a dependent-global-load latency-bound M=1 GEMV, so — like the
/// down-projection launch — it keeps improving past one wave; aim for ~2 waves
/// of resident CTAs.
const ACCURACY4_GEMV_FILL_CTAS_PER_SM: usize = 12;
const F16_SYMMETRIC_SPLITK_TARGET_WARPS_PER_SM: usize = 16;

/// Per-SM CTA count at or above which a single-warp block-32 scales-fp16 int4
/// GEMV is considered WELL-OCCUPIED and routed to the lower-register plain entry
/// instead of the prefetch-pipelined one (see [`scales_f16_pipe_well_occupied`]).
/// The pipe kernel's deeper register footprint hides the Long-Scoreboard
/// weight-load latency on grid-starved (warp-capped) projections but only shaves
/// occupancy once the launch already has many resident waves — measured on the
/// qwen3.5-0.8b LM head (N=248320, grid=31040 ≈ 235 CTAs/SM on an H200): the
/// plain entry runs ~85.0 µs vs the pipe entry's ~98.8 µs (-14%), while the
/// grid-starved projections (N=4096, grid=512 ≈ 3.9 CTAs/SM) keep the pipe entry
/// (4.66 vs 4.79 µs). One resident wave of 8-warp CTAs is 8/SM on sm_90; the
/// crossover is far above that, so the threshold is set well clear of it.
const GEMV_F16_PIPE_WELL_OCCUPIED_CTAS_PER_SM: usize = 32;

/// Per-SM CTA target for the block!=32 general_bs split-K decode GEMV. The
/// single-warp `general_bs` launch emits one 256-thread (8-warp) CTA per 8
/// output columns, so a projection is grid-starved (leaves SMs idle, starving
/// the latency-bound global loads of in-flight work) when `ceil(N/8)` CTAs is
/// below ~2 waves of resident CTAs. glm-4-9b's medium projections (N=4096) sit
/// at ~0.5 waves single-warp; K_SPLIT=2 doubles the grid to ~1 wave. The fused
/// gate_up (N~27k) is already >3 waves and stays single-warp.
const GENERAL_BS_SPLITK_TARGET_CTAS_PER_SM: usize = 16;

/// Whether the block!=32 int4 decode GEMV should take the split-K entry
/// ([`GEMV_F16_GENERAL_BS_SPLITK_ENTRY`]) instead of the single-warp
/// [`GEMV_F16_GENERAL_BS_ENTRY`]. Enabled only for grid-starved int4
/// projections so the well-occupied wide GEMVs (fused gate_up) keep the
/// single-warp path. `ONNX_GENAI_GENERAL_SPLITK=0|1` forces off/on for A/B
/// measurement; unset uses the device-derived heuristic.
fn use_general_bs_splitk(k: usize, n: usize, bits: usize, multiprocessor_count: u32) -> bool {
    if let Some(forced) = general_bs_splitk_override() {
        return forced;
    }
    let single_warp_ctas = n.div_ceil((GEMV_F16_LARGE_THREADS / 32) as usize);
    bits == 4
        && k >= 512
        // Split-K needs >= K_SPLIT warps/block; the small-shape path uses only
        // 64 threads (2 warps) < K_SPLIT, so restrict to the 256-thread large
        // path (matches the block-32 split-K gating).
        && !(n <= GEMV_F16_SMALL_N_MAX && k <= GEMV_F16_SMALL_N_MAX)
        && single_warp_ctas
            < (multiprocessor_count.max(1) as usize)
                .saturating_mul(GENERAL_BS_SPLITK_TARGET_CTAS_PER_SM)
}

/// Developer override for the general_bs split-K decode GEMV: `1`/`true`/`on`
/// forces the split-K entry, `0`/`false`/`off` forces the single-warp entry,
/// and any other/unset value defers to the device-derived heuristic.
fn general_bs_splitk_override() -> Option<bool> {
    match std::env::var("ONNX_GENAI_GENERAL_SPLITK").ok().as_deref() {
        Some("1") | Some("true") | Some("on") => Some(true),
        Some("0") | Some("false") | Some("off") => Some(false),
        _ => None,
    }
}

/// Whether the block!=32 int4 decode GEMV should take the wide-load (128-bit
/// `uint4`, software-pipelined) entries ([`GEMV_F16_GENERAL_BS_WIDE_ENTRY`] /
/// [`GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY`]) instead of the 32-bit narrow-load
/// entries. Default-on (the wide load is portable — no SM80 intrinsic — and
/// raises DRAM bandwidth on the memory-latency-bound M=1 GEMV toward ORT's).
/// Restricted to int4 with `block_size % 32 == 0 && k % 32 == 0` so each lane's
/// 32-nibble `uint4` chunk lies inside a single block; other layouts fall back
/// to the narrow entry byte-for-byte. Only wired into the `block_size != 32`
/// general_bs dispatch arm (glm-class block-128 large-N GEMV, measured +35%
/// decode); the block-32 (`scales_f16` / fused gate_up) path was measured a
/// NO-GO — those kernels are compute/SM-bound, not DRAM-bound, so the wide
/// fp32 path is flat-to-negative there (see the decode decision drop), and they
/// keep their tuned fp16-accumulate narrow entries. `ONNX_GENAI_GEMV_WIDELOAD=0`
/// forces the narrow entry for A/B measurement, `1` forces wide where permitted.
fn use_gemv_wideload(bits: usize, block_size: usize, k: usize) -> bool {
    if bits != 4 || !block_size.is_multiple_of(32) || !k.is_multiple_of(32) {
        return false;
    }
    !matches!(
        std::env::var("ONNX_GENAI_GEMV_WIDELOAD").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Whether the non-split-K wide GEMV should take the column register-blocked
/// entry ([`GEMV_F16_GENERAL_BS_WIDE_MULTICOL_ENTRY`]) instead of the
/// single-column [`GEMV_F16_GENERAL_BS_WIDE_ENTRY`]. Only consulted once
/// wide-load is already in effect (same int4 / `block_size % 32 == 0` /
/// `k % 32 == 0` preconditions, checked by the caller via [`use_gemv_wideload`]).
/// The multicol kernel decodes each activation sub-word once and reuses it
/// across [`GEMV_F16_WIDE_MULTICOL_NC`] columns; out-of-range columns are skipped
/// in-kernel, so any `n` is safe, and the result is byte-identical to the
/// single-column wide entry. `ONNX_GENAI_GEMV_WIDE_MULTICOL=0` forces the
/// single-column wide entry for A/B measurement.
fn use_gemv_wide_multicol() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_GEMV_WIDE_MULTICOL")
            .ok()
            .as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Whether the block!=32 general_bs split-K wide GEMV should take the multicol x
/// split-K hybrid entry ([`GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_ENTRY`])
/// instead of the single-column split-K wide entry
/// ([`GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY`]). Only consulted once the split-K
/// wide path is already selected (`use_general_splitk && use_gemv_wideload`). The
/// hybrid register-blocks [`GEMV_F16_WIDE_MULTICOL_NC`] columns/warp to restore
/// the memory-level parallelism the single-column split-K kernel lacks, while
/// keeping K_SPLIT grid-fill; it is byte-identical to the single-column split-K
/// wide entry. Default-on; `ONNX_GENAI_GEMV_SPLITK_MULTICOL=0` forces the
/// single-column split-K wide entry for A/B measurement.
fn use_gemv_splitk_multicol() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_GEMV_SPLITK_MULTICOL")
            .ok()
            .as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Grid-fill override for the split-K wide path on GRID-STARVED narrow-N
/// projections (e.g. a GQA k/v projection: N = kv_heads * head_dim, ~256).
///
/// The multicol hybrid ([`GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_ENTRY`])
/// register-blocks [`GEMV_F16_WIDE_MULTICOL_NC`] columns per warp group, so a
/// 256-thread (8-warp) CTA covers `(8 / GENERAL_BS_SPLITK_MULTICOL) *
/// GEMV_F16_WIDE_MULTICOL_NC` output columns and its launch grid is only
/// `ceil(N / cols_per_cta)`. That register blocking is a WIN on the medium/large
/// projections (down_proj / q / o at N~4096) where it lifts the memory-level
/// parallelism at ~1 wave, but on a narrow projection it collapses the grid so
/// far below the device SM count that most SMs sit idle (measured N=256 on GLM's
/// 2-kv-head GQA: grid=32 / ~0.06 waves / 2% DRAM / 6.1 us), starving the
/// latency-bound global loads of any in-flight parallelism to hide behind.
///
/// The single-column split-K wide entry
/// ([`GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY`]) covers only `8 /
/// GENERAL_BS_SPLITK` columns per CTA (4x fewer), so its grid is `4 *
/// GEMV_F16_WIDE_MULTICOL_NC` = 4x larger and fills those idle SMs. It is
/// BYTE-IDENTICAL to the multicol hybrid (same per-column depth0/stride, same
/// K_SPLIT reduction order — the only difference is how many columns a warp
/// register-blocks), so this is a pure occupancy swap, not a numeric change
/// (measured N=256: grid=32->128, 6.1->4.1 us / -33%). The medium/large
/// projections keep the multicol hybrid unchanged.
///
/// Returns true (prefer the single-column entry) when the multicol grid would
/// leave the device under-filled — `ceil(N / multicol_cols_per_cta) <
/// multiprocessor_count` — i.e. below ~1 CTA/SM from the column dimension. This
/// is shape/device-derived (narrow-N + SM count), not model-specific, so it
/// helps every GQA/MLA int4 model's narrow k/v projection. `unset` uses the
/// heuristic; `ONNX_GENAI_GEMV_SPLITK_SMALLN_SINGLECOL=0|1` forces off/on.
fn splitk_smalln_prefers_single_column(n: usize, multiprocessor_count: u32) -> bool {
    match std::env::var("ONNX_GENAI_GEMV_SPLITK_SMALLN_SINGLECOL")
        .ok()
        .as_deref()
    {
        Some("1") | Some("true") | Some("on") => return true,
        Some("0") | Some("false") | Some("off") => return false,
        _ => {}
    }
    // Columns covered by one 256-thread (8-warp) CTA of the multicol hybrid.
    let multicol_cols_per_cta = (8 / GENERAL_BS_SPLITK_MULTICOL) * GEMV_F16_WIDE_MULTICOL_NC;
    let multicol_grid = n.div_ceil(multicol_cols_per_cta.max(1));
    multicol_grid < multiprocessor_count.max(1) as usize
}

/// Whether the column register-blocked wide GEMV should take the fp16
/// mixed-precision entry ([`GEMV_F16_GENERAL_BS_WIDE_MULTICOL_FP16_ENTRY`])
/// instead of the fp32 [`GEMV_F16_GENERAL_BS_WIDE_MULTICOL_ENTRY`]. Only
/// consulted once wide-load + multicol are already in effect. The fp16 kernel
/// runs the per-chunk MAC in `__hfma2` (matching ORT's fp16 arithmetic, the
/// equal-conditions fp16-vs-fp16 path) at the cost of NOT being byte-identical
/// to the fp32 path — so it is OPT-IN during the accuracy-validation phase via
/// `ONNX_GENAI_GEMV_FP16=1` and gated on accuracy (error <= ORT vs an f64
/// oracle), not bit-identity. It becomes default-on only after passing that gate.
fn use_gemv_fp16() -> bool {
    matches!(
        std::env::var("ONNX_GENAI_GEMV_FP16").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Whether the M>1 half-precision path dequantizes to fp16 and runs the GEMM on
/// cuBLASLt tensor cores (`ONNX_GENAI_DEQUANT_F16_GEMM`, default ON).
///
/// Prefill on this shape is otherwise a choice between two slow kernels: the
/// `matmul_nbits_marlin_gemm_f16` int4 GEMM, which stages nothing in shared
/// memory and re-reads the whole A panel from global memory per warp, and the
/// f32 dequant fallback, which gives up tensor cores entirely. Materializing the
/// weights as fp16 and calling cuBLASLt costs one K*N pass over the weight and
/// then runs the matmul at the fp16 tensor-core rate.
///
/// Set to `0`/`false`/`off` to force the previous behaviour.
fn dequant_f16_gemm_enabled() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_DEQUANT_F16_GEMM").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Smallest M that takes the dequantize + fp16 cuBLASLt GEMM
/// (`ONNX_GENAI_DEQUANT_F16_GEMM_MIN_M`, default 8).
///
/// The dequantize pass costs a fixed K*N of bandwidth regardless of M, so it
/// only pays for itself once the GEMM has enough rows to amortize it. Below the
/// threshold the existing kernels, which read the weights in their packed form,
/// stay ahead.
fn dequant_f16_gemm_min_m() -> usize {
    std::env::var("ONNX_GENAI_DEQUANT_F16_GEMM_MIN_M")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|min_m| *min_m > 1)
        .unwrap_or(8)
}

/// Largest dequantized fp16 weight this path will materialize, in bytes
/// (`ONNX_GENAI_DEQUANT_F16_GEMM_MAX_SCRATCH`, default 1 GiB).
///
/// The scratch is transient, but a vocabulary projection is far larger than any
/// other node in the graph, and expanding one to fp16 would dwarf the GEMM it
/// feeds. Oversized nodes fall through rather than allocate.
fn dequant_f16_gemm_max_scratch_bytes() -> usize {
    std::env::var("ONNX_GENAI_DEQUANT_F16_GEMM_MAX_SCRATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1 << 30)
}

/// Opt-in gate for the TRT-LLM-style interleaved + biased int4 decode dequant/// (`ONNX_GENAI_INTERLEAVE_DEQUANT`). Default OFF: the lever bakes an offline
/// nibble-interleave into the packed weights and folds the symmetric `-8` bias
/// into the LOP3 converter, dropping the per-block zero-point `sub.f16x2` and the
/// `prmt.b32` activation reorder from the decode inner loop. Byte-identical to the
/// fp32 multicol path on symmetric weights, but glm base decode is a knife-edge,
/// so it stays opt-in and reversible until proven. Only routes symmetric
/// (no zero-point) block!=32 int4 nodes that already qualify for the wide-load
/// multicol kernel; every other node is untouched.
/// Default-on gate (`ONNX_GENAI_GEMV_BF16_DIRECT_OUT`) for letting the block-32
/// split-K decode GEMVs narrow their fp16 epilogue straight into the caller's
/// bf16 tensor, skipping the separate staging `cast_half` launch that
/// [`Bf16Kernel::run_bf16`] would otherwise issue per node. Kept as a lever so
/// the path can be A/B'd against the staged one without a rebuild.
fn gemv_bf16_direct_out_enabled() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_GEMV_BF16_DIRECT_OUT")
            .ok()
            .as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Offer published by `run_bf16` for the duration of the staged fp16 call it
/// makes on this same thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bf16DirectOut {
    /// fp16 staging buffer the fp16 path would otherwise write.
    staging: CUdeviceptr,
    /// Real bf16 destination the GEMV may write instead.
    dst: CUdeviceptr,
    /// Set once a GEMV has accepted the offer, so `run_bf16` knows to skip the
    /// narrowing cast and so a second GEMV in the same call cannot take it.
    taken: bool,
}

thread_local! {
    /// Thread-local rather than a field on the op: `run_bf16` calls the fp16
    /// path synchronously on the calling thread, so the offer cannot outlive
    /// that call or be observed by a concurrent session executing the same
    /// node. Acceptance additionally requires an exact match on the staging
    /// pointer, so an unrelated fp16 node can never claim it.
    static BF16_DIRECT_OUT: std::cell::Cell<Option<Bf16DirectOut>> =
        const { std::cell::Cell::new(None) };
}

/// Count of GEMV launches that narrowed straight into a bf16 output. Tests
/// assert this moves, so a routing change that silently reverts to the staged
/// cast fails loudly instead of passing on an unexercised equality.
static BF16_DIRECT_OUT_STORES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Accept the pending bf16 direct-store offer if it targets `staging`.
fn take_bf16_direct_out(staging: CUdeviceptr) -> Option<CUdeviceptr> {
    BF16_DIRECT_OUT.with(|cell| match cell.get() {
        Some(mut offer) if !offer.taken && offer.staging == staging => {
            offer.taken = true;
            cell.set(Some(offer));
            BF16_DIRECT_OUT_STORES.fetch_add(1, Ordering::Relaxed);
            Some(offer.dst)
        }
        _ => None,
    })
}

fn interleave_dequant_enabled() -> bool {
    matches!(
        std::env::var("ONNX_GENAI_INTERLEAVE_DEQUANT")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Module-global cache of offline nibble-interleaved int4 weights, keyed by the
/// source (packed) device pointer, byte length, and device ordinal. Mirrors the
/// Marlin repack cache: weights are immutable initializers, so the device
/// interleave runs once and every later call — including captured CUDA-graph
/// replays — reuses the cached buffer with no allocation.
struct InterleaveEntry {
    ptr: CUdeviceptr,
    runtime: Arc<CudaRuntime>,
}

struct InterleaveCache {
    map: std::collections::HashMap<(usize, usize, u32), InterleaveEntry>,
    order: std::collections::VecDeque<(usize, usize, u32)>,
}

const INTERLEAVE_CACHE_CAP: usize = 4096;

static INTERLEAVE_CACHE: std::sync::OnceLock<Mutex<InterleaveCache>> = std::sync::OnceLock::new();

fn interleave_cache() -> &'static Mutex<InterleaveCache> {
    INTERLEAVE_CACHE.get_or_init(|| {
        Mutex::new(InterleaveCache {
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
        })
    })
}

/// Launch the once-per-weight int4 nibble-interleave pass over `bytes` bytes
/// (`bytes` must be a multiple of 4). Reads `src` as 32-bit words and writes the
/// TRT-LLM interleaved order into `dst`.
fn launch_interleave_int4(
    runtime: &CudaRuntime,
    src: CUdeviceptr,
    dst: CUdeviceptr,
    bytes: usize,
) -> Result<()> {
    let words = (bytes / 4) as u64;
    let function = runtime.nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, INTERLEAVE_INT4_ENTRY)?;
    let src_ptr = cuptr(src as usize as *const c_void);
    let dst_ptr = cuptr(dst as usize as *const c_void);
    const THREADS: u32 = 256;
    let grid = u32::try_from(words.div_ceil(THREADS as u64))
        .unwrap_or(u32::MAX)
        .max(1);
    let mut builder = runtime.stream().launch_builder(&function);
    builder.arg(&src_ptr).arg(&dst_ptr).arg(&words);
    // SAFETY: static grid; `src`/`dst` each cover `bytes` bytes and the kernel
    // bounds-checks the word index against `words`.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|err| driver_err("launch MatMulNBits int4 interleave", err))
}

/// Ensure the offline-interleaved copy of the `bytes`-byte packed weight buffer
/// at `packed` exists on device, running the device interleave once and caching
/// the result. Returns `(interleaved_ptr, warm)`; `warm == true` means the
/// buffer was already cached (no allocation / interleave / sync this call, so it
/// is CUDA-graph capture-safe). A cold miss while capturing is rejected so the
/// caller falls back to the non-interleaved path rather than allocating inside a
/// capture.
fn ensure_interleaved(
    runtime: &Arc<CudaRuntime>,
    packed: CUdeviceptr,
    bytes: usize,
) -> Result<(CUdeviceptr, bool)> {
    let key = (packed as usize, bytes, runtime.ordinal());
    {
        let cache = interleave_cache()
            .lock()
            .map_err(|_| error("interleave cache mutex poisoned"))?;
        if let Some(entry) = cache.map.get(&key) {
            return Ok((entry.ptr, true));
        }
    }
    if runtime.is_capturing()? {
        return Err(error(
            "int4 interleave cannot allocate during CUDA-graph capture; the weight must be \
             interleaved during warmup before capture",
        ));
    }
    let out = runtime.alloc_raw(bytes)?;
    if let Err(e) = launch_interleave_int4(runtime, packed, out, bytes) {
        // SAFETY: `out` was just allocated here and is otherwise unreferenced.
        let _ = unsafe { runtime.free_raw(out) };
        return Err(e);
    }
    let mut cache = interleave_cache()
        .lock()
        .map_err(|_| error("interleave cache mutex poisoned"))?;
    // Another thread may have inserted the same key while we interleaved.
    if let Some(entry) = cache.map.get(&key) {
        let winner = entry.ptr;
        drop(cache);
        // SAFETY: `out` is our just-allocated duplicate; free it once.
        let _ = unsafe { runtime.free_raw(out) };
        return Ok((winner, true));
    }
    cache.map.insert(
        key,
        InterleaveEntry {
            ptr: out,
            runtime: runtime.clone(),
        },
    );
    cache.order.push_back(key);
    while cache.order.len() > INTERLEAVE_CACHE_CAP {
        if let Some(evict) = cache.order.pop_front()
            && let Some(entry) = cache.map.remove(&evict)
        {
            // SAFETY: exclusively owned by the cache; freed once on eviction.
            let _ = unsafe { entry.runtime.free_raw(entry.ptr) };
        }
    }
    Ok((out, false))
}

fn use_f16_symmetric_splitk(
    k: usize,
    n: usize,
    multiprocessor_count: u32,
    max_threads_per_block: u32,
) -> bool {
    let eligible =
        k >= 512 && k.is_multiple_of(32) && max_threads_per_block >= GEMV_F16_LARGE_THREADS;
    if !eligible {
        return false;
    }
    n < (multiprocessor_count.max(1) as usize)
        .saturating_mul(F16_SYMMETRIC_SPLITK_TARGET_WARPS_PER_SM)
}

/// Warps-per-CTA (one warp reduces one output column) for the accuracy_level=4
/// blockwise GEMV, chosen so the `ceil(N / warps)` grid fills the device. The
/// grid-starved tiled `matmul_nbits_accuracy4` fallback issued only
/// `ceil(M*N / 256)` CTAs (4 CTAs for a 1024-wide decode projection); here each
/// output column is a warp, so an 8-warp CTA already emits `ceil(N/8)` CTAs. On
/// narrow projections a many-SM device would still be under the ~2-wave target,
/// so the warp count is halved (down to a single warp) until the grid fills.
/// Keying only on `N` and the SM count keeps the choice a launch-time constant
/// that is stable across CUDA-graph replays and never hard-codes a GPU.
fn select_accuracy4_gemv_warps(n: usize, multiprocessor_count: u32) -> u32 {
    if let Some(forced) = std::env::var("ONNX_GENAI_ACC4_WARPS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|warps| matches!(warps, 1 | 2 | 4 | 8))
    {
        return forced;
    }
    let target =
        (multiprocessor_count.max(1) as usize).saturating_mul(ACCURACY4_GEMV_FILL_CTAS_PER_SM);
    for warps in [8usize, 4, 2] {
        if n.div_ceil(warps) >= target {
            return warps as u32;
        }
    }
    1
}

/// Select the wider activation stage only when the fixed 8-warp block-32 launch
/// does not provide one resident CTA wave. The estimate uses architectural warp
/// limits via [`crate::arch::decode_resident_warps_per_sm`]: datacenter
/// sm_80/sm_90 parts expose 64 warps/SM, while sm_86/sm_89 consumer parts expose
/// 48. Routing the ladder through the arch layer keeps this decode selector's
/// occupancy math byte-identical to the previous inline `match` (crucially
/// sm_90 → 64) while giving the pending RTX split-K tuning a single seam.
/// Hudson's general blockwise path keeps its separate device-derived warp-width
/// selection.
fn use_accuracy4_stage64(
    n: usize,
    multiprocessor_count: u32,
    compute_capability: (u32, u32),
    max_shared_memory_per_block: u32,
) -> bool {
    if max_shared_memory_per_block < GEMV_ACCURACY4_STAGE64_SHARED_BYTES {
        return false;
    }
    let resident_warps = crate::arch::decode_resident_warps_per_sm(compute_capability) as usize;
    let resident_ctas = resident_warps / (GEMV_ACCURACY4_THREADS as usize / 32);
    let one_wave = (multiprocessor_count.max(1) as usize).saturating_mul(resident_ctas);
    n.div_ceil(GEMV_ACCURACY4_COLUMNS_PER_BLOCK) < one_wave
}

/// Optional developer override for the down-projection columns-per-CTA, used to
/// A/B the grid-fill variants. `ONNX_GENAI_DOWN_COLS=8|4|2` forces that width;
/// any other/unset value keeps the device-driven [`select_down_columns`] choice.
fn down_columns_override() -> Option<(usize, &'static str)> {
    match std::env::var("ONNX_GENAI_DOWN_COLS").ok()?.as_str() {
        "8" => Some((GEMV_F16_DOWN_COLUMNS_PER_BLOCK, GEMV_F16_DOWN_ENTRY)),
        "4" => Some((4, GEMV_F16_DOWN_C4_ENTRY)),
        "2" => Some((2, GEMV_F16_DOWN_C2_ENTRY)),
        _ => None,
    }
}

/// Whether the single-warp block-32 scales-fp16 int4 decode GEMV takes the
/// prefetch-pipelined entry (4 weight loads in flight per lane to hide the
/// Long-Scoreboard global-load latency). Byte-identical to the original entry,
/// so it is default-on; `ONNX_GENAI_GEMV_PIPELINE=0` (or `false`/`off`) forces
/// the original single-load entry for A/B measurement.
fn scales_f16_pipeline_enabled() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_GEMV_PIPELINE").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Opt-out for the SYMMETRIC (no zero-point) int8 decode GEMV split-K grid-fill.
/// The asymmetric int8 GEMV already routes grid-starved shapes to the split-K
/// entry; the symmetric path was historically pinned to its single-warp kernel
/// purely to stay BYTE-IDENTICAL (split-K reassociates the per-column fp32
/// partial sums). Now that decode correctness is validated on the GREEDY
/// TOKEN-ID lock (not a byte-exact lock) — and fp32 accumulation is preserved,
/// so the reassociation is a ULP-level shift that leaves the argmax stable, the
/// same trade already shipped for the symmetric int4 split-K
/// ([`use_f16_symmetric_splitk`]) — the symmetric int8 path may take the same
/// grid-fill split-K on the grid-starved projections. Default-on; set
/// `ONNX_GENAI_CUDA_DISABLE_INT8_SYMMETRIC_SPLITK=1` (or `true`/`on`) to force
/// the byte-identical single-warp entry for A/B measurement or de-risking.
fn int8_symmetric_splitk_enabled() -> bool {
    !std::env::var_os("ONNX_GENAI_CUDA_DISABLE_INT8_SYMMETRIC_SPLITK")
        .is_some_and(|value| value != "0" && !value.is_empty())
}

/// Whether a single-warp block-32 scales-fp16 int4 GEMV launch is WELL-OCCUPIED
/// enough that it should take the lower-register plain entry
/// ([`GEMV_F16_SCALES_F16_ENTRY`]) instead of the prefetch-pipelined entry
/// ([`GEMV_F16_SCALES_F16_PIPE_ENTRY`]). Both entries are BYTE-IDENTICAL (same
/// lane→nibble mapping, same fp16 accumulation order), so this is a pure
/// occupancy/register trade, not a numeric change. The pipe entry's extra
/// registers pay off only on grid-starved (warp-capped) projections where the
/// scheduler has too few resident warps to hide the Long-Scoreboard weight-load
/// latency; once the launch already fills the SMs many times over (e.g. the
/// wide LM-head projection), the pipe entry's lower occupancy makes it a net
/// loss and the plain entry wins. Keys only on `N`, the launch width and the
/// live SM count (no per-model magic); returns a launch-time constant stable
/// across CUDA-graph replays. `ONNX_GENAI_GEMV_PIPELINE=0` still forces the
/// plain entry everywhere for A/B; `ONNX_GENAI_GEMV_PIPE_WELLOCC=0` forces the
/// pipe entry even on well-occupied launches (restores the pre-gate behavior).
fn scales_f16_pipe_well_occupied(
    n: usize,
    columns_per_block: usize,
    multiprocessor_count: u32,
) -> bool {
    match std::env::var("ONNX_GENAI_GEMV_PIPE_WELLOCC")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("off") => return false,
        Some("1") | Some("true") | Some("on") => return true,
        _ => {}
    }
    let ctas = n.div_ceil(columns_per_block.max(1));
    let threshold = (multiprocessor_count.max(1) as usize)
        .saturating_mul(GEMV_F16_PIPE_WELL_OCCUPIED_CTAS_PER_SM);
    ctas >= threshold
}

/// Whether the SYMMETRIC paired gate/up SwiGLU decode GEMV takes the
/// fused-symmetric (`_vec`) entry, which folds the `- 8` symmetric zero point
/// into the dequant bias constants and so issues four fewer `f16x2` ops per
/// weight word. That kernel is the largest decode kernel on qwen2.5-14b and is
/// issue-bound (ncu: ~28% DRAM, ~48% issue-active), so cutting issued
/// instructions is the right lever (prefetch/`uint4` loads do not help: the
/// loads are already 128B-coalesced per warp and strided per lane). The `_vec`
/// entry is BYTE-IDENTICAL to the default entry — every dequant intermediate is
/// an exactly-representable fp16 integer, so folding the two constant
/// subtractions changes no rounding — so it is default-on;
/// `ONNX_GENAI_GATEUP_VEC=0` (or `false`/`off`) forces the scalar entry for A/B
/// measurement.
fn gate_up_vec_enabled() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_GATEUP_VEC").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Whether the SYMMETRIC RMS-norm-fused gate/up SwiGLU decode GEMV takes the
/// occupancy-raised `_vec_occ` entry (`__launch_bounds__(256, 8)` → 32 regs,
/// 8 blocks/SM, 100% theoretical vs 75% register-limited). That kernel is the
/// largest decode kernel on qwen2.5-14b and is Short-Scoreboard bound (~51% of
/// stall cycles wait on the staged-activation shared load); the extra resident
/// warps hide the latency (isolated ncu: 57.5 -> 54.0us, occupancy 62 -> 82%;
/// E2E +2.6% on 14b). Byte-identical to the `_vec` entry — `__launch_bounds__`
/// only constrains register allocation. DEFAULT-ON; `ONNX_GENAI_GATEUP_OCC=0`
/// (or `false`/`off`) forces the non-occ path for A/B.
fn gate_up_occ_enabled() -> bool {
    !matches!(
        std::env::var("ONNX_GENAI_GATEUP_OCC").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

pub struct MatMulNBitsFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for MatMulNBitsFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let bits = optional_int_attr(node, "bits")?.unwrap_or(4);
        if !matches!(bits, 4 | 8) {
            return Err(error(format!(
                "MatMulNBits CUDA supports bits in {{4, 8}}, got bits={bits}. Why: the native \
                 kernels implement packed int4 and int8 layouts. How to fix: export bits=4 or \
                 bits=8, or select another execution provider"
            )));
        }
        let weight_prepacked = optional_int_attr(node, "weight_prepacked")?.unwrap_or(0);
        if weight_prepacked != 0 {
            return Err(error(format!(
                "weight_prepacked={weight_prepacked} is unsupported: CUDA only supports the standard (non-prepacked) layout"
            )));
        }
        let block_size = required_positive_attr(node, "block_size")?;
        if block_size < 16 || !block_size.is_power_of_two() {
            return Err(error(format!(
                "block_size must be a power of two and at least 16, got {block_size}"
            )));
        }
        // int8 at block_size==32 uses the tuned four-lane/eight-block GEMV; any
        // other power-of-two int8 block width routes to the model-agnostic
        // general-block-size GEMV/GEMM (both implement the one-byte-per-weight
        // layout), so no int8 block width is rejected here.
        let accuracy_level = node
            .attr("accuracy_level")
            .and_then(|value| value.as_int())
            .unwrap_or(0);

        let accuracy4_workspace = if bits == 4 && accuracy_level == 4 {
            Some(Mutex::new(Accuracy4Workspace::new(
                self.runtime.clone(),
                k,
                block_size,
            )?))
        } else {
            None
        };
        Ok(Box::new(MatMulNBitsKernel {
            runtime: self.runtime.clone(),
            k,
            n,
            bits: bits as usize,
            block_size,
            accuracy_level,
            accuracy4_workspace,
            fold_bias_post_round: node
                .attr(crate::optimizer::MATMUL_NBITS_FOLDED_BIAS_ATTR)
                .and_then(onnx_runtime_ir::Attribute::as_int)
                == Some(1),
            gate_up_swiglu: node
                .attr(crate::optimizer::GATE_UP_SWIGLU_FUSION_ATTR)
                .and_then(onnx_runtime_ir::Attribute::as_int)
                == Some(1),
            decomposed_silu: node
                .attr(crate::optimizer::DECOMPOSED_SILU_ATTR)
                .and_then(onnx_runtime_ir::Attribute::as_int)
                == Some(1),
            rmsnorm_prologue: node
                .attr(crate::optimizer::MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR)
                .and_then(onnx_runtime_ir::Attribute::as_int)
                == Some(1),
            rmsnorm_epsilon: node
                .attr(crate::optimizer::MATMUL_NBITS_RMSNORM_EPSILON_ATTR)
                .and_then(onnx_runtime_ir::Attribute::as_float)
                .unwrap_or(1e-5),
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(self.runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(self.runtime.clone())),
        }))
    }
}

#[derive(Debug)]
struct Accuracy4Workspace {
    runtime: Arc<CudaRuntime>,
    quantized_activation: CUdeviceptr,
    activation_scale: CUdeviceptr,
    padded_k: usize,
}

impl Accuracy4Workspace {
    fn new(runtime: Arc<CudaRuntime>, k: usize, block_size: usize) -> Result<Self> {
        let padded_k = k.div_ceil(block_size) * block_size;
        // Per-K-block int8 activation scales: one f32 per weight block.
        let k_blocks = padded_k / block_size;
        let scale_bytes = k_blocks * std::mem::size_of::<f32>();
        let quantized_activation = runtime.alloc_raw(padded_k + scale_bytes)?;
        Ok(Self {
            runtime,
            quantized_activation,
            activation_scale: quantized_activation + padded_k as CUdeviceptr,
            padded_k,
        })
    }
}

impl Drop for Accuracy4Workspace {
    fn drop(&mut self) {
        if self.quantized_activation != 0 {
            // SAFETY: this persistent buffer is exclusively owned by the kernel.
            let _ = unsafe { self.runtime.free_raw(self.quantized_activation) };
            self.quantized_activation = 0;
            self.activation_scale = 0;
        }
    }
}

/// Reusable device scratch for the BFloat16 activation path.
///
/// BFloat16 operands are staged through Float16 (see
/// [`MatMulNBitsKernel::run_bf16`]). A per-node allocation + free for each
/// staging buffer would synchronize the device on every one of the hundreds of
/// MatMulNBits nodes per decode step (`cuMemAlloc`/`cuMemFree` are synchronous),
/// collapsing throughput. This grow-only arena is carved into per-call regions
/// and reused across decode steps, so after the first step the BFloat16 path
/// issues only asynchronous casts and the tuned fp16 matmul — no device sync.
/// Each kernel instance owns its own arena, and consecutive uses are ordered on
/// the EP stream, so region reuse never races.
#[derive(Debug)]
struct Bf16Scratch {
    runtime: Arc<CudaRuntime>,
    ptr: CUdeviceptr,
    cap: usize,
}

impl Bf16Scratch {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            ptr: 0,
            cap: 0,
        }
    }

    /// Ensure the arena holds at least `bytes`, returning its base pointer.
    fn ensure(&mut self, bytes: usize) -> Result<CUdeviceptr> {
        if bytes > self.cap {
            if self.ptr != 0 {
                // SAFETY: exclusively owned; freed once before replacement.
                unsafe { self.runtime.free_raw(self.ptr)? };
                self.ptr = 0;
            }
            self.ptr = self.runtime.alloc_raw(bytes.max(1))?;
            self.cap = bytes;
        }
        Ok(self.ptr)
    }
}

impl Drop for Bf16Scratch {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: this persistent buffer is exclusively owned by the kernel.
            let _ = unsafe { self.runtime.free_raw(self.ptr) };
            self.ptr = 0;
        }
    }
}

/// Persistent per-kernel cache of the Float16-staged **constant** inputs on the
/// BFloat16 activation path. A node's scales / zero-point-scales / gamma are
/// immutable weights, but the original `run_bf16` re-cast them from BFloat16 to
/// Float16 into an ephemeral arena on *every* decode step. For Muse-Glimmer that
/// is ~3.3 GB/token of pure-copy traffic (≈25% of the int4 weight traffic) doing
/// nothing but reproducing an identical Float16 buffer. This cache stages each
/// constant once (keyed by its device pointer + element count) and reuses the
/// result across steps. Converting BFloat16→Float16 in one staging pass is
/// byte-identical whether done once or per step, so decode output is unchanged.
#[derive(Debug)]
struct Bf16ConstCache {
    runtime: Arc<CudaRuntime>,
    ptr: CUdeviceptr,
    cap: usize,
    /// One entry per cached constant: `(src_ptr, numel, byte_offset)`.
    slots: Vec<(CUdeviceptr, usize, usize)>,
}

impl Bf16ConstCache {
    fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            ptr: 0,
            cap: 0,
            slots: Vec::new(),
        }
    }

    /// Return the Float16 device pointers holding the staged copies of the
    /// BFloat16 constants described by `consts` (`(src_ptr, numel)` each),
    /// casting them on first use and reusing them thereafter. The staged buffer
    /// is allocated once (before CUDA-graph capture, during warmup) and never
    /// grows again for a given node, so replays hit only cache lookups.
    fn staged(&mut self, consts: &[(CUdeviceptr, usize)]) -> Result<Vec<CUdeviceptr>> {
        let matches =
            self.slots.len() == consts.len()
                && self.slots.iter().zip(consts).all(
                    |((src, numel, _), (want_src, want_numel))| {
                        *src == *want_src && *numel == *want_numel
                    },
                );
        if !matches {
            self.rebuild(consts)?;
        }
        Ok(self
            .slots
            .iter()
            .map(|(_, _, offset)| self.ptr + *offset as CUdeviceptr)
            .collect())
    }

    fn rebuild(&mut self, consts: &[(CUdeviceptr, usize)]) -> Result<()> {
        const ALIGN: usize = 256;
        let f16 = std::mem::size_of::<half::f16>();
        let round = |bytes: usize| bytes.div_ceil(ALIGN) * ALIGN;
        let mut slots = Vec::with_capacity(consts.len());
        let mut total = 0usize;
        for (src, numel) in consts {
            slots.push((*src, *numel, total));
            total += round(numel * f16);
        }
        if total > self.cap {
            if self.ptr != 0 {
                // SAFETY: exclusively owned; freed once before replacement.
                unsafe { self.runtime.free_raw(self.ptr)? };
                self.ptr = 0;
            }
            self.ptr = self.runtime.alloc_raw(total.max(1))?;
            self.cap = total;
        }
        for (src, numel, offset) in &slots {
            super::cast::launch_cast_raw(
                &self.runtime,
                cuptr(*src as *const c_void),
                DataType::BFloat16,
                self.ptr + *offset as CUdeviceptr,
                DataType::Float16,
                *numel,
            )?;
        }
        self.slots = slots;
        Ok(())
    }
}

impl Drop for Bf16ConstCache {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: this persistent buffer is exclusively owned by the kernel.
            let _ = unsafe { self.runtime.free_raw(self.ptr) };
            self.ptr = 0;
        }
    }
}

#[derive(Debug)]
pub struct MatMulNBitsKernel {
    runtime: Arc<CudaRuntime>,
    k: usize,
    n: usize,
    bits: usize,
    block_size: usize,
    accuracy_level: i64,
    accuracy4_workspace: Option<Mutex<Accuracy4Workspace>>,
    /// Set when this node's bias input came from folding a standalone `Add`
    /// (see [`crate::optimizer::MATMUL_NBITS_FOLDED_BIAS_ATTR`]). The fp16 GEMV
    /// then reproduces the two-op `fp16(fp16(acc) + bias)` rounding.
    fold_bias_post_round: bool,
    /// Set on a synthetic node produced by
    /// [`crate::optimizer::CudaGateUpSwiGluFusion`]: inputs are
    /// `[x, W_gate, scales_gate, W_up, scales_up]` and the kernel writes
    /// `silu(gate) * up` directly (see [`GATE_UP_SWIGLU_ENTRY`]).
    gate_up_swiglu: bool,
    decomposed_silu: bool,
    /// Set on a general fp16 GEMV whose input activation must be RMS-normalized
    /// in-kernel before the int4 dot, produced by
    /// [`crate::optimizer::CudaSkipRmsNormMatMulFusion`]. The `gamma` weight is
    /// bound at input slot 6 and the kernel reproduces
    /// `skip_rmsnorm_f16_warp_half4` bit-for-bit (see
    /// [`GEMV_F16_SCALES_F16_RMSNORM_ENTRY`]).
    rmsnorm_prologue: bool,
    /// Epsilon copied from the folded `SkipSimplifiedLayerNormalization` node so
    /// the fused prologue reproduces its `1/sqrt(mean_sq + epsilon)`.
    rmsnorm_epsilon: f32,
    last_call_capture_safe: AtomicBool,
    /// Reusable Float16 staging arena for the BFloat16 activation path.
    bf16_scratch: Mutex<Bf16Scratch>,
    /// Persistent Float16 staging of the BFloat16 *constant* inputs (scales,
    /// gamma) so they are cast once rather than re-cast every decode step.
    bf16_const_cache: Mutex<Bf16ConstCache>,
}

impl MatMulNBitsKernel {
    fn run(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        let max_inputs = if self.gate_up_swiglu { 8 } else { 7 };
        if !(3..=max_inputs).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(error(format!(
                "expected 3 to {max_inputs} inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        if inputs[0].dtype == DataType::Float16 {
            if self.gate_up_swiglu {
                return self.run_f16_gate_up_swiglu(inputs, outputs, workspace);
            }
            return self.run_f16(inputs, outputs, workspace);
        }
        if inputs[0].dtype == DataType::BFloat16 {
            return self.run_bf16(inputs, outputs, workspace);
        }
        require_dtype("A", inputs[0].dtype, DataType::Float32)?;
        require_dtype("B", inputs[1].dtype, DataType::Uint8)?;
        require_dtype("scales", inputs[2].dtype, DataType::Float32)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;
        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        if outputs[0].shape != expected_output_shape {
            return Err(error(format!(
                "Y must have shape {expected_output_shape:?}, got {:?}",
                outputs[0].shape
            )));
        }

        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size * self.bits / 8;
        require_shape("B", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        require_flat_or_matrix_shape("scales", inputs[2].shape, self.n, k_blocks)?;

        let zero_points = optional_input(inputs, 3);
        let zp_row_bytes = (k_blocks * self.bits).div_ceil(8);
        if let Some(zp) = zero_points {
            require_dtype("zero_points", zp.dtype, DataType::Uint8)?;
            require_flat_or_matrix_shape("zero_points", zp.shape, self.n, zp_row_bytes)?;
        }

        let group_indices = optional_input(inputs, 4);
        if let Some(g_idx) = group_indices {
            require_dtype("g_idx", g_idx.dtype, DataType::Int32)?;
            if !g_idx.is_contiguous() {
                return Err(error(
                    "g_idx must be contiguous on the CUDA execution provider",
                ));
            }
            let padded_k = k_blocks * self.block_size;
            if g_idx.shape != [self.k] && g_idx.shape != [padded_k] {
                return Err(error(format!(
                    "g_idx must have shape [{}] or [{padded_k}], got {:?}",
                    self.k, g_idx.shape
                )));
            }
            let mut bytes = vec![0u8; g_idx.numel() * 4];
            // SAFETY: `g_idx` is a live contiguous device tensor and `bytes`
            // exactly covers all of its i32 elements.
            unsafe {
                self.runtime
                    .dtoh(&mut bytes, cuptr(g_idx.data_ptr::<u8>() as *const c_void))?
            };
            for (index, value) in bytes.chunks_exact(4).enumerate() {
                let group = i32::from_ne_bytes([value[0], value[1], value[2], value[3]]);
                if group < 0 || group as usize >= k_blocks {
                    return Err(error(format!(
                        "g_idx[{index}]={group} is outside 0..{k_blocks}"
                    )));
                }
            }
        }

        let bias = optional_input(inputs, 5);
        if let Some(bias) = bias {
            require_dtype("bias", bias.dtype, DataType::Float32)?;
            require_shape("bias", bias.shape, &[self.n])?;
        }

        for (name, contiguous) in [
            ("A", inputs[0].is_contiguous()),
            ("B", inputs[1].is_contiguous()),
            ("scales", inputs[2].is_contiguous()),
            (
                "zero_points",
                zero_points.is_none_or(TensorView::is_contiguous),
            ),
            ("g_idx", group_indices.is_none_or(TensorView::is_contiguous)),
            ("bias", bias.is_none_or(TensorView::is_contiguous)),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }

        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let mut flops = (m as u64)
                .saturating_mul(self.n as u64)
                .saturating_mul(self.k as u64)
                .saturating_mul(2);
            if bias.is_some() {
                flops = flops.saturating_add((m as u64).saturating_mul(self.n as u64));
            }
            flops
        });
        self.last_call_capture_safe
            .store(m == 1 && group_indices.is_none(), Ordering::Relaxed);
        if m == 1 && group_indices.is_none() {
            if self.bits == 4
                && self.block_size == 128
                && let Some(zero_points) = zero_points
            {
                // The asymmetric int4/block-128 fp32 path is the dominant Qwen3
                // decode kernel. Its specialized entry preserves the generic
                // GEMV's arithmetic and reduction order while replacing block
                // division/modulo with shifts, hoisting metadata row bases, and
                // loading each warp's packed nibbles as four aligned u32 words.
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_int4_f32_block128",
                    "M==1 decode: bits=4, block_size=128, asymmetric → specialized \
                     shift-indexed int4 f32 GEMV (bit-identical to the generic path)"
                );
                return self.launch_int4_f32_gemv_block128(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                );
            }
            if self.bits == 8 && self.block_size == 128 && zero_points.is_some() {
                // Qwen3-0.6B's dominant decode kernel: int8, block-128,
                // asymmetric (per-block byte zero point), fp32 activations. The
                // generic f32 GEMV below handles this correctly but pays a
                // per-weight integer divide/modulo, a runtime bit-width branch,
                // and repeated scale/zero-point base recomputation. This
                // specialization folds block_size=128 into a shift and hoists the
                // per-column base while preserving each thread's depth order and
                // fp32 reduction, so its output is bit-for-bit identical. Launch
                // geometry is unchanged (one CTA per column, BLOCK_THREADS lanes,
                // no shared memory or arch-specific instructions) so it stays
                // capture-safe and portable across SM counts and CCs.
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_int8_f32_block128",
                    "M==1 decode: bits=8, block_size=128, asymmetric → specialized \
                     shift-indexed int8 f32 GEMV (bit-identical to the generic path)"
                );
                return self.launch_int8_f32_gemv_block128(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                );
            }
            if self.bits == 8 && self.block_size == 32 {
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_int8_f32",
                    "M==1 decode: bits=8, block_size=32 → direct capture-safe f32 GEMV"
                );
                return self.launch_int8_f32_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                );
            }
            if self.bits == 8 {
                // int8 at any non-block-32 power-of-two block size: the tuned
                // int8 f32 GEMV bakes in the block-32 geometry, so route to the
                // model-agnostic general f32 GEMV, which derives the block index
                // from the real block_size and accumulates in fp32. accuracy_level
                // 4 (int8-activation) has no block-size-general kernel, so fp32
                // accumulation is used instead (strictly higher precision than the
                // int8-activation reference), keeping the path capture-safe.
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_f32_general_bs_int8",
                    "M==1 decode: bits=8, block_size={} → model-agnostic f32 GEMV \
                     (fp32 accumulation, any power-of-two block_size)",
                    self.block_size
                );
                return self.launch_f32_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                    blob_size,
                    zp_row_bytes,
                );
            }
            if self.accuracy_level == 4 && self.block_size == 32 && zero_points.is_none() {
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_accuracy4_int8",
                    "M==1 decode: accuracy_level==4, block_size==32, symmetric (no zero_points) \
                     → int8-quantized-activation capture-safe GEMV"
                );
                return self.launch_accuracy4_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    bias,
                    &mut outputs[0],
                    k_blocks,
                );
            }
            if self.bits == 4 && self.accuracy_level == 4 {
                if self.block_size == 32 {
                    onnx_runtime_ep_api::record_kernel_variant!(
                        "gemv_accuracy4_blockwise_int8",
                        "M==1 decode: bits=4, accuracy_level==4, block_size=32 → int8-quantized \
                         activation quantized ONCE then parallelized blockwise GEMV (grid filled \
                         from the device SM count; bit-identical to the tiled accuracy4 GEMM)"
                    );
                    return self.launch_accuracy4_gemv_blockwise(
                        &inputs[0],
                        &inputs[1],
                        &inputs[2],
                        zero_points,
                        bias,
                        &mut outputs[0],
                        k_blocks,
                        blob_size,
                        zp_row_bytes,
                    );
                }
                // accuracy_level 4 requests int8-quantized activations, but the
                // int8 activation quantum is only calibrated at block_size=32
                // (the size the blockwise/tiled accuracy4 kernels bake in). At
                // larger block sizes (e.g. 128) quantizing the activations to
                // int8 discards enough precision to flip razor-thin decode logit
                // ties and diverge from the fp32 reference, so accumulate the
                // activations in fp32 instead. This mirrors the int8/non-block-32
                // path above and the int8/block-128 specialization: fp32
                // activations are strictly higher precision than the
                // int8-activation reference and match the fp32 oracle stream.
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_f32_general_bs_int4",
                    "M==1 decode: bits=4, accuracy_level==4, block_size={} (≠32) → \
                     model-agnostic f32 GEMV (fp32 activations, higher precision than the \
                     int8-activation reference)",
                    self.block_size
                );
                return self.launch_f32_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                    blob_size,
                    zp_row_bytes,
                );
            }
            if self.accuracy_level != 4 {
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_f32",
                    "M==1 decode: accuracy_level={} (non-accuracy4) → direct f32 GEMV",
                    self.accuracy_level
                );
                return self.launch_f32_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                    blob_size,
                    zp_row_bytes,
                );
            }
        }
        if self.bits == 4
            && self.accuracy_level == 4
            && self.block_size == 32
            && group_indices.is_none()
        {
            onnx_runtime_ep_api::record_kernel_variant!(
                "gemm_tiled_accuracy4",
                "M={} (GEMV requires M==1), accuracy_level==4, block_size=32, no g_idx → \
                 tiled accuracy4 GEMM (int8-quantized activations)",
                m
            );
            return self.launch_accuracy4(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                zero_points,
                bias,
                &mut outputs[0],
                m,
                k_blocks,
                blob_size,
                zp_row_bytes,
            );
        }

        onnx_runtime_ep_api::record_kernel_variant!(
            "dequant_cublas_gemm",
            "M={}, accuracy_level={}, g_idx={} → dequantize weights to f32 then cuBLAS GEMM \
             (general prefill / grouped path)",
            m,
            self.accuracy_level,
            group_indices.is_some()
        );

        let weight = self.runtime.alloc_raw(self.k * self.n * 4)?;
        let result = self
            .launch_dequant(
                &inputs[1],
                &inputs[2],
                zero_points,
                group_indices,
                weight,
                k_blocks,
                blob_size,
                zp_row_bytes,
            )
            .and_then(|()| {
                let params = GemmParams {
                    dtype: GemmDtype::F32,
                    a: cuptr(inputs[0].data_ptr::<u8>() as *const c_void),
                    b: weight,
                    c: cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void),
                    m,
                    k: self.k,
                    n: self.n,
                    batch: 1,
                    a_batch_stride: m * self.k,
                    b_batch_stride: 0,
                    epilogue: bias.map(|bias| GemmEpilogue {
                        kind: GemmEpilogueKind::Bias,
                        bias: cuptr(bias.data_ptr::<u8>() as *const c_void),
                    }),
                };
                // SAFETY: validated dense f32 A/Y and the dequantized [K,N]
                // allocation cover the complete GEMM; workspace and stream live
                // through the call and Y aliases neither input.
                unsafe {
                    blas::governed_gemm(
                        self.runtime.blas(),
                        self.runtime.stream_ptr(),
                        &params,
                        workspace,
                        "MatMulNBits",
                    )
                }
            })
            .and_then(|()| self.runtime.synchronize());

        // SAFETY: `weight` came from `alloc_raw` and is released once, after all
        // submitted work has synchronized (or the submission failed).
        let free_weight = unsafe { self.runtime.free_raw(weight) };
        result.and(free_weight)
    }

    fn uses_dequant_cublas_workspace(
        &self,
        dtype: DataType,
        a_shape: &[usize],
        group_indices_present: bool,
    ) -> bool {
        if dtype != DataType::Float32 || a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k
        {
            return false;
        }
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        if m == 1 && !group_indices_present {
            return false;
        }
        !(self.bits == 4
            && self.accuracy_level == 4
            && self.block_size == 32
            && !group_indices_present)
    }

    /// Whether a half-precision activation at M>1 would take the dequantize +
    /// fp16 cuBLASLt tensor-core GEMM, which needs the same declared workspace
    /// the f32 fallback does. BFloat16 counts: it stages through Float16 and
    /// lands on the very same path, carrying this workspace with it.
    fn uses_dequant_f16_cublas_workspace(
        &self,
        dtype: DataType,
        a_shape: &[usize],
        group_indices_present: bool,
    ) -> bool {
        if !matches!(dtype, DataType::Float16 | DataType::BFloat16)
            || a_shape.is_empty()
            || a_shape[a_shape.len() - 1] != self.k
            || group_indices_present
        {
            return false;
        }
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        // Deliberately looser than the launch-site gates: this only has to be an
        // upper bound. Declaring a workspace a call turns out not to need costs
        // nothing, whereas under-declaring makes `governed_gemm_ex` fail at
        // launch — after the dequantize has already run — and silently fall back.
        dequant_f16_gemm_enabled()
            && m >= dequant_f16_gemm_min_m()
            && self.k.saturating_mul(self.n).saturating_mul(2)
                <= dequant_f16_gemm_max_scratch_bytes()
    }

    fn workspace_requirement_for(
        &self,
        inputs: &[TensorMetadata<'_>],
    ) -> Result<WorkspaceRequirement> {
        let Some(a) = inputs.first() else {
            return Ok(WorkspaceRequirement::NONE);
        };
        let group_indices_present = inputs.get(4).is_some_and(|input| input.present);
        // The fused gate/up node has no group-index or bias slot: it spends
        // inputs 3..=4 on the up projection's packed weights and scales, and
        // 5..=6 on the two zero-point tensors. Reading them positionally, as the
        // unfused node does, makes every fused node look group-indexed and
        // bias-carrying, which silently denies it a workspace.
        let fused = self.gate_up_swiglu;
        let dtype = if self.uses_dequant_cublas_workspace(a.dtype, a.shape, group_indices_present) {
            GemmDtype::F32
        } else if self.uses_dequant_f16_cublas_workspace(
            a.dtype,
            a.shape,
            !fused && group_indices_present,
        ) {
            GemmDtype::F16
        } else {
            return Ok(WorkspaceRequirement::NONE);
        };
        let m = a.shape[..a.shape.len() - 1].iter().product::<usize>();
        if dtype == GemmDtype::F16 {
            // Take the larger of the bias and no-bias plans rather than trying to
            // predict which the launch will pick: the requirement only has to be
            // an upper bound, and under-declaring fails the launch outright.
            let mut bytes = 0;
            let biases: &[Option<u64>] = if fused { &[None] } else { &[None, Some(0)] };
            for bias in biases {
                let params = self.dequant_f16_gemm_ex(1, 1, 1, m, *bias);
                bytes = bytes.max(blas::gemm_ex_workspace_bytes(self.runtime.blas(), &params)?);
            }
            return Ok(blas::governed_workspace_requirement(bytes));
        }
        let params = GemmParams {
            dtype,
            a: 1,
            b: 1,
            c: 1,
            m,
            k: self.k,
            n: self.n,
            batch: 1,
            a_batch_stride: m * self.k,
            b_batch_stride: 0,
            epilogue: inputs
                .get(5)
                .filter(|bias| bias.present)
                .map(|_| GemmEpilogue {
                    kind: GemmEpilogueKind::Bias,
                    bias: 0,
                }),
        };
        let bytes = blas::gemm_workspace_bytes(self.runtime.blas(), &params)?;
        Ok(blas::governed_workspace_requirement(bytes))
    }

    /// BFloat16-activation path. The tuned int4/int8 GEMV/GEMM kernels are
    /// implemented for Float16 and Float32 activations only, so BFloat16 inputs
    /// are staged through Float16: every BFloat16 operand (activation, scales,
    /// and any fused residual/gamma) is converted into a scratch Float16 buffer,
    /// the reused fp16 path runs the actual matmul against the still-quantized
    /// int4/int8 weights, and the Float16 result is converted back to BFloat16.
    ///
    /// Only the small activation/scale/output buffers are converted; the packed
    /// weights (the memory-bound term in decode) keep their int4/int8 layout, so
    /// the extra work is a handful of pointwise casts per node. Float16 losslessly
    /// represents the post-normalization magnitudes MatMulNBits consumes here.
    /// The per-call scratch means this path is not graph-capture safe.
    fn run_bf16(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        require_dtype("A", inputs[0].dtype, DataType::BFloat16)?;
        require_dtype("Y", outputs[0].dtype, DataType::BFloat16)?;

        // The scale slots are immutable weights (general path: input 2; the
        // gate/up SwiGLU fusion adds a second scale at input 4). They dominate
        // the BFloat16 staging traffic (~25% of the int4 weight bytes/token for
        // Muse-Glimmer), so stage them once into the persistent const cache and
        // reuse across decode steps. Every other BFloat16 input is either the
        // per-step activation (input 0) or a per-token residual bound into the
        // bias slot — both genuinely dynamic — so they keep going through the
        // per-call arena. Caching keys on the weight pointer + element count and
        // never uses pointer identity for the dynamic slots (a reused activation
        // buffer has a stable pointer but changing contents).
        let cache_slots: &[usize] = if self.gate_up_swiglu { &[2, 4] } else { &[2] };
        let is_cached = |index: usize, dtype: DataType| {
            dtype == DataType::BFloat16 && cache_slots.contains(&index)
        };

        // Lay out per-call Float16 regions inside the reusable arena: one region
        // per dynamic BFloat16 input, plus one for the Float16 result. Regions
        // are padded so each begins aligned for the tuned fp16 kernels'
        // vectorized loads.
        const ALIGN: usize = 256;
        let f16 = std::mem::size_of::<half::f16>();
        let round = |bytes: usize| bytes.div_ceil(ALIGN) * ALIGN;

        let mut offsets: Vec<Option<usize>> = Vec::with_capacity(inputs.len());
        let mut total = 0usize;
        for (index, input) in inputs.iter().enumerate() {
            if input.dtype == DataType::BFloat16 && !is_cached(index, input.dtype) {
                offsets.push(Some(total));
                total += round(input.numel() * f16);
            } else {
                offsets.push(None);
            }
        }
        let out_off = total;
        let out_n = outputs[0].numel();
        total += round(out_n * f16);

        // Stage the constant scale slots once into the persistent cache (this is
        // allocation-free on every call after the first warmup call, so the
        // captured decode graph only replays cache hits).
        let cached: Vec<(usize, CUdeviceptr)> = {
            let consts: Vec<(CUdeviceptr, usize)> = cache_slots
                .iter()
                .filter(|index| **index < inputs.len() && is_cached(**index, inputs[**index].dtype))
                .map(|index| {
                    (
                        cuptr(inputs[*index].data_ptr::<u8>() as *const c_void) as CUdeviceptr,
                        inputs[*index].numel(),
                    )
                })
                .collect();
            if consts.is_empty() {
                Vec::new()
            } else {
                let mut cache = self
                    .bf16_const_cache
                    .lock()
                    .map_err(|_| error("MatMulNBits bf16 const cache mutex poisoned"))?;
                let ptrs = cache.staged(&consts)?;
                cache_slots
                    .iter()
                    .filter(|index| {
                        **index < inputs.len() && is_cached(**index, inputs[**index].dtype)
                    })
                    .zip(ptrs)
                    .map(|(index, ptr)| (*index, ptr))
                    .collect()
            }
        };

        // Hold the arena for the whole call: the staging casts, the reused fp16
        // matmul, and the narrowing cast all run stream-ordered against it. The
        // fp16 path never touches `bf16_scratch`, so this cannot deadlock.
        let mut arena = self
            .bf16_scratch
            .lock()
            .map_err(|_| error("MatMulNBits bf16 scratch mutex poisoned"))?;
        let base = arena.ensure(total)?;

        let mut f16_inputs: Vec<TensorView> = Vec::with_capacity(inputs.len());
        for (index, (input, offset)) in inputs.iter().zip(offsets.iter()).enumerate() {
            if let Some((_, cached_ptr)) = cached.iter().find(|(slot, _)| *slot == index) {
                f16_inputs.push(TensorView::new(
                    DevicePtr(raw_ptr(*cached_ptr) as *const c_void),
                    DataType::Float16,
                    input.shape,
                    input.strides,
                    input.device,
                ));
                continue;
            }
            match offset {
                Some(off) => {
                    let ptr = base + *off as CUdeviceptr;
                    super::cast::launch_cast_raw(
                        &self.runtime,
                        cuptr(input.data_ptr::<u8>() as *const c_void),
                        DataType::BFloat16,
                        ptr,
                        DataType::Float16,
                        input.numel(),
                    )?;
                    f16_inputs.push(TensorView::new(
                        DevicePtr(raw_ptr(ptr) as *const c_void),
                        DataType::Float16,
                        input.shape,
                        input.strides,
                        input.device,
                    ));
                }
                None => f16_inputs.push(*input),
            }
        }

        let out_ptr = base + out_off as CUdeviceptr;
        let out_shape = outputs[0].shape;
        let out_strides = outputs[0].strides;
        let out_device = outputs[0].device;
        let mut y_f16 = TensorMut::new(
            DevicePtrMut(raw_ptr(out_ptr)),
            DataType::Float16,
            out_shape,
            out_strides,
            out_device,
        );
        // Offer the real bf16 output to the fp16 path: the split-K decode GEMVs
        // can narrow into it directly and save this node's staging cast. The
        // offer is restored (not just cleared) so a nested bf16 call cannot
        // strand an outer one, and it is restored before `?` so an error cannot
        // leave it published.
        let dst_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let offer = gemv_bf16_direct_out_enabled().then_some(Bf16DirectOut {
            staging: out_ptr,
            dst: dst_ptr,
            taken: false,
        });
        let previous = BF16_DIRECT_OUT.with(|cell| cell.replace(offer));
        let run_result = self.run(&f16_inputs, std::slice::from_mut(&mut y_f16), workspace);
        let settled = BF16_DIRECT_OUT.with(|cell| cell.replace(previous));
        run_result?;
        if !settled.is_some_and(|offer| offer.taken) {
            super::cast::launch_cast_raw(
                &self.runtime,
                out_ptr,
                DataType::Float16,
                dst_ptr,
                DataType::BFloat16,
                out_n,
            )?;
        }
        // Capture-safety is inherited from the staged fp16 run: the M==1 decode
        // GEMV is capture-safe, and the staging casts add only allocation-free,
        // sync-free stream launches against the persistent arena (which is grown
        // once, before capture). Prefill (M>1) leaves the flag cleared. So the
        // flag `run` set during recursion is exactly right — do not override it.
        drop(arena);
        Ok(())
    }

    /// Direct fp16-activation x int4/int8-weight path. Scales may be fp16 or
    /// f32. M=1 uses the capture-safe decode GEMVs; M>1 uses a portable tiled
    /// CUDA-core GEMM with fp32 accumulation and fp16 output.
    fn run_f16(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        require_dtype("A", inputs[0].dtype, DataType::Float16)?;
        require_dtype("B", inputs[1].dtype, DataType::Uint8)?;
        let scales_fp16 = match inputs[2].dtype {
            DataType::Float16 => true,
            DataType::Float32 => false,
            other => {
                return Err(error(format!(
                    "scales must have dtype Float16 or Float32 for fp16 activations, got {other:?}"
                )));
            }
        };
        require_dtype("Y", outputs[0].dtype, DataType::Float16)?;

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        if outputs[0].shape != expected_output_shape {
            return Err(error(format!(
                "Y must have shape {expected_output_shape:?}, got {:?}",
                outputs[0].shape
            )));
        }

        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size * self.bits / 8;
        require_shape("B", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        require_flat_or_matrix_shape("scales", inputs[2].shape, self.n, k_blocks)?;

        let zero_points = optional_input(inputs, 3);
        let zp_row_bytes = (k_blocks * self.bits).div_ceil(8);
        if let Some(zero_points) = zero_points {
            require_dtype("zero_points", zero_points.dtype, DataType::Uint8)?;
            require_flat_or_matrix_shape("zero_points", zero_points.shape, self.n, zp_row_bytes)?;
        }
        let group_indices = optional_input(inputs, 4);
        let bias = optional_input(inputs, 5);
        let rows = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        if let Some(bias) = bias {
            require_dtype("bias", bias.dtype, DataType::Float16)?;
            // A folded residual epilogue binds the residual activation into this
            // same slot: `[1, 1, N]` (N elements) at decode, or `[1, S, N]`
            // (rows * N elements) at prefill. A genuine broadcast bias is `[N]`.
            if bias.numel() != self.n && bias.numel() != rows * self.n {
                return Err(error(format!(
                    "bias must have {} elements (broadcast [N]) or {} elements (per-token \
                     [1, S, N] residual), got {:?}",
                    self.n,
                    rows * self.n,
                    bias.shape
                )));
            }
        }
        let gamma = optional_input(inputs, 6);
        if self.rmsnorm_prologue {
            let gamma = gamma.ok_or_else(|| {
                error("rmsnorm_prologue fusion requires the normalization weight at input 6")
            })?;
            require_gamma_dtype(gamma.dtype)?;
            require_shape("gamma", gamma.shape, &[self.k])?;
        }

        for (name, contiguous) in [
            ("A", inputs[0].is_contiguous()),
            ("B", inputs[1].is_contiguous()),
            ("scales", inputs[2].is_contiguous()),
            (
                "zero_points",
                zero_points.is_none_or(TensorView::is_contiguous),
            ),
            ("bias", bias.is_none_or(TensorView::is_contiguous)),
            ("gamma", gamma.is_none_or(TensorView::is_contiguous)),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }

        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let mut flops = (m as u64)
                .saturating_mul(self.n as u64)
                .saturating_mul(self.k as u64)
                .saturating_mul(2);
            if bias.is_some() {
                flops = flops.saturating_add((m as u64).saturating_mul(self.n as u64));
            }
            if self.rmsnorm_prologue {
                let elements = (m as u64).saturating_mul(self.k as u64);
                flops = flops
                    .saturating_add(elements.saturating_mul(4))
                    .saturating_add((m as u64).saturating_mul(4));
            }
            flops
        });
        // Non-block-32 layouts are served by the model-agnostic general-block-size
        // fp16 kernels (int4/int8 decode GEMV + int4/int8 prefill GEMM). The tuned
        // block-32 fusions (rmsnorm prologue, gate/up SwiGLU, down-projection) are
        // gated to block_size==32 in the optimizer, so a non-block-32 node always
        // arrives here as a plain int4/int8 GEMV/GEMM. The general-block-size GEMV
        // selects the packed layout from `bits`, so bits==8 with a non-block-32
        // width routes to it just like int4 does.
        if group_indices.is_some() {
            return Err(error(
                "MatMulNBits CUDA fp16 activations do not support g_idx. Why: the block-32 fp16 \
                 kernels map each K block directly to its scale and zero point and do not implement \
                 group remapping. How to fix: omit g_idx, provide f32 activations, or select another \
                 execution provider",
            ));
        }

        if m > 1 {
            // Small-batch decode fast path: for M within the crossover bound,
            // reuse the capture-safe single-row decode GEMV once per row instead
            // of the tiled prefill GEMM. The tiled GEMM tiles M in 16-row blocks,
            // so M=2..16 pay the same M-independent full-weight-grid pass; the
            // looped GEMV instead reads the weights once per row and skips that
            // fixed overhead, which is faster while M x gemv_step < tiled_step
            // (measured crossover ~11 on qwen05b-q4; see DECODE_GEMV_LOOP_MAX_M_*).
            // Excludes the fused SwiGLU/decomposed-SiLU epilogues, which have no
            // single-row GEMV equivalent in `dispatch_f16_decode_gemv_row`.
            //
            // Numerics: each row is computed byte-identically to a single-sequence
            // (M==1) decode of that row — the strongest batching-correctness
            // guarantee (a batched row produces the same tokens as if run alone).
            // Capture-safe: every per-row launch is static-grid, allocation- and
            // sync-free, so the batch decode graph captures and replays it.
            // Streaming: the resident weight is read M times from VRAM, not
            // re-streamed, so the weight-offload HtoD 1/N amortization is intact.
            if m <= decode_gemv_loop_max_m() && !self.gate_up_swiglu && !self.decomposed_silu {
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_f16_batched_loop",
                    "M={} small-batch decode: {} single-row decode GEMV launches (one per row), \
                     each byte-identical to M==1 decode; skips the tiled prefill GEMM's \
                     M-independent full-weight-grid pass",
                    m,
                    m
                );
                self.last_call_capture_safe.store(true, Ordering::Relaxed);
                let per_token_bias =
                    bias.filter(|b| b.numel() == m * self.n && m * self.n != self.n);
                let a_base = inputs[0].data_ptr::<u8>();
                let y_base = outputs[0].data_ptr_mut::<u8>();
                let bias_base = per_token_bias.map(|b| b.data_ptr::<u8>());
                let a_row_shape = [1usize, self.k];
                let a_row_strides = [self.k as i64, 1];
                let y_row_shape = [1usize, self.n];
                let y_row_strides = [self.n as i64, 1];
                let a_row_bytes = self.k * 2; // fp16 activation
                let y_row_bytes = self.n * 2; // fp16 output / residual
                for row in 0..m {
                    let a_row = TensorView::new(
                        DevicePtr(a_base.wrapping_add(row * a_row_bytes) as *const c_void),
                        DataType::Float16,
                        &a_row_shape,
                        &a_row_strides,
                        inputs[0].device,
                    );
                    let mut y_row = TensorMut::new(
                        DevicePtrMut(y_base.wrapping_add(row * y_row_bytes) as *mut c_void),
                        DataType::Float16,
                        &y_row_shape,
                        &y_row_strides,
                        outputs[0].device,
                    );
                    let bias_row = bias_base.map(|base| {
                        TensorView::new(
                            DevicePtr(base.wrapping_add(row * y_row_bytes) as *const c_void),
                            DataType::Float16,
                            &y_row_shape,
                            &y_row_strides,
                            inputs[0].device,
                        )
                    });
                    let bias_ref = match bias_row {
                        Some(ref b) => Some(b),
                        None => bias,
                    };
                    self.dispatch_f16_decode_gemv_row(
                        &a_row,
                        &inputs[1],
                        &inputs[2],
                        scales_fp16,
                        zero_points,
                        bias_ref,
                        gamma,
                        &mut y_row,
                        k_blocks,
                        blob_size,
                        zp_row_bytes,
                    )?;
                }
                return Ok(());
            }
            // SAFETY: the tiled prefill kernel itself has fixed pointers and no
            // allocation or host synchronization. We nevertheless keep the
            // advertised capture contract conservative: variable-M prefill is
            // outside the persistent M=1 decode graph and has no replay coverage.
            self.last_call_capture_safe.store(false, Ordering::Relaxed);
            // A folded residual epilogue supplies a per-token residual (rows * N
            // elements) in the bias slot; index it with row stride N. A genuine
            // broadcast bias (N elements) keeps stride 0.
            let bias_row_stride = match bias {
                Some(bias) if bias.numel() == m * self.n && m * self.n != self.n => self.n,
                _ => 0,
            };
            // Dequantize to fp16 and run the GEMM on cuBLASLt tensor cores. This
            // sits ahead of Marlin because both of the int4 M>1 kernels below
            // leave most of the machine idle on this shape: Marlin stages
            // nothing in shared memory, so every warp re-reads its whole A panel
            // from global memory, and the f32 dequant fallback gives up tensor
            // cores. One K*N dequantize pass buys the full fp16 tensor-core rate
            // for the matmul itself, which prefill is large enough to amortize.
            // Ineligibility and launch errors fall through to Marlin and then
            // the tiled GEMM (Rule 11 fallback contract).
            if dequant_f16_gemm_enabled()
                && m >= dequant_f16_gemm_min_m()
                && !self.gate_up_swiglu
                && !self.decomposed_silu
                && !self.rmsnorm_prologue
                && group_indices.is_none()
            {
                match self.try_dequant_f16_cublas_gemm(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    scales_fp16,
                    zero_points,
                    bias,
                    &mut outputs[0],
                    m,
                    bias_row_stride,
                    k_blocks,
                    blob_size,
                    zp_row_bytes,
                    workspace,
                ) {
                    Ok(true) => {
                        onnx_runtime_ep_api::record_kernel_variant!(
                            "gemm_dequant_f16_cublas",
                            "M={} prefill: dequantize int{} weights to [K, N] fp16, then \
                             cuBLASLt fp16 tensor-core GEMM with f32 accumulation",
                            m,
                            self.bits
                        );
                        return Ok(());
                    }
                    Ok(false) => {
                        // Not eligible at runtime (bias form, scratch size).
                    }
                    Err(_err) => {
                        // Hard launch error; fall through rather than fail the op.
                    }
                }
            }
            // Opt-in Marlin int4 tensor-core GEMM for the M>1 path. Gated on
            // SM80+, int4, no g_idx (checked above), no fused SwiGLU/RMSNorm
            // epilogue yet, and `ONNX_GENAI_MARLIN_M_GT_1`. On any ineligibility
            // or runtime error it falls through to the byte-identical portable
            // tiled GEMM below (Rule 11 fallback contract).
            if marlin_gemm::marlin_m_gt_1_enabled()
                && self.bits == 4
                && !self.gate_up_swiglu
                && !self.decomposed_silu
                && !self.rmsnorm_prologue
                && marlin_gemm::device_supports_marlin(
                    self.runtime.capabilities().compute_capability(),
                )
            {
                match self.try_launch_marlin_gemm(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    scales_fp16,
                    zero_points,
                    bias,
                    &mut outputs[0],
                    m,
                    bias_row_stride,
                ) {
                    Ok(Some(warm)) => {
                        // Static grid + (when warm) no alloc/repack/sync this
                        // call ⇒ capture-safe. A cold first call repacked and is
                        // reported unsafe; subsequent replays hit the cache.
                        self.last_call_capture_safe.store(warm, Ordering::Relaxed);
                        onnx_runtime_ep_api::record_kernel_variant!(
                            "gemm_marlin_int4",
                            "M={} prefill/verify: fp16 activation, int4, block_size={}, \
                             zero_points={} → Marlin SM80 mma.sync int4 tensor-core GEMM \
                             (static grid, capture-safe when weights are pre-repacked)",
                            m,
                            self.block_size,
                            zero_points.is_some()
                        );
                        return Ok(());
                    }
                    Ok(None) => {
                        // Not eligible at runtime (e.g. dims); fall through.
                    }
                    Err(_err) => {
                        // Hard launch error: fall through to the portable tiled
                        // GEMM rather than fail the op. The tiled variant records
                        // its own kernel-variant note below.
                    }
                }
            }
            if self.rmsnorm_prologue {
                let gamma = gamma.ok_or_else(|| {
                    error("rmsnorm_prologue fusion requires the normalization weight at input 6")
                })?;
                // Opt-in Marlin int4 tensor-core GEMM with the fused RMS-norm
                // prologue: stage the per-token normalized activation into
                // scratch (byte-identical to the standalone prologue), then run
                // Marlin over it. The scratch allocation keeps this off the
                // advertised capture contract (like the tiled rmsnorm prefill),
                // but prefill is outside the persistent decode graph. Falls
                // through to the tiled rmsnorm GEMM on ineligibility / error.
                if marlin_gemm::marlin_m_gt_1_enabled()
                    && self.bits == 4
                    && !self.gate_up_swiglu
                    && !self.decomposed_silu
                    && marlin_gemm::device_supports_marlin(
                        self.runtime.capabilities().compute_capability(),
                    )
                {
                    match self.try_launch_marlin_gemm_rmsnorm(
                        &inputs[0],
                        &inputs[1],
                        &inputs[2],
                        scales_fp16,
                        zero_points,
                        gamma,
                        bias,
                        &mut outputs[0],
                        m,
                        bias_row_stride,
                    ) {
                        Ok(Some(warm)) => {
                            // Static grid + pooled scratch/weights ⇒ capture-safe
                            // on a warm replay; a cold call allocated and is
                            // reported unsafe.
                            self.last_call_capture_safe.store(warm, Ordering::Relaxed);
                            onnx_runtime_ep_api::record_kernel_variant!(
                                "gemm_marlin_int4_rmsnorm",
                                "M={} prefill/verify: RMS-normalization prologue \
                                 (SkipSimplifiedLayerNormalization folded) into pooled scratch, \
                                 then Marlin SM80 mma.sync int4 tensor-core GEMM (capture-safe \
                                 when weights + scratch are pre-warmed)",
                                m
                            );
                            return Ok(());
                        }
                        Ok(None) => {
                            // Not eligible (e.g. dims); fall through to tiled.
                        }
                        Err(_err) => {
                            // Hard error: fall through to the tiled rmsnorm GEMM.
                        }
                    }
                }
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemm_f16_tiled_rmsnorm",
                    "M={} prefill: RMS-normalization prologue (SkipSimplifiedLayerNormalization \
                     folded) into a per-token scratch, then portable 16x16 tiled GEMM with fp32 \
                     accumulation; not advertised as CUDA-graph capture-safe",
                    m
                );
                return self.launch_f16_gemm_rmsnorm_prefill(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    gamma,
                    bias,
                    &mut outputs[0],
                    m,
                    k_blocks,
                    bias_row_stride,
                );
            }
            onnx_runtime_ep_api::record_kernel_variant!(
                "gemm_f16_tiled",
                "M={} prefill: fp16 activation, bits={}, block_size={}, zero_points={}, \
                 scales={} → portable 16x16 CUDA-core tiled GEMM with fp32 accumulation; \
                 not advertised as CUDA-graph capture-safe",
                m,
                self.bits,
                self.block_size,
                zero_points.is_some(),
                if scales_fp16 { "fp16" } else { "fp32" }
            );
            return self.launch_f16_gemm(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                scales_fp16,
                zero_points,
                bias,
                &mut outputs[0],
                m,
                k_blocks,
                blob_size,
                bias_row_stride,
            );
        }

        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        self.dispatch_f16_decode_gemv_row(
            &inputs[0],
            &inputs[1],
            &inputs[2],
            scales_fp16,
            zero_points,
            bias,
            gamma,
            &mut outputs[0],
            k_blocks,
            blob_size,
            zp_row_bytes,
        )
    }

    /// Dispatch one single-row (M==1) fp16-activation decode GEMV for `a_row`
    /// into `y_row`, selecting the specialized kernel by bits/block_size/rmsnorm
    /// exactly as the M==1 path does. `packed`/`scales`/`zero_points`/`gamma` are
    /// the shared weight metadata; `bias` is the row's bias/residual slice (or a
    /// broadcast `[N]` bias). Reused by [`Self::run_f16`] for the true M==1 step
    /// and by the small-batch looped decode path (one call per row).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_f16_decode_gemv_row(
        &self,
        a_row: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        gamma: Option<&TensorView>,
        y_row: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        if self.bits == 8 && self.block_size == 32 {
            if self.rmsnorm_prologue {
                let gamma = gamma.ok_or_else(|| {
                    error("rmsnorm_prologue fusion requires the normalization weight at input 6")
                })?;
                if !scales_fp16 {
                    return Err(error(
                        "rmsnorm_prologue fusion requires fp16 scales (the fused kernel replicates \
                         the fp16 general scales path)",
                    ));
                }
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gemv_int8_f16_scales_f16_rmsnorm",
                    "M==1 decode: fp16 activation, bits=8, block_size=32, fp16 scales, \
                     zero_points={} → int8 GEMV with fused RMS-normalization prologue \
                     (SkipSimplifiedLayerNormalization folded)",
                    zero_points.is_some()
                );
                return self.launch_int8_f16_gemv_rmsnorm(
                    a_row,
                    packed,
                    scales,
                    zero_points,
                    gamma,
                    bias,
                    y_row,
                    k_blocks,
                );
            }
            onnx_runtime_ep_api::record_kernel_variant!(
                "gemv_int8_f16",
                "M==1 decode: fp16 activation, bits=8, block_size=32, zero_points={} → direct \
                 capture-safe GEMV",
                zero_points.is_some()
            );
            return self.launch_int8_f16_gemv(
                a_row,
                packed,
                scales,
                scales_fp16,
                zero_points,
                bias,
                y_row,
                k_blocks,
            );
        }
        if self.rmsnorm_prologue {
            let gamma = gamma.ok_or_else(|| {
                error("rmsnorm_prologue fusion requires the normalization weight at input 6")
            })?;
            if !scales_fp16 {
                return Err(error(
                    "rmsnorm_prologue fusion requires fp16 scales (the fused kernel replicates \
                     the fp16 general scales path)",
                ));
            }
            onnx_runtime_ep_api::record_kernel_variant!(
                "gemv_f16_scales_f16_rmsnorm",
                "M==1 decode: fp16 activation, bits=4, block_size=32, fp16 scales, \
                 zero_points={} → general GEMV with fused RMS-normalization prologue \
                 (SkipSimplifiedLayerNormalization folded)",
                zero_points.is_some()
            );
            return self.launch_f16_gemv_rmsnorm(
                a_row,
                packed,
                scales,
                zero_points,
                gamma,
                bias,
                y_row,
                k_blocks,
                blob_size,
                zp_row_bytes,
            );
        }
        self.launch_f16_gemv(
            a_row,
            packed,
            scales,
            scales_fp16,
            zero_points,
            bias,
            y_row,
            k_blocks,
            blob_size,
            zp_row_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Attempt the opt-in Marlin int4 tensor-core GEMM for the M>1 path.
    ///
    /// Returns `Ok(Some(warm))` when the kernel launched (`warm == true` means
    /// the repacked weights were already cached, so this call did no allocation
    /// and is CUDA-graph capture-safe), `Ok(None)` when the inputs are not
    /// eligible (caller falls through to the portable tiled GEMM), or `Err` on a
    /// hard launch failure (caller also falls through).
    ///
    /// Eligibility (SM80, int4, no fused SwiGLU/RMSNorm epilogue, opt-in flag)
    /// is checked by the caller; here we validate the numeric shape contract
    /// (`K` divisible by 16 and by the group size) and ensure the weights are
    /// repacked into the tensor-core layout exactly once.
    #[allow(clippy::too_many_arguments)]
    fn try_launch_marlin_gemm(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        bias_row_stride: usize,
    ) -> Result<Option<bool>> {
        // Shape contract: the tensor-core kernel tiles K in units of 16 and
        // applies one scale per quantization group, so K must be divisible by
        // both. Anything else stays on the portable tiled GEMM.
        if self.block_size == 0
            || !self.k.is_multiple_of(16)
            || !self.k.is_multiple_of(self.block_size)
        {
            return Ok(None);
        }
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits Marlin int4 tensor-core GEMM")?;

        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let (weights_ptr, warm) = marlin_gemm::ensure_repacked(
            &self.runtime,
            packed_ptr,
            self.n,
            self.k,
            self.block_size,
        )?;

        let args = marlin_gemm::MarlinGemmArgs {
            activation: cuptr(activation.data_ptr::<u8>() as *const c_void),
            weights: weights_ptr,
            scales: cuptr(scales.data_ptr::<u8>() as *const c_void),
            zero_points: zero_points.map(|t| cuptr(t.data_ptr::<u8>() as *const c_void)),
            bias: bias.map(|t| cuptr(t.data_ptr::<u8>() as *const c_void)),
            output: cuptr(output.data_ptr_mut::<u8>() as *const c_void),
            m,
            k: self.k,
            n: self.n,
            group_size: self.block_size,
            scales_fp16,
            bias_post_round: self.fold_bias_post_round && bias.is_some(),
            bias_row_stride,
        };
        // Split-K (opt-in): partition K across grid.z to fill idle SMs when the
        // base block count is small. Partials come from the capture-safe scratch
        // pool (slot 4) so warm replays stay allocation-free; the reduce applies
        // the same fold_bias epilogue. Falls back to the byte-identical direct
        // kernel whenever the heuristic declines to split.
        let split_warm = self.maybe_launch_marlin_splitk(&args)?;
        Ok(Some(warm && split_warm))
    }

    /// Launch `args` through split-K when enabled and the heuristic elects to
    /// split, otherwise through the direct kernel. Returns whether the launch
    /// was allocation-free (`true` for the direct kernel; for split-K, whether
    /// the pooled partials scratch was already warm) so callers can propagate
    /// capture-safety.
    fn maybe_launch_marlin_splitk(&self, args: &marlin_gemm::MarlinGemmArgs) -> Result<bool> {
        if marlin_gemm::marlin_splitk_enabled() {
            let k_blocks = self.k / self.block_size.max(1);
            let sm = self.runtime.capabilities().multiprocessor_count();
            let split_k = marlin_gemm::choose_split_k(args.m, args.n, k_blocks, sm);
            if split_k > 1 {
                let bytes = marlin_gemm::splitk_partials_len(split_k, args.m, args.n)
                    * std::mem::size_of::<f32>();
                let (partials, warm) = marlin_gemm::ensure_scratch(&self.runtime, 4, bytes)?;
                marlin_gemm::launch_marlin_gemm_splitk(&self.runtime, args, split_k, partials)?;
                return Ok(warm);
            }
        }
        marlin_gemm::launch_marlin_gemm(&self.runtime, args)?;
        Ok(true)
    }

    /// Marlin M>1 GEMM with the fused RMS-normalization prologue. Stages the
    /// per-token normalized activation into pooled scratch (byte-identical to the
    /// standalone `launch_rmsnorm_prefill` output the tiled path uses), then runs
    /// the Marlin int4 tensor-core GEMM over the normalized rows. Returns
    /// `Some(warm)` when it launched (`warm` means both the repacked weights and
    /// the scratch were already pooled ⇒ no allocation this call ⇒ capture-safe),
    /// or `None` when the shape is ineligible (caller falls through to the tiled
    /// rmsnorm GEMM).
    #[allow(clippy::too_many_arguments)]
    fn try_launch_marlin_gemm_rmsnorm(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        gamma: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        bias_row_stride: usize,
    ) -> Result<Option<bool>> {
        if self.block_size == 0
            || !self.k.is_multiple_of(16)
            || !self.k.is_multiple_of(self.block_size)
        {
            return Ok(None);
        }
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits Marlin int4 tensor-core GEMM (rmsnorm)")?;

        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let (weights_ptr, weights_warm) = marlin_gemm::ensure_repacked(
            &self.runtime,
            packed_ptr,
            self.n,
            self.k,
            self.block_size,
        )?;
        // Pooled normalized-activation scratch (slot 0). On a warm replay this is
        // already allocated, so the whole path is allocation-free.
        let scratch_bytes = m * self.k * std::mem::size_of::<half::f16>();
        let (scratch, scratch_warm) = marlin_gemm::ensure_scratch(&self.runtime, 0, scratch_bytes)?;

        self.launch_rmsnorm_prefill(activation, gamma, scratch, m)?;
        let args = marlin_gemm::MarlinGemmArgs {
            activation: scratch,
            weights: weights_ptr,
            scales: cuptr(scales.data_ptr::<u8>() as *const c_void),
            zero_points: zero_points.map(|t| cuptr(t.data_ptr::<u8>() as *const c_void)),
            bias: bias.map(|t| cuptr(t.data_ptr::<u8>() as *const c_void)),
            output: cuptr(output.data_ptr_mut::<u8>() as *const c_void),
            m,
            k: self.k,
            n: self.n,
            group_size: self.block_size,
            scales_fp16,
            bias_post_round: self.fold_bias_post_round && bias.is_some(),
            bias_row_stride,
        };
        let split_warm = self.maybe_launch_marlin_splitk(&args)?;
        Ok(Some(weights_warm && scratch_warm && split_warm))
    }

    /// Dequantize + fp16 cuBLASLt M>1 path for the paired gate/up SwiGLU MLP
    /// fusion, optionally with the fused RMS-norm prologue.
    ///
    /// These are the widest projections in the decoder and therefore most of
    /// prefill's arithmetic, so leaving them on Marlin while the unfused nodes
    /// moved to tensor cores would cap the win. The structure is the Marlin
    /// path's: stage the normalized activation once, run the two projections
    /// into pooled scratch, then apply the same fp16 SiluMul epilogue — only
    /// the two GEMMs differ.
    ///
    /// Returns `Ok(false)` when ineligible so the caller falls through.
    #[allow(clippy::too_many_arguments)]
    fn try_dequant_f16_gate_up_prefill(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        scales_fp16: bool,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        gamma: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
        workspace: Option<WorkspaceView>,
    ) -> Result<bool> {
        let Some(block_shift) = self.dequant_f16_block_shift() else {
            return Ok(false);
        };
        let weight_bytes = self
            .k
            .checked_mul(self.n)
            .and_then(|elems| elems.checked_mul(2))
            .ok_or_else(|| error("dequantized f16 weight size overflowed"))?;
        if weight_bytes > dequant_f16_gemm_max_scratch_bytes() {
            return Ok(false);
        }
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits dequant f16 gate/up SwiGLU GEMM")?;

        // Stage the RMS-normalized activation once (slot 0, as the Marlin path
        // does), so both projections read the same normalized copy.
        let act_ptr = if let Some(gamma) = gamma {
            let norm_bytes = m * self.k * std::mem::size_of::<half::f16>();
            let (norm, _warm) = marlin_gemm::ensure_scratch(&self.runtime, 0, norm_bytes)?;
            self.launch_rmsnorm_prefill(activation, gamma, norm, m)?;
            norm
        } else {
            cuptr(activation.data_ptr::<u8>() as *const c_void)
        };

        let out_bytes = output.byte_size();
        // Slot 1 holds the gate projection, as on the Marlin path; the up
        // projection writes the real output and SiluMul folds them in place.
        let (gate_buf, _warm) = marlin_gemm::ensure_scratch(&self.runtime, 1, out_bytes)?;
        // Slots 5 and 6 hold the two dequantized weights. They must differ: both
        // are live across the second GEMM.
        let (weight_gate, _warm) = marlin_gemm::ensure_scratch(&self.runtime, 5, weight_bytes)?;
        let (weight_up, _warm) = marlin_gemm::ensure_scratch(&self.runtime, 6, weight_bytes)?;
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);

        for (packed, scales, zero_points, weight, out) in [
            (packed_gate, scales_gate, zp_gate, weight_gate, gate_buf),
            (packed_up, scales_up, zp_up, weight_up, output_ptr),
        ] {
            self.launch_dequant_f16(
                packed,
                scales,
                scales_fp16,
                zero_points,
                weight,
                k_blocks,
                blob_size,
                zp_row_bytes,
                block_shift,
            )?;
            let params = self.dequant_f16_gemm_ex(act_ptr, weight, out, m, None);
            // SAFETY: the dequantized [N, K] weight was just written, the
            // activation is the validated (optionally normalized) [M, K] fp16
            // panel, and the destination is a distinct [M, N] fp16 buffer.
            unsafe {
                blas::governed_gemm_ex(
                    self.runtime.blas(),
                    self.runtime.stream_ptr(),
                    &params,
                    workspace,
                    "MatMulNBits",
                )
            }?;
        }

        crate::kernels::elementwise::launch_silu_mul_f16_raw(
            &self.runtime,
            gate_buf,
            output_ptr,
            output_ptr,
            output.numel(),
            self.decomposed_silu,
        )?;
        Ok(true)
    }

    /// Marlin M>1 path for the paired gate/up SwiGLU MLP fusion (optionally with
    /// a fused RMS-norm prologue). Runs both projections through the Marlin int4
    /// tensor-core GEMM into pooled scratch, then the same fp16 SiluMul epilogue
    /// the tiled path uses (`silu(gate) * up`, decomposed or not). Returns
    /// `Some(warm)` when it launched (`warm` ⇒ every repacked weight and scratch
    /// buffer was already pooled, so the call is allocation-free and
    /// capture-safe), or `None` when the shape is ineligible (caller falls
    /// through to the tiled gate/up prefill).
    #[allow(clippy::too_many_arguments)]
    fn try_launch_marlin_gate_up_prefill(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        gamma: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
    ) -> Result<Option<bool>> {
        if self.block_size == 0
            || !self.k.is_multiple_of(16)
            || !self.k.is_multiple_of(self.block_size)
        {
            return Ok(None);
        }
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits Marlin int4 gate/up SwiGLU GEMM")?;

        let packed_gate_ptr = cuptr(packed_gate.data_ptr::<u8>() as *const c_void);
        let packed_up_ptr = cuptr(packed_up.data_ptr::<u8>() as *const c_void);
        let (weights_gate, gate_w_warm) = marlin_gemm::ensure_repacked(
            &self.runtime,
            packed_gate_ptr,
            self.n,
            self.k,
            self.block_size,
        )?;
        let (weights_up, up_w_warm) = marlin_gemm::ensure_repacked(
            &self.runtime,
            packed_up_ptr,
            self.n,
            self.k,
            self.block_size,
        )?;

        // Optionally stage the RMS-normalized activation into pooled scratch
        // (slot 0), byte-identical to the standalone prologue; both projections
        // then read that single normalized copy.
        let (act_ptr, norm_warm) = if let Some(gamma) = gamma {
            let norm_bytes = m * self.k * std::mem::size_of::<half::f16>();
            let (norm, norm_warm) = marlin_gemm::ensure_scratch(&self.runtime, 0, norm_bytes)?;
            self.launch_rmsnorm_prefill(activation, gamma, norm, m)?;
            (norm, norm_warm)
        } else {
            (cuptr(activation.data_ptr::<u8>() as *const c_void), true)
        };

        // Pooled gate-projection scratch (slot 1). The up projection writes the
        // real output; SiluMul folds them into the output in place.
        let out_bytes = output.byte_size();
        let (gate_buf, gate_s_warm) = marlin_gemm::ensure_scratch(&self.runtime, 1, out_bytes)?;
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);

        let gate_args = marlin_gemm::MarlinGemmArgs {
            activation: act_ptr,
            weights: weights_gate,
            scales: cuptr(scales_gate.data_ptr::<u8>() as *const c_void),
            zero_points: zp_gate.map(|t| cuptr(t.data_ptr::<u8>() as *const c_void)),
            bias: None,
            output: gate_buf,
            m,
            k: self.k,
            n: self.n,
            group_size: self.block_size,
            scales_fp16: true,
            bias_post_round: false,
            bias_row_stride: 0,
        };
        // Route both projections through split-K: at the M=K speculative-verify
        // width these gate/up GEMMs (the largest MLP projections) are
        // memory/latency-bound, so `choose_split_k` elects an 8-way K split to
        // fill the SMs. Partials use the shared slot-4 scratch sequentially (gate
        // fully reduces into `gate_buf` before the up GEMM overwrites the
        // partials), so the stream ordering keeps it correct and capture-safe.
        let gate_split_warm = self.maybe_launch_marlin_splitk(&gate_args)?;

        let up_args = marlin_gemm::MarlinGemmArgs {
            activation: act_ptr,
            weights: weights_up,
            scales: cuptr(scales_up.data_ptr::<u8>() as *const c_void),
            zero_points: zp_up.map(|t| cuptr(t.data_ptr::<u8>() as *const c_void)),
            bias: None,
            output: output_ptr,
            m,
            k: self.k,
            n: self.n,
            group_size: self.block_size,
            scales_fp16: true,
            bias_post_round: false,
            bias_row_stride: 0,
        };
        let up_split_warm = self.maybe_launch_marlin_splitk(&up_args)?;

        crate::kernels::elementwise::launch_silu_mul_f16_raw(
            &self.runtime,
            gate_buf,
            output_ptr,
            output_ptr,
            output.numel(),
            self.decomposed_silu,
        )?;

        Ok(Some(
            gate_w_warm
                && up_w_warm
                && norm_warm
                && gate_s_warm
                && gate_split_warm
                && up_split_warm,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemm(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        blob_size: usize,
        bias_row_stride: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 prefill GEMM")?;
        let general_block_size = self.block_size != 32;
        let entry = if general_block_size {
            GEMM_F16_GENERAL_BS_ENTRY
        } else {
            GEMM_F16_ENTRY
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let m_i32 = as_i32("M", m)?;
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let bits = as_i32("bits", self.bits)?;
        let scales_fp16_flag = scales_fp16 as i32;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let bias_row_stride_i32 = as_i32("bias row stride", bias_row_stride)?;
        let block_size_i32 = as_i32("block_size", self.block_size)?;
        let blob_size_i32 = as_i32("block blob size", blob_size)?;
        let grid_x = u32::try_from(self.n.div_ceil(GEMM_F16_TILE))
            .map_err(|_| error(format!("N={} exceeds CUDA prefill grid limits", self.n)))?;
        let grid_y = u32::try_from(m.div_ceil(GEMM_F16_TILE))
            .map_err(|_| error(format!("M={m} exceeds CUDA prefill grid limits")))?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&m_i32)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&bits)
            .arg(&scales_fp16_flag)
            .arg(&bias_post_round_flag)
            .arg(&bias_row_stride_i32);
        // The general-block-size prefill kernel takes two extra trailing scalars
        // (`block_size`, `blob_size`) to derive the packed layout for any block
        // width; the tuned block-32 kernel bakes those in and takes neither.
        if general_block_size {
            builder.arg(&block_size_i32).arg(&blob_size_i32);
        }
        // SAFETY: dense tensors and all dimensions were validated
        // above. The 16x16 CTA uses 4 KiB of statically sized shared memory,
        // ordinary fp32 CUDA-core arithmetic, and fp16 conversions only. It has
        // no tensor-core/PTX/cp.async dependency, so the same path is the
        // portable fallback on every CUDA SM supported by this crate.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (GEMM_F16_TILE as u32, GEMM_F16_TILE as u32, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 prefill GEMM", err))
    }

    /// M>1 prefill path for the fused RMS-normalization prologue. It stages the
    /// per-token normalized activation (byte-identical to the standalone
    /// `skip_rmsnorm_f16_warp_half4` output) into scratch, then runs the
    /// portable tiled GEMM over it. Prefill is outside the persistent decode
    /// graph, so the scratch allocation here is not on any captured path.
    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemm_rmsnorm_prefill(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        gamma: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        bias_row_stride: usize,
    ) -> Result<()> {
        let scratch = self
            .runtime
            .alloc_raw(m * self.k * std::mem::size_of::<half::f16>())?;
        let scratch_shape = [m, self.k];
        let scratch_strides = [self.k as i64, 1];
        let normalized = TensorView::new(
            DevicePtr(raw_ptr(scratch) as *const c_void),
            DataType::Float16,
            &scratch_shape,
            &scratch_strides,
            activation.device,
        );
        let result = self
            .launch_rmsnorm_prefill(activation, gamma, scratch, m)
            .and_then(|()| {
                self.launch_f16_gemm(
                    &normalized,
                    packed,
                    scales,
                    true,
                    zero_points,
                    bias,
                    output,
                    m,
                    k_blocks,
                    self.block_size * self.bits / 8,
                    bias_row_stride,
                )
            });
        // SAFETY: `scratch` came from `alloc_raw` above and is freed exactly
        // once; `cuMemFree` waits for the preceding norm + GEMM stream work.
        let free_scratch = unsafe { self.runtime.free_raw(scratch) };
        result.and(free_scratch)
    }

    /// Launches the standalone RMS-normalization prologue used by prefill. One
    /// warp normalizes one token row of `activation` into `normalized`.
    fn launch_rmsnorm_prefill(
        &self,
        activation: &TensorView,
        gamma: &TensorView,
        normalized: CUdeviceptr,
        m: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 RMS-norm prefill prologue")?;
        let function =
            self.runtime
                .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, RMSNORM_PREFILL_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let normalized_ptr = normalized;
        let norm_size = as_i32("K", self.k)?;
        let num_groups = as_i32("M", m)?;
        let gamma_is_half: i32 = (gamma.dtype == DataType::Float16) as i32;
        let epsilon = self.rmsnorm_epsilon;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&gamma_ptr)
            .arg(&normalized_ptr)
            .arg(&norm_size)
            .arg(&num_groups)
            .arg(&gamma_is_half)
            .arg(&epsilon);
        // SAFETY: `activation` and `gamma` are validated contiguous fp16 tensors
        // and `normalized` is a `K * M`-half scratch buffer allocated by the
        // caller. Each of the `M` one-warp blocks reads/writes only its own row
        // with the launch-predicate-guaranteed `K % 128 == 0` half4 chunking.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (RMSNORM_PREFILL_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 RMS-norm prefill prologue", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_int8_f16_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits int8 fp16 GEMV")?;
        // Grid-starved standalone int8-zp GEMV (e.g. Phi's int8 down projection,
        // grid 384 / ~0.48 waves/SM): when K is a whole multiple of the 256-wide
        // step and the shape uses the 256-thread large path, take the split-K
        // entry (K_SPLIT warps/column, K_SPLIT x larger grid) to fill the SMs.
        // This kernel has no serial prologue, so the extra grid parallelism pays
        // off directly. Asymmetric int8 (with zero points) fires unconditionally
        // on the large path; SYMMETRIC int8 (no zero points) opts in only when
        // the live SM count says the single-warp grid is too narrow (mirrors the
        // symmetric int4 grid-fill gate [`use_f16_symmetric_splitk`]), so the
        // already-occupied wide projections keep the byte-identical single-warp
        // entry and small consumer GPUs are unaffected. The symmetric split-K
        // reassociates the per-column fp32 partials (near-equal, not
        // byte-identical) — greedy-token-stable under fp32 accumulation — and is
        // opt-out via `int8_symmetric_splitk_enabled`.
        let large_path_eligible = self.k.is_multiple_of(256)
            && !(self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX);
        let capabilities = self.runtime.capabilities();
        let use_splitk = large_path_eligible
            && if zero_points.is_some() {
                true
            } else {
                int8_symmetric_splitk_enabled()
                    && use_f16_symmetric_splitk(
                        self.k,
                        self.n,
                        capabilities.multiprocessor_count(),
                        capabilities.max_threads_per_block(),
                    )
            };
        let entry = if use_splitk {
            GEMV_INT8_F16_SPLITK_ENTRY
        } else {
            GEMV_INT8_F16_ENTRY
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let scales_fp16_flag = scales_fp16 as i32;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let threads = if self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX {
            GEMV_F16_SMALL_THREADS
        } else {
            GEMV_F16_LARGE_THREADS
        };
        let columns_per_block = if use_splitk {
            (threads / 32) as usize / GEMV_INT8_F16_SPLITK
        } else {
            (threads / 32) as usize
        };
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&scales_fp16_flag)
            .arg(&bias_post_round_flag);
        // SAFETY: validation restricts this entry to dense fp16 block-32 M=1
        // tensors. The kernel uses fixed device pointers, registers, and warp
        // shuffles only, so it performs no allocation or synchronization and is
        // legal to capture and replay on every CUDA SM supported by this crate.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits int8 fp16 GEMV", err))
    }

    /// INT8 decode GEMV with a fused RMS-normalization prologue. Mirrors
    /// [`Self::launch_f16_gemv_rmsnorm`] but dispatches the int8 sibling kernel,
    /// which shares the RMS reduction / normalized-activation staging bit-for-bit
    /// and swaps in the block-32 int8 dequant dot. Restricted to fp16 scales and
    /// block-32, matching the fusion's eligibility gates.
    #[allow(clippy::too_many_arguments)]
    fn launch_int8_f16_gemv_rmsnorm(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        gamma: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits int8 fp16 RMS-norm-prologue GEMV")?;
        if bias.is_some() {
            if self.fold_bias_post_round {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "qkv_bias_fused",
                    "folded standalone Add(MatMulNBits, bias) into GEMV epilogue with \
                     fp16-after-round semantics fp16(fp16(acc)+bias) (token-identity preserved)"
                );
            } else {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "bias_native",
                    "native MatMulNBits bias: single-round epilogue fp16(acc+bias)"
                );
            }
        }
        let entry = if zero_points.is_some() {
            GEMV_INT8_F16_SCALES_F16_RMSNORM_ZP_ENTRY
        } else {
            GEMV_INT8_F16_SCALES_F16_RMSNORM_ENTRY
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let gamma_is_half: i32 = (gamma.dtype == DataType::Float16) as i32;
        let epsilon = self.rmsnorm_epsilon;
        let threads = if self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX {
            GEMV_F16_SMALL_THREADS
        } else {
            GEMV_F16_LARGE_THREADS
        };
        let columns_per_block = (threads / 32) as usize;
        let shared_mem_bytes = (self.k * std::mem::size_of::<half::f16>()) as u32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&gamma_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&bias_post_round_flag)
            .arg(&gamma_is_half)
            .arg(&epsilon);
        // SAFETY: restricted to block-32 M=1 fp16 inputs with fp16 scales, all
        // dtype/shape/contiguity validated above. The kernel stages the
        // normalized activation in launch-time dynamic shared memory
        // (`K * sizeof(f16)`, bounded by the fusion's `K % 128 == 0` predicate)
        // and uses only registers, warp shuffles, and `__syncthreads` — no
        // per-call allocation or host synchronization — so it is legal to record
        // into and replay from a CUDA graph.
        // Clamp the K-sized activation stage to the device's shared-memory
        // budget: opt into >48 KB when the card allows it, and fail loudly
        // (rather than launch-crash on a smaller consumer GPU) if K exceeds even
        // the opt-in ceiling.
        self.runtime
            .configure_dynamic_shared_memory(&function, shared_mem_bytes)?;
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits int8 fp16 RMS-norm-prologue GEMV", err))
    }

    /// Paired gate/up projection + SwiGLU path (see
    /// [`crate::optimizer::CudaGateUpSwiGluFusion`]). Inputs are
    /// `[x, W_gate, scales_gate, W_up, scales_up]`. M=1 keeps the paired decode
    /// GEMV unchanged; M>1 reuses the portable tiled prefill GEMM for both
    /// projections before applying the existing fp16 `silu(gate)*up` kernel.
    fn run_f16_gate_up_swiglu(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        // Contract from `CudaGateUpSwiGluFusion`:
        //   [x, W_gate, scales_gate, W_up, scales_up, (gamma?)@5, (zp_gate?)@6, (zp_up?)@7]
        // Slot 5 carries the RMS-norm gamma when the skip-rmsnorm prologue is
        // folded in; slots 6/7 carry per-projection asymmetric zero points
        // (both present for asymmetric weights, both absent for symmetric ones).
        if !(5..=8).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(error(format!(
                "gate/up SwiGLU fusion expects 5 to 8 inputs [x, W_gate, scales_gate, W_up, \
                 scales_up, (gamma), (zp_gate, zp_up)] and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        require_dtype("A", inputs[0].dtype, DataType::Float16)?;
        require_dtype("W_gate", inputs[1].dtype, DataType::Uint8)?;
        require_dtype("scales_gate", inputs[2].dtype, DataType::Float16)?;
        require_dtype("W_up", inputs[3].dtype, DataType::Uint8)?;
        require_dtype("scales_up", inputs[4].dtype, DataType::Float16)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float16)?;
        let gamma = if self.rmsnorm_prologue {
            let gamma = optional_input(inputs, 5).ok_or_else(|| {
                error("rmsnorm_prologue fusion requires the normalization weight at input 5")
            })?;
            require_gamma_dtype(gamma.dtype)?;
            require_shape("gamma", gamma.shape, &[self.k])?;
            if !gamma.is_contiguous() {
                return Err(error(
                    "gamma must be contiguous on the CUDA execution provider".to_string(),
                ));
            }
            Some(gamma)
        } else {
            None
        };

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        if outputs[0].shape != expected_output_shape {
            return Err(error(format!(
                "Y must have shape {expected_output_shape:?}, got {:?}",
                outputs[0].shape
            )));
        }

        if self.block_size != 32 || self.bits != 4 {
            return Err(error(format!(
                "gate/up SwiGLU fusion received bits={} and block_size={}. Why: the fused fp16 \
                 path implements the block-32 packed int4 layout. How to fix: export 4-bit \
                 MatMulNBits weights with block_size=32 or disable this fusion",
                self.bits, self.block_size
            )));
        }
        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size / 2;
        require_shape("W_gate", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        require_shape("W_up", inputs[3].shape, &[self.n, k_blocks, blob_size])?;
        require_flat_or_matrix_shape("scales_gate", inputs[2].shape, self.n, k_blocks)?;
        require_flat_or_matrix_shape("scales_up", inputs[4].shape, self.n, k_blocks)?;

        // Optional asymmetric zero points (slots 6/7). Symmetric weights omit
        // both and the kernels apply the implicit `zp == 8` subtrahend, matching
        // the historical byte-identical path. Require them paired: mixing a
        // zero-point projection with a symmetric one is never valid.
        let zp_gate = optional_input(inputs, 6);
        let zp_up = optional_input(inputs, 7);
        if zp_gate.is_some() != zp_up.is_some() {
            return Err(error(
                "gate/up SwiGLU fusion requires zero points for both projections or neither"
                    .to_string(),
            ));
        }
        let zp_row_bytes = (k_blocks * self.bits).div_ceil(8);
        for (name, zp) in [("zp_gate", zp_gate), ("zp_up", zp_up)] {
            if let Some(zp) = zp {
                require_dtype(name, zp.dtype, DataType::Uint8)?;
                require_flat_or_matrix_shape(name, zp.shape, self.n, zp_row_bytes)?;
                if !zp.is_contiguous() {
                    return Err(error(format!(
                        "{name} must be contiguous on the CUDA execution provider"
                    )));
                }
            }
        }

        for (name, contiguous) in [
            ("A", inputs[0].is_contiguous()),
            ("W_gate", inputs[1].is_contiguous()),
            ("scales_gate", inputs[2].is_contiguous()),
            ("W_up", inputs[3].is_contiguous()),
            ("scales_up", inputs[4].is_contiguous()),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let rows = m as u64;
            let mut flops = rows
                .saturating_mul(self.n as u64)
                .saturating_mul(self.k as u64)
                .saturating_mul(4)
                .saturating_add(rows.saturating_mul(self.n as u64).saturating_mul(5));
            if self.rmsnorm_prologue {
                let elements = rows.saturating_mul(self.k as u64);
                flops = flops
                    .saturating_add(elements.saturating_mul(4))
                    .saturating_add(rows.saturating_mul(4));
            }
            flops
        });

        if m == 0 {
            self.last_call_capture_safe.store(false, Ordering::Relaxed);
            onnx_runtime_ep_api::record_kernel_variant!(
                "gate_up_swiglu_empty",
                "M=0 gate/up SwiGLU has an empty output and requires no CUDA launch"
            );
            return Ok(());
        }
        if m > 1 {
            // Small-batch decode fast path (mirrors the plain-GEMV loop in
            // `run_f16`): for M within the crossover window, loop the capture-safe
            // M==1 fused gate/up SwiGLU GEMV once per row instead of the tiled
            // prefill GEMM. Each row is computed byte-identically to a
            // single-sequence (M==1) fused SwiGLU decode of that row, and every
            // per-row launch is static-grid, allocation- and sync-free, so the
            // batch decode graph captures it as part of the whole-subgraph capture
            // instead of fragmenting into an eager seam. This is the node that
            // otherwise leaves batch-N (M>=2) decode capturing as many segments as
            // there are MLP layers, whose per-step eager replay dominates the
            // M>=2 step cost. Streaming is unaffected: the resident gate/up weight
            // is read M times from VRAM, not re-streamed (HtoD 1/N amortization
            // intact).
            //
            // GATED TO THE RMS-NORM-PROLOGUE (`gamma.is_some()`) PATH ONLY. The
            // production skip-rmsnorm fusion (`CudaGateUpSwiGluFusion` +
            // `MATMUL_NBITS_RMSNORM_PROLOGUE_ATTR`) folds the pre-MLP RMS norm
            // into this node for every RMSNorm architecture (Qwen/Llama/Phi), so
            // the resident-model decode win lives entirely on this branch. A ULP
            // sweep (`fp16_gate_up_swiglu_rmsnorm_two_op_ulp_bound_sweep`) shows
            // the fused rmsnorm decode GEMV is BYTE-IDENTICAL (0 ULP, 0/60 cases)
            // to the two-op decode reference at M==1, so routing M>1 through the
            // per-row loop is byte-exact to running each row alone as M==1 decode
            // (the batching contract) AND within the two-op decode identity.
            //
            // The PLAIN (`gamma.is_none()`) gate/up SwiGLU is intentionally NOT
            // routed here: its M==1 fused decode GEMV is measured up to 2 ULP off
            // the two-op decode reference (8/56 cases in
            // `fp16_gate_up_swiglu_two_op_ulp_bound_sweep`), so a looped M>1
            // decode-GEMV would change plain-path logits vs the two-op path by
            // more than 1 ULP. It stays on the prefill/Marlin path (advertised
            // capture-unsafe at M>1) pending a decision on that bound in #1334.
            if m <= decode_gemv_loop_max_m() {
                if let Some(gamma) = gamma {
                    onnx_runtime_ep_api::record_kernel_variant!(
                        "gate_up_swiglu_rmsnorm_batched_loop",
                        "M={} small-batch rmsnorm decode: {} single-row fused gate/up SwiGLU \
                         GEMV launches (one per row), each byte-identical to M==1 decode; keeps \
                         the batch decode graph a single captured subgraph instead of a \
                         per-MLP-layer eager seam",
                        m,
                        m
                    );
                    self.last_call_capture_safe.store(true, Ordering::Relaxed);
                    let a_base = inputs[0].data_ptr::<u8>();
                    let y_base = outputs[0].data_ptr_mut::<u8>();
                    let a_row_shape = [1usize, self.k];
                    let a_row_strides = [self.k as i64, 1];
                    let y_row_shape = [1usize, self.n];
                    let y_row_strides = [self.n as i64, 1];
                    let a_row_bytes = self.k * 2; // fp16 activation
                    let y_row_bytes = self.n * 2; // fp16 output
                    for row in 0..m {
                        let a_row = TensorView::new(
                            DevicePtr(a_base.wrapping_add(row * a_row_bytes) as *const c_void),
                            DataType::Float16,
                            &a_row_shape,
                            &a_row_strides,
                            inputs[0].device,
                        );
                        let mut y_row = TensorMut::new(
                            DevicePtrMut(y_base.wrapping_add(row * y_row_bytes) as *mut c_void),
                            DataType::Float16,
                            &y_row_shape,
                            &y_row_strides,
                            outputs[0].device,
                        );
                        self.launch_gate_up_swiglu_rmsnorm(
                            &a_row,
                            &inputs[1],
                            &inputs[2],
                            &inputs[3],
                            &inputs[4],
                            zp_gate,
                            zp_up,
                            gamma,
                            &mut y_row,
                            k_blocks,
                            blob_size,
                            zp_row_bytes,
                        )?;
                    }
                    return Ok(());
                }
            }
            self.last_call_capture_safe.store(false, Ordering::Relaxed);
            // Dequantize both projections to fp16 and run them on cuBLASLt
            // tensor cores, ahead of Marlin, for the reason given at the
            // unfused M>1 site: Marlin re-reads its whole A panel from global
            // memory per warp on these shapes. These are the widest projections
            // in the decoder, so leaving them behind would cap the win.
            if dequant_f16_gemm_enabled() && m >= dequant_f16_gemm_min_m() {
                match self.try_dequant_f16_gate_up_prefill(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &inputs[3],
                    &inputs[4],
                    inputs[2].dtype == DataType::Float16,
                    zp_gate,
                    zp_up,
                    gamma,
                    &mut outputs[0],
                    m,
                    k_blocks,
                    blob_size,
                    zp_row_bytes,
                    workspace,
                ) {
                    Ok(true) => {
                        onnx_runtime_ep_api::record_kernel_variant!(
                            "gate_up_swiglu_dequant_f16_cublas",
                            "M={} prefill: dequantize both int{} projections to [N, K] fp16, \
                             two cuBLASLt fp16 tensor-core GEMMs, then the fp16 SiluMul \
                             epilogue",
                            m,
                            self.bits
                        );
                        return Ok(());
                    }
                    Ok(false) => {
                        if std::env::var_os("ONNX_GENAI_DEQUANT_F16_DEBUG").is_some() {
                            eprintln!("dequant_f16 gate/up: ineligible m={m}");
                        }
                    }
                    Err(_err) => {
                        if std::env::var_os("ONNX_GENAI_DEQUANT_F16_DEBUG").is_some() {
                            eprintln!("dequant_f16 gate/up: {_err}");
                        }
                    }
                }
            }
            // Opt-in Marlin int4 tensor-core path for the paired gate/up MLP:
            // both projections run on tensor cores, then the same fp16 SiluMul
            // epilogue. This is the bulk of prefill/verify cost and the last
            // MatMulNBits node that otherwise falls back to the tiled GEMM at
            // M>1 (which would keep the captured forward segmented). Warm replays
            // (pooled weights + scratch) are capture-safe. Falls through to the
            // tiled gate/up prefill on ineligibility or launch error.
            if marlin_gemm::marlin_m_gt_1_enabled()
                && self.bits == 4
                && marlin_gemm::device_supports_marlin(
                    self.runtime.capabilities().compute_capability(),
                )
            {
                match self.try_launch_marlin_gate_up_prefill(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &inputs[3],
                    &inputs[4],
                    zp_gate,
                    zp_up,
                    gamma,
                    &mut outputs[0],
                    m,
                ) {
                    Ok(Some(warm)) => {
                        self.last_call_capture_safe.store(warm, Ordering::Relaxed);
                        onnx_runtime_ep_api::record_kernel_variant!(
                            "gate_up_swiglu_marlin_prefill",
                            "M={} prefill/verify: {}paired gate/up Marlin SM80 mma.sync int4 \
                             tensor-core GEMMs followed by fp16 SiluMul (capture-safe when \
                             weights + scratch are pre-warmed)",
                            m,
                            if gamma.is_some() {
                                "RMS-normalization prologue then "
                            } else {
                                ""
                            }
                        );
                        return Ok(());
                    }
                    Ok(None) => {
                        // Ineligible shape; fall through to the tiled prefill.
                    }
                    Err(_err) => {
                        // Hard error; fall through to the tiled prefill.
                    }
                }
            }
            if let Some(gamma) = gamma {
                onnx_runtime_ep_api::record_kernel_variant!(
                    "gate_up_swiglu_rmsnorm_prefill",
                    "M={} prefill: RMS-normalization prologue into scratch, then two portable \
                     block-32 int4 fp16 tiled GEMMs with fp32 accumulation, followed by fp16 \
                     SiluMul; not advertised as CUDA-graph capture-safe",
                    m
                );
                return self.launch_gate_up_swiglu_rmsnorm_prefill(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &inputs[3],
                    &inputs[4],
                    zp_gate,
                    zp_up,
                    gamma,
                    &mut outputs[0],
                    m,
                    k_blocks,
                    zp_row_bytes,
                );
            }
            onnx_runtime_ep_api::record_kernel_variant!(
                "gate_up_swiglu_prefill",
                "M={} prefill: two portable block-32 int4 fp16 tiled GEMMs with fp32 \
                 accumulation, followed by fp16 SiluMul; not advertised as CUDA-graph \
                 capture-safe",
                m
            );
            return self.launch_gate_up_swiglu_prefill(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &inputs[3],
                &inputs[4],
                zp_gate,
                zp_up,
                &mut outputs[0],
                m,
                k_blocks,
            );
        }

        if let Some(gamma) = gamma {
            onnx_runtime_ep_api::record_kernel_variant!(
                "gate_up_swiglu_rmsnorm_fused",
                "fp16 block-32 M==1 decode: fused RMS-normalization prologue + paired gate/up \
                 int4 GEMV + SwiGLU (silu(gate)*up) in one capture-safe kernel; the RMS \
                 reduction runs once for both projections and reproduces the standalone norm \
                 plus two-op fp16 rounding for byte-identical greedy tokens"
            );
            self.last_call_capture_safe.store(true, Ordering::Relaxed);
            return self.launch_gate_up_swiglu_rmsnorm(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                &inputs[3],
                &inputs[4],
                zp_gate,
                zp_up,
                gamma,
                &mut outputs[0],
                k_blocks,
                blob_size,
                zp_row_bytes,
            );
        }

        onnx_runtime_ep_api::record_kernel_variant!(
            "gate_up_swiglu_fused",
            "fp16 block-32 M==1 decode: fused paired gate/up int4 GEMV + SwiGLU \
             (silu(gate)*up) in one capture-safe kernel; reproduces the two-op fp16 \
             rounding for byte-identical greedy tokens"
        );

        self.last_call_capture_safe.store(true, Ordering::Relaxed);
        self.launch_gate_up_swiglu(
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &inputs[3],
            &inputs[4],
            zp_gate,
            zp_up,
            &mut outputs[0],
            k_blocks,
            blob_size,
            zp_row_bytes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_swiglu_prefill(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
    ) -> Result<()> {
        let scratch = self.runtime.alloc_raw(output.byte_size())?;
        let scratch_shape = output.shape.to_vec();
        let scratch_strides = output.strides.to_vec();
        let mut gate_output = TensorMut::new(
            DevicePtrMut(raw_ptr(scratch)),
            DataType::Float16,
            &scratch_shape,
            &scratch_strides,
            output.device,
        );

        let result = (|| {
            self.launch_f16_gemm(
                activation,
                packed_gate,
                scales_gate,
                true,
                zp_gate,
                None,
                &mut gate_output,
                m,
                k_blocks,
                self.block_size * self.bits / 8,
                0,
            )?;
            self.launch_f16_gemm(
                activation,
                packed_up,
                scales_up,
                true,
                zp_up,
                None,
                output,
                m,
                k_blocks,
                self.block_size * self.bits / 8,
                0,
            )?;
            let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
            crate::kernels::elementwise::launch_silu_mul_f16_raw(
                &self.runtime,
                scratch,
                output_ptr,
                output_ptr,
                output.numel(),
                self.decomposed_silu,
            )
        })();

        // Always release the prefill gate projection. `cuMemFree` waits for
        // preceding stream work that uses this allocation, including SiluMul.
        // SAFETY: `scratch` came from `alloc_raw` above and is freed exactly once.
        let free_scratch = unsafe { self.runtime.free_raw(scratch) };
        result.and(free_scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_swiglu(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 gate/up SwiGLU GEMV")?;
        // Symmetric weights launch the `HasZp == false` entry, whose PTX drops the
        // per-block zero-point load entirely; only asymmetric weights pay for it.
        // When `ONNX_GENAI_GATEUP_VEC` is armed, the symmetric entries reroute to
        // their byte-identical fused-symmetric `_vec` sibling (fewer issued ops).
        let has_zp = zp_gate.is_some() || zp_up.is_some();
        let vec = !has_zp && gate_up_vec_enabled();
        let entry = match (self.decomposed_silu, has_zp, vec) {
            (true, true, _) => GATE_UP_DECOMPOSED_SWIGLU_ZP_ENTRY,
            (true, false, true) => GATE_UP_DECOMPOSED_SWIGLU_VEC_ENTRY,
            (true, false, false) => GATE_UP_DECOMPOSED_SWIGLU_ENTRY,
            (false, true, _) => GATE_UP_SWIGLU_ZP_ENTRY,
            (false, false, true) => GATE_UP_SWIGLU_VEC_ENTRY,
            (false, false, false) => GATE_UP_SWIGLU_ENTRY,
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_gate_ptr = cuptr(packed_gate.data_ptr::<u8>() as *const c_void);
        let scales_gate_ptr = cuptr(scales_gate.data_ptr::<u8>() as *const c_void);
        let packed_up_ptr = cuptr(packed_up.data_ptr::<u8>() as *const c_void);
        let scales_up_ptr = cuptr(scales_up.data_ptr::<u8>() as *const c_void);
        let zp_gate_ptr = zp_gate
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let zp_up_ptr = zp_up
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row byte count", zp_row_bytes)?;
        let threads = GATE_UP_SWIGLU_THREADS;
        let columns_per_block = (threads / 32) as usize;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_gate_ptr)
            .arg(&scales_gate_ptr)
            .arg(&packed_up_ptr)
            .arg(&scales_up_ptr)
            .arg(&zp_gate_ptr)
            .arg(&zp_up_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes);
        // SAFETY: restricted to fp16 block-32 M=1 inputs validated above; both
        // persistent weight/scale sets and the output are fixed device pointers,
        // the scalar ABI matches the paired entry point, and the kernel uses only
        // registers + warp shuffles (no per-call alloc, shared memory, or sync),
        // so the launch is legal to record into and replay from a CUDA graph.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 gate/up SwiGLU GEMV", err))
    }

    /// Decode (M==1) paired gate/up SwiGLU with a fused RMS-normalization
    /// prologue. The block reduces the shared activation once and stages the
    /// normalized activation in launch-time dynamic shared memory (`K *
    /// sizeof(f16)`, bounded by the fusion's `K % 128 == 0` predicate), then
    /// both projections read that single staged copy. Fixing the reduction to a
    /// single pass is what recovers the double-recompute cost the fan-out-2
    /// post-attention norm otherwise pays.
    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_swiglu_rmsnorm(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        gamma: &TensorView,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 gate/up SwiGLU RMS-norm GEMV")?;
        // `ONNX_GENAI_GATEUP_VEC` reroutes the symmetric RMS-norm-fused entries to
        // their byte-identical fused-symmetric `_vec` sibling (fewer issued ops);
        // `ONNX_GENAI_GATEUP_OCC` further reroutes them to the `_vec_occ` sibling
        // (same body + `__launch_bounds__(256, 8)` to raise occupancy 62->83% and
        // hide the Short-Scoreboard shared-load latency). Both are byte-identical.
        let has_zp = zp_gate.is_some() || zp_up.is_some();
        let vec = !has_zp && gate_up_vec_enabled();
        let occ = !has_zp && gate_up_occ_enabled();
        let entry = match (self.decomposed_silu, has_zp, occ, vec) {
            (true, true, _, _) => GATE_UP_DECOMPOSED_SWIGLU_RMSNORM_ZP_ENTRY,
            (true, false, true, _) => GATE_UP_DECOMPOSED_SWIGLU_RMSNORM_VEC_OCC_ENTRY,
            (true, false, false, true) => GATE_UP_DECOMPOSED_SWIGLU_RMSNORM_VEC_ENTRY,
            (true, false, false, false) => GATE_UP_DECOMPOSED_SWIGLU_RMSNORM_ENTRY,
            (false, true, _, _) => GATE_UP_SWIGLU_RMSNORM_ZP_ENTRY,
            (false, false, true, _) => GATE_UP_SWIGLU_RMSNORM_VEC_OCC_ENTRY,
            (false, false, false, true) => GATE_UP_SWIGLU_RMSNORM_VEC_ENTRY,
            (false, false, false, false) => GATE_UP_SWIGLU_RMSNORM_ENTRY,
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_gate_ptr = cuptr(packed_gate.data_ptr::<u8>() as *const c_void);
        let scales_gate_ptr = cuptr(scales_gate.data_ptr::<u8>() as *const c_void);
        let packed_up_ptr = cuptr(packed_up.data_ptr::<u8>() as *const c_void);
        let scales_up_ptr = cuptr(scales_up.data_ptr::<u8>() as *const c_void);
        let zp_gate_ptr = zp_gate
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let zp_up_ptr = zp_up
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row byte count", zp_row_bytes)?;
        let gamma_is_half: i32 = (gamma.dtype == DataType::Float16) as i32;
        let epsilon = self.rmsnorm_epsilon;
        let threads = GATE_UP_SWIGLU_THREADS;
        let columns_per_block = (threads / 32) as usize;
        let shared_mem_bytes = (self.k * std::mem::size_of::<half::f16>()) as u32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_gate_ptr)
            .arg(&scales_gate_ptr)
            .arg(&packed_up_ptr)
            .arg(&scales_up_ptr)
            .arg(&zp_gate_ptr)
            .arg(&zp_up_ptr)
            .arg(&gamma_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes)
            .arg(&gamma_is_half)
            .arg(&epsilon);
        // SAFETY: restricted to fp16 block-32 M=1 inputs validated above; the
        // weight/scale/gamma sets and the output are fixed device pointers, the
        // scalar ABI matches the paired RMS-norm entry point, and the kernel
        // stages the normalized activation in launch-time dynamic shared memory
        // (`K * sizeof(f16)`, bounded by the fusion's `K % 128 == 0` predicate)
        // using only registers, warp shuffles, and `__syncthreads` — no per-call
        // allocation or host sync — so it is legal to record into and replay
        // from a CUDA graph.
        // Clamp the K-sized activation stage to the device's shared-memory
        // budget: opt into >48 KB when the card allows it, and fail loudly
        // (rather than launch-crash on a smaller consumer GPU) if K exceeds even
        // the opt-in ceiling.
        self.runtime
            .configure_dynamic_shared_memory(&function, shared_mem_bytes)?;
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 gate/up SwiGLU RMS-norm GEMV", err))
    }

    /// Prefill (M>1) gate/up SwiGLU with an RMS-normalization prologue.
    /// Normalizes each token row into scratch (byte-identical to
    /// `skip_rmsnorm_f16_warp_half4`), then runs the standard paired gate/up
    /// SwiGLU prefill over the normalized activation. Prefill is outside the
    /// persistent decode graph, so the scratch allocation is not on any captured
    /// path.
    #[allow(clippy::too_many_arguments)]
    fn launch_gate_up_swiglu_rmsnorm_prefill(
        &self,
        activation: &TensorView,
        packed_gate: &TensorView,
        scales_gate: &TensorView,
        packed_up: &TensorView,
        scales_up: &TensorView,
        zp_gate: Option<&TensorView>,
        zp_up: Option<&TensorView>,
        gamma: &TensorView,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        _zp_row_bytes: usize,
    ) -> Result<()> {
        let scratch = self
            .runtime
            .alloc_raw(m * self.k * std::mem::size_of::<half::f16>())?;
        let scratch_shape = [m, self.k];
        let scratch_strides = [self.k as i64, 1];
        let normalized = TensorView::new(
            DevicePtr(raw_ptr(scratch) as *const c_void),
            DataType::Float16,
            &scratch_shape,
            &scratch_strides,
            activation.device,
        );
        let result = self
            .launch_rmsnorm_prefill(activation, gamma, scratch, m)
            .and_then(|()| {
                self.launch_gate_up_swiglu_prefill(
                    &normalized,
                    packed_gate,
                    scales_gate,
                    packed_up,
                    scales_up,
                    zp_gate,
                    zp_up,
                    output,
                    m,
                    k_blocks,
                )
            });
        // SAFETY: `scratch` came from `alloc_raw` above and is freed exactly
        // once; `cuMemFree` waits for the preceding norm + GEMM stream work.
        let free_scratch = unsafe { self.runtime.free_raw(scratch) };
        result.and(free_scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let selection = select_f16_gemv_variant(
            self.k,
            self.n,
            self.block_size,
            scales_fp16,
            zero_points.is_some(),
        );
        // Non-block-32 layouts are served by the model-agnostic general_bs
        // kernel; tag them distinctly so nsys/trace timelines can tell the
        // block-size-general decode GEMV apart from a tuned block-32 general one.
        let variant_name = if self.block_size != 32 {
            "gemv_f16_general_bs"
        } else {
            match selection.variant {
                F16GemvVariant::DownProjection => "gemv_f16_down_projection",
                F16GemvVariant::General => "gemv_f16_general",
            }
        };
        onnx_runtime_ep_api::record_kernel_variant!(
            variant_name,
            "fp16-activation x int{} M==1 decode GEMV: block_size={}; zero_points={}; {}",
            self.bits,
            self.block_size,
            zero_points.is_some(),
            selection.reason
        );
        if bias.is_some() {
            if self.fold_bias_post_round {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "qkv_bias_fused",
                    "folded standalone Add(MatMulNBits, bias) into GEMV epilogue with \
                     fp16-after-round semantics fp16(fp16(acc)+bias) (token-identity preserved)"
                );
            } else {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "bias_native",
                    "native MatMulNBits bias: single-round epilogue fp16(acc+bias)"
                );
            }
        }
        self.launch_f16_gemv_variant(
            activation,
            packed,
            scales,
            scales_fp16,
            zero_points,
            bias,
            output,
            k_blocks,
            blob_size,
            zp_row_bytes,
            selection,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemv_variant(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
        selection: F16GemvSelection,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 GEMV")?;
        // Split-K routing for standalone block-32, scales-fp16 general GEMVs.
        // The existing asymmetric path retains its measured gate; symmetric
        // projections opt in only when the live SM count says their warp grid is
        // too narrow, so smaller consumer GPUs keep the current single-warp path.
        let use_scales_f16_zp_splitk = self.block_size == 32
            && scales_fp16
            && zero_points.is_some()
            && matches!(selection.variant, F16GemvVariant::General)
            && self.k.is_multiple_of(256)
            // Split-K needs >= K_SPLIT warps/block; the small-shape path uses only
            // 64 threads (2 warps), so restrict to the 256-thread large path.
            && !(self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX);
        let capabilities = self.runtime.capabilities();
        let use_scales_f16_symmetric_splitk = self.block_size == 32
            && scales_fp16
            && zero_points.is_none()
            && matches!(selection.variant, F16GemvVariant::General)
            && use_f16_symmetric_splitk(
                self.k,
                self.n,
                capabilities.multiprocessor_count(),
                capabilities.max_threads_per_block(),
            );
        let use_scales_f16_splitk = use_scales_f16_zp_splitk || use_scales_f16_symmetric_splitk;
        // Prefetch-pipeline routing for the SINGLE-WARP block-32 scales-fp16 int4
        // GEMV (the entry taken when split-K is not selected). Byte-identical to
        // the original entry (same mapping/math/order) but keeps 2 weight loads
        // in flight per lane to hide the Long-Scoreboard latency that dominates
        // the grid-starved projections. Default-on; `ONNX_GENAI_GEMV_PIPELINE=0`
        // forces the original entry.
        //
        // Occupancy gate: the pipe entry's extra registers are a net LOSS once
        // the launch already fills the SMs many waves over (the pipe kernel's
        // lower occupancy then outweighs its latency hiding). The wide LM-head
        // projection is the only such well-occupied block-32 GEMV here (measured
        // plain 85.0 µs vs pipe 98.8 µs, -14%), so it falls back to the plain
        // entry while the grid-starved q/gate/kv projections keep the pipe entry.
        // `pipe_columns_per_block` mirrors the launch's `columns_per_block`: the
        // pipe path always takes the 256-thread (8-warp) large launch unless BOTH
        // n and k are small.
        let use_scales_f16_pipeline = self.block_size == 32
            && scales_fp16
            && matches!(selection.variant, F16GemvVariant::General)
            && !use_scales_f16_splitk
            && scales_f16_pipeline_enabled();
        let pipe_columns_per_block =
            if self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX {
                (GEMV_F16_SMALL_THREADS / 32) as usize
            } else {
                (GEMV_F16_LARGE_THREADS / 32) as usize
            };
        let use_scales_f16_pipe = use_scales_f16_pipeline
            && !scales_f16_pipe_well_occupied(
                self.n,
                pipe_columns_per_block,
                self.runtime.capabilities().multiprocessor_count(),
            );
        // Down projection grid-fill: choose columns-per-CTA (8/4/2) from the
        // device SM count so a small-N (grid-starved) down launch splits enough
        // to fill the multiprocessors, bit-identically. A developer env override
        // is honored for A/B measurement.
        let down_choice = down_columns_override().unwrap_or_else(|| {
            select_down_columns(self.n, self.runtime.capabilities().multiprocessor_count())
        });
        // Grid-fill split-K for the block!=32 int4 decode GEMV: the single-warp
        // general_bs launch under-fills the SMs on the medium/KV projections
        // (~0.5 waves), starving the latency-bound global loads. K_SPLIT warps
        // per column doubles the grid to fill the device; the wide gate_up GEMV
        // (already >3 waves) keeps the single-warp entry.
        let use_general_splitk = self.block_size != 32
            && use_general_bs_splitk(
                self.k,
                self.n,
                self.bits,
                self.runtime.capabilities().multiprocessor_count(),
            );
        let entry = if self.block_size != 32 {
            // Any non-block-32 layout uses the model-agnostic general kernel; the
            // tuned DownProjection / scales_f16 / general entries bake in the
            // block-32 lane→block mapping. `select_f16_gemv_variant` already
            // returns `General` for block_size != 32, so shape/thread selection
            // below (the `General` arm) applies unchanged. The general_bs kernel
            // dequantizes an optional asymmetric zero point per block, so it is
            // correct for both symmetric (zp==8) and asymmetric layouts.
            if use_general_splitk {
                if use_gemv_wideload(self.bits, self.block_size, self.k) {
                    // The multicol hybrid register-blocks WIDE_NC columns/warp,
                    // which lifts memory-level parallelism on the medium/large
                    // projections but collapses the launch grid on a narrow
                    // (grid-starved) projection. For those, fall back to the
                    // byte-identical single-column split-K wide entry, whose 4x
                    // larger grid fills the otherwise-idle SMs.
                    if use_gemv_splitk_multicol()
                        && !splitk_smalln_prefers_single_column(
                            self.n,
                            self.runtime.capabilities().multiprocessor_count(),
                        )
                    {
                        GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_ENTRY
                    } else {
                        GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY
                    }
                } else {
                    GEMV_F16_GENERAL_BS_SPLITK_ENTRY
                }
            } else if use_gemv_wideload(self.bits, self.block_size, self.k) {
                if use_gemv_wide_multicol() {
                    if use_gemv_fp16() {
                        GEMV_F16_GENERAL_BS_WIDE_MULTICOL_FP16_ENTRY
                    } else {
                        GEMV_F16_GENERAL_BS_WIDE_MULTICOL_ENTRY
                    }
                } else {
                    GEMV_F16_GENERAL_BS_WIDE_ENTRY
                }
            } else {
                GEMV_F16_GENERAL_BS_ENTRY
            }
        } else {
            match selection.variant {
                F16GemvVariant::DownProjection => down_choice.1,
                // The vectorized `scales_f16` kernel is compiled in two
                // specializations: the symmetric entry (`HasZp == false`) folds
                // the subtrahend to the constant fp16 8.0 with zero extra memory
                // traffic (byte-identical PTX to the pre-zero-point path), while
                // the `_zp` entry reads the per-block asymmetric zero point.
                // Symmetric weights must take the constant path so the memory-
                // bound M=1 decode GEMV does not pay for an unused per-block load.
                F16GemvVariant::General if scales_fp16 => {
                    if zero_points.is_some() {
                        // Grid-starved standalone int4 zp GEMV: when K is a whole
                        // multiple of the 256-wide step, take the split-K entry
                        // (K_SPLIT warps/column, K_SPLIT x larger grid) to fill the
                        // SMs; otherwise the plain single-warp `_zp` entry (which
                        // handles the divergent K tail).
                        if use_scales_f16_zp_splitk {
                            GEMV_F16_SCALES_F16_ZP_SPLITK_ENTRY
                        } else if use_scales_f16_pipe {
                            GEMV_F16_SCALES_F16_ZP_PIPE_ENTRY
                        } else {
                            GEMV_F16_SCALES_F16_ZP_ENTRY
                        }
                    } else {
                        if use_scales_f16_symmetric_splitk {
                            GEMV_F16_SCALES_F16_SPLITK_ENTRY
                        } else if use_scales_f16_pipe {
                            GEMV_F16_SCALES_F16_PIPE_ENTRY
                        } else {
                            GEMV_F16_SCALES_F16_ENTRY
                        }
                    }
                }
                F16GemvVariant::General => GEMV_F16_ENTRY,
            }
        };
        // Opt-in TRT-LLM interleaved + biased dequant lever. Keying off the
        // ALREADY-selected `entry` (rather than re-deriving the shape predicates)
        // guarantees each swap is a drop-in with identical launch geometry. The
        // wide multicol and the grid-starved split-K wide entries (block!=32,
        // fp32-accum) consume offline nibble-interleaved weights so runtime
        // dequant drops both the per-block `sub.f16x2` (folded -8 bias) and the
        // `prmt.b32` activation reorder; they swap the packed pointer for the
        // cached interleaved buffer AND route to the interleaved sibling kernel.
        // Byte-identical on symmetric weights.
        //
        // The block-32 `scales_f16` fp16-accum family was evaluated (a pure
        // bias-fold, no weight interleave) and measured a NO-GO: byte-identical
        // but no decode speedup on qwen2.5-14b-int4 (that kernel is not
        // issue-slot-bound the way the fp32 wide path is), so it is deliberately
        // NOT wired here. Any other entry (narrow single-warp, fp16-accum
        // multicol, asymmetric) is likewise left untouched.
        let interleaved_entry = if entry == GEMV_F16_GENERAL_BS_WIDE_MULTICOL_ENTRY {
            Some(GEMV_F16_GENERAL_BS_WIDE_MULTICOL_INTERLEAVED_ENTRY)
        } else if entry == GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_ENTRY {
            Some(GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_INTERLEAVED_ENTRY)
        } else if entry == GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY {
            Some(GEMV_F16_GENERAL_BS_SPLITK_WIDE_INTERLEAVED_ENTRY)
        } else {
            None
        };
        let orig_packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let interleave_on = interleave_dequant_enabled() && self.bits == 4 && zero_points.is_none();
        let (entry, packed_ptr) = if interleave_on && interleaved_entry.is_some() {
            let target = interleaved_entry.unwrap();
            let bytes = self.n.saturating_mul(k_blocks).saturating_mul(blob_size);
            match ensure_interleaved(&self.runtime, orig_packed_ptr, bytes) {
                Ok((iptr, warm)) => {
                    if !warm {
                        self.last_call_capture_safe.store(false, Ordering::Relaxed);
                    }
                    onnx_runtime_ep_api::record_kernel_variant_stage!(
                        "dequant",
                        "interleaved_biased",
                        "TRT-LLM interleaved+biased int4 dequant: offline nibble-interleave + \
                         folded symmetric -8 bias drops the per-block sub.f16x2 and the prmt.b32 \
                         activation reorder (byte-identical to the fp32 wide/multicol/split-K path)"
                    );
                    (target, iptr)
                }
                Err(_) => (entry, orig_packed_ptr),
            }
        } else {
            (entry, orig_packed_ptr)
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        // These two entries narrow their fp16 epilogue through
        // `matmul_nbits_store_narrowed`, so they can write a bf16 consumer
        // buffer directly and let `run_bf16` drop its per-node staging cast.
        let bf16_direct_capable = entry == GEMV_F16_SCALES_F16_SPLITK_ENTRY
            || entry == GEMV_F16_SCALES_F16_ZP_SPLITK_ENTRY;
        let staging_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let direct_bf16_ptr = if bf16_direct_capable && gemv_bf16_direct_out_enabled() {
            take_bf16_direct_out(staging_ptr)
        } else {
            None
        };
        let output_ptr = direct_bf16_ptr.unwrap_or(staging_ptr);
        let out_bf16_flag: i32 = direct_bf16_ptr.is_some() as i32;
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row byte count", zp_row_bytes)?;
        let scales_fp16_flag: i32 = scales_fp16 as i32;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let bits = as_i32("bits", self.bits)?;
        let (threads, columns_per_block, shared_mem_bytes) = match selection.variant {
            F16GemvVariant::DownProjection => (GEMV_F16_DOWN_THREADS, down_choice.0, 0),
            F16GemvVariant::General => {
                let threads = if self.n <= GEMV_F16_SMALL_N_MAX
                    && self.k <= GEMV_F16_SMALL_N_MAX
                    && !use_general_splitk
                {
                    GEMV_F16_SMALL_THREADS
                } else {
                    GEMV_F16_LARGE_THREADS
                };
                // Split-K assigns K_SPLIT warps per output column, so a block of
                // `threads/32` warps now covers `warps / K_SPLIT` columns and the
                // grid grows by K_SPLIT to fill the SMs. The block-32 scales_f16
                // split-K uses K_SPLIT = GEMV_F16_SCALES_F16_ZP_SPLITK; the
                // block!=32 general_bs split-K uses GENERAL_BS_SPLITK. Both force
                // the 256-thread path so `threads/32 >= K_SPLIT`.
                let columns_per_block = if use_general_splitk {
                    // Split-K covers `warps / K_SPLIT` column groups; the multicol
                    // hybrid register-blocks WIDE_NC output columns per group, so
                    // it covers that many more columns per block.
                    if entry == GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_ENTRY
                        || entry == GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_INTERLEAVED_ENTRY
                    {
                        (threads / 32) as usize / GENERAL_BS_SPLITK_MULTICOL
                            * GEMV_F16_WIDE_MULTICOL_NC
                    } else {
                        (threads / 32) as usize / GENERAL_BS_SPLITK
                    }
                } else if use_scales_f16_splitk {
                    (threads / 32) as usize / GEMV_F16_SCALES_F16_ZP_SPLITK
                } else if entry == GEMV_F16_GENERAL_BS_WIDE_MULTICOL_ENTRY
                    || entry == GEMV_F16_GENERAL_BS_WIDE_MULTICOL_FP16_ENTRY
                    || entry == GEMV_F16_GENERAL_BS_WIDE_MULTICOL_INTERLEAVED_ENTRY
                {
                    // Each warp emits WIDE_NC columns, so a `threads/32`-warp CTA
                    // covers `warps * WIDE_NC` output columns; the grid shrinks by
                    // WIDE_NC accordingly.
                    (threads / 32) as usize * GEMV_F16_WIDE_MULTICOL_NC
                } else {
                    (threads / 32) as usize
                };
                (threads, columns_per_block, 0)
            }
        };
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes)
            .arg(&scales_fp16_flag)
            .arg(&bias_post_round_flag);
        // The model-agnostic general-block-size GEMV takes one extra trailing
        // scalar (`bits`) to select the packed int4/int8 layout for any block
        // width; the tuned block-32 entries bake in their bit width and take none.
        // The general_bs split-K entry shares the same trailing `bits` ABI.
        if entry == GEMV_F16_GENERAL_BS_ENTRY
            || entry == GEMV_F16_GENERAL_BS_SPLITK_ENTRY
            || entry == GEMV_F16_GENERAL_BS_WIDE_ENTRY
            || entry == GEMV_F16_GENERAL_BS_SPLITK_WIDE_ENTRY
            || entry == GEMV_F16_GENERAL_BS_WIDE_MULTICOL_ENTRY
            || entry == GEMV_F16_GENERAL_BS_WIDE_MULTICOL_FP16_ENTRY
            || entry == GEMV_F16_GENERAL_BS_WIDE_MULTICOL_INTERLEAVED_ENTRY
            || entry == GEMV_F16_GENERAL_BS_SPLITK_WIDE_INTERLEAVED_ENTRY
            || entry == GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_ENTRY
            || entry == GEMV_F16_GENERAL_BS_SPLITK_WIDE_MULTICOL_INTERLEAVED_ENTRY
        {
            builder.arg(&bits);
        }
        // The block-32 split-K entries take one extra trailing scalar selecting
        // the fp16 or bf16 narrowing store in their epilogue.
        if bf16_direct_capable {
            builder.arg(&out_bf16_flag);
        }
        // SAFETY: M=1 fp16 inputs; all tensors were dtype/shape/contiguity
        // validated above, including the optional packed per-block zero-point
        // rows. Block-32 layouts use the tuned entries; any other (power-of-two,
        // >=16) block size routes to the general_bs entry, which derives the
        // scale/zero-point block index from `block_size`. The scalar ABI is
        // shared by all these entries. Every variant uses only registers and
        // launch-time shared memory (no per-call alloc or sync), so the launch
        // is legal to record into and replay from a CUDA graph.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|err| {
            driver_err(
                &format!("launch MatMulNBits fp16 GEMV ({})", selection.reason),
                err,
            )
        })
    }

    /// General fp16 GEMV whose input activation is RMS-normalized in-kernel
    /// (the [`GEMV_F16_SCALES_F16_RMSNORM_ENTRY`] entry). Bit-for-bit identical
    /// to the standalone `SkipSimplifiedLayerNormalization` residual output
    /// followed by the general `scales_f16` GEMV, so decode tokens are
    /// unchanged while the separate normalization launch is removed.
    #[allow(clippy::too_many_arguments)]
    fn launch_f16_gemv_rmsnorm(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        gamma: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        self.runtime
            .require_nvrtc_half_headers("MatMulNBits fp16 RMS-norm-prologue GEMV")?;
        if bias.is_some() {
            if self.fold_bias_post_round {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "qkv_bias_fused",
                    "folded standalone Add(MatMulNBits, bias) into GEMV epilogue with \
                     fp16-after-round semantics fp16(fp16(acc)+bias) (token-identity preserved)"
                );
            } else {
                onnx_runtime_ep_api::record_kernel_variant_stage!(
                    "bias",
                    "bias_native",
                    "native MatMulNBits bias: single-round epilogue fp16(acc+bias)"
                );
            }
        }
        let capabilities = self.runtime.capabilities();
        let use_splitk = zero_points.is_none()
            && use_f16_symmetric_splitk(
                self.k,
                self.n,
                capabilities.multiprocessor_count(),
                capabilities.max_threads_per_block(),
            );
        let entry = if zero_points.is_some() {
            GEMV_F16_SCALES_F16_RMSNORM_ZP_ENTRY
        } else if use_splitk {
            GEMV_F16_SCALES_F16_RMSNORM_SPLITK_ENTRY
        } else {
            GEMV_F16_SCALES_F16_RMSNORM_ENTRY
        };
        let function = self
            .runtime
            .nvrtc_function(GEMV_F16_MODULE, GEMV_F16_SRC, entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let gamma_ptr = cuptr(gamma.data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row byte count", zp_row_bytes)?;
        let bias_post_round_flag: i32 = (self.fold_bias_post_round && bias.is_some()) as i32;
        let gamma_is_half: i32 = (gamma.dtype == DataType::Float16) as i32;
        let epsilon = self.rmsnorm_epsilon;
        let threads = if self.n <= GEMV_F16_SMALL_N_MAX && self.k <= GEMV_F16_SMALL_N_MAX {
            GEMV_F16_SMALL_THREADS
        } else {
            GEMV_F16_LARGE_THREADS
        };
        let columns_per_block = (threads / 32) as usize
            / if use_splitk {
                GEMV_F16_SCALES_F16_ZP_SPLITK
            } else {
                1
            };
        let shared_mem_bytes = (self.k * std::mem::size_of::<half::f16>()) as u32;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&gamma_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes)
            .arg(&bias_post_round_flag)
            .arg(&gamma_is_half)
            .arg(&epsilon);
        // SAFETY: restricted to block-32 M=1 fp16 inputs with fp16 scales and no
        // zero_points, all dtype/shape/contiguity validated above. The kernel
        // stages the normalized activation in launch-time dynamic shared memory
        // (`K * sizeof(f16)`, bounded by the fusion's `K % 128 == 0` predicate)
        // and uses only registers, warp shuffles, and `__syncthreads` — no
        // per-call allocation or host synchronization — so it is legal to record
        // into and replay from a CUDA graph.
        // Clamp the K-sized activation stage to the device's shared-memory
        // budget: opt into >48 KB when the card allows it, and fail loudly
        // (rather than launch-crash on a smaller consumer GPU) if K exceeds even
        // the opt-in ceiling.
        self.runtime
            .configure_dynamic_shared_memory(&function, shared_mem_bytes)?;
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n.div_ceil(columns_per_block) as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits fp16 RMS-norm-prologue GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_int8_f32_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_INT8_F32_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks);
        // SAFETY: dense tensors were validated for the one-byte-per-weight
        // block-32 layout. This fixed-geometry launch uses no dynamic allocation,
        // host synchronization, or architecture-specific instructions, so it is
        // CUDA-graph-capturable and portable across supported SM versions.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n as u32, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits int8 f32 GEMV", err))
    }

    /// Specialized asymmetric int4 / `block_size == 128` fp32-activation decode
    /// GEMV. The fixed layout makes `blob_size == 64` and
    /// `zp_row_bytes == ceil(k_blocks / 2)`, so neither is passed at runtime.
    #[allow(clippy::too_many_arguments)]
    fn launch_int4_f32_gemv_block128(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        let function =
            self.runtime
                .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_INT4_F32_BLOCK128_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = cuptr(zero_points.data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks);
        // SAFETY: validation guarantees the asymmetric int4 block-128 layout.
        // The fixed one-CTA-per-column launch has no allocation, synchronization,
        // or architecture-specific launch geometry, so it is capture-safe.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n as u32, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits int4 f32 block-128 GEMV", err))
    }

    /// Specialized int8 / `block_size == 128` / asymmetric fp32-activation decode
    /// GEMV. Structurally selected from [`Self::launch_f32_gemv`] and bit-for-bit
    /// identical to it; see [`GEMV_INT8_F32_BLOCK128_ENTRY`] for the kernel-level
    /// rationale. Only `k`, `n`, and `k_blocks` are passed because `block_size`,
    /// `blob_size`, `zp_row_bytes`, and `bits` are all fixed by the structural
    /// dispatch (128 / 128 / `k_blocks` / 8).
    #[allow(clippy::too_many_arguments)]
    fn launch_int8_f32_gemv_block128(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        let function =
            self.runtime
                .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_INT8_F32_BLOCK128_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks);
        // SAFETY: dense tensors were validated for the one-byte-per-weight
        // block-128 layout (blob_size == 128). The launch geometry (one CTA per
        // column, BLOCK_THREADS lanes, no dynamic allocation, host sync, or
        // architecture-specific instructions) matches the generic f32 GEMV, so it
        // is CUDA-graph-capturable and portable across supported SM counts/CCs.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n as u32, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits int8 f32 block-128 GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_f32_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_F32_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row size", zp_row_bytes)?;
        let bits = as_i32("bits", self.bits)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes)
            .arg(&bits);
        // SAFETY: validated dense tensors cover the complete M=1 operation and
        // the scalar ABI matches `matmul_nbits_gemv_f32`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n as u32, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits f32 GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_accuracy4_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        let workspace = self
            .accuracy4_workspace
            .as_ref()
            .ok_or_else(|| error("accuracy_level=4 GEMV workspace is unavailable"))?
            .lock()
            .map_err(|_| error("accuracy_level=4 GEMV workspace lock poisoned"))?;
        let quantize_function =
            self.runtime
                .nvrtc_function(GEMV_MODULE, GEMV_SRC, QUANTIZE_ACCURACY4_ENTRY)?;
        let capabilities = self.runtime.capabilities();
        let stage64 = use_accuracy4_stage64(
            self.n,
            capabilities.multiprocessor_count(),
            capabilities.compute_capability(),
            capabilities.max_shared_memory_per_block_optin(),
        );
        let gemv_entry = if stage64 {
            GEMV_ACCURACY4_STAGE64_ENTRY
        } else {
            GEMV_ACCURACY4_ENTRY
        };
        let gemv_function = self
            .runtime
            .nvrtc_function(GEMV_MODULE, GEMV_SRC, gemv_entry)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let padded_k = as_i32("padded K", workspace.padded_k)?;

        let mut quantize_builder = self.runtime.stream().launch_builder(&quantize_function);
        quantize_builder
            .arg(&activation_ptr)
            .arg(&workspace.quantized_activation)
            .arg(&workspace.activation_scale)
            .arg(&k)
            .arg(&padded_k);
        // SAFETY: the persistent workspace covers padded_k int8 values plus one
        // f32 scale per block-32, and the scalar ABI matches the quantization
        // entry point. One warp (CUDA block) quantizes one K-block.
        unsafe {
            quantize_builder.launch(LaunchConfig {
                grid_dim: ((workspace.padded_k / 32) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4 quantization", err))?;

        let mut gemv_builder = self.runtime.stream().launch_builder(&gemv_function);
        gemv_builder
            .arg(&workspace.quantized_activation)
            .arg(&workspace.activation_scale)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks);
        // SAFETY: this path is restricted to symmetric block-32 M=1 inputs; the
        // persistent quantized activation is initialized by the preceding stream
        // launch, and the scalar ABI matches the tiled GEMV entry point.
        unsafe {
            gemv_builder.launch(LaunchConfig {
                grid_dim: (
                    self.n.div_ceil(GEMV_ACCURACY4_COLUMNS_PER_BLOCK) as u32,
                    1,
                    1,
                ),
                block_dim: (GEMV_ACCURACY4_THREADS, 1, 1),
                shared_mem_bytes: if stage64 {
                    GEMV_ACCURACY4_STAGE64_SHARED_BYTES
                } else {
                    GEMV_ACCURACY4_SHARED_BYTES
                },
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4 GEMV", err))
    }

    /// General-block-size fp32-activation accuracy_level=4 decode GEMV. Quantizes
    /// the fp32 activation to int8 ONCE per K-block into the persistent workspace
    /// (matching the tiled reference's per-block quantization), then runs a
    /// warp-per-column blockwise GEMV whose grid width is chosen from the device
    /// multiprocessor count so it fills consumer and datacenter GPUs alike. This
    /// replaces the grid-starved tiled `matmul_nbits_accuracy4` fallback for M==1
    /// int4 decode (any power-of-two block_size, symmetric or asymmetric),
    /// bit-for-bit identically, and eliminates the per-output re-quantization.
    #[allow(clippy::too_many_arguments)]
    fn launch_accuracy4_gemv_blockwise(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let workspace = self
            .accuracy4_workspace
            .as_ref()
            .ok_or_else(|| error("accuracy_level=4 GEMV workspace is unavailable"))?
            .lock()
            .map_err(|_| error("accuracy_level=4 GEMV workspace lock poisoned"))?;
        let quantize_function = self.runtime.nvrtc_function(
            GEMV_MODULE,
            GEMV_SRC,
            QUANTIZE_ACCURACY4_BLOCKWISE_ENTRY,
        )?;
        let gemv_function =
            self.runtime
                .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_ACCURACY4_BLOCKWISE_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks_arg = as_i32("K block count", k_blocks)?;
        let blob_size_arg = as_i32("block blob size", blob_size)?;
        let zp_row_bytes_arg = as_i32("zero-point row size", zp_row_bytes)?;
        let padded_k = as_i32("padded K", workspace.padded_k)?;

        let mut quantize_builder = self.runtime.stream().launch_builder(&quantize_function);
        quantize_builder
            .arg(&activation_ptr)
            .arg(&workspace.quantized_activation)
            .arg(&workspace.activation_scale)
            .arg(&k)
            .arg(&block_size)
            .arg(&padded_k);
        // SAFETY: the persistent workspace covers padded_k int8 values plus one
        // f32 scale per K-block, and the scalar ABI matches the quantization
        // entry point. One warp (CUDA block) quantizes one K-block.
        unsafe {
            quantize_builder.launch(LaunchConfig {
                grid_dim: (k_blocks as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|err| {
            driver_err(
                "launch MatMulNBits accuracy_level=4 blockwise quantization",
                err,
            )
        })?;

        let warps =
            select_accuracy4_gemv_warps(self.n, self.runtime.capabilities().multiprocessor_count());
        let grid = self.n.div_ceil(warps as usize) as u32;
        let mut gemv_builder = self.runtime.stream().launch_builder(&gemv_function);
        gemv_builder
            .arg(&workspace.quantized_activation)
            .arg(&workspace.activation_scale)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks_arg)
            .arg(&block_size)
            .arg(&blob_size_arg)
            .arg(&zp_row_bytes_arg);
        // SAFETY: the persistent quantized activation is initialized by the
        // preceding stream launch, the scalar ABI matches the blockwise GEMV
        // entry point, and each warp reduces exactly one output column.
        unsafe {
            gemv_builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (warps * 32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4 blockwise GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_accuracy4(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let total = m.checked_mul(self.n).ok_or_else(|| {
            error(format!(
                "accuracy_level=4 output size {m} * {} overflows usize",
                self.n
            ))
        })?;
        let blocks = total.div_ceil(BLOCK_THREADS as usize).clamp(1, 65_535) as u32;
        let function =
            self.runtime
                .nvrtc_function(ACCURACY4_MODULE, ACCURACY4_SRC, ACCURACY4_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let m = as_i32("M", m)?;
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row size", zp_row_bytes)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes);
        // SAFETY: all tensors were dtype/shape/contiguity validated above and
        // the scalar ABI matches `matmul_nbits_accuracy4`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4", err))
    }

    #[allow(clippy::too_many_arguments)]
    /// Dequantize the int4/int8 weights to `[K, N]` fp16 and run the GEMM on
    /// cuBLASLt tensor cores.
    ///
    /// Returns `Ok(false)` when this call is not eligible, so the caller falls
    /// through to Marlin / the tiled GEMM under the usual fallback contract.
    #[allow(clippy::too_many_arguments)]
    fn try_dequant_f16_cublas_gemm(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        bias_row_stride: usize,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
        workspace: Option<WorkspaceView>,
    ) -> Result<bool> {
        // cuBLASLt's bias epilogue adds one value per output column into the f32
        // accumulator before rounding. That is exactly `bias_post_round == 0`
        // with a broadcast bias; a per-token residual (row stride N) or a folded
        // standalone `Add` (which must round the accumulator *first*) would be a
        // different computation, so both stay on the existing paths.
        if bias.is_some() && (bias_row_stride != 0 || self.fold_bias_post_round) {
            return Ok(false);
        }
        let Some(block_shift) = self.dequant_f16_block_shift() else {
            return Ok(false);
        };
        let weight_bytes = self
            .k
            .checked_mul(self.n)
            .and_then(|elems| elems.checked_mul(2))
            .ok_or_else(|| error("dequantized f16 weight size overflowed"))?;
        if weight_bytes > dequant_f16_gemm_max_scratch_bytes() {
            return Ok(false);
        }

        // A persistent pooled scratch, not a per-call allocation. Prefill runs
        // this hundreds of times per generation, and the VMM arena charges
        // `cuMemCreate`/`cuMemSetAccess`/`cuMemUnmap` for every map and unmap of
        // a buffer this large — enough allocator time to swamp the GEMM the
        // dequantize exists to feed. Slot 5 is unused by the Marlin fused paths.
        let (weight, _warm) = marlin_gemm::ensure_scratch(&self.runtime, 5, weight_bytes)?;
        let result = self
            .launch_dequant_f16(
                packed,
                scales,
                scales_fp16,
                zero_points,
                weight,
                k_blocks,
                blob_size,
                zp_row_bytes,
                block_shift,
            )
            .and_then(|()| {
                let params = self.dequant_f16_gemm_ex(
                    cuptr(activation.data_ptr::<u8>() as *const c_void),
                    weight,
                    cuptr(output.data_ptr_mut::<u8>() as *const c_void),
                    m,
                    bias.map(|bias| cuptr(bias.data_ptr::<u8>() as *const c_void)),
                );
                // SAFETY: A is the freshly written [N, K] fp16 dequantized
                // weight, B the validated [M, K] fp16 activation, and C the op's
                // [M, N] fp16 output, which aliases neither. The workspace and
                // stream outlive the call.
                unsafe {
                    blas::governed_gemm_ex(
                        self.runtime.blas(),
                        self.runtime.stream_ptr(),
                        &params,
                        workspace,
                        "MatMulNBits",
                    )
                }
            });

        // The scratch stays pooled for the next call; nothing to release here.
        result.map(|()| true)
    }

    /// The one place the dequantize + fp16 GEMM's column-major mapping lives, so
    /// the launch and the declared workspace requirement cannot drift apart.
    ///
    /// cuBLAS is column-major, and the row-major product `C[M, N] = A[M, K] ·
    /// W[K, N]` is the column-major product `Cᶜ[N, M] = Wᵀ · Aᶜ`. The
    /// dequantized weight is `[N, K]` row-major, i.e. `K × N` column-major, so
    /// transposing it gives the `N × K` first operand; the `[M, K]` row-major
    /// activation is already `K × M` column-major and needs no transpose. The
    /// epilogue bias then indexes the `m = N` axis, which is the output channel
    /// a MatMulNBits bias is defined over.
    fn dequant_f16_gemm_ex(
        &self,
        activation: CUdeviceptr,
        weight_nk: CUdeviceptr,
        output: CUdeviceptr,
        m: usize,
        bias: Option<CUdeviceptr>,
    ) -> GemmEx {
        GemmEx {
            dtype: GemmDtype::F16,
            transa: true,
            transb: false,
            m: self.n,
            n: m,
            k: self.k,
            alpha: 1.0,
            beta: 0.0,
            a: weight_nk,
            lda: self.k,
            b: activation,
            ldb: self.k,
            c: output,
            ldc: self.n,
            epilogue: bias.map(|bias| GemmEpilogue {
                kind: GemmEpilogueKind::Bias,
                bias,
            }),
        }
    }

    /// Whether this node's quantization layout is expressible by the packed
    /// eight-nibble word the fp16 dequant kernel reads.
    fn dequant_f16_block_shift(&self) -> Option<i32> {
        if self.bits != 4 || self.k % 8 != 0 {
            return None;
        }
        if self.block_size % 8 != 0 || !self.block_size.is_power_of_two() {
            return None;
        }
        Some(self.block_size.trailing_zeros() as i32)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_dequant_f16(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        scales_fp16: bool,
        zero_points: Option<&TensorView>,
        weight: cudarc::driver::sys::CUdeviceptr,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
        block_shift: i32,
    ) -> Result<()> {
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let k = as_i32("K", self.k)?;
        let k_blocks_i = as_i32("K block count", k_blocks)?;
        let blob_size_i = as_i32("block blob size", blob_size)?;
        let zp_row_bytes_i = as_i32("zero-point row size", zp_row_bytes)?;
        let scales_fp16_flag: i32 = scales_fp16 as i32;
        let words = self.k / 8;
        let grid_x = words.div_ceil(BLOCK_THREADS as usize) as u32;
        let grid_y = as_i32("N", self.n)? as u32;
        let function =
            self.runtime
                .nvrtc_function(DEQUANT_F16_MODULE, DEQUANT_F16_SRC, DEQUANT_F16_ENTRY)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&weight)
            .arg(&k)
            .arg(&k_blocks_i)
            .arg(&blob_size_i)
            .arg(&zp_row_bytes_i)
            .arg(&block_shift)
            .arg(&scales_fp16_flag);
        // SAFETY: argument order/types match the CUDA entry point; all device
        // buffers were shape-validated and `weight` has N*K fp16 elements.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid_x.max(1), grid_y, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits f16 dequant", err))
    }

    fn launch_dequant(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        group_indices: Option<&TensorView>,
        weight: cudarc::driver::sys::CUdeviceptr,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let group_indices_ptr = group_indices
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row size", zp_row_bytes)?;
        let bits = as_i32("bits", self.bits)?;
        let total = self.k * self.n;
        let blocks = total.div_ceil(BLOCK_THREADS as usize).clamp(1, 65_535) as u32;
        let function = self
            .runtime
            .nvrtc_function(DEQUANT_MODULE, DEQUANT_SRC, DEQUANT_ENTRY)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&group_indices_ptr)
            .arg(&weight)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes)
            .arg(&bits);
        // SAFETY: argument order/types match the CUDA entry point; all device
        // buffers were shape-validated and `weight` has K*N f32 elements.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits dequant", err))
    }
}

impl Kernel for MatMulNBitsKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs, None)
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        self.workspace_requirement_for(inputs)
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
        // The m=1 no-g_idx GEMV uses only launch-time shared memory and a
        // shape-fixed persistent accuracy-4 activation workspace; it performs no
        // per-call allocation, D2H, or synchronization. The direct fp16 GEMV is
        // likewise capture-safe: fixed grid/block geometry from the shape
        // signature and register/launch-time-shared scratch.
        //
        // The opt-in Marlin M>1 tensor-core GEMM is ALSO capture-safe once its
        // weights are repacked: its launch grid is a pure function of (M, N) and,
        // on a warm repack-cache hit, the call performs no allocation, D2H, or
        // synchronization. The dispatch records this by storing the cache-warm
        // flag into `last_call_capture_safe`, so a cold (repacking) first call is
        // reported unsupported and only warm replays advertise support — which is
        // exactly what unlocks capture-stable speculative verification at M>1.
        //
        // The portable (non-Marlin) fp16 prefill GEMM remains conservatively
        // unadvertised because variable-M prefill is outside the persistent
        // decode graph and lacks replay coverage; f32 prefill scratch and g_idx
        // validation are also non-capturable.
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires M==1 decode GEMV without group_indices, or a warm-cache Marlin M>1 GEMM; a cold Marlin repack and portable prefill are outside the advertised capture contract and group_indices validation reads D2H",
            )
        }
    }
}

fn optional_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn required_positive_attr(node: &Node, name: &str) -> Result<usize> {
    let value = optional_int_attr(node, name)?
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))?;
    if value <= 0 {
        return Err(error(format!(
            "attribute '{name}' must be positive, got {value}"
        )));
    }
    Ok(value as usize)
}

fn optional_int_attr(node: &Node, name: &str) -> Result<Option<i64>> {
    match node.attr(name) {
        Some(attribute) => attribute
            .as_int()
            .map(Some)
            .ok_or_else(|| error(format!("attribute '{name}' must be an integer"))),
        None => Ok(None),
    }
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have dtype {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

// The RMS-norm prologue accepts an fp16 OR fp32 gamma. Gamma is only a final
// multiplicand (never in the fp32 variance accumulation), so an fp32 gamma is
// numerically safe and lets fp32-gamma exports (e.g. Phi-4-mini) fuse. The
// fused kernels branch on `gamma_is_half` to read gamma at full precision.
fn require_gamma_dtype(got: DataType) -> Result<()> {
    if got != DataType::Float16 && got != DataType::Float32 {
        return Err(error(format!(
            "gamma must have dtype Float16 or Float, got {got:?}"
        )));
    }
    Ok(())
}

fn require_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn require_flat_or_matrix_shape(
    name: &str,
    got: &[usize],
    rows: usize,
    columns: usize,
) -> Result<()> {
    if got != [rows * columns] && got != [rows, columns] {
        return Err(error(format!(
            "{name} must have shape [{}] or [{rows}, {columns}], got {got:?}",
            rows * columns
        )));
    }
    Ok(())
}

fn as_i32(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| error(format!("{name}={value} exceeds i32")))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep MatMulNBits: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use half::f16;

    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
    use onnx_runtime_ir::{DataType, DeviceId};

    use super::*;

    /// Guards the `ONNX_GENAI_INTERLEAVE_DEQUANT` / `ONNX_GENAI_GEMV_PIPELINE`
    /// levers, which the interleaved-dequant byte-identity test flips by writing
    /// the PROCESS environment.
    ///
    /// A plain mutex is not enough here, and used to be the bug: it only excludes
    /// other lock *holders*, while an env var is visible to every thread in the
    /// binary. Any parity test running concurrently would silently pick up the
    /// lever, route through `ensure_interleaved`, allocate the interleaved
    /// weights on first sight and therefore report that call capture-UNSAFE —
    /// failing an assertion about a kernel it never meant to exercise, at a
    /// timing that depended on the harness's thread interleaving.
    ///
    /// So the writer takes the exclusive side and every helper that depends on
    /// the levers being at their default (OFF) takes the shared side.
    static INTERLEAVE_TEST_ENV_LOCK: std::sync::OnceLock<std::sync::RwLock<()>> =
        std::sync::OnceLock::new();

    /// Exclusive guard that clears the dispatch levers when it drops.
    ///
    /// Restoring the environment on the normal path only is not enough: these
    /// helpers `unwrap()` their kernel runs, so a failing assertion unwinds past
    /// the cleanup and leaves the lever set. The `RwLock` is then poisoned, and
    /// readers — which recover with `into_inner` rather than cascade the
    /// failure — would run with a lever nobody asked for, turning one real
    /// failure into a run of unrelated ones. Clearing from `Drop` makes the
    /// restore unwind-safe.
    struct LeverEnvGuard(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

    impl LeverEnvGuard {
        fn acquire() -> Self {
            Self(
                INTERLEAVE_TEST_ENV_LOCK
                    .get_or_init(Default::default)
                    .write()
                    .unwrap_or_else(|e| e.into_inner()),
            )
        }
    }

    impl Drop for LeverEnvGuard {
        fn drop(&mut self) {
            // SAFETY: still inside the exclusive section, so no other thread is
            // reading the environment concurrently.
            unsafe {
                std::env::remove_var("ONNX_GENAI_INTERLEAVE_DEQUANT");
                std::env::remove_var("ONNX_GENAI_GENERAL_SPLITK");
                std::env::remove_var("ONNX_GENAI_GEMV_PIPELINE");
                std::env::remove_var("ONNX_GENAI_GATEUP_VEC");
                std::env::remove_var("ONNX_GENAI_GATEUP_OCC");
            }
        }
    }

    /// Shared guard for a test that requires the dispatch levers to be OFF.
    fn default_levers_guard() -> std::sync::RwLockReadGuard<'static, ()> {
        INTERLEAVE_TEST_ENV_LOCK
            .get_or_init(Default::default)
            .read()
            .unwrap_or_else(|e| e.into_inner())
    }

    // Qwen2.5-0.5B down-projection shape (K=intermediate, N=hidden). Used as a
    // test fixture for the tall-skinny down variant and, transposed, as the
    // gate/up shape — the runtime code never keys on these values.
    const QWEN_DOWN_K: usize = 4864;
    const QWEN_DOWN_N: usize = 896;
    const STAGED_DOWN_REFERENCE_ENTRY: &str =
        "matmul_nbits_gemv_f16_scales_f16_down_staged_reference";
    const STAGED_DOWN_REFERENCE_SRC: &str = r#"
extern "C" __global__ void matmul_nbits_gemv_f16_scales_f16_down_staged_reference(
    const __half* __restrict__ activation,
    const unsigned char* __restrict__ packed,
    const void* __restrict__ scales_raw,
    const unsigned char* __restrict__ zero_points,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes,
    const int scales_fp16,
    const int bias_post_round)
{
    (void)block_size;
    (void)zero_points;
    (void)zp_row_bytes;
    (void)scales_fp16;
    extern __shared__ uint4 activation_shared[];
    __shared__ float warp_sums[8][8];
    const __half* __restrict__ scales =
        reinterpret_cast<const __half*>(scales_raw);
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int column_base = (int)blockIdx.x * 8;

    for (int vector = tid; vector * 8 < k; vector += (int)blockDim.x) {
        activation_shared[vector] =
            permute_activation_f16x8(activation + vector * 8);
    }
    __syncthreads();

    float values[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int block = tid; block < k_blocks; block += (int)blockDim.x) {
        const uint4 activation0 = activation_shared[block * 4];
        const uint4 activation1 = activation_shared[block * 4 + 1];
        const uint4 activation2 = activation_shared[block * 4 + 2];
        const uint4 activation3 = activation_shared[block * 4 + 3];
#pragma unroll
        for (int tile_column = 0; tile_column < 8; ++tile_column) {
            const int column = column_base + tile_column;
            if (column < n) {
                const long packed_start =
                    ((long)column * k_blocks + block) * blob_size;
                const uint4 packed_weights =
                    *reinterpret_cast<const uint4*>(packed + packed_start);
                const __half scale = scales[(long)column * k_blocks + block];
                values[tile_column] += dot_int4x32_f16_permuted_scaled(
                    packed_weights,
                    activation0,
                    activation1,
                    activation2,
                    activation3,
                    scale);
            }
        }
    }

#pragma unroll
    for (int tile_column = 0; tile_column < 8; ++tile_column) {
        const float value = warp_sum(values[tile_column]);
        if (lane == 0) {
            warp_sums[warp][tile_column] = value;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < 8) {
        const int column = column_base + lane;
        float value = warp_sums[0][lane];
        value += warp_sums[1][lane];
        value += warp_sums[2][lane];
        value += warp_sums[3][lane];
        value += warp_sums[4][lane];
        value += warp_sums[5][lane];
        value += warp_sums[6][lane];
        value += warp_sums[7][lane];
        output[column] = fold_bias_f16(value, bias, column, bias_post_round);
    }
}
"#;

    fn runtime() -> Option<Arc<CudaRuntime>> {
        crate::test_support::maybe_runtime()
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

    /// Apples-to-apples wall-clock comparison of the Marlin M>1 tensor-core GEMM
    /// against the portable tiled CUDA-core GEMM through the real op, across a
    /// sweep of M at a production projection shape. Prints median kernel wall per
    /// path; this is the measurement that quantifies the M=1→M=2 cliff collapse.
    /// Ignored (perf, needs a dedicated idle SM80+ GPU).
    #[test]
    #[ignore = "perf microbench; requires a dedicated idle SM80+ CUDA device"]
    fn marlin_m_gt_1_op_wall_vs_tiled() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping Marlin M>1 wall bench: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("marlin_m_gt_1_wall")
            .is_err()
            || !marlin_gemm::device_supports_marlin(runtime.capabilities().compute_capability())
        {
            eprintln!("skipping Marlin M>1 wall bench: headers unavailable or pre-SM80");
            return;
        }

        let k = 5120usize;
        let n = 13824usize;
        let block_size = 128usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);

        let mut state = 0xcafef00d_1234_5678u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let max_m = 128usize;
        let mut activation_f16 = vec![f16::ZERO; max_m * k];
        for h in activation_f16.iter_mut() {
            *h = f16::from_f32(next());
        }
        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob_size {
                    let low = quant[col * k + block * block_size + pair * 2] & 15;
                    let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                    packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                }
            }
        }
        let mut zp_packed = vec![0u8; n * zp_row_bytes];
        for byte in zp_packed.iter_mut() {
            *byte = ((next() * 0.5 + 0.5) * 255.0) as u8;
        }
        let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
        for h in scale_f16.iter_mut() {
            *h = f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5));
        }

        let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scale_f16.len() * 2).unwrap();
        let zp_dev = runtime.alloc_raw(zp_packed.len()).unwrap();
        let output_dev = runtime.alloc_raw(max_m * n * 2).unwrap();
        // SAFETY: buffers sized to their sources.
        unsafe {
            runtime
                .htod(as_bytes(&activation_f16), activation_dev)
                .unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            runtime.htod(&zp_packed, zp_dev).unwrap();
        }

        let device = DeviceId::cuda(0);
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };

        let time_path = |kernel: &MatMulNBitsKernel, m: usize| -> f64 {
            let a_shape = [m, k];
            let a_strides = [k as i64, 1];
            let y_shape = [m, n];
            let y_strides = [n as i64, 1];
            let inputs = vec![
                TensorView::new(
                    device_ptr(activation_dev),
                    DataType::Float16,
                    &a_shape,
                    &a_strides,
                    device,
                ),
                TensorView::new(
                    device_ptr(packed_dev),
                    DataType::Uint8,
                    &b_shape,
                    &b_strides,
                    device,
                ),
                TensorView::new(
                    device_ptr(scales_dev),
                    DataType::Float16,
                    &scales_shape,
                    &scales_strides,
                    device,
                ),
                TensorView::new(
                    device_ptr(zp_dev),
                    DataType::Uint8,
                    &zp_shape,
                    &zp_strides,
                    device,
                ),
            ];
            let run = || {
                let mut outputs = [TensorMut::new(
                    device_ptr_mut(output_dev),
                    DataType::Float16,
                    &y_shape,
                    &y_strides,
                    device,
                )];
                kernel.run(&inputs, &mut outputs, None).unwrap();
            };
            // Warm up (NVRTC compile, repack cache fill).
            run();
            runtime.synchronize().unwrap();
            let iters = 30;
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let start = std::time::Instant::now();
                run();
                runtime.synchronize().unwrap();
                samples.push(start.elapsed().as_secs_f64() * 1e6);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            samples[samples.len() / 2]
        };

        eprintln!("Marlin M>1 wall vs tiled @ K={k} N={n} block={block_size} (median us):");
        eprintln!("  M     tiled_us   marlin_us   speedup");
        for &m in &[1usize, 2, 4, 8, 16, 32, 64, 128] {
            // SAFETY: serial ignored bench; flag toggling is race-free here.
            // Marlin M>1 is default-ON, so the tiled arm opts out explicitly.
            unsafe {
                std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "0");
            }
            let tiled = time_path(&kernel, m);
            unsafe {
                std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
            }
            let marlin = time_path(&kernel, m);
            let marlin_used = m > 1;
            eprintln!(
                "  {m:<5} {tiled:>8.1}   {marlin:>8.1}   {:>6.2}x{}",
                tiled / marlin,
                if marlin_used {
                    ""
                } else {
                    " (M=1 stays on GEMV)"
                }
            );
        }
        // SAFETY: clear the flag so it cannot leak into other tests.
        unsafe {
            std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
        }

        // SAFETY: every pointer came from this runtime's alloc_raw, freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }
    }

    /// Parity of the fused RMS-norm + Marlin M>1 path against the already-trusted
    /// fused RMS-norm + tiled GEMM. Both share the identical `launch_rmsnorm_prefill`
    /// staging, so the residual isolates the tensor-core GEMM's reordered
    /// accumulation. Drives the real op: flag off → tiled reference, flag on →
    /// Marlin; compares within the relayout tolerance.
    #[test]
    #[ignore = "requires an SM80+ CUDA device"]
    fn marlin_m_gt_1_rmsnorm_op_parity() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping Marlin M>1 rmsnorm parity: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("marlin_m_gt_1_rmsnorm")
            .is_err()
            || !marlin_gemm::device_supports_marlin(runtime.capabilities().compute_capability())
        {
            eprintln!("skipping Marlin M>1 rmsnorm parity: headers unavailable or pre-SM80");
            return;
        }

        let m = 8usize;
        let k = 2048usize;
        let n = 64usize;
        let block_size = 128usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);

        let mut state = 0x0bad_c0de_feed_face_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation = vec![f16::ZERO; m * k];
        for h in activation.iter_mut() {
            *h = f16::from_f32(next());
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for byte in packed.iter_mut() {
            *byte = ((next() * 0.5 + 0.5) * 255.0) as u8;
        }
        let mut zp_packed = vec![0u8; n * zp_row_bytes];
        for byte in zp_packed.iter_mut() {
            *byte = ((next() * 0.5 + 0.5) * 255.0) as u8;
        }
        let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
        for h in scale_f16.iter_mut() {
            *h = f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5));
        }
        let mut gamma = vec![f16::ZERO; k];
        for h in gamma.iter_mut() {
            *h = f16::from_f32(0.5 + 0.5 * (next() * 0.5 + 0.5));
        }

        let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scale_f16.len() * 2).unwrap();
        let zp_dev = runtime.alloc_raw(zp_packed.len()).unwrap();
        let gamma_dev = runtime.alloc_raw(gamma.len() * 2).unwrap();
        let output_dev = runtime.alloc_raw(m * n * 2).unwrap();
        // SAFETY: buffers sized to their sources.
        unsafe {
            runtime.htod(as_bytes(&activation), activation_dev).unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            runtime.htod(&zp_packed, zp_dev).unwrap();
            runtime.htod(as_bytes(&gamma), gamma_dev).unwrap();
        }

        let device = DeviceId::cuda(0);
        let a_shape = [m, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];
        let gamma_shape = [k];
        let gamma_strides = [1i64];
        let y_shape = [m, n];
        let y_strides = [n as i64, 1];

        // Slots: 3=zero_points, 4=g_idx (absent), 5=bias (absent), 6=gamma.
        let inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            ),
            TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            ),
            TensorView::absent(DataType::Int32),
            TensorView::absent(DataType::Float16),
            TensorView::new(
                device_ptr(gamma_dev),
                DataType::Float16,
                &gamma_shape,
                &gamma_strides,
                device,
            ),
        ];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: true,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };

        let run = |want_marlin: bool| -> Vec<f16> {
            // SAFETY: serial ignored GPU test; flag toggling is race-free here.
            unsafe {
                if want_marlin {
                    std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
                } else {
                    // Marlin M>1 is default-ON, so the tiled arm must opt out
                    // explicitly; unsetting the variable would select Marlin.
                    std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "0");
                }
            }
            let mut outputs = [TensorMut::new(
                device_ptr_mut(output_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            )];
            kernel.run(&inputs, &mut outputs, None).unwrap();
            runtime.synchronize().unwrap();
            let mut got = vec![f16::ZERO; m * n];
            // SAFETY: `output_dev` holds `m * n` fp16 values.
            unsafe {
                runtime.dtoh(as_bytes_mut(&mut got), output_dev).unwrap();
            }
            got
        };

        let tiled = run(false);
        let marlin = run(true);
        // SAFETY: reset the flag so it cannot leak into other tests.
        unsafe {
            std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
        }

        // SAFETY: every pointer came from this runtime's alloc_raw, freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(gamma_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut max_out = 0.0f32;
        for (mv, tv) in marlin.iter().zip(tiled.iter()) {
            let mf = mv.to_f32();
            let tf = tv.to_f32();
            assert!(mf.is_finite(), "Marlin rmsnorm output must be finite");
            worst_abs = worst_abs.max((mf - tf).abs());
            max_out = max_out.max(tf.abs());
        }
        let tol = 2e-2 * max_out.max(1e-3);
        eprintln!(
            "Marlin M>1 rmsnorm parity: worst_abs={worst_abs:.5}, max_out={max_out:.5}, tol={tol:.5}"
        );
        assert!(
            worst_abs <= tol,
            "fused rmsnorm Marlin output diverges from tiled: worst_abs={worst_abs} > tol={tol}"
        );
    }

    /// Gate/up SwiGLU MLP fusion parity + capture-safety for the Marlin M>1 path.
    /// Drives the real fused op (gate_up_swiglu = true) with the flag off (tiled
    /// two-GEMM + SiluMul reference) then on (paired Marlin GEMMs + the identical
    /// SiluMul), and asserts the outputs match within the relayout tolerance.
    /// Also checks the capture-safety contract on the plain (no-gamma) variant:
    /// cold call allocates pooled weights/scratch and is NOT capture-safe; a warm
    /// replay reuses them and IS capture-safe with byte-identical output.
    fn run_marlin_gate_up_parity(with_gamma: bool, decomposed: bool, check_capture: bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping Marlin gate/up parity: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("marlin_gate_up")
            .is_err()
            || !marlin_gemm::device_supports_marlin(runtime.capabilities().compute_capability())
        {
            eprintln!("skipping Marlin gate/up parity: headers unavailable or pre-SM80");
            return;
        }

        let m = 8usize;
        let k = 1024usize;
        let n = 256usize;
        let block_size = 32usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = (k_blocks * 4).div_ceil(8);

        let mut state = 0xa5a5_1234_dead_beef_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation = vec![f16::ZERO; m * k];
        for h in activation.iter_mut() {
            *h = f16::from_f32(next() * 0.5);
        }
        let mut make_weights = |scale: f32| {
            let mut packed = vec![0u8; n * k_blocks * blob_size];
            for byte in packed.iter_mut() {
                *byte = ((next() * 0.5 + 0.5) * 255.0) as u8;
            }
            let mut scales = vec![f16::ZERO; n * k_blocks];
            for h in scales.iter_mut() {
                *h = f16::from_f32(scale * (0.5 + 0.5 * (next() * 0.5 + 0.5)));
            }
            let mut zp = vec![0u8; n * zp_row_bytes];
            for byte in zp.iter_mut() {
                *byte = ((next() * 0.5 + 0.5) * 255.0) as u8;
            }
            (packed, scales, zp)
        };
        // Modest weight magnitudes so silu stays in a well-conditioned range.
        let (packed_gate, scales_gate, zp_gate) = make_weights(0.02);
        let (packed_up, scales_up, zp_up) = make_weights(0.02);
        let mut gamma = vec![f16::ZERO; k];
        for h in gamma.iter_mut() {
            *h = f16::from_f32(0.5 + 0.5 * (next() * 0.5 + 0.5));
        }

        let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
        let packed_gate_dev = runtime.alloc_raw(packed_gate.len()).unwrap();
        let scales_gate_dev = runtime.alloc_raw(scales_gate.len() * 2).unwrap();
        let packed_up_dev = runtime.alloc_raw(packed_up.len()).unwrap();
        let scales_up_dev = runtime.alloc_raw(scales_up.len() * 2).unwrap();
        let zp_gate_dev = runtime.alloc_raw(zp_gate.len()).unwrap();
        let zp_up_dev = runtime.alloc_raw(zp_up.len()).unwrap();
        let gamma_dev = runtime.alloc_raw(gamma.len() * 2).unwrap();
        let output_dev = runtime.alloc_raw(m * n * 2).unwrap();
        // SAFETY: buffers sized to their sources.
        unsafe {
            runtime.htod(as_bytes(&activation), activation_dev).unwrap();
            runtime.htod(&packed_gate, packed_gate_dev).unwrap();
            runtime
                .htod(as_bytes(&scales_gate), scales_gate_dev)
                .unwrap();
            runtime.htod(&packed_up, packed_up_dev).unwrap();
            runtime.htod(as_bytes(&scales_up), scales_up_dev).unwrap();
            runtime.htod(&zp_gate, zp_gate_dev).unwrap();
            runtime.htod(&zp_up, zp_up_dev).unwrap();
            runtime.htod(as_bytes(&gamma), gamma_dev).unwrap();
        }

        let device = DeviceId::cuda(0);
        let a_shape = [m, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];
        let gamma_shape = [k];
        let gamma_strides = [1i64];
        let y_shape = [m, n];
        let y_strides = [n as i64, 1];

        // [x, W_gate, scales_gate, W_up, scales_up, gamma@5, zp_gate@6, zp_up@7]
        let mut inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_gate_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_gate_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_up_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_up_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            ),
        ];
        if with_gamma {
            inputs.push(TensorView::new(
                device_ptr(gamma_dev),
                DataType::Float16,
                &gamma_shape,
                &gamma_strides,
                device,
            ));
        } else {
            inputs.push(TensorView::absent(DataType::Float16));
        }
        inputs.push(TensorView::new(
            device_ptr(zp_gate_dev),
            DataType::Uint8,
            &zp_shape,
            &zp_strides,
            device,
        ));
        inputs.push(TensorView::new(
            device_ptr(zp_up_dev),
            DataType::Uint8,
            &zp_shape,
            &zp_strides,
            device,
        ));

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: true,
            decomposed_silu: decomposed,
            rmsnorm_prologue: with_gamma,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };

        let run = |want_marlin: bool| -> Vec<f16> {
            // SAFETY: serial ignored GPU test; flag toggling is race-free here.
            unsafe {
                if want_marlin {
                    std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
                } else {
                    // Marlin M>1 is default-ON, so the tiled arm must opt out
                    // explicitly; unsetting the variable would select Marlin.
                    std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "0");
                }
            }
            let mut outputs = [TensorMut::new(
                device_ptr_mut(output_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            )];
            kernel.run(&inputs, &mut outputs, None).unwrap();
            runtime.synchronize().unwrap();
            let mut got = vec![f16::ZERO; m * n];
            // SAFETY: `output_dev` holds `m * n` fp16 values.
            unsafe {
                runtime.dtoh(as_bytes_mut(&mut got), output_dev).unwrap();
            }
            got
        };

        let tiled = run(false);
        let marlin_cold = run(true);
        let cold_safe = kernel.last_call_capture_safe.load(Ordering::Relaxed);
        let marlin_warm = run(true);
        let warm_safe = kernel.last_call_capture_safe.load(Ordering::Relaxed);
        // SAFETY: clear the flag so it cannot leak into other tests.
        unsafe {
            std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
        }

        // SAFETY: every pointer came from this runtime's alloc_raw, freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_gate_dev).unwrap();
            runtime.free_raw(scales_gate_dev).unwrap();
            runtime.free_raw(packed_up_dev).unwrap();
            runtime.free_raw(scales_up_dev).unwrap();
            runtime.free_raw(zp_gate_dev).unwrap();
            runtime.free_raw(zp_up_dev).unwrap();
            runtime.free_raw(gamma_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut max_out = 0.0f32;
        for (mv, tv) in marlin_warm.iter().zip(tiled.iter()) {
            let mf = mv.to_f32();
            let tf = tv.to_f32();
            assert!(mf.is_finite(), "Marlin gate/up output must be finite");
            worst_abs = worst_abs.max((mf - tf).abs());
            max_out = max_out.max(tf.abs());
        }
        let tol = 3e-2 * max_out.max(1e-3);
        eprintln!(
            "Marlin gate/up parity (gamma={with_gamma}, decomposed={decomposed}): \
             worst_abs={worst_abs:.5}, max_out={max_out:.5}, tol={tol:.5}, \
             cold_safe={cold_safe}, warm_safe={warm_safe}"
        );
        assert!(
            worst_abs <= tol,
            "Marlin gate/up output diverges from tiled: worst_abs={worst_abs} > tol={tol}"
        );
        assert_eq!(
            marlin_cold, marlin_warm,
            "warm gate/up Marlin replay must be byte-identical to the cold run"
        );
        if check_capture {
            // The cold-miss (NOT capture-safe) contract is asserted with
            // guaranteed-unique buffers in `marlin_m_gt_1_op_parity_and_capture_safety`.
            // Here the module-global repack/scratch pools may already be warm from a
            // prior test (freed device addresses get reused), so only the positive
            // guarantee — a warm replay IS capture-safe — is order-independent.
            assert!(
                warm_safe,
                "warm gate/up Marlin call reuses pooled buffers and must be capture-safe"
            );
        }
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device"]
    fn marlin_gate_up_swiglu_matches_tiled_plain() {
        run_marlin_gate_up_parity(false, false, true);
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device"]
    fn marlin_gate_up_swiglu_matches_tiled_rmsnorm() {
        run_marlin_gate_up_parity(true, false, true);
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device"]
    fn marlin_gate_up_decomposed_swiglu_matches_tiled_rmsnorm() {
        run_marlin_gate_up_parity(true, true, false);
    }

    fn device_ptr(raw: CUdeviceptr) -> DevicePtr {
        DevicePtr(raw as usize as *const c_void)
    }

    fn device_ptr_mut(raw: CUdeviceptr) -> DevicePtrMut {
        DevicePtrMut(raw as usize as *mut c_void)
    }

    /// Direct fp16 GEMV parity against an f32/f64 dequant-and-matmul oracle that
    /// is fed the **same fp16-rounded** activations and the same (fp16- or
    /// f32-) rounded scales, so the residual covers only the kernel's documented
    /// accumulation precision and fp16 output rounding — not input quantization,
    /// which both sides share.
    fn run_parity(scales_fp16: bool, with_bias: bool) -> (f32, f32, f32, bool) {
        // K spans 128 block-32 groups (contraction depth 4096, near the model's
        // widest hidden path), N covers several 8-column CTAs plus a ragged tail.
        run_parity_dims(4096, 70, scales_fp16, with_bias, false)
    }

    /// Parametrized fp16 GEMV parity harness. `k` and `n` pick the projection
    /// shape so callers can pin the exact production dims (e.g. Qwen2.5-1.5B's
    /// gate/up K=1536,N=8960 and down-projection K=8960,N=1536) that select
    /// different GEMV variants and block-count boundaries. `explicit_zp` toggles
    /// the asymmetric per-block int4 zero-point path (see
    /// [`run_parity_dims_block`]). Delegates to [`run_parity_dims_block`] with
    /// the default block-32 layout.
    fn run_parity_dims(
        k: usize,
        n: usize,
        scales_fp16: bool,
        with_bias: bool,
        explicit_zp: bool,
    ) -> (f32, f32, f32, bool) {
        run_parity_dims_block(k, n, 32, scales_fp16, with_bias, explicit_zp)
    }

    /// Parametrized fp16 GEMV parity harness with an explicit `block_size`. The
    /// tuned block-32 kernels and the model-agnostic general-block-size kernel
    /// share this oracle; passing `block_size != 32` exercises the general
    /// decode GEMV (`matmul_nbits_gemv_f16_general_bs`) against the same f64
    /// dequant-and-matmul reference. `block_size` must be a power of two >= 16
    /// and must divide `k`. `explicit_zp` supplies a non-uniform per-block int4
    /// zero-point tensor (packed two block-nibbles per byte) instead of the
    /// symmetric zp=8 default, so a zero-point indexing regression in the
    /// general kernel's dequant path is caught.
    fn run_parity_dims_block(
        k: usize,
        n: usize,
        block_size: usize,
        scales_fp16: bool,
        with_bias: bool,
        explicit_zp: bool,
    ) -> (f32, f32, f32, bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping MatMulNBits fp16 GEMV parity test: CUDA runtime unavailable");
            return (0.0, 0.0, 0.0, true);
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping MatMulNBits fp16 GEMV parity test: fp16 NVRTC headers unavailable");
            return (0.0, 0.0, 0.0, true);
        }

        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);

        // Deterministic LCG so the test is reproducible without extra crates.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // fp16 activations (device input) plus their fp16-value-as-f32 twin so
        // the oracle consumes identical inputs.
        let mut activation_f16 = vec![f16::ZERO; k];
        let mut activation_ref = vec![0.0f32; k];
        for (dst_h, dst_f) in activation_f16.iter_mut().zip(activation_ref.iter_mut()) {
            let h = f16::from_f32(next());
            *dst_h = h;
            *dst_f = h.to_f32();
        }

        // int4 quant codes (0..15), packed two nibbles per byte in the exact
        // symmetric block-32 layout the kernel unpacks.
        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob_size {
                    let low = quant[col * k + block * block_size + pair * 2] & 15;
                    let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                    packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                }
            }
        }

        // Explicit asymmetric int4 zero points: a non-uniform per-(col, block)
        // code in 0..15, packed two block-nibbles per byte exactly as the kernel
        // unpacks (`zp[col*zp_row_bytes + block/2]`, low nibble for even blocks).
        // The symmetric default (zp=8) is used when `explicit_zp` is false.
        let mut zp_codes = vec![8i32; n * k_blocks];
        let mut zp_packed = vec![0u8; n * zp_row_bytes];
        if explicit_zp {
            for code in zp_codes.iter_mut().take(n * k_blocks) {
                *code = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as i32;
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

        // Per (col, block) scales, rounded to the storage dtype so both paths use
        // the same scale value.
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

        let mut bias_f16 = vec![f16::ZERO; n];
        let mut bias_ref = vec![0.0f32; n];
        if with_bias {
            for (h, f) in bias_f16.iter_mut().zip(bias_ref.iter_mut()) {
                let value = f16::from_f32(next());
                *h = value;
                *f = value.to_f32();
            }
        }

        // f64 dequant-and-matmul oracle over the shared fp16 activations.
        let mut expected = vec![0.0f32; n];
        for col in 0..n {
            let mut acc = 0.0f64;
            for block in 0..k_blocks {
                let scale = scale_ref[col * k_blocks + block] as f64;
                let zero_point = zp_codes[col * k_blocks + block];
                for within in 0..block_size {
                    let depth = block * block_size + within;
                    let q = quant[col * k + depth] as i32 - zero_point;
                    acc += activation_ref[depth] as f64 * q as f64 * scale;
                }
            }
            if with_bias {
                acc += bias_ref[col] as f64;
            }
            expected[col] = acc as f32;
        }

        let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime
            .alloc_raw(n * k_blocks * if scales_fp16 { 2 } else { 4 })
            .unwrap();
        let zp_dev = runtime.alloc_raw(zp_packed.len().max(1)).unwrap();
        let bias_dev = runtime.alloc_raw(n * 2).unwrap();
        let output_dev = runtime.alloc_raw(n * 2).unwrap();

        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime
                .htod(as_bytes(&activation_f16), activation_dev)
                .unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            if scales_fp16 {
                runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            } else {
                runtime.htod(as_bytes(&scale_f32), scales_dev).unwrap();
            }
            if explicit_zp {
                runtime.htod(&zp_packed, zp_dev).unwrap();
            }
            if with_bias {
                runtime.htod(as_bytes(&bias_f16), bias_dev).unwrap();
            }
        }

        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let bias_shape = [n];
        let bias_strides = [1i64];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];

        let scales_dtype = if scales_fp16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        let device = DeviceId::cuda(0);
        let mut inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                scales_dtype,
                &scales_shape,
                &scales_strides,
                device,
            ),
        ];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];
        let zp_view = TensorView::new(
            device_ptr(zp_dev),
            DataType::Uint8,
            &zp_shape,
            &zp_strides,
            device,
        );
        // Slots: 3 = zero_points, 4 = g_idx, 5 = bias. Fill only up to the last
        // present optional input so the kernel's `optional_input` indexing holds.
        if explicit_zp {
            inputs.push(zp_view);
        } else if with_bias {
            inputs.push(TensorView::absent(DataType::Uint8));
        }
        if with_bias {
            inputs.push(TensorView::absent(DataType::Int32));
            inputs.push(TensorView::new(
                device_ptr(bias_dev),
                DataType::Float16,
                &bias_shape,
                &bias_strides,
                device,
            ));
        }

        let mut outputs = [TensorMut::new(
            device_ptr_mut(output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        )];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };
        // The default decode GEMV launch is a static grid that allocates
        // nothing, so it is capture-safe on the very first call. That only holds
        // with the dispatch levers OFF: `ONNX_GENAI_INTERLEAVE_DEQUANT` routes
        // through `ensure_interleaved`, which allocates and interleaves the
        // packed weights on first sight and correctly reports that call
        // capture-unsafe. Hold the shared guard so the env-mutating interleave
        // test cannot flip the lever underneath this kernel run.
        let _levers = default_levers_guard();
        kernel.run(&inputs, &mut outputs, None).unwrap();
        runtime.synchronize().unwrap();

        assert!(
            kernel.last_call_capture_safe.load(Ordering::Relaxed),
            "fp16 decode GEMV must report capture-safe"
        );
        drop(_levers);

        let mut got_f16 = vec![f16::ZERO; n];
        // SAFETY: `output_dev` holds `n` fp16 values.
        unsafe {
            runtime
                .dtoh(as_bytes_mut(&mut got_f16), output_dev)
                .unwrap();
        }

        // SAFETY: each pointer came from this runtime's `alloc_raw` and is freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(bias_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut max_out = 0.0f32;
        let mut all_finite = true;
        for (g16, e) in got_f16.iter().zip(expected.iter()) {
            let g = g16.to_f32();
            if !g.is_finite() {
                all_finite = false;
            }
            let abs = (g - e).abs();
            let rel = abs / e.abs().max(1e-1);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
            max_out = max_out.max(e.abs());
        }
        (worst_abs, worst_rel, max_out, all_finite)
    }

    /// Raw fp16 output of a symmetric (no zero-point, no bias) int4 decode GEMV
    /// at an explicit `block_size`, used by the interleaved-dequant byte-identity
    /// test. `interleave` toggles `ONNX_GENAI_INTERLEAVE_DEQUANT`; the general
    /// split-K heuristic is pinned OFF so the non-interleaved reference always
    /// takes the wide-load multicol kernel that the interleaved path replaces.
    /// Returns `None` when no CUDA device / fp16 NVRTC headers are available.
    fn run_symmetric_block_raw(
        k: usize,
        n: usize,
        block_size: usize,
        scales_fp16: bool,
        interleave: bool,
        general_splitk: Option<bool>,
        pipeline: Option<bool>,
    ) -> Option<Vec<f16>> {
        let runtime = runtime()?;
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            return None;
        }

        let k_blocks = k / block_size;
        let blob_size = block_size / 2;

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation_f16 = vec![f16::ZERO; k];
        for dst_h in activation_f16.iter_mut() {
            *dst_h = f16::from_f32(next());
        }
        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob_size {
                    let low = quant[col * k + block * block_size + pair * 2] & 15;
                    let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                    packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                }
            }
        }
        let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
        let mut scale_f32 = vec![0.0f32; n * k_blocks];
        for i in 0..n * k_blocks {
            let raw = 0.015 + 0.01 * (next() * 0.5 + 0.5);
            if scales_fp16 {
                scale_f16[i] = f16::from_f32(raw);
            } else {
                scale_f32[i] = raw;
            }
        }

        let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime
            .alloc_raw(n * k_blocks * if scales_fp16 { 2 } else { 4 })
            .unwrap();
        let output_dev = runtime.alloc_raw(n * 2).unwrap();
        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime
                .htod(as_bytes(&activation_f16), activation_dev)
                .unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            if scales_fp16 {
                runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            } else {
                runtime.htod(as_bytes(&scale_f32), scales_dev).unwrap();
            }
        }

        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];
        let scales_dtype = if scales_fp16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        let device = DeviceId::cuda(0);
        let inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                scales_dtype,
                &scales_shape,
                &scales_strides,
                device,
            ),
        ];
        let mut outputs = [TensorMut::new(
            device_ptr_mut(output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        )];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };

        // Take the EXCLUSIVE side: the env writes below are visible to every
        // thread in the binary, so they must not overlap any test that expects
        // the levers at their default. SAFETY: the writes are confined to this
        // critical section and restored before the lock is released.
        let _guard = LeverEnvGuard::acquire();
        unsafe {
            match general_splitk {
                Some(true) => std::env::set_var("ONNX_GENAI_GENERAL_SPLITK", "on"),
                Some(false) => std::env::set_var("ONNX_GENAI_GENERAL_SPLITK", "off"),
                None => std::env::remove_var("ONNX_GENAI_GENERAL_SPLITK"),
            }
            if interleave {
                std::env::set_var("ONNX_GENAI_INTERLEAVE_DEQUANT", "1");
            } else {
                std::env::remove_var("ONNX_GENAI_INTERLEAVE_DEQUANT");
            }
            match pipeline {
                Some(true) => std::env::set_var("ONNX_GENAI_GEMV_PIPELINE", "1"),
                Some(false) => std::env::set_var("ONNX_GENAI_GEMV_PIPELINE", "0"),
                None => std::env::remove_var("ONNX_GENAI_GEMV_PIPELINE"),
            }
        }
        kernel.run(&inputs, &mut outputs, None).unwrap();
        runtime.synchronize().unwrap();
        // `_guard` clears the levers as it drops, on the unwind path too.
        drop(_guard);

        let mut got_f16 = vec![f16::ZERO; n];
        // SAFETY: `output_dev` holds `n` fp16 values.
        unsafe {
            runtime
                .dtoh(as_bytes_mut(&mut got_f16), output_dev)
                .unwrap();
            runtime.free_raw(activation_dev).unwrap();
            // NOTE: `packed_dev` is intentionally NOT freed. The interleave cache
            // is keyed by (source pointer, byte length, ordinal); freeing the
            // packed buffer would let CUDA reuse its address for a later,
            // different-content buffer of the same byte length, producing a stale
            // cache hit in a subsequent interleave run. Production weights are
            // immutable initializers that are never freed, so this aliasing
            // cannot occur there; leaking the small test buffer reproduces that
            // stable-address invariant. (Real allocations are reclaimed on
            // process exit.)
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }
        Some(got_f16)
    }

    /// The opt-in TRT-LLM interleaved + biased int4 dequant
    /// (`ONNX_GENAI_INTERLEAVE_DEQUANT`) must be BYTE-IDENTICAL to the fp32
    /// wide-load multicol decode GEMV it replaces, on symmetric weights. The
    /// converter yields `(code - 8)` in fp16 exactly as the reference path, the
    /// eight products accumulate in the same ascending element order, and the
    /// offline nibble-interleave only changes *how* the codes are decoded, so
    /// the fp16 outputs must match bit-for-bit. Covers block-128 (glm) and
    /// block-64, fp16 and fp32 scales, and a ragged N tail.
    #[test]
    fn int4_interleaved_dequant_is_bit_identical_to_multicol() {
        let mut ran = false;
        for (k, n, block_size, scales_fp16) in [
            (4096usize, 256usize, 128usize, true),
            (4096, 256, 128, false),
            (4096, 70, 128, true),
            (8192, 512, 128, true),
            (4096, 256, 64, true),
        ] {
            let Some(base) =
                run_symmetric_block_raw(k, n, block_size, scales_fp16, false, Some(false), None)
            else {
                eprintln!(
                    "skipping interleaved dequant byte-identity test: CUDA runtime/headers \
                     unavailable"
                );
                return;
            };
            let inter =
                run_symmetric_block_raw(k, n, block_size, scales_fp16, true, Some(false), None)
                    .unwrap();
            ran = true;
            let mismatches = base
                .iter()
                .zip(inter.iter())
                .filter(|(b, i)| b.to_bits() != i.to_bits())
                .count();
            assert_eq!(
                mismatches, 0,
                "interleaved int4 dequant diverged from multicol at block-{block_size} K={k} \
                 N={n} scales_fp16={scales_fp16}: {mismatches}/{n} fp16 outputs differ"
            );
        }
        assert!(
            ran,
            "interleaved dequant byte-identity test did not execute any case"
        );
    }

    /// The opt-in interleaved+biased lever must also be BYTE-IDENTICAL on the
    /// grid-starved split-K WIDE path (`ONNX_GENAI_GENERAL_SPLITK=on`), which
    /// serves glm's narrow-N qkv/down projections. The offline nibble-interleave
    /// leaves each lane's fp32 partial bit-identical to the non-interleaved
    /// split-K wide kernel, and the K_SPLIT shared-memory reduction order is
    /// unchanged, so the fp16 outputs must match bit-for-bit.
    #[test]
    fn int4_interleaved_dequant_is_bit_identical_to_splitk_wide() {
        let mut ran = false;
        for (k, n, block_size, scales_fp16) in [
            (4096usize, 256usize, 128usize, true),
            (4096, 256, 128, false),
            (4096, 70, 128, true),
            (8192, 128, 128, true),
        ] {
            let Some(base) =
                run_symmetric_block_raw(k, n, block_size, scales_fp16, false, Some(true), None)
            else {
                eprintln!(
                    "skipping split-K wide interleaved byte-identity test: CUDA runtime/headers \
                     unavailable"
                );
                return;
            };
            let inter =
                run_symmetric_block_raw(k, n, block_size, scales_fp16, true, Some(true), None)
                    .unwrap();
            ran = true;
            let mismatches = base
                .iter()
                .zip(inter.iter())
                .filter(|(b, i)| b.to_bits() != i.to_bits())
                .count();
            assert_eq!(
                mismatches, 0,
                "interleaved int4 dequant diverged from split-K wide at block-{block_size} K={k} \
                 N={n} scales_fp16={scales_fp16}: {mismatches}/{n} fp16 outputs differ"
            );
        }
        assert!(
            ran,
            "split-K wide interleaved byte-identity test did not execute any case"
        );
    }

    /// The prefetch-pipelined single-warp block-32 scales-fp16 int4 decode GEMV
    /// (`matmul_nbits_gemv_f16_scales_f16_pipe`, default-on, `use_scales_f16_pipeline`)
    /// must be BYTE-IDENTICAL to the original single-load entry it replaces
    /// (`ONNX_GENAI_GEMV_PIPELINE=0`). The pipe variant only keeps extra weight
    /// loads in flight per lane (memory-level parallelism to hide the Long
    /// Scoreboard stall); it preserves the EXACT lane→nibble mapping, the same
    /// `accumulate_int4x8_f16_zp` calls with the same fp16 arguments in the same
    /// order, and the same fp16 accumulation — so every fp16 output must match
    /// bit-for-bit. A single differing bit means the reschedule reassociated the
    /// K-reduction and the byte-identity contract is broken. Covers the real
    /// block-32 decode projection widths (Qwen2.5-14B q/o K=N=5120, the wide MLP
    /// K=13824, a small attention shape, and a ragged-N tail) across fp16 and
    /// fp32 scales.
    #[test]
    fn scales_f16_pipeline_is_bit_identical_to_scalar() {
        let mut ran = false;
        for (k, n) in [
            (5120usize, 5120usize),
            (5120, 13824),
            (896, 896),
            (896, 4870),
        ] {
            for scales_fp16 in [true, false] {
                // The block-32 General single-warp path is the only one rerouted
                // to the pipe entry; confirm the dims select it before comparing.
                assert_eq!(
                    select_f16_gemv_variant(k, n, 32, scales_fp16, false).variant,
                    F16GemvVariant::General,
                    "K={k} N={n} scales_fp16={scales_fp16} must select the General variant"
                );
                let Some(scalar) =
                    run_symmetric_block_raw(k, n, 32, scales_fp16, false, None, Some(false))
                else {
                    eprintln!(
                        "skipping scales-fp16 pipeline byte-identity test: CUDA runtime/headers \
                         unavailable"
                    );
                    return;
                };
                let pipe = run_symmetric_block_raw(k, n, 32, scales_fp16, false, None, Some(true))
                    .unwrap();
                ran = true;
                let mismatches = scalar
                    .iter()
                    .zip(pipe.iter())
                    .filter(|(s, p)| s.to_bits() != p.to_bits())
                    .count();
                assert_eq!(
                    mismatches, 0,
                    "prefetch-pipelined scales-fp16 GEMV diverged from the scalar entry at \
                     block-32 K={k} N={n} scales_fp16={scales_fp16}: {mismatches}/{n} fp16 \
                     outputs differ"
                );
            }
        }
        assert!(
            ran,
            "scales-fp16 pipeline byte-identity test did not execute any case"
        );
    }

    /// GEMM. Drives the real `MatMulNBitsKernel::run` dispatch with
    /// `ONNX_GENAI_MARLIN_M_GT_1=1` so the `m > 1` seam routes to Marlin,
    /// exercising eligibility gating, the device repack + module-level cache, and
    /// the launch. Validates the output against an f64 dequant→GEMM oracle
    /// (asymmetric zero points) within the relayout tolerance, and asserts the
    /// capture-safety contract: the first (cold) call repacks and reports
    /// NOT-capture-safe, while an identical second call hits the warm cache and
    /// reports capture-safe with byte-identical output.
    #[test]
    #[ignore = "requires an SM80+ CUDA device"]
    fn marlin_m_gt_1_op_parity_and_capture_safety() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping Marlin M>1 op parity: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("marlin_m_gt_1_op_parity")
            .is_err()
        {
            eprintln!("skipping Marlin M>1 op parity: fp16 NVRTC headers unavailable");
            return;
        }
        if !marlin_gemm::device_supports_marlin(runtime.capabilities().compute_capability()) {
            eprintln!("skipping Marlin M>1 op parity: device is pre-SM80");
            return;
        }

        let m = 8usize;
        let k = 4096usize;
        let n = 70usize;
        let block_size = 128usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation_f16 = vec![f16::ZERO; m * k];
        let mut activation_ref = vec![0.0f32; m * k];
        for (h, f) in activation_f16.iter_mut().zip(activation_ref.iter_mut()) {
            let value = f16::from_f32(next());
            *h = value;
            *f = value.to_f32();
        }

        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob_size {
                    let low = quant[col * k + block * block_size + pair * 2] & 15;
                    let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                    packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                }
            }
        }

        // Asymmetric per-(col, block) zero points, nibble-packed as the kernel
        // unpacks them (low nibble for even blocks).
        let mut zp_codes = vec![0i32; n * k_blocks];
        let mut zp_packed = vec![0u8; n * zp_row_bytes];
        for code in zp_codes.iter_mut() {
            *code = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as i32;
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

        let mut scale_ref = vec![0.0f32; n * k_blocks];
        let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
        for i in 0..n * k_blocks {
            let h = f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5));
            scale_f16[i] = h;
            scale_ref[i] = h.to_f32();
        }

        // f64 dequant→GEMM oracle: Y[row, col] = sum_k a[row,k] * (code-zp) * scale.
        let mut expected = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for block in 0..k_blocks {
                    let scale = scale_ref[col * k_blocks + block] as f64;
                    let zp = zp_codes[col * k_blocks + block];
                    for within in 0..block_size {
                        let depth = block * block_size + within;
                        let q = quant[col * k + depth] as i32 - zp;
                        acc += activation_ref[row * k + depth] as f64 * q as f64 * scale;
                    }
                }
                expected[row * n + col] = acc as f32;
            }
        }

        let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scale_f16.len() * 2).unwrap();
        let zp_dev = runtime.alloc_raw(zp_packed.len()).unwrap();
        let output_dev = runtime.alloc_raw(m * n * 2).unwrap();
        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime
                .htod(as_bytes(&activation_f16), activation_dev)
                .unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            runtime.htod(&zp_packed, zp_dev).unwrap();
        }

        let device = DeviceId::cuda(0);
        let a_shape = [m, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];
        let y_shape = [m, n];
        let y_strides = [n as i64, 1];

        let inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            ),
            TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            ),
        ];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };

        // Enable the opt-in Marlin path for the duration of this test.
        // SAFETY: this ignored GPU test runs serially; no other thread reads the
        // flag concurrently.
        unsafe {
            std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
        }

        let run_once = |capture_expectation: &str| -> Vec<f16> {
            let mut outputs = [TensorMut::new(
                device_ptr_mut(output_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            )];
            kernel.run(&inputs, &mut outputs, None).unwrap();
            runtime.synchronize().unwrap();
            let mut got = vec![f16::ZERO; m * n];
            // SAFETY: `output_dev` holds `m * n` fp16 values.
            unsafe {
                runtime.dtoh(as_bytes_mut(&mut got), output_dev).unwrap();
            }
            eprintln!(
                "Marlin M>1 op {capture_expectation}: capture_safe={}",
                kernel.last_call_capture_safe.load(Ordering::Relaxed)
            );
            got
        };

        // First call: cold repack ⇒ allocates ⇒ reported NOT capture-safe.
        let got_cold = run_once("cold");
        assert!(
            !kernel.last_call_capture_safe.load(Ordering::Relaxed),
            "cold Marlin call performs the repack allocation and must report NOT capture-safe"
        );

        // Second identical call: warm cache ⇒ no alloc ⇒ capture-safe, and the
        // static-grid kernel is deterministic so the bytes match exactly.
        let got_warm = run_once("warm");
        assert!(
            kernel.last_call_capture_safe.load(Ordering::Relaxed),
            "warm Marlin call reuses the cached repack and must report capture-safe"
        );
        assert_eq!(
            got_cold, got_warm,
            "warm replay of the static-grid Marlin kernel must be byte-identical to the cold run"
        );

        // SAFETY: reset the process-global flag so it cannot leak into other tests.
        unsafe {
            std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
        }

        // SAFETY: each pointer came from this runtime's `alloc_raw`, freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut max_out = 0.0f32;
        for (g16, e) in got_warm.iter().zip(expected.iter()) {
            let g = g16.to_f32();
            assert!(g.is_finite(), "Marlin M>1 output must be finite");
            worst_abs = worst_abs.max((g - e).abs());
            max_out = max_out.max(e.abs());
        }
        let tol = 2e-2 * max_out.max(1e-3);
        eprintln!(
            "Marlin M>1 op parity: worst_abs={worst_abs:.5}, max_out={max_out:.5}, tol={tol:.5}"
        );
        assert!(
            worst_abs <= tol,
            "Marlin M>1 op output exceeds tolerance: worst_abs={worst_abs} > tol={tol}"
        );
    }

    /// The dequantize-to-fp16 + cuBLASLt tensor-core path taken at M>1, checked
    /// against the same f64 dequant→GEMM oracle the Marlin parity test uses and,
    /// in the same run, against Marlin itself.
    ///
    /// Two things are easy to get wrong here and neither shows up as a crash.
    /// The dequantize kernel writes `[N, K]` (matching the packed weight's own
    /// storage order, which is what makes its loads and stores coalesce), so the
    /// GEMM has to transpose it back — an error in that mapping silently
    /// produces a plausible-looking but wrong matrix. And the cuBLASLt call
    /// needs a declared workspace: when none is supplied the launch fails
    /// *after* the dequantize has already run and the caller quietly falls back
    /// to Marlin, so a test that only checked the numbers would pass while
    /// measuring the old path. This asserts the path actually ran.
    #[test]
    #[ignore = "requires an SM80+ CUDA device"]
    fn dequant_f16_cublas_m_gt_1_op_parity() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping dequant-f16 cuBLASLt parity: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("dequant_f16_cublas_m_gt_1_op_parity")
            .is_err()
        {
            eprintln!("skipping dequant-f16 cuBLASLt parity: fp16 NVRTC headers unavailable");
            return;
        }

        let m = 32usize;
        let k = 1024usize;
        let n = 96usize;
        let block_size = 32usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);

        let mut state = 0x0f1e_2d3c_4b5a_6978u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation_f16 = vec![f16::ZERO; m * k];
        let mut activation_ref = vec![0.0f32; m * k];
        for (h, f) in activation_f16.iter_mut().zip(activation_ref.iter_mut()) {
            let value = f16::from_f32(next());
            *h = value;
            *f = value.to_f32();
        }

        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob_size {
                    let low = quant[col * k + block * block_size + pair * 2] & 15;
                    let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                    packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                }
            }
        }

        let mut zp_codes = vec![0i32; n * k_blocks];
        let mut zp_packed = vec![0u8; n * zp_row_bytes];
        for code in zp_codes.iter_mut() {
            *code = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as i32;
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

        let mut scale_ref = vec![0.0f32; n * k_blocks];
        let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
        for i in 0..n * k_blocks {
            let h = f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5));
            scale_f16[i] = h;
            scale_ref[i] = h.to_f32();
        }

        let mut expected = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for block in 0..k_blocks {
                    let scale = scale_ref[col * k_blocks + block] as f64;
                    let zp = zp_codes[col * k_blocks + block];
                    for within in 0..block_size {
                        let depth = block * block_size + within;
                        let q = quant[col * k + depth] as i32 - zp;
                        acc += activation_ref[row * k + depth] as f64 * q as f64 * scale;
                    }
                }
                expected[row * n + col] = acc as f32;
            }
        }

        let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scale_f16.len() * 2).unwrap();
        let zp_dev = runtime.alloc_raw(zp_packed.len()).unwrap();
        let output_dev = runtime.alloc_raw(m * n * 2).unwrap();
        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime
                .htod(as_bytes(&activation_f16), activation_dev)
                .unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            runtime.htod(&zp_packed, zp_dev).unwrap();
        }

        let device = DeviceId::cuda(0);
        let a_shape = [m, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];
        let y_shape = [m, n];
        let y_strides = [n as i64, 1];

        let inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            ),
            TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            ),
        ];

        let kernel = zc_kernel(&runtime, k, n, block_size);

        // Size the workspace exactly the way the executor does, through the
        // kernel's own declared requirement, so this also covers the plumbing
        // that denied the fused node a workspace in the first place.
        let metadata = [TensorMetadata {
            dtype: DataType::Float16,
            shape: &a_shape,
            present: true,
        }];
        let requirement = kernel.workspace_requirement_for(&metadata).unwrap();
        let workspace_bytes = requirement.bytes.max(1) as usize;
        let workspace_dev = runtime.alloc_raw(workspace_bytes).unwrap();

        let run_once = |dequant_f16: bool| -> Vec<f16> {
            // SAFETY: this ignored GPU test runs serially; no other thread reads
            // these flags concurrently.
            unsafe {
                std::env::set_var("ONNX_GENAI_DEQUANT_F16_GEMM", if dequant_f16 { "1" } else { "0" });
                std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", if dequant_f16 { "0" } else { "1" });
            }
            let mut outputs = [TensorMut::new(
                device_ptr_mut(output_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            )];
            let workspace = Some(WorkspaceView::new(
                device_ptr_mut(workspace_dev),
                workspace_bytes,
            ));
            kernel.run(&inputs, &mut outputs, workspace).unwrap();
            runtime.synchronize().unwrap();
            let mut got = vec![f16::ZERO; m * n];
            // SAFETY: `output_dev` holds `m * n` fp16 values.
            unsafe {
                runtime.dtoh(as_bytes_mut(&mut got), output_dev).unwrap();
            }
            got
        };

        assert!(
            requirement.bytes > 0,
            "the dequant-f16 path must declare a cuBLASLt workspace; a zero requirement means \
             the launch would fail and silently fall back to Marlin"
        );

        let got_dequant = run_once(true);
        let got_marlin = run_once(false);

        // SAFETY: reset the process-global flags so they cannot leak.
        unsafe {
            std::env::remove_var("ONNX_GENAI_DEQUANT_F16_GEMM");
            std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
        }
        // SAFETY: each pointer came from this runtime's `alloc_raw`, freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
            runtime.free_raw(workspace_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut worst_vs_marlin = 0.0f32;
        let mut max_out = 0.0f32;
        for ((d16, m16), e) in got_dequant
            .iter()
            .zip(got_marlin.iter())
            .zip(expected.iter())
        {
            let d = d16.to_f32();
            assert!(d.is_finite(), "dequant-f16 output must be finite");
            worst_abs = worst_abs.max((d - e).abs());
            worst_vs_marlin = worst_vs_marlin.max((d - m16.to_f32()).abs());
            max_out = max_out.max(e.abs());
        }
        let tol = 2e-2 * max_out.max(1e-3);
        eprintln!(
            "dequant-f16 cuBLASLt parity: worst_abs={worst_abs:.5} vs_marlin={worst_vs_marlin:.5} \
             max_out={max_out:.5} tol={tol:.5} workspace={workspace_bytes}B"
        );
        assert!(
            worst_abs <= tol,
            "dequant-f16 cuBLASLt output exceeds oracle tolerance: {worst_abs} > {tol}"
        );
        assert!(
            worst_vs_marlin <= tol,
            "dequant-f16 cuBLASLt output diverged from Marlin: {worst_vs_marlin} > {tol}"
        );
    }

    fn zc_env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    }

    /// Deterministic int4/fp16 decode GEMV inputs for the #864 zero-copy probe.
    /// Mirrors the block-32 asymmetric (`zp`) layout the real `qwen14b-zp`
    /// decode path feeds `matmul_nbits_gemv_f16_scales_f16_zp` / its split-K
    /// sibling: `packed` is `n * k_blocks * blob_size` bytes (two int4 nibbles
    /// per byte), `scales` is fp16, `zero_points` is a per-(col, block) nibble.
    #[allow(clippy::type_complexity)]
    fn zc_make_inputs(
        k: usize,
        n: usize,
        block_size: usize,
    ) -> (Vec<f16>, Vec<u8>, Vec<f16>, Vec<u8>) {
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);

        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation = vec![f16::ZERO; k];
        for slot in activation.iter_mut() {
            *slot = f16::from_f32(next());
        }

        // Packed int4 codes, two nibbles per byte, laid out exactly as the
        // kernel unpacks: `packed[(col * k_blocks + block) * blob_size + pair]`.
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for byte in packed.iter_mut() {
            let low = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
            let high = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
            *byte = low | (high << 4);
        }

        let mut scales = vec![f16::ZERO; n * k_blocks];
        for slot in scales.iter_mut() {
            *slot = f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5));
        }

        // Asymmetric per-(col, block) int4 zero points, packed low/high nibble.
        let mut zp_packed = vec![0u8; n * zp_row_bytes];
        for col in 0..n {
            for block in 0..k_blocks {
                let code = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
                let byte = &mut zp_packed[col * zp_row_bytes + block / 2];
                if block & 1 == 0 {
                    *byte = (*byte & 0xf0) | code;
                } else {
                    *byte = (*byte & 0x0f) | (code << 4);
                }
            }
        }

        (activation, packed, scales, zp_packed)
    }

    fn zc_kernel(
        runtime: &Arc<CudaRuntime>,
        k: usize,
        n: usize,
        block_size: usize,
    ) -> MatMulNBitsKernel {
        MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        }
    }

    /// #864 — the decisive measurement this issue asks for. The 11.4 GB/s
    /// zero-copy figure on #864/#877 was an *upper bound* measured with a
    /// perfectly sequential `cuMemcpyDtoD` pull from mapped host memory. This
    /// runs the **real** int4/fp16 decode GEMV (`MatMulNBits`, the exact kernel
    /// plus launch config the engine selects) twice — once with its `packed`
    /// weight pointer from `cuMemAlloc` (VRAM) and once from
    /// `cuMemHostGetDevicePointer` over a `cuMemHostRegister(DEVICEMAP|
    /// READ_ONLY)` host buffer — with **only the weight pointer differing**.
    /// Everything else (activation, scales, zero-points, output) stays resident.
    ///
    /// It reports effective GB/s (packed_bytes / kernel_time, best-of-N) for
    /// both arms so the numbers are directly comparable to #877's table, and
    /// asserts the two arms' outputs are **bit-identical** (a zero-copy read
    /// that changes the result is a correctness failure, not a perf question).
    ///
    /// Sizes are swept because the ~11.9 MB average per-tensor page-in fits in
    /// this GPU's L2: at that size repeated VRAM reads are L2-cached and
    /// *overstate* device bandwidth, inflating the ratio in the hybrid's favour.
    /// The larger sizes exceed L2 and report the true cold DRAM-vs-PCIe rate the
    /// real 8.33 GB layer walk sees (each weight read once per step, evicted
    /// long before reuse). The honest ratio is the large-size one.
    ///
    /// `#[ignore]`: needs a live CUDA device and page-locks host RAM; it is a
    /// measurement, not a CI gate. Run solo with the GPU otherwise idle:
    ///   cargo test -p onnx-runtime-ep-cuda --features cuda --release \
    ///     zerocopy_kernel_host_mapped_vs_vram -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore]
    fn zerocopy_kernel_host_mapped_vs_vram_bandwidth() {
        use cudarc::driver::result::event;
        use cudarc::driver::sys;

        let Some(runtime) = runtime() else {
            eprintln!("skipping zero-copy kernel probe: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("zero-copy kernel probe")
            .is_err()
        {
            eprintln!("skipping zero-copy kernel probe: fp16 NVRTC headers unavailable");
            return;
        }
        runtime.bind().unwrap();

        const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;
        const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 0x08;

        let k = zc_env_usize("ZC_K", 5120);
        let block_size = zc_env_usize("ZC_BLOCK", 32);
        let reps = zc_env_usize("ZC_REPS", 7).max(3);
        assert!(
            k.is_multiple_of(block_size),
            "K must be a multiple of block_size"
        );
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);
        let row_bytes = k_blocks * blob_size;

        let sizes_mib: Vec<usize> = std::env::var("ZC_SIZES_MIB")
            .ok()
            .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![12, 64, 128, 256, 384]);

        let caps = runtime.capabilities();
        println!(
            "\n=== #864 zero-copy REAL-kernel bandwidth probe ===\n\
             SMs={} K={k} block_size={block_size} k_blocks={k_blocks} row_bytes/col={row_bytes} best-of-{reps}\n\
             (packed = int4 weights; only the packed pointer differs VRAM vs host-mapped)\n",
            caps.multiprocessor_count(),
        );
        println!(
            "{:>10} {:>9} {:>12} {:>12} {:>8} {:>10}",
            "packedMiB", "N", "VRAM GB/s", "host GB/s", "ratio", "match"
        );

        for mib in sizes_mib {
            let target = mib * (1usize << 20);
            let n = (target / row_bytes).max(1);
            let packed_bytes = n * row_bytes;
            let packed_mib = packed_bytes as f64 / (1u64 << 20) as f64;

            let (activation, packed, scales, zp_packed) = zc_make_inputs(k, n, block_size);

            // Resident buffers (identical for both arms).
            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let scales_dev = runtime.alloc_raw(scales.len() * 2).unwrap();
            let zp_dev = runtime.alloc_raw(zp_packed.len().max(1)).unwrap();
            let output_dev = runtime.alloc_raw(n * 2).unwrap();
            // VRAM copy of the weights (the arm the managed path streams into).
            let packed_vram = runtime.alloc_raw(packed_bytes).unwrap();

            // SAFETY: each device buffer was sized to its source slice.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(as_bytes(&scales), scales_dev).unwrap();
                runtime.htod(&zp_packed, zp_dev).unwrap();
                runtime.htod(&packed, packed_vram).unwrap();
            }

            // Host-mapped weights: page-lock the same bytes READ_ONLY and take a
            // device-addressable pointer into them (the zero-copy cold path).
            let host_ptr = packed.as_ptr() as *mut c_void;
            // SAFETY: `packed` outlives the registration; it is unregistered
            // below before `packed` is dropped, and never reallocated meanwhile.
            unsafe {
                sys::cuMemHostRegister_v2(
                    host_ptr,
                    packed_bytes,
                    CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_READ_ONLY,
                )
                .result()
                .expect("cuMemHostRegister(DEVICEMAP|READ_ONLY)");
            }
            let mut packed_host_dptr: CUdeviceptr = 0;
            // SAFETY: `host_ptr` was just registered with DEVICEMAP.
            unsafe {
                sys::cuMemHostGetDevicePointer_v2(&mut packed_host_dptr, host_ptr, 0)
                    .result()
                    .expect("cuMemHostGetDevicePointer");
            }

            let kernel = zc_kernel(&runtime, k, n, block_size);

            let a_shape = [1usize, k];
            let a_strides = [k as i64, 1];
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let s_shape = [n, k_blocks];
            let s_strides = [k_blocks as i64, 1];
            let zp_shape = [n, zp_row_bytes];
            let zp_strides = [zp_row_bytes as i64, 1];
            let y_shape = [1usize, n];
            let y_strides = [n as i64, 1];
            let device = DeviceId::cuda(0);

            let make_inputs = |packed_ptr: CUdeviceptr| {
                vec![
                    TensorView::new(
                        device_ptr(activation_dev),
                        DataType::Float16,
                        &a_shape,
                        &a_strides,
                        device,
                    ),
                    TensorView::new(
                        device_ptr(packed_ptr),
                        DataType::Uint8,
                        &b_shape,
                        &b_strides,
                        device,
                    ),
                    TensorView::new(
                        device_ptr(scales_dev),
                        DataType::Float16,
                        &s_shape,
                        &s_strides,
                        device,
                    ),
                    TensorView::new(
                        device_ptr(zp_dev),
                        DataType::Uint8,
                        &zp_shape,
                        &zp_strides,
                        device,
                    ),
                ]
            };

            let time_arm = |packed_ptr: CUdeviceptr| -> Vec<f32> {
                let inputs = make_inputs(packed_ptr);
                let mut outputs = [TensorMut::new(
                    device_ptr_mut(output_dev),
                    DataType::Float16,
                    &y_shape,
                    &y_strides,
                    device,
                )];
                // Warm up (first call also compiles the NVRTC module).
                kernel.run(&inputs, &mut outputs, None).unwrap();
                runtime.synchronize().unwrap();
                let mut times = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let start = event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).unwrap();
                    let end = event::create(sys::CUevent_flags::CU_EVENT_DEFAULT).unwrap();
                    // SAFETY: both events were just created on this context; the
                    // launch enqueues on the same stream between the records.
                    unsafe {
                        event::record(start, runtime.stream_ptr()).unwrap();
                        kernel.run(&inputs, &mut outputs, None).unwrap();
                        event::record(end, runtime.stream_ptr()).unwrap();
                        event::synchronize(end).unwrap();
                        times.push(event::elapsed(start, end).unwrap());
                        event::destroy(start).ok();
                        event::destroy(end).ok();
                    }
                }
                times
            };

            // VRAM arm, then read its output.
            let vram_times = time_arm(packed_vram);
            let mut got_vram = vec![f16::ZERO; n];
            // SAFETY: `output_dev` holds `n` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut got_vram), output_dev)
                    .unwrap()
            };

            // Host-mapped arm, then read its output.
            let host_times = time_arm(packed_host_dptr);
            let mut got_host = vec![f16::ZERO; n];
            // SAFETY: `output_dev` holds `n` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut got_host), output_dev)
                    .unwrap()
            };

            let best_ms = |t: &[f32]| t.iter().cloned().fold(f32::INFINITY, f32::min);
            let med_ms = |t: &[f32]| {
                let mut s = t.to_vec();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                s[s.len() / 2]
            };
            let gbps = |ms: f32| (packed_bytes as f64) / (ms as f64 / 1e3) / 1e9;
            let vram_gbps = gbps(best_ms(&vram_times));
            let host_gbps = gbps(best_ms(&host_times));

            let bit_match = got_vram
                .iter()
                .zip(got_host.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());

            println!(
                "{:>10.1} {:>9} {:>12.2} {:>12.2} {:>7.2}x {:>10}",
                packed_mib,
                n,
                vram_gbps,
                host_gbps,
                vram_gbps / host_gbps,
                if bit_match { "yes" } else { "NO" }
            );
            println!(
                "           spread: VRAM best={:.4} med={:.4} ms  host best={:.4} med={:.4} ms",
                best_ms(&vram_times),
                med_ms(&vram_times),
                best_ms(&host_times),
                med_ms(&host_times),
            );

            // Release this size's resources before the next (host RAM is
            // page-locked; VRAM is scarce). Not in a Drop guard — asserting in
            // Drop trips STATUS_STACK_BUFFER_OVERRUN in this tree.
            // SAFETY: `host_ptr` is still registered and `packed` still alive.
            unsafe {
                sys::cuMemHostUnregister(host_ptr)
                    .result()
                    .expect("cuMemHostUnregister");
            }
            // SAFETY: each pointer came from this runtime's `alloc_raw`, freed once.
            unsafe {
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(scales_dev).unwrap();
                runtime.free_raw(zp_dev).unwrap();
                runtime.free_raw(output_dev).unwrap();
                runtime.free_raw(packed_vram).unwrap();
            }
            drop(packed);

            assert!(
                bit_match,
                "zero-copy host-mapped read produced a DIFFERENT result than the VRAM read \
                 (packed={packed_mib:.1} MiB, N={n}) — correctness failure"
            );
        }
    }

    /// #864 second question: can CUDA graph capture bake a **host-mapped**
    /// device pointer into a captured `MatMulNBits` launch and replay it
    /// correctly? Capture is a hard gate on the streaming path (`captures > 0`,
    /// `fallbacks == 0`, #796); if host-mapped pointers cannot be captured the
    /// hybrid's cold path is incompatible with capture. This captures the real
    /// decode GEMV reading its weights zero-copy from host memory, replays it,
    /// and asserts the replayed output matches an eager host-mapped run.
    ///
    /// `#[ignore]`: needs a live CUDA device. Run solo:
    ///   cargo test -p onnx-runtime-ep-cuda --features cuda --release \
    ///     zerocopy_kernel_capture_with_host_mapped_pointer -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore]
    fn zerocopy_kernel_capture_with_host_mapped_pointer() {
        use cudarc::driver::sys;

        let Some(runtime) = runtime() else {
            eprintln!("skipping capture probe: CUDA runtime unavailable");
            return;
        };
        if runtime.require_nvrtc_half_headers("capture probe").is_err() {
            eprintln!("skipping capture probe: fp16 NVRTC headers unavailable");
            return;
        }
        runtime.bind().unwrap();

        const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;
        const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 0x08;

        let k = zc_env_usize("ZC_K", 5120);
        let block_size = zc_env_usize("ZC_BLOCK", 32);
        let n = zc_env_usize("ZC_CAPTURE_N", 4608);
        assert!(k.is_multiple_of(block_size));
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);
        let packed_bytes = n * k_blocks * blob_size;

        let (activation, packed, scales, zp_packed) = zc_make_inputs(k, n, block_size);

        let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
        let scales_dev = runtime.alloc_raw(scales.len() * 2).unwrap();
        let zp_dev = runtime.alloc_raw(zp_packed.len().max(1)).unwrap();
        let output_dev = runtime.alloc_raw(n * 2).unwrap();

        // SAFETY: each device buffer was sized to its source slice.
        unsafe {
            runtime.htod(as_bytes(&activation), activation_dev).unwrap();
            runtime.htod(as_bytes(&scales), scales_dev).unwrap();
            runtime.htod(&zp_packed, zp_dev).unwrap();
        }

        let host_ptr = packed.as_ptr() as *mut c_void;
        // SAFETY: `packed` outlives the registration and is unregistered below.
        unsafe {
            sys::cuMemHostRegister_v2(
                host_ptr,
                packed_bytes,
                CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_READ_ONLY,
            )
            .result()
            .expect("cuMemHostRegister(DEVICEMAP|READ_ONLY)");
        }
        let mut packed_host_dptr: CUdeviceptr = 0;
        // SAFETY: `host_ptr` was just registered with DEVICEMAP.
        unsafe {
            sys::cuMemHostGetDevicePointer_v2(&mut packed_host_dptr, host_ptr, 0)
                .result()
                .expect("cuMemHostGetDevicePointer");
        }

        let kernel = zc_kernel(&runtime, k, n, block_size);

        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let s_shape = [n, k_blocks];
        let s_strides = [k_blocks as i64, 1];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];
        let device = DeviceId::cuda(0);

        let inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_host_dptr),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                DataType::Float16,
                &s_shape,
                &s_strides,
                device,
            ),
            TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            ),
        ];
        let mut outputs = [TensorMut::new(
            device_ptr_mut(output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        )];

        // Eager reference (also compiles the NVRTC module before capture).
        kernel.run(&inputs, &mut outputs, None).unwrap();
        runtime.synchronize().unwrap();
        let mut eager = vec![f16::ZERO; n];
        // SAFETY: `output_dev` holds `n` fp16 values.
        unsafe { runtime.dtoh(as_bytes_mut(&mut eager), output_dev).unwrap() };

        // Clobber the output so a broken replay cannot masquerade as a pass.
        // SAFETY: `output_dev` holds `n` fp16 values.
        unsafe {
            runtime
                .htod(as_bytes(&vec![f16::ONE; n]), output_dev)
                .unwrap()
        };

        let captured = runtime
            .begin_graph_capture(&[&kernel as &dyn Kernel])
            .is_ok();
        let mut replay_match = false;
        if captured {
            kernel.run(&inputs, &mut outputs, None).unwrap();
            match runtime.end_graph_capture() {
                Ok(()) => {
                    for _ in 0..4 {
                        runtime.replay_graph().unwrap();
                    }
                    runtime.synchronize().unwrap();
                    let mut got = vec![f16::ZERO; n];
                    // SAFETY: `output_dev` holds `n` fp16 values.
                    unsafe { runtime.dtoh(as_bytes_mut(&mut got), output_dev).unwrap() };
                    replay_match = got
                        .iter()
                        .zip(eager.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    runtime.reset_graph().ok();
                    println!(
                        "\n=== #864 capture-with-host-mapped-pointer probe ===\n\
                         capture: SUPPORTED; replay output bit-identical to eager: {}",
                        if replay_match { "YES" } else { "NO" }
                    );
                }
                Err(e) => {
                    runtime.abort_graph_capture().ok();
                    println!(
                        "\n=== #864 capture-with-host-mapped-pointer probe ===\n\
                         capture: end_graph_capture FAILED: {e:?}"
                    );
                }
            }
        } else {
            println!(
                "\n=== #864 capture-with-host-mapped-pointer probe ===\n\
                 capture: begin_graph_capture declined (kernel/audit not capturable in this build)"
            );
        }

        // SAFETY: `host_ptr` is still registered and `packed` still alive.
        unsafe {
            sys::cuMemHostUnregister(host_ptr)
                .result()
                .expect("cuMemHostUnregister");
        }
        // SAFETY: each pointer came from this runtime's `alloc_raw`, freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }
        drop(packed);

        if captured {
            assert!(
                replay_match,
                "graph replay with a host-mapped weight pointer produced a different result \
                 than the eager host-mapped run"
            );
        }
    }

    /// Dequant-reference parity for the int8 (bits=8) fp16-activation decode
    /// GEMV at arbitrary `(k, n)`. Exercises the vectorised four-lane/eight-block
    /// path against an f64 oracle; `explicit_zp` toggles the symmetric zp=128
    /// default versus an explicit per-block uint8 zero point.
    fn run_int8_parity_dims(
        k: usize,
        n: usize,
        block_size: usize,
        scales_fp16: bool,
        with_bias: bool,
        explicit_zp: bool,
    ) -> (f32, f32, f32, bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping MatMulNBits int8 fp16 GEMV parity test: CUDA runtime unavailable");
            return (0.0, 0.0, 0.0, true);
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!(
                "skipping MatMulNBits int8 fp16 GEMV parity test: fp16 NVRTC headers unavailable"
            );
            return (0.0, 0.0, 0.0, true);
        }

        let k_blocks = k / block_size;
        let blob_size = block_size; // one byte per weight for bits=8

        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation_f16 = vec![f16::ZERO; k];
        let mut activation_ref = vec![0.0f32; k];
        for (dst_h, dst_f) in activation_f16.iter_mut().zip(activation_ref.iter_mut()) {
            let h = f16::from_f32(next());
            *dst_h = h;
            *dst_f = h.to_f32();
        }

        // int8 quant codes (0..255), one byte per weight in [n, k_blocks, 32].
        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for within in 0..block_size {
                    packed[(col * k_blocks + block) * blob_size + within] =
                        quant[col * k + block * block_size + within];
                }
            }
        }

        // Explicit per-block uint8 zero points, or the symmetric 128 default.
        let mut zero_points = vec![0u8; n * k_blocks];
        if explicit_zp {
            for zp in zero_points.iter_mut() {
                *zp = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        let zp_ref = |col: usize, block: usize| -> i32 {
            if explicit_zp {
                zero_points[col * k_blocks + block] as i32
            } else {
                128
            }
        };

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

        let mut bias_f16 = vec![f16::ZERO; n];
        let mut bias_ref = vec![0.0f32; n];
        if with_bias {
            for (h, f) in bias_f16.iter_mut().zip(bias_ref.iter_mut()) {
                let value = f16::from_f32(next());
                *h = value;
                *f = value.to_f32();
            }
        }

        let mut expected = vec![0.0f32; n];
        for col in 0..n {
            let mut acc = 0.0f64;
            for block in 0..k_blocks {
                let scale = scale_ref[col * k_blocks + block] as f64;
                let zp = zp_ref(col, block);
                for within in 0..block_size {
                    let depth = block * block_size + within;
                    let q = quant[col * k + depth] as i32 - zp;
                    acc += activation_ref[depth] as f64 * q as f64 * scale;
                }
            }
            if with_bias {
                acc += bias_ref[col] as f64;
            }
            expected[col] = acc as f32;
        }

        let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime
            .alloc_raw(n * k_blocks * if scales_fp16 { 2 } else { 4 })
            .unwrap();
        let zp_dev = runtime.alloc_raw(zero_points.len()).unwrap();
        let bias_dev = runtime.alloc_raw(n * 2).unwrap();
        let output_dev = runtime.alloc_raw(n * 2).unwrap();

        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime
                .htod(as_bytes(&activation_f16), activation_dev)
                .unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            if scales_fp16 {
                runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            } else {
                runtime.htod(as_bytes(&scale_f32), scales_dev).unwrap();
            }
            if explicit_zp {
                runtime.htod(&zero_points, zp_dev).unwrap();
            }
            if with_bias {
                runtime.htod(as_bytes(&bias_f16), bias_dev).unwrap();
            }
        }

        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, k_blocks];
        let zp_strides = [k_blocks as i64, 1];
        let bias_shape = [n];
        let bias_strides = [1i64];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];

        let scales_dtype = if scales_fp16 {
            DataType::Float16
        } else {
            DataType::Float32
        };
        let device = DeviceId::cuda(0);
        let mut inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                scales_dtype,
                &scales_shape,
                &scales_strides,
                device,
            ),
        ];
        if explicit_zp || with_bias {
            inputs.push(if explicit_zp {
                TensorView::new(
                    device_ptr(zp_dev),
                    DataType::Uint8,
                    &zp_shape,
                    &zp_strides,
                    device,
                )
            } else {
                TensorView::absent(DataType::Uint8)
            });
        }
        if with_bias {
            inputs.push(TensorView::absent(DataType::Int32));
            inputs.push(TensorView::new(
                device_ptr(bias_dev),
                DataType::Float16,
                &bias_shape,
                &bias_strides,
                device,
            ));
        }

        let mut outputs = [TensorMut::new(
            device_ptr_mut(output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        )];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 8,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };
        kernel.run(&inputs, &mut outputs, None).unwrap();
        runtime.synchronize().unwrap();

        assert!(
            kernel.last_call_capture_safe.load(Ordering::Relaxed),
            "int8 decode GEMV must report capture-safe"
        );

        let mut got_f16 = vec![f16::ZERO; n];
        // SAFETY: `output_dev` holds `n` fp16 values.
        unsafe {
            runtime
                .dtoh(as_bytes_mut(&mut got_f16), output_dev)
                .unwrap();
        }

        // SAFETY: each pointer came from this runtime's `alloc_raw` and is freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(bias_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut max_out = 0.0f32;
        let mut all_finite = true;
        for (g16, e) in got_f16.iter().zip(expected.iter()) {
            let g = g16.to_f32();
            if !g.is_finite() {
                all_finite = false;
            }
            let abs = (g - e).abs();
            let rel = abs / e.abs().max(1e-1);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
            max_out = max_out.max(e.abs());
        }
        (worst_abs, worst_rel, max_out, all_finite)
    }

    #[test]
    fn fp16_down_projection_is_bit_exact_to_staged_kernel() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping down-projection GEMV parity test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping down-projection GEMV parity test: fp16 NVRTC headers unavailable");
            return;
        }

        // Prove the specialization matches the general GEMV bit-numerically for
        // the Qwen down shape AND an unrelated non-Qwen tall-skinny shape, so
        // the generalized selection is correct beyond one architecture.
        for (k, n) in [
            (QWEN_DOWN_K, QWEN_DOWN_N),
            (5632usize, 2048usize),
            // K=16384, N=8192 selects the 4-column grid-fill entry on a 132-SM
            // device, so this covers the `_c4` variant bit-exactly too.
            (16_384usize, 8192usize),
        ] {
            assert_eq!(
                select_f16_gemv_variant(k, n, 32, true, false).variant,
                F16GemvVariant::DownProjection,
                "shape K={k}, N={n} must select the down variant under test"
            );
            let block_size = 32usize;
            let k_blocks = k / block_size;
            let blob_size = block_size / 2;

            let activation: Vec<f16> = (0..k)
                .map(|i| f16::from_f32(((i * 17 % 257) as f32 - 128.0) / 128.0))
                .collect();
            let packed: Vec<u8> = (0..n * k_blocks * blob_size)
                .map(|i| ((i * 29 + i / 7 + 13) & 0xff) as u8)
                .collect();
            let scales: Vec<f16> = (0..n * k_blocks)
                .map(|i| f16::from_f32(0.01 + (i % 17) as f32 * 0.0005))
                .collect();

            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
            let scales_dev = runtime.alloc_raw(scales.len() * 2).unwrap();
            let staged_output_dev = runtime.alloc_raw(n * 2).unwrap();
            let down_output_dev = runtime.alloc_raw(n * 2).unwrap();
            // SAFETY: device buffers exactly cover their source slices.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(&packed, packed_dev).unwrap();
                runtime.htod(as_bytes(&scales), scales_dev).unwrap();
            }

            let device = DeviceId::cuda(0);
            let a_shape = [1usize, k];
            let a_strides = [k as i64, 1];
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let scales_shape = [n, k_blocks];
            let scales_strides = [k_blocks as i64, 1];
            let y_shape = [1usize, n];
            let y_strides = [n as i64, 1];
            let activation_view = TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            );
            let packed_view = TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_view = TensorView::new(
                device_ptr(scales_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let mut down_output = TensorMut::new(
                device_ptr_mut(down_output_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );
            let kernel = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: false,
                decomposed_silu: false,
                rmsnorm_prologue: false,
                rmsnorm_epsilon: 1e-5,
                last_call_capture_safe: AtomicBool::new(false),
                bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
                bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
            };
            let staged_source = format!("{GEMV_F16_SRC}\n{STAGED_DOWN_REFERENCE_SRC}");
            let staged_function = runtime
                .nvrtc_function(
                    "matmul_nbits_gemv_f16_down_staged_reference",
                    &staged_source,
                    STAGED_DOWN_REFERENCE_ENTRY,
                )
                .unwrap();
            let activation_ptr = cuptr(activation_view.data_ptr::<u8>() as *const c_void);
            let packed_ptr = cuptr(packed_view.data_ptr::<u8>() as *const c_void);
            let scales_ptr = cuptr(scales_view.data_ptr::<u8>() as *const c_void);
            let zero_points_ptr: CUdeviceptr = 0;
            let bias_ptr: CUdeviceptr = 0;
            let staged_output_ptr = staged_output_dev;
            let k_i32 = as_i32("K", k).unwrap();
            let n_i32 = as_i32("N", n).unwrap();
            let block_size_i32 = as_i32("block_size", block_size).unwrap();
            let k_blocks_i32 = as_i32("K block count", k_blocks).unwrap();
            let blob_size_i32 = as_i32("block blob size", blob_size).unwrap();
            let zp_row_bytes_i32 =
                as_i32("zero-point row byte count", k_blocks.div_ceil(2)).unwrap();
            let scales_fp16_flag = 1i32;
            let bias_post_round_flag = 0i32;
            let mut staged_builder = runtime.stream().launch_builder(&staged_function);
            staged_builder
                .arg(&activation_ptr)
                .arg(&packed_ptr)
                .arg(&scales_ptr)
                .arg(&zero_points_ptr)
                .arg(&bias_ptr)
                .arg(&staged_output_ptr)
                .arg(&k_i32)
                .arg(&n_i32)
                .arg(&block_size_i32)
                .arg(&k_blocks_i32)
                .arg(&blob_size_i32)
                .arg(&zp_row_bytes_i32)
                .arg(&scales_fp16_flag)
                .arg(&bias_post_round_flag);
            // SAFETY: this launches the exact pre-change down-projection entry
            // over the same validated buffers used by the replacement below.
            unsafe {
                staged_builder
                    .launch(LaunchConfig {
                        grid_dim: (n.div_ceil(GEMV_F16_DOWN_COLUMNS_PER_BLOCK) as u32, 1, 1),
                        block_dim: (GEMV_F16_DOWN_THREADS, 1, 1),
                        shared_mem_bytes: (k * std::mem::size_of::<f16>()) as u32,
                    })
                    .unwrap();
            }
            kernel
                .launch_f16_gemv_variant(
                    &activation_view,
                    &packed_view,
                    &scales_view,
                    true,
                    None,
                    None,
                    &mut down_output,
                    k_blocks,
                    blob_size,
                    k_blocks.div_ceil(2),
                    select_f16_gemv_variant(k, n, block_size, true, false),
                )
                .unwrap();
            runtime.synchronize().unwrap();

            let mut staged = vec![f16::ZERO; n];
            let mut down = vec![f16::ZERO; n];
            // SAFETY: both output allocations hold `n` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut staged), staged_output_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut down), down_output_dev)
                    .unwrap();
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(packed_dev).unwrap();
                runtime.free_raw(scales_dev).unwrap();
                runtime.free_raw(staged_output_dev).unwrap();
                runtime.free_raw(down_output_dev).unwrap();
            }

            assert_eq!(
                as_bytes(&staged),
                as_bytes(&down),
                "register-loaded down projection must be bit-exact to the pre-change staged kernel \
                 at K={k}, N={n}"
            );
        }
    }

    #[test]
    fn fp16_gemv_variant_selection_is_structural() {
        // The down variant is selected by the tall-skinny (K>N) block-32 fp16
        // shape *class*, generalizing across models — not by a magic K/N.
        let qwen = select_f16_gemv_variant(QWEN_DOWN_K, QWEN_DOWN_N, 32, true, false);
        assert_eq!(qwen.variant, F16GemvVariant::DownProjection);
        assert_eq!(
            qwen.reason,
            "variant=down_projection;class=tall_skinny(K>N);block_size=32;\
             scales=fp16;K%32==0"
        );

        // Non-Qwen tall-skinny down/output projections, including contractions
        // larger than the former activation-staging limit, must also select it.
        for (k, n) in [
            (5632, 2048),
            (11008, 4096),
            (2048, 512),
            (4096, 4096 - 32),
            (32_768, 4096),
        ] {
            let selection = select_f16_gemv_variant(k, n, 32, true, false);
            assert_eq!(
                selection.variant,
                F16GemvVariant::DownProjection,
                "tall-skinny K={k}, N={n} must select the down variant"
            );
        }

        // Wide (N>=K) projections, non-multiple-of-32 K, and non-block-32 all
        // fall back.
        let general_cases = [
            (896, 4864, 32, true),    // gate/up: N > K
            (896, 896, 32, true),     // square: K == N is not tall-skinny
            (896, 151_936, 32, true), // lm_head: N >> K
            (4880, 896, 32, true),    // 4880 % 32 != 0
            (4864, 896, 64, true),    // block_size != 32
        ];
        for (k, n, block_size, scales_fp16) in general_cases {
            let selection = select_f16_gemv_variant(k, n, block_size, scales_fp16, false);
            assert_eq!(
                selection.variant,
                F16GemvVariant::General,
                "K={k}, N={n}, block_size={block_size} must retain the general GEMV"
            );
        }

        // fp32 scales are never down-eligible even for a tall-skinny shape.
        assert_eq!(
            select_f16_gemv_variant(QWEN_DOWN_K, QWEN_DOWN_N, 32, false, false).variant,
            F16GemvVariant::General,
        );

        let asymmetric = select_f16_gemv_variant(QWEN_DOWN_K, QWEN_DOWN_N, 32, true, true);
        assert_eq!(asymmetric.variant, F16GemvVariant::General);
        assert_eq!(
            asymmetric.reason,
            "variant=general;zero_points=explicit;down_projection requires symmetric zp=8"
        );
    }

    #[test]
    fn scales_f16_pipe_well_occupied_routes_lm_head_to_plain() {
        // The pipe-vs-plain occupancy gate keys only on N, the launch's
        // columns-per-CTA and the live SM count — no model-specific magic.
        // qwen3.5-0.8b measured shapes on an H200 (132 SMs, 8-warp large launch
        // => 8 columns per CTA):
        //   LM head  N=248320 -> grid 31040 (~235 CTAs/SM): WELL-OCCUPIED -> plain
        //   proj     N=4096   -> grid 512   (~3.9 CTAs/SM): grid-starved  -> pipe
        assert!(
            scales_f16_pipe_well_occupied(248_320, 8, 132),
            "the wide LM-head projection must be treated as well-occupied (plain)"
        );
        assert!(
            !scales_f16_pipe_well_occupied(4096, 8, 132),
            "a grid-starved N=4096 projection must keep the pipe entry"
        );

        // Device-derived, not a fixed N: the same width flips with the SM count.
        // grid = ceil(N/cols); well-occupied when grid >= mp_count*32.
        // N=8192, cols=8 -> grid 1024; threshold(mp)=mp*32.
        assert!(
            scales_f16_pipe_well_occupied(8192, 8, 32),
            "grid 1024 >= 32*32=1024 on a 32-SM device is well-occupied"
        );
        assert!(
            !scales_f16_pipe_well_occupied(8192, 8, 33),
            "grid 1024 < 33*32=1056 on a 33-SM device stays grid-starved"
        );

        // Degenerate inputs must not divide by zero.
        assert!(scales_f16_pipe_well_occupied(1_000_000, 0, 1));
        assert!(!scales_f16_pipe_well_occupied(1, 8, 0));
    }

    #[test]
    fn symmetric_int8_splitk_gate_targets_grid_starved_only() {
        // The symmetric (no zero-point) int8 decode GEMV reuses the symmetric
        // int4 grid-fill predicate: split-K fires only when the single-warp grid
        // is too narrow to fill the device (N < mp_count * 16), so the
        // already-occupied wide projections keep the byte-identical single-warp
        // entry. qwen3.5-0.8b int8 shapes on an H200 (132 SMs => threshold 2112):
        //   N=1024 (down/o)  -> grid-starved  -> split-K
        //   N=2048           -> grid-starved  -> split-K
        //   N=3584 / N=6144  -> occupied      -> single-warp
        let (mp, max_threads) = (132u32, GEMV_F16_LARGE_THREADS);
        assert!(
            use_f16_symmetric_splitk(2048, 1024, mp, max_threads),
            "grid-starved N=1024 symmetric int8 must take split-K"
        );
        assert!(
            use_f16_symmetric_splitk(1024, 2048, mp, max_threads),
            "grid-starved N=2048 symmetric int8 must take split-K"
        );
        assert!(
            !use_f16_symmetric_splitk(1024, 3584, mp, max_threads),
            "occupied N=3584 symmetric int8 must keep the single-warp entry"
        );
        assert!(
            !use_f16_symmetric_splitk(1024, 6144, mp, max_threads),
            "occupied N=6144 symmetric int8 must keep the single-warp entry"
        );
        // K < 512 is ineligible (no whole 256-wide split step to spread).
        assert!(!use_f16_symmetric_splitk(256, 512, mp, max_threads));
        // Device-derived, not a fixed N: the same width flips with the SM count.
        assert!(use_f16_symmetric_splitk(1024, 3584, 256, max_threads));

        // The opt-out flag is default-on (enabled) when unset.
        // (Env-driven; asserted here only for the unset default to avoid racing
        // other tests that may set process-wide env.)
        if std::env::var_os("ONNX_GENAI_CUDA_DISABLE_INT8_SYMMETRIC_SPLITK").is_none() {
            assert!(int8_symmetric_splitk_enabled());
        }
    }

    #[test]
    fn splitk_smalln_single_column_is_shape_driven_and_general() {
        // A narrow (grid-starved) projection — e.g. a GQA k/v projection with
        // N = kv_heads * head_dim (2*128 = 256) — under-fills the device with the
        // multicol hybrid (grid = ceil(256/8) = 32 CTAs << 132 SMs), so it must
        // prefer the byte-identical single-column split-K wide entry whose 4x
        // larger grid fills the idle SMs.
        assert!(
            splitk_smalln_prefers_single_column(256, 132),
            "a narrow N=256 kv projection must fall back to the single-column entry"
        );
        // Larger GQA kv widths that still under-fill also flip (shape-driven, not
        // a single magic constant): N=512 -> grid 64 < 132.
        assert!(splitk_smalln_prefers_single_column(512, 132));

        // The medium/large projections (down_proj / q / o at N~4096) keep the
        // multicol hybrid: grid = ceil(4096/8) = 512 >= 132 SMs, so the register
        // blocking wins and the grid already fills the device.
        assert!(
            !splitk_smalln_prefers_single_column(4096, 132),
            "a wide N=4096 projection must keep the multicol hybrid"
        );
        assert!(!splitk_smalln_prefers_single_column(27_392, 132));

        // Device-derived, not model-specific: the same N flips on a bigger device
        // (more SMs to fill) and holds on a tiny one.
        assert!(
            splitk_smalln_prefers_single_column(1024, 132),
            "N=1024 -> grid 128 < 132 SMs still under-fills a large device"
        );
        assert!(
            !splitk_smalln_prefers_single_column(1024, 100),
            "N=1024 -> grid 128 >= 100 SMs fills a smaller device"
        );

        // Env override forces the decision for A/B measurement.
        // SAFETY: serial logic test (--test-threads=1); the flag is cleared below
        // so it cannot leak into other tests.
        unsafe {
            std::env::set_var("ONNX_GENAI_GEMV_SPLITK_SMALLN_SINGLECOL", "1");
        }
        assert!(splitk_smalln_prefers_single_column(4096, 132));
        unsafe {
            std::env::set_var("ONNX_GENAI_GEMV_SPLITK_SMALLN_SINGLECOL", "0");
        }
        assert!(!splitk_smalln_prefers_single_column(256, 132));
        unsafe {
            std::env::remove_var("ONNX_GENAI_GEMV_SPLITK_SMALLN_SINGLECOL");
        }
    }

    #[test]
    fn down_columns_fill_the_device_and_never_undersplit() {
        // Wide down projection: the 8-column grid already clears the ~2-wave
        // target, so keep the cheapest 8-column launch.
        assert_eq!(
            select_down_columns(65_536, 132),
            (8, GEMV_F16_DOWN_ENTRY),
            "a wide down projection must keep the 8-column launch"
        );
        // Mid-width: 8-column grid underfills but the 4-column one clears it.
        assert_eq!(
            select_down_columns(8192, 132),
            (4, GEMV_F16_DOWN_C4_ENTRY),
            "a mid-width down projection must split to 4 columns/CTA"
        );
        // Grid-starved (Qwen2.5-7B down N=3584 on an H200's 132 SMs): split to
        // the measured-optimal 2 columns/CTA.
        assert_eq!(
            select_down_columns(3584, 132),
            (2, GEMV_F16_DOWN_C2_ENTRY),
            "a grid-starved down projection must split to 2 columns/CTA"
        );
        // Even a tiny N never drops below 2 columns/CTA (1-column launches
        // over-subscribe and re-read the activation 8x, erasing the gain).
        assert_eq!(select_down_columns(64, 132).0, 2);
        // A tiny/degenerate device (clamped to >=1 SM) has a small target, so a
        // wide-enough N stays on the cheapest 8-column launch instead of
        // over-splitting.
        assert_eq!(select_down_columns(3584, 0), (8, GEMV_F16_DOWN_ENTRY));
    }

    #[test]
    fn accuracy4_stage64_is_limited_to_sub_wave_block32_grids() {
        assert!(use_accuracy4_stage64(4608, 132, (9, 0), 64 * 1024));
        assert!(!use_accuracy4_stage64(4608, 46, (8, 9), 48 * 1024));
        assert!(!use_accuracy4_stage64(4608, 28, (8, 6), 48 * 1024));
        assert!(!use_accuracy4_stage64(4608, 132, (9, 0), 1024));
        assert!(!use_accuracy4_stage64(65_536, 132, (9, 0), 64 * 1024));
    }

    /// Byte-identity guard for routing `use_accuracy4_stage64`'s resident-warp
    /// estimate through the arch layer: the arch helper must reproduce the exact
    /// pre-refactor inline ladder for every compute capability, so the stage-64
    /// decision (and therefore H200's selection) is provably unchanged.
    #[test]
    fn accuracy4_resident_warps_matches_prior_inline_ladder() {
        fn prior_inline_ladder(compute_capability: (u32, u32)) -> u32 {
            match compute_capability {
                (8, 0) | (9.., _) => 64,
                _ => 48,
            }
        }
        for major in 0u32..=13 {
            for minor in 0u32..=12 {
                let cc = (major, minor);
                assert_eq!(
                    crate::arch::decode_resident_warps_per_sm(cc),
                    prior_inline_ladder(cc),
                    "arch resident-warp ladder diverged from the frozen inline ladder at {cc:?}"
                );
            }
        }
        // And the full stage-64 decision is unchanged across a shape/arch sweep.
        for &(n, sms, cc, smem) in &[
            (4608usize, 132u32, (9u32, 0u32), 64 * 1024u32),
            (4608, 46, (8, 9), 48 * 1024),
            (4608, 28, (8, 6), 48 * 1024),
            (1024, 132, (9, 0), 64 * 1024),
            (65_536, 132, (9, 0), 64 * 1024),
            (2048, 84, (8, 9), 64 * 1024),
            (4096, 68, (8, 6), 64 * 1024),
        ] {
            let resident = prior_inline_ladder(cc) as usize;
            let resident_ctas = resident / (GEMV_ACCURACY4_THREADS as usize / 32);
            let one_wave = (sms.max(1) as usize).saturating_mul(resident_ctas);
            let expected = smem >= GEMV_ACCURACY4_STAGE64_SHARED_BYTES
                && n.div_ceil(GEMV_ACCURACY4_COLUMNS_PER_BLOCK) < one_wave;
            assert_eq!(
                use_accuracy4_stage64(n, sms, cc, smem),
                expected,
                "stage64 decision changed for n={n} sms={sms} cc={cc:?} smem={smem}"
            );
        }
    }

    #[test]
    fn symmetric_fp16_splitk_is_device_driven_and_falls_back_on_small_gpus() {
        assert!(use_f16_symmetric_splitk(896, 1152, 132, 1024));
        assert!(!use_f16_symmetric_splitk(896, 1152, 46, 1024));
        assert!(!use_f16_symmetric_splitk(896, 1152, 132, 128));
        assert!(!use_f16_symmetric_splitk(256, 1024, 132, 1024));
    }

    #[test]
    fn fp16_down_projection_loads_activation_directly_into_registers() {
        let start = GEMV_F16_SRC
            .find("void matmul_nbits_gemv_f16_scales_f16_down_tpl")
            .expect("down-projection template must exist");
        let body = &GEMV_F16_SRC[start..];
        let end = body
            .find("\n}\n\n// Default 8-column down projection")
            .expect("down-projection template must have a bounded body");
        let body = &body[..end];

        assert!(
            !body.contains("activation_shared"),
            "down projection must not round-trip activations through shared memory"
        );
        for offset in [
            "activation_block);",
            "activation_block + 8);",
            "activation_block + 16);",
            "activation_block + 24);",
        ] {
            assert!(
                body.contains(offset),
                "down projection must directly load the block-32 activation at {offset}"
            );
        }
    }

    #[test]
    fn fp16_gemv_matches_dequant_reference() {
        let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
            (0.0f32, 0.0f32, 0.0f32, true);
        for (scales_fp16, with_bias) in [(false, false), (true, false), (true, true)] {
            let (abs, rel, out, finite) = run_parity(scales_fp16, with_bias);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
            max_out = max_out.max(out);
            all_finite &= finite;
        }
        // fp16 output ULP is 2^-11 (~4.9e-4) of a value's magnitude, so the
        // absolute error floor scales with the largest output component. Bound
        // the observed abs error against that magnitude with 2x headroom for the
        // fp32-vs-f64 reduction-order drift accumulated over K=4096.
        let abs_bound = (max_out * 1e-3).max(1e-3);
        eprintln!(
            "MatMulNBits fp16 GEMV parity: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} \
             max_out={max_out:.3e} abs_bound={abs_bound:.3e}"
        );
        assert!(all_finite, "fp16 GEMV produced a non-finite output");
        assert!(
            worst_abs < abs_bound,
            "fp16 GEMV diverged from dequant reference: max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
        );
        // Relative error (against a 1e-1 floor so near-zero columns do not
        // explode the ratio) isolates the per-element accuracy from the output
        // magnitude and must stay well under 5e-2.
        assert!(
            worst_rel < 5e-2,
            "fp16 GEMV diverged from dequant reference: max_rel={worst_rel:.3e}"
        );
    }

    /// Multi-request batch decode byte-identity guard for the looped fp16 decode
    /// GEMV (limiter B fix). For `1 < m <= decode_gemv_loop_max_m()`, `run_f16`
    /// runs the specialized single-row decode GEMV once per row instead of the
    /// portable tiled prefill GEMM. Because each row is dispatched through the
    /// *identical* `dispatch_f16_decode_gemv_row` path the M==1 decode step uses,
    /// row `r` of the batched output must be BIT-for-BIT equal to running that
    /// row alone as an M==1 GEMV. This is the strong form of the bit-identity
    /// claim: batched decode is byte-identical to single-stream decode, so a
    /// batch cannot change any user's logits. A regression that reintroduced a
    /// tiled-GEMM (fp32-accumulation, different reduction order) batch path — or
    /// mis-strided the per-row sub-views — would diverge here.
    #[test]
    fn decode_gemv_loop_is_byte_identical_to_per_row_singlestream() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping looped decode GEMV byte-identity test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!(
                "skipping looped decode GEMV byte-identity test: fp16 NVRTC headers unavailable"
            );
            return;
        }

        // Qwen2.5-0.5B-ish block-32 shapes: square attention proj and the wide
        // MLP (K<N general GEMV, K>N tall-skinny GEMV). m=4 sits inside the
        // default loop window (max_m=8).
        for (k, n) in [
            (896usize, 896usize),
            (896usize, 4864usize),
            (4864usize, 896usize),
        ] {
            let m = 4usize;
            let block_size = 32usize;
            let k_blocks = k / block_size;
            let blob_size = block_size / 2;

            let mut state = 0x0123_4567_89ab_cdefu64 ^ ((k as u64) << 20) ^ (n as u64);
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            // m distinct activation rows, contiguous [m, k].
            let mut activation_f16 = vec![f16::ZERO; m * k];
            for h in activation_f16.iter_mut() {
                *h = f16::from_f32(next());
            }

            // Symmetric int4 weights (implicit zp=8), block-32, no bias.
            let mut quant = vec![0u8; n * k];
            for value in quant.iter_mut() {
                *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
            }
            let mut packed = vec![0u8; n * k_blocks * blob_size];
            for col in 0..n {
                for block in 0..k_blocks {
                    for pair in 0..blob_size {
                        let low = quant[col * k + block * block_size + pair * 2] & 15;
                        let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                        packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                    }
                }
            }
            let mut scale_f16 = vec![f16::ZERO; n * k_blocks];
            for s in scale_f16.iter_mut() {
                *s = f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5));
            }

            let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
            let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
            let scales_dev = runtime.alloc_raw(scale_f16.len() * 2).unwrap();
            let loop_out_dev = runtime.alloc_raw(m * n * 2).unwrap();
            let ref_out_dev = runtime.alloc_raw(n * 2).unwrap();
            // SAFETY: device buffers were sized to hold each source slice.
            unsafe {
                runtime
                    .htod(as_bytes(&activation_f16), activation_dev)
                    .unwrap();
                runtime.htod(&packed, packed_dev).unwrap();
                runtime.htod(as_bytes(&scale_f16), scales_dev).unwrap();
            }

            let device = DeviceId::cuda(0);
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let scales_shape = [n, k_blocks];
            let scales_strides = [k_blocks as i64, 1];
            let packed_view = TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_view = TensorView::new(
                device_ptr(scales_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );

            let kernel = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: false,
                decomposed_silu: false,
                rmsnorm_prologue: false,
                rmsnorm_epsilon: 1e-5,
                last_call_capture_safe: AtomicBool::new(false),
                bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
                bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
            };

            // Batched path: m>1 routes through the looped single-row decode GEMV.
            let a_shape_m = [m, k];
            let a_strides_m = [k as i64, 1];
            let y_shape_m = [m, n];
            let y_strides_m = [n as i64, 1];
            let inputs_m = vec![
                TensorView::new(
                    device_ptr(activation_dev),
                    DataType::Float16,
                    &a_shape_m,
                    &a_strides_m,
                    device,
                ),
                packed_view.clone(),
                scales_view.clone(),
            ];
            {
                let mut outputs_m = [TensorMut::new(
                    device_ptr_mut(loop_out_dev),
                    DataType::Float16,
                    &y_shape_m,
                    &y_strides_m,
                    device,
                )];
                // Warm once so scratch pools are allocated, then the measured run
                // must report capture-safe (mirrors the M==1 decode contract).
                kernel.run(&inputs_m, &mut outputs_m, None).unwrap();
                kernel.run(&inputs_m, &mut outputs_m, None).unwrap();
                runtime.synchronize().unwrap();
            }
            assert!(
                kernel.last_call_capture_safe.load(Ordering::Relaxed),
                "warm looped decode GEMV must report capture-safe (K={k} N={n})"
            );
            let mut loop_out = vec![f16::ZERO; m * n];
            // SAFETY: `loop_out_dev` holds `m * n` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut loop_out), loop_out_dev)
                    .unwrap();
            }

            // Reference: run each row alone as an independent M==1 GEMV over the
            // same weights and the same (offset) activation row.
            let mut ref_out = vec![f16::ZERO; m * n];
            for r in 0..m {
                let a_off = (r * k * 2) as CUdeviceptr;
                let a_shape_1 = [1usize, k];
                let a_strides_1 = [k as i64, 1];
                let y_shape_1 = [1usize, n];
                let y_strides_1 = [n as i64, 1];
                let inputs_1 = vec![
                    TensorView::new(
                        device_ptr(activation_dev + a_off),
                        DataType::Float16,
                        &a_shape_1,
                        &a_strides_1,
                        device,
                    ),
                    packed_view.clone(),
                    scales_view.clone(),
                ];
                let mut outputs_1 = [TensorMut::new(
                    device_ptr_mut(ref_out_dev),
                    DataType::Float16,
                    &y_shape_1,
                    &y_strides_1,
                    device,
                )];
                kernel.run(&inputs_1, &mut outputs_1, None).unwrap();
                runtime.synchronize().unwrap();
                // SAFETY: `ref_out_dev` holds `n` fp16 values.
                unsafe {
                    runtime
                        .dtoh(as_bytes_mut(&mut ref_out[r * n..(r + 1) * n]), ref_out_dev)
                        .unwrap();
                }
            }

            // SAFETY: each pointer came from this runtime's `alloc_raw`, freed once.
            unsafe {
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(packed_dev).unwrap();
                runtime.free_raw(scales_dev).unwrap();
                runtime.free_raw(loop_out_dev).unwrap();
                runtime.free_raw(ref_out_dev).unwrap();
            }

            let mismatches = loop_out
                .iter()
                .zip(ref_out.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                mismatches,
                0,
                "looped batch-{m} decode GEMV diverged from per-row single-stream decode at \
                 K={k} N={n}: {mismatches}/{} fp16 outputs differ",
                m * n
            );
        }
    }

    /// Multi-request batch decode byte-identity guard for the **fused gate/up
    /// SwiGLU with an RMS-norm prologue** — the node whose small-batch decode is
    /// routed through the per-row capture-safe loop by this fix. For
    /// `1 < m <= decode_gemv_loop_max_m()` and a present gamma, the production
    /// path (`run_f16_gate_up_swiglu`) dispatches `launch_gate_up_swiglu_rmsnorm`
    /// once per row over a 1×K sub-view, which is the *identical* kernel an M==1
    /// rmsnorm decode step runs. Therefore row `r` of the batched output must be
    /// BIT-for-BIT equal to running that row alone as an M==1 fused rmsnorm
    /// decode. This is THE batching-contract invariant this change must protect:
    /// a request batched with others produces the same tokens as if it had run
    /// alone. (The two-op ULP sweeps bound accuracy vs a different reference;
    /// this asserts exact single-stream equivalence.) A regression that routed
    /// M>1 through the tiled prefill GEMM / Marlin (different reduction order), or
    /// mis-strided the per-row sub-views, would diverge here.
    #[test]
    fn gate_up_swiglu_rmsnorm_loop_is_byte_identical_to_per_row_singlestream() {
        let Some(runtime) = runtime() else {
            eprintln!(
                "skipping rmsnorm gate/up SwiGLU byte-identity test: CUDA runtime unavailable"
            );
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!(
                "skipping rmsnorm gate/up SwiGLU byte-identity test: fp16 NVRTC headers unavailable"
            );
            return;
        }

        // Qwen2.5-0.5B-ish gate/up shapes (K<N wide MLP and a square case).
        // m=4 sits inside the default per-row loop window (max_m=8).
        for (k, n) in [(896usize, 4864usize), (896usize, 896usize)] {
            let m = 4usize;
            let block_size = 32usize;
            let k_blocks = k / block_size;
            let blob_size = block_size / 2;

            let mut state = 0x51ed_2701_c0ff_ee11u64 ^ ((k as u64) << 20) ^ (n as u64);
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            // m distinct activation rows, contiguous [m, k].
            let mut activation_f16 = vec![f16::ZERO; m * k];
            for h in activation_f16.iter_mut() {
                *h = f16::from_f32(next());
            }
            // fp16 gamma, length k.
            let mut gamma_f16 = vec![f16::ZERO; k];
            for g in gamma_f16.iter_mut() {
                *g = f16::from_f32(0.75 + 0.5 * (next() * 0.5 + 0.5));
            }

            // Symmetric int4 gate and up weights (implicit zp=8), block-32.
            let pack = |next: &mut dyn FnMut() -> f32| {
                let mut quant = vec![0u8; n * k];
                for value in quant.iter_mut() {
                    *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
                }
                let mut packed = vec![0u8; n * k_blocks * blob_size];
                for col in 0..n {
                    for block in 0..k_blocks {
                        for pair in 0..blob_size {
                            let low = quant[col * k + block * block_size + pair * 2] & 15;
                            let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                            packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                        }
                    }
                }
                packed
            };
            let packed_gate = pack(&mut next);
            let packed_up = pack(&mut next);
            let mut scales_gate = vec![f16::ZERO; n * k_blocks];
            for s in scales_gate.iter_mut() {
                *s = f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5));
            }
            let mut scales_up = vec![f16::ZERO; n * k_blocks];
            for s in scales_up.iter_mut() {
                *s = f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5));
            }

            let activation_dev = runtime.alloc_raw(activation_f16.len() * 2).unwrap();
            let gamma_dev = runtime.alloc_raw(gamma_f16.len() * 2).unwrap();
            let packed_gate_dev = runtime.alloc_raw(packed_gate.len()).unwrap();
            let packed_up_dev = runtime.alloc_raw(packed_up.len()).unwrap();
            let scales_gate_dev = runtime.alloc_raw(scales_gate.len() * 2).unwrap();
            let scales_up_dev = runtime.alloc_raw(scales_up.len() * 2).unwrap();
            let loop_out_dev = runtime.alloc_raw(m * n * 2).unwrap();
            let ref_out_dev = runtime.alloc_raw(n * 2).unwrap();
            // SAFETY: device buffers were sized to hold each source slice.
            unsafe {
                runtime
                    .htod(as_bytes(&activation_f16), activation_dev)
                    .unwrap();
                runtime.htod(as_bytes(&gamma_f16), gamma_dev).unwrap();
                runtime.htod(&packed_gate, packed_gate_dev).unwrap();
                runtime.htod(&packed_up, packed_up_dev).unwrap();
                runtime
                    .htod(as_bytes(&scales_gate), scales_gate_dev)
                    .unwrap();
                runtime.htod(as_bytes(&scales_up), scales_up_dev).unwrap();
            }

            let device = DeviceId::cuda(0);
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let scales_shape = [n, k_blocks];
            let scales_strides = [k_blocks as i64, 1];
            let packed_gate_view = TensorView::new(
                device_ptr(packed_gate_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let packed_up_view = TensorView::new(
                device_ptr(packed_up_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_gate_view = TensorView::new(
                device_ptr(scales_gate_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let scales_up_view = TensorView::new(
                device_ptr(scales_up_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let gamma_shape = [k];
            let gamma_strides = [1i64];
            let gamma_view = TensorView::new(
                device_ptr(gamma_dev),
                DataType::Float16,
                &gamma_shape,
                &gamma_strides,
                device,
            );

            let kernel = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: true,
                decomposed_silu: false,
                rmsnorm_prologue: true,
                rmsnorm_epsilon: 1e-5,
                last_call_capture_safe: AtomicBool::new(false),
                bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
                bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
            };

            // Batched path: m>1 with gamma present routes through the looped
            // single-row fused rmsnorm decode GEMV.
            let a_shape_m = [m, k];
            let a_strides_m = [k as i64, 1];
            let y_shape_m = [m, n];
            let y_strides_m = [n as i64, 1];
            let inputs_m = [
                TensorView::new(
                    device_ptr(activation_dev),
                    DataType::Float16,
                    &a_shape_m,
                    &a_strides_m,
                    device,
                ),
                packed_gate_view,
                scales_gate_view,
                packed_up_view,
                scales_up_view,
                gamma_view,
            ];
            {
                let mut outputs_m = [TensorMut::new(
                    device_ptr_mut(loop_out_dev),
                    DataType::Float16,
                    &y_shape_m,
                    &y_strides_m,
                    device,
                )];
                kernel
                    .run_f16_gate_up_swiglu(&inputs_m, &mut outputs_m, None)
                    .unwrap();
                runtime.synchronize().unwrap();
            }
            assert!(
                kernel.last_call_capture_safe.load(Ordering::Relaxed),
                "looped rmsnorm gate/up SwiGLU decode must report capture-safe (K={k} N={n})"
            );
            let mut loop_out = vec![f16::ZERO; m * n];
            // SAFETY: `loop_out_dev` holds `m * n` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut loop_out), loop_out_dev)
                    .unwrap();
            }

            // Reference: run each row alone as an independent M==1 fused rmsnorm
            // gate/up SwiGLU decode over the same weights, gamma, and the same
            // (offset) activation row.
            let mut ref_out = vec![f16::ZERO; m * n];
            for r in 0..m {
                let a_off = (r * k * 2) as CUdeviceptr;
                let a_shape_1 = [1usize, k];
                let a_strides_1 = [k as i64, 1];
                let y_shape_1 = [1usize, n];
                let y_strides_1 = [n as i64, 1];
                let inputs_1 = [
                    TensorView::new(
                        device_ptr(activation_dev + a_off),
                        DataType::Float16,
                        &a_shape_1,
                        &a_strides_1,
                        device,
                    ),
                    packed_gate_view,
                    scales_gate_view,
                    packed_up_view,
                    scales_up_view,
                    gamma_view,
                ];
                let mut outputs_1 = [TensorMut::new(
                    device_ptr_mut(ref_out_dev),
                    DataType::Float16,
                    &y_shape_1,
                    &y_strides_1,
                    device,
                )];
                kernel
                    .run_f16_gate_up_swiglu(&inputs_1, &mut outputs_1, None)
                    .unwrap();
                runtime.synchronize().unwrap();
                // SAFETY: `ref_out_dev` holds `n` fp16 values.
                unsafe {
                    runtime
                        .dtoh(as_bytes_mut(&mut ref_out[r * n..(r + 1) * n]), ref_out_dev)
                        .unwrap();
                }
            }

            // SAFETY: each pointer came from this runtime's `alloc_raw`, freed once.
            unsafe {
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(gamma_dev).unwrap();
                runtime.free_raw(packed_gate_dev).unwrap();
                runtime.free_raw(packed_up_dev).unwrap();
                runtime.free_raw(scales_gate_dev).unwrap();
                runtime.free_raw(scales_up_dev).unwrap();
                runtime.free_raw(loop_out_dev).unwrap();
                runtime.free_raw(ref_out_dev).unwrap();
            }

            let mismatches = loop_out
                .iter()
                .zip(ref_out.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                mismatches,
                0,
                "looped batch-{m} rmsnorm gate/up SwiGLU decode diverged from per-row \
                 single-stream decode at K={k} N={n}: {mismatches}/{} fp16 outputs differ",
                m * n
            );
        }
    }

    /// no other Qwen2.5 size hits: gate/up is K=1536,N=8960 (K<N → *general*
    /// GEMV) and the down-projection is K=8960,N=1536 (K>N → *tall-skinny*
    /// specialized GEMV). K=1536 is 48 block-32 groups and N=8960 is a whole
    /// multiple of the 8-column CTA width, exercising the block-count and column
    /// tiling boundaries at the exact production shapes. Both variants must track
    /// the f64 dequant-and-matmul oracle within the fp16 accumulation floor so a
    /// future kernel change cannot silently reintroduce a decode-step logit
    /// divergence at these dims.
    #[test]
    fn fp16_gemv_matches_dequant_reference_qwen_1_5b_dims() {
        // (k, n) → (gate/up general GEMV, down-projection tall-skinny GEMV).
        for (k, n) in [(1536usize, 8960usize), (8960usize, 1536usize)] {
            let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
                (0.0f32, 0.0f32, 0.0f32, true);
            for (scales_fp16, with_bias) in [(false, false), (true, false), (true, true)] {
                let (abs, rel, out, finite) = run_parity_dims(k, n, scales_fp16, with_bias, false);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                max_out = max_out.max(out);
                all_finite &= finite;
            }
            // Same fp16-ULP-scaled magnitude bound as the general parity test.
            // K here runs deeper (up to 8960), but the observed error stays
            // within that floor, so no extra slack is needed.
            let abs_bound = (max_out * 1e-3).max(1e-3);
            eprintln!(
                "MatMulNBits fp16 GEMV parity K={k} N={n}: max_abs={worst_abs:.3e} \
                 max_rel={worst_rel:.3e} max_out={max_out:.3e} abs_bound={abs_bound:.3e}"
            );
            assert!(
                all_finite,
                "fp16 GEMV produced a non-finite output (K={k} N={n})"
            );
            assert!(
                worst_abs < abs_bound,
                "fp16 GEMV diverged from dequant reference at K={k} N={n}: \
                 max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
            );
            assert!(
                worst_rel < 5e-2,
                "fp16 GEMV diverged from dequant reference at K={k} N={n}: max_rel={worst_rel:.3e}"
            );
        }
    }

    /// Model-agnostic block-size guard: the general-block-size fp16 decode GEMV
    /// (`matmul_nbits_gemv_f16_general_bs`, selected for any `block_size != 32`)
    /// must track the same f64 dequant-and-matmul oracle as the tuned block-32
    /// path. Exercised at `block_size = 128` — the Qwen2.5-0.5B **v4-bs128**
    /// foundry package's layout that previously failed to load — across the
    /// exact q/k/v/o/gate/up/down projection dims (K=896, and the wide MLP
    /// K=4864), a ragged-N tail, and both fp16/fp32 scales with and without a
    /// folded bias. It also drives an explicit **asymmetric** per-block int4
    /// zero-point tensor so a zero-point indexing regression in the general
    /// kernel's dequant path is caught (plausible for zp-bearing non-32-block
    /// models). A regression in the general block-index math (scale/zp stride,
    /// K-stepping, or nibble unpack) would diverge here.
    #[test]
    fn fp16_gemv_matches_dequant_reference_block128() {
        // (block_size, k, n): block-128 covers the Qwen2.5-0.5B bs128 attention
        // and MLP projection shapes (K=896, wide MLP K=4864) plus a ragged N
        // (70) spanning several 8-column CTAs with a partial tail; block-64
        // proves the block index math generalizes beyond a single width. All K
        // are whole multiples of their block size.
        for (block_size, k, n) in [
            (128usize, 896usize, 896usize),
            (128usize, 896usize, 4864usize),
            (128usize, 4864usize, 896usize),
            (128usize, 896usize, 70usize),
            (64usize, 896usize, 896usize),
            (64usize, 896usize, 70usize),
        ] {
            // Any block_size != 32 must route through the general kernel.
            assert_eq!(
                select_f16_gemv_variant(k, n, block_size, true, false).variant,
                F16GemvVariant::General,
                "block_size={block_size} K={k} N={n} must select the general variant"
            );
            let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
                (0.0f32, 0.0f32, 0.0f32, true);
            // (scales_fp16, with_bias, explicit_zp): the last two rows exercise
            // the general kernel's asymmetric int4 zero-point dequant path.
            for (scales_fp16, with_bias, explicit_zp) in [
                (false, false, false),
                (true, false, false),
                (true, true, false),
                (false, false, true),
                (true, true, true),
            ] {
                let (abs, rel, out, finite) =
                    run_parity_dims_block(k, n, block_size, scales_fp16, with_bias, explicit_zp);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                max_out = max_out.max(out);
                all_finite &= finite;
            }
            // Same fp16-ULP-scaled magnitude bound as the block-32 parity tests;
            // a wider block shares one scale across more K-elements but the oracle
            // uses the identical dequant, so the accumulation floor is unchanged.
            let abs_bound = (max_out * 1e-3).max(1e-3);
            eprintln!(
                "MatMulNBits fp16 GEMV block-{block_size} parity K={k} N={n}: \
                 max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} max_out={max_out:.3e} \
                 abs_bound={abs_bound:.3e}"
            );
            assert!(
                all_finite,
                "block-{block_size} fp16 GEMV produced a non-finite output (K={k} N={n})"
            );
            assert!(
                worst_abs < abs_bound,
                "block-{block_size} fp16 GEMV diverged from dequant reference at K={k} N={n}: \
                 max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
            );
            assert!(
                worst_rel < 5e-2,
                "block-{block_size} fp16 GEMV diverged from dequant reference at K={k} N={n}: \
                 max_rel={worst_rel:.3e}"
            );
        }
    }

    /// Asymmetric-zero-point int4 fp16 decode GEMV at Phi-4-mini's int4 dims
    /// must track an f64 dequant oracle that honors the per-block zero point.
    /// Phi carries explicit zero points on every MatMulNBits, so the vectorized
    /// `scales_f16` GEMV that decode routes to (o_proj K=3072,N=3072 and the
    /// gate/up projection K=3072,N=8192) must dequantize `(code - zp) * scale`,
    /// not the symmetric `(code - 8)`. This is the mutation guard for the shared
    /// `int4x8_to_half2x4_sub` primitive: a kernel that ignored the zero point
    /// (subtracting the implicit 8) would diverge from the oracle far beyond the
    /// fp16 floor and fail here.
    #[test]
    fn fp16_gemv_matches_dequant_reference_phi_int4_zp_dims() {
        // (K, N): Phi o_proj (K==N general GEMV) and gate/up (K<N general GEMV).
        for (k, n) in [(3072usize, 3072usize), (3072, 8192)] {
            let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
                (0.0f32, 0.0f32, 0.0f32, true);
            // Exercise both the plain asymmetric GEMV and the folded-bias
            // (residual) epilogue that the skip-rmsnorm fusion produces on the
            // preceding projection. Routed through `run_parity_dims` (block-32
            // default) so the `explicit_zp` delegation is actually driven.
            for (with_bias, explicit_zp) in [(false, true), (true, true)] {
                let (abs, rel, out, finite) = run_parity_dims(k, n, true, with_bias, explicit_zp);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                max_out = max_out.max(out);
                all_finite &= finite;
            }
            let abs_bound = (max_out * 1e-3).max(1e-3);
            eprintln!(
                "MatMulNBits int4 asymmetric-zp GEMV parity K={k} N={n}: max_abs={worst_abs:.3e} \
                 max_rel={worst_rel:.3e} max_out={max_out:.3e} abs_bound={abs_bound:.3e}"
            );
            assert!(
                all_finite,
                "int4 asymmetric-zp GEMV produced a non-finite output (K={k} N={n})"
            );
            assert!(
                worst_abs < abs_bound,
                "int4 asymmetric-zp GEMV diverged from dequant reference at K={k} N={n}: \
                 max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
            );
            assert!(
                worst_rel < 5e-2,
                "int4 asymmetric-zp GEMV diverged from dequant reference at K={k} N={n}: \
                 max_rel={worst_rel:.3e}"
            );
        }
    }

    /// Int8 (bits=8) fp16 decode GEMV must track an f64 dequant oracle at Phi's
    /// GEMV dims — the shapes ORT beat us on. QKV (K=3072, N=5120), down
    /// projection (K=8192, N=3072), and the lm_head slice (K=3072, wide N) all
    /// exercise the vectorised four-lane/eight-block path; a ragged N tail and
    /// an explicit-zero-point case guard the reduction and dequant edges.
    #[test]
    fn int8_fp16_gemv_matches_dequant_reference_phi_dims() {
        let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
            (0.0f32, 0.0f32, 0.0f32, true);
        // (K, N, scales_fp16, with_bias, explicit_zp)
        let cases = [
            (3072usize, 5120usize, true, false, false), // Phi QKV int8
            (8192, 3072, true, false, false),           // Phi down projection int8
            (3072, 5120, true, true, false),            // QKV with folded bias
            (3072, 5120, true, false, true),            // explicit per-block zero points
            (8192, 3072, true, false, true),            // down-proj zp: exercises split-K at K=8192
            (3072, 5121, true, false, false),           // ragged N tail (not warp-tile aligned)
            (8192, 3072, false, false, false),          // fp32 scales
        ];
        for (k, n, scales_fp16, with_bias, explicit_zp) in cases {
            let (abs, rel, out, finite) =
                run_int8_parity_dims(k, n, 32, scales_fp16, with_bias, explicit_zp);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
            max_out = max_out.max(out);
            all_finite &= finite;
        }
        // int8 quant codes span 0..255 so accumulated magnitudes (and thus the
        // fp16 output ULP floor) are larger than the int4 case; keep the same
        // magnitude-relative bound shape with headroom for K up to 8192.
        let abs_bound = (max_out * 2e-3).max(1e-3);
        eprintln!(
            "MatMulNBits int8 fp16 GEMV parity: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} \
             max_out={max_out:.3e} abs_bound={abs_bound:.3e}"
        );
        assert!(all_finite, "int8 fp16 GEMV produced a non-finite output");
        assert!(
            worst_abs < abs_bound,
            "int8 fp16 GEMV diverged from dequant reference: max_abs={worst_abs:.3e} \
             bound={abs_bound:.3e}"
        );
        assert!(
            worst_rel < 5e-2,
            "int8 fp16 GEMV diverged from dequant reference: max_rel={worst_rel:.3e}"
        );
    }

    /// Int8 (bits=8) fp16 decode GEMV at non-block-32 power-of-two block widths
    /// must route through the model-agnostic general-block-size kernel and track
    /// an f64 dequant oracle. This is the coverage guard for the Foundry
    /// Qwen3-0.6B artifact, whose 105 `MatMulNBits(bits=8, block_size=128)` nodes
    /// (e.g. the wide MLP down projection K=3072, and QKV/attention shapes) the
    /// old factory rejected — forcing the whole session onto the CPU EP. The
    /// block-64 rows prove the block-index math generalizes beyond a single
    /// width, and the ragged-N / explicit-zero-point rows guard the reduction
    /// tail and the asymmetric uint8 zero-point dequant.
    #[test]
    fn int8_fp16_gemv_matches_dequant_reference_block128() {
        // (block_size, k, n): all K are whole multiples of their block size.
        for (block_size, k, n) in [
            (128usize, 1024usize, 1024usize), // Qwen3-style square projection
            (128, 3072, 1024),                // tall-skinny down projection (K > N)
            (128, 1024, 3072),                // wide projection (N > K)
            (128, 1024, 1030),                // ragged N tail across CTAs
            (64, 1024, 1024),                 // block-64 generalization
        ] {
            // Any block_size != 32 must select the general variant.
            assert_eq!(
                select_f16_gemv_variant(k, n, block_size, true, false).variant,
                F16GemvVariant::General,
                "int8 block_size={block_size} K={k} N={n} must select the general variant"
            );
            let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
                (0.0f32, 0.0f32, 0.0f32, true);
            // (scales_fp16, with_bias, explicit_zp): cover fp16/fp32 scales, the
            // folded-bias epilogue, and the asymmetric uint8 zero-point path.
            for (scales_fp16, with_bias, explicit_zp) in [
                (true, false, false),
                (false, false, false),
                (true, true, false),
                (true, false, true),
            ] {
                let (abs, rel, out, finite) =
                    run_int8_parity_dims(k, n, block_size, scales_fp16, with_bias, explicit_zp);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                max_out = max_out.max(out);
                all_finite &= finite;
            }
            let abs_bound = (max_out * 2e-3).max(1e-3);
            eprintln!(
                "MatMulNBits int8 fp16 GEMV block-{block_size} parity K={k} N={n}: \
                 max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} max_out={max_out:.3e} \
                 abs_bound={abs_bound:.3e}"
            );
            assert!(
                all_finite,
                "int8 block-{block_size} fp16 GEMV produced a non-finite output (K={k} N={n})"
            );
            assert!(
                worst_abs < abs_bound,
                "int8 block-{block_size} fp16 GEMV diverged from dequant reference at K={k} \
                 N={n}: max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
            );
            assert!(
                worst_rel < 5e-2,
                "int8 block-{block_size} fp16 GEMV diverged from dequant reference at K={k} \
                 N={n}: max_rel={worst_rel:.3e}"
            );
        }
    }

    /// Int8 (bits=8) **fp32-activation** decode GEMV harness. The Foundry
    /// Qwen3-0.6B artifact emits fp32 activations, so its 105
    /// `MatMulNBits(bits=8, block_size=128)` decode nodes take the fp32 `run`
    /// path (`launch_f32_gemv` → `matmul_nbits_gemv_f32`), not the fp16
    /// `run_f16` path. This exercises the general-block-size fp32 kernel branch
    /// against an f64 dequant oracle and asserts the launch stays capture-safe.
    fn run_int8_f32_parity_dims(
        k: usize,
        n: usize,
        block_size: usize,
        with_bias: bool,
        explicit_zp: bool,
    ) -> (f32, f32, f32, bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping MatMulNBits int8 f32 GEMV parity test: CUDA runtime unavailable");
            return (0.0, 0.0, 0.0, true);
        };

        let k_blocks = k / block_size;
        let blob_size = block_size; // one byte per weight for bits=8

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation = vec![0.0f32; k];
        for value in activation.iter_mut() {
            *value = next();
        }

        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for within in 0..block_size {
                    packed[(col * k_blocks + block) * blob_size + within] =
                        quant[col * k + block * block_size + within];
                }
            }
        }

        let mut zero_points = vec![0u8; n * k_blocks];
        if explicit_zp {
            for zp in zero_points.iter_mut() {
                *zp = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        let zp_ref = |col: usize, block: usize| -> i32 {
            if explicit_zp {
                zero_points[col * k_blocks + block] as i32
            } else {
                128
            }
        };

        let mut scale = vec![0.0f32; n * k_blocks];
        for value in scale.iter_mut() {
            *value = 0.015 + 0.01 * (next() * 0.5 + 0.5);
        }

        let mut bias = vec![0.0f32; n];
        if with_bias {
            for value in bias.iter_mut() {
                *value = next();
            }
        }

        let mut expected = vec![0.0f32; n];
        for col in 0..n {
            let mut acc = 0.0f64;
            for block in 0..k_blocks {
                let s = scale[col * k_blocks + block] as f64;
                let zp = zp_ref(col, block);
                for within in 0..block_size {
                    let depth = block * block_size + within;
                    let q = quant[col * k + depth] as i32 - zp;
                    acc += activation[depth] as f64 * q as f64 * s;
                }
            }
            if with_bias {
                acc += bias[col] as f64;
            }
            expected[col] = acc as f32;
        }

        let activation_dev = runtime.alloc_raw(activation.len() * 4).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scale.len() * 4).unwrap();
        let zp_dev = runtime.alloc_raw(zero_points.len().max(1)).unwrap();
        let bias_dev = runtime.alloc_raw(n * 4).unwrap();
        let output_dev = runtime.alloc_raw(n * 4).unwrap();

        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime.htod(as_bytes(&activation), activation_dev).unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scale), scales_dev).unwrap();
            if explicit_zp {
                runtime.htod(&zero_points, zp_dev).unwrap();
            }
            if with_bias {
                runtime.htod(as_bytes(&bias), bias_dev).unwrap();
            }
        }

        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, k_blocks];
        let zp_strides = [k_blocks as i64, 1];
        let bias_shape = [n];
        let bias_strides = [1i64];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];

        let device = DeviceId::cuda(0);
        let mut inputs = vec![
            TensorView::new(
                device_ptr(activation_dev),
                DataType::Float32,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                DataType::Float32,
                &scales_shape,
                &scales_strides,
                device,
            ),
        ];
        if explicit_zp || with_bias {
            inputs.push(if explicit_zp {
                TensorView::new(
                    device_ptr(zp_dev),
                    DataType::Uint8,
                    &zp_shape,
                    &zp_strides,
                    device,
                )
            } else {
                TensorView::absent(DataType::Uint8)
            });
        }
        if with_bias {
            inputs.push(TensorView::absent(DataType::Int32));
            inputs.push(TensorView::new(
                device_ptr(bias_dev),
                DataType::Float32,
                &bias_shape,
                &bias_strides,
                device,
            ));
        }

        let mut outputs = [TensorMut::new(
            device_ptr_mut(output_dev),
            DataType::Float32,
            &y_shape,
            &y_strides,
            device,
        )];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 8,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };
        kernel.run(&inputs, &mut outputs, None).unwrap();
        runtime.synchronize().unwrap();

        assert!(
            kernel.last_call_capture_safe.load(Ordering::Relaxed),
            "int8 f32 decode GEMV must report capture-safe"
        );

        let mut got = vec![0.0f32; n];
        // SAFETY: `output_dev` holds `n` fp32 values.
        unsafe {
            runtime.dtoh(as_bytes_mut(&mut got), output_dev).unwrap();
        }

        // SAFETY: each pointer came from this runtime's `alloc_raw` and is freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(bias_dev).unwrap();
            runtime.free_raw(output_dev).unwrap();
        }

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut max_out = 0.0f32;
        let mut all_finite = true;
        for (g, e) in got.iter().zip(expected.iter()) {
            if !g.is_finite() {
                all_finite = false;
            }
            let abs = (g - e).abs();
            let rel = abs / e.abs().max(1e-1);
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
            max_out = max_out.max(e.abs());
        }
        (worst_abs, worst_rel, max_out, all_finite)
    }

    /// The Foundry Qwen3-0.6B fp32-activation int8/block-128 decode GEMV must
    /// track an f64 dequant oracle through the general-block-size fp32 kernel.
    /// This is the direct unit guard for the path the real artifact takes at
    /// decode (fp32 activations, `bits=8`, `block_size=128`).
    #[test]
    fn int8_f32_gemv_matches_dequant_reference_block128() {
        for (block_size, k, n) in [
            (128usize, 1024usize, 1024usize), // Qwen3-style square projection
            (128, 3072, 1024),                // tall-skinny down projection (K > N)
            (128, 1024, 3072),                // wide projection (N > K)
            (128, 1024, 1030),                // ragged N tail across CTAs
            (64, 1024, 1024),                 // block-64 generalization
        ] {
            let (mut worst_abs, mut worst_rel, mut max_out, mut all_finite) =
                (0.0f32, 0.0f32, 0.0f32, true);
            for (with_bias, explicit_zp) in [(false, false), (true, false), (false, true)] {
                let (abs, rel, out, finite) =
                    run_int8_f32_parity_dims(k, n, block_size, with_bias, explicit_zp);
                worst_abs = worst_abs.max(abs);
                worst_rel = worst_rel.max(rel);
                max_out = max_out.max(out);
                all_finite &= finite;
            }
            let abs_bound = (max_out * 2e-3).max(1e-3);
            eprintln!(
                "MatMulNBits int8 f32 GEMV block-{block_size} parity K={k} N={n}: \
                 max_abs={worst_abs:.3e} max_rel={worst_rel:.3e} max_out={max_out:.3e} \
                 abs_bound={abs_bound:.3e}"
            );
            assert!(
                all_finite,
                "int8 block-{block_size} f32 GEMV produced a non-finite output (K={k} N={n})"
            );
            assert!(
                worst_abs < abs_bound,
                "int8 block-{block_size} f32 GEMV diverged from dequant reference at K={k} \
                 N={n}: max_abs={worst_abs:.3e} bound={abs_bound:.3e}"
            );
            assert!(
                worst_rel < 5e-2,
                "int8 block-{block_size} f32 GEMV diverged from dequant reference at K={k} \
                 N={n}: max_rel={worst_rel:.3e}"
            );
        }
    }

    /// Runs a structurally-selected block-128 specialization and the generic
    /// `matmul_nbits_gemv_f32` kernel on identical device buffers and
    /// returns `(mismatching_columns, all_finite)`. The specialization only
    /// rewrites address arithmetic (shift instead of divide/modulo, hoisted
    /// per-column base, dropped bit-width branch) — the per-thread depth stride,
    /// per-element fp32 expression, and block reduction are unchanged, so every
    /// output column must be **bit-for-bit identical**. Includes partial final
    /// K-blocks (`k` not a multiple of 128) to exercise the padded weight row.
    fn run_f32_block128_byte_identity(
        bits: usize,
        k: usize,
        n: usize,
        with_bias: bool,
        explicit_zp: bool,
    ) -> (usize, bool) {
        let Some(runtime) = runtime() else {
            eprintln!(
                "skipping MatMulNBits int{bits} f32 block-128 byte-identity test: CUDA runtime unavailable"
            );
            return (0, true);
        };

        let block_size = 128usize;
        let k_blocks = k.div_ceil(block_size);
        let blob_size = block_size * bits / 8;
        let zp_row_bytes = (k_blocks * bits).div_ceil(8);

        let mut state = 0x0bad_c0de_1234_5678u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let mut activation = vec![0.0f32; k];
        for value in activation.iter_mut() {
            *value = next();
        }

        // Packed weights are stored per padded K-block row (`k_blocks * blob_size`
        // bytes per column); only depths `< k` are ever read.
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for value in packed.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
        }

        let mut zero_points = vec![0u8; n * zp_row_bytes];
        if explicit_zp {
            for zp in zero_points.iter_mut() {
                *zp = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }

        let mut scale = vec![0.0f32; n * k_blocks];
        for value in scale.iter_mut() {
            *value = 0.015 + 0.01 * (next() * 0.5 + 0.5);
        }

        let mut bias = vec![0.0f32; n];
        if with_bias {
            for value in bias.iter_mut() {
                *value = next();
            }
        }

        let activation_dev = runtime.alloc_raw(activation.len() * 4).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scale.len() * 4).unwrap();
        let zp_dev = runtime.alloc_raw(zero_points.len().max(1)).unwrap();
        let bias_dev = runtime.alloc_raw(n * 4).unwrap();
        let spec_dev = runtime.alloc_raw(n * 4).unwrap();
        let generic_dev = runtime.alloc_raw(n * 4).unwrap();

        // SAFETY: device buffers were sized to hold each source slice.
        unsafe {
            runtime.htod(as_bytes(&activation), activation_dev).unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scale), scales_dev).unwrap();
            if explicit_zp {
                runtime.htod(&zero_points, zp_dev).unwrap();
            }
            if with_bias {
                runtime.htod(as_bytes(&bias), bias_dev).unwrap();
            }
        }

        let device = DeviceId::cuda(0);
        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, zp_row_bytes];
        let zp_strides = [zp_row_bytes as i64, 1];
        let bias_shape = [n];
        let bias_strides = [1i64];

        let activation_view = TensorView::new(
            device_ptr(activation_dev),
            DataType::Float32,
            &a_shape,
            &a_strides,
            device,
        );
        let packed_view = TensorView::new(
            device_ptr(packed_dev),
            DataType::Uint8,
            &b_shape,
            &b_strides,
            device,
        );
        let scales_view = TensorView::new(
            device_ptr(scales_dev),
            DataType::Float32,
            &scales_shape,
            &scales_strides,
            device,
        );
        let zp_view = explicit_zp.then(|| {
            TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            )
        });
        let bias_view = with_bias.then(|| {
            TensorView::new(
                device_ptr(bias_dev),
                DataType::Float32,
                &bias_shape,
                &bias_strides,
                device,
            )
        });

        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];
        let mut spec_out = TensorMut::new(
            device_ptr_mut(spec_dev),
            DataType::Float32,
            &y_shape,
            &y_strides,
            device,
        );
        let mut generic_out = TensorMut::new(
            device_ptr_mut(generic_dev),
            DataType::Float32,
            &y_shape,
            &y_strides,
            device,
        );

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };

        match bits {
            4 => kernel
                .launch_int4_f32_gemv_block128(
                    &activation_view,
                    &packed_view,
                    &scales_view,
                    zp_view
                        .as_ref()
                        .expect("int4 block-128 specialization requires zero points"),
                    bias_view.as_ref(),
                    &mut spec_out,
                    k_blocks,
                )
                .unwrap(),
            8 => kernel
                .launch_int8_f32_gemv_block128(
                    &activation_view,
                    &packed_view,
                    &scales_view,
                    zp_view.as_ref(),
                    bias_view.as_ref(),
                    &mut spec_out,
                    k_blocks,
                )
                .unwrap(),
            _ => unreachable!("byte-identity harness only supports int4/int8"),
        }
        kernel
            .launch_f32_gemv(
                &activation_view,
                &packed_view,
                &scales_view,
                zp_view.as_ref(),
                bias_view.as_ref(),
                &mut generic_out,
                k_blocks,
                blob_size,
                zp_row_bytes,
            )
            .unwrap();
        runtime.synchronize().unwrap();

        let mut spec = vec![0.0f32; n];
        let mut generic = vec![0.0f32; n];
        // SAFETY: both device buffers hold `n` fp32 values.
        unsafe {
            runtime.dtoh(as_bytes_mut(&mut spec), spec_dev).unwrap();
            runtime
                .dtoh(as_bytes_mut(&mut generic), generic_dev)
                .unwrap();
        }

        // SAFETY: each pointer came from this runtime's `alloc_raw`, freed once.
        unsafe {
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(zp_dev).unwrap();
            runtime.free_raw(bias_dev).unwrap();
            runtime.free_raw(spec_dev).unwrap();
            runtime.free_raw(generic_dev).unwrap();
        }

        let mut mismatches = 0usize;
        let mut all_finite = true;
        for (s, g) in spec.iter().zip(generic.iter()) {
            if !s.is_finite() {
                all_finite = false;
            }
            if s.to_bits() != g.to_bits() {
                mismatches += 1;
            }
        }
        (mismatches, all_finite)
    }

    /// The specialized int8/block-128 asymmetric decode GEMV must be
    /// **bit-for-bit identical** to the generic `matmul_nbits_gemv_f32` it
    /// replaces for Qwen3-0.6B's dominant projection — greedy decode tokens must
    /// not shift. Covers square/tall/wide projections, ragged-N tails, and
    /// partial final K-blocks, with and without an explicit asymmetric zero
    /// point and bias.
    #[test]
    fn int8_f32_block128_specialization_is_bit_identical_to_generic() {
        for (k, n) in [
            (1024usize, 1024usize), // Qwen3-style square projection
            (3072, 1024),           // tall-skinny down projection (K > N)
            (1024, 3072),           // wide projection (N > K)
            (1024, 1030),           // ragged N tail across CTAs
            (1000, 1024),           // partial final K-block (k % 128 != 0)
            (896, 1027),            // partial N and exact K-blocks
        ] {
            for (with_bias, explicit_zp) in [(true, true), (false, true), (true, false)] {
                let (mismatches, all_finite) =
                    run_f32_block128_byte_identity(8, k, n, with_bias, explicit_zp);
                assert!(
                    all_finite,
                    "int8 block-128 specialization produced a non-finite output \
                     (K={k} N={n} bias={with_bias} zp={explicit_zp})"
                );
                assert_eq!(
                    mismatches, 0,
                    "int8 block-128 specialization diverged from the generic GEMV in \
                     {mismatches} column(s) (K={k} N={n} bias={with_bias} zp={explicit_zp}) — \
                     output must be bit-for-bit identical"
                );
            }
        }
    }

    /// The asymmetric int4/block-128 specialization must exactly match the
    /// generic fp32 GEMV, including a partial final K-block.
    #[test]
    fn int4_f32_block128_specialization_is_bit_identical_to_generic() {
        let (mismatches, all_finite) = run_f32_block128_byte_identity(4, 259, 37, true, true);
        assert!(
            all_finite,
            "int4 block-128 specialization produced non-finite output"
        );
        assert_eq!(
            mismatches, 0,
            "int4 block-128 specialization diverged from generic GEMV in \
             {mismatches} column(s)"
        );
    }

    /// Folding a standalone `Add(MatMulNBits, bias)` into the GEMV epilogue must
    /// stay **byte-identical** to the original two-op path so greedy decode
    /// tokens do not shift. The two-op path is `fp16(fp16(acc) + bias)`: the
    /// GEMV first rounds its accumulator to fp16, then the elementwise `Add`
    /// rounds again after an fp16 add. This reproduces that exactly by running
    /// the real kernel with no bias (the fp16 GEMV output) and adding the fp16
    /// bias on the host, then asserting the folded-bias kernel matches bit-for-
    /// bit across every output column.
    #[test]
    fn fp16_folded_bias_is_bit_exact_to_two_op_path() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping folded-bias bit-exactness test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping folded-bias bit-exactness test: fp16 NVRTC headers unavailable");
            return;
        }

        // QKV decode shape: K=896, N=1152, symmetric block-32, fp16 scales.
        let k = 896usize;
        let n = 1152usize;
        let block_size = 32usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let activation: Vec<f16> = (0..k).map(|_| f16::from_f32(next())).collect();
        let mut quant = vec![0u8; n * k];
        for value in quant.iter_mut() {
            *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
        }
        let mut packed = vec![0u8; n * k_blocks * blob_size];
        for col in 0..n {
            for block in 0..k_blocks {
                for pair in 0..blob_size {
                    let low = quant[col * k + block * block_size + pair * 2] & 15;
                    let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                    packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                }
            }
        }
        let scales: Vec<f16> = (0..n * k_blocks)
            .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
            .collect();
        // Bias with a wide magnitude range so the second fp16 round is exercised.
        let bias: Vec<f16> = (0..n).map(|_| f16::from_f32(next() * 4.0)).collect();

        let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scales.len() * 2).unwrap();
        let bias_dev = runtime.alloc_raw(bias.len() * 2).unwrap();
        let nobias_output_dev = runtime.alloc_raw(n * 2).unwrap();
        let fused_output_dev = runtime.alloc_raw(n * 2).unwrap();
        // SAFETY: device buffers exactly cover their source slices.
        unsafe {
            runtime.htod(as_bytes(&activation), activation_dev).unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scales), scales_dev).unwrap();
            runtime.htod(as_bytes(&bias), bias_dev).unwrap();
        }

        let device = DeviceId::cuda(0);
        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let bias_shape = [n];
        let bias_strides = [1i64];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];
        let activation_view = TensorView::new(
            device_ptr(activation_dev),
            DataType::Float16,
            &a_shape,
            &a_strides,
            device,
        );
        let packed_view = TensorView::new(
            device_ptr(packed_dev),
            DataType::Uint8,
            &b_shape,
            &b_strides,
            device,
        );
        let scales_view = TensorView::new(
            device_ptr(scales_dev),
            DataType::Float16,
            &scales_shape,
            &scales_strides,
            device,
        );
        let bias_view = TensorView::new(
            device_ptr(bias_dev),
            DataType::Float16,
            &bias_shape,
            &bias_strides,
            device,
        );
        let mut nobias_output = TensorMut::new(
            device_ptr_mut(nobias_output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );
        let mut fused_output = TensorMut::new(
            device_ptr_mut(fused_output_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );

        let selection = select_f16_gemv_variant(k, n, block_size, true, false);
        let kernel_nobias = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };
        let kernel_fold = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: true,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };
        kernel_nobias
            .launch_f16_gemv_variant(
                &activation_view,
                &packed_view,
                &scales_view,
                true,
                None,
                None,
                &mut nobias_output,
                k_blocks,
                blob_size,
                k_blocks.div_ceil(2),
                selection,
            )
            .unwrap();
        kernel_fold
            .launch_f16_gemv_variant(
                &activation_view,
                &packed_view,
                &scales_view,
                true,
                None,
                Some(&bias_view),
                &mut fused_output,
                k_blocks,
                blob_size,
                k_blocks.div_ceil(2),
                selection,
            )
            .unwrap();
        runtime.synchronize().unwrap();

        let mut gemv_out = vec![f16::ZERO; n];
        let mut fused_out = vec![f16::ZERO; n];
        // SAFETY: both output allocations hold `n` fp16 values.
        unsafe {
            runtime
                .dtoh(as_bytes_mut(&mut gemv_out), nobias_output_dev)
                .unwrap();
            runtime
                .dtoh(as_bytes_mut(&mut fused_out), fused_output_dev)
                .unwrap();
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_dev).unwrap();
            runtime.free_raw(scales_dev).unwrap();
            runtime.free_raw(bias_dev).unwrap();
            runtime.free_raw(nobias_output_dev).unwrap();
            runtime.free_raw(fused_output_dev).unwrap();
        }

        for col in 0..n {
            // Two-op reference: fp16(fp16(acc) + bias). gemv_out is already the
            // fp16-rounded accumulator, so add the fp16 bias in f32 and round.
            let two_op = f16::from_f32(gemv_out[col].to_f32() + bias[col].to_f32());
            assert_eq!(
                fused_out[col].to_bits(),
                two_op.to_bits(),
                "folded bias diverged at column {col}: fused={:?} two_op={:?} (gemv={:?} bias={:?})",
                fused_out[col],
                two_op,
                gemv_out[col],
                bias[col]
            );
        }
    }

    // Faithful replica of the elementwise `silu_mul_f16` scalar path (which is
    // byte-identical to its half2 path): `fp16(silu(f32(g)) * f32(u))`. Used to
    // build the two-op reference the paired kernel must reproduce bit-for-bit.
    const REF_SILU_MUL_SRC: &str = r#"
#include <cuda_fp16.h>
__device__ float ref_op_silu(float x) {
    if (x >= 0.0f) {
        const float denominator = __fadd_rn(1.0f, (float)exp((double)-x));
        return __fdiv_rn(x, denominator);
    }
    const float e = (float)exp((double)x);
    const float numerator = __fmul_rn(x, e);
    return __fdiv_rn(numerator, __fadd_rn(1.0f, e));
}
extern "C" __global__ void ref_silu_mul_f16(
    const __half* g, const __half* u, __half* y, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half_rn(
            __fmul_rn(ref_op_silu(__half2float(g[i])), __half2float(u[i])));
    }
}
"#;

    /// The paired gate/up SwiGLU path must be byte-identical to running two
    /// standalone MatMulNBits projections and then `silu_mul_f16`. Decode covers
    /// the Qwen shape and an unrelated shape; prefill covers the reported
    /// five-token Qwen case plus M/N tails on a small unrelated shape.
    #[test]
    fn fp16_gate_up_swiglu_is_bit_exact_to_two_op_path() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping gate/up SwiGLU bit-exactness test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping gate/up SwiGLU bit-exactness test: fp16 NVRTC headers unavailable");
            return;
        }

        // (M, K=hidden, N=intermediate): preserve both decode cases, then add
        // Qwen M=5 prefill and unrelated row/column tails.
        for (m, k, n) in [
            (1usize, QWEN_DOWN_N, QWEN_DOWN_K),
            (1, 2048, 5632),
            (5, QWEN_DOWN_N, QWEN_DOWN_K),
            (3, 96, 77),
        ] {
            let block_size = 32usize;
            let k_blocks = k / block_size;
            let blob_size = block_size / 2;

            let mut state = 0x0bad_c0de_dead_beefu64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            let pack = |next: &mut dyn FnMut() -> f32| -> Vec<u8> {
                let mut quant = vec![0u8; n * k];
                for value in quant.iter_mut() {
                    *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
                }
                let mut packed = vec![0u8; n * k_blocks * blob_size];
                for col in 0..n {
                    for block in 0..k_blocks {
                        for pair in 0..blob_size {
                            let low = quant[col * k + block * block_size + pair * 2] & 15;
                            let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                            packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                        }
                    }
                }
                packed
            };

            let activation: Vec<f16> = (0..m * k).map(|_| f16::from_f32(next())).collect();
            let packed_gate = pack(&mut next);
            let scales_gate: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            let packed_up = pack(&mut next);
            let scales_up: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();

            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let packed_gate_dev = runtime.alloc_raw(packed_gate.len()).unwrap();
            let scales_gate_dev = runtime.alloc_raw(scales_gate.len() * 2).unwrap();
            let packed_up_dev = runtime.alloc_raw(packed_up.len()).unwrap();
            let scales_up_dev = runtime.alloc_raw(scales_up.len() * 2).unwrap();
            let output_elements = m * n;
            let gate_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            let up_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            let ref_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            let fused_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            // SAFETY: device buffers exactly cover their source slices.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(&packed_gate, packed_gate_dev).unwrap();
                runtime
                    .htod(as_bytes(&scales_gate), scales_gate_dev)
                    .unwrap();
                runtime.htod(&packed_up, packed_up_dev).unwrap();
                runtime.htod(as_bytes(&scales_up), scales_up_dev).unwrap();
            }

            let device = DeviceId::cuda(0);
            let a_shape = [m, k];
            let a_strides = [k as i64, 1];
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let scales_shape = [n, k_blocks];
            let scales_strides = [k_blocks as i64, 1];
            let y_shape = [m, n];
            let y_strides = [n as i64, 1];
            let activation_view = TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            );
            let packed_gate_view = TensorView::new(
                device_ptr(packed_gate_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_gate_view = TensorView::new(
                device_ptr(scales_gate_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let packed_up_view = TensorView::new(
                device_ptr(packed_up_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_up_view = TensorView::new(
                device_ptr(scales_up_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let mut gate_out = TensorMut::new(
                device_ptr_mut(gate_out_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );
            let mut up_out = TensorMut::new(
                device_ptr_mut(up_out_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );
            let fused_out = TensorMut::new(
                device_ptr_mut(fused_out_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );

            let gemv_kernel = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: false,
                decomposed_silu: false,
                rmsnorm_prologue: false,
                rmsnorm_epsilon: 1e-5,
                last_call_capture_safe: AtomicBool::new(false),
                bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
                bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
            };
            // Reference: two standalone MatMulNBits projections.
            if m == 1 {
                let selection = select_f16_gemv_variant(k, n, block_size, true, false);
                assert_eq!(
                    selection.variant,
                    F16GemvVariant::General,
                    "gate/up decode projections must use the general GEMV as the reference"
                );
                gemv_kernel
                    .launch_f16_gemv_variant(
                        &activation_view,
                        &packed_gate_view,
                        &scales_gate_view,
                        true,
                        None,
                        None,
                        &mut gate_out,
                        k_blocks,
                        blob_size,
                        k_blocks.div_ceil(2),
                        selection,
                    )
                    .unwrap();
                gemv_kernel
                    .launch_f16_gemv_variant(
                        &activation_view,
                        &packed_up_view,
                        &scales_up_view,
                        true,
                        None,
                        None,
                        &mut up_out,
                        k_blocks,
                        blob_size,
                        k_blocks.div_ceil(2),
                        selection,
                    )
                    .unwrap();
            } else {
                gemv_kernel
                    .launch_f16_gemm(
                        &activation_view,
                        &packed_gate_view,
                        &scales_gate_view,
                        true,
                        None,
                        None,
                        &mut gate_out,
                        m,
                        k_blocks,
                        gemv_kernel.block_size * gemv_kernel.bits / 8,
                        0,
                    )
                    .unwrap();
                gemv_kernel
                    .launch_f16_gemm(
                        &activation_view,
                        &packed_up_view,
                        &scales_up_view,
                        true,
                        None,
                        None,
                        &mut up_out,
                        m,
                        k_blocks,
                        gemv_kernel.block_size * gemv_kernel.bits / 8,
                        0,
                    )
                    .unwrap();
            }
            // Then the reference silu_mul (byte-identical to silu_mul_f16).
            let ref_function = runtime
                .nvrtc_function(
                    "matmul_nbits_ref_silu_mul",
                    REF_SILU_MUL_SRC,
                    "ref_silu_mul_f16",
                )
                .unwrap();
            let gate_out_ptr = cuptr(device_ptr(gate_out_dev).0);
            let up_out_ptr = cuptr(device_ptr(up_out_dev).0);
            let ref_out_ptr = cuptr(device_ptr(ref_out_dev).0);
            let output_elements_i32 = output_elements as i32;
            let mut ref_builder = runtime.stream().launch_builder(&ref_function);
            ref_builder
                .arg(&gate_out_ptr)
                .arg(&up_out_ptr)
                .arg(&ref_out_ptr)
                .arg(&output_elements_i32);
            // SAFETY: all three buffers hold `output_elements` fp16 values.
            unsafe {
                ref_builder.launch(LaunchConfig {
                    grid_dim: (output_elements.div_ceil(256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();

            // Subject: exercise the fused node's real M=1/M>1 dispatch.
            let inputs = [
                activation_view,
                packed_gate_view,
                scales_gate_view,
                packed_up_view,
                scales_up_view,
            ];
            let mut outputs = [fused_out];
            gemv_kernel
                .run_f16_gate_up_swiglu(&inputs, &mut outputs, None)
                .unwrap();
            // Plain (gamma=None) gate/up SwiGLU keeps the original strict
            // invariant: ONLY M==1 decode is advertised capture-safe. Unlike the
            // rmsnorm-prologue path (see the companion assertion in
            // `run_fused_gate_up_swiglu_rmsnorm_parity`, narrowed to the per-row
            // decode-GEMV loop window), the plain path is deliberately NOT routed
            // through the small-M per-row loop: a plain fused decode GEMV is
            // measured up to 2 ULP off the two-op decode reference even at M==1
            // (`fp16_gate_up_swiglu_two_op_ulp_bound_sweep`), so looping it at M>1
            // would move plain-path logits by >1 ULP vs the reference two-op path.
            // Plain M>1 therefore still falls to the prefill/Marlin GEMM
            // (capture-unsafe unless Marlin's warm path advertises otherwise),
            // and this invariant stays `m == 1`. Tracked in #1334.
            assert_eq!(
                gemv_kernel.last_call_capture_safe.load(Ordering::Relaxed),
                m == 1,
                "only M=1 decode may be advertised capture-safe"
            );
            runtime.synchronize().unwrap();

            let mut reference = vec![f16::ZERO; output_elements];
            let mut fused = vec![f16::ZERO; output_elements];
            // SAFETY: both output allocations hold `output_elements` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut reference), ref_out_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut fused), fused_out_dev)
                    .unwrap();
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(packed_gate_dev).unwrap();
                runtime.free_raw(scales_gate_dev).unwrap();
                runtime.free_raw(packed_up_dev).unwrap();
                runtime.free_raw(scales_up_dev).unwrap();
                runtime.free_raw(gate_out_dev).unwrap();
                runtime.free_raw(up_out_dev).unwrap();
                runtime.free_raw(ref_out_dev).unwrap();
                runtime.free_raw(fused_out_dev).unwrap();
            }

            let marlin_prefill = m > 1
                && marlin_gemm::marlin_m_gt_1_enabled()
                && marlin_gemm::device_supports_marlin(runtime.capabilities().compute_capability())
                && k.is_multiple_of(16)
                && k.is_multiple_of(block_size);
            let mut max_ulp = 0i64;
            let mut worst = 0usize;
            for index in 0..output_elements {
                let ulp = f16_ulp_diff(fused[index].to_bits(), reference[index].to_bits());
                if ulp > max_ulp {
                    max_ulp = ulp;
                    worst = index;
                }
                if !marlin_prefill {
                    assert_eq!(
                        fused[index].to_bits(),
                        reference[index].to_bits(),
                        "paired gate/up SwiGLU diverged at M={m}, K={k}, N={n}, row={}, \
                         column={}: fused={:?} reference={:?}",
                        index / n,
                        index % n,
                        fused[index],
                        reference[index]
                    );
                }
            }
            if marlin_prefill {
                // On SM80+ the default M>1 gate/up dispatch is the Marlin
                // `mma.sync.aligned.m16n8k16` tensor-core GEMM, while the
                // two-op reference above is the portable 16x16 CUDA-core tiled
                // GEMM. Those are different reduction trees (tensor-core K=16
                // fragments, with per-group scaling around the mma accumulator,
                // vs scalar CUDA-core accumulation), so byte identity is not a
                // valid contract for this branch. Keep this sentinel tight: the
                // deterministic Marlin-vs-tiled gate/up cases covered here are
                // expected to stay within two fp16 representable values.
                assert!(
                    max_ulp <= 2,
                    "paired gate/up SwiGLU Marlin prefill exceeded the measured 2-ULP \
                     sentinel bound at M={m}, K={k}, N={n}, row={}, column={}: fused={:?} \
                     reference={:?}, max_ulp={max_ulp}",
                    worst / n,
                    worst % n,
                    fused[worst],
                    reference[worst]
                );
            }
        }
    }

    /// fp16 ULP distance between two half-precision bit patterns, using a
    /// sign-magnitude → monotone-integer key so that adjacent representable
    /// values are exactly 1 apart. Not meaningful for NaN/Inf; the SwiGLU
    /// sweep bounds its activations so those do not arise.
    fn f16_ulp_diff(a: u16, b: u16) -> i64 {
        let key = |bits: u16| -> i64 {
            let mag = (bits & 0x7fff) as i64;
            if bits & 0x8000 != 0 { -mag } else { mag }
        };
        (key(a) - key(b)).abs()
    }

    /// One (M, K, N, seed) case of the fused gate/up SwiGLU vs two-op reference
    /// comparison. Reference construction is IDENTICAL to
    /// [`fp16_gate_up_swiglu_is_bit_exact_to_two_op_path`]: M==1 uses two
    /// standalone General decode GEMVs, M>1 uses the tiled prefill GEMM, both
    /// followed by the reference silu_mul. Returns
    /// `(max_ulp, elements_with_nonzero_ulp, total_elements)`.
    fn gate_up_swiglu_two_op_ulp_case(
        runtime: &Arc<CudaRuntime>,
        m: usize,
        k: usize,
        n: usize,
        seed: u64,
    ) -> (i64, usize, usize) {
        const REF_SILU_MUL_SRC: &str = r#"
#include <cuda_fp16.h>
__device__ float ref_op_silu(float x) {
    if (x >= 0.0f) {
        const float denominator = __fadd_rn(1.0f, (float)exp((double)-x));
        return __fdiv_rn(x, denominator);
    }
    const float e = (float)exp((double)x);
    const float numerator = __fmul_rn(x, e);
    return __fdiv_rn(numerator, __fadd_rn(1.0f, e));
}
extern "C" __global__ void ref_silu_mul_f16(
    const __half* g, const __half* u, __half* y, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half_rn(
            __fmul_rn(ref_op_silu(__half2float(g[i])), __half2float(u[i])));
    }
}
"#;
        let block_size = 32usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;

        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let pack = |next: &mut dyn FnMut() -> f32| -> Vec<u8> {
            let mut quant = vec![0u8; n * k];
            for value in quant.iter_mut() {
                *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
            }
            let mut packed = vec![0u8; n * k_blocks * blob_size];
            for col in 0..n {
                for block in 0..k_blocks {
                    for pair in 0..blob_size {
                        let low = quant[col * k + block * block_size + pair * 2] & 15;
                        let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                        packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                    }
                }
            }
            packed
        };

        let activation: Vec<f16> = (0..m * k).map(|_| f16::from_f32(next())).collect();
        let packed_gate = pack(&mut next);
        let scales_gate: Vec<f16> = (0..n * k_blocks)
            .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
            .collect();
        let packed_up = pack(&mut next);
        let scales_up: Vec<f16> = (0..n * k_blocks)
            .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
            .collect();

        let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
        let packed_gate_dev = runtime.alloc_raw(packed_gate.len()).unwrap();
        let scales_gate_dev = runtime.alloc_raw(scales_gate.len() * 2).unwrap();
        let packed_up_dev = runtime.alloc_raw(packed_up.len()).unwrap();
        let scales_up_dev = runtime.alloc_raw(scales_up.len() * 2).unwrap();
        let output_elements = m * n;
        let gate_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
        let up_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
        let ref_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
        let fused_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
        // SAFETY: device buffers exactly cover their source slices.
        unsafe {
            runtime.htod(as_bytes(&activation), activation_dev).unwrap();
            runtime.htod(&packed_gate, packed_gate_dev).unwrap();
            runtime
                .htod(as_bytes(&scales_gate), scales_gate_dev)
                .unwrap();
            runtime.htod(&packed_up, packed_up_dev).unwrap();
            runtime.htod(as_bytes(&scales_up), scales_up_dev).unwrap();
        }

        let device = DeviceId::cuda(0);
        let a_shape = [m, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let y_shape = [m, n];
        let y_strides = [n as i64, 1];
        let activation_view = TensorView::new(
            device_ptr(activation_dev),
            DataType::Float16,
            &a_shape,
            &a_strides,
            device,
        );
        let packed_gate_view = TensorView::new(
            device_ptr(packed_gate_dev),
            DataType::Uint8,
            &b_shape,
            &b_strides,
            device,
        );
        let scales_gate_view = TensorView::new(
            device_ptr(scales_gate_dev),
            DataType::Float16,
            &scales_shape,
            &scales_strides,
            device,
        );
        let packed_up_view = TensorView::new(
            device_ptr(packed_up_dev),
            DataType::Uint8,
            &b_shape,
            &b_strides,
            device,
        );
        let scales_up_view = TensorView::new(
            device_ptr(scales_up_dev),
            DataType::Float16,
            &scales_shape,
            &scales_strides,
            device,
        );
        let mut gate_out = TensorMut::new(
            device_ptr_mut(gate_out_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );
        let mut up_out = TensorMut::new(
            device_ptr_mut(up_out_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );
        let fused_out = TensorMut::new(
            device_ptr_mut(fused_out_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );

        let gemv_kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };
        if m == 1 {
            let selection = select_f16_gemv_variant(k, n, block_size, true, false);
            gemv_kernel
                .launch_f16_gemv_variant(
                    &activation_view,
                    &packed_gate_view,
                    &scales_gate_view,
                    true,
                    None,
                    None,
                    &mut gate_out,
                    k_blocks,
                    blob_size,
                    k_blocks.div_ceil(2),
                    selection,
                )
                .unwrap();
            gemv_kernel
                .launch_f16_gemv_variant(
                    &activation_view,
                    &packed_up_view,
                    &scales_up_view,
                    true,
                    None,
                    None,
                    &mut up_out,
                    k_blocks,
                    blob_size,
                    k_blocks.div_ceil(2),
                    selection,
                )
                .unwrap();
        } else {
            gemv_kernel
                .launch_f16_gemm(
                    &activation_view,
                    &packed_gate_view,
                    &scales_gate_view,
                    true,
                    None,
                    None,
                    &mut gate_out,
                    m,
                    k_blocks,
                    gemv_kernel.block_size * gemv_kernel.bits / 8,
                    0,
                )
                .unwrap();
            gemv_kernel
                .launch_f16_gemm(
                    &activation_view,
                    &packed_up_view,
                    &scales_up_view,
                    true,
                    None,
                    None,
                    &mut up_out,
                    m,
                    k_blocks,
                    gemv_kernel.block_size * gemv_kernel.bits / 8,
                    0,
                )
                .unwrap();
        }
        let ref_function = runtime
            .nvrtc_function(
                "matmul_nbits_ref_silu_mul",
                REF_SILU_MUL_SRC,
                "ref_silu_mul_f16",
            )
            .unwrap();
        let gate_out_ptr = cuptr(device_ptr(gate_out_dev).0);
        let up_out_ptr = cuptr(device_ptr(up_out_dev).0);
        let ref_out_ptr = cuptr(device_ptr(ref_out_dev).0);
        let output_elements_i32 = output_elements as i32;
        let mut ref_builder = runtime.stream().launch_builder(&ref_function);
        ref_builder
            .arg(&gate_out_ptr)
            .arg(&up_out_ptr)
            .arg(&ref_out_ptr)
            .arg(&output_elements_i32);
        // SAFETY: all three buffers hold `output_elements` fp16 values.
        unsafe {
            ref_builder.launch(LaunchConfig {
                grid_dim: (output_elements.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();

        let inputs = [
            activation_view,
            packed_gate_view,
            scales_gate_view,
            packed_up_view,
            scales_up_view,
        ];
        let mut outputs = [fused_out];
        gemv_kernel
            .run_f16_gate_up_swiglu(&inputs, &mut outputs, None)
            .unwrap();
        runtime.synchronize().unwrap();

        let mut reference = vec![f16::ZERO; output_elements];
        let mut fused = vec![f16::ZERO; output_elements];
        // SAFETY: both output allocations hold `output_elements` fp16 values.
        unsafe {
            runtime
                .dtoh(as_bytes_mut(&mut reference), ref_out_dev)
                .unwrap();
            runtime
                .dtoh(as_bytes_mut(&mut fused), fused_out_dev)
                .unwrap();
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_gate_dev).unwrap();
            runtime.free_raw(scales_gate_dev).unwrap();
            runtime.free_raw(packed_up_dev).unwrap();
            runtime.free_raw(scales_up_dev).unwrap();
            runtime.free_raw(gate_out_dev).unwrap();
            runtime.free_raw(up_out_dev).unwrap();
            runtime.free_raw(ref_out_dev).unwrap();
            runtime.free_raw(fused_out_dev).unwrap();
        }

        let mut max_ulp = 0i64;
        let mut nonzero = 0usize;
        for index in 0..output_elements {
            let d = f16_ulp_diff(fused[index].to_bits(), reference[index].to_bits());
            if d > 0 {
                nonzero += 1;
            }
            if d > max_ulp {
                max_ulp = d;
            }
        }
        (max_ulp, nonzero, output_elements)
    }

    /// Measured ULP bound of the *plain* (no rmsnorm prologue) fused gate/up
    /// SwiGLU decode GEMV against the two-op reference (two standalone
    /// MatMulNBits projections + reference silu_mul), swept across activations
    /// (many seeds) and shapes at **M==1**.
    ///
    /// WHY THIS EXISTS — and why it is M==1-only:
    /// `fp16_gate_up_swiglu_is_bit_exact_to_two_op_path` asserts *byte* equality
    /// against the two-op reference. That identity holds only for the particular
    /// activations its four tuples happen to sample. The fused decode GEMV
    /// carries gate/up in fp32 and the two-op reference rounds each projection to
    /// fp16 before silu_mul, so the two differ by up to **2 ULP** depending on
    /// the activation — *even at M==1*, the apples-to-apples decode comparison.
    /// This measurement is the decision-rule input that HARD-STOPPED the plain
    /// half of the batch-decode capture-safe fix: because the plain fused decode
    /// GEMV is not within 1 ULP of the two-op path even at M==1, the production
    /// per-row decode-GEMV loop is gated to the rmsnorm-prologue path only
    /// (`gamma.is_some()`), and plain M>1 stays on the prefill/Marlin GEMM.
    /// See the byte-identical rmsnorm analog
    /// `fp16_gate_up_swiglu_rmsnorm_two_op_ulp_bound_sweep` (0 ULP at M==1) and
    /// issue #1334.
    ///
    /// Restricted to M==1 on purpose. With the plain path no longer looped, the
    /// plain M>1 fused gate/up runs through the direct Marlin int4 tensor-core
    /// GEMM (`marlin_m_gt_1_enabled()`, default **ON**; split-K
    /// `ONNX_GENAI_MARLIN_SPLITK` is a separate, default-OFF, deterministic
    /// path and does not engage here). That direct kernel accumulates in a
    /// different order than the tiled two-op reference and is documented
    /// *non-byte-identical* to it (its parity tests assert a `2e-2 * max_out`
    /// tolerance, not equality — see `marlin_gemm::marlin_m_gt_1_enabled`). So an
    /// M>1 plain sweep would measure Marlin's accumulation offset, not this
    /// node's decode-vs-two-op bound. It is also why `main` does not actually
    /// hold "M>1 is bit-exact to the two-op reference" under the default config:
    /// `fp16_gate_up_swiglu_is_bit_exact_to_two_op_path` reds at M=5 with Marlin
    /// default-on and only passes under `ONNX_GENAI_MARLIN_M_GT_1=0` (tiled).
    #[test]
    fn fp16_gate_up_swiglu_two_op_ulp_bound_sweep() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping gate/up SwiGLU ULP sweep: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping gate/up SwiGLU ULP sweep: fp16 NVRTC headers unavailable");
            return;
        }

        // (K=hidden, N=intermediate), K divisible by block_size (32).
        let shapes = [
            (QWEN_DOWN_N, QWEN_DOWN_K), // Qwen2.5-0.5B gate/up: 896 -> 4864
            (2048usize, 5632usize),     // Llama-7B-ish MLP
            (1024, 2816),
            (896, 896), // square
            (512, 1536),
            (128, 256), // small, grid-starved
            (96, 77),   // odd tail (matches the existing test's small case)
        ];
        // M==1 only: see the doc comment. Plain M>1 routes through the direct
        // Marlin int4 tensor-core GEMM (non-byte-identical to the two-op
        // reference by a different accumulation order), not this node's decode
        // GEMV.
        let ms = [1usize];
        let seeds: [u64; 8] = [
            0x0bad_c0de_dead_beef,
            0x1234_5678_9abc_def0,
            0xdead_beef_cafe_babe,
            0x0f0f_0f0f_f0f0_f0f0,
            0xa5a5_5a5a_1111_2222,
            0x9e37_79b9_7f4a_7c15,
            0xc2b2_ae3d_27d4_eb4f,
            0xff51_afd7_ed55_8ccd,
        ];

        let mut global_max = 0i64;
        let mut m1_max = 0i64;
        let mut m1_nonzero_cases = 0usize;
        let mut m1_cases = 0usize;
        let mut worst = (0usize, 0usize, 0usize, 0u64); // (m,k,n,seed)
        for (k, n) in shapes {
            for m in ms {
                for seed in seeds {
                    let (max_ulp, nonzero, total) =
                        gate_up_swiglu_two_op_ulp_case(&runtime, m, k, n, seed);
                    assert!(total > 0);
                    if m == 1 {
                        m1_cases += 1;
                        if nonzero > 0 {
                            m1_nonzero_cases += 1;
                        }
                        if max_ulp > m1_max {
                            m1_max = max_ulp;
                        }
                    }
                    if max_ulp > global_max {
                        global_max = max_ulp;
                        worst = (m, k, n, seed);
                    }
                }
            }
        }

        eprintln!(
            "gate/up SwiGLU two-op ULP sweep: max_ulp={global_max} \
             (worst: M={} K={} N={} seed={:#018x}); \
             M==1: max_ulp={m1_max}, {m1_nonzero_cases}/{m1_cases} cases had >=1 ULP divergence",
            worst.0, worst.1, worst.2, worst.3
        );

        // Decision-rule input (HARD STOP): the plain fused decode GEMV is a
        // deterministic approximation of the two-op reference, but NOT within
        // 1 ULP even at M==1 — the measured bound is 2 ULP. This is exactly why
        // the plain path is not routed through the M>1 per-row decode loop
        // (that would move plain-path logits >1 ULP vs the two-op path). The
        // bound is asserted at its measured value so a *regression past it*
        // (e.g. a kernel change widening the gap) still fails loudly; loosening
        // this number must be a deliberate, measured edit, not an accident.
        // The rmsnorm half, by contrast, is byte-identical at M==1 (0 ULP) and
        // IS landed — see `fp16_gate_up_swiglu_rmsnorm_two_op_ulp_bound_sweep`.
        assert_eq!(
            global_max, m1_max,
            "sweep is M==1-only; global and M==1 maxima must coincide"
        );
        assert!(
            global_max <= 2,
            "plain fused gate/up SwiGLU exceeded the measured 2 ULP bound vs the \
             two-op reference: max_ulp={global_max} at M={} K={} N={} seed={:#018x}",
            worst.0,
            worst.1,
            worst.2,
            worst.3
        );
    }

    /// One (M, K, N, seed, gamma_dtype) case of the fused gate/up SwiGLU *with an
    /// RMS-norm prologue* vs the two-op reference. Reference: normalize the
    /// activation with the production prefill norm kernel, then two standalone
    /// projections (M==1 → General GEMV, M>1 → tiled GEMM) + reference silu_mul
    /// on the normalized activation. Subject: the fused rmsnorm-prologue kernel
    /// over the raw activation with gamma at slot 5. Returns
    /// `(max_ulp, elements_with_nonzero_ulp, total_elements)`.
    ///
    /// This is the RMS-norm analog of [`gate_up_swiglu_two_op_ulp_case`]: it
    /// measures whether the rmsnorm decode-GEMV loop is byte-identical to the
    /// two-op (prefill-equivalent) rmsnorm path, i.e. whether the "rmsnorm half"
    /// is genuinely a byte-safe substitute for what `main` ships, or only
    /// byte-identical to a loop-vs-loop reference.
    fn gate_up_swiglu_rmsnorm_two_op_ulp_case(
        runtime: &Arc<CudaRuntime>,
        m: usize,
        k: usize,
        n: usize,
        seed: u64,
        gamma_dtype: DataType,
    ) -> (i64, usize, usize) {
        const REF_SILU_MUL_SRC: &str = r#"
#include <cuda_fp16.h>
__device__ float ref_op_silu(float x) {
    if (x >= 0.0f) {
        const float denominator = __fadd_rn(1.0f, (float)exp((double)-x));
        return __fdiv_rn(x, denominator);
    }
    const float e = (float)exp((double)x);
    const float numerator = __fmul_rn(x, e);
    return __fdiv_rn(numerator, __fadd_rn(1.0f, e));
}
extern "C" __global__ void ref_silu_mul_f16(
    const __half* g, const __half* u, __half* y, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half_rn(
            __fmul_rn(ref_op_silu(__half2float(g[i])), __half2float(u[i])));
    }
}
"#;
        let epsilon = 1e-5f32;
        let block_size = 32usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;

        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let pack = |next: &mut dyn FnMut() -> f32| -> Vec<u8> {
            let mut quant = vec![0u8; n * k];
            for value in quant.iter_mut() {
                *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
            }
            let mut packed = vec![0u8; n * k_blocks * blob_size];
            for col in 0..n {
                for block in 0..k_blocks {
                    for pair in 0..blob_size {
                        let low = quant[col * k + block * block_size + pair * 2] & 15;
                        let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                        packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                    }
                }
            }
            packed
        };

        let activation: Vec<f16> = (0..m * k).map(|_| f16::from_f32(next())).collect();
        let packed_gate = pack(&mut next);
        let scales_gate: Vec<f16> = (0..n * k_blocks)
            .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
            .collect();
        let packed_up = pack(&mut next);
        let scales_up: Vec<f16> = (0..n * k_blocks)
            .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
            .collect();
        let gamma_is_f32 = gamma_dtype == DataType::Float32;
        let gamma_f32: Vec<f32> = (0..k).map(|_| 0.5 + 0.5 * (next() * 0.5 + 0.5)).collect();
        let gamma_bytes: Vec<u8> = if gamma_is_f32 {
            gamma_f32.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            gamma_f32
                .iter()
                .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                .collect()
        };

        let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
        let packed_gate_dev = runtime.alloc_raw(packed_gate.len()).unwrap();
        let scales_gate_dev = runtime.alloc_raw(scales_gate.len() * 2).unwrap();
        let packed_up_dev = runtime.alloc_raw(packed_up.len()).unwrap();
        let scales_up_dev = runtime.alloc_raw(scales_up.len() * 2).unwrap();
        let gamma_dev = runtime.alloc_raw(gamma_bytes.len()).unwrap();
        let normalized_dev = runtime.alloc_raw(m * k * 2).unwrap();
        let output_elements = m * n;
        let gate_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
        let up_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
        let ref_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
        let fused_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
        // SAFETY: device buffers exactly cover their source slices.
        unsafe {
            runtime.htod(as_bytes(&activation), activation_dev).unwrap();
            runtime.htod(&packed_gate, packed_gate_dev).unwrap();
            runtime
                .htod(as_bytes(&scales_gate), scales_gate_dev)
                .unwrap();
            runtime.htod(&packed_up, packed_up_dev).unwrap();
            runtime.htod(as_bytes(&scales_up), scales_up_dev).unwrap();
            runtime.htod(&gamma_bytes, gamma_dev).unwrap();
        }

        let device = DeviceId::cuda(0);
        let a_shape = [m, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let gamma_shape = [k];
        let gamma_strides = [1i64];
        let y_shape = [m, n];
        let y_strides = [n as i64, 1];
        let activation_view = TensorView::new(
            device_ptr(activation_dev),
            DataType::Float16,
            &a_shape,
            &a_strides,
            device,
        );
        let normalized_view = TensorView::new(
            device_ptr(normalized_dev),
            DataType::Float16,
            &a_shape,
            &a_strides,
            device,
        );
        let packed_gate_view = TensorView::new(
            device_ptr(packed_gate_dev),
            DataType::Uint8,
            &b_shape,
            &b_strides,
            device,
        );
        let scales_gate_view = TensorView::new(
            device_ptr(scales_gate_dev),
            DataType::Float16,
            &scales_shape,
            &scales_strides,
            device,
        );
        let packed_up_view = TensorView::new(
            device_ptr(packed_up_dev),
            DataType::Uint8,
            &b_shape,
            &b_strides,
            device,
        );
        let scales_up_view = TensorView::new(
            device_ptr(scales_up_dev),
            DataType::Float16,
            &scales_shape,
            &scales_strides,
            device,
        );
        let gamma_view = TensorView::new(
            device_ptr(gamma_dev),
            gamma_dtype,
            &gamma_shape,
            &gamma_strides,
            device,
        );
        let mut gate_out = TensorMut::new(
            device_ptr_mut(gate_out_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );
        let mut up_out = TensorMut::new(
            device_ptr_mut(up_out_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );
        let fused_out = TensorMut::new(
            device_ptr_mut(fused_out_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        );

        let plain_kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: epsilon,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };
        let fused_kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 4,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: true,
            decomposed_silu: false,
            rmsnorm_prologue: true,
            rmsnorm_epsilon: epsilon,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };

        // Reference: normalize with the production prefill norm, then two-op on
        // the normalized activation (main's prefill-equivalent rmsnorm path).
        plain_kernel
            .launch_rmsnorm_prefill(
                &activation_view,
                &gamma_view,
                cuptr(device_ptr(normalized_dev).0),
                m,
            )
            .unwrap();
        if m == 1 {
            let selection = select_f16_gemv_variant(k, n, block_size, true, false);
            plain_kernel
                .launch_f16_gemv_variant(
                    &normalized_view,
                    &packed_gate_view,
                    &scales_gate_view,
                    true,
                    None,
                    None,
                    &mut gate_out,
                    k_blocks,
                    blob_size,
                    k_blocks.div_ceil(2),
                    selection,
                )
                .unwrap();
            plain_kernel
                .launch_f16_gemv_variant(
                    &normalized_view,
                    &packed_up_view,
                    &scales_up_view,
                    true,
                    None,
                    None,
                    &mut up_out,
                    k_blocks,
                    blob_size,
                    k_blocks.div_ceil(2),
                    selection,
                )
                .unwrap();
        } else {
            plain_kernel
                .launch_f16_gemm(
                    &normalized_view,
                    &packed_gate_view,
                    &scales_gate_view,
                    true,
                    None,
                    None,
                    &mut gate_out,
                    m,
                    k_blocks,
                    plain_kernel.block_size * plain_kernel.bits / 8,
                    0,
                )
                .unwrap();
            plain_kernel
                .launch_f16_gemm(
                    &normalized_view,
                    &packed_up_view,
                    &scales_up_view,
                    true,
                    None,
                    None,
                    &mut up_out,
                    m,
                    k_blocks,
                    plain_kernel.block_size * plain_kernel.bits / 8,
                    0,
                )
                .unwrap();
        }
        let ref_function = runtime
            .nvrtc_function(
                "matmul_nbits_ref_silu_mul",
                REF_SILU_MUL_SRC,
                "ref_silu_mul_f16",
            )
            .unwrap();
        let gate_out_ptr = cuptr(device_ptr(gate_out_dev).0);
        let up_out_ptr = cuptr(device_ptr(up_out_dev).0);
        let ref_out_ptr = cuptr(device_ptr(ref_out_dev).0);
        let output_elements_i32 = output_elements as i32;
        let mut ref_builder = runtime.stream().launch_builder(&ref_function);
        ref_builder
            .arg(&gate_out_ptr)
            .arg(&up_out_ptr)
            .arg(&ref_out_ptr)
            .arg(&output_elements_i32);
        // SAFETY: all three buffers hold `output_elements` fp16 values.
        unsafe {
            ref_builder.launch(LaunchConfig {
                grid_dim: (output_elements.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();

        // Subject: the fused rmsnorm-prologue kernel over the RAW activation.
        let inputs = [
            activation_view,
            packed_gate_view,
            scales_gate_view,
            packed_up_view,
            scales_up_view,
            gamma_view,
        ];
        let mut outputs = [fused_out];
        fused_kernel
            .run_f16_gate_up_swiglu(&inputs, &mut outputs, None)
            .unwrap();
        runtime.synchronize().unwrap();

        let mut reference = vec![f16::ZERO; output_elements];
        let mut fused = vec![f16::ZERO; output_elements];
        // SAFETY: both output allocations hold `output_elements` fp16 values.
        unsafe {
            runtime
                .dtoh(as_bytes_mut(&mut reference), ref_out_dev)
                .unwrap();
            runtime
                .dtoh(as_bytes_mut(&mut fused), fused_out_dev)
                .unwrap();
            runtime.free_raw(activation_dev).unwrap();
            runtime.free_raw(packed_gate_dev).unwrap();
            runtime.free_raw(scales_gate_dev).unwrap();
            runtime.free_raw(packed_up_dev).unwrap();
            runtime.free_raw(scales_up_dev).unwrap();
            runtime.free_raw(gamma_dev).unwrap();
            runtime.free_raw(normalized_dev).unwrap();
            runtime.free_raw(gate_out_dev).unwrap();
            runtime.free_raw(up_out_dev).unwrap();
            runtime.free_raw(ref_out_dev).unwrap();
            runtime.free_raw(fused_out_dev).unwrap();
        }

        let mut max_ulp = 0i64;
        let mut nonzero = 0usize;
        for index in 0..output_elements {
            let d = f16_ulp_diff(fused[index].to_bits(), reference[index].to_bits());
            if d > 0 {
                nonzero += 1;
            }
            if d > max_ulp {
                max_ulp = d;
            }
        }
        (max_ulp, nonzero, output_elements)
    }

    /// RMS-norm analog of [`fp16_gate_up_swiglu_two_op_ulp_bound_sweep`]. The
    /// rmsnorm parity tests (`fused_gate_up_swiglu_rmsnorm_*`) are byte-exact,
    /// but their reference reuses a *per-row plain decode-GEMV loop* as its
    /// second step, so they only prove `rmsnorm-loop == normalize + plain-loop`,
    /// NOT that the rmsnorm loop matches a genuine two-op path. This sweep
    /// compares the fused rmsnorm loop against a real two-op reference (prefill
    /// normalize + standalone projections + silu_mul), across gamma dtypes,
    /// shapes, seeds, and M. ANSWER: byte-identical (0 ULP) at M==1 only when
    /// the standalone projections use the same single-warp block-32 reduction as
    /// the paired fused kernel. On many-SM devices the standalone general GEMV
    /// may select its grid-fill split-K variant for narrow N; that intentionally
    /// reassociates the K reduction and is bounded separately below.
    #[test]
    fn fp16_gate_up_swiglu_rmsnorm_two_op_ulp_bound_sweep() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping gate/up SwiGLU rmsnorm ULP sweep: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping gate/up SwiGLU rmsnorm ULP sweep: fp16 NVRTC headers unavailable");
            return;
        }

        // (K=hidden, N=intermediate); K % 128 == 0 for the warp_half4 norm.
        let shapes = [
            (QWEN_DOWN_N, QWEN_DOWN_K), // 896 -> 4864
            (2048usize, 5632usize),
            (1024, 2816),
            (896, 896),
            (512, 1536),
        ];
        let ms = [1usize, 2, 5, 8];
        let seeds: [u64; 6] = [
            0x0bad_c0de_dead_beef,
            0x1234_5678_9abc_def0,
            0xdead_beef_cafe_babe,
            0xa5a5_5a5a_1111_2222,
            0x9e37_79b9_7f4a_7c15,
            0xff51_afd7_ed55_8ccd,
        ];
        let gammas = [DataType::Float16, DataType::Float32];

        let mut global_max = 0i64;
        let mut m1_max = 0i64;
        let mut m1_single_warp_max = 0i64;
        let mut m1_splitk_max = 0i64;
        let mut m1_nonzero_cases = 0usize;
        let mut m1_cases = 0usize;
        let mut m1_worst = (0usize, 0usize, 0u64, DataType::Float16);
        let mut worst = (0usize, 0usize, 0usize, 0u64, DataType::Float16);
        let caps = runtime.capabilities();
        for gamma in gammas {
            for (k, n) in shapes {
                let reference_uses_splitk = use_f16_symmetric_splitk(
                    k,
                    n,
                    caps.multiprocessor_count(),
                    caps.max_threads_per_block(),
                );
                for m in ms {
                    for seed in seeds {
                        let (max_ulp, nonzero, total) =
                            gate_up_swiglu_rmsnorm_two_op_ulp_case(&runtime, m, k, n, seed, gamma);
                        assert!(total > 0);
                        if m == 1 {
                            m1_cases += 1;
                            if nonzero > 0 {
                                m1_nonzero_cases += 1;
                            }
                            if max_ulp > m1_max {
                                m1_max = max_ulp;
                                m1_worst = (k, n, seed, gamma);
                            }
                            if reference_uses_splitk {
                                m1_splitk_max = m1_splitk_max.max(max_ulp);
                            } else {
                                m1_single_warp_max = m1_single_warp_max.max(max_ulp);
                            }
                        }
                        if max_ulp > global_max {
                            global_max = max_ulp;
                            worst = (m, k, n, seed, gamma);
                        }
                    }
                }
            }
        }

        eprintln!(
            "gate/up SwiGLU rmsnorm two-op ULP sweep: max_ulp={global_max} \
             (worst: M={} K={} N={} seed={:#018x} gamma={:?}); \
             M==1: max_ulp={m1_max} (worst: K={} N={} seed={:#018x} gamma={:?}), \
             {m1_nonzero_cases}/{m1_cases} cases had >=1 ULP divergence",
            worst.0,
            worst.1,
            worst.2,
            worst.3,
            worst.4,
            m1_worst.0,
            m1_worst.1,
            m1_worst.2,
            m1_worst.3
        );

        // DECISION-RULE OUTPUT: at M==1 — the apples-to-apples decode
        // comparison — the fused rmsnorm decode GEMV is BYTE-IDENTICAL to the
        // two-op reference only when the two standalone projections use the same
        // single-warp block-32 reduction shape. On this A100/sm_80, the
        // device-property selector routes narrow standalone projections
        // (`N < SM_count * F16_SYMMETRIC_SPLITK_TARGET_WARPS_PER_SM`, e.g.
        // 896/1536 < 108*16) to `matmul_nbits_gemv_f16_scales_f16_splitk`:
        // two warps own disjoint K-block ranges for one column and reduce their
        // fp32 partials through shared memory. The paired gate/up RMS kernel has
        // no split-K sibling; it keeps one warp per column and therefore cannot
        // be byte-identical to that reference reduction tree on such shapes.
        // Preserve the strict invariant where the reduction shape matches, and
        // separately bound the named split-K reassociation observed on sm_80.
        assert_eq!(
            m1_single_warp_max, 0,
            "rmsnorm fused gate/up SwiGLU decode GEMV must be byte-identical to \
             the two-op reference at M==1 when both use the single-warp reduction, \
             but observed {m1_single_warp_max} ULP"
        );
        assert!(
            m1_splitk_max <= 3,
            "rmsnorm fused gate/up SwiGLU decode GEMV exceeded the measured 3-ULP \
             bound vs the standalone split-K two-op reference at M==1: \
             max_ulp={m1_splitk_max}"
        );
        // M>1 compares the per-row decode GEMV loop against a tiled two-op GEMM
        // reference; they differ by a few ULP purely from GEMV-vs-GEMM reduction
        // order (NOT a batching-contract violation — that is guarded by the
        // per-row == single-stream byte identity test). Bound recorded, not 0.
        assert!(
            global_max <= 4,
            "rmsnorm fused gate/up SwiGLU ULP vs two-op unexpectedly large: \
             max_ulp={global_max}"
        );
    }

    /// Byte-for-byte parity of the fused gate/up SwiGLU kernel *with an RMS
    /// prologue* against the standalone two-step sequence
    /// (`RMS-normalize the activation` → `paired gate/up SwiGLU`). The reference
    /// normalizes with the production prefill norm kernel
    /// (`matmul_nbits_rmsnorm_f16_warp_half4`) and then runs the already-proven
    /// non-prologue paired kernel, so any divergence isolates the fused
    /// prologue. Exercising M==1 (the single fused decode kernel) and M>1 (the
    /// normalize-into-scratch prefill path) keeps both dispatches honest.
    #[test]
    fn fused_gate_up_swiglu_rmsnorm_is_bit_exact_to_two_step_path() {
        run_fused_gate_up_swiglu_rmsnorm_parity(DataType::Float16, false);
    }

    /// Same gate/up SwiGLU RMS-norm fusion parity, but with an fp32 gamma (as
    /// Phi-4-mini exports it). The fused decode/prefill kernels must read the
    /// fp32 gamma at full precision and stay bit-identical to the two-step path.
    #[test]
    fn fused_gate_up_swiglu_rmsnorm_fp32_gamma_is_bit_exact_to_two_step_path() {
        run_fused_gate_up_swiglu_rmsnorm_parity(DataType::Float32, false);
    }

    /// Same gate/up SwiGLU RMS-norm fusion parity, but with asymmetric int4
    /// zero points on BOTH the gate and up projections (as Phi-4-mini exports
    /// them). The fused prologue kernel and the reference non-prologue paired
    /// kernel are independently written, so byte-identity here proves both honor
    /// the per-block zero point in the packed dequant. A fused kernel that
    /// ignored the zero point would diverge from the reference and fail. fp32
    /// gamma is paired with the zero points to mirror Phi's actual export.
    #[test]
    fn fused_gate_up_swiglu_rmsnorm_zero_points_is_bit_exact_to_two_step_path() {
        run_fused_gate_up_swiglu_rmsnorm_parity(DataType::Float32, true);
    }

    fn run_fused_gate_up_swiglu_rmsnorm_parity(gamma_dtype: DataType, explicit_zp: bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping gate/up SwiGLU RMS-norm parity test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!(
                "skipping gate/up SwiGLU RMS-norm parity test: fp16 NVRTC headers unavailable"
            );
            return;
        }

        let epsilon = 1e-5f32;
        // (M, K=hidden, N=intermediate); hidden % 128 == 0 for the warp_half4
        // reduction. Decode is the capture-safe fused kernel; M=5 prefill routes
        // through the normalize-into-scratch path.
        for (m, k, n) in [
            (1usize, 896usize, 2432usize),
            (1, 3584, 4864),
            (5, 896, 2432),
        ] {
            let block_size = 32usize;
            let k_blocks = k / block_size;
            let blob_size = block_size / 2;

            let mut state = 0xf00d_1ceb_00da_5555u64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            let pack = |next: &mut dyn FnMut() -> f32| -> Vec<u8> {
                let mut quant = vec![0u8; n * k];
                for value in quant.iter_mut() {
                    *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
                }
                let mut packed = vec![0u8; n * k_blocks * blob_size];
                for col in 0..n {
                    for block in 0..k_blocks {
                        for pair in 0..blob_size {
                            let low = quant[col * k + block * block_size + pair * 2] & 15;
                            let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                            packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                        }
                    }
                }
                packed
            };

            let activation: Vec<f16> = (0..m * k).map(|_| f16::from_f32(next())).collect();
            let packed_gate = pack(&mut next);
            let scales_gate: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            let packed_up = pack(&mut next);
            let scales_up: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();

            // Optional asymmetric zero points, nibble-packed [n, zp_row_bytes]
            // exactly as `int4_block_zero_point` reads them. Symmetric weights
            // use the implicit zp == 8 and carry no zero-point input.
            let zp_row_bytes = k_blocks.div_ceil(2);
            let pack_zp = |next: &mut dyn FnMut() -> f32| -> Vec<u8> {
                let mut zp = vec![0u8; n * zp_row_bytes];
                for col in 0..n {
                    for block in 0..k_blocks {
                        let nibble =
                            (((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8) & 15;
                        let byte = &mut zp[col * zp_row_bytes + (block >> 1)];
                        if block & 1 == 1 {
                            *byte = (*byte & 0x0f) | (nibble << 4);
                        } else {
                            *byte = (*byte & 0xf0) | nibble;
                        }
                    }
                }
                zp
            };
            let zp_gate = pack_zp(&mut next);
            let zp_up = pack_zp(&mut next);
            let gamma_is_f32 = gamma_dtype == DataType::Float32;
            let gamma_f32: Vec<f32> = (0..k).map(|_| 0.5 + 0.5 * (next() * 0.5 + 0.5)).collect();
            let gamma_bytes: Vec<u8> = if gamma_is_f32 {
                gamma_f32.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else {
                gamma_f32
                    .iter()
                    .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                    .collect()
            };

            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let packed_gate_dev = runtime.alloc_raw(packed_gate.len()).unwrap();
            let scales_gate_dev = runtime.alloc_raw(scales_gate.len() * 2).unwrap();
            let packed_up_dev = runtime.alloc_raw(packed_up.len()).unwrap();
            let scales_up_dev = runtime.alloc_raw(scales_up.len() * 2).unwrap();
            let gamma_dev = runtime.alloc_raw(gamma_bytes.len()).unwrap();
            let zp_gate_dev = runtime.alloc_raw(zp_gate.len()).unwrap();
            let zp_up_dev = runtime.alloc_raw(zp_up.len()).unwrap();
            let normalized_dev = runtime.alloc_raw(m * k * 2).unwrap();
            let output_elements = m * n;
            let ref_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            let fused_out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
            // SAFETY: device buffers exactly cover their source slices.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(&packed_gate, packed_gate_dev).unwrap();
                runtime
                    .htod(as_bytes(&scales_gate), scales_gate_dev)
                    .unwrap();
                runtime.htod(&packed_up, packed_up_dev).unwrap();
                runtime.htod(as_bytes(&scales_up), scales_up_dev).unwrap();
                runtime.htod(&gamma_bytes, gamma_dev).unwrap();
                if explicit_zp {
                    runtime.htod(&zp_gate, zp_gate_dev).unwrap();
                    runtime.htod(&zp_up, zp_up_dev).unwrap();
                }
            }

            let device = DeviceId::cuda(0);
            let a_shape = [m, k];
            let a_strides = [k as i64, 1];
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let scales_shape = [n, k_blocks];
            let scales_strides = [k_blocks as i64, 1];
            let gamma_shape = [k];
            let gamma_strides = [1i64];
            let y_shape = [m, n];
            let y_strides = [n as i64, 1];

            let activation_view = TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            );
            let _normalized_view = TensorView::new(
                device_ptr(normalized_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            );
            let packed_gate_view = TensorView::new(
                device_ptr(packed_gate_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_gate_view = TensorView::new(
                device_ptr(scales_gate_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let packed_up_view = TensorView::new(
                device_ptr(packed_up_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_up_view = TensorView::new(
                device_ptr(scales_up_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let gamma_view = TensorView::new(
                device_ptr(gamma_dev),
                gamma_dtype,
                &gamma_shape,
                &gamma_strides,
                device,
            );
            let zp_shape = [n, zp_row_bytes];
            let zp_strides = [zp_row_bytes as i64, 1];
            let zp_gate_view = TensorView::new(
                device_ptr(zp_gate_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            );
            let zp_up_view = TensorView::new(
                device_ptr(zp_up_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            );
            let fused_out = TensorMut::new(
                device_ptr_mut(fused_out_dev),
                DataType::Float16,
                &y_shape,
                &y_strides,
                device,
            );

            let plain_swiglu = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: true,
                decomposed_silu: false,
                rmsnorm_prologue: false,
                rmsnorm_epsilon: epsilon,
                last_call_capture_safe: AtomicBool::new(false),
                bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
                bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
            };
            let fused_swiglu = MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits: 4,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: false,
                gate_up_swiglu: true,
                decomposed_silu: false,
                rmsnorm_prologue: true,
                rmsnorm_epsilon: epsilon,
                last_call_capture_safe: AtomicBool::new(false),
                bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
                bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
            };

            // Reference: normalize the activation (production prefill norm
            // kernel), then run the proven non-prologue paired gate/up SwiGLU.
            plain_swiglu
                .launch_rmsnorm_prefill(
                    &activation_view,
                    &gamma_view,
                    cuptr(device_ptr(normalized_dev).0),
                    m,
                )
                .unwrap();
            {
                // The reference GEMV is looped per row (M==1 each) rather than
                // run once over all M rows. RMS norm is a per-row reduction and
                // a plain M==1 gate/up SwiGLU always dispatches the capture-safe
                // fused decode GEMV, so row r of this reference is exactly a
                // standalone M==1 normalize+SwiGLU of row r. That keeps the
                // reference byte-identical to the fused rmsnorm per-row decode
                // loop under test even though the plain (gamma=None) path no
                // longer loops M>1 (it falls to the prefill/Marlin GEMM, which
                // is ~3 ULP off a decode GEMV; see the two-op ULP sweep and
                // #1334). Running the plain reference over all M rows here would
                // route through that GEMM and spuriously red this parity test.
                let norm_row_shape = [1usize, k];
                let norm_row_strides = [k as i64, 1];
                let out_row_shape = [1usize, n];
                let out_row_strides = [n as i64, 1];
                let norm_row_bytes = (k * 2) as CUdeviceptr; // fp16
                let out_row_bytes = (n * 2) as CUdeviceptr; // fp16
                for row in 0..m {
                    let norm_row = TensorView::new(
                        device_ptr(normalized_dev + row as CUdeviceptr * norm_row_bytes),
                        DataType::Float16,
                        &norm_row_shape,
                        &norm_row_strides,
                        device,
                    );
                    let out_row = TensorMut::new(
                        device_ptr_mut(ref_out_dev + row as CUdeviceptr * out_row_bytes),
                        DataType::Float16,
                        &out_row_shape,
                        &out_row_strides,
                        device,
                    );
                    let mut ref_outputs = [out_row];
                    let ref_inputs_base = [
                        norm_row,
                        packed_gate_view,
                        scales_gate_view,
                        packed_up_view,
                        scales_up_view,
                    ];
                    if explicit_zp {
                        // Slot 5 gamma absent (already normalized), slots 6/7 zp.
                        let ref_inputs = [
                            ref_inputs_base[0],
                            ref_inputs_base[1],
                            ref_inputs_base[2],
                            ref_inputs_base[3],
                            ref_inputs_base[4],
                            TensorView::absent(DataType::Float16),
                            zp_gate_view,
                            zp_up_view,
                        ];
                        plain_swiglu
                            .run_f16_gate_up_swiglu(&ref_inputs, &mut ref_outputs, None)
                            .unwrap();
                    } else {
                        plain_swiglu
                            .run_f16_gate_up_swiglu(&ref_inputs_base, &mut ref_outputs, None)
                            .unwrap();
                    }
                }
            }

            // Subject: the fused prologue kernel over the raw (residual sum)
            // activation with gamma at slot 5.
            {
                let mut fused_outputs = [fused_out];
                let fused_inputs_base = [
                    activation_view,
                    packed_gate_view,
                    scales_gate_view,
                    packed_up_view,
                    scales_up_view,
                    gamma_view,
                ];
                if explicit_zp {
                    let fused_inputs = [
                        fused_inputs_base[0],
                        fused_inputs_base[1],
                        fused_inputs_base[2],
                        fused_inputs_base[3],
                        fused_inputs_base[4],
                        fused_inputs_base[5],
                        zp_gate_view,
                        zp_up_view,
                    ];
                    fused_swiglu
                        .run_f16_gate_up_swiglu(&fused_inputs, &mut fused_outputs, None)
                        .unwrap();
                } else {
                    fused_swiglu
                        .run_f16_gate_up_swiglu(&fused_inputs_base, &mut fused_outputs, None)
                        .unwrap();
                }
            }
            // Capture-safety tracks the decode-GEMV *routing*, not a fixed
            // M==1 rule (see the companion note on the non-rmsnorm assertion
            // above). `05e1fd10` could only capture M==1 because every M>1
            // fused gate/up-SwiGLU with an RMS-norm prologue reached
            // `launch_gate_up_swiglu_rmsnorm_prefill`, which normalizes into
            // freshly `alloc_raw`'d scratch and so cannot be graph-recorded.
            // The small-M loop now dispatches `1 < m <= decode_gemv_loop_max_m()`
            // as per-row capture-safe M==1 fused kernels (each byte-identical
            // to M==1), making those M genuinely capture-safe; only
            // `m > decode_gemv_loop_max_m()` still reaches the scratch prefill
            // path. Narrowed to exactly that range, not deleted.
            assert_eq!(
                fused_swiglu.last_call_capture_safe.load(Ordering::Relaxed),
                (1..=decode_gemv_loop_max_m()).contains(&m),
                "capture-safe iff the decode-GEMV routing is per-row M==1 \
                 launches: M==1 or small-batch M<=decode_gemv_loop_max_m(); \
                 M>window still reaches the uncapturable prefill GEMM"
            );
            runtime.synchronize().unwrap();

            let mut reference = vec![f16::ZERO; output_elements];
            let mut fused = vec![f16::ZERO; output_elements];
            // SAFETY: both output allocations hold `output_elements` fp16 values.
            unsafe {
                runtime
                    .dtoh(as_bytes_mut(&mut reference), ref_out_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut fused), fused_out_dev)
                    .unwrap();
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(packed_gate_dev).unwrap();
                runtime.free_raw(scales_gate_dev).unwrap();
                runtime.free_raw(packed_up_dev).unwrap();
                runtime.free_raw(scales_up_dev).unwrap();
                runtime.free_raw(gamma_dev).unwrap();
                runtime.free_raw(zp_gate_dev).unwrap();
                runtime.free_raw(zp_up_dev).unwrap();
                runtime.free_raw(normalized_dev).unwrap();
                runtime.free_raw(ref_out_dev).unwrap();
                runtime.free_raw(fused_out_dev).unwrap();
            }

            for index in 0..output_elements {
                assert_eq!(
                    fused[index].to_bits(),
                    reference[index].to_bits(),
                    "fused gate/up SwiGLU RMS prologue diverged at M={m}, K={k}, N={n}, \
                     row={}, column={}: fused={:?} reference={:?}",
                    index / n,
                    index % n,
                    fused[index],
                    reference[index]
                );
            }
        }
    }

    /// The fused-symmetric gate/up SwiGLU decode entries (`..gate_up*_vec`,
    /// opt-in via `ONNX_GENAI_GATEUP_VEC=1`) must be RAW-16-BIT IDENTICAL to the
    /// default symmetric entries. The `_vec` entries only fold the `- 8`
    /// symmetric zero point into the dequant bias constants (see
    /// `int4x8_to_half2x4_sym8`); every `q[i]` value is unchanged, so the fp16
    /// multiply-add accumulation, the RMS-norm prologue, the SwiGLU rounding, and
    /// thus `silu(gate)*up` must match bit-for-bit — and hence match M
    /// independent M=1 greedy `_dsilu` GEMVs. A single argmax flip downstream
    /// would start here. Covers plain + RMS-norm-fused, decomposed +
    /// non-decomposed SiLU, fp16 AND fp32 gamma, the real Qwen2.5-14b gate/up
    /// dims (K=5120, N=13824), and M in {1,4,6,8}. Also asserts the
    /// occupancy-raised `_vec_occ` path (`ONNX_GENAI_GATEUP_OCC`,
    /// `__launch_bounds__(256, 8)`) is bit-identical to the scalar reference for
    /// the symmetric RMS-norm-fused kernels it applies to (it is a no-op for the
    /// non-RMS launch) — `__launch_bounds__` only constrains register
    /// allocation, so the math is unchanged.
    #[test]
    fn gate_up_swiglu_vec_is_bit_identical_to_scalar() {
        let Some(runtime) = runtime() else {
            eprintln!("skipping gate/up SwiGLU _vec bit-identity test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!(
                "skipping gate/up SwiGLU _vec bit-identity test: fp16 NVRTC headers unavailable"
            );
            return;
        }

        let epsilon = 1e-5f32;
        for (m, k, n) in [
            (1usize, 896usize, 2432usize),
            (4, 3584, 4864),
            (6, 5120, 13824),
            (8, 5120, 13824),
        ] {
            let block_size = 32usize;
            let k_blocks = k / block_size;
            let blob_size = block_size / 2;

            let mut state = 0x51de_face_c0de_1234u64 ^ ((m as u64) << 40);
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            let pack = |next: &mut dyn FnMut() -> f32| -> Vec<u8> {
                let mut quant = vec![0u8; n * k];
                for value in quant.iter_mut() {
                    *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
                }
                let mut packed = vec![0u8; n * k_blocks * blob_size];
                for col in 0..n {
                    for block in 0..k_blocks {
                        for pair in 0..blob_size {
                            let low = quant[col * k + block * block_size + pair * 2] & 15;
                            let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                            packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                        }
                    }
                }
                packed
            };

            let activation: Vec<f16> = (0..m * k).map(|_| f16::from_f32(next())).collect();
            let packed_gate = pack(&mut next);
            let scales_gate: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            let packed_up = pack(&mut next);
            let scales_up: Vec<f16> = (0..n * k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            let gamma_f32: Vec<f32> = (0..k).map(|_| 0.5 + 0.5 * (next() * 0.5 + 0.5)).collect();

            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let packed_gate_dev = runtime.alloc_raw(packed_gate.len()).unwrap();
            let scales_gate_dev = runtime.alloc_raw(scales_gate.len() * 2).unwrap();
            let packed_up_dev = runtime.alloc_raw(packed_up.len()).unwrap();
            let scales_up_dev = runtime.alloc_raw(scales_up.len() * 2).unwrap();
            // SAFETY: device buffers exactly cover their source slices.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(&packed_gate, packed_gate_dev).unwrap();
                runtime
                    .htod(as_bytes(&scales_gate), scales_gate_dev)
                    .unwrap();
                runtime.htod(&packed_up, packed_up_dev).unwrap();
                runtime.htod(as_bytes(&scales_up), scales_up_dev).unwrap();
            }

            let device = DeviceId::cuda(0);
            let a_shape = [m, k];
            let a_strides = [k as i64, 1];
            let b_shape = [n, k_blocks, blob_size];
            let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
            let scales_shape = [n, k_blocks];
            let scales_strides = [k_blocks as i64, 1];
            let gamma_shape = [k];
            let gamma_strides = [1i64];
            let y_shape = [m, n];
            let y_strides = [n as i64, 1];
            let output_elements = m * n;

            let activation_view = TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            );
            let packed_gate_view = TensorView::new(
                device_ptr(packed_gate_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_gate_view = TensorView::new(
                device_ptr(scales_gate_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );
            let packed_up_view = TensorView::new(
                device_ptr(packed_up_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            );
            let scales_up_view = TensorView::new(
                device_ptr(scales_up_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            );

            // Every (rmsnorm, decomposed, gamma dtype) symmetric gate/up variant
            // that has a `_vec` sibling, so the bit-identity gate covers all four
            // fused-symmetric entries and both gamma widths.
            for rmsnorm in [false, true] {
                for decomposed in [false, true] {
                    for gamma_dtype in [DataType::Float16, DataType::Float32] {
                        if !rmsnorm && gamma_dtype == DataType::Float32 {
                            // Gamma is only consumed by the RMS-norm prologue.
                            continue;
                        }
                        let gamma_is_f32 = gamma_dtype == DataType::Float32;
                        let gamma_bytes: Vec<u8> = if gamma_is_f32 {
                            gamma_f32.iter().flat_map(|v| v.to_le_bytes()).collect()
                        } else {
                            gamma_f32
                                .iter()
                                .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                                .collect()
                        };
                        let gamma_dev = runtime.alloc_raw(gamma_bytes.len()).unwrap();
                        // SAFETY: buffer sized to the gamma byte slice.
                        unsafe {
                            runtime.htod(&gamma_bytes, gamma_dev).unwrap();
                        }
                        let gamma_view = TensorView::new(
                            device_ptr(gamma_dev),
                            gamma_dtype,
                            &gamma_shape,
                            &gamma_strides,
                            device,
                        );

                        let kernel = MatMulNBitsKernel {
                            runtime: runtime.clone(),
                            k,
                            n,
                            bits: 4,
                            block_size,
                            accuracy_level: 4,
                            accuracy4_workspace: None,
                            fold_bias_post_round: false,
                            gate_up_swiglu: true,
                            decomposed_silu: decomposed,
                            rmsnorm_prologue: rmsnorm,
                            rmsnorm_epsilon: epsilon,
                            last_call_capture_safe: AtomicBool::new(false),
                            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
                            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
                        };

                        let run_once = |vec_on: bool, occ_on: bool| -> Vec<u16> {
                            let out_dev = runtime.alloc_raw(output_elements * 2).unwrap();
                            // Exclusive: this env write is process-wide, and the
                            // guard clears it on drop so an unwinding `unwrap`
                            // below cannot leak the lever to other tests.
                            let _guard = LeverEnvGuard::acquire();
                            // SAFETY: confined to the exclusive section.
                            unsafe {
                                std::env::set_var(
                                    "ONNX_GENAI_GATEUP_VEC",
                                    if vec_on { "1" } else { "0" },
                                );
                                std::env::set_var(
                                    "ONNX_GENAI_GATEUP_OCC",
                                    if occ_on { "1" } else { "0" },
                                );
                            }
                            let out = TensorMut::new(
                                device_ptr_mut(out_dev),
                                DataType::Float16,
                                &y_shape,
                                &y_strides,
                                device,
                            );
                            let mut outputs = [out];
                            if rmsnorm {
                                let inputs = [
                                    activation_view,
                                    packed_gate_view,
                                    scales_gate_view,
                                    packed_up_view,
                                    scales_up_view,
                                    gamma_view,
                                ];
                                kernel
                                    .run_f16_gate_up_swiglu(&inputs, &mut outputs, None)
                                    .unwrap();
                            } else {
                                let inputs = [
                                    activation_view,
                                    packed_gate_view,
                                    scales_gate_view,
                                    packed_up_view,
                                    scales_up_view,
                                ];
                                kernel
                                    .run_f16_gate_up_swiglu(&inputs, &mut outputs, None)
                                    .unwrap();
                            }
                            runtime.synchronize().unwrap();
                            unsafe {
                                std::env::remove_var("ONNX_GENAI_GATEUP_VEC");
                                std::env::remove_var("ONNX_GENAI_GATEUP_OCC");
                            }
                            drop(_guard);

                            let mut host = vec![f16::ZERO; output_elements];
                            // SAFETY: buffer holds `output_elements` fp16 values.
                            unsafe {
                                runtime.dtoh(as_bytes_mut(&mut host), out_dev).unwrap();
                                runtime.free_raw(out_dev).unwrap();
                            }
                            host.iter().map(|v| v.to_bits()).collect()
                        };

                        let reference = run_once(false, false);
                        let fused = run_once(true, false);
                        // Occupancy-raised `_vec_occ` path. The `_vec_occ` entries
                        // exist ONLY for the two symmetric RMS-norm-fused kernels,
                        // so this actually exercises `_vec_occ` when rmsnorm==true;
                        // for the non-rmsnorm launch `ONNX_GENAI_GATEUP_OCC` is a
                        // no-op and this simply re-checks the `_vec` path. Either
                        // way the result must stay bit-identical to the reference.
                        let occ = run_once(true, true);

                        // SAFETY: gamma buffer no longer referenced.
                        unsafe {
                            runtime.free_raw(gamma_dev).unwrap();
                        }

                        for index in 0..output_elements {
                            assert_eq!(
                                fused[index],
                                reference[index],
                                "fused-symmetric gate/up _vec diverged at M={m}, K={k}, N={n}, \
                                 rmsnorm={rmsnorm}, decomposed={decomposed}, \
                                 gamma_f32={gamma_is_f32}, row={}, column={}: vec=0x{:04x} \
                                 scalar=0x{:04x}",
                                index / n,
                                index % n,
                                fused[index],
                                reference[index]
                            );
                            assert_eq!(
                                occ[index],
                                reference[index],
                                "occupancy-raised gate/up _vec_occ diverged at M={m}, K={k}, \
                                 N={n}, rmsnorm={rmsnorm}, decomposed={decomposed}, \
                                 gamma_f32={gamma_is_f32}, row={}, column={}: occ=0x{:04x} \
                                 scalar=0x{:04x}",
                                index / n,
                                index % n,
                                occ[index],
                                reference[index]
                            );
                        }
                    }
                }
            }

            // SAFETY: shared input buffers are done after all variants ran.
            unsafe {
                runtime.free_raw(activation_dev).unwrap();
                runtime.free_raw(packed_gate_dev).unwrap();
                runtime.free_raw(scales_gate_dev).unwrap();
                runtime.free_raw(packed_up_dev).unwrap();
                runtime.free_raw(scales_up_dev).unwrap();
            }
        }
    }

    /// Byte-for-byte parity of the fused SkipSimplifiedLayerNormalization
    /// epilogue/prologue against the standalone three-op sequence
    /// (`preceding MatMulNBits` → `SkipSimplifiedLayerNormalization` →
    /// `following MatMulNBits`) on GPU.
    ///
    /// The reference path runs the exact production kernels: a plain preceding
    /// GEMV, the standalone `skip_rmsnorm_f16_warp_half4` kernel (producing the
    /// normalized output and the residual sum), then a plain following GEMV. The
    /// fused path folds the residual add into the preceding GEMV's bias-slot
    /// epilogue and the RMS normalization into the following GEMV's prologue. The
    /// residual sum (`preceding fused output`) must equal the standalone norm's
    /// `input_skip_bias_sum`, and the final projection must be bit-identical —
    /// for decode (M==1) and prefill (M>1), with and without a following bias.
    #[test]
    fn fused_skip_rmsnorm_is_bit_exact_to_three_op_path() {
        run_fused_skip_rmsnorm_parity(DataType::Float16, 4, false);
    }

    /// Phi-4-mini exports its `SkipSimplifiedLayerNormalization` gamma in fp32.
    /// The fused RMS-norm-prologue GEMV must accept that fp32 gamma and stay
    /// bit-identical to the standalone (fp32-gamma) norm + GEMV pair, so the
    /// fusion fires on Phi as well as on Qwen (fp16 gamma).
    #[test]
    fn fused_skip_rmsnorm_fp32_gamma_is_bit_exact_to_three_op_path() {
        run_fused_skip_rmsnorm_parity(DataType::Float32, 4, false);
    }

    /// Phi-4-mini's qkv/down projections are int8 with non-trivial asymmetric
    /// zero points. The fused int8 RMS-norm-prologue GEMV (following) and the
    /// int8 residual-fold epilogue (preceding) must stay bit-identical to the
    /// standalone int8 GEMV + skip_rmsnorm + int8 GEMV sequence at Phi's dims
    /// (down K=8192>hidden=3072, qkv hidden=3072<=N=5120), fp32 gamma. The
    /// asymmetric zero points make this a mutation guard: ignoring the zero
    /// point (or dropping to fp16 accumulation) diverges from the reference.
    #[test]
    fn fused_skip_rmsnorm_int8_asymmetric_zp_is_bit_exact_to_three_op_path() {
        run_fused_skip_rmsnorm_parity(DataType::Float32, 8, true);
    }

    fn run_fused_skip_rmsnorm_parity(gamma_dtype: DataType, bits: usize, explicit_zp: bool) {
        let Some(runtime) = runtime() else {
            eprintln!("skipping fused skip-rmsnorm parity test: CUDA runtime unavailable");
            return;
        };
        if runtime
            .require_nvrtc_half_headers("matmul_nbits_gemv_f16")
            .is_err()
        {
            eprintln!("skipping fused skip-rmsnorm parity test: fp16 NVRTC headers unavailable");
            return;
        }

        // hidden % 128 == 0 (warp_half4 gate); preceding is a down projection
        // (pre_k > hidden), the following is a general projection (hidden <= n).
        // int8 exercises Phi's actual qkv/down dims; int4 keeps the Qwen shapes.
        let (hidden, pre_k, post_n) = if bits == 8 {
            (3072usize, 8192usize, 5120usize)
        } else {
            (896usize, QWEN_DOWN_K, 1152usize)
        };
        let epsilon = 1e-5f32;
        let block_size = 32usize;
        let blob_size = block_size * bits / 8;
        let device = DeviceId::cuda(0);

        for (m, following_bias) in [(1usize, false), (1, true), (5, true)] {
            let mut state = 0x51ce_d00d_f00d_1234u64;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            };

            // Pack random int4/int8 codes into the block-32 layout the kernels
            // unpack (nibble pairs for int4, one byte per weight for int8).
            let pack = |next: &mut dyn FnMut() -> f32, n: usize, k: usize| -> Vec<u8> {
                let k_blocks = k / block_size;
                let mut packed = vec![0u8; n * k_blocks * blob_size];
                if bits == 8 {
                    for byte in packed.iter_mut() {
                        *byte = ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
                    }
                    return packed;
                }
                let mut quant = vec![0u8; n * k];
                for value in quant.iter_mut() {
                    *value = ((next() * 0.5 + 0.5) * 15.0).round().clamp(0.0, 15.0) as u8;
                }
                for col in 0..n {
                    for block in 0..k_blocks {
                        for pair in 0..blob_size {
                            let low = quant[col * k + block * block_size + pair * 2] & 15;
                            let high = quant[col * k + block * block_size + pair * 2 + 1] & 15;
                            packed[(col * k_blocks + block) * blob_size + pair] = low | (high << 4);
                        }
                    }
                }
                packed
            };

            // Non-uniform asymmetric zero points, one byte per block (int8) so the
            // dequant is `(code - zp) * scale`. `None` keeps the symmetric default
            // (zp == 128 for int8), preserving the byte-identical int4 path.
            let zp_bytes = |next: &mut dyn FnMut() -> f32, n: usize, k: usize| -> Vec<u8> {
                let k_blocks = k / block_size;
                (0..n * k_blocks)
                    .map(|_| (128.0 + (next() * 16.0)).round().clamp(96.0, 160.0) as u8)
                    .collect()
            };

            let pre_k_blocks = pre_k / block_size;
            let post_k_blocks = hidden / block_size;

            let activation: Vec<f16> = (0..m * pre_k).map(|_| f16::from_f32(next())).collect();
            let packed_pre = pack(&mut next, hidden, pre_k);
            let scales_pre: Vec<f16> = (0..hidden * pre_k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            let packed_post = pack(&mut next, post_n, hidden);
            let scales_post: Vec<f16> = (0..post_n * post_k_blocks)
                .map(|_| f16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
                .collect();
            // The residual is a plain fp16 activation, one hidden vector per token.
            let residual: Vec<f16> = (0..m * hidden).map(|_| f16::from_f32(next())).collect();
            // Gamma values are produced identically regardless of storage dtype;
            // fp16 keeps the byte-identical Qwen path, fp32 exercises Phi's export.
            let gamma_is_f32 = gamma_dtype == DataType::Float32;
            let gamma_f32: Vec<f32> = (0..hidden)
                .map(|_| 0.5 + 0.5 * (next() * 0.5 + 0.5))
                .collect();
            let gamma_bytes: Vec<u8> = if gamma_is_f32 {
                gamma_f32.iter().flat_map(|v| v.to_le_bytes()).collect()
            } else {
                gamma_f32
                    .iter()
                    .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                    .collect()
            };
            let bias_post: Vec<f16> = (0..post_n).map(|_| f16::from_f32(next())).collect();
            // Per-block asymmetric zero points (int8 only); `explicit_zp == false`
            // uses the symmetric default and omits the input entirely.
            let zp_pre: Vec<u8> = if explicit_zp {
                zp_bytes(&mut next, hidden, pre_k)
            } else {
                Vec::new()
            };
            let zp_post: Vec<u8> = if explicit_zp {
                zp_bytes(&mut next, post_n, hidden)
            } else {
                Vec::new()
            };

            // Device buffers.
            let activation_dev = runtime.alloc_raw(activation.len() * 2).unwrap();
            let packed_pre_dev = runtime.alloc_raw(packed_pre.len()).unwrap();
            let scales_pre_dev = runtime.alloc_raw(scales_pre.len() * 2).unwrap();
            let packed_post_dev = runtime.alloc_raw(packed_post.len()).unwrap();
            let scales_post_dev = runtime.alloc_raw(scales_post.len() * 2).unwrap();
            let residual_dev = runtime.alloc_raw(residual.len() * 2).unwrap();
            let gamma_dev = runtime.alloc_raw(gamma_bytes.len()).unwrap();
            let bias_post_dev = runtime.alloc_raw(bias_post.len() * 2).unwrap();
            let matmul_out_dev = runtime.alloc_raw(m * hidden * 2).unwrap();
            let normalized_dev = runtime.alloc_raw(m * hidden * 2).unwrap();
            let sum_dev = runtime.alloc_raw(m * hidden * 2).unwrap();
            let mean_dev = runtime.alloc_raw(m * 2).unwrap();
            let invstd_dev = runtime.alloc_raw(m * 2).unwrap();
            let y_ref_dev = runtime.alloc_raw(m * post_n * 2).unwrap();
            let pre_fused_dev = runtime.alloc_raw(m * hidden * 2).unwrap();
            let y_fused_dev = runtime.alloc_raw(m * post_n * 2).unwrap();
            let zp_pre_dev = explicit_zp.then(|| runtime.alloc_raw(zp_pre.len()).unwrap());
            let zp_post_dev = explicit_zp.then(|| runtime.alloc_raw(zp_post.len()).unwrap());

            // SAFETY: device buffers exactly cover their source slices.
            unsafe {
                runtime.htod(as_bytes(&activation), activation_dev).unwrap();
                runtime.htod(&packed_pre, packed_pre_dev).unwrap();
                runtime.htod(as_bytes(&scales_pre), scales_pre_dev).unwrap();
                runtime.htod(&packed_post, packed_post_dev).unwrap();
                runtime
                    .htod(as_bytes(&scales_post), scales_post_dev)
                    .unwrap();
                runtime.htod(as_bytes(&residual), residual_dev).unwrap();
                runtime.htod(&gamma_bytes, gamma_dev).unwrap();
                runtime.htod(as_bytes(&bias_post), bias_post_dev).unwrap();
                if let Some(dev) = zp_pre_dev {
                    runtime.htod(&zp_pre, dev).unwrap();
                }
                if let Some(dev) = zp_post_dev {
                    runtime.htod(&zp_post, dev).unwrap();
                }
            }

            // Tensor descriptors.
            let pre_a_shape = [m, pre_k];
            let pre_a_strides = [pre_k as i64, 1];
            let pre_b_shape = [hidden, pre_k_blocks, blob_size];
            let pre_b_strides = [(pre_k_blocks * blob_size) as i64, blob_size as i64, 1];
            let pre_scales_shape = [hidden, pre_k_blocks];
            let pre_scales_strides = [pre_k_blocks as i64, 1];
            let hidden_shape = [m, hidden];
            let hidden_strides = [hidden as i64, 1];
            let gamma_shape = [hidden];
            let gamma_strides = [1i64];
            let post_b_shape = [post_n, post_k_blocks, blob_size];
            let post_b_strides = [(post_k_blocks * blob_size) as i64, blob_size as i64, 1];
            let post_scales_shape = [post_n, post_k_blocks];
            let post_scales_strides = [post_k_blocks as i64, 1];
            let post_bias_shape = [post_n];
            let post_bias_strides = [1i64];
            let y_shape = [m, post_n];
            let y_strides = [post_n as i64, 1];
            let stat_shape = [m];
            let stat_strides = [1i64];

            let activation_view = TensorView::new(
                device_ptr(activation_dev),
                DataType::Float16,
                &pre_a_shape,
                &pre_a_strides,
                device,
            );
            let packed_pre_view = TensorView::new(
                device_ptr(packed_pre_dev),
                DataType::Uint8,
                &pre_b_shape,
                &pre_b_strides,
                device,
            );
            let scales_pre_view = TensorView::new(
                device_ptr(scales_pre_dev),
                DataType::Float16,
                &pre_scales_shape,
                &pre_scales_strides,
                device,
            );
            let packed_post_view = TensorView::new(
                device_ptr(packed_post_dev),
                DataType::Uint8,
                &post_b_shape,
                &post_b_strides,
                device,
            );
            let scales_post_view = TensorView::new(
                device_ptr(scales_post_dev),
                DataType::Float16,
                &post_scales_shape,
                &post_scales_strides,
                device,
            );
            let residual_view = TensorView::new(
                device_ptr(residual_dev),
                DataType::Float16,
                &hidden_shape,
                &hidden_strides,
                device,
            );
            let gamma_view = TensorView::new(
                device_ptr(gamma_dev),
                gamma_dtype,
                &gamma_shape,
                &gamma_strides,
                device,
            );
            let bias_post_view = TensorView::new(
                device_ptr(bias_post_dev),
                DataType::Float16,
                &post_bias_shape,
                &post_bias_strides,
                device,
            );
            let matmul_out_view = TensorView::new(
                device_ptr(matmul_out_dev),
                DataType::Float16,
                &hidden_shape,
                &hidden_strides,
                device,
            );
            let normalized_input_view = TensorView::new(
                device_ptr(normalized_dev),
                DataType::Float16,
                &hidden_shape,
                &hidden_strides,
                device,
            );
            let pre_fused_input_view = TensorView::new(
                device_ptr(pre_fused_dev),
                DataType::Float16,
                &hidden_shape,
                &hidden_strides,
                device,
            );
            // Asymmetric zero-point views (int8 only), one byte per block.
            let pre_zp_shape = [hidden, pre_k_blocks];
            let pre_zp_strides = [pre_k_blocks as i64, 1];
            let post_zp_shape = [post_n, post_k_blocks];
            let post_zp_strides = [post_k_blocks as i64, 1];
            let zp_pre_view = zp_pre_dev.map(|dev| {
                TensorView::new(
                    device_ptr(dev),
                    DataType::Uint8,
                    &pre_zp_shape,
                    &pre_zp_strides,
                    device,
                )
            });
            let zp_post_view = zp_post_dev.map(|dev| {
                TensorView::new(
                    device_ptr(dev),
                    DataType::Uint8,
                    &post_zp_shape,
                    &post_zp_strides,
                    device,
                )
            });

            let make_kernel = |k: usize, n: usize, fold: bool, rmsnorm: bool| MatMulNBitsKernel {
                runtime: runtime.clone(),
                k,
                n,
                bits,
                block_size,
                accuracy_level: 4,
                accuracy4_workspace: None,
                fold_bias_post_round: fold,
                gate_up_swiglu: false,
                decomposed_silu: false,
                rmsnorm_prologue: rmsnorm,
                rmsnorm_epsilon: epsilon,
                last_call_capture_safe: AtomicBool::new(false),
                bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
                bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
            };

            // ── Reference: preceding GEMV → skip_rmsnorm → following GEMV ──
            let preceding_ref = make_kernel(pre_k, hidden, false, false);
            {
                let mut matmul_out = TensorMut::new(
                    device_ptr_mut(matmul_out_dev),
                    DataType::Float16,
                    &hidden_shape,
                    &hidden_strides,
                    device,
                );
                preceding_ref
                    .run(
                        &{
                            let mut inputs =
                                vec![activation_view, packed_pre_view, scales_pre_view];
                            if let Some(zp) = zp_pre_view {
                                inputs.push(zp);
                            }
                            inputs
                        },
                        std::slice::from_mut(&mut matmul_out),
                        None,
                    )
                    .unwrap();
            }

            let mut skip_node = Node::new(
                onnx_runtime_ir::NodeId(0),
                "SkipSimplifiedLayerNormalization",
                Vec::new(),
                Vec::new(),
            );
            skip_node
                .attributes
                .insert("epsilon".into(), onnx_runtime_ir::Attribute::Float(epsilon));
            let skip_kernel = crate::kernels::normalization::SkipSimplifiedLayerNormFactory {
                runtime: runtime.clone(),
            }
            .create(&skip_node, &[])
            .unwrap();
            {
                let normalized = TensorMut::new(
                    device_ptr_mut(normalized_dev),
                    DataType::Float16,
                    &hidden_shape,
                    &hidden_strides,
                    device,
                );
                let mean = TensorMut::new(
                    device_ptr_mut(mean_dev),
                    DataType::Float16,
                    &stat_shape,
                    &stat_strides,
                    device,
                );
                let invstd = TensorMut::new(
                    device_ptr_mut(invstd_dev),
                    DataType::Float16,
                    &stat_shape,
                    &stat_strides,
                    device,
                );
                let sum = TensorMut::new(
                    device_ptr_mut(sum_dev),
                    DataType::Float16,
                    &hidden_shape,
                    &hidden_strides,
                    device,
                );
                skip_kernel
                    .execute(
                        &[matmul_out_view, residual_view, gamma_view],
                        &mut [normalized, mean, invstd, sum],
                    )
                    .unwrap();
            }

            let following_ref = make_kernel(hidden, post_n, false, false);
            {
                let mut y_ref = TensorMut::new(
                    device_ptr_mut(y_ref_dev),
                    DataType::Float16,
                    &y_shape,
                    &y_strides,
                    device,
                );
                let mut inputs = vec![normalized_input_view, packed_post_view, scales_post_view];
                if zp_post_view.is_some() || following_bias {
                    inputs.push(zp_post_view.unwrap_or(TensorView::absent(DataType::Uint8)));
                }
                if following_bias {
                    inputs.push(TensorView::absent(DataType::Int32));
                    inputs.push(bias_post_view);
                }
                following_ref
                    .run(&inputs, std::slice::from_mut(&mut y_ref), None)
                    .unwrap();
            }

            // ── Fused: residual epilogue in preceding, norm prologue in following ──
            let preceding_fused = make_kernel(pre_k, hidden, true, false);
            {
                let mut pre_fused = TensorMut::new(
                    device_ptr_mut(pre_fused_dev),
                    DataType::Float16,
                    &hidden_shape,
                    &hidden_strides,
                    device,
                );
                preceding_fused
                    .run(
                        &[
                            activation_view,
                            packed_pre_view,
                            scales_pre_view,
                            zp_pre_view.unwrap_or(TensorView::absent(DataType::Uint8)),
                            TensorView::absent(DataType::Int32),
                            residual_view,
                        ],
                        std::slice::from_mut(&mut pre_fused),
                        None,
                    )
                    .unwrap();
            }

            let following_fused = make_kernel(hidden, post_n, false, true);
            {
                let mut y_fused = TensorMut::new(
                    device_ptr_mut(y_fused_dev),
                    DataType::Float16,
                    &y_shape,
                    &y_strides,
                    device,
                );
                let mut inputs = vec![
                    pre_fused_input_view,
                    packed_post_view,
                    scales_post_view,
                    zp_post_view.unwrap_or(TensorView::absent(DataType::Uint8)),
                    TensorView::absent(DataType::Int32),
                ];
                if following_bias {
                    inputs.push(bias_post_view);
                } else {
                    inputs.push(TensorView::absent(DataType::Float16));
                }
                inputs.push(gamma_view);
                following_fused
                    .run(&inputs, std::slice::from_mut(&mut y_fused), None)
                    .unwrap();
            }

            runtime.synchronize().unwrap();

            let mut sum_host = vec![f16::ZERO; m * hidden];
            let mut pre_fused_host = vec![f16::ZERO; m * hidden];
            let mut y_ref_host = vec![f16::ZERO; m * post_n];
            let mut y_fused_host = vec![f16::ZERO; m * post_n];
            // SAFETY: host buffers match their device sources.
            unsafe {
                runtime.dtoh(as_bytes_mut(&mut sum_host), sum_dev).unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut pre_fused_host), pre_fused_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut y_ref_host), y_ref_dev)
                    .unwrap();
                runtime
                    .dtoh(as_bytes_mut(&mut y_fused_host), y_fused_dev)
                    .unwrap();
                for buffer in [
                    activation_dev,
                    packed_pre_dev,
                    scales_pre_dev,
                    packed_post_dev,
                    scales_post_dev,
                    residual_dev,
                    gamma_dev,
                    bias_post_dev,
                    matmul_out_dev,
                    normalized_dev,
                    sum_dev,
                    mean_dev,
                    invstd_dev,
                    y_ref_dev,
                    pre_fused_dev,
                    y_fused_dev,
                ] {
                    runtime.free_raw(buffer).unwrap();
                }
                for buffer in [zp_pre_dev, zp_post_dev].into_iter().flatten() {
                    runtime.free_raw(buffer).unwrap();
                }
            }

            // The preceding fused output is the residual sum (input + skip).
            for index in 0..m * hidden {
                assert_eq!(
                    pre_fused_host[index].to_bits(),
                    sum_host[index].to_bits(),
                    "residual epilogue diverged from skip_rmsnorm sum at M={m}, \
                     following_bias={following_bias}, token={}, column={}",
                    index / hidden,
                    index % hidden
                );
            }
            // The fused projection matches the three-op sequence. It is normally
            // bit-identical, EXCEPT two fp-reassociation sources make it merely
            // near-equal:
            //   (1) the asymmetric int8-zp M=1 case, where the three-op
            //       reference's standalone int8 GEMV routes to the split-K entry
            //       (K % 256 == 0, grid-starved), reordering the fp32 block-sum
            //       across K_SPLIT cooperating warps; and
            //   (2) any decode-shaped norm (num_groups <=
            //       SKIP_RMSNORM_BLOCK_MAX_GROUPS), where the reference's
            //       standalone SkipSimplifiedLayerNorm routes to the multi-warp
            //       `skip_rmsnorm_f16_block_half4` variant, which reduces the
            //       fp32 sum-of-squares in block-tree order while the fused norm
            //       prologue keeps the single-warp order.
            // Both are near-equal valid computations, so those paths are
            // validated to a tight magnitude-relative tolerance (which still
            // catches a dropped zero point or fp16 accumulation, both of which
            // diverge grossly) instead of byte-identity.
            let splitk_path = bits == 8 && explicit_zp && m == 1;
            let norm_block_reference =
                m as u32 <= crate::kernels::normalization::SKIP_RMSNORM_BLOCK_MAX_GROUPS;
            let near_equal = splitk_path || norm_block_reference;
            if near_equal {
                let mut max_abs = 0.0f32;
                let mut worst = 0.0f32;
                for index in 0..m * post_n {
                    let fused = y_fused_host[index].to_f32();
                    let reference = y_ref_host[index].to_f32();
                    assert!(
                        fused.is_finite(),
                        "near-equal fused int8-zp GEMV produced a non-finite output at M={m}, \
                         following_bias={following_bias}, column={}",
                        index % post_n
                    );
                    max_abs = max_abs.max(reference.abs());
                    worst = worst.max((fused - reference).abs());
                }
                let bound = (max_abs * 2e-3).max(1e-3);
                assert!(
                    worst < bound,
                    "fused norm prologue diverged (beyond fp reassociation) from \
                     skip_rmsnorm + GEMV at \
                     M={m}, following_bias={following_bias}: \
                     max_abs_diff={worst:.3e} bound={bound:.3e}"
                );
            } else {
                for index in 0..m * post_n {
                    assert_eq!(
                        y_fused_host[index].to_bits(),
                        y_ref_host[index].to_bits(),
                        "fused norm prologue diverged from skip_rmsnorm + GEMV at M={m}, \
                         following_bias={following_bias}, token={}, column={}",
                        index / post_n,
                        index % post_n
                    );
                }
            }
        }
    }

    /// The block-32 split-K decode GEMV narrows its fp16 epilogue straight into
    /// the caller's bf16 tensor instead of writing an fp16 staging buffer that
    /// `run_bf16` would then cast. That must be bit-identical to the staged
    /// route, because greedy decoding is only stable if every token is.
    ///
    /// The reference here is deliberately the staged route itself: run the fp16
    /// GEMV into an fp16 buffer, then `cast_half` it to bf16. That is exactly
    /// what the direct store replaces, and it is why the kernel rounds
    /// fp32 -> fp16 -> bf16 rather than fp32 -> bf16.
    ///
    /// The counter assertion is load-bearing. Without it a routing change that
    /// silently fell back to staging would still compare equal, and this test
    /// would pass while covering nothing.
    #[test]
    fn bf16_direct_store_matches_staged_cast_bit_for_bit() {
        use half::bf16;

        let Some(runtime) = runtime() else {
            eprintln!("skipping MatMulNBits bf16 direct-store test: CUDA runtime unavailable");
            return;
        };

        // Shaped to select `matmul_nbits_gemv_f16_scales_f16_zp_splitk`: block 32,
        // fp16-family scales, per-block zero points, K a multiple of 256, and
        // both dimensions past the small-shape (64-thread) path.
        let k = 2048usize;
        let n = 2048usize;
        let block_size = 32usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2;

        let mut state = 0x51ed_2701_c0ff_ee11u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let activation_bf16: Vec<bf16> = (0..k).map(|_| bf16::from_f32(next())).collect();
        let packed: Vec<u8> = (0..n * k_blocks * blob_size)
            .map(|_| next().to_bits() as u8)
            .collect();
        let scales_bf16: Vec<bf16> = (0..n * k_blocks)
            .map(|_| bf16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
            .collect();
        let zero_points: Vec<u8> = (0..n * k_blocks.div_ceil(2))
            .map(|_| ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        let device = DeviceId::cuda(0);
        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, k_blocks.div_ceil(2)];
        let zp_strides = [k_blocks.div_ceil(2) as i64, 1];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];

        let act_dev = runtime.alloc_raw(activation_bf16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scales_bf16.len() * 2).unwrap();
        let zp_dev = runtime.alloc_raw(zero_points.len()).unwrap();
        let out_dev = runtime.alloc_raw(n * 2).unwrap();
        let act_f16_dev = runtime.alloc_raw(k * 2).unwrap();
        let scales_f16_dev = runtime.alloc_raw(scales_bf16.len() * 2).unwrap();
        let ref_f16_dev = runtime.alloc_raw(n * 2).unwrap();
        let ref_bf16_dev = runtime.alloc_raw(n * 2).unwrap();
        // SAFETY: device buffers exactly cover their source slices.
        unsafe {
            runtime.htod(as_bytes(&activation_bf16), act_dev).unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scales_bf16), scales_dev).unwrap();
            runtime.htod(&zero_points, zp_dev).unwrap();
        }

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 0,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };

        let bf16_inputs = vec![
            TensorView::new(
                device_ptr(act_dev),
                DataType::BFloat16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                DataType::BFloat16,
                &scales_shape,
                &scales_strides,
                device,
            ),
            TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            ),
        ];

        let before = BF16_DIRECT_OUT_STORES.load(Ordering::Relaxed);
        let mut got_out = [TensorMut::new(
            device_ptr_mut(out_dev),
            DataType::BFloat16,
            &y_shape,
            &y_strides,
            device,
        )];
        kernel.run(&bf16_inputs, &mut got_out, None).unwrap();
        runtime.synchronize().unwrap();
        let direct_stores = BF16_DIRECT_OUT_STORES.load(Ordering::Relaxed) - before;

        // Reference: the staged route this replaces.
        super::super::cast::launch_cast_raw(
            &runtime,
            cuptr(act_dev as *const c_void),
            DataType::BFloat16,
            act_f16_dev,
            DataType::Float16,
            k,
        )
        .unwrap();
        super::super::cast::launch_cast_raw(
            &runtime,
            cuptr(scales_dev as *const c_void),
            DataType::BFloat16,
            scales_f16_dev,
            DataType::Float16,
            scales_bf16.len(),
        )
        .unwrap();
        let ref_inputs = vec![
            TensorView::new(
                device_ptr(act_f16_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_f16_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            ),
            TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            ),
        ];
        let mut ref_f16 = [TensorMut::new(
            device_ptr_mut(ref_f16_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        )];
        kernel.run(&ref_inputs, &mut ref_f16, None).unwrap();
        super::super::cast::launch_cast_raw(
            &runtime,
            ref_f16_dev,
            DataType::Float16,
            ref_bf16_dev,
            DataType::BFloat16,
            n,
        )
        .unwrap();
        runtime.synchronize().unwrap();

        let mut got = vec![bf16::ZERO; n];
        let mut want = vec![bf16::ZERO; n];
        // SAFETY: each host buffer matches its device source.
        unsafe {
            runtime.dtoh(as_bytes_mut(&mut got), out_dev).unwrap();
            runtime.dtoh(as_bytes_mut(&mut want), ref_bf16_dev).unwrap();
            for buffer in [
                act_dev,
                packed_dev,
                scales_dev,
                zp_dev,
                out_dev,
                act_f16_dev,
                scales_f16_dev,
                ref_f16_dev,
                ref_bf16_dev,
            ] {
                runtime.free_raw(buffer).unwrap();
            }
        }

        assert_eq!(
            direct_stores, 1,
            "the bf16 run did not take the direct-store path, so this test would \
             compare the staged route against itself"
        );
        for index in 0..n {
            assert_eq!(
                got[index].to_bits(),
                want[index].to_bits(),
                "direct bf16 store diverged from the staged cast at column {index}"
            );
        }
    }

    /// The BFloat16 activation path caches its Float16-staged **constant** scales
    /// across calls (see [`Bf16ConstCache`]) instead of re-casting them every
    /// step. This must stay bit-identical to inline per-call staging: converting
    /// BFloat16 -> Float16 yields the same bits whether done once or repeatedly,
    /// and the tuned fp16 GEMV then reads identical scales. This test runs the
    /// bf16 path twice (cache miss, then cache hit) and compares it, bit for bit,
    /// to a reference that stages every bf16 input to Float16 inline.
    #[test]
    fn bf16_scale_cache_is_bit_exact_to_inline_staging() {
        use half::bf16;

        let Some(runtime) = runtime() else {
            eprintln!("skipping MatMulNBits bf16 scale-cache test: CUDA runtime unavailable");
            return;
        };

        let k = 256usize;
        let n = 64usize;
        let block_size = 32usize;
        let k_blocks = k / block_size;
        let blob_size = block_size / 2; // int4: two weights per byte

        let mut state = 0x0bad_c0de_dead_beefu64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        let activation_bf16: Vec<bf16> = (0..k).map(|_| bf16::from_f32(next())).collect();
        let packed: Vec<u8> = (0..n * k_blocks * blob_size)
            .map(|_| next().to_bits() as u8)
            .collect();
        let scales_bf16: Vec<bf16> = (0..n * k_blocks)
            .map(|_| bf16::from_f32(0.015 + 0.01 * (next() * 0.5 + 0.5)))
            .collect();
        let zero_points: Vec<u8> = (0..n * k_blocks.div_ceil(2))
            .map(|_| ((next() * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        let device = DeviceId::cuda(0);
        let a_shape = [1usize, k];
        let a_strides = [k as i64, 1];
        let b_shape = [n, k_blocks, blob_size];
        let b_strides = [(k_blocks * blob_size) as i64, blob_size as i64, 1];
        let scales_shape = [n, k_blocks];
        let scales_strides = [k_blocks as i64, 1];
        let zp_shape = [n, k_blocks.div_ceil(2)];
        let zp_strides = [k_blocks.div_ceil(2) as i64, 1];
        let y_shape = [1usize, n];
        let y_strides = [n as i64, 1];

        let act_dev = runtime.alloc_raw(activation_bf16.len() * 2).unwrap();
        let packed_dev = runtime.alloc_raw(packed.len()).unwrap();
        let scales_dev = runtime.alloc_raw(scales_bf16.len() * 2).unwrap();
        let zp_dev = runtime.alloc_raw(zero_points.len()).unwrap();
        let out1_dev = runtime.alloc_raw(n * 2).unwrap();
        let out2_dev = runtime.alloc_raw(n * 2).unwrap();
        // SAFETY: device buffers exactly cover their source slices.
        unsafe {
            runtime.htod(as_bytes(&activation_bf16), act_dev).unwrap();
            runtime.htod(&packed, packed_dev).unwrap();
            runtime.htod(as_bytes(&scales_bf16), scales_dev).unwrap();
            runtime.htod(&zero_points, zp_dev).unwrap();
        }

        let inputs = vec![
            TensorView::new(
                device_ptr(act_dev),
                DataType::BFloat16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_dev),
                DataType::BFloat16,
                &scales_shape,
                &scales_strides,
                device,
            ),
            TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            ),
        ];

        let kernel = MatMulNBitsKernel {
            runtime: runtime.clone(),
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 0,
            accuracy4_workspace: None,
            fold_bias_post_round: false,
            gate_up_swiglu: false,
            decomposed_silu: false,
            rmsnorm_prologue: false,
            rmsnorm_epsilon: 1e-5,
            last_call_capture_safe: AtomicBool::new(false),
            bf16_scratch: Mutex::new(Bf16Scratch::new(runtime.clone())),
            bf16_const_cache: Mutex::new(Bf16ConstCache::new(runtime.clone())),
        };

        let mut out1 = [TensorMut::new(
            device_ptr_mut(out1_dev),
            DataType::BFloat16,
            &y_shape,
            &y_strides,
            device,
        )];
        kernel.run(&inputs, &mut out1, None).unwrap();
        let mut out2 = [TensorMut::new(
            device_ptr_mut(out2_dev),
            DataType::BFloat16,
            &y_shape,
            &y_strides,
            device,
        )];
        kernel.run(&inputs, &mut out2, None).unwrap();
        runtime.synchronize().unwrap();

        // Reference: stage every bf16 input to Float16 inline (no cache) and run
        // the fp16 GEMV directly, then narrow to BFloat16 exactly as run_bf16 does.
        let act_f16_dev = runtime.alloc_raw(k * 2).unwrap();
        let scales_f16_dev = runtime.alloc_raw(scales_bf16.len() * 2).unwrap();
        let ref_f16_dev = runtime.alloc_raw(n * 2).unwrap();
        let ref_bf16_dev = runtime.alloc_raw(n * 2).unwrap();
        super::super::cast::launch_cast_raw(
            &runtime,
            cuptr(act_dev as *const c_void),
            DataType::BFloat16,
            act_f16_dev,
            DataType::Float16,
            k,
        )
        .unwrap();
        super::super::cast::launch_cast_raw(
            &runtime,
            cuptr(scales_dev as *const c_void),
            DataType::BFloat16,
            scales_f16_dev,
            DataType::Float16,
            scales_bf16.len(),
        )
        .unwrap();
        let ref_inputs = vec![
            TensorView::new(
                device_ptr(act_f16_dev),
                DataType::Float16,
                &a_shape,
                &a_strides,
                device,
            ),
            TensorView::new(
                device_ptr(packed_dev),
                DataType::Uint8,
                &b_shape,
                &b_strides,
                device,
            ),
            TensorView::new(
                device_ptr(scales_f16_dev),
                DataType::Float16,
                &scales_shape,
                &scales_strides,
                device,
            ),
            TensorView::new(
                device_ptr(zp_dev),
                DataType::Uint8,
                &zp_shape,
                &zp_strides,
                device,
            ),
        ];
        let mut ref_f16 = [TensorMut::new(
            device_ptr_mut(ref_f16_dev),
            DataType::Float16,
            &y_shape,
            &y_strides,
            device,
        )];
        kernel.run(&ref_inputs, &mut ref_f16, None).unwrap();
        super::super::cast::launch_cast_raw(
            &runtime,
            ref_f16_dev,
            DataType::Float16,
            ref_bf16_dev,
            DataType::BFloat16,
            n,
        )
        .unwrap();
        runtime.synchronize().unwrap();

        let mut got1 = vec![bf16::ZERO; n];
        let mut got2 = vec![bf16::ZERO; n];
        let mut want = vec![bf16::ZERO; n];
        // SAFETY: each host buffer matches its device source.
        unsafe {
            runtime.dtoh(as_bytes_mut(&mut got1), out1_dev).unwrap();
            runtime.dtoh(as_bytes_mut(&mut got2), out2_dev).unwrap();
            runtime.dtoh(as_bytes_mut(&mut want), ref_bf16_dev).unwrap();
            for buffer in [
                act_dev,
                packed_dev,
                scales_dev,
                zp_dev,
                out1_dev,
                out2_dev,
                act_f16_dev,
                scales_f16_dev,
                ref_f16_dev,
                ref_bf16_dev,
            ] {
                runtime.free_raw(buffer).unwrap();
            }
        }

        for index in 0..n {
            assert_eq!(
                got1[index].to_bits(),
                got2[index].to_bits(),
                "cached bf16 scale path is non-deterministic across steps at column {index}"
            );
            assert_eq!(
                got1[index].to_bits(),
                want[index].to_bits(),
                "cached bf16 scale path diverged from inline staging at column {index}"
            );
        }
    }
}
