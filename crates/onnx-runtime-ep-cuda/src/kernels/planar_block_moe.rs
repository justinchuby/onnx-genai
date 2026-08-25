//! Exact-format CUDA **routed top-k MoE execution** for the DeepSeek-V4 planar
//! B2 weight formats (`block_fp8`, `fp4_planar`) — the mixed-projection routed
//! primitive stacked on the planar matmul slice ([`super::planar_block_decode`])
//! and the CPU planar oracle
//! ([`onnx_runtime_ep_cpu::kernels::planar_block_quant`]).
//!
//! ## What this slice delivers
//!
//! A **launched**, capture-safe routed-MoE pipeline that decodes per-expert
//! planar weights (packed bytes + a *separate* UE8M0 aux-scale bank) directly on
//! the device and runs the full DeepSeek/GLM expert path:
//! `route → fc1 (+ optional fc3 gate) → activate → fc2 → combine`. No host
//! mirror, dequantize-to-full-copy, or transcode runs in the launch path — the
//! expert-indexed planar linear kernel consumes the exact on-disk byte layout,
//! one CUDA block per `(route, output_feature)` task, decoding from the routed
//! expert's slice of the expert-major bank.
//!
//! The **only** format-specific step is the linear/GEMV: the routing, activation
//! (ReLU / tanh-GELU / SiLU / SwiGLU) and weighted combine kernels are
//! format-agnostic `f32` passes, so this primitive reuses
//! `block_quantized_moe`'s `bqmoe_route` / `bqmoe_activate` / `bqmoe_combine_f32`
//! **verbatim** (compiled from the same NVRTC module,
//! [`super::block_quantized_moe::MOE_MODULE`]) and adds only a new
//! `pbmoe_planar_linear_f32` kernel that embeds the planar decode device
//! functions from [`super::planar_block_decode::PLANAR_BLOCK_DECODE_CUH`]. The
//! routing/activation/combine arithmetic therefore stays single-source with the
//! interleaved-GGUF routed path.
//!
//! Every projection (`fc1`, `fc2`, `fc3`) may carry an independent planar format,
//! logical `[out, in]` geometry, block geometry (`bs0`/`bs1`), and per-expert
//! packed/scale strides, matching DeepSeek-V4 experts whose gate/up/down pack at
//! different qtypes.
//!
//! ## Capture safety
//!
//! [`warm_planar_moe`] compiles/loads every kernel signature (both the planar
//! linear and the reused route/activate/combine) **outside** any CUDA-graph
//! capture — NVRTC compile and module load synchronize the device. After it
//! returns, [`launch_planar_moe`] fetches cached functions and issues launches
//! on the EP stream with **no allocation, no host→device copy, no compile, and
//! no trailing host synchronization**, so a warmed fixed-shape routed MoE records
//! cleanly into a captured segment. (This is the key difference from the eager
//! `BlockQuantizedMoE` op kernel, whose trailing host sync makes it
//! capture-unsupported.) Ordering with a later device→host read is guaranteed by
//! the single in-order EP stream.
//!
//! ## Claim boundary (honest)
//!
//! This slice proves and advertises the routed-MoE **primitive** only:
//! [`planar_moe_capable_formats`] reports `block_fp8` / `fp4_planar` once the GPU
//! parity test passes on hardware. The op-level `BlockQuantizedMoE` node claim
//! gate in [`super::block_quantized_moe`] still **typed-rejects** these formats:
//! the current 9-input node ABI has no per-projection aux-scale *input* to carry
//! the planar UE8M0 banks, and that node ABI is co-designed with the unmerged
//! Mobius #602 / Deckard #593 exporter. Wiring the aux-scale node inputs — and
//! adding reserved-code validation (see [`super::planar_block_decode`]) — is the
//! explicit next slice. The onnx-runtime-ep-cpu `planar_block_quant` oracle owns
//! the op-level routed path until then. This primitive is the runtime capability
//! the exporter probes before it may emit a planar MoE node.

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Result};

use crate::error::driver_err;
use crate::kernels::block_quantized_matmul::decoder_prelude;
use crate::kernels::block_quantized_moe::{
    MOE_ACTIVATE_ENTRY, MOE_COMBINE_ENTRY, MOE_MODULE, MOE_ROUTE_ENTRY, moe_module_source,
};
use crate::kernels::planar_block_decode::{PLANAR_BLOCK_DECODE_CUH, PlanarLinearDims};
use crate::runtime::CudaRuntime;

/// NVRTC module cache key for the planar routed-MoE linear kernel.
pub(crate) const PLANAR_MOE_MODULE: &str = "planar_block_moe_v1";

/// Entry point of the expert-indexed planar linear kernel.
pub(crate) const PLANAR_MOE_LINEAR_ENTRY: &str = "pbmoe_planar_linear_f32";

/// Expert-indexed planar linear / GEMV. One CUDA block per `(route, out)` task:
/// selects the routed expert, slices its packed + UE8M0 scale banks by the
/// caller-supplied per-expert byte strides, decodes the planar weight
/// `W[out, in]` element by element (reusing the exact device decode functions
/// from `PLANAR_BLOCK_DECODE_CUH`), contracts it against the correct input row,
/// block-reduces in `f32`, and adds the per-`(expert, out)` bias. The
/// `input_rows_are_routes` flag mirrors `bqmoe_linear_f32`: `false` for the
/// `fc1`/`fc3` projections (input is one token row shared by that token's
/// `top_k` routes) and `true` for `fc2` (input is the per-route activation).
const PLANAR_MOE_KERNEL: &str = r#"
extern "C" __global__ void pbmoe_planar_linear_f32(
    const float* input,
    const int* selected_experts,
    const unsigned char* packed,
    const unsigned char* scale,
    const float* bias,
    float* output,
    const unsigned long long routes,
    const int input_rows_are_routes,
    const int top_k,
    const int out_features,
    const int in_features,
    const int format,
    const int bs0,
    const int bs1,
    const unsigned long long packed_expert_stride,
    const unsigned long long scale_expert_stride)
{
    const unsigned long long tasks = routes * (unsigned long long)out_features;
    for (unsigned long long task = blockIdx.x; task < tasks; task += gridDim.x) {
        const unsigned long long route = task / out_features;
        const int output_feature = (int)(task % out_features);
        const int expert = selected_experts[route];
        const unsigned long long input_row =
            input_rows_are_routes ? route : route / (unsigned long long)top_k;
        const unsigned long long input_base =
            input_row * (unsigned long long)in_features;
        const unsigned char* expert_packed =
            packed + (unsigned long long)expert * packed_expert_stride;
        const unsigned char* expert_scale =
            scale + (unsigned long long)expert * scale_expert_stride;
        float value = 0.0f;
        for (int depth = (int)threadIdx.x; depth < in_features;
             depth += (int)blockDim.x) {
            const float w = (format == 0)
                ? planar_bf8_element(
                    expert_packed, expert_scale, out_features, in_features,
                    bs0, bs1, output_feature, depth)
                : planar_fp4_element(
                    expert_packed, expert_scale, out_features, in_features,
                    output_feature, depth);
            value += input[input_base + depth] * w;
        }
        value = block_sum(value);
        if (threadIdx.x == 0) {
            const unsigned long long bias_index =
                (unsigned long long)expert * out_features + output_feature;
            output[task] = value + (bias ? bias[bias_index] : 0.0f);
        }
        __syncthreads();
    }
}
"#;

/// Full NVRTC source for [`PLANAR_MOE_MODULE`]: the shared decoder prelude (for
/// `warp_sum`/`block_sum`), the planar decode device functions, and the
/// expert-indexed planar linear kernel.
fn planar_moe_module_source() -> String {
    let mut source = decoder_prelude();
    source.push_str(PLANAR_BLOCK_DECODE_CUH);
    source.push_str(PLANAR_MOE_KERNEL);
    source
}

fn kernel_err(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep planar moe: {}", message.into()))
}

/// One projection (`fc1`, `fc2`, or `fc3`) of a planar routed MoE: a per-expert
/// planar weight of logical `[out_features, in_features]` plus its UE8M0 block
/// geometry. Every expert in the bank shares this geometry; the per-expert
/// packed/scale byte strides are derived from it.
#[derive(Clone, Copy, Debug)]
pub struct PlanarMoeProjection {
    /// [`PLANAR_FORMAT_BLOCK_FP8`] or [`PLANAR_FORMAT_FP4_PLANAR`].
    pub format: i32,
    /// Contraction dimension (`K = in_features`).
    pub in_features: usize,
    /// Output dimension (`N = out_features`).
    pub out_features: usize,
    /// Row block size of the 2D `block_fp8` scale (`bs0`); ignored by `fp4_planar`.
    pub bs0: usize,
    /// Column block size of the 2D `block_fp8` scale (`bs1`); ignored by `fp4_planar`.
    pub bs1: usize,
}

impl PlanarMoeProjection {
    fn dims(&self) -> PlanarLinearDims {
        // `m_rows` is irrelevant to per-expert packed/scale byte counts; use 1 so
        // the proven `PlanarLinearDims` geometry validator (odd/unaligned `fp4`,
        // zero `block_fp8` block, i32 overflow) covers this projection too.
        PlanarLinearDims {
            format: self.format,
            m_rows: 1,
            in_features: self.in_features,
            out_features: self.out_features,
            bs0: self.bs0,
            bs1: self.bs1,
        }
    }

    /// Exact packed-weight and aux-scale byte counts for **one** expert of this
    /// projection, typed-rejecting any geometry the kernel cannot decode.
    pub fn per_expert_bytes(&self) -> Result<(usize, usize)> {
        let lengths = self
            .dims()
            .expected_lengths()
            .map_err(|err| kernel_err(err.to_string()))?;
        Ok((lengths.packed_bytes, lengths.scale_bytes))
    }
}

/// Geometry of a planar routed top-k MoE. The activation family and its
/// parameters mirror `block_quantized_moe`'s `bqmoe_activate` kernel-id ABI
/// (`0 = ReLU`, `1 = tanh-GELU`, `2 = SiLU`, `3 = SwiGLU`, `4 = identity`).
#[derive(Clone, Copy, Debug)]
pub struct PlanarMoeDims {
    /// Token rows (`M`).
    pub rows: usize,
    /// Hidden size (`H`) — input width of `fc1`/`fc3`, output width of `fc2`.
    pub hidden: usize,
    /// Intermediate size (`I`) — activated width; `fc2` contraction.
    pub inter: usize,
    /// Number of routed experts.
    pub experts: usize,
    /// Experts selected per token (`top_k`).
    pub top_k: usize,
    /// `bqmoe_activate` kernel id.
    pub activation: i32,
    /// SwiGLU packing: `0` = split/none, `1` = fused gate/linear interleaved.
    pub swiglu_fusion: i32,
    /// SwiGLU sigmoid gain (`alpha`).
    pub activation_alpha: f32,
    /// SwiGLU linear bias (`beta`).
    pub activation_beta: f32,
    /// SwiGLU clamp limit.
    pub swiglu_limit: f32,
    /// Normalize the routed weights over the selected experts.
    pub normalize_routing_weights: bool,
    /// Gate/up projection (`in = hidden`).
    pub fc1: PlanarMoeProjection,
    /// Down projection (`in = inter`, `out = hidden`).
    pub fc2: PlanarMoeProjection,
    /// Optional separate up projection for gated activations (`in = hidden`,
    /// `out = inter`). When present the `fc1` output is the gate and `fc3` the
    /// linear branch.
    pub fc3: Option<PlanarMoeProjection>,
}

impl PlanarMoeDims {
    /// Number of `(token, slot)` routes = `rows * top_k`.
    pub fn routes(&self) -> usize {
        self.rows * self.top_k
    }

    /// Full `fc1` output width: `2 * inter` when SwiGLU is fused (gate + linear
    /// interleaved), else `inter`.
    pub fn fc1_out(&self) -> usize {
        if self.swiglu_fusion != 0 {
            self.inter * 2
        } else {
            self.inter
        }
    }
}

/// Validate a planar routed MoE's geometry and every supplied bank/workspace
/// byte length against the exact extents the kernels require. This is the
/// host-side aux/OOB guard (ragged expert banks, wrong projection widths,
/// truncated scales, mis-sized workspace, missing gate) that must pass before
/// any launch. Any mismatch is a typed rejection — never a dense-expert
/// fallback.
#[allow(clippy::too_many_arguments)]
pub fn validate_planar_moe(
    dims: &PlanarMoeDims,
    fc1_packed_bytes: usize,
    fc1_scale_bytes: usize,
    fc1_bias_elems: Option<usize>,
    fc2_packed_bytes: usize,
    fc2_scale_bytes: usize,
    fc2_bias_elems: Option<usize>,
    fc3_banks: Option<(usize, usize, Option<usize>)>,
) -> Result<()> {
    if dims.rows == 0 || dims.hidden == 0 || dims.inter == 0 || dims.experts == 0 {
        return Err(kernel_err(format!(
            "non-positive dims rows={} hidden={} inter={} experts={}",
            dims.rows, dims.hidden, dims.inter, dims.experts
        )));
    }
    if dims.top_k == 0 || dims.top_k > dims.experts {
        return Err(kernel_err(format!(
            "requires 0 < top_k <= experts, got top_k={} experts={}",
            dims.top_k, dims.experts
        )));
    }
    if !(0..=4).contains(&dims.activation) {
        return Err(kernel_err(format!(
            "unknown activation id {}",
            dims.activation
        )));
    }
    if dims.swiglu_fusion > 2 {
        return Err(kernel_err(format!(
            "swiglu_fusion must be 0, 1, or 2, got {}",
            dims.swiglu_fusion
        )));
    }
    // SwiGLU fusion packs gate+linear into fc1 and is only defined for the
    // SwiGLU activation (id 3). Any other activation with fusion set would make
    // `fc1_out()` demand a 2*inter fc1 while the reused activate kernel indexes
    // it as inter-wide (base = route*inter), silently reading misaligned data.
    if dims.swiglu_fusion != 0 && dims.activation != 3 {
        return Err(kernel_err(format!(
            "swiglu_fusion={} is only valid with the SwiGLU activation (id 3), got activation {}",
            dims.swiglu_fusion, dims.activation
        )));
    }
    // SwiGLU needs both a gate and a linear projection. The only correct sources
    // are fused packing (swiglu_fusion != 0, gate+linear in a 2*inter fc1) or a
    // separate fc3 gate (swiglu_fusion == 0, fc1=gate, fc3=linear). The remaining
    // combination — SwiGLU, no fusion, no fc3 — is the unimplemented "split"
    // layout: the reused activate kernel reads `fc1[base + inter + feature]`
    // (base = route*inter), i.e. one intermediate row past the inter-wide fc1
    // buffer, which is out of bounds on the final route. Reject rather than
    // mis-execute or read OOB.
    if dims.activation == 3 && dims.swiglu_fusion == 0 && dims.fc3.is_none() {
        return Err(kernel_err(
            "SwiGLU (activation 3) with swiglu_fusion=0 requires a separate fc3 gate; \
             the split gate-in-fc1 layout is unsupported (would read past the fc1 buffer)",
        ));
    }

    // Projection widths must line up with the pipeline's intermediate buffers.
    if dims.fc1.in_features != dims.hidden {
        return Err(kernel_err(format!(
            "fc1 in={} must equal hidden={}",
            dims.fc1.in_features, dims.hidden
        )));
    }
    if dims.fc1.out_features != dims.fc1_out() {
        return Err(kernel_err(format!(
            "fc1 out={} must equal fc1_out={} (inter={}, swiglu_fusion={})",
            dims.fc1.out_features,
            dims.fc1_out(),
            dims.inter,
            dims.swiglu_fusion
        )));
    }
    if dims.fc2.in_features != dims.inter {
        return Err(kernel_err(format!(
            "fc2 in={} must equal inter={}",
            dims.fc2.in_features, dims.inter
        )));
    }
    if dims.fc2.out_features != dims.hidden {
        return Err(kernel_err(format!(
            "fc2 out={} must equal hidden={}",
            dims.fc2.out_features, dims.hidden
        )));
    }

    validate_projection_banks(
        dims,
        &dims.fc1,
        "fc1",
        fc1_packed_bytes,
        fc1_scale_bytes,
        fc1_bias_elems,
    )?;
    validate_projection_banks(
        dims,
        &dims.fc2,
        "fc2",
        fc2_packed_bytes,
        fc2_scale_bytes,
        fc2_bias_elems,
    )?;

    match (dims.fc3.as_ref(), fc3_banks) {
        (Some(fc3), Some((packed, scale, bias))) => {
            if dims.swiglu_fusion != 0 {
                return Err(kernel_err(
                    "a separate fc3 gate is incompatible with fused SwiGLU (swiglu_fusion != 0)",
                ));
            }
            if !matches!(dims.activation, 2 | 3) {
                return Err(kernel_err(format!(
                    "a separate fc3 gate requires a gated activation (SiLU=2 or SwiGLU=3), got {}",
                    dims.activation
                )));
            }
            if fc3.in_features != dims.hidden {
                return Err(kernel_err(format!(
                    "fc3 in={} must equal hidden={}",
                    fc3.in_features, dims.hidden
                )));
            }
            if fc3.out_features != dims.inter {
                return Err(kernel_err(format!(
                    "fc3 out={} must equal inter={}",
                    fc3.out_features, dims.inter
                )));
            }
            validate_projection_banks(dims, fc3, "fc3", packed, scale, bias)?;
        }
        (None, None) => {}
        (Some(_), None) => {
            return Err(kernel_err("fc3 projection present but fc3 banks missing"));
        }
        (None, Some(_)) => {
            return Err(kernel_err("fc3 banks present but fc3 projection missing"));
        }
    }
    Ok(())
}

fn validate_projection_banks(
    dims: &PlanarMoeDims,
    projection: &PlanarMoeProjection,
    label: &str,
    packed_bytes: usize,
    scale_bytes: usize,
    bias_elems: Option<usize>,
) -> Result<()> {
    let (per_packed, per_scale) = projection.per_expert_bytes()?;
    let expected_packed = per_packed
        .checked_mul(dims.experts)
        .ok_or_else(|| kernel_err(format!("{label} packed bank byte count overflow")))?;
    let expected_scale = per_scale
        .checked_mul(dims.experts)
        .ok_or_else(|| kernel_err(format!("{label} scale bank byte count overflow")))?;
    if packed_bytes != expected_packed {
        return Err(kernel_err(format!(
            "{label} packed bank has {packed_bytes} bytes, expected experts*{per_packed} = {expected_packed}"
        )));
    }
    if scale_bytes != expected_scale {
        return Err(kernel_err(format!(
            "{label} scale bank has {scale_bytes} bytes, expected experts*{per_scale} = {expected_scale}"
        )));
    }
    if let Some(bias_elems) = bias_elems {
        let expected_bias = projection
            .out_features
            .checked_mul(dims.experts)
            .ok_or_else(|| kernel_err(format!("{label} bias element count overflow")))?;
        if bias_elems != expected_bias {
            return Err(kernel_err(format!(
                "{label} bias has {bias_elems} elements, expected experts*out = {expected_bias}"
            )));
        }
    }
    Ok(())
}

/// Device pointers for one planar routed-MoE launch. Every pointer is a live
/// device allocation the caller owns for the launch's duration; the workspace
/// pointers are caller-allocated so the launch itself allocates nothing (a
/// prerequisite for capture safety). A `0` bias/router-weight/fc3 pointer means
/// "absent" and is treated as inert by the kernels.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanarMoePtrs {
    /// `f32 [rows, hidden]` token activations.
    pub input: CUdeviceptr,
    /// `f32 [rows, experts]` router logits.
    pub router_logits: CUdeviceptr,
    /// `f32 [rows, experts]` pre-aggregated router weights, or `0` for softmax.
    pub router_weights: CUdeviceptr,

    /// `fc1` expert-major packed weight bank.
    pub fc1_packed: CUdeviceptr,
    /// `fc1` expert-major UE8M0 scale bank.
    pub fc1_scale: CUdeviceptr,
    /// `f32 [experts, fc1_out]` bias, or `0`.
    pub fc1_bias: CUdeviceptr,

    /// `fc2` expert-major packed weight bank.
    pub fc2_packed: CUdeviceptr,
    /// `fc2` expert-major UE8M0 scale bank.
    pub fc2_scale: CUdeviceptr,
    /// `f32 [experts, hidden]` bias, or `0`.
    pub fc2_bias: CUdeviceptr,

    /// `fc3` expert-major packed weight bank, or `0`.
    pub fc3_packed: CUdeviceptr,
    /// `fc3` expert-major UE8M0 scale bank, or `0`.
    pub fc3_scale: CUdeviceptr,
    /// `f32 [experts, inter]` bias, or `0`.
    pub fc3_bias: CUdeviceptr,

    /// `i32 [routes]` selected expert scratch.
    pub route_indices: CUdeviceptr,
    /// `f32 [routes]` selected weight scratch.
    pub route_weights: CUdeviceptr,
    /// `f32 [routes, fc1_out]` fc1 output scratch.
    pub fc1_output: CUdeviceptr,
    /// `f32 [routes, inter]` fc3 output scratch, or `0`.
    pub fc3_output: CUdeviceptr,
    /// `f32 [routes, inter]` activated scratch.
    pub activated: CUdeviceptr,
    /// `f32 [routes, hidden]` per-route output scratch.
    pub route_output: CUdeviceptr,

    /// `f32 [rows, hidden]` final combined output.
    pub output: CUdeviceptr,
}

/// Threads-per-block ceiling preference, mirroring `BlockQuantizedMoEKernel`.
fn preferred_threads(runtime: &CudaRuntime) -> u32 {
    let capabilities = runtime.capabilities();
    let preferred = if capabilities.compute_capability().0 >= 7 {
        256
    } else {
        128
    };
    preferred.min(capabilities.max_threads_per_block()).max(1)
}

/// Grid saturated to `16 * SM count`, mirroring `BlockQuantizedMoEKernel`.
fn saturating_grid(runtime: &CudaRuntime, units: u64) -> u32 {
    let capabilities = runtime.capabilities();
    let saturation = u64::from(capabilities.multiprocessor_count()).saturating_mul(16);
    let grid = units.min(saturation.max(1)).min(u64::from(u32::MAX)).max(1);
    grid as u32
}

/// One-thread-per-element pointwise launch config (route/activate/combine).
fn pointwise_config(runtime: &CudaRuntime, total: u64) -> LaunchConfig {
    let threads = preferred_threads(runtime);
    let blocks_needed = total.div_ceil(u64::from(threads)).max(1);
    LaunchConfig {
        grid_dim: (saturating_grid(runtime, blocks_needed), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Warm-compile every kernel the planar routed MoE launches — its own planar
/// linear plus the reused `bqmoe_route`/`bqmoe_activate`/`bqmoe_combine_f32` —
/// and pre-apply any dynamic shared-memory opt-in for the reduction launch.
///
/// Must run **outside** a CUDA-graph capture (NVRTC compile / module load
/// synchronize the device). After it returns, [`launch_planar_moe`] loads cached
/// functions only, so a warmed fixed-shape launch is capture-safe.
pub fn warm_planar_moe(runtime: &CudaRuntime) -> Result<()> {
    runtime.require_nvrtc_half_headers("planar moe")?;
    let linear = runtime.nvrtc_function(
        PLANAR_MOE_MODULE,
        &planar_moe_module_source(),
        PLANAR_MOE_LINEAR_ENTRY,
    )?;
    // Pre-apply any MAX_DYNAMIC_SHARED_SIZE opt-in outside capture. The
    // reduction needs 4 bytes/thread, comfortably below the default block
    // budget, so this is defence in depth rather than a required opt-in.
    runtime.reduction_launch_config(&linear, 1, preferred_threads(runtime), 4)?;
    for entry in [MOE_ROUTE_ENTRY, MOE_ACTIVATE_ENTRY, MOE_COMBINE_ENTRY] {
        runtime.nvrtc_function(MOE_MODULE, moe_module_source(), entry)?;
    }
    Ok(())
}

fn as_i32(label: &str, value: usize) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| kernel_err(format!("{label}={value} exceeds the i32 kernel ABI")))
}

#[allow(clippy::too_many_arguments)]
fn launch_planar_linear(
    runtime: &CudaRuntime,
    projection: &PlanarMoeProjection,
    input: CUdeviceptr,
    route_indices: CUdeviceptr,
    packed: CUdeviceptr,
    scale: CUdeviceptr,
    bias: CUdeviceptr,
    output: CUdeviceptr,
    routes: usize,
    top_k: usize,
    input_rows_are_routes: bool,
) -> Result<()> {
    let (per_packed, per_scale) = projection.per_expert_bytes()?;
    let function = runtime.nvrtc_function(
        PLANAR_MOE_MODULE,
        &planar_moe_module_source(),
        PLANAR_MOE_LINEAR_ENTRY,
    )?;
    let tasks = (routes as u64)
        .checked_mul(projection.out_features as u64)
        .ok_or_else(|| kernel_err("linear task count overflow"))?;
    let grid_x = saturating_grid(runtime, tasks);
    let config =
        runtime.reduction_launch_config(&function, grid_x, preferred_threads(runtime), 4)?;

    let routes_u64 = routes as u64;
    let input_rows_are_routes = i32::from(input_rows_are_routes);
    let top_k = as_i32("top_k", top_k)?;
    let out_features = as_i32("out_features", projection.out_features)?;
    let in_features = as_i32("in_features", projection.in_features)?;
    let format = projection.format;
    let bs0 = as_i32("bs0", projection.bs0)?;
    let bs1 = as_i32("bs1", projection.bs1)?;
    let packed_stride = per_packed as u64;
    let scale_stride = per_scale as u64;

    let stream = runtime.stream();
    let mut builder = stream.launch_builder(&function);
    builder
        .arg(&input)
        .arg(&route_indices)
        .arg(&packed)
        .arg(&scale)
        .arg(&bias)
        .arg(&output)
        .arg(&routes_u64)
        .arg(&input_rows_are_routes)
        .arg(&top_k)
        .arg(&out_features)
        .arg(&in_features)
        .arg(&format)
        .arg(&bs0)
        .arg(&bs1)
        .arg(&packed_stride)
        .arg(&scale_stride);
    // SAFETY: the scalar ABI matches `pbmoe_planar_linear_f32`; packed/scale
    // banks cover experts*per_expert bytes and the scratch buffers cover
    // routes*out_features, all validated by `validate_planar_moe`.
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|err| driver_err("launch planar MoE expert GEMV", err))
}

/// Launch the full planar routed top-k MoE pipeline on `runtime`'s EP stream:
/// `route → fc1 (+ fc3) → activate → fc2 → combine`.
///
/// Geometry is re-validated here (defence in depth); bank/workspace byte lengths
/// must be validated by the caller with [`validate_planar_moe`] before this
/// call. The launch issues **no** allocation, host→device copy, compile, or host
/// synchronization, so a warmed signature (see [`warm_planar_moe`]) records
/// cleanly into a CUDA-graph capture.
pub fn launch_planar_moe(
    runtime: &CudaRuntime,
    dims: &PlanarMoeDims,
    ptrs: &PlanarMoePtrs,
) -> Result<()> {
    if dims.rows == 0 || dims.hidden == 0 || dims.inter == 0 {
        return Ok(());
    }
    // Re-validate projection geometry (defence in depth); the caller validates
    // concrete bank byte lengths with `validate_planar_moe`.
    dims.fc1.per_expert_bytes()?;
    dims.fc2.per_expert_bytes()?;
    if let Some(fc3) = dims.fc3.as_ref() {
        fc3.per_expert_bytes()?;
    }
    let routes = dims.routes();

    // 1. Route: top-k selection + weights (telemetry pointers null / inert).
    let route_fn = runtime.nvrtc_function(MOE_MODULE, moe_module_source(), MOE_ROUTE_ENTRY)?;
    {
        let rows = dims.rows as u64;
        let experts = as_i32("experts", dims.experts)?;
        let top_k = as_i32("top_k", dims.top_k)?;
        let normalize = i32::from(dims.normalize_routing_weights);
        let telemetry_bitmap: CUdeviceptr = 0;
        let telemetry_header: CUdeviceptr = 0;
        let config = pointwise_config(runtime, rows);
        let stream = runtime.stream();
        let mut builder = stream.launch_builder(&route_fn);
        builder
            .arg(&ptrs.router_logits)
            .arg(&ptrs.router_weights)
            .arg(&ptrs.route_indices)
            .arg(&ptrs.route_weights)
            .arg(&rows)
            .arg(&experts)
            .arg(&top_k)
            .arg(&normalize)
            .arg(&telemetry_bitmap)
            .arg(&telemetry_header);
        // SAFETY: scratch buffers cover rows*top_k and the scalar ABI matches
        // `bqmoe_route`; telemetry pointers are null (disarmed / inert).
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch planar MoE routing", err))?;
    }

    // 2. FC1 (and optional FC3 gate): token rows shared across a token's routes.
    launch_planar_linear(
        runtime,
        &dims.fc1,
        ptrs.input,
        ptrs.route_indices,
        ptrs.fc1_packed,
        ptrs.fc1_scale,
        ptrs.fc1_bias,
        ptrs.fc1_output,
        routes,
        dims.top_k,
        false,
    )?;
    if let Some(fc3) = dims.fc3.as_ref() {
        launch_planar_linear(
            runtime,
            fc3,
            ptrs.input,
            ptrs.route_indices,
            ptrs.fc3_packed,
            ptrs.fc3_scale,
            ptrs.fc3_bias,
            ptrs.fc3_output,
            routes,
            dims.top_k,
            false,
        )?;
    }

    // 3. Activation: fc1 (+ fc3) -> activated[routes, inter].
    let activate_fn =
        runtime.nvrtc_function(MOE_MODULE, moe_module_source(), MOE_ACTIVATE_ENTRY)?;
    {
        let total = (routes as u64)
            .checked_mul(dims.inter as u64)
            .ok_or_else(|| kernel_err("activation element count overflow"))?;
        let fc3_output = if dims.fc3.is_some() {
            ptrs.fc3_output
        } else {
            0
        };
        let routes_u64 = routes as u64;
        let inter = as_i32("inter", dims.inter)?;
        let activation = dims.activation;
        let swiglu_fusion = dims.swiglu_fusion;
        let alpha = dims.activation_alpha;
        let beta = dims.activation_beta;
        let limit = dims.swiglu_limit;
        let config = pointwise_config(runtime, total);
        let stream = runtime.stream();
        let mut builder = stream.launch_builder(&activate_fn);
        builder
            .arg(&ptrs.fc1_output)
            .arg(&fc3_output)
            .arg(&ptrs.activated)
            .arg(&routes_u64)
            .arg(&inter)
            .arg(&activation)
            .arg(&swiglu_fusion)
            .arg(&alpha)
            .arg(&beta)
            .arg(&limit);
        // SAFETY: scratch buffers cover every routed intermediate element and
        // the ABI matches `bqmoe_activate`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch planar MoE activation", err))?;
    }

    // 4. FC2 down-projection: input rows are per-route activations.
    launch_planar_linear(
        runtime,
        &dims.fc2,
        ptrs.activated,
        ptrs.route_indices,
        ptrs.fc2_packed,
        ptrs.fc2_scale,
        ptrs.fc2_bias,
        ptrs.route_output,
        routes,
        dims.top_k,
        true,
    )?;

    // 5. Combine: weighted sum over each token's top_k routes.
    let combine_fn = runtime.nvrtc_function(MOE_MODULE, moe_module_source(), MOE_COMBINE_ENTRY)?;
    {
        let total = (dims.rows as u64)
            .checked_mul(dims.hidden as u64)
            .ok_or_else(|| kernel_err("output element count overflow"))?;
        let rows = dims.rows as u64;
        let hidden = as_i32("hidden", dims.hidden)?;
        let top_k = as_i32("top_k", dims.top_k)?;
        let config = pointwise_config(runtime, total);
        let stream = runtime.stream();
        let mut builder = stream.launch_builder(&combine_fn);
        builder
            .arg(&ptrs.route_output)
            .arg(&ptrs.route_weights)
            .arg(&ptrs.output)
            .arg(&rows)
            .arg(&hidden)
            .arg(&top_k);
        // SAFETY: routed output/weights cover rows*top_k, output covers
        // rows*hidden, and the ABI matches `bqmoe_combine_f32`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch planar MoE weighted combine", err))?;
    }
    Ok(())
}

/// Planar weight formats with a proven, launched routed top-k MoE kernel on this
/// build. These are the stable runtime capability strings the Mobius #602 /
/// Deckard #593 planar emitters probe before emitting a planar MoE node: the
/// routed path is only correct once the runtime advertises the matching format
/// here.
///
/// Scope: the routed-MoE **primitive** ([`launch_planar_moe`]), proven on device
/// against the CPU oracle. The op-level `BlockQuantizedMoE` node claim gate still
/// typed-rejects these formats until its per-projection aux-scale node inputs are
/// wired (see the module docs).
pub fn planar_moe_capable_formats() -> &'static [&'static str] {
    &["block_fp8", "fp4_planar"]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::planar_block_decode::{PLANAR_FORMAT_BLOCK_FP8, PLANAR_FORMAT_FP4_PLANAR};

    fn fp8(inp: usize, out: usize) -> PlanarMoeProjection {
        PlanarMoeProjection {
            format: PLANAR_FORMAT_BLOCK_FP8,
            in_features: inp,
            out_features: out,
            bs0: 128,
            bs1: 128,
        }
    }

    fn fp4(inp: usize, out: usize) -> PlanarMoeProjection {
        PlanarMoeProjection {
            format: PLANAR_FORMAT_FP4_PLANAR,
            in_features: inp,
            out_features: out,
            bs0: 1,
            bs1: FP4_MICROSCALE_BLOCK_TEST,
        }
    }

    const FP4_MICROSCALE_BLOCK_TEST: usize = 32;

    fn base_dims() -> PlanarMoeDims {
        PlanarMoeDims {
            rows: 3,
            hidden: 256,
            inter: 128,
            experts: 4,
            top_k: 2,
            activation: 0,
            swiglu_fusion: 0,
            activation_alpha: 1.0,
            activation_beta: 1.0,
            swiglu_limit: f32::INFINITY,
            normalize_routing_weights: true,
            fc1: fp8(256, 128),
            fc2: fp8(128, 256),
            fc3: None,
        }
    }

    fn per_expert(projection: &PlanarMoeProjection) -> (usize, usize) {
        projection.per_expert_bytes().unwrap()
    }

    fn banks(dims: &PlanarMoeDims, projection: &PlanarMoeProjection) -> (usize, usize) {
        let (p, s) = per_expert(projection);
        (p * dims.experts, s * dims.experts)
    }

    #[test]
    fn block_fp8_per_expert_bytes_match_layout() {
        // block_fp8: 1 byte/elem packed, ceil(out/bs0)*ceil(in/bs1) scale bytes.
        let projection = fp8(256, 128);
        let (packed, scale) = per_expert(&projection);
        assert_eq!(packed, 256 * 128);
        assert_eq!(scale, 128usize.div_ceil(128) * 256usize.div_ceil(128));
    }

    #[test]
    fn fp4_planar_per_expert_bytes_match_layout() {
        // fp4_planar: two nibbles/byte packed, one UE8M0 per 32 logical inputs.
        let projection = fp4(256, 128);
        let (packed, scale) = per_expert(&projection);
        assert_eq!(packed, 128 * (256 / 2));
        assert_eq!(scale, 128 * (256 / 32));
    }

    #[test]
    fn valid_geometry_accepts() {
        let dims = base_dims();
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        validate_planar_moe(
            &dims,
            fc1p,
            fc1s,
            Some(dims.fc1.out_features * dims.experts),
            fc2p,
            fc2s,
            None,
            None,
        )
        .expect("valid planar MoE geometry must be accepted");
    }

    #[test]
    fn mixed_projection_formats_accept() {
        // fc1 block_fp8 gate/up, fc2 fp4_planar down — real DeepSeek-style mix.
        let mut dims = base_dims();
        dims.fc2 = fp4(128, 256);
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        validate_planar_moe(&dims, fc1p, fc1s, None, fc2p, fc2s, None, None)
            .expect("mixed planar projection formats must be accepted");
    }

    #[test]
    fn ragged_packed_bank_is_typed_rejected() {
        let dims = base_dims();
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let err = validate_planar_moe(&dims, fc1p + 1, fc1s, None, fc2p, fc2s, None, None)
            .expect_err("a ragged fc1 packed bank must be rejected");
        assert!(format!("{err:?}").contains("fc1 packed bank"));
    }

    #[test]
    fn wrong_fc2_width_is_typed_rejected() {
        let mut dims = base_dims();
        dims.fc2 = fp8(128, 128); // out != hidden
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let err = validate_planar_moe(&dims, fc1p, fc1s, None, fc2p, fc2s, None, None)
            .expect_err("fc2 out != hidden must be rejected");
        assert!(format!("{err:?}").contains("fc2 out"));
    }

    #[test]
    fn fused_swiglu_requires_double_width_fc1() {
        let mut dims = base_dims();
        dims.activation = 3;
        dims.swiglu_fusion = 1;
        // fc1 out must be 2*inter for fused SwiGLU.
        assert_eq!(dims.fc1_out(), dims.inter * 2);
        dims.fc1 = fp8(256, 128); // wrong: only inter wide
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let err = validate_planar_moe(&dims, fc1p, fc1s, None, fc2p, fc2s, None, None)
            .expect_err("fused SwiGLU with inter-wide fc1 must be rejected");
        assert!(format!("{err:?}").contains("fc1 out"));
    }

    #[test]
    fn separate_fc3_gate_validates() {
        let mut dims = base_dims();
        dims.activation = 3;
        dims.fc3 = Some(fp8(256, 128));
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let fc3 = dims.fc3.unwrap();
        let (fc3p, fc3s) = banks(&dims, &fc3);
        validate_planar_moe(
            &dims,
            fc1p,
            fc1s,
            None,
            fc2p,
            fc2s,
            None,
            Some((fc3p, fc3s, None)),
        )
        .expect("separate fc3 SwiGLU gate must validate");
    }

    #[test]
    fn split_swiglu_without_gate_is_typed_rejected() {
        // SwiGLU, no fusion, no fc3: the buggy "split" layout the activate kernel
        // would index one intermediate row past the inter-wide fc1 buffer (OOB).
        let mut dims = base_dims();
        dims.activation = 3;
        dims.swiglu_fusion = 0;
        dims.fc3 = None;
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let err = validate_planar_moe(&dims, fc1p, fc1s, None, fc2p, fc2s, None, None)
            .expect_err("split SwiGLU without a gate must be rejected");
        assert!(format!("{err:?}").contains("requires a separate fc3 gate"));
    }

    #[test]
    fn swiglu_fusion_with_non_swiglu_activation_is_typed_rejected() {
        // Fusion is only meaningful for SwiGLU; a ReLU with fusion set would
        // demand a 2*inter fc1 the activate kernel reads as inter-wide.
        let mut dims = base_dims();
        dims.activation = 0; // ReLU
        dims.swiglu_fusion = 1;
        dims.fc1 = fp8(256, 256); // 2*inter to satisfy width, exercise the guard
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let err = validate_planar_moe(&dims, fc1p, fc1s, None, fc2p, fc2s, None, None)
            .expect_err("swiglu_fusion with a non-SwiGLU activation must be rejected");
        assert!(format!("{err:?}").contains("only valid with the SwiGLU activation"));
    }

    #[test]
    fn fc3_with_fused_swiglu_is_typed_rejected() {
        let mut dims = base_dims();
        dims.activation = 3;
        dims.swiglu_fusion = 1;
        dims.fc1 = fp8(256, 256); // 2*inter for fused
        dims.fc3 = Some(fp8(256, 128));
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let fc3 = dims.fc3.unwrap();
        let (fc3p, fc3s) = banks(&dims, &fc3);
        let err = validate_planar_moe(
            &dims,
            fc1p,
            fc1s,
            None,
            fc2p,
            fc2s,
            None,
            Some((fc3p, fc3s, None)),
        )
        .expect_err("fc3 gate + fused SwiGLU must be rejected");
        assert!(format!("{err:?}").contains("fused SwiGLU"));
    }

    #[test]
    fn top_k_greater_than_experts_is_typed_rejected() {
        let mut dims = base_dims();
        dims.top_k = 8; // > experts (4)
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let err = validate_planar_moe(&dims, fc1p, fc1s, None, fc2p, fc2s, None, None)
            .expect_err("top_k > experts must be rejected");
        assert!(format!("{err:?}").contains("top_k"));
    }

    #[test]
    fn capability_lists_both_planar_formats() {
        let formats = planar_moe_capable_formats();
        assert!(formats.contains(&"block_fp8"));
        assert!(formats.contains(&"fp4_planar"));
    }

    #[test]
    fn module_source_embeds_planar_decode_and_reduction() {
        let source = planar_moe_module_source();
        assert!(source.contains("pbmoe_planar_linear_f32"));
        assert!(source.contains("planar_bf8_element"));
        assert!(source.contains("planar_fp4_element"));
        assert!(source.contains("block_sum"));
    }
}
