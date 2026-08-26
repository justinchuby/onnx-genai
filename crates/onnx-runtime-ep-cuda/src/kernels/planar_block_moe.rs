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
//! Mobius #602 / Deckard #593 exporter. Wiring the aux-scale node inputs is the
//! explicit next slice; immutable bank value admission already reuses the CPU
//! planar oracle and is required by this primitive's launch API. The
//! onnx-runtime-ep-cpu `planar_block_quant` oracle owns the op-level routed path
//! until then. This primitive is the runtime capability the exporter probes
//! before it may emit a planar MoE node.

#[cfg(any(test, feature = "gpu-tests"))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{DeviceBuffer, EpError, ExecutionProvider, Result};
use onnx_runtime_ep_cpu::kernels::moe::{Activation, validate_moe_activation_attributes};
use onnx_runtime_ep_cpu::kernels::planar_block_quant::{
    FP4_MICROSCALE_BLOCK as CPU_FP4_MICROSCALE_BLOCK, PlanarBankIdentity, PlanarBlockFormat,
    PlanarLayout, validate_planar_expert_bank_values,
};
use onnx_runtime_memory_governor::ProviderContextIdentity;

use crate::error::driver_err;
use crate::kernels::block_quantized_matmul::decoder_prelude;
use crate::kernels::block_quantized_moe::{
    MOE_ACTIVATE_ENTRY, MOE_COMBINE_ENTRY, MOE_MODULE, MOE_ROUTE_ENTRY, moe_module_source,
};
use crate::kernels::planar_block_decode::{
    ImmutablePlanarDeviceBuffer, PLANAR_BLOCK_DECODE_CUH, PlanarLinearDims,
};
use crate::provider::CudaExecutionProvider;
use crate::runtime::{CudaRuntime, cuptr};

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
fn planar_moe_module_source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| {
        #[cfg(any(test, feature = "gpu-tests"))]
        PLANAR_MOE_SOURCE_BUILDS.fetch_add(1, Ordering::Relaxed);
        let mut source = decoder_prelude();
        source.push_str(PLANAR_BLOCK_DECODE_CUH);
        source.push_str(PLANAR_MOE_KERNEL);
        source
    })
}

#[cfg(any(test, feature = "gpu-tests"))]
static PLANAR_MOE_SOURCE_BUILDS: AtomicUsize = AtomicUsize::new(0);

/// Test-only observability for the warm-source contract. The counter advances
/// only inside the process-level [`OnceLock`] initializer, never on the launch
/// hot path.
#[cfg(any(test, feature = "gpu-tests"))]
pub fn planar_moe_source_build_count() -> usize {
    PLANAR_MOE_SOURCE_BUILDS.load(Ordering::Relaxed)
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

    fn cpu_layout(&self) -> Result<PlanarLayout> {
        let (format, block_out, block_in) = match self.format {
            crate::kernels::planar_block_decode::PLANAR_FORMAT_BLOCK_FP8 => {
                (PlanarBlockFormat::BlockFp8, self.bs0, self.bs1)
            }
            crate::kernels::planar_block_decode::PLANAR_FORMAT_FP4_PLANAR => {
                (PlanarBlockFormat::Fp4Planar, 1, CPU_FP4_MICROSCALE_BLOCK)
            }
            other => return Err(kernel_err(format!("unknown planar format id {other}"))),
        };
        PlanarLayout::new(
            format,
            self.out_features,
            self.in_features,
            block_out,
            block_in,
        )
        .map_err(|err| kernel_err(err.to_string()))
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
    pub fn fc1_out(&self) -> Result<usize> {
        if self.swiglu_fusion != 0 {
            self.inter
                .checked_mul(2)
                .ok_or_else(|| kernel_err("fused fc1 width overflow"))
        } else {
            Ok(self.inter)
        }
    }
}

/// Element counts for every non-weight buffer used by one planar routed-MoE
/// launch. Optional buffers use `None` when their matching pointer is absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanarMoeBufferLengths {
    pub input_elems: usize,
    pub router_logits_elems: usize,
    pub router_weights_elems: Option<usize>,
    pub route_indices_elems: usize,
    pub route_weights_elems: usize,
    pub fc1_output_elems: usize,
    pub fc3_output_elems: Option<usize>,
    pub activated_elems: usize,
    pub route_output_elems: usize,
    pub output_elems: usize,
}

/// Immutable host view of one expert-major planar projection bank.
#[derive(Clone, Copy, Debug)]
pub struct PlanarMoeBank<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [u8],
    pub bias_elems: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct ValidatedPlanarMoeProjection {
    projection: PlanarMoeProjection,
    packed_expert_stride: usize,
    scale_expert_stride: usize,
    bias_elems: Option<usize>,
    identity: PlanarBankIdentity,
}

/// Cached admission proof for an immutable planar routed-MoE bank set.
#[derive(Clone, Copy, Debug)]
struct ValidatedPlanarMoe {
    dims: PlanarMoeDims,
    buffers: PlanarMoeBufferLengths,
    routes: usize,
    fc1_out: usize,
    fc1: ValidatedPlanarMoeProjection,
    fc2: ValidatedPlanarMoeProjection,
    fc3: Option<ValidatedPlanarMoeProjection>,
}

impl PlanarMoeBufferLengths {
    /// Compute the exact buffer extents for `dims`, rejecting any multiplication
    /// overflow before allocation or launch.
    pub fn for_dims(dims: &PlanarMoeDims, has_router_weights: bool) -> Result<Self> {
        let fc1_out = dims.fc1_out()?;
        Self::for_dims_with_fc1_out(dims, has_router_weights, fc1_out)
    }

    fn for_dims_with_fc1_out(
        dims: &PlanarMoeDims,
        has_router_weights: bool,
        fc1_out: usize,
    ) -> Result<Self> {
        let routes = dims
            .rows
            .checked_mul(dims.top_k)
            .ok_or_else(|| kernel_err("route count overflow"))?;
        let input_elems = dims
            .rows
            .checked_mul(dims.hidden)
            .ok_or_else(|| kernel_err("input element count overflow"))?;
        let router_elems = dims
            .rows
            .checked_mul(dims.experts)
            .ok_or_else(|| kernel_err("router element count overflow"))?;
        let fc1_output_elems = routes
            .checked_mul(fc1_out)
            .ok_or_else(|| kernel_err("fc1 output element count overflow"))?;
        let inter_elems = routes
            .checked_mul(dims.inter)
            .ok_or_else(|| kernel_err("activated element count overflow"))?;
        let route_output_elems = routes
            .checked_mul(dims.hidden)
            .ok_or_else(|| kernel_err("route output element count overflow"))?;
        Ok(Self {
            input_elems,
            router_logits_elems: router_elems,
            router_weights_elems: has_router_weights.then_some(router_elems),
            route_indices_elems: routes,
            route_weights_elems: routes,
            fc1_output_elems,
            fc3_output_elems: dims.fc3.is_some().then_some(inter_elems),
            activated_elems: inter_elems,
            route_output_elems,
            output_elems: input_elems,
        })
    }
}

/// Validate a planar routed MoE's geometry, immutable host-bank values, and
/// every supplied bank/workspace extent. This is the host-side reserved-code,
/// decoded-overflow, and aux/OOB guard that must pass before upload or launch.
/// Any mismatch is a typed rejection — never a dense-expert fallback.
#[allow(clippy::too_many_arguments)]
fn validate_planar_moe_host(
    dims: &PlanarMoeDims,
    fc1_bank: PlanarMoeBank<'_>,
    fc2_bank: PlanarMoeBank<'_>,
    fc3_bank: Option<PlanarMoeBank<'_>>,
    buffers: &PlanarMoeBufferLengths,
) -> Result<ValidatedPlanarMoe> {
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
    let fc1_out = dims.fc1_out()?;
    let activation = Activation::from_kernel_id(dims.activation)
        .ok_or_else(|| kernel_err(format!("unknown activation id {}", dims.activation)))?;
    validate_moe_activation_attributes(
        activation.name(),
        i64::from(dims.swiglu_fusion),
        dims.activation_alpha,
        dims.activation_beta,
        dims.swiglu_limit,
    )
    .map_err(kernel_err)?;
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
    if dims.fc1.out_features != fc1_out {
        return Err(kernel_err(format!(
            "fc1 out={} must equal fc1_out={} (inter={}, swiglu_fusion={})",
            dims.fc1.out_features, fc1_out, dims.inter, dims.swiglu_fusion
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

    let fc1 = validate_projection_bank(dims, &dims.fc1, "fc1", fc1_bank)?;
    let fc2 = validate_projection_bank(dims, &dims.fc2, "fc2", fc2_bank)?;

    let fc3 = match (dims.fc3.as_ref(), fc3_bank) {
        (Some(fc3), Some(bank)) => {
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
            Some(validate_projection_bank(dims, fc3, "fc3", bank)?)
        }
        (None, None) => None,
        (Some(_), None) => {
            return Err(kernel_err("fc3 projection present but fc3 banks missing"));
        }
        (None, Some(_)) => {
            return Err(kernel_err("fc3 banks present but fc3 projection missing"));
        }
    };

    let expected = PlanarMoeBufferLengths::for_dims_with_fc1_out(
        dims,
        buffers.router_weights_elems.is_some(),
        fc1_out,
    )?;
    for (label, supplied, required) in [
        ("input", buffers.input_elems, expected.input_elems),
        (
            "router_logits",
            buffers.router_logits_elems,
            expected.router_logits_elems,
        ),
        (
            "route_indices",
            buffers.route_indices_elems,
            expected.route_indices_elems,
        ),
        (
            "route_weights",
            buffers.route_weights_elems,
            expected.route_weights_elems,
        ),
        (
            "fc1_output",
            buffers.fc1_output_elems,
            expected.fc1_output_elems,
        ),
        (
            "activated",
            buffers.activated_elems,
            expected.activated_elems,
        ),
        (
            "route_output",
            buffers.route_output_elems,
            expected.route_output_elems,
        ),
        ("output", buffers.output_elems, expected.output_elems),
    ] {
        if supplied != required {
            return Err(kernel_err(format!(
                "{label} has {supplied} elements, expected {required}"
            )));
        }
    }
    for (label, supplied, required) in [
        (
            "router_weights",
            buffers.router_weights_elems,
            expected.router_weights_elems,
        ),
        (
            "fc3_output",
            buffers.fc3_output_elems,
            expected.fc3_output_elems,
        ),
    ] {
        if supplied != required {
            return Err(kernel_err(format!(
                "{label} has {supplied:?} elements, expected {required:?}"
            )));
        }
    }
    Ok(ValidatedPlanarMoe {
        dims: *dims,
        buffers: *buffers,
        routes: expected.route_indices_elems,
        fc1_out,
        fc1,
        fc2,
        fc3,
    })
}

fn validate_projection_bank(
    dims: &PlanarMoeDims,
    projection: &PlanarMoeProjection,
    label: &str,
    bank: PlanarMoeBank<'_>,
) -> Result<ValidatedPlanarMoeProjection> {
    let (per_packed, per_scale) = projection.per_expert_bytes()?;
    let expected_packed = per_packed
        .checked_mul(dims.experts)
        .ok_or_else(|| kernel_err(format!("{label} packed bank byte count overflow")))?;
    let expected_scale = per_scale
        .checked_mul(dims.experts)
        .ok_or_else(|| kernel_err(format!("{label} scale bank byte count overflow")))?;
    if bank.packed.len() != expected_packed {
        return Err(kernel_err(format!(
            "{label} packed bank has {} bytes, expected experts*{per_packed} = {expected_packed}",
            bank.packed.len()
        )));
    }
    if bank.scale.len() != expected_scale {
        return Err(kernel_err(format!(
            "{label} scale bank has {} bytes, expected experts*{per_scale} = {expected_scale}",
            bank.scale.len()
        )));
    }
    if let Some(bias_elems) = bank.bias_elems {
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
    let identity = validate_planar_expert_bank_values(
        &projection.cpu_layout()?,
        dims.experts,
        bank.packed,
        bank.scale,
    )
    .map_err(|err| kernel_err(format!("{label} value admission failed: {err}")))?;
    Ok(ValidatedPlanarMoeProjection {
        projection: *projection,
        packed_expert_stride: per_packed,
        scale_expert_stride: per_scale,
        bias_elems: bank.bias_elems,
        identity,
    })
}

struct AdmittedPlanarMoeProjection {
    validation: ValidatedPlanarMoeProjection,
    packed: ImmutablePlanarDeviceBuffer,
    scale: ImmutablePlanarDeviceBuffer,
}

impl AdmittedPlanarMoeProjection {
    fn upload(
        provider: &Arc<CudaExecutionProvider>,
        validation: ValidatedPlanarMoeProjection,
        bank: PlanarMoeBank<'_>,
        label: &str,
    ) -> Result<Self> {
        Ok(Self {
            validation,
            packed: ImmutablePlanarDeviceBuffer::upload(
                provider,
                bank.packed,
                &format!("{label} packed weights"),
            )?,
            scale: ImmutablePlanarDeviceBuffer::upload(
                provider,
                bank.scale,
                &format!("{label} aux scales"),
            )?,
        })
    }
}

struct PlanarMoeBanks {
    fc1: AdmittedPlanarMoeProjection,
    fc2: AdmittedPlanarMoeProjection,
    fc3: Option<AdmittedPlanarMoeProjection>,
}

/// Sealed admission for all planar routed-MoE projection banks.
///
/// FC1, FC2, and optional FC3 each own their exact immutable packed-weight and
/// aux-scale device allocations together with their independent
/// format/geometry/stride validation. The type is not constructible, cloneable,
/// or mutable through safe external APIs. Diagnostic hashes never authorize a
/// launch; ownership of these allocations does.
///
/// No content-addressed admission cache participates in correctness. A
/// provider-internal stable-VA remap may preserve this handle only while the
/// same immutable content and allocation ownership remain bound to its runtime
/// and device. Destructive movement or replacement must invalidate/drop the
/// handle and atomically re-admit the new bytes. During CUDA graph capture the
/// graph registry strongly owns every projection bank, so release, remap, and
/// content replacement stay unreachable until every graph pin is reset or
/// destroyed.
///
/// ```
/// fn accepts(_: &onnx_runtime_ep_cuda::AdmittedPlanarMoe) {}
/// ```
///
/// ```compile_fail,E0451
/// # use onnx_runtime_ep_cuda::AdmittedPlanarMoe;
/// let forged = AdmittedPlanarMoe { runtime: todo!() };
/// ```
///
/// ```compile_fail
/// # use onnx_runtime_ep_cuda::AdmittedPlanarMoe;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<AdmittedPlanarMoe>();
/// ```
///
/// Safe upload and VMM APIs require a `DeviceBuffer`; an admission cannot be
/// converted to one, so admitted bytes cannot be overwritten or remapped.
///
/// ```compile_fail
/// # use onnx_runtime_ep_api::ExecutionProvider;
/// # use onnx_runtime_ep_cuda::{AdmittedPlanarMoe, CudaExecutionProvider};
/// # fn overwrite(ep: &CudaExecutionProvider, bank: &mut AdmittedPlanarMoe) {
/// ep.copy_from_host(&[0xff], bank);
/// # }
/// ```
///
/// ```compile_fail
/// # use onnx_runtime_ep_api::ExecutionProvider;
/// # use onnx_runtime_ep_cuda::{AdmittedPlanarMoe, CudaExecutionProvider};
/// # fn remap(ep: &CudaExecutionProvider, bank: &AdmittedPlanarMoe) {
/// ep.decommit_allocation_range(bank, 0, 1);
/// # }
/// ```
///
/// ```compile_fail
/// # use onnx_runtime_ep_cuda::AdmittedPlanarMoe;
/// # fn escape(bank: &AdmittedPlanarMoe) {
/// let _ = bank.as_ptr();
/// # }
/// ```
pub struct AdmittedPlanarMoe {
    // See `AdmittedPlanarLinear`: graphs retain only these sealed allocations,
    // never the provider/runtime that owns the graph registry.
    banks: Arc<PlanarMoeBanks>,
    provider: Arc<CudaExecutionProvider>,
    device: onnx_runtime_ir::DeviceId,
    provider_context: ProviderContextIdentity,
    dims: PlanarMoeDims,
    buffers: PlanarMoeBufferLengths,
    routes: usize,
    fc1_out: usize,
}

impl AdmittedPlanarMoe {
    pub fn dims(&self) -> &PlanarMoeDims {
        &self.dims
    }

    pub fn buffers(&self) -> &PlanarMoeBufferLengths {
        &self.buffers
    }

    pub fn diagnostic_bank_identities(&self) -> [Option<PlanarBankIdentity>; 3] {
        [
            Some(self.banks.fc1.validation.identity),
            Some(self.banks.fc2.validation.identity),
            self.banks
                .fc3
                .as_ref()
                .map(|projection| projection.validation.identity),
        ]
    }
}

/// Validate every projection and workspace extent before uploading anything,
/// then mint one sealed owner for the exact immutable device banks.
pub fn admit_planar_moe(
    provider: &Arc<CudaExecutionProvider>,
    dims: &PlanarMoeDims,
    fc1_bank: PlanarMoeBank<'_>,
    fc2_bank: PlanarMoeBank<'_>,
    fc3_bank: Option<PlanarMoeBank<'_>>,
    buffers: &PlanarMoeBufferLengths,
) -> Result<AdmittedPlanarMoe> {
    if provider.runtime().is_capturing()? {
        return Err(kernel_err(
            "cannot admit planar MoE banks during CUDA graph capture",
        ));
    }
    let validation = validate_planar_moe_host(dims, fc1_bank, fc2_bank, fc3_bank, buffers)?;
    let fc1 = AdmittedPlanarMoeProjection::upload(provider, validation.fc1, fc1_bank, "fc1")?;
    let fc2 = AdmittedPlanarMoeProjection::upload(provider, validation.fc2, fc2_bank, "fc2")?;
    let fc3 = match (validation.fc3, fc3_bank) {
        (Some(validation), Some(bank)) => Some(AdmittedPlanarMoeProjection::upload(
            provider, validation, bank, "fc3",
        )?),
        (None, None) => None,
        _ => {
            return Err(kernel_err(
                "internal fc3 admission mismatch after host validation",
            ));
        }
    };
    Ok(AdmittedPlanarMoe {
        banks: Arc::new(PlanarMoeBanks { fc1, fc2, fc3 }),
        provider: Arc::clone(provider),
        device: provider.device_id(),
        provider_context: provider.provider_context_identity(),
        dims: validation.dims,
        buffers: validation.buffers,
        routes: validation.routes,
        fc1_out: validation.fc1_out,
    })
}

/// Borrowed non-weight inputs, optional biases, workspaces, and output for one
/// launch. Packed weights and aux scales are deliberately absent: the sealed
/// [`AdmittedPlanarMoe`] is their only safe owner/source.
pub struct PlanarMoeBuffers<'a> {
    /// `f32 [rows, hidden]` token activations.
    pub input: &'a DeviceBuffer,
    /// `f32 [rows, experts]` router logits.
    pub router_logits: &'a DeviceBuffer,
    /// `f32 [rows, experts]` pre-aggregated router weights, or `None` for softmax.
    pub router_weights: Option<&'a DeviceBuffer>,
    /// Optional immutable projection biases.
    pub fc1_bias: Option<&'a DeviceBuffer>,
    pub fc2_bias: Option<&'a DeviceBuffer>,
    pub fc3_bias: Option<&'a DeviceBuffer>,

    /// `i32 [routes]` selected expert scratch.
    pub route_indices: &'a mut DeviceBuffer,
    /// `f32 [routes]` selected weight scratch.
    pub route_weights: &'a mut DeviceBuffer,
    /// `f32 [routes, fc1_out]` fc1 output scratch.
    pub fc1_output: &'a mut DeviceBuffer,
    /// `f32 [routes, inter]` fc3 output scratch, or `None`.
    pub fc3_output: Option<&'a mut DeviceBuffer>,
    /// `f32 [routes, inter]` activated scratch.
    pub activated: &'a mut DeviceBuffer,
    /// `f32 [routes, hidden]` per-route output scratch.
    pub route_output: &'a mut DeviceBuffer,

    /// `f32 [rows, hidden]` final combined output.
    pub output: &'a mut DeviceBuffer,
}

#[derive(Clone, Copy)]
struct PlanarMoeRawPtrs {
    input: CUdeviceptr,
    router_logits: CUdeviceptr,
    router_weights: CUdeviceptr,
    fc1_bias: CUdeviceptr,
    fc2_bias: CUdeviceptr,
    fc3_bias: CUdeviceptr,
    route_indices: CUdeviceptr,
    route_weights: CUdeviceptr,
    fc1_output: CUdeviceptr,
    fc3_output: CUdeviceptr,
    activated: CUdeviceptr,
    route_output: CUdeviceptr,
    output: CUdeviceptr,
}

fn exact_f32_bytes(elements: usize, label: &str) -> Result<usize> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| kernel_err(format!("{label} byte count overflow")))
}

fn require_buffer(
    device: onnx_runtime_ir::DeviceId,
    provider_context: ProviderContextIdentity,
    label: &str,
    buffer: &DeviceBuffer,
    bytes: usize,
) -> Result<()> {
    if buffer.device() != device {
        return Err(kernel_err(format!(
            "{label} device {:?} does not match admitted bank device {device:?}",
            buffer.device()
        )));
    }
    if buffer.len() != bytes {
        return Err(kernel_err(format!(
            "{label} has {} bytes, expected {bytes}",
            buffer.len()
        )));
    }
    let context = buffer
        .bound_owner()
        .ok_or_else(|| {
            kernel_err(format!(
                "{label} has no binding-issued provider-context identity"
            ))
        })?
        .identity()
        .binding()
        .provider_context();
    if context != provider_context {
        return Err(kernel_err(format!(
            "{label} provider context {context:?} does not match admitted bank context \
             {provider_context:?}"
        )));
    }
    Ok(())
}

fn require_optional_buffer(
    device: onnx_runtime_ir::DeviceId,
    provider_context: ProviderContextIdentity,
    label: &str,
    buffer: Option<&DeviceBuffer>,
    elements: Option<usize>,
) -> Result<CUdeviceptr> {
    match (buffer, elements) {
        (Some(buffer), Some(elements)) => {
            require_buffer(
                device,
                provider_context,
                label,
                buffer,
                exact_f32_bytes(elements, label)?,
            )?;
            Ok(cuptr(buffer.as_ptr()))
        }
        (None, None) => Ok(0),
        (Some(_), None) => Err(kernel_err(format!(
            "{label} was supplied but the admitted projection has no bias/output"
        ))),
        (None, Some(_)) => Err(kernel_err(format!(
            "{label} is required by the admitted projection"
        ))),
    }
}

fn validate_planar_moe_buffers(
    admission: &AdmittedPlanarMoe,
    buffers: &mut PlanarMoeBuffers<'_>,
) -> Result<PlanarMoeRawPtrs> {
    let device = admission.device;
    let provider_context = admission.provider_context;
    let lengths = admission.buffers();
    require_buffer(
        device,
        provider_context,
        "input",
        buffers.input,
        exact_f32_bytes(lengths.input_elems, "input")?,
    )?;
    require_buffer(
        device,
        provider_context,
        "router_logits",
        buffers.router_logits,
        exact_f32_bytes(lengths.router_logits_elems, "router_logits")?,
    )?;
    let router_weights = require_optional_buffer(
        device,
        provider_context,
        "router_weights",
        buffers.router_weights,
        lengths.router_weights_elems,
    )?;
    let fc1_bias = require_optional_buffer(
        device,
        provider_context,
        "fc1_bias",
        buffers.fc1_bias,
        admission.banks.fc1.validation.bias_elems,
    )?;
    let fc2_bias = require_optional_buffer(
        device,
        provider_context,
        "fc2_bias",
        buffers.fc2_bias,
        admission.banks.fc2.validation.bias_elems,
    )?;
    let fc3_bias = require_optional_buffer(
        device,
        provider_context,
        "fc3_bias",
        buffers.fc3_bias,
        admission
            .banks
            .fc3
            .as_ref()
            .and_then(|projection| projection.validation.bias_elems),
    )?;
    require_buffer(
        device,
        provider_context,
        "route_indices",
        buffers.route_indices,
        lengths
            .route_indices_elems
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| kernel_err("route_indices byte count overflow"))?,
    )?;
    require_buffer(
        device,
        provider_context,
        "route_weights",
        buffers.route_weights,
        exact_f32_bytes(lengths.route_weights_elems, "route_weights")?,
    )?;
    require_buffer(
        device,
        provider_context,
        "fc1_output",
        buffers.fc1_output,
        exact_f32_bytes(lengths.fc1_output_elems, "fc1_output")?,
    )?;
    let fc3_output = require_optional_buffer(
        device,
        provider_context,
        "fc3_output",
        buffers.fc3_output.as_deref(),
        lengths.fc3_output_elems,
    )?;
    require_buffer(
        device,
        provider_context,
        "activated",
        buffers.activated,
        exact_f32_bytes(lengths.activated_elems, "activated")?,
    )?;
    require_buffer(
        device,
        provider_context,
        "route_output",
        buffers.route_output,
        exact_f32_bytes(lengths.route_output_elems, "route_output")?,
    )?;
    require_buffer(
        device,
        provider_context,
        "output",
        buffers.output,
        exact_f32_bytes(lengths.output_elems, "output")?,
    )?;

    Ok(PlanarMoeRawPtrs {
        input: cuptr(buffers.input.as_ptr()),
        router_logits: cuptr(buffers.router_logits.as_ptr()),
        router_weights,
        fc1_bias,
        fc2_bias,
        fc3_bias,
        route_indices: cuptr(buffers.route_indices.as_mut_ptr()),
        route_weights: cuptr(buffers.route_weights.as_mut_ptr()),
        fc1_output: cuptr(buffers.fc1_output.as_mut_ptr()),
        fc3_output,
        activated: cuptr(buffers.activated.as_mut_ptr()),
        route_output: cuptr(buffers.route_output.as_mut_ptr()),
        output: cuptr(buffers.output.as_mut_ptr()),
    })
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
        planar_moe_module_source(),
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
    admission: &AdmittedPlanarMoeProjection,
    input: CUdeviceptr,
    route_indices: CUdeviceptr,
    bias: CUdeviceptr,
    output: CUdeviceptr,
    routes: usize,
    top_k: usize,
    input_rows_are_routes: bool,
    admitted_out_features: usize,
) -> Result<()> {
    let projection = &admission.validation.projection;
    if projection.out_features != admitted_out_features {
        return Err(kernel_err(format!(
            "sealed projection out={} does not match admitted launch width {admitted_out_features}",
            projection.out_features
        )));
    }
    let function = runtime.nvrtc_function(
        PLANAR_MOE_MODULE,
        planar_moe_module_source(),
        PLANAR_MOE_LINEAR_ENTRY,
    )?;
    let tasks = (routes as u64)
        .checked_mul(admitted_out_features as u64)
        .ok_or_else(|| kernel_err("linear task count overflow"))?;
    let grid_x = saturating_grid(runtime, tasks);
    let config =
        runtime.reduction_launch_config(&function, grid_x, preferred_threads(runtime), 4)?;

    let routes_u64 = routes as u64;
    let input_rows_are_routes = i32::from(input_rows_are_routes);
    let top_k = as_i32("top_k", top_k)?;
    let out_features = as_i32("out_features", admitted_out_features)?;
    let in_features = as_i32("in_features", projection.in_features)?;
    let format = projection.format;
    let bs0 = as_i32("bs0", projection.bs0)?;
    let bs1 = as_i32("bs1", projection.bs1)?;
    let packed_stride = admission.validation.packed_expert_stride as u64;
    let scale_stride = admission.validation.scale_expert_stride as u64;
    let access = super::SealedLaunchAccess::new();
    let packed = admission.packed.ptr(&access);
    let scale = admission.scale.ptr(&access);

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
    // routes*out_features, all validated by `admit_planar_moe`.
    unsafe { builder.launch(config) }
        .map(|_| ())
        .map_err(|err| driver_err("launch planar MoE expert GEMV", err))
}

/// Launch the full planar routed top-k MoE pipeline on `runtime`'s EP stream:
/// `route → fc1 (+ fc3) → activate → fc2 → combine`.
///
/// The admission owns the exact FC1/FC2/FC3 bank allocations populated by
/// [`admit_planar_moe`]. The launch issues **no** allocation, host→device copy,
/// compile, or host synchronization, so a warmed signature (see
/// [`warm_planar_moe`]) records cleanly into a CUDA-graph capture.
pub fn launch_planar_moe(
    admission: &AdmittedPlanarMoe,
    buffers: &mut PlanarMoeBuffers<'_>,
) -> Result<()> {
    let ptrs = validate_planar_moe_buffers(admission, buffers)?;
    let runtime = admission.provider.runtime();
    runtime.retain_active_graph_resource(
        Arc::as_ptr(&admission.banks) as usize,
        &admission.banks,
        "planar MoE projection banks",
    )?;
    let dims = admission.dims();
    let routes = admission.routes;

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
        &admission.banks.fc1,
        ptrs.input,
        ptrs.route_indices,
        ptrs.fc1_bias,
        ptrs.fc1_output,
        routes,
        dims.top_k,
        false,
        admission.fc1_out,
    )?;
    if let Some(fc3) = admission.banks.fc3.as_ref() {
        launch_planar_linear(
            runtime,
            fc3,
            ptrs.input,
            ptrs.route_indices,
            ptrs.fc3_bias,
            ptrs.fc3_output,
            routes,
            dims.top_k,
            false,
            dims.inter,
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
        &admission.banks.fc2,
        ptrs.activated,
        ptrs.route_indices,
        ptrs.fc2_bias,
        ptrs.route_output,
        routes,
        dims.top_k,
        true,
        dims.hidden,
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
            swiglu_limit: f32::MAX,
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

    #[allow(clippy::too_many_arguments)]
    fn validate_planar_moe(
        dims: &PlanarMoeDims,
        fc1_packed_bytes: usize,
        fc1_scale_bytes: usize,
        fc1_bias_elems: Option<usize>,
        fc2_packed_bytes: usize,
        fc2_scale_bytes: usize,
        fc2_bias_elems: Option<usize>,
        fc3_banks: Option<(usize, usize, Option<usize>)>,
    ) -> Result<()> {
        let buffers = PlanarMoeBufferLengths::for_dims(dims, false)?;
        let fc1_packed = vec![0u8; fc1_packed_bytes];
        let fc1_scale = vec![127u8; fc1_scale_bytes];
        let fc2_packed = vec![0u8; fc2_packed_bytes];
        let fc2_scale = vec![127u8; fc2_scale_bytes];
        let fc3_storage = fc3_banks
            .map(|(packed, scale, bias_elems)| (vec![0u8; packed], vec![127u8; scale], bias_elems));
        let fc3_bank = fc3_storage
            .as_ref()
            .map(|(packed, scale, bias_elems)| PlanarMoeBank {
                packed,
                scale,
                bias_elems: *bias_elems,
            });
        super::validate_planar_moe_host(
            dims,
            PlanarMoeBank {
                packed: &fc1_packed,
                scale: &fc1_scale,
                bias_elems: fc1_bias_elems,
            },
            PlanarMoeBank {
                packed: &fc2_packed,
                scale: &fc2_scale,
                bias_elems: fc2_bias_elems,
            },
            fc3_bank,
            &buffers,
        )
        .map(|_| ())
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
    fn undersized_workspace_is_typed_rejected() {
        let dims = base_dims();
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let mut buffers = PlanarMoeBufferLengths::for_dims(&dims, false).unwrap();
        buffers.fc1_output_elems -= 1;
        let fc1_packed = vec![0u8; fc1p];
        let fc1_scale = vec![127u8; fc1s];
        let fc2_packed = vec![0u8; fc2p];
        let fc2_scale = vec![127u8; fc2s];
        let err = super::validate_planar_moe_host(
            &dims,
            PlanarMoeBank {
                packed: &fc1_packed,
                scale: &fc1_scale,
                bias_elems: None,
            },
            PlanarMoeBank {
                packed: &fc2_packed,
                scale: &fc2_scale,
                bias_elems: None,
            },
            None,
            &buffers,
        )
        .expect_err("undersized fc1 workspace must be rejected");
        assert!(format!("{err:?}").contains("fc1_output"));
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
        assert_eq!(dims.fc1_out().unwrap(), dims.inter * 2);
        dims.fc1 = fp8(256, 128); // wrong: only inter wide
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let err = validate_planar_moe(&dims, fc1p, fc1s, None, fc2p, fc2s, None, None)
            .expect_err("fused SwiGLU with inter-wide fc1 must be rejected");
        assert!(format!("{err:?}").contains("fc1 out"));
    }

    #[test]
    fn fused_width_overflow_is_typed_rejected_before_bank_validation() {
        let mut dims = base_dims();
        dims.activation = 3;
        dims.swiglu_fusion = 1;
        dims.inter = usize::MAX / 2 + 1;
        dims.fc1.out_features = 0;
        let err = dims
            .fc1_out()
            .expect_err("an unrepresentable fused width must be rejected");
        assert!(format!("{err:?}").contains("fused fc1 width overflow"));

        let err = PlanarMoeBufferLengths::for_dims(&dims, false)
            .expect_err("buffer sizing must reuse the checked fused width");
        assert!(format!("{err:?}").contains("fused fc1 width overflow"));

        let empty = PlanarMoeBank {
            packed: &[],
            scale: &[],
            bias_elems: None,
        };
        let zero_buffers = PlanarMoeBufferLengths {
            input_elems: 0,
            router_logits_elems: 0,
            router_weights_elems: None,
            route_indices_elems: 0,
            route_weights_elems: 0,
            fc1_output_elems: 0,
            fc3_output_elems: None,
            activated_elems: 0,
            route_output_elems: 0,
            output_elems: 0,
        };
        let err = super::validate_planar_moe_host(&dims, empty, empty, None, &zero_buffers)
            .expect_err("admission validation must reject overflow without panicking");
        assert!(format!("{err:?}").contains("fused fc1 width overflow"));
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
        assert!(format!("{err:?}").contains("only valid when activation_type='swiglu'"));
    }

    #[test]
    fn negative_swiglu_fusion_is_typed_rejected() {
        let mut dims = base_dims();
        dims.activation = 3;
        dims.swiglu_fusion = -1;
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let err = validate_planar_moe(&dims, fc1p, fc1s, None, fc2p, fc2s, None, None)
            .expect_err("negative swiglu_fusion must be rejected");
        assert!(format!("{err:?}").contains("must be 0, 1, or 2"));
    }

    #[test]
    fn invalid_activation_parameters_are_typed_rejected() {
        for (name, alpha, beta, limit) in [
            ("activation_alpha NaN", f32::NAN, 0.0, 1.0),
            ("activation_alpha +Inf", f32::INFINITY, 0.0, 1.0),
            ("activation_alpha -Inf", f32::NEG_INFINITY, 0.0, 1.0),
            ("activation_beta NaN", 1.0, f32::NAN, 1.0),
            ("activation_beta +Inf", 1.0, f32::INFINITY, 1.0),
            ("activation_beta -Inf", 1.0, f32::NEG_INFINITY, 1.0),
            ("swiglu_limit NaN", 1.0, 0.0, f32::NAN),
            ("swiglu_limit +Inf", 1.0, 0.0, f32::INFINITY),
            ("swiglu_limit -Inf", 1.0, 0.0, f32::NEG_INFINITY),
            ("swiglu_limit zero", 1.0, 0.0, 0.0),
            ("swiglu_limit negative", 1.0, 0.0, -1.0),
        ] {
            let mut dims = base_dims();
            dims.activation = 3;
            dims.swiglu_fusion = 1;
            dims.fc1 = fp8(dims.hidden, dims.inter * 2);
            dims.activation_alpha = alpha;
            dims.activation_beta = beta;
            dims.swiglu_limit = limit;
            let (fc1p, fc1s) = banks(&dims, &dims.fc1);
            let (fc2p, fc2s) = banks(&dims, &dims.fc2);
            assert!(
                validate_planar_moe(&dims, fc1p, fc1s, None, fc2p, fc2s, None, None).is_err(),
                "{name} must fail before a launch token exists"
            );
        }
    }

    #[test]
    fn malformed_moe_banks_reject_reserved_and_overflowing_values() {
        let mut dims = base_dims();
        dims.rows = 1;
        dims.hidden = 32;
        dims.inter = 32;
        dims.experts = 2;
        dims.top_k = 1;
        dims.fc1 = fp8(32, 32);
        dims.fc2 = fp8(32, 32);
        let buffers = PlanarMoeBufferLengths::for_dims(&dims, false).unwrap();
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let mut fc1_packed = vec![0u8; fc1p];
        let mut fc1_scale = vec![127u8; fc1s];
        let fc2_packed = vec![0u8; fc2p];
        let mut fc2_scale = vec![127u8; fc2s];

        fc1_packed[0] = 0x7f;
        assert!(
            super::validate_planar_moe_host(
                &dims,
                PlanarMoeBank {
                    packed: &fc1_packed,
                    scale: &fc1_scale,
                    bias_elems: None,
                },
                PlanarMoeBank {
                    packed: &fc2_packed,
                    scale: &fc2_scale,
                    bias_elems: None,
                },
                None,
                &buffers,
            )
            .is_err()
        );

        fc1_packed[0] = 0x7e;
        fc1_scale[0] = 247;
        assert!(
            super::validate_planar_moe_host(
                &dims,
                PlanarMoeBank {
                    packed: &fc1_packed,
                    scale: &fc1_scale,
                    bias_elems: None,
                },
                PlanarMoeBank {
                    packed: &fc2_packed,
                    scale: &fc2_scale,
                    bias_elems: None,
                },
                None,
                &buffers,
            )
            .is_err()
        );

        fc1_packed[0] = 0;
        fc1_scale[0] = 127;
        fc2_scale[0] = 0xff;
        assert!(
            super::validate_planar_moe_host(
                &dims,
                PlanarMoeBank {
                    packed: &fc1_packed,
                    scale: &fc1_scale,
                    bias_elems: None,
                },
                PlanarMoeBank {
                    packed: &fc2_packed,
                    scale: &fc2_scale,
                    bias_elems: None,
                },
                None,
                &buffers,
            )
            .is_err()
        );

        dims.fc1 = fp4(32, 32);
        dims.fc2 = fp4(32, 32);
        let (fc1p, fc1s) = banks(&dims, &dims.fc1);
        let (fc2p, fc2s) = banks(&dims, &dims.fc2);
        let fc1_packed = vec![0x77u8; fc1p];
        let fc1_scale = vec![253u8; fc1s];
        let fc2_packed = vec![0u8; fc2p];
        let fc2_scale = vec![127u8; fc2s];
        assert!(
            super::validate_planar_moe_host(
                &dims,
                PlanarMoeBank {
                    packed: &fc1_packed,
                    scale: &fc1_scale,
                    bias_elems: None,
                },
                PlanarMoeBank {
                    packed: &fc2_packed,
                    scale: &fc2_scale,
                    bias_elems: None,
                },
                None,
                &buffers,
            )
            .is_err()
        );
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
        let same_source = planar_moe_module_source();
        assert!(std::ptr::eq(source, same_source));
        assert_eq!(planar_moe_source_build_count(), 1);
        assert!(source.contains("pbmoe_planar_linear_f32"));
        assert!(source.contains("planar_bf8_element"));
        assert!(source.contains("planar_fp4_element"));
        assert!(source.contains("block_sum"));
    }
}
