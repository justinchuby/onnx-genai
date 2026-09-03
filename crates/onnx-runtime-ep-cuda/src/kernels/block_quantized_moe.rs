//! CUDA implementation of the `pkg.nxrt::BlockQuantizedMoE` operator
//! (mixed-projection ABI).
//!
//! This is the CUDA counterpart to the CPU parity oracle
//! ([`onnx_runtime_ep_cpu::kernels::block_quantized_moe`]). Expert weights stay
//! resident on one GPU packed in the native GGUF block formats. Per-weight
//! dequantization reuses the exact `decode_weight` device routine that backs
//! [`super::block_quantized_matmul`], so the numeric semantics match the oracle
//! block-for-block; only the reduction/accumulation order differs (both
//! accumulate in f32).
//!
//! The mixed-projection ABI lets `fc1_format`, `fc2_format` and `fc3_format`
//! differ per projection. Each launch derives its decoder geometry from the
//! selected projection; no global qtype is assumed.
//!
//! The pipeline mirrors the CPU reference: host-free top-k routing, per-route
//! expert GEMV for FC1 (and the optional FC3 gate), a fused
//! activation/SwiGLU pass, the FC2 down-projection, and a weighted combine of
//! each token's selected experts. All heavy work runs asynchronously on the
//! EP's selected-device stream and is safe to record after warm-up.
//!
//! The concrete kernel/factory and raw telemetry record are intentionally not
//! part of the ordinary production API; callers observe traffic through
//! `onnx_runtime_session::BlockQuantizedMoeTrafficObserver`.
//!
//! ```compile_fail
//! use onnx_runtime_ep_cuda::kernels::block_quantized_moe::BlockQuantizedMoEFactory;
//! ```

use std::borrow::Cow;
use std::cell::Cell;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use arc_swap::ArcSwapOption;
use cudarc::driver::LaunchConfig;
use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    DeviceGraphResource, EpError, ExecutionProvider, Kernel, KernelConstantInput, KernelFactory,
    Result, SealedDeviceAllocation, TensorBacking, TensorMetadata, TensorMut, TensorView,
    WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ep_cpu::kernels::moe::{
    Activation, DEFAULT_SWIGLU_LIMIT, validate_moe_activation_attributes,
};
use onnx_runtime_ir::block_quant_schema::{
    BLOCK_QUANTIZED_MOE_INPUT_COUNT as INPUT_COUNT, BLOCK_QUANTIZED_MOE_INPUT_NAMES as INPUT_NAMES,
    BQMOE_FC1_SCALE, BQMOE_FC1_WEIGHT, BQMOE_FC2_SCALE, BQMOE_FC2_WEIGHT, BQMOE_FC3_SCALE,
    BQMOE_FC3_WEIGHT, PlanarBlockGeometry, planar_geometry_from_node, require_layout_v1,
};
use onnx_runtime_ir::{DataType, Node, Shape};
use onnx_runtime_memory_governor::{MemoryRole, ProviderContextIdentity};

use super::planar_block_decode::{PlanarLinearDims, validate_planar_bank_device};
use super::planar_block_moe::{
    BorrowedPlanarMoeProjection, BorrowedPlanarMoePtrs, PlanarMoeDims, PlanarMoeProjection,
    launch_planar_moe_borrowed, warm_planar_moe,
};
use crate::error::driver_err;
use crate::kernels::block_quantized_matmul::{BlockFormat, decoder_prelude};
use crate::kernels::expert_route_telemetry::{
    ArmedTelemetry, MARK_DEVICE_SRC, RouteTelemetryConfig, TelemetryUnsupported,
};
#[cfg(feature = "gpu-tests")]
use crate::kernels::expert_route_telemetry::{
    H_COUNT, H_DEVICE, H_EPOCH, H_OVERFLOW, H_POISON, H_REQUEST, TelemetrySnapshot,
};
use crate::provider::CudaExecutionProvider;
use crate::runtime::{CudaRuntime, RawCudaFunction, cuptr};

pub use onnx_runtime_ep_api::BlockQuantizedMoeTraffic;

const OP: &str = "BlockQuantizedMoE";
const DOMAIN: &str = onnx_runtime_ir::RUNTIME_DOMAIN;

const MODULE: &str = "block_quantized_moe_v2";
const ROUTE_ENTRY: &str = "bqmoe_route";
const LINEAR_ENTRY: &str = "bqmoe_linear_f32";
const ACTIVATE_ENTRY: &str = "bqmoe_activate";
const COMBINE_ENTRY: &str = "bqmoe_combine_f32";

thread_local! {
    static FORMAT_PARSE_CALLS: Cell<u64> = const { Cell::new(0) };
    static WORKSPACE_LAYOUT_BUILDS: Cell<u64> = const { Cell::new(0) };
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockQuantizedMoePreparationCounts {
    pub format_parse_calls: u64,
    pub workspace_layout_builds: u64,
}

#[cfg(feature = "gpu-tests")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockQuantizedMoeTrafficFaultForTest {
    Poison,
    Overflow,
    StaleEpoch,
    ForeignRequest,
    WrongDevice,
    NonTopKMultipleCount,
}

pub fn block_quantized_moe_preparation_counts() -> BlockQuantizedMoePreparationCounts {
    BlockQuantizedMoePreparationCounts {
        format_parse_calls: FORMAT_PARSE_CALLS.with(Cell::get),
        workspace_layout_builds: WORKSPACE_LAYOUT_BUILDS.with(Cell::get),
    }
}

fn record_format_parse() {
    FORMAT_PARSE_CALLS.with(|calls| calls.set(calls.get() + 1));
}

fn record_workspace_layout_build() {
    WORKSPACE_LAYOUT_BUILDS.with(|builds| builds.set(builds.get() + 1));
}

/// NVRTC module cache key shared with the planar routed-MoE primitive
/// ([`super::planar_block_moe`]), which reuses the format-agnostic
/// `bqmoe_route`/`bqmoe_activate`/`bqmoe_combine_f32` kernels verbatim so the
/// routing/activation/combine arithmetic stays single-source.
pub(crate) const MOE_MODULE: &str = MODULE;
/// `bqmoe_route` entry point, reused by [`super::planar_block_moe`].
pub(crate) const MOE_ROUTE_ENTRY: &str = ROUTE_ENTRY;
/// `bqmoe_activate` entry point, reused by [`super::planar_block_moe`].
pub(crate) const MOE_ACTIVATE_ENTRY: &str = ACTIVATE_ENTRY;
/// `bqmoe_combine_f32` entry point, reused by [`super::planar_block_moe`].
pub(crate) const MOE_COMBINE_ENTRY: &str = COMBINE_ENTRY;

// Kernels appended after the shared `decode_weight`/`block_sum` prelude. The
// routing, activation, and combine kernels match the CPU oracle's arithmetic
// (total-order top-k, f64 tanh GELU, stable sigmoid SwiGLU); the linear kernel
// contracts activations against decoded GGUF weights in f32.
const KERNELS: &str = r#"
__device__ __forceinline__ int bqmoe_total_order_key(float value)
{
    int bits = __float_as_int(value);
    bits ^= (bits >> 31) & 0x7fffffff;
    return bits;
}

__device__ __forceinline__ bool bqmoe_route_is_better(
    float candidate, int candidate_index, float best, int best_index)
{
    const int candidate_key = bqmoe_total_order_key(candidate);
    const int best_key = bqmoe_total_order_key(best);
    return candidate_key > best_key
        || (candidate_key == best_key && candidate_index < best_index);
}

extern "C" __global__ void bqmoe_route(
    const float* router_logits,
    const float* router_weights,
    int* selected_experts,
    float* selected_weights,
    const unsigned long long rows,
    const int experts,
    const int top_k,
    const int normalize,
    unsigned int* route_telemetry_bitmap,
    unsigned int* route_telemetry_header)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long row = first; row < rows; row += stride) {
        const float* logits = router_logits + row * (unsigned long long)experts;
        int* indices = selected_experts + row * (unsigned long long)top_k;
        float* weights = selected_weights + row * (unsigned long long)top_k;

        for (int slot = 0; slot < top_k; ++slot) {
            int best_index = -1;
            float best_value = 0.0f;
            for (int expert = 0; expert < experts; ++expert) {
                bool already_selected = false;
                for (int previous = 0; previous < slot; ++previous) {
                    already_selected |= indices[previous] == expert;
                }
                if (already_selected) {
                    continue;
                }
                const float candidate = logits[expert];
                if (best_index < 0
                    || bqmoe_route_is_better(
                        candidate, expert, best_value, best_index)) {
                    best_index = expert;
                    best_value = candidate;
                }
            }
            indices[slot] = best_index;
        }

        // Fused, inert route telemetry (issue #1810 Slice 7A): one thread owns
        // this row and has finalized indices[0..top_k]. Mark once; the helper is
        // a no-op when telemetry pointers are null (disarmed), so the outputs
        // below are byte-identical.
        route_telemetry_mark_row(
            route_telemetry_bitmap, route_telemetry_header,
            indices, top_k, experts);

        if (router_weights) {
            const float* aggregation =
                router_weights + row * (unsigned long long)experts;
            float denominator = 1.0f;
            if (normalize) {
                denominator = 0.0f;
                for (int slot = 0; slot < top_k; ++slot) {
                    denominator += aggregation[indices[slot]];
                }
            }
            for (int slot = 0; slot < top_k; ++slot) {
                weights[slot] = denominator == 0.0f
                    ? 0.0f
                    : aggregation[indices[slot]] / denominator;
            }
            continue;
        }

        float maximum = -__int_as_float(0x7f800000);
        for (int expert = 0; expert < experts; ++expert) {
            maximum = fmaxf(maximum, logits[expert]);
        }
        float all_sum = 0.0f;
        for (int expert = 0; expert < experts; ++expert) {
            all_sum += expf(logits[expert] - maximum);
        }
        float denominator = all_sum;
        if (normalize) {
            denominator = 0.0f;
            for (int slot = 0; slot < top_k; ++slot) {
                denominator += expf(logits[indices[slot]] - maximum);
            }
        }
        for (int slot = 0; slot < top_k; ++slot) {
            weights[slot] =
                expf(logits[indices[slot]] - maximum) / denominator;
        }
    }
}

extern "C" __global__ void bqmoe_linear_f32(
    const float* input,
    const int* selected_experts,
    const unsigned char* packed,
    const float* bias,
    float* output,
    const unsigned long long routes,
    const int input_rows_are_routes,
    const int top_k,
    const int out_features,
    const int in_features,
    const int blocks,
    const int block_bytes,
    const int qk,
    const int format)
{
    const unsigned long long tasks = routes * (unsigned long long)out_features;
    const unsigned long long expert_weight_bytes =
        (unsigned long long)out_features
        * (unsigned long long)blocks
        * (unsigned long long)block_bytes;
    for (unsigned long long task = blockIdx.x; task < tasks; task += gridDim.x) {
        const unsigned long long route = task / out_features;
        const int output_feature = (int)(task % out_features);
        const int expert = selected_experts[route];
        const unsigned long long input_row =
            input_rows_are_routes ? route : route / (unsigned long long)top_k;
        const unsigned long long input_base =
            input_row * (unsigned long long)in_features;
        const unsigned char* expert_packed =
            packed + (unsigned long long)expert * expert_weight_bytes;
        float value = 0.0f;
        for (int depth = (int)threadIdx.x; depth < in_features;
             depth += (int)blockDim.x) {
            value += input[input_base + depth]
                * decode_weight(
                    expert_packed, format, qk, blocks, block_bytes,
                    output_feature, depth);
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

__device__ __forceinline__ float bqmoe_stable_sigmoid(float value)
{
    if (value >= 0.0f) {
        return 1.0f / (1.0f + expf(-value));
    }
    const float exponential = expf(value);
    return exponential / (1.0f + exponential);
}

__device__ __forceinline__ float bqmoe_swiglu_value(
    float gate, float linear, float alpha, float beta, float limit)
{
    const float bounded_gate = fminf(gate, limit);
    const float bounded_linear =
        isnan(linear) ? linear : fminf(fmaxf(linear, -limit), limit);
    return bounded_gate * bqmoe_stable_sigmoid(alpha * bounded_gate)
        * (bounded_linear + beta);
}

extern "C" __global__ void bqmoe_activate(
    const float* fc1,
    const float* fc3,
    float* activated,
    const unsigned long long routes,
    const int inter,
    const int activation,
    const int swiglu_fusion,
    const float alpha,
    const float beta,
    const float swiglu_limit)
{
    const unsigned long long total = routes * (unsigned long long)inter;
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long index = first; index < total; index += stride) {
        const unsigned long long route = index / inter;
        const int feature = (int)(index % inter);
        const unsigned long long base =
            route * (unsigned long long)(activation == 3 && swiglu_fusion != 0
                ? inter * 2
                : inter);
        const float value = fc1[base + feature];
        if (activation == 0) {
            activated[index] = fmaxf(value, 0.0f);
        } else if (activation == 1) {
            const double x = (double)value;
            const double inner =
                0.7978845608028654 * (x + 0.044715 * x * x * x);
            activated[index] = (float)(0.5 * x * (1.0 + tanh(inner)));
        } else if (activation == 2 && !fc3) {
            activated[index] = value * bqmoe_stable_sigmoid(value);
        } else if (activation == 4) {
            activated[index] = value;
        } else {
            float gate;
            float linear;
            if (fc3) {
                gate = value;
                linear = fc3[index];
            } else if (swiglu_fusion == 1) {
                gate = fc1[base + 2 * feature];
                linear = fc1[base + 2 * feature + 1];
            } else {
                gate = value;
                linear = fc1[base + inter + feature];
            }
            activated[index] =
                bqmoe_swiglu_value(gate, linear, alpha, beta, swiglu_limit);
        }
    }
}

extern "C" __global__ void bqmoe_combine_f32(
    const float* route_output,
    const float* selected_weights,
    float* output,
    const unsigned long long rows,
    const int hidden,
    const int top_k)
{
    const unsigned long long total = rows * (unsigned long long)hidden;
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long index = first; index < total; index += stride) {
        const unsigned long long row = index / hidden;
        const int feature = (int)(index % hidden);
        float value = 0.0f;
        for (int slot = 0; slot < top_k; ++slot) {
            const unsigned long long route =
                row * (unsigned long long)top_k + slot;
            value += selected_weights[route]
                * route_output[route * (unsigned long long)hidden + feature];
        }
        output[index] = value;
    }
}
"#;

fn module_source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| {
        let mut source = decoder_prelude();
        // Shared route-telemetry `__device__` helpers (issue #1810 Slice 7A) so
        // `bqmoe_route`'s fused `route_telemetry_mark_row` call resolves. Uses
        // only integer atomics; inert when the route kernel is passed null
        // telemetry pointers.
        source.push_str(MARK_DEVICE_SRC);
        source.push_str(KERNELS);
        source
    })
}

/// The compiled NVRTC source for [`MOE_MODULE`], reused verbatim by
/// [`super::planar_block_moe`] so the routing/activation/combine kernels stay
/// single-source.
pub(crate) fn moe_module_source() -> &'static str {
    module_source()
}

#[derive(Clone, Copy, Debug)]
struct MoeAttributes {
    k: usize,
    activation: Activation,
    normalize_routing_weights: bool,
    swiglu_fusion: usize,
    activation_alpha: f32,
    activation_beta: f32,
    swiglu_limit: f32,
}

impl MoeAttributes {
    fn from_node(node: &Node) -> Result<Self> {
        for name in node.attributes.keys() {
            if !matches!(
                name.as_str(),
                "k" | "activation_type"
                    | "normalize_routing_weights"
                    | "swiglu_fusion"
                    | "activation_alpha"
                    | "activation_beta"
                    | "swiglu_limit"
                    | "fc1_format"
                    | "fc2_format"
                    | "fc3_format"
                    | "block_layout_version"
                    | "fc1_block_size_out"
                    | "fc1_block_size_in"
                    | "fc2_block_size_out"
                    | "fc2_block_size_in"
                    | "fc3_block_size_out"
                    | "fc3_block_size_in"
            ) {
                return Err(error(format!(
                    "attribute '{name}' is not part of the BlockQuantizedMoE ABI"
                )));
            }
        }
        let k = int_attr(node, "k", 1)?;
        if k <= 0 {
            return Err(error(format!("k must be > 0, got {k}")));
        }
        let activation_name = match node.attr("activation_type") {
            Some(value) => value
                .as_str()
                .ok_or_else(|| error("attribute activation_type must be a string"))?,
            None => "relu",
        };
        let normalize_routing_weights = bool_attr(node, "normalize_routing_weights", false)?;
        let swiglu_fusion = int_attr(node, "swiglu_fusion", 0)?;
        let activation_alpha = float_attr(node, "activation_alpha", 1.0)?;
        let activation_beta = float_attr(node, "activation_beta", 0.0)?;
        let swiglu_limit = float_attr(node, "swiglu_limit", DEFAULT_SWIGLU_LIMIT)?;
        let activation_attributes = validate_moe_activation_attributes(
            activation_name,
            swiglu_fusion,
            activation_alpha,
            activation_beta,
            swiglu_limit,
        )
        .map_err(error)?;
        Ok(Self {
            k: usize::try_from(k).map_err(|_| error("k exceeds usize limits"))?,
            activation: activation_attributes.activation,
            normalize_routing_weights,
            swiglu_fusion: activation_attributes.swiglu_fusion,
            activation_alpha: activation_attributes.activation_alpha,
            activation_beta: activation_attributes.activation_beta,
            swiglu_limit: activation_attributes.swiglu_limit,
        })
    }

    fn fc1_size(self, inter: usize) -> Result<usize> {
        if self.activation == Activation::Swiglu && self.swiglu_fusion != 0 {
            inter
                .checked_mul(2)
                .ok_or_else(|| error("fused SwiGLU FC1 width exceeds usize limits"))
        } else {
            Ok(inter)
        }
    }

    fn uses_separate_gate(self, has_fc3: bool) -> bool {
        (self.activation == Activation::Swiglu && self.swiglu_fusion == 0)
            || (self.activation == Activation::Silu && has_fc3)
    }
}

fn parse_layout_version(node: &Node) -> Result<()> {
    require_layout_v1(node, OP).map_err(error)
}

/// Per-projection native formats for CUDA validation and dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProjectionFormats {
    fc1: BlockFormat,
    fc2: BlockFormat,
    fc3: Option<BlockFormat>,
}

#[derive(Clone, Copy, Debug)]
struct PlanarProjectionFormats {
    fc1: PlanarBlockGeometry,
    fc2: PlanarBlockGeometry,
    fc3: Option<PlanarBlockGeometry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlanarBankIdentity {
    packed: CUdeviceptr,
    scale: CUdeviceptr,
}

struct PlanarKernelState {
    formats: PlanarProjectionFormats,
    constant_inputs: [bool; INPUT_COUNT],
    validation_scratch: CUdeviceptr,
    validated_banks: Mutex<[Option<PlanarBankIdentity>; 3]>,
}

fn parse_planar_projection_formats(node: &Node) -> Result<Option<PlanarProjectionFormats>> {
    let fc1 = planar_geometry_from_node(
        node,
        OP,
        "fc1_format",
        "fc1_block_size_out",
        "fc1_block_size_in",
    )
    .map_err(error)?;
    let fc2 = planar_geometry_from_node(
        node,
        OP,
        "fc2_format",
        "fc2_block_size_out",
        "fc2_block_size_in",
    )
    .map_err(error)?;
    let fc3_wired = node
        .inputs
        .get(BQMOE_FC3_WEIGHT)
        .is_some_and(Option::is_some);
    let fc3 = if fc3_wired {
        planar_geometry_from_node(
            node,
            OP,
            "fc3_format",
            "fc3_block_size_out",
            "fc3_block_size_in",
        )
        .map_err(error)?
    } else {
        None
    };
    let any_planar = fc1.is_some() || fc2.is_some() || fc3.is_some();
    if !any_planar {
        return Ok(None);
    }
    match (fc1, fc2, fc3_wired, fc3) {
        (Some(fc1), Some(fc2), false, None) => Ok(Some(PlanarProjectionFormats {
            fc1,
            fc2,
            fc3: None,
        })),
        (Some(fc1), Some(fc2), true, Some(fc3)) => Ok(Some(PlanarProjectionFormats {
            fc1,
            fc2,
            fc3: Some(fc3),
        })),
        _ => Err(error(
            "CUDA planar MoE requires every wired projection to use block_fp8 or fp4_planar; mixing planar and interleaved projections is unsupported",
        )),
    }
}

/// Immutable host description of one expert-major interleaved GGUF projection.
#[derive(Clone, Copy, Debug)]
pub struct BlockQuantizedMoeBank<'a> {
    pub format: &'a str,
    pub packed: &'a [u8],
    pub experts: usize,
    pub out_features: usize,
    pub in_features: usize,
}

/// Diagnostic content identity for one admitted projection bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockQuantizedMoeBankIdentity(u64);

impl BlockQuantizedMoeBankIdentity {
    pub fn digest(self) -> u64 {
        self.0
    }
}

/// Residency granularity honestly exposed by the current sealed admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockQuantizedMoeResidency {
    WholeProjectionBank,
}

struct AdmittedProjectionBank {
    allocation: Arc<dyn SealedDeviceAllocation>,
    identity: BlockQuantizedMoeBankIdentity,
    experts: usize,
    out_features: usize,
    in_features: usize,
    blocks: usize,
    format: BlockFormat,
    shape: [usize; 4],
    strides: [i64; 4],
    bytes_per_expert: usize,
    total_bytes: usize,
}

struct AdmittedBlockQuantizedMoeBankSet {
    fc1: AdmittedProjectionBank,
    fc2: AdmittedProjectionBank,
    fc3: Option<AdmittedProjectionBank>,
    device: onnx_runtime_ir::DeviceId,
    provider_context: ProviderContextIdentity,
    runtime_identity: usize,
}

/// Sealed owner for immutable FC1/FC2/(optional) FC3 projection banks.
///
/// This admission validates exact full-block geometry and reserved/non-finite
/// scale codes before upload. It deliberately exposes no device pointer or
/// mutable allocation. Current residency remains whole-projection-bank; a
/// selected-expert pager must mint a different admission type.
pub struct AdmittedBlockQuantizedMoeBanks {
    banks: Arc<AdmittedBlockQuantizedMoeBankSet>,
}

impl AdmittedBlockQuantizedMoeBanks {
    pub fn diagnostic_identities(&self) -> [Option<BlockQuantizedMoeBankIdentity>; 3] {
        [
            Some(self.banks.fc1.identity),
            Some(self.banks.fc2.identity),
            self.banks.fc3.as_ref().map(|bank| bank.identity),
        ]
    }

    pub fn device(&self) -> onnx_runtime_ir::DeviceId {
        self.banks.device
    }

    pub fn provider_context(&self) -> ProviderContextIdentity {
        self.banks.provider_context
    }

    pub fn residency(&self) -> BlockQuantizedMoeResidency {
        BlockQuantizedMoeResidency::WholeProjectionBank
    }

    pub fn projection_count(&self) -> usize {
        2 + usize::from(self.banks.fc3.is_some())
    }

    pub fn no_residency_traffic(
        &self,
        selected_experts: &[usize],
    ) -> Result<BlockQuantizedMoeTraffic> {
        let experts = self.banks.fc1.experts;
        if selected_experts.iter().any(|&expert| expert >= experts) {
            return Err(error(format!(
                "selected expert is outside admitted range 0..{experts}"
            )));
        }
        let mut selected = vec![false; experts];
        for &expert in selected_experts {
            selected[expert] = true;
        }
        let selected_count = selected.into_iter().filter(|selected| *selected).count();
        let projections = [&self.banks.fc1, &self.banks.fc2];
        let uploaded_whole_bank_bytes = projections
            .iter()
            .map(|bank| bank.total_bytes)
            .chain(self.banks.fc3.iter().map(|bank| bank.total_bytes))
            .try_fold(0usize, |total, bytes| total.checked_add(bytes))
            .ok_or_else(|| error("fixed load byte count overflow"))?;
        let bytes_per_expert = projections
            .iter()
            .map(|bank| bank.bytes_per_expert)
            .chain(self.banks.fc3.iter().map(|bank| bank.bytes_per_expert))
            .try_fold(0usize, |total, bytes| total.checked_add(bytes))
            .ok_or_else(|| error("per-expert byte count overflow"))?;
        let logical_route_demand_bytes = bytes_per_expert
            .checked_mul(selected_experts.len())
            .ok_or_else(|| error("logical route-demand byte count overflow"))?;
        let unique_selected_expert_bytes = bytes_per_expert
            .checked_mul(selected_count)
            .ok_or_else(|| error("unique selected-expert byte count overflow"))?;
        Ok(BlockQuantizedMoeTraffic {
            uploaded_whole_bank_bytes: as_u64(
                "uploaded whole-bank bytes",
                uploaded_whole_bank_bytes,
            )?,
            committed_whole_bank_bytes: as_u64(
                "committed whole-bank bytes",
                uploaded_whole_bank_bytes,
            )?,
            logical_route_demand_bytes: as_u64(
                "logical route-demand bytes",
                logical_route_demand_bytes,
            )?,
            unique_selected_expert_bytes: as_u64(
                "unique selected-expert bytes",
                unique_selected_expert_bytes,
            )?,
            physical_dram_bytes: None,
            page_ins: 0,
            byte_hit_rate: None,
        })
    }
}

fn bank_identity(
    format: BlockFormat,
    experts: usize,
    out_features: usize,
    in_features: usize,
    packed: &[u8],
) -> BlockQuantizedMoeBankIdentity {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in [
        format.kernel_id() as u64,
        experts as u64,
        out_features as u64,
        in_features as u64,
    ] {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for &byte in packed {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    BlockQuantizedMoeBankIdentity(hash)
}

fn validate_finite_f16(label: &str, block_index: usize, field: &str, bytes: &[u8]) -> Result<()> {
    let value = half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
    if !value.is_finite() {
        return Err(error(format!(
            "{label} block {block_index} has non-finite {field} scale"
        )));
    }
    Ok(())
}

fn validate_block_values(label: &str, format: BlockFormat, packed: &[u8]) -> Result<()> {
    let block_bytes = format.block_bytes();
    for (block_index, block) in packed.chunks_exact(block_bytes).enumerate() {
        match format {
            BlockFormat::Mxfp4 => {
                if block[0] == 0xff {
                    return Err(error(format!(
                        "{label} block {block_index} uses reserved MXFP4 E8M0 code 0xff"
                    )));
                }
            }
            BlockFormat::Iq1M => {
                let scales = &block[48..56];
                let words = [
                    u16::from_le_bytes([scales[0], scales[1]]),
                    u16::from_le_bytes([scales[2], scales[3]]),
                    u16::from_le_bytes([scales[4], scales[5]]),
                    u16::from_le_bytes([scales[6], scales[7]]),
                ];
                let scale_bits = (words[0] >> 12)
                    | ((words[1] >> 8) & 0x00f0)
                    | ((words[2] >> 4) & 0x0f00)
                    | (words[3] & 0xf000);
                validate_finite_f16(label, block_index, "IQ1_M", &scale_bits.to_le_bytes())?;
            }
            BlockFormat::Q2K => {
                validate_finite_f16(label, block_index, "Q2_K d", &block[80..82])?;
                validate_finite_f16(label, block_index, "Q2_K dmin", &block[82..84])?;
            }
            BlockFormat::Q3K => {
                validate_finite_f16(label, block_index, "Q3_K d", &block[108..110])?;
            }
            BlockFormat::Q5K => {
                validate_finite_f16(label, block_index, "Q5_K d", &block[..2])?;
                validate_finite_f16(label, block_index, "Q5_K dmin", &block[2..4])?;
            }
            BlockFormat::Q6K => {
                validate_finite_f16(label, block_index, "Q6_K d", &block[208..210])?;
            }
            _ => validate_finite_f16(label, block_index, "block", &block[..2])?,
        }
    }
    Ok(())
}

fn validate_host_bank(
    label: &str,
    bank: BlockQuantizedMoeBank<'_>,
) -> Result<(BlockFormat, BlockQuantizedMoeBankIdentity)> {
    record_format_parse();
    let format = BlockFormat::parse(bank.format)?;
    if bank.experts == 0 || bank.out_features == 0 || bank.in_features == 0 {
        return Err(error(format!(
            "{label} requires positive experts/out/in, got {}/{}/{}",
            bank.experts, bank.out_features, bank.in_features
        )));
    }
    if !bank.in_features.is_multiple_of(format.qk()) {
        return Err(error(format!(
            "{label} input width {} has a partial {:?} block tail; full blocks of {} are required",
            bank.in_features,
            format,
            format.qk()
        )));
    }
    let expected = bank
        .experts
        .checked_mul(bank.out_features)
        .and_then(|value| value.checked_mul(bank.in_features / format.qk()))
        .and_then(|value| value.checked_mul(format.block_bytes()))
        .ok_or_else(|| error(format!("{label} packed byte count overflow")))?;
    if bank.packed.len() != expected {
        return Err(error(format!(
            "{label} has {} bytes, expected {expected}",
            bank.packed.len()
        )));
    }
    validate_block_values(label, format, bank.packed)?;
    Ok((
        format,
        bank_identity(
            format,
            bank.experts,
            bank.out_features,
            bank.in_features,
            bank.packed,
        ),
    ))
}

/// Validate all projection banks before uploading any bytes, then return one
/// non-cloneable sealed owner tied to the selected CUDA provider context.
pub fn admit_block_quantized_moe_banks(
    provider: &Arc<CudaExecutionProvider>,
    fc1: BlockQuantizedMoeBank<'_>,
    fc2: BlockQuantizedMoeBank<'_>,
    fc3: Option<BlockQuantizedMoeBank<'_>>,
) -> Result<AdmittedBlockQuantizedMoeBanks> {
    let banks = admit_block_quantized_moe_bank_set(provider.as_ref(), fc1, fc2, fc3)?;
    Ok(AdmittedBlockQuantizedMoeBanks { banks })
}

fn admit_block_quantized_moe_bank_set(
    provider: &dyn ExecutionProvider,
    fc1: BlockQuantizedMoeBank<'_>,
    fc2: BlockQuantizedMoeBank<'_>,
    fc3: Option<BlockQuantizedMoeBank<'_>>,
) -> Result<Arc<AdmittedBlockQuantizedMoeBankSet>> {
    let provider_context = provider.provider_context_identity().ok_or_else(|| {
        error(format!(
            "{} does not expose a sealed-allocation context",
            provider.name()
        ))
    })?;
    let runtime_identity = provider.runtime_identity().ok_or_else(|| {
        error(format!(
            "{} does not expose a sealed-allocation runtime",
            provider.name()
        ))
    })?;
    let unfused = fc1.out_features == fc2.in_features;
    let fused = fc3.is_none()
        && fc2
            .in_features
            .checked_mul(2)
            .is_some_and(|width| fc1.out_features == width);
    if fc1.experts != fc2.experts || fc1.in_features != fc2.out_features || (!unfused && !fused) {
        return Err(error(
            "fc1/fc2 bank dimensions do not form one expert pipeline",
        ));
    }
    if let Some(fc3) = fc3
        && (fc3.experts != fc1.experts
            || fc3.in_features != fc1.in_features
            || fc3.out_features != fc1.out_features)
    {
        return Err(error("fc3 bank dimensions do not match fc1"));
    }
    let (fc1_format, fc1_identity) = validate_host_bank("fc1", fc1)?;
    let (fc2_format, fc2_identity) = validate_host_bank("fc2", fc2)?;
    let fc3_validation = fc3
        .map(|bank| validate_host_bank("fc3", bank))
        .transpose()?;
    let upload = |label: &str, bank: BlockQuantizedMoeBank<'_>, format: BlockFormat, identity| {
        let bytes_per_expert = bank.packed.len() / bank.experts;
        provider
            .upload_sealed_constant(bank.packed, 256)
            .map(|allocation| AdmittedProjectionBank {
                allocation,
                identity,
                experts: bank.experts,
                out_features: bank.out_features,
                in_features: bank.in_features,
                blocks: bank.in_features / format.qk(),
                format,
                shape: [
                    bank.experts,
                    bank.out_features,
                    bank.in_features / format.qk(),
                    format.block_bytes(),
                ],
                strides: [
                    (bank.out_features * (bank.in_features / format.qk()) * format.block_bytes())
                        as i64,
                    ((bank.in_features / format.qk()) * format.block_bytes()) as i64,
                    format.block_bytes() as i64,
                    1,
                ],
                bytes_per_expert,
                total_bytes: bank.packed.len(),
            })
            .map_err(|err| error(format!("upload immutable {label} bank: {err}")))
    };
    let fc1 = upload("fc1", fc1, fc1_format, fc1_identity)?;
    let fc2 = upload("fc2", fc2, fc2_format, fc2_identity)?;
    let fc3 = match (fc3, fc3_validation) {
        (Some(bank), Some((format, identity))) => Some(upload("fc3", bank, format, identity)?),
        (None, None) => None,
        _ => return Err(error("internal fc3 admission mismatch")),
    };
    for (label, bank) in [
        ("fc1", Some(&fc1)),
        ("fc2", Some(&fc2)),
        ("fc3", fc3.as_ref()),
    ] {
        let Some(bank) = bank else {
            continue;
        };
        if bank.allocation.ptr().is_null()
            || bank.allocation.len() != bank.total_bytes
            || bank.allocation.device() != provider.device_id()
            || bank.allocation.provider_context() != provider_context
            || bank.allocation.runtime_identity() != runtime_identity
        {
            return Err(error(format!(
                "{label} sealed allocation does not match the admitting provider"
            )));
        }
    }
    Ok(Arc::new(AdmittedBlockQuantizedMoeBankSet {
        fc1,
        fc2,
        fc3,
        device: provider.device_id(),
        provider_context,
        runtime_identity,
    }))
}

fn parse_format_attr(node: &Node, name: &str) -> Result<BlockFormat> {
    record_format_parse();
    node.attr(name)
        .ok_or_else(|| error(format!("missing required string attribute '{name}'")))?
        .as_str()
        .ok_or_else(|| error(format!("attribute '{name}' must be a UTF-8 string")))
        .and_then(BlockFormat::parse)
}

fn optional_format_attr(node: &Node, name: &str) -> Result<Option<BlockFormat>> {
    match node.attr(name) {
        None => Ok(None),
        Some(attribute) => attribute
            .as_str()
            .ok_or_else(|| error(format!("attribute '{name}' must be a UTF-8 string")))
            .and_then(|format| {
                record_format_parse();
                BlockFormat::parse(format)
            })
            .map(Some),
    }
}

/// Parse the per-projection formats. `fc3_format` must be present exactly when
/// the `fc3_experts_weights` input (index 6) is wired on the node.
fn parse_projection_formats(node: &Node) -> Result<ProjectionFormats> {
    let fc1 = parse_format_attr(node, "fc1_format")?;
    let fc2 = parse_format_attr(node, "fc2_format")?;
    let fc3_wired = node.inputs.get(6).is_some_and(Option::is_some);
    let fc3_attr = optional_format_attr(node, "fc3_format")?;
    match (fc3_wired, fc3_attr) {
        (true, Some(fc3)) => Ok(ProjectionFormats {
            fc1,
            fc2,
            fc3: Some(fc3),
        }),
        (true, None) => Err(error(
            "fc3_experts_weights is wired but the required fc3_format attribute is missing",
        )),
        (false, Some(_)) => Err(error(
            "fc3_format is only valid when fc3_experts_weights is wired",
        )),
        (false, None) => Ok(ProjectionFormats {
            fc1,
            fc2,
            fc3: None,
        }),
    }
}

/// Claim-gate variant of [`parse_projection_formats`] that yields
/// `Cow<'static, str>` rejection reasons (with the CUDA re-export guidance) so
/// [`unsupported_reason`] can decline a node without constructing an [`EpError`].
fn claim_projection_formats(
    node: &Node,
) -> std::result::Result<ProjectionFormats, Cow<'static, str>> {
    let fc1 = claim_format_attr(node, "fc1_format")?.ok_or(Cow::Borrowed(
        "BlockQuantizedMoE: missing required string attribute 'fc1_format'",
    ))?;
    let fc2 = claim_format_attr(node, "fc2_format")?.ok_or(Cow::Borrowed(
        "BlockQuantizedMoE: missing required string attribute 'fc2_format'",
    ))?;
    let fc3 = claim_format_attr(node, "fc3_format")?;
    let fc3_wired = node.inputs.get(6).is_some_and(Option::is_some);
    match (fc3_wired, fc3) {
        (true, Some(fc3)) => Ok(ProjectionFormats {
            fc1,
            fc2,
            fc3: Some(fc3),
        }),
        (true, None) => Err(Cow::Borrowed(
            "BlockQuantizedMoE: fc3_experts_weights is wired but the required fc3_format attribute is missing",
        )),
        (false, Some(_)) => Err(Cow::Borrowed(
            "BlockQuantizedMoE: fc3_format is only valid when fc3_experts_weights is wired",
        )),
        (false, None) => Ok(ProjectionFormats {
            fc1,
            fc2,
            fc3: None,
        }),
    }
}

/// Parse one projection-format attribute for the claim gate. `Ok(None)` means
/// the attribute is absent; a present-but-invalid value is a typed rejection.
fn claim_format_attr(
    node: &Node,
    name: &str,
) -> std::result::Result<Option<BlockFormat>, Cow<'static, str>> {
    let Some(attribute) = node.attr(name) else {
        return Ok(None);
    };
    let Some(text) = attribute.as_str() else {
        return Err(Cow::Owned(format!(
            "BlockQuantizedMoE: attribute '{name}' must be a string naming a CUDA-supported block format"
        )));
    };
    match BlockFormat::parse(text) {
        Ok(format) => Ok(Some(format)),
        Err(_) => Err(Cow::Owned(format!(
            "BlockQuantizedMoE: CUDA does not support format '{text}' for '{name}'"
        ))),
    }
}

fn validate_planar_claim(
    node: &Node,
    shapes: &[Shape],
    dtypes: &[DataType],
    formats: PlanarProjectionFormats,
) -> std::result::Result<(), String> {
    for &index in &[0usize, 1, BQMOE_FC1_WEIGHT, BQMOE_FC2_WEIGHT] {
        if node.inputs[index].is_none() {
            return Err(format!(
                "required input {index} ('{}') is omitted",
                INPUT_NAMES[index]
            ));
        }
    }
    for index in [0usize, 1, 3, 5, 7, 8] {
        if node.inputs[index].is_some() && dtypes[index] != DataType::Float32 {
            return Err(format!(
                "input {index} ('{}') must be Float32, got {:?}",
                INPUT_NAMES[index], dtypes[index]
            ));
        }
    }
    for (weight, scale, format) in [
        (BQMOE_FC1_WEIGHT, BQMOE_FC1_SCALE, Some(formats.fc1)),
        (BQMOE_FC2_WEIGHT, BQMOE_FC2_SCALE, Some(formats.fc2)),
        (BQMOE_FC3_WEIGHT, BQMOE_FC3_SCALE, formats.fc3),
    ] {
        let Some(format) = format else {
            if node.inputs[scale].is_some() {
                return Err(format!("{} requires fc3 weights", INPUT_NAMES[scale]));
            }
            continue;
        };
        if node.inputs[weight].is_none() || node.inputs[scale].is_none() {
            return Err(format!(
                "{} and {} are both required for planar format",
                INPUT_NAMES[weight], INPUT_NAMES[scale]
            ));
        }
        if dtypes[weight] != format.format.weight_dtype()
            || dtypes[scale] != format.format.scale_dtype()
        {
            return Err(format!(
                "{} / {} must have dtypes {:?} / {:?}",
                INPUT_NAMES[weight],
                INPUT_NAMES[scale],
                format.format.weight_dtype(),
                format.format.scale_dtype()
            ));
        }
        if shapes[weight].len() != 3 || shapes[scale].len() != 3 {
            return Err(format!(
                "{} and {} must both have rank 3",
                INPUT_NAMES[weight], INPUT_NAMES[scale]
            ));
        }
    }
    if !matches!(shapes[0].len(), 2 | 3) || shapes[1].len() != 2 {
        return Err("input/router ranks must be 2-or-3/2".into());
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct PreparedMoeGeometry {
    input_shapes: Vec<Vec<usize>>,
    rows: usize,
    hidden: usize,
    experts: usize,
    inter: usize,
    fc1_out: usize,
    routes: usize,
    has_fc3: bool,
    layout: QmoeWorkspaceLayout,
}

impl PreparedMoeGeometry {
    fn new(
        node: &Node,
        input_shapes: &[Vec<usize>],
        attributes: MoeAttributes,
        formats: ProjectionFormats,
    ) -> Result<Self> {
        if input_shapes.len() != INPUT_COUNT || input_shapes.len() != node.inputs.len() {
            return Err(error(format!(
                "expected exactly {INPUT_COUNT} positional input shapes, got {} shapes for {} node inputs",
                input_shapes.len(),
                node.inputs.len()
            )));
        }
        for index in [BQMOE_FC1_SCALE, BQMOE_FC2_SCALE, BQMOE_FC3_SCALE] {
            if node.inputs[index].is_some() || !input_shapes[index].is_empty() {
                return Err(error(format!(
                    "{} must be absent for interleaved projection formats",
                    INPUT_NAMES[index]
                )));
            }
        }
        for &index in &[0usize, 1, 2, 4] {
            if node.inputs[index].is_none() || input_shapes[index].is_empty() {
                return Err(error(format!(
                    "required input {index} ('{}') is absent",
                    INPUT_NAMES[index]
                )));
            }
        }
        let input_shape = &input_shapes[0];
        if !matches!(input_shape.len(), 2 | 3) {
            return Err(error(format!(
                "input must be 2-D [rows, hidden] or 3-D [batch, sequence, hidden], got {input_shape:?}"
            )));
        }
        let hidden = *input_shape
            .last()
            .ok_or_else(|| error("input rank unexpectedly empty"))?;
        let rows = checked_product(
            &input_shape[..input_shape.len() - 1],
            "flattened input row count",
        )?;
        let router_shape = &input_shapes[1];
        if router_shape.len() != 2 {
            return Err(error(format!(
                "router_logits must be 2-D [rows, experts], got {router_shape:?}"
            )));
        }
        require_prepared_shape("router_logits", router_shape, &[rows, router_shape[1]])?;
        let experts = router_shape[1];
        if attributes.k == 0 || attributes.k > experts {
            return Err(error(format!(
                "requires 0 < k <= num_experts, got k={} and num_experts={experts}",
                attributes.k
            )));
        }

        let fc1_shape = &input_shapes[2];
        if fc1_shape.len() != 4 {
            return Err(error(format!(
                "fc1_experts_weights must be rank 4, got {fc1_shape:?}"
            )));
        }
        let fc1_out = fc1_shape[1];
        let inter = if attributes.swiglu_fusion == 0 {
            fc1_out
        } else {
            if !fc1_out.is_multiple_of(2) {
                return Err(error(format!(
                    "fused SwiGLU fc1_out must be even, got {fc1_out}"
                )));
            }
            fc1_out / 2
        };
        if inter == 0 {
            return Err(error("inferred inter dimension must be non-zero"));
        }
        let expected_fc1 = attributes.fc1_size(inter)?;
        if fc1_out != expected_fc1 {
            return Err(error(format!(
                "fc1_experts_weights dimension 1 must be {expected_fc1}, got {fc1_out}"
            )));
        }
        require_prepared_shape(
            "fc1_experts_weights",
            fc1_shape,
            &[
                experts,
                fc1_out,
                hidden
                    .checked_div(formats.fc1.qk())
                    .filter(|_| hidden.is_multiple_of(formats.fc1.qk()))
                    .ok_or_else(|| error("fc1 input width has a partial block tail"))?,
                formats.fc1.block_bytes(),
            ],
        )?;
        require_optional_prepared_shape(
            node,
            input_shapes,
            3,
            "fc1_experts_bias",
            &[experts, fc1_out],
        )?;
        require_prepared_shape(
            "fc2_experts_weights",
            &input_shapes[4],
            &[
                experts,
                hidden,
                inter
                    .checked_div(formats.fc2.qk())
                    .filter(|_| inter.is_multiple_of(formats.fc2.qk()))
                    .ok_or_else(|| error("fc2 input width has a partial block tail"))?,
                formats.fc2.block_bytes(),
            ],
        )?;
        require_optional_prepared_shape(
            node,
            input_shapes,
            5,
            "fc2_experts_bias",
            &[experts, hidden],
        )?;

        let fc3_wired = node.inputs.get(6).is_some_and(Option::is_some);
        let uses_separate_gate = attributes.uses_separate_gate(fc3_wired);
        if fc3_wired != uses_separate_gate {
            return Err(error(
                "fc3_experts_weights is only valid for unfused swiglu or silu gated-GLU",
            ));
        }
        if fc3_wired {
            let format = formats
                .fc3
                .ok_or_else(|| error("fc3 weights require fc3_format"))?;
            require_prepared_shape(
                "fc3_experts_weights",
                &input_shapes[6],
                &[
                    experts,
                    inter,
                    hidden
                        .checked_div(format.qk())
                        .filter(|_| hidden.is_multiple_of(format.qk()))
                        .ok_or_else(|| error("fc3 input width has a partial block tail"))?,
                    format.block_bytes(),
                ],
            )?;
            require_optional_prepared_shape(
                node,
                input_shapes,
                7,
                "fc3_experts_bias",
                &[experts, inter],
            )?;
        } else if node.inputs.get(7).is_some_and(Option::is_some) {
            return Err(error(
                "fc3_experts_bias is invalid without fc3_experts_weights",
            ));
        }
        require_optional_prepared_shape(node, input_shapes, 8, "router_weights", &[rows, experts])?;

        let layout = qmoe_workspace_layout(
            input_shape,
            router_shape,
            fc1_shape,
            attributes.k,
            attributes.swiglu_fusion,
            fc3_wired,
        )?;
        let routes = checked_product(&[rows, attributes.k], "route count")?;
        Ok(Self {
            input_shapes: input_shapes.to_vec(),
            rows,
            hidden,
            experts,
            inter,
            fc1_out,
            routes,
            has_fc3: fc3_wired,
            layout,
        })
    }
}

fn require_prepared_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        return Err(error(format!(
            "{name} shape {actual:?} does not match required {expected:?}"
        )));
    }
    Ok(())
}

fn require_optional_prepared_shape(
    node: &Node,
    input_shapes: &[Vec<usize>],
    index: usize,
    name: &str,
    expected: &[usize],
) -> Result<()> {
    match node.inputs.get(index).copied().flatten() {
        Some(_) => require_prepared_shape(name, &input_shapes[index], expected),
        None => {
            if input_shapes
                .get(index)
                .is_some_and(|shape| !shape.is_empty())
            {
                return Err(error(format!(
                    "{name} has a concrete shape but the node input is absent"
                )));
            }
            Ok(())
        }
    }
}

struct PreparedMoeLaunches {
    route: RawCudaFunction,
    linear: RawCudaFunction,
    activate: RawCudaFunction,
    combine: RawCudaFunction,
    route_config: LaunchConfig,
    fc1_config: LaunchConfig,
    fc2_config: LaunchConfig,
    fc3_config: Option<LaunchConfig>,
    activate_config: LaunchConfig,
    combine_config: LaunchConfig,
}

impl PreparedMoeLaunches {
    fn new(runtime: &CudaRuntime, geometry: &PreparedMoeGeometry) -> Result<Self> {
        let linear_for_config = runtime.nvrtc_function(MODULE, module_source(), LINEAR_ENTRY)?;
        let route = runtime.nvrtc_raw_function(MODULE, module_source(), ROUTE_ENTRY)?;
        let linear = runtime.nvrtc_raw_function(MODULE, module_source(), LINEAR_ENTRY)?;
        let activate = runtime.nvrtc_raw_function(MODULE, module_source(), ACTIVATE_ENTRY)?;
        let combine = runtime.nvrtc_raw_function(MODULE, module_source(), COMBINE_ENTRY)?;
        let threads = preferred_threads_for(runtime);
        let linear_config = |out_features: usize| {
            let tasks = checked_product(
                &[geometry.routes, out_features],
                "prepared linear task count",
            )?;
            runtime.reduction_launch_config(
                &linear_for_config,
                saturating_grid_for(runtime, as_u64("prepared linear task count", tasks)?),
                threads,
                std::mem::size_of::<f32>() as u32,
            )
        };
        Ok(Self {
            route_config: pointwise_config_for(runtime, geometry.rows as u64),
            fc1_config: linear_config(geometry.fc1_out)?,
            fc2_config: linear_config(geometry.hidden)?,
            fc3_config: geometry
                .has_fc3
                .then(|| linear_config(geometry.inter))
                .transpose()?,
            activate_config: pointwise_config_for(
                runtime,
                checked_product(
                    &[geometry.routes, geometry.inter],
                    "prepared activation element count",
                )? as u64,
            ),
            combine_config: pointwise_config_for(
                runtime,
                checked_product(
                    &[geometry.rows, geometry.hidden],
                    "prepared output element count",
                )? as u64,
            ),
            route,
            linear,
            activate,
            combine,
        })
    }
}

fn preferred_threads_for(runtime: &CudaRuntime) -> u32 {
    let capabilities = runtime.capabilities();
    let preferred = if capabilities.compute_capability().0 >= 7 {
        256
    } else {
        128
    };
    preferred.min(capabilities.max_threads_per_block()).max(1)
}

fn saturating_grid_for(runtime: &CudaRuntime, units: u64) -> u32 {
    let saturation = u64::from(runtime.capabilities().multiprocessor_count()).saturating_mul(16);
    units.min(saturation.max(1)).min(u64::from(u32::MAX)).max(1) as u32
}

fn pointwise_config_for(runtime: &CudaRuntime, total: u64) -> LaunchConfig {
    let threads = preferred_threads_for(runtime);
    let blocks_needed = total.div_ceil(u64::from(threads)).max(1);
    LaunchConfig {
        grid_dim: (saturating_grid_for(runtime, blocks_needed), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(feature = "gpu-tests")]
pub struct BlockQuantizedMoEFactory {
    pub runtime: Arc<CudaRuntime>,
}

#[cfg(not(feature = "gpu-tests"))]
pub(crate) struct BlockQuantizedMoEFactory {
    pub(crate) runtime: Arc<CudaRuntime>,
}

impl KernelFactory for BlockQuantizedMoEFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(self.create_kernel(node, input_shapes)?))
    }
}

impl BlockQuantizedMoEFactory {
    fn create_kernel_impl(
        &self,
        node: &Node,
        input_shapes: &[Vec<usize>],
    ) -> Result<BlockQuantizedMoEKernel> {
        if node.inputs.len() != INPUT_COUNT {
            return Err(error(format!(
                "expected exactly {INPUT_COUNT} positional inputs, got {}",
                node.inputs.len()
            )));
        }
        parse_layout_version(node)?;
        let attributes = MoeAttributes::from_node(node)?;
        if let Some(formats) = parse_planar_projection_formats(node)? {
            warm_planar_moe(&self.runtime)?;
            return Ok(BlockQuantizedMoEKernel {
                runtime: self.runtime.clone(),
                attributes,
                interleaved_formats: None,
                interleaved_geometry: None,
                interleaved_launches: None,
                shared: None,
                uploaded_whole_bank_bytes: 0,
                committed_whole_bank_bytes: 0,
                bytes_per_expert: 0,
                planar: Some(PlanarKernelState {
                    formats,
                    constant_inputs: [false; INPUT_COUNT],
                    validation_scratch: self.runtime.alloc_raw(std::mem::size_of::<u32>())?,
                    validated_banks: Mutex::new([None; 3]),
                }),
            });
        }
        let formats = parse_projection_formats(node)?;
        let geometry = PreparedMoeGeometry::new(node, input_shapes, attributes, formats)?;
        let launches = PreparedMoeLaunches::new(&self.runtime, &geometry)?;
        Ok(BlockQuantizedMoEKernel {
            runtime: self.runtime.clone(),
            attributes,
            interleaved_formats: Some(formats),
            interleaved_geometry: Some(geometry),
            interleaved_launches: Some(launches),
            shared: None,
            uploaded_whole_bank_bytes: 0,
            committed_whole_bank_bytes: 0,
            bytes_per_expert: 0,
            planar: None,
        })
    }

    #[cfg(feature = "gpu-tests")]
    pub fn create_kernel(
        &self,
        node: &Node,
        input_shapes: &[Vec<usize>],
    ) -> Result<BlockQuantizedMoEKernel> {
        self.create_kernel_impl(node, input_shapes)
    }

    #[cfg(not(feature = "gpu-tests"))]
    pub(crate) fn create_kernel(
        &self,
        node: &Node,
        input_shapes: &[Vec<usize>],
    ) -> Result<BlockQuantizedMoEKernel> {
        self.create_kernel_impl(node, input_shapes)
    }
}

/// Placement declaration for the CUDA claim gate. The CUDA kernel implements the
/// BlockQuantizedMoE ABI over the CUDA-supported GGUF block formats with f32
/// activations. It declines any node the kernel cannot execute so those
/// nodes fall back to the CPU oracle rather than mis-executing.
pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    if node.inputs.len() != INPUT_COUNT
        || shapes.len() != INPUT_COUNT
        || input_dtypes.len() != INPUT_COUNT
    {
        return Some(Cow::Owned(format!(
            "BlockQuantizedMoE: expected exactly {INPUT_COUNT} positional inputs and matching metadata"
        )));
    }
    if let Err(error) = parse_layout_version(node) {
        return Some(Cow::Owned(error.to_string()));
    }
    match parse_planar_projection_formats(node) {
        Ok(Some(formats)) => {
            if let Err(reason) = validate_planar_claim(node, shapes, input_dtypes, formats) {
                return Some(Cow::Owned(format!("BlockQuantizedMoE: {reason}")));
            }
            if let Err(error) = MoeAttributes::from_node(node) {
                return Some(Cow::Owned(error.to_string()));
            }
            return None;
        }
        Ok(None) => {}
        Err(error) => return Some(Cow::Owned(error.to_string())),
    }
    let formats = match claim_projection_formats(node) {
        Ok(formats) => formats,
        Err(reason) => return Some(reason),
    };
    let attributes = match MoeAttributes::from_node(node) {
        Ok(attributes) => attributes,
        Err(reason) => return Some(Cow::Owned(reason.to_string())),
    };
    // The CUDA kernel is Phase-2 all-f32: activations, router logits/weights, and
    // the output are Float32; packed weights and the optional FC3 weights are
    // Uint8. Decline anything else so the CPU oracle keeps ownership.
    for index in 0..node.inputs.len() {
        if node.inputs[index].is_none() {
            continue;
        }
        let Some(dtype) = input_dtypes.get(index) else {
            continue;
        };
        let expected = if matches!(index, 2 | 4 | 6) {
            DataType::Uint8
        } else {
            DataType::Float32
        };
        if *dtype != expected {
            let name = INPUT_NAMES.get(index).copied().unwrap_or("input");
            return Some(Cow::Owned(format!(
                "BlockQuantizedMoE: CUDA requires input {index} ('{name}') to be {expected:?}, got {dtype:?} — the CUDA kernel is all-f32 (Uint8 packed weights)"
            )));
        }
    }
    let static_axis = |input: usize, axis: usize| {
        shapes
            .get(input)
            .and_then(|shape| shape.get(axis))
            .and_then(|dim| dim.as_static())
    };
    let hidden = shapes
        .first()
        .and_then(|shape| shape.last())
        .and_then(|dim| dim.as_static());
    let fc1_out = static_axis(2, 1);
    let inter = fc1_out.and_then(|width| {
        if attributes.swiglu_fusion == 0 {
            Some(width)
        } else {
            width.is_multiple_of(2).then_some(width / 2)
        }
    });
    for (index, label, format, in_features) in [
        (2, "fc1", Some(formats.fc1), hidden),
        (4, "fc2", Some(formats.fc2), inter),
        (6, "fc3", formats.fc3, hidden),
    ] {
        let Some(format) = format else {
            continue;
        };
        if let Some(width) = in_features {
            if !width.is_multiple_of(format.qk()) {
                return Some(Cow::Owned(format!(
                    "BlockQuantizedMoE: {label} input width {width} has a partial {format:?} block tail"
                )));
            }
            if let Some(blocks) = static_axis(index, 2)
                && blocks != width / format.qk()
            {
                return Some(Cow::Owned(format!(
                    "BlockQuantizedMoE: {label} block count {blocks} does not match {}",
                    width / format.qk()
                )));
            }
        }
        if let Some(bytes) = static_axis(index, 3)
            && bytes != format.block_bytes()
        {
            return Some(Cow::Owned(format!(
                "BlockQuantizedMoE: {label} block byte width {bytes} does not match {}",
                format.block_bytes()
            )));
        }
    }
    None
}

#[cfg(feature = "gpu-tests")]
pub struct BlockQuantizedMoEKernel {
    runtime: Arc<CudaRuntime>,
    attributes: MoeAttributes,
    interleaved_formats: Option<ProjectionFormats>,
    interleaved_geometry: Option<PreparedMoeGeometry>,
    interleaved_launches: Option<PreparedMoeLaunches>,
    shared: Option<Arc<BlockQuantizedMoeSharedState>>,
    uploaded_whole_bank_bytes: u64,
    committed_whole_bank_bytes: u64,
    bytes_per_expert: u64,
    planar: Option<PlanarKernelState>,
}

#[cfg(not(feature = "gpu-tests"))]
pub(crate) struct BlockQuantizedMoEKernel {
    runtime: Arc<CudaRuntime>,
    attributes: MoeAttributes,
    interleaved_formats: Option<ProjectionFormats>,
    interleaved_geometry: Option<PreparedMoeGeometry>,
    interleaved_launches: Option<PreparedMoeLaunches>,
    shared: Option<Arc<BlockQuantizedMoeSharedState>>,
    uploaded_whole_bank_bytes: u64,
    committed_whole_bank_bytes: u64,
    bytes_per_expert: u64,
    planar: Option<PlanarKernelState>,
}

struct BlockQuantizedMoeSharedState {
    banks: Arc<AdmittedBlockQuantizedMoeBankSet>,
    telemetry: ArcSwapOption<OwnedBlockQuantizedMoeTelemetry>,
}

struct OwnedBlockQuantizedMoeTelemetry {
    record: ArmedTelemetry,
    runtime: Arc<CudaRuntime>,
}

impl OwnedBlockQuantizedMoeTelemetry {
    fn arm(
        runtime: &Arc<CudaRuntime>,
        config: RouteTelemetryConfig,
    ) -> std::result::Result<Self, TelemetryUnsupported> {
        Ok(Self {
            record: ArmedTelemetry::arm(runtime, config)?,
            runtime: Arc::clone(runtime),
        })
    }
}

impl Drop for OwnedBlockQuantizedMoeTelemetry {
    fn drop(&mut self) {
        self.record.free(&self.runtime);
    }
}

fn checked_logical_traffic_bytes(
    bytes_per_expert: u64,
    route_count: u64,
    unique_experts: u64,
) -> Result<(u64, u64)> {
    let logical = bytes_per_expert
        .checked_mul(route_count)
        .ok_or_else(|| error("logical route-demand byte count overflow"))?;
    let unique = bytes_per_expert
        .checked_mul(unique_experts)
        .ok_or_else(|| error("unique selected-expert byte count overflow"))?;
    Ok((logical, unique))
}

impl BlockQuantizedMoEKernel {
    fn formats(&self) -> &ProjectionFormats {
        self.interleaved_formats
            .as_ref()
            .expect("interleaved BlockQuantizedMoE formats are absent")
    }

    fn geometry(&self) -> &PreparedMoeGeometry {
        self.interleaved_geometry
            .as_ref()
            .expect("interleaved BlockQuantizedMoE geometry is absent")
    }

    fn launches(&self) -> &PreparedMoeLaunches {
        self.interleaved_launches
            .as_ref()
            .expect("interleaved BlockQuantizedMoE launches are absent")
    }

    fn install_banks(&mut self, banks: Arc<AdmittedBlockQuantizedMoeBankSet>) -> Result<()> {
        self.validate_admitted_bank_set(&banks)?;
        let total = banks
            .fc1
            .total_bytes
            .checked_add(banks.fc2.total_bytes)
            .and_then(|bytes| {
                banks
                    .fc3
                    .as_ref()
                    .map_or(Some(bytes), |fc3| bytes.checked_add(fc3.total_bytes))
            })
            .ok_or_else(|| error("uploaded whole-bank byte count overflow"))?;
        let per_expert = banks
            .fc1
            .bytes_per_expert
            .checked_add(banks.fc2.bytes_per_expert)
            .and_then(|bytes| {
                banks
                    .fc3
                    .as_ref()
                    .map_or(Some(bytes), |fc3| bytes.checked_add(fc3.bytes_per_expert))
            })
            .ok_or_else(|| error("per-expert projection byte count overflow"))?;
        self.uploaded_whole_bank_bytes = as_u64("uploaded whole-bank bytes", total)?;
        self.committed_whole_bank_bytes = self.uploaded_whole_bank_bytes;
        self.bytes_per_expert = as_u64("per-expert projection bytes", per_expert)?;
        self.shared = Some(Arc::new(BlockQuantizedMoeSharedState {
            banks,
            telemetry: ArcSwapOption::empty(),
        }));
        Ok(())
    }

    fn shared(&self) -> Result<&Arc<BlockQuantizedMoeSharedState>> {
        self.shared
            .as_ref()
            .ok_or_else(|| error("BlockQuantizedMoE projection banks were not admitted"))
    }

    fn arm_route_telemetry_impl(
        &mut self,
        config: RouteTelemetryConfig,
    ) -> std::result::Result<(), TelemetryUnsupported> {
        if self.runtime.has_graph_executable().unwrap_or(false)
            || self
                .runtime
                .has_graph_executable_in(onnx_runtime_ep_api::DeviceGraphSlot::Verify)
                .unwrap_or(false)
        {
            return Err(TelemetryUnsupported::GraphInstalled);
        }
        if config.routes_per_row != self.attributes.k {
            return Err(TelemetryUnsupported::RouteWidthMismatch {
                config: config.routes_per_row,
                execution: self.attributes.k,
            });
        }
        let shared = self.shared.as_ref().ok_or_else(|| {
            TelemetryUnsupported::Alloc("projection banks are not admitted".into())
        })?;
        let armed = Arc::new(OwnedBlockQuantizedMoeTelemetry::arm(&self.runtime, config)?);
        shared.telemetry.store(Some(armed));
        Ok(())
    }

    fn disarm_route_telemetry_impl(&mut self) -> Result<()> {
        if self.runtime.has_graph_executable()?
            || self
                .runtime
                .has_graph_executable_in(onnx_runtime_ep_api::DeviceGraphSlot::Verify)?
        {
            return Err(error(
                "cannot disarm BlockQuantizedMoE traffic while a device graph is installed",
            ));
        }
        self.shared()?.telemetry.store(None);
        Ok(())
    }

    fn reset_route_telemetry_boundary_impl(&mut self) -> Result<()> {
        if let Some(armed) = self.shared()?.telemetry.load_full() {
            armed.record.reset_boundary(&self.runtime)?;
        }
        Ok(())
    }

    #[cfg(feature = "gpu-tests")]
    fn route_telemetry_snapshot_impl(&self) -> Result<Option<TelemetrySnapshot>> {
        self.shared()?
            .telemetry
            .load_full()
            .map(|armed| armed.record.snapshot(&self.runtime))
            .transpose()
    }

    #[cfg(feature = "gpu-tests")]
    fn route_telemetry_footprint_bytes_impl(&self) -> usize {
        self.shared
            .as_ref()
            .and_then(|shared| shared.telemetry.load_full())
            .map_or(0, |armed| armed.record.footprint_bytes())
    }

    #[cfg(feature = "gpu-tests")]
    fn route_telemetry_bitmap_addr_impl(&self) -> Option<u64> {
        self.shared
            .as_ref()
            .and_then(|shared| shared.telemetry.load_full())
            .map(|armed| armed.record.bitmap_addr())
    }

    pub(crate) fn production_traffic_snapshot(&self) -> Result<BlockQuantizedMoeTraffic> {
        let armed = self
            .shared()?
            .telemetry
            .load_full()
            .ok_or_else(|| error("BlockQuantizedMoE traffic is not armed"))?;
        let telemetry = armed.record.validated_snapshot(&self.runtime)?;
        let (logical_route_demand_bytes, unique_selected_expert_bytes) =
            checked_logical_traffic_bytes(
                self.bytes_per_expert,
                u64::from(telemetry.selected_route_count()),
                u64::from(telemetry.unique_expert_count()),
            )?;
        Ok(BlockQuantizedMoeTraffic {
            uploaded_whole_bank_bytes: self.uploaded_whole_bank_bytes,
            committed_whole_bank_bytes: self.committed_whole_bank_bytes,
            logical_route_demand_bytes,
            unique_selected_expert_bytes,
            physical_dram_bytes: None,
            page_ins: 0,
            byte_hit_rate: None,
        })
    }

    #[cfg(feature = "gpu-tests")]
    pub fn arm_route_telemetry(
        &mut self,
        config: RouteTelemetryConfig,
    ) -> std::result::Result<(), TelemetryUnsupported> {
        self.arm_route_telemetry_impl(config)
    }

    #[cfg(not(feature = "gpu-tests"))]
    pub(crate) fn arm_route_telemetry(
        &mut self,
        config: RouteTelemetryConfig,
    ) -> std::result::Result<(), TelemetryUnsupported> {
        self.arm_route_telemetry_impl(config)
    }

    #[cfg(feature = "gpu-tests")]
    pub fn disarm_route_telemetry(&mut self) -> Result<()> {
        self.disarm_route_telemetry_impl()
    }

    #[cfg(not(feature = "gpu-tests"))]
    pub(crate) fn disarm_route_telemetry(&mut self) -> Result<()> {
        self.disarm_route_telemetry_impl()
    }

    #[cfg(feature = "gpu-tests")]
    pub fn reset_route_telemetry_boundary(&mut self) -> Result<()> {
        self.reset_route_telemetry_boundary_impl()
    }

    #[cfg(not(feature = "gpu-tests"))]
    pub(crate) fn reset_route_telemetry_boundary(&mut self) -> Result<()> {
        self.reset_route_telemetry_boundary_impl()
    }

    #[cfg(feature = "gpu-tests")]
    pub fn route_telemetry_snapshot(&self) -> Result<Option<TelemetrySnapshot>> {
        self.route_telemetry_snapshot_impl()
    }

    #[cfg(feature = "gpu-tests")]
    pub fn route_telemetry_footprint_bytes(&self) -> usize {
        self.route_telemetry_footprint_bytes_impl()
    }

    #[cfg(feature = "gpu-tests")]
    pub fn route_telemetry_bitmap_addr(&self) -> Option<u64> {
        self.route_telemetry_bitmap_addr_impl()
    }

    #[cfg(feature = "gpu-tests")]
    pub fn inject_route_telemetry_fault_for_test(
        &self,
        fault: BlockQuantizedMoeTrafficFaultForTest,
    ) -> Result<()> {
        let armed = self
            .shared()?
            .telemetry
            .load_full()
            .ok_or_else(|| error("BlockQuantizedMoE traffic is not armed"))?;
        let snapshot = armed.record.snapshot(&self.runtime)?;
        let (index, value) = match fault {
            BlockQuantizedMoeTrafficFaultForTest::Poison => (H_POISON, 1),
            BlockQuantizedMoeTrafficFaultForTest::Overflow => (H_OVERFLOW, 1),
            BlockQuantizedMoeTrafficFaultForTest::StaleEpoch => {
                (H_EPOCH, snapshot.header[H_EPOCH].wrapping_add(1))
            }
            BlockQuantizedMoeTrafficFaultForTest::ForeignRequest => {
                (H_REQUEST, snapshot.header[H_REQUEST].wrapping_add(1))
            }
            BlockQuantizedMoeTrafficFaultForTest::WrongDevice => {
                (H_DEVICE, snapshot.header[H_DEVICE].wrapping_add(1))
            }
            BlockQuantizedMoeTrafficFaultForTest::NonTopKMultipleCount => {
                let count = snapshot.header[H_COUNT];
                (H_COUNT, count.saturating_add(1))
            }
        };
        armed.record.inject_header_word(&self.runtime, index, value)
    }

    fn validate_admitted_bank_set(&self, banks: &AdmittedBlockQuantizedMoeBankSet) -> Result<()> {
        let runtime_identity = Arc::as_ptr(&self.runtime) as usize;
        if banks.runtime_identity != runtime_identity
            || banks.device != onnx_runtime_ir::DeviceId::cuda(self.runtime.ordinal())
        {
            return Err(error(
                "sealed projection banks belong to a different CUDA runtime/device",
            ));
        }
        let expected = [
            (
                "fc1",
                Some(&banks.fc1),
                self.formats().fc1,
                self.geometry().fc1_out,
                self.geometry().hidden,
            ),
            (
                "fc2",
                Some(&banks.fc2),
                self.formats().fc2,
                self.geometry().hidden,
                self.geometry().inter,
            ),
            (
                "fc3",
                banks.fc3.as_ref(),
                self.formats().fc3.unwrap_or(self.formats().fc1),
                self.geometry().inter,
                self.geometry().hidden,
            ),
        ];
        for (label, bank, format, out_features, in_features) in expected {
            if label == "fc3" && bank.is_some() != self.geometry().has_fc3 {
                return Err(error(
                    "sealed FC3 presence does not match prepared geometry",
                ));
            }
            let Some(bank) = bank else {
                if label == "fc3" && !self.geometry().has_fc3 {
                    continue;
                }
                return Err(error(format!("sealed {label} projection is absent")));
            };
            if bank.format != format
                || bank.experts != self.geometry().experts
                || bank.out_features != out_features
                || bank.in_features != in_features
                || bank.allocation.runtime_identity() != runtime_identity
                || bank.allocation.provider_context() != banks.provider_context
                || bank.allocation.device() != banks.device
                || bank.allocation.len() != bank.total_bytes
                || bank.allocation.ptr().is_null()
            {
                return Err(error(format!(
                    "sealed {label} projection does not match prepared geometry/ownership"
                )));
            }
        }
        let fc1_identity = banks.fc1.allocation.allocation_identity();
        let fc2_identity = banks.fc2.allocation.allocation_identity();
        if fc1_identity == fc2_identity
            || banks.fc3.as_ref().is_some_and(|fc3| {
                let fc3_identity = fc3.allocation.allocation_identity();
                fc3_identity == fc1_identity || fc3_identity == fc2_identity
            })
        {
            return Err(error(
                "sealed projections must use distinct immutable allocations",
            ));
        }
        Ok(())
    }

    fn validate_sealed_projection<'a>(
        &self,
        label: &str,
        view: &TensorView<'_>,
        bias: Option<&TensorView<'_>>,
        bank: &'a AdmittedProjectionBank,
        banks: &AdmittedBlockQuantizedMoeBankSet,
    ) -> Result<SealedPackedExperts<'a>> {
        if view.dtype != DataType::Uint8
            || view.shape != bank.shape
            || view.strides != bank.strides
            || view.byte_offset != 0
            || view.device != banks.device
            || view.data.0 != bank.allocation.ptr().0
            || view.backing
                != (TensorBacking::Sealed {
                    provider_context: banks.provider_context,
                    allocation: bank.allocation.allocation_identity(),
                })
        {
            return Err(error(format!(
                "{label}_experts_weights must be the exact immutable admitted projection \
                 (dtype={:?}, shape={:?}, strides={:?}, offset={}, device={:?}, ptr={:?}, \
                 backing={:?}; expected shape={:?}, strides={:?}, device={:?}, ptr={:?}, \
                 backing={:?})",
                view.dtype,
                view.shape,
                view.strides,
                view.byte_offset,
                view.device,
                view.data.0,
                view.backing,
                bank.shape,
                bank.strides,
                banks.device,
                bank.allocation.ptr().0,
                TensorBacking::Sealed {
                    provider_context: banks.provider_context,
                    allocation: bank.allocation.allocation_identity(),
                }
            )));
        }
        if let Some(bias) = bias
            && (bias.dtype != DataType::Float32
                || bias.shape.len() != 2
                || bias.shape[0] != bank.experts
                || bias.shape[1] != bank.out_features
                || !bias.is_contiguous())
        {
            return Err(error(format!(
                "{label}_experts_bias must be contiguous Float32 [experts, out_features]"
            )));
        }
        Ok(SealedPackedExperts {
            bank,
            bias: bias.map(tensor_ptr).unwrap_or(0),
        })
    }
}

const WORKSPACE_ALIGNMENT: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QmoeWorkspaceLayout {
    offsets: [usize; 6],
    bytes: usize,
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or_else(|| error("BlockQuantizedMoE workspace alignment overflow"))
}

fn qmoe_workspace_layout(
    input_shape: &[usize],
    router_shape: &[usize],
    fc1_shape: &[usize],
    top_k: usize,
    swiglu_fusion: usize,
    has_fc3: bool,
) -> Result<QmoeWorkspaceLayout> {
    record_workspace_layout_build();
    if !matches!(input_shape.len(), 2 | 3) {
        return Err(error(format!(
            "input must be 2-D [rows, hidden] or 3-D [batch, sequence, hidden], got {input_shape:?}"
        )));
    }
    if router_shape.len() != 2 {
        return Err(error(format!(
            "router_logits must be rank 2, got {router_shape:?}"
        )));
    }
    if !matches!(fc1_shape.len(), 3 | 4) {
        return Err(error(format!(
            "fc1_experts_weights must be rank 3 (planar) or 4 (interleaved), got {fc1_shape:?}"
        )));
    }
    let hidden = *input_shape
        .last()
        .ok_or_else(|| error("input rank unexpectedly empty"))?;
    let rows = checked_product(
        &input_shape[..input_shape.len() - 1],
        "flattened input row count",
    )?;
    if router_shape[0] != rows {
        return Err(error(format!(
            "router_logits rows {} must equal flattened input rows {rows}",
            router_shape[0]
        )));
    }
    if top_k == 0 || top_k > router_shape[1] {
        return Err(error(format!(
            "requires 0 < k <= num_experts, got k={top_k} and num_experts={}",
            router_shape[1]
        )));
    }
    if rows == 0 || hidden == 0 {
        return Ok(QmoeWorkspaceLayout {
            offsets: [0; 6],
            bytes: 0,
        });
    }
    let fc1_out = fc1_shape[1];
    let inter = if swiglu_fusion == 0 {
        fc1_out
    } else {
        if !fc1_out.is_multiple_of(2) {
            return Err(error(format!(
                "fused SwiGLU fc1_out must be even, got {fc1_out}"
            )));
        }
        fc1_out / 2
    };
    if inter == 0 {
        return Err(error("inferred inter dimension must be non-zero"));
    }
    let routes = checked_product(&[rows, top_k], "route count")?;
    let sizes = [
        checked_bytes(routes, std::mem::size_of::<i32>(), "route indices")?,
        checked_bytes(routes, std::mem::size_of::<f32>(), "route weights")?,
        checked_bytes(
            checked_product(&[routes, fc1_out], "FC1 scratch element count")?,
            4,
            "FC1 scratch",
        )?,
        if has_fc3 {
            checked_bytes(
                checked_product(&[routes, inter], "FC3 scratch element count")?,
                4,
                "FC3 scratch",
            )?
        } else {
            0
        },
        checked_bytes(
            checked_product(&[routes, inter], "intermediate element count")?,
            4,
            "intermediate scratch",
        )?,
        checked_bytes(
            checked_product(&[routes, hidden], "route output element count")?,
            4,
            "route output scratch",
        )?,
    ];
    let mut offsets = [0; 6];
    let mut cursor = 0usize;
    for (index, size) in sizes.into_iter().enumerate() {
        if size == 0 {
            offsets[index] = 0;
            continue;
        }
        cursor = align_up(cursor, WORKSPACE_ALIGNMENT)?;
        offsets[index] = cursor;
        cursor = cursor
            .checked_add(size)
            .ok_or_else(|| error("BlockQuantizedMoE workspace byte count overflow"))?;
    }
    Ok(QmoeWorkspaceLayout {
        offsets,
        bytes: cursor,
    })
}

#[derive(Clone, Copy)]
struct SealedPackedExperts<'a> {
    bank: &'a AdmittedProjectionBank,
    bias: CUdeviceptr,
}

fn block_format_name(format: BlockFormat) -> &'static str {
    match format {
        BlockFormat::Mxfp4 => "mxfp4",
        BlockFormat::Iq4Nl => "iq4_nl",
        BlockFormat::Iq4Xs => "iq4_xs",
        BlockFormat::Iq2Xxs => "iq2_xxs",
        BlockFormat::Iq3Xxs => "iq3_xxs",
        BlockFormat::Iq2Xs => "iq2_xs",
        BlockFormat::Iq2S => "iq2_s",
        BlockFormat::Iq3S => "iq3_s",
        BlockFormat::Iq1S => "iq1_s",
        BlockFormat::Iq1M => "iq1_m",
        BlockFormat::Q2K => "q2_k",
        BlockFormat::Q3K => "q3_k",
        BlockFormat::Q5K => "q5_k",
        BlockFormat::Q6K => "q6_k",
        BlockFormat::Q8_0 => "q8_0",
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_planar_projection(
    inputs: &[TensorView],
    weight_index: usize,
    scale_index: usize,
    bias_index: usize,
    geometry: PlanarBlockGeometry,
    experts: usize,
    out_features: usize,
    in_features: usize,
) -> Result<BorrowedPlanarMoeProjection> {
    let weight = &inputs[weight_index];
    let scale = optional_input(inputs, scale_index).ok_or_else(|| {
        error(format!(
            "{} is required for {}",
            INPUT_NAMES[scale_index],
            geometry.format.capability_str()
        ))
    })?;
    require_dtype(
        INPUT_NAMES[weight_index],
        weight.dtype,
        geometry.format.weight_dtype(),
    )?;
    require_dtype(
        INPUT_NAMES[scale_index],
        scale.dtype,
        geometry.format.scale_dtype(),
    )?;
    let projection = PlanarMoeProjection {
        format: geometry.format.kernel_id(),
        in_features,
        out_features,
        bs0: geometry.block_out,
        bs1: geometry.block_in,
    };
    let (packed_bytes, scale_bytes) = projection.per_expert_bytes()?;
    require_shape(
        INPUT_NAMES[weight_index],
        weight.shape,
        &[
            experts,
            out_features,
            in_features / geometry.format.pack_factor(),
        ],
    )?;
    require_shape(
        INPUT_NAMES[scale_index],
        scale.shape,
        &[
            experts,
            out_features.div_ceil(geometry.block_out),
            in_features.div_ceil(geometry.block_in),
        ],
    )?;
    if weight.byte_size()
        != experts
            .checked_mul(packed_bytes)
            .ok_or_else(|| error("planar packed bank byte count overflow"))?
        || scale.byte_size()
            != experts
                .checked_mul(scale_bytes)
                .ok_or_else(|| error("planar scale bank byte count overflow"))?
    {
        return Err(error("planar expert bank byte extent mismatch"));
    }
    if !weight.is_contiguous() || !scale.is_contiguous() {
        return Err(error("planar expert weight and scale must be contiguous"));
    }
    let bias = optional_input(inputs, bias_index);
    if let Some(bias) = bias {
        require_dtype(INPUT_NAMES[bias_index], bias.dtype, DataType::Float32)?;
        require_shape(
            INPUT_NAMES[bias_index],
            bias.shape,
            &[experts, out_features],
        )?;
        if !bias.is_contiguous() {
            return Err(error(format!(
                "{} must be contiguous",
                INPUT_NAMES[bias_index]
            )));
        }
    }
    Ok(BorrowedPlanarMoeProjection {
        projection,
        packed: tensor_ptr(weight),
        scale: tensor_ptr(scale),
        bias: bias.map(tensor_ptr).unwrap_or(0),
    })
}

impl Kernel for BlockQuantizedMoEKernel {
    fn prepare_constant_inputs(
        &mut self,
        constants: &[Option<KernelConstantInput<'_>>],
        provider: &dyn ExecutionProvider,
    ) -> Result<()> {
        if let Some(planar) = self.planar.as_mut() {
            if provider.device_id() != onnx_runtime_ir::DeviceId::cuda(self.runtime.ordinal())
                || provider.runtime_identity() != Some(Arc::as_ptr(&self.runtime) as usize)
            {
                return Err(error(
                    "BlockQuantizedMoE constants were offered by the wrong provider runtime/device",
                ));
            }
            for (index, value) in planar.constant_inputs.iter_mut().enumerate() {
                *value = constants.get(index).is_some_and(Option::is_some);
            }
            for index in [
                BQMOE_FC1_WEIGHT,
                BQMOE_FC1_SCALE,
                BQMOE_FC2_WEIGHT,
                BQMOE_FC2_SCALE,
            ] {
                if !planar.constant_inputs[index] {
                    return Err(error(format!(
                        "{} must be an immutable graph initializer",
                        INPUT_NAMES[index]
                    )));
                }
            }
            if planar.formats.fc3.is_some()
                && (!planar.constant_inputs[BQMOE_FC3_WEIGHT]
                    || !planar.constant_inputs[BQMOE_FC3_SCALE])
            {
                return Err(error(
                    "fc3 planar weight and aux scale must be immutable graph initializers",
                ));
            }
            return Ok(());
        }
        if self.shared.is_some() {
            return Err(error("immutable projection banks were already admitted"));
        }
        if provider.device_id() != onnx_runtime_ir::DeviceId::cuda(self.runtime.ordinal())
            || provider.runtime_identity() != Some(Arc::as_ptr(&self.runtime) as usize)
        {
            return Err(error(
                "BlockQuantizedMoE constants were offered by the wrong provider runtime/device",
            ));
        }
        let projection = |index: usize,
                          label: &'static str,
                          format: BlockFormat,
                          out_features: usize,
                          in_features: usize|
         -> Result<BlockQuantizedMoeBank<'_>> {
            let constant = constants
                .get(index)
                .and_then(Option::as_ref)
                .ok_or_else(|| error(format!("{label} must be an immutable graph initializer")))?;
            if constant.dtype != DataType::Uint8
                || constant.shape != self.geometry().input_shapes[index]
            {
                return Err(error(format!(
                    "{label} initializer metadata does not match the prepared projection"
                )));
            }
            Ok(BlockQuantizedMoeBank {
                format: block_format_name(format),
                packed: constant.bytes,
                experts: self.geometry().experts,
                out_features,
                in_features,
            })
        };
        let fc1 = projection(
            2,
            "fc1",
            self.formats().fc1,
            self.geometry().fc1_out,
            self.geometry().hidden,
        )?;
        let fc2 = projection(
            4,
            "fc2",
            self.formats().fc2,
            self.geometry().hidden,
            self.geometry().inter,
        )?;
        let fc3 = if self.geometry().has_fc3 {
            Some(projection(
                6,
                "fc3",
                self.formats()
                    .fc3
                    .ok_or_else(|| error("prepared FC3 format is absent"))?,
                self.geometry().inter,
                self.geometry().hidden,
            )?)
        } else {
            None
        };
        let banks = admit_block_quantized_moe_bank_set(provider, fc1, fc2, fc3)?;
        self.install_banks(banks)
    }

    fn shareable_constant_state(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.shared
            .as_ref()
            .map(|shared| Arc::clone(shared) as Arc<dyn std::any::Any + Send + Sync>)
    }

    fn adopt_shareable_constant_state(
        &mut self,
        state: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<bool> {
        if self.planar.is_some() {
            return Ok(false);
        }
        let Ok(shared) = Arc::downcast::<BlockQuantizedMoeSharedState>(state) else {
            return Ok(false);
        };
        self.validate_admitted_bank_set(&shared.banks)?;
        let total = shared
            .banks
            .fc1
            .total_bytes
            .checked_add(shared.banks.fc2.total_bytes)
            .and_then(|bytes| {
                shared
                    .banks
                    .fc3
                    .as_ref()
                    .map_or(Some(bytes), |fc3| bytes.checked_add(fc3.total_bytes))
            })
            .ok_or_else(|| error("uploaded whole-bank byte count overflow"))?;
        let per_expert = shared
            .banks
            .fc1
            .bytes_per_expert
            .checked_add(shared.banks.fc2.bytes_per_expert)
            .and_then(|bytes| {
                shared
                    .banks
                    .fc3
                    .as_ref()
                    .map_or(Some(bytes), |fc3| bytes.checked_add(fc3.bytes_per_expert))
            })
            .ok_or_else(|| error("per-expert projection byte count overflow"))?;
        self.uploaded_whole_bank_bytes = total as u64;
        self.committed_whole_bank_bytes = total as u64;
        self.bytes_per_expert = per_expert as u64;
        self.shared = Some(shared);
        Ok(true)
    }

    fn constant_input_override(&self, input_idx: usize) -> Option<TensorView<'_>> {
        let banks = &self.shared.as_ref()?.banks;
        let bank = match input_idx {
            2 => &banks.fc1,
            4 => &banks.fc2,
            6 => banks.fc3.as_ref()?,
            _ => return None,
        };
        Some(
            TensorView::new(
                bank.allocation.ptr(),
                DataType::Uint8,
                &bank.shape,
                &bank.strides,
                banks.device,
            )
            .with_backing(TensorBacking::Sealed {
                provider_context: banks.provider_context,
                allocation: bank.allocation.allocation_identity(),
            }),
        )
    }

    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        let Some(shared) = self.shared.as_ref() else {
            return Vec::new();
        };
        let mut resources = Vec::with_capacity(2);
        resources.push(DeviceGraphResource::new(
            Arc::as_ptr(&shared.banks) as usize,
            Arc::clone(&shared.banks),
        ));
        if let Some(telemetry) = shared.telemetry.load_full() {
            resources.push(DeviceGraphResource::new(
                Arc::as_ptr(&telemetry) as usize,
                telemetry,
            ));
        }
        resources
    }

    fn arm_block_quantized_moe_traffic(&mut self, request_id: u32) -> Result<bool> {
        if self.planar.is_some() {
            return Ok(false);
        }
        self.arm_route_telemetry(RouteTelemetryConfig {
            request_id,
            device_id: self.runtime.ordinal(),
            num_experts: self.geometry().experts,
            routes_per_row: self.attributes.k,
        })
        .map_err(|error| EpError::KernelFailed(error.to_string()))?;
        Ok(true)
    }

    fn reset_block_quantized_moe_traffic(&mut self) -> Result<()> {
        if self.planar.is_some() {
            return Ok(());
        }
        self.reset_route_telemetry_boundary()
    }

    fn snapshot_block_quantized_moe_traffic(&self) -> Result<Option<BlockQuantizedMoeTraffic>> {
        if self.planar.is_some() {
            return Ok(None);
        }
        if self
            .shared
            .as_ref()
            .and_then(|shared| shared.telemetry.load_full())
            .is_none()
        {
            return Ok(None);
        }
        self.production_traffic_snapshot().map(Some)
    }

    fn disarm_block_quantized_moe_traffic(&mut self) -> Result<()> {
        if self.planar.is_some() {
            return Ok(());
        }
        self.disarm_route_telemetry()
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let _ = (inputs, outputs);
        Err(error(
            "BlockQuantizedMoE requires workspace prepared before execution",
        ))
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        if self.planar.is_some() {
            if inputs.len() != INPUT_COUNT {
                return Err(error(format!(
                    "expected exactly {INPUT_COUNT} input metadata entries, got {}",
                    inputs.len()
                )));
            }
            let layout = qmoe_workspace_layout(
                inputs[0].shape,
                inputs[1].shape,
                inputs[BQMOE_FC1_WEIGHT].shape,
                self.attributes.k,
                self.attributes.swiglu_fusion,
                inputs[BQMOE_FC3_WEIGHT].present,
            )?;
            return Ok(WorkspaceRequirement {
                bytes: u64::try_from(layout.bytes)
                    .map_err(|_| error("BlockQuantizedMoE workspace does not fit u64"))?,
                alignment: WORKSPACE_ALIGNMENT,
                lifetime: WorkspaceLifetime::SessionPersistent,
                role: MemoryRole::Workspace { step_scoped: false },
            });
        }
        if inputs.len() != self.geometry().input_shapes.len() {
            return Err(error(format!(
                "expected {} input metadata entries, got {}",
                self.geometry().input_shapes.len(),
                inputs.len()
            )));
        }
        for (index, (input, expected)) in inputs
            .iter()
            .zip(self.geometry().input_shapes.iter())
            .enumerate()
        {
            let expected_present = !expected.is_empty();
            if input.present != expected_present
                || (expected_present && input.shape != expected.as_slice())
            {
                return Err(error(format!(
                    "input {index} metadata changed after BlockQuantizedMoE preparation"
                )));
            }
        }
        Ok(WorkspaceRequirement {
            bytes: u64::try_from(self.geometry().layout.bytes)
                .map_err(|_| error("BlockQuantizedMoE workspace does not fit u64"))?,
            alignment: WORKSPACE_ALIGNMENT,
            lifetime: WorkspaceLifetime::SessionPersistent,
            role: MemoryRole::Workspace { step_scoped: false },
        })
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        if let Some(planar) = self.planar.as_ref() {
            return self.execute_planar_with_workspace(inputs, outputs, workspace, planar.formats);
        }
        if inputs.len() != self.geometry().input_shapes.len() || outputs.len() != 1 {
            return Err(error(format!(
                "expected {} inputs and exactly 1 output, got {} inputs and {} outputs",
                self.geometry().input_shapes.len(),
                inputs.len(),
                outputs.len()
            )));
        }
        let shared = self.shared()?;
        let banks = &shared.banks;
        self.validate_admitted_bank_set(banks)?;
        for (index, (input, expected)) in inputs
            .iter()
            .zip(self.geometry().input_shapes.iter())
            .enumerate()
        {
            let expected_present = !expected.is_empty();
            if input.is_absent() == expected_present
                || (expected_present && input.shape != expected.as_slice())
            {
                return Err(error(format!(
                    "input {index} changed after BlockQuantizedMoE preparation"
                )));
            }
            if expected_present && input.device != banks.device {
                return Err(error(format!("input {index} is on the wrong CUDA device")));
            }
        }
        if inputs[0].dtype != DataType::Float32
            || inputs[1].dtype != DataType::Float32
            || !inputs[0].is_contiguous()
            || !inputs[1].is_contiguous()
        {
            return Err(error(
                "input and router_logits must be contiguous Float32 tensors",
            ));
        }
        if outputs[0].dtype != DataType::Float32
            || outputs[0].shape != self.geometry().input_shapes[0]
            || !outputs[0].is_contiguous()
            || outputs[0].device != banks.device
        {
            return Err(error(
                "output must be the prepared contiguous Float32 CUDA tensor",
            ));
        }

        let fc1 = self.validate_sealed_projection(
            "fc1",
            &inputs[2],
            optional_input(inputs, 3),
            &banks.fc1,
            banks,
        )?;
        let fc2 = self.validate_sealed_projection(
            "fc2",
            &inputs[4],
            optional_input(inputs, 5),
            &banks.fc2,
            banks,
        )?;
        let fc3 = match (&banks.fc3, optional_input(inputs, 6)) {
            (Some(bank), Some(view)) => Some(self.validate_sealed_projection(
                "fc3",
                view,
                optional_input(inputs, 7),
                bank,
                banks,
            )?),
            (None, None) => None,
            _ => {
                return Err(error(
                    "FC3 execution input does not match the admitted projection set",
                ));
            }
        };
        let router_weights = optional_input(inputs, 8);
        if let Some(router_weights) = router_weights
            && (router_weights.dtype != DataType::Float32
                || !router_weights.is_contiguous()
                || router_weights.device != banks.device)
        {
            return Err(error(
                "router_weights must be the prepared contiguous Float32 CUDA tensor",
            ));
        }
        if self.geometry().rows == 0 || self.geometry().hidden == 0 {
            return Ok(());
        }

        let workspace = workspace.ok_or_else(|| {
            error("BlockQuantizedMoE execute reached compute without prepared workspace")
        })?;
        if workspace.bytes() < self.geometry().layout.bytes {
            return Err(error(format!(
                "BlockQuantizedMoE prepared-workspace invariant violated: execution requires {} bytes but only {} were prepared",
                self.geometry().layout.bytes,
                workspace.bytes()
            )));
        }
        let base = cuptr(workspace.ptr().0.cast_const());
        let ptr = |index: usize| base + self.geometry().layout.offsets[index] as u64;
        let route_indices = ptr(0);
        let route_weights_ptr = ptr(1);
        let fc1_output = ptr(2);
        let fc3_output = fc3.map(|_| ptr(3));
        let activated = ptr(4);
        let route_output = ptr(5);

        let telemetry = shared.telemetry.load();
        let (telemetry_bitmap, telemetry_header) = if telemetry
            .as_ref()
            .is_some_and(|owner| owner.record.matches_experts(self.geometry().experts))
        {
            let owner = telemetry.as_ref().expect("checked above");
            (owner.record.bitmap_ptr(), owner.record.header_ptr())
        } else {
            (0, 0)
        };

        self.runtime.require_registered_address_capture(
            Arc::as_ptr(&shared.banks) as usize,
            "BlockQuantizedMoE projection banks",
        )?;
        if let Some(owner) = telemetry.as_ref() {
            self.runtime.require_registered_address_capture(
                Arc::as_ptr(owner) as usize,
                "BlockQuantizedMoE traffic record",
            )?;
        }
        self.launch_route(
            &inputs[1],
            router_weights,
            route_indices,
            route_weights_ptr,
            telemetry_bitmap,
            telemetry_header,
        )?;
        self.launch_linear(
            tensor_ptr(&inputs[0]),
            route_indices,
            fc1,
            fc1_output,
            false,
            self.launches().fc1_config,
        )?;
        if let (Some(fc3), Some(fc3_output)) = (fc3, fc3_output) {
            self.launch_linear(
                tensor_ptr(&inputs[0]),
                route_indices,
                fc3,
                fc3_output,
                false,
                self.launches()
                    .fc3_config
                    .ok_or_else(|| error("prepared FC3 launch is absent"))?,
            )?;
        }
        self.launch_activation(fc1_output, fc3_output, activated)?;
        self.launch_linear(
            activated,
            route_indices,
            fc2,
            route_output,
            true,
            self.launches().fc2_config,
        )?;
        self.launch_combine(route_output, route_weights_ptr, &mut outputs[0])?;
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::Supported
    }
}

impl Drop for BlockQuantizedMoEKernel {
    fn drop(&mut self) {
        if let Some(planar) = self.planar.take() {
            // SAFETY: this kernel uniquely owns the planar validation word.
            let _ = unsafe { self.runtime.free_raw(planar.validation_scratch) };
        }
    }
}

impl BlockQuantizedMoEKernel {
    fn execute_planar_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
        formats: PlanarProjectionFormats,
    ) -> Result<()> {
        let planar = self
            .planar
            .as_ref()
            .ok_or_else(|| error("planar BlockQuantizedMoE state is absent"))?;
        for &index in &[0usize, 1, BQMOE_FC1_WEIGHT, BQMOE_FC2_WEIGHT] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{}') is absent",
                    INPUT_NAMES[index]
                )));
            }
        }
        for &(weight, scale) in &[
            (BQMOE_FC1_WEIGHT, BQMOE_FC1_SCALE),
            (BQMOE_FC2_WEIGHT, BQMOE_FC2_SCALE),
        ] {
            if !planar.constant_inputs[weight] || !planar.constant_inputs[scale] {
                return Err(error(format!(
                    "{} and {} must be immutable session constants",
                    INPUT_NAMES[weight], INPUT_NAMES[scale]
                )));
            }
        }
        if formats.fc3.is_some()
            && (!planar.constant_inputs[BQMOE_FC3_WEIGHT]
                || !planar.constant_inputs[BQMOE_FC3_SCALE])
        {
            return Err(error(
                "fc3 planar weight and aux scale must be immutable session constants",
            ));
        }
        require_dtype("input", inputs[0].dtype, DataType::Float32)?;
        require_dtype("router_logits", inputs[1].dtype, DataType::Float32)?;
        require_dtype("output", outputs[0].dtype, DataType::Float32)?;
        let input_shape = inputs[0].shape;
        if !matches!(input_shape.len(), 2 | 3) {
            return Err(error(format!(
                "input must be [rows,H] or [B,S,H], got {input_shape:?}"
            )));
        }
        require_shape("output", outputs[0].shape, input_shape)?;
        let hidden = *input_shape
            .last()
            .ok_or_else(|| error("input rank unexpectedly empty"))?;
        let rows = checked_product(
            &input_shape[..input_shape.len() - 1],
            "flattened input row count",
        )?;
        require_rank("router_logits", inputs[1].shape, 2)?;
        if inputs[1].shape[0] != rows {
            return Err(error("router_logits row count must match input rows"));
        }
        let experts = inputs[1].shape[1];
        if self.attributes.k > experts {
            return Err(error(format!(
                "k={} exceeds num_experts={experts}",
                self.attributes.k
            )));
        }
        require_rank("fc1_experts_weights", inputs[BQMOE_FC1_WEIGHT].shape, 3)?;
        require_rank("fc2_experts_weights", inputs[BQMOE_FC2_WEIGHT].shape, 3)?;
        let fc1_out = inputs[BQMOE_FC1_WEIGHT].shape[1];
        let inter = if self.attributes.swiglu_fusion == 0 {
            fc1_out
        } else {
            if !fc1_out.is_multiple_of(2) {
                return Err(error("fused SwiGLU fc1 output width must be even"));
            }
            fc1_out / 2
        };
        if inter == 0 || self.attributes.fc1_size(inter)? != fc1_out {
            return Err(error("invalid inferred planar MoE intermediate width"));
        }

        let fc1 = validate_planar_projection(
            inputs,
            BQMOE_FC1_WEIGHT,
            BQMOE_FC1_SCALE,
            3,
            formats.fc1,
            experts,
            fc1_out,
            hidden,
        )?;
        let fc2 = validate_planar_projection(
            inputs,
            BQMOE_FC2_WEIGHT,
            BQMOE_FC2_SCALE,
            5,
            formats.fc2,
            experts,
            hidden,
            inter,
        )?;
        let has_fc3 = optional_input(inputs, BQMOE_FC3_WEIGHT).is_some();
        if has_fc3 != formats.fc3.is_some() {
            return Err(error(
                "fc3_format must be present exactly when fc3_experts_weights is wired",
            ));
        }
        let fc3 = if self.attributes.uses_separate_gate(has_fc3) {
            Some(validate_planar_projection(
                inputs,
                BQMOE_FC3_WEIGHT,
                BQMOE_FC3_SCALE,
                7,
                formats
                    .fc3
                    .ok_or_else(|| error("fc3 planar format is missing"))?,
                experts,
                inter,
                hidden,
            )?)
        } else {
            if has_fc3
                || optional_input(inputs, 7).is_some()
                || optional_input(inputs, BQMOE_FC3_SCALE).is_some()
            {
                return Err(error(
                    "fc3 inputs are valid only for an unfused gated activation",
                ));
            }
            None
        };
        let router_weights = optional_input(inputs, 8);
        if let Some(weights) = router_weights {
            require_dtype("router_weights", weights.dtype, DataType::Float32)?;
            require_shape("router_weights", weights.shape, &[rows, experts])?;
            if !weights.is_contiguous() {
                return Err(error("router_weights must be contiguous"));
            }
        }
        for (name, tensor) in [("input", &inputs[0]), ("router_logits", &inputs[1])] {
            checked_tensor_layout(name, tensor.shape, tensor.dtype)?;
            if !tensor.is_contiguous() {
                return Err(error(format!("{name} must be contiguous")));
            }
        }
        if !outputs[0].is_contiguous() {
            return Err(error("output must be contiguous"));
        }
        if rows == 0 || hidden == 0 {
            return Ok(());
        }

        let dims = PlanarMoeDims {
            rows,
            hidden,
            inter,
            experts,
            top_k: self.attributes.k,
            activation: self.attributes.activation.kernel_id(),
            swiglu_fusion: as_i32("swiglu_fusion", self.attributes.swiglu_fusion)?,
            activation_alpha: self.attributes.activation_alpha,
            activation_beta: self.attributes.activation_beta,
            swiglu_limit: self.attributes.swiglu_limit,
            normalize_routing_weights: self.attributes.normalize_routing_weights,
            fc1: fc1.projection,
            fc2: fc2.projection,
            fc3: fc3.map(|bank| bank.projection),
        };
        self.admit_planar_banks(&dims, [Some(fc1), Some(fc2), fc3])?;

        let layout = qmoe_workspace_layout(
            input_shape,
            inputs[1].shape,
            inputs[BQMOE_FC1_WEIGHT].shape,
            self.attributes.k,
            self.attributes.swiglu_fusion,
            fc3.is_some(),
        )?;
        let workspace = workspace.ok_or_else(|| error("prepared workspace is missing"))?;
        if workspace.bytes() < layout.bytes {
            return Err(error(format!(
                "prepared workspace has {} bytes, requires {}",
                workspace.bytes(),
                layout.bytes
            )));
        }
        let base = cuptr(workspace.ptr().0.cast_const());
        let workspace_ptr = |index: usize| base + layout.offsets[index] as u64;
        let ptrs = BorrowedPlanarMoePtrs {
            input: tensor_ptr(&inputs[0]),
            router_logits: tensor_ptr(&inputs[1]),
            router_weights: router_weights.map(tensor_ptr).unwrap_or(0),
            fc1,
            fc2,
            fc3,
            route_indices: workspace_ptr(0),
            route_weights: workspace_ptr(1),
            fc1_output: workspace_ptr(2),
            fc3_output: fc3.map_or(0, |_| workspace_ptr(3)),
            activated: workspace_ptr(4),
            route_output: workspace_ptr(5),
            output: cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void),
        };
        launch_planar_moe_borrowed(&self.runtime, &dims, ptrs)
    }

    fn admit_planar_banks(
        &self,
        dims: &PlanarMoeDims,
        banks: [Option<BorrowedPlanarMoeProjection>; 3],
    ) -> Result<()> {
        let planar = self
            .planar
            .as_ref()
            .ok_or_else(|| error("planar BlockQuantizedMoE state is absent"))?;
        let mut validated = planar
            .validated_banks
            .lock()
            .map_err(|_| error("planar bank validation state is poisoned"))?;
        let scratch = planar.validation_scratch;
        for (index, (bank, projection)) in banks
            .into_iter()
            .zip([Some(dims.fc1), Some(dims.fc2), dims.fc3])
            .enumerate()
        {
            let (Some(bank), Some(projection)) = (bank, projection) else {
                validated[index] = None;
                continue;
            };
            let identity = PlanarBankIdentity {
                packed: bank.packed,
                scale: bank.scale,
            };
            if validated[index] == Some(identity) {
                continue;
            }
            let linear_dims = PlanarLinearDims {
                format: projection.format,
                m_rows: 1,
                in_features: projection.in_features,
                out_features: projection.out_features,
                bs0: projection.bs0,
                bs1: projection.bs1,
            };
            validate_planar_bank_device(
                &self.runtime,
                &linear_dims,
                dims.experts,
                bank.packed,
                bank.scale,
                scratch,
            )?;
            validated[index] = Some(identity);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_route(
        &self,
        router_logits: &TensorView,
        router_weights: Option<&TensorView>,
        route_indices: CUdeviceptr,
        route_weights: CUdeviceptr,
        route_telemetry_bitmap: CUdeviceptr,
        route_telemetry_header: CUdeviceptr,
    ) -> Result<()> {
        let router_logits_ptr = tensor_ptr(router_logits);
        let router_weights_ptr = router_weights.map(tensor_ptr).unwrap_or(0);
        let rows_u64 = self.geometry().rows as u64;
        let experts_i32 = self.geometry().experts as i32;
        let top_k = as_i32("top-k", self.attributes.k)?;
        let normalize = i32::from(self.attributes.normalize_routing_weights);
        let mut params = [
            kernel_param(&router_logits_ptr),
            kernel_param(&router_weights_ptr),
            kernel_param(&route_indices),
            kernel_param(&route_weights),
            kernel_param(&rows_u64),
            kernel_param(&experts_i32),
            kernel_param(&top_k),
            kernel_param(&normalize),
            kernel_param(&route_telemetry_bitmap),
            kernel_param(&route_telemetry_header),
        ];
        // SAFETY: scratch buffers cover rows*top_k entries and the scalar ABI
        // matches `bqmoe_route`. Telemetry pointers are null when disarmed, which
        // the kernel treats as inert.
        unsafe {
            self.launches().route.launch(
                self.runtime.stream(),
                self.launches().route_config,
                &mut params,
            )
        }
        .map_err(|err| driver_err("launch BlockQuantizedMoE routing", err))
    }

    fn launch_linear(
        &self,
        input_ptr: CUdeviceptr,
        route_indices: CUdeviceptr,
        weights: SealedPackedExperts<'_>,
        output: CUdeviceptr,
        input_rows_are_routes: bool,
        config: LaunchConfig,
    ) -> Result<()> {
        let packed = cuptr(weights.bank.allocation.ptr().0);
        let bias = weights.bias;
        let routes_u64 = self.geometry().routes as u64;
        let input_rows_are_routes = i32::from(input_rows_are_routes);
        let top_k = self.attributes.k as i32;
        let out_features = weights.bank.out_features as i32;
        let in_features = weights.bank.in_features as i32;
        let blocks = weights.bank.blocks as i32;
        let block_bytes = weights.bank.format.block_bytes() as i32;
        let qk = weights.bank.format.qk() as i32;
        let format = weights.bank.format.kernel_id();
        let mut params = [
            kernel_param(&input_ptr),
            kernel_param(&route_indices),
            kernel_param(&packed),
            kernel_param(&bias),
            kernel_param(&output),
            kernel_param(&routes_u64),
            kernel_param(&input_rows_are_routes),
            kernel_param(&top_k),
            kernel_param(&out_features),
            kernel_param(&in_features),
            kernel_param(&blocks),
            kernel_param(&block_bytes),
            kernel_param(&qk),
            kernel_param(&format),
        ];
        // SAFETY: packed weights cover experts*out_features*blocks*block_bytes,
        // scratch buffers cover routes*out_features, and the scalar ABI matches
        // `bqmoe_linear_f32`.
        unsafe {
            self.launches()
                .linear
                .launch(self.runtime.stream(), config, &mut params)
        }
        .map_err(|err| driver_err("launch BlockQuantizedMoE expert GEMV", err))
    }

    fn launch_activation(
        &self,
        fc1: CUdeviceptr,
        fc3: Option<CUdeviceptr>,
        activated: CUdeviceptr,
    ) -> Result<()> {
        let fc3 = fc3.unwrap_or(0);
        let routes_u64 = self.geometry().routes as u64;
        let inter_i32 = self.geometry().inter as i32;
        let activation = self.attributes.activation.kernel_id();
        let swiglu_fusion = self.attributes.swiglu_fusion as i32;
        let alpha = self.attributes.activation_alpha;
        let beta = self.attributes.activation_beta;
        let limit = self.attributes.swiglu_limit;
        let mut params = [
            kernel_param(&fc1),
            kernel_param(&fc3),
            kernel_param(&activated),
            kernel_param(&routes_u64),
            kernel_param(&inter_i32),
            kernel_param(&activation),
            kernel_param(&swiglu_fusion),
            kernel_param(&alpha),
            kernel_param(&beta),
            kernel_param(&limit),
        ];
        // SAFETY: scratch buffers cover every routed intermediate element and the
        // ABI matches `bqmoe_activate`.
        unsafe {
            self.launches().activate.launch(
                self.runtime.stream(),
                self.launches().activate_config,
                &mut params,
            )
        }
        .map_err(|err| driver_err("launch BlockQuantizedMoE activation", err))
    }

    fn launch_combine(
        &self,
        route_output: CUdeviceptr,
        route_weights: CUdeviceptr,
        output: &mut TensorMut,
    ) -> Result<()> {
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let rows_u64 = self.geometry().rows as u64;
        let hidden_i32 = self.geometry().hidden as i32;
        let top_k = self.attributes.k as i32;
        let mut params = [
            kernel_param(&route_output),
            kernel_param(&route_weights),
            kernel_param(&output_ptr),
            kernel_param(&rows_u64),
            kernel_param(&hidden_i32),
            kernel_param(&top_k),
        ];
        // SAFETY: routed output and weights cover rows*top_k, output covers
        // rows*hidden, and the ABI matches `bqmoe_combine_f32`.
        unsafe {
            self.launches().combine.launch(
                self.runtime.stream(),
                self.launches().combine_config,
                &mut params,
            )
        }
        .map_err(|err| driver_err("launch BlockQuantizedMoE weighted combine", err))
    }
}

fn kernel_param<T>(value: &T) -> *mut c_void {
    std::ptr::from_ref(value).cast_mut().cast()
}

fn tensor_ptr(tensor: &TensorView) -> CUdeviceptr {
    cuptr(tensor.data_ptr::<u8>() as *const c_void)
}

fn optional_input<'a, 'b>(
    inputs: &'a [TensorView<'b>],
    index: usize,
) -> Option<&'a TensorView<'b>> {
    inputs.get(index).filter(|input| !input.is_absent())
}

fn int_attr(node: &Node, name: &str, default: i64) -> Result<i64> {
    match node.attr(name) {
        Some(value) => value
            .as_int()
            .ok_or_else(|| error(format!("attribute {name} must be an integer"))),
        None => Ok(default),
    }
}

fn bool_attr(node: &Node, name: &str, default: bool) -> Result<bool> {
    match int_attr(node, name, i64::from(default))? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(error(format!(
            "attribute {name} must be 0 or 1, got {value}"
        ))),
    }
}

fn float_attr(node: &Node, name: &str, default: f32) -> Result<f32> {
    match node.attr(name) {
        Some(value) => value
            .as_float()
            .ok_or_else(|| error(format!("attribute {name} must be a float"))),
        None => Ok(default),
    }
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!("{name} requires {expected:?}, got {got:?}")));
    }
    Ok(())
}

fn require_rank(name: &str, shape: &[usize], rank: usize) -> Result<()> {
    if shape.len() != rank {
        return Err(error(format!(
            "{name} must be {rank}-D, got shape {shape:?}"
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

fn checked_product(factors: &[usize], context: &str) -> Result<usize> {
    let mut product = 1usize;
    let mut has_zero = false;
    for &factor in factors {
        if factor == 0 {
            has_zero = true;
        } else {
            product = product
                .checked_mul(factor)
                .ok_or_else(|| error(format!("{context} exceeds usize limits")))?;
        }
    }
    Ok(if has_zero { 0 } else { product })
}

fn checked_bytes(elements: usize, element_size: usize, context: &str) -> Result<usize> {
    let bytes = elements
        .checked_mul(element_size)
        .ok_or_else(|| error(format!("{context} byte count exceeds usize limits")))?;
    if bytes > isize::MAX as usize {
        return Err(error(format!(
            "{context} byte count {bytes} exceeds isize::MAX"
        )));
    }
    Ok(bytes)
}

fn checked_tensor_layout(name: &str, shape: &[usize], dtype: DataType) -> Result<usize> {
    let elements = checked_product(shape, &format!("{name} element count"))?;
    checked_bytes(elements, dtype.byte_size(), name)?;
    Ok(elements)
}

fn as_u64(name: &str, value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| error(format!("{name}={value} exceeds CUDA u64 limits")))
}

fn as_i32(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| error(format!("{name}={value} exceeds CUDA i32 limits")))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep {DOMAIN}::{OP}: {}", message.into()))
}

#[cfg(test)]
mod workspace_tests {
    use super::*;

    #[test]
    fn workspace_layout_varies_with_prompt_top_k_and_expert_width() {
        for (rows, top_k, hidden, fc1_out) in
            [(1, 1, 128, 256), (17, 2, 512, 1024), (4096, 8, 2048, 6144)]
        {
            let layout = qmoe_workspace_layout(
                &[1, rows, hidden],
                &[rows, 64],
                &[64, fc1_out, 1, 1],
                top_k,
                1,
                false,
            )
            .unwrap();
            assert!(layout.bytes > 0);
            assert!(
                layout
                    .offsets
                    .iter()
                    .filter(|&&offset| offset != 0)
                    .all(|offset| offset % WORKSPACE_ALIGNMENT == 0)
            );
        }
    }

    #[test]
    fn workspace_layout_zero_rows_requires_no_bytes() {
        let layout =
            qmoe_workspace_layout(&[1, 0, 128], &[0, 8], &[8, 256, 1, 1], 2, 1, false).unwrap();
        assert_eq!(layout.bytes, 0);
        assert_eq!(layout.offsets, [0; 6]);
    }

    #[test]
    fn workspace_layout_rejects_overflow() {
        let error = qmoe_workspace_layout(
            &[usize::MAX, 2],
            &[usize::MAX, 8],
            &[8, 256, 1, 1],
            2,
            1,
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("exceeds usize limits"));
    }

    #[test]
    fn logical_traffic_arithmetic_rejects_overflow() {
        let logical = checked_logical_traffic_bytes(u64::MAX, 2, 1).unwrap_err();
        assert!(logical.to_string().contains("route-demand"));
        let unique = checked_logical_traffic_bytes(u64::MAX, 1, 2).unwrap_err();
        assert!(unique.to_string().contains("unique selected-expert"));
    }
}

#[cfg(test)]
mod sealed_bank_tests {
    use super::*;

    fn projection_bytes(
        format: BlockFormat,
        experts: usize,
        out_features: usize,
        in_features: usize,
    ) -> u64 {
        (experts * out_features * (in_features / format.qk()) * format.block_bytes()) as u64
    }

    fn bank<'a>(
        format: &'a str,
        packed: &'a [u8],
        in_features: usize,
    ) -> BlockQuantizedMoeBank<'a> {
        BlockQuantizedMoeBank {
            format,
            packed,
            experts: 1,
            out_features: 1,
            in_features,
        }
    }

    #[test]
    fn host_admission_rejects_reserved_tail_length_and_overflow() {
        let mut reserved = vec![0u8; 17];
        reserved[0] = 0xff;
        let error = validate_host_bank("fc1", bank("mxfp4", &reserved, 32)).unwrap_err();
        assert!(error.to_string().contains("reserved MXFP4"));

        let finite = vec![0u8; 17];
        let error = validate_host_bank("fc1", bank("mxfp4", &finite, 33)).unwrap_err();
        assert!(error.to_string().contains("partial"));

        let error = validate_host_bank("fc1", bank("mxfp4", &[], 32)).unwrap_err();
        assert!(error.to_string().contains("expected 17"));

        let overflow = BlockQuantizedMoeBank {
            format: "q2_k",
            packed: &[],
            experts: usize::MAX,
            out_features: usize::MAX,
            in_features: 256,
        };
        let error = validate_host_bank("fc1", overflow).unwrap_err();
        assert!(error.to_string().contains("overflow"));
    }

    #[test]
    fn glm52_three_projection_traffic_arithmetic_is_closed() {
        let h = 6144;
        let i = 2048;
        let per_expert = |gate: BlockFormat, down: BlockFormat| {
            projection_bytes(gate, 1, i, h) * 2 + projection_bytes(down, 1, h, i)
        };
        let cases = [
            (BlockFormat::Iq1S, BlockFormat::Iq3Xxs, 9_732_096u64),
            (BlockFormat::Iq2Xxs, BlockFormat::Iq3Xxs, 11_304_960),
            (BlockFormat::Iq2Xxs, BlockFormat::Iq4Xs, 13_172_736),
            (BlockFormat::Q2K, BlockFormat::Q3K, 13_664_256),
        ];
        for (gate, down, expected) in cases {
            let bytes = per_expert(gate, down);
            assert_eq!(bytes, expected);
            assert_eq!(bytes * 8, expected * 8);
            assert_eq!(bytes * 256, expected * 256);
        }
        assert_eq!(9_732_096u64 * 256, 2_491_416_576);
        assert_eq!(11_304_960u64 * 256, 2_894_069_760);
        assert_eq!(13_172_736u64 * 256, 3_372_220_416);
        assert_eq!(13_664_256u64 * 256, 3_498_049_536);
        assert_eq!(
            53 * (9_732_096u64 * 8)
                + 18 * (11_304_960u64 * 8)
                + 4 * (13_172_736u64 * 8)
                + 13_664_256u64 * 8,
            6_285_164_544
        );
    }

    #[test]
    fn host_admission_rejects_non_finite_embedded_scale() {
        let mut q8 = vec![0u8; 34];
        q8[..2].copy_from_slice(&half::f16::NAN.to_le_bytes());
        let error = validate_host_bank("fc2", bank("q8_0", &q8, 32)).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }
}

#[cfg(test)]
mod claim_gate_tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId, ValueId, static_shape};

    /// Build a minimal `BlockQuantizedMoE` node carrying only the attributes the
    /// claim gate reads. `fc3` set means the `fc3_experts_weights` input (index
    /// 6) is wired and an `fc3_format` attribute is emitted. These nodes are not
    /// executable — they exist purely to exercise `unsupported_reason` /
    /// `claim_projection_formats`, which are pure functions over node metadata
    /// and never touch a CUDA device, so this runs on a host without a GPU.
    fn claim_node(fc1: &str, fc2: &str, fc3: Option<&str>) -> Node {
        let mut inputs = vec![None; INPUT_COUNT];
        for index in [0usize, 1, 2, 4] {
            inputs[index] = Some(ValueId(index as u32));
        }
        if fc3.is_some() {
            inputs[6] = Some(ValueId(6));
        }
        let mut node = Node::new(
            NodeId(0),
            "BlockQuantizedMoE",
            inputs.clone(),
            vec![ValueId(100)],
        );
        node.attributes.insert(
            "fc1_format".into(),
            Attribute::String(fc1.as_bytes().to_vec()),
        );
        node.attributes.insert(
            "fc2_format".into(),
            Attribute::String(fc2.as_bytes().to_vec()),
        );
        if let Some(fc3) = fc3 {
            node.attributes.insert(
                "fc3_format".into(),
                Attribute::String(fc3.as_bytes().to_vec()),
            );
        }
        for (prefix, format, scale_index) in [
            ("fc1", fc1, BQMOE_FC1_SCALE),
            ("fc2", fc2, BQMOE_FC2_SCALE),
            ("fc3", fc3.unwrap_or(""), BQMOE_FC3_SCALE),
        ] {
            if matches!(format, "block_fp8" | "fp4_planar") {
                inputs[scale_index] = Some(ValueId(scale_index as u32));
                let (block_out, block_in) = if format == "fp4_planar" {
                    (1, 32)
                } else {
                    (32, 32)
                };
                node.attributes.insert(
                    format!("{prefix}_block_size_out"),
                    Attribute::Int(block_out),
                );
                node.attributes
                    .insert(format!("{prefix}_block_size_in"), Attribute::Int(block_in));
            }
        }
        node.inputs = inputs;
        node.attributes
            .insert("block_layout_version".into(), Attribute::Int(1));
        node
    }

    fn claim_reason(node: &Node) -> Option<Cow<'static, str>> {
        let mut shapes = vec![vec![]; INPUT_COUNT];
        let mut dtypes = vec![DataType::Undefined; INPUT_COUNT];
        shapes[0] = static_shape([1, 256]);
        shapes[1] = static_shape([1, 2]);
        dtypes[0] = DataType::Float32;
        dtypes[1] = DataType::Float32;
        for (weight, scale, attr) in [
            (BQMOE_FC1_WEIGHT, BQMOE_FC1_SCALE, "fc1_format"),
            (BQMOE_FC2_WEIGHT, BQMOE_FC2_SCALE, "fc2_format"),
            (BQMOE_FC3_WEIGHT, BQMOE_FC3_SCALE, "fc3_format"),
        ] {
            if node.inputs[weight].is_none() {
                continue;
            }
            let format = node
                .attr(attr)
                .and_then(|value| value.as_str())
                .unwrap_or("");
            match format {
                "block_fp8" => {
                    shapes[weight] = static_shape([2, 256, 256]);
                    shapes[scale] = static_shape([2, 8, 8]);
                    dtypes[weight] = DataType::Float8E4M3FN;
                    dtypes[scale] = DataType::Float8E8M0;
                }
                "fp4_planar" => {
                    shapes[weight] = static_shape([2, 256, 128]);
                    shapes[scale] = static_shape([2, 256, 8]);
                    dtypes[weight] = DataType::Int8;
                    dtypes[scale] = DataType::Float8E8M0;
                }
                _ => {
                    if let Ok(format) = BlockFormat::parse(format) {
                        shapes[weight] =
                            static_shape([2, 256, 256 / format.qk(), format.block_bytes()]);
                    }
                    dtypes[weight] = DataType::Uint8;
                }
            }
        }
        unsupported_reason(node, &shapes, &dtypes)
    }

    #[test]
    fn mixed_fc1_fc2_formats_are_claimable() {
        // The real GLM-5.2 UD-IQ1_S combo: gate/up IQ1_S, down IQ3_XXS.
        let node = claim_node("iq1_s", "iq3_xxs", None);
        assert!(
            claim_reason(&node).is_none(),
            "mixed per-projection formats must be claimed"
        );
        let final_layer = claim_node("q2_k", "q3_k", None);
        assert!(claim_reason(&final_layer).is_none());
    }

    #[test]
    fn mixed_fc3_gate_format_is_claimable() {
        // Unfused gate carried at a different qtype than fc1/fc2.
        let node = claim_node("iq1_s", "iq1_s", Some("iq2_xxs"));
        assert!(claim_reason(&node).is_none());
    }

    #[test]
    fn uniform_projection_formats_remain_claimable() {
        // Uniform format across every projection stays claimable by the device
        // kernel — the mixed-rejection must not over-reject the supported case.
        let node = claim_node("iq1_s", "iq1_s", Some("iq1_s"));
        assert!(
            claim_reason(&node).is_none(),
            "uniform-format node must not be declined by the CUDA claim gate"
        );
        let fused = claim_node("iq4_nl", "iq4_nl", None);
        assert!(
            claim_reason(&fused).is_none(),
            "uniform fused-projection node must not be declined by the CUDA claim gate"
        );
    }

    #[test]
    fn invalid_activation_attributes_reject_at_claim_and_create_parsing() {
        for (name, value) in [
            ("activation_alpha", f32::NAN),
            ("activation_alpha", f32::INFINITY),
            ("activation_alpha", f32::NEG_INFINITY),
            ("activation_beta", f32::NAN),
            ("activation_beta", f32::INFINITY),
            ("activation_beta", f32::NEG_INFINITY),
            ("swiglu_limit", f32::NAN),
            ("swiglu_limit", f32::INFINITY),
            ("swiglu_limit", f32::NEG_INFINITY),
            ("swiglu_limit", 0.0),
            ("swiglu_limit", -1.0),
        ] {
            let mut node = claim_node("iq1_s", "iq1_s", None);
            node.attributes.insert(
                "activation_type".into(),
                Attribute::String(b"swiglu".to_vec()),
            );
            node.attributes
                .insert("swiglu_fusion".into(), Attribute::Int(1));
            node.attributes.insert(name.into(), Attribute::Float(value));
            let reason = claim_reason(&node)
                .unwrap_or_else(|| panic!("{name}={value} must be declined at claim time"));
            assert!(reason.contains(name), "unexpected claim reason: {reason}");
            let error = MoeAttributes::from_node(&node)
                .expect_err("the same attribute must fail factory/create parsing");
            assert!(
                error.to_string().contains(name),
                "unexpected create error: {error}"
            );
        }
    }

    #[test]
    fn unsupported_native_format_is_typed_rejected_at_the_claim_gate() {
        // A native GGUF qtype outside BlockFormat (Q4_K) is declined, not
        // dequantized or dense-fallback executed.
        let node = claim_node("q4_k", "q4_k", None);
        let reason = claim_reason(&node)
            .expect("an unsupported native format must be declined by the CUDA claim gate");
        assert!(
            reason.contains("q4_k"),
            "unexpected rejection reason: {reason}"
        );
    }

    #[test]
    fn fc3_format_without_wired_gate_is_typed_rejected() {
        // fc3_format present but the fc3 weights input is not wired: reject.
        let mut node = claim_node("iq1_s", "iq1_s", None);
        node.attributes
            .insert("fc3_format".into(), Attribute::String(b"iq1_s".to_vec()));
        let reason = claim_reason(&node).expect("fc3_format without a wired gate must be declined");
        assert!(
            reason.contains("fc3_format is only valid when fc3_experts_weights is wired"),
            "unexpected rejection reason: {reason}"
        );
    }

    #[test]
    fn planar_b2_formats_are_claimable_with_aux_scales() {
        for planar in ["block_fp8", "fp4_planar"] {
            let node = claim_node(planar, planar, None);
            assert!(claim_reason(&node).is_none(), "{planar} must be claimable");
        }
    }

    #[test]
    fn planar_b2_format_on_a_single_projection_is_typed_rejected() {
        // A mixed node where only the routed (fc2) projection is planar-FP4 must
        // still be declined; the planar recognition fires per projection.
        let node = claim_node("iq1_s", "fp4_planar", None);
        let reason = claim_reason(&node)
            .expect("a planar projection anywhere must be declined by the CUDA claim gate");
        assert!(
            reason.contains("mixing planar and interleaved"),
            "unexpected rejection reason: {reason}"
        );
    }
}
