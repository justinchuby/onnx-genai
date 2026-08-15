//! CUDA implementation of the frozen `pkg.nxrt::BlockQuantizedMoE` v1 operator.
//!
//! This is the CUDA counterpart to the CPU parity oracle
//! ([`onnx_runtime_ep_cpu::kernels::block_quantized_moe`]). Expert weights stay
//! resident on one GPU packed in the native GGUF block formats (mxfp4, iq4_nl,
//! iq4_xs, iq2_xxs, iq3_xxs, iq2_xs, iq2_s, iq3_s, iq1_s, iq1_m). Per-weight
//! dequantization reuses the exact `decode_weight` device routine that backs
//! [`super::block_quantized_matmul`], so the numeric semantics match the oracle
//! block-for-block; only the reduction/accumulation order differs (both
//! accumulate in f32).
//!
//! The pipeline mirrors the CPU reference: host-free top-k routing, per-route
//! expert GEMV for FC1 (and the optional FC3 gate), a fused
//! activation/SwiGLU pass, the FC2 down-projection, and a weighted combine of
//! each token's selected experts. All heavy work runs on the EP's non-default
//! stream; the trailing host synchronization guards the eager (non-captured)
//! path.

use std::borrow::Cow;
use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView,
    WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node, Shape};
use onnx_runtime_memory_governor::MemoryRole;

use crate::error::driver_err;
use crate::kernels::block_quantized_matmul::{BlockFormat, decoder_prelude};
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "BlockQuantizedMoE";
const DOMAIN: &str = onnx_runtime_ir::RUNTIME_DOMAIN;
const LAYOUT_VERSION: i64 = 1;

const MODULE: &str = "block_quantized_moe_v1";
const ROUTE_ENTRY: &str = "bqmoe_route";
const LINEAR_ENTRY: &str = "bqmoe_linear_f32";
const ACTIVATE_ENTRY: &str = "bqmoe_activate";
const COMBINE_ENTRY: &str = "bqmoe_combine_f32";

const INPUT_NAMES: [&str; 9] = [
    "input",
    "router_logits",
    "fc1_experts_weights",
    "fc1_experts_bias",
    "fc2_experts_weights",
    "fc2_experts_bias",
    "fc3_experts_weights",
    "fc3_experts_bias",
    "router_weights",
];

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
    const int normalize)
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
                    expert_packed, format, blocks, block_bytes,
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
        source.push_str(KERNELS);
        source
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
    Relu,
    Gelu,
    Silu,
    Swiglu,
    Identity,
}

impl Activation {
    fn parse(node: &Node) -> Result<Self> {
        let name = match node.attr("activation_type") {
            Some(value) => value
                .as_str()
                .ok_or_else(|| error("attribute activation_type must be a string"))?,
            None => "relu",
        };
        match name {
            "relu" => Ok(Self::Relu),
            "gelu" => Ok(Self::Gelu),
            "silu" => Ok(Self::Silu),
            "swiglu" => Ok(Self::Swiglu),
            "identity" => Ok(Self::Identity),
            other => Err(error(format!(
                "unsupported activation_type '{other}' (supported: relu, gelu, silu, swiglu, identity)"
            ))),
        }
    }

    fn kernel_id(self) -> i32 {
        match self {
            Self::Relu => 0,
            Self::Gelu => 1,
            Self::Silu => 2,
            Self::Swiglu => 3,
            Self::Identity => 4,
        }
    }
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
                    | "format"
                    | "block_layout_version"
            ) {
                return Err(error(format!(
                    "attribute '{name}' is not part of the frozen v1 ABI"
                )));
            }
        }
        let k = int_attr(node, "k", 1)?;
        if k <= 0 {
            return Err(error(format!("k must be > 0, got {k}")));
        }
        let activation = Activation::parse(node)?;
        let normalize_routing_weights = bool_attr(node, "normalize_routing_weights", false)?;
        let swiglu_fusion = int_attr(node, "swiglu_fusion", 0)?;
        if !(0..=2).contains(&swiglu_fusion) {
            return Err(error(format!(
                "swiglu_fusion must be 0, 1, or 2, got {swiglu_fusion}"
            )));
        }
        if activation != Activation::Swiglu && swiglu_fusion != 0 {
            return Err(error(
                "swiglu_fusion is only valid when activation_type='swiglu'",
            ));
        }
        Ok(Self {
            k: usize::try_from(k).map_err(|_| error("k exceeds usize limits"))?,
            activation,
            normalize_routing_weights,
            swiglu_fusion: swiglu_fusion as usize,
            activation_alpha: float_attr(node, "activation_alpha", 1.0)?,
            activation_beta: float_attr(node, "activation_beta", 0.0)?,
            swiglu_limit: float_attr(node, "swiglu_limit", f32::INFINITY)?,
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
    let layout_version = int_attr(node, "block_layout_version", LAYOUT_VERSION)?;
    if layout_version != LAYOUT_VERSION {
        return Err(error(format!(
            "block_layout_version must be {LAYOUT_VERSION}, got {layout_version}"
        )));
    }
    Ok(())
}

fn parse_format(node: &Node) -> Result<BlockFormat> {
    node.attr("format")
        .ok_or_else(|| error("missing required string attribute 'format'"))?
        .as_str()
        .ok_or_else(|| error("attribute 'format' must be a UTF-8 string"))
        .and_then(BlockFormat::parse)
}

pub struct BlockQuantizedMoEFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for BlockQuantizedMoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        parse_layout_version(node)?;
        let attributes = MoeAttributes::from_node(node)?;
        let format = parse_format(node)?;
        Ok(Box::new(BlockQuantizedMoEKernel {
            runtime: self.runtime.clone(),
            attributes,
            format,
        }))
    }
}

/// Placement declaration for the CUDA claim gate. The CUDA kernel implements the
/// full frozen v1 ABI over the ten CUDA-supported GGUF block formats with f32
/// activations. It declines any node the kernel cannot execute (unsupported
/// format, wrong layout version, or non-f32 activation/router dtypes) so those
/// nodes fall back to the CPU oracle rather than mis-executing.
pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    let format = match node.attr("format") {
        Some(attribute) => match attribute.as_str() {
            Some(format) => format,
            None => {
                return Some(Cow::Borrowed(
                    "BlockQuantizedMoE: attribute 'format' must be a string naming a CUDA-supported block format",
                ));
            }
        },
        None => {
            return Some(Cow::Borrowed(
                "BlockQuantizedMoE: missing required string attribute 'format' — export one of mxfp4, iq4_nl, iq4_xs, iq2_xxs, iq3_xxs, iq2_xs, iq2_s, iq3_s, iq1_s, or iq1_m",
            ));
        }
    };
    if BlockFormat::parse(format).is_err() {
        return Some(Cow::Owned(format!(
            "BlockQuantizedMoE: CUDA does not support format '{format}' — re-export weights as mxfp4, iq4_nl, iq4_xs, iq2_xxs, iq3_xxs, iq2_xs, iq2_s, iq3_s, iq1_s, or iq1_m"
        )));
    }
    if let Some(attribute) = node.attr("block_layout_version") {
        match attribute.as_int() {
            Some(version) if version == LAYOUT_VERSION => {}
            Some(version) => {
                return Some(Cow::Owned(format!(
                    "BlockQuantizedMoE: CUDA requires block_layout_version={LAYOUT_VERSION}, got {version} — re-export the packed weights with the current layout"
                )));
            }
            None => {
                return Some(Cow::Borrowed(
                    "BlockQuantizedMoE: block_layout_version must be an integer — re-export the packed weights with the current layout",
                ));
            }
        }
    }
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
    let _ = shapes;
    None
}

pub struct BlockQuantizedMoEKernel {
    runtime: Arc<CudaRuntime>,
    attributes: MoeAttributes,
    format: BlockFormat,
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
    if fc1_shape.len() != 4 {
        return Err(error(format!(
            "fc1_experts_weights must be rank 4, got {fc1_shape:?}"
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

/// A validated per-projection packed expert-weight tensor. `packed` is the
/// expert-major `[experts, out_features, blocks, block_bytes]` buffer that
/// `decode_weight` indexes one weight at a time.
#[derive(Clone, Copy)]
struct PackedExperts<'a> {
    packed: &'a TensorView<'a>,
    bias: Option<&'a TensorView<'a>>,
    out_features: usize,
    in_features: usize,
    blocks: usize,
}

impl<'a> PackedExperts<'a> {
    #[allow(clippy::too_many_arguments)]
    fn validate(
        name: &str,
        packed: &'a TensorView<'a>,
        bias: Option<&'a TensorView<'a>>,
        experts: usize,
        out_features: usize,
        in_features: usize,
        format: BlockFormat,
    ) -> Result<Self> {
        require_dtype(
            &format!("{name}_experts_weights"),
            packed.dtype,
            DataType::Uint8,
        )?;
        let blocks = checked_div_ceil(in_features, format.qk(), &format!("{name} block count"))?;
        require_shape(
            &format!("{name}_experts_weights"),
            packed.shape,
            &[experts, out_features, blocks, format.block_bytes()],
        )?;
        checked_tensor_layout(
            &format!("{name}_experts_weights"),
            packed.shape,
            packed.dtype,
        )?;
        if !packed.is_contiguous() {
            return Err(error(format!(
                "{name}_experts_weights must be contiguous on the CUDA execution provider"
            )));
        }
        if let Some(bias) = bias {
            require_dtype(
                &format!("{name}_experts_bias"),
                bias.dtype,
                DataType::Float32,
            )?;
            require_shape(
                &format!("{name}_experts_bias"),
                bias.shape,
                &[experts, out_features],
            )?;
            checked_tensor_layout(&format!("{name}_experts_bias"), bias.shape, bias.dtype)?;
            if !bias.is_contiguous() {
                return Err(error(format!(
                    "{name}_experts_bias must be contiguous on the CUDA execution provider"
                )));
            }
        }
        Ok(Self {
            packed,
            bias,
            out_features,
            in_features,
            blocks,
        })
    }
}

impl Kernel for BlockQuantizedMoEKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let _ = (inputs, outputs);
        Err(error(
            "BlockQuantizedMoE requires workspace prepared before execution",
        ))
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        if inputs.len() < 5 {
            return Err(error(format!(
                "expected at least 5 input metadata entries, got {}",
                inputs.len()
            )));
        }
        let has_fc3 = inputs.get(6).is_some_and(|input| input.present);
        let layout = qmoe_workspace_layout(
            inputs[0].shape,
            inputs[1].shape,
            inputs[2].shape,
            self.attributes.k,
            self.attributes.swiglu_fusion,
            has_fc3,
        )?;
        Ok(WorkspaceRequirement {
            bytes: u64::try_from(layout.bytes)
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
        if !(5..=9).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(error(format!(
                "expected 5 to 9 inputs and exactly 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        for &index in &[0usize, 1, 2, 4] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{}') is absent",
                    INPUT_NAMES[index]
                )));
            }
        }
        require_dtype("input", inputs[0].dtype, DataType::Float32)?;
        require_dtype("router_logits", inputs[1].dtype, DataType::Float32)?;
        if outputs[0].dtype != DataType::Float32 {
            return Err(error(format!(
                "output dtype {:?} unsupported; expected Float32",
                outputs[0].dtype
            )));
        }

        let input_shape = inputs[0].shape;
        if !matches!(input_shape.len(), 2 | 3) {
            return Err(error(format!(
                "input must be 2-D [rows, hidden] or 3-D [batch, sequence, hidden], got {input_shape:?}"
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
            return Err(error(format!(
                "router_logits rows {} must equal flattened input rows {rows}",
                inputs[1].shape[0]
            )));
        }
        let experts = inputs[1].shape[1];
        if self.attributes.k > experts {
            return Err(error(format!(
                "requires 0 < k <= num_experts, got k={} and num_experts={experts}",
                self.attributes.k
            )));
        }

        require_rank("fc1_experts_weights", inputs[2].shape, 4)?;
        require_rank("fc2_experts_weights", inputs[4].shape, 4)?;
        if inputs[2].shape[0] != experts || inputs[4].shape[0] != experts {
            return Err(error(format!(
                "expert weight counts must equal router num_experts {experts}"
            )));
        }
        if inputs[4].shape[1] != hidden {
            return Err(error(format!(
                "fc2_experts_weights must start [experts={experts}, H={hidden}], got {:?}",
                inputs[4].shape
            )));
        }
        let fc1_out = inputs[2].shape[1];
        let inter = if self.attributes.swiglu_fusion == 0 {
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
        let expected_fc1 = self.attributes.fc1_size(inter)?;
        if fc1_out != expected_fc1 {
            return Err(error(format!(
                "fc1_experts_weights dimension 1 must be {expected_fc1}, got {fc1_out}"
            )));
        }

        let fc1 = PackedExperts::validate(
            "fc1",
            &inputs[2],
            optional_input(inputs, 3),
            experts,
            fc1_out,
            hidden,
            self.format,
        )?;
        let fc2 = PackedExperts::validate(
            "fc2",
            &inputs[4],
            optional_input(inputs, 5),
            experts,
            hidden,
            inter,
            self.format,
        )?;

        let has_fc3 = optional_input(inputs, 6).is_some();
        let uses_separate_gate = self.attributes.uses_separate_gate(has_fc3);
        let fc3 = if uses_separate_gate {
            Some(PackedExperts::validate(
                "fc3",
                optional_input(inputs, 6)
                    .ok_or_else(|| error("unfused swiglu requires input 6 fc3_experts_weights"))?,
                optional_input(inputs, 7),
                experts,
                inter,
                hidden,
                self.format,
            )?)
        } else {
            for (index, name) in [(6, "fc3_experts_weights"), (7, "fc3_experts_bias")] {
                if optional_input(inputs, index).is_some() {
                    return Err(error(format!(
                        "{name} is only valid for unfused swiglu or silu gated-GLU"
                    )));
                }
            }
            None
        };

        let router_weights = optional_input(inputs, 8);
        if let Some(router_weights) = router_weights {
            require_dtype("router_weights", router_weights.dtype, DataType::Float32)?;
            require_shape("router_weights", router_weights.shape, &[rows, experts])?;
            checked_tensor_layout("router_weights", router_weights.shape, router_weights.dtype)?;
            if !router_weights.is_contiguous() {
                return Err(error(
                    "router_weights must be contiguous on the CUDA execution provider",
                ));
            }
        }
        for (name, tensor) in [("input", &inputs[0]), ("router_logits", &inputs[1])] {
            checked_tensor_layout(name, tensor.shape, tensor.dtype)?;
            if !tensor.is_contiguous() {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }
        checked_tensor_layout("output", outputs[0].shape, outputs[0].dtype)?;
        if !outputs[0].is_contiguous() {
            return Err(error(
                "output must be contiguous on the CUDA execution provider",
            ));
        }
        if rows == 0 || hidden == 0 {
            return Ok(());
        }

        let layout = qmoe_workspace_layout(
            input_shape,
            inputs[1].shape,
            inputs[2].shape,
            self.attributes.k,
            self.attributes.swiglu_fusion,
            fc3.is_some(),
        )?;
        let routes = checked_product(&[rows, self.attributes.k], "route count")?;
        let workspace = workspace.ok_or_else(|| {
            error("BlockQuantizedMoE execute reached compute without prepared workspace")
        })?;
        if workspace.bytes() < layout.bytes {
            return Err(error(format!(
                "BlockQuantizedMoE prepared-workspace invariant violated: execution requires {} bytes but only {} were prepared",
                layout.bytes,
                workspace.bytes()
            )));
        }
        let base = cuptr(workspace.ptr().0.cast_const());
        let ptr = |index: usize| base + layout.offsets[index] as u64;
        let route_indices = ptr(0);
        let route_weights_ptr = ptr(1);
        let fc1_output = ptr(2);
        let fc3_output = fc3.map(|_| ptr(3));
        let activated = ptr(4);
        let route_output = ptr(5);

        self.launch_route(
            &inputs[1],
            router_weights,
            route_indices,
            route_weights_ptr,
            rows,
            experts,
        )?;
        self.launch_linear(
            tensor_ptr(&inputs[0]),
            route_indices,
            fc1,
            fc1_output,
            routes,
            false,
        )?;
        if let (Some(fc3), Some(fc3_output)) = (fc3, fc3_output) {
            self.launch_linear(
                tensor_ptr(&inputs[0]),
                route_indices,
                fc3,
                fc3_output,
                routes,
                false,
            )?;
        }
        self.launch_activation(fc1_output, fc3_output, activated, routes, inter)?;
        self.launch_linear(activated, route_indices, fc2, route_output, routes, true)?;
        self.launch_combine(
            route_output,
            route_weights_ptr,
            &mut outputs[0],
            rows,
            hidden,
        )?;
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        onnx_runtime_ep_api::CaptureSupport::unsupported(
            "block-quantized MoE performs a trailing host stream synchronization",
        )
    }
}

impl BlockQuantizedMoEKernel {
    fn preferred_threads(&self) -> u32 {
        let capabilities = self.runtime.capabilities();
        let preferred = if capabilities.compute_capability().0 >= 7 {
            256
        } else {
            128
        };
        preferred.min(capabilities.max_threads_per_block()).max(1)
    }

    fn saturating_grid(&self, units: u64) -> u32 {
        let capabilities = self.runtime.capabilities();
        let saturation = u64::from(capabilities.multiprocessor_count()).saturating_mul(16);
        let grid = units.min(saturation.max(1)).min(u64::from(u32::MAX)).max(1);
        grid as u32
    }

    fn pointwise_launch_config(&self, total: u64) -> LaunchConfig {
        let threads = self.preferred_threads();
        let blocks_needed = total.div_ceil(u64::from(threads)).max(1);
        LaunchConfig {
            grid_dim: (self.saturating_grid(blocks_needed), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn launch_route(
        &self,
        router_logits: &TensorView,
        router_weights: Option<&TensorView>,
        route_indices: CUdeviceptr,
        route_weights: CUdeviceptr,
        rows: usize,
        experts: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(MODULE, module_source(), ROUTE_ENTRY)?;
        let router_logits_ptr = tensor_ptr(router_logits);
        let router_weights_ptr = router_weights.map(tensor_ptr).unwrap_or(0);
        let rows_u64 = as_u64("row count", rows)?;
        let experts_i32 = as_i32("expert count", experts)?;
        let top_k = as_i32("top-k", self.attributes.k)?;
        let normalize = i32::from(self.attributes.normalize_routing_weights);
        let config = self.pointwise_launch_config(rows_u64);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&router_logits_ptr)
            .arg(&router_weights_ptr)
            .arg(&route_indices)
            .arg(&route_weights)
            .arg(&rows_u64)
            .arg(&experts_i32)
            .arg(&top_k)
            .arg(&normalize);
        // SAFETY: scratch buffers cover rows*top_k entries and the scalar ABI
        // matches `bqmoe_route`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch BlockQuantizedMoE routing", err))
    }

    fn launch_linear(
        &self,
        input_ptr: CUdeviceptr,
        route_indices: CUdeviceptr,
        weights: PackedExperts<'_>,
        output: CUdeviceptr,
        routes: usize,
        input_rows_are_routes: bool,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(MODULE, module_source(), LINEAR_ENTRY)?;
        let tasks = checked_product(&[routes, weights.out_features], "linear task count")?;
        let grid_x = self.saturating_grid(as_u64("linear task count", tasks)?);
        // `block_sum` reserves 32 static shared floats; request one dynamic float
        // per thread so `reduction_launch_config` sizes the block against the
        // device's queried optin shared-memory budget without hardcoded SM caps.
        let config = self.runtime.reduction_launch_config(
            &function,
            grid_x,
            self.preferred_threads(),
            std::mem::size_of::<f32>() as u32,
        )?;
        let packed = tensor_ptr(weights.packed);
        let bias = weights.bias.map(tensor_ptr).unwrap_or(0);
        let routes_u64 = as_u64("route count", routes)?;
        let input_rows_are_routes = i32::from(input_rows_are_routes);
        let top_k = as_i32("top-k", self.attributes.k)?;
        let out_features = as_i32("output feature count", weights.out_features)?;
        let in_features = as_i32("input feature count", weights.in_features)?;
        let blocks = as_i32("block count", weights.blocks)?;
        let block_bytes = as_i32("block byte count", self.format.block_bytes())?;
        let format = self.format.kernel_id();
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&input_ptr)
            .arg(&route_indices)
            .arg(&packed)
            .arg(&bias)
            .arg(&output)
            .arg(&routes_u64)
            .arg(&input_rows_are_routes)
            .arg(&top_k)
            .arg(&out_features)
            .arg(&in_features)
            .arg(&blocks)
            .arg(&block_bytes)
            .arg(&format);
        // SAFETY: packed weights cover experts*out_features*blocks*block_bytes,
        // scratch buffers cover routes*out_features, and the scalar ABI matches
        // `bqmoe_linear_f32`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch BlockQuantizedMoE expert GEMV", err))
    }

    fn launch_activation(
        &self,
        fc1: CUdeviceptr,
        fc3: Option<CUdeviceptr>,
        activated: CUdeviceptr,
        routes: usize,
        inter: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(MODULE, module_source(), ACTIVATE_ENTRY)?;
        let total = checked_product(&[routes, inter], "activation element count")?;
        let config = self.pointwise_launch_config(as_u64("activation element count", total)?);
        let fc3 = fc3.unwrap_or(0);
        let routes_u64 = as_u64("route count", routes)?;
        let inter_i32 = as_i32("intermediate feature count", inter)?;
        let activation = self.attributes.activation.kernel_id();
        let swiglu_fusion = as_i32("swiglu_fusion", self.attributes.swiglu_fusion)?;
        let alpha = self.attributes.activation_alpha;
        let beta = self.attributes.activation_beta;
        let limit = self.attributes.swiglu_limit;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&fc1)
            .arg(&fc3)
            .arg(&activated)
            .arg(&routes_u64)
            .arg(&inter_i32)
            .arg(&activation)
            .arg(&swiglu_fusion)
            .arg(&alpha)
            .arg(&beta)
            .arg(&limit);
        // SAFETY: scratch buffers cover every routed intermediate element and the
        // ABI matches `bqmoe_activate`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch BlockQuantizedMoE activation", err))
    }

    fn launch_combine(
        &self,
        route_output: CUdeviceptr,
        route_weights: CUdeviceptr,
        output: &mut TensorMut,
        rows: usize,
        hidden: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(MODULE, module_source(), COMBINE_ENTRY)?;
        let total = checked_product(&[rows, hidden], "combined output element count")?;
        let config = self.pointwise_launch_config(as_u64("output element count", total)?);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let rows_u64 = as_u64("row count", rows)?;
        let hidden_i32 = as_i32("hidden feature count", hidden)?;
        let top_k = as_i32("top-k", self.attributes.k)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&route_output)
            .arg(&route_weights)
            .arg(&output_ptr)
            .arg(&rows_u64)
            .arg(&hidden_i32)
            .arg(&top_k);
        // SAFETY: routed output and weights cover rows*top_k, output covers
        // rows*hidden, and the ABI matches `bqmoe_combine_f32`.
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|err| driver_err("launch BlockQuantizedMoE weighted combine", err))
    }
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

fn checked_div_ceil(value: usize, divisor: usize, context: &str) -> Result<usize> {
    if divisor == 0 {
        return Err(error(format!("{context} divisor must be non-zero")));
    }
    value
        .checked_add(divisor - 1)
        .map(|adjusted| adjusted / divisor)
        .ok_or_else(|| error(format!("{context} exceeds usize limits")))
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
}
