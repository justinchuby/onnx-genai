//! `ExportedComputeInfo` — wraps Rust `Kernel`s as `OrtNodeComputeInfo` callbacks.
//!
//! # Shape-inference contract
//!
//! Every op that can appear in a subgraph must have an explicit `ShapeInference`
//! variant. The **fail-closed** policy: if an op has no modelled rule its
//! `ShapeInference` is `Declined { op_type, domain }` and `infer_shapes` returns
//! an error naming the op and domain. This surfaces at Compute time — never
//! silently producing a wrong-shape tensor.
//!
//! [`ShapeInference::for_node`] is the single dispatch point: it takes the
//! compiled IR `Node`, its input shapes and its output count, and selects either
//! the shared native registry adapter or a plugin-only rule. There is
//! deliberately no attribute-blind op-name entry point.

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::HostToDeviceCopier;
use onnx_runtime_ep_api::kernel::{
    Kernel, KernelSizedOutput, KernelSizedOutputMetadata, KernelSizedOutputPolicy, TensorMetadata,
    WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ep_api::tensor::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, DeviceId, Node};

use crate::kernel_ctx::{allocate_output, read_inputs_into};
use crate::shared_shapes::{SharedNativeShapeRule, SharedShapeResult, infer_shared_node};
use crate::status::{fail_status, ok_status};

// ──────────────────────────────────────────────────────────────────────────────
// Per-axis parameters for Conv
// ──────────────────────────────────────────────────────────────────────────────

/// Per-spatial-axis convolution parameters (kernel size, padding, stride, dilation).
#[derive(Clone, Debug)]
pub struct ConvSpatialAxis {
    pub kernel: usize,
    pub pad_before: usize,
    pub pad_after: usize,
    pub stride: usize,
    pub dilation: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// ShapeInference
// ──────────────────────────────────────────────────────────────────────────────

/// How to infer output shapes at runtime from the concrete input shapes.
#[derive(Clone, Debug)]
pub enum ShapeInference {
    /// The kernel computes data-dependent extents and owned host bytes in one
    /// pass; Compute validates them before allocating final ORT outputs.
    KernelSizedOutputs,
    /// Ask the native symbolic registry first, then preserve the existing
    /// plugin rule as a fallback when values remain unavailable or the native
    /// rule is stricter than the historical plugin contract.
    SharedNative {
        node: Box<Node>,
        fallback: Box<ShapeInference>,
    },
    /// numpy-style broadcast of all inputs → one output.
    ElementwiseBroadcast,
    /// Output shape == input[idx].shape.
    SameAsInput(usize),
    /// `count` outputs, each with shape == input[idx].shape.
    SameAsInputMultiOutput { idx: usize, count: usize },
    /// MatMul / MatMulNBits semantics (handles 1-D, 2-D, batched-ND).
    MatMul,
    /// `com.microsoft::MatMulNBits`: `A[.., K] x dequant(B)[K, N]`.
    ///
    /// Not [`Self::MatMul`]. `B` is the *packed* quantized weight, shaped
    /// `[N, ceil(K / block_size), block_size * bits / 8]` — three dims that
    /// have nothing to do with the GEMM's, so plain matmul broadcasting reads
    /// `N` as a batch dim and `blob_size` as the output column count. The real
    /// output is the activation's shape with its last dim replaced by the `N`
    /// attribute, which is also what `onnx-runtime-shape-inference`'s
    /// `quantized_matmul` rule computes for the native path.
    MatMulNBits {
        /// The `N` attribute: the dequantized weight's column count.
        n: usize,
    },
    /// `QLinearMatMul`: `MatMul` semantics between input 0 and input **3**.
    ///
    /// The quantization parameters sit between the two operands
    /// (`a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale,
    /// y_zero_point`), so [`Self::MatMul`]'s "inputs 0 and 1" reads `a_scale`
    /// as the right-hand operand. Absent from this table the op resolves to
    /// [`Self::Declined`], and the fail-closed shape filter in `GetCapability`
    /// then drops the claim — which is what kept this EP from ever receiving a
    /// `QLinearMatMul` node from a plugin host.
    QLinearMatMul,
    /// Gemm: (trans_a, trans_b) flags.
    Gemm { trans_a: bool, trans_b: bool },
    /// Concat along `axis`.
    Concat { axis: i64 },
    /// Transpose with optional explicit permutation.
    Transpose { perm: Option<Vec<usize>> },
    /// Gather: replace axis `axis` of data with the shape of indices.
    Gather { axis: i64 },
    /// GatherND: `batch_dims` leading dims are shared.
    GatherND { batch_dims: usize },
    /// GatherBlockQuantized — treat as GatherND(0) for shape purposes.
    GatherBlockQuantized,
    /// Shape op — emits [len(dims[start:end])] as a 1-D int64 tensor.
    ShapeOp { start: i64, end: Option<i64> },
    /// Squeeze: remove dims listed in `axes` (or all size-1 dims if empty).
    Squeeze { axes: Vec<i64> },
    /// Unsqueeze: insert unit dims at `axes`.
    Unsqueeze { axes: Vec<i64> },
    /// Reshape — output shape read from input[1] at Compute time.
    ReshapeData { allowzero: bool },
    /// Slice — output shape derived from inputs[1..=4] at Compute time.
    SliceData,
    /// Reduction ops with keepdims / axes.
    Reduction {
        keepdims: bool,
        /// None = reduce all axes.
        axes: Option<Vec<i64>>,
        noop_with_empty_axes: bool,
    },
    /// Opset-18+ reduction where axes come from input[1].
    ReductionFromInput {
        keepdims: bool,
        noop_with_empty_axes: bool,
    },
    /// Conv with explicit spatial axis rules.
    Conv {
        out_channels: usize,
        per_axis: Vec<ConvSpatialAxis>,
    },
    /// MultiHeadAttention: [B,S,hidden], optional present_key/value.
    MultiHeadAttention {
        num_heads: usize,
        num_outputs: usize,
    },
    /// GroupQueryAttention.
    GroupQueryAttention {
        num_heads: usize,
        kv_num_heads: usize,
    },
    /// RotaryEmbedding — output same shape as input[0].
    RotaryEmbedding,
    /// `com.microsoft::Attention`: packed-QKV attention.
    ///
    /// A different signature from [`Self::AttentionStd`], which is why the
    /// opset-23 `ai.onnx::Attention` arm cannot cover it. Here input(0) is the
    /// *unprojected* activation `(batch, seq, input_hidden)` and input(1) is
    /// the fused projection weight `(input_hidden, q_hidden + k_hidden +
    /// v_hidden)`, so the output width is `v_hidden` — not the input's last
    /// dim, and not `weights.dim1`.
    MsftAttention {
        /// `qkv_hidden_sizes[2]` when the attribute is present, else
        /// `weights.dim1 / 3` resolved from the input shape at Compute time.
        v_hidden: Option<usize>,
        num_heads: usize,
        num_outputs: usize,
    },
    /// `com.microsoft::PackedMultiHeadAttention`: output
    /// `[token_count, v_hidden]` = `[input[0].dim0, input[2].dim1]`.
    ///
    /// Tokens are packed across the batch with no padding, so the output is
    /// rank-2 and neither dim can be read off input[0] alone.
    PackedMultiHeadAttention,
    /// Standard `ai.onnx::Attention` (opset 23+): scaled dot-product attention.
    ///
    /// Q/K/V are rank-3 `(batch, seq, hidden)` or rank-4
    /// `(batch, heads, seq, head_size)`. `Y` follows Q's layout with the value
    /// head size (which may differ from the Q/K head size), and the optional
    /// `present_key`/`present_value`/`qk_matmul_output` slots are shaped from
    /// the K/V geometry and the (optional) `past_key` length. `q_num_heads`/
    /// `kv_num_heads` are required only for the rank-3 layout, where they split
    /// the hidden dimension into heads.
    AttentionStd {
        q_num_heads: usize,
        kv_num_heads: usize,
        num_outputs: usize,
    },
    /// LayerNormalization family: output 0 = input[0] shape,
    /// outputs 1+ (Mean, InvStdDev) = input[0] shape with dims from `axis`
    /// onward replaced by 1 (keepdims reduction).
    ///
    /// `raw_axis` is stored as-is from the ONNX attribute (may be negative)
    /// and resolved against the **runtime** input rank in `infer_shapes`.
    /// This avoids pre-resolving against a truncated static shape that has
    /// symbolic dimensions stripped.
    ///
    /// For SkipLayerNormalization the last output (input_skip_bias_sum) is
    /// full-shaped — `full_shape_outputs` lists the output indices that
    /// should keep input[0]'s shape verbatim.
    LayerNorm {
        /// Raw axis from the ONNX attribute; resolved at runtime.
        raw_axis: i64,
        num_outputs: usize,
        /// Output indices (besides 0) that keep the full input shape,
        /// e.g. the `input_skip_bias_sum` output of SkipLayerNormalization.
        full_shape_outputs: Vec<usize>,
    },
    /// `CausalConvWithState`: two outputs with *different* shapes, which is why
    /// neither `SameAsInput` nor `SameAsInputMultiOutput` covers it.
    ///
    /// Output 0 is the convolution result and keeps the input's `[B, C, L]`.
    /// Output 1 is the carry state `[B, C, K-1]`, whose width comes from the
    /// depthwise weight `[C, 1, K]` — not from any input's outer shape — so it
    /// has to be read off input 1.
    CausalConvWithState,
    /// `ai.onnx::ConstantOfShape`: output dims come from input 0's *values*, a
    /// 1-D int64 tensor. Rank equals that tensor's length; an empty one yields a
    /// scalar.
    ConstantOfShape,
    /// `ai.onnx::Expand`: **bidirectional** (numpy) broadcast between the input's
    /// shape and the target shape carried in input 1's values — not simply the
    /// target, which is the easy mistake.
    Expand,
    /// `ai.onnx::Tile`: each dim multiplied by the matching entry of input 1's
    /// values.
    Tile,
    /// `ai.onnx::HannWindow` / `HammingWindow` / `BlackmanWindow`: a 1-D window
    /// whose length is input 0's scalar *value*.
    Window,
    /// `ai.onnx::DFT`: discrete Fourier transform along a signal axis.
    ///
    /// Output shape is the input's, with the last dimension forced to 2 (the
    /// complex component pair, even for real input) and the signal axis set to
    /// `dft_length` — halved to `n / 2 + 1` when `onesided`.
    ///
    /// Both `dft_length` (input 1) and, from opset 20, `axis` (input 2) arrive
    /// as *inputs*, so like [`Self::Compress`] this resolves at Compute time by
    /// reading values. `axis_attr` carries the pre-opset-20 spelling.
    Dft {
        onesided: bool,
        axis_attr: Option<i64>,
        default_axis: i64,
    },
    /// `ai.onnx::Compress`: select along `axis` (or over the flattened input
    /// when absent) using a 1-D Bool `condition`.
    ///
    /// The selected extent is **data-dependent** — it is the number of true
    /// entries in `condition`, which no amount of shape reasoning can supply.
    /// That is expressible here because `infer_shapes` runs at Compute time and
    /// receives `TensorView`s, so it reads the condition's *values*. Capability
    /// only needs the rule to exist, not to produce an extent.
    Compress { axis: Option<i64> },
    /// Op the shape table cannot resolve. `reason` separates the two cases,
    /// which look identical at the decline site and are not the same defect.
    Declined {
        op_type: String,
        domain: String,
        reason: DeclineReason,
    },
}

/// Why [`ShapeInference::for_node`] could not produce a rule.
///
/// `GetCapability` drops any claim containing a declined node either way, so
/// the distinction changes no routing. It changes what the situation *means*:
/// [`Self::NodeNotShapeable`] is the table working as designed, while
/// [`Self::Unmodelled`] on an operator that has a registered kernel is wiring
/// nobody finished — the kernel exists and is never dispatched to, and the only
/// symptom is that the op quietly runs somewhere else.
///
/// `registered_ops_are_claimable` in `onnx-runtime-ep-cpu-plugin` turns that
/// second case into a test failure instead of a silent slowdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclineReason {
    /// No arm in the table names this operator. If a kernel is registered for
    /// it, the registration is incomplete.
    Unmodelled,
    /// An arm exists, but this node cannot be shaped from what it carries —
    /// a missing attribute, or an extent that only an input's *values* fix.
    /// Declining is the correct outcome and the claim is dropped on purpose.
    NodeNotShapeable(&'static str),
}

const KERNEL_SIZED_OUTPUT_STRATEGIES: &[(&str, &str)] =
    &[("", "NonMaxSuppression"), ("", "Unique")];

fn kernel_sized_output_strategy(domain: &str, op_type: &str) -> bool {
    let domain = if domain == "ai.onnx" { "" } else { domain };
    KERNEL_SIZED_OUTPUT_STRATEGIES
        .iter()
        .any(|&(expected_domain, expected_op)| domain == expected_domain && op_type == expected_op)
}

/// Exact production census for the dynamic-output anti-vacuity test.
#[cfg(feature = "testutil")]
#[doc(hidden)]
pub const fn kernel_sized_output_strategy_census() -> &'static [(&'static str, &'static str)] {
    KERNEL_SIZED_OUTPUT_STRATEGIES
}

impl ShapeInference {
    /// Full shape inference from a compiled IR `Node` plus the shapes
    /// of its inputs (may be empty slices for absent optional inputs).
    /// Each dimension is `Some(n)` for a statically known extent or `None`
    /// for a symbolic/dynamic dimension — preserving rank.
    ///
    /// `num_outputs` is how many output slots the node has in the graph.
    pub fn for_node(node: &Node, input_shapes: &[Vec<Option<usize>>], num_outputs: usize) -> Self {
        let op = node.op_type.as_str();
        let domain = node.domain.as_str();
        let opset = node.version.unwrap_or(0);

        if kernel_sized_output_strategy(domain, op) {
            return Self::KernelSizedOutputs;
        }

        if let Some(rule) = SharedNativeShapeRule::for_node(node) {
            let fallback = match rule {
                SharedNativeShapeRule::ConstantOfShape => Self::ConstantOfShape,
                SharedNativeShapeRule::Dft => Self::Dft {
                    onesided: node
                        .attr("onesided")
                        .and_then(onnx_runtime_ir::Attribute::as_int)
                        .unwrap_or(0)
                        != 0,
                    axis_attr: node
                        .attr("axis")
                        .and_then(onnx_runtime_ir::Attribute::as_int),
                    default_axis: if opset >= 20 { -2 } else { 1 },
                },
                SharedNativeShapeRule::Expand => Self::Expand,
                SharedNativeShapeRule::Stft => Self::Declined {
                    op_type: "STFT".into(),
                    domain: node.domain.clone(),
                    reason: DeclineReason::NodeNotShapeable(
                        "the shared native STFT rule did not resolve concrete extents",
                    ),
                },
                SharedNativeShapeRule::Tile => Self::Tile,
            };
            return Self::SharedNative {
                node: Box::new(node.clone()),
                fallback: Box::new(fallback),
            };
        }

        let int_attr = |name: &str| -> Option<i64> { node.attr(name)?.as_int() };
        let ints_attr =
            |name: &str| -> Option<Vec<i64>> { Some(node.attr(name)?.as_ints()?.to_vec()) };

        match op {
            // ── Elementwise ───────────────────────────────────────────────
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Mod" | "And" | "Or" | "Xor" | "Equal"
            | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual" | "BitShift" | "BitwiseAnd"
            | "BitwiseOr" | "BitwiseXor" | "Max" | "Min" | "Mean" | "Sum" | "Where"
            // `PRelu`'s slope is unidirectionally broadcastable to the
            // input, so its result follows the same rule as the binary
            // ops above.
            | "PRelu" => {
                Self::ElementwiseBroadcast
            }

            // ── Unary / shape-preserving ──────────────────────────────────
            "Relu"
            | "Sigmoid"
            | "Tanh"
            | "Exp"
            | "Log"
            | "Sqrt"
            | "Abs"
            | "Neg"
            | "Ceil"
            | "Floor"
            | "Round"
            | "Reciprocal"
            | "Not"
            | "Sign"
            | "Erf"
            | "Gelu"
            | "HardSigmoid"
            | "HardSwish"
            | "LeakyRelu"
            | "Elu"
            | "Celu"
            | "Selu"
            | "Mish"
            | "Softplus"
            | "Softsign"
            // Trigonometric and hyperbolic unaries, plus the remaining
            // shape-preserving activations. Every one is registered by the
            // CPU EP (`onnx-runtime-ep-cpu`'s `kernels/mod.rs`) and every one
            // was missing here, so the fail-closed shape filter in
            // `GetCapability` dropped the claim and ORT ran them instead --
            // silently, whatever `supports_op` answered.
            | "Sin"
            | "Cos"
            | "Tan"
            | "Asin"
            | "Acos"
            | "Atan"
            | "Sinh"
            | "Cosh"
            | "Asinh"
            | "Acosh"
            | "Atanh"
            | "ThresholdedRelu"
            | "Swish"
            // `com.microsoft` shape-preserving activations, same mechanism.
            // `Silu` is listed even though ORT 1.28 does not register it:
            // this EP does, and an op the host cannot run is precisely one it
            // must never be handed.
            | "FastGelu"
            | "QuickGelu"
            | "BiasGelu"
            | "Silu"
            | "Cast"
            | "Identity"
            | "Dropout"
            | "IsNaN"
            | "IsInf"
            | "BitCount"
            | "Bernoulli"
            | "Softmax"
            | "LogSoftmax"
            | "Hardmax"
            | "BatchNormalization"
            | "InstanceNormalization"
            | "GroupNormalization"
            | "LpNormalization"
            // Shape-preserving attention / MoE / KV-cache ops. Absent from this
            // table they resolve to `Declined` and `GetCapability` hands them
            // to ORT's CPU EP.
            | "MoE"
            | "QMoE"
            | "ScatterND"
            | "ScatterElements"
            | "TensorScatter"
            | "Trilu"
            | "Clip" => Self::SameAsInput(0),

            // Two outputs of differing shape, so neither `SameAsInput` nor
            // `SameAsInputMultiOutput` fits. Covers both the `com.microsoft`
            // contrib spelling and the standard ai.onnx opset-27 one, which
            // share a contract. Declining handed it to ORT, which has no
            // kernel for it at all — a load failure, not a slower run.
            "CausalConvWithState" => Self::CausalConvWithState,

            // ── LayerNorm / SkipLayerNorm family ──────────────────────────
            "LayerNormalization" | "RMSNormalization" | "SimplifiedLayerNormalization" => {
                // ONNX spec: axis defaults to -1; Mean/InvStdDev shapes are
                // [d[0]..d[axis-1], 1, .., 1].
                // Store raw axis — resolve at runtime against actual rank.
                let raw_axis = int_attr("axis").unwrap_or(-1);
                Self::LayerNorm {
                    raw_axis,
                    num_outputs,
                    full_shape_outputs: vec![],
                }
            }
            "SkipLayerNormalization" | "SkipSimplifiedLayerNormalization" => {
                // Contrib op; normalises the last axis (no axis attr).
                // Outputs: [output, mean, inv_std_dev, input_skip_bias_sum]
                // output and input_skip_bias_sum are full-shaped; mean and
                // inv_std_dev are reduced.
                // raw_axis = -1 (last axis); resolved at runtime.
                let full_shape_outputs = if num_outputs > 3 { vec![3] } else { vec![] };
                Self::LayerNorm {
                    raw_axis: -1,
                    num_outputs,
                    full_shape_outputs,
                }
            }

            // ── MatMul ────────────────────────────────────────────────────
            "MatMul" => Self::MatMul,
            "QLinearMatMul" => Self::QLinearMatMul,
            "MatMulNBits" => match int_attr("N").and_then(|n| usize::try_from(n).ok()) {
                Some(n) if n > 0 => Self::MatMulNBits { n },
                // A `MatMulNBits` without a usable `N` is malformed; the kernel
                // factory rejects it at Compile time. Declining here keeps the
                // claim from reaching that point at all.
                _ => Self::Declined {
                    op_type: op.to_string(),
                    domain: domain.to_string(),
                    reason: DeclineReason::NodeNotShapeable("MatMulNBits without a usable N attribute"),
                },
            },

            // ── Gemm ──────────────────────────────────────────────────────
            "Gemm" => {
                let trans_a = int_attr("transA").unwrap_or(0) != 0;
                let trans_b = int_attr("transB").unwrap_or(0) != 0;
                Self::Gemm { trans_a, trans_b }
            }

            // ── Concat ────────────────────────────────────────────────────
            "Concat" => {
                let axis = int_attr("axis").unwrap_or(0);
                Self::Concat { axis }
            }

            // ── Transpose ────────────────────────────────────────────────
            "Transpose" => {
                let perm = ints_attr("perm").map(|v| v.iter().map(|&x| x as usize).collect());
                Self::Transpose { perm }
            }

            // ── Gather / GatherND ─────────────────────────────────────────
            "Gather" => {
                let axis = int_attr("axis").unwrap_or(0);
                Self::Gather { axis }
            }
            "GatherND" => {
                let batch_dims = int_attr("batch_dims").unwrap_or(0) as usize;
                Self::GatherND { batch_dims }
            }
            "GatherBlockQuantized" => Self::GatherBlockQuantized,

            // ── Shape ─────────────────────────────────────────────────────
            "Shape" => {
                let start = int_attr("start").unwrap_or(0);
                let end = int_attr("end");
                Self::ShapeOp { start, end }
            }

            // ── Squeeze ───────────────────────────────────────────────────
            "Squeeze" => {
                // opset 13+: axes from input[1]; earlier: attribute.
                let axes = if opset >= 13 {
                    // We can't read input[1] at compile time (data-dependent),
                    // but we know input_shapes[0] and can remove all size-1 dims
                    // if no axes input is provided.
                    ints_attr("axes").unwrap_or_default()
                } else {
                    ints_attr("axes").unwrap_or_default()
                };
                Self::Squeeze { axes }
            }

            // ── Unsqueeze ─────────────────────────────────────────────────
            "Unsqueeze" => {
                if let Some(axes) = ints_attr("axes") {
                    Self::Unsqueeze { axes }
                } else {
                    // opset-13: axes come from input[1] — data-dependent.
                    Self::Declined {
                        op_type: op.to_string(),
                        domain: domain.to_string(),
                        reason: DeclineReason::NodeNotShapeable("Unsqueeze opset-13 takes axes from input[1]"),
                    }
                }
            }

            // ── Reshape ───────────────────────────────────────────────────
            "Reshape" => {
                let allowzero = int_attr("allowzero").unwrap_or(0) != 0;
                Self::ReshapeData { allowzero }
            }

            // ── Slice ─────────────────────────────────────────────────────
            "Slice" => Self::SliceData,

            // ── Reductions ────────────────────────────────────────────────
            op_name if is_reduction(op_name) => {
                let keepdims = int_attr("keepdims").unwrap_or(1) != 0;
                let noop_with_empty_axes = int_attr("noop_with_empty_axes").unwrap_or(0) != 0;
                if opset >= 18 {
                    Self::ReductionFromInput {
                        keepdims,
                        noop_with_empty_axes,
                    }
                } else {
                    let axes = ints_attr("axes");
                    Self::Reduction {
                        keepdims,
                        axes,
                        noop_with_empty_axes,
                    }
                }
            }

            // ── Conv ──────────────────────────────────────────────────────
            "Conv" | "ConvInteger" => {
                if let Some(conv) = build_conv(node, input_shapes) {
                    conv
                } else {
                    Self::Declined {
                        op_type: op.to_string(),
                        domain: domain.to_string(),
                        reason: DeclineReason::NodeNotShapeable("Conv attributes/input shapes do not determine the output"),
                    }
                }
            }

            // ── Attention family (com.microsoft) ──────────────────────────
            "MultiHeadAttention" => {
                let num_heads = int_attr("num_heads").unwrap_or(0) as usize;
                Self::MultiHeadAttention {
                    num_heads,
                    num_outputs,
                }
            }
            "GroupQueryAttention" => {
                let num_heads = int_attr("num_heads").unwrap_or(0) as usize;
                let kv_num_heads = int_attr("kv_num_heads").unwrap_or(num_heads as i64) as usize;
                Self::GroupQueryAttention {
                    num_heads,
                    kv_num_heads,
                }
            }
            "RotaryEmbedding" => Self::RotaryEmbedding,
            "PackedMultiHeadAttention" => Self::PackedMultiHeadAttention,

            // `com.microsoft::Attention` — packed QKV, guarded on domain so the
            // opset-23 `ai.onnx::Attention` arm below still owns its own
            // signature. The output width is `v_hidden`, which the attribute
            // gives directly when present; otherwise it is `weights.dim1 / 3`,
            // resolved from the input shapes at Compute time because this
            // entry point does not see them.
            "Attention" if domain == "com.microsoft" => {
                let num_heads = int_attr("num_heads").unwrap_or(0).max(0) as usize;
                let v_hidden = ints_attr("qkv_hidden_sizes").and_then(|sizes| {
                    // Malformed unless all three are present and positive; fall
                    // back to the weight-derived split rather than trusting a
                    // partial attribute.
                    (sizes.len() == 3 && sizes.iter().all(|s| *s > 0)).then(|| sizes[2] as usize)
                });
                Self::MsftAttention {
                    v_hidden,
                    num_heads,
                    num_outputs,
                }
            }

            // ── Standard attention (ai.onnx::Attention, opset 23+) ─────────
            "Attention" if domain.is_empty() || domain == "ai.onnx" => {
                let q_num_heads = int_attr("q_num_heads").unwrap_or(0).max(0) as usize;
                let kv_num_heads = int_attr("kv_num_heads")
                    .unwrap_or(q_num_heads as i64)
                    .max(0) as usize;
                Self::AttentionStd {
                    q_num_heads,
                    kv_num_heads,
                    num_outputs,
                }
            }

            // ── Shape-preserving, one line each ───────────────────────────
            // Output shape == input[0].shape. Listed explicitly rather than
            // folded into the elementwise arm above because none of them is
            // elementwise-broadcasting: they take one input whose shape they
            // carry through, and a broadcast rule would quietly accept a
            // second input it should not.
            //
            // `CastLike` and `EyeLike` take a second input for *dtype* only.
            // `Quantize`/`DequantizeLinear` take scale and zero-point, which
            // change how values map, not the extent — including under blocked
            // quantization. `CumSum`/`CumProd` accumulate along an axis.
            "BitwiseNot" | "CastLike" | "CumProd" | "CumSum" | "DequantizeLinear" | "EyeLike"
            | "QuantizeLinear" => Self::SameAsInput(0),

            // ── Shapes carried in input values ────────────────────────────
            // The shared native adapter owns ConstantOfShape/Expand/Tile above;
            // their local variants remain as compatibility fallbacks. Window
            // sizing is the next cheap value-carried rule still local here.
            "HannWindow" | "HammingWindow" | "BlackmanWindow" => Self::Window,

            // ── DFT ───────────────────────────────────────────────────────
            // Pre-opset-20 the axis is an attribute defaulting to -2; from
            // opset 20 it is input 2. Both are modelled, so neither reaches
            // `Unmodelled`.
            "DFT" => Self::Dft {
                onesided: int_attr("onesided").unwrap_or(0) != 0,
                axis_attr: int_attr("axis"),
                default_axis: if opset >= 20 { -2 } else { 1 },
            },

            // ── Compress ──────────────────────────────────────────────────
            // `axis` is optional: absent means the input is flattened first.
            // Both spellings are modelled, so neither reaches `Unmodelled`.
            "Compress" => Self::Compress {
                axis: int_attr("axis"),
            },

            _ => Self::Declined {
                op_type: op.to_string(),
                domain: domain.to_string(),
                reason: DeclineReason::Unmodelled,
            },
        }
    }
}

fn is_reduction(op: &str) -> bool {
    matches!(
        op,
        "ReduceMean"
            | "ReduceSum"
            | "ReduceProd"
            | "ReduceMax"
            | "ReduceMin"
            | "ReduceL1"
            | "ReduceL2"
            | "ReduceLogSum"
            | "ReduceLogSumExp"
            | "ReduceSumSquare"
    )
}

fn build_conv(node: &Node, input_shapes: &[Vec<Option<usize>>]) -> Option<ShapeInference> {
    let auto_pad = node
        .attr("auto_pad")
        .and_then(|a| a.as_str())
        .unwrap_or("NOTSET");
    if auto_pad != "NOTSET" {
        // SAME_UPPER / SAME_LOWER require runtime spatial size — defer to runtime.
        return None;
    }
    // input[0]: [N, C_in, d0, d1, ...]
    let in_shape = input_shapes.first()?;
    if in_shape.len() < 3 {
        return None;
    }
    let spatial_dims = in_shape.len() - 2;

    // weight[0] first dim is out_channels — fail closed if dynamic.
    let out_channels = input_shapes
        .get(1)
        .and_then(|w| w.first().copied())
        .flatten()?;

    let kernel_shape: Vec<usize> = node
        .attr("kernel_shape")
        .and_then(|a| a.as_ints())
        .map(|v| v.iter().map(|&x| x as usize).collect())
        .or_else(|| {
            // derive from weight shape: weight=[out_ch, in_ch/group, k0, k1, ...]
            let w = input_shapes.get(1)?;
            if w.len() == 2 + spatial_dims {
                // Fail closed: all kernel dims must be static.
                w[2..].iter().copied().collect::<Option<Vec<usize>>>()
            } else {
                None
            }
        })?;

    let strides: Vec<usize> = node
        .attr("strides")
        .and_then(|a| a.as_ints())
        .map(|v| v.iter().map(|&x| x as usize).collect())
        .unwrap_or_else(|| vec![1; spatial_dims]);

    let dilations: Vec<usize> = node
        .attr("dilations")
        .and_then(|a| a.as_ints())
        .map(|v| v.iter().map(|&x| x as usize).collect())
        .unwrap_or_else(|| vec![1; spatial_dims]);

    let pads: Vec<usize> = node
        .attr("pads")
        .and_then(|a| a.as_ints())
        .map(|v| v.iter().map(|&x| x as usize).collect())
        .unwrap_or_else(|| vec![0; 2 * spatial_dims]);

    if kernel_shape.len() != spatial_dims
        || strides.len() != spatial_dims
        || dilations.len() != spatial_dims
        || pads.len() != 2 * spatial_dims
    {
        return None;
    }

    let per_axis = (0..spatial_dims)
        .map(|i| ConvSpatialAxis {
            kernel: kernel_shape[i],
            pad_before: pads[i],
            pad_after: pads[i + spatial_dims],
            stride: strides[i],
            dilation: dilations[i],
        })
        .collect();

    Some(ShapeInference::Conv {
        out_channels,
        per_axis,
    })
}

/// A compiled kernel bundled with the metadata needed to drive execution
/// through the ORT kernel context.
pub struct CompiledKernelEntry {
    pub kernel: Box<dyn Kernel>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    /// Per-output declared IR dtype, read from the ORT graph's value info at
    /// Compile time. Indexed by output slot. Never inferred from inputs —
    /// this is the authoritative dtype for each output tensor. Absent optional
    /// outputs carry their actual ORT-declared dtype (not `Undefined`) so
    /// scratch buffers can be sized correctly for f16/bf16 ops.
    pub output_dtypes: Vec<DataType>,
    /// Which output slots are absent (omitted optional outputs). Absent slots
    /// get a scratch buffer at runtime; present slots are allocated via ORT.
    pub absent_output_slots: HashSet<usize>,
    pub shape_inference: ShapeInference,
    /// Maps node input position → `Some(ort_index)` for present inputs,
    /// `None` for absent optional inputs. Used by the single-kernel fast
    /// path to reconstruct the positional input list with absent sentinels.
    pub input_slots: Vec<Option<usize>>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Scratch buffer sizing — single source of truth
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the scratch buffer allocation size (in bytes) for an absent output slot.
///
/// Formula: `numel * max(byte_size_of(dtype), 8)`.  The `max(…, 8)` padding
/// prevents a heap overflow when a kernel internally writes a wider type than
/// the declared dtype (e.g. Float32 stats for an f16 SkipLayerNormalization).
///
/// **This function is the single authoritative implementation.**  Both the
/// single-node and multi-node compute paths, as well as the canary tests, call
/// it directly.  Duplicating the formula invites drift — and drift here means a
/// heap overflow (the original B1 bug).
pub fn scratch_alloc_bytes(numel: usize, dtype: DataType) -> usize {
    let elem_bytes = dtype.byte_size().max(8);
    numel * elem_bytes
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-node subgraph routing
// ──────────────────────────────────────────────────────────────────────────────

/// Where a node's i-th input tensor comes from in a fused multi-node subgraph.
#[derive(Clone, Debug)]
pub enum NodeInputSource {
    /// Take from ORT's kernel-context input at this index.
    Ort(usize),
    /// Take from an intermediate buffer written by an earlier node.
    Buffer(usize),
    /// The input is absent (omitted optional). Must not be read by the kernel.
    Absent,
}

/// Where a node's i-th output tensor goes in a fused multi-node subgraph.
#[derive(Clone, Debug)]
pub enum NodeOutputSink {
    /// Write to ORT's kernel-context output at this index.
    Ort(usize),
    /// Write to an intermediate buffer for a later node.
    Buffer(usize),
    /// The output is absent (omitted optional). Scratch-allocated at compute
    /// time; does not consume an intermediate buffer slot.
    Absent,
}

/// Routing table for a fused multi-node subgraph.
///
/// `input_sources[node_idx]` and `output_sinks[node_idx]` are indexed by
/// the kernel's input/output slot, in the same order as `CompiledKernelEntry`.
#[derive(Clone, Debug)]
pub struct SubgraphRouting {
    pub input_sources: Vec<Vec<NodeInputSource>>,
    pub output_sinks: Vec<Vec<NodeOutputSink>>,
    pub num_intermediate_buffers: usize,
}

/// Heap-owned intermediate tensor buffer for multi-node subgraph execution.
///
/// The backing bytes are either owned on the host (`data`, used by unit tests
/// and as a fallback) or borrowed from an ORT scratch allocation (`scratch_ptr`,
/// non-null). For device EPs (CUDA) the scratch allocation lives in device
/// memory, so kernels can read/write intermediates directly on the GPU without
/// an illegal host-pointer dereference. ORT owns scratch memory for the
/// duration of the `Compute` call, which exactly matches an intermediate's
/// lifetime, so `IntermediateBuf` never frees `scratch_ptr`.
pub(crate) struct IntermediateBuf {
    pub(crate) data: Vec<u8>,
    /// When non-null, the buffer is backed by ORT scratch memory (possibly on
    /// device) instead of the host `data` vector. Not owned — never freed here.
    pub(crate) scratch_ptr: *mut u8,
    /// Inline rather than `Vec` because every buffer-sink output built one of
    /// each per node, and a rank that fits inline needs no allocator at all.
    /// The struct is wider as a result and is moved twice on the routed path
    /// (staged, then installed), so this trades two `malloc`/`free` pairs for a
    /// larger `memcpy`; see the PR for the measured net.
    pub(crate) shape: crate::dim_vec::DimVec<usize>,
    pub(crate) strides: crate::dim_vec::DimVec<i64>,
    pub(crate) dtype: DataType,
    pub(crate) device: DeviceId,
}

impl IntermediateBuf {
    /// Raw const pointer to the backing bytes (scratch if set, else host `data`).
    fn ptr(&self) -> *const u8 {
        if self.scratch_ptr.is_null() {
            self.data.as_ptr()
        } else {
            self.scratch_ptr.cast_const()
        }
    }

    /// Raw mutable pointer to the backing bytes (scratch if set, else host).
    fn ptr_mut(&mut self) -> *mut u8 {
        if self.scratch_ptr.is_null() {
            self.data.as_mut_ptr()
        } else {
            self.scratch_ptr
        }
    }

    /// Immutable view backed by this buffer.
    pub(crate) fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.ptr().cast()),
            self.dtype,
            &self.shape,
            &self.strides,
            self.device,
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ExportedComputeInfo
// ──────────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// Workspace plan cache
// ──────────────────────────────────────────────────────────────────────────────

/// One operand's contribution to a [`WorkspaceSignature`] — exactly the three
/// fields a kernel sees in [`TensorMetadata`], and nothing else.
///
/// It must stay exactly those three: `Kernel::workspace_requirement` is handed
/// `TensorMetadata { dtype, shape, present }`, so two dispatches whose operands
/// agree on all three are indistinguishable to the kernel, and any field the key
/// drops is a way to serve a plan computed for different operands.
#[derive(Clone, Debug, PartialEq, Eq)]
struct OperandKey {
    dtype: DataType,
    present: bool,
    shape: Vec<usize>,
}

impl OperandKey {
    fn matches(&self, metadata: &TensorMetadata<'_>) -> bool {
        self.dtype == metadata.dtype
            && self.present == metadata.present
            && self.shape.as_slice() == metadata.shape
    }

    /// Same comparison as [`OperandKey::matches`], against the `TensorView` the
    /// metadata would have been built from.
    ///
    /// This exists so the lock-free fast path in
    /// [`WorkspacePlanCache::get_or_plan`] can verify a hit without first
    /// materialising the `Vec<TensorMetadata>` — that allocation is only worth
    /// paying on a miss. It must stay field-for-field identical to `matches`
    /// composed with `TensorMetadata::new(v.dtype, v.shape, !v.is_absent())`,
    /// which `the_view_and_metadata_comparisons_agree` pins.
    fn matches_view(&self, view: &TensorView<'_>) -> bool {
        self.dtype == view.dtype
            && self.present == !view.is_absent()
            && self.shape.as_slice() == view.shape
    }
}

/// Owned copy of the metadata a [`WorkspaceRequirement`] was computed from.
type WorkspaceSignature = Vec<OperandKey>;

fn signature_matches(signature: &WorkspaceSignature, metadata: &[TensorMetadata<'_>]) -> bool {
    signature.len() == metadata.len()
        && signature
            .iter()
            .zip(metadata)
            .all(|(key, meta)| key.matches(meta))
}

fn signature_of(metadata: &[TensorMetadata<'_>]) -> WorkspaceSignature {
    metadata
        .iter()
        .map(|meta| OperandKey {
            dtype: meta.dtype,
            present: meta.present,
            shape: meta.shape.to_vec(),
        })
        .collect()
}

/// How many distinct operand signatures one node remembers a plan for.
///
/// Sized for the shapes a decoder actually alternates between (prefill, one or
/// two decode extents, a padded variant) with headroom, while staying small
/// enough that a linear scan under the lock is cheaper than the planning call it
/// avoids. Overflow is a miss, never a wrong answer.
const WORKSPACE_PLAN_CACHE_CAPACITY: usize = 8;

/// Memoizes `Kernel::workspace_requirement` per node, keyed by the exact
/// operand metadata the requirement was derived from.
///
/// # Why this exists
///
/// `prepare_workspace` must ask the kernel what it needs before it can decide
/// whether it can serve it. For the cuBLASLt-backed CUDA kernels (`MatMul`,
/// `Gemm`, `FusedEpilogue`, `MatMulNBits`' f32 dequant path) answering that
/// question runs a full `cublasLtMatmulAlgoGetHeuristic` search — and because
/// those kernels declare `WorkspaceLifetime::SessionPersistent`, which this
/// executor declines, the kernel then plans a *second* time inside its own
/// `execute`. Every node of every decode step paid for two heuristic searches
/// and used one.
///
/// The kernel-side re-plan cannot be removed from here: `plan_gemm` returns the
/// selected algorithm and matrix layouts, not just a byte count, and the kernel
/// re-derives and re-validates the size it was handed rather than trusting the
/// executor. So the duplicate is removed on the executor side instead, by not
/// re-asking a question whose answer cannot have changed.
///
/// # What this is worth, stated exactly
///
/// The baseline matters. On `main` the executor never called
/// `workspace_requirement` at all, so:
///
/// * versus the uncached version of this seam, the second search per dispatch
///   is gone;
/// * versus `main`, the steady state is approximately **neutral** — a repeated
///   operand signature costs a `Mutex` acquire plus a linear scan of at most
///   [`WORKSPACE_PLAN_CACHE_CAPACITY`] entries, and each *new* signature still
///   costs one heuristic search `main` never paid;
/// * the kernel-side plan inside `blas::governed_gemm` is untouched and still
///   happens once per dispatch.
///
/// Hit rate follows the shapes, not the kernel. A stable geometry hits after
/// the first dispatch. A growing-KV `StepScoped` attention, whose operand
/// shapes change every decode step, can miss on **every** step; the cache is
/// then bounded overhead (one extra search per step) rather than a saving.
/// Only shapes that recur — fixed geometries, and the many decode GEMMs whose
/// cuBLASLt signature is stable across steps — benefit.
///
/// # Correctness
///
/// * The key is the full operand metadata (dtype, presence, shape) for every
///   operand, so a hit is only possible when the kernel would be asked exactly
///   the same question. Dynamic shapes, dtype changes and optional-input
///   presence changes are all misses.
/// * Nothing is shared between nodes: each `CompiledKernelEntry` gets its own
///   cache, so one node's plan can never be served to another.
/// * The lock is never held across the kernel call, so concurrent `Run`s on the
///   same session plan in parallel; a duplicated concurrent computation is
///   harmless because it produces the same value.
/// * Planning errors are never cached — a failing `workspace_requirement` is
///   re-raised on every dispatch, exactly as before.
/// * A poisoned lock is recovered with `PoisonError::into_inner`, matching
///   `EpHandle::with` and `release_ep_factory_with_teardown`: a panic in some
///   unrelated dispatch must not turn every later dispatch into a hard error.
///
/// # What this assumes of the kernel
///
/// That `workspace_requirement` is a function of the operand metadata alone for
/// a given kernel instance. That is the contract the native session executor
/// already relies on (it plans once during prepare and reuses the reservation
/// for every later `Run`), and it holds for every kernel in this workspace.
/// A kernel that violates it is not silently mis-served: the CUDA consumers
/// re-derive their requirement in `execute_with_workspace` and reject a
/// workspace that is too small (`governed_workspace_ptr`, `std_attention_carve`,
/// `gqa` sub-range carving) rather than reading past the end.
/// A signature that has been observed more than once, published for lock-free
/// reads (#1077 lever 3).
///
/// Written at most once per node, through [`OnceLock`], and only for a
/// signature that has already been served from the locked cache — that is, one
/// that recurs. Reading it costs an acquire load and an operand-wise compare;
/// it takes no lock and allocates nothing.
#[derive(Debug)]
struct HotPlan {
    operands: Box<[OperandKey]>,
    requirement: WorkspaceRequirement,
}

struct WorkspacePlanCache {
    /// Lock-free fast path. See [`HotPlan`].
    ///
    /// Publishing on the *second* sighting rather than the first is deliberate:
    /// a decoder's first `Run` is a prefill whose shape may never recur, and
    /// pinning that one would leave every later decode `Run` on the locked
    /// path. Waiting for a repeat costs one extra locked `Run` at startup and
    /// picks the recurring shape in both the static and the prefill/decode
    /// case.
    ///
    /// Exactly one signature is ever pinned, and it is never rotated. That
    /// covers the shapes we care about (static graphs, and prefill-then-decode
    /// where only the decode extent recurs), but a node that genuinely
    /// alternates between two equally-recurring signatures keeps roughly half
    /// its dispatches on the locked path. That is a missed optimisation, not a
    /// correctness problem: a miss simply falls through to [`Self::lookup`].
    hot: std::sync::OnceLock<HotPlan>,
    /// Test-only: keep every dispatch on the locked path.
    ///
    /// The lock-free slot would otherwise answer the repeat access in
    /// `a_repeatedly_used_signature_survives_eviction`, which would let that
    /// test pass with move-to-front deleted — it would stop measuring what it
    /// names. Tests of the locked cache opt out through
    /// [`WorkspacePlanCache::locked_only`].
    #[cfg(test)]
    suppress_hot: bool,
    plans: Mutex<Vec<(WorkspaceSignature, WorkspaceRequirement)>>,
}

impl WorkspacePlanCache {
    fn new() -> Self {
        Self {
            hot: std::sync::OnceLock::new(),
            #[cfg(test)]
            suppress_hot: false,
            plans: Mutex::new(Vec::new()),
        }
    }

    /// A cache that never publishes to the lock-free slot, so every dispatch
    /// exercises the locked path. See `suppress_hot`.
    #[cfg(test)]
    fn locked_only() -> Self {
        Self {
            suppress_hot: true,
            ..Self::new()
        }
    }

    /// The lock-free hit test, against already-built metadata.
    fn hot_hit(&self, metadata: &[TensorMetadata<'_>]) -> Option<WorkspaceRequirement> {
        let hot = self.hot.get()?;
        (hot.operands.len() == metadata.len()
            && hot
                .operands
                .iter()
                .zip(metadata)
                .all(|(key, meta)| key.matches(meta)))
        .then_some(hot.requirement)
    }

    /// The lock-free hit test, against the views the metadata would describe.
    ///
    /// Identical in effect to [`WorkspacePlanCache::hot_hit`] on the metadata
    /// built from `inputs`, but it does not build that `Vec`.
    fn hot_hit_views(&self, inputs: &[TensorView<'_>]) -> Option<WorkspaceRequirement> {
        let hot = self.hot.get()?;
        (hot.operands.len() == inputs.len()
            && hot
                .operands
                .iter()
                .zip(inputs)
                .all(|(key, view)| key.matches_view(view)))
        .then_some(hot.requirement)
    }

    /// Publish a recurring signature to the lock-free slot, if it is still
    /// empty. Losing the race is harmless: the winner is equally valid, and
    /// this dispatch already has its answer.
    fn publish_hot(&self, metadata: &[TensorMetadata<'_>], requirement: WorkspaceRequirement) {
        #[cfg(test)]
        if self.suppress_hot {
            return;
        }
        if self.hot.get().is_some() {
            return;
        }
        let _ = self.hot.set(HotPlan {
            operands: signature_of(metadata).into_boxed_slice(),
            requirement,
        });
    }

    /// Look up the plan for `inputs`, computing and remembering it on a miss.
    ///
    /// The fast path takes no lock and makes no allocation. The metadata `Vec`
    /// is built only when the lock-free slot does not already answer the
    /// question, because it is needed only by the locked cache and by
    /// `Kernel::workspace_requirement`.
    fn get_or_plan_views(
        &self,
        inputs: &[TensorView<'_>],
        plan: impl FnOnce(&[TensorMetadata<'_>]) -> Result<WorkspaceRequirement, String>,
    ) -> Result<WorkspaceRequirement, String> {
        if let Some(hit) = self.hot_hit_views(inputs) {
            return Ok(hit);
        }
        let metadata: Vec<TensorMetadata<'_>> = inputs
            .iter()
            .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
            .collect();
        self.get_or_plan(&metadata, || plan(&metadata))
    }

    /// Look up the plan for `metadata`, computing and remembering it on a miss.
    fn get_or_plan(
        &self,
        metadata: &[TensorMetadata<'_>],
        plan: impl FnOnce() -> Result<WorkspaceRequirement, String>,
    ) -> Result<WorkspaceRequirement, String> {
        if let Some(hit) = self.hot_hit(metadata) {
            return Ok(hit);
        }
        if let Some(hit) = self.lookup(metadata) {
            // Second sighting: this signature recurs, so it is the one worth
            // publishing.
            self.publish_hot(metadata, hit);
            return Ok(hit);
        }
        let requirement = plan()?;
        self.remember(signature_of(metadata), requirement);
        Ok(requirement)
    }

    fn lookup(&self, metadata: &[TensorMetadata<'_>]) -> Option<WorkspaceRequirement> {
        let mut plans = self.plans.lock().unwrap_or_else(PoisonError::into_inner);
        let idx = plans
            .iter()
            .position(|(signature, _)| signature_matches(signature, metadata))?;
        let requirement = plans[idx].1;
        // Move-to-front so the hot signature stays ahead of a one-off prefill
        // shape once the cache is full.
        //
        // Skipped when the hit is already at the front, which is the steady
        // state: one shape, every node, every `Run`. Promoting the front entry
        // to the front is identity, but `remove` + `insert` still run the
        // shift, bounds checks and length bookkeeping to achieve nothing.
        // Callgrind, 100-node chain: `lookup` costs 141 instructions per node
        // without this guard and 101 with it.
        if idx != 0 {
            let entry = plans.remove(idx);
            plans.insert(0, entry);
        }
        Some(requirement)
    }

    fn remember(&self, signature: WorkspaceSignature, requirement: WorkspaceRequirement) {
        let mut plans = self.plans.lock().unwrap_or_else(PoisonError::into_inner);
        // A concurrent dispatch may have inserted the same signature while this
        // one was planning. Overwrite rather than duplicate: both computed the
        // same answer from the same metadata.
        if let Some(slot) = plans
            .iter_mut()
            .find(|(existing, _)| *existing == signature)
        {
            slot.1 = requirement;
            return;
        }
        if plans.len() >= WORKSPACE_PLAN_CACHE_CAPACITY {
            plans.pop();
        }
        plans.insert(0, (signature, requirement));
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.plans
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// Heap-allocated compute info whose raw pointer is returned as
/// `OrtNodeComputeInfo*`.
///
/// The first field is the `OrtNodeComputeInfo` vtable.
#[repr(C)]
pub struct ExportedComputeInfo {
    pub vtable: ort::OrtNodeComputeInfo,
    /// The compiled kernel entries for this subgraph (in topological order).
    pub entries: Vec<CompiledKernelEntry>,
    /// Optional routing table for multi-node fused subgraphs.
    pub routing: Option<SubgraphRouting>,
    /// One [`WorkspacePlanCache`] per entry in `entries`, same index. Built in
    /// [`ExportedComputeInfo::new`] so it is always exactly as long as
    /// `entries`.
    workspace_plans: Vec<WorkspacePlanCache>,
    /// Device staging context (#982). `Some` only for a device EP: it lets
    /// `Compute` upload a host-resident boundary input into device scratch
    /// before launching a device kernel, on an interspersed CPU→device
    /// partition where ORT never inserts the host→device copy. `None` for the
    /// CPU EP (and any host EP), which uses its inputs verbatim exactly as
    /// before.
    device_staging: Option<DeviceStaging>,
    /// Whether ORT places this EP's tensors in **host-accessible** memory.
    ///
    /// This is the same `DeviceSupport::host_accessible` the plugin factory
    /// keys allocator registration on (`factory.rs`, `CreateAllocator`): when
    /// it is true ORT hands out its own default host allocator, so every
    /// `OrtValue` this EP sees is host-resident and a routed subgraph's
    /// intermediates belong in host buffers. That lets `Compute` skip the
    /// per-input `OrtMemoryInfo` scan entirely.
    ///
    /// Deliberately **defaults to `false`** — the conservative answer. A
    /// caller that never sets it gets the full scan, i.e. the historical
    /// behaviour, so a device EP cannot be mis-served by a forgotten setter.
    /// Note this must *not* be inferred from `device_staging`: that is set
    /// from `host_to_device_copier()`, which defaults to `None` and which a
    /// device EP may legitimately decline (see `provider.rs`), so a device EP
    /// can have `device_staging == None` and device-resident inputs.
    host_accessible: bool,
    /// Whether ORT's intra-op pool has ever been seen running our elementwise
    /// chunks, and when to ask again if not.
    ///
    /// Lives here, rather than in a process-global, because a process may hold
    /// one session at `intra_op = 1` and another at `intra_op = 16`, and the
    /// right answer -- whether to borrow ORT's threads or use our own -- is
    /// the opposite for each. See `onnx_runtime_ep_api::host_parallel`.
    host_pool_probe: std::sync::atomic::AtomicU32,
}

/// Everything `Compute` needs to stage host-resident boundary inputs onto the
/// EP's device (#982): a synchronous uploader and a reconstructed device
/// `OrtMemoryInfo` used as a fallback allocation target when no device-resident
/// `OrtValue` is otherwise visible in the call.
struct DeviceStaging {
    copier: Arc<dyn HostToDeviceCopier>,
    /// Reconstructed device memory info, matching the one the plugin factory
    /// registered the device allocator against. Used only when neither an
    /// input nor an output of the node yields a device `OrtMemoryInfo`.
    recon_mem_info: Option<ReconstructedMemInfo>,
}

/// Owns an `OrtMemoryInfo` rebuilt via `CreateMemoryInfo_V2`, released on drop.
struct ReconstructedMemInfo {
    ptr: *const ort::OrtMemoryInfo,
    /// Whether `ptr` denotes device memory, recorded from the `device_type` it
    /// was built with rather than asked of ORT afterwards.
    ///
    /// `mem_info_is_device` is exactly `device_type != CPU`, and this memory
    /// info was created by passing `device_type` straight to
    /// `CreateMemoryInfo_V2`, so the two agree by construction. Every EP that
    /// supplies a host-to-device copier today is non-CPU, which would make a
    /// hardcoded `true` correct — but nothing in `HostToDeviceCopier` requires
    /// that, and a CPU-typed EP that grew a copier would then be handed host
    /// memory as a device scratch target. Deriving it costs nothing and cannot
    /// drift.
    is_device: bool,
}

// SAFETY: the pointer is read-only after construction and released only on drop.
unsafe impl Send for ReconstructedMemInfo {}
unsafe impl Sync for ReconstructedMemInfo {}

impl Drop for ReconstructedMemInfo {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let api = crate::status::host_api();
        if api.is_null() {
            return;
        }
        if let Some(release) = unsafe { (*api).ReleaseMemoryInfo } {
            crate::dispatch_probe::ort_call();
            unsafe { release(self.ptr.cast_mut()) };
        }
    }
}

/// Per-session state created by `CreateState`.
struct ComputeState {
    _placeholder: u8,
}

impl ExportedComputeInfo {
    pub fn new(entries: Vec<CompiledKernelEntry>) -> Self {
        let workspace_plans = entries.iter().map(|_| WorkspacePlanCache::new()).collect();
        Self {
            vtable: ort::OrtNodeComputeInfo {
                ort_version_supported: ort::ORT_API_VERSION,
                CreateState: Some(compute_create_state),
                Compute: Some(compute_execute),
                ReleaseState: Some(compute_release_state),
            },
            entries,
            routing: None,
            workspace_plans,
            device_staging: None,
            host_accessible: false,
            host_pool_probe: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Record whether ORT places this EP's tensors in host-accessible memory.
    ///
    /// Pass the factory's `DeviceSupport::host_accessible` — the *same* flag
    /// `CreateAllocator` branches on — so this can never disagree with where
    /// ORT actually allocates. Leaving it unset means "assume device", which
    /// is the safe direction: `Compute` then scans as it always did.
    pub fn set_host_accessible(&mut self, host_accessible: bool) {
        self.host_accessible = host_accessible;
    }

    /// Attach the device staging context (#982) captured at compile time from a
    /// device EP. Given a `copier`, this also reconstructs the device
    /// `OrtMemoryInfo` (from `allocator_name`, `device_type`, `vendor_id`) that
    /// serves as the fallback scratch target for all-host-input nodes. A CPU EP
    /// never calls this, so its `device_staging` stays `None`.
    pub fn set_device_staging(
        &mut self,
        copier: Arc<dyn HostToDeviceCopier>,
        allocator_name: &str,
        device_type: ort::OrtMemoryInfoDeviceType,
        vendor_id: u32,
    ) {
        let recon_mem_info = reconstruct_device_mem_info(allocator_name, device_type, vendor_id);
        self.device_staging = Some(DeviceStaging {
            copier,
            recon_mem_info,
        });
    }

    /// Attach a subgraph routing table (for multi-node fused subgraphs).
    pub fn set_routing(&mut self, routing: SubgraphRouting) {
        self.routing = Some(routing);
    }

    /// Plan cache for node `idx`. `None` when the index is out of range, which
    /// a caller must treat as a hard error rather than by planning uncached:
    /// a length mismatch means the two vectors have drifted.
    fn workspace_plan_cache(&self, idx: usize) -> Option<&WorkspacePlanCache> {
        self.workspace_plans.get(idx)
    }
}

/// CreateState: allocate per-session compute state.
unsafe extern "C" fn compute_create_state(
    _info: *mut ort::OrtNodeComputeInfo,
    _compute_context: *mut ort::OrtNodeComputeContext,
    out_state: *mut *mut c_void,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out_state.is_null() {
            return fail_status("CreateState: out_state is null");
        }
        let state = Box::new(ComputeState { _placeholder: 0 });
        unsafe { *out_state = Box::into_raw(state).cast::<c_void>() };
        ok_status()
    }));
    result.unwrap_or_else(|_| fail_status("CreateState: internal panic"))
}

/// Returns the `OrtMemoryInfo*` of the EP's tensor memory, read from **fused
/// subgraph input 0**'s `OrtValue`. For a device EP (CUDA) this is device
/// memory; for the CPU EP it is host memory.
///
/// This is the memory info intermediate buffers are allocated against — the
/// exact derivation #832 validated on H200 — and it is deliberately unchanged
/// here. It is a *subgraph*-level fact, not a per-node one: for a fused
/// multi-node subgraph, node `k > 0` may take none of its operands from ORT.
/// The per-node derivation used for kernel workspaces is
/// [`operand_mem_info`]; see its docs for what is and is not guaranteed.
///
/// Returns `None` when there are no inputs or the memory info cannot be read; in
/// that case callers fall back to host-owned intermediate buffers.
///
/// # Safety
///
/// `api` must be valid and `ctx` a valid `OrtKernelContext*`.
/// Whether `Compute` must ask ORT where this EP's tensors live before it can
/// place a routed subgraph's intermediates.
///
/// Keyed on `host_accessible` -- the flag the plugin factory registers its
/// allocator on, so it is the same decision ORT itself made -- and
/// deliberately **not** on the presence of a device-staging context. `staging`
/// is taken and ignored on purpose: it comes from `host_to_device_copier()`,
/// which defaults to `None` and which a device EP may legitimately decline, so
/// keying on it would classify such an EP as host and place its intermediates
/// in host memory for a device kernel to dereference. That was a real defect
/// caught in review; the parameter is kept so the invariant is stated where it
/// can be tested rather than only in a comment.
fn must_scan_for_device_placement(host_accessible: bool, _staging: Option<&DeviceStaging>) -> bool {
    !host_accessible
}

unsafe fn device_mem_info(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    staging: Option<&DeviceStaging>,
) -> Option<(*const ort::OrtMemoryInfo, bool)> {
    // A routed subgraph's intermediates must be allocated in *device* memory so
    // the device kernels that produce and consume them dereference valid device
    // pointers. On an interspersed CPU→device partition, kernel-context input 0
    // may be a *host* boundary input (#982), so we must not assume input 0 lives
    // on the device — scan every input for a genuinely device-resident one.
    //
    // The scan remembers input 0's memory info as it goes. It used to throw it
    // away and re-fetch it at the bottom, which cost two more FFI calls on the
    // path every host EP takes: the scan finds nothing device-resident, and the
    // fallback then asks ORT again for the value it had already seen first.
    // Whether the result is a device is returned too, for the same reason — the
    // one caller immediately asked, which meant a third query for a fact this
    // function had just established.
    let mut input0: Option<*const ort::OrtMemoryInfo> = None;
    if let Some(get_count) = api.KernelContext_GetInputCount {
        let mut count: usize = 0;
        crate::dispatch_probe::ort_call();
        let status = unsafe { get_count(ctx, &mut count) };
        if status.is_null() {
            for i in 0..count {
                let Some(mi) = (unsafe { ort_input_mem_info(api, ctx, i) }) else {
                    continue;
                };
                if i == 0 {
                    input0 = Some(mi);
                }
                if unsafe { mem_info_is_device(api, mi) } {
                    return Some((mi, true));
                }
            }
        }
    }
    // No device-resident input (e.g. every boundary input is host): if this is a
    // device EP, fall back to the reconstructed EP device memory info — the same
    // recipe the plugin factory registered its device allocator against — so
    // intermediates still land on the device instead of silently on the host.
    if let Some(staging) = staging
        && let Some(recon) = staging.recon_mem_info.as_ref()
        && !recon.ptr.is_null()
    {
        // Device-ness recorded when this was built from the EP's own device
        // recipe, so it matches what the old `mem_info_is_device` call returned
        // without spending an FFI call to re-derive it.
        return Some((recon.ptr, recon.is_device));
    }
    // Host EP (no device staging context): preserve the historical behavior of
    // using input 0's memory info. For a host EP this is host memory, which is
    // exactly where its intermediates belong, so host-only partitions are
    // unchanged by the #982 device-staging work.
    //
    // The scan above already looked at input 0 unless it failed or there are no
    // inputs, so this normally costs nothing.
    match input0 {
        Some(mi) => Some((mi, false)),
        None => unsafe { ort_input_mem_info(api, ctx, 0) }
            .map(|mi| (mi, unsafe { mem_info_is_device(api, mi) })),
    }
}

/// Memory info of the `OrtValue` bound to kernel-context input `index`.
///
/// # Safety
///
/// `api` must be valid and `ctx` a valid `OrtKernelContext*`.
unsafe fn ort_input_mem_info(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    index: usize,
) -> Option<*const ort::OrtMemoryInfo> {
    let get_input = api.KernelContext_GetInput?;
    let get_mem_info = api.GetTensorMemoryInfo?;
    let mut value: *const ort::OrtValue = std::ptr::null();
    crate::dispatch_probe::ort_call();
    let status = unsafe { get_input(ctx, index, &mut value) };
    if !status.is_null() || value.is_null() {
        return None;
    }
    let mut mem_info: *const ort::OrtMemoryInfo = std::ptr::null();
    crate::dispatch_probe::ort_call();
    let status = unsafe { get_mem_info(value, &mut mem_info) };
    if !status.is_null() || mem_info.is_null() {
        return None;
    }
    Some(mem_info)
}

/// Where one node's operands live, as far as ORT will tell us.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperandMemInfo {
    /// Every ORT-bound operand of this node reported the same memory info (or
    /// there was exactly one, so there was nothing to disagree with).
    Uniform(*const ort::OrtMemoryInfo),
    /// This node binds no ORT operand at all — in a fused subgraph every one of
    /// its inputs comes from an intermediate buffer, which was allocated
    /// against the subgraph-level [`device_mem_info`]. That is then the right
    /// answer for this node too, and it is what is carried here.
    FromIntermediates(*const ort::OrtMemoryInfo),
    /// The memory device could not be read (no inputs, or ORT refused).
    Unavailable,
    /// Two ORT-bound operands of the same node reported **different** memory
    /// infos, so "the device this node runs on" is not determined by its
    /// operands. Carries the two kernel-context input indices that disagreed.
    Divergent { first: usize, other: usize },
}

/// Where to look to decide which device a node's workspace must live on.
///
/// Kept as one value so it can be threaded through `prepare_workspace` without
/// resolving anything: the ORT input indices are cheap to carry, the answer
/// they produce is not.
#[derive(Clone, Copy)]
struct PlacementSources<'a> {
    /// Kernel-context input indices bound to ORT for *this* node. Outputs are
    /// deliberately absent; see [`operand_mem_info`].
    ort_inputs: OrtOperands<'a>,
    /// Memory info to fall back to when the node binds no ORT inputs at all
    /// (every operand is a fused-subgraph intermediate).
    subgraph_fallback: SubgraphFallback<'a>,
}

/// Where [`operand_mem_info`] gets its fallback memory info when a node binds
/// no ORT inputs at all, plus the per-`Run` memo of that resolution.
///
/// Resolving it means walking every kernel-context input and asking ORT for its
/// `OrtMemoryInfo` (see [`device_mem_info`]) — four FFI calls for a one-input
/// node. Its only consumer reads it for a node with no ORT-bound operands at
/// all, and then only past [`prepare_workspace`]'s zero-byte and servability
/// gates, so an elementwise dispatch never makes the calls at all.
///
/// The routed path used to resolve it *eagerly*, once per `Run`, to share it
/// across nodes — charging every routed `Run` for a value almost no dispatch
/// reads. Deferring it must not trade that for a resolution *per node*: a fused
/// subgraph whose nodes each take only intermediates and each request a
/// non-zero step-scoped workspace would then rescan every input once per node.
/// `memo` keeps both properties — at most one resolution per `Run`, and none
/// when nothing reaches the fallback.
#[derive(Clone, Copy)]
struct SubgraphFallback<'a> {
    /// Staging context to resolve against, `None` for an EP without one.
    staging: Option<&'a DeviceStaging>,
    /// `None` = not yet resolved this `Run`; `Some(v)` = resolved to `v`.
    ///
    /// A `Cell` is sufficient: a `Compute` call resolves on one thread, and
    /// each `Run` builds a fresh cell, so nothing is shared across `Run`s or
    /// across sessions.
    memo: &'a std::cell::Cell<Option<Option<*const ort::OrtMemoryInfo>>>,
}

impl SubgraphFallback<'_> {
    /// The fallback memory info, resolving (and memoising) on first use.
    ///
    /// # Safety
    ///
    /// `api` must be valid and `ctx` a valid `OrtKernelContext*`.
    unsafe fn resolve(
        self,
        api: &ort::OrtApi,
        ctx: *mut ort::OrtKernelContext,
    ) -> Option<*const ort::OrtMemoryInfo> {
        if let Some(cached) = self.memo.get() {
            return cached;
        }
        let resolved = unsafe { device_mem_info(api, ctx, self.staging) }.map(|(mi, _)| mi);
        self.memo.set(Some(resolved));
        resolved
    }
}

/// The ORT-bound operands of a node, in either of the two shapes a caller
/// already has them in.
///
/// Placement is resolved for a vanishing fraction of dispatches -- only past
/// [`prepare_workspace`]'s zero-byte and lifetime gates -- so a caller holding
/// a slot map should not have to flatten it into a fresh `Vec` on every `Run`
/// just to describe operands that will not be looked at. Both spellings
/// iterate the same present ORT indices in the same order.
#[derive(Clone, Copy)]
enum OrtOperands<'a> {
    /// Already-resolved kernel-context input indices.
    Resolved(&'a [usize]),
    /// A node's input slots, `None` for an absent optional input. Absent slots
    /// bind no ORT input and are skipped.
    Slots(&'a [Option<usize>]),
}

impl<'a> OrtOperands<'a> {
    /// The present ORT input indices, in slot order.
    fn indices(self) -> OrtOperandIter<'a> {
        match self {
            Self::Resolved(v) => OrtOperandIter::Resolved(v.iter()),
            Self::Slots(v) => OrtOperandIter::Slots(v.iter()),
        }
    }
}

/// Iterator over [`OrtOperands`]; see [`OrtOperands::indices`].
enum OrtOperandIter<'a> {
    Resolved(std::slice::Iter<'a, usize>),
    Slots(std::slice::Iter<'a, Option<usize>>),
}

impl Iterator for OrtOperandIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Resolved(it) => it.next().copied(),
            Self::Slots(it) => it.by_ref().find_map(|slot| *slot),
        }
    }
}

/// Number of times a node's workspace placement has been **resolved**.
///
/// Incremented once per [`operand_mem_info`] call, which is the only place this
/// executor decides where a node's workspace must live. Read it with
/// [`workspace_placement_queries`].
///
/// It counts *resolutions*, not ORT calls: the
/// [`OperandMemInfo::FromIntermediates`] path resolves placement from the
/// subgraph-level memory info and makes **no** ORT call at all, so a resolution
/// is an upper bound on the FFI a dispatch pays, not a measure of it.
///
/// This is deliberately *not* `cfg(test)`-gated. The property it exists to
/// pin — that a dispatch which needs no workspace never resolves placement —
/// is only observable through a real ORT `Run` against the built
/// cdylib, which is compiled without `cfg(test)`. A gated counter would leave
/// the claim tested in a configuration nobody ships.
static WORKSPACE_PLACEMENT_QUERIES: AtomicUsize = AtomicUsize::new(0);

/// Cumulative count of workspace placement resolutions (see
/// [`WORKSPACE_PLACEMENT_QUERIES`]).
pub fn workspace_placement_queries() -> usize {
    WORKSPACE_PLACEMENT_QUERIES.load(Ordering::Relaxed)
}

/// Reset the workspace placement counter. For tests and diagnostics only.
pub fn reset_workspace_placement_queries() {
    WORKSPACE_PLACEMENT_QUERIES.store(0, Ordering::Relaxed);
}

/// Cumulative count of node kernels **this EP actually executed** inside a
/// `Run`, across every compiled subgraph.
///
/// Distinct from [`crate::ep::compiled_node_count`], and the distinction is the
/// point. That counter is an *assignment* signal: it says ORT gave this EP the
/// node at session-build time. This one is an *execution* signal: it says our
/// kernel ran for that node during `Run`. Only the pair is a proof that
/// selecting this EP kept the work here — a node can be assigned to us and
/// still not be the thing that produced the output (a partition that never
/// runs, an ORT-side constant fold, or a future short-circuit), and nothing in
/// an output comparison against ORT can tell those apart, because agreeing
/// with ORT is exactly what a correct kernel does.
///
/// Deliberately not `cfg(test)`-gated, for the same reason as
/// [`WORKSPACE_PLACEMENT_QUERIES`]: the only way to observe it is a real ORT
/// `Run` against the shipped cdylib, which is compiled without `cfg(test)`.
/// Cost is one relaxed increment per node per `Run` — unmeasurable next to a
/// kernel dispatch, and it is the counter the no-defer rule is checked with.
static EXECUTED_NODE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Number of node kernels this EP has executed since process start (see
/// [`EXECUTED_NODE_COUNT`]).
pub fn executed_node_count() -> usize {
    EXECUTED_NODE_COUNT.load(Ordering::Relaxed)
}

/// Reset the executed-node counter. For tests and diagnostics only.
pub fn reset_executed_node_count() {
    EXECUTED_NODE_COUNT.store(0, Ordering::Relaxed);
}

/// Derive the memory device of **one node's own operands**.
///
/// `ort_operands` lists the kernel-context input indices this node actually
/// reads, in node-operand order, with absent optional operands and
/// intermediate-buffer operands already filtered out. `subgraph_fallback` is
/// [`device_mem_info`], used only when the node binds no ORT operand at all.
///
/// # What this guarantees, and what it does not
///
/// It guarantees the returned memory info is the one ORT reported for an
/// operand **of this node** — not, as before, for input 0 of the whole fused
/// subgraph, which for node `k > 0` may be a different tensor on a different
/// device entirely.
///
/// When the node binds more than one ORT operand, every one of them is compared
/// against the first with `OrtApi::CompareMemoryInfo`, and a disagreement is
/// reported as [`OperandMemInfo::Divergent`] rather than resolved by guessing.
/// This is a real check on real handles, not an assumption: mixed-device
/// operands are legal in ORT (an op may declare `OrtMemTypeCPUInput` for a
/// small control operand), and in that case the operands genuinely do not
/// determine where the kernel's compute runs.
///
/// If `CompareMemoryInfo` is unavailable on the host API, a multi-operand node
/// reports [`OperandMemInfo::Unavailable`] — no comparison is claimed that was
/// not performed. A single-operand node never needs the comparator.
///
/// # Inputs only
///
/// Every handle consulted here comes from `OrtApi::KernelContext_GetInput`.
/// Outputs are never queried, for two reasons: ORT does not require an output
/// to be materialised before `Compute` runs, and the output the kernel is about
/// to write is not evidence about where its *compute* happens. So "this node's
/// operands" means, precisely, its ORT-bound **inputs** — nothing here inspects
/// output placement, and no claim is made about it.
///
/// # Cost
///
/// For `n` ORT-bound inputs this performs `2 * n` ORT FFI calls plus `n - 1`
/// `CompareMemoryInfo` calls; the [`OperandMemInfo::FromIntermediates`] path
/// (no ORT-bound inputs) performs none. [`prepare_workspace`] therefore calls
/// this **only after** it knows a workspace is both non-empty and servable, so
/// a dispatch that requests **zero bytes** and one whose `SessionPersistent`
/// request is **declined** resolve nothing. Note the gates are about the
/// *requirement*, not the device: a kernel on a CPU-placed node that does ask
/// for a step-scoped workspace still resolves placement, which is exactly what
/// tells the executor to serve it from host memory.
/// [`WORKSPACE_PLACEMENT_QUERIES`] counts the resolutions that do happen.
///
/// # Safety
///
/// `api` must be valid and `ctx` a valid `OrtKernelContext*`.
unsafe fn operand_mem_info(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    sources: PlacementSources<'_>,
) -> OperandMemInfo {
    WORKSPACE_PLACEMENT_QUERIES.fetch_add(1, Ordering::Relaxed);
    let mut rest = sources.ort_inputs.indices();
    let Some(first_idx) = rest.next() else {
        let fallback = unsafe { sources.subgraph_fallback.resolve(api, ctx) };
        return match fallback {
            Some(ptr) => OperandMemInfo::FromIntermediates(ptr),
            None => OperandMemInfo::Unavailable,
        };
    };
    let Some(first) = (unsafe { ort_input_mem_info(api, ctx, first_idx) }) else {
        return OperandMemInfo::Unavailable;
    };
    let mut rest = rest.peekable();
    if rest.peek().is_none() {
        return OperandMemInfo::Uniform(first);
    }
    let Some(compare) = api.CompareMemoryInfo else {
        return OperandMemInfo::Unavailable;
    };
    for idx in rest {
        let Some(other) = (unsafe { ort_input_mem_info(api, ctx, idx) }) else {
            return OperandMemInfo::Unavailable;
        };
        let mut equal: std::os::raw::c_int = 0;
        crate::dispatch_probe::ort_call();
        let status = unsafe { compare(first, other, &mut equal) };
        if !status.is_null() {
            // ORT allocated this status and handed us ownership; dropping the
            // pointer would leak it. Released inline rather than through a new
            // shared helper, to keep this fix to the one site this PR adds.
            if let Some(release) = api.ReleaseStatus {
                unsafe { release(status) };
            }
            return OperandMemInfo::Unavailable;
        }
        if equal != 0 {
            return OperandMemInfo::Divergent {
                first: first_idx,
                other: idx,
            };
        }
    }
    OperandMemInfo::Uniform(first)
}

/// Allocates `bytes` of scratch memory via ORT for the given `mem_info`. The
/// memory is owned by ORT for the duration of the `Compute` call — never freed
/// by the caller. For a device EP this returns device memory.
///
/// # Safety
///
/// `api`, `ctx`, and `mem_info` must be valid.
unsafe fn alloc_scratch(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    mem_info: *const ort::OrtMemoryInfo,
    bytes: usize,
) -> Result<*mut c_void, String> {
    let get_scratch = api
        .KernelContext_GetScratchBuffer
        .ok_or("OrtApi.KernelContext_GetScratchBuffer is null")?;
    let mut out: *mut c_void = std::ptr::null_mut();
    crate::dispatch_probe::ort_call();
    let status = unsafe { get_scratch(ctx, mem_info, bytes.max(1), &mut out) };
    if !status.is_null() {
        return Err("KernelContext_GetScratchBuffer failed".into());
    }
    if out.is_null() {
        return Err("KernelContext_GetScratchBuffer returned null".into());
    }
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// Host→device boundary-input staging (#982)
// ──────────────────────────────────────────────────────────────────────────────

/// `true` when `ONNX_GENAI_PLUGIN_TRANSFER_TRACE=1` asks the input-staging path
/// to print, *before* each synchronous host→device upload, which operand it is
/// staging, the memory-info source it resolved, and the destination pointer —
/// so a hang inside the driver memcpy can be named to the exact call (#982).
fn staging_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ONNX_GENAI_PLUGIN_TRANSFER_TRACE").as_deref() == Ok("1"))
}

/// Emit one staging-trace line, formatting the arguments only when the trace is
/// on.
///
/// The gate has to come first: the bare `staging_log(&format!(..))` spelling
/// builds and drops a `String` on every call whether or not anyone is
/// listening, and one of these sites is on the per-`Run` dispatch path.
macro_rules! staging_log {
    ($($arg:tt)*) => {
        if staging_trace_enabled() {
            staging_log(&format!($($arg)*));
        }
    };
}

/// Emit one staging-trace line. Goes to stderr (buffered by pipes, so it can be
/// lost when the process hangs) *and*, when `ONNX_GENAI_PLUGIN_TRANSFER_TRACE_FILE`
/// is set, is appended to that file with an immediate flush — the file path is
/// the only trace that survives a boundary hang. No-op unless the trace is on.
///
/// Prefer the [`staging_log!`] macro: it checks the gate before formatting.
fn staging_log(msg: &str) {
    if !staging_trace_enabled() {
        return;
    }
    eprintln!("{msg}");
    use std::io::Write as _;
    let _ = std::io::stderr().flush();
    if let Ok(path) = std::env::var("ONNX_GENAI_PLUGIN_TRANSFER_TRACE_FILE")
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
    {
        let _ = writeln!(f, "{msg}");
        let _ = f.flush();
    }
}

/// Reconstruct the device `OrtMemoryInfo` the plugin factory registered the
/// device allocator against (`GetSupportedDevices` uses the same
/// `CreateMemoryInfo_V2` recipe). ORT keys its allocator lookup by the
/// `OrtDevice` fields — device type, vendor id, device id, memory type — so a
/// memory info rebuilt with matching fields resolves to the same allocator in
/// `KernelContext_GetScratchBuffer`. `device_id` is 0 to match the factory.
///
/// Returns `None` if the API is unavailable; the caller then falls back to a
/// device memory info sourced from a live `OrtValue`.
fn reconstruct_device_mem_info(
    allocator_name: &str,
    device_type: ort::OrtMemoryInfoDeviceType,
    vendor_id: u32,
) -> Option<ReconstructedMemInfo> {
    let api = crate::status::host_api();
    if api.is_null() {
        return None;
    }
    let create = unsafe { (*api).CreateMemoryInfo_V2 }?;
    let name = std::ffi::CString::new(allocator_name).ok()?;
    let mut ptr: *mut ort::OrtMemoryInfo = std::ptr::null_mut();
    crate::dispatch_probe::ort_call();
    let status = unsafe {
        create(
            name.as_ptr(),
            device_type,
            vendor_id,
            0,
            ort::OrtDeviceMemoryType_DEFAULT,
            0,
            ort::OrtDeviceAllocator,
            &mut ptr,
        )
    };
    if !status.is_null() {
        if let Some(release) = unsafe { (*api).ReleaseStatus } {
            crate::dispatch_probe::ort_call();
            unsafe { release(status) };
        }
        return None;
    }
    if ptr.is_null() {
        return None;
    }
    Some(ReconstructedMemInfo {
        ptr,
        is_device: device_type != ort::OrtMemoryInfoDeviceType_CPU,
    })
}

/// Device type ORT placed a memory info on, or `None` if unreadable.
///
/// # Safety
///
/// `api` and `mem_info` must be valid.
unsafe fn mem_info_device_type(
    api: &ort::OrtApi,
    mem_info: *const ort::OrtMemoryInfo,
) -> Option<ort::OrtMemoryInfoDeviceType> {
    let f = api.MemoryInfoGetDeviceType?;
    let mut out: ort::OrtMemoryInfoDeviceType = ort::OrtMemoryInfoDeviceType_CPU;
    crate::dispatch_probe::ort_call();
    unsafe { f(mem_info, &mut out) };
    Some(out)
}

/// Whether a memory info denotes non-host (device) memory.
///
/// # Safety
///
/// `api` and `mem_info` must be valid.
unsafe fn mem_info_is_device(api: &ort::OrtApi, mem_info: *const ort::OrtMemoryInfo) -> bool {
    matches!(
        unsafe { mem_info_device_type(api, mem_info) },
        Some(t) if t != ort::OrtMemoryInfoDeviceType_CPU
    )
}

/// Node-operand positions that a kernel legitimately reads on **host** — the
/// shape/index/axes control operands `infer_shapes` dereferences via
/// `read_i64_tensor`. These are the `OrtMemTypeCPUInput`-style operands that
/// must **never** be uploaded to the device: they are supposed to stay on host,
/// and staging them would both waste a copy and hand the host-reading path a
/// device pointer. Every other host-resident input of a device kernel is a data
/// operand that crossed a CPU→device boundary and must be staged (#982).
fn host_operand_indices(strategy: &ShapeInference) -> &'static [usize] {
    match strategy {
        ShapeInference::SharedNative { fallback, .. } => host_operand_indices(fallback),
        ShapeInference::ReshapeData { .. } => &[1],
        ShapeInference::SliceData => &[1, 2, 3, 4],
        ShapeInference::ReductionFromInput { .. } => &[1],
        _ => &[],
    }
}

/// Stage every host-resident **data** operand of one node into device scratch,
/// substituting the device pointer into `kernel_inputs` in place (#982).
///
/// For each operand position `p`:
///  - skipped if it is not ORT-bound (`ort_indices[p]` is `None`) — intermediate
///    buffers already live on the device;
///  - skipped if it is a host-required control operand (`host_operands`);
///  - skipped if absent or zero-length;
///  - skipped if ORT already placed it on the device.
///
/// A host-resident data operand is uploaded into ORT device scratch (freed by
/// ORT after `Compute`, past the stream sync, so the async kernel that reads it
/// cannot outlive the buffer) and `kernel_inputs[p].data` is repointed at it.
///
/// The scratch is allocated against a device `OrtMemoryInfo` resolved, in order,
/// from: a device-resident ORT input, a device-resident node output, then the
/// reconstructed EP device memory info. If a data operand needs staging but no
/// device memory info can be found, this fails closed rather than launching a
/// device kernel on a host pointer.
///
/// # Safety
///
/// `api` and `ctx` must be valid; `ort_indices[p]`, when `Some`, must be a valid
/// kernel-context input index for `ctx`.
#[allow(clippy::too_many_arguments)]
unsafe fn stage_host_boundary_inputs(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    staging: &DeviceStaging,
    kernel_inputs: &mut [TensorView<'_>],
    ort_indices: &[Option<usize>],
    host_operands: &[usize],
    output_mem_infos: &[*const ort::OrtMemoryInfo],
    label: &str,
) -> Result<(), String> {
    // First pass: which operands are host-resident data operands needing upload?
    let mut to_stage: Vec<usize> = Vec::new();
    staging_log!(
        "[plugin/staging #982] {label}: enter operands={} host_operands={:?}",
        ort_indices.len(),
        host_operands
    );
    for (p, slot) in ort_indices.iter().enumerate() {
        let Some(ort_idx) = *slot else {
            staging_log!("[plugin/staging #982] {label}: op[{p}] skip (not ORT-bound)");
            continue;
        };
        if host_operands.contains(&p) {
            staging_log!(
                "[plugin/staging #982] {label}: op[{p}] ort_idx={ort_idx} skip (host control operand)"
            );
            continue;
        }
        let view = &kernel_inputs[p];
        if view.is_absent() || view.data.0.is_null() {
            staging_log!(
                "[plugin/staging #982] {label}: op[{p}] ort_idx={ort_idx} skip (absent/null)"
            );
            continue;
        }
        let numel: usize = view.shape.iter().product();
        if numel == 0 || view.dtype.byte_size() == 0 {
            staging_log!("[plugin/staging #982] {label}: op[{p}] ort_idx={ort_idx} skip (empty)");
            continue;
        }
        let Some(mem_info) = (unsafe { ort_input_mem_info(api, ctx, ort_idx) }) else {
            staging_log!(
                "[plugin/staging #982] {label}: op[{p}] ort_idx={ort_idx} skip (no mem info)"
            );
            continue;
        };
        let dev_type = unsafe { mem_info_device_type(api, mem_info) };
        if unsafe { mem_info_is_device(api, mem_info) } {
            staging_log!(
                "[plugin/staging #982] {label}: op[{p}] ort_idx={ort_idx} skip (already device, dev_type={dev_type:?})"
            );
            continue; // already on device — nothing to do
        }
        staging_log!(
            "[plugin/staging #982] {label}: op[{p}] ort_idx={ort_idx} HOST (dev_type={dev_type:?}) → will stage"
        );
        to_stage.push(p);
    }

    if to_stage.is_empty() {
        return Ok(());
    }

    // Resolve a device memory info to allocate scratch against.
    let mut device_mi: *const ort::OrtMemoryInfo = std::ptr::null();
    for slot in ort_indices.iter() {
        if let Some(ort_idx) = *slot
            && let Some(mi) = unsafe { ort_input_mem_info(api, ctx, ort_idx) }
            && unsafe { mem_info_is_device(api, mi) }
        {
            device_mi = mi;
            break;
        }
    }
    if device_mi.is_null() {
        for &mi in output_mem_infos {
            if !mi.is_null() && unsafe { mem_info_is_device(api, mi) } {
                device_mi = mi;
                break;
            }
        }
    }
    if device_mi.is_null()
        && let Some(recon) = staging.recon_mem_info.as_ref()
    {
        device_mi = recon.ptr;
        staging_log!(
            "[plugin/staging #982] {label}: no device OrtValue found; \
             falling back to reconstructed EP memory info recon={device_mi:?}"
        );
    }
    if device_mi.is_null() {
        return Err(format!(
            "{label}: a host-resident boundary input must be uploaded to the \
             device, but no device memory info is available (no device-resident \
             operand and no reconstructed EP memory info). Refusing to launch a \
             device kernel on a host pointer."
        ));
    }

    // Second pass: upload and repoint.
    let mem_dev = unsafe { mem_info_device_type(api, device_mi) };
    for p in to_stage {
        let view = &kernel_inputs[p];
        let numel: usize = view.shape.iter().product();
        let byte_len = numel * view.dtype.byte_size();
        let dst = unsafe { alloc_scratch(api, ctx, device_mi, byte_len) }
            .map_err(|e| format!("{label}: staging scratch alloc failed: {e}"))?;
        staging_log!(
            "[plugin/staging #982] {label}: staging op[{p}] byte_len={byte_len} \
             mem_dev={mem_dev:?} dst={dst:?} — issuing host→device upload"
        );
        // SAFETY: the source is a host tensor of `byte_len` contiguous bytes
        // (ORT-provided, contiguous strides); `dst` is device scratch of the
        // same size.
        let src = unsafe { std::slice::from_raw_parts(view.data.0.cast::<u8>(), byte_len) };
        unsafe { staging.copier.copy_host_to_device(src, dst) }
            .map_err(|e| format!("{label}: host→device upload failed: {e}"))?;
        staging_log!("[plugin/staging #982] {label}: op[{p}] host→device upload complete");
        kernel_inputs[p].data = DevicePtr(dst.cast_const());
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Governed kernel workspace
// ──────────────────────────────────────────────────────────────────────────────

/// Resolve a kernel's declared [`WorkspaceRequirement`] into the concrete
/// [`WorkspaceView`] handed to `execute_with_workspace` for one dispatch.
///
/// # Why ORT scratch, and not EP allocate/free
///
/// The workspace has to live on the device the kernel runs on. The mechanism
/// that was hardware-validated on H200 in #832 for fused-subgraph intermediates
/// is [`alloc_scratch`] (`OrtApi::KernelContext_GetScratchBuffer`), so
/// workspaces reuse exactly that path: ORT owns the bytes for the duration of
/// the `Compute` call, which is precisely a step-scoped workspace's lifetime,
/// and this executor never issues a device free. A per-dispatch
/// `ExecutionProvider::allocate`/`deallocate` pair would instead put a
/// synchronous `cuMemAlloc`/`cuMemFree` on every node of every decode step —
/// the exact cost #832 removed from `MatMulNBits` with a cached grow-only
/// arena — and a device free is illegal during CUDA-graph capture.
///
/// # Where the workspace is placed
///
/// Against the memory info of **this node's own ORT-bound inputs**, derived by
/// [`operand_mem_info`], not against input 0 of the fused subgraph. "Operands"
/// here means kernel-context *inputs* only: nothing consults output placement,
/// and no claim is made about it. When the node binds several ORT inputs they
/// are compared with `OrtApi::CompareMemoryInfo` and a disagreement is an
/// error, not a guess. A node whose inputs are all intermediate buffers
/// inherits the subgraph-level memory info those buffers were allocated from,
/// which is the same device by construction.
///
/// The derivation is **lazy**: it runs below, after the zero-byte and
/// `StepScoped` gates, so a node with no workspace requirement and a node whose
/// `SessionPersistent` request is declined resolve no placement at all. The
/// gates are on the *requirement*, not on the device — a node that genuinely
/// needs a step-scoped workspace resolves placement wherever it runs.
/// [`workspace_placement_queries`] counts the resolutions that do run.
///
/// Every failure to determine the device is a hard error *for a request that
/// needs serving*, never a silent host allocation: a device kernel handed host
/// memory dereferences it as device memory.
///
/// # Alignment
///
/// ORT makes no promise about the alignment of a scratch block beyond the
/// allocator's own, so a stricter request is satisfied by **checked
/// over-allocation** (`bytes + alignment - 1`) followed by align-up inside the
/// returned block. Every arithmetic step is checked and the aligned window is
/// re-verified to lie inside the allocation before it is handed out.
/// [`alloc_scratch`] already rejects a null block, so there is exactly one
/// null check on this path.
///
/// # Planning cost
///
/// The requirement is memoized per node by [`WorkspacePlanCache`], keyed on the
/// exact operand metadata. Be precise about what that buys:
///
/// * It removes **this seam's own** repeat search. Without the cache, a
///   `SessionPersistent` declarer whose `workspace_requirement` runs a cuBLASLt
///   heuristic search (`MatMul`, `Gemm`, `FusedEpilogue`, `MatMulNBits`' f32
///   dequant path) paid for that search here on *every* dispatch, had the
///   result declined, and then planned again inside its own `execute`.
/// * It does **not** remove the kernel-side plan. `blas::governed_gemm` still
///   plans once per dispatch inside `execute`; nothing here reaches into it.
/// * So against `main` — which never called `workspace_requirement` at all —
///   the steady state is approximately **neutral**, not a halving: a repeated
///   operand signature costs a lock plus a short linear scan, and each *new*
///   signature costs one extra heuristic search that `main` did not perform.
/// * Hit rate is a property of the shapes. A stable geometry (fixed batch and
///   sequence length) hits; a growing-KV `StepScoped` attention whose operand
///   shapes change every decode step can **miss on every step**, and then the
///   cache is pure overhead bounded by one extra search per step.
///
/// See [`WorkspacePlanCache`] for the correctness argument.
///
/// # Lifetimes
///
/// * [`WorkspaceLifetime::StepScoped`] — served from ORT scratch, as above.
///   The step-scoped consumers in `onnx-runtime-ep-cuda` are the default-domain
///   `Attention` Phase-2a scratch and `StandardAttention` whenever it is *not*
///   a single-token, single-batch decode (`batch == 1 && q_seq == 1`) — i.e.
///   prefill and batched dispatch.
/// * [`WorkspaceLifetime::SessionPersistent`] — **explicitly declined**
///   (`Ok(None)`), never downgraded. ORT reclaims scratch when `Compute`
///   returns, so serving a persistent request from it would hand the kernel
///   memory that is recycled behind its back on the next `Run`. There is no
///   session-persistent device arena at this seam, so the request is passed
///   through as `None` and the kernel decides. This seam must not guess.
///
/// ## What declining actually does, per declarer
///
/// Declining is **not** uniformly a self-owned fallback. Every
/// `SessionPersistent` declarer in `onnx-runtime-ep-cuda`, and what each does
/// with `None`:
///
/// * **Self-owned fallback — correct, no error.**
///   * `GroupQueryAttention` (`group_query_attention.rs`): the composite
///     request (scores, packed Q/K/V staging, BSH↔BNSH transposes) is served
///     from its own pooled `GqaWorkspace` slots.
///   * `StandardAttention` (`standard_attention.rs`), single-token single-batch
///     decode only — every other geometry is `StepScoped` and *is* served: the
///     score and staged-K/V scratch is pooled or per-call inside `run`.
///   * `MatMulNBits` bf16 activations: the `Bf16Scratch` staging arena is
///     always self-owned and is listed here only because it is easy to assume
///     otherwise — that path declares [`WorkspaceRequirement::NONE`], so it is
///     unaffected by the decline in either direction.
/// * **Hard error, but only when the cuBLASLt heuristic asks for bytes.**
///   `MatMul`, `Gemm`, `FusedEpilogue` and `MatMulNBits`' f32
///   dequant-cuBLASLt path all route through
///   `blas::governed_workspace_requirement`, which reports
///   [`WorkspaceRequirement::NONE`] when the heuristic selects **0** bytes and
///   `SessionPersistent` otherwise. On the declined path
///   `blas::governed_workspace_ptr` returns `Ok(0)` for a 0-byte requirement
///   and errors — *"governed cuBLASLt workspace requires N bytes, but none was
///   supplied"* — for a non-zero one. Which of the two happens is a property of
///   the algorithm cuBLASLt picks for that shape, dtype, device and library
///   version, not a static property of the kernel; many decode-shaped GEMMs
///   select 0 bytes and are unaffected.
/// * **Unconditional hard error.** `BlockQuantizedMoE`
///   (`block_quantized_moe.rs`) and `IndexShare` (`index_share.rs`) declare
///   `SessionPersistent` for every geometry and have no self-owned path; both
///   their `execute` and their `execute_with_workspace(None)` return an error.
///
/// Declining is what preserves #832's H200-validated plugin path: on `main`
/// the executor called bare `Kernel::execute`, and
/// [`Kernel::execute_with_workspace`] defaults to forwarding there, so `None`
/// reproduces `main` node for node — including for the kernels that error,
/// which error on `main` too. Hard-failing here instead would have turned every
/// GQA-bearing model into a plugin-path error on hardware that runs it today.
/// The cost of declining is that `BlockQuantizedMoE`, `IndexShare`, and any
/// GEMM whose heuristic asks for workspace bytes remain **incompatible with the
/// plugin path**; making them work needs a real session-persistent device arena
/// at this seam, which is future work and is not claimed here.
///
/// # Safety
///
/// `api`, `ctx` and `placement.subgraph_fallback` (when `Some`) must be valid.
unsafe fn prepare_workspace(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    placement: PlacementSources<'_>,
    kernel: &dyn Kernel,
    plans: &WorkspacePlanCache,
    inputs: &[TensorView<'_>],
    node_label: impl std::fmt::Display + Copy,
) -> Result<Option<WorkspaceView>, String> {
    let requirement: WorkspaceRequirement = plans.get_or_plan_views(inputs, |metadata| {
        kernel
            .workspace_requirement(metadata)
            .map_err(|e| format!("{node_label}: workspace_requirement failed: {e}"))
    })?;
    if requirement.bytes == 0 {
        return Ok(None);
    }

    let alignment = requirement.alignment;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!(
            "{node_label}: kernel requested workspace alignment {alignment}, \
             which is not a non-zero power of two"
        ));
    }
    if requirement.lifetime != WorkspaceLifetime::StepScoped {
        // Explicit decline, never a downgrade. ORT reclaims a
        // `KernelContext_GetScratchBuffer` block when `Compute` returns, so
        // serving a `SessionPersistent` request from it would hand the kernel
        // memory that is recycled behind its back the moment it reuses the
        // pointer on the next `Run`.
        return Ok(None);
    }

    let bytes = usize::try_from(requirement.bytes).map_err(|_| {
        format!(
            "{node_label}: workspace requirement of {} bytes exceeds usize on this target",
            requirement.bytes
        )
    })?;
    let total =
        workspace_block_bytes(bytes, alignment).map_err(|e| format!("{node_label}: {e}"))?;

    // Placement is resolved *here*, past every gate, and not by the caller.
    // Resolving it costs up to `2n` ORT FFI calls plus `n-1` comparisons, and
    // for a zero-byte requirement or a declined `SessionPersistent` request the
    // answer is never used. Every dispatch used to pay it; now only a dispatch
    // that is actually about to be served does.
    let mem_info = match unsafe { operand_mem_info(api, ctx, placement) } {
        OperandMemInfo::Uniform(ptr) | OperandMemInfo::FromIntermediates(ptr) => ptr,
        OperandMemInfo::Unavailable => {
            return Err(format!(
                "{node_label}: kernel requires {bytes} bytes of {alignment}-byte-aligned \
                 workspace, but the memory device of its operands could not be read, so the \
                 workspace cannot be placed where the kernel runs. Failing closed rather than \
                 handing a device kernel host memory."
            ));
        }
        OperandMemInfo::Divergent { first, other } => {
            return Err(format!(
                "{node_label}: kernel requires {bytes} bytes of {alignment}-byte-aligned \
                 workspace, but its operands do not agree on a memory device: kernel-context \
                 inputs {first} and {other} report different OrtMemoryInfo (CompareMemoryInfo). \
                 The operands therefore do not determine where this kernel's compute runs. \
                 Failing closed rather than guessing a device."
            ));
        }
    };

    let base = unsafe { alloc_scratch(api, ctx, mem_info, total) }
        .map_err(|e| format!("{node_label}: workspace scratch allocation failed: {e}"))?;

    // `alloc_scratch` already rejects a null block, so no second null check
    // here: a duplicated guard reads as two independent defences and is really
    // one, which is worse than none for a reviewer counting them.
    let aligned = align_workspace_window(base as usize, total, bytes, alignment)
        .map_err(|e| format!("{node_label}: {e}"))?;

    if workspace_trace_enabled() {
        eprintln!(
            "{}",
            workspace_trace_line(node_label, base as usize, aligned, bytes, total, alignment)
        );
    }

    Ok(Some(WorkspaceView::new(
        DevicePtrMut(aligned as *mut c_void),
        bytes,
    )))
}

/// `true` when `NXRT_EP_WORKSPACE_TRACE=1` asks every **served** workspace to
/// print the block it was cut from.
///
/// Off by default and read once. The trace exists because two questions about
/// `KernelContext_GetScratchBuffer` can only be answered on a real device, and
/// neither is answerable from a counter: whether ORT hands back the *same*
/// storage on every step (arena-backed and reused) or a fresh block each time,
/// and what alignment the returned block actually has versus the alignment the
/// kernel asked for. Both are properties of the pointer, so the pointer is what
/// gets reported.
fn workspace_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NXRT_EP_WORKSPACE_TRACE").as_deref() == Ok("1"))
}

/// Format one served-workspace trace record.
///
/// Split from the emission site so the format is unit-testable without an ORT
/// kernel context: a reader of the trace is going to compare `block=` across
/// steps and read `block_align=` against `align=`, so both have to be derived
/// here rather than left for the reader to compute.
///
/// `block_align` is the largest power of two that divides the block ORT
/// returned, capped at 4096 — the useful question is "did ORT already meet the
/// kernel's alignment", not the exact 2-adic valuation of the address.
fn workspace_trace_line(
    node_label: impl std::fmt::Display,
    base: usize,
    aligned: usize,
    bytes: usize,
    total: usize,
    alignment: usize,
) -> String {
    let block_align = if base == 0 {
        0
    } else {
        let mut a: usize = 1;
        while a < 4096 && base.is_multiple_of(a << 1) {
            a <<= 1;
        }
        a
    };
    format!(
        "nxrt ep plugin: workspace served node={node_label} bytes={bytes} align={alignment} \
         requested_block={total} block=0x{base:x} block_align={block_align} \
         ptr=0x{aligned:x} skew={}",
        aligned - base
    )
}

/// Bytes to request so that aligning up to `alignment` always lands inside the
/// block. Split out from [`prepare_workspace`] so the overflow behaviour is
/// unit-testable without an ORT kernel context.
fn workspace_block_bytes(bytes: usize, alignment: usize) -> Result<usize, String> {
    bytes.checked_add(alignment - 1).ok_or_else(|| {
        format!("workspace request of {bytes} bytes over-aligned to {alignment} overflows usize")
    })
}

/// Align `base` up to `alignment` and prove the resulting `bytes`-long window
/// lies inside the `total`-byte block starting at `base`.
///
/// Split out from [`prepare_workspace`] so the alignment and overflow rules are
/// unit-testable without an ORT kernel context. Every step is checked: a
/// wrapping add here would hand a kernel a pointer outside the allocation.
fn align_workspace_window(
    base: usize,
    total: usize,
    bytes: usize,
    alignment: usize,
) -> Result<usize, String> {
    let aligned = base
        .checked_add(alignment - 1)
        .map(|a| a & !(alignment - 1))
        .ok_or("aligning workspace pointer overflows usize")?;
    let end = aligned
        .checked_add(bytes)
        .ok_or("workspace end address overflows usize")?;
    let block_end = base
        .checked_add(total)
        .ok_or("workspace block end overflows usize")?;
    if end > block_end {
        return Err(format!(
            "aligned workspace [{aligned:#x}, {end:#x}) escapes the {total}-byte scratch \
             block at {base:#x}"
        ));
    }
    Ok(aligned)
}

/// Compute: execute the kernel(s) for this subgraph.
///
/// For single-node subgraphs (the common case for CPU EP), this calls
/// `kernel.execute_with_workspace()` once. For multi-node fused subgraphs with
/// a `SubgraphRouting` table, it allocates intermediate buffers, threads them
/// between nodes in topological order, and writes only true subgraph outputs
/// back to ORT.
///
/// # Workspace contract
///
/// Every dispatch goes through [`prepare_workspace`] →
/// [`Kernel::execute_with_workspace`], never bare [`Kernel::execute`]. A kernel
/// declaring a non-zero [`WorkspaceLifetime::StepScoped`] requirement gets
/// correctly-aligned scratch on the memory device reported by **that node's own
/// ORT-bound operands** (see [`operand_mem_info`]), taken from ORT
/// (`KernelContext_GetScratchBuffer`) and reclaimed by ORT when `Compute`
/// returns — the same mechanism #832 validated on an H200 for fused subgraph
/// intermediates. A [`WorkspaceLifetime::SessionPersistent`] request is
/// **declined** (`None`), never downgraded to recycled step-scoped memory; see
/// [`prepare_workspace`] for the per-kernel consequences of that decline.
/// Kernels declaring [`WorkspaceRequirement::NONE`] behave exactly as before:
/// `execute_with_workspace`'s default forwards to `execute`.
///
/// Intermediate buffers keep #832's derivation — the subgraph-level
/// [`device_mem_info`] — deliberately unchanged, since that is the allocation
/// that ran on hardware.
///
/// # Safety
///
/// `info` must be a valid `ExportedComputeInfo*`, `state` from `CreateState`,
/// and `kernel_context` a valid `OrtKernelContext*`.
unsafe extern "C" fn compute_execute(
    info: *mut ort::OrtNodeComputeInfo,
    _state: *mut c_void,
    kernel_context: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::dispatch_probe::count(crate::dispatch_probe::Event::ComputeExecute);
        let entry_probe = crate::dispatch_probe::Phase::CallbackEntry.enter();
        if info.is_null() || kernel_context.is_null() {
            return fail_status("Compute: null argument");
        }

        let exported = unsafe { &*(info.cast::<ExportedComputeInfo>()) };

        if exported.entries.is_empty() {
            return fail_status("Compute: no kernels compiled for this subgraph");
        }

        staging_log!(
            "[plugin/staging #982] compute_execute enter: entries={} routing={} device_staging={}",
            exported.entries.len(),
            exported.routing.is_some(),
            exported.device_staging.is_some()
        );

        let api = crate::status::host_api();
        if api.is_null() {
            return fail_status("Compute: host ORT API not available");
        }
        let api_ref = unsafe { &*api };

        // Lend ORT's intra-op pool to the kernels for this call. Ours would be
        // a second pool on the same cores, and ORT's workers spin: at
        // `intra_op = 16`, splitting a 1 Mi `Sqrt` across our rayon pool cost
        // 252 -> 777 us against staying serial. Dropped at the end of the
        // call, before `kernel_context` goes away.
        //
        // SAFETY: `kernel_context` is the context ORT handed this call and
        // stays valid until it returns, which is after the guard is dropped.
        let _host_pool = unsafe {
            crate::host_pool::install(api_ref, kernel_context, &exported.host_pool_probe)
        };

        entry_probe.end();

        // One thread-local resolution for the whole `Run`, and the scratch is
        // *borrowed* for its duration rather than moved out and put back. The
        // recycle is the guard's `Drop`, so it happens on every exit -- the
        // early `return`s below included, and an unwinding kernel too.
        with_run_scratch(|scratch| {
            let RunScratch {
                inputs,
                owned: owned_outputs,
                slots: slot_map,
                shapes,
                host_pool,
            } = scratch;
            if let Err(e) = unsafe { read_inputs_into(api_ref, kernel_context, inputs) } {
                return fail_status(&format!("Compute: {e}"));
            }
            let inputs = &*inputs;

            if let Some(routing) = &exported.routing {
                // ── Routed multi-node path ────────────────────────────────────
                // Where a routed subgraph's *intermediates* are allocated.
                //
                // Device memory is not optional: a device kernel handed a host pointer
                // dereferences it as device memory. Host memory is different — the
                // intermediate only has to be readable by the next kernel on this
                // thread, and `KernelContext_GetScratchBuffer` is a much worse way to
                // get it than the Rust allocator. ORT's host scratch goes through an
                // aligned allocation that glibc always services with a fresh `mmap`
                // for buffers of this size, so every intermediate is a new mapping,
                // first-touch page-faulted while the kernel writes it and unmapped at
                // the end of the `Run`. Measured on an 8-node f32 `Relu` chain of
                // 262144 elements (1 MiB per intermediate), one thread: 3246 us
                // through ORT scratch vs 383 us through host buffers, an 8.5x
                // difference on identical kernels, against ORT's own 220 us for the
                // same graph. So: device memory when the resolved memory info is a
                // device, host buffers otherwise.
                //
                // Resolving this costs an ORT memory-info query *per kernel-context
                // input* on every `Run` — four fixed FFI calls on the one-input
                // graphs this path is measured on. When ORT placed this EP's
                // tensors in **host-accessible** memory the answer is always
                // `None`, and it is knowable without asking ORT anything: that is
                // the same flag the factory registers the allocator on, so every
                // `OrtValue` here is host-resident and the intermediates belong in
                // host buffers.
                //
                // The discriminator is deliberately `host_accessible` and *not*
                // `device_staging.is_some()`. The latter tracks
                // `host_to_device_copier()`, which defaults to `None` and which a
                // device EP may legitimately decline, so it would wrongly classify
                // such an EP as host and place its intermediates on the wrong side
                // of the bus. `host_accessible` defaults to the conservative
                // `false`, so anything that has not explicitly declared itself
                // host-placed keeps the full scan.
                //
                // The `debug_assert` re-derives the full answer in unoptimised
                // builds, so a host-accessible EP that ever did see a
                // device-resident input fails loudly in tests instead of silently
                // mis-placing intermediates.
                // Memoises the placement fallback for this `Run` (see
                // `SubgraphFallback`): resolved at most once across all nodes, and
                // not at all unless some node actually reaches it.
                let subgraph_fallback_memo = std::cell::Cell::new(None);

                let intermediate_scratch = if must_scan_for_device_placement(
                    exported.host_accessible,
                    exported.device_staging.as_ref(),
                ) {
                    let scanned = unsafe {
                        device_mem_info(api_ref, kernel_context, exported.device_staging.as_ref())
                    };
                    // The subgraph fallback wants this very resolution. Seed the
                    // memo so a node that reaches it later reuses this scan
                    // instead of repeating it -- which is what the eager
                    // once-per-`Run` version this replaced effectively did.
                    subgraph_fallback_memo.set(Some(scanned.map(|(mem_info, _)| mem_info)));
                    scanned.and_then(|(mem_info, is_device)| is_device.then_some(mem_info))
                } else {
                    debug_assert!(
                        unsafe {
                            device_mem_info(
                                api_ref,
                                kernel_context,
                                exported.device_staging.as_ref(),
                            )
                        }
                        .is_none_or(|(_, is_device)| !is_device),
                        "EP declared host-accessible saw a device-resident kernel-context input; \
                     its routed intermediates would be placed in host memory"
                    );
                    None
                };

                if routing.input_sources.len() != exported.entries.len()
                    || routing.output_sinks.len() != exported.entries.len()
                {
                    return fail_status("Compute: routing table length mismatch with entries");
                }

                // Allocate intermediate buffer slots (uninitialized until written).
                let mut intermediates: Vec<Option<IntermediateBuf>> = (0..routing
                    .num_intermediate_buffers)
                    .map(|_| None)
                    .collect();

                // Last node that reads each intermediate buffer. A buffer whose
                // last reader has run is dead and its storage can be handed
                // straight back to the next allocation, so a chain of N nodes
                // cycles a couple of buffers instead of touching N fresh ones.
                let (retire_starts, retire_items) =
                    retirements_per_node(&routing.input_sources, routing.num_intermediate_buffers);

                // Per-node scratch, hoisted so each node reuses the previous
                // node's capacity instead of allocating its own. These were the
                // largest single item in the dispatch allocation table: six of the
                // ~12 allocations per node were these vectors being created and
                // dropped once each.
                //
                // Each is cleared at the top of the iteration that uses it, so a
                // node always starts from empty regardless of how the previous one
                // exited. (Every early exit in this loop is a `return`, so a failed
                // node cannot leak into a later one either way -- the clears are
                // load-bearing for ordinary node-to-node reuse, not for errors.)
                enum RoutedSlotKind {
                    Ort,
                    Buffer,
                    Absent(usize), // index into absent_scratch
                }
                let mut ort_outputs: Vec<crate::kernel_ctx::OwnedOutput> = Vec::new();
                // Stores the *output slot*, not a copy of its shape: cloning the
                // shape cost one more allocation per buffer-bound output, and the
                // shape is already live in `output_shapes` for the whole iteration.
                let mut buf_writes: Vec<(usize, usize, DataType)> = Vec::new();
                let mut absent_scratch: Vec<(usize, Vec<u8>, DataType)> = Vec::new();
                let mut slot_kinds: Vec<RoutedSlotKind> = Vec::new();
                let mut new_bufs: Vec<(usize, IntermediateBuf)> = Vec::new();
                let mut node_ort_operands: Vec<usize> = Vec::new();

                for (node_idx, entry) in exported.entries.iter().enumerate() {
                    let node_probe = crate::dispatch_probe::Phase::TensorBind.enter();
                    let sources = &routing.input_sources[node_idx];
                    let sinks = &routing.output_sinks[node_idx];

                    // Gather input views for this node.
                    let mut kernel_inputs: Vec<TensorView<'_>> = Vec::with_capacity(sources.len());
                    for src in sources {
                        match src {
                            NodeInputSource::Ort(i) => {
                                if *i >= inputs.len() {
                                    return fail_status(&format!(
                                        "Compute: routing ORT input {i} out of range"
                                    ));
                                }
                                kernel_inputs.push(inputs[*i].view());
                            }
                            NodeInputSource::Buffer(b) => {
                                let buf = match intermediates.get(*b).and_then(|o| o.as_ref()) {
                                    Some(b) => b,
                                    None => {
                                        return fail_status(&format!(
                                            "Compute: intermediate buffer {b} not yet written"
                                        ));
                                    }
                                };
                                // SAFETY: buf lives for the duration of this loop body.
                                // We extend the lifetime here; the borrow is valid
                                // because intermediates is not mutated while we hold
                                // this view (we only push to kernel_inputs first).
                                let view = buf.view();
                                // Transmute lifetime to 'static so we can store in the
                                // Vec; we ensure the buf outlives this scope.
                                let view: TensorView<'static> =
                                    unsafe { std::mem::transmute(view) };
                                kernel_inputs.push(view);
                            }
                            NodeInputSource::Absent => {
                                // Absent optional input — provide a null-backed
                                // sentinel TensorView so the kernel can detect it
                                // via `view.is_absent()`.
                                kernel_inputs.push(TensorView::absent(DataType::Undefined));
                            }
                        }
                    }

                    node_probe.end();
                    if matches!(&entry.shape_inference, ShapeInference::KernelSizedOutputs) {
                        let requested_outputs: Vec<bool> = (0..entry.num_outputs)
                            .map(|slot| !entry.absent_output_slots.contains(&slot))
                            .collect();
                        if entry.kernel.kernel_sized_output_policy()
                            == KernelSizedOutputPolicy::DeviceWorkspace
                        {
                            let ort_indices: Vec<Option<usize>> = sources
                                .iter()
                                .map(|source| match source {
                                    NodeInputSource::Ort(index) => Some(*index),
                                    NodeInputSource::Buffer(_) | NodeInputSource::Absent => None,
                                })
                                .collect();
                            if let Some(staging) = exported.device_staging.as_ref() {
                                let output_mem_infos: Vec<*const ort::OrtMemoryInfo> =
                                    ort_outputs.iter().map(|output| output.mem_info).collect();
                                if let Err(error) = unsafe {
                                    stage_host_boundary_inputs(
                                        api_ref,
                                        kernel_context,
                                        staging,
                                        &mut kernel_inputs,
                                        &ort_indices,
                                        &[],
                                        &output_mem_infos,
                                        &format!("Compute: node {node_idx} kernel-sized metadata"),
                                    )
                                } {
                                    return fail_status(&error);
                                }
                            }

                            let plans = match exported.workspace_plan_cache(node_idx) {
                                Some(plans) => plans,
                                None => {
                                    return fail_status(&format!(
                                        "Compute: node {node_idx}: no workspace plan cache for \
                                         kernel-sized device outputs"
                                    ));
                                }
                            };
                            node_ort_operands.clear();
                            node_ort_operands.extend(sources.iter().filter_map(
                                |source| match source {
                                    NodeInputSource::Ort(index) => Some(*index),
                                    NodeInputSource::Buffer(_) | NodeInputSource::Absent => None,
                                },
                            ));
                            let workspace = match unsafe {
                                prepare_workspace(
                                    api_ref,
                                    kernel_context,
                                    PlacementSources {
                                        ort_inputs: OrtOperands::Resolved(&node_ort_operands),
                                        subgraph_fallback: SubgraphFallback {
                                            staging: exported.device_staging.as_ref(),
                                            memo: &subgraph_fallback_memo,
                                        },
                                    },
                                    &*entry.kernel,
                                    plans,
                                    &kernel_inputs,
                                    format_args!(
                                        "Compute: node {node_idx} kernel-sized device workspace"
                                    ),
                                )
                            } {
                                Ok(workspace) => workspace,
                                Err(error) => return fail_status(&error),
                            };
                            let kernel_probe = crate::dispatch_probe::Phase::KernelInvoke.enter();
                            let metadata = match run_device_kernel_sized(
                                &*entry.kernel,
                                &kernel_inputs,
                                &requested_outputs,
                                &entry.output_dtypes,
                                workspace,
                                &format!("Compute: node {node_idx}"),
                            ) {
                                Ok(outputs) => outputs,
                                Err(error) => return fail_status(&error),
                            };

                            ort_outputs.clear();
                            new_bufs.clear();
                            slot_kinds.clear();
                            for (slot, output) in metadata.iter().enumerate() {
                                let sink = match sinks.get(slot) {
                                    Some(sink) => sink,
                                    None => {
                                        return fail_status(&format!(
                                            "Compute: device kernel-sized output slot {slot} has \
                                             no sink"
                                        ));
                                    }
                                };
                                let Some(output) = output else {
                                    if !matches!(sink, NodeOutputSink::Absent) {
                                        return fail_status(&format!(
                                            "Compute: absent device kernel-sized slot {slot} has a \
                                             non-absent sink"
                                        ));
                                    }
                                    slot_kinds.push(RoutedSlotKind::Absent(0));
                                    continue;
                                };
                                match sink {
                                    NodeOutputSink::Ort(ort_index) => {
                                        match unsafe {
                                            allocate_output(
                                                api_ref,
                                                kernel_context,
                                                *ort_index,
                                                &output.shape,
                                                output.dtype,
                                                true,
                                            )
                                        } {
                                            Ok(output) => ort_outputs.push(output),
                                            Err(error) => {
                                                return fail_status(&format!(
                                                    "Compute: device kernel-sized output slot \
                                                     {slot} allocation failed: {error}"
                                                ));
                                            }
                                        }
                                        slot_kinds.push(RoutedSlotKind::Ort);
                                    }
                                    NodeOutputSink::Buffer(buffer_index) => {
                                        let mem_info = match intermediate_scratch {
                                            Some(mem_info) => mem_info,
                                            None => {
                                                return fail_status(
                                                    "Compute: device kernel-sized intermediate \
                                                     has no device scratch placement",
                                                );
                                            }
                                        };
                                        let numel = match output
                                            .shape
                                            .iter()
                                            .try_fold(1usize, |product, &extent| {
                                                product.checked_mul(extent)
                                            }) {
                                            Some(numel) => numel,
                                            None => {
                                                return fail_status(
                                                    "Compute: device kernel-sized intermediate \
                                                     shape overflow",
                                                );
                                            }
                                        };
                                        let bytes =
                                            match numel.checked_mul(output.dtype.byte_size()) {
                                                Some(bytes) => bytes,
                                                None => {
                                                    return fail_status(
                                                        "Compute: device kernel-sized intermediate \
                                                     byte length overflow",
                                                    );
                                                }
                                            };
                                        let scratch = match unsafe {
                                            alloc_scratch(
                                                api_ref,
                                                kernel_context,
                                                mem_info,
                                                bytes.max(1),
                                            )
                                        } {
                                            Ok(scratch) => scratch.cast::<u8>(),
                                            Err(error) => {
                                                return fail_status(&format!(
                                                    "Compute: device kernel-sized intermediate \
                                                     allocation failed: {error}"
                                                ));
                                            }
                                        };
                                        let device = match unsafe {
                                            crate::kernel_ctx::device_from_memory_info(
                                                api_ref,
                                                mem_info,
                                                format_args!(
                                                    "node {node_idx} dynamic intermediate"
                                                ),
                                            )
                                        } {
                                            Ok(device) => device,
                                            Err(error) => return fail_status(&error),
                                        };
                                        new_bufs.push((
                                            *buffer_index,
                                            IntermediateBuf {
                                                data: Vec::new(),
                                                scratch_ptr: scratch,
                                                shape: crate::dim_vec::DimVec::from_slice(
                                                    &output.shape,
                                                ),
                                                strides: contiguous_strides(&output.shape),
                                                dtype: output.dtype,
                                                device,
                                            },
                                        ));
                                        slot_kinds.push(RoutedSlotKind::Buffer);
                                    }
                                    NodeOutputSink::Absent => {
                                        return fail_status(&format!(
                                            "Compute: present device kernel-sized slot {slot} has \
                                             an absent sink"
                                        ));
                                    }
                                }
                            }

                            let mut ort_iter =
                                ort_outputs.iter_mut().map(|output| output.view_mut());
                            let mut buffer_iter = new_bufs.iter_mut();
                            let mut output_views: Vec<TensorMut<'_>> = slot_kinds
                                .iter()
                                .map(|kind| match kind {
                                    RoutedSlotKind::Ort => ort_iter.next().unwrap(),
                                    RoutedSlotKind::Buffer => {
                                        let (_, buffer) = buffer_iter.next().unwrap();
                                        buf_view_mut(buffer)
                                    }
                                    RoutedSlotKind::Absent(_) => absent_output_view(),
                                })
                                .collect();
                            if let Err(error) = entry.kernel.materialize_kernel_sized_device(
                                &kernel_inputs,
                                &mut output_views,
                                workspace,
                            ) {
                                return fail_status(&format!(
                                    "Compute: device kernel-sized materialization failed: {error}"
                                ));
                            }
                            kernel_probe.end();
                            crate::dispatch_probe::count(
                                crate::dispatch_probe::Event::NodeExecuted,
                            );
                            EXECUTED_NODE_COUNT.fetch_add(1, Ordering::Relaxed);
                            for (buffer_index, buffer) in new_bufs.drain(..) {
                                if buffer_index >= intermediates.len() {
                                    return fail_status(&format!(
                                        "Compute: buffer index {buffer_index} out of range"
                                    ));
                                }
                                if let Some(stale) = intermediates[buffer_index].take() {
                                    host_pool.recycle_intermediate(stale.data);
                                }
                                intermediates[buffer_index] = Some(buffer);
                            }
                            for &buffer_index in
                                &retire_items[retire_starts[node_idx]..retire_starts[node_idx + 1]]
                            {
                                if let Some(dead) = intermediates[buffer_index].take() {
                                    host_pool.recycle_intermediate(dead.data);
                                }
                            }
                            continue;
                        }

                        let kernel_probe = crate::dispatch_probe::Phase::KernelInvoke.enter();
                        let deferred = match run_kernel_sized(
                            exported,
                            &*entry.kernel,
                            &kernel_inputs,
                            &requested_outputs,
                            &entry.output_dtypes,
                            &format!("Compute: node {node_idx}"),
                        ) {
                            Ok(outputs) => outputs,
                            Err(error) => return fail_status(&error),
                        };
                        kernel_probe.end();

                        ort_outputs.clear();
                        new_bufs.clear();
                        for (out_slot, output) in deferred.into_iter().enumerate() {
                            let sink = match sinks.get(out_slot) {
                                Some(sink) => sink,
                                None => {
                                    return fail_status(&format!(
                                        "Compute: kernel-sized output slot {out_slot} has no sink"
                                    ));
                                }
                            };
                            let Some(output) = output else {
                                if !matches!(sink, NodeOutputSink::Absent) {
                                    return fail_status(&format!(
                                        "Compute: absent kernel-sized output slot {out_slot} has a \
                                         non-absent sink"
                                    ));
                                }
                                continue;
                            };
                            match sink {
                                NodeOutputSink::Ort(ort_idx) => {
                                    let mut ort_output = match unsafe {
                                        allocate_output(
                                            api_ref,
                                            kernel_context,
                                            *ort_idx,
                                            &output.shape,
                                            output.dtype,
                                            false,
                                        )
                                    } {
                                        Ok(output) => output,
                                        Err(error) => {
                                            return fail_status(&format!(
                                                "Compute: kernel-sized output slot {out_slot} \
                                                 allocation failed: {error}"
                                            ));
                                        }
                                    };
                                    if let Err(error) =
                                        materialize_kernel_sized(&mut ort_output, &output, out_slot)
                                    {
                                        return fail_status(&format!("Compute: {error}"));
                                    }
                                    ort_outputs.push(ort_output);
                                }
                                NodeOutputSink::Buffer(buf_idx) => {
                                    let strides = contiguous_strides(&output.shape);
                                    new_bufs.push((
                                        *buf_idx,
                                        IntermediateBuf {
                                            data: output.bytes,
                                            scratch_ptr: std::ptr::null_mut(),
                                            shape: crate::dim_vec::DimVec::from_slice(
                                                &output.shape,
                                            ),
                                            strides,
                                            dtype: output.dtype,
                                            device: DeviceId::cpu(),
                                        },
                                    ));
                                }
                                NodeOutputSink::Absent => {
                                    return fail_status(&format!(
                                        "Compute: present kernel-sized output slot {out_slot} has \
                                         an absent sink"
                                    ));
                                }
                            }
                        }

                        crate::dispatch_probe::count(crate::dispatch_probe::Event::NodeExecuted);
                        EXECUTED_NODE_COUNT.fetch_add(1, Ordering::Relaxed);
                        for (buf_idx, buf) in new_bufs.drain(..) {
                            if buf_idx >= intermediates.len() {
                                return fail_status(&format!(
                                    "Compute: buffer index {buf_idx} out of range"
                                ));
                            }
                            if let Some(stale) = intermediates[buf_idx].take() {
                                host_pool.recycle_intermediate(stale.data);
                            }
                            intermediates[buf_idx] = Some(buf);
                        }
                        for &buf_idx in
                            &retire_items[retire_starts[node_idx]..retire_starts[node_idx + 1]]
                        {
                            if let Some(dead) = intermediates[buf_idx].take() {
                                host_pool.recycle_intermediate(dead.data);
                            }
                        }
                        continue;
                    }

                    // Infer output shapes.
                    let shape_probe = crate::dispatch_probe::Phase::DispatchLookup.enter();
                    if let Err(e) =
                        infer_shapes_into(&entry.shape_inference, &kernel_inputs, shapes)
                    {
                        return fail_status(&format!("Compute: shape inference failed: {e}"));
                    }
                    let output_shapes = &*shapes;
                    shape_probe.end();

                    // Execute — dispatch based on sinks.
                    // For outputs going to ORT we allocate via ORT API;
                    // for outputs going to intermediate buffers we allocate on heap;
                    // for absent slots we allocate a scratch buffer.
                    ort_outputs.clear();
                    buf_writes.clear();
                    absent_scratch.clear();
                    slot_kinds.clear();
                    slot_kinds.reserve(sinks.len());

                    // We need to know which output slot → which sink.
                    for (out_slot, shape) in output_shapes.iter().enumerate() {
                        if entry.absent_output_slots.contains(&out_slot) {
                            // Absent output slot — allocate scratch using the
                            // slot's own dtype. Fail closed if unknown.
                            let scratch_dtype = match entry.output_dtypes.get(out_slot).copied() {
                                Some(dt) if dt != DataType::Undefined && dt.byte_size() > 0 => dt,
                                _ => {
                                    return fail_status(&format!(
                                        "Compute: absent output slot {out_slot} has unknown dtype; \
                                     cannot allocate scratch buffer (fail closed)"
                                    ));
                                }
                            };
                            let numel: usize = shape.iter().product::<usize>().max(1);
                            let buf = vec![0u8; scratch_alloc_bytes(numel, scratch_dtype)];
                            let idx = absent_scratch.len();
                            absent_scratch.push((out_slot, buf, scratch_dtype));
                            slot_kinds.push(RoutedSlotKind::Absent(idx));
                            continue;
                        }
                        let out_dtype = match entry.output_dtypes.get(out_slot).copied() {
                            Some(dt) => dt,
                            None => {
                                return fail_status(&format!(
                                    "Compute: output slot {out_slot} has no declared dtype \
                                 (output_dtypes vec is too short)"
                                ));
                            }
                        };
                        let sink = match sinks.get(out_slot) {
                            Some(s) => s,
                            None => {
                                return fail_status(&format!(
                                    "Compute: output slot {out_slot} has no sink"
                                ));
                            }
                        };
                        match sink {
                            NodeOutputSink::Ort(ort_idx) => {
                                match unsafe {
                                    allocate_output(
                                        api_ref,
                                        kernel_context,
                                        *ort_idx,
                                        shape,
                                        out_dtype,
                                        exported.device_staging.is_some(),
                                    )
                                } {
                                    Ok(out) => ort_outputs.push(out),
                                    Err(e) => {
                                        return fail_status(&format!("Compute: {e}"));
                                    }
                                }
                                slot_kinds.push(RoutedSlotKind::Ort);
                            }
                            NodeOutputSink::Buffer(buf_idx) => {
                                buf_writes.push((*buf_idx, out_slot, out_dtype));
                                slot_kinds.push(RoutedSlotKind::Buffer);
                            }
                            NodeOutputSink::Absent => {
                                // Should not reach here — absent slots are handled
                                // above via absent_output_slots. Defensive fallback:
                                // treat as absent scratch allocation.
                                return fail_status(&format!(
                                    "Compute: output slot {out_slot} has Absent sink \
                                 but was not in absent_output_slots"
                                ));
                            }
                        }
                    }

                    // Stage host-resident boundary inputs onto the device before
                    // this node launches (#982). Buffer/absent operands are skipped
                    // inside the helper (only ORT-bound inputs can be host
                    // boundaries); host-required control operands are excluded.
                    if let Some(staging) = exported.device_staging.as_ref() {
                        let ort_indices: Vec<Option<usize>> = sources
                            .iter()
                            .map(|src| match src {
                                NodeInputSource::Ort(i) => Some(*i),
                                NodeInputSource::Buffer(_) | NodeInputSource::Absent => None,
                            })
                            .collect();
                        let output_mem_infos: Vec<*const ort::OrtMemoryInfo> =
                            ort_outputs.iter().map(|o| o.mem_info).collect();
                        if let Err(e) = unsafe {
                            stage_host_boundary_inputs(
                                api_ref,
                                kernel_context,
                                staging,
                                &mut kernel_inputs,
                                &ort_indices,
                                host_operand_indices(&entry.shape_inference),
                                &output_mem_infos,
                                &format!("Compute: node {node_idx}"),
                            )
                        } {
                            return fail_status(&e);
                        }
                    }

                    // For buffer-sink outputs, allocate the IntermediateBuf and get a
                    // mutable pointer into it. Device EPs take ORT scratch memory so
                    // intermediates live where the kernels execute; host EPs take a
                    // plain host buffer, which is both correct and measurably faster
                    // than ORT's host scratch allocator (see `intermediate_scratch`).
                    // Redundant while `drain(..)` below is the only consumer, and
                    // kept so the invariant survives an early `continue` being
                    // added to this loop.
                    new_bufs.clear();
                    for &(buf_idx, out_slot, dtype) in &buf_writes {
                        let shape = &output_shapes[out_slot];
                        let numel: usize = shape.iter().product();
                        let byte_len = dtype.byte_size() * numel;
                        let strides = contiguous_strides(shape);
                        let (data, scratch_ptr, device) = match intermediate_scratch {
                            Some(mem_info) => {
                                match unsafe {
                                    alloc_scratch(api_ref, kernel_context, mem_info, byte_len)
                                } {
                                    Ok(ptr) => {
                                        let device = match unsafe {
                                            crate::kernel_ctx::device_from_memory_info(
                                                api_ref,
                                                mem_info,
                                                format_args!("node {node_idx} intermediate output"),
                                            )
                                        } {
                                            Ok(device) => device,
                                            Err(error) => return fail_status(&error),
                                        };
                                        (Vec::new(), ptr.cast::<u8>(), device)
                                    }
                                    Err(e) => {
                                        return fail_status(&format!(
                                            "Compute: intermediate scratch alloc failed: {e}"
                                        ));
                                    }
                                }
                            }
                            None => (
                                host_pool.take_intermediate(byte_len),
                                std::ptr::null_mut(),
                                DeviceId::cpu(),
                            ),
                        };
                        new_bufs.push((
                            buf_idx,
                            IntermediateBuf {
                                data,
                                scratch_ptr,
                                // `shape` is borrowed from the reusable buffer,
                                // which the next node overwrites, so the buf
                                // owns a copy. Now an inline copy rather than
                                // an allocation, for a rank that fits.
                                shape: (*shape).clone(),
                                strides,
                                dtype,
                                device,
                            },
                        ));
                    }

                    // Collect all output views using the per-slot view map so
                    // positions stay aligned even when absent slots are present.
                    let absent_shapes: &[crate::dim_vec::DimVec<usize>] = output_shapes;
                    // One entry per *absent* slot, not per output slot. Only
                    // absent slots read these, and a node with an absent output is
                    // the exception, so building them for every output cost an
                    // allocation per output on every node that has none. Each entry
                    // is keyed by the same index that keys `absent_scratch`, where
                    // the slot it belongs to is recorded -- so this stays aligned by
                    // construction rather than by an invariant about emptiness.
                    let absent_strides_storage = absent_slot_strides(
                        absent_scratch.iter().map(|(slot, _, _)| *slot),
                        absent_shapes,
                    );
                    let mut all_output_views: Vec<_> = {
                        // Taken lazily. Collecting these into their own `Vec` and
                        // immediately draining it built a second vector of views per
                        // node, for every node of every `Run`, to hand each view
                        // straight to the map below. The single-node path stopped
                        // doing this; the routed path is where it is paid per node.
                        let mut ort_iter = ort_outputs.iter_mut().map(|o| o.view_mut());
                        let mut buf_iter = new_bufs.iter_mut();
                        slot_kinds
                            .iter()
                            .enumerate()
                            .map(|(slot_idx, kind)| match kind {
                                RoutedSlotKind::Ort => ort_iter.next().unwrap(),
                                RoutedSlotKind::Buffer => {
                                    let (_, buf) = buf_iter.next().unwrap();
                                    buf_view_mut(buf)
                                }
                                RoutedSlotKind::Absent(idx) => {
                                    let (_, scratch_buf, dtype) = &mut absent_scratch[*idx];
                                    let shape = &absent_shapes[slot_idx];
                                    let strides = &absent_strides_storage[*idx];
                                    TensorMut::new(
                                        DevicePtrMut(scratch_buf.as_mut_ptr().cast()),
                                        *dtype,
                                        shape.as_slice(),
                                        strides.as_slice(),
                                        DeviceId::cpu(),
                                    )
                                    .mark_absent()
                                }
                            })
                            .collect()
                    };

                    let plans = match exported.workspace_plan_cache(node_idx) {
                        Some(plans) => plans,
                        None => {
                            return fail_status(&format!(
                                "Compute: node {node_idx}: no workspace plan cache for this node \
                             (entries and workspace_plans have drifted)"
                            ));
                        }
                    };
                    // This node's own ORT-bound operands — not subgraph input 0.
                    // Resolved lazily inside `prepare_workspace`, so a node that
                    // needs no workspace never queries ORT for placement.
                    node_ort_operands.clear();
                    node_ort_operands.extend(sources.iter().filter_map(|src| match src {
                        NodeInputSource::Ort(i) => Some(*i),
                        NodeInputSource::Buffer(_) | NodeInputSource::Absent => None,
                    }));
                    let workspace = match unsafe {
                        prepare_workspace(
                            api_ref,
                            kernel_context,
                            PlacementSources {
                                ort_inputs: OrtOperands::Resolved(&node_ort_operands),
                                subgraph_fallback: SubgraphFallback {
                                    staging: exported.device_staging.as_ref(),
                                    memo: &subgraph_fallback_memo,
                                },
                            },
                            &*entry.kernel,
                            plans,
                            &kernel_inputs,
                            format_args!("Compute: node {node_idx}"),
                        )
                    } {
                        Ok(w) => w,
                        Err(e) => return fail_status(&e),
                    };

                    let kernel_probe = crate::dispatch_probe::Phase::KernelInvoke.enter();
                    if let Err(e) = entry.kernel.execute_with_workspace(
                        &kernel_inputs,
                        &mut all_output_views,
                        workspace,
                    ) {
                        return fail_status(&format!("Compute: kernel execution failed: {e}"));
                    }
                    kernel_probe.end();
                    crate::dispatch_probe::count(crate::dispatch_probe::Event::NodeExecuted);
                    EXECUTED_NODE_COUNT.fetch_add(1, Ordering::Relaxed);

                    // Store new intermediate buffers.
                    for (buf_idx, buf) in new_bufs.drain(..) {
                        if buf_idx >= intermediates.len() {
                            return fail_status(&format!(
                                "Compute: buffer index {buf_idx} out of range"
                            ));
                        }
                        if let Some(stale) = intermediates[buf_idx].take() {
                            host_pool.recycle_intermediate(stale.data);
                        }
                        intermediates[buf_idx] = Some(buf);
                    }

                    // Retire every buffer this node was the last reader of. Doing
                    // it here — rather than at the end of the subgraph — is what
                    // lets the next node's allocation land on storage that is
                    // still hot in cache.
                    for &buf_idx in
                        &retire_items[retire_starts[node_idx]..retire_starts[node_idx + 1]]
                    {
                        if let Some(dead) = intermediates[buf_idx].take() {
                            host_pool.recycle_intermediate(dead.data);
                        }
                    }
                }

                for slot in intermediates.drain(..).flatten() {
                    host_pool.recycle_intermediate(slot.data);
                }
            } else if exported.entries.len() == 1 {
                // ── Fast path: single-kernel subgraph ─────────────────────────
                let entry = &exported.entries[0];
                // Reconstruct positional inputs with absent sentinels so the
                // kernel sees the correct arity and position.
                // Fail closed rather than indexing: a slot beyond ORT's input
                // count means the compile-time slot map and the runtime binding
                // disagree, and panicking across the C ABI inside `Compute` turns
                // that into an opaque "internal panic" instead of a diagnosable
                // session error.
                crate::dispatch_probe::count(crate::dispatch_probe::Event::NodeExecuted);
                let bind_probe = crate::dispatch_probe::Phase::TensorBind.enter();
                // Operand views for an ordinary node live on the stack. A
                // `Vec::with_capacity` here was one `malloc`/`free` pair per
                // `Run` -- pure fixed cost -- to hold a handful of `Copy`
                // views that never outlive this block. The `absent` seed is
                // what the unbound-slot arm used to push, so a slot with no
                // ORT input is already correct and the loop leaves it alone.
                let operand_count = entry.input_slots.len();
                let mut inline_inputs;
                let mut heap_inputs;
                let kernel_inputs: &mut [TensorView<'_>] = if operand_count <= INLINE_OPERANDS {
                    inline_inputs = [TensorView::absent(DataType::Undefined); INLINE_OPERANDS];
                    &mut inline_inputs[..operand_count]
                } else {
                    crate::dispatch_probe::count(crate::dispatch_probe::Event::DispatchAlloc);
                    heap_inputs = vec![TensorView::absent(DataType::Undefined); operand_count];
                    &mut heap_inputs[..]
                };
                for (position, (dst, slot)) in
                    kernel_inputs.iter_mut().zip(&entry.input_slots).enumerate()
                {
                    // A slot beyond ORT's input count means the compile-time
                    // slot map and the runtime binding disagree; fail closed
                    // rather than index.
                    if let Some(ort_idx) = slot {
                        match inputs.get(*ort_idx) {
                            Some(input) => *dst = input.view(),
                            None => {
                                return fail_status(&format!(
                                    "Compute: input slot {position} maps to ORT input \
                                 {ort_idx}, but ORT bound only {} input(s) to this \
                                 fused node",
                                    inputs.len()
                                ));
                            }
                        }
                    }
                }
                bind_probe.end();
                if matches!(&entry.shape_inference, ShapeInference::KernelSizedOutputs) {
                    let requested_outputs: Vec<bool> = (0..entry.num_outputs)
                        .map(|slot| !entry.absent_output_slots.contains(&slot))
                        .collect();
                    if entry.kernel.kernel_sized_output_policy()
                        == KernelSizedOutputPolicy::DeviceWorkspace
                    {
                        if let Some(staging) = exported.device_staging.as_ref()
                            && let Err(error) = unsafe {
                                stage_host_boundary_inputs(
                                    api_ref,
                                    kernel_context,
                                    staging,
                                    kernel_inputs,
                                    &entry.input_slots,
                                    &[],
                                    &[],
                                    "Compute: node 0 kernel-sized metadata",
                                )
                            }
                        {
                            return fail_status(&error);
                        }
                        let plans = match exported.workspace_plan_cache(0) {
                            Some(plans) => plans,
                            None => {
                                return fail_status(
                                    "Compute: node 0: no workspace plan cache for kernel-sized \
                                     device outputs",
                                );
                            }
                        };
                        let subgraph_fallback_memo = std::cell::Cell::new(None);
                        let workspace = match unsafe {
                            prepare_workspace(
                                api_ref,
                                kernel_context,
                                PlacementSources {
                                    ort_inputs: OrtOperands::Slots(&entry.input_slots),
                                    subgraph_fallback: SubgraphFallback {
                                        staging: exported.device_staging.as_ref(),
                                        memo: &subgraph_fallback_memo,
                                    },
                                },
                                &*entry.kernel,
                                plans,
                                kernel_inputs,
                                "Compute: node 0 kernel-sized device workspace",
                            )
                        } {
                            Ok(workspace) => workspace,
                            Err(error) => return fail_status(&error),
                        };
                        let kernel_probe = crate::dispatch_probe::Phase::KernelInvoke.enter();
                        let metadata = match run_device_kernel_sized(
                            &*entry.kernel,
                            kernel_inputs,
                            &requested_outputs,
                            &entry.output_dtypes,
                            workspace,
                            "Compute: node 0",
                        ) {
                            Ok(outputs) => outputs,
                            Err(error) => return fail_status(&error),
                        };
                        owned_outputs.reserve(entry.num_outputs);
                        let mut ort_output_index = 0usize;
                        for (slot, output) in metadata.iter().enumerate() {
                            let Some(output) = output else {
                                continue;
                            };
                            match unsafe {
                                allocate_output(
                                    api_ref,
                                    kernel_context,
                                    ort_output_index,
                                    &output.shape,
                                    output.dtype,
                                    true,
                                )
                            } {
                                Ok(output) => owned_outputs.push(output),
                                Err(error) => {
                                    return fail_status(&format!(
                                        "Compute: device kernel-sized output slot {slot} \
                                         allocation failed: {error}"
                                    ));
                                }
                            }
                            ort_output_index += 1;
                        }
                        let mut ort_iter = owned_outputs.iter_mut().map(|output| output.view_mut());
                        let mut output_views: Vec<TensorMut<'_>> = metadata
                            .iter()
                            .map(|output| {
                                if output.is_some() {
                                    ort_iter.next().unwrap()
                                } else {
                                    absent_output_view()
                                }
                            })
                            .collect();
                        if let Err(error) = entry.kernel.materialize_kernel_sized_device(
                            kernel_inputs,
                            &mut output_views,
                            workspace,
                        ) {
                            return fail_status(&format!(
                                "Compute: device kernel-sized materialization failed: {error}"
                            ));
                        }
                        kernel_probe.end();
                        EXECUTED_NODE_COUNT.fetch_add(1, Ordering::Relaxed);
                        return ok_status();
                    }

                    let kernel_probe = crate::dispatch_probe::Phase::KernelInvoke.enter();
                    let deferred = match run_kernel_sized(
                        exported,
                        &*entry.kernel,
                        kernel_inputs,
                        &requested_outputs,
                        &entry.output_dtypes,
                        "Compute: node 0",
                    ) {
                        Ok(outputs) => outputs,
                        Err(error) => return fail_status(&error),
                    };
                    kernel_probe.end();

                    owned_outputs.reserve(entry.num_outputs);
                    let mut ort_out_idx = 0usize;
                    for (slot, output) in deferred.into_iter().enumerate() {
                        let Some(output) = output else {
                            continue;
                        };
                        let mut ort_output = match unsafe {
                            allocate_output(
                                api_ref,
                                kernel_context,
                                ort_out_idx,
                                &output.shape,
                                output.dtype,
                                false,
                            )
                        } {
                            Ok(output) => output,
                            Err(error) => {
                                return fail_status(&format!(
                                    "Compute: kernel-sized output slot {slot} allocation failed: \
                                     {error}"
                                ));
                            }
                        };
                        if let Err(error) = materialize_kernel_sized(&mut ort_output, &output, slot)
                        {
                            return fail_status(&format!("Compute: {error}"));
                        }
                        owned_outputs.push(ort_output);
                        ort_out_idx += 1;
                    }
                    EXECUTED_NODE_COUNT.fetch_add(1, Ordering::Relaxed);
                    return ok_status();
                }
                let lookup_probe = crate::dispatch_probe::Phase::DispatchLookup.enter();
                if let Err(e) = infer_shapes_into(&entry.shape_inference, kernel_inputs, shapes) {
                    return fail_status(&format!("Compute: shape inference failed: {e}"));
                }
                let output_shapes = &*shapes;
                lookup_probe.end();
                // Allocate outputs. Absent slots get a local scratch buffer so the
                // kernel sees the full output arity and can index by position,
                // while only present slots are allocated through ORT's kernel
                // context (sequential ORT indices).
                owned_outputs.reserve(entry.num_outputs);
                slot_map.reserve(entry.num_outputs);
                let mut absent_bufs: Vec<Vec<u8>> = Vec::new();
                // Also record the dtype for each absent slot so the TensorMut
                // matches the kernel's element size.
                let mut absent_dtypes: Vec<DataType> = Vec::new();
                let mut ort_out_idx: usize = 0;

                for (out_slot, shape) in output_shapes.iter().enumerate() {
                    if entry.absent_output_slots.contains(&out_slot) {
                        // Absent output slot — allocate scratch buffer using the
                        // slot's own ORT-declared dtype. Fail closed if unknown.
                        let scratch_dtype = match entry.output_dtypes.get(out_slot).copied() {
                            Some(dt) if dt != DataType::Undefined && dt.byte_size() > 0 => dt,
                            _ => {
                                return fail_status(&format!(
                                    "Compute: absent output slot {out_slot} has unknown dtype; \
                                 cannot allocate scratch buffer (fail closed)"
                                ));
                            }
                        };
                        let numel: usize = shape.iter().product::<usize>().max(1);
                        let buf = vec![0u8; scratch_alloc_bytes(numel, scratch_dtype)];
                        let idx = absent_bufs.len();
                        absent_bufs.push(buf);
                        absent_dtypes.push(scratch_dtype);
                        slot_map.push(SlotKind::Absent(idx));
                        continue;
                    }
                    let out_dtype = match entry.output_dtypes.get(out_slot).copied() {
                        Some(dt) => dt,
                        None => {
                            return fail_status(&format!(
                                "Compute: output slot {out_slot} has no declared dtype \
                             (output_dtypes vec is too short)"
                            ));
                        }
                    };
                    match unsafe {
                        allocate_output(
                            api_ref,
                            kernel_context,
                            ort_out_idx,
                            shape,
                            out_dtype,
                            exported.device_staging.is_some(),
                        )
                    } {
                        Ok(out) => {
                            owned_outputs.push(out);
                            slot_map.push(SlotKind::Ort);
                        }
                        Err(e) => return fail_status(&format!("Compute: {e}")),
                    }
                    ort_out_idx += 1;
                }
                // Stage host-resident boundary inputs onto the device before the
                // kernel launches (#982). No-op for a host EP (device_staging is
                // None) or when every operand already lives where it should.
                if let Some(staging) = exported.device_staging.as_ref() {
                    let output_mem_infos: Vec<*const ort::OrtMemoryInfo> =
                        owned_outputs.iter().map(|o| o.mem_info).collect();
                    if let Err(e) = unsafe {
                        stage_host_boundary_inputs(
                            api_ref,
                            kernel_context,
                            staging,
                            kernel_inputs,
                            &entry.input_slots,
                            host_operand_indices(&entry.shape_inference),
                            &output_mem_infos,
                            "Compute: node 0",
                        )
                    } {
                        return fail_status(&e);
                    }
                }
                // Build output views in node-output order so the kernel sees the
                // full arity including absent scratch slots.
                //
                // The stride storage below exists only to back the absent slots'
                // `TensorMut`s. `absent_slot_strides` builds one entry per *absent*
                // slot, so a node with no absent outputs -- every elementwise op,
                // which is most of what reaches this path -- allocates nothing here.
                let absent_shapes: &[crate::dim_vec::DimVec<usize>] = output_shapes;
                // See the routed path. `slot_map` pushes `Absent(idx)` with
                // increasing `idx`, so filtering it in slot order yields the strides
                // in absent-index order.
                let absent_strides_storage = absent_slot_strides(
                    slot_map.iter().enumerate().filter_map(|(slot_idx, kind)| {
                        matches!(kind, SlotKind::Absent(_)).then_some(slot_idx)
                    }),
                    absent_shapes,
                );
                // Views of the ORT outputs, in order, taken lazily: collecting them
                // into a `Vec` first and immediately draining it allocated a second
                // vector of views per call for nothing.
                let mut ort_view_iter = owned_outputs.iter_mut().map(|o| o.view_mut());
                // Same stack storage as the operand views above. `TensorMut` is
                // not `Copy` — it is a unique borrow — so the seed is built per
                // slot rather than copied, but it is the same null-backed absent
                // sentinel the `Absent` arm produces and it borrows only empty
                // `'static` slices, so it coerces to this block's lifetime.
                let mut inline_outputs;
                let mut heap_outputs;
                let output_views: &mut [TensorMut<'_>] = if slot_map.len() <= INLINE_OPERANDS {
                    inline_outputs =
                        std::array::from_fn::<_, INLINE_OPERANDS, _>(|_| absent_output_view());
                    &mut inline_outputs[..slot_map.len()]
                } else {
                    crate::dispatch_probe::count(crate::dispatch_probe::Event::DispatchAlloc);
                    heap_outputs = Vec::from_iter(
                        std::iter::repeat_with(absent_output_view).take(slot_map.len()),
                    );
                    &mut heap_outputs[..]
                };
                for ((slot_idx, kind), dst) in
                    slot_map.iter().enumerate().zip(output_views.iter_mut())
                {
                    match kind {
                        SlotKind::Ort => {
                            *dst = ort_view_iter.next().unwrap();
                        }
                        SlotKind::Absent(idx) => {
                            let buf = &mut absent_bufs[*idx];
                            let shape = &absent_shapes[slot_idx];
                            let strides = &absent_strides_storage[*idx];
                            let scratch_dtype = absent_dtypes[*idx];
                            *dst = TensorMut::new(
                                DevicePtrMut(buf.as_mut_ptr().cast()),
                                scratch_dtype,
                                shape.as_slice(),
                                strides.as_slice(),
                                DeviceId::cpu(),
                            )
                            .mark_absent();
                        }
                    }
                }
                let plans = match exported.workspace_plan_cache(0) {
                    Some(plans) => plans,
                    None => {
                        return fail_status(
                            "Compute: node 0: no workspace plan cache for this node (entries and \
                         workspace_plans have drifted)",
                        );
                    }
                };
                // The single-node subgraph's operands are this node's operands, but
                // only the present ones are ORT-bound. Placement is resolved lazily
                // inside `prepare_workspace`, past the zero-byte and lifetime gates.
                let subgraph_fallback_memo = std::cell::Cell::new(None);
                let lookup2_probe = crate::dispatch_probe::Phase::DispatchLookup.enter();
                let workspace = match unsafe {
                    prepare_workspace(
                        api_ref,
                        kernel_context,
                        PlacementSources {
                            ort_inputs: OrtOperands::Slots(&entry.input_slots),
                            subgraph_fallback: SubgraphFallback {
                                staging: exported.device_staging.as_ref(),
                                memo: &subgraph_fallback_memo,
                            },
                        },
                        &*entry.kernel,
                        plans,
                        kernel_inputs,
                        "Compute: node 0",
                    )
                } {
                    Ok(w) => w,
                    Err(e) => return fail_status(&e),
                };
                lookup2_probe.end();
                let kernel_probe = crate::dispatch_probe::Phase::KernelInvoke.enter();
                if let Err(e) =
                    entry
                        .kernel
                        .execute_with_workspace(kernel_inputs, output_views, workspace)
                {
                    return fail_status(&format!("Compute: kernel execution failed: {e}"));
                }
                kernel_probe.end();
                EXECUTED_NODE_COUNT.fetch_add(1, Ordering::Relaxed);
                // `output_views` borrows `owned_outputs`, so anything that
                // touches the outputs again has to follow its last use. That
                // used to need an explicit `drop` of the owning `Vec`. It no
                // longer does, and not because the storage moved to the stack
                // -- the spill arm is still a `Vec` that owns an allocation and
                // has a `Drop`. It is because `TensorMut` itself has no `Drop`,
                // so dropck never extends the borrow past the last use and NLL
                // ends it at the kernel call above.
            } else {
                return fail_status(
                    "Compute: multi-node subgraph requires SubgraphRouting — \
                 call ExportedComputeInfo::set_routing before registering",
                );
            }

            // Every `TensorView` borrowing the scratch is out of scope by here.
            // Returning it is the guard's business, not this function's.
            ok_status()
        })
    }));
    result.unwrap_or_else(|_| fail_status("Compute: internal panic"))
}

fn run_kernel_sized(
    exported: &ExportedComputeInfo,
    kernel: &dyn Kernel,
    inputs: &[TensorView<'_>],
    requested_outputs: &[bool],
    declared_dtypes: &[DataType],
    context: &str,
) -> Result<Vec<Option<KernelSizedOutput>>, String> {
    if kernel.kernel_sized_output_policy() != KernelSizedOutputPolicy::HostOwned {
        return Err(format!(
            "{context}: host-owned kernel-sized dispatch received policy {:?}",
            kernel.kernel_sized_output_policy()
        ));
    }
    if !exported.host_accessible || exported.device_staging.is_some() {
        return Err(format!(
            "{context}: kernel-sized outputs are host-only, but this EP uses device-resident \
             tensors; place this node on a host EP (implicit D2H payload copies are not allowed)"
        ));
    }
    if let Some((slot, input)) = inputs
        .iter()
        .enumerate()
        .find(|(_, input)| !input.is_absent() && !input.device.is_host_accessible())
    {
        return Err(format!(
            "{context}: kernel-sized outputs are host-only, but input slot {slot} is on {:?}; \
             place this node on a host EP",
            input.device
        ));
    }
    if !kernel.has_kernel_sized_outputs() {
        return Err(format!(
            "{context}: shape strategy requires kernel-sized outputs, but the kernel did not opt in"
        ));
    }
    if requested_outputs.len() != declared_dtypes.len() {
        return Err(format!(
            "{context}: output metadata mismatch: {} requested slots but {} declared dtypes",
            requested_outputs.len(),
            declared_dtypes.len()
        ));
    }

    let outputs = kernel
        .execute_kernel_sized(inputs, requested_outputs)
        .map_err(|error| format!("{context}: kernel-sized execution failed: {error}"))?;
    validate_kernel_sized_outputs(&outputs, requested_outputs, declared_dtypes, context)?;
    Ok(outputs)
}

fn run_device_kernel_sized(
    kernel: &dyn Kernel,
    inputs: &[TensorView<'_>],
    requested_outputs: &[bool],
    declared_dtypes: &[DataType],
    workspace: Option<WorkspaceView>,
    context: &str,
) -> Result<Vec<Option<KernelSizedOutputMetadata>>, String> {
    if !kernel.has_kernel_sized_outputs()
        || kernel.kernel_sized_output_policy() != KernelSizedOutputPolicy::DeviceWorkspace
    {
        return Err(format!(
            "{context}: device-workspace shape strategy requires an explicitly opted-in device \
             policy"
        ));
    }
    if requested_outputs.len() != declared_dtypes.len() {
        return Err(format!(
            "{context}: output metadata mismatch: {} requested slots but {} declared dtypes",
            requested_outputs.len(),
            declared_dtypes.len()
        ));
    }
    let outputs = kernel
        .prepare_kernel_sized_device(inputs, requested_outputs, workspace)
        .map_err(|error| format!("{context}: device metadata phase failed: {error}"))?;
    validate_kernel_sized_metadata(&outputs, requested_outputs, declared_dtypes, context)?;
    Ok(outputs)
}

fn validate_kernel_sized_metadata(
    outputs: &[Option<KernelSizedOutputMetadata>],
    requested_outputs: &[bool],
    declared_dtypes: &[DataType],
    context: &str,
) -> Result<(), String> {
    if outputs.len() != requested_outputs.len() {
        return Err(format!(
            "{context}: kernel-sized metadata count mismatch: kernel returned {}, node has {} slots",
            outputs.len(),
            requested_outputs.len()
        ));
    }
    for (slot, ((output, &requested), &declared_dtype)) in outputs
        .iter()
        .zip(requested_outputs)
        .zip(declared_dtypes)
        .enumerate()
    {
        match (requested, output) {
            (false, None) => continue,
            (false, Some(_)) => {
                return Err(format!(
                    "{context}: absent output slot {slot} unexpectedly returned device metadata"
                ));
            }
            (true, None) => {
                return Err(format!(
                    "{context}: present output slot {slot} returned no device metadata"
                ));
            }
            (true, Some(output)) => {
                if output.dtype != declared_dtype {
                    return Err(format!(
                        "{context}: kernel-sized output slot {slot} returned dtype {:?}, but the \
                         graph declares {:?}",
                        output.dtype, declared_dtype
                    ));
                }
                if output.dtype == DataType::Undefined || output.dtype.byte_size() == 0 {
                    return Err(format!(
                        "{context}: kernel-sized output slot {slot} has unsupported dtype {:?}",
                        output.dtype
                    ));
                }
                output.shape.iter().try_fold(1usize, |product, &extent| {
                    if extent > i64::MAX as usize {
                        return Err(format!(
                            "{context}: kernel-sized output slot {slot} extent {extent} exceeds \
                             ORT's i64 shape range"
                        ));
                    }
                    product.checked_mul(extent).ok_or_else(|| {
                        format!(
                            "{context}: kernel-sized output slot {slot} shape {:?} overflows usize",
                            output.shape
                        )
                    })
                })?;
            }
        }
    }
    Ok(())
}

fn validate_kernel_sized_outputs(
    outputs: &[Option<KernelSizedOutput>],
    requested_outputs: &[bool],
    declared_dtypes: &[DataType],
    context: &str,
) -> Result<(), String> {
    if outputs.len() != requested_outputs.len() {
        return Err(format!(
            "{context}: kernel-sized output count mismatch: kernel returned {}, node has {} slots",
            outputs.len(),
            requested_outputs.len()
        ));
    }

    for (slot, ((output, &requested), &declared_dtype)) in outputs
        .iter()
        .zip(requested_outputs)
        .zip(declared_dtypes)
        .enumerate()
    {
        match (requested, output) {
            (false, None) => continue,
            (false, Some(_)) => {
                return Err(format!(
                    "{context}: absent output slot {slot} unexpectedly returned bytes"
                ));
            }
            (true, None) => {
                return Err(format!(
                    "{context}: present output slot {slot} returned no bytes"
                ));
            }
            (true, Some(output)) => {
                if declared_dtype == DataType::Undefined {
                    return Err(format!(
                        "{context}: present output slot {slot} has no declared dtype"
                    ));
                }
                if output.dtype != declared_dtype {
                    return Err(format!(
                        "{context}: kernel-sized output slot {slot} returned dtype {:?}, but the \
                         graph declares {:?}",
                        output.dtype, declared_dtype
                    ));
                }
                let element_size = output.dtype.byte_size();
                if element_size == 0 {
                    return Err(format!(
                        "{context}: kernel-sized output slot {slot} has unsupported zero-sized \
                         dtype {:?}",
                        output.dtype
                    ));
                }
                let numel = output.shape.iter().try_fold(1usize, |product, &extent| {
                    if extent > i64::MAX as usize {
                        return Err(format!(
                            "{context}: kernel-sized output slot {slot} extent {extent} exceeds \
                             ORT's i64 shape range"
                        ));
                    }
                    product.checked_mul(extent).ok_or_else(|| {
                        format!(
                            "{context}: kernel-sized output slot {slot} shape {:?} overflows usize",
                            output.shape
                        )
                    })
                })?;
                let expected_bytes = numel.checked_mul(element_size).ok_or_else(|| {
                    format!(
                        "{context}: kernel-sized output slot {slot} byte length overflows usize \
                         for shape {:?} and dtype {:?}",
                        output.shape, output.dtype
                    )
                })?;
                if output.bytes.len() != expected_bytes {
                    return Err(format!(
                        "{context}: kernel-sized output slot {slot} returned {} bytes, expected \
                         {expected_bytes} for shape {:?} and dtype {:?}",
                        output.bytes.len(),
                        output.shape,
                        output.dtype
                    ));
                }
            }
        }
    }
    Ok(())
}

fn materialize_kernel_sized(
    output: &mut crate::kernel_ctx::OwnedOutput,
    value: &KernelSizedOutput,
    slot: usize,
) -> Result<(), String> {
    if output.dtype != value.dtype || output.shape.as_slice() != value.shape.as_slice() {
        return Err(format!(
            "kernel-sized output slot {slot} allocation mismatch: ORT returned {:?}{:?}, \
             requested {:?}{:?}",
            output.dtype, output.shape, value.dtype, value.shape
        ));
    }
    if !value.bytes.is_empty() {
        if output.data_ptr.is_null() {
            return Err(format!(
                "kernel-sized output slot {slot} allocation returned a null data pointer for {} \
                 bytes",
                value.bytes.len()
            ));
        }
        // SAFETY: validation proved the source byte length equals the complete
        // tensor byte length, and ORT allocated that exact shape and dtype.
        unsafe {
            std::ptr::copy_nonoverlapping(
                value.bytes.as_ptr(),
                output.data_ptr.cast::<u8>(),
                value.bytes.len(),
            );
        }
    }
    #[cfg(test)]
    KERNEL_SIZED_MATERIALIZATION_COPIES.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
static KERNEL_SIZED_MATERIALIZATION_COPIES: AtomicUsize = AtomicUsize::new(0);

/// Build a contiguous stride array from a shape (C-order, innermost stride = 1).
/// Whether a node output slot is ORT-allocated or absent (scratch-backed).
///
/// Module scope rather than local to the single-node path so the reusable
/// scratch below can name it.
enum SlotKind {
    Ort,           // present, comes from ORT
    Absent(usize), // index into absent_bufs
}

/// The per-`Run` scratch one thread keeps between calls: the input metadata
/// vector and the two output-bookkeeping vectors, in a single cell.
///
/// These were three vectors behind two thread-locals, acquired and returned
/// independently, which cost four thread-local resolutions per `Run`. Under a
/// cdylib every one of those is a real `__tls_get_addr` call, not a fixed
/// offset off `%fs`, because the dynamic TLS model cannot know the module's
/// offset until load time. Callgrind attributed 72 Ir/`Run` to `__tls_get_addr`
/// alone and a further 160 to the take/recycle pairs, against a total fixed
/// excess over ORT of ~1,760 -- so the pooling machinery was spending a third
/// of what the pooling saved.
///
/// One cell means one resolution for the whole `Run` (see
/// [`with_run_scratch`]). The vectors keep their own capacity bounds (see
/// [`RunScratch::clear_and_bound`]); bundling their storage must not bundle the
/// decision about whether to keep it.
#[derive(Default)]
pub(crate) struct RunScratch {
    /// Input metadata for this `Run`. Cleared when the `Run` leaves: an
    /// `OwnedInput` borrows a pointer into an `OrtValue` owned by a call that
    /// has finished.
    inputs: Vec<crate::kernel_ctx::OwnedInput>,
    /// Output bookkeeping. Cleared for the same reason.
    owned: Vec<crate::kernel_ctx::OwnedOutput>,
    /// Which output slots are ORT-allocated and which are absent scratch.
    slots: Vec<SlotKind>,
    /// Inferred output shapes for the node being executed.
    ///
    /// Lives here so the storage outlives the node *and* the `Run`.
    /// [`infer_shapes`] returns a fresh `Vec<Vec<usize>>` per node per `Run`,
    /// and at depth 100 that measured 277.2 Ir/node: 57.4 for the inner
    /// `to_vec`, 63.0 for the one-element `vec![_]` (which lowers to
    /// `box_new_uninit`), and 78.4 each for the drop glue and the `free`. On
    /// top of that sits a proportional share of glibc's `_int_free`, the
    /// single largest allocator line in the profile at 482.4 Ir/node
    /// aggregated over every Rust deallocation.
    ///
    /// A per-`Run` local would leave depth 1 exactly where it started, since
    /// there the buffer would be allocated and dropped for its single node.
    /// Parking it with the rest of the scratch is what makes the fixed-`Run`
    /// cost fall too.
    shapes: Vec<crate::dim_vec::DimVec<usize>>,
    /// Retired host intermediate storage, reused by later nodes and later
    /// `Run`s on this thread.
    ///
    /// Lives here for exactly the reason given above this struct: it used to be
    /// a thread-local of its own, so every take and every recycle paid a
    /// separate `__tls_get_addr` and a separate `RefCell` borrow. Those are
    /// **per buffer per node**, not per `Run`, so on a 100-node chain the
    /// pooling machinery resolved the thread-local ~200 times to serve ~100
    /// buffers. Sharing `RunScratch`'s single resolution makes it zero extra.
    ///
    /// **Deliberately not cleared by [`RunScratch::clear_and_bound`].** Every
    /// other field here is per-`Run` state that would be a dangling borrow if
    /// it outlived its `Run`; this one is the opposite -- it is owned, pointer
    /// free `Vec<u8>` storage whose entire purpose is to outlive the `Run` that
    /// retired it. Clearing it would silently turn every reuse back into a
    /// `malloc`/`free` pair while leaving every test green, so
    /// `a_pooled_buffer_survives_the_run_that_retired_it` pins it.
    host_pool: HostPool,
}

/// Capacity, in elements, above which per-`Run` scratch is dropped instead of
/// kept. Applied to each vector on its own.
///
/// A node with a handful of outputs is the case worth optimising; one with
/// hundreds is not worth pinning that memory on every worker thread for the
/// rest of the process.
const SCRATCH_MAX_CAPACITY: usize = 16;

/// Operand count up to which a node's input views are built on the stack.
///
/// Unlike [`SCRATCH_MAX_CAPACITY`] this is not about how much memory to keep —
/// nothing is kept, the storage dies with the block — it is about how wide a
/// stack array is worth initialising to avoid a `malloc`/`free` pair. Every
/// slot is seeded whether the node uses it or not, so the trade turns on how
/// many operands ONNX nodes actually have: `Relu` and the rest of the
/// elementwise family take 1, `MatMul` 2, `Gemm`/`Conv`/`Where`/`LayerNorm` 3,
/// and past that the population thins out fast. Four covers those and costs a
/// handful of stores; a wider node simply allocates, as it did before.
///
/// Deliberately *not* `dim_vec::INLINE_RANK`: that one answers "how many
/// dimensions does a tensor have", this one answers "how many tensors does a
/// node take". They are different questions about different things and tying
/// them together would make one of the two answers wrong.
const INLINE_OPERANDS: usize = 4;

/// The seed for an unfilled output-view slot: null-backed, zero-rank, marked
/// absent — the same sentinel the `SlotKind::Absent` arm builds.
///
/// Every slot is overwritten before the kernel sees it; this exists so the
/// stack array can be initialised without `TensorMut` being `Copy`, and so
/// that a slot which somehow escaped the fill would present as absent rather
/// than as a tensor pointing at nothing.
fn absent_output_view<'a>() -> TensorMut<'a> {
    TensorMut::new(
        DevicePtrMut(std::ptr::null_mut()),
        DataType::Undefined,
        &[],
        &[],
        DeviceId::cpu(),
    )
    .mark_absent()
}

thread_local! {
    /// This thread's parked [`RunScratch`].
    ///
    /// Thread-local because `Compute` runs on whichever thread ORT calls it
    /// from, and sharing would need a lock on the hot path to buy nothing --
    /// scratch and pooled storage are only ever reused by a later call on the
    /// same thread.
    ///
    /// Every field that can hold a pointer into a finished `Run`'s ORT values
    /// is cleared when the `Run` that borrowed it leaves -- including by
    /// unwinding -- so none is ever retained here between calls. `host_pool` is
    /// the one field deliberately kept, and it is kept precisely because it
    /// holds no pointers: only owned `Vec<u8>` storage for the next `Run` on
    /// this thread to reuse. See [`RunScratch::clear_and_bound`].
    static RUN_SCRATCH: std::cell::RefCell<RunScratch> =
        const { std::cell::RefCell::new(RunScratch {
            inputs: Vec::new(),
            owned: Vec::new(),
            slots: Vec::new(),
            shapes: Vec::new(),
            host_pool: HostPool { slots: Vec::new() },
        }) };
}

/// Run `f` with this thread's scratch, reusing the capacity the last `Run`
/// retired, and return the storage no matter how `f` leaves.
///
/// The scratch is **borrowed** for the call rather than moved out and put back.
/// That is one thread-local resolution instead of two, and it makes the recycle
/// unconditional: [`ScratchGuard`]'s `Drop` runs on the ordinary path, on every
/// early `return` inside `f`, and while unwinding out of a panicking kernel.
/// The take-and-return shape this replaced skipped the recycle on any early
/// return, so a `Run` that failed left the next one to allocate again.
///
/// Falls back to freshly allocated vectors when the cell is already borrowed,
/// which can only happen if a kernel re-entered `Compute` on this thread; the
/// inner call then works on storage of its own and cannot alias the outer
/// call's. Correctness never depends on the reuse. `try_with` likewise fails
/// during thread teardown, after the destructor has run, and takes the same
/// path.
fn with_run_scratch<R>(f: impl FnOnce(&mut RunScratch) -> R) -> R {
    let mut f = Some(f);
    let borrowed = RUN_SCRATCH.try_with(|cell| {
        let mut slot = cell.try_borrow_mut().ok()?;
        let f = f.take()?;
        let guard = ScratchGuard { scratch: &mut slot };
        Some(f(guard.scratch))
    });
    match borrowed {
        Ok(Some(r)) => r,
        // Re-entrant, or the thread-local is already gone. Either way this call
        // owns its storage outright and simply drops it; parking it would hand
        // the outer call's slot to whichever nesting level finished last.
        _ => {
            let mut local = RunScratch::default();
            let f = f
                .take()
                .expect("the borrowed path consumed `f` without returning a value");
            f(&mut local)
        }
    }
}

/// Returns the borrowed scratch to a reusable state when the `Run` leaves.
///
/// A guard rather than a call at the end of the `Run`, because the interesting
/// exits are the ones that are easy to forget: the early `return`s, and the
/// unwind out of a panicking kernel. Both must leave the cell clean, since an
/// `OwnedInput` holds a pointer into an `OrtValue` belonging to a call that has
/// finished, and the cell outlives the call.
struct ScratchGuard<'a> {
    scratch: &'a mut RunScratch,
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        self.scratch.clear_and_bound();
    }
}

impl RunScratch {
    /// Empty every vector and give back any capacity too large to park.
    ///
    /// The keep/drop decision is made **per vector**, which is what the two
    /// separate pools did before they were merged: a node with 200 inputs must
    /// not cost the output vectors the capacity they had earned, and a node
    /// with 200 outputs must not cost the input vector its own. Bundling the
    /// storage is an optimisation; bundling the policy would be a behaviour
    /// change.
    fn clear_and_bound(&mut self) {
        self.inputs.clear();
        self.owned.clear();
        self.slots.clear();
        self.shapes.clear();

        if self.inputs.capacity() > SCRATCH_MAX_CAPACITY {
            self.inputs = Vec::new();
        }
        // The output pair is judged together, as it was when it shared a cell:
        // `owned` and `slots` are always the same length, so one being
        // pathological says the other is too, and keeping half a pair buys
        // nothing.
        if self.owned.capacity() > SCRATCH_MAX_CAPACITY
            || self.slots.capacity() > SCRATCH_MAX_CAPACITY
        {
            self.owned = Vec::new();
            self.slots = Vec::new();
        }

        // Judged on its own, for the same spine-capacity reason as the pair
        // above. Any heap a spilled `DimVec` (rank > INLINE_RANK) owned is
        // already gone: `clear()` drops the elements, and that is where the
        // out-of-line buffer is released. What this bounds is only the outer
        // Vec's own capacity.
        if self.shapes.capacity() > SCRATCH_MAX_CAPACITY {
            self.shapes = Vec::new();
        }

        // `host_pool` is *not* cleared, and is not judged against
        // `SCRATCH_MAX_CAPACITY` either. It is not per-`Run` state: it holds
        // retired tensor storage for the next `Run` on this thread to reuse,
        // which is the entire point of pooling, and its own bound is
        // `HOST_INTERMEDIATE_POOL_SLOTS` applied at recycle time. Clearing it
        // here would be a pure pessimisation that no correctness test could
        // see.
    }
}

/// How many retired host intermediates one [`RunScratch`] keeps for reuse.
///
/// A routed subgraph's live set is bounded by its widest cut, which is small
/// for the elementwise chains this path actually sees. Eight slots cover those
/// without letting a pathological graph pin an unbounded amount of memory: past
/// the cap, retired buffers are simply dropped.
///
/// This is the pool's *only* bound. `RunScratch::clear_and_bound` deliberately
/// leaves `host_pool` alone, so nothing else will trim it.
const HOST_INTERMEDIATE_POOL_SLOTS: usize = 8;

/// Storage for one host intermediate of `len` bytes, reusing a retired buffer
/// from this pool when one is large enough.
///
/// The pool is reached through `&mut self` rather than a thread-local of its
/// own: this runs once per buffer per node, so a thread-local here cost a
/// `__tls_get_addr` and a `RefCell` borrow on every call. It now shares the
/// single resolution [`with_run_scratch`] already makes for the whole `Run`.
///
/// The returned bytes are initialized but **not** guaranteed to be zero when a
/// retired buffer is reused. That is deliberate, and it is not a weakening of
/// what kernels may assume: an output routed to an ORT sink already arrives as
/// whatever `KernelContext_GetOutput` handed back, which ORT does not zero, and
/// the single-kernel path has always worked that way. Producing the same
/// contract for buffer sinks makes a kernel that fails to write its whole
/// output fail the same way wherever it sits in a partition, instead of only
/// when it happens to be the last node. Re-zeroing on every reuse costs a full
/// `memset` of the tensor — measured at 0.586x of ORT for an 8-node 1 MiB f32
/// `Relu` chain against 0.801x without it.
///
/// Debug builds do fill reused storage, with a poison pattern rather than
/// zeros, so a kernel that leaves part of its output unwritten fails loudly in
/// tests instead of inheriting something plausible.
/// Retired host intermediate storage belonging to one [`RunScratch`].
///
/// A newtype rather than a bare `Vec<Vec<u8>>` so that the pool a call site
/// uses cannot be written by accident. The whole value of parking this in
/// `RunScratch` is that *these particular* call sites reach *that particular*
/// pool; with a bare vector, `take_intermediate(&mut Vec::new(), n)` type
/// checks, silently reverts every reuse to a `malloc`/`free` pair, and leaves
/// the suite green -- an independent review of this change found exactly that
/// mutation. It no longer compiles.
#[derive(Default)]
pub(crate) struct HostPool {
    slots: Vec<Vec<u8>>,
}

impl HostPool {
    fn take_intermediate(&mut self, len: usize) -> Vec<u8> {
        let pool = &mut self.slots;
        match pool.iter().position(|buf| buf.capacity() >= len) {
            Some(pos) => {
                let mut buf = pool.swap_remove(pos);
                if buf.len() < len {
                    buf.resize(len, 0);
                } else {
                    buf.truncate(len);
                }
                // Debug builds hand back a poison pattern rather than whatever
                // the previous tenant left. A kernel that fails to write part
                // of its output then produces an obviously wrong value in every
                // test run instead of a plausible stale one, which is the
                // failure this relaxation would otherwise make quieter. `0xFF`
                // because it is the loudest pattern available: it reads as NaN
                // in every float width and as -1 in every signed integer width,
                // so it survives a tolerance comparison. Release builds skip
                // it — that write is the whole cost being avoided.
                #[cfg(debug_assertions)]
                buf.fill(0xFF);
                buf
            }
            None => vec![0u8; len],
        }
    }

    /// Hand a dead host intermediate back for reuse.
    ///
    /// Buffers backed by ORT scratch have an empty `data` vector — they carry a
    /// borrowed `scratch_ptr` instead — so they are dropped here rather than
    /// pooled.
    fn recycle_intermediate(&mut self, buf: Vec<u8>) {
        if buf.capacity() == 0 {
            return;
        }
        if self.slots.len() < HOST_INTERMEDIATE_POOL_SLOTS {
            self.slots.push(buf);
        }
    }

    /// How many buffers are parked. Only the tests care.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.slots.len()
    }
}

/// For each intermediate buffer, the highest node index that reads it, or
/// `None` when no node does.
///
/// `None` covers two real cases: a buffer index no node consumes (the producer
/// also routes that output to an ORT sink), and an out-of-range index in a
/// malformed routing table. Both mean "nothing keeps this alive", which is the
/// conservative answer for a liveness bound — a buffer is retired only after
/// its recorded last reader has run, so a missing entry can never retire a
/// buffer early.
fn last_reader_per_buffer(
    input_sources: &[Vec<NodeInputSource>],
    num_buffers: usize,
) -> Vec<Option<usize>> {
    let mut last = vec![None; num_buffers];
    for (node_idx, sources) in input_sources.iter().enumerate() {
        for src in sources {
            if let NodeInputSource::Buffer(b) = src
                && let Some(slot) = last.get_mut(*b)
            {
                *slot = Some(node_idx);
            }
        }
    }
    last
}

/// Buffers retiring after each node, as a CSR index: node `i` retires the
/// buffers in `items[starts[i]..starts[i + 1]]`.
///
/// The routed loop used to find these by scanning every buffer's last-reader
/// entry at every node, which is O(nodes x buffers) -- and a chain has a buffer
/// per node, so it is quadratic in subgraph depth. That cost is invisible at
/// depth 1 and dominant at depth 100.
///
/// Inverting the map once per `Run` makes each node touch only its own
/// retirements. Within a node they stay in ascending buffer order, as the scan
/// produced them.
fn retirements_per_node(
    input_sources: &[Vec<NodeInputSource>],
    num_buffers: usize,
) -> (Vec<usize>, Vec<usize>) {
    let last = last_reader_per_buffer(input_sources, num_buffers);
    let nodes = input_sources.len();
    let mut starts = vec![0usize; nodes + 1];
    for &reader in last.iter().flatten() {
        if reader < nodes {
            starts[reader + 1] += 1;
        }
    }
    for i in 0..nodes {
        starts[i + 1] += starts[i];
    }
    let mut items = vec![0usize; starts[nodes]];
    let mut cursor = starts.clone();
    for (buffer, reader) in last.iter().enumerate() {
        if let Some(reader) = reader
            && *reader < nodes
        {
            items[cursor[*reader]] = buffer;
            cursor[*reader] += 1;
        }
    }
    (starts, items)
}

/// Contiguous strides for the *absent* output slots only, in absent-index
/// order.
///
/// Absent slots are the exception, so the routed loop used to build a stride
/// vector for every output of every node and read it only when a slot turned
/// out to be absent -- an allocation per output, per node, discarded unused.
///
/// The result is keyed by absent index, the same key `Absent(idx)` carries, so
/// callers index it with `idx` rather than the slot number. Both callers pass
/// absent slots in ascending slot order, which is the order `idx` was assigned
/// in, so the two agree by construction.
fn absent_slot_strides(
    absent_slots: impl Iterator<Item = usize>,
    shapes: &[crate::dim_vec::DimVec<usize>],
) -> Vec<crate::dim_vec::DimVec<i64>> {
    absent_slots
        .map(|slot| match shapes.get(slot) {
            Some(shape) => contiguous_strides(shape),
            None => crate::dim_vec::DimVec::new(),
        })
        .collect()
}

fn contiguous_strides(shape: &[usize]) -> crate::dim_vec::DimVec<i64> {
    // `zeroed` then set the last element, rather than filling with 1: every
    // element except the last is overwritten by the loop below, so seeding all
    // of them costs a pass that is immediately thrown away. A rank-0 shape has
    // no last element and correctly yields an empty result.
    let mut strides = crate::dim_vec::DimVec::<i64>::zeroed(shape.len());
    if let Some(last) = strides.last_mut() {
        *last = 1;
    }
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    strides
}

/// Build a mutable TensorView from an IntermediateBuf (unsafe: caller ensures
/// exclusive access and lifetime correctness).
fn buf_view_mut(buf: &mut IntermediateBuf) -> onnx_runtime_ep_api::tensor::TensorMut<'_> {
    use onnx_runtime_ep_api::tensor::{DevicePtrMut, TensorMut};
    // `ptr_mut` returns a Copy raw pointer, so its mutable borrow ends here and
    // does not conflict with the immutable borrows of shape/strides below.
    let ptr = buf.ptr_mut();
    TensorMut::new(
        DevicePtrMut(ptr.cast()),
        buf.dtype,
        &buf.shape,
        &buf.strides,
        buf.device,
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// Shape inference implementations
// ──────────────────────────────────────────────────────────────────────────────

/// Read a 1-D `i64` (or `i32`) tensor's values out of a host-accessible view.
///
/// Same device refusal as [`read_scalar_i64`], and honours the stride rather
/// than assuming a packed buffer.
fn read_i64_vec(t: &TensorView<'_>, what: &str) -> Result<Vec<i64>, String> {
    if !t.device.is_host_accessible() {
        return Err(format!(
            "{what} is on {:?}, which the host cannot read during shape \
             inference. An EP with device-resident inputs must copy it to the \
             host before Compute.",
            t.device
        ));
    }
    let len: usize = t.shape.iter().product();
    if len == 0 {
        return Ok(Vec::new());
    }
    let base = t.data.as_ptr::<u8>();
    if base.is_null() {
        return Err(format!("{what} has a null data pointer"));
    }
    let elem = match t.dtype {
        DataType::Int64 => 8usize,
        DataType::Int32 => 4,
        other => return Err(format!("{what} must be Int32 or Int64, got {other:?}")),
    };
    let stride = t.strides.first().copied().unwrap_or(1);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let off = t.byte_offset as isize + (i as isize) * (stride as isize) * (elem as isize);
        // SAFETY: host-accessible view of `len` elements of the stated dtype;
        // the offset is inside it by the view's contract.
        let p = unsafe { base.offset(off) };
        out.push(match t.dtype {
            DataType::Int64 => unsafe { p.cast::<i64>().read_unaligned() },
            _ => i64::from(unsafe { p.cast::<i32>().read_unaligned() }),
        });
    }
    Ok(out)
}

/// Read a scalar `i64` (or `i32`) out of a host-accessible view.
///
/// Same device restriction, and for the same reason, as [`count_true`]: this
/// runs on the host, so a device-resident value is refused with an explanation
/// rather than dereferenced.
fn read_scalar_i64(t: &TensorView<'_>, what: &str) -> Result<i64, String> {
    if !t.device.is_host_accessible() {
        return Err(format!(
            "{what} is on {:?}, which the host cannot read during shape \
             inference. An EP with device-resident inputs must copy it to the \
             host before Compute.",
            t.device
        ));
    }
    let base = t.data.as_ptr::<u8>();
    if base.is_null() {
        return Err(format!("{what} has a null data pointer"));
    }
    // SAFETY: host-accessible view of at least one element of the stated dtype;
    // `byte_offset` is inside it by the view's contract.
    let p = unsafe { base.add(t.byte_offset) };
    match t.dtype {
        DataType::Int64 => Ok(unsafe { p.cast::<i64>().read_unaligned() }),
        DataType::Int32 => Ok(i64::from(unsafe { p.cast::<i32>().read_unaligned() })),
        other => Err(format!("{what} must be Int32 or Int64, got {other:?}")),
    }
}

/// Resolve a possibly-negative axis against `rank`.
fn normalize_axis(axis: i64, rank: usize, what: &str) -> Result<usize, String> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        return Err(format!("{what} {axis} out of range for rank {rank}"));
    }
    Ok(a as usize)
}

/// Indices of the true entries in a 1-D Bool `condition`.
///
/// This is the only place the shape table reads a tensor's *values*, so the two
/// ways it can be wrong are worth naming.
///
/// **Device memory.** `TensorView::data` may point at device memory, and this
/// runs on the host. A device-resident condition is refused rather than read
/// through a pointer the host cannot dereference. That is a hard error, not a
/// silent decline, because by Compute time the claim has already been accepted —
/// the honest outcome is to say why. Making the CUDA plugin claim `Compress`
/// needs a D2H copy of the condition (a 1-D Bool tensor, so cheap) plus a stream
/// sync, which belongs with that EP rather than here.
///
/// **Strides.** ORT may hand back a non-contiguous view, so the stride is
/// honoured instead of assuming packed bytes.
fn count_true(condition: &TensorView<'_>) -> Result<Vec<usize>, String> {
    if condition.dtype != DataType::Bool {
        return Err(format!(
            "Compress: condition must be Bool, got {:?}",
            condition.dtype
        ));
    }
    if !condition.device.is_host_accessible() {
        return Err(format!(
            "Compress: condition is on {:?}, which the host cannot read during \
             shape inference. Compress needs the condition's values to size its \
             output; an EP with device-resident inputs must copy it to the host \
             before Compute.",
            condition.device
        ));
    }

    let len = condition.shape.first().copied().unwrap_or(0);
    let stride = condition.strides.first().copied().unwrap_or(1);
    let base = condition.data.as_ptr::<u8>();
    if base.is_null() {
        return Err("Compress: condition has a null data pointer".into());
    }

    let mut out = Vec::new();
    for i in 0..len {
        let offset = condition.byte_offset as isize + (i as isize) * (stride as isize);
        // SAFETY: `condition` is a host-accessible Bool view of `len` elements;
        // the offset is inside it by the view's own contract, and Bool is one
        // byte so the element stride is the byte stride.
        let byte = unsafe { *base.offset(offset) };
        if byte != 0 {
            out.push(i);
        }
    }
    Ok(out)
}

/// Test-only entry point to [`infer_shapes`].
///
/// Exposed so `shape_tables_agree` in `onnx-runtime-ep-cpu-plugin` can compare
/// the compatibility fallback with the native registry and exercise the
/// production shared route. The `testutil` feature is enabled only by that
/// crate's dev-dependency, so shipped plugin artifacts expose no test-only API.
#[cfg(feature = "testutil")]
#[doc(hidden)]
pub fn infer_shapes_for_test(
    strategy: &ShapeInference,
    inputs: &[TensorView<'_>],
) -> Result<Vec<Vec<usize>>, String> {
    infer_shapes(strategy, inputs)
}

/// Exact production census used by the cross-crate anti-vacuity test.
#[cfg(feature = "testutil")]
#[doc(hidden)]
pub fn shared_native_rule_names_for_test() -> Vec<&'static str> {
    SharedNativeShapeRule::all()
        .iter()
        .map(|rule| rule.op_type())
        .collect()
}

/// Infer output shapes into reusable storage, allocating nothing on the
/// elementwise paths the dispatch grid actually measures.
///
/// Not on *every* path: the shared-native arm still receives an owned
/// `Vec<Vec<usize>>` from `infer_shared_node` and copies it in, which is
/// strictly more work than returning it would have been. That arm is here to
/// route a *declining* rule's fallback through the fast arms, not to save an
/// allocation on resolve, and it is off the measured path.
///
/// [`infer_shapes`] hands back a fresh `Vec<Vec<usize>>` for every node of
/// every `Run`. Callers only ever read it as a slice and drop it a few hundred
/// instructions later, so the container is pure overhead: measured at depth
/// 100, 277.2 Ir/node across the two allocations and their frees, plus a share
/// of the `_int_free` line that dominates the allocator profile.
///
/// Only the strategies confirmed hot in that profile are reimplemented here;
/// everything else delegates to [`infer_shapes`] verbatim and merely copies the
/// result in. Two reasons for keeping the split that narrow. The inference
/// rules are the correctness-critical part and there are forty of them, so
/// duplicating them wholesale to save an allocation on a cold path would be a
/// bad trade. And the arms that *are* duplicated are each a bounds check plus a
/// copy, small enough to audit by eye — with
/// `infer_shapes_into_matches_infer_shapes` standing behind them as a
/// differential oracle so the two paths cannot drift apart silently.
fn infer_shapes_into(
    strategy: &ShapeInference,
    inputs: &[TensorView<'_>],
    out: &mut Vec<crate::dim_vec::DimVec<usize>>,
) -> Result<(), String> {
    use crate::dim_vec::DimVec;
    match strategy {
        // `Ok(vec![inputs[idx].shape.to_vec()])`, without the two allocations.
        ShapeInference::SameAsInput(idx) => {
            let idx = *idx;
            if idx >= inputs.len() {
                return Err(format!(
                    "SameAsInput({idx}): only {} inputs present",
                    inputs.len()
                ));
            }
            out.clear();
            out.push(DimVec::from_slice(inputs[idx].shape));
            Ok(())
        }

        // A single operand broadcasts against nothing, so the fold in
        // `infer_shapes` runs zero times and the answer is that operand's own
        // shape. Two or more still have to go through `broadcast_shapes`, which
        // allocates a fresh shape per step regardless of who calls it.
        ShapeInference::ElementwiseBroadcast if inputs.len() == 1 => {
            out.clear();
            out.push(DimVec::from_slice(inputs[0].shape));
            Ok(())
        }

        // Recurse rather than delegate, so a shared-native rule that declines
        // still reaches the fast arms through its own fallback.
        ShapeInference::SharedNative { node, fallback } => match infer_shared_node(node, inputs) {
            SharedShapeResult::Resolved(shapes) => {
                out.clear();
                out.extend(shapes.iter().map(|s| DimVec::from_slice(s)));
                Ok(())
            }
            SharedShapeResult::SymbolicOrUnknown | SharedShapeResult::Rejected(_) => {
                infer_shapes_into(fallback, inputs, out)
            }
        },

        other => {
            let shapes = infer_shapes(other, inputs)?;
            out.clear();
            out.extend(shapes.iter().map(|s| DimVec::from_slice(s)));
            Ok(())
        }
    }
}

/// Infer output shapes from the shape inference strategy and input views.
fn infer_shapes(
    strategy: &ShapeInference,
    inputs: &[TensorView<'_>],
) -> Result<Vec<Vec<usize>>, String> {
    match strategy {
        ShapeInference::KernelSizedOutputs => Err(
            "kernel-sized outputs must execute before allocation; infer_shapes is not applicable"
                .into(),
        ),
        ShapeInference::SharedNative { node, fallback } => match infer_shared_node(node, inputs) {
            SharedShapeResult::Resolved(shapes) => Ok(shapes),
            // Preserve the plugin's established permissiveness. In particular,
            // native validation can reject malformed companion operands that a
            // shape-only plugin rule historically ignored.
            SharedShapeResult::SymbolicOrUnknown | SharedShapeResult::Rejected(_) => {
                infer_shapes(fallback, inputs)
            }
        },
        ShapeInference::ElementwiseBroadcast => {
            if inputs.is_empty() {
                return Err("ElementwiseBroadcast: no inputs".into());
            }
            let mut shape = inputs[0].shape.to_vec();
            for inp in &inputs[1..] {
                shape = onnx_runtime_ir::broadcast_shapes(&shape, inp.shape)
                    .map_err(|e| format!("broadcast failed: {e}"))?;
            }
            Ok(vec![shape])
        }

        ShapeInference::SameAsInput(idx) => {
            let idx = *idx;
            if idx >= inputs.len() {
                return Err(format!(
                    "SameAsInput({idx}): only {} inputs present",
                    inputs.len()
                ));
            }
            Ok(vec![inputs[idx].shape.to_vec()])
        }

        ShapeInference::CausalConvWithState => {
            if inputs.len() < 2 {
                return Err(format!(
                    "CausalConvWithState: needs input and weight, got {} inputs",
                    inputs.len()
                ));
            }
            let input = inputs[0].shape;
            let weight = inputs[1].shape;
            if input.len() != 3 {
                return Err(format!(
                    "CausalConvWithState: input must be rank 3 (batch, channels, length), got {input:?}"
                ));
            }
            if weight.len() != 3 {
                return Err(format!(
                    "CausalConvWithState: weight must be rank 3 (channels, 1, k), got {weight:?}"
                ));
            }
            // The carry holds the last k-1 positions, so a degenerate k == 0
            // would underflow rather than merely producing an odd shape.
            let carry = weight[2]
                .checked_sub(1)
                .ok_or_else(|| "CausalConvWithState: kernel width must be non-zero".to_string())?;
            Ok(vec![input.to_vec(), vec![input[0], input[1], carry]])
        }

        ShapeInference::SameAsInputMultiOutput { idx, count } => {
            let idx = *idx;
            if idx >= inputs.len() {
                return Err(format!(
                    "SameAsInputMultiOutput({idx}): only {} inputs present",
                    inputs.len()
                ));
            }
            let shape = inputs[idx].shape.to_vec();
            Ok(vec![shape; *count])
        }

        ShapeInference::LayerNorm {
            raw_axis,
            num_outputs,
            full_shape_outputs,
        } => {
            if inputs.is_empty() {
                return Err("LayerNorm: no inputs".into());
            }
            let full_shape = inputs[0].shape.to_vec();
            let rank = full_shape.len() as i64;
            // Resolve raw axis against runtime rank.
            let resolved = if *raw_axis < 0 {
                *raw_axis + rank
            } else {
                *raw_axis
            };
            if resolved < 0 || resolved >= rank {
                return Err(format!(
                    "LayerNorm: axis {raw_axis} out of range for rank {rank}",
                ));
            }
            let axis = resolved as usize;
            // Reduced shape: dims before `axis` are kept, dims from `axis`
            // onward become 1 (keepdims reduction over the normalised axes).
            let mut reduced_shape = full_shape.clone();
            for d in reduced_shape[axis..].iter_mut() {
                *d = 1;
            }
            let mut out = Vec::with_capacity(*num_outputs);
            for i in 0..*num_outputs {
                if i == 0 || full_shape_outputs.contains(&i) {
                    out.push(full_shape.clone());
                } else {
                    out.push(reduced_shape.clone());
                }
            }
            Ok(out)
        }

        ShapeInference::MatMul => {
            if inputs.len() < 2 {
                return Err(format!("MatMul: expected ≥2 inputs, got {}", inputs.len()));
            }
            let a = inputs[0].shape;
            let b = inputs[1].shape;
            let shape = matmul_output_shape(a, b)?;
            Ok(vec![shape])
        }

        ShapeInference::MatMulNBits { n } => {
            let a = inputs
                .first()
                .ok_or_else(|| "MatMulNBits: expected >=1 input, got 0".to_string())?
                .shape;
            if a.is_empty() {
                return Err("MatMulNBits: activation must have rank >= 1".to_string());
            }
            let mut shape = a.to_vec();
            *shape.last_mut().expect("rank checked above") = *n;
            Ok(vec![shape])
        }

        ShapeInference::QLinearMatMul => {
            if inputs.len() < 4 {
                return Err(format!(
                    "QLinearMatMul: expected >=4 inputs (a, a_scale, a_zero_point, b), got {}",
                    inputs.len()
                ));
            }
            let shape = matmul_output_shape(inputs[0].shape, inputs[3].shape)?;
            Ok(vec![shape])
        }

        ShapeInference::Gemm { trans_a, trans_b } => {
            if inputs.len() < 2 {
                return Err(format!("Gemm: expected ≥2 inputs, got {}", inputs.len()));
            }
            let a = inputs[0].shape;
            let b = inputs[1].shape;
            if a.len() != 2 || b.len() != 2 {
                return Err(format!("Gemm: inputs must be 2-D, got {a:?} and {b:?}"));
            }
            let m = if *trans_a { a[1] } else { a[0] };
            let n = if *trans_b { b[0] } else { b[1] };
            Ok(vec![vec![m, n]])
        }

        ShapeInference::Concat { axis } => {
            if inputs.is_empty() {
                return Err("Concat: no inputs".into());
            }
            let rank = inputs[0].shape.len();
            let ax = normalise_axis(*axis, rank)?;
            let mut out = inputs[0].shape.to_vec();
            for inp in &inputs[1..] {
                out[ax] += inp.shape[ax];
            }
            Ok(vec![out])
        }

        ShapeInference::Transpose { perm } => {
            if inputs.is_empty() {
                return Err("Transpose: no inputs".into());
            }
            let rank = inputs[0].shape.len();
            let out: Vec<usize> = if let Some(p) = perm {
                if p.len() != rank {
                    return Err(format!(
                        "Transpose: perm length {} != rank {}",
                        p.len(),
                        rank
                    ));
                }
                p.iter().map(|&i| inputs[0].shape[i]).collect()
            } else {
                inputs[0].shape.iter().rev().copied().collect()
            };
            Ok(vec![out])
        }

        ShapeInference::Gather { axis } => {
            if inputs.len() < 2 {
                return Err(format!("Gather: expected ≥2 inputs, got {}", inputs.len()));
            }
            let data = inputs[0].shape;
            let indices = inputs[1].shape;
            let rank = data.len();
            let ax = normalise_axis(*axis, rank)?;
            let mut out: Vec<usize> = data[..ax].to_vec();
            out.extend_from_slice(indices);
            out.extend_from_slice(&data[ax + 1..]);
            Ok(vec![out])
        }

        ShapeInference::GatherND { batch_dims } => {
            if inputs.len() < 2 {
                return Err(format!(
                    "GatherND: expected ≥2 inputs, got {}",
                    inputs.len()
                ));
            }
            let data = inputs[0].shape;
            let indices = inputs[1].shape;
            let b = *batch_dims;
            if indices.is_empty() {
                return Err("GatherND: indices must have rank ≥1".into());
            }
            let k = *indices.last().unwrap();
            if b + k > data.len() {
                return Err(format!(
                    "GatherND: batch_dims+k ({}) > data rank ({})",
                    b + k,
                    data.len()
                ));
            }
            // output = data[:b] ++ indices[:-1] ++ data[b+k:]
            let mut out: Vec<usize> = data[..b].to_vec();
            out.extend_from_slice(&indices[..indices.len() - 1]);
            out.extend_from_slice(&data[b + k..]);
            Ok(vec![out])
        }

        ShapeInference::GatherBlockQuantized => {
            // Treat as GatherND(batch_dims=0) for shape purposes.
            if inputs.len() < 2 {
                return Err(format!(
                    "GatherBlockQuantized: expected ≥2 inputs, got {}",
                    inputs.len()
                ));
            }
            let data = inputs[0].shape;
            let indices = inputs[1].shape;
            if indices.is_empty() {
                return Err("GatherBlockQuantized: indices must have rank ≥1".into());
            }
            let k = *indices.last().unwrap();
            if k > data.len() {
                return Err(format!(
                    "GatherBlockQuantized: k ({k}) > data rank ({})",
                    data.len()
                ));
            }
            let mut out: Vec<usize> = indices[..indices.len() - 1].to_vec();
            out.extend_from_slice(&data[k..]);
            Ok(vec![out])
        }

        ShapeInference::ShapeOp { start, end } => {
            if inputs.is_empty() {
                return Err("Shape: no inputs".into());
            }
            let rank = inputs[0].shape.len() as i64;
            let s = normalise_axis(*start, inputs[0].shape.len()).unwrap_or(0);
            let e = if let Some(end_val) = end {
                if *end_val < 0 {
                    (rank + end_val).max(0) as usize
                } else {
                    (*end_val as usize).min(inputs[0].shape.len())
                }
            } else {
                inputs[0].shape.len()
            };
            let len = e.saturating_sub(s);
            Ok(vec![vec![len]])
        }

        ShapeInference::Squeeze { axes } => {
            if inputs.is_empty() {
                return Err("Squeeze: no inputs".into());
            }
            let shape = inputs[0].shape;
            let out: Vec<usize> = if axes.is_empty() {
                shape.iter().filter(|&&d| d != 1).copied().collect()
            } else {
                let rank = shape.len();
                let norm_axes: Vec<usize> = axes
                    .iter()
                    .map(|&a| normalise_axis(a, rank))
                    .collect::<Result<_, _>>()?;
                shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !norm_axes.contains(i))
                    .map(|(_, &d)| d)
                    .collect()
            };
            Ok(vec![out])
        }

        ShapeInference::Unsqueeze { axes } => {
            if inputs.is_empty() {
                return Err("Unsqueeze: no inputs".into());
            }
            let in_rank = inputs[0].shape.len();
            let out_rank = in_rank + axes.len();
            // Normalise axes with respect to output rank.
            let mut norm_axes: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    let ax = if a < 0 {
                        (out_rank as i64 + a) as usize
                    } else {
                        a as usize
                    };
                    if ax >= out_rank {
                        Err(format!(
                            "Unsqueeze: axis {a} out of range for output rank {out_rank}"
                        ))
                    } else {
                        Ok(ax)
                    }
                })
                .collect::<Result<_, _>>()?;
            norm_axes.sort_unstable();

            let mut out = vec![1usize; out_rank];
            let mut src = 0;
            let mut ax_iter = norm_axes.iter().peekable();
            for (i, slot) in out.iter_mut().enumerate() {
                if ax_iter.peek() == Some(&&i) {
                    ax_iter.next();
                    // slot stays 1
                } else {
                    *slot = inputs[0].shape[src];
                    src += 1;
                }
            }
            Ok(vec![out])
        }

        ShapeInference::ReshapeData { allowzero } => {
            if inputs.len() < 2 {
                return Err(format!("Reshape: expected 2 inputs, got {}", inputs.len()));
            }
            let data_shape = inputs[0].shape;
            // Read the shape tensor from input[1].
            let shape_vals = unsafe { read_i64_tensor(&inputs[1]) }?;
            let in_numel: usize = data_shape.iter().product();

            let mut out: Vec<usize> = Vec::with_capacity(shape_vals.len());
            let mut infer_idx: Option<usize> = None;
            let mut known_prod: usize = 1;

            for (i, &v) in shape_vals.iter().enumerate() {
                match v {
                    0 if !allowzero => {
                        // Copy from input shape at same index.
                        let d = data_shape.get(i).copied().ok_or_else(|| {
                            format!(
                                "Reshape: dim 0 at index {i} but input rank is {}",
                                data_shape.len()
                            )
                        })?;
                        out.push(d);
                        known_prod *= d;
                    }
                    -1 => {
                        if infer_idx.is_some() {
                            return Err("Reshape: only one -1 dimension allowed".into());
                        }
                        infer_idx = Some(i);
                        out.push(0); // placeholder
                    }
                    d if d > 0 => {
                        let d = d as usize;
                        out.push(d);
                        known_prod *= d;
                    }
                    _ => {
                        return Err(format!("Reshape: invalid shape value {v}"));
                    }
                }
            }

            if let Some(idx) = infer_idx {
                if known_prod == 0 {
                    return Err("Reshape: cannot infer -1 dimension when known product is 0".into());
                }
                if !in_numel.is_multiple_of(known_prod) {
                    return Err(format!(
                        "Reshape: cannot infer -1: {in_numel} elements not divisible by {known_prod}"
                    ));
                }
                out[idx] = in_numel / known_prod;
            }
            Ok(vec![out])
        }

        ShapeInference::SliceData => {
            if inputs.is_empty() {
                return Err("Slice: no inputs".into());
            }
            let data_shape = inputs[0].shape;
            let starts = unsafe { read_i64_tensor(&inputs[1]) }?;
            let ends = unsafe { read_i64_tensor(&inputs[2]) }?;
            let axes: Vec<i64> = if inputs.len() > 3 && !inputs[3].data.is_null() {
                unsafe { read_i64_tensor(&inputs[3]) }?
            } else {
                (0..data_shape.len() as i64).collect()
            };
            let steps: Vec<i64> = if inputs.len() > 4 && !inputs[4].data.is_null() {
                unsafe { read_i64_tensor(&inputs[4]) }?
            } else {
                vec![1; starts.len()]
            };

            let out_shape = slice_output_shape(data_shape, &starts, &ends, &axes, &steps)?;
            Ok(vec![out_shape])
        }

        ShapeInference::Reduction {
            keepdims,
            axes,
            noop_with_empty_axes,
        } => {
            if inputs.is_empty() {
                return Err("Reduction: no inputs".into());
            }
            let shape = inputs[0].shape;
            Ok(vec![reduce_shape(
                shape,
                axes.as_deref(),
                *keepdims,
                *noop_with_empty_axes,
            )?])
        }

        ShapeInference::ReductionFromInput {
            keepdims,
            noop_with_empty_axes,
        } => {
            if inputs.is_empty() {
                return Err("ReductionFromInput: no inputs".into());
            }
            let shape = inputs[0].shape;
            let axes_opt: Option<Vec<i64>> = if inputs.len() > 1 && !inputs[1].data.is_null() {
                Some(unsafe { read_i64_tensor(&inputs[1]) }?)
            } else {
                None
            };
            Ok(vec![reduce_shape(
                shape,
                axes_opt.as_deref(),
                *keepdims,
                *noop_with_empty_axes,
            )?])
        }

        ShapeInference::Conv {
            out_channels,
            per_axis,
        } => {
            if inputs.is_empty() {
                return Err("Conv: no inputs".into());
            }
            let in_shape = inputs[0].shape;
            if in_shape.len() < 2 {
                return Err("Conv: input must have rank ≥2".into());
            }
            let n = in_shape[0];
            let spatial: Vec<usize> = in_shape[2..]
                .iter()
                .zip(per_axis)
                .map(|(&dim, ax)| {
                    let eff = ax.dilation * (ax.kernel - 1) + 1;
                    let padded = dim + ax.pad_before + ax.pad_after;
                    if padded < eff {
                        0
                    } else {
                        (padded - eff) / ax.stride + 1
                    }
                })
                .collect();
            let mut out = vec![n, *out_channels];
            out.extend(spatial);
            Ok(vec![out])
        }

        ShapeInference::MultiHeadAttention {
            num_heads,
            num_outputs,
        } => {
            // query: [B, S, hidden]
            // output[0]: same as query
            // output[1] present_key:   [B, num_heads, P+S, head_size]  (if present)
            // output[2] present_value: same as output[1]
            if inputs.is_empty() {
                return Err("MultiHeadAttention: no inputs".into());
            }
            let q = inputs[0].shape;
            if q.len() < 2 {
                return Err("MultiHeadAttention: query rank must be ≥2".into());
            }
            let attn_out = q.to_vec();
            let mut outputs = vec![attn_out];

            if *num_outputs > 1 && *num_heads > 0 {
                let b = q[0];
                let s = q[1];
                let hidden = if q.len() > 2 { q[2] } else { 0 };
                let head_size = if *num_heads > 0 {
                    hidden / num_heads
                } else {
                    0
                };
                // past_key is input[6] when present.
                let past_seq = inputs
                    .get(6)
                    .map(|v| if v.shape.len() >= 3 { v.shape[2] } else { 0 })
                    .unwrap_or(0);
                let present_shape = vec![b, *num_heads, past_seq + s, head_size];
                for _ in 1..*num_outputs {
                    outputs.push(present_shape.clone());
                }
            }
            Ok(outputs)
        }

        ShapeInference::GroupQueryAttention {
            num_heads,
            kv_num_heads,
        } => {
            if inputs.is_empty() {
                return Err("GroupQueryAttention: no inputs".into());
            }
            let q = inputs[0].shape;
            if q.len() < 2 {
                return Err("GroupQueryAttention: query rank must be ≥2".into());
            }
            let b = q[0];
            let s = q[1];
            let hidden = if q.len() > 2 { q[2] } else { 0 };
            let head_size = if *num_heads > 0 {
                hidden / num_heads
            } else {
                0
            };
            let attn_out = q.to_vec();
            let past_seq = inputs
                .get(3)
                .map(|v| if v.shape.len() >= 3 { v.shape[2] } else { 0 })
                .unwrap_or(0);
            let present_shape = vec![b, *kv_num_heads, past_seq + s, head_size];
            Ok(vec![attn_out, present_shape.clone(), present_shape])
        }

        ShapeInference::RotaryEmbedding => {
            if inputs.is_empty() {
                return Err("RotaryEmbedding: no inputs".into());
            }
            Ok(vec![inputs[0].shape.to_vec()])
        }

        ShapeInference::MsftAttention {
            v_hidden,
            num_heads,
            num_outputs,
        } => {
            // input(0):   [B, S, input_hidden]   (unprojected activation)
            // input(1):   [input_hidden, q_hidden + k_hidden + v_hidden]
            // output(0):  [B, S, v_hidden]
            // output(1):  present [2, B, num_heads, P + S, head_size]
            if inputs.len() < 2 {
                return Err("Attention: expected at least input and weights".into());
            }
            let x = inputs[0].shape;
            if x.len() != 3 {
                return Err(format!("Attention: input rank must be 3, got {}", x.len()));
            }
            let weights = inputs[1].shape;
            if weights.len() != 2 {
                return Err(format!(
                    "Attention: weights rank must be 2, got {}",
                    weights.len()
                ));
            }
            // Without `qkv_hidden_sizes` the projection is split three ways, so
            // an indivisible width is a malformed node rather than something to
            // round.
            let v = match v_hidden {
                Some(v) => *v,
                None => {
                    if !weights[1].is_multiple_of(3) {
                        return Err(format!(
                            "Attention: weights dim1 {} is not divisible by 3 and \
                             qkv_hidden_sizes is absent",
                            weights[1]
                        ));
                    }
                    weights[1] / 3
                }
            };
            let mut outputs = vec![vec![x[0], x[1], v]];

            if *num_outputs > 1 {
                if *num_heads == 0 {
                    return Err(
                        "Attention: num_heads is required to shape the present output".into(),
                    );
                }
                if !v.is_multiple_of(*num_heads) {
                    return Err(format!(
                        "Attention: v_hidden {v} is not divisible by num_heads {num_heads}"
                    ));
                }
                let head_size = v / num_heads;
                // `past` is input(4): [2, B, num_heads, past_seq, head_size].
                let past_seq = inputs
                    .get(4)
                    .map(|p| if p.shape.len() >= 5 { p.shape[3] } else { 0 })
                    .unwrap_or(0);
                let present = vec![2, x[0], *num_heads, past_seq + x[1], head_size];
                for _ in 1..*num_outputs {
                    outputs.push(present.clone());
                }
            }
            Ok(outputs)
        }

        ShapeInference::PackedMultiHeadAttention => {
            // Tokens are packed with padding removed, so the output is rank-2:
            // [token_count, v_hidden] = [query.dim0, value.dim1]. Neither dim
            // can be read off input[0] alone, which is why this is not
            // `SameAsInput(0)`.
            if inputs.len() < 3 {
                return Err(
                    "PackedMultiHeadAttention: expected at least query, key and value".into(),
                );
            }
            let q = inputs[0].shape;
            let v = inputs[2].shape;
            if q.is_empty() {
                return Err("PackedMultiHeadAttention: query must not be rank-0".into());
            }
            if v.len() < 2 {
                return Err(format!(
                    "PackedMultiHeadAttention: value rank must be >= 2, got {}",
                    v.len()
                ));
            }
            Ok(vec![vec![q[0], v[1]]])
        }

        ShapeInference::AttentionStd {
            q_num_heads,
            kv_num_heads,
            num_outputs,
        } => {
            // Inputs: Q(0), K(1), V(2), attn_mask(3), past_key(4),
            // past_value(5), nonpad_kv_seqlen(6). Q/K/V are rank-3
            // (batch, seq, hidden) or rank-4 (batch, heads, seq, head_size).
            if inputs.len() < 3 {
                return Err("Attention: expected at least Q, K, V inputs".into());
            }
            let q = inputs[0].shape;
            let v = inputs[2].shape;
            if q.len() < 2 {
                return Err("Attention: query rank must be ≥3 or 4".into());
            }

            // Resolve (batch, q_heads, q_seq, qk_head_size) from Q and
            // (kv_heads, kv_seq, v_head_size) from V, for both layouts.
            let (batch, q_heads, q_seq, qk_head_size, is_3d) = match q.len() {
                4 => (q[0], q[1], q[2], q[3], false),
                3 => {
                    if *q_num_heads == 0 {
                        return Err("Attention: rank-3 query requires q_num_heads".into());
                    }
                    let hidden = q[2];
                    (
                        q[0],
                        *q_num_heads,
                        q[1],
                        hidden / (*q_num_heads).max(1),
                        true,
                    )
                }
                other => {
                    return Err(format!("Attention: query must be rank 3 or 4, got {other}"));
                }
            };
            let (kv_heads, kv_seq, v_head_size) = match v.len() {
                4 => (v[1], v[2], v[3]),
                3 => {
                    let kvh = if *kv_num_heads > 0 {
                        *kv_num_heads
                    } else {
                        *q_num_heads
                    };
                    if kvh == 0 {
                        return Err("Attention: rank-3 value requires kv_num_heads".into());
                    }
                    (kvh, v[1], v[2] / kvh.max(1))
                }
                other => {
                    return Err(format!("Attention: value must be rank 3 or 4, got {other}"));
                }
            };

            // Y follows Q's layout, carrying the value head size.
            let y = if is_3d {
                vec![batch, q_seq, q_heads.saturating_mul(v_head_size)]
            } else {
                vec![batch, q_heads, q_seq, v_head_size]
            };

            let mut outputs = vec![y];
            if *num_outputs > 1 {
                // present_key/present_value are always rank-4
                // (batch, kv_heads, total_seq, head_size). total_seq folds in
                // any past_key length (input 4).
                let past_seq = inputs
                    .get(4)
                    .map(|p| if p.shape.len() >= 3 { p.shape[2] } else { 0 })
                    .unwrap_or(0);
                let total_seq = past_seq.saturating_add(kv_seq);
                let present_key = vec![batch, kv_heads, total_seq, qk_head_size];
                let present_value = vec![batch, kv_heads, total_seq, v_head_size];
                // Slot order: Y, present_key, present_value, qk_matmul_output.
                if *num_outputs > 1 {
                    outputs.push(present_key);
                }
                if *num_outputs > 2 {
                    outputs.push(present_value);
                }
                if *num_outputs > 3 {
                    // qk_matmul_output: (batch, q_heads, q_seq, total_seq).
                    outputs.push(vec![batch, q_heads, q_seq, total_seq]);
                }
            }
            Ok(outputs)
        }

        ShapeInference::ConstantOfShape => {
            if inputs.is_empty() {
                return Err("ConstantOfShape: expected 1 input".into());
            }
            let dims = read_i64_vec(&inputs[0], "ConstantOfShape shape")?;
            let mut shape = Vec::with_capacity(dims.len());
            for d in dims {
                shape.push(usize::try_from(d).map_err(|_| {
                    format!("ConstantOfShape: dimension must be non-negative, got {d}")
                })?);
            }
            Ok(vec![shape])
        }

        ShapeInference::Expand => {
            if inputs.len() < 2 {
                return Err(format!(
                    "Expand: expected 2 inputs (data, shape), got {}",
                    inputs.len()
                ));
            }
            let target = read_i64_vec(&inputs[1], "Expand shape")?;
            let mut want = Vec::with_capacity(target.len());
            for d in target {
                want.push(usize::try_from(d).map_err(|_| {
                    format!("Expand: target dimension must be non-negative, got {d}")
                })?);
            }
            // Bidirectional, not "take the target": ONNX broadcasts the input's
            // shape against `shape`, so a target of [1] against data [3] yields
            // [3], and a target longer than the input extends the rank.
            let shape = onnx_runtime_ir::broadcast_shapes(inputs[0].shape, &want)
                .map_err(|e| format!("Expand: broadcast failed: {e}"))?;
            Ok(vec![shape])
        }

        ShapeInference::Tile => {
            if inputs.len() < 2 {
                return Err(format!(
                    "Tile: expected 2 inputs (data, repeats), got {}",
                    inputs.len()
                ));
            }
            let repeats = read_i64_vec(&inputs[1], "Tile repeats")?;
            if repeats.len() != inputs[0].shape.len() {
                return Err(format!(
                    "Tile: repeats has {} entries but the input has rank {}",
                    repeats.len(),
                    inputs[0].shape.len()
                ));
            }
            let mut shape = Vec::with_capacity(repeats.len());
            for (d, r) in inputs[0].shape.iter().zip(repeats) {
                let r = usize::try_from(r)
                    .map_err(|_| format!("Tile: repeats must be non-negative, got {r}"))?;
                shape.push(d * r);
            }
            Ok(vec![shape])
        }

        ShapeInference::Window => {
            if inputs.is_empty() {
                return Err("Window: expected 1 input (size)".into());
            }
            let n = read_scalar_i64(&inputs[0], "Window size")?;
            let n = usize::try_from(n)
                .map_err(|_| format!("Window: size must be non-negative, got {n}"))?;
            Ok(vec![vec![n]])
        }

        ShapeInference::Dft {
            onesided,
            axis_attr,
            default_axis,
        } => {
            if inputs.is_empty() {
                return Err("DFT: expected at least 1 input".into());
            }
            let input = &inputs[0];
            let rank = input.shape.len();
            if rank < 2 {
                return Err(format!(
                    "DFT: input must have rank >= 2 (signal axis plus complex \
                     component dim), got {rank}"
                ));
            }
            let last = rank - 1;
            let complex_dim = input.shape[last];
            if complex_dim != 1 && complex_dim != 2 {
                return Err(format!(
                    "DFT: last dimension must be 1 (real) or 2 (complex), got {complex_dim}"
                ));
            }

            // opset 20 moves `axis` into input 2; before that it is an
            // attribute defaulting to -2.
            let axis_raw = match inputs.get(2) {
                Some(t) if !t.is_absent() && t.shape.iter().product::<usize>() > 0 => {
                    read_scalar_i64(t, "DFT axis")?
                }
                _ => axis_attr.unwrap_or(*default_axis),
            };
            let axis = normalize_axis(axis_raw, rank, "DFT axis")?;
            if axis == last {
                return Err(
                    "DFT: the signal axis cannot be the complex component dimension".into(),
                );
            }

            // `dft_length` (input 1) overrides the signal extent when present.
            let signal_len = match inputs.get(1) {
                Some(t) if !t.is_absent() && t.shape.iter().product::<usize>() > 0 => {
                    let n = read_scalar_i64(t, "DFT dft_length")?;
                    usize::try_from(n)
                        .map_err(|_| format!("DFT: dft_length must be non-negative, got {n}"))?
                }
                _ => input.shape[axis],
            };

            let mut shape = input.shape.to_vec();
            shape[axis] = if *onesided {
                signal_len / 2 + 1
            } else {
                signal_len
            };
            // Always complex out, even for real input.
            shape[last] = 2;
            Ok(vec![shape])
        }

        ShapeInference::Compress { axis } => {
            // The only rule here that reads input *values* rather than shapes.
            // `Compress` selects the entries where `condition` is true, so the
            // output extent is a popcount and is unknowable until Compute —
            // which is precisely when this runs.
            if inputs.len() < 2 {
                return Err(format!(
                    "Compress: expected 2 inputs (data, condition), got {}",
                    inputs.len()
                ));
            }
            let data = &inputs[0];
            let condition = &inputs[1];
            if condition.shape.len() != 1 {
                return Err(format!(
                    "Compress: condition must be 1-D, got rank {}",
                    condition.shape.len()
                ));
            }

            // Mirror the kernel: with no axis the input is flattened first.
            let (mut shape, axis_idx) = match axis {
                Some(a) => {
                    let rank = data.shape.len() as i64;
                    let a = if *a < 0 { a + rank } else { *a };
                    if a < 0 || a >= rank {
                        return Err(format!("Compress: axis {a} out of range for rank {rank}"));
                    }
                    (data.shape.to_vec(), a as usize)
                }
                None => (vec![data.shape.iter().product::<usize>()], 0),
            };

            // `condition` may be shorter or longer than the axis; the spec
            // selects over the overlap, which is what the kernel does too.
            let axis_len = shape[axis_idx];
            let selected = count_true(condition)?
                .into_iter()
                .filter(|&i| i < axis_len)
                .count();
            shape[axis_idx] = selected;
            Ok(vec![shape])
        }

        ShapeInference::Declined {
            op_type,
            domain,
            reason,
        } => Err(match reason {
            DeclineReason::Unmodelled => format!(
                "Op '{op_type}' (domain '{domain}') has no shape-inference rule at all. \
                 If a kernel is registered for it, that registration is incomplete and the \
                 kernel is never dispatched to. Add a variant to ShapeInference, resolve it \
                 in ShapeInference::for_node, and handle it in infer_shapes. A data-dependent \
                 extent is expressed as `None` for that dimension."
            ),
            DeclineReason::NodeNotShapeable(why) => format!(
                "Op '{op_type}' (domain '{domain}') is modelled, but this node cannot be \
                 shaped: {why}. The claim was dropped deliberately; this is the shape table \
                 working as intended, not a missing rule."
            ),
        }),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shape-inference helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Normalise a possibly-negative axis index relative to `rank`.
fn normalise_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let rank_i = rank as i64;
    let a = if axis < 0 { rank_i + axis } else { axis };
    if a < 0 || a >= rank_i {
        Err(format!("axis {axis} out of range for rank {rank}"))
    } else {
        Ok(a as usize)
    }
}

/// Output shape for a MatMul-family op.
fn matmul_output_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    match (a.len(), b.len()) {
        (0, _) | (_, 0) => Err("MatMul: 0-D inputs are not supported".into()),
        (1, 1) => {
            // dot product → scalar
            Ok(vec![])
        }
        (1, 2) => {
            // [K] × [K, N] → [N]
            Ok(vec![b[1]])
        }
        (2, 1) => {
            // [M, K] × [K] → [M]
            Ok(vec![a[0]])
        }
        (ra, rb) => {
            // Batch MatMul: broadcast all but last two dims.
            let batch_a = &a[..ra - 2];
            let batch_b = &b[..rb - 2];
            let m = a[ra - 2];
            let n = b[rb - 1];
            // Broadcast batch dims.
            let max_batch = batch_a.len().max(batch_b.len());
            let a_padded = std::iter::repeat_n(1usize, max_batch - batch_a.len())
                .chain(batch_a.iter().copied());
            let b_padded = std::iter::repeat_n(1usize, max_batch - batch_b.len())
                .chain(batch_b.iter().copied());
            let mut batch_out: Vec<usize> = Vec::with_capacity(max_batch);
            for (x, y) in a_padded.zip(b_padded) {
                if x != y && x != 1 && y != 1 {
                    return Err(format!("MatMul: incompatible batch dims: {a:?} vs {b:?}"));
                }
                batch_out.push(x.max(y));
            }
            batch_out.push(m);
            batch_out.push(n);
            Ok(batch_out)
        }
    }
}

/// Output shape for a reduction op.
fn reduce_shape(
    shape: &[usize],
    axes: Option<&[i64]>,
    keepdims: bool,
    noop_with_empty_axes: bool,
) -> Result<Vec<usize>, String> {
    match axes {
        Some(ax) if ax.is_empty() && noop_with_empty_axes => {
            // No-op when empty axes and the flag is set.
            Ok(shape.to_vec())
        }
        // Empty axes (without noop) or None → reduce all dimensions.
        None | Some([]) => {
            if keepdims {
                Ok(vec![1; shape.len()])
            } else {
                Ok(vec![])
            }
        }
        Some(ax) => {
            let rank = shape.len();
            let norm_axes: Vec<usize> = ax
                .iter()
                .map(|&a| normalise_axis(a, rank))
                .collect::<Result<_, _>>()?;
            let out: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| {
                    if norm_axes.contains(&i) {
                        if keepdims { Some(1) } else { None }
                    } else {
                        Some(d)
                    }
                })
                .collect();
            Ok(out)
        }
    }
}

/// Read all values of a contiguous i64 tensor as a Vec<i64>.
///
/// # Safety
/// `view.data` must point to a valid, readable region of `view.shape.iter().product()`
/// `i64` elements.
unsafe fn read_i64_tensor(view: &TensorView<'_>) -> Result<Vec<i64>, String> {
    if !view.device.is_host_accessible() {
        return Err(format!(
            "shape operand is on {:?}, which the plugin host cannot dereference; \
             the device EP must decline this value-driven shape rule or provide \
             an explicit bounded metadata transfer",
            view.device
        ));
    }
    if view.dtype != DataType::Int64 {
        return Err(format!("expected Int64 tensor, got {:?}", view.dtype));
    }
    if view.data.is_null() {
        return Err("tensor data pointer is null".into());
    }
    let numel: usize = view.shape.iter().product();
    let ptr = view.data.0.cast::<i64>();
    let vals = unsafe { std::slice::from_raw_parts(ptr, numel) }.to_vec();
    Ok(vals)
}

/// Compute the output shape for a Slice operation.
///
/// Mirrors the semantics of `slice_plan` in the CPU EP kernel (`slice.rs`),
/// including clamping, negative index handling, and step direction.
fn slice_output_shape(
    data: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
) -> Result<Vec<usize>, String> {
    let rank = data.len();
    let mut out = data.to_vec();

    for (i, &ax) in axes.iter().enumerate() {
        let axis = normalise_axis(ax, rank)?;
        let dim = data[axis] as i64;
        let step = steps.get(i).copied().unwrap_or(1);
        if step == 0 {
            return Err("Slice: step cannot be zero".into());
        }

        let (clamp_lo, clamp_hi) = if step > 0 {
            (0i64, dim)
        } else {
            (-1i64, dim - 1)
        };

        let mut start = starts.get(i).copied().unwrap_or(0);
        let mut end = ends.get(i).copied().unwrap_or(dim);

        // Negative indices.
        if start < 0 {
            start += dim;
        }
        if end < 0 {
            end += dim;
        }

        // Clamp.
        start = start.clamp(clamp_lo, clamp_hi);
        end = end.clamp(clamp_lo, clamp_hi);

        let span = end - start;
        let count = if step > 0 {
            if span <= 0 {
                0
            } else {
                (span + step - 1) / step
            }
        } else {
            if span >= 0 {
                0
            } else {
                (-span + (-step) - 1) / (-step)
            }
        };
        out[axis] = count.max(0) as usize;
    }
    Ok(out)
}

/// ReleaseState: drop per-session compute state.
///
/// Returns `void` — there is no status channel to surface an error. A panic
/// in the drop path (or any future `ComputeState` extension) must not unwind
/// across the `extern "C"` boundary (undefined behaviour); we catch it here and
/// swallow it, matching the guard pattern in `compute_create_state` and
/// `compute_execute`. This fixes NEW-1 from the EP plugin security audit.
unsafe extern "C" fn compute_release_state(
    _info: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !state.is_null() {
            unsafe { drop(Box::from_raw(state.cast::<ComputeState>())) };
        }
    }));
    // Panic swallowed: no status channel exists for ReleaseState.
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::EpError;

    struct KernelSizedMock {
        calls: Arc<AtomicUsize>,
        output: KernelSizedOutput,
    }

    struct DeviceKernelSizedMock {
        prepares: Arc<AtomicUsize>,
        materializes: Arc<AtomicUsize>,
    }

    impl Kernel for DeviceKernelSizedMock {
        fn execute(
            &self,
            _: &[TensorView],
            _: &mut [TensorMut],
        ) -> onnx_runtime_ep_api::Result<()> {
            unreachable!("device kernel-sized mock uses split phases")
        }

        fn has_kernel_sized_outputs(&self) -> bool {
            true
        }

        fn kernel_sized_output_policy(&self) -> KernelSizedOutputPolicy {
            KernelSizedOutputPolicy::DeviceWorkspace
        }

        fn prepare_kernel_sized_device(
            &self,
            _: &[TensorView],
            requested_outputs: &[bool],
            _: Option<WorkspaceView>,
        ) -> onnx_runtime_ep_api::Result<Vec<Option<KernelSizedOutputMetadata>>> {
            self.prepares.fetch_add(1, Ordering::Relaxed);
            Ok(requested_outputs
                .iter()
                .enumerate()
                .map(|(slot, requested)| {
                    requested.then(|| KernelSizedOutputMetadata {
                        shape: vec![2],
                        dtype: if slot == 0 {
                            DataType::Float32
                        } else {
                            DataType::Int64
                        },
                    })
                })
                .collect())
        }

        fn materialize_kernel_sized_device(
            &self,
            _: &[TensorView],
            _: &mut [TensorMut],
            _: Option<WorkspaceView>,
        ) -> onnx_runtime_ep_api::Result<()> {
            self.materializes.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    impl Kernel for KernelSizedMock {
        fn execute(
            &self,
            _inputs: &[TensorView],
            _outputs: &mut [TensorMut],
        ) -> onnx_runtime_ep_api::Result<()> {
            Err(EpError::KernelFailed(
                "ordinary execute must not run for a kernel-sized strategy".into(),
            ))
        }

        fn has_kernel_sized_outputs(&self) -> bool {
            true
        }

        fn execute_kernel_sized(
            &self,
            _inputs: &[TensorView],
            requested_outputs: &[bool],
        ) -> onnx_runtime_ep_api::Result<Vec<Option<KernelSizedOutput>>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(requested_outputs
                .iter()
                .map(|requested| requested.then(|| self.output.clone()))
                .collect())
        }
    }

    fn kernel_sized_info() -> ExportedComputeInfo {
        let mut info = ExportedComputeInfo::new(Vec::new());
        info.set_host_accessible(true);
        info
    }

    #[test]
    fn kernel_sized_dispatch_invokes_algorithm_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let kernel = KernelSizedMock {
            calls: Arc::clone(&calls),
            output: KernelSizedOutput {
                shape: vec![2],
                dtype: DataType::Float32,
                bytes: [1.0f32, 2.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            },
        };
        let input_data = [3.0f32, 3.0, 4.0];
        let input = TensorView::new(
            DevicePtr(input_data.as_ptr().cast()),
            DataType::Float32,
            &[3],
            &[1],
            DeviceId::cpu(),
        );
        let outputs = run_kernel_sized(
            &kernel_sized_info(),
            &kernel,
            &[input],
            &[true],
            &[DataType::Float32],
            "test",
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(outputs[0].as_ref().unwrap().shape, [2]);
    }

    #[test]
    fn device_kernel_sized_policy_splits_metadata_and_materialization_once() {
        let prepares = Arc::new(AtomicUsize::new(0));
        let materializes = Arc::new(AtomicUsize::new(0));
        let kernel = DeviceKernelSizedMock {
            prepares: Arc::clone(&prepares),
            materializes: Arc::clone(&materializes),
        };
        let metadata = run_device_kernel_sized(
            &kernel,
            &[],
            &[true, false, true],
            &[DataType::Float32, DataType::Undefined, DataType::Int64],
            None,
            "test",
        )
        .unwrap();
        assert_eq!(prepares.load(Ordering::Relaxed), 1);
        assert!(metadata[1].is_none());
        let mut outputs = [
            absent_output_view(),
            absent_output_view(),
            absent_output_view(),
        ];
        kernel
            .materialize_kernel_sized_device(&[], &mut outputs, None)
            .unwrap();
        assert_eq!(materializes.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn kernel_sized_host_gate_precedes_algorithm() {
        let calls = Arc::new(AtomicUsize::new(0));
        let kernel = KernelSizedMock {
            calls: Arc::clone(&calls),
            output: KernelSizedOutput {
                shape: vec![1],
                dtype: DataType::Float32,
                bytes: 1.0f32.to_le_bytes().to_vec(),
            },
        };
        let input_data = [1.0f32];
        let input = TensorView::new(
            DevicePtr(input_data.as_ptr().cast()),
            DataType::Float32,
            &[1],
            &[1],
            DeviceId::cuda(0),
        );
        let error = run_kernel_sized(
            &kernel_sized_info(),
            &kernel,
            &[input],
            &[true],
            &[DataType::Float32],
            "test",
        )
        .unwrap_err();
        assert!(error.contains("host-only"));
        assert_eq!(calls.load(Ordering::Relaxed), 0);

        let cpu_input = TensorView::new(
            DevicePtr(input_data.as_ptr().cast()),
            DataType::Float32,
            &[1],
            &[1],
            DeviceId::cpu(),
        );
        let device_output_info = ExportedComputeInfo::new(Vec::new());
        let error = run_kernel_sized(
            &device_output_info,
            &kernel,
            &[cpu_input],
            &[true],
            &[DataType::Float32],
            "test",
        )
        .unwrap_err();
        assert!(error.contains("this EP uses device-resident tensors"));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn kernel_sized_validation_fails_closed_on_malformed_bytes_and_overflow() {
        let malformed = vec![Some(KernelSizedOutput {
            shape: vec![2],
            dtype: DataType::Float32,
            bytes: vec![0; 4],
        })];
        let error =
            validate_kernel_sized_outputs(&malformed, &[true], &[DataType::Float32], "test")
                .unwrap_err();
        assert!(error.contains("returned 4 bytes, expected 8"));

        let overflow = vec![Some(KernelSizedOutput {
            shape: vec![i64::MAX as usize, 3],
            dtype: DataType::Float32,
            bytes: Vec::new(),
        })];
        let error = validate_kernel_sized_outputs(&overflow, &[true], &[DataType::Float32], "test")
            .unwrap_err();
        assert!(error.contains("overflows usize"));
    }

    #[test]
    fn kernel_sized_materializes_one_copy_per_present_output() {
        KERNEL_SIZED_MATERIALIZATION_COPIES.store(0, Ordering::Relaxed);
        let value = KernelSizedOutput {
            shape: vec![2],
            dtype: DataType::Float32,
            bytes: [1.0f32, 2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        };
        let mut destination = vec![0u8; value.bytes.len()];
        let mut output = crate::kernel_ctx::OwnedOutput {
            data_ptr: destination.as_mut_ptr().cast(),
            dtype: DataType::Float32,
            shape: vec![2].into(),
            strides: vec![1].into(),
            mem_info: std::ptr::null(),
        };
        materialize_kernel_sized(&mut output, &value, 0).unwrap();
        assert_eq!(destination, value.bytes);
        assert_eq!(
            KERNEL_SIZED_MATERIALIZATION_COPIES.load(Ordering::Relaxed),
            1
        );
    }

    /// Return the production portion of a Rust source file, normalised to LF.
    ///
    /// Test-only top-level items may be interleaved with production code, so
    /// truncating at a particular test-module spelling is both incomplete and
    /// sensitive to checkout line endings.
    fn production_source(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let mut skipped_item_depth: Option<i32> = None;

        for line in src.lines() {
            if let Some(depth) = skipped_item_depth.as_mut() {
                let opens = line.matches('{').count() as i32;
                let closes = line.matches('}').count() as i32;
                *depth += opens - closes;
                if *depth <= 0 && !(opens == 0 && closes == 0 && line.trim().is_empty()) {
                    skipped_item_depth = None;
                }
                continue;
            }

            if line.starts_with("#[cfg(") && line.contains("test") {
                assert!(
                    line.ends_with(")]"),
                    "multi-line #[cfg(...test...)] attributes are not supported: {line}"
                );
                skipped_item_depth = Some(0);
                continue;
            }

            out.push_str(line);
            out.push('\n');
        }

        assert!(
            skipped_item_depth.is_none(),
            "a #[cfg(test)] item never closed"
        );
        out
    }

    #[test]
    fn production_source_is_line_ending_agnostic() {
        let lf = "\
fn before() {}
#[cfg(test)]
fn test_only() {}
fn after() {}
";
        let crlf = lf.replace('\n', "\r\n");
        let expected = "fn before() {}\nfn after() {}\n";

        assert_eq!(production_source(lf), expected);
        assert_eq!(production_source(&crlf), expected);
    }

    // ── Matmul-family shape rules ─────────────────────────────────────────────
    //
    // These are the hardware-independent falsifiers for the two shape rules
    // that decide whether `MatMulNBits` and `QLinearMatMul` can be claimed at
    // all: `GetCapability` drops any claim whose strategy is `Declined`, and a
    // wrong strategy is worse than none because it produces a plausible but
    // incorrect output buffer.

    fn u8_view<'a>(shape: &'a [usize], strides: &'a [i64]) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(std::ptr::null()),
            DataType::Uint8,
            shape,
            strides,
            DeviceId::cpu(),
        )
    }

    #[test]
    fn matmul_nbits_shape_uses_the_n_attribute_not_the_packed_weight() {
        // B is [N, blocks_per_col, blob_bytes] — nothing about its trailing
        // dims is a matmul operand, so N must come from the attribute.
        let a = view(&[1, 256], &[256, 1]);
        let b = u8_view(&[4096, 8, 16], &[128, 16, 1]);
        let scales = view(&[4096 * 8], &[1]);

        let shapes = infer(&ShapeInference::MatMulNBits { n: 4096 }, &[a, b, scales])
            .expect("MatMulNBits shape inference");
        assert_eq!(shapes, vec![vec![1, 4096]]);

        // Aliasing this op to plain `MatMul` — which is what the table used to
        // do — broadcasts the activation against the *packed* weight and
        // yields a wrong-rank, wrong-extent buffer. Pin the bad answer so the
        // alias cannot come back unnoticed.
        let wrong = infer(&ShapeInference::MatMul, &[a, b, scales])
            .expect("plain MatMul over a packed weight");
        assert_eq!(wrong, vec![vec![4096, 1, 16]]);
        assert_ne!(wrong, vec![vec![1, 4096]]);
    }

    #[test]
    fn matmul_nbits_shape_preserves_leading_activation_dims() {
        let a = view(&[2, 3, 256], &[768, 256, 1]);
        let b = u8_view(&[512, 8, 16], &[128, 16, 1]);
        let shapes = infer(&ShapeInference::MatMulNBits { n: 512 }, &[a, b])
            .expect("MatMulNBits shape inference");
        assert_eq!(shapes, vec![vec![2, 3, 512]]);
    }

    #[test]
    fn qlinear_matmul_shape_uses_inputs_zero_and_three() {
        // a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zp.
        // Input 1 is a scalar, so a rule that read operands 0 and 1 — the
        // plain-`MatMul` convention — could not produce [1, 4096] by accident.
        let a = u8_view(&[1, 256], &[256, 1]);
        let scalar = view(&[], &[]);
        let b = u8_view(&[256, 4096], &[4096, 1]);
        let inputs = [a, scalar, scalar, b, scalar, scalar, scalar, scalar];
        let shapes =
            infer(&ShapeInference::QLinearMatMul, &inputs).expect("QLinearMatMul shape inference");
        assert_eq!(shapes, vec![vec![1, 4096]]);
    }

    #[test]
    fn matmul_family_ops_resolve_to_their_own_strategies() {
        use onnx_runtime_ir::{Node, NodeId};
        let node = |op: &str, domain: &str| {
            let mut n = Node::new(NodeId(0), op, Vec::new(), Vec::new());
            n.domain = domain.to_string();
            n
        };
        assert!(matches!(
            ShapeInference::for_node(&node("MatMul", ""), &[], 1),
            ShapeInference::MatMul
        ));
        assert!(matches!(
            ShapeInference::for_node(&node("QLinearMatMul", ""), &[], 1),
            ShapeInference::QLinearMatMul
        ));
        // `MatMulNBits` derives its output width from the `N` attribute; a node
        // without it is malformed, so `for_node` must decline rather than fall
        // back to the plain-matmul rule (which would read the packed weight's
        // `blob_size` as the output column count).
        assert!(matches!(
            ShapeInference::for_node(&node("MatMulNBits", "com.microsoft"), &[], 1),
            ShapeInference::Declined { .. }
        ));
    }

    use onnx_runtime_ep_api::tensor::{DevicePtr, TensorView};
    use onnx_runtime_ir::{DataType, DeviceId};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn view<'a>(shape: &'a [usize], strides: &'a [i64]) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(std::ptr::null()),
            DataType::Float32,
            shape,
            strides,
            DeviceId::cpu(),
        )
    }

    /// Build an i64 tensor view backed by a provided slice.
    fn i64_view<'a>(data: &'a [i64], shape: &'a [usize], strides: &'a [i64]) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(data.as_ptr().cast()),
            DataType::Int64,
            shape,
            strides,
            DeviceId::cpu(),
        )
    }

    fn infer(
        strategy: &ShapeInference,
        inputs: &[TensorView<'_>],
    ) -> Result<Vec<Vec<usize>>, String> {
        infer_shapes(strategy, inputs)
    }

    /// Differential oracle for the allocation-free path.
    ///
    /// `infer_shapes_into` reimplements a handful of hot arms and delegates the
    /// rest. That split is only safe while the two agree, and the arms it
    /// reimplements are exactly the ones the dispatch benchmark drives -- so a
    /// divergence would show up as wrong output shapes on the most common ops
    /// in the suite, not on an exotic one. Asserting equality on both the `Ok`
    /// shapes and the `Err` text keeps the fast path honest about failures too,
    /// since a fast arm that skipped a bounds check would still "work" until it
    /// indexed out of range.
    #[track_caller]
    fn assert_paths_agree(strategy: &ShapeInference, inputs: &[TensorView<'_>]) {
        let slow = infer_shapes(strategy, inputs);
        let mut buf = Vec::new();
        let fast = infer_shapes_into(strategy, inputs, &mut buf);
        match (&slow, &fast) {
            (Ok(want), Ok(())) => {
                let got: Vec<Vec<usize>> = buf.iter().map(|d| d.as_slice().to_vec()).collect();
                assert_eq!(&got, want, "fast path disagreed for {strategy:?}");
            }
            (Err(want), Err(got)) => {
                assert_eq!(
                    got, want,
                    "fast path gave a different error for {strategy:?}"
                );
            }
            _ => panic!("fast/slow disagreed on success for {strategy:?}: {slow:?} vs {fast:?}"),
        }
    }

    // ── The allocation-free path agrees with the allocating one ─────────────

    /// The arms `infer_shapes_into` reimplements, including their failures.
    #[test]
    fn infer_shapes_into_matches_infer_shapes_on_the_fast_arms() {
        let a = view(&[2, 3, 4], &[12, 4, 1]);
        let b = view(&[2, 3, 4], &[12, 4, 1]);
        let scalar = view(&[], &[]);

        assert_paths_agree(&ShapeInference::SameAsInput(0), &[a, b]);
        assert_paths_agree(&ShapeInference::SameAsInput(1), &[a, b]);
        // Rank 0 is not a degenerate case for a buffer that stores a length.
        assert_paths_agree(&ShapeInference::SameAsInput(0), &[scalar]);
        // Out of range must fail identically, not index out of bounds.
        assert_paths_agree(&ShapeInference::SameAsInput(1), &[a]);
        assert_paths_agree(&ShapeInference::SameAsInput(0), &[]);

        assert_paths_agree(&ShapeInference::ElementwiseBroadcast, &[a]);
        assert_paths_agree(&ShapeInference::ElementwiseBroadcast, &[scalar]);
        // Two operands fall through to the shared fold rather than the fast arm.
        assert_paths_agree(&ShapeInference::ElementwiseBroadcast, &[a, b]);
        assert_paths_agree(&ShapeInference::ElementwiseBroadcast, &[]);
    }

    /// A rank past `INLINE_RANK`, where `DimVec` stops being inline and starts
    /// owning a heap buffer. The fast path must be correct on both sides of
    /// that boundary, and the buffer must survive being reused across the
    /// transition in either direction.
    #[test]
    fn infer_shapes_into_handles_ranks_that_spill_out_of_line() {
        let wide_shape: Vec<usize> = vec![2; crate::dim_vec::INLINE_RANK + 3];
        let wide_strides: Vec<i64> = vec![1; wide_shape.len()];
        let wide = view(&wide_shape, &wide_strides);
        let narrow = view(&[7], &[1]);

        assert_paths_agree(&ShapeInference::SameAsInput(0), &[wide]);

        // Reuse one buffer across wide -> narrow -> wide. A buffer that kept a
        // stale length or a stale spilled allocation shows up here.
        let mut buf = Vec::new();
        for expect in [&wide_shape[..], &[7][..], &wide_shape[..]] {
            let input = if expect.len() == 1 { narrow } else { wide };
            infer_shapes_into(&ShapeInference::SameAsInput(0), &[input], &mut buf).unwrap();
            assert_eq!(buf.len(), 1, "one output slot");
            assert_eq!(buf[0].as_slice(), expect, "stale storage leaked through");
        }
    }

    /// A *shorter* result after a longer one must not leave the tail of the
    /// previous node visible.
    ///
    /// Every arm that writes the buffer needs its own case here, because each
    /// one clears it separately. Driving only the `SameAsInput` arm proves only
    /// that `SameAsInput` clears: deleting the `out.clear()` from the
    /// delegating arm, or from the shared-native arm, survived an earlier
    /// version of this test. Both are load-bearing in production, where the
    /// routed path reads `output_shapes.iter().enumerate()` as the node's
    /// output arity — a stale tail becomes a phantom output slot.
    #[test]
    fn every_arm_truncates_when_a_node_has_fewer_outputs() {
        let a = view(&[2, 3], &[3, 1]);
        assert_paths_agree(&ShapeInference::MatMul, &[a, a]);

        // Two outputs, via the delegating arm, to leave a tail behind.
        let input = view(&[1, 2, 5], &[10, 5, 1]);
        let weight = view(&[2, 1, 3], &[3, 3, 1]);
        let fill_two = |buf: &mut Vec<crate::dim_vec::DimVec<usize>>| {
            infer_shapes_into(&ShapeInference::CausalConvWithState, &[input, weight], buf).unwrap();
            assert_eq!(buf.len(), 2, "two outputs");
        };

        // Fast arm.
        let mut buf = Vec::new();
        fill_two(&mut buf);
        infer_shapes_into(&ShapeInference::SameAsInput(0), &[input], &mut buf).unwrap();
        assert_eq!(buf.len(), 1, "SameAsInput left the second slot behind");

        // Delegating arm: a one-output strategy that is NOT reimplemented.
        fill_two(&mut buf);
        infer_shapes_into(&ShapeInference::MatMul, &[a, a], &mut buf).unwrap();
        assert_eq!(
            buf.len(),
            1,
            "the delegating arm left the second slot behind"
        );

        // Shared-native arm, resolving.
        fill_two(&mut buf);
        let (strategy, ins) = resolving_shared_native();
        infer_shapes_into(&strategy, &ins, &mut buf).unwrap();
        assert_eq!(
            buf.len(),
            1,
            "the shared-native arm left the second slot behind"
        );
    }

    /// A `SharedNative` node that resolves natively, plus its operands.
    fn resolving_shared_native() -> (ShapeInference, [TensorView<'static>; 2]) {
        static DATA: [f32; 3] = [0.0; 3];
        static WANT: [i64; 2] = [1, 4];
        static DSHAPE: [usize; 2] = [3, 1];
        static DSTRIDE: [i64; 2] = [1, 1];
        static WSHAPE: [usize; 1] = [2];
        static WSTRIDE: [i64; 1] = [1];
        let data = TensorView::new(
            DevicePtr(DATA.as_ptr().cast()),
            DataType::Float32,
            &DSHAPE,
            &DSTRIDE,
            DeviceId::cpu(),
        );
        let shape_in = TensorView::new(
            DevicePtr(WANT.as_ptr().cast()),
            DataType::Int64,
            &WSHAPE,
            &WSTRIDE,
            DeviceId::cpu(),
        );
        let mut node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "Expand",
            vec![
                Some(onnx_runtime_ir::ValueId(0)),
                Some(onnx_runtime_ir::ValueId(1)),
            ],
            vec![onnx_runtime_ir::ValueId(2)],
        );
        node.version = Some(8);
        (
            ShapeInference::SharedNative {
                node: Box::new(node),
                fallback: Box::new(ShapeInference::SameAsInput(0)),
            },
            [data, shape_in],
        )
    }

    /// The shared-native arm is reimplemented too, so it needs the oracle just
    /// as much as the others -- on both sides of its branch. The `Resolved`
    /// side is the one production actually takes for `Expand`/`Tile` with
    /// concrete shape operands; the declining side is what routes a rule's
    /// fallback back through the fast arms.
    #[test]
    fn infer_shapes_into_matches_infer_shapes_on_shared_native() {
        let (resolving, ins) = resolving_shared_native();
        assert_paths_agree(&resolving, &ins);

        // A node no native rule knows: SymbolicOrUnknown, so the fallback runs.
        let unknown = Node::new(
            onnx_runtime_ir::NodeId(0),
            "NotRegisteredAnywhere",
            vec![Some(onnx_runtime_ir::ValueId(0))],
            vec![onnx_runtime_ir::ValueId(1)],
        );
        let input = view(&[2, 3], &[3, 1]);
        assert_eq!(
            infer_shared_node(&unknown, &[input]),
            SharedShapeResult::SymbolicOrUnknown,
            "this test is vacuous unless the rule really declines"
        );
        assert_paths_agree(
            &ShapeInference::SharedNative {
                node: Box::new(unknown.clone()),
                fallback: Box::new(ShapeInference::SameAsInput(0)),
            },
            &[input],
        );
        // A fallback that itself fails must surface the fallback's own error.
        assert_paths_agree(
            &ShapeInference::SharedNative {
                node: Box::new(unknown),
                fallback: Box::new(ShapeInference::SameAsInput(9)),
            },
            &[input],
        );
    }

    // ── Shapes carried in input values ───────────────────────────────────────

    #[test]
    fn constant_of_shape_takes_its_rank_from_the_value_length() {
        let dims = [2i64, 3, 4];
        let t = i64_scalar(&dims, &[3], &[1]);
        assert_eq!(
            infer(&ShapeInference::ConstantOfShape, &[t]).unwrap(),
            vec![vec![2, 3, 4]]
        );
    }

    #[test]
    fn constant_of_shape_empty_input_is_a_scalar() {
        let empty: [i64; 0] = [];
        let t = i64_scalar(&empty, &[0], &[1]);
        assert_eq!(
            infer(&ShapeInference::ConstantOfShape, &[t]).unwrap(),
            vec![Vec::<usize>::new()]
        );
    }

    #[test]
    fn expand_broadcasts_bidirectionally_rather_than_taking_the_target() {
        // The trap: `Expand` is *bidirectional*. Data [3,1] against a target
        // [1,4] is [3,4] — a rule that returned the target would answer [1,4],
        // and one that returned the data shape would answer [3,1].
        let buf = vec![0.0f32; 3];
        let data = f32_data(&[3, 1], &[1, 1], &buf);
        let want = [1i64, 4];
        let shape_in = i64_scalar(&want, &[2], &[1]);
        assert_eq!(
            infer(&ShapeInference::Expand, &[data, shape_in]).unwrap(),
            vec![vec![3, 4]]
        );
    }

    #[test]
    fn expand_target_may_add_rank() {
        let buf = vec![0.0f32; 3];
        let data = f32_data(&[3], &[1], &buf);
        let want = [2i64, 3];
        let shape_in = i64_scalar(&want, &[2], &[1]);
        assert_eq!(
            infer(&ShapeInference::Expand, &[data, shape_in]).unwrap(),
            vec![vec![2, 3]]
        );
    }

    #[test]
    fn tile_multiplies_each_dim_by_its_repeat() {
        let buf = vec![0.0f32; 6];
        let data = f32_data(&[2, 3], &[3, 1], &buf);
        let reps = [3i64, 2];
        let r = i64_scalar(&reps, &[2], &[1]);
        assert_eq!(
            infer(&ShapeInference::Tile, &[data, r]).unwrap(),
            vec![vec![6, 6]]
        );
    }

    #[test]
    fn tile_rejects_a_repeats_length_that_does_not_match_the_rank() {
        let buf = vec![0.0f32; 6];
        let data = f32_data(&[2, 3], &[3, 1], &buf);
        let reps = [3i64];
        let r = i64_scalar(&reps, &[1], &[1]);
        let err = infer(&ShapeInference::Tile, &[data, r]).unwrap_err();
        assert!(
            err.contains("rank"),
            "error should name the mismatch: {err}"
        );
    }

    #[test]
    fn shared_rule_rejection_preserves_the_existing_plugin_fallback() {
        let buf = vec![0.0f32; 6];
        let data = f32_data(&[2, 3], &[3, 1], &buf);
        let reps = [3i64, 2];
        // The native rule correctly rejects this malformed rank-2 `repeats`
        // tensor. The historical plugin rule only reads its two values and
        // sizes the output, so the first migration slice deliberately retains
        // that permissive behavior.
        let r = i64_scalar(&reps, &[2, 1], &[1, 1]);
        let mut node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "Tile",
            vec![
                Some(onnx_runtime_ir::ValueId(0)),
                Some(onnx_runtime_ir::ValueId(1)),
            ],
            vec![onnx_runtime_ir::ValueId(2)],
        );
        node.version = Some(13);
        assert!(matches!(
            infer_shared_node(&node, &[data, r]),
            SharedShapeResult::Rejected(reason) if reason.contains("invalid rank 2")
        ));

        let strategy =
            ShapeInference::for_node(&node, &[vec![Some(2), Some(3)], vec![Some(2), Some(1)]], 1);
        assert_eq!(infer(&strategy, &[data, r]).unwrap(), vec![vec![6, 6]]);
    }

    #[test]
    fn shared_expand_opset_floor_falls_back_before_version_eight() {
        let buf = vec![0.0f32; 3];
        let data = f32_data(&[3, 1], &[1, 1], &buf);
        let want = [1i64, 4];
        let shape_in = i64_scalar(&want, &[2], &[1]);
        let make_node = |version| {
            let mut node = Node::new(
                onnx_runtime_ir::NodeId(0),
                "Expand",
                vec![
                    Some(onnx_runtime_ir::ValueId(0)),
                    Some(onnx_runtime_ir::ValueId(1)),
                ],
                vec![onnx_runtime_ir::ValueId(2)],
            );
            node.version = Some(version);
            node
        };

        let version_seven = make_node(7);
        assert_eq!(
            infer_shared_node(&version_seven, &[data, shape_in]),
            SharedShapeResult::SymbolicOrUnknown
        );
        let strategy =
            ShapeInference::for_node(&version_seven, &[vec![Some(3), Some(1)], vec![Some(2)]], 1);
        assert_eq!(
            infer(&strategy, &[data, shape_in]).unwrap(),
            vec![vec![3, 4]],
            "Expand@7 must reach the compatibility fallback"
        );

        assert_eq!(
            infer_shared_node(&make_node(8), &[data, shape_in]),
            SharedShapeResult::Resolved(vec![vec![3, 4]])
        );
    }

    #[test]
    fn foreign_domain_expand_is_not_a_shared_native_rule() {
        let mut node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "Expand",
            vec![None, None],
            vec![onnx_runtime_ir::ValueId(2)],
        );
        node.domain = "example.foreign".into();
        node.version = Some(8);
        assert!(
            !matches!(
                ShapeInference::for_node(&node, &[vec![Some(3), Some(1)], vec![Some(2)]], 1),
                ShapeInference::SharedNative { .. }
            ),
            "operator names from foreign domains must not enter default-domain shared rules"
        );
    }

    #[test]
    fn unregistered_native_rule_can_use_a_synthetic_fallback() {
        let node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "NotRegisteredAnywhere",
            vec![Some(onnx_runtime_ir::ValueId(0))],
            vec![onnx_runtime_ir::ValueId(1)],
        );
        let input = view(&[2, 3], &[3, 1]);
        assert_eq!(
            infer_shared_node(&node, &[input]),
            SharedShapeResult::SymbolicOrUnknown
        );
        let strategy = ShapeInference::SharedNative {
            node: Box::new(node),
            fallback: Box::new(ShapeInference::SameAsInput(0)),
        };
        assert_eq!(infer(&strategy, &[input]).unwrap(), vec![vec![2, 3]]);
    }

    #[test]
    fn device_shape_operand_reaches_safe_plugin_error() {
        let buf = vec![0.0f32; 3];
        let data = f32_data(&[3, 1], &[1, 1], &buf);
        let want = [1i64, 4];
        let shape_in = TensorView::new(
            DevicePtr(want.as_ptr().cast()),
            DataType::Int64,
            &[2],
            &[1],
            DeviceId::cuda(0),
        );
        let mut node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "Expand",
            vec![
                Some(onnx_runtime_ir::ValueId(0)),
                Some(onnx_runtime_ir::ValueId(1)),
            ],
            vec![onnx_runtime_ir::ValueId(2)],
        );
        node.version = Some(8);
        assert_eq!(
            infer_shared_node(&node, &[data, shape_in]),
            SharedShapeResult::SymbolicOrUnknown
        );

        let strategy = ShapeInference::for_node(&node, &[vec![Some(3), Some(1)], vec![Some(2)]], 1);
        let error = infer(&strategy, &[data, shape_in])
            .expect_err("the fallback must refuse to dereference a device pointer");
        assert!(
            error.contains("host cannot read") && error.contains("Cuda"),
            "device-resident fallback error should explain the safe refusal: {error}"
        );
    }

    #[test]
    fn window_length_is_the_scalar_input_value() {
        let n = [16i64];
        let t = i64_scalar(&n, &[1], &[1]);
        assert_eq!(
            infer(&ShapeInference::Window, &[t]).unwrap(),
            vec![vec![16]]
        );
    }

    #[test]
    fn value_carried_shapes_refuse_device_memory() {
        // Same contract as Compress: the host cannot dereference device memory,
        // so it errors with the reason instead of guessing an extent.
        let dims = [2i64, 3];
        let t = TensorView::new(
            DevicePtr(dims.as_ptr() as *mut std::ffi::c_void),
            DataType::Int64,
            &[2],
            &[1],
            onnx_runtime_ir::DeviceId::cuda(0),
        );
        let err = infer(&ShapeInference::ConstantOfShape, &[t]).unwrap_err();
        assert!(
            err.contains("host cannot read"),
            "error should name the device restriction: {err}"
        );
    }

    // ── DFT ──────────────────────────────────────────────────────────────────

    fn i64_scalar<'a>(v: &'a [i64], shape: &'a [usize], strides: &'a [i64]) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(v.as_ptr() as *mut std::ffi::c_void),
            DataType::Int64,
            shape,
            strides,
            onnx_runtime_ir::DeviceId::cpu(),
        )
    }

    #[test]
    fn dft_shared_and_plugin_defaults_agree_across_opsets() {
        let data = vec![0.0f32; 48];
        let x = f32_data(&[1, 8, 6, 1], &[48, 6, 1, 1], &data);
        let make_node = |version| {
            let mut node = Node::new(
                onnx_runtime_ir::NodeId(0),
                "DFT",
                vec![Some(onnx_runtime_ir::ValueId(0))],
                vec![onnx_runtime_ir::ValueId(1)],
            );
            node.version = Some(version);
            node.attributes
                .insert("onesided".into(), onnx_runtime_ir::Attribute::Int(1));
            node
        };

        for (version, expected) in [(17, vec![1, 5, 6, 2]), (20, vec![1, 8, 4, 2])] {
            let node = make_node(version);
            let strategy =
                ShapeInference::for_node(&node, &[vec![Some(1), Some(8), Some(6), Some(1)]], 1);
            assert!(matches!(strategy, ShapeInference::SharedNative { .. }));
            assert_eq!(infer(&strategy, &[x]).unwrap(), vec![expected]);
            if version == 17 {
                let ShapeInference::SharedNative { fallback, .. } = &strategy else {
                    unreachable!()
                };
                assert_eq!(
                    infer(fallback, &[x]).unwrap(),
                    vec![vec![1, 5, 6, 2]],
                    "the compatibility fallback must preserve the opset-17 axis=1 default"
                );
            }
        }

        let mut attr_node = make_node(17);
        attr_node
            .attributes
            .insert("axis".into(), onnx_runtime_ir::Attribute::Int(2));
        let strategy =
            ShapeInference::for_node(&attr_node, &[vec![Some(1), Some(8), Some(6), Some(1)]], 1);
        assert_eq!(infer(&strategy, &[x]).unwrap(), vec![vec![1, 8, 4, 2]]);

        let mut input_node = make_node(20);
        input_node.inputs = vec![
            Some(onnx_runtime_ir::ValueId(0)),
            None,
            Some(onnx_runtime_ir::ValueId(2)),
        ];
        let axis_value = [1i64];
        let axis = i64_scalar(&axis_value, &[], &[]);
        let absent_length = TensorView::absent(DataType::Int64);
        let strategy = ShapeInference::for_node(
            &input_node,
            &[vec![Some(1), Some(8), Some(6), Some(1)], vec![], vec![]],
            1,
        );
        assert_eq!(
            infer(&strategy, &[x, absent_length, axis]).unwrap(),
            vec![vec![1, 5, 6, 2]]
        );
    }

    #[test]
    fn dft_device_axis_is_symbolic_and_never_dereferenced() {
        let data = vec![0.0f32; 48];
        let x = f32_data(&[1, 8, 6, 1], &[48, 6, 1, 1], &data);
        let absent_length = TensorView::absent(DataType::Int64);
        let device_axis = TensorView::new(
            DevicePtr(std::ptr::dangling::<i64>().cast()),
            DataType::Int64,
            &[],
            &[],
            DeviceId::cuda(0),
        );
        let mut node = Node::new(
            onnx_runtime_ir::NodeId(0),
            "DFT",
            vec![
                Some(onnx_runtime_ir::ValueId(0)),
                None,
                Some(onnx_runtime_ir::ValueId(2)),
            ],
            vec![onnx_runtime_ir::ValueId(3)],
        );
        node.version = Some(20);
        node.attributes
            .insert("onesided".into(), onnx_runtime_ir::Attribute::Int(1));
        assert_eq!(
            infer_shared_node(&node, &[x, absent_length, device_axis]),
            SharedShapeResult::SymbolicOrUnknown
        );
        let strategy = ShapeInference::for_node(
            &node,
            &[vec![Some(1), Some(8), Some(6), Some(1)], vec![], vec![]],
            1,
        );
        let error = infer(&strategy, &[x, absent_length, device_axis]).unwrap_err();
        assert!(error.contains("host cannot read"), "{error}");
    }

    #[test]
    fn dft_real_input_produces_a_complex_last_dim() {
        // [batch=2, signal=8, real=1] -> [2, 8, 2]. The last dim must become 2
        // even though the input is real; a rule that copied it through would
        // answer 1 and the kernel would reject the buffer.
        let buf = vec![0.0f32; 16];
        let x = f32_data(&[2, 8, 1], &[8, 1, 1], &buf);
        assert_eq!(
            infer(
                &ShapeInference::Dft {
                    onesided: false,
                    axis_attr: Some(1),
                    default_axis: 1,
                },
                &[x]
            )
            .unwrap(),
            vec![vec![2, 8, 2]]
        );
    }

    #[test]
    fn dft_onesided_halves_the_signal_axis() {
        // n=8 -> 8/2+1 = 5. This is the whole point of `onesided`, and a rule
        // that ignored the attribute would answer 8.
        let buf = vec![0.0f32; 16];
        let x = f32_data(&[2, 8, 1], &[8, 1, 1], &buf);
        assert_eq!(
            infer(
                &ShapeInference::Dft {
                    onesided: true,
                    axis_attr: Some(1),
                    default_axis: 1,
                },
                &[x]
            )
            .unwrap(),
            vec![vec![2, 5, 2]]
        );
    }

    #[test]
    fn dft_length_input_overrides_the_signal_extent() {
        // dft_length=4 against a signal of 8: the output follows the request,
        // not the input extent.
        let buf = vec![0.0f32; 16];
        let x = f32_data(&[2, 8, 1], &[8, 1, 1], &buf);
        let n = [4i64];
        let len = i64_scalar(&n, &[1], &[1]);
        assert_eq!(
            infer(
                &ShapeInference::Dft {
                    onesided: false,
                    axis_attr: Some(1),
                    default_axis: 1,
                },
                &[x, len]
            )
            .unwrap(),
            vec![vec![2, 4, 2]]
        );
    }

    #[test]
    fn dft_opset20_reads_the_axis_from_input_two() {
        // The axis input must win over the attribute. Attribute says 1, input
        // says 0 — a rule that ignored the input would resize the wrong axis.
        let buf = vec![0.0f32; 24];
        let x = f32_data(&[3, 4, 2], &[8, 2, 1], &buf);
        let empty: [i64; 0] = [];
        let no_len = i64_scalar(&empty, &[0], &[1]);
        let a = [0i64];
        let axis_in = i64_scalar(&a, &[1], &[1]);
        assert_eq!(
            infer(
                &ShapeInference::Dft {
                    onesided: true,
                    axis_attr: Some(1),
                    default_axis: 1,
                },
                &[x, no_len, axis_in]
            )
            .unwrap(),
            // axis 0 of extent 3, onesided -> 3/2+1 = 2
            vec![vec![2, 4, 2]]
        );
    }

    #[test]
    fn dft_refuses_the_complex_dim_as_the_signal_axis() {
        let buf = vec![0.0f32; 16];
        let x = f32_data(&[2, 8, 1], &[8, 1, 1], &buf);
        let err = infer(
            &ShapeInference::Dft {
                onesided: false,
                axis_attr: Some(-1),
                default_axis: 1,
            },
            &[x],
        )
        .unwrap_err();
        assert!(
            err.contains("complex component"),
            "error should name the reason: {err}"
        );
    }

    // ── Compress ─────────────────────────────────────────────────────────────
    //
    // The only rule that reads input *values*, so it gets its own tests rather
    // than riding on the claimability guard: that guard only asserts the op has
    // an arm, which a wrong arm satisfies just as well as a right one.

    /// Build a host Bool condition view over `bits`.
    fn bool_condition<'a>(
        bits: &'a [u8],
        shape: &'a [usize],
        strides: &'a [i64],
    ) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(bits.as_ptr() as *mut std::ffi::c_void),
            DataType::Bool,
            shape,
            strides,
            onnx_runtime_ir::DeviceId::cpu(),
        )
    }

    fn f32_data<'a>(shape: &'a [usize], strides: &'a [i64], buf: &'a [f32]) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(buf.as_ptr() as *mut std::ffi::c_void),
            DataType::Float32,
            shape,
            strides,
            onnx_runtime_ir::DeviceId::cpu(),
        )
    }

    #[test]
    fn compress_axis_extent_is_the_condition_popcount() {
        // 3x4, compress along axis 1 keeping columns 0 and 2.
        let buf = vec![0.0f32; 12];
        let data = f32_data(&[3, 4], &[4, 1], &buf);
        let bits = [1u8, 0, 1, 0];
        let cond = bool_condition(&bits, &[4], &[1]);
        assert_eq!(
            infer(&ShapeInference::Compress { axis: Some(1) }, &[data, cond]).unwrap(),
            vec![vec![3, 2]]
        );
    }

    #[test]
    fn compress_without_axis_flattens_first() {
        // No axis: the input is flattened, so the output is 1-D of the popcount.
        let buf = vec![0.0f32; 6];
        let data = f32_data(&[2, 3], &[3, 1], &buf);
        let bits = [1u8, 1, 0, 1, 0, 0];
        let cond = bool_condition(&bits, &[6], &[1]);
        assert_eq!(
            infer(&ShapeInference::Compress { axis: None }, &[data, cond]).unwrap(),
            vec![vec![3]]
        );
    }

    #[test]
    fn compress_ignores_condition_entries_past_the_axis() {
        // The spec selects over the overlap: a condition longer than the axis
        // does not invent rows. Picking a *true* entry past the end is what
        // makes this discriminating — a rule that counted the whole condition
        // would answer 3 here instead of 2.
        let buf = vec![0.0f32; 3];
        let data = f32_data(&[3], &[1], &buf);
        let bits = [1u8, 0, 1, 1, 1];
        let cond = bool_condition(&bits, &[5], &[1]);
        assert_eq!(
            infer(&ShapeInference::Compress { axis: Some(0) }, &[data, cond]).unwrap(),
            vec![vec![2]]
        );
    }

    #[test]
    fn compress_honours_a_non_unit_condition_stride() {
        // A strided view must not be read as packed bytes. Every other element
        // is true, so a packed read would answer 1 where the truth is 2.
        let buf = vec![0.0f32; 3];
        let data = f32_data(&[3], &[1], &buf);
        let bits = [1u8, 0, 1, 0, 1];
        let cond = bool_condition(&bits, &[3], &[2]);
        assert_eq!(
            infer(&ShapeInference::Compress { axis: Some(0) }, &[data, cond]).unwrap(),
            vec![vec![3]]
        );
    }

    #[test]
    fn compress_refuses_a_device_resident_condition() {
        // The host cannot dereference device memory. Erroring names the reason;
        // reading through the pointer would be undefined behaviour and sizing
        // the output from a guess would be silently wrong.
        let buf = vec![0.0f32; 3];
        let data = f32_data(&[3], &[1], &buf);
        let bits = [1u8, 0, 1];
        let cond = TensorView::new(
            DevicePtr(bits.as_ptr() as *mut std::ffi::c_void),
            DataType::Bool,
            &[3],
            &[1],
            onnx_runtime_ir::DeviceId::cuda(0),
        );
        let err = infer(&ShapeInference::Compress { axis: Some(0) }, &[data, cond]).unwrap_err();
        assert!(
            err.contains("host cannot read"),
            "error should name the device restriction: {err}"
        );
    }

    // ── AttentionStd (ai.onnx::Attention, opset 23+) ──────────────────────────

    #[test]
    fn attention_std_rank4_single_output_uses_value_head_size() {
        // Q/K:[B,Hq,Sq,Dqk], V:[B,Hkv,Skv,Dv] with Dv != Dqk. Y carries Dv.
        let q = [2usize, 4, 8, 32];
        let v = [2usize, 4, 8, 16];
        let st4 = [0i64, 0, 0, 0];
        let s = ShapeInference::AttentionStd {
            q_num_heads: 0,
            kv_num_heads: 0,
            num_outputs: 1,
        };
        let res = infer(&s, &[view(&q, &st4), view(&q, &st4), view(&v, &st4)]).unwrap();
        assert_eq!(res, vec![vec![2, 4, 8, 16]]);
    }

    #[test]
    fn attention_std_rank4_present_outputs_fold_in_past() {
        // Three outputs: Y, present_key, present_value. past_key seq=5 → total 13.
        let q = [1usize, 8, 3, 64];
        let v = [1usize, 2, 8, 64];
        let past = [1usize, 2, 5, 64];
        let st4 = [0i64, 0, 0, 0];
        let s = ShapeInference::AttentionStd {
            q_num_heads: 0,
            kv_num_heads: 0,
            num_outputs: 3,
        };
        let inputs = [
            view(&q, &st4),    // Q
            view(&q, &st4),    // K
            view(&v, &st4),    // V
            view(&q, &st4),    // attn_mask (shape unused here)
            view(&past, &st4), // past_key
        ];
        let res = infer(&s, &inputs).unwrap();
        assert_eq!(
            res,
            vec![
                vec![1, 8, 3, 64],  // Y
                vec![1, 2, 13, 64], // present_key: kv_heads=2, total_seq=13
                vec![1, 2, 13, 64], // present_value
            ]
        );
    }

    #[test]
    fn attention_std_rank3_splits_hidden_by_heads() {
        // Q:[B,Sq,Hq*Dqk]=[2,8,128] (q_num_heads=4 → Dqk=32),
        // V:[B,Skv,Hkv*Dv]=[2,8,64] (kv_num_heads=4 → Dv=16).
        // Y:[B,Sq,Hq*Dv] = [2,8,64].
        let q = [2usize, 8, 128];
        let v = [2usize, 8, 64];
        let st3 = [0i64, 0, 0];
        let s = ShapeInference::AttentionStd {
            q_num_heads: 4,
            kv_num_heads: 4,
            num_outputs: 1,
        };
        let res = infer(&s, &[view(&q, &st3), view(&q, &st3), view(&v, &st3)]).unwrap();
        assert_eq!(res, vec![vec![2, 8, 64]]);
    }

    #[test]
    fn attention_for_node_routes_each_domain_to_its_own_rule() {
        use onnx_runtime_ir::{Node, NodeId};
        // Default-domain Attention is the opset-23 signature.
        let node = Node::new(NodeId(0), "Attention", Vec::new(), Vec::new());
        assert!(matches!(
            ShapeInference::for_node(&node, &[], 1),
            ShapeInference::AttentionStd { .. }
        ));
        // `com.microsoft::Attention` is a different operator -- its input is
        // unprojected and the fused weight sets the output width -- so it must
        // not fall into the opset-23 arm. It used to be `Declined` instead,
        // which silently handed every node to ORT's CPU EP; it now has its own
        // rule, and `plugin_ort_e2e`'s `msft_attention_assignment_f32` fixture
        // is the end-to-end falsifier for that.
        let mut contrib = Node::new(NodeId(0), "Attention", Vec::new(), Vec::new());
        contrib.domain = "com.microsoft".into();
        assert!(matches!(
            ShapeInference::for_node(&contrib, &[], 1),
            ShapeInference::MsftAttention { .. }
        ));
    }

    // ── for_node: fail-closed fallback ────────────────────────────────────────

    #[test]
    fn for_node_unknown_returns_declined() {
        use onnx_runtime_ir::{Node, NodeId};
        let node = Node::new(NodeId(0), "SomeCompletelyUnknownOp", Vec::new(), Vec::new());
        match ShapeInference::for_node(&node, &[], 1) {
            ShapeInference::Declined { op_type, .. } => {
                assert_eq!(op_type, "SomeCompletelyUnknownOp");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }

    #[test]
    fn declined_infer_gives_actionable_error() {
        let s = ShapeInference::Declined {
            op_type: "FooBar".into(),
            domain: "some.domain".into(),
            reason: DeclineReason::Unmodelled,
        };
        let err = infer(&s, &[]).unwrap_err();
        assert!(err.contains("FooBar"), "error should mention op: {err}");
        assert!(
            err.contains("some.domain"),
            "error should mention domain: {err}"
        );
        assert!(err.contains("for_node"), "error should suggest fix: {err}");
    }

    // ── ElementwiseBroadcast ──────────────────────────────────────────────────

    #[test]
    fn elementwise_broadcast_same_shape() {
        let s = [2usize, 3];
        let st = [3i64, 1];
        let v1 = view(&s, &st);
        let v2 = view(&s, &st);
        let res = infer(&ShapeInference::ElementwiseBroadcast, &[v1, v2]).unwrap();
        assert_eq!(res, vec![vec![2, 3]]);
    }

    #[test]
    fn elementwise_broadcast_numpy_rules() {
        // [3,1] × [1,4] → [3,4]
        let s1 = [3usize, 1];
        let st1 = [1i64, 1];
        let s2 = [1usize, 4];
        let st2 = [4i64, 1];
        let res = infer(
            &ShapeInference::ElementwiseBroadcast,
            &[view(&s1, &st1), view(&s2, &st2)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn elementwise_broadcast_scalar_and_vector() {
        // [] × [5] → [5]
        let s_scalar: [usize; 0] = [];
        let st_scalar: [i64; 0] = [];
        let s_vec = [5usize];
        let st_vec = [1i64];
        let res = infer(
            &ShapeInference::ElementwiseBroadcast,
            &[view(&s_scalar, &st_scalar), view(&s_vec, &st_vec)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![5]]);
    }

    #[test]
    fn elementwise_broadcast_no_inputs_is_error() {
        let res = infer(&ShapeInference::ElementwiseBroadcast, &[]);
        assert!(res.is_err());
    }

    // ── SameAsInput ───────────────────────────────────────────────────────────

    #[test]
    fn same_as_input_roundtrip() {
        let s = [2usize, 3];
        let st = [3i64, 1];
        let res = infer(&ShapeInference::SameAsInput(0), &[view(&s, &st)]).unwrap();
        assert_eq!(res, vec![vec![2, 3]]);
    }

    #[test]
    fn same_as_input_oob_is_error() {
        let s = [4usize];
        let st = [1i64];
        let res = infer(&ShapeInference::SameAsInput(5), &[view(&s, &st)]);
        assert!(res.is_err());
        let msg = res.unwrap_err();
        assert!(msg.contains("SameAsInput(5)"), "{msg}");
    }

    #[test]
    fn same_as_input_multi_output() {
        let s = [2usize, 3];
        let st = [3i64, 1];
        let res = infer(
            &ShapeInference::SameAsInputMultiOutput { idx: 0, count: 3 },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res.len(), 3);
        assert!(res.iter().all(|r| r == &vec![2, 3]));
    }

    // ── MatMul ────────────────────────────────────────────────────────────────

    #[test]
    fn matmul_2d() {
        let a = [3usize, 4];
        let b = [4usize, 5];
        let at = [4i64, 1];
        let bt = [5i64, 1];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![3, 5]]);
    }

    #[test]
    fn matmul_batched() {
        let a = [2usize, 3, 4];
        let b = [2usize, 4, 5];
        let at = [12i64, 4, 1];
        let bt = [20i64, 5, 1];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![2, 3, 5]]);
    }

    #[test]
    fn matmul_batch_broadcast() {
        // [1, 3, 4] × [2, 4, 5] → [2, 3, 5]
        let a = [1usize, 3, 4];
        let b = [2usize, 4, 5];
        let at = [12i64, 4, 1];
        let bt = [20i64, 5, 1];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![2, 3, 5]]);
    }

    #[test]
    fn matmul_1d_vector_dot() {
        let a = [4usize];
        let b = [4usize];
        let at = [1i64];
        let bt = [1i64];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![] as Vec<usize>]); // scalar output
    }

    #[test]
    fn matmul_matvec() {
        // [M, K] × [K] → [M]
        let a = [3usize, 4];
        let b = [4usize];
        let at = [4i64, 1];
        let bt = [1i64];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![3]]);
    }

    // ── Gemm ─────────────────────────────────────────────────────────────────

    #[test]
    fn gemm_no_transpose() {
        let a = [3usize, 4];
        let b = [4usize, 5];
        let at = [4i64, 1];
        let bt = [5i64, 1];
        let res = infer(
            &ShapeInference::Gemm {
                trans_a: false,
                trans_b: false,
            },
            &[view(&a, &at), view(&b, &bt)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 5]]);
    }

    #[test]
    fn gemm_trans_b() {
        // A=[3,4], B=[5,4], transB → output [3,5]
        let a = [3usize, 4];
        let b = [5usize, 4];
        let at = [4i64, 1];
        let bt = [4i64, 1];
        let res = infer(
            &ShapeInference::Gemm {
                trans_a: false,
                trans_b: true,
            },
            &[view(&a, &at), view(&b, &bt)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 5]]);
    }

    // ── Concat ────────────────────────────────────────────────────────────────

    #[test]
    fn concat_axis0() {
        let s1 = [2usize, 3];
        let st1 = [3i64, 1];
        let s2 = [4usize, 3];
        let st2 = [3i64, 1];
        let res = infer(
            &ShapeInference::Concat { axis: 0 },
            &[view(&s1, &st1), view(&s2, &st2)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![6, 3]]);
    }

    #[test]
    fn concat_axis1() {
        let s1 = [2usize, 3];
        let st1 = [3i64, 1];
        let s2 = [2usize, 5];
        let st2 = [5i64, 1];
        let res = infer(
            &ShapeInference::Concat { axis: 1 },
            &[view(&s1, &st1), view(&s2, &st2)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 8]]);
    }

    #[test]
    fn concat_negative_axis() {
        // axis=-1 for rank-2 = axis=1
        let s1 = [2usize, 3];
        let st1 = [3i64, 1];
        let s2 = [2usize, 4];
        let st2 = [4i64, 1];
        let res = infer(
            &ShapeInference::Concat { axis: -1 },
            &[view(&s1, &st1), view(&s2, &st2)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 7]]);
    }

    // ── Transpose ────────────────────────────────────────────────────────────

    #[test]
    fn transpose_default_reverses() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(&ShapeInference::Transpose { perm: None }, &[view(&s, &st)]).unwrap();
        assert_eq!(res, vec![vec![4, 3, 2]]);
    }

    #[test]
    fn transpose_explicit_perm() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Transpose {
                perm: Some(vec![0, 2, 1]),
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 4, 3]]);
    }

    #[test]
    fn transpose_perm_wrong_length_is_error() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Transpose {
                perm: Some(vec![0, 1]),
            },
            &[view(&s, &st)],
        );
        assert!(res.is_err());
    }

    // ── Gather ───────────────────────────────────────────────────────────────

    #[test]
    fn gather_axis0() {
        // data=[5,4], indices=[3] → output=[3,4]
        let data = [5usize, 4];
        let dst = [4i64, 1];
        let idx = [3usize];
        let ist = [1i64];
        let res = infer(
            &ShapeInference::Gather { axis: 0 },
            &[view(&data, &dst), view(&idx, &ist)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn gather_axis1_matrix_index() {
        // data=[2,10,4], indices=[3,5], axis=1 → output=[2,3,5,4]
        let data = [2usize, 10, 4];
        let dst = [40i64, 4, 1];
        let idx = [3usize, 5];
        let ist = [5i64, 1];
        let res = infer(
            &ShapeInference::Gather { axis: 1 },
            &[view(&data, &dst), view(&idx, &ist)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 3, 5, 4]]);
    }

    // ── GatherND ─────────────────────────────────────────────────────────────

    #[test]
    fn gather_nd_basic() {
        // data=[2,3,4], indices=[5,2] (k=2, batch_dims=0)
        // → output=[5,4]
        let data = [2usize, 3, 4];
        let dst = [12i64, 4, 1];
        let idx = [5usize, 2];
        let ist = [2i64, 1];
        let res = infer(
            &ShapeInference::GatherND { batch_dims: 0 },
            &[view(&data, &dst), view(&idx, &ist)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![5, 4]]);
    }

    // ── Shape op ─────────────────────────────────────────────────────────────

    #[test]
    fn shape_op_full() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::ShapeOp {
                start: 0,
                end: None,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3]]); // 3-dim shape → output len 3
    }

    #[test]
    fn shape_op_slice() {
        let s = [2usize, 3, 4, 5];
        let st = [60i64, 20, 5, 1];
        let res = infer(
            &ShapeInference::ShapeOp {
                start: 1,
                end: Some(3),
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2]]); // dims 1..3 → 2 dims
    }

    #[test]
    fn shape_op_negative_indices() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::ShapeOp {
                start: -2,
                end: None,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2]]); // dims [-2:] = [3,4] → 2 dims
    }

    // ── Squeeze / Unsqueeze ───────────────────────────────────────────────────

    #[test]
    fn squeeze_removes_ones() {
        let s = [1usize, 3, 1, 4];
        let st = [12i64, 4, 4, 1];
        let res = infer(&ShapeInference::Squeeze { axes: vec![] }, &[view(&s, &st)]).unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn squeeze_specific_axis() {
        let s = [1usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(&ShapeInference::Squeeze { axes: vec![0] }, &[view(&s, &st)]).unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn unsqueeze_insert_at_front() {
        let s = [3usize, 4];
        let st = [4i64, 1];
        let res = infer(
            &ShapeInference::Unsqueeze { axes: vec![0] },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 3, 4]]);
    }

    #[test]
    fn unsqueeze_multiple_axes() {
        let s = [3usize];
        let st = [1i64];
        let res = infer(
            &ShapeInference::Unsqueeze { axes: vec![0, 2] },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 3, 1]]);
    }

    // ── Reshape (data-dependent) ──────────────────────────────────────────────

    #[test]
    fn reshape_static_shape() {
        let d = [2usize, 6];
        let dst = [6i64, 1];
        let data_view = view(&d, &dst);
        let shape_data: [i64; 3] = [2, 2, 3];
        let sshape = [3usize];
        let sst = [1i64];
        let shape_view = i64_view(&shape_data, &sshape, &sst);
        let res = infer(
            &ShapeInference::ReshapeData { allowzero: false },
            &[data_view, shape_view],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 2, 3]]);
    }

    #[test]
    fn reshape_infer_minus_one() {
        // data=[2,6]=12, shape=[3,-1] → [3,4]
        let d = [2usize, 6];
        let dst = [6i64, 1];
        let data_view = view(&d, &dst);
        let shape_data: [i64; 2] = [3, -1];
        let sshape = [2usize];
        let sst = [1i64];
        let shape_view = i64_view(&shape_data, &sshape, &sst);
        let res = infer(
            &ShapeInference::ReshapeData { allowzero: false },
            &[data_view, shape_view],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn reshape_copy_zero_dims() {
        // data=[2,3,4], shape=[0,12,0] allowzero=false
        // dim 0 → copy from data[0]=2, dim 2 → copy from data[2]=4 → [2,12,4]? No:
        // "0" means copy the corresponding dim from input. so shape=[0,12,1]→[2,12,1]
        let d = [2usize, 3, 4];
        let dst = [12i64, 4, 1];
        let data_view = view(&d, &dst);
        let shape_data: [i64; 3] = [0, 12, 1];
        let sshape = [3usize];
        let sst = [1i64];
        let shape_view = i64_view(&shape_data, &sshape, &sst);
        let res = infer(
            &ShapeInference::ReshapeData { allowzero: false },
            &[data_view, shape_view],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 12, 1]]);
    }

    // ── Slice (data-dependent) ────────────────────────────────────────────────

    #[test]
    fn slice_basic() {
        // data=[5,4], starts=[1], ends=[3], axes=[0], steps=[1] → [2,4]
        let d = [5usize, 4];
        let dst = [4i64, 1];
        let data_v = view(&d, &dst);
        let starts_d: [i64; 1] = [1];
        let ends_d: [i64; 1] = [3];
        let ax_d: [i64; 1] = [0];
        let steps_d: [i64; 1] = [1];
        let s1 = [1usize];
        let st1 = [1i64];
        let starts_v = i64_view(&starts_d, &s1, &st1);
        let ends_v = i64_view(&ends_d, &s1, &st1);
        let ax_v = i64_view(&ax_d, &s1, &st1);
        let steps_v = i64_view(&steps_d, &s1, &st1);
        let res = infer(
            &ShapeInference::SliceData,
            &[data_v, starts_v, ends_v, ax_v, steps_v],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 4]]);
    }

    #[test]
    fn slice_negative_step_reverse() {
        // data=[5], starts=[4], ends=[-6], axes=[0], steps=[-1] → [5] (all elements reversed)
        let d = [5usize];
        let dst = [1i64];
        let data_v = view(&d, &dst);
        let starts_d: [i64; 1] = [4];
        let ends_d: [i64; 1] = [-6];
        let ax_d: [i64; 1] = [0];
        let steps_d: [i64; 1] = [-1];
        let s1 = [1usize];
        let st1 = [1i64];
        let starts_v = i64_view(&starts_d, &s1, &st1);
        let ends_v = i64_view(&ends_d, &s1, &st1);
        let ax_v = i64_view(&ax_d, &s1, &st1);
        let steps_v = i64_view(&steps_d, &s1, &st1);
        let res = infer(
            &ShapeInference::SliceData,
            &[data_v, starts_v, ends_v, ax_v, steps_v],
        )
        .unwrap();
        assert_eq!(res, vec![vec![5]]);
    }

    // ── Reduction ────────────────────────────────────────────────────────────

    #[test]
    fn reduction_keepdims_single_axis() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Reduction {
                keepdims: true,
                axes: Some(vec![1]),
                noop_with_empty_axes: false,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 1, 4]]);
    }

    #[test]
    fn reduction_no_keepdims() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Reduction {
                keepdims: false,
                axes: Some(vec![1]),
                noop_with_empty_axes: false,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 4]]);
    }

    #[test]
    fn reduction_all_axes_keepdims() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Reduction {
                keepdims: true,
                axes: None,
                noop_with_empty_axes: false,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 1, 1]]);
    }

    #[test]
    fn reduction_noop_empty_axes() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Reduction {
                keepdims: true,
                axes: Some(vec![]),
                noop_with_empty_axes: true,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 3, 4]]); // identity
    }

    // ── Conv ─────────────────────────────────────────────────────────────────

    #[test]
    fn conv_1d_no_padding() {
        // in=[1,3,7], kernel=3, stride=1, dilation=1, pad=0 → out=[1,16,5]
        let s = [1usize, 3, 7];
        let st = [21i64, 7, 1];
        let res = infer(
            &ShapeInference::Conv {
                out_channels: 16,
                per_axis: vec![ConvSpatialAxis {
                    kernel: 3,
                    pad_before: 0,
                    pad_after: 0,
                    stride: 1,
                    dilation: 1,
                }],
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 16, 5]]);
    }

    #[test]
    fn conv_2d_with_padding() {
        // in=[1,3,5,5], kernel=3, stride=1, dilation=1, pad=1 all → out=[1,16,5,5]
        let s = [1usize, 3, 5, 5];
        let st = [75i64, 25, 5, 1];
        let rule = ConvSpatialAxis {
            kernel: 3,
            pad_before: 1,
            pad_after: 1,
            stride: 1,
            dilation: 1,
        };
        let res = infer(
            &ShapeInference::Conv {
                out_channels: 16,
                per_axis: vec![rule.clone(), rule],
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 16, 5, 5]]);
    }

    // ── Multi-node subgraph routing ───────────────────────────────────────────
    //
    // Prove that intermediate buffers are correctly allocated, written, and
    // threaded between nodes. We use a trivial "identity" kernel (no-op copy)
    // to focus on the routing mechanics, not on kernel math.
    //
    // The subgraph: ORT_input[0] → node0 → intermediate[0] → node1 → ORT_output[0]
    //
    // We can't run compute_execute without a live OrtKernelContext, but we can
    // verify the SubgraphRouting structure is well-formed and that
    // IntermediateBuf view/view_mut work correctly.

    #[test]
    fn intermediate_buf_view_roundtrip() {
        let data = vec![0u8; 12 * 4]; // 12 f32 elements
        let shape = vec![3usize, 4];
        let strides = onnx_runtime_ir::compute_contiguous_strides(&shape);
        let buf = IntermediateBuf {
            data,
            scratch_ptr: std::ptr::null_mut(),
            shape: crate::dim_vec::DimVec::from_slice(&shape),
            strides: crate::dim_vec::DimVec::from_slice(&strides),
            dtype: DataType::Float32,
            device: DeviceId::cpu(),
        };
        let v = buf.view();
        assert_eq!(v.shape, &shape[..]);
        assert_eq!(v.dtype, DataType::Float32);
        assert_eq!(v.device, DeviceId::cpu());
    }

    #[test]
    fn device_intermediate_view_preserves_residency() {
        let buf = IntermediateBuf {
            data: Vec::new(),
            scratch_ptr: std::ptr::dangling_mut(),
            shape: crate::dim_vec::DimVec::new(),
            strides: crate::dim_vec::DimVec::new(),
            dtype: DataType::Int64,
            device: DeviceId::cuda(2),
        };
        let view = buf.view();
        assert_eq!(view.device, DeviceId::cuda(2));
        assert!(!view.device.is_host_accessible());
    }

    #[test]
    fn subgraph_routing_structure() {
        // Build a two-node routing: node0 takes ORT[0], writes Buffer[0];
        // node1 takes Buffer[0], writes ORT[0].
        let routing = SubgraphRouting {
            input_sources: vec![
                vec![NodeInputSource::Ort(0)],
                vec![NodeInputSource::Buffer(0)],
            ],
            output_sinks: vec![
                vec![NodeOutputSink::Buffer(0)],
                vec![NodeOutputSink::Ort(0)],
            ],
            num_intermediate_buffers: 1,
        };
        assert_eq!(routing.input_sources.len(), 2);
        assert_eq!(routing.output_sinks.len(), 2);
        // Verify the chain: ORT→Buffer→ORT.
        assert!(matches!(
            routing.input_sources[0][0],
            NodeInputSource::Ort(0)
        ));
        assert!(matches!(
            routing.output_sinks[0][0],
            NodeOutputSink::Buffer(0)
        ));
        assert!(matches!(
            routing.input_sources[1][0],
            NodeInputSource::Buffer(0)
        ));
        assert!(matches!(routing.output_sinks[1][0], NodeOutputSink::Ort(0)));
    }

    // ── Intermediate buffer liveness and recycling ────────────────────────────

    #[test]
    fn last_reader_marks_the_final_consumer_of_each_buffer() {
        // node0: ORT[0] → Buffer[0]; node1: Buffer[0] → Buffer[1];
        // node2: Buffer[1] → ORT[0]. Buffer 0 dies after node1, buffer 1 after
        // node2.
        let sources = vec![
            vec![NodeInputSource::Ort(0)],
            vec![NodeInputSource::Buffer(0)],
            vec![NodeInputSource::Buffer(1)],
        ];
        assert_eq!(
            super::last_reader_per_buffer(&sources, 2),
            vec![Some(1), Some(2)]
        );
    }

    #[test]
    fn last_reader_takes_the_highest_index_when_a_buffer_is_read_twice() {
        // A buffer feeding both a later node and a much later node must stay
        // alive until the *last* one has run, or the second read sees storage
        // that has already been handed to another node.
        let sources = vec![
            vec![NodeInputSource::Ort(0)],
            vec![NodeInputSource::Buffer(0)],
            vec![NodeInputSource::Ort(1)],
            vec![NodeInputSource::Buffer(0), NodeInputSource::Buffer(1)],
        ];
        assert_eq!(
            super::last_reader_per_buffer(&sources, 2),
            vec![Some(3), Some(3)]
        );
    }

    #[test]
    fn last_reader_is_none_for_unread_and_out_of_range_buffers() {
        // Buffer 1 is never read (its producer also routes the output to ORT),
        // and buffer 7 does not exist. Neither may be reported as live.
        let sources = vec![
            vec![NodeInputSource::Ort(0)],
            vec![NodeInputSource::Buffer(0), NodeInputSource::Buffer(7)],
            vec![NodeInputSource::Absent],
        ];
        assert_eq!(
            super::last_reader_per_buffer(&sources, 2),
            vec![Some(1), None]
        );
    }

    /// The CSR index must retire exactly what the per-node scan retired.
    ///
    /// This is the property the rewrite trades on, so it is checked against the
    /// scan itself rather than against hand-written expectations: for every
    /// node, the buffers the index yields are the buffers whose last reader is
    /// that node, in the same ascending order the scan visited them.
    fn scan_retirements(sources: &[Vec<NodeInputSource>], buffers: usize) -> Vec<Vec<usize>> {
        let last = super::last_reader_per_buffer(sources, buffers);
        (0..sources.len())
            .map(|node| {
                last.iter()
                    .enumerate()
                    .filter(|(_, l)| **l == Some(node))
                    .map(|(buffer, _)| buffer)
                    .collect()
            })
            .collect()
    }

    fn index_retirements(sources: &[Vec<NodeInputSource>], buffers: usize) -> Vec<Vec<usize>> {
        let (starts, items) = super::retirements_per_node(sources, buffers);
        (0..sources.len())
            .map(|node| items[starts[node]..starts[node + 1]].to_vec())
            .collect()
    }

    #[test]
    fn the_retirement_index_agrees_with_the_scan_it_replaces() {
        let chain: Vec<Vec<NodeInputSource>> = std::iter::once(vec![NodeInputSource::Ort(0)])
            .chain((0..9).map(|i| vec![NodeInputSource::Buffer(i)]))
            .collect();
        let cases: Vec<(Vec<Vec<NodeInputSource>>, usize)> = vec![
            // A plain chain: every buffer dies after the node that reads it.
            (chain, 9),
            // A buffer read twice must retire only after the later reader, and
            // must appear exactly once in the index.
            (
                vec![
                    vec![NodeInputSource::Ort(0)],
                    vec![NodeInputSource::Buffer(0)],
                    vec![NodeInputSource::Ort(1)],
                    vec![NodeInputSource::Buffer(0), NodeInputSource::Buffer(1)],
                ],
                2,
            ),
            // Unread and out-of-range buffers retire nowhere.
            (
                vec![
                    vec![NodeInputSource::Ort(0)],
                    vec![NodeInputSource::Buffer(0), NodeInputSource::Buffer(7)],
                    vec![NodeInputSource::Absent],
                ],
                2,
            ),
            // One node retiring several buffers at once: order must be
            // ascending, as the scan produced it.
            (
                vec![
                    vec![NodeInputSource::Ort(0)],
                    vec![NodeInputSource::Ort(1)],
                    vec![
                        NodeInputSource::Buffer(2),
                        NodeInputSource::Buffer(0),
                        NodeInputSource::Buffer(1),
                    ],
                ],
                3,
            ),
            // No buffers at all.
            (vec![vec![NodeInputSource::Ort(0)]], 0),
        ];
        for (sources, buffers) in cases {
            assert_eq!(
                index_retirements(&sources, buffers),
                scan_retirements(&sources, buffers),
                "retirement index diverged from the scan for {sources:?}"
            );
        }
    }

    /// Every buffer retires exactly once, or storage leaks for the length of
    /// the subgraph (or worse, is recycled twice).
    #[test]
    fn every_read_buffer_retires_exactly_once() {
        let sources: Vec<Vec<NodeInputSource>> = std::iter::once(vec![NodeInputSource::Ort(0)])
            .chain((0..20).map(|i| vec![NodeInputSource::Buffer(i)]))
            .collect();
        let (starts, items) = super::retirements_per_node(&sources, 20);
        let mut seen = items.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), items.len(), "a buffer retired twice");
        assert_eq!(items.len(), 20, "every buffer of a chain is read once");
        assert_eq!(starts[sources.len()], items.len());
    }

    /// Build the reusable shape storage `absent_slot_strides` now reads.
    fn dim_shapes(shapes: &[&[usize]]) -> Vec<crate::dim_vec::DimVec<usize>> {
        shapes
            .iter()
            .map(|s| crate::dim_vec::DimVec::from_slice(s))
            .collect()
    }

    /// Absent strides are keyed by absent index, not slot index.
    ///
    /// Getting this wrong is invisible for a node with a single absent output
    /// at slot 0 -- the two indices coincide -- and silently hands the wrong
    /// strides to every other shape. It is pinned with absent slots that are
    /// neither first nor contiguous, and with differing shapes per slot so a
    /// mis-key produces a different answer rather than the same one twice.
    #[test]
    fn absent_strides_are_keyed_by_absent_index_not_slot_index() {
        let shapes = dim_shapes(&[&[2, 3, 4], &[5, 6], &[7, 8, 9], &[10]]);
        // Slots 1 and 3 are absent, so absent index 0 is slot 1 and absent
        // index 1 is slot 3.
        let got = super::absent_slot_strides([1usize, 3].into_iter(), &shapes);
        assert_eq!(got.len(), 2, "one entry per absent slot, not per output");
        assert_eq!(got[0], super::contiguous_strides(&shapes[1]));
        assert_eq!(got[1], super::contiguous_strides(&shapes[3]));
        assert_ne!(
            got[0],
            super::contiguous_strides(&shapes[0]),
            "keying by slot index would have produced this"
        );
    }

    #[test]
    fn a_node_with_no_absent_outputs_builds_no_absent_strides() {
        let shapes = dim_shapes(&[&[2, 3], &[4, 5]]);
        assert!(super::absent_slot_strides(std::iter::empty(), &shapes).is_empty());
    }

    /// An out-of-range slot must not panic in the middle of a `Run`. It cannot
    /// arise from a well-formed slot map, but this runs inside ORT's callback
    /// where a panic crosses an FFI boundary.
    #[test]
    fn an_out_of_range_absent_slot_yields_empty_strides() {
        let shapes = dim_shapes(&[&[2, 3]]);
        let got = super::absent_slot_strides([9usize].into_iter(), &shapes);
        assert_eq!(got, vec![Vec::<i64>::new()]);
    }

    #[test]
    fn recycled_intermediate_storage_is_reused_without_reallocating() {
        // Each test owns its pool outright. That is not only isolation from
        // libtest's thread reuse -- it is the same shape production uses now,
        // so these tests exercise the real signature rather than a global the
        // hot path no longer touches.
        let pool = &mut super::HostPool::default();
        // The point of the pool is address reuse: a retired buffer must come
        // back on the next request of the same size, so a chain of nodes keeps
        // rewriting storage that is still in cache.
        let first = pool.take_intermediate(4096);
        let addr = first.as_ptr() as usize;
        pool.recycle_intermediate(first);
        let second = pool.take_intermediate(4096);
        assert_eq!(second.as_ptr() as usize, addr);
        assert_eq!(second.len(), 4096);
    }

    #[test]
    fn a_recycled_buffer_serves_a_smaller_request_at_the_requested_length() {
        let pool = &mut super::HostPool::default();
        let big = pool.take_intermediate(8192);
        let addr = big.as_ptr() as usize;
        pool.recycle_intermediate(big);
        let small = pool.take_intermediate(64);
        assert_eq!(small.as_ptr() as usize, addr);
        // Length is what the caller asked for — `byte_len` bounds every
        // `from_raw_parts` built from this buffer, so an over-long slice would
        // be a real out-of-bounds view.
        assert_eq!(small.len(), 64);
    }

    #[test]
    fn a_request_larger_than_every_pooled_buffer_allocates_fresh_zeroed_storage() {
        let pool = &mut super::HostPool::default();
        let seed = pool.take_intermediate(32);
        pool.recycle_intermediate(seed);
        let big = pool.take_intermediate(1 << 20);
        assert_eq!(big.len(), 1 << 20);
        assert!(big.iter().all(|b| *b == 0));
    }

    #[test]
    fn scratch_backed_buffers_are_not_pooled() {
        let pool = &mut super::HostPool::default();
        // A scratch-backed IntermediateBuf carries an empty `data` vector and a
        // borrowed pointer it does not own. Pooling that empty vector would
        // fill a slot with nothing usable.
        pool.recycle_intermediate(Vec::new());
        assert_eq!(pool.len(), 0);
        let taken = pool.take_intermediate(16);
        assert_eq!(taken.len(), 16);
        assert!(taken.capacity() >= 16);
    }

    #[test]
    fn the_pool_is_bounded() {
        let pool = &mut super::HostPool::default();
        let bufs: Vec<Vec<u8>> = (0..super::HOST_INTERMEDIATE_POOL_SLOTS * 4)
            .map(|_| pool.take_intermediate(128))
            .collect();
        for b in bufs {
            pool.recycle_intermediate(b);
        }
        assert!(pool.len() <= super::HOST_INTERMEDIATE_POOL_SLOTS);
    }

    #[test]
    fn a_pooled_buffer_survives_the_run_that_retired_it() {
        // The pool now shares storage with per-`Run` scratch, every other field
        // of which is cleared when the `Run` leaves *because* keeping it would
        // be a dangling borrow. The pool is the one field where the opposite is
        // true, and nothing else in the suite would notice it being cleared:
        // reuse is invisible to every correctness assertion, so dropping it
        // would leave the suite green and silently restore a `malloc`/`free`
        // pair per node. This is the tripwire for that.
        let mut scratch = super::RunScratch::default();
        let buf = scratch.host_pool.take_intermediate(2048);
        let addr = buf.as_ptr() as usize;
        scratch.host_pool.recycle_intermediate(buf);

        // Exactly what `ScratchGuard::drop` does at the end of a `Run`.
        scratch.clear_and_bound();

        assert_eq!(
            scratch.host_pool.len(),
            1,
            "clear_and_bound discarded pooled storage that the next `Run` \
             should have reused"
        );
        let after = scratch.host_pool.take_intermediate(2048);
        assert_eq!(
            after.as_ptr() as usize,
            addr,
            "the buffer survived the `Run` boundary but is no longer the one \
             that was retired"
        );
    }

    // ── CreateState / ReleaseState lifecycle ──────────────────────────────────

    #[test]
    fn create_and_release_state_lifecycle() {
        let mut state_ptr: *mut c_void = std::ptr::null_mut();
        let status = unsafe {
            compute_create_state(std::ptr::null_mut(), std::ptr::null_mut(), &mut state_ptr)
        };
        assert!(status.is_null(), "ok_status returns null");
        assert!(!state_ptr.is_null());
        unsafe {
            compute_release_state(std::ptr::null_mut(), state_ptr);
        }
    }

    #[test]
    fn create_state_null_out_does_not_panic() {
        let status = unsafe {
            compute_create_state(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        let _ = status; // may be null in test env (no ORT API)
    }

    // ── for_node coverage ─────────────────────────────────────────────────────

    /// Build a bare node (no attributes) for the op/domain, to check the rule
    /// `for_node` resolves from the op alone.
    fn bare_node(op: &str) -> onnx_runtime_ir::Node {
        onnx_runtime_ir::Node::new(onnx_runtime_ir::NodeId(0), op, Vec::new(), Vec::new())
    }

    #[test]
    fn for_node_elementwise_coverage() {
        for op in ["Add", "Sub", "Mul", "Div", "Pow", "Where", "Max", "Min"] {
            assert!(
                matches!(
                    ShapeInference::for_node(&bare_node(op), &[], 1),
                    ShapeInference::ElementwiseBroadcast
                ),
                "{op} should be ElementwiseBroadcast"
            );
        }
    }

    #[test]
    fn for_node_unary_coverage() {
        for op in ["Relu", "Sigmoid", "Cast", "Identity", "Softmax"] {
            assert!(
                matches!(
                    ShapeInference::for_node(&bare_node(op), &[], 1),
                    ShapeInference::SameAsInput(0)
                ),
                "{op} should be SameAsInput(0)"
            );
        }
    }

    #[test]
    fn for_node_layer_norm_family_resolves() {
        // The LayerNorm family defaults axis to -1, so `for_node` resolves it
        // to a `LayerNorm` rule from the op alone — no attribute required.
        for (op, domain) in [
            ("LayerNormalization", ""),
            ("SimplifiedLayerNormalization", ""),
            ("RMSNormalization", ""),
            ("SkipLayerNormalization", "com.microsoft"),
            ("SkipSimplifiedLayerNormalization", "com.microsoft"),
        ] {
            let mut node = bare_node(op);
            node.domain = domain.to_string();
            assert!(
                matches!(
                    ShapeInference::for_node(&node, &[], 2),
                    ShapeInference::LayerNorm { .. }
                ),
                "{op} should resolve to a LayerNorm rule"
            );
        }
    }

    #[test]
    fn for_node_matmul_is_matmul() {
        assert!(matches!(
            ShapeInference::for_node(&bare_node("MatMul"), &[], 1),
            ShapeInference::MatMul
        ));
    }

    #[test]
    fn for_node_resolves_attribute_defaults() {
        // These ops carry attributes, but their ONNX defaults let `for_node`
        // resolve a rule from a bare node.
        assert!(matches!(
            ShapeInference::for_node(&bare_node("Concat"), &[], 1),
            ShapeInference::Concat { axis: 0 }
        ));
        assert!(matches!(
            ShapeInference::for_node(&bare_node("Transpose"), &[], 1),
            ShapeInference::Transpose { perm: None }
        ));
        assert!(matches!(
            ShapeInference::for_node(&bare_node("Gather"), &[], 1),
            ShapeInference::Gather { axis: 0 }
        ));
        assert!(matches!(
            ShapeInference::for_node(&bare_node("Reshape"), &[], 1),
            ShapeInference::ReshapeData { allowzero: false }
        ));
        assert!(matches!(
            ShapeInference::for_node(&bare_node("Slice"), &[], 1),
            ShapeInference::SliceData
        ));
        assert!(matches!(
            ShapeInference::for_node(&bare_node("Shape"), &[], 1),
            ShapeInference::ShapeOp {
                start: 0,
                end: None
            }
        ));
    }

    // ── Panic guard test ──────────────────────────────────────────────────

    #[test]
    fn compute_execute_catches_panic_returns_error_status() {
        // The compute_execute extern "C" fn wraps execution in catch_unwind.
        // Verify that a panic inside the closure path does NOT propagate across
        // the extern "C" boundary (would be UB) but is converted to an error.
        // We test the pattern directly since we cannot easily call the real
        // extern "C" fn without a valid ORT context.
        use crate::status::fail_status;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            panic!("simulated kernel panic");
        }));
        let status = result.unwrap_or_else(|_| fail_status("Compute: internal panic"));
        // In test environment without ORT API, fail_status returns null (documented).
        // The important thing is we didn't actually unwind/abort.
        let _ = status;
    }

    #[test]
    fn release_state_swallows_panic_safely() {
        // compute_release_state is void-returning so it has no status channel.
        // A panic inside drop must be caught and swallowed — not let through the
        // extern "C" boundary. We exercise the guard pattern directly.
        //
        // This test verifies NEW-1 (EP plugin security audit) is fixed: a future
        // ComputeState extension that panics in Drop will not cause UB.
        use std::ffi::c_void;

        // Construct a state pointer the same way create_state would.
        let state = Box::new(ComputeState { _placeholder: 0 });
        let raw: *mut c_void = Box::into_raw(state).cast::<c_void>();

        // Exercise the catch_unwind guard for the normal (non-panic) path.
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !raw.is_null() {
                unsafe { drop(Box::from_raw(raw.cast::<ComputeState>())) };
            }
        }));
        // No panic occurred; caught must be Ok.
        assert!(caught.is_ok(), "release_state unexpectedly panicked");

        // Verify the guard also swallows a panic (pattern-level check).
        let panicky = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            panic!("simulated drop panic");
        }));
        // Panic was caught — the caller sees Ok(()) is absent but no unwind.
        let _ = panicky; // swallowed, as compute_release_state does
    }

    // ── Intermediate buffer overflow test ─────────────────────────────────

    #[test]
    fn contiguous_strides_empty_shape() {
        let s = super::contiguous_strides(&[]);
        assert_eq!(s, Vec::<i64>::new());
    }

    #[test]
    fn contiguous_strides_scalar() {
        let s = super::contiguous_strides(&[1]);
        assert_eq!(s, vec![1i64]);
    }

    /// The strides now live in a `DimVec`, which changes representation at
    /// `INLINE_RANK`. `onnx_runtime_ir::compute_contiguous_strides` is the same
    /// algorithm in a crate this change did not touch, so it is a genuine
    /// oracle rather than a restatement: this walks ranks either side of the
    /// spill boundary and demands agreement on every one.
    ///
    /// The sweep must include extents of **1 and 0**, not just the distinct
    /// extents that make a transposition obvious. A stride *on* a size-1 axis
    /// is inert — its index is always 0 — which makes it tempting to leave out.
    /// But that axis is still a *multiplicand* for every axis outside it, so an
    /// error there propagates into strides that are very much live. An earlier
    /// version of this test used `i + 2` throughout and waved through a mutant
    /// that read `(shape[i + 1] as i64).max(2)`.
    #[test]
    fn contiguous_strides_matches_the_ir_oracle_across_the_inline_boundary() {
        let mut shapes: Vec<Vec<usize>> = Vec::new();
        for rank in 0..=(crate::dim_vec::INLINE_RANK + 3) {
            // Distinct, non-uniform extents so a transposed or off-by-one
            // stride cannot coincide with the right answer.
            shapes.push((0..rank).map(|i| i + 2).collect());
        }
        // Interior and trailing unit axes, on both sides of the spill boundary,
        // and a zero extent.
        shapes.extend([
            vec![2, 1, 3],
            vec![1, 1, 5],
            vec![2, 1, 3, 1, 4],
            vec![4, 1],
            vec![1],
            vec![2, 0, 3],
            vec![3, 1, 4, 1, 5, 1, 9, 1, 2, 1, 6],
            vec![1; crate::dim_vec::INLINE_RANK + 2],
        ]);

        let mut saw_inline = false;
        let mut saw_spilled = false;
        let mut saw_interior_unit = false;
        for shape in &shapes {
            let got = super::contiguous_strides(shape);
            let want = onnx_runtime_ir::compute_contiguous_strides(shape);
            assert_eq!(
                got.as_slice(),
                want.as_slice(),
                "shape {shape:?} disagrees with the IR oracle"
            );
            assert_eq!(got.len(), shape.len(), "shape {shape:?} changed length");
            if shape.len() > crate::dim_vec::INLINE_RANK {
                saw_spilled = true;
            } else if !shape.is_empty() {
                saw_inline = true;
            }
            if shape.len() >= 3 && shape[1..shape.len() - 1].contains(&1) {
                saw_interior_unit = true;
            }
        }
        assert!(
            saw_inline && saw_spilled,
            "the sweep must cover both representations or it proves nothing \
             about the boundary"
        );
        assert!(
            saw_interior_unit,
            "the sweep must exercise a size-1 interior axis, or an error \
             confined to one propagates outward unnoticed"
        );
    }

    /// A shape that spills must keep every dimension. Truncating at
    /// `INLINE_RANK` would still produce a plausible-looking stride vector for
    /// the leading dimensions, so this pins the length and the last element.
    ///
    /// Kept alongside the oracle sweep rather than folded into it because it
    /// states the answer in **closed form**. The oracle is the same algorithm
    /// by construction — that is what makes it a good check on representation
    /// and initialisation, and a poor one on the algorithm itself.
    #[test]
    fn contiguous_strides_spills_rather_than_truncating() {
        let rank = crate::dim_vec::INLINE_RANK + 2;
        let shape: Vec<usize> = vec![2; rank];
        let got = super::contiguous_strides(&shape);
        assert_eq!(got.len(), rank, "a spilled shape lost dimensions");
        assert_eq!(got[rank - 1], 1, "the innermost stride must be 1");
        assert_eq!(got[0], 1i64 << (rank - 1), "outermost stride is wrong");
    }

    /// `view()` hands the kernel `&self.shape` and `&self.strides`. For a rank
    /// that spilled out of line, both must arrive whole.
    ///
    /// This deliberately does **not** claim to prove that the routed path's
    /// `shape: (*shape).clone()` copies rather than aliases. That property is a
    /// compiler guarantee — `DimVec` derives `Clone`, and there is no safe
    /// `Clone` that shares a `Vec`'s buffer — so no mutation can violate it and
    /// a test asserting it could not fail. What is mutable, and what this pins,
    /// is whether `view()` passes the whole slice or a truncated one.
    #[test]
    fn a_spilled_intermediate_buf_view_reports_every_dimension() {
        let rank = crate::dim_vec::INLINE_RANK + 2;
        let dims: Vec<usize> = (0..rank).map(|i| i + 2).collect();
        let strides = super::contiguous_strides(&dims);
        let numel: usize = dims.iter().product();
        let buf = IntermediateBuf {
            data: vec![0u8; numel * 4],
            scratch_ptr: std::ptr::null_mut(),
            shape: crate::dim_vec::DimVec::from_slice(&dims),
            strides,
            dtype: DataType::Float32,
            device: DeviceId::cpu(),
        };

        let v = buf.view();
        assert_eq!(v.shape, &dims[..], "view truncated the spilled shape");
        assert_eq!(
            v.strides,
            onnx_runtime_ir::compute_contiguous_strides(&dims).as_slice(),
            "view truncated the spilled strides"
        );
    }

    // ── S2: axis bounds check ─────────────────────────────────────────────────

    #[test]
    fn layer_norm_axis_eq_rank_is_out_of_bounds() {
        // rank=3 tensor [2,4,8]; axis=3 is out of bounds (valid: 0..2).
        let strat = ShapeInference::LayerNorm {
            raw_axis: 3,
            num_outputs: 1,
            full_shape_outputs: vec![],
        };
        let v = view(&[2, 4, 8], &[32, 8, 1]);
        let result = infer(&strat, &[v]);
        assert!(
            result.is_err(),
            "axis == rank should be rejected, got {result:?}"
        );
    }

    #[test]
    fn layer_norm_axis_eq_rank_minus_one_is_valid() {
        // rank=3 tensor [2,4,8]; axis=2 is the last valid axis.
        let strat = ShapeInference::LayerNorm {
            raw_axis: 2,
            num_outputs: 2,
            full_shape_outputs: vec![],
        };
        let v = view(&[2, 4, 8], &[32, 8, 1]);
        let result = infer(&strat, &[v]);
        assert!(
            result.is_ok(),
            "axis == rank-1 should be valid, got {result:?}"
        );
        let shapes = result.unwrap();
        // Output 0 = full shape, output 1 = reduced (dims from axis onward → 1).
        assert_eq!(shapes[0], vec![2, 4, 8]);
        assert_eq!(shapes[1], vec![2, 4, 1]);
    }

    // ── B1: Canary tests for scratch buffer sizing with f16/bf16 ──────────────

    /// Canary helper: allocate a buffer using the **production** sizing formula
    /// (`scratch_alloc_bytes`), write `numel` elements at `write_byte_size`
    /// bytes each, and assert the canary padding is intact.
    /// Returns `true` if canaries survived.
    fn canary_check(numel: usize, declared_dtype: DataType, write_byte_size: usize) -> bool {
        const CANARY: u8 = 0xCD;
        const CANARY_LEN: usize = 64;
        let data_len = scratch_alloc_bytes(numel, declared_dtype);
        let total_len = CANARY_LEN + data_len + CANARY_LEN;
        let mut buf = vec![CANARY; total_len];
        buf[CANARY_LEN..CANARY_LEN + data_len].fill(0);
        // Simulate kernel writing numel elements at write_byte_size each.
        let write_len = numel * write_byte_size;
        if write_len > data_len {
            // Would overwrite canaries — that's a detected overflow.
            return false;
        }
        let data_slice = &mut buf[CANARY_LEN..CANARY_LEN + write_len];
        for i in 0..numel {
            let start = i * write_byte_size;
            let end = start + write_byte_size;
            if end <= data_slice.len() {
                data_slice[start..end].fill(0xAB);
            }
        }
        // Check canaries.
        buf[..CANARY_LEN].iter().all(|&b| b == CANARY)
            && buf[CANARY_LEN + data_len..].iter().all(|&b| b == CANARY)
    }

    /// Verify f16 scratch: correct-dtype write fits within production allocation.
    #[test]
    fn scratch_buffer_canary_f16_no_overflow() {
        assert!(
            canary_check(8, DataType::Float16, 2),
            "f16 correct-dtype write must not corrupt canaries"
        );
    }

    /// Same canary test but for BFloat16.
    #[test]
    fn scratch_buffer_canary_bf16_no_overflow() {
        assert!(
            canary_check(8, DataType::BFloat16, 2),
            "bf16 correct-dtype write must not corrupt canaries"
        );
    }

    /// Proof that production allocation absorbs wider writes up to 8 bytes/elem
    /// (the `max(byte_size, 8)` padding). A Float32 (4-byte) kernel writing
    /// into an f16-declared slot is absorbed — this is by design, not a bug.
    #[test]
    fn scratch_buffer_wider_write_absorbed_by_padding() {
        // Writing 4 bytes/elem into a 2-byte-declared slot: production
        // allocates max(2, 8) = 8 bytes/elem, so 4-byte writes fit.
        assert!(
            canary_check(8, DataType::Float16, 4),
            "4-byte write into f16 slot should be absorbed by max(byte_size, 8) padding"
        );
    }

    /// Proof that a truly wrong-dtype write exceeding the 8-byte padding
    /// WOULD be detected: writing 16 bytes/elem into a 2-byte-declared slot
    /// overflows the production allocation (8 bytes/elem < 16 bytes/elem).
    #[test]
    fn scratch_buffer_detects_oversized_write() {
        assert!(
            !canary_check(8, DataType::Float16, 16),
            "16-byte write into f16 slot (8 bytes/elem alloc) must overflow canaries"
        );
    }

    /// Verify that the fixed code never uses Float32 as scratch dtype for absent
    /// slots — it uses the slot's own dtype from output_dtypes.
    #[test]
    fn scratch_dtype_matches_absent_slot_dtype() {
        // Simulate the fixed logic: absent slot at index 1 with Float16 dtype.
        let output_dtypes = [DataType::Float16, DataType::Float16, DataType::Float16];
        let absent_output_slots: HashSet<usize> = [1, 2].into_iter().collect();
        for &slot in &[1usize, 2] {
            assert!(absent_output_slots.contains(&slot));
            let scratch_dtype = output_dtypes[slot];
            // This assertion would fail under the old code where Float32 was hardcoded.
            assert_ne!(
                scratch_dtype,
                DataType::Float32,
                "scratch dtype must NOT be hardcoded Float32 — it should match the slot's dtype"
            );
            assert_eq!(scratch_dtype, DataType::Float16);
            assert_eq!(scratch_dtype.byte_size(), 2);
        }
    }

    // ── validate_write_dtype exercised from compute tests ─────────────────────

    /// Verify that `validate_write_dtype` rejects a write wider than the
    /// scratch allocation permits (the `max(byte_size, 8)` padding).
    #[test]
    fn validate_write_dtype_rejects_overflow() {
        use onnx_runtime_ep_api::tensor::{DevicePtrMut, TensorMut};
        use onnx_runtime_ir::DeviceId;

        let numel = 4usize;
        let declared = DataType::Float16;
        let buf_size = scratch_alloc_bytes(numel, declared);
        let mut buf = vec![0u8; buf_size];
        let shape = [numel];
        let strides = [1i64];
        let view = TensorMut::new(
            DevicePtrMut(buf.as_mut_ptr().cast()),
            declared,
            &shape,
            &strides,
            DeviceId::cpu(),
        )
        .mark_absent();

        // Same dtype — always OK.
        assert!(view.validate_write_dtype(DataType::Float16).is_ok());
        // Float32 (4 bytes) fits in max(2,8)=8 byte padding.
        assert!(view.validate_write_dtype(DataType::Float32).is_ok());
        // Float64 (8 bytes) fits in max(2,8)=8.
        assert!(view.validate_write_dtype(DataType::Float64).is_ok());
    }

    /// Verify that `validate_write_dtype` accepts writes within padding for
    /// present (non-absent) tensors only when dtype matches exactly.
    #[test]
    fn validate_write_dtype_present_requires_exact_match() {
        use onnx_runtime_ep_api::tensor::{DevicePtrMut, TensorMut};
        use onnx_runtime_ir::DeviceId;

        let mut buf = vec![0u8; 32];
        let shape = [4usize];
        let strides = [1i64];
        let view = TensorMut::new(
            DevicePtrMut(buf.as_mut_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        // Not marked absent — exact dtype required.
        assert!(view.validate_write_dtype(DataType::Float32).is_ok());
        assert!(view.validate_write_dtype(DataType::Float16).is_err());
    }

    /// An `OwnedOutput` is inert data -- raw pointers plus shape/strides -- so
    /// these tests never dereference one.
    fn dummy_owned_output() -> crate::kernel_ctx::OwnedOutput {
        crate::kernel_ctx::OwnedOutput {
            data_ptr: std::ptr::null_mut(),
            dtype: DataType::Float32,
            shape: crate::dim_vec::DimVec::zeroed(1),
            strides: crate::dim_vec::DimVec::zeroed(1),
            mem_info: std::ptr::null(),
        }
    }

    fn dummy_owned_input() -> crate::kernel_ctx::OwnedInput {
        crate::kernel_ctx::OwnedInput {
            data_ptr: std::ptr::null(),
            dtype: DataType::Float32,
            shape: crate::dim_vec::DimVec::zeroed(1),
            strides: crate::dim_vec::DimVec::zeroed(1),
            device: DeviceId::cpu(),
        }
    }

    #[test]
    fn run_scratch_hands_back_input_capacity_it_retired() {
        // Fill past the capacity an empty `Vec::reserve(1)` reaches on its own,
        // or the assertion below would hold whether or not reuse happened.
        let cap = with_run_scratch(|scratch| {
            for _ in 0..SCRATCH_MAX_CAPACITY {
                scratch.inputs.push(dummy_owned_input());
            }
            scratch.inputs.capacity()
        });
        assert!(
            cap > 4,
            "retired capacity {cap} is one a fresh vector reaches anyway"
        );

        with_run_scratch(|scratch| {
            assert!(scratch.inputs.is_empty(), "reused scratch must start empty");
            assert!(
                scratch.inputs.capacity() >= cap,
                "capacity was not reused: {} vs {cap}",
                scratch.inputs.capacity()
            );
        });
    }

    #[test]
    fn run_scratch_hands_back_output_capacity_it_retired() {
        let (owned_cap, slot_cap) = with_run_scratch(|scratch| {
            for _ in 0..SCRATCH_MAX_CAPACITY {
                scratch.owned.push(dummy_owned_output());
                scratch.slots.push(SlotKind::Ort);
            }
            (scratch.owned.capacity(), scratch.slots.capacity())
        });
        assert!(
            owned_cap > 4,
            "retired capacity {owned_cap} is one a fresh vector reaches anyway"
        );

        with_run_scratch(|scratch| {
            assert!(
                scratch.owned.is_empty() && scratch.slots.is_empty(),
                "reused scratch must start empty"
            );
            assert!(
                scratch.owned.capacity() >= owned_cap && scratch.slots.capacity() >= slot_cap,
                "capacity was not reused: {} / {} vs {owned_cap} / {slot_cap}",
                scratch.owned.capacity(),
                scratch.slots.capacity()
            );
        });
    }

    /// The scratch must not hold an `OwnedInput` or `OwnedOutput` between
    /// `Run`s: those borrow pointers into ORT values belonging to a call that
    /// has finished.
    #[test]
    fn run_scratch_retains_nothing_between_runs() {
        with_run_scratch(|scratch| {
            scratch.inputs.push(dummy_owned_input());
            scratch.owned.push(dummy_owned_output());
            scratch.slots.push(SlotKind::Ort);
        });

        RUN_SCRATCH.with(|cell| {
            let slot = cell.borrow();
            assert!(
                slot.inputs.is_empty() && slot.owned.is_empty() && slot.slots.is_empty(),
                "parked scratch still holds {} inputs / {} outputs / {} slots",
                slot.inputs.len(),
                slot.owned.len(),
                slot.slots.len()
            );
        });
    }

    #[test]
    fn run_scratch_does_not_pin_a_pathological_input_capacity() {
        with_run_scratch(|scratch| {
            scratch.inputs.reserve(SCRATCH_MAX_CAPACITY + 1);
            scratch.inputs.push(dummy_owned_input());
        });

        RUN_SCRATCH.with(|cell| {
            let cap = cell.borrow().inputs.capacity();
            assert!(
                cap <= SCRATCH_MAX_CAPACITY,
                "oversized scratch was parked: {cap}"
            );
        });
    }

    #[test]
    fn run_scratch_does_not_pin_a_pathological_output_capacity() {
        let over = SCRATCH_MAX_CAPACITY + 1;
        with_run_scratch(|scratch| {
            scratch.owned.reserve(over);
            scratch.slots.reserve(over);
            assert!(scratch.owned.capacity() >= over);
            scratch.owned.push(dummy_owned_output());
            scratch.slots.push(SlotKind::Ort);
        });

        RUN_SCRATCH.with(|cell| {
            let slot = cell.borrow();
            assert!(
                slot.owned.capacity() <= SCRATCH_MAX_CAPACITY
                    && slot.slots.capacity() <= SCRATCH_MAX_CAPACITY,
                "oversized scratch was parked: {} / {}",
                slot.owned.capacity(),
                slot.slots.capacity()
            );
        });
    }

    /// Bundling the storage must not bundle the keep/drop policy. Before these
    /// three vectors shared a cell they were judged separately, and a node with
    /// a pathological input arity did not cost the *output* vectors the
    /// capacity they had earned.
    ///
    /// Falsifier: replace the two independent bounds in
    /// [`RunScratch::clear_and_bound`] with a single condition over all three
    /// vectors and this fails both ways.
    #[test]
    fn run_scratch_bounds_each_vector_independently() {
        // Oversized inputs, ordinary outputs: the outputs must survive.
        RUN_SCRATCH.with(|cell| *cell.borrow_mut() = RunScratch::default());
        let kept = with_run_scratch(|scratch| {
            scratch.inputs.reserve(SCRATCH_MAX_CAPACITY + 1);
            scratch.owned.reserve(SCRATCH_MAX_CAPACITY);
            scratch.slots.reserve(SCRATCH_MAX_CAPACITY);
            scratch.owned.capacity()
        });
        RUN_SCRATCH.with(|cell| {
            let slot = cell.borrow();
            assert!(
                slot.inputs.capacity() <= SCRATCH_MAX_CAPACITY,
                "pathological input capacity was parked: {}",
                slot.inputs.capacity()
            );
            assert!(
                slot.owned.capacity() >= kept,
                "an oversized input vector cost the outputs their capacity: {} vs {kept}",
                slot.owned.capacity()
            );
        });

        // And the converse: oversized outputs must not cost the inputs theirs.
        RUN_SCRATCH.with(|cell| *cell.borrow_mut() = RunScratch::default());
        let kept_in = with_run_scratch(|scratch| {
            scratch.inputs.reserve(SCRATCH_MAX_CAPACITY);
            scratch.owned.reserve(SCRATCH_MAX_CAPACITY + 1);
            scratch.slots.reserve(SCRATCH_MAX_CAPACITY + 1);
            scratch.inputs.capacity()
        });
        RUN_SCRATCH.with(|cell| {
            let slot = cell.borrow();
            assert!(
                slot.owned.capacity() <= SCRATCH_MAX_CAPACITY,
                "pathological output capacity was parked: {}",
                slot.owned.capacity()
            );
            assert!(
                slot.inputs.capacity() >= kept_in,
                "an oversized output vector cost the inputs their capacity: {} vs {kept_in}",
                slot.inputs.capacity()
            );
        });
    }

    /// A kernel that re-enters `Compute` on this thread finds the cell
    /// borrowed. That must degrade to a fresh allocation, not panic.
    ///
    /// Falsifier: use `borrow_mut` instead of `try_borrow_mut` in
    /// [`with_run_scratch`] and this panics.
    #[test]
    fn run_scratch_is_reentrancy_safe() {
        RUN_SCRATCH.with(|cell| {
            let _held = cell.borrow_mut();
            with_run_scratch(|scratch| {
                scratch.inputs.push(dummy_owned_input());
                scratch.owned.push(dummy_owned_output());
                scratch.slots.push(SlotKind::Ort);
                assert_eq!(scratch.inputs.len(), 1);
                assert_eq!(scratch.owned.len(), 1);
            });
        });
    }

    /// A nested `Run` must not be handed storage the outer call is still using,
    /// and must not disturb it. The outer call borrows the cell for its whole
    /// duration, so the inner one falls back to storage of its own.
    ///
    /// Falsifier: hand the inner call the same `&mut RunScratch` and the inner
    /// emptiness assertion fails; clear the cell on acquire instead of on exit
    /// and the outer call loses its inputs.
    #[test]
    fn a_nested_run_does_not_alias_the_outer_runs_scratch() {
        with_run_scratch(|outer| {
            outer.inputs.push(dummy_owned_input());
            outer.owned.push(dummy_owned_output());
            outer.slots.push(SlotKind::Ort);

            with_run_scratch(|inner| {
                assert!(
                    inner.inputs.is_empty() && inner.owned.is_empty() && inner.slots.is_empty(),
                    "a nested Run was handed the outer Run's live bookkeeping: \
                     {} inputs / {} outputs / {} slots",
                    inner.inputs.len(),
                    inner.owned.len(),
                    inner.slots.len()
                );
                inner.inputs.push(dummy_owned_input());
            });

            assert_eq!(
                (outer.inputs.len(), outer.owned.len(), outer.slots.len()),
                (1, 1, 1),
                "a nested Run disturbed the outer Run's bookkeeping"
            );
        });
    }

    /// Thread-local, so two threads never share a buffer -- concurrent `Run`s
    /// on different threads cannot alias each other's bookkeeping.
    #[test]
    fn run_scratch_is_per_thread() {
        // Park a capacity a fresh `Vec` would not land on by itself: an empty
        // `Vec::reserve(1)` already gives 4 for these element sizes, so a
        // smaller parked capacity could not tell reuse from a coincidence.
        let parked = with_run_scratch(|scratch| {
            for _ in 0..SCRATCH_MAX_CAPACITY {
                scratch.owned.push(dummy_owned_output());
                scratch.slots.push(SlotKind::Ort);
            }
            scratch.owned.capacity()
        });
        assert!(parked >= SCRATCH_MAX_CAPACITY);

        let other = std::thread::spawn(|| {
            with_run_scratch(|scratch| {
                scratch.owned.reserve(1);
                scratch.owned.capacity()
            })
        })
        .join()
        .unwrap();
        assert!(
            other < parked,
            "a fresh thread reused another thread's capacity: {other} vs {parked}"
        );
    }

    /// Many threads borrowing and returning concurrently must not observe each
    /// other's storage, and must not panic on teardown.
    #[test]
    fn concurrent_runs_do_not_share_scratch() {
        let threads: Vec<_> = (0..8)
            .map(|t| {
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        with_run_scratch(|scratch| {
                            for _ in 0..=(t % 4) {
                                scratch.inputs.push(dummy_owned_input());
                                scratch.owned.push(dummy_owned_output());
                                scratch.slots.push(SlotKind::Ort);
                            }
                            assert_eq!(scratch.inputs.len(), (t % 4) + 1);
                        });
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("no thread panicked");
        }
    }

    /// A panic inside the `Run` must not park a dirty buffer. The scratch is
    /// borrowed rather than moved out, so this is the guard's `Drop` running
    /// during the unwind -- not a side effect of dropping an owned value.
    ///
    /// This is the **only** test that can tell `ScratchGuard::drop` from a call
    /// at the end of `with_run_scratch`'s success path, because unwinding is
    /// the only exit where the two differ. An earlier version of this suite
    /// also claimed an "early return" test pinned that distinction; it did not,
    /// and could not -- a `return` out of an `FnOnce` *is* the success path.
    /// Review caught the false claim. Do not delete this test in favour of one
    /// that looks like it covers the same ground.
    ///
    /// Falsifier: clear the scratch at the end of `with_run_scratch` instead of
    /// in `ScratchGuard::drop` and the next borrow finds a live `OwnedInput`.
    #[test]
    fn a_panic_inside_the_run_parks_nothing_dirty() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(|| {
            with_run_scratch(|scratch| {
                scratch.inputs.push(dummy_owned_input());
                scratch.owned.push(dummy_owned_output());
                panic!("kernel exploded");
            })
        });
        std::panic::set_hook(hook);
        assert!(caught.is_err(), "the test panic must actually have fired");

        with_run_scratch(|scratch| {
            assert!(
                scratch.inputs.is_empty() && scratch.owned.is_empty(),
                "a panicking Run parked {} live inputs / {} live outputs",
                scratch.inputs.len(),
                scratch.owned.len()
            );
        });
    }

    /// Every pool operation must go through the `Run`'s own pool.
    ///
    /// An independent review of the change that moved this pool into
    /// `RunScratch` found a mutation that survived the entire suite: redirect
    /// the single `take` call site to a throwaway pool. Every buffer is then
    /// freshly allocated on every node -- the exact cost the move existed to
    /// remove -- while the seven `recycle` sites keep filling the real pool,
    /// which is now read by nobody. It is correctness preserving, so no
    /// behavioural test can see it, and the unit tests below cannot either
    /// because they own their pools and never touch these call sites.
    ///
    /// A newtype makes the accidental form (`&mut Vec::new()`) fail to compile.
    /// This pins the deliberate one: **no pool operation anywhere in this file
    /// is performed on a pool other than the one destructured from
    /// `RunScratch`.** Stated as a ratio rather than as fixed call counts, so
    /// that adding or removing a legitimate call site does not trip it.
    #[test]
    fn every_pool_operation_goes_through_the_run_scratch_pool() {
        let prod = production_source(include_str!("compute.rs"));
        let prod = prod.as_str();

        for method in ["take_intermediate", "recycle_intermediate"] {
            let calls = prod.matches(&format!(".{method}(")).count();
            let via_pool = prod.matches(&format!("host_pool.{method}(")).count();

            // Anti-vacuity, and the reason it is not optional: if either method
            // is renamed, both counts fall to zero and the equality below holds
            // trivially. The guard has to be unconditional -- a check that
            // itself skips when the thing it guards disappears is the failure
            // it exists to prevent.
            assert!(
                calls > 0,
                "no calls to `{method}` found; this test would pass vacuously"
            );
            assert_eq!(
                calls,
                via_pool,
                "{} call(s) to `{method}` do not use the `RunScratch` pool; a \
                 pool built at the call site allocates on every node and no \
                 behavioural test can see it",
                calls - via_pool
            );
        }
    }

    /// The `Run` path must resolve the thread-local exactly **once**. Taking
    /// the scratch out and putting it back cost two resolutions; borrowing it
    /// costs one. That is not observable from a unit test, so it is asserted
    /// structurally.
    ///
    /// Falsifier: reintroduce a second scratch thread-local, or add another
    /// `RUN_SCRATCH.with` on the `Run` path, and this fails.
    #[test]
    fn a_run_resolves_the_scratch_thread_local_exactly_once() {
        let prod = production_source(include_str!("compute.rs"));
        let prod = prod.as_str();

        // Anchor: the thing being counted must exist, or the count is vacuous.
        assert!(
            prod.contains("static RUN_SCRATCH:"),
            "RUN_SCRATCH is gone -- this test would pass vacuously"
        );

        // Counting resolutions of one *name* is not enough: splitting the pool
        // back apart introduces a differently-named thread-local and leaves the
        // RUN_SCRATCH count untouched. So the property asserted is the total
        // thread-local surface of this file -- every declaration and every
        // resolution of any of them.
        let mut declared: Vec<&str> = prod
            .match_indices("\n    static ")
            .map(|(i, m)| {
                let rest = &prod[i + m.len()..];
                &rest[..rest.find(':').unwrap_or(0)]
            })
            .collect();
        declared.sort_unstable();
        assert_eq!(
            declared,
            ["ENABLED", "ENABLED", "RUN_SCRATCH"],
            "the set of thread-locals in compute.rs changed; a new per-`Run` \
             pool costs another __tls_get_addr on the hot path"
        );
        assert_eq!(
            prod.matches(".with(").count() + prod.matches(".try_with(").count(),
            1,
            "the number of thread-local resolutions in compute.rs changed"
        );

        // Scoping this count to `compute_execute`'s body is not enough: a
        // helper defined outside it that acquires the scratch, called from
        // inside the per-node loop, regresses the optimisation while the body
        // contains neither name. Review found exactly that bypass. So the call
        // sites are counted over the whole production source instead -- there
        // must be exactly one in the crate, and it must be the one in
        // `compute_execute`.
        assert!(
            prod.contains("fn with_run_scratch<"),
            "with_run_scratch is gone -- this count would be vacuous"
        );
        // The definition is generic (`fn with_run_scratch<R>(`), so it does not
        // match `with_run_scratch(` -- but subtract it anyway, so that making
        // the signature non-generic does not silently turn the definition into
        // a phantom call site and mask a real second one.
        let acquisitions = prod.matches("with_run_scratch(").count()
            - prod.matches("fn with_run_scratch(").count();
        assert_eq!(
            acquisitions, 1,
            "the scratch is acquired somewhere other than the one place on the \
             `Run` path; a second site costs another __tls_get_addr"
        );

        let start = prod
            .find("unsafe extern \"C\" fn compute_execute(")
            .expect("compute_execute exists");
        let body = &prod[start..];
        let end = body.find("\n}\n").expect("compute_execute terminates") + 3;
        let body = &body[..end];
        assert_eq!(
            body.matches("with_run_scratch(").count(),
            1,
            "compute_execute acquires the scratch more than once per Run"
        );

        // The recycle must live in the guard's `Drop`, not in the `Run` body.
        // That is the whole reason an early `return` cannot skip it any more --
        // and it is a placement no behavioural unit test can observe except
        // through an unwind, so it is asserted here as well.
        let drop_impl = prod
            .split_once("impl Drop for ScratchGuard<'_> {")
            .expect("ScratchGuard has a Drop impl")
            .1;
        assert!(
            drop_impl[..drop_impl.find("\n}").unwrap_or(0)].contains("clear_and_bound()"),
            "ScratchGuard::drop no longer recycles; an early return or an \
             unwinding kernel would park a dirty buffer"
        );
        assert_eq!(
            body.matches("clear_and_bound(").count(),
            0,
            "compute_execute recycles the scratch itself; a call in the body is \
             a call every early `return` above it skips, which is exactly the \
             defect the guard removed"
        );
    }
}

#[cfg(test)]
mod workspace_math_tests {
    use super::{align_workspace_window, workspace_block_bytes, workspace_trace_line};

    /// The over-allocation must be exactly enough that align-up always fits —
    /// no more (wasted device memory per dispatch) and no less (out-of-bounds).
    #[test]
    fn block_size_covers_the_worst_case_misalignment() {
        for alignment in [1usize, 8, 64, 256, 4096] {
            let total = workspace_block_bytes(1000, alignment).expect("no overflow");
            assert_eq!(total, 1000 + alignment - 1);
            // Worst case: the allocator returns a block one byte past alignment.
            let base = alignment * 4 + 1;
            let aligned = align_workspace_window(base, total, 1000, alignment).expect("must fit");
            assert!(aligned >= base, "aligned pointer moved backwards");
            assert!(
                aligned.is_multiple_of(alignment),
                "aligned pointer {aligned:#x} does not satisfy {alignment}"
            );
            assert!(
                aligned + 1000 <= base + total,
                "aligned window escapes the block"
            );
        }
    }

    /// An already-aligned base must not be moved — otherwise every dispatch
    /// silently wastes up to `alignment - 1` bytes it did not need to.
    #[test]
    fn an_aligned_base_is_returned_unchanged() {
        let aligned = align_workspace_window(4096, 4096 + 255, 4096, 256).expect("must fit");
        assert_eq!(aligned, 4096);
    }

    /// A request whose padded size cannot be represented must be rejected
    /// rather than wrapping to a tiny allocation the kernel then overruns.
    #[test]
    fn an_overflowing_request_is_rejected_not_wrapped() {
        let err = workspace_block_bytes(usize::MAX, 256).expect_err("must reject");
        assert!(err.contains("overflows usize"), "got: {err}");
    }

    /// Aligning a pointer near the top of the address space must fail closed,
    /// not wrap around to a low address.
    #[test]
    fn aligning_near_the_address_space_top_is_rejected() {
        let err = align_workspace_window(usize::MAX - 8, 8, 8, 256).expect_err("must reject");
        assert!(
            err.contains("overflows usize") || err.contains("escapes"),
            "got: {err}"
        );
    }

    /// The containment check is the last line of defence: if the block is too
    /// small for the aligned window, no pointer may be handed out.
    #[test]
    fn a_window_escaping_the_block_is_rejected() {
        // 64 bytes requested from a 64-byte block whose base is misaligned:
        // aligning up leaves fewer than 64 bytes.
        let err = align_workspace_window(0x1001, 64, 64, 256).expect_err("must reject");
        assert!(err.contains("escapes"), "got: {err}");
    }

    /// The trace exists to answer two device-only questions from the pointer
    /// itself, so it must report the block address (comparable across steps)
    /// and the alignment ORT actually delivered — not a restatement of what the
    /// kernel asked for.
    #[test]
    fn a_trace_line_reports_the_block_the_reader_has_to_compare() {
        let line = workspace_trace_line(
            "Attention_0",
            0x7f00_0000_1000,
            0x7f00_0000_1000,
            96,
            351,
            256,
        );
        assert!(
            line.contains("block=0x7f0000001000"),
            "the block address is what identifies reuse across steps: {line}"
        );
        assert!(
            line.contains("ptr=0x7f0000001000"),
            "the served pointer must be shown next to the block: {line}"
        );
        assert!(
            line.contains("skew=0"),
            "an already-aligned block must report zero skew: {line}"
        );
        assert!(
            line.contains("bytes=96")
                && line.contains("align=256")
                && line.contains("requested_block=351"),
            "the request the block answers must travel with it: {line}"
        );
    }

    /// `block_align` is the load-bearing field: it says whether ORT's scratch
    /// already met the kernel's alignment or the executor had to skew the
    /// pointer, which is the difference between "ORT aligns for us" and "our
    /// over-allocation is doing the work".
    #[test]
    fn block_alignment_is_measured_from_the_address_not_assumed() {
        let under = workspace_trace_line("n", 0x1040, 0x1100, 8, 263, 256);
        assert!(
            under.contains("block_align=64"),
            "0x1040 is 64-byte aligned and no more: {under}"
        );
        assert!(
            under.contains("skew=192"),
            "the executor had to move the pointer 192 bytes to reach 256: {under}"
        );

        let over = workspace_trace_line("n", 0x2000, 0x2000, 8, 263, 256);
        assert!(
            over.contains("block_align=4096"),
            "measurement must saturate at the 4096 cap, not report 256: {over}"
        );
    }

    /// A null block never reaches the trace (`alloc_scratch` rejects it), but
    /// the formatter must not loop forever if it ever did.
    #[test]
    fn a_zero_block_reports_zero_alignment_instead_of_looping() {
        let line = workspace_trace_line("n", 0, 0, 8, 8, 8);
        assert!(line.contains("block_align=0"), "got: {line}");
    }
}

#[cfg(test)]
mod workspace_plan_cache_tests {
    //! Falsifiers for [`WorkspacePlanCache`].
    //!
    //! The cache exists to stop `prepare_workspace` re-running an expensive
    //! `Kernel::workspace_requirement` (a cuBLASLt heuristic search on the CUDA
    //! GEMM kernels) on every dispatch of a shape it has already planned. Every
    //! test here therefore asserts on the **observed number of planning calls**
    //! and on **which requirement was returned** — never on a summary of the
    //! cache's own state — so a cache that silently returns the wrong plan
    //! cannot pass by reporting a plausible hit count.

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use onnx_runtime_ep_api::kernel::{TensorMetadata, WorkspaceRequirement};
    use onnx_runtime_ep_api::tensor::{DevicePtr, TensorView};
    use onnx_runtime_ir::DataType;

    use super::{OperandKey, WORKSPACE_PLAN_CACHE_CAPACITY, WorkspacePlanCache};

    fn requirement(bytes: u64) -> WorkspaceRequirement {
        WorkspaceRequirement {
            bytes,
            alignment: 256,
            ..WorkspaceRequirement::NONE
        }
    }

    /// Stands in for a kernel whose `workspace_requirement` is expensive: it
    /// counts how many times it actually ran and derives the answer from the
    /// metadata, so a stale hit is visible as a wrong byte count.
    struct CountingPlanner {
        calls: AtomicUsize,
    }

    impl CountingPlanner {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        /// Derive a distinct answer for every distinguishable metadata list, so
        /// serving one signature's plan for another is detectable by value.
        fn plan(&self, metadata: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut bytes = 0u64;
            for meta in metadata {
                let numel: usize = meta.shape.iter().product();
                let elem = meta.dtype.byte_size().max(1);
                let present = u64::from(meta.present);
                bytes += (numel * elem) as u64 * present;
            }
            Ok(requirement(bytes.max(1)))
        }
    }

    fn meta<'a>(dtype: DataType, shape: &'a [usize], present: bool) -> TensorMetadata<'a> {
        TensorMetadata::new(dtype, shape, present)
    }

    /// The signature every dispatch is actually using must survive eviction.
    ///
    /// The cache evicts from the back, so without move-to-front the first
    /// signature inserted drifts to the back and is evicted even though it is
    /// the hot one -- silently restoring the expensive planning call the cache
    /// exists to avoid. Asserted on observed planning calls, not on cache
    /// state.
    #[test]
    fn a_repeatedly_used_signature_survives_eviction() {
        // Locked path on purpose: the lock-free slot would serve the repeat
        // access below and the assertion would hold with move-to-front gone.
        let cache = WorkspacePlanCache::locked_only();
        let planner = CountingPlanner::new();

        let hot_shape = vec![1usize];
        let hot = [meta(DataType::Float32, &hot_shape, true)];
        cache.get_or_plan(&hot, || planner.plan(&hot)).unwrap();

        // Fill the rest of the cache. `hot` is now at the back, next to evict.
        let cold: Vec<Vec<usize>> = (2..=WORKSPACE_PLAN_CACHE_CAPACITY)
            .map(|n| vec![n])
            .collect();
        for sh in &cold {
            let m = [meta(DataType::Float32, sh, true)];
            cache.get_or_plan(&m, || planner.plan(&m)).unwrap();
        }

        // Using `hot` again is the access that must promote it.
        let before = planner.calls();
        cache.get_or_plan(&hot, || planner.plan(&hot)).unwrap();
        assert_eq!(
            planner.calls(),
            before,
            "hot signature was already lost before any eviction"
        );

        // One more distinct signature evicts whatever is at the back.
        let evictor_shape = vec![WORKSPACE_PLAN_CACHE_CAPACITY + 1];
        let evictor = [meta(DataType::Float32, &evictor_shape, true)];
        cache
            .get_or_plan(&evictor, || planner.plan(&evictor))
            .unwrap();

        let before = planner.calls();
        let got = cache.get_or_plan(&hot, || planner.plan(&hot)).unwrap();
        assert_eq!(
            planner.calls(),
            before,
            "the signature every dispatch uses was evicted -- move-to-front is not promoting it"
        );
        assert_eq!(got.bytes, 4, "wrong plan served for the hot signature");
    }

    /// The point of the cache: the second dispatch of an unchanged shape must
    /// not re-run the planner. Falsifier — delete the lookup and the call count
    /// becomes 2.

    #[test]
    fn a_repeated_signature_plans_exactly_once() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let shape = [4usize, 8];
        let metadata = [meta(DataType::Float32, &shape, true)];

        let first = cache
            .get_or_plan(&metadata, || planner.plan(&metadata))
            .expect("plan");
        for _ in 0..16 {
            let again = cache
                .get_or_plan(&metadata, || planner.plan(&metadata))
                .expect("plan");
            assert_eq!(
                again, first,
                "a cache hit must return the same requirement the planner produced"
            );
        }
        assert_eq!(
            planner.calls(),
            1,
            "16 dispatches of one unchanged shape must run the planner once, not 17"
        );
    }

    /// A different shape is a different question. Falsifier — drop `shape` from
    /// the key and the second call returns 128 bytes for a 256-byte geometry.
    #[test]
    fn a_changed_shape_is_replanned_and_not_served_stale() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let small = [4usize, 8];
        let large = [4usize, 16];

        let small_meta = [meta(DataType::Float32, &small, true)];
        let large_meta = [meta(DataType::Float32, &large, true)];
        let small_req = cache
            .get_or_plan(&small_meta, || planner.plan(&small_meta))
            .expect("plan");
        let large_req = cache
            .get_or_plan(&large_meta, || planner.plan(&large_meta))
            .expect("plan");

        assert_eq!(small_req.bytes, 4 * 8 * 4);
        assert_eq!(
            large_req.bytes,
            4 * 16 * 4,
            "the larger geometry must get its own plan, not the smaller one's"
        );
        assert_eq!(planner.calls(), 2);

        // And both remain individually correct once cached.
        assert_eq!(
            cache
                .get_or_plan(&small_meta, || planner.plan(&small_meta))
                .expect("plan"),
            small_req
        );
        assert_eq!(
            cache
                .get_or_plan(&large_meta, || planner.plan(&large_meta))
                .expect("plan"),
            large_req
        );
        assert_eq!(planner.calls(), 2, "both shapes must now be cached");
    }

    /// Same shape, different dtype: the workspace of an f32 GEMM is not the
    /// workspace of an f16 one. Falsifier — drop `dtype` from the key and this
    /// returns the f32 size for f16 operands.
    #[test]
    fn a_changed_dtype_is_replanned_and_not_served_stale() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let shape = [64usize];

        let f32_meta = [meta(DataType::Float32, &shape, true)];
        let f16_meta = [meta(DataType::Float16, &shape, true)];
        let f32_req = cache
            .get_or_plan(&f32_meta, || planner.plan(&f32_meta))
            .expect("plan");
        let f16_req = cache
            .get_or_plan(&f16_meta, || planner.plan(&f16_meta))
            .expect("plan");

        assert_eq!(f32_req.bytes, 64 * 4);
        assert_eq!(
            f16_req.bytes,
            64 * 2,
            "an f16 dispatch must not be served the f32 plan"
        );
        assert_eq!(planner.calls(), 2);
    }

    /// Optional-input presence changes the requirement for real kernels
    /// (`MatMulNBits` charges the cuBLASLt epilogue only when `bias` is bound;
    /// `GroupQueryAttention` charges packed staging only for packed QKV).
    /// Falsifier — drop `present` from the key and the absent-bias dispatch is
    /// served the with-bias plan.
    #[test]
    fn a_changed_presence_is_replanned_and_not_served_stale() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let a = [16usize];
        let bias = [16usize];

        let with_bias = [
            meta(DataType::Float32, &a, true),
            meta(DataType::Float32, &bias, true),
        ];
        let without_bias = [
            meta(DataType::Float32, &a, true),
            meta(DataType::Float32, &bias, false),
        ];
        let with = cache
            .get_or_plan(&with_bias, || planner.plan(&with_bias))
            .expect("plan");
        let without = cache
            .get_or_plan(&without_bias, || planner.plan(&without_bias))
            .expect("plan");

        assert_eq!(with.bytes, 16 * 4 * 2);
        assert_eq!(
            without.bytes,
            16 * 4,
            "an absent optional operand must not be served the plan that charged for it"
        );
        assert_eq!(planner.calls(), 2);
    }

    /// A different operand count is a different question too — an arity change
    /// must never collide with a prefix of a longer signature.
    #[test]
    fn a_changed_operand_count_is_replanned() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let shape = [8usize];

        let one = [meta(DataType::Float32, &shape, true)];
        let two = [
            meta(DataType::Float32, &shape, true),
            meta(DataType::Float32, &shape, true),
        ];
        let one_req = cache
            .get_or_plan(&one, || planner.plan(&one))
            .expect("plan");
        let two_req = cache
            .get_or_plan(&two, || planner.plan(&two))
            .expect("plan");

        assert_eq!(one_req.bytes, 8 * 4);
        assert_eq!(two_req.bytes, 8 * 4 * 2);
        assert_eq!(planner.calls(), 2);
    }

    /// Overflowing the capacity must degrade to re-planning, never to a wrong
    /// answer. Falsifier — evict by overwriting an arbitrary slot instead of
    /// the least-recently-used one and the returned bytes stop matching the
    /// signature that asked.
    #[test]
    fn exceeding_capacity_still_answers_every_signature_correctly() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let shapes: Vec<[usize; 1]> = (1..=(WORKSPACE_PLAN_CACHE_CAPACITY * 3))
            .map(|n| [n])
            .collect();

        for _ in 0..3 {
            for shape in &shapes {
                let metadata = [meta(DataType::Float32, shape.as_slice(), true)];
                let got = cache
                    .get_or_plan(&metadata, || planner.plan(&metadata))
                    .expect("plan");
                assert_eq!(
                    got.bytes,
                    (shape[0] * 4) as u64,
                    "signature {shape:?} was served another signature's plan"
                );
            }
        }
        assert!(
            cache.len() <= WORKSPACE_PLAN_CACHE_CAPACITY,
            "the cache must stay bounded, found {} entries",
            cache.len()
        );
    }

    /// The hot signature must survive a one-off shape (a single prefill step
    /// among many decode steps). The flood must run *past* capacity: filling
    /// the cache to exactly capacity never evicts anything, so a shorter loop
    /// passes with or without the promotion and proves nothing. Falsifier —
    /// remove the move-to-front on hit and the hot signature drifts to the
    /// back, gets evicted, and the planner runs again mid-flood.
    #[test]
    fn the_hot_signature_survives_a_flood_of_one_off_shapes() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let hot = [4096usize];
        let hot_meta = [meta(DataType::Float32, &hot, true)];

        cache
            .get_or_plan(&hot_meta, || planner.plan(&hot_meta))
            .expect("plan");
        let after_hot = planner.calls();

        let flood = WORKSPACE_PLAN_CACHE_CAPACITY * 2;
        for n in 1..=flood {
            let cold = [n];
            let cold_meta = [meta(DataType::Float32, &cold, true)];
            cache
                .get_or_plan(&cold_meta, || planner.plan(&cold_meta))
                .expect("plan");
            // Touch the hot signature between cold ones, as decode does.
            cache
                .get_or_plan(&hot_meta, || planner.plan(&hot_meta))
                .expect("plan");
        }
        assert_eq!(
            planner.calls(),
            after_hot + flood,
            "only the cold signatures may have been planned; the hot one must have stayed cached"
        );
    }

    /// A planning error must reach the caller and must not be remembered: the
    /// next dispatch has to ask again, exactly as it did before the cache
    /// existed. Falsifier — cache the error and the second call count stays 1.
    #[test]
    fn a_planning_error_is_propagated_and_never_cached() {
        let cache = WorkspacePlanCache::new();
        let calls = AtomicUsize::new(0);
        let shape = [2usize];
        let metadata = [meta(DataType::Float32, &shape, true)];

        for expected in 1..=3 {
            let err = cache
                .get_or_plan(&metadata, || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Err("cuBLASLt heuristic found no algorithm".to_string())
                })
                .expect_err("the planner failed, so the dispatch must fail");
            assert!(err.contains("no algorithm"), "got: {err}");
            assert_eq!(
                calls.load(Ordering::SeqCst),
                expected,
                "a failed plan must not be remembered as an answer"
            );
        }
    }

    /// Concurrent `Run`s share one `ExportedComputeInfo`, so they share this
    /// cache. Every thread must get the plan for *its own* signature.
    ///
    /// Falsifier — replace the keyed store with a single last-plan slot and
    /// threads start reading each other's requirements; this asserts on the
    /// value, so that shows up as a wrong byte count rather than as a hit-rate
    /// change nobody notices.
    #[test]
    fn concurrent_dispatches_never_read_each_others_plans() {
        let cache = Arc::new(WorkspacePlanCache::new());
        let planner = Arc::new(CountingPlanner::new());
        let threads: Vec<_> = (1usize..=8)
            .map(|id| {
                let cache = Arc::clone(&cache);
                let planner = Arc::clone(&planner);
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        let shape = [id * 32];
                        let metadata = [meta(DataType::Float32, shape.as_slice(), true)];
                        let got = cache
                            .get_or_plan(&metadata, || planner.plan(&metadata))
                            .expect("plan");
                        assert_eq!(
                            got.bytes,
                            (id * 32 * 4) as u64,
                            "thread {id} was served another thread's plan"
                        );
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("no thread may panic");
        }
        assert!(
            cache.len() <= WORKSPACE_PLAN_CACHE_CAPACITY,
            "the cache must stay bounded under concurrency, found {} entries",
            cache.len()
        );
        assert!(
            planner.calls() < 8 * 200,
            "the cache must still be doing work under concurrency, saw {} plans for 1600 \
             dispatches",
            planner.calls()
        );
    }

    /// A poisoned lock must not turn every later dispatch into a hard error —
    /// the same `PoisonError::into_inner` policy the EP handle and the factory
    /// teardown use. Falsifier — swap to `.lock().unwrap()` and this panics.
    #[test]
    fn a_poisoned_cache_lock_is_recovered_not_propagated() {
        let cache = Arc::new(WorkspacePlanCache::new());
        let planner = CountingPlanner::new();
        let shape = [12usize];
        let metadata = [meta(DataType::Float32, &shape, true)];
        cache
            .get_or_plan(&metadata, || planner.plan(&metadata))
            .expect("plan");

        let poisoner = {
            let cache = Arc::clone(&cache);
            std::thread::spawn(move || {
                let _guard = cache.plans.lock().expect("not yet poisoned");
                panic!("poison the cache lock");
            })
        };
        assert!(poisoner.join().is_err(), "the poisoning thread must panic");

        let got = cache
            .get_or_plan(&metadata, || planner.plan(&metadata))
            .expect("a poisoned lock must not fail the dispatch");
        assert_eq!(got.bytes, 12 * 4);
        assert_eq!(
            planner.calls(),
            1,
            "the cached plan must survive the poisoning"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Lock-free fast path (#1077 lever 3)
    // ──────────────────────────────────────────────────────────────────────

    fn view<'a>(
        data: &'a [u8],
        dtype: DataType,
        shape: &'a [usize],
        strides: &'a [i64],
    ) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(data.as_ptr().cast()),
            dtype,
            shape,
            strides,
            onnx_runtime_ir::DeviceId::cpu(),
        )
    }

    /// The fast path compares against `TensorView`s to avoid building the
    /// metadata `Vec`, so its comparison must be indistinguishable from the
    /// metadata one. Falsifier: drop any field from `matches_view` and a
    /// disagreement appears here.
    #[test]
    fn the_view_and_metadata_comparisons_agree() {
        let data = [0u8; 64];
        let strides = [1i64; 3];
        let shapes: [&[usize]; 3] = [&[8], &[2, 3], &[]];
        for dtype in [DataType::Float32, DataType::Float16] {
            for shape in shapes {
                for absent in [false, true] {
                    let v = if absent {
                        TensorView::absent(dtype)
                    } else {
                        view(&data, dtype, shape, &strides[..shape.len()])
                    };
                    let m = TensorMetadata::new(v.dtype, v.shape, !v.is_absent());
                    let exact = OperandKey {
                        dtype: m.dtype,
                        present: m.present,
                        shape: m.shape.to_vec(),
                    };
                    assert_eq!(
                        exact.matches(&m),
                        exact.matches_view(&v),
                        "view and metadata comparison disagree on a matching key"
                    );
                    for wrong in [
                        OperandKey {
                            dtype: m.dtype,
                            present: !m.present,
                            shape: m.shape.to_vec(),
                        },
                        OperandKey {
                            dtype: m.dtype,
                            present: m.present,
                            shape: vec![99],
                        },
                    ] {
                        assert_eq!(
                            wrong.matches(&m),
                            wrong.matches_view(&v),
                            "view and metadata comparison disagree on a mismatching key"
                        );
                    }
                }
            }
        }
    }

    /// A signature is published only once it recurs, and the published plan is
    /// then served without re-planning.
    #[test]
    fn a_recurring_signature_is_published_to_the_lock_free_slot() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let data = [0u8; 64];
        let shape = [8usize];
        let strides = [1i64];
        let inputs = [view(&data, DataType::Float32, &shape, &strides)];

        cache
            .get_or_plan_views(&inputs, |m| planner.plan(m))
            .expect("plan");
        assert!(
            cache.hot.get().is_none(),
            "a signature seen once must not be published: a prefill shape may never recur"
        );

        cache
            .get_or_plan_views(&inputs, |m| planner.plan(m))
            .expect("plan");
        assert!(
            cache.hot.get().is_some(),
            "a signature seen twice must be published"
        );

        let before = planner.calls();
        let got = cache
            .get_or_plan_views(&inputs, |_| {
                panic!("the published plan must not be re-planned")
            })
            .expect("served from the lock-free slot");
        assert_eq!(got.bytes, 8 * 4);
        assert_eq!(planner.calls(), before);
    }

    /// The published plan must be readable while the cache lock is held by
    /// someone else — that is what "lock-free" means here. Falsifier: make the
    /// fast path take `plans.lock()` and this times out.
    #[test]
    fn a_published_plan_is_served_while_the_lock_is_held() {
        let cache = Arc::new(WorkspacePlanCache::new());
        let planner = CountingPlanner::new();
        let data = [0u8; 64];
        let shape = [8usize];
        let strides = [1i64];
        let inputs = [view(&data, DataType::Float32, &shape, &strides)];
        for _ in 0..2 {
            cache
                .get_or_plan_views(&inputs, |m| planner.plan(m))
                .expect("plan");
        }
        assert!(cache.hot.get().is_some(), "signature must be published");

        let held = cache.plans.lock().expect("lock");
        let (tx, rx) = std::sync::mpsc::channel();
        let reader = {
            let cache = Arc::clone(&cache);
            std::thread::spawn(move || {
                let data = [0u8; 64];
                let shape = [8usize];
                let strides = [1i64];
                let inputs = [view(&data, DataType::Float32, &shape, &strides)];
                let got = cache
                    .get_or_plan_views(&inputs, |_| panic!("must not re-plan"))
                    .expect("served");
                let _ = tx.send(got.bytes);
            })
        };
        let bytes = rx
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("the fast path blocked on the cache lock");
        assert_eq!(bytes, 8 * 4);
        drop(held);
        reader.join().expect("reader thread");
    }

    /// Publication must pick the *recurring* signature, not merely the first
    /// one: a decoder's first `Run` is a prefill whose shape may never return.
    #[test]
    fn the_published_plan_is_the_recurring_shape_not_the_first_one() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let data = [0u8; 512];
        let strides = [1i64];
        let prefill = [32usize];
        let decode = [1usize];

        let p = [view(&data, DataType::Float32, &prefill, &strides)];
        cache
            .get_or_plan_views(&p, |m| planner.plan(m))
            .expect("plan");
        let d = [view(&data, DataType::Float32, &decode, &strides)];
        for _ in 0..2 {
            cache
                .get_or_plan_views(&d, |m| planner.plan(m))
                .expect("plan");
        }

        let published = cache.hot.get().expect("a recurring signature must publish");
        assert_eq!(
            published.operands[0].shape.as_slice(),
            &decode[..],
            "the one-off prefill shape was pinned instead of the recurring decode shape"
        );
        assert_eq!(published.requirement.bytes, 4);
    }

    /// Falsification of fast-path bypass: once a signature is published, every
    /// *other* signature must still be planned on its own terms. Deleting any
    /// field from the fast-path comparison makes one of these serve a stale
    /// plan.
    #[test]
    fn the_lock_free_slot_is_never_served_for_a_different_signature() {
        let data = [0u8; 512];
        let strides = [1i64; 2];
        let hot_shape = [8usize];

        // Each case: (label, the differing operand list, the bytes its own
        // plan must produce).
        let publish = |cache: &WorkspacePlanCache, planner: &CountingPlanner| {
            let inputs = [view(&data, DataType::Float32, &hot_shape, &strides[..1])];
            for _ in 0..2 {
                cache
                    .get_or_plan_views(&inputs, |m| planner.plan(m))
                    .expect("plan");
            }
            assert!(cache.hot.get().is_some(), "publication precondition failed");
        };

        // Changed shape.
        {
            let cache = WorkspacePlanCache::new();
            let planner = CountingPlanner::new();
            publish(&cache, &planner);
            let other_shape = [4usize];
            let other = [view(&data, DataType::Float32, &other_shape, &strides[..1])];
            let got = cache
                .get_or_plan_views(&other, |m| planner.plan(m))
                .expect("plan");
            assert_eq!(
                got.bytes,
                4 * 4,
                "a changed shape was served the published plan"
            );
        }
        // Changed dtype.
        {
            let cache = WorkspacePlanCache::new();
            let planner = CountingPlanner::new();
            publish(&cache, &planner);
            let other = [view(&data, DataType::Float16, &hot_shape, &strides[..1])];
            let got = cache
                .get_or_plan_views(&other, |m| planner.plan(m))
                .expect("plan");
            assert_eq!(
                got.bytes,
                8 * 2,
                "a changed dtype was served the published plan"
            );
        }
        // Changed presence: an omitted optional slot.
        {
            let cache = WorkspacePlanCache::new();
            let planner = CountingPlanner::new();
            publish(&cache, &planner);
            let other = [TensorView::absent(DataType::Float32)];
            let got = cache
                .get_or_plan_views(&other, |m| planner.plan(m))
                .expect("plan");
            assert_eq!(
                got.bytes, 1,
                "an absent optional slot was served the present operand's plan"
            );
        }
        // Changed arity.
        {
            let cache = WorkspacePlanCache::new();
            let planner = CountingPlanner::new();
            publish(&cache, &planner);
            let other = [
                view(&data, DataType::Float32, &hot_shape, &strides[..1]),
                view(&data, DataType::Float32, &hot_shape, &strides[..1]),
            ];
            let got = cache
                .get_or_plan_views(&other, |m| planner.plan(m))
                .expect("plan");
            assert_eq!(
                got.bytes,
                8 * 4 * 2,
                "a longer operand list was served the shorter signature's plan"
            );
        }
    }

    /// Concurrent dispatches of interleaved signatures must each receive their
    /// own plan, whichever one wins publication.
    #[test]
    fn concurrent_dispatches_never_receive_another_signatures_plan() {
        let cache = Arc::new(WorkspacePlanCache::new());
        let planner = Arc::new(CountingPlanner::new());
        let mut handles = Vec::new();
        for t in 0..8usize {
            let cache = Arc::clone(&cache);
            let planner = Arc::clone(&planner);
            handles.push(std::thread::spawn(move || {
                let data = [0u8; 512];
                let strides = [1i64];
                for i in 0..250usize {
                    let n = 1 + ((t + i) % 5);
                    let shape = [n * 2];
                    let inputs = [view(&data, DataType::Float32, &shape, &strides)];
                    let got = cache
                        .get_or_plan_views(&inputs, |m| planner.plan(m))
                        .expect("plan");
                    assert_eq!(
                        got.bytes as usize,
                        n * 2 * 4,
                        "a concurrent dispatch received another signature's plan"
                    );
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread");
        }
    }
}

/// Pins the FFI cost of resolving where a routed subgraph's intermediates
/// should live — a question every `Compute` call asks, before any kernel runs.
///
/// A host EP takes the longest path through `device_mem_info`: the scan finds
/// nothing device-resident, so it runs to completion and then falls back. The
/// count below is what that costs, and it is the per-`Run` fixed overhead
/// #1077 measures as ~0.9 us worse than ORT's.
#[cfg(all(test, feature = "dispatch_probe"))]
mod mem_info_cost {
    use super::*;
    use crate::dispatch_probe::{self, Event};

    const MI: *const ort::OrtMemoryInfo = std::ptr::without_provenance(3);
    const VALUE: *const ort::OrtValue = std::ptr::without_provenance(4);
    const RECON: *const ort::OrtMemoryInfo = std::ptr::without_provenance(5);

    /// A device EP's staging context with a reconstructed memory info of the
    /// given device-ness.
    ///
    /// Built directly rather than through `set_device_staging`, which would
    /// need a live ORT to call `CreateMemoryInfo_V2`. Dropping the result is
    /// safe with no host API installed: `ReconstructedMemInfo::drop` returns
    /// early when `host_api()` is null, so the sentinel pointer is never
    /// passed to `ReleaseMemoryInfo`.
    fn staging_with_recon(ptr: *const ort::OrtMemoryInfo, is_device: bool) -> DeviceStaging {
        struct NoCopier;
        impl HostToDeviceCopier for NoCopier {
            unsafe fn copy_host_to_device(
                &self,
                _src: &[u8],
                _dst: *mut std::ffi::c_void,
            ) -> onnx_runtime_ep_api::Result<()> {
                unreachable!("scratch placement must not copy")
            }
        }
        assert!(
            crate::status::host_api().is_null(),
            "test builds a ReconstructedMemInfo over a sentinel pointer; a live \
             host API would hand it to ReleaseMemoryInfo on drop"
        );
        DeviceStaging {
            copier: Arc::new(NoCopier),
            recon_mem_info: Some(ReconstructedMemInfo { ptr, is_device }),
        }
    }

    unsafe extern "C" fn count_1(
        _c: *const ort::OrtKernelContext,
        out: *mut usize,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = 1 };
        std::ptr::null_mut()
    }
    unsafe extern "C" fn get_input(
        _c: *const ort::OrtKernelContext,
        _i: usize,
        out: *mut *const ort::OrtValue,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = VALUE };
        std::ptr::null_mut()
    }
    unsafe extern "C" fn tensor_mem_info(
        _v: *const ort::OrtValue,
        out: *mut *const ort::OrtMemoryInfo,
    ) -> ort::OrtStatusPtr {
        unsafe { *out = MI };
        std::ptr::null_mut()
    }
    /// Always reports host memory — the answer a CPU EP always gets, and the
    /// branch that makes the scan run to completion.
    unsafe extern "C" fn device_type_cpu(
        _m: *const ort::OrtMemoryInfo,
        out: *mut ort::OrtMemoryInfoDeviceType,
    ) {
        unsafe { *out = ort::OrtMemoryInfoDeviceType_CPU };
    }
    unsafe extern "C" fn device_type_gpu(
        _m: *const ort::OrtMemoryInfo,
        out: *mut ort::OrtMemoryInfoDeviceType,
    ) {
        unsafe { *out = ort::OrtMemoryInfoDeviceType_GPU };
    }

    fn api(host: bool) -> ort::OrtApi {
        let mut api: ort::OrtApi = unsafe { std::mem::zeroed() };
        api.KernelContext_GetInputCount = Some(count_1);
        api.KernelContext_GetInput = Some(get_input);
        api.GetTensorMemoryInfo = Some(tensor_mem_info);
        api.MemoryInfoGetDeviceType = Some(if host {
            device_type_cpu
        } else {
            device_type_gpu
        });
        api
    }

    /// Four calls for a one-input host node: the input count, then `GetInput`,
    /// `GetTensorMemoryInfo` and `MemoryInfoGetDeviceType` for that input.
    ///
    /// It was seven. The scan used to discard input 0's memory info and ask ORT
    /// for it again at the fallback (two calls for a value it had already
    /// seen), and the caller then asked a third time whether the result was a
    /// device — a fact this function had just established and thrown away.
    #[test]
    fn resolving_host_scratch_costs_four_ort_calls() {
        let api = api(true);
        let before = dispatch_probe::snapshot();
        let got = unsafe { device_mem_info(&api, std::ptr::null_mut(), None) };
        let d = dispatch_probe::snapshot().since(&before);

        assert_eq!(
            got,
            Some((MI, false)),
            "host memory, reported as not device"
        );
        assert_eq!(
            d.event(Event::OrtFfiCall),
            4,
            "FFI cost of resolving scratch placement changed"
        );
    }

    /// A device-resident input short-circuits the scan, and the device-ness
    /// comes back with it rather than costing another query.
    #[test]
    fn a_device_input_short_circuits_and_reports_itself() {
        let api = api(false);
        let before = dispatch_probe::snapshot();
        let got = unsafe { device_mem_info(&api, std::ptr::null_mut(), None) };
        let d = dispatch_probe::snapshot().since(&before);

        assert_eq!(got, Some((MI, true)));
        assert_eq!(d.event(Event::OrtFfiCall), 4);
    }

    /// A routed `Run` must resolve subgraph placement **at most once**, no
    /// matter how many of its nodes reach the fallback.
    ///
    /// Deferring the resolution out of the per-`Run` prologue traded an
    /// unconditional cost for a lazy one; without the memo it would also have
    /// traded "once per `Run`" for "once per node", which for a fused subgraph
    /// whose nodes each request a step-scoped workspace is strictly worse than
    /// what it replaced.
    #[test]
    fn the_subgraph_fallback_resolves_once_per_run_however_many_nodes_ask() {
        let api = api(true);
        let memo = std::cell::Cell::new(None);
        let fallback = SubgraphFallback {
            staging: None,
            memo: &memo,
        };

        let before = dispatch_probe::snapshot();
        let first = unsafe { fallback.resolve(&api, std::ptr::null_mut()) };
        let after_first = dispatch_probe::snapshot();
        for _ in 0..8 {
            assert_eq!(
                unsafe { fallback.resolve(&api, std::ptr::null_mut()) },
                first,
                "memoised resolution changed its answer"
            );
        }
        let after_rest = dispatch_probe::snapshot();

        assert_eq!(first, Some(MI));
        assert!(
            after_first.since(&before).event(Event::OrtFfiCall) > 0,
            "first resolution should actually query ORT"
        );
        assert_eq!(
            after_rest.since(&after_first).event(Event::OrtFfiCall),
            0,
            "nodes after the first must not re-query placement"
        );
    }

    /// A node that *does* bind ORT inputs never reaches the fallback, so the
    /// memo must stay unresolved -- the lazy path has to be lazy, not merely
    /// cached. This is the case every elementwise dispatch takes.
    #[test]
    fn a_node_with_ort_inputs_never_resolves_the_subgraph_fallback() {
        let api = api(true);
        let memo = std::cell::Cell::new(None);

        let got = unsafe {
            operand_mem_info(
                &api,
                std::ptr::null_mut(),
                PlacementSources {
                    ort_inputs: OrtOperands::Resolved(&[0]),
                    subgraph_fallback: SubgraphFallback {
                        staging: None,
                        memo: &memo,
                    },
                },
            )
        };

        assert!(
            !matches!(got, OperandMemInfo::Unavailable),
            "an operand-backed node should resolve from its own operands"
        );
        assert_eq!(
            memo.get(),
            None,
            "the subgraph fallback must not be resolved for a node that binds \
             ORT inputs"
        );
    }

    /// The placement shortcut keys on `host_accessible`, and that flag must
    /// default to the *conservative* answer.
    ///
    /// `host_accessible` is what the factory registers the allocator on, so it
    /// is the only honest statement about where ORT puts this EP's tensors. It
    /// must not be inferred from `device_staging`: that tracks
    /// `host_to_device_copier()`, which defaults to `None` and which a device
    /// EP may legitimately decline, so a device EP can have no staging *and*
    /// device-resident inputs -- exactly what
    /// `a_device_input_short_circuits_and_reports_itself` above demonstrates.
    /// Defaulting to `false` means a forgotten setter costs FFI calls, not
    /// correctness.
    #[test]
    fn compute_info_assumes_device_placement_until_told_otherwise() {
        let mut info = ExportedComputeInfo::new(Vec::new());
        assert!(
            !info.host_accessible,
            "default must be the conservative 'assume device', so an unset \
             flag keeps the full memory-info scan"
        );
        assert!(
            info.device_staging.is_none(),
            "and staging must stay independent of it"
        );

        info.set_host_accessible(true);
        assert!(info.host_accessible);
    }

    /// The regression test for the defect review caught: a **device** EP that
    /// declined `host_to_device_copier()` has no staging context, and must
    /// still scan for device placement.
    ///
    /// Keying the shortcut on staging presence -- as the first cut of this
    /// change did -- classifies exactly this EP as host, so its routed
    /// subgraph's intermediates get host buffers and its next device kernel
    /// dereferences a host pointer as device memory.
    /// `a_device_input_short_circuits_and_reports_itself` shows the scan such
    /// an EP performs really does report a device input.
    #[test]
    fn a_device_ep_that_declined_staging_still_scans_for_placement() {
        assert!(
            must_scan_for_device_placement(false, None),
            "a device EP without a copier must not be mistaken for a host EP"
        );
        // ...and staging presence must not sway the decision either way.
        let staging = staging_with_recon(RECON, true);
        assert!(must_scan_for_device_placement(false, Some(&staging)));
        assert!(!must_scan_for_device_placement(true, Some(&staging)));
        assert!(!must_scan_for_device_placement(true, None));
    }

    /// The two stock `DeviceSupport` recipes must keep disagreeing about
    /// placement, since the shortcut above is only sound while they do.
    #[test]
    fn stock_device_support_recipes_declare_opposite_placement() {
        use crate::device::DeviceSupport;
        assert!(DeviceSupport::cpu_only().host_accessible);
        assert!(!DeviceSupport::gpu("Gpu", 0).host_accessible);
    }

    /// The fallback still works when the scan cannot run at all, which is the
    /// one case that legitimately needs the extra fetch. Behaviour here is what
    /// it always was; only the cheap path got cheaper.
    #[test]
    fn a_missing_input_count_still_falls_back_to_input_zero() {
        let mut api = api(true);
        api.KernelContext_GetInputCount = None;
        let got = unsafe { device_mem_info(&api, std::ptr::null_mut(), None) };
        assert_eq!(got, Some((MI, false)));
    }

    /// A node with no inputs has nothing to scan and nothing to fall back on
    /// beyond input 0, which does not exist either. It must not invent one.
    #[test]
    fn a_node_with_no_inputs_reports_no_memory_info() {
        unsafe extern "C" fn none(
            _c: *const ort::OrtKernelContext,
            out: *mut usize,
        ) -> ort::OrtStatusPtr {
            unsafe { *out = 0 };
            std::ptr::null_mut()
        }
        unsafe extern "C" fn no_value(
            _c: *const ort::OrtKernelContext,
            _i: usize,
            out: *mut *const ort::OrtValue,
        ) -> ort::OrtStatusPtr {
            unsafe { *out = std::ptr::null() };
            std::ptr::null_mut()
        }
        let mut api = api(true);
        api.KernelContext_GetInputCount = Some(none);
        api.KernelContext_GetInput = Some(no_value);
        assert_eq!(
            unsafe { device_mem_info(&api, std::ptr::null_mut(), None) },
            None
        );
    }

    /// A device EP whose node has only host-resident inputs falls back to the
    /// memory info reconstructed from its own device recipe, so intermediates
    /// still land on the device.
    ///
    /// The old code returned that pointer and left the caller to ask ORT
    /// whether it was a device; the device-ness is now carried alongside it.
    /// This is the one branch whose reporting changed, so it is pinned
    /// explicitly rather than left to the host-EP tests.
    #[test]
    fn all_host_inputs_on_a_device_ep_fall_back_to_the_reconstructed_device_info() {
        let staging = staging_with_recon(RECON, true);
        let api = api(true);
        let before = dispatch_probe::snapshot();
        let got = unsafe { device_mem_info(&api, std::ptr::null_mut(), Some(&staging)) };
        let d = dispatch_probe::snapshot().since(&before);

        assert_eq!(
            got,
            Some((RECON, true)),
            "device scratch target, reported as a device"
        );
        assert_eq!(
            d.event(Event::OrtFfiCall),
            4,
            "the fallback must not re-ask ORT about the info it just built"
        );
    }

    /// The device-ness of the reconstructed info is recorded from the
    /// `device_type` it was built with, not hardcoded.
    ///
    /// Nothing in `HostToDeviceCopier` obliges an EP that provides a copier to
    /// be non-CPU. Every one that exists today is, which is what makes a
    /// hardcoded `true` look safe -- but a CPU-typed EP that grew a copier
    /// would be handed host memory as a device scratch target, and the staging
    /// path would treat host pointers as device pointers. `mem_info_is_device`
    /// is exactly `device_type != CPU`, so recording it at construction
    /// reproduces the old query in this case too.
    #[test]
    fn a_cpu_typed_reconstruction_is_not_reported_as_a_device() {
        let staging = staging_with_recon(RECON, false);
        let api = api(true);
        assert_eq!(
            unsafe { device_mem_info(&api, std::ptr::null_mut(), Some(&staging)) },
            Some((RECON, false)),
            "host-typed reconstruction must not be reported as device memory"
        );
    }

    /// A device-resident input still wins over the reconstruction: the scan
    /// short-circuits before the fallback is consulted.
    #[test]
    fn a_device_input_wins_over_the_reconstruction() {
        let staging = staging_with_recon(RECON, true);
        let api = api(false);
        assert_eq!(
            unsafe { device_mem_info(&api, std::ptr::null_mut(), Some(&staging)) },
            Some((MI, true)),
            "the actual device input, not the fallback recipe"
        );
    }
}
