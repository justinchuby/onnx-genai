//! CPU kernels for the Phase-1 BERT-on-CPU correctness milestone (`docs/ORT2.md`
//! §4.4). One [`Kernel`] per ONNX op, keyed purely by op type — there are **no**
//! model-specific shapes or names anywhere in this crate; BERT is only the
//! validation target.
//!
//! ## Pure-Rust reference kernels (architecture decision)
//!
//! These are straightforward, **correct** pure-Rust kernels — the ops other than
//! the GEMM hot spot use naive loops with no FFI or `cc` build dependency. The
//! MatMul GEMM went through the Phase-1.5 perf pass (`docs/ORT2.md` §25.2): its
//! default backend is a blocked, register-tiled, rayon-parallelized pure-Rust
//! kernel, with an optional statically-linked oneDNN (`dnnl_sgemm`) backend
//! behind the non-default `onednn` feature. Every kernel sits behind the
//! [`Kernel`] trait, so backends swap in **without touching the EP contract or
//! the session**. The seam is [`Kernel`] itself; see [`matmul`] for the hot spot
//! and [`crate::backend`] for backend selection.
//!
//! ## Strided inputs
//!
//! Kernels accept non-contiguous inputs by reading through
//! [`to_dense_f32`]/[`to_dense_i64`], which materialize a view (applying its
//! strides and byte offset) into a dense row-major buffer. This keeps the
//! per-kernel `unsafe` surface to the two element accessors in this module.

use onnx_runtime_ep_api::{EpError, OpKey, OpRegistry, Result, TensorMut, TensorView};
use onnx_runtime_ir::DataType;

use crate::strided::{elem_offset, next_index, numel};

pub mod activations;
pub mod add;
pub mod attention;
pub mod cast;
pub mod concat;
pub mod constant;
pub mod constant_of_shape;
pub mod contrib_fused;
pub mod elementwise;
pub mod expand;
pub mod fused_attention;
pub mod fused_gemm;
pub mod fused_matmul_bias;
pub mod gather;
pub mod gelu;
pub mod gemm;
pub mod group_query_attention;
pub mod identity;
pub mod indexing;
pub mod layernorm;
pub mod log_softmax;
pub mod logical;
pub mod matmul;
pub mod matmul_nbits;
pub mod moe;
pub mod movement_ops;
pub mod norm_ops;
#[cfg(feature = "onednn")]
pub mod onednn;
pub mod pad;
pub mod pooling;
pub mod quantization;
pub mod reduce;
pub mod reduce_ops;
pub mod relu;
pub mod reshape;
pub mod rmsnorm;
pub mod rotary_embedding;
pub mod selection;
pub mod sequence;
pub mod shape;
pub mod skip_simplified_layernorm;
pub mod slice;
pub mod softmax;
pub mod split;
pub mod transpose;
pub mod unary_math;
pub mod unsqueeze;
pub mod where_op;

/// The set of ops the CPU EP implements for the Phase-1 BERT-on-CPU milestone.
pub const PHASE1_OPS: &[&str] = &[
    "MatMul",
    "Add",
    "Relu",
    "Reshape",
    "Transpose",
    "Gather",
    "LayerNormalization",
    // Elementwise binary (numpy broadcasting).
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Min",
    "Max",
    "Sum",
    "Mean",
    // Elementwise unary.
    "Sqrt",
    "Erf",
    "Tanh",
    "Cast",
    "CastLike",
    // Additional elementwise unary math (unary_math.rs).
    "Abs",
    "Neg",
    "Reciprocal",
    "Exp",
    "Log",
    "Sign",
    "Floor",
    "Ceil",
    "Round",
    "Sin",
    "Cos",
    "Sigmoid",
    "Softplus",
    "Softsign",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Cosh",
    "Sinh",
    "Tan",
    "Elu",
    "LeakyRelu",
    "HardSigmoid",
    // Logical / selection.
    "Not",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Where",
    // Reduction / normalization.
    "ReduceMean",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "Softmax",
    "LogSoftmax",
    // Shape / data movement.
    "Shape",
    "Unsqueeze",
    "Expand",
    "Slice",
    "Constant",
    "Identity",
    "Concat",
    "Flatten",
    "Squeeze",
    "Split",
    "Pad",
    "ConstantOfShape",
    "Size",
    "Trilu",
    "GatherElements",
    "GatherND",
    "ScatterElements",
    "OneHot",
    "Tile",
    "Range",
    "CumSum",
    "Clip",
    "ArgMax",
    "ArgMin",
    "TopK",
    "NonZero",
    // GEMM.
    "Gemm",
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
];

/// Whether `op_type` is one of the Phase-1 ops the CPU EP can run.
pub fn is_phase1_op(op_type: &str) -> bool {
    PHASE1_OPS.contains(&op_type)
}

/// Build an [`OpRegistry`] populated with every Phase-1 CPU kernel factory.
///
/// The provider consults this to instantiate kernels, and Track D (session) can
/// reuse the same registry for its own placement/lookup. All ops are registered
/// under the default domain (`""`) at `since_version` 1; the registry's
/// `lookup` picks the highest applicable version, so future opset-specialized
/// kernels can be added alongside these.
pub fn build_cpu_registry() -> OpRegistry {
    let mut reg = OpRegistry::new();
    reg.register(OpKey::new("MatMul", "", 1), Box::new(matmul::MatMulFactory));
    reg.register(
        OpKey::new("MatMulNBits", "com.microsoft", 1),
        Box::new(matmul_nbits::MatMulNBitsFactory),
    );
    reg.register(OpKey::new("Add", "", 1), Box::new(add::AddFactory));
    reg.register(OpKey::new("Relu", "", 1), Box::new(relu::ReluFactory));
    reg.register(
        OpKey::new("Reshape", "", 1),
        Box::new(reshape::ReshapeFactory),
    );
    reg.register(
        OpKey::new("Transpose", "", 1),
        Box::new(transpose::TransposeFactory),
    );
    reg.register(OpKey::new("Gather", "", 1), Box::new(gather::GatherFactory));
    reg.register(
        OpKey::new("LayerNormalization", "", 1),
        Box::new(layernorm::LayerNormFactory),
    );
    // The optimizer emits fused `LayerNormalization` in the private contrib
    // domain (`com.microsoft`); bind the same kernel there so dispatch resolves
    // the fused op by (domain, op_type). The default-domain registration above
    // still serves standard ONNX `LayerNormalization`.
    reg.register(
        OpKey::new("LayerNormalization", "com.microsoft", 1),
        Box::new(layernorm::LayerNormFactory),
    );
    // The optimizer's `MatMul + Add(bias)` fusion emits `FusedMatMulBias` in the
    // contrib domain; bind its kernel there so dispatch resolves the fused op by
    // (domain, op_type). It reuses the shared MatMul GEMM + broadcast-Add.
    reg.register(
        OpKey::new("FusedMatMulBias", "com.microsoft", 1),
        Box::new(fused_matmul_bias::FusedMatMulBiasFactory),
    );
    // The optimizer's `MatMul + Add(bias) + Relu` fusion emits `FusedGemm` in
    // the contrib domain; bind its kernel there so dispatch resolves the fused
    // op by (domain, op_type). It reuses the shared MatMul GEMM + broadcast-Add
    // + elementwise Relu.
    reg.register(
        OpKey::new("FusedGemm", "com.microsoft", 1),
        Box::new(fused_gemm::FusedGemmFactory),
    );
    // The optimizer's SDPA-core fusion (MatMul(QKᵀ) → scale → [+mask] → Softmax
    // → MatMul(·V)) emits `FusedAttention` in the contrib domain; bind its
    // kernel there so dispatch resolves the fused op by (domain, op_type). It
    // reuses the shared MatMul GEMM (twice), broadcast-Add (mask) and the
    // extracted last-axis softmax helper.
    reg.register(
        OpKey::new("FusedAttention", "com.microsoft", 1),
        Box::new(fused_attention::FusedAttentionFactory),
    );
    reg.register(
        OpKey::new("GroupQueryAttention", "com.microsoft", 1),
        Box::new(group_query_attention::GroupQueryAttentionFactory),
    );
    // Standard `ai.onnx::Attention`: the richer SDPA op with 3D/4D inputs,
    // GQA/MQA head sharing, a KV cache (`past_*`/`present_*`), causal masking,
    // softcap, and up to four outputs. Distinct from the contrib
    // `FusedAttention` above. Added at opset 23 and revised at opset 24; since
    // no newer version exists, the opset-24 kernel serves model opsets 24, 25
    // and 26 (the registry resolves the highest `since_version <= opset`). Both
    // versions are registered so opset-23 models keep the original
    // `qk_matmul_output_mode` 1↔2 ordering while opset-24+ models get the
    // swapped ordering and `nonpad_kv_seqlen` support.
    reg.register(
        OpKey::new("Attention", "", 23),
        Box::new(attention::AttentionFactory { since_version: 23 }),
    );
    reg.register(
        OpKey::new("Attention", "", 24),
        Box::new(attention::AttentionFactory { since_version: 24 }),
    );
    // The optimizer's exact-GELU fusion emits `com.microsoft::Gelu`; bind its
    // CPU kernel in the same contrib domain (there is no standard-domain `Gelu`
    // op, so it is registered only under `com.microsoft`).
    reg.register(
        OpKey::new("Gelu", "com.microsoft", 1),
        Box::new(gelu::GeluFactory),
    );
    reg.register(
        OpKey::new("BiasGelu", "com.microsoft", 1),
        Box::new(contrib_fused::BiasGeluFactory),
    );
    reg.register(
        OpKey::new("FastGelu", "com.microsoft", 1),
        Box::new(contrib_fused::FastGeluFactory),
    );
    reg.register(
        OpKey::new("QuickGelu", "com.microsoft", 1),
        Box::new(contrib_fused::QuickGeluFactory),
    );
    reg.register(
        OpKey::new("SkipLayerNormalization", "com.microsoft", 1),
        Box::new(contrib_fused::SkipLayerNormFactory),
    );
    reg.register(
        OpKey::new("SimplifiedLayerNormalization", "com.microsoft", 1),
        Box::new(contrib_fused::SimplifiedLayerNormFactory),
    );
    reg.register(
        OpKey::new("SkipSimplifiedLayerNormalization", "com.microsoft", 1),
        Box::new(skip_simplified_layernorm::SkipSimplifiedLayerNormFactory),
    );
    reg.register(
        OpKey::new("MoE", "com.microsoft", 1),
        Box::new(moe::MoEFactory),
    );
    // Standard-domain LLM/transformer primitives (ai.onnx). Registered at their
    // ONNX since_version; the registry resolves the highest since_version <=
    // model opset.
    //
    // `ai.onnx::Gelu` was added at opset 20 with the `approximate` attribute
    // ("none" = exact erf, "tanh" = tanh approximation). Distinct from the
    // com.microsoft::Gelu contrib op above.
    reg.register(OpKey::new("Gelu", "", 20), Box::new(gelu::StdGeluFactory));
    // `ai.onnx::RMSNormalization` added at opset 23.
    reg.register(
        OpKey::new("RMSNormalization", "", 23),
        Box::new(rmsnorm::RmsNormFactory),
    );
    reg.register(
        OpKey::new("BatchNormalization", "", 15),
        Box::new(norm_ops::BatchNormFactory),
    );
    reg.register(
        OpKey::new("InstanceNormalization", "", 6),
        Box::new(norm_ops::InstanceNormFactory),
    );
    // GroupNormalization v18 uses per-group scale/bias. Opset 21 changed the
    // affine inputs to per-channel, so keep versioned factories for both schemas.
    reg.register(
        OpKey::new("GroupNormalization", "", 18),
        Box::new(norm_ops::GroupNormFactory { since_version: 18 }),
    );
    reg.register(
        OpKey::new("GroupNormalization", "", 21),
        Box::new(norm_ops::GroupNormFactory { since_version: 21 }),
    );
    reg.register(
        OpKey::new("PRelu", "", 16),
        Box::new(norm_ops::PReluFactory),
    );
    // `ai.onnx::RotaryEmbedding` added at opset 23.
    reg.register(
        OpKey::new("RotaryEmbedding", "", 23),
        Box::new(rotary_embedding::RotaryEmbeddingFactory),
    );
    // `ai.onnx::Swish` added at opset 24: y = x·sigmoid(alpha·x).
    reg.register(
        OpKey::new("Swish", "", 24),
        Box::new(activations::SwishFactory),
    );
    // Elementwise binary broadcasting ops.
    reg.register(OpKey::new("Sub", "", 1), Box::new(elementwise::SubFactory));
    reg.register(OpKey::new("Mul", "", 1), Box::new(elementwise::MulFactory));
    reg.register(OpKey::new("Div", "", 1), Box::new(elementwise::DivFactory));
    reg.register(OpKey::new("Pow", "", 1), Box::new(elementwise::PowFactory));
    reg.register(OpKey::new("Min", "", 1), Box::new(elementwise::MinFactory));
    reg.register(OpKey::new("Max", "", 1), Box::new(elementwise::MaxFactory));
    reg.register(OpKey::new("Sum", "", 1), Box::new(elementwise::SumFactory));
    reg.register(
        OpKey::new("Mean", "", 1),
        Box::new(elementwise::MeanFactory),
    );
    // Elementwise unary ops.
    reg.register(
        OpKey::new("Sqrt", "", 1),
        Box::new(elementwise::SqrtFactory),
    );
    reg.register(OpKey::new("Erf", "", 1), Box::new(elementwise::ErfFactory));
    reg.register(
        OpKey::new("Tanh", "", 1),
        Box::new(elementwise::TanhFactory),
    );
    reg.register(OpKey::new("Cast", "", 1), Box::new(cast::CastFactory));
    reg.register(
        OpKey::new("CastLike", "", 15),
        Box::new(cast::CastLikeFactory),
    );
    // Identity: dtype-agnostic passthrough (raw byte copy).
    reg.register(
        OpKey::new("Identity", "", 1),
        Box::new(identity::IdentityFactory),
    );
    reg.register(
        OpKey::new("ReduceMean", "", 1),
        Box::new(reduce::ReduceMeanFactory),
    );
    // Softmax: legacy coerce-to-2D at opset ≤ 12, per-axis at opset ≥ 13. The
    // provider's opset-aware lookup selects the version-correct kernel.
    reg.register(
        OpKey::new("Softmax", "", 1),
        Box::new(softmax::SoftmaxLegacyFactory),
    );
    reg.register(
        OpKey::new("Softmax", "", 13),
        Box::new(softmax::SoftmaxFactory),
    );
    // LogSoftmax shares Softmax's opset split: legacy flattened trailing axes
    // through opset 12, then one-axis normalization from opset 13.
    reg.register(
        OpKey::new("LogSoftmax", "", 1),
        Box::new(log_softmax::LogSoftmaxLegacyFactory),
    );
    reg.register(
        OpKey::new("LogSoftmax", "", 13),
        Box::new(log_softmax::LogSoftmaxFactory),
    );
    // Shape / data movement.
    reg.register(OpKey::new("Shape", "", 1), Box::new(shape::ShapeFactory));
    reg.register(
        OpKey::new("Unsqueeze", "", 1),
        Box::new(unsqueeze::UnsqueezeFactory),
    );
    reg.register(OpKey::new("Expand", "", 1), Box::new(expand::ExpandFactory));
    reg.register(OpKey::new("Slice", "", 1), Box::new(slice::SliceFactory));
    reg.register(OpKey::new("Split", "", 1), Box::new(split::SplitFactory));
    reg.register(OpKey::new("Pad", "", 1), Box::new(pad::PadFactory));
    reg.register(
        OpKey::new("ConstantOfShape", "", 1),
        Box::new(constant_of_shape::ConstantOfShapeFactory),
    );
    reg.register(
        OpKey::new("Constant", "", 1),
        Box::new(constant::ConstantFactory),
    );
    // GEMM.
    reg.register(OpKey::new("Gemm", "", 1), Box::new(gemm::GemmFactory));
    // Linear quantization evolved at opsets 10, 13, 19, 21, 23, and 25. The
    // implementation accepts the newest parameter set for all these revisions.
    for version in [10, 13, 19, 21, 23, 25] {
        reg.register(
            OpKey::new("QuantizeLinear", "", version),
            Box::new(quantization::QuantizeLinearFactory),
        );
        reg.register(
            OpKey::new("DequantizeLinear", "", version),
            Box::new(quantization::DequantizeLinearFactory),
        );
    }
    reg.register(
        OpKey::new("DynamicQuantizeLinear", "", 11),
        Box::new(quantization::DynamicQuantizeLinearFactory),
    );
    // Spatial pooling. Newer registrations preserve version-specific attributes.
    reg.register(
        OpKey::new("AveragePool", "", 1),
        Box::new(pooling::AveragePoolFactory),
    );
    reg.register(
        OpKey::new("AveragePool", "", 7),
        Box::new(pooling::AveragePoolFactory),
    );
    reg.register(
        OpKey::new("AveragePool", "", 10),
        Box::new(pooling::AveragePoolFactory),
    );
    reg.register(
        OpKey::new("AveragePool", "", 11),
        Box::new(pooling::AveragePoolFactory),
    );
    reg.register(
        OpKey::new("AveragePool", "", 19),
        Box::new(pooling::AveragePoolFactory),
    );
    reg.register(
        OpKey::new("MaxPool", "", 1),
        Box::new(pooling::MaxPoolFactory),
    );
    reg.register(
        OpKey::new("MaxPool", "", 8),
        Box::new(pooling::MaxPoolFactory),
    );
    reg.register(
        OpKey::new("MaxPool", "", 10),
        Box::new(pooling::MaxPoolFactory),
    );
    reg.register(
        OpKey::new("MaxPool", "", 11),
        Box::new(pooling::MaxPoolFactory),
    );
    reg.register(
        OpKey::new("MaxPool", "", 12),
        Box::new(pooling::MaxPoolFactory),
    );
    reg.register(
        OpKey::new("GlobalAveragePool", "", 1),
        Box::new(pooling::GlobalAveragePoolFactory),
    );
    reg.register(
        OpKey::new("GlobalMaxPool", "", 1),
        Box::new(pooling::GlobalMaxPoolFactory),
    );
    // --- Additional ep-cpu op coverage (op-coverage wave) ---------------------
    // Elementwise unary math (f32). Additive, default-domain-only registrations.
    reg.register(OpKey::new("Abs", "", 1), Box::new(unary_math::AbsFactory));
    reg.register(OpKey::new("Neg", "", 1), Box::new(unary_math::NegFactory));
    reg.register(
        OpKey::new("Reciprocal", "", 1),
        Box::new(unary_math::ReciprocalFactory),
    );
    reg.register(OpKey::new("Exp", "", 1), Box::new(unary_math::ExpFactory));
    reg.register(OpKey::new("Log", "", 1), Box::new(unary_math::LogFactory));
    reg.register(OpKey::new("Sign", "", 1), Box::new(unary_math::SignFactory));
    reg.register(
        OpKey::new("Floor", "", 1),
        Box::new(unary_math::FloorFactory),
    );
    reg.register(OpKey::new("Ceil", "", 1), Box::new(unary_math::CeilFactory));
    reg.register(
        OpKey::new("Round", "", 1),
        Box::new(unary_math::RoundFactory),
    );
    reg.register(OpKey::new("Sin", "", 1), Box::new(unary_math::SinFactory));
    reg.register(OpKey::new("Cos", "", 1), Box::new(unary_math::CosFactory));
    reg.register(
        OpKey::new("Sigmoid", "", 1),
        Box::new(unary_math::SigmoidFactory),
    );
    reg.register(
        OpKey::new("Softplus", "", 1),
        Box::new(unary_math::SoftplusFactory),
    );
    reg.register(
        OpKey::new("Softsign", "", 1),
        Box::new(unary_math::SoftsignFactory),
    );
    reg.register(OpKey::new("Acos", "", 1), Box::new(unary_math::AcosFactory));
    reg.register(
        OpKey::new("Acosh", "", 1),
        Box::new(unary_math::AcoshFactory),
    );
    reg.register(OpKey::new("Asin", "", 1), Box::new(unary_math::AsinFactory));
    reg.register(
        OpKey::new("Asinh", "", 1),
        Box::new(unary_math::AsinhFactory),
    );
    reg.register(OpKey::new("Atan", "", 1), Box::new(unary_math::AtanFactory));
    reg.register(
        OpKey::new("Atanh", "", 1),
        Box::new(unary_math::AtanhFactory),
    );
    reg.register(OpKey::new("Cosh", "", 1), Box::new(unary_math::CoshFactory));
    reg.register(OpKey::new("Sinh", "", 1), Box::new(unary_math::SinhFactory));
    reg.register(OpKey::new("Tan", "", 1), Box::new(unary_math::TanFactory));
    reg.register(OpKey::new("Elu", "", 1), Box::new(activations::EluFactory));
    reg.register(
        OpKey::new("LeakyRelu", "", 1),
        Box::new(activations::LeakyReluFactory),
    );
    reg.register(
        OpKey::new("HardSigmoid", "", 1),
        Box::new(activations::HardSigmoidFactory),
    );
    // Logical / selection.
    reg.register(OpKey::new("Not", "", 1), Box::new(logical::NotFactory));
    reg.register(OpKey::new("Equal", "", 1), Box::new(logical::EqualFactory));
    reg.register(
        OpKey::new("Greater", "", 1),
        Box::new(logical::GreaterFactory),
    );
    reg.register(
        OpKey::new("GreaterOrEqual", "", 1),
        Box::new(logical::GreaterOrEqualFactory),
    );
    reg.register(OpKey::new("Less", "", 1), Box::new(logical::LessFactory));
    reg.register(
        OpKey::new("LessOrEqual", "", 1),
        Box::new(logical::LessOrEqualFactory),
    );
    reg.register(OpKey::new("Where", "", 1), Box::new(where_op::WhereFactory));
    // Reductions (axes attribute or opset-13/18 axes input).
    reg.register(
        OpKey::new("ReduceSum", "", 1),
        Box::new(reduce_ops::ReduceSumFactory),
    );
    reg.register(
        OpKey::new("ReduceMax", "", 1),
        Box::new(reduce_ops::ReduceMaxFactory),
    );
    reg.register(
        OpKey::new("ReduceMin", "", 1),
        Box::new(reduce_ops::ReduceMinFactory),
    );
    reg.register(
        OpKey::new("ReduceProd", "", 1),
        Box::new(reduce_ops::ReduceProdFactory),
    );
    reg.register(
        OpKey::new("ReduceSumSquare", "", 1),
        Box::new(reduce_ops::ReduceSumSquareFactory),
    );
    reg.register(
        OpKey::new("ReduceL1", "", 1),
        Box::new(reduce_ops::ReduceL1Factory),
    );
    reg.register(
        OpKey::new("ReduceL2", "", 1),
        Box::new(reduce_ops::ReduceL2Factory),
    );
    reg.register(
        OpKey::new("ReduceLogSum", "", 1),
        Box::new(reduce_ops::ReduceLogSumFactory),
    );
    reg.register(
        OpKey::new("ReduceLogSumExp", "", 1),
        Box::new(reduce_ops::ReduceLogSumExpFactory),
    );
    // Shape / data movement (dtype-agnostic byte movers).
    reg.register(OpKey::new("Concat", "", 1), Box::new(concat::ConcatFactory));
    reg.register(
        OpKey::new("Flatten", "", 1),
        Box::new(movement_ops::FlattenFactory),
    );
    reg.register(
        OpKey::new("Squeeze", "", 1),
        Box::new(movement_ops::SqueezeFactory),
    );
    reg.register(
        OpKey::new("Size", "", 1),
        Box::new(movement_ops::SizeFactory),
    );
    reg.register(
        OpKey::new("Trilu", "", 14),
        Box::new(movement_ops::TriluFactory),
    );
    // Indexed data movement and sequence construction.
    reg.register(
        OpKey::new("GatherElements", "", 11),
        Box::new(indexing::GatherElementsFactory),
    );
    reg.register(
        OpKey::new("GatherND", "", 11),
        Box::new(indexing::GatherNDFactory),
    );
    // ScatterElements gained its reduction attribute at opset 16.
    reg.register(
        OpKey::new("ScatterElements", "", 11),
        Box::new(indexing::ScatterElementsFactory),
    );
    reg.register(
        OpKey::new("ScatterElements", "", 16),
        Box::new(indexing::ScatterElementsFactory),
    );
    reg.register(
        OpKey::new("OneHot", "", 9),
        Box::new(indexing::OneHotFactory),
    );
    reg.register(OpKey::new("Tile", "", 6), Box::new(sequence::TileFactory));
    reg.register(
        OpKey::new("Range", "", 11),
        Box::new(sequence::RangeFactory),
    );
    reg.register(
        OpKey::new("CumSum", "", 11),
        Box::new(sequence::CumSumFactory),
    );
    // Value selection.
    reg.register(OpKey::new("Clip", "", 1), Box::new(selection::ClipFactory));
    reg.register(
        OpKey::new("ArgMax", "", 1),
        Box::new(selection::ArgMaxFactory),
    );
    reg.register(
        OpKey::new("ArgMin", "", 1),
        Box::new(selection::ArgMinFactory),
    );
    reg.register(OpKey::new("TopK", "", 10), Box::new(selection::TopKFactory));
    reg.register(
        OpKey::new("NonZero", "", 9),
        Box::new(selection::NonZeroFactory),
    );
    reg
}

// ---------------------------------------------------------------------------
// Shared view accessors — the only `unsafe` in the kernel layer.
// ---------------------------------------------------------------------------

/// Materialize an `f32` view into a dense, row-major `Vec<f32>`, applying the
/// view's strides and byte offset. Rejects non-`Float32` views.
pub fn to_dense_f32(view: &TensorView) -> Result<Vec<f32>> {
    view.validate()?;
    require_dtype(view.dtype, DataType::Float32, "f32 kernel input")?;
    let n = numel(view.shape);
    let origin = view.data_ptr::<f32>();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    let mut idx = vec![0usize; view.shape.len()];
    loop {
        let off = elem_offset(view.strides, &idx);
        // SAFETY: `origin` is the element origin of a validated view; `off` is
        // an in-shape element offset (each index component is `< shape[d]`), so
        // the address lies within the range the view describes. The owning EP
        // has already checked that range against the backing allocation via
        // `strided::view_in_bounds` (ep-api safety invariant #1). We never read
        // past the addressed extent, and `f32` has no invalid bit patterns.
        out.push(unsafe { *origin.offset(off) });
        if !next_index(view.shape, &mut idx) {
            break;
        }
    }
    Ok(out)
}

/// Materialize an integer index view (`Int64` or `Int32`) into a dense
/// `Vec<i64>`. Used for `Gather` indices.
pub fn to_dense_i64(view: &TensorView) -> Result<Vec<i64>> {
    view.validate()?;
    let n = numel(view.shape);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    let mut idx = vec![0usize; view.shape.len()];
    match view.dtype {
        DataType::Int64 => {
            let origin = view.data_ptr::<i64>();
            loop {
                let off = elem_offset(view.strides, &idx);
                // SAFETY: see `to_dense_f32` — in-shape offset over a validated,
                // bounds-checked view; `i64` has no invalid bit patterns.
                out.push(unsafe { *origin.offset(off) });
                if !next_index(view.shape, &mut idx) {
                    break;
                }
            }
        }
        DataType::Int32 => {
            let origin = view.data_ptr::<i32>();
            loop {
                let off = elem_offset(view.strides, &idx);
                // SAFETY: as above, for a 4-byte element type.
                out.push(unsafe { *origin.offset(off) } as i64);
                if !next_index(view.shape, &mut idx) {
                    break;
                }
            }
        }
        other => {
            return Err(EpError::InvalidTensorView {
                reason: format!("index tensor must be Int64 or Int32, got {other:?}"),
            });
        }
    }
    Ok(out)
}

/// Write a dense, row-major `f32` slice into `out`, applying the output view's
/// strides and byte offset. `data.len()` must equal the output element count.
pub fn write_dense_f32(out: &mut TensorMut, data: &[f32]) -> Result<()> {
    out.validate()?;
    require_dtype(out.dtype, DataType::Float32, "f32 kernel output")?;
    let n = numel(out.shape);
    if data.len() != n {
        return Err(EpError::KernelFailed(format!(
            "output element count {n} does not match produced {}",
            data.len()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let origin = out.data_ptr_mut::<f32>();
    let strides = out.strides;
    let shape = out.shape;
    let mut idx = vec![0usize; shape.len()];
    let mut i = 0usize;
    loop {
        let off = elem_offset(strides, &idx);
        // SAFETY: `origin` is the element origin of a validated output view;
        // `off` is an in-shape offset, so it lies within the extent the view
        // describes (bounds-checked against the backing allocation by the EP
        // per invariant #1). Each address is written exactly once because the
        // row-major walk visits every logical index once.
        unsafe {
            *origin.offset(off) = data[i];
        }
        i += 1;
        if !next_index(shape, &mut idx) {
            break;
        }
    }
    Ok(())
}

/// The fixed element byte-width of `dtype`. Errors for variable-width
/// ([`DataType::String`]) and sub-byte-packed (`Int4`/`Uint4`) types, which the
/// dtype-generic byte movers below cannot address one-element-at-a-time.
pub fn elem_size(dtype: DataType) -> Result<usize> {
    let size = dtype.byte_size();
    if size == 0 {
        return Err(EpError::InvalidTensorView {
            reason: format!("dtype {dtype:?} has no fixed-width byte layout"),
        });
    }
    Ok(size)
}

/// Materialize any fixed-width view into a dense, row-major byte buffer,
/// applying the view's strides and byte offset. This is the dtype-agnostic
/// counterpart to [`to_dense_f32`]: it copies raw element bytes without
/// interpreting them, so it serves the pure data-movement ops (Unsqueeze,
/// Expand, Slice, Cast source read) uniformly across dtypes.
pub fn to_dense_bytes(view: &TensorView) -> Result<Vec<u8>> {
    view.validate()?;
    let esize = elem_size(view.dtype)?;
    let n = numel(view.shape);
    let mut out = vec![0u8; n * esize];
    if n == 0 {
        return Ok(out);
    }
    // Byte origin of the element at logical index 0 (applies `byte_offset`).
    let origin = view.data_ptr::<u8>();
    let mut idx = vec![0usize; view.shape.len()];
    let mut w = 0usize;
    loop {
        let elem_off = elem_offset(view.strides, &idx);
        let byte_off = elem_off * esize as isize;
        // SAFETY: `origin` is the byte origin of a validated view; `elem_off` is
        // an in-shape element offset, so `byte_off .. byte_off + esize` lies
        // within the extent the view describes (bounds-checked against the
        // backing allocation by the EP per invariant #1). `out[w..w + esize]` is
        // a fresh, uniquely-owned buffer. The regions do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(origin.offset(byte_off), out.as_mut_ptr().add(w), esize);
        }
        w += esize;
        if !next_index(view.shape, &mut idx) {
            break;
        }
    }
    Ok(out)
}

/// Write a dense, row-major byte buffer into `out`, applying the output view's
/// strides and byte offset. `data.len()` must equal `numel(out) * elem_size`.
/// The dtype-agnostic counterpart to [`write_dense_f32`].
pub fn write_dense_bytes(out: &mut TensorMut, data: &[u8]) -> Result<()> {
    out.validate()?;
    let esize = elem_size(out.dtype)?;
    let n = numel(out.shape);
    if data.len() != n * esize {
        return Err(EpError::KernelFailed(format!(
            "output byte count {} does not match produced {}",
            n * esize,
            data.len()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let origin = out.data_ptr_mut::<u8>();
    let strides = out.strides;
    let shape = out.shape;
    let mut idx = vec![0usize; shape.len()];
    let mut r = 0usize;
    loop {
        let elem_off = elem_offset(strides, &idx);
        let byte_off = elem_off * esize as isize;
        // SAFETY: `origin` is the byte origin of a validated output view;
        // `byte_off .. byte_off + esize` is an in-shape offset lying within the
        // extent the view describes (bounds-checked by the EP per invariant #1).
        // Each destination range is written exactly once because the row-major
        // walk visits every logical index once; source and destination buffers
        // are distinct.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(r), origin.offset(byte_off), esize);
        }
        r += esize;
        if !next_index(shape, &mut idx) {
            break;
        }
    }
    Ok(())
}

/// Error out unless `got == want`.
fn require_dtype(got: DataType, want: DataType, ctx: &str) -> Result<()> {
    if got != want {
        return Err(EpError::InvalidTensorView {
            reason: format!("{ctx} requires {want:?}, got {got:?}"),
        });
    }
    Ok(())
}

/// Validate the arity of a kernel's input/output slices.
fn check_arity(
    op: &str,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    min_inputs: usize,
    max_inputs: usize,
    outputs_wanted: usize,
) -> Result<()> {
    if inputs.len() < min_inputs || inputs.len() > max_inputs {
        return Err(EpError::KernelFailed(format!(
            "{op}: expected {min_inputs}..={max_inputs} inputs, got {}",
            inputs.len()
        )));
    }
    if outputs.len() < outputs_wanted {
        return Err(EpError::KernelFailed(format!(
            "{op}: expected at least {outputs_wanted} output(s), got {}",
            outputs.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Helpers to build owning-buffer-backed views for kernel unit tests.

    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
    use onnx_runtime_ir::{DataType, DeviceId, compute_contiguous_strides};

    /// A dense f32 buffer plus the shape/stride metadata a view needs.
    pub struct Owned {
        pub bytes: Vec<u8>,
        pub shape: Vec<usize>,
        pub strides: Vec<i64>,
        pub dtype: DataType,
    }

    impl Owned {
        pub fn f32(shape: &[usize], data: &[f32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Float32,
            }
        }

        pub fn f64(shape: &[usize], data: &[f64]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 8);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Float64,
            }
        }

        pub fn i64(shape: &[usize], data: &[i64]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 8);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Int64,
            }
        }

        pub fn i32(shape: &[usize], data: &[i32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Int32,
            }
        }

        /// An f16 buffer built by rounding `data` (given in f32) to half.
        pub fn f16(shape: &[usize], data: &[f32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &v in data {
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Float16,
            }
        }

        /// An f16 buffer built from raw 16-bit patterns (for adversarial
        /// NaN/inf/denormal cases that must survive without f32-reinterpret).
        pub fn f16_bits(shape: &[usize], bits: &[u16]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(bits.len() * 2);
            for &b in bits {
                bytes.extend_from_slice(&b.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Float16,
            }
        }

        /// A bf16 buffer built by rounding `data` (given in f32) to bfloat16.
        pub fn bf16(shape: &[usize], data: &[f32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &v in data {
                bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::BFloat16,
            }
        }

        /// A bf16 buffer built from raw 16-bit patterns.
        pub fn bf16_bits(shape: &[usize], bits: &[u16]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(bits.len() * 2);
            for &b in bits {
                bytes.extend_from_slice(&b.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::BFloat16,
            }
        }

        /// A u8 buffer.
        pub fn u8(shape: &[usize], data: &[u8]) -> Self {
            let strides = compute_contiguous_strides(shape);
            Self {
                bytes: data.to_vec(),
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Uint8,
            }
        }

        pub fn bool_(shape: &[usize], data: &[bool]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let bytes = data.iter().map(|&b| b as u8).collect();
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Bool,
            }
        }

        /// A zero-filled f32 output buffer of `shape`.
        pub fn zeros_f32(shape: &[usize]) -> Self {
            let n: usize = shape.iter().product();
            Self::f32(shape, &vec![0.0; n])
        }

        /// A zero-filled output buffer of `shape` with element type `dtype`.
        pub fn zeros(dtype: DataType, shape: &[usize]) -> Self {
            let n: usize = shape.iter().product();
            let strides = compute_contiguous_strides(shape);
            let esize = dtype.byte_size();
            Self {
                bytes: vec![0u8; n * esize],
                shape: shape.to_vec(),
                strides,
                dtype,
            }
        }

        /// Override strides/shape to expose the same bytes as a strided view
        /// (e.g. a transpose without copying).
        pub fn with_view(mut self, shape: &[usize], strides: &[i64]) -> Self {
            self.shape = shape.to_vec();
            self.strides = strides.to_vec();
            self
        }

        pub fn view(&self) -> TensorView<'_> {
            TensorView::new(
                DevicePtr(self.bytes.as_ptr() as *const std::ffi::c_void),
                self.dtype,
                &self.shape,
                &self.strides,
                DeviceId::cpu(),
            )
        }

        pub fn view_mut(&mut self) -> TensorMut<'_> {
            TensorMut::new(
                DevicePtrMut(self.bytes.as_mut_ptr() as *mut std::ffi::c_void),
                self.dtype,
                &self.shape,
                &self.strides,
                DeviceId::cpu(),
            )
        }

        pub fn to_f32(&self) -> Vec<f32> {
            self.bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        pub fn to_f64(&self) -> Vec<f64> {
            self.bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }

        pub fn to_i64(&self) -> Vec<i64> {
            self.bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }

        pub fn to_i32(&self) -> Vec<i32> {
            self.bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        pub fn to_bool(&self) -> Vec<bool> {
            self.bytes.iter().map(|&b| b != 0).collect()
        }

        /// Widen an f16 buffer to f32 for comparison.
        pub fn to_f16_as_f32(&self) -> Vec<f32> {
            self.bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }

        /// The raw 16-bit patterns of an f16/bf16 buffer (to assert no
        /// f32-reinterpret corruption of NaN/inf/denormal inputs).
        pub fn to_u16_bits(&self) -> Vec<u16> {
            self.bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        }

        /// Widen a bf16 buffer to f32 for comparison.
        pub fn to_bf16_as_f32(&self) -> Vec<f32> {
            self.bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }

        pub fn to_u8(&self) -> Vec<u8> {
            self.bytes.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strided::view_in_bounds;
    use testutil::Owned;

    #[test]
    fn dense_roundtrip_contiguous() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let v = a.view();
        assert_eq!(to_dense_f32(&v).unwrap(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn dense_reads_transposed_view() {
        // Backing [2,3] row-major; expose as transposed [3,2] with strides [1,3].
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]).with_view(&[3, 2], &[1, 3]);
        let v = a.view();
        // Transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]].
        assert_eq!(to_dense_f32(&v).unwrap(), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn registry_has_all_phase1_ops() {
        let reg = build_cpu_registry();
        // Every Phase-1 op has at least one factory, and each resolves at a
        // modern opset. `Softmax` is registered twice (legacy v1 + per-axis
        // v13), and `LayerNormalization`, `FusedMatMulBias`, `FusedGemm`,
        // `FusedAttention` and the fused exact-GELU `Gelu` add contrib
        // (`com.microsoft`) entries. Standard `ai.onnx::Attention` is registered
        // at both opset 23 and 24 (two default-domain entries not in
        // `PHASE1_OPS`). The standard LLM primitives `Gelu` (opset 20),
        // `RMSNormalization` (23), `RotaryEmbedding` (23) and `Swish` (24) add
        // four more default-domain entries not in `PHASE1_OPS`; `Softmax` and
        // `LogSoftmax` each have a legacy and an opset-13 entry. Five contrib
        // (`com.microsoft`) fused transformer entries (BiasGelu, FastGelu,
        // QuickGelu, SkipLayerNormalization, SimplifiedLayerNormalization,
        // SkipSimplifiedLayerNormalization) add six more; `MoE` and
        // `GroupQueryAttention` add one contrib entry each.
        // QuantizeLinear and
        // DequantizeLinear each add six versioned entries, while
        // DynamicQuantizeLinear adds one (twenty-eight over the
        // op-name count). Pooling adds twelve more versioned entries: five each
        // for AveragePool and MaxPool, plus the two global pool operators, for
        // forty over the op-name count. ScatterElements also has distinct
        // opset-11 and opset-16 registrations. BatchNormalization,
        // InstanceNormalization and PRelu add one registration each, while
        // GroupNormalization adds opset-18 and opset-21 entries, for forty-seven
        // registrations over the Phase-1 op-name count in total.
        // MatMulNBits and GroupQueryAttention each add one more contrib-domain
        // registration.
        assert_eq!(reg.len(), PHASE1_OPS.len() + 50);
        for op in PHASE1_OPS {
            assert!(reg.lookup(op, "", 21).is_some(), "missing factory for {op}");
        }
        // Softmax selects legacy at opset ≤ 12 and per-axis at opset ≥ 13.
        assert!(reg.lookup("Softmax", "", 12).is_some());
        assert!(reg.lookup("Softmax", "", 13).is_some());
        assert!(reg.lookup("LogSoftmax", "", 12).is_some());
        assert!(reg.lookup("LogSoftmax", "", 13).is_some());
        assert!(reg.lookup("MatMulNBits", "com.microsoft", 1).is_some());
        assert!(reg.lookup("Conv", "", 21).is_none());
        assert!(
            reg.lookup("GroupQueryAttention", "com.microsoft", 1)
                .is_some()
        );
        // The fused contrib-domain LayerNormalization resolves to the same
        // kernel as the standard default-domain op.
        assert!(
            reg.lookup("LayerNormalization", "com.microsoft", 1)
                .is_some()
        );
        assert!(reg.supports("LayerNormalization", "com.microsoft"));
        assert!(reg.supports("MatMul", "ai.onnx"));
        // The `MatMul + Add` fusion's contrib op now has a CPU kernel.
        assert!(reg.supports("FusedMatMulBias", "com.microsoft"));
        // The `MatMul + Add + Relu` fusion's contrib op now has a CPU kernel.
        assert!(reg.supports("FusedGemm", "com.microsoft"));
        assert!(reg.lookup("FusedGemm", "com.microsoft", 1).is_some());
        // The exact-GELU fusion's contrib op has a CPU kernel (contrib-only).
        assert!(reg.supports("Gelu", "com.microsoft"));
        assert!(reg.supports("MoE", "com.microsoft"));
        assert!(reg.lookup("Gelu", "com.microsoft", 1).is_some());
        for op in [
            "BiasGelu",
            "FastGelu",
            "QuickGelu",
            "SkipLayerNormalization",
            "SimplifiedLayerNormalization",
            "SkipSimplifiedLayerNormalization",
        ] {
            assert!(
                reg.lookup(op, "com.microsoft", 1).is_some(),
                "missing contrib factory for {op}"
            );
        }
        // Standard `ai.onnx::Gelu` (opset 20) is now registered in the default
        // domain; it resolves at opset ≥ 20 but not below its since-version.
        assert!(reg.lookup("Gelu", "", 21).is_some());
        assert!(reg.lookup("Gelu", "", 20).is_some());
        assert!(reg.lookup("Gelu", "", 19).is_none());
        // Standard LLM primitives resolve at/after their since-versions.
        assert!(reg.lookup("RMSNormalization", "", 23).is_some());
        assert!(reg.lookup("RMSNormalization", "", 22).is_none());
        assert!(reg.lookup("BatchNormalization", "", 15).is_some());
        assert!(reg.lookup("BatchNormalization", "", 14).is_none());
        assert!(reg.lookup("InstanceNormalization", "", 6).is_some());
        assert!(reg.lookup("GroupNormalization", "", 18).is_some());
        assert!(reg.lookup("GroupNormalization", "", 21).is_some());
        assert!(reg.lookup("GroupNormalization", "", 17).is_none());
        assert!(reg.lookup("PRelu", "", 16).is_some());
        assert!(reg.lookup("PRelu", "", 15).is_none());
        assert!(reg.lookup("RotaryEmbedding", "", 23).is_some());
        assert!(reg.lookup("RotaryEmbedding", "", 22).is_none());
        assert!(reg.lookup("Swish", "", 24).is_some());
        assert!(reg.lookup("Swish", "", 23).is_none());
        // Standard ai.onnx::Attention resolves at opsets 23–26 (default domain
        // and the `ai.onnx` alias), but not below its since-version. Opset 23
        // resolves to the v23 kernel; 24/25/26 resolve to the v24 kernel.
        assert!(reg.lookup("Attention", "", 23).is_some());
        assert!(reg.lookup("Attention", "", 24).is_some());
        assert!(reg.lookup("Attention", "", 25).is_some());
        assert!(reg.lookup("Attention", "", 26).is_some());
        assert!(reg.lookup("Attention", "ai.onnx", 23).is_some());
        assert!(reg.lookup("Attention", "ai.onnx", 26).is_some());
        assert!(reg.lookup("Attention", "", 22).is_none());
        assert!(reg.supports("Attention", ""));
    }

    #[test]
    fn dense_read_stays_in_bounds() {
        let a = Owned::f32(&[3, 2], &[1., 4., 2., 5., 3., 6.]);
        let v = a.view();
        view_in_bounds(v.shape, v.strides, v.byte_offset, 4, a.bytes.len()).unwrap();
    }
}
