//! CUDA kernels for the Phase-2a slice (`docs/ORT2.md` §15). Standard **GEMM**
//! (`MatMul`, cuBLASLt) plus the SDPA/GQA **Attention** baseline (`Attention` in
//! the `com.microsoft` domain — cuBLAS batched GEMM + NVRTC softmax). One
//! [`Kernel`] per op, keyed purely by (op type, domain) — there are **no**
//! model-specific shapes or constants anywhere in this crate (the §15.1
//! model-agnostic hard rule; attention dims are runtime data / attributes).
//!
//! ## Deferred to later slices (Phase 2b+)
//!
//! Custom fused norm/RoPE kernels, cuDNN-fused SDPA / FlashAttention-3 behind
//! the same [`attention::AttentionKernel`] binding (§13.3), paged-KV (§13.4),
//! and FP8 GEMM are **not** implemented here. Ops we don't cover are simply not
//! registered, so [`crate::CudaExecutionProvider::supports_op`] reports them
//! Unsupported and the session routes them to another EP (e.g. CPU). A direct
//! [`crate::CudaExecutionProvider::get_kernel`] for an unregistered op returns
//! an actionable [`onnx_runtime_ep_api::EpError`] — never a panic.

use std::sync::Arc;

use onnx_runtime_ep_api::{OpKey, OpRegistry};

use crate::runtime::CudaRuntime;

pub mod activations;
pub mod argreduce;
pub mod attention;
pub mod batch_normalization;
pub mod bitwise;
pub mod block_quant;
pub mod block_quantized_matmul;
pub mod block_quantized_moe;
pub mod cast;
pub mod causal_conv_with_state;
pub mod compressed_sparse_attention;
pub mod constant;
pub mod constant_of_shape;
pub mod conv;
pub mod conv_transpose;
pub mod csa_checkpoint;
pub mod csa_device_state;
pub mod cumprod;
pub mod cumsum;
pub mod data_transform;
pub(crate) mod device_argmax;
pub mod dropout;
pub mod elementwise;
mod flash_attention;
pub mod fused_gelu;
pub mod fused_gemm;
pub mod gather;
pub mod gather_block_quantized;
pub mod gemm;
pub mod global_reduction;
mod gqa_decode;
mod gqa_decode_bf16;
mod gqa_decode_fp16;
pub mod grid_sample;
pub mod group_normalization;
pub mod group_query_attention;
pub mod hardmax;
pub mod index_share;
pub mod index_transform;
pub mod indexing;
pub(crate) mod kv_stride;
pub mod linear_attention;
pub mod log_softmax;
pub mod matmul;
pub mod matmul_nbits;
pub mod mod_op;
pub mod movement;
pub mod nary;
pub mod nonzero;
pub mod normalization;
pub mod onehot;
pub mod packed_varlen_attention;
pub mod pad;
pub mod pointwise;
pub mod pooling;
pub mod prelu;
pub mod qlinear_matmul;
pub mod qmoe;
mod qmoe_gemm;
mod qmoe_grouping;
pub mod quantization;
pub mod range;
pub mod reduce;
pub mod resize;
pub mod rotary_embedding;
pub mod shape;
pub mod size;
pub mod softmax;
pub mod sparse_kv_gather;
pub mod standard_attention;
pub(crate) mod standard_claims;
pub mod structural;
pub mod topk;
pub mod trilu;
pub mod unary_predicate;
pub mod varlen_attention;
pub mod where_op;
pub mod window;

use activations::ActivationFactory;
use elementwise::{BinaryFactory, BinaryOp, StandardGeluFactory, UnaryFactory, UnaryOp};
use pointwise::{
    CmpFactory, CmpOp, LogicalFactory, LogicalOp, NotFactory, UnaryMathFactory, UnaryMathOp,
};

/// The ops the CUDA EP implements today.
///
/// * **GEMM family** — `MatMul`, `Gemm`, `FusedMatMulBias`, and `FusedGemm`
///   (cuBLASLt; the fused ops use native bias/activation epilogues).
/// * **Elementwise unary** — `Relu`, `Sqrt`, `Erf`, `Tanh` (+ `Sigmoid`),
///   standard and `com.microsoft` `Gelu`, and `com.microsoft` `Silu`, via
///   runtime-compiled NVRTC kernels (`Silu` matches the CPU EP's f32 coverage;
///   the others support f32/f16/bf16).
/// * **Elementwise binary (NumPy broadcasting)** — `Add`, `Sub`, `Mul`, `Div`,
///   `Pow`, `Min`, `Max`, via f32/f16/bf16 NVRTC kernels.
/// * **Attention** — the SDPA/GQA baseline (`com.microsoft` domain; cuBLAS
///   batched GEMM + NVRTC softmax), the §13.3 binding a cuDNN-fused SDPA /
///   FlashAttention-3 shim drops in behind.
/// * **Softmax** — cuDNN `cudnnSoftmaxForward` (f32/f16/bf16; legacy
///   coerce-to-2D at opset ≤ 12, per-axis at opset ≥ 13), with f32 NVRTC fallback.
/// * **Normalization** — fused NVRTC `LayerNormalization` (ai.onnx +
///   `com.microsoft`), `RMSNormalization` / `SimplifiedLayerNormalization`, and
///   `SkipLayerNormalization` and `SkipSimplifiedLayerNormalization` (residual add fused into the norm).
/// * **Cast / CastLike** — NVRTC element-wise dtype conversion (f32/f64/f16/bf16/
///   int8-64/uint8-64/bool).
/// * **Reductions** — cuDNN `ReduceSum`/`ReduceMean` (f32/f16/bf16, f32 NVRTC
///   fallback) plus NVRTC `ReduceMax`/`ReduceMin`, and the f32 NVRTC family
///   `ReduceProd`, `ReduceSumSquare`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`,
///   `ReduceLogSumExp`; arbitrary axes and keepdims.
/// * **Pooling** — cuDNN `MaxPool`/`AveragePool` for 2-D NCHW f32/f16/bf16.
/// * **Pointwise unary math** — `Abs`, `Neg`, `Reciprocal`, `Exp`, `Log`,
///   `Sign`, `Floor`, `Ceil`, `Round`, `Sin`, `Cos`, `Softplus`, and the
///   trigonometric/hyperbolic family `Tan`, `Sinh`, `Cosh`, `Asin`, `Acos`,
///   `Atan`, `Asinh`, `Acosh`, `Atanh` (NVRTC f32/f16/bf16, formulas matched to
///   the CPU EP `unary_math.rs`).
/// * **Logical** — `Not` (bool), `And`, `Or`, `Xor` (bool, broadcasting).
/// * **Comparison** — `Equal`, `Greater`, `Less`, `GreaterOrEqual`,
///   `LessOrEqual` (f32/i32/i64 operands → bool, broadcasting; `Equal` also
///   accepts bool operands).
/// * **Movement/construction** — `Concat`, `Expand`, `Pad`, `Range`, `Reshape`, `Slice`, `Split`,
///   `Squeeze`, `Tile`, `Transpose`, `Unsqueeze`, `Identity`, `Flatten`, plus
///   broadcasting `Where` and triangular-mask `Trilu`.
/// * **Metadata** — `Shape`, `Size` (host-computed Int64, uploaded to device).
/// * **Activations (extended)** — `Swish` (opset 24) and `ThresholdedRelu`
///   (opset 10), attribute-driven f32/f16/bf16.
/// * **Variadic elementwise** — `Sum` and `Mean` (f32/f16/bf16, NumPy
///   broadcasting across a variadic input list).
/// * **Modulo** — `Mod` (f32 with `fmod=1`, plus i32/i64 truncated and
///   floor modulo).
/// * **Bitwise** — `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `BitwiseNot`
///   (all integer dtypes, broadcasting) and unsigned `BitShift`
///   (LEFT/RIGHT), matched to the CPU EP `bitwise.rs`/`bitshift.rs`.
/// * **LogSoftmax / Hardmax** — numerically-stable axis-reduction ops
///   (f32/f16/bf16), matched to the CPU EP; `LogSoftmax` uses the stable
///   shifted-logsumexp formulation.
/// * **Fused GELU (`com.microsoft`)** — `BiasGelu` (exact GELU of `X+bias`),
///   `FastGelu` (tanh GELU of `X`+ optional bias), and `QuickGelu`
///   (`X·sigmoid(alpha·X)`), f32/f16/bf16, matched to the CPU EP
///   `contrib_fused.rs` (GELU evaluated in `double`).
/// * **CumProd** — deterministic per-lane cumulative product (f32/i64) with the
///   `exclusive`/`reverse` attributes, mirroring the `CumSum` scan.
/// * **ArgMax / ArgMin** — axis reduction to `Int64` indices (f32/f16/bf16)
///   honouring `keepdims` and `select_last_index`, matched to the CPU EP
///   `selection.rs` tie-breaking.
///
/// See `docs/CUDA_COVERAGE.md` for the full op → backend mapping matrix and the
/// prioritised list of remaining / custom-kernel ops.
pub const CUDA_COVERED_OPS: &[&str] = &[
    "MatMul",
    "MatMulNBits",
    "QMoE",
    "BlockQuantizedMatMul",
    "BlockQuantizedMoE",
    "SparseKvGather",
    "CompressedSparseAttention",
    "IndexShare",
    "PackedVarlenAttention",
    "VarlenAttention",
    "Gemm",
    "FusedMatMulBias",
    "FusedGemm",
    "Conv",
    "MaxPool",
    "AveragePool",
    "LpPool",
    "Relu",
    "Sqrt",
    "Erf",
    "Tanh",
    "Sigmoid",
    "Gelu",
    "Silu",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Min",
    "Max",
    "Attention",
    "GroupQueryAttention",
    "RotaryEmbedding",
    "Softmax",
    "LayerNormalization",
    "SkipLayerNormalization",
    "SkipSimplifiedLayerNormalization",
    "SimplifiedLayerNormalization",
    "RMSNormalization",
    "Cast",
    "CastLike",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
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
    "Softplus",
    "Tan",
    "Sinh",
    "Cosh",
    "Asin",
    "Acos",
    "Atan",
    "Asinh",
    "Acosh",
    "Atanh",
    "Not",
    "And",
    "Or",
    "Xor",
    "Equal",
    "Greater",
    "Less",
    "GreaterOrEqual",
    "LessOrEqual",
    "LeakyRelu",
    "Elu",
    "HardSigmoid",
    "Clip",
    "Softsign",
    "Selu",
    "Gather",
    "Shape",
    "Constant",
    "ConstantOfShape",
    "Concat",
    "Expand",
    "Reshape",
    "Slice",
    "Split",
    "Squeeze",
    "Tile",
    "Transpose",
    "Unsqueeze",
    "Where",
    "TopK",
    "CumSum",
    "GatherElements",
    "ScatterElements",
    "OneHot",
    "Identity",
    "Flatten",
    "Size",
    "Trilu",
    "Swish",
    "ThresholdedRelu",
    "Sum",
    "Mean",
    "Mod",
    "IsInf",
    "IsNaN",
    "PRelu",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BitwiseNot",
    "BitShift",
    "LogSoftmax",
    "Hardmax",
    "BiasGelu",
    "FastGelu",
    "QuickGelu",
    "CumProd",
    "ArgMax",
    "ArgMin",
    "GatherND",
    "SpaceToDepth",
    "EyeLike",
    "Pad",
    "Range",
    "ScatterND",
    "HannWindow",
    "HammingWindow",
    "BlackmanWindow",
    "QuantizeLinear",
    "DequantizeLinear",
    "Dropout",
    "NonZero",
    "AffineGrid",
    "BatchNormalization",
    "Compress",
    "DynamicQuantizeLinear",
    "GlobalAveragePool",
    "GlobalLpPool",
    "GlobalMaxPool",
    "LpNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "CenterCropPad",
    "Col2Im",
    "QLinearMatMul",
    "Resize",
    "ConvTranspose",
    "GridSample",
    "GatherBlockQuantized",
    "CausalConvWithState",
    "LinearAttention",
];

/// Build an [`OpRegistry`] populated with the CUDA kernel factories.
///
/// The shared [`CudaRuntime`] (context + stream + cuBLASLt handle) is threaded
/// into every factory so kernels submit onto the EP's single stream.
pub fn build_cuda_registry(runtime: Arc<CudaRuntime>) -> OpRegistry {
    build_cuda_registry_with_metrics(runtime, Arc::new(csa_checkpoint::CsaMetrics::default()))
}

/// Like [`build_cuda_registry`] but threads a shared [`CsaMetrics`] telemetry
/// surface (§8) into the CSA factory so the owning EP can read per-layer CSA
/// observability after execution.
///
/// [`CsaMetrics`]: csa_checkpoint::CsaMetrics
pub fn build_cuda_registry_with_metrics(
    runtime: Arc<CudaRuntime>,
    csa_metrics: Arc<csa_checkpoint::CsaMetrics>,
) -> OpRegistry {
    let mut reg = OpRegistry::new();

    // GEMM family (cuBLASLt).
    reg.register(
        OpKey::new("MatMul", "", 1),
        Box::new(matmul::MatMulFactory {
            runtime: runtime.clone(),
        }),
    );
    // Metadata / indexed data movement. Shape and Constant construct their small
    // results on the host and upload them; Gather is an NVRTC indexed-copy kernel.
    reg.register(
        OpKey::new("Gather", "", 1),
        Box::new(gather::GatherFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("GatherElements", "", 11),
        Box::new(indexing::GatherElementsFactory {
            runtime: runtime.clone(),
        }),
    );
    for opset in [11, 16] {
        reg.register(
            OpKey::new("ScatterElements", "", opset),
            Box::new(indexing::ScatterElementsFactory {
                runtime: runtime.clone(),
            }),
        );
    }
    for opset in [11, 16, 18] {
        reg.register(
            OpKey::new("ScatterND", "", opset),
            Box::new(indexing::ScatterNdFactory {
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("CumSum", "", 11),
        Box::new(cumsum::CumSumFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("CumProd", "", 26),
        Box::new(cumprod::CumProdFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ArgMax", "", 1),
        Box::new(argreduce::ArgReduceFactory {
            op: argreduce::ArgOp::Max,
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ArgMin", "", 1),
        Box::new(argreduce::ArgReduceFactory {
            op: argreduce::ArgOp::Min,
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("GatherND", "", 11),
        Box::new(structural::GatherNdFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("SpaceToDepth", "", 13),
        Box::new(structural::SpaceToDepthFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("EyeLike", "", 9),
        Box::new(structural::EyeLikeFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("AffineGrid", "", 20),
        Box::new(data_transform::AffineGridFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Compress", "", 11),
        Box::new(data_transform::CompressFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("BatchNormalization", "", 7),
        Box::new(batch_normalization::BatchNormalizationFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("LpNormalization", "", 1),
        Box::new(global_reduction::LpNormalizationFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("InstanceNormalization", "", 6),
        Box::new(group_normalization::InstanceNormalizationFactory {
            runtime: runtime.clone(),
        }),
    );
    for since_version in [18, 21] {
        reg.register(
            OpKey::new("GroupNormalization", "", since_version),
            Box::new(group_normalization::GroupNormalizationFactory {
                runtime: runtime.clone(),
                since_version,
            }),
        );
    }
    for (op, kind) in [
        (
            "GlobalAveragePool",
            global_reduction::GlobalPoolKind::Average,
        ),
        ("GlobalMaxPool", global_reduction::GlobalPoolKind::Max),
    ] {
        reg.register(
            OpKey::new(op, "", 1),
            Box::new(global_reduction::GlobalPoolFactory {
                kind,
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("GlobalLpPool", "", 2),
        Box::new(global_reduction::GlobalPoolFactory {
            kind: global_reduction::GlobalPoolKind::Lp(2),
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Pad", "", 1),
        Box::new(pad::PadFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Range", "", 11),
        Box::new(range::RangeFactory {
            runtime: runtime.clone(),
        }),
    );
    for version in [10, 13, 19, 21, 23, 25] {
        reg.register(
            OpKey::new("QuantizeLinear", "", version),
            Box::new(quantization::LinearQuantFactory {
                op: quantization::LinearQuantOp::Quantize,
                runtime: runtime.clone(),
            }),
        );
        reg.register(
            OpKey::new("DequantizeLinear", "", version),
            Box::new(quantization::LinearQuantFactory {
                op: quantization::LinearQuantOp::Dequantize,
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("DynamicQuantizeLinear", "", 11),
        Box::new(quantization::DynamicQuantizeLinearFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("QLinearMatMul", "", 10),
        Box::new(qlinear_matmul::QLinearMatMulFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Resize", "", 10),
        Box::new(resize::ResizeFactory {
            runtime: runtime.clone(),
            since_version: 10,
        }),
    );
    reg.register(
        OpKey::new("Resize", "", 11),
        Box::new(resize::ResizeFactory {
            runtime: runtime.clone(),
            since_version: 11,
        }),
    );
    reg.register(
        OpKey::new("ConvTranspose", "", 1),
        Box::new(conv_transpose::ConvTransposeFactory {
            runtime: runtime.clone(),
        }),
    );
    for since_version in [16_u32, 20] {
        reg.register(
            OpKey::new("GridSample", "", u64::from(since_version)),
            Box::new(grid_sample::GridSampleFactory {
                runtime: runtime.clone(),
                since_version,
            }),
        );
    }
    for version in [13, 22] {
        reg.register(
            OpKey::new("Dropout", "", version),
            Box::new(dropout::DropoutFactory {
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("NonZero", "", 9),
        Box::new(nonzero::NonZeroFactory {
            runtime: runtime.clone(),
        }),
    );
    for (op_type, kind) in [
        ("HannWindow", window::WindowKind::Hann),
        ("HammingWindow", window::WindowKind::Hamming),
        ("BlackmanWindow", window::WindowKind::Blackman),
    ] {
        reg.register(
            OpKey::new(op_type, "", 17),
            Box::new(window::WindowFactory {
                kind,
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("TopK", "", 10),
        Box::new(topk::TopKFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Shape", "", 1),
        Box::new(shape::ShapeFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Size", "", 1),
        Box::new(size::SizeFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Trilu", "", 14),
        Box::new(trilu::TriluFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Constant", "", 1),
        Box::new(constant::ConstantFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ConstantOfShape", "", 9),
        Box::new(constant_of_shape::ConstantOfShapeFactory {
            runtime: runtime.clone(),
        }),
    );
    for (opset, wrap_negative) in [(9, false), (11, true)] {
        reg.register(
            OpKey::new("OneHot", "", opset),
            Box::new(onehot::OneHotFactory {
                runtime: runtime.clone(),
                wrap_negative,
            }),
        );
    }
    for (op_type, factory) in [
        (
            "Concat",
            Box::new(movement::ConcatFactory {
                runtime: runtime.clone(),
            }) as Box<dyn onnx_runtime_ep_api::KernelFactory>,
        ),
        (
            "Expand",
            Box::new(movement::ExpandFactory {
                runtime: runtime.clone(),
            }),
        ),
        (
            "Identity",
            Box::new(movement::IdentityFactory {
                runtime: runtime.clone(),
            }),
        ),
        (
            "Flatten",
            Box::new(movement::FlattenFactory {
                runtime: runtime.clone(),
            }),
        ),
        (
            "Reshape",
            Box::new(movement::ReshapeFactory {
                runtime: runtime.clone(),
            }),
        ),
        (
            "Slice",
            Box::new(movement::SliceFactory {
                runtime: runtime.clone(),
            }),
        ),
        (
            "Split",
            Box::new(movement::SplitFactory {
                runtime: runtime.clone(),
            }),
        ),
        (
            "Squeeze",
            Box::new(movement::SqueezeFactory {
                runtime: runtime.clone(),
            }),
        ),
        (
            "Transpose",
            Box::new(movement::TransposeFactory {
                runtime: runtime.clone(),
            }),
        ),
        (
            "Unsqueeze",
            Box::new(movement::UnsqueezeFactory {
                runtime: runtime.clone(),
            }),
        ),
        (
            "Where",
            Box::new(where_op::WhereFactory {
                runtime: runtime.clone(),
            }),
        ),
    ] {
        reg.register(OpKey::new(op_type, "", 1), factory);
    }
    reg.register(
        OpKey::new("Tile", "", 6),
        Box::new(movement::TileFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("MatMulNBits", "com.microsoft", 1),
        Box::new(matmul_nbits::MatMulNBitsFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("GatherBlockQuantized", "com.microsoft", 1),
        Box::new(gather_block_quantized::GatherBlockQuantizedFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("CausalConvWithState", "com.microsoft", 1),
        Box::new(causal_conv_with_state::CausalConvWithStateFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("LinearAttention", "com.microsoft", 1),
        Box::new(linear_attention::LinearAttentionFactory {
            runtime: runtime.clone(),
        }),
    );
    // Standard ONNX-domain spelling (onnx/onnx#7689), semantically identical to
    // the com.microsoft op — served by the same fused kernel.
    reg.register(
        OpKey::new("LinearAttention", "", 1),
        Box::new(linear_attention::LinearAttentionFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("QMoE", "com.microsoft", 1),
        Box::new(qmoe::QMoEFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("BlockQuantizedMatMul", "pkg.nxrt", 1),
        Box::new(block_quantized_matmul::BlockQuantizedMatMulFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("BlockQuantizedMoE", "pkg.nxrt", 1),
        Box::new(block_quantized_moe::BlockQuantizedMoEFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("SparseKvGather", "pkg.nxrt", 1),
        Box::new(sparse_kv_gather::SparseKvGatherFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("CompressedSparseAttention", "pkg.nxrt", 1),
        Box::new(
            compressed_sparse_attention::CompressedSparseAttentionFactory {
                runtime: runtime.clone(),
                metrics: csa_metrics.clone(),
            },
        ),
    );
    reg.register(
        OpKey::new("IndexShare", "pkg.nxrt", 1),
        Box::new(index_share::IndexShareFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("PackedVarlenAttention", "pkg.nxrt", 1),
        Box::new(packed_varlen_attention::PackedVarlenAttentionFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("VarlenAttention", "pkg.nxrt", 1),
        Box::new(varlen_attention::VarlenAttentionFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Gemm", "", 1),
        Box::new(gemm::GemmFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("FusedMatMulBias", "com.microsoft", 1),
        Box::new(fused_gemm::FusedMatMulBiasFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("FusedGemm", "com.microsoft", 1),
        Box::new(fused_gemm::FusedGemmFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Conv", "", 1),
        Box::new(conv::ConvFactory {
            runtime: runtime.clone(),
        }),
    );
    for (op_type, kind) in [
        ("MaxPool", pooling::PoolKind::Max),
        ("AveragePool", pooling::PoolKind::Average),
    ] {
        reg.register(
            OpKey::new(op_type, "", 1),
            Box::new(pooling::PoolFactory {
                kind,
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("LpPool", "", 18),
        Box::new(pooling::LpPoolFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("CenterCropPad", "", 18),
        Box::new(index_transform::CenterCropPadFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Col2Im", "", 18),
        Box::new(index_transform::Col2ImFactory {
            runtime: runtime.clone(),
        }),
    );

    // Elementwise unary activations (NVRTC pointwise). The loop includes the
    // contrib Gelu/Silu forms; standard Gelu is registered separately below so
    // its `approximate` attribute can select exact-erf or tanh semantics.
    for (op_type, domain, op) in [
        ("Relu", "", UnaryOp::Relu),
        ("Sqrt", "", UnaryOp::Sqrt),
        ("Erf", "", UnaryOp::Erf),
        ("Tanh", "", UnaryOp::Tanh),
        ("Sigmoid", "", UnaryOp::Sigmoid),
        ("Gelu", "com.microsoft", UnaryOp::Gelu),
        ("Silu", "com.microsoft", UnaryOp::Silu),
    ] {
        reg.register(
            OpKey::new(op_type, domain, 1),
            Box::new(UnaryFactory {
                op,
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("Gelu", "", 20),
        Box::new(StandardGeluFactory {
            runtime: runtime.clone(),
        }),
    );

    // Fused GELU activations (`com.microsoft`): BiasGelu / FastGelu add a
    // broadcast bias before an exact / tanh GELU; QuickGelu is x*sigmoid(alpha*x).
    for (op_type, op) in [
        ("BiasGelu", fused_gelu::FusedGeluOp::Bias),
        ("FastGelu", fused_gelu::FusedGeluOp::Fast),
        ("QuickGelu", fused_gelu::FusedGeluOp::Quick),
    ] {
        reg.register(
            OpKey::new(op_type, "com.microsoft", 1),
            Box::new(fused_gelu::FusedGeluFactory {
                op,
                runtime: runtime.clone(),
            }),
        );
    }

    // CUDA Wave 4 — attribute-driven f32/f16/bf16 activations.
    for op_type in [
        "LeakyRelu",
        "Elu",
        "HardSigmoid",
        "Clip",
        "Softsign",
        "Selu",
    ] {
        reg.register(
            OpKey::new(op_type, "", 1),
            Box::new(ActivationFactory {
                name: op_type,
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("ThresholdedRelu", "", 10),
        Box::new(ActivationFactory {
            name: "ThresholdedRelu",
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Swish", "", 24),
        Box::new(ActivationFactory {
            name: "Swish",
            runtime: runtime.clone(),
        }),
    );
    for (op_type, op) in [
        ("Add", BinaryOp::Add),
        ("Sub", BinaryOp::Sub),
        ("Mul", BinaryOp::Mul),
        ("Div", BinaryOp::Div),
        ("Pow", BinaryOp::Pow),
        ("Min", BinaryOp::Min),
        ("Max", BinaryOp::Max),
    ] {
        reg.register(
            OpKey::new(op_type, "", 1),
            Box::new(BinaryFactory {
                op,
                runtime: runtime.clone(),
            }),
        );
    }

    // Attention (Phase-2b fused prefill with a Phase-2a fallback).
    reg.register(
        OpKey::new("Attention", "com.microsoft", 1),
        Box::new(attention::AttentionFactory {
            runtime: runtime.clone(),
        }),
    );
    for opset in [23, 24] {
        reg.register(
            OpKey::new("Attention", "", opset),
            Box::new(standard_attention::StandardAttentionFactory {
                runtime: runtime.clone(),
                since_version: opset as u32,
            }),
        );
    }
    reg.register(
        OpKey::new("RotaryEmbedding", "", 23),
        Box::new(rotary_embedding::RotaryEmbeddingFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("RotaryEmbedding", "com.microsoft", 1),
        Box::new(rotary_embedding::RotaryEmbeddingContribFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("GroupQueryAttention", "com.microsoft", 1),
        Box::new(group_query_attention::GroupQueryAttentionFactory {
            runtime: runtime.clone(),
        }),
    );

    // ── CUDA Wave 2 — transformer-critical ops (see docs/CUDA_COVERAGE.md) ──

    // Softmax (cuDNN, with f32 NVRTC fallback). Legacy coerce-to-2D at opset
    // ≤ 12, per-axis at opset ≥ 13.
    reg.register(
        OpKey::new("Softmax", "", 1),
        Box::new(softmax::SoftmaxLegacyFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Softmax", "", 13),
        Box::new(softmax::SoftmaxFactory {
            runtime: runtime.clone(),
        }),
    );

    // LayerNormalization (fused NVRTC). Standard domain + the optimizer's
    // `com.microsoft` fused form share identical semantics.
    for domain in ["", "com.microsoft"] {
        reg.register(
            OpKey::new("LayerNormalization", domain, 1),
            Box::new(normalization::LayerNormFactory {
                runtime: runtime.clone(),
            }),
        );
    }

    // RMSNormalization (fused NVRTC, no mean subtraction): both CPU-registered
    // SimplifiedLayerNormalization domains and ai.onnx RMSNormalization share
    // the same computation.
    for domain in ["", "com.microsoft"] {
        reg.register(
            OpKey::new("SimplifiedLayerNormalization", domain, 1),
            Box::new(normalization::RmsNormFactory {
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("RMSNormalization", "", 1),
        Box::new(normalization::RmsNormFactory {
            runtime: runtime.clone(),
        }),
    );

    // SkipLayerNormalization (fused residual add + layernorm).
    reg.register(
        OpKey::new("SkipLayerNormalization", "com.microsoft", 1),
        Box::new(normalization::SkipLayerNormFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("SkipSimplifiedLayerNormalization", "com.microsoft", 1),
        Box::new(normalization::SkipSimplifiedLayerNormFactory {
            runtime: runtime.clone(),
        }),
    );

    // Cast / CastLike (NVRTC element-wise dtype conversion).
    for op_type in ["Cast", "CastLike"] {
        reg.register(
            OpKey::new(op_type, "", 1),
            Box::new(cast::CastFactory {
                runtime: runtime.clone(),
            }),
        );
    }

    // Reductions (sum/mean cuDNN with f32 NVRTC fallback; max/min NVRTC).
    reg.register(
        OpKey::new("ReduceSum", "", 1),
        Box::new(reduce::ReduceSumFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ReduceMean", "", 1),
        Box::new(reduce::ReduceMeanFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ReduceMax", "", 1),
        Box::new(reduce::ReduceMaxFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ReduceMin", "", 1),
        Box::new(reduce::ReduceMinFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ReduceProd", "", 1),
        Box::new(reduce::ReduceProdFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ReduceSumSquare", "", 1),
        Box::new(reduce::ReduceSumSquareFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ReduceL1", "", 1),
        Box::new(reduce::ReduceL1Factory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ReduceL2", "", 1),
        Box::new(reduce::ReduceL2Factory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ReduceLogSum", "", 1),
        Box::new(reduce::ReduceLogSumFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("ReduceLogSumExp", "", 1),
        Box::new(reduce::ReduceLogSumExpFactory {
            runtime: runtime.clone(),
        }),
    );

    // Variadic elementwise (nary.rs) — Sum/Mean, f32/f16/bf16 broadcasting.
    reg.register(
        OpKey::new("Sum", "", 1),
        Box::new(nary::NaryFactory {
            is_mean: false,
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Mean", "", 1),
        Box::new(nary::NaryFactory {
            is_mean: true,
            runtime: runtime.clone(),
        }),
    );

    // Mod (mod_op.rs) — f32 (fmod=1) plus i32/i64 truncated and floor modulo.
    reg.register(
        OpKey::new("Mod", "", 10),
        Box::new(mod_op::ModFactory {
            runtime: runtime.clone(),
        }),
    );

    // ── CUDA Wave 3 — pointwise math / logical / comparison (pointwise.rs) ──

    // Pointwise unary math (NVRTC f32/f16/bf16; formulas matched to the CPU EP
    // `unary_math.rs`). Standard domain, single input/output, equal shape.
    for (op_type, op) in [
        ("Abs", UnaryMathOp::Abs),
        ("Neg", UnaryMathOp::Neg),
        ("Reciprocal", UnaryMathOp::Reciprocal),
        ("Exp", UnaryMathOp::Exp),
        ("Log", UnaryMathOp::Log),
        ("Sign", UnaryMathOp::Sign),
        ("Floor", UnaryMathOp::Floor),
        ("Ceil", UnaryMathOp::Ceil),
        ("Round", UnaryMathOp::Round),
        ("Sin", UnaryMathOp::Sin),
        ("Cos", UnaryMathOp::Cos),
        ("Softplus", UnaryMathOp::Softplus),
        ("Tan", UnaryMathOp::Tan),
        ("Sinh", UnaryMathOp::Sinh),
        ("Cosh", UnaryMathOp::Cosh),
        ("Asin", UnaryMathOp::Asin),
        ("Acos", UnaryMathOp::Acos),
        ("Atan", UnaryMathOp::Atan),
        ("Asinh", UnaryMathOp::Asinh),
        ("Acosh", UnaryMathOp::Acosh),
        ("Atanh", UnaryMathOp::Atanh),
    ] {
        reg.register(
            OpKey::new(op_type, "", 1),
            Box::new(UnaryMathFactory {
                op,
                runtime: runtime.clone(),
            }),
        );
    }

    // Logical `Not` (bool → bool; matched to the CPU EP `logical.rs`).
    reg.register(
        OpKey::new("Not", "", 1),
        Box::new(NotFactory {
            runtime: runtime.clone(),
        }),
    );

    // Logical binary (bool operands → bool; NumPy broadcasting).
    for (op_type, op) in [
        ("And", LogicalOp::And),
        ("Or", LogicalOp::Or),
        ("Xor", LogicalOp::Xor),
    ] {
        reg.register(
            OpKey::new(op_type, "", 1),
            Box::new(LogicalFactory {
                op,
                runtime: runtime.clone(),
            }),
        );
    }

    // Comparison (f32 operands → bool; NumPy broadcasting).
    for (op_type, op) in [
        ("Equal", CmpOp::Equal),
        ("Greater", CmpOp::Greater),
        ("Less", CmpOp::Less),
        ("GreaterOrEqual", CmpOp::GreaterOrEqual),
        ("LessOrEqual", CmpOp::LessOrEqual),
    ] {
        reg.register(
            OpKey::new(op_type, "", 1),
            Box::new(CmpFactory {
                op,
                runtime: runtime.clone(),
            }),
        );
    }

    // ── CUDA op-coverage batch 3 (issue #67) ──

    // Unary float predicates (f32/f16/bf16 → bool). `IsInf` honours the
    // detect_positive/detect_negative attributes; formulas mirror the CPU EP
    // (`is_inf.rs`/`is_nan.rs`).
    reg.register(
        OpKey::new("IsInf", "", 10),
        Box::new(unary_predicate::PredicateFactory {
            op: unary_predicate::PredicateOp::IsInf,
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("IsNaN", "", 9),
        Box::new(unary_predicate::PredicateFactory {
            op: unary_predicate::PredicateOp::IsNaN,
            runtime: runtime.clone(),
        }),
    );

    // PRelu (f32/f16/bf16, NumPy-broadcastable slope; matches the CPU EP
    // `norm_ops.rs::prelu_typed`).
    reg.register(
        OpKey::new("PRelu", "", 16),
        Box::new(prelu::PReluFactory {
            runtime: runtime.clone(),
        }),
    );

    // ── CUDA op-coverage batch 4 (issue #67) ──

    // Integer bitwise binary (same-dtype operands → same-dtype output; NumPy
    // broadcasting). Matches the CPU EP `bitwise.rs` across every integer dtype.
    for (op_type, op) in [
        ("BitwiseAnd", bitwise::BitwiseBinaryOp::And),
        ("BitwiseOr", bitwise::BitwiseBinaryOp::Or),
        ("BitwiseXor", bitwise::BitwiseBinaryOp::Xor),
    ] {
        reg.register(
            OpKey::new(op_type, "", 18),
            Box::new(bitwise::BitwiseBinaryFactory {
                op,
                runtime: runtime.clone(),
            }),
        );
    }
    reg.register(
        OpKey::new("BitwiseNot", "", 18),
        Box::new(bitwise::BitwiseNotFactory {
            runtime: runtime.clone(),
        }),
    );
    // Unsigned BitShift (LEFT/RIGHT via the `direction` attribute); matches the
    // CPU EP `bitshift.rs` checked-shift contract.
    reg.register(
        OpKey::new("BitShift", "", 11),
        Box::new(bitwise::BitShiftFactory {
            runtime: runtime.clone(),
        }),
    );

    // LogSoftmax (numerically-stable shifted-logsumexp NVRTC kernel). Shares
    // Softmax's opset split: legacy coerce-to-2D at opset ≤ 12, per-axis at
    // opset ≥ 13.
    reg.register(
        OpKey::new("LogSoftmax", "", 1),
        Box::new(log_softmax::LogSoftmaxLegacyFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("LogSoftmax", "", 13),
        Box::new(log_softmax::LogSoftmaxFactory {
            runtime: runtime.clone(),
        }),
    );

    // Hardmax (first-argmax one-hot along `axis`, opset ≥ 13 semantics).
    reg.register(
        OpKey::new("Hardmax", "", 13),
        Box::new(hardmax::HardmaxFactory {
            runtime: runtime.clone(),
        }),
    );

    reg
}

#[cfg(test)]
mod tests {
    use super::CUDA_COVERED_OPS;

    #[test]
    fn wave2_ops_are_listed_in_coverage() {
        for op in [
            "Softmax",
            "LayerNormalization",
            "SkipLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "SimplifiedLayerNormalization",
            "RMSNormalization",
            "Cast",
            "CastLike",
            "ReduceSum",
            "ReduceMean",
            "ReduceMax",
            "ReduceMin",
        ] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn coverage_batch3_ops_are_listed_in_coverage() {
        for op in ["IsInf", "IsNaN", "PRelu"] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn coverage_batch4_ops_are_listed_in_coverage() {
        for op in [
            "BitwiseAnd",
            "BitwiseOr",
            "BitwiseXor",
            "BitwiseNot",
            "BitShift",
            "LogSoftmax",
            "Hardmax",
        ] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn covered_ops_have_no_duplicates() {
        let unique_ops = CUDA_COVERED_OPS
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            CUDA_COVERED_OPS.len(),
            unique_ops.len(),
            "CUDA_COVERED_OPS contains duplicate entries"
        );
    }

    #[test]
    fn indexing_and_scan_ops_are_listed_in_coverage() {
        for op in [
            "TopK",
            "CumSum",
            "GatherElements",
            "ScatterElements",
            "OneHot",
        ] {
            assert!(CUDA_COVERED_OPS.contains(&op));
        }
    }

    #[test]
    fn group_query_attention_is_listed_in_coverage() {
        assert!(CUDA_COVERED_OPS.contains(&"GroupQueryAttention"));
    }

    #[test]
    fn coverage_batch2_ops_are_listed_in_coverage() {
        for op in [
            "ReduceProd",
            "ReduceSumSquare",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "Swish",
            "ThresholdedRelu",
            "Sum",
            "Mean",
            "Mod",
        ] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn wave3_pointwise_ops_are_listed_in_coverage() {
        for op in [
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
            "Softplus",
            "Not",
            "And",
            "Or",
            "Xor",
            "Equal",
            "Greater",
            "Less",
            "GreaterOrEqual",
            "LessOrEqual",
        ] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn movement_and_where_ops_are_listed_in_coverage() {
        for op in [
            "Concat",
            "Expand",
            "Reshape",
            "Slice",
            "Split",
            "Squeeze",
            "Tile",
            "Transpose",
            "Unsqueeze",
            "Where",
        ] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn wave4_activations_are_listed_in_coverage() {
        for op in [
            "LeakyRelu",
            "Elu",
            "HardSigmoid",
            "Clip",
            "Softsign",
            "Selu",
        ] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn cudnn_pooling_ops_are_listed_in_coverage() {
        for op in ["MaxPool", "AveragePool"] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn fused_epilogue_ops_are_listed_in_coverage() {
        for op in ["FusedMatMulBias", "FusedGemm"] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn standard_attention_and_rope_are_listed_in_coverage() {
        for op in ["Attention", "RotaryEmbedding"] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn trig_hyperbolic_unary_ops_are_listed_in_coverage() {
        for op in [
            "Tan", "Sinh", "Cosh", "Asin", "Acos", "Atan", "Asinh", "Acosh", "Atanh",
        ] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }

    #[test]
    fn shape_movement_ops_are_listed_in_coverage() {
        for op in ["Identity", "Flatten", "Size", "Trilu"] {
            assert!(
                CUDA_COVERED_OPS.contains(&op),
                "{op} missing from CUDA_COVERED_OPS"
            );
        }
    }
}
