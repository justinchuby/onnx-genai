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
pub mod attention;
pub mod block_quant;
pub mod block_quantized_matmul;
pub mod cast;
pub mod compressed_sparse_attention;
pub mod constant;
pub mod constant_of_shape;
pub mod conv;
pub mod csa_checkpoint;
pub mod csa_device_state;
pub mod cumsum;
pub(crate) mod device_argmax;
pub mod elementwise;
mod flash_attention;
pub mod fused_gemm;
pub mod gather;
pub mod gather_block_quantized;
pub mod gemm;
mod gqa_decode;
mod gqa_decode_fp16;
pub mod group_query_attention;
pub mod index_share;
pub mod indexing;
pub mod matmul;
pub mod matmul_nbits;
pub mod movement;
pub mod normalization;
pub mod onehot;
pub mod pointwise;
pub mod pooling;
pub mod qmoe;
mod qmoe_gemm;
mod qmoe_grouping;
pub mod reduce;
pub mod rotary_embedding;
pub mod shape;
pub mod softmax;
pub mod sparse_kv_gather;
pub mod standard_attention;
pub(crate) mod standard_claims;
pub mod topk;
pub mod where_op;

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
///   fallback) plus NVRTC `ReduceMax`/`ReduceMin`; arbitrary axes and keepdims.
/// * **Pooling** — cuDNN `MaxPool`/`AveragePool` for 2-D NCHW f32/f16/bf16.
/// * **Pointwise unary math** — `Abs`, `Neg`, `Reciprocal`, `Exp`, `Log`,
///   `Sign`, `Floor`, `Ceil`, `Round`, `Sin`, `Cos`, `Softplus` (NVRTC
///   f32/f16/bf16, formulas matched to the CPU EP `unary_math.rs`).
/// * **Logical** — `Not` (bool), `And`, `Or`, `Xor` (bool, broadcasting).
/// * **Comparison** — `Equal`, `Greater`, `Less`, `GreaterOrEqual`,
///   `LessOrEqual` (f32/i32/i64 operands → bool, broadcasting; `Equal` also
///   accepts bool operands).
/// * **Movement/construction** — `Concat`, `Expand`, `Reshape`, `Slice`, `Split`,
///   `Squeeze`, `Tile`, `Transpose`, `Unsqueeze`, plus broadcasting `Where`.
///
/// See `docs/CUDA_COVERAGE.md` for the full op → backend mapping matrix and the
/// prioritised list of remaining / custom-kernel ops.
pub const CUDA_COVERED_OPS: &[&str] = &[
    "MatMul",
    "MatMulNBits",
    "QMoE",
    "BlockQuantizedMatMul",
    "SparseKvGather",
    "CompressedSparseAttention",
    "IndexShare",
    "Gemm",
    "FusedMatMulBias",
    "FusedGemm",
    "Conv",
    "MaxPool",
    "AveragePool",
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
    reg.register(
        OpKey::new("CumSum", "", 11),
        Box::new(cumsum::CumSumFactory {
            runtime: runtime.clone(),
        }),
    );
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

    // Elementwise binary (NVRTC f32/f16/bf16, NumPy broadcasting).
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
    fn covered_ops_have_no_duplicates() {
        assert_eq!(CUDA_COVERED_OPS.len(), 88);

        let mut seen = std::collections::HashSet::new();
        for op in CUDA_COVERED_OPS {
            assert!(seen.insert(*op), "duplicate op {op} in CUDA_COVERED_OPS");
        }
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
}
