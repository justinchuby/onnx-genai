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
pub mod cast;
pub mod conv;
pub mod elementwise;
pub mod gemm;
pub mod matmul;
pub mod normalization;
pub mod pointwise;
pub mod pooling;
pub mod reduce;
pub mod softmax;

use activations::ActivationFactory;
use elementwise::{BinaryFactory, BinaryOp, UnaryFactory, UnaryOp};
use pointwise::{
    CmpFactory, CmpOp, LogicalFactory, LogicalOp, NotFactory, UnaryMathFactory, UnaryMathOp,
};

/// The ops the CUDA EP implements today.
///
/// * **GEMM family** — `MatMul` and `Gemm` (cuBLASLt; `Gemm` adds a fused NVRTC
///   `beta·C` broadcast-bias epilogue).
/// * **Elementwise unary** — `Relu`, `Sqrt`, `Erf`, `Tanh` (+ `Sigmoid`) and the
///   `com.microsoft` `Gelu`, via runtime-compiled f32/f16/bf16 NVRTC kernels.
/// * **Elementwise binary (NumPy broadcasting)** — `Add`, `Sub`, `Mul`, `Div`,
///   `Pow`, `Min`, `Max`, via f32/f16/bf16 NVRTC kernels.
/// * **Attention** — the SDPA/GQA baseline (`com.microsoft` domain; cuBLAS
///   batched GEMM + NVRTC softmax), the §13.3 binding a cuDNN-fused SDPA /
///   FlashAttention-3 shim drops in behind.
/// * **Softmax** — cuDNN `cudnnSoftmaxForward` (f32/f16/bf16; legacy
///   coerce-to-2D at opset ≤ 12, per-axis at opset ≥ 13), with f32 NVRTC fallback.
/// * **Normalization** — fused NVRTC `LayerNormalization` (ai.onnx +
///   `com.microsoft`), `RMSNormalization` / `SimplifiedLayerNormalization`, and
///   `SkipLayerNormalization` (residual add fused into the norm).
/// * **Cast / CastLike** — NVRTC element-wise dtype conversion (f32/f64/f16/bf16/
///   int8-64/uint8-64/bool).
/// * **Reductions** — cuDNN `ReduceSum`/`ReduceMean` (f32/f16/bf16, f32 NVRTC
///   fallback) plus NVRTC `ReduceMax`/`ReduceMin`; arbitrary axes and keepdims.
/// * **Pooling** — cuDNN `MaxPool`/`AveragePool` for 2-D NCHW f32/f16/bf16.
/// * **Pointwise unary math** — `Abs`, `Neg`, `Reciprocal`, `Exp`, `Log`,
///   `Sign`, `Floor`, `Ceil`, `Round`, `Sin`, `Cos`, `Softplus` (NVRTC
///   f32/f16/bf16, formulas matched to the CPU EP `unary_math.rs`).
/// * **Logical** — `Not` (bool), `And`, `Or`, `Xor` (bool, equal-shape).
/// * **Comparison** — `Equal`, `Greater`, `Less`, `GreaterOrEqual`,
///   `LessOrEqual` (f32 operands → bool, equal-shape).
///
/// See `docs/CUDA_COVERAGE.md` for the full op → backend mapping matrix and the
/// prioritised list of remaining / custom-kernel ops.
pub const CUDA_COVERED_OPS: &[&str] = &[
    "MatMul",
    "Gemm",
    "Conv",
    "MaxPool",
    "AveragePool",
    "Relu",
    "Sqrt",
    "Erf",
    "Tanh",
    "Sigmoid",
    "Gelu",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Min",
    "Max",
    "Attention",
    "Softmax",
    "LayerNormalization",
    "SkipLayerNormalization",
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
];

/// Build an [`OpRegistry`] populated with the CUDA kernel factories.
///
/// The shared [`CudaRuntime`] (context + stream + cuBLASLt handle) is threaded
/// into every factory so kernels submit onto the EP's single stream.
pub fn build_cuda_registry(runtime: Arc<CudaRuntime>) -> OpRegistry {
    let mut reg = OpRegistry::new();

    // GEMM family (cuBLASLt).
    reg.register(
        OpKey::new("MatMul", "", 1),
        Box::new(matmul::MatMulFactory {
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

    // Elementwise unary activations (NVRTC f32/f16/bf16 pointwise). `Gelu` is a
    // `com.microsoft` contrib op; the rest are standard-domain.
    for (op_type, domain, op) in [
        ("Relu", "", UnaryOp::Relu),
        ("Sqrt", "", UnaryOp::Sqrt),
        ("Erf", "", UnaryOp::Erf),
        ("Tanh", "", UnaryOp::Tanh),
        ("Sigmoid", "", UnaryOp::Sigmoid),
        ("Gelu", "com.microsoft", UnaryOp::Gelu),
    ] {
        reg.register(
            OpKey::new(op_type, domain, 1),
            Box::new(UnaryFactory {
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

    // Attention (SDPA/GQA baseline).
    reg.register(
        OpKey::new("Attention", "com.microsoft", 1),
        Box::new(attention::AttentionFactory {
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

    // RMSNormalization (fused NVRTC, no mean subtraction): the ai.onnx op and the
    // `com.microsoft` `SimplifiedLayerNormalization` are the same computation.
    reg.register(
        OpKey::new("SimplifiedLayerNormalization", "com.microsoft", 1),
        Box::new(normalization::RmsNormFactory {
            runtime: runtime.clone(),
        }),
    );
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

    // Logical binary (bool operands → bool; equal-shape, non-zero byte = true).
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

    // Comparison (f32 operands → bool; equal-shape, ONNX comparison semantics).
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
        let mut seen = std::collections::HashSet::new();
        for op in CUDA_COVERED_OPS {
            assert!(seen.insert(*op), "duplicate op {op} in CUDA_COVERED_OPS");
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
}
