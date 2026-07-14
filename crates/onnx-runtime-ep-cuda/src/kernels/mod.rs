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

pub mod attention;
pub mod elementwise;
pub mod gemm;
pub mod matmul;

use elementwise::{BinaryFactory, BinaryOp, UnaryFactory, UnaryOp};

/// The ops the CUDA EP implements today.
///
/// * **GEMM family** — `MatMul` and `Gemm` (cuBLASLt; `Gemm` adds a fused NVRTC
///   `beta·C` broadcast-bias epilogue).
/// * **Elementwise unary** — `Relu`, `Sqrt`, `Erf`, `Tanh` (+ `Sigmoid`) and the
///   `com.microsoft` `Gelu`, via runtime-compiled (NVRTC) f32 pointwise kernels.
/// * **Elementwise binary (equal-shape)** — `Add`, `Sub`, `Mul`, `Div`, `Pow`,
///   `Min`, `Max`, via NVRTC f32 pointwise kernels.
/// * **Attention** — the SDPA/GQA baseline (`com.microsoft` domain; cuBLAS
///   batched GEMM + NVRTC softmax), the §13.3 binding a cuDNN-fused SDPA /
///   FlashAttention-3 shim drops in behind.
///
/// See `docs/CUDA_COVERAGE.md` for the full op → backend mapping matrix and the
/// prioritised list of remaining / custom-kernel ops.
pub const CUDA_COVERED_OPS: &[&str] = &[
    "MatMul", "Gemm", "Relu", "Sqrt", "Erf", "Tanh", "Sigmoid", "Gelu", "Add", "Sub", "Mul", "Div",
    "Pow", "Min", "Max", "Attention",
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

    // Elementwise unary activations (NVRTC f32 pointwise). `Gelu` is a
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

    // Elementwise binary (NVRTC f32 pointwise, equal-shape operands).
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
    reg
}
