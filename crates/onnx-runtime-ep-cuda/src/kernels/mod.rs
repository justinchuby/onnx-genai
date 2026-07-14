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
pub mod matmul;

/// The ops the CUDA EP implements in Phase 2a.
///
/// `MatMul` (cuBLASLt GEMM) plus the SDPA/GQA `Attention` baseline
/// (`com.microsoft` domain — cuBLAS batched GEMM + NVRTC softmax; the §13.3
/// binding that a cuDNN-fused SDPA / FlashAttention-3 shim drops in behind).
pub const CUDA_PHASE2A_OPS: &[&str] = &["MatMul", "Attention"];

/// Build an [`OpRegistry`] populated with the Phase-2a CUDA kernel factories.
///
/// The shared [`CudaRuntime`] (context + stream + cuBLASLt handle) is threaded
/// into every factory so kernels submit onto the EP's single stream.
pub fn build_cuda_registry(runtime: Arc<CudaRuntime>) -> OpRegistry {
    let mut reg = OpRegistry::new();
    reg.register(
        OpKey::new("MatMul", "", 1),
        Box::new(matmul::MatMulFactory {
            runtime: runtime.clone(),
        }),
    );
    reg.register(
        OpKey::new("Attention", "com.microsoft", 1),
        Box::new(attention::AttentionFactory {
            runtime: runtime.clone(),
        }),
    );
    reg
}
